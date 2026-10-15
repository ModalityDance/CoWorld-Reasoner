#!/usr/bin/env python3
"""Run VLM-VWM Harness Co-Evolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any
from harness_state import EXPERIMENT_FILES


STAGES = ("P1", "P2", "P3", "P4")
MAX_REVISION_ROUNDS = 5


def _read_env_file(path: Path) -> None:
    """Load unset KEY=VALUE pairs without ever printing their values."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value


def _json_from_last_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _same_population(left: Path, right: Path) -> bool:
    """Return whether two JSONL manifests contain the same sample IDs."""
    try:
        def ids(path: Path) -> list[str]:
            return [
                str(json.loads(line)["sample_id"])
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        a, b = ids(left), ids(right)
        return bool(a) and len(a) == len(set(a)) == len(b) == len(set(b)) and set(a) == set(b)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


class CommandFailed(RuntimeError):
    def __init__(self, returncode: int, command: list[str], detail: str):
        self.returncode = returncode
        super().__init__(f"command_failed:{returncode}:{' '.join(command)}:{detail}")


def _run(command: list[str], *, cwd: Path) -> dict[str, Any]:
    label = Path(command[1]).name if len(command) > 1 else Path(command[0]).name
    started = time.monotonic()
    print(f"[co-evolution] {label} started", flush=True)
    messages: queue.Queue[str | None] = queue.Queue()
    tail: deque[str] = deque(maxlen=100)
    result: dict[str, Any] = {}
    with subprocess.Popen(command, cwd=str(cwd),
                          env={**os.environ, "PYTHONUNBUFFERED": "1"},
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        def read_output() -> None:
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            while True:
                try:
                    line = messages.get(timeout=30)
                except queue.Empty:
                    print(f"[co-evolution] {label} running ({time.monotonic() - started:.0f}s)", flush=True)
                    continue
                if line is None:
                    break
                print(line, end="", flush=True)
                tail.append(line[-4000:])
                parsed = _json_from_last_line(line)
                if parsed:
                    result = parsed
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        finally:
            reader.join(timeout=1)
    if returncode != 0:
        raise CommandFailed(returncode, command, "".join(tail)[-4000:])
    return result


def _check_retry_directory(path: Path, output_root: Path) -> None:
    """Keep candidate retries within this run while preserving completed work."""
    if not path.exists():
        return
    resolved = path.resolve()
    root = output_root.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"retry_directory_outside_output_root:{resolved}")


def _run_candidate_with_retries(
    command: list[str], *, cwd: Path, candidate_dir: Path, output_root: Path, retries: int
) -> dict[str, Any]:
    """Retry a failed candidate execution before continuing the stage."""
    for attempt in range(max(0, retries) + 1):
        try:
            return _run(command, cwd=cwd)
        except RuntimeError:
            if attempt >= retries:
                raise
            _check_retry_directory(candidate_dir, output_root)
    raise RuntimeError("candidate_retry_loop_exhausted")


def _stage_command(
    root: Path,
    stage: str,
    input_jsonl: Path,
    input_root: Path,
    output_dir: Path,
    upstream: Path | None,
    candidate_id: str,
    round_index: int,
    workers: int,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "execution.py"),
        "--stage", stage,
        "--input-jsonl", str(input_jsonl),
        "--input-root", str(input_root),
        "--output-dir", str(output_dir),
        "--candidate-id", candidate_id,
        "--round-index", str(round_index),
        "--workers", str(workers),
    ]
    if upstream is not None:
        command.extend(["--upstream-dir", str(upstream)])
    return command


def _scorer_command(
    root: Path,
    candidate_dir: Path,
    public_jsonl: Path,
    evaluator_jsonl: Path,
    bundle_root: Path,
    timeout_sec: int,
) -> list[str]:
    return [
        sys.executable,
        str(root / "external_scorer.py"),
        "--candidate-dir", str(candidate_dir),
        "--public-jsonl", str(public_jsonl),
        "--evaluator-jsonl", str(evaluator_jsonl),
        "--bundle-root", str(bundle_root),
        "--timeout-sec", str(timeout_sec),
    ]


def _write_state(path: Path, state: dict[str, Any]) -> None:
    from execution import _write
    _write(path, state)


def _restore_selected_harness(root: Path, candidate_dir: Path, backup_dir: Path) -> None:
    """Install the selected candidate's harness for subsequent execution."""
    from harness_state import restore_harness
    restore_harness(root, candidate_dir, backup_dir, resume=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    env_file = args.env_file if args.env_file.is_absolute() else root / args.env_file
    _read_env_file(env_file)
    from runtime.vwm_harness.codex_session import require_codex
    require_codex()
    os.environ["AGENTIC_MAX_RETAKES"] = str(getattr(args, "retakes", 0))
    input_jsonl = args.input_jsonl.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    resuming = output_root.exists() and any(output_root.iterdir())
    if resuming and not getattr(args, "resume", False):
        raise ValueError(f"output_root_must_be_new_or_empty:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    stages = tuple(args.stages)
    if not stages or any(stage not in STAGES for stage in stages):
        raise ValueError("stages_must_be_subset_of_P1_P2_P3_P4")
    positions = [STAGES.index(stage) for stage in stages]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise ValueError("stages_must_be_contiguous_and_in_order")
    if "P2" in stages:
        from execution import _require_live_backend
        _require_live_backend("P2")
    if stages[0] == "P1" and args.initial_upstream is not None:
        raise ValueError("initial_upstream_is_only_valid_when_starting_after_P1")
    if stages[0] != "P1" and args.initial_upstream is None:
        raise ValueError("initial_upstream_required_when_starting_after_P1")
    evolution_rounds = int(getattr(args, "evolution_rounds", 1))
    if evolution_rounds < 1:
        raise ValueError("evolution_rounds_must_be_positive")
    if evolution_rounds > 1 and stages != STAGES:
        raise ValueError("multiple_evolution_rounds_require_all_four_stages")
    evaluator_jsonl = getattr(args, "evaluator_jsonl", None)
    if evaluator_jsonl is None:
        default_evaluator = root / "scorer_bundle" / "data" / "dev120_evaluator_only.jsonl"
        if input_jsonl.name == "public.jsonl" and default_evaluator.is_file() and _same_population(input_jsonl, default_evaluator):
            evaluator_jsonl = default_evaluator
            args.evaluator_jsonl = evaluator_jsonl
    if evaluator_jsonl is not None:
        evaluator_jsonl = evaluator_jsonl.resolve()
        args.evaluator_jsonl = evaluator_jsonl
        if not evaluator_jsonl.is_file():
            raise FileNotFoundError(f"evaluator_jsonl_missing:{evaluator_jsonl}")
        if not _same_population(input_jsonl, evaluator_jsonl):
            raise ValueError("public_and_evaluator_population_mismatch")
    from execution import _api_config, _public_rows, _row_paths, _resume_identity
    _api_config()
    rows = _public_rows(input_jsonl, input_root)
    media = {}
    for row in rows:
        for path in _row_paths(row)[0]:
            media[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {
        "input_jsonl": str(input_jsonl), "input_root": str(input_root),
        "input_sha256": hashlib.sha256(input_jsonl.read_bytes()).hexdigest(),
        "evaluator_jsonl": str(evaluator_jsonl) if evaluator_jsonl else None,
        "evaluator_sha256": hashlib.sha256(evaluator_jsonl.read_bytes()).hexdigest() if evaluator_jsonl else None,
        "media": media, "stages": list(stages), "evolution_rounds": evolution_rounds,
        "max_revisions": args.max_revisions,
        "initial_upstream": str(args.initial_upstream.resolve()) if args.initial_upstream else None,
        "candidate_id": args.candidate_id,
        "supervisor": {"harness": "codex_native_goal", "max_cases": args.max_cases},
        "settings": {key: value for key, value in sorted(os.environ.items())
                     if key.startswith(("AGENTIC_", "COEVOLVE_"))
                     and not any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))},
        "runner": {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                   for relative in EXPERIMENT_FILES},
    }
    identity_path = output_root / "COEVOLUTION_CONFIG.json"
    if resuming:
        if not identity_path.is_file() or _resume_identity(json.loads(identity_path.read_text(encoding="utf-8"))) != _resume_identity(identity):
            raise ValueError("coevolution_resume_configuration_mismatch")
    else:
        _write_state(identity_path, identity)
    from runtime.vwm_harness.supervisor_agent import run as run_native_evolution
    return run_native_evolution(root, output_root, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional local env file; values are never printed.")
    parser.add_argument("--initial-upstream", type=Path, help="Existing preceding-stage candidate when starting at P2/P3/P4.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the same evolution, reusing completed samples, the supervisor conversation, and selections.")
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--evolution-rounds", type=int, default=3, help="Outer evolution rounds R from the paper.")
    parser.add_argument("--retakes", type=int, choices=[0, 1], default=0,
                        help="Paper budget is 0; optionally permit one same-operation repair retake.")
    parser.add_argument("--evaluator-jsonl", type=Path, help="Reference answers for the external answer scorer.")
    parser.add_argument("--candidate-id", default="auto")
    parser.add_argument("--max-revisions", type=int, default=MAX_REVISION_ROUNDS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stage-retries",
        type=int,
        default=2,
        help="Retries for a failed candidate stage before aborting co-evolution.",
    )
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--timeout-sec", type=int, default=900)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.max_revisions <= MAX_REVISION_ROUNDS:
        raise SystemExit(f"max_revisions_must_be_between_0_and_{MAX_REVISION_ROUNDS}")
    try:
        from runtime.workflow_lock import workflow_lock
        with workflow_lock(Path(__file__).resolve().parent):
            result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}:{exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": result["status"], "final_candidate_dir": result["final_candidate_dir"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
