#!/usr/bin/env python3
"""Run a complete evaluation with the Video World Model-Driven Harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


STAGES = ("P1", "P2", "P3", "P4")


def _load_env(path: Path) -> None:
    """Load unset KEY=VALUE pairs without printing credentials."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value


def _run(command: list[str], *, root: Path) -> None:
    completed = subprocess.run(command, cwd=str(root), env=os.environ.copy(), check=False)
    if completed.returncode:
        raise RuntimeError(f"evaluation_command_failed:{completed.returncode}")


def _check_retry_directory(path: Path, output_root: Path) -> None:
    """Keep stage retries within this run while preserving completed work."""
    if not path.exists():
        return
    resolved = path.resolve()
    root = output_root.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"retry_directory_outside_output_root:{resolved}")


def _run_stage_with_retries(
    command: list[str], *, root: Path, stage_dir: Path, output_root: Path, retries: int
) -> None:
    """Retry transient stage failures within the run."""
    for attempt in range(max(0, retries) + 1):
        try:
            _run(command, root=root)
            return
        except RuntimeError:
            if attempt >= retries:
                raise
            _check_retry_directory(stage_dir, output_root)


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


def _run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    _load_env((args.env_file if args.env_file.is_absolute() else root / args.env_file).resolve())
    input_jsonl = args.input_jsonl.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    retakes = getattr(args, "retakes", 0)
    os.environ["AGENTIC_MAX_RETAKES"] = str(retakes)
    if output_root.exists() and any(output_root.iterdir()) and not getattr(args, "resume", False):
        raise ValueError(f"output_root_must_be_new_or_empty:{output_root}")
    from execution import _api_config, _public_rows, _row_paths, _require_live_backend
    for row in _public_rows(input_jsonl, input_root):
        _row_paths(row)
    _require_live_backend("P2")
    _api_config()
    evaluator = args.evaluator_jsonl
    if evaluator is None and input_jsonl.name == "public.jsonl":
        candidate = root / "scorer_bundle" / "data" / "dev120_evaluator_only.jsonl"
        if candidate.is_file() and _same_population(input_jsonl, candidate):
            evaluator = candidate
    if evaluator is not None:
        evaluator = evaluator.resolve()
        if not evaluator.is_file():
            raise FileNotFoundError(f"evaluator_jsonl_missing:{evaluator}")
        if not _same_population(input_jsonl, evaluator):
            raise ValueError("public_and_evaluator_population_mismatch")
    identity = {"input_jsonl": str(input_jsonl), "input_root": str(input_root),
                "retakes": retakes, "evaluator": str(evaluator) if evaluator else None,
                "input_sha256": hashlib.sha256(input_jsonl.read_bytes()).hexdigest(),
                "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest() if evaluator else None}
    identity_path = output_root / "EVALUATION_CONFIG.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise ValueError("evaluation_resume_configuration_mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")

    upstream: Path | None = None
    stage_outputs: list[dict[str, Any]] = []
    for stage in STAGES:
        output_dir = output_root / stage.lower()
        command = [
            sys.executable,
            str(root / "execution.py"),
            "--stage", stage,
            "--input-jsonl", str(input_jsonl),
            "--input-root", str(input_root),
            "--output-dir", str(output_dir),
            "--workers", str(args.workers),
            "--candidate-id", "evaluation",
        ]
        if upstream is not None:
            command.extend(["--upstream-dir", str(upstream)])
        _run_stage_with_retries(
            command,
            root=root,
            stage_dir=output_dir,
            output_root=output_root,
            retries=args.stage_retries,
        )
        stage_outputs.append({"stage": stage, "output_dir": str(output_dir)})
        upstream = output_dir

    scoring: dict[str, Any] | None = None
    if evaluator is not None:
        evaluator = evaluator.resolve()
        if not evaluator.is_file():
            raise FileNotFoundError(f"evaluator_jsonl_missing:{evaluator}")
        command = [
            sys.executable,
            str(root / "external_scorer.py"),
            "--candidate-dir", str(upstream),
            "--public-jsonl", str(input_jsonl),
            "--evaluator-jsonl", str(evaluator),
            "--bundle-root", str(root / "scorer_bundle"),
            "--timeout-sec", str(args.timeout_sec),
        ]
        _run(command, root=root)
        scoring = {"candidate_dir": str(upstream), "evaluator": str(evaluator)}

    result = {"status": "completed", "retakes": retakes, "stage_outputs": stage_outputs, "scoring": scoring}
    (output_root / "EVALUATION_RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the current harness with exclusive access to its source tree."""
    from runtime.workflow_lock import workflow_lock

    with workflow_lock(Path(__file__).resolve().parent):
        return _run_evaluation(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--evaluator-jsonl", type=Path)
    parser.add_argument("--retakes", type=int, choices=[0, 1], default=0,
                        help="Paper evaluation uses 0; 1 enables one same-operation repair retake.")
    parser.add_argument("--resume", action="store_true", help="Resume the same inputs and method without rerunning committed samples.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument(
        "--stage-retries",
        type=int,
        default=2,
        help="Retries for a failed stage before aborting the evaluation.",
    )
    args = parser.parse_args()
    try:
        result = run_evaluation(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}:{exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": result["status"], "output_root": str(args.output_root.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
