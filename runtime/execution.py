#!/usr/bin/env python3
"""Portable execution runner for CoWorld Reasoner.

The complete evaluation and co-evolution entry points invoke this runner
automatically and carry each sample's interaction history forward.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _resume_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Compare experiment settings independently of request scheduling limits."""
    scheduling_settings = {
        "AGENTIC_VLM_MAX_CONCURRENCY",
        "AGENTIC_FAL_H3_MAX_INFLIGHT",
        "AGENTIC_FAL_H3_TRANSFER_CONCURRENCY",
    }
    return {**identity, "settings": {
        key: value for key, value in identity.get("settings", {}).items()
        if key not in scheduling_settings
    }}


def _json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(path, json.dumps(_json(value), ensure_ascii=False, indent=2) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(path, "".join(json.dumps(_json(row), ensure_ascii=False) + "\n" for row in rows))


def _completed_records(output: Path, rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    """Recover committed samples even if the aggregate was never written."""
    completed = []
    for index, row in enumerate(rows):
        record_path = output / "sample_records" / f"{row['sample_id']}.json"
        history_path = output / "sample_runs" / f"{index:03d}" / "INTERACTION_HISTORY.json"
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (record.get("sample_id") == str(row["sample_id"]) and record.get("stage") == stage
                and record.get("index") == index and not record.get("error")
                and record.get("stage_output") is not None and isinstance(history, list) and history):
            completed.append(record)
    return completed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty_public_manifest:{path}")
    return rows


def _public_rows(path: Path, input_root: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    ids = [str(row.get("sample_id")) for row in rows]
    if any(not value or value == "None" for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("public_manifest_requires_unique_sample_id")
    for row in rows:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"sample_requires_question:{row['sample_id']}")
        if str(row["sample_id"]) in {".", ".."} or any(char in str(row["sample_id"]) for char in "/\\"):
            raise ValueError("sample_id_must_be_a_filename")
        forbidden = {"answer", "gold", "ground_truth", "correct", "evaluator_detail"}
        if forbidden.intersection(row):
            raise ValueError(f"evaluator_state_in_public_manifest:{row['sample_id']}")
        images = row.get("image_paths")
        if not isinstance(images, list) or not images:
            raise ValueError(f"sample_requires_image_paths:{row['sample_id']}")
        resolved = []
        for value in images:
            path_value = Path(str(value))
            resolved.append((path_value if path_value.is_absolute() else input_root / path_value).resolve())
        row["_resolved_image_paths"] = resolved
    return rows


def _api_config() -> tuple[str, str, str]:
    from runtime.local_env import load_local_env
    from runtime.model_config import default_live_base_url, live_api_key
    from runtime.agent_runtime import configured_model
    load_local_env()
    base_url = os.environ.get("AGENTIC_VLM_BASE_URL") or default_live_base_url()
    api_key = live_api_key("AGENTIC_VLM_API_KEY", "AGENTIC_LLM_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing_agent_api_key_set_AGENTIC_VLM_API_KEY")
    return base_url, api_key, configured_model()


def _row_paths(row: dict[str, Any]) -> tuple[list[Path], list[str]]:
    paths = list(row["_resolved_image_paths"])
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing_input_media:{row['sample_id']}:{missing}")
    return paths, [f"original_observation_{i:02d}" for i in range(len(paths))]


def _require_live_backend(stage: str) -> None:
    if stage != "P2":
        return
    backend = os.environ.get("AGENTIC_VWM_TOOL_BACKEND", "fal_h3").strip().lower() or "fal_h3"
    if backend != "fal_h3":
        raise ValueError(f"unsupported_vwm_harness_backend:{backend}")
    if backend == "fal_h3":
        missing = []
        if os.environ.get("AGENTIC_FAL_H3_EXECUTE") != "1":
            missing.append("AGENTIC_FAL_H3_EXECUTE=1")
        if not os.environ.get("FAL_KEY", "").strip():
            missing.append("FAL_KEY")
        if missing:
            raise RuntimeError("live_video_backend_not_configured:" + ",".join(missing))


def _backend_config(output_dir: Path):
    from evolved_harness.visual_generation import VWMBackendConfig
    return VWMBackendConfig(
        output_root=output_dir / "generated_evidence",
    )


def _record_base(row: dict[str, Any], stage: str, index: int) -> dict[str, Any]:
    return {"sample_id": str(row["sample_id"]), "index": index, "benchmark": row.get("benchmark"), "stage": stage}


def _generation_failure(record: dict[str, Any]) -> str | None:
    """Return a hard failure for an attempted request with no usable artifact."""
    for event in record.get("events") or []:
        if event.get("event_type") != "vwm_result":
            continue
        payload = event.get("payload") or {}
        status = str(payload.get("status") or "")
        if status in {"failed", "generation_failed", "failed_fal_h3_request", "invalid_or_unreadable_generated_artifact"}:
            return f"vwm_generation_failed:{payload.get('error') or status}"
    for generation in record.get("generation_records") or []:
        if not (generation.get("generation") or {}).get("technically_valid", False):
            result = (generation.get("generation") or {}).get("tool_return") or {}
            return f"vwm_generation_failed:{result.get('status') or 'invalid_artifact'}"
    return None


def _run_one(stage: str, row: dict[str, Any], index: int, output: Path, upstream: Path | None, base_url: str, api_key: str) -> dict[str, Any]:
    from runtime.agent_runtime import run_agent
    paths, roles = _row_paths(row)
    sample_dir = output / "sample_runs" / f"{index:03d}"
    common = {
        "output_dir": sample_dir,
        "observed_image_paths": paths,
        "observed_image_roles": roles,
        "interleaved_observations": bool(row.get("interleaved_observations", False)),
    }
    record = _record_base(row, stage, index)
    retakes = int(os.environ.get("AGENTIC_MAX_RETAKES", "0"))
    if retakes not in (0, 1):
        raise ValueError("AGENTIC_MAX_RETAKES_must_be_0_or_1")
    try:
        if stage == "P1":
            from runtime.vwm_harness.method_rules import SKILL_DISCOVERY
            from evolved_harness import p1_capability_assessment as guidance
            from evolved_harness.capability_assessment import (
                CapabilityAssessment, CAPABILITY_ASSESSMENT_INSTRUCTIONS, build_capability_assessment_tools,
            )
            from evolved_harness.visual_tools import HarnessToolContext
            context = HarnessToolContext(**common, max_vwm_calls=0, max_skill_loads=1)
            result = run_agent(
                runtime_context=context, task_prompt=str(row["question"]),
                output_type=CapabilityAssessment, tools=build_capability_assessment_tools(),
                base_url=base_url, api_key=api_key, max_turns=4, max_output_tokens=4096,
                timeout_sec=600, system_instructions=CAPABILITY_ASSESSMENT_INSTRUCTIONS.replace("{question}", str(row["question"])) + "\n" + guidance.STAGE_GUIDANCE
                + "\nFixed method boundary: " + SKILL_DISCOVERY,
                agent_name="coworld_p1", continuation_path=sample_dir / "INTERACTION_HISTORY.json",
                checkpoint_dir=sample_dir / "checkpoint",
            )
            record.update({"p1_output": result.final_output, "stage_output": result.final_output,
                           "loaded_skill_id": context.loaded_skill_id, "skill_loads": context.skill_loads})
        elif stage == "P2":
            from runtime.vwm_harness.method_rules import P2_AUTONOMY
            from evolved_harness import p2_vwm_request as guidance
            from evolved_harness.evidence_tools import EvidenceToolContext
            from runtime.generation_execution import execute_generation_request
            from evolved_harness.vwm_request import VWMRequestDecision, build_vwm_assistance_tools
            from runtime.vwm_harness.interaction_history import continue_capability_assessment, refresh_loaded_skill, stop_after_generation
            prior = json.loads((upstream / "sample_records" / f"{row['sample_id']}.json").read_text(encoding="utf-8"))
            context = EvidenceToolContext(
                **common, sample_id=str(row["sample_id"]), benchmark=str(row.get("benchmark") or ""),
                question=str(row["question"]),
                loaded_skill_id=prior.get("loaded_skill_id"), skill_loads=prior.get("skill_loads", 0),
                backend_config=_backend_config(output), max_vwm_calls=1 + retakes, max_vwm_attempts=2,
                max_retake_calls=retakes, single_retake_policy=True,
                max_verifier_calls=0, sampled_frame_budget=9,
            )
            context.vwm_dispatch_executor = execute_generation_request
            history = continue_capability_assessment(upstream, prior, str(row["question"]), retakes=retakes)
            refreshed = refresh_loaded_skill(history, context.loaded_skill_id)
            record["skill_guidance_refreshed"] = len(refreshed) != len(history)
            history = refreshed
            instruction = (
                "Continue your P1 reasoning with VWM request tools now available. "
                "Use the loaded skill and original observations. Decide autonomously "
                "whether predicted evidence can help and author an answer-free request. "
                "The available operations are independent of your earlier skill choice; "
                "you may choose an appropriate operation even if that earlier choice was incomplete. "
                "If no request is useful, return decision_reason. Otherwise the SDK "
                "will stop after generation. Do not solve the final task in this stage."
            )
            tools = build_vwm_assistance_tools()
            if retakes:
                instruction = instruction.replace("Otherwise the SDK will stop after generation.",
                    "After generation, check media quality and optionally repeat the same operation once when needed.")
            result = run_agent(
                runtime_context=context, task_prompt=str(row["question"]), initial_input=history,
                output_type=VWMRequestDecision, tools=tools, base_url=base_url, api_key=api_key,
                max_turns=6, max_output_tokens=4096, timeout_sec=900,
                system_instructions=(instruction + guidance.STAGE_GUIDANCE + guidance.REQUEST_GROUNDING_GUIDANCE
                    + "\nFixed method boundary: " + P2_AUTONOMY),
                tool_use_behavior=stop_after_generation, agent_name="coworld_p2",
                continuation_path=sample_dir / "INTERACTION_HISTORY.json",
                checkpoint_dir=sample_dir / "checkpoint",
            )
            record.update({"p1_output": prior.get("p1_output"), "stage_output": result.final_output,
                           "accepted_dispatches": context.accepted_dispatches,
                           "generation_records": context.generation_records,
                           "loaded_skill_id": context.loaded_skill_id})
        else:
            from evolved_harness import p3_video_interpretation as p3, p4_evidence_integration as p4
            from evolved_harness.evidence_tools import EvidenceToolContext, build_verifier_tool, ACTIVE_VERIFIER_PROFILE_IDS
            from evolved_harness.evidence_reasoning import EvidenceIntegration, VideoInterpretation, video_interpretation_instructions
            from runtime.vwm_harness.interaction_history import refresh_loaded_skill
            prior = json.loads((upstream / "sample_records" / f"{row['sample_id']}.json").read_text(encoding="utf-8"))
            generation_records = prior.get("generation_records", [])
            context = EvidenceToolContext(
                **common, sample_id=str(row["sample_id"]), benchmark=str(row.get("benchmark") or ""),
                question=str(row["question"]),
                loaded_skill_id=prior.get("loaded_skill_id"), max_vwm_calls=0, max_verifier_calls=2 if stage == "P3" else 0,
                verifier_guidance=p3.VERIFIER_GUIDANCE,
                sampled_frame_budget=9,
            )
            context.generation_records = generation_records
            context.valid_media_records = [item for item in generation_records if item.get("generation", {}).get("technically_valid") and item.get("sampled_frames")]
            history_path = upstream / "sample_runs" / f"{prior['index']:03d}" / "INTERACTION_HISTORY.json"
            history = json.loads(history_path.read_text(encoding="utf-8"))
            refreshed = refresh_loaded_skill(history, context.loaded_skill_id)
            record["skill_guidance_refreshed"] = len(refreshed) != len(history)
            history = refreshed
            if stage == "P3":
                instruction = video_interpretation_instructions(generation_records) + p3.STAGE_GUIDANCE + p3.REFERENCE_FRAME_GUIDANCE
                output_type, tools, agent_name = VideoInterpretation, [build_verifier_tool(p) for p in ACTIVE_VERIFIER_PROFILE_IDS], "coworld_p3"
            else:
                instruction = (
                    "Continue into P4. Use your preceding observations and evidence analysis "
                    "to answer the original task. Decide autonomously what evidence to use "
                    "and explain any reference-frame conversion or unresolved contradiction. "
                    "No runtime rule determines or replaces your answer."
                    + p4.STAGE_GUIDANCE
                )
                output_type, tools, agent_name = EvidenceIntegration, [], "coworld_p4"
            history.append({"role": "user", "content": instruction + "\n" + str(row["question"])})
            result = run_agent(
                runtime_context=context, task_prompt=str(row["question"]), initial_input=history,
                output_type=output_type, tools=tools, base_url=base_url, api_key=api_key,
                max_turns=6, max_output_tokens=4096, timeout_sec=900,
                system_instructions=instruction, agent_name=agent_name,
                continuation_path=sample_dir / "INTERACTION_HISTORY.json",
                checkpoint_dir=sample_dir / "checkpoint",
            )
            record.update({"p1_output": prior.get("p1_output"), "stage_output": result.final_output,
                           "generation_records": generation_records,
                           "verifier_records": [*prior.get("verifier_records", []), *context.verifier_records],
                           "accepted_dispatches": prior.get("accepted_dispatches", []),
                           "loaded_skill_id": context.loaded_skill_id})
        record.update({
            "error": result.error,
            "events": result.events,
            "usage": result.usage,
            "provider_errors": result.provider_errors,
        })
        from runtime.failures import verifier_transport_failure
        infrastructure_failure = _generation_failure(record) or verifier_transport_failure(record.get("verifier_records") or [])
        if infrastructure_failure and not record.get("error"):
            record["error"] = infrastructure_failure
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}:{str(exc)[:2000]}"
    _write(output / "sample_records" / f"{row['sample_id']}.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["P1", "P2", "P3", "P4"], required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-id", default="initial", help="Name for this candidate.")
    parser.add_argument("--round-index", type=int, default=0, help="Candidate revision round.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the public manifest without contacting a model.")
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    rows = _public_rows(args.input_jsonl.resolve(), input_root)
    if args.stage != "P1" and args.upstream_dir is None:
        parser.error("P2_P3_P4_require_upstream_dir")
    output = args.output_dir.resolve()
    if args.round_index < 0 or args.round_index > 5:
        parser.error("round_index_must_be_between_0_and_5")
    if args.dry_run:
        print(json.dumps({"status": "pass", "stage": args.stage, "rows": len(rows)}))
        return 0
    for row in rows:
        _row_paths(row)
    # A retry may reuse a partially completed stage. Successful sample
    # records are retained and only missing/failed samples are executed again.
    existing_records = _completed_records(output, rows, args.stage)
    upstream = args.upstream_dir.resolve() if args.upstream_dir else None
    if upstream and not (upstream / "records.jsonl").is_file():
        raise FileNotFoundError(f"upstream_records_missing:{upstream}")
    base_url, api_key, vlm_model = _api_config()
    _require_live_backend(args.stage)
    identity = {
        "stage": args.stage, "rows": _json(rows), "upstream": str(upstream),
        "candidate_id": args.candidate_id, "round_index": args.round_index,
        "base_url": base_url, "model": vlm_model,
        "media": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for row in rows for path in row["_resolved_image_paths"]},
        "upstream_records_sha256": hashlib.sha256((upstream / "records.jsonl").read_bytes()).hexdigest() if upstream else None,
        "settings": {key: value for key, value in sorted(os.environ.items())
                     if key.startswith("AGENTIC_") and not any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))},
    }
    identity_path = output / "RESUME_IDENTITY.json"
    if identity_path.is_file():
        if _resume_identity(json.loads(identity_path.read_text(encoding="utf-8"))) != _resume_identity(identity):
            raise ValueError("resume_inputs_or_configuration_changed_use_new_output_dir")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("resume_identity_missing_use_new_output_dir")
    else:
        _write(identity_path, identity)
    # Restore selected code before importing the task runtime.
    from util.harness_state import restore_harness, save_harness
    if upstream:
        upstream_config = json.loads((upstream / "RUN_CONFIG.json").read_text(encoding="utf-8"))
        if upstream_config.get("stage") != f"P{int(args.stage[1]) - 1}":
            raise ValueError("upstream_must_be_immediately_preceding_stage")
        if args.round_index == 0:
            snapshot = output / "harness_snapshot"
            restore_harness(ROOT, upstream, snapshot, resume=True)
        from runtime.vwm_harness.candidate_evaluation import candidate_record
        inherited, prior_records = candidate_record(upstream)
        if not inherited["eligible_for_selection"]:
            raise ValueError("upstream_candidate_not_ready_for_continuation:" + ";".join(inherited["continuation_issues"]))
        if inherited["stage"] != f"P{int(args.stage[1]) - 1}":
            raise ValueError("upstream_must_be_immediately_preceding_stage")
        if set(prior_records) != {str(row["sample_id"]) for row in rows}:
            raise ValueError("upstream_population_must_match_current_population")
        for prior in prior_records.values():
            history = upstream / "sample_runs" / f"{prior['index']:03d}" / "INTERACTION_HISTORY.json"
            if not history.is_file():
                raise ValueError(f"upstream_history_missing:{prior['sample_id']}")
    if not (output / "harness").exists():
        save_harness(ROOT, output)
    from util.harness_state import _source_paths
    if {p.relative_to(ROOT) for p in _source_paths(ROOT)} != {p.relative_to(output / "harness") for p in _source_paths(output / "harness")}:
        raise ValueError("resume_harness_file_set_changed_use_new_output_dir")
    for saved in _source_paths(output / "harness"):
        current = ROOT / saved.relative_to(output / "harness")
        if not current.is_file() or hashlib.sha256(current.read_bytes()).digest() != hashlib.sha256(saved.read_bytes()).digest():
            raise ValueError("resume_harness_changed_use_new_output_dir")
    _write(output / "RUN_CONFIG.json", {
        "stage": args.stage,
        "input_jsonl": str(args.input_jsonl.resolve()),
        "input_root": str(input_root),
        "upstream": str(upstream) if upstream else None,
        "workers": args.workers,
        "provider_base_url": base_url,
        "vlm_model": vlm_model,
        "candidate_id": args.candidate_id,
        "round_index": args.round_index,
        "max_revision_rounds": 5,
    })
    completed_ids = {
        str(item.get("sample_id")) for item in existing_records
        if not item.get("error") and (output / "sample_records" / f"{item.get('sample_id')}.json").is_file()
    }
    pending = [(index, row) for index, row in enumerate(rows) if str(row["sample_id"]) not in completed_ids]
    started = time.monotonic()
    finished, errors = len(completed_ids), 0
    print(f"[{args.stage}] {finished}/{len(rows)} complete; {len(pending)} pending", flush=True)
    fresh = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_run_one, args.stage, row, index, output, upstream, base_url, api_key)
                   for index, row in pending}
        while futures:
            done, futures = concurrent.futures.wait(futures, timeout=30,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                record = future.result()
                fresh.append(record)
                finished += 1
                errors += bool(record.get("error"))
            print(f"[{args.stage}] {finished}/{len(rows)} complete; {errors} errors; "
                  f"{time.monotonic() - started:.0f}s elapsed", flush=True)
    by_id = {str(item.get("sample_id")): item for item in existing_records}
    by_id.update({str(item.get("sample_id")): item for item in fresh})
    results = [by_id[str(row["sample_id"])] for row in rows if str(row["sample_id"]) in by_id]
    _write_jsonl(output / "records.jsonl", results)
    status = {"stage": args.stage, "expected_rows": len(rows), "returned_rows": len(results), "errors": sum(bool(row.get("error")) for row in results)}
    _write(output / "STAGE_RESULT.json", status)
    print(json.dumps(status, ensure_ascii=False))
    return 0 if status["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
