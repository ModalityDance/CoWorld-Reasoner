"""Summarize candidate behavior and prepare candidates for stage selection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections import Counter

MAX_REVISION_ROUNDS = 5


def population_metrics(records: dict[str, dict[str, Any]], stage: str) -> dict[str, Any]:
    """Summarize feedback available at the completed stage."""
    if stage not in {"P1", "P2", "P3", "P4"}:
        raise ValueError(f"unknown_stage:{stage}")
    rows = list(records.values())
    known = [r for r in rows if "skill_loads" in r or "loaded_skill_id" in r]
    loaded = [r for r in known if int(r.get("skill_loads") or 0) > 0 or r.get("loaded_skill_id")]
    metrics = {
        "rows": len(rows),
        "benchmarks": dict(Counter(str(r.get("benchmark") or "unknown") for r in rows)),
        "execution_error_count": sum(bool(r.get("error")) for r in rows),
        "skill_load_observed_rows": len(known),
        "skill_loaded_sample_count": len(loaded),
        "skill_load_rate": len(loaded) / len(known) if known else None,
        "skill_load_rate_denominator": "rows with explicit skill_loads or loaded_skill_id",
        "loaded_skill_counts": dict(Counter(str(r.get("loaded_skill_id") or "unknown") for r in loaded)),
    }
    if stage != "P1":
        metrics.update({
            "accepted_dispatch_count": sum(len(r.get("accepted_dispatches") or []) for r in rows),
            "samples_with_generation": sum(bool(r.get("generation_records")) for r in rows),
            "generation_count": sum(len(r.get("generation_records") or []) for r in rows),
        })
    if stage in {"P3", "P4"}:
        metrics["verifier_count"] = sum(len(r.get("verifier_records") or []) for r in rows)
    if stage == "P4":
        scored = [(r.get("scoring") or {})["correct"] for r in rows
                  if (r.get("scoring") or {}).get("correct") is not None]
        metrics["scored_sample_count"] = len(scored)
        metrics["answer_correctness_counts"] = dict(Counter(str(value) for value in scored))
    return metrics


def candidate_record(directory: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    directory = directory.resolve()
    config = json.loads((directory / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    status = json.loads((directory / "STAGE_RESULT.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records = {str(row["sample_id"]): row for row in rows}
    if not rows or len(records) != len(rows) or status.get("expected_rows") != len(rows) or status.get("returned_rows") != len(rows):
        raise ValueError(f"candidate_population_incomplete_or_duplicate:{directory}")
    round_index = config.get("round_index")
    if not isinstance(round_index, int) or isinstance(round_index, bool) or not 0 <= round_index <= MAX_REVISION_ROUNDS:
        raise ValueError("candidate_round_index_must_be_between_0_and_5")
    if not str(config.get("candidate_id") or "").strip() or config.get("stage") not in {"P1", "P2", "P3", "P4"}:
        raise ValueError("candidate_id_and_stage_required")
    if status.get("stage") != config["stage"] or any(row.get("stage") != config["stage"] for row in rows):
        raise ValueError("candidate_stage_mismatch")
    upstream = config.get("upstream")
    if not (directory / "harness").is_dir():
        raise ValueError(f"candidate_harness_missing:{directory}")
    continuation_issues = []
    for row in rows:
        sample_path = directory / "sample_records" / f"{row['sample_id']}.json"
        history_path = directory / "sample_runs" / f"{row['index']:03d}" / "INTERACTION_HISTORY.json"
        if not sample_path.is_file():
            raise ValueError(f"candidate_execution_files_missing:{row['sample_id']}")
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        if row.get("error") or sample.get("error"):
            continuation_issues.append(f"{row['sample_id']}:execution_error")
        if row.get("stage_output") is None or sample.get("stage_output") is None:
            continuation_issues.append(f"{row['sample_id']}:missing_stage_output")
        if config["stage"] == "P1" and sample.get("p1_output") is None:
            continuation_issues.append(f"{row['sample_id']}:missing_p1_decision")
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, list) or not history or any(not isinstance(item, dict) or not (item.get("role") or item.get("type")) for item in history):
                continuation_issues.append(f"{row['sample_id']}:invalid_sdk_history")
        except (OSError, ValueError):
            continuation_issues.append(f"{row['sample_id']}:missing_or_invalid_sdk_history")
    return {
        "candidate_id": config["candidate_id"], "round_index": round_index,
        "stage": config["stage"], "candidate_dir": str(directory), "status": "completed",
        "upstream": str(Path(upstream).resolve()) if upstream else None,
        "sample_ids": sorted(records),
        "harness_source_dir": str(directory / "harness"),
        "eligible_for_selection": not continuation_issues,
        "continuation_issues": continuation_issues,
        "vlm_model": config.get("vlm_model"), "provider_base_url": config.get("provider_base_url"),
        "metrics": population_metrics(records, config["stage"]),
    }, records


def validate_candidates(candidates: list[dict[str, Any]], stage: str) -> None:
    if not candidates or len(candidates) > MAX_REVISION_ROUNDS + 1:
        raise ValueError("candidate_count_must_be_between_1_and_6")
    ids = [c["candidate_id"] for c in candidates]
    rounds = [c["round_index"] for c in candidates]
    if len(set(ids)) != len(ids) or len(set(rounds)) != len(rounds):
        raise ValueError("candidate_ids_and_rounds_must_be_unique")
    if 0 not in rounds:
        raise ValueError("initial_candidate_required_supply_candidate_dir")
    for c in candidates:
        if c["stage"] != stage or c["status"] != "completed":
            raise ValueError("only_completed_same_stage_candidates_allowed")
    for key in ("upstream", "sample_ids"):
        if any(c.get(key) != candidates[0].get(key) for c in candidates[1:]):
            raise ValueError(f"candidate_comparison_mismatch:{key}")
