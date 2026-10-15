"""Candidate comparisons and supervisor context for staged co-evolution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def decision_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Separate executable choices from changes to their prose justification."""
    stage = row.get("stage", "P1")
    output = row.get("stage_output") or row.get("p1_output") or {}
    result = {"loaded_skill_id": row.get("loaded_skill_id")}
    if stage == "P1":
        result.update({key: output.get(key) for key in (
            "relation_visibility",
            "visual_operation_opportunity", "selected_task_skill", "skip_reason", "provisional_answer",
        )})
    elif stage == "P2":
        result["requests"] = [{key: request.get(key) for key in (
            "tool_interface", "source_image_indices", "source_order_mode",
            "reference_image_index", "candidate_image_index", "generation_instruction",
            "operation_controls", "preserve",
        )} for request in row.get("accepted_dispatches") or []]
    elif stage == "P3":
        result["verifiers"] = [record.get("profile_id", record.get("tool_name"))
                               for record in row.get("verifier_records") or []]
    else:
        result["answer"] = str(output.get("answer") or "").strip()
    return result


def comparison_rows(current, reference) -> list[dict[str, Any]]:
    rows = []
    for sid, row in sorted(current.items()):
        old = reference.get(sid)
        now = decision_fields(row)
        before = decision_fields(old) if old is not None else None
        correct = (row.get("scoring") or {}).get("correct")
        prior = (old.get("scoring") or {}).get("correct") if old is not None else None
        score_change = None
        if isinstance(correct, bool) and isinstance(prior, bool):
            score_change = ("stable_correct" if correct else "persistent_error") if correct == prior else (
                "improved" if correct else "regressed")
        elif correct is False:
            score_change = "error"
        rows.append({
            "sample_id": sid, "benchmark": row.get("benchmark"),
            "changed_fields": [key for key in now if before is not None and now[key] != before.get(key)],
            "current_decision": now, "reference_decision": before,
            "current_output": row.get("stage_output") or row.get("p1_output"),
            "reference_output": (old.get("stage_output") or old.get("p1_output")) if old else None,
            "current_verification": verifier_observations(row),
            "reference_verification": verifier_observations(old) if old is not None else None,
            "output_text_changed": old is not None and row.get("stage_output") != old.get("stage_output"),
            "score_change": score_change, "correct": correct,
            "execution_error": bool(row.get("error")),
            "skill_loaded": bool(row.get("loaded_skill_id")),
            "generation_count": len(row.get("generation_records") or []),
            "verifier_count": len(row.get("verifier_records") or []),
            **({"answer": now["answer"], "reference_answer": before.get("answer") if before else None}
               if "answer" in now else {}),
        })
    return rows


def verifier_observations(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose the verifier's reading alongside the task agent's interpretation."""
    return [{"profile_id": v.get("profile_id"),
             "status": (v.get("result") or {}).get("status"),
             "observations": ((v.get("result") or {}).get("quality") or {}).get("observations"),
             "failure_modes": (v.get("result") or {}).get("failure_modes")}
            for v in row.get("verifier_records") or []]


def decision_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair choices with bounded excerpts of requests, reasoning and verifier readings."""
    def excerpt(value):
        if isinstance(value, str) and len(value) > 480:
            return value[:480] + " [excerpt; full text in comparison.jsonl or inspect_case]"
        if isinstance(value, list):
            items = [excerpt(item) for item in value[:4]]
            if len(value) > 4:
                items.append(f"[{len(value) - 4} more items; see comparison.jsonl or inspect_case]")
            return items
        if isinstance(value, dict):
            return {key: excerpt(item) for key, item in value.items()}
        return value

    result = []
    for row in rows:
        item = dict(row)
        for field in ("current_output", "reference_output", "current_verification", "reference_verification"):
            item[field] = excerpt(row.get(field))
        for side in ("current_decision", "reference_decision"):
            if row[side] is not None:
                decision = dict(row[side])
                if "requests" in decision:
                    decision["requests"] = [{key: excerpt(value) if key in {
                        "generation_instruction", "operation_controls", "preserve"} else value
                        for key, value in request.items()} for request in decision["requests"]]
                item[side] = decision
        result.append(item)
    return result


def sample_cases(current, reference, limit: int, target_ids=()):
    stage = next((r.get("stage") for r in current.values() if r.get("stage")), "P1")
    used, skipped = ("skill_loaded", "no_skill") if stage == "P1" else ("generation", "no_generation")
    pools = {key: [] for key in (
        "experiment_target", "regressed", "persistent_error", "improved", "execution_error",
        "decision_changed", used, skipped, "stable_control", "interpretation_changed",
    )}
    for item in comparison_rows(current, reference):
        sid, row = item["sample_id"], current[item["sample_id"]]
        if sid in target_ids:
            pools["experiment_target"].append(sid)
        outcome = item["score_change"]
        if outcome in {"persistent_error", "error"}:
            pools["persistent_error"].append(sid)
        elif outcome in {"regressed", "improved"}:
            pools[outcome].append(sid)
        if item["execution_error"]:
            pools["execution_error"].append(sid)
        if item["changed_fields"]:
            pools["decision_changed"].append(sid)
        elif stage == "P3" and item["output_text_changed"]:
            pools["interpretation_changed"].append(sid)
        active = item["skill_loaded"] if stage == "P1" else bool(row.get("generation_records") or row.get("accepted_dispatches"))
        pools[used if active else skipped].append(sid)
        if not item["changed_fields"] and outcome not in {"regressed", "persistent_error", "error"}:
            pools["stable_control"].append(sid)

    def interleave(ids):
        groups = {}
        for sid in ids:
            groups.setdefault(str(current[sid].get("benchmark") or "unknown"), []).append(sid)
        ordered = []
        while groups:
            for group in sorted(list(groups)):
                ordered.append(groups[group].pop(0))
                if not groups[group]:
                    del groups[group]
        return ordered

    queues = {key: interleave(ids) for key, ids in pools.items()}
    selected = []
    # Rotate categories to distribute inspections across behavior types.
    while len(selected) < limit and any(queues.values()):
        for ids in queues.values():
            while ids and ids[0] in selected:
                ids.pop(0)
            if ids and len(selected) < limit:
                selected.append(ids.pop(0))
    return selected, pools


def preceding_outputs(candidate: Path, sample_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Read recorded outputs from the preceding candidates."""
    outputs = {sid: {} for sid in sample_ids}
    visited = {candidate.resolve()}
    config = json.loads((candidate / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    for _ in range(3):
        upstream = config.get("upstream")
        if not upstream:
            break
        parent = Path(upstream)
        parent = (parent if parent.is_absolute() else candidate / parent).resolve()
        if parent in visited:
            raise ValueError("candidate_upstream_cycle")
        visited.add(parent)
        config = json.loads((parent / "RUN_CONFIG.json").read_text(encoding="utf-8"))
        for line in (parent / "records.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["sample_id"] in outputs:
                outputs[row["sample_id"]][config["stage"]] = row.get("stage_output")
        candidate = parent
    return outputs
