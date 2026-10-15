#!/usr/bin/env python3
"""Run the external answer scorer on a completed candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_answer(row: dict[str, Any]) -> str | None:
    output = row.get("stage_output")
    if not isinstance(output, dict):
        return None
    answer = output.get("answer")
    if answer is None:
        answer = output.get("final_answer")
    return str(answer).strip() if answer is not None and str(answer).strip() else None


def _load_bundle(bundle_root: Path):
    bundle_root = bundle_root.resolve()
    if not (bundle_root / "answer_scorer" / "scorer.py").is_file():
        raise FileNotFoundError(f"scorer_bundle_missing:{bundle_root}")
    sys.path.insert(0, str(bundle_root))
    from answer_scorer.scorer import score_predictions

    return score_predictions


def _configured_judge_value(judge_name: str, agent_name: str, *, default: str = "") -> str:
    """Use an explicit judge override, otherwise mirror the agent transport."""
    return (os.environ.get(judge_name) or os.environ.get(agent_name) or default).strip()


def _scoring_signature(evaluator_rows, predictions, bundle_root, model, base_url, api_surface):
    sources = [Path(__file__), *sorted((bundle_root / "answer_scorer").glob("*.py")),
               *sorted((bundle_root / "answer_scorer").glob("*.txt"))]
    inputs = {
        "references": evaluator_rows, "predictions": predictions,
        "model": model, "base_url": base_url, "api_surface": api_surface,
        "scorer": [(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources],
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def score_candidate(
    *,
    candidate_dir: Path,
    public_jsonl: Path,
    evaluator_jsonl: Path,
    bundle_root: Path = Path("scorer_bundle"),
    timeout_sec: int = 180,
) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    evaluator_jsonl = evaluator_jsonl.resolve()
    public_list = _read_jsonl(public_jsonl.resolve())
    public_rows = {str(row["sample_id"]): row for row in public_list}
    evaluator_rows = _read_jsonl(evaluator_jsonl)
    records_path = candidate_dir / "records.jsonl"
    rows = _read_jsonl(records_path)
    for name, population in (("public", public_list), ("evaluator", evaluator_rows), ("candidate", rows)):
        ids = [str(row.get("sample_id") or "") for row in population]
        if not ids or not all(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{name}_population_empty_or_duplicate")
    if any(row.get("error") or row.get("stage") != "P4" for row in rows):
        raise ValueError("only_successful_P4_records_can_be_scored")
    if {str(row["sample_id"]) for row in rows} != set(public_rows):
        raise ValueError("candidate_and_public_population_mismatch")
    if {str(row["sample_id"]) for row in evaluator_rows} != set(public_rows):
        raise ValueError("public_and_evaluator_population_mismatch")

    predictions = {str(row["sample_id"]): _candidate_answer(row) for row in rows}
    score_predictions_fn = _load_bundle(bundle_root)
    # The external answer scorer uses a fully independent evaluation context.
    # Its transport mirrors the supervisor's model deployment and endpoint
    # credentials by default; explicit COEVOLVE_JUDGE_* values can override it.
    base_url = _configured_judge_value("COEVOLVE_JUDGE_BASE_URL", "AGENTIC_VLM_BASE_URL")
    api_key = _configured_judge_value("COEVOLVE_JUDGE_API_KEY", "AGENTIC_VLM_API_KEY")
    from runtime.model_config import shared_vlm_model
    model = _configured_judge_value("COEVOLVE_JUDGE_MODEL", "AGENTIC_VLM_MODEL", default=shared_vlm_model())
    api_surface = _configured_judge_value(
        "COEVOLVE_JUDGE_API_SURFACE", "AGENTIC_AGENTS_SDK_MODEL_API", default="responses"
    ).lower()
    signature = _scoring_signature(evaluator_rows, predictions, bundle_root, model, base_url, api_surface)
    summary_path = candidate_dir / "SCORING.json"
    cached = {}
    if summary_path.is_file():
        try:
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            if saved.get("input_signature") == signature:
                cached = {str(item["sample_id"]): item for item in saved["results"]
                          if item.get("status") == "scored" and isinstance(item.get("correct"), bool)}
        except (ValueError, KeyError, TypeError):
            pass
    pending = [row for row in evaluator_rows if str(row["sample_id"]) not in cached]
    results = score_predictions_fn(
        pending,
        predictions,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout_sec,
        api_surface=api_surface,
        prompt_path=bundle_root / "answer_scorer" / "perceptionbench_official_judge_prompt.txt",
    ) if pending else []
    results = list(cached.values()) + results
    order = {str(row["sample_id"]): i for i, row in enumerate(evaluator_rows)}
    results.sort(key=lambda item: order.get(str(item["sample_id"]), len(order)))
    by_id = {str(item["sample_id"]): item for item in results}
    for row in rows:
        row["scoring"] = by_id.get(str(row["sample_id"]), {
            "status": "evaluator_error",
            "score": None,
            "correct": None,
            "error": "missing_scorer_result",
        })
    raw_records = candidate_dir / "records.raw.jsonl"
    if not raw_records.exists():
        raw_records.write_bytes(records_path.read_bytes())
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    scored = [item for item in results if item.get("status") == "scored"]
    correct = [item for item in scored if item.get("correct") is True]
    complete = (len(results) == len(evaluator_rows) and len(scored) == len(evaluator_rows)
                and len(by_id) == len(results) and set(by_id) == set(public_rows))
    summary = {
        "status": "completed" if complete else "incomplete",
        "eligible": complete,
        "candidate_dir": str(candidate_dir),
        "bundle_root": str(bundle_root),
        "scoring_contract": "external_answer_scorer_v1",
        "independent_context": True,
        "judge_model": model,
        "input_signature": signature,
        "items": len(results),
        "scored_items": len(scored),
        "correct_items": len(correct),
        "accuracy": len(correct) / len(evaluator_rows) if complete else None,
        "results": results,
    }
    from execution import _atomic_text
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if not summary_path.is_file() or summary_path.read_text(encoding="utf-8") != summary_text:
        _atomic_text(summary_path, summary_text)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--public-jsonl", type=Path, required=True)
    parser.add_argument("--evaluator-jsonl", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, default=Path("scorer_bundle"))
    parser.add_argument("--timeout-sec", type=int, default=180)
    args = parser.parse_args()
    try:
        summary = score_candidate(
            candidate_dir=args.candidate_dir,
            public_jsonl=args.public_jsonl,
            evaluator_jsonl=args.evaluator_jsonl,
            bundle_root=args.bundle_root,
            timeout_sec=args.timeout_sec,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}:{exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": summary["status"], "accuracy": summary["accuracy"]}, ensure_ascii=False))
    return 0 if summary["status"] == "completed" and summary.get("eligible") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
