"""Command-line entry point for the external answer scorer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .scorer import score_predictions


def _setting(judge: str, agent: str, default: str = "") -> str:
    return os.environ.get(judge) or os.environ.get(agent) or default


def main() -> int:
    parser = argparse.ArgumentParser(description="Score saved CoWorld Reasoner answers with the external answer scorer.")
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="JSONL rows with sample_id and prediction (or answer).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=_setting("COEVOLVE_JUDGE_BASE_URL", "AGENTIC_VLM_BASE_URL"))
    parser.add_argument("--api-key", default=_setting("COEVOLVE_JUDGE_API_KEY", "AGENTIC_VLM_API_KEY"))
    parser.add_argument("--judge-model", default=_setting("COEVOLVE_JUDGE_MODEL", "AGENTIC_VLM_MODEL"))
    parser.add_argument("--api-surface", choices=["responses", "chat"], default=_setting("COEVOLVE_JUDGE_API_SURFACE", "AGENTIC_AGENTS_SDK_MODEL_API", "responses"))
    args = parser.parse_args()
    evaluator = [json.loads(line) for line in args.evaluator.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluator_ids = [str(row.get("sample_id") or "") for row in evaluator]
    if not evaluator_ids or not all(evaluator_ids) or len(set(evaluator_ids)) != len(evaluator_ids):
        parser.error("evaluator_population_empty_or_duplicate")
    predictions = {}
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            sid = str(row["sample_id"])
            if sid in predictions:
                parser.error("duplicate_prediction_sample_id")
            predictions[sid] = row.get("prediction", row.get("answer"))
    if set(predictions) != set(evaluator_ids):
        parser.error("prediction_and_evaluator_population_mismatch")
    results = score_predictions(evaluator, predictions, base_url=args.base_url, api_key=args.api_key,
        model=args.judge_model, api_surface=args.api_surface)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")
    scored = [r for r in results if r.get("status") == "scored"]
    correct = sum(r.get("correct") is True for r in scored)
    complete = len(results) == len(evaluator) == len(scored)
    print(json.dumps({"status": "completed" if complete else "incomplete",
                      "rows": len(results), "scored": len(scored), "correct": correct,
                      "accuracy": correct / len(evaluator) if complete else None}, ensure_ascii=False))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
