"""Portable external answer scorer for the fixed CoWorld Reasoner evaluator contract.

The scorer consumes evaluator data and saved predictions after agent execution.
Multiple-choice rows are deterministic; open-ended answers use the bundled
judge prompt and an OpenAI-compatible endpoint only when needed.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

OPTION_PATTERN = re.compile(r"^\s*\(?([A-Z])\)?(?:\s*[:.)-].*)?\s*$", re.I)
OPTION_LINE_PATTERN = re.compile(r"^\s*([A-Z])\s*[.:)]\s*(.*?)\s*$", re.M)
PERCEPTION_SCORER = "perceptionbench_official_general_qa_judge_v1"
DEFAULT_JUDGE_MODEL = ""
DEFAULT_PROMPT = Path(__file__).with_name("perceptionbench_official_judge_prompt.txt")
EXTENDED_PROMPT = Path(__file__).with_name("extended_short_answer_judge.txt")


def normalize_option_letter(value: Any) -> str | None:
    match = OPTION_PATTERN.fullmatch(str(value or ""))
    return match.group(1).upper() if match else None


def score_exact_match(evaluator: dict[str, Any], prediction: Any) -> dict[str, Any]:
    expected = normalize_option_letter(evaluator.get("answer"))
    actual = normalize_option_letter(prediction)
    if expected is None:
        raise ValueError("evaluator answer is not a single option letter")
    return {
        "status": "scored",
        "score": float(actual == expected),
        "correct": actual == expected,
        "normalized_prediction": actual,
        "scorer": "normalized_exact_match_v1",
    }


def _option_map(question: Any) -> dict[str, str]:
    return {m.group(1).upper(): m.group(2).strip() for m in OPTION_LINE_PATTERN.finditer(str(question or ""))}


def _deterministic_open_choice(evaluator: dict[str, Any], prediction: Any) -> dict[str, Any] | None:
    question = evaluator.get("official_problem") or evaluator.get("question")
    reference = evaluator.get("answer")
    ref = reference.get("answer") if isinstance(reference, dict) else reference
    match = re.fullmatch(r"\s*([A-Z])\s*[.:)]?\s*", str(ref or ""))
    options = _option_map(question)
    if not match or match.group(1).upper() not in options:
        return None
    text = str(prediction or "").strip()
    label = re.match(r"^\s*([A-Z])(?:\s*[.:)]|\s*$)", text)
    predicted = label.group(1).upper() if label else None
    mapping = "explicit_option_label" if predicted else None
    if predicted is None:
        norm = re.sub(r"[^a-z0-9]+", "", text.casefold())
        candidates = [k for k, v in options.items() if norm and norm == re.sub(r"[^a-z0-9]+", "", v.casefold())]
        if len(candidates) == 1:
            predicted, mapping = candidates[0], "exact_option_content"
    if predicted is None:
        return None
    expected = match.group(1).upper()
    return {"is_correct": predicted == expected, "predicted_option": predicted,
            "reference_option": expected, "mapping": mapping, "option_text": options.get(predicted)}


def _judge_prompt(evaluator: dict[str, Any], prediction: Any, prompt_path: Path) -> str:
    template = prompt_path.read_text(encoding="utf-8")
    reference = evaluator.get("answer")
    if isinstance(reference, dict):
        reference = reference.get("answer")
    return (template.replace("{problem}", str(evaluator.get("official_problem") or evaluator.get("question") or "").strip())
            .replace("{assistant_answer}", str(prediction or "").strip())
            .replace("{reference_answer}", str(reference or "").strip()))


def _decode_judge(raw: str) -> tuple[bool, str]:
    if "[reason]" not in raw or "[judge]" not in raw:
        raise ValueError("judge output is missing [reason] or [judge]")
    reason = raw.split("[judge]", 1)[0].split("[reason]", 1)[-1].strip()
    verdict = re.search(r"\b(True|False)\b", raw.split("[judge]", 1)[-1], re.I)
    if not verdict:
        raise ValueError("judge output is missing a boolean verdict")
    return verdict.group(1).lower() == "true", reason[:500]


def score_open_answer(
    evaluator: dict[str, Any], prediction: Any, *, base_url: str, api_key: str,
    model: str = DEFAULT_JUDGE_MODEL, timeout: int = 180, retries: int = 3,
    api_surface: str = "responses", prompt_path: Path = DEFAULT_PROMPT,
) -> dict[str, Any]:
    if not str(prediction or "").strip():
        return {"status": "scored", "score": 0.0, "correct": False,
                "scorer": PERCEPTION_SCORER, "reason": "failed to obtain answer"}
    choice = _deterministic_open_choice(evaluator, prediction)
    if choice is not None:
        return {"status": "scored", "score": float(choice["is_correct"]),
                "correct": bool(choice["is_correct"]), "scorer": PERCEPTION_SCORER,
                "model": model, "scoring_route": "deterministic_multiple_choice_mapping",
                "reason": f"Student {choice['predicted_option']} vs. reference {choice['reference_option']}.",
                "choice_mapping": choice}
    if not api_key:
        raise ValueError("judge API key required for non-choice open answer")
    if not model or not base_url:
        raise ValueError("judge model and base URL required for non-choice open answer")
    prompt = _judge_prompt(evaluator, prediction, prompt_path)
    surface = api_surface.lower()
    if surface in {"chat", "chat_completions", "chat-completions"}:
        endpoint, body = "/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3,
            "max_completion_tokens" if model.startswith("gpt-5") else "max_tokens": 1024}
    else:
        endpoint, body = "/responses", {"model": model, "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}], "max_output_tokens": 1024, "stream": False}
        if model.startswith("gpt-5"):
            body["reasoning"] = {"effort": "low"}
    last: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        req = urllib.request.Request(base_url.rstrip("/") + endpoint, data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": os.environ.get("AGENTIC_HTTP_USER_AGENT") or "OpenAI/Python",
            }, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            if endpoint == "/responses":
                raw = "\n".join(str(c.get("text") or "") for item in payload.get("output", [])
                    if isinstance(item, dict) for c in item.get("content", [])
                    if isinstance(c, dict) and c.get("type") == "output_text").strip()
            else:
                raw = str(payload["choices"][0]["message"]["content"])
            verdict, reason = _decode_judge(raw)
            return {"status": "scored", "score": float(verdict), "correct": verdict,
                    "scorer": PERCEPTION_SCORER, "model": model, "attempts": attempt + 1,
                    "reason": reason, "raw_judge_response": raw}
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries:
                time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"judge_failed:{type(last).__name__}:{last}")


def score_prediction(evaluator: dict[str, Any], prediction: Any, **kwargs: Any) -> dict[str, Any]:
    score_type = str((evaluator.get("score_spec") or {}).get("type") or "")
    if score_type == "normalized_exact_match":
        return score_exact_match(evaluator, prediction)
    if score_type == "extended_short_answer_judge_v1":
        result = score_open_answer(evaluator, prediction, **{**kwargs, "prompt_path": EXTENDED_PROMPT})
        result["scorer"] = score_type
        return result
    if score_type in {PERCEPTION_SCORER, "perceptionbench_official_judge"}:
        return score_open_answer(evaluator, prediction, **kwargs)
    raise ValueError(f"unsupported_static_score_type:{score_type}")


def score_predictions(evaluator_rows: Iterable[dict[str, Any]], predictions: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
    results = []
    for row in evaluator_rows:
        sid = row.get("sample_id")
        try:
            result = score_prediction(row, predictions.get(sid), **kwargs)
        except Exception as exc:  # keep row-level diagnostics and continue
            result = {"status": "evaluator_error", "score": None, "correct": None,
                      "error": f"{type(exc).__name__}:{exc}"}
        results.append({"sample_id": sid, "benchmark": row.get("benchmark"), **result})
    return results
