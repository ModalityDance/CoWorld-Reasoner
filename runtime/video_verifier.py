"""VLM client for observations of generated video.

The configured VLM returns structured evidence descriptions without requesting option
letters, correctness judgments, or final answers.
"""

from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

from .contracts import ToolRequest, ToolReturn
from .agents_sdk_model import get_vlm_concurrency_gate
from .model_config import (
    default_live_base_url,
    default_live_model,
    live_api_key,
    live_model_candidates,
)
from .local_env import load_local_env


load_local_env()


def verifier_model_candidates(model: str | None) -> list[str]:
    raw_key = (
        "AGENTIC_VLM_MODEL_CANDIDATES"
        if os.environ.get("AGENTIC_VLM_MODEL_CANDIDATES")
        else "AGENTIC_LLM_MODEL_CANDIDATES"
    )
    return live_model_candidates(model, env_var=raw_key)


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _verifier_image_detail(*, controls: dict[str, Any]) -> str:
    requested = str(controls.get("image_detail") or "").strip().lower()
    return requested if requested in {"low", "auto", "high"} else "high"


def _int_control_or_env(
    controls: dict[str, Any],
    key: str,
    env_key: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    raw = controls.get(key)
    if raw is None:
        raw = os.environ.get(env_key)
    try:
        value = int(raw) if raw is not None and str(raw).strip() else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, value)


def _evenly_sample(values: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return list(values)
    if limit == 1:
        # A generated sequence usually starts at an observed anchor. When only
        # one predicted frame fits, the temporal midpoint carries more of the
        # missing-state evidence than that duplicated endpoint.
        return [values[len(values) // 2]]
    last = len(values) - 1
    indices = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    sampled = [values[index] for index in indices]
    return sampled[:limit]


def _select_generated_frames(
    artifact_refs: list[str],
    controls: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Sample generated frames in temporal order within the image budget."""
    frame_budget = _int_control_or_env(
        controls,
        "verifier_frame_budget",
        "AGENTIC_VERIFIER_FRAME_BUDGET",
        9,
        minimum=1,
    )

    manifest_by_path = {
        str(item.get("path") or item.get("artifact_ref") or ""): item
        for item in controls.get("attachment_manifest") or []
        if isinstance(item, dict)
    }
    predicted: list[str] = []
    for ref in artifact_refs:
        item = manifest_by_path.get(str(ref)) or {}
        source_type = str(item.get("source_type") or "").lower()
        oracle_status = str(item.get("oracle_status") or "").lower()
        evidence_modality = str(item.get("evidence_modality") or "").lower()
        if (
            "observed" not in source_type
            and oracle_status not in {"observed_input", "observed_anchor"}
            and ("predicted" in source_type or "predicted" in oracle_status
                 or evidence_modality == "video_frame")
        ):
            predicted.append(ref)

    selected = _evenly_sample(predicted, frame_budget)
    return selected, {
        "artifact_refs_original_count": len(artifact_refs),
        "artifact_refs_used_count": len(selected),
        "predicted_refs_count": len(predicted),
        "frame_budget": frame_budget,
        "policy": "generated_frames_in_temporal_order",
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        start = 0
        while True:
            start = cleaned.find("{", start)
            if start < 0:
                break
            try:
                candidate, _ = decoder.raw_decode(cleaned[start:])
            except json.JSONDecodeError:
                start += 1
                continue
            if isinstance(candidate, dict):
                candidates.append(candidate)
            start += 1
        if not candidates:
            raise

        # Providers occasionally emit a short JSON progress object followed by
        # the actual evidence object. Prefer the latter by its semantic shape,
        # while retaining support for a single object wrapped in prose.
        def evidence_shape(value: dict[str, Any]) -> tuple[int, int, int, int]:
            return (
                int(
                    value.get("schema_version") == "generated_evidence_observations_v2"
                ),
                int(isinstance(value.get("observations"), list)),
                int(isinstance(value.get("request_execution"), dict)),
                int("summary" in value),
            )

        parsed = max(candidates, key=evidence_shape)
    if isinstance(parsed, str):
        nested_text = parsed.strip()
        if nested_text.startswith("{"):
            try:
                nested_parsed = json.loads(nested_text)
                if isinstance(nested_parsed, dict):
                    parsed = nested_parsed
            except json.JSONDecodeError:
                pass
    if not isinstance(parsed, dict):
        raise ValueError("video verifier response is not a JSON object")
    return parsed


def _compact_attachment_manifest(controls: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in controls.get("attachment_manifest") or []:
        if not isinstance(item, dict):
            continue
        artifact_ref = item.get("artifact_ref")
        if artifact_ref:
            artifact_ref = Path(str(artifact_ref)).name
        compact.append(
            {
                key: value
                for key, value in {
                    "evidence_id": item.get("evidence_id"),
                    "source_type": item.get("source_type"),
                    "evidence_modality": item.get("evidence_modality"),
                    "role": item.get("role"),
                    "task_role": item.get("task_role"),
                    "candidate_label": item.get("candidate_label"),
                    "image_attachment_index": item.get("image_attachment_index"),
                    "frame_index_in_video": item.get("frame_index_in_video"),
                    "frame_count_in_video": item.get("frame_count_in_video"),
                    "oracle_status": item.get("oracle_status"),
                    "artifact_ref": artifact_ref,
                }.items()
                if value is not None
            }
        )
    return compact


def _align_evidence_attachments(
    artifact_refs: list[str],
    controls: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Bind the review manifest to the images actually sent after frame selection."""
    by_path = {
        str(item.get("path") or item.get("artifact_ref") or ""): item
        for item in controls.get("attachment_manifest") or []
        if isinstance(item, dict)
    }
    selected = [
        ref
        for ref in artifact_refs
        if Path(ref).is_file()
        and Path(ref).suffix.lower() not in {".mp4", ".mov", ".webm", ".mkv"}
    ]
    manifest = [
        {
            **by_path.get(str(ref), {}),
            "path": str(ref),
            "image_attachment_index": index,
            "evidence_id": by_path.get(str(ref), {}).get("evidence_id")
            or f"attachment_{index:02d}",
        }
        for index, ref in enumerate(selected, 1)
    ]
    return selected, {
        **controls,
        "artifact_refs": selected,
        "attachment_manifest": manifest,
    }


def validate_observation_response(
    parsed: dict[str, Any],
    controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate generated-frame observations and their attached evidence sources."""
    from evolved_harness.verifier_observations import validate_verifier_observations

    return validate_verifier_observations(parsed, controls or {})


def _verifier_retry_count(transient_retries: int | None) -> int:
    if transient_retries is not None:
        return max(0, int(transient_retries))
    try:
        return max(
            0,
            int(os.environ.get("AGENTIC_VERIFIER_TRANSIENT_RETRIES", "3")),
        )
    except ValueError:
        return 3


def _verifier_retry_delay(attempt_index: int) -> float:
    try:
        base_delay = max(
            0.0,
            float(os.environ.get("AGENTIC_VERIFIER_RETRY_DELAY_SEC", "2")),
        )
    except ValueError:
        base_delay = 2.0
    try:
        max_delay = max(
            base_delay,
            float(
                os.environ.get("AGENTIC_VERIFIER_MAX_RETRY_DELAY_SEC", "20")
            ),
        )
    except ValueError:
        max_delay = 20.0
    return min(max_delay, base_delay * (2 ** max(0, attempt_index)))


def _verifier_http_error_text(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(2048)
    except Exception:
        body = b""
    try:
        body_text = body.decode("utf-8", errors="replace")
    except Exception:
        body_text = ""
    return f"HTTP {exc.code}: {body_text or exc.reason or exc.msg}"


def _verifier_error_is_transient(exc: BaseException, message: str) -> bool:
    lower = message.lower()
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {429, 500, 502, 503, 504}:
            return True
        if exc.code == 400 and any(
            token in lower
            for token in ("upstream", "temporar", "timeout", "rate limit")
        ):
            return True
        return False
    if isinstance(
        exc,
        (
            urllib.error.URLError,
            TimeoutError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
        ),
    ):
        return True
    if isinstance(exc, OSError):
        return any(
            token in lower
            for token in (
                "timed out",
                "timeout",
                "temporar",
                "connection reset",
                "remote end closed",
                "connection aborted",
                "broken pipe",
                "network is unreachable",
            )
        )
    return False


def _verifier_post_json(
    *,
    url: str,
    api_key: str,
    body: dict[str, Any],
    timeout: int,
    transient_retries: int | None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": os.environ.get("AGENTIC_HTTP_USER_AGENT") or "OpenAI/Python",
        },
        method="POST",
    )
    max_attempts = _verifier_retry_count(transient_retries) + 1
    last_error: str | None = None
    gate = get_vlm_concurrency_gate()
    for attempt_index in range(max_attempts):
        gate.acquire_sync()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise RuntimeError("verifier_provider_response_must_be_json_object")
                return payload
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            last_error = f"verifier_provider_invalid_json:{type(exc).__name__}"
            if attempt_index >= max_attempts - 1:
                raise RuntimeError(last_error) from exc
        except urllib.error.HTTPError as exc:
            message = _verifier_http_error_text(exc)
            last_error = message
            if attempt_index >= max_attempts - 1 or not _verifier_error_is_transient(
                exc, message
            ):
                raise RuntimeError(message) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            OSError,
        ) as exc:
            message = str(exc)
            last_error = f"{type(exc).__name__}: {message}"
            if attempt_index >= max_attempts - 1 or not _verifier_error_is_transient(
                exc, message
            ):
                raise RuntimeError(last_error) from exc
        finally:
            gate.release()
        time.sleep(_verifier_retry_delay(attempt_index))
    raise RuntimeError(last_error or "video_verifier_request_failed")


class _ProviderText(str):
    provider_usage: dict[str, int]

    def __new__(cls, value: str, provider_usage: dict[str, int]) -> "_ProviderText":
        instance = super().__new__(cls, value)
        instance.provider_usage = provider_usage
        return instance


def _normalized_provider_usage(raw_usage: Any) -> dict[str, int]:
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    input_details = (
        usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    )
    output_details = (
        usage.get("output_tokens_details")
        or usage.get("completion_tokens_details")
        or {}
    )
    return {
        "requests": 1,
        "input_tokens": int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        ),
        "cached_input_tokens": int(input_details.get("cached_tokens") or 0),
        "output_tokens": int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        ),
        "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def sampling_controls(model: str) -> dict[str, Any]:
    return {"temperature": 0.0} if model == "gpt-5.6-luna" else {}


def _call_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
    transient_retries: int | None = None,
    response_schema: dict[str, Any] | None = None,
) -> str:

    body = {
        "model": model,
        "messages": messages,
        "max_completion_tokens"
        if model.startswith("gpt-5")
        else "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    body.update(sampling_controls(model))
    if response_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "generated_evidence_observations",
                "schema": response_schema,
                "strict": True,
            },
        }
    if model.startswith("gpt-5"):
        body["reasoning_effort"] = reasoning_effort or os.environ.get(
            "AGENTIC_LLM_REASONING_EFFORT", "low"
        )
    data = _verifier_post_json(
        url=base_url.rstrip("/") + "/chat/completions",
        api_key=api_key,
        body=body,
        timeout=timeout,
        transient_retries=transient_retries,
    )
    return _ProviderText(
        str(data["choices"][0]["message"]["content"]),
        _normalized_provider_usage(data.get("usage")),
    )


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        raw_content = message.get("content")
        parts = (
            raw_content
            if isinstance(raw_content, list)
            else [{"type": "text", "text": str(raw_content or "")}]
        )
        content: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                content.append({"type": "input_text", "text": str(part)})
                continue
            part_type = str(part.get("type") or "")
            if part_type in {"text", "input_text"}:
                content.append(
                    {"type": "input_text", "text": str(part.get("text") or "")}
                )
            elif part_type in {"image_url", "input_image"}:
                image = part.get("image_url")
                image_url = image.get("url") if isinstance(image, dict) else image
                detail = (
                    image.get("detail")
                    if isinstance(image, dict)
                    else part.get("detail")
                )
                converted_image = {
                    "type": "input_image",
                    "image_url": str(image_url or ""),
                }
                if detail:
                    converted_image["detail"] = str(detail)
                content.append(converted_image)
        converted.append(
            {"role": str(message.get("role") or "user"), "content": content}
        )
    return converted


def _call_responses(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
    transient_retries: int | None = None,
    response_schema: dict[str, Any] | None = None,
) -> str:
    body: dict[str, Any] = {
        "model": model,
        "input": _responses_input(messages),
        "max_output_tokens": max_tokens,
        "text": {"format": {"type": "json_object"}},
        "stream": False,
    }
    body.update(sampling_controls(model))
    if response_schema is not None:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "generated_evidence_observations",
                "schema": response_schema,
                "strict": True,
            }
        }
    if model.startswith("gpt-5"):
        body["reasoning"] = {
            "effort": reasoning_effort
            or os.environ.get("AGENTIC_LLM_REASONING_EFFORT", "low")
        }
    data = _verifier_post_json(
        url=base_url.rstrip("/") + "/responses",
        api_key=api_key,
        body=body,
        timeout=timeout,
        transient_retries=transient_retries,
    )
    output_text = [
        str(content.get("text") or "")
        for item in data.get("output") or []
        if isinstance(item, dict)
        for content in item.get("content") or []
        if isinstance(content, dict) and content.get("type") == "output_text"
    ]
    text = "\n".join(part for part in output_text if part).strip()
    if not text:
        raise RuntimeError("video_verifier_responses_output_text_missing")
    return _ProviderText(text, _normalized_provider_usage(data.get("usage")))


def _verifier_prompt(request: ToolRequest) -> str:
    from evolved_harness.verifier_observations import build_verifier_prompt

    return build_verifier_prompt(request.controls, _compact_attachment_manifest(request.controls))


def verify_generated_video(request: ToolRequest) -> ToolReturn:
    """Describe generated frames using the configured verifier VLM."""
    controls = dict(request.controls)
    controls.setdefault("observation_schema", "generated_evidence_observations_v2")
    controls.setdefault("answer_free", True)
    controls.setdefault(
        "verifier_evidence_id",
        f"video_verifier::{request.request_id or 'runtime'}",
    )
    base_url = (
        os.environ.get("AGENTIC_VLM_BASE_URL")
        or os.environ.get("AGENTIC_LLM_BASE_URL")
        or default_live_base_url()
    )
    api_key = live_api_key(
        "AGENTIC_VLM_API_KEY", "AGENTIC_LLM_API_KEY", "OPENAI_API_KEY"
    )
    model = (
        os.environ.get("AGENTIC_VLM_MODEL")
        or os.environ.get("AGENTIC_LLM_MODEL")
        or default_live_model()
    )
    api_surface = (
        (
            os.environ.get("AGENTIC_VERIFIER_API")
            or os.environ.get("AGENTIC_AGENTS_SDK_MODEL_API")
            or os.environ.get("AGENTIC_OPENAI_MODEL_API")
            or "chat"
        )
        .strip()
        .lower()
    )
    call_provider = (
        _call_chat_completions
        if api_surface in {"chat", "chat_completions", "chat-completions"}
        else _call_responses
    )
    verifier_id = str(
        controls.get("verifier_evidence_id")
        or f"verifier::{request.request_id or 'video_verifier'}"
    )
    original_artifact_refs = [
        str(path) for path in controls.get("artifact_refs") or [] if path
    ]
    artifact_refs, artifact_selection_metadata = (
        _select_generated_frames(
            original_artifact_refs,
            controls,
        )
    )
    artifact_refs, controls = _align_evidence_attachments(artifact_refs, controls)
    artifact_selection_metadata.update(
        artifact_refs_used_count=len(artifact_refs),
        delivered_attachment_manifest=_compact_attachment_manifest(controls),
        attachment_alignment="selected_images_with_inline_evidence_ids",
    )
    request_for_model = replace(request, controls=controls)
    prompt = _verifier_prompt(request_for_model)
    from evolved_harness.verifier_observations import VerifierObservations
    from agents.strict_schema import ensure_strict_json_schema

    response_schema = ensure_strict_json_schema(VerifierObservations.model_json_schema())
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    image_detail = _verifier_image_detail(controls=controls)
    reasoning_effort = str(controls.get("reasoning_effort") or "").strip().lower()
    if reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        reasoning_effort = "medium"
    for index, path_text in enumerate(artifact_refs):
        path = Path(path_text)
        if (
            path.exists()
            and path.is_file()
            and path.suffix.lower() not in {".mp4", ".mov", ".webm", ".mkv"}
        ):
            item = controls["attachment_manifest"][index]
            label = {
                key: item[key]
                for key in (
                    "image_attachment_index",
                    "evidence_id",
                    "source_type",
                    "frame_index_in_video",
                )
                if key in item
            }
            content.append({"type": "text", "text": json.dumps(label, sort_keys=True)})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(path), "detail": image_detail},
                }
            )
    failure_modes: list[str] = []
    parsed: dict[str, Any] = {}
    raw_preview = ""
    status = "success"
    model_candidates = verifier_model_candidates(model)
    model_errors: list[dict[str, str]] = []
    model_used: str | None = None
    provider_usage: dict[str, int] = {}
    if not base_url or not api_key:
        status = "video_verifier_not_configured"
        failure_modes.append("video_verifier_api_not_configured")
    else:
        raw: str | None = None
        for candidate_model in model_candidates:
            try:
                raw = call_provider(
                    base_url=base_url,
                    api_key=api_key,
                    model=candidate_model,
                    messages=[{"role": "user", "content": content}],
                    response_schema=response_schema,
                    max_tokens=int(controls.get("max_tokens") or 512),
                    timeout=int(controls.get("request_timeout") or 180),
                    reasoning_effort=reasoning_effort,
                    transient_retries=int(controls.get("transient_retries", 2)),
                )
                provider_usage = dict(getattr(raw, "provider_usage", {}) or {})
                model_used = candidate_model
                break
            except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
                model_errors.append({"model": candidate_model, "error": str(exc)[:500]})
        try:
            if raw is None:
                raise RuntimeError(
                    model_errors[-1]["error"]
                    if model_errors
                    else "unknown_verifier_model_error"
                )
            raw_preview = str(raw)[:1000]
            try:
                parsed = _extract_json_object(raw)
            except (ValueError, json.JSONDecodeError):
                parsed = {"unstructured_response": str(raw).strip()}
            quality_check = validate_observation_response(parsed, controls)
            if not quality_check["is_usable"]:
                status = "video_verifier_low_quality"
                failure_modes.extend(quality_check["low_quality_reasons"])
        except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
            status = "video_verifier_failed"
            failure_modes.append(f"video_verifier_failed:{type(exc).__name__}")
            if not model_errors:
                model_errors.append({"model": model, "error": str(exc)[:500]})
            parsed = {"error": str(exc)[:500]}
            quality_check = {
                "is_usable": False,
                "low_quality_reasons": list(failure_modes),
            }
    if not parsed:
        quality_check = {"is_usable": False, "low_quality_reasons": list(failure_modes)}
    success = status == "success"
    return ToolReturn(
        tool_return_id=f"video_verifier_return::{request.request_id or 'runtime'}",
        tool_interface=request.tool_interface,
        status=status,
        input_evidence_ids=list(request.input_evidence_ids),
        output_evidence_ids=[verifier_id] if success else [],
        artifacts={"artifact_refs": artifact_refs},
        provenance_label="online_answer_free_video_verifier",
        quality={
            "overall_usability": "review_available" if success else "unusable",
            "verifier_decision": "describe_evidence_only",
            "verifier_type": controls.get("observation_schema"),
            "observations": parsed,
            "observation_validation": quality_check,
            "prompt_char_count": len(prompt),
        },
        failure_modes=failure_modes,
        cost={
            "video_verifier_calls": 1 if base_url and api_key else 0,
            "video_verifier_model_attempts": len(model_errors)
            + (1 if model_used else 0),
        },
        permission_decision={
            "executed": success,
            "policy": "online_answer_free_verifier",
        },
        execution_trace={
            "request_id": request.request_id,
            "step_index": request.step_index,
            "requested_by_skill_id": request.requested_by_skill_id,
            "verifier_type": controls.get("observation_schema"),
            "model_used": model_used,
            "api_surface": api_surface,
            "provider_usage": provider_usage,
            "prompt_char_count": len(prompt),
        },
        raw_backend_metadata={
            "video_verifier_response": parsed,
            "raw_response_preview": raw_preview,
            "model_candidates_attempted": model_candidates
            if base_url and api_key
            else [],
            "model_used": model_used,
            "api_surface": api_surface,
            "provider_usage": provider_usage,
            "model_errors": model_errors,
            "artifact_selection": artifact_selection_metadata,
            "image_detail": image_detail,
        },
    )
