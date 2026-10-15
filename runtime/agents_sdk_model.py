"""Agents SDK provider transport, request concurrency, and API compatibility."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from .vlm_concurrency_config import VLM_CONCURRENCY_LIMIT


from agents.models.interface import Model as AgentsModel


@dataclass(frozen=True)
class VLMRequestLease:
    waited_seconds: float
    active: int
    peak: int


class VLMConcurrencyGate:
    """Process-wide cap for in-flight VLM HTTP requests.

    Sample workers may remain alive while a synchronous VWM-driven tool waits. The gate
    is therefore placed around the provider request itself, not around the
    sample worker or the whole Agent trajectory.
    """

    def __init__(self, limit: int) -> None:
        if int(limit) < 1:
            raise ValueError("vlm_concurrency_limit_must_be_positive")
        if int(limit) > VLM_CONCURRENCY_LIMIT:
            raise ValueError(
                f"vlm_concurrency_limit_exceeds_global_cap:{VLM_CONCURRENCY_LIMIT}"
            )
        self.limit = int(limit)
        self._slots = threading.BoundedSemaphore(self.limit)
        self._lock = threading.Lock()
        self._active = 0
        self._peak = 0
        self._acquired = 0
        self._waited = 0
        self._waited_seconds = 0.0
        self._max_wait_seconds = 0.0

    async def acquire(self) -> VLMRequestLease:
        started = time.monotonic()
        waited = False
        while not self._slots.acquire(blocking=False):
            waited = True
            await asyncio.sleep(0.005)
        waited_seconds = time.monotonic() - started
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)
            self._acquired += 1
            if waited:
                self._waited += 1
                self._waited_seconds += waited_seconds
                self._max_wait_seconds = max(self._max_wait_seconds, waited_seconds)
            return VLMRequestLease(
                waited_seconds=waited_seconds,
                active=self._active,
                peak=self._peak,
            )

    def acquire_sync(self) -> VLMRequestLease:
        """Acquire a slot for synchronous provider paths in worker threads."""

        started = time.monotonic()
        waited = not self._slots.acquire(blocking=False)
        if waited:
            self._slots.acquire()
        waited_seconds = time.monotonic() - started
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)
            self._acquired += 1
            if waited:
                self._waited += 1
                self._waited_seconds += waited_seconds
                self._max_wait_seconds = max(self._max_wait_seconds, waited_seconds)
            return VLMRequestLease(
                waited_seconds=waited_seconds,
                active=self._active,
                peak=self._peak,
            )

    def release(self) -> None:
        with self._lock:
            if self._active < 1:
                raise RuntimeError("vlm_concurrency_gate_release_without_acquire")
            self._active -= 1
        self._slots.release()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "limit": self.limit,
                "active": self._active,
                "peak": self._peak,
                "acquired": self._acquired,
                "waited_requests": self._waited,
                "waited_seconds": round(self._waited_seconds, 6),
                "max_wait_seconds": round(self._max_wait_seconds, 6),
            }


_VLM_GATE_LOCK = threading.Lock()
_VLM_GATE: VLMConcurrencyGate | None = None


def get_vlm_concurrency_gate() -> VLMConcurrencyGate:
    global _VLM_GATE
    with _VLM_GATE_LOCK:
        if _VLM_GATE is None:
            configured = int(
                os.environ.get(
                    "AGENTIC_VLM_MAX_CONCURRENCY", str(VLM_CONCURRENCY_LIMIT)
                )
            )
            _VLM_GATE = VLMConcurrencyGate(configured)
        return _VLM_GATE


def _short_call_id(value: str, *, limit: int = 64) -> str:
    if len(value) <= limit:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:24]
    return f"call_{digest}"


def _normalize_call_ids(value: Any, mapping: dict[str, str] | None = None) -> Any:
    """Shorten tool call ids for OpenAI-compatible servers with 64-char caps."""

    call_id_map = {} if mapping is None else mapping
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"call_id", "tool_call_id"} and isinstance(item, str):
                short = call_id_map.setdefault(item, _short_call_id(item))
                normalized[key] = short
            elif (
                key == "id"
                and isinstance(item, str)
                and value.get("type")
                in {
                    "function_call",
                    "function_call_output",
                }
            ):
                short = call_id_map.setdefault(item, _short_call_id(item))
                normalized[key] = short
            else:
                normalized[key] = _normalize_call_ids(item, call_id_map)
        return normalized
    if isinstance(value, list):
        return [_normalize_call_ids(item, call_id_map) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_call_ids(item, call_id_map) for item in value)
    if hasattr(value, "model_copy"):
        item_type = getattr(value, "type", None)
        updates: dict[str, Any] = {}
        for attr in ("call_id", "tool_call_id"):
            current = getattr(value, attr, None)
            if isinstance(current, str):
                updates[attr] = call_id_map.setdefault(current, _short_call_id(current))
        current_id = getattr(value, "id", None)
        if item_type in {"function_call", "function_call_output"} and isinstance(
            current_id, str
        ):
            updates["id"] = call_id_map.setdefault(
                current_id, _short_call_id(current_id)
            )
        if updates:
            try:
                return value.model_copy(update=updates)
            except Exception:
                return value
        return value
    return value


def _promote_chat_tool_output_images(value: Any) -> Any:
    """Expose structured tool-result images through Chat Completions user content.

    The Agents SDK records images inside ``function_call_output.output``. Its
    Chat Completions converter intentionally drops those non-text parts because
    that API cannot attach images to a ``tool`` message. Keep the original tool
    result in place and add one user multimodal message after each contiguous
    tool-output group so the model receives the same visual artifacts. Strip
    images and transport-only attachment labels from the copied tool message;
    so each image appears once in the outgoing request.
    """

    if not isinstance(value, list):
        return value

    promoted: list[Any] = []
    pending_visual_content: list[dict[str, Any]] = []

    def flush_visual_content() -> None:
        if not pending_visual_content:
            return
        promoted.append({"role": "user", "content": list(pending_visual_content)})
        pending_visual_content.clear()

    for item in value:
        item_type = (
            item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        )
        if item_type != "function_call_output":
            flush_visual_content()
            promoted.append(item)
            continue

        output = (
            item.get("output")
            if isinstance(item, dict)
            else getattr(item, "output", None)
        )
        if not isinstance(output, (list, tuple)):
            promoted.append(item)
            continue
        images = [
            content
            for content in output
            if (
                content.get("type")
                if isinstance(content, dict)
                else getattr(content, "type", None)
            )
            == "input_image"
            and (
                content.get("image_url")
                if isinstance(content, dict)
                else getattr(content, "image_url", None)
            )
        ]
        if not images:
            promoted.append(item)
            continue

        call_id = (
            item.get("call_id")
            if isinstance(item, dict)
            else getattr(item, "call_id", None)
        )
        captions: dict[int, str] = {}
        caption = ""
        image_position = 0
        retained_output: list[Any] = []
        for part in output:
            part_type = (
                part.get("type")
                if isinstance(part, dict)
                else getattr(part, "type", None)
            )
            if part_type == "input_text":
                text = (
                    part.get("text")
                    if isinstance(part, dict)
                    else getattr(part, "text", "")
                )
                try:
                    parsed = json.loads(text)
                except (ValueError, TypeError):
                    parsed = {}
                if isinstance(parsed, dict) and isinstance(
                    parsed.get("image_attachment_label"), str
                ):
                    caption = parsed["image_attachment_label"]
                else:
                    retained_output.append(part)
            elif part_type == "input_image":
                image_position += 1
                if caption:
                    captions[image_position] = caption
                    caption = ""
            else:
                retained_output.append(part)
        if not retained_output:
            retained_output = [
                {
                    "type": "input_text",
                    "text": "Visual attachments are delivered in the following user message.",
                }
            ]
        if isinstance(item, dict):
            promoted.append({**item, "output": retained_output})
        else:
            try:
                promoted.append(item.model_copy(update={"output": retained_output}))
            except Exception:
                promoted.append(item)
        for image_index, image in enumerate(images, start=1):
            pending_visual_content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"Tool output visual attachment from call {call_id or 'unknown'} "
                        f"({image_index}/{len(images)})."
                        + (
                            f" {captions[image_index]}"
                            if image_index in captions
                            else ""
                        )
                    ),
                }
            )
            image_url = (
                image.get("image_url")
                if isinstance(image, dict)
                else getattr(image, "image_url", None)
            )
            image_part: dict[str, Any] = {
                "type": "input_image",
                "image_url": image_url,
            }
            detail = (
                image.get("detail")
                if isinstance(image, dict)
                else getattr(image, "detail", None)
            )
            if detail is not None:
                image_part["detail"] = detail
            pending_visual_content.append(image_part)

    flush_visual_content()
    return promoted


def _looks_like_agent_output(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(
        {
            "answer_bearing_relation",
            "decision_reason",
            "observations",
            "answer",
            "candidate_assessments",
        }.intersection(value)
    )


def _extract_agent_json_text(text: str) -> str:
    """Return embedded agent JSON when a compatible model adds prose around it.

    Strip surrounding prose only when a parseable JSON object contains
    recognized agent output fields.
    """

    stripped = text.strip()
    if not stripped or "{" not in stripped:
        return text
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        return stripped if _looks_like_agent_output(parsed) else text

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if _looks_like_agent_output(parsed):
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        if end:
            continue
    return text


def _sanitize_embedded_json_output(value: Any) -> Any:
    """Clean model output text before Agents SDK structured-output parsing."""

    if isinstance(value, dict):
        item_type = value.get("type")
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if (
                key == "text"
                and isinstance(item, str)
                and item_type in {"output_text", "text"}
            ):
                sanitized[key] = _extract_agent_json_text(item)
            else:
                sanitized[key] = _sanitize_embedded_json_output(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_embedded_json_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_embedded_json_output(item) for item in value)
    if hasattr(value, "model_copy"):
        updates: dict[str, Any] = {}
        item_type = getattr(value, "type", None)
        text = getattr(value, "text", None)
        if isinstance(text, str):
            cleaned = _extract_agent_json_text(text)
            if cleaned != text:
                updates["text"] = cleaned
        for attr in ("content", "output"):
            current = getattr(value, attr, None)
            if isinstance(current, (dict, list, tuple)):
                cleaned = _sanitize_embedded_json_output(current)
                if cleaned != current:
                    updates[attr] = cleaned
        if updates:
            try:
                return value.model_copy(update=updates)
            except Exception:
                return value
        return value
    return value


def _agents_sdk_client_retry_count(env: dict[str, str] | None = None) -> int:
    values = os.environ if env is None else env
    raw = (
        values.get("AGENTIC_AGENTS_SDK_CLIENT_MAX_RETRIES")
        or values.get("AGENTIC_LLM_CLIENT_MAX_RETRIES")
        or "4"
    )
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 4


def _agents_sdk_client_retry_delay(
    attempt_index: int, env: dict[str, str] | None = None
) -> float:
    values = os.environ if env is None else env
    try:
        base_delay = max(
            0.0,
            float(values.get("AGENTIC_AGENTS_SDK_CLIENT_RETRY_DELAY_SEC", "2")),
        )
    except (TypeError, ValueError):
        base_delay = 2.0
    try:
        max_delay = max(
            base_delay,
            float(values.get("AGENTIC_AGENTS_SDK_CLIENT_MAX_RETRY_DELAY_SEC", "8")),
        )
    except (TypeError, ValueError):
        max_delay = 8.0
    return min(max_delay, base_delay * (2 ** max(0, attempt_index)))


def _is_agents_sdk_transient_model_error(exc: BaseException) -> bool:
    name = exc.__class__.__name__
    text = str(exc).lower()
    if name in {
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "RateLimitError",
        "InternalServerError",
        "TimeoutError",
    }:
        return True
    return any(
        token in text
        for token in (
            "connection error",
            "timed out",
            "timeout",
            "temporarily",
            "connection reset",
            "remote end closed",
            "502",
            "503",
            "504",
            "rate limit",
        )
    )


def _response_from_sse_text(text: str) -> Any:
    """Convert nonstandard Responses SSE text into an OpenAI Response object."""

    if not isinstance(text, str) or "event:" not in text or "data:" not in text:
        return text
    from openai.types.responses import Response

    completed_response: dict[str, Any] | None = None
    output_items: list[dict[str, Any]] = []
    for block in text.strip().split("\n\n"):
        data_lines = [
            line[len("data:") :].strip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        payload_text = "\n".join(data_lines).strip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        event_type = str(payload.get("type") or "")
        if event_type == "response.output_item.done" and isinstance(
            payload.get("item"), dict
        ):
            output_items.append(payload["item"])
        elif event_type == "response.completed" and isinstance(
            payload.get("response"), dict
        ):
            completed_response = dict(payload["response"])
    if completed_response is None:
        return text
    if not completed_response.get("output") and output_items:
        completed_response["output"] = output_items
    return Response.model_validate(completed_response)


class _ResponsesSSECompatResource:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        response = await self.inner.create(*args, **kwargs)
        if isinstance(response, str):
            return _response_from_sse_text(response)
        return response


class _ResponsesSSECompatClient:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.responses = _ResponsesSSECompatResource(inner.responses)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class AgentsSDKCompatModel(AgentsModel):
    """Agents SDK model wrapper for transport compatibility and bounded retries."""

    def __init__(
        self,
        inner: Any,
        *,
        client: Any | None = None,
        max_retries: int | None = None,
        timeout_seconds: int | float | None = None,
        promote_tool_output_images: bool = False,
        request_gate: VLMConcurrencyGate | None = None,
    ) -> None:
        self.inner = inner
        self.client = client
        self.max_retries = (
            _agents_sdk_client_retry_count()
            if max_retries is None
            else max(0, int(max_retries))
        )
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None and float(timeout_seconds) > 0
            else None
        )
        self.promote_tool_output_images = promote_tool_output_images
        self.request_gate = request_gate

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def close(self) -> Any:
        client_close = getattr(self.client, "close", None)
        if client_close is not None:
            result = client_close()
            if hasattr(result, "__await__"):
                return await result
            return result
        client_aclose = getattr(self.client, "aclose", None)
        if client_aclose is not None:
            return await client_aclose()
        close = getattr(self.inner, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                return await result
            return result
        return None

    def get_retry_advice(self, request: Any) -> Any | None:
        get_retry_advice = getattr(self.inner, "get_retry_advice", None)
        if get_retry_advice is None:
            return None
        return get_retry_advice(request)

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        normalized_args = list(args)
        if len(normalized_args) >= 2:
            normalized_args[1] = _normalize_call_ids(copy.deepcopy(normalized_args[1]))
            if self.promote_tool_output_images:
                normalized_args[1] = _promote_chat_tool_output_images(
                    normalized_args[1]
                )
        elif "input" in kwargs:
            kwargs = dict(kwargs)
            kwargs["input"] = _normalize_call_ids(copy.deepcopy(kwargs["input"]))
            if self.promote_tool_output_images:
                kwargs["input"] = _promote_chat_tool_output_images(kwargs["input"])
        response = None
        for attempt_index in range(self.max_retries + 1):
            lease = (
                await self.request_gate.acquire()
                if self.request_gate is not None
                else None
            )
            retry_delay = 0.0
            try:
                call = self.inner.get_response(*normalized_args, **kwargs)
                if self.timeout_seconds is not None:
                    response = await asyncio.wait_for(
                        call, timeout=self.timeout_seconds
                    )
                else:
                    response = await call
                break
            except Exception as exc:
                if (
                    attempt_index >= self.max_retries
                    or not _is_agents_sdk_transient_model_error(exc)
                ):
                    raise
                retry_delay = _agents_sdk_client_retry_delay(attempt_index)
            finally:
                if lease is not None:
                    self.request_gate.release()
            if retry_delay:
                await asyncio.sleep(retry_delay)
        if response is None:
            raise RuntimeError("agents_sdk_model_response_missing_after_retry")
        output = getattr(response, "output", None)
        if output is not None:
            response.output = _normalize_call_ids(
                _sanitize_embedded_json_output(output)
            )
        return response

    async def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        normalized_args = list(args)
        if len(normalized_args) >= 2:
            normalized_args[1] = _normalize_call_ids(copy.deepcopy(normalized_args[1]))
            if self.promote_tool_output_images:
                normalized_args[1] = _promote_chat_tool_output_images(
                    normalized_args[1]
                )
        elif "input" in kwargs:
            kwargs = dict(kwargs)
            kwargs["input"] = _normalize_call_ids(copy.deepcopy(kwargs["input"]))
            if self.promote_tool_output_images:
                kwargs["input"] = _promote_chat_tool_output_images(kwargs["input"])
        for attempt_index in range(self.max_retries + 1):
            lease = (
                await self.request_gate.acquire()
                if self.request_gate is not None
                else None
            )
            retry_delay = 0.0
            try:
                stream = self.inner.stream_response(*normalized_args, **kwargs)
                while True:
                    try:
                        if self.timeout_seconds is not None:
                            event = await asyncio.wait_for(
                                stream.__anext__(),
                                timeout=self.timeout_seconds,
                            )
                        else:
                            event = await stream.__anext__()
                    except StopAsyncIteration:
                        break
                    yield _normalize_call_ids(
                        _sanitize_embedded_json_output(event)
                    )
                break
            except Exception as exc:
                if (
                    attempt_index >= self.max_retries
                    or not _is_agents_sdk_transient_model_error(exc)
                ):
                    raise
                retry_delay = _agents_sdk_client_retry_delay(attempt_index)
            finally:
                if lease is not None:
                    self.request_gate.release()
            if retry_delay:
                await asyncio.sleep(retry_delay)
