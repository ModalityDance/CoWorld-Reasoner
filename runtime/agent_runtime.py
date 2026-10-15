"""Agents SDK loop for model transport and agent-selected tool execution."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from runtime.agents_sdk_input import image_data_url
from runtime.agents_sdk_model import (
    AgentsSDKCompatModel,
    _ResponsesSSECompatClient,
    get_vlm_concurrency_gate,
)
from runtime.model_config import shared_vlm_model


def configured_model() -> str:
    """Resolve the model after local dotenv loading as well as at import time."""

    return shared_vlm_model()


REASONING_EFFORT = "medium"
TEMPERATURE = 0.0
PROVIDER_MAX_ATTEMPTS = 3


def _provider_retry_delay(exc: Exception, attempt: int) -> float:
    """Honor provider backoff hints before retrying transient model failures."""

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return min(180.0, max(1.0, float(raw)))
            except (TypeError, ValueError):
                pass
    # Keep retries bounded while allowing the shared gateway queue to drain.
    return min(120.0, 15.0 * (2 ** (attempt - 1)))


CORE_SYSTEM_PROMPT = """You are one autonomous multimodal reasoning agent.
Reason from the task, original observed media, and observations returned by the capabilities that
are actually available in this run. You autonomously decide whether, when, and how to use an
available capability. There is no required action sequence or action count. Use a capability when
it can expose decision-relevant evidence that the current observations do not provide; answer from
the current observations when they are sufficient or the available capability is incompatible.
Returned observations are evidence, not answers or oracle truth. Integrate useful evidence, keep
uncertainty when evidence is weak, and finish with the typed task output.

Use only the supplied task premises, observations and available tools. Make the reasoning
decisions and answer selection yourself. Keep the final answer in the exact compact format
requested by the task.
"""


@dataclass
class AgentRuntimeContext:
    output_dir: Path
    observed_image_paths: list[Path]
    observed_image_roles: list[str]
    interleaved_observations: bool = field(default=False, kw_only=True)
    started_at: float = field(default_factory=time.time)
    provider_errors: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append(
            {
                "event_index": len(self.events),
                "event_type": event_type,
                "elapsed_sec": round(time.time() - self.started_at, 3),
                "payload": payload,
            }
        )


@dataclass(frozen=True)
class AgentRunResult:
    final_output: dict[str, Any] | None
    raw_final_output: str | None
    events: list[dict[str, Any]]
    usage: dict[str, int]
    provider_errors: list[dict[str, Any]]
    error: str | None = None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _strict_args(raw: str, required: set[str], allowed: set[str]) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("tool_arguments_must_be_object")
    missing = sorted(required - set(parsed))
    unknown = sorted(set(parsed) - allowed)
    if missing:
        raise ValueError(f"tool_arguments_missing:{missing}")
    if unknown:
        raise ValueError(f"tool_arguments_unknown:{unknown}")
    return parsed


class _TraceHooks:
    """Lazily subclass RunHooks so importing the module stays dependency-light."""

    @staticmethod
    def build(runtime_context: AgentRuntimeContext) -> Any:
        from agents import RunHooks

        class Hooks(RunHooks[AgentRuntimeContext]):
            def __init__(self) -> None:
                self.provider_error_cursor = 0

            async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
                usage = _jsonable(response.usage)
                new_provider_errors = runtime_context.provider_errors[
                    self.provider_error_cursor :
                ]
                self.provider_error_cursor = len(runtime_context.provider_errors)
                runtime_context.emit(
                    "model_turn",
                    {
                        "agent_name": str(agent.name),
                        "response_id": response.response_id,
                        "request_id": response.request_id,
                        "output": _jsonable(response.output),
                        "usage": usage,
                        "provider_attempts": 1 + len(new_provider_errors),
                        "provider_errors": list(new_provider_errors),
                        "parse_error": None,
                    },
                )

        return Hooks()


class _RetryLoggingModel(AgentsSDKCompatModel):
    """Retry only transient provider failures with an identical SDK request."""

    def __init__(
        self,
        inner: Any,
        *,
        client: Any,
        context: AgentRuntimeContext,
        timeout_sec: int,
        promote_tool_output_images: bool = True,
        request_gate: Any | None = None,
    ) -> None:
        self.context = context
        self.api_surface = "chat" if promote_tool_output_images else "responses"
        super().__init__(
            inner,
            client=client,
            max_retries=0,
            timeout_seconds=timeout_sec,
            promote_tool_output_images=promote_tool_output_images,
            request_gate=request_gate,
        )

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        from runtime.agents_sdk_model import (
            _agents_sdk_client_retry_count,
            _is_agents_sdk_transient_model_error,
        )

        configured = any(
            name in os.environ
            for name in (
                "AGENTIC_AGENTS_SDK_CLIENT_MAX_RETRIES",
                "AGENTIC_LLM_CLIENT_MAX_RETRIES",
            )
        )
        max_attempts = (
            _agents_sdk_client_retry_count() + 1
            if configured
            else PROVIDER_MAX_ATTEMPTS
        )
        for attempt in range(1, max_attempts + 1):
            try:
                return await super().get_response(*args, **kwargs)
            except Exception as exc:
                transient = _is_agents_sdk_transient_model_error(exc)
                self.context.provider_errors.append(
                    {
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                        "transient": transient,
                    }
                )
                if not transient or attempt >= max_attempts:
                    raise
                await asyncio.sleep(_provider_retry_delay(exc, attempt))
        raise RuntimeError("provider_attempts_exhausted")

    async def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        async for event in super().stream_response(*args, **kwargs):
            yield event


def _build_live_model(
    *,
    base_url: str,
    api_key: str,
    runtime_context: AgentRuntimeContext,
    timeout_sec: int,
) -> Any:
    import httpx
    from agents import AsyncOpenAI, OpenAIChatCompletionsModel, OpenAIResponsesModel

    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        timeout=timeout_sec,
    )
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_sec,
        max_retries=0,
        http_client=http_client,
    )
    api_surface = (
        (
            os.environ.get("AGENTIC_AGENTS_SDK_MODEL_API")
            or os.environ.get("AGENTIC_OPENAI_MODEL_API")
            or "chat"
        )
        .strip()
        .lower()
    )
    if api_surface in {"chat", "chat_completions", "chat-completions"}:
        inner = OpenAIChatCompletionsModel(
            model=configured_model(), openai_client=client
        )
        promote_tool_output_images = True
    else:
        inner = OpenAIResponsesModel(
            model=configured_model(),
            openai_client=_ResponsesSSECompatClient(client),
            model_is_explicit=True,
        )
        promote_tool_output_images = False
    return _RetryLoggingModel(
        inner,
        client=client,
        context=runtime_context,
        timeout_sec=timeout_sec,
        promote_tool_output_images=promote_tool_output_images,
        request_gate=get_vlm_concurrency_gate(),
    )


def initial_multimodal_input(
    prompt: str, paths: list[Path], roles: list[str], *, interleaved: bool = False
) -> list[dict[str, Any]]:
    from runtime.media_context import observed_attachment_label

    if len(paths) != len(roles):
        raise ValueError("observed_image_paths_and_roles_must_have_equal_length")
    markers = [f"[[observed_{index:02d}]]" for index in range(len(paths))]
    if interleaved and any(prompt.count(marker) != 1 for marker in markers):
        raise ValueError("interleaved_observations_require_each_image_exactly_once")
    content: list[dict[str, Any]] = (
        [] if interleaved else [{"type": "input_text", "text": prompt}]
    )
    cursor = 0
    for index, (path, role) in enumerate(zip(paths, roles)):
        if interleaved:
            position = prompt.index(markers[index])
            if position < cursor:
                raise ValueError("interleaved_observation_order_mismatch")
            content.append({"type": "input_text", "text": prompt[cursor:position]})
            cursor = position + len(markers[index])
        data_url = image_data_url(path)
        if not data_url:
            raise ValueError(f"unreadable_observed_image:{path}")
        content.append(
            {"type": "input_text", "text": observed_attachment_label(index, role)}
        )
        content.append({"type": "input_image", "image_url": data_url, "detail": "high"})
    if interleaved:
        content.append({"type": "input_text", "text": prompt[cursor:]})
    return [{"role": "user", "content": content}]


def run_agent(
    *,
    runtime_context: AgentRuntimeContext,
    task_prompt: str,
    output_type: type[BaseModel],
    tools: list[Any],
    base_url: str,
    api_key: str,
    max_turns: int,
    max_output_tokens: int,
    timeout_sec: int,
    model_override: Any | None = None,
    system_instructions: Any = CORE_SYSTEM_PROMPT,
    agent_name: str = "coworld_reasoner",
    initial_input: list[dict[str, Any]] | None = None,
    continuation_path: Path | None = None,
    tool_use_behavior: Any = "run_llm_again",
    checkpoint_dir: Path | None = None,
) -> AgentRunResult:
    """Run one autonomous SDK loop and return its complete local trace."""

    from agents import Agent, ModelSettings, Runner, set_tracing_disabled
    from agents.model_settings import Reasoning

    set_tracing_disabled(True)
    model = model_override or _build_live_model(
        base_url=base_url,
        api_key=api_key,
        runtime_context=runtime_context,
        timeout_sec=timeout_sec,
    )
    if checkpoint_dir is not None:
        from runtime.durable_agent import ReplayModel, journal_tools

        model = ReplayModel(model, checkpoint_dir, runtime_context)
        tools = journal_tools(tools, checkpoint_dir)
    agent_kwargs: dict[str, Any] = {
        "name": agent_name,
        "instructions": system_instructions,
        "model": model,
        "model_settings": ModelSettings(
            temperature=TEMPERATURE,
            max_tokens=max_output_tokens,
            reasoning=Reasoning(effort=REASONING_EFFORT),
            parallel_tool_calls=False,
            include_usage=True,
        ),
        "tools": tools,
        "output_type": output_type,
        "tool_use_behavior": tool_use_behavior,
        "reset_tool_choice": True,
    }
    agent = Agent[AgentRuntimeContext](**agent_kwargs)
    final_output: dict[str, Any] | None = None
    raw_final: str | None = None
    error: str | None = None
    try:
        current_input = (
            initial_input
            if initial_input is not None
            else initial_multimodal_input(
                task_prompt,
                runtime_context.observed_image_paths,
                runtime_context.observed_image_roles,
                interleaved=runtime_context.interleaved_observations,
            )
        )
        runtime_context.emit(
            "sdk_runner_call", {"max_turns": max_turns}
        )
        result = Runner.run_sync(
            agent,
            input=current_input,
            context=runtime_context,
            hooks=_TraceHooks.build(runtime_context),
            max_turns=max_turns,
        )
        output = result.final_output
        final_output = _jsonable(output)
        raw_final = json.dumps(final_output, ensure_ascii=False, sort_keys=True)
        if continuation_path is not None:
            # Preserve the SDK's actual ordered inputs/tool results for cumulative
            # stage continuation.
            from runtime.durable_agent import atomic_json

            atomic_json(continuation_path, _jsonable(result.to_input_list()))
    except Exception as exc:  # Record the failure for stage execution.
        error = f"{type(exc).__name__}:{str(exc)[:2000]}"
        runtime_context.emit("error", {"error": error})
    finally:
        if model_override is None:
            try:
                asyncio.run(model.close())
            except (
                Exception
            ) as exc:  # model response is already durable; record close failure
                runtime_context.emit(
                    "error",
                    {"error": f"model_close:{type(exc).__name__}:{str(exc)[:1000]}"},
                )
    model_events = [
        event for event in runtime_context.events if event["event_type"] == "model_turn"
    ]
    usage = {
        "input_tokens": sum(
            int((event["payload"].get("usage") or {}).get("input_tokens") or 0)
            for event in model_events
        ),
        "output_tokens": sum(
            int((event["payload"].get("usage") or {}).get("output_tokens") or 0)
            for event in model_events
        ),
        "reasoning_tokens": sum(
            int(
                (
                    (event["payload"].get("usage") or {}).get("output_tokens_details")
                    or {}
                ).get("reasoning_tokens")
                or 0
            )
            for event in model_events
        ),
    }
    return AgentRunResult(
        final_output=final_output,
        raw_final_output=raw_final,
        events=list(runtime_context.events),
        usage=usage,
        provider_errors=list(runtime_context.provider_errors),
        error=error,
    )
