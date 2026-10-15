"""Validate and dispatch model-selected VWM-driven tool requests.

Validate source observations and answer-free controls before dispatching through
the runtime executor. Requests and execution results are recorded in the trajectory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from runtime.agent_runtime import AgentRuntimeContext


BACKEND_OPERATIONS: dict[str, frozenset[str]] = {
    "fal_h3": frozenset(
        {"temporal_interpolation", "view_change", "sequence_continuation", "intervention"}
    ),
}
OPERATION_CONTROL_FIELDS = frozenset(
    {
        "camera_instruction",
        "target_object",
    }
)


_ANSWER_DIRECTING_PATTERNS = (
    re.compile(
        r"\b(correct|gold|ground[ -]?truth)\s+(answer|option|choice|label|response)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(answer|option|choice|label|response)\s+(is\s+)?(correct|gold|ground[ -]?truth)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(answer|choose|select)\s+(is\s+)?(option\s+)?[A-H]\b", re.IGNORECASE
    ),
    re.compile(r"\b(option|choice)\s+[A-H]\b", re.IGNORECASE),
    re.compile(
        r"\bprove\s+(?:that\s+)?(?:the\s+)?(?:correct\s+)?"
        r"(?:answer|option|choice|label|response)(?:\s+[A-H])?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdesired\s+(answer|outcome|result)\b", re.IGNORECASE),
)


@dataclass
class VisualDispatchContext(AgentRuntimeContext):
    max_vwm_calls: int = 2
    vwm_calls: int = 0
    accepted_dispatches: list[dict[str, Any]] = field(default_factory=list)
    vwm_dispatch_executor: (
        Callable[["VisualDispatchContext", dict[str, Any], str], Any] | None
    ) = None


def _answer_free_violation(texts: list[str]) -> str | None:
    combined = "\n".join(texts)
    for pattern in _ANSWER_DIRECTING_PATTERNS:
        if pattern.search(combined):
            return f"answer_directing_control:{pattern.pattern}"
    return None


def _dispatch_id(context: VisualDispatchContext, call_id: str) -> str:
    return f"visual_dispatch_{len(context.accepted_dispatches) + 1:02d}_{call_id or 'unknown'}"


def _validate_operation_controls(operation: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("operation_controls_must_be_object")
    if set(value) != OPERATION_CONTROL_FIELDS:
        missing = sorted(OPERATION_CONTROL_FIELDS - set(value))
        extra = sorted(set(value) - OPERATION_CONTROL_FIELDS)
        raise ValueError(
            f"operation_controls_fields_mismatch:missing={missing}:extra={extra}"
        )
    controls = dict(value)
    for key in OPERATION_CONTROL_FIELDS:
        item = controls[key]
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ValueError(f"{key}_must_be_nonempty_string_or_null")
    required_non_null: dict[str, set[str]] = {
        "view_change": {"camera_instruction"},
        "intervention": {"target_object"},
    }
    missing = sorted(
        key for key in required_non_null.get(operation, set()) if controls[key] is None
    )
    if missing:
        raise ValueError(f"missing_operation_controls:{','.join(missing)}")

    allowed_non_null: dict[str, set[str]] = {
        "temporal_interpolation": set(),
        "sequence_continuation": set(),
        "view_change": {"camera_instruction"},
        "intervention": required_non_null["intervention"],
    }
    unexpected = sorted(
        key
        for key, item in controls.items()
        if item is not None and key not in allowed_non_null.get(operation, set())
    )
    if unexpected:
        raise ValueError(f"operation_controls_not_applicable:{','.join(unexpected)}")

    return controls


def _tool_interface(operation: str) -> str:
    return {
        "temporal_interpolation": "vwm_temporal_interpolation_tool",
        "view_change": "vwm_new_view_tool",
        "sequence_continuation": "vwm_temporal_continuation_tool",
        "intervention": "vwm_simulate_intervention_tool",
    }[operation]


def _backend_preflight(
    *,
    backend: str,
    operation: str,
    source_paths: list[Any],
) -> dict[str, Any]:
    """Validate backend settings and source-image count before generation."""

    if backend == "fal_h3":
        from evolved_harness.fal_h3_backend_adapter import FalH3Config

        config = FalH3Config.from_env()
        if len(source_paths) not in {1, 2}:
            raise ValueError("fal_h3_requires_one_or_two_source_images")
        if config.duration < 5 or config.duration > 15:
            raise ValueError("fal_h3_duration_must_be_5_to_15")
        if config.resolution not in {"480P", "768P", "1080P"}:
            raise ValueError("fal_h3_invalid_resolution")
        return {
            "status": "pass",
            "validator": "evolved_harness.fal_h3_backend_adapter.FalH3Config",
            "backend": backend,
            "tool_interface": _tool_interface(operation),
            "source_image_count": len(source_paths),
            "pipeline": "fal_h3_max_turbo",
            "duration": config.duration,
            "resolution": config.resolution,
        }

    raise ValueError(f"unsupported_backend:{backend}")
