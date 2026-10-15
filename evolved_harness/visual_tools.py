"""Progressively disclosed, backend-neutral VWM-driven tools for the task agent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from runtime.agent_runtime import _strict_args
from runtime.visual_dispatch import (
    OPERATION_CONTROL_FIELDS,
    VisualDispatchContext,
    _answer_free_violation,
    _backend_preflight,
    _dispatch_id,
    _validate_operation_controls,
)
from evolved_harness.catalog import (
    ACTIVE_SKILL_IDS,
    ACTIVE_VWM_TOOL_IDS,
    SEQUENCE_CONTINUATION_TOOL,
    INTERVENTION_TOOL,
    MOTION_DECOMPOSITION_TOOL,
    VIEW_CHANGE_TOOL,
    TEMPORAL_INTERPOLATION_TOOL,
    VISIBILITY_REVEAL_TOOL,
    skill_catalog,
    skill_package,
)


TOOL_OPERATION = {
    TEMPORAL_INTERPOLATION_TOOL: "temporal_interpolation",
    SEQUENCE_CONTINUATION_TOOL: "sequence_continuation",
    VIEW_CHANGE_TOOL: "view_change",
    VISIBILITY_REVEAL_TOOL: "view_change",
    MOTION_DECOMPOSITION_TOOL: "view_change",
    INTERVENTION_TOOL: "intervention",
}

SOURCE_ORDER_MODES = (
    "single_anchor",
    "ordered_transition",
    "reference_to_anchor",
    "independent_candidates",
)

# Task-facing agents choose the operation and describe how to realize it, while
# this wording keeps the operation description focused on visual evidence.
OPERATION_INFORMATION_GAPS = {
    "temporal_interpolation": (
        "The task-relevant visual transition is not fully observed in the supplied images."
    ),
    "view_change": (
        "The task-relevant viewpoint relation is not fully observed in the supplied images."
    ),
    "intervention": (
        "The task-relevant intervention outcome is not observed in the supplied image."
    ),
}
DEFAULT_INFORMATION_GAP = "The task-relevant visual evidence is not fully observed in the supplied observations."
DEFAULT_EXPECTED_INFORMATION_GAIN = (
    "Inspect the generated visual evidence for task-relevant information."
)
DEFAULT_EXPECTED_EVIDENCE = "Visible continuity, geometry, motion, or outcome grounded in the supplied observations."


def operation_information_gap(operation: str) -> str:
    return OPERATION_INFORMATION_GAPS.get(
        operation, DEFAULT_INFORMATION_GAP
    )


@dataclass
class HarnessToolContext(VisualDispatchContext):
    loaded_skill_id: str | None = None
    skill_loads: int = 0
    max_skill_loads: int = 1
    vwm_attempts: int = 0
    max_vwm_attempts: int = 1


def render_initial_skill_catalog() -> str:
    lines = ["Available task skills:"]
    for package in skill_catalog():
        lines.append(f"- {package.skill_id}: {package.description}")
    lines.extend(
        [
            "First decide whether the answer-bearing relation is resolved by the original",
            "observations. Load a skill only when the relation is unresolved and the skill",
            "matches the missing state. Call a VWM-driven tool only when its visual operation could add discriminative",
            "evidence; answer directly when the originals are sufficient.",
        ]
    )
    return "\n".join(lines)


def _operation_for_tool(tool_name: str) -> str:
    try:
        return TOOL_OPERATION[tool_name]
    except KeyError as exc:
        raise ValueError(f"unsupported_vwm_harness_tool:{tool_name}") from exc


def _backend_for_operation(operation: str) -> str:
    configured = os.environ.get("AGENTIC_VWM_TOOL_BACKEND", "").strip().lower()
    if configured in {"", "fal_h3"} and operation in {
        "temporal_interpolation",
        "view_change",
        "sequence_continuation",
        "intervention",
    }:
        return "fal_h3"
    raise ValueError(f"unsupported_vwm_harness_backend:{configured or '<unset>'}")


def _tool_enabled():
    def enabled(wrapper: Any, _agent: Any) -> bool:
        context = wrapper.context
        if not isinstance(context, VisualDispatchContext):
            return False
        if context.vwm_calls >= context.max_vwm_calls or getattr(
            context, "vwm_attempts", 0
        ) >= getattr(context, "max_vwm_attempts", context.max_vwm_calls):
            return False
        if getattr(context, "single_retake_policy", False) and context.vwm_calls > 0:
            return False
        return True

    return enabled


def selected_skill_id(context: Any) -> str | None:
    """Return the task skill actually loaded in this trajectory."""

    value = str(getattr(context, "loaded_skill_id", "") or "").strip()
    return value or None


def _empty_operation_controls() -> dict[str, Any]:
    return {key: None for key in OPERATION_CONTROL_FIELDS}


def _operation_controls(
    operation: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    controls = _empty_operation_controls()
    if operation == "intervention":
        target = args.get("target_object")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target_object_must_be_nonempty_string")
        controls["target_object"] = target.strip()
        return controls
    if operation == "view_change":
        controls["camera_instruction"] = (
            args.get("camera_instruction")
            or args.get("reveal_direction")
            or (
                "Use a controlled viewpoint probe while keeping fixed scene landmarks and "
                "the subject stationary."
                if args.get("motion_focus")
                else None
            )
            or args.get("generation_instruction")
        )
    return _validate_operation_controls(operation, controls)


def _common_request_properties(
    *, operation: str, candidate_conditioned_fields: bool = False,
    source_order_fields: bool = False,
) -> dict[str, Any]:
    min_sources, max_sources, source_description = {
        "temporal_interpolation": (2, 2, "Supply exactly two ordered start and end images."),
        "sequence_continuation": (1, 1, "Supply exactly one observed state to extend."),
        "intervention": (1, 1, "Supply exactly one image of the initial scene."),
        "view_change": (1, 2, "Supply one starting view, or two views defining the start and end."),
    }[operation]
    properties = {
        "source_image_indices": {
            "type": "array",
            "minItems": min_sources,
            "maxItems": max_sources,
            "items": {"type": "integer", "minimum": 0},
            "description": (
                "Ordered zero-based original image indices that ground the request. "
                + source_description
                + (
                    " When non-null candidate indices are supplied, this must be exactly "
                    "[reference_image_index, candidate_image_index]."
                    if candidate_conditioned_fields
                    else ""
                )
            ),
        },
        "generation_instruction": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Write a short, concrete English instruction for this input: describe the "
                "visual path or action, what moves, and what should remain stable. Keep it "
                "answer-neutral and grounded in the task and images. Do not mention an answer, "
                "option, verifier, or tool. Example: 'Generate one continuous camera "
                "path from the viewpoint shown in Picture 1 to the viewpoint shown in Picture 2; "
                "preserve the same subject, scene geometry, and fixed landmarks.'"
            ),
        },
        "preserve": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string", "minLength": 1},
            "description": "Observed identities, anchors, and relations that must remain stable.",
        },
    }
    if source_order_fields:
        properties.update(
            {
                "source_order_mode": {
                    "type": "string",
                    "enum": list(SOURCE_ORDER_MODES),
                    "description": (
                        "State the meaning of the ordered source_image_indices. Use single_anchor for one "
                        "image; ordered_transition only for a genuine continuous sequence; reference_to_anchor "
                        "when the question names a reference viewpoint and another image supplies a complementary "
                        "anchor; independent_candidates for one reference-to-one-candidate bridge."
                    ),
                },
                "reference_image_index": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "null"},
                    ],
                    "description": (
                        "Zero-based original image index that defines the question's viewpoint. Required for "
                        "reference_to_anchor and independent_candidates; otherwise use null."
                    ),
                },
            }
        )
    if candidate_conditioned_fields:
        properties.update(
            {
                "reference_image_index": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "null"},
                    ],
                    "description": (
                        "Zero-based original image index for the reference side of this one "
                        "candidate-conditioned relation; use null when this request is not "
                        "candidate-conditioned."
                    ),
                },
                "candidate_image_index": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "null"},
                    ],
                    "description": (
                        "Zero-based original image index for exactly one candidate in this "
                        "relation; use null when this request is not candidate-conditioned."
                    ),
                },
            }
        )
    return properties


def _operation_properties(
    operation: str,
) -> dict[str, Any]:
    if operation == "view_change":
        return {
            "camera_instruction": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Answer-neutral, grounded camera motion for a viewpoint operation. Use "
                    "only when the requested change is a camera/viewpoint path; do not encode "
                    "the answer or candidate relation. Distinguish in-place turning, camera "
                    "translation, camera orbit, and a static camera observing object motion. "
                    "Use local Picture 1/2 labels, not original question numbering."
                ),
            }
        }
    if operation == "intervention":
        return {
            "target_object": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "The acted-on object, identified by visible appearance and location "
                    "in Picture 1. H3 uses this text and generation_instruction, not "
                    "numeric boxes or tracked coordinates. Put the specified action "
                    "and its extent in generation_instruction; leave its unknown "
                    "consequence for the simulation."
                ),
            },
        }
    return {}


def _tool_specific_properties(tool_name: str) -> dict[str, Any]:
    """Expose controls that distinguish an agent-facing VWM-driven tool capability."""

    if tool_name == SEQUENCE_CONTINUATION_TOOL:
        return {
            "continuation_direction": {
                "type": "string",
                "enum": ["forward", "backward"],
                "description": (
                    "Direction of the missing short process continuation from the single supplied "
                    "state; this is not a bridge between two supplied endpoints."
                ),
            }
        }
    if tool_name == VISIBILITY_REVEAL_TOOL:
        return {
            "focus_target": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Concrete target or region whose visibility is unresolved because it is "
                    "occluded, cropped, or hidden by the current viewpoint."
                ),
            },
            "reveal_direction": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "A grounded viewpoint change that could expose the target or relation while "
                    "preserving the observed scene and identity."
                ),
            },
        }
    if tool_name == MOTION_DECOMPOSITION_TOOL:
        return {
            "motion_focus": {
                "type": "string",
                "enum": ["camera_motion", "object_motion", "camera_vs_object"],
                "description": (
                    "The unresolved motion attribution: inspect camera_motion, object_motion, or "
                    "camera_vs_object while keeping fixed scene anchors visible."
                ),
            }
        }
    return {}


def build_load_skill_tool(
    *,
    timeout_seconds: float = 30.0,
    vwm_tools_callable: bool = True,
    compact_output: bool = False,
) -> Any:
    from agents import FunctionTool

    async def invoke(wrapper: Any, raw_args: str) -> str:
        context = wrapper.context
        call_id = str(getattr(wrapper, "tool_call_id", ""))
        if not isinstance(context, VisualDispatchContext) or not hasattr(
            context, "loaded_skill_id"
        ):
            raise TypeError("vwm_harness_context_required")
        try:
            args = _strict_args(
                raw_args,
                required={"skill_id"},
                allowed={"skill_id"},
            )
            package = skill_package(str(args["skill_id"]))
            skill_body = package.load_body()
            if context.loaded_skill_id not in {None, package.skill_id}:
                raise ValueError(
                    f"one_skill_limit_already_loaded:{context.loaded_skill_id}"
                )
            if context.loaded_skill_id is None:
                if context.skill_loads >= context.max_skill_loads:
                    raise ValueError("vwm_task_skill_load_budget_exhausted")
                context.loaded_skill_id = package.skill_id
                context.skill_loads += 1
            currently_callable = (
                [
                    tool_name
                    for tool_name in package.allowed_tools
                    if tool_name in ACTIVE_VWM_TOOL_IDS
                ]
                if vwm_tools_callable
                else []
            )
            runtime_payload = {
                "tool_name": "load_vwm_task_skill",
                "call_id": call_id,
                "status": "loaded",
                "skill_id": package.skill_id,
                "matched_tool_interfaces": list(package.allowed_tools),
                "available_tool_interfaces": currently_callable,
                "vwm_tools_callable": vwm_tools_callable,
                "skill_body_delivery": "tool_output",
                "skill_body": skill_body,
            }
            context.emit(
                "skill_selection",
                {
                    "call_id": call_id,
                    "selected_skill": package.skill_id,
                    "matched_tool_interfaces": list(package.allowed_tools),
                    "available_tool_interfaces": runtime_payload[
                        "available_tool_interfaces"
                    ],
                    "vwm_tools_callable": vwm_tools_callable,
                },
            )
            payload = (
                {
                    "status": "loaded",
                    "skill_id": package.skill_id,
                    "available_tools": currently_callable,
                    "skill": skill_body,
                }
                if compact_output
                else runtime_payload
            )
        except (ValueError, json.JSONDecodeError) as exc:
            payload = {
                "tool_name": "load_vwm_task_skill",
                "call_id": call_id,
                "status": "invalid_request",
                "error": str(exc),
            }
        context.emit(
            "tool_result",
            {
                "tool_name": "load_vwm_task_skill",
                "call_id": call_id,
                **payload,
            },
        )
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return FunctionTool(
        name="load_vwm_task_skill",
        description=(
            (
                "Load one matching task skill when the original observations leave an "
                "answer-relevant relation unresolved and the skill matches that missing relation. "
                "Do not load a skill when the originals are sufficient or the relation is outside "
                "the available VWM-driven tool capabilities. Loading a skill does not require a VWM-driven tool call."
            )
            if compact_output
            else (
                "Load at most one task-shape VWM-driven skill when a useful unobserved visual-world state "
                "remains. Do not load a skill when original observations are sufficient. "
                + (
                    "Matched VWM-driven tools become callable after loading."
                    if vwm_tools_callable
                    else "This P1 boundary records skill discovery only; no VWM-driven tool is callable."
                )
            )
        ),
        params_json_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["skill_id"],
            "properties": {
                "skill_id": {"type": "string", "enum": list(ACTIVE_SKILL_IDS)}
            },
        },
        on_invoke_tool=invoke,
        strict_json_schema=True,
        is_enabled=lambda wrapper, _agent: (
            isinstance(wrapper.context, VisualDispatchContext)
            and hasattr(wrapper.context, "loaded_skill_id")
            and wrapper.context.loaded_skill_id is None
            and getattr(wrapper.context, "skill_loads", 0)
            < getattr(wrapper.context, "max_skill_loads", 0)
        ),
        timeout_seconds=timeout_seconds,
        timeout_behavior="error_as_result",
    )


def build_visual_operation_tool(
    tool_name: str,
    *,
    timeout_seconds: float = 30.0,
    candidate_conditioned_fields: bool = False,
) -> Any:
    from agents import FunctionTool

    operation = _operation_for_tool(tool_name)
    candidate_fields_enabled = candidate_conditioned_fields and operation == "view_change"
    source_order_fields = operation in {"temporal_interpolation", "view_change"}
    properties = {
        **(
            _common_request_properties(
                operation=operation,
                candidate_conditioned_fields=candidate_fields_enabled,
                source_order_fields=source_order_fields,
            )
        ),
        **_operation_properties(operation),
        **_tool_specific_properties(tool_name),
    }
    source_schema = properties["source_image_indices"]
    min_sources, max_sources = source_schema["minItems"], source_schema["maxItems"]
    optional = {"candidate_image_index"} if candidate_fields_enabled else set()
    # Targeted view-change tools use operation-specific controls; generic camera
    # and reference-order fields are optional.
    if tool_name in {VISIBILITY_REVEAL_TOOL, MOTION_DECOMPOSITION_TOOL}:
        optional.update(
            {"camera_instruction", "source_order_mode", "reference_image_index"}
        )
    required = set(properties) - optional

    async def invoke(wrapper: Any, raw_args: str) -> Any:
        context = wrapper.context
        call_id = str(getattr(wrapper, "tool_call_id", ""))
        if not isinstance(context, VisualDispatchContext) or not hasattr(
            context, "loaded_skill_id"
        ):
            raise TypeError("vwm_harness_context_required")
        try:
            args = _strict_args(
                raw_args,
                required=required,
                allowed=set(properties),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            payload = {
                "tool_name": tool_name,
                "call_id": call_id,
                "status": "invalid_arguments",
                "error": str(exc),
            }
            context.emit("tool_result", payload)
            return json.dumps(payload, sort_keys=True)

        context.emit(
            "tool_call",
            {"tool_name": tool_name, "call_id": call_id, "arguments": args},
        )
        if getattr(context, "single_retake_policy", False):
            policy_error = (
                "additional_generation_requires_single_defect_retake"
                if context.vwm_calls > 0
                else None
            )
            if policy_error:
                payload = {
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "status": "budget_exhausted",
                    "error": policy_error,
                }
                context.emit("tool_result", payload)
                return json.dumps(payload, sort_keys=True)
        status = "dispatch_accepted"
        error: str | None = None
        indices = args["source_image_indices"]
        preserve = args["preserve"]
        source_order_mode = (
            args.get("source_order_mode") if source_order_fields else None
        )
        reference_image_index = (
            args.get("reference_image_index") if source_order_fields else None
        )
        candidate_image_index = (
            args.get("candidate_image_index") if candidate_fields_enabled else None
        )
        # `reference_image_index` is also required by ordinary
        # `reference_to_anchor` requests. Only a non-null candidate index
        # activates candidate-conditioned validation.
        candidate_fields_supplied = (
            candidate_fields_enabled and args.get("candidate_image_index") is not None
        )
        candidate_conditioned_request = (
            candidate_fields_supplied
            and args.get("reference_image_index") is not None
            and args.get("candidate_image_index") is not None
        )
        generation_instruction = str(args["generation_instruction"]).strip()
        evidence_focus = str(args.get("evidence_focus") or "").strip()
        if tool_name == SEQUENCE_CONTINUATION_TOOL:
            generation_instruction = (
                f"{generation_instruction} Extend the observed state "
                f"{args['continuation_direction']} in time."
            )
        elif tool_name == VISIBILITY_REVEAL_TOOL:
            generation_instruction = (
                f"{generation_instruction} Change the viewpoint toward {args['reveal_direction']} "
                f"to expose {args['focus_target']}; preserve all observed scene anchors and "
                "do not invent unrelated content."
            )
        elif tool_name == MOTION_DECOMPOSITION_TOOL:
            generation_instruction = (
                f"{generation_instruction} Produce a controlled visual probe for "
                f"{args['motion_focus']}: keep fixed scene landmarks stable and make the "
                "relative motion of the subject and camera inspectable."
            )
        controls: dict[str, Any] | None = None
        backend_preflight: dict[str, Any] | None = None
        if context.vwm_calls >= context.max_vwm_calls:
            status, error = "budget_exhausted", "maximum_vwm_dispatches_reached"
        elif context.vwm_attempts >= context.max_vwm_attempts:
            status, error = "budget_exhausted", "maximum_vwm_attempts_reached"
        elif not isinstance(indices, list) or not indices:
            status, error = (
                "invalid_dispatch",
                "source_image_indices_must_be_nonempty_array",
            )
        elif any(
            not isinstance(index, int) or isinstance(index, bool) for index in indices
        ):
            status, error = "invalid_dispatch", "source_image_indices_must_be_integers"
        elif len(set(indices)) != len(indices):
            status, error = "invalid_dispatch", "source_image_indices_must_be_unique"
        elif any(
            index < 0 or index >= len(context.observed_image_paths) for index in indices
        ):
            status, error = "invalid_dispatch", "source_image_index_out_of_range"
        elif not min_sources <= len(indices) <= max_sources:
            count = str(min_sources) if min_sources == max_sources else f"{min_sources}_or_{max_sources}"
            status, error = "invalid_dispatch", f"{operation}_requires_{count}_source_images"
        elif (
            source_order_fields
            and source_order_mode is not None
            and source_order_mode not in SOURCE_ORDER_MODES
        ):
            status, error = "invalid_dispatch", "source_order_mode_is_invalid"
        elif (
            source_order_fields
            and source_order_mode == "single_anchor"
            and len(indices) != 1
        ):
            status, error = (
                "invalid_dispatch",
                "single_anchor_requires_one_source_image",
            )
        elif (
            source_order_fields
            and source_order_mode == "ordered_transition"
            and len(indices) != 2
        ):
            status, error = (
                "invalid_dispatch",
                "ordered_transition_requires_two_source_images",
            )
        elif (
            source_order_fields
            and source_order_mode == "reference_to_anchor"
            and (
                not isinstance(reference_image_index, int)
                or isinstance(reference_image_index, bool)
                or len(indices) != 2
                or indices[0] != reference_image_index
            )
        ):
            status, error = (
                "invalid_dispatch",
                "reference_to_anchor_requires_reference_as_first_of_two_sources",
            )
        elif (
            source_order_fields
            and source_order_mode == "independent_candidates"
            and not candidate_fields_enabled
        ):
            status, error = (
                "invalid_dispatch",
                "independent_candidates_requires_candidate_fields",
            )
        elif (
            source_order_fields
            and source_order_mode == "independent_candidates"
            and (
                not candidate_conditioned_request
                or len(indices) != 2
                or indices != [reference_image_index, candidate_image_index]
            )
        ):
            status, error = (
                "invalid_dispatch",
                "independent_candidates_requires_one_reference_to_one_candidate_pair",
            )
        elif candidate_fields_supplied and not candidate_conditioned_request:
            status, error = (
                "invalid_dispatch",
                "candidate_conditioned_indices_must_be_provided_together",
            )
        elif candidate_conditioned_request and (
            not isinstance(reference_image_index, int)
            or isinstance(reference_image_index, bool)
            or not isinstance(candidate_image_index, int)
            or isinstance(candidate_image_index, bool)
        ):
            status, error = (
                "invalid_dispatch",
                "candidate_conditioned_indices_must_be_integers",
            )
        elif (
            candidate_conditioned_request
            and reference_image_index == candidate_image_index
        ):
            status, error = (
                "invalid_dispatch",
                "candidate_conditioned_reference_and_candidate_must_differ",
            )
        elif candidate_conditioned_request and (
            reference_image_index not in indices
            or candidate_image_index not in indices
            or len(indices) != 2
        ):
            status, error = (
                "invalid_dispatch",
                "candidate_conditioned_request_requires_reference_and_one_candidate",
            )
        elif candidate_conditioned_request and indices != [
            reference_image_index,
            candidate_image_index,
        ]:
            status, error = (
                "invalid_dispatch",
                "candidate_conditioned_source_order_must_be_reference_then_candidate",
            )
        elif not all(str(args[key]).strip() for key in (("generation_instruction",))):
            status, error = "invalid_dispatch", "dispatch_text_fields_must_be_nonempty"
        elif (
            not isinstance(preserve, list)
            or not preserve
            or any(not str(item).strip() for item in preserve)
        ):
            status, error = "invalid_dispatch", "preserve_must_be_nonempty_string_array"
        else:
            try:
                controls = _operation_controls(
                    operation,
                    args,
                )
                if operation == "intervention":
                    generation_instruction = (
                        f"Acted-on object in Picture 1: {controls['target_object']}. "
                        f"{generation_instruction}"
                    )
            except ValueError as exc:
                status, error = "invalid_dispatch", str(exc)

        information_gap = operation_information_gap(operation)
        text_fields = [
            information_gap,
            generation_instruction,
            *([str(item) for item in preserve] if isinstance(preserve, list) else []),
        ]
        if controls:
            text_fields.extend(
                str(controls[key])
                for key in ("camera_instruction", "target_object")
                if controls[key] is not None
            )
        if error is None:
            error = _answer_free_violation(text_fields)
            if error:
                status = "invalid_dispatch"

        backend = _backend_for_operation(operation)
        if status == "dispatch_accepted" and controls is not None:
            try:
                backend_preflight = _backend_preflight(
                    backend=backend,
                    operation=operation,
                    source_paths=[
                        context.observed_image_paths[index] for index in indices
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                status = "invalid_dispatch"
                error = f"backend_preflight_failed:{type(exc).__name__}:{exc}"

        context.vwm_attempts += 1
        if status == "dispatch_accepted":
            context.vwm_calls += 1
        request = {
            "dispatch_id": _dispatch_id(context, call_id),
            "backend": backend,
            "operation": operation,
            "tool_interface": tool_name,
            "source_image_indices": indices,
            "source_order_mode": source_order_mode,
            "candidate_conditioned": candidate_conditioned_request,
            "reference_image_index": reference_image_index,
            "candidate_image_index": candidate_image_index,
            "candidate_relation": (
                {
                    "reference_image_index": reference_image_index,
                    "candidate_image_index": candidate_image_index,
                }
                if candidate_conditioned_request
                else None
            ),
            "information_gap": information_gap,
            "expected_information_gain": (DEFAULT_EXPECTED_INFORMATION_GAIN),
            "expected_evidence": (DEFAULT_EXPECTED_EVIDENCE),
            "evidence_focus": evidence_focus or None,
            "generation_instruction": generation_instruction,
            "tool_arguments": dict(args),
            "preserve": preserve,
            "operation_controls": controls,
            "control_provenance": "agent_model_tool_arguments",
            "backend_preflight": backend_preflight,
            "answer_free_control": error is None,
            "requested_by_skill_id": selected_skill_id(context),
            "evidence_sufficiency": ("runtime_inferred_missing"),
            "visual_operation_opportunity": ("agent_selected_operation"),
        }
        context.emit(
            "planning",
            {
                "tool_name": tool_name,
                "call_id": call_id,
                "selected_skill": selected_skill_id(context),
                "information_gap": request["information_gap"],
            },
        )
        context.emit("vwm_call", {"call_id": call_id, **request})
        vwm_payload = {
            "tool_name": tool_name,
            "call_id": call_id,
            **request,
            "status": status,
            "error": error,
            "readable": False,
            "artifact_sha256": None,
            "stage_boundary": "dispatch_only_no_generation_or_media",
        }
        context.emit("vwm_result", vwm_payload)
        if status == "dispatch_accepted":
            context.accepted_dispatches.append(request)
            if context.vwm_dispatch_executor is not None:
                return context.vwm_dispatch_executor(context, request, call_id)
        payload = {
            **vwm_payload,
            "instruction": (
                "Invalid request acquired no evidence and did not consume the accepted-call "
                "budget. Rewrite all request fields without answer, option, choice, label, "
                "candidate letter, or correctness language, then retry once."
                if status == "invalid_dispatch"
                else "Dispatch receipt only; no generated artifact or verifier evidence exists yet."
            ),
        }
        context.emit("tool_result", payload)
        model_payload = {
            "status": status,
            "operation": operation,
            "error": error,
            "evidence_returned": False,
        }
        return json.dumps(model_payload, ensure_ascii=False, sort_keys=True)

    descriptions = {
        TEMPORAL_INTERPOLATION_TOOL: (
            "Generate one predicted interpolation between ordered observations when the "
            "answer-bearing transition is unresolved. The result is visual evidence, not ground "
            "truth or calibrated geometry."
        ),
        SEQUENCE_CONTINUATION_TOOL: (
            "Extend one observed state forward or backward to expose an unresolved short temporal "
            "outcome. Use for continuation, not for connecting two supplied endpoints. The result "
            "is predicted visual evidence."
        ),
        VIEW_CHANGE_TOOL: (
            "Generate one predicted grounded view when an answer-bearing viewpoint or cross-view "
            "mapping is unresolved. The result is visual evidence, not calibrated camera control "
            "or ground truth."
        ),
        VISIBILITY_REVEAL_TOOL: (
            "Generate a targeted viewpoint change to expose a target or relation hidden by "
            "occlusion or the current view. This is a visibility probe, not a generic scene change."
        ),
        MOTION_DECOMPOSITION_TOOL: (
            "Generate a controlled visual probe for an unresolved attribution between camera and "
            "object motion. Keep scene landmarks and subject identity stable."
        ),
        INTERVENTION_TOOL: (
            "Generate one predicted outcome for a bounded intervention when that outcome is "
            "unobserved. Use one target and one action. The result is visual evidence, not ground truth."
        ),
    }
    if operation == "intervention":
        descriptions[INTERVENTION_TOOL] = (
            "Simulate a task-specified object action from one original image using "
            "text-conditioned H3. Identify the acting object and describe its visible "
            "action and stopping condition. The camera and unrelated scene normally "
            "remain fixed. This tool has no numeric bounding-box or trajectory control; "
            "the target description and generation_instruction are the actual controls. "
            "Inspect the resulting motion and consequence, then request its generated visual evidence review."
        )
    return FunctionTool(
        name=tool_name,
        description=descriptions[tool_name],
        params_json_schema={
            "type": "object",
            "additionalProperties": False,
            "required": sorted(required),
            "properties": properties,
        },
        on_invoke_tool=invoke,
        strict_json_schema=True,
        is_enabled=_tool_enabled(),
        timeout_seconds=timeout_seconds,
        timeout_behavior="error_as_result",
    )


def build_vwm_tools(
    *,
    timeout_seconds: float = 30.0,
    candidate_conditioned_fields: bool = False,
) -> list[Any]:
    """Build the task-facing VWM-driven skill and tool surface."""

    tools: list[Any] = []
    tools.extend(
        build_visual_operation_tool(
            tool_name,
            timeout_seconds=timeout_seconds,
            candidate_conditioned_fields=candidate_conditioned_fields,
        )
        for tool_name in ACTIVE_VWM_TOOL_IDS
    )
    return tools
