"""P1 output, guidance, and skill tools used by the shared stage runner."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evolved_harness.catalog import render_capability_boundary_assessment
from evolved_harness.visual_tools import (
    build_load_skill_tool,
    render_initial_skill_catalog,
)


TaskSkillId = Literal[
    "reason-temporal-spatial-continuity",
    "complete-cross-view-spatial",
    "simulate-causal-intervention",
]
RelationVisibility = Literal[
    "directly_jointly_observed",
    "not_directly_jointly_observed",
    "uncertain",
]
WMOpportunity = Literal["clear", "plausible", "none"]
LegalSkipReason = Literal[
    "directly_jointly_observed",
    "outside_active_visual_operation_capability",
    "non_discriminative_expected_artifact",
]


class CapabilityAssessment(BaseModel):
    """Per-sample P1 attribution plus an observation-only provisional answer."""

    model_config = ConfigDict(extra="forbid")

    answer_bearing_relation: str = Field(min_length=1)
    relation_visibility: RelationVisibility
    missing_state: str | None
    visual_operation_capability_fit: str = Field(min_length=1)
    expected_discriminative_value: str = Field(min_length=1)
    visual_operation_opportunity: WMOpportunity
    selected_task_skill: TaskSkillId | None = Field(description="Skill loaded for capability assessment, or null if none was loaded. May remain non-null when the final operation opportunity is none.")
    decision_reason: str = Field(min_length=1)
    skip_reason: LegalSkipReason | None
    reasoning_summary: str = Field(
        min_length=1,
        description="Concise task reasoning from original observations only.",
    )
    provisional_answer: str = Field(
        min_length=1,
        description="Compact answer in the exact task-requested format.",
    )

    @model_validator(mode="after")
    def validate_opportunity_boundary(self) -> "CapabilityAssessment":
        if self.relation_visibility == "directly_jointly_observed":
            if self.missing_state is not None:
                raise ValueError("jointly_observed_relation_cannot_have_missing_state")
            if self.visual_operation_opportunity != "none":
                raise ValueError("jointly_observed_relation_requires_none_opportunity")
            if self.skip_reason != "directly_jointly_observed":
                raise ValueError("jointly_observed_relation_requires_matching_skip_reason")
        else:
            if not str(self.missing_state or "").strip():
                raise ValueError("unobserved_or_uncertain_relation_requires_missing_state")
        if self.visual_operation_opportunity in {"clear", "plausible"}:
            if self.selected_task_skill is None:
                raise ValueError("positive_visual_operation_opportunity_requires_loaded_task_skill")
            if self.skip_reason is not None:
                raise ValueError("positive_visual_operation_opportunity_cannot_have_skip_reason")
        elif self.skip_reason is None:
            raise ValueError("none_visual_operation_opportunity_requires_legal_skip_reason")
        return self


CAPABILITY_ASSESSMENT_INSTRUCTIONS = f"""Solve the task from the complete original observations.

{render_initial_skill_catalog()}

P1: Capability Boundary Assessment. Identify the answer-bearing relation and decide whether it is
directly and jointly visible in the original observations. If not, name one answer-free missing
visual-world state. Separately judge whether a currently described VWM-driven tool capability could expose that
state at the required fidelity and with discriminative value. Classify the VWM-driven tool opportunity as
clear, plausible, or none. Load at most one matching task skill when its complete capability and
boundary guidance is needed.

{render_capability_boundary_assessment()}

Treat VWM-driven tools as evidence acquisition, not as a last resort after confidence is exhausted. Ask whether
available generated visual evidence could test, falsify, or distinguish the provisional inference. When the
answer-bearing relation depends on an unobserved state and a matching capability could expose
discriminative evidence, actively load the matching skill even if you can already make a likely
guess from the original observations.

Use these answer-free, cross-sample experience patterns only as attention priors, never as fixed
routes. Sparse endpoint motion can hide an intermediate egomotion transform; relations split
across views can hide a connecting pose; canonical side identity from sparse profiles can remain
view-dependent; and an outcome or compatibility judgment from a static initial state can depend on
an unobserved intervention. In those shapes, explicitly test whether a matching temporal,
cross-view, or intervention skill could expose discriminative state. Conversely, static source
detail has no active zoom or crop capability, and metric or scale recovery from sparse views is
often outside the active VWM-driven tool boundary. Always decide from the current observations and actual
capability fit rather than a task name, sample identity, remembered answer, or historical result.

Apply a strict observation-sufficiency test to the answer-bearing relation itself. Visible cues are
not the same as direct observation of a relation: image-plane shift is only a cue for a camera
transform, and front/profile appearance is only a cue for canonical left/right handedness. If the
answer still requires constructing a missing transform, connecting pose, or handedness-preserving
view, treat that state as unobserved and assess the matching skill. By contrast, an explicitly
visible rear surface identified from direct rear features is already observed. Absolute metric
recovery remains unsupported, but a qualitative comparison between objects never jointly visible
may still fit a grounded cross-view bridge. These distinctions refine capability discovery; they
do not require a call when the current relation itself is genuinely jointly visible.

Legal none-opportunity reasons are only: the relation is directly and jointly observed; the needed
change is outside active VWM-driven tool capability; or the expected artifact would be non-discriminative.
Confidence in a guess, likely answer correctness, and generation cost are not sufficient skip
reasons. No VWM-driven tool, generation, artifact, retake, verifier, or post-generation evidence exists at
this boundary. Give a provisional answer from original observations only.

Task:
{{question}}
"""


def build_capability_assessment_tools(*, timeout_seconds: float = 30.0) -> list[Any]:
    return [
        build_load_skill_tool(
            timeout_seconds=timeout_seconds,
            vwm_tools_callable=False,
        )
    ]
