"""Task-shape skill catalog and capability boundaries for the VWM harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SKILL_ROOT = PACKAGE_ROOT / "task_skills"

TEMPORAL_SKILL = "reason-temporal-spatial-continuity"
CROSS_VIEW_SKILL = "complete-cross-view-spatial"
CAUSAL_SKILL = "simulate-causal-intervention"

TEMPORAL_INTERPOLATION_TOOL = "vwm_temporal_interpolation_tool"
SEQUENCE_CONTINUATION_TOOL = "vwm_temporal_continuation_tool"
VIEW_CHANGE_TOOL = "vwm_new_view_tool"
VISIBILITY_REVEAL_TOOL = "vwm_visibility_reveal_tool"
MOTION_DECOMPOSITION_TOOL = "vwm_motion_decomposition_tool"
INTERVENTION_TOOL = "vwm_simulate_intervention_tool"
RETAKE_TOOL = "vwm_retake_tool"

ACTIVE_SKILL_IDS = (TEMPORAL_SKILL, CROSS_VIEW_SKILL, CAUSAL_SKILL)
ACTIVE_VWM_TOOL_IDS = (
    TEMPORAL_INTERPOLATION_TOOL,
    VIEW_CHANGE_TOOL,
    INTERVENTION_TOOL,
    SEQUENCE_CONTINUATION_TOOL,
    VISIBILITY_REVEAL_TOOL,
    MOTION_DECOMPOSITION_TOOL,
)

def render_capability_boundary_assessment() -> str:
    """Return the shared relation-level self-capability check used before VWM-driven tool discovery."""

    return """P1 capability check:

1. Identify the answer-bearing visual relation and the fact that would change the answer.
2. Ask whether the supplied pixels jointly determine that relation. If different hidden states
   could produce the same pixels but require different answers, the relation is missing.
3. Describe the missing state visually: an ordered transition, connecting pose, reference-frame or
   handedness mapping, or bounded intervention outcome.
4. Match the missing state to an active skill. Load it and use its operation when the result could
   distinguish the live interpretations.
5. Continue directly when the relation is jointly observed or no available operation can test it.

Resolve the common frame ambiguities before labeling a gap:
- If the question asks where an object is relative to the camera/person in a named photo, judge
  that photo's requested reference frame first. The presence of another attached photo does not by
  itself make the relation cross-view. If the target and the needed egocentric anchors are visible
  in the named photo, this can be directly observed.
- Labeled candidate images are alternatives, not a time sequence. A front/reference view plus a
  candidate whose front/rear, visible flank, and stable identity make the requested canonical side
  unambiguous can be solved directly. Load cross-view only when the physical identity, connecting
  pose, handedness, or reference mapping remains genuinely ambiguous and a grounded view could
  distinguish it.
- If one image is the current camera view and another unordered image shows the target view,
  destination, or desired orientation, a question about how to move from the current view toward
  that target is cross-view completion, not temporal interpolation. The target may be visible in
  the other image; it is not an observed future frame. Use temporal continuity only when the task
  explicitly supplies an ordered start/end sequence (or other observed time order) and the missing
  interval itself is answer-bearing.
- A qualitative comparison across separate views may still be directly reasoned when each target
  has sufficient local scale anchors and the question does not require exact metric recovery. Do
  not load merely because the objects are in different images; do load when the comparison needs a
  missing grounded common view and the artifact would add discriminative qualitative evidence.
- Temporal continuity requires ordered observed anchors and an answer-bearing unobserved interval
  or process. A hypothetical movement, fall, collision, removal, or occlusion change starting from
  one static scene is causal intervention, even though its outcome unfolds over time. Conversely,
  an ordered pair is not automatically temporal when the question only asks a static relation in
  one named frame.

Output consistency check before finalizing P1:
- directly_jointly_observed -> missing_state=null, visual_operation_opportunity=none,
  skip_reason=directly_jointly_observed;
- not_directly_jointly_observed or uncertain + a useful active operation -> non-empty missing_state,
  clear/plausible opportunity, one matching selected skill, skip_reason=null;
- not_directly_jointly_observed or uncertain + no useful active operation -> non-empty missing_state,
  visual_operation_opportunity=none and one precise legal skip reason.
In each case, retain selected_task_skill if a skill was loaded to understand the
capability boundary; use null only when no skill was loaded. Skill discovery does
not commit to generation, and a no-operation judgment does not erase a skill load.

High-value boundary distinctions:
- Separate images that show local anchors do not jointly show the global relation if a connecting
  camera pose, frame transform, or handedness-preserving view is still needed.
- Image-plane displacement is evidence about a possible camera transform, not the transform itself;
  ordered endpoints can therefore fit temporal continuity when the missing interval matters.
- A front/profile cue does not by itself establish canonical left/right identity across views;
  use cross-view completion when a grounded connecting view could distinguish the mapping.
- A visible actor, path, and obstacle do not show the result of a future movement, collision,
  removal, or occlusion change; use causal intervention when one bounded target action can expose it.
- Static count, text, source detail, exact metric/scale, and a relation already visible in one
  required reference frame remain outside the active VWM-driven tool capabilities.

Do not use a fixed call quota, benchmark name, remembered sample, answer label, or historical
accuracy to make this decision. The protocol decides whether the model's current visual evidence
is sufficient; it does not decide the answer or force a VWM-driven tool call."""


@dataclass(frozen=True)
class SkillPackage:
    skill_id: str
    description: str
    allowed_tools: tuple[str, ...]

    @property
    def body_path(self) -> Path:
        return SKILL_ROOT / self.skill_id / "SKILL.md"

    def load_body(self) -> str:
        return self.body_path.read_text(encoding="utf-8")


_CATALOG = (
    SkillPackage(
        skill_id=TEMPORAL_SKILL,
        description=(
            "For an answer-bearing transition or motion interval between ordered observations "
            "when the missing intermediate state matters; choose it before answering from endpoints "
            "alone. It also covers a missing short continuation from one observed state."
        ),
        allowed_tools=(TEMPORAL_INTERPOLATION_TOOL, SEQUENCE_CONTINUATION_TOOL, RETAKE_TOOL),
    ),
    SkillPackage(
        skill_id=CROSS_VIEW_SKILL,
        description=(
            "For an answer-bearing relation between separate viewpoints that needs a connecting view, "
            "reference-frame transform, canonical-side mapping, or global layout relation; choose it "
            "when separate images do not jointly establish that relation. It also covers revealing an "
            "occluded target from a grounded viewpoint."
        ),
        allowed_tools=(
            TEMPORAL_INTERPOLATION_TOOL,
            VIEW_CHANGE_TOOL,
            VISIBILITY_REVEAL_TOOL,
            MOTION_DECOMPOSITION_TOOL,
            RETAKE_TOOL,
        ),
    ),
    SkillPackage(
        skill_id=CAUSAL_SKILL,
        description=(
            "For an answer-bearing outcome of a bounded physical intervention or counterfactual "
            "action that is not already observed; choose it before answering from a static scene."
        ),
        allowed_tools=(INTERVENTION_TOOL, RETAKE_TOOL),
    ),
)


def skill_catalog() -> tuple[SkillPackage, ...]:
    return _CATALOG


def skill_package(skill_id: str) -> SkillPackage:
    for package in skill_catalog():
        if package.skill_id == skill_id:
            return package
    raise ValueError(f"unknown_vwm_harness_skill:{skill_id}")
