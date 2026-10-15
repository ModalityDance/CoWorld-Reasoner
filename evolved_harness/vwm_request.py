"""Request-stage output schema and tool construction."""

from __future__ import annotations

from pydantic import BaseModel


class VWMRequestDecision(BaseModel):
    decision_reason: str


def build_vwm_assistance_tools():
    """Expose executable requests independently of the retained P1 skill choice."""
    from evolved_harness import visual_tools
    from evolved_harness.evidence_tools import build_video_retake_tool

    # Include source-binding indices for reference and candidate view requests.
    operations = visual_tools.build_vwm_tools(
        timeout_seconds=7500,
        candidate_conditioned_fields=True,
    )
    return [
        visual_tools.build_load_skill_tool(compact_output=True),
        *operations,
        build_video_retake_tool(),
    ]
