"""Stage-specific criteria and evidence for harness candidate assessment."""
from __future__ import annotations

from typing import Any


STAGE_FEEDBACK = {
    "P1": ("original_observations", "stage_output", "skill_loading", "execution_feedback"),
    "P2": ("original_observations", "stage_output", "skill_loading", "execution_feedback",
           "vwm_request", "generation"),
    "P3": ("original_observations", "stage_output", "skill_loading", "execution_feedback",
           "vwm_request", "generation", "verification"),
    "P4": ("original_observations", "stage_output", "skill_loading", "execution_feedback",
           "vwm_request", "generation", "verification", "answer_score"),
}

STAGE_INSTRUCTIONS = {
    "P1": """Capability Boundary Assessment.
Form a reliable capability judgment: decide whether the supplied observations already
determine the answer-bearing relation, identify the missing relation when they do not,
and select a useful skill or visual operation only when it could distinguish the live
interpretations.""",
    "P2": """Request VWM Assistance.
Form a reliable request from the information gap: bind the right source images and reference relationships,
identities and neutral operation controls, and make sure the request's
semantics agree with the actual tool and backend execution.""",
    "P3": """Video Interpretation.
Form reliable generated evidence: separate original facts, task conditions, request
assumptions, generated observations and verifier findings; track identity, anchors,
motion and defects; and preserve only relations the media actually supports.""",
    "P4": """Evidence Integration and Reasoning.
Form a reliable final derivation: establish the task reference frame, weigh original and
generated evidence by what each can support, resolve conflicts and uncertainty, and produce
the requested answer without silently converting screen placement into a new relation.""",
}


def available_feedback(row: dict[str, Any], stage: str) -> set[str]:
    kinds = {"original_observations", "execution_feedback"}
    output = row.get("stage_output") or (row.get("p1_output") if stage == "P1" else None)
    if output:
        kinds.add("stage_output")
    if "skill_loads" in row or "loaded_skill_id" in row or row.get("p1_output"):
        kinds.add("skill_loading")
    if row.get("accepted_dispatches"):
        kinds.add("vwm_request")
    if row.get("generation_records"):
        kinds.add("generation")
    if row.get("verifier_records"):
        kinds.add("verification")
    if (row.get("scoring") or {}).get("correct") is not None:
        kinds.add("answer_score")
    return kinds.intersection(STAGE_FEEDBACK[stage])
