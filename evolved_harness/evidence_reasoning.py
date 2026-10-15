"""Evidence-reading and answer schemas for the staged reasoning loop."""
from __future__ import annotations

from pydantic import BaseModel, Field


class VideoInterpretation(BaseModel):
    observations: list[str]
    reference_frame_mapping: str
    defects_and_uncertainty: list[str]
    evidence_use_reasoning: str


class EvidenceIntegration(BaseModel):
    task_reference: str = Field(min_length=1, description='Target and task-specified state/reference frame.')
    discriminating_evidence: str = Field(min_length=1, description='Concrete source cue separating the alternatives, including material contrary evidence.')
    reasoning_summary: str = Field(min_length=1, description='Derive the required relation from the evidence and task premises before selecting the answer.')
    answer: str = Field(min_length=1)


def video_interpretation_instructions(records):
    has_frames = any(r.get('sampled_frames') for r in records)
    evidence = (
        'Inspect the generated frames actually present in the preceding tool results '
        'alongside the original observations. A compatible verifier can provide an '
        'independent reading of the generated artifact if useful. '
        if has_frames else
        'No generated frames were delivered in the preceding stage. Review the '
        'original observations and preceding request/no-request reasoning; do not '
        'describe a predicted artifact as observed evidence. '
    )
    return (
        'Continue into P3. ' + evidence +
        'Explain identity, the exact task reference frame, visible changes, defects '
        'and unresolved relations. Reassess whether the preceding capability judgment '
        'covered all task-relevant observations and whether any claimed evidence '
        'actually distinguishes the alternatives. Decide autonomously how useful '
        'the available evidence is; do not select the final answer here.'
    )
