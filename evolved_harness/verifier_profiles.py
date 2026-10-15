"""Artifact-dependent logical verifier profiles."""

from __future__ import annotations

from dataclasses import dataclass

from evolved_harness.catalog import (
    SEQUENCE_CONTINUATION_TOOL,
    INTERVENTION_TOOL,
    MOTION_DECOMPOSITION_TOOL,
    VIEW_CHANGE_TOOL,
    TEMPORAL_INTERPOLATION_TOOL,
    VISIBILITY_REVEAL_TOOL,
)


@dataclass(frozen=True)
class VerifierProfile:
    profile_id: str
    source_tools: tuple[str, ...]
    media_types: tuple[str, ...]
    purpose: str


PROFILES = (
    VerifierProfile(
        profile_id="temporal_motion_verifier",
        source_tools=(TEMPORAL_INTERPOLATION_TOOL, SEQUENCE_CONTINUATION_TOOL, MOTION_DECOMPOSITION_TOOL),
        media_types=("video",),
        purpose="Read bridge motion, temporal order, continuity, and camera-versus-object motion.",
    ),
    VerifierProfile(
        profile_id="viewpoint_layout_verifier",
        source_tools=(VIEW_CHANGE_TOOL, TEMPORAL_INTERPOLATION_TOOL, VISIBILITY_REVEAL_TOOL, MOTION_DECOMPOSITION_TOOL),
        media_types=("image", "video"),
        purpose=(
            "Read identity, persistent anchors, viewpoint execution, topology, and layout "
            "from a generated view or connecting-view sequence."
        ),
    ),
    VerifierProfile(
        profile_id="intervention_outcome_verifier",
        source_tools=(INTERVENTION_TOOL,),
        media_types=("video",),
        purpose="Read target adherence, displacement, contact, occlusion, and stable-anchor relation.",
    ),
)
