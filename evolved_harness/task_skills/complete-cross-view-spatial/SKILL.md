---
name: complete-cross-view-spatial
description: "Use for a visual question whose answer depends on a missing connection between separate observations: identity, viewpoint, handedness, or layout."
---

# Cross-View Spatial Completion

## Use when

Use when separate observations do not completely establish an answer-bearing identity, viewpoint,
handedness, or layout relation. A guess from separate local cues is not a complete observation when
a grounded connecting view could distinguish the remaining interpretations. Do not use for a
relation already completely visible in the required view, or for static text, counts, source detail,
or exact metrics.

## Decide and act

- Use `vwm_new_view_tool` for a missing grounded viewpoint or cross-view mapping.
- Use `vwm_temporal_interpolation_tool` only for genuinely ordered endpoints whose connecting motion
  is answer-relevant.
- Use `vwm_retake_tool` only after a returned artifact has one concrete, repairable defect.

If the originals resolve the relation, answer directly. Otherwise, use the matching VWM operation
only when it could distinguish the remaining interpretations. Loading the skill does not require
a VWM call. For mutually exclusive
candidates, query one unresolved reference-to-candidate relation with `independent_candidates`;
put the reference first, set both image indices, and pass exactly
`[reference_image_index, candidate_image_index]` as `source_image_indices`. Use
`reference_to_anchor` for a complementary anchor and `ordered_transition` only for observed order.
Choose whether another query is useful after reading the new evidence.

Decision test: when the answer depends on how a scene connects across viewpoints, two endpoint
images or a change in apparent size do not by themselves establish the intervening viewpoint, motion,
or depth relation. Treat that relation as unresolved unless it is directly visible under the question's
reference frame; then use the matching VWM operation when it can distinguish the remaining
interpretations.

## Request

State the missing relation, minimum source images, persistent anchors, and viewpoint change. Keep the
request answer-neutral: do not name an answer, option, preferred candidate, or conclusion.

## Inspect and integrate

Compare returned media with the originals for identity, anchors, viewpoint, scene structure, and the
requested relation. Distinguish camera motion from object motion. Reject endpoint mismatch,
identity replacement, or duplicated structure. If the artifact may affect the answer, use
`viewpoint_layout_verifier` for an independent relation comment. The main Agent makes the final
decision.

Generated completion supplies predicted evidence. It does not recover hidden source detail or exact
measurements.
