---
name: reason-temporal-spatial-continuity
description: "Use when a missing answer-relevant transition lies between genuinely ordered observations."
---

# Temporal-Spatial Continuity

## Use when

Use when the answer depends on what happens between ordered observations: camera or viewpoint
motion, actor or object movement, or a change in proximity. Do not use for a static fact already
visible in one observation. A hypothetical action from one scene is an intervention outcome and
belongs to the causal skill.

## Decide and choose an operation

Use `vwm_temporal_interpolation_tool` for the shortest ordered interval containing the missing
state. Choose endpoints that constrain the relevant identities and motion. The result is predicted
visual continuity, not an exact geometry, distance, speed, or angle measurement.

If the originals resolve the transition, answer directly. Otherwise, call the matching VWM operation
only when its predicted transition could distinguish the remaining interpretations. Loading this
skill does not require a VWM call.

Decision test: a pair of ordered images shows two states, not the motion between them. If the answer
depends on the direction or manner of the transition, do not treat a static displacement or apparent
size change alone as a complete observation; use the matching VWM operation when the transition would
resolve the relation.

Use `vwm_retake_tool` only for a concrete, repairable defect in returned media. The main Agent still
decides whether another call is useful.

## Request

State the time order, relevant entities, stable anchors, and relation that should become visible.
Keep the request neutral: do not include an answer, option, preferred direction, or conclusion.

## Read and integrate

Read the sequence in time order and compare its beginning and end with the original observations.
Check identity, order, the requested motion, and camera-versus-object movement. Use only relations
supported by the pixels and preserve partial results, conflicts, or uncertainty.

If the generated media may affect the answer, use the matching verifier for an independent
answer-free comment. Combine it with the media and originals; it cannot choose the final answer.

## Limits

Interpolation supplies plausible predicted continuity, not physical truth. A coherent sequence that
does not expose the requested relation is not useful evidence.
