---
name: simulate-causal-intervention
description: "Use when the answer depends on one unobserved bounded action on a visible target."
---

# Causal Intervention Simulation

## Use when

Use when the question asks for the outcome of one bounded action on a visible target and that
outcome is not already observed. This includes movement, falling, collision, removal, or an
occlusion change. Static recognition and relations already visible in the original scene belong to
direct reasoning. If ordered observed anchors define a missing interval, use temporal continuity.

## Choose an operation

Use `vwm_simulate_intervention_tool` for one target and one action supported by one source
observation. The structured controls and the generation request must describe the same target and
action. Do not add a second moving target or an unrelated scene edit.

If the outcome is already established by the originals, answer directly. Otherwise, call the
matching VWM operation only when its predicted outcome could distinguish the remaining
interpretations. Loading this skill does not require a VWM call.

Use `vwm_retake_tool` for a concrete, repairable defect in returned media. The main Agent decides
whether another call is useful.

## Request

Name the target, intended action, preserved scene elements, and outcome relation to expose. Keep
the request neutral: do not include an answer, option, preferred outcome, or conclusion. Describe
only the intervention; do not ask the generated media to prove the answer.

## Read and integrate

Separate the requested action from what visibly occurred. Check target identity, action adherence,
stable anchors, contact, occlusion, displacement, and relevant secondary motion. Generated media is
predicted evidence, not ground truth. Use only supported relations and preserve partial results,
conflicts, or uncertainty.

If the generated media may affect the answer, use the matching verifier for an independent
answer-free comment. Combine it with the media and originals; it does not choose the final answer.

## Limits

The result is a plausible visual consequence, not exact mechanics or guaranteed real-world
causality. Unsupported multi-target or composite edits are outside this skill.
