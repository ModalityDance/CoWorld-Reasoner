"""Guidance for constructing visual generation requests."""

STAGE_GUIDANCE = """
Before writing the generation request, resolve which original image defines each
target and reference frame. Bind Picture numbers to the source order actually
submitted, not to a different order remembered from the question. Identify a
visible stable anchor for the requested movement direction. If forward is object
relative, use the object's visible front/back rather than assuming forward means
toward a nearby destination or the camera. Preserve the stated direction and
extent; do not silently substitute the action whose outcome seems likely.

Your P1 judgment is revisable. If reconsidering a prior no-request judgment,
explain which concrete ambiguity remains despite its original evidence, rather
than merely saying that the future frame or connecting video is unobserved.
Consider whether the operation can expose a discriminating relation at the
required fidelity. An unobserved face, gaze, label, or other fine source detail
may be invented in a generated view; realism is not recovery of the original fact.
Decide yourself whether generation still offers useful evidence and state its
limits. Do not treat a supplied endpoint or intended direction as independent
confirmation of the hypothesis used to request it. No rule here forces a tool
call, suppresses one, or chooses an answer.
"""

REQUEST_GROUNDING_GUIDANCE = """
Translate the original question into an executable request after selecting its
sources. In generation text Picture 1 is the first submitted source and Picture 2
is the second: for sources [1, 0], original image 2 becomes local Picture 1.
Use visible landmark names where that is clearer. H3 sees only those submitted
images, not the other original images in your context. If a known target or
endpoint matters, consider how its identity is actually supplied; a one-image
request cannot preserve an unseen target merely by naming it.

Choose the operation by the unresolved relation, not only by the skill selected
earlier. A single observed image can support a bounded causal intervention; it
does not require a recorded temporal interval. Conversely a missing future frame
alone does not establish useful information gain. Explain which competing spatial
or motion interpretations the generated observation could distinguish.

Ground object-relative movement with visible front/back features before writing
the direction. Saying 'visible front' without identifying its cue does not resolve
an uncertain heading. For a fixed-camera intervention do not simultaneously request
a camera path. Describe the task's action without adding its unknown consequence.
For a bridge to a candidate endpoint, do not label the path with the very side or
heading you are trying to infer. The generated path can test consistency, while
the relation still requires interpretation from the original and generated anchors.
"""
