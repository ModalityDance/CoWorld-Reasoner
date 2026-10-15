"""Guidance for interpreting original and generated visual evidence."""

STAGE_GUIDANCE = """
For each consequential inference, distinguish what the original pixels show,
what the task stipulates, what the generation request assumes, and what the
generated frames add. Explain the actual connection from a visible change to
the requested relation. A request being executed faithfully does not independently
validate the direction, endpoint or identity assumption used to write that request.
Inspect whether the video preserved those assumptions and whether they match the
original question and sources. A realistic newly exposed gaze, label or small
detail is not recovery of an unseen original fact; explain what original anchors
would make that detail informative before relying on it.

Use the frame stipulated by the question. If a source heading is defined as north,
no external compass is necessary; the issue is whether relative pose and identity
are sufficiently grounded. For ego-motion distinguish rotation, translation and
object deformation before converting image-plane movement into camera direction.
For side identity compare the initial front reference, camera orbit and persistent
features, rather than equating a final screen direction with intrinsic left/right.
Keep viewer-relative and object-intrinsic side names distinct throughout the
explanation; apply the convention the question actually specifies. Two targets
need not appear in one frame if persistent anchors support their cross-frame
relation; explain whether that tracking and transform are actually established.

Neither a caveat nor a review status determines evidence usefulness. You may
combine indirect cues, keep unaffected relations when a segment is defective,
or find the evidence inconclusive. State the concrete basis and limitations of
your own judgment; do not automatically revert to a prior answer or automatically
accept or discard the artifact. Do not select the final answer in P3.
"""

REFERENCE_FRAME_GUIDANCE = """
Anchor the subject before revising a spatial claim. If several plants, chairs or
similar objects appear, retain the question's identifying features and original
image binding; the object nearest the moving subject is not necessarily the
queried target. Explain any changed identity with actual distinguishing features.
When revising front/back, distinguish the windshield/mirrors/hood from rear
glass/tailgate/spoiler or other visible front/rear features. Reassessment alone
is not evidence that the earlier claim was wrong.
For fine facial detail, separate a coarse head/body pose already visible in the
original from newly synthesized eye direction. Explain which source cue supports
the particular claim; calling invented detail low-confidence does not establish
that connection.

For a claimed camera orbit, compare the motion of the subject with at least two
background anchors across early, middle and late frames. A changing subject pose
against an almost fixed background can indicate object/platform rotation or
deformation, rather than the requested camera movement. Describe the observed
motion first, then infer what it permits about the original reference frame.
The verifier can repeat a request assumption too; evaluate its cited visible
cues against the frames before adopting a consequential transform.

Do not make your analysis more certain merely by adding caveats or more skeptical
merely because it is generated. Keep the original task identity and stipulated
frame fixed while assessing which relations the actual anchors support. If a
relation is supported, carry out its frame conversion explicitly rather than
only saying the images are sufficient and leaving the conversion unexplained.
"""

VERIFIER_GUIDANCE = """
Observe the actual temporal change before labeling it with the requested action.
For viewpoint motion, compare a subject landmark and two background landmarks in
early, middle and late frames: if the background barely moves while the subject
turns, consider object/platform rotation or deformation instead of camera orbit.
For forward/backward motion identify visible front and rear features; the request
is not evidence for which end is front. Keep the question's target identity
distinct from nearby similar objects. When multiple objects could match, report
the visible distinguishing features and ambiguity instead of substituting one.
Only generated frames are attached. Do not invent checks of the original images
or derive an unknown side from the side named in the request. Preserve legitimate
partial cues and leave evidence integration to the main Agent.
"""
