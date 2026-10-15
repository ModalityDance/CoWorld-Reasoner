"""Guidance for integrating visual evidence into the final answer."""

STAGE_GUIDANCE = """
Derive the discriminating relation before selecting the final answer. In your
structured response first state the task reference and required state, then the
concrete visible cue that separates the remaining alternatives, then perform the
needed conversion or calculation, and finally give the answer. Use these fields
to express your own reasoning.

For spatial tasks, build the comparison from the original subject's distinctive
features in the specified reference frame. A front-view observer's left/right,
the object's own left/right, and a side photograph's screen axes are different
quantities. Track the front-to-rear axis and the visible flank through the pose
change rather than naming a flank from the final nose direction without a frame.
For movement, evaluate the requested endpoint or interval explicitly; initial
overlap and intermediate crossing need not equal final obstruction. Apply any
task-supplied rule or size/fit criterion as stated.

Check the strongest contrary cue from the actual available images or task
premises against your proposed relation. Earlier model/verifier statements are
interpretations, and generated motion may have inherited request assumptions.
If a generated cue and an original cue appear inconsistent, identify which
specific feature, transform, identity or action accounts for the difference.
Keep whatever indirect evidence is justified; do not automatically prefer an
earlier answer, reject generation, or claim a missing detail is visible. A
statement of confidence or uncertainty does not replace the distinguishing cue.
The final answer remains your own judgment from that derivation.
"""
