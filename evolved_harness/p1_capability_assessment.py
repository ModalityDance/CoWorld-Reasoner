"""Capability-assessment guidance for the first reasoning stage."""

STAGE_GUIDANCE = """
For the P1 capability judgment, evaluate the described VWM operations as capabilities
available in later stages. Their execution is deliberately disabled in P1. That
experimental restriction is not a reason to classify an operation as incapable.

Judge the value of a reasoning skill separately from the value of generating media.
Loading a skill may improve interpretation even when no VWM request is useful. If
you loaded a skill and then decided not to request VWM, retain its identity in
selected_task_skill and explain the no-VWM decision independently.

Separate direct observation from an inference supported by visual cues. Objects
visible in separate photographs are not jointly observed at a calibrated scale.
A supported inference can still justify no VWM when generation adds no discriminative
information; describe that as non-discriminative or outside capability, as appropriate.

For a useful VWM opportunity, explain which visible change would distinguish the
remaining interpretations and which original anchor/reference frame makes that
change meaningful. An unobserved state alone does not establish capability fit.
Smooth continuity or reaching an endpoint alone does not determine signed 3D
displacement, intrinsic handedness, or object identity. Consider whether the
operation preserves the specific anchors needed, and state uncertainty when it
cannot guarantee them. A familiar object or plausible path is not calibration.
Do not call solely because a question describes motion; visible static geometry
may already determine the requested outcome. These considerations guide your
autonomous decision, without requiring a call, a skip, or an answer change.

Before declaring the relation directly observed, identify where each required
target, condition and reference appears in the original images. A locally open
path does not identify a destination shown in another view; the corresponding
cross-view relation still needs to be established. If original shared anchors
already establish it, explain that mapping rather than silently ignoring a view.
Do not assume a generated bridge calibrates lengths in separate images. For a
size comparison, explain a real shared scale cue or acknowledge the limitation;
visual plausibility alone cannot make generation discriminative.
When motion language merely specifies a footprint at an opening or an explicit
straight path, assess whether that static relation suffices before proposing
unconstrained dynamics. Preserve the task's stated conditions.
"""
