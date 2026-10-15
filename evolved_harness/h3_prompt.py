"""Compile model-authored visual requests into the hosted H3 prompt format."""
from __future__ import annotations

from runtime.contracts import ToolRequest
from runtime.request_media import artifact_refs_from_request


def build_h3_prompt(request: ToolRequest) -> str:
    """Compile an answer-free VWM-driven tool request into H3's documented prompt shape."""

    controls = request.controls or {}
    image_refs = artifact_refs_from_request(request)
    picture_lines = [
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced."
    ]
    if len(image_refs) >= 2:
        picture_lines.append(
            "At the end of the target video, <Picture 2> (from [Shot 1]) is fully referenced."
        )

    tool_name = request.tool_interface
    instruction_parts = [
        controls.get("generation_instruction"),
        controls.get("camera_instruction"),
        controls.get("motion_prompt"),
    ]
    instructions = " ".join(
        str(value).strip().rstrip(".")
        for value in instruction_parts
        if str(value or "").strip()
    ) or "show a conservative, grounded change of viewpoint"
    preserve = [
        str(value).strip().rstrip(".")
        for value in (controls.get("preserve") or [])
        if str(value or "").strip()
    ]
    preserve_text = "; ".join(preserve) or "all visible scene geometry and object identity"
    camera_request = bool(controls.get("camera_instruction") or tool_name in {
        "vwm_new_view_tool", "vwm_visibility_reveal_tool", "vwm_motion_decomposition_tool",
    })
    if tool_name == "vwm_visibility_reveal_tool":
        action_text = (
            "Generate one continuous grounded viewpoint change that exposes the requested "
            "hidden region or target from the supplied observation."
        )
        motion_constraint = (
            "Reveal only by a plausible camera/viewpoint change; preserve fixed landmarks, "
            "subject identity, scene geometry, and the visible content already observed."
        )
    elif tool_name == "vwm_motion_decomposition_tool":
        action_text = (
            "Generate one continuous controlled visual probe that makes the relative motion "
            "of the camera, subject, and fixed scene landmarks inspectable."
        )
        motion_constraint = (
            "Keep fixed scene landmarks and subject identity consistent; do not replace camera "
            "motion with object enlargement or replace object motion with a camera pan."
        )
    elif tool_name == "vwm_temporal_continuation_tool":
        action_text = (
            "Generate one continuous temporal continuation from the supplied observed state."
        )
        motion_constraint = (
            "Preserve the same physical scene and subject identity while showing only a coherent "
            "short continuation in the requested temporal direction."
        )
    elif camera_request:
        action_text = (
            "Generate one continuous camera path from the viewpoint shown in Picture 1 "
            "to the viewpoint shown in Picture 2."
            if len(image_refs) >= 2
            else "Generate one continuous camera path from the provided viewpoint."
        )
        motion_constraint = (
            "Move the camera smoothly through the same physical scene; keep fixed scene "
            "landmarks and subject identity consistent, and do not replace camera motion "
            "with independent object enlargement or object motion."
        )
    else:
        action_text = (
            "Generate one continuous visual action from Picture 1 to Picture 2."
            if len(image_refs) >= 2
            else "Generate one continuous visual action from the provided visual anchor."
        )
        motion_constraint = (
            "Keep the same physical scene and persistent subject identity throughout the "
            "action, with a coherent progression between the supplied visual anchors."
        )
    shot = (
        "[Shot 1] Generate a photorealistic single continuous visual-evidence clip. "
        f"{action_text} The requested visual action is: {instructions}. "
        f"Preserve: {preserve_text}. "
        f"{motion_constraint} "
        "Do not add captions, labels, option letters, arrows, check marks, watermarks, or other "
        "answer-bearing marks; do not introduce a scene cut, dissolve, overlay, object swap, or unrelated content."
    )
    return "\n\n".join(
        [
            "integrated_multimodal_description: "
            + " ".join(picture_lines)
            + "\n"
            + shot,
            "overall_soundscape: N/A. No dialogue or audio cue is needed; prioritize the visual evidence.",
            "non_diegetic_music: N/A.",
        ]
    )
