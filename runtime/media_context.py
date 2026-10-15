"""Stable model-visible identifiers for observed and generated visual evidence."""

from __future__ import annotations

from typing import Any


def observed_evidence_id(index: int) -> str:
    return f"observed_{int(index):02d}"


def predicted_evidence_id(generation_index: int, frame_position: int) -> str:
    return f"predicted_{int(generation_index):02d}_{int(frame_position):02d}"


def observed_attachment_label(index: int, role: str) -> str:
    return (
        f"Original {observed_evidence_id(index)}: task-order image {index + 1}; "
        f"source_image_index={index}; role={role}."
    )


def generation_media_context(
    dispatch: dict[str, Any],
    sampled_frames: list[dict[str, Any]],
    generation_index: int,
) -> dict[str, Any]:
    source_indices = [
        int(index)
        for index in dispatch.get("source_image_indices") or []
        if isinstance(index, int) and not isinstance(index, bool)
    ]
    return {
        "generation_index": generation_index,
        "source_order_mode": dispatch.get("source_order_mode"),
        "source_images": [
            {
                "evidence_id": observed_evidence_id(source_index),
                "source_image_index": source_index,
                "task_order_image": source_index + 1,
                "source_picture": local_position + 1,
            }
            for local_position, source_index in enumerate(source_indices)
        ],
        "generated_frames": [
            {
                "evidence_id": predicted_evidence_id(generation_index, frame_position),
                "temporal_position": frame_position + 1,
                "temporal_count": len(sampled_frames),
                "frame_index_in_video": int(frame.get("frame_index") or 0),
            }
            for frame_position, frame in enumerate(sampled_frames)
        ],
    }
