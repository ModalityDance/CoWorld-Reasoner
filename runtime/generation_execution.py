"""Execute VWM requests and attach ordered generated frames to the task history."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime.agents_sdk_input import image_data_url
from runtime.visual_dispatch import VisualDispatchContext
from evolved_harness.visual_generation import (
    VWMBackendConfig,
    execute_visual_generation,
    technically_valid_generation,
)
from runtime.ffmpeg_runtime import ffmpeg_executable


@dataclass
class GenerationContext(VisualDispatchContext):
    sample_id: str = ""
    benchmark: str = ""
    backend_config: VWMBackendConfig | None = None
    max_verifier_calls: int = 2
    verifier_calls: int = 0
    generation_records: list[dict[str, Any]] = field(default_factory=list)
    valid_media_records: list[dict[str, Any]] = field(default_factory=list)
    verifier_records: list[dict[str, Any]] = field(default_factory=list)
    sampled_frame_budget: int = 9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_video_frames(
    video: Path,
    output_dir: Path,
    *,
    frame_count: int,
    budget: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = max(2, int(frame_count or 0))
    indices = sorted(
        {
            round(index * (count - 1) / max(1, budget - 1))
            for index in range(max(2, budget))
        }
    )
    frames: list[dict[str, Any]] = []
    executable = ffmpeg_executable()
    if not executable:
        raise FileNotFoundError("ffmpeg executable is unavailable")
    for frame_index in indices:
        output = output_dir / f"frame_{frame_index:04d}.jpg"
        proc = subprocess.run(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"select=eq(n\\,{frame_index})",
                "-frames:v",
                "1",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if proc.returncode != 0 or not output.is_file():
            raise RuntimeError(
                f"generation_frame_extraction_failed:{frame_index}:{proc.stderr[-500:]}"
            )
        frames.append(
            {
                "path": str(output.resolve()),
                "frame_index": frame_index,
                "sha256": _sha256(output),
            }
        )
    return frames


def _sample_generation_artifacts(
    context: GenerationContext,
    generation: dict[str, Any],
    *,
    generation_index: int,
) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for artifact_index, artifact in enumerate(generation.get("artifact_records") or []):
        path = Path(str(artifact.get("path") or "")).resolve()
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            raise ValueError("generation_artifact_hash_mismatch")
        if path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
            probe = artifact.get("media_probe") or {}
            frames = _extract_video_frames(
                path,
                context.output_dir
                / "generated_frames"
                / f"generation_{generation_index:02d}_artifact_{artifact_index:02d}",
                frame_count=int(probe.get("frame_count") or 0),
                budget=context.sampled_frame_budget,
            )
            for item in frames:
                item.update(
                    {
                        "artifact_path": str(path),
                        "artifact_sha256": artifact["sha256"],
                        "artifact_index": artifact_index,
                        "evidence_modality": "video_frame",
                    }
                )
            sampled.extend(frames)
        else:
            sampled.append(
                {
                    "path": str(path),
                    "frame_index": 0,
                    "sha256": artifact["sha256"],
                    "artifact_path": str(path),
                    "artifact_sha256": artifact["sha256"],
                    "artifact_index": artifact_index,
                    "evidence_modality": "generated_image",
                }
            )
    return sampled


def execute_generation_request(
    raw_context: VisualDispatchContext,
    dispatch: dict[str, Any],
    call_id: str,
) -> Any:
    """Execute one validated model-authored request and return media to the same agent."""

    from agents.tool import ToolOutputImage, ToolOutputText

    if not isinstance(raw_context, GenerationContext):
        raise TypeError("generation_context_required")
    context = raw_context
    model_tool_name = str(dispatch.get("tool_interface") or "vwm_dispatch_tool")
    try:
        if context.backend_config is None:
            raise RuntimeError("generation_backend_config_missing")
        generation = execute_visual_generation(
            sample_id=context.sample_id,
            benchmark=context.benchmark,
            dispatch=dispatch,
            observed_paths=list(context.observed_image_paths),
            config=context.backend_config,
        )
        generation["technically_valid"] = technically_valid_generation(generation)
        generation_index = len(context.generation_records)
        sampled = (
            _sample_generation_artifacts(
                context,
                generation,
                generation_index=generation_index,
            )
            if generation["technically_valid"]
            else []
        )
        record = {
            "generation_index": generation_index,
            "call_id": call_id,
            "dispatch": dispatch,
            "generation": generation,
            "sampled_frames": sampled,
        }
        context.generation_records.append(record)
        if generation["technically_valid"] and sampled:
            context.valid_media_records.append(record)
        status = (
            "success"
            if generation["technically_valid"] and sampled
            else "invalid_or_unreadable_generated_artifact"
        )
        payload = {
            "tool_name": model_tool_name,
            "call_id": call_id,
            "status": status,
            "backend": generation.get("backend"),
            "operation": generation.get("operation"),
            "generation_index": generation_index,
            "technically_valid": generation["technically_valid"],
            "predicted_evidence_not_oracle": True,
            "sampled_frame_count": len(sampled),
            "artifact_hashes": [
                item.get("sha256") for item in generation.get("artifact_records") or []
            ],
            "verifier_capability_now_available": bool(context.valid_media_records),
            "observation": (
                "New generated visual evidence is attached below. Compare it with the original "
                "observations for the requested relation before deciding the answer."
            ),
        }
        evidence_focus = str(dispatch.get("evidence_focus") or "").strip()
        if evidence_focus:
            payload["evidence_focus"] = evidence_focus
        from runtime.media_context import generation_media_context

        media_context = generation_media_context(dispatch, sampled, generation_index)
        payload["media_context"] = media_context
        context.emit("vwm_result", payload)
        context.emit("tool_result", payload)
        model_payload = {
            "status": status,
            "operation": generation.get("operation"),
            "generation_index": generation_index,
            "sampled_frame_count": len(sampled),
            "predicted_evidence_not_oracle": True,
            "observation": (
                "New generated visual evidence is attached below. Compare it with the original "
                "observations for the requested relation before deciding the answer."
            ),
        }
        if isinstance(model_payload, dict) and evidence_focus:
            model_payload["evidence_focus"] = evidence_focus
        if isinstance(model_payload, dict):
            model_payload["media_context"] = media_context
        outputs: list[Any] = [
            ToolOutputText(
                text=json.dumps(model_payload, ensure_ascii=False, sort_keys=True)
            )
        ]
        for frame_number, item in enumerate(sampled):
            data_url = image_data_url(Path(item["path"]))
            if not data_url:
                raise RuntimeError(
                    f"generation_sampled_frame_unreadable:{item['path']}"
                )
            outputs.append(ToolOutputImage(image_url=data_url, detail="high"))
        return outputs
    except (
        Exception
    ) as exc:  # the failed observation remains inside the same autonomous run
        payload = {
            "tool_name": model_tool_name,
            "call_id": call_id,
            "status": "generation_failed",
            "error": f"{type(exc).__name__}:{str(exc)[:1000]}",
            "verifier_capability_now_available": bool(context.valid_media_records),
            "instruction": (
                "This call produced no new usable visual evidence. Continue autonomously from "
                "the observations that are actually available."
            ),
        }
        context.generation_records.append(
            {
                "generation_index": len(context.generation_records),
                "call_id": call_id,
                "dispatch": dispatch,
                "error": payload["error"],
                "sampled_frames": [],
            }
        )
        context.emit("vwm_result", payload)
        context.emit("tool_result", payload)
        model_payload = {
            "status": "generation_failed",
            "operation": dispatch.get("operation"),
            "error": payload["error"],
            "evidence_returned": False,
        }
        return json.dumps(model_payload, ensure_ascii=False, sort_keys=True)
