"""Execute model-authored requests through configured VWM backends."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from runtime.contracts import ToolRequest, ToolReturn


OPERATION_TOOL_INTERFACES = {
    "temporal_interpolation": "vwm_temporal_interpolation_tool",
    "view_change": "vwm_new_view_tool",
    "sequence_continuation": "vwm_temporal_continuation_tool",
    "intervention": "vwm_simulate_intervention_tool",
}


@dataclass(frozen=True)
class VWMBackendConfig:
    output_root: Path


def build_generation_request(
    *,
    sample_id: str,
    benchmark: str,
    dispatch: dict[str, Any],
    observed_paths: list[Path],
) -> ToolRequest:
    """Compile a validated visual dispatch into a backend generation request."""

    operation = str(dispatch.get("operation") or "")
    if operation not in OPERATION_TOOL_INTERFACES:
        raise ValueError(f"unsupported_generation_operation:{operation}")
    # Several distinct tools share a low-level operation. Preserve the selected
    # tool so the adapter can apply its specific generation instructions.
    tool_interface = dispatch.get("tool_interface")
    if tool_interface == "vwm_retake_tool":
        tool_interface = dispatch.get("retake_source_tool_interface")
    tool_interface = tool_interface or OPERATION_TOOL_INTERFACES[operation]
    tool_operations = {
        **{tool: op for op, tool in OPERATION_TOOL_INTERFACES.items()},
        "vwm_visibility_reveal_tool": "view_change",
        "vwm_motion_decomposition_tool": "view_change",
    }
    if tool_operations.get(tool_interface) != operation:
        raise ValueError(f"generation_tool_operation_mismatch:{tool_interface}:{operation}")
    indices = dispatch.get("source_image_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError("generation_dispatch_requires_source_image_indices")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("generation_source_image_indices_must_be_integers")
    if any(index < 0 or index >= len(observed_paths) for index in indices):
        raise ValueError("generation_source_image_index_out_of_range")
    source_paths = [observed_paths[index].resolve() for index in indices]
    if any(not path.is_file() for path in source_paths):
        raise FileNotFoundError("generation_source_image_missing")

    operation_controls = dispatch.get("operation_controls")
    if not isinstance(operation_controls, dict):
        raise ValueError("generation_operation_controls_missing")
    generation_instruction = str(dispatch.get("generation_instruction") or "").strip()
    if not generation_instruction:
        raise ValueError("generation_instruction_missing")
    controls = {
        key: value for key, value in operation_controls.items() if value is not None
    }
    controls.update(
        {
            "input_artifact_refs": [str(path) for path in source_paths],
            "world_state_gap": str(dispatch.get("information_gap") or "").strip(),
            "visual_task_focus": str(dispatch.get("expected_evidence") or "").strip(),
            "evidence_inspection_policy": str(
                dispatch.get("expected_evidence") or ""
            ).strip(),
            "generation_instruction": generation_instruction,
            "preserve": list(dispatch.get("preserve") or []),
            "benchmark": benchmark,
            "operation": operation,
        }
    )
    # Preserve the model-authored meaning of the ordered inputs. The paths
    # already follow source_image_indices; these fields make that order
    # explicit to adapters and to the evidence manifest.
    for key in (
        "source_order_mode",
        "reference_image_index",
        "candidate_image_index",
    ):
        value = dispatch.get(key)
        if value is not None:
            controls[key] = value
    if dispatch.get("source_order_mode") == "independent_candidates":
        controls.update(
            {
                "candidate_conditioned_bridge": True,
                "candidate_alternatives_not_temporal_order": True,
                "candidate_bridge_segments_are_independent": True,
                "input_strategy": "reference_to_candidate_bridge",
            }
        )

    request = ToolRequest(
        tool_interface=tool_interface,
        input_evidence_ids=[f"{sample_id}/observed/{index:02d}" for index in indices],
        request_id=f"generation/{sample_id}/{dispatch.get('dispatch_id') or operation}",
        controls=controls,
        constraints={
            "answer_free_controls": True,
            "predicted_evidence_not_oracle": True,
            "validated_visual_dispatch": True,
        },
        requested_by_skill_id=str(dispatch.get("requested_by_skill_id") or ""),
        permission_policy={
            "generation_only": True,
            "agent_media_reading": False,
            "verifier_disclosed": False,
        },
    )
    return request


def execute_visual_generation(
    *,
    sample_id: str,
    benchmark: str,
    dispatch: dict[str, Any],
    observed_paths: list[Path],
    config: VWMBackendConfig,
) -> dict[str, Any]:
    requested_backend = str(dispatch.get("backend") or "")
    if requested_backend != "fal_h3":
        raise ValueError(f"unsupported_generation_backend:{requested_backend}")
    # Carry the selected backend identity through the request so the evidence
    # manifest matches the operation authored by the agent.
    effective_dispatch = dispatch
    request = build_generation_request(
        sample_id=sample_id,
        benchmark=benchmark,
        dispatch=effective_dispatch,
        observed_paths=observed_paths,
    )
    backend = str(effective_dispatch.get("backend") or "")
    output_root = config.output_root.resolve()
    candidate_requests = [request]
    from evolved_harness.fal_h3_backend_adapter import FalH3Config, fal_h3_tool_function

    fal_config = replace(
        FalH3Config.from_env(),
        output_root=output_root / "fal_h3",
    )
    result = fal_h3_tool_function(request, fal_config)

    return {
        "requested_backend": requested_backend,
        "backend": backend,
        "effective_backend": backend,
        "backend_provenance": effective_dispatch.get(
            "backend_provenance",
            {
                "requested_by_agent": requested_backend,
                "effective_runtime_backend": backend,
                "selection": "agent_request",
            },
        ),
        "operation": dispatch["operation"],
        "dispatch_id": dispatch.get("dispatch_id"),
        "tool_request": asdict(request),
        "candidate_tool_requests": [asdict(item) for item in candidate_requests],
        "tool_return": asdict(result),
        "artifact_records": artifact_records(result),
        "stage_boundary": {
            "generation_executed": True,
            "agent_media_reading": False,
            "verifier_disclosed": False,
            "answer_updated": False,
        },
    }


def artifact_records(result: ToolReturn) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    refs = (result.artifacts or {}).get("artifact_refs") or []
    for raw_path in refs:
        path = Path(str(raw_path)).resolve()
        record: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": _sha256_file(path) if path.is_file() else None,
            "suffix": path.suffix.lower(),
        }
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".webm"}:
            record["media_probe"] = _probe_video(path)
        elif path.is_file() and path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            record["media_probe"] = _probe_image(path)
        records.append(record)
    return records


def technically_valid_generation(record: dict[str, Any]) -> bool:
    tool_return = record.get("tool_return") or {}
    artifacts = record.get("artifact_records") or []
    if tool_return.get("status") != "success" or not artifacts:
        return False
    for artifact in artifacts:
        if (
            not artifact.get("exists")
            or not artifact.get("sha256")
            or artifact.get("bytes", 0) <= 0
        ):
            return False
        if artifact.get("suffix") in {".mp4", ".mov", ".webm"}:
            probe = artifact.get("media_probe") or {}
            if not probe.get("valid") or int(probe.get("frame_count") or 0) < 2:
                return False
        elif artifact.get("suffix") in {".png", ".jpg", ".jpeg", ".webp"}:
            probe = artifact.get("media_probe") or {}
            if (
                not probe.get("valid")
                or int(probe.get("width") or 0) < 1
                or int(probe.get("height") or 0) < 1
            ):
                return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            return {
                "valid": width > 0 and height > 0,
                "frame_count": 1,
                "width": width,
                "height": height,
                "format": image.format,
                "error": None,
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "valid": False,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "format": None,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
        payload = json.loads(proc.stdout) if proc.returncode == 0 else {}
        stream = (payload.get("streams") or [{}])[0]
        frame_count = int(stream.get("nb_read_frames") or 0)
        return {
            "valid": proc.returncode == 0 and frame_count > 0,
            "frame_count": frame_count,
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "duration": stream.get("duration"),
            "returncode": proc.returncode,
            "error": proc.stderr[-1000:] if proc.returncode else None,
        }
    except FileNotFoundError:
        try:
            import av

            with av.open(str(path)) as container:
                stream = next(
                    stream for stream in container.streams if stream.type == "video"
                )
                width = int(stream.codec_context.width or 0)
                height = int(stream.codec_context.height or 0)
                fps = str(stream.average_rate or "")
                duration = (
                    float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None
                    else None
                )
                frame_count = sum(1 for _ in container.decode(stream))
            return {
                "valid": bool(width and height and frame_count),
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "avg_frame_rate": fps,
                "duration": duration,
                "returncode": 0,
                "error": None,
            }
        except Exception as exc:
            return {
                "valid": False,
                "frame_count": 0,
                "error": f"{type(exc).__name__}:{exc}",
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "valid": False,
            "frame_count": 0,
            "error": f"{type(exc).__name__}:{exc}",
        }
