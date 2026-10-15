"""Fal-hosted MiniMax H3 Max Turbo adapter for the normalized VWM boundary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

import fal_client
from PIL import Image

from runtime.contracts import ToolRequest, ToolReturn
from runtime.request_media import artifact_refs_from_request, stable_seed
from .h3_prompt import build_h3_prompt


FAL_H3_APPLICATION = "minimax/h3-max-turbo/image-to-video"
_FAL_INFLIGHT = BoundedSemaphore(
    max(1, int(os.environ.get("AGENTIC_FAL_H3_MAX_INFLIGHT", "48")))
)
_TRANSFER_SLOTS = BoundedSemaphore(
    max(1, int(os.environ.get("AGENTIC_FAL_H3_TRANSFER_CONCURRENCY", "2")))
)


@dataclass(frozen=True)
class FalH3Config:
    output_root: Path = Path("runs/generated_evidence/fal_h3")
    duration: int = 5
    resolution: str = "480P"
    prompt_expansion_mode: str = "balanced"
    enable_safety_checker: bool = True
    timeout_seconds: int = 3600
    poll_interval_seconds: float = 2.0
    execute: bool = False

    @classmethod
    def from_env(cls) -> "FalH3Config":
        return cls(
            output_root=Path(os.environ.get("AGENTIC_FAL_H3_OUTPUT_ROOT", str(cls.output_root))),
            duration=int(os.environ.get("AGENTIC_FAL_H3_DURATION", str(cls.duration))),
            resolution=os.environ.get("AGENTIC_FAL_H3_RESOLUTION", cls.resolution),
            prompt_expansion_mode=os.environ.get(
                "AGENTIC_FAL_H3_PROMPT_EXPANSION_MODE", cls.prompt_expansion_mode
            ),
            enable_safety_checker=os.environ.get(
                "AGENTIC_FAL_H3_SAFETY_CHECKER", "1"
            ) == "1",
            timeout_seconds=int(os.environ.get("AGENTIC_FAL_H3_TIMEOUT_SECONDS", str(cls.timeout_seconds))),
            poll_interval_seconds=float(os.environ.get("AGENTIC_FAL_H3_POLL_INTERVAL_SECONDS", "2")),
            execute=os.environ.get("AGENTIC_FAL_H3_EXECUTE", "0") == "1",
        )


def _load_key() -> None:
    if os.environ.get("FAL_KEY"):
        return
    path = Path(os.environ.get("AGENTIC_FAL_ENV_FILE", ".env.fal.local"))
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.startswith("FAL_KEY="):
                value = line.split("=", 1)[1].strip()
                if value:
                    os.environ["FAL_KEY"] = value
                    return
    raise RuntimeError("fal_key_missing")


def _input_images(request: ToolRequest) -> list[Path]:
    refs = (request.controls or {}).get("input_artifact_refs") or artifact_refs_from_request(request)
    return [Path(str(ref)).resolve() for ref in refs if Path(str(ref)).is_file()]


def _fal_ready_image(path: Path, cache_root: Path) -> Path:
    """Prepare an uncropped RGB image within Fal's accepted aspect-ratio range."""
    with Image.open(path) as image:
        width, height = image.size
        ratio = width / height
        target_width = max(width, 256)
        target_height = max(height, 256)
        if ratio > 2.5:
            target_height = max(target_height, (target_width * 2 + 4) // 5)
        elif ratio < 0.4:
            target_width = max(target_width, (target_height * 2 + 4) // 5)
        if width >= 256 and height >= 256 and 0.4 <= ratio <= 2.5:
            return path
        cache_root.mkdir(parents=True, exist_ok=True)
        scale = min(target_width / width, target_height / height)
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas_size = (max(256, target_width), max(256, target_height))
        digest = hashlib.sha256(
            path.read_bytes() + f":{resized}:{canvas_size}".encode()
        ).hexdigest()
        target = cache_root / f"{digest}{path.suffix.lower() or '.png'}"
        if not target.is_file():
            converted = image.convert("RGB").resize(resized, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", canvas_size, (255, 255, 255))
            canvas.paste(converted, ((canvas_size[0] - resized[0]) // 2,
                                     (canvas_size[1] - resized[1]) // 2))
            canvas.save(target, quality=95)
        return target


def _upload_cached(path: Path, cache_root: Path) -> str:
    path = _fal_ready_image(path, cache_root)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    record = cache_root / f"{digest}.json"
    if record.is_file():
        try:
            value = json.loads(record.read_text(encoding="utf-8")).get("url")
            if value:
                return str(value)
        except (OSError, ValueError):
            pass
    last_error = None
    for attempt in range(2):
        try:
            # Use the CDN repository supported by this image-upload endpoint.
            url = fal_client.upload_file(str(path), repository="fal_v3", fallback_repository=[])
            break
        except Exception as exc:
            last_error = exc
            if attempt == 1:
                raise
            time.sleep(1.0)
    else:
        raise RuntimeError(f"fal_upload_failed:{last_error}")
    record.write_text(json.dumps({"url": url}) + "\n",
                      encoding="utf-8")
    return str(url)


def _media_probe(path: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=False, timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"fal_h3_ffprobe_failed:{proc.stderr[-500:]}")
        stream = (json.loads(proc.stdout).get("streams") or [{}])[0]
        frame_count = int(stream.get("nb_frames") or 0)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        fps = str(stream.get("avg_frame_rate") or "")
        duration = stream.get("duration")
    except FileNotFoundError:
        try:
            import av
            with av.open(str(path)) as container:
                stream_obj = next(stream for stream in container.streams if stream.type == "video")
                width = int(stream_obj.codec_context.width or 0)
                height = int(stream_obj.codec_context.height or 0)
                fps = str(stream_obj.average_rate or "")
                duration = (
                    float(stream_obj.duration * stream_obj.time_base)
                    if stream_obj.duration is not None and stream_obj.time_base is not None
                    else None
                )
                frame_count = sum(1 for _ in container.decode(stream_obj))
        except Exception as exc:
            raise RuntimeError(f"fal_h3_video_probe_failed:{type(exc).__name__}:{exc}") from exc
    return {
        "valid": bool(width and height and frame_count),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration,
    }


class FalAPIError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"fal_api_error:{status_code}:{detail}")


def _fal_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Key {os.environ['FAL_KEY']}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read(8192).decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
            detail = payload.get("detail") or payload.get("message") or payload.get("error") or exc.reason
        except (ValueError, AttributeError):
            detail = body or exc.reason
        raise FalAPIError(exc.code, str(detail)[:2000]) from exc


def _download(url: str, destination: Path, timeout: int) -> dict[str, Any]:
    partial = destination.with_suffix(destination.suffix + ".partial")
    waiting = time.monotonic()
    with _TRANSFER_SLOTS:
        acquired = time.monotonic()
        try:
            subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error",
                 "--connect-timeout", "30", "--max-time", str(int(timeout)),
                 "-o", str(partial), url],
                check=True, capture_output=True, timeout=timeout + 60,
            )
        except FileNotFoundError:
            # Use the standard-library downloader when curl is unavailable,
            # retaining the same transfer timeout.
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as sink:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    sink.write(chunk)
        size = partial.stat().st_size
        if size <= 0:
            raise RuntimeError("fal_h3_empty_local_download")
        partial.replace(destination)
        return {
            "mode": "local_download", "bytes": size,
            "wait_seconds": round(acquired - waiting, 3),
            "transfer_seconds": round(time.monotonic() - acquired, 3),
        }


def fal_h3_tool_function(
    request: ToolRequest, config: FalH3Config | None = None
) -> ToolReturn:
    config = config or FalH3Config.from_env()
    started = time.time()
    images = _input_images(request)
    trace: dict[str, Any] = {
        "backend": "fal_h3_max_turbo",
        "application": FAL_H3_APPLICATION,
        "input_images": [str(path) for path in images],
        "duration": config.duration,
        "resolution": config.resolution,
        "prompt_expansion_mode": config.prompt_expansion_mode,
        "enable_safety_checker": config.enable_safety_checker,
        "predicted_not_oracle": True,
        "new_generation_requests": 0,
    }
    if len(images) not in (1, 2):
        return _failure(request, "fal_h3_requires_one_or_two_images", trace, started)
    if not all(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for path in images):
        return _failure(request, "fal_h3_unsupported_image_format", trace, started)
    if config.duration < 5 or config.duration > 15:
        return _failure(request, "fal_h3_duration_must_be_5_to_15", trace, started)
    if config.resolution not in {"480P", "768P", "1080P"}:
        return _failure(request, "fal_h3_invalid_resolution", trace, started)
    if not config.execute:
        return _failure(request, "dry_run_fal_h3_preflight_ready", trace, started)

    try:
        _load_key()
        prompt = build_h3_prompt(request)
        output = (config.output_root / (request.request_id or "fal_h3")).resolve()
        output.mkdir(parents=True, exist_ok=True)
        prior = {}
        prior_file = output / "fal_request.json"
        if prior_file.is_file():
            try:
                prior = json.loads(prior_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prior = {}
        arguments: dict[str, Any] = {
            "prompt": prompt,
            "duration": config.duration,
            "resolution": config.resolution,
            "seed": stable_seed(request.request_id or "fal_h3"),
            "enable_safety_checker": config.enable_safety_checker,
            "sync_mode": False,
            "prompt_expansion_mode": config.prompt_expansion_mode,
        }
        from runtime.durable_agent import atomic_json, digest
        fingerprint = digest({'arguments': arguments, 'images': [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in images]})
        intent_file = output / 'fal_submit_intent.json'
        if intent_file.exists():
            intent = json.loads(intent_file.read_text())
            if intent.get('fingerprint') != fingerprint:
                raise RuntimeError('fal_resume_request_fingerprint_mismatch')
            if not prior.get('request_id') and intent.get('status') != 'rejected':
                raise RuntimeError('fal_submitted_unknown_do_not_resubmit')
        if (prior.get('terminal_error') or {}).get('retry_after_account_recovery'):
            # A completed billing rejection produced no video. Preserve its
            # receipt, then allow the unchanged request to be submitted again.
            atomic_json(output / 'failed_requests' / (str(prior['request_id']) + '.json'), prior)
            atomic_json(intent_file, {'fingerprint': fingerprint, 'status': 'rejected'})
            atomic_json(prior_file, {})
            prior = {}
        if not prior.get("request_id"):
            arguments["image_url"] = _upload_cached(images[0], config.output_root / "_upload_cache")
            if len(images) == 2:
                arguments["end_image_url"] = _upload_cached(images[1], config.output_root / "_upload_cache")

        with _FAL_INFLIGHT:
            if prior.get("request_id"):
                request_id = str(prior["request_id"])
                status_url = str(prior["status_url"])
                response_url = str(prior["response_url"])
                trace["resumed_request_id"] = request_id
            else:
                atomic_json(intent_file, {'fingerprint': fingerprint,
                                         'status': 'submitting', 'time': time.time()})
                try:
                    handle = fal_client.submit(FAL_H3_APPLICATION, arguments)
                except Exception as exc:
                    # An explicit client-side rejection did not enqueue a job.
                    # Keep that evidence so a retry can submit after recovery.
                    # Timeouts and server failures retain an ambiguous state;
                    # submit again only after a receipt or explicit rejection.
                    code = getattr(exc, 'status_code', None)
                    if code in {400, 401, 402, 403, 404, 422, 429}:
                        atomic_json(intent_file, {'fingerprint': fingerprint,
                            'status': 'rejected', 'status_code': code,
                            'error': f'{type(exc).__name__}:{exc}', 'time': time.time()})
                    raise
                trace["new_generation_requests"] = 1
                request_id, status_url, response_url = handle.request_id, handle.status_url, handle.response_url
            receipt = {
                "backend": "fal_h3_max_turbo",
                "application": FAL_H3_APPLICATION,
                "request_id": request_id,
                "status_url": status_url,
                "response_url": response_url,
                "submitted_at": time.time(),
                "input_image_count": len(images),
            }
            if not prior.get("request_id"):
                atomic_json(output / 'fal_request.json', receipt)
            trace.update({
                "request_id": request_id,
                "status_url": status_url,
                "response_url": response_url,
            })
            deadline = time.monotonic() + config.timeout_seconds
            status = None
            while time.monotonic() < deadline:
                status = handle.status() if not prior.get("request_id") else _fal_json(status_url)
                status_name = getattr(status, "status", None)
                if status_name is None and isinstance(status, dict):
                    status_name = status.get("status")
                trace["last_status"] = str(status_name or status)
                if (isinstance(status, fal_client.Completed) or
                        str(status_name or "").lower() == "completed"):
                    break
                time.sleep(max(0.2, config.poll_interval_seconds))
            else:
                raise TimeoutError(f"fal_h3_timeout:{request_id}")

            try:
                result = _fal_json(response_url)
            except FalAPIError as exc:
                if exc.status_code == 402 or (exc.status_code == 403 and 'TOP_UP' in exc.detail):
                    receipt['terminal_error'] = {
                        'status_code': exc.status_code, 'detail': exc.detail,
                        'retry_after_account_recovery': True,
                    }
                    atomic_json(prior_file, receipt)
                raise
            atomic_json(output / "fal_response.json", result)
        video = result.get("video") if isinstance(result, dict) else None
        video_url = video.get("url") if isinstance(video, dict) else None
        if not video_url:
            raise RuntimeError("fal_h3_result_missing_video_url")
        output_path = output / "video.mp4"
        download = _download(
            str(video_url), output_path, min(config.timeout_seconds, 600),
        )
        expected_size = video.get("file_size")
        if expected_size is not None and output_path.stat().st_size != int(expected_size):
            raise RuntimeError("fal_h3_provider_file_size_mismatch")
        probe = _media_probe(output_path)
        if not probe.get("valid"):
            raise RuntimeError("fal_h3_invalid_downloaded_video")
        trace.update({
            "video_url": str(video_url),
            "output_path": str(output_path),
            "media_probe": probe,
            "expanded_prompt": result.get("expanded_prompt"),
            "timings": result.get("timings"),
            "download": download,
        })
        evidence_id = f"fal_h3_video::{request.request_id or request_id}"
        manifest = {
            "schema_version": "fal_h3_predicted_evidence_v1",
            "backend": "fal_h3_max_turbo",
            "application": FAL_H3_APPLICATION,
            "artifact_refs": [str(output_path)],
            "input_evidence_ids": list(request.input_evidence_ids),
            "predicted_not_oracle": True,
            "request_id": request_id,
            "media_probe": probe,
            "expanded_prompt": result.get("expanded_prompt"),
        }
        return ToolReturn(
            tool_return_id=f"fal_h3_return::{request.request_id or request_id}",
            tool_interface=request.tool_interface,
            status="success",
            input_evidence_ids=list(request.input_evidence_ids),
            output_evidence_ids=[evidence_id],
            artifacts={"artifact_refs": [str(output_path)], "video_paths": [str(output_path)], "evidence_manifest": manifest, "media_probe": probe},
            provenance_label="fal_h3_max_turbo_predicted_video_evidence",
            quality={"overall_usability": "requires_verification", "backend": "fal_h3_max_turbo"},
            cost={"runtime_sec": round(time.time() - started, 3), "backend": "fal_h3_max_turbo",
                  "generation_calls": trace["new_generation_requests"]},
            permission_decision={"executed": True, "policy": "AGENTIC_FAL_H3_EXECUTE=1"},
            execution_trace=trace,
            raw_backend_metadata={"provider": "fal", "video_url": str(video_url)},
        )
    except Exception as exc:
        trace["error"] = f"{type(exc).__name__}:{exc}"
        return _failure(request, "failed_fal_h3_request", trace, started)


def _failure(request: ToolRequest, status: str, trace: dict[str, Any], started: float) -> ToolReturn:
    return ToolReturn(
        tool_return_id=f"fal_h3_return::{request.request_id or int(started)}",
        tool_interface=request.tool_interface,
        status=status,
        input_evidence_ids=list(request.input_evidence_ids),
        failure_modes=[str(trace.get("error") or status)],
        cost={"runtime_sec": round(time.time() - started, 3), "backend": "fal_h3_max_turbo",
              "generation_calls": trace.get("new_generation_requests", 0)},
        permission_decision={"executed": False, "policy": "AGENTIC_FAL_H3_EXECUTE=1"},
        execution_trace=trace,
    )
