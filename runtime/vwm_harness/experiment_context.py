"""Readable execution evidence and the supervisor's persistent workspace."""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
from util.harness_state import _source_paths
from runtime.vwm_harness.stage_assessment import available_feedback
from runtime.vwm_harness.supervisor_feedback import preceding_outputs, verifier_observations
from runtime.vwm_harness.interaction_history import _SKILL_UPDATE_PREFIX
from runtime.vwm_harness.project_context import component_for_path, project_guide

STAGES = ("P1", "P2", "P3", "P4")

def _stage_output(row: dict[str, Any]) -> Any:
    output = row.get("stage_output")
    if output:
        return output
    return (row.get("p1_output") or {}) if row.get("stage", "P1") == "P1" else {}


def _records(root: Path) -> dict[str, dict[str, Any]]:
    jsonl = root / "records.jsonl"
    if jsonl.exists():
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "sample_records").glob("*.json"))]
    return {str(row["sample_id"]): row for row in rows}


def _public_rows(candidate: Path) -> dict[str, dict[str, Any]]:
    config = json.loads((candidate / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    path = Path(config["input_jsonl"])
    input_root = Path(config["input_root"])
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["image_paths"] = [
            str((image if image.is_absolute() else input_root / image).resolve())
            for image in map(Path, row.get("image_paths") or [])
        ]
        rows[str(row["sample_id"])] = row
    return rows


def _compact(value: Any, limit: int = 1800) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...[truncated]"
    if isinstance(value, list):
        return [_compact(v, limit) for v in value[:12]]
    if isinstance(value, dict):
        return {str(k): _compact(v, limit) for k, v in list(value.items())[:30]}
    return value


def _execution_feedback(row: dict[str, Any]) -> dict[str, Any]:
    """Summarize execution failures for supervisor feedback."""
    events = row.get("events") or []
    failures = []
    for event in events:
        if not isinstance(event, dict):
            continue
        payload = event.get("payload") or {}
        status = str(payload.get("status") or "").lower()
        event_type = str(event.get("event_type") or "").lower()
        if status in {"error", "failed", "failure", "invalid_arguments", "invalid_request", "invalid_dispatch", "unavailable", "budget_exhausted"} or "error" in event_type or "fail" in event_type:
            failures.append({
                "event_index": event.get("event_index"),
                "event_type": event.get("event_type"),
                "payload": _compact(payload, 1200),
            })
    feedback = {
        "record_error": _compact(row.get("error"), 2000),
        "failure_events": failures[:16],
        "event_count": len(events),
    }
    if row.get("provider_errors"):
        stage_output = row.get("stage_output") or (
            row.get("p1_output") if row.get("stage", "P1") == "P1" else None
        )
        feedback["provider_retries"] = {
            "stage_completed": not row.get("error") and bool(stage_output),
            "errors": _compact(row["provider_errors"], 2000),
        }
    outcomes = []
    generations = [] if row.get("stage") == "P1" else row.get("generation_records") or []
    for index, record in enumerate(generations):
        if not isinstance(record, dict):
            continue
        generation = record.get("generation") if isinstance(record.get("generation"), dict) else {}
        tool_return = generation.get("tool_return") if isinstance(generation.get("tool_return"), dict) else {}
        permission = tool_return.get("permission_decision") if isinstance(tool_return.get("permission_decision"), dict) else {}
        outcomes.append({
            "generation_index": record.get("generation_index", index),
            "status": tool_return.get("status"),
            "executed": permission.get("executed"),
            "technically_valid": generation.get("technically_valid"),
            "failure_modes": _compact(generation.get("failure_modes") or tool_return.get("failure_modes") or [], 1200),
            "sampled_frame_count": len(record.get("sampled_frames") or []),
        })
    if outcomes:
        feedback["generation_outcomes"] = outcomes
    return {key: value for key, value in feedback.items() if value not in (None, [], "")}


def _summary(
    row: dict[str, Any], previous: dict[str, Any] | None = None,
    *, public: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage = row.get("stage") or "P1"
    stage_output = _stage_output(row)
    dispatches = row.get("accepted_dispatches") or []
    generations = row.get("generation_records") or []
    final = stage_output if row.get("stage") == "P4" else None
    result = {
        "sample_id": row.get("sample_id"),
        "benchmark": row.get("benchmark"),
        "task_type": row.get("task_type"),
        "stage": stage,
        "status": row.get("status"),
        "error": row.get("error"),
        "execution_feedback": _execution_feedback(row),
        "loaded_skill_id": row.get("loaded_skill_id") or (row.get("p1_output") or {}).get("selected_task_skill"),
        "skill_guidance_refreshed": row.get("skill_guidance_refreshed", False),
        "stage_output": _compact(stage_output, 1500),
        "available_feedback": sorted(available_feedback(row, stage)),
    }
    if stage != "P1":
        result.update(vwm_call_count=len(dispatches), generation_count=len(generations),
                      p1_output=_compact(row.get("p1_output") or {}, 900))
    if stage in {"P3", "P4"}:
        result["verifier_count"] = len(row.get("verifier_records") or [])
        if row.get("verifier_records"):
            result["verification"] = _compact(row["verifier_records"], 1000)
    if stage == "P4" and row.get("scoring"):
        result["scoring"] = row["scoring"]
    if public is not None:
        # Copy the public task fields verbatim; they define the task context.
        result["original_question"] = public["question"]
        result["original_images"] = [
            {"source_image_index": index,
             "label": f"{row['sample_id']}:original_{index:02d}", "path": path}
            for index, path in enumerate(public.get("image_paths") or [])
        ]
    if stage != "P1" and dispatches:
        result["requests"] = _compact([
            {k: d.get(k) for k in (
                "operation", "tool_interface", "source_image_indices", "source_order_mode",
                "candidate_conditioned", "reference_image_index", "candidate_image_index",
                "candidate_relation", "generation_instruction", "preserve",
                "tool_arguments", "operation_controls", "control_provenance",
            ) if k in d}
            for d in dispatches
        ], 1500)
    if previous is not None:
        result["previous_candidate_output"] = _compact(_stage_output(previous), 1100)
        if stage == "P4" and previous.get("scoring"):
            result["previous_scoring"] = _compact(previous["scoring"], 300)
    if final is not None:
        result["final"] = _compact(final, 1500)
    return result


def _media_roots(candidate: Path) -> set[Path]:
    """Locate the candidate and its configured preceding-stage directories."""
    roots: set[Path] = set()
    current = candidate.resolve()
    expected_stage = None
    for _ in STAGES:
        if current in roots:
            raise ValueError("candidate_media_upstream_cycle")
        config = json.loads((current / "RUN_CONFIG.json").read_text(encoding="utf-8"))
        stage = config.get("stage")
        if stage not in STAGES or (expected_stage is not None and stage != expected_stage):
            raise ValueError("candidate_media_upstream_stage_mismatch")
        roots.add(current)
        upstream = config.get("upstream")
        if not upstream:
            return roots
        expected_stage = f"P{int(stage[1]) - 1}"
        parent = Path(upstream)
        current = (parent if parent.is_absolute() else current / parent).resolve()
    raise ValueError("candidate_media_upstream_depth_exceeded")


def _candidate_images(
    candidate: Path, records: dict[str, dict[str, Any]], public: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    roots = _media_roots(candidate)
    images: dict[str, list[dict[str, str]]] = {}
    for sid, row in records.items():
        entries = []
        for index, raw in enumerate((public.get(sid) or {}).get("image_paths") or []):
            path = Path(raw).resolve()
            if path.is_file():
                entries.append({"label": f"{sid}:original_{index:02d}", "path": str(path)})
        for generation in row.get("generation_records") or []:
            frames = sorted(generation.get("sampled_frames") or [], key=lambda f: f.get("frame_index", 0))
            for index, frame in enumerate(frames):
                raw = frame.get("path")
                if not raw:
                    continue
                path = Path(raw)
                path = (path if path.is_absolute() else candidate / path).resolve()
                if path.is_file() and any(root in path.parents for root in roots):
                    label = f"{sid}:generated_{generation.get('generation_index', 0)}:frame_{frame.get('frame_index', index)}"
                    entries.append({"label": label, "path": str(path)})
        images[sid] = entries
    return images


def prepare_candidate_context(output: Path, candidates: list[dict[str, Any]]):
    """Prepare the supervisor workspace and evidence for each candidate."""
    baselines = {}
    git_environment = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        if Path(cid).name != cid or cid in {".", ".."}:
            raise ValueError("invalid_candidate_id")
        directory = output / "workspace" / "project"
        source = Path(candidate["harness_source_dir"])
        fresh = not directory.exists()
        if fresh:
            shutil.copytree(source, directory, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            # Keep the project guide inside this Git root for native discovery.
            (directory / "AGENTS.md").write_text(project_guide(directory), encoding="utf-8")
        baselines[cid] = {path.relative_to(source).as_posix(): path.read_bytes()
                          for path in _source_paths(source)}
        for arguments in ((["init", "-q"], ["add", "--", "."],
                          ["-c", "user.name=CoWorld", "-c", "user.email=supervisor@localhost",
                           "-c", "core.hooksPath=" + os.devnull,
                           "commit", "--no-gpg-sign", "-qm", "Initial project"]) if fresh else ()):
            subprocess.run(["git", *arguments], cwd=directory, env=git_environment,
                           check=True, capture_output=True, text=True)
        candidate["workspace"] = str(directory)
        if candidate.get("record_path"):
            records = Path(candidate["record_path"])
            evidence_root = output / "evidence" / cid
            readable_records = evidence_root / "records.jsonl"
            _render_evidence(records, readable_records, jsonl=True)
            candidate["raw_record_path"] = str(records)
            candidate["record_path"] = str(readable_records)
            histories = {}
            for line in records.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                raw = records.parent / "sample_runs" / f"{row['index']:03d}" / "INTERACTION_HISTORY.json"
                if raw.is_file():
                    readable = evidence_root / str(row["sample_id"]) / "history.json"
                    _render_evidence(raw, readable)
                    histories[row["sample_id"]] = str(readable)
            candidate["interaction_histories"] = histories
            population_path = evidence_root / "population.jsonl"
            public = _public_rows(records.parent)
            rows = _records(records.parent)
            ancestors = preceding_outputs(records.parent, list(rows))
            media = _candidate_images(records.parent, rows, public)
            population_path.write_text("".join(json.dumps(dict(
                _summary(row, public=public.get(sid)),
                preceding_stage_outputs=ancestors[sid],
                interaction_history_path=histories.get(sid), images=media[sid],
            ), ensure_ascii=False) + "\n" for sid, row in rows.items()), encoding="utf-8")
            candidate["population_path"] = str(population_path)
    return baselines


def case_guidance(candidate: dict, row: dict) -> dict:
    """Relate delivered skill versions to source and editable entry points."""
    snapshot = Path(candidate["harness_source_dir"])
    working = Path(candidate["workspace"])
    history_path = candidate.get("interaction_histories", {}).get(row["sample_id"])
    history = json.loads(Path(history_path).read_text(encoding="utf-8")) if history_path else []

    def source_reference(relative):
        executed, current = snapshot / relative, working / relative
        return {"path": relative,
                "executed_source": str(executed) if executed.is_file() else None,
                "working_source": str(current) if current.is_file() else None,
                "working_differs_from_snapshot": (
                    executed.read_bytes() != current.read_bytes()
                    if executed.is_file() and current.is_file() else None)}

    skills = {path.parent.name: path.relative_to(snapshot).as_posix()
              for path in (snapshot / "evolved_harness/task_skills").glob("*/SKILL.md")}
    calls, deliveries = set(), []

    def add_delivery(payload, index, delivery):
        if not isinstance(payload, dict):
            return
        body = payload.get("skill", payload.get("skill_body"))
        if not isinstance(body, str):
            return
        skill_id = payload.get("skill_id")
        relative = skills.get(skill_id)
        reference = source_reference(relative) if relative else None
        deliveries.append({"history_item_index": index, "delivery": delivery,
            "skill_id": skill_id, "body": body, "source": reference,
            "matches_executed_snapshot": (
                body == (snapshot / relative).read_text(encoding="utf-8") if relative else None),
            "matches_working_file": (
                body == (working / relative).read_text(encoding="utf-8")
                if relative and (working / relative).is_file() else None)})

    for index, item in enumerate(history):
        if item.get("type") == "function_call" and item.get("name") == "load_vwm_task_skill":
            calls.add(item.get("call_id"))
        if item.get("type") == "function_call_output" and item.get("call_id") in calls:
            payload = item.get("output")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    continue
            if isinstance(payload, dict) and payload.get("status") == "loaded":
                add_delivery(payload, index, "skill_load")
        if item.get("role") == "user":
            content = item.get("content")
            texts = [content] if isinstance(content, str) else [
                part.get("text", "") for part in content or [] if isinstance(part, dict)]
            for text in texts:
                if text.startswith(_SKILL_UPDATE_PREFIX):
                    add_delivery(json.loads(text[len(_SKILL_UPDATE_PREFIX):]), index, "guidance_refresh")

    stage = row.get("stage", "P1")
    stage_modules = {"P1": "p1_capability_assessment.py", "P2": "p2_vwm_request.py",
                     "P3": "p3_video_interpretation.py", "P4": "p4_evidence_integration.py"}
    paths = ["evolved_harness/" + stage_modules[stage], "runtime/execution.py",
             "runtime/agent_runtime.py", "runtime/agents_sdk_input.py", "runtime/media_context.py",
             "runtime/vwm_harness/interaction_history.py", "evolved_harness/catalog.py"]
    if stage in {"P1", "P2"}:
        paths.append("evolved_harness/visual_tools.py")
    if stage in {"P3", "P4"}:
        paths.extend(["evolved_harness/evidence_reasoning.py", "evolved_harness/evidence_tools.py",
                      "evolved_harness/verifier_observations.py", "evolved_harness/verifier_profiles.py",
                      "runtime/video_verifier.py"])
    if row.get("accepted_dispatches"):
        paths.extend(["evolved_harness/h3_prompt.py", "evolved_harness/fal_h3_backend_adapter.py",
                      "runtime/visual_dispatch.py", "runtime/generation_execution.py",
                      "runtime/request_media.py"])
    generation_steps = []
    for index, record in enumerate(row.get("generation_records") or []):
        generation = record.get("generation") if isinstance(record.get("generation"), dict) else {}
        returned = generation.get("tool_return") if isinstance(generation.get("tool_return"), dict) else {}
        generation_steps.append({
            "generation_index": record.get("generation_index", index),
            "request_index": record.get("request_index", index),
            "status": returned.get("status"),
            "technically_valid": generation.get("technically_valid"),
            "failure_modes": generation.get("failure_modes") or returned.get("failure_modes") or [],
            "frame_count": len(record.get("sampled_frames") or []),
            "frame_order": [frame.get("frame_index") for frame in record.get("sampled_frames") or []],
        })
    verifier_steps = []
    for verifier in row.get("verifier_records") or []:
        request = verifier.get("request") if isinstance(verifier.get("request"), dict) else {}
        controls = request.get("controls") if isinstance(request.get("controls"), dict) else {}
        manifest = controls.get("attachment_manifest") or []
        verifier_steps.append({
            "profile_id": verifier.get("profile_id"),
            "status": (verifier.get("result") or {}).get("status"),
            "attachment_sources": sorted({item.get("source_type") for item in manifest}),
            "attachment_count": len(manifest),
            "generated_only": bool(manifest) and all(
                item.get("source_type") == "predicted_vwm_frame" for item in manifest),
        })
    execution_chain = {
        "stage": stage,
        "history": {"path": str(history_path) if history_path else None,
                    "item_count": len(history), "skill_refresh_count": sum(
                        1 for delivery in deliveries if delivery["delivery"] == "guidance_refresh")},
        "skill": {"requested": row.get("loaded_skill_id"),
                   "delivery_count": len(deliveries),
                   "delivery_indices": [delivery["history_item_index"] for delivery in deliveries]},
        "tool_requests": [{"index": index, "operation": dispatch.get("operation"),
                            "tool_interface": dispatch.get("tool_interface"),
                            "source_image_indices": dispatch.get("source_image_indices"),
                            "source_order_mode": dispatch.get("source_order_mode")}
                           for index, dispatch in enumerate(row.get("accepted_dispatches") or [])],
        "generation": generation_steps,
        "verifier": verifier_steps,
        "answer_or_stage_output_present": bool(row.get("stage_output") or row.get("p1_output")),
        "entry_point_components": sorted({component_for_path(path) for path in paths}),
    }
    return {"recorded_skill_id": row.get("loaded_skill_id"),
            "history_available": bool(history_path), "delivered_skill_versions": deliveries,
            "source_entry_points": [source_reference(path) for path in paths],
            "execution_chain": execution_chain,
            "interpretation": "Skill bodies above are copied from the executed interaction in delivery order. "
                "Source entry points locate the candidate snapshot and current working files. "
                "The history contains the actual calls, results and continuation messages."}


def case_evidence(candidates: list[dict], sample_id: str) -> list[dict]:
    """Bind the original task and ordered media to one or two executed candidates."""
    content = []
    shown = {}

    def text(value):
        content.append({"type": "inputText", "text": value if isinstance(value, str)
                        else json.dumps(value, ensure_ascii=False)})

    def picture(path, label):
        path = Path(path).resolve()
        text(label)
        if path in shown:
            text("Same image as " + shown[path])
            return
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif"}.get(path.suffix.lower())
        if mime is None:
            raise ValueError(f"unsupported_evidence_image:{path.suffix}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "inputImage", "imageUrl": f"data:{mime};base64,{encoded}"})
        shown[path] = label

    source_task = None
    for candidate in candidates:
        directory = Path(candidate["candidate_dir"])
        public = _public_rows(directory)[sample_id]
        task = {"question": public["question"], "image_paths": public.get("image_paths") or []}
        if source_task is None:
            source_task = task
            text({"sample_id": sample_id, "original_question": task["question"],
                  "original_image_count": len(task["image_paths"]),
                  "history_path": candidate["interaction_histories"].get(sample_id),
                  "readable_history_path": str(Path(candidate["interaction_histories"][sample_id]).with_suffix(".md"))})
            for index, path in enumerate(task["image_paths"]):
                picture(path, f"Original task-order image {index + 1}; source_image_index={index}.")
        elif task != source_task:
            raise ValueError("case_comparison_original_task_mismatch")
        records = _records(directory)
        row = records[sample_id]
        text({"candidate_id": candidate["candidate_id"], "stage": row["stage"],
              "history_path": candidate["interaction_histories"].get(sample_id),
              "readable_history_path": str(Path(candidate["interaction_histories"][sample_id]).with_suffix(".md")),
              "stage_output": _stage_output(row), "p1_output": row.get("p1_output"),
              "requests": row.get("accepted_dispatches") or [],
              "verifier_observations": verifier_observations(row),
              "guidance": case_guidance(candidate, row),
              "scoring": row.get("scoring"), "execution_feedback": _execution_feedback(row)})
        media = _candidate_images(directory, {sample_id: row}, {sample_id: public})[sample_id]
        for entry in media[len(task["image_paths"]):]:
            picture(entry["path"], candidate["candidate_id"] + ": " + entry["label"])
    text("Images are attached in original task order and generated-frame temporal order, labeled by candidate and generation. Generated frames are predictions. The linked history.md contains complete tool calls, results and loaded skill guidance.")
    return content


def render_history(history: list[dict]) -> str:
    """Render visible SDK messages and pair calls/results with their content."""
    names = {str(item.get("call_id") or item.get("id")): item.get("name")
             for item in history if item.get("type") == "function_call"}

    def readable(value, indent=0):
        if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        pad = "  " * indent
        if isinstance(value, dict):
            return "\n".join(pad + str(key) + ":\n" + readable(child, indent + 1)
                             for key, child in value.items())
        if isinstance(value, list):
            return "\n".join(pad + f"[{i}]\n" + readable(child, indent + 1)
                             for i, child in enumerate(value))
        return "\n".join(pad + line for line in str(value).splitlines()) if isinstance(value, str) else pad + json.dumps(value)
    parts = ["# Interaction history\n\nImage paths refer to media files; use native view_image to view them."]
    for index, item in enumerate(history):
        kind = item.get("type") or item.get("role") or "message"
        call = str(item.get("call_id") or "")
        name = item.get("name") or names.get(call) or ""
        parts.append(f"\n## Item {index}: {kind} {name}" + (f" (call_id={call})" if call else ""))
        parts.append(readable(item))
    return "\n\n".join(parts) + "\n"


def _render_evidence(source: Path, destination: Path, *, jsonl: bool = False) -> None:
    """Render the visible conversation and store inline media as local files."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    def render(value):
        if isinstance(value, list):
            return [render(item) for item in value]
        if isinstance(value, dict):
            return {key: render(item) if key != "encrypted_content" else "[opaque provider state]"
                    for key, item in value.items()}
        if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
            header, encoded = value.split(",", 1)
            data = base64.b64decode(encoded, validate=True)
            extension = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
                         "image/gif": ".gif"}.get(header[5:].split(";", 1)[0], ".img")
            media = destination.parent / (hashlib.sha256(data).hexdigest() + extension)
            if not media.exists():
                media.write_bytes(data)
            return "image file: " + str(media)
        if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            try:
                return json.dumps(render(json.loads(value)), ensure_ascii=False)
            except ValueError:
                pass
        return value

    text = source.read_text(encoding="utf-8")
    if jsonl:
        content = "".join(json.dumps(render(json.loads(line)), ensure_ascii=False) + "\n"
                          for line in text.splitlines() if line.strip())
    else:
        content = json.dumps(render(json.loads(text)), ensure_ascii=False, indent=2) + "\n"
    destination.write_text(content, encoding="utf-8")
    if not jsonl:
        history = json.loads(content)
        if isinstance(history, list):
            destination.with_suffix(".md").write_text(render_history(history), encoding="utf-8")
