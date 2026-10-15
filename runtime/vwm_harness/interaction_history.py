"""Carry the agent's interaction context across the four stages."""
from __future__ import annotations

import json
from pathlib import Path


_SKILL_UPDATE_PREFIX = (
    "Updated guidance for your already loaded skill follows. Use this version "
    "instead of its earlier guidance for the remaining reasoning. Earlier "
    "observations and actions are unchanged; this is guidance, not new visual evidence.\n"
)


def refresh_loaded_skill(history: list[dict], skill_id: str | None) -> list[dict]:
    """Append revised guidance to the continuing interaction."""
    if not skill_id:
        return history
    from evolved_harness.catalog import skill_package

    current = skill_package(skill_id).load_body()
    delivered = None
    skill_calls = set()
    for item in history:
        if item.get("type") == "function_call" and item.get("name") == "load_vwm_task_skill":
            skill_calls.add(item.get("call_id"))
        if item.get("type") == "function_call_output" and item.get("call_id") in skill_calls:
            output = item.get("output")
            if isinstance(output, str):
                try:
                    payload = json.loads(output)
                except ValueError:
                    continue
                if isinstance(payload, dict) and payload.get("status") == "loaded" and payload.get("skill_id") == skill_id:
                    delivered = payload.get("skill", payload.get("skill_body"))
        if item.get("role") == "user":
            content = item.get("content")
            texts = [content] if isinstance(content, str) else [
                part.get("text", "") for part in content or [] if isinstance(part, dict)
            ]
            for text in texts:
                if text.startswith(_SKILL_UPDATE_PREFIX):
                    payload = json.loads(text[len(_SKILL_UPDATE_PREFIX):])
                    if payload.get("skill_id") == skill_id:
                        delivered = payload["skill"]
    if delivered is None:
        raise ValueError(f"loaded_skill_guidance_missing_from_history:{skill_id}")
    if delivered == current:
        return history
    return history + [{"role": "user", "content": _SKILL_UPDATE_PREFIX + json.dumps(
        {"skill_id": skill_id, "skill": current}, ensure_ascii=False,
    )}]


async def stop_after_generation(wrapper, tool_results):
    """End P2 when generation is complete and no retake remains available."""
    from agents.agent import ToolsToFinalOutputResult
    from evolved_harness.evidence_tools import _retake_enabled

    records = getattr(wrapper.context, 'generation_records', [])
    if not records or _retake_enabled(wrapper, None):
        return ToolsToFinalOutputResult(is_final_output=False)
    latest = records[-1] if isinstance(records[-1], dict) else {}
    generation = latest.get('generation') if isinstance(latest.get('generation'), dict) else {}
    tool_return = generation.get('tool_return') if isinstance(generation.get('tool_return'), dict) else {}
    permission = tool_return.get('permission_decision') if isinstance(tool_return.get('permission_decision'), dict) else {}
    status = str(tool_return.get('status') or '').lower()
    executed = permission.get('executed') is True
    technically_valid = generation.get('technically_valid') is True
    sampled_frames = latest.get('sampled_frames')
    has_frames = isinstance(sampled_frames, list) and bool(sampled_frames)
    if executed and status == 'success' and technically_valid and has_frames:
        state = 'generation_completed'
    elif not executed and status.startswith('dry_run_'):
        state = 'generation_not_executed'
    else:
        state = 'generation_failed'
    return ToolsToFinalOutputResult(is_final_output=True, final_output={
        'stage': 'P2', 'state': state,
        'generation_count': len(records), 'answer_selected': False,
    })


def continue_capability_assessment(root: Path, record: dict, question: str, *, retakes: int = 0) -> list[dict]:
    """Continue P1's source images, skill calls, and decision into P2."""
    path = root / 'sample_runs' / f"{record['index']:03d}" / 'INTERACTION_HISTORY.json'
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError('empty_sdk_stage_history')
    if record.get('p1_output') is None or record.get('error'):
        raise ValueError('P1_has_no_completed_decision')
    instruction = (
        'Continue this trajectory into P2. Your preceding output was a P1 '
        'capability judgment, not the final task answer. VWM execution is now '
        'available. Reuse your existing observations and loaded skill; decide '
        'autonomously whether and how to request the missing visual evidence. '
        'This stage stops at generation, before interpreting generated images '
        'or calling a verifier. The original task remains:\n'
    )
    if retakes:
        instruction = instruction.replace('This stage stops at generation, before interpreting generated images ',
            'After generation, check media quality and optionally retake the same operation once for a concrete defect. '
            'Finish with the request decision, before interpreting generated images ')
    return history + [{
        'role': 'user',
        'content': [{
            'type': 'input_text',
            'text': instruction + question,
        }],
    }]
