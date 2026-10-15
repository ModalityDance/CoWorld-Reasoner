"""Common request media utilities shared by active VWM backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import ToolRequest


def stable_seed(identifier: str) -> int:
    """Return a deterministic bounded seed for a logical request identity."""
    value = 2166136261
    for char in str(identifier):
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return 1000 + (value % 900000)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def artifact_refs_from_request(request: ToolRequest) -> list[Path]:
    """Resolve unique image inputs declared by a normalized tool request."""
    controls = request.controls or {}
    refs = [str(item) for item in _as_list(controls.get("input_artifact_refs"))]
    refs.extend(str(item) for item in _as_list(controls.get("artifact_refs")))
    evidence_artifacts = controls.get("evidence_artifact_refs") or {}
    if isinstance(evidence_artifacts, dict):
        for evidence_id in request.input_evidence_ids:
            refs.extend(str(item) for item in _as_list(evidence_artifacts.get(evidence_id)))
    resolved: list[Path] = []
    seen: set[str] = set()
    for raw in refs:
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        normalized = str(path.resolve(strict=False))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and normalized not in seen:
            seen.add(normalized)
            resolved.append(path)
    return resolved
