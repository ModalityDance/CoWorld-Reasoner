"""Typed requests and results for the active visual tools."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ToolRequest:
    tool_interface: str
    input_evidence_ids: list[str]
    request_id: str | None = None
    step_index: int | None = None
    controls: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    requested_by_skill_id: str | None = None
    permission_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolReturn:
    tool_return_id: str
    tool_interface: str
    status: str
    input_evidence_ids: list[str]
    output_evidence_ids: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    provenance_label: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    permission_decision: dict[str, Any] = field(default_factory=dict)
    execution_trace: dict[str, Any] = field(default_factory=dict)
    raw_backend_metadata: dict[str, Any] = field(default_factory=dict)
