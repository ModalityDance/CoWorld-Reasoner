"""Source context and component feedback for harness co-evolution."""
from __future__ import annotations

import difflib
from pathlib import Path

from harness_state import _source_paths, EXPERIMENT_FILES


def component_for_path(path: str) -> str:
    """Assign an edited task path to the component it primarily owns."""
    normalized = path.replace("\\", "/")
    if "/task_skills/" in normalized:
        return "skills"
    if normalized.startswith("evolved_harness/"):
        if any(token in normalized for token in ("verifier", "evidence_tools")):
            return "verifier"
        if any(token in normalized for token in ("catalog.py", "visual_tools", "vwm_request", "h3_prompt",
                                                "visual_generation", "fal_h3_backend_adapter")):
            return "tools"
        if any(token in normalized for token in ("p1_", "p2_", "p3_", "p4_",
                                                "capability_assessment", "evidence_reasoning")):
            return "stage_instructions"
    if normalized in {"runtime/video_verifier.py"}:
        return "verifier"
    if normalized in {"runtime/visual_dispatch.py", "runtime/generation_execution.py",
                      "runtime/request_media.py"}:
        return "tools"
    if normalized.startswith("runtime/") or normalized == "execution.py":
        return "execution_and_context"
    return "other_task_source"


SOURCE_GUIDE = """The files below are the current task implementation for inspection and editing.
Prompts within them configure the task agent.
The full project is available in the workspace.
"""


COMPONENTS = {
    "skills": (
        "Develop systematic workflows and reusable experience for similar tasks here: applicability, operation choice, "
        "request construction and evidence interpretation.",
        ("evolved_harness/task_skills/*/*",)),
    "stage_instructions": (
        "Keep these focused on cross-task general principles, stage responsibilities and basic output requirements; avoid placing task-family procedures here.",
        ("evolved_harness/p[1-4]_*.py", "evolved_harness/capability_assessment.py",
         "evolved_harness/evidence_reasoning.py")),
    "tools": (
        "Skill discovery, operation descriptions and parameters, request compilation and "
        "generation adapters. Follow the request into the actual backend arguments.",
        ("evolved_harness/catalog.py", "evolved_harness/visual_tools.py",
         "evolved_harness/vwm_request.py", "evolved_harness/h3_prompt.py",
         "evolved_harness/visual_generation.py", "evolved_harness/fal_h3_backend_adapter.py",
         "runtime/visual_dispatch.py", "runtime/generation_execution.py", "runtime/request_media.py")),
    "verifier": (
        "VLM observation instructions, profiles, output schema and delivery of generated-frame "
        "findings to the main agent. Visual inputs are generated frames only; the main agent "
        "compares with originals and chooses the answer.",
        ("evolved_harness/evidence_tools.py", "evolved_harness/verifier*.py",
         "runtime/video_verifier.py")),
    "execution_and_context": (
        "Stage execution, model interaction, ordered media, skill loading/refresh, history "
        "continuation and tool-result delivery. Inspect how each component reaches the model.",
        ("execution.py", "runtime/agent_runtime.py", "runtime/agents_sdk_input.py",
         "runtime/agents_sdk_model.py", "runtime/durable_agent.py", "runtime/media_context.py",
         "runtime/vwm_harness/interaction_history.py")),
}


def component_map(project: Path) -> dict:
    """Locate component entry points in the current project, including skill resources."""
    result = {}
    for name, (role, patterns) in COMPONENTS.items():
        paths = {p.relative_to(project).as_posix() for pattern in patterns
                 for p in project.glob(pattern) if p.is_file()}
        if name == "skills":
            paths.update(p.relative_to(project).as_posix()
                         for p in (project / "evolved_harness/task_skills").rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts)
        result[name] = {"role": role, "entry_points": sorted(paths)}
    return result


def project_guide(project: Path) -> str:
    parts = ["# CoWorld Reasoner\n\nThis is the editable task project. README.md describes usage.\n",
             "## Components\n"]
    for name, component in component_map(project).items():
        paths = ", ".join(f"`{p}`" for p in component["entry_points"])
        parts.append(f"\n- **{name.replace('_', ' ')}**: {component['role']}\n  Entry points: {paths}.\n")
    parts.append("\nFollow these entry points through their imports and calls to explore the full implementation.\n"
                 "`experiment_state` locates candidates, comparisons, histories and media.\n"
                 "The workspace contains current source and completed execution evidence.\n")
    return "".join(parts)


def source_context(project: Path) -> str:
    """Render current task sources for on-demand inspection."""
    paths = [path for path in _source_paths(project)
             if path.relative_to(project).as_posix() not in EXPERIMENT_FILES]
    parts = [SOURCE_GUIDE]
    for path in paths:
        if not path.is_file() or path.suffix not in {".py", ".md", ".txt", ".json"}:
            continue
        parts.append(f"\n=== {path.relative_to(project)} ===\n{path.read_text(encoding='utf-8')}")
    return "\n".join(parts)


def project_feedback(output: Path, initial: Path) -> dict:
    project = output / "workspace" / "project"
    packet = output / "workspace" / "source_context.txt"
    packet.write_text(source_context(project), encoding="utf-8")
    before = {p.relative_to(initial).as_posix(): p for p in _source_paths(initial)}
    after = {p.relative_to(project).as_posix(): p for p in _source_paths(project)}
    patches = []
    for name in sorted(set(before) | set(after)):
        if name in EXPERIMENT_FILES:
            continue
        old = before[name].read_bytes() if name in before else b""
        new = after[name].read_bytes() if name in after else b""
        if old == new:
            continue
        try:
            patch = "".join(difflib.unified_diff(old.decode().splitlines(True),
                new.decode().splitlines(True), fromfile="initial/" + name, tofile="project/" + name))
        except UnicodeDecodeError:
            patch = "Binary resource changed"
        patches.append({"path": name, "diff": patch})
    notes = output / "workspace" / "stage_findings.md"
    content = notes.read_text(encoding="utf-8") if notes.is_file() else ""
    return {"source_context_path": str(packet), "components": component_map(project),
            "changes_from_initial_source": patches,
            "stage_findings": content[:16000], "stage_findings_truncated": len(content) > 16000,
            "stage_findings_path": str(notes)}


def trial_feedback(candidate: dict, rows: list[dict]) -> dict:
    experiment = candidate.get("experiment") or {}
    targets = set(experiment.get("target_case_ids") or [])
    return {"candidate_id": candidate["candidate_id"],
            "base_candidate_id": experiment.get("base_candidate_id"),
            "hypothesis": experiment.get("hypothesis"),
            "expected_behavior": experiment.get("expected_behavior"),
            "preserve_behavior": experiment.get("preserve_behavior"),
            "target_results": [r for r in rows if r["sample_id"] in targets],
            "improved": [r["sample_id"] for r in rows if r["score_change"] == "improved"],
            "regressed": [r["sample_id"] for r in rows if r["score_change"] == "regressed"],
            "persistent_errors": [r["sample_id"] for r in rows if r["score_change"] == "persistent_error"],
            "skill_guidance_refreshed": [r["sample_id"] for r in rows
                                         if r.get("skill_guidance_refreshed")],
            "generation_observed": sum(bool(r.get("generation_records")) for r in rows),
            "verifier_observed": sum(bool(r.get("verifier_records")) for r in rows),
            "changed_source": candidate.get("revision_diff") or []}
