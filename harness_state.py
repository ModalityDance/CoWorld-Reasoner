"""Snapshot and restore task code, skills and their supporting files."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


EXPERIMENT_FILES = frozenset({
    "harness_state.py", "co_evolution.py", "evaluate.py", "download_data.py",
    "external_scorer.py",
    "runtime/vwm_harness/codex_session.py",
    "runtime/vwm_harness/experiment_context.py",
    "runtime/vwm_harness/supervisor_agent.py",
    "runtime/vwm_harness/supervisor_feedback.py",
    "runtime/vwm_harness/candidate_evaluation.py",
    "runtime/vwm_harness/stage_assessment.py",
    "runtime/vwm_harness/project_context.py",
})
_IGNORED_PARTS = {".git", "__pycache__", ".venv", ".pytest_cache"}


def is_source_path(relative: Path) -> bool:
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        return False
    if any(part in _IGNORED_PARTS or part.startswith(".env") for part in relative.parts):
        return False
    if relative.suffix in {".pyc", ".pyo"}:
        return False
    return relative.parts[0] in {"runtime", "evolved_harness"} or (
        len(relative.parts) == 1 and relative.suffix == ".py"
    )


def editable_scope() -> dict:
    """Describe task implementation boundaries, independent of stage or function."""
    return {
        "task_source": ["evolved_harness/", "runtime/", "top-level Python modules"],
        "experiment_files": sorted(EXPERIMENT_FILES),
        "file_operations": ["modify", "add", "delete", "rename"],
    }


def validate_revision(root: Path, relative: str, content: bytes | None) -> None:
    """Check file location and Python syntax without prescribing its implementation."""
    path = Path(relative)
    if not is_source_path(path) or relative in EXPERIMENT_FILES:
        raise ValueError(f"not_task_source:{relative}")
    _target(root, path)
    if content is not None and path.suffix == ".py":
        try:
            compile(content, relative, "exec")
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid_revision_python:{relative}:{exc}") from exc


def _source_paths(root: Path) -> list[Path]:
    paths = list(root.glob("*.py"))
    for directory in (root / "runtime", root / "evolved_harness"):
        if directory.is_symlink():
            raise ValueError(f"source_symlink:{directory}")
        paths.extend(directory.rglob("*"))
    result = []
    for path in paths:
        if not is_source_path(path.relative_to(root)):
            continue
        if path.is_symlink():
            raise ValueError(f"source_symlink:{path}")
        if path.is_file():
            result.append(path)
    return sorted(result)


def _target(root: Path, relative: Path) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"harness_path_outside_root:{relative}")
    return path


def _clear_bytecode(path: Path) -> None:
    if path.suffix == ".py":
        for cache in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
            cache.unlink(missing_ok=True)


def install_source_file(source: Path, destination: Path) -> None:
    """Atomically install source and invalidate timestamp-based Python caches."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    _clear_bytecode(destination)


def remove_source_file(path: Path) -> None:
    path.unlink(missing_ok=True)
    _clear_bytecode(path)


def save_harness(root: Path, candidate: Path) -> None:
    """Save the complete code and resources used by a candidate."""
    target = candidate / "harness"
    if target.exists():
        raise ValueError("candidate_harness_exists_use_new_output_dir")
    paths = _source_paths(root)
    if not paths:
        raise ValueError("harness_source_missing")
    candidate.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".harness-", dir=candidate))
    for source in paths:
        relative = source.relative_to(root)
        source = _target(root, relative)
        destination = _target(staging, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in ("README.md", "requirements.txt"):
        if (root / name).is_file():
            shutil.copy2(root / name, staging / name)
    os.replace(staging, target)


def restore_harness(root: Path, candidate: Path, backup: Path, *, resume: bool = False) -> None:
    """Install an exact candidate file set, retaining replaced or removed files."""
    saved = candidate / "harness"
    sources = {path.relative_to(saved): path for path in _source_paths(saved)}
    if not sources:
        raise ValueError(f"candidate_harness_missing:{candidate}")
    if not resume and backup.exists() and any(backup.iterdir()):
        raise ValueError("restore_backup_directory_must_be_new_or_empty")
    current = {path.relative_to(root): path for path in _source_paths(root)}
    # The runner owns experiment orchestration; the task agent evolves harness source.
    for relative in sorted(set(sources) | set(current)):
        if relative.as_posix() in EXPERIMENT_FILES:
            continue
        destination = _target(root, relative)
        previous = _target(backup, relative)
        if destination.is_file() and not previous.exists():
            previous.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, previous)
        if relative in sources:
            install_source_file(_target(saved, relative), destination)
        else:
            remove_source_file(destination)
