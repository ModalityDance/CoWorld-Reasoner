"""Load local API settings, giving existing environment variables priority."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILES = (ROOT / ".env.local", ROOT / ".env")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_local_env(env_files: tuple[Path, ...] = DEFAULT_ENV_FILES) -> list[str]:
    """Load unset KEY=VALUE pairs and return the names of loaded variables."""

    if os.environ.get("AGENTIC_DISABLE_LOCAL_ENV") == "1":
        return []

    loaded: list[str] = []
    for env_file in env_files:
        if not env_file.exists():
            continue
        for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = _unquote(value)
            loaded.append(key)
    return loaded
