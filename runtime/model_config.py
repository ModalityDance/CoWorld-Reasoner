"""Model and API settings for the task agent and supervisor."""

from __future__ import annotations

import os


DEFAULT_VLM_MODEL = "gpt-5.6-luna"
DEFAULT_OPENAI_COMPAT_BASE_URL = "https://api.openai.com/v1"


def default_live_model() -> str:
    return os.environ.get("AGENTIC_DEFAULT_LIVE_MODEL") or DEFAULT_VLM_MODEL


def shared_vlm_model() -> str:
    """Resolve the shared model name from the supported environment variables."""

    llm_model = os.environ.get("AGENTIC_LLM_MODEL")
    vlm_model = os.environ.get("AGENTIC_VLM_MODEL")
    if llm_model and vlm_model and llm_model.strip() != vlm_model.strip():
        raise RuntimeError("shared_vlm_model_mismatch:AGENTIC_LLM_MODEL!=AGENTIC_VLM_MODEL")
    return (llm_model or vlm_model or default_live_model()).strip()


def default_live_base_url() -> str:
    return os.environ.get("AGENTIC_DEFAULT_LIVE_BASE_URL") or DEFAULT_OPENAI_COMPAT_BASE_URL


def live_api_key(*env_keys: str) -> str | None:
    """Read the first configured API key in the requested priority order."""
    for key in env_keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


def live_model_candidates(model: str | None, *, env_var: str | None = None) -> list[str]:
    """Use the configured model or an explicitly supplied fallback list."""

    raw = os.environ.get(env_var or "") if env_var else None
    if raw:
        return list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    return [model or default_live_model()]
