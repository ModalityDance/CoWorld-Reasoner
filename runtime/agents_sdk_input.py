"""Shared multimodal input helpers for Agents SDK runtimes."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def image_data_url(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    mime_type = mimetypes.guess_type(path.name)[0]
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(path.suffix.lower(), "image/jpeg")
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
