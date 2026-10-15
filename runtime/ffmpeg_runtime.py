"""Resolve FFmpeg executables available to the active Python runtime."""

from __future__ import annotations

import shutil


def ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (OSError, RuntimeError):
        return None
