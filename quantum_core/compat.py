"""
QuantumA Core - Compatibility helpers.

Utility leggere per rendere il progetto più robusto su sistemi con
encoding console diversi e ambienti cross-platform.
"""

from __future__ import annotations

import os
import sys
from typing import Any


def configure_stdio() -> None:
    """Rende stdout/stderr più tolleranti agli encoding non UTF-8."""
    try:
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def safe_print(*args: Any, **kwargs: Any) -> None:
    """Print tollerante agli errori di encoding."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(text, **kwargs)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
