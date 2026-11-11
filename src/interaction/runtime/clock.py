"""Clock shared by live input workers and interaction state."""

from __future__ import annotations

from collections.abc import Callable
import time


Clock = Callable[[], int]


def monotonic_ms() -> int:
    """Return milliseconds in one process-wide monotonic time domain."""
    return time.monotonic_ns() // 1_000_000
