"""Paced camera reads with measured timestamps for live interaction."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import time
from typing import Any

from .clock import Clock, monotonic_ms


@dataclass(frozen=True)
class TimedReading:
    reading: Any | None
    observed_at_ms: int
    elapsed_ms: int
    clock_continuous: bool


def timed_readings(
    provider,
    *,
    frames: int,
    interval_ms: int,
    clock: Clock = monotonic_ms,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[TimedReading]:
    """The first frame supplies zero fixation time; delays are measured, not assumed."""
    previous = None
    for _ in range(max(0, frames)):
        if previous is not None:
            remaining = max(0, interval_ms) - (clock() - previous)
            if remaining > 0:
                sleep(remaining / 1000)
        reading = provider.read(delta_ms=1)
        now = clock()
        continuous = previous is None or now >= previous
        elapsed = max(0, now - previous) if previous is not None else 0
        yield TimedReading(reading, now, elapsed, continuous)
        previous = now
