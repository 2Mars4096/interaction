"""Shared helpers for webcam-provider backends."""

from __future__ import annotations

from contextlib import contextmanager
import os

import cv2


def clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def effective_delta_ms(
    requested_delta_ms: int,
    *,
    last_frame_monotonic: float | None,
    current_monotonic: float,
) -> int:
    fallback_delta_ms = max(1, int(requested_delta_ms))
    if last_frame_monotonic is None:
        return fallback_delta_ms
    observed_delta_ms = int(round((current_monotonic - last_frame_monotonic) * 1000.0))
    if observed_delta_ms <= 0:
        return fallback_delta_ms
    return observed_delta_ms


def silence_opencv_logging() -> None:
    logging = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging is None:
        return
    set_log_level = getattr(logging, "setLogLevel", None)
    silent_level = getattr(logging, "LOG_LEVEL_SILENT", None)
    if callable(set_log_level) and silent_level is not None:
        set_log_level(silent_level)


@contextmanager
def suppress_native_stderr():
    try:
        original_stderr = os.dup(2)
    except OSError:
        yield
        return

    devnull_fd = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(original_stderr, 2)
        os.close(original_stderr)
        if devnull_fd is not None:
            os.close(devnull_fd)
