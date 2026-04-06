"""Shared webcam provider types and context-manager support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np

from .gaze import GazeSample


class WebcamProviderError(RuntimeError):
    """Raised when a webcam provider cannot initialize or read from the camera."""

    def __init__(self, code: str, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.payload = payload or {}


@dataclass(frozen=True)
class WebcamGazeReading:
    sample: GazeSample
    frame_size: tuple[int, int]
    face_bbox: tuple[int, int, int, int]
    eye_boxes: tuple[tuple[int, int, int, int], ...]
    frame: np.ndarray | None = None
    pupil_points: tuple[tuple[int, int], ...] = ()
    signals: dict[str, float] | None = None
    raw_sample: GazeSample | None = None
    quality: dict[str, float | int | bool] | None = None


@dataclass(frozen=True)
class WebcamSampleAggregate:
    sample: GazeSample
    used_frames: int
    attempted_frames: int
    signals: dict[str, float] | None = None
    quality: dict[str, float | int | bool] | None = None


class WebcamProvider(Protocol):
    camera_index: int

    def open(self) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    def read(self, *, delta_ms: int = 100) -> WebcamGazeReading | None: ...

    def capture_average_sample(
        self,
        *,
        required_frames: int = 6,
        max_attempts: int | None = None,
        delta_ms: int = 100,
    ) -> WebcamSampleAggregate | None: ...


class WebcamProviderContextMixin:
    """Provide safe `with` support for providers that hold camera resources."""

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
