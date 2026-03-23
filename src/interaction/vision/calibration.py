"""Calibration helpers for normalized gaze points."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


@dataclass(frozen=True)
class CalibrationSample:
    raw: NormalizedPoint
    expected: NormalizedPoint


@dataclass(frozen=True)
class CalibrationProfile:
    x_scale: float = 1.0
    y_scale: float = 1.0
    x_offset: float = 0.0
    y_offset: float = 0.0

    @classmethod
    def fit(cls, samples: list[CalibrationSample]) -> "CalibrationProfile":
        if not samples:
            return cls()
        raw_xs = [sample.raw.x for sample in samples]
        raw_ys = [sample.raw.y for sample in samples]
        expected_xs = [sample.expected.x for sample in samples]
        expected_ys = [sample.expected.y for sample in samples]

        x_scale = _fit_scale(raw_xs, expected_xs)
        y_scale = _fit_scale(raw_ys, expected_ys)
        x_offset = sum(expected - raw * x_scale for raw, expected in zip(raw_xs, expected_xs)) / len(samples)
        y_offset = sum(expected - raw * y_scale for raw, expected in zip(raw_ys, expected_ys)) / len(samples)
        return cls(x_scale=x_scale, y_scale=y_scale, x_offset=x_offset, y_offset=y_offset)

    def apply(self, point: NormalizedPoint) -> NormalizedPoint:
        return NormalizedPoint(
            x=_clamp(point.x * self.x_scale + self.x_offset),
            y=_clamp(point.y * self.y_scale + self.y_offset),
        )


def _fit_scale(raw_values: list[float], expected_values: list[float]) -> float:
    raw_span = max(raw_values) - min(raw_values)
    expected_span = max(expected_values) - min(expected_values)
    if raw_span <= 1e-6:
        return 1.0
    return expected_span / raw_span


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
