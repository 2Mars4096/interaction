"""Calibration helpers for normalized gaze points."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from .provider_common import clamp_unit_interval


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
    """Affine calibration with optional per-axis piecewise remaps.

    The affine fields remain populated even when piecewise knots are active so
    persisted profiles stay backward-compatible and still have a linear fallback
    for composition and legacy callers.
    """

    x_scale: float = 1.0
    y_scale: float = 1.0
    x_offset: float = 0.0
    y_offset: float = 0.0
    fit_mode: str = "linear"
    x_input_knots: tuple[float, ...] = ()
    x_output_knots: tuple[float, ...] = ()
    y_input_knots: tuple[float, ...] = ()
    y_output_knots: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit_mode", str(self.fit_mode or "linear"))
        object.__setattr__(self, "x_input_knots", _coerce_knots(self.x_input_knots))
        object.__setattr__(self, "x_output_knots", _coerce_knots(self.x_output_knots))
        object.__setattr__(self, "y_input_knots", _coerce_knots(self.y_input_knots))
        object.__setattr__(self, "y_output_knots", _coerce_knots(self.y_output_knots))
        if len(self.x_input_knots) != len(self.x_output_knots) or len(self.x_input_knots) < 2:
            object.__setattr__(self, "x_input_knots", ())
            object.__setattr__(self, "x_output_knots", ())
        if len(self.y_input_knots) != len(self.y_output_knots) or len(self.y_input_knots) < 2:
            object.__setattr__(self, "y_input_knots", ())
            object.__setattr__(self, "y_output_knots", ())

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
        use_piecewise = len(samples) >= 7
        x_curve = _fit_axis_curve(raw_xs, expected_xs, linear_scale=x_scale, linear_offset=x_offset) if use_piecewise else None
        y_curve = _fit_axis_curve(raw_ys, expected_ys, linear_scale=y_scale, linear_offset=y_offset) if use_piecewise else None
        return cls(
            x_scale=x_scale,
            y_scale=y_scale,
            x_offset=x_offset,
            y_offset=y_offset,
            fit_mode="piecewise_axis" if x_curve or y_curve else "linear",
            x_input_knots=x_curve[0] if x_curve is not None else (),
            x_output_knots=x_curve[1] if x_curve is not None else (),
            y_input_knots=y_curve[0] if y_curve is not None else (),
            y_output_knots=y_curve[1] if y_curve is not None else (),
        )

    def apply(self, point: NormalizedPoint) -> NormalizedPoint:
        return NormalizedPoint(
            x=self._apply_x(point.x),
            y=self._apply_y(point.y),
        )

    def compose(self, correction: "CalibrationProfile") -> "CalibrationProfile":
        if not self._has_piecewise_mapping() and not correction._has_piecewise_mapping():
            return CalibrationProfile(
                x_scale=self.x_scale * correction.x_scale,
                y_scale=self.y_scale * correction.y_scale,
                x_offset=self.x_offset * correction.x_scale + correction.x_offset,
                y_offset=self.y_offset * correction.y_scale + correction.y_offset,
            )
        x_inputs = _compose_axis_sample_inputs(self.x_input_knots)
        y_inputs = _compose_axis_sample_inputs(self.y_input_knots)
        x_outputs = tuple(correction._apply_x(self._apply_x(value)) for value in x_inputs)
        y_outputs = tuple(correction._apply_y(self._apply_y(value)) for value in y_inputs)
        x_scale, x_offset = _fit_affine(x_inputs, x_outputs)
        y_scale, y_offset = _fit_affine(y_inputs, y_outputs)
        return CalibrationProfile(
            x_scale=x_scale,
            y_scale=y_scale,
            x_offset=x_offset,
            y_offset=y_offset,
            fit_mode="piecewise_axis",
            x_input_knots=x_inputs,
            x_output_knots=x_outputs,
            y_input_knots=y_inputs,
            y_output_knots=y_outputs,
        )

    def _apply_x(self, value: float) -> float:
        return _apply_axis_mapping(
            float(value),
            scale=self.x_scale,
            offset=self.x_offset,
            input_knots=self.x_input_knots,
            output_knots=self.x_output_knots,
        )

    def _apply_y(self, value: float) -> float:
        return _apply_axis_mapping(
            float(value),
            scale=self.y_scale,
            offset=self.y_offset,
            input_knots=self.y_input_knots,
            output_knots=self.y_output_knots,
        )

    def _has_piecewise_mapping(self) -> bool:
        return len(self.x_input_knots) >= 2 or len(self.y_input_knots) >= 2


def _fit_scale(raw_values: list[float], expected_values: list[float]) -> float:
    count = len(raw_values)
    if count == 0:
        return 1.0
    raw_mean = sum(raw_values) / count
    expected_mean = sum(expected_values) / count
    variance = sum((raw - raw_mean) ** 2 for raw in raw_values)
    if variance <= 1e-9:
        return 1.0
    covariance = sum((raw - raw_mean) * (expected - expected_mean) for raw, expected in zip(raw_values, expected_values))
    return covariance / variance

def _coerce_knots(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _fit_affine(raw_values: tuple[float, ...], expected_values: tuple[float, ...]) -> tuple[float, float]:
    scale = _fit_scale(list(raw_values), list(expected_values))
    offset = sum(expected - raw * scale for raw, expected in zip(raw_values, expected_values)) / len(raw_values)
    return scale, offset


def _fit_axis_curve(
    raw_values: list[float],
    expected_values: list[float],
    *,
    linear_scale: float,
    linear_offset: float,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if len(raw_values) < 2:
        return None
    pairs = sorted((float(raw), float(expected)) for raw, expected in zip(raw_values, expected_values))
    grouped_pairs = _merge_duplicate_axis_inputs(pairs)
    if len(grouped_pairs) < 2:
        return None
    monotonic_score = _axis_rank_correlation(grouped_pairs)
    if monotonic_score < 0.82:
        return None
    inputs = tuple(raw for raw, _, _ in grouped_pairs)
    fitted_outputs = tuple(_isotonic_regression([expected for _, expected, _ in grouped_pairs], [weight for _, _, weight in grouped_pairs]))
    if len(inputs) < 2:
        return None
    linear_mae = _axis_linear_mean_abs_error(raw_values, expected_values, scale=linear_scale, offset=linear_offset)
    piecewise_mae = _axis_piecewise_mean_abs_error(raw_values, expected_values, inputs=inputs, outputs=fitted_outputs)
    if piecewise_mae >= linear_mae * 0.92:
        return None
    return inputs, fitted_outputs


def _merge_duplicate_axis_inputs(
    pairs: list[tuple[float, float]],
    *,
    tolerance: float = 1e-6,
) -> list[tuple[float, float, float]]:
    grouped: list[tuple[float, float, float]] = []
    for raw, expected in pairs:
        if grouped and abs(grouped[-1][0] - raw) <= tolerance:
            prev_raw, prev_expected, prev_weight = grouped[-1]
            next_weight = prev_weight + 1.0
            grouped[-1] = (
                (prev_raw * prev_weight + raw) / next_weight,
                (prev_expected * prev_weight + expected) / next_weight,
                next_weight,
            )
            continue
        grouped.append((raw, expected, 1.0))
    return grouped


def _isotonic_regression(values: list[float], weights: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    counts: list[int] = []
    for value, weight in zip(values, weights):
        blocks.append([float(value), float(weight)])
        counts.append(1)
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            merged_weight = blocks[-2][1] + blocks[-1][1]
            merged_value = ((blocks[-2][0] * blocks[-2][1]) + (blocks[-1][0] * blocks[-1][1])) / merged_weight
            merged_count = counts[-2] + counts[-1]
            blocks[-2:] = [[merged_value, merged_weight]]
            counts[-2:] = [merged_count]
    result: list[float] = []
    for (value, _weight), count in zip(blocks, counts):
        result.extend([value] * count)
    return result


def _apply_axis_mapping(
    value: float,
    *,
    scale: float,
    offset: float,
    input_knots: tuple[float, ...],
    output_knots: tuple[float, ...],
) -> float:
    if len(input_knots) >= 2 and len(input_knots) == len(output_knots):
        return _apply_piecewise_axis_curve(value, input_knots, output_knots)
    return clamp_unit_interval(value * scale + offset)


def _apply_piecewise_axis_curve(
    value: float,
    input_knots: tuple[float, ...],
    output_knots: tuple[float, ...],
) -> float:
    if len(input_knots) < 2:
        return clamp_unit_interval(value)
    if value <= input_knots[0]:
        return clamp_unit_interval(output_knots[0])
    if value >= input_knots[-1]:
        return clamp_unit_interval(output_knots[-1])
    index = bisect_right(input_knots, value) - 1
    return clamp_unit_interval(
        _interpolate_segment(
            value,
            input_knots[index],
            input_knots[index + 1],
            output_knots[index],
            output_knots[index + 1],
        )
    )


def _interpolate_segment(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if abs(x1 - x0) <= 1e-9:
        return (y0 + y1) / 2.0
    position = (value - x0) / (x1 - x0)
    return y0 + (y1 - y0) * position


def _compose_axis_sample_inputs(existing_knots: tuple[float, ...]) -> tuple[float, ...]:
    base_inputs = set(existing_knots) if existing_knots else set()
    base_inputs.update({0.0, 0.18, 0.34, 0.5, 0.66, 0.82, 1.0})
    return tuple(sorted(base_inputs))


def _axis_linear_mean_abs_error(
    raw_values: list[float],
    expected_values: list[float],
    *,
    scale: float,
    offset: float,
) -> float:
    if not raw_values:
        return 0.0
    return sum(abs(clamp_unit_interval(raw * scale + offset) - expected) for raw, expected in zip(raw_values, expected_values)) / len(raw_values)


def _axis_piecewise_mean_abs_error(
    raw_values: list[float],
    expected_values: list[float],
    *,
    inputs: tuple[float, ...],
    outputs: tuple[float, ...],
) -> float:
    if not raw_values:
        return 0.0
    return sum(abs(_apply_piecewise_axis_curve(raw, inputs, outputs) - expected) for raw, expected in zip(raw_values, expected_values)) / len(raw_values)


def _axis_rank_correlation(grouped_pairs: list[tuple[float, float, float]]) -> float:
    if len(grouped_pairs) < 2:
        return 1.0
    expected_values = [expected for _, expected, _ in grouped_pairs]
    expected_ranks = _ordinal_ranks(expected_values)
    index_ranks = list(range(len(grouped_pairs)))
    return _pearson_correlation(index_ranks, expected_ranks)


def _ordinal_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    current_rank = 0
    while current_rank < len(ordered):
        start = current_rank
        value = ordered[current_rank][1]
        while current_rank < len(ordered) and ordered[current_rank][1] == value:
            current_rank += 1
        average_rank = (start + current_rank - 1) / 2.0
        for original_index, _ in ordered[start:current_rank]:
            ranks[original_index] = average_rank
    return ranks


def _pearson_correlation(xs: list[int] | list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(float(value) for value in xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance_x = sum((float(value) - mean_x) ** 2 for value in xs)
    variance_y = sum((value - mean_y) ** 2 for value in ys)
    if variance_x <= 1e-9 or variance_y <= 1e-9:
        return 0.0
    covariance = sum((float(x) - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return covariance / ((variance_x * variance_y) ** 0.5)
