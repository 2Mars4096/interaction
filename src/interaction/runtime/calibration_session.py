"""Calibration and validation helpers for webcam-based gaze providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import random
from typing import Any

from interaction.vision import CalibrationProfile, CalibrationSample, NormalizedPoint, WebcamProvider

from .cursor_controllers import clamp_normalized


class WebcamCalibrationError(RuntimeError):
    """Raised when webcam calibration cannot collect enough usable samples."""

    def __init__(self, code: str, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.payload = payload or {}


@dataclass(frozen=True)
class WebcamCalibrationTarget:
    label: str
    expected: NormalizedPoint


def default_webcam_calibration_targets() -> tuple[WebcamCalibrationTarget, ...]:
    return (
        WebcamCalibrationTarget(label="upper left", expected=NormalizedPoint(0.2, 0.2)),
        WebcamCalibrationTarget(label="upper center", expected=NormalizedPoint(0.5, 0.2)),
        WebcamCalibrationTarget(label="upper right", expected=NormalizedPoint(0.8, 0.2)),
        WebcamCalibrationTarget(label="middle left", expected=NormalizedPoint(0.2, 0.5)),
        WebcamCalibrationTarget(label="center", expected=NormalizedPoint(0.5, 0.5)),
        WebcamCalibrationTarget(label="middle right", expected=NormalizedPoint(0.8, 0.5)),
        WebcamCalibrationTarget(label="lower left", expected=NormalizedPoint(0.2, 0.8)),
        WebcamCalibrationTarget(label="lower center", expected=NormalizedPoint(0.5, 0.8)),
        WebcamCalibrationTarget(label="lower right", expected=NormalizedPoint(0.8, 0.8)),
    )


def generate_single_screen_calibration_targets(
    *,
    seed: int | None = None,
    rows: int = 3,
) -> tuple[WebcamCalibrationTarget, ...]:
    """Generate a randomized single-screen calibration grid."""

    if int(rows) == 3:
        return _generate_three_row_calibration_targets(seed=seed)
    if int(rows) != 5:
        raise ValueError("Calibration grid rows must be either 3 or 5.")

    rng = random.Random(seed)
    row_names = ("upper", "upper middle", "center", "lower middle", "lower")
    row_positions = (0.14, 0.32, 0.5, 0.68, 0.86)
    column_names = ("left", "center", "right")
    column_positions = (0.18, 0.5, 0.82)
    anchors: list[tuple[str, float, float]] = []
    for row_name, base_y in zip(row_names, row_positions):
        for column_name, base_x in zip(column_names, column_positions):
            if row_name == "center" and column_name == "center":
                label = "center"
            elif column_name == "center":
                label = row_name
            else:
                label = f"{row_name} {column_name}"
            anchors.append((label, base_x, base_y))
    targets = [
        WebcamCalibrationTarget(
            label=label,
            expected=NormalizedPoint(
                x=clamp_normalized(base_x + rng.uniform(-0.03, 0.03)),
                y=clamp_normalized(base_y + rng.uniform(-0.03, 0.03)),
            ),
        )
        for label, base_x, base_y in anchors
    ]
    rng.shuffle(targets)
    return tuple(targets)


def generate_single_screen_validation_targets(
    *,
    count: int = 6,
    seed: int | None = None,
) -> tuple[WebcamCalibrationTarget, ...]:
    """Generate well-spread randomized validation points across one display."""

    rng = random.Random(seed)
    candidates = [
        ("upper left", 0.22, 0.22),
        ("upper center", 0.5, 0.18),
        ("upper right", 0.78, 0.22),
        ("middle left", 0.22, 0.5),
        ("middle center", 0.5, 0.5),
        ("middle right", 0.78, 0.5),
        ("lower left", 0.22, 0.78),
        ("lower center", 0.5, 0.82),
        ("lower right", 0.78, 0.78),
    ]
    desired_count = max(1, min(len(candidates), int(count)))
    chosen = rng.sample(candidates, k=desired_count)
    targets = [
        WebcamCalibrationTarget(
            label=f"validation {index + 1} ({label})",
            expected=NormalizedPoint(
                x=clamp_normalized(base_x + rng.uniform(-0.03, 0.03)),
                y=clamp_normalized(base_y + rng.uniform(-0.03, 0.03)),
            ),
        )
        for index, (label, base_x, base_y) in enumerate(chosen)
    ]
    rng.shuffle(targets)
    return tuple(targets)


def collect_webcam_calibration(
    provider: WebcamProvider,
    *,
    targets: tuple[WebcamCalibrationTarget, ...] | None = None,
    frames_per_step: int = 6,
    delta_ms: int = 100,
    max_attempts_multiplier: int = 4,
    before_step: Callable[[WebcamCalibrationTarget, int, int], None] | None = None,
    after_step: Callable[[dict[str, Any], int, int], None] | None = None,
) -> tuple[CalibrationProfile, list[dict[str, Any]], dict[str, Any]]:
    samples: list[CalibrationSample] = []
    step_results: list[dict[str, Any]] = []
    step_quality_summaries: list[dict[str, float]] = []
    calibration_targets = targets or default_webcam_calibration_targets()
    total_steps = len(calibration_targets)
    for step_index, step in enumerate(calibration_targets, start=1):
        if before_step is not None:
            before_step(step, step_index, total_steps)
        aggregate = provider.capture_average_sample(
            required_frames=frames_per_step,
            max_attempts=max(frames_per_step, 1) * max_attempts_multiplier,
            delta_ms=delta_ms,
        )
        if aggregate is None:
            step_result = {
                "label": step.label,
                "expected": {"x": step.expected.x, "y": step.expected.y},
                "success": False,
                "used_frames": 0,
                "attempted_frames": max(frames_per_step, 1) * max_attempts_multiplier,
            }
            step_results.append(step_result)
            if after_step is not None:
                after_step(step_result, step_index, total_steps)
            continue
        raw_point = aggregate.sample.point
        samples.append(CalibrationSample(raw=raw_point, expected=step.expected))
        step_result = {
            "label": step.label,
            "expected": {"x": step.expected.x, "y": step.expected.y},
            "raw": {"x": raw_point.x, "y": raw_point.y},
            "success": True,
            "confidence": aggregate.sample.confidence,
            "used_frames": aggregate.used_frames,
            "attempted_frames": aggregate.attempted_frames,
        }
        if isinstance(aggregate.quality, dict) and aggregate.quality:
            step_result["quality"] = aggregate.quality
            step_quality_summaries.append({key: float(value) for key, value in aggregate.quality.items() if isinstance(value, (int, float))})
        step_results.append(step_result)
        if after_step is not None:
            after_step(step_result, step_index, total_steps)
    if len(samples) < 3:
        raise WebcamCalibrationError(
            "insufficient_calibration_samples",
            "Unable to collect enough stable webcam calibration samples.",
            payload={"steps": step_results, "successful_steps": len(samples)},
        )
    quality = _calibration_signal_quality(samples)
    quality.update(_aggregate_quality_maps(step_quality_summaries))
    quality.update(_calibration_axis_diagnostics(quality))
    if quality["blocked_axes"]:
        blocked_axes = [str(axis) for axis in quality["blocked_axes"]]
        if len(blocked_axes) == 1:
            profile = _fit_profile_with_axis_fallback(samples, blocked_axes=blocked_axes)
            quality["fallback_profile_used"] = True
            quality["fallback_axes"] = blocked_axes
            quality["warnings"] = [
                *list(quality.get("warnings", [])),
                f"Saved a degraded calibration profile with identity mapping on the blocked {blocked_axes[0].upper()} axis.",
            ]
            quality.update(_calibration_fit_error_metrics(samples, profile))
            return profile, step_results, quality
        blocked_axes = ", ".join(str(axis) for axis in quality["blocked_axes"])
        raise WebcamCalibrationError(
            "unstable_calibration_signal",
            (
                "The live gaze signal was not stable enough during calibration. "
                f"Blocked axes: {blocked_axes}. "
                f'Observed raw span was x={quality["raw_x_span"]:.3f}, y={quality["raw_y_span"]:.3f}. '
                "Retry and exaggerate the eye movement between dots while keeping your head steadier."
            ),
            payload={
                "steps": step_results,
                "successful_steps": len(samples),
                "quality": quality,
            },
        )
    profile = CalibrationProfile.fit(samples)
    quality.update(_calibration_fit_error_metrics(samples, profile))
    return profile, step_results, quality


def _generate_three_row_calibration_targets(*, seed: int | None = None) -> tuple[WebcamCalibrationTarget, ...]:
    rng = random.Random(seed)
    anchors = [
        ("upper left", 0.18, 0.18),
        ("upper center", 0.5, 0.18),
        ("upper right", 0.82, 0.18),
        ("middle left", 0.18, 0.5),
        ("center", 0.5, 0.5),
        ("middle right", 0.82, 0.5),
        ("lower left", 0.18, 0.82),
        ("lower center", 0.5, 0.82),
        ("lower right", 0.82, 0.82),
    ]
    targets = [
        WebcamCalibrationTarget(
            label=label,
            expected=NormalizedPoint(
                x=clamp_normalized(base_x + rng.uniform(-0.03, 0.03)),
                y=clamp_normalized(base_y + rng.uniform(-0.03, 0.03)),
            ),
        )
        for label, base_x, base_y in anchors
    ]
    rng.shuffle(targets)
    return tuple(targets)


def collect_webcam_validation(
    provider: WebcamProvider,
    profile: CalibrationProfile,
    *,
    targets: tuple[WebcamCalibrationTarget, ...] | None = None,
    frames_per_step: int = 4,
    delta_ms: int = 100,
    max_attempts_multiplier: int = 4,
    before_step: Callable[[WebcamCalibrationTarget, int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    validation_targets = targets or generate_single_screen_validation_targets()
    total_steps = len(validation_targets)
    step_results: list[dict[str, Any]] = []

    for step_index, step in enumerate(validation_targets, start=1):
        if before_step is not None:
            before_step(step, step_index, total_steps)
        aggregate = provider.capture_average_sample(
            required_frames=frames_per_step,
            max_attempts=max(frames_per_step, 1) * max_attempts_multiplier,
            delta_ms=delta_ms,
        )
        if aggregate is None:
            step_results.append(
                {
                    "label": step.label,
                    "expected": {"x": step.expected.x, "y": step.expected.y},
                    "success": False,
                    "used_frames": 0,
                    "attempted_frames": max(frames_per_step, 1) * max_attempts_multiplier,
                }
            )
            continue

        raw_point = aggregate.sample.point
        tracked_point = profile.apply(raw_point)
        error_x = tracked_point.x - step.expected.x
        error_y = tracked_point.y - step.expected.y
        error_distance = math.hypot(error_x, error_y)
        step_results.append(
            {
                "label": step.label,
                "expected": {"x": step.expected.x, "y": step.expected.y},
                "raw": {"x": raw_point.x, "y": raw_point.y},
                "tracked": {"x": tracked_point.x, "y": tracked_point.y},
                "error": {"dx": error_x, "dy": error_y, "distance": error_distance},
                "success": True,
                "confidence": aggregate.sample.confidence,
                "used_frames": aggregate.used_frames,
                "attempted_frames": aggregate.attempted_frames,
            }
        )

    successful_results = [result for result in step_results if result["success"]]
    distances = [float(result["error"]["distance"]) for result in successful_results]
    summary: dict[str, float | int] = {
        "attempted_points": total_steps,
        "successful_points": len(successful_results),
        "missing_points": total_steps - len(successful_results),
        "mean_error_distance": (sum(distances) / len(distances)) if distances else 0.0,
        "max_error_distance": max(distances) if distances else 0.0,
    }
    return step_results, summary


def _calibration_signal_quality(samples: list[CalibrationSample]) -> dict[str, Any]:
    raw_xs = [sample.raw.x for sample in samples]
    raw_ys = [sample.raw.y for sample in samples]
    return {
        "sample_count": len(samples),
        "raw_x_span": max(raw_xs) - min(raw_xs),
        "raw_y_span": max(raw_ys) - min(raw_ys),
        "raw_x_mean": sum(raw_xs) / len(raw_xs),
        "raw_y_mean": sum(raw_ys) / len(raw_ys),
    }


def _calibration_fit_error_metrics(samples: list[CalibrationSample], profile: CalibrationProfile) -> dict[str, float]:
    distances: list[float] = []
    dxs: list[float] = []
    dys: list[float] = []
    for sample in samples:
        adjusted = profile.apply(sample.raw)
        dx = adjusted.x - sample.expected.x
        dy = adjusted.y - sample.expected.y
        dxs.append(abs(dx))
        dys.append(abs(dy))
        distances.append(math.hypot(dx, dy))
    if not distances:
        return {
            "fit_mean_error_distance": 0.0,
            "fit_max_error_distance": 0.0,
            "fit_mean_error_x": 0.0,
            "fit_mean_error_y": 0.0,
        }
    return {
        "fit_mean_error_distance": sum(distances) / len(distances),
        "fit_max_error_distance": max(distances),
        "fit_mean_error_x": sum(dxs) / len(dxs),
        "fit_mean_error_y": sum(dys) / len(dys),
    }


def _calibration_axis_diagnostics(quality: dict[str, Any]) -> dict[str, Any]:
    x_diagnostics = _calibration_axis_status(axis="x", quality=quality)
    y_diagnostics = _calibration_axis_status(axis="y", quality=quality)
    blocked_axes = [axis for axis, diagnostics in (("x", x_diagnostics), ("y", y_diagnostics)) if diagnostics["status"] == "blocked"]
    weak_axes = [axis for axis, diagnostics in (("x", x_diagnostics), ("y", y_diagnostics)) if diagnostics["status"] == "weak"]
    warnings = [diagnostics["message"] for diagnostics in (x_diagnostics, y_diagnostics) if diagnostics["status"] == "weak"]
    return {
        "axis_diagnostics": {
            "x": x_diagnostics,
            "y": y_diagnostics,
        },
        "blocked_axes": blocked_axes,
        "weak_axes": weak_axes,
        "warnings": warnings,
    }


def _calibration_axis_status(*, axis: str, quality: dict[str, Any]) -> dict[str, Any]:
    raw_span_key = f"raw_{axis}_span"
    raw_span = float(quality.get(raw_span_key, 0.0))
    min_span = 0.08 if axis == "x" else 0.05
    warn_span = min_span * 1.2
    confidence = _quality_metric(quality, f"{axis}_axis_confidence_mean", f"{axis}_axis_confidence")
    saturated_mean = _quality_metric(
        quality,
        f"{axis}_saturated_mean",
        f"{axis}_saturated_ratio_mean",
        f"{axis}_saturated_ratio",
    )
    disagreement = _quality_metric(
        quality,
        "horizontal_disagreement_mean" if axis == "x" else "vertical_disagreement_mean",
        "horizontal_disagreement" if axis == "x" else "vertical_disagreement",
    )
    issues: list[str] = []
    status = "good"
    if raw_span < min_span:
        status = "blocked"
        issues.append("compressed_span")
    elif raw_span < warn_span:
        status = "weak"
        issues.append("narrow_span")
    if confidence is not None:
        blocked_threshold = 0.55 if axis == "x" else 0.48
        warning_threshold = 0.65 if axis == "x" else 0.58
        if confidence < blocked_threshold:
            status = "blocked"
            issues.append("low_axis_confidence")
        elif confidence < warning_threshold and status != "blocked":
            status = "weak"
            issues.append("soft_axis_confidence")
    if saturated_mean is not None:
        if saturated_mean > 0.45:
            status = "blocked"
            issues.append("high_saturation")
        elif saturated_mean > 0.20 and status != "blocked":
            status = "weak"
            issues.append("moderate_saturation")
    if disagreement is not None and disagreement > 0.16 and status == "good":
        status = "weak"
        issues.append("eye_disagreement")
    if status == "blocked":
        message = f"{axis.upper()} axis is not reliable enough for calibration."
    elif status == "weak":
        message = f"{axis.upper()} axis is weaker than ideal and may need refinement."
    else:
        message = f"{axis.upper()} axis quality looks usable."
    return {
        "status": status,
        "raw_span": raw_span,
        "min_span": min_span,
        "confidence": confidence,
        "saturated_mean": saturated_mean,
        "disagreement": disagreement,
        "issues": issues,
        "message": message,
    }


def _quality_metric(quality: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = quality.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _aggregate_quality_maps(quality_maps: list[dict[str, float]]) -> dict[str, float]:
    if not quality_maps:
        return {}
    aggregated: dict[str, list[float]] = {}
    for quality_map in quality_maps:
        for key, value in quality_map.items():
            aggregated.setdefault(key, []).append(float(value))
    return {f"{key}_mean": sum(values) / len(values) for key, values in aggregated.items()}


def _fit_profile_with_axis_fallback(
    samples: list[CalibrationSample],
    *,
    blocked_axes: list[str],
) -> CalibrationProfile:
    fitted = CalibrationProfile.fit(samples)
    block_x = "x" in blocked_axes
    block_y = "y" in blocked_axes
    fit_mode = "linear"
    if (not block_x and len(fitted.x_input_knots) >= 2) or (not block_y and len(fitted.y_input_knots) >= 2):
        fit_mode = "piecewise_axis"
    return CalibrationProfile(
        x_scale=1.0 if block_x else fitted.x_scale,
        y_scale=1.0 if block_y else fitted.y_scale,
        x_offset=0.0 if block_x else fitted.x_offset,
        y_offset=0.0 if block_y else fitted.y_offset,
        fit_mode=fit_mode,
        x_input_knots=() if block_x else fitted.x_input_knots,
        x_output_knots=() if block_x else fitted.x_output_knots,
        y_input_knots=() if block_y else fitted.y_input_knots,
        y_output_knots=() if block_y else fitted.y_output_knots,
    )


__all__ = [
    "WebcamCalibrationError",
    "WebcamCalibrationTarget",
    "collect_webcam_calibration",
    "collect_webcam_validation",
    "default_webcam_calibration_targets",
    "generate_single_screen_calibration_targets",
    "generate_single_screen_validation_targets",
]
