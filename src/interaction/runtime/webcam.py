"""Live webcam calibration and gaze-session helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Callable
import math
import random
from typing import Any

from interaction.contracts import ActionName, ActionProposal, EnvironmentSnapshot, GazeObservation, GroundedTarget, RiskLevel
from interaction.control import CommandBroker
from interaction.feedback import GazeFeedbackEvent, GazeLoopPhase
from interaction.platform import PlatformAdapter
from interaction.vision import (
    CalibrationProfile,
    CalibrationSample,
    GazeSample,
    GazeSmoother,
    GazeTargetInferencer,
    NormalizedPoint,
    NormalizedScreenTarget,
    WebcamProvider,
)

from .gaze import GazeTrackingLoop


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


@dataclass(frozen=True)
class LiveGazeContext:
    observation: GazeObservation
    target: GroundedTarget
    attempted_frames: int
    successful_readings: int
    missing_readings: int
    grounded_readings: int
    dominant_target_hits: int


@dataclass
class CursorFollowController:
    smoothing: float = 0.1
    deadzone: float = 0.05
    edge_padding: float = 0.05
    max_step: float = 0.04
    dispatched_point: tuple[float, float] | None = None

    def next_point(self, observation: GazeObservation) -> tuple[float, float] | None:
        if observation.x_norm is None or observation.y_norm is None:
            return None
        target = (
            _clamp_with_padding(observation.x_norm, self.edge_padding),
            _clamp_with_padding(observation.y_norm, self.edge_padding),
        )
        if self.dispatched_point is None:
            self.dispatched_point = target
            return target

        dx = target[0] - self.dispatched_point[0]
        dy = target[1] - self.dispatched_point[1]
        distance = math.hypot(dx, dy)
        if distance < self.deadzone:
            return None

        dynamic_smoothing = min(0.24, self.smoothing + distance * 0.18)
        step_x = dx * dynamic_smoothing
        step_y = dy * dynamic_smoothing
        step_distance = math.hypot(step_x, step_y)
        if step_distance > self.max_step and step_distance > 0:
            scale = self.max_step / step_distance
            step_x *= scale
            step_y *= scale

        next_point = (
            _clamp_with_padding(self.dispatched_point[0] + step_x, self.edge_padding),
            _clamp_with_padding(self.dispatched_point[1] + step_y, self.edge_padding),
        )
        self.dispatched_point = next_point
        return next_point


def default_webcam_calibration_targets() -> tuple[WebcamCalibrationTarget, ...]:
    return (
        WebcamCalibrationTarget(label="upper left", expected=NormalizedPoint(0.2, 0.2)),
        WebcamCalibrationTarget(label="upper right", expected=NormalizedPoint(0.8, 0.2)),
        WebcamCalibrationTarget(label="center", expected=NormalizedPoint(0.5, 0.5)),
        WebcamCalibrationTarget(label="lower left", expected=NormalizedPoint(0.2, 0.8)),
        WebcamCalibrationTarget(label="lower right", expected=NormalizedPoint(0.8, 0.8)),
    )


def generate_single_screen_calibration_targets(*, seed: int | None = None) -> tuple[WebcamCalibrationTarget, ...]:
    """Generate four well-spread primary-screen anchors in randomized order."""

    rng = random.Random(seed)
    anchors = [
        ("upper left", 0.18, 0.18),
        ("upper right", 0.82, 0.18),
        ("lower left", 0.18, 0.82),
        ("lower right", 0.82, 0.82),
    ]
    targets = [
        WebcamCalibrationTarget(
            label=label,
            expected=NormalizedPoint(
                x=_clamp_normalized(base_x + rng.uniform(-0.045, 0.045)),
                y=_clamp_normalized(base_y + rng.uniform(-0.045, 0.045)),
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
                x=_clamp_normalized(base_x + rng.uniform(-0.03, 0.03)),
                y=_clamp_normalized(base_y + rng.uniform(-0.03, 0.03)),
            ),
        )
        for index, (label, base_x, base_y) in enumerate(chosen)
    ]
    rng.shuffle(targets)
    return tuple(targets)


def default_webcam_targets() -> list[NormalizedScreenTarget]:
    regions: list[NormalizedScreenTarget] = []
    horizontal = ("left", "center", "right")
    vertical = ("upper", "middle", "lower")
    width = 1.0 / 3.0
    height = 1.0 / 3.0
    for row_index, vertical_label in enumerate(vertical):
        for col_index, horizontal_label in enumerate(horizontal):
            x = col_index * width
            y = row_index * height
            regions.append(
                NormalizedScreenTarget(
                    target_id=f"{vertical_label}_{horizontal_label}",
                    label=f"{vertical_label.title()} {horizontal_label.title()} region",
                    role="region",
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                )
            )
    return regions


def collect_webcam_calibration(
    provider: WebcamProvider,
    *,
    targets: tuple[WebcamCalibrationTarget, ...] | None = None,
    frames_per_step: int = 6,
    delta_ms: int = 100,
    max_attempts_multiplier: int = 4,
    before_step: Callable[[WebcamCalibrationTarget, int, int], None] | None = None,
) -> tuple[CalibrationProfile, list[dict[str, Any]], dict[str, float | int]]:
    samples: list[CalibrationSample] = []
    step_results: list[dict[str, Any]] = []
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
        samples.append(CalibrationSample(raw=raw_point, expected=step.expected))
        step_results.append(
            {
                "label": step.label,
                "expected": {"x": step.expected.x, "y": step.expected.y},
                "raw": {"x": raw_point.x, "y": raw_point.y},
                "success": True,
                "confidence": aggregate.sample.confidence,
                "used_frames": aggregate.used_frames,
                "attempted_frames": aggregate.attempted_frames,
            }
        )
    if len(samples) < 3:
        raise WebcamCalibrationError(
            "insufficient_calibration_samples",
            "Unable to collect enough stable webcam calibration samples.",
            payload={"steps": step_results, "successful_steps": len(samples)},
        )
    quality = _calibration_signal_quality(samples)
    if quality["raw_x_span"] < 0.08 or quality["raw_y_span"] < 0.05:
        raise WebcamCalibrationError(
            "unstable_calibration_signal",
            (
                "The live gaze signal did not move enough during calibration. "
                f'Observed raw span was x={quality["raw_x_span"]:.3f}, y={quality["raw_y_span"]:.3f}. '
                "Retry and exaggerate the eye movement between dots while keeping your head steadier."
            ),
            payload={
                "steps": step_results,
                "successful_steps": len(samples),
                "quality": quality,
            },
        )
    return CalibrationProfile.fit(samples), step_results, quality


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


def run_live_webcam_trace(
    provider: WebcamProvider,
    loop: GazeTrackingLoop,
    *,
    frames: int,
    delta_ms: int,
    targets: list[NormalizedScreenTarget] | None = None,
    environment: EnvironmentSnapshot | None = None,
    point_visualizer: Any | None = None,
    camera_preview: Any | None = None,
) -> tuple[list[GazeFeedbackEvent], dict[str, int]]:
    if targets is None:
        targets = default_webcam_targets()
    if environment is None:
        environment = EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera")
    events: list[GazeFeedbackEvent] = []
    successful_readings = 0
    missing_readings = 0
    triggered_events = 0
    for _ in range(frames):
        reading = provider.read(delta_ms=delta_ms)
        if reading is None:
            missing_readings += 1
            _hide_point_visualizer(point_visualizer)
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="No coarse gaze reading from webcam provider."))
            continue
        successful_readings += 1
        calibrated_point = loop.calibration_profile.apply(reading.sample.point)
        if point_visualizer is not None:
            _show_point_visualizer(point_visualizer, calibrated_point.x, calibrated_point.y)
        if camera_preview is not None:
            _show_camera_preview(camera_preview, reading, calibrated_point=(calibrated_point.x, calibrated_point.y))
        sample_events = loop.process_sample(reading.sample, targets, environment)
        triggered_events += sum(1 for event in sample_events if event.phase == GazeLoopPhase.TRIGGERED)
        events.extend(sample_events)
    events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
    return events, {
        "attempted_frames": frames,
        "successful_readings": successful_readings,
        "missing_readings": missing_readings,
        "triggered_events": triggered_events,
    }


def run_live_cursor_follow(
    provider: WebcamProvider,
    profile: CalibrationProfile,
    *,
    adapter: PlatformAdapter,
    frames: int,
    delta_ms: int,
    environment: EnvironmentSnapshot | None = None,
    min_confidence: float = 0.6,
    min_move_distance: float = 0.05,
    smoothing: float = 0.1,
    edge_padding: float = 0.05,
    max_step: float = 0.04,
    response_gain_x: float = 0.25,
    response_gain_y: float = 0.5,
    inferencer: GazeTargetInferencer | None = None,
    smoother: GazeSmoother | None = None,
    targets: list[NormalizedScreenTarget] | None = None,
    point_visualizer: Any | None = None,
    initial_cursor_point: tuple[float, float] | None = None,
    camera_preview: Any | None = None,
) -> tuple[list[GazeFeedbackEvent], dict[str, int]]:
    if environment is None:
        environment = EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera")
    if targets is None:
        targets = default_webcam_targets()
    inferencer = inferencer or GazeTargetInferencer()
    smoother = smoother or GazeSmoother()
    broker = CommandBroker()
    cursor = CursorFollowController(
        smoothing=smoothing,
        deadzone=min_move_distance,
        edge_padding=edge_padding,
        max_step=max_step,
    )
    anchor_point = (
        _clamp_with_padding(initial_cursor_point[0], edge_padding),
        _clamp_with_padding(initial_cursor_point[1], edge_padding),
    ) if initial_cursor_point is not None else (0.5, 0.5)
    if initial_cursor_point is not None:
        cursor.dispatched_point = anchor_point

    events: list[GazeFeedbackEvent] = []
    successful_readings = 0
    missing_readings = 0
    moved_events = 0
    low_confidence_readings = 0
    held_events = 0

    if point_visualizer is not None and cursor.dispatched_point is not None:
        _show_point_visualizer(point_visualizer, cursor.dispatched_point[0], cursor.dispatched_point[1])

    for _ in range(frames):
        reading = provider.read(delta_ms=delta_ms)
        if reading is None:
            missing_readings += 1
            _hide_point_visualizer(point_visualizer)
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="No coarse gaze reading from webcam provider."))
            continue

        successful_readings += 1
        calibrated_point = _compress_point_around_anchor(
            profile.apply(reading.sample.point),
            anchor_point=anchor_point,
            gain_x=response_gain_x,
            gain_y=response_gain_y,
            edge_padding=edge_padding,
        )
        observation = smoother.smooth(
            GazeSample(
                point=calibrated_point,
                confidence=reading.sample.confidence,
                delta_ms=reading.sample.delta_ms,
            )
        )
        target = inferencer.infer(observation, targets)
        if observation.confidence < min_confidence or observation.x_norm is None or observation.y_norm is None:
            low_confidence_readings += 1
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="Live gaze confidence is too low for cursor follow.",
                    observation=observation,
                    target=target,
                )
            )
            continue

        point = cursor.next_point(observation)
        visual_point = point if point is not None else cursor.dispatched_point
        if point_visualizer is not None and visual_point is not None:
            _show_point_visualizer(point_visualizer, visual_point[0], visual_point[1])
        if camera_preview is not None:
            _show_camera_preview(
                camera_preview,
                reading,
                calibrated_point=(observation.x_norm, observation.y_norm) if observation.x_norm is not None and observation.y_norm is not None else None,
                cursor_point=visual_point,
                target_label=target.label if target is not None else None,
            )
        if point is None:
            held_events += 1
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message="Live gaze cursor hold is within the movement threshold.",
                    observation=observation,
                    target=target,
                )
            )
            continue

        proposal = ActionProposal(
            action=ActionName.FOCUS_TARGET,
            arguments={
                "normalized_point": {"x": point[0], "y": point[1]},
                **({"target_ref": target.target_id, "target_label": target.label} if target is not None else {}),
                **({"target_bounds": target.bounds.model_dump(mode="json")} if target is not None and target.bounds is not None else {}),
            },
            confidence=observation.confidence,
            risk=RiskLevel.L1,
            requires_confirmation=False,
            rationale="Live gaze cursor-follow mode moved the pointer to the current normalized gaze point.",
        )
        decision = broker.decide(proposal)
        request = broker.build_execution_request(decision, environment)
        result = adapter.execute(request)
        moved_events += 1
        message = 'Moved cursor with live gaze follow.'
        if target is not None:
            message = f'Moved cursor with live gaze follow toward "{target.label}".'
        events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRACKING,
                message=message,
                observation=observation,
                target=target,
                proposal=proposal,
                result=result,
            )
        )

    events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
    return events, {
        "attempted_frames": frames,
        "successful_readings": successful_readings,
        "missing_readings": missing_readings,
        "moved_events": moved_events,
        "held_events": held_events,
        "low_confidence_readings": low_confidence_readings,
    }


def capture_live_gaze_context(
    provider: WebcamProvider,
    profile: CalibrationProfile,
    *,
    frames: int,
    delta_ms: int,
    targets: list[NormalizedScreenTarget] | None = None,
    inferencer: GazeTargetInferencer | None = None,
    smoother: GazeSmoother | None = None,
) -> tuple[LiveGazeContext | None, list[GazeFeedbackEvent], dict[str, int]]:
    if targets is None:
        targets = default_webcam_targets()
    inferencer = inferencer or GazeTargetInferencer()
    smoother = smoother or GazeSmoother()

    events: list[GazeFeedbackEvent] = []
    successful_readings = 0
    missing_readings = 0
    grounded_readings = 0
    hits: Counter[str] = Counter()
    latest_match: dict[str, tuple[GazeObservation, GroundedTarget, int]] = {}

    for index in range(frames):
        reading = provider.read(delta_ms=delta_ms)
        if reading is None:
            missing_readings += 1
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="No coarse gaze reading from webcam provider.",
                )
            )
            continue

        successful_readings += 1
        calibrated_point = profile.apply(reading.sample.point)
        observation = smoother.smooth(
            GazeSample(
                point=calibrated_point,
                confidence=reading.sample.confidence,
                delta_ms=reading.sample.delta_ms,
            )
        )
        target = inferencer.infer(observation, targets)
        if target is None:
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="No grounded live gaze target is stable enough yet.",
                    observation=observation,
                )
            )
            continue

        grounded_readings += 1
        hits[target.target_id] += 1
        latest_match[target.target_id] = (observation, target, index)
        events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRACKING,
                message=f'Grounded live gaze target "{target.label}".',
                observation=observation,
                target=target,
            )
        )

    summary = {
        "attempted_frames": frames,
        "successful_readings": successful_readings,
        "missing_readings": missing_readings,
        "grounded_readings": grounded_readings,
    }
    if not hits:
        return None, events, summary

    dominant_target_id = max(
        hits,
        key=lambda target_id: (hits[target_id], latest_match[target_id][2]),
    )
    observation, target, _ = latest_match[dominant_target_id]
    context = LiveGazeContext(
        observation=observation,
        target=target,
        attempted_frames=frames,
        successful_readings=successful_readings,
        missing_readings=missing_readings,
        grounded_readings=grounded_readings,
        dominant_target_hits=hits[dominant_target_id],
    )
    return context, events, {
        **summary,
        "dominant_target_hits": context.dominant_target_hits,
    }


def _clamp_with_padding(value: float, padding: float) -> float:
    padding = max(0.0, min(0.45, padding))
    return max(padding, min(1.0 - padding, value))


def _clamp_normalized(value: float) -> float:
    return max(0.0, min(1.0, value))


def _calibration_signal_quality(samples: list[CalibrationSample]) -> dict[str, float | int]:
    raw_xs = [sample.raw.x for sample in samples]
    raw_ys = [sample.raw.y for sample in samples]
    return {
        "sample_count": len(samples),
        "raw_x_span": max(raw_xs) - min(raw_xs),
        "raw_y_span": max(raw_ys) - min(raw_ys),
        "raw_x_mean": sum(raw_xs) / len(raw_xs),
        "raw_y_mean": sum(raw_ys) / len(raw_ys),
    }


def _compress_point_around_anchor(
    point: NormalizedPoint,
    *,
    anchor_point: tuple[float, float],
    gain_x: float,
    gain_y: float,
    edge_padding: float,
) -> NormalizedPoint:
    gain_x = max(0.05, min(1.0, float(gain_x)))
    gain_y = max(0.05, min(1.0, float(gain_y)))
    anchor_x = _clamp_with_padding(anchor_point[0], edge_padding)
    anchor_y = _clamp_with_padding(anchor_point[1], edge_padding)
    return NormalizedPoint(
        x=_clamp_with_padding(anchor_x + (point.x - anchor_x) * gain_x, edge_padding),
        y=_clamp_with_padding(anchor_y + (point.y - anchor_y) * gain_y, edge_padding),
    )


def _show_point_visualizer(point_visualizer: Any | None, x_norm: float, y_norm: float) -> None:
    if point_visualizer is None:
        return
    try:
        point_visualizer.show_point(x_norm, y_norm)
    except Exception:
        return


def _hide_point_visualizer(point_visualizer: Any | None) -> None:
    if point_visualizer is None:
        return
    try:
        point_visualizer.hide()
    except Exception:
        return


def _show_camera_preview(
    camera_preview: Any | None,
    reading: Any,
    *,
    calibrated_point: tuple[float, float] | None = None,
    cursor_point: tuple[float, float] | None = None,
    target_label: str | None = None,
) -> None:
    if camera_preview is None:
        return
    try:
        camera_preview.show_reading(
            reading,
            calibrated_point=calibrated_point,
            cursor_point=cursor_point,
            target_label=target_label,
        )
    except Exception:
        return
