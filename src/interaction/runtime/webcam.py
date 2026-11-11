"""Live webcam calibration and gaze-session helpers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import math
import time
from typing import Any

from interaction.contracts import ActionName, ActionProposal, BrokerDecisionType, EnvironmentSnapshot, GazeObservation, GroundedTarget, RiskLevel
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
    OpenCVWebcamGazeProvider,
)

from .gaze import GazeTrackingLoop
from .clock import Clock, monotonic_ms
from .sampling import timed_readings
from interaction.vision.stability import StableGazeLock


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
    phase: str = "fitting"


@dataclass(frozen=True)
class LiveGazeContext:
    observation: GazeObservation
    target: GroundedTarget
    attempted_frames: int
    successful_readings: int
    missing_readings: int
    grounded_readings: int
    dominant_target_hits: int
    observed_at_ms: int = 0
    stable_duration_ms: int = 0


@dataclass
class CursorFollowController:
    smoothing: float = 0.28
    deadzone: float = 0.015
    edge_padding: float = 0.03
    max_step: float = 0.16
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

        dynamic_smoothing = min(0.72, self.smoothing + distance * 0.6)
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


def default_webcam_validation_targets() -> tuple[WebcamCalibrationTarget, ...]:
    """Independent points that are never included in the fitting data."""
    return (
        WebcamCalibrationTarget("upper middle", NormalizedPoint(0.5, 0.2), "validation"),
        WebcamCalibrationTarget("middle right", NormalizedPoint(0.8, 0.5), "validation"),
        WebcamCalibrationTarget("lower middle", NormalizedPoint(0.5, 0.8), "validation"),
        WebcamCalibrationTarget("middle left", NormalizedPoint(0.2, 0.5), "validation"),
    )


def collect_webcam_calibration(
    provider: OpenCVWebcamGazeProvider,
    *,
    frames_per_step: int = 6,
    delta_ms: int = 100,
    max_attempts_multiplier: int = 4,
    before_step: Callable[[WebcamCalibrationTarget], None] | None = None,
    on_frame: Callable[[], None] | None = None,
) -> tuple[CalibrationProfile, list[dict[str, Any]]]:
    if frames_per_step < 3 or delta_ms < 1 or max_attempts_multiplier < 1:
        raise WebcamCalibrationError("invalid_calibration_options", "Calibration needs at least 3 frames per point and a positive sampling interval.")
    step_results: list[dict[str, Any]] = []

    def collect(targets):
        samples, aggregates = [], []
        for step in targets:
            if before_step is not None:
                before_step(step)
            aggregate = provider.capture_average_sample(
                required_frames=frames_per_step,
                max_attempts=frames_per_step * max_attempts_multiplier,
                delta_ms=delta_ms,
                on_frame=on_frame,
            )
            result = {
                "label": step.label, "phase": step.phase,
                "expected": {"x": step.expected.x, "y": step.expected.y},
                "success": False, "used_frames": 0,
                "attempted_frames": frames_per_step * max_attempts_multiplier,
            }
            step_results.append(result)
            if aggregate is None:
                result["reason"] = "No gaze samples were collected."
                continue
            point = aggregate.sample.point
            dispersion = aggregate.raw_stddev
            valid = (
                aggregate.used_frames >= frames_per_step
                and aggregate.sample_span_ms >= (frames_per_step - 1) * delta_ms * 0.8
                and math.isfinite(aggregate.sample.confidence) and aggregate.sample.confidence >= 0.6
                and all(math.isfinite(value) and 0 <= value <= 1 for value in (point.x, point.y))
                and dispersion is not None
                and all(math.isfinite(value) and value >= 0 for value in (dispersion.x, dispersion.y))
            )
            result.update({
                "raw": {"x": point.x, "y": point.y}, "success": valid,
                "confidence": aggregate.sample.confidence,
                "used_frames": aggregate.used_frames, "attempted_frames": aggregate.attempted_frames,
                "sample_span_ms": aggregate.sample_span_ms,
                "raw_stddev": {"x": dispersion.x, "y": dispersion.y} if dispersion else None,
            })
            if valid:
                samples.append(CalibrationSample(raw=point, expected=step.expected))
                aggregates.append(aggregate)
            else:
                result["reason"] = "Samples were incomplete, too brief, low-confidence, or missing finite dispersion evidence."
        if len(samples) != len(targets):
            raise WebcamCalibrationError(
                "insufficient_calibration_samples", "Every reference point needs a complete stable sample set. Please retry calibration.",
                payload={"steps": step_results, "successful_steps": len(samples)},
            )
        return samples, aggregates

    training, training_aggregates = collect(default_webcam_calibration_targets())
    try:
        profile = CalibrationProfile.fit(training)
    except ValueError as error:
        raise WebcamCalibrationError("invalid_calibration_signal", str(error), payload={"steps": step_results}) from error
    for result, aggregate in zip(step_results, training_aggregates):
        dispersion = aggregate.raw_stddev
        mapped = {"x": abs(profile.x_scale) * dispersion.x, "y": abs(profile.y_scale) * dispersion.y}
        result["mapped_stddev"] = mapped
        if max(mapped.values()) > 0.05:
            raise WebcamCalibrationError(
                "unstable_calibration_samples", "Camera noise becomes too large after calibration. Keep a stable position and retry.",
                payload={"steps": step_results},
            )
    validation, validation_aggregates = collect(default_webcam_validation_targets())
    for result, sample, aggregate in zip(step_results[len(training):], validation, validation_aggregates):
        projected = profile.project(sample.raw)
        result["projected"] = {"x": projected.x, "y": projected.y}
        result["error"] = math.hypot(projected.x - sample.expected.x, projected.y - sample.expected.y)
        result["mapped_stddev"] = {
            "x": abs(profile.x_scale) * aggregate.raw_stddev.x,
            "y": abs(profile.y_scale) * aggregate.raw_stddev.y,
        }
    try:
        profile = profile.validate(
            validation,
            raw_stddevs=[aggregate.raw_stddev for aggregate in validation_aggregates],
            sample_counts=[aggregate.used_frames for aggregate in validation_aggregates],
            camera_index=provider.camera_index,
        )
    except ValueError as error:
        raise WebcamCalibrationError("calibration_validation_failed", str(error), payload={"steps": step_results}) from error
    return profile, step_results


def run_live_webcam_trace(
    provider: OpenCVWebcamGazeProvider,
    loop: GazeTrackingLoop,
    *,
    frames: int,
    delta_ms: int,
    targets: list[NormalizedScreenTarget] | None = None,
    environment: EnvironmentSnapshot | None = None,
    point_visualizer: Any | None = None,
    clock: Clock = monotonic_ms,
    sleep: Callable[[float], None] = time.sleep,
    max_gap_ms: int = 250,
    event_sink: list[GazeFeedbackEvent] | None = None,
) -> tuple[list[GazeFeedbackEvent], dict[str, int]]:
    if targets is None:
        targets = default_webcam_targets()
    if environment is None:
        environment = EnvironmentSnapshot()
    events = event_sink if event_sink is not None else []
    successful_readings = 0
    missing_readings = 0
    triggered_events = 0
    try:
        for timed in timed_readings(provider, frames=frames, interval_ms=delta_ms, clock=clock, sleep=sleep):
            reading = timed.reading
            if reading is None:
                missing_readings += 1
                _hide_point_visualizer(point_visualizer)
                events.extend(loop.tracking_lost(elapsed_ms=timed.elapsed_ms))
                continue
            successful_readings += 1
            elapsed = timed.elapsed_ms
            if not timed.clock_continuous or elapsed > max_gap_ms:
                events.extend(loop.tracking_lost(elapsed_ms=elapsed, reason="Camera sampling paused; reacquiring gaze."))
                elapsed = 0
            sample_events = loop.process_sample(
                GazeSample(reading.sample.point, confidence=reading.sample.confidence, delta_ms=elapsed), targets, environment,
                measured_timing=True,
            )
            current = next((event.observation for event in reversed(sample_events)
                            if event.observation is not None and event.phase in {GazeLoopPhase.TRACKING, GazeLoopPhase.TRIGGERED}), None)
            if current is not None:
                _show_point_visualizer(point_visualizer, current.x_norm, current.y_norm)
            else:
                _hide_point_visualizer(point_visualizer)
            triggered_events += sum(1 for event in sample_events if event.phase == GazeLoopPhase.TRIGGERED)
            events.extend(sample_events)
    finally:
        had_drag = loop.pending_drag_origin is not None
        cancelled = loop.tracking_lost(reason="The live gaze session ended; start a new action to continue.")
        if had_drag:
            events.extend(cancelled)
    events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
    return events, {
        "attempted_frames": frames,
        "successful_readings": successful_readings,
        "missing_readings": missing_readings,
        "triggered_events": triggered_events,
    }


def run_live_cursor_follow(
    provider: OpenCVWebcamGazeProvider,
    profile: CalibrationProfile,
    *,
    adapter: PlatformAdapter,
    frames: int,
    delta_ms: int,
    environment: EnvironmentSnapshot | None = None,
    min_confidence: float = 0.6,
    min_move_distance: float = 0.015,
    smoothing: float = 0.28,
    edge_padding: float = 0.03,
    max_step: float = 0.16,
    inferencer: GazeTargetInferencer | None = None,
    smoother: GazeSmoother | None = None,
    targets: list[NormalizedScreenTarget] | None = None,
    point_visualizer: Any | None = None,
    clock: Clock = monotonic_ms,
    sleep: Callable[[float], None] = time.sleep,
    max_gap_ms: int = 250,
    event_sink: list[GazeFeedbackEvent] | None = None,
) -> tuple[list[GazeFeedbackEvent], dict[str, int]]:
    environment = environment or EnvironmentSnapshot()
    targets = default_webcam_targets() if targets is None else targets
    inferencer = inferencer or GazeTargetInferencer()
    smoother = smoother or GazeSmoother()
    lock = StableGazeLock(min_confidence=min_confidence, max_gap_ms=max_gap_ms)
    broker = CommandBroker()
    cursor = CursorFollowController(smoothing=smoothing, deadzone=min_move_distance, edge_padding=edge_padding, max_step=max_step)
    events = event_sink if event_sink is not None else []
    successful_readings = missing_readings = moved_events = low_confidence_readings = held_events = acquiring_readings = 0

    for timed in timed_readings(provider, frames=frames, interval_ms=delta_ms, clock=clock, sleep=sleep):
        reading = timed.reading
        if not timed.clock_continuous or timed.elapsed_ms > max_gap_ms:
            lock.reset()
            smoother.reset()
        if reading is None:
            missing_readings += 1
            lock.reset()
            smoother.reset()
            _hide_point_visualizer(point_visualizer)
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="Gaze tracking was lost; cursor movement is paused."))
            continue
        successful_readings += 1
        evidence = _live_observation(profile, reading, elapsed_ms=timed.elapsed_ms, min_confidence=min_confidence)
        if evidence is None:
            low_confidence_readings += 1
            lock.reset()
            smoother.reset()
            _hide_point_visualizer(point_visualizer)
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="No valid gaze evidence for cursor follow."))
            continue
        target = inferencer.infer(evidence, targets)
        previous_revision = lock.revision
        stable = lock.update(evidence, target, observed_at_ms=timed.observed_at_ms)
        if lock.revision != previous_revision:
            smoother.reset()
        observation = smoother.smooth(GazeSample(
            NormalizedPoint(evidence.x_norm, evidence.y_norm), confidence=evidence.confidence, delta_ms=timed.elapsed_ms,
        ))
        _show_point_visualizer(point_visualizer, observation.x_norm, observation.y_norm)
        if not stable:
            acquiring_readings += 1
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.TRACKING, message="Acquiring stable gaze before moving the cursor.", observation=observation, target=target))
            continue
        point = cursor.next_point(observation)
        if point is None:
            held_events += 1
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.TRACKING, message="Live gaze cursor hold is within the movement threshold.", observation=observation, target=target))
            continue
        proposal = ActionProposal(
            action=ActionName.FOCUS_TARGET,
            arguments={
                "normalized_point": {"x": point[0], "y": point[1]},
                "target_ref": target.target_id, "target_label": target.label,
                **({"target_bounds": target.bounds.model_dump(mode="json")} if target.bounds is not None else {}),
            },
            confidence=observation.confidence, risk=RiskLevel.L1, requires_confirmation=False,
            rationale="Stable current gaze evidence moved the pointer to the normalized point.",
        )
        decision = broker.decide(proposal)
        if decision.decision != BrokerDecisionType.ALLOW:
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=decision.reason, proposal=proposal, target=target))
            continue
        result = adapter.execute(broker.build_execution_request(decision, environment))
        moved_events += 1
        events.append(GazeFeedbackEvent(
            phase=GazeLoopPhase.TRACKING, message=f'Moved cursor with live gaze follow toward "{target.label}".',
            observation=observation, target=target, proposal=proposal, result=result,
        ))
    events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
    return events, {
        "attempted_frames": max(0, frames), "successful_readings": successful_readings,
        "missing_readings": missing_readings, "moved_events": moved_events, "held_events": held_events,
        "low_confidence_readings": low_confidence_readings, "acquiring_readings": acquiring_readings,
    }


def capture_live_gaze_context(
    provider: OpenCVWebcamGazeProvider,
    profile: CalibrationProfile,
    *,
    frames: int,
    delta_ms: int,
    targets: list[NormalizedScreenTarget] | None = None,
    inferencer: GazeTargetInferencer | None = None,
    smoother: GazeSmoother | None = None,
    clock: Clock = monotonic_ms,
    sleep: Callable[[float], None] = time.sleep,
    max_gap_ms: int = 250,
) -> tuple[LiveGazeContext | None, list[GazeFeedbackEvent], dict[str, int]]:
    targets = default_webcam_targets() if targets is None else targets
    inferencer = inferencer or GazeTargetInferencer()
    smoother = smoother or GazeSmoother()
    lock = StableGazeLock(max_gap_ms=max_gap_ms)
    events = []
    successful_readings = missing_readings = grounded_readings = 0
    latest = None
    for timed in timed_readings(provider, frames=frames, interval_ms=delta_ms, clock=clock, sleep=sleep):
        if not timed.clock_continuous or timed.elapsed_ms > max_gap_ms:
            lock.reset()
            smoother.reset()
        reading = timed.reading
        if reading is None:
            missing_readings += 1
            evidence = None
        else:
            successful_readings += 1
            evidence = _live_observation(profile, reading, elapsed_ms=timed.elapsed_ms)
        if evidence is None:
            lock.reset()
            smoother.reset()
            latest = None
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="No current valid gaze evidence."))
            continue
        target = inferencer.infer(evidence, targets)
        grounded_readings += int(target is not None)
        previous_revision = lock.revision
        stable = lock.update(evidence, target, observed_at_ms=timed.observed_at_ms)
        if lock.revision != previous_revision:
            smoother.reset()
        observation = smoother.smooth(GazeSample(
            NormalizedPoint(evidence.x_norm, evidence.y_norm), confidence=evidence.confidence, delta_ms=timed.elapsed_ms,
        ))
        latest = (observation, target, timed.observed_at_ms) if stable else None
        events.append(GazeFeedbackEvent(
            phase=GazeLoopPhase.TRACKING,
            message="Stable live gaze target acquired." if stable else "Acquiring stable gaze evidence.",
            observation=observation, target=target,
        ))
    summary = {
        "attempted_frames": max(0, frames), "successful_readings": successful_readings,
        "missing_readings": missing_readings, "grounded_readings": grounded_readings,
    }
    if latest is None or not 0 <= clock() - latest[2] <= max_gap_ms:
        return None, events, summary
    observation, target, observed_at = latest
    context = LiveGazeContext(
        observation=observation, target=target, attempted_frames=max(0, frames),
        successful_readings=successful_readings, missing_readings=missing_readings,
        grounded_readings=grounded_readings, dominant_target_hits=lock.sample_count,
        observed_at_ms=observed_at, stable_duration_ms=lock.duration_ms,
    )
    return context, events, {**summary, "dominant_target_hits": context.dominant_target_hits, "stable_duration_ms": lock.duration_ms}


def _live_observation(profile, reading, *, elapsed_ms: int, min_confidence: float = 0.6) -> GazeObservation | None:
    sample = reading.sample
    try:
        values = (sample.point.x, sample.point.y, sample.confidence)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values) or sample.confidence < min_confidence:
            return None
        point = profile.project(sample.point)
        if not all(-1e-9 <= value <= 1 + 1e-9 for value in (point.x, point.y)):
            return None
        return GazeObservation(
            confidence=sample.confidence, x_norm=min(1.0, max(0.0, point.x)), y_norm=min(1.0, max(0.0, point.y)),
            fixation_ms=max(0, elapsed_ms),
        )
    except (ValueError, TypeError, OverflowError):
        return None


def _clamp_with_padding(value: float, padding: float) -> float:
    padding = max(0.0, min(0.45, padding))
    return max(padding, min(1.0 - padding, value))


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
