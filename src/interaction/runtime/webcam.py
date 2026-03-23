"""Live webcam calibration and gaze-session helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from interaction.contracts import EnvironmentSnapshot, GazeObservation, GroundedTarget
from interaction.feedback import GazeFeedbackEvent, GazeLoopPhase
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


def collect_webcam_calibration(
    provider: OpenCVWebcamGazeProvider,
    *,
    frames_per_step: int = 6,
    delta_ms: int = 100,
    max_attempts_multiplier: int = 4,
    before_step: Callable[[WebcamCalibrationTarget], None] | None = None,
) -> tuple[CalibrationProfile, list[dict[str, Any]]]:
    samples: list[CalibrationSample] = []
    step_results: list[dict[str, Any]] = []
    for step in default_webcam_calibration_targets():
        if before_step is not None:
            before_step(step)
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
    return CalibrationProfile.fit(samples), step_results


def run_live_webcam_trace(
    provider: OpenCVWebcamGazeProvider,
    loop: GazeTrackingLoop,
    *,
    frames: int,
    delta_ms: int,
    targets: list[NormalizedScreenTarget] | None = None,
    environment: EnvironmentSnapshot | None = None,
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
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="No coarse gaze reading from webcam provider."))
            continue
        successful_readings += 1
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


def capture_live_gaze_context(
    provider: OpenCVWebcamGazeProvider,
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
