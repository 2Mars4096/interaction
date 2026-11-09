"""Live webcam calibration and gaze-session helpers."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from collections.abc import Callable
import math
import time
from typing import Any

from interaction.contracts import ActionName, ActionProposal, EnvironmentSnapshot, ExecutionStatus, GazeObservation, GroundedTarget, RiskLevel
from interaction.control import CommandBroker
from interaction.feedback import GazeFeedbackEvent, GazeLoopPhase
from interaction.platform import PlatformAdapter, current_position_normalized
from interaction.vision import (
    CalibrationProfile,
    GazeSample,
    GazeSmoother,
    GazeTargetInferencer,
    NormalizedPoint,
    NormalizedScreenTarget,
    WebcamProvider,
    WebcamSampleAggregate,
)

from .calibration_session import (
    WebcamCalibrationError,
    WebcamCalibrationTarget,
    collect_webcam_calibration,
    collect_webcam_validation,
    default_webcam_calibration_targets,
    generate_single_screen_calibration_targets,
    generate_single_screen_validation_targets,
)
from .cursor_controllers import (
    HEAD_ASSIST_ACTIVE_THRESHOLD,
    HEAD_ASSIST_FILTER_WEIGHT,
    ONLINE_BIAS_LEARN_WINDOW_SCALE,
    POSTURE_STABILIZE_MS,
    POSTURE_STABLE_PITCH,
    POSTURE_STABLE_ROLL,
    POSTURE_STABLE_SIZE,
    POSTURE_STABLE_TRANSLATION,
    POSTURE_STABLE_YAW,
    CursorFollowController,
    OnlineBiasCorrector,
    PostureState,
    RelativeCursorController,
    adaptive_relative_deadzone as _adaptive_relative_deadzone,
    average_posture_states as _average_posture_states,
    clamp_normalized as _clamp_normalized,
    clamp_signed as _clamp_signed,
    clamp_with_padding as _clamp_with_padding,
    compress_point_around_anchor as _compress_point_around_anchor,
    gated_head_assist_component as _gated_head_assist_component,
    head_assist_targets as _head_assist_targets,
    is_large_posture_change as _is_large_posture_change,
    lerp as _lerp,
    reading_posture_state as _reading_posture_state,
    relative_controls as _relative_controls,
    update_fixation_window as _update_fixation_window,
    update_posture_window as _update_posture_window,
)
from .gaze import GazeTrackingLoop


@dataclass(frozen=True)
class LiveGazeContext:
    observation: GazeObservation
    target: GroundedTarget
    attempted_frames: int
    successful_readings: int
    missing_readings: int
    grounded_readings: int
    dominant_target_hits: int


__all__ = [
    "CursorFollowController",
    "LiveGazeContext",
    "OnlineBiasCorrector",
    "PostureState",
    "RelativeCursorController",
    "WebcamCalibrationError",
    "WebcamCalibrationTarget",
    "_compress_point_around_anchor",
    "capture_live_gaze_context",
    "collect_webcam_calibration",
    "collect_webcam_validation",
    "default_webcam_calibration_targets",
    "default_webcam_snap_targets",
    "default_webcam_targets",
    "generate_single_screen_calibration_targets",
    "generate_single_screen_validation_targets",
    "run_live_assisted_cursor_follow",
    "run_live_cursor_follow",
    "run_live_cursor_snap",
    "run_live_relative_cursor_follow",
    "run_live_webcam_benchmark",
    "run_live_webcam_trace",
    "run_live_wink_click_test",
]


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


def default_webcam_snap_targets() -> list[NormalizedScreenTarget]:
    regions: list[NormalizedScreenTarget] = []
    horizontal = ("far_left", "left", "mid_left", "left_center", "right_center", "mid_right", "right", "far_right")
    vertical = ("upper", "upper_mid", "middle", "lower_mid", "lower", "bottom")
    horizontal_margin = 0.04
    vertical_margin = 0.05
    width = (1.0 - horizontal_margin * 2.0) / len(horizontal)
    height = (1.0 - vertical_margin * 2.0) / len(vertical)
    for row_index, vertical_label in enumerate(vertical):
        for col_index, horizontal_label in enumerate(horizontal):
            x = horizontal_margin + col_index * width
            y = vertical_margin + row_index * height
            human_horizontal = horizontal_label.replace("_", " ").title()
            human_vertical = vertical_label.replace("_", " ").title()
            regions.append(
                NormalizedScreenTarget(
                    target_id=f"{vertical_label}_{horizontal_label}",
                    label=f"{human_vertical} {human_horizontal} region",
                    role="region",
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                )
            )
    return regions


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
    set_bias_correction_enabled = getattr(loop, "set_bias_correction_enabled", None)
    if callable(set_bias_correction_enabled):
        set_bias_correction_enabled(True)
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
        "bias_update_events": int(getattr(loop, "bias_update_events", 0)),
    }


def run_live_webcam_benchmark(
    provider: WebcamProvider,
    *,
    frames: int,
    delta_ms: int,
    wink_closed_threshold: float = 0.7,
    wink_open_threshold: float = 0.35,
    point_visualizer: Any | None = None,
    camera_preview: Any | None = None,
) -> tuple[list[GazeFeedbackEvent], dict[str, Any]]:
    events = [
        GazeFeedbackEvent(
            phase=GazeLoopPhase.TRACKING,
            message="Benchmarking live webcam gaze stability and wink signals.",
        )
    ]
    readings: list[Any] = []
    successful_readings = 0
    missing_readings = 0
    left_wink_frames = 0
    right_wink_frames = 0
    left_wink_episodes = 0
    right_wink_episodes = 0
    wink_signal_available = False
    left_eye_closed_max = 0.0
    right_eye_closed_max = 0.0
    previous_wink_side: str | None = None

    for _ in range(frames):
        reading = provider.read(delta_ms=delta_ms)
        if reading is None:
            missing_readings += 1
            _hide_point_visualizer(point_visualizer)
            continue
        readings.append(reading)
        successful_readings += 1

        fused_point = reading.sample.point
        _show_point_visualizer(point_visualizer, fused_point.x, fused_point.y)
        _show_camera_preview(
            camera_preview,
            reading,
            calibrated_point=(fused_point.x, fused_point.y),
        )

        left_present, left_eye_closed = _reading_signal(reading, "left_eye_closed")
        right_present, right_eye_closed = _reading_signal(reading, "right_eye_closed")
        wink_signal_available = wink_signal_available or left_present or right_present
        left_eye_closed_max = max(left_eye_closed_max, left_eye_closed)
        right_eye_closed_max = max(right_eye_closed_max, right_eye_closed)
        wink_side = _classify_wink(
            left_eye_closed,
            right_eye_closed,
            closed_threshold=wink_closed_threshold,
            open_threshold=wink_open_threshold,
        )
        if wink_side == "left":
            left_wink_frames += 1
            if previous_wink_side != "left":
                left_wink_episodes += 1
        elif wink_side == "right":
            right_wink_frames += 1
            if previous_wink_side != "right":
                right_wink_episodes += 1
        previous_wink_side = wink_side

    if successful_readings == 0:
        events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.RECOVERING,
                message="No webcam readings were available during benchmark capture.",
            )
        )
    events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))

    raw_points = [(_reading_raw_sample(reading).point.x, _reading_raw_sample(reading).point.y) for reading in readings]
    fused_points = [(reading.sample.point.x, reading.sample.point.y) for reading in readings]
    sample_deltas = [max(1, int(reading.sample.delta_ms)) for reading in readings]
    confidences = [float(reading.sample.confidence) for reading in readings]
    summary: dict[str, Any] = {
        "attempted_frames": frames,
        "successful_readings": successful_readings,
        "missing_readings": missing_readings,
        "sample_rate_hz": (1000.0 / (sum(sample_deltas) / len(sample_deltas))) if sample_deltas else 0.0,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "wink_signal_available": wink_signal_available,
        "left_eye_closed_max": left_eye_closed_max,
        "right_eye_closed_max": right_eye_closed_max,
        "left_wink_frames": left_wink_frames,
        "right_wink_frames": right_wink_frames,
        "left_wink_episodes": left_wink_episodes,
        "right_wink_episodes": right_wink_episodes,
    }
    summary.update(_point_series_metrics(raw_points, prefix="raw"))
    summary.update(_point_series_metrics(fused_points, prefix="fused"))
    summary.update(_raw_fused_delta_metrics(raw_points, fused_points))
    summary.update(_head_signal_metrics(readings))
    summary.update(_aggregate_reading_quality(readings))
    return events, summary


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
    event_sink: Callable[[GazeFeedbackEvent], None] | None = None,
) -> tuple[list[GazeFeedbackEvent], dict[str, Any]]:
    if environment is None:
        environment = EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera")
    if targets is None:
        targets = default_webcam_targets()
    inferencer = inferencer or GazeTargetInferencer(fallback_to_nearest=True)
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

    events: deque[GazeFeedbackEvent] = deque(maxlen=256)
    last_emitted_at = float("-inf")
    last_phase = None
    last_message = None

    def emit(event: GazeFeedbackEvent) -> None:
        nonlocal last_emitted_at, last_phase, last_message
        now = time.monotonic()
        urgent = (event.result is not None and event.result.status != ExecutionStatus.SUCCESS)
        changed_recovery = event.phase == GazeLoopPhase.RECOVERING and event.message != last_message
        if urgent or changed_recovery or event.phase != last_phase or now - last_emitted_at >= 1.0 or event.phase == GazeLoopPhase.IDLE:
            events.append(event)
            if event_sink is not None:
                event_sink(event)
            last_emitted_at, last_phase, last_message = now, event.phase, event.message

    attempted_moves = 0
    failed_moves = 0
    preview_moves = 0
    stopped = False
    attempted_frames = 0
    successful_readings = 0
    missing_readings = 0
    moved_events = 0
    low_confidence_readings = 0
    held_events = 0
    fixation_points: list[tuple[tuple[float, float], int]] = []
    fixation_elapsed_ms = 0
    fixation_radius = max(0.015, min_move_distance * 0.75)
    bias_corrector = OnlineBiasCorrector()
    bias_update_events = 0

    if point_visualizer is not None and cursor.dispatched_point is not None:
        _show_point_visualizer(point_visualizer, cursor.dispatched_point[0], cursor.dispatched_point[1])

    try:
        for _ in range(frames):
            attempted_frames += 1
            reading = provider.read(delta_ms=delta_ms)
            if reading is None:
                missing_readings += 1
                _hide_point_visualizer(point_visualizer)
                emit(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="No coarse gaze reading from webcam provider."))
                continue

            successful_readings += 1
            unbiased_point = _compress_point_around_anchor(
                profile.apply(reading.sample.point),
                anchor_point=anchor_point,
                gain_x=response_gain_x,
                gain_y=response_gain_y,
                edge_padding=edge_padding,
            )
            corrected_point = bias_corrector.apply((unbiased_point.x, unbiased_point.y), edge_padding=edge_padding)
            observation = smoother.smooth(
                GazeSample(
                    point=NormalizedPoint(*corrected_point),
                    confidence=reading.sample.confidence,
                    delta_ms=reading.sample.delta_ms,
                )
            )
            target = inferencer.infer(observation, targets)
            sample_delta_ms = max(0, reading.sample.delta_ms)
            fixation_points, fixation_elapsed_ms = _update_fixation_window(
                fixation_points,
                corrected_point,
                sample_delta_ms,
                radius=fixation_radius,
            )
            if observation.confidence < min_confidence or observation.x_norm is None or observation.y_norm is None:
                low_confidence_readings += 1
                fixation_points = []
                fixation_elapsed_ms = 0
                emit(
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
                if target is not None and target.bounds is not None:
                    bias_updated = bias_corrector.maybe_update(
                        (unbiased_point.x, unbiased_point.y),
                        reference_point=(
                            float(target.bounds.x) + float(target.bounds.width) / 2.0,
                            float(target.bounds.y) + float(target.bounds.height) / 2.0,
                        ),
                        stable_ms=fixation_elapsed_ms,
                        sample_delta_ms=sample_delta_ms,
                        learn_window_x=max(fixation_radius, float(target.bounds.width) * 0.45),
                        learn_window_y=max(fixation_radius, float(target.bounds.height) * 0.45),
                    )
                    if bias_updated:
                        bias_update_events += 1
                held_events += 1
                emit(
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
            attempted_moves += 1
            if result.status != ExecutionStatus.SUCCESS:
                failed_moves += 1
                emit(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING,
                    message=f"Cursor control stopped: {result.message}", observation=observation,
                    target=target, proposal=proposal, result=result))
                break
            moved_events += 1
            if result.details.get("dry_run", False):
                preview_moves += 1
                message = "Previewing gaze position; desktop pointer is unchanged."
            else:
                message = "Cursor following gaze."
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message=message,
                    observation=observation,
                    target=target,
                    proposal=proposal,
                    result=result,
                )
            )

    except KeyboardInterrupt:
        stopped = True
        emit(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="Gaze session stopped. Diagnostics saved."))

    emit(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
    return list(events), {
        "status": "error" if failed_moves else "stopped" if stopped else "success",
        "message": "Cursor execution failed; see session log." if failed_moves else "Gaze session stopped." if stopped else "Gaze session completed.",
        "attempted_frames": attempted_frames,
        "attempted_moves": attempted_moves,
        "failed_moves": failed_moves,
        "preview_moves": preview_moves,
        "executed_moves": moved_events - preview_moves,
        "successful_readings": successful_readings,
        "missing_readings": missing_readings,
        "moved_events": moved_events - preview_moves,
        "successful_move_requests": moved_events,
        "held_events": held_events,
        "low_confidence_readings": low_confidence_readings,
        "bias_update_events": bias_update_events,
    }


def run_live_cursor_snap(
    provider: WebcamProvider,
    profile: CalibrationProfile,
    *,
    adapter: PlatformAdapter,
    frames: int,
    delta_ms: int,
    environment: EnvironmentSnapshot | None = None,
    min_confidence: float = 0.6,
    dwell_ms: int = 700,
    second_dwell_action: ActionName | None = None,
    wink_confirm: bool = False,
    wink_hold_ms: int = 180,
    wink_closed_threshold: float = 0.7,
    wink_open_threshold: float = 0.35,
    unlock_lookaway_ms: int = 700,
    cooldown_ms: int = 900,
    preview_lock_ms: int = 250,
    preview_smoothing: float = 0.24,
    inferencer: GazeTargetInferencer | None = None,
    smoother: GazeSmoother | None = None,
    targets: list[NormalizedScreenTarget] | None = None,
    point_visualizer: Any | None = None,
    camera_preview: Any | None = None,
    event_sink: Callable[[GazeFeedbackEvent], None] | None = None,
) -> tuple[list[GazeFeedbackEvent], dict[str, Any]]:
    if environment is None:
        environment = EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera")
    if targets is None:
        targets = default_webcam_snap_targets()
    inferencer = inferencer or GazeTargetInferencer(fallback_to_nearest=True)
    smoother = smoother or GazeSmoother()
    broker = CommandBroker()

    def _execute_pointer_request(request):
        if request.proposal.action in {
            ActionName.CLICK_TARGET,
            ActionName.RIGHT_CLICK_TARGET,
            ActionName.DOUBLE_CLICK_TARGET,
            ActionName.DRAG_TARGET,
        }:
            _dismiss_live_visualizers_for_pointer_action(point_visualizer, camera_preview)
        else:
            _dismiss_point_visualizer_for_pointer_action(point_visualizer)
        return adapter.execute(request)

    events: list[GazeFeedbackEvent] = []

    def emit(event: GazeFeedbackEvent) -> None:
        events.append(event)
        if event_sink is not None:
            event_sink(event)
    successful_readings = 0
    missing_readings = 0
    low_confidence_readings = 0
    preview_frames = 0
    snap_events = 0
    click_events = 0
    left_wink_events = 0
    right_wink_events = 0
    arm_cancel_events = 0
    wink_signal_available = False
    left_eye_closed_max = 0.0
    right_eye_closed_max = 0.0
    armed_target_id: str | None = None
    current_target_id: str | None = None
    candidate_target_id: str | None = None
    candidate_elapsed_ms = 0
    preview_target: GroundedTarget | None = None
    preview_point: tuple[float, float] | None = None
    armed_precision_point: tuple[float, float] | None = None
    dwell_elapsed_ms = 0
    cooldown_remaining_ms = 0
    left_wink_elapsed_ms = 0
    right_wink_elapsed_ms = 0
    arm_lookaway_elapsed_ms = 0
    left_eye_closed_filtered = 0.0
    right_eye_closed_filtered = 0.0

    for _ in range(frames):
        reading = provider.read(delta_ms=delta_ms)
        if reading is None:
            missing_readings += 1
            _hide_point_visualizer(point_visualizer)
            current_target_id = None
            dwell_elapsed_ms = 0
            emit(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="No coarse gaze reading from webcam provider."))
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
        sample_delta_ms = max(0, reading.sample.delta_ms)
        cooldown_remaining_ms = max(0, cooldown_remaining_ms - sample_delta_ms)
        left_present, left_eye_closed = _reading_signal(reading, "left_eye_closed")
        right_present, right_eye_closed = _reading_signal(reading, "right_eye_closed")
        if wink_confirm:
            left_eye_closed_filtered = _smooth_signal(
                left_eye_closed_filtered,
                left_eye_closed if left_present else 0.0,
                weight=0.68,
            )
            right_eye_closed_filtered = _smooth_signal(
                right_eye_closed_filtered,
                right_eye_closed if right_present else 0.0,
                weight=0.68,
            )
        else:
            left_eye_closed_filtered = 0.0
            right_eye_closed_filtered = 0.0
        wink_signal_available = wink_signal_available or left_present or right_present
        left_signal = max(left_eye_closed_filtered, left_eye_closed)
        right_signal = max(right_eye_closed_filtered, right_eye_closed)
        left_eye_closed_max = max(left_eye_closed_max, left_signal)
        right_eye_closed_max = max(right_eye_closed_max, right_signal)
        active_wink_side = (
            _classify_wink(
                left_signal,
                right_signal,
                closed_threshold=wink_closed_threshold,
                open_threshold=wink_open_threshold,
            )
            if wink_confirm and armed_target_id is not None
            else None
        )
        observed_target = inferencer.infer(observation, targets)
        locked_target = _target_by_id(targets, armed_target_id) if wink_confirm and armed_target_id is not None else None
        target_locked = locked_target is not None
        target = locked_target if target_locked else observed_target

        if target_locked and active_wink_side is None:
            if observed_target is not None and observed_target.target_id != armed_target_id:
                arm_lookaway_elapsed_ms += sample_delta_ms
            else:
                arm_lookaway_elapsed_ms = 0
            if arm_lookaway_elapsed_ms >= max(0, unlock_lookaway_ms):
                armed_target_id = None
                arm_cancel_events += 1
                arm_lookaway_elapsed_ms = 0
                current_target_id = None
                candidate_target_id = None
                candidate_elapsed_ms = 0
                dwell_elapsed_ms = 0
                preview_target = observed_target
                target = observed_target
                locked_target = None
                target_locked = False
                armed_precision_point = None
                left_wink_elapsed_ms = 0
                right_wink_elapsed_ms = 0
                emit(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.TRACKING,
                        message="Canceled the armed target after you looked away. Snap preview is active again.",
                        observation=observation,
                        target=observed_target,
                    )
                )
        else:
            arm_lookaway_elapsed_ms = 0

        if (observation.confidence < min_confidence or observation.x_norm is None or observation.y_norm is None) and not target_locked:
            low_confidence_readings += 1
            current_target_id = None
            dwell_elapsed_ms = 0
            left_wink_elapsed_ms = 0
            right_wink_elapsed_ms = 0
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="Live gaze confidence is too low for cursor snap.",
                    observation=observation,
                    target=target,
                )
            )
            continue

        if target is None or target.bounds is None:
            _hide_point_visualizer(point_visualizer)
            current_target_id = None
            candidate_target_id = None
            candidate_elapsed_ms = 0
            preview_target = None
            preview_point = None
            armed_precision_point = None
            dwell_elapsed_ms = 0
            left_wink_elapsed_ms = 0
            right_wink_elapsed_ms = 0
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="No stable snap target is grounded yet.",
                    observation=observation,
                )
            )
            continue

        if not target_locked:
            if target.target_id == candidate_target_id:
                candidate_elapsed_ms += sample_delta_ms
            else:
                candidate_target_id = target.target_id
                candidate_elapsed_ms = sample_delta_ms

        preview_target_changed = False
        if preview_target is None or preview_target.target_id == target.target_id:
            preview_target_changed = preview_target is None
            preview_target = target
        elif target_locked:
            preview_target = target
        elif candidate_elapsed_ms >= max(0, preview_lock_ms):
            preview_target = target
            preview_target_changed = True
            dwell_elapsed_ms = 0
            current_target_id = None

        assert preview_target.bounds is not None
        refined_target_point = _refine_point_within_bounds(
            observation.x_norm,
            observation.y_norm,
            preview_target.bounds,
        )
        precision_smoothing = max(0.18, min(0.65, float(preview_smoothing) + 0.12))
        if preview_target.target_id == current_target_id:
            dwell_elapsed_ms += sample_delta_ms
        else:
            current_target_id = preview_target.target_id
            dwell_elapsed_ms = sample_delta_ms

        armed_target_active = armed_target_id == preview_target.target_id
        if armed_target_active:
            if armed_precision_point is None:
                armed_precision_point = refined_target_point
            elif active_wink_side is None:
                armed_precision_point = (
                    _lerp(armed_precision_point[0], refined_target_point[0], precision_smoothing),
                    _lerp(armed_precision_point[1], refined_target_point[1], precision_smoothing),
                )
            target_point = armed_precision_point
        else:
            armed_precision_point = None
            if preview_target_changed or preview_point is None:
                preview_point = refined_target_point
            else:
                preview_point = (
                    _lerp(preview_point[0], refined_target_point[0], precision_smoothing),
                    _lerp(preview_point[1], refined_target_point[1], precision_smoothing),
                )
            target_point = preview_point

        preview_frames += 1
        _show_snap_visualizer(
            point_visualizer,
            target_x=target_point[0],
            target_y=target_point[1],
            tracked_x=observation.x_norm,
            tracked_y=observation.y_norm,
            armed=armed_target_active,
            wink_side=active_wink_side if armed_target_active else None,
        )
        if camera_preview is not None:
            _show_camera_preview(
                camera_preview,
                reading,
                calibrated_point=(observation.x_norm, observation.y_norm),
                cursor_point=target_point,
                target_label=preview_target.label,
            )

        if wink_confirm and armed_target_active:
            wink_side = active_wink_side
            left_wink_active = wink_side == "left"
            right_wink_active = wink_side == "right"
            left_wink_elapsed_ms = left_wink_elapsed_ms + sample_delta_ms if left_wink_active else 0
            right_wink_elapsed_ms = right_wink_elapsed_ms + sample_delta_ms if right_wink_active else 0
        else:
            left_wink_elapsed_ms = 0
            right_wink_elapsed_ms = 0

        if cooldown_remaining_ms > 0:
            message = f'Previewing snap target "{preview_target.label}" while cooldown is active.'
            if armed_target_active and second_dwell_action is not None:
                message = f'Previewing armed target "{preview_target.label}" while cooldown is active.'
            elif armed_target_active and wink_confirm:
                message = f'Previewing armed target "{preview_target.label}" while cooldown is active.'
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message=message,
                    observation=observation,
                    target=preview_target,
                )
            )
            continue

        if wink_confirm and armed_target_active:
            wink_action: ActionName | None = None
            wink_message: str | None = None
            if left_wink_elapsed_ms >= max(0, wink_hold_ms):
                wink_action = ActionName.CLICK_TARGET
                wink_message = f'Clicked the armed snap target "{preview_target.label}" with a left wink.'
            elif right_wink_elapsed_ms >= max(0, wink_hold_ms):
                wink_action = ActionName.RIGHT_CLICK_TARGET
                wink_message = f'Right-clicked the armed snap target "{preview_target.label}" with a right wink.'
            if wink_action is not None and wink_message is not None:
                proposal = ActionProposal(
                    action=wink_action,
                    arguments={
                        "target_ref": preview_target.target_id,
                        "target_label": preview_target.label,
                        "target_bounds": preview_target.bounds.model_dump(mode="json"),
                        "normalized_point": {"x": target_point[0], "y": target_point[1]},
                    },
                    confidence=preview_target.confidence,
                    risk=RiskLevel.L2,
                    requires_confirmation=False,
                    rationale="A deliberate one-eye wink confirmed the armed snap target.",
                )
                decision = broker.decide(proposal)
                if decision.decision.value == "confirm":
                    decision = broker.confirm(decision)
                request = broker.build_execution_request(decision, environment)
                result = _execute_pointer_request(request)
                click_events += 1
                left_wink_events += int(wink_action == ActionName.CLICK_TARGET)
                right_wink_events += int(wink_action == ActionName.RIGHT_CLICK_TARGET)
                armed_target_id = None
                cooldown_remaining_ms = max(0, cooldown_ms)
                dwell_elapsed_ms = 0
                current_target_id = None
                candidate_target_id = None
                candidate_elapsed_ms = 0
                left_wink_elapsed_ms = 0
                right_wink_elapsed_ms = 0
                emit(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.TRIGGERED,
                        message=wink_message,
                        observation=observation,
                        target=preview_target,
                        proposal=proposal,
                        result=result,
                    )
                )
                continue
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message=_armed_wink_feedback_message(
                        preview_target.label,
                        wink_side=active_wink_side,
                        candidate_wink_side=(
                            _classify_wink_candidate(
                                left_signal,
                                right_signal,
                                closed_threshold=wink_closed_threshold,
                                open_threshold=wink_open_threshold,
                            )
                            if active_wink_side is None
                            else None
                        ),
                        left_elapsed_ms=left_wink_elapsed_ms,
                        right_elapsed_ms=right_wink_elapsed_ms,
                        wink_hold_ms=max(0, wink_hold_ms),
                        left_signal=left_signal,
                        right_signal=right_signal,
                    ),
                    observation=observation,
                    target=preview_target,
                )
            )
            continue

        if dwell_elapsed_ms < max(0, dwell_ms):
            if wink_confirm and armed_target_active:
                message = (
                    f'Previewing armed target "{preview_target.label}". '
                    "Left wink to click, right wink to right-click, or look away briefly to cancel."
                )
            elif second_dwell_action is not None and armed_target_active:
                action_label = "right-click" if second_dwell_action == ActionName.RIGHT_CLICK_TARGET else "click"
                message = f'Previewing armed target "{preview_target.label}". Dwell again to {action_label}.'
            else:
                message = f'Previewing snap target "{preview_target.label}". Hold steady to snap.'
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message=message,
                    observation=observation,
                    target=preview_target,
                )
            )
            continue

        if second_dwell_action is not None and armed_target_active:
            action_label = "right-click" if second_dwell_action == ActionName.RIGHT_CLICK_TARGET else "click"
            proposal = ActionProposal(
                action=second_dwell_action,
                arguments={
                    "target_ref": preview_target.target_id,
                    "target_label": preview_target.label,
                    "target_bounds": preview_target.bounds.model_dump(mode="json"),
                    "normalized_point": {"x": target_point[0], "y": target_point[1]},
                },
                confidence=preview_target.confidence,
                risk=RiskLevel.L2,
                requires_confirmation=False,
                rationale=f"Second stable dwell on the armed snap target triggered a {action_label}.",
            )
            decision = broker.decide(proposal)
            if decision.decision.value == "confirm":
                decision = broker.confirm(decision)
            request = broker.build_execution_request(decision, environment)
            result = _execute_pointer_request(request)
            click_events += 1
            armed_target_id = None
            cooldown_remaining_ms = max(0, cooldown_ms)
            dwell_elapsed_ms = 0
            current_target_id = None
            candidate_target_id = None
            candidate_elapsed_ms = 0
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRIGGERED,
                    message=f'Second dwell {action_label}ed the armed snap target "{preview_target.label}".',
                    observation=observation,
                    target=preview_target,
                    proposal=proposal,
                    result=result,
                )
            )
            continue

        proposal = ActionProposal(
            action=ActionName.FOCUS_TARGET,
            arguments={
                "target_ref": preview_target.target_id,
                "target_label": preview_target.label,
                "target_bounds": preview_target.bounds.model_dump(mode="json"),
                "normalized_point": {"x": target_point[0], "y": target_point[1]},
            },
            confidence=preview_target.confidence,
            risk=RiskLevel.L1,
            requires_confirmation=False,
            rationale="Stable dwell on a grounded snap target moved the pointer to the refined point inside that target.",
        )
        decision = broker.decide(proposal)
        request = broker.build_execution_request(decision, environment)
        result = _execute_pointer_request(request)
        snap_events += 1
        dwell_elapsed_ms = 0
        current_target_id = None
        if second_dwell_action is not None or wink_confirm:
            armed_target_id = preview_target.target_id
            armed_precision_point = target_point
            if wink_confirm:
                message = (
                    f'Snapped to "{preview_target.label}". '
                    "Left wink to click, right wink to right-click, or look away briefly to cancel."
                )
            else:
                action_label = "right-click" if second_dwell_action == ActionName.RIGHT_CLICK_TARGET else "click"
                message = f'Snapped to "{preview_target.label}". Dwell again on the same target to {action_label}.'
            cooldown_remaining_ms = 0
        else:
            armed_target_id = None
            armed_precision_point = None
            message = f'Snapped cursor to "{preview_target.label}".'
            cooldown_remaining_ms = max(0, cooldown_ms)
        emit(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRIGGERED,
                message=message,
                observation=observation,
                target=preview_target,
                proposal=proposal,
                result=result,
            )
        )

    emit(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
    return events, {
        "attempted_frames": frames,
        "successful_readings": successful_readings,
        "missing_readings": missing_readings,
        "low_confidence_readings": low_confidence_readings,
        "preview_frames": preview_frames,
        "snap_events": snap_events,
        "click_events": click_events,
        "wink_confirm": wink_confirm,
        "wink_signal_available": wink_signal_available,
        "wink_hold_ms": max(0, wink_hold_ms),
        "wink_closed_threshold": max(0.0, min(1.0, float(wink_closed_threshold))),
        "wink_open_threshold": max(0.0, min(1.0, float(wink_open_threshold))),
        "left_wink_events": left_wink_events,
        "right_wink_events": right_wink_events,
        "left_eye_closed_max": left_eye_closed_max,
        "right_eye_closed_max": right_eye_closed_max,
        "arm_cancel_events": arm_cancel_events,
        "armed_target_id": armed_target_id,
        "precision_mode": "armed_refinement",
        "target_grid": _target_grid_dimensions(targets),
        "unlock_lookaway_ms": max(0, unlock_lookaway_ms),
        "preview_lock_ms": max(0, preview_lock_ms),
        "preview_smoothing": max(0.05, min(1.0, float(preview_smoothing))),
    }


def run_live_wink_click_test(
    provider: WebcamProvider,
    *,
    adapter: PlatformAdapter,
    frames: int,
    delta_ms: int,
    environment: EnvironmentSnapshot | None = None,
    min_confidence: float = 0.6,
    wink_hold_ms: int = 180,
    wink_closed_threshold: float = 0.7,
    wink_open_threshold: float = 0.35,
    cooldown_ms: int = 900,
    stop_after_clicks: int = 2,
    point_visualizer: Any | None = None,
    camera_preview: Any | None = None,
    current_point_getter: Callable[[], tuple[float, float]] | None = None,
    event_sink: Callable[[GazeFeedbackEvent], None] | None = None,
) -> tuple[list[GazeFeedbackEvent], dict[str, Any]]:
    if environment is None:
        environment = EnvironmentSnapshot(active_app="Wink Click Test", active_window_title="Wink Click Test")
    current_point_getter = current_point_getter or current_position_normalized
    broker = CommandBroker()

    def _cursor_point() -> tuple[float, float] | None:
        try:
            point = current_point_getter()
        except Exception:
            return None
        if not isinstance(point, tuple) or len(point) != 2:
            return None
        return (_clamp_normalized(float(point[0])), _clamp_normalized(float(point[1])))

    def _execute_pointer_request(request):
        _dismiss_live_visualizers_for_pointer_action(point_visualizer, camera_preview)
        return adapter.execute(request)

    events: list[GazeFeedbackEvent] = []

    def emit(event: GazeFeedbackEvent) -> None:
        events.append(event)
        if event_sink is not None:
            event_sink(event)

    emit(
        GazeFeedbackEvent(
            phase=GazeLoopPhase.TRACKING,
            message=(
                "Wink click test is active. Keep the cursor over a safe target. "
                "Left wink left-clicks. Right wink right-clicks."
            ),
        )
    )
    successful_readings = 0
    missing_readings = 0
    low_confidence_readings = 0
    left_wink_events = 0
    right_wink_events = 0
    left_click_events = 0
    right_click_events = 0
    pointer_lookup_failures = 0
    wink_signal_available = False
    left_eye_closed_max = 0.0
    right_eye_closed_max = 0.0
    cooldown_remaining_ms = 0
    left_wink_elapsed_ms = 0
    right_wink_elapsed_ms = 0
    left_eye_closed_filtered = 0.0
    right_eye_closed_filtered = 0.0
    last_cursor_point: tuple[float, float] | None = None
    stop_after_clicks = max(0, int(stop_after_clicks))
    stopped_after_clicks = False

    for _ in range(frames):
        reading = provider.read(delta_ms=delta_ms)
        if reading is None:
            missing_readings += 1
            _hide_point_visualizer(point_visualizer)
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="No webcam reading is available for wink click testing.",
                )
            )
            continue

        successful_readings += 1
        sample_delta_ms = max(0, reading.sample.delta_ms)
        cooldown_remaining_ms = max(0, cooldown_remaining_ms - sample_delta_ms)
        observation = GazeObservation(
            confidence=reading.sample.confidence,
            x_norm=reading.sample.point.x,
            y_norm=reading.sample.point.y,
            fixation_ms=sample_delta_ms,
        )
        left_present, left_eye_closed = _reading_signal(reading, "left_eye_closed")
        right_present, right_eye_closed = _reading_signal(reading, "right_eye_closed")
        left_eye_closed_filtered = _smooth_signal(
            left_eye_closed_filtered,
            left_eye_closed if left_present else 0.0,
            weight=0.68,
        )
        right_eye_closed_filtered = _smooth_signal(
            right_eye_closed_filtered,
            right_eye_closed if right_present else 0.0,
            weight=0.68,
        )
        wink_signal_available = wink_signal_available or left_present or right_present
        left_signal = max(left_eye_closed_filtered, left_eye_closed)
        right_signal = max(right_eye_closed_filtered, right_eye_closed)
        left_eye_closed_max = max(left_eye_closed_max, left_signal)
        right_eye_closed_max = max(right_eye_closed_max, right_signal)
        active_wink_side = _classify_wink(
            left_signal,
            right_signal,
            closed_threshold=wink_closed_threshold,
            open_threshold=wink_open_threshold,
        )

        needs_cursor_point = point_visualizer is not None or camera_preview is not None or active_wink_side is not None
        if needs_cursor_point:
            cursor_point = _cursor_point()
            if cursor_point is None:
                pointer_lookup_failures += 1
            else:
                last_cursor_point = cursor_point
        else:
            cursor_point = last_cursor_point

        if cursor_point is not None:
            if point_visualizer is not None:
                _show_point_visualizer(point_visualizer, cursor_point[0], cursor_point[1])
                if active_wink_side == "left":
                    try:
                        point_visualizer.set_style(fill="#30d158", outline="#ffffff")
                    except Exception:
                        pass
                elif active_wink_side == "right":
                    try:
                        point_visualizer.set_style(fill="#bf5af2", outline="#ffffff")
                    except Exception:
                        pass
                else:
                    _set_live_control_visualizer_state(point_visualizer, "drive_ready")
        else:
            _hide_point_visualizer(point_visualizer)

        if camera_preview is not None:
            _show_camera_preview(
                camera_preview,
                reading,
                calibrated_point=(observation.x_norm, observation.y_norm),
                cursor_point=cursor_point,
                target_label=(
                    "left wink"
                    if active_wink_side == "left"
                    else "right wink" if active_wink_side == "right" else "wink test"
                ),
            )

        if observation.confidence < min_confidence:
            low_confidence_readings += 1
            left_wink_elapsed_ms = 0
            right_wink_elapsed_ms = 0
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="Webcam confidence is too low for reliable wink testing.",
                    observation=observation,
                )
            )
            continue

        left_wink_elapsed_ms = left_wink_elapsed_ms + sample_delta_ms if active_wink_side == "left" else 0
        right_wink_elapsed_ms = right_wink_elapsed_ms + sample_delta_ms if active_wink_side == "right" else 0

        if cooldown_remaining_ms > 0:
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message="Wink click test cooldown is active.",
                    observation=observation,
                )
            )
            continue

        wink_action: ActionName | None = None
        wink_message: str | None = None
        if left_wink_elapsed_ms >= max(0, wink_hold_ms):
            wink_action = ActionName.CLICK_TARGET
            wink_message = "Detected a deliberate left wink and issued a left click at the current cursor position."
        elif right_wink_elapsed_ms >= max(0, wink_hold_ms):
            wink_action = ActionName.RIGHT_CLICK_TARGET
            wink_message = "Detected a deliberate right wink and issued a right click at the current cursor position."

        if wink_action is not None and wink_message is not None:
            action_point = cursor_point or _cursor_point()
            if action_point is None:
                pointer_lookup_failures += 1
                left_wink_elapsed_ms = 0
                right_wink_elapsed_ms = 0
                emit(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.RECOVERING,
                        message="Detected a wink, but could not read the current cursor position for the click test.",
                        observation=observation,
                    )
                )
                continue
            last_cursor_point = action_point
            proposal = ActionProposal(
                action=wink_action,
                arguments={
                    "normalized_point": {"x": action_point[0], "y": action_point[1]},
                    "target_label": "current cursor position",
                },
                confidence=observation.confidence,
                risk=RiskLevel.L2,
                requires_confirmation=False,
                rationale="A deliberate one-eye wink triggered the dedicated click test at the current cursor position.",
            )
            decision = broker.decide(proposal)
            if decision.decision.value == "confirm":
                decision = broker.confirm(decision)
            request = broker.build_execution_request(decision, environment)
            result = _execute_pointer_request(request)
            left_wink_events += int(wink_action == ActionName.CLICK_TARGET)
            right_wink_events += int(wink_action == ActionName.RIGHT_CLICK_TARGET)
            left_click_events += int(wink_action == ActionName.CLICK_TARGET)
            right_click_events += int(wink_action == ActionName.RIGHT_CLICK_TARGET)
            cooldown_remaining_ms = max(0, cooldown_ms)
            left_wink_elapsed_ms = 0
            right_wink_elapsed_ms = 0
            emit(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRIGGERED,
                    message=wink_message,
                    observation=observation,
                    proposal=proposal,
                    result=result,
                )
            )
            if stop_after_clicks > 0 and (left_click_events + right_click_events) >= stop_after_clicks:
                stopped_after_clicks = True
                break
            continue

        emit(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRACKING,
                message=_wink_test_feedback_message(
                    active_wink_side=active_wink_side,
                    candidate_wink_side=(
                        _classify_wink_candidate(
                            left_signal,
                            right_signal,
                            closed_threshold=wink_closed_threshold,
                            open_threshold=wink_open_threshold,
                        )
                        if active_wink_side is None
                        else None
                    ),
                    left_elapsed_ms=left_wink_elapsed_ms,
                    right_elapsed_ms=right_wink_elapsed_ms,
                    wink_hold_ms=max(0, wink_hold_ms),
                    left_signal=left_signal,
                    right_signal=right_signal,
                ),
                observation=observation,
            )
        )

    emit(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
    return events, {
        "attempted_frames": frames,
        "successful_readings": successful_readings,
        "missing_readings": missing_readings,
        "low_confidence_readings": low_confidence_readings,
        "wink_signal_available": wink_signal_available,
        "wink_hold_ms": max(0, wink_hold_ms),
        "wink_closed_threshold": max(0.0, min(1.0, float(wink_closed_threshold))),
        "wink_open_threshold": max(0.0, min(1.0, float(wink_open_threshold))),
        "left_wink_events": left_wink_events,
        "right_wink_events": right_wink_events,
        "left_eye_closed_max": left_eye_closed_max,
        "right_eye_closed_max": right_eye_closed_max,
        "left_click_events": left_click_events,
        "right_click_events": right_click_events,
        "pointer_lookup_failures": pointer_lookup_failures,
        "last_cursor_point": (
            {"x": last_cursor_point[0], "y": last_cursor_point[1]}
            if last_cursor_point is not None
            else None
        ),
        "cooldown_ms": max(0, cooldown_ms),
        "stop_after_clicks": stop_after_clicks,
        "stopped_after_clicks": stopped_after_clicks,
    }


def run_live_relative_cursor_follow(
    provider: WebcamProvider,
    profile: CalibrationProfile,
    *,
    adapter: PlatformAdapter,
    frames: int,
    delta_ms: int,
    environment: EnvironmentSnapshot | None = None,
    min_confidence: float = 0.6,
    min_move_distance: float = 0.08,
    edge_padding: float = 0.05,
    max_step: float = 0.04,
    response_gain_x: float = 0.25,
    response_gain_y: float = 0.5,
    neutral_frames: int = 6,
    recenter_ms: int = 320,
    fixation_distance: float = 0.015,
    click_on_dwell: bool = False,
    dwell_ms: int = 700,
    click_cooldown_ms: int = 900,
    smoother: GazeSmoother | None = None,
    point_visualizer: Any | None = None,
    initial_cursor_point: tuple[float, float] | None = None,
    camera_preview: Any | None = None,
) -> tuple[list[GazeFeedbackEvent], dict[str, Any]]:
    del profile  # Relative mode uses raw webcam gaze around a neutral point.
    if environment is None:
        environment = EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera")
    smoother = smoother or GazeSmoother(window_size=4)
    broker = CommandBroker()

    anchor_point = (
        _clamp_with_padding(initial_cursor_point[0], edge_padding),
        _clamp_with_padding(initial_cursor_point[1], edge_padding),
    ) if initial_cursor_point is not None else (0.5, 0.5)

    neutral_aggregate, neutral_quality, posture_baseline = _capture_average_raw_sample_stats(
        provider,
        required_frames=max(1, neutral_frames),
        max_attempts=max(1, neutral_frames) * 4,
        delta_ms=delta_ms,
    )
    if neutral_aggregate is None:
        neutral_point = (0.5, 0.5)
        neutral_used_frames = 0
        neutral_attempted_frames = max(1, neutral_frames) * 4
        neutral_confidence = 0.0
        neutral_quality = {
            "sample_count": 0,
            "raw_x_span": 0.0,
            "raw_y_span": 0.0,
            "raw_x_mean": neutral_point[0],
            "raw_y_mean": neutral_point[1],
        }
    else:
        neutral_point = (
            _clamp_with_padding(neutral_aggregate.sample.point.x, edge_padding),
            _clamp_with_padding(neutral_aggregate.sample.point.y, edge_padding),
        )
        neutral_used_frames = neutral_aggregate.used_frames
        neutral_attempted_frames = neutral_aggregate.attempted_frames
        neutral_confidence = neutral_aggregate.sample.confidence

    effective_deadzone_x = _adaptive_relative_deadzone(
        base_deadzone=min_move_distance,
        observed_span=float(neutral_quality["raw_x_span"]),
    )
    effective_deadzone_y = _adaptive_relative_deadzone(
        base_deadzone=min_move_distance,
        observed_span=float(neutral_quality["raw_y_span"]),
    )
    fixation_radius = max(
        0.0,
        max(
            fixation_distance,
            min(effective_deadzone_x, effective_deadzone_y) * 0.75,
        ),
    )

    cursor = RelativeCursorController(
        deadzone=min_move_distance,
        deadzone_x=effective_deadzone_x,
        deadzone_y=effective_deadzone_y,
        edge_padding=edge_padding,
        max_step=max_step,
        speed_gain_x=response_gain_x,
        speed_gain_y=response_gain_y,
        neutral_point=neutral_point,
        dispatched_point=anchor_point,
    )

    events: list[GazeFeedbackEvent] = [
        GazeFeedbackEvent(
            phase=GazeLoopPhase.CALIBRATING,
            message=(
                "Established neutral gaze point for relative cursor mode. "
                f'Neutral point: x={neutral_point[0]:.3f}, y={neutral_point[1]:.3f}.'
            ),
        )
    ]
    successful_readings = 0
    missing_readings = 0
    moved_events = 0
    low_confidence_readings = 0
    held_events = 0
    click_events = 0
    recenter_events = 0
    stationary_elapsed_ms = 0
    click_cooldown_remaining_ms = 0
    click_armed = False
    fixation_elapsed_ms = 0
    fixation_points: list[tuple[tuple[float, float], int]] = []
    posture_signal_available = posture_baseline is not None
    posture_frozen = False
    posture_freeze_events = 0
    posture_recenter_events = 0
    posture_points: list[tuple[PostureState, int]] = []
    head_assist_x = 0.0
    head_assist_y = 0.0
    head_assist_active_frames = 0
    max_head_assist_x = 0.0
    max_head_assist_y = 0.0
    bias_corrector = OnlineBiasCorrector()
    bias_update_events = 0

    if point_visualizer is not None and cursor.dispatched_point is not None:
        _show_point_visualizer(point_visualizer, cursor.dispatched_point[0], cursor.dispatched_point[1])

    for _ in range(frames):
        reading = provider.read(delta_ms=delta_ms)
        if reading is None:
            missing_readings += 1
            _hide_point_visualizer(point_visualizer)
            fixation_points = []
            fixation_elapsed_ms = 0
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="No coarse gaze reading from webcam provider."))
            continue

        successful_readings += 1
        raw_point_unbiased = (
            _clamp_with_padding(reading.sample.point.x, edge_padding),
            _clamp_with_padding(reading.sample.point.y, edge_padding),
        )
        raw_point = bias_corrector.apply(raw_point_unbiased, edge_padding=edge_padding)
        current_posture = _reading_posture_state(reading)
        posture_signal_available = posture_signal_available or current_posture is not None
        if posture_baseline is None and current_posture is not None:
            posture_baseline = current_posture
        observation = smoother.smooth(
            GazeSample(
                point=NormalizedPoint(raw_point[0], raw_point[1]),
                confidence=reading.sample.confidence,
                delta_ms=reading.sample.delta_ms,
            )
        )
        sample_delta_ms = max(0, reading.sample.delta_ms)
        click_cooldown_remaining_ms = max(0, click_cooldown_remaining_ms - sample_delta_ms)
        if observation.confidence < min_confidence or observation.x_norm is None or observation.y_norm is None:
            low_confidence_readings += 1
            stationary_elapsed_ms = 0
            fixation_elapsed_ms = 0
            fixation_points = []
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="Live gaze confidence is too low for relative cursor follow.",
                    observation=observation,
                )
            )
            continue

        if current_posture is not None and posture_baseline is not None:
            if posture_frozen:
                posture_points, posture_elapsed_ms = _update_posture_window(
                    posture_points,
                    current_posture,
                    sample_delta_ms,
                    translation_threshold=POSTURE_STABLE_TRANSLATION,
                    size_threshold=POSTURE_STABLE_SIZE,
                    yaw_threshold=POSTURE_STABLE_YAW,
                    pitch_threshold=POSTURE_STABLE_PITCH,
                    roll_threshold=POSTURE_STABLE_ROLL,
                )
                if posture_elapsed_ms >= POSTURE_STABILIZE_MS:
                    posture_baseline = _average_posture_states(state for state, _ in posture_points) or current_posture
                    cursor.recenter(raw_point)
                    reset_smoother = getattr(smoother, "reset", None)
                    if callable(reset_smoother):
                        reset_smoother()
                    posture_frozen = False
                    posture_points = []
                    head_assist_x = 0.0
                    head_assist_y = 0.0
                    bias_corrector.reset()
                    posture_recenter_events += 1
                    stationary_elapsed_ms = 0
                    fixation_elapsed_ms = 0
                    fixation_points = []
                    click_armed = False
                    events.append(
                        GazeFeedbackEvent(
                            phase=GazeLoopPhase.TRACKING,
                            message="Head posture stabilized. Recentered relative cursor control at the new pose.",
                            observation=observation,
                        )
                    )
                else:
                    held_events += 1
                    events.append(
                        GazeFeedbackEvent(
                            phase=GazeLoopPhase.TRACKING,
                            message="Detected a larger head movement. Holding cursor motion until posture stabilizes.",
                            observation=observation,
                        )
                    )
                continue
            if _is_large_posture_change(current_posture, posture_baseline):
                cursor.recenter(cursor.neutral_point, weight=0.0)
                reset_smoother = getattr(smoother, "reset", None)
                if callable(reset_smoother):
                    reset_smoother()
                posture_frozen = True
                posture_freeze_events += 1
                posture_points = [(current_posture, sample_delta_ms)]
                head_assist_x = 0.0
                head_assist_y = 0.0
                bias_corrector.reset()
                stationary_elapsed_ms = 0
                fixation_elapsed_ms = 0
                fixation_points = []
                click_armed = False
                held_events += 1
                events.append(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.TRACKING,
                        message="Detected a larger head movement. Holding cursor motion until posture stabilizes.",
                        observation=observation,
                    )
                )
                continue

        control_point = (observation.x_norm, observation.y_norm)
        fixation_points, fixation_elapsed_ms = _update_fixation_window(
            fixation_points,
            raw_point,
            sample_delta_ms,
            radius=fixation_radius,
        )

        raw_control_x, raw_control_y = _relative_controls(
            control_point[0],
            control_point[1],
            neutral_point=cursor.neutral_point,
            deadzone_x=effective_deadzone_x,
            deadzone_y=effective_deadzone_y,
        )
        target_head_assist_x, target_head_assist_y = _head_assist_targets(current_posture, posture_baseline)
        gated_target_head_assist_x = _gated_head_assist_component(raw_control_x, target_head_assist_x)
        gated_target_head_assist_y = _gated_head_assist_component(raw_control_y, target_head_assist_y)
        head_assist_x = _lerp(head_assist_x, gated_target_head_assist_x, HEAD_ASSIST_FILTER_WEIGHT) if abs(gated_target_head_assist_x) >= 1e-6 else 0.0
        head_assist_y = _lerp(head_assist_y, gated_target_head_assist_y, HEAD_ASSIST_FILTER_WEIGHT) if abs(gated_target_head_assist_y) >= 1e-6 else 0.0
        max_head_assist_x = max(max_head_assist_x, abs(head_assist_x))
        max_head_assist_y = max(max_head_assist_y, abs(head_assist_y))
        combined_control_x = _clamp_signed(raw_control_x + head_assist_x)
        combined_control_y = _clamp_signed(raw_control_y + head_assist_y)
        combined_within_deadzone = abs(combined_control_x) < 1e-6 and abs(combined_control_y) < 1e-6
        if max(abs(head_assist_x), abs(head_assist_y)) >= HEAD_ASSIST_ACTIVE_THRESHOLD:
            head_assist_active_frames += 1
        if (
            fixation_elapsed_ms >= max(0, recenter_ms)
            and math.dist(raw_point, cursor.neutral_point) > max(fixation_radius, min(effective_deadzone_x, effective_deadzone_y) * 0.75)
        ):
            cursor.recenter(raw_point)
            recenter_events += 1
            stationary_elapsed_ms = 0
            fixation_points = []
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message="Stable fixation recentered relative cursor control at the new look point.",
                    observation=observation,
                )
            )
            continue

        if combined_within_deadzone:
            bias_updated = bias_corrector.maybe_update(
                raw_point_unbiased,
                reference_point=cursor.neutral_point,
                stable_ms=fixation_elapsed_ms,
                sample_delta_ms=sample_delta_ms,
                learn_window_x=max(fixation_radius, effective_deadzone_x * ONLINE_BIAS_LEARN_WINDOW_SCALE),
                learn_window_y=max(fixation_radius, effective_deadzone_y * ONLINE_BIAS_LEARN_WINDOW_SCALE),
            )
            if bias_updated:
                bias_update_events += 1

        point = cursor.next_point_from_controls(combined_control_x, combined_control_y)
        visual_point = point if point is not None else cursor.dispatched_point
        if point_visualizer is not None and visual_point is not None:
            _show_point_visualizer(point_visualizer, visual_point[0], visual_point[1])
        if camera_preview is not None:
            _show_camera_preview(
                camera_preview,
                reading,
                calibrated_point=(observation.x_norm, observation.y_norm) if observation.x_norm is not None and observation.y_norm is not None else None,
                cursor_point=visual_point,
                target_label="relative-cursor",
            )
        if combined_within_deadzone and click_on_dwell and click_armed and cursor.dispatched_point is not None:
            stationary_elapsed_ms += sample_delta_ms
        else:
            stationary_elapsed_ms = 0
        if point is None:
            held_events += 1
            if combined_within_deadzone and click_on_dwell and click_armed and cursor.dispatched_point is not None:
                if stationary_elapsed_ms >= max(0, dwell_ms) and click_cooldown_remaining_ms <= 0:
                    proposal = ActionProposal(
                        action=ActionName.CLICK_TARGET,
                        arguments={
                            "normalized_point": {"x": cursor.dispatched_point[0], "y": cursor.dispatched_point[1]},
                            "control_mode": "relative",
                            "neutral_point": {"x": neutral_point[0], "y": neutral_point[1]},
                            "trigger": "dwell",
                        },
                        confidence=observation.confidence,
                        risk=RiskLevel.L2,
                        requires_confirmation=False,
                        rationale="Relative gaze mode held the cursor steady long enough to trigger a dwell click.",
                    )
                    decision = broker.decide(proposal)
                    if decision.decision.value == "confirm":
                        decision = broker.confirm(decision)
                    request = broker.build_execution_request(decision, environment)
                    result = adapter.execute(request)
                    click_events += 1
                    click_armed = False
                    stationary_elapsed_ms = 0
                    click_cooldown_remaining_ms = max(0, click_cooldown_ms)
                    events.append(
                        GazeFeedbackEvent(
                            phase=GazeLoopPhase.TRIGGERED,
                            message="Stable dwell triggered a relative gaze click action.",
                            observation=observation,
                            proposal=proposal,
                            result=result,
                        )
                    )
                    continue
            message = "Live relative gaze cursor hold is within the movement deadzone."
            if not combined_within_deadzone:
                message = "Live relative gaze input did not produce additional cursor movement."
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message=message,
                    observation=observation,
                )
            )
            continue

        stationary_elapsed_ms = 0
        click_armed = True
        proposal = ActionProposal(
            action=ActionName.FOCUS_TARGET,
            arguments={
                "normalized_point": {"x": point[0], "y": point[1]},
                "control_mode": "relative",
                "neutral_point": {"x": neutral_point[0], "y": neutral_point[1]},
            },
            confidence=observation.confidence,
            risk=RiskLevel.L1,
            requires_confirmation=False,
            rationale="Live relative gaze mode nudged the pointer based on gaze direction with small head movement assist around the neutral pose.",
        )
        decision = broker.decide(proposal)
        request = broker.build_execution_request(decision, environment)
        result = adapter.execute(request)
        moved_events += 1
        events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRACKING,
                message="Moved cursor with live relative gaze control.",
                observation=observation,
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
        "click_events": click_events,
        "recenter_events": recenter_events,
        "low_confidence_readings": low_confidence_readings,
        "neutral_point": {"x": neutral_point[0], "y": neutral_point[1]},
        "final_neutral_point": {"x": cursor.neutral_point[0], "y": cursor.neutral_point[1]},
        "neutral_used_frames": neutral_used_frames,
        "neutral_attempted_frames": neutral_attempted_frames,
        "neutral_confidence": neutral_confidence,
        "effective_deadzone": {
            "x": effective_deadzone_x,
            "y": effective_deadzone_y,
        },
        "neutral_quality": neutral_quality,
        "posture_signal_available": posture_signal_available,
        "posture_freeze_events": posture_freeze_events,
        "posture_recenter_events": posture_recenter_events,
        "head_assist_active_frames": head_assist_active_frames,
        "max_head_assist": {
            "x": max_head_assist_x,
            "y": max_head_assist_y,
        },
        "online_bias_correction": {
            "enabled": True,
            "updates": bias_update_events,
            "active_ms": bias_corrector.active_ms,
            "offset": {
                "x": bias_corrector.offset_x,
                "y": bias_corrector.offset_y,
            },
        },
    }


def run_live_assisted_cursor_follow(
    provider: WebcamProvider,
    profile: CalibrationProfile,
    *,
    adapter: PlatformAdapter,
    frames: int,
    delta_ms: int,
    environment: EnvironmentSnapshot | None = None,
    min_confidence: float = 0.6,
    min_move_distance: float = 0.08,
    edge_padding: float = 0.05,
    max_step: float = 0.04,
    response_gain_x: float = 0.25,
    response_gain_y: float = 0.5,
    control_signal_name: str = "mouth_open",
    control_signal_threshold: float = 0.18,
    arm_hold_ms: int = 100,
    arm_settle_ms: int = 100,
    drive_commit_ms: int = 140,
    idle_disarm_ms: int = 1200,
    recenter_ms: int = 200,
    fixation_distance: float = 0.015,
    click_on_dwell: bool = False,
    dwell_ms: int = 700,
    click_cooldown_ms: int = 900,
    smoother: GazeSmoother | None = None,
    point_visualizer: Any | None = None,
    initial_cursor_point: tuple[float, float] | None = None,
    camera_preview: Any | None = None,
) -> tuple[list[GazeFeedbackEvent], dict[str, Any]]:
    del profile  # Assisted mode uses raw webcam gaze around a local neutral point.
    if environment is None:
        environment = EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera")
    smoother = smoother or GazeSmoother()
    broker = CommandBroker()

    anchor_point = (
        _clamp_with_padding(initial_cursor_point[0], edge_padding),
        _clamp_with_padding(initial_cursor_point[1], edge_padding),
    ) if initial_cursor_point is not None else (0.5, 0.5)
    cursor = RelativeCursorController(
        deadzone=min_move_distance,
        edge_padding=edge_padding,
        max_step=max_step,
        speed_gain_x=response_gain_x,
        speed_gain_y=response_gain_y,
        neutral_point=anchor_point,
        dispatched_point=anchor_point,
    )

    events: list[GazeFeedbackEvent] = [
        GazeFeedbackEvent(
            phase=GazeLoopPhase.CALIBRATING,
            message=(
                "Passive look mode is active. Open your mouth slightly to steer the cursor. "
                "Keep it open while nudging, then let your gaze settle to stop."
            ),
        )
    ]
    successful_readings = 0
    missing_readings = 0
    moved_events = 0
    held_events = 0
    click_events = 0
    low_confidence_readings = 0
    passive_frames = 0
    active_frames = 0
    engaged_events = 0
    released_events = 0
    recenter_events = 0
    signal_available = False
    max_control_signal = 0.0
    control_hold_ms = 0
    stationary_elapsed_ms = 0
    click_cooldown_remaining_ms = 0
    arm_settle_remaining_ms = 0
    fixation_elapsed_ms = 0
    clutch_active = False
    movement_since_activation = False
    rearm_blocked_until_signal_drop = False
    last_raw_point: tuple[float, float] | None = None
    posture_baseline: PostureState | None = None
    posture_signal_available = False
    posture_frozen = False
    posture_freeze_events = 0
    posture_recenter_events = 0
    posture_points: list[tuple[PostureState, int]] = []
    head_assist_x = 0.0
    head_assist_y = 0.0
    head_assist_active_frames = 0
    max_head_assist_x = 0.0
    max_head_assist_y = 0.0
    drive_ready_frames = 0
    drive_frames = 0
    armed_frames = 0
    paused_frames = 0
    auto_disarm_events = 0
    candidate_direction: tuple[int, int] | None = None
    candidate_elapsed_ms = 0
    control_state = "observe"
    bias_corrector = OnlineBiasCorrector()
    bias_update_events = 0

    if point_visualizer is not None and cursor.dispatched_point is not None:
        _set_live_control_visualizer_state(point_visualizer, control_state)
        _show_point_visualizer(point_visualizer, cursor.dispatched_point[0], cursor.dispatched_point[1])

    for _ in range(frames):
        reading = provider.read(delta_ms=delta_ms)
        if reading is None:
            missing_readings += 1
            control_state = "paused"
            paused_frames += 1
            _set_live_control_visualizer_state(point_visualizer, control_state)
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="No coarse gaze reading from webcam provider."))
            continue

        successful_readings += 1
        raw_point_unbiased = (
            _clamp_with_padding(reading.sample.point.x, edge_padding),
            _clamp_with_padding(reading.sample.point.y, edge_padding),
        )
        current_posture = _reading_posture_state(reading)
        posture_signal_available = posture_signal_available or current_posture is not None
        sample_delta_ms = max(0, reading.sample.delta_ms)
        click_cooldown_remaining_ms = max(0, click_cooldown_remaining_ms - sample_delta_ms)
        signal_present, control_signal = _reading_signal(reading, control_signal_name)
        signal_available = signal_available or signal_present
        max_control_signal = max(max_control_signal, control_signal)
        signal_active = control_signal >= max(0.0, min(1.0, control_signal_threshold))
        corrected_point = (
            bias_corrector.apply(raw_point_unbiased, edge_padding=edge_padding)
            if clutch_active or signal_active
            else raw_point_unbiased
        )
        observation = smoother.smooth(
            GazeSample(
                point=NormalizedPoint(corrected_point[0], corrected_point[1]),
                confidence=reading.sample.confidence,
                delta_ms=reading.sample.delta_ms,
            )
        )

        if not signal_active:
            rearm_blocked_until_signal_drop = False

        if observation.confidence < min_confidence or observation.x_norm is None or observation.y_norm is None:
            low_confidence_readings += 1
            stationary_elapsed_ms = 0
            fixation_elapsed_ms = 0
            control_state = "paused"
            paused_frames += 1
            _set_live_control_visualizer_state(point_visualizer, control_state)
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="Live gaze confidence is too low for assisted cursor follow.",
                    observation=observation,
                )
            )
            continue

        if camera_preview is not None:
            state_label = f"assist-{control_state}"
            _show_camera_preview(
                camera_preview,
                reading,
                calibrated_point=(observation.x_norm, observation.y_norm),
                cursor_point=cursor.dispatched_point,
                target_label=state_label,
            )

        if rearm_blocked_until_signal_drop and signal_active:
            passive_frames += 1
            control_state = "observe"
            _set_live_control_visualizer_state(point_visualizer, control_state)
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message="Release the mouth-open control signal to rearm assisted cursor control.",
                    observation=observation,
                )
            )
            continue

        if signal_active:
            control_hold_ms += sample_delta_ms
            if not clutch_active and control_hold_ms >= max(0, arm_hold_ms):
                clutch_active = True
                engaged_events += 1
                movement_since_activation = False
                stationary_elapsed_ms = 0
                fixation_elapsed_ms = 0
                candidate_direction = None
                candidate_elapsed_ms = 0
                arm_settle_remaining_ms = max(0, arm_settle_ms)
                posture_frozen = False
                posture_points = []
                posture_baseline = current_posture
                bias_corrector.reset()
                last_raw_point = corrected_point
                cursor.neutral_point = last_raw_point
                control_state = "drive_ready"
                drive_ready_frames += 1
                _set_live_control_visualizer_state(point_visualizer, control_state)
                events.append(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.TRIGGERED,
                        message="Assisted cursor control engaged from the mouth-open signal.",
                        observation=observation,
                    )
                )
                continue
        else:
            control_hold_ms = 0
            if clutch_active:
                clutch_active = False
                released_events += 1
                arm_settle_remaining_ms = 0
                fixation_elapsed_ms = 0
                stationary_elapsed_ms = 0
                movement_since_activation = False
                last_raw_point = None
                head_assist_x = 0.0
                head_assist_y = 0.0
                bias_corrector.reset()
                posture_frozen = False
                posture_points = []
                posture_baseline = None
                candidate_direction = None
                candidate_elapsed_ms = 0
                control_state = "observe"
                _set_live_control_visualizer_state(point_visualizer, control_state)
                events.append(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.TRACKING,
                        message="Assisted cursor control released. Passive look mode resumed.",
                        observation=observation,
                    )
                )
                continue

            passive_frames += 1
            control_state = "observe"
            _set_live_control_visualizer_state(point_visualizer, control_state)
            message = "Passive look mode is active."
            if not signal_available:
                message = (
                    f'Passive look mode is active. The current tracker has not exposed a "{control_signal_name}" control signal yet.'
                )
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message=message,
                    observation=observation,
                )
            )
            continue

        active_frames += 1
        raw_point = corrected_point
        if posture_baseline is None and current_posture is not None:
            posture_baseline = current_posture

        if arm_settle_remaining_ms > 0:
            arm_settle_remaining_ms = max(0, arm_settle_remaining_ms - sample_delta_ms)
            last_raw_point = raw_point
            control_state = "drive_ready"
            drive_ready_frames += 1
            _set_live_control_visualizer_state(point_visualizer, control_state)
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message="Assisted cursor control is waiting for the first eye movement to settle.",
                    observation=observation,
                )
            )
            continue

        if current_posture is not None and posture_baseline is not None:
            if posture_frozen:
                posture_points, posture_elapsed_ms = _update_posture_window(
                    posture_points,
                    current_posture,
                    sample_delta_ms,
                    translation_threshold=POSTURE_STABLE_TRANSLATION,
                    size_threshold=POSTURE_STABLE_SIZE,
                    yaw_threshold=POSTURE_STABLE_YAW,
                    pitch_threshold=POSTURE_STABLE_PITCH,
                    roll_threshold=POSTURE_STABLE_ROLL,
                )
                if posture_elapsed_ms >= POSTURE_STABILIZE_MS:
                    posture_baseline = _average_posture_states(state for state, _ in posture_points) or current_posture
                    cursor.recenter(raw_point)
                    reset_smoother = getattr(smoother, "reset", None)
                    if callable(reset_smoother):
                        reset_smoother()
                    posture_frozen = False
                    posture_points = []
                    head_assist_x = 0.0
                    head_assist_y = 0.0
                    bias_corrector.reset()
                    posture_recenter_events += 1
                    movement_since_activation = False
                    stationary_elapsed_ms = 0
                    fixation_elapsed_ms = 0
                    candidate_direction = None
                    candidate_elapsed_ms = 0
                    control_state = "drive_ready"
                    drive_ready_frames += 1
                    _set_live_control_visualizer_state(point_visualizer, control_state)
                    events.append(
                        GazeFeedbackEvent(
                            phase=GazeLoopPhase.TRACKING,
                            message="Head posture stabilized. Recentered assisted cursor control at the new pose.",
                            observation=observation,
                        )
                    )
                else:
                    held_events += 1
                    control_state = "paused"
                    paused_frames += 1
                    _set_live_control_visualizer_state(point_visualizer, control_state)
                    events.append(
                        GazeFeedbackEvent(
                            phase=GazeLoopPhase.TRACKING,
                            message="Detected a larger head movement. Holding assisted cursor motion until posture stabilizes.",
                            observation=observation,
                        )
                    )
                continue
            if _is_large_posture_change(current_posture, posture_baseline):
                cursor.recenter(cursor.neutral_point, weight=0.0)
                reset_smoother = getattr(smoother, "reset", None)
                if callable(reset_smoother):
                    reset_smoother()
                posture_frozen = True
                posture_freeze_events += 1
                posture_points = [(current_posture, sample_delta_ms)]
                head_assist_x = 0.0
                head_assist_y = 0.0
                bias_corrector.reset()
                movement_since_activation = False
                stationary_elapsed_ms = 0
                fixation_elapsed_ms = 0
                candidate_direction = None
                candidate_elapsed_ms = 0
                held_events += 1
                control_state = "paused"
                paused_frames += 1
                _set_live_control_visualizer_state(point_visualizer, control_state)
                events.append(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.TRACKING,
                        message="Detected a larger head movement. Holding assisted cursor motion until posture stabilizes.",
                        observation=observation,
                    )
                )
                continue

        if last_raw_point is not None and math.dist(raw_point, last_raw_point) <= max(0.0, fixation_distance):
            fixation_elapsed_ms += sample_delta_ms
        else:
            fixation_elapsed_ms = sample_delta_ms
        last_raw_point = raw_point

        if (
            movement_since_activation
            and fixation_elapsed_ms >= max(0, recenter_ms)
            and math.dist(raw_point, cursor.neutral_point) > max(0.0, fixation_distance)
        ):
            cursor.recenter(raw_point, weight=1.0)
            reset_smoother = getattr(smoother, "reset", None)
            if callable(reset_smoother):
                reset_smoother()
            head_assist_x = 0.0
            head_assist_y = 0.0
            stationary_elapsed_ms = 0
            fixation_elapsed_ms = 0
            bias_corrector.reset()
            recenter_events += 1
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message="Stable fixation recentered assisted cursor control at the new look point.",
                    observation=observation,
                )
            )
            continue

        smoothed_control_x, smoothed_control_y = _relative_controls(
            observation.x_norm,
            observation.y_norm,
            neutral_point=cursor.neutral_point,
            deadzone_x=cursor.deadzone_x if cursor.deadzone_x is not None else cursor.deadzone,
            deadzone_y=cursor.deadzone_y if cursor.deadzone_y is not None else cursor.deadzone,
        )
        raw_control_x, raw_control_y = _relative_controls(
            raw_point[0],
            raw_point[1],
            neutral_point=cursor.neutral_point,
            deadzone_x=cursor.deadzone_x if cursor.deadzone_x is not None else cursor.deadzone,
            deadzone_y=cursor.deadzone_y if cursor.deadzone_y is not None else cursor.deadzone,
        )
        target_head_assist_x, target_head_assist_y = _head_assist_targets(current_posture, posture_baseline)
        gated_target_head_assist_x = _gated_head_assist_component(smoothed_control_x, target_head_assist_x)
        gated_target_head_assist_y = _gated_head_assist_component(smoothed_control_y, target_head_assist_y)
        head_assist_x = _lerp(head_assist_x, gated_target_head_assist_x, HEAD_ASSIST_FILTER_WEIGHT) if abs(gated_target_head_assist_x) >= 1e-6 else 0.0
        head_assist_y = _lerp(head_assist_y, gated_target_head_assist_y, HEAD_ASSIST_FILTER_WEIGHT) if abs(gated_target_head_assist_y) >= 1e-6 else 0.0
        max_head_assist_x = max(max_head_assist_x, abs(head_assist_x))
        max_head_assist_y = max(max_head_assist_y, abs(head_assist_y))
        combined_control_x = _clamp_signed(smoothed_control_x + head_assist_x)
        combined_control_y = _clamp_signed(smoothed_control_y + head_assist_y)
        raw_neutral_radius = max(0.0, max(fixation_distance, min_move_distance * 0.8))
        raw_within_neutral = math.dist(raw_point, cursor.neutral_point) <= raw_neutral_radius
        if movement_since_activation and raw_within_neutral:
            combined_control_x = 0.0
            combined_control_y = 0.0
            head_assist_x = 0.0
            head_assist_y = 0.0
            cursor.velocity_x = 0.0
            cursor.velocity_y = 0.0
        combined_within_deadzone = abs(combined_control_x) < 1e-6 and abs(combined_control_y) < 1e-6
        if max(abs(head_assist_x), abs(head_assist_y)) >= HEAD_ASSIST_ACTIVE_THRESHOLD:
            head_assist_active_frames += 1

        if raw_within_neutral and combined_within_deadzone:
            bias_updated = bias_corrector.maybe_update(
                raw_point_unbiased,
                reference_point=cursor.neutral_point,
                stable_ms=fixation_elapsed_ms,
                sample_delta_ms=sample_delta_ms,
                learn_window_x=max(fixation_distance, min_move_distance * ONLINE_BIAS_LEARN_WINDOW_SCALE),
                learn_window_y=max(fixation_distance, min_move_distance * ONLINE_BIAS_LEARN_WINDOW_SCALE),
            )
            if bias_updated:
                bias_update_events += 1

        if not movement_since_activation:
            current_direction = _control_direction(raw_control_x, raw_control_y)
            if current_direction is None:
                candidate_direction = None
                candidate_elapsed_ms = 0
                control_state = "drive_ready"
                drive_ready_frames += 1
                _set_live_control_visualizer_state(point_visualizer, control_state)
                if point_visualizer is not None and cursor.dispatched_point is not None:
                    _show_point_visualizer(point_visualizer, cursor.dispatched_point[0], cursor.dispatched_point[1])
                events.append(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.TRACKING,
                        message="Drive mode is armed. Hold the same direction briefly to move the cursor.",
                        observation=observation,
                    )
                )
                continue
            if current_direction == candidate_direction:
                candidate_elapsed_ms += sample_delta_ms
            else:
                candidate_direction = current_direction
                candidate_elapsed_ms = sample_delta_ms
            if candidate_elapsed_ms < max(0, drive_commit_ms):
                control_state = "drive_ready"
                drive_ready_frames += 1
                _set_live_control_visualizer_state(point_visualizer, control_state)
                if point_visualizer is not None and cursor.dispatched_point is not None:
                    _show_point_visualizer(point_visualizer, cursor.dispatched_point[0], cursor.dispatched_point[1])
                events.append(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.TRACKING,
                        message="Drive mode is armed. Hold the same direction briefly to move the cursor.",
                        observation=observation,
                    )
                )
                continue

        point = cursor.next_point_from_controls(combined_control_x, combined_control_y)
        visual_point = point if point is not None else cursor.dispatched_point
        if click_on_dwell and movement_since_activation and point is None and combined_within_deadzone:
            control_state = "armed"
            armed_frames += 1
        elif point is not None:
            control_state = "drive"
            drive_frames += 1
        else:
            control_state = "drive_ready"
            drive_ready_frames += 1
        _set_live_control_visualizer_state(point_visualizer, control_state)
        if point_visualizer is not None and visual_point is not None:
            _show_point_visualizer(point_visualizer, visual_point[0], visual_point[1])
        if camera_preview is not None:
            _show_camera_preview(
                camera_preview,
                reading,
                calibrated_point=(observation.x_norm, observation.y_norm),
                cursor_point=visual_point,
                target_label=f"assist-{control_state}",
            )
        if (
            click_on_dwell
            and movement_since_activation
            and cursor.dispatched_point is not None
            and raw_within_neutral
        ):
            stationary_elapsed_ms += sample_delta_ms
        elif not click_on_dwell and movement_since_activation and raw_within_neutral:
            stationary_elapsed_ms += sample_delta_ms
        elif point is not None:
            stationary_elapsed_ms = 0
        else:
            stationary_elapsed_ms = 0

        if (
            click_on_dwell
            and movement_since_activation
            and cursor.dispatched_point is not None
            and raw_within_neutral
            and stationary_elapsed_ms >= max(0, dwell_ms)
            and click_cooldown_remaining_ms <= 0
        ):
            cursor.recenter(cursor.neutral_point, weight=0.0)
            proposal = ActionProposal(
                action=ActionName.CLICK_TARGET,
                arguments={
                    "normalized_point": {"x": cursor.dispatched_point[0], "y": cursor.dispatched_point[1]},
                    "control_mode": "assist",
                    "control_signal": control_signal_name,
                    "trigger": "dwell",
                },
                confidence=observation.confidence,
                risk=RiskLevel.L2,
                requires_confirmation=False,
                rationale="Assisted cursor mode held the pointer steady long enough to trigger a dwell click.",
            )
            decision = broker.decide(proposal)
            if decision.decision.value == "confirm":
                decision = broker.confirm(decision)
            request = broker.build_execution_request(decision, environment)
            result = adapter.execute(request)
            click_events += 1
            clutch_active = False
            rearm_blocked_until_signal_drop = True
            control_hold_ms = 0
            movement_since_activation = False
            stationary_elapsed_ms = 0
            fixation_elapsed_ms = 0
            candidate_direction = None
            candidate_elapsed_ms = 0
            click_cooldown_remaining_ms = max(0, click_cooldown_ms)
            control_state = "observe"
            _set_live_control_visualizer_state(point_visualizer, control_state)
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRIGGERED,
                    message="Assisted cursor control triggered a dwell click and returned to passive look mode.",
                    observation=observation,
                    proposal=proposal,
                    result=result,
                )
            )
            continue

        if (
            not click_on_dwell
            and movement_since_activation
            and raw_within_neutral
            and stationary_elapsed_ms >= max(0, idle_disarm_ms)
        ):
            cursor.recenter(cursor.neutral_point, weight=0.0)
            clutch_active = False
            rearm_blocked_until_signal_drop = True
            released_events += 1
            auto_disarm_events += 1
            control_hold_ms = 0
            movement_since_activation = False
            stationary_elapsed_ms = 0
            fixation_elapsed_ms = 0
            candidate_direction = None
            candidate_elapsed_ms = 0
            last_raw_point = None
            head_assist_x = 0.0
            head_assist_y = 0.0
            posture_frozen = False
            posture_points = []
            posture_baseline = None
            bias_corrector.reset()
            control_state = "observe"
            _set_live_control_visualizer_state(point_visualizer, control_state)
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message="Drive mode auto-disarmed after inactivity. Observe mode resumed.",
                    observation=observation,
                )
            )
            continue

        if point is None:
            held_events += 1
            if click_on_dwell and movement_since_activation and cursor.dispatched_point is not None and raw_within_neutral:
                if stationary_elapsed_ms >= max(0, dwell_ms) and click_cooldown_remaining_ms <= 0:
                    proposal = ActionProposal(
                        action=ActionName.CLICK_TARGET,
                        arguments={
                            "normalized_point": {"x": cursor.dispatched_point[0], "y": cursor.dispatched_point[1]},
                            "control_mode": "assist",
                            "control_signal": control_signal_name,
                            "trigger": "dwell",
                        },
                        confidence=observation.confidence,
                        risk=RiskLevel.L2,
                        requires_confirmation=False,
                        rationale="Assisted cursor mode held the pointer steady long enough to trigger a dwell click.",
                    )
                    decision = broker.decide(proposal)
                    if decision.decision.value == "confirm":
                        decision = broker.confirm(decision)
                    request = broker.build_execution_request(decision, environment)
                    result = adapter.execute(request)
                    click_events += 1
                    clutch_active = False
                    rearm_blocked_until_signal_drop = True
                    control_hold_ms = 0
                    movement_since_activation = False
                    stationary_elapsed_ms = 0
                    fixation_elapsed_ms = 0
                    candidate_direction = None
                    candidate_elapsed_ms = 0
                    click_cooldown_remaining_ms = max(0, click_cooldown_ms)
                    control_state = "observe"
                    _set_live_control_visualizer_state(point_visualizer, control_state)
                    events.append(
                        GazeFeedbackEvent(
                            phase=GazeLoopPhase.TRIGGERED,
                            message="Assisted cursor control triggered a dwell click and returned to passive look mode.",
                            observation=observation,
                            proposal=proposal,
                            result=result,
                        )
                    )
                    continue
            elif not click_on_dwell and movement_since_activation and raw_within_neutral:
                if stationary_elapsed_ms >= max(0, idle_disarm_ms):
                    clutch_active = False
                    rearm_blocked_until_signal_drop = True
                    released_events += 1
                    auto_disarm_events += 1
                    control_hold_ms = 0
                    movement_since_activation = False
                    stationary_elapsed_ms = 0
                    fixation_elapsed_ms = 0
                    candidate_direction = None
                    candidate_elapsed_ms = 0
                    last_raw_point = None
                    head_assist_x = 0.0
                    head_assist_y = 0.0
                    posture_frozen = False
                    posture_points = []
                    posture_baseline = None
                    bias_corrector.reset()
                    control_state = "observe"
                    _set_live_control_visualizer_state(point_visualizer, control_state)
                    events.append(
                        GazeFeedbackEvent(
                            phase=GazeLoopPhase.TRACKING,
                            message="Drive mode auto-disarmed after inactivity. Observe mode resumed.",
                            observation=observation,
                        )
                    )
                    continue
            else:
                stationary_elapsed_ms = 0
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRACKING,
                    message="Drive mode is holding steady at the current fixation.",
                    observation=observation,
                )
            )
            continue

        stationary_elapsed_ms = 0
        movement_since_activation = True
        candidate_direction = None
        candidate_elapsed_ms = 0
        proposal = ActionProposal(
            action=ActionName.FOCUS_TARGET,
            arguments={
                "normalized_point": {"x": point[0], "y": point[1]},
                "control_mode": "assist",
                "control_signal": control_signal_name,
            },
            confidence=observation.confidence,
            risk=RiskLevel.L1,
            requires_confirmation=False,
            rationale="Assisted cursor mode moved the pointer only while the explicit control signal was active, with small head movement assist around the current pose.",
        )
        decision = broker.decide(proposal)
        request = broker.build_execution_request(decision, environment)
        result = adapter.execute(request)
        moved_events += 1
        control_state = "drive"
        drive_frames += 1
        _set_live_control_visualizer_state(point_visualizer, control_state)
        events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRACKING,
                message="Assisted cursor control nudged the pointer while the control signal was active.",
                observation=observation,
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
        "click_events": click_events,
        "low_confidence_readings": low_confidence_readings,
        "passive_frames": passive_frames,
        "active_frames": active_frames,
        "engaged_events": engaged_events,
        "released_events": released_events,
        "recenter_events": recenter_events,
        "posture_signal_available": posture_signal_available,
        "posture_freeze_events": posture_freeze_events,
        "posture_recenter_events": posture_recenter_events,
        "control_signal": control_signal_name,
        "control_signal_available": signal_available,
        "control_signal_threshold": control_signal_threshold,
        "max_control_signal": max_control_signal,
        "head_assist_active_frames": head_assist_active_frames,
        "max_head_assist": {
            "x": max_head_assist_x,
            "y": max_head_assist_y,
        },
        "control_state": control_state,
        "drive_ready_frames": drive_ready_frames,
        "drive_frames": drive_frames,
        "armed_frames": armed_frames,
        "paused_frames": paused_frames,
        "auto_disarm_events": auto_disarm_events,
        "drive_commit_ms": max(0, drive_commit_ms),
        "idle_disarm_ms": max(0, idle_disarm_ms),
        "online_bias_correction": {
            "enabled": True,
            "updates": bias_update_events,
            "active_ms": bias_corrector.active_ms,
            "offset": {
                "x": bias_corrector.offset_x,
                "y": bias_corrector.offset_y,
            },
        },
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
    observation, target, last_index = latest_match[dominant_target_id]
    latest_target_id = max(latest_match, key=lambda target_id: latest_match[target_id][2])
    if latest_target_id != dominant_target_id or frames - 1 - last_index > 2 or hits[dominant_target_id] < 2:
        events.append(GazeFeedbackEvent(
            phase=GazeLoopPhase.RECOVERING,
            message="The gaze target changed or tracking was lost near the end of capture. Look steadily at the target and retry.",
        ))
        return None, events, summary
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


def _refine_point_within_bounds(x_norm: float, y_norm: float, bounds) -> tuple[float, float]:
    min_padding = 0.006
    max_padding = 0.016
    pad_x = min(max_padding, max(min_padding, float(bounds.width) * 0.06))
    pad_y = min(max_padding, max(min_padding, float(bounds.height) * 0.06))
    left = float(bounds.x) + pad_x
    right = float(bounds.x) + float(bounds.width) - pad_x
    top = float(bounds.y) + pad_y
    bottom = float(bounds.y) + float(bounds.height) - pad_y
    if right <= left:
        center_x = float(bounds.x) + float(bounds.width) / 2.0
    else:
        center_x = max(left, min(right, float(x_norm)))
    if bottom <= top:
        center_y = float(bounds.y) + float(bounds.height) / 2.0
    else:
        center_y = max(top, min(bottom, float(y_norm)))
    return center_x, center_y


def _reading_signal(reading: Any, name: str) -> tuple[bool, float]:
    signals = getattr(reading, "signals", None)
    if not isinstance(signals, dict):
        return False, 0.0
    value = signals.get(name)
    if not isinstance(value, (int, float)):
        return False, 0.0
    return True, max(0.0, min(1.0, float(value)))


def _classify_wink(
    left_eye_closed: float,
    right_eye_closed: float,
    *,
    closed_threshold: float,
    open_threshold: float,
) -> str | None:
    closed_threshold = max(0.0, min(1.0, float(closed_threshold)))
    open_threshold = max(0.0, min(closed_threshold, float(open_threshold)))
    dominance_threshold = max(0.12, (closed_threshold - open_threshold) * 0.40)
    bilateral_blink_threshold = min(0.98, closed_threshold + 0.12)
    open_ceiling = min(0.62, max(open_threshold + 0.18, open_threshold))
    dominance_ratio_threshold = 1.45

    left = max(0.0, min(1.0, float(left_eye_closed)))
    right = max(0.0, min(1.0, float(right_eye_closed)))
    if left >= bilateral_blink_threshold and right >= bilateral_blink_threshold:
        return None

    if left >= closed_threshold and right <= open_ceiling and (left - right) >= dominance_threshold and (left + 0.06) / (right + 0.06) >= dominance_ratio_threshold:
        return "left"
    if right >= closed_threshold and left <= open_ceiling and (right - left) >= dominance_threshold and (right + 0.06) / (left + 0.06) >= dominance_ratio_threshold:
        return "right"
    return None


def _classify_wink_candidate(
    left_eye_closed: float,
    right_eye_closed: float,
    *,
    closed_threshold: float,
    open_threshold: float,
) -> str | None:
    candidate_threshold = max(open_threshold + 0.05, min(0.68, closed_threshold * 0.72))
    dominance_threshold = max(0.08, (closed_threshold - open_threshold) * 0.24)
    dominance_ratio_threshold = 1.2
    left = max(0.0, min(1.0, float(left_eye_closed)))
    right = max(0.0, min(1.0, float(right_eye_closed)))
    if left >= candidate_threshold and (left - right) >= dominance_threshold and (left + 0.06) / (right + 0.06) >= dominance_ratio_threshold:
        return "left"
    if right >= candidate_threshold and (right - left) >= dominance_threshold and (right + 0.06) / (left + 0.06) >= dominance_ratio_threshold:
        return "right"
    return None


def _smooth_signal(previous: float, current: float, *, weight: float) -> float:
    weight = max(0.0, min(1.0, float(weight)))
    return _lerp(previous, current, weight)


def _target_grid_dimensions(targets: list[NormalizedScreenTarget]) -> dict[str, int]:
    if not targets:
        return {"columns": 0, "rows": 0}
    columns = len({round(float(target.x), 6) for target in targets})
    rows = len({round(float(target.y), 6) for target in targets})
    return {"columns": columns, "rows": rows}


def _reading_raw_sample(reading: Any) -> GazeSample:
    raw_sample = getattr(reading, "raw_sample", None)
    if raw_sample is not None:
        return raw_sample
    return reading.sample


def _aggregate_reading_quality(readings: list[Any]) -> dict[str, float | int]:
    quality_samples = [getattr(reading, "quality", None) for reading in readings]
    quality_samples = [sample for sample in quality_samples if isinstance(sample, dict)]
    if not quality_samples:
        return {}

    aggregated: dict[str, float | int] = {}
    numeric_keys = (
        "x_axis_confidence",
        "y_axis_confidence",
        "horizontal_disagreement",
        "vertical_disagreement",
        "fused_delta_x",
        "fused_delta_y",
        "fusion_samples",
    )
    boolean_keys = (
        "x_saturated",
        "y_saturated",
        "temporal_fusion_active",
        "fusion_wink_freeze",
    )
    for key in numeric_keys:
        values = [float(sample[key]) for sample in quality_samples if key in sample]
        if values:
            aggregated[f"{key}_mean"] = sum(values) / len(values)
    for key in boolean_keys:
        aggregated[f"{key}_count"] = sum(1 for sample in quality_samples if bool(sample.get(key)))
    return aggregated


def _point_series_metrics(points: list[tuple[float, float]], *, prefix: str) -> dict[str, float]:
    if not points:
        return {
            f"{prefix}_x_span": 0.0,
            f"{prefix}_y_span": 0.0,
            f"{prefix}_jitter_x": 0.0,
            f"{prefix}_jitter_y": 0.0,
            f"{prefix}_drift_x": 0.0,
            f"{prefix}_drift_y": 0.0,
            f"{prefix}_drift_distance": 0.0,
        }
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_steps = [abs(current[0] - previous[0]) for previous, current in zip(points, points[1:])]
    y_steps = [abs(current[1] - previous[1]) for previous, current in zip(points, points[1:])]
    drift_x = points[-1][0] - points[0][0]
    drift_y = points[-1][1] - points[0][1]
    return {
        f"{prefix}_x_span": max(xs) - min(xs),
        f"{prefix}_y_span": max(ys) - min(ys),
        f"{prefix}_jitter_x": (sum(x_steps) / len(x_steps)) if x_steps else 0.0,
        f"{prefix}_jitter_y": (sum(y_steps) / len(y_steps)) if y_steps else 0.0,
        f"{prefix}_drift_x": drift_x,
        f"{prefix}_drift_y": drift_y,
        f"{prefix}_drift_distance": math.hypot(drift_x, drift_y),
    }


def _raw_fused_delta_metrics(
    raw_points: list[tuple[float, float]],
    fused_points: list[tuple[float, float]],
) -> dict[str, float]:
    if not raw_points or not fused_points:
        return {
            "raw_fused_mean_delta_x": 0.0,
            "raw_fused_mean_delta_y": 0.0,
            "raw_fused_mean_distance": 0.0,
            "raw_fused_max_distance": 0.0,
        }
    deltas = [
        (abs(raw_x - fused_x), abs(raw_y - fused_y), math.dist((raw_x, raw_y), (fused_x, fused_y)))
        for (raw_x, raw_y), (fused_x, fused_y) in zip(raw_points, fused_points)
    ]
    return {
        "raw_fused_mean_delta_x": sum(delta_x for delta_x, _, _ in deltas) / len(deltas),
        "raw_fused_mean_delta_y": sum(delta_y for _, delta_y, _ in deltas) / len(deltas),
        "raw_fused_mean_distance": sum(distance for _, _, distance in deltas) / len(deltas),
        "raw_fused_max_distance": max(distance for _, _, distance in deltas),
    }


def _head_signal_metrics(readings: list[Any]) -> dict[str, float]:
    def _collect(name: str) -> list[float]:
        values: list[float] = []
        for reading in readings:
            signals = getattr(reading, "signals", None)
            if not isinstance(signals, dict):
                continue
            value = signals.get(name)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    head_yaw_values = _collect("head_yaw")
    head_pitch_values = _collect("head_pitch")
    head_roll_values = _collect("head_roll")
    return {
        "head_yaw_abs_mean": (sum(abs(value) for value in head_yaw_values) / len(head_yaw_values)) if head_yaw_values else 0.0,
        "head_pitch_abs_mean": (sum(abs(value) for value in head_pitch_values) / len(head_pitch_values)) if head_pitch_values else 0.0,
        "head_roll_abs_mean": (sum(abs(value) for value in head_roll_values) / len(head_roll_values)) if head_roll_values else 0.0,
        "head_yaw_abs_max": max((abs(value) for value in head_yaw_values), default=0.0),
        "head_pitch_abs_max": max((abs(value) for value in head_pitch_values), default=0.0),
        "head_roll_abs_max": max((abs(value) for value in head_roll_values), default=0.0),
    }


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


def _dismiss_point_visualizer_for_pointer_action(point_visualizer: Any | None, *, settle_ms: int = 60) -> None:
    if point_visualizer is None:
        return
    try:
        point_visualizer.hide()
    except Exception:
        pass
    try:
        point_visualizer.close()
    except Exception:
        pass
    if settle_ms > 0:
        time.sleep(max(0.0, float(settle_ms) / 1000.0))


def _dismiss_live_visualizers_for_pointer_action(
    point_visualizer: Any | None,
    camera_preview: Any | None,
    *,
    settle_ms: int = 60,
) -> None:
    _dismiss_point_visualizer_for_pointer_action(point_visualizer, settle_ms=0)
    if camera_preview is not None:
        try:
            close = getattr(camera_preview, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
    if settle_ms > 0:
        time.sleep(max(0.0, float(settle_ms) / 1000.0))


def _set_live_control_visualizer_state(point_visualizer: Any | None, state: str) -> None:
    if point_visualizer is None or not hasattr(point_visualizer, "set_style"):
        return
    palette = {
        "observe": {"fill": "#0a84ff", "outline": "#ffffff"},
        "drive_ready": {"fill": "#64d2ff", "outline": "#ffffff"},
        "drive": {"fill": "#30d158", "outline": "#ffffff"},
        "armed": {"fill": "#ffd60a", "outline": "#ffffff"},
        "paused": {"fill": "#bf5af2", "outline": "#ffffff"},
    }
    style = palette.get(state, palette["observe"])
    try:
        point_visualizer.set_style(**style)
    except Exception:
        return


def _control_direction(control_x: float, control_y: float) -> tuple[int, int] | None:
    x = 1 if control_x > 1e-6 else -1 if control_x < -1e-6 else 0
    y = 1 if control_y > 1e-6 else -1 if control_y < -1e-6 else 0
    if x == 0 and y == 0:
        return None
    return x, y


def _show_snap_visualizer(
    point_visualizer: Any | None,
    *,
    target_x: float,
    target_y: float,
    tracked_x: float,
    tracked_y: float,
    armed: bool = False,
    wink_side: str | None = None,
) -> None:
    if point_visualizer is None:
        return
    try:
        if hasattr(point_visualizer, "show_validation_point"):
            try:
                point_visualizer.show_validation_point(
                    target_x,
                    target_y,
                    tracked_x=tracked_x,
                    tracked_y=tracked_y,
                    armed=armed,
                    wink_side=wink_side,
                )
            except TypeError:
                point_visualizer.show_validation_point(
                    target_x,
                    target_y,
                    tracked_x=tracked_x,
                    tracked_y=tracked_y,
                )
            return
        point_visualizer.show_point(target_x, target_y)
    except Exception:
        return


def _armed_wink_feedback_message(
    target_label: str,
    *,
    wink_side: str | None,
    candidate_wink_side: str | None,
    left_elapsed_ms: int,
    right_elapsed_ms: int,
    wink_hold_ms: int,
    left_signal: float,
    right_signal: float,
) -> str:
    if wink_side == "left":
        remaining_ms = max(0, wink_hold_ms - left_elapsed_ms)
        return f'Left wink detected on "{target_label}". Hold for {remaining_ms} ms more to left-click.'
    if wink_side == "right":
        remaining_ms = max(0, wink_hold_ms - right_elapsed_ms)
        return f'Right wink detected on "{target_label}". Hold for {remaining_ms} ms more to right-click.'
    if candidate_wink_side == "left":
        return (
            f'Left wink candidate on "{target_label}". '
            f"L={left_signal:.2f} R={right_signal:.2f}. Close a bit more to left-click."
        )
    if candidate_wink_side == "right":
        return (
            f'Right wink candidate on "{target_label}". '
            f"L={left_signal:.2f} R={right_signal:.2f}. Close a bit more to right-click."
        )
    return (
        f'Previewing armed target "{target_label}". '
        "Left wink to click, right wink to right-click, or look away briefly to cancel."
    )


def _wink_test_feedback_message(
    *,
    active_wink_side: str | None,
    candidate_wink_side: str | None,
    left_elapsed_ms: int,
    right_elapsed_ms: int,
    wink_hold_ms: int,
    left_signal: float,
    right_signal: float,
) -> str:
    if active_wink_side == "left":
        remaining_ms = max(0, wink_hold_ms - left_elapsed_ms)
        return f"Left wink detected. Hold for {remaining_ms} ms more to left-click at the current cursor position."
    if active_wink_side == "right":
        remaining_ms = max(0, wink_hold_ms - right_elapsed_ms)
        return f"Right wink detected. Hold for {remaining_ms} ms more to right-click at the current cursor position."
    if candidate_wink_side == "left":
        return (
            f"Left wink candidate. L={left_signal:.2f} R={right_signal:.2f}. "
            "Close a bit more to left-click at the current cursor position."
        )
    if candidate_wink_side == "right":
        return (
            f"Right wink candidate. L={left_signal:.2f} R={right_signal:.2f}. "
            "Close a bit more to right-click at the current cursor position."
        )
    return "Watching for deliberate left and right winks at the current cursor position."


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


def _target_by_id(targets: list[NormalizedScreenTarget], target_id: str | None) -> GroundedTarget | None:
    if not target_id:
        return None
    for target in targets:
        if target.target_id == target_id:
            return target.to_grounded_target(confidence=1.0)
    return None


def _capture_average_raw_sample_stats(
    provider: WebcamProvider,
    *,
    required_frames: int,
    max_attempts: int,
    delta_ms: int,
) -> tuple[WebcamSampleAggregate | None, dict[str, float | int], PostureState | None]:
    readings = []
    attempted_frames = 0
    while attempted_frames < max_attempts and len(readings) < required_frames:
        attempted_frames += 1
        reading = provider.read(delta_ms=delta_ms)
        if reading is None:
            continue
        readings.append(reading)
    if not readings:
        return None, {
            "sample_count": 0,
            "raw_x_span": 0.0,
            "raw_y_span": 0.0,
            "raw_x_mean": 0.0,
            "raw_y_mean": 0.0,
        }, None
    raw_samples = [_reading_raw_sample(reading) for reading in readings]
    avg_x = sum(sample.point.x for sample in raw_samples) / len(raw_samples)
    avg_y = sum(sample.point.y for sample in raw_samples) / len(raw_samples)
    avg_confidence = sum(sample.confidence for sample in raw_samples) / len(raw_samples)
    quality = {
        "sample_count": len(readings),
        "raw_x_span": max(sample.point.x for sample in raw_samples) - min(sample.point.x for sample in raw_samples),
        "raw_y_span": max(sample.point.y for sample in raw_samples) - min(sample.point.y for sample in raw_samples),
        "raw_x_mean": avg_x,
        "raw_y_mean": avg_y,
    }
    quality.update(_aggregate_reading_quality(readings))
    posture_baseline = _average_posture_states(
        posture_state for posture_state in (_reading_posture_state(reading) for reading in readings) if posture_state is not None
    )
    return (
        WebcamSampleAggregate(
            sample=GazeSample(
                point=NormalizedPoint(x=_clamp_normalized(avg_x), y=_clamp_normalized(avg_y)),
                confidence=max(0.0, min(1.0, avg_confidence)),
                delta_ms=delta_ms,
            ),
            used_frames=len(readings),
            attempted_frames=attempted_frames,
        ),
        quality,
        posture_baseline,
    )
