"""Cursor control and posture helpers for webcam-driven live modes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Any

from interaction.contracts import GazeObservation
from interaction.vision import NormalizedPoint

POSTURE_TRANSLATION_THRESHOLD = 0.035
POSTURE_SIZE_THRESHOLD = 0.05
POSTURE_YAW_THRESHOLD = 0.28
POSTURE_PITCH_THRESHOLD = 0.28
POSTURE_ROLL_THRESHOLD = 0.24
POSTURE_STABLE_TRANSLATION = 0.014
POSTURE_STABLE_SIZE = 0.02
POSTURE_STABLE_YAW = 0.10
POSTURE_STABLE_PITCH = 0.10
POSTURE_STABLE_ROLL = 0.12
POSTURE_STABILIZE_MS = 240
HEAD_ASSIST_YAW_DEADZONE = 0.03
HEAD_ASSIST_PITCH_DEADZONE = 0.03
HEAD_ASSIST_YAW_RANGE = POSTURE_YAW_THRESHOLD * 0.8
HEAD_ASSIST_PITCH_RANGE = POSTURE_PITCH_THRESHOLD * 0.8
HEAD_ASSIST_MAX_X = 0.32
HEAD_ASSIST_MAX_Y = 0.38
HEAD_ASSIST_FILTER_WEIGHT = 0.34
HEAD_ASSIST_ACTIVE_THRESHOLD = 0.035
ONLINE_BIAS_LEARN_WINDOW_SCALE = 1.35


def clamp_normalized(value: float) -> float:
    return max(0.0, min(1.0, value))


def clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def clamp_with_padding(value: float, padding: float) -> float:
    padding = max(0.0, min(0.45, padding))
    return max(padding, min(1.0 - padding, value))


def lerp(start: float, end: float, weight: float) -> float:
    clamped_weight = max(0.0, min(1.0, weight))
    return start + (end - start) * clamped_weight


def relative_axis_control(value: float, neutral: float, deadzone: float) -> float:
    offset = float(value) - float(neutral)
    if abs(offset) <= deadzone:
        return 0.0
    available = (1.0 - neutral - deadzone) if offset > 0 else (neutral - deadzone)
    available = max(1e-6, available)
    normalized = max(0.0, min(1.0, (abs(offset) - deadzone) / available))
    curved = normalized**1.1
    boosted = min(1.0, 0.18 + curved * 0.82)
    return math.copysign(boosted, offset)


def relative_controls(
    x_value: float,
    y_value: float,
    *,
    neutral_point: tuple[float, float],
    deadzone_x: float,
    deadzone_y: float,
) -> tuple[float, float]:
    return (
        relative_axis_control(x_value, neutral_point[0], deadzone_x),
        relative_axis_control(y_value, neutral_point[1], deadzone_y),
    )


def signed_axis_control(value: float, *, deadzone: float, max_input: float) -> float:
    deadzone = max(0.0, float(deadzone))
    max_input = max(deadzone + 1e-6, float(max_input))
    magnitude = abs(float(value))
    if magnitude <= deadzone:
        return 0.0
    normalized = max(0.0, min(1.0, (magnitude - deadzone) / (max_input - deadzone)))
    curved = normalized**1.15
    boosted = min(1.0, 0.16 + curved * 0.84)
    return math.copysign(boosted, value)


@dataclass(frozen=True)
class PostureState:
    center_x: float
    center_y: float
    width: float
    height: float
    yaw: float
    pitch: float
    roll: float


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
            clamp_with_padding(observation.x_norm, self.edge_padding),
            clamp_with_padding(observation.y_norm, self.edge_padding),
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
            clamp_with_padding(self.dispatched_point[0] + step_x, self.edge_padding),
            clamp_with_padding(self.dispatched_point[1] + step_y, self.edge_padding),
        )
        self.dispatched_point = next_point
        return next_point


@dataclass
class RelativeCursorController:
    deadzone: float = 0.08
    deadzone_x: float | None = None
    deadzone_y: float | None = None
    edge_padding: float = 0.05
    max_step: float = 0.04
    speed_gain_x: float = 0.25
    speed_gain_y: float = 0.5
    neutral_point: tuple[float, float] = (0.5, 0.5)
    dispatched_point: tuple[float, float] | None = None
    acceleration: float = 0.28
    friction: float = 0.45
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    def next_point(self, observation: GazeObservation) -> tuple[float, float] | None:
        if observation.x_norm is None or observation.y_norm is None:
            return None
        control_x, control_y = relative_controls(
            observation.x_norm,
            observation.y_norm,
            neutral_point=self.neutral_point,
            deadzone_x=self.deadzone_x if self.deadzone_x is not None else self.deadzone,
            deadzone_y=self.deadzone_y if self.deadzone_y is not None else self.deadzone,
        )
        return self.next_point_from_controls(control_x, control_y)

    def next_point_from_controls(self, control_x: float, control_y: float) -> tuple[float, float] | None:
        if self.dispatched_point is None:
            self.dispatched_point = (0.5, 0.5)

        desired_step_x = control_x * self.max_step * self.speed_gain_x
        desired_step_y = control_y * self.max_step * self.speed_gain_y

        if abs(control_x) < 1e-6:
            self.velocity_x *= max(0.0, min(1.0, self.friction))
        else:
            self.velocity_x = lerp(self.velocity_x, desired_step_x, max(0.0, min(1.0, self.acceleration)))

        if abs(control_y) < 1e-6:
            self.velocity_y *= max(0.0, min(1.0, self.friction))
        else:
            self.velocity_y = lerp(self.velocity_y, desired_step_y, max(0.0, min(1.0, self.acceleration)))

        step_x = self.velocity_x
        step_y = self.velocity_y
        step_distance = math.hypot(step_x, step_y)
        if step_distance < 1.5e-4:
            self.velocity_x = 0.0
            self.velocity_y = 0.0
            return None
        if step_distance > self.max_step and step_distance > 0:
            scale = self.max_step / step_distance
            step_x *= scale
            step_y *= scale
            self.velocity_x = step_x
            self.velocity_y = step_y

        next_point = (
            clamp_with_padding(self.dispatched_point[0] + step_x, self.edge_padding),
            clamp_with_padding(self.dispatched_point[1] + step_y, self.edge_padding),
        )
        self.dispatched_point = next_point
        return next_point

    def recenter(self, point: tuple[float, float], *, weight: float = 1.0) -> None:
        self.neutral_point = (
            clamp_with_padding(lerp(self.neutral_point[0], point[0], max(0.0, min(1.0, weight))), self.edge_padding),
            clamp_with_padding(lerp(self.neutral_point[1], point[1], max(0.0, min(1.0, weight))), self.edge_padding),
        )
        self.velocity_x = 0.0
        self.velocity_y = 0.0


@dataclass
class OnlineBiasCorrector:
    max_offset: float = 0.035
    learning_rate: float = 0.18
    min_stable_ms: int = 220
    min_residual: float = 0.003
    offset_x: float = 0.0
    offset_y: float = 0.0
    updates: int = 0
    active_ms: int = 0

    def apply(self, point: tuple[float, float], *, edge_padding: float) -> tuple[float, float]:
        return (
            clamp_with_padding(point[0] + self.offset_x, edge_padding),
            clamp_with_padding(point[1] + self.offset_y, edge_padding),
        )

    def maybe_update(
        self,
        raw_point: tuple[float, float],
        *,
        reference_point: tuple[float, float],
        stable_ms: int,
        sample_delta_ms: int,
        learn_window_x: float,
        learn_window_y: float,
    ) -> bool:
        if stable_ms < max(0, self.min_stable_ms):
            return False
        updated = False
        residual_x = reference_point[0] - raw_point[0]
        residual_y = reference_point[1] - raw_point[1]
        if self._within_learning_window(residual_x, learn_window_x):
            self.offset_x = self._updated_offset(self.offset_x, residual_x)
            updated = True
        if self._within_learning_window(residual_y, learn_window_y):
            self.offset_y = self._updated_offset(self.offset_y, residual_y)
            updated = True
        if updated:
            self.updates += 1
            self.active_ms += max(0, sample_delta_ms)
        return updated

    def reset(self) -> None:
        self.offset_x = 0.0
        self.offset_y = 0.0

    def _within_learning_window(self, residual: float, learn_window: float) -> bool:
        return self.min_residual <= abs(residual) <= max(self.min_residual, float(learn_window))

    def _updated_offset(self, current_offset: float, desired_offset: float) -> float:
        return max(-self.max_offset, min(self.max_offset, lerp(current_offset, desired_offset, self.learning_rate)))


def adaptive_relative_deadzone(*, base_deadzone: float, observed_span: float) -> float:
    base_deadzone = max(0.0, float(base_deadzone))
    if base_deadzone <= 0.0:
        return 0.0
    return min(base_deadzone, max(0.010, max(0.0, float(observed_span)) * 1.6))


def update_fixation_window(
    points: list[tuple[tuple[float, float], int]],
    point: tuple[float, float],
    delta_ms: int,
    *,
    radius: float,
) -> tuple[list[tuple[tuple[float, float], int]], int]:
    next_points = [*points, (point, max(0, delta_ms))]
    radius = max(0.0, float(radius))
    while next_points and any(math.dist(candidate_point, point) > radius for candidate_point, _ in next_points):
        next_points.pop(0)
    return next_points, sum(duration_ms for _, duration_ms in next_points)


def reading_posture_state(reading: Any) -> PostureState | None:
    frame_size = getattr(reading, "frame_size", None)
    face_bbox = getattr(reading, "face_bbox", None)
    if (
        not isinstance(frame_size, tuple)
        or len(frame_size) != 2
        or not isinstance(face_bbox, tuple)
        or len(face_bbox) != 4
    ):
        return None
    frame_width, frame_height = frame_size
    face_x, face_y, face_width, face_height = face_bbox
    if frame_width <= 0 or frame_height <= 0 or face_width <= 0 or face_height <= 0:
        return None
    signals = getattr(reading, "signals", None)
    signal_map = signals if isinstance(signals, dict) else {}
    return PostureState(
        center_x=clamp_normalized((float(face_x) + float(face_width) / 2.0) / float(frame_width)),
        center_y=clamp_normalized((float(face_y) + float(face_height) / 2.0) / float(frame_height)),
        width=clamp_normalized(float(face_width) / float(frame_width)),
        height=clamp_normalized(float(face_height) / float(frame_height)),
        yaw=max(-1.0, min(1.0, float(signal_map.get("head_yaw", 0.0)))),
        pitch=max(-1.0, min(1.0, float(signal_map.get("head_pitch", 0.0)))),
        roll=max(-1.0, min(1.0, float(signal_map.get("head_roll", 0.0)))),
    )


def average_posture_states(states: Iterable[PostureState]) -> PostureState | None:
    collected = list(states)
    if not collected:
        return None
    count = float(len(collected))
    return PostureState(
        center_x=sum(state.center_x for state in collected) / count,
        center_y=sum(state.center_y for state in collected) / count,
        width=sum(state.width for state in collected) / count,
        height=sum(state.height for state in collected) / count,
        yaw=sum(state.yaw for state in collected) / count,
        pitch=sum(state.pitch for state in collected) / count,
        roll=sum(state.roll for state in collected) / count,
    )


def posture_metrics(current: PostureState, baseline: PostureState) -> dict[str, float]:
    return {
        "translation": math.dist((current.center_x, current.center_y), (baseline.center_x, baseline.center_y)),
        "size": max(abs(current.width - baseline.width), abs(current.height - baseline.height)),
        "yaw": abs(current.yaw - baseline.yaw),
        "pitch": abs(current.pitch - baseline.pitch),
        "roll": abs(current.roll - baseline.roll),
    }


def is_large_posture_change(current: PostureState, baseline: PostureState) -> bool:
    metrics = posture_metrics(current, baseline)
    return (
        metrics["translation"] >= POSTURE_TRANSLATION_THRESHOLD
        or metrics["size"] >= POSTURE_SIZE_THRESHOLD
        or metrics["yaw"] >= POSTURE_YAW_THRESHOLD
        or metrics["pitch"] >= POSTURE_PITCH_THRESHOLD
        or metrics["roll"] >= POSTURE_ROLL_THRESHOLD
    )


def head_assist_targets(current: PostureState | None, baseline: PostureState | None) -> tuple[float, float]:
    if current is None or baseline is None:
        return 0.0, 0.0
    yaw_delta = clamp_signed(current.yaw - baseline.yaw)
    pitch_delta = clamp_signed(current.pitch - baseline.pitch)
    return (
        signed_axis_control(
            yaw_delta,
            deadzone=HEAD_ASSIST_YAW_DEADZONE,
            max_input=HEAD_ASSIST_YAW_RANGE,
        ) * HEAD_ASSIST_MAX_X,
        signed_axis_control(
            pitch_delta,
            deadzone=HEAD_ASSIST_PITCH_DEADZONE,
            max_input=HEAD_ASSIST_PITCH_RANGE,
        ) * HEAD_ASSIST_MAX_Y,
    )


def gated_head_assist_component(eye_control: float, head_assist: float) -> float:
    if abs(eye_control) < 1e-6 or abs(head_assist) < 1e-6:
        return 0.0
    if math.copysign(1.0, eye_control) != math.copysign(1.0, head_assist):
        return 0.0
    return head_assist


def update_posture_window(
    points: list[tuple[PostureState, int]],
    state: PostureState,
    delta_ms: int,
    *,
    translation_threshold: float,
    size_threshold: float,
    yaw_threshold: float,
    pitch_threshold: float,
    roll_threshold: float,
) -> tuple[list[tuple[PostureState, int]], int]:
    next_points = [*points, (state, max(0, delta_ms))]
    while next_points:
        latest_state = next_points[-1][0]
        if all(
            posture_metrics(candidate_state, latest_state)["translation"] <= translation_threshold
            and posture_metrics(candidate_state, latest_state)["size"] <= size_threshold
            and posture_metrics(candidate_state, latest_state)["yaw"] <= yaw_threshold
            and posture_metrics(candidate_state, latest_state)["pitch"] <= pitch_threshold
            and posture_metrics(candidate_state, latest_state)["roll"] <= roll_threshold
            for candidate_state, _ in next_points
        ):
            break
        next_points.pop(0)
    return next_points, sum(duration_ms for _, duration_ms in next_points)


def compress_point_around_anchor(
    point: NormalizedPoint,
    *,
    anchor_point: tuple[float, float],
    gain_x: float,
    gain_y: float,
    edge_padding: float,
) -> NormalizedPoint:
    gain_x = max(0.05, min(1.0, float(gain_x)))
    gain_y = max(0.05, min(1.0, float(gain_y)))
    anchor_x = clamp_with_padding(anchor_point[0], edge_padding)
    anchor_y = clamp_with_padding(anchor_point[1], edge_padding)
    return NormalizedPoint(
        x=clamp_with_padding(anchor_x + (point.x - anchor_x) * gain_x, edge_padding),
        y=clamp_with_padding(anchor_y + (point.y - anchor_y) * gain_y, edge_padding),
    )


__all__ = [
    "CursorFollowController",
    "HEAD_ASSIST_ACTIVE_THRESHOLD",
    "HEAD_ASSIST_FILTER_WEIGHT",
    "ONLINE_BIAS_LEARN_WINDOW_SCALE",
    "OnlineBiasCorrector",
    "POSTURE_STABILIZE_MS",
    "POSTURE_STABLE_PITCH",
    "POSTURE_STABLE_ROLL",
    "POSTURE_STABLE_SIZE",
    "POSTURE_STABLE_TRANSLATION",
    "POSTURE_STABLE_YAW",
    "PostureState",
    "RelativeCursorController",
    "adaptive_relative_deadzone",
    "average_posture_states",
    "clamp_normalized",
    "clamp_signed",
    "clamp_with_padding",
    "compress_point_around_anchor",
    "gated_head_assist_component",
    "head_assist_targets",
    "is_large_posture_change",
    "lerp",
    "reading_posture_state",
    "relative_controls",
    "signed_axis_control",
    "update_fixation_window",
    "update_posture_window",
]
