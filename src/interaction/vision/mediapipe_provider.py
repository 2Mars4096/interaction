"""MediaPipe-backed webcam gaze provider using face mesh and iris landmarks."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import math
from pathlib import Path
import time
from typing import Any, TypedDict

import cv2
import numpy as np

from .calibration import NormalizedPoint
from .gaze import GazeSample
from .provider_common import clamp_unit_interval, effective_delta_ms, silence_opencv_logging, suppress_native_stderr
from .provider_types import (
    WebcamGazeReading,
    WebcamProviderContextMixin,
    WebcamProviderError,
    WebcamSampleAggregate,
)

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - import availability depends on env
    mp = None


RIGHT_EYE_CORNERS = (33, 133)
RIGHT_EYE_LIDS = (159, 145)
RIGHT_IRIS = (468, 469, 470, 471, 472)

LEFT_EYE_CORNERS = (362, 263)
LEFT_EYE_LIDS = (386, 374)
LEFT_IRIS = (473, 474, 475, 476, 477)
MOUTH_CORNERS = (78, 308)
MOUTH_LIPS = (13, 14)
FACE_SIDE_POINTS = (234, 454)
FACE_VERTICAL_POINTS = (10, 152)
NOSE_TIP = 1
OUTER_EYE_CORNERS = (33, 263)

HORIZONTAL_BLENDSHAPE_WEIGHT = 0.72
HORIZONTAL_IRIS_WEIGHT = 0.28
VERTICAL_BLENDSHAPE_WEIGHT = 0.35
VERTICAL_IRIS_WEIGHT = 0.45
# Vertical component weights are relative priors before agreement-based scaling.
VERTICAL_EYELID_WEIGHT = 0.20
HEAD_YAW_COMPENSATION_WEIGHT = 0.10
HEAD_PITCH_COMPENSATION_WEIGHT = 0.12
TEMPORAL_FUSION_WINDOW_SIZE = 5
TEMPORAL_FUSION_WINK_BLOCK_THRESHOLD = 0.55
TEMPORAL_FUSION_MIN_WEIGHT = 0.12
BLENDSHAPE_BASELINE_WINDOW_SIZE = 45
BLENDSHAPE_BASELINE_MIN_SAMPLES = 4
VERTICAL_BLENDSHAPE_BASELINE_GAIN = 2.4
VERTICAL_OPENNESS_BASELINE_GAIN = 3.6
VERTICAL_OPENNESS_PARTIAL_SQUINT_THRESHOLD = 0.035
VERTICAL_EYELID_SQUINT_ATTENUATION = 0.80
CAPTURE_WARMUP_SUCCESS_FRAMES = 1


class EyeMeasurement(TypedDict):
    x_ratio: float
    y_ratio: float
    openness: float
    x_reliability: float
    y_reliability: float
    box: tuple[int, int, int, int]
    iris_center: tuple[int, int]


_clamp = clamp_unit_interval


class MediaPipeWebcamGazeProvider(WebcamProviderContextMixin):
    """Estimate webcam gaze using MediaPipe face mesh landmarks."""

    def __init__(
        self,
        camera_index: int = 0,
        *,
        model_asset_path: str | None = None,
        face_mesh_factory: Callable[[], Any] | None = None,
        capture_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self.camera_index = camera_index
        self.model_asset_path = model_asset_path
        self.capture = None
        self.face_mesh = None
        self.face_mesh_factory = face_mesh_factory or self._default_face_mesh_factory
        self.capture_factory = capture_factory or cv2.VideoCapture
        self._video_timestamp_ms = 0
        self._last_read_monotonic: float | None = None
        self._fused_history: deque[dict[str, float]] = deque(maxlen=TEMPORAL_FUSION_WINDOW_SIZE)
        self._vertical_blendshape_history: deque[float] = deque(maxlen=BLENDSHAPE_BASELINE_WINDOW_SIZE)
        self._vertical_openness_history: deque[float] = deque(maxlen=BLENDSHAPE_BASELINE_WINDOW_SIZE)

    @staticmethod
    def is_available() -> bool:
        return mp is not None and (_supports_legacy_face_mesh_api() or _supports_tasks_face_landmarker_api())

    @staticmethod
    def describe_capabilities() -> dict[str, Any]:
        blendshape_mode = _blendshape_capability_mode()
        signal_notes = "Supports richer facial control signals from MediaPipe landmarks and blendshapes."
        if blendshape_mode == "legacy_facemesh_iris_only":
            signal_notes = (
                "Legacy MediaPipe FaceMesh mode is available, but blendshape gaze signals are not. "
                "Tracking falls back to iris-driven gaze only, which is noticeably weaker horizontally."
            )
        return {
            "tracker": "mediapipe",
            "supported_signals": [
                "mouth_open",
                "left_eye_closed",
                "right_eye_closed",
                "head_yaw",
                "head_pitch",
                "head_roll",
            ],
            "blendshape_tracking": blendshape_mode,
            "horizontal_signal_quality": "degraded" if blendshape_mode == "legacy_facemesh_iris_only" else "full",
            "signal_notes": signal_notes,
        }

    def open(self) -> None:
        if not self.is_available():
            raise WebcamProviderError(
                "mediapipe_unavailable",
                "MediaPipe is not installed. Install the mediapipe package to use this tracker backend.",
            )
        if self.capture is not None and self.face_mesh is not None:
            return
        silence_opencv_logging()
        with suppress_native_stderr():
            capture = self.capture_factory(self.camera_index)
            is_opened = capture.isOpened()
            if not is_opened:
                capture.release()
        if not is_opened:
            raise WebcamProviderError(
                "camera_unavailable",
                f"Unable to open webcam at index {self.camera_index}",
                payload={"camera_index": self.camera_index},
            )
        self.capture = capture
        try:
            self.face_mesh = self.face_mesh_factory()
        except WebcamProviderError:
            capture.release()
            self.capture = None
            raise
        except Exception as error:
            capture.release()
            self.capture = None
            raise WebcamProviderError(
                "mediapipe_init_failed",
                f"Failed to initialize the MediaPipe face mesh tracker: {error}",
            ) from error

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.face_mesh is not None:
            close = getattr(self.face_mesh, "close", None)
            if callable(close):
                close()
            self.face_mesh = None
        self.reset_tracking_state(preserve_video_timestamp=False, preserve_blendshape_baseline=False)

    def reset_tracking_state(
        self,
        *,
        preserve_video_timestamp: bool | None = None,
        preserve_blendshape_baseline: bool = False,
    ) -> None:
        if preserve_video_timestamp is None:
            preserve_video_timestamp = self.face_mesh is not None and hasattr(self.face_mesh, "detect_for_video")
        if not preserve_video_timestamp:
            self._video_timestamp_ms = 0
        self._last_read_monotonic = None
        self._fused_history.clear()
        if not preserve_blendshape_baseline:
            self._vertical_blendshape_history.clear()
            self._vertical_openness_history.clear()

    def read(self, *, delta_ms: int = 100) -> WebcamGazeReading | None:
        if self.capture is None or self.face_mesh is None:
            self.open()
        assert self.capture is not None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        now = time.monotonic()
        delta_ms = effective_delta_ms(
            delta_ms,
            last_frame_monotonic=self._last_read_monotonic,
            current_monotonic=now,
        )
        self._last_read_monotonic = now
        return self.process_frame(frame, delta_ms=delta_ms)

    def process_frame(self, frame: np.ndarray, *, delta_ms: int = 100) -> WebcamGazeReading | None:
        if frame.size == 0:
            return None
        if self.face_mesh is None:
            self.open()
        assert self.face_mesh is not None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        landmarks, blendshapes = self._extract_landmarks(rgb, delta_ms=delta_ms)
        if not landmarks:
            return None

        height, width = frame.shape[:2]
        right_eye_measurement = _eye_measurement(landmarks, width, height, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, RIGHT_IRIS)
        left_eye_measurement = _eye_measurement(landmarks, width, height, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, LEFT_IRIS)
        eye_measurements = [measurement for measurement in (right_eye_measurement, left_eye_measurement) if measurement is not None]
        if not eye_measurements:
            return None

        head_pose = _head_pose_signals(landmarks, width, height)
        component_signals = _component_signals(
            eye_measurements,
            blendshapes,
            vertical_blendshape_override=_normalize_dynamic_blendshape_axis(
                _blendshape_vertical_signal(blendshapes),
                self._vertical_blendshape_history,
                min_samples=BLENDSHAPE_BASELINE_MIN_SAMPLES,
                gain=VERTICAL_BLENDSHAPE_BASELINE_GAIN,
            ),
            vertical_eyelid_override=_normalize_dynamic_openness_vertical(
                _mean_measurement_openness(eye_measurements),
                self._vertical_openness_history,
                min_samples=BLENDSHAPE_BASELINE_MIN_SAMPLES,
                gain=VERTICAL_OPENNESS_BASELINE_GAIN,
            ),
        )
        horizontal = _apply_head_pose_compensation(
            component_signals["horizontal"],
            head_pose["head_yaw"],
            weight=HEAD_YAW_COMPENSATION_WEIGHT,
        )
        vertical = _apply_head_pose_compensation(
            component_signals["vertical"],
            head_pose["head_pitch"],
            weight=HEAD_PITCH_COMPENSATION_WEIGHT,
        )
        axis_quality = _axis_quality(
            left_measurement=left_eye_measurement,
            right_measurement=right_eye_measurement,
            component_signals=component_signals,
        )
        base_confidence = _estimate_confidence(eye_measurements)
        face_bbox = _face_bbox(landmarks, width, height)
        signals = {
            "mouth_open": max(
                _mouth_open_signal(blendshapes),
                _mouth_open_landmark_signal(landmarks, width, height),
            ),
            "left_eye_closed": _eye_closed_signal(left_eye_measurement, blendshapes, side="Left"),
            "right_eye_closed": _eye_closed_signal(right_eye_measurement, blendshapes, side="Right"),
            **head_pose,
        }
        raw_sample = GazeSample(
            point=NormalizedPoint(x=_clamp(horizontal), y=_clamp(vertical)),
            confidence=base_confidence,
            delta_ms=delta_ms,
        )
        fused_point, fusion_quality = self._fuse_temporal_sample(
            raw_sample.point,
            x_weight=axis_quality["x_axis_confidence"],
            y_weight=axis_quality["y_axis_confidence"],
            wink_block=max(signals["left_eye_closed"], signals["right_eye_closed"]) >= TEMPORAL_FUSION_WINK_BLOCK_THRESHOLD,
        )
        confidence = round(
            min(
                0.98,
                max(
                    0.0,
                    base_confidence * 0.72
                    + ((axis_quality["x_axis_confidence"] + axis_quality["y_axis_confidence"]) / 2.0) * 0.28,
                ),
            ),
            3,
        )
        quality = {
            **axis_quality,
            **fusion_quality,
            "left_eye_present": left_eye_measurement is not None,
            "right_eye_present": right_eye_measurement is not None,
            "left_eye_x_ratio": _measurement_value(left_eye_measurement, "x_ratio", default=0.5),
            "right_eye_x_ratio": _measurement_value(right_eye_measurement, "x_ratio", default=0.5),
            "left_eye_y_ratio": _measurement_value(left_eye_measurement, "y_ratio", default=0.5),
            "right_eye_y_ratio": _measurement_value(right_eye_measurement, "y_ratio", default=0.5),
            "left_eye_x_reliability": _measurement_value(left_eye_measurement, "x_reliability"),
            "right_eye_x_reliability": _measurement_value(right_eye_measurement, "x_reliability"),
            "left_eye_y_reliability": _measurement_value(left_eye_measurement, "y_reliability"),
            "right_eye_y_reliability": _measurement_value(right_eye_measurement, "y_reliability"),
            "left_eye_openness": _measurement_value(left_eye_measurement, "openness"),
            "right_eye_openness": _measurement_value(right_eye_measurement, "openness"),
            "horizontal_iris_component": component_signals["iris_horizontal"],
            "vertical_iris_component": component_signals["iris_vertical"],
            "horizontal_blendshape_available": component_signals["blendshape_horizontal_available"],
            "vertical_blendshape_available": component_signals["blendshape_vertical_available"],
            "vertical_eyelid_available": component_signals["eyelid_vertical_available"],
            "horizontal_blendshape_value": component_signals["blendshape_horizontal"],
            "vertical_blendshape_value": component_signals["blendshape_vertical"],
            "vertical_eyelid_value": component_signals["eyelid_vertical"],
            "vertical_eyelid_noise_risk": component_signals["eyelid_noise_risk"],
            "raw_sample_x": raw_sample.point.x,
            "raw_sample_y": raw_sample.point.y,
            "fused_sample_x": fused_point.x,
            "fused_sample_y": fused_point.y,
        }

        return WebcamGazeReading(
            sample=GazeSample(
                point=fused_point,
                confidence=confidence,
                delta_ms=delta_ms,
            ),
            frame_size=(width, height),
            face_bbox=face_bbox,
            eye_boxes=tuple(measurement["box"] for measurement in eye_measurements),
            frame=frame.copy(),
            pupil_points=tuple(measurement["iris_center"] for measurement in eye_measurements),
            signals=signals,
            raw_sample=raw_sample,
            quality=quality,
        )

    def capture_average_sample(
        self,
        *,
        required_frames: int = 6,
        max_attempts: int | None = None,
        delta_ms: int = 100,
    ) -> WebcamSampleAggregate | None:
        self.reset_tracking_state(preserve_blendshape_baseline=True)
        if max_attempts is None:
            max_attempts = max(required_frames, 1) * 4
        warmup_remaining = min(CAPTURE_WARMUP_SUCCESS_FRAMES, max(0, max_attempts - max(required_frames, 1)))
        readings: list[WebcamGazeReading] = []
        attempted_frames = 0
        while attempted_frames < max_attempts and warmup_remaining > 0:
            attempted_frames += 1
            reading = self.read(delta_ms=delta_ms)
            if reading is None:
                continue
            warmup_remaining -= 1
        while attempted_frames < max_attempts and len(readings) < required_frames:
            attempted_frames += 1
            reading = self.read(delta_ms=delta_ms)
            if reading is None:
                continue
            readings.append(reading)
        if not readings:
            return None
        raw_samples = [reading.raw_sample or reading.sample for reading in readings]
        avg_x = sum(sample.point.x for sample in raw_samples) / len(raw_samples)
        avg_y = sum(sample.point.y for sample in raw_samples) / len(raw_samples)
        avg_confidence = sum(sample.confidence for sample in raw_samples) / len(raw_samples)
        return WebcamSampleAggregate(
            sample=GazeSample(
                point=NormalizedPoint(x=_clamp(avg_x), y=_clamp(avg_y)),
                confidence=max(0.0, min(1.0, avg_confidence)),
                delta_ms=delta_ms,
            ),
            used_frames=len(readings),
            attempted_frames=attempted_frames,
            signals=_average_numeric_signal_map(reading.signals for reading in readings),
            quality=_aggregate_quality_map(reading.quality for reading in readings),
        )

    def _fuse_temporal_sample(
        self,
        raw_point: NormalizedPoint,
        *,
        x_weight: float,
        y_weight: float,
        wink_block: bool,
    ) -> tuple[NormalizedPoint, dict[str, float | int | bool]]:
        if not wink_block:
            self._fused_history.append(
                {
                    "x": float(raw_point.x),
                    "y": float(raw_point.y),
                    "x_weight": max(TEMPORAL_FUSION_MIN_WEIGHT, float(x_weight)),
                    "y_weight": max(TEMPORAL_FUSION_MIN_WEIGHT, float(y_weight)),
                }
            )

        if not self._fused_history:
            return raw_point, {
                "temporal_fusion_active": False,
                "fusion_samples": 0,
                "fusion_wink_freeze": wink_block,
                "fused_delta_x": 0.0,
                "fused_delta_y": 0.0,
            }

        fused_x = _trimmed_weighted_axis_value(self._fused_history, axis_key="x", weight_key="x_weight")
        fused_y = _trimmed_weighted_axis_value(self._fused_history, axis_key="y", weight_key="y_weight")
        return NormalizedPoint(x=_clamp(fused_x), y=_clamp(fused_y)), {
            "temporal_fusion_active": len(self._fused_history) > 1,
            "fusion_samples": len(self._fused_history),
            "fusion_wink_freeze": wink_block,
            "fused_delta_x": abs(float(raw_point.x) - fused_x),
            "fused_delta_y": abs(float(raw_point.y) - fused_y),
        }

    def _default_face_mesh_factory(self) -> Any:
        assert mp is not None
        if _supports_legacy_face_mesh_api():
            return mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        if _supports_tasks_face_landmarker_api():
            model_path = self._resolved_model_asset_path()
            base_options = mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            return mp.tasks.vision.FaceLandmarker.create_from_options(options)
        raise WebcamProviderError(
            "mediapipe_api_unavailable",
            "The installed MediaPipe package does not expose either the legacy FaceMesh API or the newer Face Landmarker Tasks API.",
        )

    def _resolved_model_asset_path(self) -> Path:
        if not self.model_asset_path:
            raise WebcamProviderError(
                "mediapipe_model_required",
                (
                    "This MediaPipe install exposes the Tasks API, which requires a local face landmarker model. "
                    "Download a Face Landmarker `.task` file and pass `--tracker-model-path /absolute/path/to/face_landmarker.task`."
                ),
            )
        model_path = Path(self.model_asset_path).expanduser()
        if not model_path.is_file():
            raise WebcamProviderError(
                "mediapipe_model_not_found",
                f"Face Landmarker model file was not found at {model_path}",
                payload={"model_asset_path": str(model_path)},
            )
        return model_path

    def _extract_landmarks(self, rgb_frame: np.ndarray, *, delta_ms: int) -> tuple[list[Any] | None, dict[str, float]]:
        tracker = self.face_mesh
        assert tracker is not None
        if hasattr(tracker, "process"):
            result = tracker.process(rgb_frame)
            face_landmarks = getattr(result, "multi_face_landmarks", None)
            if not face_landmarks:
                return None, {}
            return list(face_landmarks[0].landmark), {}
        if hasattr(tracker, "detect_for_video"):
            self._video_timestamp_ms += max(1, int(delta_ms))
            if mp is not None:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
            else:  # pragma: no cover - exercised by fake task tracker tests
                image = rgb_frame
            result = tracker.detect_for_video(image, self._video_timestamp_ms)
            face_landmarks = getattr(result, "face_landmarks", None)
            if not face_landmarks:
                return None, {}
            return list(face_landmarks[0]), _blendshape_map(result)
        if hasattr(tracker, "detect"):
            if mp is not None:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
            else:  # pragma: no cover - exercised by fake task tracker tests
                image = rgb_frame
            result = tracker.detect(image)
            face_landmarks = getattr(result, "face_landmarks", None)
            if not face_landmarks:
                return None, {}
            return list(face_landmarks[0]), _blendshape_map(result)
        raise WebcamProviderError(
            "mediapipe_tracker_unsupported",
            "The configured MediaPipe tracker does not support a known detection interface.",
        )


def _supports_legacy_face_mesh_api() -> bool:
    solutions = getattr(mp, "solutions", None)
    face_mesh = getattr(solutions, "face_mesh", None)
    return face_mesh is not None


def _supports_tasks_face_landmarker_api() -> bool:
    tasks = getattr(mp, "tasks", None)
    vision = getattr(tasks, "vision", None)
    return (
        tasks is not None
        and getattr(tasks, "BaseOptions", None) is not None
        and vision is not None
        and getattr(vision, "FaceLandmarker", None) is not None
        and getattr(vision, "FaceLandmarkerOptions", None) is not None
        and getattr(vision, "RunningMode", None) is not None
    )


def _blendshape_capability_mode() -> str:
    if _supports_tasks_face_landmarker_api():
        return "tasks_api"
    if _supports_legacy_face_mesh_api():
        return "legacy_facemesh_iris_only"
    return "unavailable"


def _eye_measurement(
    landmarks: list[Any],
    frame_width: int,
    frame_height: int,
    corners: tuple[int, int],
    lids: tuple[int, int],
    iris_indices: tuple[int, ...],
) -> EyeMeasurement | None:
    corner_a = _point(landmarks, corners[0], frame_width, frame_height)
    corner_b = _point(landmarks, corners[1], frame_width, frame_height)
    lid_top = _point(landmarks, lids[0], frame_width, frame_height)
    lid_bottom = _point(landmarks, lids[1], frame_width, frame_height)
    iris_points = [_point(landmarks, index, frame_width, frame_height) for index in iris_indices]

    eye_width = math.dist(corner_a, corner_b)
    eye_height = math.dist(lid_top, lid_bottom)
    if eye_width < 4.0 or eye_height < 1.0:
        return None

    iris_x = sum(point[0] for point in iris_points) / len(iris_points)
    iris_y = sum(point[1] for point in iris_points) / len(iris_points)
    min_x = min(corner_a[0], corner_b[0])
    max_x = max(corner_a[0], corner_b[0])
    min_y = min(lid_top[1], lid_bottom[1])
    max_y = max(lid_top[1], lid_bottom[1])
    if max_x - min_x < 1.0 or max_y - min_y < 1.0:
        return None

    padding_x = eye_width * 0.18
    padding_y = max(eye_height * 0.9, 2.0)
    box_left = int(max(0, round(min_x - padding_x)))
    box_right = int(min(frame_width - 1, round(max_x + padding_x)))
    box_top = int(max(0, round(min_y - padding_y)))
    box_bottom = int(min(frame_height - 1, round(max_y + padding_y)))

    horizontal_axis = (corner_b[0] - corner_a[0], corner_b[1] - corner_a[1])
    horizontal_axis_length_sq = max(1.0, horizontal_axis[0] ** 2 + horizontal_axis[1] ** 2)
    vertical_axis = (lid_bottom[0] - lid_top[0], lid_bottom[1] - lid_top[1])
    vertical_axis_length_sq = max(1.0, vertical_axis[0] ** 2 + vertical_axis[1] ** 2)
    iris_offset_horizontal = (iris_x - corner_a[0], iris_y - corner_a[1])
    iris_offset_vertical = (iris_x - lid_top[0], iris_y - lid_top[1])
    x_ratio = _clamp(
        (iris_offset_horizontal[0] * horizontal_axis[0] + iris_offset_horizontal[1] * horizontal_axis[1])
        / horizontal_axis_length_sq
    )
    y_ratio = _clamp(
        (iris_offset_vertical[0] * vertical_axis[0] + iris_offset_vertical[1] * vertical_axis[1])
        / vertical_axis_length_sq
    )
    openness = eye_height / eye_width
    openness_score = _clamp((openness - 0.05) / 0.12)
    x_margin_score = _clamp((min(x_ratio, 1.0 - x_ratio) - 0.01) / 0.18)
    y_margin_score = _clamp((min(y_ratio, 1.0 - y_ratio) - 0.01) / 0.18)

    return {
        "x_ratio": x_ratio,
        "y_ratio": y_ratio,
        "openness": openness,
        "x_reliability": _clamp(openness_score * 0.65 + x_margin_score * 0.35),
        "y_reliability": _clamp(openness_score * 0.65 + y_margin_score * 0.35),
        "box": (
            box_left,
            box_top,
            max(1, box_right - box_left),
            max(1, box_bottom - box_top),
        ),
        "iris_center": (int(round(iris_x)), int(round(iris_y))),
    }


def _estimate_confidence(eye_measurements: list[EyeMeasurement]) -> float:
    openness = sum(measurement["openness"] for measurement in eye_measurements) / len(eye_measurements)
    openness_score = _clamp((openness - 0.06) / 0.12)
    return round(min(0.98, 0.56 + 0.16 * len(eye_measurements) + 0.14 * openness_score), 3)


def _component_signals(
    eye_measurements: list[EyeMeasurement],
    blendshapes: dict[str, float],
    *,
    vertical_blendshape_override: float | None = None,
    vertical_eyelid_override: float | None = None,
) -> dict[str, float | bool]:
    iris_horizontal = 1.0 - sum(measurement["x_ratio"] for measurement in eye_measurements) / len(eye_measurements)
    iris_vertical = sum(measurement["y_ratio"] for measurement in eye_measurements) / len(eye_measurements)
    blendshape_horizontal = _blendshape_horizontal_signal(blendshapes)
    blendshape_vertical = vertical_blendshape_override
    if blendshape_vertical is None:
        blendshape_vertical = _blendshape_vertical_signal(blendshapes)
    eyelid_vertical = vertical_eyelid_override
    return {
        "iris_horizontal": _clamp(iris_horizontal),
        "iris_vertical": _clamp(iris_vertical),
        "blendshape_horizontal": _clamp(blendshape_horizontal) if blendshape_horizontal is not None else 0.5,
        "blendshape_vertical": _clamp(blendshape_vertical) if blendshape_vertical is not None else 0.5,
        "eyelid_vertical": _clamp(eyelid_vertical) if eyelid_vertical is not None else 0.5,
        "eyelid_noise_risk": _vertical_eyelid_noise_risk(blendshapes),
        "blendshape_horizontal_available": blendshape_horizontal is not None,
        "blendshape_vertical_available": blendshape_vertical is not None,
        "eyelid_vertical_available": eyelid_vertical is not None,
        "horizontal": _clamp(
            _fused_axis_component(
                iris_horizontal,
                blendshape_horizontal,
                blendshape_weight=HORIZONTAL_BLENDSHAPE_WEIGHT,
                iris_weight=HORIZONTAL_IRIS_WEIGHT,
            )
        ),
        "vertical": _clamp(
            _fused_vertical_component(
                iris_vertical,
                blendshape_vertical,
                eyelid_vertical,
                blendshape_weight=VERTICAL_BLENDSHAPE_WEIGHT,
                iris_weight=VERTICAL_IRIS_WEIGHT,
                eyelid_weight=VERTICAL_EYELID_WEIGHT,
                eyelid_noise_risk=float(_vertical_eyelid_noise_risk(blendshapes)),
            )
        ),
    }


def _blendshape_horizontal_signal(blendshapes: dict[str, float]) -> float | None:
    if not blendshapes:
        return None
    left_score = (
        blendshapes.get("eyeLookOutLeft", 0.0)
        + blendshapes.get("eyeLookInRight", 0.0)
    ) / 2.0
    right_score = (
        blendshapes.get("eyeLookInLeft", 0.0)
        + blendshapes.get("eyeLookOutRight", 0.0)
    ) / 2.0
    return _clamp(0.5 + (right_score - left_score) * 1.8)


def _blendshape_vertical_signal(blendshapes: dict[str, float]) -> float | None:
    if not blendshapes:
        return None
    up_score = (
        blendshapes.get("eyeLookUpLeft", 0.0)
        + blendshapes.get("eyeLookUpRight", 0.0)
    ) / 2.0
    down_score = (
        blendshapes.get("eyeLookDownLeft", 0.0)
        + blendshapes.get("eyeLookDownRight", 0.0)
    ) / 2.0
    return _clamp(0.5 + (down_score - up_score) * 1.8)


def _normalize_dynamic_blendshape_axis(
    value: float | None,
    history: deque[float],
    *,
    min_samples: int,
    gain: float,
) -> float | None:
    if value is None:
        return None
    clamped_value = _clamp(float(value))
    required_samples = max(1, int(min_samples))
    baseline: float | None = None
    if len(history) >= required_samples:
        baseline = float(np.median(np.asarray(history, dtype=float)))
    history.append(clamped_value)
    if baseline is None:
        return clamped_value
    return _clamp(0.5 + (clamped_value - baseline) * max(0.0, float(gain)))


def _normalize_dynamic_openness_vertical(
    value: float | None,
    history: deque[float],
    *,
    min_samples: int,
    gain: float,
) -> float | None:
    if value is None:
        return None
    openness = max(0.0, float(value))
    required_samples = max(1, int(min_samples))
    baseline: float | None = None
    if len(history) >= required_samples:
        baseline = float(np.median(np.asarray(history, dtype=float)))
    history.append(openness)
    if baseline is None:
        return None
    normalized = _clamp(0.5 + (baseline - openness) * max(0.0, float(gain)))
    if normalized <= 0.5:
        return normalized
    partial_squint_risk = _clamp((baseline - openness - VERTICAL_OPENNESS_PARTIAL_SQUINT_THRESHOLD) / 0.08)
    if partial_squint_risk <= 0.0:
        return normalized
    attenuation = 1.0 - VERTICAL_EYELID_SQUINT_ATTENUATION * partial_squint_risk
    return _clamp(0.5 + (normalized - 0.5) * max(0.0, attenuation))


def _fused_axis_component(
    iris_component: float,
    blendshape_component: float | None,
    *,
    blendshape_weight: float,
    iris_weight: float,
) -> float:
    if blendshape_component is None:
        return _clamp(iris_component)
    return _clamp(blendshape_component * blendshape_weight + iris_component * iris_weight)


def _fused_vertical_component(
    iris_component: float,
    blendshape_component: float | None,
    eyelid_component: float | None,
    *,
    blendshape_weight: float,
    iris_weight: float,
    eyelid_weight: float,
    eyelid_noise_risk: float = 0.0,
) -> float:
    values = [_clamp(iris_component)]
    weights = [max(0.12, float(iris_weight))]

    if blendshape_component is not None:
        blendshape_agreement = _support_agreement_score(iris_component, blendshape_component)
        blendshape_excursion = _clamp(abs(float(blendshape_component) - 0.5) / 0.35)
        adjusted_blendshape_weight = max(
            0.0,
            float(blendshape_weight) * (0.25 + 0.50 * blendshape_excursion + 0.25 * blendshape_agreement),
        )
        if eyelid_component is not None:
            adjusted_blendshape_weight *= 0.05 + 0.95 * _support_agreement_score(eyelid_component, blendshape_component)
        if adjusted_blendshape_weight > 0.0:
            values.append(_clamp(blendshape_component))
            weights.append(adjusted_blendshape_weight)

    if eyelid_component is not None:
        eyelid_agreement = _support_agreement_score(iris_component, eyelid_component)
        eyelid_excursion = _clamp(abs(float(eyelid_component) - 0.5) / 0.28)
        adjusted_eyelid_weight = max(
            0.0,
            float(eyelid_weight) * (0.35 + 0.65 * eyelid_agreement) * (0.45 + 0.55 * eyelid_excursion),
        )
        if float(eyelid_component) >= 0.5:
            adjusted_eyelid_weight *= 1.0 - _clamp(float(eyelid_noise_risk)) * VERTICAL_EYELID_SQUINT_ATTENUATION
        if adjusted_eyelid_weight > 0.0:
            values.append(_clamp(eyelid_component))
            weights.append(adjusted_eyelid_weight)

    total_weight = sum(weights)
    if total_weight <= 0.0:
        return _clamp(iris_component)
    return _clamp(sum(value * weight for value, weight in zip(values, weights)) / total_weight)


def _axis_quality(
    *,
    left_measurement: EyeMeasurement | None,
    right_measurement: EyeMeasurement | None,
    component_signals: dict[str, float | bool],
) -> dict[str, float | bool]:
    left_x_reliability = _measurement_value(left_measurement, "x_reliability")
    right_x_reliability = _measurement_value(right_measurement, "x_reliability")
    left_y_reliability = _measurement_value(left_measurement, "y_reliability")
    right_y_reliability = _measurement_value(right_measurement, "y_reliability")
    x_reliabilities = [
        reliability
        for measurement, reliability in ((left_measurement, left_x_reliability), (right_measurement, right_x_reliability))
        if measurement is not None
    ]
    y_reliabilities = [
        reliability
        for measurement, reliability in ((left_measurement, left_y_reliability), (right_measurement, right_y_reliability))
        if measurement is not None
    ]
    if left_measurement is None or right_measurement is None:
        horizontal_disagreement = 0.0
        vertical_disagreement = 0.0
    else:
        horizontal_disagreement = abs(
            _measurement_value(left_measurement, "x_ratio")
            - _measurement_value(right_measurement, "x_ratio")
        )
        vertical_disagreement = abs(
            _measurement_value(left_measurement, "y_ratio")
            - _measurement_value(right_measurement, "y_ratio")
        )
    horizontal_disagreement_score = _clamp(1.0 - horizontal_disagreement / 0.35)
    vertical_disagreement_score = _clamp(1.0 - vertical_disagreement / 0.35)
    horizontal_component_alignment = _component_alignment_score(
        float(component_signals["iris_horizontal"]),
        float(component_signals["blendshape_horizontal"]),
        enabled=bool(component_signals["blendshape_horizontal_available"]),
    )
    vertical_component_alignment = _component_alignment_score(
        float(component_signals["iris_vertical"]),
        float(component_signals["blendshape_vertical"]),
        enabled=bool(component_signals["blendshape_vertical_available"]),
    )
    vertical_eyelid_alignment = _component_alignment_score(
        float(component_signals["iris_vertical"]),
        float(component_signals["eyelid_vertical"]),
        enabled=bool(component_signals["eyelid_vertical_available"]),
    )
    vertical_support_alignment = _mean_sequence(
        [vertical_component_alignment, vertical_eyelid_alignment],
        default=0.8,
    )
    x_axis_confidence = _clamp(
        (_mean_sequence(x_reliabilities, default=0.0)) * 0.55
        + horizontal_disagreement_score * 0.30
        + horizontal_component_alignment * 0.15
    )
    y_axis_confidence = _clamp(
        (_mean_sequence(y_reliabilities, default=0.0)) * 0.52
        + vertical_disagreement_score * 0.24
        + vertical_support_alignment * 0.24
    )
    return {
        "horizontal_disagreement": horizontal_disagreement,
        "vertical_disagreement": vertical_disagreement,
        "horizontal_disagreement_score": horizontal_disagreement_score,
        "vertical_disagreement_score": vertical_disagreement_score,
        "horizontal_component_alignment": horizontal_component_alignment,
        "vertical_component_alignment": vertical_component_alignment,
        "vertical_eyelid_alignment": vertical_eyelid_alignment,
        "x_axis_confidence": x_axis_confidence,
        "y_axis_confidence": y_axis_confidence,
        "x_saturated": _axis_saturated([measurement for measurement in (left_measurement, right_measurement) if measurement is not None], axis="x"),
        "y_saturated": _axis_saturated([measurement for measurement in (left_measurement, right_measurement) if measurement is not None], axis="y"),
    }


def _component_alignment_score(iris_component: float, blendshape_component: float, *, enabled: bool) -> float:
    if not enabled:
        return 0.8
    return _clamp(1.0 - abs(float(iris_component) - float(blendshape_component)) / 0.35)


def _support_agreement_score(reference_component: float, support_component: float) -> float:
    return _clamp(1.0 - abs(float(reference_component) - float(support_component)) / 0.28)


def _mean_sequence(values: list[float], *, default: float) -> float:
    if not values:
        return float(default)
    return sum(values) / len(values)


def _mean_measurement_openness(eye_measurements: list[EyeMeasurement]) -> float | None:
    if not eye_measurements:
        return None
    return sum(float(measurement["openness"]) for measurement in eye_measurements) / len(eye_measurements)


def _vertical_eyelid_noise_risk(blendshapes: dict[str, float]) -> float:
    if not blendshapes:
        return 0.0
    blink_score = (
        blendshapes.get("eyeBlinkLeft", 0.0)
        + blendshapes.get("eyeBlinkRight", 0.0)
    ) / 2.0
    squint_score = (
        blendshapes.get("eyeSquintLeft", 0.0)
        + blendshapes.get("eyeSquintRight", 0.0)
    ) / 2.0
    return _clamp(max(squint_score, blink_score * 0.75))


def _axis_saturated(eye_measurements: list[EyeMeasurement], *, axis: str) -> bool:
    key = f"{axis}_ratio"
    return any(min(float(measurement[key]), 1.0 - float(measurement[key])) <= 0.04 for measurement in eye_measurements)


def _measurement_value(measurement: EyeMeasurement | None, key: str, *, default: float = 0.0) -> float:
    if measurement is None:
        return float(default)
    return float(measurement.get(key, default))


def _blendshape_map(result: Any) -> dict[str, float]:
    face_blendshapes = getattr(result, "face_blendshapes", None)
    if not face_blendshapes:
        return {}
    categories = face_blendshapes[0]
    mapping: dict[str, float] = {}
    for category in categories:
        name = getattr(category, "category_name", None) or getattr(category, "display_name", None)
        score = getattr(category, "score", None)
        if not name or score is None:
            continue
        mapping[str(name)] = float(score)
    return mapping


def _mouth_open_signal(blendshapes: dict[str, float]) -> float:
    return _clamp(
        max(
            blendshapes.get("jawOpen", 0.0),
            blendshapes.get("mouthOpen", 0.0),
        )
    )


def _eye_closed_signal(
    measurement: EyeMeasurement | None,
    blendshapes: dict[str, float],
    *,
    side: str,
) -> float:
    blink_score = _clamp(blendshapes.get(f"eyeBlink{side}", 0.0))
    squint_score = _clamp(blendshapes.get(f"eyeSquint{side}", 0.0))
    if measurement is None:
        return max(blink_score, squint_score * 0.82)
    openness = float(measurement["openness"])
    landmark_closed = _clamp((0.16 - openness) / 0.12)
    fused_closed = max(blink_score, landmark_closed, squint_score * 0.82)
    if blink_score > 0.0 or squint_score > 0.0:
        fused_closed = max(fused_closed, _clamp(max(blink_score, squint_score) * 0.72 + landmark_closed * 0.28))
    return fused_closed


def _mouth_open_landmark_signal(landmarks: list[Any], frame_width: int, frame_height: int) -> float:
    left_corner = _point(landmarks, MOUTH_CORNERS[0], frame_width, frame_height)
    right_corner = _point(landmarks, MOUTH_CORNERS[1], frame_width, frame_height)
    upper_lip = _point(landmarks, MOUTH_LIPS[0], frame_width, frame_height)
    lower_lip = _point(landmarks, MOUTH_LIPS[1], frame_width, frame_height)
    mouth_width = math.dist(left_corner, right_corner)
    if mouth_width < 6.0:
        return 0.0
    mouth_height = math.dist(upper_lip, lower_lip)
    openness = mouth_height / mouth_width
    return _clamp((openness - 0.02) / 0.20)


def _head_pose_signals(landmarks: list[Any], frame_width: int, frame_height: int) -> dict[str, float]:
    left_face = _point(landmarks, FACE_SIDE_POINTS[0], frame_width, frame_height)
    right_face = _point(landmarks, FACE_SIDE_POINTS[1], frame_width, frame_height)
    top_face = _point(landmarks, FACE_VERTICAL_POINTS[0], frame_width, frame_height)
    bottom_face = _point(landmarks, FACE_VERTICAL_POINTS[1], frame_width, frame_height)
    nose_tip = _point(landmarks, NOSE_TIP, frame_width, frame_height)
    right_outer_eye = _point(landmarks, OUTER_EYE_CORNERS[0], frame_width, frame_height)
    left_outer_eye = _point(landmarks, OUTER_EYE_CORNERS[1], frame_width, frame_height)

    face_width = max(1.0, math.dist(left_face, right_face))
    face_height = max(1.0, math.dist(top_face, bottom_face))
    face_center_x = (left_face[0] + right_face[0]) / 2.0
    face_center_y = (top_face[1] + bottom_face[1]) / 2.0
    yaw = _clamp_signed((nose_tip[0] - face_center_x) / (face_width * 0.18))
    pitch = _clamp_signed((nose_tip[1] - face_center_y) / (face_height * 0.16))
    roll_radians = math.atan2(left_outer_eye[1] - right_outer_eye[1], left_outer_eye[0] - right_outer_eye[0])
    roll = _clamp_signed(roll_radians / math.radians(18.0))
    return {
        "head_yaw": yaw,
        "head_pitch": pitch,
        "head_roll": roll,
    }


def _apply_head_pose_compensation(value: float, pose_signal: float, *, weight: float) -> float:
    pose = _clamp_signed(pose_signal)
    base_weight = max(0.0, float(weight))
    # Keep mid-range pose behavior unchanged while scaling compensation up for
    # larger head movement and down for near-neutral pose.
    effective_weight = base_weight * (0.6 + abs(pose) * 0.8)
    return _clamp(value - pose * effective_weight)


def _trimmed_weighted_axis_value(
    history: deque[dict[str, float]],
    *,
    axis_key: str,
    weight_key: str,
) -> float:
    points = sorted((float(entry[axis_key]), max(TEMPORAL_FUSION_MIN_WEIGHT, float(entry[weight_key]))) for entry in history)
    if len(points) >= 4:
        points = points[1:-1]
    total_weight = sum(weight for _, weight in points)
    if total_weight <= 0.0:
        return sum(value for value, _ in points) / max(1, len(points))
    return sum(value * weight for value, weight in points) / total_weight


def _average_numeric_signal_map(signal_maps: Any) -> dict[str, float] | None:
    aggregated: dict[str, list[float]] = {}
    for signal_map in signal_maps:
        if not isinstance(signal_map, dict):
            continue
        for key, value in signal_map.items():
            if isinstance(value, bool):
                aggregated.setdefault(key, []).append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                aggregated.setdefault(key, []).append(float(value))
    if not aggregated:
        return None
    return {key: sum(values) / len(values) for key, values in aggregated.items()}


def _aggregate_quality_map(quality_maps: Any) -> dict[str, float] | None:
    aggregated: dict[str, list[float]] = {}
    boolean_keys = {
        "x_saturated",
        "y_saturated",
        "temporal_fusion_active",
        "fusion_wink_freeze",
        "left_eye_present",
        "right_eye_present",
        "horizontal_blendshape_available",
        "vertical_blendshape_available",
        "vertical_eyelid_available",
    }
    for quality_map in quality_maps:
        if not isinstance(quality_map, dict):
            continue
        for key, value in quality_map.items():
            if key in boolean_keys:
                aggregated.setdefault(f"{key}_ratio", []).append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                aggregated.setdefault(key, []).append(float(value))
    if not aggregated:
        return None
    return {key: sum(values) / len(values) for key, values in aggregated.items()}


def _face_bbox(landmarks: list[Any], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    xs = [max(0.0, min(frame_width - 1, landmark.x * frame_width)) for landmark in landmarks]
    ys = [max(0.0, min(frame_height - 1, landmark.y * frame_height)) for landmark in landmarks]
    min_x = int(round(min(xs)))
    max_x = int(round(max(xs)))
    min_y = int(round(min(ys)))
    max_y = int(round(max(ys)))
    return (
        min_x,
        min_y,
        max(1, max_x - min_x),
        max(1, max_y - min_y),
    )


def _point(landmarks: list[Any], index: int, frame_width: int, frame_height: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return (
        max(0.0, min(frame_width - 1, landmark.x * frame_width)),
        max(0.0, min(frame_height - 1, landmark.y * frame_height)),
    )


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))
