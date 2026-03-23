"""MediaPipe-backed webcam gaze provider using face mesh and iris landmarks."""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import NormalizedPoint
from .gaze import GazeSample
from .opencv_provider import (
    WebcamGazeReading,
    WebcamProviderError,
    WebcamSampleAggregate,
    _clamp,
    _silence_opencv_logging,
    _suppress_native_stderr,
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

VERTICAL_BLENDSHAPE_WEIGHT = 0.85
VERTICAL_IRIS_WEIGHT = 0.15


class MediaPipeWebcamGazeProvider:
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

    @staticmethod
    def is_available() -> bool:
        return mp is not None and (_supports_legacy_face_mesh_api() or _supports_tasks_face_landmarker_api())

    def open(self) -> None:
        if not self.is_available():
            raise WebcamProviderError(
                "mediapipe_unavailable",
                "MediaPipe is not installed. Install the mediapipe package to use this tracker backend.",
            )
        if self.capture is not None and self.face_mesh is not None:
            return
        _silence_opencv_logging()
        with _suppress_native_stderr():
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

    def read(self, *, delta_ms: int = 100) -> WebcamGazeReading | None:
        if self.capture is None or self.face_mesh is None:
            self.open()
        assert self.capture is not None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
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
        eye_measurements = [
            measurement
            for measurement in (
                _eye_measurement(landmarks, width, height, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, RIGHT_IRIS),
                _eye_measurement(landmarks, width, height, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, LEFT_IRIS),
            )
            if measurement is not None
        ]
        if not eye_measurements:
            return None

        horizontal = 1.0 - sum(measurement["x_ratio"] for measurement in eye_measurements) / len(eye_measurements)
        vertical = _vertical_signal(eye_measurements, blendshapes)
        confidence = _estimate_confidence(eye_measurements)
        face_bbox = _face_bbox(landmarks, width, height)

        return WebcamGazeReading(
            sample=GazeSample(
                point=NormalizedPoint(x=_clamp(horizontal), y=_clamp(vertical)),
                confidence=confidence,
                delta_ms=delta_ms,
            ),
            frame_size=(width, height),
            face_bbox=face_bbox,
            eye_boxes=tuple(measurement["box"] for measurement in eye_measurements),
            frame=frame.copy(),
            pupil_points=tuple(measurement["iris_center"] for measurement in eye_measurements),
        )

    def capture_average_sample(
        self,
        *,
        required_frames: int = 6,
        max_attempts: int | None = None,
        delta_ms: int = 100,
    ) -> WebcamSampleAggregate | None:
        if max_attempts is None:
            max_attempts = max(required_frames, 1) * 4
        readings: list[WebcamGazeReading] = []
        attempted_frames = 0
        while attempted_frames < max_attempts and len(readings) < required_frames:
            attempted_frames += 1
            reading = self.read(delta_ms=delta_ms)
            if reading is None:
                continue
            readings.append(reading)
        if not readings:
            return None
        avg_x = sum(reading.sample.point.x for reading in readings) / len(readings)
        avg_y = sum(reading.sample.point.y for reading in readings) / len(readings)
        avg_confidence = sum(reading.sample.confidence for reading in readings) / len(readings)
        return WebcamSampleAggregate(
            sample=GazeSample(
                point=NormalizedPoint(x=_clamp(avg_x), y=_clamp(avg_y)),
                confidence=max(0.0, min(1.0, avg_confidence)),
                delta_ms=delta_ms,
            ),
            used_frames=len(readings),
            attempted_frames=attempted_frames,
        )

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
                output_facial_transformation_matrixes=False,
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
            assert mp is not None
            self._video_timestamp_ms += max(1, int(delta_ms))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
            result = tracker.detect_for_video(image, self._video_timestamp_ms)
            face_landmarks = getattr(result, "face_landmarks", None)
            if not face_landmarks:
                return None, {}
            return list(face_landmarks[0]), _blendshape_map(result)
        if hasattr(tracker, "detect"):
            assert mp is not None
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
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


def _eye_measurement(
    landmarks: list[Any],
    frame_width: int,
    frame_height: int,
    corners: tuple[int, int],
    lids: tuple[int, int],
    iris_indices: tuple[int, ...],
) -> dict[str, Any] | None:
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

    return {
        "x_ratio": _clamp((iris_x - min_x) / (max_x - min_x)),
        "y_ratio": _clamp((iris_y - min_y) / (max_y - min_y)),
        "openness": eye_height / eye_width,
        "box": (
            box_left,
            box_top,
            max(1, box_right - box_left),
            max(1, box_bottom - box_top),
        ),
        "iris_center": (int(round(iris_x)), int(round(iris_y))),
    }


def _estimate_confidence(eye_measurements: list[dict[str, Any]]) -> float:
    openness = sum(measurement["openness"] for measurement in eye_measurements) / len(eye_measurements)
    openness_score = _clamp((openness - 0.06) / 0.12)
    return round(min(0.98, 0.56 + 0.16 * len(eye_measurements) + 0.14 * openness_score), 3)


def _vertical_signal(eye_measurements: list[dict[str, Any]], blendshapes: dict[str, float]) -> float:
    iris_vertical = sum(measurement["y_ratio"] for measurement in eye_measurements) / len(eye_measurements)
    if not blendshapes:
        return iris_vertical

    up_score = (
        blendshapes.get("eyeLookUpLeft", 0.0)
        + blendshapes.get("eyeLookUpRight", 0.0)
    ) / 2.0
    down_score = (
        blendshapes.get("eyeLookDownLeft", 0.0)
        + blendshapes.get("eyeLookDownRight", 0.0)
    ) / 2.0
    blendshape_vertical = _clamp(0.5 + (down_score - up_score) * 1.8)
    return _clamp(blendshape_vertical * VERTICAL_BLENDSHAPE_WEIGHT + iris_vertical * VERTICAL_IRIS_WEIGHT)


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
