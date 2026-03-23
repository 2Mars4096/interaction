"""OpenCV-backed webcam gaze provider for coarse large-target grounding."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import NormalizedPoint
from .gaze import GazeSample


class WebcamProviderError(RuntimeError):
    """Raised when the webcam provider cannot open or read from the camera."""

    def __init__(self, code: str, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.payload = payload or {}


@dataclass(frozen=True)
class WebcamGazeReading:
    sample: GazeSample
    frame_size: tuple[int, int]
    face_bbox: tuple[int, int, int, int]
    eye_boxes: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class WebcamSampleAggregate:
    sample: GazeSample
    used_frames: int
    attempted_frames: int


class OpenCVWebcamGazeProvider:
    """Coarse webcam gaze estimator using OpenCV cascades and pupil heuristics."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.capture: cv2.VideoCapture | None = None
        self.face_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        self.eye_cascades = [
            cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml")),
            cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml")),
        ]

    def open(self) -> None:
        if self.capture is not None:
            return
        _silence_opencv_logging()
        with _suppress_native_stderr():
            capture = cv2.VideoCapture(self.camera_index)
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

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def read(self, *, delta_ms: int = 100) -> WebcamGazeReading | None:
        if self.capture is None:
            self.open()
        assert self.capture is not None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        return self.process_frame(frame, delta_ms=delta_ms)

    def process_frame(self, frame: np.ndarray, *, delta_ms: int = 100) -> WebcamGazeReading | None:
        if frame.size == 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(60, 60))
        if len(faces) == 0:
            return None

        face = max(faces, key=lambda box: box[2] * box[3])
        fx, fy, fw, fh = face
        face_gray = gray[fy : fy + fh, fx : fx + fw]
        upper_half = face_gray[: max(1, fh // 2), :]
        eyes = self._detect_eyes(upper_half)
        if len(eyes) == 0:
            return None

        eye_boxes = self._select_eye_boxes(eyes)
        pupil_points: list[tuple[float, float]] = []
        for ex, ey, ew, eh in eye_boxes:
            eye_roi = upper_half[ey : ey + eh, ex : ex + ew]
            pupil = self._estimate_pupil_center(eye_roi)
            if pupil is None:
                continue
            pupil_points.append((fx + ex + pupil[0], fy + ey + pupil[1]))
        if not pupil_points:
            return None

        height, width = gray.shape[:2]
        avg_x = sum(point[0] for point in pupil_points) / len(pupil_points)
        avg_y = sum(point[1] for point in pupil_points) / len(pupil_points)
        confidence = min(1.0, 0.45 + 0.2 * len(pupil_points) + 0.15 * min(len(eye_boxes), 2))
        sample = GazeSample(
            point=NormalizedPoint(x=_clamp(avg_x / width), y=_clamp(avg_y / height)),
            confidence=confidence,
            delta_ms=delta_ms,
        )
        return WebcamGazeReading(
            sample=sample,
            frame_size=(width, height),
            face_bbox=(int(fx), int(fy), int(fw), int(fh)),
            eye_boxes=tuple((int(ex), int(ey), int(ew), int(eh)) for ex, ey, ew, eh in eye_boxes),
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

    @staticmethod
    def _select_eye_boxes(eyes: np.ndarray) -> list[tuple[int, int, int, int]]:
        boxes = sorted((tuple(map(int, eye)) for eye in eyes), key=lambda box: box[0])
        if len(boxes) <= 2:
            return boxes
        widest = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)[:2]
        return sorted(widest, key=lambda box: box[0])

    def _detect_eyes(self, upper_half: np.ndarray) -> np.ndarray:
        for cascade in self.eye_cascades:
            if cascade.empty():
                continue
            eyes = cascade.detectMultiScale(upper_half, scaleFactor=1.05, minNeighbors=4, minSize=(16, 16))
            if len(eyes) > 0:
                return eyes
        return np.empty((0, 4), dtype=np.int32)

    @staticmethod
    def _estimate_pupil_center(eye_roi: np.ndarray) -> tuple[float, float] | None:
        if eye_roi.size == 0:
            return None
        blurred = cv2.GaussianBlur(eye_roi, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return cx, cy


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _silence_opencv_logging() -> None:
    logging = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging is None:
        return
    set_log_level = getattr(logging, "setLogLevel", None)
    silent_level = getattr(logging, "LOG_LEVEL_SILENT", None)
    if callable(set_log_level) and silent_level is not None:
        set_log_level(silent_level)


@contextmanager
def _suppress_native_stderr():
    try:
        original_stderr = os.dup(2)
    except OSError:
        yield
        return

    devnull_fd = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(original_stderr, 2)
        os.close(original_stderr)
        if devnull_fd is not None:
            os.close(devnull_fd)
