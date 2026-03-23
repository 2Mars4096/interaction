import os

import cv2
import numpy as np

from interaction.vision import GazeSample, NormalizedPoint, OpenCVWebcamGazeProvider, WebcamGazeReading, WebcamProviderError


def test_estimate_pupil_center_detects_dark_blob() -> None:
    eye_roi = np.full((40, 80), 255, dtype=np.uint8)
    cv2.circle(eye_roi, center=(30, 20), radius=8, color=0, thickness=-1)

    center = OpenCVWebcamGazeProvider._estimate_pupil_center(eye_roi)

    assert center is not None
    assert center[0] == pytest_approx(30, tolerance=3.0)
    assert center[1] == pytest_approx(20, tolerance=3.0)


def test_select_eye_boxes_prefers_two_largest() -> None:
    provider = OpenCVWebcamGazeProvider()
    eyes = np.array(
        [
            [10, 10, 15, 15],
            [40, 12, 25, 20],
            [75, 10, 23, 18],
        ]
    )

    selected = provider._select_eye_boxes(eyes)

    assert len(selected) == 2
    assert selected[0][0] < selected[1][0]


def test_capture_average_sample_averages_successful_readings(monkeypatch) -> None:
    provider = OpenCVWebcamGazeProvider()
    readings = iter(
        [
            None,
            WebcamGazeReading(
                sample=GazeSample(point=NormalizedPoint(0.2, 0.3), confidence=0.7, delta_ms=100),
                frame_size=(640, 480),
                face_bbox=(0, 0, 10, 10),
                eye_boxes=(),
            ),
            WebcamGazeReading(
                sample=GazeSample(point=NormalizedPoint(0.4, 0.5), confidence=0.9, delta_ms=100),
                frame_size=(640, 480),
                face_bbox=(0, 0, 10, 10),
                eye_boxes=(),
            ),
        ]
    )

    monkeypatch.setattr(provider, "read", lambda delta_ms=100: next(readings))

    aggregate = provider.capture_average_sample(required_frames=2, max_attempts=3, delta_ms=100)

    assert aggregate is not None
    assert aggregate.used_frames == 2
    assert aggregate.attempted_frames == 3
    assert aggregate.sample.point.x == pytest_approx(0.3, tolerance=0.001)
    assert aggregate.sample.point.y == pytest_approx(0.4, tolerance=0.001)


def test_open_suppresses_native_camera_stderr(monkeypatch, capfd) -> None:
    class FakeCapture:
        def __init__(self, _camera_index: int) -> None:
            os.write(2, b"native camera backend noise\n")

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    monkeypatch.setattr("interaction.vision.opencv_provider.cv2.VideoCapture", FakeCapture)

    provider = OpenCVWebcamGazeProvider()

    try:
        provider.open()
    except WebcamProviderError as error:
        assert error.code == "camera_unavailable"
    else:
        raise AssertionError("Expected camera_unavailable when fake capture cannot open.")

    captured = capfd.readouterr()
    assert captured.err == ""


def pytest_approx(value: float, tolerance: float) -> float:
    """Small helper to avoid importing pytest in this cv2-focused test module."""
    return ApproxFloat(value, tolerance)


class ApproxFloat(float):
    def __new__(cls, value: float, tolerance: float):  # type: ignore[override]
        obj = float.__new__(cls, value)
        obj.tolerance = tolerance
        return obj

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (int, float)):
            return False
        return abs(float(self) - float(other)) <= self.tolerance
