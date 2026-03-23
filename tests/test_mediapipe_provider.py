from types import SimpleNamespace

import numpy as np

from interaction.vision import MediaPipeWebcamGazeProvider, WebcamProviderError


def test_mediapipe_provider_process_frame_extracts_eye_boxes_and_pupil_points() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceMesh(_landmarks_payload())

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.frame_size == (640, 480)
    assert len(reading.eye_boxes) == 2
    assert len(reading.pupil_points) == 2
    assert 0.0 <= reading.sample.point.x <= 1.0
    assert 0.0 <= reading.sample.point.y <= 1.0
    assert reading.sample.confidence >= 0.8


def test_mediapipe_provider_returns_none_without_landmarks() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceMesh([])

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert provider.process_frame(frame) is None


def test_mediapipe_provider_process_frame_accepts_tasks_style_result() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(_landmarks_payload())

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert len(reading.eye_boxes) == 2
    assert len(reading.pupil_points) == 2


def test_mediapipe_provider_uses_blendshapes_for_vertical_direction() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(
        _landmarks_payload(),
        blendshapes=[
            SimpleNamespace(category_name="eyeLookUpLeft", score=0.8),
            SimpleNamespace(category_name="eyeLookUpRight", score=0.7),
            SimpleNamespace(category_name="eyeLookDownLeft", score=0.0),
            SimpleNamespace(category_name="eyeLookDownRight", score=0.0),
        ],
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.sample.point.y < 0.3


def test_mediapipe_provider_requires_model_path_for_tasks_api() -> None:
    provider = MediaPipeWebcamGazeProvider(model_asset_path=None)

    try:
        provider._resolved_model_asset_path()
    except WebcamProviderError as error:
        assert error.code == "mediapipe_model_required"
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected a WebcamProviderError when no model path is configured.")


class FakeFaceMesh:
    def __init__(self, landmarks) -> None:
        self.landmarks = landmarks

    def process(self, _frame):
        if not self.landmarks:
            return SimpleNamespace(multi_face_landmarks=[])
        return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=self.landmarks)])


class FakeFaceLandmarker:
    def __init__(self, landmarks, blendshapes=None) -> None:
        self.landmarks = landmarks
        self.blendshapes = blendshapes or []

    def detect_for_video(self, _frame, _timestamp_ms):
        if not self.landmarks:
            return SimpleNamespace(face_landmarks=[], face_blendshapes=[])
        return SimpleNamespace(face_landmarks=[self.landmarks], face_blendshapes=[self.blendshapes])


def _landmarks_payload():
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    points = {
        10: (0.45, 0.22),
        152: (0.45, 0.8),
        234: (0.24, 0.5),
        454: (0.76, 0.5),
        33: (0.30, 0.40),
        133: (0.40, 0.40),
        159: (0.35, 0.38),
        145: (0.35, 0.42),
        468: (0.34, 0.40),
        469: (0.34, 0.395),
        470: (0.345, 0.40),
        471: (0.34, 0.405),
        472: (0.335, 0.40),
        362: (0.60, 0.40),
        263: (0.70, 0.40),
        386: (0.65, 0.38),
        374: (0.65, 0.42),
        473: (0.66, 0.40),
        474: (0.66, 0.395),
        475: (0.665, 0.40),
        476: (0.66, 0.405),
        477: (0.655, 0.40),
    }
    for index, (x, y) in points.items():
        landmarks[index] = SimpleNamespace(x=x, y=y)
    return landmarks
