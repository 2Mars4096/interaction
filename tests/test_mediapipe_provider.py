from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from interaction.vision import GazeSample, MediaPipeWebcamGazeProvider, NormalizedPoint, WebcamProviderError
from interaction.vision.mediapipe_provider import (
    _apply_head_pose_compensation,
    _head_pose_signals,
    _normalize_dynamic_blendshape_axis,
)


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
    assert reading.raw_sample is not None
    assert reading.quality is not None


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


def test_mediapipe_provider_supports_context_manager() -> None:
    provider = MediaPipeWebcamGazeProvider()
    calls: list[str] = []
    provider.open = lambda: calls.append("open")  # type: ignore[method-assign]
    provider.close = lambda: calls.append("close")  # type: ignore[method-assign]

    with provider as managed:
        assert managed is provider

    assert calls == ["open", "close"]


def test_mediapipe_provider_uses_blendshapes_for_vertical_direction() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(
        _landmarks_payload(),
        blendshapes=_vertical_blendshapes(up=0.75, down=0.0),
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.raw_sample is not None
    assert reading.raw_sample.point.y < 0.35


def test_normalize_dynamic_blendshape_axis_recenters_persistent_bias() -> None:
    history = deque([0.82, 0.83, 0.84, 0.85], maxlen=45)

    normalized = _normalize_dynamic_blendshape_axis(0.84, history, min_samples=4, gain=2.4)
    upward = _normalize_dynamic_blendshape_axis(0.72, history, min_samples=4, gain=2.4)

    assert normalized is not None
    assert upward is not None
    assert 0.5 <= normalized <= 0.56
    assert upward < 0.3


def test_mediapipe_provider_preserves_vertical_baseline_across_capture_steps() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeSequenceFaceLandmarker(
        [
            _frame_payload(blendshapes=_vertical_blendshapes(up=0.0, down=0.20))
            for _ in range(6)
        ]
        + [
            _frame_payload(blendshapes=_vertical_blendshapes(up=0.0, down=0.10))
            for _ in range(6)
        ]
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    provider.read = lambda *, delta_ms=100: provider.process_frame(frame, delta_ms=delta_ms)  # type: ignore[method-assign]

    first = provider.capture_average_sample(required_frames=6, max_attempts=6, delta_ms=120)
    second = provider.capture_average_sample(required_frames=6, max_attempts=6, delta_ms=120)

    assert first is not None
    assert second is not None
    assert second.sample.point.y < first.sample.point.y - 0.05


def test_mediapipe_provider_uses_eyelid_openness_as_vertical_support_signal() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeSequenceFaceLandmarker(
        [_frame_payload() for _ in range(4)]
        + [_frame_payload(overrides=_symmetric_lid_overrides(top=0.36, bottom=0.44)) for _ in range(4)]
        + [_frame_payload(overrides=_symmetric_lid_overrides(top=0.39, bottom=0.41)) for _ in range(4)]
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(4):
        provider.process_frame(frame, delta_ms=120)
    upward_readings = [provider.process_frame(frame, delta_ms=120) for _ in range(4)]
    downward_readings = [provider.process_frame(frame, delta_ms=120) for _ in range(4)]
    upward = upward_readings[-1]
    downward = downward_readings[-1]

    assert upward is not None
    assert downward is not None
    assert upward.raw_sample is not None
    assert downward.raw_sample is not None
    assert bool(upward.quality["vertical_eyelid_available"]) is True
    assert float(upward.quality["vertical_eyelid_value"]) < 0.5
    assert float(downward.quality["vertical_eyelid_value"]) > 0.5
    assert upward.raw_sample.point.y < downward.raw_sample.point.y - 0.07
    assert upward.sample.point.y < downward.sample.point.y - 0.05


def test_mediapipe_provider_downweights_conflicting_vertical_blendshape_with_eyelid_support() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeSequenceFaceLandmarker(
        [_frame_payload() for _ in range(4)]
        + [
            _frame_payload(
                overrides={
                    **_symmetric_lid_overrides(top=0.36, bottom=0.44),
                    **_shift_irises_vertical(vertical="up"),
                },
                blendshapes=_vertical_blendshapes(up=0.0, down=0.95),
            )
            for _ in range(4)
        ]
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(4):
        provider.process_frame(frame, delta_ms=120)
    reading = None
    for _ in range(4):
        reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.raw_sample is not None
    assert float(reading.quality["vertical_blendshape_value"]) > 0.8
    assert float(reading.quality["vertical_eyelid_value"]) < 0.5
    assert reading.raw_sample.point.y < 0.36
    assert reading.sample.point.y < 0.36


def test_mediapipe_provider_downweights_eyelid_vertical_signal_during_partial_squint() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeSequenceFaceLandmarker(
        [_frame_payload() for _ in range(4)]
        + [
            _frame_payload(
                overrides=_symmetric_lid_overrides(top=0.392, bottom=0.408),
                blendshapes=[
                    SimpleNamespace(category_name="eyeSquintLeft", score=0.72),
                    SimpleNamespace(category_name="eyeSquintRight", score=0.69),
                ],
            )
            for _ in range(4)
        ]
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(4):
        provider.process_frame(frame, delta_ms=120)
    reading = None
    for _ in range(4):
        reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.raw_sample is not None
    assert float(reading.quality["vertical_eyelid_value"]) > 0.5
    assert float(reading.quality["vertical_eyelid_noise_risk"]) > 0.65
    assert 0.45 <= reading.raw_sample.point.y <= 0.58
    assert 0.45 <= reading.sample.point.y <= 0.58


def test_mediapipe_capture_average_sample_discards_first_warmup_read_when_budget_allows() -> None:
    provider = MediaPipeWebcamGazeProvider()
    samples = iter(
        [
            SimpleNamespace(sample=GazeSample(point=NormalizedPoint(0.1, 0.1), confidence=0.9, delta_ms=120), raw_sample=None, signals=None, quality=None),
            SimpleNamespace(sample=GazeSample(point=NormalizedPoint(0.5, 0.5), confidence=0.9, delta_ms=120), raw_sample=None, signals=None, quality=None),
            SimpleNamespace(sample=GazeSample(point=NormalizedPoint(0.5, 0.5), confidence=0.9, delta_ms=120), raw_sample=None, signals=None, quality=None),
        ]
    )
    provider.read = lambda *, delta_ms=100: next(samples, None)  # type: ignore[method-assign]

    aggregate = provider.capture_average_sample(required_frames=2, max_attempts=3, delta_ms=120)

    assert aggregate is not None
    assert aggregate.used_frames == 2
    assert aggregate.sample.point.x == pytest.approx(0.5)
    assert aggregate.sample.point.y == pytest.approx(0.5)


def test_mediapipe_provider_uses_blendshapes_for_horizontal_direction() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(
        _landmarks_payload(),
        blendshapes=[
            SimpleNamespace(category_name="eyeLookInLeft", score=0.8),
            SimpleNamespace(category_name="eyeLookOutRight", score=0.7),
            SimpleNamespace(category_name="eyeLookOutLeft", score=0.0),
            SimpleNamespace(category_name="eyeLookInRight", score=0.0),
        ],
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.sample.point.x > 0.7


def test_mediapipe_provider_describes_richer_control_capabilities() -> None:
    capabilities = MediaPipeWebcamGazeProvider.describe_capabilities()

    assert capabilities["tracker"] == "mediapipe"
    assert "mouth_open" in capabilities["supported_signals"]
    assert "left_eye_closed" in capabilities["supported_signals"]
    assert "head_yaw" in capabilities["supported_signals"]
    assert capabilities["blendshape_tracking"] in {"tasks_api", "legacy_facemesh_iris_only", "unavailable"}


def test_mediapipe_provider_reports_legacy_blendshape_degradation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("interaction.vision.mediapipe_provider._supports_tasks_face_landmarker_api", lambda: False)
    monkeypatch.setattr("interaction.vision.mediapipe_provider._supports_legacy_face_mesh_api", lambda: True)

    capabilities = MediaPipeWebcamGazeProvider.describe_capabilities()

    assert capabilities["blendshape_tracking"] == "legacy_facemesh_iris_only"
    assert capabilities["horizontal_signal_quality"] == "degraded"
    assert "weaker horizontally" in capabilities["signal_notes"]


def test_mediapipe_provider_exposes_mouth_open_control_signal() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(
        _landmarks_payload(),
        blendshapes=[
            SimpleNamespace(category_name="jawOpen", score=0.82),
        ],
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.signals is not None
    assert reading.signals["mouth_open"] >= 0.82


def test_mediapipe_provider_exposes_mouth_open_control_signal_from_landmarks() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(_landmarks_payload())

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.signals is not None
    assert reading.signals["mouth_open"] > 0.3


def test_mediapipe_provider_exposes_unilateral_eye_closed_signals_from_landmarks() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(
        _landmarks_payload(
            overrides={
                386: (0.65, 0.396),
                374: (0.65, 0.404),
            }
        )
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.signals is not None
    assert reading.signals["left_eye_closed"] > 0.7
    assert reading.signals["right_eye_closed"] < 0.2


def test_mediapipe_provider_exposes_unilateral_eye_closed_signal_from_squint_blendshape() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(
        _landmarks_payload(),
        blendshapes=[
            SimpleNamespace(category_name="eyeSquintRight", score=0.92),
            SimpleNamespace(category_name="eyeSquintLeft", score=0.05),
        ],
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.signals is not None
    assert reading.signals["right_eye_closed"] > 0.7
    assert reading.signals["left_eye_closed"] < 0.2


def test_mediapipe_provider_exposes_head_pose_signals_from_landmarks() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(
        _landmarks_payload(
            overrides={
                1: (0.56, 0.58),
                263: (0.70, 0.44),
                33: (0.30, 0.36),
            }
        )
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.signals is not None
    assert reading.signals["head_yaw"] > 0.0
    assert reading.signals["head_pitch"] > 0.0
    assert reading.signals["head_roll"] > 0.0


def test_mediapipe_provider_axis_quality_handles_single_detected_eye() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(
        _landmarks_payload(
            overrides={
                362: (0.60, 0.40),
                263: (0.603, 0.40),
            }
        )
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.quality is not None
    assert bool(reading.quality["left_eye_present"]) is False
    assert bool(reading.quality["right_eye_present"]) is True
    assert float(reading.quality["right_eye_x_reliability"]) > 0.0
    assert float(reading.quality["x_axis_confidence"]) > 0.5


def test_mediapipe_provider_exposes_per_eye_axis_quality_metrics() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeFaceLandmarker(_landmarks_payload())

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reading = provider.process_frame(frame, delta_ms=120)

    assert reading is not None
    assert reading.quality is not None
    assert 0.0 <= float(reading.quality["left_eye_x_ratio"]) <= 1.0
    assert 0.0 <= float(reading.quality["right_eye_x_ratio"]) <= 1.0
    assert 0.0 <= float(reading.quality["x_axis_confidence"]) <= 1.0
    assert 0.0 <= float(reading.quality["y_axis_confidence"]) <= 1.0
    assert "horizontal_disagreement" in reading.quality
    assert "vertical_disagreement" in reading.quality
    assert "temporal_fusion_active" in reading.quality


def test_mediapipe_provider_temporal_fusion_damps_single_frame_outlier() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeSequenceFaceLandmarker(
        [
            _frame_payload(),
            _frame_payload(),
            _frame_payload(),
            _frame_payload(),
            _frame_payload(overrides=_shift_irises(horizontal="hard_left")),
        ]
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    baseline_readings = [provider.process_frame(frame, delta_ms=120) for _ in range(4)]
    outlier_reading = provider.process_frame(frame, delta_ms=120)

    assert all(reading is not None for reading in baseline_readings)
    assert outlier_reading is not None
    baseline_x = baseline_readings[-1].sample.point.x
    assert outlier_reading.raw_sample is not None
    assert abs(outlier_reading.sample.point.x - baseline_x) < abs(outlier_reading.raw_sample.point.x - baseline_x)
    assert outlier_reading.quality is not None
    assert bool(outlier_reading.quality["temporal_fusion_active"]) is True


def test_mediapipe_provider_freezes_temporal_fusion_during_wink_distortion() -> None:
    provider = MediaPipeWebcamGazeProvider()
    provider.face_mesh = FakeSequenceFaceLandmarker(
        [
            _frame_payload(),
            _frame_payload(),
            _frame_payload(),
            _frame_payload(
                overrides=_shift_irises(horizontal="hard_right"),
                blendshapes=[SimpleNamespace(category_name="eyeSquintRight", score=0.95)],
            ),
        ]
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    stable_readings = [provider.process_frame(frame, delta_ms=120) for _ in range(3)]
    wink_reading = provider.process_frame(frame, delta_ms=120)

    assert all(reading is not None for reading in stable_readings)
    assert wink_reading is not None
    assert wink_reading.raw_sample is not None
    assert wink_reading.quality is not None
    assert bool(wink_reading.quality["fusion_wink_freeze"]) is True
    assert abs(wink_reading.sample.point.x - stable_readings[-1].sample.point.x) < abs(
        wink_reading.raw_sample.point.x - stable_readings[-1].sample.point.x
    )


def test_apply_head_pose_compensation_offsets_bias_in_opposite_direction() -> None:
    assert _apply_head_pose_compensation(0.7, 0.5, weight=0.1) == pytest.approx(0.65)
    assert _apply_head_pose_compensation(0.3, -0.5, weight=0.1) == pytest.approx(0.35)


def test_apply_head_pose_compensation_scales_with_larger_pose_magnitude() -> None:
    small_pose = _apply_head_pose_compensation(0.7, 0.1, weight=0.1)
    large_pose = _apply_head_pose_compensation(0.7, 0.9, weight=0.1)

    assert large_pose < small_pose


def test_head_pose_signals_are_centered_for_neutral_face() -> None:
    signals = _head_pose_signals(_landmarks_payload(), 640, 480)

    assert abs(signals["head_yaw"]) < 0.1
    assert abs(signals["head_pitch"]) < 0.12


def test_mediapipe_provider_requires_model_path_for_tasks_api() -> None:
    provider = MediaPipeWebcamGazeProvider(model_asset_path=None)

    try:
        provider._resolved_model_asset_path()
    except WebcamProviderError as error:
        assert error.code == "mediapipe_model_required"
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected a WebcamProviderError when no model path is configured.")


def test_mediapipe_provider_preserves_tasks_video_timestamp_across_tracking_reset() -> None:
    provider = MediaPipeWebcamGazeProvider()
    tracker = StrictMonotonicFaceLandmarker(_landmarks_payload())
    provider.face_mesh = tracker

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    first = provider.process_frame(frame, delta_ms=120)
    provider.reset_tracking_state()
    second = provider.process_frame(frame, delta_ms=120)

    assert first is not None
    assert second is not None
    assert tracker.timestamps == [120, 240]


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


class FakeSequenceFaceLandmarker:
    def __init__(self, frames) -> None:
        self.frames = list(frames)
        self.index = 0

    def detect_for_video(self, _frame, _timestamp_ms):
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        landmarks = frame["landmarks"]
        blendshapes = frame["blendshapes"]
        return SimpleNamespace(face_landmarks=[landmarks], face_blendshapes=[blendshapes])


class StrictMonotonicFaceLandmarker:
    def __init__(self, landmarks) -> None:
        self.landmarks = landmarks
        self.last_timestamp = -1
        self.timestamps: list[int] = []

    def detect_for_video(self, _frame, timestamp_ms):
        if timestamp_ms <= self.last_timestamp:
            raise ValueError("Input timestamp must be monotonically increasing.")
        self.last_timestamp = timestamp_ms
        self.timestamps.append(timestamp_ms)
        return SimpleNamespace(face_landmarks=[self.landmarks], face_blendshapes=[[]])


def _landmarks_payload(overrides: dict[int, tuple[float, float]] | None = None):
    return _frame_payload(overrides=overrides)["landmarks"]


def _frame_payload(
    overrides: dict[int, tuple[float, float]] | None = None,
    *,
    blendshapes: list[SimpleNamespace] | None = None,
):
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
        13: (0.50, 0.46),
        14: (0.50, 0.54),
        78: (0.42, 0.50),
        308: (0.58, 0.50),
    }
    if overrides:
        points.update(overrides)
    for index, (x, y) in points.items():
        landmarks[index] = SimpleNamespace(x=x, y=y)
    return {"landmarks": landmarks, "blendshapes": blendshapes or []}


def _shift_irises(*, horizontal: str) -> dict[int, tuple[float, float]]:
    if horizontal == "hard_left":
        return {
            468: (0.305, 0.40),
            469: (0.305, 0.395),
            470: (0.31, 0.40),
            471: (0.305, 0.405),
            472: (0.30, 0.40),
            473: (0.605, 0.40),
            474: (0.605, 0.395),
            475: (0.61, 0.40),
            476: (0.605, 0.405),
            477: (0.60, 0.40),
        }
    if horizontal == "hard_right":
        return {
            468: (0.395, 0.40),
            469: (0.395, 0.395),
            470: (0.39, 0.40),
            471: (0.395, 0.405),
            472: (0.40, 0.40),
            473: (0.695, 0.40),
            474: (0.695, 0.395),
            475: (0.69, 0.40),
            476: (0.695, 0.405),
            477: (0.70, 0.40),
        }
    raise AssertionError(f"Unsupported horizontal iris shift: {horizontal}")


def _shift_irises_vertical(*, vertical: str) -> dict[int, tuple[float, float]]:
    if vertical == "up":
        return {
            468: (0.34, 0.39),
            469: (0.34, 0.385),
            470: (0.345, 0.39),
            471: (0.34, 0.395),
            472: (0.335, 0.39),
            473: (0.66, 0.39),
            474: (0.66, 0.385),
            475: (0.665, 0.39),
            476: (0.66, 0.395),
            477: (0.655, 0.39),
        }
    if vertical == "down":
        return {
            468: (0.34, 0.41),
            469: (0.34, 0.405),
            470: (0.345, 0.41),
            471: (0.34, 0.415),
            472: (0.335, 0.41),
            473: (0.66, 0.41),
            474: (0.66, 0.405),
            475: (0.665, 0.41),
            476: (0.66, 0.415),
            477: (0.655, 0.41),
        }
    raise AssertionError(f"Unsupported vertical iris shift: {vertical}")


def _symmetric_lid_overrides(*, top: float, bottom: float) -> dict[int, tuple[float, float]]:
    return {
        159: (0.35, top),
        145: (0.35, bottom),
        386: (0.65, top),
        374: (0.65, bottom),
    }


def _vertical_blendshapes(*, up: float, down: float) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(category_name="eyeLookUpLeft", score=up),
        SimpleNamespace(category_name="eyeLookUpRight", score=up),
        SimpleNamespace(category_name="eyeLookDownLeft", score=down),
        SimpleNamespace(category_name="eyeLookDownRight", score=down),
    ]
