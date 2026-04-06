import argparse
import io
import json
import math
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from interaction.app import (
    _append_calibration_capture_feedback,
    _append_calibration_prompt,
    _build_parser,
    _run_gaze_benchmark_report,
    main,
)
from interaction.contracts import ActionName, GazeObservation
from interaction.feedback import GazeLoopPhase
from interaction.platform import MacOSPlatformAdapter
from interaction.persistence import JsonStateStore, RuntimePaths, UserSettings
from interaction.runtime import GazeTrackingLoop
from interaction.runtime.webcam import (
    CursorFollowController,
    RelativeCursorController,
    WebcamCalibrationTarget,
    _compress_point_around_anchor,
    default_webcam_snap_targets,
    default_webcam_targets,
    generate_single_screen_calibration_targets,
    generate_single_screen_validation_targets,
    run_live_assisted_cursor_follow,
    run_live_webcam_benchmark,
    run_live_cursor_follow,
    run_live_cursor_snap,
    run_live_relative_cursor_follow,
    run_live_wink_click_test,
    run_live_webcam_trace,
)
from interaction.ui import GazeDotGeometry
from interaction.vision import CalibrationProfile, GazeSample, NormalizedPoint, WebcamProviderError, WebcamSampleAggregate


class FakeCalibrationProvider:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.opened = False
        self.closed = False
        self._samples = [
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.18, 0.19), confidence=0.8, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.50, 0.18), confidence=0.83, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.78, 0.18), confidence=0.82, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.21, 0.50), confidence=0.85, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.5, 0.5), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.79, 0.49), confidence=0.86, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.22, 0.79), confidence=0.81, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.5, 0.81), confidence=0.82, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.79, 0.82), confidence=0.8, delta_ms=100), used_frames=3, attempted_frames=3),
        ]

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def capture_average_sample(self, *, required_frames: int = 6, max_attempts: int | None = None, delta_ms: int = 100):
        return self._samples.pop(0)


class FakeSparseCalibrationProvider(FakeCalibrationProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._samples = [
            None,
            None,
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.5, 0.5), confidence=0.9, delta_ms=100), used_frames=1, attempted_frames=3),
            None,
            None,
            None,
            None,
            None,
            None,
        ]


class FakeCompressedCalibrationProvider(FakeCalibrationProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._samples = [
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.47, 0.49), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.48, 0.52), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.49, 0.51), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.50, 0.50), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.51, 0.50), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.54, 0.51), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.53, 0.50), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.52, 0.49), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.54, 0.50), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
        ]


class FakeQualityCalibrationProvider(FakeCalibrationProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._samples = [
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.18, 0.19), confidence=0.8, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={
                    "x_axis_confidence": 0.72,
                    "y_axis_confidence": 0.64,
                    "x_saturated_ratio": 0.0,
                    "y_saturated_ratio": 0.0,
                },
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.50, 0.18), confidence=0.83, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={
                    "x_axis_confidence": 0.73,
                    "y_axis_confidence": 0.63,
                    "x_saturated_ratio": 0.0,
                    "y_saturated_ratio": 0.0,
                },
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.78, 0.18), confidence=0.82, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={
                    "x_axis_confidence": 0.70,
                    "y_axis_confidence": 0.60,
                    "x_saturated_ratio": 0.0,
                    "y_saturated_ratio": 0.0,
                },
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.21, 0.50), confidence=0.85, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={
                    "x_axis_confidence": 0.71,
                    "y_axis_confidence": 0.65,
                    "x_saturated_ratio": 0.0,
                    "y_saturated_ratio": 0.0,
                },
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.5, 0.5), confidence=0.9, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={
                    "x_axis_confidence": 0.74,
                    "y_axis_confidence": 0.68,
                    "x_saturated_ratio": 0.0,
                    "y_saturated_ratio": 0.0,
                },
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.79, 0.49), confidence=0.86, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={
                    "x_axis_confidence": 0.72,
                    "y_axis_confidence": 0.66,
                    "x_saturated_ratio": 0.0,
                    "y_saturated_ratio": 0.0,
                },
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.22, 0.79), confidence=0.81, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={
                    "x_axis_confidence": 0.68,
                    "y_axis_confidence": 0.66,
                    "x_saturated_ratio": 0.0,
                    "y_saturated_ratio": 0.0,
                },
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.5, 0.81), confidence=0.82, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={
                    "x_axis_confidence": 0.69,
                    "y_axis_confidence": 0.67,
                    "x_saturated_ratio": 0.0,
                    "y_saturated_ratio": 0.0,
                },
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.79, 0.82), confidence=0.8, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={
                    "x_axis_confidence": 0.71,
                    "y_axis_confidence": 0.67,
                    "x_saturated_ratio": 0.0,
                    "y_saturated_ratio": 0.0,
                },
            ),
        ]


class FakeLowAxisConfidenceCalibrationProvider(FakeCalibrationProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._samples = [
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.18, 0.19), confidence=0.8, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={"x_axis_confidence": 0.74, "y_axis_confidence": 0.34, "x_saturated": 0.0, "y_saturated": 0.0},
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.50, 0.18), confidence=0.83, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={"x_axis_confidence": 0.75, "y_axis_confidence": 0.35, "x_saturated": 0.0, "y_saturated": 0.0},
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.78, 0.18), confidence=0.82, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={"x_axis_confidence": 0.72, "y_axis_confidence": 0.36, "x_saturated": 0.0, "y_saturated": 0.0},
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.21, 0.50), confidence=0.85, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={"x_axis_confidence": 0.74, "y_axis_confidence": 0.35, "x_saturated": 0.0, "y_saturated": 0.0},
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.5, 0.5), confidence=0.9, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={"x_axis_confidence": 0.76, "y_axis_confidence": 0.36, "x_saturated": 0.0, "y_saturated": 0.0},
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.79, 0.49), confidence=0.86, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={"x_axis_confidence": 0.73, "y_axis_confidence": 0.37, "x_saturated": 0.0, "y_saturated": 0.0},
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.22, 0.79), confidence=0.81, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={"x_axis_confidence": 0.73, "y_axis_confidence": 0.35, "x_saturated": 0.0, "y_saturated": 0.0},
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.5, 0.81), confidence=0.82, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={"x_axis_confidence": 0.72, "y_axis_confidence": 0.36, "x_saturated": 0.0, "y_saturated": 0.0},
            ),
            WebcamSampleAggregate(
                sample=GazeSample(point=NormalizedPoint(0.79, 0.82), confidence=0.8, delta_ms=100),
                used_frames=3,
                attempted_frames=3,
                quality={"x_axis_confidence": 0.71, "y_axis_confidence": 0.37, "x_saturated": 0.0, "y_saturated": 0.0},
            ),
        ]


class FakeLiveProvider:
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {"tracker": "fake", "supported_signals": []}

    def __init__(self, camera_index: int = 0, model_asset_path: str | None = None) -> None:
        self.camera_index = camera_index
        self.model_asset_path = model_asset_path
        self.opened = False
        self.closed = False
        self._points = [
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
        ]

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def read(self, *, delta_ms: int = 100):
        if not self._points:
            return None
        point = self._points.pop(0)
        signals = {}
        if hasattr(self, "_signals") and self._signals:
            signals = dict(self._signals.pop(0))
        return type(
            "Reading",
            (),
            {
                "sample": GazeSample(point=point, confidence=0.9, delta_ms=delta_ms),
                "frame_size": (640, 480),
                "face_bbox": (0, 0, 10, 10),
                "eye_boxes": (),
                "signals": signals,
            },
        )()


class FakeBenchmarkProvider(FakeLiveProvider):
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {
            "tracker": "fake-benchmark",
            "supported_signals": ["left_eye_closed", "right_eye_closed", "head_yaw", "head_pitch", "head_roll"],
        }

    def __init__(self, camera_index: int = 0, model_asset_path: str | None = None) -> None:
        super().__init__(camera_index, model_asset_path)
        self._frames = [
            {
                "sample": NormalizedPoint(0.50, 0.50),
                "raw_sample": NormalizedPoint(0.48, 0.50),
                "signals": {"head_yaw": 0.02, "head_pitch": 0.01, "head_roll": 0.0, "left_eye_closed": 0.0, "right_eye_closed": 0.0},
                "quality": {"x_axis_confidence": 0.80, "y_axis_confidence": 0.70, "temporal_fusion_active": True, "fusion_samples": 3},
            },
            {
                "sample": NormalizedPoint(0.505, 0.502),
                "raw_sample": NormalizedPoint(0.56, 0.53),
                "signals": {"head_yaw": 0.04, "head_pitch": 0.02, "head_roll": 0.01, "left_eye_closed": 0.0, "right_eye_closed": 0.0},
                "quality": {"x_axis_confidence": 0.82, "y_axis_confidence": 0.72, "temporal_fusion_active": True, "fusion_samples": 4},
            },
            {
                "sample": NormalizedPoint(0.507, 0.504),
                "raw_sample": NormalizedPoint(0.49, 0.51),
                "signals": {"head_yaw": 0.03, "head_pitch": 0.02, "head_roll": 0.01, "left_eye_closed": 0.92, "right_eye_closed": 0.10},
                "quality": {"x_axis_confidence": 0.81, "y_axis_confidence": 0.74, "temporal_fusion_active": True, "fusion_samples": 5},
            },
            {
                "sample": NormalizedPoint(0.509, 0.505),
                "raw_sample": NormalizedPoint(0.52, 0.50),
                "signals": {"head_yaw": 0.01, "head_pitch": 0.01, "head_roll": 0.0, "left_eye_closed": 0.0, "right_eye_closed": 0.0},
                "quality": {"x_axis_confidence": 0.83, "y_axis_confidence": 0.73, "temporal_fusion_active": True, "fusion_samples": 5},
            },
        ]

    def read(self, *, delta_ms: int = 100):
        if not self._frames:
            return None
        frame = self._frames.pop(0)
        return type(
            "Reading",
            (),
            {
                "sample": GazeSample(point=frame["sample"], confidence=0.91, delta_ms=delta_ms),
                "raw_sample": GazeSample(point=frame["raw_sample"], confidence=0.91, delta_ms=delta_ms),
                "frame_size": (640, 480),
                "face_bbox": (0, 0, 10, 10),
                "eye_boxes": (),
                "signals": frame["signals"],
                "quality": frame["quality"],
            },
        )()


class FakeSnapWinkLeftProvider(FakeLiveProvider):
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {"tracker": "fake-snap-wink", "supported_signals": ["left_eye_closed", "right_eye_closed"]}

    def __init__(self, camera_index: int = 0, model_asset_path: str | None = None) -> None:
        super().__init__(camera_index, model_asset_path=model_asset_path)
        self._points = [
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
        ]
        self._signals = [
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.95, "right_eye_closed": 0.1},
            {"left_eye_closed": 0.95, "right_eye_closed": 0.1},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
        ]


class FakeSnapWinkRightProvider(FakeLiveProvider):
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {"tracker": "fake-snap-wink", "supported_signals": ["left_eye_closed", "right_eye_closed"]}

    def __init__(self, camera_index: int = 0, model_asset_path: str | None = None) -> None:
        super().__init__(camera_index, model_asset_path=model_asset_path)
        self._points = [
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
        ]
        self._signals = [
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.1, "right_eye_closed": 0.95},
            {"left_eye_closed": 0.1, "right_eye_closed": 0.95},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
        ]


class FakeSnapWinkRightLeakyProvider(FakeLiveProvider):
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {"tracker": "fake-snap-wink", "supported_signals": ["left_eye_closed", "right_eye_closed"]}

    def __init__(self, camera_index: int = 0, model_asset_path: str | None = None) -> None:
        super().__init__(camera_index, model_asset_path=model_asset_path)
        self._points = [
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
        ]
        self._signals = [
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.42, "right_eye_closed": 0.92},
            {"left_eye_closed": 0.44, "right_eye_closed": 0.90},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
        ]


class FakeSnapBlinkProvider(FakeLiveProvider):
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {"tracker": "fake-snap-wink", "supported_signals": ["left_eye_closed", "right_eye_closed"]}

    def __init__(self, camera_index: int = 0, model_asset_path: str | None = None) -> None:
        super().__init__(camera_index, model_asset_path=model_asset_path)
        self._points = [
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
        ]
        self._signals = [
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.94, "right_eye_closed": 0.88},
            {"left_eye_closed": 0.95, "right_eye_closed": 0.90},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
        ]


class FakeSnapWinkRightDriftProvider(FakeLiveProvider):
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {"tracker": "fake-snap-wink", "supported_signals": ["left_eye_closed", "right_eye_closed"]}

    def __init__(self, camera_index: int = 0, model_asset_path: str | None = None) -> None:
        super().__init__(camera_index, model_asset_path=model_asset_path)
        self._points = [
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.88, 0.82),
            NormalizedPoint(0.86, 0.80),
            NormalizedPoint(0.84, 0.78),
        ]
        self._signals = [
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.42, "right_eye_closed": 0.92},
            {"left_eye_closed": 0.44, "right_eye_closed": 0.90},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
        ]


class FakeSnapWinkCancelProvider(FakeLiveProvider):
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {"tracker": "fake-snap-wink", "supported_signals": ["left_eye_closed", "right_eye_closed"]}

    def __init__(self, camera_index: int = 0, model_asset_path: str | None = None) -> None:
        super().__init__(camera_index, model_asset_path=model_asset_path)
        self._points = [
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.15, 0.15),
            NormalizedPoint(0.88, 0.82),
            NormalizedPoint(0.88, 0.82),
            NormalizedPoint(0.88, 0.82),
            NormalizedPoint(0.88, 0.82),
            NormalizedPoint(0.88, 0.82),
            NormalizedPoint(0.88, 0.82),
            NormalizedPoint(0.88, 0.82),
        ]
        self._signals = [
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
            {"left_eye_closed": 0.0, "right_eye_closed": 0.0},
        ]


class FakeErrorProvider:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index

    def open(self) -> None:
        raise WebcamProviderError("camera_unavailable", "Unable to open webcam at index 0", payload={"camera_index": self.camera_index})

    def close(self) -> None:
        return None


class FakePointVisualizer:
    def __init__(self) -> None:
        self.points: list[tuple[float, float]] = []
        self.hidden_calls = 0
        self.wait_calls = 0
        self.styles: list[dict[str, str | None]] = []

    def show_point(self, x_norm: float, y_norm: float) -> None:
        self.points.append((x_norm, y_norm))

    def hide(self) -> None:
        self.hidden_calls += 1

    def set_style(self, *, fill: str | None = None, outline: str | None = None) -> None:
        self.styles.append({"fill": fill, "outline": outline})

    def wait_for_confirmation(self) -> None:
        self.wait_calls += 1


class FakeInteractiveValidationVisualizer(FakePointVisualizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.validation_points: list[tuple[float, float, float | None, float | None, bool, str | None]] = []
        self.click_points: list[tuple[float, float]] = [(0.72, 0.28), (0.28, 0.72), (0.5, 0.5)]

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def show_validation_point(
        self,
        target_x: float,
        target_y: float,
        *,
        tracked_x: float | None = None,
        tracked_y: float | None = None,
        armed: bool = False,
        wink_side: str | None = None,
        step_index: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        self.validation_points.append((target_x, target_y, tracked_x, tracked_y, armed, wink_side))

    def wait_for_confirmation(self, *, on_iteration=None) -> None:
        self.wait_calls += 1
        if on_iteration is not None:
            for _ in range(3):
                on_iteration()

    def wait_for_click_point(self, *, on_iteration=None) -> tuple[float, float]:
        self.wait_calls += 1
        if on_iteration is not None:
            for _ in range(3):
                on_iteration()
        if self.click_points:
            return self.click_points.pop(0)
        return (0.5, 0.5)


class FakeCameraPreview:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.close_calls = 0

    def show_reading(
        self,
        reading,
        *,
        calibrated_point=None,
        cursor_point=None,
        target_label=None,
    ) -> None:
        self.calls.append(
            {
                "reading": reading,
                "calibrated_point": calibrated_point,
                "cursor_point": cursor_point,
                "target_label": target_label,
            }
        )

    def close(self) -> None:
        self.close_calls += 1


class FakeInteractiveValidationProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.4, 0.6),
            NormalizedPoint(0.4, 0.6),
            NormalizedPoint(0.4, 0.6),
        ]


class FakeInteractiveRefinementProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.25, 0.35),
            NormalizedPoint(0.65, 0.50),
            NormalizedPoint(0.45, 0.72),
        ]


class FakeInteractiveRefinementYSaturatedProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.20, 0.95),
            NormalizedPoint(0.80, 0.95),
            NormalizedPoint(0.50, 0.95),
        ]


class FakeRelativeLiveProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.80, 0.45),
            NormalizedPoint(0.80, 0.45),
            NormalizedPoint(0.80, 0.45),
        ]


class FakeRelativeSubtleProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.53, 0.50),
            NormalizedPoint(0.535, 0.50),
            NormalizedPoint(0.54, 0.50),
        ]


class FakeRelativeDriftProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.508, 0.50),
            NormalizedPoint(0.516, 0.50),
            NormalizedPoint(0.524, 0.50),
        ]


class FakeRelativePostureShiftProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
        ]
        self._signals = [
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.82, "head_pitch": 0.18, "head_roll": 0.0},
            {"head_yaw": 0.86, "head_pitch": 0.22, "head_roll": 0.0},
            {"head_yaw": 0.84, "head_pitch": 0.20, "head_roll": 0.0},
            {"head_yaw": 0.03, "head_pitch": 0.02, "head_roll": 0.0},
            {"head_yaw": 0.02, "head_pitch": 0.01, "head_roll": 0.0},
            {"head_yaw": 0.02, "head_pitch": 0.01, "head_roll": 0.0},
        ]


class FakeRelativePostureRecoverProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.56, 0.50),
            NormalizedPoint(0.58, 0.50),
            NormalizedPoint(0.60, 0.50),
            NormalizedPoint(0.62, 0.50),
            NormalizedPoint(0.64, 0.50),
            NormalizedPoint(0.66, 0.50),
        ]
        self._signals = [
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.88, "head_pitch": 0.20, "head_roll": 0.0},
            {"head_yaw": 0.90, "head_pitch": 0.22, "head_roll": 0.0},
            {"head_yaw": 0.86, "head_pitch": 0.18, "head_roll": 0.0},
            {"head_yaw": 0.03, "head_pitch": 0.01, "head_roll": 0.0},
            {"head_yaw": 0.02, "head_pitch": 0.01, "head_roll": 0.0},
            {"head_yaw": 0.02, "head_pitch": 0.01, "head_roll": 0.0},
            {"head_yaw": 0.02, "head_pitch": 0.01, "head_roll": 0.0},
            {"head_yaw": 0.02, "head_pitch": 0.01, "head_roll": 0.0},
            {"head_yaw": 0.02, "head_pitch": 0.01, "head_roll": 0.0},
        ]


class FakeRelativeEyeOnlyProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.54, 0.50),
            NormalizedPoint(0.58, 0.50),
            NormalizedPoint(0.62, 0.50),
            NormalizedPoint(0.66, 0.50),
            NormalizedPoint(0.70, 0.50),
        ]
        self._signals = [
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.01, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.01, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
        ]


class FakeRelativeNeutralProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
        ]
        self._signals = [
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
        ]


class FakeRelativeBiasLearningProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.507, 0.50),
            NormalizedPoint(0.507, 0.50),
            NormalizedPoint(0.507, 0.50),
            NormalizedPoint(0.507, 0.50),
        ]
        self._signals = [
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
        ]


class FakeRelativeHeadOnlyProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
        ]
        self._signals = [
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.16, "head_pitch": 0.08, "head_roll": 0.0},
            {"head_yaw": 0.18, "head_pitch": 0.09, "head_roll": 0.0},
            {"head_yaw": 0.15, "head_pitch": 0.08, "head_roll": 0.0},
            {"head_yaw": 0.14, "head_pitch": 0.07, "head_roll": 0.0},
            {"head_yaw": 0.12, "head_pitch": 0.06, "head_roll": 0.0},
        ]


class FakeRelativeSmallHeadAssistProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.62, 0.49),
            NormalizedPoint(0.63, 0.49),
            NormalizedPoint(0.64, 0.49),
            NormalizedPoint(0.65, 0.49),
            NormalizedPoint(0.66, 0.49),
            NormalizedPoint(0.67, 0.49),
        ]
        self._signals = [
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.14, "head_pitch": 0.08, "head_roll": 0.0},
            {"head_yaw": 0.16, "head_pitch": 0.09, "head_roll": 0.0},
            {"head_yaw": 0.15, "head_pitch": 0.08, "head_roll": 0.0},
            {"head_yaw": 0.14, "head_pitch": 0.07, "head_roll": 0.0},
            {"head_yaw": 0.12, "head_pitch": 0.06, "head_roll": 0.0},
        ]


class FakeRelativeLargeHeadAssistProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.53, 0.49),
            NormalizedPoint(0.54, 0.49),
            NormalizedPoint(0.55, 0.49),
            NormalizedPoint(0.56, 0.49),
            NormalizedPoint(0.57, 0.49),
            NormalizedPoint(0.58, 0.49),
        ]
        self._signals = [
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"head_yaw": 0.82, "head_pitch": 0.18, "head_roll": 0.0},
            {"head_yaw": 0.86, "head_pitch": 0.22, "head_roll": 0.0},
            {"head_yaw": 0.84, "head_pitch": 0.20, "head_roll": 0.0},
            {"head_yaw": 0.03, "head_pitch": 0.02, "head_roll": 0.0},
            {"head_yaw": 0.02, "head_pitch": 0.01, "head_roll": 0.0},
            {"head_yaw": 0.02, "head_pitch": 0.01, "head_roll": 0.0},
        ]


class FakeRelativeClickProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
        ]


class FakeAssistedLiveProvider(FakeLiveProvider):
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {
            "tracker": "fake-assist",
            "supported_signals": ["mouth_open", "head_yaw", "head_pitch", "head_roll"],
        }

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
        ]
        self._signals = [
            {},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {},
        ]


class FakeRecalibrateThenAssistProvider(FakeAssistedLiveProvider):
    instances: list["FakeRecalibrateThenAssistProvider"] = []

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        type(self).instances.append(self)
        self.open_count = 0
        self.close_count = 0
        self._samples = [
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.18, 0.18), confidence=0.9, delta_ms=100), used_frames=4, attempted_frames=4),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.50, 0.18), confidence=0.9, delta_ms=100), used_frames=4, attempted_frames=4),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.82, 0.18), confidence=0.9, delta_ms=100), used_frames=4, attempted_frames=4),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.18, 0.50), confidence=0.9, delta_ms=100), used_frames=4, attempted_frames=4),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.50, 0.50), confidence=0.9, delta_ms=100), used_frames=4, attempted_frames=4),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.82, 0.50), confidence=0.9, delta_ms=100), used_frames=4, attempted_frames=4),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.18, 0.82), confidence=0.9, delta_ms=100), used_frames=4, attempted_frames=4),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.50, 0.82), confidence=0.9, delta_ms=100), used_frames=4, attempted_frames=4),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.82, 0.82), confidence=0.9, delta_ms=100), used_frames=4, attempted_frames=4),
        ]

    def open(self) -> None:
        if self.opened:
            return
        self.opened = True
        self.open_count += 1

    def close(self) -> None:
        if not self.opened:
            return
        self.opened = False
        self.closed = True
        self.close_count += 1

    def capture_average_sample(self, *, required_frames: int = 6, max_attempts: int | None = None, delta_ms: int = 100):
        return self._samples.pop(0)


class FakeAssistedClickProvider(FakeLiveProvider):
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {
            "tracker": "fake-assist",
            "supported_signals": ["mouth_open", "head_yaw", "head_pitch", "head_roll"],
        }

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
        ]
        self._signals = [
            {},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
        ]


class FakeAssistedBiasLearningProvider(FakeLiveProvider):
    @staticmethod
    def describe_capabilities() -> dict[str, object]:
        return {
            "tracker": "fake-assist",
            "supported_signals": ["mouth_open", "head_yaw", "head_pitch", "head_roll"],
        }

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.507, 0.50),
            NormalizedPoint(0.507, 0.50),
            NormalizedPoint(0.507, 0.50),
            NormalizedPoint(0.507, 0.50),
            NormalizedPoint(0.507, 0.50),
        ]
        self._signals = [
            {},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
        ]


class FakeAssistedHeadOnlyProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
        ]
        self._signals = [
            {},
            {"mouth_open": 0.85, "head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"mouth_open": 0.85, "head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            {"mouth_open": 0.85, "head_yaw": 0.18, "head_pitch": 0.08, "head_roll": 0.0},
            {"mouth_open": 0.85, "head_yaw": 0.18, "head_pitch": 0.08, "head_roll": 0.0},
            {"mouth_open": 0.85, "head_yaw": 0.16, "head_pitch": 0.07, "head_roll": 0.0},
            {},
        ]


class FakeAssistedSaccadeProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.20, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.20, 0.50),
        ]
        self._signals = [
            {},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
        ]


class FakeAssistedIdleDisarmProvider(FakeLiveProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._points = [
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.80, 0.50),
            NormalizedPoint(0.50, 0.50),
            NormalizedPoint(0.50, 0.50),
        ]
        self._signals = [
            {},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
            {"mouth_open": 0.85},
        ]


def test_gaze_calibrate_cli_saves_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeCalibrationProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-calibrate", "--runtime-dir", str(tmp_path / ".interaction"), "--settle-ms", "0", "--frames-per-step", "3"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["calibration"]["status"] == "success"
    assert len(payload["calibration_targets"]) == 9
    assert payload["calibration_visualization"]["status"] == "disabled"
    assert (tmp_path / ".interaction" / "profiles" / "webcam-live.json").exists()


def test_gaze_calibrate_cli_reports_insufficient_samples(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeSparseCalibrationProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-calibrate", "--runtime-dir", str(tmp_path / ".interaction"), "--settle-ms", "0", "--frames-per-step", "3"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["calibration"]["status"] == "error"
    assert payload["calibration"]["error"] == "insufficient_calibration_samples"


def test_gaze_calibrate_cli_rejects_compressed_signal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeCompressedCalibrationProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-calibrate", "--runtime-dir", str(tmp_path / ".interaction"), "--settle-ms", "0", "--frames-per-step", "3"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["calibration"]["status"] == "error"
    assert payload["calibration"]["error"] == "unstable_calibration_signal"
    assert payload["calibration_quality"]["raw_x_span"] < 0.08
    assert payload["calibration_quality"]["raw_y_span"] < 0.05


def test_gaze_calibrate_cli_includes_axis_quality_summary_when_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeQualityCalibrationProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-calibrate", "--runtime-dir", str(tmp_path / ".interaction"), "--settle-ms", "0", "--frames-per-step", "3"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["calibration"]["status"] == "success"
    assert payload["calibration_quality"]["x_axis_confidence_mean"] == pytest.approx((0.72 + 0.73 + 0.70 + 0.71 + 0.74 + 0.72 + 0.68 + 0.69 + 0.71) / 9.0)
    assert payload["calibration_quality"]["y_axis_confidence_mean"] == pytest.approx((0.64 + 0.63 + 0.60 + 0.65 + 0.68 + 0.66 + 0.66 + 0.67 + 0.67) / 9.0)
    assert payload["calibration_quality"]["blocked_axes"] == []
    assert payload["calibration_quality"]["axis_diagnostics"]["x"]["status"] == "good"


def test_gaze_calibrate_cli_rejects_low_axis_confidence_even_when_spans_are_large(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLowAxisConfidenceCalibrationProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-calibrate", "--runtime-dir", str(tmp_path / ".interaction"), "--settle-ms", "0", "--frames-per-step", "3"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["calibration"]["status"] == "success"
    assert payload["calibration"]["degraded"] is True
    assert payload["calibration"]["fallback_axes"] == ["y"]
    assert payload["calibration_quality"]["raw_x_span"] > 0.08
    assert payload["calibration_quality"]["raw_y_span"] > 0.05
    assert payload["calibration_quality"]["blocked_axes"] == ["y"]
    assert payload["calibration_quality"]["axis_diagnostics"]["y"]["status"] == "blocked"
    assert "low_axis_confidence" in payload["calibration_quality"]["axis_diagnostics"]["y"]["issues"]
    assert payload["calibration_quality"]["fallback_profile_used"] is True
    assert payload["calibration_profile"]["y_scale"] == pytest.approx(1.0)
    assert payload["calibration_profile"]["y_offset"] == pytest.approx(0.0)


def test_generate_single_screen_calibration_targets_covers_full_display() -> None:
    targets = generate_single_screen_calibration_targets(seed=7)

    assert len(targets) == 9
    xs = [target.expected.x for target in targets]
    ys = [target.expected.y for target in targets]
    labels = {target.label for target in targets}

    assert labels == {
        "upper left",
        "upper center",
        "upper right",
        "middle left",
        "center",
        "middle right",
        "lower left",
        "lower center",
        "lower right",
    }
    assert min(xs) < 0.3
    assert max(xs) > 0.7
    assert min(ys) < 0.3
    assert max(ys) > 0.7


def test_generate_single_screen_calibration_targets_supports_five_rows() -> None:
    targets = generate_single_screen_calibration_targets(seed=7, rows=5)

    assert len(targets) == 15
    labels = {target.label for target in targets}
    ys = sorted(target.expected.y for target in targets)

    assert "upper middle" in labels
    assert "lower middle" in labels
    assert "center left" in labels
    assert len({round(y, 1) for y in ys}) >= 5


def test_append_calibration_prompt_waits_on_visualizer_confirmation() -> None:
    events: list[object] = []
    visualizer = FakePointVisualizer()
    step = WebcamCalibrationTarget(label="upper left", expected=NormalizedPoint(0.2, 0.2))

    _append_calibration_prompt(
        events,
        step,
        step_index=1,
        total_steps=4,
        frames_per_step=3,
        settle_ms=0,
        interactive=True,
        point_visualizer=visualizer,
    )

    assert len(events) == 1
    assert visualizer.points == [(0.2, 0.2)]
    assert visualizer.wait_calls == 1


def test_append_calibration_capture_feedback_reports_quality_hint() -> None:
    events: list[object] = []

    _append_calibration_capture_feedback(
        events,
        {
            "label": "upper center",
            "success": True,
            "confidence": 0.98,
            "quality": {
                "x_axis_confidence": 0.82,
                "y_axis_confidence": 0.44,
            },
        },
        step_index=2,
        total_steps=9,
    )

    assert len(events) == 1
    assert "Captured calibration point 2/9" in events[0].message
    assert "Y signal looked weak." in events[0].message


def test_generate_single_screen_validation_targets_spreads_points() -> None:
    targets = generate_single_screen_validation_targets(count=6, seed=7)

    assert len(targets) == 6
    xs = [target.expected.x for target in targets]
    ys = [target.expected.y for target in targets]

    assert min(xs) < 0.3
    assert max(xs) > 0.7
    assert min(ys) < 0.3
    assert max(ys) > 0.7


def test_gaze_live_requires_saved_calibration_profile(tmp_path: Path) -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8", "--delta-ms", "100"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze"]["error"] == "missing_calibration_profile"


def test_gaze_validate_requires_saved_calibration_profile(tmp_path: Path) -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-validate", "--runtime-dir", str(tmp_path / ".interaction"), "--points", "3"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["validation"]["error"] == "missing_calibration_profile"


def test_gaze_validate_reports_gap_metrics(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeCalibrationProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-validate",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--points",
                "3",
                "--frames-per-step",
                "3",
                "--random-seed",
                "7",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["validation"]["status"] == "success"
    assert payload["validation"]["successful_points"] == 3
    assert len(payload["validation_points"]) == 3
    assert all(point["success"] for point in payload["validation_points"])
    assert payload["validation"]["mean_error_distance"] >= 0.0


def test_gaze_validate_rejects_stale_profile_after_failed_calibration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeCompressedCalibrationProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        calibrate_exit = main(["gaze-calibrate", "--runtime-dir", str(tmp_path / ".interaction"), "--settle-ms", "0", "--frames-per-step", "3"])
    calibration_payload = json.loads(stdout.getvalue())

    assert calibrate_exit == 0
    assert calibration_payload["calibration"]["error"] == "unstable_calibration_signal"

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        validate_exit = main(["gaze-validate", "--runtime-dir", str(tmp_path / ".interaction"), "--points", "3"])
    validation_payload = json.loads(stdout.getvalue())

    assert validate_exit == 0
    assert validation_payload["validation"]["error"] == "invalid_calibration_profile"


def test_gaze_validate_interactive_captures_live_tracked_point_on_confirmation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeInteractiveValidationProvider)
    monkeypatch.setattr("interaction.app._is_interactive_calibration_session", lambda: True)
    monkeypatch.setattr("interaction.app.CalibrationDotOverlay", FakeInteractiveValidationVisualizer)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-validate",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--points",
                "1",
                "--random-seed",
                "7",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["validation"]["status"] == "success"
    assert payload["validation_points"][0]["captured_on_confirmation"] is True
    assert payload["validation_points"][0]["tracked"]["x"] == pytest.approx(0.4)
    assert payload["validation_points"][0]["tracked"]["y"] == pytest.approx(0.6)
    assert payload["validation_points"][0]["used_frames"] >= 1


def test_gaze_refine_interactive_updates_saved_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeInteractiveRefinementProvider)
    monkeypatch.setattr("interaction.app._is_interactive_calibration_session", lambda: True)
    monkeypatch.setattr("interaction.app.CalibrationDotOverlay", FakeInteractiveValidationVisualizer)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-refine",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--points",
                "3",
                "--random-seed",
                "7",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["refinement"]["status"] == "success"
    assert payload["refinement_points"][0]["captured_on_click"] is True
    assert payload["refinement_points"][0]["clicked"]["x"] == pytest.approx(0.72)
    assert payload["refined_profile"] is not None
    assert store.load_calibration_profile("webcam-live") is not None


def test_gaze_refine_ignores_saturated_vertical_axis(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeInteractiveRefinementYSaturatedProvider)
    monkeypatch.setattr("interaction.app._is_interactive_calibration_session", lambda: True)
    monkeypatch.setattr("interaction.app.CalibrationDotOverlay", FakeInteractiveValidationVisualizer)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-refine",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--points",
                "3",
                "--random-seed",
                "7",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["refinement"]["status"] == "success"
    assert payload["refinement"]["suppressed_axes"] == ["y"]
    assert payload["refinement_quality"]["tracked_y_span"] == pytest.approx(0.0)
    assert payload["correction_profile"]["y_scale"] == pytest.approx(1.0)
    assert payload["correction_profile"]["y_offset"] == pytest.approx(0.0)


def test_gaze_refine_can_recalibrate_first(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeInteractiveRefinementProvider)
    monkeypatch.setattr("interaction.app._is_interactive_calibration_session", lambda: True)
    monkeypatch.setattr("interaction.app.CalibrationDotOverlay", FakeInteractiveValidationVisualizer)
    monkeypatch.setattr(
        "interaction.app._perform_live_gaze_calibration",
        lambda *args, **kwargs: {
            "events": [],
            "calibration": {"status": "success"},
            "calibration_steps": [],
            "calibration_targets": [],
            "calibration_visualization": {"status": "enabled"},
            "calibration_quality": {"raw_x_span": 0.2, "raw_y_span": 0.2},
            "calibration_profile": {"x_scale": 1.0, "y_scale": 1.0, "x_offset": 0.0, "y_offset": 0.0},
            "profile": CalibrationProfile(),
        },
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-refine",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--points",
                "3",
                "--random-seed",
                "7",
                "--recalibrate-first",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["initial_calibration"]["calibration"]["status"] == "success"
    assert payload["refinement"]["status"] == "success"


def test_gaze_live_cli_uses_saved_profile_and_triggers_highlight(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8", "--delta-ms", "100"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze"]["status"] == "success"
    assert any(event["phase"] == "triggered" for event in payload["events"])


def test_gaze_live_cli_click_mode_plans_normalized_pointer_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8", "--delta-ms", "100", "--action", "click"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze"]["status"] == "success"
    assert payload["gaze_action"] == "click_target"
    assert any(
        event["result"]
        and event["result"]["details"]["commands"][0][2] == "interaction.platform.macos_runtime"
        and event["result"]["details"]["commands"][0][3] == "click-normalized"
        for event in payload["events"]
    )


def test_gaze_live_cli_cursor_mode_plans_continuous_move_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)
    monkeypatch.setattr("interaction.app.current_position_normalized", lambda: (0.62, 0.58))
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8", "--action", "cursor"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze"]["status"] == "success"
    assert payload["gaze_mode"] == "cursor"
    assert payload["gaze_action"] == "focus_target"
    assert payload["live_gaze"]["moved_events"] >= 1
    assert payload["live_gaze"]["moved_events"] + payload["live_gaze"]["held_events"] >= 1
    assert any(
        event["result"]
        and event["result"]["details"]["commands"][0][2] == "interaction.platform.macos_runtime"
        and event["result"]["details"]["commands"][0][3] == "move-normalized"
        for event in payload["events"]
    )


def test_gaze_live_cli_cursor_mode_reports_initial_cursor_point(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)
    monkeypatch.setattr("interaction.app.current_position_normalized", lambda: (0.62, 0.58))
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8", "--action", "cursor"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["initial_cursor_point"] == {"x": 0.62, "y": 0.58}


def test_gaze_live_cli_cursor_snap_mode_uses_grounded_focus_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    store.save_settings(UserSettings(dwell_ms=300))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8", "--action", "cursor-snap"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gaze_mode"] == "cursor-snap"
    assert payload["gaze_action"] == "focus_target"
    assert any(event["phase"] == "triggered" for event in payload["events"])


def test_gaze_live_cli_cursor_snap_click_mode_uses_grounded_click_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    store.save_settings(UserSettings(dwell_ms=300))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8", "--delta-ms", "100", "--action", "cursor-snap-click"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gaze_mode"] == "cursor-snap-click"
    assert payload["gaze_action"] == "click_target"
    assert any(
        event["result"]
        and event["result"]["details"]["commands"][0][2] == "interaction.platform.macos_runtime"
        and event["result"]["details"]["commands"][0][3] == "click-normalized"
        for event in payload["events"]
    )


def test_gaze_live_cli_cursor_snap_right_click_mode_uses_grounded_secondary_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    store.save_settings(UserSettings(dwell_ms=300))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8", "--delta-ms", "100", "--action", "cursor-snap-right-click"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gaze_mode"] == "cursor-snap-right-click"
    assert payload["gaze_action"] == "right_click_target"
    assert any(
        event["result"]
        and event["result"]["details"]["commands"][0][2] == "interaction.platform.macos_runtime"
        and event["result"]["details"]["commands"][0][3] == "right-click-normalized"
        for event in payload["events"]
    )


def test_gaze_live_cli_cursor_snap_wink_mode_uses_wink_confirmations(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeSnapWinkLeftProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    store.save_settings(UserSettings(dwell_ms=300))

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "6", "--delta-ms", "100", "--action", "cursor-snap-wink"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert "[gaze] TARGET ARMED: wink to confirm" in stderr.getvalue()
    assert "[gaze] WINK DETECTED: LEFT" in stderr.getvalue()
    assert "[gaze] WINK TRIGGERED: LEFT" in stderr.getvalue()
    assert payload["gaze_mode"] == "cursor-snap-wink"
    assert payload["gaze_action"] == "focus_target"
    assert payload["live_gaze"]["wink_confirm"] is True
    assert payload["live_gaze"]["wink_signal_available"] is True
    assert payload["live_gaze"]["left_wink_events"] == 1
    assert any(
        event["result"]
        and event["result"]["details"]["commands"][0][2] == "interaction.platform.macos_runtime"
        and event["result"]["details"]["commands"][0][3] == "click-normalized"
        for event in payload["events"]
    )


def test_gaze_wink_test_cli_uses_left_wink_click_at_current_cursor(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.MediaPipeWebcamGazeProvider", FakeSnapWinkLeftProvider)
    monkeypatch.setattr("interaction.runtime.webcam.current_position_normalized", lambda: (0.62, 0.41))

    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_settings(UserSettings(dry_run=True))

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(["gaze-wink-test", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "6", "--delta-ms", "100"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert "[gaze] WINK DETECTED: LEFT" in stderr.getvalue()
    assert "[gaze] WINK TRIGGERED: LEFT" in stderr.getvalue()
    assert payload["gaze_mode"] == "wink-click-test"
    assert payload["wink_test"]["status"] == "success"
    assert payload["wink_test"]["left_wink_events"] == 1
    assert payload["wink_test"]["left_click_events"] == 1
    assert payload["wink_test"]["right_click_events"] == 0
    assert any(
        event["result"]
        and event["result"]["details"]["commands"][0][2] == "interaction.platform.macos_runtime"
        and event["result"]["details"]["commands"][0][3] == "click-normalized"
        for event in payload["events"]
    )


def test_gaze_wink_test_cli_uses_right_wink_click_at_current_cursor(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.MediaPipeWebcamGazeProvider", FakeSnapWinkRightProvider)
    monkeypatch.setattr("interaction.runtime.webcam.current_position_normalized", lambda: (0.58, 0.37))

    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_settings(UserSettings(dry_run=True))

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(["gaze-wink-test", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "6", "--delta-ms", "100"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert "[gaze] WINK DETECTED: RIGHT" in stderr.getvalue()
    assert "[gaze] WINK TRIGGERED: RIGHT" in stderr.getvalue()
    assert payload["wink_test"]["right_wink_events"] == 1
    assert payload["wink_test"]["right_click_events"] == 1
    assert payload["wink_test"]["left_click_events"] == 0
    assert any(
        event["result"]
        and event["result"]["details"]["commands"][0][2] == "interaction.platform.macos_runtime"
        and event["result"]["details"]["commands"][0][3] == "right-click-normalized"
        for event in payload["events"]
    )


def test_gaze_wink_test_cli_rejects_tracker_without_wink_signals(tmp_path: Path) -> None:
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_settings(UserSettings(dry_run=True))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-wink-test",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--tracker",
                "opencv",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["wink_test"]["status"] == "error"
    assert payload["wink_test"]["error"] == "unsupported_tracker_mode"
    assert payload["gaze_mode_support"]["missing_signals"] == ["left_eye_closed", "right_eye_closed"]


def test_gaze_live_cli_reports_preview_only_warning_when_dry_run_is_enabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeSnapWinkLeftProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    store.save_settings(UserSettings(dwell_ms=300, dry_run=True))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "6", "--action", "cursor-snap-wink"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["settings"]["effective_dry_run"] is True
    assert payload["warnings"] == [
        "Dry-run mode is enabled. Gaze preview is live, but cursor movement and click actions are only simulated. Pass --execute to control macOS for real."
    ]
    assert payload["events"][0]["message"].startswith("Dry-run mode is enabled.")


def test_gaze_live_cli_clears_preview_only_warning_when_execute_is_enabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeSnapWinkLeftProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    store.save_settings(UserSettings(dwell_ms=300, dry_run=True))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-live",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--frames",
                "6",
                "--action",
                "cursor-snap-wink",
                "--execute",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["settings"]["effective_dry_run"] is False
    assert payload["warnings"] == []


def test_gaze_live_cli_cursor_snap_show_dot_uses_calibration_overlay(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)
    overlay_choices: list[str] = []

    class RecordingSnapOverlay(FakeInteractiveValidationVisualizer):
        def open(self) -> None:
            overlay_choices.append("snap")

    class RecordingLiveOverlay(FakePointVisualizer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def open(self) -> None:
            overlay_choices.append("live")

        def close(self) -> None:
            return None

    monkeypatch.setattr("interaction.app.LiveSnapDotsOverlay", RecordingSnapOverlay)
    monkeypatch.setattr("interaction.app.LiveGazeDotOverlay", RecordingLiveOverlay)

    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    store.save_settings(UserSettings(dwell_ms=300))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-live",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--frames",
                "8",
                "--action",
                "cursor-snap",
                "--show-dot",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert overlay_choices == ["snap"]
    assert payload["gaze_visualization"]["status"] == "enabled"


def test_gaze_live_easy_defaults_to_snap_click_visual_preset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)
    calibration_calls: list[bool] = []
    calibration_args: list[object] = []
    overlay_choices: list[str] = []

    def fake_live_calibration(args, *_args, **_kwargs):
        calibration_calls.append(True)
        calibration_args.append(args)
        profile = CalibrationProfile()
        return {
            "events": [],
            "calibration": {"status": "success"},
            "calibration_steps": [],
            "calibration_targets": [],
            "calibration_visualization": {"status": "enabled", "dot_size": args.calibration_dot_size},
            "calibration_quality": {"sample_count": 4},
            "calibration_profile": {
                "x_scale": profile.x_scale,
                "y_scale": profile.y_scale,
                "x_offset": profile.x_offset,
                "y_offset": profile.y_offset,
            },
            "profile": profile,
        }

    class RecordingSnapOverlay(FakeInteractiveValidationVisualizer):
        def open(self) -> None:
            overlay_choices.append("snap")

    class RecordingCameraPreview(FakeCameraPreview):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def open(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr("interaction.app.LiveSnapDotsOverlay", RecordingSnapOverlay)
    monkeypatch.setattr("interaction.app.LiveWebcamDebugPreview", RecordingCameraPreview)
    monkeypatch.setattr("interaction.app._perform_live_gaze_calibration", fake_live_calibration)

    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_settings(UserSettings(dwell_ms=300))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-live-easy",
                "--tracker",
                "opencv",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--frames",
                "8",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gaze_mode"] == "cursor-snap"
    assert payload["tracker"] == "opencv"
    assert payload["gaze_visualization"]["status"] == "enabled"
    assert payload["gaze_visualization"]["camera_status"] == "disabled"
    assert payload["gaze_visualization"]["dot_size"] == 24
    assert payload["initial_calibration"]["calibration_visualization"]["dot_size"] == 44
    assert payload["initial_calibration"]["calibration"]["status"] == "success"
    assert calibration_calls == [True]
    assert calibration_args[0].calibration_dot_size == 44
    assert calibration_args[0].calibration_frames_per_step == 6
    assert overlay_choices == ["snap"]


def test_gaze_live_defaults_to_long_running_session() -> None:
    parser = _build_parser()

    args = parser.parse_args(["gaze-live"])

    assert args.frames == 10000
    assert args.calibration_dot_size == 44
    assert args.calibration_frames_per_step == 6


def test_gaze_live_recalibrate_first_fails_before_tracking(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)

    def fake_live_calibration(*_args, **_kwargs):
        return {
            "events": [],
            "calibration": {
                "status": "error",
                "error": "unstable_calibration_signal",
                "message": "Calibration failed.",
                "payload": {"quality": {"raw_x_span": 0.1}},
            },
            "calibration_steps": [],
            "calibration_targets": [],
            "calibration_visualization": {"status": "enabled"},
            "calibration_quality": {"sample_count": 4},
            "calibration_profile": None,
            "profile": None,
        }

    monkeypatch.setattr("interaction.app._perform_live_gaze_calibration", fake_live_calibration)

    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_settings(UserSettings(dwell_ms=300))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-live",
                "--tracker",
                "opencv",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--frames",
                "8",
                "--action",
                "cursor-snap",
                "--recalibrate-first",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze"]["status"] == "error"
    assert payload["live_gaze"]["error"] == "initial_calibration_failed"
    assert payload["initial_calibration"]["calibration"]["status"] == "error"


def test_gaze_live_easy_keeps_large_calibration_dots_when_live_dots_are_small(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeLiveProvider)
    calibration_dot_sizes: list[int] = []

    class RecordingSnapOverlay(FakeInteractiveValidationVisualizer):
        def open(self) -> None:
            return None

    def fake_live_calibration(args, *_args, **_kwargs):
        calibration_dot_sizes.append(args.calibration_dot_size)
        profile = CalibrationProfile()
        return {
            "events": [],
            "calibration": {"status": "success"},
            "calibration_steps": [],
            "calibration_targets": [],
            "calibration_visualization": {"status": "enabled", "dot_size": args.calibration_dot_size},
            "calibration_quality": {"sample_count": 4},
            "calibration_profile": {
                "x_scale": profile.x_scale,
                "y_scale": profile.y_scale,
                "x_offset": profile.x_offset,
                "y_offset": profile.y_offset,
            },
            "profile": profile,
        }

    monkeypatch.setattr("interaction.app.LiveSnapDotsOverlay", RecordingSnapOverlay)
    monkeypatch.setattr("interaction.app._perform_live_gaze_calibration", fake_live_calibration)

    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_settings(UserSettings(dwell_ms=300))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-live-easy",
                "--tracker",
                "opencv",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--frames",
                "8",
                "--dot-size",
                "18",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gaze_visualization"]["dot_size"] == 18
    assert payload["initial_calibration"]["calibration_visualization"]["dot_size"] == 44
    assert calibration_dot_sizes == [44]


def test_cursor_follow_controller_applies_padding_deadzone_and_max_step() -> None:
    controller = CursorFollowController(smoothing=0.2, deadzone=0.02, edge_padding=0.05, max_step=0.1)

    first = controller.next_point(GazeObservation(confidence=0.9, x_norm=0.0, y_norm=1.0, fixation_ms=100))
    assert first == (0.05, 0.95)

    second = controller.next_point(GazeObservation(confidence=0.9, x_norm=0.055, y_norm=0.945, fixation_ms=100))
    assert second is None

    third = controller.next_point(GazeObservation(confidence=0.9, x_norm=1.0, y_norm=0.0, fixation_ms=100))
    assert third is not None
    assert 0.05 <= third[0] <= 0.95
    assert 0.05 <= third[1] <= 0.95
    assert math.hypot(third[0] - 0.05, third[1] - 0.95) <= 0.100001


def test_relative_cursor_controller_moves_from_neutral_point() -> None:
    controller = RelativeCursorController(
        deadzone=0.05,
        max_step=0.04,
        speed_gain_x=1.0,
        speed_gain_y=1.0,
        neutral_point=(0.5, 0.5),
        dispatched_point=(0.5, 0.5),
    )

    point = controller.next_point(GazeObservation(confidence=0.9, x_norm=0.8, y_norm=0.45, fixation_ms=100))

    assert point is not None
    assert point[0] > 0.5
    assert point[1] == pytest.approx(0.5)


def test_compress_point_around_anchor_reduces_cursor_extent() -> None:
    point = _compress_point_around_anchor(
        NormalizedPoint(0.9, 0.1),
        anchor_point=(0.5, 0.5),
        gain_x=0.25,
        gain_y=0.5,
        edge_padding=0.05,
    )

    assert point.x == 0.6
    assert point.y == 0.3


def test_gaze_dot_geometry_clamps_to_screen_bounds() -> None:
    geometry = GazeDotGeometry.from_normalized(x_norm=1.2, y_norm=-0.2, screen_width=100, screen_height=80, size=28)

    assert geometry.left == 72
    assert geometry.top == 0
    assert geometry.size == 28


def test_run_live_cursor_follow_updates_point_visualizer() -> None:
    provider = FakeLiveProvider()
    visualizer = FakePointVisualizer()

    events, summary = run_live_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=2,
        delta_ms=100,
        point_visualizer=visualizer,
    )

    assert summary["successful_readings"] == 2
    assert len(visualizer.points) == 2
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in visualizer.points)
    assert any(event.result for event in events)


def test_run_live_cursor_follow_updates_camera_preview() -> None:
    provider = FakeLiveProvider()
    preview = FakeCameraPreview()

    events, summary = run_live_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=2,
        delta_ms=100,
        camera_preview=preview,
    )

    assert summary["successful_readings"] == 2
    assert len(preview.calls) == 2
    assert preview.calls[0]["calibrated_point"] is not None
    assert any(event.result for event in events)


def test_run_live_cursor_follow_initializes_visualizer_from_current_cursor_point() -> None:
    provider = FakeLiveProvider()
    visualizer = FakePointVisualizer()

    _events, summary = run_live_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=1,
        delta_ms=100,
        point_visualizer=visualizer,
        initial_cursor_point=(0.62, 0.58),
    )

    assert summary["successful_readings"] == 1
    assert visualizer.points[0] == (0.62, 0.58)


def test_run_live_cursor_follow_uses_gain_to_limit_movement_extent() -> None:
    provider = FakeLiveProvider()

    events, summary = run_live_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=1,
        delta_ms=100,
        initial_cursor_point=(0.62, 0.58),
        response_gain_x=0.25,
        response_gain_y=0.6,
    )

    moved = [event for event in events if event.result]
    assert summary["successful_readings"] == 1
    assert moved
    normalized_point = moved[0].proposal.arguments["normalized_point"]
    assert abs(normalized_point["x"] - 0.62) < abs(0.15 - 0.62)
    assert abs(normalized_point["y"] - 0.58) < abs(0.15 - 0.58)


def test_run_live_cursor_follow_supports_asymmetric_axis_gains() -> None:
    provider = FakeLiveProvider()

    events, summary = run_live_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=1,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        response_gain_x=0.2,
        response_gain_y=0.6,
    )

    moved = [event for event in events if event.result]
    assert summary["successful_readings"] == 1
    assert moved
    normalized_point = moved[0].proposal.arguments["normalized_point"]
    assert abs(normalized_point["x"] - 0.5) < abs(normalized_point["y"] - 0.5)


def test_run_live_relative_cursor_follow_moves_cursor_from_raw_signal() -> None:
    provider = FakeRelativeLiveProvider()

    events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=2,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=6,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    moved = [event for event in events if event.result]
    assert summary["successful_readings"] == 2
    assert summary["neutral_point"] == {"x": 0.5, "y": 0.5}
    assert moved
    assert moved[0].proposal.arguments["control_mode"] == "relative"
    assert moved[0].proposal.arguments["normalized_point"]["x"] > 0.5


def test_run_live_relative_cursor_follow_moves_on_subtle_post_neutral_shift() -> None:
    provider = FakeRelativeSubtleProvider()

    events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=3,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=6,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    moved = [event for event in events if event.result]
    assert moved
    assert summary["moved_events"] >= 1
    assert summary["effective_deadzone"]["x"] < 0.05
    assert summary["online_bias_correction"]["updates"] == 0


def test_relative_cursor_controller_recenter_resets_velocity_and_updates_neutral_point() -> None:
    controller = RelativeCursorController(
        neutral_point=(0.5, 0.5),
        dispatched_point=(0.5, 0.5),
        velocity_x=0.01,
        velocity_y=-0.01,
    )

    controller.recenter((0.56, 0.52))

    assert controller.neutral_point == pytest.approx((0.56, 0.52))
    assert controller.velocity_x == 0.0
    assert controller.velocity_y == 0.0


def test_run_live_relative_cursor_follow_does_not_recenter_while_signal_is_still_offset() -> None:
    provider = FakeRelativeLiveProvider()

    _events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=2,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=6,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert summary["recenter_events"] == 0


def test_run_live_relative_cursor_follow_does_not_recenter_on_slow_drift() -> None:
    provider = FakeRelativeDriftProvider()

    _events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=3,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=6,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert summary["recenter_events"] == 0


def test_run_live_relative_cursor_follow_freezes_on_large_head_posture_change() -> None:
    provider = FakeRelativePostureShiftProvider()

    _events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=9,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=6,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert summary["posture_freeze_events"] >= 1
    assert summary["moved_events"] == 0


def test_run_live_relative_cursor_follow_recenters_and_resumes_after_posture_stabilizes() -> None:
    provider = FakeRelativePostureRecoverProvider()

    _events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=12,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=6,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert summary["posture_freeze_events"] >= 1
    assert summary["posture_recenter_events"] >= 1
    assert summary["moved_events"] >= 1


def test_run_live_relative_cursor_follow_does_not_false_freeze_on_eye_only_motion() -> None:
    provider = FakeRelativeEyeOnlyProvider()

    _events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=8,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=6,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert summary["posture_freeze_events"] == 0
    assert summary["moved_events"] >= 1


def test_run_live_relative_cursor_follow_does_not_move_from_head_assist_alone() -> None:
    provider = FakeRelativeHeadOnlyProvider()

    _events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=8,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=3,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert summary["posture_signal_available"] is True
    assert summary["posture_freeze_events"] == 0
    assert summary["moved_events"] == 0
    assert summary["head_assist_active_frames"] == 0


def test_run_live_relative_cursor_follow_learns_small_stable_bias_near_neutral() -> None:
    provider = FakeRelativeBiasLearningProvider()

    _events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=4,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=6,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert summary["moved_events"] == 0
    assert summary["online_bias_correction"]["updates"] >= 1
    assert summary["online_bias_correction"]["active_ms"] > 0
    assert summary["online_bias_correction"]["offset"]["x"] < 0.0


def test_run_live_relative_cursor_follow_small_head_assist_moves_when_eye_only_is_idle() -> None:
    eye_only_provider = FakeRelativeNeutralProvider()
    assist_provider = FakeRelativeSmallHeadAssistProvider()

    _eye_events, eye_only_summary = run_live_relative_cursor_follow(
        eye_only_provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=9,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=3,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    _assist_events, assist_summary = run_live_relative_cursor_follow(
        assist_provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=9,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=3,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert eye_only_summary["moved_events"] == 0
    assert assist_summary["moved_events"] >= 1
    assert assist_summary["posture_freeze_events"] == 0
    assert assist_summary["posture_signal_available"] is True


def test_run_live_relative_cursor_follow_large_head_motion_still_freezes_with_assist_signal() -> None:
    provider = FakeRelativeLargeHeadAssistProvider()

    _events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=9,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=6,
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert summary["posture_signal_available"] is True
    assert summary["posture_freeze_events"] >= 1
    assert summary["moved_events"] == 0


def test_run_live_relative_cursor_follow_can_trigger_dwell_click() -> None:
    provider = FakeRelativeClickProvider()
    provider._points.extend([NormalizedPoint(0.50, 0.50) for _ in range(24)])

    events, summary = run_live_relative_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=30,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        neutral_frames=3,
        response_gain_x=1.0,
        response_gain_y=1.0,
        click_on_dwell=True,
        dwell_ms=300,
    )

    clicks = [
        event
        for event in events
        if event.result
        and event.result.details["commands"][0][3] == "click-normalized"
    ]
    assert summary["click_events"] == 1
    assert clicks


def test_run_live_assisted_cursor_follow_only_moves_when_control_signal_is_active() -> None:
    provider = FakeAssistedLiveProvider()

    events, summary = run_live_assisted_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=7,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    moved = [event for event in events if event.result and event.proposal and event.proposal.arguments["control_mode"] == "assist"]
    assert summary["engaged_events"] == 1
    assert summary["released_events"] == 1
    assert summary["recenter_events"] >= 1
    assert summary["control_signal_available"] is True
    assert summary["online_bias_correction"]["updates"] == 0
    assert moved


def test_run_live_assisted_cursor_follow_does_not_move_from_head_assist_alone() -> None:
    provider = FakeAssistedHeadOnlyProvider()

    _events, summary = run_live_assisted_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=7,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert summary["engaged_events"] == 1
    assert summary["control_signal_available"] is True
    assert summary["posture_freeze_events"] == 0
    assert summary["moved_events"] == 0
    assert summary["head_assist_active_frames"] == 0


def test_run_live_assisted_cursor_follow_can_trigger_dwell_click() -> None:
    provider = FakeAssistedClickProvider()

    events, summary = run_live_assisted_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=9,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        response_gain_x=1.0,
        response_gain_y=1.0,
        click_on_dwell=True,
        dwell_ms=200,
    )

    clicks = [
        event
        for event in events
        if event.result
        and event.result.details["commands"][0][3] == "click-normalized"
    ]
    assert summary["click_events"] == 1
    assert clicks


def test_run_live_assisted_cursor_follow_learns_small_stable_bias_near_neutral() -> None:
    provider = FakeAssistedBiasLearningProvider()

    _events, summary = run_live_assisted_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        response_gain_x=1.0,
        response_gain_y=1.0,
    )

    assert summary["moved_events"] == 0
    assert summary["online_bias_correction"]["updates"] >= 1
    assert summary["online_bias_correction"]["active_ms"] > 0
    assert summary["online_bias_correction"]["offset"]["x"] < 0.0


def test_run_live_assisted_cursor_follow_suppresses_inconsistent_starting_saccades() -> None:
    provider = FakeAssistedSaccadeProvider()

    events, summary = run_live_assisted_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        response_gain_x=1.0,
        response_gain_y=1.0,
        drive_commit_ms=180,
    )

    assert summary["engaged_events"] == 1
    assert summary["moved_events"] == 0
    assert summary["drive_ready_frames"] >= 2
    assert any("Hold the same direction briefly" in event.message for event in events)


def test_run_live_assisted_cursor_follow_auto_disarms_after_idle() -> None:
    provider = FakeAssistedIdleDisarmProvider()

    events, summary = run_live_assisted_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=7,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        response_gain_x=1.0,
        response_gain_y=1.0,
        drive_commit_ms=140,
        idle_disarm_ms=200,
    )

    assert summary["moved_events"] >= 1
    assert summary["auto_disarm_events"] == 1
    assert summary["control_state"] == "observe"
    assert any("auto-disarmed after inactivity" in event.message for event in events)


def test_run_live_assisted_cursor_follow_updates_visualizer_styles_for_control_states() -> None:
    provider = FakeAssistedIdleDisarmProvider()
    visualizer = FakePointVisualizer()

    _events, summary = run_live_assisted_cursor_follow(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=7,
        delta_ms=100,
        initial_cursor_point=(0.5, 0.5),
        response_gain_x=1.0,
        response_gain_y=1.0,
        drive_commit_ms=140,
        idle_disarm_ms=200,
        point_visualizer=visualizer,
    )

    fills = {style["fill"] for style in visualizer.styles}
    assert "#0a84ff" in fills
    assert "#64d2ff" in fills
    assert "#30d158" in fills
    assert summary["drive_frames"] >= 1


def test_default_webcam_snap_targets_use_denser_grid() -> None:
    targets = default_webcam_snap_targets()

    assert len(targets) == 48
    widths = [target.width for target in targets]
    heights = [target.height for target in targets]
    assert all(width == pytest.approx(0.115) for width in widths)
    assert all(height == pytest.approx(0.15) for height in heights)
    assert min(target.x for target in targets) == pytest.approx(0.04)
    assert min(target.y for target in targets) == pytest.approx(0.05)
    assert max(target.x + target.width for target in targets) == pytest.approx(0.96)
    assert max(target.y + target.height for target in targets) == pytest.approx(0.95)


def test_run_live_cursor_snap_can_click_on_second_dwell() -> None:
    provider = FakeLiveProvider()

    events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=7,
        delta_ms=100,
        dwell_ms=300,
        second_dwell_action=ActionName.CLICK_TARGET,
    )

    assert summary["snap_events"] == 1
    assert summary["click_events"] == 1
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "move-normalized"
        for event in events
    )
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "click-normalized"
        for event in events
    )


def test_run_live_cursor_snap_uses_refined_point_for_focus_action() -> None:
    provider = FakeLiveProvider()

    events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=4,
        delta_ms=100,
        dwell_ms=300,
    )

    assert summary["snap_events"] == 1
    trigger = next(event for event in events if event.result and event.result.details["commands"][0][3] == "move-normalized")
    normalized_point = trigger.proposal.arguments["normalized_point"]
    assert normalized_point["x"] == pytest.approx(0.1481)
    assert normalized_point["y"] == pytest.approx(0.15)


def test_run_live_cursor_snap_can_right_click_on_second_dwell() -> None:
    provider = FakeLiveProvider()

    events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=7,
        delta_ms=100,
        dwell_ms=300,
        second_dwell_action=ActionName.RIGHT_CLICK_TARGET,
    )

    assert summary["snap_events"] == 1
    assert summary["click_events"] == 1
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "move-normalized"
        for event in events
    )
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "right-click-normalized"
        for event in events
    )


def test_run_live_cursor_snap_can_trigger_left_wink_click_on_armed_target() -> None:
    provider = FakeSnapWinkLeftProvider()

    events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        dwell_ms=300,
        wink_confirm=True,
    )

    assert summary["snap_events"] == 1
    assert summary["click_events"] == 1
    assert summary["wink_confirm"] is True
    assert summary["wink_signal_available"] is True
    assert summary["left_wink_events"] == 1
    assert summary["right_wink_events"] == 0
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "click-normalized"
        for event in events
    )


def test_run_live_cursor_snap_can_trigger_right_wink_click_on_armed_target() -> None:
    provider = FakeSnapWinkRightProvider()

    events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        dwell_ms=300,
        wink_confirm=True,
    )

    assert summary["snap_events"] == 1
    assert summary["click_events"] == 1
    assert summary["wink_confirm"] is True
    assert summary["wink_signal_available"] is True
    assert summary["left_wink_events"] == 0
    assert summary["right_wink_events"] == 1
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "right-click-normalized"
        for event in events
    )
    assert any("Right wink detected" in event.message for event in events)


def test_run_live_cursor_snap_hides_overlay_before_right_wink_click() -> None:
    provider = FakeSnapWinkRightProvider()

    class RecordingSnapVisualizer(FakeInteractiveValidationVisualizer):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    visualizer = RecordingSnapVisualizer()

    events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        dwell_ms=300,
        wink_confirm=True,
        point_visualizer=visualizer,
    )

    assert summary["snap_events"] == 1
    assert summary["click_events"] == 1
    assert summary["right_wink_events"] == 1
    assert visualizer.hidden_calls >= 2
    assert visualizer.close_calls >= 2
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "right-click-normalized"
        for event in events
    )


def test_run_live_cursor_snap_closes_camera_preview_before_right_wink_click() -> None:
    provider = FakeSnapWinkRightProvider()
    preview = FakeCameraPreview()

    events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        dwell_ms=300,
        wink_confirm=True,
        camera_preview=preview,
    )

    assert summary["right_wink_events"] == 1
    assert preview.close_calls >= 1
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "right-click-normalized"
        for event in events
    )


def test_run_live_cursor_snap_can_trigger_right_wink_click_when_other_eye_partially_closes() -> None:
    provider = FakeSnapWinkRightLeakyProvider()

    events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        dwell_ms=300,
        wink_confirm=True,
    )

    assert summary["snap_events"] == 1
    assert summary["click_events"] == 1
    assert summary["right_wink_events"] == 1
    assert summary["left_wink_events"] == 0
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "right-click-normalized"
        for event in events
    )


def test_run_live_cursor_snap_does_not_treat_bilateral_blink_as_wink() -> None:
    provider = FakeSnapBlinkProvider()

    events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        dwell_ms=300,
        wink_confirm=True,
    )

    assert summary["snap_events"] == 1
    assert summary["click_events"] == 0
    assert summary["left_wink_events"] == 0
    assert summary["right_wink_events"] == 0
    assert not any(
        event.result
        and event.result.details["commands"][0][3] in {"click-normalized", "right-click-normalized"}
        for event in events
    )


def test_run_live_cursor_snap_locks_armed_target_during_wink_confirmation() -> None:
    provider = FakeSnapWinkRightDriftProvider()
    visualizer = FakePointVisualizer()

    events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        dwell_ms=300,
        wink_confirm=True,
        point_visualizer=visualizer,
    )

    assert summary["snap_events"] == 1
    assert summary["click_events"] == 1
    assert summary["right_wink_events"] == 1
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "right-click-normalized"
        for event in events
    )
    assert len(visualizer.points) >= 5
    refined_target_point = (0.1481, 0.15)
    assert visualizer.points[3] == pytest.approx(refined_target_point)
    assert visualizer.points[4] == pytest.approx(refined_target_point)


def test_run_live_cursor_snap_can_cancel_armed_target_by_looking_away() -> None:
    provider = FakeSnapWinkCancelProvider()

    _events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=10,
        delta_ms=100,
        dwell_ms=300,
        wink_confirm=True,
    )

    assert summary["snap_events"] >= 1
    assert summary["click_events"] == 0
    assert summary["arm_cancel_events"] >= 1
    assert summary["armed_target_id"] is None


def test_run_live_wink_click_test_can_trigger_left_click() -> None:
    provider = FakeSnapWinkLeftProvider()

    events, summary = run_live_wink_click_test(
        provider,
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        current_point_getter=lambda: (0.61, 0.43),
    )

    assert summary["left_wink_events"] == 1
    assert summary["left_click_events"] == 1
    assert summary["right_click_events"] == 0
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "click-normalized"
        for event in events
    )


def test_run_live_wink_click_test_can_trigger_right_click() -> None:
    provider = FakeSnapWinkRightProvider()

    events, summary = run_live_wink_click_test(
        provider,
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        current_point_getter=lambda: (0.57, 0.39),
    )

    assert summary["right_wink_events"] == 1
    assert summary["right_click_events"] == 1
    assert summary["left_click_events"] == 0
    assert any(
        event.result
        and event.result.details["commands"][0][3] == "right-click-normalized"
        for event in events
    )


def test_run_live_wink_click_test_does_not_treat_blink_as_click() -> None:
    provider = FakeSnapBlinkProvider()

    events, summary = run_live_wink_click_test(
        provider,
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=6,
        delta_ms=100,
        current_point_getter=lambda: (0.5, 0.5),
    )

    assert summary["left_click_events"] == 0
    assert summary["right_click_events"] == 0
    assert summary["left_wink_events"] == 0
    assert summary["right_wink_events"] == 0
    assert not any(
        event.result
        and event.result.details["commands"][0][3] in {"click-normalized", "right-click-normalized"}
        for event in events
    )


def test_run_live_cursor_snap_preview_uses_refined_point_within_target() -> None:
    class FakeOffsetSnapProvider(FakeLiveProvider):
        def __init__(self) -> None:
            super().__init__()
            self._points = [
                NormalizedPoint(0.24, 0.28),
                NormalizedPoint(0.24, 0.28),
                NormalizedPoint(0.24, 0.28),
            ]

    provider = FakeOffsetSnapProvider()
    visualizer = FakePointVisualizer()

    _events, summary = run_live_cursor_snap(
        provider,
        CalibrationProfile(),
        adapter=MacOSPlatformAdapter(dry_run=True),
        frames=3,
        delta_ms=100,
        dwell_ms=1000,
        point_visualizer=visualizer,
    )

    assert summary["preview_frames"] == 3
    assert len(visualizer.points) == 3
    first_point = visualizer.points[0]
    last_point = visualizer.points[-1]
    assert first_point == pytest.approx((0.24, 0.28))
    assert last_point == pytest.approx((0.24, 0.28))


def test_gaze_live_cli_cursor_relative_mode_reports_neutral_point(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeRelativeLiveProvider)
    monkeypatch.setattr("interaction.app.current_position_normalized", lambda: (0.62, 0.58))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-live",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--frames",
                "2",
                "--action",
                "cursor-relative",
                "--dwell-ms",
                "321",
                "--dwell-min-confidence",
                "0.77",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gaze_mode"] == "cursor-relative"
    assert payload["live_gaze"]["status"] == "success"
    assert payload["gaze_mode_support"]["requires_calibration_profile"] is False
    assert payload["live_gaze"]["neutral_point"] == {"x": 0.5, "y": 0.5}
    assert payload["initial_cursor_point"] == {"x": 0.62, "y": 0.58}
    assert payload["cursor_settings"]["deadzone"] == pytest.approx(0.025)
    assert payload["cursor_settings"]["max_step"] == pytest.approx(0.05)
    assert payload["cursor_settings"]["gain_x"] == pytest.approx(0.6)
    assert payload["cursor_settings"]["gain_y"] == pytest.approx(0.6)
    assert payload["gaze_control_settings"]["dwell_ms"] == 321
    assert payload["gaze_control_settings"]["dwell_min_confidence"] == pytest.approx(0.77)
    assert payload["calibration_profile"] is None


def test_gaze_live_parser_defaults_to_lower_latency_sampling() -> None:
    parser = _build_parser()

    gaze_live_args = parser.parse_args(["gaze-live"])
    gaze_wink_args = parser.parse_args(["gaze-wink-test"])
    gaze_benchmark_args = parser.parse_args(["gaze-benchmark"])

    assert gaze_live_args.delta_ms == 41
    assert gaze_live_args.wink_closed_threshold == 0.45
    assert gaze_live_args.wink_open_threshold == 0.25
    assert gaze_wink_args.delta_ms == 41
    assert gaze_wink_args.wink_closed_threshold == 0.45
    assert gaze_wink_args.wink_open_threshold == 0.25
    assert gaze_benchmark_args.delta_ms == 41
    assert gaze_benchmark_args.wink_closed_threshold == 0.45
    assert gaze_benchmark_args.wink_open_threshold == 0.25


def test_gaze_live_cli_cursor_relative_click_mode_triggers_click(monkeypatch, tmp_path: Path) -> None:
    class FakeRelativeCliClickProvider(FakeLiveProvider):
        def __init__(self, camera_index: int = 0) -> None:
            super().__init__(camera_index)
            self._points = [
                NormalizedPoint(0.50, 0.50),
                NormalizedPoint(0.50, 0.50),
                NormalizedPoint(0.50, 0.50),
                NormalizedPoint(0.50, 0.50),
                NormalizedPoint(0.50, 0.50),
                NormalizedPoint(0.50, 0.50),
                NormalizedPoint(0.80, 0.50),
                NormalizedPoint(0.80, 0.50),
                NormalizedPoint(0.80, 0.50),
                NormalizedPoint(0.50, 0.50),
                NormalizedPoint(0.50, 0.50),
                NormalizedPoint(0.50, 0.50),
                NormalizedPoint(0.50, 0.50),
                NormalizedPoint(0.50, 0.50),
            ]

    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeRelativeCliClickProvider)
    monkeypatch.setattr("interaction.app.current_position_normalized", lambda: (0.62, 0.58))
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    store.save_settings(UserSettings(dwell_ms=300))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-live",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--frames",
                "8",
                "--action",
                "cursor-relative-click",
                "--cursor-gain",
                "1.0",
                "--cursor-max-step",
                "0.04",
                "--cursor-deadzone",
                "0.05",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gaze_mode"] == "cursor-relative-click"
    assert payload["gaze_action"] == "click_target"
    assert payload["live_gaze"]["status"] == "success"
    assert payload["live_gaze"]["successful_readings"] == 8


def test_gaze_live_cli_cursor_assist_mode_uses_control_signal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeAssistedLiveProvider)
    monkeypatch.setattr("interaction.app.current_position_normalized", lambda: (0.62, 0.58))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "6", "--action", "cursor-assist"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gaze_mode"] == "cursor-assist"
    assert payload["gaze_action"] == "focus_target"
    assert payload["live_gaze"]["engaged_events"] == 1
    assert payload["live_gaze"]["control_signal_available"] is True
    assert payload["gaze_mode_support"]["requires_calibration_profile"] is False


def test_gaze_live_cli_cursor_assist_rejects_opencv_without_required_signals(tmp_path: Path) -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-live",
                "--tracker",
                "opencv",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--frames",
                "6",
                "--action",
                "cursor-assist",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze"]["status"] == "error"
    assert payload["live_gaze"]["error"] == "unsupported_tracker_mode"
    assert payload["live_gaze"]["payload"]["required_signals"] == ["mouth_open"]


def test_gaze_live_cli_cursor_snap_wink_rejects_opencv_without_required_signals(tmp_path: Path) -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-live",
                "--tracker",
                "opencv",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--frames",
                "6",
                "--action",
                "cursor-snap-wink",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze"]["status"] == "error"
    assert payload["live_gaze"]["error"] == "unsupported_tracker_mode"
    assert payload["live_gaze"]["payload"]["required_signals"] == ["left_eye_closed", "right_eye_closed"]


def test_gaze_live_cli_cursor_assist_reuses_provider_after_recalibration(monkeypatch, tmp_path: Path) -> None:
    FakeRecalibrateThenAssistProvider.instances = []
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeRecalibrateThenAssistProvider)
    monkeypatch.setattr("interaction.app.current_position_normalized", lambda: (0.62, 0.58))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-live",
                "--tracker",
                "opencv",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--frames",
                "6",
                "--action",
                "cursor-assist",
                "--recalibrate-first",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["initial_calibration"]["calibration"]["status"] == "success"
    assert payload["live_gaze"]["status"] == "success"
    assert payload["live_gaze"]["successful_readings"] == 6
    assert len(FakeRecalibrateThenAssistProvider.instances) == 1
    provider = FakeRecalibrateThenAssistProvider.instances[0]
    assert provider.open_count == 1
    assert provider.close_count == 1


def test_gaze_live_cli_cursor_assist_click_mode_triggers_click(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeAssistedClickProvider)
    monkeypatch.setattr("interaction.app.current_position_normalized", lambda: (0.62, 0.58))
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    store.save_settings(UserSettings(dwell_ms=200))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "7", "--action", "cursor-assist-click"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gaze_mode"] == "cursor-assist-click"
    assert payload["gaze_action"] == "click_target"
    assert payload["live_gaze"]["status"] == "success"
    assert payload["live_gaze"]["successful_readings"] == 7


def test_run_live_webcam_trace_hides_point_visualizer_when_readings_are_missing() -> None:
    class FakeNoReadingProvider(FakeLiveProvider):
        def read(self, *, delta_ms: int = 100):
            return None

    provider = FakeNoReadingProvider()
    visualizer = FakePointVisualizer()
    loop = GazeTrackingLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    _events, summary = run_live_webcam_trace(
        provider,
        loop,
        frames=3,
        delta_ms=100,
        targets=default_webcam_targets(),
        point_visualizer=visualizer,
    )

    assert summary["missing_readings"] == 3
    assert visualizer.hidden_calls == 3


def test_run_live_webcam_trace_updates_camera_preview() -> None:
    provider = FakeLiveProvider()
    preview = FakeCameraPreview()
    loop = GazeTrackingLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    _events, summary = run_live_webcam_trace(
        provider,
        loop,
        frames=2,
        delta_ms=100,
        targets=default_webcam_targets(),
        camera_preview=preview,
    )

    assert summary["successful_readings"] == 2
    assert len(preview.calls) == 2
    assert preview.calls[0]["calibrated_point"] is not None


def test_run_live_webcam_benchmark_reports_stability_and_wink_metrics() -> None:
    provider = FakeBenchmarkProvider()
    preview = FakeCameraPreview()
    visualizer = FakePointVisualizer()

    events, summary = run_live_webcam_benchmark(
        provider,
        frames=4,
        delta_ms=100,
        point_visualizer=visualizer,
        camera_preview=preview,
    )

    assert summary["successful_readings"] == 4
    assert summary["missing_readings"] == 0
    assert summary["wink_signal_available"] is True
    assert summary["left_wink_frames"] >= 1
    assert summary["left_wink_episodes"] == 1
    assert summary["raw_jitter_x"] > summary["fused_jitter_x"]
    assert summary["x_axis_confidence_mean"] > 0.0
    assert len(preview.calls) == 4
    assert visualizer.points
    assert events[-1].phase == GazeLoopPhase.IDLE


def test_gaze_benchmark_cli_reports_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.MediaPipeWebcamGazeProvider", FakeBenchmarkProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-benchmark", "--tracker", "mediapipe", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "4"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gaze_mode"] == "benchmark"
    assert payload["benchmark"]["status"] == "success"
    assert payload["benchmark"]["successful_readings"] == 4
    assert payload["benchmark"]["left_wink_episodes"] == 1
    assert payload["benchmark"]["raw_fused_mean_distance"] > 0.0
    summary_path = Path(payload["summary_path"])
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["benchmark"]["successful_readings"] == 4
    assert summary_payload["session_log"] == payload["session_log"]


def test_gaze_benchmark_report_cli_summarizes_recent_runs(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    logs_dir = runtime_dir / "logs"
    logs_dir.mkdir(parents=True)
    first_summary = logs_dir / "gaze-benchmark.summary.json"
    second_summary = logs_dir / "gaze-benchmark-2.summary.json"
    first_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:00:00Z",
                "session_log": str(logs_dir / "gaze-benchmark.jsonl"),
                "summary_path": str(first_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 95,
                    "missing_readings": 5,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.90,
                    "fused_jitter_x": 0.020,
                    "fused_jitter_y": 0.030,
                    "fused_drift_distance": 0.040,
                    "raw_fused_mean_distance": 0.025,
                    "x_axis_confidence_mean": 0.82,
                    "y_axis_confidence_mean": 0.60,
                    "head_yaw_abs_mean": 0.02,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 0,
                },
            },
            indent=2,
        )
        + "\n"
    )
    second_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:05:00Z",
                "session_log": str(logs_dir / "gaze-benchmark-2.jsonl"),
                "summary_path": str(second_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 99,
                    "missing_readings": 1,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.93,
                    "fused_jitter_x": 0.010,
                    "fused_jitter_y": 0.020,
                    "fused_drift_distance": 0.030,
                    "raw_fused_mean_distance": 0.015,
                    "x_axis_confidence_mean": 0.88,
                    "y_axis_confidence_mean": 0.72,
                    "head_yaw_abs_mean": 0.01,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-benchmark-report", "--runtime-dir", str(runtime_dir), "--limit", "5"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["benchmark_report"]["status"] == "success"
    assert payload["benchmark_report"]["rendering"] == {
        "preset": None,
        "format": "json",
        "brief": False,
        "top": None,
        "output": None,
    }
    assert payload["benchmark_report"]["run_count"] == 2
    assert payload["benchmark_report"]["latest"]["session_log"].endswith("gaze-benchmark-2.jsonl")
    assert payload["benchmark_report"]["aggregate"]["mean_fused_jitter_distance"] > 0.0
    assert payload["benchmark_report"]["aggregate"]["mean_quality_score"] > 0.0
    assert payload["benchmark_report"]["quality_summary"]["latest_quality_score"] >= payload["benchmark_report"]["quality_summary"]["mean_quality_score"]
    assert payload["benchmark_report"]["trend"]["overall"] == "improving"
    assert payload["benchmark_report"]["trend"]["metrics"]["fused_jitter_distance"]["status"] == "improved"
    assert payload["benchmark_report"]["trend"]["metrics"]["y_axis_confidence_mean"]["status"] == "improved"
    assert payload["benchmark_report"]["best_runs"]["highest_quality_score"]["session_log"].endswith("gaze-benchmark-2.jsonl")
    assert payload["benchmark_report"]["groups_by_tracker"]["mediapipe"]["run_count"] == 2
    assert payload["benchmark_report"]["groups_by_tracker"]["mediapipe"]["trend"]["overall"] == "improving"
    assert payload["benchmark_report"]["groups_by_session_prefix"]["gaze-benchmark"]["run_count"] == 2
    assert payload["benchmark_report"]["groups_by_session_prefix"]["gaze-benchmark"]["best_runs"]["highest_quality_score"]["session_log"].endswith("gaze-benchmark-2.jsonl")
    assert payload["benchmark_report"]["delta_from_previous_successful"]["fused_jitter_distance_delta"] < 0.0
    assert payload["benchmark_report"]["best_runs"]["lowest_dropout_rate"]["session_log"].endswith("gaze-benchmark-2.jsonl")
    assert payload["benchmark_report"]["dominant_bottleneck"]["category"] == "vertical_axis"
    assert payload["benchmark_report"]["groups_by_tracker"]["mediapipe"]["dominant_bottleneck"]["category"] == "vertical_axis"
    assert payload["benchmark_report"]["group_comparisons"]["tracker"]["best_group"]["group"] == "mediapipe"
    assert payload["benchmark_report"]["group_comparisons"]["tracker"]["weakest_group"]["group"] == "mediapipe"
    assert payload["benchmark_report"]["group_comparisons"]["session_prefix"]["group_count"] == 1
    assert payload["benchmark_report"]["group_metric_hotspots"]["tracker"]["metrics"]["mean_quality_score"]["best_group"]["group"] == "mediapipe"
    assert payload["benchmark_report"]["group_metric_hotspots"]["session_prefix"]["metrics"]["mean_fused_jitter_distance"]["weakest_group"]["group"] == "gaze-benchmark"
    assert payload["benchmark_report"]["metric_leaderboards"]["run_count"] == 2
    assert payload["benchmark_report"]["metric_leaderboards"]["metrics"]["quality_score"]["best_run"]["session_log"].endswith("gaze-benchmark-2.jsonl")
    assert payload["benchmark_report"]["metric_leaderboards"]["metrics"]["fused_jitter_distance"]["weakest_run"]["session_log"].endswith("gaze-benchmark.jsonl")
    assert payload["benchmark_report"]["recent_changes"]["transition_count"] == 1
    assert payload["benchmark_report"]["recent_changes"]["latest_transition"]["to_session_log"].endswith("gaze-benchmark-2.jsonl")
    assert payload["benchmark_report"]["recent_changes"]["top_improvements"][0]["metric"] == "quality_score"
    assert payload["benchmark_report"]["recent_changes"]["top_regressions"] == []
    assert payload["benchmark_report"]["notes"]
    assert payload["benchmark_report"]["recommendations"]


def test_gaze_live_cli_reports_camera_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeErrorProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze"]["status"] == "error"
    assert payload["live_gaze"]["error"] == "camera_unavailable"


def test_gaze_benchmark_report_cli_text_format(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    logs_dir = runtime_dir / "logs"
    logs_dir.mkdir(parents=True)
    first_summary = logs_dir / "gaze-benchmark.summary.json"
    second_summary = logs_dir / "gaze-benchmark-2.summary.json"
    first_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:00:00Z",
                "session_log": str(logs_dir / "gaze-benchmark.jsonl"),
                "summary_path": str(first_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 95,
                    "missing_readings": 5,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.90,
                    "fused_jitter_x": 0.020,
                    "fused_jitter_y": 0.030,
                    "fused_drift_distance": 0.040,
                    "raw_fused_mean_distance": 0.025,
                    "x_axis_confidence_mean": 0.82,
                    "y_axis_confidence_mean": 0.60,
                    "head_yaw_abs_mean": 0.02,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 0,
                },
            },
            indent=2,
        )
        + "\n"
    )
    second_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:05:00Z",
                "session_log": str(logs_dir / "gaze-benchmark-2.jsonl"),
                "summary_path": str(second_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 99,
                    "missing_readings": 1,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.93,
                    "fused_jitter_x": 0.010,
                    "fused_jitter_y": 0.020,
                    "fused_drift_distance": 0.030,
                    "raw_fused_mean_distance": 0.015,
                    "x_axis_confidence_mean": 0.88,
                    "y_axis_confidence_mean": 0.72,
                    "head_yaw_abs_mean": 0.01,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-benchmark-report", "--runtime-dir", str(runtime_dir), "--limit", "5", "--format", "text"])
    output = stdout.getvalue()

    assert exit_code == 0
    assert "Gaze Benchmark Report" in output
    assert "Status: success" in output
    assert "Runs: 2 total, 2 successful" in output
    assert "Latest Successful Run" in output
    assert "gaze-benchmark-2.jsonl" in output
    assert "Trend" in output
    assert "overall improving" in output
    assert "Recommendations" in output
    assert "Bottleneck" in output
    assert "vertical_axis" in output
    assert "By Tracker" in output
    assert "mediapipe: 2 runs" in output
    assert "By Session Prefix" in output
    assert "gaze-benchmark: 2 runs" in output
    assert "Tracker Comparison" in output
    assert "best mediapipe" in output
    assert "Session Comparison" in output
    assert "Tracker Hotspots" in output
    assert "quality: best mediapipe" in output
    assert "Metric Leaderboards" in output
    assert "quality: best gaze-benchmark-2.jsonl" in output
    assert "Recent Changes" in output
    assert "quality improved" in output


def test_gaze_benchmark_report_cli_text_brief_format(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    logs_dir = runtime_dir / "logs"
    logs_dir.mkdir(parents=True)
    first_summary = logs_dir / "gaze-benchmark.summary.json"
    second_summary = logs_dir / "gaze-benchmark-2.summary.json"
    first_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:00:00Z",
                "session_log": str(logs_dir / "gaze-benchmark.jsonl"),
                "summary_path": str(first_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 95,
                    "missing_readings": 5,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.90,
                    "fused_jitter_x": 0.020,
                    "fused_jitter_y": 0.030,
                    "fused_drift_distance": 0.040,
                    "raw_fused_mean_distance": 0.025,
                    "x_axis_confidence_mean": 0.82,
                    "y_axis_confidence_mean": 0.60,
                    "head_yaw_abs_mean": 0.02,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 0,
                },
            },
            indent=2,
        )
        + "\n"
    )
    second_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:05:00Z",
                "session_log": str(logs_dir / "gaze-benchmark-2.jsonl"),
                "summary_path": str(second_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 99,
                    "missing_readings": 1,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.93,
                    "fused_jitter_x": 0.010,
                    "fused_jitter_y": 0.020,
                    "fused_drift_distance": 0.030,
                    "raw_fused_mean_distance": 0.015,
                    "x_axis_confidence_mean": 0.88,
                    "y_axis_confidence_mean": 0.72,
                    "head_yaw_abs_mean": 0.01,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-benchmark-report",
                "--runtime-dir",
                str(runtime_dir),
                "--limit",
                "5",
                "--format",
                "text",
                "--brief",
            ]
        )
    output = stdout.getvalue()

    assert exit_code == 0
    assert "Gaze Benchmark Report" in output
    assert "Latest" in output
    assert "Bottleneck" in output
    assert "Recent" in output
    assert "Leaders" in output
    assert "Metric Leaderboards" not in output
    assert "Tracker Hotspots" not in output
    assert "Session Comparison" not in output


def test_gaze_benchmark_report_cli_text_top_limit(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    logs_dir = runtime_dir / "logs"
    logs_dir.mkdir(parents=True)
    first_summary = logs_dir / "gaze-benchmark.summary.json"
    second_summary = logs_dir / "gaze-benchmark-2.summary.json"
    first_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:00:00Z",
                "session_log": str(logs_dir / "gaze-benchmark.jsonl"),
                "summary_path": str(first_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 95,
                    "missing_readings": 5,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.90,
                    "fused_jitter_x": 0.020,
                    "fused_jitter_y": 0.030,
                    "fused_drift_distance": 0.040,
                    "raw_fused_mean_distance": 0.025,
                    "x_axis_confidence_mean": 0.82,
                    "y_axis_confidence_mean": 0.60,
                    "head_yaw_abs_mean": 0.02,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 0,
                },
            },
            indent=2,
        )
        + "\n"
    )
    second_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:05:00Z",
                "session_log": str(logs_dir / "gaze-benchmark-2.jsonl"),
                "summary_path": str(second_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 99,
                    "missing_readings": 1,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.93,
                    "fused_jitter_x": 0.010,
                    "fused_jitter_y": 0.020,
                    "fused_drift_distance": 0.030,
                    "raw_fused_mean_distance": 0.015,
                    "x_axis_confidence_mean": 0.88,
                    "y_axis_confidence_mean": 0.72,
                    "head_yaw_abs_mean": 0.01,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-benchmark-report",
                "--runtime-dir",
                str(runtime_dir),
                "--limit",
                "5",
                "--format",
                "text",
                "--top",
                "1",
            ]
        )
    output = stdout.getvalue()

    assert exit_code == 0
    assert "Tracker Hotspots" in output
    assert "quality: best mediapipe" in output
    assert "jitter: best mediapipe" not in output
    assert "Metric Leaderboards" in output
    assert "quality: best gaze-benchmark-2.jsonl" in output
    assert "jitter: best gaze-benchmark-2.jsonl" not in output
    assert "Recent Changes" in output
    assert "quality improved" in output
    assert "dropout improved" not in output


def test_gaze_benchmark_report_cli_json_output_file(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    logs_dir = runtime_dir / "logs"
    logs_dir.mkdir(parents=True)
    summary_path = logs_dir / "gaze-benchmark.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:00:00Z",
                "session_log": str(logs_dir / "gaze-benchmark.jsonl"),
                "summary_path": str(summary_path),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 99,
                    "missing_readings": 1,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.94,
                    "fused_jitter_x": 0.010,
                    "fused_jitter_y": 0.015,
                    "fused_drift_distance": 0.020,
                    "raw_fused_mean_distance": 0.015,
                    "x_axis_confidence_mean": 0.88,
                    "y_axis_confidence_mean": 0.77,
                    "head_yaw_abs_mean": 0.01,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )
    output_path = tmp_path / "benchmark-report.json"

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-benchmark-report",
                "--runtime-dir",
                str(runtime_dir),
                "--limit",
                "5",
                "--output",
                str(output_path),
            ]
        )

    assert exit_code == 0
    assert stdout.getvalue() == ""
    payload = json.loads(output_path.read_text())
    assert payload["benchmark_report"]["status"] == "success"
    assert payload["benchmark_report"]["run_count"] == 1
    assert payload["benchmark_report"]["rendering"] == {
        "preset": None,
        "format": "json",
        "brief": False,
        "top": None,
        "output": str(output_path),
    }


def test_gaze_benchmark_report_cli_text_output_file(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    logs_dir = runtime_dir / "logs"
    logs_dir.mkdir(parents=True)
    summary_path = logs_dir / "gaze-benchmark.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:00:00Z",
                "session_log": str(logs_dir / "gaze-benchmark.jsonl"),
                "summary_path": str(summary_path),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 99,
                    "missing_readings": 1,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.94,
                    "fused_jitter_x": 0.010,
                    "fused_jitter_y": 0.015,
                    "fused_drift_distance": 0.020,
                    "raw_fused_mean_distance": 0.015,
                    "x_axis_confidence_mean": 0.88,
                    "y_axis_confidence_mean": 0.77,
                    "head_yaw_abs_mean": 0.01,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )
    output_path = tmp_path / "benchmark-report.txt"

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-benchmark-report",
                "--runtime-dir",
                str(runtime_dir),
                "--limit",
                "5",
                "--format",
                "text",
                "--brief",
                "--output",
                str(output_path),
            ]
        )
    output = output_path.read_text()

    assert exit_code == 0
    assert stdout.getvalue() == ""
    assert "Gaze Benchmark Report" in output
    assert "Latest" in output
    assert "Bottleneck" in output
    assert "Leaders" in output
    payload = _run_gaze_benchmark_report(
        argparse.Namespace(
            runtime_dir=str(runtime_dir),
            summary_path=None,
            tracker=None,
            limit=5,
            format="text",
            brief=True,
            output=str(output_path),
            top=None,
        )
    )
    assert payload["benchmark_report"]["rendering"] == {
        "preset": None,
        "format": "text",
        "brief": True,
        "top": None,
        "output": str(output_path),
    }


def test_gaze_benchmark_report_cli_text_brief_top_limit(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    logs_dir = runtime_dir / "logs"
    logs_dir.mkdir(parents=True)
    first_summary = logs_dir / "gaze-benchmark.summary.json"
    second_summary = logs_dir / "gaze-benchmark-2.summary.json"
    first_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:00:00Z",
                "session_log": str(logs_dir / "gaze-benchmark.jsonl"),
                "summary_path": str(first_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 95,
                    "missing_readings": 5,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.90,
                    "fused_jitter_x": 0.020,
                    "fused_jitter_y": 0.030,
                    "fused_drift_distance": 0.040,
                    "raw_fused_mean_distance": 0.025,
                    "x_axis_confidence_mean": 0.82,
                    "y_axis_confidence_mean": 0.60,
                    "head_yaw_abs_mean": 0.02,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 0,
                },
            },
            indent=2,
        )
        + "\n"
    )
    second_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:05:00Z",
                "session_log": str(logs_dir / "gaze-benchmark-2.jsonl"),
                "summary_path": str(second_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 99,
                    "missing_readings": 1,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.93,
                    "fused_jitter_x": 0.010,
                    "fused_jitter_y": 0.020,
                    "fused_drift_distance": 0.030,
                    "raw_fused_mean_distance": 0.015,
                    "x_axis_confidence_mean": 0.88,
                    "y_axis_confidence_mean": 0.72,
                    "head_yaw_abs_mean": 0.01,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-benchmark-report",
                "--runtime-dir",
                str(runtime_dir),
                "--limit",
                "5",
                "--format",
                "text",
                "--brief",
                "--top",
                "1",
            ]
        )
    output = stdout.getvalue()

    assert exit_code == 0
    assert "Leaders" in output
    assert "quality: best gaze-benchmark-2.jsonl" in output
    assert "jitter: best gaze-benchmark-2.jsonl" not in output
    assert "Recent" in output
    assert "improvements quality improved" in output
    assert "dropout improved" not in output


def test_gaze_benchmark_report_cli_preset_brief_text(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    logs_dir = runtime_dir / "logs"
    logs_dir.mkdir(parents=True)
    first_summary = logs_dir / "gaze-benchmark.summary.json"
    second_summary = logs_dir / "gaze-benchmark-2.summary.json"
    first_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:00:00Z",
                "session_log": str(logs_dir / "gaze-benchmark.jsonl"),
                "summary_path": str(first_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 95,
                    "missing_readings": 5,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.90,
                    "fused_jitter_x": 0.020,
                    "fused_jitter_y": 0.030,
                    "fused_drift_distance": 0.040,
                    "raw_fused_mean_distance": 0.025,
                    "x_axis_confidence_mean": 0.82,
                    "y_axis_confidence_mean": 0.60,
                    "head_yaw_abs_mean": 0.02,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 0,
                },
            },
            indent=2,
        )
        + "\n"
    )
    second_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:05:00Z",
                "session_log": str(logs_dir / "gaze-benchmark-2.jsonl"),
                "summary_path": str(second_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 99,
                    "missing_readings": 1,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.93,
                    "fused_jitter_x": 0.010,
                    "fused_jitter_y": 0.020,
                    "fused_drift_distance": 0.030,
                    "raw_fused_mean_distance": 0.015,
                    "x_axis_confidence_mean": 0.88,
                    "y_axis_confidence_mean": 0.72,
                    "head_yaw_abs_mean": 0.01,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-benchmark-report",
                "--runtime-dir",
                str(runtime_dir),
                "--limit",
                "5",
                "--preset",
                "brief-text",
            ]
        )
    output = stdout.getvalue()
    payload = _run_gaze_benchmark_report(
        argparse.Namespace(
            runtime_dir=str(runtime_dir),
            summary_path=None,
            tracker=None,
            limit=5,
            preset="brief-text",
            format="json",
            brief=False,
            output=None,
            top=None,
        )
    )

    assert exit_code == 0
    assert "Gaze Benchmark Report" in output
    assert "Latest" in output
    assert "Leaders" in output
    assert "Metric Leaderboards" not in output
    assert payload["benchmark_report"]["rendering"] == {
        "preset": "brief-text",
        "format": "text",
        "brief": True,
        "top": None,
        "output": None,
    }


def test_gaze_benchmark_brief_cli_alias(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    logs_dir = runtime_dir / "logs"
    logs_dir.mkdir(parents=True)
    first_summary = logs_dir / "gaze-benchmark.summary.json"
    second_summary = logs_dir / "gaze-benchmark-2.summary.json"
    first_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:00:00Z",
                "session_log": str(logs_dir / "gaze-benchmark.jsonl"),
                "summary_path": str(first_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 95,
                    "missing_readings": 5,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.90,
                    "fused_jitter_x": 0.020,
                    "fused_jitter_y": 0.030,
                    "fused_drift_distance": 0.040,
                    "raw_fused_mean_distance": 0.025,
                    "x_axis_confidence_mean": 0.82,
                    "y_axis_confidence_mean": 0.60,
                    "head_yaw_abs_mean": 0.02,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 0,
                },
            },
            indent=2,
        )
        + "\n"
    )
    second_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:05:00Z",
                "session_log": str(logs_dir / "gaze-benchmark-2.jsonl"),
                "summary_path": str(second_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 99,
                    "missing_readings": 1,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.93,
                    "fused_jitter_x": 0.010,
                    "fused_jitter_y": 0.020,
                    "fused_drift_distance": 0.030,
                    "raw_fused_mean_distance": 0.015,
                    "x_axis_confidence_mean": 0.88,
                    "y_axis_confidence_mean": 0.72,
                    "head_yaw_abs_mean": 0.01,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-benchmark-brief", "--runtime-dir", str(runtime_dir), "--limit", "5"])
    output = stdout.getvalue()
    payload = _run_gaze_benchmark_report(
        argparse.Namespace(
            runtime_dir=str(runtime_dir),
            summary_path=None,
            tracker=None,
            limit=5,
            preset="brief-text",
            format="text",
            brief=True,
            output=None,
            top=None,
        )
    )

    assert exit_code == 0
    assert "Gaze Benchmark Report" in output
    assert "Latest" in output
    assert "Leaders" in output
    assert "Metric Leaderboards" not in output
    assert payload["benchmark_report"]["rendering"]["preset"] == "brief-text"


def test_gaze_benchmark_report_cli_group_hotspots_rank_multiple_trackers(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    logs_dir = runtime_dir / "logs"
    logs_dir.mkdir(parents=True)
    mediapipe_summary = logs_dir / "gaze-benchmark.summary.json"
    opencv_summary = logs_dir / "gaze-benchmark-opencv.summary.json"
    mediapipe_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:00:00Z",
                "session_log": str(logs_dir / "gaze-benchmark.jsonl"),
                "summary_path": str(mediapipe_summary),
                "tracker": "mediapipe",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 99,
                    "missing_readings": 1,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.94,
                    "fused_jitter_x": 0.010,
                    "fused_jitter_y": 0.015,
                    "fused_drift_distance": 0.020,
                    "raw_fused_mean_distance": 0.015,
                    "x_axis_confidence_mean": 0.88,
                    "y_axis_confidence_mean": 0.77,
                    "head_yaw_abs_mean": 0.01,
                    "head_pitch_abs_mean": 0.01,
                    "head_roll_abs_mean": 0.0,
                    "left_wink_episodes": 1,
                    "right_wink_episodes": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )
    opencv_summary.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-04-02T00:05:00Z",
                "session_log": str(logs_dir / "gaze-benchmark-opencv.jsonl"),
                "summary_path": str(opencv_summary),
                "tracker": "opencv",
                "benchmark": {
                    "status": "success",
                    "attempted_frames": 100,
                    "successful_readings": 90,
                    "missing_readings": 10,
                    "sample_rate_hz": 30.0,
                    "mean_confidence": 0.78,
                    "fused_jitter_x": 0.030,
                    "fused_jitter_y": 0.040,
                    "fused_drift_distance": 0.050,
                    "raw_fused_mean_distance": 0.040,
                    "x_axis_confidence_mean": 0.70,
                    "y_axis_confidence_mean": 0.48,
                    "head_yaw_abs_mean": 0.02,
                    "head_pitch_abs_mean": 0.02,
                    "head_roll_abs_mean": 0.01,
                    "left_wink_episodes": 0,
                    "right_wink_episodes": 0,
                },
            },
            indent=2,
        )
        + "\n"
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "gaze-benchmark-report",
                "--summary-path",
                str(mediapipe_summary),
                "--summary-path",
                str(opencv_summary),
            ]
        )
    payload = json.loads(stdout.getvalue())

    tracker_hotspots = payload["benchmark_report"]["group_metric_hotspots"]["tracker"]["metrics"]

    assert exit_code == 0
    assert tracker_hotspots["mean_quality_score"]["best_group"]["group"] == "mediapipe"
    assert tracker_hotspots["mean_quality_score"]["weakest_group"]["group"] == "opencv"
    assert tracker_hotspots["mean_fused_jitter_distance"]["best_group"]["group"] == "mediapipe"
    assert tracker_hotspots["mean_fused_jitter_distance"]["weakest_group"]["group"] == "opencv"
    assert tracker_hotspots["mean_y_axis_confidence"]["best_group"]["group"] == "mediapipe"
    assert tracker_hotspots["mean_y_axis_confidence"]["weakest_group"]["group"] == "opencv"
    leaderboards = payload["benchmark_report"]["metric_leaderboards"]["metrics"]
    assert leaderboards["quality_score"]["best_run"]["session_log"].endswith("gaze-benchmark.jsonl")
    assert leaderboards["quality_score"]["weakest_run"]["session_log"].endswith("gaze-benchmark-opencv.jsonl")
    assert leaderboards["dropout_rate"]["best_run"]["session_log"].endswith("gaze-benchmark.jsonl")
    assert leaderboards["dropout_rate"]["weakest_run"]["session_log"].endswith("gaze-benchmark-opencv.jsonl")
