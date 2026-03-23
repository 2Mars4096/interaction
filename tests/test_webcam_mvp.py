import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from interaction.app import _append_calibration_prompt, main
from interaction.contracts import GazeObservation
from interaction.platform import MacOSPlatformAdapter
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.runtime import GazeTrackingLoop
from interaction.runtime.webcam import (
    CursorFollowController,
    WebcamCalibrationTarget,
    _compress_point_around_anchor,
    default_webcam_targets,
    generate_single_screen_calibration_targets,
    generate_single_screen_validation_targets,
    run_live_cursor_follow,
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
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.78, 0.18), confidence=0.82, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.5, 0.5), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.22, 0.79), confidence=0.81, delta_ms=100), used_frames=3, attempted_frames=3),
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
        self._samples = [None, None, WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.5, 0.5), confidence=0.9, delta_ms=100), used_frames=1, attempted_frames=3), None, None]


class FakeCompressedCalibrationProvider(FakeCalibrationProvider):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(camera_index)
        self._samples = [
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.47, 0.49), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.48, 0.52), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.54, 0.51), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
            WebcamSampleAggregate(sample=GazeSample(point=NormalizedPoint(0.54, 0.50), confidence=0.9, delta_ms=100), used_frames=3, attempted_frames=3),
        ]


class FakeLiveProvider:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
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
        return type(
            "Reading",
            (),
            {
                "sample": GazeSample(point=point, confidence=0.9, delta_ms=delta_ms),
                "frame_size": (640, 480),
                "face_bbox": (0, 0, 10, 10),
                "eye_boxes": (),
            },
        )()


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

    def show_point(self, x_norm: float, y_norm: float) -> None:
        self.points.append((x_norm, y_norm))

    def hide(self) -> None:
        self.hidden_calls += 1

    def wait_for_confirmation(self) -> None:
        self.wait_calls += 1


class FakeInteractiveValidationVisualizer(FakePointVisualizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.validation_points: list[tuple[float, float, float | None, float | None]] = []
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
        step_index: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        self.validation_points.append((target_x, target_y, tracked_x, tracked_y))

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


def test_gaze_calibrate_cli_saves_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeCalibrationProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-calibrate", "--runtime-dir", str(tmp_path / ".interaction"), "--settle-ms", "0", "--frames-per-step", "3"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["calibration"]["status"] == "success"
    assert len(payload["calibration_targets"]) == 4
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


def test_generate_single_screen_calibration_targets_covers_full_display() -> None:
    targets = generate_single_screen_calibration_targets(seed=7)

    assert len(targets) == 4
    xs = [target.expected.x for target in targets]
    ys = [target.expected.y for target in targets]
    labels = {target.label for target in targets}

    assert labels == {"upper left", "upper right", "lower left", "lower right"}
    assert min(xs) < 0.3
    assert max(xs) > 0.7
    assert min(ys) < 0.3
    assert max(ys) > 0.7


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
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8"])
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
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8"])
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
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8", "--action", "click"])
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
