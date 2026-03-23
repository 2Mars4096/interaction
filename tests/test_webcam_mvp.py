import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path

from interaction.app import main
from interaction.contracts import GazeObservation
from interaction.platform import MacOSPlatformAdapter
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.runtime import GazeTrackingLoop
from interaction.runtime.webcam import CursorFollowController, default_webcam_targets, run_live_cursor_follow, run_live_webcam_trace
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

    def show_point(self, x_norm: float, y_norm: float) -> None:
        self.points.append((x_norm, y_norm))

    def hide(self) -> None:
        self.hidden_calls += 1


def test_gaze_calibrate_cli_saves_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeCalibrationProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-calibrate", "--runtime-dir", str(tmp_path / ".interaction"), "--settle-ms", "0", "--frames-per-step", "3"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["calibration"]["status"] == "success"
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


def test_gaze_live_requires_saved_calibration_profile(tmp_path: Path) -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["gaze-live", "--runtime-dir", str(tmp_path / ".interaction"), "--frames", "8"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze"]["error"] == "missing_calibration_profile"


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
    assert payload["live_gaze"]["held_events"] >= 1
    assert any(
        event["result"]
        and event["result"]["details"]["commands"][0][2] == "interaction.platform.macos_runtime"
        and event["result"]["details"]["commands"][0][3] == "move-normalized"
        for event in payload["events"]
    )
    assert any(event["message"] == "Live gaze cursor hold is within the movement threshold." for event in payload["events"])


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
