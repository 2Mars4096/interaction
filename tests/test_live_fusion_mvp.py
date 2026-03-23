import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from interaction.app import main
from interaction.audio import LiveSpeechCapture, SpeechCaptureError
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.vision import CalibrationProfile, GazeSample, NormalizedPoint, WebcamProviderError


class FakeFusionGazeProvider:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._points = [NormalizedPoint(0.15, 0.15)] * 8

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, *, delta_ms: int = 100):
        if not self._points:
            return None
        point = self._points.pop(0)
        return type(
            "Reading",
            (),
            {
                "sample": GazeSample(point=point, confidence=0.92, delta_ms=delta_ms),
                "frame_size": (640, 480),
                "face_bbox": (0, 0, 10, 10),
                "eye_boxes": (),
            },
        )()


class FakeNoReadingGazeProvider(FakeFusionGazeProvider):
    def read(self, *, delta_ms: int = 100):
        return None


class FakeFusionCameraErrorProvider:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index

    def open(self) -> None:
        raise WebcamProviderError(
            "camera_unavailable",
            "Unable to open webcam at index 0",
            payload={"camera_index": self.camera_index},
        )

    def close(self) -> None:
        return None


class FakeFusionSpeechProvider:
    def __init__(self, _paths) -> None:
        self.calls = 0

    def capture_turn(self, *, duration_s: float = 4.0, locale: str = "en-US") -> LiveSpeechCapture:
        self.calls += 1
        if self.calls == 1:
            return LiveSpeechCapture(
                transcript="open this",
                confidence=0.91,
                locale=locale,
                duration_s=duration_s,
                provider="fake",
                used_on_device=True,
                permission_state="granted",
            )
        if self.calls == 2:
            return LiveSpeechCapture(
                transcript="yes",
                confidence=0.95,
                locale=locale,
                duration_s=duration_s,
                provider="fake",
                used_on_device=True,
                permission_state="granted",
            )
        raise AssertionError("Unexpected extra speech capture turn.")


class FakeFusionSpeechErrorProvider:
    def __init__(self, _paths) -> None:
        pass

    def capture_turn(self, *, duration_s: float = 4.0, locale: str = "en-US") -> LiveSpeechCapture:
        raise SpeechCaptureError(
            "microphone_denied",
            "Microphone permission was denied for the speech helper.",
            payload={"status": "error", "error": "microphone_denied"},
        )


def test_fusion_live_requires_saved_calibration_profile(tmp_path: Path) -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["fusion-live", "--runtime-dir", str(tmp_path / ".interaction"), "--gaze-frames", "6"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze_context"]["error"] == "missing_calibration_profile"


def test_fusion_live_reports_camera_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeFusionCameraErrorProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["fusion-live", "--runtime-dir", str(tmp_path / ".interaction"), "--gaze-frames", "6"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze_context"]["status"] == "error"
    assert payload["live_gaze_context"]["error"] == "camera_unavailable"


def test_fusion_live_reports_missing_gaze_lock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeNoReadingGazeProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["fusion-live", "--runtime-dir", str(tmp_path / ".interaction"), "--gaze-frames", "6"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze_context"]["status"] == "error"
    assert payload["live_gaze_context"]["error"] == "missing_gaze_lock"


def test_fusion_live_reports_speech_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeFusionGazeProvider)
    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeFusionSpeechErrorProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["fusion-live", "--runtime-dir", str(tmp_path / ".interaction"), "--gaze-frames", "6", "--duration", "1.0"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze_context"]["status"] == "success"
    assert payload["live_capture"]["status"] == "error"
    assert payload["live_capture"]["error"] == "microphone_denied"


def test_fusion_live_runs_confirmation_required_flow(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", FakeFusionGazeProvider)
    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeFusionSpeechProvider)
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "fusion-live",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--gaze-frames",
                "6",
                "--duration",
                "1.0",
                "--confirm-duration",
                "0.5",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_gaze_context"]["status"] == "success"
    assert payload["live_gaze_context"]["target"]["target_id"] == "upper_left"
    assert payload["live_capture"]["status"] == "success"
    assert payload["confirmation_capture"]["status"] == "success"
    assert payload["confirmation_capture"]["transcript"] == "yes"
    assert payload["metrics"]["confirmations"] >= 1
    assert payload["metrics"]["executions"] >= 1
    assert any(
        event["result"]
        and event["result"]["details"]["commands"][0][2] == "interaction.platform.macos_runtime"
        and event["result"]["details"]["commands"][0][3] == "double-click-normalized"
        for event in payload["events"]
    )
