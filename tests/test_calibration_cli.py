"""Exercise visible calibration, independent validation, and save transactions."""

from __future__ import annotations

import json

import pytest

from interaction.app import main
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.runtime.webcam import default_webcam_calibration_targets, default_webcam_validation_targets
from interaction.ui.calibration import CalibrationCancelled, CalibrationUIError
from interaction.vision import GazeSample, NormalizedPoint, WebcamProviderError, WebcamSampleAggregate


class CalibrationProvider:
    camera_index = 0

    def __init__(self, screen, *, failure=None):
        self.screen = screen
        self.failure = failure
        self.targets = (*default_webcam_calibration_targets(), *default_webcam_validation_targets())
        self.constructed = False
        self.opened = False
        self.close_calls = 0
        self.capture_calls = 0
        self.settled_targets = []

    def open(self):
        assert self.screen.opened
        self.opened = True
        if self.failure == "provider_open":
            raise WebcamProviderError("camera_unavailable", "Camera could not be opened.")

    def close(self):
        self.close_calls += 1
        if self.failure == "provider_cleanup":
            raise RuntimeError("Camera cleanup failed.")

    def capture_average_sample(self, *, required_frames, max_attempts, delta_ms, on_frame):
        assert self.opened and self.close_calls == 0
        assert required_frames == 6 and delta_ms == 100 and max_attempts >= required_frames
        index = self.capture_calls
        target = self.targets[index]
        # The real orchestration must show and settle this reference point before
        # asking for its samples, including the independent validation points.
        assert len(self.screen.targets) == index + 1
        assert len(self.settled_targets) == index + 1
        x, y, label, phase = self.screen.targets[-1]
        assert (x, y, phase) == (target.expected.x, target.expected.y, target.phase)
        assert f"{index + 1}/9" in label
        self.capture_calls += 1
        on_frame()
        if self.failure == "ui_late_cancel" and index == len(self.targets) - 1:
            # A close/Escape event can arrive during the final camera read,
            # after its last event pump. The CLI must pump again before saving.
            def cancel_on_next_pump():
                raise CalibrationCancelled()

            self.screen.pump = cancel_on_next_pump
        if self.failure == "provider_capture":
            raise WebcamProviderError("camera_read_failed", "Camera stopped delivering frames.")
        point = target.expected
        if self.failure == "validation" and target.phase == "validation":
            point = NormalizedPoint(point.x + 0.15, point.y)
        used_frames = 1 if self.failure == "partial_samples" and index == 0 else 6
        return WebcamSampleAggregate(
            GazeSample(point, confidence=0.95),
            used_frames=used_frames,
            attempted_frames=6,
            raw_stddev=NormalizedPoint(0.001, 0.001),
            sample_span_ms=500,
        )


def configure_session(monkeypatch, screen, *, failure=None):
    provider = CalibrationProvider(screen, failure=failure)

    def make_provider(*, camera_index):
        assert camera_index == provider.camera_index
        provider.constructed = True
        return provider

    def wait(ms):
        assert ms == 30
        provider.settled_targets.append(screen.targets[-1])
        if failure == "ui_cancel" and len(screen.targets) == 2:
            raise CalibrationCancelled()
        screen.pump()

    original_close = screen.close
    screen.close_calls = 0

    def close():
        screen.close_calls += 1
        original_close()
        if failure == "ui_cleanup":
            raise RuntimeError("Screen cleanup failed.")

    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", make_provider)
    monkeypatch.setattr(screen, "wait", wait)
    monkeypatch.setattr(screen, "close", close)
    if failure == "ui_open":
        def fail_open():
            raise CalibrationUIError("calibration_ui_unavailable", "A desktop display is required.")

        monkeypatch.setattr(screen, "open", fail_open)
    return provider


def run_calibration(runtime_dir, capsys):
    exit_code = main([
        "gaze-calibrate", "--runtime-dir", str(runtime_dir),
        "--settle-ms", "30", "--frames-per-step", "6", "--delta-ms", "100",
    ])
    output = capsys.readouterr()
    # One machine-readable JSON result remains available even after an error;
    # immediate user feedback goes to stderr without corrupting that result.
    payload = json.loads(output.out)
    assert exit_code == 0
    return payload


def test_cli_shows_each_point_before_sampling_and_saves_validated_quality(
    monkeypatch, tmp_path, capsys, fake_calibration_screen,
):
    screen = fake_calibration_screen
    provider = configure_session(monkeypatch, screen)
    runtime_dir = tmp_path / "runtime"
    payload = run_calibration(runtime_dir, capsys)

    assert payload["calibration"] == {"status": "success", "display_size": [1920, 1080]}
    assert provider.capture_calls == 9
    assert screen.closed and screen.close_calls == 1
    assert provider.close_calls == 1
    steps = payload["calibration_steps"]
    fitting = {(step["expected"]["x"], step["expected"]["y"]) for step in steps if step["phase"] == "fitting"}
    validation = {(step["expected"]["x"], step["expected"]["y"]) for step in steps if step["phase"] == "validation"}
    assert len(fitting) == 5 and len(validation) == 4 and fitting.isdisjoint(validation)
    assert all(step["success"] and step["used_frames"] == 6 for step in steps)
    profile_payload = payload["calibration_profile"]
    assert profile_payload["schema_version"] == 2
    assert profile_payload["camera_index"] == 0
    assert profile_payload["quality"]["point_count"] == 4
    assert profile_payload["quality"]["sample_count"] == 24
    assert profile_payload["quality"]["mean_error"] == pytest.approx(0)
    saved = JsonStateStore(RuntimePaths(runtime_dir)).load_calibration_profile(
        "webcam-live", require_validated=True, camera_index=0,
    )
    assert saved is not None and saved.is_usable(camera_index=0)
    assert saved.quality.mean_error == pytest.approx(0)


@pytest.mark.parametrize("existing_profile", [False, True])
@pytest.mark.parametrize(
    "failure,expected_error",
    [
        ("ui_open", "calibration_ui_unavailable"),
        ("ui_cancel", "calibration_cancelled"),
        ("ui_late_cancel", "calibration_cancelled"),
        ("partial_samples", "insufficient_calibration_samples"),
        ("validation", "calibration_validation_failed"),
        ("provider_open", "camera_unavailable"),
        ("provider_capture", "camera_read_failed"),
        ("provider_cleanup", "calibration_cleanup_failed"),
        ("ui_cleanup", "calibration_cleanup_failed"),
    ],
)
def test_failed_cli_session_cleans_resources_and_cannot_replace_saved_profile(
    monkeypatch, tmp_path, capsys, fake_calibration_screen, validated_profile,
    existing_profile, failure, expected_error,
):
    screen = fake_calibration_screen
    provider = configure_session(monkeypatch, screen, failure=failure)
    runtime_dir = tmp_path / "runtime"
    store = JsonStateStore(RuntimePaths(runtime_dir))
    profile_path = runtime_dir / "profiles" / "webcam-live.json"
    if existing_profile:
        assert store.save_calibration_profile(validated_profile, name="webcam-live") == profile_path
        previous_bytes = profile_path.read_bytes()

    payload = run_calibration(runtime_dir, capsys)

    assert payload["calibration"]["status"] == "error"
    assert payload["calibration"]["error"] == expected_error
    assert payload["calibration"]["message"]
    assert payload["calibration_profile"] is None
    assert screen.closed and screen.close_calls == 1
    if failure == "ui_open":
        assert not provider.constructed
        assert provider.close_calls == 0 and provider.capture_calls == 0
    else:
        assert provider.constructed and provider.close_calls == 1
    if failure in {"validation", "provider_cleanup", "ui_cleanup", "ui_late_cancel"}:
        assert provider.capture_calls == 9
    if existing_profile:
        assert profile_path.read_bytes() == previous_bytes
    else:
        assert not profile_path.exists()
