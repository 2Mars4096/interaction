"""Exercise live speech/gaze coordination without camera, microphone, or OS actions."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from interaction.app import main
from interaction.audio import LiveSpeechCapture
from interaction.contracts import EnvironmentSnapshot
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.platform import MacOSPlatformAdapter
from interaction.runtime.live_gaze import LiveGazeSampler
from interaction.vision import CalibrationProfile, GazeSample, NormalizedPoint


class ControlledCamera:
    def __init__(self):
        self.drop_next = Event()
        self.tracking_lost = Event()
        self.closed = Event()
        self.owner = None

    def open(self):
        self.owner = get_ident()

    def read(self, *, delta_ms=100):
        assert self.owner == get_ident()
        if self.tracking_lost.is_set():
            return None
        if self.drop_next.is_set():
            self.drop_next.clear()
            return None
        return SimpleNamespace(
            sample=GazeSample(NormalizedPoint(0.15, 0.15), confidence=0.95, delta_ms=delta_ms)
        )

    def close(self):
        assert self.owner == get_ident()
        self.closed.set()


def run_live_session(
    monkeypatch, tmp_path, *, lose_during_confirmation=False,
    keep_tracking_lost=False, change_context_turn=None,
):
    camera = ControlledCamera()
    samplers = []
    capture_progress = []
    reacquired_targets = []
    environment = [EnvironmentSnapshot(active_app="Safari", active_window_title="Example page")]

    def sampler_factory(provider, profile, **kwargs):
        sampler = LiveGazeSampler(provider, profile, **kwargs)
        samplers.append(sampler)
        return sampler

    def dry_run_adapter(*, dry_run):
        assert dry_run, "Live context tests must never execute OS actions."
        return MacOSPlatformAdapter(dry_run=True)

    class BlockingSpeech:
        def __init__(self, _paths):
            self.calls = 0

        def capture_turn(self, *, duration_s=4.0, locale="en-US"):
            self.calls += 1
            assert self.calls <= 2
            sampler = samplers[0]
            with sampler._condition:
                initial_frames = sampler.attempted_frames
                # This capture is blocked on the main thread. Only the real
                # camera worker can make the predicate true and wake it.
                assert sampler._condition.wait_for(
                    lambda: sampler.attempted_frames >= initial_frames + 3, timeout=2.0
                ), "Gaze sampling stopped while speech capture was waiting."
                capture_progress.append((initial_frames, sampler.attempted_frames))

                if self.calls == 2 and lose_during_confirmation:
                    original = sampler._current()
                    assert original is not None
                    missing_before = sampler.missing_readings
                    if keep_tracking_lost:
                        camera.tracking_lost.set()
                    else:
                        camera.drop_next.set()
                    assert sampler._condition.wait_for(
                        lambda: sampler.missing_readings > missing_before, timeout=2.0
                    ), "The worker did not observe the injected tracking loss."
                    if not keep_tracking_lost:
                        assert sampler._condition.wait_for(
                            lambda: sampler._current() is not None
                            and sampler._current().revision > original.revision,
                            timeout=2.0,
                        ), "The worker did not reacquire a stable target after loss."
                        reacquired = sampler._current()
                        reacquired_targets.append((original, reacquired))

            if self.calls == change_context_turn:
                environment[0] = EnvironmentSnapshot(active_app="Notes", active_window_title="Untitled")
            return LiveSpeechCapture(
                transcript="click" if self.calls == 1 else "yes",
                confidence=0.95,
                locale=locale,
                duration_s=duration_s,
                provider="fake",
                used_on_device=True,
                permission_state="granted",
            )

    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", lambda **kwargs: camera)
    monkeypatch.setattr("interaction.app.LiveGazeSampler", sampler_factory)
    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", BlockingSpeech)
    monkeypatch.setattr("interaction.app.MacOSPlatformAdapter", dry_run_adapter)
    monkeypatch.setattr("interaction.app.capture_environment", lambda: environment[0].model_copy())
    store = JsonStateStore(RuntimePaths(tmp_path))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(io.StringIO()):
        exit_code = main([
            "fusion-live", "--runtime-dir", str(tmp_path),
            "--gaze-frames", "40", "--gaze-delta-ms", "10",
        ])

    assert exit_code == 0
    assert camera.owner != get_ident()
    assert camera.closed.is_set()
    assert not samplers[0]._thread.is_alive()
    return json.loads(output.getvalue()), capture_progress, reacquired_targets


def test_camera_continues_during_command_and_confirmation_capture(monkeypatch, tmp_path):
    payload, progress, _ = run_live_session(monkeypatch, tmp_path)

    assert len(progress) == 2
    assert all(after >= before + 3 for before, after in progress)
    assert payload["live_capture"]["status"] == "success"
    assert payload["confirmation_capture"]["status"] == "success"
    assert payload["metrics"]["executions"] == 1
    results = [event["result"] for event in payload["events"] if event["result"]]
    assert len(results) == 1
    assert results[0]["details"]["dry_run"] is True
    assert results[0]["details"]["commands"][0][3] == "click-normalized"


def test_loss_and_reacquisition_during_confirmation_cannot_restore_pending_click(monkeypatch, tmp_path):
    payload, progress, reacquired_targets = run_live_session(
        monkeypatch, tmp_path, lose_during_confirmation=True
    )

    assert len(progress) == 2
    original, reacquired = reacquired_targets[0]
    assert original.target.target_id == reacquired.target.target_id == "upper_left"
    assert reacquired.revision > original.revision
    assert payload["confirmation_capture"]["transcript"] == "yes"
    assert payload["metrics"]["confirmations"] == 1
    assert payload["metrics"]["cancellations"] == 1
    assert payload["metrics"]["executions"] == 0
    assert all(event["result"] is None for event in payload["events"])
    assert any("tracking was lost" in (event["message"] or "") for event in payload["events"])


def test_tracking_loss_during_confirmation_removes_old_target_from_final_payload(monkeypatch, tmp_path):
    payload, progress, _ = run_live_session(
        monkeypatch, tmp_path, lose_during_confirmation=True, keep_tracking_lost=True
    )

    assert len(progress) == 2
    assert payload["confirmation_capture"]["transcript"] == "yes"
    assert payload["metrics"]["executions"] == 0
    assert all(event["result"] is None for event in payload["events"])
    context = payload["live_gaze_context"]
    assert context["status"] == "unavailable"
    assert context["error"] == "gaze_tracking_lost"
    assert "target" not in context
    assert "observation" not in context


@pytest.mark.parametrize("capture_turn", [1, 2])
def test_foreground_context_change_during_capture_cancels_action(monkeypatch, tmp_path, capture_turn):
    payload, progress, _ = run_live_session(monkeypatch, tmp_path, change_context_turn=capture_turn)

    assert len(progress) == capture_turn
    capture_key = "live_capture" if capture_turn == 1 else "confirmation_capture"
    assert payload[capture_key]["status"] == "error"
    assert payload[capture_key]["error"] == "context_changed"
    assert payload["metrics"]["executions"] == 0
    assert all(event["result"] is None for event in payload["events"])
