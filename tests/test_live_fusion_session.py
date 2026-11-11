"""Repeated fusion sessions with real worker coordination and fake OS inputs."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace

import pytest

from interaction.app import main
from interaction.audio import LiveSpeechCapture, SpeechCaptureError
from interaction.contracts import EnvironmentSnapshot
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.platform import MacOSPlatformAdapter
from interaction.runtime.live_gaze import LiveGazeSampler
from interaction.session import SessionReplay
from interaction.vision import GazeSample, NormalizedPoint, WebcamProviderError


class SessionCamera:
    def __init__(self):
        self.open_count = 0
        self.close_count = 0
        self.owner = None
        self.lost = False
        self.read_error = None
        self.close_error = None

    def open(self):
        self.open_count += 1
        self.owner = get_ident()

    def read(self, *, delta_ms=100):
        assert self.owner == get_ident()
        if self.read_error is not None:
            raise self.read_error
        if self.lost:
            return None
        return SimpleNamespace(sample=GazeSample(NormalizedPoint(0.15, 0.15), confidence=0.95, delta_ms=delta_ms))

    def close(self):
        assert self.owner == get_ident()
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


def wait_for_worker(sampler, predicate):
    with sampler._condition:
        assert sampler._condition.wait_for(predicate, timeout=2), "Camera worker did not reach the expected state."


def run_session(monkeypatch, tmp_path, validated_profile, utterances, *options, on_capture=None, initial_environment=None):
    camera = SessionCamera()
    samplers = []
    captures = []
    environment = [initial_environment or EnvironmentSnapshot(active_app="Safari", active_window_title="Example")]
    speech = iter(utterances)
    feedback = io.StringIO()

    def sampler_factory(provider, profile, **kwargs):
        sampler = LiveGazeSampler(provider, profile, **kwargs)
        samplers.append(sampler)
        return sampler

    class Speech:
        def __init__(self, _paths):
            pass

        def capture_turn(self, *, duration_s, locale):
            sampler = samplers[0]
            previous_frames = sampler.attempted_frames
            wait_for_worker(sampler, lambda: sampler.attempted_frames > previous_frames)
            captures.append({"duration_s": duration_s, "feedback": feedback.getvalue()})
            if on_capture is not None:
                on_capture(len(captures), camera, sampler, environment)
            utterance = next(speech)
            if isinstance(utterance, BaseException):
                raise utterance
            if utterance is None:
                raise SpeechCaptureError("empty_transcript", "No speech heard.")
            text, confidence = utterance if isinstance(utterance, tuple) else (utterance, 0.95)
            return LiveSpeechCapture(text, confidence, locale, duration_s)

    monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", lambda **kwargs: camera)
    monkeypatch.setattr("interaction.app.LiveGazeSampler", sampler_factory)
    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", Speech)
    monkeypatch.setattr("interaction.app.capture_environment", lambda: environment[0].model_copy())
    # Even tests of --execute validation must never move the host pointer.
    monkeypatch.setattr("interaction.app.MacOSPlatformAdapter", lambda **kwargs: MacOSPlatformAdapter(dry_run=True))
    store = JsonStateStore(RuntimePaths(tmp_path))
    store.save_calibration_profile(validated_profile, name="webcam-live")
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(feedback):
        assert main([
            "fusion-live", "--runtime-dir", str(tmp_path), "--gaze-frames", "40", "--gaze-delta-ms", "10", *options,
        ]) == 0
    assert len(samplers) == 1
    assert camera.open_count == camera.close_count == 1
    assert camera.owner != get_ident()
    assert not samplers[0]._thread.is_alive()
    return json.loads(output.getvalue()), captures


def results(payload):
    return [event["result"] for event in payload["events"] if event["result"]]


def test_bounded_session_keeps_one_camera_for_successive_commands(monkeypatch, tmp_path, validated_profile):
    payload, captures = run_session(monkeypatch, tmp_path, validated_profile, ["open Safari", "scroll down"], "--turns", "2")
    assert len(captures) == 2
    assert payload["session"] == {"commands": 2, "stopped": False, "failed": False}
    assert [item["proposal"]["action"] for item in results(payload)] == ["open_app", "scroll"]
    assert payload["live_capture"] == payload["captures"][-1]
    assert payload["confirmation_capture"] is None
    assert len(SessionReplay.load(Path(payload["session_log"]))) == len(payload["events"])


def test_continuous_session_handles_confirmation_and_next_command(monkeypatch, tmp_path, validated_profile):
    payload, captures = run_session(
        monkeypatch, tmp_path, validated_profile, ["click", "yes", "scroll down", "Stop listening."], "--continuous",
    )
    assert len(captures) == 4
    assert "Listening" in captures[0]["feedback"]
    assert "Confirm clicking" in captures[1]["feedback"]
    assert len(payload["captures"]) == 3
    assert len(payload["confirmation_captures"]) == 1
    assert payload["confirmation_capture"] == payload["confirmation_captures"][-1]
    assert payload["session"] == {"commands": 3, "stopped": True, "failed": False}
    assert [item["proposal"]["action"] for item in results(payload)] == ["click_target", "scroll"]


def test_confirmation_silence_and_unclear_yes_retry_within_budget(monkeypatch, tmp_path, validated_profile):
    payload, captures = run_session(
        monkeypatch, tmp_path, validated_profile, ["click", None, ("yes", 0.2), "yes"], "--confirmation-attempts", "3",
    )
    assert len(captures) == 4
    assert len(payload["confirmation_captures"]) == 3
    assert len(results(payload)) == 1
    assert not payload["session"]["failed"]


def test_exhausted_confirmation_cancels_then_accepts_another_command(monkeypatch, tmp_path, validated_profile):
    payload, captures = run_session(
        monkeypatch, tmp_path, validated_profile, ["click", None, None, "scroll up"], "--turns", "2",
    )
    assert len(captures) == 4
    assert payload["metrics"]["cancellations"] == 1
    assert [item["proposal"]["action"] for item in results(payload)] == ["scroll"]
    assert any("no clear confirmation" in (event["message"] or "") for event in payload["events"])


def test_cancel_does_not_end_session_or_execute_pending_action(monkeypatch, tmp_path, validated_profile):
    payload, _ = run_session(
        monkeypatch, tmp_path, validated_profile, ["click", "cancel", "scroll up", "stop listening"], "--continuous",
    )
    assert payload["session"]["stopped"]
    assert payload["metrics"]["cancellations"] == 1
    assert [item["proposal"]["action"] for item in results(payload)] == ["scroll"]


@pytest.mark.parametrize("utterances, expected_action", [
    (["click", "scroll up"], "scroll"),
    (["click", "open Safari"], "open_app"),
    (["click", "type replacement text", "yes"], "type_text"),
])
def test_new_command_during_confirmation_supersedes_pending_action(monkeypatch, tmp_path, validated_profile, utterances, expected_action):
    payload, _ = run_session(monkeypatch, tmp_path, validated_profile, utterances)
    assert [item["proposal"]["action"] for item in results(payload)] == [expected_action]


def test_replacement_command_cannot_extend_confirmation_budget(monkeypatch, tmp_path, validated_profile):
    payload, captures = run_session(monkeypatch, tmp_path, validated_profile, ["click", "type replacement text", None])
    assert len(captures) == 3
    assert len(payload["confirmation_captures"]) == 2
    assert not results(payload)
    assert any("no clear confirmation" in (event["message"] or "") for event in payload["events"])


def test_stop_during_confirmation_cancels_without_another_capture(monkeypatch, tmp_path, validated_profile):
    payload, captures = run_session(monkeypatch, tmp_path, validated_profile, ["click", "stop listening"], "--continuous")
    assert len(captures) == 2
    assert payload["session"]["stopped"]
    assert payload["metrics"]["cancellations"] == 1
    assert not results(payload)


def test_continuous_silence_does_not_end_session(monkeypatch, tmp_path, validated_profile):
    payload, captures = run_session(monkeypatch, tmp_path, validated_profile, [None, "scroll down", "stop listening"], "--continuous")
    assert len(captures) == 3
    assert payload["captures"][0]["error"] == "empty_transcript"
    assert payload["session"] == {"commands": 2, "stopped": True, "failed": False}
    assert len(results(payload)) == 1


def test_bounded_session_silence_ends_without_unbounded_retries(monkeypatch, tmp_path, validated_profile):
    payload, captures = run_session(monkeypatch, tmp_path, validated_profile, [None], "--turns", "2")
    assert len(captures) == 1
    assert payload["session"] == {"commands": 0, "stopped": False, "failed": True}
    assert not results(payload)


def test_context_change_cancels_only_affected_command_in_continuous_mode(monkeypatch, tmp_path, validated_profile):
    def change_context(number, _camera, _sampler, environment):
        if number == 1:
            environment[0] = EnvironmentSnapshot(active_app="Notes", active_window_title="New note")

    payload, _ = run_session(
        monkeypatch, tmp_path, validated_profile, ["scroll down", "scroll up", "stop listening"], "--continuous", on_capture=change_context,
    )
    assert payload["captures"][0]["error"] == "context_changed"
    assert [item["proposal"]["arguments"]["direction"] for item in results(payload)] == ["up"]
    assert payload["session"]["stopped"] and not payload["session"]["failed"]


def test_context_change_during_confirmation_cannot_leak_pending_click(monkeypatch, tmp_path, validated_profile):
    def change_context(number, _camera, _sampler, environment):
        if number == 2:
            environment[0] = EnvironmentSnapshot(active_app="Notes")

    payload, _ = run_session(
        monkeypatch, tmp_path, validated_profile, ["click", "yes", "yes", "scroll up", "stop listening"], "--continuous", on_capture=change_context,
    )
    assert payload["confirmation_captures"][0]["error"] == "context_changed"
    assert payload["metrics"]["cancellations"] == 1
    assert [item["proposal"]["action"] for item in results(payload)] == ["scroll"]


def test_stop_is_honored_when_focus_changes_during_capture(monkeypatch, tmp_path, validated_profile):
    def change_context(_number, _camera, _sampler, environment):
        environment[0] = EnvironmentSnapshot(active_app="Notes")

    payload, captures = run_session(
        monkeypatch, tmp_path, validated_profile, ["stop listening"], "--continuous", on_capture=change_context,
    )
    assert len(captures) == 1
    assert payload["session"]["stopped"] and not payload["session"]["failed"]
    assert not results(payload)


def test_stop_is_honored_without_known_active_app_in_execute_mode(monkeypatch, tmp_path, validated_profile):
    payload, captures = run_session(
        monkeypatch, tmp_path, validated_profile, ["stop listening"], "--execute", "--continuous",
        initial_environment=EnvironmentSnapshot(),
    )
    assert len(captures) == 1
    assert payload["session"]["stopped"] and not payload["session"]["failed"]
    assert not results(payload)


def test_normal_action_without_known_active_app_is_rejected(monkeypatch, tmp_path, validated_profile):
    payload, _ = run_session(
        monkeypatch, tmp_path, validated_profile, ["scroll down"], "--execute", initial_environment=EnvironmentSnapshot(),
    )
    assert payload["live_capture"]["error"] == "context_unavailable"
    assert payload["session"]["failed"]
    assert not results(payload)


def test_gaze_loss_allows_voice_commands_and_recovery_clears_old_error(monkeypatch, tmp_path, validated_profile):
    def change_tracking(number, camera, sampler, _environment):
        if number == 1:
            missed = sampler.missing_readings
            camera.lost = True
            wait_for_worker(sampler, lambda: sampler.missing_readings > missed)
        elif number == 2:
            camera.lost = False
            wait_for_worker(sampler, lambda: sampler._current() is not None)

    payload, _ = run_session(
        monkeypatch, tmp_path, validated_profile, ["scroll down", "focus this"], "--turns", "2", on_capture=change_tracking,
    )
    assert [item["proposal"]["action"] for item in results(payload)] == ["scroll", "focus_target"]
    assert payload["live_gaze_context"]["status"] == "success"
    assert "error" not in payload["live_gaze_context"]
    assert "message" not in payload["live_gaze_context"]


@pytest.mark.parametrize("camera_failure", [False, True])
def test_stop_does_not_depend_on_gaze_after_capture(monkeypatch, tmp_path, validated_profile, camera_failure):
    def lose_tracking(_number, camera, sampler, _environment):
        if camera_failure:
            camera.read_error = WebcamProviderError("camera_disconnected", "Camera disconnected.")
            wait_for_worker(sampler, lambda: sampler._error is not None)
        else:
            missed = sampler.missing_readings
            camera.lost = True
            wait_for_worker(sampler, lambda: sampler.missing_readings > missed)

    payload, captures = run_session(
        monkeypatch, tmp_path, validated_profile, ["stop listening"], "--continuous", on_capture=lose_tracking,
    )
    assert len(captures) == 1
    assert payload["session"]["stopped"]
    assert not results(payload)


@pytest.mark.parametrize("utterances", [[KeyboardInterrupt()], ["click", KeyboardInterrupt()]])
def test_interrupt_in_command_or_confirmation_stops_camera_and_returns_json(monkeypatch, tmp_path, validated_profile, utterances):
    payload, _ = run_session(monkeypatch, tmp_path, validated_profile, utterances, "--continuous")
    assert payload["session"]["stopped"] and not payload["session"]["failed"]
    assert not results(payload)


def test_fatal_speech_error_stops_continuous_session(monkeypatch, tmp_path, validated_profile):
    payload, captures = run_session(
        monkeypatch, tmp_path, validated_profile, ["scroll down", SpeechCaptureError("microphone_denied", "Microphone denied.")], "--continuous",
    )
    assert len(captures) == 2
    assert payload["live_capture"]["error"] == "microphone_denied"
    assert payload["session"]["failed"]
    assert len(results(payload)) == 1


def test_camera_failure_after_first_command_is_fatal(monkeypatch, tmp_path, validated_profile):
    def fail_camera(number, camera, sampler, _environment):
        if number == 2:
            camera.read_error = WebcamProviderError("camera_disconnected", "Camera disconnected.")
            wait_for_worker(sampler, lambda: sampler._error is not None)

    payload, captures = run_session(
        monkeypatch, tmp_path, validated_profile, ["scroll down", "scroll up"], "--continuous", on_capture=fail_camera,
    )
    assert len(captures) == 2
    assert payload["live_gaze_context"]["error"] == "camera_disconnected"
    assert payload["session"]["failed"]
    assert len(results(payload)) == 1


def test_camera_close_error_is_reported_as_failure(monkeypatch, tmp_path, validated_profile):
    def fail_close(_number, camera, _sampler, _environment):
        camera.close_error = OSError("Camera cleanup failed.")

    payload, _ = run_session(monkeypatch, tmp_path, validated_profile, ["stop listening"], on_capture=fail_close)
    assert payload["live_gaze_context"]["error"] == "camera_close_failed"
    assert payload["session"]["stopped"] and payload["session"]["failed"]


@pytest.mark.parametrize("flag", ["--turns", "--confirmation-attempts"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_session_limits_fail_before_capture(flag, value):
    with pytest.raises(SystemExit) as error:
        main(["fusion-live", flag, value])
    assert error.value.code == 2
