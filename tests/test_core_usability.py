"""Behavioral regressions for the first-use live command flows (no hardware actions)."""
import io
import json
from contextlib import redirect_stdout

import pytest

from interaction.app import main
from interaction.audio import LiveSpeechCapture, LiveSpeechStreamCapture, SpeechCaptureError
from interaction.contracts import EnvironmentSnapshot, GazeObservation
from interaction.feedback import VoiceFeedbackEvent, VoiceLoopPhase
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.platform import MacOSPlatformAdapter
from interaction.runtime import MultimodalFusionLoop
from interaction.ui.overlay import OverlayController
from interaction.ui.shell import build_shell_view_state
from interaction.vision import CalibrationProfile
from tests.test_fusion_loop import _compose_target
from tests.test_live_fusion_mvp import FakeFusionGazeProvider, FakeFusionSpeechProvider, FakeNoReadingGazeProvider


ENV = EnvironmentSnapshot(active_app="Notes", active_bundle_id="com.apple.Notes", active_window_title="Draft")


def install_speech(monkeypatch, transcripts):
    queue = iter(transcripts)
    calls = []

    class Provider:
        def __init__(self, _paths):
            pass

        def capture_turn(self, **kwargs):
            calls.append(kwargs)
            value = next(queue)
            if isinstance(value, Exception):
                raise value
            return LiveSpeechCapture(value, .95, "en-US", kwargs.get("duration_s", 4), provider="fake")

        def stream_turn(self, **kwargs):
            return LiveSpeechStreamCapture.from_capture(self.capture_turn(**kwargs))

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", Provider)
    monkeypatch.setattr("interaction.app._capture_runtime_environment", lambda **kwargs: ENV)
    return calls


def run_cli(tmp_path, *args):
    output = io.StringIO()
    with redirect_stdout(output):
        main([*args, "--runtime-dir", str(tmp_path / "runtime")])
    return json.loads(output.getvalue())


@pytest.mark.parametrize("answer, executed", [("Yes.", True), ("cancel", False)])
def test_live_typing_finishes_confirmation(monkeypatch, tmp_path, capsys, answer, executed):
    calls = install_speech(monkeypatch, ["type hello world", answer])
    result = run_cli(tmp_path, "voice-live")
    assert len(calls) == 2
    assert bool([event for event in result["events"] if event.get("result")]) is executed
    assert result["voice_session_state"]["has_pending_decision"] is False
    assert "Say yes or cancel" in capsys.readouterr().err


def test_unrelated_confirmation_cannot_execute_another_command(monkeypatch, tmp_path):
    install_speech(monkeypatch, ["type hello", "open Safari", "perhaps"])
    result = run_cli(tmp_path, "voice-live")
    assert not any(event.get("result") for event in result["events"])
    assert not result["voice_session_state"]["has_pending_decision"]
    assert "cancelled" in result["final_overlay"]["message"]


def test_confirmation_capture_failure_cancels(monkeypatch, tmp_path):
    install_speech(monkeypatch, ["type hello", SpeechCaptureError("no_speech", "No speech detected.")])
    result = run_cli(tmp_path, "voice-live")
    assert result["confirmation_captures"][0]["error"] == "no_speech"
    assert not any(event.get("result") for event in result["events"])
    assert not result["voice_session_state"]["has_pending_decision"]


def test_changed_window_cancels_typing_confirmation(monkeypatch, tmp_path):
    install_speech(monkeypatch, ["type hello", "yes"])
    environments = iter([ENV, ENV, ENV.model_copy(update={"active_window_title": "Other draft"})])
    monkeypatch.setattr("interaction.app._capture_runtime_environment", lambda **kwargs: next(environments))
    result = run_cli(tmp_path, "voice-live")
    assert not any(event.get("result") for event in result["events"])
    assert "window changed" in result["final_overlay"]["message"]


def test_continuous_dictation_keeps_history_and_stops(monkeypatch, tmp_path):
    calls = install_speech(monkeypatch, ["hello team", "more news", "stop listening"])
    result = run_cli(tmp_path, "voice-dictate", "--continuous", "--max-turns", "5")
    assert len(calls) == 3
    assert result["session_turns"] == 3
    assert result["session_stopped"]
    assert result["voice_session_state"]["clause_count"] == 2
    assert "stop listening" not in result["voice_session_state"]["assembled_text"]
    assert len(list((tmp_path / "runtime" / "logs").glob("*.jsonl"))) == 3


def test_dictation_history_does_not_cross_windows(monkeypatch, tmp_path):
    install_speech(monkeypatch, ["hello team", "scratch that"])
    other = ENV.model_copy(update={"active_window_title": "Other draft"})
    environments = iter([ENV, ENV, other, other])
    monkeypatch.setattr("interaction.app._capture_runtime_environment", lambda **kwargs: next(environments))
    result = run_cli(tmp_path, "voice-dictate", "--continuous", "--max-turns", "2")
    assert not any(event.get("result") for event in result["events"])
    assert result["voice_session_state"]["clause_count"] == 0


def pending_fusion():
    loop = MultimodalFusionLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    observation = GazeObservation(confidence=.95, x_norm=.56, y_norm=.2, fixation_ms=250)
    loop.update_gaze_context(observation, _compose_target())
    loop.run_voice_turn("click this", ENV)
    assert loop.pending_decision
    return loop, observation


@pytest.mark.parametrize("change", ["expired", "moved", "window"])
def test_fusion_confirmation_rejects_changed_context(change):
    loop, observation = pending_fusion()
    env = ENV
    if change == "expired":
        loop.state.advance(5000)
    elif change == "moved":
        loop.update_gaze_context(observation.model_copy(update={"x_norm": .8}), _compose_target())
    else:
        env = ENV.model_copy(update={"active_window_title": "Other"})
    events = loop.run_voice_turn("yes", env)
    assert not any(event.result for event in events)
    assert loop.pending_decision is None
    assert any("cancelled" in (event.message or "") for event in events)


def test_fusion_confirmation_accepts_reacquired_same_target():
    loop, observation = pending_fusion()
    loop.state.advance(5000)
    loop.update_gaze_context(observation, _compose_target())
    events = loop.run_voice_turn("yes", ENV)
    assert any(event.result for event in events)


@pytest.mark.parametrize("lose_lock", [False, True])
def test_live_fusion_reacquires_after_speech_and_confirmation(monkeypatch, tmp_path, lose_lock):
    providers = []

    def provider_factory(*args, **kwargs):
        provider = FakeNoReadingGazeProvider() if lose_lock and providers else FakeFusionGazeProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr("interaction.app._build_webcam_provider", provider_factory)
    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeFusionSpeechProvider)
    monkeypatch.setattr("interaction.app._capture_runtime_environment", lambda **kwargs: ENV)
    monkeypatch.setattr("interaction.app._collect_fusion_candidate_targets", lambda **kwargs: ([], []))
    store = JsonStateStore(RuntimePaths(tmp_path / "runtime"))
    store.save_calibration_profile(CalibrationProfile(), name="webcam-live")
    result = run_cli(tmp_path, "fusion-live", "--gaze-frames", "6")
    assert len(providers) == (2 if lose_lock else 3)
    assert bool(result["metrics"]["executions"]) is not lose_lock


def test_idle_preserves_recovery_until_next_capture():
    overlay = OverlayController()
    overlay.apply_event("voice", VoiceFeedbackEvent(VoiceLoopPhase.RECOVERING, message="Permission denied", transcript="hello"))
    state = overlay.apply_event("voice", VoiceFeedbackEvent(VoiceLoopPhase.IDLE, message="Idle"))
    assert state.message == "Permission denied"
    assert state.transcript == "hello"
    state = overlay.apply_event("voice", VoiceFeedbackEvent(VoiceLoopPhase.LISTENING, message="Listening"))
    assert state.message == "Listening"
    assert state.transcript is None


def test_shell_does_not_advertise_unavailable_translation_or_precision():
    state = build_shell_view_state(source="voice", overlay_state=OverlayController().state, environment=ENV)
    assert not state.exact_pointing_available
    commands = [command.utterance for pane in (*state.panes, *state.panels) for command in pane.commands]
    assert not any(command.startswith("translate ") for command in commands)


def test_execute_dispatches_confirmed_typing_to_os_runner(monkeypatch, tmp_path):
    install_speech(monkeypatch, ["type hello world", "yes"])
    commands = []
    monkeypatch.setattr("interaction.app.MacOSPlatformAdapter", lambda **kwargs: MacOSPlatformAdapter(**kwargs, runner=commands.append))
    result = run_cli(tmp_path, "voice-live", "--execute")
    assert commands
    assert any("hello world" in " ".join(command) for command in commands)
    executions = [event["result"] for event in result["events"] if event.get("result")]
    assert len(executions) == 1
    assert executions[0]["details"]["dry_run"] is False


def test_continuous_session_survives_silence(monkeypatch, tmp_path):
    calls = install_speech(monkeypatch, [SpeechCaptureError("empty_transcript", "No speech."), "stop listening"])
    result = run_cli(tmp_path, "voice-live", "--continuous", "--max-turns", "3")
    assert len(calls) == 2
    assert result["session_stopped"]


def test_stop_listening_cancels_pending_confirmation(monkeypatch, tmp_path):
    install_speech(monkeypatch, ["type hello", "stop listening"])
    result = run_cli(tmp_path, "voice-live", "--continuous")
    assert result["session_stopped"]
    assert not result["voice_session_state"]["has_pending_decision"]
    assert not any(event.get("result") for event in result["events"])


def test_streaming_interrupt_terminates_microphone_helper(monkeypatch, tmp_path):
    from interaction.audio import MacOSLiveSpeechProvider

    class Process:
        terminated = False
        waited = False

        @property
        def stdout(self):
            raise KeyboardInterrupt

        def terminate(self):
            self.terminated = True

        def wait(self, **kwargs):
            self.waited = True
            return 0

    process = Process()
    provider = MacOSLiveSpeechProvider(RuntimePaths(tmp_path), process_runner=lambda cmd: process)
    monkeypatch.setattr(provider, "ensure_helper", lambda: tmp_path / "helper")
    with pytest.raises(KeyboardInterrupt):
        provider.stream_turn()
    assert process.terminated and process.waited


def test_gaze_capture_does_not_reuse_target_after_tracking_loss():
    from interaction.runtime import capture_live_gaze_context
    provider = FakeFusionGazeProvider()
    context, events, summary = capture_live_gaze_context(provider, CalibrationProfile(), frames=14, delta_ms=100)
    assert summary["successful_readings"] == 8
    assert context is None
    assert "tracking was lost" in events[-1].message
