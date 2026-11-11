"""Streamed previews remain display-only until a verified final capture returns."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from interaction.app import main
from interaction.audio import LiveSpeechCapture, SpeechCaptureError, SpeechStreamEvent
from interaction.contracts import EnvironmentSnapshot
from interaction.intent import VoiceIntentInterpreter
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.platform import MacOSPlatformAdapter
from interaction.runtime.live_gaze import LiveGazeSampler
from interaction.session import SessionReplay
from tests.test_live_fusion_session import SessionCamera


def run_session(monkeypatch, tmp_path, validated_profile, command, turns, *options, mode="stream", after_previews=None):
    output = io.StringIO()
    feedback = io.StringIO()
    scripted_turns = iter(turns)
    calls = []
    executions = []
    interpretations = []
    samplers = []
    camera = SessionCamera()
    environment = [EnvironmentSnapshot(active_app="Safari", active_window_title="Example")]
    paths = RuntimePaths(tmp_path)
    interpreter = VoiceIntentInterpreter.interpret

    def interpret(self, text):
        interpretations.append(text)
        return interpreter(self, text)

    class Adapter(MacOSPlatformAdapter):
        def __init__(self, **_kwargs):
            super().__init__(dry_run=True)

        def execute(self, request):
            executions.append(request)
            return super().execute(request)

    class Speech:
        def __init__(self, _paths):
            pass

        def capture_turn(self, *, duration_s, locale):
            calls.append({"mode": "recorded", "duration_s": duration_s, "locale": locale})
            turn = next(scripted_turns)
            return LiveSpeechCapture(turn["text"], turn.get("confidence", 0.95), locale, duration_s)

        def capture_stream(self, *, duration_s, locale, silence_duration_s, on_event):
            calls.append({"mode": "stream", "duration_s": duration_s, "locale": locale, "silence_duration_s": silence_duration_s})
            turn = next(scripted_turns)
            action_count = len(executions)
            interpretation_count = len(interpretations)
            on_event(SpeechStreamEvent(kind="ready"))
            assert "Microphone ready." in feedback.getvalue()
            for index, text in enumerate(turn.get("partials", [])):
                on_event(SpeechStreamEvent(kind="partial", transcript=text, elapsed_s=index * 0.01))
                assert f"Hearing: {text}" in feedback.getvalue(), "Preview must be visible before capture returns."
                assert output.getvalue() == "", "Partial feedback must not break stdout's one-document contract."
                assert len(executions) == action_count
                assert len(interpretations) == interpretation_count, "Previews must never enter the intent interpreter."
                logs = list(tmp_path.glob("logs/*.jsonl"))
                assert len(logs) == 1
                record = json.loads(logs[0].read_text().splitlines()[-1])
                assert record["transcript"] == text, "Preview must be journaled before capture returns."
                assert record["decision"] is None and record["result"] is None
            if after_previews is not None:
                after_previews(len(calls), environment, camera)
            if "error" in turn:
                raise turn["error"]
            return LiveSpeechCapture(
                turn["text"], turn.get("confidence", 0.95), locale, min(duration_s, 0.75),
                capture_mode="stream", endpoint_reason="silence", max_duration_s=duration_s,
            )

    def sampler_factory(provider, profile, **kwargs):
        sampler = LiveGazeSampler(provider, profile, **kwargs)
        samplers.append(sampler)
        return sampler

    monkeypatch.setattr(VoiceIntentInterpreter, "interpret", interpret)
    monkeypatch.setattr("interaction.app.MacOSPlatformAdapter", Adapter)
    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", Speech)
    monkeypatch.setattr("interaction.app.capture_environment", lambda: environment[0].model_copy())
    arguments = [command, "--runtime-dir", str(tmp_path), *options]
    if mode is not None:
        arguments += ["--speech-mode", mode]
    if command == "fusion-live":
        JsonStateStore(paths).save_calibration_profile(validated_profile, name="webcam-live")
        monkeypatch.setattr("interaction.app.OpenCVWebcamGazeProvider", lambda **kwargs: camera)
        monkeypatch.setattr("interaction.app.LiveGazeSampler", sampler_factory)
        arguments += ["--gaze-frames", "40", "--gaze-delta-ms", "10"]
    with redirect_stdout(output), redirect_stderr(feedback):
        assert main(arguments) == 0
    if samplers:
        assert camera.open_count == camera.close_count == 1
        assert not samplers[0]._thread.is_alive()
    payload = json.loads(output.getvalue())
    records = SessionReplay.load(Path(payload["session_log"]))
    return payload, records, calls, executions


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_revisable_previews_are_immediate_and_only_final_executes_once(monkeypatch, tmp_path, validated_profile, command):
    payload, records, calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command,
        [{"partials": ["open Safar", "open Safari"], "text": "open Notes"}],
    )
    assert len(executions) == 1
    results = [record.result for record in records if record.result]
    assert results[0]["proposal"]["arguments"]["app_name"] == "Notes"
    assert [record.transcript for record in records if record.message == "Speech preview."] == ["open Safar", "open Safari"]
    assert sum(record.phase == "interpreting" for record in records) == 1
    assert not any("Partial" in (record.message or "") for record in records)
    assert calls == [{"mode": "stream", "duration_s": 4.0, "locale": "en-US", "silence_duration_s": 0.8}]
    assert payload["live_capture"]["capture_mode"] == "stream"
    assert payload["live_capture"]["endpoint_reason"] == "silence"
    assert payload["live_capture"]["duration_s"] == 0.75
    assert payload["live_capture"]["max_duration_s"] == 4.0


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_partial_then_error_never_executes_or_falls_back(monkeypatch, tmp_path, validated_profile, command):
    payload, records, calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command,
        [{"partials": ["open Safari"], "error": SpeechCaptureError("recognition_timeout", "No final result.")}],
    )
    assert executions == []
    assert payload["session"]["failed"]
    assert payload["live_capture"]["error"] == "recognition_timeout"
    assert [call["mode"] for call in calls] == ["stream"]
    assert all(record.phase != "interpreting" for record in records)


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_commands_and_confirmations_use_selected_stream_settings(monkeypatch, tmp_path, validated_profile, command):
    payload, records, calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command,
        [{"partials": ["type Hello"], "text": "type Hello,  world!"}, {"partials": ["yes"], "text": "yes"}],
        "--duration", "3", "--confirm-duration", "1.5", "--silence-duration", "1.2", "--locale", "en-GB",
    )
    assert len(executions) == 1
    result = next(record.result for record in records if record.result)
    assert result["proposal"]["arguments"]["text"] == "Hello,  world!"
    assert calls == [
        {"mode": "stream", "duration_s": 3.0, "locale": "en-GB", "silence_duration_s": 1.2},
        {"mode": "stream", "duration_s": 1.5, "locale": "en-GB", "silence_duration_s": 1.2},
    ]
    assert payload["confirmation_captures"][0]["capture_mode"] == "stream"
    assert sum(record.phase == "interpreting" for record in records) == 2


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_error_after_partial_confirmation_cannot_authorize_pending_action(monkeypatch, tmp_path, validated_profile, command):
    payload, _records, calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command,
        [{"text": "type example"}, {"partials": ["yes"], "error": SpeechCaptureError("recognition_failed", "Recognition failed.")}],
    )
    assert executions == []
    assert len(calls) == 2
    assert payload["confirmation_captures"][0]["error"] == "recognition_failed"


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_context_change_during_previews_cancels_final_action(monkeypatch, tmp_path, validated_profile, command):
    def change_context(_number, environment, _camera):
        environment[0] = EnvironmentSnapshot(active_app="Terminal")

    payload, records, _calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command,
        [{"partials": ["scroll down"], "text": "scroll down"}], after_previews=change_context,
    )
    assert executions == []
    assert payload["live_capture"]["error"] == "context_changed"
    assert all(record.phase != "interpreting" for record in records)


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_streamed_stop_cancels_pending_work_even_when_focus_changes(monkeypatch, tmp_path, validated_profile, command):
    def change_context(number, environment, camera):
        if number == 2:
            environment[0] = EnvironmentSnapshot()
            camera.lost = True

    payload, _records, calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command,
        [{"text": "type example"}, {"partials": ["stop listening"], "text": "Stop listening."}],
        "--continuous", "--execute", after_previews=change_context,
    )
    assert executions == []
    assert len(calls) == 2
    assert payload["session"]["stopped"] and not payload["session"]["failed"]


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_revised_stop_preview_does_not_stop_final_command(monkeypatch, tmp_path, validated_profile, command):
    payload, _records, _calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command,
        [{"partials": ["stop listening"], "text": "open Safari"}],
    )
    assert len(executions) == 1
    assert not payload["session"]["stopped"]


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_recorded_default_also_avoids_synthetic_live_previews(monkeypatch, tmp_path, validated_profile, command):
    _payload, records, calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command, [{"text": "open Safari"}], mode=None,
    )
    assert len(executions) == 1
    assert [call["mode"] for call in calls] == ["recorded"]
    assert sum(record.phase == "interpreting" for record in records) == 1
    assert not any("Partial" in (record.message or "") for record in records)


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_many_streamed_previews_respect_history_limit_without_losing_journal(monkeypatch, tmp_path, validated_profile, command):
    previews = [f"open candidate {index}" for index in range(20)]
    payload, records, _calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command,
        [{"partials": previews, "text": "open Safari"}], "--history-limit", "1",
    )
    assert len(executions) == 1
    assert len(payload["events"]) == len(payload["overlay_snapshots"]) == 1
    assert payload["history"]["events"]["total"] == len(records)
    assert payload["history"]["events"]["dropped"] == len(records) - 1
    assert [record.transcript for record in records if record.message == "Speech preview."] == previews


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_interrupt_during_stream_never_dispatches_preview(monkeypatch, tmp_path, validated_profile, command):
    payload, _records, _calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command,
        [{"partials": ["open Safari"], "error": KeyboardInterrupt()}],
    )
    assert executions == []
    assert payload["session"]["stopped"] and not payload["session"]["failed"]


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
@pytest.mark.parametrize("confirmation", [False, True])
def test_clipped_turn_is_retryable_without_executing_its_partial(monkeypatch, tmp_path, validated_profile, command, confirmation):
    message = "Speech was still active at the time limit. Please repeat a shorter command."
    clipped = {"partials": ["yes" if confirmation else "open Safari"],
               "error": SpeechCaptureError("capture_limit_reached", message)}
    turns = [{"text": "type example"}, clipped, {"text": "yes"}] if confirmation else [clipped, {"text": "open Notes"}, {"text": "stop listening"}]
    options = [] if confirmation else ["--continuous"]
    payload, records, calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command, turns, *options,
    )
    assert len(executions) == 1
    assert len(calls) == 3
    assert not payload["session"]["failed"]
    assert any(record.message == message and record.phase == "recovering" for record in records)
    result = next(record.result for record in records if record.result)
    assert result["proposal"]["action"] == ("type_text" if confirmation else "open_app")
    if not confirmation:
        assert result["proposal"]["arguments"]["app_name"] == "Notes"


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
def test_bounded_session_reports_clipped_turn_failure(monkeypatch, tmp_path, validated_profile, command):
    payload, _records, calls, executions = run_session(
        monkeypatch, tmp_path, validated_profile, command,
        [{"partials": ["open Safari"], "error": SpeechCaptureError("capture_limit_reached", "Repeat a shorter command.")}],
    )
    assert executions == []
    assert len(calls) == 1
    assert payload["session"]["failed"]
    assert payload["live_capture"]["error"] == "capture_limit_reached"


@pytest.mark.parametrize("command", ["voice-live", "fusion-live"])
@pytest.mark.parametrize("value", ["0.1", "5.1", "nan", "inf", "-inf"])
def test_invalid_silence_duration_is_rejected_before_hardware(command, value):
    with pytest.raises(SystemExit) as error:
        main([command, f"--silence-duration={value}"])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["voice-smoke", "fusion-smoke"])
def test_smoke_paths_retain_scripted_partial_transcripts(tmp_path, command):
    output = io.StringIO()
    with redirect_stdout(output):
        assert main([command, "--runtime-dir", str(tmp_path)]) == 0
    payload = json.loads(output.getvalue())
    assert any("Partial" in (event["message"] or "") for event in payload["events"])
