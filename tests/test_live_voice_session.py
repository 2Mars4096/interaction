import io
import json
from contextlib import redirect_stderr, redirect_stdout

from interaction.app import main
from interaction.audio import LiveSpeechCapture, SpeechCaptureError
from interaction.session import SessionReplay
from interaction.contracts import EnvironmentSnapshot


def run_session(monkeypatch, tmp_path, utterances, *options):
    output = io.StringIO()
    feedback = io.StringIO()
    heard = iter(utterances)
    before_capture = []

    class Provider:
        def __init__(self, _paths):
            pass

        def capture_turn(self, *, duration_s, locale):
            before_capture.append(feedback.getvalue())
            utterance = next(heard)
            if utterance is None:
                raise SpeechCaptureError("empty_transcript", "No speech heard.")
            return LiveSpeechCapture(utterance, 0.95, locale, duration_s)

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", Provider)
    with redirect_stdout(output), redirect_stderr(feedback):
        main(["voice-live", "--runtime-dir", str(tmp_path), *options])
    return json.loads(output.getvalue()), before_capture, feedback.getvalue()


def test_live_typing_can_be_confirmed_in_same_process(monkeypatch, tmp_path):
    payload, before, _ = run_session(monkeypatch, tmp_path, ["type Hello, world!", "Yes."])
    assert "Listening" in before[0]
    assert 'Confirm typing: "Hello, world!"?' in before[1]
    results = [event["result"] for event in payload["events"] if event["result"]]
    assert len(results) == 1
    assert results[0]["proposal"]["arguments"]["text"] == "Hello, world!"
    assert len(payload["confirmation_captures"]) == 1
    from pathlib import Path

    assert len(SessionReplay.load(Path(payload["session_log"]))) == len(payload["events"])


def test_continuous_session_accepts_successive_commands_and_stops(monkeypatch, tmp_path):
    payload, before, _ = run_session(
        monkeypatch, tmp_path, ["open Safari.", "scroll down.", "Stop listening."], "--continuous"
    )
    assert len(before) == 3
    assert payload["session"]["stopped"]
    results = [event["result"] for event in payload["events"] if event["result"]]
    assert [result["proposal"]["action"] for result in results] == ["open_app", "scroll"]


def test_pending_action_cancellation_never_executes(monkeypatch, tmp_path):
    payload, _, _ = run_session(monkeypatch, tmp_path, ["type example", "Cancel."])
    assert not any(event["result"] for event in payload["events"])
    assert any("cancelled" in (event["message"] or "") for event in payload["events"])


def test_session_can_be_bounded_without_restarting_cli(monkeypatch, tmp_path):
    payload, before, _ = run_session(monkeypatch, tmp_path, ["open Safari", "open Notes"], "--turns", "2")
    assert len(before) == 2
    assert payload["session"]["commands"] == 2


def test_continuous_session_survives_silence(monkeypatch, tmp_path):
    payload, before, _ = run_session(monkeypatch, tmp_path, [None, "open Safari", "stop listening"], "--continuous")
    assert len(before) == 3
    assert not payload["session"]["failed"]
    assert payload["session"]["commands"] == 2
    assert any(event["result"] for event in payload["events"])


def test_confirmation_retries_silence_within_budget(monkeypatch, tmp_path):
    payload, before, _ = run_session(monkeypatch, tmp_path, ["type example", None, "yes"])
    assert len(before) == 3
    assert any(event["result"] for event in payload["events"])


def test_confirmation_silence_eventually_cancels(monkeypatch, tmp_path):
    payload, before, _ = run_session(monkeypatch, tmp_path, ["type example", None, None])
    assert len(before) == 3
    assert not any(event["result"] for event in payload["events"])
    assert any("no clear confirmation" in (event["message"] or "") for event in payload["events"])


def test_keyboard_interrupt_ends_session_with_json(monkeypatch, tmp_path):
    class Provider:
        def __init__(self, _paths):
            pass

        def capture_turn(self, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", Provider)
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(io.StringIO()):
        main(["voice-live", "--continuous", "--runtime-dir", str(tmp_path)])
    assert json.loads(output.getvalue())["session"]["stopped"]


def test_application_change_during_speech_cancels_execution(monkeypatch, tmp_path):
    contexts = iter([EnvironmentSnapshot(active_app="Notes"), EnvironmentSnapshot(active_app="Terminal")])
    monkeypatch.setattr("interaction.app.capture_environment", lambda: next(contexts))
    payload, _, _ = run_session(monkeypatch, tmp_path, ["type example"])
    assert payload["live_capture"]["error"] == "context_changed"
    assert not any(event["result"] for event in payload["events"])


def test_continuous_session_recovers_from_changed_context(monkeypatch, tmp_path):
    notes = EnvironmentSnapshot(active_app="Notes")
    terminal = EnvironmentSnapshot(active_app="Terminal")
    contexts = iter([notes, terminal, notes, notes, notes, notes])
    monkeypatch.setattr("interaction.app.capture_environment", lambda: next(contexts))
    payload, _, _ = run_session(monkeypatch, tmp_path, ["scroll down", "scroll up", "stop listening"], "--continuous")
    assert payload["captures"][0]["error"] == "context_changed"
    results = [event["result"] for event in payload["events"] if event["result"]]
    assert len(results) == 1
    assert results[0]["proposal"]["arguments"]["direction"] == "up"
    assert payload["session"]["stopped"]
    assert not payload["session"]["failed"]


def test_stop_does_not_require_a_second_matching_context_probe(monkeypatch, tmp_path):
    probes = []

    def environment():
        probes.append(True)
        return EnvironmentSnapshot(active_app="Notes" if len(probes) == 1 else "Terminal")

    monkeypatch.setattr("interaction.app.capture_environment", environment)
    payload, before, _ = run_session(monkeypatch, tmp_path, ["stop listening"], "--continuous")
    assert len(before) == 1
    assert len(probes) == 1
    assert payload["session"]["stopped"] and not payload["session"]["failed"]
    assert not any(event["result"] for event in payload["events"])


def test_stop_is_heard_with_unknown_focus_in_execute_mode(monkeypatch, tmp_path):
    monkeypatch.setattr("interaction.app.capture_environment", EnvironmentSnapshot)
    payload, before, _ = run_session(monkeypatch, tmp_path, ["stop listening"], "--execute", "--continuous")
    assert len(before) == 1
    assert payload["session"]["stopped"] and not payload["session"]["failed"]
    assert not any(event["result"] for event in payload["events"])


def test_unknown_focus_still_blocks_normal_live_execution(monkeypatch, tmp_path):
    monkeypatch.setattr("interaction.app.capture_environment", EnvironmentSnapshot)
    payload, _, _ = run_session(monkeypatch, tmp_path, ["scroll down"], "--execute")
    assert payload["live_capture"]["error"] == "context_unavailable"
    assert payload["session"]["failed"]
    assert not any(event["result"] for event in payload["events"])


def test_stop_during_confirmation_cancels_pending_typing(monkeypatch, tmp_path):
    payload, before, _ = run_session(monkeypatch, tmp_path, ["type example", "stop listening"], "--continuous")
    assert len(before) == 2
    assert payload["session"]["stopped"] and not payload["session"]["failed"]
    assert not any(event["result"] for event in payload["events"])
