"""Bound live presentation history without losing actions or the event journal."""

import io
import json
import weakref
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from interaction.app import _build_payload, main
from interaction.audio import LiveSpeechCapture, SpeechCaptureError
from interaction.feedback import VoiceFeedbackEvent, VoiceLoopPhase
from interaction.session import SessionLogger, SessionReplay
from interaction.session.live import BoundedHistory, LiveEventBuffer
from interaction.ui import ConsoleOverlayRenderer, OverlayController
from tests.test_live_fusion_session import run_session as run_fusion_session, wait_for_worker


def run_voice_session(monkeypatch, tmp_path, utterances, *, limit):
    speech = iter(utterances)

    class Speech:
        def __init__(self, _paths):
            pass

        def capture_turn(self, *, duration_s, locale):
            utterance = next(speech)
            if utterance is None:
                raise SpeechCaptureError("empty_transcript", "No speech heard.")
            return LiveSpeechCapture(utterance, 0.95, locale, duration_s)

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", Speech)
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(io.StringIO()):
        assert main(["voice-live", "--continuous", "--history-limit", str(limit), "--runtime-dir", str(tmp_path)]) == 0
    return json.loads(output.getvalue())


def journal(payload):
    return SessionReplay.load(Path(payload["session_log"]))


def assert_history_matches_payload(payload):
    for name in ("events", "overlay_snapshots", "captures", "confirmation_captures"):
        if name not in payload["history"]:
            continue
        counts = payload["history"][name]
        assert type(payload[name]) is list
        assert counts["retained"] == len(payload[name]) <= payload["history"]["limit"]
        assert counts["total"] == counts["retained"] + counts["dropped"]
    assert payload["history"]["events"]["total"] == len(journal(payload))


def test_partial_feedback_eviction_releases_objects_but_keeps_complete_journal(tmp_path):
    logger = SessionLogger(tmp_path / "events.jsonl")
    overlay = OverlayController()
    buffer = LiveEventBuffer("voice", logger, overlay, history_limit=3, stream=io.StringIO())
    first = VoiceFeedbackEvent(phase=VoiceLoopPhase.LISTENING, transcript="first")
    first_reference = weakref.ref(first)
    buffer.append(first)
    del first
    for number in range(500):
        buffer.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.LISTENING, transcript=f"partial {number}"))
        assert len(buffer) <= 3
        assert len(buffer.serialized_events) <= 3
        assert len(buffer.snapshots) <= 3
    assert first_reference() is None
    assert [event.transcript for event in buffer] == ["partial 497", "partial 498", "partial 499"]
    assert [event["transcript"] for event in buffer.serialized_events] == ["partial 497", "partial 498", "partial 499"]
    assert [snapshot["transcript"] for snapshot in buffer.snapshots] == ["partial 497", "partial 498", "partial 499"]
    assert buffer.summary() == {"total": 501, "retained": 3, "dropped": 498}
    before = logger.path.read_bytes()
    payload = _build_payload(
        source="voice", events=buffer, logger=logger, overlay=overlay, renderer=ConsoleOverlayRenderer(), extra={},
    )
    assert logger.path.read_bytes() == before, "Building stdout must not append the retained events a second time."
    assert_history_matches_payload(payload)
    records = journal(payload)
    assert [record.index for record in records] == list(range(1, 502))
    assert records[0].transcript == "first"
    assert records[-1].transcript == "partial 499"
    assert payload["final_overlay"]["transcript"] == "partial 499"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_buffer_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="positive integer"):
        BoundedHistory(limit)


@pytest.mark.parametrize("command", ["voice-live", "fusion-live", "gaze-live", "gaze-calibrate"])
def test_cli_rejects_zero_limit_before_using_hardware(command):
    with pytest.raises(SystemExit) as error:
        main([command, "--history-limit", "0"])
    assert error.value.code == 2


def test_long_voice_session_bounds_silent_captures_and_preserves_lifetime_counts(monkeypatch, tmp_path):
    payload = run_voice_session(monkeypatch, tmp_path, ["scroll down", None] * 100 + ["stop listening"], limit=3)
    assert_history_matches_payload(payload)
    assert payload["history"]["captures"] == {"total": 201, "retained": 3, "dropped": 198}
    assert payload["history"]["confirmation_captures"] == {"total": 0, "retained": 0, "dropped": 0}
    assert payload["session"] == {"commands": 101, "stopped": True, "failed": False}
    assert [capture["status"] for capture in payload["captures"]] == ["success", "error", "success"]
    assert payload["live_capture"]["transcript"] == "stop listening"
    records = journal(payload)
    assert sum(record.message == "No speech heard." for record in records) == 100
    assert sum(record.result is not None for record in records) == 100
    assert payload["final_overlay"]["phase"] == "idle"


def test_history_limit_one_does_not_erase_pending_typing_or_confirmation_counts(monkeypatch, tmp_path):
    payload = run_voice_session(
        monkeypatch, tmp_path, ["type first text", None, "yes", "type second text", "yes", "stop listening"], limit=1,
    )
    assert_history_matches_payload(payload)
    assert payload["history"]["captures"] == {"total": 3, "retained": 1, "dropped": 2}
    assert payload["history"]["confirmation_captures"] == {"total": 3, "retained": 1, "dropped": 2}
    results = [record.result for record in journal(payload) if record.result]
    assert [result["proposal"]["arguments"]["text"] for result in results] == ["first text", "second text"]
    assert all(event["result"] is None for event in payload["events"]), "Execution evidence remains in the journal after eviction."
    assert payload["session"] == {"commands": 3, "stopped": True, "failed": False}


def test_long_fusion_session_keeps_all_execution_metrics_after_eviction(monkeypatch, tmp_path, validated_profile):
    utterances = ["click", "yes", None, "scroll up", None] * 10 + ["stop listening"]
    payload, _ = run_fusion_session(
        monkeypatch, tmp_path, validated_profile, utterances, "--continuous", "--history-limit", "1",
    )
    assert_history_matches_payload(payload)
    assert payload["history"]["captures"] == {"total": 41, "retained": 1, "dropped": 40}
    assert payload["history"]["confirmation_captures"] == {"total": 10, "retained": 1, "dropped": 9}
    assert payload["session"] == {"commands": 21, "stopped": True, "failed": False}
    assert payload["metrics"]["executions"] == 20
    assert payload["metrics"]["confirmations"] == 10
    assert payload["live_capture"]["transcript"] == "stop listening"
    assert payload["confirmation_capture"]["transcript"] == "yes"
    records = journal(payload)
    assert sum(record.message == "No speech heard." for record in records) == 20
    assert sum(record.result is not None for record in records) == 20


def test_evicted_confirmation_does_not_authorize_click_after_gaze_loss(monkeypatch, tmp_path, validated_profile):
    def lose_and_reacquire(number, camera, sampler, _environment):
        if number == 2:
            missing = sampler.missing_readings
            camera.lost = True
            wait_for_worker(sampler, lambda: sampler.missing_readings > missing)
            camera.lost = False
            wait_for_worker(sampler, lambda: sampler._current() is not None)

    payload, _ = run_fusion_session(
        monkeypatch, tmp_path, validated_profile, ["click", "yes", "scroll down", "stop listening"],
        "--continuous", "--history-limit", "1", on_capture=lose_and_reacquire,
    )
    assert_history_matches_payload(payload)
    assert payload["metrics"]["cancellations"] == 1
    results = [record.result for record in journal(payload) if record.result]
    assert [result["proposal"]["action"] for result in results] == ["scroll"]


def test_gaze_startup_error_honors_history_limit_and_journals_dropped_event(tmp_path):
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(io.StringIO()):
        assert main(["gaze-live", "--history-limit", "1", "--runtime-dir", str(tmp_path)]) == 0
    payload = json.loads(output.getvalue())
    assert_history_matches_payload(payload)
    assert payload["history"]["events"] == {"total": 2, "retained": 1, "dropped": 1}
    assert payload["live_gaze"]["error"] == "missing_calibration_profile"
    assert "No current validated" in journal(payload)[0].message


def test_short_live_payload_keeps_complete_arrays_and_replay_remains_complete(monkeypatch, tmp_path):
    payload = run_voice_session(monkeypatch, tmp_path, ["open Safari", "stop listening"], limit=200)
    assert_history_matches_payload(payload)
    assert all(counts["dropped"] == 0 for name, counts in payload["history"].items() if name != "limit")
    output = io.StringIO()
    with redirect_stdout(output):
        assert main(["replay", "--session-log", payload["session_log"]]) == 0
    replayed = json.loads(output.getvalue())
    assert len(replayed["records"]) == len(payload["events"])
    assert replayed["final_overlay"] == payload["final_overlay"]
    assert "history" not in replayed


def test_scripted_smoke_keeps_full_history_without_live_retention_metadata(tmp_path):
    output = io.StringIO()
    with redirect_stdout(output):
        assert main(["voice-smoke", "--runtime-dir", str(tmp_path), "--utterance", "type hello", "--follow-up", "yes"]) == 0
    payload = json.loads(output.getvalue())
    assert "history" not in payload
    assert len(journal(payload)) == len(payload["events"])
    assert any(event["result"] for event in payload["events"])
