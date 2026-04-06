import io
import json
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from interaction.app import main
from interaction.audio import LiveSpeechCapture, MacOSLiveSpeechProvider, SpeechCaptureError
from interaction.contracts import EnvironmentSnapshot
from interaction.persistence import RuntimePaths


def test_live_speech_provider_builds_and_parses_success(tmp_path: Path) -> None:
    paths = RuntimePaths(tmp_path / ".interaction")
    helper_source = tmp_path / "bridge.m"
    helper_source.write_text("int main(void) { return 0; }\n")
    seen_commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        seen_commands.append(command)
        if command[0] == "clang":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "status": "success",
                    "provider": "macos_speech",
                    "transcript": "open Safari",
                    "locale": "en-US",
                    "duration_s": 1.5,
                    "confidence": 0.88,
                    "used_on_device": True,
                    "permission_state": "granted",
                }
            ),
            stderr="",
        )

    provider = MacOSLiveSpeechProvider(paths, helper_source=helper_source, command_runner=runner)
    capture = provider.capture_turn(duration_s=1.5, locale="en-US")

    assert capture == LiveSpeechCapture(
        transcript="open Safari",
        confidence=0.88,
        locale="en-US",
        duration_s=1.5,
        provider="macos_speech",
        used_on_device=True,
        permission_state="granted",
    )
    assert seen_commands[0][0] == "clang"
    assert "QuartzCore" in seen_commands[0]
    assert seen_commands[1][0].endswith("macos_speech_bridge")


def test_live_speech_provider_stream_turn_wraps_single_capture(tmp_path: Path) -> None:
    paths = RuntimePaths(tmp_path / ".interaction")
    helper_source = tmp_path / "bridge.m"
    helper_source.write_text("int main(void) { return 0; }\n")

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = [
                json.dumps(
                    {
                        "event": "final",
                        "status": "success",
                        "provider": "macos_streaming_speech",
                        "transcript": "tab 3",
                        "locale": "en-US",
                        "duration_s": 1.0,
                        "confidence": 0.91,
                    }
                ),
            ]
            self.stderr = []

        def wait(self) -> int:
            return 0

    provider = MacOSLiveSpeechProvider(
        paths,
        helper_source=helper_source,
        command_runner=runner,
        process_runner=lambda _command: FakeProcess(),
    )
    capture = provider.stream_turn(duration_s=1.0, locale="en-US")

    assert capture.transcript == "tab 3"
    assert len(capture.clauses) == 1
    assert capture.clauses[0].is_final is True
    assert capture.clauses[0].transcript == "tab 3"


def test_live_speech_provider_passes_record_path_to_helper(tmp_path: Path) -> None:
    paths = RuntimePaths(tmp_path / ".interaction")
    helper_source = tmp_path / "bridge.m"
    helper_source.write_text("int main(void) { return 0; }\n")
    seen_commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        seen_commands.append(command)
        if command[0] == "clang":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "status": "success",
                    "provider": "macos_speech",
                    "transcript": "open Safari",
                    "locale": "en-US",
                    "duration_s": 1.5,
                    "recording_path": str(tmp_path / "capture.caf"),
                }
            ),
            stderr="",
        )

    provider = MacOSLiveSpeechProvider(paths, helper_source=helper_source, command_runner=runner)
    capture = provider.capture_turn(duration_s=1.5, locale="en-US", recording_path=str(tmp_path / "capture.caf"))

    assert capture.recording_path == str(tmp_path / "capture.caf")
    assert "--record-path" in seen_commands[1]


def test_live_speech_provider_surfaces_structured_error(tmp_path: Path) -> None:
    paths = RuntimePaths(tmp_path / ".interaction")
    helper_source = tmp_path / "bridge.m"
    helper_source.write_text("int main(void) { return 0; }\n")

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[0] == "clang":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(
                {
                    "status": "error",
                    "error": "microphone_denied",
                    "message": "Microphone permission was denied for the speech helper.",
                }
            ),
            stderr="",
        )

    provider = MacOSLiveSpeechProvider(paths, helper_source=helper_source, command_runner=runner)

    try:
        provider.capture_turn(duration_s=1.0, locale="en-US")
    except SpeechCaptureError as error:
        assert error.code == "microphone_denied"
        assert error.payload["status"] == "error"
    else:
        raise AssertionError("Expected SpeechCaptureError for denied microphone access.")


def test_live_speech_provider_parses_streaming_clauses(tmp_path: Path) -> None:
    paths = RuntimePaths(tmp_path / ".interaction")
    helper_source = tmp_path / "bridge.m"
    helper_source.write_text("int main(void) { return 0; }\n")

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = [
                json.dumps({"event": "partial", "transcript": "open"}),
                json.dumps({"event": "clause", "transcript": "open Safari", "confidence": 0.91, "clause_index": 0}),
                json.dumps(
                    {
                        "event": "final",
                        "status": "success",
                        "transcript": "open Safari",
                        "locale": "en-US",
                        "duration_s": 2.5,
                        "provider": "macos_streaming_speech",
                        "used_on_device": True,
                        "permission_state": "granted",
                    }
                ),
            ]
            self.stderr = []

        def wait(self) -> int:
            return 0

    provider = MacOSLiveSpeechProvider(
        paths,
        helper_source=helper_source,
        command_runner=runner,
        process_runner=lambda _command: FakeProcess(),
    )

    capture = provider.stream_turn(duration_s=2.5, locale="en-US", pause_ms=600)

    assert capture.final_transcript == "open Safari"
    assert capture.partials == ["open"]
    assert len(capture.clauses) == 1
    assert capture.clauses[0].transcript == "open Safari"
    assert capture.provider == "macos_streaming_speech"


def test_live_speech_provider_parses_stream_recording_path(tmp_path: Path) -> None:
    paths = RuntimePaths(tmp_path / ".interaction")
    helper_source = tmp_path / "bridge.m"
    helper_source.write_text("int main(void) { return 0; }\n")

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = [
                json.dumps(
                    {
                        "event": "final",
                        "status": "success",
                        "transcript": "open Safari",
                        "locale": "en-US",
                        "duration_s": 2.5,
                        "provider": "macos_streaming_speech",
                        "recording_path": str(tmp_path / "stream.caf"),
                    }
                ),
            ]
            self.stderr = []

        def wait(self) -> int:
            return 0

    provider = MacOSLiveSpeechProvider(
        paths,
        helper_source=helper_source,
        command_runner=runner,
        process_runner=lambda _command: FakeProcess(),
    )

    capture = provider.stream_turn(duration_s=2.5, locale="en-US", pause_ms=600, recording_path=str(tmp_path / "stream.caf"))

    assert capture.recording_path == str(tmp_path / "stream.caf")


def test_live_speech_provider_streaming_emits_live_events(tmp_path: Path) -> None:
    paths = RuntimePaths(tmp_path / ".interaction")
    helper_source = tmp_path / "bridge.m"
    helper_source.write_text("int main(void) { return 0; }\n")

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = [
                json.dumps({"event": "partial", "transcript": "open"}),
                json.dumps({"event": "clause", "transcript": "open Safari", "confidence": 0.91, "clause_index": 0}),
                json.dumps({"event": "final", "status": "success", "transcript": "open Safari"}),
            ]
            self.stderr = []

        def wait(self) -> int:
            return 0

    provider = MacOSLiveSpeechProvider(
        paths,
        helper_source=helper_source,
        command_runner=runner,
        process_runner=lambda _command: FakeProcess(),
    )
    seen_events: list[dict[str, object]] = []
    provider.live_event_callback = lambda payload: seen_events.append(payload)

    capture = provider.stream_turn(duration_s=2.5, locale="en-US", pause_ms=600)

    assert capture.final_transcript == "open Safari"
    assert [event["event"] for event in seen_events] == ["partial", "clause", "final"]
    assert seen_events[0]["transcript"] == "open"
    assert seen_events[1]["transcript"] == "open Safari"


def test_live_speech_provider_streaming_emits_events_before_wait(tmp_path: Path) -> None:
    paths = RuntimePaths(tmp_path / ".interaction")
    helper_source = tmp_path / "bridge.m"
    helper_source.write_text("int main(void) { return 0; }\n")

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    callback_events: list[dict[str, object]] = []

    class TrackingStdout:
        def __iter__(self):
            yield json.dumps({"event": "partial", "transcript": "open"})
            assert callback_events, "Expected live callback before stdout iteration completed."
            yield json.dumps({"event": "final", "status": "success", "transcript": "open Safari"})

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = TrackingStdout()
            self.stderr = []

        def wait(self) -> int:
            return 0

    provider = MacOSLiveSpeechProvider(
        paths,
        helper_source=helper_source,
        command_runner=runner,
        process_runner=lambda _command: FakeProcess(),
    )
    provider.live_event_callback = lambda payload: callback_events.append(payload)

    capture = provider.stream_turn(duration_s=2.5, locale="en-US", pause_ms=600)

    assert capture.final_transcript == "open Safari"
    assert callback_events[0]["event"] == "partial"


def test_live_speech_provider_streaming_surfaces_structured_error(tmp_path: Path) -> None:
    paths = RuntimePaths(tmp_path / ".interaction")
    helper_source = tmp_path / "bridge.m"
    helper_source.write_text("int main(void) { return 0; }\n")

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = [
                json.dumps(
                    {
                        "event": "final",
                        "status": "error",
                        "error": "recognition_failed",
                        "message": "Streaming speech recognition failed.",
                    }
                )
            ]
            self.stderr = []

        def wait(self) -> int:
            return 1

    provider = MacOSLiveSpeechProvider(
        paths,
        helper_source=helper_source,
        command_runner=runner,
        process_runner=lambda _command: FakeProcess(),
    )

    try:
        provider.stream_turn(duration_s=2.5, locale="en-US", pause_ms=600)
    except SpeechCaptureError as error:
        assert error.code == "recognition_failed"
    else:
        raise AssertionError("Expected SpeechCaptureError for streaming recognition failure.")


def test_app_voice_live_success_path_executes_broker_flow(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths):
            pass

        def capture_turn(self, *, duration_s: float = 4.0, locale: str = "en-US") -> LiveSpeechCapture:
            return LiveSpeechCapture(
                transcript="open Safari",
                confidence=0.91,
                locale=locale,
                duration_s=duration_s,
                provider="fake",
                used_on_device=True,
                permission_state="granted",
            )

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["voice-live", "--runtime-dir", str(tmp_path / ".interaction"), "--duration", "1.0"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_capture"]["status"] == "success"
    assert any(event["result"] and event["result"]["details"]["commands"] == [["open", "-a", "Safari"]] for event in payload["events"])


def test_app_voice_live_uses_frontmost_environment_when_helper_is_available(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths):
            pass

        def capture_turn(self, *, duration_s: float = 4.0, locale: str = "en-US") -> LiveSpeechCapture:
            return LiveSpeechCapture(
                transcript="tab 3",
                confidence=0.94,
                locale=locale,
                duration_s=duration_s,
                provider="fake",
                used_on_device=True,
                permission_state="granted",
            )

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeProvider)
    monkeypatch.setattr(
        "interaction.app.capture_frontmost_environment",
        lambda: EnvironmentSnapshot(
            active_app="Google Chrome",
            active_bundle_id="com.google.Chrome",
            active_window_title="Docs",
        ),
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["voice-live", "--runtime-dir", str(tmp_path / ".interaction"), "--duration", "1.0"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["environment"]["active_bundle_id"] == "com.google.Chrome"
    assert any(event["result"] and event["result"]["details"]["navigation"]["scope"] == "tab" for event in payload["events"])


def test_app_voice_live_save_audio_defaults_into_logs(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths):
            self.live_event_callback = None

        def stream_turn(self, *, duration_s: float = 8.0, locale: str = "en-US", pause_ms: int = 700, recording_path: str | None = None):
            from interaction.audio.macos_speech import LiveSpeechClause, LiveSpeechStreamCapture

            return LiveSpeechStreamCapture(
                clauses=[LiveSpeechClause(transcript="hello there", confidence=0.95, clause_index=0)],
                partials=["hello"],
                final_transcript="hello there",
                locale=locale,
                duration_s=duration_s,
                provider="fake_stream",
                used_on_device=True,
                permission_state="granted",
                recording_path=recording_path,
            )

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "voice-dictate",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--duration",
                "2.0",
                "--save-audio",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_capture"]["recording_path"].endswith("voice-dictate-audio.caf")


def test_app_voice_live_error_path_reports_recovery(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths):
            pass

        def capture_turn(self, *, duration_s: float = 4.0, locale: str = "en-US") -> LiveSpeechCapture:
            raise SpeechCaptureError(
                "microphone_denied",
                "Microphone permission was denied for the speech helper.",
                payload={"status": "error", "error": "microphone_denied"},
            )

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["voice-live", "--runtime-dir", str(tmp_path / ".interaction"), "--duration", "1.0"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_capture"]["status"] == "error"
    assert any(event["phase"] == "recovering" for event in payload["events"])


def test_app_voice_live_dictation_lane_types_transcript(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths):
            pass

        def capture_turn(self, *, duration_s: float = 4.0, locale: str = "en-US") -> LiveSpeechCapture:
            return LiveSpeechCapture(
                transcript="hello there",
                confidence=0.93,
                locale=locale,
                duration_s=duration_s,
                provider="fake",
                used_on_device=True,
                permission_state="granted",
            )

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "voice-live",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--duration",
                "1.0",
                "--lane",
                "dictation",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["voice_lane"] == "dictation"
    assert payload["voice_session_state"]["dictated_clause_history"] == ["Hello there"]
    assert payload["voice_session_state"]["assembled_text"] == "Hello there"
    assert payload["voice_session_state"]["clause_count"] == 1
    assert any(
        event["result"] and event["result"]["details"]["commands"][0][0] == "osascript"
        for event in payload["events"]
    )


def test_app_voice_live_streaming_hybrid_processes_each_clause(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths):
            pass

        def stream_turn(self, *, duration_s: float = 8.0, locale: str = "en-US", pause_ms: int = 700):
            from interaction.audio.macos_speech import LiveSpeechClause, LiveSpeechStreamCapture

            return LiveSpeechStreamCapture(
                clauses=[
                    LiveSpeechClause(transcript="open Safari", confidence=0.95, clause_index=0),
                    LiveSpeechClause(transcript="hello there", confidence=0.9, clause_index=1),
                ],
                partials=["open", "open Safari hello"],
                final_transcript="open Safari hello there",
                locale=locale,
                duration_s=duration_s,
                provider="fake_stream",
                used_on_device=True,
                permission_state="granted",
            )

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "voice-live",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--duration",
                "1.0",
                "--streaming",
                "--lane",
                "hybrid",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_capture"]["mode"] == "streaming"
    assert payload["voice_lane"] == "hybrid"
    assert payload["voice_session_state"]["dictated_clause_history"] == ["Hello there"]
    assert payload["voice_session_state"]["assembled_text"] == "Hello there"
    assert payload["voice_session_state"]["clause_count"] == 1
    assert any(
        event["result"] and event["result"]["details"].get("commands") == [["open", "-a", "Safari"]]
        for event in payload["events"]
    )
    assert any(
        event["phase"] == "interpreting"
        and event["message"] == "Hybrid lane did not match a bounded command and fell back to dictation."
        for event in payload["events"]
    )


def test_app_voice_live_streaming_mirrors_transcripts_to_stderr(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths):
            self.live_event_callback = None

        def stream_turn(self, *, duration_s: float = 8.0, locale: str = "en-US", pause_ms: int = 700):
            callback = self.live_event_callback
            if callable(callback):
                callback({"event": "partial", "transcript": "open"})
                callback({"event": "clause", "transcript": "open Safari", "clause_index": 0})
                callback({"event": "final", "transcript": "open Safari"})
            from interaction.audio.macos_speech import LiveSpeechClause, LiveSpeechStreamCapture

            return LiveSpeechStreamCapture(
                clauses=[LiveSpeechClause(transcript="open Safari", confidence=0.95, clause_index=0)],
                partials=["open"],
                final_transcript="open Safari",
                locale=locale,
                duration_s=duration_s,
                provider="fake_stream",
                used_on_device=True,
                permission_state="granted",
            )

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeProvider)

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(
            [
                "voice-live",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--duration",
                "1.0",
                "--streaming",
                "--lane",
                "hybrid",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_capture"]["mode"] == "streaming"
    stderr_text = stderr.getvalue()
    assert "[voice] partial: open" in stderr_text
    assert "[voice] clause 1: open Safari" in stderr_text
    assert "[voice] final: open Safari" in stderr_text


def test_app_voice_live_streaming_mirrors_error_to_stderr(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths):
            self.live_event_callback = None

        def stream_turn(self, *, duration_s: float = 8.0, locale: str = "en-US", pause_ms: int = 700):
            callback = self.live_event_callback
            if callable(callback):
                callback(
                    {
                        "event": "final",
                        "status": "error",
                        "message": "No speech detected",
                        "error": "recognition_failed",
                        "audio_peak_dbfs": -91.0,
                        "audio_rms_dbfs": -91.0,
                        "audio_signal_detected": False,
                    }
                )
            raise SpeechCaptureError(
                "recognition_failed",
                "No speech detected",
                payload={
                        "status": "error",
                        "message": "No speech detected",
                        "event": "final",
                        "error": "recognition_failed",
                        "audio_peak_dbfs": -91.0,
                        "audio_rms_dbfs": -91.0,
                        "audio_signal_detected": False,
                    },
                )

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeProvider)

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(
            [
                "voice-live",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--duration",
                "1.0",
                "--streaming",
                "--lane",
                "hybrid",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["live_capture"]["status"] == "error"
    stderr_text = stderr.getvalue()
    assert "[voice] mic: no usable audio detected (peak -91.0 dBFS, rms -91.0 dBFS)" in stderr_text
    assert "[voice] error: No speech detected" in stderr_text


def test_app_voice_dictate_alias_uses_streaming_dictation_defaults(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths):
            pass

        def stream_turn(self, *, duration_s: float = 10.0, locale: str = "en-US", pause_ms: int = 700):
            from interaction.audio.macos_speech import LiveSpeechClause, LiveSpeechStreamCapture

            return LiveSpeechStreamCapture(
                clauses=[
                    LiveSpeechClause(transcript="um hello there", confidence=0.94, clause_index=0),
                ],
                partials=["um hello"],
                final_transcript="um hello there",
                locale=locale,
                duration_s=duration_s,
                provider="fake_stream",
                used_on_device=True,
                permission_state="granted",
            )

    monkeypatch.setattr("interaction.app.MacOSLiveSpeechProvider", FakeProvider)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "voice-dictate",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--duration",
                "2.0",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["voice_lane"] == "dictation"
    assert payload["live_capture"]["mode"] == "streaming"
    assert payload["live_capture"]["pause_ms"] == 700
    assert payload["voice_session_state"]["dictated_clause_history"] == ["Hello there"]
    assert payload["voice_session_state"]["assembled_text"] == "Hello there"
    assert payload["voice_session_state"]["clause_count"] == 1
    assert any(
        event["result"] and event["result"]["details"]["commands"][0][0] == "osascript"
        for event in payload["events"]
    )
