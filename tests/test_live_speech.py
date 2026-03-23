import io
import json
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

from interaction.app import main
from interaction.audio import LiveSpeechCapture, MacOSLiveSpeechProvider, SpeechCaptureError
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
    assert seen_commands[1][0].endswith("macos_speech_bridge")


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
