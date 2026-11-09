import io
import json
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

from interaction.app import main
from interaction.audio import LiveSpeechCapture, MacOSFileSpeechProvider
from interaction.persistence import RuntimePaths


def test_file_speech_provider_passes_audio_file_to_helper(tmp_path: Path) -> None:
    paths = RuntimePaths(tmp_path / ".interaction")
    helper_source = tmp_path / "bridge.m"
    helper_source.write_text("int main(void) { return 0; }\n")
    audio_path = tmp_path / "fixture.wav"
    audio_path.write_bytes(b"RIFF")
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
                    "provider": "macos_file_speech",
                    "transcript": "open Safari",
                    "locale": "en-US",
                    "duration_s": 1.2,
                    "confidence": 0.92,
                    "used_on_device": True,
                    "permission_state": "granted",
                    "recording_path": str(audio_path),
                }
            ),
            stderr="",
        )

    provider = MacOSFileSpeechProvider(
        paths,
        audio_path=audio_path,
        helper_source=helper_source,
        command_runner=runner,
    )
    capture = provider.capture_turn(locale="en-US")

    assert capture.transcript == "open Safari"
    assert capture.provider == "macos_file_speech"
    assert capture.recording_path == str(audio_path)
    assert seen_commands[1][0].endswith("macos_speech_bridge")
    assert "--audio-file" in seen_commands[1]
    assert str(audio_path.resolve()) in seen_commands[1]


def test_voice_eval_cli_reports_signal_only_pass(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths, *, audio_path: str):
            self.audio_path = audio_path

        def capture_turn(self, *, duration_s: float = 0.0, locale: str = "en-US", recording_path: str | None = None) -> LiveSpeechCapture:
            del duration_s, recording_path
            return LiveSpeechCapture(
                transcript="open Safari",
                confidence=0.91,
                locale=locale,
                duration_s=1.0,
                provider="fake_file",
                used_on_device=True,
                permission_state="granted",
                recording_path=self.audio_path,
            )

    monkeypatch.setattr("interaction.app.MacOSFileSpeechProvider", FakeProvider)

    audio_path = tmp_path / "fixture.wav"
    audio_path.write_bytes(b"RIFF")
    manifest_path = tmp_path / "fixture.json"
    manifest_path.write_text(
        json.dumps(
            {
                "reference_transcript": "open Safari",
                "expected_action": "open_app",
                "expected_arguments": {"app_name": "Safari"},
                "expected_risk": "L1",
                "expected_broker_decision": "allow",
            }
        )
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "voice-eval",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--audio",
                str(audio_path),
                "--manifest",
                str(manifest_path),
                "--active-app",
                "Finder",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["signal_only"] is True
    assert payload["audio_capture"]["status"] == "success"
    assert payload["signal_summary"]["proposal"]["action"] == "open_app"
    assert payload["signal_summary"]["decision"]["decision"] == "allow"
    assert payload["evaluation"]["passed"] is True
    assert payload["latency_breakdown_ms"]["total_ms"] == payload["duration_ms"]
    assert payload["latency_breakdown_ms"]["transcription_ms"] >= 0.0
    assert payload["latency_breakdown_ms"]["signal_ms"] >= 0.0
    assert payload["latency_breakdown_ms"]["postprocess_ms"] >= 0.0
    assert any(
        event["result"]
        and event["result"]["status"] == "blocked"
        and event["result"]["details"].get("signal_only") is True
        for event in payload["events"]
    )
    assert all(
        not (
            event["result"]
            and isinstance(event["result"].get("details"), dict)
            and "commands" in event["result"]["details"]
        )
        for event in payload["events"]
    )


def test_voice_eval_cli_reports_dictation_route(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths, *, audio_path: str):
            self.audio_path = audio_path

        def capture_turn(self, *, duration_s: float = 0.0, locale: str = "en-US", recording_path: str | None = None) -> LiveSpeechCapture:
            del duration_s, recording_path
            return LiveSpeechCapture(
                transcript="hello team this should stay in dictation mode",
                confidence=0.89,
                locale=locale,
                duration_s=2.0,
                provider="fake_file",
                used_on_device=True,
                permission_state="granted",
                recording_path=self.audio_path,
            )

    monkeypatch.setattr("interaction.app.MacOSFileSpeechProvider", FakeProvider)

    audio_path = tmp_path / "fixture.wav"
    audio_path.write_bytes(b"RIFF")
    manifest_path = tmp_path / "fixture.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prompt_text": "hello team this should stay in dictation mode",
                "lane": "dictation",
                "reference_output_text": "Hello team this should stay in dictation mode",
                "expected_route": "dictation",
                "expected_action": "type_text",
                "expected_risk": "L2",
                "expected_broker_decision": "allow",
                "expected_message_contains": "Dictation lane",
            }
        )
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "voice-eval",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--audio",
                str(audio_path),
                "--manifest",
                str(manifest_path),
                "--lane",
                "dictation",
                "--active-app",
                "Notes",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["signal_summary"]["route"] == "dictation"
    assert payload["evaluation"]["passed"] is True
    checks_by_name = {check["name"]: check for check in payload["evaluation"]["checks"]}
    assert checks_by_name["reference_output_text"]["passed"] is True


def test_voice_eval_cli_reports_translation_route_with_manifest_backend(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths, *, audio_path: str):
            self.audio_path = audio_path

        def capture_turn(self, *, duration_s: float = 0.0, locale: str = "en-US", recording_path: str | None = None) -> LiveSpeechCapture:
            del duration_s, recording_path
            return LiveSpeechCapture(
                transcript="translate hello to French",
                confidence=0.9,
                locale=locale,
                duration_s=1.0,
                provider="fake_file",
                used_on_device=True,
                permission_state="granted",
                recording_path=self.audio_path,
            )

    monkeypatch.setattr("interaction.app.MacOSFileSpeechProvider", FakeProvider)

    audio_path = tmp_path / "fixture.wav"
    audio_path.write_bytes(b"RIFF")
    manifest_path = tmp_path / "fixture.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prompt_text": "translate hello to French",
                "reference_output_text": "bonjour",
                "expected_route": "translation",
                "expected_action": "type_text",
                "expected_arguments": {"text": "bonjour", "voice_route": "translation", "target_language": "French"},
                "expected_risk": "L2",
                "gemma_response": {
                    "route": "translation",
                    "confidence": 0.93,
                    "translated_text": "bonjour",
                    "target_language": "French",
                    "source_language": "English",
                    "rationale": "Normalized translation for fixture translation.",
                },
            }
        )
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "voice-eval",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--audio",
                str(audio_path),
                "--manifest",
                str(manifest_path),
                "--active-app",
                "Notes",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["signal_summary"]["route"] == "translation"
    assert payload["signal_summary"]["proposal"]["arguments"]["text"] == "bonjour"
    assert payload["signal_summary"]["proposal"]["arguments"]["voice_route"] == "translation"
    assert payload["signal_summary"]["decision"]["decision"] == "confirm"
    assert payload["evaluation"]["passed"] is True


def test_voice_eval_cli_allows_reference_transcript_drift_when_signal_matches(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths, *, audio_path: str):
            self.audio_path = audio_path

        def capture_turn(self, *, duration_s: float = 0.0, locale: str = "en-US", recording_path: str | None = None) -> LiveSpeechCapture:
            del duration_s, recording_path
            return LiveSpeechCapture(
                transcript="Flick this",
                confidence=0.76,
                locale=locale,
                duration_s=1.0,
                provider="fake_file",
                used_on_device=True,
                permission_state="granted",
                recording_path=self.audio_path,
            )

    monkeypatch.setattr("interaction.app.MacOSFileSpeechProvider", FakeProvider)

    audio_path = tmp_path / "fixture.wav"
    audio_path.write_bytes(b"RIFF")
    manifest_path = tmp_path / "fixture.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prompt_text": "click this",
                "reference_transcript": "click this",
                "expected_route": "clarify",
                "expected_message_contains": "gaze grounding",
            }
        )
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "voice-eval",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--audio",
                str(audio_path),
                "--manifest",
                str(manifest_path),
                "--active-app",
                "Finder",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["signal_summary"]["transcript"] == "Flick this"
    assert payload["signal_summary"]["route"] == "clarify"
    assert payload["evaluation"]["passed"] is True
    checks_by_name = {check["name"]: check for check in payload["evaluation"]["checks"]}
    assert checks_by_name["reference_transcript"]["passed"] is False
    assert checks_by_name["expected_route"]["passed"] is True
    assert checks_by_name["expected_message_contains"]["passed"] is True
