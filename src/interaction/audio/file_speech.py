"""File-backed speech capture via the macOS speech bridge."""

from __future__ import annotations

from pathlib import Path

from .macos_speech import MacOSLiveSpeechProvider
from .provider_types import LiveSpeechCapture, LiveSpeechStreamCapture, SpeechCaptureError


class MacOSFileSpeechProvider(MacOSLiveSpeechProvider):
    """Replay a prerecorded audio file through the macOS speech bridge."""

    def __init__(
        self,
        paths,
        *,
        audio_path: str | Path,
        helper_source: Path | None = None,
        command_runner=None,
        process_runner=None,
    ) -> None:
        super().__init__(
            paths,
            helper_source=helper_source,
            command_runner=command_runner,
            process_runner=process_runner,
        )
        self.audio_path = Path(audio_path).expanduser()

    def capture_turn(
        self,
        *,
        duration_s: float = 0.0,
        locale: str = "en-US",
        recording_path: str | None = None,
    ) -> LiveSpeechCapture:
        audio_path = self.audio_path.resolve()
        if not audio_path.is_file():
            raise SpeechCaptureError(
                "audio_file_missing",
                f'Audio file not found: "{audio_path}"',
                payload={"audio_path": str(audio_path)},
            )

        binary = self.ensure_helper()
        command = [
            str(binary),
            "--audio-file",
            str(audio_path),
            "--locale",
            locale,
        ]
        completed = self.command_runner(command)
        payload = self._parse_payload(completed.stdout, completed.stderr)
        if completed.returncode != 0 or payload.get("status") != "success":
            raise SpeechCaptureError(
                str(payload.get("error", "capture_failed")),
                str(payload.get("message", "Speech file transcription failed.")),
                payload=payload,
            )

        transcript = str(payload.get("transcript", "")).strip()
        if not transcript:
            raise SpeechCaptureError(
                "empty_transcript",
                "The speech helper completed but returned an empty transcript.",
                payload=payload,
            )

        raw_confidence = payload.get("confidence")
        confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else None
        raw_duration = payload.get("duration_s")
        resolved_duration = float(raw_duration) if isinstance(raw_duration, (int, float)) else float(max(0.0, duration_s))
        capture = LiveSpeechCapture(
            transcript=transcript,
            confidence=confidence,
            locale=str(payload.get("locale", locale)),
            duration_s=resolved_duration,
            provider=str(payload.get("provider", "macos_file_speech")),
            used_on_device=bool(payload["used_on_device"]) if "used_on_device" in payload else None,
            permission_state=str(payload.get("permission_state")) if payload.get("permission_state") is not None else None,
            recording_path=str(payload.get("recording_path") or recording_path or audio_path),
        )
        self._emit_live_event(
            {
                "event": "final",
                "status": "success",
                "transcript": capture.transcript,
                "confidence": capture.confidence if capture.confidence is not None else None,
                "locale": capture.locale,
                "duration_s": capture.duration_s,
                "provider": capture.provider,
                "recording_path": capture.recording_path,
            }
        )
        return capture

    def stream_turn(
        self,
        *,
        duration_s: float = 0.0,
        locale: str = "en-US",
        pause_ms: int = 450,
        recording_path: str | None = None,
    ) -> LiveSpeechStreamCapture:
        del pause_ms
        return LiveSpeechStreamCapture.from_capture(
            self.capture_turn(
                duration_s=duration_s,
                locale=locale,
                recording_path=recording_path,
            )
        )
