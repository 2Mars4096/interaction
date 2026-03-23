"""macOS live speech capture via a small Objective-C bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess

from interaction.persistence import RuntimePaths


@dataclass(frozen=True)
class LiveSpeechCapture:
    transcript: str
    confidence: float | None
    locale: str
    duration_s: float
    provider: str = "macos_speech"
    used_on_device: bool | None = None
    permission_state: str | None = None


class SpeechCaptureError(RuntimeError):
    """Raised when the native speech helper cannot produce a transcript."""

    def __init__(self, code: str, message: str, *, payload: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.payload = payload or {}


class MacOSLiveSpeechProvider:
    """Compile and invoke the macOS speech bridge on demand."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        helper_source: Path | None = None,
        command_runner=None,
    ) -> None:
        self.paths = paths
        self.helper_source = helper_source or Path(__file__).with_name("macos_speech_bridge.m")
        self.command_runner = command_runner or self._default_runner

    @property
    def helper_binary_path(self) -> Path:
        return self.paths.bin_dir / "macos_speech_bridge"

    def ensure_helper(self) -> Path:
        self.paths.ensure()
        binary = self.helper_binary_path
        if binary.exists() and binary.stat().st_mtime >= self.helper_source.stat().st_mtime:
            return binary
        command = [
            "clang",
            "-fobjc-arc",
            "-framework",
            "Foundation",
            "-framework",
            "Speech",
            "-framework",
            "AVFoundation",
            str(self.helper_source),
            "-o",
            str(binary),
        ]
        completed = self.command_runner(command)
        if completed.returncode != 0:
            raise SpeechCaptureError(
                "helper_build_failed",
                "Failed to build the macOS speech helper.",
                payload={
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                },
            )
        return binary

    def capture_turn(self, *, duration_s: float = 4.0, locale: str = "en-US") -> LiveSpeechCapture:
        binary = self.ensure_helper()
        completed = self.command_runner(
            [
                str(binary),
                "--duration",
                f"{duration_s:.2f}",
                "--locale",
                locale,
            ]
        )
        payload = self._parse_payload(completed.stdout, completed.stderr)
        if completed.returncode != 0 or payload.get("status") != "success":
            raise SpeechCaptureError(
                str(payload.get("error", "capture_failed")),
                str(payload.get("message", "Live speech capture failed.")),
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
        return LiveSpeechCapture(
            transcript=transcript,
            confidence=confidence,
            locale=str(payload.get("locale", locale)),
            duration_s=float(payload.get("duration_s", duration_s)),
            provider=str(payload.get("provider", "macos_speech")),
            used_on_device=bool(payload["used_on_device"]) if "used_on_device" in payload else None,
            permission_state=str(payload.get("permission_state")) if payload.get("permission_state") is not None else None,
        )

    @staticmethod
    def _default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, check=False, capture_output=True, text=True)

    @staticmethod
    def _parse_payload(stdout: str, stderr: str) -> dict[str, object]:
        for candidate in (stdout.strip(), stderr.strip()):
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return {
            "status": "error",
            "error": "invalid_helper_output",
            "message": "The speech helper did not return valid JSON output.",
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }
