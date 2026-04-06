"""macOS live speech capture via a small Objective-C bridge."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
import json
import subprocess
from typing import TextIO

from interaction.persistence import RuntimePaths

from .provider_types import LiveSpeechCapture, LiveSpeechClause, LiveSpeechStreamCapture, SpeechCaptureError


class MacOSLiveSpeechProvider:
    """Compile and invoke the macOS speech bridge on demand."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        helper_source: Path | None = None,
        command_runner=None,
        process_runner=None,
    ) -> None:
        self.paths = paths
        self.helper_source = helper_source or Path(__file__).with_name("macos_speech_bridge.m")
        self.command_runner = command_runner or self._default_runner
        self.process_runner = process_runner or self._default_process_runner
        self.live_event_callback: Callable[[dict[str, object]], None] | None = None

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
            "QuartzCore",
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

    def capture_turn(
        self,
        *,
        duration_s: float = 4.0,
        locale: str = "en-US",
        recording_path: str | None = None,
    ) -> LiveSpeechCapture:
        binary = self.ensure_helper()
        command = [
            str(binary),
            "--duration",
            f"{duration_s:.2f}",
            "--locale",
            locale,
        ]
        if recording_path:
            command.extend(["--record-path", recording_path])
        completed = self.command_runner(
            command
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
        capture = LiveSpeechCapture(
            transcript=transcript,
            confidence=confidence,
            locale=str(payload.get("locale", locale)),
            duration_s=float(payload.get("duration_s", duration_s)),
            provider=str(payload.get("provider", "macos_speech")),
            used_on_device=bool(payload["used_on_device"]) if "used_on_device" in payload else None,
            permission_state=str(payload.get("permission_state")) if payload.get("permission_state") is not None else None,
            recording_path=str(payload.get("recording_path")) if payload.get("recording_path") is not None else None,
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
            }
        )
        return capture

    def stream_turn(
        self,
        *,
        duration_s: float = 8.0,
        locale: str = "en-US",
        pause_ms: int = 700,
        recording_path: str | None = None,
    ) -> LiveSpeechStreamCapture:
        binary = self.ensure_helper()
        command = [
            str(binary),
            "--streaming",
            "--duration",
            f"{duration_s:.2f}",
            "--locale",
            locale,
            "--pause-ms",
            str(max(0, int(pause_ms))),
        ]
        if recording_path:
            command.extend(["--record-path", recording_path])
        process = self.process_runner(command)
        clauses: list[LiveSpeechClause] = []
        partials: list[str] = []
        final_payload: dict[str, object] | None = None
        stderr_lines: list[str] = []

        for raw_line in self._iter_process_lines(getattr(process, "stdout", None)):
            payload = self._parse_stream_line(raw_line)
            if payload is None:
                continue
            event = str(payload.get("event", "")).strip().lower()
            if event == "partial":
                transcript = str(payload.get("transcript", "")).strip()
                if transcript:
                    partials.append(transcript)
                    self._emit_live_event({"event": "partial", "transcript": transcript})
                continue
            if event == "clause":
                transcript = str(payload.get("transcript", "")).strip()
                if not transcript:
                    continue
                raw_confidence = payload.get("confidence")
                confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else None
                clause_index = int(payload.get("clause_index", len(clauses)))
                clauses.append(
                    LiveSpeechClause(
                        transcript=transcript,
                        confidence=confidence,
                        clause_index=clause_index,
                    )
                )
                self._emit_live_event(
                    {
                        "event": "clause",
                        "transcript": transcript,
                        "confidence": confidence if confidence is not None else None,
                        "clause_index": clause_index,
                    }
                )
                continue
            if event == "final" or payload.get("status") in {"success", "error"}:
                final_payload = payload
                self._emit_live_event(payload)

        stderr_lines.extend(self._iter_process_lines(getattr(process, "stderr", None)))
        returncode = process.wait()
        if final_payload is None:
            raise SpeechCaptureError(
                "invalid_helper_output",
                "The streaming speech helper did not return a final JSON payload.",
                payload={
                    "stdout_events": len(partials) + len(clauses),
                    "stderr": "".join(stderr_lines).strip(),
                },
            )
        if returncode != 0 or final_payload.get("status") != "success":
            raise SpeechCaptureError(
                str(final_payload.get("error", "capture_failed")),
                str(final_payload.get("message", "Live streaming speech capture failed.")),
                payload=final_payload,
            )

        final_transcript = str(final_payload.get("transcript", "")).strip()
        if not clauses and final_transcript:
            raw_confidence = final_payload.get("confidence")
            clauses.append(
                LiveSpeechClause(
                    transcript=final_transcript,
                    confidence=float(raw_confidence) if isinstance(raw_confidence, (int, float)) else None,
                    clause_index=0,
                )
            )
        if not final_transcript and clauses:
            final_transcript = " ".join(clause.transcript for clause in clauses).strip()
        if not final_transcript:
            raise SpeechCaptureError(
                "empty_transcript",
                "The streaming speech helper completed but returned an empty transcript.",
                payload=final_payload,
            )

        return LiveSpeechStreamCapture(
            clauses=clauses,
            partials=partials,
            final_transcript=final_transcript,
            locale=str(final_payload.get("locale", locale)),
            duration_s=float(final_payload.get("duration_s", duration_s)),
            provider=str(final_payload.get("provider", "macos_streaming_speech")),
            used_on_device=bool(final_payload["used_on_device"]) if "used_on_device" in final_payload else None,
            permission_state=(
                str(final_payload.get("permission_state"))
                if final_payload.get("permission_state") is not None
                else None
            ),
            recording_path=str(final_payload.get("recording_path")) if final_payload.get("recording_path") is not None else None,
            confidence=float(final_payload.get("confidence")) if isinstance(final_payload.get("confidence"), (int, float)) else None,
        )

    @staticmethod
    def _default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, check=False, capture_output=True, text=True)

    @staticmethod
    def _default_process_runner(command: list[str]) -> subprocess.Popen[str]:
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

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

    @staticmethod
    def _parse_stream_line(line: str) -> dict[str, object] | None:
        candidate = line.strip()
        if not candidate:
            return None
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _iter_process_lines(stream: TextIO | list[str] | None) -> Iterator[str]:
        if stream is None:
            return
        if isinstance(stream, list):
            for line in stream:
                yield str(line)
            return
        for line in stream:
            yield str(line)

    def _emit_live_event(self, payload: dict[str, object]) -> None:
        callback = self.live_event_callback
        if callback is None:
            return
        callback(dict(payload))
