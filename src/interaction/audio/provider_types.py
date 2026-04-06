"""Shared speech-provider contracts for live capture integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class LiveSpeechCapture:
    transcript: str
    confidence: float | None
    locale: str
    duration_s: float
    provider: str = "macos_speech"
    used_on_device: bool | None = None
    permission_state: str | None = None
    recording_path: str | None = None


@dataclass(frozen=True)
class LiveSpeechClause:
    transcript: str
    confidence: float | None = None
    clause_index: int = 0
    locale: str = "en-US"
    duration_s: float | None = None
    provider: str = "macos_speech"
    is_final: bool = True


@dataclass(frozen=True)
class LiveSpeechStreamCapture:
    clauses: list[LiveSpeechClause] = field(default_factory=list)
    partials: list[str] = field(default_factory=list)
    final_transcript: str = ""
    locale: str = "en-US"
    duration_s: float = 0.0
    provider: str = "macos_streaming_speech"
    used_on_device: bool | None = None
    permission_state: str | None = None
    recording_path: str | None = None
    confidence: float | None = None

    @property
    def transcript(self) -> str:
        return self.final_transcript

    @classmethod
    def from_capture(cls, capture: LiveSpeechCapture) -> "LiveSpeechStreamCapture":
        return cls(
            clauses=[
                LiveSpeechClause(
                    transcript=capture.transcript,
                    confidence=capture.confidence,
                    clause_index=0,
                    locale=capture.locale,
                    duration_s=capture.duration_s,
                    provider=capture.provider,
                    is_final=True,
                )
            ],
            partials=[],
            final_transcript=capture.transcript,
            locale=capture.locale,
            duration_s=capture.duration_s,
            provider=capture.provider,
            used_on_device=capture.used_on_device,
            permission_state=capture.permission_state,
            recording_path=capture.recording_path,
            confidence=capture.confidence,
        )


class SpeechCaptureError(RuntimeError):
    """Raised when a speech provider cannot produce a usable transcript."""

    def __init__(self, code: str, message: str, *, payload: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.payload = payload or {}


class SpeechProvider(Protocol):
    """Protocol shared by live speech integrations."""

    def capture_turn(
        self,
        *,
        duration_s: float = 4.0,
        locale: str = "en-US",
        recording_path: str | None = None,
    ) -> LiveSpeechCapture:
        """Capture a single turn and return the final transcript."""

    def stream_turn(
        self,
        *,
        duration_s: float = 4.0,
        locale: str = "en-US",
        pause_ms: int = 450,
        recording_path: str | None = None,
    ) -> LiveSpeechStreamCapture:
        """Capture a turn as streamed clauses when the provider supports it."""
