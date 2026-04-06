"""Language-layer models shared by the router and optional Gemma backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from interaction.contracts import ActionName, EnvironmentSnapshot, GazeObservation, GroundedTarget, NavigationScope, RiskLevel, SessionMode


class InterpreterStrategy(StrEnum):
    REGEX = "regex"
    GEMMA = "gemma"
    HYBRID = "hybrid"


class VoiceRoute(StrEnum):
    COMMAND = "command"
    DICTATION = "dictation"
    TRANSLATION = "translation"
    CLARIFY = "clarify"
    REJECT = "reject"


class InterpretationSource(StrEnum):
    FAST_PATH = "fast_path"
    REGEX = "regex"
    GEMMA = "gemma"
    DICTATION_FALLBACK = "dictation_fallback"


@dataclass(frozen=True)
class TranscriptHistoryItem:
    utterance: str
    confidence: float | None = None


@dataclass(frozen=True)
class VoiceInterpretationContext:
    mode: SessionMode
    environment: EnvironmentSnapshot
    active_scope: NavigationScope | None = None
    transcript_history: tuple[TranscriptHistoryItem, ...] = ()
    latest_target: GroundedTarget | None = None
    candidate_targets: tuple[GroundedTarget, ...] = ()
    latest_observation: GazeObservation | None = None
    pending_confirmation: bool = False


class GemmaPromptContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode_context: str
    app_context: str
    scope_context: str
    transcript_context: str
    gaze_context: str

    def render(self) -> str:
        return "\n".join(
            (
                f"Mode: {self.mode_context}",
                f"App/Profile: {self.app_context}",
                f"Scope: {self.scope_context}",
                f"Recent transcripts: {self.transcript_context}",
                f"Gaze context: {self.gaze_context}",
            )
        )


class GemmaInterpretationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str | None = Field(default=None, min_length=1)
    raw_audio: bytes | None = None
    context: GemmaPromptContext
    mode: SessionMode
    environment: EnvironmentSnapshot
    active_scope: NavigationScope | None = None
    recent_transcripts: list[str] = Field(default_factory=list)
    has_gaze_context: bool = False
    supported_actions: list[str] = Field(default_factory=list)


class GemmaInterpretationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: VoiceRoute
    action: ActionName | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk: RiskLevel | None = None
    rationale: str | None = None
    normalized_scope: NavigationScope | None = None
    normalized_command: str | None = None
    cleaned_text: str | None = None
    translated_text: str | None = None
    clarification: str | None = None
    spoken_response: str | None = None
    is_deictic: bool = False
    source_language: str | None = None
    target_language: str | None = None


class GemmaBackend(Protocol):
    """Injectable backend used by the Gemma interpreter seam."""

    def interpret(
        self,
        request: GemmaInterpretationRequest,
    ) -> GemmaInterpretationResult | dict[str, Any] | None:
        """Return a validated interpretation payload or a raw dict-like response."""
