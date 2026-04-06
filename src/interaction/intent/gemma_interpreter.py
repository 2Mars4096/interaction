"""Optional Gemma-backed language interpretation seam."""

from __future__ import annotations

from typing import Any

from interaction.contracts import ActionName

from .gemma_models import GemmaBackend, GemmaInterpretationRequest, GemmaInterpretationResult, VoiceInterpretationContext
from .gemma_prompt_context import build_prompt_context

_SUPPORTED_ACTIONS = [action.value for action in ActionName if action not in {ActionName.CLARIFY, ActionName.REJECT}]


class GemmaIntentInterpreter:
    """Normalize backend responses into typed language-layer results."""

    def __init__(self, backend: GemmaBackend | None = None) -> None:
        self.backend = backend

    @property
    def is_available(self) -> bool:
        return self.backend is not None

    def interpret_text(
        self,
        utterance: str,
        context: VoiceInterpretationContext,
    ) -> GemmaInterpretationResult | None:
        return self._interpret(
            transcript=utterance,
            raw_audio=None,
            context=context,
        )

    def interpret_audio(
        self,
        raw_audio: bytes,
        context: VoiceInterpretationContext,
        *,
        transcript: str | None = None,
    ) -> GemmaInterpretationResult | None:
        return self._interpret(
            transcript=transcript,
            raw_audio=raw_audio,
            context=context,
        )

    def _interpret(
        self,
        *,
        transcript: str | None,
        raw_audio: bytes | None,
        context: VoiceInterpretationContext,
    ) -> GemmaInterpretationResult | None:
        if self.backend is None:
            return None
        request = GemmaInterpretationRequest(
            transcript=transcript.strip() if transcript is not None else None,
            raw_audio=raw_audio,
            context=build_prompt_context(context),
            mode=context.mode,
            environment=context.environment,
            active_scope=context.active_scope,
            recent_transcripts=[entry.utterance for entry in context.transcript_history[-4:]],
            has_gaze_context=context.latest_target is not None,
            supported_actions=_SUPPORTED_ACTIONS,
        )
        try:
            raw_response = self.backend.interpret(request)
        except Exception:
            return None
        if raw_response is None:
            return None
        return self._validate_response(raw_response)

    @staticmethod
    def _validate_response(raw_response: GemmaInterpretationResult | dict[str, Any]) -> GemmaInterpretationResult | None:
        try:
            if isinstance(raw_response, GemmaInterpretationResult):
                return raw_response
            return GemmaInterpretationResult.model_validate(raw_response)
        except Exception:
            return None
