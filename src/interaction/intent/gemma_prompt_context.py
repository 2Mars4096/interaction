"""Helpers for building prompt context for the optional Gemma language layer."""

from __future__ import annotations

from interaction.contracts import NavigationScope, SessionMode

from .gemma_models import GemmaPromptContext, VoiceInterpretationContext


def build_prompt_context(context: VoiceInterpretationContext) -> GemmaPromptContext:
    return GemmaPromptContext(
        mode_context=_mode_context(context.mode),
        app_context=_app_context(context),
        scope_context=_scope_context(context.active_scope),
        transcript_context=_transcript_context(context),
        gaze_context=_gaze_context(context),
    )


def _mode_context(mode: SessionMode) -> str:
    if mode == SessionMode.DICTATION:
        return "dictation mode: prefer cleaned text unless the utterance is explicitly command-shaped"
    return "command mode: prefer bounded desktop actions and clarifications over free-form output"


def _app_context(context: VoiceInterpretationContext) -> str:
    environment = context.environment
    parts = []
    if environment.active_app:
        parts.append(f"app={environment.active_app}")
    if environment.active_bundle_id:
        parts.append(f"bundle={environment.active_bundle_id}")
    if environment.active_profile_id:
        parts.append(f"profile={environment.active_profile_id}")
    if environment.active_window_title:
        parts.append(f"window={environment.active_window_title}")
    return ", ".join(parts) if parts else "no active app context"


def _scope_context(scope: NavigationScope | None) -> str:
    if scope is None:
        return "no active structured-navigation scope"
    return f"active scope={scope.value}"


def _transcript_context(context: VoiceInterpretationContext) -> str:
    if not context.transcript_history:
        return "none"
    return " | ".join(
        entry.utterance
        for entry in context.transcript_history[-4:]
        if entry.utterance.strip()
    ) or "none"


def _gaze_context(context: VoiceInterpretationContext) -> str:
    if context.latest_target is None:
        return "no grounded target"
    parts = [
        f'target="{context.latest_target.label}"',
        f"id={context.latest_target.target_id}",
        f"confidence={context.latest_target.confidence:.2f}",
    ]
    if context.latest_observation is not None and context.latest_observation.x_norm is not None and context.latest_observation.y_norm is not None:
        parts.append(
            "point="
            f"({context.latest_observation.x_norm:.2f}, {context.latest_observation.y_norm:.2f})"
        )
    if context.candidate_targets:
        labels = ", ".join(target.label for target in context.candidate_targets[:3])
        parts.append(f"candidates={labels}")
    return ", ".join(parts)
