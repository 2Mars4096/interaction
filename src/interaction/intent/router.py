"""Central voice interpretation router for deterministic and optional Gemma paths."""

from __future__ import annotations

from typing import Any

from interaction.contracts import (
    ActionName,
    ActionProposal,
    NormalizedIntent,
    RiskLevel,
    SessionMode,
    StructuredNavigationRequest,
)

from .gemma_interpreter import GemmaIntentInterpreter
from .gemma_models import GemmaInterpretationResult, InterpretationSource, InterpreterStrategy, VoiceInterpretationContext, VoiceRoute
from .shortcut_grammar import FastPathMatch, match_fast_path
from .voice import VoiceDirective, VoiceDirectiveType, VoiceIntentInterpreter, match_review_token, normalize_utterance

_ACTION_RISK_FLOORS = {
    ActionName.HIGHLIGHT_TARGET: RiskLevel.L0,
    ActionName.FOCUS_TARGET: RiskLevel.L1,
    ActionName.SCROLL: RiskLevel.L1,
    ActionName.OPEN_APP: RiskLevel.L1,
    ActionName.SWITCH_APP: RiskLevel.L1,
    ActionName.PRESS_KEY: RiskLevel.L1,
    ActionName.NAVIGATE_STRUCTURE: RiskLevel.L1,
    ActionName.CLICK_TARGET: RiskLevel.L2,
    ActionName.DOUBLE_CLICK_TARGET: RiskLevel.L2,
    ActionName.RIGHT_CLICK_TARGET: RiskLevel.L2,
    ActionName.DRAG_TARGET: RiskLevel.L2,
    ActionName.TYPE_TEXT: RiskLevel.L2,
    ActionName.TRANSLATE_TEXT: RiskLevel.L2,
}

_DEICTIC_ACTIONS = {
    ActionName.HIGHLIGHT_TARGET,
    ActionName.FOCUS_TARGET,
    ActionName.CLICK_TARGET,
    ActionName.DOUBLE_CLICK_TARGET,
    ActionName.RIGHT_CLICK_TARGET,
    ActionName.DRAG_TARGET,
}


class VoiceInterpretationRouter:
    """Route utterances through review-token handling, fast path, Gemma, and fallback parsing."""

    def __init__(
        self,
        *,
        strategy: InterpreterStrategy = InterpreterStrategy.HYBRID,
        deterministic_interpreter: VoiceIntentInterpreter | None = None,
        gemma_interpreter: GemmaIntentInterpreter | None = None,
    ) -> None:
        self.strategy = strategy
        self.deterministic_interpreter = deterministic_interpreter or VoiceIntentInterpreter()
        self.gemma_interpreter = gemma_interpreter or GemmaIntentInterpreter()

    def interpret(
        self,
        utterance: str,
        context: VoiceInterpretationContext,
    ) -> VoiceDirective:
        text = normalize_utterance(utterance)
        if not text:
            return VoiceDirective(
                kind=VoiceDirectiveType.CLARIFY,
                utterance=text,
                message="I did not hear a command.",
                source=InterpretationSource.REGEX.value,
            )

        review = match_review_token(
            text,
            pending_confirmation=context.pending_confirmation,
            allow_without_pending=context.mode != SessionMode.DICTATION,
        )
        if review is not None:
            return review

        fast_path = match_fast_path(
            text,
            mode=context.mode,
            active_scope=context.active_scope,
        )
        if fast_path is not None:
            return self._directive_from_fast_path(text, fast_path)

        gemma_result = None
        if self.strategy in {InterpreterStrategy.GEMMA, InterpreterStrategy.HYBRID}:
            gemma_result = self.gemma_interpreter.interpret_text(text, context)
            directive = self._directive_from_gemma(text, context, gemma_result)
            if directive is not None:
                return directive

        directive = self.deterministic_interpreter.interpret(text)
        directive = self._postprocess_fallback_directive(directive, context)
        if directive is not None:
            return directive

        if context.mode == SessionMode.DICTATION:
            return self._dictation_directive(
                utterance=text,
                text=text,
                rationale="Dictation mode fell back to the raw transcript because no richer language result was available.",
                confidence=0.72,
                source=InterpretationSource.DICTATION_FALLBACK,
            )

        if gemma_result is None and self.strategy == InterpreterStrategy.GEMMA:
            return VoiceDirective(
                kind=VoiceDirectiveType.REJECT,
                utterance=text,
                message="No Gemma language backend is configured for this session.",
                source=InterpretationSource.GEMMA.value,
            )
        return VoiceDirective(
            kind=VoiceDirectiveType.REJECT,
            utterance=text,
            message="I could not map that request to the current bounded voice command set.",
            source=InterpretationSource.REGEX.value,
        )

    def _directive_from_fast_path(self, utterance: str, match: FastPathMatch) -> VoiceDirective:
        return self._directive_from_proposal(
            utterance=utterance,
            intent_name=match.intent_name,
            proposal=match.proposal,
            source=InterpretationSource.FAST_PATH,
        )

    def _directive_from_gemma(
        self,
        utterance: str,
        context: VoiceInterpretationContext,
        result: GemmaInterpretationResult | None,
    ) -> VoiceDirective | None:
        if result is None:
            return None
        if result.route == VoiceRoute.CLARIFY:
            return VoiceDirective(
                kind=VoiceDirectiveType.CLARIFY,
                utterance=utterance,
                message=result.clarification or result.spoken_response or "I need a bit more detail before acting on that.",
                source=InterpretationSource.GEMMA.value,
            )
        if result.route == VoiceRoute.REJECT:
            return VoiceDirective(
                kind=VoiceDirectiveType.REJECT,
                utterance=utterance,
                message=result.clarification or result.spoken_response or "That request is outside the current bounded command set.",
                source=InterpretationSource.GEMMA.value,
            )
        if result.route == VoiceRoute.DICTATION:
            dictation_text = (result.cleaned_text or utterance).strip()
            if not dictation_text:
                return VoiceDirective(
                    kind=VoiceDirectiveType.REJECT,
                    utterance=utterance,
                    message="The language backend did not return any dictation text.",
                    source=InterpretationSource.GEMMA.value,
                )
            return self._dictation_directive(
                utterance=utterance,
                text=dictation_text,
                rationale=result.rationale or "Gemma cleaned the transcript for dictation.",
                confidence=max(0.0, result.confidence or 0.0),
                source=InterpretationSource.GEMMA,
            )
        if result.route == VoiceRoute.TRANSLATION:
            translated_text = (result.translated_text or "").strip()
            if not translated_text:
                return VoiceDirective(
                    kind=VoiceDirectiveType.REJECT,
                    utterance=utterance,
                    message="Translation requires a Gemma backend that returns normalized translated_text output.",
                    source=InterpretationSource.GEMMA.value,
                )
            return self._dictation_directive(
                utterance=utterance,
                text=translated_text,
                rationale=result.rationale or "Gemma returned translated text for insertion.",
                confidence=max(0.0, result.confidence or 0.0),
                source=InterpretationSource.GEMMA,
                modifiers={
                    "route": VoiceRoute.TRANSLATION.value,
                    "target_language": result.target_language,
                    "source_language": result.source_language,
                },
            )
        if result.route != VoiceRoute.COMMAND or result.action is None:
            return None
        return self._proposal_from_gemma_command(utterance, context, result)

    def _proposal_from_gemma_command(
        self,
        utterance: str,
        context: VoiceInterpretationContext,
        result: GemmaInterpretationResult,
    ) -> VoiceDirective:
        action = result.action
        assert action is not None
        if action in {ActionName.CLARIFY, ActionName.REJECT}:
            message = result.clarification or result.spoken_response or "The language backend returned a non-executable route."
            kind = VoiceDirectiveType.CLARIFY if action == ActionName.CLARIFY else VoiceDirectiveType.REJECT
            return VoiceDirective(kind=kind, utterance=utterance, message=message, source=InterpretationSource.GEMMA.value)

        arguments = dict(result.arguments)
        if action == ActionName.NAVIGATE_STRUCTURE:
            if "scope" not in arguments and result.normalized_scope is not None:
                arguments["scope"] = result.normalized_scope.value
            if "command" not in arguments and result.normalized_command is not None:
                arguments["command"] = result.normalized_command
            try:
                StructuredNavigationRequest.from_action_arguments(arguments)
            except Exception:
                return VoiceDirective(
                    kind=VoiceDirectiveType.REJECT,
                    utterance=utterance,
                    message="The language backend returned malformed structured navigation arguments.",
                    source=InterpretationSource.GEMMA.value,
                )

        if action in _DEICTIC_ACTIONS:
            arguments = self._attach_grounding(arguments, context)
            if "target_ref" not in arguments and action != ActionName.HIGHLIGHT_TARGET:
                return VoiceDirective(
                    kind=VoiceDirectiveType.CLARIFY,
                    utterance=utterance,
                    message="I need a recent gaze target before I can resolve that command.",
                    source=InterpretationSource.GEMMA.value,
                )

        proposal = self._validated_proposal(
            action=action,
            arguments=arguments,
            confidence=max(0.0, result.confidence or 0.0),
            risk=result.risk,
            rationale=result.rationale or f'Gemma normalized "{utterance}" into a bounded action proposal.',
            spoken_response=result.spoken_response,
        )
        if proposal is None:
            return VoiceDirective(
                kind=VoiceDirectiveType.REJECT,
                utterance=utterance,
                message="The language backend returned an unsupported action proposal.",
                source=InterpretationSource.GEMMA.value,
            )
        modifiers = {
            "route": result.route.value,
            "source": InterpretationSource.GEMMA.value,
        }
        if result.normalized_scope is not None:
            modifiers["normalized_scope"] = result.normalized_scope.value
        if result.normalized_command is not None:
            modifiers["normalized_command"] = result.normalized_command
        return self._directive_from_proposal(
            utterance=utterance,
            intent_name=action.value,
            proposal=proposal,
            source=InterpretationSource.GEMMA,
            modifiers=modifiers,
        )

    def _postprocess_fallback_directive(
        self,
        directive: VoiceDirective,
        context: VoiceInterpretationContext,
    ) -> VoiceDirective | None:
        if directive.kind != VoiceDirectiveType.PROPOSAL or directive.proposal is None:
            if context.mode == SessionMode.DICTATION and directive.kind in {
                VoiceDirectiveType.CLARIFY,
                VoiceDirectiveType.REJECT,
            }:
                return self._dictation_directive(
                    utterance=directive.utterance,
                    text=directive.utterance,
                    rationale="Dictation mode kept the raw transcript instead of treating it as a command.",
                    confidence=0.7,
                    source=InterpretationSource.DICTATION_FALLBACK,
                )
            return directive

        proposal = directive.proposal
        if proposal.action == ActionName.TRANSLATE_TEXT:
            return VoiceDirective(
                kind=VoiceDirectiveType.REJECT,
                utterance=directive.utterance,
                message="Translation is unavailable because no Gemma language backend is configured.",
                source=directive.source or InterpretationSource.REGEX.value,
            )
        if proposal.action in _DEICTIC_ACTIONS:
            proposal = proposal.model_copy(update={"arguments": self._attach_grounding(proposal.arguments, context)})
        return self._directive_from_proposal(
            utterance=directive.utterance,
            intent_name=directive.intent.intent if directive.intent is not None else proposal.action.value,
            proposal=proposal,
            source=InterpretationSource(directive.source or InterpretationSource.REGEX.value),
            modifiers=directive.intent.modifiers if directive.intent is not None else None,
        )

    def _directive_from_proposal(
        self,
        *,
        utterance: str,
        intent_name: str,
        proposal: ActionProposal,
        source: InterpretationSource,
        modifiers: dict[str, Any] | None = None,
    ) -> VoiceDirective:
        intent = NormalizedIntent(
            intent=intent_name,
            modifiers={
                **(modifiers or {}),
                "source": source.value,
            },
            confidence=proposal.confidence,
            needs_clarification=False,
            utterance=utterance,
        )
        return VoiceDirective(
            kind=VoiceDirectiveType.PROPOSAL,
            utterance=utterance,
            intent=intent,
            proposal=proposal,
            source=source.value,
        )

    def _dictation_directive(
        self,
        *,
        utterance: str,
        text: str,
        rationale: str,
        confidence: float,
        source: InterpretationSource,
        modifiers: dict[str, Any] | None = None,
    ) -> VoiceDirective:
        normalized_modifiers = {"route": VoiceRoute.DICTATION.value, **(modifiers or {})}
        arguments: dict[str, Any] = {
            "text": text,
            "voice_route": normalized_modifiers.get("route", VoiceRoute.DICTATION.value),
        }
        if normalized_modifiers.get("target_language") is not None:
            arguments["target_language"] = normalized_modifiers["target_language"]
        if normalized_modifiers.get("source_language") is not None:
            arguments["source_language"] = normalized_modifiers["source_language"]
        proposal = self._validated_proposal(
            action=ActionName.TYPE_TEXT,
            arguments=arguments,
            confidence=confidence,
            risk=RiskLevel.L2,
            rationale=rationale,
        )
        assert proposal is not None
        return self._directive_from_proposal(
            utterance=utterance,
            intent_name="type_text",
            proposal=proposal,
            source=source,
            modifiers=normalized_modifiers,
        )

    def _validated_proposal(
        self,
        *,
        action: ActionName,
        arguments: dict[str, Any],
        confidence: float,
        risk: RiskLevel | None,
        rationale: str,
        spoken_response: str | None = None,
    ) -> ActionProposal | None:
        floor = _ACTION_RISK_FLOORS.get(action)
        if floor is None:
            return None
        proposal_risk = self._max_risk(floor, risk or floor)
        proposal_confidence = min(0.99, max(0.55, confidence or 0.0))
        try:
            return ActionProposal(
                action=action,
                arguments=arguments,
                confidence=proposal_confidence,
                risk=proposal_risk,
                requires_confirmation=proposal_risk in {RiskLevel.L2, RiskLevel.L3},
                rationale=rationale,
                spoken_response=spoken_response,
            )
        except Exception:
            return None

    @staticmethod
    def _attach_grounding(
        arguments: dict[str, Any],
        context: VoiceInterpretationContext,
    ) -> dict[str, Any]:
        if "target_ref" in arguments or context.latest_target is None:
            return dict(arguments)
        grounded_arguments = dict(arguments)
        grounded_arguments["target_ref"] = context.latest_target.target_id
        grounded_arguments["target_label"] = context.latest_target.label
        if context.latest_target.bounds is not None:
            grounded_arguments["target_bounds"] = context.latest_target.bounds.model_dump(mode="json")
        if context.latest_observation is not None and context.latest_observation.x_norm is not None and context.latest_observation.y_norm is not None:
            grounded_arguments["normalized_point"] = {
                "x": context.latest_observation.x_norm,
                "y": context.latest_observation.y_norm,
            }
        return grounded_arguments

    @staticmethod
    def _max_risk(left: RiskLevel, right: RiskLevel) -> RiskLevel:
        order = {
            RiskLevel.L0: 0,
            RiskLevel.L1: 1,
            RiskLevel.L2: 2,
            RiskLevel.L3: 3,
        }
        return left if order[left] >= order[right] else right
