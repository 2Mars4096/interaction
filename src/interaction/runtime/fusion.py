"""Shared-state multimodal fusion loop."""

from __future__ import annotations

from dataclasses import dataclass

from interaction.audio import ScriptedTranscriber
from interaction.contracts import BrokerDecision, BrokerDecisionType, EnvironmentSnapshot, GazeObservation, GroundedTarget
from interaction.control import CommandBroker
from interaction.feedback import FusionFeedbackEvent, FusionLoopPhase
from interaction.intent import FusionIntentResolver, VoiceDirectiveType, VoiceIntentInterpreter
from interaction.platform import MacOSPlatformAdapter, PlatformAdapter

from .pending import PendingAction
from .state import SharedInteractionState


@dataclass
class FusionMetrics:
    total_turns: int = 0
    clarifications: int = 0
    confirmations: int = 0
    executions: int = 0
    cancellations: int = 0
    recoveries: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_turns": self.total_turns,
            "clarifications": self.clarifications,
            "confirmations": self.confirmations,
            "executions": self.executions,
            "cancellations": self.cancellations,
            "recoveries": self.recoveries,
        }


class MultimodalFusionLoop:
    """Combine recent gaze context with voice commands."""

    def __init__(
        self,
        *,
        broker: CommandBroker | None = None,
        adapter: PlatformAdapter | None = None,
        voice_interpreter: VoiceIntentInterpreter | None = None,
        fusion_resolver: FusionIntentResolver | None = None,
        transcriber: ScriptedTranscriber | None = None,
        state: SharedInteractionState | None = None,
        metrics: FusionMetrics | None = None,
        min_recognition_confidence: float = 0.6,
        pending_timeout_ms: int = 15_000,
        max_confirmation_gaze_distance: float = 0.08,
    ) -> None:
        self.broker = broker or CommandBroker()
        self.adapter = adapter or MacOSPlatformAdapter(dry_run=True)
        self.voice_interpreter = voice_interpreter or VoiceIntentInterpreter()
        self.fusion_resolver = fusion_resolver or FusionIntentResolver()
        self.transcriber = transcriber or ScriptedTranscriber()
        self.state = state or SharedInteractionState()
        self.metrics = metrics or FusionMetrics()
        self.min_recognition_confidence = min_recognition_confidence
        self.pending_timeout_ms = max(1, pending_timeout_ms)
        self.max_confirmation_gaze_distance = max(0.0, max_confirmation_gaze_distance)
        self.pending_decision = None

    @property
    def pending_decision(self) -> BrokerDecision | None:
        return self._pending_decision

    @pending_decision.setter
    def pending_decision(self, decision: BrokerDecision | None) -> None:
        self._pending_decision = decision
        self._pending_action: PendingAction | None = None

    def _set_pending(self, decision: BrokerDecision, environment: EnvironmentSnapshot) -> None:
        self.pending_decision = decision
        self._pending_action = PendingAction(
            decision=decision,
            environment=environment.model_copy(deep=True),
            created_at_ms=self.state.now_ms,
            gaze_revision=self.state.gaze_revision,
        )

    def _pending_invalid_reason(self, environment: EnvironmentSnapshot) -> str | None:
        if self.pending_decision is None:
            return None
        if self._pending_action is None:
            return "The pending action has no captured context. Please repeat the command."
        target = self.state.latest_target()
        return self._pending_action.invalid_reason(
            now_ms=self.state.now_ms,
            timeout_ms=self.pending_timeout_ms,
            environment=environment,
            current_target_id=target.target_id if target is not None else None,
            observation=self.state.latest_observation(),
            gaze_revision=self.state.gaze_revision,
            max_gaze_distance=self.max_confirmation_gaze_distance,
        )

    def clear_gaze_context(self, *, observed_at_ms: int | None = None) -> FusionFeedbackEvent:
        self.state.clear_gaze(observed_at_ms=observed_at_ms)
        message = "Gaze tracking was lost. Look at the target to reacquire it."
        if self._pending_action is not None and self._pending_action.requires_gaze:
            self.pending_decision = None
            self.metrics.cancellations += 1
            message = "The pending pointer action was cancelled because gaze tracking was lost."
        return FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=message)

    def update_gaze_context(
        self,
        observation: GazeObservation,
        target: GroundedTarget | None,
        *,
        observed_at_ms: int | None = None,
    ) -> FusionFeedbackEvent:
        self.state.record_gaze(observation, target, observed_at_ms=observed_at_ms)
        if self._pending_action is not None and self._pending_action.requires_gaze:
            reason = self._pending_invalid_reason(self._pending_action.environment)
            if reason is not None:
                self.pending_decision = None
                self.metrics.cancellations += 1
                return FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=reason, target=self.state.latest_target())
        current_target = self.state.latest_target()
        message = f'Updated gaze context with "{current_target.label}".' if current_target else "No fresh grounded target was recorded."
        return FusionFeedbackEvent(
            phase=FusionLoopPhase.TRACKING,
            message=message,
            target=current_target,
        )

    def run_voice_turn(
        self,
        utterance: str,
        environment: EnvironmentSnapshot,
        *,
        final_confidence: float | None = 1.0,
    ) -> list[FusionFeedbackEvent]:
        self.metrics.total_turns += 1
        events: list[FusionFeedbackEvent] = []
        for segment in self.transcriber.stream(utterance, final_confidence=final_confidence):
            if not segment.is_final:
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.TRACKING, message="Partial fusion transcript update.", transcript=segment.text))
                continue
            events.extend(self._handle_final_transcript(segment.text, environment, transcript_confidence=segment.confidence))
        if not events:
            self.metrics.recoveries += 1
            events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message="I did not receive any multimodal command input."))
            events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
        return events

    def _handle_final_transcript(
        self,
        text: str,
        environment: EnvironmentSnapshot,
        *,
        transcript_confidence: float | None = None,
    ) -> list[FusionFeedbackEvent]:
        latest_target = self.state.latest_target()
        events = [
            FusionFeedbackEvent(
                phase=FusionLoopPhase.INTERPRETING,
                message="Final transcript received for multimodal fusion.",
                transcript=text,
                target=latest_target,
            )
        ]
        self.state.record_transcript(text, confidence=transcript_confidence)
        directive = self.voice_interpreter.interpret(text)
        invalid_pending = self._pending_invalid_reason(environment)
        if invalid_pending is not None:
            self.pending_decision = None
            self.metrics.cancellations += 1
            events.append(FusionFeedbackEvent(
                phase=FusionLoopPhase.RECOVERING, transcript=text,
                message=invalid_pending, target=latest_target,
            ))

        if directive.kind not in {VoiceDirectiveType.CONFIRM, VoiceDirectiveType.CANCEL}:
            self.pending_decision = None
        if (
            directive.kind != VoiceDirectiveType.CANCEL
            and transcript_confidence is not None
            and transcript_confidence < self.min_recognition_confidence
        ):
            self.metrics.clarifications += 1
            events.append(FusionFeedbackEvent(
                phase=FusionLoopPhase.RECOVERING,
                transcript=text,
                message="I could not hear that clearly. Please repeat the command or say cancel.",
                target=latest_target,
            ))
            events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.CONFIRM:
            if self.pending_decision is None:
                self.metrics.recoveries += 1
                if invalid_pending is None:
                    events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message="There is no pending fused action to confirm.", transcript=text))
            else:
                allow_decision = self.broker.confirm(self.pending_decision)
                self.pending_decision = None
                events.extend(self._execute_allow_decision(allow_decision, environment))
            events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.CANCEL:
            if self.pending_decision is not None:
                self.pending_decision = None
                self.metrics.cancellations += 1
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message="Pending fused action cancelled.", transcript=text))
            else:
                self.metrics.recoveries += 1
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message="Nothing is pending to cancel.", transcript=text))
            events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
            return events

        fused_confidence = self._fused_confidence(
            transcript_confidence=transcript_confidence,
            latest_gaze_confidence=self.state.latest_gaze_confidence(),
            latest_target=latest_target,
        )
        latest_observation = self.state.latest_observation() if self.state.latest_target_is_fresh() else None
        fusion = self.fusion_resolver.resolve(
            text,
            latest_target if self.state.latest_target_is_fresh() else None,
            observation=latest_observation,
            fused_confidence=fused_confidence,
            stale_reason=self.state.stale_reason(),
            candidate_targets=self.state.candidate_targets(limit=3),
        )
        if fusion.proposal is not None:
            decision = self.broker.decide(fusion.proposal)
            if decision.decision == BrokerDecisionType.ALLOW:
                events.extend(self._execute_allow_decision(decision, environment))
            elif decision.decision == BrokerDecisionType.CONFIRM:
                self._set_pending(decision, environment)
                self.metrics.confirmations += 1
                events.append(
                    FusionFeedbackEvent(
                        phase=FusionLoopPhase.CONFIRMING,
                        message=decision.confirmation_prompt or "Confirm this fused action?",
                        transcript=text,
                        target=latest_target,
                        decision=decision,
                    )
                )
            else:
                self.metrics.clarifications += 1
                events.append(
                    FusionFeedbackEvent(
                        phase=FusionLoopPhase.RECOVERING,
                        message=decision.reason,
                        transcript=text,
                        target=latest_target,
                        decision=decision,
                    )
                )
            if decision.decision != BrokerDecisionType.CONFIRM:
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
            return events

        if fusion.clarification is not None:
            self.metrics.clarifications += 1
            events.append(
                FusionFeedbackEvent(
                    phase=FusionLoopPhase.RECOVERING,
                    message=fusion.clarification,
                    transcript=text,
                    target=latest_target,
                )
            )
            events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.PROPOSAL and directive.proposal is not None:
            # Fall back to the voice-only path for non-deictic commands.
            decision = self.broker.decide(directive.proposal)
            if decision.decision == BrokerDecisionType.ALLOW:
                events.extend(self._execute_allow_decision(decision, environment))
            elif decision.decision == BrokerDecisionType.CONFIRM:
                self._set_pending(decision, environment)
                self.metrics.confirmations += 1
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.CONFIRMING, message=decision.confirmation_prompt or "Confirm this action?", transcript=text, decision=decision))
            else:
                self.metrics.recoveries += 1
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=decision.reason, transcript=text, decision=decision))
            if decision.decision != BrokerDecisionType.CONFIRM:
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
            return events

        self.metrics.recoveries += 1
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=directive.message or "The fused command could not be resolved.", transcript=text))
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
        return events

    def _execute_allow_decision(self, decision, environment: EnvironmentSnapshot) -> list[FusionFeedbackEvent]:
        self.metrics.executions += 1
        request = self.broker.build_execution_request(decision, environment)
        result = self.adapter.execute(request)
        return [
            FusionFeedbackEvent(
                phase=FusionLoopPhase.EXECUTING,
                message=decision.reason,
                target=self.state.latest_target(),
                decision=decision,
            ),
            FusionFeedbackEvent(
                phase=FusionLoopPhase.EXECUTING,
                message=result.message,
                target=self.state.latest_target(),
                decision=decision,
                result=result,
            ),
            FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."),
        ]

    def _fused_confidence(
        self,
        *,
        transcript_confidence: float | None,
        latest_gaze_confidence: float | None,
        latest_target: GroundedTarget | None,
    ) -> float:
        transcript_score = transcript_confidence if transcript_confidence is not None else 0.8
        if latest_target is None:
            return round(min(0.95, transcript_score), 3)
        gaze_score = latest_gaze_confidence if latest_gaze_confidence is not None else latest_target.confidence
        combined = 0.3 * transcript_score + 0.3 * gaze_score + 0.4 * latest_target.confidence
        return round(min(0.98, combined), 3)
