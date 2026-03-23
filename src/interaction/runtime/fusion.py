"""Shared-state multimodal fusion loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from interaction.audio import ScriptedTranscriber
from interaction.contracts import BrokerDecisionType, EnvironmentSnapshot, GazeObservation, GroundedTarget
from interaction.control import CommandBroker
from interaction.feedback import FusionFeedbackEvent, FusionLoopPhase
from interaction.intent import FusionIntentResolver, VoiceDirectiveType, VoiceIntentInterpreter
from interaction.platform import MacOSPlatformAdapter, PlatformAdapter

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
    ) -> None:
        self.broker = broker or CommandBroker()
        self.adapter = adapter or MacOSPlatformAdapter(dry_run=True)
        self.voice_interpreter = voice_interpreter or VoiceIntentInterpreter()
        self.fusion_resolver = fusion_resolver or FusionIntentResolver()
        self.transcriber = transcriber or ScriptedTranscriber()
        self.state = state or SharedInteractionState()
        self.metrics = metrics or FusionMetrics()
        self.pending_decision = None

    def update_gaze_context(self, observation: GazeObservation, target: GroundedTarget | None) -> FusionFeedbackEvent:
        self.state.record_gaze(observation, target)
        message = f'Updated gaze context with "{target.label}".' if target else "No grounded target was recorded."
        return FusionFeedbackEvent(
            phase=FusionLoopPhase.TRACKING,
            message=message,
            target=target,
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

        if directive.kind == VoiceDirectiveType.CONFIRM:
            if self.pending_decision is None:
                self.metrics.recoveries += 1
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
                self.pending_decision = decision
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
                self.pending_decision = decision
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
