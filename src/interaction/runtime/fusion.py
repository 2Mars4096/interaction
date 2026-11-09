"""Shared-state multimodal fusion loop."""

from __future__ import annotations

from dataclasses import dataclass

from interaction.audio import ScriptedTranscriber
from interaction.contracts import (
    ActionName,
    BrokerDecisionType,
    EnvironmentSnapshot,
    ExecutionStatus,
    GazeObservation,
    GroundedTarget,
    NavigationScope,
    SessionMode,
)
from interaction.control import CommandBroker
from interaction.feedback import FusionFeedbackEvent, FusionLoopPhase
from interaction.intent import (
    FusionIntentResolver,
    FusionResolution,
    InterpreterStrategy,
    TranscriptHistoryItem,
    VoiceDirectiveType,
    VoiceInterpretationContext,
    VoiceInterpretationRouter,
)
from interaction.platform.base import PlatformAdapter
from interaction.platform.macos import MacOSPlatformAdapter

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
        router: VoiceInterpretationRouter | None = None,
        fusion_resolver: FusionIntentResolver | None = None,
        transcriber: ScriptedTranscriber | None = None,
        state: SharedInteractionState | None = None,
        metrics: FusionMetrics | None = None,
    ) -> None:
        self.broker = broker or CommandBroker()
        self.adapter = adapter or MacOSPlatformAdapter(dry_run=True)
        self.router = router or VoiceInterpretationRouter(strategy=InterpreterStrategy.HYBRID)
        self.fusion_resolver = fusion_resolver or FusionIntentResolver()
        self.transcriber = transcriber or ScriptedTranscriber()
        self.state = state or SharedInteractionState()
        self.metrics = metrics or FusionMetrics()
        self.pending_decision = None
        self.pending_environment: EnvironmentSnapshot | None = None
        self.pending_target: GroundedTarget | None = None
        self.pending_observation: GazeObservation | None = None

    def state_snapshot(self) -> dict[str, object]:
        active_scope = self.state.active_scope()
        latest_environment = self.state.latest_environment
        latest_target = self.state.latest_target()
        return {
            "has_pending_decision": self.pending_decision is not None,
            "active_scope": active_scope.value if active_scope is not None else None,
            "latest_environment": latest_environment.model_dump(mode="json") if latest_environment is not None else None,
            "latest_target": latest_target.model_dump(mode="json") if latest_target is not None else None,
            "candidate_targets": [target.model_dump(mode="json") for target in self.state.candidate_targets(limit=3)],
        }

    def update_gaze_context(
        self,
        observation: GazeObservation,
        target: GroundedTarget | None,
        *,
        candidate_targets: list[GroundedTarget] | None = None,
    ) -> FusionFeedbackEvent:
        self.state.record_gaze(observation, target, candidate_targets=candidate_targets)
        primary_target = candidate_targets[0] if candidate_targets else target
        message = f'Updated gaze context with "{primary_target.label}".' if primary_target else "No grounded target was recorded."
        return FusionFeedbackEvent(
            phase=FusionLoopPhase.TRACKING,
            message=message,
            target=primary_target,
        )

    def run_voice_turn(
        self,
        utterance: str,
        environment: EnvironmentSnapshot,
        *,
        final_confidence: float | None = 1.0,
    ) -> list[FusionFeedbackEvent]:
        self.metrics.total_turns += 1
        self.state.record_environment(environment)
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
        self.state.record_environment(environment)
        self.state.record_transcript(text, confidence=transcript_confidence)
        directive = self.router.interpret(text, self._interpretation_context(environment))

        if directive.kind == VoiceDirectiveType.CONFIRM:
            if self.pending_decision is None:
                self.metrics.recoveries += 1
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message="There is no pending fused action to confirm.", transcript=text))
            elif (reason := self._pending_context_error(environment)) is not None:
                self.pending_decision = None
                self.metrics.cancellations += 1
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=reason, transcript=text))
            else:
                allow_decision = self.broker.confirm(self.pending_decision)
                self.pending_decision = None
                events.extend(self._execute_allow_decision(allow_decision, environment))
            events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.CANCEL:
            if self.pending_decision is not None:
                self.pending_decision = None
                self.state.clear_active_scope()
                self.metrics.cancellations += 1
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message="Pending fused action cancelled.", transcript=text))
            else:
                self.metrics.recoveries += 1
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message="Nothing is pending to cancel.", transcript=text))
            events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.PROPOSAL and directive.proposal is not None:
            events.extend(self._handle_decision(directive.proposal, environment, transcript=text, target=latest_target))
            return events

        fusion = self._resolve_fusion_fallback(text, transcript_confidence=transcript_confidence)
        if fusion.proposal is not None:
            events.extend(self._handle_decision(fusion.proposal, environment, transcript=text, target=latest_target))
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

        self.metrics.recoveries += 1
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=directive.message or "The fused command could not be resolved.", transcript=text))
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
        return events

    def _interpretation_context(self, environment: EnvironmentSnapshot) -> VoiceInterpretationContext:
        return VoiceInterpretationContext(
            mode=SessionMode.COMMAND,
            environment=environment,
            active_scope=self.state.active_scope(),
            transcript_history=tuple(
                TranscriptHistoryItem(utterance=entry.utterance, confidence=entry.confidence)
                for entry in self.state.recent_transcripts(limit=4)
            ),
            latest_target=self.state.latest_target() if self.state.latest_target_is_fresh() else None,
            candidate_targets=tuple(self.state.candidate_targets(limit=3)),
            latest_observation=self.state.latest_observation() if self.state.latest_target_is_fresh() else None,
            pending_confirmation=self.pending_decision is not None,
        )

    def _resolve_fusion_fallback(
        self,
        text: str,
        *,
        transcript_confidence: float | None,
    ) -> FusionResolution:
        latest_target = self.state.latest_target()
        return self.fusion_resolver.resolve(
            text,
            latest_target if self.state.latest_target_is_fresh() else None,
            observation=self.state.latest_observation() if self.state.latest_target_is_fresh() else None,
            fused_confidence=self._fused_confidence(
                transcript_confidence=transcript_confidence,
                latest_gaze_confidence=self.state.latest_gaze_confidence(),
                latest_target=latest_target,
            ),
            stale_reason=self.state.stale_reason(),
            candidate_targets=self.state.candidate_targets(limit=3),
        )

    def _handle_decision(
        self,
        proposal,
        environment: EnvironmentSnapshot,
        *,
        transcript: str,
        target: GroundedTarget | None,
    ) -> list[FusionFeedbackEvent]:
        decision = self.broker.decide(proposal)
        if decision.decision == BrokerDecisionType.ALLOW:
            return self._execute_allow_decision(decision, environment)
        if decision.decision == BrokerDecisionType.CONFIRM:
            self.pending_decision = decision
            self.pending_environment = environment
            self.pending_target = self.state.latest_target()
            self.pending_observation = self.state.latest_observation()
            self.metrics.confirmations += 1
            return [
                FusionFeedbackEvent(
                    phase=FusionLoopPhase.CONFIRMING,
                    message=decision.confirmation_prompt or "Confirm this fused action?",
                    transcript=transcript,
                    target=target,
                    decision=decision,
                )
            ]
        self.metrics.clarifications += 1
        return [
            FusionFeedbackEvent(
                phase=FusionLoopPhase.RECOVERING,
                message=decision.reason,
                transcript=transcript,
                target=target,
                decision=decision,
            ),
            FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."),
        ]

    def _pending_context_error(self, environment: EnvironmentSnapshot) -> str | None:
        previous = self.pending_environment
        if previous is not None and (
            previous.active_app, previous.active_bundle_id, previous.active_window_title
        ) != (environment.active_app, environment.active_bundle_id, environment.active_window_title):
            return "The active app or window changed. Pending action cancelled; repeat your command."
        proposal = self.pending_decision.proposal
        if not proposal.arguments.get("target_ref"):
            return None
        target = self.state.latest_target()
        old = self.pending_target
        if target is None or old is None or target.target_id != old.target_id:
            return "The gaze target expired or changed. Pending action cancelled; look at the intended target and repeat your command."
        candidate_fields = {"candidate_id", "source", "label", "role", "screen_point", "screen_bounds", "normalized_bounds", "enabled", "pid", "window_id", "window_title"}
        old_candidate = old.candidate.model_dump(include=candidate_fields) if old.candidate else None
        new_candidate = target.candidate.model_dump(include=candidate_fields) if target.candidate else None
        if target.bounds != old.bounds or new_candidate != old_candidate:
            return "The desktop target moved or changed. Pending action cancelled; repeat your command."
        observation = self.state.latest_observation()
        before = self.pending_observation
        if before is not None and observation is not None:
            if all(v is not None for v in (before.x_norm, before.y_norm, observation.x_norm, observation.y_norm)):
                distance = ((before.x_norm - observation.x_norm) ** 2 + (before.y_norm - observation.y_norm) ** 2) ** 0.5
                if distance > 0.06:
                    return "Your gaze moved away from the proposed point. Pending action cancelled; repeat your command."
        return None

    def _execute_allow_decision(self, decision, environment: EnvironmentSnapshot) -> list[FusionFeedbackEvent]:
        self.metrics.executions += 1
        request = self.broker.build_execution_request(decision, environment)
        result = self.adapter.execute(request)
        if result.status == ExecutionStatus.SUCCESS:
            self._remember_successful_action(decision.proposal)
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

    def _remember_successful_action(self, proposal) -> None:
        if proposal is None or proposal.action != ActionName.NAVIGATE_STRUCTURE:
            return
        scope_value = proposal.arguments.get("scope")
        if not isinstance(scope_value, str):
            return
        try:
            self.state.set_active_scope(NavigationScope(scope_value))
        except ValueError:
            return

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
