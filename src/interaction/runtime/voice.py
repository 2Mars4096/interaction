"""Push-to-talk voice command loop."""

from __future__ import annotations

from interaction.audio import PushToTalkTurnManager, ScriptedTranscriber
from interaction.contracts import ActionName, BrokerDecisionType, EnvironmentSnapshot, TranscriptSegment
from interaction.control import CommandBroker
from interaction.feedback import VoiceFeedbackEvent, VoiceLoopPhase
from interaction.intent import VoiceDirectiveType, VoiceIntentInterpreter
from interaction.platform import MacOSPlatformAdapter, PlatformAdapter


class VoiceCommandLoop:
    """Drive a push-to-talk voice session from transcript events to broker results."""

    def __init__(
        self,
        *,
        broker: CommandBroker | None = None,
        adapter: PlatformAdapter | None = None,
        interpreter: VoiceIntentInterpreter | None = None,
        transcriber: ScriptedTranscriber | None = None,
    ) -> None:
        self.broker = broker or CommandBroker()
        self.adapter = adapter or MacOSPlatformAdapter(dry_run=True)
        self.interpreter = interpreter or VoiceIntentInterpreter()
        self.transcriber = transcriber or ScriptedTranscriber()
        self.turn_manager = PushToTalkTurnManager()
        self.pending_decision = None

    def run_text_turn(
        self,
        utterance: str,
        environment: EnvironmentSnapshot,
        *,
        final_confidence: float | None = 1.0,
        begin_turn: bool = True,
    ) -> list[VoiceFeedbackEvent]:
        events: list[VoiceFeedbackEvent] = []
        if begin_turn:
            events.append(self.begin_turn())
        segments = self.transcriber.stream(utterance, final_confidence=final_confidence)
        if not segments:
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, message="I did not receive any speech content."))
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events
        for segment in segments:
            events.extend(self.handle_segment(segment, environment))
        return events

    def begin_turn(self) -> VoiceFeedbackEvent:
        self.turn_manager.begin()
        return VoiceFeedbackEvent(
            phase=VoiceLoopPhase.LISTENING,
            message="Push-to-talk listening started.",
        )

    def handle_segment(self, segment: TranscriptSegment, environment: EnvironmentSnapshot) -> list[VoiceFeedbackEvent]:
        events: list[VoiceFeedbackEvent] = []
        if not segment.is_final:
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.LISTENING,
                    transcript=segment.text,
                    message="Partial transcript update.",
                )
            )
            return events

        self.turn_manager.end()
        events.append(
            VoiceFeedbackEvent(
                phase=VoiceLoopPhase.INTERPRETING,
                transcript=segment.text,
                message="Final transcript received.",
            )
        )
        directive = self.interpreter.interpret(segment.text)

        if directive.kind == VoiceDirectiveType.CONFIRM:
            if self.pending_decision is None:
                events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, transcript=segment.text, message="There is no pending action to confirm."))
            else:
                allow_decision = self.broker.confirm(self.pending_decision)
                self.pending_decision = None
                events.extend(self._execute_allow_decision(allow_decision, environment))
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.CANCEL:
            if self.pending_decision is not None:
                self.pending_decision = None
                events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, transcript=segment.text, message="Pending action cancelled."))
            else:
                events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, transcript=segment.text, message="Nothing is pending to cancel."))
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.CLARIFY:
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, transcript=segment.text, message=directive.message))
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.REJECT:
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, transcript=segment.text, message=directive.message))
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        assert directive.proposal is not None
        if directive.proposal.action == ActionName.TRANSLATE_TEXT:
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.RECOVERING,
                    transcript=segment.text,
                    message="Translation is recognized, but no local translation provider is configured yet.",
                )
            )
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events
        decision = self.broker.decide(directive.proposal)
        if decision.decision == BrokerDecisionType.ALLOW:
            events.extend(self._execute_allow_decision(decision, environment))
        elif decision.decision == BrokerDecisionType.CONFIRM:
            self.pending_decision = decision
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.CONFIRMING,
                    transcript=segment.text,
                    message=decision.confirmation_prompt,
                    decision=decision,
                )
            )
        else:
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.RECOVERING,
                    transcript=segment.text,
                    message=decision.reason,
                    decision=decision,
                )
            )

        if decision.decision != BrokerDecisionType.CONFIRM:
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
        return events

    def _execute_allow_decision(self, decision, environment: EnvironmentSnapshot) -> list[VoiceFeedbackEvent]:
        request = self.broker.build_execution_request(decision, environment)
        result = self.adapter.execute(request)
        return [
            VoiceFeedbackEvent(
                phase=VoiceLoopPhase.EXECUTING,
                message=decision.reason,
                decision=decision,
            ),
            VoiceFeedbackEvent(
                phase=VoiceLoopPhase.EXECUTING,
                message=result.message,
                decision=decision,
                result=result,
            ),
        ]
