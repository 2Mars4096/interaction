"""Push-to-talk voice command loop."""

from __future__ import annotations

from interaction.audio import LiveSpeechClause, LiveSpeechStreamCapture, PushToTalkTurnManager, ScriptedTranscriber
from interaction.contracts import (
    ActionName,
    ActionProposal,
    BrokerDecisionType,
    EnvironmentSnapshot,
    ExecutionResult,
    ExecutionStatus,
    NavigationScope,
    RiskLevel,
    SessionMode,
    TranscriptSegment,
)
from interaction.control import CommandBroker
from interaction.feedback import VoiceFeedbackEvent, VoiceLoopPhase
from interaction.intent import (
    InterpreterStrategy,
    TranscriptHistoryItem,
    VoiceDirectiveType,
    VoiceIntentInterpreter,
    VoiceInterpretationContext,
    VoiceInterpretationRouter,
)
from interaction.platform.base import PlatformAdapter
from interaction.platform.macos import MacOSPlatformAdapter

from .state import SharedInteractionState


class VoiceCommandLoop:
    """Drive a push-to-talk voice session from transcript events to broker results."""

    def __init__(
        self,
        *,
        broker: CommandBroker | None = None,
        adapter: PlatformAdapter | None = None,
        router: VoiceInterpretationRouter | None = None,
        interpreter: VoiceIntentInterpreter | None = None,
        transcriber: ScriptedTranscriber | None = None,
        state: SharedInteractionState | None = None,
        default_mode: SessionMode = SessionMode.COMMAND,
        signal_only: bool = False,
    ) -> None:
        self.broker = broker or CommandBroker()
        self.adapter = adapter or MacOSPlatformAdapter(dry_run=True)
        self.router = router or VoiceInterpretationRouter(
            strategy=InterpreterStrategy.HYBRID,
            deterministic_interpreter=interpreter,
        )
        self.transcriber = transcriber or ScriptedTranscriber()
        self.state = state or SharedInteractionState()
        self.default_mode = default_mode
        self.signal_only = signal_only
        self.turn_manager = PushToTalkTurnManager()
        self.pending_decision = None
        self.dictated_clause_history: list[str] = []
        self.pending_dictation_history: list[str] | None = None

    def state_snapshot(self) -> dict[str, object]:
        active_scope = self.state.active_scope()
        latest_environment = self.state.latest_environment
        return {
            "dictated_clause_history": list(self.dictated_clause_history),
            "assembled_text": "".join(self.dictated_clause_history),
            "clause_count": len(self.dictated_clause_history),
            "pending_dictation_history": list(self.pending_dictation_history) if self.pending_dictation_history is not None else None,
            "has_pending_decision": self.pending_decision is not None,
            "active_scope": active_scope.value if active_scope is not None else None,
            "latest_environment": latest_environment.model_dump(mode="json") if latest_environment is not None else None,
        }

    def run_text_turn(
        self,
        utterance: str,
        environment: EnvironmentSnapshot,
        *,
        final_confidence: float | None = 1.0,
        begin_turn: bool = True,
        mode: SessionMode | None = None,
        fallback_to_dictation: bool = False,
    ) -> list[VoiceFeedbackEvent]:
        events: list[VoiceFeedbackEvent] = []
        effective_mode = mode or self.default_mode
        if begin_turn:
            events.append(self.begin_turn())
        segments = self.transcriber.stream(utterance, final_confidence=final_confidence)
        if not segments:
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, message="I did not receive any speech content."))
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events
        for segment in segments:
            events.extend(
                self.handle_segment(
                    segment,
                    environment,
                    mode=effective_mode,
                    fallback_to_dictation=fallback_to_dictation,
                )
            )
        return events

    def run_stream_capture(
        self,
        capture: LiveSpeechStreamCapture,
        environment: EnvironmentSnapshot,
        *,
        begin_turn: bool = True,
        mode: SessionMode | None = None,
        fallback_to_dictation: bool = False,
    ) -> list[VoiceFeedbackEvent]:
        events: list[VoiceFeedbackEvent] = []
        effective_mode = mode or self.default_mode
        self.state.record_environment(environment)
        if begin_turn:
            events.append(self.begin_turn())
        for partial in capture.partials:
            if partial.strip():
                events.append(
                    VoiceFeedbackEvent(
                        phase=VoiceLoopPhase.LISTENING,
                        transcript=partial,
                        message="Streaming clause update.",
                    )
                )
        for clause in capture.clauses:
            events.extend(
                self.handle_live_clause(
                    clause,
                    environment,
                    mode=effective_mode,
                    fallback_to_dictation=fallback_to_dictation,
                )
            )
        if not capture.clauses and capture.final_transcript.strip():
            events.extend(
                self._handle_final_utterance(
                    capture.final_transcript,
                    environment,
                    confidence=capture.confidence,
                    mode=effective_mode,
                    fallback_to_dictation=fallback_to_dictation,
                )
            )
        if not events:
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
        return events

    def begin_turn(self) -> VoiceFeedbackEvent:
        self.turn_manager.begin()
        return VoiceFeedbackEvent(
            phase=VoiceLoopPhase.LISTENING,
            message="Push-to-talk listening started.",
        )

    def handle_segment(
        self,
        segment: TranscriptSegment,
        environment: EnvironmentSnapshot,
        *,
        mode: SessionMode | None = None,
        fallback_to_dictation: bool = False,
    ) -> list[VoiceFeedbackEvent]:
        if not segment.is_final:
            return [
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.LISTENING,
                    transcript=segment.text,
                    message="Partial transcript update.",
                )
            ]
        return self._handle_final_utterance(
            segment.text,
            environment,
            confidence=segment.confidence,
            mode=mode or self.default_mode,
            fallback_to_dictation=fallback_to_dictation,
        )

    def handle_live_clause(
        self,
        clause: LiveSpeechClause,
        environment: EnvironmentSnapshot,
        *,
        mode: SessionMode | None = None,
        fallback_to_dictation: bool = False,
    ) -> list[VoiceFeedbackEvent]:
        if not clause.is_final:
            return [
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.LISTENING,
                    transcript=clause.transcript,
                    message="Streaming clause update.",
                )
            ]
        dedupe_key = f"clause:{clause.transcript.strip().lower()}"
        if self.state.is_duplicate_command(dedupe_key):
            return []
        self.state.record_command_key(dedupe_key)
        return self._handle_final_utterance(
            clause.transcript,
            environment,
            confidence=clause.confidence,
            mode=mode or self.default_mode,
            fallback_to_dictation=fallback_to_dictation,
        )

    def _handle_final_utterance(
        self,
        utterance: str,
        environment: EnvironmentSnapshot,
        *,
        confidence: float | None,
        mode: SessionMode,
        fallback_to_dictation: bool,
    ) -> list[VoiceFeedbackEvent]:
        self.turn_manager.end()
        self.state.record_environment(environment)
        self.state.record_transcript(utterance, confidence=confidence)
        events = [
            VoiceFeedbackEvent(
                phase=VoiceLoopPhase.INTERPRETING,
                transcript=utterance,
                message="Final transcript received.",
            )
        ]
        directive = self.router.interpret(
            utterance,
            self._interpretation_context(environment, mode=mode),
        )

        if mode == SessionMode.DICTATION and (self.pending_decision is None or directive.kind not in {VoiceDirectiveType.CONFIRM, VoiceDirectiveType.CANCEL}):
            events.extend(
                self._handle_dictation_text(
                    utterance,
                    environment,
                    auto_confirm=True,
                    message="Dictation lane treated the spoken clause as text input.",
                )
            )
            return events

        if directive.kind == VoiceDirectiveType.CONFIRM:
            if self.pending_decision is None:
                events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, transcript=utterance, message="There is no pending action to confirm."))
            else:
                allow_decision = self.broker.confirm(self.pending_decision)
                pending_dictation_history = self.pending_dictation_history
                self.pending_decision = None
                self.pending_dictation_history = None
                execution_events = self._execute_allow_decision(allow_decision, environment)
                events.extend(execution_events)
                self._commit_dictation_history_if_success(pending_dictation_history, execution_events)
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.CANCEL:
            if self.pending_decision is not None:
                self.pending_decision = None
                self.pending_dictation_history = None
                self.state.clear_active_scope()
                events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, transcript=utterance, message="Pending action cancelled."))
            else:
                events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, transcript=utterance, message="Nothing is pending to cancel."))
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.CLARIFY:
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, transcript=utterance, message=directive.message))
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        if directive.kind == VoiceDirectiveType.REJECT:
            if fallback_to_dictation:
                events.extend(
                    self._handle_dictation_text(
                        utterance,
                        environment,
                        auto_confirm=True,
                        message="Hybrid lane did not match a bounded command and fell back to dictation.",
                    )
                )
                return events
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, transcript=utterance, message=directive.message))
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        assert directive.proposal is not None
        if directive.proposal.action == ActionName.TRANSLATE_TEXT:
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.RECOVERING,
                    transcript=utterance,
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
                    transcript=utterance,
                    message=decision.confirmation_prompt,
                    decision=decision,
                )
            )
        else:
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.RECOVERING,
                    transcript=utterance,
                    message=decision.reason,
                    decision=decision,
                )
            )

        if decision.decision != BrokerDecisionType.CONFIRM:
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
        return events

    def _handle_dictation_text(
        self,
        text: str,
        environment: EnvironmentSnapshot,
        *,
        auto_confirm: bool,
        message: str,
    ) -> list[VoiceFeedbackEvent]:
        events: list[VoiceFeedbackEvent] = []
        edit_request = _parse_dictation_edit_request(
            text,
            allow_self_correction=bool(self.dictated_clause_history),
        )
        if edit_request is not None:
            normalized_text = _normalize_dictation_text(text)
        else:
            normalized_text = _normalize_dictation_text(
                text,
                sentence_start=_history_starts_sentence(self.dictated_clause_history),
            )
        if not normalized_text:
            self.turn_manager.reset()
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.RECOVERING,
                    transcript=text,
                    message="I did not receive any dictation text to type.",
                )
            )
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        edit_command = self._dictation_edit_command(edit_request)
        if edit_command is not None:
            delete_characters, replacement_text, description, next_history = edit_command
            if delete_characters <= 0:
                self.turn_manager.reset()
                events.append(
                    VoiceFeedbackEvent(
                        phase=VoiceLoopPhase.RECOVERING,
                        transcript=normalized_text,
                        message="There is no recent dictated clause to revise.",
                    )
                )
                events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
                return events
            if not getattr(self.adapter, "dry_run", False):
                self.dictated_clause_history.clear()
                self.pending_dictation_history = None
                self.turn_manager.reset()
                return [VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.RECOVERING, transcript=normalized_text,
                    message="Automatic text correction is disabled because the current field and caret cannot be verified. Edit the text manually.",
                ), VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle.")]
            proposal = ActionProposal(
                action=ActionName.TYPE_TEXT,
                arguments={
                    "delete_characters": delete_characters,
                    "text": replacement_text,
                },
                confidence=0.9,
                risk=RiskLevel.L2,
                requires_confirmation=not auto_confirm,
                rationale="The dictation lane revised the most recent dictated clause in place.",
            )
            decision = self.broker.decide(proposal)
            if auto_confirm and decision.decision == BrokerDecisionType.CONFIRM:
                decision = self.broker.confirm(decision)
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.INTERPRETING,
                    transcript=normalized_text,
                    message=description,
                    decision=decision,
                )
            )
            if decision.decision == BrokerDecisionType.ALLOW:
                execution_events = self._execute_allow_decision(decision, environment)
                events.extend(execution_events)
                self._commit_dictation_history_if_success(next_history, execution_events)
                self.turn_manager.reset()
                events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
                return events
            if decision.decision == BrokerDecisionType.CONFIRM:
                self.pending_decision = decision
                self.pending_dictation_history = next_history
                events.append(
                    VoiceFeedbackEvent(
                        phase=VoiceLoopPhase.CONFIRMING,
                        transcript=normalized_text,
                        message=decision.confirmation_prompt,
                        decision=decision,
                    )
                )
                return events
            self.turn_manager.reset()
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.RECOVERING,
                    transcript=normalized_text,
                    message=decision.reason,
                    decision=decision,
                )
            )
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        if _is_duplicate_dictation_clause(normalized_text, self.dictated_clause_history):
            self.turn_manager.reset()
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.INTERPRETING,
                    transcript=normalized_text,
                    message="Dictation lane ignored a duplicated clause.",
                )
            )
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events

        prepared_text = self._prepare_dictation_clause(normalized_text)
        proposal = ActionProposal(
            action=ActionName.TYPE_TEXT,
            arguments={"text": prepared_text},
            confidence=0.9,
            risk=RiskLevel.L2,
            requires_confirmation=not auto_confirm,
            rationale="The voice lane routed the spoken clause into text insertion for the active field.",
        )
        decision = self.broker.decide(proposal)
        if auto_confirm and decision.decision == BrokerDecisionType.CONFIRM:
            decision = self.broker.confirm(decision)

        events.append(
            VoiceFeedbackEvent(
                phase=VoiceLoopPhase.INTERPRETING,
                transcript=normalized_text,
                message=message,
                decision=decision,
            )
        )

        if decision.decision == BrokerDecisionType.ALLOW:
            execution_events = self._execute_allow_decision(decision, environment)
            events.extend(execution_events)
            self._commit_dictation_history_if_success(self.dictated_clause_history + [prepared_text], execution_events)
            self.turn_manager.reset()
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
            return events
        if decision.decision == BrokerDecisionType.CONFIRM:
            self.pending_decision = decision
            self.pending_dictation_history = self.dictated_clause_history + [prepared_text]
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.CONFIRMING,
                    transcript=normalized_text,
                    message=decision.confirmation_prompt,
                    decision=decision,
                )
            )
            return events

        recovery_message = decision.reason if decision.reason else "The dictation clause could not be executed."
        self.turn_manager.reset()
        events.append(
            VoiceFeedbackEvent(
                phase=VoiceLoopPhase.RECOVERING,
                transcript=normalized_text,
                message=recovery_message,
                decision=decision,
            )
        )
        events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
        return events

    def _dictation_edit_command(self, edit_request: tuple[str, str, str] | None) -> tuple[int, str, str, list[str]] | None:
        if edit_request is None:
            return None
        command, target, replacement_raw = edit_request
        delete_characters, base_history = _edit_span_from_history(self.dictated_clause_history, target)
        if command == "delete":
            return (
                delete_characters,
                "",
                _edit_description(command, target),
                base_history,
            )
        if command != "replace":
            return None
        replacement_text = _normalize_dictation_text(
            replacement_raw,
            sentence_start=_history_starts_sentence(base_history),
        )
        prepared_replacement_text = self._prepare_dictation_clause(replacement_text, history=base_history)
        return (
            delete_characters,
            prepared_replacement_text,
            _edit_description(command, target),
            base_history + ([prepared_replacement_text] if prepared_replacement_text else []),
        )

    def _commit_dictation_history_if_success(
        self,
        next_history: list[str] | None,
        execution_events: list[VoiceFeedbackEvent],
    ) -> None:
        if next_history is None:
            return
        for event in execution_events:
            if event.result is not None and (
                event.result.status == "success"
                or (
                    event.result.status == ExecutionStatus.BLOCKED
                    and event.result.details.get("signal_only") is True
                )
            ):
                self.dictated_clause_history = list(next_history)
                return

    def _prepare_dictation_clause(self, normalized_text: str, *, history: list[str] | None = None) -> str:
        if not normalized_text:
            return normalized_text
        effective_history = self.dictated_clause_history if history is None else history
        if not effective_history:
            return normalized_text
        previous_clause = effective_history[-1]
        if normalized_text.startswith("\n") or previous_clause.endswith("\n") or previous_clause.endswith("\n\n"):
            return normalized_text
        return f" {normalized_text}"

    def _execute_allow_decision(self, decision, environment: EnvironmentSnapshot) -> list[VoiceFeedbackEvent]:
        if self.signal_only:
            self._remember_successful_action(decision.proposal)
            return [
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.EXECUTING,
                    message=decision.reason,
                    decision=decision,
                ),
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.EXECUTING,
                    message="Signal-only mode skipped platform execution.",
                    decision=decision,
                    result=ExecutionResult(
                        status=ExecutionStatus.BLOCKED,
                        message="Signal-only mode skipped platform execution.",
                        proposal=decision.proposal,
                        details={"signal_only": True},
                    ),
                ),
            ]
        request = self.broker.build_execution_request(decision, environment)
        result = self.adapter.execute(request)
        if result.status == ExecutionStatus.SUCCESS:
            self._remember_successful_action(decision.proposal)
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

    def _interpretation_context(
        self,
        environment: EnvironmentSnapshot,
        *,
        mode: SessionMode,
    ) -> VoiceInterpretationContext:
        return VoiceInterpretationContext(
            mode=mode,
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

    def _remember_successful_action(self, proposal) -> None:
        if proposal is None:
            return
        command_key = f"proposal:{proposal.action.value}:{proposal.arguments}"
        self.state.record_command_key(command_key)
        if proposal.action != ActionName.NAVIGATE_STRUCTURE:
            return
        scope_value = proposal.arguments.get("scope")
        if not isinstance(scope_value, str):
            return
        try:
            self.state.set_active_scope(NavigationScope(scope_value))
        except ValueError:
            return


def _normalize_dictation_text(text: str, *, sentence_start: bool = True) -> str:
    tokens = _strip_leading_fillers(text.strip().split())
    if not tokens:
        return ""
    normalized_tokens: list[tuple[str, str]] = []
    last_word_lowered: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lowered = token.lower()
        next_lowered = tokens[index + 1].lower() if index + 1 < len(tokens) else None

        phrase_type: tuple[str, str] | None = None
        phrase_advance = 1
        if lowered == "question" and next_lowered == "mark":
            phrase_type = ("punct", "?")
            phrase_advance = 2
        elif lowered == "exclamation" and next_lowered == "mark":
            phrase_type = ("punct", "!")
            phrase_advance = 2
        elif lowered == "full" and next_lowered == "stop":
            phrase_type = ("punct", ".")
            phrase_advance = 2
        elif lowered == "new" and next_lowered == "line":
            phrase_type = ("newline", "\n")
            phrase_advance = 2
        elif lowered == "new" and next_lowered == "paragraph":
            phrase_type = ("newline", "\n\n")
            phrase_advance = 2
        elif lowered == "open" and next_lowered == "quote":
            phrase_type = ("punct", '"')
            phrase_advance = 2
        elif lowered == "close" and next_lowered == "quote":
            phrase_type = ("punct", '"')
            phrase_advance = 2
        elif lowered == "open" and next_lowered in {"parenthesis", "paren"}:
            phrase_type = ("punct", "(")
            phrase_advance = 2
        elif lowered == "close" and next_lowered in {"parenthesis", "paren"}:
            phrase_type = ("punct", ")")
            phrase_advance = 2
        elif lowered == "open" and next_lowered == "bracket":
            phrase_type = ("punct", "[")
            phrase_advance = 2
        elif lowered == "close" and next_lowered == "bracket":
            phrase_type = ("punct", "]")
            phrase_advance = 2
        elif lowered == "new" and next_lowered == "sentence":
            phrase_type = ("punct", ".")
            phrase_advance = 2
        elif lowered in {"comma", "period", "colon", "semicolon"}:
            phrase_type = ("punct", {"comma": ",", "period": ".", "colon": ":", "semicolon": ";"}[lowered])
        elif lowered in {"dash", "hyphen"}:
            phrase_type = ("punct", "-")

        if phrase_type is not None:
            normalized_tokens.append(phrase_type)
            last_word_lowered = None
            index += phrase_advance
            continue

        if normalized_tokens and lowered == last_word_lowered:
            index += 1
            continue

        normalized_tokens.append(("word", token))
        last_word_lowered = lowered
        index += 1

    normalized_tokens = _collapse_repeated_word_phrases(normalized_tokens)
    return _render_dictation_tokens(normalized_tokens, sentence_start=sentence_start)


def _parse_dictation_edit_request(text: str, *, allow_self_correction: bool = False) -> tuple[str, str, str] | None:
    cleaned_tokens = _strip_leading_fillers(text.strip().split())
    if not cleaned_tokens:
        return None
    cleaned_text = " ".join(cleaned_tokens)
    lowered = cleaned_text.lower()
    if lowered in {"scratch that", "delete that", "undo that"}:
        return ("delete", "clause", "")
    if lowered in {"delete last sentence", "scratch last sentence", "delete the last sentence", "scratch the last sentence"}:
        return ("delete", "sentence", "")
    for prefix in ("scratch that ", "replace that with "):
        if lowered.startswith(prefix):
            return ("replace", "clause", cleaned_text[len(prefix):].strip())
    for prefix in ("replace last sentence with ", "replace the last sentence with "):
        if lowered.startswith(prefix):
            return ("replace", "sentence", cleaned_text[len(prefix):].strip())
    if allow_self_correction:
        for prefix in ("i mean ", "or rather "):
            if lowered.startswith(prefix):
                return ("replace", "clause", cleaned_text[len(prefix):].strip())
    return None


def _edit_span_from_history(history: list[str], target: str) -> tuple[int, list[str]]:
    if not history:
        return 0, []
    if target == "sentence":
        start_index = _last_sentence_start_index(history)
        return sum(len(clause) for clause in history[start_index:]), history[:start_index]
    return len(history[-1]), history[:-1]


def _last_sentence_start_index(history: list[str]) -> int:
    if not history:
        return 0
    index = len(history) - 1
    while index > 0 and not _clause_ends_sentence(history[index - 1]):
        index -= 1
    return index


def _edit_description(command: str, target: str) -> str:
    target_label = "most recent dictated clause" if target == "clause" else "most recent dictated sentence"
    verb = "removed" if command == "delete" else "replaced"
    return f"Dictation lane {verb} the {target_label}."


def _history_starts_sentence(history: list[str]) -> bool:
    if not history:
        return True
    return _clause_ends_sentence(history[-1])


def _clause_ends_sentence(text: str) -> bool:
    if not text:
        return True
    trimmed = text.rstrip()
    if not trimmed:
        return True
    index = len(trimmed) - 1
    while index >= 0 and trimmed[index] in {'"', "'", ")", "]"}:
        index -= 1
    if index < 0:
        return True
    return trimmed[index] in {".", "!", "?", "\n"}


def _is_duplicate_dictation_clause(normalized_text: str, history: list[str]) -> bool:
    if not normalized_text or not history:
        return False
    previous_clause = history[-1].lstrip()
    return previous_clause.casefold() == normalized_text.casefold()


def _strip_leading_fillers(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    lowered_tokens = [token.lower() for token in tokens]
    filler_phrases = (
        ("um",),
        ("uh",),
        ("erm",),
        ("ah",),
        ("hmm",),
        ("mm",),
        ("mhm",),
        ("you", "know"),
    )
    index = 0
    while index < len(lowered_tokens):
        matched = False
        for filler in filler_phrases:
            end_index = index + len(filler)
            if tuple(lowered_tokens[index:end_index]) == filler:
                index = end_index
                matched = True
                break
        if not matched:
            break
    return tokens[index:]


def _collapse_repeated_word_phrases(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not tokens:
        return []
    collapsed: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens):
        if tokens[index][0] != "word":
            collapsed.append(tokens[index])
            index += 1
            continue
        phrase_length = 0
        for offset in range(index, min(len(tokens), index + 3)):
            if tokens[offset][0] != "word":
                break
            phrase_length += 1
        collapsed_repeat = False
        for repeat_length in range(min(3, phrase_length), 1, -1):
            next_index = index + repeat_length
            repeat_end_index = next_index + repeat_length
            if repeat_end_index > len(tokens):
                continue
            if any(kind != "word" for kind, _ in tokens[index:repeat_end_index]):
                continue
            first_phrase = [value.lower() for kind, value in tokens[index:next_index]]
            second_phrase = [value.lower() for kind, value in tokens[next_index:repeat_end_index]]
            if first_phrase == second_phrase:
                collapsed.extend(tokens[index:next_index])
                index = repeat_end_index
                collapsed_repeat = True
                break
        if collapsed_repeat:
            continue
        collapsed.append(tokens[index])
        index += 1
    return collapsed


def _render_dictation_tokens(tokens: list[tuple[str, str]], *, sentence_start: bool = True) -> str:
    if not tokens:
        return ""
    parts: list[str] = []
    at_sentence_start = sentence_start
    for kind, value in tokens:
        if kind == "newline":
            if parts and parts[-1].endswith(" "):
                parts[-1] = parts[-1].rstrip()
            parts.append(value)
            at_sentence_start = True
            continue
        if kind == "punct":
            if not parts:
                parts.append(value)
            else:
                parts[-1] = parts[-1].rstrip() + value
            if value in {".", "!", "?"}:
                at_sentence_start = True
            continue

        word = value
        if at_sentence_start:
            word = _capitalize_token(word)
            at_sentence_start = False
        if not parts:
            parts.append(word)
            continue
        previous = parts[-1]
        if previous.endswith("\n") or previous.endswith("\n\n") or previous.endswith('"') or previous.endswith("(") or previous.endswith("["):
            parts.append(word)
        else:
            parts.append(f" {word}")
    rendered = "".join(parts)
    rendered = rendered.replace(" \n", "\n").replace(" \n\n", "\n\n")
    return rendered.strip()


def _capitalize_token(token: str) -> str:
    for index, character in enumerate(token):
        if character.isalpha():
            return f"{token[:index]}{character.upper()}{token[index + 1:]}"
    return token
