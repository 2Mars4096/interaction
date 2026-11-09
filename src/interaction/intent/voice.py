"""Pattern-based voice interpretation for the Phase 2 voice loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from interaction.contracts import ActionName, ActionProposal, NormalizedIntent, RiskLevel

_DEICTIC_TOKENS = {"this", "that", "these", "those", "here", "there", "it", "them"}


class VoiceDirectiveType(StrEnum):
    PROPOSAL = "proposal"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    CLARIFY = "clarify"
    REJECT = "reject"


@dataclass(frozen=True)
class VoiceDirective:
    kind: VoiceDirectiveType
    utterance: str
    intent: NormalizedIntent | None = None
    proposal: ActionProposal | None = None
    message: str | None = None
    source: str | None = None


def normalize_utterance(utterance: str) -> str:
    return " ".join(utterance.strip().split())


def repair_command_utterance(utterance: str) -> str:
    text = normalize_utterance(utterance)
    if not text:
        return text

    tokens = text.split(" ")
    if len(tokens) >= 2 and tokens[0].lower() == "crawl" and tokens[1].lower() in {"up", "down"}:
        tokens[0] = "scroll"
        return " ".join(tokens)
    if len(tokens) >= 2 and tokens[0].lower() == "flick" and tokens[1].lower() in _DEICTIC_TOKENS:
        tokens[0] = "click"
        return " ".join(tokens)
    return text


def match_review_token(
    utterance: str,
    *,
    pending_confirmation: bool,
    allow_without_pending: bool = True,
) -> VoiceDirective | None:
    lowered = normalize_utterance(utterance).lower()
    if not lowered:
        return None
    if lowered in VoiceIntentInterpreter._CONFIRM_TOKENS:
        if pending_confirmation or allow_without_pending:
            return VoiceDirective(kind=VoiceDirectiveType.CONFIRM, utterance=normalize_utterance(utterance), source="regex")
        return None
    if lowered in VoiceIntentInterpreter._CANCEL_TOKENS:
        if pending_confirmation or allow_without_pending:
            return VoiceDirective(kind=VoiceDirectiveType.CANCEL, utterance=normalize_utterance(utterance), source="regex")
        return None
    return None


class VoiceIntentInterpreter:
    """Interpret typed or transcribed speech into bounded actions."""

    _DEICTIC_TOKENS = _DEICTIC_TOKENS
    _COMMAND_PATTERNS = (
        (re.compile(r"^(?:open|launch)\s+(.+)$", re.IGNORECASE), "open_app"),
        (re.compile(r"^(?:switch to|go to)\s+(.+)$", re.IGNORECASE), "switch_app"),
        (re.compile(r"^(?:press|hit)\s+(escape|tab|enter|back)$", re.IGNORECASE), "press_key"),
        (re.compile(r"^scroll\s+(up|down)(?:\s+(faster|slower))?$", re.IGNORECASE), "scroll"),
        (re.compile(r"^(?:type|dictate)\s+(.+)$", re.IGNORECASE), "type_text"),
        (re.compile(r"^translate\s+(.+?)\s+to\s+([a-zA-Z]+)$", re.IGNORECASE), "translate_text"),
        (
            re.compile(
                r"^(?:click|right(?: |-)?click|secondary click|open|focus|move(?: cursor)?|show)\s+"
                r"(?:this|that|these|those|here|there|it|them)$",
                re.IGNORECASE,
            ),
            "deictic",
        ),
    )

    _CONFIRM_TOKENS = {"yes", "confirm", "do it", "go ahead", "okay", "ok"}
    _CANCEL_TOKENS = {"no", "cancel", "stop", "never mind", "abort", "nope"}

    def interpret(self, utterance: str) -> VoiceDirective:
        text = repair_command_utterance(utterance)
        if not text:
            return VoiceDirective(
                kind=VoiceDirectiveType.CLARIFY,
                utterance=text,
                message="I did not hear a command.",
                source="regex",
            )
        review_token = match_review_token(text, pending_confirmation=True)
        if review_token is not None:
            return review_token

        for pattern, command in self._COMMAND_PATTERNS:
            match = pattern.match(text)
            if not match:
                continue
            if command == "deictic":
                return VoiceDirective(
                    kind=VoiceDirectiveType.CLARIFY,
                    utterance=text,
                    message="That command needs gaze grounding before I can act on it.",
                    source="regex",
                )
            return self._proposal_from_match(command, text, match.groups())

        return VoiceDirective(
            kind=VoiceDirectiveType.REJECT,
            utterance=text,
            message="I could not map that request to the current bounded voice command set.",
            source="regex",
        )

    def _proposal_from_match(self, command: str, utterance: str, groups: tuple[str | None, ...]) -> VoiceDirective:
        if command == "open_app":
            app_name = groups[0].strip()
            if self._looks_deictic_target(app_name):
                return VoiceDirective(
                    kind=VoiceDirectiveType.CLARIFY,
                    utterance=utterance,
                    message="That command needs gaze grounding before I can act on it.",
                    source="regex",
                )
            return self._proposal(
                utterance=utterance,
                intent_name="open_app",
                action=ActionName.OPEN_APP,
                arguments={"app_name": app_name},
                risk=RiskLevel.L1,
                rationale=f'Voice command requested opening the app "{app_name}".',
            )
        if command == "switch_app":
            app_name = groups[0].strip()
            if self._looks_deictic_target(app_name):
                return VoiceDirective(
                    kind=VoiceDirectiveType.CLARIFY,
                    utterance=utterance,
                    message="That command needs gaze grounding before I can act on it.",
                    source="regex",
                )
            return self._proposal(
                utterance=utterance,
                intent_name="switch_app",
                action=ActionName.SWITCH_APP,
                arguments={"app_name": app_name},
                risk=RiskLevel.L1,
                rationale=f'Voice command requested switching to the app "{app_name}".',
            )
        if command == "press_key":
            key = groups[0].strip().lower()
            return self._proposal(
                utterance=utterance,
                intent_name="press_key",
                action=ActionName.PRESS_KEY,
                arguments={"key": key},
                risk=RiskLevel.L1,
                rationale=f'Voice command requested pressing the key "{key}".',
            )
        if command == "scroll":
            direction = groups[0].strip().lower()
            speed = groups[1].strip().lower() if len(groups) > 1 and groups[1] else None
            arguments = {"direction": direction}
            if speed:
                arguments["speed"] = speed
            return self._proposal(
                utterance=utterance,
                intent_name="scroll",
                action=ActionName.SCROLL,
                arguments=arguments,
                risk=RiskLevel.L1,
                rationale=f'Voice command requested scrolling {direction}.',
            )
        if command == "type_text":
            text = groups[0].strip()
            return self._proposal(
                utterance=utterance,
                intent_name="type_text",
                action=ActionName.TYPE_TEXT,
                arguments={"text": text},
                risk=RiskLevel.L2,
                rationale="Voice command requested typing text into the active field.",
                requires_confirmation=True,
            )
        if command == "translate_text":
            source_text = groups[0].strip()
            target_language = groups[1].strip()
            return self._proposal(
                utterance=utterance,
                intent_name="translate_text",
                action=ActionName.TRANSLATE_TEXT,
                arguments={"text": source_text, "target_language": target_language},
                risk=RiskLevel.L2,
                rationale=f'Voice command requested translating text into "{target_language}".',
                requires_confirmation=True,
            )
        raise ValueError(f"Unsupported command mapping: {command}")

    def _proposal(
        self,
        *,
        utterance: str,
        intent_name: str,
        action: ActionName,
        arguments: dict[str, object],
        risk: RiskLevel,
        rationale: str,
        requires_confirmation: bool | None = None,
    ) -> VoiceDirective:
        confidence = 0.95 if risk in {RiskLevel.L0, RiskLevel.L1} else 0.9
        proposal = ActionProposal(
            action=action,
            arguments=arguments,
            confidence=confidence,
            risk=risk,
            requires_confirmation=requires_confirmation if requires_confirmation is not None else risk in {RiskLevel.L2, RiskLevel.L3},
            rationale=rationale,
        )
        intent = NormalizedIntent(
            intent=intent_name,
            modifiers={},
            confidence=confidence,
            needs_clarification=False,
            utterance=utterance,
        )
        return VoiceDirective(
            kind=VoiceDirectiveType.PROPOSAL,
            utterance=utterance,
            intent=intent,
            proposal=proposal,
            source="regex",
        )

    @classmethod
    def _looks_deictic_target(cls, value: str) -> bool:
        lowered = normalize_utterance(value).lower()
        if not lowered:
            return False
        if lowered in cls._DEICTIC_TOKENS:
            return True
        first_token = lowered.split(" ", 1)[0]
        return first_token in cls._DEICTIC_TOKENS
