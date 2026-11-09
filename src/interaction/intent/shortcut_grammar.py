"""Small deterministic fast-path grammar for low-latency shortcut commands."""

from __future__ import annotations

from dataclasses import dataclass
import re

from interaction.contracts import ActionName, ActionProposal, NavigationCommand, NavigationScope, RiskLevel, SessionMode
from .voice import repair_command_utterance


@dataclass(frozen=True)
class FastPathMatch:
    intent_name: str
    proposal: ActionProposal


_SAFE_KEYS = {"escape", "tab", "enter", "back"}
_SCROLL_DIRECTIONS = {"up", "down"}
_DEICTIC_TOKENS = {"this", "that", "these", "those", "here", "there", "it", "them"}
_STRUCTURED_COMMANDS = {
    "next": NavigationCommand.NEXT,
    "back": NavigationCommand.BACK,
    "left": NavigationCommand.LEFT,
    "right": NavigationCommand.RIGHT,
    "up": NavigationCommand.UP,
    "down": NavigationCommand.DOWN,
    "new": NavigationCommand.NEW,
    "close": NavigationCommand.CLOSE,
    "reopen": NavigationCommand.REOPEN,
}
_STRUCTURED_SCOPES = {
    "app": NavigationScope.APP,
    "window": NavigationScope.WINDOW,
    "tab": NavigationScope.TAB,
    "pane": NavigationScope.PANE,
    "space": NavigationScope.SPACE,
    "display": NavigationScope.DISPLAY,
}

_COUNT_PATTERN = re.compile(r"^(?P<scope>app|window|tab|pane|space|display)\s+(?P<index>[1-9]\d*)$", re.IGNORECASE)
_SCOPED_COMMAND_PATTERN = re.compile(
    r"^(?P<scope>app|window|tab|pane|space|display)\s+"
    r"(?P<command>next|back|left|right|up|down|new|close|reopen)$",
    re.IGNORECASE,
)
_FOLLOW_UP_COMMAND_PATTERN = re.compile(r"^(?P<command>next|back|left|right|up|down|new|close|reopen)$", re.IGNORECASE)
_FOLLOW_UP_INDEX_PATTERN = re.compile(r"^(?P<index>[1-9]\d*)$", re.IGNORECASE)


def match_fast_path(
    utterance: str,
    *,
    mode: SessionMode = SessionMode.COMMAND,
    active_scope: NavigationScope | None = None,
) -> FastPathMatch | None:
    text = repair_command_utterance(utterance)
    if not text or mode == SessionMode.DICTATION:
        return None

    lowered = text.lower()
    if lowered.startswith("open ") and len(text) > 5:
        app_name = text[5:].strip()
        if _looks_deictic_target(app_name):
            return None
        return _proposal(
            intent_name="open_app",
            action=ActionName.OPEN_APP,
            arguments={"app_name": app_name},
            risk=RiskLevel.L1,
            rationale=f'Fast-path shortcut requested opening "{app_name}".',
        )
    switch_prefix = None
    if lowered.startswith("switch to ") and len(text) > 10:
        switch_prefix = "switch to "
    elif lowered.startswith("switch ") and len(text) > 7:
        switch_prefix = "switch "
    if switch_prefix is not None:
        app_name = text[len(switch_prefix) :].strip()
        if _looks_deictic_target(app_name):
            return None
        return _proposal(
            intent_name="switch_app",
            action=ActionName.SWITCH_APP,
            arguments={"app_name": app_name},
            risk=RiskLevel.L1,
            rationale=f'Fast-path shortcut requested switching to "{app_name}".',
        )
    if lowered.startswith("press "):
        key = lowered[6:].strip()
        if key in _SAFE_KEYS:
            return _proposal(
                intent_name="press_key",
                action=ActionName.PRESS_KEY,
                arguments={"key": key},
                risk=RiskLevel.L1,
                rationale=f'Fast-path shortcut requested pressing "{key}".',
            )
    if lowered.startswith("scroll "):
        direction = lowered[7:].strip()
        if direction in _SCROLL_DIRECTIONS:
            return _proposal(
                intent_name="scroll",
                action=ActionName.SCROLL,
                arguments={"direction": direction},
                risk=RiskLevel.L1,
                rationale=f"Fast-path shortcut requested scrolling {direction}.",
            )
    if lowered.startswith("type ") and len(text) > 5:
        typed_text = text[5:].strip()
        return _proposal(
            intent_name="type_text",
            action=ActionName.TYPE_TEXT,
            arguments={"text": typed_text},
            risk=RiskLevel.L2,
            rationale="Fast-path shortcut requested typing text into the active field.",
            requires_confirmation=True,
        )

    structured_match = _match_structured_navigation(text, active_scope=active_scope)
    if structured_match is not None:
        return structured_match
    return None


def _match_structured_navigation(
    utterance: str,
    *,
    active_scope: NavigationScope | None,
) -> FastPathMatch | None:
    count_match = _COUNT_PATTERN.match(utterance)
    if count_match is not None:
        scope = _STRUCTURED_SCOPES[count_match.group("scope").lower()]
        index = int(count_match.group("index"))
        return _structured_navigation(scope=scope, command=NavigationCommand.INDEX, index=index)

    command_match = _SCOPED_COMMAND_PATTERN.match(utterance)
    if command_match is not None:
        scope = _STRUCTURED_SCOPES[command_match.group("scope").lower()]
        command = _STRUCTURED_COMMANDS[command_match.group("command").lower()]
        return _structured_navigation(scope=scope, command=command)

    if active_scope is None:
        return None

    follow_up_match = _FOLLOW_UP_COMMAND_PATTERN.match(utterance)
    if follow_up_match is not None:
        command = _STRUCTURED_COMMANDS[follow_up_match.group("command").lower()]
        return _structured_navigation(scope=active_scope, command=command)

    follow_up_index = _FOLLOW_UP_INDEX_PATTERN.match(utterance)
    if follow_up_index is not None:
        return _structured_navigation(
            scope=active_scope,
            command=NavigationCommand.INDEX,
            index=int(follow_up_index.group("index")),
        )
    return None


def _structured_navigation(
    *,
    scope: NavigationScope,
    command: NavigationCommand,
    index: int | None = None,
) -> FastPathMatch:
    arguments: dict[str, object] = {
        "scope": scope.value,
        "command": command.value,
    }
    if index is not None:
        arguments["index"] = index
    label = f"{scope.value} {index}" if command == NavigationCommand.INDEX and index is not None else f"{scope.value} {command.value}"
    return _proposal(
        intent_name="navigate_structure",
        action=ActionName.NAVIGATE_STRUCTURE,
        arguments=arguments,
        risk=RiskLevel.L1,
        rationale=f'Fast-path shortcut requested structured navigation "{label}".',
    )


def _looks_deictic_target(value: str) -> bool:
    lowered = " ".join(value.strip().split()).lower()
    if not lowered:
        return False
    if lowered in _DEICTIC_TOKENS:
        return True
    return lowered.split(" ", 1)[0] in _DEICTIC_TOKENS


def _proposal(
    *,
    intent_name: str,
    action: ActionName,
    arguments: dict[str, object],
    risk: RiskLevel,
    rationale: str,
    requires_confirmation: bool | None = None,
) -> FastPathMatch:
    confidence = 0.98 if risk in {RiskLevel.L0, RiskLevel.L1} else 0.94
    proposal = ActionProposal(
        action=action,
        arguments=arguments,
        confidence=confidence,
        risk=risk,
        requires_confirmation=requires_confirmation if requires_confirmation is not None else risk in {RiskLevel.L2, RiskLevel.L3},
        rationale=rationale,
    )
    return FastPathMatch(intent_name=intent_name, proposal=proposal)
