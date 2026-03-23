"""Push-to-talk turn state management."""

from __future__ import annotations

from enum import StrEnum


class TurnPhase(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    COMPLETE = "complete"


class PushToTalkTurnManager:
    """Track the lifecycle of a single push-to-talk turn."""

    def __init__(self) -> None:
        self.phase = TurnPhase.IDLE

    def begin(self) -> TurnPhase:
        self.phase = TurnPhase.LISTENING
        return self.phase

    def end(self) -> TurnPhase:
        self.phase = TurnPhase.COMPLETE
        return self.phase

    def reset(self) -> TurnPhase:
        self.phase = TurnPhase.IDLE
        return self.phase
