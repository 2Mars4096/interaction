"""Voice-loop feedback events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from interaction.contracts import BrokerDecision, ExecutionResult


class VoiceLoopPhase(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    INTERPRETING = "interpreting"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class VoiceFeedbackEvent:
    phase: VoiceLoopPhase
    transcript: str | None = None
    message: str | None = None
    decision: BrokerDecision | None = None
    result: ExecutionResult | None = None
