"""Feedback events for multimodal fusion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from interaction.contracts import BrokerDecision, ExecutionResult, GroundedTarget


class FusionLoopPhase(StrEnum):
    TRACKING = "tracking"
    INTERPRETING = "interpreting"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    RECOVERING = "recovering"
    IDLE = "idle"


@dataclass(frozen=True)
class FusionFeedbackEvent:
    phase: FusionLoopPhase
    message: str
    transcript: str | None = None
    target: GroundedTarget | None = None
    decision: BrokerDecision | None = None
    result: ExecutionResult | None = None
