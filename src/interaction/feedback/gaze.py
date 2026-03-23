"""Feedback events for gaze tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from interaction.contracts import ActionProposal, ExecutionResult, GazeObservation, GroundedTarget


class GazeLoopPhase(StrEnum):
    CALIBRATING = "calibrating"
    TRACKING = "tracking"
    TRIGGERED = "triggered"
    RECOVERING = "recovering"
    IDLE = "idle"


@dataclass(frozen=True)
class GazeFeedbackEvent:
    phase: GazeLoopPhase
    message: str
    observation: GazeObservation | None = None
    target: GroundedTarget | None = None
    proposal: ActionProposal | None = None
    result: ExecutionResult | None = None
