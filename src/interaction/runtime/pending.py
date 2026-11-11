"""Context bound to a pending action until deliberate confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from interaction.contracts import ActionName, BrokerDecision, EnvironmentSnapshot, GazeObservation


_POINTER_ACTIONS = {
    ActionName.FOCUS_TARGET,
    ActionName.CLICK_TARGET,
    ActionName.DOUBLE_CLICK_TARGET,
    ActionName.RIGHT_CLICK_TARGET,
    ActionName.DRAG_TARGET,
}


@dataclass(frozen=True)
class PendingAction:
    decision: BrokerDecision
    environment: EnvironmentSnapshot
    created_at_ms: int
    gaze_revision: int | None = None

    @property
    def requires_gaze(self) -> bool:
        return self.decision.proposal is not None and self.decision.proposal.action in _POINTER_ACTIONS

    def invalid_reason(
        self,
        *,
        now_ms: int,
        timeout_ms: int,
        environment: EnvironmentSnapshot,
        current_target_id: str | None = None,
        observation: GazeObservation | None = None,
        gaze_revision: int | None = None,
        max_gaze_distance: float = 0.08,
    ) -> str | None:
        age = now_ms - self.created_at_ms
        if age < 0 or age >= timeout_ms:
            return "The pending action expired. Please repeat the command."
        if self.environment != environment:
            return "The application or window changed. Please repeat the command in the intended location."
        if not self.requires_gaze:
            return None
        proposal = self.decision.proposal
        assert proposal is not None
        intended_target = proposal.arguments.get("target_ref")
        if (
            not isinstance(intended_target, str)
            or not intended_target
            or current_target_id != intended_target
            or observation is None
            or self.gaze_revision != gaze_revision
        ):
            return "The gaze target changed or tracking was lost. Please look at the target and repeat the command."
        point = proposal.arguments.get("normalized_point")
        if (
            not isinstance(point, dict)
            or not isinstance(point.get("x"), (int, float))
            or not isinstance(point.get("y"), (int, float))
            or not isfinite(point["x"])
            or not isfinite(point["y"])
            or not 0 <= point["x"] <= 1
            or not 0 <= point["y"] <= 1
            or observation.x_norm is None
            or observation.y_norm is None
        ):
            return "The pending pointer action has no current calibrated gaze point. Please repeat the command."
        distance = hypot(observation.x_norm - point["x"], observation.y_norm - point["y"])
        if distance > max_gaze_distance:
            return "Your gaze moved away from the pending point. Please repeat the command at the intended location."
        return None
