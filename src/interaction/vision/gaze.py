"""Gaze smoothing, target inference, and dwell-trigger logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from interaction.contracts import ActionName, ActionProposal, BoundingBox, GazeObservation, GroundedTarget, RiskLevel

from .calibration import NormalizedPoint


@dataclass(frozen=True)
class GazeSample:
    point: NormalizedPoint
    confidence: float = 1.0
    delta_ms: int = 100


@dataclass(frozen=True)
class NormalizedScreenTarget:
    target_id: str
    label: str
    role: str
    x: float
    y: float
    width: float
    height: float

    def contains(self, point: NormalizedPoint) -> bool:
        epsilon = 1e-6
        return (
            self.x - epsilon <= point.x <= self.x + self.width + epsilon
            and self.y - epsilon <= point.y <= self.y + self.height + epsilon
        )

    def center(self) -> NormalizedPoint:
        return NormalizedPoint(self.x + self.width / 2.0, self.y + self.height / 2.0)

    def to_grounded_target(self, confidence: float) -> GroundedTarget:
        return GroundedTarget(
            target_id=self.target_id,
            label=self.label,
            role=self.role,
            confidence=max(0.0, min(1.0, confidence)),
            screen_region=_region_label(self.center()),
            bounds=BoundingBox(x=self.x, y=self.y, width=self.width, height=self.height),
        )


class GazeSmoother:
    """Simple moving-average smoother for normalized gaze points."""

    def __init__(self, window_size: int = 4) -> None:
        self.window: deque[NormalizedPoint] = deque(maxlen=window_size)

    def reset(self) -> None:
        self.window.clear()

    def smooth(self, sample: GazeSample) -> GazeObservation:
        self.window.append(sample.point)
        x = sum(point.x for point in self.window) / len(self.window)
        y = sum(point.y for point in self.window) / len(self.window)
        return GazeObservation(
            confidence=sample.confidence,
            x_norm=x,
            y_norm=y,
            fixation_ms=sample.delta_ms,
        )


class GazeTargetInferencer:
    """Infer the most likely large target under the current gaze point."""

    def __init__(self, *, min_width: float = 0.08, min_height: float = 0.05, min_confidence: float = 0.6) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.min_confidence = min_confidence

    def infer(self, observation: GazeObservation, targets: list[NormalizedScreenTarget]) -> GroundedTarget | None:
        if not _valid_observation(observation, min_confidence=self.min_confidence):
            return None
        point = NormalizedPoint(observation.x_norm, observation.y_norm)
        candidates = [
            target
            for target in targets
            if target.width >= self.min_width and target.height >= self.min_height and target.contains(point)
        ]
        if not candidates:
            return None

        def score(target: NormalizedScreenTarget) -> float:
            center = target.center()
            distance = math.dist((point.x, point.y), (center.x, center.y))
            area_bonus = target.width * target.height
            return area_bonus - distance

        best = max(candidates, key=score)
        return best.to_grounded_target(confidence=observation.confidence)


class DwellTrigger:
    """Trigger a conservative highlight action after stable dwell on the same target."""

    def __init__(
        self,
        *,
        dwell_ms: int = 700,
        min_confidence: float = 0.65,
        action: ActionName = ActionName.HIGHLIGHT_TARGET,
        cooldown_ms: int = 900,
        rearm_ms: int = 200,
        max_radius: float = 0.05,
    ) -> None:
        self.dwell_ms = dwell_ms
        self.min_confidence = min_confidence
        self.action = action
        self.cooldown_ms = max(0, cooldown_ms)
        self.rearm_ms = max(0, rearm_ms)
        if not math.isfinite(max_radius) or max_radius <= 0:
            raise ValueError("Dwell radius must be positive and finite.")
        self.max_radius = max_radius
        self.current_target_id: str | None = None
        self.accumulated_ms = 0
        self.triggered_target_id: str | None = None
        self.cooldown_remaining_ms = 0
        self.lookaway_ms = 0
        self.anchor_point: NormalizedPoint | None = None

    def start_cooldown(self) -> None:
        self.reset()
        self.cooldown_remaining_ms = self.cooldown_ms

    def reset(self, *, elapsed_ms: int = 0, preserve_trigger: bool = True) -> None:
        """Clear interrupted dwell; dropped tracking never rearms a fired target."""
        self._clear_partial()
        self.lookaway_ms = 0
        self.cooldown_remaining_ms = max(0, self.cooldown_remaining_ms - max(0, elapsed_ms))
        if not preserve_trigger:
            self.triggered_target_id = None
            self.cooldown_remaining_ms = 0

    def _clear_partial(self) -> None:
        self.current_target_id = None
        self.accumulated_ms = 0
        self.anchor_point = None

    def update(
        self, observation: GazeObservation, target: GroundedTarget | None, *, elapsed_ms: int | None = None,
    ) -> ActionProposal | None:
        delta_ms = max(0, observation.fixation_ms or 0)
        elapsed_ms = delta_ms if elapsed_ms is None else max(0, elapsed_ms)
        if not _valid_observation(observation, min_confidence=self.min_confidence) or (
            target is not None and (not math.isfinite(target.confidence) or target.confidence < self.min_confidence)
        ):
            self.reset(elapsed_ms=elapsed_ms)
            return None
        self.cooldown_remaining_ms = max(0, self.cooldown_remaining_ms - elapsed_ms)

        # A valid point away from the fired target is deliberate evidence.
        # Missing/weak readings use reset(), so interrupted look-away cannot
        # accumulate across an interval in which the eyes were not tracked.
        if self.triggered_target_id is not None:
            if target is None or target.target_id != self.triggered_target_id:
                self.lookaway_ms += delta_ms
                if self.lookaway_ms >= self.rearm_ms:
                    self.triggered_target_id = None
                    self.lookaway_ms = 0
                    self._clear_partial()
                    # The look-away interval establishes deliberate rearming;
                    # it is not also credited toward a new activation dwell.
                    return None
            else:
                self.lookaway_ms = 0

        if target is None or self.cooldown_remaining_ms > 0 or target.target_id == self.triggered_target_id:
            self._clear_partial()
            return None

        point = NormalizedPoint(observation.x_norm, observation.y_norm)

        if target.target_id == self.current_target_id:
            if self.anchor_point is not None and math.dist((point.x, point.y), (self.anchor_point.x, self.anchor_point.y)) > self.max_radius:
                self.accumulated_ms = 0
                self.anchor_point = point
            else:
                self.accumulated_ms += delta_ms
        else:
            self.current_target_id = target.target_id
            self.accumulated_ms = delta_ms
            self.anchor_point = point

        if self.accumulated_ms >= self.dwell_ms and self.triggered_target_id is None:
            self.triggered_target_id = target.target_id
            if self.action != ActionName.DRAG_TARGET:
                self.start_cooldown()
            return self._build_proposal(observation, target)
        return None

    def _build_proposal(self, observation: GazeObservation, target: GroundedTarget) -> ActionProposal:
        arguments: dict[str, object] = {
            "target_ref": target.target_id,
            "target_label": target.label,
        }
        if target.bounds is not None:
            arguments["target_bounds"] = target.bounds.model_dump(mode="json")
        if observation.x_norm is not None and observation.y_norm is not None:
            arguments["normalized_point"] = {
                "x": observation.x_norm,
                "y": observation.y_norm,
            }

        rationale = "Stable dwell on a large target triggered a conservative highlight action."
        risk = RiskLevel.L0
        requires_confirmation = False

        if self.action == ActionName.FOCUS_TARGET:
            rationale = "Stable dwell on a large target triggered a gaze-only pointer move action."
            risk = RiskLevel.L1
        elif self.action == ActionName.CLICK_TARGET:
            rationale = "Stable dwell on a large target triggered a gaze-only click action."
            risk = RiskLevel.L2
            requires_confirmation = True
        elif self.action == ActionName.RIGHT_CLICK_TARGET:
            rationale = "Stable dwell on a large target triggered a gaze-only right-click action."
            risk = RiskLevel.L2
            requires_confirmation = True
        elif self.action == ActionName.DOUBLE_CLICK_TARGET:
            rationale = "Stable dwell on a large target triggered a gaze-only double-click action."
            risk = RiskLevel.L2
            requires_confirmation = True
        elif self.action == ActionName.DRAG_TARGET:
            rationale = "Stable dwell on a large target triggered a gaze-only drag stage."
            risk = RiskLevel.L2
            requires_confirmation = True

        return ActionProposal(
            action=self.action,
            arguments=arguments,
            confidence=target.confidence,
            risk=risk,
            requires_confirmation=requires_confirmation,
            rationale=rationale,
        )


def _valid_observation(observation: GazeObservation, *, min_confidence: float) -> bool:
    values = (observation.x_norm, observation.y_norm, observation.confidence)
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values):
        return False
    return (
        0 <= observation.x_norm <= 1
        and 0 <= observation.y_norm <= 1
        and min_confidence <= observation.confidence <= 1
    )


def _region_label(point: NormalizedPoint) -> str:
    horizontal = "left" if point.x < 0.33 else "right" if point.x > 0.66 else "center"
    vertical = "upper" if point.y < 0.33 else "lower" if point.y > 0.66 else "middle"
    return f"{vertical}_{horizontal}"
