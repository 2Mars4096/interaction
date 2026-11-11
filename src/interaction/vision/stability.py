"""Stable target evidence from consecutive, unsmoothed gaze measurements."""

from __future__ import annotations

import math

from interaction.contracts import GazeObservation, GroundedTarget


class StableGazeLock:
    """Require sample count, elapsed time, and bounded dispersion together.

    Callers must pass calibrated coordinates before smoothing. An invalid sample
    ends the current sequence immediately. A changed target, displaced point, or
    long sampling gap starts a new sequence, whose revision cannot revive an old
    pending action even if the user subsequently looks back at the same target.
    """

    def __init__(
        self,
        *,
        min_samples: int = 3,
        min_stable_ms: float = 120,
        max_gap_ms: float = 250,
        max_radius: float = 0.05,
        min_confidence: float = 0.6,
    ) -> None:
        if type(min_samples) is not int or min_samples < 2:
            raise ValueError("A stable gaze lock requires at least two samples.")
        if not _finite_number(min_stable_ms) or min_stable_ms < 0:
            raise ValueError("Stable gaze duration must be finite and non-negative.")
        if not _finite_number(max_gap_ms) or max_gap_ms <= 0:
            raise ValueError("The maximum gaze sampling gap must be positive and finite.")
        if not _finite_number(max_radius) or max_radius < 0:
            raise ValueError("The stable gaze radius must be finite and non-negative.")
        if not _finite_number(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("The minimum gaze confidence must be between zero and one.")
        self.min_samples = min_samples
        self.min_stable_ms = min_stable_ms
        self.max_gap_ms = max_gap_ms
        self.max_radius = max_radius
        self.min_confidence = min_confidence
        self._revision = 0
        self._sample_count = 0
        self._target_key: tuple | None = None
        self._anchor: tuple[float, float] | None = None
        self._started_at_ms: float | None = None
        self._last_at_ms: float | None = None

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def duration_ms(self) -> int:
        if self._started_at_ms is None or self._last_at_ms is None:
            return 0
        return int(self._last_at_ms - self._started_at_ms)

    def reset(self) -> None:
        """Invalidate accumulated evidence; repeated empty resets are harmless."""
        if self._sample_count:
            self._revision += 1
        self._sample_count = 0
        self._target_key = None
        self._anchor = None
        self._started_at_ms = None
        self._last_at_ms = None

    def update(
        self,
        observation: GazeObservation | None,
        target: GroundedTarget | None,
        *,
        observed_at_ms: float | None,
    ) -> bool:
        if (
            observation is None
            or target is None
            or not _finite_number(observed_at_ms)
            or observed_at_ms < 0
            or not _finite_number(observation.confidence)
            or not self.min_confidence <= observation.confidence <= 1
            or not _finite_number(target.confidence)
            or not self.min_confidence <= target.confidence <= 1
            or not _normalized_coordinate(observation.x_norm)
            or not _normalized_coordinate(observation.y_norm)
            or not target.target_id
            or (observation.target_id is not None and observation.target_id != target.target_id)
        ):
            self.reset()
            return False
        if self._last_at_ms is not None and observed_at_ms < self._last_at_ms:
            self.reset()
            return False

        bounds = target.bounds
        key = (
            target.target_id,
            (bounds.x, bounds.y, bounds.width, bounds.height) if bounds is not None else None,
        )
        point = (observation.x_norm, observation.y_norm)
        if self._sample_count and (
            key != self._target_key
            or observed_at_ms - self._last_at_ms > self.max_gap_ms
            or math.dist(point, self._anchor) > self.max_radius
        ):
            self.reset()
        if not self._sample_count:
            self._target_key = key
            self._anchor = point
            self._started_at_ms = observed_at_ms
        self._sample_count += 1
        self._last_at_ms = observed_at_ms
        return self._sample_count >= self.min_samples and self.duration_ms >= self.min_stable_ms


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _normalized_coordinate(value: object) -> bool:
    return _finite_number(value) and 0 <= value <= 1
