"""Shared multimodal interaction state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from interaction.contracts import GazeObservation, GroundedTarget

from .clock import Clock, monotonic_ms


@dataclass(frozen=True)
class GazeContextEntry:
    observed_at_ms: int
    observation: GazeObservation
    target: GroundedTarget


@dataclass(frozen=True)
class TranscriptEntry:
    observed_at_ms: int
    utterance: str
    confidence: float | None = None


class SharedInteractionState:
    """Keep a short history of gaze and voice context for fusion."""

    def __init__(
        self,
        *,
        gaze_context_window_ms: int = 1500,
        max_history: int = 12,
        clock: Clock = monotonic_ms,
    ) -> None:
        self.gaze_context_window_ms = gaze_context_window_ms
        self._clock = clock
        self._simulation_offset_ms = 0
        self._last_gaze_at_ms: int | None = None
        self.gaze_revision = 0
        self.gaze_history: deque[GazeContextEntry] = deque(maxlen=max_history)
        self.transcript_history: deque[TranscriptEntry] = deque(maxlen=max_history)

    @property
    def now_ms(self) -> int:
        return self._clock() + self._simulation_offset_ms

    def advance(self, delta_ms: int) -> int:
        """Advance scripted time explicitly; live input uses actual timestamps."""
        self._simulation_offset_ms += max(0, delta_ms)
        return self.now_ms

    def record_gaze(
        self,
        observation: GazeObservation,
        target: GroundedTarget | None,
        *,
        delta_ms: int | None = None,
        observed_at_ms: int | None = None,
    ) -> None:
        if delta_ms is not None:
            self.advance(delta_ms)
        now = self.now_ms
        observed_at = now if observed_at_ms is None else observed_at_ms
        # Never turn delayed input into a fresh sample, or let an out-of-order
        # reading revive a target that was subsequently lost.
        if observed_at > now or (self._last_gaze_at_ms is not None and observed_at < self._last_gaze_at_ms):
            return
        self._last_gaze_at_ms = observed_at
        if target is None:
            self.clear_gaze(observed_at_ms=observed_at)
            return
        previous = self.gaze_history[-1].target.target_id if self.gaze_history else None
        if previous != target.target_id:
            self.gaze_revision += 1
        self.gaze_history.append(
            GazeContextEntry(
                observed_at_ms=observed_at,
                observation=observation,
                target=target,
            )
        )

    def clear_gaze(self, *, observed_at_ms: int | None = None) -> None:
        self.gaze_history.clear()
        self.gaze_revision += 1
        self._last_gaze_at_ms = self.now_ms if observed_at_ms is None else observed_at_ms

    def record_transcript(self, utterance: str, *, confidence: float | None = None, delta_ms: int = 0) -> None:
        self.advance(delta_ms)
        self.transcript_history.append(
            TranscriptEntry(
                observed_at_ms=self.now_ms,
                utterance=utterance,
                confidence=confidence,
            )
        )

    def latest_target(self) -> GroundedTarget | None:
        entry = self._latest_entry()
        return entry.target if entry is not None else None

    def latest_target_age_ms(self) -> int | None:
        entry = self.gaze_history[-1] if self.gaze_history else None
        if entry is None:
            return None
        return self.now_ms - entry.observed_at_ms

    def latest_observation(self) -> GazeObservation | None:
        entry = self._latest_entry()
        return entry.observation if entry is not None else None

    def latest_gaze_confidence(self) -> float | None:
        observation = self.latest_observation()
        return observation.confidence if observation is not None else None

    def candidate_targets(self, *, limit: int = 3) -> list[GroundedTarget]:
        if limit <= 0:
            return []
        candidates: list[GroundedTarget] = []
        seen_ids: set[str] = set()
        now = self.now_ms
        for entry in reversed(self.gaze_history):
            if not 0 <= now - entry.observed_at_ms <= self.gaze_context_window_ms:
                continue
            if entry.target.target_id in seen_ids:
                continue
            seen_ids.add(entry.target.target_id)
            candidates.append(entry.target)
            if len(candidates) >= limit:
                break
        return candidates

    def latest_target_is_fresh(self) -> bool:
        age = self.latest_target_age_ms()
        return age is not None and 0 <= age <= self.gaze_context_window_ms

    def stale_reason(self) -> str:
        candidates = self.candidate_targets(limit=2)
        if not candidates:
            return "I do not have a recent grounded gaze target for that command."
        labels = ", ".join(target.label for target in candidates)
        return f'My recent gaze context is stale or ambiguous. Recent targets: {labels}.'

    def _latest_entry(self) -> GazeContextEntry | None:
        if not self.gaze_history:
            return None
        entry = self.gaze_history[-1]
        return entry if 0 <= self.now_ms - entry.observed_at_ms <= self.gaze_context_window_ms else None
