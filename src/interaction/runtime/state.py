"""Shared multimodal interaction state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from interaction.contracts import GazeObservation, GroundedTarget


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

    def __init__(self, *, gaze_context_window_ms: int = 1500, max_history: int = 12) -> None:
        self.gaze_context_window_ms = gaze_context_window_ms
        self.now_ms = 0
        self.gaze_history: deque[GazeContextEntry] = deque(maxlen=max_history)
        self.transcript_history: deque[TranscriptEntry] = deque(maxlen=max_history)

    def advance(self, delta_ms: int) -> int:
        self.now_ms += max(0, delta_ms)
        return self.now_ms

    def record_gaze(self, observation: GazeObservation, target: GroundedTarget | None, *, delta_ms: int | None = None) -> None:
        if delta_ms is None:
            delta_ms = observation.fixation_ms or 0
        self.advance(delta_ms)
        if target is None:
            return
        self.gaze_history.append(
            GazeContextEntry(
                observed_at_ms=self.now_ms,
                observation=observation,
                target=target,
            )
        )

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
        entry = self._latest_entry()
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
        candidates: list[GroundedTarget] = []
        seen_ids: set[str] = set()
        for entry in reversed(self.gaze_history):
            if self.now_ms - entry.observed_at_ms > self.gaze_context_window_ms:
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
        return age is not None and age <= self.gaze_context_window_ms

    def stale_reason(self) -> str:
        candidates = self.candidate_targets(limit=2)
        if not candidates:
            return "I do not have a recent grounded gaze target for that command."
        labels = ", ".join(target.label for target in candidates)
        return f'My recent gaze context is stale or ambiguous. Recent targets: {labels}.'

    def _latest_entry(self) -> GazeContextEntry | None:
        for entry in reversed(self.gaze_history):
            if self.now_ms - entry.observed_at_ms <= self.gaze_context_window_ms:
                return entry
        return None
