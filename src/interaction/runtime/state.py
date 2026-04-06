"""Shared multimodal interaction state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from interaction.contracts import EnvironmentSnapshot, GazeObservation, GroundedTarget, NavigationScope


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


@dataclass(frozen=True)
class CandidateTargetEntry:
    observed_at_ms: int
    targets: tuple[GroundedTarget, ...]


@dataclass(frozen=True)
class CommandEntry:
    observed_at_ms: int
    key: str


@dataclass(frozen=True)
class ActiveScopeEntry:
    observed_at_ms: int
    scope: NavigationScope


class SharedInteractionState:
    """Keep a short history of gaze, transcript, scope, and dedupe context."""

    def __init__(
        self,
        *,
        gaze_context_window_ms: int = 1500,
        scope_context_window_ms: int = 6000,
        dedupe_window_ms: int = 1200,
        max_history: int = 12,
    ) -> None:
        self.gaze_context_window_ms = gaze_context_window_ms
        self.scope_context_window_ms = scope_context_window_ms
        self.dedupe_window_ms = dedupe_window_ms
        self.now_ms = 0
        self.gaze_history: deque[GazeContextEntry] = deque(maxlen=max_history)
        self.explicit_candidate_history: deque[CandidateTargetEntry] = deque(maxlen=max_history)
        self.transcript_history: deque[TranscriptEntry] = deque(maxlen=max_history)
        self.command_history: deque[CommandEntry] = deque(maxlen=max_history)
        self.latest_environment: EnvironmentSnapshot | None = None
        self.active_scope_entry: ActiveScopeEntry | None = None

    def advance(self, delta_ms: int) -> int:
        self.now_ms += max(0, delta_ms)
        return self.now_ms

    def record_gaze(
        self,
        observation: GazeObservation,
        target: GroundedTarget | None,
        *,
        delta_ms: int | None = None,
        candidate_targets: list[GroundedTarget] | None = None,
    ) -> None:
        if delta_ms is None:
            delta_ms = observation.fixation_ms or 0
        self.advance(delta_ms)
        if target is None:
            if candidate_targets:
                self.explicit_candidate_history.append(
                    CandidateTargetEntry(
                        observed_at_ms=self.now_ms,
                        targets=tuple(candidate_targets),
                    )
                )
            return
        self.gaze_history.append(
            GazeContextEntry(
                observed_at_ms=self.now_ms,
                observation=observation,
                target=target,
            )
        )
        if candidate_targets:
            self.explicit_candidate_history.append(
                CandidateTargetEntry(
                    observed_at_ms=self.now_ms,
                    targets=tuple(candidate_targets),
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

    def recent_transcripts(self, *, limit: int = 4) -> tuple[TranscriptEntry, ...]:
        if limit <= 0:
            return ()
        return tuple(list(self.transcript_history)[-limit:])

    def record_environment(self, environment: EnvironmentSnapshot, *, delta_ms: int = 0) -> None:
        self.advance(delta_ms)
        self.latest_environment = environment

    def set_active_scope(self, scope: NavigationScope, *, delta_ms: int = 0) -> None:
        self.advance(delta_ms)
        self.active_scope_entry = ActiveScopeEntry(observed_at_ms=self.now_ms, scope=scope)

    def clear_active_scope(self) -> None:
        self.active_scope_entry = None

    def active_scope(self) -> NavigationScope | None:
        if self.active_scope_entry is None:
            return None
        if self.now_ms - self.active_scope_entry.observed_at_ms > self.scope_context_window_ms:
            return None
        return self.active_scope_entry.scope

    def record_command_key(self, key: str, *, delta_ms: int = 0) -> None:
        normalized = key.strip().lower()
        if not normalized:
            return
        self.advance(delta_ms)
        self.command_history.append(CommandEntry(observed_at_ms=self.now_ms, key=normalized))

    def is_duplicate_command(self, key: str) -> bool:
        normalized = key.strip().lower()
        if not normalized:
            return False
        for entry in reversed(self.command_history):
            if self.now_ms - entry.observed_at_ms > self.dedupe_window_ms:
                break
            if entry.key == normalized:
                return True
        return False

    def latest_target(self) -> GroundedTarget | None:
        explicit = self._latest_explicit_target()
        if explicit is not None:
            return explicit
        entry = self._latest_entry()
        return entry.target if entry is not None else None

    def latest_target_age_ms(self) -> int | None:
        explicit_entry = self._latest_explicit_entry()
        if explicit_entry is not None and explicit_entry.targets:
            return self.now_ms - explicit_entry.observed_at_ms
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
        for entry in reversed(self.explicit_candidate_history):
            if self.now_ms - entry.observed_at_ms > self.gaze_context_window_ms:
                continue
            for target in entry.targets:
                if target.target_id in seen_ids:
                    continue
                seen_ids.add(target.target_id)
                candidates.append(target)
                if len(candidates) >= limit:
                    return candidates
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

    def _latest_explicit_entry(self) -> CandidateTargetEntry | None:
        for entry in reversed(self.explicit_candidate_history):
            if self.now_ms - entry.observed_at_ms <= self.gaze_context_window_ms and entry.targets:
                return entry
        return None

    def _latest_explicit_target(self) -> GroundedTarget | None:
        entry = self._latest_explicit_entry()
        if entry is None or not entry.targets:
            return None
        return entry.targets[0]
