"""Deliver live feedback immediately while retaining a replayable event history."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
import sys
from typing import Generic, TextIO, TypeVar

from interaction.ui import OverlayController

from .logging import SessionLogger, serialize_feedback_event


DEFAULT_HISTORY_LIMIT = 200
T = TypeVar("T")


class BoundedHistory(Sequence[T], Generic[T]):
    """Retain the latest entries while keeping exact lifetime counts.

    This history is presentation data, never the source of pending-action or
    gaze freshness state. Iteration and indexing expose retained entries only.
    """

    def __init__(self, limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("History limit must be a positive integer.")
        self.limit = limit
        self.total_count = 0
        self._items: deque[T] = deque(maxlen=limit)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __getitem__(self, index: int | slice) -> T | list[T]:
        if isinstance(index, slice):
            return list(self._items)[index]
        return self._items[index]

    def append(self, item: T) -> None:
        self._items.append(item)
        self.total_count += 1

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.append(item)

    def summary(self) -> dict[str, int]:
        retained = len(self)
        return {"total": self.total_count, "retained": retained, "dropped": self.total_count - retained}


class LiveEventBuffer(BoundedHistory[object]):
    """Bounded event sink: journal and overlay update when an event arrives.

    Human feedback goes to stderr so stdout remains one final JSON document.
    Callers may use the same append/extend interface as their scripted harnesses.
    Every event is journaled before the oldest in-memory event is evicted.
    """

    def __init__(self, source: str, logger: SessionLogger, overlay: OverlayController, *, stream: TextIO | None = None,
                 history_limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        super().__init__(history_limit)
        self.source = source
        self.logger = logger
        self.overlay = overlay
        self.stream = stream if stream is not None else sys.stderr
        self.snapshots: BoundedHistory[dict] = BoundedHistory(history_limit)
        self.serialized_events: BoundedHistory[dict] = BoundedHistory(history_limit)

    def append(self, event: object) -> None:
        self.logger.record_event(self.source, event)
        self.snapshots.append(self.overlay.apply_event(self.source, event).to_dict())
        payload = serialize_feedback_event(event)
        self.serialized_events.append(payload)
        super().append(event)
        message = payload["message"]
        if message:
            print(f'[{payload["phase"]}] {message}', file=self.stream, flush=True)
        if payload["transcript"] and payload["phase"] == "interpreting":
            print(f'Heard: {payload["transcript"]}', file=self.stream, flush=True)
