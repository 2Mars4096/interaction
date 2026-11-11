"""Deliver live feedback immediately while retaining a replayable event history."""

from __future__ import annotations

from collections.abc import Iterable
import sys
from typing import TextIO

from interaction.ui import OverlayController

from .logging import SessionLogger, serialize_feedback_event


class LiveEventBuffer(list[object]):
    """Append-only event sink: JSONL and overlay update when an event arrives.

    Human feedback goes to stderr so stdout remains one final JSON document.
    Callers may use the same append/extend interface as their scripted harnesses.
    """

    def __init__(self, source: str, logger: SessionLogger, overlay: OverlayController, *, stream: TextIO | None = None) -> None:
        super().__init__()
        self.source = source
        self.logger = logger
        self.overlay = overlay
        self.stream = stream if stream is not None else sys.stderr
        self.snapshots: list[dict] = []
        self.serialized_events: list[dict] = []

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

    def extend(self, events: Iterable[object]) -> None:
        for event in events:
            self.append(event)
