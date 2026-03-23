"""Structured session logging and replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionLogRecord:
    index: int
    source: str
    phase: str
    message: str | None
    transcript: str | None
    observation: dict[str, Any] | None = None
    target: dict[str, Any] | None = None
    proposal: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source": self.source,
            "phase": self.phase,
            "message": self.message,
            "transcript": self.transcript,
            "observation": self.observation,
            "target": self.target,
            "proposal": self.proposal,
            "decision": self.decision,
            "result": self.result,
        }


class SessionLogger:
    """Append structured JSONL event records for a run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.index = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_event(self, source: str, event: object) -> SessionLogRecord:
        self.index += 1
        record = SessionLogRecord(index=self.index, source=source, **serialize_feedback_event(event))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict()) + "\n")
        return record


class SessionReplay:
    """Load and inspect structured session logs."""

    @staticmethod
    def load(path: Path) -> list[SessionLogRecord]:
        records: list[SessionLogRecord] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                records.append(SessionLogRecord(**payload))
        return records


def serialize_feedback_event(event: object) -> dict[str, Any]:
    return {
        "phase": _enum_value(getattr(event, "phase", "unknown")),
        "message": getattr(event, "message", None),
        "transcript": getattr(event, "transcript", None),
        "observation": _model_dump(getattr(event, "observation", None)),
        "target": _model_dump(getattr(event, "target", None)),
        "proposal": _model_dump(getattr(event, "proposal", None)),
        "decision": _model_dump(getattr(event, "decision", None)),
        "result": _model_dump(getattr(event, "result", None)),
    }


def _model_dump(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return None


def _enum_value(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)
