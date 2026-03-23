"""Terminal-friendly overlay state for repeated local testing."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OverlayState:
    source: str
    phase: str
    message: str | None = None
    transcript: str | None = None
    target_label: str | None = None
    decision: str | None = None
    result_status: str | None = None
    pending_confirmation: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class OverlayController:
    """Build a small inspectable overlay snapshot from runtime feedback events."""

    def __init__(self) -> None:
        self.state = OverlayState(source="system", phase="idle", message="No activity yet.")

    def apply_event(self, source: str, event: object) -> OverlayState:
        decision = getattr(event, "decision", None)
        result = getattr(event, "result", None)
        target = getattr(event, "target", None)
        decision_value = _decision_value(decision)
        result_value = _result_value(result)
        target_label = _target_label(target)
        self.state = OverlayState(
            source=source,
            phase=_enum_value(getattr(event, "phase", "unknown")),
            message=getattr(event, "message", None),
            transcript=getattr(event, "transcript", None),
            target_label=target_label,
            decision=decision_value,
            result_status=result_value,
            pending_confirmation=decision_value == "confirm" and result is None,
        )
        return self.state


class ConsoleOverlayRenderer:
    """Render the overlay state as a compact plain-text block."""

    def render(self, state: OverlayState) -> str:
        lines = [
            f"[{state.source}] {state.phase}",
            f"message: {state.message or '-'}",
            f"transcript: {state.transcript or '-'}",
            f"target: {state.target_label or '-'}",
            f"decision: {state.decision or '-'}",
            f"result: {state.result_status or '-'}",
            f"pending_confirmation: {'yes' if state.pending_confirmation else 'no'}",
        ]
        return "\n".join(lines)


def _enum_value(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def _decision_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        decision = value.get("decision")
        return str(decision) if decision is not None else None
    decision = getattr(value, "decision", None)
    if hasattr(decision, "value"):
        return str(getattr(decision, "value"))
    return str(decision) if decision is not None else None


def _result_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        status = value.get("status")
        return str(status) if status is not None else None
    status = getattr(value, "status", None)
    if hasattr(status, "value"):
        return str(getattr(status, "value"))
    return str(status) if status is not None else None


def _target_label(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        label = value.get("label")
        return str(label) if label is not None else None
    label = getattr(value, "label", None)
    return str(label) if label is not None else None
