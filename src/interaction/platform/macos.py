"""macOS platform scaffolding for the first bounded control slice."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from typing import Any

from interaction.contracts import ActionName, ExecutionRequest, ExecutionResult, ExecutionStatus

from .base import CommandSpec, PlatformAdapter, PlatformCapabilities

Runner = Callable[[tuple[str, ...]], None]
Point = tuple[float, float]

KEY_CODE_MAP = {
    "escape": 53,
    "tab": 48,
    "enter": 36,
    "back": 51,
    "page_up": 116,
    "page_down": 121,
}


class MacOSPlatformAdapter(PlatformAdapter):
    """Thin command-planning adapter for early macOS experiments."""

    def __init__(self, *, dry_run: bool = True, runner: Runner | None = None) -> None:
        self.dry_run = dry_run
        self.runner = runner or self._default_runner

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        plan = self.plan(request)
        if plan is None:
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                message=f"Action {request.proposal.action.value} is not yet supported by the macOS adapter.",
                proposal=request.proposal,
                details={"dry_run": self.dry_run},
            )

        if self.dry_run:
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                message="Dry-run execution prepared successfully.",
                proposal=request.proposal,
                details={
                    "dry_run": True,
                    "commands": [list(spec.argv) for spec in plan],
                },
            )

        try:
            for spec in plan:
                self.runner(spec.argv)
        except Exception as exc:  # pragma: no cover - defensive live-execution guard
            return ExecutionResult(
                status=ExecutionStatus.FAILURE,
                message=f"macOS action execution failed: {exc}",
                proposal=request.proposal,
                details={"dry_run": False},
            )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            message="macOS action executed successfully.",
            proposal=request.proposal,
            details={"dry_run": False},
        )

    def describe_capabilities(self) -> PlatformCapabilities:
        return PlatformCapabilities(
            platform_name="macos",
            supported_actions=(
                ActionName.HIGHLIGHT_TARGET.value,
                ActionName.FOCUS_TARGET.value,
                ActionName.CLICK_TARGET.value,
                ActionName.DOUBLE_CLICK_TARGET.value,
                ActionName.RIGHT_CLICK_TARGET.value,
                ActionName.DRAG_TARGET.value,
                ActionName.SCROLL.value,
                ActionName.OPEN_APP.value,
                ActionName.SWITCH_APP.value,
                ActionName.PRESS_KEY.value,
                ActionName.TYPE_TEXT.value,
            ),
            coordinate_actions=(
                ActionName.FOCUS_TARGET.value,
                ActionName.CLICK_TARGET.value,
                ActionName.DOUBLE_CLICK_TARGET.value,
                ActionName.RIGHT_CLICK_TARGET.value,
                ActionName.DRAG_TARGET.value,
            ),
            notes=(
                "Target-only pointer actions can use normalized gaze points or normalized target bounds.",
                "Live pointer actions require macOS accessibility permissions.",
            ),
        )

    def plan(self, request: ExecutionRequest) -> list[CommandSpec] | None:
        proposal = request.proposal
        action = proposal.action
        point = self._extract_screen_point(proposal.arguments)
        normalized_point = self._extract_normalized_point(proposal.arguments)
        if normalized_point is None:
            normalized_point = self._extract_target_bounds_center(proposal.arguments)
        if action in {ActionName.OPEN_APP, ActionName.SWITCH_APP}:
            app_name = str(proposal.arguments.get("app_name", "")).strip()
            if not app_name:
                return None
            return [CommandSpec(("open", "-a", app_name))]
        if action == ActionName.HIGHLIGHT_TARGET:
            return [self._notification_spec("Interaction Highlight", self._target_message(proposal.arguments, fallback="Would highlight the current target."))]
        if action == ActionName.FOCUS_TARGET:
            if point is not None:
                return [self._python_runtime_spec("move", point)]
            if normalized_point is not None:
                return [self._python_runtime_spec("move", normalized_point, normalized=True)]
            if "app_name" in proposal.arguments:
                app_name = str(proposal.arguments.get("app_name", "")).strip()
                if app_name:
                    return [CommandSpec(("open", "-a", app_name))]
            return [self._notification_spec("Interaction Focus", self._target_message(proposal.arguments, fallback="Would focus the current target."))]
        if action == ActionName.PRESS_KEY:
            key = str(proposal.arguments.get("key", "")).strip().lower()
            if key not in KEY_CODE_MAP:
                return None
            script = f'tell application "System Events" to key code {KEY_CODE_MAP[key]}'
            return [CommandSpec(("osascript", "-e", script))]
        if action == ActionName.SCROLL:
            direction = str(proposal.arguments.get("direction", "")).strip().lower()
            if direction == "up":
                script = f'tell application "System Events" to key code {KEY_CODE_MAP["page_up"]}'
                return [CommandSpec(("osascript", "-e", script))]
            if direction == "down":
                script = f'tell application "System Events" to key code {KEY_CODE_MAP["page_down"]}'
                return [CommandSpec(("osascript", "-e", script))]
            return None
        if action == ActionName.TYPE_TEXT:
            text = str(proposal.arguments.get("text", "")).strip()
            if not text:
                return None
            escaped = self._escape_applescript(text)
            script = f'tell application "System Events" to keystroke "{escaped}"'
            return [CommandSpec(("osascript", "-e", script))]
        if action == ActionName.DRAG_TARGET:
            drag_plan = self._drag_spec(proposal.arguments)
            if drag_plan is not None:
                return [drag_plan]
            if self.dry_run:
                return [self._notification_spec("Interaction Pointer", self._target_message(proposal.arguments, fallback="Would drag from the current source to the current target."))]
            return None
        if action in {ActionName.CLICK_TARGET, ActionName.DOUBLE_CLICK_TARGET, ActionName.RIGHT_CLICK_TARGET}:
            if point is None:
                if normalized_point is not None:
                    command = self._pointer_command(action)
                    return [self._python_runtime_spec(command, normalized_point, normalized=True)]
                if self.dry_run:
                    command = self._pointer_message(action)
                    return [self._notification_spec("Interaction Pointer", self._target_message(proposal.arguments, fallback=command))]
                return None
            command = self._pointer_command(action)
            return [self._python_runtime_spec(command, point)]
        return None

    @staticmethod
    def _default_runner(argv: tuple[str, ...]) -> None:
        subprocess.run(argv, check=True)

    def _notification_spec(self, title: str, message: str) -> CommandSpec:
        escaped_title = self._escape_applescript(title)
        escaped_message = self._escape_applescript(message)
        script = f'display notification "{escaped_message}" with title "{escaped_title}"'
        return CommandSpec(("osascript", "-e", script))

    def _python_runtime_spec(self, command: str, point: Point, *, normalized: bool = False) -> CommandSpec:
        x, y = point
        runtime_command = f"{command}-normalized" if normalized else command
        return CommandSpec(
            (
                sys.executable,
                "-m",
                "interaction.platform.macos_runtime",
                runtime_command,
                f"{x}",
                f"{y}",
            )
        )

    def _drag_spec(self, arguments: dict[str, Any]) -> CommandSpec | None:
        start_point = self._extract_prefixed_screen_point(arguments, prefix="start_")
        end_point = self._extract_prefixed_screen_point(arguments, prefix="end_")
        if start_point is not None and end_point is not None:
            return self._python_runtime_multi_spec("drag", start_point, end_point)

        start_normalized = self._extract_prefixed_normalized_point(arguments, prefix="start_")
        end_normalized = self._extract_prefixed_normalized_point(arguments, prefix="end_")
        if start_normalized is None:
            start_normalized = self._extract_prefixed_target_bounds_center(arguments, prefix="start_")
        if end_normalized is None:
            end_normalized = self._extract_prefixed_target_bounds_center(arguments, prefix="end_")
        if start_normalized is not None and end_normalized is not None:
            return self._python_runtime_multi_spec("drag-normalized", start_normalized, end_normalized)
        return None

    def _python_runtime_multi_spec(self, command: str, *points: Point) -> CommandSpec:
        coordinates: list[str] = []
        for x, y in points:
            coordinates.extend((f"{x}", f"{y}"))
        return CommandSpec(
            (
                sys.executable,
                "-m",
                "interaction.platform.macos_runtime",
                command,
                *coordinates,
            )
        )

    @staticmethod
    def _extract_screen_point(arguments: dict[str, Any]) -> Point | None:
        raw_point = arguments.get("screen_point")
        return MacOSPlatformAdapter._coerce_point_dict(raw_point, clamp=False)

    @staticmethod
    def _extract_prefixed_screen_point(arguments: dict[str, Any], *, prefix: str) -> Point | None:
        raw_point = arguments.get(f"{prefix}screen_point")
        return MacOSPlatformAdapter._coerce_point_dict(raw_point, clamp=False)

    @staticmethod
    def _extract_normalized_point(arguments: dict[str, Any]) -> Point | None:
        raw_point = arguments.get("normalized_point")
        return MacOSPlatformAdapter._coerce_point_dict(raw_point, clamp=True)

    @staticmethod
    def _extract_prefixed_normalized_point(arguments: dict[str, Any], *, prefix: str) -> Point | None:
        raw_point = arguments.get(f"{prefix}normalized_point")
        return MacOSPlatformAdapter._coerce_point_dict(raw_point, clamp=True)

    @staticmethod
    def _coerce_point_dict(raw_point: Any, *, clamp: bool) -> Point | None:
        if not isinstance(raw_point, dict):
            return None
        x = raw_point.get("x")
        y = raw_point.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        if clamp:
            return MacOSPlatformAdapter._clamp(float(x)), MacOSPlatformAdapter._clamp(float(y))
        return float(x), float(y)

    @staticmethod
    def _extract_target_bounds_center(arguments: dict[str, Any]) -> Point | None:
        raw_bounds = arguments.get("target_bounds")
        return MacOSPlatformAdapter._bounds_center(raw_bounds)

    @staticmethod
    def _extract_prefixed_target_bounds_center(arguments: dict[str, Any], *, prefix: str) -> Point | None:
        raw_bounds = arguments.get(f"{prefix}target_bounds")
        return MacOSPlatformAdapter._bounds_center(raw_bounds)

    @staticmethod
    def _bounds_center(raw_bounds: Any) -> Point | None:
        if not isinstance(raw_bounds, dict):
            return None
        x = raw_bounds.get("x")
        y = raw_bounds.get("y")
        width = raw_bounds.get("width")
        height = raw_bounds.get("height")
        if not all(isinstance(value, (int, float)) for value in (x, y, width, height)):
            return None
        return (
            MacOSPlatformAdapter._clamp(float(x) + float(width) / 2.0),
            MacOSPlatformAdapter._clamp(float(y) + float(height) / 2.0),
        )

    @staticmethod
    def _target_message(arguments: dict[str, Any], *, fallback: str) -> str:
        label = str(arguments.get("target_label", "")).strip()
        target_ref = str(arguments.get("target_ref", "")).strip()
        if label:
            return f"Target: {label}"
        if target_ref:
            return f"Target ref: {target_ref}"
        return fallback

    @staticmethod
    def _escape_applescript(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _pointer_command(action: ActionName) -> str:
        if action == ActionName.DRAG_TARGET:
            return "drag"
        if action == ActionName.DOUBLE_CLICK_TARGET:
            return "double-click"
        if action == ActionName.RIGHT_CLICK_TARGET:
            return "right-click"
        return "click"

    @staticmethod
    def _pointer_message(action: ActionName) -> str:
        if action == ActionName.DRAG_TARGET:
            return "Would drag from the current source to the current target."
        if action == ActionName.DOUBLE_CLICK_TARGET:
            return "Would double-click the current target."
        if action == ActionName.RIGHT_CLICK_TARGET:
            return "Would right-click the current target."
        return "Would click the current target."

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
