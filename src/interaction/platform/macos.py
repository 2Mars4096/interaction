"""macOS platform scaffolding for the first bounded control slice."""

from __future__ import annotations

import ctypes
from ctypes import util
import importlib
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from interaction.contracts import (
    ActionName,
    DesktopTargetCandidate,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ScreenPoint,
    StructuredNavigationRequest,
    TargetCandidateSource,
)

from .base import CommandSpec, PlatformAdapter, PlatformCapabilities, TargetCandidateQuery
from .macos_profiles import (
    DEFAULT_MACOS_PROFILE_REGISTRY,
    MacOSProfileRegistry,
    ShortcutModifier,
    ShortcutPress,
)

Runner = Callable[[tuple[str, ...]], None]
CandidateCollector = Callable[[TargetCandidateQuery], tuple[DesktopTargetCandidate, ...]]
Point = tuple[float, float]
CFTypeRef = ctypes.c_void_p
AXUIElementRef = ctypes.c_void_p
UTF8_ENCODING = 0x08000100

KEY_CODE_MAP = {
    "escape": 53,
    "tab": 48,
    "enter": 36,
    "back": 51,
    "page_up": 116,
    "page_down": 121,
}

MODIFIER_ORDER = (
    ShortcutModifier.CONTROL,
    ShortcutModifier.OPTION,
    ShortcutModifier.COMMAND,
    ShortcutModifier.SHIFT,
)


class MacOSPlatformAdapter(PlatformAdapter):
    """Thin command-planning adapter for early macOS experiments."""

    def __init__(
        self,
        *,
        dry_run: bool = True,
        runner: Runner | None = None,
        target_candidate_collector: CandidateCollector | None = None,
        profile_registry: MacOSProfileRegistry | None = None,
    ) -> None:
        self.dry_run = dry_run
        self.runner = runner or self._default_runner
        self.target_candidate_collector = target_candidate_collector or _collect_accessibility_candidates
        self.profile_registry = profile_registry or DEFAULT_MACOS_PROFILE_REGISTRY

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if (not self.dry_run and request.proposal.action == ActionName.TYPE_TEXT
                and int(request.proposal.arguments.get("delete_characters", 0) or 0) > 0):
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                message="Automatic text correction is disabled because the current field and caret cannot be verified. Edit the text manually.",
                proposal=request.proposal, details={"dry_run": False, "reason": "unverified_text_ownership"},
            )
        plan_details: dict[str, Any] = {}
        blocked_message: str | None = None
        if request.proposal.action == ActionName.NAVIGATE_STRUCTURE:
            plan, plan_details, blocked_message = self._structured_navigation_plan(request)
        else:
            plan = self.plan(request)
        if plan is None:
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                message=blocked_message or f"Action {request.proposal.action.value} is not yet supported by the macOS adapter.",
                proposal=request.proposal,
                details={"dry_run": self.dry_run, **plan_details},
            )

        if self.dry_run:
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                message="Dry-run execution prepared successfully. macOS input was not executed; re-run with --execute for real control.",
                proposal=request.proposal,
                details={
                    "dry_run": True,
                    "commands": [list(spec.argv) for spec in plan],
                    **plan_details,
                },
            )

        try:
            for spec in plan:
                self._run_spec(spec)
        except Exception as exc:  # pragma: no cover - defensive live-execution guard
            return ExecutionResult(
                status=ExecutionStatus.FAILURE,
                message=f"macOS action execution failed: {exc}",
                proposal=request.proposal,
                details={"dry_run": False, **plan_details},
            )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            message="macOS action executed successfully.",
            proposal=request.proposal,
            details={"dry_run": False, **plan_details},
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
                ActionName.NAVIGATE_STRUCTURE.value,
                ActionName.TYPE_TEXT.value,
            ),
            coordinate_actions=(
                ActionName.FOCUS_TARGET.value,
                ActionName.CLICK_TARGET.value,
                ActionName.DOUBLE_CLICK_TARGET.value,
                ActionName.RIGHT_CLICK_TARGET.value,
                ActionName.DRAG_TARGET.value,
            ),
            candidate_collection=True,
            structured_navigation_scopes=self.profile_registry.supported_scopes(),
            app_profiles=self.profile_registry.profile_ids(),
            notes=(
                "Structured navigation resolves through deterministic macOS app profiles.",
                "Target-only pointer actions can use normalized gaze points or normalized target bounds.",
                "Live pointer actions require macOS accessibility permissions.",
                "Desktop target candidates can be collected via macOS accessibility hit testing.",
            ),
        )

    def collect_target_candidates(self, query: TargetCandidateQuery) -> tuple[DesktopTargetCandidate, ...]:
        try:
            candidates = self.target_candidate_collector(query)
        except Exception:
            return ()
        return tuple(candidates[: max(1, query.limit)])

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
            delete_characters = max(0, int(proposal.arguments.get("delete_characters", 0) or 0))
            raw_text = proposal.arguments.get("text", "")
            text = str(raw_text) if raw_text is not None else ""
            if delete_characters <= 0 and not text:
                return None
            script_parts: list[str] = ['tell application "System Events"']
            if delete_characters > 0:
                script_parts.append(f"repeat {delete_characters} times")
                script_parts.append("key code 51")
                script_parts.append("end repeat")
            if text:
                escaped = self._escape_applescript(text)
                script_parts.append(f'keystroke "{escaped}"')
            script_parts.append("end tell")
            script = "\n".join(script_parts)
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

    def _run_spec(self, spec: CommandSpec) -> None:
        argv = spec.argv
        if len(argv) >= 5 and argv[0] == sys.executable and argv[1] == "-m" and argv[2] == "interaction.platform.macos_runtime":
            self._run_runtime_command(argv[3], argv[4:])
            return
        self.runner(argv)

    @staticmethod
    def _run_runtime_command(command: str, coordinate_args: tuple[str, ...]) -> None:
        runtime = importlib.import_module("interaction.platform.macos_runtime")
        command_name = command.replace("-", "_")
        handler = getattr(runtime, command_name, None)
        if not callable(handler):
            raise RuntimeError(f"Unsupported macOS runtime command: {command}")
        coordinates = tuple(float(value) for value in coordinate_args)
        handler(*coordinates)

    def _notification_spec(self, title: str, message: str) -> CommandSpec:
        escaped_title = self._escape_applescript(title)
        escaped_message = self._escape_applescript(message)
        script = f'display notification "{escaped_message}" with title "{escaped_title}"'
        return CommandSpec(("osascript", "-e", script))

    def _structured_navigation_plan(
        self,
        request: ExecutionRequest,
    ) -> tuple[list[CommandSpec] | None, dict[str, Any], str | None]:
        try:
            navigation = StructuredNavigationRequest.from_action_arguments(request.proposal.arguments)
        except Exception:
            return None, {}, "Structured navigation arguments are malformed."

        resolution = self.profile_registry.resolve_navigation(request.environment, navigation)
        details: dict[str, Any] = {
            "navigation": navigation.model_dump(mode="json"),
        }
        if resolution.profile is not None:
            details["profile_id"] = resolution.profile.profile_id
        if resolution.fallback_profile_id is not None:
            details["fallback_profile_id"] = resolution.fallback_profile_id
        if not resolution.supported:
            details["profile_supported"] = False
            return (
                None,
                details,
                resolution.message or "Structured navigation could not be resolved for the active app profile.",
            )
        details["profile_supported"] = True
        return list(self._shortcut_specs(resolution.shortcut_presses)), details, None

    def _shortcut_specs(
        self,
        shortcut_presses: tuple[ShortcutPress, ...],
    ) -> tuple[CommandSpec, ...]:
        return tuple(self._shortcut_spec(shortcut) for shortcut in shortcut_presses)

    def _shortcut_spec(self, shortcut: ShortcutPress) -> CommandSpec:
        if shortcut.key_code is not None:
            key_action = f"key code {shortcut.key_code}"
        else:
            escaped_keystroke = self._escape_applescript(shortcut.keystroke or "")
            key_action = f'keystroke "{escaped_keystroke}"'
        modifiers = self._modifier_clause(shortcut.modifiers)
        script = f'tell application "System Events" to {key_action}{modifiers}'
        return CommandSpec(("osascript", "-e", script))

    def _modifier_clause(
        self,
        modifiers: tuple[ShortcutModifier, ...],
    ) -> str:
        ordered_modifiers = [
            modifier for modifier in MODIFIER_ORDER if modifier in set(modifiers)
        ]
        if not ordered_modifiers:
            return ""
        apple_modifiers = ", ".join(f"{modifier.value} down" for modifier in ordered_modifiers)
        return f" using {{{apple_modifiers}}}"

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


def normalized_to_screen_point(x_norm: float, y_norm: float) -> ScreenPoint:
    library = util.find_library("ApplicationServices")
    if not library:
        raise RuntimeError("ApplicationServices framework not found")
    quartz = ctypes.cdll.LoadLibrary(library)
    quartz.CGMainDisplayID.restype = ctypes.c_uint32
    quartz.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]
    quartz.CGDisplayPixelsWide.restype = ctypes.c_size_t
    quartz.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
    quartz.CGDisplayPixelsHigh.restype = ctypes.c_size_t
    display_id = quartz.CGMainDisplayID()
    width = max(1, int(quartz.CGDisplayPixelsWide(display_id)))
    height = max(1, int(quartz.CGDisplayPixelsHigh(display_id)))
    return ScreenPoint(
        x=MacOSPlatformAdapter._clamp(float(x_norm)) * (width - 1),
        y=MacOSPlatformAdapter._clamp(float(y_norm)) * (height - 1),
    )


def _collect_accessibility_candidates(query: TargetCandidateQuery) -> tuple[DesktopTargetCandidate, ...]:
    runtime = _load_accessibility_runtime()
    if runtime is None:
        return ()

    seen: dict[str, DesktopTargetCandidate] = {}
    for point in _candidate_probe_points(query):
        element = runtime.copy_element_at_position(point)
        if element is None:
            continue
        try:
            candidate = runtime.element_to_candidate(element, point)
        finally:
            runtime.release(element)
        if candidate is None:
            continue
        if query.require_actionable and candidate.invocable is False and not candidate.available_actions:
            continue
        if query.active_app and candidate.app_name and candidate.app_name != query.active_app:
            continue
        seen.setdefault(candidate.candidate_id, candidate)
        if len(seen) >= query.limit:
            break
    return tuple(seen.values())


def _candidate_probe_points(query: TargetCandidateQuery) -> tuple[ScreenPoint, ...]:
    center = query.screen_point
    radius = max(8.0, float(query.radius_px))
    offsets = (
        (0.0, 0.0),
        (radius * 0.45, 0.0),
        (-radius * 0.45, 0.0),
        (0.0, radius * 0.45),
        (0.0, -radius * 0.45),
        (radius * 0.35, radius * 0.35),
        (-radius * 0.35, radius * 0.35),
        (radius * 0.35, -radius * 0.35),
        (-radius * 0.35, -radius * 0.35),
    )
    return tuple(
        ScreenPoint(x=center.x + dx, y=center.y + dy)
        for dx, dy in offsets[: max(1, query.limit)]
    )


class _MacOSAccessibilityRuntime:
    def __init__(self, *, ax: ctypes.CDLL, cf: ctypes.CDLL) -> None:
        self.ax = ax
        self.cf = cf
        self.system_wide = self.ax.AXUIElementCreateSystemWide()

    def release(self, value: CFTypeRef | None) -> None:
        if value:
            self.cf.CFRelease(value)

    def copy_element_at_position(self, point: ScreenPoint) -> AXUIElementRef | None:
        if not self.system_wide:
            return None
        result = AXUIElementRef()
        status = self.ax.AXUIElementCopyElementAtPosition(
            self.system_wide,
            float(point.x),
            float(point.y),
            ctypes.byref(result),
        )
        if status != 0 or not result:
            return None
        return result

    def element_to_candidate(
        self,
        element: AXUIElementRef,
        point: ScreenPoint,
    ) -> DesktopTargetCandidate | None:
        pid = self._copy_pid(element)
        role = self._copy_string_attribute(element, "AXRole")
        subrole = self._copy_string_attribute(element, "AXSubrole")
        identifier = self._copy_string_attribute(element, "AXIdentifier")
        title = self._copy_string_attribute(element, "AXTitle")
        description = self._copy_string_attribute(element, "AXDescription")
        value_text = self._copy_string_attribute(element, "AXValue")
        enabled = self._copy_bool_attribute(element, "AXEnabled")
        window_title = self._copy_window_title(element)
        actions = self._copy_actions(element)
        app_name = _process_name_for_pid(pid)
        label = title or description or value_text or role or "desktop target"
        invocable = bool(actions) or role in {"AXButton", "AXMenuItem", "AXTabGroup", "AXLink"}
        candidate_id = _candidate_id(
            pid=pid,
            identifier=identifier,
            role=role,
            subrole=subrole,
            label=label,
            point=point,
        )
        try:
            return DesktopTargetCandidate(
                candidate_id=candidate_id,
                source=TargetCandidateSource.ACCESSIBILITY,
                display_label=label,
                label=title or description,
                role=role,
                subrole=subrole,
                value_text=value_text,
                screen_point=point,
                enabled=enabled,
                focusable=None,
                editable=role in {"AXTextField", "AXTextArea", "AXComboBox"},
                invocable=invocable,
                available_actions=actions,
                app_name=app_name,
                pid=pid,
                window_title=window_title,
            )
        except Exception:
            return None

    def _copy_pid(self, element: AXUIElementRef) -> int | None:
        pid = ctypes.c_int(0)
        status = self.ax.AXUIElementGetPid(element, ctypes.byref(pid))
        if status != 0:
            return None
        return max(0, int(pid.value))

    def _copy_window_title(self, element: AXUIElementRef) -> str | None:
        window = self._copy_attribute_value(element, "AXWindow")
        if window is None:
            return None
        try:
            return self._copy_string_attribute(window, "AXTitle")
        finally:
            self.release(window)

    def _copy_actions(self, element: AXUIElementRef) -> list[str]:
        actions = CFTypeRef()
        status = self.ax.AXUIElementCopyActionNames(element, ctypes.byref(actions))
        if status != 0 or not actions:
            return []
        try:
            return self._cf_array_to_strings(actions)
        finally:
            self.release(actions)

    def _copy_string_attribute(self, element: AXUIElementRef, name: str) -> str | None:
        value = self._copy_attribute_value(element, name)
        if value is None:
            return None
        try:
            return self._cfstring_to_str(value)
        finally:
            self.release(value)

    def _copy_bool_attribute(self, element: AXUIElementRef, name: str) -> bool | None:
        value = self._copy_attribute_value(element, name)
        if value is None:
            return None
        try:
            if self.cf.CFGetTypeID(value) != self.cf.CFBooleanGetTypeID():
                return None
            return bool(self.cf.CFBooleanGetValue(value))
        finally:
            self.release(value)

    def _copy_attribute_value(self, element: AXUIElementRef, name: str) -> CFTypeRef | None:
        attribute = self._cfstring_from_str(name)
        if attribute is None:
            return None
        value = CFTypeRef()
        try:
            status = self.ax.AXUIElementCopyAttributeValue(element, attribute, ctypes.byref(value))
        finally:
            self.release(attribute)
        if status != 0 or not value:
            return None
        return value

    def _cfstring_from_str(self, value: str) -> CFTypeRef | None:
        return self.cf.CFStringCreateWithCString(None, value.encode("utf-8"), UTF8_ENCODING)

    def _cfstring_to_str(self, value: CFTypeRef) -> str | None:
        if self.cf.CFGetTypeID(value) != self.cf.CFStringGetTypeID():
            return None
        buffer = ctypes.create_string_buffer(4096)
        success = self.cf.CFStringGetCString(value, buffer, len(buffer), UTF8_ENCODING)
        if not success:
            return None
        return buffer.value.decode("utf-8").strip() or None

    def _cf_array_to_strings(self, value: CFTypeRef) -> list[str]:
        count = int(self.cf.CFArrayGetCount(value))
        items: list[str] = []
        for index in range(count):
            item = self.cf.CFArrayGetValueAtIndex(value, index)
            text = self._cfstring_to_str(item)
            if text:
                items.append(text)
        return items


def _load_accessibility_runtime() -> _MacOSAccessibilityRuntime | None:
    application_services = util.find_library("ApplicationServices")
    core_foundation = util.find_library("CoreFoundation")
    if not application_services or not core_foundation:
        return None
    try:
        ax = ctypes.cdll.LoadLibrary(application_services)
        cf = ctypes.cdll.LoadLibrary(core_foundation)
    except OSError:
        return None

    try:
        ax.AXUIElementCreateSystemWide.restype = AXUIElementRef
        ax.AXUIElementCopyElementAtPosition.argtypes = [
            AXUIElementRef,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(AXUIElementRef),
        ]
        ax.AXUIElementCopyElementAtPosition.restype = ctypes.c_int
        ax.AXUIElementCopyAttributeValue.argtypes = [
            AXUIElementRef,
            CFTypeRef,
            ctypes.POINTER(CFTypeRef),
        ]
        ax.AXUIElementCopyAttributeValue.restype = ctypes.c_int
        ax.AXUIElementCopyActionNames.argtypes = [
            AXUIElementRef,
            ctypes.POINTER(CFTypeRef),
        ]
        ax.AXUIElementCopyActionNames.restype = ctypes.c_int
        ax.AXUIElementGetPid.argtypes = [AXUIElementRef, ctypes.POINTER(ctypes.c_int)]
        ax.AXUIElementGetPid.restype = ctypes.c_int

        cf.CFRelease.argtypes = [CFTypeRef]
        cf.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        cf.CFStringCreateWithCString.restype = CFTypeRef
        cf.CFStringGetCString.argtypes = [
            CFTypeRef,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_uint32,
        ]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetTypeID.restype = ctypes.c_ulong
        cf.CFGetTypeID.argtypes = [CFTypeRef]
        cf.CFGetTypeID.restype = ctypes.c_ulong
        cf.CFBooleanGetTypeID.restype = ctypes.c_ulong
        cf.CFBooleanGetValue.argtypes = [CFTypeRef]
        cf.CFBooleanGetValue.restype = ctypes.c_bool
        cf.CFArrayGetCount.argtypes = [CFTypeRef]
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetValueAtIndex.argtypes = [CFTypeRef, ctypes.c_long]
        cf.CFArrayGetValueAtIndex.restype = CFTypeRef
    except AttributeError:
        return None

    try:
        return _MacOSAccessibilityRuntime(ax=ax, cf=cf)
    except Exception:
        return None


def _candidate_id(
    *,
    pid: int | None,
    identifier: str | None,
    role: str | None,
    subrole: str | None,
    label: str,
    point: ScreenPoint,
) -> str:
    stable_id = identifier or label or role or "candidate"
    normalized = "".join(char if char.isalnum() else "-" for char in stable_id.lower()).strip("-") or "candidate"
    return (
        f"ax-{pid or 0}-{role or 'unknown'}-{subrole or 'unknown'}-"
        f"{normalized}-{int(round(point.x))}-{int(round(point.y))}"
    )


def _process_name_for_pid(pid: int | None) -> str | None:
    if pid is None or pid <= 0:
        return None
    try:
        result = subprocess.run(
            ("ps", "-p", str(pid), "-o", "comm="),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None
