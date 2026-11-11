"""Bounded, read-only foreground application inspection.

An unavailable probe yields unknown context rather than a fabricated application.
The snapshot deliberately excludes field values, document content, and screen data.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable

from interaction.contracts import EnvironmentSnapshot, PlatformName


_FOREGROUND_SCRIPT = """
function run() {
    var result = {active_app: null, active_window_title: null};
    var system = Application('System Events');
    var processes = system.applicationProcesses.whose({frontmost: true})();
    if (processes.length === 0) return JSON.stringify(result);
    var process = processes[0];
    result.active_app = process.name();
    try {
        var windows = process.windows();
        if (windows.length > 0) result.active_window_title = windows[0].name();
    } catch (error) {
        // A windowless app or unavailable Accessibility data is still useful
        // application context; never reuse an earlier window title.
    }
    return JSON.stringify(result);
}
"""


def capture_environment(
    *,
    platform: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    timeout_s: float = 1.5,
) -> EnvironmentSnapshot:
    """Read the frontmost app/window on macOS without activating anything.

    ``runner`` and ``platform`` can be injected for tests. The subprocess has a
    finite timeout so a denied or unresponsive Accessibility/Automation request
    cannot block the interaction loop indefinitely.
    """
    platform = sys.platform if platform is None else platform
    platform_name = _platform_name(platform)
    unknown = EnvironmentSnapshot(platform=platform_name)
    if platform_name != PlatformName.MACOS:
        return unknown
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        return unknown
    try:
        completed = (runner or subprocess.run)(
            ["osascript", "-l", "JavaScript", "-e", _FOREGROUND_SCRIPT],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if completed.returncode != 0:
            return unknown
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return unknown
    if not isinstance(payload, dict):
        return unknown

    app = _optional_text(payload.get("active_app"))
    # A window title without an identified owning app is not a usable context.
    if app is None:
        return unknown
    return EnvironmentSnapshot(
        platform=platform_name,
        active_app=app,
        active_window_title=_optional_text(payload.get("active_window_title")),
    )


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _platform_name(platform: str) -> PlatformName:
    if platform == "darwin":
        return PlatformName.MACOS
    if platform.startswith("linux"):
        return PlatformName.LINUX
    if platform in {"win32", "cygwin"}:
        return PlatformName.WINDOWS
    return PlatformName.UNKNOWN
