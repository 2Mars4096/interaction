"""Helpers for capturing the current frontmost macOS application context."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from interaction.contracts import EnvironmentSnapshot

FIELD_SEPARATOR = "\x1f"
FrontmostContextRunner = Callable[[tuple[str, ...]], str]

FRONTMOST_ENVIRONMENT_SCRIPT = """
tell application "System Events"
    set frontApp to first application process whose frontmost is true
    set appName to name of frontApp
    try
        set bundleId to bundle identifier of frontApp
    on error
        set bundleId to ""
    end try
    try
        set windowTitle to name of front window of frontApp
    on error
        set windowTitle to ""
    end try
end tell
set delimiterToken to ASCII character 31
return appName & delimiterToken & bundleId & delimiterToken & windowTitle
""".strip()


def capture_frontmost_environment(
    runner: FrontmostContextRunner | None = None,
) -> EnvironmentSnapshot:
    """Capture active app, bundle id, and window title from macOS."""

    output = (runner or _default_runner)(
        ("osascript", "-e", FRONTMOST_ENVIRONMENT_SCRIPT)
    )
    return parse_frontmost_environment(output)


def parse_frontmost_environment(raw_output: str) -> EnvironmentSnapshot:
    """Parse the frontmost-app script output into an environment snapshot."""

    parts = raw_output.split(FIELD_SEPARATOR, maxsplit=2)
    while len(parts) < 3:
        parts.append("")
    app_name, bundle_id, window_title = (part.strip() or None for part in parts)
    return EnvironmentSnapshot(
        active_app=app_name,
        active_bundle_id=bundle_id,
        active_window_title=window_title,
    )


def _default_runner(argv: tuple[str, ...]) -> str:
    completed = subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
