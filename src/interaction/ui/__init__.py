"""Overlay state and rendering helpers."""

from .camera_preview import LiveWebcamDebugPreview
from .gaze_dot import CalibrationDotOverlay, GazeDotGeometry, LiveGazeDotOverlay, LiveSnapDotsOverlay
from .overlay import ConsoleOverlayRenderer, OverlayController, OverlayState
from .shell import ShellViewState, TerminalShellRenderer, build_shell_view_state, infer_shell_surface
from .shell_gui import ShellDemoController, ShellDemoRunResult, run_shell_gui_demo

__all__ = [
    "CalibrationDotOverlay",
    "ConsoleOverlayRenderer",
    "GazeDotGeometry",
    "LiveWebcamDebugPreview",
    "LiveGazeDotOverlay",
    "LiveSnapDotsOverlay",
    "OverlayController",
    "OverlayState",
    "ShellViewState",
    "ShellDemoController",
    "ShellDemoRunResult",
    "TerminalShellRenderer",
    "build_shell_view_state",
    "infer_shell_surface",
    "run_shell_gui_demo",
]
