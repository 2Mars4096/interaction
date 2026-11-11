"""Visible, cancellable calibration targets on the primary display.

The overlay uses Tk's primary-screen logical coordinates. It is deliberately
borderless rather than using a native fullscreen transition, and must be driven
from the main thread while the caller collects camera samples.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import math
import threading
import time
from typing import Any


class CalibrationUIError(RuntimeError):
    """A calibration screen could not be shown or kept responsive."""

    def __init__(self, code: str, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.payload = payload or {}


class CalibrationCancelled(CalibrationUIError):
    def __init__(self) -> None:
        super().__init__("calibration_cancelled", "Calibration was cancelled.")


def target_center(
    x_norm: float,
    y_norm: float,
    *,
    screen_width: int,
    screen_height: int,
) -> tuple[float, float]:
    """Map a valid normalized reference point to a primary-screen position.

    Invalid coordinates are rejected instead of being moved to a different
    reference position. The target's center, rather than its outer edge, is the
    point the participant should look at.
    """
    try:
        x, y = float(x_norm), float(y_norm)
        valid = (
            math.isfinite(x)
            and math.isfinite(y)
            and 0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and isinstance(screen_width, int)
            and isinstance(screen_height, int)
            and not isinstance(screen_width, bool)
            and not isinstance(screen_height, bool)
            and screen_width > 0
            and screen_height > 0
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise CalibrationUIError(
            "invalid_calibration_target",
            "Calibration requires finite screen coordinates between zero and one and a valid screen size.",
        )
    return x * (screen_width - 1), y * (screen_height - 1)


class CalibrationTargetOverlay:
    """Render one reference target at a time and service Escape/close events.

    ``pump()`` must be called between camera frames; ``wait()`` services events
    during settling delays. A close request is raised by the next pump, never
    inside a Tk callback where it could be swallowed. Tk and time dependencies
    are injectable so tests do not require a display or real waiting.
    """

    def __init__(
        self,
        *,
        tk_module: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._tk_module = tk_module
        self._clock = clock
        self._sleep = sleep
        self._root: Any | None = None
        self._canvas: Any | None = None
        self._screen_size: tuple[int, int] | None = None
        self._cancelled = False

    @property
    def screen_size(self) -> tuple[int, int] | None:
        """Primary display's logical (width, height), available while open."""
        return self._screen_size

    def open(self) -> None:
        self._require_main_thread()
        if self._root is not None:
            self.pump()
            return
        self._cancelled = False
        try:
            tk = self._tk_module if self._tk_module is not None else importlib.import_module("tkinter")
            root = tk.Tk()
            self._root = root
            root.withdraw()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.configure(bg="#0b1220")
            width, height = int(root.winfo_screenwidth()), int(root.winfo_screenheight())
            target_center(0.5, 0.5, screen_width=width, screen_height=height)
            self._screen_size = (width, height)
            root.geometry(f"{width}x{height}+0+0")
            root.bind("<Escape>", self._cancel)
            root.protocol("WM_DELETE_WINDOW", self._cancel)
            canvas = tk.Canvas(root, width=width, height=height, bg="#0b1220", highlightthickness=0, bd=0)
            self._canvas = canvas
            canvas.pack(fill="both", expand=True)
            canvas.create_text(
                width / 2, height * 0.06,
                text="Look at the center of each dot",
                fill="#f1f5f9", font=("Helvetica", 22, "bold"), tags="heading",
            )
            canvas.create_text(
                width / 2, height * 0.11,
                text="Keep your head comfortably still. The dot will move automatically.",
                fill="#cbd5e1", font=("Helvetica", 14), tags="instructions",
            )
            canvas.create_text(
                width / 2, height * 0.89,
                text="Preparing calibration", fill="#f1f5f9", font=("Helvetica", 18), tags="progress",
            )
            canvas.create_text(
                width / 2, height * 0.95,
                text="Press Esc to cancel", fill="#cbd5e1", font=("Helvetica", 13), tags="cancel_hint",
            )
            root.deiconify()
            root.lift()
            self.pump()
            root.focus_force()
        except CalibrationUIError:
            self._dispose()
            raise
        except Exception as error:
            self._dispose()
            raise CalibrationUIError(
                "calibration_ui_unavailable",
                "A visible calibration screen could not be opened. Tk and a desktop display are required.",
                payload={"reason": type(error).__name__},
            ) from error

    def show_target(self, x_norm: float, y_norm: float, label: str, phase: str = "calibration") -> None:
        self._require_main_thread()
        if self._root is None:
            self.open()
        assert self._screen_size is not None and self._canvas is not None
        width, height = self._screen_size
        x, y = target_center(x_norm, y_norm, screen_width=width, screen_height=height)
        try:
            canvas = self._canvas
            canvas.itemconfigure(
                "heading",
                text="Checking accuracy: look at the center of each dot"
                if phase == "validation" else "Calibration: look at the center of each dot",
            )
            canvas.itemconfigure("progress", text=label)
            canvas.delete("target")
            canvas.create_oval(x - 14, y - 14, x + 14, y + 14, fill="#0e7490", outline="#67e8f9", width=2, tags="target")
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#ffffff", outline="", tags="target")
        except Exception as error:
            raise self._display_error(error) from error
        self.pump()

    def pump(self) -> None:
        self._require_main_thread()
        if self._cancelled:
            raise CalibrationCancelled()
        if self._root is None:
            raise CalibrationUIError("calibration_ui_closed", "The calibration screen is not open.")
        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception as error:
            if self._cancelled:
                raise CalibrationCancelled() from error
            raise self._display_error(error) from error
        if self._cancelled:
            raise CalibrationCancelled()

    def wait(self, ms: float) -> None:
        """Wait for a settling interval while checking for cancellation every 16 ms."""
        self._require_main_thread()
        try:
            duration = float(ms) / 1000.0
            valid = math.isfinite(duration) and duration >= 0.0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise CalibrationUIError("invalid_calibration_wait", "The calibration wait must be finite and non-negative.")
        self.pump()
        deadline = self._clock() + duration
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                return
            self._sleep(min(0.016, remaining))
            self.pump()

    def close(self) -> None:
        """Release the window; repeated cleanup, including after cancellation, is safe."""
        self._require_main_thread()
        self._dispose()

    def _cancel(self, _event: Any = None) -> None:
        self._cancelled = True

    def _dispose(self) -> None:
        root = self._root
        self._root = None
        self._canvas = None
        self._screen_size = None
        if root is not None:
            try:
                root.destroy()
            except Exception:
                # Tk also raises when the native window has already disappeared.
                pass

    @staticmethod
    def _require_main_thread() -> None:
        if threading.current_thread() is not threading.main_thread():
            raise CalibrationUIError("calibration_ui_thread", "The calibration screen must run on the main thread.")

    @staticmethod
    def _display_error(error: Exception) -> CalibrationUIError:
        return CalibrationUIError(
            "calibration_ui_failed",
            "The visible calibration screen stopped responding.",
            payload={"reason": type(error).__name__},
        )
