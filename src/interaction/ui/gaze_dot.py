"""Small optional live gaze-point visualizer for local testing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GazeDotGeometry:
    left: int
    top: int
    size: int

    @classmethod
    def from_normalized(
        cls,
        *,
        x_norm: float,
        y_norm: float,
        screen_width: int,
        screen_height: int,
        size: int = 28,
    ) -> GazeDotGeometry:
        size = max(12, int(size))
        screen_width = max(size, int(screen_width))
        screen_height = max(size, int(screen_height))
        center_x = int(_clamp(x_norm) * max(0, screen_width - 1))
        center_y = int(_clamp(y_norm) * max(0, screen_height - 1))
        left = max(0, min(screen_width - size, center_x - size // 2))
        top = max(0, min(screen_height - size, center_y - size // 2))
        return cls(left=left, top=top, size=size)


class LiveGazeDotOverlay:
    """Render a small always-on-top window that follows normalized gaze points."""

    def __init__(self, *, size: int = 28, fill: str = "#ff3b30", outline: str = "#ffffff") -> None:
        self.size = max(12, int(size))
        self.fill = fill
        self.outline = outline
        self._root = None
        self._canvas = None
        self._visible = False
        self._screen_width = 0
        self._screen_height = 0

    def open(self) -> None:
        if self._root is not None:
            return
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.85)
        except tk.TclError:
            pass
        root.configure(bg="black")
        canvas = tk.Canvas(
            root,
            width=self.size,
            height=self.size,
            bg="black",
            highlightthickness=0,
            bd=0,
        )
        canvas.pack()
        inset = max(2, self.size // 7)
        canvas.create_oval(
            inset,
            inset,
            self.size - inset,
            self.size - inset,
            fill=self.fill,
            outline=self.outline,
            width=max(1, self.size // 14),
        )
        root.update_idletasks()
        self._screen_width = max(self.size, int(root.winfo_screenwidth()))
        self._screen_height = max(self.size, int(root.winfo_screenheight()))
        self._root = root
        self._canvas = canvas

    def show_point(self, x_norm: float, y_norm: float) -> None:
        if self._root is None:
            self.open()
        assert self._root is not None
        geometry = GazeDotGeometry.from_normalized(
            x_norm=x_norm,
            y_norm=y_norm,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
            size=self.size,
        )
        self._root.geometry(f"{geometry.size}x{geometry.size}+{geometry.left}+{geometry.top}")
        if not self._visible:
            self._root.deiconify()
            self._visible = True
        self._root.update_idletasks()
        self._root.update()

    def hide(self) -> None:
        if self._root is None or not self._visible:
            return
        self._root.withdraw()
        self._root.update_idletasks()
        self._root.update()
        self._visible = False

    def close(self) -> None:
        if self._root is None:
            return
        try:
            self._root.destroy()
        finally:
            self._root = None
            self._canvas = None
            self._visible = False


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
