"""Small optional live gaze-point visualizers for local testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from interaction.vision.provider_common import clamp_unit_interval


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
        center_x = int(clamp_unit_interval(x_norm) * max(0, screen_width - 1))
        center_y = int(clamp_unit_interval(y_norm) * max(0, screen_height - 1))
        left = max(0, min(screen_width - size, center_x - size // 2))
        top = max(0, min(screen_height - size, center_y - size // 2))
        return cls(left=left, top=top, size=size)


class LiveGazeDotOverlay:
    """Render a small always-on-top window that follows normalized gaze points."""

    def __init__(self, *, size: int = 28, fill: str = "#ff3b30", outline: str = "#ffffff") -> None:
        self.size = max(12, int(size))
        self.fill = fill
        self.outline = outline
        self._background = "#ff00ff"
        self._root = None
        self._canvas = None
        self._visible = False
        self._screen_width = 0
        self._screen_height = 0
        self._ring_id = None
        self._core_id = None

    def open(self) -> None:
        if self._root is not None:
            return
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg=self._background)
        transparent_configured = False
        try:
            root.wm_attributes("-transparentcolor", self._background)
            transparent_configured = True
        except tk.TclError:
            try:
                root.configure(bg="systemTransparent")
                root.attributes("-transparent", True)
                transparent_configured = True
            except tk.TclError:
                root.configure(bg=self._background)
        if not transparent_configured:
            try:
                root.attributes("-alpha", 0.92)
            except tk.TclError:
                pass
        canvas = tk.Canvas(
            root,
            width=self.size,
            height=self.size,
            bg=root.cget("bg"),
            highlightthickness=0,
            bd=0,
        )
        canvas.pack()
        canvas.create_oval(
            1,
            1,
            self.size - 1,
            self.size - 1,
            fill="",
            outline="#000000",
            width=max(1, self.size // 12),
        )
        inset = max(3, self.size // 7)
        self._ring_id = canvas.create_oval(
            inset // 2,
            inset // 2,
            self.size - inset // 2,
            self.size - inset // 2,
            fill="",
            outline=self.outline,
            width=max(2, self.size // 12),
        )
        self._core_id = canvas.create_oval(
            inset,
            inset,
            self.size - inset,
            self.size - inset,
            fill=self.fill,
            outline=self.outline,
            width=max(2, self.size // 14),
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
        try:
            self._root.lift()
        except Exception:
            pass
        try:
            self._root.attributes("-topmost", True)
        except Exception:
            pass
        self._root.update_idletasks()
        self._root.update()

    def set_style(self, *, fill: str | None = None, outline: str | None = None) -> None:
        if fill is not None:
            self.fill = fill
        if outline is not None:
            self.outline = outline
        if self._canvas is None:
            return
        try:
            if self._ring_id is not None:
                self._canvas.itemconfig(self._ring_id, outline=self.outline)
            if self._core_id is not None:
                self._canvas.itemconfig(self._core_id, fill=self.fill, outline=self.outline)
            self._canvas.update_idletasks()
        except Exception:
            return

    def wait_for_confirmation(self) -> None:
        if self._root is None:
            self.open()
        assert self._root is not None
        import tkinter as tk

        confirmed = tk.BooleanVar(master=self._root, value=False)
        tick_id = None

        def _confirm(_event=None) -> None:
            confirmed.set(True)

        def _tick() -> None:
            nonlocal tick_id
            if confirmed.get():
                return
            self._root.update_idletasks()
            tick_id = self._root.after(10, _tick)

        return_binding = self._root.bind("<Return>", _confirm)
        space_binding = self._root.bind("<space>", _confirm)
        click_binding = self._root.bind("<Button-1>", _confirm)
        try:
            self._root.lift()
            try:
                self._root.focus_force()
            except Exception:
                pass
            _tick()
            self._root.wait_variable(confirmed)
        finally:
            if tick_id is not None:
                try:
                    self._root.after_cancel(tick_id)
                except Exception:
                    pass
            if return_binding is not None:
                self._root.unbind("<Return>", return_binding)
            if space_binding is not None:
                self._root.unbind("<space>", space_binding)
            if click_binding is not None:
                self._root.unbind("<Button-1>", click_binding)

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


class LiveSnapDotsOverlay:
    """Render small target and tracked dots without a full-screen overlay."""

    def __init__(self, *, size: int = 24) -> None:
        target_size = max(12, int(size))
        tracked_size = max(10, int(round(target_size * 0.7)))
        self._target_dot = LiveGazeDotOverlay(size=target_size, fill="#0a84ff", outline="#ffffff")
        self._tracked_dot = LiveGazeDotOverlay(size=tracked_size, fill="#ff3b30", outline="#ffffff")

    def open(self) -> None:
        self._target_dot.open()
        self._tracked_dot.open()

    def show_validation_point(
        self,
        target_x: float,
        target_y: float,
        *,
        tracked_x: float | None = None,
        tracked_y: float | None = None,
        armed: bool = False,
        wink_side: str | None = None,
        step_index: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        del step_index, total_steps
        if armed:
            fill = "#ff9f0a"
            if wink_side == "left":
                fill = "#30d158"
            elif wink_side == "right":
                fill = "#bf5af2"
            self._target_dot.set_style(fill=fill, outline="#ffffff")
        else:
            self._target_dot.set_style(fill="#0a84ff", outline="#ffffff")
        self._target_dot.show_point(target_x, target_y)
        if armed:
            self._tracked_dot.hide()
        elif tracked_x is not None and tracked_y is not None:
            self._tracked_dot.show_point(tracked_x, tracked_y)
        else:
            self._tracked_dot.hide()

    def hide(self) -> None:
        self._target_dot.hide()
        self._tracked_dot.hide()

    def close(self) -> None:
        self._target_dot.close()
        self._tracked_dot.close()

class CalibrationDotOverlay:
    """Render a full-screen calibration overlay on the primary display."""

    def __init__(
        self,
        *,
        size: int = 44,
        fill: str = "#0a84ff",
        outline: str = "#ffffff",
        background: str = "#08111f",
        instruction_color: str = "#f5f7fa",
    ) -> None:
        self.size = max(16, int(size))
        self.fill = fill
        self.outline = outline
        self.background = background
        self.instruction_color = instruction_color
        self._root = None
        self._canvas = None
        self._screen_width = 0
        self._screen_height = 0
        self._visible = False
        self._background_item = None
        self._dot_item = None
        self._tracked_dot_item = None
        self._instruction_item = None
        self._last_instruction: str | None = None

    def open(self) -> None:
        if self._root is not None:
            return
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg=self.background)
        root.update_idletasks()
        self._screen_width = max(self.size, int(root.winfo_screenwidth()))
        self._screen_height = max(self.size, int(root.winfo_screenheight()))
        root.geometry(f"{self._screen_width}x{self._screen_height}+0+0")
        canvas = tk.Canvas(
            root,
            width=self._screen_width,
            height=self._screen_height,
            bg=self.background,
            highlightthickness=0,
            bd=0,
        )
        canvas.pack(fill="both", expand=True)
        self._root = root
        self._canvas = canvas

    def show_point(
        self,
        x_norm: float,
        y_norm: float,
        *,
        step_index: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        instruction = "Look at the dot. Press Enter, Space, or click to capture."
        if step_index is not None and total_steps is not None:
            instruction = f"Calibration {step_index}/{total_steps}. {instruction}"
        self._render_points(
            target_x=x_norm,
            target_y=y_norm,
            tracked_x=None,
            tracked_y=None,
            instruction=instruction,
        )

    def show_validation_point(
        self,
        target_x: float,
        target_y: float,
        *,
        tracked_x: float | None = None,
        tracked_y: float | None = None,
        step_index: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        instruction = "Look at the blue target. Click when the red dot sits on the blue target."
        if step_index is not None and total_steps is not None:
            instruction = f"Validation {step_index}/{total_steps}. {instruction}"
        self._render_points(
            target_x=target_x,
            target_y=target_y,
            tracked_x=tracked_x,
            tracked_y=tracked_y,
            instruction=instruction,
        )

    def _render_points(
        self,
        *,
        target_x: float,
        target_y: float,
        tracked_x: float | None,
        tracked_y: float | None,
        instruction: str,
    ) -> None:
        if self._root is None or self._canvas is None:
            self.open()
        assert self._root is not None
        assert self._canvas is not None

        geometry = GazeDotGeometry.from_normalized(
            x_norm=target_x,
            y_norm=target_y,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
            size=self.size,
        )
        if self._background_item is None:
            self._background_item = self._canvas.create_rectangle(
                0,
                0,
                self._screen_width,
                self._screen_height,
                fill=self.background,
                outline=self.background,
            )
        inset = max(2, self.size // 8)
        target_coords = (
            geometry.left + inset,
            geometry.top + inset,
            geometry.left + geometry.size - inset,
            geometry.top + geometry.size - inset,
        )
        if self._dot_item is None:
            self._dot_item = self._canvas.create_oval(
                *target_coords,
                fill=self.fill,
                outline=self.outline,
                width=max(1, self.size // 16),
            )
        else:
            self._canvas.coords(self._dot_item, *target_coords)
        if tracked_x is not None and tracked_y is not None:
            tracked_size = max(12, self.size // 2)
            tracked_geometry = GazeDotGeometry.from_normalized(
                x_norm=tracked_x,
                y_norm=tracked_y,
                screen_width=self._screen_width,
                screen_height=self._screen_height,
                size=tracked_size,
            )
            tracked_inset = max(2, tracked_size // 8)
            tracked_coords = (
                tracked_geometry.left + tracked_inset,
                tracked_geometry.top + tracked_inset,
                tracked_geometry.left + tracked_geometry.size - tracked_inset,
                tracked_geometry.top + tracked_geometry.size - tracked_inset,
            )
            if self._tracked_dot_item is None:
                self._tracked_dot_item = self._canvas.create_oval(
                    *tracked_coords,
                    fill="#ff453a",
                    outline="#ffffff",
                    width=max(1, tracked_size // 14),
                )
            else:
                self._canvas.coords(self._tracked_dot_item, *tracked_coords)
        else:
            if self._tracked_dot_item is not None:
                self._canvas.delete(self._tracked_dot_item)
                self._tracked_dot_item = None
        if self._instruction_item is None:
            self._instruction_item = self._canvas.create_text(
                self._screen_width // 2,
                max(self.size * 2, self._screen_height - max(64, self.size * 2)),
                text=instruction,
                fill=self.instruction_color,
                font=("Helvetica", max(16, self.size // 2), "bold"),
            )
            self._last_instruction = instruction
        elif instruction != self._last_instruction:
            self._canvas.itemconfigure(self._instruction_item, text=instruction)
            self._last_instruction = instruction
        if not self._visible:
            self._root.deiconify()
            self._visible = True
        self._refresh_focus()

    def wait_for_confirmation(self, *, on_iteration: Callable[[], None] | None = None) -> None:
        if self._root is None or self._canvas is None:
            self.open()
        assert self._root is not None
        assert self._canvas is not None
        import tkinter as tk

        confirmed = tk.BooleanVar(master=self._root, value=False)
        tick_id = None

        def _confirm(_event=None) -> None:
            confirmed.set(True)

        def _tick() -> None:
            nonlocal tick_id
            if confirmed.get():
                return
            if on_iteration is not None:
                on_iteration()
            self._refresh_focus()
            tick_id = self._root.after(10, _tick)

        return_binding = self._root.bind("<Return>", _confirm)
        space_binding = self._root.bind("<space>", _confirm)
        click_binding = self._canvas.bind("<Button-1>", _confirm)
        try:
            try:
                self._root.grab_set()
            except Exception:
                pass
            self._refresh_focus()
            _tick()
            self._root.wait_variable(confirmed)
        finally:
            if tick_id is not None:
                try:
                    self._root.after_cancel(tick_id)
                except Exception:
                    pass
            try:
                self._root.grab_release()
            except Exception:
                pass
            if return_binding is not None:
                self._root.unbind("<Return>", return_binding)
            if space_binding is not None:
                self._root.unbind("<space>", space_binding)
            if click_binding is not None:
                self._canvas.unbind("<Button-1>", click_binding)

    def wait_for_click_point(self, *, on_iteration: Callable[[], None] | None = None) -> tuple[float, float]:
        if self._root is None or self._canvas is None:
            self.open()
        assert self._root is not None
        assert self._canvas is not None
        import tkinter as tk

        last_point: tuple[float, float] | None = None
        click_state = tk.StringVar(master=self._root, value="")
        tick_id = None

        def _capture_click(event) -> None:
            nonlocal last_point
            last_point = (
                clamp_unit_interval(event.x / max(1, self._screen_width)),
                clamp_unit_interval(event.y / max(1, self._screen_height)),
            )
            click_state.set("captured")

        def _tick() -> None:
            nonlocal tick_id
            if click_state.get():
                return
            if on_iteration is not None:
                on_iteration()
            self._refresh_focus()
            tick_id = self._root.after(10, _tick)

        click_binding = self._canvas.bind("<Button-1>", _capture_click)
        try:
            try:
                self._root.grab_set()
            except Exception:
                pass
            self._refresh_focus()
            _tick()
            self._root.wait_variable(click_state)
        finally:
            if tick_id is not None:
                try:
                    self._root.after_cancel(tick_id)
                except Exception:
                    pass
            try:
                self._root.grab_release()
            except Exception:
                pass
            if click_binding is not None:
                self._canvas.unbind("<Button-1>", click_binding)
        assert last_point is not None
        return last_point

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
        root = self._root
        try:
            try:
                root.grab_release()
            except Exception:
                pass
            try:
                root.withdraw()
                root.update_idletasks()
                root.update()
            except Exception:
                pass
            try:
                root.quit()
            except Exception:
                pass
            root.destroy()
        finally:
            self._root = None
            self._canvas = None
            self._visible = False
            self._background_item = None
            self._dot_item = None
            self._tracked_dot_item = None
            self._instruction_item = None
            self._last_instruction = None

    def _refresh_focus(self) -> None:
        assert self._root is not None
        assert self._canvas is not None
        self._root.lift()
        try:
            self._root.focus_force()
        except Exception:
            pass
        try:
            self._canvas.focus_set()
        except Exception:
            pass
        self._root.update_idletasks()
        self._root.update()
