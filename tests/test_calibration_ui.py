from __future__ import annotations

import threading

import pytest

from interaction.ui.calibration import CalibrationCancelled, CalibrationTargetOverlay, CalibrationUIError, target_center


class FakeRoot:
    def __init__(self) -> None:
        self.width = 1920
        self.height = 1080
        self.callbacks = {}
        self.options = {}
        self.updates = 0
        self.destroy_calls = 0
        self.update_callback = None
        self.fail_at = None

    def withdraw(self):
        self.options["withdraw"] = True

    def overrideredirect(self, value):
        self.options["borderless"] = value

    def attributes(self, key, value):
        self.options[key] = value

    def configure(self, **kwargs):
        self.options.update(kwargs)

    def winfo_screenwidth(self):
        return self.width

    def winfo_screenheight(self):
        return self.height

    def geometry(self, value):
        self.options["geometry"] = value
        if self.fail_at == "geometry":
            raise RuntimeError("display unavailable")

    def bind(self, event, callback):
        self.callbacks[event] = callback

    def protocol(self, event, callback):
        self.callbacks[event] = callback

    def deiconify(self):
        self.options["visible"] = True

    def lift(self):
        pass

    def focus_force(self):
        self.options["focused"] = True

    def update_idletasks(self):
        pass

    def update(self):
        self.updates += 1
        if self.update_callback:
            self.update_callback()
        if self.fail_at == "update":
            raise RuntimeError("window unavailable")

    def destroy(self):
        self.destroy_calls += 1
        if self.fail_at == "destroy":
            raise RuntimeError("already destroyed")


class FakeCanvas:
    def __init__(self, root, **kwargs):
        self.root = root
        self.options = kwargs
        self.text = {}
        self.ovals = []

    def pack(self, **kwargs):
        self.options.update(kwargs)

    def create_text(self, x, y, **kwargs):
        self.text[kwargs["tags"]] = kwargs["text"]

    def itemconfigure(self, tag, **kwargs):
        self.text[tag] = kwargs["text"]

    def delete(self, tag):
        self.ovals = [oval for oval in self.ovals if oval[1]["tags"] != tag]

    def create_oval(self, *bounds, **kwargs):
        self.ovals.append((bounds, kwargs))


class FakeTk:
    def __init__(self) -> None:
        self.root = FakeRoot()
        self.canvas = None
        self.created = 0

    def Tk(self):
        self.created += 1
        return self.root

    def Canvas(self, root, **kwargs):
        self.canvas = FakeCanvas(root, **kwargs)
        return self.canvas


def test_target_center_matches_primary_screen_endpoints_and_center():
    assert target_center(0.0, 0.0, screen_width=1920, screen_height=1080) == (0.0, 0.0)
    assert target_center(1.0, 1.0, screen_width=1920, screen_height=1080) == (1919.0, 1079.0)
    assert target_center(0.5, 0.5, screen_width=1920, screen_height=1080) == (959.5, 539.5)


@pytest.mark.parametrize("x,y", [(float("nan"), 0.5), (0.5, float("inf")), (-0.1, 0.5), (0.5, 1.1), (None, 0.5)])
def test_invalid_targets_are_rejected_without_clamping(x, y):
    with pytest.raises(CalibrationUIError) as failure:
        target_center(x, y, screen_width=1920, screen_height=1080)
    assert failure.value.code == "invalid_calibration_target"


@pytest.mark.parametrize("width,height", [(0, 1080), (1920, -1), (True, 1080), (1920.5, 1080)])
def test_invalid_screen_size_is_rejected(width, height):
    with pytest.raises(CalibrationUIError):
        target_center(0.5, 0.5, screen_width=width, screen_height=height)


def test_overlay_is_visible_before_target_collection_and_reuses_window():
    tk = FakeTk()
    overlay = CalibrationTargetOverlay(tk_module=tk)
    overlay.open()
    overlay.open()
    assert overlay.screen_size == (1920, 1080)
    assert tk.created == 1
    assert tk.root.options["borderless"] is True
    assert tk.root.options["-topmost"] is True
    assert tk.root.options["geometry"] == "1920x1080+0+0"
    assert tk.root.options["visible"] is True
    assert tk.root.options["focused"] is True
    updates = tk.root.updates
    overlay.show_target(0.25, 0.75, "Target 2 of 5")
    assert tk.root.updates > updates
    assert tk.canvas.text["progress"] == "Target 2 of 5"
    for bounds, _ in tk.canvas.ovals:
        assert (bounds[0] + bounds[2]) / 2 == 0.25 * 1919
        assert (bounds[1] + bounds[3]) / 2 == 0.75 * 1079
    overlay.show_target(0.7, 0.3, "Validation 1 of 4", phase="validation")
    assert len(tk.canvas.ovals) == 2
    assert "Checking accuracy" in tk.canvas.text["heading"]
    assert tk.canvas.text["progress"] == "Validation 1 of 4"
    overlay.close()
    overlay.close()
    assert tk.root.destroy_calls == 1
    assert overlay.screen_size is None


@pytest.mark.parametrize("event", ["<Escape>", "WM_DELETE_WINDOW"])
def test_window_events_raise_cancellation_from_pump_not_callback(event):
    tk = FakeTk()
    overlay = CalibrationTargetOverlay(tk_module=tk)
    overlay.open()
    tk.root.callbacks[event]()
    with pytest.raises(CalibrationCancelled) as failure:
        overlay.pump()
    assert failure.value.code == "calibration_cancelled"
    overlay.close()
    assert tk.root.destroy_calls == 1


def test_cancellation_is_seen_during_settling_wait():
    tk = FakeTk()
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
        if now[0] >= 0.03:
            tk.root.callbacks["<Escape>"]()

    overlay = CalibrationTargetOverlay(tk_module=tk, clock=lambda: now[0], sleep=sleep)
    overlay.open()
    with pytest.raises(CalibrationCancelled):
        overlay.wait(1000)
    assert 0.03 <= now[0] < 0.05
    assert max(sleeps) <= 0.016
    overlay.close()


def test_wait_pumps_events_until_requested_interval_is_complete():
    tk = FakeTk()
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    overlay = CalibrationTargetOverlay(tk_module=tk, clock=lambda: now[0], sleep=sleep)
    overlay.open()
    before = tk.root.updates
    overlay.wait(45)
    assert now[0] == pytest.approx(0.045)
    assert max(sleeps) <= 0.016
    assert tk.root.updates - before >= 4
    overlay.wait(0)
    overlay.close()


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), None])
def test_invalid_wait_has_structured_failure(duration):
    overlay = CalibrationTargetOverlay(tk_module=FakeTk())
    with pytest.raises(CalibrationUIError) as failure:
        overlay.wait(duration)
    assert failure.value.code == "invalid_calibration_wait"


def test_missing_tk_is_reported_without_invisible_fallback(monkeypatch):
    def fail_import(_name):
        raise ModuleNotFoundError("No module named tkinter")

    monkeypatch.setattr("interaction.ui.calibration.importlib.import_module", fail_import)
    overlay = CalibrationTargetOverlay()
    with pytest.raises(CalibrationUIError) as failure:
        overlay.open()
    assert failure.value.code == "calibration_ui_unavailable"
    assert failure.value.payload == {"reason": "ModuleNotFoundError"}
    overlay.close()


@pytest.mark.parametrize("failure_point", ["geometry", "update"])
def test_partial_open_failure_destroys_window(failure_point):
    tk = FakeTk()
    tk.root.fail_at = failure_point
    overlay = CalibrationTargetOverlay(tk_module=tk)
    with pytest.raises(CalibrationUIError):
        overlay.open()
    assert tk.root.destroy_calls == 1
    assert overlay.screen_size is None
    overlay.close()
    assert tk.root.destroy_calls == 1


def test_render_failure_is_structured_and_cleanup_is_idempotent():
    tk = FakeTk()
    overlay = CalibrationTargetOverlay(tk_module=tk)
    overlay.open()
    tk.root.fail_at = "update"
    with pytest.raises(CalibrationUIError) as failure:
        overlay.show_target(0.5, 0.5, "Center")
    assert failure.value.code == "calibration_ui_failed"
    tk.root.fail_at = "destroy"
    overlay.close()
    overlay.close()
    assert tk.root.destroy_calls == 1


def test_cancel_during_event_processing_is_not_misreported_as_display_error():
    tk = FakeTk()
    overlay = CalibrationTargetOverlay(tk_module=tk)
    overlay.open()
    tk.root.update_callback = tk.root.callbacks["WM_DELETE_WINDOW"]
    tk.root.fail_at = "update"
    with pytest.raises(CalibrationCancelled):
        overlay.pump()
    overlay.close()


def test_closed_pump_reports_structured_error():
    with pytest.raises(CalibrationUIError) as failure:
        CalibrationTargetOverlay(tk_module=FakeTk()).pump()
    assert failure.value.code == "calibration_ui_closed"


def test_background_thread_is_rejected_before_creating_tk_window():
    tk = FakeTk()
    overlay = CalibrationTargetOverlay(tk_module=tk)
    errors = []

    def open_in_worker():
        try:
            overlay.open()
        except CalibrationUIError as error:
            errors.append(error)

    worker = threading.Thread(target=open_in_worker)
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert errors[0].code == "calibration_ui_thread"
    assert tk.created == 0
