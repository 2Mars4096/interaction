"""Minimal macOS runtime helpers for cursor movement, clicks, and drag."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import util
import math


class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class CGRect(ctypes.Structure):
    _fields_ = [("origin", CGPoint), ("size", CGSize)]


def _load_quartz() -> ctypes.CDLL:
    library = util.find_library("ApplicationServices")
    if not library:
        raise RuntimeError("ApplicationServices framework not found")
    quartz = ctypes.cdll.LoadLibrary(library)
    quartz.CGEventCreateMouseEvent.restype = ctypes.c_void_p
    quartz.CGEventCreateMouseEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, CGPoint, ctypes.c_uint32]
    quartz.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    quartz.CFRelease.argtypes = [ctypes.c_void_p]
    quartz.CGMainDisplayID.restype = ctypes.c_uint32
    quartz.CGDisplayBounds.argtypes = [ctypes.c_uint32]
    quartz.CGDisplayBounds.restype = CGRect
    return quartz


def _post_mouse_event(quartz: ctypes.CDLL, event_type: int, point: CGPoint, button: int) -> None:
    event = quartz.CGEventCreateMouseEvent(None, event_type, point, button)
    if not event:
        raise RuntimeError("Failed to create Quartz mouse event")
    try:
        quartz.CGEventPost(0, event)
    finally:
        quartz.CFRelease(event)


def move(x: float, y: float) -> None:
    quartz = _load_quartz()
    point = CGPoint(x, y)
    _post_mouse_event(quartz, 5, point, 0)


def click(x: float, y: float) -> None:
    quartz = _load_quartz()
    point = CGPoint(x, y)
    _post_mouse_event(quartz, 5, point, 0)
    _post_mouse_event(quartz, 1, point, 0)
    _post_mouse_event(quartz, 2, point, 0)


def double_click(x: float, y: float) -> None:
    click(x, y)
    click(x, y)


def right_click(x: float, y: float) -> None:
    quartz = _load_quartz()
    point = CGPoint(x, y)
    _post_mouse_event(quartz, 5, point, 0)
    _post_mouse_event(quartz, 3, point, 1)
    _post_mouse_event(quartz, 4, point, 1)


def move_normalized(x: float, y: float) -> None:
    quartz = _load_quartz()
    point = _normalized_to_point(quartz, x, y)
    _post_mouse_event(quartz, 5, point, 0)


def click_normalized(x: float, y: float) -> None:
    point = _normalized_to_point(_load_quartz(), x, y)
    click(point.x, point.y)


def double_click_normalized(x: float, y: float) -> None:
    point = _normalized_to_point(_load_quartz(), x, y)
    double_click(point.x, point.y)


def right_click_normalized(x: float, y: float) -> None:
    point = _normalized_to_point(_load_quartz(), x, y)
    right_click(point.x, point.y)


def drag(start_x: float, start_y: float, end_x: float, end_y: float) -> None:
    quartz = _load_quartz()
    start = CGPoint(start_x, start_y)
    end = CGPoint(end_x, end_y)
    _post_mouse_event(quartz, 5, start, 0)
    _post_mouse_event(quartz, 1, start, 0)
    _post_mouse_event(quartz, 6, end, 0)
    _post_mouse_event(quartz, 2, end, 0)


def drag_normalized(start_x: float, start_y: float, end_x: float, end_y: float) -> None:
    quartz = _load_quartz()
    start = _normalized_to_point(quartz, start_x, start_y)
    end = _normalized_to_point(quartz, end_x, end_y)
    drag(start.x, start.y, end.x, end.y)


def _normalized_to_point(quartz: ctypes.CDLL, x: float, y: float) -> CGPoint:
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Pointer coordinates must be finite.")
    display_id = quartz.CGMainDisplayID()
    # Mouse events use global display coordinates, not Retina backing pixels.
    bounds = quartz.CGDisplayBounds(display_id)
    if not all(math.isfinite(value) for value in (bounds.origin.x, bounds.origin.y, bounds.size.width, bounds.size.height)):
        raise RuntimeError("Primary display bounds are unavailable.")
    if bounds.size.width < 1 or bounds.size.height < 1:
        raise RuntimeError("Primary display bounds are empty.")
    return CGPoint(
        bounds.origin.x + _clamp(x) * (bounds.size.width - 1),
        bounds.origin.y + _clamp(y) * (bounds.size.height - 1),
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal macOS cursor action.")
    parser.add_argument(
        "command",
        choices=[
            "move",
            "click",
            "double-click",
            "right-click",
            "drag",
            "move-normalized",
            "click-normalized",
            "double-click-normalized",
            "right-click-normalized",
            "drag-normalized",
        ],
    )
    parser.add_argument("coordinates", nargs="+", type=float)
    args = parser.parse_args()
    required = 4 if "drag" in args.command else 2
    if len(args.coordinates) != required:
        parser.error(f"{args.command} requires {required} numeric coordinates.")

    if args.command == "move":
        move(args.coordinates[0], args.coordinates[1])
    elif args.command == "click":
        click(args.coordinates[0], args.coordinates[1])
    elif args.command == "double-click":
        double_click(args.coordinates[0], args.coordinates[1])
    elif args.command == "right-click":
        right_click(args.coordinates[0], args.coordinates[1])
    elif args.command == "drag":
        drag(*args.coordinates)
    elif args.command == "move-normalized":
        move_normalized(args.coordinates[0], args.coordinates[1])
    elif args.command == "click-normalized":
        click_normalized(args.coordinates[0], args.coordinates[1])
    elif args.command == "double-click-normalized":
        double_click_normalized(args.coordinates[0], args.coordinates[1])
    elif args.command == "right-click-normalized":
        right_click_normalized(args.coordinates[0], args.coordinates[1])
    else:
        drag_normalized(*args.coordinates)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
