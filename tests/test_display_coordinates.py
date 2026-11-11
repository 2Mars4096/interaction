from types import SimpleNamespace

import pytest

from interaction.platform.macos_runtime import CGPoint, CGRect, CGSize, _normalized_to_point
from interaction.ui.calibration import target_center


def display(width=1512, height=982):
    return SimpleNamespace(
        CGMainDisplayID=lambda: 1,
        CGDisplayBounds=lambda display_id: CGRect(CGPoint(0, 0), CGSize(width, height)),
        CGDisplayPixelsWide=lambda display_id: 3024,
        CGDisplayPixelsHigh=lambda display_id: 1964,
    )


@pytest.mark.parametrize("x,y", [(0, 0), (.2, .8), (.5, .5), (1, 1)])
def test_pointer_and_calibration_use_same_primary_display_coordinates(x, y):
    point = _normalized_to_point(display(), x, y)
    target = target_center(x, y, screen_width=1512, screen_height=982)
    assert (point.x, point.y) == target


def test_pointer_uses_display_bounds_instead_of_retina_backing_pixels():
    point = _normalized_to_point(display(), 1, 1)
    assert (point.x, point.y) == (1511, 981)


@pytest.mark.parametrize("x,y", [(float("nan"), 0), (0, float("inf"))])
def test_pointer_rejects_nonfinite_coordinates(x, y):
    with pytest.raises(ValueError):
        _normalized_to_point(display(), x, y)
