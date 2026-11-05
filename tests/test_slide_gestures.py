"""--slides on synthetic hands: quick strokes change slides and zoom; everything a presenter
does in between (returning the hand, raising or lowering it, moving a fist) does nothing."""

from __future__ import annotations

from typing import Any

import pytest

from config import AppConfig
from src.gesture_engine import Mode
from src.os_controller import Action, KeyTap, Zoom
from src.slide_gestures import Phase, SlideGestures
from tests.handgen import FIST, OPEN, Driver

LEFT, MID, RIGHT = (0.35, 0.55), (0.5, 0.55), (0.65, 0.55)
UP, DOWN = (0.5, 0.3), (0.5, 0.8)


def events(actions: list[Action]) -> list[str]:
    out = []
    for a in actions:
        if isinstance(a, KeyTap):
            out += [a.key] * a.count
        elif isinstance(a, Zoom):
            out.append("zoom in" if a.steps > 0 else "zoom out")
    return out


def driver(fps: float = 60.0, **overrides: Any) -> Driver:
    cfg = AppConfig()
    for name, value in overrides.items():
        setattr(cfg.slides, name, value)
    cfg.validate()
    d = Driver(SlideGestures(cfg.slides, cfg.gesture), fps=fps)
    d.hold(OPEN, 0.4, center=MID)  # the hand rests: ready
    assert d.engine.phase is Phase.RESTING
    return d


def swipe(
    d: Driver, start: tuple[float, float], end: tuple[float, float], pose: Any = OPEN, **view: Any
) -> list[str]:
    """A quick stroke (0.2 s) followed by the hand resting at the end."""
    actions = d.sweep(pose, start, end, 0.2, **view)
    return events(actions + d.hold(pose, 0.4, center=end, **view))


def test_quick_strokes_change_slides_and_zoom() -> None:
    d = driver()
    assert swipe(d, MID, RIGHT) == ["right"]
    d.hold(OPEN, 1.0, center=RIGHT)
    assert swipe(d, RIGHT, MID) == ["left"]
    d.hold(OPEN, 1.0, center=MID)
    assert swipe(d, MID, UP) == ["zoom in"]
    d.hold(OPEN, 1.0, center=UP)
    assert swipe(d, UP, MID) == ["zoom out"]
    assert d.outputs[-1].mode is Mode.SLIDES


def test_zoom_sends_ctrl_wheel_steps() -> None:
    d = driver(zoom_steps=3)
    actions = d.sweep(OPEN, MID, UP, 0.2) + d.hold(OPEN, 0.4, center=UP)
    assert [a for a in actions if isinstance(a, Zoom)] == [Zoom(3)]


def test_the_way_back_after_a_swipe_is_ignored() -> None:
    d = driver()
    pressed = []
    for _ in range(3):  # next, next, next: swipe right, bring the hand back, ...
        pressed += swipe(d, MID, RIGHT)
        pressed += swipe(d, RIGHT, MID)
    assert pressed == ["right", "right", "right"]


def test_slow_moves_diagonals_and_fists_do_nothing() -> None:
    d = driver()
    slow = d.sweep(OPEN, MID, RIGHT, 1.5) + d.hold(OPEN, 0.5, center=RIGHT)
    assert events(slow) == []
    d.hold(OPEN, 1.0, center=RIGHT)
    assert swipe(d, RIGHT, (0.5, 0.3)) == []  # 45 degrees: neither left nor up
    d.hold(FIST, 1.0, center=(0.5, 0.3))
    assert swipe(d, (0.5, 0.3), (0.65, 0.3), pose=FIST) == []  # a fist repositions the hand


def test_raising_the_hand_into_view_and_lowering_it_out_do_nothing() -> None:
    cfg = AppConfig()
    d = Driver(SlideGestures(cfg.slides, cfg.gesture))
    # The hand appears while still moving up, then stops: no rest before the stroke.
    assert events(d.sweep(OPEN, (0.5, 0.85), MID, 0.2) + d.hold(OPEN, 0.5, center=MID)) == []
    # Lowered quickly out of the frame: the stroke never ends with the hand resting in view.
    actions = d.sweep(OPEN, MID, (0.5, 0.9), 0.15) + d.hold(None, 0.5)
    assert events(actions) == []
    assert d.outputs[-1].mode is Mode.NO_HAND


def test_a_short_tracking_dropout_mid_swipe_is_tolerated() -> None:
    d = driver()
    actions = d.sweep(OPEN, MID, (0.58, 0.55), 0.1)
    actions += d.hold(None, 0.05)  # motion blur: two frames without a hand
    actions += d.sweep(OPEN, (0.6, 0.55), RIGHT, 0.08) + d.hold(OPEN, 0.4, center=RIGHT)
    assert events(actions) == ["right"]


@pytest.mark.parametrize("palm_px", [80.0, 250.0])
def test_hand_distance_does_not_matter(palm_px: float) -> None:
    cfg = AppConfig()
    d = Driver(SlideGestures(cfg.slides, cfg.gesture))
    d.hold(OPEN, 0.4, center=MID, palm_px=palm_px)
    shift = 1.4 * palm_px / 1280  # the same stroke in palm lengths, whatever the hand's size
    assert swipe(d, MID, (0.5 + shift, 0.55), palm_px=palm_px) == ["right"]


def test_works_at_a_webcam_30_fps() -> None:
    d = driver(fps=30.0)
    assert swipe(d, MID, RIGHT) == ["right"]
    d.hold(OPEN, 1.0, center=RIGHT)
    assert swipe(d, RIGHT, MID) == ["left"]
