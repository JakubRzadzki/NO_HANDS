"""Scroll with a stop sign (src/stop_hand.py): the other hand raised, palm to the camera, fingers
up; this hand's up / down movement scrolls."""

from __future__ import annotations

from typing import Any

from src.gesture_engine import LeftState, Mode
from src.os_controller import Action, MoveTo, Press, Scroll
from tests.handgen import CENTER, OPEN, RELAXED, STOP, Driver, make_engine, pinch


def settled(**overrides: Any) -> Driver:
    d = Driver(make_engine(**overrides), fps=30.0)
    d.hold(OPEN, 0.5)
    return d


def other(pose: dict[str, Any], **view: Any) -> dict[str, Any]:
    return {"center": (0.2, 0.55), **view, **pose}


def of(actions: list[Action], kind: type) -> list[Any]:
    return [a for a in actions if isinstance(a, kind)]


def test_a_stop_sign_and_this_hand_up_or_down_scrolls() -> None:
    d = settled()
    d.hold(OPEN, 0.3, second=other(STOP))
    assert d.engine.mode is Mode.SCROLL and d.outputs[-1].cursor_frozen
    up = d.hold(OPEN, 1.0, second=other(STOP), center=(CENTER[0], CENTER[1] - 0.12))
    assert sum(s.dy for s in of(up, Scroll)) > 3  # up = scroll up
    assert of(up, MoveTo) == []  # the cursor holds still
    down = d.hold(OPEN, 1.0, second=other(STOP), center=(CENTER[0], CENTER[1] + 0.12))
    assert sum(s.dy for s in of(down, Scroll)) < -3


def test_lowering_the_stop_hand_brings_the_cursor_back_without_a_click() -> None:
    d = settled()
    d.hold(OPEN, 0.3, second=other(STOP))
    during = d.hold(pinch("index"), 0.3, second=other(STOP))  # a pinch while scrolling
    after = d.hold(pinch("index"), 0.3) + d.hold(OPEN, 0.3)  # the stop hand goes down
    assert of(during + after, Press) == []
    assert d.engine.mode is Mode.POINTER
    moves = d.sweep(OPEN, CENTER, (0.6, 0.55), 0.3)
    assert of(moves, MoveTo)


def test_a_hand_merely_in_view_is_not_a_stop_sign() -> None:
    d = settled()
    for pose, view in (
        (RELAXED, {}),  # fingers loose
        (STOP, {"roll_deg": 90.0}),  # fingers pointing sideways
        (STOP, {"yaw_deg": 80.0}),  # edge to the camera
        (pinch("index"), {}),  # pinching: that is the two-hand zoom
    ):
        actions = d.hold(OPEN, 0.5, second=other(pose, **view))
        assert d.engine.mode is not Mode.SCROLL, (pose, view)
        assert of(actions, Scroll) == []


def test_a_stop_sign_for_a_moment_does_nothing() -> None:
    d = settled()
    d.hold(OPEN, 3 / 30, second=other(STOP))
    d.hold(OPEN, 0.3)
    assert all(out.mode is Mode.POINTER for out in d.outputs[-9:])


def test_a_stop_sign_does_not_end_a_drag() -> None:
    d = settled()
    d.hold(pinch("index"), 0.2)
    d.sweep(pinch("index"), CENTER, (0.65, 0.55), 0.3)
    assert d.engine.left_state is LeftState.DRAG
    d.hold(pinch("index"), 0.4, second=other(STOP), center=(0.65, 0.55))
    assert d.engine.left_state is LeftState.DRAG and d.engine.mode is not Mode.SCROLL


def test_the_preview_says_what_the_stop_sign_is_missing() -> None:
    d = settled()
    d.hold(OPEN, 0.1, second=other(RELAXED))
    assert d.outputs[-1].stop_hint.startswith("straighten")
    d.hold(OPEN, 0.3, second=other(STOP))
    assert d.outputs[-1].stop_hint == "STOP"


def test_stop_scroll_can_be_switched_off() -> None:
    d = settled(scroll_with_other_hand=False)
    d.hold(OPEN, 0.5, second=other(STOP))
    assert d.engine.mode is Mode.POINTER
