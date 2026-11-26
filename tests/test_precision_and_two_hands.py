"""Precision (fold fingers to slow the cursor), the relaxed horns and the zoom with
two hands, on synthetic hands."""

from __future__ import annotations

import itertools
from typing import Any

from config import AppConfig
from src.gesture_engine import LeftState, Mode
from src.os_controller import Action, KeyTap, MoveTo, Press, Release, Zoom
from src.slide_gestures import SlideGestures
from tests.handgen import CENTER, OPEN, RELAXED, VICTORY, Driver, make_engine, pinch

PINKY_FOLDED: dict[str, Any] = {"curl": {"pinky": 1.0}}


def of(actions: list[Action], kind: type) -> list[Any]:
    return [a for a in actions if isinstance(a, kind)]


def settled(**overrides: Any) -> Driver:
    d = Driver(make_engine(**overrides), fps=60.0)
    d.hold(OPEN, 0.5)
    return d


def cursor_travel(d: Driver, pose: dict[str, Any], dx: float) -> float:
    """Screen px the cursor moves while the hand moves dx (frame widths) to the right."""
    d.hold(pose, 0.3)  # the new speed is confirmed
    start = d.engine.mapper.target[0]
    d.sweep(pose, CENTER, (CENTER[0] + dx, CENTER[1]), 0.4)
    d.hold(pose, 0.5, center=(CENTER[0] + dx, CENTER[1]))  # the filter catches up
    end = d.engine.mapper.target[0]
    d.sweep(pose, (CENTER[0] + dx, CENTER[1]), CENTER, 0.4)
    d.hold(pose, 0.5)
    return float(end - start)


# -- precision -----------------------------------------------------------------------------------
def test_folding_fingers_slows_the_cursor_down() -> None:
    d = settled()
    full = cursor_travel(d, OPEN, 0.08)
    assert d.outputs[-1].speed == 1.0
    half = cursor_travel(d, PINKY_FOLDED, 0.08)
    assert d.outputs[-1].speed == 0.5
    quarter = cursor_travel(d, VICTORY, 0.08)
    assert d.outputs[-1].speed == 0.25
    assert abs(half / full - 0.5) < 0.05 and abs(quarter / full - 0.25) < 0.05


def test_back_to_full_speed_the_cursor_does_not_jump() -> None:
    """After slow movement the cursor is off the spot the hand points at. That offset fades
    while the hand moves; it never jumps the cursor."""
    d = settled()
    d.hold(VICTORY, 0.3)
    d.sweep(VICTORY, CENTER, (0.6, 0.55), 0.5)
    actions = d.hold(OPEN, 0.5, center=(0.6, 0.55)) + d.sweep(OPEN, (0.6, 0.55), CENTER, 0.6)
    moves = [(m.x, m.y) for m in of(actions, MoveTo)]
    steps = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in itertools.pairwise(moves)]
    assert max(steps) < 60  # px per frame at 60 FPS: smooth, no jump
    d.sweep(OPEN, CENTER, (0.6, 0.55), 0.6)
    d.sweep(OPEN, (0.6, 0.55), CENTER, 0.6)
    d.hold(OPEN, 0.5)
    assert abs(float(d.engine.mapper._offset[0])) < 1.0  # caught up with the hand again


def test_a_v_slows_the_cursor_to_a_quarter() -> None:
    d = settled()
    d.hold(VICTORY, 0.3)
    assert d.outputs[-1].speed == 0.25


def test_clicking_in_precision_mode_keeps_the_speed() -> None:
    """The index makes the pinch and is not counted: a click never changes the speed."""
    d = settled()
    d.hold(VICTORY, 0.3)
    actions = d.hold({**VICTORY, **pinch("index")}, 0.15) + d.hold(VICTORY, 0.3)
    assert of(actions, Press) == [Press("left")] and of(actions, Release) == [Release("left")]
    assert all(out.speed == 0.25 for out in d.outputs[-30:])


# -- zoom ----------------------------------------------------------------------------------------
def test_the_horns_with_half_folded_fingers_still_zoom() -> None:
    """Few people fold middle and ring all the way while the pinky is out."""
    half = {"curl": {"middle": 0.47, "ring": 0.47}}  # ratio ~1.09: not straight, not folded
    d = settled()
    d.hold(half, 0.3)
    assert d.engine.mode is Mode.ZOOM
    d = settled(horns_relaxed=False)
    d.hold(half, 0.3)
    assert d.engine.mode is Mode.POINTER


def left_hand(x: float, pose: dict[str, Any] | None = None) -> dict[str, Any]:
    """The other hand, relaxed unless a pose is given (an open hand raised is a stop sign)."""
    return {"center": (x, 0.55), **(pose or RELAXED)}


def test_two_hands_pinched_and_pulled_apart_zoom_in_and_together_zoom_out() -> None:
    d = settled()
    d.hold(OPEN, 0.2, second=left_hand(0.3))
    assert d.engine.mode is Mode.POINTER  # a second hand that does not pinch changes nothing
    grab = pinch("index")
    d.hold(OPEN, 0.1, second=left_hand(0.3, grab))
    assert d.engine.mode is Mode.TWO_HANDS and d.outputs[-1].cursor_frozen
    actions = d.hold(grab, 0.2, second=left_hand(0.3, grab))
    assert of(actions, Press) == []  # both hands pinched: a zoom, never a click
    apart: list[Action] = []
    for i in range(1, 31):  # the left hand moves away: 0.2 -> 0.3 frame widths apart (+50%)
        apart += d.step(grab, left_hand(0.3 - 0.1 * i / 30, grab)).actions
    assert of(apart, Zoom) == [Zoom(1)] * 3  # one step per 12%: 1.12^3 < 1.5 < 1.12^4
    together: list[Action] = []
    for i in range(1, 31):
        together += d.step(grab, left_hand(0.2 + 0.1 * i / 30, grab)).actions
    assert of(together, Zoom) == [Zoom(-1)] * 3
    assert of(apart + together, Press) == []


def test_both_hands_pinching_at_once_never_click() -> None:
    """Both hands come up and pinch at about the same time; the cursor hand must not win the
    race and click before the other hand's pinch is seen."""
    d = settled()
    grab = pinch("index")
    actions = d.hold(grab, 0.4, second=left_hand(0.3, grab)) + d.hold(OPEN, 0.3)
    assert of(actions, Press) == [] and of(actions, Release) == []


def test_after_two_hands_the_still_pinched_hand_does_not_click() -> None:
    d = settled()
    grab = pinch("index")
    d.hold(grab, 0.3, second=left_hand(0.3, grab))
    actions = d.hold(grab, 0.3) + d.hold(OPEN, 0.3)  # the left hand goes, the right still pinched
    assert d.engine.mode is Mode.POINTER
    assert of(actions, Press) == [] and of(actions, Release) == []
    assert of(d.hold(grab, 0.15) + d.hold(OPEN, 0.3), Press) == [Press("left")]  # clicks again


def test_a_resting_second_hand_changes_nothing() -> None:
    """The presenter's other hand on the forehead, on the desk, holding a remote: the cursor
    moves, clicks work. (A first version stopped everything as soon as two hands were seen.)"""
    d = settled()
    other = left_hand(0.3)
    d.hold(OPEN, 0.5, second=other)
    moves = d.sweep(OPEN, CENTER, (0.55, 0.55), 0.3, second=other)
    assert of(moves, MoveTo) and all(out.mode is Mode.POINTER for out in d.outputs[-20:])
    grab = pinch("index")
    clicked = d.hold(grab, 0.2, center=(0.55, 0.55), second=other)
    clicked += d.hold(OPEN, 0.3, center=(0.55, 0.55), second=other)
    assert of(clicked, Press) == [Press("left")] and of(clicked, Release) == [Release("left")]


def test_a_second_hand_does_not_end_a_drag() -> None:
    d = settled()
    grab = pinch("index")
    d.hold(grab, 0.2)
    d.sweep(grab, CENTER, (0.65, 0.55), 0.3)
    assert d.engine.left_state is LeftState.DRAG
    actions = d.hold(grab, 0.3, second=left_hand(0.3, grab), center=(0.65, 0.55))
    assert of(actions, Release) == [] and d.engine.left_state is LeftState.DRAG


def test_two_hand_zoom_can_be_switched_off() -> None:
    d = settled(two_hand_zoom=False)
    grab = pinch("index")
    d.hold(OPEN, 0.3, second=left_hand(0.3, grab))
    assert d.engine.mode is Mode.POINTER


def slides_driver() -> Driver:
    cfg = AppConfig()
    d = Driver(SlideGestures(cfg.slides, cfg.gesture), fps=60.0)
    d.hold(OPEN, 0.6)
    return d


def test_two_hand_zoom_in_slides_mode_is_not_a_next_slide() -> None:
    d = slides_driver()
    grab = pinch("index")
    d.hold(grab, 0.2, second=left_hand(0.3, grab))
    apart: list[Action] = []
    for i in range(1, 31):
        apart += d.step(grab, left_hand(0.3 - 0.1 * i / 30, grab)).actions
    assert of(apart, Zoom) == [Zoom(1)] * 3
    after = d.hold(grab, 0.3) + d.hold(OPEN, 0.5)  # the left hand goes; the right lets go
    assert after == []  # letting go of that pinch is not "next slide"


def test_a_resting_second_hand_does_not_block_swipes() -> None:
    """Found on the test video: a presenter with one hand on her forehead."""
    d = slides_driver()
    other = left_hand(0.2)
    d.hold(OPEN, 0.5, second=other)
    swipe = d.sweep(OPEN, CENTER, (0.7, 0.55), 0.3, second=other)
    swipe += d.hold(OPEN, 0.5, center=(0.7, 0.55), second=other)
    assert [a.key for a in swipe if hasattr(a, "key")] == ["right"]


# -- slides with the other hand ------------------------------------------------------------------
def other_swipe(d: Driver, x0: float, x1: float, seconds: float, pose: Any = None) -> list[Action]:
    """The other hand moves from x0 to x1 (frame widths) the way people move, speeding up and
    slowing down (minimum jerk); the cursor hand holds still."""
    n = max(1, round(seconds / d.dt))
    actions: list[Action] = []
    for i in range(1, n + 1):
        u = i / n
        k = 10 * u**3 - 15 * u**4 + 6 * u**5
        actions += d.step(OPEN, left_hand(x0 + (x1 - x0) * k, pose)).actions
    return actions


def keys(actions: list[Action]) -> list[str]:
    return [a.key for a in actions if isinstance(a, KeyTap)]


def test_the_other_hand_swipes_the_slides_while_this_hand_points() -> None:
    """The cursor hand only has the strict flick (it also points). The other hand does not
    point, so its swipes use the lenient --slides thresholds."""
    d = settled(other_hand_slides=True)
    d.hold(OPEN, 0.5, second=left_hand(0.15))  # the other hand comes up and rests
    actions = other_swipe(d, 0.15, 0.35, 0.3) + d.hold(OPEN, 0.5, second=left_hand(0.35))
    assert keys(actions) == ["right"]
    actions = d.hold(OPEN, 0.6, second=left_hand(0.35))
    actions += other_swipe(d, 0.35, 0.15, 0.3) + d.hold(OPEN, 0.5, second=left_hand(0.15))
    assert keys(actions) == ["left"]
    assert of(actions, Press) == []  # the cursor hand did nothing


def test_a_careful_swipe_of_the_other_hand_counts() -> None:
    d = settled(other_hand_slides=True)
    d.hold(OPEN, 0.5, second=left_hand(0.12))
    actions = other_swipe(d, 0.12, 0.36, 0.9) + d.hold(OPEN, 0.5, second=left_hand(0.36))
    assert keys(actions) == ["right"]  # 2 palm lengths taking 0.9 s


def test_the_other_hand_coming_into_view_does_not_swipe() -> None:
    d = settled(other_hand_slides=True)
    raising = other_swipe(d, 0.05, 0.3, 0.3)  # appears already moving
    assert keys(raising + d.hold(OPEN, 0.5, second=left_hand(0.3))) == []


def test_other_hand_slides_can_be_switched_off() -> None:
    d = settled(other_hand_slides=False)
    d.hold(OPEN, 0.5, second=left_hand(0.15))
    assert keys(other_swipe(d, 0.15, 0.35, 0.3) + d.hold(OPEN, 0.5)) == []
