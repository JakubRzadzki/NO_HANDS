"""State machine tests on synthetic hands. Several are regression tests for bugs of the
original single-file script (named in the test docstrings)."""

from __future__ import annotations

from typing import Any

import pytest

from src.gesture_engine import HysteresisLatch, LeftState, Mode
from src.os_controller import Action, Click, MoveBy, MoveTo, Press, Release, Scroll, Zoom
from tests.handgen import (
    CENTER,
    FIST,
    OPEN,
    SCROLL,
    VICTORY,
    Driver,
    hand_points,
    make_engine,
    pinch,
    rock,
)


def of(actions: list[Action], kind: type) -> list[Any]:
    return [a for a in actions if isinstance(a, kind)]


def settled(mode: str = "absolute", fps: float = 60.0, **overrides: Any) -> Driver:
    d = Driver(make_engine(mode, **overrides), fps=fps)
    d.hold(OPEN, 0.5)  # hand acquired, filter converged
    return d


# -- primitives -----------------------------------------------------------------------------
def test_hysteresis_latch_needs_both_thresholds_and_confirmation() -> None:
    latch = HysteresisLatch(engage_at=0.22, release_at=0.32, confirm_frames=2)
    assert latch.update(0.10) is False  # one frame is not enough
    assert latch.update(0.10) is True
    assert latch.update(0.28) is True  # inside the band: stays engaged
    assert latch.update(0.40) is True  # release needs confirmation too
    assert latch.update(0.40) is False
    with pytest.raises(ValueError):
        HysteresisLatch(engage_at=0.3, release_at=0.2)


# -- pointer ---------------------------------------------------------------------------------
def test_open_hand_moves_cursor_in_pointer_mode() -> None:
    d = settled()
    actions = d.sweep(OPEN, CENTER, (0.65, 0.55), 0.5) + d.hold(OPEN, 0.5, center=(0.65, 0.55))
    moves = of(actions, MoveTo)
    assert d.outputs[-1].mode is Mode.POINTER
    assert moves and moves[-1].x > 1300 and abs(moves[-1].y - 540) < 30


def test_quick_pinch_is_one_click_at_the_frozen_position() -> None:
    d = settled()
    # The hand drifts a little while pinching (less than drag_start_px on screen).
    during = d.sweep(pinch("index"), CENTER, (0.505, 0.55), 0.1)
    after = d.hold(OPEN, 0.3, center=(0.505, 0.55))
    assert of(during, MoveTo) == []  # cursor frozen: the click lands where the pinch began
    assert of(during + after, Click) == [Click("left")]
    assert of(during + after, Press) == []
    first_move = next(i for i, a in enumerate(after) if isinstance(a, MoveTo))
    assert after.index(Click("left")) < first_move


def test_long_pinch_drags_and_cursor_follows_during_drag() -> None:
    """Regression: the original froze the cursor whenever thumb-middle < 0.34, which is
    always true during a thumb-index pinch, so drag-and-drop could never move anything."""
    d = settled()
    hold = d.hold(pinch("index"), 0.4)
    assert of(hold, Press) == [Press("left")] and of(hold, Click) == []
    assert d.engine.left_state is LeftState.DRAG
    drag = d.sweep(pinch("index"), CENTER, (0.65, 0.55), 0.4)
    assert len(of(drag, MoveTo)) > 5
    drop = d.hold(OPEN, 0.2, center=(0.65, 0.55))
    assert of(drop, Release) == [Release("left")] and of(drop, Click) == []


def test_fast_motion_while_pinched_starts_drag_before_hold_time() -> None:
    d = settled()
    actions = d.sweep(pinch("index"), CENTER, (0.7, 0.55), 0.15)
    assert of(actions, Press) == [Press("left")]  # 0.15 s < click_max_hold
    press_at = actions.index(Press("left"))
    assert any(isinstance(a, MoveTo) for a in actions[press_at:])


def test_hysteresis_prevents_click_chatter() -> None:
    d = settled()
    actions = d.hold(pinch("index", 0.10), 0.05)
    for _ in range(3):  # distance wobbles inside the hysteresis band
        actions += d.hold(pinch("index", 0.30), 1 / 60) + d.hold(pinch("index", 0.18), 1 / 60)
    actions += d.hold(OPEN, 0.2)
    assert of(actions, Click) == [Click("left")]


def test_right_click_fires_once_per_pinch() -> None:
    """Regression: the original re-fired the right click every 0.45 s while held."""
    d = settled()
    actions = d.hold(pinch("middle"), 2.0) + d.hold(OPEN, 0.3)
    assert of(actions, Click) == [Click("right")]
    assert of(actions[: len(actions) // 2], MoveTo) == []  # frozen while latched


def test_right_click_cooldown_and_rearm() -> None:
    d = settled()
    fast = d.hold(pinch("middle"), 0.08) + d.hold(OPEN, 0.06) + d.hold(pinch("middle"), 0.08)
    assert of(fast, Click) == [Click("right")]  # second pinch inside the 0.4 s cooldown
    d.hold(OPEN, 0.5)
    again = d.hold(pinch("middle"), 0.1)
    assert of(again, Click) == [Click("right")]


def test_ring_pinch_double_clicks() -> None:
    d = settled()
    actions = d.hold(pinch("ring"), 0.2) + d.hold(OPEN, 0.2)
    assert of(actions, Click) == [Click("left", 2)]


def test_ambiguous_pinch_does_nothing() -> None:
    pts = hand_points()
    between = (pts[8] + pts[12]) / 2  # thumb tip exactly between index and middle tips
    d = settled()
    actions = d.hold({"thumb_tip": between}, 0.5) + d.hold(OPEN, 0.3)
    assert of(actions, Click) == [] and of(actions, Press) == []


def test_pinch_works_at_any_distance_from_camera() -> None:
    for palm_px in (50.0, 150.0, 320.0):  # far ... near
        d = Driver(make_engine())
        d.hold(OPEN, 0.5, palm_px=palm_px)
        actions = d.hold(pinch("index"), 0.1, palm_px=palm_px)
        actions += d.hold(OPEN, 0.2, palm_px=palm_px)
        assert of(actions, Click) == [Click("left")], palm_px


# -- tracking loss --------------------------------------------------------------------------
def test_hand_lost_mid_pinch_does_not_click_later() -> None:
    """Regression: the original kept its 'pinched' flag across a lost hand and fired a click
    when the hand re-appeared open."""
    d = settled()
    actions = d.hold(pinch("index"), 0.1) + d.hold(None, 0.5) + d.hold(OPEN, 0.5)
    assert of(actions, Click) == [] and of(actions, Press) == []


def test_hand_reappearing_already_pinched_needs_a_fresh_pinch() -> None:
    d = settled()
    d.hold(None, 0.5)
    actions = d.hold(pinch("index"), 0.4) + d.hold(OPEN, 0.3)
    assert of(actions, Click) == [] and of(actions, Press) == []


def test_short_dropout_keeps_drag_long_dropout_releases() -> None:
    d = settled()
    d.hold(pinch("index"), 0.4)
    assert d.engine.left_state is LeftState.DRAG
    brief = d.hold(None, 0.1) + d.hold(pinch("index"), 0.2)  # < hand_lost_grace
    assert of(brief, Release) == [] and d.engine.left_state is LeftState.DRAG
    long = d.hold(None, 0.5)
    assert of(long, Release) == [Release("left")]
    assert d.outputs[-1].mode is Mode.NO_HAND


def test_reset_returns_release_when_dragging() -> None:
    d = settled()
    d.hold(pinch("index"), 0.4)
    assert d.engine.reset() == [Release("left")]
    assert d.engine.reset() == []


# -- posture modes ---------------------------------------------------------------------------
def scroll_ticks(fps: float, palm_lengths_up: float, palm_px: float = 150.0) -> int:
    d = settled(fps=fps)
    d.hold(SCROLL, 0.2, palm_px=palm_px)
    assert d.outputs[-1].mode is Mode.SCROLL
    up = (CENTER[0], CENTER[1] - palm_lengths_up * palm_px / 720)
    actions = d.hold(SCROLL, 1.0, center=up, palm_px=palm_px)
    assert of(actions, MoveTo) == []  # cursor locked while scrolling
    assert all(s.dx == 0 for s in of(actions, Scroll))
    return sum(s.dy for s in of(actions, Scroll))


def test_scroll_speed_is_rate_based_not_per_frame() -> None:
    """Regression: the original scrolled N ticks *per frame*, so the speed doubled at 60 FPS."""
    expected = 30.0 * (0.5 - 0.15)  # gain * (displacement - dead zone) * 1 s = 10.5 ticks
    at_30, at_60 = scroll_ticks(30, 0.5), scroll_ticks(60, 0.5)
    assert abs(at_30 - expected) <= 1.5 and abs(at_60 - expected) <= 1.5
    assert abs(at_30 - at_60) <= 1


def test_scroll_direction_dead_zone_and_scale_invariance() -> None:
    assert scroll_ticks(60, 0.1) == 0  # inside the dead zone
    assert scroll_ticks(60, -0.5) < 0  # hand down -> scroll down
    assert abs(scroll_ticks(60, 0.5, palm_px=70) - scroll_ticks(60, 0.5, palm_px=280)) <= 1


def test_spread_fingers_are_not_scroll() -> None:
    d = settled()
    d.hold(VICTORY, 0.3)
    assert d.outputs[-1].mode is Mode.POINTER


def test_zoom_follows_the_hand_up_and_down() -> None:
    """Horns, then drag: up = zoom in, down = zoom out, one wheel tick per 0.25 palm lengths.
    The thumb is free, so zooming out never runs out of range (the thumb-spread version did:
    with the index straight, the thumb cannot get much closer to it than where it started)."""
    d = settled()
    d.hold(rock(0.8), 0.3)
    assert d.outputs[-1].mode is Mode.ZOOM
    palm = 150.0 / 720  # handgen's palm length, in normalised y
    up = (CENTER[0], CENTER[1] - 1.1 * palm)
    zoom_in = d.sweep(rock(0.8), CENTER, up, 0.6) + d.hold(rock(0.8), 0.3, center=up)
    assert [z.steps for z in of(zoom_in, Zoom)] == [1, 1, 1, 1]  # 1.1 / 0.25 -> 4 detents
    down = (CENTER[0], CENTER[1] + 0.4 * palm)
    zoom_out = d.sweep(rock(0.8), up, down, 0.8) + d.hold(rock(0.8), 0.3, center=down)
    # from the last detent (-1.0) down to +0.4: 1.4 / 0.25 -> 5 detents, one tick at a time
    assert [z.steps for z in of(zoom_out, Zoom)] == [-1] * 5
    assert of(zoom_in + zoom_out, Click) == [] and of(zoom_in + zoom_out, MoveTo) == []


def test_zoom_ignores_the_thumb_and_small_wobble() -> None:
    d = settled()
    d.hold(rock(0.8), 0.3)
    actions: list[Action] = []
    for i in range(30):  # the thumb moves, the hand wobbles by a fifth of a detent
        wobble = (CENTER[0], CENTER[1] + (0.05 if i % 2 else -0.05) * 150.0 / 720)
        actions += d.step(rock(0.5 + 0.5 * (i % 3) / 2), center=wobble).actions
    assert of(actions, Zoom) == []


def test_leaving_zoom_with_fingers_together_does_not_click() -> None:
    d = settled()
    d.hold(rock(0.5), 0.3)
    d.hold(rock(0.08), 0.3)  # thumb on the index tip, still in ZOOM
    actions = d.hold(pinch("index", 0.08), 0.4) + d.hold(OPEN, 0.3)  # back to POINTER
    assert of(actions, Click) == [] and of(actions, Press) == []


def test_fist_clutch_relative_mode_lets_hand_reposition() -> None:
    d = settled(mode="relative")
    moved = d.sweep(OPEN, CENTER, (0.62, 0.55), 0.3) + d.hold(OPEN, 0.3, center=(0.62, 0.55))
    assert sum(m.dx for m in of(moved, MoveBy)) > 200
    d.hold(FIST, 0.1, center=(0.62, 0.55))
    assert d.outputs[-1].mode is Mode.CLUTCH
    lifted = d.sweep(FIST, (0.62, 0.55), (0.40, 0.55), 0.5)  # carry the "mouse" back
    assert lifted == []
    after = d.hold(OPEN, 0.3, center=(0.40, 0.55))
    assert sum(abs(m.dx) for m in of(after, MoveBy)) <= 3  # no jump on release


def test_fist_in_absolute_mode_pauses_output() -> None:
    d = settled()
    d.hold(FIST, 0.1)
    assert d.sweep(FIST, CENTER, (0.7, 0.55), 0.3) == []


def test_single_glitch_frame_does_not_switch_mode() -> None:
    d = settled()
    for _ in range(3):
        d.step(FIST)
        d.step(FIST)  # two frames < posture_confirm_frames
        d.hold(OPEN, 0.05)
    assert all(o.mode is Mode.POINTER for o in d.outputs[-15:])


@pytest.mark.parametrize("roll_deg", [0.0, 45.0, 90.0, 150.0])
def test_postures_are_rotation_invariant(roll_deg: float) -> None:
    """Regression: `tip.y < pip.y` (the original) misreads every finger once the hand is
    rotated ~90 degrees; ratios of distances do not care about rotation."""
    for pose, expected in ((OPEN, Mode.POINTER), (FIST, Mode.CLUTCH), (SCROLL, Mode.SCROLL)):
        d = Driver(make_engine())
        d.hold(pose, 0.2, roll_deg=roll_deg)
        assert d.outputs[-1].mode is expected, (pose, roll_deg)
    d = Driver(make_engine())
    d.hold(rock(0.5), 0.2, roll_deg=roll_deg)
    assert d.outputs[-1].mode is Mode.ZOOM
