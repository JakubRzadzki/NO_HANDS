"""State machine tests on synthetic hands. Several are regression tests for bugs of the
original single-file script (named in the test docstrings)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.gesture_engine import HysteresisLatch, LeftState, Mode
from src.os_controller import (
    Action,
    Click,
    KeyTap,
    MoveBy,
    MoveTo,
    Press,
    Release,
    Scroll,
    Zoom,
)
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


def test_latch_can_let_go_more_slowly_than_it_engages() -> None:
    latch = HysteresisLatch(0.27, 0.36, confirm_frames=2, release_frames=3)
    assert [latch.update(0.1) for _ in range(2)] == [False, True]
    assert [latch.update(0.9) for _ in range(2)] == [True, True]  # two bad frames: still held
    assert latch.update(0.1) is True  # ... and the count starts again
    assert [latch.update(0.9) for _ in range(3)] == [True, True, False]


# -- pointer ---------------------------------------------------------------------------------
def test_open_hand_moves_cursor_in_pointer_mode() -> None:
    d = settled()
    actions = d.sweep(OPEN, CENTER, (0.65, 0.55), 0.5) + d.hold(OPEN, 0.5, center=(0.65, 0.55))
    moves = of(actions, MoveTo)
    assert d.outputs[-1].mode is Mode.POINTER
    assert moves and moves[-1].x > 1300 and abs(moves[-1].y - 540) < 30


def test_quick_pinch_is_one_click_at_the_frozen_position() -> None:
    """Press when the fingers meet, release when they part, and no movement in between: the
    OS sees one clean click on one pixel."""
    d = settled()
    # The hand drifts a little while pinching (less than drag_start_px on screen).
    during = d.sweep(pinch("index"), CENTER, (0.505, 0.55), 0.1)
    after = d.hold(OPEN, 0.3, center=(0.505, 0.55))
    actions = during + after
    assert of(actions, Press) == [Press("left")] and of(actions, Release) == [Release("left")]
    assert of(actions, Click) == []
    pressed, released = actions.index(Press("left")), actions.index(Release("left"))
    assert of(actions[pressed:released], MoveTo) == []  # frozen while the button is down
    assert of(during, MoveTo) == []
    assert d.outputs[-1].left_state is LeftState.IDLE


def test_the_button_is_down_from_the_first_moment_of_the_pinch() -> None:
    """Feedback from a real user: "thumb + index is not a continuous press". The button used
    to go down only after 0.25 s of holding; now it is down as long as the fingers touch."""
    d = settled()  # 60 FPS
    first = d.hold(pinch("index"), 0.05)  # three frames
    assert of(first, Press) == [Press("left")]  # pressed after two confirming frames
    held = d.hold(pinch("index"), 3.0)
    assert of(held, Press) == [] and of(held, Release) == []  # one press, held for 3 s
    assert d.engine.left_state is LeftState.DRAG
    assert of(d.hold(OPEN, 0.2), Release) == [Release("left")]


def test_a_held_pinch_survives_bad_frames_and_a_wobbling_grip() -> None:
    """Real MediaPipe output while a pinching hand moves: the distance wobbles (p95 0.30 palm
    lengths, the odd frame at 0.5), and with the old 0.22 / 0.32 thresholds the hold broke."""
    d = settled()
    d.hold(pinch("index"), 0.4)
    actions: list[Action] = []
    for i in range(120):  # 2 s of dragging with a loose, noisy grip
        ratio = 0.30 if i % 3 else 0.12
        if i % 20 == 0:
            ratio = 0.55  # a single wild frame
        if i % 41 in (5, 6):
            ratio = 0.50  # two in a row (three would be letting go)
        actions += d.step(pinch("index", ratio)).actions
    assert of(actions, Release) == [] and d.engine.left_state is LeftState.DRAG
    assert of(d.hold(OPEN, 0.2), Release) == [Release("left")]


def test_a_relaxed_open_hand_lets_the_button_go_and_can_pinch_again() -> None:
    """A relaxed open hand keeps the thumb close to the index finger: 0.41-0.46 palm lengths
    on a real photo. The button must be up there, and the next pinch must work from there."""
    d = settled()
    relaxed = pinch("index", 0.42)
    assert of(d.hold(pinch("index"), 0.5), Press) == [Press("left")]
    assert of(d.hold(relaxed, 0.2), Release) == [Release("left")]
    again = d.hold(pinch("index"), 0.1) + d.hold(relaxed, 0.2)
    assert of(again, Press) == [Press("left")] and of(again, Release) == [Release("left")]


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
        actions += d.hold(pinch("index", 0.33), 1 / 60) + d.hold(pinch("index", 0.18), 1 / 60)
    actions += d.hold(OPEN, 0.2)
    assert of(actions, Press) == [Press("left")] and of(actions, Release) == [Release("left")]


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


def test_ring_pinch_double_click_is_optional() -> None:
    d = settled()
    assert of(d.hold(pinch("ring"), 0.2) + d.hold(OPEN, 0.2), Click) == []  # off by default
    d = settled(ring_double_click=True)
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
        assert of(actions, Press) == [Press("left")], palm_px
        assert of(actions, Release) == [Release("left")], palm_px


# -- what is not a pinch -----------------------------------------------------------------------
def test_thumb_resting_on_a_folded_finger_is_not_a_pinch() -> None:
    """On real photos of the V sign the thumb lies on the folded ring finger: that fired the
    thumb + ring double click in 36% of them. A pinching finger reaches for the thumb."""
    ring_tip = hand_points(curl={"ring": 1.0, "pinky": 1.0})[16]
    v_sign: dict[str, Any] = {
        "curl": {"ring": 1.0, "pinky": 1.0},
        "spread": {"index": 10.0, "middle": -8.0},
        "thumb_tip": ring_tip + np.array([0.005, 0.0, -0.005]),
    }
    d = settled()
    actions = d.hold(v_sign, 0.5) + d.hold(OPEN, 0.3)
    assert of(actions, Click) == [] and of(actions, Press) == []
    assert d.outputs[-1].mode is Mode.POINTER


def test_a_fist_with_the_thumb_on_the_index_is_a_fist_not_a_click() -> None:
    """In a quarter of real fists the thumb tip lies right next to the folded index tip."""
    index_tip = hand_points(**FIST)[8]
    fist: dict[str, Any] = {
        "curl": FIST["curl"],
        "thumb_tip": index_tip + np.array([0.008, 0.0, -0.004]),
    }
    d = settled()
    actions = d.hold(fist, 0.6)
    assert d.outputs[-1].mode is Mode.CLUTCH
    assert of(actions + d.hold(OPEN, 0.3), Press) == []


def test_a_pinch_needs_an_open_hand_but_the_hold_does_not() -> None:
    """Starting a pinch takes an open hand (another finger straight): with all the other
    fingers folded, real fists look the same. Once the button is down, the other fingers may
    relax and curl: the hold only watches thumb and index."""
    pen: dict[str, Any] = {"curl": {"middle": 1.0, "ring": 1.0, "pinky": 1.0}}
    d = settled()
    d.hold(pen, 0.2)
    assert of(d.hold({**pen, **pinch("index")}, 0.5) + d.hold(pen, 0.2), Press) == []
    d.hold(OPEN, 0.3)
    assert of(d.hold(pinch("index"), 0.3), Press) == [Press("left")]  # open hand: pressed
    curled = d.hold({**pen, **pinch("index")}, 1.0)  # ... and the other fingers relax
    assert of(curled, Release) == [] and d.engine.left_state is LeftState.DRAG
    assert of(d.hold(OPEN, 0.2), Release) == [Release("left")]


def test_two_quick_pinches_are_a_double_click_on_one_pixel() -> None:
    d = settled()
    actions: list[Action] = []
    for i in range(2):  # the hand drifts a few pixels' worth between the two pinches
        actions += d.hold(pinch("index"), 0.08, center=(CENTER[0] + 0.002 * i, CENTER[1]))
        actions += d.hold(OPEN, 0.10, center=(CENTER[0] + 0.003 * i + 0.001, CENTER[1]))
    assert of(actions, Press) == [Press("left")] * 2
    assert of(actions, Release) == [Release("left")] * 2
    first, last = (
        actions.index(Press("left")),
        len(actions) - 1 - actions[::-1].index(Release("left")),
    )
    assert of(actions[first:last], MoveTo) == []  # both clicks on the same pixel
    assert of(d.hold(OPEN, 0.6, center=(0.6, 0.55)), MoveTo)  # then the cursor is free again


# -- flick: previous / next slide from the hand mouse ---------------------------------------------
def flick(d: Driver, to_x: float, out: float = 0.15, back: float = 0.25) -> list[Action]:
    """The open hand goes sideways to ``to_x`` and straight back."""
    there = (to_x, CENTER[1])
    actions = d.sweep(OPEN, CENTER, there, out) + d.sweep(OPEN, there, CENTER, back)
    return actions + d.hold(OPEN, 0.5)


def keys(actions: list[Action]) -> list[str]:
    return [a.key for a in of(actions, KeyTap)]


def test_flick_and_back_changes_the_slide_and_leaves_the_cursor_where_it_was() -> None:
    d = settled()
    before = d.outputs[-1]
    actions = flick(d, CENTER[0] + 0.25)  # 2.1 palm lengths to the right and back
    assert keys(actions) == ["right"]
    assert of(actions, Click) == [] and of(actions, Press) == []
    moves = of(actions, MoveTo)
    assert before.mode is Mode.POINTER and abs(moves[-1].x - 960) < 15  # back where it was
    d.hold(OPEN, 1.0)
    assert keys(flick(d, CENTER[0] - 0.25)) == ["left"]


def test_moving_the_cursor_fast_without_coming_back_is_not_a_flick() -> None:
    d = settled()
    there = (CENTER[0] + 0.25, CENTER[1])
    one_way = d.sweep(OPEN, CENTER, there, 0.15) + d.hold(OPEN, 1.0, center=there)
    assert keys(one_way) == []
    slow = d.sweep(OPEN, there, CENTER, 0.8) + d.hold(OPEN, 0.5)  # ... and a slow way back
    assert keys(slow) == []


def test_small_or_slow_out_and_back_moves_are_not_flicks() -> None:
    d = settled()
    assert keys(flick(d, CENTER[0] + 0.10)) == []  # 0.85 palm lengths: too short
    d.hold(OPEN, 1.0)
    assert keys(flick(d, CENTER[0] + 0.25, out=0.8, back=0.8)) == []  # too slow


def test_dragging_back_and_forth_is_not_a_flick() -> None:
    d = settled()
    d.hold(pinch("index"), 0.4)
    there = (CENTER[0] + 0.25, CENTER[1])
    shake = d.sweep(pinch("index"), CENTER, there, 0.15) + d.sweep(
        pinch("index"), there, CENTER, 0.2
    )
    assert keys(shake + d.hold(pinch("index"), 0.3)) == []
    assert d.engine.left_state is LeftState.DRAG


def test_flick_can_be_switched_off() -> None:
    d = Driver(make_engine(flick_slides=False))
    d.hold(OPEN, 0.5)
    assert keys(flick(d, CENTER[0] + 0.25)) == []


# -- tracking loss --------------------------------------------------------------------------
def test_hand_lost_mid_pinch_releases_and_does_not_click_later() -> None:
    """Regression: the original kept its 'pinched' flag across a lost hand and fired a click
    when the hand re-appeared open. The button that the pinch pressed is let go when the hand
    stays lost, and nothing more happens when it comes back."""
    d = settled()
    pinched = d.hold(pinch("index"), 0.1)
    lost = d.hold(None, 0.8)
    back = d.hold(OPEN, 0.5)
    assert of(pinched, Press) == [Press("left")] and of(lost, Release) == [Release("left")]
    assert of(back, Press) == [] and of(back, Release) == [] and of(back, Click) == []


def test_hand_reappearing_already_pinched_needs_a_fresh_pinch() -> None:
    d = settled()
    d.hold(None, 0.5)
    actions = d.hold(pinch("index"), 0.4) + d.hold(OPEN, 0.3)
    assert of(actions, Click) == [] and of(actions, Press) == []


def test_short_dropout_keeps_drag_long_dropout_releases() -> None:
    d = settled()
    d.hold(pinch("index"), 0.4)
    assert d.engine.left_state is LeftState.DRAG
    brief = d.hold(None, 0.4) + d.hold(pinch("index"), 0.2)  # < drag_lost_grace (0.5 s)
    assert of(brief, Release) == [] and d.engine.left_state is LeftState.DRAG
    long = d.hold(None, 0.8)
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
