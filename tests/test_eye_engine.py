"""--eyes on synthetic faces: gaze / nose pointer, head nudge, blinks, winks, scroll, pause."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from config import EyeConfig, FaceConfig, FilterConfig
from src.eye_engine import EyeEngine, FaceMode
from src.os_controller import Action, Click, MoveTo, Scroll
from tests.facegen import SCREEN, FaceDriver, gaze_for, linear_calibration

CLOSED = (0.05, 0.05)
RIGHT_WINK = (1.0, 0.05)  # (user's left eye, user's right eye)
LEFT_WINK = (0.05, 1.0)


def of(actions: list[Action], kind: type) -> list[Any]:
    return [a for a in actions if isinstance(a, kind)]


def driver(pointer: str = "gaze", noise: float = 0.0, **overrides: Any) -> FaceDriver:
    cfg = EyeConfig(pointer=pointer)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    model = linear_calibration(noise=noise) if pointer == "gaze" else None
    d = FaceDriver(EyeEngine(cfg, FaceConfig(), SCREEN, FilterConfig(), model))
    d.hold(1.0, gaze=gaze_for(960, 540))  # learn the open-eye baseline, settle the cursor
    return d


# -- pointer ------------------------------------------------------------------------------------
def test_looking_at_a_point_moves_the_cursor_there() -> None:
    d = driver()
    moves = of(d.hold(1.0, gaze=gaze_for(1500, 300)), MoveTo)
    assert moves and abs(moves[-1].x - 1500) < 25 and abs(moves[-1].y - 300) < 25


def test_gaze_noise_does_not_jitter_the_cursor() -> None:
    d = driver(noise=50.0)  # calibration measured 50 px of jitter: the radius adapts to it
    rng = np.random.default_rng(3)
    pointers = []
    for _ in range(90):
        out = d.step(gaze=gaze_for(960 + rng.normal(0, 50), 540 + rng.normal(0, 50)))
        if out.pointer is not None:
            pointers.append(out.pointer)
    assert np.std(np.array(pointers), axis=0).max() < 15  # raw gaze noise is 50 px


def test_fixation_radius_comes_from_the_calibration_jitter() -> None:
    def radius(noise: float, **cfg: Any) -> float:
        model = linear_calibration(noise=noise)
        engine = EyeEngine(EyeConfig(**cfg), FaceConfig(), SCREEN, FilterConfig(), model)
        return engine.fixation_radius

    assert radius(80.0) == pytest.approx(240.0)  # 3 sigma
    assert radius(0.0) == pytest.approx(0.03 * SCREEN[0])  # never below 3% of the screen
    assert radius(500.0) == pytest.approx(0.20 * SCREEN[0])  # ... or above 20%
    assert radius(80.0, fixation_radius=0.1) == pytest.approx(192.0)  # fixed in the config


def test_gaze_pointer_needs_a_calibration() -> None:
    with pytest.raises(ValueError):
        EyeEngine(EyeConfig(), FaceConfig(), SCREEN, FilterConfig(), None)


# -- head nudge -----------------------------------------------------------------------------------
def test_a_small_head_turn_nudges_the_cursor_and_a_gaze_jump_resets_it() -> None:
    d = driver()
    d.hold(1.0, gaze=gaze_for(1200, 500))
    out = d.step(gaze=gaze_for(1200, 500))
    assert out.pointer == pytest.approx((1200, 500), abs=10)
    d.hold(1.5, gaze=gaze_for(1200, 500), yaw=4.0, pitch=-3.0)  # eyes stay, head turns
    out = d.step(gaze=gaze_for(1200, 500), yaw=4.0, pitch=-3.0)
    per_degree = 0.03 * SCREEN[0]  # head_nudge, beyond the 1.5 degree dead zone
    expected = (2.5 * per_degree, -1.5 * per_degree)
    assert out.pointer == pytest.approx((1200 + expected[0], 500 + expected[1]), abs=12)
    assert out.nudge == pytest.approx(expected, abs=12)
    # The eyes jump elsewhere: the cursor lands where they look, whatever the head did before.
    d.hold(1.0, gaze=gaze_for(400, 300), yaw=3.0, pitch=-2.0)
    out = d.step(gaze=gaze_for(400, 300), yaw=3.0, pitch=-2.0)
    assert out.pointer == pytest.approx((400, 300), abs=10)


def test_natural_head_drift_does_not_move_the_cursor() -> None:
    """A head never holds still. Within the dead zone it must not drag the cursor away from
    where the eyes look (without one, in simulation, a fidgety head added 36% error)."""
    d = driver()
    d.hold(1.0, gaze=gaze_for(1200, 500))
    pointers = []
    for i in range(60):  # +-1.2 degrees of drift while the eyes stay on the target
        out = d.step(gaze=gaze_for(1200, 500), yaw=1.2 * ((i % 20) / 10 - 1), pitch=0.8)
        if out.pointer is not None:
            pointers.append(out.pointer)
    assert max(abs(p[0] - 1200) for p in pointers) < 10
    assert max(abs(p[1] - 500) for p in pointers) < 10


def test_the_gaze_map_data_is_reported() -> None:
    d = driver(noise=40.0)
    out = d.step(gaze=gaze_for(960, 540))
    assert out.gaze_raw == pytest.approx((960, 540), abs=5)
    assert out.fixation == pytest.approx((960, 540), abs=5)
    assert out.radius == pytest.approx(120.0)  # 3 x the calibration jitter


def test_the_nudge_waits_for_the_head_to_finish_its_part_of_a_gaze_shift() -> None:
    d = driver()
    pointers = []
    for i in range(12):  # the eyes jump, the head follows them for another 0.4 s
        out = d.step(gaze=gaze_for(1500, 540), yaw=min(i, 8) * 1.5)
        if out.pointer is not None:
            pointers.append(out.pointer[0])
    assert max(pointers[2:]) < 1500 + 20  # no overshoot from the late head movement


def test_nudge_can_be_switched_off_and_is_limited() -> None:
    d = driver(head_nudge=0.0)
    d.hold(1.0, gaze=gaze_for(960, 540), yaw=10.0)
    assert d.step(gaze=gaze_for(960, 540), yaw=10.0).pointer == pytest.approx((960, 540), abs=10)
    d = driver(head_nudge_max=0.05)
    d.hold(1.0, gaze=gaze_for(960, 540))
    d.hold(1.5, gaze=gaze_for(960, 540), yaw=20.0)
    out = d.step(gaze=gaze_for(960, 540), yaw=20.0)
    assert out.pointer is not None and out.pointer[0] == pytest.approx(960 + 96, abs=5)


# -- blinks ---------------------------------------------------------------------------------------
def test_natural_blink_is_ignored() -> None:
    d = driver()
    assert of(d.hold(0.15, openness=CLOSED) + d.hold(0.5), Click) == []


def test_deliberate_blink_clicks_where_you_looked() -> None:
    d = driver()
    d.hold(0.6, gaze=gaze_for(800, 400))
    during = d.hold(0.5, openness=CLOSED, gaze=gaze_for(100, 1000))  # iris is garbage when shut
    after = d.hold(0.5, gaze=gaze_for(800, 400))
    assert of(during, MoveTo) == []  # the cursor did not follow the closing eyelids
    assert of(after, Click) == [Click("left")]
    assert isinstance(after[0], Click)  # clicked before any further movement


def test_two_deliberate_blinks_make_a_double_click() -> None:
    d = driver()
    actions = (
        d.hold(0.5, openness=CLOSED) + d.hold(0.3) + d.hold(0.5, openness=CLOSED) + d.hold(0.3)
    )
    assert of(actions, Click) == [Click("left"), Click("left", 2)]


def test_lowered_lids_when_looking_down_are_not_a_blink() -> None:
    d = driver()
    assert of(d.hold(1.0, openness=(0.65, 0.65)) + d.hold(0.3), Click) == []


def test_narrow_eyes_get_their_own_open_baseline() -> None:
    d = FaceDriver(
        EyeEngine(EyeConfig(), FaceConfig(), SCREEN, FilterConfig(), linear_calibration())
    )
    d.hold(1.0, eye_aspect=0.18)  # open eyes of this user are narrower than average
    actions = d.hold(0.5, eye_aspect=0.18, openness=CLOSED) + d.hold(0.3, eye_aspect=0.18)
    assert of(actions, Click) == [Click("left")]


def test_losing_the_face_mid_blink_does_not_click() -> None:
    d = driver()
    actions = d.hold(0.3, openness=CLOSED) + d.hold(0.3, present=False) + d.hold(0.5)
    assert of(actions, Click) == []


# -- winks, scroll, pause -------------------------------------------------------------------------
def test_winks_right_click_and_toggle_scroll_mode() -> None:
    d = driver()
    assert of(d.hold(0.5, openness=RIGHT_WINK) + d.hold(0.3), Click) == [Click("right")]

    d.hold(0.5, openness=LEFT_WINK)
    assert d.step().mode is FaceMode.EYE_SCROLL
    down = d.hold(1.0, gaze=gaze_for(960, 1060))  # look near the bottom edge
    assert sum(s.dy for s in of(down, Scroll)) < -5
    assert of(down, MoveTo) == []  # the cursor stays put while scrolling
    assert of(d.hold(1.0, gaze=gaze_for(960, 540)), Scroll) == []  # centre: no scrolling

    d.hold(0.5, openness=LEFT_WINK)
    assert d.step().mode is FaceMode.EYES


def test_winks_can_be_swapped() -> None:
    d = driver(swap_winks=True)
    assert of(d.hold(0.5, openness=LEFT_WINK) + d.hold(0.3), Click) == [Click("right")]
    d.hold(0.5, openness=RIGHT_WINK)
    assert d.step().mode is FaceMode.EYE_SCROLL


def test_long_closure_pauses_and_resumes() -> None:
    d = driver()
    d.hold(2.0, openness=CLOSED)
    assert d.step().mode is FaceMode.PAUSED
    ignored = d.hold(0.5, openness=CLOSED) + d.hold(0.5, gaze=gaze_for(100, 100))
    assert of(ignored, Click) == [] and of(ignored, MoveTo) == []
    d.hold(2.0, openness=CLOSED)
    assert d.step().mode is FaceMode.EYES


# -- nose pointer ---------------------------------------------------------------------------------
def test_nose_pointer_follows_the_head_and_recentres_after_a_pause() -> None:
    d = driver(pointer="nose")
    moves = of(d.hold(1.0, center=(700.0, 330.0)), MoveTo)  # head 60 px to the right
    assert moves and moves[-1].x > 1500

    d.hold(2.0, openness=CLOSED)
    d.step()  # paused
    elsewhere: dict[str, Any] = {"center": (500.0, 300.0)}
    d.hold(0.5, **elsewhere)
    d.hold(2.0, openness=CLOSED, **elsewhere)
    d.step(**elsewhere)  # resumed: this head position becomes the new centre
    moves = of(d.hold(1.0, **elsewhere), MoveTo)
    assert moves and abs(moves[-1].x - 960) < 60 and abs(moves[-1].y - 540) < 60


# -- dwell click (EyeDrive) -----------------------------------------------------------------------
def test_dwell_clicks_once_per_spot() -> None:
    d = driver(click="dwell")
    looked = d.hold(2.5, gaze=gaze_for(1400, 300))  # jump to a spot and keep looking
    assert of(looked, Click) == [Click("left")]  # once, however long you look
    d.hold(0.5, gaze=gaze_for(400, 800))
    assert of(d.hold(1.5, gaze=gaze_for(400, 800)), Click) == [Click("left")]  # a new spot


def test_dwell_needs_the_cursor_to_move_first() -> None:
    """Starting, resuming or coming back from eyes off with the eyes on one spot must not
    click: the user has not chosen that spot."""
    d = driver(click="dwell")  # the driver starts by looking at the centre for 1 s
    assert of(d.hold(2.0, gaze=gaze_for(960, 540)), Click) == []


def test_dwell_progress_is_reported_and_a_natural_blink_only_pauses_it() -> None:
    d = driver(click="dwell", dwell_time=1.0)
    d.hold(0.7, gaze=gaze_for(1400, 300))
    progress = d.step(gaze=gaze_for(1400, 300)).dwell
    assert 0.0 < progress < 1.0
    d.hold(0.12, openness=CLOSED, gaze=gaze_for(1400, 300))  # a natural blink
    d.hold(0.2, gaze=gaze_for(1400, 300))  # the cursor is frozen just after a blink
    assert d.step(gaze=gaze_for(1400, 300)).dwell >= progress  # not restarted
    assert of(d.hold(1.0, gaze=gaze_for(1400, 300)), Click) == [Click("left")]


def test_with_dwell_only_a_blink_does_not_click() -> None:
    d = driver(click="dwell")
    assert of(d.hold(0.5, openness=CLOSED) + d.hold(0.3), Click) == []


def test_with_both_a_blink_click_is_not_followed_by_a_dwell_click() -> None:
    d = driver(click="both", dwell_time=1.5)
    d.hold(0.6, gaze=gaze_for(1400, 300))
    actions = d.hold(0.5, openness=CLOSED) + d.hold(2.0, gaze=gaze_for(1400, 300))
    assert of(actions, Click) == [Click("left")]  # the blink's, not a second one from the dwell


def test_no_dwell_clicks_while_scrolling_or_paused() -> None:
    d = driver(click="dwell")
    d.hold(0.5, openness=LEFT_WINK)  # scroll mode
    assert of(d.hold(2.0, gaze=gaze_for(960, 900)), Click) == []


# -- eyes off (EyeDrive) --------------------------------------------------------------------------
def test_looking_off_the_screen_moves_and_scrolls_nothing() -> None:
    d = driver()
    d.hold(1.0, gaze=gaze_for(1200, 500))
    keyboard = gaze_for(960, 1700)  # well below the screen
    assert of(d.hold(1.0, gaze=keyboard), MoveTo) == []
    assert d.step(gaze=keyboard).mode is FaceMode.EYES_OFF
    back = d.hold(1.0, gaze=gaze_for(700, 400))
    assert of(back, MoveTo) and abs(of(back, MoveTo)[-1].x - 700) < 25
    assert d.step(gaze=gaze_for(700, 400)).mode is FaceMode.EYES

    d.hold(0.5, openness=LEFT_WINK)  # scroll mode: looking at the keyboard must not scroll
    assert of(d.hold(1.0, gaze=keyboard), Scroll) == []


def test_eyes_off_cancels_a_running_dwell() -> None:
    d = driver(click="dwell", dwell_time=1.0)
    d.hold(0.8, gaze=gaze_for(1400, 300))  # most of the way to a click
    d.hold(0.5, gaze=gaze_for(960, 1800))  # glance at the keyboard
    assert of(d.hold(1.5, gaze=gaze_for(1400, 300)), Click) == []  # back: no surprise click
    d.hold(0.5, present=False)  # the face lost for a moment works the same way
    assert of(d.hold(1.5, gaze=gaze_for(1400, 300)), Click) == []


# -- the scroll joystick --------------------------------------------------------------------------
def test_scroll_joystick_scrolls_sideways_near_the_side_edges() -> None:
    d = driver()
    d.hold(0.5, openness=LEFT_WINK)
    right = d.hold(1.0, gaze=gaze_for(1900, 540))
    assert sum(s.dx for s in of(right, Scroll)) > 5 and sum(s.dy for s in of(right, Scroll)) == 0
    left = d.hold(1.0, gaze=gaze_for(20, 540))
    assert sum(s.dx for s in of(left, Scroll)) < -5
    d.hold(0.2, gaze=gaze_for(1300, 540))  # the cursor's jump takes a couple of frames
    assert of(d.hold(1.0, gaze=gaze_for(1300, 540)), Scroll) == []  # inside the stop zone

    d = driver(scroll_horizontal=False)
    d.hold(0.5, openness=LEFT_WINK)
    assert of(d.hold(1.0, gaze=gaze_for(1900, 540)), Scroll) == []
