"""Media mode: the shaka toggles it; swipe = previous/next, turning the hand = volume,
fist = play/pause. Uses the synthetic hands from tests/handgen.py."""

from __future__ import annotations

from typing import Any

import pytest

from src.gesture_engine import Mode
from src.media_gestures import wrap_degrees
from src.os_controller import Action, Click, KeyTap, MoveTo, Press
from tests.handgen import CENTER, FIST, OPEN, SHAKA, Driver, make_engine, pinch

WRIST_VIEW: dict[str, Any] = {"pivot": "wrist", "center": (0.5, 0.75)}  # turn around the wrist


def of(actions: list[Action], kind: type) -> list[Any]:
    return [a for a in actions if isinstance(a, kind)]


def taps(actions: list[Action]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tap in of(actions, KeyTap):
        counts[tap.key] = counts.get(tap.key, 0) + tap.count
    return counts


def media_driver(view: dict[str, Any] | None = None, **media: Any) -> Driver:
    view = view or {}
    d = Driver(make_engine(media=media))
    d.hold(OPEN, 0.3, **view)
    d.hold(SHAKA, 0.9, **view)  # 3 debounce frames + 0.6 s hold
    assert d.engine.media_active
    d.hold(OPEN, 0.5, **view)  # a steady open hand becomes the knob's neutral
    assert d.outputs[-1].mode is Mode.MEDIA
    return d


# -- switching ----------------------------------------------------------------------------------
def test_shaka_hold_toggles_media_mode_and_back() -> None:
    d = media_driver()
    assert taps(d.hold(OPEN, 0.3)) == {}  # a steady open hand does nothing
    d.hold(SHAKA, 0.9)
    assert not d.engine.media_active
    d.hold(OPEN, 0.2)
    assert d.outputs[-1].mode is Mode.POINTER


def test_short_shaka_does_not_toggle() -> None:
    d = Driver(make_engine())
    d.hold(OPEN, 0.3)
    d.hold(SHAKA, 0.4)
    d.hold(OPEN, 0.3)
    assert not d.engine.media_active
    assert d.outputs[-1].mode is Mode.POINTER


def test_holding_the_shaka_toggles_only_once() -> None:
    d = Driver(make_engine())
    d.hold(OPEN, 0.3)
    d.hold(SHAKA, 2.5)
    assert d.engine.media_active


def test_media_mode_never_moves_or_clicks_the_mouse() -> None:
    d = media_driver()
    actions = d.sweep(OPEN, CENTER, (0.55, 0.55), 1.0)
    actions += d.hold(pinch("index"), 0.5) + d.hold(OPEN, 0.3)
    assert of(actions, MoveTo) == [] and of(actions, Click) == [] and of(actions, Press) == []


def test_lowering_the_hand_keeps_media_mode() -> None:
    d = media_driver()
    d.hold(None, 1.0)
    assert d.step(OPEN).mode is Mode.MEDIA
    assert of(d.hold(OPEN, 0.3), MoveTo) == []
    assert d.engine.media_active


# -- swipe --------------------------------------------------------------------------------------
def test_fast_swipes_skip_tracks_and_ignore_the_return_stroke() -> None:
    d = media_driver()
    right = d.sweep(OPEN, CENTER, (0.75, 0.55), 0.2)  # ~2 palm lengths in 0.2 s
    assert taps(right) == {"next_track": 1}
    back = d.sweep(OPEN, (0.75, 0.55), CENTER, 0.2)  # hand travelling back, inside the cooldown
    assert taps(back) == {}
    d.hold(OPEN, 0.8)
    left = d.sweep(OPEN, CENTER, (0.25, 0.55), 0.2)
    assert taps(left) == {"previous_track": 1}


def test_slow_sideways_movement_is_not_a_swipe() -> None:
    d = media_driver()
    assert taps(d.sweep(OPEN, CENTER, (0.75, 0.55), 2.0)) == {}


def test_swipe_can_send_arrow_keys_instead() -> None:
    d = media_driver(swipe_action="arrows")
    assert taps(d.sweep(OPEN, CENTER, (0.75, 0.55), 0.2)) == {"right": 1}


# -- volume knob --------------------------------------------------------------------------------
# handgen's roll_deg is counter-clockwise on screen, so a turn "to the right" (clockwise) is < 0.
def test_turning_right_raises_and_turning_left_lowers_the_volume() -> None:
    d = media_driver(WRIST_VIEW)
    up: list[Action] = []
    for i in range(1, 16):  # turn 45 degrees clockwise in 0.25 s ...
        up += d.step(OPEN, roll_deg=-45.0 * i / 15, **WRIST_VIEW).actions
    up += d.hold(OPEN, 1.0, roll_deg=-45.0, **WRIST_VIEW)  # ... and hold it for 1 s
    # rate = 0.25 presses/s per degree beyond the 25-degree dead zone = 5 presses/s
    assert 4 <= taps(up).get("volume_up", 0) <= 7 and "volume_down" not in taps(up)

    for i in range(1, 16):  # back to neutral ...
        d.step(OPEN, roll_deg=-45.0 + 45.0 * i / 15, **WRIST_VIEW)
    assert taps(d.hold(OPEN, 1.0, **WRIST_VIEW)) == {}  # ... stops the volume change

    down = d.hold(OPEN, 1.0, roll_deg=45.0, **WRIST_VIEW)
    assert 4 <= taps(down).get("volume_down", 0) <= 6 and "volume_up" not in taps(down)


def test_small_turns_inside_the_dead_zone_do_nothing() -> None:
    d = media_driver(WRIST_VIEW)
    assert taps(d.hold(OPEN, 1.0, roll_deg=-20.0, **WRIST_VIEW)) == {}


def test_knob_direction_can_be_inverted() -> None:
    d = media_driver(WRIST_VIEW, knob_invert=True)
    assert "volume_down" in taps(d.hold(OPEN, 1.0, roll_deg=-45.0, **WRIST_VIEW))


# -- play / pause -------------------------------------------------------------------------------
def test_fist_plays_or_pauses_once_per_closing() -> None:
    d = media_driver()
    assert taps(d.hold(FIST, 1.0)) == {"play_pause": 1}  # holding it does not repeat
    d.hold(OPEN, 0.5)
    assert taps(d.hold(FIST, 0.5)) == {"play_pause": 1}


def test_a_quick_second_fist_inside_the_cooldown_is_ignored() -> None:
    d = media_driver()
    d.hold(FIST, 0.1)
    d.hold(OPEN, 0.1)
    assert taps(d.hold(FIST, 0.5)) == {}


@pytest.mark.parametrize(
    ("value", "expected"), [(0.0, 0.0), (190.0, -170.0), (-190.0, 170.0), (359.0, -1.0)]
)
def test_wrap_degrees(value: float, expected: float) -> None:
    assert wrap_degrees(value) == pytest.approx(expected)
