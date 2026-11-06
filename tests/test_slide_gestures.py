"""--slides on synthetic hands that move like people: minimum-jerk strokes that come straight
back, start while the hand is still moving, or are just a flick of the wrist. And everything a
presenter does in between, which must do nothing."""

from __future__ import annotations

from typing import Any

import pytest

from config import AppConfig
from src.gesture_engine import Mode
from src.os_controller import Action, KeyTap, Zoom
from src.slide_gestures import Phase, SlideGestures
from tests.handgen import FIST, OPEN, Driver, pinch

PALM_PX = 110.0
PL = PALM_PX / 1280  # one palm length in normalised x
MID = (0.5, 0.6)  # wrist position


def at(dx_pl: float, dy_pl: float = 0.0) -> tuple[float, float]:
    """A wrist position dx / dy palm lengths from MID (+ = right / down)."""
    return MID[0] + dx_pl * PL, MID[1] + dy_pl * PALM_PX / 720


def events(actions: list[Action]) -> list[str]:
    out = []
    for a in actions:
        if isinstance(a, KeyTap):
            out += [a.key] * a.count
        elif isinstance(a, Zoom):
            out += ["zoom in" if a.steps > 0 else "zoom out"] * abs(a.steps)
    return out


class Hand:
    """Drives SlideGestures with a hand whose wrist follows minimum-jerk strokes."""

    def __init__(self, fps: float = 30.0, palm_px: float = PALM_PX, **slides: Any) -> None:
        cfg = AppConfig()
        for name, value in slides.items():
            setattr(cfg.slides, name, value)
        cfg.validate()
        self.d = Driver(SlideGestures(cfg.slides, cfg.gesture), fps=fps)
        self.palm_px = palm_px
        self.pos, self.roll = MID, 0.0

    def frame(self, pose: dict[str, Any] | None = OPEN) -> list[Action]:
        if pose is None:
            return self.d.step(None).actions
        view = {"center": self.pos, "roll_deg": self.roll, "pivot": "wrist"}
        return self.d.step(pose, palm_px=self.palm_px, **view).actions

    def hold(self, seconds: float, pose: dict[str, Any] | None = OPEN) -> list[Action]:
        actions: list[Action] = []
        for _ in range(max(1, round(seconds / self.d.dt))):
            actions += self.frame(pose)
        return actions

    def move(
        self,
        to: tuple[float, float],
        seconds: float,
        roll: float | None = None,
        pose: dict[str, Any] | None = OPEN,
    ) -> list[Action]:
        start, roll0 = self.pos, self.roll
        roll1 = roll0 if roll is None else roll
        n = max(1, round(seconds / self.d.dt))
        actions: list[Action] = []
        for i in range(1, n + 1):
            s = i / n
            k = 10 * s**3 - 15 * s**4 + 6 * s**5  # minimum-jerk profile
            self.pos = (start[0] + (to[0] - start[0]) * k, start[1] + (to[1] - start[1]) * k)
            self.roll = roll0 + (roll1 - roll0) * k
            actions += self.frame(pose)
        return actions


def ready(**kw: Any) -> Hand:
    h = Hand(**kw)
    h.hold(0.6)
    assert h.d.engine.phase is Phase.READY
    return h


# -- swipes that people actually make ------------------------------------------------------------
def test_swipe_and_hold() -> None:
    h = ready()
    assert events(h.move(at(2.5), 0.3) + h.hold(0.5)) == ["right"]


def test_swipe_that_comes_straight_back() -> None:
    """The most natural swipe. The first version waited for the hand to rest after the stroke
    and missed every one of these."""
    h = ready()
    actions = h.move(at(2.5), 0.3) + h.hold(0.05) + h.move(MID, 0.4) + h.hold(0.6)
    assert events(actions) == ["right"]


def test_swipe_started_while_moving_into_place() -> None:
    h = Hand()
    h.pos = at(-1.5, 1.5)
    h.hold(0.5)
    actions = h.move(MID, 0.6) + h.move(at(2.5), 0.3) + h.hold(0.5)
    assert events(actions) == ["right"]


def test_flick_of_the_wrist() -> None:
    """The arm barely moves (0.6 palm lengths), the hand swings 35 degrees around the wrist."""
    h = ready()
    actions = h.move(at(0.6), 0.25, roll=-35.0) + h.hold(0.1) + h.move(MID, 0.35, roll=0.0)
    assert events(actions + h.hold(0.6)) == ["right"]


def test_an_unhurried_swipe_still_counts() -> None:
    h = ready()
    assert events(h.move(at(1.6), 0.4) + h.hold(0.5)) == ["right"]  # ~14 cm in 0.4 s


def test_slower_swipe_and_a_swipe_left() -> None:
    h = ready()
    assert events(h.move(at(2.5), 0.5) + h.hold(1.0)) == ["right"]
    assert events(h.move(MID, 0.3) + h.hold(0.5)) == ["left"]  # after a pause: not a return


def test_swipe_back_and_forth_turns_one_slide_per_swipe() -> None:
    h = ready()
    pressed = []
    for _ in range(3):
        pressed += events(h.move(at(2.5), 0.3) + h.move(MID, 0.4) + h.hold(0.2))
    assert pressed == ["right", "right", "right"]  # the way back is never "left"


def test_a_short_tracking_dropout_mid_swipe_is_tolerated() -> None:
    h = ready()
    actions = h.move(at(1.0), 0.12)
    actions += h.hold(0.07, pose=None)  # motion blur: two frames without a hand
    h.pos = at(1.8)
    actions += h.move(at(2.5), 0.12) + h.hold(0.5)
    assert events(actions) == ["right"]


@pytest.mark.parametrize("palm_px", [70.0, 220.0])
def test_hand_distance_does_not_matter(palm_px: float) -> None:
    shift = 2.0 * palm_px / 1280  # the same stroke in palm lengths, whatever the hand's size
    h = Hand(palm_px=palm_px)
    h.pos = (MID[0] - shift / 2, MID[1])
    h.hold(0.6)
    assert events(h.move((MID[0] + shift / 2, MID[1]), 0.3) + h.hold(0.5)) == ["right"]


def test_works_at_60_fps_too() -> None:
    h = ready(fps=60.0)
    assert events(h.move(at(2.5), 0.3) + h.move(MID, 0.4) + h.hold(0.5)) == ["right"]


# -- what a presenter does that must not change the slide ------------------------------------------
def test_slow_repositioning_does_nothing() -> None:
    h = ready()
    assert events(h.move(at(2.0), 1.2) + h.hold(0.5) + h.move(MID, 1.2) + h.hold(0.5)) == []


def test_talking_gestures_do_nothing() -> None:
    h = ready()
    actions: list[Action] = []
    for side in (1, -1, 1, -1):  # bobbing sideways while talking
        actions += h.move(at(0.8 * side, -0.4), 0.35) + h.move(MID, 0.35)
    for _ in range(3):  # up/down "beats": as zoom strokes these fired every time
        actions += h.move(at(0.0, 1.5), 0.25) + h.move(MID, 0.3) + h.hold(0.3)
    actions += h.move(at(1.5), 0.5) + h.move(MID, 0.5) + h.hold(0.5)  # a slow, wide sweep
    assert events(actions) == []


def test_raising_the_hand_into_view_and_lowering_it_out_do_nothing() -> None:
    h = Hand()
    h.pos = at(0.0, 4.0)  # appears at the bottom while still moving up
    assert events(h.move(MID, 0.4) + h.hold(0.6)) == []
    actions = h.move(at(0.0, 5.0), 0.3) + h.hold(0.5, pose=None)  # lowered out of view
    assert events(actions) == []
    assert h.d.outputs[-1].mode is Mode.NO_HAND


def test_a_moving_fist_does_not_swipe() -> None:
    h = ready()
    h.hold(0.3, pose=FIST)
    assert events(h.move(at(2.5), 0.3, pose=FIST) + h.hold(0.5, pose=FIST)) == []


def test_a_swipe_off_the_frame_edge_does_nothing() -> None:
    h = ready()
    assert events(h.move((0.99, MID[1]), 0.3) + h.hold(0.5)) == []


# -- zoom: pinch and drag ------------------------------------------------------------------------
def test_pinch_and_move_up_or_down_zooms_one_step_per_detent() -> None:
    h = ready()
    grab = pinch("index", 0.05)
    h.hold(0.3, pose=grab)
    assert h.d.engine.phase is Phase.ZOOM
    zoom_in = h.move(at(0.0, -1.0), 0.8, pose=grab) + h.hold(0.3, pose=grab)
    assert events(zoom_in) == ["zoom in"] * 3  # 1.0 palm lengths / 0.3 per step
    # The last detent was at -0.9; down to +0.7 is 1.6 palm lengths: 5 whole detents.
    zoom_out = h.move(at(0.0, 0.7), 1.0, pose=grab) + h.hold(0.3, pose=grab)
    assert events(zoom_out) == ["zoom out"] * 5
    released = h.hold(0.3) + h.move(at(2.5), 0.3) + h.hold(0.5)
    assert events(released) == ["right"]  # after the pinch, swipes work again


def test_moving_while_pinched_never_swipes() -> None:
    h = ready()
    grab = pinch("index", 0.05)
    h.hold(0.3, pose=grab)
    assert events(h.move(at(2.5), 0.3, pose=grab) + h.hold(0.5, pose=grab)) == []
