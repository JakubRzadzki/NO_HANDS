"""Slides by finger gun (src/gun.py): the barrel's direction picks the slide, either hand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import AppConfig
from src.gesture_engine import HandFeatures, Mode
from src.gun import gun_direction
from src.hand_tracker import HandObservation
from src.os_controller import Action, KeyTap, MoveTo
from src.slide_gestures import SlideGestures
from tests.handgen import CENTER, OPEN, THUMB_CMC, Driver, make_engine, make_obs

FOLDED = {"middle": 1.0, "ring": 1.0, "pinky": 1.0}
GUN_THUMB = THUMB_CMC + np.array([0.075, 0.005, -0.01])  # standing away from the index
GUN_RIGHT: dict[str, Any] = {"roll_deg": -90.0, "curl": FOLDED, "thumb_tip": GUN_THUMB}
GUN_LEFT: dict[str, Any] = {"roll_deg": 90.0, "curl": FOLDED, "thumb_tip": GUN_THUMB}
POINT_SIDEWAYS: dict[str, Any] = {"roll_deg": -90.0, "curl": FOLDED, "thumb": "tucked"}
POINT_UP_THUMB_OUT: dict[str, Any] = {"curl": FOLDED, "thumb_tip": GUN_THUMB}


def keys(actions: list[Action]) -> list[str]:
    return [a.key for a in actions if isinstance(a, KeyTap)]


def settled(**overrides: Any) -> Driver:
    d = Driver(make_engine(**overrides), fps=30.0)
    d.hold(OPEN, 0.5)
    return d


def aim(pose: dict[str, Any]) -> str | None:
    obs = make_obs(0.0, **pose)
    return gun_direction(obs, HandFeatures.from_observation(obs), AppConfig().gesture)


def test_the_barrel_says_where_to_go() -> None:
    assert aim(GUN_RIGHT) == "right"
    assert aim(GUN_LEFT) == "left"
    assert aim(POINT_SIDEWAYS) is None  # thumb along the hand: pointing at something
    assert aim(POINT_UP_THUMB_OUT) is None  # barrel up: the slowest cursor, not a gun
    assert aim(OPEN) is None


def test_no_real_hand_is_taken_for_a_gun() -> None:
    """Fists, open hands, V signs, pinches and horns from real photos (MediaPipe output)."""
    hands = json.loads((Path(__file__).parent / "data" / "real_hands.json").read_text("utf-8"))
    cfg = AppConfig().gesture
    for group in ("fist", "open", "v_sign", "pinch", "horns"):
        for hand in hands[group]:
            obs = HandObservation.from_normalized(hand["norm"], tuple(hand["frame"]), 0.0)
            assert gun_direction(obs, HandFeatures.from_observation(obs), cfg) is None, hand


def test_one_slide_per_gun_and_the_cursor_holds_still() -> None:
    d = settled()
    held = d.hold(GUN_RIGHT, 2.0)
    assert keys(held) == ["right"]  # held for 2 s: still one slide
    assert d.engine.mode is Mode.GUN and not any(isinstance(a, MoveTo) for a in held[10:])
    again = d.hold(OPEN, 0.3) + d.hold(GUN_RIGHT, 0.5)  # lowered, then a new gun
    assert keys(again) == ["right"]
    assert keys(d.hold(OPEN, 0.3) + d.hold(GUN_LEFT, 0.5)) == ["left"]


def test_a_gun_for_a_moment_does_nothing() -> None:
    d = settled()
    assert keys(d.hold(GUN_RIGHT, 3 / 30) + d.hold(OPEN, 0.5)) == []  # 3 frames < 6


def test_pointing_never_changes_the_slide() -> None:
    d = settled()
    assert keys(d.hold(POINT_SIDEWAYS, 1.0) + d.hold(POINT_UP_THUMB_OUT, 1.0)) == []


def test_the_other_hand_fires_while_this_hand_points() -> None:
    d = settled()
    other = {"center": (0.25, 0.55), **GUN_LEFT}
    actions = d.sweep(OPEN, CENTER, (0.6, 0.55), 0.6, second=other)
    assert keys(actions) == ["left"]
    assert sum(isinstance(a, MoveTo) for a in actions) > 5  # the cursor moved meanwhile


def test_guns_can_be_switched_off() -> None:
    d = settled(gun_slides=False)
    assert keys(d.hold(GUN_RIGHT, 1.0)) == []


def test_guns_in_slides_mode_from_either_hand() -> None:
    cfg = AppConfig()
    d = Driver(SlideGestures(cfg.slides, cfg.gesture), fps=30.0)
    d.hold(OPEN, 0.6)
    assert keys(d.hold(GUN_RIGHT, 0.5) + d.hold(OPEN, 0.3)) == ["right"]
    other = {"center": (0.25, 0.55), **GUN_LEFT}
    assert keys(d.hold(OPEN, 0.5, second=other) + d.hold(OPEN, 0.3)) == ["left"]
