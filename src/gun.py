"""Slides by "finger gun": index out, thumb up, the other fingers folded. The barrel says where to
go: pointing right = next slide (Right arrow), pointing left = previous slide (Left arrow).
Either hand; naturally the right hand points right and the left hand points left.

A pose, not a movement: it does not depend on how fast the hand moves, on motion blur or on the
frame rate, which is what made swipes miss. One slide per gun: the gun has to be held for a
moment (gun_confirm_frames) and lowered again before it fires again.

Why the barrel and not which hand it is: MediaPipe's left / right label is good (0.95+ on photos
of known hands) but reversed in a mirrored preview and sometimes wrong. The barrel's direction
is measured directly.

What tells a gun from plain pointing, measured: a gun's thumb stands away from the index at
~75 degrees, its tip ~1 palm length from the index knuckle; pointing at something keeps the
thumb along the hand (25 degrees, 0.2 palm lengths on a real photo). The barrel must also be
roughly level: pointing up (the slowest cursor) never fires.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from config import GestureConfig
from src.hand_tracker import INDEX_MCP, INDEX_TIP, THUMB_MCP, THUMB_TIP, HandObservation


class FingerExtensions(Protocol):
    """What is needed of HandFeatures (src/gesture_engine.py), without importing it."""

    @property
    def extension(self) -> Mapping[Any, float]: ...


# The other fingers only have to be clearly lower than the index, not folded all the way: a
# finger gun made in passing often has the pinky half out (ratio ~1.0-1.13), and the first
# rule (all folded, < 1.0) read that as the horns, which zoom.
FOLD_MARGIN = 0.2


def gun_check(
    obs: HandObservation, f: FingerExtensions, cfg: GestureConfig
) -> tuple[str | None, str]:
    """(which way a finger gun points, or None; why, in a few words for the preview)."""
    e = {finger.name: ratio for finger, ratio in f.extension.items()}
    index = e["INDEX"]
    if index <= cfg.finger_extended_above:
        return None, f"index not straight ({index:.2f})"
    others = max(e["MIDDLE"], e["RING"], e["PINKY"])
    if others >= index - FOLD_MARGIN or max(e["MIDDLE"], e["RING"]) >= cfg.finger_extended_above:
        return None, f"fold middle, ring, pinky ({others:.2f})"
    barrel = obs.px[INDEX_TIP, :2] - obs.px[INDEX_MCP, :2]
    thumb = obs.px[THUMB_TIP, :2] - obs.px[THUMB_MCP, :2]
    tilt = math.degrees(math.atan2(abs(barrel[1]), abs(barrel[0])))
    if tilt > cfg.gun_max_tilt:
        return None, f"point to the side (index {tilt:.0f} deg from level)"
    norm = float(np.linalg.norm(barrel) * np.linalg.norm(thumb)) or 1.0
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(barrel, thumb)) / norm))))
    if angle < cfg.gun_thumb_angle:
        return None, f"thumb up ({angle:.0f} deg from the index, need {cfg.gun_thumb_angle:.0f})"
    reach = float(np.linalg.norm(obs.px[THUMB_TIP, :2] - obs.px[INDEX_MCP, :2])) / obs.palm_scale
    if reach < cfg.gun_thumb_out:
        return None, f"thumb further out ({reach:.2f}, need {cfg.gun_thumb_out:.2f})"
    direction = "right" if barrel[0] > 0 else "left"
    return direction, f"GUN {direction}"


def gun_direction(obs: HandObservation, f: FingerExtensions, cfg: GestureConfig) -> str | None:
    """Which way a finger gun points: "right" / "left" (in the mirrored preview the user's right
    is the image's right), or None when the hand is not a gun."""
    return gun_check(obs, f, cfg)[0]


class GunTrigger:
    """One key per gun: fires after the gun is held gun_confirm_frames, re-arms after it is gone
    for gun_release_frames. Keep one per hand."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._direction: str | None = None
        self._held = 0
        self._gone = 0
        self.fired = False

    def update(self, direction: str | None) -> str | None:
        if direction is None:
            self._gone += 1
            if self._gone >= self.cfg.gun_release_frames:
                self._direction, self._held, self.fired = None, 0, False
            return None
        self._gone = 0
        if direction != self._direction:  # a new gun, or turned the other way
            self._direction, self._held, self.fired = direction, 0, False
        self._held += 1
        if not self.fired and self._held >= self.cfg.gun_confirm_frames:
            self.fired = True
            return direction
        return None

    @property
    def aiming(self) -> str | None:
        """The direction of a gun that is being held (fired or not)."""
        return self._direction if self._held else None
