"""Scroll with a "stop" sign: the other hand raised, palm towards the camera, fingers up and
straight - like a traffic police officer's stop. While it is held, moving the cursor hand up or
down scrolls (a joystick: the further from where it started, the faster); lowering it brings the
cursor back.

The stop hand is a modifier key, so scroll needs no pose of the cursor hand at all and clashes
with none of them (the V of the slowest cursor, the finger gun, the pinch). A hand that is
merely in view - on the desk, on the forehead, relaxed - is not a stop sign: all four fingers
must be straight, pointing up, with the palm (or the back of the hand) facing the camera.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol

from config import GestureConfig
from src.hand_tracker import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST, HandObservation
from src.pinch import pinch_distance


class StopFeatures(Protocol):
    """What is needed of HandFeatures (src/gesture_engine.py), without importing it."""

    @property
    def extension(self) -> Mapping[Any, float]: ...

    @property
    def palm_width(self) -> float: ...


def stop_check(obs: HandObservation, f: StopFeatures, cfg: GestureConfig) -> tuple[bool, str]:
    """(is the hand a stop sign; why, in a few words for the preview)."""
    bent = [
        finger.name.lower()
        for finger, ratio in f.extension.items()
        if ratio <= cfg.finger_extended_above
    ]
    if bent:
        return False, "straighten " + ", ".join(bent)
    if pinch_distance(obs, THUMB_TIP, INDEX_TIP, cfg.pinch_depth_weight) < cfg.pinch_exit:
        return False, "thumb off the index (that is a pinch: zoom)"
    up = obs.px[MIDDLE_MCP, :2] - obs.px[WRIST, :2]
    tilt = math.degrees(math.atan2(abs(float(up[0])), -float(up[1])))  # 0 = fingers straight up
    if tilt > cfg.stop_max_tilt:
        return False, f"fingers up ({tilt:.0f} deg from upright)"
    if f.palm_width < cfg.stop_palm_width:
        return False, f"palm to the camera ({f.palm_width:.2f})"
    return True, "STOP"
