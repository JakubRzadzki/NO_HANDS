"""Zoom with two hands, like two fingers on a touchscreen.

Pinch thumb + index on both hands, then pull the hands apart (zoom in) or bring them together
(zoom out). One zoom step per ``two_hand_zoom_step`` change of the distance between the two
pinches (12% by default), measured as a ratio: the same gesture works close to the camera and
far from it, and a small hand tremor never adds up to a step.

The second hand matters only while it pinches. A hand that is merely in view - resting on the
desk, scratching a nose, the presenter's other hand on a remote - changes nothing. (A first
version switched modes as soon as two hands were in view; on a test video of a woman with one
hand on her forehead that blocked every swipe.)

Why it helps: the one-hand zoom needs a pose (the horns, or a pinch held while moving up and
down) that is easy to lose halfway. Two pinching hands are hard to make by accident and
impossible to confuse with anything else, and the zoom amount is how far apart the hands are.
"""

from __future__ import annotations

import math

import numpy as np

from config import GestureConfig
from src.hand_tracker import INDEX_TIP, THUMB_TIP, HandObservation
from src.os_controller import Zoom
from src.pinch import HysteresisLatch, pinch_distance

Point = tuple[float, float]


def pinch_point(obs: HandObservation) -> Point:
    """Where thumb and index meet, px."""
    p = (obs.px[THUMB_TIP, :2] + obs.px[INDEX_TIP, :2]) / 2.0
    return float(p[0]), float(p[1])


class TwoHandZoom:
    """Per frame: ``second_pinched(other hand)`` says whether the two-hand zoom is on; while it is,
    ``update(cursor hand)`` returns the zoom steps."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._latches = [
            HysteresisLatch(
                cfg.pinch_enter,
                cfg.pinch_exit,
                confirm_frames=cfg.pinch_confirm_frames,
                release_frames=cfg.pinch_release_frames,
            )
            for _ in range(2)
        ]
        self._missing = 0  # frames in a row without the other hand
        self._last_second: HandObservation | None = None
        self._ref: float | None = None  # hands' distance (palm lengths) at the last step
        self.points: tuple[Point, Point] | None = None  # the two pinches while both are held

    @property
    def pinched(self) -> tuple[bool, bool]:
        return self._latches[0].active, self._latches[1].active

    def reset(self) -> None:
        for latch in self._latches:
            latch.reset()
        self._missing = 0
        self._last_second = None
        self._ref = None
        self.points = None

    def second_pinched(self, second: HandObservation | None) -> bool:
        """Feed the other hand (None when it is not seen). True while it pinches; a few frames
        without it (``two_hand_grace_frames``) do not end the zoom."""
        latch = self._latches[1]
        if second is None:
            self._missing += 1
            if self._missing > self.cfg.two_hand_grace_frames:
                self.reset()
            return latch.active
        self._missing = 0
        self._last_second = second
        w = self.cfg.pinch_depth_weight
        latch.update(pinch_distance(second, THUMB_TIP, INDEX_TIP, w))
        if not latch.active:
            self._latches[0].reset()
            self._ref, self.points = None, None
        return latch.active

    def update(self, a: HandObservation) -> list[Zoom]:
        """The cursor hand, after ``second_pinched`` said yes for this frame."""
        b = self._last_second
        w = self.cfg.pinch_depth_weight
        self._latches[0].update(pinch_distance(a, THUMB_TIP, INDEX_TIP, w))
        if b is None or not all(self.pinched):
            self._ref, self.points = None, None
            return []
        if self._missing:  # the other hand is out of sight this frame: hold still
            return []
        pa, pb = pinch_point(a), pinch_point(b)
        self.points = (pa, pb)
        palm = (a.palm_scale + b.palm_scale) / 2.0
        distance = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) / max(palm, 1e-6)
        if self._ref is None:
            self._ref = max(distance, 1e-3)
            return []
        step = math.log1p(self.cfg.two_hand_zoom_step)
        steps = int(math.log(max(distance, 1e-3) / self._ref) / step)  # + = apart = zoom in
        if steps == 0:
            return []
        steps = max(-1, min(1, steps))  # one wheel tick per frame at most: no zoom jumps
        self._ref *= math.exp(steps * step)  # whole detents: noise cannot flicker in and out
        return [Zoom(steps)]
