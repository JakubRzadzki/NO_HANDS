"""--slides: quick hand strokes change slides; a pinch and a vertical drag zooms.

    quick stroke left / right .............. Left / Right arrow (previous / next slide)
    pinch thumb + index, move up / down .... zoom in / out (Ctrl + mouse wheel)

A stroke is the fast part of a hand movement, cut out by its speed: it starts when the hand
speeds up and ends as soon as it slows down or turns back, and it fires right then. So the
natural swipes all count: a swipe that goes straight back, a swipe started while the hand was
still moving into place, and a flick of the wrist. (A first version waited for the hand to come
to rest after the stroke. In simulated natural swipes that missed every swipe-and-return and
every wrist flick.)

It counts as a swipe if it is fast (peak speed), long enough, mostly sideways, and made with an
open hand. Everything else a presenter does is ruled out:

* the way back after a swipe: an opposite stroke right after a swipe is the hand returning;
* raising the hand into view: strokes in the first moments after the hand appears are ignored;
* a hand leaving the frame: the stroke ends at the frame edge or the tracking is lost;
* slow repositioning, and moving a fist: not fast, not an open hand.

Vertical strokes do nothing. As zoom gestures they also fired on ordinary up/down "beat"
gestures made while talking, in every simulated case. Zoom therefore needs a pinch, which
nobody makes by accident.

Positions are the centroid of all 21 landmarks: a flick swings the fingers around the wrist,
so the wrist alone barely moves. Distances are in palm lengths, so the same stroke works close
to the camera and far from it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from config import GestureConfig, SlidesConfig
from src.gesture_engine import EngineOutput, Finger, HandFeatures, HysteresisLatch, Mode
from src.hand_tracker import HandObservation
from src.os_controller import Action, KeyTap, Zoom
from src.smoother import FloatArray

SPEED_LAG = 0.06  # s: speed = displacement over this long (two frames at 30 FPS)
START_LAG = 0.1  # s: a stroke starts this long before its speed was noticed
HISTORY = 1.5  # s of positions kept (speed, the HUD trail)
MAX_GAP = 0.15  # s: tracking may drop out this long mid-stroke (motion blur) without a reset
END_FRACTION = 0.35  # a stroke ends when the speed falls below this fraction of its peak ...
END_SPEED = 0.8  # ... or below this (palm lengths/s)
TURN_MIN = 0.3  # palm lengths a stroke must cover before turning back can end it
BOUNCE = 0.15  # s after a swipe in which no stroke at all can fire
EDGE = 0.06  # strokes ending this close to the frame edge they head for are the hand leaving


class Phase(str, Enum):
    WAITING = "WAITING"  # the hand just appeared: strokes do not count yet
    READY = "READY"
    STROKE = "STROKE"  # a fast movement is under way
    ZOOM = "ZOOM"  # pinched: vertical movement zooms


@dataclass(frozen=True)
class _Sample:
    t: float
    p: FloatArray  # centroid of the hand, px
    palm: float  # palm length, px
    open: bool  # enough fingers extended


@dataclass
class _Stroke:
    start: _Sample
    peak: float = 0.0  # palm lengths/s
    far: FloatArray = field(default_factory=lambda: np.zeros(2))  # farthest point from start
    far_distance: float = 0.0  # palm lengths


@dataclass(frozen=True)
class Stroke:
    """A finished stroke, in palm lengths (+x = right, +y = down, as in the mirrored image)."""

    dx: float
    dy: float
    peak_speed: float  # palm lengths/s
    duration: float  # s
    open_start: bool


ACTIONS = {"right": "RIGHT >>", "left": "<< LEFT"}
OPPOSITE = {"right": "left", "left": "right"}


class SlideGestures:
    def __init__(self, cfg: SlidesConfig, gesture: GestureConfig) -> None:
        self.cfg = cfg
        self.gesture = gesture
        self._history: deque[_Sample] = deque()
        self._stroke: _Stroke | None = None
        self._pinch = HysteresisLatch(
            gesture.pinch_enter, gesture.pinch_exit, confirm_frames=gesture.pinch_confirm_frames
        )
        self._zoom_ref: float | None = None  # hand height (px) of the last zoom detent
        self._seen_since: float | None = None
        self._last_seen: float | None = None
        self._last_fire: tuple[str, float] | None = None  # direction, time
        self._frame_width = 1.0  # px
        self._flash = ""
        self._flash_until = -math.inf
        self.last_stroke: Stroke | None = None  # for the HUD and the tests

    @property
    def phase(self) -> Phase:
        if self._zoom_ref is not None:
            return Phase.ZOOM
        if self._stroke is not None:
            return Phase.STROKE
        if not self._history or self._seen_since is None:
            return Phase.WAITING
        if self._history[-1].t - self._seen_since < self.cfg.arm_time:
            return Phase.WAITING
        return Phase.READY

    def reset(self) -> list[Action]:
        """Forget motion state (pause, hand lost). Nothing is ever held, so nothing to release."""
        self._history.clear()
        self._stroke = None
        self._pinch.reset()
        self._zoom_ref = None
        self._seen_since = self._last_seen = None
        return []

    def update(self, obs: HandObservation | None, t: float) -> EngineOutput:
        if obs is None:
            if self._last_seen is None or t - self._last_seen > MAX_GAP:
                self.reset()
                return EngineOutput(Mode.NO_HAND, self._label(t, "NO HAND"))
            return EngineOutput(Mode.SLIDES, self._label(t, "TRACKING LOST"))
        if self._seen_since is None:
            self._seen_since = t
        self._last_seen = t
        self._frame_width = float(obs.frame_size[0])

        features = HandFeatures.from_observation(obs)
        extended = {f: features.extension[f] > self.gesture.finger_extended_above for f in Finger}
        sample = _Sample(
            t,
            obs.px[:, :2].mean(axis=0),
            obs.palm_scale,
            sum(extended.values()) >= self.cfg.min_fingers,
        )
        self._history.append(sample)
        while self._history and self._history[0].t < t - HISTORY:
            self._history.popleft()

        pinched = self._pinch.update(features.pinch[Finger.INDEX])
        if pinched:
            self._stroke = None  # a pinch is for zooming: no swipes meanwhile
            actions = self._zoom(sample)
        else:
            self._zoom_ref = None
            actions = self._strokes(sample)

        stroke = None
        if self._stroke is not None:
            d = (sample.p - self._stroke.start.p) / self._palm()
            stroke = (float(d[0]), float(d[1]))
        phase = self.phase
        return EngineOutput(
            Mode.SLIDES,
            self._label(t, phase.value),
            actions,
            features=features,
            extended=extended,
            trail=[(float(s.p[0]), float(s.p[1])) for s in self._history],
            stroke=stroke,
        )

    # -- swipes ------------------------------------------------------------------------------
    def _strokes(self, now: _Sample) -> list[Action]:
        speed, velocity = self._velocity(now)
        if self._stroke is None:
            if speed > self.cfg.start_speed:
                start = self._sample_before(now.t - START_LAG) or self._history[0]
                self._stroke = _Stroke(start, speed, start.p.copy())
            return []

        s = self._stroke
        s.peak = max(s.peak, speed)
        palm = self._palm()
        distance = float(np.linalg.norm(now.p - s.start.p)) / palm
        if distance > s.far_distance:
            s.far_distance, s.far = distance, now.p.copy()
        heading = (s.far - s.start.p) / palm
        turned_back = s.far_distance > TURN_MIN and float(np.dot(velocity, heading)) < 0
        slowed = speed < max(END_FRACTION * s.peak, END_SPEED)
        if now.t - s.start.t > self.cfg.max_stroke + START_LAG:
            self._stroke = None  # too long for a swipe: the hand was being repositioned
            return []
        if not (slowed or turned_back):
            return []
        self._stroke = None
        dx, dy = (float(v) for v in (s.far - s.start.p) / palm)
        start_open = self._open_before(s.start.t)
        stroke = Stroke(dx, dy, s.peak, now.t - s.start.t, start_open)
        self.last_stroke = stroke
        return self._fire(stroke, s, now)

    def _fire(self, stroke: Stroke, s: _Stroke, now: _Sample) -> list[Action]:
        cfg = self.cfg
        if self._seen_since is None or s.start.t - self._seen_since < cfg.arm_time:
            return []  # the hand was still coming into view
        if not stroke.open_start or stroke.peak_speed < cfg.swipe_speed:
            return []
        horizontal = abs(stroke.dx) >= cfg.swipe_distance
        if not horizontal or abs(stroke.dx) < cfg.axis_ratio * abs(stroke.dy):
            return []
        direction = "right" if stroke.dx > 0 else "left"
        edge_x = s.far[0] / self._frame_width
        if (direction == "right" and edge_x > 1 - EDGE) or (direction == "left" and edge_x < EDGE):
            return []  # the hand is leaving the frame sideways
        if self._last_fire is not None:
            last, when = self._last_fire
            if s.start.t - when < BOUNCE:
                return []
            if direction == OPPOSITE[last] and s.start.t - when <= cfg.return_window:
                return []  # the hand coming back after the previous swipe
        self._last_fire = (direction, now.t)
        self._flash, self._flash_until = ACTIONS[direction], now.t + 0.6
        return [KeyTap(direction)]

    # -- zoom: pinch and move up / down -----------------------------------------------------
    def _zoom(self, now: _Sample) -> list[Action]:
        y = float(now.p[1])
        if self._zoom_ref is None:
            self._zoom_ref = y
            return []
        step = self.cfg.zoom_travel * self._palm()
        steps = int((self._zoom_ref - y) / step)  # + = the hand moved up = zoom in
        if steps == 0:
            return []
        steps = max(-1, min(1, steps))  # one wheel tick per frame at most: no zoom jumps
        self._zoom_ref -= steps * step  # whole detents: noise cannot flicker in and out
        self._flash, self._flash_until = ("ZOOM IN" if steps > 0 else "ZOOM OUT"), now.t + 0.4
        return [Zoom(steps)]

    # -- helpers -------------------------------------------------------------------------------
    def _palm(self) -> float:
        return float(np.median([s.palm for s in self._history]))

    def _sample_before(self, t: float) -> _Sample | None:
        older = [s for s in self._history if s.t <= t]
        return older[-1] if older else None

    def _velocity(self, now: _Sample) -> tuple[float, FloatArray]:
        """Speed and velocity in palm lengths per second over the last SPEED_LAG seconds."""
        old = self._sample_before(now.t - SPEED_LAG)
        if old is None or now.t <= old.t:
            return 0.0, np.zeros(2)
        velocity: FloatArray = (now.p - old.p) / self._palm() / (now.t - old.t)
        return float(np.linalg.norm(velocity)), velocity

    def _open_before(self, t: float) -> bool:
        """Was the hand open just before the stroke started (blur makes later frames unreliable)?"""
        before = [s.open for s in self._history if t - 0.2 <= s.t <= t]
        return bool(before) and sum(before) * 2 >= len(before)

    def _label(self, t: float, default: str) -> str:
        return self._flash if t < self._flash_until else default
