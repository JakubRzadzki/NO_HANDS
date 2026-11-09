"""Swipe detection shared by --slides and the hand mouse.

A stroke is the fast part of a hand movement, cut out by its speed: it starts when the hand
speeds up and ends as soon as it slows down or turns back. So the swipes people really make
all count: one that goes straight back, one started while the hand was still moving into
place, and a flick of the wrist. (A first version waited for the hand to rest before and
after. In simulated natural swipes it missed every swipe-and-return and every wrist flick.)

A stroke is a swipe if it is fast (peak speed), long enough, mostly sideways and made with an
open hand. Not swipes:

* the way back after a swipe: an opposite stroke right after a swipe is the hand returning;
* raising the hand into view: strokes in the first moments after the hand appears;
* a hand leaving the frame: the stroke ends at the frame edge, or the tracking is lost;
* slow repositioning, and moving a fist.

Positions are the centroid of all 21 landmarks: a flick swings the fingers around the wrist,
so the wrist alone barely moves. Distances are in palm lengths, so the same stroke works close
to the camera and far from it.

``require_return`` is for the hand mouse, where the same hand also moves the cursor: there a
swipe only counts once the hand has come most of the way back. Nobody moves a cursor far,
fast, and straight back, so pointing never changes the slide.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from config import SlidesConfig
from src.smoother import FloatArray

SPEED_LAG = 0.06  # s: speed = displacement over this long (two frames at 30 FPS)
START_LAG = 0.1  # s: a stroke starts this long before its speed was noticed
HISTORY = 1.5  # s of positions kept (speed, the HUD trail)
END_FRACTION = 0.35  # a stroke ends when the speed falls below this fraction of its peak ...
END_SPEED = 0.8  # ... or below this (palm lengths/s)
TURN_MIN = 0.3  # palm lengths a stroke must cover before turning back can end it
BOUNCE = 0.15  # s after a swipe in which no stroke at all can fire
EDGE = 0.06  # strokes ending this close to the frame edge they head for are the hand leaving
OPPOSITE = {"right": "left", "left": "right"}


@dataclass(frozen=True)
class _Sample:
    t: float
    p: FloatArray  # centroid of the hand, px
    palm: float  # palm length, px
    open: bool  # enough fingers are not curled


@dataclass
class _Moving:
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


@dataclass(frozen=True)
class _Pending:
    """A swipe waiting for the hand to come back (hand mouse)."""

    direction: str
    far_x: float  # px
    length: float  # px, sideways
    ended: float  # time


class SwipeDetector:
    """Feed it the hand every frame; ``update`` returns "left" / "right" when a swipe fires."""

    def __init__(self, cfg: SlidesConfig, require_return: bool = False) -> None:
        self.cfg = cfg
        self.require_return = require_return
        self._history: deque[_Sample] = deque()
        self._moving: _Moving | None = None
        self._pending: _Pending | None = None
        self._seen_since: float | None = None
        self._last_fire: tuple[str, float] | None = None  # direction, time
        self.last_stroke: Stroke | None = None  # for the HUD and the tests

    # -- state for the HUD -------------------------------------------------------------------
    @property
    def moving(self) -> bool:
        return self._moving is not None

    @property
    def ready(self) -> bool:
        """The hand has been in view long enough for strokes to count."""
        if not self._history or self._seen_since is None:
            return False
        return self._history[-1].t - self._seen_since >= self.cfg.arm_time

    @property
    def stroke(self) -> tuple[float, float] | None:
        """The stroke under way, in palm lengths."""
        if self._moving is None or not self._history:
            return None
        d = (self._history[-1].p - self._moving.start.p) / self._palm()
        return float(d[0]), float(d[1])

    @property
    def trail(self) -> list[tuple[float, float]]:
        return [(float(s.p[0]), float(s.p[1])) for s in self._history]

    def reset(self) -> None:
        """The hand is gone (or busy with something else): forget the motion."""
        self._history.clear()
        self._moving = None
        self._pending = None
        self._seen_since = None

    def cancel(self) -> None:
        """Drop the stroke under way but keep watching (e.g. the hand pinched)."""
        self._moving = None
        self._pending = None

    # -- per frame ---------------------------------------------------------------------------
    def update(
        self, t: float, point: FloatArray, palm: float, is_open: bool, frame_width: float
    ) -> str | None:
        if self._seen_since is None:
            self._seen_since = t
        now = _Sample(t, np.asarray(point, dtype=np.float64), palm, is_open)
        self._history.append(now)
        while self._history and self._history[0].t < t - HISTORY:
            self._history.popleft()

        returned = self._returned(now)
        if returned is not None:
            return returned
        finished = self._track(now)
        if finished is None:
            return None
        return self._judge(*finished, now, frame_width)

    def _track(self, now: _Sample) -> tuple[Stroke, _Moving] | None:
        """Stroke segmentation. Returns the stroke that just ended, if one did."""
        speed, velocity = self._velocity(now)
        if self._moving is None:
            if speed > self.cfg.start_speed:
                start = self._sample_before(now.t - START_LAG) or self._history[0]
                self._moving = _Moving(start, speed, start.p.copy())
            return None

        m = self._moving
        m.peak = max(m.peak, speed)
        palm = self._palm()
        distance = float(np.linalg.norm(now.p - m.start.p)) / palm
        if distance > m.far_distance:
            m.far_distance, m.far = distance, now.p.copy()
        heading = (m.far - m.start.p) / palm
        turned_back = m.far_distance > TURN_MIN and float(np.dot(velocity, heading)) < 0
        slowed = speed < max(END_FRACTION * m.peak, END_SPEED)
        if now.t - m.start.t > self.cfg.max_stroke + START_LAG:
            self._moving = None  # too long for a swipe: the hand was being repositioned
            return None
        if not (slowed or turned_back):
            return None
        self._moving = None
        dx, dy = (float(v) for v in (m.far - m.start.p) / palm)
        stroke = Stroke(dx, dy, m.peak, now.t - m.start.t, self._open_before(m.start.t))
        self.last_stroke = stroke
        return stroke, m

    def _judge(self, stroke: Stroke, m: _Moving, now: _Sample, frame_width: float) -> str | None:
        cfg = self.cfg
        if self._seen_since is None or m.start.t - self._seen_since < cfg.arm_time:
            return None  # the hand was still coming into view
        if not stroke.open_start or stroke.peak_speed < cfg.swipe_speed:
            return None
        if abs(stroke.dx) < cfg.swipe_distance or abs(stroke.dx) < cfg.axis_ratio * abs(stroke.dy):
            return None
        direction = "right" if stroke.dx > 0 else "left"
        edge_x = m.far[0] / max(frame_width, 1.0)
        if (direction == "right" and edge_x > 1 - EDGE) or (direction == "left" and edge_x < EDGE):
            return None  # the hand is leaving the frame sideways
        if self._last_fire is not None:
            last, when = self._last_fire
            if m.start.t - when < BOUNCE:
                return None
            if direction == OPPOSITE[last] and m.start.t - when <= cfg.return_window:
                return None  # the hand coming back after the previous swipe
        if self.require_return:
            length = abs(float(m.far[0] - m.start.p[0]))
            self._pending = _Pending(direction, float(m.far[0]), length, now.t)
            return self._returned(now)
        self._last_fire = (direction, now.t)
        return direction

    def _returned(self, now: _Sample) -> str | None:
        """Hand mouse: fire the pending swipe once the hand is most of the way back."""
        pending = self._pending
        if pending is None:
            return None
        if now.t - pending.ended > self.cfg.return_time:
            self._pending = None  # it stayed there: the hand moved the cursor, it did not swipe
            return None
        sign = 1.0 if pending.direction == "right" else -1.0
        back = (pending.far_x - float(now.p[0])) * sign
        if back < self.cfg.return_fraction * pending.length:
            return None
        self._pending = None
        self._last_fire = (pending.direction, now.t)
        return pending.direction

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
