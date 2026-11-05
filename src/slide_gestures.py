"""--slides: quick hand strokes control a presentation.

    quick stroke left / right ...... Left / Right arrow (previous / next slide)
    quick stroke up / down ......... zoom in / out (Ctrl + mouse wheel)

A stroke counts when it starts from a resting hand, is quick, mostly along one axis, is made
with an open hand, and ends with the hand resting again inside the frame. Each rule rules out
something a presenter does all the time:

* bringing the hand back after a swipe: an opposite stroke right after a swipe is a return;
* raising the hand into the frame, or lowering it out: those strokes never start (or never
  end) with the hand resting in view;
* moving a fist or a pointing hand around: not an open hand (a fist is the way to move the
  hand without swiping);
* slowly repositioning the hand: not quick.

Waiting for the hand to stop costs ~0.2 s after the stroke; a wrong slide in front of an
audience costs more. Positions are the wrist (it barely moves when the fingers do), measured in
palm lengths, so the same stroke works close to the camera or far from it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from config import GestureConfig, SlidesConfig
from src.gesture_engine import EngineOutput, Finger, HandFeatures, Mode
from src.hand_tracker import WRIST, HandObservation
from src.os_controller import Action, KeyTap, Zoom

SPEED_LAG = 0.1  # s: speed = displacement over this long (one frame alone is too noisy)
HISTORY = 1.5  # s of wrist positions kept (for the speed and the HUD trail)
MAX_GAP = 0.15  # s: tracking may drop out this long mid-stroke (motion blur) without a reset


class Phase(str, Enum):
    UNSETTLED = "UNSETTLED"  # hand just found, or a stroke was discarded: wait for a rest
    RESTING = "RESTING"
    MOVING = "MOVING"


@dataclass(frozen=True)
class _Sample:
    t: float
    x: float  # wrist, px
    y: float
    palm: float  # palm length, px
    open: bool  # enough fingers extended


@dataclass(frozen=True)
class Stroke:
    """A finished stroke, in palm lengths (+x = right, +y = down, as in the mirrored image)."""

    dx: float
    dy: float
    duration: float
    open_fraction: float


ACTIONS: dict[str, str] = {
    "right": "RIGHT >>",
    "left": "<< LEFT",
    "up": "ZOOM IN",
    "down": "ZOOM OUT",
}
OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}


class SlideGestures:
    def __init__(self, cfg: SlidesConfig, gesture: GestureConfig) -> None:
        self.cfg = cfg
        self.gesture = gesture
        self._history: deque[_Sample] = deque()
        self._phase = Phase.UNSETTLED
        self._rest_since: float | None = None
        self._start: _Sample | None = None  # where the current stroke began
        self._stroke_frames = 0
        self._stroke_open = 0
        self._last_fire: tuple[str, float] | None = None  # direction, time
        self._last_seen: float | None = None
        self._flash = ""
        self._flash_until = -math.inf
        self.last_stroke: Stroke | None = None  # for the HUD and the tests

    @property
    def phase(self) -> Phase:
        return self._phase

    def reset(self) -> list[Action]:
        """Forget motion state (pause, hand lost). Nothing is ever held, so nothing to release."""
        self._history.clear()
        self._phase = Phase.UNSETTLED
        self._rest_since = None
        self._start = None
        self._last_seen = None
        return []

    def update(self, obs: HandObservation | None, t: float) -> EngineOutput:
        if obs is None:
            if self._last_seen is None or t - self._last_seen > MAX_GAP:
                self.reset()
                return EngineOutput(Mode.NO_HAND, self._label(t, "NO HAND"))
            return EngineOutput(Mode.SLIDES, self._label(t, "TRACKING LOST"))
        self._last_seen = t

        features = HandFeatures.from_observation(obs)
        extended = {f: features.extension[f] > self.gesture.finger_extended_above for f in Finger}
        sample = _Sample(
            t,
            float(obs.px[WRIST, 0]),
            float(obs.px[WRIST, 1]),
            obs.palm_scale,
            sum(extended.values()) >= self.cfg.min_fingers,
        )
        self._history.append(sample)
        while self._history and self._history[0].t < t - HISTORY:
            self._history.popleft()

        actions = self._step(sample)
        start = self._start if self._phase is Phase.MOVING else None
        stroke = None if start is None else self._displacement(start, sample)
        return EngineOutput(
            Mode.SLIDES,
            self._label(t, self._phase.value if self._phase is not Phase.RESTING else "READY"),
            actions,
            features=features,
            extended=extended,
            trail=[(s.x, s.y) for s in self._history],
            stroke=stroke,
        )

    # -- stroke segmentation -------------------------------------------------------------------
    def _step(self, now: _Sample) -> list[Action]:
        cfg = self.cfg
        speed = self._speed(now)
        resting = speed is not None and speed < cfg.rest_speed
        if resting:
            self._rest_since = now.t if self._rest_since is None else self._rest_since
        else:
            self._rest_since = None
        rested = self._rest_since is not None and now.t - self._rest_since >= cfg.rest_time

        if self._phase is Phase.UNSETTLED:
            if rested:
                self._phase = Phase.RESTING
            return []
        if self._phase is Phase.RESTING:
            if speed is not None and speed > cfg.move_speed:
                self._phase = Phase.MOVING
                self._start = self._sample_before(now.t - SPEED_LAG) or now
                self._stroke_frames = self._stroke_open = 0
            else:
                return []
        # MOVING
        self._stroke_frames += 1
        self._stroke_open += now.open
        assert self._start is not None
        # The start is SPEED_LAG before the motion was noticed, the end rest_time after it stopped.
        overhead = cfg.rest_time + SPEED_LAG
        if now.t - self._start.t > cfg.max_stroke + overhead:
            self._phase = Phase.UNSETTLED  # too slow for a swipe: the hand was repositioned
            return []
        if not rested:
            return []
        self._phase = Phase.RESTING
        dx, dy = self._displacement(self._start, now)
        stroke = Stroke(
            dx,
            dy,
            now.t - overhead - self._start.t,
            self._stroke_open / max(self._stroke_frames, 1),
        )
        self.last_stroke = stroke
        return self._fire(stroke, self._start.t, now.t)

    def _fire(self, stroke: Stroke, started: float, t: float) -> list[Action]:
        cfg = self.cfg
        if stroke.open_fraction < 0.5:
            return []  # a fist or a pointing hand moved: not a swipe
        direction: str | None = None
        if abs(stroke.dx) >= cfg.swipe_distance and abs(stroke.dx) >= cfg.axis_ratio * abs(
            stroke.dy
        ):
            direction = "right" if stroke.dx > 0 else "left"
        elif abs(stroke.dy) >= cfg.zoom_distance and abs(stroke.dy) >= cfg.axis_ratio * abs(
            stroke.dx
        ):
            direction = "down" if stroke.dy > 0 else "up"
        if direction is None:
            return []
        if self._last_fire is not None:
            last, when = self._last_fire
            if direction == OPPOSITE[last] and started - when <= cfg.return_window:
                return []  # the hand coming back after the previous swipe
        self._last_fire = (direction, t)
        self._flash, self._flash_until = ACTIONS[direction], t + 0.6
        if direction in ("right", "left"):
            return [KeyTap(direction)]
        steps = cfg.zoom_steps if direction == "up" else -cfg.zoom_steps
        return [Zoom(steps)]

    # -- helpers -------------------------------------------------------------------------------
    def _sample_before(self, t: float) -> _Sample | None:
        older = [s for s in self._history if s.t <= t]
        return older[-1] if older else None

    def _speed(self, now: _Sample) -> float | None:
        """Palm lengths per second over the last SPEED_LAG seconds (None: not enough history)."""
        old = self._sample_before(now.t - SPEED_LAG)
        if old is None or now.t - old.t > SPEED_LAG + MAX_GAP:
            return None
        palm = float(np.median([s.palm for s in self._history]))
        return math.hypot(now.x - old.x, now.y - old.y) / palm / (now.t - old.t)

    def _displacement(self, start: _Sample, end: _Sample) -> tuple[float, float]:
        palm = (start.palm + end.palm) / 2.0
        return (end.x - start.x) / palm, (end.y - start.y) / palm

    def _label(self, t: float, default: str) -> str:
        return self._flash if t < self._flash_until else default
