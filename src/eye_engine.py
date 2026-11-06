"""--eyes: hands-free mouse. The cursor follows the gaze (or the nose), eye gestures click.

    look at a point ............ cursor jumps there and stays (gaze) / follows the nose
    small head turn ............ nudges the cursor the last bit (gaze pointer)
    deliberate blink 0.35-1 s .. left click; a second one within 1.2 s = double click
    right wink ................. right click
    left wink .................. scroll mode on/off: look above / below the screen centre
    eyes closed >= 1.5 s ....... pause / resume (the nose pointer also re-centres)

Natural blinks last 0.1-0.3 s and are ignored. A gesture is judged when the eyes open again,
and the cursor stays frozen while they are closing and briefly after, so a click lands where
you looked before blinking.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from config import EyeConfig, FaceConfig, FilterConfig
from src.face_tracker import NOSE_TIP, EyeMeasure, FaceObservation, gaze_features, measure_eyes
from src.gaze import FixationFilter, GazeModel
from src.gesture_engine import HysteresisLatch
from src.os_controller import Action, Click, MoveTo, Scroll
from src.pointer import CursorMapper, Roi
from src.smoother import OneEuroFilter

REOPEN_SETTLE = 0.15  # s after the eyes open before the gaze is trusted again
NOSE_NEUTRAL_FRAMES = 10  # frames that define the nose pointer's neutral position
RADIUS_SIGMAS = 3.0  # automatic fixation radius = this many times the calibration jitter
RADIUS_LIMITS = (0.03, 0.20)  # ... kept within these fractions of the screen width


class FaceMode(str, Enum):
    NO_FACE = "NO FACE"
    EYES = "EYES"
    EYE_SCROLL = "EYE SCROLL"
    HEAD = "HEAD"
    PAUSED = "PAUSED"


@dataclass
class FaceOutput:
    mode: FaceMode
    label: str
    actions: list[Action] = field(default_factory=list)
    eyes: tuple[EyeMeasure, EyeMeasure] | None = None  # user's left, right
    openness: tuple[float, float] = (1.0, 1.0)
    closed: tuple[bool, bool] = (False, False)
    pointer: tuple[float, float] | None = None  # screen px the gaze / nose points at
    feature: tuple[float, float] | None = None  # iris offset (--eyes) or head offset (--head)
    nudge: tuple[float, float] | None = None  # px the head adds to the gaze point (--eyes)
    gaze_raw: tuple[float, float] | None = None  # px, this frame's gaze estimate (unfiltered)
    fixation: tuple[float, float] | None = None  # px, the fixation the cursor sits on
    radius: float = 0.0  # px, gaze moves smaller than this do not jump the cursor


class _OpenBaseline:
    """An eye's "fully open" aspect ratio: 80th percentile of recent not-closed frames."""

    def __init__(self, size: int = 150) -> None:
        self._values: deque[float] = deque(maxlen=size)

    def openness(self, aspect: float) -> float:
        if not self._values:
            return 1.0
        return aspect / max(float(np.percentile(self._values, 80)), 1e-6)

    def add(self, aspect: float) -> None:
        self._values.append(aspect)


class ClosureDetector:
    """Per-eye closed state (with hysteresis) grouped into closure episodes.

    ``update`` returns ``(kind, seconds)`` when the eyes open again after a closure, where kind
    is "both", "left" or "right" (the user's eye that was closed for most of the episode).
    """

    def __init__(self, cfg: FaceConfig) -> None:
        self.cfg = cfg
        self._baselines = (_OpenBaseline(), _OpenBaseline())
        self._latches = (
            HysteresisLatch(cfg.closed_below, cfg.open_above),
            HysteresisLatch(cfg.closed_below, cfg.open_above),
        )
        self._start: float | None = None
        self._counts: dict[str, int] = {}
        self.openness = (1.0, 1.0)
        self.closed = (False, False)

    @property
    def fully_open(self) -> bool:
        return min(self.openness) >= self.cfg.open_above

    def cancel(self) -> None:
        """Forget the running episode (e.g. the face was lost: it cannot be judged)."""
        self._start, self._counts = None, {}

    def update(self, left: EyeMeasure, right: EyeMeasure, t: float) -> tuple[str, float] | None:
        eyes = (left, right)
        self.openness = (
            self._baselines[0].openness(left.aspect),
            self._baselines[1].openness(right.aspect),
        )
        self.closed = (
            self._latches[0].update(self.openness[0]),
            self._latches[1].update(self.openness[1]),
        )
        for baseline, eye, closed in zip(self._baselines, eyes, self.closed, strict=True):
            if not closed:
                baseline.add(eye.aspect)

        if any(self.closed):
            kind = "both" if all(self.closed) else ("left" if self.closed[0] else "right")
            if self._start is None:
                self._start = t
            self._counts[kind] = self._counts.get(kind, 0) + 1
            return None
        if self._start is None:
            return None
        kind = max(self._counts, key=lambda k: self._counts[k])
        duration = t - self._start
        self.cancel()
        return kind, duration


class HeadNudge:
    """MAGIC pointing (Zhai et al. 1999): the eyes jump the cursor close to the target, a small
    head turn moves it the rest of the way.

    The offset is the head turn since the last gaze jump, in px. Right after a jump the head is
    often still finishing its own part of the gaze shift, so for ``SETTLE`` seconds the
    reference follows the head instead of moving the cursor. Turns within ``deadzone`` degrees
    do nothing: a head never holds still, and without the dead zone its drift moved the cursor
    away from where the eyes were looking (in simulation +36% error for a fidgety head)."""

    SETTLE = 0.4  # s

    def __init__(
        self,
        gain_px_per_deg: float,
        max_px: float,
        filter_cfg: FilterConfig,
        deadzone_deg: float = 0.0,
    ) -> None:
        self.gain = gain_px_per_deg
        self.max = max_px
        self.deadzone = deadzone_deg
        self._filter = OneEuroFilter(filter_cfg.min_cutoff, filter_cfg.beta, filter_cfg.d_cutoff)
        self._ref: np.ndarray | None = None
        self._settle_until = -math.inf

    def restart(self, t: float) -> None:
        self._ref = None
        self._settle_until = t + self.SETTLE
        self._filter.reset()

    def update(self, head: tuple[float, float], t: float) -> tuple[float, float]:
        if self.gain <= 0:
            return 0.0, 0.0
        angles = np.asarray(head, dtype=np.float64)
        if self._ref is None or t < self._settle_until:
            self._ref = angles
        turn = angles - self._ref
        beyond = np.sign(turn) * np.maximum(np.abs(turn) - self.deadzone, 0.0)
        offset = np.clip(self.gain * beyond, -self.max, self.max)
        x, y = self._filter(offset, t)
        return float(x), float(y)


class EyeEngine:
    def __init__(
        self,
        cfg: EyeConfig,
        face: FaceConfig,
        screen_size: tuple[int, int],
        filter_cfg: FilterConfig,
        model: GazeModel | None = None,
        mirrored: bool = True,
    ) -> None:
        if cfg.pointer == "gaze" and model is None:
            raise ValueError("the gaze pointer needs a calibration")
        self.cfg = cfg
        self.face = face
        self.screen_w, self.screen_h = screen_size
        self.model = model
        self.mirrored = mirrored
        self._filter_cfg = filter_cfg
        self._closure = ClosureDetector(face)
        self._fixation = FixationFilter(
            self._fixation_radius(), cfg.saccade_frames, cfg.fixation_settle, cfg.fixation_dwell
        )
        self._nudge = HeadNudge(
            cfg.head_nudge * self.screen_w,
            cfg.head_nudge_max * self.screen_w,
            filter_cfg,
            cfg.head_nudge_deadzone,
        )
        self._gaze_raw: tuple[float, float] | None = None
        self._nudge_offset: tuple[float, float] | None = None
        self._nose_samples: list[tuple[float, float]] = []
        self._nose_mapper: CursorMapper | None = None
        self._paused = False
        self._scroll = False
        self._scroll_accum = 0.0
        self._last_blink = -math.inf
        self._settle_until = -math.inf
        self._last_pos: tuple[int, int] | None = None
        self._last_t: float | None = None
        self._flash = ""
        self._flash_until = -math.inf

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def scrolling(self) -> bool:
        return self._scroll

    @property
    def fixation_radius(self) -> float:
        return self._fixation.radius

    def _fixation_radius(self) -> float:
        """px. Automatic: 3x the jitter measured at calibration - at that distance noise alone
        practically never looks like a jump, and a real jump of that size is still easy."""
        w = self.screen_w
        if self.cfg.fixation_radius > 0:
            return self.cfg.fixation_radius * w
        noise = self.model.noise if self.model is not None else 0.0
        return min(max(RADIUS_SIGMAS * noise, RADIUS_LIMITS[0] * w), RADIUS_LIMITS[1] * w)

    def reset(self) -> None:
        """Forget motion state (global pause, recalibration). Pause/scroll modes are kept."""
        self._closure.cancel()
        self._fixation.reset()
        self._nose_samples.clear()
        self._nose_mapper = None
        self._last_pos = None
        self._last_t = None

    def set_model(self, model: GazeModel) -> None:
        self.model = model
        self._fixation.radius = self._fixation_radius()
        self.reset()

    # -- per frame -----------------------------------------------------------------------
    def update(self, obs: FaceObservation | None, t: float) -> FaceOutput:
        if obs is None:
            self._closure.cancel()  # a closure cannot be judged without seeing it end
            mode = FaceMode.PAUSED if self._paused else FaceMode.NO_FACE
            return FaceOutput(mode, self._label(t, mode))
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 0.1)
        self._last_t = t

        left, right = measure_eyes(obs, self.mirrored)
        episode = self._closure.update(left, right, t)
        actions = self._gesture(*episode, t) if episode is not None else []

        if not self._closure.fully_open:
            self._settle_until = t + REOPEN_SETTLE  # eyelids moving: iris landmarks unreliable
        pointer: tuple[float, float] | None = None
        features = gaze_features(obs, left, right, self.mirrored)
        if t >= self._settle_until:
            pointer = self._pointer(obs, features, t)
        if pointer is not None and not self._paused:
            if self._scroll:
                actions += self._scroll_actions(pointer[1], dt)
            else:
                move = self._move(pointer)
                if move is not None:
                    actions.append(move)

        mode = (
            FaceMode.PAUSED
            if self._paused
            else FaceMode.EYE_SCROLL
            if self._scroll
            else FaceMode.EYES
        )
        gaze = self.cfg.pointer == "gaze"
        return FaceOutput(
            mode,
            self._label(t, mode),
            actions,
            (left, right),
            self._closure.openness,
            self._closure.closed,
            pointer,
            (features[0], features[1]),
            self._nudge_offset if gaze else None,
            self._gaze_raw if gaze else None,
            self._fixation.center if gaze else None,
            self._fixation.radius if gaze else 0.0,
        )

    # -- gestures --------------------------------------------------------------------------
    def _gesture(self, kind: str, duration: float, t: float) -> list[Action]:
        cfg = self.cfg
        if kind == "both":
            if duration >= self.face.pause_hold:
                self._toggle_pause(t)
                return []
            if self._paused or not cfg.blink_min <= duration <= cfg.blink_max:
                return []  # natural blink, or a closure of ambiguous length
            if t - self._last_blink <= cfg.double_click_window:
                # The first blink already clicked; two more fast clicks make the OS see a double
                # click even though two deliberate blinks take longer than its 0.5 s window.
                self._last_blink = -math.inf
                self._set_flash("DOUBLE CLICK", t)
                return [Click("left", 2)]
            self._last_blink = t
            self._set_flash("LEFT CLICK", t)
            return [Click("left")]
        if self._paused or duration < cfg.wink_min:
            return []
        right_click_eye = "left" if cfg.swap_winks else "right"
        if kind == right_click_eye:
            self._set_flash("RIGHT CLICK", t)
            return [Click("right")]
        self._scroll = not self._scroll
        self._scroll_accum = 0.0
        self._set_flash("SCROLL ON" if self._scroll else "SCROLL OFF", t)
        return []

    def _toggle_pause(self, t: float) -> None:
        self._paused = not self._paused
        self._scroll = False
        self._fixation.reset()
        if not self._paused:  # nose pointer: wherever the head is now becomes the centre
            self._nose_samples.clear()
            self._nose_mapper = None
        self._set_flash("PAUSED" if self._paused else "RESUMED", t, 1.0)

    # -- pointer ---------------------------------------------------------------------------
    def _pointer(
        self, obs: FaceObservation, features: tuple[float, ...], t: float
    ) -> tuple[float, float] | None:
        if self.cfg.pointer == "gaze":
            assert self.model is not None
            gaze = self.model.predict(features, obs.head_position[2])
            self._gaze_raw = gaze
            x, y = self._fixation.update(gaze, t)
            if self._fixation.jumped:
                self._nudge.restart(t)
            dx, dy = self._nudge.update((obs.head[0], obs.head[1]), t)
            self._nudge_offset = (dx, dy)
            return x + dx, y + dy
        nose = (float(obs.norm[NOSE_TIP, 0]), float(obs.norm[NOSE_TIP, 1]))
        if self._nose_mapper is None:
            self._nose_samples.append(nose)
            if len(self._nose_samples) < NOSE_NEUTRAL_FRAMES:
                return None
            cx, cy = np.median(np.asarray(self._nose_samples), axis=0)
            screen = (self.screen_w, self.screen_h)
            roi = Roi.for_screen(obs.frame_size, screen, self.cfg.nose_roi_width, (cx, cy))
            self._nose_mapper = CursorMapper(roi, screen, self._filter_cfg)
        self._nose_mapper.update(nose, t)
        target = self._nose_mapper.target
        return None if target is None else (float(target[0]), float(target[1]))

    def _move(self, pointer: tuple[float, float]) -> MoveTo | None:
        x = round(min(max(pointer[0], 0.0), self.screen_w - 1.0))
        y = round(min(max(pointer[1], 0.0), self.screen_h - 1.0))
        if (x, y) == self._last_pos:
            return None
        self._last_pos = (x, y)
        return MoveTo(x, y)

    def _scroll_actions(self, y: float, dt: float) -> list[Action]:
        offset = (y - self.screen_h / 2.0) / self.screen_h  # + = below the centre
        excess = abs(offset) - self.cfg.scroll_deadzone
        rate = 0.0
        if excess > 0:  # looking below the centre scrolls down (negative wheel ticks)
            rate = math.copysign(
                min(self.cfg.scroll_gain * excess, self.cfg.scroll_max_rate), -offset
            )
        self._scroll_accum += rate * dt
        ticks = int(self._scroll_accum)
        self._scroll_accum -= ticks
        return [Scroll(0, ticks)] if ticks else []

    # -- HUD label -------------------------------------------------------------------------
    def _set_flash(self, text: str, t: float, duration: float = 0.5) -> None:
        self._flash, self._flash_until = text, t + duration

    def _label(self, t: float, mode: FaceMode) -> str:
        if t < self._flash_until:
            return self._flash
        if mode is FaceMode.EYES:
            return "EYES" if self.cfg.pointer == "gaze" else "EYES + NOSE"
        return mode.value
