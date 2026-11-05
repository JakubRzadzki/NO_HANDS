"""Gesture recognition: hand landmarks -> mouse/keyboard actions.

Per frame:

1. Features - dimensionless measurements: pinch ratios, finger-extension ratios.
2. Posture  - which mode the hand asks for (POINTER / SCROLL / ZOOM / CLUTCH), debounced over a
              few frames so one misdetected frame cannot switch modes.
3. Mode     - POINTER runs the pinch latches (click, drag, right click, double click);
              SCROLL and ZOOM turn hand motion into wheel ticks; CLUTCH does nothing.
4. Cursor   - the CursorMapper turns the anchor (knuckles, not fingertips) into a move unless
              the engine freezes it (pinch approaching or pending, posture change pending).

The engine never touches the OS: it returns action objects. That keeps the whole state machine
deterministic and unit-testable with synthetic landmarks (see tests/handgen.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from config import GestureConfig, MediaConfig
from src.hand_tracker import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    HandObservation,
)
from src.media_gestures import MediaGestures
from src.os_controller import Action, Click, Press, Release, Scroll, Zoom
from src.pointer import CursorMapper
from src.smoother import FloatArray, OneEuroFilter


class Mode(str, Enum):
    NO_HAND = "NO HAND"
    POINTER = "POINTER"
    SCROLL = "SCROLL"
    ZOOM = "ZOOM"
    CLUTCH = "CLUTCH"
    SWITCH = "SWITCH"  # shaka held: toggles between mouse and media mode
    MEDIA = "MEDIA"  # music control instead of the mouse


class LeftState(str, Enum):
    IDLE = "IDLE"
    PENDING = "PENDING"  # pinched, not yet decided between click and drag
    DRAG = "DRAG"  # button is held down


class Finger(Enum):
    INDEX = (INDEX_PIP, INDEX_TIP)
    MIDDLE = (MIDDLE_PIP, MIDDLE_TIP)
    RING = (RING_PIP, RING_TIP)
    PINKY = (PINKY_PIP, PINKY_TIP)

    @property
    def pip(self) -> int:
        return self.value[0]

    @property
    def tip(self) -> int:
        return self.value[1]


# Thumb pinches: INDEX = left click / drag, MIDDLE = right click, RING = double click.
PINCH_FINGERS = (Finger.INDEX, Finger.MIDDLE, Finger.RING)


class HysteresisLatch:
    """Boolean latch with separate engage/release thresholds plus frame debouncing.

    low_active=True:  engages when value < engage_at, releases when value > release_at.
    low_active=False: engages when value > engage_at, releases when value < release_at.

    Two thresholds stop a signal that hovers around one boundary from toggling every frame (the
    classic "machine-gun click"); ``confirm_frames`` additionally ignores one-frame outliers.
    """

    def __init__(
        self,
        engage_at: float,
        release_at: float,
        low_active: bool = True,
        confirm_frames: int = 1,
    ) -> None:
        if (low_active and engage_at >= release_at) or (not low_active and engage_at <= release_at):
            raise ValueError("the release threshold must lie on the inactive side of engage_at")
        self.engage_at = engage_at
        self.release_at = release_at
        self.low_active = low_active
        self.confirm_frames = max(1, confirm_frames)
        self.active = False
        self._streak = 0

    def update(self, value: float) -> bool:
        if self.active:
            flip = value > self.release_at if self.low_active else value < self.release_at
        else:
            flip = value < self.engage_at if self.low_active else value > self.engage_at
        if flip:
            self._streak += 1
            if self._streak >= self.confirm_frames:
                self.active = not self.active
                self._streak = 0
        else:
            self._streak = 0
        return self.active

    def reset(self) -> None:
        self.active = False
        self._streak = 0


@dataclass(frozen=True)
class HandFeatures:
    palm_scale: float  # px
    pinch: dict[Finger, float]  # thumb tip <-> finger tip, palm lengths
    extension: dict[Finger, float]  # dist(wrist, tip) / dist(wrist, pip)
    index_middle_gap: float  # index tip <-> middle tip, palm lengths
    anchor: tuple[float, float]  # cursor anchor, normalized frame coordinates
    scroll_point: tuple[float, float]  # px, midpoint of the index and middle tips

    @classmethod
    def from_observation(
        cls, obs: HandObservation, anchor_index_weight: float = 0.5
    ) -> HandFeatures:
        pinch = {finger: obs.ndist(THUMB_TIP, finger.tip) for finger in PINCH_FINGERS}
        # A straight finger puts its tip much farther from the wrist than its PIP joint (~1.3x+),
        # a curled one folds the tip back towards the palm (<0.9x). Ratios of distances do not
        # care how the hand is rotated in the image, unlike the common `tip.y < pip.y` test,
        # which silently breaks as soon as the hand tilts sideways.
        extension = {
            finger: obs.dist(WRIST, finger.tip) / max(obs.dist(WRIST, finger.pip), 1e-6)
            for finger in Finger
        }
        w = anchor_index_weight
        anchor = w * obs.norm[INDEX_MCP, :2] + (1.0 - w) * obs.norm[MIDDLE_MCP, :2]
        scroll_point = (obs.px[INDEX_TIP, :2] + obs.px[MIDDLE_TIP, :2]) / 2.0
        return cls(
            palm_scale=obs.palm_scale,
            pinch=pinch,
            extension=extension,
            index_middle_gap=obs.ndist(INDEX_TIP, MIDDLE_TIP),
            anchor=(float(anchor[0]), float(anchor[1])),
            scroll_point=(float(scroll_point[0]), float(scroll_point[1])),
        )


@dataclass
class EngineOutput:
    mode: Mode
    label: str
    actions: list[Action] = field(default_factory=list)
    cursor_frozen: bool = False
    left_state: LeftState = LeftState.IDLE
    features: HandFeatures | None = None
    extended: dict[Finger, bool] = field(default_factory=dict)
    pending_mode: Mode | None = None  # posture change waiting for confirmation
    scroll_origin: tuple[float, float] | None = None  # px, for the HUD
    scroll_rate: tuple[float, float] = (0.0, 0.0)  # ticks/s, for the HUD
    media_active: bool = False  # media mode instead of mouse mode
    knob_offset: float | None = None  # degrees from neutral, for the HUD
    volume_rate: float = 0.0  # volume key presses per second, for the HUD


class GestureEngine:
    def __init__(
        self,
        cfg: GestureConfig,
        mapper: CursorMapper,
        anchor_index_weight: float = 0.5,
        media: MediaConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.mapper = mapper
        self._anchor_weight = anchor_index_weight
        self.media_cfg = media or MediaConfig()
        self._media = MediaGestures(self.media_cfg)
        self._media_active = False  # survives tracking loss: lowering the hand is normal here
        self._switch_since: float | None = None
        self._switch_done = False
        self._finger = {
            finger: HysteresisLatch(
                cfg.finger_extended_above, cfg.finger_curled_below, low_active=False
            )
            for finger in Finger
        }
        self._joined = HysteresisLatch(cfg.scroll_join_enter, cfg.scroll_join_exit)
        self._pinch = {
            finger: HysteresisLatch(
                cfg.pinch_enter, cfg.pinch_exit, confirm_frames=cfg.pinch_confirm_frames
            )
            for finger in PINCH_FINGERS
        }
        self._zoom_filter = OneEuroFilter(min_cutoff=3.0, beta=0.0)  # plain ~3 Hz low-pass
        # A pinch may only engage after its fingers were seen apart in POINTER mode. Otherwise
        # leaving ZOOM with thumb and index together, or the hand re-appearing mid-pinch, would
        # fire a click nobody asked for. (Edge-triggered, not level-triggered.)
        self._armed = dict.fromkeys(PINCH_FINGERS, False)

        self._mode = Mode.NO_HAND
        self._candidate: Mode | None = None
        self._candidate_frames = 0
        self._left = LeftState.IDLE
        self._left_t0 = 0.0
        self._pending_origin: FloatArray | None = None
        self._cursor_frozen = False
        self._last_right = -math.inf
        self._last_double = -math.inf
        self._scroll_origin: tuple[float, float] | None = None
        self._scroll_accum = np.zeros(2)
        self._scroll_rate = np.zeros(2)
        self._zoom_ref: float | None = None
        self._last_seen: float | None = None
        self._last_t: float | None = None
        self._flash = ""
        self._flash_until = -math.inf

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def left_state(self) -> LeftState:
        return self._left

    @property
    def media_active(self) -> bool:
        return self._media_active

    # -- public API --------------------------------------------------------------------------
    def reset(self) -> list[Action]:
        """Forget all gesture state. Returns the actions needed to release anything held."""
        actions: list[Action] = [Release("left")] if self._left is LeftState.DRAG else []
        self._left = LeftState.IDLE
        self._pending_origin = None
        self._cursor_frozen = False
        for latch in (*self._pinch.values(), *self._finger.values(), self._joined):
            latch.reset()
        self._armed = dict.fromkeys(PINCH_FINGERS, False)
        self._mode = Mode.NO_HAND
        self._candidate, self._candidate_frames = None, 0
        self._exit_scroll()
        self._exit_zoom()
        self._media.reset()
        self._switch_since, self._switch_done = None, False
        self._last_seen = self._last_t = None
        self.mapper.reset()
        return actions

    def update(self, obs: HandObservation | None, t: float) -> EngineOutput:
        if obs is None:
            return self._on_no_hand(t)

        self._last_seen = t
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 0.1)
        self._last_t = t
        features = HandFeatures.from_observation(obs, self._anchor_weight)
        extended = {f: latch.update(features.extension[f]) for f, latch in self._finger.items()}
        joined = self._joined.update(features.index_middle_gap)
        if self._mode is Mode.NO_HAND:  # hand (re)acquired
            self._mode = Mode.MEDIA if self._media_active else Mode.POINTER

        actions: list[Action] = []
        # An engaged pinch owns the hand: curling the other fingers mid-drag must not turn the
        # pose into a "fist" (clutch) and drop whatever is being dragged.
        if self._left is LeftState.IDLE:
            raw = self._classify(extended, joined, features)
            if self._media_active and raw is not Mode.SWITCH:
                raw = Mode.MEDIA  # media mode reads every other pose itself
        else:
            raw = Mode.POINTER
        previous_mode = self._debounce(raw)
        if previous_mode is not None:
            actions += self._on_mode_change(previous_mode)
        pending = raw is not self._mode

        if self._mode is Mode.POINTER:
            actions += self._pointer(features, t, posture_pending=pending)
        else:
            self._cursor_frozen = True
            self.mapper.update(features.anchor, t, frozen=True)  # keep the filter converged
            if self._mode is Mode.SCROLL:
                actions += self._scroll(features, dt)
            elif self._mode is Mode.ZOOM:
                actions += self._zoom(features, t)
            elif self._mode is Mode.SWITCH:
                self._switch(t)
            elif self._mode is Mode.MEDIA:
                fist = not any(extended.values())
                media_actions, event = self._media.update(obs, fist, t, dt)
                actions += media_actions
                if event is not None:
                    self._set_flash(event, t, 0.5)

        return EngineOutput(
            mode=self._mode,
            label=self._label(t),
            actions=actions,
            cursor_frozen=self._cursor_frozen,
            left_state=self._left,
            features=features,
            extended=extended,
            pending_mode=raw if pending else None,
            scroll_origin=self._scroll_origin,
            scroll_rate=(float(self._scroll_rate[0]), float(self._scroll_rate[1])),
            media_active=self._media_active,
            knob_offset=self._media.knob_offset if self._mode is Mode.MEDIA else None,
            volume_rate=self._media.volume_rate if self._mode is Mode.MEDIA else 0.0,
        )

    # -- tracking loss -----------------------------------------------------------------------
    def _on_no_hand(self, t: float) -> EngineOutput:
        if self._last_seen is not None and t - self._last_seen <= self.cfg.hand_lost_grace:
            # Short dropout (motion blur, fingers crossing the palm): hold everything as it is.
            return EngineOutput(
                self._mode,
                "TRACKING LOST",
                cursor_frozen=True,
                left_state=self._left,
                media_active=self._media_active,
            )
        actions = self.reset() if self._mode is not Mode.NO_HAND else []
        return EngineOutput(Mode.NO_HAND, "NO HAND", actions, media_active=self._media_active)

    # -- posture -----------------------------------------------------------------------------
    def _classify(self, ext: dict[Finger, bool], joined: bool, f: HandFeatures) -> Mode:
        i, m, r, p = (ext[Finger.INDEX], ext[Finger.MIDDLE], ext[Finger.RING], ext[Finger.PINKY])
        if not (i or m or r or p) and f.pinch[Finger.INDEX] > self.cfg.pinch_enter:
            return Mode.CLUTCH  # fist (and not a pinch made with curled fingers)
        if p and not (i or m or r):
            return Mode.SWITCH  # shaka: only the pinky (and the thumb) out
        if i and p and not m and not r:
            return Mode.ZOOM
        if i and m and not r and not p and joined:
            return Mode.SCROLL
        return Mode.POINTER

    def _debounce(self, raw: Mode) -> Mode | None:
        """Switch mode only after ``raw`` was seen N frames in a row. Returns the old mode."""
        if raw is self._mode:
            self._candidate, self._candidate_frames = None, 0
            return None
        if raw is self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate, self._candidate_frames = raw, 1
        if self._candidate_frames < self.cfg.posture_confirm_frames:
            return None
        previous, self._mode = self._mode, raw
        self._candidate, self._candidate_frames = None, 0
        return previous

    def _on_mode_change(self, previous: Mode) -> list[Action]:
        actions: list[Action] = []
        if previous is Mode.POINTER:
            if self._left is LeftState.DRAG:  # defensive: an engaged pinch pins POINTER mode
                actions.append(Release("left"))
            self._left = LeftState.IDLE
            self._pending_origin = None
            for latch in self._pinch.values():
                latch.reset()
        elif previous is Mode.SCROLL:
            self._exit_scroll()
        elif previous is Mode.ZOOM:
            self._exit_zoom()
        elif previous is Mode.SWITCH:
            self._switch_since, self._switch_done = None, False
        elif previous is Mode.MEDIA:
            self._media.reset()
        if self._mode is Mode.POINTER:
            self._armed = dict.fromkeys(PINCH_FINGERS, False)
        return actions

    # -- POINTER: move, click, drag, right click, double click -----------------------------------
    def _pointer(self, f: HandFeatures, t: float, posture_pending: bool) -> list[Action]:
        cfg = self.cfg
        actions: list[Action] = []
        d = f.pinch

        # Arbitration: only the clearly closest finger may start a pinch. When the thumb meets
        # the index, the middle fingertip is often close too; without this rule one pinch can
        # fire a left *and* a right click.
        ranked = sorted(PINCH_FINGERS, key=d.__getitem__)
        best, runner_up = ranked[0], ranked[1]
        clear_winner = d[runner_up] - d[best] >= cfg.pinch_ambiguity_margin
        busy = any(latch.active for latch in self._pinch.values())
        for finger in PINCH_FINGERS:
            if d[finger] > cfg.pinch_exit:
                self._armed[finger] = True

        rose: dict[Finger, bool] = {}
        fell: dict[Finger, bool] = {}
        for finger, latch in self._pinch.items():
            before = latch.active
            if before:
                latch.update(d[finger])  # release depends only on this finger's own distance
            else:
                allowed = (
                    finger is best
                    and clear_winner
                    and self._armed[finger]
                    and not busy
                    and not posture_pending
                )
                latch.update(d[finger] if allowed else math.inf)
            rose[finger] = latch.active and not before
            fell[finger] = before and not latch.active

        # Left button. Pressing immediately would turn every few-pixel wobble into a micro-drag
        # (e.g. starting a drag-select in a file manager), so the decision is deferred:
        # released quickly -> click; held long enough or moved far enough -> press and drag.
        if rose[Finger.INDEX]:
            self._left = LeftState.PENDING
            self._left_t0 = t
            self._pending_origin = self.mapper.target
        if self._left is LeftState.PENDING:
            if fell[Finger.INDEX]:
                actions.append(Click("left"))
                self._left = LeftState.IDLE
                self._set_flash("LEFT CLICK", t)
            elif (
                t - self._left_t0 >= cfg.click_max_hold
                or self._pending_travel() >= cfg.drag_start_px
            ):
                actions.append(Press("left"))  # at the frozen position = where the pinch began
                self._left = LeftState.DRAG
        elif self._left is LeftState.DRAG and fell[Finger.INDEX]:
            actions.append(Release("left"))
            self._left = LeftState.IDLE
            self._set_flash("DROP", t)

        # Right click fires once per pinch (the latch must release before it can fire again);
        # the cooldown additionally absorbs a pinch that "bounces" through the hysteresis band.
        if rose[Finger.MIDDLE] and t - self._last_right >= cfg.right_click_cooldown:
            actions.append(Click("right"))
            self._last_right = t
            self._set_flash("RIGHT CLICK", t)
        if rose[Finger.RING] and t - self._last_double >= cfg.double_click_cooldown:
            actions.append(Click("left", 2))
            self._last_double = t
            self._set_flash("DOUBLE CLICK", t)

        # Cursor. Freeze while a pinch is approaching or pending: clicks land exactly where the
        # cursor was, and two quick pinches stay within the OS double-click distance (~4 px).
        approaching = cfg.approach_freeze > 0 and d[best] < cfg.approach_freeze
        latched = any(latch.active for latch in self._pinch.values())
        if self._left is LeftState.DRAG:
            frozen = False
        else:
            frozen = posture_pending or self._left is LeftState.PENDING or approaching or latched
        self._cursor_frozen = frozen
        move = self.mapper.update(f.anchor, t, frozen=frozen)
        if move is not None:
            actions.append(move)  # after Press: the drag starts where the pinch began
        return actions

    def _pending_travel(self) -> float:
        target = self.mapper.target
        if self._pending_origin is None or target is None:
            return 0.0
        return float(np.linalg.norm(target - self._pending_origin))

    # -- SCROLL ------------------------------------------------------------------------------
    def _scroll(self, f: HandFeatures, dt: float) -> list[Action]:
        cfg = self.cfg
        if self._scroll_origin is None:
            self._scroll_origin = f.scroll_point
            self._scroll_accum[:] = 0.0
            return []
        (ox, oy), (x, y) = self._scroll_origin, f.scroll_point
        # Displacement in palm lengths (scale invariant): right = +x, up = +y (image y is down).
        displacement = np.array([x - ox, oy - y]) / f.palm_scale
        rate = np.zeros(2)
        for axis in (0, 1):
            excess = abs(displacement[axis]) - cfg.scroll_deadzone
            if excess > 0:
                speed = min(cfg.scroll_gain * excess, cfg.scroll_max_rate)
                rate[axis] = math.copysign(speed, displacement[axis])
        if not cfg.scroll_horizontal:
            rate[0] = 0.0
        # One axis at a time: slight diagonal drift must not scroll sideways while reading.
        off_axis = 1 if abs(rate[0]) > abs(rate[1]) else 0
        rate[off_axis] = 0.0
        self._scroll_accum[off_axis] = 0.0
        if cfg.scroll_invert:
            rate = -rate
        self._scroll_rate = rate
        # Rate x elapsed time, not "N ticks per frame": scroll speed must not depend on FPS.
        self._scroll_accum += rate * dt
        ticks = np.trunc(self._scroll_accum)
        self._scroll_accum -= ticks
        if ticks[0] == 0 and ticks[1] == 0:
            return []
        return [Scroll(int(ticks[0]), int(ticks[1]))]

    def _exit_scroll(self) -> None:
        self._scroll_origin = None
        self._scroll_accum[:] = 0.0
        self._scroll_rate = np.zeros(2)

    # -- ZOOM --------------------------------------------------------------------------------
    def _zoom(self, f: HandFeatures, t: float) -> list[Action]:
        spread = float(self._zoom_filter(f.pinch[Finger.INDEX], t))
        if self._zoom_ref is None:
            self._zoom_ref = spread
            return []
        # Detents: one tick per zoom_step of spread change. The reference moves by whole steps
        # only, so noise smaller than one step can never produce in-out-in-out flicker.
        steps = int((spread - self._zoom_ref) / self.cfg.zoom_step)
        if steps == 0:
            return []
        cap = self.cfg.zoom_max_steps_per_frame
        steps = max(-cap, min(cap, steps))
        self._zoom_ref += steps * self.cfg.zoom_step
        self._set_flash("ZOOM IN" if steps > 0 else "ZOOM OUT", t)
        return [Zoom(steps)]

    def _exit_zoom(self) -> None:
        self._zoom_ref = None
        self._zoom_filter.reset()

    # -- SWITCH: mouse <-> media -------------------------------------------------------------
    def _switch(self, t: float) -> None:
        if self._switch_since is None:
            self._switch_since = t
        # Toggles once per hold; the shaka has to be released before it can toggle again.
        if not self._switch_done and t - self._switch_since >= self.media_cfg.toggle_hold:
            self._switch_done = True
            self._media_active = not self._media_active
            self._media.reset()
            self._set_flash("MEDIA MODE" if self._media_active else "MOUSE MODE", t, 1.0)

    # -- HUD label ---------------------------------------------------------------------------
    def _set_flash(self, text: str, t: float, duration: float = 0.35) -> None:
        self._flash, self._flash_until = text, t + duration

    def _label(self, t: float) -> str:
        if t < self._flash_until:
            return self._flash
        if self._mode is Mode.SWITCH:
            return f"HOLD -> {'MOUSE' if self._media_active else 'MEDIA'}"
        if self._mode is Mode.MEDIA:
            rate = self._media.volume_rate
            return "VOL +" if rate > 0 else "VOL -" if rate < 0 else "MEDIA"
        if self._mode is not Mode.POINTER:
            return self._mode.value
        if self._left is LeftState.PENDING:
            return "PINCH"
        if self._left is LeftState.DRAG:
            return "DRAG"
        return "HOLD" if self._cursor_frozen else "MOVE"
