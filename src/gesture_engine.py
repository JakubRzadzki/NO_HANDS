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
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from config import GestureConfig, MediaConfig, SlidesConfig
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
from src.os_controller import Action, Click, KeyTap, Press, Release, Scroll, Zoom
from src.pointer import CursorMapper
from src.smoother import FloatArray
from src.swipes import SwipeDetector


class Mode(str, Enum):
    NO_HAND = "NO HAND"
    POINTER = "POINTER"
    SCROLL = "SCROLL"
    ZOOM = "ZOOM"
    CLUTCH = "CLUTCH"
    SWITCH = "SWITCH"  # shaka held: toggles between mouse and media mode
    MEDIA = "MEDIA"  # music control instead of the mouse
    SLIDES = "SLIDES"  # --slides: hand strokes change slides and zoom (src/slide_gestures.py)


class LeftState(str, Enum):
    IDLE = "IDLE"
    PENDING = "PENDING"  # button down, cursor still frozen: a click if released now
    DRAG = "DRAG"  # button down, the cursor follows the hand


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
    ``release_frames`` (default: the same) can make letting go slower than engaging: a held
    mouse button must survive a couple of bad frames while the hand moves.
    """

    def __init__(
        self,
        engage_at: float,
        release_at: float,
        low_active: bool = True,
        confirm_frames: int = 1,
        release_frames: int | None = None,
    ) -> None:
        if (low_active and engage_at >= release_at) or (not low_active and engage_at <= release_at):
            raise ValueError("the release threshold must lie on the inactive side of engage_at")
        self.engage_at = engage_at
        self.release_at = release_at
        self.low_active = low_active
        self.confirm_frames = max(1, confirm_frames)
        self.release_frames = self.confirm_frames if release_frames is None else release_frames
        self.active = False
        self._streak = 0

    def update(self, value: float) -> bool:
        if self.active:
            flip = value > self.release_at if self.low_active else value < self.release_at
        else:
            flip = value < self.engage_at if self.low_active else value > self.engage_at
        if flip:
            self._streak += 1
            if self._streak >= (self.release_frames if self.active else self.confirm_frames):
                self.active = not self.active
                self._streak = 0
        else:
            self._streak = 0
        return self.active

    def reset(self) -> None:
        self.active = False
        self._streak = 0


def pinch_distance(obs: HandObservation, a: int, b: int, depth_weight: float) -> float:
    """Distance between two fingertips in palm lengths, depth counted at ``depth_weight``.

    MediaPipe's depth (z) is its least reliable output, and its noise is as large as a pinch is
    small. On real photos of touching fingers the image distance was 0.09-0.18 palm lengths
    while the full 3D distance read 0.13-0.26. Some depth is still worth having: seen from
    the side, fingers that overlap in the image can be far apart."""
    d = obs.px[a] - obs.px[b]
    return float(math.sqrt(d[0] ** 2 + d[1] ** 2 + (depth_weight * d[2]) ** 2)) / obs.palm_scale


@dataclass(frozen=True)
class HandFeatures:
    palm_scale: float  # px
    pinch: dict[Finger, float]  # thumb tip <-> finger tip, palm lengths
    extension: dict[Finger, float]  # dist(wrist, tip) / dist(wrist, pip)
    index_middle_gap: float  # index tip <-> middle tip, palm lengths
    anchor: tuple[float, float]  # cursor anchor, normalized frame coordinates
    anchor_px: tuple[float, float]  # the same anchor in px (zoom measures its travel)
    scroll_point: tuple[float, float]  # px, midpoint of the index and middle tips

    @classmethod
    def from_observation(
        cls, obs: HandObservation, anchor_index_weight: float = 0.5, depth_weight: float = 0.5
    ) -> HandFeatures:
        pinch = {
            finger: pinch_distance(obs, THUMB_TIP, finger.tip, depth_weight)
            for finger in PINCH_FINGERS
        }
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
        anchor_px = w * obs.px[INDEX_MCP, :2] + (1.0 - w) * obs.px[MIDDLE_MCP, :2]
        scroll_point = (obs.px[INDEX_TIP, :2] + obs.px[MIDDLE_TIP, :2]) / 2.0
        return cls(
            palm_scale=obs.palm_scale,
            pinch=pinch,
            extension=extension,
            index_middle_gap=obs.ndist(INDEX_TIP, MIDDLE_TIP),
            anchor=(float(anchor[0]), float(anchor[1])),
            anchor_px=(float(anchor_px[0]), float(anchor_px[1])),
            scroll_point=(float(scroll_point[0]), float(scroll_point[1])),
        )


def is_pinch_shape(f: HandFeatures, finger: Finger, cfg: GestureConfig) -> bool:
    """Is the hand shaped like a pinch with this finger, apart from the distance?

    The finger reaches for the thumb (bent, not folded), and the hand is open: at least one
    other finger is straight. A thumb resting on a folded finger - a fist, a V sign, the horns -
    is not a pinch. Checked on 480 real photos: all the real pinches pass, no fist, V sign or
    open hand does."""
    if f.extension[finger] < cfg.pinch_min_extension:
        return False
    others = (ratio for other, ratio in f.extension.items() if other is not finger)
    return any(ratio > cfg.finger_extended_above for ratio in others)


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
    zoom_origin: tuple[float, float] | None = None  # px, where the zoom drag started (HUD)
    scroll_rate: tuple[float, float] = (0.0, 0.0)  # ticks/s, for the HUD
    media_active: bool = False  # media mode instead of mouse mode
    knob_offset: float | None = None  # degrees from neutral, for the HUD
    volume_rate: float = 0.0  # volume key presses per second, for the HUD
    trail: list[tuple[float, float]] = field(default_factory=list)  # --slides: wrist path, px
    stroke: tuple[float, float] | None = None  # --slides: stroke so far, palm lengths


class GestureEngine:
    def __init__(
        self,
        cfg: GestureConfig,
        mapper: CursorMapper,
        anchor_index_weight: float = 0.5,
        media: MediaConfig | None = None,
        slides: SlidesConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.mapper = mapper
        self._anchor_weight = anchor_index_weight
        self.media_cfg = media or MediaConfig()
        self._media = MediaGestures(self.media_cfg)
        # Flick the open hand sideways and straight back = previous / next slide. Same detector
        # as --slides, with a longer and faster stroke and the way back required.
        self._slides = replace(
            slides or SlidesConfig(),
            swipe_distance=cfg.flick_distance,
            swipe_speed=max(cfg.flick_speed, (slides or SlidesConfig()).start_speed + 0.1),
        )
        self._swipes = (
            SwipeDetector(self._slides, require_return=True) if cfg.flick_slides else None
        )
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
                cfg.pinch_enter,
                cfg.pinch_exit,
                confirm_frames=cfg.pinch_confirm_frames,
                release_frames=cfg.pinch_release_frames,
            )
            for finger in PINCH_FINGERS
        }
        # A pinch may only engage after its fingers were seen apart. Otherwise leaving ZOOM with
        # thumb and index together, or the hand re-appearing mid-pinch, would fire a click
        # nobody asked for. (Edge-triggered, not level-triggered.)
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
        self._settle_until = -math.inf  # the cursor waits here after a click
        self._scroll_origin: tuple[float, float] | None = None
        self._scroll_accum = np.zeros(2)
        self._scroll_rate = np.zeros(2)
        self._zoom_ref: float | None = None  # hand height (px) of the last zoom detent
        self._zoom_origin: tuple[float, float] | None = None
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
        actions: list[Action] = [] if self._left is LeftState.IDLE else [Release("left")]
        self._left = LeftState.IDLE
        self._pending_origin = None
        self._cursor_frozen = False
        if self._swipes is not None:
            self._swipes.reset()
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
        features = HandFeatures.from_observation(
            obs, self._anchor_weight, self.cfg.pinch_depth_weight
        )
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

        if self._mode in (Mode.POINTER, Mode.CLUTCH):
            # Fingers seen apart arm the pinch. A fist counts too: a pinch made with the other
            # fingers curled passes through "fist" for a few frames on its way.
            for finger in PINCH_FINGERS:
                if features.pinch[finger] > self.cfg.pinch_exit:
                    self._armed[finger] = True

        if self._mode is Mode.POINTER:
            actions += self._pointer(obs, features, t, posture_pending=pending)
        else:
            if self._swipes is not None:
                self._swipes.reset()
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
            zoom_origin=self._zoom_origin,
            scroll_rate=(float(self._scroll_rate[0]), float(self._scroll_rate[1])),
            media_active=self._media_active,
            knob_offset=self._media.knob_offset if self._mode is Mode.MEDIA else None,
            volume_rate=self._media.volume_rate if self._mode is Mode.MEDIA else 0.0,
        )

    # -- tracking loss -----------------------------------------------------------------------
    def _on_no_hand(self, t: float) -> EngineOutput:
        held = self._left is not LeftState.IDLE
        grace = self.cfg.drag_lost_grace if held else self.cfg.hand_lost_grace
        if self._last_seen is not None and t - self._last_seen <= grace:
            # Short dropout (motion blur, fingers crossing the palm): hold everything as it is.
            # A held button waits longer: a drag must not drop because of a few lost frames.
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
        if not (i or m or r or p):
            # A fist, wherever the thumb is: in a quarter of real fists its tip lies right next
            # to the folded index tip, which must not count as a pinch.
            return Mode.CLUTCH
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
            if self._left is not LeftState.IDLE:  # defensive: an engaged pinch pins POINTER mode
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
        if self._mode is Mode.POINTER and previous is not Mode.CLUTCH:
            self._armed = dict.fromkeys(PINCH_FINGERS, False)
        return actions

    # -- POINTER: move, click, drag, right click, double click -----------------------------------
    def _pointer(
        self, obs: HandObservation, f: HandFeatures, t: float, posture_pending: bool
    ) -> list[Action]:
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
                    and (finger is not Finger.RING or cfg.ring_double_click)
                    and is_pinch_shape(f, finger, cfg)
                )
                latch.update(d[finger] if allowed else math.inf)
            rose[finger] = latch.active and not before
            fell[finger] = before and not latch.active

        # Left button: down the moment thumb and index meet, up when they part - held exactly as
        # long as the pinch, like a finger on a real mouse button. The cursor stays frozen for
        # the first click_max_hold seconds (or until the hand has clearly moved), so a quick
        # pinch presses and releases on the same pixel: a clean click, never a micro-drag.
        # After that the cursor follows the hand with the button still down: a drag.
        if rose[Finger.INDEX]:
            actions.append(Press("left"))
            self._left = LeftState.PENDING
            self._left_t0 = t
            self._pending_origin = self.mapper.target
        elif self._left is not LeftState.IDLE and fell[Finger.INDEX]:
            actions.append(Release("left"))
            if self._left is LeftState.PENDING:
                self._set_flash("LEFT CLICK", t)
                self._settle_until = t + cfg.click_settle  # a second pinch = a double click
            else:
                self._set_flash("DROP", t)
            self._left = LeftState.IDLE
        elif self._left is LeftState.PENDING and (
            t - self._left_t0 >= cfg.click_max_hold or self._pending_travel() >= cfg.drag_start_px
        ):
            self._left = LeftState.DRAG

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

        # Cursor. Freeze while a pinch is approaching or just pressed: clicks land exactly where
        # the cursor was, and two quick pinches stay within the OS double-click distance (~4 px).
        approaching = cfg.approach_freeze > 0 and d[best] < cfg.approach_freeze
        latched = any(latch.active for latch in self._pinch.values())
        # After a click the cursor waits a moment where the click landed, so that a second
        # pinch lands on the same pixel (the OS allows ~4 px between the clicks of a double
        # click) - unless the hand is clearly moving on.
        settling = t < self._settle_until and self._pending_travel() < cfg.drag_start_px
        if self._left is LeftState.DRAG:
            frozen = False
        else:
            frozen = (
                posture_pending
                or self._left is LeftState.PENDING
                or approaching
                or latched
                or settling
            )
        self._cursor_frozen = frozen
        move = self.mapper.update(f.anchor, t, frozen=frozen)
        if move is not None:
            actions.append(move)  # after Press: the drag starts where the pinch began

        actions += self._flick(obs, f, t, busy=latched or approaching or posture_pending)
        return actions

    def _flick(self, obs: HandObservation, f: HandFeatures, t: float, busy: bool) -> list[Action]:
        """Open hand flicked sideways and straight back = Left / Right (previous / next slide).

        The cursor follows the hand out and back, so it ends where it was. Moving the cursor
        never does this: nobody points far, fast, and straight back."""
        if self._swipes is None:
            return []
        if busy:
            self._swipes.cancel()  # the hand is pinching: it is not swiping
            return []
        not_curled = sum(ratio > self.cfg.finger_curled_below for ratio in f.extension.values())
        direction = self._swipes.update(
            t,
            obs.px[:, :2].mean(axis=0),
            f.palm_scale,
            not_curled >= self._slides.min_fingers,
            float(obs.frame_size[0]),
        )
        if direction is None:
            return []
        self._set_flash("RIGHT >>" if direction == "right" else "<< LEFT", t, 0.6)
        return [KeyTap(direction)]

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
        """Drag to zoom: in the horns pose, moving the hand up zooms in, down zooms out, one
        wheel tick per zoom_travel palm lengths, measured at the knuckles (the thumb is free)."""
        y = f.anchor_px[1]
        if self._zoom_ref is None:
            self._zoom_ref = y
            self._zoom_origin = f.anchor_px
            return []
        step = self.cfg.zoom_travel * f.palm_scale
        steps = int((self._zoom_ref - y) / step)  # image y grows downwards: up = positive
        if steps == 0:
            return []
        cap = self.cfg.zoom_max_steps_per_frame
        steps = max(-cap, min(cap, steps))
        # The reference moves by whole detents only, so noise smaller than one detent can never
        # produce in-out-in-out flicker; going back needs a full detent of travel.
        self._zoom_ref -= steps * step
        self._set_flash("ZOOM IN" if steps > 0 else "ZOOM OUT", t)
        return [Zoom(steps)]

    def _exit_zoom(self) -> None:
        self._zoom_ref = None
        self._zoom_origin = None

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
            return "BUTTON DOWN"
        if self._left is LeftState.DRAG:
            return "BUTTON DOWN: DRAG"
        return "HOLD" if self._cursor_frozen else "MOVE"
