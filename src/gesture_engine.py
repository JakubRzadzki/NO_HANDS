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
from src.gun import GunTrigger, gun_check
from src.hand_tracker import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    HandObservation,
)
from src.media_gestures import MediaGestures
from src.os_controller import Action, Click, KeyTap, Press, Release, Scroll, Zoom
from src.pinch import HysteresisLatch as HysteresisLatch
from src.pinch import pinch_distance as pinch_distance
from src.pointer import CursorMapper
from src.smoother import FloatArray
from src.stop_hand import stop_check
from src.swipes import SwipeDetector
from src.two_hands import TwoHandZoom


class Mode(str, Enum):
    NO_HAND = "NO HAND"
    POINTER = "POINTER"
    SCROLL = "SCROLL"
    ZOOM = "ZOOM"
    CLUTCH = "CLUTCH"
    SWITCH = "SWITCH"  # shaka held: toggles between mouse and media mode
    MEDIA = "MEDIA"  # music control instead of the mouse
    SLIDES = "SLIDES"  # --slides: hand strokes change slides and zoom (src/slide_gestures.py)
    TWO_HANDS = "TWO HANDS"  # both hands in view: pinch both and pull apart / together = zoom
    GUN = "GUN"  # finger gun: the barrel's direction changes the slide (src/gun.py)


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


# s the other hand may be lost (motion blur) before its swipe is forgotten.
OTHER_HAND_GAP = 0.15

# Relaxed horns: middle and ring this much (extension ratio) below index and pinky.
HORNS_MARGIN = 0.15

# Thumb pinches: INDEX = left click / drag, MIDDLE = right click, RING = double click.
PINCH_FINGERS = (Finger.INDEX, Finger.MIDDLE, Finger.RING)


@dataclass(frozen=True)
class HandFeatures:
    palm_scale: float  # px
    pinch: dict[Finger, float]  # thumb tip <-> finger tip, palm lengths
    extension: dict[Finger, float]  # dist(wrist, tip) / dist(wrist, pip)
    palm_width: float  # index knuckle <-> pinky knuckle in the image, palm lengths
    index_middle_gap: float  # index tip <-> middle tip, palm lengths
    anchor: tuple[float, float]  # cursor anchor, normalized frame coordinates
    anchor_px: tuple[float, float]  # the same anchor in px (zoom measures its travel)
    scroll_point: tuple[float, float]  # px, centre of the palm (the whole hand scrolls)

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
        scroll_point = obs.px[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2].mean(axis=0)
        # In the image only: turned sideways, the knuckles line up one behind the other.
        knuckles = obs.px[INDEX_MCP, :2] - obs.px[PINKY_MCP, :2]
        return cls(
            palm_scale=obs.palm_scale,
            pinch=pinch,
            extension=extension,
            palm_width=float(math.hypot(knuckles[0], knuckles[1])) / obs.palm_scale,
            index_middle_gap=obs.ndist(INDEX_TIP, MIDDLE_TIP),
            anchor=(float(anchor[0]), float(anchor[1])),
            anchor_px=(float(anchor_px[0]), float(anchor_px[1])),
            scroll_point=(float(scroll_point[0]), float(scroll_point[1])),
        )


def is_pinch_shape(f: HandFeatures, finger: Finger, cfg: GestureConfig) -> bool:
    """Is the hand shaped like a pinch with this finger, apart from the distance?

    The finger reaches for the thumb (bent, not folded), and the hand is open: at least one
    other finger is not folded. A thumb resting on a folded finger - a fist, a V sign, the
    horns - is not a pinch. Checked on 480 real photos: all the real pinches pass, no fist,
    V sign or open hand does."""
    if f.extension[finger] < cfg.pinch_min_extension:
        return False
    others = (ratio for other, ratio in f.extension.items() if other is not finger)
    return any(ratio > cfg.pinch_other_finger_above for ratio in others)


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
    speed: float = 1.0  # cursor speed: < 1 = precision (fingers folded)
    gun_hint: str = ""  # why the hand is or is not a finger gun, for the preview
    stop_hint: str = ""  # why the other hand is or is not a stop sign (scroll), for the preview
    two_hands: tuple[tuple[float, float], tuple[float, float]] | None = None  # the two pinches
    pinched_hands: tuple[bool, bool] = (False, False)  # two-hand mode: which hand pinches


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
        # The other hand's swipes change slides, with the lenient --slides thresholds.
        self._other_swipes = (
            SwipeDetector(slides or SlidesConfig()) if cfg.other_hand_slides else None
        )
        self._other_seen: float | None = None  # when the other hand was last seen
        # Finger guns, one trigger per hand: pointing right = next slide, left = previous.
        self._guns = (GunTrigger(cfg), GunTrigger(cfg))
        self._gun_hint = ""
        # The other hand's stop sign = scroll, debounced: fed 0 (seen) or 1 (not seen).
        self._stop = HysteresisLatch(
            0.5,
            0.5 + 1e-9,
            confirm_frames=cfg.stop_confirm_frames,
            release_frames=cfg.stop_release_frames,
        )
        self._stop_scrolling = False
        self._stop_hint = ""
        self._last_second: HandObservation | None = None
        self._media_active = False  # survives tracking loss: lowering the hand is normal here
        self._two = TwoHandZoom(cfg) if cfg.two_hand_zoom else None
        self._speed_level = 0  # index into cfg.precision_speeds
        self._speed_candidate = 0
        self._speed_frames = 0
        self._switch_since: float | None = None
        self._switch_done = False
        self._finger = {
            finger: HysteresisLatch(
                cfg.finger_extended_above, cfg.finger_curled_below, low_active=False
            )
            for finger in Finger
        }
        self._sideways = HysteresisLatch(cfg.scroll_palm_enter, cfg.scroll_palm_exit)
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
        self._pending_origin: FloatArray | None = None  # cursor position at the press
        self._press_anchor: tuple[float, float] | None = None  # hand (px) at the press / release
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
        for gun in self._guns:
            gun.reset()
        self._stop.reset()
        self._stop_scrolling = False
        self._speed_level = self._speed_candidate = self._speed_frames = 0
        if self._two is not None:
            self._two.reset()
        self._left = LeftState.IDLE
        self._pending_origin = None
        self._press_anchor = None
        self._cursor_frozen = False
        if self._swipes is not None:
            self._swipes.reset()
        for latch in (
            *self._pinch.values(),
            *self._finger.values(),
            self._sideways,
            self._joined,
        ):
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

    def update(
        self, obs: HandObservation | None, t: float, second: HandObservation | None = None
    ) -> EngineOutput:
        """``second``: the other hand, when two are in view (two-hand zoom)."""
        if obs is None:
            return self._on_no_hand(t)

        self._last_seen = t
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 0.1)
        self._last_t = t
        features = HandFeatures.from_observation(
            obs, self._anchor_weight, self.cfg.pinch_depth_weight
        )
        extended = {f: latch.update(features.extension[f]) for f, latch in self._finger.items()}
        sideways = self._sideways.update(features.palm_width) and self.cfg.scroll_sideways
        joined = self._joined.update(features.index_middle_gap)
        if self._mode is Mode.NO_HAND:  # hand (re)acquired
            self._mode = Mode.MEDIA if self._media_active else Mode.POINTER

        actions: list[Action] = []
        # The other hand: a finger gun or a swipe changes the slide. Neither changes anything
        # about this hand.
        other = (
            None
            if second is None
            else HandFeatures.from_observation(second, depth_weight=self.cfg.pinch_depth_weight)
        )
        self._last_second = second
        aim = self._aim(obs, features)
        other_aim = self._aim(second, other)
        actions += self._fire_guns(aim, other_aim, t)
        actions += self._other_hand_swipe(second, other, other_aim is not None, t)
        if self._two is not None and not self._media_active:
            # Both hands pinch at about the same time for a two-hand zoom: with a second hand in
            # view the cursor hand's pinch waits a few frames longer, so it cannot click first.
            delay = self.cfg.two_hand_click_delay_frames if second is not None else 0
            for latch in self._pinch.values():
                latch.confirm_frames = self.cfg.pinch_confirm_frames + delay
            # Only a pinching second hand counts. A held button keeps the hand mouse: a second
            # hand must not end a drag.
            pinching = self._two.second_pinched(second)
            if pinching and self._left is LeftState.IDLE:
                return self._two_hands(obs, features, extended, t)
            if self._mode is Mode.TWO_HANDS:
                actions += self._leave_two_hands()
        # The other hand's stop sign: this hand's up / down movement scrolls.
        stop_seen, self._stop_hint = False, ""
        if self.cfg.scroll_with_other_hand and second is not None and other is not None:
            stop_seen, self._stop_hint = stop_check(second, other, self.cfg)
        stopped = self._stop.update(0.0 if stop_seen else 1.0)
        if stopped and self._left is LeftState.IDLE and not self._media_active:
            return self._stop_scroll(features, extended, dt, t)
        if self._stop_scrolling:
            actions += self._leave_stop_scroll()
        # An engaged pinch owns the hand: curling the other fingers mid-drag must not turn the
        # pose into a "fist" (clutch) and drop whatever is being dragged.
        if self._left is LeftState.IDLE:
            raw = (
                Mode.GUN
                if aim is not None
                else self._classify(extended, joined, sideways, features)
            )
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

        if self._mode is Mode.POINTER and self._left is LeftState.IDLE:
            self._update_speed(extended)  # kept while the button is held: drags stay steady
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
            speed=self.speed,
            gun_hint=self._gun_hint,
            stop_hint=self._stop_hint,
        )

    # -- precision ---------------------------------------------------------------------------
    @property
    def speed(self) -> float:
        """Cursor speed factor: 1 = full, smaller = precision (fingers folded)."""
        return self.cfg.precision_speeds[self._speed_level]

    def _update_speed(self, extended: dict[Finger, bool]) -> None:
        """Fold middle / ring / pinky to slow the cursor. The index is not counted: it makes the
        pinch, and clicking must never change the speed."""
        folded = sum(not extended[f] for f in (Finger.MIDDLE, Finger.RING, Finger.PINKY))
        level = min(folded, len(self.cfg.precision_speeds) - 1)
        if level == self._speed_level:
            self._speed_frames = 0
            return
        if level == self._speed_candidate:
            self._speed_frames += 1
        else:
            self._speed_candidate, self._speed_frames = level, 1
        if self._speed_frames >= self.cfg.precision_confirm_frames:
            self._speed_level, self._speed_frames = level, 0

    def _aim(self, obs: HandObservation | None, f: HandFeatures | None) -> str | None:
        """Where a finger gun points, if the hand is one (and guns are on)."""
        if obs is None or f is None or not self.cfg.gun_slides or self._media_active:
            return None
        direction, hint = gun_check(obs, f, self.cfg)
        if obs is not self._last_second:
            self._gun_hint = hint  # the cursor hand's, shown in the preview
        return direction

    def _fire_guns(self, aim: str | None, other_aim: str | None, t: float) -> list[Action]:
        """One slide per gun, from either hand. Not while the button is held: a drag is not a
        slide change."""
        if self._left is not LeftState.IDLE:
            aim = None
        actions: list[Action] = []
        for gun, direction in zip(self._guns, (aim, other_aim), strict=True):
            fired = gun.update(direction)
            if fired is not None:
                actions.append(KeyTap(fired))
                self._set_flash("RIGHT >>" if fired == "right" else "<< LEFT", t, 0.6)
        return actions

    def _other_hand_swipe(
        self, second: HandObservation | None, f: HandFeatures | None, gun: bool, t: float
    ) -> list[Action]:
        """The other hand swipes the slides: it does not point, so the --slides thresholds fit."""
        swipes = self._other_swipes
        if swipes is None:
            return []
        if second is None or f is None:
            if self._other_seen is not None and t - self._other_seen > OTHER_HAND_GAP:
                swipes.reset()  # gone, not just a blurred frame or two
                self._other_seen = None
            return []
        self._other_seen = t
        busy = gun or (self._two is not None and self._two.pinched[1])
        if busy or self._media_active:
            swipes.cancel()  # holding a gun or zooming is not swiping
            return []
        slides = swipes.cfg
        not_curled = sum(ratio > self.cfg.finger_curled_below for ratio in f.extension.values())
        direction = swipes.update(
            t,
            second.px[:, :2].mean(axis=0),
            second.palm_scale,
            not_curled >= slides.min_fingers,
            float(second.frame_size[0]),
        )
        if direction is None:
            return []
        self._set_flash("RIGHT >>" if direction == "right" else "<< LEFT", t, 0.6)
        return [KeyTap(direction)]

    # -- two hands ---------------------------------------------------------------------------
    def _two_hands(
        self, obs: HandObservation, features: HandFeatures, extended: dict[Finger, bool], t: float
    ) -> EngineOutput:
        """The other hand pinches: the cursor holds still, nothing clicks, and with this hand
        pinched too, pulling the hands apart / together zooms."""
        assert self._two is not None
        actions: list[Action] = []
        if self._mode is not Mode.TWO_HANDS:
            previous, self._mode = self._mode, Mode.TWO_HANDS
            self._candidate, self._candidate_frames = None, 0
            actions += self._on_mode_change(previous)
        if self._swipes is not None:
            self._swipes.reset()
        self._cursor_frozen = True
        self.mapper.update(features.anchor, t, frozen=True)  # keep the filter converged
        zoom = self._two.update(obs)
        if zoom:
            self._set_flash("ZOOM IN" if zoom[0].steps > 0 else "ZOOM OUT", t, 0.4)
        actions += zoom
        return EngineOutput(
            mode=Mode.TWO_HANDS,
            label=self._label(t),
            actions=actions,
            cursor_frozen=True,
            features=features,
            extended=extended,
            media_active=self._media_active,
            speed=self.speed,
            two_hands=self._two.points,
            pinched_hands=self._two.pinched,
        )

    def _stop_scroll(
        self, features: HandFeatures, extended: dict[Finger, bool], dt: float, t: float
    ) -> EngineOutput:
        """The other hand shows a stop sign: this hand's up / down movement scrolls, the cursor
        holds still and nothing clicks."""
        actions: list[Action] = []
        if not self._stop_scrolling:
            previous, self._mode = self._mode, Mode.SCROLL
            self._candidate, self._candidate_frames = None, 0
            if previous is not Mode.SCROLL:
                actions += self._on_mode_change(previous)
            self._exit_scroll()  # the scroll starts where this hand is now
            self._stop_scrolling = True
        if self._swipes is not None:
            self._swipes.reset()
        self._cursor_frozen = True
        self.mapper.update(features.anchor, t, frozen=True)  # keep the filter converged
        actions += self._scroll(features, dt)
        return EngineOutput(
            mode=Mode.SCROLL,
            label="SCROLL (other hand: stop)",
            actions=actions,
            cursor_frozen=True,
            features=features,
            extended=extended,
            scroll_origin=self._scroll_origin,
            scroll_rate=(float(self._scroll_rate[0]), float(self._scroll_rate[1])),
            media_active=self._media_active,
            speed=self.speed,
            gun_hint=self._gun_hint,
            stop_hint=self._stop_hint,
        )

    def _leave_stop_scroll(self) -> list[Action]:
        """The stop hand went down: back to pointing. A pinch made meanwhile must not click."""
        self._stop_scrolling = False
        self._exit_scroll()
        self._mode = Mode.MEDIA if self._media_active else Mode.POINTER
        self._candidate, self._candidate_frames = None, 0
        for latch in self._pinch.values():
            latch.reset()
        self._armed = dict.fromkeys(PINCH_FINGERS, False)
        return []

    def _leave_two_hands(self) -> list[Action]:
        """Back to one hand. Its fingers may still be pinched: that must not click."""
        assert self._two is not None
        self._two.reset()
        self._mode = Mode.MEDIA if self._media_active else Mode.POINTER
        self._candidate, self._candidate_frames = None, 0
        for latch in self._pinch.values():
            latch.reset()
        self._armed = dict.fromkeys(PINCH_FINGERS, False)
        return []

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
    def _classify(
        self, ext: dict[Finger, bool], joined: bool, sideways: bool, f: HandFeatures
    ) -> Mode:
        i, m, r, p = (ext[Finger.INDEX], ext[Finger.MIDDLE], ext[Finger.RING], ext[Finger.PINKY])
        if not (i or m or r or p):
            # A fist, wherever the thumb is: in a quarter of real fists its tip lies right next
            # to the folded index tip, which must not count as a pinch.
            return Mode.CLUTCH
        if sideways and i + m + r + p >= 2:
            # The whole hand turned sideways (scroll_sideways): scroll. A fist turned sideways
            # is still a fist.
            return Mode.SCROLL
        if i and m and not r and not p and joined and self.cfg.scroll_two_fingers:
            return Mode.SCROLL  # index + middle together; apart they are the slowest cursor
        if p and not (i or m or r):
            return Mode.SWITCH  # shaka: only the pinky (and the thumb) out
        # The horns need the pinky straight now, not just not yet folded: a finger gun with the
        # pinky half out (ratio ~1.0-1.13) used to zoom the screen instead of changing slides.
        pinky_straight = f.extension[Finger.PINKY] > self.cfg.finger_extended_above
        if i and p and not m and not r and pinky_straight:
            return Mode.ZOOM
        if i and p and pinky_straight and self.cfg.horns_relaxed:
            # Middle and ring only half folded still make the horns - but only when they are
            # clearly lower than index and pinky: a relaxed open hand has all four alike.
            e = f.extension
            low = min(e[Finger.INDEX], e[Finger.PINKY]) - HORNS_MARGIN
            straight = self.cfg.finger_extended_above
            if max(e[Finger.MIDDLE], e[Finger.RING]) < min(straight, low):
                return Mode.ZOOM
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
        # The index finger needs no margin: when the thumb is about as close to the index as to
        # the middle finger, that is the left click people mean. Before, nothing happened at
        # all, and that was one way clicks got "missed".
        best = min(PINCH_FINGERS, key=d.__getitem__)
        if best is not Finger.INDEX and d[Finger.INDEX] - d[best] < cfg.pinch_ambiguity_margin:
            best = Finger.INDEX
        runner_up = min((x for x in PINCH_FINGERS if x is not best), key=d.__getitem__)
        clear_winner = best is Finger.INDEX or d[runner_up] - d[best] >= cfg.pinch_ambiguity_margin
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
        # long as the pinch, like a finger on a real mouse button. The cursor stays where the
        # press landed until the hand has clearly moved (drag_start palm lengths), however long
        # the pinch lasts: a still hand always makes a clean click, never a micro-drag. Then
        # the cursor follows with the button down, starting from where the press landed.
        if rose[Finger.INDEX]:
            actions.append(Press("left"))
            self._left = LeftState.PENDING
            self._pending_origin = self.mapper.target
            self._press_anchor = f.anchor_px
        elif self._left is not LeftState.IDLE and fell[Finger.INDEX]:
            actions.append(Release("left"))
            if self._left is LeftState.PENDING:
                self._set_flash("LEFT CLICK", t)
                self._settle_until = t + cfg.click_settle  # a second pinch = a double click
                self._press_anchor = f.anchor_px  # the settle ends when the hand moves on
            else:
                self._set_flash("DROP", t)
            self._left = LeftState.IDLE
        elif self._left is LeftState.PENDING and self._hand_travel(f) >= cfg.drag_start:
            self._left = LeftState.DRAG
            self.mapper.hold_at(self._pending_origin)  # no jump: the drag starts at the press

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
        settling = t < self._settle_until and self._hand_travel(f) < cfg.drag_start
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
        move = self.mapper.update(f.anchor, t, frozen=frozen, speed=self.speed)
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

    def _hand_travel(self, f: HandFeatures) -> float:
        """How far the hand has moved since the press (or the click), in palm lengths: the
        same on any screen and at any distance from the camera."""
        if self._press_anchor is None:
            return 0.0
        dx = f.anchor_px[0] - self._press_anchor[0]
        dy = f.anchor_px[1] - self._press_anchor[1]
        return math.hypot(dx, dy) / f.palm_scale

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
