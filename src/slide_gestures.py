"""--slides: a presenter's remote in the hand. No cursor.

    quick stroke left / right .............. Left / Right arrow (previous / next slide)
    quick pinch thumb + index .............. Right arrow (next slide), like a click would
    pinch thumb + index, move up / down .... zoom in / out (Ctrl + mouse wheel)
    pinch both hands, pull apart / together  zoom in / out (src/two_hands.py)

The strokes come from ``src.swipes.SwipeDetector`` (see there for what counts as a swipe and
what does not). Here a swipe counts the moment the stroke ends; the hand does not have to come
back, because nothing else is listening to it.

Vertical strokes do nothing. As zoom gestures they also fired on ordinary up/down "beat"
gestures made while talking, in every simulated case. Zoom therefore needs a pinch, which
nobody makes by accident. A pinch that is let go without moving is "next slide": people who
advance slides with a mouse click pinch by habit, and a key press cannot click on whatever the
cursor happens to be over.
"""

from __future__ import annotations

import math
from enum import Enum

from config import GestureConfig, SlidesConfig
from src.gesture_engine import (
    EngineOutput,
    Finger,
    HandFeatures,
    HysteresisLatch,
    Mode,
    is_pinch_shape,
)
from src.gun import GunTrigger, gun_direction
from src.hand_tracker import HandObservation
from src.os_controller import Action, KeyTap, Zoom
from src.swipes import Stroke, SwipeDetector
from src.two_hands import TwoHandZoom

MAX_GAP = 0.15  # s: tracking may drop out this long mid-stroke (motion blur) without a reset
ACTIONS = {"right": "RIGHT >>", "left": "<< LEFT"}


class Phase(str, Enum):
    WAITING = "WAITING"  # the hand just appeared: strokes do not count yet
    READY = "READY"
    STROKE = "STROKE"  # a fast movement is under way
    ZOOM = "ZOOM"  # pinched: vertical movement zooms
    TWO_HANDS = "TWO HANDS"  # both hands in view: pinch both, pull apart / together = zoom


class SlideGestures:
    def __init__(self, cfg: SlidesConfig, gesture: GestureConfig) -> None:
        self.cfg = cfg
        self.gesture = gesture
        self._swipes = SwipeDetector(cfg)
        self._pinch = HysteresisLatch(
            gesture.pinch_enter,
            gesture.pinch_exit,
            confirm_frames=gesture.pinch_confirm_frames,
            release_frames=gesture.pinch_release_frames,
        )
        self._armed = False  # thumb and index were seen apart since the hand appeared
        self._zoom_ref: float | None = None  # hand height (px) of the last zoom detent
        self._zoomed = False  # this pinch already zoomed: letting go is not "next slide"
        self._palm = 1.0
        self._last_seen: float | None = None
        self._flash = ""
        self._flash_until = -math.inf
        self._two = TwoHandZoom(gesture) if gesture.two_hand_zoom else None
        self._two_active = False  # the other hand pinches: two-hand zoom
        self._guns = (GunTrigger(gesture), GunTrigger(gesture))  # this hand, the other hand

    @property
    def phase(self) -> Phase:
        if self._two_active:
            return Phase.TWO_HANDS
        if self._pinch.active:
            return Phase.ZOOM
        if self._swipes.moving:
            return Phase.STROKE
        return Phase.READY if self._swipes.ready else Phase.WAITING

    @property
    def last_stroke(self) -> Stroke | None:
        return self._swipes.last_stroke

    def reset(self) -> list[Action]:
        """Forget motion state (pause, hand lost). Nothing is ever held, so nothing to release."""
        self._swipes.reset()
        self._pinch.reset()
        if self._two is not None:
            self._two.reset()
        self._two_active = False
        for gun in self._guns:
            gun.reset()
        self._armed = False
        self._zoom_ref = None
        self._last_seen = None
        return []

    def update(
        self, obs: HandObservation | None, t: float, second: HandObservation | None = None
    ) -> EngineOutput:
        """``second``: the other hand, when two are in view (two-hand zoom)."""
        if obs is None:
            if self._last_seen is None or t - self._last_seen > MAX_GAP:
                self.reset()
                return EngineOutput(Mode.NO_HAND, self._label(t, "NO HAND"))
            return EngineOutput(Mode.SLIDES, self._label(t, "TRACKING LOST"))
        self._last_seen = t

        g = self.gesture
        features = HandFeatures.from_observation(obs, depth_weight=g.pinch_depth_weight)
        not_curled = {f: features.extension[f] > g.finger_curled_below for f in Finger}
        centroid = obs.px[:, :2].mean(axis=0)
        self._palm = obs.palm_scale

        if self._two is not None:
            # A pinch of this hand waits a little longer while a second hand is in view: both
            # hands pinch at about the same time for a zoom, and this one must not turn the
            # slide first. Only a pinching second hand counts; any other changes nothing.
            delay = self.gesture.two_hand_click_delay_frames if second is not None else 0
            self._pinch.confirm_frames = self.gesture.pinch_confirm_frames + delay
            was_two, self._two_active = self._two_active, self._two.second_pinched(second)
            if self._two_active:
                return self._two_hands(obs, features, t)
            self._two_active = False
            if was_two:  # back to one hand, maybe still pinched: that is not "next slide"
                self._pinch.reset()
                self._armed = False

        # Finger guns: either hand, pointing right = next slide, left = previous.
        aim = gun_direction(obs, features, g) if g.gun_slides else None
        other_aim = None
        if second is not None and g.gun_slides:
            other = HandFeatures.from_observation(second, depth_weight=g.pinch_depth_weight)
            other_aim = gun_direction(second, other, g)
        actions: list[Action] = []
        for gun, direction in zip(self._guns, (aim, other_aim), strict=True):
            fired = gun.update(direction)
            if fired is not None:
                actions.append(KeyTap(fired))
                self._set_flash(ACTIONS[fired], t)
        if aim is not None:
            self._swipes.cancel()  # forming the gun is not a swipe
            return EngineOutput(
                Mode.SLIDES,
                self._label(t, "GUN"),
                actions,
                features=features,
                extended=not_curled,
                trail=self._swipes.trail,
            )

        actions += self._pinch_actions(features, float(centroid[1]), t)
        if self._pinch.active:
            self._swipes.cancel()  # a pinch is for zooming: no swipes meanwhile
        else:
            is_open = sum(not_curled.values()) >= self.cfg.min_fingers
            direction = self._swipes.update(
                t, centroid, obs.palm_scale, is_open, float(obs.frame_size[0])
            )
            if direction is not None:
                actions.append(KeyTap(direction))
                self._set_flash(ACTIONS[direction], t)

        return EngineOutput(
            Mode.SLIDES,
            self._label(t, self.phase.value),
            actions,
            features=features,
            extended=not_curled,
            trail=self._swipes.trail,
            stroke=self._swipes.stroke,
        )

    def _two_hands(self, obs: HandObservation, features: HandFeatures, t: float) -> EngineOutput:
        assert self._two is not None
        self._swipes.cancel()
        self._pinch.reset()  # this hand's pinch now belongs to the zoom: no "next slide" later
        self._zoom_ref = None
        zoom = self._two.update(obs)
        if zoom:
            self._set_flash("ZOOM IN" if zoom[0].steps > 0 else "ZOOM OUT", t, 0.4)
        return EngineOutput(
            Mode.SLIDES,
            self._label(t, Phase.TWO_HANDS.value),
            list(zoom),
            features=features,
            two_hands=self._two.points,
            pinched_hands=self._two.pinched,
        )

    # -- pinch: next slide, or zoom ------------------------------------------------------------
    def _pinch_actions(self, f: HandFeatures, y: float, t: float) -> list[Action]:
        g = self.gesture
        d = f.pinch[Finger.INDEX]
        if d > g.pinch_exit:
            self._armed = True  # a hand that appears already pinched does nothing
        before = self._pinch.active
        if before:
            self._pinch.update(d)
        else:
            shaped = is_pinch_shape(f, Finger.INDEX, g)  # an open hand, not a fist
            self._pinch.update(d if self._armed and shaped else math.inf)

        if self._pinch.active and not before:
            self._zoom_ref, self._zoomed = y, False
            return []
        if before and not self._pinch.active:
            self._zoom_ref = None
            if self._zoomed:
                return []
            self._set_flash(ACTIONS["right"], t)
            return [KeyTap("right")]  # a pinch let go without moving: next slide
        if not self._pinch.active or self._zoom_ref is None:
            return []

        step = self.cfg.zoom_travel * self._palm
        steps = int((self._zoom_ref - y) / step)  # + = the hand moved up = zoom in
        if steps == 0:
            return []
        steps = max(-1, min(1, steps))  # one wheel tick per frame at most: no zoom jumps
        self._zoom_ref -= steps * step  # whole detents: noise cannot flicker in and out
        self._zoomed = True
        self._set_flash("ZOOM IN" if steps > 0 else "ZOOM OUT", t, 0.4)
        return [Zoom(steps)]

    def _set_flash(self, text: str, t: float, duration: float = 0.6) -> None:
        self._flash, self._flash_until = text, t + duration

    def _label(self, t: float, default: str) -> str:
        return self._flash if t < self._flash_until else default
