"""Media-mode gestures: swipe = previous/next, hand turned like a knob = volume, fist = play/pause.

Media mode is separate from mouse mode (GestureEngine toggles it with the "shaka") because the
same movements already drive the cursor: moving the open hand sideways *is* moving the mouse,
so it cannot also mean "next track".

* Swipe: the *wrist* travels >= swipe_distance palm lengths sideways within swipe_window. The
  wrist is used, not the knuckles, because turning the hand (the volume knob) swings the
  knuckles around the wrist while the wrist itself barely moves.
* Knob: the angle of the wrist -> middle-knuckle line, relative to the hand's neutral pose. A
  wrist only turns about +-70 degrees, so a position-based knob would run out of range, and
  turning back would undo the change. It works like a spring-loaded jog dial instead: the
  further past the dead zone, the faster the volume changes; back to neutral stops it.
* Fist: play/pause, once per closing of the hand.
"""

from __future__ import annotations

import math
from collections import deque

from config import MediaConfig
from src.hand_tracker import MIDDLE_MCP, WRIST, HandObservation
from src.os_controller import Action, KeyTap

FIST_CONFIRM_FRAMES = 3  # a fist must be seen this many frames in a row (blur glitches)


def wrap_degrees(angle: float) -> float:
    """Map an angle difference to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


class MediaGestures:
    def __init__(self, cfg: MediaConfig) -> None:
        self.cfg = cfg
        self._history: deque[tuple[float, float, float]] = deque()  # (t, wrist x, wrist y), px
        self._knob_ref: float | None = None
        self._knob_accum = 0.0
        self._last_swipe = -math.inf
        self._last_play = -math.inf
        self._fist_frames = 0
        self.knob_offset: float | None = None  # degrees from neutral, + = clockwise (HUD)
        self.volume_rate = 0.0  # key presses per second, + = louder (HUD)

    def reset(self) -> None:
        """Forget motion history (hand lost, mode left). The next steady pose is the neutral."""
        self._history.clear()
        self._knob_ref = None
        self._knob_accum = 0.0
        self._fist_frames = 0
        self.knob_offset = None
        self.volume_rate = 0.0

    def update(
        self, obs: HandObservation, fist: bool, t: float, dt: float
    ) -> tuple[list[Action], str | None]:
        """Returns the actions to send and the name of a discrete event for the HUD, if any."""
        if fist:
            return self._fist(t)
        self._fist_frames = 0

        wrist_x, wrist_y = float(obs.px[WRIST, 0]), float(obs.px[WRIST, 1])
        self._history.append((t, wrist_x, wrist_y))
        while t - self._history[0][0] > self.cfg.swipe_window:
            self._history.popleft()
        t0, x0, y0 = self._history[0]
        dx = (wrist_x - x0) / obs.palm_scale  # palm lengths; + = to the right (mirrored view)
        dy = (wrist_y - y0) / obs.palm_scale
        speed = math.hypot(dx, dy) / max(t - t0, 1e-3)  # palm lengths per second

        swipe = self._swipe(dx, dy, t)
        if swipe is not None:
            return [swipe[0]], swipe[1]
        return self._knob(obs, speed, dt), None

    def _swipe(self, dx: float, dy: float, t: float) -> tuple[KeyTap, str] | None:
        cfg = self.cfg
        if t - self._last_swipe < cfg.swipe_cooldown:
            return None  # also swallows the hand travelling back after a swipe
        if abs(dx) < cfg.swipe_distance or abs(dx) < 2.0 * abs(dy):
            return None
        self._last_swipe = t
        self._history.clear()
        self._knob_ref = None  # the hand moved: re-zero the knob once it is steady again
        right = dx > 0
        if cfg.swipe_action == "arrows":
            return (KeyTap("right"), "RIGHT >>") if right else (KeyTap("left"), "<< LEFT")
        return (KeyTap("next_track"), "NEXT >>") if right else (KeyTap("previous_track"), "<< PREV")

    def _knob(self, obs: HandObservation, speed: float, dt: float) -> list[Action]:
        cfg = self.cfg
        if speed > cfg.knob_max_wrist_speed:
            # A moving hand is swiping or being repositioned, not turning a knob.
            self.volume_rate, self._knob_accum, self.knob_offset = 0.0, 0.0, None
            return []
        vx, vy = obs.px[MIDDLE_MCP, :2] - obs.px[WRIST, :2]
        # Image y points down, so a growing angle is clockwise on screen. With the mirrored
        # camera view that is also clockwise for the user ("turning right").
        angle = math.degrees(math.atan2(vy, vx))
        if self._knob_ref is None:
            self._knob_ref, self._knob_accum = angle, 0.0
        offset = wrap_degrees(angle - self._knob_ref)
        if cfg.knob_invert:
            offset = -offset
        self.knob_offset = offset
        excess = abs(offset) - cfg.knob_deadzone_deg
        rate = 0.0
        if excess > 0:
            rate = math.copysign(min(cfg.knob_rate * excess, cfg.knob_max_rate), offset)
        self.volume_rate = rate
        self._knob_accum += rate * dt  # presses accumulate with time, not per frame
        presses = int(self._knob_accum)
        self._knob_accum -= presses
        if presses > 0:
            return [KeyTap("volume_up", presses)]
        if presses < 0:
            return [KeyTap("volume_down", -presses)]
        return []

    def _fist(self, t: float) -> tuple[list[Action], str | None]:
        self._history.clear()
        self._knob_ref = None
        self.volume_rate, self.knob_offset = 0.0, None
        self._fist_frames += 1
        # Fires exactly once per closing (== not >=), so holding the fist never repeats it.
        if (
            self._fist_frames == FIST_CONFIRM_FRAMES
            and t - self._last_play >= self.cfg.play_pause_cooldown
        ):
            self._last_play = t
            return [KeyTap("play_pause")], "PLAY/PAUSE"
        return [], None
