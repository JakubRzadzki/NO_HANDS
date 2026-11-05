"""--head: quick head movements press the arrow keys.

    quick turn left / right ......... Left / Right arrow
    quick tilt up / nod down ........ Up / Down arrow
    eyes closed >= 1.5 s ............ pause / resume (resuming re-learns the neutral pose)

The pose is MediaPipe's 3D head pose in degrees, relative to a neutral pose learned at start
(0.5 s of holding still) that slowly follows your posture while the head rests in the centre.

A move fires like a swipe: the head has to get from the neutral zone past the threshold within
``gesture_time``. Turning slowly to look at something else, or a posture change, never presses
anything. After firing, the head must come back to the neutral zone before the next move, so
returning to the centre never fires the opposite arrow. Holding the head there repeats the key
only if ``repeat_delay`` > 0. Tilting the head sideways (roll) is ignored.

A quick look away (someone calls you) is a quick turn too: pause first (eyes closed 1.5 s).
"""

from __future__ import annotations

import math

import numpy as np

from config import FaceConfig, HeadConfig
from src.eye_engine import ClosureDetector, FaceMode, FaceOutput
from src.face_tracker import FaceObservation, measure_eyes
from src.gaze import FloatArray
from src.os_controller import Action, KeyTap

NEUTRAL_FRAMES = 15  # frames (~0.5 s) of a still head that define the neutral pose


class HeadArrows:
    def __init__(self, cfg: HeadConfig, face: FaceConfig, mirrored: bool = True) -> None:
        self.cfg = cfg
        self.face = face
        self.mirrored = mirrored  # only for telling the eyes apart; the pose is already mirrored
        self._closure = ClosureDetector(face)
        self._samples: list[tuple[float, float]] = []
        self._neutral: FloatArray | None = None
        self._engaged: str | None = None  # the arrow fired for the current move
        self._armed = False  # True once the head was in the neutral zone (a move may start)
        self._centred_at = -math.inf  # last time the head was in the neutral zone
        self._next_repeat = math.inf
        self._paused = False
        self._flash = ""
        self._flash_until = -math.inf

    @property
    def paused(self) -> bool:
        return self._paused

    def reset(self) -> None:
        """Re-learn the neutral pose (resume, global pause)."""
        self._closure.cancel()
        self._samples.clear()
        self._neutral = None
        self._engaged = None
        self._armed = False

    def update(self, obs: FaceObservation | None, t: float) -> FaceOutput:
        if obs is None:
            self._closure.cancel()
            self._engaged = None  # a returning face must come back to neutral first
            self._armed = False
            mode = FaceMode.PAUSED if self._paused else FaceMode.NO_FACE
            return FaceOutput(mode, self._label(t, mode))

        eyes = measure_eyes(obs, self.mirrored)
        episode = self._closure.update(*eyes, t)
        if episode is not None and episode[0] == "both" and episode[1] >= self.face.pause_hold:
            self._paused = not self._paused
            if not self._paused:
                self.reset()
            self._set_flash("PAUSED" if self._paused else "RESUMED", t, 1.0)

        pose = np.array(obs.head[:2], dtype=np.float64)  # yaw (+ right), pitch (+ down), deg
        actions: list[Action] = []
        offset: tuple[float, float] | None = None
        if self._neutral is None:
            self._samples.append((float(pose[0]), float(pose[1])))
            if len(self._samples) >= NEUTRAL_FRAMES:
                self._neutral = np.median(np.asarray(self._samples), axis=0)
        else:
            delta = pose - self._neutral
            offset = (float(delta[0]), float(delta[1]))
            if not self._paused:
                actions = self._arrows(offset, t)
            if self._engaged is None and self._centred(offset):
                # Follow slow posture drift, but only while the head is at rest in the centre.
                self._neutral += self.cfg.neutral_adapt * (pose - self._neutral)

        mode = FaceMode.PAUSED if self._paused else FaceMode.HEAD
        label = self._label(t, mode) if self._neutral is not None else "HOLD STILL"
        return FaceOutput(
            mode,
            label,
            actions,
            eyes,
            self._closure.openness,
            self._closure.closed,
            feature=offset,
        )

    def _centred(self, offset: tuple[float, float]) -> bool:
        return abs(offset[0]) < self.cfg.yaw_release and abs(offset[1]) < self.cfg.pitch_release

    def _arrows(self, offset: tuple[float, float], t: float) -> list[Action]:
        cfg = self.cfg
        if self._centred(offset):
            self._engaged = None  # back at neutral: ready for the next move
            self._armed = True
            self._centred_at = t
            return []
        if self._engaged is not None:
            if cfg.repeat_delay > 0 and t >= self._next_repeat:
                self._next_repeat = t + cfg.repeat_interval
                return [KeyTap(self._engaged)]
            return []
        if not self._armed:
            return []
        if t - self._centred_at > cfg.gesture_time:
            self._armed = False  # too slow for a gesture: looking around or changing posture
            return []
        # Scale each axis by its own threshold so the dominant direction wins on diagonals.
        yaw, pitch = offset[0] / cfg.yaw_threshold, offset[1] / cfg.pitch_threshold
        if max(abs(yaw), abs(pitch)) < 1.0:
            return []
        if abs(yaw) >= abs(pitch):
            direction = "right" if yaw > 0 else "left"
        else:
            direction = "down" if pitch > 0 else "up"
        self._engaged = direction
        self._armed = False
        self._next_repeat = t + cfg.repeat_delay
        self._set_flash(direction.upper(), t)
        return [KeyTap(direction)]

    def _set_flash(self, text: str, t: float, duration: float = 0.4) -> None:
        self._flash, self._flash_until = text, t + duration

    def _label(self, t: float, mode: FaceMode) -> str:
        return self._flash if t < self._flash_until else mode.value
