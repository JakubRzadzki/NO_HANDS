"""A simulated laptop user for the gaze tests: where they look and how their head sits -> the
gaze features MediaPipe would report (see src.face_tracker.gaze_features).

Geometry in cm, oriented like the screen: x right, y down, z = distance in front of the camera.
The camera sits at the origin, in the screen plane, 1 cm above the screen's top edge. The eyes
turn inside the head: eye angle = gaze angle - head angle, and the iris shifts by
eyeball radius * sin(eye angle), measured in eye widths like MediaPipe's iris offset.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.gaze import CalibrationSession, Sample

Pose = dict[str, float]  # yaw, pitch (deg), x, y, distance (cm)
REST: Pose = {"yaw": 0.0, "pitch": 0.0, "x": 0.0, "y": 8.0, "distance": 60.0}


@dataclass
class Viewer:
    screen_px: tuple[int, int] = (1920, 1080)
    screen_cm: tuple[float, float] = (34.5, 19.4)  # a 15.6" 16:9 laptop screen
    screen_top: float = 1.0  # cm from the camera down to the top edge of the screen
    eye_radius: float = 0.4  # eyeball radius in eye widths (12 mm / 30 mm)
    iris_bias: tuple[float, float] = (0.01, -0.07)  # measured on real faces: iris sits high
    iris_noise: float = 0.0  # eye widths, per frame (both eyes averaged)
    angle_noise: float = 0.0  # degrees
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def point_cm(self, target_px: tuple[float, float]) -> tuple[float, float]:
        (w, h), (wc, hc) = self.screen_px, self.screen_cm
        return (target_px[0] / (w - 1) - 0.5) * wc, self.screen_top + target_px[1] / (h - 1) * hc

    def features(self, target_px: tuple[float, float], **pose: float) -> Sample:
        p = {**REST, **pose}
        tx, ty = self.point_cm(target_px)
        dx, dy, d = tx - p["x"], ty - p["y"], p["distance"]
        gaze_yaw = math.degrees(math.atan2(dx, d))
        gaze_pitch = math.degrees(math.atan2(dy, math.hypot(dx, d)))
        eye_yaw = math.radians(gaze_yaw - p["yaw"])
        eye_pitch = math.radians(gaze_pitch - p["pitch"])
        iris = np.array(
            [
                self.eye_radius * math.sin(eye_yaw) + self.iris_bias[0],
                self.eye_radius * math.sin(eye_pitch) + self.iris_bias[1],
            ]
        )
        angles = np.array([p["yaw"], p["pitch"]])
        if self.iris_noise > 0:
            iris = iris + self.rng.normal(0.0, self.iris_noise, 2)
        if self.angle_noise > 0:
            angles = angles + self.rng.normal(0.0, self.angle_noise, 2)
        features = (*iris.tolist(), *angles.tolist(), p["x"], p["y"])
        return features, d

    def random_pose(self, rng: np.random.Generator) -> Pose:
        """A head somewhere a laptop user's head ends up during normal use."""
        return {
            "yaw": rng.uniform(-8, 8),
            "pitch": rng.uniform(-6, 6),
            "x": rng.uniform(-4, 4),
            "y": 8 + rng.uniform(-2, 2),
            "distance": rng.uniform(50, 70),
        }


def head_tour(t: float) -> Pose:
    """The head moving around while the eyes stay on one dot (calibration phase 2)."""
    return {
        "yaw": 10 * math.sin(2 * math.pi * t / 3.0),
        "pitch": 7 * math.sin(2 * math.pi * t / 4.3 + 1.0),
        "x": 3 * math.sin(2 * math.pi * t / 5.1),
        "y": 8 + 2 * math.sin(2 * math.pi * t / 3.7),
        "distance": 60 + 5 * math.sin(2 * math.pi * t / 6.0),
    }


def follow_arrows(session: CalibrationSession, previous: Pose, fps: float) -> Pose:
    """A user in the head phase: turns the head towards the arrow, ~10 degrees, at 25 deg/s.
    Turning the head also moves the eyes a little (they sit in front of the neck)."""
    hint = session.head_hint
    goal = (
        {"yaw": 0.0, "pitch": 0.0}
        if hint is None
        else {
            "yaw": 10.0 * hint[1][0],
            "pitch": 8.0 * hint[1][1],
        }
    )
    pose = dict(previous)
    for axis in ("yaw", "pitch"):
        step = goal[axis] - pose.get(axis, 0.0)
        pose[axis] = pose.get(axis, 0.0) + max(-25.0 / fps, min(25.0 / fps, step))
    pose["x"] = 10.0 * math.sin(math.radians(pose["yaw"]))
    pose["y"] = REST["y"] + 10.0 * math.sin(math.radians(pose["pitch"]))
    return pose


def calibrate(
    session: CalibrationSession,
    viewer: Viewer,
    head: Callable[[float, str], Pose] | None = None,
    fps: float = 30.0,
    glitch: Callable[[float], bool] | None = None,
) -> None:
    """Plays a user who looks at every dot. ``head(t, phase)`` gives the head pose (default:
    still, following the arrows in the head phase); ``glitch(t)`` marks frames with garbage
    features."""
    t = 0.0
    pose: Pose = dict(REST)
    while not session.done:
        target = session.target
        assert target is not None
        if head is not None:
            pose = head(t, session.phase)
        elif session.phase == "head":
            pose = follow_arrows(session, pose, fps)
        else:
            pose = dict(REST)
        sample = viewer.features(target, **pose)
        if glitch is not None and glitch(t):
            features, distance = sample
            sample = ((features[0] + 0.3, features[1] - 0.2, *features[2:]), distance)
        session.update(t, sample)
        t += 1.0 / fps
