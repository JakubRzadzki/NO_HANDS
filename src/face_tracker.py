"""MediaPipe Face Landmarker wrapper plus the eye and head measurements used by --eyes / --head.

Which eye is "left" is decided by position in the image, never by landmark or blendshape names:
in the mirrored (selfie) frame the user's left eye is the one on the image's left, exactly like
in a mirror. That keeps winks correct whether or not the frame is mirrored.

The head pose comes from MediaPipe's facial transformation matrix: a metric 3D fit of a canonical
face to all 478 landmarks, giving real angles in degrees and the head position in cm. It is far
steadier than any single landmark. (An earlier 2D shortcut, the nose offset from the eyes, gave
the wrong sign on real photos as soon as the head was tilted, which is why head control
misbehaved.)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from config import FaceConfig
from src.hand_tracker import FloatArray, ensure_model

NOSE_TIP = 1
IRIS_CENTERS = (468, 473)
# Eye outlines in MediaPipe's 478-point face mesh: two corners, two upper- and two lower-lid
# points. Verified on real photos: open eyes give an aspect ratio of about 0.29-0.40.
EYE_OUTLINES: tuple[dict[str, tuple[int, int]], ...] = (
    {"corners": (33, 133), "upper": (160, 158), "lower": (144, 153)},
    {"corners": (263, 362), "upper": (387, 385), "lower": (373, 380)},
)


@dataclass(frozen=True)
class FaceObservation:
    timestamp: float
    norm: FloatArray  # (478, 3) normalized MediaPipe coordinates
    px: FloatArray  # (478, 2) pixel coordinates
    frame_size: tuple[int, int]
    # From the user's point of view: yaw > 0 = turned right, pitch > 0 = head down (degrees).
    head: tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, pitch, roll
    head_position: tuple[float, float, float] = (0.0, 0.0, 60.0)  # cm: right, up, distance

    @classmethod
    def from_normalized(
        cls,
        norm: npt.ArrayLike,
        frame_size: tuple[int, int],
        timestamp: float,
        head: tuple[float, float, float] = (0.0, 0.0, 0.0),
        head_position: tuple[float, float, float] = (0.0, 0.0, 60.0),
    ) -> FaceObservation:
        norm_arr = np.asarray(norm, dtype=np.float64).reshape(-1, 3)
        width, height = frame_size
        px = norm_arr[:, :2] * np.array([width, height], dtype=np.float64)
        return cls(timestamp, norm_arr, px, frame_size, head, head_position)


@dataclass(frozen=True)
class EyeMeasure:
    center: tuple[float, float]  # px
    width: float  # px, corner to corner
    aspect: float  # lid opening / width: ~0.3 open, < 0.1 closed
    iris: tuple[float, float]  # iris offset from the eye centre in eye widths (+x right, +y down)


def head_from_matrix(
    matrix: npt.ArrayLike, mirrored: bool
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(yaw, pitch, roll) in degrees and (x, y, distance) in cm from the 4x4 face transform.

    The transform maps the canonical face into the camera frame (x right, y up, z towards the
    camera); its third column is where the face points. In a mirrored frame the image's right
    is the user's right, so the frame's angles already read from the user's point of view."""
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rotation, forward = m[:3, :3], m[:3, 2]
    yaw = math.degrees(math.atan2(forward[0], forward[2]))
    pitch = math.degrees(math.asin(min(max(-forward[1], -1.0), 1.0)))  # + = facing down
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[1, 1]))
    x, y, z = (float(v) for v in m[:3, 3])
    if not mirrored:
        yaw, roll, x = -yaw, -roll, -x
    return (yaw, pitch, roll), (x, y, -z)


def _unit_axes(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Unit vector from a to b, its perpendicular (pointing down in the image), and |b - a|."""
    length = max(float(np.linalg.norm(b - a)), 1e-6)
    u = (b - a) / length
    return u, np.array([-u[1], u[0]]), length


def measure_eye(px: FloatArray, outline: dict[str, tuple[int, int]]) -> EyeMeasure:
    c0, c1 = px[outline["corners"][0]], px[outline["corners"][1]]
    left, right = (c0, c1) if c0[0] <= c1[0] else (c1, c0)  # axis runs left -> right in the image
    u, v, width = _unit_axes(left, right)
    center = (left + right) / 2.0
    (u0, u1), (l0, l1) = outline["upper"], outline["lower"]
    opening = (np.linalg.norm(px[u0] - px[l0]) + np.linalg.norm(px[u1] - px[l1])) / 2.0
    iris_index = min(IRIS_CENTERS, key=lambda i: float(np.linalg.norm(px[i] - center)))
    offset = px[iris_index] - center
    return EyeMeasure(
        center=(float(center[0]), float(center[1])),
        width=width,
        aspect=float(opening) / width,
        iris=(float(offset @ u) / width, float(offset @ v) / width),
    )


def measure_eyes(obs: FaceObservation, mirrored: bool = True) -> tuple[EyeMeasure, EyeMeasure]:
    """(user's left eye, user's right eye)."""
    a, b = (measure_eye(obs.px, outline) for outline in EYE_OUTLINES)
    image_left, image_right = (a, b) if a.center[0] <= b.center[0] else (b, a)
    return (image_left, image_right) if mirrored else (image_right, image_left)


def gaze_feature(left: EyeMeasure, right: EyeMeasure) -> tuple[float, float]:
    """Mean iris offset of both eyes: where the eyes point inside the head."""
    return (left.iris[0] + right.iris[0]) / 2.0, (left.iris[1] + right.iris[1]) / 2.0


class FaceTracker:
    """BGR frame in, ``FaceObservation`` (or None) out. Blendshapes are not computed: eye
    openness comes from the landmarks, which is cheaper and does not depend on naming."""

    def __init__(self, cfg: FaceConfig, mirrored: bool = True) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = ensure_model(cfg.model_path, cfg.model_url)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.mirrored = mirrored
        self._mp: Any = mp
        self._landmarker: Any = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        self.last_inference_ms = 0.0

    def process(self, frame_bgr: npt.NDArray[Any], timestamp: float) -> FaceObservation | None:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int(timestamp * 1000.0), self._last_ts_ms + 1)  # strictly increasing
        self._last_ts_ms = ts_ms
        started = time.perf_counter()
        result = self._landmarker.detect_for_video(image, ts_ms)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        if not result.face_landmarks:
            return None
        norm = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float64)
        head, position = (0.0, 0.0, 0.0), (0.0, 0.0, 60.0)
        if result.facial_transformation_matrixes:
            head, position = head_from_matrix(
                result.facial_transformation_matrixes[0], self.mirrored
            )
        return FaceObservation.from_normalized(norm, (width, height), timestamp, head, position)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
