"""MediaPipe Face Landmarker wrapper plus the eye and head measurements used by --eyes / --head.

Which eye is "left" is decided by position in the image, never by landmark or blendshape names:
in the mirrored (selfie) frame the user's left eye is the one on the image's left, exactly like
in a mirror. That keeps winks correct whether or not the frame is mirrored.
"""

from __future__ import annotations

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

    @classmethod
    def from_normalized(
        cls, norm: npt.ArrayLike, frame_size: tuple[int, int], timestamp: float
    ) -> FaceObservation:
        norm_arr = np.asarray(norm, dtype=np.float64).reshape(-1, 3)
        width, height = frame_size
        px = norm_arr[:, :2] * np.array([width, height], dtype=np.float64)
        return cls(timestamp, norm_arr, px, frame_size)


@dataclass(frozen=True)
class EyeMeasure:
    center: tuple[float, float]  # px
    width: float  # px, corner to corner
    aspect: float  # lid opening / width: ~0.3 open, < 0.1 closed
    iris: tuple[
        float, float
    ]  # iris centre offset from the eye centre, in eye widths (+x right, +y down)


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


def head_pose(
    obs: FaceObservation, eyes: tuple[EyeMeasure, EyeMeasure], mirrored: bool = True
) -> tuple[float, float]:
    """Head (yaw, pitch) as the nose tip's offset from the eyes' midpoint, in eye distances.

    Measured in the face's own frame (along and across the line through the eyes), so tilting
    the head sideways does not leak into yaw or pitch, and dividing by the eye distance makes
    it independent of how far the user sits. yaw > 0: turned to the user's right;
    pitch > 0: nose lower than usual (head down).
    """
    p, q = np.array(eyes[0].center), np.array(eyes[1].center)
    a, b = (p, q) if p[0] <= q[0] else (q, p)
    u, v, distance = _unit_axes(a, b)
    rel = obs.px[NOSE_TIP] - (a + b) / 2.0
    yaw = float(rel @ u) / distance  # + = towards the image's right
    pitch = float(rel @ v) / distance
    return (yaw if mirrored else -yaw), pitch


class FaceTracker:
    """BGR frame in, ``FaceObservation`` (or None) out. Blendshapes are not computed: eye
    openness comes from the landmarks, which is cheaper and does not depend on naming."""

    def __init__(self, cfg: FaceConfig) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = ensure_model(cfg.model_path, cfg.model_url)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
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
        return FaceObservation.from_normalized(norm, (width, height), timestamp)

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
