"""MediaPipe Hand Landmarker wrapper (Tasks API).

The legacy ``mp.solutions.hands`` API no longer exists in current MediaPipe releases (1.x ships
only the Tasks API), so this module drives ``HandLandmarker`` in VIDEO mode: synchronous,
deterministic, and it reuses the previous frame's hand box for tracking instead of running the
palm detector on every frame.

Far hands get a second look: MediaPipe's palm detector sees the whole frame shrunk to 192 px,
where a hand far from the camera is too small to find. When the normal search finds nothing,
a second landmarker searches the centre crop of the frame (``TrackerConfig.zoom_crop``), which
it sees magnified. In a test with real hand photos at several sizes this found a hand in ~37%
more frames, almost all of them small hands. What was measured and *not* adopted: MediaPipe's
metric "world" landmarks classify fingers worse than the image landmarks used here (86 vs 89
of 92 finger states on hands tilted by up to 50 degrees, and a fist only just counts as a fist).
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from config import TrackerConfig

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
NUM_LANDMARKS = 21
# With no hand anywhere for a while (hands on the keyboard), the centre-crop search runs only on
# every 3rd frame: it would otherwise double the CPU time spent looking at an empty scene.
ZOOM_IDLE_AFTER = 30  # frames without a hand
ZOOM_IDLE_EVERY = 3

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),  # palm edge
)  # fmt: skip


@dataclass(frozen=True)
class HandObservation:
    """One detected hand in one frame.

    ``norm``: (21, 3) MediaPipe output. x, y are fractions of frame width/height; z is depth
    relative to the wrist, on roughly the same scale as x.

    ``px``: (21, 3) the same points in *isotropic pixel space*: (x * W, y * H, z * W).

    All geometry uses ``px``. In normalized coordinates one unit of x is W pixels but one unit of
    y is only H pixels, so on a 16:9 frame equal physical gaps differ by 1.78x depending on
    direction, and ``hypot(dx, dy)`` on raw normalized values changes when the hand rotates.
    """

    timestamp: float
    norm: FloatArray
    px: FloatArray
    frame_size: tuple[int, int]
    handedness: str = "Unknown"
    score: float = 1.0

    @classmethod
    def from_normalized(
        cls,
        norm: npt.ArrayLike,
        frame_size: tuple[int, int],
        timestamp: float,
        handedness: str = "Unknown",
        score: float = 1.0,
    ) -> HandObservation:
        norm_arr = np.asarray(norm, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
        width, height = frame_size
        px = norm_arr * np.array([width, height, width], dtype=np.float64)
        return cls(timestamp, norm_arr, px, frame_size, handedness, score)

    @cached_property
    def palm_scale(self) -> float:
        """Palm length in px (wrist -> middle MCP): the unit of every dimensionless threshold."""
        return max(self.dist(WRIST, MIDDLE_MCP), 1e-6)

    def dist(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.px[i] - self.px[j]))

    def ndist(self, i: int, j: int) -> float:
        """Scale-invariant distance between landmarks i and j, in palm lengths."""
        return self.dist(i, j) / self.palm_scale


def centre_crop(frame_size: tuple[int, int], fraction: float) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of the centred crop with ``fraction`` of the frame's size."""
    width, height = frame_size
    crop_w, crop_h = max(1, round(width * fraction)), max(1, round(height * fraction))
    x0, y0 = (width - crop_w) // 2, (height - crop_h) // 2
    return x0, y0, x0 + crop_w, y0 + crop_h


def uncrop(
    norm: npt.ArrayLike, box: tuple[int, int, int, int], frame_size: tuple[int, int]
) -> FloatArray:
    """Normalized landmarks found in a crop -> normalized coordinates of the whole frame."""
    x0, y0, x1, y1 = box
    width, height = frame_size
    out = np.array(norm, dtype=np.float64).reshape(-1, 3)
    out[:, 0] = (x0 + out[:, 0] * (x1 - x0)) / width
    out[:, 1] = (y0 + out[:, 1] * (y1 - y0)) / height
    out[:, 2] *= (x1 - x0) / width  # MediaPipe's z is on the scale of the image width
    return out


def ensure_model(path: str | os.PathLike[str], url: str) -> Path:
    """Return the model path, downloading the model first if it is missing."""
    target = Path(path)
    if target.is_file() and zipfile.is_zipfile(target):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    log.info("Downloading the model %s ...", target)
    try:
        with urllib.request.urlopen(url, timeout=30) as response, open(partial, "wb") as fh:
            shutil.copyfileobj(response, fh)
        # A .task file is a zip of two TFLite models. Anything else is most likely an HTML error
        # page from a proxy or captive portal, which MediaPipe would reject with a cryptic error.
        if not zipfile.is_zipfile(partial):
            raise RuntimeError("downloaded file is not a MediaPipe .task bundle")
        os.replace(partial, target)  # atomic: never leaves a half-written model behind
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the model from {url} ({exc}). "
            f"Download it manually and save it as {target}."
        ) from exc
    return target


class HandTracker:
    """BGR frame in, ``HandObservation`` (or None) out."""

    def __init__(self, cfg: TrackerConfig) -> None:
        # Imported lazily: mediapipe is heavy and the pure-logic modules/tests do not need it.
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = ensure_model(cfg.model_path, cfg.model_url)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._mp: Any = mp
        self._landmarker: Any = vision.HandLandmarker.create_from_options(options)
        self.zoom_crop = cfg.zoom_crop
        # A landmarker of its own: VIDEO mode tracks the hand from frame to frame, and the crop
        # has other coordinates than the whole frame.
        self._zoomed: Any = (
            vision.HandLandmarker.create_from_options(options) if 0 < cfg.zoom_crop < 1 else None
        )
        self._last_ts_ms = -1
        self._misses = 0  # frames in a row without a hand
        self.last_inference_ms = 0.0
        self.zoomed = False  # the last hand was found by the centre-crop search

    def process(self, frame_bgr: npt.NDArray[np.uint8], timestamp: float) -> HandObservation | None:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode demands *strictly* increasing integer timestamps; an equal value raises.
        ts_ms = max(int(timestamp * 1000.0), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        started = time.perf_counter()
        result = self._landmarker.detect_for_video(image, ts_ms)
        box = None
        idle = self._misses >= ZOOM_IDLE_AFTER and self._misses % ZOOM_IDLE_EVERY != 0
        if not result.hand_landmarks and self._zoomed is not None and not idle:
            box = centre_crop((width, height), self.zoom_crop)
            x0, y0, x1, y1 = box
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=crop)
            result = self._zoomed.detect_for_video(image, ts_ms)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        self.zoomed = False
        if not result.hand_landmarks:
            self._misses += 1
            return None
        self._misses = 0
        norm = np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], dtype=np.float64)
        if box is not None:
            norm = uncrop(norm, box, (width, height))
            self.zoomed = True
        label, score = "Unknown", 1.0
        if result.handedness and result.handedness[0]:
            category = result.handedness[0][0]
            label, score = category.category_name or "Unknown", float(category.score)
        return HandObservation.from_normalized(norm, (width, height), timestamp, label, score)

    def close(self) -> None:
        self._landmarker.close()
        if self._zoomed is not None:
            self._zoomed.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
