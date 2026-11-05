"""Full-screen gaze calibration: look at each dot until it moves on (9 dots, ~15 s)."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from config import EyeConfig
from src.camera import FramePacket
from src.face_tracker import FaceObservation, gaze_feature, measure_eyes
from src.gaze import CalibrationSession, GazeCalibration

log = logging.getLogger(__name__)
WINDOW = "Gaze calibration"
MIN_OPEN_ASPECT = 0.15  # (nearly) closed eyes give a garbage iris position: skip those frames


class FrameSource(Protocol):
    @property
    def is_running(self) -> bool: ...
    def read(self, after_seq: int = -1, timeout: float = 1.0) -> FramePacket | None: ...


class FaceSource(Protocol):
    def process(self, frame_bgr: npt.NDArray[Any], timestamp: float) -> FaceObservation | None: ...


def run_calibration(
    camera: FrameSource,
    tracker: FaceSource,
    cfg: EyeConfig,
    screen_size: tuple[int, int],
    mirrored: bool = True,
) -> GazeCalibration | None:
    """Returns the fitted calibration, or None if cancelled (ESC / q) or unusable."""
    session = CalibrationSession(screen_size, cfg.calib_settle, cfg.calib_sample)
    width, height = screen_size
    canvas = np.zeros((height, width, 3), np.uint8)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    last_seq = -1
    try:
        while not session.done:
            packet = camera.read(after_seq=last_seq, timeout=1.0)
            if packet is None:
                if not camera.is_running:
                    return None
                continue
            last_seq = packet.seq
            obs = tracker.process(packet.frame, packet.timestamp)
            feature = None
            if obs is not None:
                left, right = measure_eyes(obs, mirrored)
                if min(left.aspect, right.aspect) >= MIN_OPEN_ASPECT:
                    feature = gaze_feature(left, right)
            session.update(packet.timestamp, feature)
            _draw(canvas, session, face_found=obs is not None)
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                log.info("Gaze calibration cancelled.")
                return None
        calibration = session.result()
    except ValueError as exc:
        log.error("Gaze calibration failed: %s", exc)
        return None
    finally:
        cv2.destroyWindow(WINDOW)
    log.info("Gaze calibration done, fit error %.0f px.", calibration.rms_error)
    return calibration


def _draw(canvas: npt.NDArray[np.uint8], session: CalibrationSession, face_found: bool) -> None:
    canvas[:] = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        f"Look at the dot until it moves on ({session.index + 1}/{len(session.targets)}). "
        "Keep your head still. ESC = cancel",
        (40, 50),
        font,
        0.8,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    if not face_found:
        cv2.putText(
            canvas, "Face not found: look at the screen", (40, 95), font, 0.8, (0, 0, 255), 2
        )
    target = session.target
    if target is not None:
        center = (round(target[0]), round(target[1]))
        ring = 6 + round(34 * (1.0 - session.progress))  # shrinks while the dot is sampled
        cv2.circle(canvas, center, ring, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, center, 6, (0, 0, 255), -1, cv2.LINE_AA)
