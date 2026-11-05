"""Full-screen gaze calibration (~30 s): 9 dots, a head-movement phase, 4 check dots."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from config import EyeConfig
from src.camera import FramePacket
from src.face_tracker import FaceObservation, gaze_features, measure_eyes
from src.gaze import MIN_HEAD_RANGE, CalibrationSession, GazeModel

log = logging.getLogger(__name__)
WINDOW = "Gaze calibration"
MIN_OPEN_ASPECT = 0.15  # (nearly) closed eyes give a garbage iris position: skip those frames
RESULT_SECONDS = 4.0  # how long the result screen stays up (any key closes it)
POOR_ACCURACY = 0.08  # check-dot error above this fraction of the screen width
HIGH_JITTER = 0.08  # per-frame jitter above this fraction of the screen width
FONT = cv2.FONT_HERSHEY_SIMPLEX


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
) -> GazeModel | None:
    """Returns the fitted gaze model, or None if cancelled (ESC / q) or unusable."""
    session = CalibrationSession(screen_size, cfg.calib_settle, cfg.calib_sample, cfg.calib_head)
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
            sample = None
            if obs is not None:
                left, right = measure_eyes(obs, mirrored)
                if min(left.aspect, right.aspect) >= MIN_OPEN_ASPECT:
                    sample = (gaze_features(obs, left, right, mirrored), obs.head_position[2])
            session.update(packet.timestamp, sample)
            _draw(canvas, session, face_found=obs is not None)
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                log.info("Gaze calibration cancelled.")
                return None
        model = session.result()
        _show_result(canvas, model)
    except ValueError as exc:
        log.error("Gaze calibration failed: %s", exc)
        return None
    finally:
        cv2.destroyWindow(WINDOW)
    return model


def describe(model: GazeModel) -> str:
    """One line on the quality of a calibration, for the log and the result screen."""
    width = model.screen_size[0]
    parts = []
    if model.validation_error is not None:
        error = model.validation_error
        parts.append(f"accuracy ~{error:.0f} px ({100 * error / width:.1f}% of the screen width)")
    parts.append(f"jitter ~{model.noise:.0f} px")
    return ", ".join(parts)


def quality_warnings(model: GazeModel) -> list[str]:
    """What is wrong with a calibration, in words a user can act on."""
    width = model.screen_size[0]
    warnings = []
    if model.validation_error is not None and model.validation_error > POOR_ACCURACY * width:
        warnings.append(
            "Poor accuracy: recalibrate with your face well lit and your eyes on each dot, "
            "or use --eyes-pointer nose."
        )
    if model.noise > HIGH_JITTER * width:
        warnings.append("High jitter: more light on your face helps most.")
    yaw, pitch = model.head_range
    if yaw < MIN_HEAD_RANGE[0] or pitch < MIN_HEAD_RANGE[1]:
        warnings.append(
            f"The head barely moved in the arrow phase (spread yaw {yaw:.1f}, pitch {pitch:.1f} "
            "deg), so head movement will move the cursor. Recalibrate and turn the head more."
        )
    return warnings


def _show_result(canvas: npt.NDArray[np.uint8], model: GazeModel) -> None:
    log.info("Gaze calibration done: %s.", describe(model))
    warnings = quality_warnings(model)
    for warning in warnings:
        log.warning("%s", warning)
    canvas[:] = 0
    lines = [("Calibration done", 1.2, (255, 255, 255)), (describe(model), 0.8, (200, 200, 200))]
    lines += [(warning, 0.6, (0, 165, 255)) for warning in warnings]
    hint = "The cursor jumps where you look; turn the head slightly to fine-tune."
    lines.append((hint, 0.7, (200, 200, 200)))
    y = canvas.shape[0] // 3
    for text, scale, color in lines:
        cv2.putText(canvas, text, (60, y), FONT, scale, color, 2, cv2.LINE_AA)
        y += round(70 * scale)
    cv2.imshow(WINDOW, canvas)
    end = time.monotonic() + RESULT_SECONDS
    while time.monotonic() < end:
        if cv2.waitKey(50) != -1:
            break


def _draw(canvas: npt.NDArray[np.uint8], session: CalibrationSession, face_found: bool) -> None:
    canvas[:] = 0
    step = session.index + 1
    cv2.putText(
        canvas,
        f"{session.instruction}  ({step}/{len(session.steps)})  ESC = cancel",
        (40, 50),
        FONT,
        0.8,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    if not face_found:
        cv2.putText(
            canvas, "Face not found: look at the screen", (40, 95), FONT, 0.8, (0, 0, 255), 2
        )
    target = session.target
    if target is None:
        return
    center = (round(target[0]), round(target[1]))
    ring = 6 + round(34 * (1.0 - session.progress))  # shrinks while the dot is sampled
    cv2.circle(canvas, center, ring, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, center, 6, (0, 0, 255), -1, cv2.LINE_AA)
    hint = session.head_hint
    if hint is not None:  # an arrow next to the dot: turn the head that way, eyes stay put
        name, (dx, dy) = hint
        start = (center[0] + 60 * dx, center[1] + 60 * dy)
        end = (center[0] + 160 * dx, center[1] + 160 * dy)
        cv2.arrowedLine(canvas, start, end, (0, 200, 255), 6, cv2.LINE_AA, tipLength=0.35)
        label = f"turn head {name}"
        (text_w, _), _ = cv2.getTextSize(label, FONT, 0.9, 2)
        anchor = (end[0] - text_w // 2, end[1] + 45 if dy >= 0 else end[1] - 25)
        cv2.putText(canvas, label, anchor, FONT, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
