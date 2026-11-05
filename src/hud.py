"""OpenCV HUD: ROI box coloured by state, landmarks, pinch proximity lines, live metrics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from config import GestureConfig
from src.gesture_engine import PINCH_FINGERS, EngineOutput, Finger, LeftState, Mode
from src.hand_tracker import (
    HAND_CONNECTIONS,
    INDEX_TIP,
    MIDDLE_MCP,
    THUMB_TIP,
    WRIST,
    HandObservation,
)
from src.pointer import Roi

Color = tuple[int, int, int]  # BGR
CYAN: Color = (255, 255, 0)
GREEN: Color = (0, 220, 0)
YELLOW: Color = (0, 255, 255)
MAGENTA: Color = (255, 0, 255)
RED: Color = (0, 0, 255)
ORANGE: Color = (0, 165, 255)
GRAY: Color = (160, 160, 160)
WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CLICK_LABELS = {"LEFT CLICK", "RIGHT CLICK", "DOUBLE CLICK", "DROP"}


@dataclass
class LoopStats:
    """Rolling performance numbers (exponential moving averages)."""

    fps: float = 0.0  # processed frames per second
    camera_fps: float = 0.0
    latency_ms: float = 0.0  # frame arrived -> actions sent to the OS
    inference_ms: float = 0.0  # MediaPipe time per frame
    dropped_frames: int = 0
    _last: float | None = None

    def update(
        self,
        now: float,
        frame_timestamp: float,
        inference_ms: float,
        camera_fps: float,
        dropped_frames: int,
        alpha: float = 0.1,
    ) -> None:
        def ema(old: float, new: float) -> float:
            return new if old == 0.0 else (1.0 - alpha) * old + alpha * new

        if self._last is not None and now > self._last:
            self.fps = ema(self.fps, 1.0 / (now - self._last))
        self._last = now
        self.latency_ms = ema(self.latency_ms, (now - frame_timestamp) * 1000.0)
        self.inference_ms = ema(self.inference_ms, inference_ms)
        self.camera_fps = camera_fps
        self.dropped_frames = dropped_frames


def state_color(out: EngineOutput, paused: bool) -> Color:
    if paused or out.mode is Mode.CLUTCH:
        return RED
    if out.mode is Mode.SCROLL:
        return YELLOW
    if out.mode is Mode.ZOOM:
        return MAGENTA
    if out.mode is Mode.MEDIA:
        return ORANGE
    if out.mode is Mode.SWITCH:
        return WHITE
    if out.mode is Mode.NO_HAND:
        return GRAY
    if out.left_state is not LeftState.IDLE or out.label in CLICK_LABELS:
        return GREEN
    return CYAN


def proximity_color(distance: float, enter: float, exit_: float) -> Color:
    """Red when far, through yellow, to green at the click threshold."""
    far = 2.0 * exit_
    t = float(np.clip((far - distance) / (far - enter), 0.0, 1.0))
    if t < 0.5:
        return (0, int(510 * t), 255)
    return (0, 255, int(255 * (2.0 - 2.0 * t)))


def _pt(p: npt.NDArray[np.int_]) -> tuple[int, int]:
    return int(p[0]), int(p[1])


class Hud:
    def __init__(self, gesture_cfg: GestureConfig, show_landmarks: bool = True) -> None:
        self.cfg = gesture_cfg
        self.show_landmarks = show_landmarks

    def draw(
        self,
        frame: npt.NDArray[np.uint8],
        roi: Roi,
        obs: HandObservation | None,
        out: EngineOutput,
        stats: LoopStats,
        paused: bool,
    ) -> None:
        """Draw in place on ``frame`` (the mirrored camera image)."""
        h, w = frame.shape[:2]
        color = state_color(out, paused)
        x0, y0, x1, y1 = roi.to_pixels((w, h))
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)

        if obs is not None:
            pts = np.rint(obs.norm[:, :2] * (w, h)).astype(int)
            if self.show_landmarks:
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, _pt(pts[a]), _pt(pts[b]), GRAY, 1, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(frame, _pt(p), 3, WHITE, -1, cv2.LINE_AA)
            if out.features is not None and not paused:
                self._draw_gesture_overlay(frame, pts, out)
                ax, ay = out.features.anchor
                anchor = (int(ax * w), int(ay * h))
                cv2.circle(frame, anchor, 8, color, -1, cv2.LINE_AA)
                cv2.circle(frame, anchor, 9, BLACK, 1, cv2.LINE_AA)

        self._draw_panel(frame, out, stats, paused)

    def _draw_gesture_overlay(
        self, frame: npt.NDArray[np.uint8], pts: npt.NDArray[np.int_], out: EngineOutput
    ) -> None:
        f = out.features
        assert f is not None
        if out.mode is Mode.POINTER:
            closest = min(PINCH_FINGERS, key=f.pinch.__getitem__)
            for finger in PINCH_FINGERS:
                c = proximity_color(f.pinch[finger], self.cfg.pinch_enter, self.cfg.pinch_exit)
                thickness = 3 if finger is closest else 1
                cv2.line(
                    frame, _pt(pts[THUMB_TIP]), _pt(pts[finger.tip]), c, thickness, cv2.LINE_AA
                )
        elif out.mode is Mode.SCROLL and out.scroll_origin is not None:
            origin = (int(out.scroll_origin[0]), int(out.scroll_origin[1]))
            current = (int(f.scroll_point[0]), int(f.scroll_point[1]))
            radius = max(int(self.cfg.scroll_deadzone * f.palm_scale), 1)
            cv2.circle(frame, origin, radius, YELLOW, 1, cv2.LINE_AA)
            cv2.arrowedLine(frame, origin, current, YELLOW, 2, cv2.LINE_AA, tipLength=0.2)
        elif out.mode is Mode.ZOOM:
            cv2.line(frame, _pt(pts[THUMB_TIP]), _pt(pts[INDEX_TIP]), MAGENTA, 3, cv2.LINE_AA)
        elif out.mode is Mode.MEDIA:  # the "knob needle": its angle drives the volume
            cv2.line(frame, _pt(pts[WRIST]), _pt(pts[MIDDLE_MCP]), ORANGE, 3, cv2.LINE_AA)

    def _draw_panel(
        self, frame: npt.NDArray[np.uint8], out: EngineOutput, stats: LoopStats, paused: bool
    ) -> None:
        g = self.cfg
        lines: list[tuple[str, float, Color]] = [
            ("PAUSED" if paused else out.label, 0.8, state_color(out, paused)),
            (
                f"FPS {stats.fps:3.0f}  cam {stats.camera_fps:3.0f}  "
                f"latency {stats.latency_ms:3.0f} ms  model {stats.inference_ms:3.0f} ms  "
                f"dropped {stats.dropped_frames}",
                0.45,
                WHITE,
            ),
        ]
        if out.features is not None:
            p = out.features.pinch
            lines.append(
                (
                    f"pinch I {p[Finger.INDEX]:.2f}  M {p[Finger.MIDDLE]:.2f}  "
                    f"R {p[Finger.RING]:.2f}   (on <{g.pinch_enter:.2f}, off >{g.pinch_exit:.2f})",
                    0.45,
                    WHITE,
                )
            )
            fingers = " ".join(
                name if out.extended.get(finger) else "."
                for name, finger in zip("IMRP", Finger, strict=True)
            )
            ext = "/".join(f"{out.features.extension[finger]:.2f}" for finger in Finger)
            lines.append((f"extended {fingers}  ({ext})", 0.45, WHITE))
            if out.mode is Mode.SCROLL:
                rx, ry = out.scroll_rate
                lines.append((f"scroll rate x {rx:+.1f}  y {ry:+.1f} ticks/s", 0.45, YELLOW))
            if out.mode is Mode.MEDIA:
                if out.knob_offset is None:
                    knob = "knob: hold the hand still to set neutral"
                else:
                    knob = f"knob {out.knob_offset:+.0f} deg   volume {out.volume_rate:+.1f}/s"
                lines.append((knob, 0.45, ORANGE))
            if out.pending_mode is not None:
                lines.append((f"switching to {out.pending_mode.value}...", 0.45, GRAY))
        if out.media_active:
            lines.append(
                ("MEDIA MODE: swipe = prev/next, turn = volume, fist = play", 0.45, ORANGE)
            )
        lines.append(("q/ESC quit   p pause   shaka 0.6 s: mouse <-> media", 0.4, GRAY))
        lines.append(("ctrl+alt+p / ctrl+alt+q: pause / quit from anywhere", 0.4, GRAY))

        pad, gap = 8, 6
        sizes = [cv2.getTextSize(text, FONT, scale, 1)[0] for text, scale, _ in lines]
        panel_w = min(max(sw for sw, _ in sizes) + 2 * pad, frame.shape[1])
        panel_h = min(sum(sh + gap for _, sh in sizes) + 2 * pad, frame.shape[0])
        region = frame[:panel_h, :panel_w]
        region[:] = cv2.convertScaleAbs(region, alpha=0.35, beta=0)  # darken for legibility

        y = pad
        for (text, scale, color), (_, text_h) in zip(lines, sizes, strict=True):
            y += text_h
            thickness = 2 if scale >= 0.8 else 1
            cv2.putText(frame, text, (pad, y), FONT, scale, color, thickness, cv2.LINE_AA)
            y += gap
