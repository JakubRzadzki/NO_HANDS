"""Threaded webcam capture that always hands out the *newest* frame.

Why a thread? ``cv2.VideoCapture.read()`` blocks for up to one frame interval, and drivers queue
several frames internally. In a naive ``read -> process`` loop, any frame that takes longer to
process than the camera interval leaves frames piling up in that queue, and from then on every
processed frame is already stale: input lag grows by tens of milliseconds. Here a daemon thread
drains the camera continuously and keeps only the latest frame; the consumer never sees an old
one, and frames it could not keep up with are simply dropped (and counted).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from config import CameraConfig

log = logging.getLogger(__name__)


class CaptureLike(Protocol):
    """The subset of ``cv2.VideoCapture`` used here (lets tests inject a fake camera)."""

    def isOpened(self) -> bool: ...  # noqa: N802 - OpenCV's name
    def read(self) -> tuple[bool, Any]: ...
    def set(self, prop_id: int, value: float) -> bool: ...
    def get(self, prop_id: int) -> float: ...
    def release(self) -> None: ...


@dataclass(frozen=True)
class FramePacket:
    frame: npt.NDArray[Any]  # BGR uint8, already mirrored if configured
    timestamp: float  # time.perf_counter() when the frame arrived
    seq: int  # strictly increasing frame number


def configure_capture(cap: CaptureLike, cfg: CameraConfig) -> None:
    """Ask for the configured format. Drivers may silently grant something else."""
    # Order matters: some backends reset the pixel format when the resolution changes.
    if cfg.use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    if cfg.buffer_size > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)
    if cfg.open_settings and not cap.set(cv2.CAP_PROP_SETTINGS, 1):
        log.warning("This camera backend has no settings dialog (it is a Windows/DirectShow one).")


def fourcc_name(value: float) -> str:
    code = int(value)
    return code.to_bytes(4, "little").decode("ascii", "replace") if code > 0 else "?"


def open_capture(cfg: CameraConfig) -> CaptureLike:
    """Open the webcam with the best backend for the platform."""
    if sys.platform == "win32":
        # DirectShow opens in ~1 s and honours the MJPG request; the default Media Foundation
        # backend can take several seconds to open and often ignores FOURCC.
        cap = cv2.VideoCapture(cfg.device_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(cfg.device_id)


FALLBACK_SIZES: tuple[tuple[int, int], ...] = ((1280, 720), (640, 480))
WARMUP_FRAMES = 5  # the first frames are slow while the sensor and the exposure settle


def measure_fps(cap: CaptureLike, frames: int = 15, max_seconds: float = 1.5) -> float | None:
    """The frame rate the camera delivers right now, or None if it cannot be measured.

    Measured, not asked: CAP_PROP_FPS reports the request, and drivers accept modes they then
    deliver at a fraction of the speed."""
    for _ in range(WARMUP_FRAMES):
        cap.read()
    delivered = 0
    started = time.perf_counter()
    while delivered < frames and time.perf_counter() - started < max_seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        delivered += 1
    elapsed = time.perf_counter() - started
    if delivered < 3 or elapsed <= 0:
        return None
    return delivered / elapsed


class ThreadedCamera:
    def __init__(
        self,
        cfg: CameraConfig,
        capture_factory: Callable[[CameraConfig], CaptureLike] = open_capture,
        stall_timeout: float = 2.0,
        fps_check_frames: int = 90,
        fps_probe_frames: int = 15,
    ) -> None:
        self._cfg = cfg
        self._fps_check_frames = fps_check_frames
        self._fps_probe_frames = fps_probe_frames
        self._factory = capture_factory
        self._stall_timeout = stall_timeout
        self._cap: CaptureLike | None = None
        self._cond = threading.Condition()
        self._packet: FramePacket | None = None
        self._last_read_seq = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.frame_size = (cfg.width, cfg.height)  # replaced by the real size in start()
        self.measured_fps = 0.0
        self.dropped_frames = 0

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self) -> ThreadedCamera:
        cap, frame = self._open(self._cfg)
        if self._cfg.auto_fallback:
            cap, frame = self._usable_mode(cap, frame)
        height, width = frame.shape[:2]
        self.frame_size = (int(width), int(height))
        self._cap = cap
        self._publish(frame, time.perf_counter())
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        log.info(
            "Camera %d: %dx%d, driver reports %.0f FPS",
            self._cfg.device_id,
            width,
            height,
            cap.get(cv2.CAP_PROP_FPS),
        )
        return self

    def _open(self, cfg: CameraConfig) -> tuple[CaptureLike, npt.NDArray[Any]]:
        cap = self._factory(cfg)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cfg.device_id}")
        configure_capture(cap, cfg)
        # Read one frame synchronously: the driver may not grant the requested resolution, and
        # everything downstream (ROI, HUD) must use the size we really get.
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Camera {cfg.device_id} opened but delivered no frame")
        return cap, frame

    def _usable_mode(
        self, cap: CaptureLike, frame: npt.NDArray[Any]
    ) -> tuple[CaptureLike, npt.NDArray[Any]]:
        """Keep the requested mode if it is fast enough; otherwise take the first smaller size
        that is, or failing that the fastest one."""
        cfg = self._cfg
        fps = measure_fps(cap, self._fps_probe_frames)
        if fps is None or fps >= cfg.min_fps:
            return cap, frame
        height, width = frame.shape[:2]
        tried: list[tuple[float, CameraConfig]] = [(fps, cfg)]
        for size in FALLBACK_SIZES:
            if size[0] >= width:
                continue
            trial = replace(cfg, width=size[0], height=size[1], open_settings=False)
            cap.release()
            cap, frame = self._open(trial)
            trial_fps = measure_fps(cap, self._fps_probe_frames)
            if trial_fps is None:
                continue
            tried.append((trial_fps, trial))
            if trial_fps >= cfg.min_fps:
                log.warning(
                    "Camera gives only %.0f FPS at %dx%d: using %dx%d at %.0f FPS instead. "
                    "(--width / --height keep a size; --probe-camera shows every mode.)",
                    fps,
                    width,
                    height,
                    frame.shape[1],
                    frame.shape[0],
                    trial_fps,
                )
                return cap, frame
        # Nothing is fast enough (dim light: the camera exposes longer at every size).
        best_fps, best = max(tried, key=lambda item: (round(item[0]), item[1].width))
        cap.release()
        cap, frame = self._open(replace(best, open_settings=False))
        log.warning(
            "Camera is slow in every mode (best: %.0f FPS at %dx%d). More light usually helps: "
            "webcams lower their frame rate to expose longer.",
            best_fps,
            frame.shape[1],
            frame.shape[0],
        )
        return cap, frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._cond:
            self._cond.notify_all()

    def __enter__(self) -> ThreadedCamera:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    # -- consumer API ------------------------------------------------------------------------
    def read(self, after_seq: int = -1, timeout: float = 1.0) -> FramePacket | None:
        """Wait for a frame newer than ``after_seq``.

        Returns None on timeout or when the camera thread has stopped (check ``error``).
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._packet is None or self._packet.seq <= after_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.is_running:
                    return None
                self._cond.wait(remaining)
            self._last_read_seq = self._packet.seq
            return self._packet

    # -- producer side -----------------------------------------------------------------------
    def _publish(self, raw: npt.NDArray[np.uint8], timestamp: float) -> None:
        # flip() returns a new array, so the consumer may draw on its frame freely.
        frame: npt.NDArray[Any] = cv2.flip(raw, 1) if self._cfg.mirror else raw
        with self._cond:
            previous = self._packet
            if previous is not None:
                if previous.seq > self._last_read_seq:
                    self.dropped_frames += 1  # nobody consumed it: processing is the bottleneck
                interval = timestamp - previous.timestamp
                if interval > 0:
                    instant = 1.0 / interval
                    self.measured_fps = (
                        instant
                        if self.measured_fps == 0
                        else 0.9 * self.measured_fps + 0.1 * instant
                    )
            seq = 0 if previous is None else previous.seq + 1
            self._packet = FramePacket(frame, timestamp, seq)
            self._cond.notify_all()
        if seq == self._fps_check_frames:
            self._check_fps()

    def _check_fps(self) -> None:
        """Say so when the camera is much slower than asked: it is the usual "why 30 FPS?"."""
        c = self._cfg
        if self.measured_fps < 0.8 * c.fps:
            width, height = self.frame_size
            log.warning(
                "Camera delivers %.0f FPS at %dx%d (asked for %d). The camera, its driver or dim "
                "light (longer exposure) is the limit. See what it can do: python main.py "
                "--probe-camera",
                self.measured_fps,
                width,
                height,
                c.fps,
            )

    def _run(self) -> None:
        cap = self._cap
        assert cap is not None
        failing_since: float | None = None
        while not self._stop.is_set():
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                failing_since = failing_since if failing_since is not None else now
                if now - failing_since > self._stall_timeout:
                    self.error = "camera stopped delivering frames (unplugged or in use?)"
                    log.error(self.error)
                    break
                time.sleep(0.01)
                continue
            failing_since = None
            self._publish(frame, now)
        with self._cond:
            self._cond.notify_all()  # wake a waiting consumer so it notices we stopped


# -- probing ---------------------------------------------------------------------------------
PROBE_MODES: tuple[tuple[int, int, int], ...] = (
    (1920, 1080, 60),
    (1920, 1080, 30),
    (1280, 720, 60),
    (1280, 720, 30),
    (640, 480, 60),
    (640, 480, 30),
)


@dataclass(frozen=True)
class ProbeResult:
    requested: tuple[int, int, int]  # width, height, fps
    size: tuple[int, int] | None  # what the camera really delivered (None: did not open)
    fps: float  # measured, not what the driver claims
    fourcc: str


def probe_camera(
    device_id: int,
    modes: tuple[tuple[int, int, int], ...] = PROBE_MODES,
    frames: int = 40,
    capture_factory: Callable[[CameraConfig], CaptureLike] = open_capture,
) -> list[ProbeResult]:
    """Try each mode and *measure* the frame rate. Drivers often accept a request and then
    deliver something else, and CAP_PROP_FPS reports the request, not the reality."""
    results = []
    for width, height, fps in modes:
        cfg = CameraConfig(device_id=device_id, width=width, height=height, fps=fps, mirror=False)
        cap = capture_factory(cfg)
        try:
            if not cap.isOpened():
                results.append(ProbeResult((width, height, fps), None, 0.0, "?"))
                continue
            configure_capture(cap, cfg)
            for _ in range(5):  # the first frames are often slow while the sensor settles
                cap.read()
            size: tuple[int, int] | None = None
            delivered = 0
            started = time.perf_counter()
            for _ in range(frames):
                ok, frame = cap.read()
                if ok and frame is not None:
                    delivered += 1
                    size = (int(frame.shape[1]), int(frame.shape[0]))
            elapsed = time.perf_counter() - started
            measured = delivered / elapsed if elapsed > 0 else 0.0
            fourcc = fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))
            results.append(ProbeResult((width, height, fps), size, measured, fourcc))
        finally:
            cap.release()
    return results
