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
from dataclasses import dataclass
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


class ThreadedCamera:
    def __init__(
        self,
        cfg: CameraConfig,
        capture_factory: Callable[[CameraConfig], CaptureLike] = open_capture,
        stall_timeout: float = 2.0,
    ) -> None:
        self._cfg = cfg
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
        cap = self._factory(self._cfg)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._cfg.device_id}")
        self._configure(cap)
        # Read one frame synchronously: the driver may not grant the requested resolution, and
        # everything downstream (ROI, HUD) must use the size we really get.
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Camera {self._cfg.device_id} opened but delivered no frame")
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
    def _configure(self, cap: CaptureLike) -> None:
        c = self._cfg
        # Order matters: some backends reset the pixel format when the resolution changes.
        if c.use_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.fps)
        if c.buffer_size > 0:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, c.buffer_size)

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
