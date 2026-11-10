from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
import pytest

from config import CameraConfig
from src.camera import ThreadedCamera, fourcc_name, probe_camera


class FakeCapture:
    """Frames whose left half is dark and right half bright, numbered in pixel (0, 0)."""

    def __init__(
        self, opened: bool = True, fail_after: int | None = None, frame_time: float = 0.002
    ) -> None:
        self.opened = opened
        self.fail_after = fail_after
        self.frame_time = frame_time
        self.count = 0
        self.released = threading.Event()
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's name
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(self.frame_time)
        self.count += 1
        if self.fail_after is not None and self.count > self.fail_after:
            return False, None
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, 32:] = 200
        frame[0, 0] = self.count % 256
        return True, frame

    def set(self, prop_id: int, value: float) -> bool:
        self.props[prop_id] = value
        return True

    def get(self, prop_id: int) -> float:
        return self.props.get(prop_id, 0.0)

    def release(self) -> None:
        self.released.set()


def make_camera(fake: FakeCapture, mirror: bool = False, **kwargs: Any) -> ThreadedCamera:
    cfg = CameraConfig(mirror=mirror)
    return ThreadedCamera(cfg, capture_factory=lambda _cfg: fake, **kwargs)


def test_delivers_newer_frames_with_increasing_seq_and_real_size() -> None:
    fake = FakeCapture()
    with make_camera(fake) as cam:
        assert cam.frame_size == (64, 48)  # what the camera delivers, not what was requested
        first = cam.read(after_seq=-1, timeout=1.0)
        assert first is not None
        second = cam.read(after_seq=first.seq, timeout=1.0)
        assert second is not None and second.seq > first.seq
        assert second.timestamp > first.timestamp
    assert fake.released.is_set()


def test_slow_consumer_gets_latest_frame_and_drops_are_counted() -> None:
    fake = FakeCapture()
    with make_camera(fake) as cam:
        time.sleep(0.1)  # the camera keeps producing while we "process"
        packet = cam.read(after_seq=-1, timeout=1.0)
        assert packet is not None and packet.seq > 5
        assert cam.dropped_frames > 0


def test_mirror_flips_horizontally() -> None:
    with make_camera(FakeCapture(), mirror=True) as cam:
        packet = cam.read(timeout=1.0)
        assert packet is not None
        assert packet.frame[10, 5, 0] == 200 and packet.frame[10, 60, 0] == 0


def test_stop_unblocks_a_waiting_reader() -> None:
    cam = make_camera(FakeCapture()).start()
    cam.stop()
    started = time.monotonic()
    assert cam.read(after_seq=10**9, timeout=2.0) is None
    assert time.monotonic() - started < 0.5


def test_stalled_camera_reports_error() -> None:
    with make_camera(FakeCapture(fail_after=3), stall_timeout=0.05) as cam:
        deadline = time.monotonic() + 2.0
        while cam.is_running and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not cam.is_running
        assert cam.error is not None
        assert cam.read(after_seq=10**9, timeout=0.1) is None


def test_unopenable_camera_raises() -> None:
    with pytest.raises(RuntimeError, match="Cannot open camera"):
        make_camera(FakeCapture(opened=False)).start()


def test_a_camera_slower_than_requested_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeCapture(frame_time=1 / 25)  # a 25 FPS camera, asked for 60
    with caplog.at_level("WARNING"), make_camera(fake, fps_check_frames=8) as cam:
        cam.read(after_seq=9, timeout=2.0)
    assert "Camera delivers" in caplog.text and "--probe-camera" in caplog.text


class SizedCapture(FakeCapture):
    """A camera whose speed depends on the size it is asked for, with frames of that size:
    like a webcam that sends Full HD uncompressed at a few frames per second."""

    def __init__(self, frame_times: dict[int, float]) -> None:
        super().__init__()
        self.frame_times = frame_times  # requested width -> seconds per frame
        self.size = (1920, 1080)

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self.size = (int(value), self.size[1])
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self.size = (self.size[0], int(value))
        return super().set(prop_id, value)

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(self.frame_times[self.size[0]])
        return True, np.zeros((self.size[1], self.size[0], 3), np.uint8)


def sized_camera(frame_times: dict[int, float], **cfg: Any) -> ThreadedCamera:
    config = CameraConfig(mirror=False, **{"auto_fallback": True, **cfg})
    return ThreadedCamera(config, lambda _cfg: SizedCapture(frame_times), fps_probe_frames=4)


def test_a_camera_too_slow_at_full_hd_falls_back_to_a_usable_size(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reported by a user: "works better on Windows than on Linux". Some cameras deliver Full
    HD only uncompressed, at a few frames per second; 8 FPS makes every gesture miss."""
    monkeypatch.setattr("src.camera.WARMUP_FRAMES", 1)
    slow_full_hd = {1920: 1 / 8, 1280: 1 / 100, 640: 1 / 100}
    with caplog.at_level("WARNING"), sized_camera(slow_full_hd) as cam:
        assert cam.frame_size == (1280, 720)
        assert cam.read(timeout=1.0) is not None
    assert "using 1280x720" in caplog.text


def test_a_camera_slow_everywhere_keeps_the_largest_size(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.camera.WARMUP_FRAMES", 1)
    dim_light = {1920: 1 / 14, 1280: 1 / 14, 640: 1 / 14}  # exposure, not the size, is the limit
    with caplog.at_level("WARNING"), sized_camera(dim_light) as cam:
        assert cam.frame_size == (1920, 1080)
    assert "slow in every mode" in caplog.text


class FlakyReopen:
    """A camera factory like a real driver. The first ``failures`` opens after the very first
    one fail (not opened, or opened but without a frame), and sizes in ``refuse`` never open.
    Every capture has the speed of its size."""

    def __init__(
        self,
        frame_times: dict[int, float],
        failures: int = 0,
        how: str = "closed",
        refuse: tuple[int, ...] = (),
    ) -> None:
        self.frame_times, self.failures, self.how, self.refuse = frame_times, failures, how, refuse
        self.opens = 0

    def __call__(self, cfg: CameraConfig) -> FakeCapture:
        self.opens += 1
        cap = SizedCapture(self.frame_times)
        failing = self.opens > 1 and (self.opens <= 1 + self.failures or cfg.width in self.refuse)
        if failing and self.how == "closed":
            cap.opened = False
        elif failing:
            cap.read = lambda: (False, None)  # type: ignore[method-assign]
        return cap


@pytest.mark.parametrize("how", ["closed", "no frame"])
def test_a_reopen_that_fails_the_first_time_is_retried(
    how: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window must never stay empty because the driver was not ready to open the camera
    again right after it was released."""
    monkeypatch.setattr("src.camera.WARMUP_FRAMES", 1)
    monkeypatch.setattr("src.camera.REOPEN_DELAY", 0.01)
    factory = FlakyReopen({1920: 1 / 8, 1280: 1 / 100, 640: 1 / 100}, failures=1, how=how)
    config = CameraConfig(mirror=False, auto_fallback=True)
    cam = ThreadedCamera(config, factory, fps_probe_frames=4)
    with caplog.at_level("WARNING"), cam:
        assert cam.frame_size == (1280, 720)
        assert cam.read(timeout=1.0) is not None


def test_a_camera_that_cannot_be_reopened_at_a_smaller_size_keeps_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smaller sizes refuse to open (every attempt), the original size comes back."""
    monkeypatch.setattr("src.camera.WARMUP_FRAMES", 1)
    monkeypatch.setattr("src.camera.REOPEN_DELAY", 0.01)
    times = {1920: 1 / 8, 1280: 1 / 100, 640: 1 / 100}
    factory = FlakyReopen(times, refuse=(1280, 640))
    config = CameraConfig(mirror=False, auto_fallback=True)
    cam = ThreadedCamera(config, factory, fps_probe_frames=4)
    with cam:
        assert cam.frame_size == (1920, 1080)
        assert cam.read(timeout=1.0) is not None


def test_fast_cameras_and_explicit_sizes_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.camera.WARMUP_FRAMES", 1)
    with sized_camera({1920: 1 / 100, 1280: 1 / 100, 640: 1 / 100}) as cam:
        assert cam.frame_size == (1920, 1080)
    with sized_camera({1920: 1 / 8, 1280: 1 / 100, 640: 1 / 100}, auto_fallback=False) as cam:
        assert cam.frame_size == (1920, 1080)  # --width / --height: kept, however slow


def test_probe_measures_what_each_mode_really_delivers() -> None:
    fakes: list[FakeCapture] = []

    def factory(cfg: CameraConfig) -> FakeCapture:
        fakes.append(FakeCapture(frame_time=1 / 100))
        return fakes[-1]

    modes = ((1920, 1080, 60), (640, 480, 30))
    results = probe_camera(0, modes=modes, frames=10, capture_factory=factory)
    assert [r.requested for r in results] == list(modes)
    assert all(r.size == (64, 48) for r in results)  # what the camera gave, not what we asked
    assert all(50 < r.fps < 120 for r in results)  # measured: ~100 FPS fake camera
    assert all(f.released.is_set() for f in fakes)


def test_probe_reports_a_camera_that_does_not_open() -> None:
    results = probe_camera(
        0, modes=((640, 480, 30),), capture_factory=lambda cfg: FakeCapture(opened=False)
    )
    assert results[0].size is None and results[0].fps == 0.0


def test_fourcc_name() -> None:
    assert fourcc_name(float(int.from_bytes(b"MJPG", "little"))) == "MJPG"
    assert fourcc_name(0.0) == "?"
