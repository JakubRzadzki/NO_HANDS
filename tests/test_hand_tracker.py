from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from config import TrackerConfig
from src.hand_tracker import HandObservation, HandTracker, ensure_model


def test_pixel_space_is_isotropic() -> None:
    """100 px right and 100 px down must be the same distance. In raw normalized coordinates
    of a 16:9 frame they differ by 1.78x, so thresholds would depend on hand orientation."""
    norm = np.zeros((21, 3))
    norm[:] = (0.5, 0.5, 0.0)
    norm[1] = (0.5 + 100 / 1280, 0.5, 0.0)
    norm[2] = (0.5, 0.5 + 100 / 720, 0.0)
    obs = HandObservation.from_normalized(norm, (1280, 720), 0.0)
    assert obs.dist(0, 1) == pytest.approx(100.0)
    assert obs.dist(0, 2) == pytest.approx(100.0)
    naive_ratio = np.hypot(*(norm[2, :2] - norm[0, :2])) / np.hypot(*(norm[1, :2] - norm[0, :2]))
    assert naive_ratio == pytest.approx(1280 / 720)


def test_ndist_is_relative_to_palm_length() -> None:
    norm = np.zeros((21, 3))
    norm[9] = (0.0, 0.2, 0.0)  # palm: wrist (0,0) -> middle MCP, 0.2 * 1000 px
    norm[4] = (0.1, 0.0, 0.0)  # 0.1 * 1000 px from the wrist
    obs = HandObservation.from_normalized(norm, (1000, 1000), 0.0)
    assert obs.palm_scale == pytest.approx(200.0)
    assert obs.ndist(0, 4) == pytest.approx(0.5)


def _zip_bundle(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("hand_detector.tflite", b"fake")
    return path


def test_ensure_model_downloads_a_task_bundle(tmp_path: Path) -> None:
    source = _zip_bundle(tmp_path / "source.task")
    target = tmp_path / "models" / "hand_landmarker.task"
    assert ensure_model(target, source.as_uri()) == target
    assert zipfile.is_zipfile(target)


def test_ensure_model_rejects_an_html_error_page(tmp_path: Path) -> None:
    page = tmp_path / "error.html"
    page.write_text("<html>Proxy error</html>")
    target = tmp_path / "models" / "hand_landmarker.task"
    with pytest.raises(RuntimeError, match="Download it manually"):
        ensure_model(target, page.as_uri())
    assert not target.exists()
    assert list(target.parent.iterdir()) == []  # no .part file left behind


def test_existing_model_is_not_downloaded_again(tmp_path: Path) -> None:
    target = _zip_bundle(tmp_path / "hand_landmarker.task")
    assert ensure_model(target, "http://invalid.invalid/model.task") == target


@pytest.fixture
def tracker() -> Iterator[HandTracker]:
    model = Path("models/hand_landmarker.task")
    if not model.is_file():
        pytest.skip("model not downloaded yet (run main.py once)")
    try:
        instance = HandTracker(TrackerConfig(model_path=str(model)))
    except (ImportError, OSError, RuntimeError) as exc:  # e.g. libEGL missing on headless Linux
        pytest.skip(f"MediaPipe unavailable here: {exc}")
    yield instance
    instance.close()


def test_tracker_survives_blank_frames_and_non_increasing_timestamps(tracker: HandTracker) -> None:
    frame = np.zeros((480, 640, 3), np.uint8)
    assert tracker.process(frame, 10.0) is None
    assert tracker.process(frame, 10.0) is None  # MediaPipe raises on equal timestamps
    assert tracker.process(frame, 9.0) is None  # ... and on going backwards
