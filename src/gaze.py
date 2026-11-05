"""Gaze calibration (eye feature -> screen pixels) and fixation-based cursor stabilisation."""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
log = logging.getLogger(__name__)

# 3x3 grid in screen fractions, centre first: that is where the eyes are when it starts.
CALIBRATION_TARGETS: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (0.1, 0.1),
    (0.5, 0.1),
    (0.9, 0.1),
    (0.9, 0.5),
    (0.9, 0.9),
    (0.5, 0.9),
    (0.1, 0.9),
    (0.1, 0.5),
)


def _design(z: FloatArray) -> FloatArray:
    x, y = z[:, 0], z[:, 1]
    return np.column_stack([np.ones_like(x), x, y, x * y, x * x, y * y])


@dataclass
class GazeCalibration:
    """Second-order polynomial from the gaze feature (mean iris offset) to screen pixels.

    Fitted with a little ridge regularisation on standardised features: 9 noisy points and 6
    coefficients per axis would otherwise let the quadratic terms swing wildly at the edges.
    """

    mean: FloatArray
    scale: FloatArray
    coef: FloatArray  # (6, 2)
    screen_size: tuple[int, int]
    rms_error: float = 0.0  # px, on the calibration points themselves

    @classmethod
    def fit(
        cls,
        features: npt.ArrayLike,
        targets: npt.ArrayLike,
        screen_size: tuple[int, int],
        ridge: float = 1e-2,
    ) -> GazeCalibration:
        f = np.asarray(features, dtype=np.float64).reshape(-1, 2)
        y = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
        if len(f) < 6:
            raise ValueError("a gaze calibration needs at least 6 points")
        mean = f.mean(axis=0)
        scale = f.std(axis=0)
        scale[scale < 1e-9] = 1.0
        a = _design((f - mean) / scale)
        penalty = ridge * np.eye(a.shape[1])
        penalty[0, 0] = 0.0  # never shrink the intercept
        coef = np.linalg.solve(a.T @ a + penalty, a.T @ y)
        residual = a @ coef - y
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        return cls(mean, scale, coef, (int(screen_size[0]), int(screen_size[1])), rms)

    def predict(self, feature: tuple[float, float]) -> tuple[float, float]:
        z = (np.asarray(feature, dtype=np.float64).reshape(1, 2) - self.mean) / self.scale
        x, y = (_design(z) @ self.coef)[0]
        return float(x), float(y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "screen_size": list(self.screen_size),
            "rms_error": self.rms_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GazeCalibration:
        return cls(
            np.asarray(data["mean"], dtype=np.float64),
            np.asarray(data["scale"], dtype=np.float64),
            np.asarray(data["coef"], dtype=np.float64).reshape(6, 2),
            (int(data["screen_size"][0]), int(data["screen_size"][1])),
            float(data.get("rms_error", 0.0)),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, screen_size: tuple[int, int]) -> GazeCalibration | None:
        """The saved calibration, or None if missing, unreadable or made for another screen."""
        try:
            cal = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("Ignoring unreadable gaze calibration %s: %s", path, exc)
            return None
        if cal.screen_size != tuple(screen_size):
            log.info("Saved gaze calibration is for another screen size; recalibrating.")
            return None
        return cal


class CalibrationSession:
    """Shows each target for settle + sample seconds and collects gaze features meanwhile.

    Independent of drawing, so the timing and the fit are testable without a screen. Each dot
    uses the *median* of its samples, which shrugs off blinks and tracking glitches.
    """

    def __init__(
        self,
        screen_size: tuple[int, int],
        settle: float,
        sample: float,
        targets: tuple[tuple[float, float], ...] = CALIBRATION_TARGETS,
    ) -> None:
        width, height = screen_size
        self.screen_size = screen_size
        self.settle = settle
        self.sample = sample
        self.targets = [(fx * (width - 1), fy * (height - 1)) for fx, fy in targets]
        self.samples: list[list[tuple[float, float]]] = [[] for _ in self.targets]
        self.index = 0
        self._started: float | None = None
        self._elapsed = 0.0

    @property
    def done(self) -> bool:
        return self.index >= len(self.targets)

    @property
    def target(self) -> tuple[float, float] | None:
        return None if self.done else self.targets[self.index]

    @property
    def progress(self) -> float:
        """0..1 within the current dot (for the shrinking ring)."""
        return min(self._elapsed / (self.settle + self.sample), 1.0)

    def update(self, t: float, feature: tuple[float, float] | None) -> None:
        if self.done:
            return
        if self._started is None:
            self._started = t
        self._elapsed = t - self._started
        if self._elapsed >= self.settle and feature is not None:
            self.samples[self.index].append(feature)
        if self._elapsed >= self.settle + self.sample:
            self.index += 1
            self._started, self._elapsed = t, 0.0

    def result(self, min_samples: int = 5) -> GazeCalibration:
        features, targets = [], []
        for samples, target in zip(self.samples, self.targets, strict=True):
            if len(samples) >= min_samples:
                features.append(
                    (
                        statistics.median(s[0] for s in samples),
                        statistics.median(s[1] for s in samples),
                    )
                )
                targets.append(target)
        if len(features) < 6:
            raise ValueError(
                f"only {len(features)} of {len(self.targets)} dots got enough samples "
                "(face lost or eyes closed?)"
            )
        return GazeCalibration.fit(features, targets, self.screen_size)


class FixationFilter:
    """Keeps the cursor still during a fixation and jumps on a saccade.

    The eyes either rest on a point (fixation, small noise) or jump (saccade). Webcam gaze
    noise is ~100 px, so a plain low-pass filter would have to be very slow to hide it, adding
    lag to every jump. Here gaze inside the fixation radius only nudges the cursor (averaging
    the noise away), while ``saccade_frames`` samples in a row outside the radius move it at
    once to their mean (velocity-threshold identification, I-VT).
    """

    def __init__(self, radius_px: float, saccade_frames: int = 2, smoothing: float = 0.12) -> None:
        self.radius = radius_px
        self.saccade_frames = max(1, saccade_frames)
        self.smoothing = smoothing
        self._center: FloatArray | None = None
        self._pending: list[FloatArray] = []

    def reset(self) -> None:
        self._center = None
        self._pending.clear()

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        p = np.asarray(point, dtype=np.float64)
        if self._center is None:
            self._center = p.copy()
        elif np.linalg.norm(p - self._center) > self.radius:
            self._pending.append(p)
            if len(self._pending) >= self.saccade_frames:
                self._center = np.mean(self._pending, axis=0)
                self._pending.clear()
        else:
            self._pending.clear()
            self._center = self._center + self.smoothing * (p - self._center)
        return float(self._center[0]), float(self._center[1])
