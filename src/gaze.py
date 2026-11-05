"""Gaze estimation for --eyes: a head-aware gaze model, its calibration and the fixation filter.

Where you look on the screen depends on three things a webcam can see:

    where the eyes point inside the head ...... iris offset in the eye (tiny: ~0.1 eye width
                                                across a laptop screen)
    where the head points ..................... yaw / pitch from MediaPipe's face transform
    where the head is ......................... x / y / distance in cm

A model of the iris alone (the first version) is only right while the head stays exactly where
it was during calibration. Turn the head by 3 degrees while looking at the same spot and the
eyes counter-rotate by 3 degrees - that alone moved its cursor by ~3 cm. On a laptop the head
never stays still, which is why the cursor "flew wherever it wanted".

So the calibration has three phases:

    1. grid ...... 9 dots, head still: how the iris maps onto the screen
    2. head ...... one dot, the head moves around: how head motion must be compensated
    3. check ..... 4 new dots: the honest accuracy, measured on points the fit never saw

Even then a laptop webcam gives a few centimetres of accuracy at best (one pixel of iris noise is
about 100 screen pixels). The fixation filter therefore treats gaze the way the eyes work: they
rest on a point (fixation) or jump (saccade). The cursor jumps with the eyes, settles, and then
stays put until the eyes clearly jump again; looking at the cursor itself never drags it away.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
log = logging.getLogger(__name__)

MODEL_VERSION = 2
N_FEATURES = 6  # iris x, iris y, head yaw, head pitch, head x, head y (see GAZE_FEATURES)
N_TERMS = 8  # intercept, eye x/y, head x/y, three quadratic eye terms
# Spreads below these (eye term in eye widths, head term in cm) count as "did not vary" when the
# terms are standardised, so a head held still during calibration cannot get huge,
# noise-fitted coefficients.
MIN_SCALE = np.array([0.02, 0.02, 1.5, 1.5])
REFERENCE_DISTANCE = 60.0  # cm, used when the distance is unknown

# 3x3 grid in screen fractions, centre first: that is where the eyes are when it starts.
GRID_TARGETS: tuple[tuple[float, float], ...] = (
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
VALIDATION_TARGETS: tuple[tuple[float, float], ...] = (
    (0.3, 0.3),
    (0.7, 0.3),
    (0.7, 0.7),
    (0.3, 0.7),
)
# Head phase: where to turn the head next, as an arrow next to the dot (screen orientation).
# Every direction matters: in simulation, turning only left/right during this phase left 4x the
# error for later up/down head movements (~180 px instead of ~40).
HEAD_MOVES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("left", (-1, 0)),
    ("right", (1, 0)),
    ("up", (0, -1)),
    ("down", (0, 1)),
)
HEAD_PHASE_WEIGHT = 3.0  # the head phase counts as much as this many grid dots in the fit
INSTRUCTIONS = {
    "grid": "Look at the dot until it moves on. Keep your head still.",
    "head": "Keep your eyes on the dot and slowly turn your head where the arrow points.",
    "check": "Checking accuracy: look at each dot until it moves on.",
}
MIN_HEAD_RANGE = (3.0, 2.0)  # deg, yaw / pitch spread below which compensation stays weak

Sample = tuple[Sequence[float], float]  # (gaze features, viewing distance in cm)


def physical_terms(
    features: FloatArray, distances: FloatArray, reference: float, eye_center: FloatArray
) -> FloatArray:
    """(n, 6) gaze features + distances -> (n, 4): eye term x/y and head term x/y.

    The gaze point = eye position + distance * tan(head angle + eye angle). Linearised:

        head term = head position + distance * tan(head angle)   [cm on the screen plane]
        eye term  = (iris offset - eye centre) * distance / calibration distance

    The eye centre is the mean iris offset during calibration: about where the iris sits when
    the eye looks straight ahead in the head. Measured from the eye corners that is not zero
    (on real faces the iris sits ~0.07 eye widths high), and an uncentred offset would grow
    with the distance as if it were an eye angle: in simulation sitting 15 cm closer then put
    the cursor ~150 px off instead of ~40.

    The head term is where the head's straight-ahead line meets the screen. Folding rotation
    and position into it matters: during calibration they move together (turning the head
    moves the eyes too), so fitted as separate features their weights came out arbitrary, and
    in simulation ordinary head movement afterwards put the cursor ~6 cm off. The physics fixes
    the mix; only the overall gains are fitted. A wrong metric scale in MediaPipe's pose (it
    assumes an average face) scales the head term as a whole, which the fit absorbs."""
    d = np.clip(distances, 20.0, 200.0)
    ratio = d / reference
    yaw, pitch = np.radians(features[:, 2]), np.radians(features[:, 3])
    return np.column_stack(
        [
            (features[:, 0] - eye_center[0]) * ratio,
            (features[:, 1] - eye_center[1]) * ratio,
            features[:, 4] + d * np.tan(yaw),
            features[:, 5] + d * np.tan(pitch),
        ]
    )


def _expand(z: FloatArray) -> FloatArray:
    """Design matrix: intercept, the standardised terms and quadratic eye terms (the eye is a
    ball and the screen is flat, so the mapping bends towards the edges)."""
    ex, ey = z[:, 0], z[:, 1]
    return np.column_stack([np.ones(len(z)), z, ex * ey, ex * ex, ey * ey])


@dataclass
class GazeModel:
    """Gaze features (+ viewing distance) -> screen pixels.

    Linear in the head term and quadratic in the eye term (see ``physical_terms``), fitted by
    weighted ridge regression on standardised terms with outlier rejection."""

    mean: FloatArray  # (4,) of the physical terms
    scale: FloatArray  # (4,)
    coef: FloatArray  # (8, 2)
    screen_size: tuple[int, int]
    distance: float = REFERENCE_DISTANCE  # cm, median viewing distance during calibration
    eye_center: FloatArray = field(default_factory=lambda: np.zeros(2))  # mean iris offset
    fit_error: float = 0.0  # px, RMS over the calibration dots
    validation_error: float | None = None  # px, mean over the check dots (None = not measured)
    noise: float = 0.0  # px, per-axis standard deviation of single-frame predictions
    head_range: tuple[float, float] = (0.0, 0.0)  # deg, yaw / pitch spread in the head phase

    @classmethod
    def fit(
        cls,
        features: npt.ArrayLike,
        targets: npt.ArrayLike,
        screen_size: tuple[int, int],
        distances: npt.ArrayLike | None = None,
        weights: npt.ArrayLike | None = None,
        ridge: float = 0.003,
        outlier_sigma: float = 3.5,
    ) -> GazeModel:
        f = np.asarray(features, dtype=np.float64).reshape(-1, N_FEATURES)
        y = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
        n = len(f)
        if n < 20:
            raise ValueError(f"a gaze model needs at least 20 samples, got {n}")
        d = np.full(n, REFERENCE_DISTANCE) if distances is None else np.asarray(distances, float)
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
        reference = float(np.median(d))
        eye_center = np.average(f[:, :2], axis=0, weights=w)
        terms = physical_terms(f, d, reference, eye_center)
        mean = np.average(terms, axis=0, weights=w)
        spread = np.sqrt(np.average((terms - mean) ** 2, axis=0, weights=w))
        scale = np.maximum(spread, MIN_SCALE)
        a = _expand((terms - mean) / scale)

        # Fit, drop outliers (a glance away, a tracking glitch), refit. The robust sigma comes
        # from the median residual: for 2D Gaussian noise the median radius is 1.1774 sigma.
        keep = np.ones(n, dtype=bool)
        coef = _ridge_solve(a, y, w, ridge)
        for _ in range(2):
            residual = np.linalg.norm(a @ coef - y, axis=1)
            sigma = float(np.median(residual[keep])) / 1.1774
            new_keep = residual <= max(outlier_sigma * sigma, 0.01 * screen_size[0])
            if new_keep.sum() < 20 or np.array_equal(new_keep, keep):
                break
            keep = new_keep
            coef = _ridge_solve(a[keep], y[keep], w[keep], ridge)
        residual = np.linalg.norm(a @ coef - y, axis=1)
        fit_error = float(np.sqrt(np.average(residual[keep] ** 2, weights=w[keep])))
        screen = (int(screen_size[0]), int(screen_size[1]))
        return cls(mean, scale, coef, screen, reference, eye_center, fit_error)

    def predict_many(
        self, features: npt.ArrayLike, distances: npt.ArrayLike | None = None
    ) -> FloatArray:
        f = np.asarray(features, dtype=np.float64).reshape(-1, N_FEATURES)
        d = np.full(len(f), self.distance) if distances is None else np.asarray(distances, float)
        terms = physical_terms(f, d.reshape(-1), self.distance, self.eye_center)
        z = (terms - self.mean) / self.scale
        result: FloatArray = _expand(z) @ self.coef
        return result

    def predict(
        self, features: Sequence[float], distance: float | None = None
    ) -> tuple[float, float]:
        x, y = self.predict_many([features], None if distance is None else [distance])[0]
        return float(x), float(y)

    # -- persistence ----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MODEL_VERSION,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "screen_size": list(self.screen_size),
            "distance": self.distance,
            "eye_center": self.eye_center.tolist(),
            "fit_error": self.fit_error,
            "validation_error": self.validation_error,
            "noise": self.noise,
            "head_range": list(self.head_range),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GazeModel:
        if data.get("version") != MODEL_VERSION:
            raise ValueError("made by an older version")
        validation = data.get("validation_error")
        head_range = data.get("head_range", (0.0, 0.0))
        return cls(
            np.asarray(data["mean"], dtype=np.float64).reshape(len(MIN_SCALE)),
            np.asarray(data["scale"], dtype=np.float64).reshape(len(MIN_SCALE)),
            np.asarray(data["coef"], dtype=np.float64).reshape(N_TERMS, 2),
            (int(data["screen_size"][0]), int(data["screen_size"][1])),
            float(data["distance"]),
            np.asarray(data["eye_center"], dtype=np.float64).reshape(2),
            float(data.get("fit_error", 0.0)),
            None if validation is None else float(validation),
            float(data.get("noise", 0.0)),
            (float(head_range[0]), float(head_range[1])),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, screen_size: tuple[int, int]) -> GazeModel | None:
        """The saved model, or None if missing, unreadable, outdated or for another screen."""
        try:
            model = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            log.warning("Ignoring saved gaze calibration %s (%s); recalibrating.", path, exc)
            return None
        if model.screen_size != tuple(screen_size):
            log.info("Saved gaze calibration is for another screen size; recalibrating.")
            return None
        return model


def _ridge_solve(a: FloatArray, y: FloatArray, w: FloatArray, ridge: float) -> FloatArray:
    w = w * (len(w) / w.sum())  # weights sum to n, so the penalty means the same for any n
    aw = a * w[:, None]
    penalty = ridge * len(a) * np.eye(a.shape[1])
    penalty[0, 0] = 0.0  # never shrink the intercept
    coef: FloatArray = np.linalg.solve(a.T @ aw + penalty, aw.T @ y)
    return coef


# -- calibration procedure ------------------------------------------------------------------
@dataclass
class _Step:
    target: tuple[float, float]  # px
    phase: str  # "grid", "head" or "check"
    duration: float  # s of sampling after the settle time
    samples: list[Sample] = field(default_factory=list)


class CalibrationSession:
    """Runs the three calibration phases and collects (features, distance) samples.

    Independent of drawing, so the timing and the fit are testable without a screen."""

    def __init__(
        self,
        screen_size: tuple[int, int],
        settle: float,
        sample: float,
        head_seconds: float,
        grid: tuple[tuple[float, float], ...] = GRID_TARGETS,
        check: tuple[tuple[float, float], ...] = VALIDATION_TARGETS,
    ) -> None:
        width, height = screen_size
        self.screen_size = screen_size
        self.settle = settle

        def px(fx: float, fy: float) -> tuple[float, float]:
            return fx * (width - 1), fy * (height - 1)

        self.steps = [_Step(px(*f), "grid", sample) for f in grid]
        if head_seconds > 0:
            self.steps.append(_Step(px(0.5, 0.5), "head", head_seconds))
        self.steps += [_Step(px(*f), "check", sample) for f in check]
        self.index = 0
        self._started: float | None = None
        self._elapsed = 0.0

    @property
    def done(self) -> bool:
        return self.index >= len(self.steps)

    @property
    def step(self) -> _Step | None:
        return None if self.done else self.steps[self.index]

    @property
    def target(self) -> tuple[float, float] | None:
        step = self.step
        return None if step is None else step.target

    @property
    def phase(self) -> str:
        step = self.step
        return "done" if step is None else step.phase

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS.get(self.phase, "")

    @property
    def head_hint(self) -> tuple[str, tuple[int, int]] | None:
        """In the head phase: which way to turn the head now (name, screen direction)."""
        step = self.step
        if step is None or step.phase != "head":
            return None
        into = max(self._elapsed - self.settle, 0.0) / step.duration
        return HEAD_MOVES[min(int(into * len(HEAD_MOVES)), len(HEAD_MOVES) - 1)]

    @property
    def progress(self) -> float:
        """0..1 within the current dot (for the shrinking ring)."""
        step = self.step
        if step is None:
            return 1.0
        return min(self._elapsed / (self.settle + step.duration), 1.0)

    def update(self, t: float, sample: Sample | None) -> None:
        step = self.step
        if step is None:
            return
        if self._started is None:
            self._started = t
        self._elapsed = t - self._started
        if self._elapsed >= self.settle and sample is not None:
            step.samples.append((tuple(sample[0]), float(sample[1])))
        if self._elapsed >= self.settle + step.duration:
            self.index += 1
            self._started, self._elapsed = t, 0.0

    def result(self, min_samples: int = 5) -> GazeModel:
        usable = [s for s in self.steps if len(s.samples) >= min_samples]
        grid = [s for s in usable if s.phase == "grid"]
        if len(grid) < 6:
            total = sum(s.phase == "grid" for s in self.steps)
            raise ValueError(
                f"only {len(grid)} of {total} dots got enough samples (face lost or eyes closed?)"
            )
        train = [s for s in usable if s.phase != "check"]
        check = [s for s in usable if s.phase == "check"]
        model = _fit_steps(train, self.screen_size)
        if check:  # judged on dots the fit never saw...
            validation = float(np.mean([_dot_error(model, s) for s in check]))
            model = _fit_steps(train + check, self.screen_size)  # ...then use them too
            model.validation_error = validation
        model.noise = _jitter([s for s in usable if s.phase != "head"], model)
        head = [s for s in usable if s.phase == "head"]
        if head:
            angles = np.array([sample[0][2:4] for sample in head[0].samples], dtype=np.float64)
            spread = angles.std(axis=0)
            model.head_range = (float(spread[0]), float(spread[1]))
        return model


def _fit_steps(steps: list[_Step], screen_size: tuple[int, int]) -> GazeModel:
    features, distances, targets, weights = [], [], [], []
    for step in steps:
        weight = (HEAD_PHASE_WEIGHT if step.phase == "head" else 1.0) / len(step.samples)
        for f, d in step.samples:
            features.append(f)
            distances.append(d)
            targets.append(step.target)
            weights.append(weight)
    return GazeModel.fit(features, targets, screen_size, distances, weights)


def _predictions(model: GazeModel, step: _Step) -> FloatArray:
    return model.predict_many([f for f, _ in step.samples], [d for _, d in step.samples])


def _dot_error(model: GazeModel, step: _Step) -> float:
    """Distance (px) between a dot and where the model puts the median gaze for it."""
    centre = np.median(_predictions(model, step), axis=0)
    return float(np.linalg.norm(centre - np.asarray(step.target)))


def _jitter(steps: list[_Step], model: GazeModel) -> float:
    """Per-axis standard deviation (px) of single-frame predictions while looking at one dot.

    Robust: the median distance from each dot's median prediction, converted to sigma (for a
    2D Gaussian the median radius is 1.1774 sigma), so a glance away does not inflate it."""
    distances = []
    for step in steps:
        p = _predictions(model, step)
        distances.append(np.linalg.norm(p - np.median(p, axis=0), axis=1))
    if not distances:
        return 0.0
    return float(np.median(np.concatenate(distances)) / math.sqrt(2.0 * math.log(2.0)))


# -- fixation filter ------------------------------------------------------------------------
class FixationFilter:
    """Turns noisy gaze points into a cursor that jumps with the eyes and otherwise stays put.

    The eyes either rest on a point (fixation) or jump (saccade). Per frame:

    * A point within ``radius`` of the current fixation belongs to it. During the first
      ``settle`` seconds the fixation is the mean of its points (the cursor lands, then
      steadies); after that it is frozen. Freezing is what stops the classic gaze-cursor
      runaway: the cursor lands a bit off, you look at it, the estimate moves by the same error
      again, you look again...
    * Saccade: the last ``saccade_frames`` points are all outside the radius, agree with each
      other, and got there in a fast step from the points just before. Scattered outliers never
      agree; slow wandering of the estimate (webcam noise is strongly correlated from frame to
      frame) never makes a fast step. In simulation the step test cut false jumps from ~30 to
      under 1 per minute.
    * Dwell: the mean of the last ``dwell`` seconds is outside the radius. This catches real
      gaze shifts too small to stand out from the noise in two frames.

    ``jumped`` is True after an update that started a new fixation.
    """

    def __init__(
        self, radius_px: float, saccade_frames: int = 2, settle: float = 0.6, dwell: float = 0.5
    ) -> None:
        self.radius = radius_px
        self.saccade_frames = max(1, saccade_frames)
        self.settle = settle
        self.dwell = dwell
        self.jumped = False
        self._center: FloatArray | None = None
        self._points: list[FloatArray] = []  # the current fixation's points while it settles
        self._started = 0.0
        self._recent: deque[tuple[float, FloatArray]] = deque()

    @property
    def center(self) -> tuple[float, float] | None:
        c = self._center
        return None if c is None else (float(c[0]), float(c[1]))

    def reset(self) -> None:
        self._center = None
        self._points.clear()
        self._recent.clear()
        self.jumped = False

    def update(self, point: tuple[float, float], t: float) -> tuple[float, float]:
        p = np.asarray(point, dtype=np.float64)
        self.jumped = False
        self._recent.append((t, p))
        while self._recent and self._recent[0][0] < t - max(self.dwell, 0.2):
            self._recent.popleft()
        if self._center is None:
            self._start([p], t)
        else:
            landing = self._saccade()
            if landing is None:
                landing = self._dwell(t)
            settling = t - self._started <= self.settle
            if landing is not None:
                self._start(landing, t)
            elif settling and np.linalg.norm(p - self._center) <= self.radius:
                self._points.append(p)
                self._center = np.mean(self._points, axis=0)
        assert self._center is not None
        return float(self._center[0]), float(self._center[1])

    def _saccade(self) -> list[FloatArray] | None:
        k = self.saccade_frames
        points = [q for _, q in self._recent]
        if len(points) < k:
            return None
        assert self._center is not None
        # After a gap (a blink, a lost face) the step is measured from the fixation itself:
        # people often move their eyes during a blink.
        last, before = points[-k:], points[-2 * k : -k] or [self._center]
        if any(np.linalg.norm(q - self._center) <= self.radius for q in last):
            return None
        landing = np.mean(last, axis=0)
        if any(np.linalg.norm(q - landing) > self.radius for q in last):
            return None  # scattered: noise or a glitch
        if np.linalg.norm(landing - np.mean(before, axis=0)) <= self.radius:
            return None  # no fast step: the estimate wandered off slowly
        return last

    def _dwell(self, t: float) -> list[FloatArray] | None:
        if self.dwell <= 0 or t - self._started < self.dwell:
            return None
        if t - self._recent[0][0] < 0.8 * self.dwell:
            return None  # not enough history (e.g. frames were dropped)
        points = [q for _, q in self._recent]
        assert self._center is not None
        if np.linalg.norm(np.mean(points, axis=0) - self._center) <= self.radius:
            return None
        return points

    def _start(self, points: list[FloatArray], t: float) -> None:
        self._points = list(points)
        self._center = np.mean(self._points, axis=0)
        self._started = t
        self._recent.clear()
        self._recent.extend((t, q) for q in points)
        self.jumped = True
