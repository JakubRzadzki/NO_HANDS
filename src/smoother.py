"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012) for N-dimensional signals.

The idea in one sentence: an exponential low-pass filter whose cutoff frequency rises with the
signal's speed, so it smooths hard while the hand is (almost) still and gets out of the way when
the hand moves fast.

    cutoff = min_cutoff + beta * |velocity|
    alpha  = 1 / (1 + tau / dt),   tau = 1 / (2 * pi * cutoff)
    x_hat  = alpha * x + (1 - alpha) * x_hat_prev

Details that matter in practice:

* alpha is recomputed from the *measured* dt of every sample, so the filter stays correct when
  the frame rate fluctuates (webcams do that all the time).
* For 2D cursor input the *magnitude* of the velocity vector drives one shared cutoff. Filtering
  x and y independently smooths a diagonal movement unevenly and bends straight lines.
* The velocity is estimated from consecutive *raw* samples, as in the reference implementation.
  A common variant differentiates against the previous *filtered* value, which mixes the
  filter's own lag into the speed estimate.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def smoothing_factor(cutoff_hz: float, dt: float) -> float:
    """EMA weight of a first-order low-pass filter with the given cutoff, sampled every dt."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Adaptive low-pass filter for scalars or vectors (e.g. a 2D cursor position)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        if beta < 0:
            raise ValueError("beta must be >= 0")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.last_cutoff = min_cutoff  # exposed for diagnostics
        self._x_hat: FloatArray | None = None  # last filtered value
        self._x_raw: FloatArray | None = None  # last raw value (velocity estimate)
        self._dx_hat: FloatArray | None = None  # low-passed velocity
        self._t: float | None = None

    def reset(self) -> None:
        """Forget history; the next sample is passed through unchanged."""
        self._x_hat = self._x_raw = self._dx_hat = None
        self._t = None
        self.last_cutoff = self.min_cutoff

    @property
    def value(self) -> FloatArray | None:
        return None if self._x_hat is None else self._x_hat.copy()

    def __call__(self, x: npt.ArrayLike, t: float) -> FloatArray:
        """Filter sample ``x`` taken at time ``t`` (seconds, monotonic clock)."""
        x_arr = np.array(x, dtype=np.float64)
        if self._x_hat is None or self._x_raw is None or self._dx_hat is None or self._t is None:
            # First sample: there is no history to smooth against yet.
            self._x_hat = x_arr.copy()
            self._x_raw = x_arr.copy()
            self._dx_hat = np.zeros_like(x_arr)
            self._t = t
            return x_arr

        dt = t - self._t
        if dt <= 0.0:
            # Duplicate or out-of-order timestamp: there is nothing new to integrate.
            return self._x_hat.copy()

        # 1) Speed estimate, itself low-passed so that jitter does not look like motion.
        a_d = smoothing_factor(self.d_cutoff, dt)
        self._dx_hat = a_d * (x_arr - self._x_raw) / dt + (1.0 - a_d) * self._dx_hat

        # 2) Faster movement -> higher cutoff -> less smoothing -> less lag.
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx_hat))
        a = smoothing_factor(cutoff, dt)
        self._x_hat = a * x_arr + (1.0 - a) * self._x_hat

        self._x_raw = x_arr
        self._t = t
        self.last_cutoff = cutoff
        return self._x_hat.copy()
