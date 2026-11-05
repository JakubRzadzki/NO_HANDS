from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from src.smoother import OneEuroFilter, smoothing_factor

FPS = 60.0


def test_smoothing_factor_matches_rc_low_pass() -> None:
    dt, cutoff = 1 / 60, 1.0
    tau = 1 / (2 * math.pi * cutoff)
    assert smoothing_factor(cutoff, dt) == pytest.approx(dt / (dt + tau))


def test_first_sample_passes_through() -> None:
    f = OneEuroFilter()
    assert f([3.0, 4.0], 0.0).tolist() == [3.0, 4.0]


def test_step_response_is_monotonic_first_order_and_converges() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out = [float(f(0.0, 0.0))] + [float(f(1.0, i / FPS)) for i in range(1, 181)]
    assert all(b >= a for a, b in itertools.pairwise(out))  # no overshoot, no ringing
    tau = 1 / (2 * math.pi * 1.0)  # ~0.159 s
    assert 0.55 < out[round(tau * FPS)] < 0.72  # ~63% after one time constant
    assert out[-1] == pytest.approx(1.0, abs=1e-3)


def test_beta_removes_lag_during_fast_motion() -> None:
    def lag(beta: float) -> float:
        f = OneEuroFilter(min_cutoff=1.0, beta=beta)
        x = 0.0
        for i in range(31):  # 0.5 s ramp at 2000 px/s
            x = 2000.0 * i / FPS
            y = float(f(x, i / FPS))
        return x - y

    assert lag(0.05) < lag(0.0) / 10


def test_jitter_is_suppressed_at_rest() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 4.0, 600)  # ~MediaPipe jitter at rest, in screen px
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    out = np.array([float(f(500.0 + n, i / FPS)) for i, n in enumerate(noise)])
    assert np.std(out[60:]) < 0.4 * np.std(noise)


def test_vector_filter_keeps_straight_lines_straight() -> None:
    """One shared cutoff for both axes: a straight diagonal path stays on its line. Two
    independent scalar filters would lag more on the slower axis and bend the path."""
    direction = np.array([1.0, 0.2])
    shared = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    fx, fy = OneEuroFilter(min_cutoff=1.0, beta=0.05), OneEuroFilter(min_cutoff=1.0, beta=0.05)
    worst_shared = worst_independent = 0.0
    for i in range(40):
        t = i / FPS
        p = direction * 1500.0 * t
        s = shared(p, t)
        ind = np.array([float(fx(p[0], t)), float(fy(p[1], t))])
        worst_shared = max(worst_shared, abs(s[1] - 0.2 * s[0]))
        worst_independent = max(worst_independent, abs(ind[1] - 0.2 * ind[0]))
    assert worst_shared < 1e-9
    assert worst_independent > 1.0


def test_duplicate_timestamp_returns_previous_value() -> None:
    f = OneEuroFilter()
    f(0.0, 0.0)
    first = float(f(10.0, 0.1))
    assert float(f(99.0, 0.1)) == first  # same t: nothing to integrate
    assert float(f(99.0, 0.05)) == first  # out of order: ignored


def test_reset_forgets_history() -> None:
    f = OneEuroFilter()
    f(0.0, 0.0)
    f(10.0, 0.1)
    f.reset()
    assert f.value is None
    assert float(f(42.0, 5.0)) == 42.0


@pytest.mark.parametrize("kwargs", [{"min_cutoff": 0}, {"d_cutoff": -1}, {"beta": -0.1}])
def test_invalid_parameters_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(**kwargs)
