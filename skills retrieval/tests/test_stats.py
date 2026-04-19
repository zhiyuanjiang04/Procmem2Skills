import numpy as np

from skills_retrieval.stats import bootstrap_rate_ci


def test_ci_zero_variance():
    mean, lo, hi = bootstrap_rate_ci([1, 1, 1, 1], n_boot=500, seed=0)
    assert mean == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_ci_mean_matches_sample_mean():
    values = [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]  # sample mean = 0.6
    mean, lo, hi = bootstrap_rate_ci(values, n_boot=2000, seed=0)
    assert abs(mean - 0.6) < 1e-9
    assert 0.25 < lo < 0.6
    assert 0.6 < hi < 0.95


def test_ci_brackets_mean():
    values = [1, 0] * 20
    mean, lo, hi = bootstrap_rate_ci(values, n_boot=1000, seed=42)
    assert lo <= mean <= hi


def test_ci_empty_returns_nans():
    import math
    mean, lo, hi = bootstrap_rate_ci([], n_boot=100, seed=0)
    assert math.isnan(mean)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_ci_deterministic_with_seed():
    values = [0, 1] * 15
    a = bootstrap_rate_ci(values, n_boot=500, seed=7)
    b = bootstrap_rate_ci(values, n_boot=500, seed=7)
    assert a == b
