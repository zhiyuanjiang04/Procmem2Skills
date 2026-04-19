from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def bootstrap_rate_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of a 0/1 or float sequence.

    Returns (sample_mean, lo, hi) where [lo, hi] is the (1-alpha)/2..1-(1-alpha)/2
    percentile interval over n_boot resamples. NaNs on empty input.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return (math.nan, math.nan, math.nan)
    mean = float(arr.mean())
    if n == 1 or np.allclose(arr, arr[0]):
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = 1.0 - ci
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return (mean, lo, hi)
