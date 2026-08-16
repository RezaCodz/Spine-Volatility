"""Auto- and cross-correlation functions with bootstrap error bars.

Methods: "For each morphological feature, we computed auto-correlations by
pooling all same-spine measurement pairs separated by a given time lag.
Cross-correlations were calculated analogously across different features at
matched time points. Pearson correlation coefficients were used throughout.
To quantify uncertainty, we applied nonparametric bootstrap resampling
(n=100) of the paired observations and report the standard error across
resamples."
"""

import numpy as np

from .pooling import collect_pairs_diff_feat_pooled, collect_pairs_same_feat_pooled

NUM_BOOTSTRAP = 100


def correlation_from_pairs(pairs: np.ndarray) -> float:
    if pairs.shape[1] < 2:
        return np.nan
    x, y = pairs[0, :], pairs[1, :]
    std_x, std_y = np.std(x), np.std(y)
    if std_x == 0 or std_y == 0:
        return np.nan
    cov = np.mean(x * y) - np.mean(x) * np.mean(y)
    return cov / (std_x * std_y)


def _draw_from_set(pairs: np.ndarray, rng: np.random.Generator, num_samples: int = -1) -> np.ndarray:
    if num_samples == -1:
        num_samples = pairs.shape[1]
    idx = rng.integers(0, pairs.shape[1], size=num_samples)
    return pairs[:, idx]


def bootstrap_error(
    pairs: np.ndarray, rng: np.random.Generator, num_bootstrap: int = NUM_BOOTSTRAP
) -> float:
    if pairs.shape[1] < 2:
        return np.nan
    r = np.empty(num_bootstrap)
    for i in range(num_bootstrap):
        r[i] = correlation_from_pairs(_draw_from_set(pairs, rng))
    return float(np.nanstd(r))


def auto_correlation(
    data: np.ndarray, times: np.ndarray, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Auto-correlation of one feature at each lag in ``times`` (lag 0 first).

    Returns ``(times, r, r_error)``.
    """
    rng = rng or np.random.default_rng(0)
    r = np.empty(len(times))
    r_err = np.empty(len(times))
    for lag in range(len(times)):
        pairs = collect_pairs_same_feat_pooled(data, lag)
        r[lag] = correlation_from_pairs(pairs)
        r_err[lag] = bootstrap_error(pairs, rng)
    return np.asarray(times, dtype=float), r, r_err


def cross_correlation(
    data1: np.ndarray, data2: np.ndarray, times: np.ndarray, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-correlation between two features across positive and negative lags.

    ``data2`` lagging ``data1`` gives the positive-lag half; ``data1`` lagging
    ``data2`` gives the negative-lag half (time-reversed). Returns
    ``(lags, r, r_error)`` ordered from the most negative lag to the most
    positive.
    """
    rng = rng or np.random.default_rng(0)
    times = np.asarray(times, dtype=float)
    lags = np.concatenate((-times[:0:-1], times))
    r = np.empty_like(lags)
    r_err = np.empty_like(lags)

    i = 0
    for lag in range(len(times) - 1, 0, -1):
        pairs = collect_pairs_diff_feat_pooled(data2, data1, lag)
        r[i] = correlation_from_pairs(pairs)
        r_err[i] = bootstrap_error(pairs, rng)
        i += 1
    for lag in range(len(times)):
        pairs = collect_pairs_diff_feat_pooled(data1, data2, lag)
        r[i] = correlation_from_pairs(pairs)
        r_err[i] = bootstrap_error(pairs, rng)
        i += 1
    return lags, r, r_err


def auto_pair_counts(data: np.ndarray, n_lags: int) -> list[int]:
    """Number of same-spine pairs at each lag 1..n_lags-1 (Table S1 rows)."""
    return [collect_pairs_same_feat_pooled(data, lag).shape[1] for lag in range(1, n_lags)]


def cross_pair_counts(data1: np.ndarray, data2: np.ndarray, n_lags: int) -> list[int]:
    """Number of cross pairs at shift 0..n_lags-1 (Table S2 rows).

    Matches the manuscript's Table S2 exactly: despite the "+-Delta t"
    column headers, each column reports one direction's pair count (feature2
    at t paired with feature1 at t+lag), not the sum of both directions --
    confirmed by reproducing the published counts.
    """
    return [collect_pairs_diff_feat_pooled(data2, data1, lag).shape[1] for lag in range(n_lags)]
