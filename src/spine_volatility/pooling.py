"""Canonical per-spine pairing and weighting primitives.

All arrays here follow one convention throughout the package: shape
``(n_timepoints, n_spines)``, with ``-1`` (or any non-positive value)
marking a missing measurement.

These few functions were independently reimplemented, with only cosmetic
variation, close to a dozen times across the original notebooks
(``collect_pairs_same_feat_pooled`` alone: three independent
implementations; ``mean_per_spine`` / ``pooled_values_with_spine_weights``:
nine redefinitions across two notebooks). Every spine contributes each of
its valid measurements weighted by ``1/n_valid`` so that spines followed for
more time points don't dominate a pooled estimate.
"""

import numpy as np


def collect_pairs_same_feat_pooled(data: np.ndarray, delta_steps: int) -> np.ndarray:
    """Pairs ``(x(t), x(t+delta_steps))`` for one feature, pooled across all spines.

    Returns an array of shape ``(2, n_pairs)``.
    """
    n_times, n_spines = data.shape
    pairs = []
    for spine in range(n_spines):
        for t in range(n_times - delta_steps):
            x, y = data[t, spine], data[t + delta_steps, spine]
            if x > 0 and y > 0:
                pairs.append((x, y))
    if not pairs:
        return np.empty((2, 0))
    return np.asarray(pairs, dtype=float).T


def collect_pairs_diff_feat_pooled(data1: np.ndarray, data2: np.ndarray, delta_steps: int) -> np.ndarray:
    """Pairs ``(feature1(t), feature2(t+delta_steps))``, pooled across all spines."""
    n_times, n_spines = data1.shape
    pairs = []
    for spine in range(n_spines):
        for t in range(n_times - delta_steps):
            x, y = data1[t, spine], data2[t + delta_steps, spine]
            if x > 0 and y > 0:
                pairs.append((x, y))
    if not pairs:
        return np.empty((2, 0))
    return np.asarray(pairs, dtype=float).T


def weighted_mean_std(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean and standard deviation. Scale-invariant in ``w`` (dividing
    by ``sum(w)`` internally), so callers may pass either raw 1/n_valid
    weights or weights pre-normalized to sum to 1."""
    wsum = np.sum(w)
    mean = np.sum(w * x) / wsum
    var = np.sum(w * (x - mean) ** 2) / wsum
    return mean, float(np.sqrt(var))


def pooled_values_with_spine_weights(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All valid raw values of one feature, each weighted by 1/(valid count for its spine).

    Returns ``(values, weights)``. Weights are *not* normalized to sum to 1
    -- each spine contributes total weight 1 (split across its own valid
    measurements), so the total weight mass reflects the number of spines.
    This matters wherever weights feed an extensive quantity (e.g. a
    log-likelihood sum for AIC comparison in :mod:`distributions`); for a
    weighted mean/std or a KDE, the result is unaffected by this choice.
    """
    values, weights = [], []
    for spine in range(data.shape[1]):
        valid = data[:, spine]
        valid = valid[valid > 0]
        n = len(valid)
        if n > 0:
            values.extend(valid)
            weights.extend([1.0 / n] * n)
    return np.asarray(values, dtype=float), np.asarray(weights, dtype=float)


def pooled_triplets_with_spine_weights(
    data1: np.ndarray, data2: np.ndarray, data3: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Matched (feature1, feature2, feature3) triplets, each weighted by 1/(valid count).

    A time point only contributes if all three features are valid there.
    Returns ``(x, y, z, weights)``; weights are not normalized (see
    :func:`pooled_values_with_spine_weights`).
    """
    x_vals, y_vals, z_vals, weights = [], [], [], []
    for spine in range(data1.shape[1]):
        v1, v2, v3 = data1[:, spine], data2[:, spine], data3[:, spine]
        mask = (v1 > 0) & (v2 > 0) & (v3 > 0)
        v1, v2, v3 = v1[mask], v2[mask], v3[mask]
        n = len(v1)
        if n > 0:
            x_vals.extend(v1)
            y_vals.extend(v2)
            z_vals.extend(v3)
            weights.extend([1.0 / n] * n)
    return (
        np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float),
        np.asarray(z_vals, dtype=float), np.asarray(weights, dtype=float),
    )
