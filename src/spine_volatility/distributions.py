"""Weighted distribution fits: raw values vs. log-normal, increments vs. Gaussian.

Supplementary Material: "The raw values of head size, neck length, and neck
width were right-skewed and better described by a log-normal than a
Gaussian distribution ... Their between-timepoint increments, by contrast,
were Gaussian ... Distributions were spine-weighted and compared by weighted
[Delta]AIC; because the variance of the increments grows with time lag, each
lag was standardized before pooling."

Ported from ``figure_1.ipynb``, which pools each lag's increments only after
standardizing by that lag's own spine-weighted mean/std -- combining
different-variance lags into one shape comparison would otherwise produce a
variance-mixture that isn't Gaussian even if every individual lag is.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, norm

from .pooling import pooled_values_with_spine_weights, weighted_mean_std


def population_variance(data: np.ndarray) -> float:
    """Spine-weighted variance of one feature's raw (positive) values."""
    values, weights = pooled_values_with_spine_weights(data)
    _, std = weighted_mean_std(values, weights)
    return float(std ** 2)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cdf = np.cumsum(w) - 0.5 * w
    cdf /= np.sum(w)
    return np.interp(quantiles, cdf, v)


def weighted_aic(logpdf_vals: np.ndarray, weights: np.ndarray, k: int = 2) -> float:
    log_likelihood = np.sum(weights * logpdf_vals)
    return 2 * k - 2 * log_likelihood


def pooled_deltas_with_spine_weights(data: np.ndarray, dt: int) -> tuple[np.ndarray, np.ndarray]:
    """Increments ``x(t+dt) - x(t)``, pooled across spines with 1/n_valid weighting."""
    n_times, n_spines = data.shape
    dx_vals, weights = [], []
    for spine in range(n_spines):
        col = data[:, spine]
        dxs = [col[i + dt] - col[i] for i in range(n_times - dt) if col[i] > 0 and col[i + dt] > 0]
        n = len(dxs)
        if n > 0:
            dx_vals.extend(dxs)
            weights.extend([1.0 / n] * n)
    return np.asarray(dx_vals, dtype=float), np.asarray(weights, dtype=float)


def pooled_standardized_deltas_across_lags(
    data: np.ndarray, n_timepoints_total: int
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Increments pooled across every lag 1..n_timepoints_total-1, each lag
    standardized by its own spine-weighted observed mean/std before pooling.

    Returns ``(z_all, weights_all, per_lag_stats)`` where each entry of
    ``per_lag_stats`` is ``(dt, n_pairs, mean, std)``.
    """
    z_all, w_all, lag_stats = [], [], []
    for dt in range(1, n_timepoints_total):
        dx_dt, w_dt = pooled_deltas_with_spine_weights(data, dt)
        mu_dt, sigma_dt = weighted_mean_std(dx_dt, w_dt)
        z_all.extend((dx_dt - mu_dt) / sigma_dt)
        w_all.extend(w_dt)
        lag_stats.append((dt, len(dx_dt), mu_dt, sigma_dt))
    return np.asarray(z_all), np.asarray(w_all), lag_stats


def fit_weighted_shifted_lognormal(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """Weighted MLE fit of a 3-parameter (shape, loc, scale) log-normal.

    ``loc`` is free (not fixed at 0) so the family can be fit to signed data
    (e.g. increments), with the data determining the shift. The likelihood is
    known to be multimodal for a free-loc log-normal, so this restarts from a
    grid of initial shifts/shapes and keeps the best optimum found.
    """
    data_min = values.min()
    margin = 0.01 * (values.max() - values.min())
    loc_upper = data_min - margin
    data_std = np.std(values)

    def neg_loglik(params):
        shape, loc, scale = params
        if shape <= 0 or scale <= 0 or loc > loc_upper:
            return np.inf
        return -np.sum(weights * lognorm.logpdf(values, shape, loc=loc, scale=scale))

    best = None
    for offset in [0.5, 1, 2, 5, 10, 30, 100, 300]:
        for shape0 in [0.3, 1.0, 2.0]:
            loc0 = loc_upper - offset * data_std
            scale0 = offset * data_std if offset > 1 else data_std
            res = minimize(
                neg_loglik, [shape0, loc0, scale0], method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000},
            )
            if best is None or res.fun < best.fun:
                best = res
    return tuple(best.x)


def raw_value_fit(data: np.ndarray) -> dict:
    """Log-normal vs. Gaussian fit to a feature's raw values (Fig. S1, top row)."""
    values, weights = pooled_values_with_spine_weights(data)
    mu_log, sigma_log = weighted_mean_std(np.log(values), weights)
    lognorm_shape, lognorm_scale = sigma_log, np.exp(mu_log)
    mu_raw, sigma_raw = weighted_mean_std(values, weights)

    logL_lognorm = lognorm.logpdf(values, lognorm_shape, loc=0, scale=lognorm_scale)
    logL_gauss = norm.logpdf(values, mu_raw, sigma_raw)
    aic_lognorm = weighted_aic(logL_lognorm, weights)
    aic_gauss = weighted_aic(logL_gauss, weights)

    return {
        "values": values,
        "weights": weights,
        "lognorm_shape": lognorm_shape,
        "lognorm_scale": lognorm_scale,
        "gauss_mean": mu_raw,
        "gauss_std": sigma_raw,
        "delta_aic": aic_gauss - aic_lognorm,  # positive favors log-normal
    }


def increment_fit(data: np.ndarray, n_timepoints_total: int) -> dict:
    """Gaussian vs. shifted-log-normal fit to a feature's standardized increments
    pooled across all lags (Fig. S1, bottom row)."""
    z, w, lag_stats = pooled_standardized_deltas_across_lags(data, n_timepoints_total)
    shape, loc, scale = fit_weighted_shifted_lognormal(z, w)

    logL_gauss = norm.logpdf(z, 0, 1)
    logL_lognorm = lognorm.logpdf(z, shape, loc=loc, scale=scale)
    aic_gauss = weighted_aic(logL_gauss, w, k=0)
    aic_lognorm = weighted_aic(logL_lognorm, w, k=3)

    return {
        "z": z,
        "weights": w,
        "lag_stats": lag_stats,
        "lognorm_shape": shape,
        "lognorm_loc": loc,
        "lognorm_scale": scale,
        "delta_aic": aic_lognorm - aic_gauss,  # positive favors Gaussian
    }
