"""Bayesian model comparison for the auto-/cross-correlation decay functions.

Implements the correlation model of Eq. 4 in the manuscript,

    C_ij(dt) = a_ij * exp(-|dt| / tau) + q_ij,

fit by Bayesian nested sampling (dynesty) with Gaussian errors from the
bootstrap correlation estimates, plus the alternative models it is compared
against (two time constants, with and without the persistent offset q_ij;
and a constant-only null with no temporal structure at all) -- Fig. 3,
Fig. 5, Figs. S2-S4.

The quenched-offset plateaus (q_ij, called ``c_*`` below) are treated as
fixed calibration constants rather than free parameters: they are estimates
of a static, spine-specific baseline covariance and can be read directly off
the long-lag data, so nested sampling is reserved for what actually needs a
posterior -- the timescale tau and the volatile amplitudes a_ij (Methods).

Every dynesty run is cached to disk (pickled ``results``) because a single
17-dimensional fit with ``nlive=1000`` takes tens of minutes; rerunning four
of them on every figure regeneration was the single most expensive part of
the original notebooks, and was never cached there.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..correlations import auto_correlation, cross_correlation

# ============================================================
# Fixed quenched-offset plateaus (Methods: q_ij), held fixed while
# tau/a_ij/n_i are inferred. Estimated once from the long-lag correlation
# data.
# ============================================================

FIXED_OFFSETS = {
    "hs": 0.68, "nl": 0.70, "nw": 0.22,
    "nw_hs": 0.25, "nl_hs": -0.02, "nw_nl": -0.04,
}
ZERO_OFFSETS = {k: 0.0 for k in FIXED_OFFSETS}

ONE_EXP_LABELS = ["tau", "a_nw_hs", "a_nl_hs", "a_hs", "a_nl", "a_nw", "a_nw_nl", "n_hs", "n_nl", "n_nw"]
TWO_EXP_LABELS = [
    "tau_1", "tau_2",
    "a_hs_1", "a_hs_2", "n_hs",
    "a_nl_1", "a_nl_2", "n_nl",
    "a_nw_1", "a_nw_2", "n_nw",
    "a_nw_hs_1", "a_nw_hs_2",
    "a_nl_hs_1", "a_nl_hs_2",
    "a_nw_nl_1", "a_nw_nl_2",
]
CONSTANT_LABELS = ["c_hs", "c_nl", "c_nw", "c_nw_hs", "c_nl_hs", "c_nw_nl", "n_hs", "n_nl", "n_nw"]


@dataclass
class RelaxationData:
    """Auto- and cross-correlation arrays, combined across short- and
    long-term measurements, in the layout the likelihoods below expect."""

    x_pos: np.ndarray  # non-negative lags (days), for auto-correlations
    x: np.ndarray  # signed lags (days), for cross-correlations
    HS_ac: np.ndarray
    HS_ac_error: np.ndarray
    NL_ac: np.ndarray
    NL_ac_error: np.ndarray
    NW_ac: np.ndarray
    NW_ac_error: np.ndarray
    y_nw_hs: np.ndarray
    yerr_nw_hs: np.ndarray
    y_nl_hs: np.ndarray
    yerr_nl_hs: np.ndarray
    y_nw_nl: np.ndarray
    yerr_nw_nl: np.ndarray
    n_short_auto: int  # number of short-term auto-correlation lags (for plotting)


def _combine_auto(short, long_):
    _, short_corr, short_err = short
    _, long_corr, long_err = long_
    return (
        np.concatenate((short_corr, long_corr[1:])),
        np.concatenate((short_err, long_err[1:])),
    )


def _split_pos_neg(short_values, long_values):
    mid_short = (len(short_values) - 1) // 2
    mid_long = (len(long_values) - 1) // 2
    neg = np.concatenate((long_values[:mid_long], short_values[: mid_short + 1]))
    pos = np.concatenate((short_values[mid_short:], long_values[mid_long + 1 :]))
    return neg, pos


def _combine_cross(short, long_):
    _, short_corr, short_err = short
    _, long_corr, long_err = long_
    corr_neg, corr_pos = _split_pos_neg(short_corr, long_corr)
    err_neg, err_pos = _split_pos_neg(short_err, long_err)
    return (
        np.concatenate((corr_neg, corr_pos[1:])),
        np.concatenate((err_neg, err_pos[1:])),
    )


def build_relaxation_data(
    HS_long, NL_long, NW_long, HS_short, NL_short, NW_short,
    delta_long_days, delta_short_days,
    rng: np.random.Generator | None = None,
) -> RelaxationData:
    """Compute auto-/cross-correlations from the raw data and combine
    short- and long-term measurements into one lag axis per pair."""
    rng = rng or np.random.default_rng(0)

    HS_auto_short = auto_correlation(HS_short, delta_short_days, rng)
    HS_auto_long = auto_correlation(HS_long, delta_long_days, rng)
    NL_auto_short = auto_correlation(NL_short, delta_short_days, rng)
    NL_auto_long = auto_correlation(NL_long, delta_long_days, rng)
    NW_auto_short = auto_correlation(NW_short, delta_short_days, rng)
    NW_auto_long = auto_correlation(NW_long, delta_long_days, rng)

    HS_ac, HS_ac_error = _combine_auto(HS_auto_short, HS_auto_long)
    NL_ac, NL_ac_error = _combine_auto(NL_auto_short, NL_auto_long)
    NW_ac, NW_ac_error = _combine_auto(NW_auto_short, NW_auto_long)
    # Nested sampling needs a finite lag-0 error; the bootstrap estimate at
    # lag 0 (a perfect self-pair) is 0.
    HS_ac_error[0] = NL_ac_error[0] = NW_ac_error[0] = 0.001

    x_pos = np.concatenate((delta_short_days, delta_long_days[1:]))

    nl_hs_short = cross_correlation(NL_short, HS_short, delta_short_days, rng)
    nl_hs_long = cross_correlation(NL_long, HS_long, delta_long_days, rng)
    nw_hs_short = cross_correlation(NW_short, HS_short, delta_short_days, rng)
    nw_hs_long = cross_correlation(NW_long, HS_long, delta_long_days, rng)
    nw_nl_short = cross_correlation(NW_short, NL_short, delta_short_days, rng)
    nw_nl_long = cross_correlation(NW_long, NL_long, delta_long_days, rng)

    y_nl_hs, yerr_nl_hs = _combine_cross(nl_hs_short, nl_hs_long)
    y_nw_hs, yerr_nw_hs = _combine_cross(nw_hs_short, nw_hs_long)
    y_nw_nl, yerr_nw_nl = _combine_cross(nw_nl_short, nw_nl_long)

    dt_neg, dt_pos = _split_pos_neg(nw_hs_short[0], nw_hs_long[0])
    x = np.concatenate((dt_neg, dt_pos[1:]))

    return RelaxationData(
        x_pos=x_pos, x=x,
        HS_ac=HS_ac, HS_ac_error=HS_ac_error,
        NL_ac=NL_ac, NL_ac_error=NL_ac_error,
        NW_ac=NW_ac, NW_ac_error=NW_ac_error,
        y_nw_hs=y_nw_hs, yerr_nw_hs=yerr_nw_hs,
        y_nl_hs=y_nl_hs, yerr_nl_hs=yerr_nl_hs,
        y_nw_nl=y_nw_nl, yerr_nw_nl=yerr_nw_nl,
        n_short_auto=len(delta_short_days),
    )


# ============================================================
# Model functions
# ============================================================


def one_exp_model(x_values, a, tau, c, n):
    """Auto-correlation: a*exp(-|dt|/tau) + c, with a noise bump n at lag 0."""
    y = a * np.exp(-np.abs(x_values) / tau) + c
    y = np.array(y, copy=True)
    y[0] = a * np.exp(-x_values[0] / tau) + c + n
    return y


def one_exp_model_cross(x_values, a, tau, c):
    return a * np.exp(-np.abs(x_values) / tau) + c


def two_exp_model(x_values, a, b, tau_1, tau_2, c, n):
    y = a * np.exp(-x_values / tau_1) + b * np.exp(-x_values / tau_2) + c
    y = np.array(y, copy=True)
    y[0] = a * np.exp(-x_values[0] / tau_1) + b * np.exp(-x_values[0] / tau_2) + c + n
    return y


def two_exp_model_cross(x_values, a, b, tau_1, tau_2, c):
    return a * np.exp(-x_values / tau_1) + b * np.exp(-x_values / tau_2) + c


def constant_model(x_values, c, n=0.0):
    y = np.full_like(np.asarray(x_values, dtype=float), c)
    y[0] += n
    return y


def constant_model_cross(x_values, c):
    return np.full_like(np.asarray(x_values, dtype=float), c)


# ============================================================
# Likelihoods and priors
# ============================================================


def _gaussian_loglike(model, data, error):
    return -0.5 * np.sum(((model - data) / error) ** 2)


def make_one_exp_loglike(data: RelaxationData, offsets=FIXED_OFFSETS):
    def loglike(p):
        tau, a_nw_hs, a_nl_hs, a_hs, a_nl, a_nw, a_nw_nl, n_hs, n_nl, n_nw = p
        if tau <= 0:
            return -np.inf
        total = 0.0
        total += _gaussian_loglike(one_exp_model_cross(data.x, a_nw_hs, tau, offsets["nw_hs"]), data.y_nw_hs, data.yerr_nw_hs)
        total += _gaussian_loglike(one_exp_model_cross(data.x, a_nl_hs, tau, offsets["nl_hs"]), data.y_nl_hs, data.yerr_nl_hs)
        total += _gaussian_loglike(one_exp_model_cross(data.x, a_nw_nl, tau, offsets["nw_nl"]), data.y_nw_nl, data.yerr_nw_nl)
        total += _gaussian_loglike(one_exp_model(data.x_pos, a_hs, tau, offsets["hs"], n_hs), data.HS_ac, data.HS_ac_error)
        total += _gaussian_loglike(one_exp_model(data.x_pos, a_nl, tau, offsets["nl"], n_nl), data.NL_ac, data.NL_ac_error)
        total += _gaussian_loglike(one_exp_model(data.x_pos, a_nw, tau, offsets["nw"], n_nw), data.NW_ac, data.NW_ac_error)
        return total

    return loglike


def prior_one_exp(u):
    x = np.array(u, copy=True)
    x[0] = 200.0 * x[0]  # tau ~ Uniform(0, 200) days
    x[1:7] = -2.0 * x[1:7] + 1.0  # amplitudes ~ Uniform(-1, 1)
    x[7:10] = 0.5 * x[7:10]  # noise ~ Uniform(0, 0.5)
    return x


def make_two_exp_loglike(data: RelaxationData, offsets):
    def loglike(p):
        tau1, tau2 = p[0], p[1]
        if tau1 <= 0 or tau2 <= 0:
            return -np.inf
        a_hs, a_hs2, n_hs = p[2], p[3], p[4]
        a_nl, a_nl2, n_nl = p[5], p[6], p[7]
        a_nw, a_nw2, n_nw = p[8], p[9], p[10]
        a_nw_hs, a_nw_hs2 = p[11], p[12]
        a_nl_hs, a_nl_hs2 = p[13], p[14]
        a_nw_nl, a_nw_nl2 = p[15], p[16]

        total = 0.0
        total += _gaussian_loglike(two_exp_model_cross(data.x, a_nw_hs, a_nw_hs2, tau1, tau2, offsets["nw_hs"]), data.y_nw_hs, data.yerr_nw_hs)
        total += _gaussian_loglike(two_exp_model_cross(data.x, a_nl_hs, a_nl_hs2, tau1, tau2, offsets["nl_hs"]), data.y_nl_hs, data.yerr_nl_hs)
        total += _gaussian_loglike(two_exp_model_cross(data.x, a_nw_nl, a_nw_nl2, tau1, tau2, offsets["nw_nl"]), data.y_nw_nl, data.yerr_nw_nl)
        total += _gaussian_loglike(two_exp_model(data.x_pos, a_hs, a_hs2, tau1, tau2, offsets["hs"], n_hs), data.HS_ac, data.HS_ac_error)
        total += _gaussian_loglike(two_exp_model(data.x_pos, a_nl, a_nl2, tau1, tau2, offsets["nl"], n_nl), data.NL_ac, data.NL_ac_error)
        total += _gaussian_loglike(two_exp_model(data.x_pos, a_nw, a_nw2, tau1, tau2, offsets["nw"], n_nw), data.NW_ac, data.NW_ac_error)
        return total

    return loglike


def _two_exp_amplitude_noise_prior(x):
    # NOTE: indices 4 and 10 are n_hs/n_nw (noise terms) but fall inside this
    # Uniform(-1, 1) loop rather than the Uniform(0, 0.5) noise range below,
    # and indices 8/9 (a_nw, a_nw_2, true amplitudes) get the noise range
    # instead. This reproduces the exact prior used for the published
    # two-timescale fits (Figs. 3, S3, S4) rather than a corrected version
    # that would no longer match them -- both alternative models are already
    # reported as poorly identified/degenerate regardless.
    for idx in [2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]:
        x[idx] = -2.0 * x[idx] + 1.0
    x[7] = 0.5 * x[7]
    x[8] = 0.5 * x[8]
    x[9] = 0.5 * x[9]
    return x


def prior_two_exp_with_offset(u):
    """Both taus ~ Uniform(0, 200): the fixed offset already carries the
    asymptote, so neither exponential has reason to reach further than the
    one-exp model's tau range."""
    x = np.array(u, copy=True)
    x[0] = 200.0 * x[0]
    x[1] = 200.0 * x[1]
    return _two_exp_amplitude_noise_prior(x)


def prior_two_exp_no_offset(u):
    """tau_1 ~ Uniform(0, 10) covers the fast relaxation; tau_2 ~
    log-uniform(10, 1e4) is pushed into a separate, much slower regime so it
    can carry the apparent plateau back to zero far outside the observed
    window, instead of competing with tau_1 for the same range."""
    x = np.array(u, copy=True)
    x[0] = 10.0 * x[0]
    tau2_min, tau2_max = 1e1, 1e4
    x[1] = tau2_min * (tau2_max / tau2_min) ** x[1]
    return _two_exp_amplitude_noise_prior(x)


def make_constant_loglike(data: RelaxationData):
    def loglike(p):
        c_hs, c_nl, c_nw, c_nw_hs, c_nl_hs, c_nw_nl, n_hs, n_nl, n_nw = p
        total = 0.0
        total += _gaussian_loglike(constant_model(data.x_pos, c_hs, n_hs), data.HS_ac, data.HS_ac_error)
        total += _gaussian_loglike(constant_model(data.x_pos, c_nl, n_nl), data.NL_ac, data.NL_ac_error)
        total += _gaussian_loglike(constant_model(data.x_pos, c_nw, n_nw), data.NW_ac, data.NW_ac_error)
        total += _gaussian_loglike(constant_model_cross(data.x, c_nw_hs), data.y_nw_hs, data.yerr_nw_hs)
        total += _gaussian_loglike(constant_model_cross(data.x, c_nl_hs), data.y_nl_hs, data.yerr_nl_hs)
        total += _gaussian_loglike(constant_model_cross(data.x, c_nw_nl), data.y_nw_nl, data.yerr_nw_nl)
        return total

    return loglike


def prior_constant(u):
    x = np.array(u, copy=True)
    x[:6] = -2.0 * x[:6] + 1.0
    x[6:] = 0.5 * x[6:]
    return x


MODEL_SPECS = {
    "constant": dict(ndim=9, labels=CONSTANT_LABELS, prior=prior_constant, make_loglike=lambda data: make_constant_loglike(data)),
    "one_exp_offset": dict(ndim=10, labels=ONE_EXP_LABELS, prior=prior_one_exp, make_loglike=lambda data: make_one_exp_loglike(data, FIXED_OFFSETS)),
    "two_exp_no_offset": dict(ndim=17, labels=TWO_EXP_LABELS, prior=prior_two_exp_no_offset, make_loglike=lambda data: make_two_exp_loglike(data, ZERO_OFFSETS)),
    "two_exp_with_offset": dict(ndim=17, labels=TWO_EXP_LABELS, prior=prior_two_exp_with_offset, make_loglike=lambda data: make_two_exp_loglike(data, FIXED_OFFSETS)),
}


# ============================================================
# Nested-sampling runner with disk caching
# ============================================================


def fit_model(name: str, data: RelaxationData, cache_dir: Path, force: bool = False, seed: int = 123):
    """Run (or load a cached) dynesty fit for one of ``MODEL_SPECS``."""
    import dynesty

    cache_path = Path(cache_dir) / f"{name}.pkl"
    if cache_path.exists() and not force:
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    spec = MODEL_SPECS[name]
    sampler = dynesty.NestedSampler(
        spec["make_loglike"](data),
        spec["prior"],
        ndim=spec["ndim"],
        nlive=1000,
        rstate=np.random.default_rng(seed),
        sample="rslice",
    )
    sampler.run_nested(dlogz=0.01)
    results = sampler.results

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(results, fh)
    return results


def run_all_models(data: RelaxationData, cache_dir: Path, force: bool = False) -> dict:
    return {name: fit_model(name, data, cache_dir, force=force) for name in MODEL_SPECS}


def dynesty_weights(results) -> np.ndarray:
    w = np.exp(results.logwt - results.logz[-1])
    return w / np.sum(w)


def posterior_mean(results) -> np.ndarray:
    return np.average(results.samples, weights=dynesty_weights(results), axis=0)


def posterior_std(results) -> np.ndarray:
    weights = dynesty_weights(results)
    mean = np.average(results.samples, weights=weights, axis=0)
    var = np.average((results.samples - mean) ** 2, weights=weights, axis=0)
    return np.sqrt(var)


def posterior_quantile(results, quantiles) -> np.ndarray:
    weights = dynesty_weights(results)
    out = np.empty((results.samples.shape[1], len(np.atleast_1d(quantiles))))
    for i in range(results.samples.shape[1]):
        values = results.samples[:, i]
        order = np.argsort(values)
        cdf = np.cumsum(weights[order]) - 0.5 * weights[order]
        cdf /= cdf[-1] + 0.5 * weights[order][-1]
        out[i] = np.interp(quantiles, cdf, values[order])
    return out


# ============================================================
# Illustrative kernel model (Fig. 4 schematic; Eq. 2-3)
#
# The general two-timescale-per-feature kernel that Eq. 4's single-timescale
# fit above is a special case of (alpha_i=1, tau_i1=tau_i2=tau). Fig. 4 uses
# this with hand-picked, deliberately spread-apart illustrative timescales
# (own comments in the source notebook: "schematic ... not fitted to data")
# to make the forward/inverse correlation-matrix logic visually clear, not
# to report a fit.
# ============================================================


def kernel_f(t, alpha, tau1, tau2):
    """Eq. 2: causal two-timescale plasticity kernel for one feature."""
    t = np.asarray(t, dtype=float)
    kernel = np.zeros_like(t)
    causal = t >= 0
    kernel[causal] = (alpha / tau1) * np.exp(-t[causal] / tau1) + ((1.0 - alpha) / tau2) * np.exp(-t[causal] / tau2)
    return kernel


def auto_weights(alpha, tau1, tau2):
    """Eq. 3 with i=j: relative weight of a feature's two timescales in its own auto-correlation."""
    w1 = alpha * (alpha / (2 * tau1) + (1 - alpha) / (tau1 + tau2))
    w2 = (1 - alpha) * (alpha / (tau1 + tau2) + (1 - alpha) / (2 * tau2))
    return w1, w2


def auto_corr_curve(dt, amplitude, alpha, tau1, tau2):
    w1, w2 = auto_weights(alpha, tau1, tau2)
    total = w1 + w2
    a1, a2 = amplitude * w1 / total, amplitude * w2 / total
    dt = np.asarray(dt, dtype=float)
    return a1 * np.exp(-np.abs(dt) / tau1) + a2 * np.exp(-np.abs(dt) / tau2)


def cross_weights(alpha_i, tau_i1, tau_i2, alpha_j, tau_j1, tau_j2):
    """Eq. 3, general (asymmetric) case: weights for dt>0 decaying with
    feature i's timescales, and dt<0 decaying with feature j's timescales."""
    w1_pos = alpha_i * (alpha_j / (tau_i1 + tau_j1) + (1 - alpha_j) / (tau_i1 + tau_j2))
    w2_pos = (1 - alpha_i) * (alpha_j / (tau_i2 + tau_j1) + (1 - alpha_j) / (tau_i2 + tau_j2))
    w1_neg = alpha_j * (alpha_i / (tau_i1 + tau_j1) + (1 - alpha_i) / (tau_i2 + tau_j1))
    w2_neg = (1 - alpha_j) * (alpha_i / (tau_i1 + tau_j2) + (1 - alpha_i) / (tau_i2 + tau_j2))
    return w1_pos, w2_pos, w1_neg, w2_neg


def cross_corr_curve(dt, peak_amplitude, alpha_i, tau_i1, tau_i2, alpha_j, tau_j1, tau_j2):
    """Cross-correlation curve for ordered feature pair (i, j); asymmetric in
    dt whenever the two features' timescales differ (Eq. 3)."""
    w1_pos, w2_pos, w1_neg, w2_neg = cross_weights(alpha_i, tau_i1, tau_i2, alpha_j, tau_j1, tau_j2)
    total = w1_pos + w2_pos
    scale = peak_amplitude / total

    dt = np.asarray(dt, dtype=float)
    out = np.empty_like(dt)
    pos = dt >= 0
    out[pos] = scale * (w1_pos * np.exp(-dt[pos] / tau_i1) + w2_pos * np.exp(-dt[pos] / tau_i2))
    out[~pos] = scale * (w1_neg * np.exp(dt[~pos] / tau_j1) + w2_neg * np.exp(dt[~pos] / tau_j2))
    return out
