"""Three-event-class covariance-amplitude surface (Fig. 5).

Results: "We therefore considered an extended model with three event
classes -- coordinated, head-only, and neck-only -- whose kernels we
inferred from the autocorrelation functions ... When all three are allowed,
however, the manifold of achievable covariance amplitudes intersects the
experimental values, constraining coordinated events to e=0.12-0.27,
head-only events to e1=0.34-0.47, and neck-only events to e2=0.32-0.48 of
the total."

Physical covariance amplitudes (``A_hs``, ``A_nl``, ``A_nw`` and the
measured cross-covariance point) are computed from the fitted posterior via
:func:`physical_amplitudes`, rather than hand-copy-typed -- the original
notebook's constants were exactly the posterior medians from the one-
exponential-plus-offset fit, transcribed by hand.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicalAmplitudes:
    tau: float
    A_hs: float
    A_nl: float
    A_nw: float
    measured: np.ndarray  # [A_nl_hs, A_nl_nw, A_nw_hs]
    errors_2sigma: np.ndarray  # 2*sigma physical uncertainty on `measured`


def physical_amplitudes(
    tau: float,
    var_hs: float, var_nl: float, var_nw: float,
    a_hs: float, a_nl: float, a_nw: float,
    a_nl_hs: float, a_nl_nw: float, a_nw_hs: float,
    s_nl_hs: float, s_nl_nw: float, s_nw_hs: float,
) -> PhysicalAmplitudes:
    """Convert dimensionless fitted amplitudes (posterior means `a_*`, and
    standard errors `s_*` for the three cross terms) into physical
    auto-/cross-covariance amplitudes."""
    A_hs = a_hs * var_hs
    A_nl = a_nl * var_nl
    A_nw = a_nw * var_nw
    A_nl_hs = a_nl_hs * np.sqrt(var_nl * var_hs)
    A_nl_nw = a_nl_nw * np.sqrt(var_nl * var_nw)
    A_nw_hs = a_nw_hs * np.sqrt(var_nw * var_hs)

    sigma_nl_hs = s_nl_hs * np.sqrt(var_nl * var_hs)
    sigma_nl_nw = s_nl_nw * np.sqrt(var_nl * var_nw)
    sigma_nw_hs = s_nw_hs * np.sqrt(var_nw * var_hs)

    return PhysicalAmplitudes(
        tau=tau, A_hs=A_hs, A_nl=A_nl, A_nw=A_nw,
        measured=np.array([A_nl_hs, A_nl_nw, A_nw_hs]),
        errors_2sigma=2 * np.array([sigma_nl_hs, sigma_nl_nw, sigma_nw_hs]),
    )


def event_point(amp: PhysicalAmplitudes, e: float, e1: float, e2: float) -> np.ndarray:
    """Predicted [A_nl_hs, A_nl_nw, A_nw_hs] for a mixture of coordinated (e),
    head-only (e1), and neck-only (e2) events.

    Targeting convention for this static covariance-surface model: head size
    and neck width are both hit by (e+e1), neck length by (e+e2) -- i.e.
    "head-only" events here move head size and neck width together. This
    reproduces the plausible event-fraction ranges actually reported in the
    manuscript (e=0.12-0.27, e1=0.34-0.47, e2=0.32-0.48) when combined with
    the fitted posterior; a targeting split with neck width instead grouped
    with neck length (as in :mod:`event_knockout`'s dynamic simulation for
    Fig. 6c/S5) does not reproduce them. The two figures use different,
    independently-defined targeting conventions in the original analysis,
    and each is kept faithful to its own here rather than forced to agree.
    """
    J_hs = np.sqrt(amp.A_hs / (e + e1))
    J_nl = -np.sqrt(amp.A_nl / (e + e2))
    J_nw = np.sqrt(amp.A_nw / (e + e1))
    return np.array([J_nl * J_hs * e, J_nl * J_nw * (e + e2), J_nw * J_hs * e])


def covariance_surface(amp: PhysicalAmplitudes, n: int = 90):
    """Predicted cross-covariance surface over the full 2-simplex of event
    fractions (e, e1, e2), e + e1 + e2 = 1. Returns (X, Y, Z) grids, NaN
    outside the simplex."""
    e_vals = np.linspace(1e-3, 1 - 1e-3, n)
    E, E1 = np.meshgrid(e_vals, e_vals)
    E2 = 1 - E - E1
    mask = E2 > 0

    J_hs = np.sqrt(amp.A_hs / (E + E1))
    J_nl = -np.sqrt(amp.A_nl / (E + E2))
    J_nw = np.sqrt(amp.A_nw / (E + E1))

    X = np.where(mask, J_nl * J_hs * E, np.nan)
    Y = np.where(mask, J_nl * J_nw * (E + E2), np.nan)
    Z = np.where(mask, J_nw * J_hs * E, np.nan)
    return X, Y, Z


def surface_crossing(amp: PhysicalAmplitudes, n: int = 130):
    """Which points of the covariance surface fall within the measured
    point's 2-sigma ellipsoid, and the resulting plausible ranges of
    (e, e1, e2). Returns a dict with the grids, a boolean mask, and the
    plausible min/max fraction for each event class."""
    e_vals = np.linspace(1e-3, 1 - 1e-3, n)
    E, E1 = np.meshgrid(e_vals, e_vals)
    E2 = 1 - E - E1

    X, Y, Z = covariance_surface(amp, n=n)
    distance_sq = (
        ((X - amp.measured[0]) / amp.errors_2sigma[0]) ** 2
        + ((Y - amp.measured[1]) / amp.errors_2sigma[1]) ** 2
        + ((Z - amp.measured[2]) / amp.errors_2sigma[2]) ** 2
    )
    finite = np.isfinite(distance_sq)
    inside = finite & (distance_sq <= 1.0)

    result = {"X": X, "Y": Y, "Z": Z, "E": E, "E1": E1, "E2": E2, "inside": inside}
    if inside.any():
        result["e_range"] = (float(E[inside].min()), float(E[inside].max()))
        result["e1_range"] = (float(E1[inside].min()), float(E1[inside].max()))
        result["e2_range"] = (float(E2[inside].min()), float(E2[inside].max()))
    return result
