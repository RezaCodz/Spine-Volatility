"""Event-class-gated counterfactual current reconstructions (Fig. 6c, Fig. S5).

Given the same three event classes as :mod:`spine_volatility.models.event_mixture`
(coordinated / head-only / neck-only), simulate forward morphology
trajectories from a starting point using a discretized (AR(1)) OU process
whose innovation covariance is built from whichever subset of event classes
is switched on -- "normal" (all three), "only head-only", "neck-only plus
coordinated", or "only coordinated" -- to see which event classes are
responsible for the breadth of predicted synaptic-current fluctuations.

This exact simulation is reused for both Fig. 6c (single representative
spine) and Fig. S5 (six locations across morphology space); the original
notebook had two near-identical copies of it, one per figure.
"""

from dataclasses import dataclass, field

import numpy as np

EVENT_RATE_MODELS = {
    "normal": np.array([0.2, 0.4, 0.4]),
    "only_head_events": np.array([0.0, 0.4, 0.0]),
    "neck_and_coordinated_events": np.array([0.2, 0.0, 0.4]),
    "only_coordinated_events": np.array([0.2, 0.0, 0.0]),
}
PLOT_ORDER = ["normal", "only_head_events", "neck_and_coordinated_events", "only_coordinated_events"]

# Individual trajectory traces are only drawn for these two classes (matching
# the original figures); "neck_and_coordinated_events" and
# "only_coordinated_events" appear only in the adjacent density panel, to
# keep the trace plot legible.
TRAJECTORY_MODELS = {"normal", "only_head_events"}

STYLE = {
    "normal": dict(color="blue", linestyle="-", label="Normal", trace_alpha=0.08),
    "only_head_events": dict(color="#D55E00", linestyle="-", label="Only head", trace_alpha=0.18),
    "neck_and_coordinated_events": dict(color="#0072B2", linestyle=(0, (8, 2, 2, 2)), label="Neck + coordinated", trace_alpha=0.18),
    "only_coordinated_events": dict(color="#009E73", linestyle=(0, (5, 2, 1.5, 2)), label="Only coordinated", trace_alpha=0.18),
}


@dataclass
class EventJumpSizes:
    """Per-feature jump sizes for each event class, calibrated once from the
    "normal" event rates so that the resulting covariance matches the
    fitted auto-covariance amplitudes A_hs/A_nl/A_nw."""

    coordinated: np.ndarray
    head_only: np.ndarray
    neck_only: np.ndarray


def calibrate_jump_sizes(A_hs: float, A_nl: float, A_nw: float, normal_rates=EVENT_RATE_MODELS["normal"]) -> EventJumpSizes:
    e0, e1_0, e2_0 = normal_rates
    J_hs = np.sqrt(A_hs / (e0 + e1_0))
    J_nl = -np.sqrt(A_nl / (e0 + e2_0))
    J_nw = np.sqrt(A_nw / (e0 + e2_0))
    return EventJumpSizes(
        coordinated=np.array([J_hs, J_nl, J_nw]),
        head_only=np.array([J_hs, 0.0, 0.0]),
        neck_only=np.array([0.0, J_nl, J_nw]),
    )


def covariance_from_event_rates(jumps: EventJumpSizes, event_rates: np.ndarray) -> np.ndarray:
    e_coord, e_head, e_neck = event_rates
    cov = (
        e_coord * np.outer(jumps.coordinated, jumps.coordinated)
        + e_head * np.outer(jumps.head_only, jumps.head_only)
        + e_neck * np.outer(jumps.neck_only, jumps.neck_only)
    )
    return 0.5 * (cov + cov.T)


def _covariance_sqrt(cov: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues))


def simulate_ar1_from_start(
    start_point: np.ndarray,
    cov: np.ndarray,
    tau: float,
    t: np.ndarray,
    feature_min: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    max_attempts: int = 120_000,
) -> np.ndarray:
    """Discretized (AR(1)) OU trajectories starting exactly at ``start_point``,
    rejecting any trajectory that ever dips below ``feature_min`` in any
    feature. Returns an array of shape ``(n_accepted, len(t), 3)``.
    """
    dt = float(t[1] - t[0])
    phi = np.exp(-dt / tau)
    innovation_sqrt = _covariance_sqrt((1.0 - phi ** 2) * cov)

    accepted = []
    attempts = 0
    while len(accepted) < n_samples and attempts < max_attempts:
        attempts += 1
        X = np.empty((len(t), 3))
        X[0] = start_point
        for i in range(1, len(t)):
            noise = innovation_sqrt @ rng.standard_normal(3)
            X[i] = start_point + phi * (X[i - 1] - start_point) + noise
        if np.all(X >= feature_min):
            accepted.append(X)

    return np.asarray(accepted)
