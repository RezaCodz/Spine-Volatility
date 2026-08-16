"""Instantaneous synaptic-current proxy (Methods: "Synaptic strength estimation").

S proportional to h * w^gamma / l, with gamma=1 (spine-neck resistance is
non-ohmic and scales more weakly with neck radius than the naive R ~ L/r^2
cable model). The proportionality constant is fixed so the mean strength
across the long-term measurements corresponds to a typical current of 15 pA.
"""

from dataclasses import dataclass

import numpy as np

GAMMA = 1.0
TYPICAL_CURRENT_PA = 15.0


@dataclass
class CurrentCalibration:
    gamma: float
    scale: float  # pA per unit of h * w^gamma / l


def calibrate(HS: np.ndarray, NL: np.ndarray, NW: np.ndarray, gamma: float = GAMMA) -> CurrentCalibration:
    """Fit the scale constant from all valid raw long-term morphology measurements.

    ``HS``, ``NL``, ``NW`` have shape ``(n_timepoints, n_spines)``.
    """
    valid = (HS > 0) & (NL > 0) & (NW > 0)
    strength = HS[valid] * (NW[valid] ** gamma) / NL[valid]
    scale = TYPICAL_CURRENT_PA / np.mean(strength)
    return CurrentCalibration(gamma=gamma, scale=scale)


def current_from_point(calib: CurrentCalibration, point: np.ndarray) -> float:
    h, l, w = point
    return calib.scale * h * (w ** calib.gamma) / l


def current_ci(calib: CurrentCalibration, point: np.ndarray, error_bounds: np.ndarray) -> float:
    """Propagated 1-sigma uncertainty on the current at one measured point,
    from the measurement-noise widths of h, l, w."""
    h, l, w = point
    err_h, err_l, err_w = error_bounds
    rel_err = np.sqrt((err_h / h) ** 2 + (calib.gamma * err_w / w) ** 2 + (err_l / l) ** 2)
    return rel_err * current_from_point(calib, point)


def current_from_trajectories(calib: CurrentCalibration, trajectories: np.ndarray) -> np.ndarray:
    """Current along each trajectory. ``trajectories`` has shape
    ``(n_trajectories, n_times, 3)`` with feature order (h, l, w)."""
    h, l, w = trajectories[..., 0], trajectories[..., 1], trajectories[..., 2]
    return calib.scale * h * (w ** calib.gamma) / l
