"""Loading and unit-scaling of the raw spine-morphology measurements.

Long-term data: control-mouse measurements from Steffens et al. 2021
(motor cortex, layer 1 apical dendrites), 8 recorded time points per spine,
of which the first 5 (0, 3.5, 7, 10.5, 14 days) are used throughout this
package -- matching the manuscript. Short-term data: control-mouse
measurements from Wegner et al. 2022 (visual cortex), 4 time points
(0, 30, 60, 120 minutes).

Consolidates what was previously copy-pasted, byte-identical, at the top of
``figure_1.ipynb``, ``figure_2.ipynb``, ``figure_4.ipynb`` and others.
"""

from dataclasses import dataclass

import numpy as np

from .paths import LONG_TERM_DIR, SHORT_TERM_DIR

# Recorded time points.
DELTA_SHORT_MINUTES = np.array([0.0, 30.0, 60.0, 120.0])
DELTA_SHORT_DAYS = DELTA_SHORT_MINUTES / (60 * 24)
DELTA_LONG_DAYS = np.array([0.0, 3.5, 7.0, 10.5, 14.0])

# Measurement-noise widths (Methods: "derived from the autocorrelation
# function at zero lag", i.e. sqrt(noise fraction n_i * population variance
# var_i) from an early iteration of the correlation fit). Held fixed as
# calibration constants rather than re-derived, consistent with how they
# were used throughout the original analysis.
MEASUREMENT_NOISE_HS = np.sqrt(0.0940 * 0.052)  # um^2
MEASUREMENT_NOISE_NL = np.sqrt(0.1044 * 0.194)  # um
MEASUREMENT_NOISE_NW = np.sqrt(0.3986 * 0.0029)  # um
MEASUREMENT_NOISE = np.array([MEASUREMENT_NOISE_HS, MEASUREMENT_NOISE_NL, MEASUREMENT_NOISE_NW])


def _extract_columns(data: np.ndarray, columns: list[int]) -> np.ndarray:
    return np.array([data[col] for col in columns])


@dataclass(frozen=True)
class LongTermData:
    """Shape (8 time points, n_spines); -1 marks a missing measurement."""

    HS: np.ndarray  # um^2
    NL: np.ndarray  # um
    NW: np.ndarray  # um


@dataclass(frozen=True)
class ShortTermData:
    """Shape (4 time points, n_spines); -1 marks a missing measurement."""

    HS: np.ndarray  # um^2
    NL: np.ndarray  # um
    NW: np.ndarray  # um


def load_long_term() -> LongTermData:
    HS = np.genfromtxt(
        LONG_TERM_DIR / "WT_headarea.csv",
        delimiter=";", skip_header=1, unpack=True, filling_values=-1,
    ) / 1e6  # nm^2 -> um^2
    NL = np.genfromtxt(
        LONG_TERM_DIR / "WT_necklength.csv",
        delimiter=";", skip_header=1, unpack=True, filling_values=-1,
    ) / 1e3  # nm -> um
    NW = np.genfromtxt(
        LONG_TERM_DIR / "WT_neckwidth.csv",
        delimiter=";", skip_header=1, unpack=True, filling_values=-1,
    ) / 1e3  # nm -> um
    return LongTermData(HS=HS, NL=NL, NW=NW)


def load_short_term() -> ShortTermData:
    raw = np.genfromtxt(
        SHORT_TERM_DIR / "final_Ctr_short.csv",
        delimiter=";", skip_header=1, unpack=True, filling_values=-1,
    )
    HS = _extract_columns(raw, [1, 5, 9, 13]) / 1e6  # nm^2 -> um^2
    NL = _extract_columns(raw, [2, 6, 10, 14]) / 1e3  # nm -> um
    NW = _extract_columns(raw, [3, 7, 11, 15]) / 1e3  # nm -> um
    return ShortTermData(HS=HS, NL=NL, NW=NW)
