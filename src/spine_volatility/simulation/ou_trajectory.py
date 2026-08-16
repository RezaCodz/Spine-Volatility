"""Spectral-domain Ornstein-Uhlenbeck trajectory generator (Fig. 1d, Fig. 6a).

Methods: "To generate surrogate trajectories, we used the fact that
correlation functions decay exponentially with time constant tau=3.6 days,
and constructed the corresponding spectral density S(w) = 2*tau /
(1+(2*pi*tau*w)^2). Multivariate Gaussian processes were then sampled in the
frequency domain with covariance matrices matched to the empirically
inferred auto- and cross-covariance amplitudes. After inverse Fourier
transformation and rescaling, this yielded synthetic trajectories whose
covariance structure reproduced the measured correlations."

This same algorithm was independently reimplemented, with only cosmetic
variation, six times across the original notebooks. The covariance matrix
was also hand-copy-typed as a literal in two of those places, with slightly
different values each time (e.g. neck-length variance given as both 0.194
and 0.2) -- here it is always built from the actual fitted posterior via
:func:`covariance_from_amplitudes`, so there is exactly one source of truth.
"""

import numpy as np

FEATURE_INDEX = {"hs": 0, "nl": 1, "nw": 2}


def covariance_from_amplitudes(
    variances: np.ndarray, diag_amplitudes: np.ndarray, cross_amplitudes: dict
) -> np.ndarray:
    """Physical 3x3 covariance matrix from fitted dimensionless amplitudes.

    ``variances`` are the raw (spine-weighted) population variances of
    [HS, NL, NW]; ``diag_amplitudes`` are the fitted a_hs/a_nl/a_nw; and
    ``cross_amplitudes`` maps ``(i, j)`` index pairs (0=hs, 1=nl, 2=nw) to
    the fitted cross-correlation amplitude a_ij (e.g. ``a_nl_hs`` for
    ``(0, 1)``). Off-diagonal covariance is the geometric mean of the two
    diagonal entries scaled by that correlation amplitude, guaranteeing a
    valid (positive semi-definite) correlation structure.
    """
    A = np.zeros((3, 3))
    for j in range(3):
        A[j, j] = diag_amplitudes[j] * variances[j]
    for (j, k), r in cross_amplitudes.items():
        A[j, k] = A[k, j] = r * np.sqrt(A[j, j] * A[k, k])
    return A


def spectral_cholesky(A: np.ndarray, tau: float, N: int, dt: float) -> np.ndarray:
    """Per-frequency Cholesky factor of the OU spectral density scaled by ``A``."""
    freqs = np.fft.fftfreq(N, d=dt)
    S = (2 * tau) / (1 + (2 * np.pi * tau * freqs) ** 2)
    return np.linalg.cholesky(A[None, :, :] * S[:, None, None] + 1e-10 * np.eye(3)[None, :, :])


def generate_trajectory(A: np.ndarray, tau: float, N: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """One centered, 3-feature OU trajectory of length ``N`` with covariance ``A``."""
    L_freq = spectral_cholesky(A, tau, N, dt)

    z_raw = rng.standard_normal((N, 2, 3))
    z = (z_raw[:, 0, :] + 1j * z_raw[:, 1, :]) / np.sqrt(2)

    Xf = np.einsum("kij,kj->ik", L_freq, z)
    Xf[:, 0] = 0
    for k in range(1, N // 2):
        Xf[:, -k] = np.conj(Xf[:, k])
    if N % 2 == 0:
        Xf[:, N // 2] = Xf[:, N // 2].real

    X = np.fft.ifft(Xf, axis=1).real * np.sqrt(N)

    # Rescale so the empirical covariance of this particular realization
    # exactly matches the target A, rather than only matching in expectation.
    L_emp = np.linalg.cholesky(np.cov(X))
    L_tgt = np.linalg.cholesky(A)
    X_white = np.linalg.solve(L_emp, X)
    return (L_tgt @ X_white).T


class RejectionSamplingFailed(RuntimeError):
    """Raised when :func:`sample_posterior` cannot find enough trajectories
    consistent with the measured points within ``max_attempts``."""


def sample_posterior(
    A: np.ndarray,
    tau: float,
    N: int,
    dt: float,
    target_points: np.ndarray,
    target_days: np.ndarray,
    t: np.ndarray,
    feature_min: np.ndarray,
    error_bounds: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    max_attempts: int = 200_000,
) -> list[np.ndarray]:
    """Trajectories consistent with sparse measurements (rejection sampling).

    A trajectory is accepted if it never dips below the smallest physically
    observed value of any feature, and its values at ``target_days`` fall
    within 1.96 x measurement error of the actual measured points. The
    original implementation looped until ``n_samples`` were accepted with no
    upper bound on attempts; this caps attempts and raises instead of
    hanging when the target is effectively unreachable.
    """
    target_mean = np.mean(target_points, axis=0)
    accepted = []
    attempts = 0
    while len(accepted) < n_samples and attempts < max_attempts:
        attempts += 1
        X = generate_trajectory(A, tau, N, dt, rng) + target_mean
        if np.any(X < feature_min):
            continue
        X_at_days = np.column_stack([np.interp(target_days, t, X[:, dim]) for dim in range(3)])
        if np.all(np.abs(X_at_days - target_points) <= 1.96 * error_bounds):
            accepted.append(X)

    if len(accepted) < n_samples:
        raise RejectionSamplingFailed(
            f"Accepted only {len(accepted)}/{n_samples} trajectories after {attempts} attempts."
        )
    return accepted
