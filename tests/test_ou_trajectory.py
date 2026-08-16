import numpy as np
import pytest

from spine_volatility.simulation.ou_trajectory import (
    RejectionSamplingFailed,
    covariance_from_amplitudes,
    generate_trajectory,
    sample_posterior,
)


def test_covariance_from_amplitudes_is_symmetric_positive_semidefinite():
    variances = np.array([0.05, 0.2, 0.003])
    diag_amplitudes = np.array([0.23, 0.20, 0.38])
    cross_amplitudes = {(0, 1): -0.09, (0, 2): 0.07, (1, 2): -0.28}
    A = covariance_from_amplitudes(variances, diag_amplitudes, cross_amplitudes)
    np.testing.assert_allclose(A, A.T)
    eigenvalues = np.linalg.eigvalsh(A)
    assert np.all(eigenvalues >= -1e-12)


def test_generate_trajectory_matches_target_covariance():
    A = covariance_from_amplitudes(
        np.array([1.0, 1.0, 1.0]), np.array([0.3, 0.3, 0.3]), {(0, 1): 0.5, (0, 2): -0.2, (1, 2): 0.1}
    )
    rng = np.random.default_rng(0)
    X = generate_trajectory(A, tau=3.6, N=4096, dt=0.01, rng=rng)
    # generate_trajectory whitens/recolors each realization to exactly match A.
    np.testing.assert_allclose(np.cov(X.T), A, atol=1e-8)


def test_sample_posterior_raises_when_unreachable():
    A = covariance_from_amplitudes(np.array([0.05, 0.2, 0.003]), np.array([0.2, 0.2, 0.2]), {(0, 1): 0, (0, 2): 0, (1, 2): 0})
    rng = np.random.default_rng(0)
    N, dt = 200, 0.1
    t = np.linspace(0, (N - 1) * dt, N)
    target_points = np.array([[100.0, 100.0, 100.0]])  # unreachable given tiny variances
    with pytest.raises(RejectionSamplingFailed):
        sample_posterior(
            A, tau=3.6, N=N, dt=dt, target_points=target_points, target_days=np.array([0.0]), t=t,
            feature_min=np.array([-1e9, -1e9, -1e9]), error_bounds=np.array([1e-6, 1e-6, 1e-6]),
            n_samples=5, rng=rng, max_attempts=200,
        )
