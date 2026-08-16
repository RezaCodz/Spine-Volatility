import numpy as np

from spine_volatility.correlations import auto_correlation, correlation_from_pairs, cross_correlation


def test_correlation_from_pairs_perfect_correlation():
    pairs = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    assert np.isclose(correlation_from_pairs(pairs), 1.0)


def test_correlation_from_pairs_perfect_anticorrelation():
    pairs = np.array([[1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]])
    assert np.isclose(correlation_from_pairs(pairs), -1.0)


def test_correlation_from_pairs_nan_for_too_few_points():
    assert np.isnan(correlation_from_pairs(np.empty((2, 1))))


def test_auto_correlation_lag_zero_is_one():
    rng = np.random.default_rng(0)
    data = rng.uniform(0.1, 1.0, size=(5, 20))
    times, r, r_err = auto_correlation(data, [0, 1, 2], rng)
    assert np.isclose(r[0], 1.0)
    assert r_err[0] == 0 or np.isnan(r_err[0]) or r_err[0] >= 0


def test_cross_correlation_is_antisymmetric_lag_layout():
    rng = np.random.default_rng(0)
    data1 = rng.uniform(0.1, 1.0, size=(5, 30))
    data2 = rng.uniform(0.1, 1.0, size=(5, 30))
    times = [0, 1, 2]
    lags, r, r_err = cross_correlation(data1, data2, times, rng)
    assert len(lags) == 2 * len(times) - 1
    assert lags[0] == -times[-1]
    assert lags[-1] == times[-1]
    assert np.all(np.diff(lags) > 0)
