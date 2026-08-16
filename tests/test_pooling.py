import numpy as np

from spine_volatility.pooling import (
    collect_pairs_diff_feat_pooled,
    collect_pairs_same_feat_pooled,
    pooled_triplets_with_spine_weights,
    pooled_values_with_spine_weights,
    weighted_mean_std,
)


def test_collect_pairs_same_feat_pooled_skips_missing():
    # 2 timepoints x 3 spines; spine 1 has a missing (-1) second measurement.
    data = np.array([[1.0, 2.0, 3.0], [4.0, -1.0, 6.0]])
    pairs = collect_pairs_same_feat_pooled(data, delta_steps=1)
    assert pairs.shape == (2, 2)
    assert list(pairs[0]) == [1.0, 3.0]
    assert list(pairs[1]) == [4.0, 6.0]


def test_collect_pairs_diff_feat_pooled_matches_same_feat_when_equal():
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    same = collect_pairs_same_feat_pooled(data, 1)
    diff = collect_pairs_diff_feat_pooled(data, data, 1)
    np.testing.assert_array_equal(same, diff)


def test_weighted_mean_std_scale_invariant():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    w_raw = np.array([1.0, 1.0, 2.0, 2.0])
    w_norm = w_raw / w_raw.sum()
    mean_raw, std_raw = weighted_mean_std(x, w_raw)
    mean_norm, std_norm = weighted_mean_std(x, w_norm)
    assert np.isclose(mean_raw, mean_norm)
    assert np.isclose(std_raw, std_norm)
    # Expected weighted mean: (1+2+3*2+4*2)/6 = 17/6
    assert np.isclose(mean_raw, 17 / 6)


def test_pooled_values_with_spine_weights_weights_are_not_normalized():
    # Two spines, each contributing 2 valid values -> total weight mass = 2 spines.
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    values, weights = pooled_values_with_spine_weights(data)
    assert len(values) == 4
    assert np.isclose(weights.sum(), 2.0)


def test_pooled_triplets_requires_all_three_features_valid():
    data1 = np.array([[1.0], [2.0]])
    data2 = np.array([[1.0], [-1.0]])  # second timepoint invalid
    data3 = np.array([[1.0], [2.0]])
    x, y, z, w = pooled_triplets_with_spine_weights(data1, data2, data3)
    assert len(x) == 1
    assert x[0] == 1.0 and y[0] == 1.0 and z[0] == 1.0
