import numpy as np

from spine_volatility.current_proxy import calibrate, current_from_point, current_from_trajectories


def test_calibrate_gives_mean_current_of_15pa():
    rng = np.random.default_rng(0)
    HS = rng.uniform(0.2, 1.0, size=(5, 50))
    NL = rng.uniform(0.5, 2.0, size=(5, 50))
    NW = rng.uniform(0.1, 0.4, size=(5, 50))
    calib = calibrate(HS, NL, NW)

    strengths = []
    for t in range(5):
        for s in range(50):
            strengths.append(current_from_point(calib, [HS[t, s], NL[t, s], NW[t, s]]))
    assert np.isclose(np.mean(strengths), 15.0, rtol=1e-6)


def test_current_from_trajectories_matches_pointwise():
    calib = calibrate(np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]))
    trajectories = np.array([[[1.0, 2.0, 0.5], [2.0, 1.0, 0.5]]])  # shape (1, 2, 3)
    currents = current_from_trajectories(calib, trajectories)
    expected0 = current_from_point(calib, [1.0, 2.0, 0.5])
    expected1 = current_from_point(calib, [2.0, 1.0, 0.5])
    np.testing.assert_allclose(currents[0], [expected0, expected1])
