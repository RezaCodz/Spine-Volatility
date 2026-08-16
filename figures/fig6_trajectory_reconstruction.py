"""Figure 6: generative reconstruction of spine morphology and synaptic-current trajectories.

Panel a: reconstructed head-size/current trajectories for representative
large and small spines (days 0-14), plus their projection into the
population morphospace. Panel c: predicted synaptic-current trajectories
for a representative small spine (days 14-20), comparing the full model
against counterfactuals that retain only subsets of the three event classes
(Fig. 5). Panel b (schematic) is a hand-drawn illustration, not reproduced.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spine_volatility import current_proxy, data
from spine_volatility.distributions import population_variance
from spine_volatility.models import relaxation
from spine_volatility.plotting import kde3d
from spine_volatility.plotting.axes3d import setup_3d_axis
from spine_volatility.plotting.style import apply_base_style
from spine_volatility.paths import FIGURES_DIR, INFERENCE_DIR, ensure_output_dirs
from spine_volatility.pooling import pooled_triplets_with_spine_weights, weighted_mean_std
from spine_volatility.simulation import event_knockout, ou_trajectory

RNG_SEED = 123


def fit_covariance(long, rel_data):
    results = relaxation.fit_model("one_exp_offset", rel_data, INFERENCE_DIR)
    mean = relaxation.posterior_mean(results)
    tau = mean[0]
    a_nw_hs, a_nl_hs, a_hs, a_nl, a_nw, a_nw_nl = mean[1:7]

    variances = np.array([population_variance(long.HS), population_variance(long.NL), population_variance(long.NW)])
    diag_amplitudes = np.array([a_hs, a_nl, a_nw])
    cross_amplitudes = {(0, 1): a_nl_hs, (0, 2): a_nw_hs, (1, 2): a_nw_nl}
    A = ou_trajectory.covariance_from_amplitudes(variances, diag_amplitudes, cross_amplitudes)
    return A, tau, variances, diag_amplitudes


def choose_example_spines(long):
    valid = [i for i in range(long.HS.shape[1])
             if np.all(long.HS[:5, i] > 0) and np.all(long.NL[:5, i] > 0) and np.all(long.NW[:5, i] > 0)]
    heads_t0 = np.array([long.HS[0, i] for i in valid])
    small_idx = valid[int(np.argmin(heads_t0))]
    large_idx = valid[int(np.argmax(heads_t0))]
    return valid, small_idx, large_idx


def spine_point(long, idx, col):
    return np.array([long.HS[col, idx], long.NL[col, idx], long.NW[col, idx]])


def main():
    ensure_output_dirs()
    apply_base_style()
    out_dir = FIGURES_DIR / "fig6"
    out_dir.mkdir(parents=True, exist_ok=True)

    long = data.load_long_term()
    short = data.load_short_term()
    rel_data = relaxation.build_relaxation_data(
        long.HS, long.NL, long.NW, short.HS, short.NL, short.NW,
        data.DELTA_LONG_DAYS, data.DELTA_SHORT_DAYS,
    )
    A, tau, variances, diag_amplitudes = fit_covariance(long, rel_data)
    calib = current_proxy.calibrate(long.HS, long.NL, long.NW)

    valid_spines, small_idx, large_idx = choose_example_spines(long)
    target_small = np.array([spine_point(long, small_idx, c) for c in [0, 1, 2]])
    target_days_small = data.DELTA_LONG_DAYS[[0, 1, 2]]
    target_large = np.array([spine_point(long, large_idx, c) for c in [0, 2, 4]])
    target_days_large = data.DELTA_LONG_DAYS[[0, 2, 4]]

    feature_min = np.array([
        np.min(long.HS[long.HS > 0]), np.min(long.NL[long.NL > 0]), np.min(long.NW[long.NW > 0]),
    ])

    T_end, N = 30.0, 5000
    t = np.linspace(0, T_end, N)
    dt = t[1] - t[0]
    rng = np.random.default_rng(RNG_SEED)

    trajectories_small = ou_trajectory.sample_posterior(
        A, tau, N, dt, target_small, target_days_small, t, feature_min, data.MEASUREMENT_NOISE, 80, rng,
    )
    trajectories_large = ou_trajectory.sample_posterior(
        A, tau, N, dt, target_large, target_days_large, t, feature_min, data.MEASUREMENT_NOISE, 80, rng,
    )
    trajectories_small = np.array(trajectories_small)
    trajectories_large = np.array(trajectories_large)

    current_small = current_proxy.current_from_trajectories(calib, trajectories_small)
    current_large = current_proxy.current_from_trajectories(calib, trajectories_large)

    # Population morphospace contour (95% HDR, log-z), for the 3D projection panel.
    HS_l, NL_l, NW_l, w_l = pooled_triplets_with_spine_weights(long.HS, long.NL, long.NW)
    log_hs, log_nl, log_nw = np.log(HS_l), np.log(NL_l), np.log(NW_l)
    means = [weighted_mean_std(v, w_l)[0] for v in (log_hs, log_nl, log_nw)]
    stds = [weighted_mean_std(v, w_l)[1] for v in (log_hs, log_nl, log_nw)]
    points_z = np.column_stack([(log_hs - means[0]) / stds[0], (log_nl - means[1]) / stds[1], (log_nw - means[2]) / stds[2]])
    axis_min = np.array([kde3d.bounds(points_z[:, d])[0] for d in range(3)])
    axis_max = np.array([kde3d.bounds(points_z[:, d])[1] for d in range(3)])
    grid = kde3d.fit_density_grid(points_z, w_l, axis_min, axis_max, grid_size=90)
    verts_95, faces_95 = kde3d.hdr_surface(grid, 0.95)

    def to_logz(X):
        return np.column_stack([(np.log(X[:, d]) - means[d]) / stds[d] for d in range(3)])

    # ------------------------------------------------------------
    # Panel a
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(13, 7))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.28)
    left_gs = outer[0, 0].subgridspec(2, 1, hspace=0.30)
    ax_hs = fig.add_subplot(left_gs[0, 0])
    ax_current = fig.add_subplot(left_gs[1, 0])
    ax3d = fig.add_subplot(outer[0, 1], projection="3d")

    observed_mask = t <= 14.0
    for X in trajectories_small:
        ax_hs.plot(t[observed_mask], X[observed_mask, 0], color="blue", alpha=0.08, linewidth=0.6)
    for X in trajectories_large:
        ax_hs.plot(t[observed_mask], X[observed_mask, 0], color="gray", alpha=0.08, linewidth=0.6)
    ax_hs.errorbar(target_days_small, target_small[:, 0], yerr=1.96 * data.MEASUREMENT_NOISE[0], fmt="o", color="blue", capsize=4)
    ax_hs.errorbar(target_days_large, target_large[:, 0], yerr=1.96 * data.MEASUREMENT_NOISE[0], fmt="o", color="black", capsize=4)
    ax_hs.set_ylabel(r"Head size ($\mu m^2$)")
    ax_hs.set_xlim(-0.5, 14.5)

    for I in current_small:
        ax_current.plot(t[observed_mask], I[observed_mask], color="blue", alpha=0.05, linewidth=0.8)
    for I in current_large:
        ax_current.plot(t[observed_mask], I[observed_mask], color="gray", alpha=0.05, linewidth=0.8)
    ax_current.set_yscale("log")
    ax_current.set_xlabel("Time (days)")
    ax_current.set_ylabel("Synaptic current (pA)")
    ax_current.set_xlim(-0.5, 14.5)

    setup_3d_axis(ax3d, np.column_stack([axis_min, axis_max]), axis_min, [-2, 0, 2], [-2, 0, 2], [-2, 0, 2])
    ax3d.add_collection3d(Poly3DCollection(verts_95[faces_95], facecolor="tab:blue", edgecolor="none", alpha=0.16, rasterized=True))
    for X in trajectories_small:
        Xz = to_logz(X[observed_mask])
        ax3d.plot(Xz[:, 0], Xz[:, 1], Xz[:, 2], color="blue", alpha=0.035, linewidth=0.6)
    for X in trajectories_large:
        Xz = to_logz(X[observed_mask])
        ax3d.plot(Xz[:, 0], Xz[:, 1], Xz[:, 2], color="gray", alpha=0.035, linewidth=0.6)
    ax3d.set_title("Population morphospace projection")

    fig.suptitle("Fig. 6a: reconstructed trajectories (small=blue, large=black/gray)")
    fig.savefig(out_dir / "fig6a_trajectory_reconstruction.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig6a_trajectory_reconstruction.svg", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # Panel c: event-class counterfactuals for a representative small spine
    # ------------------------------------------------------------
    A_hs_event = diag_amplitudes[0] * variances[0]
    A_nl_event = diag_amplitudes[1] * variances[1]
    A_nw_event = diag_amplitudes[2] * variances[2]
    jumps = event_knockout.calibrate_jump_sizes(A_hs_event, A_nl_event, A_nw_event)

    start_col = int(np.where(np.isclose(data.DELTA_LONG_DAYS, 14.0))[0][0])
    start_point = spine_point(long, small_idx, start_col)
    t_future = np.linspace(14.0, 20.0, 1200)
    rng2 = np.random.default_rng(790)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {"normal": "blue", "only_head_events": "#D55E00",
              "neck_and_coordinated_events": "#0072B2", "only_coordinated_events": "#009E73"}
    for name in event_knockout.PLOT_ORDER:
        cov = event_knockout.covariance_from_event_rates(jumps, event_knockout.EVENT_RATE_MODELS[name])
        trajectories = event_knockout.simulate_ar1_from_start(start_point, cov, tau, t_future, feature_min, 200, rng2)
        if len(trajectories) == 0:
            print(f"Warning: no accepted trajectories for {name}")
            continue
        currents = current_proxy.current_from_trajectories(calib, trajectories)
        mean_current = currents.mean(axis=0)
        ax.plot(t_future, mean_current, color=colors[name], label=name.replace("_", " "), linewidth=2)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Synaptic current (pA)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Fig. 6c: event-class counterfactual current reconstructions")
    fig.tight_layout()
    fig.savefig(out_dir / "fig6c_event_counterfactuals.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig6c_event_counterfactuals.svg", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 6 to {out_dir}")


if __name__ == "__main__":
    main()
