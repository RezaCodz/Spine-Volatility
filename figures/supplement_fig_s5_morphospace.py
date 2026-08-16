"""Figure S5: event-class counterfactual reconstructions at six morphospace locations.

Repeats Fig. 6c's counterfactual comparison at six starting points, each
setting one feature to its smallest/largest observed day-14 value while
holding the other two at the population mean, to check the small-spine
result isn't specific to that one example spine.
"""

import matplotlib.pyplot as plt
import numpy as np

from spine_volatility import current_proxy, data
from spine_volatility.distributions import population_variance
from spine_volatility.models import relaxation
from spine_volatility.paths import FIGURES_DIR, INFERENCE_DIR, ensure_output_dirs
from spine_volatility.plotting.style import apply_base_style
from spine_volatility.simulation import event_knockout, ou_trajectory

COLORS = {"normal": "blue", "only_head_events": "#D55E00",
          "neck_and_coordinated_events": "#0072B2", "only_coordinated_events": "#009E73"}


def main():
    ensure_output_dirs()
    apply_base_style()
    out_dir = FIGURES_DIR / "supplement"
    out_dir.mkdir(parents=True, exist_ok=True)

    long = data.load_long_term()
    short = data.load_short_term()
    rel_data = relaxation.build_relaxation_data(
        long.HS, long.NL, long.NW, short.HS, short.NL, short.NW,
        data.DELTA_LONG_DAYS, data.DELTA_SHORT_DAYS,
    )
    results = relaxation.fit_model("one_exp_offset", rel_data, INFERENCE_DIR)
    mean = relaxation.posterior_mean(results)
    tau = mean[0]
    a_hs, a_nl, a_nw = mean[3], mean[4], mean[5]
    var_hs, var_nl, var_nw = population_variance(long.HS), population_variance(long.NL), population_variance(long.NW)
    calib = current_proxy.calibrate(long.HS, long.NL, long.NW)

    jumps = event_knockout.calibrate_jump_sizes(a_hs * var_hs, a_nl * var_nl, a_nw * var_nw)
    feature_min = np.array([np.min(long.HS[long.HS > 0]), np.min(long.NL[long.NL > 0]), np.min(long.NW[long.NW > 0])])

    valid_spines = [i for i in range(long.HS.shape[1])
                    if np.all(long.HS[:5, i] > 0) and np.all(long.NL[:5, i] > 0) and np.all(long.NW[:5, i] > 0)]
    start_col = int(np.where(np.isclose(data.DELTA_LONG_DAYS, 14.0))[0][0])
    hs14 = {i: long.HS[start_col, i] for i in valid_spines}
    nl14 = {i: long.NL[start_col, i] for i in valid_spines}
    nw14 = {i: long.NW[start_col, i] for i in valid_spines}
    pop_mean = np.array([np.mean(list(hs14.values())), np.mean(list(nl14.values())), np.mean(list(nw14.values()))])

    smallest_head = min(valid_spines, key=lambda i: hs14[i])
    largest_head = max(valid_spines, key=lambda i: hs14[i])
    largest_neck_length = max(valid_spines, key=lambda i: nl14[i])
    smallest_neck_width = min(valid_spines, key=lambda i: nw14[i])
    # Neck length sits in the current formula's denominator: restrict the
    # short/wide-neck picks to spines a safe number of std devs from zero.
    nl_std = np.sqrt(event_knockout.covariance_from_event_rates(jumps, event_knockout.EVENT_RATE_MODELS["normal"])[1, 1])
    used = {smallest_head, largest_head, largest_neck_length, smallest_neck_width}
    safe_pool = [i for i in valid_spines if nl14[i] >= 3.0 * nl_std and i not in used]
    safe_short_neck = min(safe_pool, key=lambda i: nl14[i])
    safe_wide_neck = max(safe_pool, key=lambda i: nw14[i])

    specs = [
        ("Smallest head size", smallest_head, 0), ("Largest head size", largest_head, 0),
        ("Smallest neck length", safe_short_neck, 1), ("Largest neck length", largest_neck_length, 1),
        ("Smallest neck width", smallest_neck_width, 2), ("Largest neck width", safe_wide_neck, 2),
    ]
    features_at_day14 = [hs14, nl14, nw14]

    t_future = np.linspace(14.0, 30.0, 2500)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)

    for ax, (title, spine_idx, feature_dim) in zip(axes.ravel(), specs):
        start_point = pop_mean.copy()
        start_point[feature_dim] = features_at_day14[feature_dim][spine_idx]

        rng = np.random.default_rng(792)
        for name in event_knockout.PLOT_ORDER:
            cov = event_knockout.covariance_from_event_rates(jumps, event_knockout.EVENT_RATE_MODELS[name])
            trajectories = event_knockout.simulate_ar1_from_start(start_point, cov, tau, t_future, feature_min, 100, rng)
            if len(trajectories) == 0:
                continue
            currents = current_proxy.current_from_trajectories(calib, trajectories)
            ax.plot(t_future, currents.mean(axis=0), color=COLORS[name], label=name.replace("_", " "), linewidth=1.6)

        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 50)

    axes[1, 0].set_xlabel("Time (days)")
    axes[1, 1].set_xlabel("Time (days)")
    axes[1, 2].set_xlabel("Time (days)")
    axes[0, 0].set_ylabel("Current (pA)")
    axes[1, 0].set_ylabel("Current (pA)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right")

    fig.suptitle("Fig. S5: event-class counterfactuals across morphospace", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "fig_s5_morphospace_counterfactuals.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_s5_morphospace_counterfactuals.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure S5 to {out_dir}")


if __name__ == "__main__":
    main()
