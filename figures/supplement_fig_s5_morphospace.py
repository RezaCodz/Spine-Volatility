"""Figure S5: event-class counterfactual reconstructions at six morphospace locations.

Repeats Fig. 6c's counterfactual comparison at six starting points, each
setting one feature to its smallest/largest observed day-14 value while
holding the other two at the population mean, to check the small-spine
result isn't specific to that one example spine.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from spine_volatility import current_proxy, data
from spine_volatility.distributions import population_variance
from spine_volatility.models import relaxation
from spine_volatility.paths import FIGURES_DIR, INFERENCE_DIR, ensure_output_dirs
from spine_volatility.plotting.density_panel import add_kde_density
from spine_volatility.plotting.style import apply_base_style
from spine_volatility.simulation import event_knockout, ou_trajectory

CURRENT_YLIM = (0, 50)


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
    density_mask = t_future >= 20.0

    fig = plt.figure(figsize=(14, 7.5))
    outer = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.35)
    legend_handles = None

    for cell, (title, spine_idx, feature_dim) in zip(outer, specs):
        inner = cell.subgridspec(1, 2, width_ratios=[1.0, 0.22], wspace=0.05)
        ax = fig.add_subplot(inner[0, 0])
        ax_density = fig.add_subplot(inner[0, 1], sharey=ax)

        start_point = pop_mean.copy()
        start_point[feature_dim] = features_at_day14[feature_dim][spine_idx]

        rng = np.random.default_rng(792)
        handles = []
        for name in event_knockout.PLOT_ORDER:
            style = event_knockout.STYLE[name]
            cov = event_knockout.covariance_from_event_rates(jumps, event_knockout.EVENT_RATE_MODELS[name])
            trajectories = event_knockout.simulate_ar1_from_start(start_point, cov, tau, t_future, feature_min, 150, rng)
            if len(trajectories) == 0:
                continue
            currents = current_proxy.current_from_trajectories(calib, trajectories)

            if name in event_knockout.TRAJECTORY_MODELS:
                for current in currents:
                    ax.plot(t_future, current, color=style["color"], alpha=style["trace_alpha"],
                             linewidth=0.6, linestyle=style["linestyle"])
            add_kde_density(ax_density, currents[:, density_mask], CURRENT_YLIM, style["color"], style["linestyle"])
            handles.append(Line2D([0], [0], color=style["color"], linestyle=style["linestyle"], label=style["label"]))
        legend_handles = legend_handles or handles

        ax.set_title(title, fontsize=9)
        ax.set_ylim(*CURRENT_YLIM)
        ax.set_xlim(14.0, 30.0)
        ax_density.set_ylim(*CURRENT_YLIM)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Current (pA)")

    fig.legend(handles=legend_handles, frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Fig. S5: event-class counterfactuals across morphospace", y=1.06)
    fig.savefig(out_dir / "fig_s5_morphospace_counterfactuals.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_s5_morphospace_counterfactuals.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure S5 to {out_dir}")


if __name__ == "__main__":
    main()
