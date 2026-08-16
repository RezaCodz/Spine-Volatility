"""Figure 3: Bayesian model comparison (evidence bars + winning-model posterior marginals)."""

import matplotlib.pyplot as plt
import numpy as np

from spine_volatility import data
from spine_volatility.models import relaxation
from spine_volatility.paths import FIGURES_DIR, INFERENCE_DIR, ensure_output_dirs
from spine_volatility.plotting.style import apply_base_style

MODEL_DISPLAY_NAMES = {
    "constant": "Constant null",
    "one_exp_offset": "One time constant + offset",
    "two_exp_no_offset": "Two time constants, no offset",
    "two_exp_with_offset": "Two time constants + offset",
}
MODEL_COLORS = {
    "one_exp_offset": "#0072B2",
    "two_exp_no_offset": "#D55E00",
    "two_exp_with_offset": "#009E73",
}


def panel_a(all_results, out_dir):
    """Schematic decay curves for the three candidate models + Delta log evidence bars."""
    model_names = list(MODEL_DISPLAY_NAMES)
    logz = np.array([all_results[name].logz[-1] for name in model_names])
    logzerr = np.array([all_results[name].logzerr[-1] for name in model_names])
    null_idx = model_names.index("constant")
    delta_logz = logz - logz[null_idx]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    t = np.linspace(0.01, 15, 500)
    mean_one_exp = relaxation.posterior_mean(all_results["one_exp_offset"])
    mean_two_offset = relaxation.posterior_mean(all_results["two_exp_with_offset"])
    mean_two_no_offset = relaxation.posterior_mean(all_results["two_exp_no_offset"])

    axes[0].plot(t, relaxation.one_exp_model(t, mean_one_exp[3], mean_one_exp[0], relaxation.FIXED_OFFSETS["hs"], 0),
                 color=MODEL_COLORS["one_exp_offset"], label="One time constant + offset")
    axes[0].plot(t, relaxation.two_exp_model(t, mean_two_offset[2], mean_two_offset[3], mean_two_offset[0], mean_two_offset[1],
                                              relaxation.FIXED_OFFSETS["hs"], 0),
                 color=MODEL_COLORS["two_exp_with_offset"], label="Two time constants + offset")
    axes[0].plot(t, relaxation.two_exp_model(t, mean_two_no_offset[2], mean_two_no_offset[3], mean_two_no_offset[0], mean_two_no_offset[1], 0, 0),
                 color=MODEL_COLORS["two_exp_no_offset"], label="Two time constants, no offset")
    axes[0].set_xlabel("Time Lag")
    axes[0].set_ylabel("Correlation")
    axes[0].legend(frameon=False, fontsize=8)

    plot_names = [n for n in model_names if n != "constant"]
    y = np.arange(len(plot_names))
    axes[1].barh(y, [delta_logz[model_names.index(n)] for n in plot_names],
                 xerr=[logzerr[model_names.index(n)] for n in plot_names],
                 color=[MODEL_COLORS[n] for n in plot_names], capsize=4)
    axes[1].set_xlabel(r"$\Delta$ Log evidence")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([MODEL_DISPLAY_NAMES[n] for n in plot_names])

    plt.tight_layout()
    fig.savefig(out_dir / "fig3a_model_comparison.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig3a_model_comparison.svg", bbox_inches="tight")
    plt.close(fig)

    print("Log evidence relative to constant null:")
    for name in model_names:
        print(f"  {MODEL_DISPLAY_NAMES[name]:32s}: logZ={logz[model_names.index(name)]:.2f}  "
              f"Delta={delta_logz[model_names.index(name)]:.2f}")


def panel_b(results_one_exp, out_dir):
    """Marginal posteriors of tau, a_hs, a_nl, a_nw for the winning model."""
    dims = [0, 3, 4, 5]  # tau, a_hs, a_nl, a_nw
    labels = [r"$\tau$", r"$a_{hs}$", r"$a_{nl}$", r"$a_{nw}$"]
    weights = relaxation.dynesty_weights(results_one_exp)
    lo, hi = relaxation.posterior_quantile(results_one_exp, [0.025, 0.975]).T
    mean = relaxation.posterior_mean(results_one_exp)

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.6))
    for ax, dim, label in zip(axes, dims, labels):
        samples = results_one_exp.samples[:, dim]
        ax.hist(samples, weights=weights, bins=60, color="tab:blue", alpha=0.7)
        ax.axvline(lo[dim], color="black", ls="--", lw=1)
        ax.axvline(hi[dim], color="black", ls="--", lw=1)
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.set_title(f"{mean[dim]:.3g}", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "fig3b_posterior_marginals.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig3b_posterior_marginals.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_output_dirs()
    apply_base_style()
    out_dir = FIGURES_DIR / "fig3"
    out_dir.mkdir(parents=True, exist_ok=True)

    long = data.load_long_term()
    short = data.load_short_term()
    rel_data = relaxation.build_relaxation_data(
        long.HS, long.NL, long.NW, short.HS, short.NL, short.NW,
        data.DELTA_LONG_DAYS, data.DELTA_SHORT_DAYS,
    )

    all_results = {name: relaxation.fit_model(name, rel_data, INFERENCE_DIR) for name in relaxation.MODEL_SPECS}

    panel_a(all_results, out_dir)
    panel_b(all_results["one_exp_offset"], out_dir)
    print(f"Saved Figure 3 to {out_dir}")


if __name__ == "__main__":
    main()
