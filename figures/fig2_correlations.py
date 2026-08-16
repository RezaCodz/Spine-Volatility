"""Figure 2: measured auto-/cross-correlation functions with the one-exponential+offset fit."""

import matplotlib.pyplot as plt
import numpy as np

from spine_volatility import data
from spine_volatility.models import relaxation
from spine_volatility.paths import FIGURES_DIR, INFERENCE_DIR, ensure_output_dirs
from spine_volatility.plotting.style import apply_base_style


def _weighted_avg_and_err(y, yerr):
    y, yerr = np.asarray(y), np.asarray(yerr)
    yerr = np.where(yerr <= 0, np.nan, yerr)
    w = 1.0 / (yerr ** 2)
    return np.nansum(w * y) / np.nansum(w), np.sqrt(1.0 / np.nansum(w))


def _collapse_auto_short_term(x_pos, y, yerr, n_short, x_short_plot=0.3):
    """Collapse the short-term nonzero auto-correlation lags into a single
    representative point, so the minutes-scale points (a few hundredths of a
    day apart) don't overplot each other next to the days-scale points."""
    idx_short_nonzero = np.arange(1, n_short)
    y_avg, y_avg_err = _weighted_avg_and_err(y[idx_short_nonzero], yerr[idx_short_nonzero])
    x_long, y_long, yerr_long = x_pos[n_short:], y[n_short:], yerr[n_short:]
    return (
        np.concatenate([[0.0, x_short_plot], x_long]),
        np.concatenate([[1.0, y_avg], y_long]),
        np.concatenate([[0.0, y_avg_err], yerr_long]),
    )


def _collapse_cross_near_zero(x, y, yerr, tmax=0.5, x0=0.0):
    """Collapse all short-term cross-correlation lags with |t|<=tmax into one point."""
    x, y, yerr = np.asarray(x), np.asarray(y), np.asarray(yerr)
    mask = np.abs(x) <= tmax
    y_avg, y_avg_err = _weighted_avg_and_err(y[mask], yerr[mask])
    x_new = np.concatenate([[x0], x[~mask]])
    y_new = np.concatenate([[y_avg], y[~mask]])
    yerr_new = np.concatenate([[y_avg_err], yerr[~mask]])
    order = np.argsort(x_new)
    return x_new[order], y_new[order], yerr_new[order]


def main():
    ensure_output_dirs()
    apply_base_style()
    out_dir = FIGURES_DIR / "fig2"
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
    a_nw_hs, a_nl_hs, a_hs, a_nl, a_nw, a_nw_nl = mean[1:7]
    n_hs, n_nl, n_nw = mean[7:10]

    offsets = relaxation.FIXED_OFFSETS

    auto_panels = [
        ("Head Size", rel_data.HS_ac, rel_data.HS_ac_error, a_hs, offsets["hs"], n_hs),
        ("Neck Length", rel_data.NL_ac, rel_data.NL_ac_error, a_nl, offsets["nl"], n_nl),
        ("Neck Width", rel_data.NW_ac, rel_data.NW_ac_error, a_nw, offsets["nw"], n_nw),
    ]
    cross_panels = [
        ("Neck Width - Head Size", rel_data.y_nw_hs, rel_data.yerr_nw_hs, a_nw_hs, offsets["nw_hs"]),
        ("Neck Length - Head Size", rel_data.y_nl_hs, rel_data.yerr_nl_hs, a_nl_hs, offsets["nl_hs"]),
        ("Neck Width - Neck Length", rel_data.y_nw_nl, rel_data.yerr_nw_nl, a_nw_nl, offsets["nw_nl"]),
    ]

    x1 = np.linspace(0.01, 15, 1000)
    x2 = np.linspace(-15, 15, 2000)
    n_short = rel_data.n_short_auto

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax, (title, y, yerr, a_val, c, n_val) in zip(axes[0], auto_panels):
        xplot, yplot, eplot = _collapse_auto_short_term(rel_data.x_pos, y, yerr, n_short)
        ax.errorbar(xplot, yplot, yerr=eplot, marker="o", ls=" ", color="black")
        ax.plot(x1, relaxation.one_exp_model(x1, a_val, tau, c, n_val), color="C0")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-0.1, 1.1)

    for ax, (title, y, yerr, a_val, c) in zip(axes[1], cross_panels):
        xplot, yplot, eplot = _collapse_cross_near_zero(rel_data.x, y, yerr)
        ax.errorbar(xplot, yplot, yerr=eplot, marker="o", ls=" ", color="black")
        ax.plot(x2, relaxation.one_exp_model_cross(x2, a_val, tau, c), color="C0")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-0.5, 0.5)

    fig.suptitle(f"Auto- and cross-correlations, one-exponential+offset fit (tau={tau:.2f} d)", y=1.02)
    plt.tight_layout(h_pad=2.0)
    fig.savefig(out_dir / "fig2_correlations.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_correlations.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 2 to {out_dir}")


if __name__ == "__main__":
    main()
