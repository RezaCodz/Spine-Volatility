"""Figure 5: inferred event-class kernels (A-C) and covariance-amplitude surface (D).

Physical amplitudes are loaded from the cached one-exponential+offset
posterior (results/inference/one_exp_offset.pkl) rather than hand-typed, so
they can never silently drift from a rerun of the fit.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spine_volatility import data
from spine_volatility.distributions import population_variance
from spine_volatility.models import relaxation
from spine_volatility.models.event_mixture import covariance_surface, event_point, physical_amplitudes, surface_crossing
from spine_volatility.paths import FIGURES_DIR, INFERENCE_DIR, ensure_output_dirs
from spine_volatility.plotting.axes3d import draw_ellipsoid, setup_3d_axis
from spine_volatility.plotting.style import apply_base_style


def panel_abc(amp, out_dir):
    t = np.linspace(-5, 20, 500)
    k_hs = np.where(t >= 0, amp.A_hs * np.exp(-t / amp.tau), 0.0)
    k_nl = np.where(t >= 0, -amp.A_nl * np.exp(-t / amp.tau), 0.0)
    k_nw = np.where(t >= 0, amp.A_nw * np.exp(-t / amp.tau), 0.0)

    blue, green, orange = "#005BBB", "#009E3D", "#F0A500"
    fig, axs = plt.subplots(3, 1, figsize=(5.6, 6.2), sharex=True)

    axs[0].plot(t, k_hs, color=blue, lw=2.0, ls="-", label="Head size")
    axs[0].plot(t, k_nl, color=blue, lw=2.0, ls=":", label="Neck length")
    axs[0].plot(t, k_nw, color=blue, lw=2.0, ls="--", label="Neck width")
    axs[0].set_title("A Coordinated", loc="left", fontweight="bold")
    axs[0].set_ylabel(r"$\mu m^2$ / $\mu m$")
    axs[0].legend(frameon=False, loc="lower right")

    axs[1].plot(t, k_hs, color=green, lw=2.0, ls="-", label="Head size")
    axs[1].set_title("B Head-only", loc="left", fontweight="bold")
    axs[1].set_ylabel(r"$\mu m^2$")
    axs[1].legend(frameon=False, loc="lower right")

    axs[2].plot(t, k_nl, color=orange, lw=2.0, ls=":", label="Neck length")
    axs[2].plot(t, k_nw, color=orange, lw=2.0, ls="--", label="Neck width")
    axs[2].set_title("C Neck-only", loc="left", fontweight="bold")
    axs[2].set_ylabel(r"$\mu m$")
    axs[2].set_xlabel("Time since event (days)")
    axs[2].legend(frameon=False, loc="lower right")

    for ax in axs:
        ax.set_xlim(-6, 21)
        ax.axhline(0, color="0.70", lw=0.8, ls="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(h_pad=1.0)
    fig.savefig(out_dir / "fig5_abc_kernels.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig5_abc_kernels.svg", bbox_inches="tight")
    plt.close(fig)


def panel_d(amp, out_dir):
    X, Y, Z = covariance_surface(amp, n=90)
    all_coordinated = event_point(amp, 1.0, 0.0, 0.0)
    head_events_99 = event_point(amp, 0.001, 0.998, 0.001)
    neck_events_99 = event_point(amp, 0.001, 0.001, 0.998)

    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    plot_limits = np.array([[-0.024, 0.002], [-0.158, 0.006], [-0.0002, 0.00425]])
    setup_3d_axis(ax, plot_limits, plot_limits[:, 0], [-0.02, 0.0], [-0.08, 0.0], [0.0, 0.004], box_aspect=(1, 1, 0.9))

    ax.plot_surface(X, Y, Z, color="#9FB3C2", alpha=0.48, linewidth=0.20, edgecolor=(1, 1, 1, 0.22),
                     shade=False, antialiased=True, rasterized=True)
    draw_ellipsoid(ax, amp.measured, amp.errors_2sigma, color="#D62728", alpha=0.28)

    point_style = dict(s=150, edgecolor="black", linewidth=0.9, depthshade=False)
    ax.scatter(*all_coordinated, color="blue", **point_style)
    ax.scatter(*head_events_99, color="green", **point_style)
    ax.scatter(*neck_events_99, color="orange", **point_style)
    ax.scatter(*amp.measured, color="#B2182B", s=120, edgecolor="black", linewidth=0.8, depthshade=False)

    ax.text2D(0.49, 0.82, "All Coordinated", transform=ax.transAxes, color="blue", fontsize=13, ha="center")
    ax.text2D(0.56, 0.24, "99% Head Events", transform=ax.transAxes, color="green", fontsize=12, ha="center")
    ax.text2D(0.64, 0.43, "99% Neck Events", transform=ax.transAxes, color="orange", fontsize=12, ha="center")
    ax.text2D(0.36, 0.36, "Measured", transform=ax.transAxes, color="red", fontsize=12, ha="center")
    ax.text2D(0.67, 0.06, r"$A_{\mathrm{nl,hs}}$ ($\mu m^3$)", transform=ax.transAxes, rotation=-13, ha="center", fontsize=13)
    ax.text2D(0.27, 0.07, r"$A_{\mathrm{nl,nw}}$ ($\mu m^2$)", transform=ax.transAxes, rotation=20, ha="center", fontsize=13)
    ax.text2D(0.04, 0.55, r"$A_{\mathrm{nw,hs}}$ ($\mu m^3$)", transform=ax.transAxes, rotation=90, ha="center", fontsize=13)
    ax.text2D(0.02, 0.96, "D", transform=ax.transAxes, fontweight="bold", fontsize=14)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    fig.savefig(out_dir / "fig5_d_surface.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig5_d_surface.svg", bbox_inches="tight")
    plt.close(fig)

    crossing = surface_crossing(amp, n=130)
    if "e_range" in crossing:
        print(f"Plausible event fractions: coordinated e={crossing['e_range']}, "
              f"head-only e1={crossing['e1_range']}, neck-only e2={crossing['e2_range']}")
        print("Paper: e=0.12-0.27, e1=0.34-0.47, e2=0.32-0.48")
    else:
        print("No surface points found within the measured 2-sigma ellipsoid.")


def main():
    ensure_output_dirs()
    apply_base_style()
    out_dir = FIGURES_DIR / "fig5"
    out_dir.mkdir(parents=True, exist_ok=True)

    long = data.load_long_term()
    short = data.load_short_term()
    rel_data = relaxation.build_relaxation_data(
        long.HS, long.NL, long.NW, short.HS, short.NL, short.NW,
        data.DELTA_LONG_DAYS, data.DELTA_SHORT_DAYS,
    )
    results = relaxation.fit_model("one_exp_offset", rel_data, INFERENCE_DIR)
    mean = relaxation.posterior_mean(results)
    std = relaxation.posterior_std(results)
    a_nw_hs, a_nl_hs, a_hs, a_nl, a_nw, a_nw_nl = mean[1:7]
    s_nw_hs, s_nl_hs, s_hs, s_nl, s_nw, s_nw_nl = std[1:7]

    var_hs = population_variance(long.HS)
    var_nl = population_variance(long.NL)
    var_nw = population_variance(long.NW)

    amp = physical_amplitudes(
        tau=mean[0], var_hs=var_hs, var_nl=var_nl, var_nw=var_nw,
        a_hs=a_hs, a_nl=a_nl, a_nw=a_nw,
        a_nl_hs=a_nl_hs, a_nl_nw=a_nw_nl, a_nw_hs=a_nw_hs,
        s_nl_hs=s_nl_hs, s_nl_nw=s_nw_nl, s_nw_hs=s_nw_hs,
    )

    panel_abc(amp, out_dir)
    panel_d(amp, out_dir)
    print(f"Saved Figure 5 to {out_dir}")


if __name__ == "__main__":
    main()
