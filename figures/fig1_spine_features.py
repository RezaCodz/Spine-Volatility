"""Figure 1 b-d: example-spine measurements and averaged population morphospace.

Panel a (schematic) and the STED micrographs in panels b/c are not
code-reproducible (hand illustration / raw microscopy images) and are
skipped here.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spine_volatility import data
from spine_volatility.pooling import pooled_triplets_with_spine_weights, weighted_mean_std
from spine_volatility.plotting import kde3d
from spine_volatility.plotting.axes3d import setup_3d_axis
from spine_volatility.plotting.style import apply_base_style
from spine_volatility.paths import FIGURES_DIR, ensure_output_dirs

COLORS = ["gold", "tab:blue", "tab:red"]


def panels_bc(long, short, out_dir):
    """Example short-term (spine 427) and long-term (spine 135) 3D scatter with error bars."""
    spine_427_id, spine_427_idx = 427, [0, 2, 3]
    spine_427_labels = ["0 min", "60 min", "120 min"]
    spine_427_points = np.column_stack([
        short.HS[spine_427_idx, spine_427_id], short.NL[spine_427_idx, spine_427_id], short.NW[spine_427_idx, spine_427_id],
    ])

    long_spine_id, long_spine_idx = 135, [0, 2, 4]
    long_spine_labels = ["Day 1", "Day 8", "Day 15"]
    long_spine_points = np.column_stack([
        long.HS[long_spine_idx, long_spine_id], long.NL[long_spine_idx, long_spine_id], long.NW[long_spine_idx, long_spine_id],
    ])

    fig = plt.figure(figsize=(13.5, 6.0))
    ax_short = fig.add_subplot(121, projection="3d")
    ax_long = fig.add_subplot(122, projection="3d")

    plot_limits = np.array([[0.0, 1.5], [0.0, 1.5], [0.18, 0.3]])
    axis_origin = np.array([0.4, 0.4, 0.18])
    ticks = ([0.5, 1.5], [0.5, 1.5], [0.2, 0.3])

    for ax, points, labels, title in [
        (ax_short, spine_427_points, spine_427_labels, f"Spine {spine_427_id}: short"),
        (ax_long, long_spine_points, long_spine_labels, f"Spine {long_spine_id}: long"),
    ]:
        setup_3d_axis(ax, plot_limits, axis_origin, *ticks)
        ax.text2D(0.28, 0.05, r"Head Size ($\mu m^2$)", transform=ax.transAxes, rotation=-17, ha="center", va="center")
        ax.text2D(0.63, 0.10, r"Neck Length ($\mu m$)", transform=ax.transAxes, rotation=23, ha="center", va="center")
        ax.text2D(0.95, 0.55, r"Neck Width ($\mu m$)", transform=ax.transAxes, rotation=90, ha="center", va="center")
        for label, point, color in zip(labels, points, COLORS):
            ax.scatter(*point, s=135, color=color, edgecolor="black", linewidth=0.9, depthshade=False, label=label)
            ax.errorbar(*point, xerr=data.MEASUREMENT_NOISE[0], yerr=data.MEASUREMENT_NOISE[1],
                        zerr=data.MEASUREMENT_NOISE[2], fmt="none", color=color, ecolor=color, alpha=0.8, capsize=3)
        ax.set_title(title, pad=8)
        ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.08, wspace=0.08)
    fig.savefig(out_dir / "fig1_bc_spine_examples.svg", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig1_bc_spine_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_d(long, short, out_dir):
    """Averaged 68%/95% highest-density-region contours of short- and long-term
    populations in log-transformed, z-scored morphospace."""
    HS_l, NL_l, NW_l, w_l = pooled_triplets_with_spine_weights(long.HS, long.NL, long.NW)
    HS_s, NL_s, NW_s, w_s = pooled_triplets_with_spine_weights(short.HS, short.NL, short.NW)

    def logz(values, weights):
        log_values = np.log(values)
        mean, std = weighted_mean_std(log_values, weights)
        return (log_values - mean) / std, mean, std

    HS_l_z, HS_l_mean, HS_l_std = logz(HS_l, w_l)
    NL_l_z, NL_l_mean, NL_l_std = logz(NL_l, w_l)
    NW_l_z, NW_l_mean, NW_l_std = logz(NW_l, w_l)
    HS_s_z, HS_s_mean, HS_s_std = logz(HS_s, w_s)
    NL_s_z, NL_s_mean, NL_s_std = logz(NL_s, w_s)
    NW_s_z, NW_s_mean, NW_s_std = logz(NW_s, w_s)

    grid_size = 90
    axis_min = np.array([kde3d.bounds(HS_l_z)[0], kde3d.bounds(NL_l_z)[0], kde3d.bounds(NW_l_z)[0]])
    axis_max = np.array([kde3d.bounds(HS_l_z)[1], kde3d.bounds(NL_l_z)[1], kde3d.bounds(NW_l_z)[1]])

    grid_long = kde3d.fit_density_grid(np.column_stack([HS_l_z, NL_l_z, NW_l_z]), w_l, axis_min, axis_max, grid_size)
    grid_short = kde3d.fit_density_grid(np.column_stack([HS_s_z, NL_s_z, NW_s_z]), w_s, axis_min, axis_max, grid_size)
    density_avg = 0.5 * (grid_long.density + grid_short.density)
    grid_avg = kde3d.DensityGrid(density=density_avg, axis_min=axis_min, axis_max=axis_max,
                                  grid_size=grid_size, cell_volume=grid_long.cell_volume)

    verts_68, faces_68 = kde3d.hdr_surface(grid_avg, 0.68)
    verts_95, faces_95 = kde3d.hdr_surface(grid_avg, 0.95)

    # Example spines in the same log-z space, for reference.
    def to_logz(points, means, stds):
        return np.column_stack([(np.log(points[:, d]) - means[d]) / stds[d] for d in range(3)])

    long_idx, long_time_idx = 135, [0, 2, 4]
    long_points_raw = np.column_stack([
        long.HS[long_time_idx, long_idx], long.NL[long_time_idx, long_idx], long.NW[long_time_idx, long_idx],
    ])
    long_points_z = to_logz(long_points_raw, [HS_l_mean, NL_l_mean, NW_l_mean], [HS_l_std, NL_l_std, NW_l_std])

    short_idx, short_time_idx = 427, [0, 2, 3]
    short_points_raw = np.column_stack([
        short.HS[short_time_idx, short_idx], short.NL[short_time_idx, short_idx], short.NW[short_time_idx, short_idx],
    ])
    short_points_z = to_logz(short_points_raw, [HS_s_mean, NL_s_mean, NW_s_mean], [HS_s_std, NL_s_std, NW_s_std])

    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    plot_limits = np.column_stack([axis_min, axis_max])
    setup_3d_axis(ax, plot_limits, axis_min, [-2, 0, 2], [-2, 0, 2], [-2, 0, 2])

    ax.add_collection3d(Poly3DCollection(verts_95[faces_95], facecolor="tab:blue", edgecolor="none", alpha=0.16, rasterized=True))
    ax.add_collection3d(Poly3DCollection(verts_68[faces_68], facecolor="tab:red", edgecolor="none", alpha=0.28, rasterized=True))

    for idx, point in enumerate(long_points_z):
        ax.scatter(*point, s=140, color="blue", edgecolor="black", linewidth=0.9, depthshade=False, marker="o",
                   label="Long" if idx == 0 else "_nolegend_")
    for idx, point in enumerate(short_points_z):
        ax.scatter(*point, s=140, color="orange", edgecolor="black", linewidth=0.9, depthshade=False, marker="^",
                   label="Short" if idx == 0 else "_nolegend_")

    ax.text2D(0.28, 0.04, "z-score log(Head size)", transform=ax.transAxes, rotation=-17, ha="center", va="center")
    ax.text2D(0.64, 0.09, "z-score log(Neck length)", transform=ax.transAxes, rotation=23, ha="center", va="center")
    ax.text2D(0.96, 0.54, "z-score log(Neck width)", transform=ax.transAxes, rotation=90, ha="center", va="center")
    ax.set_title("Fig. 1d: averaged 68%/95% morphospace contours", pad=8)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)
    fig.savefig(out_dir / "fig1_d_morphospace_contours.svg", dpi=180)
    fig.savefig(out_dir / "fig1_d_morphospace_contours.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_output_dirs()
    apply_base_style()
    plt.rcParams["svg.fonttype"] = "none"
    out_dir = FIGURES_DIR / "fig1"
    out_dir.mkdir(parents=True, exist_ok=True)

    long = data.load_long_term()
    short = data.load_short_term()

    panels_bc(long, short, out_dir)
    panel_d(long, short, out_dir)
    print(f"Saved Figure 1 panels to {out_dir}")


if __name__ == "__main__":
    main()
