"""Figure 4: event-based model schematic (traces + forward/inverse correlation-matrix diagram).

Illustrative only -- these panels explain the modeling logic (Eq. 2-3), not
a fit to data. Timescales are deliberately spread far apart per feature so
the asymmetric cross-correlation kink is visually obvious, and no
quenched-disorder offset is added (every curve decays fully to zero).
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

from spine_volatility.models.relaxation import auto_corr_curve, cross_corr_curve, kernel_f
from spine_volatility.paths import FIGURES_DIR, ensure_output_dirs
from spine_volatility.plotting.style import apply_base_style

FEATURE_NAMES = ["Head size", "Neck length", "Neck width"]
FEATURE_COLORS = {"Head size": "#2B6CB0", "Neck length": "#C2410C", "Neck width": "#2F855A"}
J = {"Head size": 0.150, "Neck length": -0.125, "Neck width": 0.095}
ALPHA = {"Head size": 0.55, "Neck length": 0.50, "Neck width": 0.60}
TAU1 = {"Head size": 0.5, "Neck length": 1.2, "Neck width": 0.12}
TAU2 = {"Head size": 6.0, "Neck length": 4.0, "Neck width": 0.9}
A_AUTO = {"Head size": 0.35, "Neck length": 0.40, "Neck width": 0.55}


def cross_amplitude(i, j):
    sign = np.sign(J[i] * J[j])
    return sign * 0.5 * np.sqrt(A_AUTO[i] * A_AUTO[j])


def panel_a(out_dir):
    rng = np.random.default_rng(11)
    t = np.linspace(0, 12, 1600)
    event_times = np.sort(rng.uniform(0.5, 11.5, size=4))
    event_sign = rng.choice([1.0, -1.0], size=len(event_times), p=[0.5, 0.5])
    event_sign[0] = 1.0
    event_size = rng.gamma(shape=2.5, scale=0.45, size=len(event_times))
    event_marks = event_sign * event_size
    is_potentiation = event_sign > 0

    def coordinated_trace(direction, base_amplitude, alpha, tau1, tau2):
        trace = np.zeros_like(t)
        for event_time, mark in zip(event_times, event_marks):
            trace += direction * base_amplitude * mark * kernel_f(t - event_time, alpha, tau1, tau2)
        return trace

    traces = {
        name: coordinated_trace(np.sign(J[name]), abs(J[name]), ALPHA[name], TAU1[name], TAU2[name])
        for name in FEATURE_NAMES
    }

    fig = plt.figure(figsize=(7.55, 2.85))
    outer = fig.add_gridspec(1, 3, width_ratios=[0.72, 2.55, 1.45],
                              left=0.055, right=0.985, bottom=0.18, top=0.88, wspace=0.12)
    label_grid = outer[0, 0].subgridspec(3, 1, hspace=0.10)
    trace_grid = outer[0, 1].subgridspec(3, 1, hspace=0.10)
    ax_formula = fig.add_subplot(outer[0, 2])
    label_axes = [fig.add_subplot(label_grid[row, 0]) for row in range(3)]
    trace_axes = [fig.add_subplot(trace_grid[row, 0]) for row in range(3)]

    fig.text(0.012, 0.92, "A", fontsize=14, fontweight="bold")
    trace_axes[0].set_title("coordinated events (potentiation & depression)", loc="left",
                             fontsize=10, fontweight="bold", pad=7)

    ax_formula.axis("off")
    ax_formula.text(0.02, 0.94, "general model", transform=ax_formula.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_formula.text(0.02, 0.70, r"$x_{ik}(t)=x_i^0+\delta x_{ik}$" "\n" r"$\qquad+\sum_l J_i^l\,f_i(t-t^l)$",
                     transform=ax_formula.transAxes, fontsize=9.4, va="top")
    ax_formula.text(0.02, 0.44,
                     r"$f_i(t)=\theta(t)\left(\frac{\alpha_i}{\tau_{i1}}e^{-t/\tau_{i1}}\right.$" "\n"
                     r"$\left.\qquad\qquad+\frac{1-\alpha_i}{\tau_{i2}}e^{-t/\tau_{i2}}\right)$",
                     transform=ax_formula.transAxes, fontsize=8.6, va="top")
    ax_formula.text(0.02, 0.16, r"$J_i^l=\pm|J_i^l|$ (potentiation/", transform=ax_formula.transAxes, fontsize=8.6, va="top")
    ax_formula.text(0.02, 0.06, "depression, variable size)", transform=ax_formula.transAxes, fontsize=8.6, va="top")

    for label_ax, name in zip(label_axes, FEATURE_NAMES):
        label_ax.axis("off")
        label_ax.text(0.95, 0.52, name, transform=label_ax.transAxes, ha="right", va="center",
                       fontsize=9.4, fontweight="bold", color=FEATURE_COLORS[name])

    marker_scale = 55
    for row, name in enumerate(FEATURE_NAMES):
        ax = trace_axes[row]
        y = traces[name]
        y_lo, y_hi = y.min(), y.max()
        y_range = max(y_hi - y_lo, 1e-9)
        ax.plot(t, y, lw=1.9, color=FEATURE_COLORS[name])
        ax.axhline(0, color="0.45", lw=0.75, ls=(0, (2, 2)), zorder=0)

        if row == len(trace_axes) - 1:
            rug_y = y_lo - 0.14 * y_range
            ax.scatter(event_times[is_potentiation], np.full(is_potentiation.sum(), rug_y), marker="^",
                       s=marker_scale * event_size[is_potentiation], facecolor="0.05", edgecolor="none", zorder=4, clip_on=False)
            ax.scatter(event_times[~is_potentiation], np.full((~is_potentiation).sum(), rug_y), marker="v",
                       s=marker_scale * event_size[~is_potentiation], facecolor="none", edgecolor="0.05",
                       linewidths=1.1, zorder=4, clip_on=False)
            ax.set_ylim(y_lo - 0.34 * y_range, y_hi + 0.12 * y_range)
        else:
            ax.set_ylim(y_lo - 0.12 * y_range, y_hi + 0.12 * y_range)

        ax.set_xlim(t.min(), t.max())
        ax.set_yticks([])
        ax.set_xticks(event_times)
        ax.set_xticklabels([f"{v:.1f}" for v in event_times])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("0.25")
        ax.tick_params(axis="x", colors="0.25", length=3)
        ax.tick_params(axis="y", length=0)
        if row < len(trace_axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time", labelpad=6)

    trace_axes[1].text(1.01, 0.5, "change from baseline", transform=trace_axes[1].transAxes, rotation=90,
                        ha="left", va="center", fontsize=8.5, color="0.25")

    legend_handles = [
        plt.Line2D([0], [0], marker="^", color="0.05", linestyle="none", markersize=7, label="potentiation event"),
        plt.Line2D([0], [0], marker="v", color="0.05", linestyle="none", markersize=7, markerfacecolor="none", label="depression event"),
    ]
    trace_axes[-1].legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.62), ncol=2,
                           frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.2)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"fig4_panel_a.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def style_matrix_cell(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_flow_arrow(fig, pos_left, pos_right, color="0.25"):
    y_mid = 0.25 * (pos_left.y0 + pos_left.y1 + pos_right.y0 + pos_right.y1)
    fig.add_artist(FancyArrowPatch(
        (pos_left.x1 + 0.006, y_mid), (pos_right.x0 - 0.006, y_mid), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=color, clip_on=False, zorder=10,
    ))


def panel_bc(out_dir):
    t_range = np.linspace(-10, 10, 400)
    diag_ylim_solid = (-0.08, 0.62)
    diag_ylim_ambiguous = (-0.78, 0.78)
    offdiag_ylim = (-0.5, 0.5)
    ambiguous_style = ((1.0, "black", 0.16), (-1.0, "black", -0.16))

    def draw_matrix(fig, gs, diagonal_mode="solid", offdiag_mode="solid"):
        for row in range(3):
            for col in range(3):
                ax = fig.add_subplot(gs[row, col])
                ax.axhline(0, color="0.8", lw=0.6, ls="--", zorder=0)
                if row == col:
                    name = FEATURE_NAMES[row]
                    if diagonal_mode == "solid":
                        curve = auto_corr_curve(t_range, A_AUTO[name], ALPHA[name], TAU1[name], TAU2[name])
                        ax.plot(t_range, curve, color=FEATURE_COLORS[name], lw=1.6)
                        ax.set_ylim(*diag_ylim_solid)
                    elif diagonal_mode == "ambiguous":
                        for sign, color, y_shift in ambiguous_style:
                            curve = sign * auto_corr_curve(t_range, A_AUTO[name], ALPHA[name], TAU1[name], TAU2[name]) + y_shift
                            ax.plot(t_range, curve, color=color, lw=1.6, ls="--")
                        ax.set_ylim(*diag_ylim_ambiguous)
                    else:
                        ax.set_ylim(*diag_ylim_solid)
                else:
                    i, j = FEATURE_NAMES[row], FEATURE_NAMES[col]
                    if offdiag_mode == "solid":
                        curve = cross_corr_curve(t_range, cross_amplitude(i, j), ALPHA[i], TAU1[i], TAU2[i], ALPHA[j], TAU1[j], TAU2[j])
                        ax.plot(t_range, curve, color="black", lw=1.6)
                    elif offdiag_mode == "ambiguous":
                        for sign, color, y_shift in ambiguous_style:
                            peak = sign * abs(cross_amplitude(i, j))
                            curve = cross_corr_curve(t_range, peak, ALPHA[i], TAU1[i], TAU2[i], ALPHA[j], TAU1[j], TAU2[j]) + y_shift
                            ax.plot(t_range, curve, color=color, lw=1.6, ls="--")
                    ax.set_ylim(*offdiag_ylim)
                ax.set_xlim(t_range.min(), t_range.max())
                style_matrix_cell(ax)

    fig = plt.figure(figsize=(13.4, 4.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.20, left=0.035, right=0.99, top=0.82, bottom=0.10)
    panel_b_gs = outer[0, 0].subgridspec(1, 2, width_ratios=[0.8, 1.55], wspace=0.32)
    panel_c_gs = outer[0, 1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)

    fig.text(0.008, 0.93, "B", fontsize=14, fontweight="bold")
    fig.text(0.548, 0.93, "C", fontsize=14, fontweight="bold")

    kernel_gs = panel_b_gs[0, 0].subgridspec(3, 1, hspace=0.35)
    t_kernel = np.linspace(-0.1, 3.0, 400)
    kernel_y_max = max(abs(J[name]) * kernel_f(np.array([0.0]), ALPHA[name], TAU1[name], TAU2[name])[0] for name in FEATURE_NAMES)
    fig.text(0.045, 0.885, "single-event response", fontsize=9, fontweight="bold")
    for row, name in enumerate(FEATURE_NAMES):
        ax = fig.add_subplot(kernel_gs[row, 0])
        y = J[name] * kernel_f(t_kernel, ALPHA[name], TAU1[name], TAU2[name])
        ax.plot(t_kernel, y, color=FEATURE_COLORS[name], lw=1.7)
        ax.axhline(0, color="0.75", lw=0.7, ls="--", zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, ls=":", zorder=0)
        ax.set_title(name, loc="left", fontsize=7.6, color=FEATURE_COLORS[name], fontweight="bold", pad=3)
        ax.set_ylim(-1.15 * kernel_y_max, 1.15 * kernel_y_max)
        style_blank(ax)

    fig.text(0.045, 0.045,
              r"$f_i(t)=\theta(t)\left(\frac{\alpha_i}{\tau_{i1}}e^{-t/\tau_{i1}}"
              r"+\frac{1-\alpha_i}{\tau_{i2}}e^{-t/\tau_{i2}}\right)$" "\n"
              r"different $\alpha_i,\tau_{i1},\tau_{i2}$ per feature", fontsize=7.2, color="0.30")

    matrix_gs_b = panel_b_gs[0, 1].subgridspec(3, 3, wspace=0.10, hspace=0.10)
    fig.text(0.315, 0.885, "predicted correlation matrix", fontsize=9, fontweight="bold")
    draw_matrix(fig, matrix_gs_b, "solid", "solid")

    matrix_gs_c1 = panel_c_gs[0, 0].subgridspec(3, 3, wspace=0.10, hspace=0.10)
    fig.text(0.585, 0.885, "auto-correlations (fit first)", fontsize=9, fontweight="bold")
    draw_matrix(fig, matrix_gs_c1, "solid", "empty")

    matrix_gs_c2 = panel_c_gs[0, 1].subgridspec(3, 3, wspace=0.10, hspace=0.10)
    fig.text(0.775, 0.885, "cross-correlations (inferred)", fontsize=9, fontweight="bold")
    draw_matrix(fig, matrix_gs_c2, "empty", "ambiguous")

    add_flow_arrow(fig, panel_b_gs[0, 0].get_position(fig), panel_b_gs[0, 1].get_position(fig))
    add_flow_arrow(fig, panel_c_gs[0, 0].get_position(fig), panel_c_gs[0, 1].get_position(fig))

    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"fig4_panel_bc.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_output_dirs()
    apply_base_style()
    out_dir = FIGURES_DIR / "fig4"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_a(out_dir)
    panel_bc(out_dir)
    print(f"Saved Figure 4 panels to {out_dir}")


if __name__ == "__main__":
    main()
