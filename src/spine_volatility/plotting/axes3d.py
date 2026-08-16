"""Shared "publication style" 3D axis helpers.

This exact trio of functions (tick formatting, custom grid lines, custom
tick marks replacing matplotlib's default 3D axis box) was copy-pasted with
only cosmetic variation at least ten times across the original notebooks --
four times within a single notebook (``figure_1.ipynb``) alone.
"""

import numpy as np


def format_tick(tick: float) -> str:
    return f"{tick:g}"


def draw_axis_grid(ax, axis_limits, axis_origin, x_ticks, y_ticks, z_ticks, **style):
    xmin, ymin, zmin = axis_origin
    xmax, ymax, zmax = axis_limits[:, 1]
    grid_style = dict(color="0.82", linewidth=0.7, alpha=0.8, clip_on=False)
    grid_style.update(style)

    for tick in x_ticks:
        ax.plot([tick, tick], [ymin, ymax], [zmin, zmin], **grid_style)
        ax.plot([tick, tick], [ymin, ymin], [zmin, zmax], **grid_style)
    for tick in y_ticks:
        ax.plot([xmin, xmax], [tick, tick], [zmin, zmin], **grid_style)
        ax.plot([xmin, xmin], [tick, tick], [zmin, zmax], **grid_style)
    for tick in z_ticks:
        ax.plot([xmin, xmax], [ymin, ymin], [tick, tick], **grid_style)
        ax.plot([xmin, xmin], [ymin, ymax], [tick, tick], **grid_style)


def draw_axis_ticks(ax, axis_limits, axis_origin, x_ticks, y_ticks, z_ticks):
    xmin, ymin, zmin = axis_origin
    x_span, y_span, z_span = axis_limits[:, 1] - axis_origin
    x_tick_len, y_tick_len, z_tick_len = 0.035 * x_span, 0.035 * y_span, 0.035 * z_span

    for tick in x_ticks:
        ax.plot([tick, tick], [ymin, ymin + y_tick_len], [zmin, zmin], color="black", linewidth=1.1, clip_on=False)
        ax.text(tick, ymin - 0.05 * y_span, zmin - 0.02 * z_span, format_tick(tick), ha="center", va="top", clip_on=False)
    for tick in y_ticks:
        ax.plot([xmin, xmin + x_tick_len], [tick, tick], [zmin, zmin], color="black", linewidth=1.1, clip_on=False)
        ax.text(xmin - 0.05 * x_span, tick, zmin - 0.02 * z_span, format_tick(tick), ha="right", va="center", clip_on=False)
    for tick in z_ticks:
        ax.plot([xmin, xmin + x_tick_len], [ymin, ymin], [tick, tick], color="black", linewidth=1.1, clip_on=False)
        ax.text(xmin - 0.06 * x_span, ymin, tick, format_tick(tick), ha="right", va="center", clip_on=False)


def setup_3d_axis(ax, plot_limits, axis_origin, x_ticks, y_ticks, z_ticks, elev=20, azim=-48, box_aspect=(1, 1, 1)):
    """Configure a matplotlib 3D axis with the shared publication style:
    orthographic projection, no default box, custom grid + tick marks."""
    axis_limits = np.column_stack([axis_origin, plot_limits[:, 1]])
    ax.set_xlim(*plot_limits[0])
    ax.set_ylim(*plot_limits[1])
    ax.set_zlim(*plot_limits[2])
    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()

    draw_axis_grid(ax, axis_limits, axis_origin, x_ticks, y_ticks, z_ticks)

    axis_ends = [
        [axis_limits[0, 1], axis_origin[1], axis_origin[2]],
        [axis_origin[0], axis_limits[1, 1], axis_origin[2]],
        [axis_origin[0], axis_origin[1], axis_limits[2, 1]],
    ]
    for end in axis_ends:
        ax.plot([axis_origin[0], end[0]], [axis_origin[1], end[1]], [axis_origin[2], end[2]],
                color="black", linewidth=2.0, clip_on=False)

    draw_axis_ticks(ax, axis_limits, axis_origin, x_ticks, y_ticks, z_ticks)
    return axis_limits


def draw_ellipsoid(ax, center, stds, n=48, color="#D62728", alpha=0.28, wireframe=False, **kwargs):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = stds[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = stds[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = stds[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    if wireframe:
        ax.plot_wireframe(x, y, z, color=color, linewidth=kwargs.pop("linewidth", 0.65), alpha=alpha, **kwargs)
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=False, rasterized=True, **kwargs)
