"""Shared helper for the small KDE "adjacent distribution" panels (Fig. 6c, Fig. S5)."""

import numpy as np
from scipy.stats import gaussian_kde


def add_kde_density(ax, values, y_limits, color, linestyle="-", linewidth=1.4, alpha=0.9, n_grid=300):
    """Plot a horizontal KDE of ``values`` (e.g. terminal current samples) on
    ``ax``, oriented to sit beside a shared-y-axis trace plot."""
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 2 or np.std(values) == 0:
        return
    y_grid = np.linspace(y_limits[0], y_limits[1], n_grid)
    density = gaussian_kde(values, bw_method="silverman")(y_grid)
    ax.plot(density, y_grid, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
