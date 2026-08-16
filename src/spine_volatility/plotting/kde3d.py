"""3D kernel-density highest-density-region (HDR) isosurfaces (Fig. 1d, Fig. 6a).

Consolidates a KDE -> marching-cubes isosurface pattern that appeared three
times within a single notebook (``figure_1.ipynb``), each rebuilding the
same grid/threshold/rescale logic from scratch.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde
from skimage.measure import marching_cubes


@dataclass
class DensityGrid:
    density: np.ndarray  # shape (grid_size, grid_size, grid_size)
    axis_min: np.ndarray  # [x, y, z]
    axis_max: np.ndarray
    grid_size: int
    cell_volume: float


def bounds(v: np.ndarray, margin: float = 1.0) -> tuple[float, float]:
    return float(v.min() - margin * np.std(v)), float(v.max() + margin * np.std(v))


def fit_density_grid(
    points: np.ndarray, weights: np.ndarray, axis_min: np.ndarray, axis_max: np.ndarray, grid_size: int = 90
) -> DensityGrid:
    """Weighted Gaussian KDE of 3D ``points``, evaluated on a regular grid."""
    kde = gaussian_kde(points.T, weights=weights, bw_method=1.0)
    axes = [np.linspace(axis_min[d], axis_max[d], grid_size) for d in range(3)]
    HH, NN, WW = np.meshgrid(*axes, indexing="ij")
    grid_points = np.vstack([HH.ravel(), NN.ravel(), WW.ravel()])
    density = kde(grid_points).reshape(HH.shape)

    cell_volume = np.prod([(axes[d][1] - axes[d][0]) for d in range(3)])
    return DensityGrid(density=density, axis_min=np.asarray(axis_min, dtype=float),
                        axis_max=np.asarray(axis_max, dtype=float), grid_size=grid_size, cell_volume=cell_volume)


def hdr_surface(grid: DensityGrid, mass_level: float):
    """Vertices/faces of the ``mass_level`` highest-density-region isosurface
    (e.g. 0.68 or 0.95), suitable for ``Poly3DCollection``."""
    flat = grid.density.ravel()
    order = np.argsort(flat)[::-1]
    sorted_density = flat[order]
    cum_mass = np.cumsum(sorted_density * grid.cell_volume)
    cum_mass /= cum_mass[-1]
    threshold = sorted_density[np.searchsorted(cum_mass, mass_level)]

    verts, faces, _, _ = marching_cubes(grid.density, level=threshold)
    for d in range(3):
        verts[:, d] = grid.axis_min[d] + verts[:, d] * (grid.axis_max[d] - grid.axis_min[d]) / (grid.grid_size - 1)
    return verts, faces
