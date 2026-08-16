"""Shared matplotlib style used across the figure scripts."""

import matplotlib as mpl

BASE_RCPARAMS = {
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_base_style():
    mpl.rcdefaults()
    mpl.rcParams.update(BASE_RCPARAMS)
