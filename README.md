# spine-volatility

Analysis pipeline for **"Signatures of Hebbian plasticity in the nanoscale
morphodynamics of cortical spines"** (Soltanipour, Nagel, Willig & Wolf).

This is a clean, from-scratch reimplementation of the paper's analysis as an
importable Python package, organized around the manuscript's own structure
rather than around the original research notebooks. See
[`docs/scope.md`](docs/scope.md) for exactly which notebook this project's
code descends from, cell by cell, and what was intentionally left out.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python >= 3.10.

## Data

Not included in this repository (see `.gitignore`). Place the raw
measurement files at:

```
data/
├── short_term/final_Ctr_short.csv
└── long_term/{WT_headarea,WT_necklength,WT_neckwidth}.csv
```

- **Long-term** data: control-mouse measurements from Steffens et al.,
  *Science Advances* 2021 (motor cortex, layer 1 apical dendrites).
- **Short-term** data: control-mouse measurements from Wegner et al.,
  *eLife* 2022 (visual cortex).

## Reproducing the figures

```bash
python figures/run_bayesian_inference.py   # ~1-3 hours; fits 4 models via dynesty, caches results/inference/*.pkl
python figures/fig1_spine_features.py
python figures/fig2_correlations.py
python figures/fig3_model_comparison.py
python figures/fig4_event_kernel_schematic.py     # no data/inference dependency
python figures/fig5_event_kernels_surface.py
python figures/fig6_trajectory_reconstruction.py
python figures/tables_s1_s2_pair_counts.py
python figures/supplement_fig_s1_distributions.py
python figures/supplement_figs_s2_s4_corners.py
python figures/supplement_fig_s5_morphospace.py
```

Every script but `fig4` and `run_bayesian_inference.py` needs the cached
inference results, so run that one first. Output goes to
`results/figures/<fig>/` and `results/inference/*.pkl` (both gitignored).

| Script | Reproduces |
|---|---|
| `fig1_spine_features.py` | Fig. 1b-d (example-spine measurements + averaged morphospace contours; panel a and the STED micrographs are not code-reproducible) |
| `fig2_correlations.py` | Fig. 2 |
| `run_bayesian_inference.py` | the nested-sampling fits behind Figs. 3, 5, S2-S4 |
| `fig3_model_comparison.py` | Fig. 3 |
| `fig4_event_kernel_schematic.py` | Fig. 4 (illustrative only, not fit to data) |
| `fig5_event_kernels_surface.py` | Fig. 5 |
| `fig6_trajectory_reconstruction.py` | Fig. 6a, 6c (panel b is a hand-drawn schematic) |
| `tables_s1_s2_pair_counts.py` | Tables S1-S2 |
| `supplement_fig_s1_distributions.py` | Fig. S1 |
| `supplement_figs_s2_s4_corners.py` | Figs. S2-S4 |
| `supplement_fig_s5_morphospace.py` | Fig. S5 |

## Package layout

```
src/spine_volatility/
├── data.py                      loading + unit scaling
├── pooling.py                   per-spine pairing/weighting primitives
├── distributions.py             weighted log-normal/Gaussian fits, AIC
├── correlations.py              auto-/cross-correlation, bootstrap CI
├── current_proxy.py             synaptic-current proxy (Methods)
├── models/
│   ├── relaxation.py            Eq. 2-4 kernels; dynesty model comparison, disk-cached
│   └── event_mixture.py         Fig. 5's 3-event-class covariance surface
├── simulation/
│   ├── ou_trajectory.py         spectral-domain OU trajectory generator
│   └── event_knockout.py        event-class-gated counterfactual simulator (Fig. 6c/S5)
├── plotting/                    shared 3D-axis and KDE-isosurface helpers
└── paths.py                     data/results directory configuration
```

`tests/` covers the consolidated numerical primitives (pooling, correlation,
OU-trajectory covariance matching, current-proxy calibration) against
closed-form small examples: `pytest`.

### Validation against the published results

Reproducing this package's numbers against everything the manuscript
reports directly (not just qualitatively) was the actual acceptance test
during development:

- Table S1/S2 pair counts: exact match.
- Fig. S1 weighted ΔAIC (log-normal vs. Gaussian raw values; Gaussian vs.
  log-normal increments): matches the published values to the first
  decimal (e.g. head size +68.6 vs. published +69).
- One-exponential+offset posterior (Fig. S2): tau=3.68 [95% CI 2.67-4.89]
  vs. published tau=3.62 [+1.24/-0.97]; every amplitude within ~0.02 of the
  published value.
- Fig. 5's plausible event-fraction ranges (from the fitted posterior
  through `models/event_mixture.py`): e=0.11-0.26, e1=0.33-0.45,
  e2=0.35-0.50 vs. published e=0.12-0.27, e1=0.34-0.47, e2=0.32-0.48.

## License

MIT.
