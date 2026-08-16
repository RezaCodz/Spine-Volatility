# Scope: what this package ports, and what it doesn't

The original research repository (`Synaptic_volatility`, private history)
contains 9 Jupyter notebooks, ~16,500 lines of code, with the same core
algorithms independently reimplemented up to nine times. This package
reproduces the manuscript's actual figures and tables (6 main figures, 5
supplementary figures, 2 supplementary tables) from a much smaller,
deduplicated set of that logic, chosen by matching each paper figure to its
source notebook cell rather than porting every notebook wholesale.

## Figure -> source mapping

| Paper item | Source notebook / cell |
|---|---|
| Fig. 1b-c, d | `figure_1.ipynb` cell 1 (example spines) + cell 4 (averaged KDE contours) |
| Fig. 2 | `figure_2.ipynb` cell 1 (correlation + bootstrap) + cell 4 (fit overlay) |
| Fig. 3 | `figure_2.ipynb` cells 7-8 (two-exp models, constant null) + cell 10 (posterior) |
| Fig. 4 | `figure_6.ipynb` cells 0-1 (illustrative schematic; own comments say "not fitted to data") |
| Fig. 5 | `figure_3.ipynb` cells 0-3 |
| Fig. 6 | `figure_4.ipynb` cells 1-2 (trajectory/current) + cell 3 (event-subset knockout -> panel c) |
| Fig. S1 | `figure_1.ipynb` cells 5-6 |
| Figs. S2-S4 | `figure_2.ipynb` cell 10, one corner per model variant |
| Fig. S5 | `figure_4.ipynb` cell 5 (morphospace-location knockout) |
| Tables S1-S2 | pooling/correlation pair-count functions |

Not ported: `01_raw_statistics_messy.ipynb`, `basic_stat.ipynb`,
`analyse_long_term.ipynb`, and `nested_sampeling_copy.ipynb` (an earlier,
uncached draft of the same model comparison superseded by the cleaner,
complete `figure_2.ipynb`) -- none corresponds to a numbered figure or table
in the manuscript. `figure_4.ipynb` cell 4 (a `merged_*`-prefixed near-copy
of cell 3) is likewise superseded by cell 3 and not ported.

## Bugs found and fixed while porting

- **Unbounded rejection-sampling loop.** The original OU trajectory
  posterior sampler looped until it accepted enough trajectories with no
  cap on attempts. `simulation/ou_trajectory.sample_posterior` caps
  attempts and raises `RejectionSamplingFailed` instead of hanging.
- **Corner plot saved as SVG content into a `.pdf`-named file**, and **a
  figure silently overwritten three times by duplicate cells** with the
  same output filename -- both are structural non-issues here: each figure
  has exactly one script and one unambiguous output path
  (`paths.FIGURES_DIR`).
- **The single most expensive step was never cached.** A full model
  comparison means four dynesty nested-sampling runs (ndim 9, 10, 17, 17,
  nlive=1000); the original notebooks re-ran all of them on every
  execution. `models/relaxation.fit_model` pickles each run to
  `results/inference/{model}.pkl` and every downstream script loads from
  there.
- **Hand-copy-typed posterior constants.** `figure_3.ipynb`'s kernel/event
  constants (`a_hs=0.23`, etc.) turned out to be exactly the one-exponential
  posterior medians, transcribed by hand rather than loaded programmatically
  -- and a second, independent hardcoded copy of the same OU covariance
  matrix in `figure_4.ipynb` differed slightly (e.g. neck-length variance
  given as both 0.194 and 0.2). Both are now built from the same cached fit
  via `simulation/ou_trajectory.covariance_from_amplitudes` and
  `models/event_mixture.physical_amplitudes`.
- **Inconsistent event-targeting convention between Fig. 5 and Fig. 6c.**
  `figure_3.ipynb`'s static covariance-surface model (Fig. 5) has "head-only"
  events move head size *and* neck width together; `figure_4.ipynb`'s
  dynamic knockout simulator (Fig. 6c/S5) instead groups neck width with
  neck length. These produce materially different plausible event-fraction
  ranges, and only the first reproduces the manuscript's reported
  e/e1/e2 ranges. Rather than force artificial agreement, `models/event_mixture.py`
  and `simulation/event_knockout.py` each keep their own source notebook's
  convention, documented at the point of difference.
- **A two-timescale model's prior indices don't line up with its own
  parameter layout** (`figure_2.ipynb`'s `_amplitude_and_noise_prior`: two
  noise parameters get an amplitude-range prior and vice versa). This
  produced the published Figs. 3/S3/S4, which already report those two
  alternative models as poorly identified/degenerate regardless -- ported
  as-is, with the mismatch documented in `models/relaxation.py`, rather than
  "corrected" into a different result than what was actually published.

## Porting mistakes caught by comparing rendered output against the paper

These were bugs introduced in this port, not in the original notebooks --
caught by rendering every script's output and comparing it panel-by-panel
against the actual manuscript figures, which is what surfaced them:

- **`pooling.pooled_values_with_spine_weights`/`pooled_triplets_with_spine_weights`
  were normalizing weights to sum to 1.** The original notebooks intentionally
  leave them unnormalized (each spine contributes total weight 1, split
  across its own measurements), so a weighted mean/std is unaffected, but an
  AIC log-likelihood sum (`distributions.py`) is extensive in the number of
  spines. Normalizing collapsed Fig. S1's ΔAIC values by ~150-700x (e.g.
  head size: +0.4 instead of the published +69). Fixed by leaving the
  weights unnormalized and making `weighted_mean_std` divide by `sum(w)`
  internally instead (correct either way).
- **Figs. 6c and S5 were plotting the ensemble *mean* current instead of
  individual trajectories.** These panels' entire point is to contrast a
  *broad* current distribution (when coordinated events are included)
  against a *narrow* one (head-only events only) -- averaging across the
  ensemble erases exactly that signal, and produced a smooth, misleadingly
  monotonic-looking curve instead. Fixed by plotting individual trajectories
  (for the "normal"/"only head" classes, matching the original) plus an
  adjacent KDE panel of the terminal-time distribution for all four classes
  (`plotting/density_panel.py`), matching the original notebook's own
  approach.
- **Fig. 3's schematic left panel used the actual (numerically messy,
  poorly-identified) two-timescale posterior means** instead of an
  illustrative sketch, producing a curve ordering that contradicted the
  panel's own conceptual point (a model *without* a persistent offset should
  visibly decay toward zero; because the real two-timescale-no-offset fit
  has a ~130-day second timescale, using its actual fitted values instead
  made it look nearly flat instead). Replaced with hand-picked illustrative
  decay constants, consistent with how Fig. 4 is already explicitly
  illustrative rather than fit to data.
- **Fig. 3's posterior-marginal histograms weren't axis-limited**, so they
  defaulted to the full nested-sampling prior range (e.g. 0-200 days for
  tau) instead of the converged posterior, making every histogram look like
  a thin spike near the axis edge. Fixed by limiting each histogram to
  mean +/- 6 sigma of its own posterior.
- **Fig. S3's corner plot was missing the with-offset model's tau-axis
  clipping** that the original applies (both timescales are only weakly
  constrained past ~20 days once the offset already carries the asymptote),
  making the with-offset corner far more zoomed-out than Fig. S4's.

## Validation performed during the port

Pair counts, ΔAIC values, and posterior parameters were checked against the
manuscript's own reported numbers (Tables S1-S2, Fig. S1, Fig. S2, Fig. 5)
as the primary numeric acceptance test; every figure script's rendered
output was then also visually compared panel-by-panel against the actual
manuscript figures, which is what caught the issues listed above -- neither
check alone would have been sufficient.
