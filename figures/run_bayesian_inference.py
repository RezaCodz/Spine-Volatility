"""Fit all four correlation-decay models by nested sampling (Methods; Figs. 3, 5, S2-S4).

This is the expensive step: four dynesty runs (one at ndim=9, one at
ndim=10, two at ndim=17), each with nlive=1000. Every other figure script
loads the cached results this produces from results/inference/*.pkl instead
of re-fitting.

Usage: python figures/run_bayesian_inference.py [--force]
"""

import argparse
import time

import numpy as np

from spine_volatility import data
from spine_volatility.models import relaxation
from spine_volatility.paths import INFERENCE_DIR, ensure_output_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-fit even if a cached result exists.")
    args = parser.parse_args()

    ensure_output_dirs()
    long = data.load_long_term()
    short = data.load_short_term()

    rel_data = relaxation.build_relaxation_data(
        long.HS, long.NL, long.NW, short.HS, short.NL, short.NW,
        data.DELTA_LONG_DAYS, data.DELTA_SHORT_DAYS,
    )

    for name in relaxation.MODEL_SPECS:
        cache_path = INFERENCE_DIR / f"{name}.pkl"
        if cache_path.exists() and not args.force:
            print(f"[{name}] using cached fit at {cache_path}")
            continue
        print(f"[{name}] running nested sampling (ndim={relaxation.MODEL_SPECS[name]['ndim']})...")
        t0 = time.time()
        results = relaxation.fit_model(name, rel_data, INFERENCE_DIR, force=args.force)
        print(f"[{name}] done in {time.time() - t0:.0f}s, logZ = {results.logz[-1]:.2f} +/- {results.logzerr[-1]:.2f}")

    print("\nOne-exponential + offset posterior summary:")
    results = relaxation.fit_model("one_exp_offset", rel_data, INFERENCE_DIR)
    mean = relaxation.posterior_mean(results)
    lo, hi = relaxation.posterior_quantile(results, [0.025, 0.975]).T
    for label, m, l, h in zip(relaxation.ONE_EXP_LABELS, mean, lo, hi):
        print(f"  {label:>8s}: {m:.4g}  (95% CI [{l:.4g}, {h:.4g}])")
    print("Paper (Fig. S2): tau=3.62 [+1.24/-0.97], a_hs=0.23, a_nl=0.20, a_nw=0.38, "
          "a_nl_nw=-0.28, a_nl_hs=-0.09, a_nw_hs=0.07")


if __name__ == "__main__":
    main()
