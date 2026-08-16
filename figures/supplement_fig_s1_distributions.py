"""Figure S1: raw values are log-normal, increments are Gaussian."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, lognorm, norm

from spine_volatility import data
from spine_volatility.distributions import increment_fit, raw_value_fit
from spine_volatility.paths import FIGURES_DIR, ensure_output_dirs
from spine_volatility.plotting.style import apply_base_style

FEATURES = [("Head size", "HS", r"$\mu m^2$"), ("Neck length", "NL", r"$\mu m$"), ("Neck width", "NW", r"$\mu m$")]


def main():
    ensure_output_dirs()
    apply_base_style()
    out_dir = FIGURES_DIR / "supplement"
    out_dir.mkdir(parents=True, exist_ok=True)

    long = data.load_long_term()
    n_timepoints = len(data.DELTA_LONG_DAYS)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, (name, attr, unit) in enumerate(FEATURES):
        feature_data = getattr(long, attr)

        raw = raw_value_fit(feature_data)
        ax = axes[0, col]
        x = np.linspace(raw["values"].min(), raw["values"].max(), 400)
        kde = gaussian_kde(raw["values"], weights=raw["weights"])
        ax.plot(x, kde(x), color="black", label="Data (KDE)")
        ax.plot(x, lognorm.pdf(x, raw["lognorm_shape"], loc=0, scale=raw["lognorm_scale"]), color="red", label="Log-normal")
        ax.plot(x, norm.pdf(x, raw["gauss_mean"], raw["gauss_std"]), color="tab:blue", ls="--", label="Gaussian")
        ax.set_title(f"{name}\n" + r"$\Delta$AIC = " + f"{raw['delta_aic']:+.0f}")
        ax.set_xlabel(f"{name} ({unit})")
        ax.set_ylabel("Density" if col == 0 else "")
        if col == 0:
            ax.legend(frameon=False, fontsize=8)

        inc = increment_fit(feature_data, n_timepoints)
        ax = axes[1, col]
        zz = np.linspace(inc["z"].min(), inc["z"].max(), 400)
        kde_z = gaussian_kde(inc["z"], weights=inc["weights"])
        ax.plot(zz, kde_z(zz), color="black", label="Data (KDE)")
        ax.plot(zz, norm.pdf(zz, 0, 1), color="red", label="N(0,1)")
        ax.set_title(r"$\Delta$AIC = " + f"{inc['delta_aic']:+.0f}")
        ax.set_xlabel(f"Standardized {name} increment")
        ax.set_ylabel("Density" if col == 0 else "")

        print(f"{name}: raw dAIC={raw['delta_aic']:+.1f} (lognormal favored if >0), "
              f"increment dAIC={inc['delta_aic']:+.1f} (Gaussian favored if >0)")

    fig.suptitle("Fig. S1: morphological features are log-normal, increments Gaussian", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "fig_s1_distributions.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_s1_distributions.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure S1 to {out_dir}")


if __name__ == "__main__":
    main()
