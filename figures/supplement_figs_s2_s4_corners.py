"""Figures S2-S4: posterior corner plots for the one- and two-timescale models."""

import matplotlib.pyplot as plt
from dynesty import plotting as dyplot

from spine_volatility import data
from spine_volatility.models import relaxation
from spine_volatility.paths import FIGURES_DIR, INFERENCE_DIR, ensure_output_dirs
from spine_volatility.plotting.style import apply_base_style

MM_TO_INCH = 1.0 / 25.4
FIGSIZE_MM = (183.0, 170.0)


def main():
    ensure_output_dirs()
    apply_base_style()
    out_dir = FIGURES_DIR / "supplement"
    out_dir.mkdir(parents=True, exist_ok=True)

    long = data.load_long_term()
    short = data.load_short_term()
    rel_data = relaxation.build_relaxation_data(
        long.HS, long.NL, long.NW, short.HS, short.NL, short.NW,
        data.DELTA_LONG_DAYS, data.DELTA_SHORT_DAYS,
    )

    results_one_exp = relaxation.fit_model("one_exp_offset", rel_data, INFERENCE_DIR)
    results_two_offset = relaxation.fit_model("two_exp_with_offset", rel_data, INFERENCE_DIR)
    results_two_no_offset = relaxation.fit_model("two_exp_no_offset", rel_data, INFERENCE_DIR)

    one_exp_dims = [0, 1, 2, 3, 4, 5, 6]  # tau + all six amplitudes
    two_tau_dims = [0, 1, 2, 3, 13, 14]  # both taus, a_hs pair, a_nl_hs pair
    # dynesty's own default span (fraction of weighted samples) when none is given.
    default_span = 0.999999426697
    # The with-offset model's two taus have a long, uninformative tail (the
    # fixed offset already carries the asymptote, so the data don't pin them
    # down past ~20 days); clip both to 0-20 days for readability, matching
    # the two-timescale-no-offset model's naturally tighter posterior.
    span_two_exp_with_offset = [(0, 20), (0, 20)] + [default_span] * 4

    corner_models = [
        (results_one_exp, relaxation.ONE_EXP_LABELS, one_exp_dims, "Fig. S2: one-exponential + offset", "fig_s2_corner_one_exp", None),
        (results_two_offset, relaxation.TWO_EXP_LABELS, two_tau_dims, "Fig. S3: two-timescale + offset", "fig_s3_corner_two_exp_offset", span_two_exp_with_offset),
        (results_two_no_offset, relaxation.TWO_EXP_LABELS, two_tau_dims, "Fig. S4: two-timescale, no offset", "fig_s4_corner_two_exp_no_offset", None),
    ]

    with plt.rc_context({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                          "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42}):
        for results, all_labels, dims, title, basename, span in corner_models:
            selected_labels = [all_labels[i] for i in dims]
            fig, axes = dyplot.cornerplot(results, dims=dims, span=span, labels=selected_labels, show_titles=True,
                                           title_kwargs={"fontsize": 7}, label_kwargs={"fontsize": 7})
            fig.set_size_inches(FIGSIZE_MM[0] * MM_TO_INCH, FIGSIZE_MM[1] * MM_TO_INCH)
            fig.suptitle(title, y=1.02, fontsize=7)
            fig.savefig(out_dir / f"{basename}.pdf", bbox_inches="tight")
            fig.savefig(out_dir / f"{basename}.svg", bbox_inches="tight")
            plt.close(fig)

    print(f"Saved Figures S2-S4 to {out_dir}")


if __name__ == "__main__":
    main()
