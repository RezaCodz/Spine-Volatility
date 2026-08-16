"""Tables S1-S2: number of data pairs used in the auto-/cross-correlation calculations."""

from spine_volatility import data
from spine_volatility.correlations import auto_pair_counts, cross_pair_counts


def main():
    long = data.load_long_term()
    short = data.load_short_term()

    print("Table S1: auto-correlation pair counts")
    print(f"{'':8s} {'30m':>6s} {'60m':>6s} {'120m':>6s} {'3.5d':>6s} {'7d':>6s} {'10.5d':>6s} {'14d':>6s}")
    for name, s, l in [("HS", short.HS, long.HS), ("NL", short.NL, long.NL), ("NW", short.NW, long.NW)]:
        counts_short = auto_pair_counts(s, len(data.DELTA_SHORT_MINUTES))
        counts_long = auto_pair_counts(l, len(data.DELTA_LONG_DAYS))
        row = counts_short + counts_long
        print(f"{name:8s} " + " ".join(f"{c:6d}" for c in row))

    print("\nTable S2: cross-correlation pair counts")
    print(f"{'':8s} {'0':>6s} {'30m':>6s} {'60m':>6s} {'120m':>6s} {'3.5d':>6s} {'7d':>6s} {'10.5d':>6s} {'14d':>6s}")
    pairs = [
        ("NL-HS", short.NL, short.HS, long.NL, long.HS),
        ("NW-HS", short.NW, short.HS, long.NW, long.HS),
        ("NW-NL", short.NW, short.NL, long.NW, long.NL),
    ]
    for name, d1s, d2s, d1l, d2l in pairs:
        counts_short = cross_pair_counts(d1s, d2s, len(data.DELTA_SHORT_MINUTES))
        counts_long = cross_pair_counts(d1l, d2l, len(data.DELTA_LONG_DAYS))
        row = counts_short + counts_long[1:]  # one shared "0" column
        print(f"{name:8s} " + " ".join(f"{c:6d}" for c in row))


if __name__ == "__main__":
    main()
