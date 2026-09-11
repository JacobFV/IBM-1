#!/usr/bin/env python3
"""Where in the LibriBrain corpus does the heavy tail actually live?

The batch-size hypothesis rests on a measured fact: the paired MEG target's per-batch mean
square has median 4.55e-05, mean 1.24e-02 and max 1.60e+01 -- five orders of magnitude -- so
with `--batch 4` the gradient is dominated by the rare enormous batch. The proposed fix was a
larger batch, so the median stops being drowned.

Then the held-out tail (the last 10%) turned out to have a per-batch spread of **2x**: median
3.50e-05, max 7.74e-05 over 512 rows. No tail at all. A property present in the whole array
and absent from its last tenth is not a property of the signal; it is **localised**.

Which changes what the hypothesis is. "The target is heavy-tailed" and "some stretches of this
recording are corrupt" predict the same per-batch histogram and call for completely different
fixes -- a larger batch for the first, excluding or repairing those stretches for the second.
The histogram cannot tell them apart because it throws position away.

THE FIRST INSTRUMENT HERE FAILED ITS OWN KNOWN ANSWER, and the failure was mine, not the
corpus's. It binned by position and compared against bins drawn by SHUFFLING, expecting the
shuffled arm to show FEWER bins above 10x the median. It showed far more: 66 of 200 against 2.
That is not a broken corpus, it is a mis-specified control -- a shuffled 400-row sample drawn
from 3.5M rows is much LIKELIER to contain one of a few thousand extreme rows than a 400-row
contiguous window is, so shuffling SPREADS rare extremes across many bins instead of
concentrating them. The direction of the prediction was simply wrong, and the count of high
bins is the wrong statistic for clustering. Recorded rather than quietly re-specified, and the
conclusion was NOT claimed on it.

WHAT THIS DOES INSTEAD. Compute the mean square of EVERY row, not of sampled bins. Call a row
extreme if it exceeds `--thresh` times the global median. Then ask one question of the extreme
rows' POSITIONS: how many distinct windows of `--window` rows do they occupy? Clustered
positions occupy few windows; uniform positions occupy nearly one each.

KNOWN ANSWER, and this one does break the symmetry it tests: draw the SAME NUMBER of positions
uniformly at random over the same range and compute the same occupancy. That control shares the
count, the window size and the statistic, and differs only in whether position carries
structure. It must come out near the uniform expectation -- and it CAN fail, which is what the
previous version could not do.
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-neural", required=True)
    ap.add_argument("--thresh", type=float, default=100.0,
                    help="a row is extreme if its mean square exceeds thresh x the global median")
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--out", default="out/meg_tail_position.json")
    a = ap.parse_args()

    Y = np.load(a.paired_neural, mmap_mode="r")
    n = len(Y)
    sc = a.paired_neural.replace("meg_250hz", "meg_scale")
    megsc = np.load(sc) if Path(sc).exists() else None
    print(f"{a.paired_neural}: {n:,} rows x {Y.shape[1]} sensors"
          f"   meg_scale {'applied' if megsc is not None else 'ABSENT -- raw'}")

    # every row, streamed
    ms = np.empty(n, np.float32)
    for s in range(0, n, a.chunk):
        y = np.ascontiguousarray(Y[s:s + a.chunk]).astype(np.float32)
        if megsc is not None:
            y = np.clip((y - megsc[0]) / megsc[1], -6, 6)
        ms[s:s + len(y)] = (y ** 2).mean(1)
    med = float(np.median(ms))
    print(f"\nper-ROW mean square over all {n:,} rows")
    print(f"  median {med:.3e}   mean {ms.mean():.3e}   max {ms.max():.3e}   "
          f"max/median {ms.max()/med:,.0f}x")
    for q in (99.0, 99.9, 99.99):
        print(f"  p{q:<6} {np.percentile(ms, q):.3e}  ({np.percentile(ms, q)/med:9.1f}x median)")

    ext = np.flatnonzero(ms > a.thresh * med)
    print(f"\nextreme rows (> {a.thresh:g}x median): {len(ext):,}  "
          f"({len(ext)/n:.4%} of the corpus)")
    if len(ext) == 0:
        print("  none; nothing to localise")
        return

    occ = len(np.unique(ext // a.window))
    rng = np.random.default_rng(0)
    ctrl = np.array([len(np.unique(rng.integers(0, n, size=len(ext)) // a.window))
                     for _ in range(20)])
    print(f"\nKNOWN ANSWER: {len(ext):,} positions drawn UNIFORMLY, same count, same "
          f"{a.window}-row window, same statistic.")
    print(f"  uniform occupancy: {ctrl.mean():,.0f} +/- {ctrl.std():.0f} windows  "
          f"(of {int(np.ceil(n/a.window)):,} available)")
    lo = ctrl.mean() - 5 * max(ctrl.std(), 1.0)
    print(f"  observed occupancy: {occ:,} windows")
    clustered = occ < lo
    print(f"  {'CLUSTERED -- far fewer windows than uniform' if clustered else
             'NOT distinguishable from uniform'}"
          f"   (bar: below {lo:,.0f})")

    # where, concretely
    runs, start, prev = [], ext[0], ext[0]
    for p in ext[1:]:
        if p - prev > a.window:
            runs.append((start, prev)); start = p
        prev = p
    runs.append((start, prev))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    span = sum(e - s + 1 for s, e in runs)
    print(f"\n  the extreme rows fall in {len(runs)} run(s) spanning {span:,} rows "
          f"({span/n:.3%} of the corpus). Largest:")
    for s, e in runs[:8]:
        print(f"    rows {s:>10,} - {e:>10,}   {e-s+1:>9,} rows   "
              f"peak {ms[s:e+1].max():.3e} ({ms[s:e+1].max()/med:,.0f}x median)")

    print(f"\n  VERDICT: " + (
        f"the extremes are LOCALISED to {span/n:.3%} of the corpus by position. "
        f"'The target is heavy-tailed'\n  and 'these stretches are corrupt' predict the same "
        f"histogram and call for different fixes;\n  this says it is the second."
        if clustered else
        "the extremes are spread through the corpus, which is what the heavy-tailed reading\n"
        "  assumed. A larger batch remains the right lever."))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"rows": int(n), "median_ms": med, "mean_ms": float(ms.mean()), "max_ms": float(ms.max()),
         "thresh_x_median": a.thresh, "n_extreme": int(len(ext)),
         "window": a.window, "observed_occupancy": int(occ),
         "uniform_occupancy_mean": float(ctrl.mean()), "uniform_occupancy_sd": float(ctrl.std()),
         "clustered": bool(clustered), "span_rows": int(span),
         "runs": [[int(s), int(e)] for s, e in runs[:50]]}, indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
