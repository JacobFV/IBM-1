#!/usr/bin/env python3
"""Does the weak per-sample MEG signal AGGREGATE into something retrieval can use?

Seven hypotheses about the paired head are refuted and the objective is what is left. The
proposed replacement is contrastive/retrieval, on the strength of `VisualContrastiveLoop`'s
measured comparison: on THINGS-EEG2, same encoder and split, waveform regression peaked at +0.011
then went negative while contrastive retrieval reached 43x chance. Before building that for the
MEG term, it is worth knowing whether it has anything to work with here.

The reason to doubt it: a per-sample correlation of **+0.0466** is tiny. The reason to expect it
anyway: retrieval does not need per-sample accuracy, it needs one stimulus window to be more like
its own MEG window than like other windows -- and a weak per-sample correlation aggregated over N
timepoints grows as roughly sqrt(N) if the errors are independent. **Whether it actually does is a
measurement, not an argument**, because MEG noise is heavily autocorrelated and correlated errors
do not aggregate at all.

So: take the ridge's predicted MEG on held-out rows, cut both prediction and truth into windows of
W contiguous timepoints, and ask whether each predicted window retrieves its own true window from
a pool. Sweep W over 1, 5, 25, 125 (0.5 s at 250 Hz). The shape of top-1 against W is the answer.

KNOWN ANSWERS, both printed before any real number:
  1. **Chance.** A pool of P windows retrieves at 1/P. Measured by scoring with the pairings
     SHUFFLED, which destroys the correspondence and nothing else -- it must land at 1/P.
  2. **The ceiling.** Retrieving the TRUE window against itself must be 100%, or the retrieval
     metric is broken independently of any model. This one cannot fail for an interesting reason,
     which is exactly why it is paired with the shuffled arm above -- together they bracket the
     scale, and the shuffled arm CAN fail.

Pools are averaged over many draws because single-pool retrieval has sd ~2.8% (ledger row 9) and
this file's rule is to average pools before comparing, never to quote one.

PREDICTED, before the run: top-1 rises with window length, clearly above chance by W=25 and
well above by W=125, because that is what aggregating a weak stimulus-locked signal looks like.
If it stays at chance at every window, the signal does not aggregate -- correlated noise -- and a
contrastive objective has no more to work with here than regression did, which would be a strong
argument against building one for this corpus and would need saying loudly.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-stim", required=True)
    ap.add_argument("--paired-neural", required=True)
    ap.add_argument("--ctx", type=int, default=125)
    ap.add_argument("--lags", type=int, default=25)
    ap.add_argument("--n-train", type=int, default=40_000)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--exclude", default="1619681:1627268")
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--windows", default="1,5,25,125")
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=25_000, help="contiguous held-out rows scored")
    ap.add_argument("--out", default="out/paired_retrieval_aggregation.json")
    a = ap.parse_args()

    X = np.load(a.paired_stim, mmap_mode="r")
    Y = np.load(a.paired_neural, mmap_mode="r")
    sc = a.paired_neural.replace("meg_250hz", "meg_scale")
    megsc = np.load(sc) if Path(sc).exists() else None
    n = min(len(X), len(Y)) - 2
    lo = int(n * (1.0 - a.holdout))
    lag_idx = np.linspace(0, a.ctx - 1, a.lags).astype(int)

    def target(j):
        y = np.ascontiguousarray(Y[np.asarray(j)]).astype(np.float32)
        out = np.clip((y - megsc[0]) / megsc[1], -6, 6) if megsc is not None else y
        return np.ascontiguousarray(out, dtype=np.float64)

    def feats(j):
        return np.stack([np.asarray(X[q - a.ctx:q])[lag_idx].ravel() for q in j]).astype(np.float64)

    excl = [tuple(int(v) for v in r.split(":")) for r in a.exclude.split(",") if r.strip()]
    pool_idx = np.arange(a.ctx, lo)
    for x0, x1 in excl:
        pool_idx = pool_idx[~((pool_idx >= x0) & (pool_idx < x1))]
    tr = np.sort(np.random.default_rng(7).choice(pool_idx, size=min(a.n_train, len(pool_idx)),
                                                 replace=False))
    # CONTIGUOUS held-out rows: windows must be real stretches of recording, not scattered draws.
    ev = np.arange(lo + a.ctx, min(lo + a.ctx + a.n_eval, n))
    print(f"train {len(tr):,} rows | eval {len(ev):,} CONTIGUOUS rows from {ev[0]:,}\n")

    Ftr, Ttr = feats(tr), target(tr)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    W = np.linalg.solve(((Ftr - mu) / sd).T @ ((Ftr - mu) / sd)
                        + a.alpha * np.eye(Ftr.shape[1]),
                        ((Ftr - mu) / sd).T @ (Ttr - Ttr.mean(0)))
    Pev = ((feats(ev) - mu) / sd) @ W + Ttr.mean(0)
    Tev = target(ev)
    r = float(((Pev - Pev.mean(0)).ravel() @ (Tev - Tev.mean(0)).ravel())
              / (np.linalg.norm(Pev - Pev.mean(0)) * np.linalg.norm(Tev - Tev.mean(0))))
    print(f"the ridge's per-sample held-out correlation on these rows: {r:+.4f}\n")

    rng = np.random.default_rng(20260911)

    def retrieve(Wn, shuffled=False, self_test=False):
        """Top-1 over `draws` pools of `pool` windows each, averaged."""
        nwin = len(ev) // Wn
        if nwin < a.pool + 1:
            return None
        Pw = Pev[:nwin * Wn].reshape(nwin, Wn * Pev.shape[1])
        Tw = (Pev if self_test else Tev)[:nwin * Wn].reshape(nwin, Wn * Tev.shape[1])
        Pw = Pw - Pw.mean(1, keepdims=True); Tw = Tw - Tw.mean(1, keepdims=True)
        Pw /= np.linalg.norm(Pw, axis=1, keepdims=True) + 1e-30
        Tw /= np.linalg.norm(Tw, axis=1, keepdims=True) + 1e-30
        hits = 0
        for _ in range(a.draws):
            idx = rng.choice(nwin, size=a.pool, replace=False)
            q = Pw[idx]
            bank = Tw[rng.permutation(idx)] if shuffled else Tw[idx]
            hits += int((np.argmax(q @ bank.T, axis=1) == np.arange(a.pool)).sum())
        return hits / (a.draws * a.pool)

    chance = 1.0 / a.pool
    Ws = [int(v) for v in a.windows.split(",")]
    print(f"KNOWN ANSWERS (pool of {a.pool}, {a.draws} draws averaged; single-pool sd ~2.8%)")
    ceil = retrieve(Ws[0], self_test=True)
    print(f"  ceiling -- retrieving a window against ITSELF: {ceil:.1%}  "
          f"{'PASS' if ceil > 0.999 else 'FAIL -- the retrieval metric is broken'}")
    shuf = retrieve(Ws[-1], shuffled=True)
    ok = abs(shuf - chance) < 4 * np.sqrt(chance * (1 - chance) / (a.draws * a.pool))
    print(f"  chance  -- pairings SHUFFLED at W={Ws[-1]}: {shuf:.1%} against 1/{a.pool} = "
          f"{chance:.1%}  {'PASS' if ok else 'FAIL -- the pairing leaks'}")
    if ceil <= 0.999 or not ok:
        sys.exit("a known answer FAILED; no retrieval number below is interpretable")

    print(f"\nTOP-1 RETRIEVAL against window length (chance {chance:.1%}):")
    rows = {}
    for Wn in Ws:
        t = retrieve(Wn)
        if t is None:
            print(f"  W={Wn:4d}  too few windows in the held-out stretch, skipped"); continue
        rows[Wn] = t
        print(f"  W={Wn:4d} ({Wn/250:6.3f} s)  top-1 {t:6.1%}   {t/chance:5.2f}x chance")

    best = max(rows, key=rows.get)
    aggregates = rows[best] > chance + 4 * np.sqrt(chance * (1 - chance) / (a.draws * a.pool))
    print("\n  VERDICT: " + (
        f"the signal DOES aggregate -- {rows[best]:.1%} at W={best} against {chance:.1%} chance,\n"
        f"  {rows[best]/chance:.2f}x. A per-sample correlation of {r:+.4f} is too small to regress\n"
        f"  but large enough to discriminate once pooled over {best/250:.3f} s. A contrastive\n"
        f"  objective has real material here, and that is now measured rather than argued by\n"
        f"  analogy with THINGS-EEG2." if aggregates else
        f"the signal does NOT aggregate -- {rows[best]:.1%} at best against {chance:.1%} chance.\n"
        f"  The per-sample correlation is real but its errors are correlated, so pooling buys\n"
        f"  nothing. A contrastive objective would have no more to work with here than regression\n"
        f"  did, and building one for this corpus on the THINGS-EEG2 analogy would be a mistake."))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(
        per_sample_correlation=r, pool=a.pool, draws=a.draws, chance=chance,
        ceiling=ceil, shuffled=shuf, top1_by_window=rows, aggregates=bool(aggregates)),
        indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
