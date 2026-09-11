#!/usr/bin/env python3
"""Is this MEG target predictable from this stimulus AT ALL? A linear ceiling on the same split.

Four hypotheses about the paired head's failure have now been refuted by measurement -- the
readout, the target's apparent heavy tail, calibration, and batch size -- and through all four the
head's correlation with its target stayed indistinguishable from zero. The conclusion drawn was
that the objective is wrong. That conclusion has a cheaper competitor which has never been tested:
**that nothing could predict this target from this stimulus on this split**, in which case no
objective would help and the data is what needs checking.

A ridge from the cochleagram context to the MEG separates them. It is not a better model of a
brain; it is a floor under the question. A linear map either finds stimulus-locked structure in
this target or it does not, and its answer does not depend on cortical dynamics, a lead field,
an output gain, or a batch size.

WHY A RIDGE IS THE RIGHT INSTRUMENT HERE. Speech-envelope tracking in auditory MEG is among the
most robust findings in the field, and this corpus's own v3 rebuild was justified by recovering
it: resampling onto the fitted clock line took three windows from p=0.171/0.463/0.902 to p=0.024.
So there is a specific, independently established effect that a linear map on a cochleagram ought
to see. If it does not see it here, the failure is upstream of every model tried.

KNOWN ANSWER, and it breaks the symmetry it tests: the same ridge, same features, same
regularisation, fitted against a **row-shuffled** target. Shuffling destroys the stimulus-target
correspondence and nothing else -- not the target's distribution, not the feature covariance, not
the fit's capacity to overfit. It MUST score ~0 held-out correlation. If the shuffled arm scores
like the intact one, the pipeline is leaking and neither number means anything.

THE SPLIT is the same one the trained arms were scored on: train from the head, evaluate on the
reserved tail, artefact rows excluded from training exactly as the trainer excludes them.

PREDICTED, before the fit: the ridge reaches a held-out correlation clearly above its shuffled
control -- small in absolute terms (single-digit percent is a normal effect size for this), but
unambiguously nonzero. If instead the ridge also reads ~0, then on this split this stimulus
representation carries nothing about this target, and the paired line needs its data audited
before any model is blamed for failing on it.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-stim", required=True)
    ap.add_argument("--paired-neural", required=True)
    ap.add_argument("--ctx", type=int, default=125)
    ap.add_argument("--lags", type=int, default=25, help="cochleagram frames kept, evenly spaced over ctx")
    ap.add_argument("--n-train", type=int, default=40_000)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--exclude", default="1619681:1627268")
    ap.add_argument("--alphas", default="1e2,1e3,1e4,1e5,1e6")
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="out/paired_ridge_ceiling.json")
    a = ap.parse_args()

    X = np.load(a.paired_stim, mmap_mode="r")
    Y = np.load(a.paired_neural, mmap_mode="r")
    sc = a.paired_neural.replace("meg_250hz", "meg_scale")
    megsc = np.load(sc) if Path(sc).exists() else None
    n = min(len(X), len(Y)) - 2
    lo = int(n * (1.0 - a.holdout))
    lag_idx = np.linspace(0, a.ctx - 1, a.lags).astype(int)

    def target(j):
        y = np.ascontiguousarray(Y[np.asarray(j)]).astype(np.float64)
        return np.clip((y - megsc[0]) / megsc[1], -6, 6) if megsc is not None else y

    def feats(j):
        return np.stack([np.asarray(X[q - a.ctx:q])[lag_idx].ravel() for q in j]).astype(np.float64)

    # ---- training pool: head of the corpus, artefact rows removed as the trainer removes them
    excl = [tuple(int(v) for v in r.split(":")) for r in a.exclude.split(",") if r.strip()]
    pool = np.arange(a.ctx, lo)
    for x0, x1 in excl:
        pool = pool[~((pool >= x0) & (pool < x1))]
    rng = np.random.default_rng(7)
    tr = np.sort(rng.choice(pool, size=min(a.n_train, len(pool)), replace=False))

    # ---- the SAME fixed evaluation draw the trained arms were scored on
    g = np.random.default_rng(20260911)
    ev = np.sort(np.concatenate([g.integers(lo, n, size=a.batch) for _ in range(a.batches)]))

    print(f"train {len(tr):,} rows from [{a.ctx:,}, {lo:,}) minus {len(excl)} excluded range(s)")
    print(f"eval  {len(ev):,} rows from the reserved tail [{lo:,}, {n:,})  (seed 20260911)")
    print(f"features: {a.lags} cochleagram frames x {X.shape[-1]} bands = {a.lags*X.shape[-1]:,}\n")

    Ftr, Ttr = feats(tr), target(tr)
    Fev, Tev = feats(ev), target(ev)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    Ftr = (Ftr - mu) / sd; Fev = (Fev - mu) / sd
    ym = Ttr.mean(0)

    def corr(P, T):
        p, t = (P - P.mean(0)).ravel(), (T - T.mean(0)).ravel()
        return float(p @ t / max(np.linalg.norm(p) * np.linalg.norm(t), 1e-30))

    def fit_eval(Tfit, tag):
        C = Ftr.T @ Ftr
        R = Ftr.T @ (Tfit - Tfit.mean(0))
        best = None
        for al in (float(v) for v in a.alphas.split(",")):
            W = np.linalg.solve(C + al * np.eye(C.shape[0]), R)
            P = Fev @ W + Tfit.mean(0)
            c = corr(P, Tev)
            mse = float(((P - Tev) ** 2).mean())
            sk = 1.0 - mse / float((Tev ** 2).mean())
            print(f"  {tag:9s} alpha {al:8.0e}  held-out corr {c:+.4f}  skill vs zero {sk:+8.3f}")
            if best is None or c > best["correlation"]:
                best = dict(alpha=al, correlation=c, mse=mse, skill_vs_zero=sk)
        return best

    print("KNOWN ANSWER FIRST -- the same ridge against a ROW-SHUFFLED target.")
    print("Shuffling destroys the stimulus-target correspondence and nothing else; it MUST read ~0.")
    sh = Ttr[np.random.default_rng(11).permutation(len(Ttr))]
    ctrl = fit_eval(sh, "shuffled")
    se = 1.0 / np.sqrt(Tev.size)
    ok = abs(ctrl["correlation"]) < 5 * se
    print(f"  -> shuffled best corr {ctrl['correlation']:+.4f}, standard error {se:.4f}  "
          f"{'PASS' if ok else 'FAIL -- the pipeline leaks; no number below means anything'}")
    if not ok:
        sys.exit(1)

    print("\nTHE RIDGE, on correctly paired data:")
    real = fit_eval(Ttr, "intact")
    print(f"\n  intact best corr {real['correlation']:+.4f} = {real['correlation']/se:+.1f} sd  "
          f"against shuffled {ctrl['correlation']:+.4f} = {ctrl['correlation']/se:+.1f} sd")

    beats = real["correlation"] > ctrl["correlation"] + 5 * se
    print("\n  VERDICT: " + (
        f"a LINEAR map finds stimulus-locked structure this target does carry.\n"
        f"  The trained head's +0.0022/+0.0006 is therefore not a property of the data, and the\n"
        f"  objective -- not the data -- is what to change."
        if beats else
        f"a linear map finds NOTHING either, at {real['correlation']:+.4f} against a shuffled\n"
        f"  control of {ctrl['correlation']:+.4f}. On this split this stimulus representation carries\n"
        f"  nothing about this target. The paired line needs its DATA audited before any model is\n"
        f"  blamed for failing on it, and changing the objective would not have helped."))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        dict(train_rows=len(tr), eval_rows=len(ev), lags=a.lags, excluded=a.exclude,
             standard_error=float(se), intact=real, shuffled_control=ctrl,
             known_answer_pass=bool(ok), intact_beats_control=bool(beats)), indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
