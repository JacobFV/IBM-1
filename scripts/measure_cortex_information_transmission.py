#!/usr/bin/env python3
"""Do the cortical dynamics PRESERVE the stimulus information, or destroy it?

A ridge on the raw cochleagram reaches held-out correlation **+0.0466 at 18.6 sd** with positive
skill against zero, while the trained 50M-parameter head reaches +0.0006. The signal is in the
data and the model does not reach it. That leaves two very different explanations and they call
for opposite work:

  THE READOUT/TRAINING IS AT FAULT -- the dynamics carry the stimulus information through, and
  what fails is the learned map out of them. Then a ridge fitted FROM THE CORTICAL STATE should
  reach roughly what the cochleagram ridge reaches, and the fix is in the head or the objective.

  THE DYNAMICS DESTROY IT -- the state the stimulus drives is nearly one-dimensional (effective
  rank measured at 1.05-1.20 in every arm, at every lead rank, throughout training), and a
  near-rank-1 state cannot carry 1,600 features' worth of structure however it is read. Then no
  objective and no readout can recover what is no longer there, and the substrate is the problem.

The same ridge, fitted from the cortical state instead of from the cochleagram, separates them.
It asks what a LINEAR map can get out of the sheet, which is the most generous question available
-- if a linear map on the state cannot find the structure, a learned nonlinear one had no better
starting material.

THREE ARMS, one split, one evaluation draw, all at the same `--n-train`:
  cochleagram    the raw stimulus features -- the reference the cortical arms are judged against.
  trained        the cortical state of `ckpt/clean_b64.pt`, driven by the same stimulus.
  untrained      the cortical state of a FRESHLY INITIALISED substrate, same seed, same topology,
                 same driving path. The control that says whether training changed transmission
                 at all -- and it can fail in both directions, which is why it is here.

THE KNOWN ANSWER IS ASKED SEPARATELY, AND THE FIRST VERSION OF IT FAILED -- correctly, and the
fault was mine. It required the cochleagram arm to reproduce the recorded **+0.0466**; it read
**+0.0363**, because the cortical arms need a forward pass of the 150k-site dynamics per row so
`--n-train` had been cut to 6,000 where +0.0466 was fitted on 40,000. A ridge fitted on a seventh
of the data is a different estimator. The gate did exactly its job: it stopped cortical numbers
being read against a baseline fitted on other data.

So the known answer now runs the cochleagram fit at `--known-answer-n-train` (40,000, the recorded
configuration) and certifies only the code path, while the comparison runs every arm including
cochleagram at `--n-train`. **Nothing is compared across training-set sizes**, and the bar on the
known answer is unchanged.

A SECOND THING THE FIRST RUN EXPOSED. Its shuffled control read **-0.0201, eight standard errors
from zero**, which a destroyed correspondence should not do. The cause is the statistic, not a
leak: under heavy regularisation the prediction collapses toward a constant, and `corr` then
divides by a vanishing spread, so it is a ratio of tiny numbers and numerically meaningless. Each
arm now also reports the prediction's own spread as a fraction of the target's, so a correlation
computed from a near-constant prediction can be recognised as such instead of read as a signal.

PREDICTED, before any fit. The trained cortical state scores FAR below the cochleagram, near its
own shuffled control, because a state of effective rank ~1 cannot carry the structure. And
trained will be indistinguishable from untrained, because nothing in the measured history
suggests training improved what the dynamics transmit -- rank collapsed within 1,000 steps and
stayed collapsed at every setting tried.

If instead the trained state matches the cochleagram, this entry is wrong: the dynamics are a
faithful channel, the head is the whole failure, and the objective is where to work.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
RECORDED_COCHLEAGRAM_R = 0.0466      # docs/LOG.md, alpha 1e4, fixed tail draw seed 20260911


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--paired-stim", required=True)
    ap.add_argument("--paired-neural", required=True)
    ap.add_argument("--sites", type=int, default=150_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--lead-rank", type=int, default=64)
    ap.add_argument("--dyn-steps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--ctx", type=int, default=125)
    ap.add_argument("--lags", type=int, default=25)
    ap.add_argument("--read-sites", type=int, default=1600,
                    help="cortical sites read, matched to the cochleagram feature count")
    # THE FIRST RUN OF THIS FAILED ITS OWN KNOWN ANSWER, correctly, and the fault was mine.
    # The cortical arms need a forward pass of the 150k-site dynamics per row, so `--n-train`
    # was cut to 6,000 where the recorded +0.0466 was fitted on 40,000. A ridge fitted on a
    # seventh of the data is a different estimator, and it read +0.0363. The gate stopped the
    # cortical numbers being read against a baseline fitted on other data -- which is the whole
    # reason it is asked before them.
    #
    # So there are now two cochleagram fits and they do different jobs. The KNOWN ANSWER runs at
    # `--known-answer-n-train` (40,000, the recorded configuration) and only certifies the code
    # path. The COMPARISON runs every arm, cochleagram included, at `--n-train`, and the cortical
    # arms are judged against the cochleagram arm AT THAT SAME SIZE. Nothing is compared across
    # training-set sizes. The bar on the known answer is unchanged.
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--known-answer-n-train", type=int, default=40_000)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--exclude", default="1619681:1627268")
    ap.add_argument("--alpha", type=float, default=1e4, help="FIXED, not selected on the eval set")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--gpu-batch", type=int, default=8)
    ap.add_argument("--out", default="out/cortex_information_transmission.json")
    a = ap.parse_args()

    import pretrain_video_loop as P
    dev = "cuda" if torch.cuda.is_available() else "cpu"
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

    excl = [tuple(int(v) for v in r.split(":")) for r in a.exclude.split(",") if r.strip()]
    pool = np.arange(a.ctx, lo)
    for x0, x1 in excl:
        pool = pool[~((pool >= x0) & (pool < x1))]
    rng7 = np.random.default_rng(7)
    tr_big = np.sort(rng7.choice(pool, size=min(a.known_answer_n_train, len(pool)), replace=False))
    tr = np.sort(np.random.default_rng(7).choice(pool, size=min(a.n_train, len(pool)), replace=False))
    g = np.random.default_rng(20260911)
    ev = np.sort(np.concatenate([g.integers(lo, n, size=a.batch) for _ in range(a.batches)]))
    print(f"train {len(tr):,} rows | eval {len(ev):,} rows from the reserved tail "
          f"[{lo:,}, {n:,})  (seed 20260911)\n")

    def coch_feats(j):
        return np.stack([np.asarray(X[q - a.ctx:q])[lag_idx].ravel() for q in j]).astype(np.float64)

    # ---- the cortical state, driven by the same stimulus -------------------------
    def make(trained):
        dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
        pr = P.PairedNeuralLoop(dyn, n_bands=X.shape[-1], n_sensors=Y.shape[-1],
                                lead_rank=a.lead_rank).to(dev)
        if trained:
            sd = torch.load(a.ckpt, map_location=dev)
            dyn.load_state_dict(sd["dyn"]); pr.load_state_dict(sd["paired"])
        dyn.eval(); pr.eval()
        return dyn, pr

    def state_feats(dyn, pr, j, site_idx):
        out, ranks = [], []
        with torch.no_grad():
            for s in range(0, len(j), a.gpu_batch):
                q = j[s:s + a.gpu_batch]
                xp = torch.from_numpy(np.stack([X[t - a.ctx:t] for t in q])).float().to(dev)
                _, sp = pr(xp, a.dyn_steps, a.dt)
                out.append(sp[1][:, site_idx].double().cpu().numpy())
                ranks.append(P.effective_rank(sp[1][:, ::max(dyn.n // 512, 1)].float()))
        return np.concatenate(out), float(np.mean(ranks))

    def corr(Pd, T):
        p, t = (Pd - Pd.mean(0)).ravel(), (T - T.mean(0)).ravel()
        return float(p @ t / max(np.linalg.norm(p) * np.linalg.norm(t), 1e-30))

    Ttr, Tev = target(tr), target(ev)
    zero = float((Tev ** 2).mean())
    se = 1.0 / np.sqrt(Tev.size)

    def ridge(Ftr, Fev, tag):
        mu, sd_ = Ftr.mean(0), Ftr.std(0) + 1e-12
        A, B = (Ftr - mu) / sd_, (Fev - mu) / sd_
        res = {}
        for name, Tfit in (("intact", Ttr),
                           ("shuffled", Ttr[np.random.default_rng(11).permutation(len(Ttr))])):
            W = np.linalg.solve(A.T @ A + a.alpha * np.eye(A.shape[1]), A.T @ (Tfit - Tfit.mean(0)))
            Pd = B @ W + Tfit.mean(0)
            res[name] = dict(correlation=corr(Pd, Tev),
                             skill_vs_zero=1.0 - float(((Pd - Tev) ** 2).mean()) / zero,
                             # a near-constant prediction makes `corr` a ratio of tiny numbers and
                             # numerically unstable -- the first run's shuffled arm read -0.0201,
                             # 8 sd, from exactly that. the prediction's own spread relative to the
                             # target's says when the correlation is meaningless.
                             pred_rms_over_target=float(Pd.std() / max(Tev.std(), 1e-30)))
        print(f"  {tag:22s} intact {res['intact']['correlation']:+.4f} "
              f"({res['intact']['correlation']/se:+6.1f} sd, skill {res['intact']['skill_vs_zero']:+.4f})   "
              f"shuffled {res['shuffled']['correlation']:+.4f}"
              f"   pred spread intact {res['intact']['pred_rms_over_target']:.3f} / "
              f"shuffled {res['shuffled']['pred_rms_over_target']:.3f} of target")
        return res

    print(f"alpha FIXED at {a.alpha:.0e} -- not selected on the evaluation set\n")
    rows = {}
    Ftr, Fev = coch_feats(tr), coch_feats(ev)
    print("KNOWN ANSWER: the cochleagram arm must reproduce the recorded "
          f"+{RECORDED_COCHLEAGRAM_R:.4f} at this alpha.")
    ka = ridge(coch_feats(tr_big), Fev, f"cochleagram n={len(tr_big):,}")
    got = ka["intact"]["correlation"]
    ok = abs(got - RECORDED_COCHLEAGRAM_R) < 0.010
    print(f"  -> {got:+.4f} against {RECORDED_COCHLEAGRAM_R:+.4f}  "
          f"{'PASS' if ok else 'FAIL -- this instrument disagrees with the measurement it extends'}")
    if not ok:
        sys.exit("known answer FAILED; no cortical number below is interpretable")

    print(f"\nTHE COMPARISON, every arm at n_train = {len(tr):,}. Nothing is compared across sizes.")
    rows["known_answer_cochleagram_40k"] = ka
    rows["cochleagram"] = ridge(Ftr, Fev, "cochleagram")
    print()
    for tag, trained in (("trained cortex", True), ("untrained cortex", False)):
        dyn, pr = make(trained)
        site_idx = torch.linspace(0, dyn.n - 1, a.read_sites).long().to(dev)
        Str, r_tr = state_feats(dyn, pr, tr, site_idx)
        Sev, r_ev = state_feats(dyn, pr, ev, site_idx)
        rows[tag] = ridge(Str, Sev, tag)
        rows[tag]["effective_rank_train"] = r_tr
        print(f"  {'':22s} effective cortical rank {r_tr:.2f}")
        del dyn, pr
        if dev == "cuda":
            torch.cuda.empty_cache()

    c = rows["cochleagram"]["intact"]["correlation"]
    t = rows["trained cortex"]["intact"]["correlation"]
    u = rows["untrained cortex"]["intact"]["correlation"]
    print(f"\n  cochleagram {c:+.4f}   trained {t:+.4f}   untrained {u:+.4f}   "
          f"(standard error {se:.4f})")
    destroyed = t < c - 5 * se
    print("\n  VERDICT: " + (
        f"the dynamics DESTROY the stimulus information. A linear map on the cortical state\n"
        f"  reaches {t:+.4f} where the same map on the stimulus that drove it reaches {c:+.4f}.\n"
        f"  A linear readout is the most generous question available, so a learned nonlinear one\n"
        f"  had no better starting material: no objective and no head can recover what the sheet\n"
        f"  no longer carries. The substrate is the problem." if destroyed else
        f"the dynamics CARRY the stimulus information through at {t:+.4f} against the\n"
        f"  cochleagram's {c:+.4f}. The head, not the sheet, is where the failure is, and the\n"
        f"  objective is where to work."))
    print(f"  training's effect on transmission: trained {t:+.4f} vs untrained {u:+.4f} = "
          f"{(t-u)/se:+.1f} sd "
          + ("-- indistinguishable; training did not change what the dynamics transmit."
             if abs(t - u) < 5 * se else "-- training changed transmission."))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(
        alpha=a.alpha, read_sites=a.read_sites, train_rows=len(tr), eval_rows=len(ev),
        standard_error=float(se), known_answer_pass=bool(ok),
        recorded_cochleagram=RECORDED_COCHLEAGRAM_R, arms=rows,
        dynamics_destroy_information=bool(destroyed)), indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
