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

THREE ARMS, one split, one evaluation draw:
  cochleagram    the raw stimulus features. **This is the known answer**: it must reproduce the
                 +0.0466 already recorded at alpha 1e4, from the same code path and the same
                 rows. If it does not, this instrument disagrees with the measurement it is
                 built to extend and nothing else it prints can be read.
  trained        the cortical state of `ckpt/clean_b64.pt`, driven by the same stimulus.
  untrained      the cortical state of a FRESHLY INITIALISED substrate, same seed, same topology,
                 same driving path. The control that says whether training changed transmission
                 at all -- and it can fail in both directions, which is why it is here.

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
    ap.add_argument("--n-train", type=int, default=6000)
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
                             skill_vs_zero=1.0 - float(((Pd - Tev) ** 2).mean()) / zero)
        print(f"  {tag:22s} intact {res['intact']['correlation']:+.4f} "
              f"({res['intact']['correlation']/se:+6.1f} sd, skill {res['intact']['skill_vs_zero']:+.4f})   "
              f"shuffled {res['shuffled']['correlation']:+.4f}")
        return res

    print(f"alpha FIXED at {a.alpha:.0e} -- not selected on the evaluation set\n")
    rows = {}
    Ftr, Fev = coch_feats(tr), coch_feats(ev)
    print("KNOWN ANSWER: the cochleagram arm must reproduce the recorded "
          f"+{RECORDED_COCHLEAGRAM_R:.4f} at this alpha.")
    rows["cochleagram"] = ridge(Ftr, Fev, "cochleagram")
    got = rows["cochleagram"]["intact"]["correlation"]
    ok = abs(got - RECORDED_COCHLEAGRAM_R) < 0.010
    print(f"  -> {got:+.4f} against {RECORDED_COCHLEAGRAM_R:+.4f}  "
          f"{'PASS' if ok else 'FAIL -- this instrument disagrees with the measurement it extends'}")
    if not ok:
        sys.exit("known answer FAILED; no cortical number below is interpretable")

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
