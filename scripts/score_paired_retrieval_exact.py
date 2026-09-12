#!/usr/bin/env python3
"""Score the contrastive arms against the ridge exactly, on the full held-out tail, paired.

The training runs scored retrieval by drawing 200 pools of 32 from **200 windows**, which
overstates precision badly: the 6,400 trials are not independent, the error scales with the
windows, and treating them as independent understated the standard error 5.7x (docs/LOG.md, the
correction committed before this result existed). Two things are fixed here.

**The full tail, not a twentieth of it.** The reserved tail holds 2,821 non-overlapping 0.5 s
windows; the training evaluation used 200 because `--n-eval` was left at a value chosen when
evaluation cost a ridge solve. Every arm is re-scored on all of them, by the same procedure, so
enlarging the set is not moving a bar.

**Exact top-1, not sampled pools.** Drawing pools estimates a quantity that can be computed in
closed form. For window `i`, let `r_i` be the rank of its true partner among all `N` candidates
(1 = best). A random pool of size `P` containing `i` retrieves it iff none of the `P-1` other
members outranks it, so

    p_i = C(N - r_i, P - 1) / C(N - 1, P - 1)

and top-1 = mean(p_i). No sampling noise at all, and -- the point -- **`p_i` is a per-window
scalar**, so the comparison between two arms can be bootstrapped over windows, paired, which is
what actually cancels the shared window-sampling error.

KNOWN ANSWERS, both printed before any arm is compared:
  1. An arm whose ranks are UNIFORM must score exactly chance, `1/P`. Checked against a
     random-embedding arm generated here, which cannot help but be uniform.
  2. `p_i` at `r_i = 1` must be exactly 1.0 and at `r_i = N` exactly 0.0, by construction.
Both must hold or the closed form is wrong and every number below with it.

The SHUFFLED arm is the control that can fail: trained on destroyed pairings, it must land at
chance on this measure too. If it does not, the evaluation leaks and the intact number is void.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def top1_from_ranks(ranks, N, P):
    """Exact probability that each item is retrieved top-1 in a random pool of size P."""
    # C(N-r, P-1) / C(N-1, P-1), computed in logs for stability
    lg = lambda m, k: (math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)
                       if m >= k >= 0 else -math.inf)
    denom = lg(N - 1, P - 1)
    return np.array([math.exp(lg(N - r, P - 1) - denom) if N - r >= P - 1 else 0.0
                     for r in ranks])


def ranks_of_diagonal(S):
    """Rank of each row's own column in the similarity matrix (1 = best)."""
    diag = np.diag(S).copy()
    return (S > diag[:, None]).sum(1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-stim", required=True)
    ap.add_argument("--paired-neural", required=True)
    ap.add_argument("--intact", default="ckpt/contrastive_intact.pt")
    ap.add_argument("--shuffled", default="ckpt/contrastive_shuffled.pt")
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ridge-train", type=int, default=40_000)
    ap.add_argument("--ridge-alpha", type=float, default=1e4)
    ap.add_argument("--lags", type=int, default=25)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--exclude", default="1619681:1627268")
    ap.add_argument("--gpu-batch", type=int, default=16)
    ap.add_argument("--out", default="out/paired_retrieval_exact.json")
    a = ap.parse_args()

    import pretrain_video_loop as P
    from train_paired_contrastive import CorticalStimulusEncoder, MEGEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(a.paired_stim, mmap_mode="r")
    Y = np.load(a.paired_neural, mmap_mode="r")
    sc = a.paired_neural.replace("meg_250hz", "meg_scale")
    megsc = np.load(sc) if Path(sc).exists() else None
    n = min(len(X), len(Y)) - 2

    cfg = torch.load(a.intact, map_location="cpu")["config"]
    ctx, win, lo = cfg["ctx"], cfg["window"], int(n * (1.0 - a.holdout))
    ev = np.arange(lo + ctx, n - win, win)                 # EVERY window in the tail
    N, Pn = len(ev), a.pool
    print(f"held-out tail: {N:,} non-overlapping {win}-sample ({win/250:.2f} s) windows "
          f"-- the training runs used 200\n")

    def meg_win(starts):
        o = np.stack([np.asarray(Y[s:s + win]) for s in starts]).astype(np.float32)
        if megsc is not None:
            o = np.clip((o - megsc[0]) / megsc[1], -6, 6)
        return np.ascontiguousarray(o, dtype=np.float32)

    def coch(starts):
        return np.stack([np.asarray(X[s - ctx:s]) for s in starts]).astype(np.float32)

    chance = 1.0 / Pn
    print("KNOWN ANSWERS")
    r1 = top1_from_ranks([1], N, Pn)[0]; rN = top1_from_ranks([N], N, Pn)[0]
    ok_edge = abs(r1 - 1.0) < 1e-12 and abs(rN) < 1e-12
    print(f"  p(rank 1) = {r1:.6f} must be 1;  p(rank N) = {rN:.6f} must be 0   "
          f"{'PASS' if ok_edge else 'FAIL'}")
    unif = top1_from_ranks(np.arange(1, N + 1), N, Pn).mean()
    ok_unif = abs(unif - chance) < 1e-6
    print(f"  UNIFORM ranks must score exactly chance {chance:.6f}; got {unif:.6f}   "
          f"{'PASS' if ok_unif else 'FAIL'}")
    if not (ok_edge and ok_unif):
        sys.exit("a known answer FAILED; the closed form is wrong")

    # ---------- the ridge arm, in sensor space -------------------------------------
    lag_idx = np.linspace(0, ctx - 1, a.lags).astype(int)
    excl = [tuple(int(v) for v in r.split(":")) for r in a.exclude.split(",") if r.strip()]
    pool_idx = np.arange(ctx, lo - win)
    for x0, x1 in excl:
        pool_idx = pool_idx[~((pool_idx >= x0 - win) & (pool_idx < x1))]
    tr = np.sort(np.random.default_rng(7).choice(pool_idx, size=min(a.ridge_train, len(pool_idx)),
                                                 replace=False))

    def feats(j):
        return np.stack([np.asarray(X[q - ctx:q])[lag_idx].ravel() for q in j]).astype(np.float64)

    def tgt(j):
        y = np.ascontiguousarray(Y[np.asarray(j)]).astype(np.float32)
        o = np.clip((y - megsc[0]) / megsc[1], -6, 6) if megsc is not None else y
        return np.ascontiguousarray(o, dtype=np.float64)

    Ftr, Ttr = feats(tr), tgt(tr)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    A = (Ftr - mu) / sd
    Wr = np.linalg.solve(A.T @ A + a.ridge_alpha * np.eye(A.shape[1]), A.T @ (Ttr - Ttr.mean(0)))

    def ridge_windows():
        Pw, Tw = [], []
        for s in range(0, N, 256):
            q = ev[s:s + 256]
            pr = np.stack([(((feats(np.arange(t, t + win)) - mu) / sd) @ Wr + Ttr.mean(0))
                           for t in q])
            Pw.append(pr.reshape(len(q), -1))
            Tw.append(meg_win(q).reshape(len(q), -1).astype(np.float64))
        return np.concatenate(Pw), np.concatenate(Tw)

    print("\nscoring the ridge arm ...", flush=True)
    Pw, Tw = ridge_windows()
    Pw -= Pw.mean(1, keepdims=True); Tw -= Tw.mean(1, keepdims=True)
    Pw /= np.linalg.norm(Pw, axis=1, keepdims=True) + 1e-30
    Tw /= np.linalg.norm(Tw, axis=1, keepdims=True) + 1e-30
    p_ridge = top1_from_ranks(ranks_of_diagonal(Pw @ Tw.T), N, Pn)

    # ---------- the contrastive arms ------------------------------------------------
    def score_ckpt(path, tag):
        sd_ = torch.load(path, map_location=dev)
        c = sd_["config"]
        dyn = P.CorticalDynamics(c["sites"], c["embed"], c["k"], dev,
                                 long_range=c["long_range"]).to(dev)
        es = CorticalStimulusEncoder(dyn, X.shape[-1], c["ctx"], c["dim"],
                                     bypass=(sd_["arm"] == "bypass")).to(dev)
        em = MEGEncoder(Y.shape[-1], c["window"], c["dim"]).to(dev)
        dyn.load_state_dict(sd_["dyn"]); es.load_state_dict(sd_["enc_s"]); em.load_state_dict(sd_["enc_m"])
        es.eval(); em.eval()
        Aa, Bb = [], []
        with torch.no_grad():
            for s in range(0, N, a.gpu_batch):
                q = ev[s:s + a.gpu_batch]
                Aa.append(es(torch.from_numpy(coch(q)).to(dev), c["dyn_steps"], c["dt"]).cpu())
                Bb.append(em(torch.from_numpy(meg_win(q)).to(dev)).cpu())
        Aa = F.normalize(torch.cat(Aa), dim=1).numpy()
        Bb = F.normalize(torch.cat(Bb), dim=1).numpy()
        del dyn, es, em
        if dev == "cuda":
            torch.cuda.empty_cache()
        return top1_from_ranks(ranks_of_diagonal(Aa @ Bb.T), N, Pn)

    print("scoring the intact arm ...", flush=True)
    p_intact = score_ckpt(a.intact, "intact")
    print("scoring the shuffled control ...", flush=True)
    p_shuf = score_ckpt(a.shuffled, "shuffled")

    def report(p, tag):
        print(f"  {tag:22s} top-1 {p.mean():7.2%} = {p.mean()/chance:5.2f}x chance")
        return float(p.mean())

    print(f"\nEXACT TOP-1 over all {N:,} windows (pool {Pn}, chance {chance:.2%}):")
    m_ridge = report(p_ridge, "ridge (the bar)")
    m_intact = report(p_intact, "contrastive intact")
    m_shuf = report(p_shuf, "contrastive shuffled")

    rng = np.random.default_rng(20260911)
    def boot(pa, pb):
        d = np.array([(pa[i] - pb[i]).mean()
                      for i in (rng.integers(0, N, size=N) for _ in range(a.boot))])
        return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)

    print(f"\nPAIRED BOOTSTRAP over windows ({a.boot:,} resamples), in units of chance:")
    for tag, pa, pb in (("intact - ridge", p_intact, p_ridge),
                        ("intact - shuffled", p_intact, p_shuf),
                        ("shuffled - ridge", p_shuf, p_ridge)):
        m, lo_, hi_ = boot(pa, pb)
        sig = "excludes 0" if (lo_ > 0 or hi_ < 0) else "includes 0 -- NOT significant"
        print(f"  {tag:20s} {m/chance:+6.2f}x   95% CI [{lo_/chance:+.2f}, {hi_/chance:+.2f}]   {sig}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(
        n_windows=N, pool=Pn, chance=chance, exact=True, bootstrap=a.boot,
        top1=dict(ridge=m_ridge, intact=m_intact, shuffled=m_shuf),
        x_chance=dict(ridge=m_ridge / chance, intact=m_intact / chance,
                      shuffled=m_shuf / chance)), indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
