#!/usr/bin/env python3
"""Score a multi-materialization checkpoint's paired head on ONE FIXED set of batches.

Why a fixed set, fixed before the runs it compares (docs/LOG.md, 2026-09-11): raising the training
batch size ALSO raises the zero baseline, because the MEG target is heavy-tailed and a larger draw
converges toward its mean (1.24e-02) rather than its median (4.55e-05). An arm scored on its own
training draw would therefore look better for a reason that has nothing to do with the model. One
fixed evaluation draw, one zero baseline, and batch size affects training only.

THE SPLIT. Until 2026-09-11 this trainer drew every paired index with `np.random.randint(pctx, lim)`
over the WHOLE array and had no train/test split anywhere in it, so every number it had ever printed
was in-sample. It now reserves a tail fraction (`--holdout`, default 0.1) and never draws from it.
This script scores on that tail and REFUSES to run against a checkpoint whose recorded
`paired_train_lim` does not match the split it is about to evaluate on -- otherwise the held-out set
would silently be training data for one arm and not the other, which is the ledger's commonest shape.

THE SCALE. The trainer normalises the target: `yn = clip((yn - megsc[0]) / megsc[1], -6, 6)` when a
`meg_scale` sidecar exists, and its `skill/0` is therefore against the SCALED target. This script
applies exactly the same transform, from the same file. Scoring raw against a scaled-trained head
would be the wrong-units error the ledger already carries once.

KNOWN ANSWERS, both printed before any checkpoint is scored:
  1. the zero predictor must score exactly 0.0 skill on the fixed set, by construction;
  2. the mean predictor -- the per-sensor mean of the evaluation targets -- must score >= 0, because
     a constant cannot be worse than zero unless the target is not centred. If it comes out
     negative, the target has an offset and every skill here is measured against the wrong origin.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def fixed_draw(lo, hi, n_batches, batch, seed=20260911):
    """The one fixed evaluation draw, from the held-out tail [lo, hi)."""
    g = np.random.default_rng(seed)
    return [g.integers(lo, hi, size=batch) for _ in range(n_batches)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--paired-stim", required=True)
    ap.add_argument("--paired-neural", required=True)
    ap.add_argument("--sites", type=int, default=150_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--lead-rank", type=int, default=64)
    ap.add_argument("--dyn-steps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--out", default="out/paired_eval.json")
    a = ap.parse_args()

    import pretrain_video_loop as P
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pstim = np.load(a.paired_stim, mmap_mode="r")
    pneur = np.load(a.paired_neural, mmap_mode="r")
    pctx = 125

    # the same transform the trainer applies, from the same sidecar
    _sc = a.paired_neural.replace("meg_250hz", "meg_scale")
    megsc = np.load(_sc) if Path(_sc).exists() else None
    print(f"meg scale: {'applied from ' + _sc if megsc is not None else 'NONE -- raw target'}")

    def target(j):
        y = np.ascontiguousarray(pneur[j]).astype(np.float32)
        return np.clip((y - megsc[0]) / megsc[1], -6, 6) if megsc is not None else y

    p_rows = min(len(pstim), len(pneur)) - 2
    lo = int(p_rows * (1.0 - a.holdout))
    draw = fixed_draw(lo, p_rows, a.batches, a.batch)
    Y = np.concatenate([target(j) for j in draw])
    zero_mse = float((Y ** 2).mean())
    mean_pred = Y.mean(0, keepdims=True)
    mean_mse = float(((Y - mean_pred) ** 2).mean())

    print(f"fixed evaluation set: {a.batches} batches x {a.batch} timepoints = {len(Y):,} rows, "
          f"{Y.shape[1]} sensors, drawn from the held-out tail [{lo:,}, {p_rows:,})  (seed 20260911)")
    print("KNOWN ANSWERS")
    print(f"  zero predictor  skill {0.0:+.6f}   (must be exactly 0 by construction)  PASS")
    sk_mean = 1.0 - mean_mse / zero_mse
    ok = sk_mean >= -1e-9
    print(f"  mean predictor  skill {sk_mean:+.6f}   (must be >= 0; negative means the target is "
          f"not centred)  {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit("known answer FAILED -- the target has an offset; no checkpoint is scored")
    # HOW HEAVY IS THIS PARTICULAR DRAW? the target's per-batch mean square spans five orders
    # of magnitude (median 4.55e-05, mean 1.24e-02, max 1.60e+01), so the zero baseline of any
    # one fixed draw is itself a high-variance quantity -- a draw that samples none of the tail
    # reports a baseline near the median and flatters nothing, but is not representative.
    # The ARM-TO-ARM comparison does not depend on it, because both arms are scored on this
    # same draw; the absolute skill does. Both are printed so neither gets read as the other.
    bms = np.array([float((target(j) ** 2).mean()) for j in draw])
    print(f"  zero-baseline mse on this fixed set: {zero_mse:.6e}")
    print(f"  the draw's own per-batch spread:     median {np.median(bms):.3e}  "
          f"mean {bms.mean():.3e}  max {bms.max():.3e}  ({bms.max()/np.median(bms):.0f}x)")

    rows = []
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        sd = torch.load(path, map_location=dev)
        # THE REFUSAL.  a checkpoint trained before the split existed, or under a
        # different holdout, has SEEN these rows.  scoring it here would report
        # training loss for one arm and held-out loss for the other and call the
        # difference a result.
        tl = sd.get("paired_train_lim")
        if tl is None:
            sys.exit(f"{name}: no paired_train_lim in the checkpoint -- it was trained before the "
                     f"split existed, so the held-out tail is training data for it. Not scored.")
        if tl != lo:
            sys.exit(f"{name}: trained with paired_train_lim {tl:,} but this evaluation set starts "
                     f"at {lo:,}. Mismatched splits. Not scored.")
        dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
        pr = P.PairedNeuralLoop(dyn, n_bands=pstim.shape[-1], n_sensors=pneur.shape[-1],
                                lead_rank=a.lead_rank).to(dev)
        dyn.load_state_dict(sd["dyn"]); pr.load_state_dict(sd["paired"])
        dyn.eval(); pr.eval()
        se, n, ranks = 0.0, 0, []
        with torch.no_grad():
            for j in draw:
                xp = torch.from_numpy(np.stack([pstim[q - pctx:q] for q in j])).float().to(dev)
                yp = torch.from_numpy(target(j)).to(dev)
                pp, sp = pr(xp, a.dyn_steps, a.dt)
                se += float(((pp - yp) ** 2).sum()); n += yp.numel()
                ranks.append(P.effective_rank(sp[1][:, ::max(dyn.n // 512, 1)].float()))
        mse = se / n
        rows.append({"name": name, "path": path, "step": int(sd.get("step", -1)),
                     "mse": mse, "skill_vs_zero": 1.0 - mse / zero_mse,
                     "mean_effective_rank": float(np.mean(ranks))})
        print(f"  {name:12s} step {rows[-1]['step']:5d}  mse {mse:.6e}  "
              f"skill vs zero {rows[-1]['skill_vs_zero']:+11.2f}  rank {rows[-1]['mean_effective_rank']:5.2f}")

    # the baseline-free statement of the comparison.  whatever the zero baseline happens to be
    # on this draw, it divides out of a ratio between two arms scored on the same rows.
    if len(rows) > 1:
        best = min(r["mse"] for r in rows)
        print("\n  arm-to-arm, which does NOT depend on the zero baseline:")
        for r in sorted(rows, key=lambda r: r["mse"]):
            r["mse_ratio_to_best"] = r["mse"] / best
            print(f"    {r['name']:12s} mse {r['mse']:.6e}  = {r['mse_ratio_to_best']:8.3f}x the best arm")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"fixed_set": {"batches": a.batches, "batch": a.batch, "seed": 20260911,
                       "holdout": a.holdout, "rows": [lo, p_rows],
                       "meg_scale": _sc if megsc is not None else None,
                       "zero_mse": zero_mse, "mean_predictor_skill": sk_mean,
                       "per_batch_ms": {"median": float(np.median(bms)), "mean": float(bms.mean()),
                                        "max": float(bms.max())}},
         "arms": rows}, indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
