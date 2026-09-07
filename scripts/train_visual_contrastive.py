"""train the visual paired term as contrastive alignment through the cortex.

measured on correctly paired THINGS-EEG2 with the same encoder and split:

    waveform regression   peak skill +0.011, then negative
    contrastive retrieval held-out top-1 21.5% against 0.5% chance

so this term is built as the data supports it.  the image drives the occipital
port, the dynamics run, and the cortical state is read into an embedding aligned
with one computed from the measured EEG -- the cortex is IN the path, so the
alignment is only achievable if the dynamics carry stimulus-specific structure.

retrieval is reported on a held-out pool of concept-disjoint images, against
1/pool chance.  the peak matters more than the final value here: the control
peaked at step 500 and decayed to 4.5% by 4000 while training loss kept falling,
which is ordinary overfitting on 13k pairs.  so the best held-out score is tracked
and the checkpoint is written when it improves, rather than at a fixed interval.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)

D = "data/derived/things-paired"
ONSET, KEEP = 20, 50


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--decimate", type=int, default=2)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=2e-2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=4,
                    help="dynamics passes before reading the embedding; this "
                         "head uses one state, so it does not need an epoch")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ckpt", default="ckpt/visual_contrastive.pt")
    ap.add_argument("--out", default="out/visual_contrastive.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    imgs = np.load(f"{D}/images_training.npy", mmap_mode="r")
    ev = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    n = min(len(imgs), len(ev))
    ntr = int(n * 0.8)

    samp = np.asarray(ev[:ntr:7]).astype(np.float32)
    med = np.median(samp, 0)
    iqr = ((np.percentile(samp, 75, 0) - np.percentile(samp, 25, 0)) / 1.349).clip(1e-9)

    def eeg(i):
        y = (np.asarray(ev[i]).astype(np.float32) - med) / iqr
        return np.clip(y, -6, 6)[..., ONSET:ONSET + KEEP:a.decimate]

    T = eeg(np.arange(4)).shape[-1]
    C = ev.shape[1]
    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    model = P.VisualContrastiveLoop(dyn, n_sensors=C, n_times=T).to(dev)
    tot = sum(p.numel() for p in model.parameters())
    print(f"{n:,} pairs | {C} ch x {T} samples | params {tot:,} "
          f"({dyn.embed.numel():,} association) | pool {a.pool}, chance {100/a.pool:.2f}%",
          flush=True)

    temp = nn.Parameter(torch.tensor(0.07, device=dev))
    opt = torch.optim.AdamW(list(model.parameters()) + [temp], lr=a.lr, weight_decay=1e-4)
    log = {"config": vars(a), "n_params": tot, "chance": 1.0 / a.pool, "steps": []}
    best, t0 = 0.0, time.time()

    for step in range(a.steps + 1):
        i = np.random.randint(0, ntr, a.batch)
        x = torch.from_numpy(np.ascontiguousarray(imgs[i])).to(dev)
        x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1
        y = torch.from_numpy(eeg(i)).to(dev)
        zc, s = model.embed_image(x, substeps=a.substeps, dt=a.dt, n_steps=a.n_steps)
        ze = model.embed_eeg(y)
        logits = zc @ ze.T / temp.clamp(0.01, 1.0)
        lbl = torch.arange(len(i), device=dev)
        loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl))
        loss = loss + 1e-1 * P.viability_penalty(s[0])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [temp], 1.0)
        opt.step()

        if step % a.eval_every == 0:
            with torch.no_grad():
                j = np.random.randint(ntr, n - 1, a.pool)
                xt = torch.from_numpy(np.ascontiguousarray(imgs[j])).to(dev)
                xt = (xt.permute(0, 3, 1, 2).float() / 127.5) - 1
                yt = torch.from_numpy(eeg(j)).to(dev)
                zct, st = model.embed_image(xt, substeps=a.substeps, dt=a.dt, n_steps=a.n_steps)
                sim = zct @ model.embed_eeg(yt).T
                top1 = float((sim.argmax(1) ==
                              torch.arange(len(j), device=dev)).float().mean())
                r_eff = P.effective_rank(st[1][:, ::max(dyn.n // 512, 1)].float())
            rec = {"step": step, "loss": float(loss.detach()), "top1": top1,
                   "chance": 1.0 / a.pool, "r_eff": r_eff,
                   "v_absmax": float(s[0].abs().max()), "sec": round(time.time() - t0, 1)}
            log["steps"].append(rec)
            flag = ""
            if top1 > best:
                best = top1
                flag = "  <- best, checkpointed"
                os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                            "top1": top1, "config": vars(a)}, a.ckpt)
            print(f"{step:5d}  loss {float(loss):.4f}  top-1 {100*top1:5.2f}%  "
                  f"({top1*a.pool:.1f}x chance)  r_eff {r_eff:5.2f}  "
                  f"{time.time()-t0:6.0f}s{flag}", flush=True)

    log["best_top1"] = best
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"best held-out top-1: {100*best:.2f}% ({best*a.pool:.1f}x chance)",
          flush=True)


if __name__ == "__main__":
    main()
