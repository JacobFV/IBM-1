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

the score that gate reads is the MEAN over several pools, not one.  a single pool
of 200 carries sd ~2.8 points, and taking the best single draw across a few hundred
evaluations selects the lucky pool rather than the better weights -- the same shape
of error as reporting a metric against the wrong population.
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
    ap.add_argument("--eval-pools", type=int, default=8,
                    help="held-out pools averaged per evaluation.  a single pool of "
                         "200 has sd ~2.8 points, so checkpointing on the best single "
                         "draw selects lucky pools rather than better weights -- the "
                         "mean over several pools is what the checkpoint gate reads")
    ap.add_argument("--n-steps", type=int, default=4,
                    help="dynamics passes before reading the embedding; this "
                         "head uses one state, so it does not need an epoch")
    ap.add_argument("--eval-every", type=int, default=100)
    # ---- the sheet and its long-range wiring (docs/DISCONNECTS.md 2 and 3) ----
    ap.add_argument("--geometry", default="surface", choices=("surface", "sphere"),
                    help="surface: fsaverage white-surface vertices with a DK "
                         "atlas.  sphere: the area-matched spherical proxy every "
                         "checkpoint before this was trained on, kept so the "
                         "change can be MEASURED against what it replaces")
    ap.add_argument("--long-topology", default="random", choices=("random", "tract"),
                    help="where the long-range partners go.  random is the "
                         "status quo and the control; tract draws them from the "
                         "braingraph HCP group connectome.  the two are matched "
                         "by construction on edge count and on row L1")
    ap.add_argument("--tract-threshold", type=float, default=0.5,
                    help="edge-existence consensus: the fraction of the 1064 "
                         "subjects an edge must appear in")
    ap.add_argument("--tract-delays", action="store_true",
                    help="carry the declared conduction delays on the long-range "
                         "edges.  they only bite at a dt that resolves them -- "
                         "the median declared delay is 3.8 ms")
    ap.add_argument("--delay-shuffle", action="store_true",
                    help="delay-matched control: the same multiset of delays, "
                         "moved to randomly chosen edges")
    ap.add_argument("--port-region", default=None,
                    help="the region the image drives.  the default is None, "
                         "which reproduces the arbitrary `[:n//8]` slice every "
                         "existing checkpoint used and whose docstring called it "
                         "'the occipital port' -- exactly the habit "
                         "cortical_regions was written to end.  pass occipital "
                         "to make the name true")
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="ckpt/visual_contrastive.pt")
    ap.add_argument("--out", default="out/visual_contrastive.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
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
    dyn = P.CorticalDynamics(
        a.sites, a.embed, a.k, dev, long_range=a.long_range,
        geometry=a.geometry, graph_seed=a.graph_seed,
        long_topology=a.long_topology, tract_threshold=a.tract_threshold,
        tract_delays=a.tract_delays, delay_shuffle=a.delay_shuffle).to(dev)
    model = P.VisualContrastiveLoop(dyn, n_sensors=C, n_times=T,
                                    port_region=a.port_region).to(dev)
    print(f"sheet: {a.geometry}, long-range {a.long_topology}"
          + (f" (consensus {a.tract_threshold}, "
             f"delays {'on' if a.tract_delays else 'off'}"
             f"{', SHUFFLED' if a.delay_shuffle else ''})"
             if a.long_topology == "tract" else "")
          + f" | port {a.port_region or 'slice[:n//8]'} = {model.port_size} sites",
          flush=True)
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
            accs = []
            with torch.no_grad():
                # the pools are drawn from a generator seeded on the STEP, so every
                # evaluation within a run sees a fresh sample while two runs compared
                # against each other see the same one at the same step.
                rng = np.random.default_rng(90_000 + step)
                for _ in range(a.eval_pools):
                    j = rng.integers(ntr, n - 1, a.pool)
                    xt = torch.from_numpy(np.ascontiguousarray(imgs[j])).to(dev)
                    xt = (xt.permute(0, 3, 1, 2).float() / 127.5) - 1
                    yt = torch.from_numpy(eeg(j)).to(dev)
                    zct, st = model.embed_image(xt, substeps=a.substeps, dt=a.dt,
                                                n_steps=a.n_steps)
                    sim = zct @ model.embed_eeg(yt).T
                    accs.append(float((sim.argmax(1) ==
                                       torch.arange(len(j), device=dev)).float().mean()))
                r_eff = P.effective_rank(st[1][:, ::max(dyn.n // 512, 1)].float())
            top1, top1_sd = float(np.mean(accs)), float(np.std(accs))
            rec = {"step": step, "loss": float(loss.detach()), "top1": top1,
                   "top1_sd": top1_sd, "pools": a.eval_pools,
                   "chance": 1.0 / a.pool, "r_eff": r_eff,
                   "v_absmax": float(s[0].abs().max()), "sec": round(time.time() - t0, 1)}
            log["steps"].append(rec)
            flag = ""
            if top1 > best:
                best = top1
                flag = "  <- best, checkpointed"
                os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                            "top1": top1, "top1_sd": top1_sd, "config": vars(a)},
                           a.ckpt)
            print(f"{step:5d}  loss {float(loss):.4f}  top-1 {100*top1:5.2f}% "
                  f"+/-{100*top1_sd:.2f}  ({top1*a.pool:.1f}x chance)  "
                  f"r_eff {r_eff:5.2f}  {time.time()-t0:6.0f}s{flag}", flush=True)

    log["best_top1"] = best
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"best held-out top-1: {100*best:.2f}% ({best*a.pool:.1f}x chance)",
          flush=True)


if __name__ == "__main__":
    main()
