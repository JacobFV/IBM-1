"""train the auditory paired term as contrastive alignment through the cortex.

the twin of `train_visual_contrastive.py`, and the first auditory term the
programme has been able to attempt honestly.  every earlier one was measured
against `data/derived/libribrain-paired` as it stood before 2026-09-07, whose
audio and MEG streams drift up to +/-3.4 s apart; four negative results came out
of that and are void.  the corpus now passes `scripts/check_pairing.py` on all
five windows (p = 0.024, mean coupling 0.108).

**the bar is measured, not assumed.**  the dynamics-free control on this corpus
and split, averaged over 8 held-out pools of 200:

    200 ms window   4.62% +/- 0.99    9.2x chance
    1 s    window   7.12% +/- 1.71   14.2x chance

so this has to beat **7.12%** to be worth keeping.  a cortex that merely matches
its own bypass is decorative, which is the failure mode s5.ablate exists to catch
and which the visual branch had to survive before its number meant anything.

the split is contiguous with a guard band.  MEG at 250 Hz is heavily
autocorrelated and a random split leaks neighbouring windows across it.

evaluation averages `--eval-pools` held-out pools: one pool of 200 has sd ~2.8
points, and checkpointing on the best single draw selects lucky pools rather than
better weights (ledger row 9, which recurred three times in one day).
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

D = os.environ.get("IBM_PAIRED_MEG", "data/derived/libribrain-paired")
GAP = 2500          # 10 s guard band


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--win", type=int, default=250,
                    help="MEG samples per window; 250 = 1 s, where the control "
                         "reached 14.2x against 9.2x at 200 ms")
    ap.add_argument("--ctx", type=int, default=250, help="cochleagram frames in")
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=2e-2)
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--eval-pools", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ckpt", default="ckpt/audio_contrastive.pt")
    ap.add_argument("--out", default="out/audio_contrastive.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    stim = np.load(f"{D}/cochleagram_250hz.npy", mmap_mode="r")
    neur = np.load(f"{D}/meg_250hz.npy", mmap_mode="r")
    n = min(len(stim), len(neur))
    ntr = int(n * 0.8)
    C = neur.shape[1]

    def coch(i):
        out = np.empty((len(i), a.ctx, stim.shape[1]), np.float32)
        for b, q in enumerate(i):
            out[b] = np.asarray(stim[q - a.ctx:q])
        return out

    def meg(i):
        # contiguous slices, never fancy indexing: neur[q:q+win] is one run of
        # memory, while meg(arange(q, q+win)) is `win` scattered reads into a 4 GB
        # memmap and stalls the run outright.
        out = np.empty((len(i), C, a.win), np.float32)
        for b, q in enumerate(i):
            out[b] = np.clip(np.asarray(neur[q:q + a.win]).astype(np.float32), -6, 6).T
        return out

    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    model = P.AudioContrastiveLoop(dyn, n_sensors=C, n_bands=stim.shape[1],
                                   stim_len=a.ctx, n_times=a.win).to(dev)
    tot = sum(p.numel() for p in model.parameters())
    print(f"{n:,} samples ({n/250/60:.0f} min) | {C} ch | win {a.win} "
          f"({a.win*4} ms) | params {tot:,} ({dyn.embed.numel():,} association) | "
          f"pool {a.pool}, chance {100/a.pool:.2f}% | CONTROL CEILING 7.12% "
          f"(14.2x)", flush=True)

    temp = nn.Parameter(torch.tensor(0.07, device=dev))
    opt = torch.optim.AdamW(list(model.parameters()) + [temp], lr=a.lr, weight_decay=1e-4)
    log = {"config": vars(a), "n_params": tot, "chance": 1.0 / a.pool,
           "control_ceiling_top1": 0.0712, "steps": []}
    best, t0 = 0.0, time.time()

    for step in range(a.steps + 1):
        i = np.random.randint(a.ctx, ntr - a.win, a.batch)
        x = torch.from_numpy(coch(i)).to(dev).permute(0, 2, 1)
        y = torch.from_numpy(meg(i)).to(dev)
        zc, s = model.embed_audio(x, substeps=a.substeps, dt=a.dt, n_steps=a.n_steps)
        zm = model.embed_meg(y)
        logits = zc @ zm.T / temp.clamp(0.01, 1.0)
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
                rng = np.random.default_rng(90_000 + step)
                for _ in range(a.eval_pools):
                    j = rng.integers(ntr + GAP, n - a.win - 1, a.pool)
                    xt = torch.from_numpy(coch(j)).to(dev).permute(0, 2, 1)
                    yt = torch.from_numpy(meg(j)).to(dev)
                    zt, st = model.embed_audio(xt, substeps=a.substeps, dt=a.dt,
                                               n_steps=a.n_steps)
                    sim = zt @ model.embed_meg(yt).T
                    accs.append(float((sim.argmax(1) ==
                                torch.arange(len(j), device=dev)).float().mean()))
                r_eff = P.effective_rank(st[1][:, ::max(dyn.n // 512, 1)].float())
            top1, sd_ = float(np.mean(accs)), float(np.std(accs))
            log["steps"].append({"step": step, "loss": float(loss.detach()),
                                 "top1": top1, "top1_sd": sd_, "r_eff": r_eff,
                                 "sec": round(time.time() - t0, 1)})
            flag = ""
            if top1 > best:
                best = top1
                flag = "  <- best, checkpointed"
                os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step, "top1": top1,
                            "top1_sd": sd_, "config": vars(a)}, a.ckpt)
            bar = " OVER CEILING" if top1 > 0.0712 else ""
            print(f"{step:5d}  loss {float(loss):.4f}  top-1 {100*top1:5.2f}% "
                  f"+/-{100*sd_:.2f}  ({top1*a.pool:.1f}x)  r_eff {r_eff:5.2f}  "
                  f"{time.time()-t0:6.0f}s{flag}{bar}", flush=True)

    log["best_top1"] = best
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"best {100*best:.2f}% ({best*a.pool:.1f}x) against a control ceiling of "
          f"7.12% (14.2x)", flush=True)


if __name__ == "__main__":
    main()
