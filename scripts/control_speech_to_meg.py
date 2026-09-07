"""what ceiling does speech -> MEG have, and does it move under a contrastive loss?

the MEG head has sat at chance for 26,000 steps while its ablation says the cortex
contributes 35% -- doing real work that produces no prediction.  the visual term
had exactly that shape and was rescued by changing the OBJECTIVE: waveform
regression peaked at skill +0.011 there while contrastive retrieval reached 42x
chance on the same pairs.

so this asks the same question of the auditory side, with no dynamics in the way.
two heads on identical data and splits:

  regression  cochleagram window -> 306-channel MEG at the window's end
  contrastive cochleagram window and MEG window into a shared embedding

if regression has a low ceiling and retrieval does not, the MEG term should be
rebuilt contrastively too, and 26,000 steps of chance was an objective error
rather than a capacity or architecture one.

splits are contiguous in time with a guard band -- MEG at 250 Hz is heavily
autocorrelated and a random split would leak neighbouring samples across it.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os as _os
D = _os.environ.get("IBM_PAIRED_MEG", "data/derived/libribrain-paired")
CTX = 125          # 500 ms of cochleagram at 250 Hz
GAP = 2500         # 10 s guard band between train and test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("regression", "contrastive"), default="regression")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--win", type=int, default=50, help="MEG samples per contrastive window")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stim = np.load(f"{D}/cochleagram_250hz.npy", mmap_mode="r")
    neur = np.load(f"{D}/meg_250hz.npy", mmap_mode="r")
    sc = np.load(f"{D}/meg_scale.npy")
    n = min(len(stim), len(neur))
    ntr = int(n * 0.8)
    C = neur.shape[1]

    def meg(i):
        y = (np.ascontiguousarray(neur[i]).astype(np.float32) - sc[0]) / sc[1]
        return np.clip(y, -6, 6)

    def meg_win(i):
        # contiguous SLICES, not fancy indexing.  neur[q:q+win] reads one run of
        # memory; meg(np.arange(q, q+win)) issues `win` scattered reads into a 4 GB
        # memmap, and at batch 64 that is 3,200 of them per step -- enough to stall
        # the run entirely.
        out = np.empty((len(i), C, a.win), np.float32)
        for b, q in enumerate(i):
            y = (np.asarray(neur[q:q + a.win]).astype(np.float32) - sc[0]) / sc[1]
            out[b] = np.clip(y, -6, 6).T
        return out

    def coch(i):
        out = np.empty((len(i), CTX, stim.shape[1]), np.float32)
        for b, q in enumerate(i):
            out[b] = np.asarray(stim[q - CTX:q])
        return out

    if a.mode == "regression":
        base = float((meg(np.arange(ntr + GAP, n, 37)) ** 2).mean())
        print(f"[regression] zero baseline {base:.4f}  train 0-{ntr:,} "
              f"test {ntr+GAP:,}-{n:,}", flush=True)
        m = nn.Sequential(
            nn.Conv1d(stim.shape[1], 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Flatten(), nn.Linear(128 * ((CTX + 1) // 2), 512), nn.GELU(),
            nn.Linear(512, C)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        print(f"params {sum(p.numel() for p in m.parameters()):,}", flush=True)
        for step in range(a.steps + 1):
            i = np.random.randint(CTX, ntr, a.batch)
            x = torch.from_numpy(coch(i)).to(dev).permute(0, 2, 1)
            y = torch.from_numpy(meg(i)).to(dev)
            loss = F.mse_loss(m(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % 500 == 0:
                with torch.no_grad():
                    j = np.random.randint(ntr + GAP, n - 1, 512)
                    xt = torch.from_numpy(coch(j)).to(dev).permute(0, 2, 1)
                    yt = torch.from_numpy(meg(j)).to(dev)
                    h = float(F.mse_loss(m(xt), yt))
                print(f"  {step:5d}  train {float(loss):.4f}  held {h:.4f}  "
                      f"SKILL {1 - h/base:+.4f}", flush=True)
    else:
        print(f"[contrastive] pool {a.pool}, chance {100/a.pool:.2f}%  "
              f"window {a.win} samples ({a.win*4} ms)", flush=True)
        se = nn.Sequential(
            nn.Conv1d(stim.shape[1], 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Flatten(), nn.Linear(128 * ((CTX + 1) // 2), 512), nn.GELU(),
            nn.Linear(512, 128)).to(dev)
        me = nn.Sequential(
            nn.Conv1d(C, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Flatten(), nn.Linear(128 * ((a.win + 1) // 2), 512), nn.GELU(),
            nn.Linear(512, 128)).to(dev)
        temp = nn.Parameter(torch.tensor(0.07, device=dev))
        opt = torch.optim.AdamW(list(se.parameters()) + list(me.parameters()) + [temp],
                                lr=3e-4, weight_decay=1e-4)
        print(f"params {sum(p.numel() for p in list(se.parameters())+list(me.parameters())):,}",
              flush=True)
        for step in range(a.steps + 1):
            i = np.random.randint(CTX, ntr - a.win, a.batch)
            x = torch.from_numpy(coch(i)).to(dev).permute(0, 2, 1)
            y = torch.from_numpy(meg_win(i)).to(dev)
            zs = F.normalize(se(x), dim=-1); zm = F.normalize(me(y), dim=-1)
            logits = zs @ zm.T / temp.clamp(0.01, 1.0)
            lbl = torch.arange(len(i), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % 500 == 0:
                with torch.no_grad():
                    j = np.random.randint(ntr + GAP, n - a.win - 1, a.pool)
                    xt = torch.from_numpy(coch(j)).to(dev).permute(0, 2, 1)
                    yt = torch.from_numpy(meg_win(j)).to(dev)
                    sim = F.normalize(se(xt), dim=-1) @ F.normalize(me(yt), dim=-1).T
                    top1 = float((sim.argmax(1) ==
                                  torch.arange(len(j), device=dev)).float().mean())
                print(f"  {step:5d}  loss {float(loss):.4f}  top-1 {100*top1:5.2f}%  "
                      f"({top1*a.pool:.1f}x chance)", flush=True)


if __name__ == "__main__":
    main()
