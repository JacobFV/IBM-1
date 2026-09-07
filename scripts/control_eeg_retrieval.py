"""is there ANY mutual information between these images and this EEG?

regression in the image->EEG direction fails for every configuration tried: group
mean and per-subject, concept-disjoint and shared-concept splits, with and without
the cortical dynamics.  before concluding the corpus is unusable, the question has
to be asked in the form the corpus was built for.

THINGS-EEG2 is a DECODING dataset.  its published results are EEG -> image
identity, not image -> EEG waveform, and those are very different problems: a
waveform regression has to reproduce the amplitude of every channel at every
sample, while retrieval only has to preserve enough structure to tell one image
from another.  a signal far too weak for the first can be ample for the second.

so this trains a contrastive pair -- one encoder per side, a shared embedding --
and reports top-1 retrieval among a held-out pool.  chance is 1/pool.

what the outcome means for the programme:
  retrieval works    -> the corpus is fine and the paired VISUAL term should be
                        built as retrieval or a contrastive alignment, not as
                        waveform regression
  retrieval also fails -> there is no usable image-EEG mutual information here at
                        the single-image level, and the term should be dropped
                        rather than engineered around
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = "data/derived/things-paired"
ONSET, KEEP, DEC = 20, 50, 2


class ImgEnc(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.GELU(), nn.Linear(512, dim))

    def forward(self, x):
        return F.normalize(self.f(x), dim=-1)


class EegEnc(nn.Module):
    def __init__(self, ch, t, dim=128):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(ch, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Flatten(), nn.Linear(128 * ((t + 1) // 2), 512), nn.GELU(),
            nn.Linear(512, dim))

    def forward(self, x):
        return F.normalize(self.f(x), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=("groupmean", "persubject"), default="groupmean")
    ap.add_argument("--subject", type=int, default=0)
    ap.add_argument("--pool", type=int, default=200, help="retrieval pool; chance = 1/pool")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = np.load(f"{D}/images_training.npy", mmap_mode="r")
    if a.target == "persubject":
        ev = np.load(f"{D}/evoked_training_persubject.npy", mmap_mode="r")[a.subject]
    else:
        ev = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    n = min(len(imgs), len(ev))
    ntr = int(n * 0.8)

    samp = np.asarray(ev[:ntr:7]).astype(np.float32)
    med = np.median(samp, 0)
    iqr = ((np.percentile(samp, 75, 0) - np.percentile(samp, 25, 0)) / 1.349).clip(1e-9)

    def eeg(i):
        y = (np.asarray(ev[i]).astype(np.float32) - med) / iqr
        return np.clip(y, -6, 6)[..., ONSET:ONSET + KEEP:DEC]

    T = eeg(np.arange(4)).shape[-1]
    C = ev.shape[1]
    ie, ee = ImgEnc().to(dev), EegEnc(C, T).to(dev)
    opt = torch.optim.AdamW(list(ie.parameters()) + list(ee.parameters()),
                            lr=3e-4, weight_decay=1e-4)
    temp = nn.Parameter(torch.tensor(0.07, device=dev))
    print(f"[{a.target}] {n:,} pairs, {C} ch x {T} samples, pool {a.pool} "
          f"(chance {100/a.pool:.2f}%)", flush=True)

    for step in range(a.steps + 1):
        i = np.random.randint(0, ntr, a.batch)
        x = torch.from_numpy(np.ascontiguousarray(imgs[i])).to(dev)
        x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1
        y = torch.from_numpy(eeg(i)).to(dev)
        zi, ze = ie(x), ee(y)
        logits = zi @ ze.T / temp.clamp(0.01, 1.0)
        lbl = torch.arange(len(i), device=dev)
        loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 500 == 0:
            with torch.no_grad():
                j = np.random.randint(ntr, n - 1, a.pool)
                xt = torch.from_numpy(np.ascontiguousarray(imgs[j])).to(dev)
                xt = (xt.permute(0, 3, 1, 2).float() / 127.5) - 1
                yt = torch.from_numpy(eeg(j)).to(dev)
                s = ie(xt) @ ee(yt).T
                top1 = float((s.argmax(1) == torch.arange(len(j), device=dev)).float().mean())
            print(f"  {step:5d}  loss {float(loss):.4f}  held-out top-1 "
                  f"{100*top1:.2f}%  (chance {100/a.pool:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
