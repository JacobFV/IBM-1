"""control: can ANY model beat chance on image -> evoked EEG with this data?

both paired heads plateau at exactly skill 0, which is what a model does when it
gives up and predicts the conditional mean.  that has two very different causes --
the cortical path cannot carry stimulus-specific information, or the target as
prepared is not predictable from the stimulus at all -- and they call for opposite
fixes.  a plain convnet with no dynamics in it separates them:

  control beats chance  -> the target is fine, the cortical path is the bottleneck
  control also at chance -> the target as prepared carries no stimulus information,
                            and no amount of architecture will help

this is the cheapest decisive experiment available and it should have been run
before either head was built.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = "data/derived/things-paired"
ONSET, KEEP, DEC = 20, 50, 2


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("groupmean", "persubject"), default="groupmean",
                    help="groupmean averages the evoked response over 9 subjects, which "
                         "suppresses everything except the generic ERP shape -- and that "
                         "shape is identical for every image, so predicting the mean is "
                         "optimal by construction. persubject keeps one subject, where "
                         "image-specific structure survives at the cost of SNR")
    ap.add_argument("--subject", type=int, default=0)
    ap.add_argument("--split", choices=("contiguous", "random"), default="contiguous",
                    help="contiguous is by image index, and THINGS orders by concept, "
                         "so train and test hold DISJOINT object categories -- the model "
                         "must generalise to categories it has never seen. random shares "
                         "categories across the split, which is the weaker task the "
                         "decoding literature usually reports")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = np.load(f"{D}/images_training.npy", mmap_mode="r")
    if a.target == "persubject":
        _e = np.load(f"{D}/evoked_training_persubject.npy", mmap_mode="r")
        ev = _e[a.subject]
    else:
        ev = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    n = min(len(imgs), len(ev))
    ntr = int(n * 0.8)
    if a.split == "random":
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        tr_idx, te_idx = perm[:ntr], perm[ntr:]
    else:
        tr_idx, te_idx = np.arange(ntr), np.arange(ntr, n)

    samp = np.asarray(ev[np.sort(tr_idx)[::7]]).astype(np.float32)
    med = np.median(samp, 0)
    iqr = ((np.percentile(samp, 75, 0) - np.percentile(samp, 25, 0)) / 1.349).clip(1e-9)

    def tgt(i):
        y = (np.asarray(ev[i]).astype(np.float32) - med) / iqr
        return np.clip(y, -6, 6)[..., ONSET:ONSET + KEEP:DEC]

    base = float((tgt(np.sort(te_idx)[::13]) ** 2).mean())
    T = tgt(np.arange(0, 4)).shape[-1]
    C = ev.shape[1]
    print(f"[{a.target}] baseline {base:.4f} | target {C} channels x {T} samples | "
          f"{a.split} split: {len(tr_idx):,} train / {len(te_idx):,} test", flush=True)

    m = nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),
        nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
        nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
        nn.Flatten(), nn.Linear(128 * 8 * 8, 512), nn.GELU(),
        nn.Linear(512, C * T)).to(dev)
    print(f"control params {sum(p.numel() for p in m.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)

    for step in range(2001):
        i = tr_idx[np.random.randint(0, len(tr_idx), 64)]
        x = torch.from_numpy(np.ascontiguousarray(imgs[i])).to(dev)
        x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1
        y = torch.from_numpy(tgt(i)).to(dev).reshape(len(i), -1)
        loss = F.mse_loss(m(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0:
            with torch.no_grad():
                j = te_idx[np.random.randint(0, len(te_idx), 256)]
                xt = torch.from_numpy(np.ascontiguousarray(imgs[j])).to(dev)
                xt = (xt.permute(0, 3, 1, 2).float() / 127.5) - 1
                yt = torch.from_numpy(tgt(j)).to(dev).reshape(len(j), -1)
                h = float(F.mse_loss(m(xt), yt))
            print(f"  {step:5d}  train {float(loss):.4f}  held {h:.4f}  "
                  f"SKILL {1 - h / base:+.4f}", flush=True)


if __name__ == "__main__":
    main()
