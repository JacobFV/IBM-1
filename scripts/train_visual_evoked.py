"""train image -> evoked EEG on THINGS-EEG2.

the first measured brain the visual pathway has ever had.  every prior visual term
was self-supervised with a learned encoder AND decoder, which cannot by itself
force information through the cortex; here the target is what a real occipital
cortex produced when shown this image, so the trajectory has to be right.

the target is a 64-channel response over -0.2 to +0.79 s at 100 Hz, averaged over
repetitions and over 9 subjects.  splits are contiguous over IMAGE index, not
random: images from the same THINGS concept sit together in the ordering, so a
random split would put near-duplicates of a training image in the test set and
report memorisation as generalisation.

skill is reported against the zero baseline throughout, per ibm/evaluate.py.  a
raw MSE here would be meaningless -- the evoked response has a small amplitude and
an unscaled loss looks excellent while predicting nothing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)

D = "data/derived/things-paired"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", type=int, default=150_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--dt", type=float, default=2e-2)     # output interval after decimation
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--viability-weight", type=float, default=1e-1)
    ap.add_argument("--lead-rank", type=int, default=32)
    ap.add_argument("--substeps", type=int, default=4,
                    help="integrator steps per output sample; dt/substeps must stay "
                         "well under the 15 ms membrane constant")
    ap.add_argument("--decimate", type=int, default=2,
                    help="keep every Nth EEG sample. 100 -> 50 Hz still resolves P1 "
                         "and N1 (~100 and ~170 ms) and halves a cost dominated by the "
                         "time dimension: this head runs the dynamics once per output "
                         "sample where every other head runs them once per example")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--crop-ms", type=float, default=500.0,
                    help="keep 0..crop_ms of the epoch. the stored window is -200 to "
                         "+790 ms, but this head's cost is LINEAR in the number of "
                         "output samples -- it runs the dynamics once per sample -- and "
                         "the pre-stimulus baseline is by construction signal-free while "
                         "everything after ~500 ms is late/endogenous. cropping to the "
                         "evoked window is a 2x saving that discards no evoked response")
    ap.add_argument("--ckpt", default="ckpt/visual_evoked.pt")
    ap.add_argument("--upload-every", type=int, default=1000)
    ap.add_argument("--out", default="out/visual_evoked.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    imgs = np.load(f"{D}/images_training.npy", mmap_mode="r")
    ev = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    n = min(len(imgs), len(ev))
    # the stored epoch is -0.2..+0.79 s at 100 Hz.  index 20 is stimulus onset.
    onset_i = 20
    keep = int(round(a.crop_ms / 10.0))
    n_sensors = ev.shape[1]
    n_times = len(range(onset_i, min(onset_i + keep, ev.shape[2]), a.decimate))
    print(f"images {imgs.shape} | evoked {ev.shape} | pairs {n:,}", flush=True)

    # a robust scale, computed on the TRAIN portion only.  a scale fitted over the
    # whole array leaks test statistics into training, which is a small leak and
    # exactly the kind this project has been bitten by.
    n_train = int(n * 0.8)
    samp = np.asarray(ev[:n_train:7]).astype(np.float32)
    med = np.median(samp, axis=0)
    iqr = (np.percentile(samp, 75, axis=0) - np.percentile(samp, 25, axis=0)) / 1.349
    scale = np.clip(iqr, 1e-9, None).astype(np.float32)
    med = med.astype(np.float32)

    def target(i):
        y = (np.asarray(ev[i]).astype(np.float32) - med) / scale
        return np.clip(y, -6.0, 6.0)[..., onset_i:onset_i + keep:a.decimate]

    base_zero = float((target(np.arange(n_train, n, 13)) ** 2).mean())
    print(f"held-out zero baseline: {base_zero:.4f}  "
          f"(train 0-{n_train:,}, test {n_train:,}-{n:,}, contiguous by image)", flush=True)

    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    model = P.VisualEvokedLoop(dyn, n_sensors=n_sensors, n_times=n_times,
                               lead_rank=a.lead_rank).to(dev)
    tot = sum(p.numel() for p in model.parameters())
    print(f"params {tot:,} ({dyn.embed.numel():,} association)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    log = {"config": vars(a), "n_params": tot, "baseline_zero": base_zero, "steps": []}
    t0 = time.time()

    for step in range(a.steps):
        i = np.random.randint(0, n_train, size=a.batch)
        x = torch.from_numpy(np.ascontiguousarray(imgs[i])).to(dev)
        x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1.0
        y = torch.from_numpy(target(i)).to(dev)
        pred, s = model(x, n_times, a.dt, substeps=a.substeps)
        recon = F.mse_loss(pred, y)
        viab = P.viability_penalty(s[0])
        loss = recon + a.viability_weight * viab
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.eval_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                j = np.random.randint(n_train, n - 1, size=min(32, a.batch * 2))
                xt = torch.from_numpy(np.ascontiguousarray(imgs[j])).to(dev)
                xt = (xt.permute(0, 3, 1, 2).float() / 127.5) - 1.0
                yt = torch.from_numpy(target(j)).to(dev)
                pt, st = model(xt, n_times, a.dt, substeps=a.substeps)
                held = float(F.mse_loss(pt, yt))
                r_eff = P.effective_rank(st[1][:, ::max(dyn.n // 512, 1)].float())
            skill = 1.0 - held / base_zero
            rec = {"step": step, "train_mse": float(recon.detach()),
                   "heldout_mse": held, "skill_vs_zero": skill,
                   "r_eff": r_eff, "v_absmax": float(s[0].abs().max()),
                   "sec": round(time.time() - t0, 1)}
            log["steps"].append(rec)
            print(f"{step:5d}  train {float(recon):.4f}  held {held:.4f}  "
                  f"SKILL {skill:+.3f}  r_eff {r_eff:5.2f}  "
                  f"|v| {float(s[0].abs().max()):6.1f}  {time.time()-t0:6.0f}s", flush=True)
            if not math.isfinite(float(loss)):
                print("DIVERGED", flush=True)
                break

        if a.ckpt and a.upload_every and step and step % a.upload_every == 0:
            os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step,
                        "config": vars(a)}, a.ckpt)
            try:
                from ibm.release import CheckpointName, sidecar, upload
                nm = CheckpointName(modality="ve", sites=a.sites, embed=a.embed,
                                    degree=a.k, objective="evoked",
                                    viability_weight=a.viability_weight, step=step)
                meta = sidecar(nm, geometry="spherical shell, area-matched",
                               n_params=tot, n_assoc=dyn.embed.numel(),
                               metrics=log["steps"][-1], config=vars(a))
                print("  uploaded", upload(a.ckpt, meta), flush=True)
            except Exception as e:
                print(f"  upload failed: {e}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
