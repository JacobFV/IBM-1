"""evaluate a visual checkpoint on the DESIGNATED THINGS-EEG2 test set.

every visual number this programme has reported was measured on the last 20% of
the TRAINING split.  that is a legitimate held-out set -- the split is contiguous
and THINGS-EEG2 orders training images by concept, so it is concept-disjoint --
but it is not the set the corpus was built to be scored on, and it is noisier:
training images carry 4 repetitions each against the test set's **80**, which is
a factor of sqrt(20) in the target's SNR.

the test set also removes the last sampling artefact.  it holds exactly 200
images, so a retrieval pool of 200 is the WHOLE SET: chance is 1/200 by
construction, there are no pools to average, and the measurement is deterministic
rather than an estimate with a standard deviation.  ledger row 9 cannot recur
here.

`images_test.npy` did not exist until now -- the evoked test responses had been
built and their images never were, so the designated set had never been usable.
it is built in the order `image_metadata.npy` declares, which is the ledger row 8
discipline: take an ordering from the dataset that defines it.

the four ablation arms are reported alongside, because a retrieval number without
them says nothing about whether the substrate earned it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import torch

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)

D = "data/derived/things-paired"
ONSET, KEEP = 20, 50


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="ckpt/visual_contrastive_v2.pt")
    ap.add_argument("--out", default="out/things_test_eval.json")
    a = ap.parse_args()

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, sd = d["config"], d["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_sites, embed = sd["dyn.embed"].shape
    k = sd["dyn.idx"].shape[1]

    # the normalisation must come from TRAINING, not from the test set: fitting
    # the centre and scale on the evaluation data leaks it.
    tr = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    ntr = int(min(len(tr), len(np.load(f"{D}/images_training.npy", mmap_mode="r"))) * 0.8)
    samp = np.asarray(tr[:ntr:7]).astype(np.float32)
    med = np.median(samp, 0)
    iqr = ((np.percentile(samp, 75, 0) - np.percentile(samp, 25, 0)) / 1.349).clip(1e-9)

    imgs = np.load(f"{D}/images_test.npy")
    ev = np.load(f"{D}/evoked_test_groupmean.npy")
    n = len(imgs)
    dec = cfg["decimate"]
    y = np.clip((ev.astype(np.float32) - med) / iqr, -6, 6)[..., ONSET:ONSET + KEEP:dec]
    T, C = y.shape[-1], ev.shape[1]

    dyn = P.CorticalDynamics(n_sites, embed, k, dev, long_range=cfg["long_range"]).to(dev)
    model = P.VisualContrastiveLoop(dyn, n_sensors=C, n_times=T).to(dev)
    model.load_state_dict(sd)
    model.eval()
    print(f"{a.ckpt}: step {d['step']}", flush=True)
    print(f"designated test set: {n} images, {C} ch x {T} samples, 80 repetitions "
          f"each.  pool = the whole set, chance = {100.0/n:.2f}%\n", flush=True)

    x = torch.from_numpy(np.ascontiguousarray(imgs)).to(dev)
    x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1
    yt = torch.from_numpy(y).to(dev)
    geo0, emb0 = dyn.geo.clone(), dyn.embed.data.clone()
    embR = (torch.randn_like(dyn.embed) * 0.02).to(dev)
    lbl = torch.arange(n, device=dev)

    @torch.no_grad()
    def arm(name: str) -> dict:
        dyn.geo.copy_(geo0); dyn.embed.data.copy_(emb0)
        if name == "frozen":
            dyn.embed.data.copy_(embR)
        elif name == "no_assoc":
            dyn.geo.zero_()
        if name == "bypass":
            drive = torch.zeros(n, dyn.n, device=dev)
            drive[:, :model.port] = model.to_cortex(model.enc(x))
            z = torch.nn.functional.normalize(
                model.cortex_head(drive[:, model.read_idx.to(dev)]), dim=-1)
        else:
            z, _ = model.embed_image(x, substeps=cfg["substeps"], dt=cfg["dt"],
                                     n_steps=cfg.get("n_steps", 4))
        sim = z @ model.embed_eeg(yt).T
        rank = (sim > sim.gather(1, lbl[:, None])).sum(1)
        dyn.geo.copy_(geo0); dyn.embed.data.copy_(emb0)
        return {"top1": float((rank == 0).float().mean()),
                "top5": float((rank < 5).float().mean()),
                "median_rank": float(rank.float().median()) + 1}

    res = {"n": n, "chance": 1.0 / n, "step": d["step"], "ckpt": a.ckpt}
    for name in ("full", "frozen", "no_assoc", "bypass"):
        res[name] = arm(name)
        r = res[name]
        print(f"  {name:10s} top-1 {100*r['top1']:5.2f}%  top-5 {100*r['top5']:5.2f}%  "
              f"median rank {r['median_rank']:5.1f}  ({r['top1']*n:.1f}x chance)",
              flush=True)

    full = res["full"]["top1"]
    print(f"\n  arm         top-1    retained")
    for name in ("full", "frozen", "no_assoc", "bypass"):
        v = res[name]["top1"]
        print(f"  {name:10s} {100*v:6.2f}%  {100*v/full:7.1f}%")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
