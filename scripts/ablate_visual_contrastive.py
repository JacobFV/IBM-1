"""does the cortex contribute to visual retrieval, or is it a pass-through?

the contrastive head reaches 18.5% held-out top-1 against 0.5% chance -- 37x --
and a dynamics-free control on the same pairs reached 21.5%.  comparable numbers,
which admits two very different readings: the cortex carries the stimulus
structure nearly as well as a direct encoder, or the encoder and the readout do
all the work and the dynamics are a fixed transform the heads learned around.

magnitude has misled this project twice, so this asks by damage rather than by
inspection.  four arms, all evaluated on the same held-out pool:

*full*      -- the trained model.
*frozen*    -- association embeddings reset to initialisation.  removes what was
               learned about cortico-cortical connectivity, keeps the dynamics.
*no_assoc*  -- association zeroed.  the sheet becomes independent E/I units and
               the only path from the occipital port to the readout sites is
               whatever the drive projection itself covers.
*bypass*    -- dynamics skipped; the drive vector is read straight into the
               cortex head.  **if retrieval survives this, the dynamics are
               decorative for the visual term.**
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
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--out", default="out/ablation_visual.json")
    a = ap.parse_args()

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = d["config"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = d["model"]
    n_sites, embed = sd["dyn.embed"].shape
    k = sd["dyn.idx"].shape[1]

    imgs = np.load(f"{D}/images_training.npy", mmap_mode="r")
    ev = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    n = min(len(imgs), len(ev))
    ntr = int(n * 0.8)
    samp = np.asarray(ev[:ntr:7]).astype(np.float32)
    med = np.median(samp, 0)
    iqr = ((np.percentile(samp, 75, 0) - np.percentile(samp, 25, 0)) / 1.349).clip(1e-9)
    dec = cfg["decimate"]

    def eeg(i):
        y = (np.asarray(ev[i]).astype(np.float32) - med) / iqr
        return np.clip(y, -6, 6)[..., ONSET:ONSET + KEEP:dec]

    T = eeg(np.arange(4)).shape[-1]
    C = ev.shape[1]
    dyn = P.CorticalDynamics(n_sites, embed, k, dev, long_range=cfg["long_range"]).to(dev)
    model = P.VisualContrastiveLoop(dyn, n_sensors=C, n_times=T).to(dev)
    model.load_state_dict(sd)
    model.eval()
    print(f"{a.ckpt}: step {d['step']}, trained top-1 {100*d['top1']:.2f}%", flush=True)

    geo0 = dyn.geo.clone()
    emb0 = dyn.embed.data.clone()
    embR = (torch.randn_like(dyn.embed) * 0.02).to(dev)

    @torch.no_grad()
    def run(arm: str) -> tuple[float, float]:
        dyn.geo.copy_(geo0)
        dyn.embed.data.copy_(emb0)
        if arm == "frozen":
            dyn.embed.data.copy_(embR)
        elif arm == "no_assoc":
            dyn.geo.zero_()
        accs = []
        for r in range(a.repeats):
            rng = np.random.default_rng(700 + r)
            j = rng.integers(ntr, n - 1, a.pool)
            x = torch.from_numpy(np.ascontiguousarray(imgs[j])).to(dev)
            x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1
            y = torch.from_numpy(eeg(j)).to(dev)
            if arm == "bypass":
                drive = torch.zeros(len(j), dyn.n, device=dev)
                drive[:, :model.port] = model.to_cortex(model.enc(x))
                z = torch.nn.functional.normalize(
                    model.cortex_head(drive[:, model.read_idx.to(dev)]), dim=-1)
            else:
                z, _ = model.embed_image(x, substeps=cfg["substeps"], dt=cfg["dt"],
                                         n_steps=cfg.get("n_steps", 4))
            sim = z @ model.embed_eeg(y).T
            accs.append(float((sim.argmax(1) ==
                               torch.arange(len(j), device=dev)).float().mean()))
        dyn.geo.copy_(geo0)
        dyn.embed.data.copy_(emb0)
        return float(np.mean(accs)), float(np.std(accs))

    chance = 1.0 / a.pool
    res = {"chance": chance}
    for arm in ("full", "frozen", "no_assoc", "bypass"):
        m, sd_ = run(arm)
        res[arm] = {"top1": m, "sd": sd_}
        print(f"  {arm:10s} top-1 {100*m:5.2f}% +/- {100*sd_:.2f}  "
              f"({m/chance:.1f}x chance)", flush=True)

    full = res["full"]["top1"]
    print(f"\n  arm         top-1     x chance   retained")
    for arm in ("full", "frozen", "no_assoc", "bypass"):
        v = res[arm]["top1"]
        print(f"  {arm:10s} {100*v:6.2f}%  {v/chance:8.1f}   {100*v/full:6.1f}%")
    keep = 100 * res["bypass"]["top1"] / full
    verdict = ("DYNAMICS CONTRIBUTE -- bypassing loses most of the retrieval"
               if keep < 70 else
               "DYNAMICS DECORATIVE for retrieval -- the encoder and head suffice")
    print(f"\n  bypass retains {keep:.1f}% of full  ->  {verdict}", flush=True)
    res["verdict"] = verdict
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
