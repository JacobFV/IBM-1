"""is the MEG head using the cortex, or leaning on its lead field again?

`r_meg` has sat at 1.01-1.05 for the whole joint run while `r_av` stays at 2.2-3.4
on the same shared substrate.  a rank-one cortical state feeding a 306-channel
prediction is the exact signature that made the standalone paired head suspect,
and constraining the lead field to rank 64 was supposed to have removed it.

so this measures contribution rather than reading rank.  four arms:

*full*        -- the trained joint model's paired head.
*frozen*      -- association embeddings reset to their initialisation.  removes
                 everything learned about cortico-cortical connectivity while
                 leaving the dynamics, the ports and the lead field intact.
*no_assoc*    -- association zeroed.  the sheet becomes independent E/I units.
*bypass*      -- the dynamics skipped; drive read straight through the lead field.
                 **if this is close to full, the cortex is decorative for MEG and
                 the rank-64 constraint did not fix what it was meant to.**
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--neural", required=True)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--out", default="out/ablation_meg.json")
    a = ap.parse_args()

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = d.get("config", {})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dyn_sd = d["dyn"]
    pr_sd = d["paired"]
    n_sites, embed = dyn_sd["embed"].shape
    k = dyn_sd["idx"].shape[1]

    stim = np.load(a.stim, mmap_mode="r")
    neur = np.load(a.neural, mmap_mode="r")
    sc = np.load(a.neural.replace("meg_250hz", "meg_scale"))

    dyn = P.CorticalDynamics(n_sites, embed, k, dev,
                             long_range=cfg.get("long_range", .25)).to(dev)
    dyn.load_state_dict(dyn_sd)
    model = P.PairedNeuralLoop(dyn, n_bands=stim.shape[-1],
                               n_sensors=neur.shape[-1]).to(dev)
    # the paired head's own weights sit under a "dyn." prefix for the shared part
    model.load_state_dict(pr_sd)
    model.eval()
    print(f"{a.ckpt}: {n_sites:,} sites, step {d.get('step')}", flush=True)

    ctx = 125
    ds, dt = cfg.get("dyn_steps", 8), cfg.get("dt", 5e-3)
    bs = cfg.get("batch", 8)
    embed0 = (torch.randn_like(dyn.embed) * 0.02).to(dev)
    geo0 = dyn.geo.clone()
    emb0 = dyn.embed.data.clone()

    # the baseline every number is quoted against
    r = np.random.default_rng(0)
    lim = min(len(stim), len(neur)) - 2
    ys = []
    for b in range(a.batches):
        j = np.random.default_rng(500 + b).integers(ctx, lim, size=bs)
        yn = np.clip((np.ascontiguousarray(neur[j]) - sc[0]) / sc[1], -6, 6)
        ys.append(yn.astype(np.float32))
    base_zero = float((np.concatenate(ys) ** 2).mean())
    print(f"  zero baseline: {base_zero:.4f}", flush=True)

    @torch.no_grad()
    def run(arm: str) -> float:
        dyn.geo.copy_(geo0)
        dyn.embed.data.copy_(emb0)
        if arm == "frozen":
            dyn.embed.data.copy_(embed0)
        elif arm == "no_assoc":
            dyn.geo.zero_()
        tot, n = 0.0, 0
        for b in range(a.batches):
            j = np.random.default_rng(500 + b).integers(ctx, lim, size=bs)
            x = torch.from_numpy(np.stack([stim[q - ctx:q] for q in j])).float().to(dev)
            yn = np.clip((np.ascontiguousarray(neur[j]) - sc[0]) / sc[1], -6, 6)
            y = torch.from_numpy(yn.astype(np.float32)).to(dev)
            if arm == "bypass":
                drive = torch.zeros(x.shape[0], dyn.n, device=dev)
                drive[:, model.off:model.off + model.port] = model.to_cortex(model.enc(x))
                pred = model.lead_v(model.lead_u(drive[:, model.read_idx.to(dev)]))
            else:
                pred, _ = model(x, ds, dt)
            tot += float(F.mse_loss(pred, y))
            n += 1
        dyn.geo.copy_(geo0)
        dyn.embed.data.copy_(emb0)
        return tot / n

    res = {"baseline_zero": base_zero}
    for arm in ("full", "frozen", "no_assoc", "bypass"):
        res[arm] = run(arm)
        print(f"  {arm:10s} mse {res[arm]:.5f}   skill {1 - res[arm]/base_zero:+.3f}",
              flush=True)

    full = res["full"]
    print("\n  arm         mse       skill     vs full")
    for arm in ("full", "frozen", "no_assoc", "bypass"):
        v = res[arm]
        print(f"  {arm:10s} {v:8.5f}  {1 - v/base_zero:+7.3f}  {100*(v-full)/full:+8.2f}%")
    dbp = 100 * (res["bypass"] - full) / full
    verdict = ("LOAD-BEARING for MEG" if dbp > 10
               else "DECORATIVE for MEG -- the lead field is doing the work")
    print(f"\n  bypassing the cortex costs {dbp:+.2f}%  ->  {verdict}")
    res["verdict"] = verdict
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
