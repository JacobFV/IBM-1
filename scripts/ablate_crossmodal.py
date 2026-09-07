"""does the learned occipito-temporal coupling actually carry anything?

`xmod` -- the mean |weight| on occipital->temporal edges -- has climbed from 0.138
to 0.80 since the long-range prior was flattened.  that is a magnitude, and
magnitude has already been wrong here once: at 4.4x the random baseline, severing
every one of those edges cost **+0.1% loss**.  large and functionally inert.

so this measures contribution rather than size.  four arms, each removing a
different claim, evaluated on held-out batches:

*full*        -- the trained model.
*sever xmod*  -- zero ONLY the occipital<->temporal edges.  everything else,
                 including all other long-range edges, is untouched.  this is the
                 arm that isolates cross-modal communication from long-range
                 communication in general.
*sever far*   -- zero every long-range edge.  the arm that failed before.
*drop audio*  -- the audio port receives no drive at all.  if severing the
                 cross-modal edges costs the same as removing the audio entirely,
                 those edges ARE the channel.

the gate for s4c is the second arm: severing the cross-modal edges specifically
must cost materially more than the 0.1% the previous configuration paid.
"""
from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np
import torch
import torch.nn.functional as F

sp = importlib.util.spec_from_file_location(
    "ptrain", __file__.replace("ablate_crossmodal.py", "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--audio-frames", required=True)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--out", default="out/ablation_crossmodal.json")
    a = ap.parse_args()

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = d.get("config", {})
    sd = d.get("av") or d.get("model")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    n_sites, embed = sd["dyn.embed"].shape
    k = sd["dyn.idx"].shape[1]
    H = cfg.get("horizon", 8)
    ds, dt = cfg.get("dyn_steps", 8), cfg.get("dt", 5e-3)
    bs = cfg.get("batch", 4)
    lr_frac = cfg.get("long_range", 0.25)

    frames = np.load(a.frames, mmap_mode="r")
    coch = np.load(a.audio_frames, mmap_mode="r")
    dyn = P.CorticalDynamics(n_sites, embed, k, dev, long_range=lr_frac)
    model = P.AudioVisualLoop(dyn, n_bands=coch.shape[-1]).to(dev)
    model.load_state_dict(sd)
    model.eval()
    print(f"{a.ckpt}: {n_sites:,} sites, k={k}, step {d.get('step')}", flush=True)
    print(f"  xmod as reported by the trainer: {model.cross_modal_weight():.4f}", flush=True)

    n_loc = k - int(k * lr_frac)
    occ_lo, occ_hi = model.occ
    tmp_lo, tmp_hi = model.tmp
    idx = dyn.idx
    src = torch.arange(n_sites, device=dev).unsqueeze(1).expand_as(idx)
    # an edge is cross-modal if it joins the two ports in either direction
    o2t = ((src >= occ_lo) & (src < occ_hi) & (idx >= tmp_lo) & (idx < tmp_hi))
    t2o = ((src >= tmp_lo) & (src < tmp_hi) & (idx >= occ_lo) & (idx < occ_hi))
    xmask = o2t | t2o
    print(f"  cross-modal edges: {int(xmask.sum()):,} of {idx.numel():,} "
          f"({100*float(xmask.float().mean()):.3f}%)", flush=True)

    geo0 = dyn.geo.clone()

    @torch.no_grad()
    def run(arm: str) -> float:
        dyn.geo.copy_(geo0)
        if arm == "sever_xmod":
            g = dyn.geo.clone(); g[xmask] = 0.0; dyn.geo.copy_(g)
        elif arm == "sever_far":
            g = dyn.geo.clone(); g[:, n_loc:] = 0.0; dyn.geo.copy_(g)
        tot, n = 0.0, 0
        for b in range(a.batches):
            r = np.random.default_rng(2000 + b)
            lim = min(len(frames), len(coch)) - H - 2
            i = r.integers(8, lim, size=bs)
            xv = torch.from_numpy(np.ascontiguousarray(frames[i])).to(dev)
            yv = torch.from_numpy(np.ascontiguousarray(frames[i + H])).to(dev)
            xv = (xv.permute(0, 3, 1, 2).float() / 127.5) - 1.0
            yv = (yv.permute(0, 3, 1, 2).float() / 127.5) - 1.0
            xa = torch.from_numpy(np.stack([coch[j - 8:j] for j in i])).float().to(dev)
            ya = torch.from_numpy(np.ascontiguousarray(coch[i + H])).float().to(dev)
            drop = "audio" if arm == "drop_audio" else ""
            pv, pa, _ = model(xv, xa, ds, dt, drop=drop)
            tot += float(F.mse_loss(pv, yv) + 0.5 * F.mse_loss(pa, ya))
            n += 1
        dyn.geo.copy_(geo0)
        return tot / n

    res = {}
    for arm in ("full", "sever_xmod", "sever_far", "drop_audio"):
        res[arm] = run(arm)
        print(f"  {arm:12s} {res[arm]:.5f}", flush=True)

    base = res["full"]
    print("\n  arm            loss      vs full")
    for arm, v in res.items():
        print(f"  {arm:12s} {v:9.5f}  {100*(v-base)/base:+8.2f}%")
    dx = 100 * (res["sever_xmod"] - base) / base
    verdict = ("LOAD-BEARING -- the cross-modal edges carry signal"
               if dx > 5.0 else
               "INERT -- large weights, no contribution (as before the prior fix)")
    print(f"\n  s4c gate: severing cross-modal edges costs {dx:+.2f}%  ->  {verdict}")
    res["verdict"] = verdict
    res["xmod_cost_pct"] = dx
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
