"""is the cortex responsible for video continuation losing to persistence?

the multifilm run settled at 2.2x worse than persistence on held-out films with
the overfitting gone, so the failure is not data.  before that attaches to the
substrate, the same encoder and decoder have to be measured WITHOUT it.  magnitude
has misled this project twice; damage has not.

four arms, all on the same held-out films, all against the same persistence
baseline computed on the same batches:

  full      the trained model.
  frozen    association embeddings reset to init -- removes what was learned about
            cortico-cortical connectivity, keeps the dynamics.
  no_assoc  association zeroed; the sheet becomes independent E/I units.
  bypass    dynamics skipped entirely, the drive read straight into the decoder.
            `to_cortex` and the anterior readout are both n/8 wide, so this is a
            clean skip rather than a reshaped substitute.

**the informative case is bypass.**  if it also loses to persistence by roughly
2x, the cortex is not what is failing -- the frame-prediction setup is, and no
amount of substrate work fixes it.  if bypass is much worse, the dynamics are
carrying the task and the gap to persistence is a capacity or objective problem.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, os
import numpy as np
import torch
import torch.nn.functional as F

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="ckpt/video_multifilm.pt")
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--run-json", default="out/video_multifilm.json",
                    help="the run log, which records WHICH films were held out")
    ap.add_argument("--out", default="out/ablation_video_multifilm.json")
    a = ap.parse_args()

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, sd = d["config"], d["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_sites, embed = sd["dyn.embed"].shape
    k = sd["dyn.idx"].shape[1]
    H, ds, dt = cfg["horizon"], cfg["dyn_steps"], cfg["dt"]

    # cfg["holdout_films"] is the argparse COUNT, not the list; the film paths
    # live in the run log beside the checkpoint.
    import json as _j
    films = _j.load(open(a.run_json))["holdout_films"]
    TE = [np.load(f, mmap_mode="r") for f in films]
    print(f"{a.ckpt}: step {d['step']}, skill {d['skill_vs_persistence']:+.4f}")
    print(f"held out: {[os.path.basename(f) for f in films]}\n",
          flush=True)

    dyn = P.CorticalDynamics(n_sites, embed, k, dev,
                             long_range=cfg["long_range"]).to(dev)
    model = P.VideoLoop(dyn).to(dev)
    model.load_state_dict(sd)
    model.eval()
    geo0, emb0 = dyn.geo.clone(), dyn.embed.data.clone()
    embR = (torch.randn_like(dyn.embed) * 0.02).to(dev)

    def draw(rng):
        xs, ys = [], []
        for _ in range(a.batch):
            v = TE[rng.integers(len(TE))]
            i = int(rng.integers(0, len(v) - H - 1))
            xs.append(np.asarray(v[i])); ys.append(np.asarray(v[i + H]))
        f = lambda t: (torch.from_numpy(np.stack(t)).to(dev)
                       .permute(0, 3, 1, 2).float() / 127.5) - 1.0
        return f(xs), f(ys)

    @torch.no_grad()
    def run(arm):
        dyn.geo.copy_(geo0); dyn.embed.data.copy_(emb0)
        if arm == "frozen":
            dyn.embed.data.copy_(embR)
        elif arm == "no_assoc":
            dyn.geo.zero_()
        mse, per = [], []
        for r in range(a.batches):
            x, y = draw(np.random.default_rng(500 + r))
            if arm == "bypass":
                # build the drive and read it AT THE READOUT SITES without running
                # the dynamics -- the same shape the trained path sees, minus the
                # substrate.  the old form fed to_cortex's n/8 output straight into
                # from_cortex, which was only width-compatible while the readout
                # was also n/8; since the readout fix it is 4096 sites spanning the
                # sheet, and the arm has to skip the dynamics without also changing
                # the readout geometry it is being compared against.
                drive = torch.zeros(len(x), dyn.n, device=dev)
                drive[:, :model.n_in] = model.to_cortex(model.enc(x))
                pred = model.dec(model.from_cortex(
                    drive[:, model.read_idx.to(dev)]))
            else:
                pred, _ = model(x, ds, dt)
            mse.append(float(F.mse_loss(pred, y)))
            per.append(float(F.mse_loss(x, y)))
        dyn.geo.copy_(geo0); dyn.embed.data.copy_(emb0)
        return float(np.mean(mse)), float(np.std(mse)), float(np.mean(per))

    res = {}
    for arm in ("full", "frozen", "no_assoc", "bypass"):
        m, s_, p_ = run(arm)
        res[arm] = {"mse": m, "sd": s_, "persistence": p_,
                    "skill_vs_persistence": 1 - m / p_}
        print(f"  {arm:10s} MSE {m:.5f} +/- {s_:.5f}   persistence {p_:.5f}   "
              f"skill {1-m/p_:+.4f}", flush=True)

    f, b = res["full"]["mse"], res["bypass"]["mse"]
    print(f"\n  bypass / full = {b/f:.2f}x")
    if abs(b / f - 1.0) < 0.15:
        v = ("THE CORTEX IS NOT THE PROBLEM -- bypassing it changes almost nothing, "
             "so the frame-prediction setup loses to persistence on its own")
    elif b > f:
        v = ("the dynamics ARE carrying the task -- bypass is materially worse, so "
             "the gap to persistence is capacity or objective, not the substrate")
    else:
        v = ("bypass BEATS the full model -- the dynamics are actively hurting this "
             "task")
    print(f"  -> {v}", flush=True)
    res["verdict"] = v
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
