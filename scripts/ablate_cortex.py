"""is the cortex load-bearing, or is it a delay line?

the pivotal test.  in every loop trained so far the encoder and the decoder are
both learned, and in the paired loop the readout is a linear lead field.  nothing
in that arrangement forces information to pass THROUGH the dynamics -- a model
that routed around them would still show a falling loss, and a rank-one cortical
state explaining 95% of MEG is exactly what being routed around looks like.

so: take a trained checkpoint, damage the cortex in four specific ways, and
measure what the loss does.  the arms are chosen so that each one removes a
different claim.

*frozen*      -- association embeddings reset to their initialization.  removes
                 everything LEARNED about cortico-cortical connectivity while
                 leaving the geometry, the E/I dynamics and both learned codecs
                 intact.  if loss does not move, the 32M association parameters
                 bought nothing.
*local only*  -- long-range edges severed, local k-NN kept.  removes the
                 possibility of cross-area and cross-modal communication while
                 leaving local recurrence.  this is the arm that tests §4b.
*no assoc*    -- association set to zero.  the sheet becomes 250k independent E/I
                 units with no lateral communication at all.
*bypass*      -- the dynamics skipped entirely, drive read straight out.  the
                 pure codec baseline.  **if the full model does not beat this,
                 the cortex is decorative and every conclusion drawn from these
                 runs is about an autoencoder.**
"""
from __future__ import annotations

import argparse, json
import numpy as np, torch, torch.nn.functional as F

import importlib.util
spec = importlib.util.spec_from_file_location(
    "ptrain", __file__.replace("ablate_cortex.py", "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--audio-frames", default="")
    ap.add_argument("--neural", default="")
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--out", default="out/ablation.json")
    a = ap.parse_args()

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = d.get("config") or d["model"] and {}
    cfg = d.get("config", {})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = d["model"]
    n_sites = sd["dyn.embed"].shape[0]; embed = sd["dyn.embed"].shape[1]
    k = sd["dyn.idx"].shape[1]
    mod = cfg.get("modality", "av")
    print(f"{a.ckpt}: {mod}, {n_sites:,} sites, embed {embed}, k {k}", flush=True)

    frames = np.load(a.frames, mmap_mode="r")
    coch = np.load(a.audio_frames, mmap_mode="r") if a.audio_frames else None
    neural = np.load(a.neural, mmap_mode="r") if a.neural else None

    dyn = P.CorticalDynamics(n_sites, embed, k, dev, long_range=cfg.get("long_range", .25))
    if mod == "av":
        model = P.AudioVisualLoop(dyn, n_bands=coch.shape[-1]).to(dev)
    elif mod == "paired":
        model = P.PairedNeuralLoop(dyn, n_bands=frames.shape[-1],
                                   n_sensors=neural.shape[-1]).to(dev)
    else:
        model = P.VideoLoop(dyn).to(dev)
    model.load_state_dict(sd); model.eval()

    H = cfg.get("horizon", 8); ctx = 25 if mod == "paired" else 8
    dyn_steps, dt = cfg.get("dyn_steps", 8), cfg.get("dt", 5e-3)
    bs = cfg.get("batch", 4)
    embed0 = torch.randn_like(dyn.embed) * 0.02          # the initialization
    local_k = k - int(k * cfg.get("long_range", .25))
    rng = np.random.default_rng(0)

    @torch.no_grad()
    def evaluate(arm):
        saved_e = dyn.embed.data.clone(); saved_geo = dyn.geo.clone()
        if arm == "frozen":
            dyn.embed.data = embed0.to(dev)
        elif arm == "local_only":
            g = dyn.geo.clone(); g[:, local_k:] = 0.0; dyn.geo.copy_(g)
        elif arm == "no_assoc":
            dyn.geo.zero_()
        tot, n = 0.0, 0
        for b in range(a.batches):
            r = np.random.default_rng(1000 + b)
            if mod == "paired":
                lim = min(len(frames), len(neural)) - H - 2
                i = r.integers(ctx, lim, size=bs)
                x = torch.from_numpy(np.stack([frames[j-ctx:j] for j in i])).float().to(dev)
                y = torch.from_numpy(np.ascontiguousarray(neural[i+H])).float().to(dev)
                if arm == "bypass":
                    drive = torch.zeros(bs, dyn.n, device=dev)
                    drive[:, model.off:model.off+model.port] = model.to_cortex(model.enc(x))
                    pred = model.lead(drive[:, model.read_idx.to(dev)])
                else:
                    pred, _ = model(x, dyn_steps, dt)
                loss = F.mse_loss(pred, y)
            else:
                lim = min(len(frames), len(coch) if coch is not None else len(frames)) - H - 2
                i = r.integers(ctx, lim, size=bs)
                xv = torch.from_numpy(np.ascontiguousarray(frames[i])).to(dev)
                yv = torch.from_numpy(np.ascontiguousarray(frames[i+H])).to(dev)
                xv = (xv.permute(0,3,1,2).float()/127.5)-1.0
                yv = (yv.permute(0,3,1,2).float()/127.5)-1.0
                xa = torch.from_numpy(np.stack([coch[j-ctx:j] for j in i])).float().to(dev)
                ya = torch.from_numpy(np.ascontiguousarray(coch[i+H])).float().to(dev)
                if arm == "bypass":
                    drive = torch.zeros(bs, dyn.n, device=dev)
                    drive[:, model.occ[0]:model.occ[1]] = model.v_in(model.v_enc(xv))
                    drive[:, model.tmp[0]:model.tmp[1]] = model.a_in(model.a_enc(xa))
                    pv = model.v_dec(model.v_out(drive[:, model.v_read[0]:model.v_read[1]]))
                    pa = model.a_dec(model.a_out(drive[:, model.a_read[0]:model.a_read[1]]))
                else:
                    pv, pa, _ = model(xv, xa, dyn_steps, dt)
                loss = F.mse_loss(pv, yv) + 0.5*F.mse_loss(pa, ya)
            tot += float(loss); n += 1
        dyn.embed.data = saved_e; dyn.geo.copy_(saved_geo)
        return tot / n

    res = {}
    for arm in ("full", "frozen", "local_only", "no_assoc", "bypass"):
        res[arm] = evaluate(arm)
        print(f"  {arm:12s} loss {res[arm]:.5f}", flush=True)

    base = res["full"]
    print("\n  arm           loss     vs full")
    for arm, v in res.items():
        print(f"  {arm:12s} {v:8.5f}  {100*(v-base)/base:+7.1f}%")
    verdict = ("LOAD-BEARING" if res["bypass"] > base * 1.10 and res["frozen"] > base * 1.02
               else "DECORATIVE -- the cortex is being routed around")
    print(f"\n  VERDICT: {verdict}")
    res["verdict"] = verdict
    import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
