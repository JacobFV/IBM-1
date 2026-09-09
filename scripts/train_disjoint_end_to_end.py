"""does the concentrated kernel TRAIN end to end, where the unmodified one cannot?

the gradient measurement says it should.  the encoder sees 3.292e-01 of gradient
when the readout sits at the driven port and 1.883e-03 when it sits in a disjoint
region -- 175x lost crossing the sheet -- and concentrating the long-range budget
recovers most of that, leaving only 7.5x.  that is a mechanism for every
end-to-end motor failure in this repo: training was starved of gradient, not of
data and not of a correct objective.

but a gradient norm is not a training run.  it says the gradient is THERE, 21x
more of it; it does not say the encoder learns anything with it.  this is the run.

image -> occipital port -> the sheet -> a PRECENTRAL-ONLY readout, contrastively
aligned with measured EEG, trained END TO END: the encoder, the kernel and the
head all receive gradient through the dynamics.  that is the configuration every
previous attempt failed at, and the only difference between the two arms is the
kernel's edge configuration.

  base      the trained kernel as it stands
  aniso4d   the same weights, long-range budget concentrated onto 4 edges per
            site at >=120 mm separation with long_gain 8

the prediction is directional and falsifiable: base stays at or near chance and
aniso4d rises.  if BOTH rise, the earlier failures were something else and the
gradient story does not explain them.  if NEITHER rises, transport was never the
binding constraint for end-to-end training and the whole line needs rethinking.
either of those is a result and is reported as one.

the ports are verified disjoint before anything runs -- a readout that can see
the region it is measuring transport from measures nothing, which is the error
that produced ledger row 20.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

D = "data/derived/things-paired"
ONSET, KEEP = 20, 50


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/ibm1_curriculum16.pt")
    ap.add_argument("--configs",
                    default="base:2.0,1.0,0,1.0,0.0;aniso4d:2.0,8.0,4,1.0,120.0")
    ap.add_argument("--drive-region", default="occipital")
    ap.add_argument("--read-region", default="precentral")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dyn-steps", type=int, default=4)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=2e-2)
    ap.add_argument("--decimate", type=int, default=2)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--eval-pools", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/disjoint_end_to_end.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = d
    if "dyn.idx" not in sd:
        # three checkpoint shapes exist in ckpt/: the graph is at the top level,
        # inside a named head, or -- for anything written by
        # train_visual_contrastive -- under "model".  a reader that knows two of
        # them silently refuses the third, which is how transfer_sweep.py once
        # left the curriculum run out of its own evaluation.
        cands = [d.get("model")] + list((d.get("heads") or {}).values())
        for h in cands:
            if isinstance(h, dict) and "dyn.idx" in h:
                sd = h; break
    if "dyn.idx" not in sd:
        raise SystemExit(f"{a.ckpt}: no association graph; refusing to redraw it")
    emb = sd.get("dyn.embed", d.get("dyn.embed"))
    n_sites, e_dim = emb.shape
    k = sd["dyn.idx"].shape[1]

    imgs = np.load(f"{D}/images_training.npy", mmap_mode="r")
    ev = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    n = min(len(imgs), len(ev)); ntr = int(n * 0.8)
    s_ = np.asarray(ev[:ntr:7]).astype(np.float32)
    med = np.median(s_, 0)
    iqr = ((np.percentile(s_, 75, 0) - np.percentile(s_, 25, 0)) / 1.349).clip(1e-9)

    def eeg(i):
        y = (np.asarray(ev[i]).astype(np.float32) - med) / iqr
        return np.clip(y, -6, 6)[..., ONSET:ONSET + KEEP:a.decimate]

    T = eeg(np.arange(2)).shape[-1]; C = ev.shape[1]
    print(f"{a.ckpt}: {n_sites:,} sites, k={k} | {n:,} pairs, "
          f"train 0-{ntr:,} | EEG {C}ch x {T}", flush=True)

    cfgs = {}
    for spec in a.configs.split(";"):
        nm, f = spec.split(":"); v = [float(t) for t in f.split(",")]
        cfgs[nm] = dict(tanh_slope=v[0], long_gain=v[1], long_topm=int(v[2]),
                        local_gain=v[3] if len(v) > 3 else 1.0,
                        long_min_dist=v[4] if len(v) > 4 else 0.0)

    def build(cfg):
        """identical initialisation for every arm; only the kernel config differs."""
        torch.manual_seed(a.seed)
        dyn = P.CorticalDynamics(n_sites, e_dim, k, dev).to(dev)
        with torch.no_grad():
            dyn.embed.copy_(emb.to(dev))
            for nm in ("idx", "geo", "pos"):
                if f"dyn.{nm}" in sd:
                    getattr(dyn, nm).copy_(sd[f"dyn.{nm}"].to(dev))
            for nm in ("w_ee", "w_ei", "w_assoc", "a_gain", "log_len"):
                if f"dyn.{nm}" in sd:
                    getattr(dyn, nm).copy_(sd[f"dyn.{nm}"].to(dev))
        for nm, v in cfg.items():
            setattr(dyn, nm, v)
        port = P.region_index(dyn.pos, a.drive_region).to(dev)
        read = P.region_index(dyn.pos, a.read_region).to(dev)
        inter = len(np.intersect1d(port.cpu().numpy(), read.cpu().numpy()))
        if inter:
            raise SystemExit(f"ports overlap in {inter} sites -- not a transport test")
        torch.manual_seed(a.seed)
        enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.GELU(), nn.Linear(256, len(port))).to(dev)
        head = nn.Sequential(nn.Linear(len(read), 512), nn.GELU(),
                             nn.Linear(512, a.dim)).to(dev)
        eh = nn.Sequential(
            nn.Conv1d(C, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(), nn.Flatten(),
            nn.Linear(128 * ((T + 1) // 2), 512), nn.GELU(),
            nn.Linear(512, a.dim)).to(dev)
        return dyn, enc, head, eh, port, read

    def batch(lo, hi, m, rng):
        i = rng.integers(lo, hi, m)
        x = torch.from_numpy(np.ascontiguousarray(imgs[i])).to(dev)
        return (x.permute(0, 3, 1, 2).float() / 127.5) - 1, \
            torch.from_numpy(eeg(i)).to(dev)

    chance = 1.0 / a.pool
    res = {"ckpt": a.ckpt, "chance": chance, "config": vars(a), "arms": {}}
    for name, cfg in cfgs.items():
        dyn, enc, head, eh, port, read = build(cfg)
        temp = nn.Parameter(torch.tensor(0.07, device=dev))
        params = (list(enc.parameters()) + list(head.parameters())
                  + list(eh.parameters()) + [dyn.embed, temp])
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
        print(f"\n### {name}: long_gain={cfg['long_gain']} topm={cfg['long_topm']} "
              f"min_dist={cfg['long_min_dist']} | drive {len(port)} -> read "
              f"{len(read)} sites, disjoint", flush=True)

        def embed_img(x):
            s = dyn.init_state(x.shape[0], dev)
            w = dyn.edge_weights()
            drive = torch.zeros(x.shape[0], dyn.n, device=dev).index_copy(
                1, port, enc(x))
            h = a.dt / a.substeps
            for _ in range(a.dyn_steps):
                for _ in range(a.substeps):
                    s = dyn.step(s, drive, h, w)
            return F.normalize(head(s[1][:, read]), dim=-1), s

        @torch.no_grad()
        def ev_(step):
            accs = []
            rng = np.random.default_rng(90_000 + step)
            for _ in range(a.eval_pools):
                x, y = batch(ntr, n - 1, a.pool, rng)
                z, _ = embed_img(x)
                sim = z @ F.normalize(eh(y), dim=-1).T
                accs.append(float((sim.argmax(1) ==
                            torch.arange(len(x), device=dev)).float().mean()))
            return float(np.mean(accs)), float(np.std(accs))

        hist, best, t0 = [], 0.0, time.time()
        rng = np.random.default_rng(a.seed)
        for step in range(a.steps + 1):
            x, y = batch(0, ntr, a.batch, rng)
            z, s = embed_img(x)
            lg = z @ F.normalize(eh(y), dim=-1).T / temp.clamp(0.01, 1.0)
            lbl = torch.arange(len(x), device=dev)
            loss = 0.5 * (F.cross_entropy(lg, lbl) + F.cross_entropy(lg.T, lbl))
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = float(torch.sqrt(sum((p.grad ** 2).sum()
                                      for p in enc.parameters() if p.grad is not None)))
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % a.eval_every == 0:
                m, sdv = ev_(step)
                best = max(best, m)
                hist.append({"step": step, "loss": float(loss), "top1": m,
                             "sd": sdv, "x_chance": m / chance, "enc_grad": gn})
                print(f"  {step:5d}  loss {float(loss):.4f}  top-1 {100*m:5.2f}%"
                      f"+/-{100*sdv:4.2f} ({m/chance:5.1f}x chance)  "
                      f"|grad_enc| {gn:.3e}  {time.time()-t0:5.0f}s", flush=True)
                res["arms"][name] = {"history": hist, "best_top1": best,
                                     "best_x_chance": best / chance}
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                json.dump(res, open(a.out, "w"), indent=2)

    if len(res["arms"]) == 2:
        nms = list(res["arms"])
        b, c = res["arms"][nms[0]]["best_x_chance"], res["arms"][nms[1]]["best_x_chance"]
        print(f"\n  {nms[0]} best {b:.1f}x chance | {nms[1]} best {c:.1f}x chance",
              flush=True)
        if c > 2 and b < 2:
            v = (f"CONFIRMED: {nms[1]} trains end to end and {nms[0]} does not. "
                 f"transport was the binding constraint on end-to-end training.")
        elif c > 2 and b > 2:
            v = (f"BOTH train. the earlier end-to-end failures were NOT explained "
                 f"by transport, and the gradient story does not account for them.")
        else:
            v = (f"NEITHER trains. transport is not the binding constraint for "
                 f"end-to-end training and this line needs rethinking.")
        print(f"  -> {v}", flush=True)
        res["verdict"] = v
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
