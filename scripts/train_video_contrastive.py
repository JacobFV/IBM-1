"""video continuation as DISCRIMINATION, not reconstruction.

pixel-space L2 has now failed twice on this task for one reason: it asks for the
conditional mean of the target.  predicting the frame, that mean is a blur and
loses to persistence by 2.2x; predicting the residual, that mean is zero and the
model degenerates to emitting nothing (residual 1.9% of true magnitude, cosine
+0.002, measured over 73 evaluations).  no amount of data or substrate moves
either -- 15.2 h did not, and the substrate is the part that works, since
bypassing it costs 4.41x.

this is the third branch to reach that conclusion, and the visual branch shows
the way out: waveform regression on THINGS-EEG2 peaked at skill +0.011 while
contrastive retrieval on the SAME pairs and the same encoder reached 53x chance.
same data, different question.  reconstruction demands every pixel; discrimination
demands only enough structure to tell the true future from a false one.

so: frame t drives the cortex, the dynamics run, and the resulting state is
aligned contrastively with an embedding of frame t+H.  negatives are other
positions IN THE SAME FILM, so the task cannot be solved by recognising the
scene -- it has to be solved by knowing what happens next.

**the baseline is persistence in embedding space** and it is computed on the same
batches: how well does frame t's OWN target-embedding retrieve frame t+H?  that
is what the model has to beat, and it is strong for the same reason pixel
persistence was strong.  a model that merely reproduces its input scores exactly
this and no more.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)


class TargetEnc(nn.Module):
    """the future frame's embedding.  separate from the context path on purpose --
    a shared encoder lets the model satisfy the objective by matching a frame to
    itself, which is the collapse this baseline exists to detect."""

    def __init__(self, dim=128):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.GELU(), nn.Linear(512, dim))

    def forward(self, x):
        return F.normalize(self.f(x), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/derived/pd-film")
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--dyn-steps", type=int, default=8)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout-films", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--ckpt", default="ckpt/video_contrastive.pt")
    ap.add_argument("--out", default="out/video_contrastive.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    films = sorted(glob.glob(f"{a.corpus}/*_frames.npy"))
    tr_f, te_f = films[:-a.holdout_films], films[-a.holdout_films:]
    TR = [np.load(f, mmap_mode="r") for f in tr_f]
    TE = [np.load(f, mmap_mode="r") for f in te_f]
    print(f"train {len(TR)} films ({sum(map(len,TR)):,} frames, "
          f"{sum(map(len,TR))/25/3600:.1f} h) | held out {len(te_f)} films "
          f"({sum(map(len,TE))/25/3600:.2f} h)", flush=True)
    print(f"negatives are other positions IN THE SAME FILM; chance = "
          f"{100.0/a.batch:.2f}%", flush=True)

    def draw(pool, m, rng):
        """one film per batch, m positions in it -- so negatives share the scene."""
        v = pool[rng.integers(len(pool))]
        i = rng.integers(0, len(v) - a.horizon - 1, m)
        f = lambda idx: ((torch.from_numpy(np.stack([np.asarray(v[j]) for j in idx]))
                          .to(dev).permute(0, 3, 1, 2).float() / 127.5) - 1.0)
        return f(i), f(i + a.horizon)

    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    model = P.VideoLoop(dyn).to(dev)
    # the cortical readout becomes an embedding rather than a frame
    head = nn.Sequential(nn.Linear(dyn.n // 8, 512), nn.GELU(),
                         nn.Linear(512, 128)).to(dev)
    tgt = TargetEnc().to(dev)
    temp = nn.Parameter(torch.tensor(0.07, device=dev))
    params = list(model.enc.parameters()) + list(model.to_cortex.parameters()) + \
             list(dyn.parameters()) + list(head.parameters()) + \
             list(tgt.parameters()) + [temp]
    tot = sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable {tot:,} ({dyn.embed.numel():,} association)", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)

    def context(x):
        b = x.shape[0]
        drive = torch.zeros(b, dyn.n, device=x.device)
        drive[:, :model.n_in] = model.to_cortex(model.enc(x))
        s = dyn.init_state(b, x.device)
        w = dyn.edge_weights()
        for _ in range(a.dyn_steps):
            s = dyn.step(s, drive, a.dt, w)
        return F.normalize(head(s[1][:, -dyn.n // 8:]), dim=-1), s

    log = {"config": vars(a), "n_params": tot, "chance": 1.0 / a.batch, "steps": []}
    best, t0 = -1e9, time.time()
    rng = np.random.default_rng(0)
    for step in range(a.steps + 1):
        x, y = draw(TR, a.batch, rng)
        z, s = context(x)
        zt = tgt(y)
        logits = z @ zt.T / temp.clamp(0.01, 1.0)
        lbl = torch.arange(len(x), device=dev)
        loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl))
        loss = loss + 1e-1 * P.viability_penalty(s[0])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], 1.0)
        opt.step()

        if step % a.eval_every == 0:
            accs, bases = [], []
            with torch.no_grad():
                r2 = np.random.default_rng(5000 + step)
                for _ in range(a.eval_batches):
                    xt, yt = draw(TE, a.batch, r2)
                    zc, st = context(xt)
                    zy = tgt(yt)
                    lb = torch.arange(len(xt), device=dev)
                    accs.append(float((( zc @ zy.T).argmax(1) == lb).float().mean()))
                    # persistence in embedding space: frame t's own target-embedding
                    bases.append(float(((tgt(xt) @ zy.T).argmax(1) == lb)
                                       .float().mean()))
                r_eff = P.effective_rank(st[1][:, ::max(dyn.n // 512, 1)].float())
            m, b = float(np.mean(accs)), float(np.mean(bases))
            sd_ = float(np.std(accs))
            log["steps"].append({"step": step, "loss": float(loss.detach()),
                                 "top1": m, "top1_sd": sd_, "persistence_top1": b,
                                 "r_eff": r_eff, "sec": round(time.time() - t0, 1)})
            flag = ""
            if m > best:
                best = m
                flag = "  <- best"
                os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "head": head.state_dict(),
                            "tgt": tgt.state_dict(), "step": step, "top1": m,
                            "persistence_top1": b, "config": vars(a)}, a.ckpt)
            mark = "  BEATS PERSISTENCE" if m > b + 2 * sd_ else ""
            print(f"{step:6d}  loss {float(loss):.4f}  top-1 {100*m:5.2f}% +/-{100*sd_:.2f}  "
                  f"persistence {100*b:5.2f}%  chance {100/a.batch:.2f}%  "
                  f"r_eff {r_eff:5.2f}  {time.time()-t0:5.0f}s{flag}{mark}", flush=True)

    log["best_top1"] = best
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
