#!/usr/bin/env python3
"""The paired MEG term as RETRIEVAL through the cortex, not regression from it.

Eight measurements on 2026-09-11 closed the regression route and specified this one. In order:
the signal is in the corpus (ridge +0.0466), it survives the dynamics (+0.0400), it is reachable
at the head's own rank-64 constraint (+0.0400), it can be installed there and verified
(+0.039779, to 9.9e-06) -- and 600 steps of the head's own MSE objective then drive it to +0.0136
**while the training loss falls by half**. The objective discards the signal because +0.04 of
correlation is worth almost nothing in least-squares terms against a target dominated by variance
no stimulus can predict.

And the replacement has measured material rather than borrowed material: cutting the ridge's
predictions into windows and retrieving each window's own MEG from a pool of 32 reaches **10.0%
top-1, 3.19x chance, at W=125 (0.5 s)**, with a shuffled-pairing control on chance and a
self-retrieval ceiling at 100%.

THE BAR, FIXED HERE BEFORE THE RUN. This arm must beat **3.19x chance** -- the LINEAR ridge's
retrieval, measured on the same held-out stretch, same pool size, same window, same averaging
over 200 draws. A cortical model that does not beat a ridge has not earned its 50 million
parameters. Reporting a number above chance is not the result; beating 3.19x is.

AND THE EXPECTATION, also fixed here. THINGS-EEG2 contrastive retrieval reaches 43x chance; this
corpus offers 3.19x from a linear map. **Anything near 43x here should be disbelieved before it is
celebrated** -- it would mean a leak, not a breakthrough.

CONTROLS, both run in the same script so neither can be skipped:
  shuffled  the identical arm trained on SHUFFLED stimulus-MEG pairings. It must not exceed
            chance. If it does, the evaluation leaks and the intact number is void. This is the
            control that can fail, and it is the one that matters.
  bypass    the identical arm with the cortical state replaced by the drive that would have
            entered it -- the dynamics removed, nothing else changed. Says whether the sheet is
            load-bearing for retrieval, which regression could never answer because regression
            never worked at all.

Held-out rows are the reserved tail, CONTIGUOUS, and the artefact rows found today
(1,619,681-1,627,268, a 30.3 s saturating burst) are excluded from training as everywhere else.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
RIDGE_BAR_X_CHANCE = 3.19       # docs/LOG.md, pool 32, W=125, 200 draws


class MEGEncoder(nn.Module):
    """A window of measured MEG -> embedding. Deliberately small: this is the side that must NOT
    become the model. If it has the capacity to solve the task alone the cortex is decorative."""

    def __init__(self, n_sensors, window, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_sensors, 64, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(64, 64, 9, stride=4, padding=4), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * math.ceil(window / 16), hidden), nn.GELU(),
            nn.Linear(hidden, dim))

    def forward(self, y):            # (b, window, sensors)
        return self.net(y.transpose(1, 2))


class CorticalStimulusEncoder(nn.Module):
    """cochleagram context -> auditory port -> dynamics -> embedding.

    The same path `PairedNeuralLoop` uses, up to the readout: encode the cochleagram, drive the
    auditory port, run the dynamics, read sites. Only the head differs -- a projection into the
    embedding instead of a lead field -- because the objective is what is being changed here, not
    the substrate.
    """

    def __init__(self, dyn, n_bands, ctx, dim, hidden=256, read_sites=4096, bypass=False):
        super().__init__()
        self.dyn, self.ctx, self.bypass = dyn, ctx, bypass
        self.port = dyn.n // 8
        self.off = dyn.n // 3                       # auditory: temporal, not occipital
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(n_bands * ctx, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.to_cortex = nn.Linear(hidden, self.port)
        self.register_buffer("read_idx", torch.linspace(0, dyn.n - 1, read_sites).long())
        self.proj = nn.Sequential(nn.Linear(read_sites if not bypass else self.port, hidden),
                                  nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, coch, n_steps, dt):
        b = coch.shape[0]
        port = self.to_cortex(self.enc(coch))
        if self.bypass:
            # THE DYNAMICS REMOVED AND NOTHING ELSE CHANGED: the projection reads the drive that
            # would have entered the sheet, at the same width, through the same head shape.
            return self.proj(port)
        drive = torch.zeros(b, self.dyn.n, device=coch.device)
        drive[:, self.off:self.off + self.port] = port
        s = self.dyn.init_state(b, coch.device)
        w = self.dyn.edge_weights()
        for _ in range(n_steps):
            s = self.dyn.step(s, drive, dt, w)
        return self.proj(s[1][:, self.read_idx])


def info_nce(a, b, temp):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = a @ b.T / temp
    lab = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-stim", required=True)
    ap.add_argument("--paired-neural", required=True)
    ap.add_argument("--sites", type=int, default=150_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=125)
    ap.add_argument("--window", type=int, default=125, help="MEG window; 0.5 s is where retrieval works")
    ap.add_argument("--dyn-steps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=25_000)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--exclude", default="1619681:1627268")
    ap.add_argument("--arm", choices=("intact", "shuffled", "bypass"), default="intact")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--ckpt", default="ckpt/paired_contrastive.pt")
    ap.add_argument("--out", default="out/paired_contrastive.json")
    a = ap.parse_args()

    import pretrain_video_loop as P
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    X = np.load(a.paired_stim, mmap_mode="r")
    Y = np.load(a.paired_neural, mmap_mode="r")
    sc = a.paired_neural.replace("meg_250hz", "meg_scale")
    megsc = np.load(sc) if os.path.exists(sc) else None
    n = min(len(X), len(Y)) - 2
    lo = int(n * (1.0 - a.holdout))

    def meg_window(starts):
        out = np.stack([np.asarray(Y[s:s + a.window]) for s in starts]).astype(np.float32)
        if megsc is not None:
            out = np.clip((out - megsc[0]) / megsc[1], -6, 6)
        return np.ascontiguousarray(out, dtype=np.float32)

    def coch_ctx(starts):
        return np.stack([np.asarray(X[s - a.ctx:s]) for s in starts]).astype(np.float32)

    excl = [tuple(int(v) for v in r.split(":")) for r in a.exclude.split(",") if r.strip()]
    pool_idx = np.arange(a.ctx, lo - a.window)
    for x0, x1 in excl:
        pool_idx = pool_idx[~((pool_idx >= x0 - a.window) & (pool_idx < x1))]
    ev0 = lo + a.ctx
    ev = np.arange(ev0, min(ev0 + a.n_eval, n - a.window), a.window)   # non-overlapping windows
    print(f"arm={a.arm}  train pool {len(pool_idx):,} rows | {len(ev):,} held-out windows of "
          f"{a.window} ({a.window/250:.2f} s) from {ev0:,}", flush=True)
    print(f"THE BAR: the linear ridge reaches {RIDGE_BAR_X_CHANCE}x chance here. "
          f"Beating it is the result; being above chance is not.", flush=True)

    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    enc_s = CorticalStimulusEncoder(dyn, X.shape[-1], a.ctx, a.dim,
                                    bypass=(a.arm == "bypass")).to(dev)
    enc_m = MEGEncoder(Y.shape[-1], a.window, a.dim).to(dev)
    params = [p for p in enc_s.parameters()] + list(enc_m.parameters())
    if a.arm == "bypass":
        params = [p for nm, p in enc_s.named_parameters() if not nm.startswith("dyn.")] + list(enc_m.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    print(f"stimulus encoder {sum(p.numel() for p in enc_s.parameters()):,} | "
          f"meg encoder {sum(p.numel() for p in enc_m.parameters()):,}", flush=True)

    rng = np.random.default_rng(0)
    erng = np.random.default_rng(20260911)

    @torch.no_grad()
    def evaluate():
        enc_s.eval(); enc_m.eval()
        A, B = [], []
        for s in range(0, len(ev), 16):
            q = ev[s:s + 16]
            A.append(enc_s(torch.from_numpy(coch_ctx(q)).to(dev), a.dyn_steps, a.dt))
            B.append(enc_m(torch.from_numpy(meg_window(q)).to(dev)))
        A = F.normalize(torch.cat(A), dim=1); B = F.normalize(torch.cat(B), dim=1)
        hits = shuf_hits = 0
        for _ in range(a.draws):
            idx = erng.choice(len(A), size=a.pool, replace=False)
            sim = A[idx] @ B[idx].T
            hits += int((sim.argmax(1).cpu().numpy() == np.arange(a.pool)).sum())
            sim2 = A[idx] @ B[erng.permutation(idx)].T
            shuf_hits += int((sim2.argmax(1).cpu().numpy() == np.arange(a.pool)).sum())
        enc_s.train(); enc_m.train()
        tot = a.draws * a.pool
        return hits / tot, shuf_hits / tot

    chance = 1.0 / a.pool
    log = {"config": vars(a), "ridge_bar_x_chance": RIDGE_BAR_X_CHANCE, "chance": chance,
           "evals": []}
    t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            top1, shuf = evaluate()
            log["evals"].append(dict(step=step, top1=top1, shuffled_pool=shuf,
                                     x_chance=top1 / chance))
            print(f"  step {step:5d}  top-1 {top1:6.2%} = {top1/chance:5.2f}x chance   "
                  f"(shuffled pool {shuf:5.2%}, chance {chance:.2%})   "
                  f"bar {RIDGE_BAR_X_CHANCE}x   {time.time()-t0:6.0f}s", flush=True)
        if step == a.steps:
            break
        i = rng.choice(pool_idx, size=a.batch, replace=False)
        xs = torch.from_numpy(coch_ctx(i)).to(dev)
        j = rng.permutation(i) if a.arm == "shuffled" else i
        ym = torch.from_numpy(meg_window(j)).to(dev)
        loss = info_nce(enc_s(xs, a.dyn_steps, a.dt), enc_m(ym), a.temp)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"    {step:5d} loss {float(loss):.4f}", flush=True)
        if a.ckpt and a.save_every and step and step % a.save_every == 0:
            os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
            torch.save({"dyn": dyn.state_dict(), "enc_s": enc_s.state_dict(),
                        "enc_m": enc_m.state_dict(), "step": step, "arm": a.arm,
                        "config": vars(a)}, a.ckpt)

    os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
    torch.save({"dyn": dyn.state_dict(), "enc_s": enc_s.state_dict(),
                "enc_m": enc_m.state_dict(), "step": a.steps, "arm": a.arm,
                "config": vars(a)}, a.ckpt)
    best = max(e["x_chance"] for e in log["evals"])
    log["best_x_chance"] = best
    log["beats_ridge"] = bool(best > RIDGE_BAR_X_CHANCE)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"\n  best {best:.2f}x chance against the ridge's {RIDGE_BAR_X_CHANCE}x -> "
          f"{'BEATS the linear baseline' if best > RIDGE_BAR_X_CHANCE else 'DOES NOT beat it'}")
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
