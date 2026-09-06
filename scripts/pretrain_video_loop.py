"""large-scale pretraining: raw video -> cortical dynamics -> next frame.

STATE.md 7b. a frame is encoded into cortical drive, the IBM dynamics are evolved
for one frame interval, and the resulting cortical state is decoded to the next
frame.  the brain is in the middle of a self-supervised video model, so raw
naturalistic video becomes an optimization source for cortical parameters -- which
is the only regime in which 10^5 PER_SITE embeddings are identifiable at all.

**what is being learned is the fine cortico-cortical connectivity.**
ARCHITECTURE.md's division of labour is that tractography constrains the coarse
modular graph and the fine graph is learned; `ibm.processes.neural`'s association
weight is the declared factorization of exactly that

    w_ij  =  M[pi(i), pi(j)]  x  exp(-d_ij / l)  x  sigma(<e_i, e_j>)
             ^ tractography      ^ geometry        ^ LEARNED, per site

and this script optimizes the third factor.  the embeddings are the parameters;
the geometry and the parcel-scale prior are held.

three things are deliberately NOT the loss:

*effective rank* and *the count of distinct metastable sets* are tracked and
reported and never optimized.  STATE.md 7d: predictive loss alone is a capture
curriculum -- it is minimized by exploiting what the model can already represent,
so it deepens existing basins and installs no new ones.  if loss falls while these
do not rise, the run is capturing rather than teaching, and holding them out of
the loss is what keeps them diagnostic (ONTOLOGY.md 7).

*the fan-in normalization is not a tunable.*  `|L(0)| = 7218` was measured before
it, three orders of magnitude above unity, so without it the dynamics diverge
before any gradient is meaningful (STATE.md 4.10).

the encoder and decoder are learned jointly rather than being a held TRIBEv2.
that is fitting the brain to an encoder, and it is legitimate HERE because this
stage is a pretraining structuring of the weights, not the final fit -- the
measured-recording likelihoods in CURRICULUM.md stages 2 and 5 are what anchor
the result to a brain.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# the cortical sheet
# ---------------------------------------------------------------------------

def cortical_sites(n: int, device, seed: int = 0):
    """positions on a folded sheet, and a k-NN association support over them.

    a real materialization reads `cortical_surface`; this uses a spherical shell
    with the measured white-surface area so that distances and therefore the
    exp(-d/l) prior are in millimetres and not arbitrary units.  the learned
    factor is what this run is about and it is indifferent to the substitution --
    but the substitution is recorded rather than hidden.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    # measured white-surface area 202,437 mm^2 -> radius of the equivalent sphere
    radius = math.sqrt(202437.0 / (4.0 * math.pi))
    z = torch.rand(n, generator=g) * 2 - 1
    theta = torch.rand(n, generator=g) * 2 * math.pi
    r = torch.sqrt(1 - z * z)
    pos = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], 1) * radius
    return pos.to(device)


def knn_edges(pos, k: int, chunk: int = 4096):
    """k nearest neighbours, chunked so the N x N distance matrix is never formed."""
    n = pos.shape[0]
    idx = torch.empty(n, k, dtype=torch.long, device=pos.device)
    dist = torch.empty(n, k, device=pos.device)
    for i in range(0, n, chunk):
        d = torch.cdist(pos[i:i + chunk], pos)
        dd, ii = torch.topk(d, k + 1, largest=False)
        idx[i:i + chunk] = ii[:, 1:]
        dist[i:i + chunk] = dd[:, 1:]
    return idx, dist


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class CorticalDynamics(nn.Module):
    """E/I population dynamics with a learned per-site association kernel."""

    def __init__(self, n_sites: int, embed_dim: int, k: int, device,
                 length_scale_mm: float = 40.0):
        super().__init__()
        self.n, self.k = n_sites, k
        pos = cortical_sites(n_sites, device)
        idx, dist = knn_edges(pos, k)
        self.register_buffer("idx", idx)
        # the geometric prior, held: exp(-d/l), normalized by fan-in so that the
        # total drive onto a node is O(1) rather than O(k).
        self.register_buffer("geo", torch.exp(-dist / length_scale_mm) / k)

        # THE PARAMETERS.  one embedding per site; the learned factor of w_ij.
        self.embed = nn.Parameter(torch.randn(n_sites, embed_dim) * 0.02)
        self.log_len = nn.Parameter(torch.tensor(math.log(length_scale_mm)))

        # E/I population parameters, initialized at the declared priors
        self.w_ee = nn.Parameter(torch.tensor(0.30))
        self.w_ei = nn.Parameter(torch.tensor(0.20))
        self.w_assoc = nn.Parameter(torch.tensor(0.50))
        self.tau_m = 0.015
        self.tau_a = 0.30          # STATE.md 4.10: puts the SO at 0.533 Hz, in band
        self.a_gain = nn.Parameter(torch.tensor(0.10))
        self.v_half, self.slope, self.r_max = -55.0, 4.0, 100.0
        self.e_rest, self.e_rev = -65.0, -70.0

    def association(self, r):
        """message passing on the learned kernel.  r: (B, N)."""
        e = F.normalize(self.embed, dim=-1)
        sim = (e.unsqueeze(1) * e[self.idx]).sum(-1)          # (N, k)
        w = self.geo * torch.sigmoid(4.0 * sim)               # (N, k)
        return (r[:, self.idx] * w).sum(-1)                   # (B, N)

    def rate(self, v):
        return self.r_max * torch.sigmoid((v - self.v_half) / self.slope)

    def step(self, s, drive, dt):
        v, r, a, gi = s
        r_inf = self.rate(v)
        assoc = self.association(r)
        g_e = self.w_ee * r / self.r_max + self.w_assoc * assoc / self.r_max
        dv = (-(v - self.e_rest) + 20.0 * g_e - a + drive) / self.tau_m
        # conductance-based shunting, LINEAR in g_i (STATE.md 4.11)
        dv = dv - (v - self.e_rev) * gi.clamp_min(0.0) / self.tau_m
        dr = (r_inf - r) / 5e-3
        da = (self.a_gain * r_inf - a) / self.tau_a
        dgi = (self.w_ei * r / self.r_max - gi) / 8e-3
        return (v + dt * dv, r + dt * dr, a + dt * da, gi + dt * dgi)

    def init_state(self, b, device):
        z = torch.zeros(b, self.n, device=device)
        return (torch.full_like(z, self.e_rest), z, z.clone(), z.clone())


class VideoLoop(nn.Module):
    """frame -> cortical drive -> dynamics -> cortical state -> next frame."""

    def __init__(self, dyn: CorticalDynamics, img: int = 64, hidden: int = 256):
        super().__init__()
        self.dyn, self.img = dyn, img
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),      # 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),     # 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),    # 8
            nn.Flatten(), nn.Linear(128 * 8 * 8, hidden), nn.GELU())
        # drive reaches a posterior subset -- the occipital port
        self.n_in = dyn.n // 8
        self.to_cortex = nn.Linear(hidden, self.n_in)
        self.from_cortex = nn.Linear(dyn.n // 8, hidden)
        self.dec = nn.Sequential(
            nn.Linear(hidden, 128 * 8 * 8), nn.GELU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))

    def forward(self, frame, n_steps: int, dt: float):
        b = frame.shape[0]
        h = self.enc(frame)
        drive = torch.zeros(b, self.dyn.n, device=frame.device)
        drive[:, :self.n_in] = self.to_cortex(h)
        s = self.dyn.init_state(b, frame.device)
        for _ in range(n_steps):
            s = self.dyn.step(s, drive, dt)
        read = s[1][:, -self.dyn.n // 8:]          # anterior readout
        return self.dec(self.from_cortex(read)), s


# ---------------------------------------------------------------------------
# the diagnostics that are NOT the loss
# ---------------------------------------------------------------------------

def effective_rank(x):
    """(tr C)^2 / tr(C^2) -- ONTOLOGY.md's expansion signal."""
    x = x - x.mean(0, keepdim=True)
    c = (x.T @ x) / max(x.shape[0] - 1, 1)
    t1 = torch.diagonal(c).sum()
    t2 = (c * c).sum()
    return float((t1 * t1 / t2.clamp_min(1e-12)).item())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", type=int, default=100_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dyn-steps", type=int, default=8)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--frames", default="/home/brandonin/Documents/win-data/derived/"
                                        "koyaanisqatsi-full/frames_64x64.npy")
    ap.add_argument("--out", default="out/pretrain_video.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    frames = np.load(a.frames, mmap_mode="r")
    print(f"frames {frames.shape} from {a.frames}", flush=True)

    dyn = CorticalDynamics(a.sites, a.embed, a.k, dev).to(dev)
    model = VideoLoop(dyn).to(dev)
    n_assoc = dyn.embed.numel()
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"association embeddings: {n_assoc:,}   total trainable: {n_tot:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n_frames = frames.shape[0]
    log = {"config": vars(a), "n_params": n_tot, "n_assoc": n_assoc, "steps": []}
    t0 = time.time()

    for step in range(a.steps):
        i = np.random.randint(0, n_frames - 2, size=a.batch)
        x = torch.from_numpy(np.ascontiguousarray(frames[i])).to(dev)
        y = torch.from_numpy(np.ascontiguousarray(frames[i + 1])).to(dev)
        x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1.0
        y = (y.permute(0, 3, 1, 2).float() / 127.5) - 1.0

        pred, s = model(x, a.dyn_steps, a.dt)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 25 == 0 or step == a.steps - 1:
            with torch.no_grad():
                r_eff = effective_rank(s[1][:, ::max(dyn.n // 512, 1)].float())
                vmax = float(s[0].abs().max())
            rec = {"step": step, "loss": float(loss), "r_eff": r_eff,
                   "v_absmax": vmax, "grad_norm": float(gn),
                   "sec": round(time.time() - t0, 1)}
            log["steps"].append(rec)
            print(f"{step:5d}  loss {float(loss):.5f}  r_eff {r_eff:7.2f}  "
                  f"|v|max {vmax:8.1f}  gn {float(gn):7.3f}  "
                  f"{time.time()-t0:6.0f}s", flush=True)
            if not math.isfinite(float(loss)):
                print("DIVERGED", flush=True); break

    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
