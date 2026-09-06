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
                 length_scale_mm: float = 40.0, long_range: float = 0.25):
        super().__init__()
        self.n, self.k = n_sites, k
        pos = cortical_sites(n_sites, device)
        n_far = int(k * long_range)
        idx, dist = knn_edges(pos, k - n_far)
        if n_far:
            # patchy long-range association fibres.  a pure k-NN graph is a local
            # sheet, and on a local sheet occipital and temporal sites are simply
            # not connected -- so no amount of training could associate them.
            # cortical association fibres are long-range and patchy, and this is
            # the minimal declaration of that: a fraction of each node's budget
            # spent on distant partners drawn uniformly, whose weight the learned
            # factor is then free to keep or discard.
            far = torch.randint(0, n_sites, (n_sites, n_far), device=device)
            far_d = (pos[far] - pos[:, None, :]).norm(dim=-1)
            idx = torch.cat([idx, far], 1)
            dist = torch.cat([dist, far_d], 1)
        self.register_buffer("pos", pos)
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


class AudioLoop(nn.Module):
    """cochleagram frame -> cortical drive -> dynamics -> next cochleagram frame.

    the same loop as `VideoLoop` with a different port.  the input is the output
    of `transduction:gammatone_cochleagram` -- ERB-spaced bands, phase-blind --
    so the pretraining signal enters through the representation the model's own
    cochlea produces rather than through a spectrogram chosen for convenience.
    drive reaches a temporal-lobe subset instead of an occipital one; the cortical
    parameters in between are SHARED with the video loop, which is the whole point
    of there being no standard materialization (STATE.md 7c).
    """

    def __init__(self, dyn: CorticalDynamics, n_bands: int = 64, ctx: int = 8,
                 hidden: int = 256):
        super().__init__()
        self.dyn, self.n_bands, self.ctx = dyn, n_bands, ctx
        self.enc = nn.Sequential(
            nn.Flatten(), nn.Linear(n_bands * ctx, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.n_in = dyn.n // 8
        self.to_cortex = nn.Linear(hidden, self.n_in)
        self.from_cortex = nn.Linear(dyn.n // 8, hidden)
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_bands))

    def forward(self, ctx, n_steps: int, dt: float):
        b = ctx.shape[0]
        h = self.enc(ctx)
        drive = torch.zeros(b, self.dyn.n, device=ctx.device)
        # the auditory port sits in the middle of the sheet, not the posterior pole
        off = self.dyn.n // 3
        drive[:, off:off + self.n_in] = self.to_cortex(h)
        s = self.dyn.init_state(b, ctx.device)
        for _ in range(n_steps):
            s = self.dyn.step(s, drive, dt)
        read = s[1][:, -self.dyn.n // 8:]
        return self.dec(self.from_cortex(read)), s


class AudioVisualLoop(nn.Module):
    """one cortex, two ports, two predictions -- and the association between them.

    the video loop drives an occipital port and the audio loop a temporal one.
    THE CORTICAL PARAMETERS BETWEEN THEM ARE THE SAME TENSOR.  so a model trained
    here has to explain both streams with one association kernel, and the only way
    to do that better than two independent models is to use the fact that the
    streams are correlated -- which is what learning an occipito-temporal
    association means.

    this is why the audio has to be the movie's OWN soundtrack.  pairing these
    frames with an unrelated audiobook would present two independent streams and
    the correct thing to learn would be that vision and hearing do not interact.

    `cross_modal_weight` reports the mean learned association between the two
    ports, and it is the number that says whether anything was actually learned
    across modalities rather than in each separately.
    """

    def __init__(self, dyn: CorticalDynamics, img: int = 64, n_bands: int = 64,
                 ctx: int = 8, hidden: int = 256):
        super().__init__()
        self.dyn, self.n_bands, self.ctx = dyn, n_bands, ctx
        n = dyn.n
        self.port = n // 8
        self.occ = (0, self.port)                       # occipital: visual drive
        self.tmp = (n // 3, n // 3 + self.port)         # temporal: auditory drive
        self.v_read = (n - self.port, n)                # visual readout
        self.a_read = (n // 2, n // 2 + self.port)      # auditory readout

        self.v_enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden), nn.GELU())
        self.a_enc = nn.Sequential(nn.Flatten(), nn.Linear(n_bands * ctx, hidden),
                                   nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.v_in = nn.Linear(hidden, self.port)
        self.a_in = nn.Linear(hidden, self.port)
        self.v_out = nn.Linear(self.port, hidden)
        self.a_out = nn.Linear(self.port, hidden)
        self.v_dec = nn.Sequential(
            nn.Linear(hidden, 128 * 8 * 8), nn.GELU(), nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))
        self.a_dec = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Linear(hidden, n_bands))

    def forward(self, frame, coch_ctx, n_steps: int, dt: float, drop: str = ""):
        b = frame.shape[0]
        drive = torch.zeros(b, self.dyn.n, device=frame.device)
        if drop != "video":
            drive[:, self.occ[0]:self.occ[1]] = self.v_in(self.v_enc(frame))
        if drop != "audio":
            drive[:, self.tmp[0]:self.tmp[1]] = self.a_in(self.a_enc(coch_ctx))
        s = self.dyn.init_state(b, frame.device)
        for _ in range(n_steps):
            s = self.dyn.step(s, drive, dt)
        r = s[1]
        v = self.v_dec(self.v_out(r[:, self.v_read[0]:self.v_read[1]]))
        a = self.a_dec(self.a_out(r[:, self.a_read[0]:self.a_read[1]]))
        return v, a, s

    @torch.no_grad()
    def cross_modal_weight(self):
        """mean learned association from the occipital port to the temporal one.

        the quantity the joint materialization exists to produce.  it is read off
        the learned kernel rather than inferred from behaviour, so it says
        directly whether the two lobes became coupled.
        """
        e = F.normalize(self.dyn.embed, dim=-1)
        occ = e[self.occ[0]:self.occ[1]]
        tmp = e[self.tmp[0]:self.tmp[1]]
        m = min(2048, occ.shape[0], tmp.shape[0])
        sim = occ[:m] @ tmp[:m].T
        return float(torch.sigmoid(4.0 * sim).mean())


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

def viability_penalty(v, lo=-90.0, hi=50.0):
    """how far the membrane potential is outside the range a neuron can occupy.

    ONTOLOGY.md §7 defines the alignment relation as viability-manifold
    containment and asks whether minimising the loss drives the model outside its
    own viability set.  measured on run v1: it does.  |v|max drifted 64 -> 430 mV
    over 375 steps while the loss fell, because nothing stopped the encoder from
    driving cortex arbitrarily hard and a 430 mV membrane predicts frames just
    fine.  it is not a brain, so the association weights learned under it mean
    nothing.

    this is the check working, not a surprise: an objective is parasitic on its
    substrate exactly when lowering the loss requires physiology the substrate
    could not sustain.  the penalty is the regularizer that makes the objective
    mutualistic instead, and it is one-sided -- zero cost anywhere inside the
    range, so it constrains nothing the model is entitled to do.
    """
    return (F.relu(v - hi) ** 2 + F.relu(lo - v) ** 2).mean()


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
    ap.add_argument("--viability-weight", type=float, default=1e-1,
                    help="ONTOLOGY.md §7: cost of leaving the physiological range")
    ap.add_argument("--modality", choices=("video", "audio", "av"), default="video")
    ap.add_argument("--horizon", type=int, default=8,
                    help="predict t+H, not t+1.  at 25 fps consecutive frames barely "
                         "differ, so a 1-step target is close to an identity map and "
                         "is minimized by COLLAPSING the representation -- measured: "
                         "effective rank fell 2.45 -> 1.04 while loss fell.  a longer "
                         "horizon forces the dynamics to do work (STATE.md 7d)")
    ap.add_argument("--audio-frames", default="")
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--upload-every", type=int, default=1000)
    ap.add_argument("--ckpt", default="", help="where to save weights")
    ap.add_argument("--out", default="out/pretrain_video.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    frames = np.load(a.frames, mmap_mode="r")
    print(f"frames {frames.shape} from {a.frames}", flush=True)

    coch = np.load(a.audio_frames, mmap_mode="r") if a.audio_frames else None
    dyn = CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    if a.modality == "av":
        model = AudioVisualLoop(dyn, n_bands=coch.shape[-1]).to(dev)
    elif a.modality == "audio":
        model = AudioLoop(dyn, n_bands=frames.shape[-1]).to(dev)
    else:
        model = VideoLoop(dyn).to(dev)
    n_assoc = dyn.embed.numel()
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"association embeddings: {n_assoc:,}   total trainable: {n_tot:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n_frames = frames.shape[0]
    log = {"config": vars(a), "n_params": n_tot, "n_assoc": n_assoc, "steps": []}
    t0 = time.time()

    for step in range(a.steps):
        H, ctx = a.horizon, 8
        if a.modality == "av":
            lim = min(n_frames, coch.shape[0]) - H - 2
            i = np.random.randint(ctx, lim, size=a.batch)
            xv = torch.from_numpy(np.ascontiguousarray(frames[i])).to(dev)
            yv = torch.from_numpy(np.ascontiguousarray(frames[i + H])).to(dev)
            xv = (xv.permute(0, 3, 1, 2).float() / 127.5) - 1.0
            yv = (yv.permute(0, 3, 1, 2).float() / 127.5) - 1.0
            xa = torch.from_numpy(np.stack([coch[j - ctx:j] for j in i])).float().to(dev)
            ya = torch.from_numpy(np.ascontiguousarray(coch[i + H])).float().to(dev)
            pv, pa, s = model(xv, xa, a.dyn_steps, a.dt)
            recon = F.mse_loss(pv, yv) + 0.5 * F.mse_loss(pa, ya)
        elif a.modality == "audio":
            i = np.random.randint(ctx, n_frames - H - 2, size=a.batch)
            x = torch.from_numpy(np.stack([frames[j - ctx:j] for j in i])).float().to(dev)
            y = torch.from_numpy(np.ascontiguousarray(frames[i + H])).float().to(dev)
            pred, s = model(x, a.dyn_steps, a.dt)
            recon = F.mse_loss(pred, y)
        else:
            i = np.random.randint(0, n_frames - H - 2, size=a.batch)
            x = torch.from_numpy(np.ascontiguousarray(frames[i])).to(dev)
            y = torch.from_numpy(np.ascontiguousarray(frames[i + H])).to(dev)
            x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1.0
            y = (y.permute(0, 3, 1, 2).float() / 127.5) - 1.0
            pred, s = model(x, a.dyn_steps, a.dt)
            recon = F.mse_loss(pred, y)
        viab = viability_penalty(s[0])
        loss = recon + a.viability_weight * viab
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 25 == 0 or step == a.steps - 1:
            with torch.no_grad():
                r_eff = effective_rank(s[1][:, ::max(dyn.n // 512, 1)].float())
                vmax = float(s[0].abs().max())
            rec = {"step": step, "loss": float(loss.detach()),
                   "recon": float(recon.detach()), "viability": float(viab.detach()),
                   "r_eff": r_eff, "v_absmax": vmax, "grad_norm": float(gn),
                   "sec": round(time.time() - t0, 1)}
            log["steps"].append(rec)
            xm = (model.cross_modal_weight() if a.modality == "av" else float("nan"))
            rec["cross_modal"] = xm
            print(f"{step:5d}  recon {float(recon):.5f}  viab {float(viab):8.3f}  "
                  f"r_eff {r_eff:7.2f}  |v|max {vmax:7.1f}  xmod {xm:.4f}  "
                  f"{time.time()-t0:6.0f}s", flush=True)
            if not math.isfinite(float(loss)):
                print("DIVERGED", flush=True); break
        if a.ckpt and a.upload_every and step % a.upload_every == 0 and step:
            from ibm.release import CheckpointName, sidecar, upload
            obj = ("av" if a.modality == "av" else f"nfh{a.horizon}")
            nm = CheckpointName(modality={"video": "v", "audio": "a", "av": "av"}[a.modality],
                                sites=a.sites, embed=a.embed, degree=a.k,
                                objective=obj, viability_weight=a.viability_weight,
                                step=step)
            torch.save({"model": model.state_dict(), "step": step, "name": str(nm)}, a.ckpt)
            meta = sidecar(nm, geometry="spherical shell, area-matched to the measured "
                                        "202,437 mm^2 white surface",
                           n_params=n_tot, n_assoc=n_assoc,
                           metrics=log["steps"][-1], config=vars(a))
            try:
                dest = upload(a.ckpt, meta)
                print(f"  uploaded {dest}", flush=True)
            except Exception as e:
                print(f"  upload failed: {e}", flush=True)

    import os
    if a.ckpt:
        os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "config": vars(a)}, a.ckpt)
        print(f"wrote {a.ckpt}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
