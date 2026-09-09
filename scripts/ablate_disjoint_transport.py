"""does anything the motor strip can USE arrive there, on a real task?

`measure_hop_transfer.py` and `measure_multimodal_convergence.py` measure
amplitude.  amplitude is not skill, and CLAUDE.md's whole ledger of withdrawn
claims is quantities computed correctly and compared against the wrong thing.
so this is the task version of the same question, on real data, with the
ablation that has to gate it.

**the task.**  THINGS-EEG2 image retrieval.  a real image drives the OCCIPITAL
region; the sheet runs; the state is read from the PRECENTRAL region ONLY; a
head is trained to align that read with the measured EEG of the same image; and
the score is held-out top-1 retrieval against chance.  every previous
contrastive head in this repo reads `linspace(0, n-1)`, which samples the driven
region directly -- measured, those samples carry 3,134x the variance of the rest
and a permuted kernel gives an identical answer.  reading precentral only means
the score is transport or it is nothing.

**why the encoder is frozen.**  the drive comes from the checkpoint's own
trained image encoder and `to_cortex`, run under no_grad, and only a small head
on the precentral rates is fitted.  that is not a shortcut, it is the control:
every arm sees the SAME stimulus-specific drive pattern, so the arms differ only
in what the sheet does with it.  training the encoder per-arm would let a strong
encoder compensate for a weak sheet and the comparison would stop being about
the sheet.

**the arms.**  each kernel configuration is scored with the kernel intact, with
it SEVERED (w = 0), and with its site rows PERMUTED.  severed is the ablation
the fix has to survive: with no kernel nothing crosses, the precentral read is
identical for every image, and retrieval must fall to chance.  permuted is
IHM-1's control and this project's recurring one -- a permuted kernel has the
same weight statistics and the same graph, so anything it recovers was never
about what the kernel learned.

**the noise arm, and why it is not optional.**  in a noiseless float32
simulation a linear head can amplify a perturbation of 1e-5 Hz and retrieve
perfectly, which would make the ablation report a large sever ratio for a sheet
that transports nothing usable.  a cortex is not noiseless.  so the score is
reported as a function of additive membrane noise, and the honest summary of a
transport improvement is how far up that curve it moves the arm -- an amplitude
gain of X buys X times the tolerable noise, and nothing else.

single-pool retrieval has sd ~ 2.8% in this repo's experience, so every score
here is the mean over `--pools` disjoint held-out pools, never a single one.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SP = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(_SP)
_SP.loader.exec_module(P)

D = "data/derived/things-paired"
ONSET, KEEP, DEC = 20, 50, 2


def load_head(ckpt, device, head="visual_eeg"):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = d["heads"][head]
    n, emb = sd["dyn.embed"].shape
    k = sd["dyn.idx"].shape[1]
    dyn = P.CorticalDynamics(n, emb, k, device).to(device)
    miss = dyn.load_state_dict({kk[4:]: v.to(device) for kk, v in sd.items()
                                if kk.startswith("dyn.")}, strict=False)
    assert not miss.missing_keys, miss.missing_keys
    model = P.VisualContrastiveLoop(dyn, n_sensors=64,
                                    n_times=KEEP // DEC).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, dyn, d.get("step", -1)


@torch.no_grad()
def cortical_features(model, dyn, imgs, drive_idx, read_idx, *, mode, batch,
                      n_steps, substeps, dt, device, gen):
    """precentral rates for every image, under one kernel treatment.

    `mode` is intact | severed | permuted.  permuted shuffles the ROWS of the
    edge-weight matrix, which keeps the graph, the fan-in and the weight
    statistics and destroys only the correspondence between a site and the
    weights it learned.
    """
    w = dyn.edge_weights().detach()
    if mode == "severed":
        w = torch.zeros_like(w)
    elif mode == "permuted":
        w = w[torch.randperm(w.shape[0], generator=gen).to(w.device)]
    elif mode != "intact":
        raise ValueError(mode)
    h = dt / substeps
    out = []
    for i in range(0, len(imgs), batch):
        x = imgs[i:i + batch].to(device)
        b = x.shape[0]
        drive = torch.zeros(b, dyn.n, device=device)
        # the occipital REGION, not `[:n//8]`.  the trained port is an arbitrary
        # eighth of a random point cloud and it intersects precentral, which
        # would let the readout see the drive directly -- the exact contamination
        # this script exists to avoid.
        u = model.to_cortex(model.enc(x))
        # the trained port has `dyn.n // 8` channels and the occipital region has
        # its own count; take the overlap rather than assuming they match, and
        # note in the JSON how many channels were actually used.
        m = min(u.shape[1], len(drive_idx))
        drive[:, drive_idx[:m]] = u[:, :m]
        s = dyn.init_state(b, device)
        for _ in range(n_steps * substeps):
            s = dyn.step(s, drive, h, w)
        out.append(s[1][:, read_idx].cpu())
    return torch.cat(out)


class Retrieval(nn.Module):
    """two small heads aligned by InfoNCE.  fitted per arm on cached features."""

    def __init__(self, n_read, n_sensors, n_times, dim=128, hidden=512):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(n_read, hidden), nn.GELU(),
                               nn.Linear(hidden, dim))
        self.b = nn.Sequential(nn.Flatten(), nn.Linear(n_sensors * n_times, hidden),
                               nn.GELU(), nn.Linear(hidden, dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, f, e):
        return (F.normalize(self.a(f), dim=-1), F.normalize(self.b(e), dim=-1))


def fit_and_score(feat_tr, eeg_tr, feat_te, eeg_te, *, pools, pool, steps, lr,
                  device, seed, noise_sd):
    """fit the head on train, report mean top-1 over disjoint held-out pools."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    # NOISE IS ADDED IN THE UNITS OF THE RATE, before any standardisation, so
    # that a sheet delivering a smaller perturbation is genuinely harder to read.
    # standardising first and then adding noise would rescale the signal back up
    # and hide exactly the quantity under test.
    if noise_sd > 0:
        feat_tr = feat_tr + noise_sd * torch.randn(feat_tr.shape, generator=g)
        feat_te = feat_te + noise_sd * torch.randn(feat_te.shape, generator=g)
    mu, sd = feat_tr.mean(0, keepdim=True), feat_tr.std(0, keepdim=True)
    sd = sd.clamp_min(1e-12)
    ftr = ((feat_tr - mu) / sd).to(device)
    fte = ((feat_te - mu) / sd).to(device)
    etr, ete = eeg_tr.to(device), eeg_te.to(device)
    m = Retrieval(ftr.shape[1], etr.shape[1], etr.shape[2]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = len(ftr)
    for _ in range(steps):
        idx = torch.randint(0, n, (256,), device=device)
        za, zb = m(ftr[idx], etr[idx])
        logit = m.logit_scale.exp().clamp(max=100) * za @ zb.T
        lbl = torch.arange(len(idx), device=device)
        loss = 0.5 * (F.cross_entropy(logit, lbl) + F.cross_entropy(logit.T, lbl))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    accs = []
    with torch.no_grad():
        for p in range(pools):
            sl = slice(p * pool, (p + 1) * pool)
            if sl.stop > len(fte):
                break
            za, zb = m(fte[sl], ete[sl])
            sim = za @ zb.T
            lbl = torch.arange(sim.shape[0], device=device)
            accs.append(float((sim.argmax(1) == lbl).float().mean()))
    return float(np.mean(accs)), float(np.std(accs)), len(accs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/ibm1_curriculum16.pt")
    ap.add_argument("--head", default="visual_eeg")
    ap.add_argument("--drive-region", default="occipital")
    ap.add_argument("--read-region", default="precentral")
    ap.add_argument("--configs", default="base:2.0,1.0,0;aniso:2.0,4.0,1",
                    help="name:tanh_slope,long_gain,long_topm; ';'-separated")
    ap.add_argument("--modes", default="intact,severed,permuted")
    ap.add_argument("--noise", default="0,1e-3,1e-2,3e-2,1e-1,3e-1,1,3")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--pools", type=int, default=8)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=2e-2)
    ap.add_argument("--fit-steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/disjoint_transport.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, dyn, step = load_head(a.ckpt, dev, a.head)
    drive_idx = P.region_index(dyn.pos, a.drive_region).to(dev)
    if a.read_region == "all":
        # the CONTROL readout, not a transport measurement: `linspace(0, n-1)`
        # is what every working head in this repo reads, and it samples the
        # driven region directly.  it is here so the cost of the kernel change
        # can be measured on the task the sheet already does, with the head
        # refitted per arm.  the overlap check is skipped on purpose and the
        # number must NOT be read as transport.
        read_idx = torch.linspace(0, dyn.n - 1, 2048).long().to(dev)
        print("  WHOLE-SHEET readout: this arm samples the driven region and "
              "is a regression check, not a transport measurement")
    else:
        read_idx = P.region_index(dyn.pos, a.read_region).to(dev)
        ov = len(np.intersect1d(drive_idx.cpu().numpy(), read_idx.cpu().numpy()))
        if ov:
            raise SystemExit(f"drive and read regions overlap in {ov} sites -- a "
                             f"readout that can see its own drive measures nothing")
    print(f"{a.ckpt} step={step}: drive {a.drive_region} ({len(drive_idx)}) -> "
          f"read {a.read_region} ({len(read_idx)}), disjoint")

    # ---- data ------------------------------------------------------------
    imgs = np.load(f"{D}/images_training.npy", mmap_mode="r")
    ev = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    n_all = min(len(imgs), len(ev))
    n_tr = min(a.n_train, n_all - a.pools * a.pool)
    n_te = a.pools * a.pool
    sl_tr, sl_te = slice(0, n_tr), slice(n_all - n_te, n_all)

    def prep_img(sl):
        x = torch.from_numpy(np.ascontiguousarray(imgs[sl]))
        return (x.permute(0, 3, 1, 2).float() / 127.5) - 1

    def prep_eeg(sl, med, iqr):
        y = np.asarray(ev[sl]).astype(np.float32)
        y = np.clip((y - med) / iqr, -6, 6)[..., ONSET:ONSET + KEEP:DEC]
        return torch.from_numpy(y)

    s_ = np.asarray(ev[0:n_tr:7]).astype(np.float32)
    med = np.median(s_, 0)
    iqr = ((np.percentile(s_, 75, 0) - np.percentile(s_, 25, 0)) / 1.349).clip(1e-9)
    x_tr, x_te = prep_img(sl_tr), prep_img(sl_te)
    e_tr, e_te = prep_eeg(sl_tr, med, iqr), prep_eeg(sl_te, med, iqr)
    print(f"train {n_tr}, held-out {n_te} = {a.pools} pools of {a.pool} "
          f"(chance {100.0/a.pool:.2f}%)")

    gen = torch.Generator().manual_seed(a.seed)
    noises = [float(x) for x in a.noise.split(",")]
    res = {"ckpt": a.ckpt, "step": step, "drive": a.drive_region,
           "read": a.read_region, "n_train": n_tr, "pools": a.pools,
           "pool": a.pool, "chance": 1.0 / a.pool, "arms": {}}

    for spec in a.configs.split(";"):
        name, vals = spec.split(":")
        slope, gain, topm = vals.split(",")
        dyn.tanh_slope, dyn.long_gain, dyn.long_topm = \
            float(slope), float(gain), int(topm)
        dyn.local_gain = float(vals.split(",")[3]) if len(vals.split(",")) > 3 else 1.0
        w = dyn.edge_weights().detach()
        l1 = float(w.abs().sum(-1).mean())
        print(f"\n### {name}: tanh_slope={slope} long_gain={gain} long_topm={topm}"
              f"   row L1 gain {l1:.4f}  |w| mean {float(w.abs().mean()):.3e}")
        for mode in a.modes.split(","):
            kw = dict(mode=mode, batch=a.batch, n_steps=a.n_steps,
                      substeps=a.substeps, dt=a.dt, device=dev, gen=gen)
            f_tr = cortical_features(model, dyn, x_tr, drive_idx, read_idx, **kw)
            f_te = cortical_features(model, dyn, x_te, drive_idx, read_idx, **kw)
            # the raw scale of the arriving perturbation, so the retrieval
            # numbers below can be read against what is physically there.
            spread = float(f_tr.std(0).mean())
            print(f"  {mode:9s}  across-image sd of the precentral rate "
                  f"{spread:.4e} Hz")
            for nz in noises:
                acc, sd, npool = fit_and_score(
                    f_tr, e_tr, f_te, e_te, pools=a.pools, pool=a.pool,
                    steps=a.fit_steps, lr=a.lr, device=dev, seed=a.seed,
                    noise_sd=nz)
                x_chance = acc * a.pool
                print(f"    noise {nz:8.1e} Hz -> top-1 {100*acc:6.2f}% "
                      f"+/- {100*sd:4.2f}  ({x_chance:6.2f}x chance, "
                      f"{npool} pools)")
                res["arms"].setdefault(name, {}).setdefault(mode, {})[str(nz)] = {
                    "top1": acc, "sd": sd, "x_chance": x_chance,
                    "n_pools": npool, "feature_sd_hz": spread, "row_l1": l1}

    # ---- the sever ratio, which is the number that decides this ------------
    print(f"\n{'config':10s} {'noise':>9s} {'intact':>9s} {'severed':>9s} "
          f"{'permuted':>9s} {'sever ratio':>12s}")
    for name, arms in res["arms"].items():
        for nz in [str(x) for x in noises]:
            g_ = arms.get("intact", {}).get(nz)
            s_ = arms.get("severed", {}).get(nz)
            p_ = arms.get("permuted", {}).get(nz)
            if not (g_ and s_):
                continue
            # the ratio of skill-above-chance multiples.  the severed arm sits
            # at exactly 1.00x whenever nothing crosses, so a
            # (x-1)/(x-1) form divides by zero and prints 1e10; this is the
            # form the rest of the repo uses for a sever ratio.
            ratio = g_["x_chance"] / max(s_["x_chance"], 1e-9)
            print(f"{name:10s} {float(nz):9.1e} {g_['x_chance']:8.2f}x "
                  f"{s_['x_chance']:8.2f}x "
                  f"{(p_['x_chance'] if p_ else float('nan')):8.2f}x "
                  f"{ratio:12.2f}")
            res.setdefault("sever_ratio", {}).setdefault(name, {})[nz] = ratio

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
