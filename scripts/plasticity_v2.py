"""does EXPERIENCE give the v2 substrate coordinated metastable states?

pre-registered in docs/LOG.md (2026-09-18, "PRE-REGISTRATION: experience-driven
plasticity").  one arm-seed per invocation:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/plasticity_v2.py --arm A --seed 0

arms, identical except where stated:
  A  plastic, real film + soundtrack
  B  plastic, the SAME input with every site's drive circularly shifted by its own
     random offset: each site keeps its marginal and its autocorrelation, and every
     cross-site co-occurrence is destroyed.  the covariance rule should find nothing
     structured to write.
  C  not plastic (eta = 0), real input

learning: 4 parallel streams (4 films), 300 s each, noise sigma 0.12.
test: 30 s spontaneous (no input), same noise, from rest.  the G3 statistics plus the
coordination measures from the diagnostics.

input ports (declared here, not fitted):
  visual   8x8 greyscale luminance of each 64x64 frame, z-scored per film, into the
           pericalcarine / cuneus / lingual / lateraloccipital sites through ONE fixed
           random projection (seeded, drawn once).  not retinotopy -- a later refinement.
  auditory the 64-band cochleagram, z-scored per film, one band per site into the
           transversetemporal / superiortemporal sites in TONOTOPIC order (sites sorted
           along the anterior-posterior axis).
  both scaled by DRIVE; 25 fps, each frame held for 40 steps of 1 ms.

cpu only, no_grad throughout, memory-guarded (CLAUDE.md, Jobs).
"""
from __future__ import annotations

import argparse, json, math, os, time
from collections import Counter
import numpy as np
import torch

from ibm.substrate import Priors, build_sheet

FILMS_DIR = "data/derived/pd-film"
VISUAL = ("pericalcarine", "cuneus", "lingual", "lateraloccipital")
AUDITORY = ("transversetemporal", "superiortemporal")
DT, FPS, SIGMA, DRIVE = 1e-3, 25, 0.12, 0.30
LEARN_S, TEST_S, STREAMS, START_S = 300.0, 30.0, 4, 600.0
ETA, LAM, APPLY_EVERY = 0.033, 0.5, 200


def rss_gb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return 0.0


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"<ndarray {o.shape}>"
    if torch.is_tensor(o):
        return o.tolist() if o.numel() <= 64 else f"<tensor {tuple(o.shape)}>"
    return str(o)


def load_streams(n_frames):
    """the first STREAMS films in sorted id order, START_S in, n_frames each."""
    man = json.load(open(os.path.join(FILMS_DIR, "manifest.json")))
    ids = sorted(f["id"] for f in man["films"])
    vis, aud, used = [], [], []
    for fid in ids:
        fr = np.load(os.path.join(FILMS_DIR, f"{fid}_frames.npy"), mmap_mode="r")
        co = np.load(os.path.join(FILMS_DIR, f"{fid}_coch.npy"), mmap_mode="r")
        a = int(START_S * FPS)
        if len(fr) < a + n_frames:
            continue
        f = np.asarray(fr[a:a + n_frames], dtype=np.float32).mean(-1)          # grey
        f = f.reshape(n_frames, 8, 8, 8, 8).mean((2, 4)).reshape(n_frames, 64)  # 8x8
        f = (f - f.mean(0)) / (f.std(0) + 1e-6)
        c = np.asarray(co[a:a + n_frames], dtype=np.float32)
        c = (c - c.mean(0)) / (c.std(0) + 1e-6)
        vis.append(f); aud.append(c); used.append(fid)
        if len(used) == STREAMS:
            break
    assert len(used) == STREAMS, f"only {len(used)} films long enough"
    return np.stack(vis), np.stack(aud), used                                   # (S, F, 64)


def ports(field, seed):
    names = [r.split(".", 1)[-1] for r in field.region_names]
    vis = np.flatnonzero(np.isin(names, VISUAL))
    aud = np.flatnonzero(np.isin(names, AUDITORY))
    g = np.random.default_rng(seed + 101)                     # drawn ONCE
    Pv = g.standard_normal((64, len(vis))).astype(np.float32) / 8.0
    # tonotopy: auditory sites in order along the anterior-posterior (RAS y) axis
    order = aud[np.argsort(field.pos[aud, 1].cpu().numpy())]
    band = np.floor(np.linspace(0, 64, len(order), endpoint=False)).astype(int)
    return vis, Pv, order, band


def drive_frames(field, vis_feat, aud_feat, vis, Pv, order, band):
    """(S, F, N) cortical drive for every frame."""
    S, F, _ = vis_feat.shape
    d = np.zeros((S, F, field.n), dtype=np.float32)
    d[:, :, vis] = vis_feat @ Pv
    d[:, :, order] = aud_feat[:, :, band]
    return d * DRIVE


def circ_shift_sites(d, seed):
    """control B: every (stream, site) drive series shifted by its own offset."""
    g = np.random.default_rng(seed + 202)
    S, F, N = d.shape
    out = np.empty_like(d)
    off = g.integers(0, F, size=(S, N))
    idx = (np.arange(F)[None, :, None] + off[:, None, :]) % F
    for s in range(S):
        out[s] = np.take_along_axis(d[s], idx[s], axis=0)
    return out


@torch.no_grad()
def spontaneous(field, seed):
    st = field.init_state(1)
    W = field.edge_weights()
    g = torch.Generator().manual_seed(seed + 303)
    rec = []
    zero = torch.zeros(1, field.n)
    for t in range(int(TEST_S / DT)):
        st = field.step(st, zero, DT, W=W, noise=torch.randn(1, field.n, generator=g))
        if t * DT >= 2.0 and t % 5 == 0:
            rec.append(st["E"][0].clone())
    E = torch.stack(rec).numpy()
    rid = field.region_id.numpy(); R = len(field.region_list)
    reg = np.stack([E[:, rid == r].mean(1) for r in range(R)], 1)
    code = reg > 0.5
    keys = [c.tobytes() for c in code]
    trans = [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
    dwell = np.diff([0] + trans + [len(keys)]) * 0.005
    visits = Counter(); prev = None
    for k in keys:
        if k != prev: visits[k] += 1
        prev = k
    runs = []
    for r in range(R):
        c = code[:, r]; ch = np.flatnonzero(c[1:] != c[:-1]) + 1
        rr = np.diff(np.r_[0, ch, len(c)]) * 0.005
        if len(rr) > 2: runs.extend(rr[1:-1].tolist())
    act = code[:, code.std(0) > 0].astype(float)
    pc = float(np.nanmean(np.corrcoef(act.T)[np.triu_indices(act.shape[1], 1)])) if act.shape[1] >= 2 else float("nan")
    revisited = sum(v >= 2 for v in visits.values())
    return {"transitions": len(trans), "median_joint_dwell_s": float(np.median(dwell)),
            "distinct": len(visits), "revisited_ge2": revisited,
            "region_dwell_median_s": float(np.median(runs)) if runs else None,
            "regions_that_switch": int((code.std(0) > 0).sum()),
            "mean_pairwise_corr": pc, "mean_rate_hz": float(E.mean() * 100),
            "passes_G3_thresholds": bool(len(trans) >= 10 and np.median(dwell) > 0.05 and revisited >= 3)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=("A", "B", "C"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learn-s", type=float, default=LEARN_S)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"out/plasticity_v2/{a.arm}_seed{a.seed}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    t0 = time.time()
    pr = Priors(); pr.sigma = SIGMA
    field = build_sheet(1024, 32, seed=a.seed, priors=pr, learn_hetero=False, long_topology="tract")
    n_frames = int(a.learn_s * FPS)
    vf, af, films = load_streams(n_frames)
    vis, Pv, order, band = ports(field, a.seed)
    d = drive_frames(field, vf, af, vis, Pv, order, band)
    if a.arm == "B":
        d = circ_shift_sites(d, a.seed)
    eta = 0.0 if a.arm == "C" else ETA
    res = {"arm": a.arm, "seed": a.seed, "films": films, "learn_s": a.learn_s, "eta": eta, "lam": LAM,
           "sigma": SIGMA, "drive": DRIVE, "n_visual_sites": int(len(vis)), "n_auditory_sites": int(len(order))}
    json.dump(res, open(out, "w"), indent=2, default=jdefault)

    st = field.init_state(STREAMS)
    pst = field.plasticity_init(STREAMS)
    W = field.edge_weights()
    g = torch.Generator().manual_seed(a.seed + 404)
    D = torch.from_numpy(d)
    steps_per_frame = int(round(1.0 / (FPS * DT)))
    t = 0
    for fi in range(n_frames):
        dr = D[:, fi]
        for _ in range(steps_per_frame):
            st = field.step(st, dr, DT, W=W, noise=torch.randn(STREAMS, field.n, generator=g))
            if eta:
                field.plasticity_accumulate(pst, st, DT)
                if (t + 1) % APPLY_EVERY == 0:
                    field.plasticity_apply(pst, DT, eta, LAM)
                    W = field.edge_weights()          # the kernel changed
            t += 1
        if fi % 1500 == 0:
            if rss_gb() > 12:
                raise MemoryError(f"RSS {rss_gb():.1f} GB")
            print(f"  frame {fi}/{n_frames}  t={t*DT:.0f}s  |P| mean {float(field.P.abs().mean()):.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    Pn = field.P[:, field.k - field.n_far:]
    res["P_long_mean_abs"] = float(Pn.abs().mean())
    res["P_long_frac_pos"] = float((Pn > 0).float().mean())
    res["P_long_p99_abs"] = float(Pn.abs().quantile(0.99))
    res["P_local_mean_abs"] = float(field.P[:, :field.k - field.n_far].abs().mean())
    json.dump(res, open(out, "w"), indent=2, default=jdefault)
    res["test"] = spontaneous(field, a.seed)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=2, default=jdefault)
    print(json.dumps(res["test"]), flush=True)


if __name__ == "__main__":
    main()
