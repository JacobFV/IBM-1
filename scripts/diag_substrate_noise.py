"""DIAGNOSTIC, not a gate: does background noise alone carry the v2 substrate into an
itinerant regime at its declared priors, and at what noise level?

written AFTER G3 of `scripts/gate_substrate_v2.py` FAILED at the declared noise (0.04):
zero transitions in 60 s (docs/LOG.md 2026-09-18).  this does not re-score G3.  it maps
the G3 statistics over noise so that setting the noise can be done by a FIT against
resting-state data, under its own pre-registration, knowing whether the target regime is
reachable at all.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/diag_substrate_noise.py
"""
from __future__ import annotations

import json, math, os
from collections import Counter
import numpy as np
import torch

from ibm.substrate import Priors, build_sheet

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("sigmas", nargs="*", type=float)
_ap.add_argument("--topology", default="random", choices=("random", "tract"))
_ap.add_argument("--G_L", type=float, default=None)
_ap.add_argument("--G_F", type=float, default=None)
ARGS = _ap.parse_args()
SIGMAS = ARGS.sigmas or [0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
TAG = "_".join([ARGS.topology] + [f"{x:g}" for x in ARGS.sigmas]
               + ([f"GL{ARGS.G_L:g}"] if ARGS.G_L is not None else [])
               + ([f"GF{ARGS.G_F:g}"] if ARGS.G_F is not None else []))
T, DT = 30.0, 1e-3


@torch.no_grad()
def one(sigma, seed=0):
    pr = Priors(); pr.sigma = sigma
    if ARGS.G_L is not None: pr.G_L = ARGS.G_L
    if ARGS.G_F is not None: pr.G_F = ARGS.G_F
    f = build_sheet(1024, 32, seed=seed, priors=pr, learn_hetero=False,
                    long_topology=ARGS.topology)
    st = f.init_state(1)
    W = f.edge_weights()
    g = torch.Generator().manual_seed(seed + 6)
    rec = []
    for t in range(int(T / DT)):
        st = f.step(st, torch.zeros(1, f.n), DT, W=W, noise=torch.randn(1, f.n, generator=g))
        if t * DT >= 2.0 and t % 5 == 0:
            rec.append(st["E"][0].clone())
    E = torch.stack(rec).numpy()
    rid = f.region_id.numpy(); R = len(f.region_list)
    reg = np.stack([E[:, rid == r].mean(1) for r in range(R)], 1)
    code = reg > 0.5
    keys = [c.tobytes() for c in code]
    trans = [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
    dwell = np.diff([0] + trans + [len(keys)]) * 0.005
    visits = Counter(); prev = None
    for k in keys:
        if k != prev: visits[k] += 1
        prev = k
    # per-REGION dwell (each region's own up/down runs) and the mean pairwise
    # correlation of region up-states: the joint 68-bit code changes whenever ANY region
    # flips, so a short joint dwell with long per-region dwell means regions switch
    # independently -- a product of columns, not coordinated global states.
    reg_runs = []
    for r in range(R):
        c = code[:, r]
        ch = np.flatnonzero(c[1:] != c[:-1]) + 1
        runs = np.diff(np.r_[0, ch, len(c)]) * 0.005
        if len(ch):
            reg_runs.extend(runs[1:-1].tolist() if len(runs) > 2 else [])
    act = code[:, code.std(0) > 0].astype(float)
    if act.shape[1] >= 2:
        C = np.corrcoef(act.T); iu = np.triu_indices_from(C, 1)
        pair_corr = float(np.nanmean(C[iu]))
    else:
        pair_corr = float("nan")
    h = f.h.numpy()
    site_up = (E > 0.5).mean(0)
    bands = {f"{a}-{b}": float(site_up[(h >= a) & (h < b)].mean()) for a, b in [(0, .3), (.3, .6), (.6, .8), (.8, 1.01)]}
    return {"sigma": sigma, "topology": ARGS.topology, "G_L": pr.G_L, "G_F": pr.G_F,
            "transitions": len(trans), "median_dwell_s": float(np.median(dwell)),
            "distinct_macrostates": len(visits), "revisited_ge2": sum(v >= 2 for v in visits.values()),
            "mean_regions_up": float(code.sum(1).mean()), "mean_E": float(E.mean()),
            "mean_rate_hz": float(E.mean() * 100), "site_frac_time_up_by_h": bands,
            "region_dwell_median_s": float(np.median(reg_runs)) if reg_runs else None,
            "region_dwell_p90_s": float(np.percentile(reg_runs, 90)) if reg_runs else None,
            "regions_that_switch": int((code.std(0) > 0).sum()),
            "mean_pairwise_corr_region_up": pair_corr,
            "would_pass_G3_thresholds": bool(len(trans) >= 10 and np.median(dwell) > 0.05
                                              and sum(v >= 2 for v in visits.values()) >= 3)}


if __name__ == "__main__":
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    out = []
    for s in SIGMAS:
        r = one(s); out.append(r); print(json.dumps(r))
        with open(f"out/diag_substrate_noise_{TAG}.json", "w") as fh:
            json.dump({"diagnostic": True, "T_s": T, "rows": out}, fh, indent=2)
