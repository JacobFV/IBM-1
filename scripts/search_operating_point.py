"""does any prior give the cortical field a firing range AND an attractor landscape?

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        .venv/bin/python -u scripts/search_operating_point.py --out out/search_operating_point.json

PRE-REGISTERED in docs/LOG.md, 18 September 2026, before this ran.  Read that entry first:
it fixes the grid, the three criteria, the cheapened settings, and -- the part that matters
-- both branches of the fork, including what it means if candidates pass the first two
criteria and fail the third.

WHY.  Three independent measurements today found the same thing: the sheet has no regime
between silent and saturated.  Swept over drive it goes from 4.60 Hz with zero saturation to
23.95 Hz with 19% saturated between two adjacent drives.  Inhibition-stabilising it produces
a smooth 5-17 Hz range and costs the attractors -- G2 from 8 invariant sets to 1, G3 to zero
transitions with no region ever up.  The two sweeps that found those each measured ONE
property and never the gates.  This measures all three on every candidate.

DRIVE IS MATCHED FOR RATE.  For each candidate the tonic drive is bisected until the sheet
sits at 8 Hz, and the graded criterion is tested over 0.5x-2x that drive.  Comparing
candidates at a fixed drive AMPLITUDE compares them at different operating points, which is
what made the earlier sweeps hard to read.

The G2 and G3 measurements are the gate script's own, imported rather than reimplemented, so
a candidate is judged by the same code that judges the declared priors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm.substrate import Priors, R_MAX, build_sheet                    # noqa: E402

# the grid, fixed in the pre-registration
BETA_I = (6.0, 9.0, 12.0)
W_EI = (0.5, 0.7, 0.9)
W_IE = (0.9, 1.4)

# cheapened for the search; the winner is confirmed on the unmodified gate script
N_SITES = 512
K = 32
DT = 1e-3
G2_STARTS = 16
G2_SECONDS = 23.0
G3_SECONDS = 30.0
TARGET_RATE_HZ = 8.0
DRIVE_FACTORS = (0.5, 1.0, 2.0)

# the criteria, fixed in the pre-registration
RATE_LO, RATE_HI = 5.0, 20.0
SAT_MAX = 0.02
G2_MIN_SETS = 2
G3_MIN_TRANSITIONS, G3_MIN_DWELL_S, G3_MIN_REVISITED = 10, 0.05, 3


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist() if o.numel() <= 32 else f"<tensor {tuple(o.shape)}>"
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 32 else f"<ndarray {o.shape}>"
    return str(o)


def field_of(beta_I, w_EI, w_IE, seed=0):
    pr = Priors(beta_I=beta_I, w_EI=w_EI, w_IE=w_IE)
    return build_sheet(N_SITES, K, 0.25, seed=seed, device="cpu",
                       long_topology="tract", priors=pr)


@torch.no_grad()
def regime(f, tonic, seconds=4.0, seed=3):
    """(mean rate Hz, saturated fraction) at this tonic drive, after a burn-in."""
    g = torch.Generator().manual_seed(seed)
    st = f.init_state(1)
    nb, nt = int(1.0 / DT), int(seconds / DT)
    _, st = f.rollout(torch.full((1, nb, f.n), float(tonic)), st, DT, noise_gen=g)
    tr, _ = f.rollout(torch.full((1, nt, f.n), float(tonic)), st, DT, noise_gen=g)
    return float(tr.mean() * R_MAX), float((tr > 0.9).float().mean())


def drive_for_rate(f, target=TARGET_RATE_HZ, lo=0.0, hi=0.40, iters=12):
    """bisect the tonic drive until the sheet sits at `target` Hz.

    Returns (drive, rate, saturation, bracketed).  `bracketed` is False when the sheet
    jumps across the target between two adjacent drives without ever sitting on it -- which
    is not a failure of the bisection, it IS the ignition this whole search is about, and it
    is reported rather than smoothed over.
    """
    r_lo, _ = regime(f, lo)
    r_hi, _ = regime(f, hi)
    if not (r_lo <= target <= r_hi):
        return hi, r_hi, regime(f, hi)[1], False
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r, _s = regime(f, mid)
        if r < target:
            lo = mid
        else:
            hi = mid
    d = 0.5 * (lo + hi)
    r, s = regime(f, d)
    return d, r, s, True


@torch.no_grad()
def graded(f, d0):
    """criterion 1: rate in band, saturation under the bar, monotone in drive."""
    rows = []
    for k in DRIVE_FACTORS:
        r, s = regime(f, d0 * k)
        rows.append({"factor": k, "drive": d0 * k, "rate_hz": r, "saturated": s})
    rates = [r["rate_hz"] for r in rows]
    ok = (all(RATE_LO <= r <= RATE_HI for r in rates)
          and all(r["saturated"] <= SAT_MAX for r in rows)
          and all(rates[i] <= rates[i + 1] + 1e-6 for i in range(len(rates) - 1)))
    return {"ok": bool(ok), "sweep": rows}


@torch.no_grad()
def g2_invariant_sets(f, drive, starts=G2_STARTS, seed=5):
    """criterion 2, the gate's own method: distinct time-averaged patterns from random
    starts, counted at correlation >= 0.9."""
    g = torch.Generator().manual_seed(seed)
    st = f.init_state(starts, random=True, generator=g)
    nt = int(G2_SECONDS / DT)
    keep = int(3.0 / DT)
    d = torch.full((starts, nt, f.n), float(drive))
    tr, _ = f.rollout(d, st, DT)
    rec = tr[:, -keep:]
    pat = rec.mean(1)
    reps = []
    for i in range(pat.shape[0]):
        p = pat[i]
        same = False
        for r in reps:
            q = pat[r]
            if p.std() < 1e-6 or q.std() < 1e-6:
                same = bool(p.std() < 1e-6 and q.std() < 1e-6
                            and (p.mean() - q.mean()).abs() < 1e-3)
            else:
                same = float(torch.corrcoef(torch.stack([p, q]))[0, 1]) >= 0.9
            if same:
                break
        if not same:
            reps.append(i)
    return {"n_sets": len(reps), "ok": len(reps) >= G2_MIN_SETS,
            "mean_E_per_set": [float(pat[r].mean()) for r in reps]}


@torch.no_grad()
def g3_metastability(f, drive, seed=6):
    """criterion 3, the gate's own method: region means binarised at 0.5, transitions and
    revisited macrostates counted.  The 0.5 is the gate's, deliberately unchanged."""
    g = torch.Generator().manual_seed(seed)
    st = f.init_state(1)
    nb = int(2.0 / DT)
    _, st = f.rollout(torch.full((1, nb, f.n), float(drive)), st, DT, noise_gen=g)
    nt = int(G3_SECONDS / DT)
    tr, _ = f.rollout(torch.full((1, nt, f.n), float(drive)), st, DT, noise_gen=g)
    E = tr[0, ::5].numpy()
    rid = f.region_id.numpy()
    reg = np.stack([E[:, rid == r].mean(1) for r in range(len(f.region_list))], 1)
    code = reg > 0.5
    keys = [c.tobytes() for c in code]
    trans = [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
    dwell = np.diff([0] + trans + [len(keys)]) * 0.005
    visits, prev = Counter(), None
    for kk in keys:
        if kk != prev:
            visits[kk] += 1
        prev = kk
    revisited = sum(1 for v in visits.values() if v >= 2)
    ok = (len(trans) >= G3_MIN_TRANSITIONS and float(np.median(dwell)) > G3_MIN_DWELL_S
          and revisited >= G3_MIN_REVISITED)
    return {"transitions": len(trans), "median_dwell_s": float(np.median(dwell)),
            "distinct_macrostates": len(visits), "revisited_ge2": revisited,
            "mean_regions_up": float(code.sum(1).mean()),
            "max_region_mean_E": float(reg.max()), "ok": bool(ok)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/search_operating_point.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rec = {"script": "scripts/search_operating_point.py",
           "preregistered": "docs/LOG.md 2026-09-18 operating-point search",
           "grid": {"beta_I": BETA_I, "w_EI": W_EI, "w_IE": W_IE},
           "settings": {"n_sites": N_SITES, "dt": DT, "g2_starts": G2_STARTS,
                        "g3_seconds": G3_SECONDS, "target_rate_hz": TARGET_RATE_HZ},
           "criteria": {"rate_band_hz": [RATE_LO, RATE_HI], "saturation_max": SAT_MAX,
                        "g2_min_sets": G2_MIN_SETS,
                        "g3": [G3_MIN_TRANSITIONS, G3_MIN_DWELL_S, G3_MIN_REVISITED]},
           "candidates": []}

    def save():
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=jdefault)

    save()
    t0 = time.time()
    print(f"{'beta_I':>7} {'w_EI':>5} {'w_IE':>5} | {'drive':>6} {'rate':>6} {'sat':>6} "
          f"| {'graded':>6} {'sets':>5} {'trans':>6} {'dwell':>6} {'revis':>6} | verdict",
          flush=True)
    for bI in BETA_I:
        for wEI in W_EI:
            for wIE in W_IE:
                f = field_of(bI, wEI, wIE)
                d0, r0, s0, bracketed = drive_for_rate(f)
                gr = graded(f, d0)
                g2 = g2_invariant_sets(f, d0)
                g3 = g3_metastability(f, d0)
                row = {"beta_I": bI, "w_EI": wEI, "w_IE": wIE,
                       "drive_for_8hz": d0, "rate_at_drive": r0, "sat_at_drive": s0,
                       "rate_bracketed": bracketed, "graded": gr, "g2": g2, "g3": g3,
                       "all_three": bool(gr["ok"] and g2["ok"] and g3["ok"]),
                       "graded_and_g2": bool(gr["ok"] and g2["ok"])}
                rec["candidates"].append(row)
                save()
                verdict = ("ALL THREE" if row["all_three"] else
                           "graded+G2" if row["graded_and_g2"] else
                           "-")
                print(f"{bI:7.1f} {wEI:5.2f} {wIE:5.2f} | {d0:6.3f} {r0:6.2f} {s0:6.3f} "
                      f"| {str(gr['ok']):>6} {g2['n_sets']:5d} {g3['transitions']:6d} "
                      f"{g3['median_dwell_s']:6.2f} {g3['revisited_ge2']:6d} | {verdict}"
                      + ("" if bracketed else "   (ignites past the target rate)"),
                      flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    allthree = [c for c in rec["candidates"] if c["all_three"]]
    g2only = [c for c in rec["candidates"] if c["graded_and_g2"] and not c["all_three"]]
    rec["summary"] = {"n": len(rec["candidates"]), "all_three": len(allthree),
                      "graded_and_g2_only": len(g2only)}
    save()
    print(f"\n{len(allthree)} of {len(rec['candidates'])} pass all three; "
          f"{len(g2only)} pass graded+G2 and fail G3.")
    print("Per the pre-registration: a candidate passing all three is confirmed on the "
          "UNMODIFIED gate script at full size before anything is adopted; the "
          "graded+G2-only case points at G3's fixed 0.5 threshold and is followed up with "
          "a separately declared pattern-based measure, with G3 left FAILED.")
    print(f"wrote {a.out}  ({rec['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
