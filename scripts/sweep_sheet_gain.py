"""how much can the cortical operator's gain be raised before it goes unstable,
and does anything useful arrive before it does?

`measure_hop_transfer.py` establishes the diagnosis: the sheet loses ~99% of a
coherent perturbation per hop, and the loss is a flat amplitude attenuation --
the linearised operator's row gain is 0.169, six times below the value at which
a hop would be lossless.

the obvious response is "raise the gain", and that has already been tried the
obvious way: `w_assoc` 0.5 -> 3.0 with a tonic drive reached a sever ratio of
1.254 and then collapsed at higher tonic.  this sweeps the question properly,
separating three knobs that the earlier attempt moved together:

  all       scale every edge weight (equivalent to replacing geo's 1/k)
  long      scale ONLY the n_far long-range edges, leaving the local sheet alone
  assoc     scale w_assoc, the knob that was already swept

and it reports, at each setting, both what arrives and whether the sheet is
still a sheet.  the second half matters: L1 row gain is NOT the stability
threshold for a signed kernel.  for a random signed matrix with `k` nonzeros per
row the spectral radius is about rms(w) * sqrt(k), not sum|w|, so a row gain of
1 on 48 signed edges sits at a radius near 0.14 and is nowhere near unstable.
that is the headroom this looks for.

the stability read-outs are measured, not assumed: the resting rate (which must
stay off both rails of the sigmoid), the largest |v| reached, and whether the
integration produced a non-finite state.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os

import torch

_SP = importlib.util.spec_from_file_location(
    "hoptx", os.path.join(os.path.dirname(__file__), "measure_hop_transfer.py"))
H = importlib.util.module_from_spec(_SP)
_SP.loader.exec_module(H)
P = H.P


@torch.no_grad()
def probe(dyn, w, drive_idx, read_idx, lab, amp, n_steps, h, tonic):
    """arrival at hop 1, at hop 2 and in the read region, plus stability."""
    n = dyn.n
    base = torch.full((1, n), float(tonic), device=dyn.pos.device)
    pert = base.clone()
    pert[0, drive_idx] += amp

    def go(d):
        s = dyn.init_state(1, d.device)
        vmax = 0.0
        for _ in range(n_steps):
            s = dyn.step(s, d, h, w)
            vmax = max(vmax, float(s[0].abs().max()))
        return s, vmax

    s0, v0 = go(base)
    s1, v1 = go(pert)
    finite = bool(torch.isfinite(s1[1]).all() and torch.isfinite(s0[1]).all())
    if not finite:
        return {"finite": False}
    d = (s1[1] - s0[1]).abs()[0]
    h0 = float(d[lab == 0].mean())
    out = {"finite": True,
           "rest_hz": float(s0[1].mean()),
           "rest_hz_max": float(s0[1].max()),
           "v_abs_max": max(v0, v1),
           "hop0": h0,
           "hop1": float(d[lab == 1].mean()) / max(h0, 1e-30),
           "hop2": float(d[lab == 2].mean()) / max(h0, 1e-30) if bool((lab == 2).any()) else float("nan"),
           "read": float(d[read_idx].mean()) / max(h0, 1e-30)}
    return out


def row_stats(w, dyn, rest_hz):
    """L1 row gain and the circular-law radius estimate, at the resting point."""
    s = rest_hz / dyn.r_max
    drdv = float(dyn.r_max * s * (1 - s) / dyn.slope)
    c = drdv * 20.0 * float(dyn.w_assoc) / dyn.r_max
    g = c * w
    l1 = float(g.abs().sum(-1).mean())
    rms = float((g ** 2).mean().sqrt())
    return {"drdv": drdv, "l1_row_gain": l1,
            "radius_est": rms * math.sqrt(w.shape[1] * w.shape[0] /
                                          w.shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/ibm1_curriculum16.pt")
    ap.add_argument("--head", default="visual_eeg")
    ap.add_argument("--knob", default="all", choices=("all", "long", "assoc"))
    ap.add_argument("--scales", default="1,2,4,8,16,32,48,64,96,128")
    ap.add_argument("--drive-region", default="postcentral")
    ap.add_argument("--read-region", default="precentral")
    ap.add_argument("--amp", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--tonic", type=float, default=0.0)
    ap.add_argument("--out", default="out/sheet_gain_sweep.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dyn, ck = H.build(a.ckpt, dev, a.head)
    h = a.dt / a.substeps
    drive_idx = P.region_index(dyn.pos, a.drive_region).to(dev)
    read_idx = P.region_index(dyn.pos, a.read_region).to(dev)
    lab = H.hop_labels(dyn.idx, drive_idx, 3)
    w0 = dyn.edge_weights().detach()
    assoc0 = float(dyn.w_assoc)
    n_loc = dyn.k - dyn.n_far

    print(f"{a.ckpt} n={dyn.n} k={dyn.k} n_far={dyn.n_far}  knob={a.knob}")
    print(f"drive {a.drive_region} ({len(drive_idx)} sites) -> "
          f"read {a.read_region} ({len(read_idx)} sites), {a.steps} steps "
          f"= {a.steps*h/dyn.tau_m:.0f} tau_m")
    print(f"hop sizes: 0={int((lab==0).sum())} 1={int((lab==1).sum())} "
          f"2={int((lab==2).sum())} unreached={int((lab<0).sum())}")
    print(f"\n{'scale':>7s} {'rest Hz':>8s} {'restmax':>8s} {'|v|max':>9s} "
          f"{'L1 row':>8s} {'hop1':>10s} {'hop2':>10s} {'read':>10s}")

    rows = {}
    for sc in [float(x) for x in a.scales.split(",")]:
        w = w0.clone()
        if a.knob == "all":
            w = w * sc
        elif a.knob == "long":
            w[:, n_loc:] = w[:, n_loc:] * sc
        else:
            dyn.w_assoc.data.fill_(assoc0 * sc)
        r = probe(dyn, w, drive_idx, read_idx, lab, a.amp, a.steps, h, a.tonic)
        if not r["finite"]:
            print(f"{sc:7.1f}   DIVERGED (non-finite state)")
            rows[str(sc)] = {"finite": False}
            dyn.w_assoc.data.fill_(assoc0)
            continue
        st = row_stats(w, dyn, r["rest_hz"])
        print(f"{sc:7.1f} {r['rest_hz']:8.2f} {r['rest_hz_max']:8.2f} "
              f"{r['v_abs_max']:9.1f} {st['l1_row_gain']:8.3f} "
              f"{r['hop1']:10.3e} {r['hop2']:10.3e} {r['read']:10.3e}")
        rows[str(sc)] = {**r, **st}
        dyn.w_assoc.data.fill_(assoc0)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"ckpt": a.ckpt, "knob": a.knob, "drive": a.drive_region,
               "read": a.read_region, "steps": a.steps, "rows": rows},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
