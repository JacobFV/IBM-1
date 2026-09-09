"""how far does a perturbation travel across the cortical sheet, hop by hop?

the standing finding (docs/LOG.md, 2026-09-09) is that driving postcentral puts
0.03-0.07% of the signal into precentral, and that the fraction scales LINEARLY
with `w_assoc` rather than compounding.  linear scaling is the signature of a
SINGLE DIRECT HOP: whatever arrives came down one long-range edge, and the sheet
is not relaying at all.  that diagnosis is inferred, though, and the region
measurement cannot separate three very different causes:

  A. every hop attenuates enormously (an amplitude problem);
  B. hop 1 is healthy and hop 2 is dead (cancellation or saturation);
  C. transport is fine and the READOUT cannot see it.

this script settles it by measuring the thing directly.  it perturbs one site,
integrates the real `CorticalDynamics`, and reports the rate perturbation binned
by GRAPH HOP DISTANCE from the driven site, where the distance is a BFS on the
actual message-passing graph `dyn.idx`.

**direction matters and is easy to get backwards.**  `association` computes
`(r[:, idx] * w).sum(-1)`, so site i reads from the sites listed in `idx[i]`:
information flows j -> i exactly when j appears in row i.  the influence
frontier of a source S is therefore the set of ROWS CONTAINING an element of S,
which is not the same set as `idx[S]`.  the BFS below expands rows-containing,
and the severed-kernel gate proves it: with w = 0 the perturbation must be
exactly zero at every hop >= 1, and any leakage into hop 1 would mean the
labelling is wrong.

gates, per CLAUDE.md ("check a metric against a case whose answer you know"):

  amp = 0            every hop must read exactly 0.0
  severed kernel     hop 0 nonzero, hops >= 1 exactly 0.0
  hop 0              ratio exactly 1.0 by construction (printed, not trusted)
  linear theory      the measured hop-1 ratio is compared against the analytic
                     one-hop gain of the linearised dynamics; agreement means
                     the number is the dynamics and not a bug.

no training, no gradients, no baselines to beat -- this is a physical
measurement of an operator, and the quantity it reports is a ratio, so it needs
no external baseline beyond the two null cases above.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os

import torch

_SP = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(_SP)
_SP.loader.exec_module(P)


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

def hop_labels(idx: torch.Tensor, seeds: torch.Tensor, max_hop: int) -> torch.Tensor:
    """BFS hop distance FROM `seeds` along the direction information travels.

    returns (N,) int64, -1 for unreached.  `idx` is (N, k) and row i lists the
    sources site i reads from, so the successors of a set S are the rows that
    contain any member of S.
    """
    n = idx.shape[0]
    lab = torch.full((n,), -1, dtype=torch.long, device=idx.device)
    lab[seeds] = 0
    reached = torch.zeros(n, dtype=torch.bool, device=idx.device)
    reached[seeds] = True
    frontier = reached.clone()
    for h in range(1, max_hop + 1):
        # a row is in the next frontier if any of its sources is in the frontier
        nxt = frontier[idx].any(dim=1) & ~reached
        if not bool(nxt.any()):
            break
        lab[nxt] = h
        reached |= nxt
        frontier = nxt
    return lab


# ---------------------------------------------------------------------------
# the dynamics
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(dyn, drive, n_steps: int, h: float, w, hold: int | None = None):
    """integrate and return the final rate vector (B, N)."""
    b = drive.shape[0]
    s = dyn.init_state(b, drive.device)
    zero = torch.zeros_like(drive)
    for i in range(n_steps):
        s = dyn.step(s, drive if (hold is None or i < hold) else zero, h, w)
    return s[1]


@torch.no_grad()
def transfer_profile(dyn, seeds, lab, amp: float, n_steps: int, h: float,
                     w, tonic: float = 0.0, max_hop: int = 6):
    """rate perturbation binned by hop distance.

    the perturbation is a DIFFERENCE of two integrations of the same system --
    one with the seed drive, one without -- so anything the sheet does on its
    own (the resting point, the adaptation transient) cancels and what remains
    is the response to the perturbation alone.
    """
    n = dyn.n
    base = torch.full((1, n), float(tonic), device=dyn.pos.device)
    pert = base.clone()
    pert[0, seeds] += amp
    r0 = run(dyn, base, n_steps, h, w)
    r1 = run(dyn, pert, n_steps, h, w)
    d = (r1 - r0).abs()[0]
    out = {}
    for hh in range(0, max_hop + 1):
        m = lab == hh
        cnt = int(m.sum())
        if cnt == 0:
            continue
        out[hh] = {"n_sites": cnt, "mean_abs": float(d[m].mean()),
                   "max_abs": float(d[m].max())}
    m = lab < 0
    if bool(m.any()):
        out["unreached"] = {"n_sites": int(m.sum()),
                            "mean_abs": float(d[m].mean()),
                            "max_abs": float(d[m].max())}
    return out, d


# ---------------------------------------------------------------------------
# the linearisation, as an independent prediction of the same number
# ---------------------------------------------------------------------------

def linear_gain(dyn, v_op: float):
    """the one-hop gain of the linearised sheet at operating potential `v_op`.

    at steady state dv = 0 gives, ignoring adaptation and shunting,
        dv_i = 20 * w_assoc * (sum_j w_ij dr_j) / r_max
    and dr_i = (dr/dv)|v_op * dv_i, so one hop multiplies by
        g = (dr/dv) * 20 * w_assoc / r_max * w_ij .
    summing |w_ij| over the fan-in gives the row gain, whose maximum bounds the
    spectral radius of the linearised operator: below 1 the sheet is a strict
    attenuator and CANNOT relay, whatever the topology says.
    """
    s = torch.sigmoid(torch.tensor((v_op - dyn.v_half) / dyn.slope))
    drdv = float(dyn.r_max * s * (1 - s) / dyn.slope)
    w = dyn.edge_weights()
    c = drdv * 20.0 * float(dyn.w_assoc) / dyn.r_max
    row = (c * w.abs()).sum(-1)
    return {"drdv_hz_per_mv": drdv,
            "per_edge_gain_mean": float((c * w.abs()).mean()),
            "row_gain_mean": float(row.mean()),
            "row_gain_max": float(row.max()),
            "signed_row_gain_absmean": float((c * w).sum(-1).abs().mean())}


# ---------------------------------------------------------------------------

def build(ckpt_path, device, head="visual_eeg"):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = d["heads"][head] if "heads" in d else d["model"]
    # the GRAPH comes from the checkpoint, never from a fresh construction.
    # `CorticalDynamics.__init__` draws the long-range partners with an unseeded
    # `torch.randint`, so a rebuilt sheet has a DIFFERENT graph that happens to
    # have the right shape -- the exact failure mode CLAUDE.md records for the
    # THINGS-EEG2 image order.
    n, emb = sd["dyn.embed"].shape
    k = sd["dyn.idx"].shape[1]
    dyn = P.CorticalDynamics(n, emb, k, device).to(device)
    got = dyn.load_state_dict({kk[4:]: v.to(device) for kk, v in sd.items()
                               if kk.startswith("dyn.")}, strict=False)
    assert not got.missing_keys, got.missing_keys
    dyn.eval()
    return dyn, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/ibm1_curriculum16.pt")
    ap.add_argument("--head", default="visual_eeg")
    ap.add_argument("--seeds", type=int, default=8, help="driven sites, one run each")
    ap.add_argument("--amp", type=float, default=10.0, help="drive, mV")
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=8, help="the loops' own budget")
    ap.add_argument("--long-steps", type=int, default=400, help="to steady state")
    ap.add_argument("--tonic", type=float, default=0.0)
    ap.add_argument("--max-hop", type=int, default=6)
    ap.add_argument("--out", default="out/hop_transfer.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dyn, ck = build(a.ckpt, dev, a.head)
    h = a.dt / a.substeps
    print(f"{a.ckpt} head={a.head} step={ck.get('step')}  "
          f"n={dyn.n} k={dyn.k} n_far={dyn.n_far} device={dev}")
    print(f"w_assoc={float(dyn.w_assoc):.4f} w_ee={float(dyn.w_ee):.4f} "
          f"w_ei={float(dyn.w_ei):.4f} a_gain={float(dyn.a_gain):.4f}")
    print(f"integrator h={h*1e3:.2f} ms, tau_m={dyn.tau_m*1e3:.0f} ms; "
          f"short run {a.n_steps*a.substeps} steps = "
          f"{a.n_steps*a.substeps*h/dyn.tau_m:.2f} tau_m, "
          f"long run {a.long_steps} steps = {a.long_steps*h/dyn.tau_m:.1f} tau_m")

    w = dyn.edge_weights()
    print(f"\nedge weights: |w| mean {float(w.abs().mean()):.3e} "
          f"max {float(w.abs().max()):.3e}; geo max {float(dyn.geo.max()):.3e} "
          f"(= 1/k = {1.0/dyn.k:.3e})")

    # ---- resting point, needed for the linearisation -----------------------
    rest = run(dyn, torch.full((1, dyn.n), float(a.tonic), device=dev),
               a.long_steps, h, w)
    v_op = float(dyn.v_half + dyn.slope * math.log(
        max(float(rest.mean()) / dyn.r_max, 1e-12) /
        max(1 - float(rest.mean()) / dyn.r_max, 1e-12)))
    print(f"resting rate {float(rest.mean()):.3f} Hz  -> v_op {v_op:.2f} mV")
    lin = linear_gain(dyn, v_op)
    print("linearised one-hop gain at that operating point:")
    print(f"  dr/dv                     {lin['drdv_hz_per_mv']:.4f} Hz/mV")
    print(f"  per-edge gain (mean|w|)   {lin['per_edge_gain_mean']:.3e}")
    print(f"  row gain sum_j|w_ij|      mean {lin['row_gain_mean']:.4e} "
          f"max {lin['row_gain_max']:.4e}")
    print(f"  signed row gain |sum_j|   mean {lin['signed_row_gain_absmean']:.4e}")
    print("  (row gain is the spectral-radius scale of the linearised sheet;"
          " < 1 means strict attenuation)")

    torch.manual_seed(0)
    seeds_all = torch.randperm(dyn.n)[:a.seeds].to(dev)

    results = {"ckpt": a.ckpt, "head": a.head, "step": ck.get("step"),
               "n_sites": dyn.n, "k": dyn.k, "n_far": dyn.n_far,
               "w_assoc": float(dyn.w_assoc), "amp": a.amp, "tonic": a.tonic,
               "h_ms": h * 1e3, "linear": lin, "resting_hz": float(rest.mean()),
               "gates": {}, "runs": {}}

    # ---- GATE 1: zero perturbation must move nothing -----------------------
    prof, _ = transfer_profile(dyn, seeds_all[:1], hop_labels(dyn.idx, seeds_all[:1], 2),
                               0.0, a.long_steps, h, w, a.tonic, 2)
    z = max(v["max_abs"] for v in prof.values())
    print(f"\nGATE zero-amplitude: max |dr| anywhere = {z:.3e}  "
          f"{'PASS' if z == 0.0 else 'FAIL'}")
    results["gates"]["zero_amp_max_abs"] = z

    # ---- GATE 2: severed kernel must confine the perturbation to hop 0 -----
    lab1 = hop_labels(dyn.idx, seeds_all[:1], 3)
    prof_s, _ = transfer_profile(dyn, seeds_all[:1], lab1, a.amp, a.long_steps, h,
                                 torch.zeros_like(w), a.tonic, 3)
    off = max(v["max_abs"] for k_, v in prof_s.items() if k_ != 0)
    print(f"GATE severed kernel: hop0 |dr| = {prof_s[0]['mean_abs']:.4f} Hz, "
          f"max |dr| off-seed = {off:.3e}  "
          f"{'PASS' if off == 0.0 and prof_s[0]['mean_abs'] > 0 else 'FAIL'}")
    results["gates"]["severed_hop0"] = prof_s[0]["mean_abs"]
    results["gates"]["severed_offhop_max"] = off

    # ---- the measurement ---------------------------------------------------
    for tag, nst in (("short", a.n_steps * a.substeps), ("steady", a.long_steps)):
        acc = {}
        counts = {}
        for s in seeds_all:
            ss = s.reshape(1)
            lab = hop_labels(dyn.idx, ss, a.max_hop)
            prof, _ = transfer_profile(dyn, ss, lab, a.amp, nst, h, w,
                                       a.tonic, a.max_hop)
            for kk, v in prof.items():
                acc.setdefault(kk, []).append(v["mean_abs"])
                counts[kk] = v["n_sites"]
        print(f"\n=== {tag}: {nst} steps = {nst*h*1e3:.0f} ms "
              f"({nst*h/dyn.tau_m:.1f} tau_m), drive {a.amp} mV at 1 site, "
              f"mean over {a.seeds} seeds ===")
        print(f"{'hop':>9s} {'sites':>7s} {'mean|dr| Hz':>13s} "
              f"{'vs hop0':>10s} {'per hop':>10s}")
        prev = None
        rows = {}
        for kk in sorted([x for x in acc if x != "unreached"]) + \
                (["unreached"] if "unreached" in acc else []):
            m = sum(acc[kk]) / len(acc[kk])
            h0 = sum(acc[0]) / len(acc[0])
            ratio = m / h0 if h0 else float("nan")
            per = (m / prev) if (prev not in (None, 0.0)) else float("nan")
            print(f"{str(kk):>9s} {counts[kk]:7d} {m:13.6e} {ratio:10.3e} "
                  f"{per:10.3e}")
            rows[str(kk)] = {"n_sites": counts[kk], "mean_abs": m,
                             "vs_hop0": ratio, "per_hop": per}
            if kk != "unreached":
                prev = m
        results["runs"][tag] = rows

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
