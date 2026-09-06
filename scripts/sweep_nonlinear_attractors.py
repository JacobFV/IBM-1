"""does the nonlinear cortical loop have more than one invariant set anywhere?

DYNAMICS.md 6.2 is the gate on everything: a linear system has exactly one fixed
point, so it has no basins, no transitions, and nothing for a cognitive object to
be.  27 Form.RATE implementations are declared, and STATE.md 7c records that the
one nonlinear run attempted diverged -- but a direct integration of the closed E/I
loop at prior medians is BOUNDED and settles to a single ~5 Hz fixed point.  so
the question the gate actually asks is not "is it stable" but "is there anywhere
in prior-plausible parameter space where it holds more than one invariant set".

this sweeps the recurrent weights against adaptation and drive, integrates many
initial conditions per parameter set, and classifies the asymptotic structure.
it is the measurement that decides whether the substrate can carry an attractor
landscape at all.

the f's are `wilson_cowan_excitatory` and `shunting_inhibition_rate` from
ibm/processes/neural.py, transcribed to run batched on GPU.  `--shunt` selects
the declared form or the conductance-based one; they differ because the declared
tau_eff double-counts g_i (quadratic rather than linear), which matters only at
high inhibitory conductance.
"""
from __future__ import annotations

import argparse
import json

import torch

PRIOR = dict(tau_m=0.015, e_rest=-65.0, v_half=-55.0, slope=4.0, r_max=100.0,
             tau_a=0.5, drive=1.0, e_rev=-70.0, g_leak=1.0,
             tau_ampa=0.005, tau_gaba=0.008, tau_i=0.005,
             # fast-spiking interneurons: shorter membrane, lower half-activation,
             # steeper curve and a higher ceiling than pyramidal cells
             tau_m_i=0.008, v_half_i=-58.0, slope_i=3.0, r_max_i=250.0, w_ie=1.0)


def derivs(v, r_e, a, g_e, v_i, r_i, g_i, w_ee, w_ei, a_gain, drive, p, shunt):
    """two populations, each with its own membrane -- as local_excitation and
    local_inhibition declare them.  slaving r_i to the excitatory potential
    instead (an earlier version of this script) makes inhibition an instantaneous
    same-gain mirror of excitation, which erases bistability for any appreciable
    w_ei and is an artefact of the reduction, not a property of the model."""
    r_inf = p["r_max"] / (1.0 + torch.exp(-(v - p["v_half"]) / p["slope"]))
    dv = (-(v - p["e_rest"]) + drive * g_e - a) / p["tau_m"]
    gi = g_i.clamp_min(0.0)
    if shunt == "declared":
        tau_eff = (p["tau_m"] * p["g_leak"] / (p["g_leak"] + gi)).clamp_min(1e-6)
        dv = dv - (v - p["e_rev"]) * gi / tau_eff
    else:
        dv = dv - (v - p["e_rev"]) * gi / (p["tau_m"] * p["g_leak"])
    # inhibitory population: its own membrane, lower threshold, higher ceiling
    r_i_inf = p["r_max_i"] / (1.0 + torch.exp(-(v_i - p["v_half_i"]) / p["slope_i"]))
    dv_i = (-(v_i - p["e_rest"]) + p["w_ie"] * g_e) / p["tau_m_i"]
    return (dv,
            (r_inf - r_e) / 5e-3,
            (a_gain * r_inf - a) / p["tau_a"],
            (w_ee * r_e - g_e) / p["tau_ampa"],
            dv_i,
            (r_i_inf - r_i) / p["tau_i"],
            (w_ei * r_i - g_i) / p["tau_gaba"])


def integrate(w_ee, w_ei, a_gain, drive, *, shunt, n_ic, dt, t_total, t_record,
              device, p=PRIOR):
    """rk2 over a batch of (parameter set x initial condition) trajectories."""
    shape = w_ee.shape
    # initial conditions spread over the physiological range of v and g_e
    v0 = torch.linspace(-80.0, -40.0, n_ic, device=device)
    v = v0.view(*(1,) * len(shape), n_ic).expand(*shape, n_ic).clone()
    r_e = torch.zeros_like(v); a = torch.zeros_like(v)
    g_e = torch.zeros_like(v); g_i = torch.zeros_like(v)
    v_i = torch.full_like(v, p["e_rest"]); r_i = torch.zeros_like(v)
    W_ee, W_ei = w_ee.unsqueeze(-1), w_ei.unsqueeze(-1)
    A, D = a_gain.unsqueeze(-1), drive.unsqueeze(-1)

    n_steps = int(t_total / dt)
    n_rec = int(t_record / dt)
    v_min = torch.full_like(v, float("inf"))
    v_max = torch.full_like(v, float("-inf"))
    diverged = torch.zeros_like(v, dtype=torch.bool)

    for i in range(n_steps):
        s = (v, r_e, a, g_e, v_i, r_i, g_i)
        k1 = derivs(*s, W_ee, W_ei, A, D, p, shunt)
        mid = tuple(x + 0.5 * dt * k for x, k in zip(s, k1))
        k2 = derivs(*mid, W_ee, W_ei, A, D, p, shunt)
        v, r_e, a, g_e, v_i, r_i, g_i = tuple(x + dt * k for x, k in zip(s, k2))
        bad = ~torch.isfinite(v) | (v.abs() > 1e3)
        diverged |= bad
        v = torch.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        g_e = torch.nan_to_num(g_e).clamp(-1e3, 1e3)
        g_i = torch.nan_to_num(g_i).clamp(-1e3, 1e3)
        a = torch.nan_to_num(a).clamp(-1e3, 1e3)
        v_i = torch.nan_to_num(v_i).clamp(-1e3, 1e3)
        if i >= n_steps - n_rec:
            v_min = torch.minimum(v_min, v)
            v_max = torch.maximum(v_max, v)
    return v_min, v_max, diverged


def classify(v_min, v_max, diverged, *, osc_mv, sep_mv):
    """per parameter set: diverged / oscillatory / multistable / single."""
    any_div = diverged.any(dim=-1)
    swing = (v_max - v_min)                       # per-IC amplitude in the record window
    osc = (swing > osc_mv).any(dim=-1)
    centre = 0.5 * (v_max + v_min)                # per-IC asymptotic level
    spread = centre.max(dim=-1).values - centre.min(dim=-1).values
    multi = spread > sep_mv
    return any_div, osc, multi, spread, swing.max(dim=-1).values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shunt", choices=("declared", "conductance"), default="declared")
    ap.add_argument("--n", type=int, default=96, help="grid points per weight axis")
    ap.add_argument("--n-ic", type=int, default=24)
    ap.add_argument("--dt", type=float, default=5e-5)
    ap.add_argument("--t-total", type=float, default=4.0)
    ap.add_argument("--t-record", type=float, default=1.0)
    ap.add_argument("--osc-mv", type=float, default=0.5)
    ap.add_argument("--sep-mv", type=float, default=1.0)
    ap.add_argument("--out", default="out/nonlinear_phase.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} shunt={args.shunt} grid={args.n}^2 ic={args.n_ic} "
          f"dt={args.dt} T={args.t_total}s")

    w_ee_ax = torch.linspace(0.0, 0.60, args.n, device=device)
    w_ei_ax = torch.linspace(0.0, 0.60, args.n, device=device)
    # adaptation_gain prior is lognormal(0.05, 6.0) -- very wide; drive_gain weak(1.0, 5.0)
    slices = [(0.0, 1.0), (0.05, 1.0), (0.20, 1.0), (0.05, 3.0)]

    report = {"shunt": args.shunt, "grid": args.n, "n_ic": args.n_ic,
              "dt": args.dt, "t_total": args.t_total, "slices": []}
    for a_gain_v, drive_v in slices:
        W_ee, W_ei = torch.meshgrid(w_ee_ax, w_ei_ax, indexing="ij")
        A = torch.full_like(W_ee, a_gain_v)
        D = torch.full_like(W_ee, drive_v)
        v_min, v_max, div = integrate(W_ee, W_ei, A, D, shunt=args.shunt,
                                      n_ic=args.n_ic, dt=args.dt,
                                      t_total=args.t_total, t_record=args.t_record,
                                      device=device)
        any_div, osc, multi, spread, swing = classify(
            v_min, v_max, div, osc_mv=args.osc_mv, sep_mv=args.sep_mv)
        n = W_ee.numel()
        row = {
            "adaptation_gain": a_gain_v, "drive_gain": drive_v,
            "diverged_frac": float(any_div.float().mean()),
            "oscillatory_frac": float((osc & ~any_div).float().mean()),
            "multistable_frac": float((multi & ~any_div).float().mean()),
            "max_spread_mv": float(spread[~any_div].max()) if (~any_div).any() else None,
            "max_swing_mv": float(swing[~any_div].max()) if (~any_div).any() else None,
            "n_cells": int(n),
        }
        # where the interesting region sits
        if (multi & ~any_div).any():
            idx = (spread * (~any_div) * multi).argmax()
            row["multistable_at"] = {"w_ee": float(W_ee.flatten()[idx]),
                                     "w_ei": float(W_ei.flatten()[idx])}
        if (osc & ~any_div).any():
            idx = (swing * (~any_div) * osc).argmax()
            row["oscillatory_at"] = {"w_ee": float(W_ee.flatten()[idx]),
                                     "w_ei": float(W_ei.flatten()[idx])}
        report["slices"].append(row)
        print(f"  a_gain={a_gain_v:<5} drive={drive_v:<4} | "
              f"div={row['diverged_frac']:6.3f} osc={row['oscillatory_frac']:6.3f} "
              f"multi={row['multistable_frac']:6.3f} "
              f"max_spread={row['max_spread_mv']} max_swing={row['max_swing_mv']}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
