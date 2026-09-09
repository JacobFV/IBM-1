#!/usr/bin/env python
"""does the thalamo-cortico-thalamic loop actually run, and is it a loop?

`docs/DYNAMICS.md` §2 makes the TCT loop the central claim of the architecture:
thalamus does not add a state variable, it changes the effective cortical
operator, and the round-trip delay plus the reticular nucleus's inhibition
*predict* alpha and spindle frequencies rather than fitting them.  §6 then
records that `thalamocortical_coupling` is declared and that the running cortical
model -- `scripts/pretrain_video_loop.py`'s `CorticalDynamics` -- has no thalamus
in it at all.  `ibm/processes/tct.py` closes that gap.  this script asks whether
closing it did anything.

the question is not "does the mean cortical rate have a spectral peak".  every
finite trace has an argmax, and `docs/LOG.md` has ten entries whose shared shape
is a quantity computed correctly and compared against the wrong thing.  so the
measurement is built as a set of severances of one running model:

  cortex_only      the sheet exactly as it runs today, zero drive.  this is the
                   "before" and it is the number every other row is against.
  full             cortex + thalamus + TRN, closed, zero external input.
  sever_tc         thalamo-cortical projection cut.  the loop's ascending limb.
  sever_ct         cortico-thalamic projection cut.  the DESCENDING limb -- the
                   one that decides whether cortex is inside the loop or merely
                   downstream of a thalamic oscillator.
  sever_trn        reticular inhibition cut.  the negative feedback.
  sever_t_current  the T-type calcium conductance cut.  the burst mechanism.

every row is the SAME integration of the SAME equations with one gain set to
zero, run from the same initial condition, for the same duration.  nothing is
re-fitted between rows.

three checks the numbers have to survive before any of them is read:

1. the estimator is run on cases whose answer is known -- a 12 Hz sine (must
   return 12 Hz with a large peak ratio), white noise (must return a small peak
   ratio; the value it returns is what calibrates "large"), and a constant (must
   return zero amplitude).  a failure here aborts before any model is run.
2. `sever_tc` must reproduce `cortex_only` to floating-point.  if cutting the
   ascending limb does not leave the cortex in exactly the state it would have
   been in with no thalamus, the ablation is leaking somewhere and no ablation
   delta below means anything.  this is asserted, not hoped for.
3. a peak is reported as a peak only if BOTH the peak-to-median power ratio
   clears a threshold read off the white-noise control AND the amplitude of the
   oscillation is a physiologically visible number rather than 1e-9 Hz.  an
   argmax on a flat spectrum is not an oscillation.

everything is reported across `--seeds` independent initial conditions and
quoted as mean +/- sd, because `CLAUDE.md` records that single-draw bests in this
project select for lucky draws.

run:
    .venv/bin/python scripts/measure_tct.py
    .venv/bin/python scripts/measure_tct.py --sweep      # the declared prediction
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibm.processes.tct import ThalamoCortical
from pretrain_video_loop import CorticalDynamics

#: the bands the claim is stated in.  spindle overlaps alpha on purpose -- that
#: is how the literature defines them, and pretending otherwise by moving an
#: edge would decide a band assignment by construction.
BANDS = {
    "slow": (0.3, 1.5),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "spindle": (10.0, 16.0),
    "beta": (16.0, 30.0),
}

#: the analysis band.  the low edge is above the DC/drift the mean removal
#: leaves and below the slow oscillation; the high edge is well under the
#: 1/(2*dt) nyquist of every dt this script will accept.
ANALYSIS_BAND = (0.3, 60.0)


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------


def spectrum(x: np.ndarray, dt: float):
    """one-sided power spectrum of a 1-D trace, mean removed, Hann windowed.

    Hann rather than rectangular: a self-sustained nonlinear oscillation is not
    an integer number of cycles in an arbitrary window, and rectangular leakage
    from a large low-frequency component can put a spurious ridge across the
    whole band.  the window costs a factor ~1.5 in resolution and is corrected
    for in power by its own mean square, so peak ratios are comparable across
    conditions.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    w = np.hanning(len(x))
    xw = x * w
    f = np.fft.rfftfreq(len(x), dt)
    P = np.abs(np.fft.rfft(xw)) ** 2 / (len(x) * np.mean(w ** 2))
    return f, P


def peak_stats(x: np.ndarray, dt: float, band=ANALYSIS_BAND) -> dict:
    """peak frequency, peak prominence, amplitude and band powers.

    `peak_ratio` is peak power over the MEDIAN power in the band.  the median
    and not the mean, because the mean of a spectrum containing a large peak is
    dominated by that peak, so a peak-over-mean ratio compresses toward 1 exactly
    when there is something to see.  the median is an estimate of the aperiodic
    floor the peak is standing on.
    """
    f, P = spectrum(x, dt)
    m = (f >= band[0]) & (f <= band[1])
    fb, Pb = f[m], P[m]
    k = int(np.argmax(Pb))
    med = float(np.median(Pb))
    total = float(np.trapezoid(Pb, fb)) if hasattr(np, "trapezoid") else float(
        np.trapz(Pb, fb))
    out = {
        "peak_hz": float(fb[k]),
        "peak_power": float(Pb[k]),
        "peak_ratio": float(Pb[k] / med) if med > 0 else float("inf"),
        "amplitude_ptp": float(np.ptp(x)),
        "amplitude_sd": float(np.std(x)),
        "mean": float(np.mean(x)),
        "band_power_total": total,
        "resolution_hz": float(fb[1] - fb[0]) if len(fb) > 1 else float("nan"),
    }
    for name, (lo, hi) in BANDS.items():
        sel = (fb >= lo) & (fb <= hi)
        if sel.sum() > 1:
            bp = (float(np.trapezoid(Pb[sel], fb[sel])) if hasattr(np, "trapezoid")
                  else float(np.trapz(Pb[sel], fb[sel])))
        else:
            bp = 0.0
        out[f"bandpower_{name}"] = bp
        out[f"bandfrac_{name}"] = bp / total if total > 0 else 0.0
    return out


def self_test(dt: float, n: int, rng_seed: int = 0) -> dict:
    """run the estimator on three signals whose answers are known.

    the white-noise row is not decoration.  it is where the threshold for
    "there is a peak" comes from: whatever peak ratio a trace of this length
    with no oscillation in it produces is the number a real peak has to beat.
    """
    t = np.arange(n) * dt
    rng = np.random.default_rng(rng_seed)
    cases = {
        "sine_12hz": np.sin(2 * math.pi * 12.0 * t),
        "sine_12hz_plus_noise": np.sin(2 * math.pi * 12.0 * t) + 3.0 * rng.standard_normal(n),
        "white_noise": rng.standard_normal(n),
        "constant": np.full(n, 7.5),
    }
    res = {k: peak_stats(v, dt) for k, v in cases.items()}
    res["_pass_sine"] = abs(res["sine_12hz"]["peak_hz"] - 12.0) <= 2 * res[
        "sine_12hz"]["resolution_hz"]
    res["_pass_sine_in_noise"] = abs(
        res["sine_12hz_plus_noise"]["peak_hz"] - 12.0) <= 2 * res[
            "sine_12hz_plus_noise"]["resolution_hz"]
    res["_pass_constant"] = res["constant"]["amplitude_ptp"] == 0.0
    res["_noise_peak_ratio"] = res["white_noise"]["peak_ratio"]
    return res


# ---------------------------------------------------------------------------
# the model runs
# ---------------------------------------------------------------------------


def build(args, device):
    torch.manual_seed(0)
    return CorticalDynamics(args.sites, args.embed, args.k, device,
                            long_range=args.long_range).to(device)


def run_conditions(dyn, args, conditions: dict, device) -> dict:
    """integrate every condition x seed in ONE batched pass.

    the batch axis carries (condition, seed).  running them together is not only
    faster -- it guarantees the cortical sheet, its edge weights and the
    integration schedule are bit-identical across conditions, so a difference
    between two rows cannot be a difference between two models.
    """
    names = list(conditions)
    S = args.seeds
    B = len(names) * S
    n = int(round(args.seconds / args.dt))
    burn = int(round(args.burn / args.dt))

    tct = ThalamoCortical(
        n_sites=dyn.n, n_thal=args.thal, device=device,
        g_t=args.g_t, i_bg_mv=args.i_bg,
        t_deinactivation_s=args.tau_h, w_trn_gaba_a=args.w_gaba_a,
        tc_delay_s=args.tc_delay, ct_delay_s=args.ct_delay,
    ).to(device)
    tct.reset(B, args.dt, device=device, jitter=args.jitter)

    # per-batch-element gains: the ablation is a gain vector, so every condition
    # runs inside the same tensor op and cannot diverge by code path.
    def col(fn):
        return torch.tensor([[fn(conditions[nm])] for nm in names
                             for _ in range(S)], device=device).float()

    tct.w_tc = col(lambda c: c.get("w_tc", 1.0)) * args.w_tc
    tct.w_ct_relay = col(lambda c: c.get("w_ct", 1.0)) * args.w_ct_relay
    tct.w_ct_trn = col(lambda c: c.get("w_ct", 1.0)) * args.w_ct_trn
    tct.w_trn_gaba_a = col(lambda c: c.get("w_trn", 1.0)) * args.w_gaba_a
    tct.w_trn_gaba_b = col(lambda c: c.get("w_trn", 1.0)) * args.w_gaba_b
    tct.g_t = col(lambda c: c.get("g_t", 1.0)) * args.g_t

    # independent initial conditions per seed, identical ACROSS conditions: the
    # s-th seed of every row starts from the same relay potentials, so a
    # difference between rows is the severance and not the draw.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    v0 = (-70.0 + args.jitter * torch.randn(S, args.thal, generator=g)).to(device)
    tct.v_t = v0.repeat(len(names), 1)
    tct.h = tct.h_inf(tct.v_t).clone()
    tct.r_t = tct.rate(tct.v_t).clone()
    tct.buf_tc = tct.r_t[None].repeat(tct.buf_tc.shape[0], 1, 1)

    with torch.no_grad():
        s = dyn.init_state(B, device)
        w = dyn.edge_weights()
        keep = n - burn
        # recorded ON DEVICE and transferred once.  a `.cpu()` inside the loop is
        # a synchronization point per step, which on a gpu costs more than the
        # integration does -- it made a 15,000-step run take minutes.
        rc = torch.empty(keep, B, device=device)
        rt = torch.empty(keep, B, device=device)
        vc = torch.empty(keep, B, device=device)
        vt = torch.empty(keep, B, device=device)
        for i in range(n):
            drive = tct.step(s[1], args.dt)
            s = dyn.step(s, drive, args.dt, w)
            if i >= burn:
                j = i - burn
                rc[j] = s[1].mean(1)
                vc[j] = s[0].mean(1)
                rt[j] = tct.r_t.mean(1)
                vt[j] = tct.v_t.mean(1)
    return {"names": names, "seeds": S,
            "cortex_rate": rc.double().cpu().numpy(),
            "cortex_v": vc.double().cpu().numpy(),
            "thal_rate": rt.double().cpu().numpy(),
            "thal_v": vt.double().cpu().numpy()}


def summarize(rec, dt, signal="cortex_rate") -> dict:
    """per-condition mean +/- sd over seeds of every peak statistic."""
    names, S = rec["names"], rec["seeds"]
    X = rec[signal]
    out = {}
    for ci, nm in enumerate(names):
        per = [peak_stats(X[:, ci * S + s], dt) for s in range(S)]
        agg = {}
        for k in per[0]:
            vals = np.array([p[k] for p in per], dtype=float)
            agg[k] = float(vals.mean())
            agg[k + "_sd"] = float(vals.std())
        agg["per_seed_peak_hz"] = [p["peak_hz"] for p in per]
        out[nm] = agg
    return out


# ---------------------------------------------------------------------------
# the declared prediction
# ---------------------------------------------------------------------------


def sweep(dyn, args, device) -> list:
    """`burst_relay_rate` predicts what moves the frequency; check that it does.

    the declaration says `t_deinactivation_s` is the T current's recovery time
    and that `trn_gain` "sets whether the oscillator runs at spindle or at delta
    frequency".  those are two falsifiable monotone predictions and this is the
    grid that tests them.  it is a PREDICTION CHECK and not a fit: nothing here
    is selected against a target frequency.
    """
    taus = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
    gains = [0.5, 1.1, 2.0, 3.0]
    combos = [(a, b) for a in taus for b in gains]
    B = len(combos)
    n = int(round(args.sweep_seconds / args.dt))
    burn = int(round(args.burn / args.dt))
    tct = ThalamoCortical(n_sites=dyn.n, n_thal=args.thal, device=device,
                          g_t=args.g_t, i_bg_mv=args.i_bg,
                          tc_delay_s=args.tc_delay, ct_delay_s=args.ct_delay).to(device)
    tct.reset(B, args.dt, device=device, jitter=args.jitter)
    tct.tau_h = torch.tensor([[a] for a, _ in combos], device=device).float()
    tct.w_trn_gaba_a = torch.tensor([[b] for _, b in combos], device=device).float()
    with torch.no_grad():
        s = dyn.init_state(B, device)
        w = dyn.edge_weights()
        rc_t = torch.empty(n - burn, B, device=device)
        for i in range(n):
            d = tct.step(s[1], args.dt)
            s = dyn.step(s, d, args.dt, w)
            if i >= burn:
                rc_t[i - burn] = s[1].mean(1)
        rc = rc_t.double().cpu().numpy()
    rows = []
    for j, (a, b) in enumerate(combos):
        st = peak_stats(rc[:, j], args.dt)
        rows.append({"t_deinactivation_s": a, "trn_gaba_a_gain": b,
                     "peak_hz": st["peak_hz"], "peak_ratio": st["peak_ratio"],
                     "amplitude_ptp": st["amplitude_ptp"]})
    return rows


def dt_convergence(dyn, args, device) -> dict:
    """the same closed loop at dt and dt/2.  the frequency must not move much.

    everything here is integrated with forward euler, which is what
    `CorticalDynamics.step` already uses, and a forward-euler frequency can be a
    function of the timestep rather than of the mechanism.  a reported "7 Hz
    rhythm" that becomes 5 Hz at half the step is a property of the integrator.
    this halves dt, keeps the total simulated time fixed, and reports the shift.
    """
    out = {}
    import copy
    for lab, dtv in (("dt", args.dt), ("dt_half", args.dt / 2.0)):
        b = copy.copy(args)
        b.dt = dtv
        b.seconds = args.sweep_seconds
        b.seeds = 1
        rec = run_conditions(dyn, b, {"full": {}}, device)
        out[lab] = {"dt": dtv,
                    **peak_stats(rec["cortex_rate"][:, 0], dtv)}
    out["peak_shift_hz"] = out["dt_half"]["peak_hz"] - out["dt"]["peak_hz"]
    out["amplitude_ratio"] = (out["dt_half"]["amplitude_ptp"]
                              / out["dt"]["amplitude_ptp"]
                              if out["dt"]["amplitude_ptp"] > 0 else float("inf"))
    return out


def bifurcation(dyn, args, device) -> list:
    """where does the oscillation switch on, with and without the descending limb?

    this is the measurement that separates "cortex is inside the loop" from
    "cortex is downstream of a thalamic oscillator", and it is stronger than the
    single-point ablation because it does not depend on having picked a lucky
    operating point.  the T-current conductance is swept through the intrathalamic
    oscillator's own bifurcation twice: once with the cortico-thalamic projection
    intact and once with it cut.  everything else -- sheet, weights, seeds,
    schedule -- is identical, and the two arms are columns of the same batch.

    if the two arms switch on at the same g_t, cortical feedback contributes
    nothing to whether the thing oscillates, and calling it a loop would be
    decoration on a relay.  if the intact arm switches on EARLIER, the descending
    limb is supplying part of the gain that sustains the rhythm, and the gap
    between the two thresholds is the size of that contribution in the units the
    mechanism is written in.
    """
    gs = [round(x, 3) for x in np.arange(args.bif_lo, args.bif_hi + 1e-9,
                                         args.bif_step)]
    combos = [(g, 1.0) for g in gs] + [(g, 0.0) for g in gs]
    B = len(combos)
    n = int(round(args.sweep_seconds / args.dt))
    burn = int(round(args.burn / args.dt))
    tct = ThalamoCortical(n_sites=dyn.n, n_thal=args.thal, device=device,
                          i_bg_mv=args.i_bg, t_deinactivation_s=args.tau_h,
                          w_trn_gaba_a=args.w_gaba_a, w_trn_gaba_b=args.w_gaba_b,
                          tc_delay_s=args.tc_delay, ct_delay_s=args.ct_delay).to(device)
    tct.reset(B, args.dt, device=device, jitter=args.jitter)
    tct.g_t = torch.tensor([[g] for g, _ in combos], device=device).float()
    tct.w_ct_relay = torch.tensor([[args.w_ct_relay * c] for _, c in combos],
                                  device=device).float()
    tct.w_ct_trn = torch.tensor([[args.w_ct_trn * c] for _, c in combos],
                                device=device).float()
    with torch.no_grad():
        s = dyn.init_state(B, device)
        w = dyn.edge_weights()
        rc = torch.empty(n - burn, B, device=device)
        rt = torch.empty(n - burn, B, device=device)
        for i in range(n):
            d = tct.step(s[1], args.dt)
            s = dyn.step(s, d, args.dt, w)
            if i >= burn:
                rc[i - burn] = s[1].mean(1)
                rt[i - burn] = tct.r_t.mean(1)
    rc = rc.double().cpu().numpy()
    rt = rt.double().cpu().numpy()
    rows = []
    for j, (g, c) in enumerate(combos):
        a = peak_stats(rc[:, j], args.dt)
        b = peak_stats(rt[:, j], args.dt)
        rows.append({"g_t": g, "ct_intact": bool(c),
                     "cortex_peak_hz": a["peak_hz"], "cortex_peak_ratio": a["peak_ratio"],
                     "cortex_amplitude_ptp": a["amplitude_ptp"],
                     "thal_peak_hz": b["peak_hz"], "thal_peak_ratio": b["peak_ratio"],
                     "thal_amplitude_ptp": b["amplitude_ptp"]})
    return rows


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", type=int, default=2048)
    ap.add_argument("--embed", type=int, default=32)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--thal", type=int, default=16)
    ap.add_argument("--dt", type=float, default=1e-4,
                    help="must resolve the 5/8 ms loop delays; tct.reset asserts it")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--burn", type=float, default=2.0,
                    help="discarded transient; the spectrum is of what is left")
    ap.add_argument("--sweep-seconds", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=2.0)
    # --- thalamic operating point -------------------------------------------
    ap.add_argument("--g-t", type=float, default=3.25,
                    help="T-current conductance relative to leak.  the DEFAULT IS "
                         "BELOW the isolated thalamus's own bifurcation, so an "
                         "oscillation here requires the cortical limb")
    ap.add_argument("--i-bg", type=float, default=0.0)
    ap.add_argument("--tau-h", type=float, default=0.05)
    ap.add_argument("--w-gaba-a", type=float, default=2.0)
    ap.add_argument("--w-gaba-b", type=float, default=0.15)
    ap.add_argument("--w-tc", type=float, default=0.60)
    ap.add_argument("--w-ct-relay", type=float, default=0.35)
    ap.add_argument("--w-ct-trn", type=float, default=0.50)
    ap.add_argument("--tc-delay", type=float, default=5e-3)
    ap.add_argument("--ct-delay", type=float, default=8e-3)
    ap.add_argument("--device", default="",
                    help="cpu or cuda; default picks cuda when present")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--dt-check", action="store_true",
                    help="rerun the closed loop at dt/2 and report the frequency shift")
    ap.add_argument("--bifurcation", action="store_true",
                    help="sweep the T-current conductance with the descending "
                         "limb intact and cut; the gap between the two onsets is "
                         "the cortical contribution to sustaining the rhythm")
    ap.add_argument("--bif-lo", type=float, default=2.0)
    ap.add_argument("--bif-hi", type=float, default=4.5)
    ap.add_argument("--bif-step", type=float, default=0.25)
    ap.add_argument("--out", default="out/tct_report.json")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = int(round(a.seconds / a.dt)) - int(round(a.burn / a.dt))

    # ---- 1. the estimator, on cases whose answers are known ---------------
    st = self_test(a.dt, n)
    print("estimator self-test")
    for k in ("sine_12hz", "sine_12hz_plus_noise", "white_noise", "constant"):
        s_ = st[k]
        print(f"  {k:<22} peak {s_['peak_hz']:8.3f} Hz   ratio {s_['peak_ratio']:10.1f}"
              f"   ptp {s_['amplitude_ptp']:.4g}")
    print(f"  resolution {st['white_noise']['resolution_hz']:.4f} Hz;  "
          f"white-noise peak ratio {st['_noise_peak_ratio']:.1f} "
          f"-> a peak must beat this")
    if not (st["_pass_sine"] and st["_pass_sine_in_noise"] and st["_pass_constant"]):
        print("ESTIMATOR SELF-TEST FAILED -- aborting before running the model")
        sys.exit(1)
    thresh = 10.0 * st["_noise_peak_ratio"]
    print(f"  PASS.  peak criterion: ratio > {thresh:.1f} (10x the noise control) "
          f"AND amplitude > 0.01 Hz\n", flush=True)

    dyn = build(a, dev)
    print(f"cortex: {a.sites} sites, k={a.k}; thalamus: {a.thal} units; "
          f"dt={a.dt:g}s, {a.seconds:g}s ({a.burn:g}s discarded), "
          f"{a.seeds} seeds, device={dev}", flush=True)
    print(f"loop delays: tc {a.tc_delay*1e3:.0f} ms + ct {a.ct_delay*1e3:.0f} ms "
          f"= {(a.tc_delay+a.ct_delay)*1e3:.0f} ms round trip "
          f"({int(round(a.tc_delay/a.dt))}/{int(round(a.ct_delay/a.dt))} steps)\n",
          flush=True)

    conditions = {
        "cortex_only": {"w_tc": 0.0, "w_ct": 0.0, "w_trn": 0.0, "g_t": 0.0},
        "full": {},
        "sever_tc": {"w_tc": 0.0},
        "sever_ct": {"w_ct": 0.0},
        "sever_trn": {"w_trn": 0.0},
        "sever_t_current": {"g_t": 0.0},
    }
    rec = run_conditions(dyn, a, conditions, dev)
    cortex = summarize(rec, a.dt, "cortex_rate")
    thal = summarize(rec, a.dt, "thal_rate")

    # ---- 2. the ablation machinery has to be airtight ---------------------
    # severing the ascending limb must leave the cortex in exactly the state a
    # cortex with no thalamus at all would be in.  bitwise, not approximately.
    S = a.seeds
    i_only = rec["names"].index("cortex_only") * S
    i_tc = rec["names"].index("sever_tc") * S
    d = np.abs(rec["cortex_rate"][:, i_only:i_only + S]
               - rec["cortex_rate"][:, i_tc:i_tc + S]).max()
    print(f"ablation integrity: max |cortex_only - sever_tc| = {d:.3e}  "
          f"({'PASS' if d == 0.0 else 'FAIL'})")
    if d != 0.0:
        print("  the severed condition does not reproduce the no-thalamus cortex; "
              "every delta below would be uninterpretable.  aborting.")
        sys.exit(1)
    print()

    base = cortex["cortex_only"]
    full = cortex["full"]

    def oscillates(s_):
        return s_["peak_ratio"] > thresh and s_["amplitude_ptp"] > 0.01

    print(f"{'condition':<17} {'peak Hz':>16} {'ratio':>9} {'ptp Hz':>10} "
          f"{'sd Hz':>8}  {'osc':>4}  {'vs cortex_only'}")
    rows = {}
    for nm in conditions:
        c = cortex[nm]
        dpow = (c["band_power_total"] / base["band_power_total"]
                if base["band_power_total"] > 0 else float("inf"))
        rows[nm] = {
            "cortex": c, "thalamus": thal[nm],
            "oscillates": bool(oscillates(c)),
            "amplitude_ratio_vs_cortex_only": (c["amplitude_ptp"] / base["amplitude_ptp"]
                                               if base["amplitude_ptp"] > 0 else float("inf")),
            "power_ratio_vs_cortex_only": dpow,
            "amplitude_ratio_vs_full": (c["amplitude_ptp"] / full["amplitude_ptp"]
                                        if full["amplitude_ptp"] > 0 else float("inf")),
            "power_ratio_vs_full": (c["band_power_total"] / full["band_power_total"]
                                    if full["band_power_total"] > 0 else float("inf")),
        }
        print(f"{nm:<17} {c['peak_hz']:8.2f}+/-{c['peak_hz_sd']:<5.2f} "
              f"{c['peak_ratio']:9.1f} {c['amplitude_ptp']:10.4f} "
              f"{c['amplitude_sd']:8.4f}  {'YES' if oscillates(c) else ' no':>4}  "
              f"x{rows[nm]['amplitude_ratio_vs_cortex_only']:.3g} amp, "
              f"x{dpow:.3g} power")

    print(f"\n{'condition':<17} {'THALAMIC peak Hz':>18} {'ratio':>9} {'ptp Hz':>10}  osc")
    for nm in conditions:
        t_ = thal[nm]
        print(f"{nm:<17} {t_['peak_hz']:10.2f}+/-{t_['peak_hz_sd']:<5.2f} "
              f"{t_['peak_ratio']:9.1f} {t_['amplitude_ptp']:10.4f}  "
              f"{'YES' if oscillates(t_) else ' no'}")

    # ---- 3. the verdict, stated as the conjunction it is ------------------
    v = {
        "cortex_oscillates_with_loop": bool(oscillates(full)),
        "cortex_oscillates_without_loop": bool(oscillates(base)),
        "killed_by_severing_tc": bool(oscillates(full) and not oscillates(cortex["sever_tc"])),
        "killed_by_severing_ct": bool(oscillates(full) and not oscillates(cortex["sever_ct"])),
        "killed_by_severing_trn": bool(oscillates(full) and not oscillates(cortex["sever_trn"])),
        "killed_by_severing_t_current": bool(
            oscillates(full) and not oscillates(cortex["sever_t_current"])),
        "thalamus_oscillates_without_cortex": bool(oscillates(thal["sever_ct"])),
    }
    v["is_a_closed_loop"] = bool(
        v["cortex_oscillates_with_loop"] and not v["cortex_oscillates_without_loop"]
        and v["killed_by_severing_tc"] and v["killed_by_severing_ct"])
    v["is_a_driven_relay"] = bool(
        v["cortex_oscillates_with_loop"] and v["killed_by_severing_tc"]
        and not v["killed_by_severing_ct"])
    print("\nverdict")
    for k, val in v.items():
        print(f"  {k:<34} {val}")
    if v["is_a_closed_loop"]:
        print("\n  CLOSED LOOP: the rhythm needs both limbs.  cutting the descending\n"
              "  limb stops it, so cortex is inside the oscillator and not downstream\n"
              "  of one.")
    elif v["is_a_driven_relay"]:
        print("\n  NOT A CLOSED LOOP at this operating point: the oscillator is\n"
              "  intrathalamic and cortex is driven by it.  cutting the ascending limb\n"
              "  removes the cortical rhythm, but cutting the descending limb does not,\n"
              "  so cortex is not part of what generates the rhythm.")
    elif not v["cortex_oscillates_with_loop"]:
        print("\n  NEGATIVE RESULT: the loop does not produce a cortical oscillation at\n"
              "  this operating point.")

    report = {
        "config": vars(a),
        "device": dev,
        "self_test": {k: val for k, val in st.items()},
        "peak_criterion": {"ratio_threshold": thresh, "amplitude_threshold_hz": 0.01,
                           "basis": "10x the peak ratio of a white-noise trace of the "
                                    "same length"},
        "ablation_integrity_max_abs_diff": float(d),
        "conditions": rows,
        "verdict": v,
    }

    if a.sweep:
        print("\ndeclared prediction: shorter t_deinactivation_s and stronger TRN\n"
              "inhibition both raise the frequency (burst_relay_rate docstring).")
        rows_s = sweep(dyn, a, dev)
        print(f"  {'tau_h':>7} {'trn_gain':>9} {'peak Hz':>9} {'ratio':>9} {'ptp':>9}")
        for r in rows_s:
            print(f"  {r['t_deinactivation_s']:7.3f} {r['trn_gaba_a_gain']:9.2f} "
                  f"{r['peak_hz']:9.2f} {r['peak_ratio']:9.1f} {r['amplitude_ptp']:9.3f}")
        osc = [r for r in rows_s if r["peak_ratio"] > thresh and r["amplitude_ptp"] > 0.01]
        mono = {}
        if osc:
            for gsel in sorted({r["trn_gaba_a_gain"] for r in osc}):
                seq = sorted([r for r in osc if r["trn_gaba_a_gain"] == gsel],
                             key=lambda r: r["t_deinactivation_s"])
                if len(seq) > 2:
                    fs = [r["peak_hz"] for r in seq]
                    mono[f"trn_gain_{gsel}"] = {
                        "tau_h": [r["t_deinactivation_s"] for r in seq],
                        "peak_hz": fs,
                        "monotone_decreasing_in_tau_h": bool(
                            all(fs[i] >= fs[i + 1] - 1e-9 for i in range(len(fs) - 1))),
                        "spearman": float(np.corrcoef(
                            np.argsort(np.argsort([r["t_deinactivation_s"] for r in seq])),
                            np.argsort(np.argsort(fs)))[0, 1]),
                    }
            print("\n  monotonicity in t_deinactivation_s (frequency should FALL as it rises)")
            for k, mv in mono.items():
                print(f"    {k:<16} monotone={mv['monotone_decreasing_in_tau_h']}  "
                      f"rank r={mv['spearman']:+.3f}")
            fr = [r["peak_hz"] for r in osc]
            print(f"\n  reachable frequency range over the declared priors: "
                  f"{min(fr):.2f} - {max(fr):.2f} Hz")
        report["sweep"] = rows_s
        report["sweep_monotonicity"] = mono

    if a.dt_check:
        dc = dt_convergence(dyn, a, dev)
        print(f"\ntimestep convergence (is the frequency the mechanism's or the "
              f"integrator's?)\n"
              f"  dt={dc['dt']['dt']:.1e}s -> {dc['dt']['peak_hz']:.3f} Hz, "
              f"ptp {dc['dt']['amplitude_ptp']:.4f}\n"
              f"  dt={dc['dt_half']['dt']:.1e}s -> {dc['dt_half']['peak_hz']:.3f} Hz, "
              f"ptp {dc['dt_half']['amplitude_ptp']:.4f}\n"
              f"  shift {dc['peak_shift_hz']:+.3f} Hz "
              f"({100*abs(dc['peak_shift_hz'])/max(dc['dt']['peak_hz'],1e-9):.1f}%), "
              f"amplitude x{dc['amplitude_ratio']:.3f}")
        report["dt_convergence"] = dc

    if a.bifurcation:
        print("\nbifurcation: T-current conductance vs the descending limb.")
        rows_b = bifurcation(dyn, a, dev)
        def osc_row(r, pre):
            return r[pre + "_peak_ratio"] > thresh and r[pre + "_amplitude_ptp"] > 0.01
        print(f"  {'g_t':>6} {'ct':>4} {'cx Hz':>7} {'cx ptp':>9} {'cx osc':>7} "
              f"{'th Hz':>7} {'th ptp':>9} {'th osc':>7}")
        for r in rows_b:
            print(f"  {r['g_t']:6.2f} {'on' if r['ct_intact'] else 'CUT':>4} "
                  f"{r['cortex_peak_hz']:7.2f} {r['cortex_amplitude_ptp']:9.4f} "
                  f"{'YES' if osc_row(r,'cortex') else 'no':>7} "
                  f"{r['thal_peak_hz']:7.2f} {r['thal_amplitude_ptp']:9.4f} "
                  f"{'YES' if osc_row(r,'thal') else 'no':>7}")
        def onset(intact):
            c = [r["g_t"] for r in rows_b if r["ct_intact"] == intact
                 and osc_row(r, "thal")]
            return min(c) if c else None
        on_i, on_c = onset(True), onset(False)
        print(f"\n  onset of the thalamic oscillation:  descending limb intact "
              f"g_t = {on_i}   cut g_t = {on_c}")
        if on_i is not None and on_c is not None:
            print(f"  gap = {on_c - on_i:+.2f} in g_t.  a POSITIVE gap means the "
                  f"cortical limb\n  supplies gain the thalamus alone does not have.")
        elif on_i is not None and on_c is None:
            print("  the oscillation exists ONLY with the descending limb intact "
                  "anywhere in\n  the swept range.")
        report["bifurcation"] = rows_b
        report["bifurcation_onset"] = {"ct_intact_g_t": on_i, "ct_cut_g_t": on_c,
                                       "gap": (on_c - on_i) if (on_i is not None
                                                                and on_c is not None)
                                       else None}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
