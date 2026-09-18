"""known answers for the thalamic module, and the sweep that sets its one fitted constant.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/gate_thalamus.py
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/gate_thalamus.py --sweep

Every gate's verdict is fixed here, in the code, before it runs, and a failure is recorded
as FAILED rather than rescored (CLAUDE.md).  The gates are about the MECHANISM, not about
whether the module is better than something else -- there is nothing to be better than.

  T0  BOUNDED.  Every state variable stays in [0, 1] under extreme drive, at three
      timesteps.  The exponential-Euler step makes this a property rather than a hope, so
      the gate exists to check that the claim is true of the code and not only of the
      algebra.
  T1  IDEMPOTENCE.  The same rollout with the same generator, twice, bit-identical.
  T2  REBOUND, the mechanism.  A hyperpolarising pulse, then release: the relay peak must
      arrive AFTER the pulse ends.  A filter of the input cannot do this, and a module that
      failed T2 while passing the frequency gates would be ringing for the wrong reason.
  T3  SPINDLE BAND.  At NREM2 polarisation the isolated loop's peak lands in 11-16 Hz.
      The catalogue's 13.45 Hz is a MEASUREMENT on held-out subjects, so the gate is the
      band and the measured value is reported beside it.
  T4  AROUSAL ORDER.  Sweeping arousal from waking to deep sleep, the peak frequency must
      fall -- tonic, then spindle, then delta.  Declared before the sweep is run.
  T5  GATING.  At waking polarisation a sensory drive passes through the relay (high
      correlation); at sleep polarisation it does not.  This is what "the thalamus gates
      the cortex" has to mean quantitatively.
  T5b GAIN, the replacement instrument for T5.  T5 stays FAILED in the log; it measured
      transmission as a CORRELATION, which is scale-free and cannot see a gain change at
      all.  T5b measures the transfer SLOPE, with its own threshold declared here before it
      runs: waking gain must be at least twice deep sleep's.
  T6  THE LOOP.  Cortex and thalamus closed on each other run bounded, and both conduction
      delays resolve to at least one step at the dt used.
  T7  DELTA.  At deep hyperpolarisation the relay must produce a 1-4 Hz peak above its own
      background: prominence > 0.  Declared in the module's docstring from the first
      version, so it is gated whether or not it passes.
  T8  SENSITIVITY, the instrument this module needed and did not have.  Every declared
      constant is swept +-50% and the spindle frequency re-measured.  A constant the module
      CLAIMS sets the clock (tau_A, tau_R) must move it by more than 1 Hz; every other
      constant's movement is reported.  Four separate sweeps in this module's first day
      each ended in "this parameter changes nothing", and each time that meant a mechanism
      was not engaging -- an inverted activation term, a gating variable reading the wrong
      voltage, a sag that shut off below the window it was meant to reach.  A parameter
      that does nothing is the cheapest available detector of a mechanism that is not
      wired in.

`--sweep` runs the tau_h_up sweep: the de-inactivation time constant against the isolated
loop's frequency.  It is the one constant in `ibm/thalamus.py` set by a measurement rather
than declared from the literature, and the sweep is how it is set -- transparently, with
the whole curve recorded, rather than by nudging it until a gate passes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import spectral as SP                                    # noqa: E402
from ibm.rhythms import RHYTHM                                    # noqa: E402
from ibm.thalamus import ThalamicField, ThalamicPriors, ThalamoCortical, units_from_regions  # noqa: E402

DT = 0.001
SPINDLE = RHYTHM["spindles"]          # band (11, 16), measured peak 13.45 Hz
DELTA = RHYTHM["delta"]               # band (1, 4)
AROUSAL_WAKE, AROUSAL_N2, AROUSAL_N3 = 1.0, 0.45, 0.10


def peak_hz(trace, dt, lo, hi, nperseg=2048):
    fs = 1.0 / dt
    freqs, psd = SP.welch_psd(trace, fs, nperseg=min(nperseg, trace.shape[-1]))
    return float(SP.peak_frequency(psd, freqs, lo, hi).mean()), freqs, psd


def run(th, steps, arousal, drive_sense=None, seed=0, b=1, record="R"):
    g = torch.Generator().manual_seed(seed)
    tr, st = th.rollout(steps, DT, arousal=arousal, drive_sense=drive_sense,
                        noise_gen=g, b=b, record=record)
    return tr, st


def gate_bounded(th):
    worst, detail = 0.0, []
    for dt in (0.001, 0.005, 0.020):
        for drive in (-2.0, 0.0, 2.0):
            g = torch.Generator().manual_seed(11)
            state = th.init_state(1)
            for _ in range(int(4.0 / dt)):
                z = torch.randn(1, th.n, generator=g)
                state = th.step(state, dt, drive_sense=drive, arousal=0.45, noise=z)
            for k in ("R", "T", "h", "sA", "sB"):
                v = state[k]
                bad = float(max((v.min() * -1).clamp_min(0).max(), (v - 1).clamp_min(0).max()))
                worst = max(worst, bad)
            detail.append({"dt": dt, "drive": drive,
                           "R": [float(state["R"].min()), float(state["R"].max())]})
    return {"ok": worst <= 0.0, "worst_excursion": worst, "cases": detail}


def gate_idempotent(th):
    a, _ = run(th, 2000, AROUSAL_N2, seed=7)
    b, _ = run(th, 2000, AROUSAL_N2, seed=7)
    return {"ok": bool(torch.equal(a, b)), "max_abs_diff": float((a - b).abs().max())}


def gate_rebound(th):
    """a hyperpolarising pulse, then release.  The peak must come AFTER the release."""
    steps = 1200
    pulse = torch.zeros(1, steps, th.n)
    t0, t1 = 200, 500                      # 300 ms of hyperpolarisation
    pulse[:, t0:t1] = -0.8
    g = torch.Generator().manual_seed(3)
    tr, _ = th.rollout(steps, DT, arousal=AROUSAL_N2, drive_sense=pulse, noise_gen=None, b=1)
    seg = tr[0, t0:, 0]
    k = int(torch.argmax(seg)) + t0
    return {"ok": k > t1, "peak_step": k, "pulse_ends": t1,
            "peak_ms_after_release": (k - t1) * DT * 1000,
            "peak_value": float(tr[0, k, 0]),
            "value_during_pulse": float(tr[0, t0:t1, 0].max())}


def gate_spindle(th):
    tr, _ = run(th, 20000, AROUSAL_N2, seed=5)
    f, freqs, psd = peak_hz(tr[..., 0], DT, 5.0, 25.0)
    prom = float(SP.peak_prominence(psd, freqs, SPINDLE.band[0], SPINDLE.band[1],
                                    fit_lo=1.0, fit_hi=45.0).mean())
    lo, hi = SPINDLE.band
    return {"ok": lo <= f <= hi, "measured_hz": f, "band": [lo, hi],
            "catalogue_peak_hz": SPINDLE.peak, "prominence_decades": prom,
            "off_measured_hz": f - SPINDLE.peak}


def gate_arousal_order(th):
    rows = []
    for a in (1.0, 0.8, 0.6, 0.45, 0.3, 0.15, 0.05):
        tr, _ = run(th, 20000, a, seed=9)
        f, freqs, psd = peak_hz(tr[..., 0], DT, 0.5, 25.0)
        prom_sp = float(SP.peak_prominence(psd, freqs, 11.0, 16.0).mean())
        prom_dl = float(SP.peak_prominence(psd, freqs, 1.0, 4.0).mean())
        rows.append({"arousal": a, "peak_hz": f, "spindle_prominence": prom_sp,
                     "delta_prominence": prom_dl, "mean_rate_hz": float(tr.mean() * 100)})
    fs = [r["peak_hz"] for r in rows]
    falling = all(fs[i] >= fs[i + 1] - 1.0 for i in range(len(fs) - 1))
    return {"ok": falling, "monotone_falling_within_1Hz": falling, "sweep": rows}


def gate_gating(th):
    """the sensory drive must reach the cortex when awake and not when asleep."""
    steps = 8000
    g = torch.Generator().manual_seed(21)
    sig = 0.25 * torch.randn(1, steps, 1, generator=g).repeat(1, 1, th.n)
    out = {}
    for name, a in (("wake", AROUSAL_WAKE), ("n2", AROUSAL_N2), ("n3", AROUSAL_N3)):
        tr, _ = th.rollout(steps, DT, arousal=a, drive_sense=sig, noise_gen=None, b=1)
        x = tr[0, 2000:, 0]
        y = sig[0, 2000:, 0]
        x = x - x.mean()
        y = y - y.mean()
        r = float((x * y).sum() / (x.norm() * y.norm() + 1e-12))
        out[name] = {"corr_with_drive": r, "mean_rate_hz": float(tr.mean() * 100)}
    ok = out["wake"]["corr_with_drive"] > out["n3"]["corr_with_drive"] + 0.1
    return {"ok": ok, "states": out,
            "rule": "wake transmission must exceed deep-sleep transmission by > 0.1"}


def gate_gain(th):
    """T5b.  Transmission is a GAIN, so measure the slope, not the correlation."""
    steps = 8000
    g = torch.Generator().manual_seed(21)
    sig = 0.25 * torch.randn(1, steps, 1, generator=g).repeat(1, 1, th.n)
    out = {}
    for name, a in (("wake", AROUSAL_WAKE), ("n2", AROUSAL_N2), ("n3", AROUSAL_N3)):
        tr, _ = th.rollout(steps, DT, arousal=a, drive_sense=sig, noise_gen=None, b=1)
        x = tr[0, 2000:, 0]
        y = sig[0, 2000:, 0]
        x = x - x.mean()
        y = y - y.mean()
        slope = float((x * y).sum() / (y * y).sum())
        out[name] = {"transfer_slope": slope, "mean_rate_hz": float(tr.mean() * 100)}
    ratio = out["wake"]["transfer_slope"] / max(1e-9, out["n3"]["transfer_slope"])
    return {"ok": ratio >= 2.0, "wake_over_n3": ratio, "states": out,
            "rule": "waking transfer slope >= 2x deep sleep's",
            "replaces": "T5, which used a scale-free correlation and stays FAILED"}


def gate_delta(th):
    """T7.  The module's docstring claims delta at deep hyperpolarisation.  Does it?"""
    tr, _ = run(th, 30000, 0.05, seed=13)
    x = tr[..., 0]
    fs = 1.0 / DT
    freqs, psd = SP.welch_psd(x, fs, nperseg=8192)
    prom = float(SP.peak_prominence(psd, freqs, DELTA.band[0], DELTA.band[1]).mean())
    pk = float(SP.peak_frequency(psd, freqs, DELTA.band[0], DELTA.band[1]).mean())
    v = x[0, 2000:]
    bursts = int(((v[:-1] < 0.1) & (v[1:] >= 0.1)).sum()) / (len(v) * DT)
    return {"ok": prom > 0.0, "delta_prominence_decades": prom,
            "in_band_peak_hz": pk, "burst_rate_hz": bursts,
            "note": "peak_frequency returns a number whether or not there is a peak; "
                    "the prominence is the gate"}


def gate_sensitivity():
    """T8.  Sweep every declared constant +-50% and watch the spindle frequency."""
    base = ThalamicPriors()
    fields = [f for f in base.__dataclass_fields__
              if not f.startswith("d_") and f not in ("sigma", "arousal_span")]

    def freq(pr):
        th = ThalamicField(1, priors=pr, learn=False)
        tr, _ = run(th, 12000, AROUSAL_N2, seed=5)
        f, _fr, _p = peak_hz(tr[..., 0], DT, 4.0, 30.0, nperseg=4096)
        return f

    f0 = freq(base)
    rows = []
    for name in fields:
        v = getattr(base, name)
        lo = ThalamicPriors(**{**{k: getattr(base, k) for k in base.__dataclass_fields__},
                               name: v * 0.5})
        hi = ThalamicPriors(**{**{k: getattr(base, k) for k in base.__dataclass_fields__},
                               name: v * 1.5})
        flo, fhi = freq(lo), freq(hi)
        rows.append({"param": name, "value": v, "hz_at_half": flo, "hz_at_1p5x": fhi,
                     "span_hz": abs(fhi - flo)})
    rows.sort(key=lambda r: -r["span_hz"])
    clock = {r["param"]: r["span_hz"] for r in rows if r["param"] in ("tau_A", "tau_R")}
    ok = all(v > 1.0 for v in clock.values())
    inert = [r["param"] for r in rows if r["span_hz"] < 0.2]
    return {"ok": ok, "baseline_hz": f0, "claimed_clock_span_hz": clock,
            "rule": "tau_A and tau_R must each move the spindle frequency by > 1 Hz",
            "inert_within_0.2hz": inert, "sweep": rows}


def gate_loop():
    """cortex and thalamus closed on each other: bounded, and the delays resolve."""
    from ibm.substrate import build_sheet
    cx = build_sheet(256, 16, 0.25, seed=0, device="cpu", long_topology="random")
    uos, unames = units_from_regions(cx.region_names)
    th = ThalamicField(len(unames))
    tc = ThalamoCortical(cx, th, uos)
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        ctx, thal, info = tc.rollout(3000, 0.002, arousal=AROUSAL_N2, noise_gen=g, b=1)
    lags = info["delays_in_steps"]
    bad = float(max((ctx.min() * -1).clamp_min(0).max(), (ctx - 1).clamp_min(0).max(),
                    (thal.min() * -1).clamp_min(0).max(), (thal - 1).clamp_min(0).max()))
    return {"ok": bad <= 0.0 and min(lags.values()) >= 1, "worst_excursion": bad,
            "delays_in_steps": lags, "units": unames,
            "cortex_rate_hz": float(ctx.mean() * 100),
            "thalamus_rate_hz": float(thal.mean() * 100)}


def sweep_tau_h(values):
    """the constant the measurement sets: tau_h_up against the isolated loop's frequency."""
    rows = []
    for v in values:
        pr = ThalamicPriors(tau_h_up=v)
        th = ThalamicField(1, priors=pr, learn=False)
        tr, _ = run(th, 20000, AROUSAL_N2, seed=5)
        f, freqs, psd = peak_hz(tr[..., 0], DT, 4.0, 30.0)
        prom = float(SP.peak_prominence(psd, freqs, 11.0, 16.0).mean())
        rows.append({"tau_h_up_ms": v * 1000, "peak_hz": f, "spindle_prominence": prom})
        print(f"  tau_h_up {v * 1000:6.1f} ms -> {f:6.2f} Hz  (spindle prominence "
              f"{prom:+.3f})", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="out/gate_thalamus.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rec = {"script": "scripts/gate_thalamus.py", "dt": DT, "gates": {}}

    def save():
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=str)

    save()
    if a.sweep:
        print("tau_h_up sweep (isolated relay+TRN unit, NREM2 polarisation):", flush=True)
        rec["sweep_tau_h"] = sweep_tau_h([0.020, 0.030, 0.040, 0.050, 0.060, 0.080,
                                          0.100, 0.130])
        save()
        print(f"\nwrote {a.out}")
        return 0

    th = ThalamicField(4, learn=False)
    order = [("T0_bounded", lambda: gate_bounded(th)),
             ("T1_idempotent", lambda: gate_idempotent(th)),
             ("T2_rebound", lambda: gate_rebound(th)),
             ("T3_spindle_band", lambda: gate_spindle(th)),
             ("T4_arousal_order", lambda: gate_arousal_order(th)),
             ("T5_gating", lambda: gate_gating(th)),
             ("T5b_gain", lambda: gate_gain(th)),
             ("T6_loop", gate_loop),
             ("T7_delta", lambda: gate_delta(th)),
             ("T8_sensitivity", gate_sensitivity)]
    ok_all = True
    for name, fn in order:
        r = fn()
        rec["gates"][name] = r
        save()
        ok_all &= bool(r.get("ok"))
        head = {k: v for k, v in r.items() if k not in ("sweep", "cases", "states")}
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {name}: {head}", flush=True)
        if name == "T4_arousal_order":
            for row in r["sweep"]:
                print(f"      arousal {row['arousal']:.2f}  peak {row['peak_hz']:6.2f} Hz  "
                      f"spindle {row['spindle_prominence']:+.3f}  "
                      f"delta {row['delta_prominence']:+.3f}  "
                      f"rate {row['mean_rate_hz']:5.2f} Hz", flush=True)
        if name == "T5_gating":
            for k, v in r["states"].items():
                print(f"      {k:5s} corr {v['corr_with_drive']:+.3f}  "
                      f"rate {v['mean_rate_hz']:5.2f} Hz", flush=True)
        if name == "T5b_gain":
            for k, v in r["states"].items():
                print(f"      {k:5s} slope {v['transfer_slope']:+.4f}  "
                      f"rate {v['mean_rate_hz']:5.2f} Hz", flush=True)
        if name == "T8_sensitivity":
            for row in r["sweep"][:6]:
                print(f"      {row['param']:12s} {row['hz_at_half']:6.2f} -> "
                      f"{row['hz_at_1p5x']:6.2f} Hz   span {row['span_hz']:5.2f}", flush=True)
            print(f"      inert (< 0.2 Hz over a 3x sweep): "
                  f"{', '.join(r['inert_within_0.2hz']) or 'none'}", flush=True)
    rec["all_gates_ok"] = ok_all
    save()
    print(f"\n{'ALL GATES PASS' if ok_all else 'SOME GATES FAILED'} -- wrote {a.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
