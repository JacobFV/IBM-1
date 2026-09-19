"""known answers for `ibm/cord.py`, and the sweep that sets its one fitted constant.

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/gate_cord.py
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/gate_cord.py --sweep

CPU ONLY.  On the GB10 the GPU shares one memory pool with the CPU, so a GPU OOM is a
machine-wide OOM (CLAUDE.md).  Nothing here needs a GPU.

Every verdict below is fixed HERE, in this docstring and in the code, before any of it
runs.  A failure is recorded as FAILED and left failed; instruments may change after a
failure, thresholds may not (CLAUDE.md).

  S0  BOUNDED.  Every state variable stays in [0, 1] under extreme drive, extreme
      cortical input and extreme joint perturbation, at four timesteps spanning 50x.
      The exponential-Euler construction makes this a property rather than a hope, so
      the gate checks that the claim is true of the CODE and not only of the algebra.
  S1  IDEMPOTENCE.  Two things, because they are different claims.  (a) The same
      rollout with the same generator, twice, bit-identical.  (b) `step` called TWICE
      from the same state with the same arguments returns the same state -- the general
      form of CLAUDE.md's row 23, and the only check that tells you the thing is a
      function at all rather than a process with memory you did not declare.
  S2  ALTERNATION.  Flexor and extensor must be in antiphase, 150-210 degrees, as a
      measured PHASE ANGLE and not a correlation.  Left flexor against right flexor
      likewise: `flexor_extensor_antiphase` declares both in one row.  The phase-locking
      value is reported beside every angle, because a phase difference computed on two
      signals that are not locked is the same kind of non-measurement as a peak
      frequency with no peak under it.
      The gate is measured from THREE initial conditions -- the declared diagonal, a
      larger one, and a SYNCHRONOUS start with all four half-centres equal.  The third
      is the one that can fail: if the antiphase were the declared initial condition
      persisting rather than an attractor, a synchronous start would stay synchronous
      and come back at 0 degrees.
      A MIRRORED start was tried here first and is not a control at all: 180 degrees is
      its own mirror image, so that arm returns 180 whether the phase is an attractor
      or a memory.  Recorded because it looked like a control and could not fail.
  S3  FREQUENCY FOLLOWS DRIVE, PHASE DOES NOT.  The descending drive is swept across the
      model's operating window and the locomotor frequency measured at each level.
      Four requirements, all declared here:
        (a) MONOTONE -- strictly rising at every step of the sweep;
        (b) IN BAND -- every measured frequency inside the catalogue's 0.5-3.0 Hz;
        (c) PHASE HELD -- the antiphase angle inside S2's 150-210 window at every level;
        (d) COVERS THE BAND -- the sweep must reach the outer quarters of the declared
            band in log-frequency: lowest <= 0.783 Hz and highest >= 1.917 Hz, those
            being 0.5*6^0.25 and 0.5*6^0.75 for the declared interval (0.5, 3.0).
            This is what "monotonically across the declared 0.5-3 Hz range" is taken to
            mean, since a sweep cannot literally touch both open ends while also lying
            inside them.  The measured endpoints are reported raw beside it either way.
      This is the actual specification of a pattern generator (`locomotor_cpg`'s own
      note in `ibm/rhythms.py`); the band alone is not.
  S4  IT GENERATES RATHER THAN FOLLOWS.  With the descending drive a CONSTANT and no
      rhythmic input anywhere, the locomotor band must still rise above its own 1/f
      background: prominence > 0.  Reported with the frequency, never without it.
      The separating CONTROL is an arm with `g_nap = 0` -- the inward current that is
      the whole mechanism removed, every other constant, the drive and the NOISE DRAW
      identical.  Its prominence must be <= 0.  Without that arm the gate could pass
      because the injected noise happened to have power in 0.5-3 Hz, which is a
      certificate of nothing (CLAUDE.md: a control that can pass for a reason unrelated
      to what it tests).
  S5  REFLEX LATENCY.  A step stretch is imposed on the joint and the motoneuron pool's
      response must arrive at the catalogue's declared conduction delay and NOT sooner.
      `ibm/rhythms.py`'s `stretch_reflex` declares spindle_afferent -> mn_pool at
      10-18 ms and `ibm/cord.py` takes the midpoint, 14 ms.  Measured as the first step
      at which a perturbed run differs from an unperturbed one that is identical in
      every other respect, so the onset is exact rather than threshold-dependent.
      VERDICT: latency in [10.0, 18.0] ms (the declared interval) AND >= 14.0 ms (the
      declared value; it cannot arrive before the axon does).  The measurement carries
      a known +2 steps of integration overhead -- the perturbation reaches the joint
      state on one step and the spindle on the next -- so at DT_FINE = 1 ms the expected
      reading is 16 ms, and that arithmetic is written down here before the run.
  S6  DEAFFERENTATION.  `afferent_gain = 0` cuts the spindle afferents, both the
      monosynaptic arc and their input to the half-centres.  The rhythm must STAND
      (prominence > 0) -- fictive locomotion -- while the muscle-force amplitude
      changes by at least 10%.  If the rhythm dies, what was built is a reflex loop and
      not a CPG, and this gate FAILS.  It is the gate in this file that can actually
      fail, and it is the reason `w_aff_hc` exists at all: with no afferent path to the
      rhythm generator there would be nothing to cut and the gate could not fail.
  S7  SENSITIVITY.  Every one of `CordPriors`' declared constants swept +-50% and the
      locomotor frequency AND the antiphase angle re-measured.  Sorted by movement,
      with everything inert over the 3x sweep listed.  A parameter that changes nothing
      is the cheapest available detector of a mechanism that is not wired in -- five
      separate diagnoses in `ibm/thalamus.py`'s first day were exactly this.
      VERDICT: the constants this module CLAIMS are its clock -- `tau_h_up` and
      `g_nap` -- must each move the locomotor frequency by more than 0.3 Hz, and
      `w_recip` must move something.  Everything else is reported, not gated.

      S7 FAILED ON ITS FIRST RUN AND IS LEFT FAILED.  `g_nap` moved the frequency by
      0.005 Hz, which reads as the most inert constant in the module and is the
      opposite of the truth: at BOTH endpoints of its 3x sweep the rhythm is DESTROYED
      (prominence -1.33 and -1.29), and the two "frequencies", 3.325 and 3.330 Hz, are
      `peak_frequency`'s flat-spectrum answer -- the midpoint of the 0.5-6.0 Hz band
      the query asked about.  The first version of this gate quoted those frequencies
      with no prominence beside them, which is the one thing CLAUDE.md says never to
      do, in the instrument built to catch exactly that.  The sweep table now carries
      `alive_at_half` / `alive_at_1p5x` and a per-row verdict, and `inert` means BOTH
      endpoints alive and neither moving; a constant whose sweep kills the rhythm is
      listed under `breaks_the_rhythm` instead.  The THRESHOLD is untouched: `g_nap`
      still does not move the frequency by 0.3 Hz, so S7 is still FAILED, and the
      honest reading is that the declared verdict was the wrong test for a constant
      whose sweep leaves the oscillatory regime entirely.

ONE INSTRUMENT, ONE SETTING.  Every spectral number in this file comes from
`ibm/spectral.py` at the settings pinned below (`SEG_S`, `SECS`, `BURN`), because the
prominence of a near-limit-cycle is sensitive to the Welch segment length -- measured,
it moved 2.3 decades between a 4 s and a 16 s segment on the SAME trace -- and two
numbers from two settings are a comparison between two instruments.

AND THE NOISE HAS TO BE ON.  With noise off this cord is an almost noiseless limit
cycle: the inter-harmonic bins are numerically empty, `aperiodic_fit` has no background
to fit, and `peak_prominence` returned -4.5 decades for a clean 1.25 Hz oscillation and
-0.4 for a DEAD one -- the ordering inverted.  Every rhythm gate here runs with the
module's declared background noise, from a generator created once and passed in.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import spectral as SP                                          # noqa: E402
from ibm.cord import CordPriors, SpinalCord, POOL_NAMES                 # noqa: E402
from ibm.rhythms import LOOPS, RHYTHM                                   # noqa: E402

# ---- the catalogue, read rather than retyped -----------------------------------------
LOCO = RHYTHM["locomotor_cpg"]                  # band (0.5, 3.0), declared peak 1.2
ANTI = RHYTHM["flexor_extensor_antiphase"]      # target (150, 210) degrees
_REFLEX_EDGE = next(e for e in LOOPS["stretch_reflex"].edges
                    if e.src == "spindle_afferent" and e.dst == "mn_pool")
REFLEX_MS = _REFLEX_EDGE.delay_ms               # (10.0, 18.0)
BAND = LOCO.band
PHASE_LO, PHASE_HI = ANTI.measure["target"]

# ---- pinned instrument settings ------------------------------------------------------
DT = 0.002            # rhythm gates
DT_FINE = 0.001       # S5, so the 14 ms delay resolves to 14 whole steps
DT_SWEEP = 0.004      # S7 and --sweep; delays reported in steps at this dt too
SECS, BURN, SEG_S = 60.0, 12.0, 8.0
SECS_SWEEP, BURN_SWEEP = 32.0, 10.0
EDGE_S = 2.0          # discarded at each end before any phase average: bandpass_analytic
                      # is circular, so the first and last cycles wrap into each other
NOISE_SEED = 1704
DRIVE_NOMINAL = 0.50
# the drive sweep's grid.  Its top is just under the co-activation bifurcation and its
# bottom is zero descending command; both were located before the gate was written and
# are reported in the JSON, so the grid is a declared operating window and not a search.
DRIVE_LEVELS = (0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05)
# S3 (d): the outer quarters of the declared band in log-frequency.
_R = (BAND[1] / BAND[0]) ** 0.25
COVER_LO, COVER_HI = BAND[0] * _R, BAND[0] * _R ** 3
# S7 verdicts
CLOCK_MIN_HZ = 0.30
INERT_HZ, INERT_DEG = 0.05, 5.0


# --------------------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------------------
def run(cord, drive, dt=DT, secs=SECS, burn=BURN, gain=1.0, seed=NOISE_SEED,
        state=None, record=("V", "M", "F", "S")):
    g = torch.Generator().manual_seed(seed)
    return cord.rollout(int(round(secs / dt)), dt, state=state, drive=drive,
                        afferent_gain=gain, noise_gen=g, b=1,
                        burn=int(round(burn / dt)), record=record)


def spectrum(x, dt, seg_s=SEG_S):
    fs = 1.0 / dt
    return SP.welch_psd(x, fs, nperseg=int(round(seg_s * fs)))


def peak_and_prominence(x, dt, seg_s=SEG_S):
    """the only way a frequency leaves this file: with its prominence beside it."""
    freqs, psd = spectrum(x, dt, seg_s)
    f = float(SP.peak_frequency(psd, freqs, BAND[0], BAND[1] * 2.0))
    p = float(SP.peak_prominence(psd, freqs, BAND[0], BAND[1]))
    return f, p


def phase_difference_deg(x, y, dt, edge_s=EDGE_S):
    """circular-mean phase of x relative to y in the locomotor band, in [0, 360).

    Built on `ibm/spectral.py`'s `bandpass_analytic`, so the filter is the same one
    every other phase quantity in this programme uses.  The phase-locking value comes
    back with it: an angle between two signals that do not hold a phase is not a
    measurement, in exactly the way a peak frequency on a flat spectrum is not one.
    """
    k = int(round(edge_s / dt))
    zx = SP.bandpass_analytic(x, 1.0 / dt, BAND[0], BAND[1])[k:-k]
    zy = SP.bandpass_analytic(y, 1.0 / dt, BAND[0], BAND[1])[k:-k]
    d = zx * zy.conj()
    u = d / (d.abs() + 1e-20)
    ang = math.degrees(math.atan2(float(u.imag.mean()), float(u.real.mean()))) % 360.0
    plv = float(SP.phase_locking_value(x[k:-k], y[k:-k], 1.0 / dt, BAND[0], BAND[1]))
    return ang, plv


def _in_window(a):
    return PHASE_LO <= a <= PHASE_HI


def measure(cord, drive, dt=DT, secs=SECS, burn=BURN, gain=1.0, seed=NOISE_SEED,
            state=None):
    """one arm: frequency, prominence, both antiphase angles, amplitudes."""
    tr, _st, info = run(cord, drive, dt=dt, secs=secs, burn=burn, gain=gain, seed=seed,
                        state=state)
    V, F = tr["V"][0], tr["F"][0]
    f, p = peak_and_prominence(V[:, 0], dt)
    fe_ang, fe_plv = phase_difference_deg(V[:, 0], V[:, 1], dt)      # Lflex vs Lext
    lr_ang, lr_plv = phase_difference_deg(V[:, 0], V[:, 2], dt)      # Lflex vs Rflex
    return {"peak_hz": f, "prominence_decades": p,
            "flex_ext_deg": fe_ang, "flex_ext_plv": fe_plv,
            "left_right_deg": lr_ang, "left_right_plv": lr_plv,
            "hc_amplitude_sd": float(V[:, 0].std()),
            "force_amplitude_sd": float(F[:, 0].std()),
            "force_mean": float(F[:, 0].mean()),
            "delays_in_steps": info["delays_in_steps"]}


# --------------------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------------------
BOUNDED_KEYS = ("V", "h", "sR", "sC", "M", "F", "th", "S", "Lm")


def gate_bounded(_cord):
    """S0.  Nothing can leave [0, 1], at any dt, under any input."""
    worst, cases = 0.0, []
    for dt in (0.001, 0.005, 0.020, 0.050):
        for drive in (-2.0, 0.0, 2.0, 5.0):
            for cx in (-3.0, 3.0):
                c = SpinalCord(learn=False)
                g = torch.Generator().manual_seed(11)
                st = c.init_state(1, dt)
                cxd = torch.zeros(1, 4) + cx
                pert = torch.zeros(1, 2) + (0.5 if cx > 0 else -0.5)
                for _ in range(int(round(6.0 / dt))):
                    z = torch.randn(1, 4, generator=g)
                    st = c.step(st, dt, drive=drive, drive_cortex=cxd, perturb=pert,
                                noise=z)
                bad = 0.0
                for k in BOUNDED_KEYS + ("ring_V", "ring_M", "ring_S"):
                    v = st[k]
                    bad = max(bad, float((-v).clamp_min(0).max()),
                              float((v - 1.0).clamp_min(0).max()))
                worst = max(worst, bad)
                cases.append({"dt": dt, "drive": drive, "cortex": cx,
                              "worst": bad, "nonfinite": bool(
                                  any(not torch.isfinite(st[k]).all()
                                      for k in BOUNDED_KEYS))})
    nf = any(c["nonfinite"] for c in cases)
    return {"ok": worst <= 0.0 and not nf, "worst_excursion": worst,
            "any_nonfinite": nf, "n_cases": len(cases), "cases": cases}


def gate_idempotent(cord):
    """S1.  (a) two identical rollouts; (b) `step` called twice at the same input."""
    pert = torch.zeros(1, int(round(8.0 / DT)), 2)
    pert[:, 2000:, 0] = 0.2
    cx = torch.zeros(1, 4) + 0.1

    def one():
        g = torch.Generator().manual_seed(99)
        return cord.rollout(int(round(8.0 / DT)), DT, drive=0.5, drive_cortex=cx,
                            perturb=pert, noise_gen=g, b=1, record=("V", "M", "F", "S"))

    a, _, _ = one()
    b, _, _ = one()
    roll = max(float((a[k] - b[k]).abs().max()) for k in a)

    st = cord.init_state(1, DT)
    for _ in range(500):
        st = cord.step(st, DT, drive=0.5)
    z = torch.randn(1, 4, generator=torch.Generator().manual_seed(5))
    s1 = cord.step(st, DT, drive=0.5, drive_cortex=cx, perturb=pert[:, 0], noise=z)
    s2 = cord.step(st, DT, drive=0.5, drive_cortex=cx, perturb=pert[:, 0], noise=z)
    stepdiff = max(float((s1[k] - s2[k]).abs().max()) for k in s1)
    return {"ok": roll == 0.0 and stepdiff == 0.0,
            "rollout_max_abs_diff": roll, "step_twice_max_abs_diff": stepdiff,
            "rule": "both must be exactly 0.0"}


def gate_alternation(cord):
    """S2.  A measured phase angle, from three initial conditions."""
    arms = {}
    for name, asym, sync in (("declared_diagonal", 0.20, False),
                             ("larger_diagonal", 0.45, False),
                             ("synchronous_start", 0.20, True)):
        st = cord.init_state(1, DT, asym=asym)
        if sync:
            # all four equal: no asymmetry at all to persist.  The antiphase has to be
            # built by the coupling or this arm returns 0 degrees.
            st["V"] = torch.zeros_like(st["V"]) + asym
        arms[name] = measure(cord, DRIVE_NOMINAL, state=st,
                             seed=NOISE_SEED + (7 if sync else 0))
    fe = [a["flex_ext_deg"] for a in arms.values()]
    lr = [a["left_right_deg"] for a in arms.values()]
    ok = all(_in_window(x) for x in fe) and all(_in_window(x) for x in lr)
    ok = ok and all(a["flex_ext_plv"] > 0.9 and a["left_right_plv"] > 0.9
                    for a in arms.values())
    return {"ok": bool(ok), "window_deg": [PHASE_LO, PHASE_HI],
            "flex_ext_deg": [round(x, 2) for x in fe],
            "left_right_deg": [round(x, 2) for x in lr],
            "flex_ext_spread_deg": round(max(fe) - min(fe), 2),
            "left_right_spread_deg": round(max(lr) - min(lr), 2),
            "rule": "every angle in the declared window and every PLV > 0.9",
            "arms": arms}


def gate_frequency_follows_drive(cord):
    """S3.  The specification of a pattern generator, not its band."""
    rows = []
    for d in DRIVE_LEVELS:
        m = measure(cord, d)
        m["drive"] = d
        rows.append(m)
    fs = [r["peak_hz"] for r in rows]
    mono = all(fs[i] < fs[i + 1] for i in range(len(fs) - 1))
    in_band = all(BAND[0] <= f <= BAND[1] for f in fs)
    phase_held = all(_in_window(r["flex_ext_deg"]) and _in_window(r["left_right_deg"])
                     for r in rows)
    covers = (min(fs) <= COVER_LO) and (max(fs) >= COVER_HI)
    alive = all(r["prominence_decades"] > 0.0 for r in rows)
    return {"ok": bool(mono and in_band and phase_held and covers and alive),
            "monotone_rising": mono, "all_in_band": in_band,
            "phase_held_in_window": phase_held, "rhythm_alive_at_every_level": alive,
            "covers_declared_band": covers,
            "coverage_rule_hz": [round(COVER_LO, 3), round(COVER_HI, 3)],
            "measured_hz_range": [round(min(fs), 3), round(max(fs), 3)],
            "declared_band_hz": list(BAND), "declared_peak_hz": LOCO.peak,
            "span_ratio": round(max(fs) / max(min(fs), 1e-9), 2),
            "sweep": rows}


def gate_generates(cord):
    """S4.  Constant drive, no rhythmic input, plus the g_nap = 0 control."""
    intact = measure(cord, DRIVE_NOMINAL)
    pr0 = CordPriors(g_nap=0.0)
    control = measure(SpinalCord(priors=pr0, learn=False), DRIVE_NOMINAL)
    ok = intact["prominence_decades"] > 0.0 and control["prominence_decades"] <= 0.0
    return {"ok": bool(ok),
            "prominence_decades": intact["prominence_decades"],
            "peak_hz": intact["peak_hz"],
            "control_g_nap_0_prominence": control["prominence_decades"],
            "control_g_nap_0_peak_hz": control["peak_hz"],
            "control_g_nap_0_amplitude_sd": control["hc_amplitude_sd"],
            "drive_was_constant": float(DRIVE_NOMINAL),
            "rule": "intact prominence > 0 AND the g_nap=0 control <= 0; the control "
                    "shares the drive, every other constant and the noise draw"}


def gate_reflex_latency(cord):
    """S5.  Onset measured as the first step a perturbed run differs from a twin."""
    dt = DT_FINE
    n = int(round(2.0 / dt))
    t0 = int(round(1.0 / dt))
    pert = torch.zeros(1, n, 2)
    pert[:, t0:, 0] = 0.30                      # a step stretch of the LEFT joint
    base, _, info = cord.rollout(n, dt, drive=DRIVE_NOMINAL, b=1,
                                 record=("M", "S", "th"))
    test, _, _ = cord.rollout(n, dt, drive=DRIVE_NOMINAL, perturb=pert, b=1,
                              record=("M", "S", "th"))
    lag = info["delays_in_steps"]["spindle_to_mn"]
    out = {}
    for key in ("th", "S", "M"):
        j = 0 if key == "th" else 1        # the STRETCHED muscle is the left extensor
        d = (test[key][0, :, j] - base[key][0, :, j]).abs()
        hit = (d > 0.0).nonzero()
        out[key] = (float(int(hit[0]) - t0) * dt * 1000.0) if len(hit) else float("nan")
    lat = out["M"]
    declared = cord.pr.d_spindle_mn_s * 1000.0
    ok = (REFLEX_MS[0] <= lat <= REFLEX_MS[1]) and lat >= declared
    return {"ok": bool(ok), "measured_latency_ms": lat,
            "declared_delay_ms": declared, "declared_interval_ms": list(REFLEX_MS),
            "delay_in_steps": lag, "dt_ms": dt * 1000.0,
            "joint_onset_ms": out["th"], "spindle_onset_ms": out["S"],
            "expected_ms": declared + 2.0 * dt * 1000.0,
            "rule": "in the declared 10-18 ms interval AND not sooner than 14 ms; the "
                    "+2 step integration overhead is declared in the docstring"}


def gate_deafferentation(cord):
    """S6.  Cut the spindles.  The rhythm must stand; the amplitude must change."""
    intact = measure(cord, DRIVE_NOMINAL, gain=1.0)
    cut = measure(cord, DRIVE_NOMINAL, gain=0.0)
    amp_ratio = cut["force_amplitude_sd"] / max(1e-9, intact["force_amplitude_sd"])
    stands = cut["prominence_decades"] > 0.0
    changed = abs(amp_ratio - 1.0) >= 0.10
    return {"ok": bool(stands and changed),
            "rhythm_stands": stands, "amplitude_changed": changed,
            "intact_peak_hz": intact["peak_hz"],
            "intact_prominence": intact["prominence_decades"],
            "cut_peak_hz": cut["peak_hz"], "cut_prominence": cut["prominence_decades"],
            "force_amplitude_ratio_cut_over_intact": round(amp_ratio, 4),
            "frequency_shift_hz": round(cut["peak_hz"] - intact["peak_hz"], 4),
            "flex_ext_deg_cut": round(cut["flex_ext_deg"], 2),
            "rule": "prominence > 0 with the afferents cut AND force amplitude moves "
                    ">= 10%.  A rhythm that dies here was a reflex loop.",
            "intact": intact, "cut": cut}


def _sweep_point(pr):
    c = SpinalCord(priors=pr, learn=False)
    m = measure(c, DRIVE_NOMINAL, dt=DT_SWEEP, secs=SECS_SWEEP, burn=BURN_SWEEP)
    return m["peak_hz"], m["prominence_decades"], m["flex_ext_deg"], m["flex_ext_plv"]


def gate_sensitivity():
    """S7.  Every declared constant +-50%, frequency and antiphase angle re-measured."""
    base = CordPriors()
    fields = list(base.__dataclass_fields__)
    f0, p0, a0, _ = _sweep_point(base)
    rows = []
    for name in fields:
        v = getattr(base, name)
        kw = {k: getattr(base, k) for k in fields}
        flo, plo, alo, _ = _sweep_point(CordPriors(**{**kw, name: v * 0.5}))
        fhi, phi, ahi, _ = _sweep_point(CordPriors(**{**kw, name: v * 1.5}))
        d_ang = abs(((ahi - alo + 180.0) % 360.0) - 180.0)
        # A frequency with no prominence under it is not a frequency, so every endpoint
        # is labelled before its span is read.  `peak_frequency` is a soft-argmax and
        # returns the band's midpoint on a flat spectrum, which looks exactly like a
        # measurement and is the reason this classification exists (see the docstring).
        live_lo, live_hi = plo > 0.0, phi > 0.0
        if live_lo and live_hi:
            verdict = ("moves" if (abs(fhi - flo) >= INERT_HZ or d_ang >= INERT_DEG)
                       else "inert")
        elif live_lo or live_hi:
            verdict = "breaks_the_rhythm_at_one_end"
        else:
            verdict = "breaks_the_rhythm_at_both_ends"
        rows.append({"param": name, "value": v, "verdict": verdict,
                     "hz_at_half": round(flo, 3), "hz_at_1p5x": round(fhi, 3),
                     "span_hz": round(abs(fhi - flo), 3),
                     "prom_at_half": round(plo, 2), "prom_at_1p5x": round(phi, 2),
                     "alive_at_half": live_lo, "alive_at_1p5x": live_hi,
                     "deg_at_half": round(alo, 1), "deg_at_1p5x": round(ahi, 1),
                     "span_deg": round(d_ang, 1)})
    # live rows first, sorted by movement; the rows whose sweep kills the rhythm after
    # them, because their span is not a movement of anything.
    rows.sort(key=lambda r: (not (r["alive_at_half"] and r["alive_at_1p5x"]),
                             -r["span_hz"]))
    by = {r["param"]: r for r in rows}
    clock = {k: by[k]["span_hz"] for k in ("tau_h_up", "g_nap")}
    inert = [r["param"] for r in rows if r["verdict"] == "inert"]
    breaks = [r["param"] for r in rows if r["verdict"].startswith("breaks")]
    # THE DECLARED RULE, UNCHANGED.  `g_nap` fails it, and the reason is recorded in
    # the docstring rather than repaired by moving the bar.
    ok = all(v > CLOCK_MIN_HZ for v in clock.values()) and \
        by["w_recip"]["span_hz"] + by["w_recip"]["span_deg"] > 0.0
    return {"ok": bool(ok), "baseline_hz": round(f0, 3),
            "baseline_prominence": round(p0, 2), "baseline_deg": round(a0, 1),
            "claimed_clock_span_hz": clock,
            "claimed_clock_verdict": {k: by[k]["verdict"] for k in ("tau_h_up", "g_nap")},
            "rule": f"tau_h_up and g_nap must each move the locomotor frequency by "
                    f"> {CLOCK_MIN_HZ} Hz over a 3x sweep, and w_recip must move "
                    f"something",
            "inert_definition": f"BOTH endpoints alive (prominence > 0) and "
                                f"< {INERT_HZ} Hz AND < {INERT_DEG} deg over the 3x sweep",
            "inert": inert, "breaks_the_rhythm": breaks, "n_constants": len(fields),
            "dt_used": DT_SWEEP,
            "delays_in_steps_at_sweep_dt": SpinalCord(learn=False).delays_in_steps(DT_SWEEP),
            "sweep": rows}


# --------------------------------------------------------------------------------------
# the one fitted constant
# --------------------------------------------------------------------------------------
def sweep_tau_h(values):
    """`tau_h_up` against the locomotor frequency, at every drive level.

    The declared rule, written before the sweep is read: `tau_h_up` is chosen so the
    GEOMETRIC MEAN of the drive sweep sits on the catalogue's declared 1.2 Hz peak.
    The geometric mean rather than one drive level, because the catalogue declares a
    BAND with a peak in it and the drive is what moves the model within the band -- a
    point target at one arbitrary drive would be inventing an operating point nobody
    measured.  The whole curve is recorded either way.
    """
    rows = []
    for v in values:
        pr = CordPriors(tau_h_up=v, tau_h_dn=0.25 * v)
        c = SpinalCord(priors=pr, learn=False)
        fs, ok = [], True
        for d in DRIVE_LEVELS:
            m = measure(c, d, dt=DT_SWEEP, secs=SECS_SWEEP, burn=BURN_SWEEP)
            fs.append(m["peak_hz"])
            ok = ok and m["prominence_decades"] > 0.0
        gm = math.exp(sum(math.log(max(f, 1e-9)) for f in fs) / len(fs))
        rows.append({"tau_h_up_ms": round(v * 1000, 1), "geometric_mean_hz": round(gm, 3),
                     "min_hz": round(min(fs), 3), "max_hz": round(max(fs), 3),
                     "alive_at_every_level": ok,
                     "hz": [round(f, 3) for f in fs]})
        print(f"  tau_h_up {v * 1000:6.1f} ms -> geo-mean {gm:6.3f} Hz  "
              f"[{min(fs):.2f}, {max(fs):.2f}]  alive {ok}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true")
    # SEPARATE default paths on purpose.  One default for both modes means `--sweep`
    # silently overwrites the gate record with a file that has no gates in it, and the
    # only tell would be a timestamp (CLAUDE.md: a stage that rewrites its
    # predecessor's output file can silently undo it).
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = "out/gate_cord_sweep.json" if a.sweep else "out/gate_cord.json"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cord = SpinalCord(learn=False)
    rec = {"script": "scripts/gate_cord.py", "module": "ibm/cord.py",
           "dt": DT, "dt_fine": DT_FINE, "dt_sweep": DT_SWEEP,
           "welch_segment_s": SEG_S, "trace_s": SECS, "burn_s": BURN,
           "noise_seed": NOISE_SEED, "drive_levels": list(DRIVE_LEVELS),
           "pools": list(POOL_NAMES),
           "catalogue": {"locomotor_band_hz": list(BAND), "declared_peak_hz": LOCO.peak,
                         "antiphase_window_deg": [PHASE_LO, PHASE_HI],
                         "reflex_edge_ms": list(REFLEX_MS)},
           "delays_in_steps": cord.delays_in_steps(DT),
           "priors": {k: getattr(cord.pr, k) for k in CordPriors.__dataclass_fields__},
           "gates": {}}

    def save():
        # written after EVERY gate, and the cheap summary is written first, so a gate
        # that raises cannot destroy the record of the ones that already ran.
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=str)

    save()
    if a.sweep:
        print("tau_h_up sweep (whole drive sweep at each value):", flush=True)
        rec["sweep_tau_h"] = sweep_tau_h([0.18, 0.25, 0.31, 0.37, 0.45, 0.55, 0.70])
        rec["declared_peak_hz"] = LOCO.peak
        save()
        print(f"\nwrote {a.out}")
        return 0

    order = [("S0_bounded", lambda: gate_bounded(cord)),
             ("S1_idempotent", lambda: gate_idempotent(cord)),
             ("S2_alternation", lambda: gate_alternation(cord)),
             ("S3_frequency_follows_drive", lambda: gate_frequency_follows_drive(cord)),
             ("S4_generates", lambda: gate_generates(cord)),
             ("S5_reflex_latency", lambda: gate_reflex_latency(cord)),
             ("S6_deafferentation", lambda: gate_deafferentation(cord)),
             ("S7_sensitivity", gate_sensitivity)]
    ok_all = True
    failed = []
    for name, fn in order:
        r = fn()
        rec["gates"][name] = r
        save()
        ok_all &= bool(r.get("ok"))
        if not r.get("ok"):
            failed.append(name)
        head = {k: v for k, v in r.items()
                if k not in ("sweep", "cases", "arms", "intact", "cut")}
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {name}: {head}", flush=True)
        if name == "S3_frequency_follows_drive":
            for row in r["sweep"]:
                print(f"      drive {row['drive']:.2f}  {row['peak_hz']:6.3f} Hz  "
                      f"prom {row['prominence_decades']:+.2f}  "
                      f"flex/ext {row['flex_ext_deg']:6.1f} deg (plv "
                      f"{row['flex_ext_plv']:.3f})  L/R {row['left_right_deg']:6.1f} deg",
                      flush=True)
        if name == "S7_sensitivity":
            for row in r["sweep"]:
                print(f"      {row['param']:16s} {row['hz_at_half']:6.3f} -> "
                      f"{row['hz_at_1p5x']:6.3f} Hz  span {row['span_hz']:5.3f}  "
                      f"phase {row['span_deg']:5.1f} deg  prom "
                      f"{row['prom_at_half']:+5.2f}/{row['prom_at_1p5x']:+5.2f}  "
                      f"{row['verdict']}", flush=True)
            print(f"      inert ({r['inert_definition']}): "
                  f"{', '.join(r['inert']) or 'none'}", flush=True)
            print(f"      breaks the rhythm: "
                  f"{', '.join(r['breaks_the_rhythm']) or 'none'}", flush=True)
    rec["all_gates_ok"] = ok_all
    rec["failed_gates"] = failed
    save()
    print(f"\n{'ALL GATES PASS' if ok_all else 'FAILED: ' + ', '.join(failed)}"
          f" -- wrote {a.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
