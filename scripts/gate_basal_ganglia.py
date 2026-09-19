"""known answers for the basal-ganglia module, and the sweep that sets its one fitted constant.

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/gate_basal_ganglia.py
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/gate_basal_ganglia.py --sweep

CPU only.  On this machine the GPU shares the system memory pool, so a GPU OOM is a
machine-wide OOM (CLAUDE.md, "Jobs"); nothing here needs one.

EVERY VERDICT BELOW IS FIXED IN THIS FILE BEFORE IT RUNS.  A gate that fails is recorded
as FAILED, never rescored and never re-run with a different bar (CLAUDE.md, "Write it down
where it will be found").  The gates are about the MECHANISM: two of them are about the
loop's actual job, which is selecting one action out of several and cancelling it again,
and only then about the band it hums at.  `ibm/rhythms.py`'s `action_selection` row says
why in its own note -- "a model that reproduced every band in this file while selecting
nothing would have missed the point."

  B0  BOUNDED.  Every rate and every synaptic activation stays in [0, 1] at three
      timesteps (1, 5, 20 ms) under extreme drive, including a saturating stop pulse.
      The exponential-Euler step makes this a property rather than a hope, so the gate
      exists to check the claim is true of the code and not only of the algebra.
      The OU background currents are deliberately NOT tested: they are currents, not
      rates, and are not in the box.  A gate that quietly tested something other than
      what it claims is worse than no gate at all.
  B1  IDEMPOTENCE.  The same rollout with the same generator, twice, bit-identical on
      every recorded trace.  Not a known-answer check: it tests whether the model is a
      FUNCTION of its arguments at all (CLAUDE.md, "call it twice at the same input").
  B2  SELECTION, the loop's actual job.  Eight channels get eight different cortical
      drives.  Declared before the run:
          * EXACTLY ONE channel's GPi falls below SELECT_MAX = 0.50 of its own resting
            rate -- selection is a PAUSE in a tonically firing output nucleus, so the
            bar is a fraction of rest and not an absolute rate;
          * every other channel stays at or above LOSER_FLOOR = 0.98 of its resting rate;
          * the channel that wins is the one with the largest cortical drive.
      Reported: the winner's GPi drop, the runner-up's, and the margin between them.
      A band is not a substitute for this and no spectral gate below can stand in for it.
  B3  STOP.  With a selection already granted, a hyperdirect pulse must re-raise GPi to
      at least LOSER_FLOOR of resting on ALL EIGHT channels.  Declared bar: it happens
      within STOP_MAX_MS = 200 ms of the pulse, the upper end of `stop_signal`'s declared
      120-200 ms.  ONE-SIDED, and the reason is declared here before the measurement: the
      catalogue's 120-200 ms is a behavioural stop-signal reaction time, which includes
      the cortical detection of the stop cue that this module does not contain, so a
      latency BELOW 120 ms is reported and is not a failure.  Reported: the measured
      latency and the GPi trajectory on the winner.
  B4  BETA.  The STN-GPe pair must produce a 13-30 Hz rhythm at rest: `peak_prominence`
      over the band, from `ibm/spectral.py`, strictly greater than 0, with the peak
      frequency reported BESIDE it and never alone.  `peak_frequency` is a soft-argmax and
      returns the band centre on a flat spectrum, so a frequency without its prominence is
      not evidence (CLAUDE.md).
  B5  BURSTS.  That beta must ARRIVE IN BURSTS, not as a tone.  `beta_bursts` declares
      100-500 ms.  Three numbers, all declared here:
          * median burst duration in [100, 500] ms;
          * burst rate at least 0.5/s, so there is something to have a duration;
          * envelope coefficient of variation at least CV_MIN = 0.30.
      The third is not decoration and it is the gate that does the work.  Bursts are
      detected by the literature's rule -- the 13-30 Hz envelope above its own 75th
      percentile (Tinkhauser 2017) -- and a PERCENTILE threshold always produces a 25%
      duty cycle whatever the signal is, so a constant tone scores a burst rate and a
      burst duration too.  Measured on this model: a supercritical loop gives an envelope
      CV of 0.05-0.14 and still reports "47 ms bursts".  Without the CV floor this gate
      could pass for a reason unrelated to what it tests.
      DECLARED HERE BEFORE THE RUN, because it is the kind of thing that is easy to
      forget afterwards: the module's loop gain was CHOSEN on the CV, from `--sweep`, as
      the sharpest resonance still clearly noise-driven.  So the CV clause is satisfied by
      construction and is NOT independent evidence -- it is a guard that stops the
      duration clause being satisfied by a tone, nothing more.  The duration clause is the
      one that tests the model, and it is the one that decides this gate.
  B6  DOPAMINE, and the direction is declared BEFORE the run.  Lowering dopamine from 0.5
      to 0.2 must move BOTH of:
          * beta prominence UP   (less dopamine -> a tonically over-driven indirect arm
            -> a released STN -> the pair closer to its bifurcation);
          * the winner's GPi drop DOWN, i.e. selection weaker.
      Both numbers are reported whichever way they go, together with the selection
      latency, which is expected to lengthen.  `dopamine` is a parameter of the model and
      there is no Parkinsonian branch anywhere in `ibm/basal_ganglia.py` -- the same
      equations run at both levels.
  B7  SENSITIVITY.  Every declared constant swept +-50% and BOTH outputs re-measured: the
      beta peak frequency and the selection margin.  Sorted by how far each moves them,
      and every constant that is inert over the whole 3x sweep is listed.  Declared bars,
      from what the module CLAIMS:
          * `tau_gaba_pal` and `tau_ampa_stn` are stated to set the beta frequency and
            must each move it by more than 1 Hz;
          * `w_d1_gpi` is stated to be the inhibition whose withdrawal IS the selection
            and must move the selection margin by more than 0.05.
      This gate is the cheapest available detector of a mechanism that is not wired in.
      Building `ibm/thalamus.py` produced five flat sweeps in one day and every one was a
      feedback loop that had not been closed (CLAUDE.md, "A parameter that changes
      nothing").  This module has already produced one: the dopamine gain term multiplied
      a cortical drive that is zero at rest, so the resting spectrum was bit-identical at
      dopamine 0.5 and 0.2 until an excitability term was added beside it.

`--sweep` runs the one sweep in this module whose answer a MEASUREMENT fixes rather than
the literature: the STN-GPe loop gain against the burst duration.  `beta_bursts` declares
100-500 ms; the loop gain is what sets how long the pair rings after it is knocked, so the
sweep is how that constant is chosen -- transparently, with the whole curve recorded,
rather than by nudging it until a gate passes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import spectral as SP                                          # noqa: E402
from ibm.basal_ganglia import BasalGanglia, BasalGangliaPriors, R_MAX   # noqa: E402
from ibm.rhythms import RHYTHM                                          # noqa: E402

DT = 0.001
N_CH = 8
BETA = RHYTHM["beta_bg"]                 # band (13, 30), declared peak 20 Hz
BURSTS = RHYTHM["beta_bursts"]           # 100-500 ms, and "not a tone"
STOP = RHYTHM["stop_signal"]             # latency_ms (120, 200)

# ---- the bars, declared here, before anything runs -----------------------------------
SELECT_MAX = 0.50          # winner's GPi, as a fraction of its own resting rate
LOSER_FLOOR = 0.98         # every other channel's GPi, same units
STOP_MAX_MS = 200.0        # the upper end of `stop_signal`'s declared 120-200 ms
STOP_PULSE = 1.0           # amplitude of the cortical stop drive
CV_MIN = 0.30              # envelope CV floor that separates bursting from a tone
BURST_Q = 0.75             # Tinkhauser 2017's threshold: the envelope's 75th percentile
CLOCK_HZ = 1.0             # a claimed beta clock must move the peak by more than this
SELECT_SENS = 0.05         # a claimed selection weight must move the margin by more
INERT_HZ, INERT_MARGIN = 0.20, 0.02      # below BOTH of these a constant is inert

# eight different cortical proposals.  Descending, well separated at the top and closely
# spaced below, so "picked the biggest" is not the same as "picked the only one".
DRIVES = torch.tensor([[0.90, 0.70, 0.62, 0.55, 0.50, 0.45, 0.40, 0.35]])


def model(priors=None, n=N_CH):
    return BasalGanglia(n, priors=priors, learn=False)


def beta_trace(bg, seconds=60.0, dopamine=0.5, seed=5):
    """the resting STN trace, channel-averaged.

    Channel-averaged because an LFP is a population signal and because the STN's efferent
    is diffuse anyway; `beta_bg`'s stations are ("stn", "gpe") and both are returned.
    """
    g = torch.Generator().manual_seed(seed)
    steps = int(round(seconds / DT))
    tr, _, _ = bg.rollout(steps, DT, ctx=0.0, dopamine=dopamine, noise_gen=g, b=1,
                          record=("STN", "GPe"), burn=int(round(1.0 / DT)))
    return tr["STN"][0].mean(-1), tr["GPe"][0].mean(-1)


def beta_peak(x, nperseg=8192):
    """peak frequency AND prominence, always together (CLAUDE.md)."""
    freqs, psd = SP.welch_psd(x, 1.0 / DT, nperseg=min(nperseg, x.shape[-1]))
    lo, hi = BETA.band
    return (float(SP.peak_frequency(psd, freqs, lo, hi)),
            float(SP.peak_prominence(psd, freqs, lo, hi)))


def burst_stats(x):
    """rate, median duration and envelope CV of the 13-30 Hz bursts.

    The threshold is the envelope's own 75th percentile, which is the literature's rule
    and is also why the CV is returned beside the duration: a percentile threshold cuts a
    constant tone into "bursts" as happily as it cuts real ones.
    """
    lo, hi = BETA.band
    env = SP.bandpass_analytic(x, 1.0 / DT, lo, hi).abs()
    env = env[int(0.5 / DT):-int(0.5 / DT)]          # the FFT filter wraps at the ends
    th = float(env.quantile(BURST_Q))
    above = (env > th).to(torch.int8)
    d = above[1:] - above[:-1]
    on = (d == 1).nonzero().flatten()
    off = (d == -1).nonzero().flatten()
    if on.numel() and off.numel() and off[0] < on[0]:
        off = off[1:]
    k = int(min(on.numel(), off.numel()))
    if k == 0:
        return {"burst_rate_hz": 0.0, "median_duration_ms": float("nan"),
                "max_duration_ms": float("nan"),
                "envelope_cv": float(env.std() / env.mean().clamp_min(1e-12)),
                "n_bursts": 0}
    dur = (off[:k] - on[:k]).float() * DT * 1000.0
    return {"burst_rate_hz": k / (env.numel() * DT),
            "median_duration_ms": float(dur.median()),
            "max_duration_ms": float(dur.max()),
            "envelope_cv": float(env.std() / env.mean().clamp_min(1e-12)),
            "n_bursts": k}


def selection_run(bg, seconds=1.5, dopamine=0.5, seed=17, stop=None, noise=True):
    """quiet, then eight proposals at once.  Returns the GPi/Thal traces and the rest."""
    steps = int(round(seconds / DT))
    on = int(round(0.30 / DT))
    ctx = DRIVES.unsqueeze(1).repeat(1, steps, 1).clone()
    ctx[:, :on] = 0.0
    g = torch.Generator().manual_seed(seed) if noise else None
    tr, _, info = bg.rollout(steps, DT, ctx=ctx, stop=stop, dopamine=dopamine,
                             noise_gen=g, b=1, record=("GPi", "Thal", "STN", "GPe"))
    # resting GPi is MEASURED on this run, per channel, over the 100 ms before the
    # proposals arrive -- not read off the declared constant, so the comparison is against
    # the rate this run actually had.
    rest = tr["GPi"][0, on - int(0.10 / DT):on].mean(0)
    return tr, rest, on, info


def _drop(tr, rest, on, window=0.20):
    """1 - GPi/rest per channel, averaged over the last `window` of the run."""
    final = tr["GPi"][0, -int(window / DT):].mean(0)
    return 1.0 - final / rest, final


# --------------------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------------------
def gate_bounded(bg):
    """B0.  Rates and synaptic activations in [0, 1] at three dt under extreme drive."""
    keys = ("D1", "D2", "GPe", "GPi", "STN", "Thal",
            "sD1", "sD2", "sSG", "sSI", "sGS", "sGG", "sGT", "aSTN")
    worst, cases = 0.0, []
    for dt in (0.001, 0.005, 0.020):
        for drive in (-3.0, 0.0, 3.0):
            for stop in (0.0, 3.0):
                g = torch.Generator().manual_seed(11)
                st = bg.init_state(1)
                pars = bg.params()
                for _ in range(int(4.0 / dt)):
                    z = torch.randn(1, 6, bg.n, generator=g)
                    st = bg.step(st, dt, ctx=drive, stop=stop, noise=z, params=pars)
                bad = 0.0
                for k in keys:
                    v = st[k]
                    bad = max(bad, float(max((-v).clamp_min(0).max(), (v - 1).clamp_min(0).max())))
                worst = max(worst, bad)
                cases.append({"dt": dt, "ctx": drive, "stop": stop, "excursion": bad,
                              "GPi_hz": float(st["GPi"].mean()) * R_MAX,
                              "STN_hz": float(st["STN"].mean()) * R_MAX})
    return {"ok": worst <= 0.0, "worst_excursion": worst,
            "variables_tested": list(keys),
            "not_tested": "eta -- the OU background is a CURRENT, not a rate, and is not "
                          "claimed to be in [0, 1]",
            "cases": cases}


def gate_idempotent(bg):
    """B1.  Same generator in, bit-identical out, on every trace."""
    a, _, _ = selection_run(bg, seconds=1.0, seed=7)
    b, _, _ = selection_run(bg, seconds=1.0, seed=7)
    diffs = {k: float((a[k] - b[k]).abs().max()) for k in a}
    return {"ok": all(torch.equal(a[k], b[k]) for k in a),
            "max_abs_diff": diffs, "traces": list(a)}


def gate_selection(bg):
    """B2.  Exactly one GPi pause, everyone else at or above rest, and the right winner."""
    tr, rest, on, info = selection_run(bg)
    drop, final = _drop(tr, rest, on)
    ratio = final / rest
    order = torch.argsort(ratio)
    win, run2 = int(order[0]), int(order[1])
    n_below = int((ratio < SELECT_MAX).sum())
    losers_ok = bool((ratio[[i for i in range(bg.n) if i != win]] >= LOSER_FLOOR).all())
    ok = (n_below == 1) and losers_ok and win == int(torch.argmax(DRIVES[0]))
    return {"ok": bool(ok),
            "n_channels_below_threshold": n_below, "threshold_frac_of_rest": SELECT_MAX,
            "winner_channel": win, "winner_has_largest_drive": win == int(torch.argmax(DRIVES[0])),
            "winner_gpi_drop": float(drop[win]), "runner_up_channel": run2,
            "runner_up_gpi_drop": float(drop[run2]),
            "margin_in_drop": float(drop[win] - drop[run2]),
            "losers_all_at_or_above_rest": losers_ok, "loser_floor": LOSER_FLOOR,
            "gpi_hz_rest": [round(float(v) * R_MAX, 2) for v in rest],
            "gpi_hz_final": [round(float(v) * R_MAX, 2) for v in final],
            "thal_hz_final": [round(float(v) * R_MAX, 2)
                              for v in tr["Thal"][0, -200:].mean(0)],
            "drives": DRIVES[0].tolist(), "delays_in_steps": info["delays_in_steps"]}


def gate_stop(bg):
    """B3.  A hyperdirect pulse re-raises GPi on ALL channels, and how long it takes."""
    seconds, t_stop = 2.5, 1.5
    steps, k0 = int(round(seconds / DT)), int(round(t_stop / DT))
    stop = torch.zeros(1, steps, N_CH)
    stop[:, k0:k0 + int(round(0.30 / DT))] = STOP_PULSE
    tr, rest, on, _ = selection_run(bg, seconds=seconds, stop=stop)
    gpi = tr["GPi"][0]
    # was anything actually selected when the pulse arrived?  A stop gate that ran on an
    # unselected loop would certify a cancellation that never happened.
    pre = gpi[k0 - 200:k0].mean(0) / rest
    selected = int((pre < SELECT_MAX).sum())
    all_up = (gpi >= LOSER_FLOOR * rest).all(-1)
    idx = [i for i in range(k0, steps) if bool(all_up[i])]
    lat = (idx[0] - k0) * DT * 1000.0 if idx else None
    return {"ok": bool(selected == 1 and lat is not None and lat <= STOP_MAX_MS),
            "channels_selected_before_pulse": selected,
            "latency_ms": lat, "declared_bar_ms": STOP_MAX_MS,
            "catalogue_latency_ms": list(STOP.metric["latency_ms"]),
            "one_sided": "the catalogue's 120-200 ms is a behavioural SSRT including "
                         "cortical detection this module does not contain; a shorter "
                         "neural latency is reported, not failed",
            "gpi_hz_before_pulse": [round(float(v) * R_MAX, 2) for v in gpi[k0 - 1]],
            "gpi_hz_after_100ms": [round(float(v) * R_MAX, 2) for v in gpi[k0 + 100]],
            "thal_hz_before_pulse": [round(float(v) * R_MAX, 2) for v in tr["Thal"][0, k0 - 1]],
            "thal_hz_after_100ms": [round(float(v) * R_MAX, 2) for v in tr["Thal"][0, k0 + 100]]}


def gate_beta(bg):
    """B4.  13-30 Hz in the STN-GPe pair, gated on prominence, frequency beside it."""
    stn, gpe = beta_trace(bg)
    f_s, p_s = beta_peak(stn)
    f_g, p_g = beta_peak(gpe)
    lo, hi = BETA.band
    return {"ok": bool(p_s > 0.0 and lo <= f_s <= hi),
            "stn_peak_hz": f_s, "stn_prominence_decades": p_s,
            "gpe_peak_hz": f_g, "gpe_prominence_decades": p_g,
            "band": [lo, hi], "catalogue_peak_hz": BETA.peak,
            "stn_mean_hz": float(stn.mean()) * R_MAX,
            "stn_sd_hz": float(stn.std()) * R_MAX,
            "note": "peak_frequency is a soft-argmax and returns the band centre on a "
                    "flat spectrum; the prominence is the gate"}


def gate_bursts(bg):
    """B5.  Bursts of 100-500 ms, and an envelope that is not a tone's."""
    stn, _ = beta_trace(bg)
    st = burst_stats(stn)
    lo_ms, hi_ms = BURSTS.metric["target_duration_ms"]
    dur_ok = lo_ms <= st["median_duration_ms"] <= hi_ms
    ok = bool(dur_ok and st["burst_rate_hz"] >= 0.5 and st["envelope_cv"] >= CV_MIN)
    return {"ok": ok, **st, "target_duration_ms": [lo_ms, hi_ms],
            "cv_floor": CV_MIN, "duration_in_range": bool(dur_ok),
            "rule": "median duration in the catalogue's 100-500 ms, rate >= 0.5/s, and "
                    "envelope CV >= 0.30 -- the CV is what a constant tone fails, and a "
                    "75th-percentile threshold on its own cannot tell the two apart"}


def gate_dopamine(bg):
    """B6.  Low dopamine: beta UP, selection DOWN.  Declared above, before the run."""
    out = {}
    for name, da in (("normal_0.50", 0.50), ("low_0.20", 0.20)):
        stn, _ = beta_trace(bg, seconds=40.0, dopamine=da)
        f, p = beta_peak(stn)
        tr, rest, on, _ = selection_run(bg, dopamine=da)
        drop, final = _drop(tr, rest, on)
        ratio = final / rest
        # PAIRED on the same channel in both conditions -- the one with the largest
        # cortical drive, which is the channel that ought to win.  Taking argmin(ratio)
        # separately at each dopamine level would compare two different channels and
        # would report a "winner" even when nothing was selected at all.
        win = int(torch.argmax(DRIVES[0]))
        below = (tr["GPi"][0, :, win] < SELECT_MAX * rest[win]).nonzero().flatten()
        out[name] = {"beta_peak_hz": f, "beta_prominence_decades": p,
                     "channel": win, "winner_gpi_drop": float(drop[win]),
                     "best_drop_any_channel": float(drop.max()),
                     "select_latency_ms": (float(int(below[0]) - on) if below.numel() else None),
                     "n_channels_selected": int((ratio < SELECT_MAX).sum()),
                     "stn_mean_hz": float(stn.mean()) * R_MAX}
    beta_up = out["low_0.20"]["beta_prominence_decades"] > out["normal_0.50"]["beta_prominence_decades"]
    weaker = out["low_0.20"]["winner_gpi_drop"] < out["normal_0.50"]["winner_gpi_drop"]
    return {"ok": bool(beta_up and weaker),
            "beta_rose": bool(beta_up), "selection_weakened": bool(weaker),
            "d_prominence": (out["low_0.20"]["beta_prominence_decades"]
                             - out["normal_0.50"]["beta_prominence_decades"]),
            "d_winner_drop": (out["low_0.20"]["winner_gpi_drop"]
                              - out["normal_0.50"]["winner_gpi_drop"]),
            "declared_direction": "lowering dopamine raises beta prominence and shrinks "
                                  "the winner's GPi drop (the Parkinsonian direction)",
            "states": out}


def gate_sensitivity():
    """B7.  Every declared constant +-50%, against the beta peak AND the selection margin."""
    base = BasalGangliaPriors()
    fields = list(base.__dataclass_fields__)

    def measure(pr):
        bg = model(pr)
        stn, _ = beta_trace(bg, seconds=12.0)
        f, p = beta_peak(stn, nperseg=4096)
        tr, rest, on, _ = selection_run(bg, seconds=1.2)
        drop, final = _drop(tr, rest, on, window=0.15)
        o = torch.argsort(final / rest)
        return f, p, float(drop[int(o[0])] - drop[int(o[1])])

    f0, p0, m0 = measure(base)
    rows = []
    for name in fields:
        v = float(getattr(base, name))
        got = {}
        for tag, scale in (("half", 0.5), ("1p5x", 1.5)):
            kw = {k: getattr(base, k) for k in fields}
            kw[name] = v * scale
            got[tag] = measure(BasalGangliaPriors(**kw))
        rows.append({"param": name, "value": v,
                     "hz_at_half": got["half"][0], "hz_at_1p5x": got["1p5x"][0],
                     "span_hz": abs(got["1p5x"][0] - got["half"][0]),
                     "prom_at_half": got["half"][1], "prom_at_1p5x": got["1p5x"][1],
                     "margin_at_half": got["half"][2], "margin_at_1p5x": got["1p5x"][2],
                     "span_margin": abs(got["1p5x"][2] - got["half"][2])})
        print(f"      {name:14s} beta {got['half'][0]:6.2f} -> {got['1p5x'][0]:6.2f} Hz "
              f"(span {abs(got['1p5x'][0] - got['half'][0]):5.2f})   margin "
              f"{got['half'][2]:+.3f} -> {got['1p5x'][2]:+.3f} "
              f"(span {abs(got['1p5x'][2] - got['half'][2]):5.3f})", flush=True)
    by_hz = sorted(rows, key=lambda r: -r["span_hz"])
    clock = {r["param"]: r["span_hz"] for r in rows
             if r["param"] in ("tau_gaba_pal", "tau_ampa_stn")}
    sel = {r["param"]: r["span_margin"] for r in rows if r["param"] == "w_d1_gpi"}
    inert = [r["param"] for r in rows
             if r["span_hz"] < INERT_HZ and r["span_margin"] < INERT_MARGIN]
    ok = all(v > CLOCK_HZ for v in clock.values()) and all(v > SELECT_SENS for v in sel.values())
    return {"ok": bool(ok), "baseline_beta_hz": f0, "baseline_prominence": p0,
            "baseline_margin": m0,
            "claimed_clock_span_hz": clock, "claimed_selection_span_margin": sel,
            "rule": "tau_gaba_pal and tau_ampa_stn must each move the beta peak by > "
                    f"{CLOCK_HZ} Hz; w_d1_gpi must move the selection margin by > {SELECT_SENS}",
            "inert_both": inert,
            "inert_definition": f"span < {INERT_HZ} Hz in beta AND < {INERT_MARGIN} in margin",
            "sweep": by_hz}


def sweep_loop_gain(values):
    """the constant a measurement sets: the STN-GPe loop gain against burst duration.

    `beta_bursts` declares 100-500 ms.  The loop gain is what decides how long the pair
    rings after the background knocks it, so this curve is how that constant is chosen --
    the whole of it recorded, including the part where the pair crosses its bifurcation
    and the envelope collapses into a tone.
    """
    rows = []
    for v in values:
        pr = BasalGangliaPriors(w_stn_gpe=v, w_gpe_stn=v)
        bg = model(pr)
        stn, _ = beta_trace(bg, seconds=60.0)
        f, p = beta_peak(stn)
        st = burst_stats(stn)
        rows.append({"loop_weight": v, "peak_hz": f, "prominence_decades": p, **st})
        print(f"  w {v:5.2f} -> {f:6.2f} Hz  prominence {p:+.3f}  "
              f"{st['burst_rate_hz']:5.2f} bursts/s  median {st['median_duration_ms']:6.1f} ms"
              f"  envelope CV {st['envelope_cv']:.3f}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="out/gate_basal_ganglia.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.set_num_threads(2)
    rec = {"script": "scripts/gate_basal_ganglia.py", "dt": DT, "n_channels": N_CH,
           "bars": {"select_max_frac_of_rest": SELECT_MAX, "loser_floor": LOSER_FLOOR,
                    "stop_max_ms": STOP_MAX_MS, "cv_min": CV_MIN, "burst_quantile": BURST_Q,
                    "clock_hz": CLOCK_HZ, "select_sens": SELECT_SENS},
           "gates": {}}

    def save():
        # the cheap summary is written BEFORE anything that can raise, and after every
        # gate, so a crash in gate k leaves gates 0..k-1 on disk (CLAUDE.md, "Jobs")
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=str)

    save()
    if a.sweep:
        print("STN-GPe loop gain sweep (resting, 60 s, channel-averaged STN):", flush=True)
        rec["sweep_loop_gain"] = sweep_loop_gain(
            [1.20, 1.35, 1.45, 1.50, 1.55, 1.60, 1.80, 2.00])
        save()
        print(f"\nwrote {a.out}")
        return 0

    bg = model()
    rec["declared_rest_hz"] = bg.rest_rates()
    rec["delays_in_steps"] = bg.delays_in_steps(DT)
    save()
    order = [("B0_bounded", lambda: gate_bounded(bg)),
             ("B1_idempotent", lambda: gate_idempotent(bg)),
             ("B2_selection", lambda: gate_selection(bg)),
             ("B3_stop", lambda: gate_stop(bg)),
             ("B4_beta", lambda: gate_beta(bg)),
             ("B5_bursts", lambda: gate_bursts(bg)),
             ("B6_dopamine", lambda: gate_dopamine(bg)),
             ("B7_sensitivity", gate_sensitivity)]
    ok_all = True
    for name, fn in order:
        t0 = time.time()
        if name == "B7_sensitivity":
            print(f"[....] {name}: sweeping every declared constant +-50% ...", flush=True)
        r = fn()
        r["seconds"] = round(time.time() - t0, 1)
        rec["gates"][name] = r
        save()
        ok_all &= bool(r.get("ok"))
        head = {k: v for k, v in r.items()
                if k not in ("sweep", "cases", "states", "max_abs_diff")}
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {name}: {head}", flush=True)
        if name == "B6_dopamine":
            for k, v in r["states"].items():
                print(f"      {k:12s} beta {v['beta_peak_hz']:6.2f} Hz "
                      f"prom {v['beta_prominence_decades']:+.3f}   winner drop "
                      f"{v['winner_gpi_drop']:+.3f}  latency {v['select_latency_ms']} ms  "
                      f"selected {v['n_channels_selected']}", flush=True)
        if name == "B7_sensitivity":
            print(f"      inert over the whole 3x sweep: "
                  f"{', '.join(r['inert_both']) or 'none'}", flush=True)
    rec["all_gates_ok"] = ok_all
    rec["failed"] = [k for k, v in rec["gates"].items() if not v.get("ok")]
    save()
    print(f"\n{'ALL GATES PASS' if ok_all else 'SOME GATES FAILED: ' + ', '.join(rec['failed'])}"
          f" -- wrote {a.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
