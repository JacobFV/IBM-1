"""known answers for `ibm/brain/basal_ganglia.py`, the v3 declaration of the basal ganglia.

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python \
        scripts/gate_brain_basal_ganglia.py
    ... scripts/gate_brain_basal_ganglia.py --gates B7          # the sweep on its own
    ... scripts/gate_brain_basal_ganglia.py --sweep             # the loop-gain curve

CPU only.  On this machine the GPU shares the system memory pool, so a GPU OOM is a
machine-wide OOM (CLAUDE.md, "Jobs"); nothing here needs one.

EVERY VERDICT BELOW IS FIXED IN THIS FILE BEFORE IT RUNS, including the DIRECTION of B5.
A gate that fails is recorded as FAILED, never rescored and never re-run with a different
bar (CLAUDE.md, "Write it down where it will be found").

THE HARNESS, AND WHAT IS A STAND-IN
-----------------------------------
`ibm/brain/basal_ganglia.py` names seven populations it does not own: four cortical
sources, one cortical stopping area, `nm.snc` / `nm.vta`, and the two thalamic relays it
returns through.  All seven exist in `ibm/brain/`, and a full assembly keeps every edge
this module declares with ZERO orphans -- but that assembly is 134 populations and 1,042
projections, and running it here would make every number below partly a measurement of the
cortex, the thalamus and the neuromodulatory nuclei at once.  So this gate builds the
circuit itself, out of the module's own `pops()`, `internal()`, `external()` and `mods()`
plus SEVEN STUB POPULATIONS of one unit each whose only job is to sit at a commanded rate:

    ctx.precentral.E  ctx.caudalmiddlefrontal.E  ctx.rostralmiddlefrontal.E
    ctx.medialorbitofrontal.E  ctx.parsopercularis.E  nm.snc  nm.vta

Two things follow and both matter for reading the numbers.  (1) A stub has no declared
`sparsity`, so `ibm/circuit.py` normalises its projection by 1.0 and an edge of weight `w`
from a stub at rate `x` delivers `w * x`.  The REAL cortex is declared sparse, and the same
edge from it delivers `w * mean(r) / sparsity` -- so a "proposal of 0.90" here corresponds
to a cortical population whose mean rate is `0.90 * sparsity`, not to one at 90 Hz.  (2)
Nothing about the cortex's own dynamics, its rhythms or its return leg is tested here.
This gate is about the basal ganglia's mechanism and says so rather than quietly testing a
network it did not declare.

THE GATES
---------
  B0  BOUNDED.  Every population's rate stays in [0, 1] at three timesteps (1, 5, 20 ms)
      under extreme drive -- +-5.0 of injected current on every basal-ganglia population at
      once, with the cortical proposals saturated and with and without a saturating stop
      pulse -- checked at EVERY STEP of every case, not at the endpoint, because "for any
      dt and any input" is a promise about the trajectory and a variable that leaves the box
      and returns has still been represented outside it.  The OU background currents are
      deliberately NOT tested: they are currents, not rates, and are not in the box.  A gate
      that quietly tested something other than what it claims is worse than no gate at all.
      At dt = 20 ms every declared delay rounds to zero; that is reported, not hidden, and
      it is a different circuit rather than an approximation.
  B1  IDEMPOTENCE.  Three separate claims, because "call it twice at the same input" is
      about whether the thing is a FUNCTION and not about any one value (CLAUDE.md):
          * the DECLARATIONS are pure -- `pops`, `internal`, `external`, `mods`, `drives`
            called twice return identical lists, so nothing in the module draws from a
            generator while declaring;
          * two `Circuit`s built from the same seed have bit-identical weight matrices;
          * the same rollout with the same generator, twice, is bit-identical on every
            recorded trace.
  B2  SELECTION, the loop's actual job.  Four different proposals arrive at once (motor
      0.90, oculomotor 0.70, associative 0.62, limbic 0.55).  Declared before the run:
          * EXACTLY ONE channel's GPi falls below SELECT_MAX = 0.50 of its own MEASURED
            resting rate -- selection is a PAUSE in a tonically firing output nucleus, so
            the bar is a fraction of rest and not an absolute rate;
          * every other channel stays at or above LOSER_FLOOR = 0.98 of its resting rate;
          * the channel that wins is the one with the largest proposal.
      Reported: the winner's drop, the runner-up's, and the margin between them.
  B3  STOP.  With a selection already granted, a pulse into `ctx.parsopercularis.E` -- the
      stopping network, down the hyperdirect tract, onto every channel's STN -- must
      re-raise GPi to at least LOSER_FLOOR of resting on ALL FOUR channels.  Declared bar:
      within STOP_MAX_MS = 200 ms, the upper end of `stop_signal`'s declared 120-200 ms.
      ONE-SIDED, and the reason is declared here before the measurement: the catalogue's
      interval is a behavioural stop-signal reaction time and includes the cortical
      detection of the stop cue that this module does not contain, so a latency BELOW
      120 ms is REPORTED AS A FINDING and is not a failure.  v2 measured 22 ms and said so.
      The gate also checks that something was actually selected when the pulse arrived: a
      stop gate run on an unselected loop would certify a cancellation that never happened.
  B4  BETA.  The STN-GPe pair must produce 13-30 Hz at rest.  Gated on `peak_prominence`
      from `ibm/spectral.py` strictly greater than 0, with the peak frequency reported
      BESIDE it and never alone -- `peak_frequency` is a soft-argmax and returns the band
      centre on a flat spectrum, so a frequency without its prominence is not evidence
      (CLAUDE.md).  Measured on the channel average AND on each single channel, because
      where those differ by 2x at most one of them is about the circuit; and the envelope's
      coefficient of variation is REPORTED, not gated, beside the reminder that a 17 Hz-wide
      band has an envelope correlation time near 59 ms whatever generates the signal.
  B5  DOPAMINE, and the direction is declared BEFORE the run.  Lowering the rate of
      `nm.snc` and `nm.vta` from 0.50 to 0.20 must move BOTH of:
          * beta prominence UP   (a depleted striatum tonically over-drives the indirect
            arm -> less GPe -> a released STN -> the pair nearer its bifurcation);
          * the winner's GPi drop DOWN, i.e. selection weaker.
      Both numbers are reported whichever way they go.  There is no dopamine float and no
      Parkinsonian branch anywhere in the module: the same equations run at both levels and
      the only thing that changes is the rate of two populations this module does not own.
  B6  CHANNELS ARE INDEPENDENT.  Each channel is driven ALONE, in turn, at 0.90.  Declared:
      the driven channel is released and NO undriven channel's GPi falls by more than
      LEAK_MAX = 0.05 of its resting rate.  Reported: the leakage on every undriven channel,
      signed, because the prediction is that they RISE (the only thing crossing channels is
      the subthalamic efferent, and it pushes GPi up everywhere).
  B7  SENSITIVITY.  Every declared constant swept +-50% and BOTH outputs re-measured: the
      beta peak frequency and the selection margin.  Sorted by how far each moves them, with
      every constant inert over the whole 3x sweep listed AND diagnosed.  Declared bars,
      from what the module CLAIMS:
          * `tau_stn` and `tau_gpe` are stated to be the pair's clock and must each move the
            beta peak by more than CLOCK_HZ = 1.0 Hz;
          * `w_d1_gpi` is stated to be the inhibition whose withdrawal IS the selection and
            must move the selection margin by more than SELECT_SENS = 0.05.
      This gate is the cheapest available detector of a mechanism that is not wired in
      (CLAUDE.md, "A parameter that changes nothing"), and a long inert list is informative
      rather than embarrassing PROVIDED it is diagnosed -- v2 found 28 of 57 inert with a
      four-way diagnosis and that diagnosis is worth more than the count.
      THE SWEEP'S OWN INSTRUMENT IS DECLARED HERE, because it is not B4's.  Each arm is ONE
      rollout of 0.6 s quiet then 0.35 s driven, at a batch of 64 independent background
      realisations, with the SAME generator seed in every arm -- so the comparison is PAIRED
      on the noise and only the constant differs.  Its spectrum is 512-point Welch, 1.95 Hz
      bins, which reads the baseline peak about +0.6 Hz high against B4's long run; that
      bias is common to every arm and the gate reports SPANS, not levels.  The pairing was
      checked: `w_gpi_thal`, whose target does not exist, moves the peak by exactly 0.000 Hz.

`--sweep` runs the one curve in this module that a MEASUREMENT sets rather than the
literature: the STN-GPe loop gain against the resonance, recorded whole -- including the
part where the pair crosses its bifurcation and becomes a tone -- rather than nudged until
a gate passed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import spectral as SP                              # noqa: E402
from ibm.brain import basal_ganglia as BG                   # noqa: E402
from ibm.circuit import Circuit, Pop                        # noqa: E402
from ibm.rhythms import RHYTHM                              # noqa: E402

DT = 0.001
R_MAX = BG.R_MAX
BETA = RHYTHM["beta_bg"]            # band (13, 30) Hz, catalogue peak 20
STOP_ROW = RHYTHM["stop_signal"]    # latency_ms (120, 200)

# ---- the bars, declared here, before anything runs -----------------------------------
SELECT_MAX = 0.50          # winner's GPi, as a fraction of its own measured resting rate
LOSER_FLOOR = 0.98         # every other channel's GPi, same units
STOP_MAX_MS = 200.0        # the upper end of `stop_signal`'s declared 120-200 ms
STOP_PULSE = 1.0           # the stopping area's rate during the pulse
LEAK_MAX = 0.05            # largest tolerated GPi drop on an UNDRIVEN channel
CLOCK_HZ = 1.0             # a claimed beta clock must move the peak by more than this
SELECT_SENS = 0.05         # a claimed selection weight must move the margin by more
INERT_HZ, INERT_MARGIN = 0.20, 0.02      # below BOTH of these a constant is inert
DA_TONIC, DA_LOW = 0.50, 0.20            # the two nigral rates B5 compares

#: four proposals, descending and unevenly spaced, so "picked the biggest" is not the same
#: as "responded to the only input".
PROPOSALS = {"motor": 0.90, "oculomotor": 0.70, "associative": 0.62, "limbic": 0.55}

#: the stub populations.  A slope of 10 about a threshold of 0.5 is invertible, so a
#: commanded rate is delivered exactly rather than approached.
STUB_BETA, STUB_THETA = 10.0, 0.5
STUB_IDS = (list(BG.CTX_SOURCE.values()) + [BG.CTX_STOP, "nm.snc", "nm.vta"])


def stub_current(rate: float) -> float:
    """the drive that puts a stub population exactly at `rate`."""
    r = min(max(float(rate), 1e-4), 1.0 - 1e-4)
    return STUB_THETA + math.log(r / (1.0 - r)) / STUB_BETA


def build(c=None, seed: int = 0):
    """the module's own declarations plus the stubs, and nothing else.

    External edges whose other end is not here (the pallidothalamic ones, while
    `ibm/brain/thalamus.py` does not exist) are dropped exactly as `ibm.brain.collect`
    drops them, and they are listed in the run record as orphans rather than forgotten.
    """
    c = c or BG.C
    ps = list(BG.pops(c)) + [Pop(id=i, n=1, kind="mod", tau=0.005,
                                 beta=STUB_BETA, theta=STUB_THETA,
                                 note="GATE STUB, not part of the brain")
                             for i in STUB_IDS]
    have = {p.id for p in ps}
    ext = list(BG.external(c))
    projs = list(BG.internal(c)) + [e for e in ext if e.src in have and e.dst in have]
    mods = [m for m in BG.mods(c) if m.src in have and m.dst in have]
    orphans = [e.key for e in ext if e.src not in have or e.dst not in have]
    return Circuit(ps, projs, mods, seed=seed), orphans


def base_drive(c, proposals=None, stop=0.0, da=DA_TONIC) -> dict:
    d = dict(BG.drives(c))
    proposals = proposals or {}
    for ch, src in BG.CTX_SOURCE.items():
        d[src] = stub_current(proposals.get(ch, 0.0))
    d[BG.CTX_STOP] = stub_current(stop)
    d["nm.snc"] = stub_current(da)
    d["nm.vta"] = stub_current(da)
    return d


def pop_ids(kind):
    return [f"bg.{kind}.{ch}" for ch in BG.CHANNELS]


def chan_mean(tr, kind):
    """(B, T) per channel, stacked: (n_ch, B, T).  Units-within-a-channel are averaged."""
    return torch.stack([tr[f"bg.{kind}.{ch}"].mean(-1) for ch in BG.CHANNELS], 0)


def spectrum(x, nperseg):
    """peak frequency AND prominence, always together (CLAUDE.md), over the beta band.

    `x` is (B, T); the PSD is averaged over the batch, which is an average over independent
    background realisations of the same circuit and not over anything structural.
    """
    lo, hi = BETA.band
    freqs, psd = SP.welch_psd(x, 1.0 / DT, nperseg=min(nperseg, x.shape[-1]))
    psd = psd.mean(0) if psd.dim() > 1 else psd
    return (float(SP.peak_frequency(psd, freqs, lo, hi)),
            float(SP.peak_prominence(psd, freqs, lo, hi)))


def envelope_cv(x):
    """the 13-30 Hz envelope's coefficient of variation.  REPORTED, never gated.

    Beside it, always, the instrument's own number: a 17 Hz-wide band has an envelope
    correlation time near 1/17 Hz = 59 ms whatever generates the signal, so envelope
    statistics from inside this band are partly a measurement of the filter.
    """
    lo, hi = BETA.band
    env = SP.bandpass_analytic(x, 1.0 / DT, lo, hi).abs()
    env = env[..., int(0.15 / DT):-int(0.15 / DT)]      # the FFT filter wraps at the ends
    return float(env.std() / env.mean().clamp_min(1e-12))


def run(circ, c, seconds, proposals=None, stop=None, da=DA_TONIC, b=8, seed=17,
        burn=0.15, record=None, on=None):
    """one rollout.  `on` (seconds) is when the proposals arrive; `stop` is (t0, t1, rate).

    The burn-in exists for the delay rings, which `init_state` fills with zeros: the
    populations themselves start exactly at the fixed point `drives()` solves for, so what
    is being run off is the rings' cold start and nothing else.
    """
    base = base_drive(c, None, 0.0, da)
    with_prop = base_drive(c, proposals or {}, 0.0, da)
    k_on = None if on is None else int(round(on / DT))
    k_stop = None if stop is None else (int(round(stop[0] / DT)), int(round(stop[1] / DT)))

    def drive(i):
        d = dict(base if (k_on is None or i < k_on) else with_prop)
        if k_stop is not None and k_stop[0] <= i < k_stop[1]:
            d[BG.CTX_STOP] = stub_current(stop[2])
        return d

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        tr, _ = circ.rollout(int(round(seconds / DT)), DT,
                             drive=(drive if (k_on is not None or k_stop is not None)
                                    else base),
                             noise_gen=g, b=b, record=record, burn=int(round(burn / DT)))
    return tr


def gpi_ratio(tr, k_on, window=0.15):
    """GPi as a fraction of its OWN measured resting rate, per channel.

    Rest is measured on this run over the 100 ms before the proposals arrive, not read off
    the declared constant, so the comparison is against the rate this run actually had.
    """
    w = int(round(window / DT))
    g = chan_mean(tr, "gpi")                       # (n_ch, B, T)
    rest = g[:, :, k_on - 100:k_on].mean((1, 2))
    final = g[:, :, -w:].mean((1, 2))
    return rest, final, final / rest


# ======================================================================================
# the gates
# ======================================================================================
def gate_bounded(c):
    """B0.  Every rate in [0, 1] at three dt under extreme drive, checked at every step."""
    circ, _ = build(c)
    ids = [p.id for p in BG.pops(c)]
    extra = torch.tensor([-5.0, -5.0, 0.0, 0.0, 5.0, 5.0])
    stops = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    b = 6
    worst, cases = 0.0, []
    for dt in (0.001, 0.005, 0.020):
        d = base_drive(c, {ch: 0.999 for ch in BG.CHANNELS}, 0.0)
        drive = {}
        for k, v in d.items():
            if k in STUB_IDS:
                drive[k] = torch.tensor([[stub_current(stops[i])] if k == BG.CTX_STOP
                                         else [v] for i in range(b)])
            else:
                drive[k] = (torch.full((b, BG.N_UNITS), float(v))
                            + extra[:, None].expand(b, BG.N_UNITS))
        g = torch.Generator().manual_seed(11)
        st = circ.init_state(b, dt)
        bad, n_steps = 0.0, int(round(2.0 / dt))
        with torch.no_grad():
            for _ in range(n_steps):
                z = {p.id: torch.randn(b, p.n, generator=g)
                     for p in circ.pops.values() if p.sigma}
                st = circ.step(st, dt, drive=drive, noise=z)
                for pid in ids:
                    v = st[pid]["r"]
                    bad = max(bad, float(max((-v).clamp_min(0).max(),
                                             (v - 1).clamp_min(0).max())))
        worst = max(worst, bad)
        cases.append({"dt": dt, "steps": n_steps,
                      "worst_excursion_over_all_steps": bad,
                      "delays_that_vanish_at_this_dt": circ.vanishing_delays(dt),
                      "gpi_hz_end": [round(float(st[p]["r"].mean()) * R_MAX, 2)
                                     for p in pop_ids("gpi")]})
    return {"ok": worst <= 0.0, "worst_excursion": worst,
            "checked": "every basal-ganglia population's rate at every step of every case, "
                       "not the endpoint",
            "cases_per_dt": "6 in the batch: injected current -5/0/+5 on EVERY bg "
                            "population at once, crossed with a saturating stop pulse, all "
                            "four cortical proposals at 0.999",
            "populations_tested": len(ids),
            "not_tested": "eta -- the OU background is a CURRENT, not a rate, and the "
                          "engine does not claim it is in [0, 1]",
            "cases": cases}


def gate_idempotent(c):
    """B1.  The declarations are pure, the wiring is a function of the seed, the run repeats."""
    decl = {}
    for name in ("pops", "internal", "external", "mods", "drives"):
        a, b = getattr(BG, name)(c), getattr(BG, name)(c)
        decl[name] = bool(a == b)
    ca, _ = build(c, seed=3)
    cb, _ = build(c, seed=3)
    wiring = all(torch.equal(ca._W[k], cb._W[k]) for k in ca._W)
    rec = pop_ids("gpi") + pop_ids("stn")
    ta = run(ca, c, 0.8, PROPOSALS, on=0.3, b=4, seed=7, record=rec)
    tb = run(ca, c, 0.8, PROPOSALS, on=0.3, b=4, seed=7, record=rec)
    same = all(torch.equal(ta[k], tb[k]) for k in ta)
    return {"ok": bool(all(decl.values()) and wiring and same),
            "declarations_pure": decl, "wiring_identical_at_same_seed": bool(wiring),
            "rollout_bit_identical": bool(same),
            "max_abs_diff": {k: float((ta[k] - tb[k]).abs().max()) for k in ta},
            "note": "not a known-answer check: it tests whether the module is a FUNCTION "
                    "of its arguments at all (CLAUDE.md, 'call it twice at the same input')"}


def gate_selection(c):
    """B2.  Exactly one GPi pause, everyone else at or above rest, and the right winner."""
    circ, _ = build(c)
    rec = pop_ids("gpi") + pop_ids("stn") + pop_ids("gpe") + pop_ids("d1") + pop_ids("d2")
    k_on = 300
    tr = run(circ, c, 1.2, PROPOSALS, on=0.3, b=8, record=rec)
    rest, final, ratio = gpi_ratio(tr, k_on)
    drop = 1.0 - ratio
    order = torch.argsort(ratio)
    win, run2 = int(order[0]), int(order[1])
    n_below = int((ratio < SELECT_MAX).sum())
    losers = [i for i in range(len(BG.CHANNELS)) if i != win]
    losers_ok = bool((ratio[losers] >= LOSER_FLOOR).all())
    expect = list(BG.CHANNELS).index(max(PROPOSALS, key=PROPOSALS.get))
    ok = (n_below == 1) and losers_ok and win == expect
    w = 150
    return {"ok": bool(ok),
            "n_channels_below_threshold": n_below, "threshold_frac_of_rest": SELECT_MAX,
            "winner": BG.CHANNELS[win], "winner_has_largest_proposal": win == expect,
            "winner_gpi_drop": float(drop[win]),
            "runner_up": BG.CHANNELS[run2], "runner_up_gpi_drop": float(drop[run2]),
            "margin_in_drop": float(drop[win] - drop[run2]),
            "losers_all_at_or_above_rest": losers_ok, "loser_floor": LOSER_FLOOR,
            "proposals": PROPOSALS,
            "gpi_hz_rest": {ch: round(float(rest[i]) * R_MAX, 2)
                            for i, ch in enumerate(BG.CHANNELS)},
            "gpi_hz_final": {ch: round(float(final[i]) * R_MAX, 2)
                             for i, ch in enumerate(BG.CHANNELS)},
            "other_nuclei_hz_final": {
                k: {ch: round(float(chan_mean(tr, k)[i, :, -w:].mean()) * R_MAX, 2)
                    for i, ch in enumerate(BG.CHANNELS)}
                for k in ("d1", "d2", "gpe", "stn")},
            "declared_rest_hz": {k: round(v, 2) for k, v in BG.rest_rates(c).items()}}


def gate_stop(c):
    """B3.  A hyperdirect pulse re-raises GPi on ALL channels, and how long it takes."""
    circ, _ = build(c)
    seconds, t_pulse = 2.0, 1.2
    k_on, k0 = 300, int(round(t_pulse / DT))
    tr = run(circ, c, seconds, PROPOSALS, on=0.3, b=4,
             stop=(t_pulse, t_pulse + 0.30, STOP_PULSE),
             record=pop_ids("gpi") + pop_ids("stn"))
    g = chan_mean(tr, "gpi").mean(1)                    # (n_ch, T), batch averaged
    rest = g[:, k_on - 100:k_on].mean(1)
    pre = g[:, k0 - 100:k0].mean(1) / rest
    selected = int((pre < SELECT_MAX).sum())
    allup = (g >= LOSER_FLOOR * rest[:, None]).all(0)
    idx = [i for i in range(k0, g.shape[1]) if bool(allup[i])]
    lat = (idx[0] - k0) * DT * 1000.0 if idx else None
    lo_ms, hi_ms = STOP_ROW.measure["latency_ms"]
    faster = lat is not None and lat < lo_ms
    return {"ok": bool(selected == 1 and lat is not None and lat <= STOP_MAX_MS),
            "channels_selected_before_pulse": selected,
            "latency_ms": lat, "declared_bar_ms": STOP_MAX_MS,
            "catalogue_latency_ms": [lo_ms, hi_ms],
            "faster_than_catalogue_floor": bool(faster),
            "finding_if_faster": "the catalogue's 120-200 ms is a behavioural stop-signal "
                                 "reaction time and includes the cortical detection of the "
                                 "cue, which this module does not contain; a shorter NEURAL "
                                 "latency is reported as a finding, not failed.  v2 "
                                 "measured 22.0 ms",
            "hyperdirect_path_ms": round((c.d_ctx_stn + c.d_stn_gpi) * 1000, 2),
            "gpi_hz_at_pulse": {ch: round(float(g[i, k0 - 1]) * R_MAX, 2)
                                for i, ch in enumerate(BG.CHANNELS)},
            "gpi_hz_50ms_after": {ch: round(float(g[i, k0 + 50]) * R_MAX, 2)
                                  for i, ch in enumerate(BG.CHANNELS)},
            "stn_hz_50ms_after": {ch: round(float(chan_mean(tr, "stn")[i, :, k0 + 50].mean())
                                            * R_MAX, 2)
                                  for i, ch in enumerate(BG.CHANNELS)}}


def gate_beta(c):
    """B4.  13-30 Hz in the STN-GPe pair at rest, gated on prominence, frequency beside it."""
    circ, _ = build(c)
    tr = run(circ, c, 6.0, b=32, seed=5, burn=0.3,
             record=pop_ids("stn") + pop_ids("gpe"))
    out, lo, hi = {}, *BETA.band
    for kind in ("stn", "gpe"):
        per = chan_mean(tr, kind)                       # (n_ch, B, T)
        avg = per.mean(0)                               # (B, T)
        f, p = spectrum(avg, 4096)
        singles = {ch: spectrum(per[i], 4096) for i, ch in enumerate(BG.CHANNELS)}
        out[kind] = {
            "channel_average_peak_hz": f, "channel_average_prominence_decades": p,
            "single_channel": {ch: {"peak_hz": v[0], "prominence_decades": v[1]}
                               for ch, v in singles.items()},
            "mean_hz": float(avg.mean()) * R_MAX, "sd_hz": float(avg.std()) * R_MAX,
            "envelope_cv_reported_not_gated": envelope_cv(avg[0]),
        }
    f, p = out["stn"]["channel_average_peak_hz"], out["stn"]["channel_average_prominence_decades"]
    return {"ok": bool(p > 0.0 and lo <= f <= hi),
            "band": [lo, hi], "catalogue_peak_hz": BETA.peak,
            "gated_on": "stn channel-average prominence > 0 and its peak inside the band",
            "phase_condition_prediction_hz": _phase_crossing(c),
            "populations": out,
            "note": "peak_frequency is a soft-argmax and returns the band centre on a flat "
                    "spectrum, so the prominence is the gate and the frequency is context. "
                    "The single-channel readings are here because a statistic computed on "
                    "the population average and on one unit can differ by 2x, and where "
                    "they do at most one of them is about the circuit (CLAUDE.md). The "
                    "envelope CV is reported only: a 17 Hz-wide band has an envelope "
                    "correlation time near 59 ms whatever generates the signal"}


def _phase_crossing(c):
    """where the STN-GPe ring's phase reaches pi, from the declared delays and taus.

    Not a fit and not a measurement: the algebra the module's constants imply, printed
    beside the measured peak so the two can disagree visibly.
    """
    d = c.d_gpe_stn + c.d_stn_gpe

    def phase(f):
        w = 2 * math.pi * f
        return w * d + math.atan(w * c.tau_stn) + math.atan(w * c.tau_gpe)
    lo, hi = 1.0, 300.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if phase(mid) < math.pi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gate_dopamine(c):
    """B5.  Lower nigral rate: beta UP, selection DOWN.  Declared above, before the run."""
    circ, _ = build(c)
    states = {}
    for name, da in (("tonic_0.50", DA_TONIC), ("low_0.20", DA_LOW)):
        q = run(circ, c, 2.0, b=32, seed=5, burn=0.3, da=da,
                record=pop_ids("stn") + pop_ids("gpe") + pop_ids("d2"))
        f, p = spectrum(chan_mean(q, "stn").mean(0), 1024)
        tr = run(circ, c, 1.2, PROPOSALS, on=0.3, b=8, da=da,
                 record=pop_ids("gpi"))
        rest, final, ratio = gpi_ratio(tr, 300)
        # PAIRED on the same channel in both conditions -- the one with the largest
        # proposal, which is the channel that ought to win.  Taking argmin separately at
        # each level would compare two different channels and would report a "winner" even
        # when nothing was selected at all.
        win = list(BG.CHANNELS).index(max(PROPOSALS, key=PROPOSALS.get))
        states[name] = {
            "nigral_rate": da,
            "beta_peak_hz": f, "beta_prominence_decades": p,
            "stn_mean_hz": float(chan_mean(q, "stn").mean()) * R_MAX,
            "gpe_mean_hz": float(chan_mean(q, "gpe").mean()) * R_MAX,
            "d2_rest_hz": float(chan_mean(q, "d2").mean()) * R_MAX,
            "channel": BG.CHANNELS[win],
            "winner_gpi_drop": float(1.0 - ratio[win]),
            "n_channels_selected": int((ratio < SELECT_MAX).sum()),
            "gpi_hz_final": {ch: round(float(final[i]) * R_MAX, 2)
                             for i, ch in enumerate(BG.CHANNELS)}}
    beta_up = (states["low_0.20"]["beta_prominence_decades"]
               > states["tonic_0.50"]["beta_prominence_decades"])
    weaker = (states["low_0.20"]["winner_gpi_drop"]
              < states["tonic_0.50"]["winner_gpi_drop"])
    return {"ok": bool(beta_up and weaker),
            "beta_rose": bool(beta_up), "selection_weakened": bool(weaker),
            "d_prominence": (states["low_0.20"]["beta_prominence_decades"]
                             - states["tonic_0.50"]["beta_prominence_decades"]),
            "d_winner_drop": (states["low_0.20"]["winner_gpi_drop"]
                              - states["tonic_0.50"]["winner_gpi_drop"]),
            "declared_direction": "lowering the rate of nm.snc/nm.vta raises the resting "
                                  "beta prominence AND shrinks the winner's GPi drop -- the "
                                  "Parkinsonian direction, from one change, with no branch "
                                  "on dopamine anywhere in the module",
            "states": states}


def gate_channels(c):
    """B6.  Drive each channel alone; nothing else may be released."""
    circ, _ = build(c)
    rows, worst = {}, -9.9
    for ch in BG.CHANNELS:
        tr = run(circ, c, 1.0, {ch: 0.90}, on=0.3, b=4, record=pop_ids("gpi"))
        rest, final, ratio = gpi_ratio(tr, 300, window=0.12)
        drop = 1.0 - ratio
        i = list(BG.CHANNELS).index(ch)
        leak = {o: float(drop[j]) for j, o in enumerate(BG.CHANNELS) if j != i}
        worst = max(worst, max(leak.values()))
        rows[ch] = {"driven_channel_drop": float(drop[i]),
                    "driven_channel_released": bool(ratio[i] < SELECT_MAX),
                    "leakage_drop_on_undriven": leak,
                    "n_released": int((ratio < SELECT_MAX).sum()),
                    "gpi_hz_final": {o: round(float(final[j]) * R_MAX, 2)
                                     for j, o in enumerate(BG.CHANNELS)}}
    all_clean = all(r["n_released"] == 1 and r["driven_channel_released"]
                    for r in rows.values())
    return {"ok": bool(all_clean and worst <= LEAK_MAX),
            "worst_leakage_drop": worst, "leak_max": LEAK_MAX,
            "sign_convention": "leakage is a DROP, so negative means the undriven channel's "
                               "GPi ROSE, which is the prediction: the only thing crossing "
                               "channels is the subthalamic efferent and it pushes GPi up",
            "per_driven_channel": rows}


# ---- B7 -------------------------------------------------------------------------------
SWEEP_QUIET, SWEEP_DRIVE, SWEEP_BURN, SWEEP_B = 0.6, 0.35, 0.10, 64
#: constants that are rates or mixing fractions and live in (0, 1).  1.5x takes `rest_gpi`
#: to 1.05 and `stn_diffuse` to 1.2, which are not large values but impossible ones -- the
#: threshold solve inverts a sigmoid at the resting rates.  They are CLAMPED to the edge of
#: their range and the clamp is recorded in the row, so the sweep reports what it actually
#: swept rather than silently narrowing it.
UNIT_CONSTANTS = {"rest_gpe", "rest_gpi", "rest_stn", "stn_diffuse", "da_baseline"}

#: THE DIAGNOSIS, written before the sweep runs, so that a constant found inert is placed in
#: a class rather than explained after the fact.  A large inert list is informative only if
#: every entry in it has a reason; an entry that lands in "UNDIAGNOSED" is the finding --
#: it means a mechanism was declared and is not wired in (CLAUDE.md, "A parameter that
#: changes nothing").  The four classes are v2's, because they are the four ways a sweep can
#: come back flat for a reason that is not "the model does not care".
INERT_CLASS = {
    "w_gpi_thal": "open loop in THIS HARNESS: no thalamic relay here, so the edge is "
                  "dropped as an orphan.  A full assembly keeps it",
    "d_gpi_thal": "open loop in THIS HARNESS: no thalamic relay here, so the edge is "
                  "dropped as an orphan.  A full assembly keeps it",
    "k_da_gain": "not exercised: measured at the nigral tonic rate, where Mod's factor is 1",
    "k_da_theta": "not exercised: measured at the nigral tonic rate, where Mod's factor is 1",
    "w_stop_stn": "not exercised: no stop pulse in either measurement (B3 moves it)",
    "d_ctx_stn": "not exercised: no stop pulse, and the proposal's collateral is a delay "
                 "into a steady state",
    "tau_msn": "steady-state observables: sets how fast a selection arrives, not its level",
    "tau_gpi": "steady-state observables: sets how fast the pause arrives, not its depth",
    "tau_eta": "steady-state observables: the background's corner, well outside the band",
    "d_ctx_str": "steady-state observables: a delay in a feedforward chain",
    "d_str_gpi": "steady-state observables: a delay into GPi, which nothing reads back",
    "d_str_gpe": "steady-state observables: a delay in a feedforward chain",
    "d_stn_gpi": "steady-state observables: a delay into GPi, which nothing reads back",
    "d_gpe_gpi": "steady-state observables: a delay into GPi, which nothing reads back",
    "d_str_lat": "steady-state observables: a delay inside a winner-take-all that settles",
}


def sweep_measure(c, seed=5):
    """ONE rollout: quiet (the spectrum) then driven (the selection margin).

    Declared in the module docstring above: same seed in every arm, so the background is
    the same draw and the comparison between arms is paired on it.
    """
    circ, _ = build(c)
    k_on = int(round(SWEEP_QUIET / DT))
    tr = run(circ, c, SWEEP_QUIET + SWEEP_DRIVE, PROPOSALS, on=SWEEP_QUIET,
             b=SWEEP_B, seed=seed, burn=SWEEP_BURN,
             record=pop_ids("stn") + pop_ids("gpi"))
    stn = chan_mean(tr, "stn")[:, :, :k_on].mean(0)
    f, p = spectrum(stn, 512)
    _, _, ratio = gpi_ratio(tr, k_on, window=0.10)
    drop = 1.0 - ratio
    o = torch.argsort(ratio)
    return f, p, float(drop[int(o[0])] - drop[int(o[1])])


def gate_sensitivity(verbose=True):
    """B7.  Every declared constant +-50%, against the beta peak AND the selection margin."""
    base = BG.Constants()
    names = BG.constant_names()
    f0, p0, m0 = sweep_measure(base)
    rows = []
    for name in names:
        v = float(getattr(base, name))
        got, used, clamped = {}, {}, False
        for tag, scale in (("half", 0.5), ("1p5x", 1.5)):
            want = v * scale
            val = min(max(want, 0.005), 0.995) if name in UNIT_CONSTANTS else want
            clamped |= (val != want)
            used[tag] = val
            got[tag] = sweep_measure(dataclasses.replace(base, **{name: val}))
        rows.append({"param": name, "value": v, "swept_to": used,
                     "clamped_to_unit_interval": clamped,
                     "hz_at_half": got["half"][0], "hz_at_1p5x": got["1p5x"][0],
                     "span_hz": abs(got["1p5x"][0] - got["half"][0]),
                     "prom_at_half": got["half"][1], "prom_at_1p5x": got["1p5x"][1],
                     "span_prominence": abs(got["1p5x"][1] - got["half"][1]),
                     "margin_at_half": got["half"][2], "margin_at_1p5x": got["1p5x"][2],
                     "span_margin": abs(got["1p5x"][2] - got["half"][2])})
        if verbose:
            r = rows[-1]
            print(f"      {name:14s} beta {r['hz_at_half']:6.2f} -> {r['hz_at_1p5x']:6.2f} Hz"
                  f" (span {r['span_hz']:5.2f})   margin {r['margin_at_half']:+.3f} -> "
                  f"{r['margin_at_1p5x']:+.3f} (span {r['span_margin']:5.3f})", flush=True)
    clock = {r["param"]: r["span_hz"] for r in rows if r["param"] in ("tau_stn", "tau_gpe")}
    sel = {r["param"]: r["span_margin"] for r in rows if r["param"] == "w_d1_gpi"}
    inert = [r["param"] for r in rows
             if r["span_hz"] < INERT_HZ and r["span_margin"] < INERT_MARGIN]
    zeroed = [r["param"] for r in rows if r["value"] == 0.0]
    ok = (all(v > CLOCK_HZ for v in clock.values())
          and all(v > SELECT_SENS for v in sel.values()))
    return {"ok": bool(ok),
            "baseline_beta_hz": f0, "baseline_prominence": p0, "baseline_margin": m0,
            "claimed_clock_span_hz": clock, "claimed_selection_span_margin": sel,
            "rule": f"tau_stn and tau_gpe must each move the beta peak by > {CLOCK_HZ} Hz; "
                    f"w_d1_gpi must move the selection margin by > {SELECT_SENS}",
            "inert_both": [p for p in inert if p not in zeroed],
            "inert_but_declared_zero": zeroed,
            "inert_diagnosis": {p: INERT_CLASS.get(p, "UNDIAGNOSED -- a mechanism that was "
                                                      "declared and is not wired in")
                                for p in inert if p not in zeroed},
            "inert_predicted_and_not_found": [p for p in INERT_CLASS if p not in inert],
            "inert_definition": f"span < {INERT_HZ} Hz in beta AND < {INERT_MARGIN} in the "
                                f"selection margin, over the whole 3x sweep",
            "instrument": {"quiet_s": SWEEP_QUIET, "driven_s": SWEEP_DRIVE,
                           "batch": SWEEP_B, "nperseg": 512, "paired_seed": 5,
                           "bias_vs_gate_B4": "about +0.6 Hz high, common to every arm; "
                                              "this gate reports SPANS, not levels"},
            "measured_at": {"nigral_rate": DA_TONIC, "resting_cortical_drive": 0.0,
                            "proposals": PROPOSALS, "stop_pulse": 0.0},
            "blind_spots": {
                "w_gpi_thal/d_gpi_thal": "the pallidothalamic edge's TARGET IS NOT IN THIS "
                    "HARNESS -- the gate builds the basal ganglia with stub cortical and "
                    "nigral populations and no thalamus, so the edge is dropped exactly as "
                    "ibm.brain.collect drops an orphan.  These two cannot move anything "
                    "here by construction, and the exactly-zero span they produce is what "
                    "calibrated this sweep's pairing.  They ARE kept in a full assembly "
                    "with ibm/brain/thalamus.py, where they close the cbgtc loop, and "
                    "measuring them needs that assembly and not this gate.",
                "k_da_gain/k_da_theta": "both observables are measured at a nigral rate of "
                    f"{DA_TONIC}, which is `da_baseline`, where Mod's factor is exactly "
                    "1 + gain * 0 = 1.  They cannot move anything here by construction; "
                    "gate B5 is what exercises them and it moves both numbers.",
                "w_stop_stn/d_ctx_stn": "no stop pulse is delivered in either measurement, "
                    "so the stopping projection is never engaged.  Gate B3 exercises it.",
                "steady-state observables": "the margin is averaged over the last 100 ms of "
                    "the driven window and the spectrum is measured at rest, so a constant "
                    "that only sets how fast a selection ARRIVES, or that shifts the phase "
                    "of a signal on its way into an output nucleus nothing feeds back from, "
                    "has nothing to move here.  That is a limitation of this gate's choice "
                    "of observables and B3's 37 ms latency is where those constants live.",
                "GPi is a leaf": "nothing in this module reads GPi -- its only efferent is "
                    "the orphaned pallidothalamic edge -- so every constant that belongs to "
                    "GPi alone can move the selection margin (which is measured on it) and "
                    "cannot move the STN spectrum.  A real structural fact about an open "
                    "cortico-basal-ganglia loop, not a broken mechanism."},
            "sweep": sorted(rows, key=lambda r: -r["span_hz"])}


def sweep_loop_gain(values):
    """the one constant a MEASUREMENT sets: the STN-GPe loop gain against the resonance.

    Recorded whole, including the part where the pair crosses its bifurcation and the
    envelope collapses into a tone, rather than nudged until a gate passed.
    """
    rows = []
    for v in values:
        c = dataclasses.replace(BG.C, w_gpe_stn=v, w_stn_gpe=v)
        circ, _ = build(c)
        tr = run(circ, c, 4.0, b=16, seed=5, burn=0.3, record=pop_ids("stn"))
        avg = chan_mean(tr, "stn").mean(0)
        f, p = spectrum(avg, 2048)
        cv = envelope_cv(avg[0])
        rows.append({"loop_weight": v, "peak_hz": f, "prominence_decades": p,
                     "envelope_cv": cv, "stn_mean_hz": float(avg.mean()) * R_MAX,
                     "stn_sd_hz": float(avg.std()) * R_MAX})
        print(f"  w {v:5.2f} -> {f:6.2f} Hz  prominence {p:+.3f}  envelope CV {cv:.3f}  "
              f"STN {float(avg.mean()) * R_MAX:5.2f} +- {float(avg.std()) * R_MAX:.2f} Hz",
              flush=True)
    return rows


# ======================================================================================
def _coerce(o):
    if torch.is_tensor(o):
        return {"tensor_shape": list(o.shape)}
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", default="",
                    help="comma-separated subset, e.g. B0,B1,B7.  Default: all.")
    ap.add_argument("--sweep", action="store_true", help="the loop-gain curve, not a gate")
    ap.add_argument("--out", default="out/gate_brain_basal_ganglia.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.set_num_threads(2)
    c = BG.C
    circ, orphans = build(c)

    rec = {"script": "scripts/gate_brain_basal_ganglia.py",
           "structure": BG.STRUCTURE, "dt": DT,
           "channels": list(BG.CHANNELS), "units_per_pop": BG.N_UNITS,
           "n_populations": len(BG.pops(c)), "n_internal": len(BG.internal(c)),
           "n_external": len(BG.external(c)), "n_mods": len(BG.mods(c)),
           "stub_populations": STUB_IDS,
           "external_edges_orphaned_in_this_harness": orphans,
           "declared_rest_hz": BG.rest_rates(c),
           "solved_tonic_drives": BG.drives(c),
           "delays_in_steps": circ.delays_in_steps(DT),
           "vanishing_delays": circ.vanishing_delays(DT),
           "path_delays_ms": {
               "hyperdirect_ctx_to_gpi": round((c.d_ctx_stn + c.d_stn_gpi) * 1000, 2),
               "direct_ctx_to_gpi": round((c.d_ctx_str + c.d_str_gpi) * 1000, 2),
               "indirect_ctx_to_gpi": round((c.d_ctx_str + c.d_str_gpe + c.d_gpe_stn
                                             + c.d_stn_gpi) * 1000, 2)},
           "bars": {"select_max_frac_of_rest": SELECT_MAX, "loser_floor": LOSER_FLOOR,
                    "stop_max_ms": STOP_MAX_MS, "leak_max": LEAK_MAX,
                    "clock_hz": CLOCK_HZ, "select_sens": SELECT_SENS,
                    "inert_hz": INERT_HZ, "inert_margin": INERT_MARGIN},
           "gates": {}}

    def save():
        # the cheap summary is written BEFORE anything that can raise, and again after
        # every gate, so a crash in gate k leaves gates 0..k-1 on disk (CLAUDE.md, "Jobs")
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=_coerce)

    save()
    if a.sweep:
        print("STN-GPe loop gain against the resonance (4 s rest, batch 16, "
              "channel-averaged STN):", flush=True)
        rec["sweep_loop_gain"] = sweep_loop_gain([1.10, 1.25, 1.40, 1.55, 1.70, 1.85, 2.00])
        save()
        print(f"\nwrote {a.out}")
        return 0

    order = [("B0_bounded", lambda: gate_bounded(c)),
             ("B1_idempotent", lambda: gate_idempotent(c)),
             ("B2_selection", lambda: gate_selection(c)),
             ("B3_stop", lambda: gate_stop(c)),
             ("B4_beta", lambda: gate_beta(c)),
             ("B5_dopamine", lambda: gate_dopamine(c)),
             ("B6_channels", lambda: gate_channels(c)),
             ("B7_sensitivity", gate_sensitivity)]
    want = [g.strip() for g in a.gates.split(",") if g.strip()]
    if want:
        order = [(n, f) for n, f in order if n.split("_")[0] in want]
        assert order, f"--gates {a.gates} matched nothing"

    ok_all, ran = True, []
    for name, fn in order:
        t0 = time.time()
        if name == "B7_sensitivity":
            print(f"[....] {name}: sweeping every declared constant +-50% "
                  f"({len(BG.constant_names())} of them) ...", flush=True)
        r = fn()
        r["seconds"] = round(time.time() - t0, 1)
        rec["gates"][name] = r
        ran.append(name)
        save()
        ok_all &= bool(r.get("ok"))
        head = {k: v for k, v in r.items()
                if k not in ("sweep", "cases", "states", "max_abs_diff", "populations",
                             "per_driven_channel", "blind_spots", "other_nuclei_hz_final")}
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {name}: {head}", flush=True)
        if name == "B4_beta":
            for k, v in r["populations"].items():
                print(f"      {k}: channel average {v['channel_average_peak_hz']:6.2f} Hz "
                      f"prominence {v['channel_average_prominence_decades']:+.3f}   "
                      f"singles "
                      f"{[round(s['peak_hz'], 2) for s in v['single_channel'].values()]} / "
                      f"{[round(s['prominence_decades'], 2) for s in v['single_channel'].values()]}"
                      f"   envelope CV {v['envelope_cv_reported_not_gated']:.3f}", flush=True)
        if name == "B5_dopamine":
            for k, v in r["states"].items():
                print(f"      {k:11s} beta {v['beta_peak_hz']:6.2f} Hz prom "
                      f"{v['beta_prominence_decades']:+.3f}   STN {v['stn_mean_hz']:5.2f} Hz "
                      f"D2 {v['d2_rest_hz']:5.2f} Hz   winner drop "
                      f"{v['winner_gpi_drop']:+.3f}  selected {v['n_channels_selected']}",
                      flush=True)
        if name == "B6_channels":
            for k, v in r["per_driven_channel"].items():
                print(f"      drive {k:12s} -> released {v['n_released']}, drop "
                      f"{v['driven_channel_drop']:+.3f}, leakage "
                      f"{ {o: round(x, 4) for o, x in v['leakage_drop_on_undriven'].items()} }",
                      flush=True)
        if name == "B7_sensitivity":
            print(f"      inert over the whole 3x sweep: "
                  f"{', '.join(r['inert_both']) or 'none'}", flush=True)
            print(f"      inert because declared at zero (not a finding): "
                  f"{', '.join(r['inert_but_declared_zero']) or 'none'}", flush=True)
            und = [p for p, v in r["inert_diagnosis"].items() if v.startswith("UNDIAGNOSED")]
            print(f"      UNDIAGNOSED inert constants: {', '.join(und) or 'none'}",
                  flush=True)
            print(f"      predicted inert and NOT inert: "
                  f"{', '.join(r['inert_predicted_and_not_found']) or 'none'}", flush=True)

    rec["gates_run"] = ran
    rec["all_gates_ok"] = ok_all
    rec["failed"] = [k for k, v in rec["gates"].items() if not v.get("ok")]
    save()
    print(f"\n{'ALL GATES PASS' if ok_all else 'SOME GATES FAILED: ' + ', '.join(rec['failed'])}"
          f" -- wrote {a.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
