"""known answers for `ibm/brain/hypothalamus.py`, and the sweep that says which of its
constants are doing anything.

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        .venv/bin/python scripts/gate_brain_hypothalamus.py
    ... scripts/gate_brain_hypothalamus.py --gates Y0 Y1 Y2 Y3 Y4 Y5     # then
    ... scripts/gate_brain_hypothalamus.py --gates Y6 --merge            # merges the JSON

CPU ONLY.  On the GB10 the GPU shares the machine's one 121 GB pool, so a GPU OOM is a
machine OOM (CLAUDE.md, Jobs).  Nothing here needs a GPU and nothing here should be run
with one.

EVERY VERDICT BELOW IS FIXED HERE, IN THE CODE, BEFORE IT RUNS.  A gate that fails is
recorded FAILED and left failed; it is never rescored and never re-run with a different
bar.  Instruments may change after a failure, thresholds may not.

THE TIMESTEP, AND WHAT IT COSTS
-------------------------------
The circadian period is 86 400 s.  Simulating it at the 1 ms step the rest of the brain
runs at would be 8.6e7 steps per day of model time, so **Y2 and Y3 run at dt = 60 s**
(1 440 steps per cycle), Y4 and Y5 at dt = 60 s, and Y6's sweep at dt = 150 s.  Three
things are traded away and all three are measured rather than assumed:

  * **every conduction delay in the module vanishes.**  The module's delays run 5-40 ms;
    at dt = 60 s all of them round to zero steps.  Y0 prints `vanishing_delays` at each
    dt it uses, so the count is in the record.  Nothing gated here is set by a delay --
    `ibm/rhythms.py` declares `circadian` as `set_by="time constants"`, and the `circ`
    ring's 80 ms round trip puts its CEILING at ~12 Hz, six orders of magnitude above the
    rhythm, so the bound is nowhere near binding -- but the
    sleep switch's 15 ms mutual inhibition becomes instantaneous, so the flip is a single
    step rather than a 15 ms race.
  * **the fast populations are at their fixed point every step.**  `hyp.vlpo` and
    `hyp.lh` have tau = 60 s, so at dt = 60 s their exponential-Euler factor is 0.63 and
    they are resolved, barely; `val.nacc_shell` in Y5 has tau = 0.05 s and is exactly at
    its fixed point, so Y5 measures a STEADY-STATE response and says so.
  * **the period estimate itself moves with dt.**  Measured on this module: 24.181 h at
    dt = 60, 24.353 h at dt = 120, 24.438 h at dt = 150, 24.525 h at dt = 180.  Y2 and
    Y3 therefore quote dt = 60 and Y6 compares only the SPAN between the two ends of each
    sweep, where the bias is common to both ends.

Y0  BOUNDED.  Every rate stays in [0, 1] under extreme drive (+-5.0 on every population at
    once, and on each one alone), at three timesteps -- 1e-3, 60 and 300 s -- including
    two too coarse to resolve any delay in the module, which is reported rather than
    hidden.  The adaptation variable `a` is NOT a rate and is not in the box; it is
    checked separately against [0, adapt_g].  PASS iff no excursion anywhere.

Y1  IDEMPOTENCE.  Five forms, because they fail differently.  (b), (c) and (e) are run at
    dt = 1e-3 as well as dt = 60 s ON PURPOSE: at dt = 60 s every delay in this module
    rounds to zero steps and `init_state` allocates no ring at all, so the three forms
    that exist to catch a ring mutated in place would otherwise be testing a code path
    that is never taken.
      (a) the same rollout twice from the same state, bit-identical;
      (b) `step` called TWICE on the SAME state returns bit-identical states;
      (c) the same under `torch.no_grad()`, which is a DIFFERENT code path in
          `Circuit.step`: the ring is cloned only when grad is enabled;
      (d) the declaration itself: `pops`, `internal`, `external`, `mods`, `drives` called
          twice must return equal objects, and two `Circuit`s built at the same seed must
          have identical weights;
      (e) BRANCHING.  `ibm/circuit.py`'s class docstring claims the rings are "carried in
          the state rather than on the module, so two rollouts from the same state cannot
          interfere".  Branch A from a state, branch B from the same state with a
          different drive, re-run A.  A must come back identical.
    This is not a known-answer check.  A known answer tests the value; idempotence tests
    whether the thing is a function at all (CLAUDE.md).  PASS iff (a), (b), (d) and (e)
    are exactly equal.  (c) is REPORTED and not gated.
    (e) was written after (c) came back clean, and it is GATED, and it FAILS -- see the
    record.  It is left failed: a gate is never rescored and never quietly re-run with a
    different bar, and the fact that the defect is in a file this task may not edit is a
    reason to report it, not a reason to lower the bar.

Y2  THE CLOCK FREE-RUNS.  `hyp.scn` is run for 40 days at dt = 60 s with **no input at
    all** -- `drives()` gives the suprachiasmatic nucleus nothing and no projection in the
    assembled module targets it, and the gate asserts both before measuring.  The period
    is the mean interval between upward mean-crossings; the spectrum is Welch with
    nperseg = 16384 (11.4 days per segment, df = 1.02e-6 Hz) and `peak_prominence` is
    quoted BESIDE `peak_frequency`, never instead of it.
    DECLARED WINDOW: period in [22.0, 26.0] h.  PASS iff the crossing period is in that
    window AND `peak_prominence` over `circadian`'s declared band (1.0e-5 - 1.3e-5 Hz) is
    > 0 AND the cycle-to-cycle sd is under 0.25 h.
    Three known-answer controls run through the identical estimator, because a frequency
    read off a spectrum with no peak in it is the band you chose and not a measurement:
    a 24.000 h sine (must return a positive prominence, and its period error is the
    instrument's bias, quoted so the SCN's spectral period can be read against it), white
    noise (must return prominence <= 0) and a constant (must return exactly 0.000 and the
    band's midpoint).  If a control fails, the gate is reported FAILED whatever the SCN
    did, because the instrument is then not measuring what it is being read as measuring.
    THIS GATE FAILED AND IS LEFT FAILED.  Two defects, both in the instrument and neither
    in the module:
      * the precondition was written as "no projection targets hyp.scn", and `hyp.scn ->
        hyp.scn` -- the recurrent self-excitation that IS the oscillator -- trips it.  A
        clock's own loop is not an input.
      * the white-noise control is ONE draw.  With ~7 Welch segments and three bins in the
        band, a single realisation's prominence is a coin flip: -0.145 at seed 0 in
        development and +0.145 at seed 7 here.  "prominence <= 0 for white noise" is not a
        known answer, it is a 50% test.
    The SCN's own numbers from that run are kept in the record: period 24.181 +- 0.006 h
    over 26 cycles, prominence +4.130 decades.

Y2b THE CLOCK FREE-RUNS, the replacement instrument, declared here before it is run.  Y2
    stays FAILED.  The THRESHOLD is untouched -- the declared window is still [22.0, 26.0]
    h and the cycle-to-cycle sd bar is still 0.25 h -- and only the instrument changes:
      * the precondition asks whether any projection from a population OTHER THAN hyp.scn
        targets it, which is the question it was always meant to ask;
      * the null is an ENSEMBLE.  Sixteen white-noise draws, from one generator seeded
        here, go through the identical estimator, and their prominences give the null's
        mean and sd at this record length and resolution.  PASS requires the SCN's
        prominence to exceed that null mean by more than 5 null sd, which is a statement
        about a distribution rather than about one coin flip;
      * the 24.000 h sine and the constant stay as they were, because both are genuine
        known answers: a sine has a peak and a constant has a flat spectrum.
    PASS iff the precondition holds AND the sine and constant controls hold AND the
    crossing period is in [22.0, 26.0] h AND its sd is under 0.25 h AND the prominence
    clears the null by 5 sd.

Y3  THE CLOCK ENTRAINS, AND FAILS TO OUTSIDE ITS RANGE.  A square light signal -- 12 h on,
    12 h off, amplitude `W_LIGHT_SCN * LIGHT_IRRADIANCE` added to `hyp.scn`, which is what
    the orphaned `thal.lgn.relay -> hyp.scn` edge would deliver -- is applied at eight
    periods for 24 days at dt = 60 s.
    DECLARED: the clock MUST lock (|measured - T_light| <= 0.05 h) at T_light = 22, 23,
    24, 25 and 26 h, and MUST FAIL to lock (|measured - T_light| > 0.5 h) at 20 h and
    32 h.  16 h is run and REPORTED but not gated either way, because a relaxation
    oscillator frequency-demultiplies: it locks 3:2 onto a 16 h cycle and returns 24 h,
    which is neither entrainment to 16 h nor a failure to be pulled, and declaring it
    either way after seeing it would be the rule collision CLAUDE.md records.
    PASS iff every must-lock period locks AND both must-fail periods fail.  An oscillator
    that entrained to everything would be a filter, so the failure half is the half that
    says the clock has a period of its own.

Y4  SLEEP PRESSURE ACCUMULATES AND DISCHARGES.  Two measurements.
      (a) SPONTANEOUS, 20 days at dt = 60 s with nothing clamped.  Sleep is
          `hyp.vlpo > hyp.lh` -- a state the network is in, not a flag -- and `hyp.som`
          must RISE across every complete wake bout and FALL across every complete sleep
          bout.  Bout durations and the sleep fraction are reported.
      (b) CLAMPED, from `hyp.som` = 0: 60 h of forced wake (drive +3.0 on `hyp.lh`, -3.0
          on `hyp.vlpo`) then 30 h of forced sleep (the reverse).  The rise and fall time
          constants are fitted by regressing dr/dt on r over each phase, and the 10-90%
          rise time and 90-10% fall time are reported in hours.
    DECLARED: every bout must move `hyp.som` the right way (no exceptions, not a mean);
    the fitted rise tau must be within 25% of `TAU_SOM`; the fitted fall tau must be
    within 25% of `TAU_SOM * clamp(1 + SOM_CLEAR_GAIN * mean(hyp.vlpo))`, computed from
    the MEASURED `hyp.vlpo` in that phase rather than from an assumed 1.0; and the rise
    must be slower than the fall.  PASS iff all four.

Y5  A DRIVE CHANGES A VALUATION.  This script assembles the hypothalamus ALONE, so the
    target of the module's valuation edges is not in its circuit and has to be supplied:
    a STAND-IN `val.nacc_shell` population built inside this script and nowhere else --
    said plainly rather than faked.  `ibm/brain/valuation.py` was written the same day by
    another hand and DOES now declare a population of exactly that id; the gate looks it
    up and reports whether the name matches, because the first version of this module
    wrote `val.nac.shell` and three edges sat in the orphan list looking like a missing
    structure.  The edges into the stand-in are not invented here either: they are read
    off `hypothalamus.external()` and `hypothalamus.mods()` by filtering for that target,
    so what is tested is the module's own declaration.
    Four arms at dt = 60 s.  Hungry: the module's own `drives()`, with no relief, which is
    what a hypothalamus with no body attached does.  Sated: the same, plus the drive the
    orphaned `bs.nts -> hyp.arc` edge would deliver at a saturated NTS
    (`-W_NTS_ARC * 1.0`).  Both then receive the SAME cue -- a fixed drive on
    `val.nacc_shell` -- and the response is its mean rate over the last half hour.
    The two arms are repeated with the `hyp -> val` edges SEVERED, which is the control:
    if the hungry-sated difference survives severing, it was never carried by the declared
    edges (CLAUDE.md: gate on an ablation, never on a magnitude).
    DECLARED: hungry response - sated response > 0.05 intact, and |difference| < 0.005
    severed.  PASS iff both.  Because the val population's tau is 0.05 s and dt is 60 s
    this is a steady-state response, not a transient; the claim is about the level.
    THIS GATE FAILED AND IS LEFT FAILED.  The defect is in the stand-in, not the module:
    it was declared with `sparsity=None`, so it had no divisive normalisation, and three
    drive edges at weight 0.7 pinned it AT THE CEILING -- 1.0000 hungry against 0.9943
    sated, with the cue worth +0.0005 when hungry and +0.0806 when sated.  The ordering
    is right and the size is meaningless: everything is squashed against 1.  This is the
    exact property `docs/BRAIN_SPEC.md` says every population must have and that
    `ibm/brain/cortex.py` gates as `no_ignition`, and the stand-in did not have it.

Y5b A DRIVE CHANGES A VALUATION, the replacement instrument.  Y5 stays FAILED.
    The stand-in now carries a declared `sparsity`, which is what buys it divisive
    normalisation, and the gate is run at SIX stand-in sparsities -- None, 0.06, 0.08,
    0.10, 0.12 and 0.20 -- rather than at one.
    THE REASON IT IS SIX, stated because it changes what may be concluded: diagnosing
    Y5's failure, the same protocol was run at sparsity None, 0.10 and 0.20 and returned
    intact differences of +0.0057, +0.0495 and +0.0782, with the severed difference
    exactly 0.0000 at all three.  The DIRECTION is the same everywhere and the SIZE is
    monotone in a parameter of the instrument -- and it crosses Y5's 0.05 bar between two
    defensible choices of it.  A bar that a choice about the measuring device can move is
    not a threshold about the model, so Y5b does not gate on a magnitude at all.
    DECLARED: PASS iff the intact hungry-minus-sated difference is POSITIVE at every one
    of the six sparsities AND the severed difference is exactly 0.0 at every one.  The
    severed arm is the error bar: the system is deterministic, so a difference that
    survives cutting the only edges that could carry it would be a bookkeeping error and
    nothing else.  The magnitude at each sparsity, the active fraction, and whether the
    population saturated are all REPORTED and none of them is gated.

Y6  SENSITIVITY.  EVERY module-level numeric constant in `ibm.brain.hypothalamus` --
    discovered by introspection, so a constant added later cannot be forgotten -- swept
    x0.5 and x1.5, with the free-running circadian period and both somnogen time constants
    re-measured at each end (dt = 150 s, 7 days for the period, 90 h for the somnogen).
    A sweep that stops the oscillator is recorded as "no oscillation", which is the
    loudest possible non-inert result rather than a missing number.
    DECLARED, before the numbers: the six constants the module CLAIMS are the clock --
    `PER_TAU`, `PER_G`, `W_SCN_REC`, `BETA_SCN`, `THETA_SCN`, `TAU_SCN` -- must EACH move
    the period by more than 0.5 h over the 3x sweep, or the module's account of where its
    period comes from is wrong.  `TAU_SOM` and `SOM_CLEAR_GAIN` must each move a somnogen
    tau by more than 0.5 h.  Everything with a period span under 0.05 h AND both somnogen
    spans under 0.05 h is listed as inert.
    Inert constants are split three ways, because the three mean different things:
      orphan      it weights an edge whose other end is outside the hypothalamus, and
                  this gate assembles the hypothalamus alone, so the edge is not in the
                  circuit it sweeps;
      unmeasured  it is wired in, but neither of this gate's two measurements can see it
                  (`hyp.pvn` has no efferent inside the hypothalamus -- its only outputs
                  are to the brainstem and the pituitary -- so it is a SINK here and all
                  seven of its constants are expected in this bucket);
      unexplained  neither, and that is the bucket CLAUDE.md's "a parameter that changes
                  nothing" is about: five mechanisms in `ibm/thalamus.py` were found this
                  way in one day and every one was a feedback loop that was not closed.
    PASS iff the eight declared-live constants all move their quantity AND the
    `unexplained` bucket is empty.

Y6b SENSITIVITY, the replacement instrument, declared here before it is run.  Y6's verdict
    stands as recorded.  Its defect is COVERAGE, and it was visible in its first printed
    row: `BETA_ARC` came back with spans of exactly 0.000 on all three quantities, because
    the somnogen protocol CLAMPS `hyp.lh` and `hyp.vlpo` and the period run touches only
    the clock -- so between them the two measurements cannot see the hunger, thirst,
    thermal or stress pathway at all.  Every constant on those pathways would land in
    `unexplained`, and that would be a statement about the measurement, not about the
    module.  A parameter that changes nothing is only evidence when something was looking.
    Y6b adds a THIRD quantity and changes nothing else: a 6-day UNCLAMPED run at
    dt = 150 s with the first 2 days discarded, from which the mean rate of every
    population and the sleep fraction are read.  `span_state` is the largest change in any
    of those nine numbers across the 3x sweep.
    THE THRESHOLDS ON THE FIRST TWO QUANTITIES ARE UNCHANGED, and the eight
    declared-live constants are the same eight.  INERT now means span_period < 0.05 h AND
    both somnogen spans < 0.05 h AND span_state < 0.005.
    The inert buckets are declared here, before the run, so that no constant can be
    excused after the fact:
      orphan            weights an edge whose other end is outside the hypothalamus, and
                        this gate assembles the hypothalamus alone, so the edge is not in
                        the circuit it sweeps;
      vanishes_at_dt    is a conduction delay far below the sweep's 150 s timestep, so it
                        rounds to zero steps (`D_SLEEP_SWITCH`, 15 ms);
      clamp             is a `Mod` clamp rather than a mechanism.  A clamp that never
                        binds changes nothing BY DESIGN -- it is a guard.  The gate
                        reports, per clamp, whether it binds at baseline, so a guard that
                        is doing nothing and a guard that is holding a parameter down are
                        distinguishable;
      unmeasured        wired in, and still invisible to all three quantities.  `N` is the
                        only one declared here: every projection in the module is dense
                        and row-normalised, so every unit receives the same input whatever
                        n is, and `hyp.pvn` -- which Y6 could not see -- IS visible to
                        span_state through its own mean rate;
      unexplained       none of the above, and the bucket CLAUDE.md's "a parameter that
                        changes nothing" is about.
    PASS iff the eight declared-live constants all move their quantity AND `unexplained`
    is empty.
    THIS GATE ALSO FAILED AND IS ALSO LEFT FAILED.  Seven constants landed in
    `unexplained`: `GC_TAU`, `TAU_ARC`, `TAU_LH`, `TAU_MNPO`, `TAU_POA`, `TAU_PVN`,
    `TAU_VLPO`.  Every one of them is a TIME CONSTANT, and all three of Y6b's quantities
    are steady-state or asymptotic -- a long-run mean, a sleep fraction, an approach rate
    measured under a clamp that fixes the approach's endpoints.  A time constant does not
    move a fixed point.  So the bucket is not evidence that these seven do nothing; it is
    the same coverage defect as Y6's, one level in, and the gate is reported FAILED for it
    rather than excused.
    CLAUDE.md says a flat sweep is a reason to go and look, not a reason to conclude, so
    each of the seven was probed directly on a TRANSIENT -- a step in its own population's
    drive -- and the result is recorded here:

        const       measured transient          x0.5      x1.5    ratio
        TAU_ARC     10-90% of hyp.arc          3.069 h   4.547 h   1.48
        TAU_MNPO    10-90% of hyp.mnpo         2.175 h   4.246 h   1.95
        TAU_POA     10-90% of hyp.poa          0.375 h   1.119 h   2.99
        TAU_PVN     10-90% of hyp.pvn          0.076 h   0.232 h   3.04
        TAU_LH      10-90% of hyp.lh           0.018 h   0.056 h   3.08
        TAU_VLPO    10-90% of hyp.vlpo         0.018 h   0.056 h   3.08
        GC_TAU      peak-to-half decay of      0.600 h   1.281 h   2.14
                    hyp.pvn after a step

    All seven are wired in.  The four that scale at almost exactly 3.0x are doing what a
    time constant does and nothing else; `TAU_ARC` and `TAU_MNPO` scale less because their
    own sigmoid saturates within the step.  `GC_TAU` needed a different probe entirely:
    the 10-90% RISE of `hyp.pvn` is set by `TAU_PVN`, and the glucocorticoid surrogate acts
    on the DECAY back off the peak, so the first probe measured it at 1.01x and was wrong
    about it.  A Y6c would add one transient quantity to the sweep; it is not written here,
    because the finding is already established and a third replacement instrument in one
    file is a sign to stop and report rather than to keep instrumenting.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import spectral as SP                        # noqa: E402
from ibm.brain import hypothalamus as H               # noqa: E402
from ibm.circuit import Circuit, Pop                  # noqa: E402
from ibm.rhythms import RHYTHM                        # noqa: E402

DAY = 86400.0

# ---------------------------------------------------------------------------- declared
DT_SLOW = 60.0            # the circadian and sleep measurements
DT_SWEEP = 150.0          # Y6 only; the period bias at this dt is +0.26 h and common to
                          # both ends of every sweep, which is what a span is
CIRCADIAN = RHYTHM["circadian"]          # band (1.0e-5, 1.3e-5) Hz, loop `circ`

Y2_DAYS = 40
Y2_NPERSEG = 16384        # 11.4 days per segment at dt = 60 s; df = 1.017e-6 Hz
Y2_FIT = (2.0e-6, 2.0e-4)  # the 1/f background's fit range, band excluded by default
Y2_SEARCH = (5.0e-6, 2.6e-5)   # the peak_frequency search window: WIDER than the band,
                               # so a peak outside it is reported where it is
Y2_PERIOD_H = (22.0, 26.0)     # the declared free-running window
Y2_SD_H = 0.25                 # cycle-to-cycle sd

Y3_DAYS = 24
LIGHT_IRRADIANCE = 0.06   # the retinal rate the orphaned RHT edge would carry, in the
                          # engine's [0, 1] rate units.  the drive delivered is
                          # W_LIGHT_SCN * this, which is what the edge computes
Y3_LOCK = (22.0, 23.0, 24.0, 25.0, 26.0)
Y3_FAIL = (20.0, 32.0)
Y3_REPORT_ONLY = (16.0,)  # frequency demultiplication; declared un-gated BEFORE the run
Y3_LOCK_TOL_H = 0.05
Y3_FAIL_TOL_H = 0.50

Y4_DAYS = 20
Y4_WAKE_H = 60.0
Y4_SLEEP_H = 30.0
Y4_CLAMP = 3.0            # how hard forced wake and forced sleep are held
Y4_TAU_TOL = 0.25         # fitted tau must be within this fraction of the declared one

Y5_CUE = 0.35             # the cue drive on the stand-in valuation population
Y5_SETTLE_H = 24.0
Y5_CUE_H = 1.0
Y5_READ_H = 0.5           # the last half hour of the cue window
Y5_MIN_EFFECT = 0.05
Y5_MAX_SEVERED = 0.005
Y5_NTS_SATED = 1.0        # a saturated nucleus of the solitary tract

Y6_DAYS = 7
Y6_WAKE_H = 60.0
Y6_SLEEP_H = 30.0
Y6_INERT_PERIOD_H = 0.05
Y6_INERT_TAU_H = 0.05
Y6_CLOCK = ("PER_TAU", "PER_G", "W_SCN_REC", "BETA_SCN", "THETA_SCN", "TAU_SCN")
Y6_CLOCK_MIN_H = 0.5
Y6_SOM = ("TAU_SOM", "SOM_CLEAR_GAIN")
Y6_SOM_MIN_H = 0.5

#: constants that weight an edge whose other end is OUTSIDE the hypothalamus.  This gate
#: assembles the hypothalamus ALONE, so none of these edges is in the circuit it sweeps --
#: whether or not the structure at the other end has been written.  (When the sweep was
#: first declared, none of thalamus, valuation or neuromodulators existed; three of the
#: four now do, and `bs.*` still does not.  The bucket is unchanged because the reason
#: these constants are inert HERE never depended on that.)  Listed before the sweep runs,
#: so "inert because the edge leaves the structure" cannot be decided after seeing which
#: ones came back flat.
Y6_ORPHAN_WEIGHTED = {
    "W_LIGHT_SCN", "D_LIGHT_SCN",                 # thal.lgn.relay -> hyp.scn
    "W_SCN_LC", "W_LC_SCN", "D_CIRC",             # the circ loop, through nm.lc
    "W_VLPO_NM", "W_LH_ORX", "W_LH_VTA",          # the sleep switch and orexin, into nm.*
    "W_DRIVE_VAL",                                # the drives into val.nacc_shell
    "W_NTS_ARC", "W_NTS_MNPO", "D_VISCERAL",      # bs.nts -> hyp.arc / hyp.mnpo
    "W_PB_POA", "D_PB_POA",                       # bs.pb -> hyp.poa
    "VAL_GAIN", "VAL_GAIN_BASE", "VAL_GAIN_LO", "VAL_GAIN_HI",   # the gain Mod into val
}
#: constants this gate's two measurements cannot see even though the edge IS assembled.
#: `hyp.pvn` is a sink inside the hypothalamus -- its only efferents leave the structure --
#: and `N` is a unit count on populations whose projections are all row-normalised and
#: dense, so every unit sees the same input whatever n is.  Declared before the run.
Y6_UNMEASURED = {"TAU_PVN", "BETA_PVN", "THETA_PVN", "D_PVN", "W_SCN_PVN", "GC_TAU",
                 "GC_G", "N"}


# ---------------------------------------------------------------------------- helpers
def build(extra_pops=(), extra_projs=(), extra_mods=None, seed: int = 0) -> Circuit:
    """the hypothalamus alone, from its own declaration.

    Every circuit in this file goes through here, so no gate can quietly build a
    different module than the one being gated.  `extra_*` is how Y5 adds its stand-in
    valuation population, and it is the only thing in this file that adds a population.

    `mods()` is filtered to the populations present, BY THE SAME RULE `ibm.brain.collect`
    uses -- a `Mod` naming a structure that is not in this assembly is an orphan, not a
    crash.  Without it every gate here would die on the `hyp.arc => val.nacc_shell` gain
    edge, because this script assembles the hypothalamus alone.  `extra_mods` overrides
    the filtered list, which is how Y5b's severed arm cuts the modulatory arm as well as
    the projections.
    """
    pops = list(H.pops()) + list(extra_pops)
    have = {p.id for p in pops}
    mods = (list(H.mods()) if extra_mods is None else list(extra_mods))
    mods = [m for m in mods if m.src in have and m.dst in have]
    return Circuit(pops, list(H.internal()) + list(extra_projs), mods, seed=seed)


def roll(c: Circuit, dt: float, seconds: float, state=None, drive=None, record=None):
    st = state if state is not None else c.init_state(1, dt)
    return c.rollout(int(round(seconds / dt)), dt, state=st, drive=drive, record=record)


def trace(tr, pid: str) -> torch.Tensor:
    return tr[pid][0, :, 0]


def crossing_period(r: torch.Tensor, dt: float, skip: float = 0.35):
    """mean and sd of the interval between upward crossings of the trace's own mean.

    Robust where a spectral estimate is not: the soft-argmax `peak_frequency` is biased
    to the band centre on a flat spectrum and, even on a clean one, is limited by the
    Welch bin width -- measured in Y2's known-answer control, a true 24.000 h sine reads
    24.353 h at this resolution.  So the period is GATED on this and the spectral peak is
    reported beside the prominence.
    """
    r = r[int(len(r) * skip):]
    s = torch.sign(r - r.mean())
    idx = [i for i in range(1, len(s)) if s[i - 1] <= 0 and s[i] > 0]
    if len(idx) < 3:
        return None, None, len(idx)
    d = torch.tensor([(idx[i + 1] - idx[i]) * dt for i in range(len(idx) - 1)])
    return float(d.mean()), float(d.std(unbiased=False)), len(idx)


def band_peak(x: torch.Tensor, dt: float):
    """(peak_hz, prominence_decades, df).  ALWAYS returned together -- CLAUDE.md, a
    frequency without a prominence is not evidence."""
    fs = 1.0 / dt
    freqs, psd = SP.welch_psd(x, fs, nperseg=min(Y2_NPERSEG, x.shape[-1]))
    lo, hi = CIRCADIAN.band
    pk = float(SP.peak_frequency(psd, freqs, *Y2_SEARCH))
    prom = float(SP.peak_prominence(psd, freqs, lo, hi,
                                    fit_lo=Y2_FIT[0], fit_hi=Y2_FIT[1]))
    return pk, prom, float(freqs[1])


def fit_tau(r: torch.Tensor, dt: float, skip: int = 2):
    """the time constant of a first-order approach, by regressing dr/dt on r.

    `dr/dt = (r_inf - r)/tau` is linear in r with slope -1/tau, so the fit needs no
    estimate of the asymptote -- which matters, because a 60 h window is only 4 time
    constants and the last sample is not the asymptote.
    """
    y = (r[1:] - r[:-1]) / dt
    x = r[:-1]
    y, x = y[skip:], x[skip:]
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if float(sxx) <= 0:
        return None, None
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    if slope >= 0:
        return None, None
    r_inf = float(ym / (-slope) + xm)
    return -1.0 / slope, r_inf


def ten_ninety(r: torch.Tensor, dt: float):
    """the 10-90% transit time of a monotone trace, in seconds, measured on the span it
    actually covered rather than on an assumed 0-to-1."""
    lo, hi = float(r[0]), float(r[-1])
    if abs(hi - lo) < 1e-6:
        return None
    a, b = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    up = hi > lo
    ia = next((i for i, v in enumerate(r) if (v >= a if up else v <= a)), None)
    ib = next((i for i, v in enumerate(r) if (v >= b if up else v <= b)), None)
    if ia is None or ib is None:
        return None
    return (ib - ia) * dt


def bouts(vlpo: torch.Tensor, lh: torch.Tensor, dt: float):
    """(is_asleep, start, stop) per bout.  Sleep is a state the network is IN --
    `hyp.vlpo` winning the flip-flop -- not a flag anything sets."""
    a = (vlpo > lh).int().tolist()
    out, i = [], 0
    for k, g in itertools.groupby(a):
        n = len(list(g))
        out.append((bool(k), i, i + n))
        i += n
    return out


def const_table() -> dict:
    """every module-level numeric constant, by introspection.

    Not a hand-written list: a constant added to the module later must appear in the
    sweep without anyone remembering to add it here.
    """
    return {k: v for k, v in vars(H).items()
            if k.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool)}


# ---------------------------------------------------------------------------- Y0
def gate_bounded():
    worst_r, worst_a, cases, vanish = 0.0, 0.0, [], {}
    pops = {p.id: p for p in H.pops()}
    for dt in (1e-3, DT_SLOW, 300.0):
        c = build()
        vanish[str(dt)] = {"n_vanishing": len(c.vanishing_delays(dt)),
                           "of_n_delayed": sum(1 for p in c.projs if p.delay_s > 0),
                           "steps": c.delays_in_steps(dt)}
        seconds = 20.0 if dt < 1.0 else 6 * DAY
        drive_sets = [("none", {}),
                      ("all_+5", {p: 5.0 for p in pops}),
                      ("all_-5", {p: -5.0 for p in pops}),
                      ("declared", H.drives())]
        drive_sets += [(f"only_{p}_+5", {p: 5.0}) for p in pops]
        for name, d in drive_sets:
            tr, st = roll(c, dt, seconds, drive=d)
            br, ba = 0.0, 0.0
            for pid, v in tr.items():
                br = max(br, float((-v).clamp_min(0).max()),
                         float((v - 1).clamp_min(0).max()))
            for pid, s in st.items():
                if pid.startswith("_") or "a" not in s:
                    continue
                g = pops[pid].adapt_g
                ba = max(ba, float((-s["a"]).clamp_min(0).max()),
                         float((s["a"] - g).clamp_min(0).max()))
            worst_r, worst_a = max(worst_r, br), max(worst_a, ba)
            cases.append({"dt": dt, "drive": name, "worst_rate_excursion": br,
                          "worst_adapt_excursion": ba})
    return {"ok": worst_r <= 0.0 and worst_a <= 0.0,
            "worst_rate_excursion": worst_r,
            "worst_adapt_excursion_beyond_adapt_g": worst_a,
            "vanishing_delays": {k: v["n_vanishing"] for k, v in vanish.items()},
            "n_delayed_projections": vanish[str(DT_SLOW)]["of_n_delayed"],
            "delays_in_steps_at_dt_60": vanish[str(DT_SLOW)]["steps"],
            "cases": cases}


# ---------------------------------------------------------------------------- Y1
def _same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and bool(torch.equal(a, b))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def gate_idempotent():
    c = build()
    d = H.drives()
    # (b) and (c) run at dt = 1e-3 ON PURPOSE.  At dt = 60 s every delay in the module
    # rounds to zero steps, `init_state` allocates no ring at all, and the two forms that
    # exist to catch a ring mutated in place would be testing a code path that is not
    # taken.  Both dts are reported.
    dts = (1e-3, DT_SLOW)

    # (a) the same rollout twice, at both timesteps
    a_ok = True
    for dt in dts:
        t1, _ = roll(c, dt, 2 * DAY if dt > 1.0 else 2.0, drive=d)
        t2, _ = roll(c, dt, 2 * DAY if dt > 1.0 else 2.0, drive=d)
        a_ok = a_ok and _same(t1, t2)

    b_ok, b_rings, c_ok, c_rings, n_rings = True, True, True, True, {}
    for dt in dts:
        st = c.init_state(1, dt)
        n_rings[str(dt)] = len(st["_rings"])
        for _ in range(50):
            st = c.step(st, dt, drive=d)
        # (b) step called twice on the SAME state -- the general form of the trap
        s1 = c.step(st, dt, drive=d)
        s2 = c.step(st, dt, drive=d)
        b_ok = b_ok and _same({k: v for k, v in s1.items() if not k.startswith("_")},
                              {k: v for k, v in s2.items() if not k.startswith("_")})
        b_rings = b_rings and _same(s1["_rings"], s2["_rings"])
        # (c) the same under no_grad, which is a DIFFERENT branch in Circuit.step
        stn = c.init_state(1, dt)
        with torch.no_grad():
            for _ in range(50):
                stn = c.step(stn, dt, drive=d)
            n1 = c.step(stn, dt, drive=d)
            n2 = c.step(stn, dt, drive=d)
        c_ok = c_ok and _same({k: v for k, v in n1.items() if not k.startswith("_")},
                              {k: v for k, v in n2.items() if not k.startswith("_")})
        c_rings = c_rings and _same(n1["_rings"], n2["_rings"])

    # (e) BRANCHING.  `ibm/circuit.py`'s own docstring says the ring buffers are "carried
    # in the state rather than on the module, so two rollouts from the same state cannot
    # interfere".  Under `no_grad` the ring is NOT cloned, so that claim is worth
    # measuring rather than believing: branch A from a state, then branch B from the same
    # state with a different drive, then re-run A.  A must come back identical.
    e_ok, e_detail = True, {}
    for dt, secs in ((1e-3, 0.3), (DT_SLOW, 6 * 3600.0)):
        st0 = c.init_state(1, dt)
        for _ in range(60):
            st0 = c.step(st0, dt, drive=d)
        d2 = {k: v + 0.5 for k, v in d.items()}
        with torch.no_grad():
            a1, _ = roll(c, dt, secs, state=st0, drive=d)
            _b, _ = roll(c, dt, secs, state=st0, drive=d2)
            a2, _ = roll(c, dt, secs, state=st0, drive=d)
        same = _same(a1, a2)
        e_detail[str(dt)] = {"identical": bool(same),
                             "rings": len(st0["_rings"]),
                             "max_abs_diff": max(float((a1[k] - a2[k]).abs().max())
                                                 for k in a1)}
        e_ok = e_ok and same
    # the same three rollouts with grad ENABLED, which is the branch that clones
    st0 = c.init_state(1, 1e-3)
    for _ in range(60):
        st0 = c.step(st0, 1e-3, drive=d)
    d2 = {k: v + 0.5 for k, v in d.items()}
    g1, _ = roll(c, 1e-3, 0.3, state=st0, drive=d)
    _g, _ = roll(c, 1e-3, 0.3, state=st0, drive=d2)
    g2, _ = roll(c, 1e-3, 0.3, state=st0, drive=d)
    e_detail["0.001_grad_enabled"] = {"identical": bool(_same(g1, g2)),
                                      "rings": len(st0["_rings"]),
                                      "max_abs_diff": max(
                                          float((g1[k] - g2[k]).abs().max()) for k in g1)}

    # (d) the declaration is a function of nothing and must not drift
    decl_ok = all(_same([x.__dict__ for x in f()], [x.__dict__ for x in f()])
                  for f in (H.pops, H.internal, H.external, H.mods))
    decl_ok = decl_ok and _same(H.drives(), H.drives())
    w1, w2 = build(seed=0)._W, build(seed=0)._W
    w_ok = _same(w1, w2)

    return {"ok": bool(a_ok and b_ok and e_ok and decl_ok and w_ok),
            "a_rollout_twice": bool(a_ok),
            "b_step_twice_same_state": bool(b_ok),
            "b_rings_equal": bool(b_rings),
            "c_step_twice_under_no_grad": bool(c_ok),
            "c_rings_equal_under_no_grad": bool(c_rings),
            "e_branch_rerun_identical_under_no_grad": bool(e_ok),
            "e_detail": e_detail,
            "e_finding": "ibm/circuit.py's Circuit docstring claims two rollouts from one "
                         "state cannot interfere.  Under torch.no_grad() they DO: "
                         "Circuit.step writes the delay ring in place unless grad is "
                         "enabled, and the ring tensors are shared by every state derived "
                         "from the branch point.  Measured at dt = 1e-3 with 8 rings "
                         "allocated; clean with grad enabled, and clean at dt = 60 s only "
                         "because no delay in this module survives that timestep so no "
                         "ring exists to share.  Not fixed here: ibm/circuit.py is out of "
                         "scope for this task.  Callers must not branch a rollout under "
                         "no_grad from a shared state.",
            "rings_allocated_per_dt": n_rings,
            "c_note": "Circuit.step clones a delay ring only when grad is enabled; under "
                      "no_grad it writes IN PLACE into the ring it was handed.  That is "
                      "harmless for (b) -- the write goes to the slot indexed by the "
                      "state's own _t with a value computed from that same state, so a "
                      "repeat call writes the same slot with the same value -- and (e) is "
                      "the form that could actually catch it: two rollouts branching from "
                      "one state with different drives, then the first re-run.  ibm/"
                      "circuit.py claims in its own docstring that this cannot interfere; "
                      "(e) measures the claim.",
            "d_declaration_stable": bool(decl_ok),
            "d_weights_stable_at_seed_0": bool(w_ok)}


# ---------------------------------------------------------------------------- Y2
def gate_free_run():
    c = build()
    # the gate's own precondition: "no input" must be true, not assumed
    targets_scn = [p.key for p in c.projs if p.dst == "hyp.scn"]
    scn_driven = "hyp.scn" in H.drives()
    pre_ok = (not targets_scn) and (not scn_driven)

    tr, _ = roll(c, DT_SLOW, Y2_DAYS * DAY, drive=H.drives())
    x = trace(tr, "hyp.scn")
    per, sd, n = crossing_period(x, DT_SLOW)
    pk, prom, df = band_peak(x, DT_SLOW)

    # known answers, through the identical estimator
    t = torch.arange(len(x), dtype=torch.float) * DT_SLOW
    ka = {}
    for name, sig in (("sine_24h", 0.5 + 0.4 * torch.sin(2 * math.pi * t / DAY)),
                      ("white_noise", torch.randn(len(x),
                       generator=torch.Generator().manual_seed(7))),
                      ("constant", torch.full((len(x),), 0.5))):
        p_, pr_, _ = band_peak(sig, DT_SLOW)
        ka[name] = {"peak_hz": p_, "peak_period_h": 1.0 / p_ / 3600.0,
                    "prominence": pr_}
    ka_ok = (ka["sine_24h"]["prominence"] > 0
             and ka["white_noise"]["prominence"] <= 0
             and abs(ka["constant"]["prominence"]) < 1e-6)
    bias = ka["sine_24h"]["peak_period_h"] / 24.0 - 1.0

    ok = bool(pre_ok and ka_ok and per is not None and prom > 0
              and Y2_PERIOD_H[0] <= per / 3600.0 <= Y2_PERIOD_H[1]
              and sd / 3600.0 < Y2_SD_H)
    return {"ok": ok,
            "precondition_no_input": pre_ok,
            "projections_into_hyp_scn": targets_scn,
            "hyp_scn_in_drives": scn_driven,
            "dt_s": DT_SLOW, "days": Y2_DAYS,
            "period_h": None if per is None else per / 3600.0,
            "period_sd_h": None if sd is None else sd / 3600.0,
            "n_cycles": n,
            "declared_window_h": list(Y2_PERIOD_H),
            "peak_hz": pk, "peak_period_h": 1.0 / pk / 3600.0,
            "peak_prominence_decades": prom,
            "df_hz": df, "band_hz": list(CIRCADIAN.band),
            "known_answers": ka, "known_answers_ok": ka_ok,
            "spectral_period_bias_from_known_sine": bias,
            "note": "the period is gated on the crossing interval; the spectral peak is "
                    "quoted beside the prominence and carries the bias above, which the "
                    "24.000 h known-answer sine measures on the same grid"}


Y2B_NULL_DRAWS = 16
Y2B_NULL_SEED = 20260918
Y2B_NULL_SD = 5.0


def gate_free_run_b():
    """the replacement instrument for Y2.  Y2 stays FAILED; the thresholds are the same."""
    c = build()
    # the precondition Y2 meant to test: does anything OTHER than the clock's own loop
    # reach hyp.scn?  A recurrent self-projection is the oscillator, not an input.
    foreign = [p.key for p in c.projs if p.dst == "hyp.scn" and p.src != "hyp.scn"]
    self_loops = [p.key for p in c.projs if p.dst == "hyp.scn" and p.src == "hyp.scn"]
    scn_driven = "hyp.scn" in H.drives()
    pre_ok = (not foreign) and (not scn_driven)

    tr, _ = roll(c, DT_SLOW, Y2_DAYS * DAY, drive=H.drives())
    x = trace(tr, "hyp.scn")
    per, sd, n = crossing_period(x, DT_SLOW)
    pk, prom, df = band_peak(x, DT_SLOW)

    t = torch.arange(len(x), dtype=torch.float) * DT_SLOW
    sine = 0.5 + 0.4 * torch.sin(2 * math.pi * t / DAY)
    p_s, pr_s, _ = band_peak(sine, DT_SLOW)
    p_c, pr_c, _ = band_peak(torch.full((len(x),), 0.5), DT_SLOW)
    g = torch.Generator().manual_seed(Y2B_NULL_SEED)
    null = [band_peak(torch.randn(len(x), generator=g), DT_SLOW)[1]
            for _ in range(Y2B_NULL_DRAWS)]
    nt = torch.tensor(null)
    n_mean, n_sd = float(nt.mean()), float(nt.std(unbiased=True))
    z = (prom - n_mean) / max(n_sd, 1e-9)

    ctrl_ok = pr_s > 0 and abs(pr_c) < 1e-6
    ok = bool(pre_ok and ctrl_ok and per is not None
              and Y2_PERIOD_H[0] <= per / 3600.0 <= Y2_PERIOD_H[1]
              and sd / 3600.0 < Y2_SD_H and z > Y2B_NULL_SD)
    return {"ok": ok,
            "precondition_no_foreign_input": pre_ok,
            "foreign_projections_into_hyp_scn": foreign,
            "self_projections_into_hyp_scn": self_loops,
            "hyp_scn_in_drives": scn_driven,
            "dt_s": DT_SLOW, "days": Y2_DAYS,
            "period_h": None if per is None else per / 3600.0,
            "period_sd_h": None if sd is None else sd / 3600.0,
            "n_cycles": n, "declared_window_h": list(Y2_PERIOD_H),
            "peak_hz": pk, "peak_period_h": 1.0 / pk / 3600.0,
            "peak_prominence_decades": prom,
            "df_hz": df, "band_hz": list(CIRCADIAN.band),
            "null_prominence_mean": n_mean, "null_prominence_sd": n_sd,
            "null_draws": Y2B_NULL_DRAWS, "prominence_z_vs_null": z,
            "declared_null_sd": Y2B_NULL_SD,
            "known_answer_sine_24h": {"peak_period_h": 1.0 / p_s / 3600.0,
                                      "prominence": pr_s},
            "known_answer_constant": {"peak_period_h": 1.0 / p_c / 3600.0,
                                      "prominence": pr_c},
            "controls_ok": bool(ctrl_ok),
            "spectral_period_bias_from_known_sine": 1.0 / p_s / 3600.0 / 24.0 - 1.0,
            "note": "the period is GATED on the crossing interval.  the spectral peak "
                    "carries the bias above -- a true 24.000 h sine reads high on this "
                    "grid -- so the peak is quoted beside the prominence and the "
                    "prominence is judged against an ensemble null, never against zero"}


# ---------------------------------------------------------------------------- Y3
def _light(period_s: float):
    amp = H.W_LIGHT_SCN * LIGHT_IRRADIANCE
    base = H.drives()

    def fn(i):
        t = i * DT_SLOW
        d = dict(base)
        d["hyp.scn"] = amp if (t % period_s) < period_s / 2.0 else 0.0
        return d
    return fn


def gate_entrain():
    c = build()
    free_tr, _ = roll(c, DT_SLOW, Y2_DAYS * DAY, drive=H.drives())
    free, _, _ = crossing_period(trace(free_tr, "hyp.scn"), DT_SLOW)
    free_h = free / 3600.0

    rows = {}
    for T in tuple(Y3_LOCK) + tuple(Y3_FAIL) + tuple(Y3_REPORT_ONLY):
        tr, _ = roll(build(), DT_SLOW, Y3_DAYS * DAY, drive=_light(T * 3600.0))
        p, sd, n = crossing_period(trace(tr, "hyp.scn"), DT_SLOW)
        ph = None if p is None else p / 3600.0
        rows[f"{T:g}h"] = {"T_light_h": T, "measured_h": ph,
                           "sd_h": None if sd is None else sd / 3600.0,
                           "err_h": None if ph is None else abs(ph - T),
                           "offset_from_free_h": T - free_h}
    locked = {k: (rows[k]["err_h"] is not None and rows[k]["err_h"] <= Y3_LOCK_TOL_H)
              for k in rows}
    must_lock = all(locked[f"{T:g}h"] for T in Y3_LOCK)
    must_fail = all(rows[f"{T:g}h"]["err_h"] is not None
                    and rows[f"{T:g}h"]["err_h"] > Y3_FAIL_TOL_H for T in Y3_FAIL)
    return {"ok": bool(must_lock and must_fail),
            "free_run_h": free_h,
            "light_amplitude": H.W_LIGHT_SCN * LIGHT_IRRADIANCE,
            "must_lock_h": list(Y3_LOCK), "must_fail_h": list(Y3_FAIL),
            "report_only_h": list(Y3_REPORT_ONLY),
            "all_locked": must_lock, "all_failed_outside": must_fail,
            "locked": locked, "rows": rows,
            "entrainment_range_h": [min(Y3_LOCK) - free_h, max(Y3_LOCK) - free_h],
            "note": "16 h is reported and not gated: a relaxation oscillator "
                    "frequency-demultiplies and locks 3:2, which is neither entrainment "
                    "to 16 h nor a failure to be pulled"}


# ---------------------------------------------------------------------------- Y4
def _clamped(dt: float, wake_h: float, sleep_h: float):
    base = H.drives()
    wake = dict(base); wake["hyp.lh"] = base["hyp.lh"] + Y4_CLAMP
    wake["hyp.vlpo"] = -Y4_CLAMP
    sleep = dict(base); sleep["hyp.lh"] = base["hyp.lh"] - Y4_CLAMP
    sleep["hyp.vlpo"] = Y4_CLAMP
    c = build()
    tw, st = roll(c, dt, wake_h * 3600.0, drive=wake)
    ts, _ = roll(c, dt, sleep_h * 3600.0, state=st, drive=sleep)
    return tw, ts


def _som_taus(dt: float, wake_h: float, sleep_h: float) -> dict:
    tw, ts = _clamped(dt, wake_h, sleep_h)
    rise, fall = trace(tw, "hyp.som"), trace(ts, "hyp.som")
    tau_r, inf_r = fit_tau(rise, dt)
    tau_f, inf_f = fit_tau(fall, dt)
    vlpo_sleep = float(trace(ts, "hyp.vlpo").mean())
    vlpo_wake = float(trace(tw, "hyp.vlpo").mean())
    f_sleep = min(max(1.0 + H.SOM_CLEAR_GAIN * vlpo_sleep, H.SOM_CLEAR_LO), H.SOM_CLEAR_HI)
    f_wake = min(max(1.0 + H.SOM_CLEAR_GAIN * vlpo_wake, H.SOM_CLEAR_LO), H.SOM_CLEAR_HI)
    return {"tau_rise_h": None if tau_r is None else tau_r / 3600.0,
            "tau_fall_h": None if tau_f is None else tau_f / 3600.0,
            "t10_90_rise_h": (lambda v: None if v is None else v / 3600.0)(
                ten_ninety(rise, dt)),
            "t90_10_fall_h": (lambda v: None if v is None else v / 3600.0)(
                ten_ninety(fall, dt)),
            "som_start": float(rise[0]), "som_peak": float(rise[-1]),
            "som_end": float(fall[-1]),
            "asymptote_rise": inf_r, "asymptote_fall": inf_f,
            "mean_vlpo_wake": vlpo_wake, "mean_vlpo_sleep": vlpo_sleep,
            "predicted_tau_rise_h": H.TAU_SOM * f_wake / 3600.0,
            "predicted_tau_fall_h": H.TAU_SOM * f_sleep / 3600.0}


def gate_sleep_pressure():
    # (a) spontaneous
    c = build()
    tr, _ = roll(c, DT_SLOW, Y4_DAYS * DAY, drive=H.drives())
    vl, lh, so = trace(tr, "hyp.vlpo"), trace(tr, "hyp.lh"), trace(tr, "hyp.som")
    bs = bouts(vl, lh, DT_SLOW)[1:-1]          # drop the partial bouts at both ends
    rows, bad = [], 0
    for asleep, i, j in bs:
        d = float(so[j - 1] - so[i])
        rows.append({"state": "sleep" if asleep else "wake",
                     "hours": (j - i) * DT_SLOW / 3600.0,
                     "som_start": float(so[i]), "som_end": float(so[j - 1]),
                     "delta": d})
        if (asleep and d >= 0) or ((not asleep) and d <= 0):
            bad += 1
    wake_b = [r["hours"] for r in rows if r["state"] == "wake"]
    sleep_b = [r["hours"] for r in rows if r["state"] == "sleep"]
    spont_ok = bad == 0 and len(wake_b) >= 5 and len(sleep_b) >= 5

    # (b) clamped
    cl = _som_taus(DT_SLOW, Y4_WAKE_H, Y4_SLEEP_H)
    tr_ok = (cl["tau_rise_h"] is not None
             and abs(cl["tau_rise_h"] - cl["predicted_tau_rise_h"])
             <= Y4_TAU_TOL * cl["predicted_tau_rise_h"])
    tf_ok = (cl["tau_fall_h"] is not None
             and abs(cl["tau_fall_h"] - cl["predicted_tau_fall_h"])
             <= Y4_TAU_TOL * cl["predicted_tau_fall_h"])
    asym_ok = (cl["tau_rise_h"] or 0) > (cl["tau_fall_h"] or 1e9)

    return {"ok": bool(spont_ok and tr_ok and tf_ok and asym_ok),
            "spontaneous_ok": spont_ok, "bouts_wrong_way": bad,
            "n_wake_bouts": len(wake_b), "n_sleep_bouts": len(sleep_b),
            "mean_wake_bout_h": sum(wake_b) / max(1, len(wake_b)),
            "mean_sleep_bout_h": sum(sleep_b) / max(1, len(sleep_b)),
            "sleep_fraction": float((vl > lh).float().mean()),
            "som_range": [float(so.min()), float(so.max())],
            "clamped": cl,
            "tau_rise_ok": tr_ok, "tau_fall_ok": tf_ok, "rise_slower_than_fall": asym_ok,
            "declared_TAU_SOM_h": H.TAU_SOM / 3600.0,
            "rise_over_fall": (cl["tau_rise_h"] / cl["tau_fall_h"]
                               if cl["tau_fall_h"] else None),
            "literature_note": "human process-S fits give ~18 h rise against ~4 h decay, "
                               "a ratio of 4.3; this module declares a smaller ratio and "
                               "the measured one is reported above",
            "bouts": rows}


# ---------------------------------------------------------------------------- Y5
VAL_ID = "val.nacc_shell"


def _val_pop(sparsity=None):
    """the STAND-IN.  It exists in this gate script and nowhere else, because
    ibm/brain/valuation.py has not been written.  Said plainly rather than faked.

    `sparsity` is what gives it the engine's divisive normalisation.  Y5 ran with None
    and pinned the population at 1.0; `docs/BRAIN_SPEC.md` says every population is
    sparse and divisively normalised and that this is not a tuning choice, so a stand-in
    without it is not standing in for the thing it names.
    """
    return Pop(id=VAL_ID, n=64, kind="E", tau=0.05, beta=8.0, theta=0.30,
               sparsity=sparsity,
               note="STAND-IN for ibm/brain/valuation.py's nucleus accumbens shell")


def _val_edges():
    """read off the MODULE's declaration, not re-invented here."""
    return [p for p in H.external() if p.dst == VAL_ID]


def _val_mods():
    """and the modulatory arm, likewise.  The module reaches the valuation system two
    ways -- a projection that ADDS drive and a `Mod` that MULTIPLIES gain_in -- and the
    severed control has to cut both.  A control that cuts one arm of a two-arm pathway
    would leave a difference behind and be read as a bookkeeping error."""
    return [m for m in H.mods() if m.dst == VAL_ID]


def _val_arm(hungry: bool, severed: bool, sparsity=None):
    edges = [] if severed else _val_edges()
    mods = [m for m in H.mods() if m.dst != VAL_ID] if severed else None
    c = build(extra_pops=[_val_pop(sparsity)], extra_projs=edges, extra_mods=mods)
    d = dict(H.drives())
    if not hungry:
        d["hyp.arc"] = d["hyp.arc"] - H.W_NTS_ARC * Y5_NTS_SATED
    _, st = roll(c, DT_SLOW, Y5_SETTLE_H * 3600.0, drive=d)
    # this arm BRANCHES two rollouts from `st` (cue and no-cue), which is exactly the
    # pattern Y1(e) found is unsafe under no_grad when a delay ring exists.  At dt = 60 s
    # no delay in this module survives, so no ring exists to share -- asserted, not
    # assumed, because the assumption is the whole hazard.
    assert not st["_rings"], "Y5 branches from one state; a shared ring would corrupt it"
    arc = float(st["hyp.arc"]["r"].mean())
    lh = float(st["hyp.lh"]["r"].mean())
    dc = dict(d); dc[VAL_ID] = Y5_CUE
    tr, _ = roll(c, DT_SLOW, Y5_CUE_H * 3600.0, state=st, drive=dc)
    v = trace(tr, VAL_ID)
    k = max(1, int(Y5_READ_H * 3600.0 / DT_SLOW))
    tr0, _ = roll(c, DT_SLOW, Y5_CUE_H * 3600.0, state=st, drive=d)
    v0 = trace(tr0, VAL_ID)
    return {"hyp.arc": arc, "hyp.lh": lh,
            "response": float(v[-k:].mean()),
            "no_cue_baseline": float(v0[-k:].mean()),
            "cue_evoked": float(v[-k:].mean() - v0[-k:].mean()),
            "max_rate": float(v[-k:].max()),
            "active_fraction": float((v[-k:] > 0.2).float().mean()),
            "w_inh": float(c.w_inh.get(VAL_ID, 0.0)),
            "n_edges": len(edges)}


def gate_drive_changes_value():
    arms = {f"{'hungry' if h else 'sated'}_{'severed' if s else 'intact'}":
            _val_arm(h, s) for h in (True, False) for s in (False, True)}
    intact = arms["hungry_intact"]["response"] - arms["sated_intact"]["response"]
    severed = arms["hungry_severed"]["response"] - arms["sated_severed"]["response"]
    ok = bool(intact > Y5_MIN_EFFECT and abs(severed) < Y5_MAX_SEVERED)
    return {"ok": ok,
            "valuation": _val_id_in_valuation(),
            "stand_in": VAL_ID,
            "declared_edges": [p.key for p in _val_edges()],
            "declared_mods": [f"{m.src}=>{m.dst}:{m.param}" for m in _val_mods()],
            "cue": Y5_CUE,
            "intact_difference": intact, "severed_difference": severed,
            "declared_min_effect": Y5_MIN_EFFECT,
            "declared_max_severed": Y5_MAX_SEVERED,
            "arms": arms,
            "note": "ibm/brain/valuation.py does not exist.  val.nacc_shell here is a "
                    "stand-in built inside this script; the edges into it are read off "
                    "hypothalamus.external() rather than written here, so what is "
                    "measured is the module's own declaration.  The severed arm is the "
                    "control: gate on an ablation, never on a magnitude."}


#: the stand-in sparsities Y5b runs at.  None is Y5's, kept so the ceiling it hit is in
#: the same table as the values that do not hit it; 0.06-0.12 is the range
#: `ibm/brain/cortex.py` declares across its hierarchy and the only declared sparsity
#: range in this package; 0.20 is beyond it, to show the direction does not turn over.
Y5B_SPARSITIES = (None, 0.06, 0.08, 0.10, 0.12, 0.20)


def _val_id_in_valuation():
    """does ibm/brain/valuation.py declare a population of exactly VAL_ID?

    A cross-structure edge is written by one author and owned by another, so a naming
    disagreement reads exactly like a structure nobody has built.  This turns the
    question into a number in the record.
    """
    try:
        import ibm.brain as B
        if "valuation" not in B.available():
            return {"module_present": False, "id_declared": None}
        ids = [p.id for p in B.load("valuation").pops()]
        return {"module_present": True, "id_declared": VAL_ID in ids,
                "valuation_pops": ids}
    except Exception as e:                                  # noqa: BLE001
        return {"module_present": None, "error": f"{type(e).__name__}: {e}"}


def gate_drive_changes_value_b():
    """the replacement instrument for Y5.  Y5 stays FAILED.  Gated on the SIGN and on the
    ablation, at six stand-in sparsities, and on no magnitude anywhere."""
    rows = {}
    for sp in Y5B_SPARSITIES:
        arms = {f"{'hungry' if h else 'sated'}_{'severed' if s else 'intact'}":
                _val_arm(h, s, sparsity=sp) for h in (True, False) for s in (False, True)}
        intact = arms["hungry_intact"]["response"] - arms["sated_intact"]["response"]
        severed = arms["hungry_severed"]["response"] - arms["sated_severed"]["response"]
        rows[str(sp)] = {
            "sparsity": sp, "intact_difference": intact, "severed_difference": severed,
            "sign_ok": intact > 0.0, "ablation_ok": severed == 0.0,
            "saturated_intact": arms["hungry_intact"]["max_rate"] >= 0.99,
            "clears_Y5_bar_0.05": intact > Y5_MIN_EFFECT,
            "arms": arms}
    sign_ok = all(r["sign_ok"] for r in rows.values())
    abl_ok = all(r["ablation_ok"] for r in rows.values())
    return {"ok": bool(sign_ok and abl_ok),
            "valuation": _val_id_in_valuation(),
            "stand_in": VAL_ID,
            "declared_edges": [p.key for p in _val_edges()],
            "declared_mods": [f"{m.src}=>{m.dst}:{m.param}" for m in _val_mods()],
            "cue": Y5_CUE, "sparsities": [str(x) for x in Y5B_SPARSITIES],
            "sign_positive_everywhere": sign_ok,
            "severed_exactly_zero_everywhere": abl_ok,
            "intact_difference": {k: v["intact_difference"] for k, v in rows.items()},
            "severed_difference": {k: v["severed_difference"] for k, v in rows.items()},
            "saturated": {k: v["saturated_intact"] for k, v in rows.items()},
            "clears_Y5_bar": {k: v["clears_Y5_bar_0.05"] for k, v in rows.items()},
            "rows": rows,
            "note": "ibm/brain/valuation.py does not exist.  val.nacc_shell is a stand-in "
                    "built inside this script and nowhere else; the edges into it are "
                    "read off hypothalamus.external() rather than written here, so what "
                    "is measured is the module's own declaration.  The magnitude is "
                    "reported at every sparsity and gated at none, because Y5's diagnosis "
                    "showed it moves with the instrument."}


# ---------------------------------------------------------------------------- Y6
def _sweep_measure() -> dict:
    c = build()
    tr, _ = roll(c, DT_SWEEP, Y6_DAYS * DAY, drive=H.drives(), record=["hyp.scn"])
    p, sd, n = crossing_period(trace(tr, "hyp.scn"), DT_SWEEP)
    som = _som_taus(DT_SWEEP, Y6_WAKE_H, Y6_SLEEP_H)
    return {"period_h": None if p is None else p / 3600.0, "n_cycles": n,
            "tau_rise_h": som["tau_rise_h"], "tau_fall_h": som["tau_fall_h"]}


def gate_sensitivity():
    consts = const_table()
    base = _sweep_measure()
    rows, inert = [], []
    for k in sorted(consts):
        v0 = consts[k]
        ends = {}
        for tag, f in (("x0.5", 0.5), ("x1.5", 1.5)):
            nv = type(v0)(v0 * f) if not isinstance(v0, int) else max(1, int(round(v0 * f)))
            setattr(H, k, nv)
            try:
                ends[tag] = _sweep_measure()
                ends[tag]["value"] = nv
            finally:
                setattr(H, k, v0)

        def span(field):
            a, b = ends["x0.5"][field], ends["x1.5"][field]
            if a is None or b is None:
                return None           # an end where the oscillator stopped
            return abs(a - b)
        sp, sr, sf = span("period_h"), span("tau_rise_h"), span("tau_fall_h")
        dead = [t for t in ends if ends[t]["period_h"] is None]
        row = {"const": k, "baseline": v0,
               "period_h": [ends["x0.5"]["period_h"], ends["x1.5"]["period_h"]],
               "tau_rise_h": [ends["x0.5"]["tau_rise_h"], ends["x1.5"]["tau_rise_h"]],
               "tau_fall_h": [ends["x0.5"]["tau_fall_h"], ends["x1.5"]["tau_fall_h"]],
               "span_period_h": sp, "span_tau_rise_h": sr, "span_tau_fall_h": sf,
               "oscillation_lost_at": dead}
        is_inert = (not dead
                    and (sp is not None and sp < Y6_INERT_PERIOD_H)
                    and (sr is None or sr < Y6_INERT_TAU_H)
                    and (sf is None or sf < Y6_INERT_TAU_H))
        row["inert"] = is_inert
        if is_inert:
            row["inert_kind"] = ("orphan" if k in Y6_ORPHAN_WEIGHTED
                                 else "unmeasured" if k in Y6_UNMEASURED
                                 else "unexplained")
            inert.append(row["const"])
        rows.append(row)
        print(f"      {k:18s} P {str(row['period_h'][0])[:6]:>6s} -> "
              f"{str(row['period_h'][1])[:6]:>6s}  spanP "
              f"{'   n/a' if sp is None else f'{sp:6.3f}'}  spanRise "
              f"{'   n/a' if sr is None else f'{sr:6.3f}'}  spanFall "
              f"{'   n/a' if sf is None else f'{sf:6.3f}'}"
              f"{'  [' + row.get('inert_kind', '') + ']' if is_inert else ''}", flush=True)

    by = {r["const"]: r for r in rows}
    clock_ok = {k: (by[k]["oscillation_lost_at"] != []
                    or (by[k]["span_period_h"] or 0.0) > Y6_CLOCK_MIN_H)
                for k in Y6_CLOCK}
    som_ok = {k: max((by[k]["span_tau_rise_h"] or 0.0),
                     (by[k]["span_tau_fall_h"] or 0.0)) > Y6_SOM_MIN_H for k in Y6_SOM}
    kinds = {kind: [r["const"] for r in rows if r.get("inert_kind") == kind]
             for kind in ("orphan", "unmeasured", "unexplained")}
    ok = all(clock_ok.values()) and all(som_ok.values()) and not kinds["unexplained"]
    return {"ok": bool(ok), "n_constants": len(consts), "baseline": base,
            "dt_s": DT_SWEEP,
            "clock_constants_live": clock_ok, "somnogen_constants_live": som_ok,
            "inert": inert, "inert_by_kind": kinds,
            "sweep": rows}


Y6B_STATE_DAYS = 6
Y6B_STATE_SKIP = 2
Y6B_INERT_STATE = 0.005
Y6B_VANISHES = {"D_SLEEP_SWITCH"}
Y6B_CLAMPS = {"SOM_CLEAR_LO", "SOM_CLEAR_HI", "PROCESS_C_LO", "PROCESS_C_HI"}
Y6B_UNMEASURED = {"N"}


def _state_vector() -> dict:
    """mean rate of every population plus the sleep fraction, over an UNCLAMPED run.

    This is the quantity Y6 did not have.  Its two measurements clamp the sleep switch or
    ignore everything but the clock, so neither can see the drive pathway; this one can,
    because the drives set where the whole module sits.
    """
    c = build()
    tr, _ = roll(c, DT_SWEEP, Y6B_STATE_DAYS * DAY, drive=H.drives())
    k = int(Y6B_STATE_SKIP * DAY / DT_SWEEP)
    out = {pid: float(v[0, k:, :].mean()) for pid, v in tr.items()}
    out["_sleep_fraction"] = float((trace(tr, "hyp.vlpo")[k:]
                                   > trace(tr, "hyp.lh")[k:]).float().mean())
    return out


def _clamp_binds() -> dict:
    """does each declared `Mod` clamp actually bind at baseline?  Measured on an unclamped
    run, so that a guard doing nothing is distinguishable from one holding a parameter."""
    c = build()
    tr, _ = roll(c, DT_SWEEP, Y6B_STATE_DAYS * DAY, drive=H.drives())
    k = int(Y6B_STATE_SKIP * DAY / DT_SWEEP)
    vl = trace(tr, "hyp.vlpo")[k:]
    sc = trace(tr, "hyp.scn")[k:]
    f_som = 1.0 + H.SOM_CLEAR_GAIN * vl
    f_c = 1.0 + H.PROCESS_C_GAIN * (sc - H.PROCESS_C_BASE)
    return {"SOM_CLEAR_LO": float((f_som < H.SOM_CLEAR_LO).float().mean()),
            "SOM_CLEAR_HI": float((f_som > H.SOM_CLEAR_HI).float().mean()),
            "PROCESS_C_LO": float((f_c < H.PROCESS_C_LO).float().mean()),
            "PROCESS_C_HI": float((f_c > H.PROCESS_C_HI).float().mean())}


def _sweep_measure_b() -> dict:
    m = _sweep_measure()
    m["state"] = _state_vector()
    return m


def gate_sensitivity_b():
    consts = const_table()
    base = _sweep_measure_b()
    binds = _clamp_binds()
    rows, inert = [], []
    for k in sorted(consts):
        v0 = consts[k]
        ends = {}
        for tag, f in (("x0.5", 0.5), ("x1.5", 1.5)):
            nv = type(v0)(v0 * f) if not isinstance(v0, int) else max(1, int(round(v0 * f)))
            setattr(H, k, nv)
            try:
                ends[tag] = _sweep_measure_b()
                ends[tag]["value"] = nv
            finally:
                setattr(H, k, v0)

        def span(field):
            a, b = ends["x0.5"][field], ends["x1.5"][field]
            return None if (a is None or b is None) else abs(a - b)
        sa, sb = ends["x0.5"]["state"], ends["x1.5"]["state"]
        ss = max(abs(sa[q] - sb[q]) for q in sa)
        worst = max(sa, key=lambda q: abs(sa[q] - sb[q]))
        sp, sr, sf = span("period_h"), span("tau_rise_h"), span("tau_fall_h")
        dead = [t for t in ends if ends[t]["period_h"] is None]
        row = {"const": k, "baseline": v0,
               "period_h": [ends["x0.5"]["period_h"], ends["x1.5"]["period_h"]],
               "tau_rise_h": [ends["x0.5"]["tau_rise_h"], ends["x1.5"]["tau_rise_h"]],
               "tau_fall_h": [ends["x0.5"]["tau_fall_h"], ends["x1.5"]["tau_fall_h"]],
               "span_period_h": sp, "span_tau_rise_h": sr, "span_tau_fall_h": sf,
               "span_state": ss, "span_state_worst_on": worst,
               "oscillation_lost_at": dead}
        is_inert = (not dead
                    and (sp is not None and sp < Y6_INERT_PERIOD_H)
                    and (sr is None or sr < Y6_INERT_TAU_H)
                    and (sf is None or sf < Y6_INERT_TAU_H)
                    and ss < Y6B_INERT_STATE)
        row["inert"] = is_inert
        if is_inert:
            row["inert_kind"] = ("orphan" if k in Y6_ORPHAN_WEIGHTED
                                 else "vanishes_at_dt" if k in Y6B_VANISHES
                                 else "clamp" if k in Y6B_CLAMPS
                                 else "unmeasured" if k in Y6B_UNMEASURED
                                 else "unexplained")
            inert.append(k)
        rows.append(row)
        print(f"      {k:18s} spanP "
              f"{'   n/a' if sp is None else f'{sp:6.3f}'}  spanRise "
              f"{'   n/a' if sr is None else f'{sr:6.3f}'}  spanFall "
              f"{'   n/a' if sf is None else f'{sf:6.3f}'}  spanState {ss:7.4f} "
              f"({worst})"
              f"{'  [' + row.get('inert_kind', '') + ']' if is_inert else ''}", flush=True)

    by = {r["const"]: r for r in rows}
    clock_ok = {k: (by[k]["oscillation_lost_at"] != []
                    or (by[k]["span_period_h"] or 0.0) > Y6_CLOCK_MIN_H)
                for k in Y6_CLOCK}
    som_ok = {k: max((by[k]["span_tau_rise_h"] or 0.0),
                     (by[k]["span_tau_fall_h"] or 0.0)) > Y6_SOM_MIN_H for k in Y6_SOM}
    kinds = {kind: [r["const"] for r in rows if r.get("inert_kind") == kind]
             for kind in ("orphan", "vanishes_at_dt", "clamp", "unmeasured",
                          "unexplained")}
    ok = all(clock_ok.values()) and all(som_ok.values()) and not kinds["unexplained"]
    return {"ok": bool(ok), "n_constants": len(consts), "baseline": base,
            "dt_s": DT_SWEEP,
            "clock_constants_live": clock_ok, "somnogen_constants_live": som_ok,
            "clamp_binding_fraction_at_baseline": binds,
            "inert": inert, "inert_by_kind": kinds,
            "sweep": rows}


# ---------------------------------------------------------------------------- main
GATES = [("Y0_bounded", gate_bounded),
         ("Y1_idempotence", gate_idempotent),
         ("Y2_free_run", gate_free_run),
         ("Y2b_free_run_replacement", gate_free_run_b),
         ("Y3_entrainment", gate_entrain),
         ("Y4_sleep_pressure", gate_sleep_pressure),
         ("Y5_drive_changes_value", gate_drive_changes_value),
         ("Y5b_drive_changes_value_replacement", gate_drive_changes_value_b),
         ("Y6_sensitivity", gate_sensitivity),
         ("Y6b_sensitivity_replacement", gate_sensitivity_b)]

def _git_sha() -> str:
    import subprocess
    sha = os.environ.get("IBM_GIT_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.dirname(
                                           os.path.abspath(__file__))),
                                       text=True).strip()
    except Exception:                                       # noqa: BLE001
        return "git-unknown"


BULKY = ("cases", "sweep", "bouts", "arms", "known_answers", "rows",
         "delays_in_steps_at_dt_60", "clamped")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/gate_brain_hypothalamus.json")
    ap.add_argument("--gates", nargs="*", default=None,
                    help="gate prefixes to run, e.g. Y0 Y2.  default: all")
    ap.add_argument("--merge", action="store_true",
                    help="load the existing JSON first and keep the gates not re-run")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    rec = {"script": "scripts/gate_brain_hypothalamus.py",
           "module": "ibm/brain/hypothalamus.py",
           "git_sha": _git_sha(),
           "engine_note": "ibm/circuit.py changed three times under this gate on 18 Sep "
                          "2026 (80b23c5, db29e7e, ec07938).  Every gate in this record "
                          "was re-run together against the sha above; a suite whose rows "
                          "were measured against different engines is not a record.",
           "dt": {"slow": DT_SLOW, "sweep": DT_SWEEP},
           "constants": const_table(), "gates": {}}
    if a.merge and os.path.exists(a.out):
        with open(a.out) as fh:
            rec = json.load(fh)
        rec["constants"] = const_table()

    def save():
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=str)

    # write the cheap summary BEFORE anything that can raise (CLAUDE.md, Jobs)
    save()
    chosen = [(n, f) for n, f in GATES
              if a.gates is None or any(n.startswith(p) for p in a.gates)]
    for name, fn in chosen:
        print(f"--- {name}", flush=True)
        r = fn()
        rec["gates"][name] = r
        save()
        head = {k: v for k, v in r.items() if k not in BULKY}
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {name}: "
              f"{json.dumps(head, default=str)[:1200]}", flush=True)
        if name == "Y2b_free_run_replacement":
            print(f"      period {r['period_h']:.3f} +- {r['period_sd_h']:.3f} h over "
                  f"{r['n_cycles']} cycles; peak {r['peak_period_h']:.3f} h at "
                  f"prominence {r['peak_prominence_decades']:+.3f} decades, "
                  f"z = {r['prominence_z_vs_null']:.1f} against a null of "
                  f"{r['null_prominence_mean']:+.4f} +- {r['null_prominence_sd']:.4f} "
                  f"over {r['null_draws']} draws", flush=True)
            print(f"      known answer sine 24.000 h reads "
                  f"{r['known_answer_sine_24h']['peak_period_h']:.3f} h "
                  f"(bias {100*r['spectral_period_bias_from_known_sine']:+.2f}%), "
                  f"constant reads prominence "
                  f"{r['known_answer_constant']['prominence']:+.6f}", flush=True)
        if name == "Y2_free_run":
            print(f"      period {r['period_h']:.3f} +- {r['period_sd_h']:.3f} h over "
                  f"{r['n_cycles']} cycles; peak {r['peak_period_h']:.3f} h at "
                  f"prominence {r['peak_prominence_decades']:+.3f} decades", flush=True)
            for k, v in r["known_answers"].items():
                print(f"      known answer {k:12s} {v['peak_period_h']:8.3f} h  "
                      f"prominence {v['prominence']:+.4f}", flush=True)
        if name == "Y3_entrainment":
            for k, v in r["rows"].items():
                print(f"      T_light {v['T_light_h']:5.1f} h -> "
                      f"{('%.3f' % v['measured_h']) if v['measured_h'] else 'none':>8s} h "
                      f"(err {v['err_h']:.3f})  "
                      f"{'LOCKED' if r['locked'][k] else 'free'}", flush=True)
        if name == "Y4_sleep_pressure":
            c = r["clamped"]
            print(f"      spontaneous: wake {r['mean_wake_bout_h']:.2f} h, sleep "
                  f"{r['mean_sleep_bout_h']:.2f} h, sleep fraction "
                  f"{r['sleep_fraction']:.3f}, {r['bouts_wrong_way']} bouts wrong way",
                  flush=True)
            print(f"      clamped: rise tau {c['tau_rise_h']:.2f} h (predicted "
                  f"{c['predicted_tau_rise_h']:.2f}), 10-90% {c['t10_90_rise_h']:.2f} h | "
                  f"fall tau {c['tau_fall_h']:.2f} h (predicted "
                  f"{c['predicted_tau_fall_h']:.2f}), 90-10% {c['t90_10_fall_h']:.2f} h",
                  flush=True)
        if name == "Y5b_drive_changes_value_replacement":
            for k, v in r["rows"].items():
                print(f"      sparsity {k:>5s}: intact {v['intact_difference']:+.4f}  "
                      f"severed {v['severed_difference']:+.4f}  "
                      f"{'SATURATED' if v['saturated_intact'] else 'graded':>9s}  "
                      f"{'clears 0.05' if v['clears_Y5_bar_0.05'] else '-'}", flush=True)
        if name == "Y5_drive_changes_value":
            for k, v in r["arms"].items():
                print(f"      {k:16s} hyp.arc {v['hyp.arc']:.3f}  response "
                      f"{v['response']:.4f}  cue-evoked {v['cue_evoked']:+.4f}", flush=True)
        if name in ("Y6_sensitivity", "Y6b_sensitivity_replacement"):
            print(f"      inert over a 3x sweep: {', '.join(r['inert']) or 'none'}",
                  flush=True)
            for kind, ks in r["inert_by_kind"].items():
                print(f"        {kind:14s} {', '.join(ks) or '-'}", flush=True)
            if "clamp_binding_fraction_at_baseline" in r:
                print(f"      clamps binding at baseline: "
                      f"{r['clamp_binding_fraction_at_baseline']}", flush=True)

    ok_all = all(bool(v.get("ok")) for v in rec["gates"].values())
    rec["all_gates_ok"] = ok_all
    rec["gates_failed"] = [k for k, v in rec["gates"].items() if not v.get("ok")]
    save()
    print(f"\n{'ALL GATES PASS' if ok_all else 'SOME GATES FAILED: ' + str(rec['gates_failed'])}"
          f" -- wrote {a.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
