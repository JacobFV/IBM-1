"""known answers for `ibm/brain/valuation.py`, the structure that makes some states better
than others.

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python \\
        scripts/gate_brain_valuation.py
    ... scripts/gate_brain_valuation.py --no-sweep     # V0-V5 only, ~25 s
    ... scripts/gate_brain_valuation.py --fit          # the w_lhb_vta sweep, see below

EVERY VERDICT BELOW IS FIXED HERE, IN THIS DOCSTRING, BEFORE IT RUNS, and a failure is
recorded as FAILED rather than rescored with a different bar (CLAUDE.md, "A gate that fails
is recorded as FAILED").  The gates are about the MECHANISM -- whether what was built is a
prediction error -- not about whether this module beats something else; there is nothing to
beat.

THE PREPARATION.  `val` is run ALONE, on purpose.  Its fifteen external edges -- seven onto
and from `nm.vta`, seven touching cortex, one onto the lateral hypothalamus -- are absent in
this preparation, so the
dopamine drive is computed from `valuation.vta_inputs()`, which is the same declaration
`assemble()` would wire, read at the same conduction delays.  A0 is the separate check that
those edges resolve where they can.

Running it alone is what makes V2 a measurement of THIS structure: with `nm.vta` in the
circuit the drive would be fed back through `nm.vta -> val.bla` and every number below would
be a property of a loop rather than of the comparison the loop is built around.  Nothing
here measures `reward_theta`, `amygdala_theta` or any resonance -- those are properties of
the assembled brain and belong to `ibm/modes.py`, and no gate here claims one.

THE BASELINE, which every number in V2, V4 and V5 is quoted against, is ONE number: the
dopamine drive in the quiescent naive state, after a 6 s warm-up, with no outcome ever
delivered and the expectation at rest.  It is reported in absolute units beside every
response.  The response itself is the mean drive over the 200 ms window at the outcome,
minus that baseline.

  A0  ASSEMBLY, stated as a rule that survives its siblings landing.  For every external
      edge this structure declares, if the other end's STRUCTURE MODULE EXISTS ON DISK the
      population id must RESOLVE; if that module is not written, the edge is an orphan and
      is listed, not counted against this module.  An id that fails to resolve against a
      module that exists is a TYPO and is the only way A0 fails.  Writing it the other way
      -- "the seven nm.vta edges must be orphans" -- would have passed for twenty minutes
      and then become a false alarm, because ibm/brain/neuromodulators.py landed while this
      gate was being written.  A0 also MEASURES AND REPORTS, without folding them into its
      verdict, two conflicts that live in sibling modules and cannot be fixed from here:
      foreign edges naming a val population that does not exist, and duplicate `src->dst`
      keys declared by two structures (which `Circuit` does not raise on -- it applies the
      pathway twice at the sum of the two weights).  Not one of the six numbered gates; an
      assembly check, and it counts toward the exit status.

  V0  BOUNDED.  Every state variable stays in [0, 1] at four timesteps (1, 2, 5, 20 ms)
      under a drive of -2, 0 and +2 applied to every population at once.  Exponential-Euler
      makes this a property rather than a hope, so the gate checks the claim is true of the
      code.  V0 also reads the RESTING STATE and requires no population to be pinned: every
      rate strictly inside (0.02, 0.98), the two tonically active nuclei within 0.05 of
      their declared r_rest, and the two declared resting rates `r_core_rest`/`r_cea_rest`
      -- from which `drives()` DERIVES the pallidum's tonic drive -- within 0.02 of what
      the circuit actually rests at.  A pinned variable is what five flat sweeps in this
      repository turned out to be (CLAUDE.md, "A parameter that changes nothing").

  V1  IDEMPOTENCE.  The same protocol, branched twice from the SAME warmed snapshot with
      the same noise tape, must be bit-identical.  This is not a formality here: under
      `no_grad` `Circuit.step` writes into the ring buffer of a delayed projection IN
      PLACE, so two branches taken from one snapshot without a deep clone silently corrupt
      each other's conduction history -- and V2's three conditions are exactly two branches
      from one snapshot.  V1 also builds the noise tape twice from one seed and requires
      `torch.equal`, rather than printing both for a reader to compare (CLAUDE.md,
      "Randomness").

  V2  THE THREE-WAY RESPONSE, which is the specification of a prediction error and the gate
      that decides whether one was built.  With DA_MARGIN = 0.05 and RATIO = 0.25, declared
      here:
        a) an UNEXPECTED reward (naive, no prior rewards) must raise the drive:
           response > +0.05
        b) a FULLY PREDICTED reward (the same reward after 8 acquisition trials) must fall
           silent: |response| < 0.25 x the unexpected response
        c) an OMITTED expected reward must push the drive BELOW baseline: response < -0.05
      and, as the control that can fail rather than a gate that cannot,
        d) NULL OUTCOME: the identical protocol with the outcome amplitude set to zero must
           give all three numbers within +-0.05 of baseline.  An acquisition that drifts the
           circuit on its own would produce a "prediction error" out of nothing, and d is
           what separates that from the real thing.

  V3  APPROACH AND AVOIDANCE ARE SEPARABLE.  A positive outcome (into val.bla) and a
      negative one of the SAME DRIVE AMPLITUDE (into val.cea) are measured as six-population
      response vectors from the same naive state.  The separation measure is the COSINE
      between them, and the verdict is |cos| <= 0.5: a model in which value is one scalar
      with a sign gives exactly -1.0, and the same state twice gives +1.0.  The metric is
      checked against that known answer in the same run -- a synthetic pair (v, -v) built
      from the measured positive response must print -1.000000 (CLAUDE.md, "Check a metric
      against a case whose answer you know").

  V4  THE HABENULA IS LOAD-BEARING.  Set `w_vp_lhb = 0`, which also zeroes the habenula's
      DERIVED tonic drive so it freezes at r_rest and can no longer vary, and re-run V2.
      V2's third case must FAIL: the omission response must not clear -0.05.  This control
      can fail -- if the accumbo-tegmental edge `val.nacc_core -| nm.vta` alone produced the
      dip it would pass for a reason unrelated to what it tests, which is why `w_core_vta`
      is small and why the ablated omission number is reported, not just its sign.

  V5  EXTINCTION.  Eight consecutive omissions after acquisition.  The expectation lives in
      val.nacc_core's membrane state, so repeated omission must decay it and the dip must
      shrink: |last| < 0.5 x |first|, and val.nacc_core's rate at the outcome must fall.
      The whole trial-by-trial series is reported, and the FINAL trial is the verdict --
      the deepest is reported separately, labelled as a maximum (CLAUDE.md, "The maximum
      over a run's evaluations is not the run's result").

  V6  SENSITIVITY.  Every float constant declared in `ValParams` -- all 51, weights, time
      constants, transfer parameters and conduction delays alike -- swept to 0.5x and 1.5x
      and V2's three numbers re-measured at each.  A constant whose largest movement of any
      of the three, over that 3x sweep, is under 0.02 in drive units is listed as INERT.
      The constants this module CLAIMS carry the mechanism -- tau_core (the expectation's
      memory), w_bla_core and w_core_vp (its gain), w_vp_lhb (the inversion), w_lhb_vta (the
      subtraction) and w_actual (what arrived) -- must each move at least one of the three
      by more than 0.02, and that is V6's verdict.  Everything inert is listed.
      **Sixteen of the 51 belong only to edges ABSENT from the isolated preparation** --
      the three dopaminergic return weights, the seven cortical weights, the accumbens'
      hypothalamic efferent, and their five delays -- and cannot move anything here.  They are reported in their own list rather
      than as inert, because "not wired in" and "not present in this preparation" are
      different findings and only the first is a bug.  (The four weights and four delays on
      the edges ONTO nm.vta are NOT in that list: the drive is computed from them, so the
      sweep exercises them and they must move.)  V6's verdict also requires that every one
      of the sixteen moves NOTHING: if one does, the partition is wrong and the inert list
      cannot be trusted.

      AND EVERY INERT CONSTANT MUST CARRY A DIAGNOSIS, or V6 fails.  "This parameter does
      nothing" is not a result in this repository: five flat sweeps in one day while
      `ibm/thalamus.py` was built were each a mechanism that had not engaged (CLAUDE.md, "A
      parameter that changes nothing").  So each inert constant is given a reason computed
      from the declaration -- off the path to nm.vta in the projection graph, a delay or a
      time constant below what a 200 ms window can resolve, noise averaged over 48 units,
      or a weight deliberately small -- and a constant with no reason found is reported as
      `NO DIAGNOSIS -- look at this one` and FAILS the gate.  It was this step, not the
      inert list, that found `val.nacc_shell` had no outgoing edge at all.

`--fit` runs the `w_lhb_vta` sweep.  That constant is the ONE number in
`ibm/brain/valuation.py` set by a measurement rather than declared: it is the gain at which
a fully predicted reward is silent, which is a weight-matching condition a brain learns and
this circuit cannot.  It was set to 1.25 from the run recorded in `out/` -- the sweep is
printed in full, including the settings that fail, rather than nudged until V2b passed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ibm.brain as B                                              # noqa: E402
import ibm.brain.valuation as V                                    # noqa: E402
from ibm.circuit import Circuit                                    # noqa: E402

DT = V.DT                       # 0.002 s; every declared delay resolves to >= 2 steps
POPS = [V.BLA, V.CEA, V.CORE, V.SHELL, V.VP, V.LHB]

WARM = 3000                     # 6.0 s, four time constants of the expectation trace
TRIAL = 450                     # 0.9 s per trial
ONSET = 100                     # outcome at 0.2 s into the trial
DUR = 175                       # outcome lasts 0.35 s
WIN = (ONSET, ONSET + 100)      # the response window: 200 ms from the outcome onset
N_ACQ = 8                       # acquisition trials (the trace is within 1% of asymptote)
N_EXT = 8                       # consecutive omissions in the extinction run
AMP = 0.35                      # outcome drive amplitude, appetitive and aversive alike

DA_MARGIN = 0.05                # V2a / V2c / V2d, in units of the drive onto nm.vta
RATIO = 0.25                    # V2b: |predicted| must be under this times |unexpected|
COS_MAX = 0.5                   # V3
EXT_RATIO = 0.5                 # V5
INERT = 0.02                    # V6, in the same drive units
CLAIMED = ("tau_core", "w_bla_core", "w_core_vp", "w_vp_lhb", "w_lhb_vta", "w_actual")


# ======================================================================================
# the preparation
# ======================================================================================
def clone_state(st: dict) -> dict:
    """a DEEP copy, ring buffers included.

    `Circuit.step` under `no_grad` writes the current source rate into the ring buffer in
    place and returns the same tensor, so two branches from a shallow copy share -- and
    overwrite -- each other's conduction history.  V1 is the check that this is right.
    """
    out = {"_t": st["_t"], "_rings": {k: v.clone() for k, v in st["_rings"].items()}}
    for k, v in st.items():
        if k in ("_t", "_rings"):
            continue
        out[k] = {kk: vv.clone() for kk, vv in v.items()}
    return out


def noise_tape(c: Circuit, steps: int, seed: int) -> dict:
    """the whole run's noise, drawn ONCE from one dedicated generator and passed in.

    Never drawn inside the step loop and never from the global RNG: two branches that must
    see the same draw are given the same seed here and `torch.equal` is asserted in V1.
    """
    g = torch.Generator().manual_seed(seed)
    return {p.id: torch.randn(steps, 1, p.n, generator=g)
            for p in c.pops.values() if p.sigma}


def run(c: Circuit, st: dict, steps: int, drive: dict, events, seed: int):
    """step `steps` times, recording each population's mean rate.  Returns (traces, state)."""
    tape = noise_tape(c, steps, seed)
    tr = {k: [] for k in POPS}
    with torch.no_grad():
        for i in range(steps):
            d = dict(drive)
            for pop, amp, a, b in events:
                if a <= i < b:
                    d[pop] = d.get(pop, 0.0) + amp
            z = {k: tape[k][i] for k in tape}
            st = c.step(st, DT, drive=d, noise=z)
            for k in POPS:
                tr[k].append(float(st[k]["r"].mean()))
    return {k: torch.tensor(v) for k, v in tr.items()}, st


def da_trace(tr: dict, p: V.ValParams) -> torch.Tensor:
    """the drive onto `nm.vta`, from `valuation.vta_inputs()` -- the declaration itself.

    Each term is read at its OWN conduction delay.  `ibm/circuit.py` delivers
    `sign * weight * mean(r_src)` per unit for a dense row-normalised projection, so this is
    what nm.vta will receive the day `ibm/brain/neuromodulators.py` declares it.
    """
    out = None
    for src, sign, w, delay in V.vta_inputs(p):
        lag = int(round(delay / DT))
        s = tr[src]
        if lag > 0:
            s = torch.cat([s[:1].repeat(lag), s[:-lag]])
        term = sign * w * s
        out = term if out is None else out + term
    return out


def build(p: V.ValParams):
    return Circuit(V.pops(p), V.internal(p), seed=0), V.drives(p)


def reward(amp=AMP):
    return [(V.BLA, amp, ONSET, ONSET + DUR)]


def aversive(amp=AMP):
    return [(V.CEA, amp, ONSET, ONSET + DUR)]


def three_way(p: V.ValParams, amp: float = AMP, seed0: int = 1, want_traces=False):
    """the warm-up, the naive trial, the acquisition, and the two branches from its end.

    One baseline for all three numbers: the drive in the quiescent naive state.  The
    predicted and omitted conditions are two branches from ONE acquisition state, deep-cloned
    so they cannot touch each other's ring buffers, and given the same noise seed so they
    differ only in whether the reward arrives.
    """
    c, dr = build(p)
    st = c.init_state(1, DT)
    warm, st = run(c, st, WARM, dr, [], seed0)
    base = float(da_trace(warm, p)[-100:].mean())
    rest = {k: float(warm[k][-1]) for k in POPS}

    naive, _ = run(c, clone_state(st), TRIAL, dr, reward(amp), seed0 + 1)
    acq = clone_state(st)
    for k in range(N_ACQ):
        _, acq = run(c, acq, TRIAL, dr, reward(amp), seed0 + 10 + k)
    pred_tr, _ = run(c, clone_state(acq), TRIAL, dr, reward(amp), seed0 + 99)
    omit_tr, _ = run(c, clone_state(acq), TRIAL, dr, [], seed0 + 99)

    def resp(t):
        return float(da_trace(t, p)[WIN[0]:WIN[1]].mean()) - base

    out = {"baseline": base, "resting": rest,
           "unexpected": resp(naive), "predicted": resp(pred_tr), "omitted": resp(omit_tr),
           "core_at_outcome": {"naive": float(naive[V.CORE][WIN[0]:WIN[1]].mean()),
                               "predicted": float(pred_tr[V.CORE][WIN[0]:WIN[1]].mean()),
                               "omitted": float(omit_tr[V.CORE][WIN[0]:WIN[1]].mean())},
           "lhb_at_outcome": {"naive": float(naive[V.LHB][WIN[0]:WIN[1]].mean()),
                              "predicted": float(pred_tr[V.LHB][WIN[0]:WIN[1]].mean()),
                              "omitted": float(omit_tr[V.LHB][WIN[0]:WIN[1]].mean())}}
    if want_traces:
        out["_state"] = (c, dr, st, acq, warm, base)
    return out


# ======================================================================================
# gates
# ======================================================================================
def gate_assembly():
    """A0.  Do this structure's declared edges resolve everywhere they can?

    THE DURABLE RULE, not a snapshot of which siblings happen to exist.  For every external
    edge `val` declares, look at the other end's STRUCTURE.  If that structure's module is on
    disk, the population id must RESOLVE -- that is what catches a typo, and it is the only
    form of this check that stays meaningful while siblings land: `nm.vta` did not exist when
    this gate was written and existed twenty minutes later, and a gate that asserted "the
    seven VTA edges are orphans" would have turned from a check into a false alarm.  If the
    module is not written, the edge is an orphan, which `collect()` reports and this lists.

    A0 judges what THIS module declares.  Two classes of conflict that live in sibling
    modules are measured and reported beside the verdict rather than folded into it, because
    they cannot be fixed here and must not be lost:

      - FOREIGN EDGES naming a val population that does not exist.  `ibm/brain/__init__.py`
        reports an orphan without saying whether the id was wrong or the module was late, and
        those need opposite fixes.
      - DUPLICATE EDGE KEYS: the same `src->dst` declared by two structures.  `Circuit` keys
        its weight matrices by `Proj.key`, so a duplicate does NOT raise -- it applies the
        pathway twice in the step loop while storing one matrix, and the pathway silently
        runs at the sum of the two declared weights.  The contract in `ibm/brain/__init__.py`
        says an edge belongs to whichever structure the pathway is NAMED for, which is the
        rule that resolves each one; nothing enforces it.
    """
    # A sibling module can be MID-WRITE: `ibm/brain/thalamus.py` was an unterminated string
    # literal for the minutes it took another author to save it, and an assembly check that
    # dies on that is a check nobody can run.  A module that will not import is recorded and
    # the structures it would have resolved are treated as unknown rather than as typos.
    avail = B.available()
    prefix_of, loaded, broken = {}, {}, {}
    for s in avail:
        try:
            m = B.load(s)
            prefix_of[m.STRUCTURE] = s
            loaded[s] = m
        except Exception as exc:                                   # noqa: BLE001
            broken[s] = f"{type(exc).__name__}: {exc}"
    have, foreign, dup_src = set(), [], {}
    for s, m in loaded.items():
        have |= {p.id for p in m.pops()}
        for e in m.external():
            dup_src.setdefault(e.key, []).append(s)
            if s != "valuation" and (e.src.startswith("val.") or e.dst.startswith("val.")):
                foreign.append({"declared_by": s, "edge": e.key, "sign": e.sign,
                                "weight": e.weight})
    resolved, orphan_expected, orphan_unexpected = [], [], []
    for e in V.external():
        other = e.dst if e.src.startswith(V.STRUCTURE + ".") else e.src
        pref = other.split(".", 1)[0]
        if other in have:
            resolved.append(e.key)
        elif pref in prefix_of:
            orphan_unexpected.append({"edge": e.key, "missing_id": other,
                                      "but_module_exists": prefix_of[pref]})
        else:
            orphan_expected.append({"edge": e.key, "waiting_on_prefix": pref})
    bad_foreign = [f for f in foreign
                   if not all(x in have or not x.startswith("val.")
                              for x in f["edge"].split("->"))]
    dups = {k: v for k, v in dup_src.items() if len(v) > 1}
    _, _, _, _, _, solo = B.collect(["valuation"])
    return {"ok": bool(not orphan_unexpected and solo["n_pops"] == len(V.pops())),
            "structures_on_disk": avail, "missing_structures": B.missing(),
            "modules_that_would_not_import": broken,
            "val_pops": solo["n_pops"], "val_units": solo["units"],
            "n_external_declared": len(V.external()),
            "resolved": resolved,
            "orphan_waiting_on_an_unwritten_structure": orphan_expected,
            "orphan_though_the_module_exists_THIS_IS_A_TYPO": orphan_unexpected,
            "CONFLICT_foreign_edges_naming_a_val_pop_that_does_not_exist": bad_foreign,
            "CONFLICT_duplicate_edge_keys_across_structures": dups,
            "rule": "every external edge whose other end's structure module exists on disk "
                    "must resolve; edges waiting on an unwritten structure are listed, not "
                    "counted against this module",
            "note": "the two CONFLICT lists are defects in sibling modules.  They are "
                    "measured here and reported rather than fixed, because this gate's "
                    "module is not allowed to edit them, and rather than dropped, because "
                    "an edge that quietly disappears is a loop that quietly stops existing"}


def gate_bounded():
    """V0.  Bounded at four timesteps under extreme drive, and nothing pinned at rest."""
    p = V.PARAMS
    c, dr = build(p)
    worst, cases = 0.0, []
    for dt in (0.001, 0.002, 0.005, 0.020):
        for u in (-2.0, 0.0, 2.0):
            st = c.init_state(1, dt)
            d = {pid: u for pid in c.pops}
            with torch.no_grad():
                for _ in range(int(2.0 / dt)):
                    st = c.step(st, dt, drive=d)
            bad = 0.0
            for pid in c.pops:
                for key, v in st[pid].items():
                    bad = max(bad, float((-v).clamp_min(0).max()),
                              float((v - 1).clamp_min(0).max()) if key == "r" else 0.0)
            worst = max(worst, bad)
            cases.append({"dt": dt, "drive": u, "worst": bad,
                          "rates": {pid: round(float(st[pid]["r"].mean()), 4)
                                    for pid in c.pops}})
    # the resting state: is anything pinned, and do the two derived constants still hold?
    st = c.init_state(1, DT)
    warm, _ = run(c, st, WARM, dr, [], 1)
    rest = {k: float(warm[k][-1]) for k in POPS}
    pinned = [k for k, v in rest.items() if v <= 0.02 or v >= 0.98]
    tonic_off = {V.VP: abs(rest[V.VP] - p.r_rest), V.LHB: abs(rest[V.LHB] - p.r_rest)}
    derived_off = {"r_core_rest": abs(rest[V.CORE] - p.r_core_rest),
                   "r_cea_rest": abs(rest[V.CEA] - p.r_cea_rest)}
    ok = (worst <= 0.0 and not pinned
          and max(tonic_off.values()) <= 0.05 and max(derived_off.values()) <= 0.02)
    return {"ok": bool(ok), "worst_excursion": worst, "resting": rest,
            "pinned": pinned, "tonic_offset_from_r_rest": tonic_off,
            "declared_minus_measured": derived_off,
            "vanishing_delays_at_dt": c.vanishing_delays(DT),
            "rule": "worst excursion 0.0; no rate outside (0.02, 0.98); vp and lhb within "
                    "0.05 of r_rest; r_core_rest and r_cea_rest within 0.02 of measured",
            "cases": cases}


def gate_idempotent():
    """V1.  Two branches from one snapshot, same tape: bit-identical, or nothing is a
    function of its arguments and every number below is void."""
    p = V.PARAMS
    c, dr = build(p)
    t1 = noise_tape(c, 64, 4242)
    t2 = noise_tape(c, 64, 4242)
    tapes_equal = all(torch.equal(t1[k], t2[k]) for k in t1)
    st = c.init_state(1, DT)
    _, st = run(c, st, 600, dr, [], 1)
    a, _ = run(c, clone_state(st), TRIAL, dr, reward(), 77)
    b, _ = run(c, clone_state(st), TRIAL, dr, reward(), 77)
    diff = max(float((a[k] - b[k]).abs().max()) for k in POPS)
    da_diff = float((da_trace(a, p) - da_trace(b, p)).abs().max())
    # and the whole V2 measurement twice
    r1 = three_way(p)
    r2 = three_way(p)
    three = max(abs(r1[k] - r2[k]) for k in ("unexpected", "predicted", "omitted"))
    return {"ok": bool(tapes_equal and diff == 0.0 and three == 0.0),
            "noise_tapes_equal": bool(tapes_equal),
            "max_abs_rate_diff": diff, "max_abs_da_diff": da_diff,
            "max_abs_three_way_diff": three,
            "rule": "same seed, same snapshot, bit-identical; the noise tapes torch.equal"}


def gate_three_way():
    """V2.  Unexpected, fully predicted, omitted -- and the null-outcome control."""
    p = V.PARAMS
    r = three_way(p)
    null = three_way(p, amp=0.0)
    ratio = abs(r["predicted"]) / max(abs(r["unexpected"]), 1e-12)
    a = r["unexpected"] > DA_MARGIN
    b = ratio < RATIO
    cc = r["omitted"] < -DA_MARGIN
    d = all(abs(null[k]) <= DA_MARGIN for k in ("unexpected", "predicted", "omitted"))
    return {"ok": bool(a and b and cc and d),
            "baseline_da_drive": r["baseline"],
            "unexpected": r["unexpected"], "predicted": r["predicted"],
            "omitted": r["omitted"], "predicted_over_unexpected": ratio,
            "a_unexpected_raises": bool(a), "b_predicted_silent": bool(b),
            "c_omitted_below_baseline": bool(cc), "d_null_outcome_control": bool(d),
            "null_outcome": {k: null[k] for k in ("unexpected", "predicted", "omitted")},
            "expectation_val_nacc_core": r["core_at_outcome"],
            "val_lhb_at_outcome": r["lhb_at_outcome"],
            "rule": f"unexpected > +{DA_MARGIN}; |predicted| < {RATIO} x unexpected; "
                    f"omitted < -{DA_MARGIN}; null outcome within +-{DA_MARGIN} on all three"}


def gate_separable():
    """V3.  A positive and a negative outcome of the same amplitude, from the same state."""
    p = V.PARAMS
    c, dr = build(p)
    st = c.init_state(1, DT)
    warm, st = run(c, st, WARM, dr, [], 1)
    rest = {k: float(warm[k][-100:].mean()) for k in POPS}
    pos, _ = run(c, clone_state(st), TRIAL, dr, reward(), 5)
    neg, _ = run(c, clone_state(st), TRIAL, dr, aversive(), 5)

    def vec(t):
        return torch.tensor([float(t[k][WIN[0]:WIN[1]].mean()) - rest[k] for k in POPS])

    v_pos, v_neg = vec(pos), vec(neg)
    cos = float((v_pos @ v_neg) / (v_pos.norm() * v_neg.norm()))
    known = float((v_pos @ (-v_pos)) / (v_pos.norm() * v_pos.norm()))  # must be -1.000000
    base = float(da_trace(warm, p)[-100:].mean())
    return {"ok": bool(abs(cos) <= COS_MAX and abs(known + 1.0) < 1e-5),
            "cosine": cos, "angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, cos)))),
            "known_answer_signed_scalar_cosine": known,
            "positive_response": {k: float(v) for k, v in zip(POPS, v_pos)},
            "negative_response": {k: float(v) for k, v in zip(POPS, v_neg)},
            "norms": {"positive": float(v_pos.norm()), "negative": float(v_neg.norm())},
            "da_positive": float(da_trace(pos, p)[WIN[0]:WIN[1]].mean()) - base,
            "da_negative": float(da_trace(neg, p)[WIN[0]:WIN[1]].mean()) - base,
            "rule": f"|cos| <= {COS_MAX}; a one-signed-scalar model gives exactly -1, and "
                    f"the metric is checked against that case in the same run"}


def gate_habenula():
    """V4.  Silence the habenula and V2's third case must fail."""
    p = V.PARAMS
    intact = three_way(p)
    # w_vp_lhb = 0 also zeroes the habenula's DERIVED tonic drive (drives() computes it as
    # w_vp_lhb * r_rest), so val.lhb freezes at r_rest and can no longer vary at all.
    lesion = three_way(replace(p, w_vp_lhb=0.0))
    dip_gone = not (lesion["omitted"] < -DA_MARGIN)
    shrink = 1.0 - abs(lesion["omitted"]) / max(abs(intact["omitted"]), 1e-12)
    return {"ok": bool(dip_gone),
            "intact_omitted": intact["omitted"], "silenced_omitted": lesion["omitted"],
            "dip_removed_fraction": shrink,
            "silenced_unexpected": lesion["unexpected"],
            "silenced_predicted": lesion["predicted"],
            "silenced_lhb_at_outcome": lesion["lhb_at_outcome"],
            "rule": f"with w_vp_lhb = 0 the omission response must NOT clear "
                    f"-{DA_MARGIN}, i.e. V2c must fail",
            "note": "the silenced arm's PREDICTED number is reported too: with no habenula "
                    "there is nothing to subtract, so a fully predicted reward bursts as "
                    "hard as an unexpected one"}


def gate_extinction():
    """V5.  Repeated omission decays the expectation and the dip shrinks."""
    p = V.PARAMS
    c, dr = build(p)
    st = c.init_state(1, DT)
    warm, st = run(c, st, WARM, dr, [], 1)
    base = float(da_trace(warm, p)[-100:].mean())
    s = clone_state(st)
    for k in range(N_ACQ):
        _, s = run(c, s, TRIAL, dr, reward(), 10 + k)
    rows = []
    for k in range(N_EXT):
        t, s = run(c, s, TRIAL, dr, [], 200 + k)
        rows.append({"trial": k + 1,
                     "response": float(da_trace(t, p)[WIN[0]:WIN[1]].mean()) - base,
                     "val_nacc_core": float(t[V.CORE][WIN[0]:WIN[1]].mean()),
                     "val_lhb": float(t[V.LHB][WIN[0]:WIN[1]].mean())})
    first, last = rows[0], rows[-1]
    deepest = min(rows, key=lambda r: r["response"])
    ok = (abs(last["response"]) < EXT_RATIO * abs(first["response"])
          and last["val_nacc_core"] < first["val_nacc_core"])
    return {"ok": bool(ok),
            "first_trial": first, "last_trial": last,
            "deepest_trial_reported_as_a_maximum": deepest,
            "series": rows,
            "rule": f"|last| < {EXT_RATIO} x |first| AND val.nacc_core falls; the FINAL "
                    f"trial is the verdict, the deepest is reported as a maximum"}


def _reaches_vta():
    """the val populations from which `nm.vta` is reachable in the DECLARED graph.

    A constant belonging to a population outside this set, or to an edge landing in one,
    cannot move the dopamine drive -- not because the mechanism failed to engage but because
    there is no path.  This is computed from `internal()` + `external()`, so the diagnosis
    of a flat sweep is a graph fact and not a story told after seeing it flat.
    """
    edges = [(e.src, e.dst) for e in list(V.internal()) + list(V.external())]
    reach, changed = {V.VTA}, True
    while changed:
        changed = False
        for s, d in edges:
            if d in reach and s not in reach:
                reach.add(s)
                changed = True
    return reach


def gate_sensitivity(record_cb=None):
    """V6.  Every declared float constant +-50%, V2's three numbers re-measured at each."""
    p = V.PARAMS
    orphan_names = {"w_vta_bla", "w_vta_core", "w_vta_shell", "w_bla_mofc", "w_bla_racc",
                    "w_mofc_core", "w_mofc_shell", "w_mofc_bla", "w_racc_core",
                    "w_racc_cea", "d_vta_bla", "d_vta_nacc", "d_bla_ctx", "d_ctx_val",
                    "w_shell_lh", "d_shell_hyp"}
    reach = _reaches_vta()
    off_path = {"w_bla_shell": V.SHELL, "tau_shell": V.SHELL}
    off_path = {k: v for k, v in off_path.items() if v not in reach}
    base = three_way(p)
    b3 = (base["unexpected"], base["predicted"], base["omitted"])
    rows = []
    for name in p.swept():
        v = getattr(p, name)
        arm = {"param": name, "value": v, "on_orphan_edge": name in orphan_names}
        span = 0.0
        for tag, f in (("half", 0.5), ("x1p5", 1.5)):
            r = three_way(replace(p, **{name: v * f}))
            arm[tag] = {"unexpected": r["unexpected"], "predicted": r["predicted"],
                        "omitted": r["omitted"]}
            span = max(span, max(abs(r["unexpected"] - b3[0]),
                                 abs(r["predicted"] - b3[1]),
                                 abs(r["omitted"] - b3[2])))
        arm["span"] = span
        rows.append(arm)
        if record_cb:
            record_cb(rows)
    rows.sort(key=lambda r: -r["span"])
    claimed = {r["param"]: r["span"] for r in rows if r["param"] in CLAIMED}
    inert = [r["param"] for r in rows if r["span"] < INERT and not r["on_orphan_edge"]]
    not_exercised = [r["param"] for r in rows if r["on_orphan_edge"]]
    not_exercised_moved = [r["param"] for r in rows if r["on_orphan_edge"]
                           and r["span"] >= INERT]
    # THE DIAGNOSIS STEP.  A flat sweep is not a finding until the reason is found: five
    # flat sweeps in this repository were mechanisms that had not engaged (CLAUDE.md).  Each
    # inert constant is given a reason computed from the declaration -- a graph fact, an
    # instrument limit, or NONE, which is the case that needs a human.
    window_ms = (WIN[1] - WIN[0]) * DT * 1000.0
    diagnosis = {}
    for name in inert:
        v = getattr(p, name)
        if name in off_path:
            diagnosis[name] = (f"off the path to {V.VTA}: {off_path[name]} has no route to "
                               f"it in the declared graph, so this constant cannot move the "
                               f"drive.  The pull is an OUTPUT of this structure")
        elif name.startswith("d_"):
            diagnosis[name] = (f"a conduction delay of {v * 1000:.1f} ms read through a "
                               f"{window_ms:.0f} ms response window: +-50% shifts a term by "
                               f"{v * 500:.1f} ms inside that mean.  Delays are the "
                               f"conduction budget of a LOOP, and this preparation is open")
        elif name.startswith("tau_"):
            diagnosis[name] = (f"a membrane time constant of {v * 1000:.0f} ms read through "
                               f"a {window_ms:.0f} ms window: the population is at its "
                               f"plateau either way")
        elif name == "sigma":
            diagnosis[name] = ("background noise, averaged over 32-48 units before it "
                               "reaches any projection")
        elif name == "w_core_vta":
            diagnosis[name] = ("small BY DESIGN at 0.10: it closes the `val` loop and is "
                               "deliberately not the subtraction, which is what lets V4 "
                               "fail.  Measured not carrying it, as declared")
        else:
            diagnosis[name] = "NO DIAGNOSIS -- look at this one"
    undiagnosed = [k for k, v in diagnosis.items() if v.startswith("NO DIAGNOSIS")]
    return {"ok": bool(all(v > INERT for v in claimed.values())
                       and not not_exercised_moved and not undiagnosed),
            "baseline_three": {"unexpected": b3[0], "predicted": b3[1], "omitted": b3[2]},
            "claimed_mechanism_span": claimed,
            "inert_under_%.3f_over_3x" % INERT: inert,
            "n_inert": len(inert), "n_swept": len(rows),
            "inert_diagnosis": diagnosis, "undiagnosed_inert": undiagnosed,
            "pops_that_reach_nm_vta": sorted(reach),
            "not_exercised_orphan_edges": not_exercised,
            "not_exercised_but_moved": not_exercised_moved,
            "rule": f"every constant in {CLAIMED} must move one of the three numbers by "
                    f"more than {INERT}; the 16 constants on edges absent from this "
                    f"preparation are listed separately and must move NOTHING (if one moves "
                    f"the partition is wrong); and every INERT constant must carry a "
                    f"diagnosis -- a flat sweep with no reason found fails this gate",
            "sweep": rows}


def fit_w_lhb_vta(values):
    """the one fitted constant, swept in the open, failures printed with the rest."""
    rows = []
    for w in values:
        r = three_way(replace(V.PARAMS, w_lhb_vta=w))
        ratio = abs(r["predicted"]) / max(abs(r["unexpected"]), 1e-12)
        rows.append({"w_lhb_vta": w, "unexpected": r["unexpected"],
                     "predicted": r["predicted"], "omitted": r["omitted"],
                     "predicted_over_unexpected": ratio,
                     "passes_V2": bool(r["unexpected"] > DA_MARGIN and ratio < RATIO
                                       and r["omitted"] < -DA_MARGIN)})
        print(f"  w_lhb_vta {w:5.2f}  unexpected {r['unexpected']:+.4f}  "
              f"predicted {r['predicted']:+.4f}  omitted {r['omitted']:+.4f}  "
              f"ratio {ratio:6.3f}  {'V2 passes' if rows[-1]['passes_V2'] else 'V2 FAILS'}",
              flush=True)
    return rows


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/gate_brain_valuation.json")
    ap.add_argument("--no-sweep", action="store_true", help="skip V6")
    ap.add_argument("--fit", action="store_true", help="the w_lhb_vta sweep only")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rec = {"script": "scripts/gate_brain_valuation.py", "structure": V.STRUCTURE,
           "dt_s": DT, "protocol": {"warm_steps": WARM, "trial_steps": TRIAL,
                                    "onset_step": ONSET, "outcome_steps": DUR,
                                    "response_window_steps": list(WIN), "n_acq": N_ACQ,
                                    "n_ext": N_EXT, "amplitude": AMP},
           "thresholds": {"da_margin": DA_MARGIN, "ratio": RATIO, "cos_max": COS_MAX,
                          "ext_ratio": EXT_RATIO, "inert": INERT, "claimed": list(CLAIMED)},
           "params": {f: getattr(V.PARAMS, f) for f in V.PARAMS.__dataclass_fields__},
           "gates": {}}

    def save():
        # written before anything that can raise, and numpy/tensors degraded rather than
        # allowed to destroy the record (CLAUDE.md)
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=lambda o: getattr(o, "tolist", lambda: str(o))())

    save()
    if a.fit:
        print("w_lhb_vta: the gain at which a fully predicted reward is silent", flush=True)
        rec["fit_w_lhb_vta"] = fit_w_lhb_vta([0.75, 0.90, 1.05, 1.20, 1.25, 1.35, 1.50])
        save()
        print(f"\nwrote {a.out}")
        return 0

    order = [("A0_assembly", gate_assembly),
             ("V0_bounded", gate_bounded),
             ("V1_idempotence", gate_idempotent),
             ("V2_three_way_prediction_error", gate_three_way),
             ("V3_separable_valence", gate_separable),
             ("V4_habenula_load_bearing", gate_habenula),
             ("V5_extinction", gate_extinction)]
    if not a.no_sweep:
        order.append(("V6_sensitivity", gate_sensitivity))

    ok_all = True
    t0 = time.time()
    for name, fn in order:
        r = fn()
        rec["gates"][name] = r
        save()
        ok_all &= bool(r.get("ok"))
        skip = ("cases", "sweep", "series", "positive_response", "negative_response")
        head = {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in r.items() if k not in skip}
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {name}", flush=True)
        for k, v in head.items():
            if k == "ok":
                continue
            print(f"      {k}: {v}", flush=True)
        if name == "V3_separable_valence":
            print("      per-population response (positive | negative):", flush=True)
            for pid in POPS:
                print(f"        {pid:16s} {r['positive_response'][pid]:+.4f}   "
                      f"{r['negative_response'][pid]:+.4f}", flush=True)
        if name == "V5_extinction":
            for row in r["series"]:
                print(f"        omission {row['trial']}: response "
                      f"{row['response']:+.4f}   val.nacc_core {row['val_nacc_core']:.4f}   "
                      f"val.lhb {row['val_lhb']:.4f}", flush=True)
        if name == "V6_sensitivity":
            print("      largest movers over the 3x sweep:", flush=True)
            for row in r["sweep"][:10]:
                print(f"        {row['param']:14s} {row['value']:+.4g}  span "
                      f"{row['span']:.4f}", flush=True)
        print(flush=True)
    rec["all_gates_ok"] = ok_all
    rec["wall_seconds"] = time.time() - t0
    save()
    print(f"{'ALL GATES PASS' if ok_all else 'SOME GATES FAILED'} -- wrote {a.out} "
          f"({rec['wall_seconds']:.0f} s)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
