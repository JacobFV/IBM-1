"""known answers for `ibm/brain/neuromodulators.py`, declared before they run.

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        .venv/bin/python scripts/gate_brain_neuromodulators.py
    ... --only N2,N4          run a subset (the JSON then covers only that subset)

CPU only.  Every verdict below is fixed HERE, in this docstring and in the code, before any
of it is measured, and a failure is recorded as FAILED rather than rescored (CLAUDE.md).
The module is the thing under test; the thresholds are not.

THE STATE VARIABLE, once, so every gate below means the same thing by "state":

    S(t) = mean rate of (nm.lc, nm.raphe, nm.tmn)  -  rate of nm.ppt

the monoaminergic (REM-off, wake-active) side minus the brainstem cholinergic (REM-on) one.
State assignment is a SCHMITT TRIGGER at S = +0.25 / -0.25: the network is called HIGH once
S rises past +0.25 and stays HIGH until S falls past -0.25.  Both trigger levels are
reported beside every number they produce, because a dwell measured with a deadband is
partly a measurement of the deadband.

  N0  BOUNDED.  Every state variable in [0, 1] at dt = 1, 5 and 20 ms, with the declared
      drives scaled by -2, 0, +1 and +3.  The exponential-Euler step makes this a property
      rather than a hope; the gate checks it is true of the code and not only of the
      algebra.  VERDICT: worst excursion outside [0, 1] is exactly 0.
      Reported beside it, not gated: the resting rates of nm.vta and nm.snc against
      `ibm/brain/basal_ganglia.py`'s `da_baseline` = 0.50, the rate at which that module's
      striatal weights are stated.  If these two nuclei do not RUN near 0.50, every
      declared corticostriatal weight is off by a factor nobody declared.
  N1  IDEMPOTENCE.  Two things, because they fail differently.  (a) Each of the six
      contract functions called TWICE with the same argument returns the same declaration
      -- a declaration that drew from a generator would make the brain a different brain
      each time it was assembled.  (b) The same rollout with the same generator, twice,
      bit-identical.  VERDICT: both, exactly.
  N2  TWO STATES, NOT A CONTINUUM.  The drive that pushes the flip-flop (a bias added to
      the aminergic drives and subtracted from PPT's) is swept over 9 levels spanning
      +-0.20, and S is pooled over the whole sweep.  A network that slides through the
      middle as the bias is swept fills the middle; a flip-flop moves the FRACTION OF TIME
      at each end and leaves the ends where they are.  Two measures, both declared:
        * occupancy of the central third of the gap between the two state means.  A
          continuum gives about 0.33 by construction.  VERDICT: < 0.15.
        * the bimodality coefficient (skew^2 + 1) / (excess kurtosis + 3).  A uniform
          distribution gives exactly 0.5556 and a two-point one gives 1.0.
          VERDICT: > 0.5556.
      The per-level table is printed too, because it is the table and not either scalar
      that distinguishes "two states" from "one state being dragged".
  N3  THE TRANSITION IS FAST AGAINST THE DWELL.  Crossing time is the 10%-90% passage
      between the two state means; dwell is the time between Schmitt flips.
      VERDICT: mean crossing < 0.10 x mean dwell.
      THE INSTRUMENT'S OWN FLOOR IS REPORTED BESIDE IT: no crossing of this network can be
      faster than the fastest switching population's tau (tau_ppt = 200 ms), whatever the
      switch does, exactly as a band-limited envelope cannot be shorter than 1/bandwidth
      (CLAUDE.md).  A crossing at the floor is a measurement of tau, not of the flip-flop.
  N4  OREXIN STABILISES.  `nm.orx` silenced by driving it to -1.0 -- the population, the
      wiring and the seeds are otherwise IDENTICAL, so nothing but its rate changes.  Three
      seeds, paired.  VERDICT: on every seed, transitions UP and mean dwell DOWN.
      This gate can fail.  Orexin's only edges are `beta` modulations, and if a `beta`
      modulation of a saturated flip-flop does nothing then the arms read identical and the
      module has a parameter that changes nothing.
  N5  THE MODULATION REACHES.  A two-population circuit built from the REAL declarations --
      `nm.lc` from this module and `ctx.pericalcarine.E` from `ibm/brain/cortex.py`, with
      the `Mod` objects taken verbatim from `mods()`, not rewritten -- and the modulator's
      rate swept from silent to saturated.  Three arms:
        a  the `beta` edge alone.  GAIN IS A SLOPE, NOT A LEVEL, and this is the T5 -> T5b
           lesson from `scripts/gate_thalamus.py` applied before the fact: a steeper
           sigmoid at a SUBTHRESHOLD operating point gives a LOWER mean rate, so measuring
           the rate would report noradrenaline's sign backwards.  The target is driven with
           a random input and the transfer slope regressed.
           VERDICT: slope at saturated modulator >= 1.5 x slope at silent modulator.
        b  the `sigma` edge alone, under a CONSTANT drive so the only variation left is the
           background current.  VERDICT: per-unit temporal sd at saturated modulator
           <= 0.7 x its value at silent modulator.
        c  THE CONTROL, and it is the one that makes a and b mean anything: the same two
           populations with NO `Mod` edges.  VERDICT: bit-identical output at every
           modulator level.  A control that cannot fail is worth nothing; this one fails if
           the modulator reaches the target by any route other than the edge under test.
  N6  SENSITIVITY.  Every constant in `CONST` swept +-50% and the mean dwell re-measured,
      WITH THE NOISE OFF so the dwell is a deterministic period and the comparison carries
      no sampling error (`sigma_nm` is therefore inert by construction and is listed as
      such rather than left in the inert list to be misread).
      VERDICT: `adapt_tau`, `adapt_g` and `orx_beta_gain` -- the three the module CLAIMS
      set the dwell -- each move it by more than 20% over the 3x sweep.
      A constant is INERT only if it moves neither the dwell (the CLOCK) nor the fraction
      of time spent high (the BALANCE) by 5%, and a constant whose sweep LOCKS the
      flip-flop is never inert however flat the dwell reads -- `measure` reports the whole
      run as the dwell when nothing flips, so a locked arm and an unchanged one were
      indistinguishable until that was fixed.  Each inert constant carries a column saying
      whether its target even exists in the nm-only circuit: a modulation gain onto
      `thal.*` is inert here because the thalamus is not in this circuit, which is a
      different fact from a mechanism that is not wired in.
  N7  ONE EDGE, ONE FACTOR.  Over every structure module that exists on disk, two checks
      with the same shape.  No (src, dst, param) `Mod` triple declared by two modules:
      `Circuit._modulation` MULTIPLIES factors onto the same pair, so a duplicate does not
      raise, it applies the modulator twice.  And no (src, dst, sign) external `Proj`
      declared by two modules: `Circuit.step` SUMS contributions, so a duplicate makes the
      pathway twice as strong as either module says.  The modules disagree about where a
      `Mod` belongs (`thalamus.py` expects the source to own it, `basal_ganglia.py`
      declares it at the target), so this will happen again -- and the `Proj` half already
      did, twice, the first time this gate was run.
      Also checked: every foreign id this module names inside a structure that EXISTS must
      be a real population of it; and every sibling module must be READABLE, because a
      cross-module check that skipped an unparseable module would report a clean sheet it
      did not earn.  VERDICT: no duplicates, no dead live-structure ids, nothing skipped.

WHAT THIS GATE DOES NOT TEST, said plainly: the SLEEP-WAKE flip-flop.  Every gate here runs
the nm populations ALONE, so the two states measured are the REM-off (monoaminergic) and
REM-on (cholinergic) sides.  `hyp.vlpo` now exists in `ibm/brain/hypothalamus.py` and its
edges are live in the full assembly, which means wake-versus-NREM is now measurable -- and
it has NOT been measured here.  Likewise `hyp.lh -> nm.orx`: assembled, orexin is driven
by the lateral hypothalamus and will not sit at the 0.45 this module's own drive holds it
at, so N4's silencing arm is a statement about the nm-only circuit.
And the dwell is on the module's declared COMPRESSED clock (~1000x fast against
`ultradian_rem`'s 4200-6600 s); no number here is a claim about the ninety-minute cycle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm.brain import Mod, Pop, Proj  # noqa: E402,F401
import ibm.brain as B                 # noqa: E402
import ibm.brain.cortex as CX         # noqa: E402
import ibm.brain.neuromodulators as NM  # noqa: E402
from ibm.circuit import Circuit       # noqa: E402

DT = 0.005            # the gates' working timestep; taus are 200-400 ms and delays 20 ms
DT_LONG = 0.010       # N2 and N4, where the run has to be long enough for many dwells
DT_SWEEP = 0.020      # N6 only.  Checked against DT: the same run reads a 13.90 s dwell at
                      # dt = 20 ms and 13.81 s at dt = 5 ms, 0.6% apart
HI, LO = 0.25, -0.25  # the Schmitt trigger, declared once
SECONDS = 180.0       # the mean dwell is ~13 s, so this is ~13 dwells
ORX_SILENCE_DRIVE = -1.0

#: the constants the module CLAIMS set the dwell.  N6's verdict is about these three.
CLAIMED_CLOCK = ("adapt_tau", "adapt_g", "orx_beta_gain")


# ======================================================================================
# reading the sibling structure modules
# ======================================================================================
def load_structures():
    """(name -> module, name -> why it could not be read).

    Other agents edit the sibling modules while this runs, so one of them can be briefly
    unparseable.  A cross-module check that dies on that reports nothing about THIS module;
    one that silently skips it reports a clean sheet it did not earn.  So the failure is
    carried into the record by name, and any gate that depends on a module it could not
    read says so in its verdict rather than passing.
    """
    ok, bad = {}, {}
    for n in B.available():
        try:
            ok[n] = B.load(n)
        except Exception as exc:                      # noqa: BLE001
            bad[n] = f"{type(exc).__name__}: {exc}"
    return ok, bad


# ======================================================================================
# building and running the nm-only circuit
# ======================================================================================
def build(c=None, seed: int = 0):
    """the nm populations, their internal projections, and the `Mod` edges that land inside.

    `ibm.brain.collect()` is not used, because this gate needs the nm populations ALONE and
    `collect()` takes structure names rather than a population set.  The filtering it does
    is reproduced here and named rather than left implicit.

    (When this gate was first written `collect()` passed every `Mod` straight through while
    `Circuit.__init__` asserted its endpoints, so assembling a partial structure list
    RAISED where the identical `Proj` would have been reported as an orphan.  That was
    fixed in `ibm/brain/__init__.py` while this was being written; the note stays because
    the gate's own filtering exists for that reason.)
    """
    pops = NM.pops(c)
    have = {p.id for p in pops}
    projs = [e for e in NM.internal(c) if e.src in have and e.dst in have]
    mods = [m for m in NM.mods(c) if m.src in have and m.dst in have]
    return Circuit(pops, projs, mods, seed=seed, device="cpu")


def run(c=None, seconds=SECONDS, dt=DT, seed=1, bias=0.0, orx_drive=None, burn=3.0,
        circ=None):
    circ = circ or build(c)
    d = dict(NM.drives(c))
    for p in NM.AMINERGIC:
        d[p] += bias
    d[NM.PPT] -= bias
    if orx_drive is not None:
        d[NM.ORX] = orx_drive
    g = torch.Generator().manual_seed(seed)
    tr, _ = circ.rollout(int(seconds / dt), dt, drive=d, noise_gen=g, b=1,
                         record=list(circ.pops), burn=int(burn / dt))
    return tr


def state_var(tr) -> torch.Tensor:
    amin = sum(tr[p] for p in NM.AMINERGIC) / len(NM.AMINERGIC)
    return (amin.mean(-1) - tr[NM.PPT].mean(-1))[0]


# ======================================================================================
# the instrument: states, dwells, crossings, bimodality
# ======================================================================================
def schmitt(S, hi=HI, lo=LO):
    """(state per sample or None before the first commitment, flip indices)."""
    st, states, flips = None, [], []
    for i, s in enumerate(S.tolist()):
        if s > hi:
            if st == -1:
                flips.append(i)
            st = 1
        elif s < lo:
            if st == 1:
                flips.append(i)
            st = -1
        states.append(st)
    return states, flips


def measure(S, dt):
    """everything N2, N3, N4 and N6 read, from one pass over S."""
    states, flips = schmitt(S)
    st = torch.tensor([0 if s is None else s for s in states])
    n_hi, n_lo = int((st == 1).sum()), int((st == -1).sum())
    out = {"n_transitions": len(flips), "seconds": len(S) * dt,
           "trigger_hi": HI, "trigger_lo": LO,
           "frac_high": n_hi / max(1, n_hi + n_lo)}
    if n_hi < 10 or n_lo < 10:
        # the flip-flop did not flip.  The dwell is at least the whole run; say so rather
        # than returning None and letting a caller read a missing value as a small one.
        out.update({"dwell_s": float(len(S) * dt), "dwell_is_lower_bound": True,
                    "gap": None, "central_occupancy": None, "bimodality": None,
                    "crossing_s": None, "state_hi": None, "state_lo": None})
        return out
    a = float(S[st == 1].mean())
    b = float(S[st == -1].mean())
    gap, mid = a - b, (a + b) / 2
    dwells = [(flips[i + 1] - flips[i]) * dt for i in range(len(flips) - 1)]
    out.update({"state_hi": a, "state_lo": b, "gap": gap,
                "dwell_s": (sum(dwells) / len(dwells)) if dwells else float(len(S) * dt),
                "dwell_is_lower_bound": not dwells, "n_dwells": len(dwells),
                "central_occupancy": float(((S - mid).abs() < gap / 6).float().mean())})
    # bimodality coefficient, large-sample form.  Uniform = 0.5556, two-point = 1.0.
    x = S - S.mean()
    v = float((x * x).mean())
    if v > 0:
        g1 = float((x ** 3).mean()) / v ** 1.5
        g2 = float((x ** 4).mean()) / v ** 2 - 3.0
        out["bimodality"] = (g1 * g1 + 1.0) / (g2 + 3.0)
        out["skew"], out["excess_kurtosis"] = g1, g2
    else:
        out["bimodality"] = None
    # 10%-90% crossing time between the two state levels
    lo10, hi90 = mid - 0.4 * gap, mid + 0.4 * gap
    cross, last_lo, last_hi = [], None, None
    for i, s in enumerate(S.tolist()):
        if s <= lo10:
            if last_hi is not None:
                cross.append((i - last_hi) * dt)
                last_hi = None
            last_lo = i
        elif s >= hi90:
            if last_lo is not None:
                cross.append((i - last_lo) * dt)
                last_lo = None
            last_hi = i
    out["crossing_s"] = (sum(cross) / len(cross)) if cross else None
    out["n_crossings"] = len(cross)
    return out


# ======================================================================================
# N0
# ======================================================================================
#: the state variables that are RATES or RESOURCES and are bounded in [0, 1].  `eta` is an
#: Ornstein-Uhlenbeck background CURRENT and `a` is an adaptation current: both are signed
#: and neither has a bound, so gating them would fail the module for the OU noise doing
#: exactly what it is for.  Their ranges are reported instead.
BOUNDED_VARS = ("r", "x", "u", "inh")
UNBOUNDED_VARS = ("a", "eta")


def gate_bounded():
    worst, cases, unbounded = 0.0, [], {}
    base = NM.drives()
    for dt in (0.001, 0.005, 0.020):
        for k in (-2.0, 0.0, 1.0, 3.0):
            circ = build()
            d = {p: v * k for p, v in base.items()}
            g = torch.Generator().manual_seed(11)
            st = circ.init_state(1, dt)
            for _ in range(int(4.0 / dt)):
                z = {p.id: torch.randn(1, p.n, generator=g) for p in circ.pops.values()
                     if p.sigma}
                st = circ.step(st, dt, drive=d, noise=z)
            bad = 0.0
            for pid in circ.pops:
                for key, v in st[pid].items():
                    if not torch.is_tensor(v):
                        continue
                    if key in BOUNDED_VARS:
                        hi_ex = float((v - 1).clamp_min(0).max()) if key != "inh" else 0.0
                        bad = max(bad, float((-v).clamp_min(0).max()), hi_ex)
                    elif key in UNBOUNDED_VARS:
                        lo, hi = unbounded.get(key, (0.0, 0.0))
                        unbounded[key] = (min(lo, float(v.min())), max(hi, float(v.max())))
            worst = max(worst, bad)
            cases.append({"dt": dt, "drive_scale": k, "worst": bad})
    # reported beside the verdict, not gated: the dopaminergic operating point
    tr = run(seconds=40.0, seed=1)
    da = {p: float(tr[p].mean()) for p in (NM.VTA, NM.SNC)}
    rest = {p: float(tr[p].mean()) for p in tr}
    return {"ok": worst <= 0.0, "worst_excursion": worst, "cases": cases,
            "bounded_vars": list(BOUNDED_VARS),
            "unbounded_var_ranges": {k: list(v) for k, v in unbounded.items()},
            "mean_rates": rest, "dopaminergic_rates": da,
            "da_rest_target": NM.CONST["da_rest_target"],
            "note": "basal_ganglia.py states its striatal weights at da_baseline = 0.50; "
                    "nm.vta and nm.snc must RUN there or those weights are off by an "
                    "undeclared factor.  Reported, not gated."}


# ======================================================================================
# N1
# ======================================================================================
def gate_idempotent():
    decl = {}
    for name in ("pops", "internal", "external", "mods", "drives", "targets"):
        f = getattr(NM, name)
        decl[name] = (repr(f()) == repr(f()))
    a = state_var(run(seconds=20.0, seed=7))
    b = state_var(run(seconds=20.0, seed=7))
    same = bool(torch.equal(a, b))
    return {"ok": all(decl.values()) and same,
            "declarations_idempotent": decl, "rollout_bit_identical": same,
            "max_abs_diff": float((a - b).abs().max()),
            "note": "a declaration that drew from a generator would make the brain a "
                    "different brain every time it was assembled"}


# ======================================================================================
# N2
# ======================================================================================
def gate_two_states():
    biases = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
    pooled, rows = [], []
    circ = build()
    for bs in biases:
        S = state_var(run(seconds=120.0, dt=DT_LONG, seed=1, bias=bs, circ=circ))
        m = measure(S, DT_LONG)
        rows.append({"bias": bs, "mean_S": float(S.mean()), "frac_high": m["frac_high"],
                     "state_hi": m["state_hi"], "state_lo": m["state_lo"],
                     "n_transitions": m["n_transitions"]})
        pooled.append(S)
    P = torch.cat(pooled)
    m = measure(P, DT_LONG)
    ok = (m["central_occupancy"] is not None and m["central_occupancy"] < 0.15
          and m["bimodality"] is not None and m["bimodality"] > 0.5556)
    return {"ok": bool(ok), "central_occupancy": m["central_occupancy"],
            "central_occupancy_bar": 0.15, "continuum_would_give": 0.333,
            "bimodality": m["bimodality"], "bimodality_bar": 0.5556,
            "uniform_gives": 0.5556, "two_point_gives": 1.0,
            "state_hi": m["state_hi"], "state_lo": m["state_lo"], "gap": m["gap"],
            "skew": m.get("skew"), "excess_kurtosis": m.get("excess_kurtosis"),
            "sweep": rows, "dt": DT_LONG, "seconds_per_level": 120.0,
            "rule": "pooled over a 9-level bias sweep: central-third occupancy < 0.15 "
                    "AND bimodality coefficient > 0.5556"}


# ======================================================================================
# N3
# ======================================================================================
def gate_transition():
    S = state_var(run(seconds=300.0, dt=DT, seed=1))
    m = measure(S, DT)
    taus = [NM.CONST[k] for k in ("tau_lc", "tau_raphe", "tau_tmn", "tau_ppt")]
    floor = min(taus)
    ratio = (m["crossing_s"] / m["dwell_s"]) if m["crossing_s"] else None
    ok = ratio is not None and ratio < 0.10
    return {"ok": bool(ok), "crossing_s": m["crossing_s"], "dwell_s": m["dwell_s"],
            "crossing_over_dwell": ratio, "bar": 0.10,
            "n_crossings": m.get("n_crossings"), "n_dwells": m.get("n_dwells"),
            "instrument_floor_s": floor,
            "crossing_over_floor": (m["crossing_s"] / floor) if m["crossing_s"] else None,
            "gap": m["gap"], "trigger": [HI, LO], "dt": DT,
            "rule": "mean 10-90% crossing < 0.10 x mean dwell",
            "note": "no crossing can be faster than the fastest switching population's "
                    "tau; that floor is quoted beside the measurement, and a crossing AT "
                    "the floor would be a measurement of tau and not of the switch"}


# ======================================================================================
# N4
# ======================================================================================
def gate_orexin():
    seeds = (1, 2, 3)
    arms = {}
    for label, od in (("intact", None), ("orx_silenced", ORX_SILENCE_DRIVE)):
        rows = []
        for s in seeds:
            tr = run(seconds=SECONDS, dt=DT_LONG, seed=s, orx_drive=od)
            m = measure(state_var(tr), DT_LONG)
            m["orx_rate"] = float(tr[NM.ORX].mean())
            rows.append(m)
        arms[label] = rows
    per_seed = []
    for i, s in enumerate(seeds):
        a, b = arms["intact"][i], arms["orx_silenced"][i]
        per_seed.append({"seed": s,
                         "transitions_intact": a["n_transitions"],
                         "transitions_silenced": b["n_transitions"],
                         "dwell_intact_s": a["dwell_s"], "dwell_silenced_s": b["dwell_s"],
                         "gap_intact": a["gap"], "gap_silenced": b["gap"],
                         "occupancy_intact": a["central_occupancy"],
                         "occupancy_silenced": b["central_occupancy"],
                         "more_transitions": b["n_transitions"] > a["n_transitions"],
                         "shorter_dwell": b["dwell_s"] < a["dwell_s"]})
    ok = all(r["more_transitions"] and r["shorter_dwell"] for r in per_seed)
    mean = lambda k: sum(r[k] for r in per_seed) / len(per_seed)  # noqa: E731
    return {"ok": bool(ok), "per_seed": per_seed,
            "mean_transitions_intact": mean("transitions_intact"),
            "mean_transitions_silenced": mean("transitions_silenced"),
            "mean_dwell_intact_s": mean("dwell_intact_s"),
            "mean_dwell_silenced_s": mean("dwell_silenced_s"),
            "orx_rate_intact": arms["intact"][0]["orx_rate"],
            "orx_rate_silenced": arms["orx_silenced"][0]["orx_rate"],
            "rule": "on EVERY seed: more transitions AND a shorter dwell when nm.orx is "
                    "silenced",
            "note": "the population, the wiring seed and the noise seeds are identical in "
                    "the two arms; only nm.orx's drive differs.  The across-seed spread is "
                    "small because the dwell is set by a deterministic adaptation cycle "
                    "with noise only jittering its phase -- three seeds here are three "
                    "phases of one clock, not three independent samples of a distribution"}


# ======================================================================================
# N5
# ======================================================================================
def _two_pop(mods):
    lc = [p for p in NM.pops() if p.id == NM.LC][0]
    tgt = [p for p in CX.pops() if p.id == N5_TARGET][0]
    return Circuit([lc, tgt], [], mods, seed=0, device="cpu")


N5_TARGET = "ctx.pericalcarine.E"
N5_DRIVE_SILENT, N5_DRIVE_SATURATED = -1.0, 1.2


def gate_modulation_reaches():
    edges = [m for m in NM.mods() if m.src == NM.LC and m.dst == N5_TARGET]
    beta_e = [m for m in edges if m.param == "beta"]
    sigma_e = [m for m in edges if m.param == "sigma"]
    assert beta_e and sigma_e, "the edges under test are not declared"

    # ---- a: the beta edge, measured as a SLOPE
    steps, burn = 6000, 1000
    gi = torch.Generator().manual_seed(21)
    sig = 0.35 + 0.25 * torch.randn(steps + burn, generator=gi)

    def slope_arm(mods, dlc):
        circ = _two_pop(mods)
        g = torch.Generator().manual_seed(4)
        tr, _ = circ.rollout(steps, 0.001, drive=lambda i: {NM.LC: dlc,
                                                            N5_TARGET: float(sig[i + burn])},
                             noise_gen=g, b=1, burn=burn)
        y = tr[N5_TARGET][0].mean(-1)
        x = sig[burn:burn + steps]
        yc, xc = y - y.mean(), x - x.mean()
        return {"modulator_rate": float(tr[NM.LC].mean()),
                "transfer_slope": float((yc * xc).sum() / (xc * xc).sum()),
                "mean_rate": float(y.mean())}

    a_lo = slope_arm(beta_e, N5_DRIVE_SILENT)
    a_hi = slope_arm(beta_e, N5_DRIVE_SATURATED)
    slope_ratio = a_hi["transfer_slope"] / max(1e-12, a_lo["transfer_slope"])

    # ---- b: the sigma edge, under a CONSTANT drive
    def sd_arm(mods, dlc):
        circ = _two_pop(mods)
        g = torch.Generator().manual_seed(4)
        tr, _ = circ.rollout(8000, 0.001, drive={NM.LC: dlc, N5_TARGET: 0.35},
                             noise_gen=g, b=1, burn=2000)
        y = tr[N5_TARGET][0]
        return {"modulator_rate": float(tr[NM.LC].mean()),
                "unit_temporal_sd": float(y.std(0).mean()), "mean_rate": float(y.mean()),
                "trace": y}

    b_lo = sd_arm(sigma_e, N5_DRIVE_SILENT)
    b_hi = sd_arm(sigma_e, N5_DRIVE_SATURATED)
    sd_ratio = b_hi["unit_temporal_sd"] / max(1e-12, b_lo["unit_temporal_sd"])

    # ---- c: THE CONTROL.  No Mod edges: the modulator must not reach by any other route.
    c_lo = sd_arm([], N5_DRIVE_SILENT)
    c_hi = sd_arm([], N5_DRIVE_SATURATED)
    control_identical = bool(torch.equal(c_lo.pop("trace"), c_hi.pop("trace")))
    b_lo.pop("trace"), b_hi.pop("trace")

    ok = slope_ratio >= 1.5 and sd_ratio <= 0.7 and control_identical
    return {"ok": bool(ok), "target": N5_TARGET,
            "edges_under_test": [{"param": m.param, "gain": m.gain, "baseline": m.baseline,
                                  "lo": m.lo, "hi": m.hi} for m in edges],
            "a_beta": {"silent": a_lo, "saturated": a_hi, "slope_ratio": slope_ratio,
                       "bar": 1.5,
                       "note": "the MEAN RATE falls as the modulator rises (%.4f -> %.4f) "
                               "because a steeper sigmoid at a subthreshold operating "
                               "point emits less; the GAIN is what rises, and measuring "
                               "the rate would have reported noradrenaline's sign "
                               "backwards" % (a_lo["mean_rate"], a_hi["mean_rate"])},
            "b_sigma": {"silent": b_lo, "saturated": b_hi, "sd_ratio": sd_ratio,
                        "bar": 0.7},
            "c_control": {"silent": c_lo, "saturated": c_hi,
                          "bit_identical": control_identical},
            "rule": "beta slope ratio >= 1.5 AND sigma sd ratio <= 0.7 AND the no-Mod "
                    "control bit-identical across modulator levels"}


# ======================================================================================
# N6
# ======================================================================================
NM_POPS = {"nm.lc", "nm.raphe", "nm.tmn", "nm.ppt", "nm.nbm", "nm.orx", "nm.vta", "nm.snc"}


def _reachable_in_nm(name: str) -> bool:
    """does the constant touch anything inside the nm-only circuit?

    A modulation gain whose only target is `thal.*` is inert HERE because the thalamus is
    not in this circuit.  That is a different fact from a mechanism that is not wired in,
    and the inert list is unreadable unless the two are separated.
    """
    base = build()
    alt = build({name: NM.CONST[name] * 1.5})
    if repr([p for p in base.pops.values()]) != repr([p for p in alt.pops.values()]):
        return True
    if repr(base.projs) != repr(alt.projs):
        return True
    return repr(base.mods) != repr(alt.mods)


SWEEP_SECONDS = 90.0


def gate_sensitivity():
    """NOISE OFF for this gate, and declared.

    The dwell is set by a deterministic adaptation cycle with the OU background only
    jittering its phase, so with `sigma_nm` = 0 the period is exact and a 90 s run measures
    it with no sampling error at all.  With noise on, a run short enough to sweep 56
    constants twice would carry a dwell error of order 15% and manufacture "this constant
    is not inert" out of it.  The one constant this cannot speak for is `sigma_nm` itself,
    which is swept from 0 to 0 here and is therefore reported as inert BY CONSTRUCTION --
    said here rather than left in the list to be misread.
    """
    def dwell(c):
        c = {**(c or {}), "sigma_nm": 0.0}
        S = state_var(run(c, seconds=SWEEP_SECONDS, dt=DT_SWEEP, seed=1))
        m = measure(S, DT_SWEEP)
        return (m["dwell_s"], m["n_transitions"], m.get("dwell_is_lower_bound", False),
                m["frac_high"])

    d0, n0, _, f0 = dwell(None)
    rows = []
    for name, v in NM.CONST.items():
        lo, lo_n, lo_lock, lo_f = dwell({name: v * 0.5})
        hi, hi_n, hi_lock, hi_f = dwell({name: v * 1.5})
        # A LOCKED FLIP-FLOP IS NOT AN UNCHANGED ONE, and until this line existed the two
        # were indistinguishable.  `measure` reports the whole run as the dwell when the
        # network never flips, so `d_ppt` -- which LOCKS the switch into the aminergic
        # state at 0.5x and into the cholinergic state at 1.5x, destroying the oscillation
        # in both directions -- gave 90.0 s at both ends, a span of 0.000, and was listed
        # as INERT.  It is the single most consequential constant in the file.  Same shape
        # as a guard that cannot fire looking exactly like a guard that passed.
        locked = lo_lock or hi_lock
        span = float("inf") if locked else abs(hi - lo) / max(1e-9, d0)
        rows.append({"const": name, "value": v, "dwell_at_half_s": lo,
                     "dwell_at_1p5x_s": hi, "relative_span": span, "locks": locked,
                     "transitions_half": lo_n, "transitions_1p5x": hi_n,
                     "locked_at_half": lo_lock, "locked_at_1p5x": hi_lock,
                     # the dwell is the CLOCK; the fraction of time high is the BALANCE.
                     # A constant can be flat on one and move the other, and calling that
                     # "inert" would be false -- `d_tmn` is exactly it (3.8% on the dwell,
                     # 0.465 -> 0.671 on the balance).
                     "frac_high_half": lo_f, "frac_high_1p5x": hi_f,
                     "balance_span": abs(hi_f - lo_f),
                     "reachable_in_nm_circuit": _reachable_in_nm(name)})
    rows.sort(key=lambda r: -r["relative_span"])
    clock = {r["const"]: r["relative_span"] for r in rows if r["const"] in CLAIMED_CLOCK}
    inert = [r["const"] for r in rows
             if r["relative_span"] < 0.05 and r["balance_span"] < 0.05]
    inert_reach = [r["const"] for r in rows
                   if r["const"] in inert and r["reachable_in_nm_circuit"]]
    locks = [r["const"] for r in rows if r["locks"]]
    balance_only = [r["const"] for r in rows
                    if r["relative_span"] < 0.05 <= r["balance_span"]]
    ok = all(v > 0.20 for v in clock.values()) and len(clock) == len(CLAIMED_CLOCK)
    return {"ok": bool(ok), "baseline_dwell_s": d0, "baseline_transitions": n0,
            "claimed_clock_relative_span": clock, "bar": 0.20,
            "rule": "adapt_tau, adapt_g and orx_beta_gain must each move the mean dwell by "
                    "> 20% over a 3x sweep",
            "inert_below_5pct": inert,
            "inert_and_reachable_in_nm_circuit": inert_reach,
            "locks_the_flipflop": locks,
            "flat_on_dwell_but_moves_the_balance": balance_only,
            "n_constants": len(rows), "dt": DT_SWEEP, "seconds": SWEEP_SECONDS,
            "noise": "off (sigma_nm forced to 0) so the dwell is a deterministic period",
            "inert_by_construction": ["sigma_nm"], "sweep": rows}


# ======================================================================================
# N7
# ======================================================================================
def gate_one_edge_one_factor():
    mods_by_name, unreadable = load_structures()
    names = list(mods_by_name)
    seen, dup, all_pops = {}, [], set()
    seen_p, dup_p = {}, []
    for n, m in mods_by_name.items():
        all_pops |= {p.id for p in m.pops()}
    for n, m in mods_by_name.items():
        for e in m.mods():
            k = (e.src, e.dst, e.param)
            if k in seen and seen[k] != n:
                dup.append({"edge": f"{e.src}->{e.dst}:{e.param}",
                            "declared_by": [seen[k], n]})
            seen.setdefault(k, n)
        # and the same check for PROJECTIONS, which is the form this actually took: two
        # Proj with the same endpoints do not raise, Circuit.step SUMS their contributions,
        # and the pathway is silently twice as strong as either module declares.
        for e in m.external():
            k = (e.src, e.dst, e.sign)
            if k in seen_p and seen_p[k] != n:
                dup_p.append({"edge": e.key, "declared_by": [seen_p[k], n]})
            seen_p.setdefault(k, n)
    live_structures = {p.split(".", 1)[0] for p in all_pops}
    named = set()
    for e in NM.external():
        named |= {e.src, e.dst}
    for e in NM.mods():
        named |= {e.src, e.dst}
    dead = sorted(p for p in named
                  if p.split(".", 1)[0] in live_structures and p not in all_pops)
    orphan = sorted({p for p in named if p.split(".", 1)[0] not in live_structures})
    return {"ok": not dup and not dup_p and not dead and not unreadable,
            "structures_on_disk": names, "structures_unreadable": unreadable,
            "duplicate_mod_triples": dup,
            "duplicate_external_projections": dup_p,
            "n_mod_edges_total": len(seen), "n_external_edges_total": len(seen_p),
            "dead_ids_in_live_structures": dead,
            "ids_in_structures_not_yet_written": orphan,
            "rule": "no (src, dst, param) Mod and no (src, dst, sign) external Proj "
                    "declared by two modules; every foreign id inside an EXISTING "
                    "structure is one of its populations"}


# ======================================================================================
def orphan_report():
    """what `ibm.brain.collect()` would report for this module's edges, printed for the
    record.  Not a gate: an orphan is the correct outcome, not a failure."""
    mods_by_name, unreadable = load_structures()
    have = set()
    for m in mods_by_name.values():
        have |= {p.id for p in m.pops()}
    ext = list(NM.external())
    kept = [e.key for e in ext if e.src in have and e.dst in have]
    orph = [e.key for e in ext if e.src not in have or e.dst not in have]
    mods = list(NM.mods())
    mk = [f"{m.src}->{m.dst}:{m.param}" for m in mods if m.src in have and m.dst in have]
    mo = [f"{m.src}->{m.dst}:{m.param}" for m in mods
          if m.src not in have or m.dst not in have]
    return {"structures_present": list(mods_by_name), "structures_missing": B.missing(),
            "structures_unreadable": unreadable,
            "external_kept": kept, "external_orphan": orph,
            "n_mods": len(mods), "mods_kept": len(mk), "mods_orphan": len(mo),
            "mods_orphan_list": mo,
            "note": "an orphan is the correct outcome for an edge into a structure that "
                    "is not written yet, not a failure.  collect() reports both Proj and "
                    "Mod orphans as of the fix in ibm/brain/__init__.py"}


GATES = [("N0_bounded", gate_bounded),
         ("N1_idempotent", gate_idempotent),
         ("N2_two_states", gate_two_states),
         ("N3_transition", gate_transition),
         ("N4_orexin", gate_orexin),
         ("N5_modulation_reaches", gate_modulation_reaches),
         ("N6_sensitivity", gate_sensitivity),
         ("N7_one_edge_one_factor", gate_one_edge_one_factor)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/gate_brain_neuromodulators.json")
    ap.add_argument("--only", default="", help="comma-separated gate prefixes, e.g. N2,N4")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pick = [s.strip() for s in a.only.split(",") if s.strip()]
    order = [g for g in GATES if not pick or any(g[0].startswith(p) for p in pick)]

    rec = {"script": "scripts/gate_brain_neuromodulators.py", "module": NM.__name__,
           "dt": DT, "dt_sweep": DT_SWEEP, "trigger": [HI, LO],
           "subset": pick or "all", "constants": dict(NM.CONST), "gates": {}}

    def save():
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=str)

    # the cheap summary FIRST, before anything that can raise (CLAUDE.md)
    rec["orphans"] = orphan_report()
    save()
    o = rec["orphans"]
    print(f"structures on disk: {', '.join(o['structures_present'])}")
    print(f"still missing:      {', '.join(o['structures_missing'])}")
    if o["structures_unreadable"]:
        print(f"UNREADABLE (edited under us): {o['structures_unreadable']}")
    print(f"external edges: {len(o['external_kept'])} kept, "
          f"{len(o['external_orphan'])} orphan;  Mod edges: {o['mods_kept']} kept, "
          f"{o['mods_orphan']} orphan (of {o['n_mods']})\n", flush=True)

    ok_all, t0 = True, time.time()
    for name, fn in order:
        t = time.time()
        r = fn()
        rec["gates"][name] = r
        save()
        ok_all &= bool(r.get("ok"))
        head = {k: v for k, v in r.items()
                if k not in ("sweep", "cases", "per_seed", "mean_rates", "a_beta",
                             "b_sigma", "c_control", "edges_under_test", "note", "rule")}
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {name}  ({time.time()-t:.0f}s)")
        print(f"       {head}", flush=True)
        if name == "N2_two_states":
            for row in r["sweep"]:
                print(f"       bias {row['bias']:+.2f}  mean S {row['mean_S']:+.3f}  "
                      f"frac high {row['frac_high']:.2f}  "
                      f"hi {row['state_hi'] if row['state_hi'] is None else round(row['state_hi'],3)}  "
                      f"lo {row['state_lo'] if row['state_lo'] is None else round(row['state_lo'],3)}  "
                      f"flips {row['n_transitions']}", flush=True)
        if name == "N4_orexin":
            for row in r["per_seed"]:
                print(f"       seed {row['seed']}  transitions "
                      f"{row['transitions_intact']} -> {row['transitions_silenced']}  "
                      f"dwell {row['dwell_intact_s']:.2f} -> {row['dwell_silenced_s']:.2f} s"
                      f"  gap {row['gap_intact']:.2f} -> {row['gap_silenced']:.2f}",
                      flush=True)
        if name == "N5_modulation_reaches":
            ab, bs, cc = r["a_beta"], r["b_sigma"], r["c_control"]
            print(f"       beta : slope {ab['silent']['transfer_slope']:.4f} -> "
                  f"{ab['saturated']['transfer_slope']:.4f}  "
                  f"({ab['slope_ratio']:.2f}x, bar 1.5)   mean rate "
                  f"{ab['silent']['mean_rate']:.4f} -> {ab['saturated']['mean_rate']:.4f}")
            print(f"       sigma: unit sd {bs['silent']['unit_temporal_sd']:.5f} -> "
                  f"{bs['saturated']['unit_temporal_sd']:.5f}  "
                  f"({bs['sd_ratio']:.2f}x, bar 0.7)")
            print(f"       control (no Mod): bit-identical "
                  f"{cc['bit_identical']}", flush=True)
        if name == "N6_sensitivity":
            for row in r["sweep"][:8]:
                print(f"       {row['const']:22s} {row['dwell_at_half_s']:6.2f} -> "
                      f"{row['dwell_at_1p5x_s']:6.2f} s   span {row['relative_span']:6.1%}",
                      flush=True)
            print(f"       LOCKS the flip-flop at one or both ends: "
                  f"{', '.join(r['locks_the_flipflop']) or 'none'}")
            print(f"       flat on the dwell but moves the BALANCE: "
                  f"{', '.join(r['flat_on_dwell_but_moves_the_balance']) or 'none'}")
            print(f"       inert on BOTH (< 5% over a 3x sweep), "
                  f"{len(r['inert_below_5pct'])} of "
                  f"{r['n_constants']}: {', '.join(r['inert_below_5pct']) or 'none'}")
            print(f"       of those, REACHABLE inside the nm-only circuit: "
                  f"{', '.join(r['inert_and_reachable_in_nm_circuit']) or 'none'}",
                  flush=True)
    rec["all_gates_ok"] = ok_all
    rec["wall_seconds"] = time.time() - t0
    save()
    print(f"\n{'ALL GATES PASS' if ok_all else 'SOME GATES FAILED'} "
          f"({time.time()-t0:.0f}s) -- wrote {a.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
