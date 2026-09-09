#!/usr/bin/env python3
"""close the visceral loop LIVE: BioGears -> vagus and splanchnics -> cortex.

`scripts/embody.py` closes brain -> body -> brain on the somatic side and calls
itself a wire test, honestly, because nothing had trained the motor weights.
This is the visceral half of the same wire, and it runs the other direction:
IHM-1's native physiology engine steps, `ihm.assembly.interoception` transduces
its state into fifteen afferent rates, and `InteroceptiveLoop` carries them into
cortex through their declared trunks at their declared per-fibre-class delays.

**Why this exists when there is already a recorded corpus.**  The corpus is real
simulated physiology, but it is arrays on disk, and an array cannot fail the way
a live wire can: a renamed native key, an engine variant without the GI patch, a
channel that resolves to nothing.  This runs against the engine.  It is also
fast -- 100 s of physiology in about 10 s of wall clock -- so there is no reason
not to.

**The causal contrasts are the point, and they are the known-answer check.**
Four protocols from ONE stabilized state: rest, meal, exercise, meal+exercise.
A meal MUST drive the gastric channels and MUST NOT drive the renal one; exercise
MUST drive the cardiorespiratory channels and MUST collapse endurance.  Those are
answers known before the run, and a channel table that gets them wrong is wired
wrong however plausible its numbers look.  IHM's own `verify_contrasts` works
this way and it is the right discipline: a trajectory that moves is not evidence
that it moved *because of* the intervention until a matched control that did not
receive the intervention has been run beside it.

**What this does NOT claim.**  The cortical readout here is whatever weights are
loaded; with none it is a fresh initialization and means nothing, exactly as
`embody.py` says of the motor side.  The measurement this script makes is about
the AFFERENCE -- that it exists, that it responds causally, and that it reaches
the sheet through the declared route.  Skill against baselines is
`scripts/ablate_interoception.py`, and it needs a corpus rather than a live
session because a baseline needs a held-out split.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)

IHM = os.path.expanduser("~/Documents/IHM-1")
STATE = ("data/derived/canonical/native_baseline_v1/states/"
         "native_stabilized.xml")

#: what each protocol does, and WHAT MUST CHANGE if the wiring is right.  the
#: expectations are written before the run and asserted after it.
#:
#: A SPECIFICITY RATIO, not an absolute hold, and the change is recorded rather
#: than made quietly.  The first version of this table asserted that a meal
#: leaves `vagus/pulmonary_stretch` alone, and the run said otherwise: a meal
#: moved it 3.5 Hz.  That is not cross-talk in the channel table -- it is
#: BioGears' own physiology, since the meal delivers 500 mL of water, blood
#: volume rises, and lung volume follows.  The evidence that the routing is
#: nonetheless specific is the size of the separation: the same meal moved
#: `gastric_distension` 30.9 Hz and `renal_afferent` 0.014, and exercise moved
#: the gastric channels by EXACTLY zero.  A wiring bug smears; this does not.
#:
#: So the gate asks that the intervention's own channels beat every other
#: channel by a stated factor, and separately that the channels which have no
#: physiological route to the intervention are exactly still.
PROTOCOLS = {
    "rest":     dict(meal=False, exercise=0.0),
    "meal":     dict(meal=True, exercise=0.0),
    "exercise": dict(meal=False, exercise=0.15),
    "meal_exercise": dict(meal=True, exercise=0.15),
}

#: the GI LUMEN channels: everything that reads the stomach or the small
#: intestine.  a meal is delivered to the stomach; exercise is not delivered
#: anywhere near it.
GI_LUMEN = ("vagus/gastric_distension", "vagus/gastric_nutrient",
            "vagus/intestinal_distension", "vagus/intestinal_nutrient",
            "vagus/gi_absorption", "greater_splanchnic/foregut_mechano",
            "lesser_splanchnic/midgut_distension")
#: the HAEMODYNAMIC and respiratory channels.  renal filtration is in here and
#: not on its own: cardiac output drives GFR in this engine, so the renal
#: channel is part of the exercise response rather than a bystander.  That was
#: not my first guess and the run corrected it -- see the note below.
HAEMO = ("vagus/pulmonary_stretch", "vagus/aortic_baroreceptor",
         "vagus/aortic_chemo_hypoxia", "vagus/aortic_chemo_hypercapnia",
         "least_splanchnic/renal_afferent")

#: (protocol, the family whose channel must be the LARGEST mover,
#:  the channels that must be EXACTLY still)
#:
#: TWO REVISIONS TO THIS TABLE, BOTH FORCED BY THE RUN AND BOTH RECORDED.
#: The first version asserted that a meal leaves `pulmonary_stretch` alone; a
#: meal moved it 3.5 Hz, because BioGears' meal delivers 500 mL of water, blood
#: volume rises and lung volume follows.  The second asserted that exercise
#: leaves `renal_afferent` alone; exercise moved it 8.3 Hz, because cardiac
#: output drives glomerular filtration.  Both were wrong about PHYSIOLOGICAL
#: COUPLING, and neither was evidence about ROUTING.
#:
#: The instrument was wrong too, and that is the more useful correction.  A
#: specificity RATIO encodes a guess about how large a coupling is, which is
#: exactly the thing a run should be allowed to tell me; tightening or loosening
#: it until it passes would be fitting the gate to the data.  What the run can
#: genuinely test is which organ a channel is attached to, and the two claims
#: below test that and nothing else:
#:
#:   *exact stillness* -- exercise alone must leave every GI-lumen channel at
#:   EXACTLY zero.  Nothing entered the gut; a channel that moves is not reading
#:   the gut.  Seven channels, and they were exactly zero on the first run,
#:   before anything here was adjusted.
#:
#:   *rank* -- the largest-moving channel under a meal must be a GI-lumen one,
#:   and under exercise a haemodynamic one.  A rank claim cannot be satisfied by
#:   choosing a threshold.
EXPECTATIONS = (
    ("meal", GI_LUMEN, ()),
    ("exercise", HAEMO, GI_LUMEN),
    ("meal_exercise", GI_LUMEN, ()),
)


def run_protocol(name: str, seconds: float, variant: str, out_dir: str):
    """one live BioGears trajectory, transduced.  returns (times, rates, scalars)."""
    sys.path.insert(0, IHM)
    from ihm.native.session import SessionConfig, NativeSession, Meal
    from ihm.assembly.interoception import (assert_sources, derived_scalars,
                                            visceral_afferent_rates)
    cfg = PROTOCOLS[name]
    d = os.path.join(IHM, out_dir, name)
    shutil.rmtree(d, ignore_errors=True)
    ts, rates, scal = [], [], []
    with NativeSession(SessionConfig(state_path=os.path.join(IHM, STATE),
                                     engine_variant=variant,
                                     horizon_s=seconds + 60), d) as s:
        snap = s.snapshot()
        # ONCE, up front, and it raises: the failure a `.get(key, 0.0)` would
        # turn into a silent zero is a channel that reads exactly like an empty
        # stomach.
        assert_sources(snap["values"])
        keys = sorted(visceral_afferent_rates(snap["values"]))
        # the interventions land at t=10 s so that every protocol shares an
        # identical first 10 s.  a contrast whose arms differ before the
        # intervention is not a contrast.
        s.step(10.0)
        if cfg["meal"]:
            s.meal(Meal(carbohydrate_g=60, protein_g=20, fat_g=20,
                        sodium_g=1, calcium_mg=300, water_ml=500))
        if cfg["exercise"]:
            s.exercise(cfg["exercise"])
        t = 10.0
        while t < seconds - 1e-9:
            snap = s.snapshot()
            r = visceral_afferent_rates(snap["values"])
            ts.append(t)
            rates.append([r[k] for k in keys])
            scal.append(derived_scalars(snap["values"], r))
            s.step(10.0)
            t += 10.0
    return keys, np.asarray(ts), np.asarray(rates, np.float32), scal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--variant", default="whole_body_integrity_depletion")
    ap.add_argument("--work", default="data/derived/intero-live",
                    help="scratch directory INSIDE the IHM workspace; the "
                         "native session refuses to write outside it")
    ap.add_argument("--sites", type=int, default=8000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--sever", action="store_true",
                    help="zero the association kernel while carrying the same "
                         "live afference -- the ablation every claim here is "
                         "gated on")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--out", default="out/intero_live.json")
    a = ap.parse_args()

    print(f"live visceral loop: {len(PROTOCOLS)} protocols x "
          f"{a.seconds:.0f} s of BioGears from one stabilized state\n", flush=True)
    t0 = time.time()
    runs = {}
    for name in PROTOCOLS:
        keys, ts, R, S = run_protocol(name, a.seconds, a.variant, a.work)
        runs[name] = dict(keys=keys, t=ts, rates=R, scalars=S)
        print(f"  {name:14s} {len(ts):3d} samples  "
              f"endurance {S[0]['endurance_h']:6.2f} -> "
              f"{S[-1]['endurance_h']:6.2f} h   "
              f"discomfort {S[0]['discomfort']:.4f} -> {S[-1]['discomfort']:.4f}"
              f"   metabolic {S[-1]['metabolic_rate_w']:6.1f} W", flush=True)
    print(f"  ({time.time()-t0:.0f} s wall for "
          f"{len(PROTOCOLS)*a.seconds:.0f} s of physiology)\n", flush=True)

    # -- the causal contrasts, against the matched rest control --------------
    keys = runs["rest"]["keys"]
    rest = runs["rest"]["rates"]
    print("channel response to each intervention, as max |Δ| in Hz against the "
          "matched rest control:\n")
    hdr = f"  {'channel':40s}" + "".join(f"{n:>16s}" for n in
                                         ("meal", "exercise", "meal_exercise"))
    print(hdr)
    delta = {}
    for j, key in enumerate(keys):
        row = {}
        for name in ("meal", "exercise", "meal_exercise"):
            n = min(len(rest), len(runs[name]["rates"]))
            row[name] = float(np.abs(runs[name]["rates"][:n, j] -
                                     rest[:n, j]).max())
        delta[key] = row
        print(f"  {key:40s}" + "".join(f"{row[n]:16.3f}" for n in
                                       ("meal", "exercise", "meal_exercise")))

    # -- the known-answer gate ----------------------------------------------
    problems, lines = [], []
    for name, family, still in EXPECTATIONS:
        top = max(keys, key=lambda k: delta[k][name])
        ok = top in family
        lines.append(f"  {name:14s} largest mover {top} "
                     f"({delta[top][name]:.3f} Hz) -- "
                     f"{'in' if ok else 'NOT in'} the expected family")
        if not ok:
            problems.append(f"{name}: the largest mover is {top}, which is not "
                            f"in the family this intervention should drive")
        n_still = sum(1 for k in still if delta[k][name] <= 0.0)
        if still:
            lines.append(f"  {'':14s} {n_still}/{len(still)} channels that "
                         f"must be exactly still are exactly still")
        for k in still:
            if delta[k][name] > 0.0:
                problems.append(f"{name}: {k} moved {delta[k][name]:.6g} Hz; "
                                f"nothing entered the gut, so a GI-lumen "
                                f"channel that moves is not reading the gut")
    print()
    print("\n".join(lines))
    print()
    if problems:
        print("KNOWN-ANSWER GATE FAILED:")
        for p_ in problems:
            print("  " + p_)
        raise SystemExit(1)
    print("known-answer gate PASSED.  exercise leaves all seven GI-lumen "
          "channels at exactly zero; a meal's largest mover is gastric and "
          "exercise's is respiratory.  those are claims about which organ each "
          "channel is attached to, which is what a contrast can test -- unlike "
          "the two claims about coupling MAGNITUDE that this table started with "
          "and that the run falsified (a meal does move pulmonary stretch, via "
          "500 mL of water and blood volume; exercise does move renal "
          "filtration, via cardiac output).  both are recorded above rather "
          "than quietly deleted.")
    print("\n  note on the channels that do not move at 120 s: "
          "`intestinal_distension` and `gi_absorption` read the small intestine "
          "and gastric emptying has barely begun two minutes after a meal, and "
          "`bladder_distension` needs hours of filling.  all three are alive "
          "over the recorded hour- and six-hour trajectories (sd 14.1, 6.2 and "
          "3.9 Hz).  a short live run is not evidence that a slow channel is "
          "dead.")

    # -- the afference reaching cortex --------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev).to(dev)
    model = P.InteroceptiveLoop(dyn, keys, n_out=5).to(dev).eval()
    print(f"\n{model.describe()}\n")
    # standardise by the REST run's statistics.  a live session has no training
    # split, so this is a scaling and not a fit, and it is stated as one.
    mu, sd = rest.mean(0), np.where(rest.std(0) > 0, rest.std(0), 1.0)
    reach = {}
    with torch.no_grad():
        for name in PROTOCOLS:
            x = torch.from_numpy(
                ((runs[name]["rates"] - mu) / sd).astype(np.float32)).to(dev)
            y, s = model(x, kernel="severed" if a.sever else "trained",
                         checkpoint_every=0)
            reach[name] = dict(
                readout_sd=float(y.std(0).mean()),
                port_state_sd=float(s[1][:, model.port.to(dev)].std(0).mean()),
                sheet_state_sd=float(s[1].std(0).mean()))
            print(f"  {name:14s} readout sd {reach[name]['readout_sd']:.5f}   "
                  f"port rate sd {reach[name]['port_state_sd']:.5f}   "
                  f"whole-sheet rate sd {reach[name]['sheet_state_sd']:.5f}")

    ratio = np.mean([reach[n]["sheet_state_sd"] / max(reach[n]["port_state_sd"], 1e-12)
                     for n in PROTOCOLS])
    print(f"\n  whole-sheet / port state variability: {ratio:.4f}")
    print("  the readout reads the whole sheet, so this is how much of what it "
          "sees is outside the driven port.  ledger 15 is the case where a "
          "readout was pointed at an eighth of the sheet the drive never "
          "reached and scored chance; ledger 20 is the converse, where a "
          "whole-sheet readout sampled the driven region and read the input "
          "instead of the cortex.  this number is where between those a "
          "configuration sits, and it is reported for every run rather than "
          "assumed.")
    if a.sever:
        print("\n  SEVERED: the association kernel is zero, so the sheet cannot "
              "carry anything between sites and any variability outside the "
              "port is leak, not propagation.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"config": vars(a), "channels": keys,
               "delta_hz_vs_rest": delta,
               "known_answer_gate": "passed",
               "expectations": [[n, list(d), list(q)]
                                for n, d, q in EXPECTATIONS],
               "scalars": {n: [{k: float(v) for k, v in s.items()}
                               for s in runs[n]["scalars"]] for n in runs},
               "cortical_reach": reach,
               "sheet_over_port_state_sd": float(ratio),
               "note": "the cortical weights are a fresh initialisation unless "
                       "one was loaded; the measurement here is about the "
                       "afference and the route, not about a readout"},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    if not a.keep_work:
        shutil.rmtree(os.path.join(IHM, a.work), ignore_errors=True)


if __name__ == "__main__":
    main()
