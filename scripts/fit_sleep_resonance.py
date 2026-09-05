#!/usr/bin/env python
"""can the declared thalamocortical resonator express a sleep spindle?

`sleep_dynamics` says, under `constrained`, that this materialization pins
"spindle density, duration and frequency".  the process it names for the
thalamocortical loop is `thalamocortical_coupling`, and the implementation a
scalp-spectrum materialization would select is `alpha_resonator` -- two
parameters, a centre frequency and a Q, whose own docstring says it is offered
"as a deliberately cheap alternative to the delay-and-feedback form" because "an
eeg alpha peak constrains a centre frequency and a width, and nothing else".

its priors are `f0_hz` = normal(10.0, 1.2) and `q` = lognormal(4.0, x/ 6.0),
sourced to "individual alpha peak frequency, 8-13 Hz".  a sleep spindle is at
11-16 Hz.  so the declaration contains a tension it has never been made to face:
the same implementation is claimed to carry both the waking alpha rhythm and the
NREM spindle, and its prior is centred on only one of them.

this script makes it face that.

    fit     `f0_hz` and `q` of the REGISTERED `alpha_resonator` transfer function
            against the 9-20 Hz spectrum of N2 sleep, on TRAIN subjects
    test    score the fitted posterior and the untouched prior on the N2 spectra
            of HELD-OUT subjects, on identical terms
    falsify three predictions that could each fail and would each matter

the model of the spectrum is multiplicative and that is a choice with content:

    S(f) = g * f^(-e) * |H(f; f0, q)|^2

a linear loop driven by broadband input produces its input's spectrum SHAPED by
|H|^2, not its input's spectrum PLUS a bump.  writing it as a sum would be
describing a separate spindle generator sitting beside the background; writing it
as a product is the statement `thalamocortical_coupling` actually makes, that the
loop is what the cortex's own drive passes through.  `f^(-e)` is the drive, and
`e` and the overall scale `g` are BOTH nuisances profiled identically for the
prior and for the posterior, so nothing below is a comparison between a model with
a free exponent and one without.

the split is BY SUBJECT and not by night.  sleep-edfx records two consecutive
nights per person; the two share a skull, an electrode placement and an individual
spindle frequency, so splitting by night would measure night-to-night stability in
one person and report it as generalization.

run:  ./.venv/bin/python scripts/fit_sleep_resonance.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ibm
from eval_sleep_state import EPOCH_S, STAGES, evidence_by_stage, pairs
from fit_neural_spectra import paired, rule
from ibm.forge.fit import Task, fit_map
from ibm.forge.priors import ParameterSpace, assemble
from ibm.forge.spectra import SpectralEvidence, local_root
from ibm.processes.neural import alpha_resonance_transfer
from ibm.registry import REGISTRY
from ibm.vocabulary import Band, Tying

MODEL_ID = "sleep_dynamics"
SOURCE = "sleep-edfx"
IMPL = "thalamocortical_coupling:alpha_resonator"

#: 9-20 Hz.  the low edge is above the alpha band's floor and well above the
#: slow-oscillation and theta structure that dominate NREM, so nothing in the
#: likelihood is the delta power that `eval_sleep_state.py` already showed can
#: masquerade as an aperiodic change.  the high edge stops below the beta/EMG
#: rise and comfortably below the 50 Hz nyquist of a 100 Hz recording.  the band
#: is chosen to contain both the waking alpha peak and the spindle peak, because a
#: band that contained only one of them would decide the falsification test by
#: construction.
BAND = Band(9.0, 20.0)

#: the drive exponent is profiled over this grid rather than optimized, because it
#: is a nuisance and a grid maximum is reproducible where an inner optimizer's
#: convergence is not.  the range is wide enough to contain a flat drive (0) and a
#: steep one (4); the fitted values are printed so a boundary hit is visible.
E_GRID = np.linspace(-1.0, 5.0, 61)


# ---------------------------------------------------------------------------
# the model spectrum
# ---------------------------------------------------------------------------


def parameter_space() -> ParameterSpace:
    """`alpha_resonator`'s own declared priors, collapsed to one global value each.

    pulled out of `REGISTRY` rather than restated, so this script tests the
    declaration and cannot drift from it.  the declared tying is PER_PARTITION --
    one resonator per thalamic nucleus and cortical area -- and two bipolar
    derivations identify exactly one, so it is collapsed to GLOBAL here.  that is
    a narrowing of the declaration and it is what `sleep_dynamics` itself predicts
    under `prior_dominated`: "spatial structure of anything.  a two-channel
    montage constrains no map".

    `gain` is dropped from the space entirely.  it is declared `weak(1.0, 5.0)`
    and it is exactly degenerate with the overall psd scale, which the gamma
    likelihood profiles in closed form; carrying it would put a parameter in the
    optimizer whose posterior is its prior times an amplifier.
    """
    src = REGISTRY.implementations[IMPL]
    proc, name = IMPL.split(":")
    impl = replace(src, params={k: v for k, v in src.params.items() if k != "gain"},
                   tying=Tying.GLOBAL)
    return assemble(implementations=[impl])


def model_psd(freqs: np.ndarray, f0: float, q: float, e: float) -> np.ndarray:
    """g f^-e |H(f; f0, q)|^2, at unit g, on the evidence's own frequency axis.

    `H` is `ibm.processes.neural.alpha_resonance_transfer`, which is the transfer
    the registry's `alpha_resonator` implementation points at -- not a
    reimplementation of it.  the difference matters: if it were reimplemented, the
    thing fitted here and the thing a materialization of `sleep_dynamics` would
    actually run could disagree and nothing would notice.
    """
    class _B:                                   # the transfer only reads `omega`
        omega = 2.0 * np.pi * freqs
    h = alpha_resonance_transfer(_B, f0_hz=float(f0), q=float(q), gain=1.0)
    f = np.maximum(freqs, 1e-6)
    return np.abs(h) ** 2 * f ** (-float(e))


def profile_loglik(ev: SpectralEvidence, f0: float, q: float
                   ) -> tuple[float, float, float]:
    """log p(psd | f0, q), maximized over the drive exponent and the overall scale.

    the scale goes in closed form through `SpectralEvidence.log_likelihood`, which
    is the exact gamma maximum; the exponent goes over `E_GRID`.  both nuisances
    are profiled the SAME way for the prior and for the posterior, which is the
    only arrangement under which the difference between the two is about `f0` and
    `q`.
    """
    idx = ev.basis.band_indices(BAND)
    f = ev.basis.freqs_hz[idx]
    best, best_e, best_r2 = -np.inf, float("nan"), float("nan")
    for e in E_GRID:
        m = np.zeros(ev.basis.k)
        m[idx] = model_psd(f, f0, q, e)
        v = ev.log_likelihood(m, band=BAND)
        if v > best:
            best, best_e = v, float(e)
            best_r2 = ev.r2_log(m, band=BAND)
    return best, best_e, best_r2


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def fit_stage(evs: list[SpectralEvidence], space: ParameterSpace,
              theta0: np.ndarray | None = None) -> tuple[np.ndarray, object]:
    """one (f0, q) for a whole set of recordings, through `ibm.forge.fit`.

    the likelihood is a sum over recordings, which is §4's product form with one
    factor per recording; nothing is averaged and a recording with more surviving
    windows carries more weight, because its psd estimate has more degrees of
    freedom and `SpectralEvidence` knows exactly how many.
    """
    bf, bq = space["f0_hz"], space["q"]

    def logp(theta: np.ndarray) -> float:
        f0, q = float(theta[bf.slice][0]), float(theta[bq.slice][0])
        if not (1.0 < f0 < 45.0 and 0.2 < q < 200.0):
            return -1e12
        v = sum(profile_loglik(e, f0, q)[0] for e in evs)
        return v if np.isfinite(v) else -1e12

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return fit_map(space, [Task(name=f"{SOURCE}.n2.train", logp=logp,
                                    moves=(bf.key, bq.key), source=SOURCE, kind="fit",
                                    note=f"{len(evs)} recordings")],
                       theta0=theta0, max_iter=200)


def per_recording(evs: list[SpectralEvidence], space: ParameterSpace) -> list[dict]:
    """refit (f0, q) against ONE recording, for the within-night contrasts.

    the falsification tests are within-subject differences, so they need a number
    per (night, stage) rather than a cohort value.  the prior is the declared one,
    unwidened: `alpha_resonator` says f0 is 10 +/- 1.2 Hz and this test is asking
    whether NREM pulls it off that, so widening the prior first would be assuming
    the answer.
    """
    out = []
    for ev in evs:
        t, _ = fit_stage([ev], space)
        f0 = float(t[space["f0_hz"].slice][0])
        q = float(t[space["q"].slice][0])
        ll, e, r2 = profile_loglik(ev, f0, q)
        out.append({"source": ev.source, "f0_hz": f0, "q": q, "drive_exponent": e,
                    "r2": r2, "loglik": ll, "n_windows": ev.n_windows})
    return out


def rec_key(name: str) -> str:
    """`SC4001E0-PSG/N2` -> `SC4001`, the night; the subject is its first five."""
    return name.split("/")[0][:6]


def subject_of(night: str) -> str:
    """`SC4001` -> `SC400`.  sleep-edfx numbers the subject in characters 3-4 and
    the night in character 5, so two nights of one person differ in one digit and
    a split that ignored it would leak a skull across the split."""
    return night[:5]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recordings", type=int, default=40)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    ibm.load_all(seal=True, strict=True)
    from ibm.materialize.library import MODELS
    model = MODELS[MODEL_ID]

    rule(f"the model: {MODEL_ID}")
    print(f"{SOURCE} is a declared FIT source of `{MODEL_ID}`: "
          f"{SOURCE in model.fit_sources}")
    print(f"`thalamocortical_coupling` is among its declared processes: "
          f"{'thalamocortical_coupling' in model.request.processes}")
    print("\nthe claim under test, quoted from the model's own `constrained`:")
    for c in model.constrained:
        print(f"  + {c}")
    print("and what it says will not move:")
    for c in model.prior_dominated:
        print(f"  - {c}")
    print("\nnote the first `constrained` entry credits `mass`'s expert spindle annotations "
          "for the\nspindle numbers, and `mass` is not held here.  so this script tests the "
          "spindle claim\nwith sleep-edfx alone, which has stage labels and no event labels: "
          "it can ask where the\nsigma resonance sits and how sharp it is, and it cannot ask "
          "about spindle DENSITY at\nall.  one third of that `constrained` entry is out of "
          "reach and saying so is cheaper\nthan quietly answering a smaller question.")

    space = parameter_space()
    rule("p(theta): the declaration being tested")
    print(space.describe())
    src = REGISTRY.implementations[IMPL]
    for b in space.blocks:
        print(f"  {b.name:8s} prior {b.prior.dist:10s} median "
              f"{(math.exp(b.prior.loc) if b.prior.dist == 'lognormal' else b.prior.loc):7.3f}"
              f"   [{b.prior.provenance.value}]  {b.prior.source or b.prior.note}")

    # -- data ------------------------------------------------------------
    rule("data")
    folder = local_root(SOURCE) / "1.0.0" / "sleep-cassette"
    print(f"{SOURCE} root {folder}  (from data/sources/{SOURCE}/raw/.location.yaml)")
    todo = pairs(folder)[: args.recordings]
    ev_by_night: dict[str, dict[str, SpectralEvidence]] = {}
    for psg, hyp in todo:
        try:
            e, _ = evidence_by_stage(psg, hyp, window_s=args.window_s, band=BAND,
                                     max_epochs=args.max_epochs,
                                     min_epochs=args.min_epochs)
        except Exception as exc:
            print(f"  {psg.name}: skipped ({type(exc).__name__}: {exc})")
            continue
        ev_by_night[psg.stem[:6]] = e
    nights = sorted(ev_by_night)
    subjects = sorted({subject_of(n) for n in nights})
    print(f"{len(nights)} nights from {len(subjects)} participants; up to "
          f"{args.max_epochs} scored 30 s epochs\nper stage per night, cut into "
          f"{args.window_s:g} s non-overlapping windows inside each epoch")
    ex = ev_by_night[nights[0]].get("N2")
    if ex is not None:
        print(f"  example: {ex}")
        print(f"  fitted over {BAND!r}: {len(ex.basis.band_indices(BAND))} bins at "
              f"{1.0 / ex.basis.duration_s:.3g} Hz resolution")

    # -- split by subject ------------------------------------------------
    rng = np.random.default_rng(args.seed)
    order = list(subjects)
    rng.shuffle(order)
    n_tr = int(round(args.train_frac * len(order)))
    tr_sub, te_sub = sorted(order[:n_tr]), sorted(order[n_tr:])
    tr_nights = [n for n in nights if subject_of(n) in tr_sub]
    te_nights = [n for n in nights if subject_of(n) in te_sub]

    rule("split")
    print("BY PARTICIPANT.  sleep-edfx records two consecutive nights per person and both "
          "go to\nthe same side; a night-level split would report one person's night-to-night "
          "stability\nas generalization to new people.")
    print(f"train {len(tr_sub)} participants / {len(tr_nights)} nights: "
          f"{' '.join(tr_nights)}")
    print(f"test  {len(te_sub)} participants / {len(te_nights)} nights: "
          f"{' '.join(te_nights)}")

    tr_n2 = [ev_by_night[n]["N2"] for n in tr_nights if "N2" in ev_by_night[n]]
    te_n2 = [ev_by_night[n]["N2"] for n in te_nights if "N2" in ev_by_night[n]]
    if len(tr_n2) < 4 or len(te_n2) < 4:
        print("not enough N2 spectra on one side of the split to say anything.")
        return 2

    # -- the fit ---------------------------------------------------------
    rule("forging: one (f0, q) for N2, over the train participants")
    theta_prior = space.median()
    theta_hat, rep = fit_stage(tr_n2, space)
    print(rep)
    f0_p, q_p = (float(theta_prior[space["f0_hz"].slice][0]),
                 float(theta_prior[space["q"].slice][0]))
    f0_h, q_h = (float(theta_hat[space["f0_hz"].slice][0]),
                 float(theta_hat[space["q"].slice][0]))
    print(f"\nprior      f0 {f0_p:6.3f} Hz   q {q_p:7.3f}")
    print(f"posterior  f0 {f0_h:6.3f} Hz   q {q_h:7.3f}")

    # -- held out --------------------------------------------------------
    rule("held-out: does the fitted resonance beat the declared one on unseen people?")
    print("both sides profile the drive exponent and the overall scale identically, so the "
          "only\ndifference scored is where the resonance sits and how sharp it is.  the "
          "likelihood is\nthe exact gamma density of an averaged periodogram, so a night "
          "with more surviving\nwindows carries more weight by construction rather than by "
          "a chosen weight.")
    rows = []
    for tag, nightset in (("train", tr_nights), ("test", te_nights)):
        lp, lq, names = [], [], []
        for n in nightset:
            e = ev_by_night[n].get("N2")
            if e is None:
                continue
            lp.append(profile_loglik(e, f0_p, q_p)[0])
            lq.append(profile_loglik(e, f0_h, q_h)[0])
            names.append(n)
        lp, lq = np.array(lp), np.array(lq)
        st = paired(lp, lq)
        n_bins = len(te_n2[0].basis.band_indices(BAND))
        rows.append((tag, lp, lq, st, names))
        print(f"\n{tag}: {len(lp)} nights, {n_bins} bins each")
        print(f"  mean log-likelihood per night    prior {lp.mean():12.2f}   "
              f"posterior {lq.mean():12.2f}")
        print(f"  ... per bin                      prior {lp.mean() / n_bins:12.4f}   "
              f"posterior {lq.mean() / n_bins:12.4f}")
        print(f"  paired delta {st['mean']:+.2f} +/- {st['sem']:.2f} sem, dz {st['dz']:+.2f}, "
              f"t {st['t']:+.2f}, p {st['p']:.3g}, posterior wins {st['win_rate']:.0%}")
    st_te = rows[-1][3]
    held_out_pass = bool(st_te["mean"] > 0 and st_te["p"] < 0.01
                         and st_te["win_rate"] > 0.5)
    print(f"\nHELD-OUT: {'PASSED' if held_out_pass else 'FAILED'} -- the fitted resonance "
          f"{'beats' if st_te['mean'] > 0 else 'does not beat'} the declared one on\n"
          f"{len(te_sub)} participants the fit never saw "
          f"({st_te['mean']:+.1f} nats per night, p {st_te['p']:.3g}, "
          f"wins {st_te['win_rate']:.0%}).")
    if not held_out_pass:
        print("a prior a fit cannot improve on held-out data is either already right or has "
              "the\nwrong functional form, and neither is repaired by refitting.")

    # -- per (night, stage) ----------------------------------------------
    rule("per-night, per-stage resonance (each night refitted on its own)")
    per_stage: dict[str, dict[str, dict]] = {}
    for st_name in STAGES:
        evs = [ev_by_night[n][st_name] for n in nights if st_name in ev_by_night[n]]
        if not evs:
            continue
        fits = per_recording(evs, space)
        per_stage[st_name] = {rec_key(f["source"]): f for f in fits}
    print(f"{'stage':6s} {'n':>4s} {'f0 Hz':>16s} {'q':>16s} {'drive e':>9s} {'r^2':>7s} "
          f"{'Q<0.5':>7s}")
    print("the last column is the fraction of nights whose fitted Q is below one half, the "
          "damping\nat which a second-order loop stops having a peak at all.  it is printed "
          "because it\ndecides how the f0 column may be read: in a cell with Q < 0.5 there is "
          "no resonance, and\nf0 there is a corner frequency rather than the location of "
          "anything visible in the\nspectrum.")
    table = {}
    for st_name in STAGES:
        d = per_stage.get(st_name)
        if not d:
            continue
        f0 = np.array([v["f0_hz"] for v in d.values()])
        qq = np.array([v["q"] for v in d.values()])
        ee = np.array([v["drive_exponent"] for v in d.values()])
        r2 = np.array([v["r2"] for v in d.values()])
        sem = f0.std(ddof=1) / math.sqrt(f0.size) if f0.size > 1 else float("nan")
        qsem = qq.std(ddof=1) / math.sqrt(qq.size) if qq.size > 1 else float("nan")
        overdamped = float((qq < 0.5).mean())
        table[st_name] = {"n": int(f0.size), "f0_mean": float(f0.mean()),
                          "f0_sem": float(sem), "q_mean": float(qq.mean()),
                          "q_sem": float(qsem), "e_mean": float(ee.mean()),
                          "r2_mean": float(r2.mean()), "fraction_overdamped": overdamped}
        print(f"{st_name:6s} {f0.size:4d} {f0.mean():8.3f} +/- {sem:5.3f} "
              f"{qq.mean():8.3f} +/- {qsem:5.3f} {ee.mean():9.3f} {r2.mean():7.3f} "
              f"{overdamped:6.0%}")

    results = {"posterior": {"f0_hz": f0_h, "q": q_h},
               "prior": {"f0_hz": f0_p, "q": q_p},
               "held_out": {"mean": st_te["mean"], "p": st_te["p"],
                            "dz": st_te["dz"], "win_rate": st_te["win_rate"],
                            "n": st_te["n"]},
               "held_out_pass": held_out_pass,
               "per_stage": table,
               "train_subjects": tr_sub, "test_subjects": te_sub}

    # -- falsification 1: the spindle shift ------------------------------
    rule("FALSIFICATION 1: does the resonance move up in N2, within night?")
    print("a sleep spindle is a 11-16 Hz event and a waking alpha rhythm is 8-13 Hz.  "
          "`sleep_dynamics`\nclaims to constrain spindle FREQUENCY through this loop, and "
          "`alpha_resonator` is the only\nresonance it has.  so the fitted f0 must be HIGHER "
          "in N2 than in W, in the same night.\nif it is not, the implementation cannot "
          "express the spindle, and the `constrained` entry\nthat promises spindle frequency "
          "is promising something the selected f cannot deliver.")
    contrasts = {}
    for a, b in (("W", "N2"), ("W", "N3"), ("W", "REM"), ("REM", "N2")):
        ks = [k for k in per_stage.get(a, {}) if k in per_stage.get(b, {})]
        if len(ks) < 4:
            continue
        xa = np.array([per_stage[a][k]["f0_hz"] for k in ks])
        xb = np.array([per_stage[b][k]["f0_hz"] for k in ks])
        stt = paired(xa, xb)
        contrasts[f"f0:{b}-{a}"] = stt
        print(f"  f0  {b} - {a:3s}  {stt['mean']:+7.3f} Hz +/- {stt['sem']:.3f} sem   "
              f"dz {stt['dz']:+6.2f}  t {stt['t']:+7.2f}  p {stt['p']:9.3g}  "
              f"higher in {stt['win_rate']:.0%} of {stt['n']} nights")
    c = contrasts.get("f0:N2-W")
    f1 = bool(c and c["mean"] > 0 and c["p"] < 0.01 and c["win_rate"] >= 0.7)
    print(f"\n  FALSIFICATION 1: {'PASSED' if f1 else 'FAILED'} -- the fitted resonance "
          f"{'does' if f1 else 'does NOT'} move up from wake to N2.")
    if not f1:
        print("  that is a statement about the DECLARATION and not about sleep.  spindles "
              "are in this\n  data; if a second-order resonance with one centre and one Q "
              "cannot find them on two\n  bipolar derivations, then `alpha_resonator` is the "
              "wrong f for `sleep_dynamics` to\n  select and the right fix is to select "
              "`burst_relay_rate` -- which the registry declares\n  precisely because 'that "
              "is a relaxation oscillator, not a resonator, and it is what a\n  sleep spindle "
              "actually is'.")
    results["falsification_spindle_shift"] = {"contrasts": contrasts, "passed": f1}

    # -- falsification 2: Q stays physiological --------------------------
    rule("FALSIFICATION 2: does Q stay inside the range the declaration calls physiological?")
    print("`alpha_resonator`'s own note: \"alpha peaks are narrow but not ringing; q above "
          "~10 would\nbe a pathological loop\".  that is a falsifiable statement about "
          "healthy sleep, and it is\nthe kind that a curve-fitting routine will happily "
          "violate to buy a sharper peak.")
    allq = np.array([v["q"] for d in per_stage.values() for v in d.values()])
    frac_hi = float((allq > 10.0).mean())
    print(f"\n  fitted Q over all {allq.size} (night, stage) cells: median {np.median(allq):.2f}, "
          f"90th pct {np.percentile(allq, 90):.2f}, max {allq.max():.2f}")
    print(f"  fraction above the declared pathological threshold of 10: {frac_hi:.0%}")
    f2 = frac_hi <= 0.1
    print(f"\n  FALSIFICATION 2: {'PASSED' if f2 else 'FAILED'} -- the fit "
          f"{'stays inside' if f2 else 'LEAVES'} the range the declaration calls\n  "
          "physiological.")
    if not f2:
        print("  when Q runs past 10 the bump has stopped being a rhythm and has become a "
              "free\n  curve-fitting element that absorbs whatever the drive exponent "
              "cannot.  every f0 in\n  a cell with Q above 10 is correspondingly less "
              "interpretable, and the stage table\n  above should be read with that in mind.")
    results["falsification_q_physiological"] = {
        "median": float(np.median(allq)), "p90": float(np.percentile(allq, 90)),
        "max": float(allq.max()), "fraction_above_10": frac_hi, "passed": f2}

    # -- falsification 3: REM has no spindles ----------------------------
    rule("FALSIFICATION 3: REM is the negative control, because REM has no spindles")
    print("spindles are an NREM phenomenon: they are generated by the thalamic reticular "
          "nucleus\nunder the hyperpolarization of NREM and they are absent in REM, which is "
          "not a matter\nof degree.  so whatever sigma resonance the fit finds in N2 must NOT "
          "be there in REM,\nin the same nights.  a pipeline that reports a spindle in REM is "
          "reporting its own\nfilter, and every N2 number above would be that filter too.")
    ks = [k for k in per_stage.get("N2", {}) if k in per_stage.get("REM", {})]
    if len(ks) >= 4:
        qn = np.array([per_stage["N2"][k]["q"] for k in ks])
        qr = np.array([per_stage["REM"][k]["q"] for k in ks])
        fn = np.array([per_stage["N2"][k]["f0_hz"] for k in ks])
        fr = np.array([per_stage["REM"][k]["f0_hz"] for k in ks])
        sq, sf = paired(qr, qn), paired(fr, fn)
        print(f"\n  Q   N2 - REM  {sq['mean']:+7.3f} +/- {sq['sem']:.3f} sem, dz "
              f"{sq['dz']:+.2f}, p {sq['p']:.3g}, sharper in N2 in "
              f"{sq['win_rate']:.0%} of {sq['n']} nights")
        print(f"  f0  N2 - REM  {sf['mean']:+7.3f} Hz +/- {sf['sem']:.3f} sem, dz "
              f"{sf['dz']:+.2f}, p {sf['p']:.3g}, higher in N2 in "
              f"{sf['win_rate']:.0%} of {sf['n']} nights")
        f3 = bool(sq["mean"] > 0 and sq["p"] < 0.05)
        print(f"\n  FALSIFICATION 3: {'PASSED' if f3 else 'FAILED'} -- the resonance is "
              f"{'sharper' if sq['mean'] > 0 else 'NOT sharper'} in N2 than in REM.")
        if not f3:
            print("  the sigma structure the fit reports in N2 is not distinguishable from "
                  "what it reports\n  in REM, where the physiology says there is none.  that "
                  "points at the estimator rather\n  than at the thalamus, and it caps how "
                  "much of falsification 1 can be believed.")
        results["falsification_rem_control"] = {"q_n2_minus_rem": sq,
                                                "f0_n2_minus_rem": sf, "passed": f3}
    else:
        print("  fewer than four nights have both N2 and REM; the control cannot be run.")
        results["falsification_rem_control"] = {"skipped": True}

    # -- caveats ---------------------------------------------------------
    rule("where this is weaker than it looks")
    print("""
HALF THE CONTRAST IN FALSIFICATION 1 IS NOT A FREQUENCY SHIFT.  the Q<0.5 column
above is the reason.  in W, N1 and REM the fitted loop is usually OVERDAMPED -- it
has no peak at all -- while in N2 and N3 it is not.  so "f0 rises from wake to N2"
is partly "a resonance appears", and the wake f0 it is being compared against is a
corner frequency of a shape with no maximum.  the test still means something,
because a form that could not produce a sigma resonance anywhere would fail it,
but it is weaker than a clean peak-to-peak comparison and must not be quoted as
one.

TWO CHANNELS.  sleep-edfx cassette gives Fpz-Cz and Pz-Oz, and the psds are
averaged across them.  spindles are maximal frontocentrally and alpha is maximal
occipitally, so averaging the two derivations mixes the very topography that would
separate the two rhythms.  a fitted f0 that sits between 10 and 13 Hz can be one
generator at 11.5 Hz or two generators being averaged, and nothing here can tell
those apart.  the model's own `prior_dominated` list says exactly this: "a
two-channel montage constrains no map".

DENSITY IS OUT OF REACH.  the `constrained` entry promises spindle "density,
duration and frequency" and credits `mass`'s event annotations for it.  `mass` is
not held.  a stage-averaged spectrum carries frequency and, through Q, something
like bandwidth; it carries no information about how many discrete events produced
it, so two thirds of that promise is untested here and one third of it is
untestable with this source at all.

INDEPENDENCE.  the degrees of freedom that set the likelihood's precision treat
every 4 s window of a stage as independent.  windows minutes apart in the same
night are not: sleep is non-stationary within a stage and spindle rate varies
across the night.  the per-cell fits are therefore more confident than they should
be.  the by-participant paired statistics are unaffected because they use one
number per night.

THE MULTIPLICATIVE FORM IS A COMMITMENT.  S = g f^-e |H|^2 says the loop shapes a
broadband drive.  a spindle that is genuinely a transient burst on an otherwise
unchanged background is better described by a sum, and the two forms are not
nested, so a poor fit here is evidence against the product form and not directly
evidence about f0 at all.  choosing the product was a reading of what
`thalamocortical_coupling` claims, and a different reading would give different
numbers.""")

    out = args.out or (Path(__file__).resolve().parents[1] / "data" / "sources" / SOURCE /
                       "evidence" / "fit_sleep_resonance@v1" / "posterior.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL_ID, "source": SOURCE, "band_hz":
                               [BAND.lo_hz, BAND.hi_hz], "seed": args.seed,
                               "results": results, "runtime_s": time.time() - t0},
                              indent=2, default=float))
    rule("summary")
    print(f"held-out (N2, unseen participants): "
          f"{'PASS' if held_out_pass else 'FAIL'}   "
          f"delta {st_te['mean']:+.1f} nats/night, p {st_te['p']:.3g}, "
          f"wins {st_te['win_rate']:.0%}")
    for k in ("falsification_spindle_shift", "falsification_q_physiological",
              "falsification_rem_control"):
        v = results.get(k, {})
        if "passed" in v:
            print(f"{k:36s} {'PASS' if v['passed'] else 'FAIL'}")
    print(f"\nwritten to {out}\n{time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
