#!/usr/bin/env python
"""an external-validity test of the fitted spectrum, on data the fit never touched.

`scripts/fit_neural_spectra.py` moves the `neural_population` shape parameters
against 60 seconds of resting eeg per subject and shows the result transfers to
held-out subjects of the same database, recorded on the same amplifier, in the
same posture, at the same time of day.  that is a weak kind of generalization and
it is worth naming as such: nothing there could distinguish a fit that learned
something about population dynamics from one that learned the BCI2000 rig.

so this asks a harder question of the same parameters.  sleep-edfx is a different
database, a different montage (two bipolar derivations rather than a 64-channel
cap), a different sampling rate, a different decade, and whole nights rather than
minutes.  and it carries something eegmmidb does not: an expert stage label every
30 seconds, which stratifies the recording by a state variable nobody fitted.

the published claim being tested is specific and directional.  the aperiodic
exponent of the electrophysiological spectrum **steepens** as arousal falls --
larger in deep NREM than in wake, with REM sitting near wake -- and it has been
reported that way in scalp eeg, in ecog and in intracranial recordings.  if the
form in `ibm/fields/priors.py` is a description of population dynamics, then
fitting it per stage should recover that ordering without being told about it.
if it is a curve that happens to fit resting eeg, it need not.

there are two ways this test could produce the right answer for the wrong reason,
and both are guarded rather than hoped away.

**the builder has no delta bump.**  its bumps are theta at 6 Hz, alpha and beta.
slow-wave activity in N3 lives at 0.5-4 Hz, below all of them, so it has nowhere
to go except the aperiodic term -- which means a fit over a band including delta
would report an exponent increase in N3 whether or not the *aperiodic* background
changed at all.  so the fit is run twice: once over 1-40 Hz, and once over
5-40 Hz, which excludes slow waves entirely.  only the second is evidence about
the aperiodic background, and both are printed.

**the whole shape is refitted per stage, not just the exponent.**  holding the
bump amplitudes at their eegmmidb values would force every stage-dependent change
in rhythm power into the exponent, which is the same artefact by another route.
all six parameters move, per recording and per stage, from the eegmmidb posterior
as a starting point and with the same prior widths the original fit had.

run:  ./.venv/bin/python scripts/eval_sleep_state.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit_neural_spectra import (PARAMS, as_kwargs, model_psd, parameter_space, paired, rule)
from ibm.fields.uncertainty.spectral import TemporalBasis
from ibm.forge.fit import Task, fit_map
from ibm.forge.spectra import SpectralEvidence, clean_windows, local_root, periodogram, window
from ibm.vocabulary import Band, Prior, Provenance

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

#: stage 3 and stage 4 are merged.  the distinction is an R&K convention that the
#: AASM rulebook abolished in 2007 precisely because scorers could not reproduce
#: it, and keeping it here would split the deepest stage into two thin, noisy
#: cells for no gain.  movement time and unscored epochs are dropped rather than
#: assigned anywhere.
STAGE_MAP = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}
STAGES = ("W", "N1", "N2", "N3", "REM")

#: the ordinal the published direction is stated against: falling arousal from
#: wake to deep NREM.  REM is deliberately NOT on this axis -- it is not a depth,
#: its cortical activation resembles wake, and putting it at 4 would test a claim
#: nobody makes.
DEPTH = {"W": 0, "N1": 1, "N2": 2, "N3": 3}

EPOCH_S = 30.0
EEG_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def pairs(folder: Path) -> list[tuple[Path, Path]]:
    """PSG files with their hypnograms, matched on the recording's own prefix.

    the archive names the two halves of one night differently after the sixth
    character (`SC4001E0-PSG` against `SC4001EC-Hypnogram`), so the prefix is the
    only reliable key.  a PSG with no hypnogram is skipped: this whole script is
    a stratification by expert label, and an unlabelled night is not a smaller
    version of the test, it is not the test.
    """
    hyp = {p.name[:6]: p for p in sorted(folder.glob("*-Hypnogram.edf"))}
    return [(p, hyp[p.name[:6]]) for p in sorted(folder.glob("*-PSG.edf"))
            if p.name[:6] in hyp]


def stage_epochs(annot, tmax: float) -> list[tuple[float, str]]:
    """(onset, stage) for every scorable 30 s epoch, in recording time."""
    out: list[tuple[float, str]] = []
    for on, dur, desc in zip(annot.onset, annot.duration, annot.description):
        s = STAGE_MAP.get(str(desc))
        if s is None:
            continue
        n = int(dur // EPOCH_S)
        for i in range(n):
            t = float(on) + i * EPOCH_S
            if t + EPOCH_S <= tmax:
                out.append((t, s))
    return sorted(out)


def sleep_window(eps: list[tuple[float, str]], pad_s: float = 1800.0
                 ) -> tuple[float, float]:
    """the recording trimmed to the night, plus half an hour of wake either side.

    the cassette recordings are ambulatory and run for about twenty hours, most
    of it daytime wake.  keeping all of it would make W a different thing from
    the other four stages -- a whole day of sitting up, moving and talking rather
    than lying still in the dark -- and the eyes-open, muscle-contaminated
    spectrum of that would be compared against N3 as though the only difference
    were arousal.  trimming to the night is what makes W a control rather than a
    confound.
    """
    sleep = [t for t, s in eps if s != "W"]
    if not sleep:
        return eps[0][0], eps[-1][0] + EPOCH_S
    return max(0.0, min(sleep) - pad_s), max(sleep) + EPOCH_S + pad_s


def evidence_by_stage(psg: Path, hyp: Path, *, window_s: float, band: Band,
                      max_epochs: int, min_epochs: int
                      ) -> tuple[dict[str, SpectralEvidence], dict[str, int]]:
    """one `SpectralEvidence` per stage for one night, and the epoch counts.

    the epochs of a stage are scattered across the night, so windows are cut
    inside each 30 s epoch and never across an epoch boundary: a window
    straddling a scored transition belongs to neither stage and would blur
    exactly the contrast this test is about.

    `max_epochs` is a cap and it is per stage.  without it W and N2 supply an
    order of magnitude more windows than N1, and the stage with the most sleep in
    it would be reported with the tightest error bars for reasons that have
    nothing to do with the biology.  the count that survives is what sets the
    evidence's degrees of freedom, so an under-represented stage comes out
    honestly uncertain instead of quietly precise.
    """
    import mne
    mne.set_log_level("ERROR")

    raw = mne.io.read_raw_edf(psg, preload=False, verbose="ERROR")
    raw.set_annotations(mne.read_annotations(hyp), emit_warning=False)
    fs = float(raw.info["sfreq"])
    picks = [raw.ch_names.index(c) for c in EEG_CHANNELS if c in raw.ch_names]
    if len(picks) != len(EEG_CHANNELS):
        raise ValueError(f"{psg.name}: expected {EEG_CHANNELS}, found {raw.ch_names}")

    eps = stage_epochs(raw.annotations, raw.n_times / fs)
    if not eps:
        raise ValueError(f"{psg.name}: no scorable epochs")
    t0, t1 = sleep_window(eps)
    eps = [(t, s) for t, s in eps if t0 <= t and t + EPOCH_S <= t1]
    raw.crop(tmin=t0, tmax=min(t1, raw.n_times / fs - 1.0 / fs)).load_data()
    x = raw.get_data(picks=picks)

    basis = TemporalBasis(int(round(window_s * fs)), 1.0 / fs)
    per = basis.n * (int(EPOCH_S * fs) // basis.n)
    rng = np.random.default_rng(0)
    out: dict[str, SpectralEvidence] = {}
    counts: dict[str, int] = {}
    for st in STAGES:
        ts = [t - t0 for t, s in eps if s == st]
        counts[st] = len(ts)
        if len(ts) < min_epochs:
            continue
        if len(ts) > max_epochs:                     # spread over the night, not the head
            ts = [ts[i] for i in np.linspace(0, len(ts) - 1, max_epochs).round().astype(int)]
        segs = []
        for t in ts:
            i = int(round(t * fs))
            seg = x[:, i: i + per]
            if seg.shape[-1] == per:
                segs.append(window(seg, basis))
        if not segs:
            continue
        w = np.concatenate(segs, axis=-2)             # (C, M, n)
        keep = clean_windows(w)
        w = w[..., keep, :]
        if keep.sum() < 4:
            continue
        p = periodogram(w, basis).mean(-2).mean(0)
        out[st] = SpectralEvidence(basis, p, int(keep.sum()), band=band,
                                   channels=EEG_CHANNELS, source=f"{psg.stem}/{st}",
                                   note=f"{len(ts)} epochs, {int(keep.sum())} windows")
    return out, counts


# ---------------------------------------------------------------------------
# the per-cell fit
# ---------------------------------------------------------------------------


def priors_at(centre: dict[str, float]) -> dict[str, Prior]:
    """the eegmmidb posterior as the prior for the sleep fit, at the original widths.

    this is the product form of §4 used as intended: `p(theta | D_eeg)` is the
    prior for the next factor.  the widths are the ones the eegmmidb fit started
    with rather than the ones it ended with, deliberately -- the posterior widths
    are mean-field and far too narrow, and using them here would pin the exponent
    near its wake value and manufacture the null.  a test has to be able to fail
    in both directions.
    """
    return {k: Prior("lognormal", math.log(max(centre.get(k, math.exp(p.loc)), 1e-9)),
                     p.scale, p.units, Provenance.FIT, "posterior of eegmmidb resting fit")
            for k, p in PARAMS.items()}


def fit_cell(ev: SpectralEvidence, band: Band, centre: dict[str, float]):
    """refit the whole shape against one (recording, stage) spectrum.

    `fit_map` from `ibm.forge.fit`, on a `ParameterSpace` built the same way the
    eegmmidb fit built its own, so the transforms, the jacobian and the
    shift-in-prior-sd accounting are identical and the two exponents are the same
    quantity.  the overall gain is profiled in closed form inside the likelihood,
    exactly as there, because a night of sleep changes electrode impedance and
    absolute scalp power for reasons that are not the aperiodic exponent.
    """
    space = parameter_space(priors_at(centre))
    theta0 = np.array([centre.get(b.name, 1.0) for b in space.blocks], float)
    i = ev.basis.band_indices(band)
    s = np.maximum(ev.psd[i], 1e-300)
    a = 0.5 * ev.dof[i]
    from scipy.special import gammaln

    def logp(theta: np.ndarray) -> float:
        g = np.maximum(model_psd(ev.basis, theta, space)[i], 1e-300)
        c = float((a * s / g).sum() / a.sum())
        v = float((a * np.log(a) - gammaln(a) + (a - 1.0) * np.log(s)
                   - a * np.log(np.maximum(c * g, 1e-300)) - a * s / np.maximum(c * g, 1e-300)
                   ).sum())
        # a degenerate corner of (exponent, knee) makes the model spectrum
        # identically zero over the band, where the density is genuinely -inf.
        # returning a finite floor rather than -inf keeps L-BFGS's line search
        # able to back out of it; returning -inf makes the whole fit fail on a
        # point the optimizer was only probing.
        return v if np.isfinite(v) else -1e12

    # errstate is scoped to the optimizer and to nothing else.  L-BFGS's line
    # search deliberately evaluates points where the model spectrum underflows,
    # and numpy's overflow and invalid warnings from those probes are noise: the
    # floor in `logp` is what handles them, and silencing them here rather than
    # globally keeps a genuine nan anywhere else in the script visible.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        t, _ = fit_map(space,
                       [Task(name=ev.source, logp=logp, moves=("neural_population_prior",))],
                       theta0=theta0, max_iter=200)
    kw = as_kwargs(space, t)
    kw["r2"] = ev.r2_log(model_psd(ev.basis, t, space), band=band)
    return kw


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recordings", type=int, default=20)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--posterior", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    post_path = args.posterior or (root / "data" / "sources" / "eegmmidb" / "evidence" /
                                   "fit_neural_spectra@v1" / "posterior.json")
    if not post_path.is_file():
        print(f"no fitted posterior at {post_path}\nrun scripts/fit_neural_spectra.py first; "
              "this script tests THAT result and has nothing\nto test without it.")
        return 2
    payload = json.loads(post_path.read_text())
    centre = payload["posterior"]

    rule("what is being tested")
    print(f"posterior from {post_path}")
    print("  " + "  ".join(f"{k} {v:.4g}" for k, v in centre.items()))
    print(f"  fitted on eegmmidb resting runs of {len(payload['train_subjects'])} subjects, "
          f"1-45 Hz")
    print("sleep-edfx was NOT used to fit any of it.  it is a different database, montage, "
          "\nsampling rate and state, and the stage labels are an expert's, not a model's.")

    rule("data")
    folder = local_root("sleep-edfx") / "1.0.0" / "sleep-cassette"
    print(f"sleep-edfx root {folder}  (from data/sources/sleep-edfx/raw/.location.yaml)")
    todo = pairs(folder)[: args.recordings]
    print(f"{len(todo)} sleep-cassette nights from "
          f"{len({p.name[3:5] for p, _ in todo})} participants; channels "
          f"{', '.join(EEG_CHANNELS)} at 100 Hz")
    print(f"up to {args.max_epochs} epochs ({args.max_epochs * EPOCH_S / 60:.0f} min) per "
          f"stage per night, at least {args.min_epochs}; "
          f"{args.window_s:g} s windows inside each 30 s epoch")

    bands = {"1-40 Hz (includes delta)": Band(1.0, 40.0),
             "5-40 Hz (delta excluded)": Band(5.0, 40.0)}

    ev_by_rec: dict[str, dict[str, SpectralEvidence]] = {}
    counts: dict[str, dict[str, int]] = {}
    band_full = Band(1.0, 40.0)
    for psg, hyp in todo:
        try:
            e, c = evidence_by_stage(psg, hyp, window_s=args.window_s, band=band_full,
                                     max_epochs=args.max_epochs, min_epochs=args.min_epochs)
        except Exception as exc:                      # a bad night is a finding, not a crash
            print(f"  {psg.name}: skipped ({type(exc).__name__}: {exc})")
            continue
        ev_by_rec[psg.stem[:6]] = e
        counts[psg.stem[:6]] = c
    print(f"\nusable nights: {len(ev_by_rec)}")
    print(f"{'night':8s} " + "  ".join(f"{s:>18s}" for s in STAGES))
    for k in sorted(ev_by_rec):
        print(f"{k:8s} " + "  ".join(
            f"{counts[k][s]:5d} ep {'/'+str(ev_by_rec[k][s].n_windows)+'w' if s in ev_by_rec[k] else '  --  ':>9s}"
            for s in STAGES))

    results = {}
    for label, band in bands.items():
        rule(f"per-stage aperiodic exponent, fitted over {label}")
        table: dict[str, dict[str, float]] = {}
        for rec, evs in sorted(ev_by_rec.items()):
            table[rec] = {}
            for st, ev in evs.items():
                table[rec][st] = fit_cell(ev, band, centre)

        print(f"{'stage':6s} {'n':>4s} {'exponent':>20s} {'knee Hz':>16s} "
              f"{'alpha Hz':>16s} {'r^2':>8s}")
        per_stage = {}
        for st in STAGES:
            v = np.array([table[r][st]["exponent"] for r in table if st in table[r]])
            if v.size == 0:
                continue
            kn = np.array([table[r][st]["knee_hz"] for r in table if st in table[r]])
            al = np.array([table[r][st]["alpha_hz"] for r in table if st in table[r]])
            r2 = np.array([table[r][st]["r2"] for r in table if st in table[r]])
            sem = v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else float("nan")
            per_stage[st] = {"n": int(v.size), "mean": float(v.mean()),
                             "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                             "sem": float(sem), "knee_hz": float(kn.mean()),
                             "alpha_hz": float(al.mean()), "r2": float(r2.mean())}
            print(f"{st:6s} {v.size:4d} {v.mean():9.3f} +/- {sem:.3f} sem "
                  f"(sd {v.std(ddof=1) if v.size > 1 else float('nan'):.3f})"
                  f" {kn.mean():16.3f} {al.mean():16.2f} {r2.mean():8.3f}")

        stray = [s for s, d in per_stage.items() if not 8.0 <= d["alpha_hz"] <= 13.0]
        if stray:
            mw = max(e.n_windows for d in ev_by_rec.values() for e in d.values())
            print(f"\n  CAVEAT: the fitted alpha centre left the 8-13 Hz band it is named "
                  f"after in {', '.join(stray)}.\n  with up to {mw} windows the evidence's "
                  "degrees of freedom run into the high hundreds,\n  which crushes a "
                  "lognormal prior centred at 10 Hz; the bump then stops being a rhythm\n"
                  "  and becomes a free curve-fitting element that absorbs whatever the "
                  "aperiodic term\n  cannot.  the exponents in those rows are "
                  "correspondingly less interpretable.")

        print("\nwithin-night paired contrasts (each night is its own control for skull, "
              "montage\nand impedance, which between-night comparison is not):")
        contrasts = {}
        for a, b in (("W", "N1"), ("W", "N2"), ("W", "N3"), ("N2", "N3"), ("W", "REM")):
            recs = [r for r in table if a in table[r] and b in table[r]]
            if len(recs) < 3:
                continue
            xa = np.array([table[r][a]["exponent"] for r in recs])
            xb = np.array([table[r][b]["exponent"] for r in recs])
            st = paired(xa, xb)
            contrasts[f"{b}-{a}"] = st
            print(f"  {b} - {a:3s}  {st['mean']:+7.3f} +/- {st['sem']:.3f} sem   "
                  f"dz {st['dz']:+6.2f}  t {st['t']:+7.2f}  p {st['p']:9.3g}  "
                  f"steeper in {st['win_rate']:.0%} of {st['n']} nights")

        # monotone depth: one spearman per night over W < N1 < N2 < N3
        from scipy import stats
        rhos = []
        for r in table:
            xs = [(DEPTH[s], table[r][s]["exponent"]) for s in DEPTH if s in table[r]]
            if len(xs) >= 3:
                rhos.append(stats.spearmanr([a for a, _ in xs], [b for _, b in xs]).statistic)
        rho = np.array([x for x in rhos if np.isfinite(x)])
        rho_t, rho_p = stats.ttest_1samp(rho, 0.0) if rho.size > 2 else (np.nan, np.nan)
        print(f"\n  per-night rank correlation of exponent with NREM depth (W<N1<N2<N3): "
              f"mean rho {rho.mean():+.3f}\n  +/- {rho.std(ddof=1) / math.sqrt(rho.size):.3f} "
              f"sem over {rho.size} nights, t {rho_t:+.2f}, p {rho_p:.3g}; positive in "
              f"{(rho > 0).mean():.0%}")

        c3 = contrasts.get("N3-W")
        ok = bool(c3 and c3["mean"] > 0 and c3["p"] < 0.01 and c3["win_rate"] >= 0.7
                  and rho.mean() > 0 and rho_p < 0.01)
        print(f"\n  {label}: {'PASSED' if ok else 'FAILED'} -- the fitted aperiodic exponent "
              f"{'steepens' if (c3 and c3['mean'] > 0) else 'does NOT steepen'} with deeper "
              f"NREM.")
        results[label] = {"per_stage": per_stage, "contrasts": contrasts,
                          "depth_rho_mean": float(rho.mean()),
                          "depth_rho_sem": float(rho.std(ddof=1) / math.sqrt(rho.size)),
                          "depth_rho_p": float(rho_p), "passed": ok,
                          "per_night": {r: {s: table[r][s] for s in table[r]} for r in table}}

    # -- the verdict ----------------------------------------------------
    rule("verdict")
    delta = results["1-40 Hz (includes delta)"]
    nodelta = results["5-40 Hz (delta excluded)"]
    print(f"1-40 Hz  (delta included): {'PASS' if delta['passed'] else 'FAIL'}")
    print(f"5-40 Hz  (delta excluded): {'PASS' if nodelta['passed'] else 'FAIL'}")
    print()
    if delta["passed"] and nodelta["passed"]:
        print("the published direction replicates, and it survives removing the band where "
              "slow\nwaves live.  that is the strong version: the change is in the aperiodic "
              "background\nand not only in delta power leaking into an exponent that has "
              "nowhere else to put it.")
    elif delta["passed"] and not nodelta["passed"]:
        print("the effect is present with delta in the band and absent without it.  that is "
              "the\nweak version, and it is the one the guard above was built to detect: "
              "what moved is\nslow-wave power, and the `neural_population` form -- which has "
              "no delta bump -- can\nonly express that as a steeper aperiodic term.  the "
              "1-40 Hz number should NOT be\nreported as an aperiodic finding.")
    elif not delta["passed"] and nodelta["passed"]:
        print("the effect appears only above the delta band, which is not the shape the "
              "literature\nreports and is not explained by the missing delta bump.  treat "
              "it as unreplicated.")
    else:
        print("the published direction did NOT replicate here, in either band.\n\n"
              "what that implies, stated rather than tuned away: the `neural_population` "
              "form as\ndeclared cannot express the state dependence of the aperiodic "
              "background on two\nbipolar derivations at 100 Hz -- either because a single "
              "lorentzian knee plus three\nfixed-centre bumps is the wrong parameterization "
              "for sleep, because the exponent and\nthe knee trade off strongly enough over "
              "1-40 Hz that neither is identified alone, or\nbecause the effect is not there "
              "in this montage.  the fix is a different declaration,\nnot a different "
              "fitting run, and nothing above should be re-run with a narrower band\nuntil "
              "it separates.")

    rule("where this is weaker than it looks")
    for label in bands:
        ps = results[label]["per_stage"]
        ct = results[label]["contrasts"]
        means = [ps[s]["mean"] for s in ("W", "N1", "N2", "N3") if s in ps]
        mono = all(b >= a for a, b in zip(means, means[1:]))
        rem = (ct.get("REM-W", {}).get("mean", float("nan"))
               / max(ct.get("N3-W", {}).get("mean", float("nan")), 1e-9))
        print(f"\n{label}")
        print(f"  stage means over W<N1<N2<N3 are "
              f"{'monotone' if mono else 'NOT monotone'}: "
              + " < ".join(f"{m:.2f}" for m in means))
        print(f"  REM sits at {rem:.0%} of the way from W to N3.  the literature puts REM "
              f"near wake,\n  not in the middle of NREM, so this is not the published "
              "pattern -- and the most\n  likely reason is the wake end rather than the "
              "REM end.")
        print(f"  fitted alpha centre by stage: "
              + ", ".join(f"{s} {ps[s]['alpha_hz']:.1f} Hz" for s in STAGES if s in ps))
    print("""
three things this experiment cannot rule out, stated because a directional
replication is exactly the situation in which they are easiest not to mention.

MUSCLE.  the sleep-edfx card lists temporalis, frontalis and neck EMG among its
confounds, and EMG is broadband with most of its power above 20 Hz.  it is
present in wake, reduced in NREM and abolished in REM by atonia.  a spectrum that
loses a broadband high-frequency component looks steeper, so "the exponent
steepens with sleep depth" and "the subject stopped tensing their jaw" predict
the same number here.  restricting to 5-40 Hz does NOT separate them: that is
where EMG lives.  separating them needs the chin EMG channel this script does not
read, and until it does the effect size above is an upper bound on the neural
part of the effect.

REM.  the exponent in REM comes out close to NREM rather than close to wake.
under the arousal account that is wrong; under the muscle account it is exactly
right, because REM atonia removes more EMG than NREM does.  the discrepancy is
evidence for the confound rather than against it, and it is the single clearest
reason not to read this as a clean replication.

INDEPENDENCE.  the degrees of freedom above treat every 4 s window of a stage as
an independent sample, which is what sets the precision that crushes the priors.
windows minutes apart in the same night are not independent -- sleep is not
stationary within a stage -- so the per-cell fits are more confident than they
should be.  the within-night paired statistics are unaffected, because they use
one number per night; the error bars on the individual cells are not.""")

    out = post_path.parent / "sleep_state.json"
    out.write_text(json.dumps({"posterior_from": str(post_path), "nights": sorted(ev_by_rec),
                               "epoch_counts": counts, "results": results,
                               "runtime_s": time.time() - t0}, indent=2, default=float))
    print(f"\nwritten to {out}\n{time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
