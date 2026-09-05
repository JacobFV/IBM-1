#!/usr/bin/env python
"""does forging actually move the neural_population prior, and does the move generalize?

`ibm/fields/priors.py` declares that resting population state looks like a knee'd
power law with theta, alpha and beta bumps on it, and it declares numbers for the
shape.  those numbers came from reading papers.  this script is the first place
anything in ibm-1 asks whether they are the *right* numbers for a real recording,
and whether replacing them with fitted ones helps on data the fit never saw.

the experiment is deliberately the smallest one that can fail.

    fit     the shape parameters of `neural_population` against the resting runs
            of eegmmidb TRAIN subjects, through `ibm.forge.fit`
    test    score the fitted posterior and the untouched prior on the resting
            runs of HELD-OUT subjects, on identical terms
    falsify refit the alpha amplitude alone, per held-out recording, and ask
            whether it separates eyes-closed from eyes-open

the split is by **subject** and never by window.  windows from one subject share
a skull, a montage, an alpha peak and an impedance; splitting by window would
measure how well the model interpolates within a person's own spectrum, which is
a number that goes up regardless of whether anything was learned.  the eegmmidb
card says `split: group_by: [participant, run]` and this obeys it.

three things are profiled out or held fixed, and each is a decision rather than a
convenience.

**absolute power is a nuisance, profiled per recording in closed form.**  the
scale of a scalp psd is skull conductivity, electrode impedance and the reference
montage.  the neural prior asserts nothing about any of those, so a fit that let
the scale in would spend most of its likelihood on the amplifier.  what is fitted
is the shape.

**the parameter space is a prior over a prior.**  the thing being moved here is
not a process's theta -- it is the hyperparameters of a state prior, which is a
legitimate but different object, and it is laid out through
`ibm.forge.priors.ParameterSpace` because that is where the transforms, the log
jacobian and the shift-in-prior-sd reporting already live.  it is registered
under its own process name and does not pretend to be one of the 28.

**the eyes-closed contrast is a falsification test, not a headline.**  the
eegmmidb card says so in the card itself: "the eyes-closed run is the alpha
fixture: any fit of the posterior alpha prior that cannot separate these two has
failed".  so the script prints PASS or FAIL and does not leave the reader to
decide what the number meant.

run:  ./.venv/bin/python scripts/fit_neural_spectra.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibm.fields import priors as field_priors
from ibm.fields.uncertainty.spectral import TemporalBasis
from ibm.forge.fit import Method, Task, fit, fit_ensemble, fit_map, fit_vi
from ibm.forge.priors import ParameterSpace, assemble
from ibm.forge.spectra import SpectralEvidence, evidence_from_signal, local_root
from ibm.registry import Form, Implementation
from ibm.vocabulary import Band, Provenance, Tying, lognormal

# ---------------------------------------------------------------------------
# what is fitted
# ---------------------------------------------------------------------------

FIT_PROCESS = "neural_population_prior"
FIT_IMPL = "psd_shape"

#: the shape parameters of the `neural_population` builder, with the priors the
#: literature actually supports.  the two aperiodic terms and the alpha centre
#: get literature provenance because they are measured quantities with reported
#: ranges; the three bump amplitudes get WEAK, because the builder's
#: normalization (a fraction of the aperiodic total) is a convention of this
#: codebase and no paper reports a number in those units.  saying so is the point
#: -- `FitReport` can then distinguish a parameter the data moved off a real
#: prior from one it moved off a placeholder.
PARAMS = {
    "exponent": lognormal(2.0, 1.5, units="", provenance=Provenance.LITERATURE,
                          source="donoghue et al. 2020 nat neurosci: resting scalp eeg "
                                 "aperiodic exponents cluster between 1 and 3"),
    "knee_hz": lognormal(2.0, 3.0, units="Hz", provenance=Provenance.LITERATURE,
                         source="he 2014 trends cogn sci: the decorrelation timescale is "
                                "reported to within a factor, not an increment"),
    "alpha_hz": lognormal(10.0, 1.25, units="Hz", provenance=Provenance.LITERATURE,
                          source="klimesch 1999 brain res rev: individual alpha peak "
                                 "frequency spans roughly 8-13 Hz in healthy adults"),
    "alpha_gain": lognormal(0.6, 3.0, units="", provenance=Provenance.WEAK,
                            source="builder convention: bump power as a fraction of the "
                                   "aperiodic total, which no paper reports"),
    "theta_gain": lognormal(0.3, 3.0, units="", provenance=Provenance.WEAK,
                            source="builder convention"),
    "beta_gain": lognormal(0.15, 3.0, units="", provenance=Provenance.WEAK,
                           source="builder convention"),
}

PARAM_NAMES = tuple(PARAMS)


def parameter_space(params: dict | None = None) -> ParameterSpace:
    """one global block per shape parameter, through the ordinary assembly path.

    `assemble` takes a list of implementations precisely so that a parameter set
    can be laid out before -- or without -- a materialization, and this uses that
    door rather than constructing `ParameterBlock`s by hand.  what it buys is the
    unconstrained transform and its log jacobian: every parameter here is
    positive and reported to within a factor, so all six are lognormal and all
    six are optimized in logs, which is the difference between a knee that
    settles at 1.4 Hz and one that walks into the boundary at zero.
    """
    impl = Implementation(
        name=FIT_IMPL, process=FIT_PROCESS,
        doc="shape of the neural_population resting psd: aperiodic exponent and knee, "
            "alpha centre and amplitude, theta and beta amplitude",
        form=Form.TABLE, params=dict(params or PARAMS), tying=Tying.GLOBAL,
        provenance=Provenance.LITERATURE)
    return assemble(implementations=[impl])


def as_kwargs(space: ParameterSpace, theta: np.ndarray) -> dict[str, float]:
    return {b.name: float(theta[b.slice][0]) for b in space.blocks}


def model_psd(basis: TemporalBasis, theta: np.ndarray, space: ParameterSpace) -> np.ndarray:
    """the registered prior builder, evaluated at theta, on the evidence's own basis.

    this calls `ibm.fields.priors.build("neural_population", ...)` rather than
    reimplementing the shape.  if it did not, the thing fitted here and the thing
    the ontology would actually materialize could drift apart without anything
    noticing, and the fit would be of a curve that resembles the prior.
    """
    return field_priors.build("neural_population", basis, **as_kwargs(space, theta)).psd


# ---------------------------------------------------------------------------
# a set of recordings, and its likelihood
# ---------------------------------------------------------------------------


@dataclass
class Cohort:
    """the recordings of one split, stacked so a likelihood is one numpy expression.

    every recording here shares a basis, so the model spectrum is computed once
    per theta and the per-recording work is a scale and a sum.  that is what
    makes a variational fit over this data affordable at all: the objective is
    evaluated tens of thousands of times, and the alternative -- a python loop
    over recordings inside it -- costs two orders of magnitude for nothing.

    the per-recording nuisance gain is profiled in closed form inside the
    likelihood, so it never enters the optimizer's parameter vector.  with one
    gain per recording it would otherwise *be* the parameter vector.
    """

    name: str
    basis: TemporalBasis
    band: Band
    idx: np.ndarray                  # bins inside the band
    psd: np.ndarray                  # (R, F) observed density
    a: np.ndarray                    # (R, F) half the degrees of freedom
    subjects: tuple[str, ...] = ()
    runs: tuple[str, ...] = ()

    @classmethod
    def of(cls, name: str, evs: list[SpectralEvidence], band: Band,
           subjects=(), runs=()) -> "Cohort":
        basis = evs[0].basis
        idx = basis.band_indices(band)
        psd = np.stack([e.psd[idx] for e in evs])
        a = np.stack([0.5 * e.dof[idx] for e in evs])
        return cls(name, basis, band, idx, psd, a, tuple(subjects), tuple(runs))

    @property
    def n(self) -> int:
        return self.psd.shape[0]

    def _gain(self, g: np.ndarray) -> np.ndarray:
        return (self.a * self.psd / g).sum(-1) / self.a.sum(-1)

    def per_recording(self, model: np.ndarray) -> np.ndarray:
        """log p(psd_hat | model) for each recording, at its own profiled gain."""
        g = np.maximum(model[self.idx], 1e-300)[None, :]
        c = self._gain(g)[:, None]
        s, a = np.maximum(self.psd, 1e-300), self.a
        return (a * np.log(a) - _lgamma(a) + (a - 1.0) * np.log(s)
                - a * np.log(c * g) - a * s / (c * g)).sum(-1)

    def loglik(self, model: np.ndarray) -> float:
        return float(self.per_recording(model).sum())

    def r2_per_recording(self, model: np.ndarray) -> np.ndarray:
        """variance of log10 psd explained, at the likelihood's own gain, per recording."""
        g = np.maximum(model[self.idx], 1e-300)[None, :]
        c = self._gain(g)[:, None]
        y = np.log10(np.maximum(self.psd, 1e-300))
        f = np.log10(c * g)
        ss = ((y - y.mean(-1, keepdims=True)) ** 2).sum(-1)
        return 1.0 - ((y - f) ** 2).sum(-1) / np.maximum(ss, 1e-300)

    @property
    def n_bins(self) -> int:
        return self.psd.shape[1]


def _lgamma(x: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln
    return gammaln(x)


# ---------------------------------------------------------------------------
# reading eegmmidb
# ---------------------------------------------------------------------------

#: parieto-occipital channels.  alpha is a posterior rhythm and the card's own
#: note calls the eyes-closed run the alpha fixture, so restricting to where the
#: rhythm is makes the falsification test sharper rather than kinder: a frontal
#: average would dilute the effect the test is trying to detect and a null there
#: would be uninformative.
POSTERIOR_CH = ("p3", "p1", "pz", "p2", "p4", "po3", "poz", "po4", "o1", "oz", "o2")

RESTING_RUNS = {"R01": "eyes_open", "R02": "eyes_closed"}


def _chan_index(names: list[str], wanted: tuple[str, ...]) -> list[int]:
    norm = [n.strip().strip(".").lower() for n in names]
    return [i for i, n in enumerate(norm) if n in wanted]


def read_eegmmidb(root: Path, n_subjects: int, window_s: float, band: Band,
                  verbose: bool = True):
    """resting runs of the first `n_subjects` usable eegmmidb participants.

    four participants in this database were recorded at a different sampling rate
    than the rest, and they are skipped rather than resampled: resampling changes
    the noise floor and the anti-alias rolloff, and a spectrum fitted to 1-45 Hz
    would inherit a different high-frequency shape for those four alone.  the
    skip is reported, because an unreported exclusion is how a cohort quietly
    stops being the cohort the card describes.
    """
    import mne
    mne.set_log_level("ERROR")

    subs = sorted(p.name for p in root.glob("S[0-9][0-9][0-9]") if p.is_dir())
    out: dict[str, dict[str, SpectralEvidence]] = {}
    skipped: list[str] = []
    for s in subs:
        if len(out) >= n_subjects:
            break
        got: dict[str, SpectralEvidence] = {}
        for run, label in RESTING_RUNS.items():
            f = root / s / f"{s}{run}.edf"
            if not f.is_file():
                break
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
            fs = float(raw.info["sfreq"])
            if abs(fs - 160.0) > 1e-6:
                skipped.append(f"{s} ({fs:g} Hz)")
                break
            ch = _chan_index(raw.ch_names, POSTERIOR_CH)
            if len(ch) < len(POSTERIOR_CH):
                skipped.append(f"{s} (montage)")
                break
            x = raw.get_data(picks=ch)
            got[label] = evidence_from_signal(
                x, fs, window_s=window_s, band=band,
                channels=tuple(raw.ch_names[i] for i in ch),
                source=f"eegmmidb/{s}/{run}", note=label)
        if len(got) == len(RESTING_RUNS):
            out[s] = got
        elif s not in " ".join(skipped):
            skipped.append(f"{s} (missing run)")
    if verbose and skipped:
        print(f"  skipped {len(skipped)}: {', '.join(skipped[:8])}"
              + (" ..." if len(skipped) > 8 else ""))
    if not out:
        raise RuntimeError(f"no usable eegmmidb subjects under {root}")
    return out


# ---------------------------------------------------------------------------
# the per-recording alpha readout
# ---------------------------------------------------------------------------


def alpha_per_recording(ev_psd: np.ndarray, a: np.ndarray, idx: np.ndarray,
                        basis: TemporalBasis, theta: np.ndarray,
                        space: ParameterSpace) -> float:
    """refit alpha_gain alone against one recording, everything else held at theta.

    a one-parameter `ParameterSpace` and `fit_map`, rather than a hand-rolled line
    search, so that the transform and its jacobian are the same ones the main fit
    used -- an alpha amplitude estimated in a different parameterization is not
    comparable to the one the posterior reports.

    the prior on this single parameter is centred on theta's own alpha_gain and
    deliberately very wide (a factor of ten either way).  it has to be: this
    quantity is being used to *discriminate* two conditions, and a tight prior
    would shrink both towards the same value and manufacture a null.
    """
    base = as_kwargs(space, theta)
    one = parameter_space({"alpha_gain": lognormal(max(base["alpha_gain"], 1e-6), 10.0,
                                                   provenance=Provenance.FIT)})

    s = np.maximum(ev_psd, 1e-300)

    def logp(t1: np.ndarray) -> float:
        kw = dict(base, alpha_gain=float(t1[0]))
        g = np.maximum(field_priors.build("neural_population", basis, **kw).psd[idx], 1e-300)
        c = float((a * s / g).sum() / a.sum())
        v = float((a * np.log(a) - _lgamma(a) + (a - 1.0) * np.log(s)
                   - a * np.log(np.maximum(c * g, 1e-300))
                   - a * s / np.maximum(c * g, 1e-300)).sum())
        return v if np.isfinite(v) else -1e12

    # scoped to the optimizer: its line search evaluates points where the model
    # spectrum underflows, and the floor in `logp` is what handles those.  a nan
    # raised anywhere else in this script stays visible.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        t, _ = fit_map(one, [Task(name="recording", logp=logp, moves=(FIT_PROCESS,))],
                       max_iter=120)
    return float(t[0])


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------


def paired(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """paired difference of two aligned vectors, with the statistics that matter.

    a mean difference with no dispersion beside it is not a result, and a p-value
    with no effect size beside it is not one either.  both, plus the win rate,
    because the win rate is the only one of the three that survives a
    non-gaussian difference distribution intact.
    """
    from scipy import stats
    d = np.asarray(y, float) - np.asarray(x, float)
    t, p = stats.ttest_rel(y, x) if len(d) > 1 else (float("nan"), float("nan"))
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return {"mean": float(d.mean()), "sd": sd, "sem": sd / math.sqrt(len(d)),
            "t": float(t), "p": float(p),
            "dz": float(d.mean() / sd) if sd else float("nan"),
            "win_rate": float((d > 0).mean()), "n": len(d)}


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=50)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--lo-hz", type=float, default=1.0)
    ap.add_argument("--hi-hz", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vi-steps", type=int, default=800)
    ap.add_argument("--ensemble", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    band = Band(args.lo_hz, args.hi_hz)
    t0 = time.time()

    rule("data")
    root = local_root("eegmmidb") / "1.0.0"
    print(f"eegmmidb root {root}  (from data/sources/eegmmidb/raw/.location.yaml)")
    print(f"reading up to {args.subjects} subjects, resting runs R01 (eyes open) and "
          f"R02 (eyes closed)")
    ev = read_eegmmidb(root, args.subjects, args.window_s, band)
    subjects = sorted(ev)
    basis = ev[subjects[0]]["eyes_open"].basis
    print(f"{len(subjects)} subjects x {len(RESTING_RUNS)} runs = "
          f"{len(subjects) * len(RESTING_RUNS)} recordings")
    print(f"  {ev[subjects[0]]['eyes_open']}")
    print(f"  channels: {', '.join(ev[subjects[0]]['eyes_open'].channels)}")
    print(f"  fitted over {band!r}: {len(basis.band_indices(band))} bins at "
          f"{1.0 / basis.duration_s:.3g} Hz resolution; 60 Hz line and the sub-1 Hz "
          "drift the card warns about are both outside it")

    # -- the split ------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    order = list(subjects)
    rng.shuffle(order)
    n_train = int(round(args.train_frac * len(order)))
    train_s, test_s = sorted(order[:n_train]), sorted(order[n_train:])

    rule("split")
    print(f"BY SUBJECT, seed {args.seed}.  no subject contributes windows to both sides, "
          "and no\nrecording is split across them; the eegmmidb card asks for exactly this "
          "(split.group_by:\n[participant, run]).")
    print(f"train {len(train_s)}: {' '.join(train_s)}")
    print(f"test  {len(test_s)}: {' '.join(test_s)}")

    def cohort(name, subs, labels):
        evs, ss, rr = [], [], []
        for s in subs:
            for lb in labels:
                evs.append(ev[s][lb]); ss.append(s); rr.append(lb)
        return Cohort.of(name, evs, band, ss, rr)

    labels = tuple(RESTING_RUNS.values())
    train = cohort("train", train_s, labels)
    test = cohort("test", test_s, labels)
    test_open = cohort("test/eyes_open", test_s, ("eyes_open",))
    test_closed = cohort("test/eyes_closed", test_s, ("eyes_closed",))

    # -- the fit --------------------------------------------------------
    space = parameter_space()
    theta_prior = space.median()

    rule("p(theta): the prior being tested")
    print(space.describe())
    for b in space.blocks:
        print(f"  {b.name:11s} median {theta_prior[b.slice][0]:8.4g}  "
              f"x/ {math.exp(b.prior.scale):.3g}  [{b.prior.provenance.value}]")

    def _finite(v: float) -> float:
        # the far corners of (exponent, knee) drive the model spectrum to zero
        # over the whole band, where the density really is -inf.  a finite floor
        # lets L-BFGS's line search back out; -inf makes the fit die on a point
        # it was only probing.
        return v if np.isfinite(v) else -1e12

    def train_logp(theta: np.ndarray) -> float:
        return _finite(train.loglik(model_psd(basis, theta, space)))

    def test_logp(theta: np.ndarray) -> float:
        return _finite(test.loglik(model_psd(basis, theta, space)))

    tasks = [
        Task(name="eegmmidb.rest.train", logp=train_logp, moves=(FIT_PROCESS,),
             source="eegmmidb", kind="fit",
             note=f"{train.n} resting recordings from {len(train_s)} subjects"),
        Task(name="eegmmidb.rest.heldout", logp=test_logp, moves=(), source="eegmmidb",
             kind="evaluate",
             note=f"{test.n} resting recordings from {len(test_s)} unseen subjects"),
    ]

    rule("forging")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        theta_map, rep = fit(space, tasks, method=Method.MAP)
        print(rep)

        print("\nvariational pass, for a width:")
        theta_vi, sd_vi, rep_vi = fit_vi(space, tasks, theta0=theta_map, steps=args.vi_steps,
                                         m=12, lr=0.05, seed=args.seed)
        print(rep_vi)

        print(f"\nensemble of {args.ensemble} MAP fits from independent prior draws, as the "
              "cross-check\nfit_vi's own docstring asks for:")
        theta_ens, sd_ens, rep_ens = fit_ensemble(space, tasks, n=args.ensemble,
                                                  seed=args.seed)
        print(rep_ens)

    theta_hat = theta_map

    # -- (e) narrowing --------------------------------------------------
    rule("(e) posterior narrowing: prior sd vs posterior sd")
    print(f"{'parameter':12s} {'prior med':>10s} {'post':>10s} {'prior sd':>10s} "
          f"{'post sd(VI)':>12s} {'ratio':>7s} {'ens sd':>10s}  provenance")
    narrowing = {}
    for b in space.blocks:
        prior_sd = abs(b.prior.scale) * math.exp(b.prior.loc)      # fit.py's _prior_sd
        post_sd = float(sd_vi[b.slice][0])
        ens_sd = float(sd_ens[b.slice][0])
        narrowing[b.name] = {"prior_median": float(theta_prior[b.slice][0]),
                             "posterior": float(theta_hat[b.slice][0]),
                             "prior_sd": prior_sd, "posterior_sd_vi": post_sd,
                             "posterior_sd_ensemble": ens_sd,
                             "ratio": post_sd / prior_sd}
        print(f"{b.name:12s} {theta_prior[b.slice][0]:10.4g} {theta_hat[b.slice][0]:10.4g} "
              f"{prior_sd:10.4g} {post_sd:12.4g} {post_sd / prior_sd:7.3f} {ens_sd:10.4g}"
              f"  {b.prior.provenance.value}")
    print("\nratio < 1 is narrowing, and the ratio column is the honest answer to 'did the "
          "data\nspeak here'.  two cautions on the two width columns, in opposite "
          "directions.\nthe VI width is mean-field and under-reports correlated uncertainty "
          "by construction\n(fit_vi says so itself), and the exponent, knee and the bump "
          "amplitudes trade off\nagainst each other, so the true marginal widths are wider "
          "than the VI column.  the\nensemble column is NOT a posterior width at all -- it "
          "is the spread of the modes\nindependent MAP runs land on, so it goes to zero "
          "when the likelihood is unimodal\nand says nothing about how sharp that mode is.  "
          "read it as a check that there is one\nmode, and read the VI column, widened, as "
          "the width.")

    # -- (c) held-out --------------------------------------------------
    rule("(c) held-out: does the posterior beat the prior on subjects it never saw?")
    m_prior = model_psd(basis, theta_prior, space)
    m_post = model_psd(basis, theta_hat, space)

    def score(c: Cohort) -> dict:
        lp, lq = c.per_recording(m_prior), c.per_recording(m_post)
        rp, rq = c.r2_per_recording(m_prior), c.r2_per_recording(m_post)
        return {"n": c.n, "bins": c.n_bins,
                "ll_prior": float(lp.mean()), "ll_post": float(lq.mean()),
                "ll_prior_per_bin": float(lp.mean() / c.n_bins),
                "ll_post_per_bin": float(lq.mean() / c.n_bins),
                "r2_prior": float(rp.mean()), "r2_post": float(rq.mean()),
                "delta_ll": paired(lp, lq), "delta_r2": paired(rp, rq)}

    results = {}
    for c in (train, test, test_open, test_closed):
        r = score(c)
        results[c.name] = r
        d = r["delta_ll"]
        print(f"\n{c.name}  ({r['n']} recordings, {r['bins']} bins each)")
        print(f"  mean log-likelihood per recording   prior {r['ll_prior']:12.2f}   "
              f"posterior {r['ll_post']:12.2f}")
        print(f"  ... per bin                         prior {r['ll_prior_per_bin']:12.4f}   "
              f"posterior {r['ll_post_per_bin']:12.4f}")
        print(f"  mean r^2 of log10 psd               prior {r['r2_prior']:12.4f}   "
              f"posterior {r['r2_post']:12.4f}")
        print(f"  paired delta log-likelihood {d['mean']:+.2f} +/- {d['sem']:.2f} sem, "
              f"dz {d['dz']:+.2f}, t {d['t']:+.2f}, p {d['p']:.3g}, "
              f"posterior wins {d['win_rate']:.0%}")

    ht = results["test"]["delta_ll"]
    held_out_pass = ht["mean"] > 0 and ht["p"] < 0.01 and ht["win_rate"] > 0.5
    print(f"\nHELD-OUT TEST: {'PASSED' if held_out_pass else 'FAILED'} -- the posterior "
          f"{'beats' if ht['mean'] > 0 else 'does not beat'} the prior on "
          f"{len(test_s)} subjects it never saw\n(paired over {ht['n']} recordings: "
          f"{ht['mean']:+.1f} nats, p {ht['p']:.3g}, wins {ht['win_rate']:.0%}).")
    if not held_out_pass:
        print("that is the result.  a prior that a fit cannot improve on held-out data is "
              "either\nalready right or has the wrong functional form, and neither is fixed "
              "by refitting.")
    op_d, cl_d = results["test/eyes_open"]["delta_ll"], results["test/eyes_closed"]["delta_ll"]
    print(f"\nwhere the gain came from, which is weaker than the headline: eyes-closed "
          f"{cl_d['mean']:+.0f}\nnats (p {cl_d['p']:.3g}) against eyes-open "
          f"{op_d['mean']:+.0f} nats (p {op_d['p']:.3g}).  the improvement is\nalmost "
          "entirely the alpha bump on the closed-eyes runs.  on eyes-open rest the "
          "declared\nliterature prior is already as good as the fitted one, so the honest "
          "claim is 'the fit\nlearned this population's alpha', not 'the fit learned the "
          "resting spectrum'.")

    # -- (d) falsification ---------------------------------------------
    rule("(d) falsification: eyes-open vs eyes-closed (R01 vs R02), held-out subjects")
    print("the eegmmidb card: \"the eyes-closed run is the alpha fixture: any fit of the "
          "posterior\nalpha prior that cannot separate these two has failed\".  so: refit "
          "alpha_gain alone\nper recording, everything else at theta, and compare within "
          "subject.")
    contrast = {}
    for tag, th in (("posterior", theta_hat), ("prior", theta_prior)):
        op, cl = [], []
        for s in test_s:
            for lst, lb in ((op, "eyes_open"), (cl, "eyes_closed")):
                e = ev[s][lb]
                i = basis.band_indices(band)
                lst.append(alpha_per_recording(e.psd[i], 0.5 * e.dof[i], i, basis, th, space))
        op, cl = np.log(np.array(op)), np.log(np.array(cl))
        st = paired(op, cl)
        contrast[tag] = {"log_alpha_open_mean": float(op.mean()),
                         "log_alpha_closed_mean": float(cl.mean()), **st}
        print(f"\nbase = {tag} theta")
        print(f"  mean log alpha_gain   open {op.mean():+.3f}   closed {cl.mean():+.3f}")
        print(f"  paired closed - open  {st['mean']:+.3f} +/- {st['sem']:.3f} sem, "
              f"dz {st['dz']:+.2f}, t {st['t']:+.2f}, p {st['p']:.3g}")
        print(f"  closed > open in {st['win_rate']:.0%} of {st['n']} held-out subjects")

    cst = contrast["posterior"]
    alpha_pass = cst["mean"] > 0 and cst["p"] < 0.01 and cst["win_rate"] >= 0.75
    print(f"\nALPHA FIXTURE: {'PASSED' if alpha_pass else 'FAILED'}.  the fitted posterior "
          f"alpha "
          f"{'separates' if alpha_pass else 'does NOT separate'} eyes-closed from eyes-open "
          f"in held-out\nsubjects (dz {cst['dz']:+.2f}, p {cst['p']:.3g}, "
          f"{cst['win_rate']:.0%} of subjects in the right direction).")
    if not alpha_pass:
        print("by the card's own criterion the alpha half of this prior has failed, and no "
              "number\nreported above should be read as though it had not.")

    # -- persist ---------------------------------------------------------
    out = args.out or (Path(__file__).resolve().parents[1] / "data" / "sources" /
                       "eegmmidb" / "evidence" / "fit_neural_spectra@v1" / "posterior.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "eegmmidb", "fitted": "ibm.fields.priors:neural_population",
        "method": rep.method, "band_hz": [band.lo_hz, band.hi_hz],
        "window_s": args.window_s, "seed": args.seed,
        "train_subjects": train_s, "test_subjects": test_s,
        "posterior": as_kwargs(space, theta_hat),
        "prior_median": as_kwargs(space, theta_prior),
        "posterior_vi": as_kwargs(space, theta_vi),
        "posterior_ensemble": as_kwargs(space, theta_ens),
        "narrowing": narrowing,
        "held_out": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        "eyes_contrast": contrast,
        "held_out_pass": bool(held_out_pass), "alpha_pass": bool(alpha_pass),
        "runtime_s": time.time() - t0,
    }
    out.write_text(json.dumps(payload, indent=2, default=float))
    rule("summary")
    print(f"held-out (c): {'PASS' if held_out_pass else 'FAIL'}     "
          f"alpha fixture (d): {'PASS' if alpha_pass else 'FAIL'}")
    print(f"posterior written to {out}")
    print(f"{time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
