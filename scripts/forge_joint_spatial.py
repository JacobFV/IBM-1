#!/usr/bin/env python
"""is the joint-forging conflict a fact about cortex, or about three montages?

`scripts/forge_joint.py` fitted one global value per parameter against four
sources and found the three electrophysiological ones disagree at 6 to 88 sigma
on 18 of 19 shared parameters -- `alpha_resonator.f0_hz` at 4.33, 10.4 and
17.4 Hz -- while the joint theta beat each source's own fit on 0 of 4.  a
wake-against-N2 positive control fires, so the detector has power.  taken at face
value that falsifies the shared-parameter architecture, which is the single
claim the whole approach rests on.

it cannot be taken at face value, and that script says why in its own
`parameter_space` docstring: every tying was collapsed from the declaration to
GLOBAL, while `ei_loop_lti` and `alpha_resonator` both declare PER_PARTITION.
so the falsified claim -- one global parameter set describes a bipolar Fpz-Cz
derivation, a posterior scalp average and a magnetometer array simultaneously --
is a claim the architecture does not make.  the previous agent was also right
that expanding to the declared tying is NOT the fix: three montages that each
integrate the whole sheet into one number cannot identify 34 areas by 6 layers,
and the map would come back as the prior wearing a fit's provenance.

### what makes it identifiable instead

two changes, and only two, so that what they buy is measurable.

**a low-dimensional spatial factor.**  each shared parameter gets a global value
and ONE anterior-posterior gradient coefficient,

    theta_p(q)  =  theta_p0  exp( g_p a(q) )

with `a(q)` the standardized anterior-posterior coordinate of the white surface.
that is two numbers per parameter rather than 204, and three montages have rank
three, so it is a design that can actually be estimated.  multiplicative rather
than additive because every one of these parameters is positive and disagreed
about by factors -- the same reason `priors.py` fits in the log -- and because an
additive gradient can drive a time constant negative at one end of the sheet,
where the likelihood is a flat penalty and there is no gradient to walk back out.

**a real forward weighting per source.**  this is the factor the earlier
experiment omitted entirely and named as one of three causes it could not
separate.  `ibm.forge.montage` solves the mne `sample` subject's three-layer BEM
over the white surface and projects it onto each source's actual montage: eleven
monopolar parieto-occipital electrodes for eegmmidb, the Fpz-Cz and Pz-Oz bipolar
pair for sleep-edfx, ds000117's own digitised 74-electrode cap, the Vectorview
magnetometer array, and -- for a grey-matter median -- the sheet's own area.
what a source reports is then a genuine mixture,

    P_s(f)  =  sum_k w_sk  S(f; theta_0 exp(g a_k))

over equal-area bins of `a`, with `w_sk` that montage's power sensitivity.  every
approximation in that weighting is listed by `MontageWeights.approximations` and
printed below rather than buried.

### the control, and why it is run in this process

the OLD model is re-run here as `--mode old`: gradients pinned at zero and every
weight uniform, which is algebraically the mixture collapsing to the single
global shape `forge_joint` fitted.  it shares this file's readers, splits, bands,
nuisance grids, overdispersion weighting and conflict statistic with the NEW
model, so the difference between the two columns is the spatial factor and the
montage weighting and nothing else.  it also has to reproduce
`data/joint/forge_joint@v1/posterior.json` exactly, and if it does not that is
the finding and the rest of this script means nothing.

### how to read the z column, and how not to

adding nine gradient coefficients gives every source more freedom, and more
freedom widens a marginal laplace sd whether or not it explains anything.  a z
that falls because the error bars grew is not a conflict that collapsed.  so the
table reports the disagreement and the width SEPARATELY -- `d_u`, the difference
in the unconstrained coordinate, beside `sd`, the quadrature width -- and the
verdict is read off `d_u`.  and §4.3(ii) stands: `Method.JOINT` on this graph is
multimodal, restarts move the posterior by up to 209 marginal sd, so no claim
below rests on a posterior having moved.  what it rests on is held-out
likelihood, swept nats, and laplace widths from each source's own hessian.

run:  ./.venv/bin/python scripts/forge_joint_spatial.py --mode both
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ibm
from forge_joint import (AR, BOLD_TR_S, EI, ELECTRO, FITTED, NVC, VASCULAR, WK, Axis,
                         Cohort, Dataset, ML_O2_PER_UMOL, Scalars, bold_shape,
                         calibrated_weights, local_shape, paired, prior_sd_natural,
                         prior_sd_u, read_ds000117, read_ds004873, read_eegmmidb,
                         read_sleep, rule, short, sleep_subject, split_by_subject)
from ibm.forge.fit import Method, Task, fit, laplace_sd
from ibm.forge.montage import cortical_field, montage_weight_table
from ibm.forge.priors import ParameterSpace, assemble
from ibm.registry import REGISTRY, Form, Implementation
from ibm.vocabulary import Band, Prior, Provenance, Tying

#: the process id the gradient coefficients are filed under.  a made-up id and
#: not one of the registry's, because these are not a declared process's
#: parameters: they are a reparameterization of a PER_PARTITION tying down to the
#: rank three montages can support, and filing them under `local_excitation`
#: would put a number in that process's posterior that its card never declared.
GRAD = "spatial_gradient"


def gradient_prior() -> Prior:
    """normal(0, 1) on `g`, in log-parameter per sd of anterior-posterior position.

    deliberately loose.  the honest physiological expectation for the best-known
    of these gradients -- alpha slowing from occiput to front -- is a couple of Hz
    across the sheet, which is |g| near 0.15; a width of 1 lets the data ask for a
    factor of e per sd, roughly fifty-fold across the cortex's four-sd span.  a
    tighter prior would guarantee the gradient could not absorb the conflict and
    would make the null uninterpretable, which is the one outcome this experiment
    cannot afford.  the cost is real and is reported: nine loose coefficients
    widen every marginal width, which is exactly why the table below prints the
    disagreement `d_u` beside the width instead of only their ratio.
    """
    return Prior("normal", 0.0, 1.0, provenance=Provenance.WEAK,
                 source="weak by construction; see forge_joint_spatial.gradient_prior")


def parameter_space(*, gradient: bool) -> ParameterSpace:
    """`forge_joint.parameter_space`, plus one gradient coefficient per shared parameter.

    the process blocks are assembled through the ordinary `assemble` door and
    collapsed to GLOBAL exactly as before, so the global half of this space is
    bit-identical to the one the conflict was found in.  what is added is a second
    implementation whose parameters are named after the electrophysiological
    blocks they modulate -- `ei_loop_lti.tau_membrane_s` and so on -- so that a
    reader of the posterior can never mistake a gradient for a value.

    only the ELECTRO blocks get a gradient.  the vascular chain is identified here
    by a grey-matter median and a whole-brain BOLD residual, both of which weigh
    the sheet uniformly, so a gradient on them would be a parameter no factor in
    this corpus can distinguish from zero -- the exact failure the per-partition
    expansion was rejected for.
    """
    impls = []
    for key, names in FITTED.items():
        src = REGISTRY.implementations[key]
        impls.append(replace(src, params={n: src.params[n] for n in names},
                             tying=Tying.GLOBAL))
    if gradient:
        impls.append(Implementation(
            name="ap", process=GRAD,
            doc="anterior-posterior gradient of each shared electrophysiological parameter, "
                "in log units per sd of position; theta(q) = theta_0 exp(g a(q))",
            form=Form.LTI, tying=Tying.GLOBAL, provenance=Provenance.WEAK,
            params={short(k): gradient_prior() for k in ELECTRO}))
    return assemble(implementations=impls)


GRAD_KEYS = tuple(f"{GRAD}:ap.{short(k)}" for k in ELECTRO)


def unpack(space: ParameterSpace, theta: np.ndarray) -> dict[str, dict[str, float]]:
    return {k: {n: float(v[0]) for n, v in d.items()}
            for k, d in space.unpack(theta).items()}


# ---------------------------------------------------------------------------
# the mixture over cortex
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mixture:
    """one source's discretized distribution over the anterior-posterior coordinate.

    equal-AREA bins of `a`, with each bin's representative coordinate taken as
    that SOURCE's own weighted mean inside the bin rather than the sheet's.  the
    difference matters at the ends: a parieto-occipital montage puts most of its
    sensitivity in the posterior bin and none of it at that bin's anterior edge,
    and using the sheet's bin centre would report the montage as looking further
    forward than it does.
    """

    name: str
    a: np.ndarray                 # (K,) representative coordinate per bin
    w: np.ndarray                 # (K,) sums to 1
    mean_a: float
    sd_a: float

    @property
    def k(self) -> int:
        return int(self.a.size)


def mixtures(field, table, *, k_bins: int) -> dict[str, Mixture]:
    """bin every montage's sensitivity onto one set of equal-area bins of `a`.

    one set of edges for every source, because the bins are a discretization of
    the same integral and letting each source choose its own would make the
    numerical error a function of the montage -- which is precisely the thing
    under test.
    """
    order = np.argsort(field.ap)
    cum = np.cumsum(field.area_mm2[order])
    cum /= cum[-1]
    edges = [-np.inf]
    for j in range(1, k_bins):
        edges.append(float(field.ap[order][int(np.searchsorted(cum, j / k_bins))]))
    edges.append(np.inf)
    idx = np.clip(np.searchsorted(np.array(edges[1:-1]), field.ap, side="right"),
                  0, k_bins - 1)
    out = {}
    for name, mw in table.items():
        w = np.array([mw.w[idx == j].sum() for j in range(k_bins)])
        a = np.array([float(np.dot(mw.w[idx == j], field.ap[idx == j]) / w[j]) if w[j] > 0
                      else float(np.mean(field.ap[idx == j])) for j in range(k_bins)])
        keep = w > 1e-9
        a, w = a[keep], w[keep] / w[keep].sum()
        mu = float(np.dot(w, a))
        out[name] = Mixture(name, a, w, mu,
                            float(math.sqrt(max(np.dot(w, (a - mu) ** 2), 0.0))))
    return out


#: the flat mixture the OLD model is: one bin at the sheet's own centre, weight
#: one.  written as a `Mixture` rather than as a branch so that the two models go
#: through the same code and the control is a control of THIS code.
FLAT = Mixture("flat", np.zeros(1), np.ones(1), 0.0, 0.0)


def scaled(th: dict[str, dict[str, float]], g: dict[str, float] | None, a: float):
    """theta at anterior-posterior coordinate `a`."""
    if g is None or a == 0.0:
        return th
    out = {k: dict(v) for k, v in th.items()}
    for key in ELECTRO:
        impl, name = key.rsplit(".", 1)
        out[impl][name] = th[impl][name] * math.exp(g[short(key)] * a)
    return out


class _ShapeCache:
    """the K per-bin shapes for the current theta, keyed by the frequency axis.

    ds000117 contributes two cohorts on identical bins, and every restart,
    polish, hessian and sweep re-evaluates the same objective, so a one-theta
    memo removes a factor of two from the inner loop for free.  keyed by theta's
    bytes, so a stale entry is impossible rather than unlikely.
    """

    def __init__(self) -> None:
        self.key = None
        self.store: dict[tuple[int, int], list] = {}

    def get(self, theta_bytes, axis, k, build):
        if theta_bytes != self.key:
            self.key, self.store = theta_bytes, {}
        ident = (id(axis), k)  # one axis belongs to one cohort, and one cohort to one mixture
        if ident not in self.store:
            self.store[ident] = build()
        return self.store[ident]


CACHE = _ShapeCache()


@dataclass
class SpatialDataset(Dataset):
    """`forge_joint.Dataset` whose factors see cortex through their own montage.

    the only override is `shape_for`.  the splits, the nuisance profile, the
    overdispersion weighting, the held-out scoring and the task construction are
    inherited unchanged, which is what makes the OLD and NEW columns comparable:
    there is no second copy of any of them to drift.
    """

    mix: dict[str, Mixture] = field(default_factory=dict)
    gradient: bool = False

    def shape_for(self, factor, th):
        if isinstance(factor, Scalars):
            return None
        m = self.mix.get(factor.name, FLAT)
        g = th.get(f"{GRAD}:ap") if self.gradient else None
        base = bold_shape if factor.kind == "bold" else local_shape
        key = repr(sorted((k, tuple(sorted(v.items()))) for k, v in th.items()))

        def build():
            return [base(factor.axis, scaled(th, g, float(ai))) for ai in m.a]

        parts = CACHE.get(key, factor.axis, m.k, build)
        if any(p is None for p in parts):
            return None
        out = np.zeros_like(parts[0])
        for wi, p in zip(m.w, parts):
            out = out + wi * p
        return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build(args, cached, mix):
    """read the four sources, split by subject, and hand back both models' datasets.

    read ONCE and split once.  the old and the new model must see the same
    spectra, the same subjects on the same side of the same split, the same bands
    and the same nuisance grids, or the difference between their conflict tables
    is a difference of cohorts.
    """
    print("\neegmmidb -- 64-ch scalp EEG at 160 Hz, resting eyes-closed (R02)")
    eeg_ev, eeg_root = cached["eeg"] if cached else read_eegmmidb(args.eeg_subjects, 4.0)
    print(f"  {eeg_root}\n  {len(eeg_ev)} subjects, one run each.")
    sleep_band = Band(1.0, 25.0)
    print("\nsleep-edfx -- 2 EEG derivations at 100 Hz, expert-scored, WAKE epochs")
    sl_ev, sl_root = cached["sleep"] if cached else read_sleep(
        args.sleep_nights, 4.0, args.sleep_epochs, sleep_band)
    sl_nights = sorted(sl_ev)
    sl_subj = sorted({sleep_subject(n) for n in sl_nights})
    print(f"  {sl_root}\n  {len(sl_nights)} nights from {len(sl_subj)} participants.")
    print("\nds000117 -- 102 magnetometers and 70 EEG channels, simultaneous, same head")
    meg_ev, meg_root = cached["meg"] if cached else read_ds000117(
        args.meg_subjects, 4.0, args.meg_seconds)
    print(f"  {meg_root}\n  {len(meg_ev)} participants x 2 instruments.")
    print("\nds004873 -- quantitative CBF/CBV/OEF/CMRO2 and a task BOLD series")
    fm_recs, fm_root = cached["fmri"] if cached else read_ds004873(
        args.fmri_subjects, 128 * BOLD_TR_S)
    print(f"  {fm_root}\n  {len(fm_recs)} subjects.")

    if args.cache is not None and cached is None:
        import pickle
        sig = (args.eeg_subjects, args.sleep_nights, args.meg_subjects, args.fmri_subjects,
               args.sleep_epochs, args.meg_seconds)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with args.cache.open("wb") as fh:
            pickle.dump({"signature": sig, "eeg": (eeg_ev, eeg_root),
                         "sleep": (sl_ev, sl_root), "meg": (meg_ev, meg_root),
                         "fmri": (fm_recs, fm_root)}, fh)

    rule("splits: by subject, always -- and identical for both models")
    eeg_tr, eeg_te = split_by_subject(sorted(eeg_ev), args.train_frac, args.seed)
    sl_str, sl_ste = split_by_subject(sl_subj, args.train_frac, args.seed + 1)
    sl_tr = [n for n in sl_nights if sleep_subject(n) in sl_str]
    sl_te = [n for n in sl_nights if sleep_subject(n) in sl_ste]
    meg_tr, meg_te = split_by_subject(sorted(meg_ev), args.train_frac, args.seed + 2)
    fm_ids = [r["subject"] for r in fm_recs]
    fm_tr, fm_te = split_by_subject(fm_ids, args.train_frac, args.seed + 3)
    print(f"eegmmidb   train {len(eeg_tr):3d} / test {len(eeg_te):3d} subjects")
    print(f"sleep-edfx train {len(sl_str):3d} / test {len(sl_ste):3d} participants")
    print(f"ds000117   train {len(meg_tr):3d} / test {len(meg_te):3d} participants")
    print(f"ds004873   train {len(fm_tr):3d} / test {len(fm_te):3d} subjects")

    e_grid = np.arange(-2.0, 3.41, 0.2)
    floor_grid = np.array([0.0, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])

    def spectral(name, source, evs, subs, band, kind="electro", floors=None):
        b = evs[0].basis
        idx = b.band_indices(band)
        if source == "eegmmidb":
            f = b.freqs_hz[idx]
            idx = idx[~((f >= 57.0) & (f < 63.0))]
        return Cohort(name, source, tuple(subs), b.freqs_hz[idx],
                      np.stack([e.psd[idx] for e in evs]),
                      np.stack([0.5 * e.dof[idx] for e in evs]),
                      e_grid, floors, kind)

    eeg_band, meg_band = Band(1.0, 75.0), Band(2.0, 45.0)
    by_id = {r["subject"]: r for r in fm_recs}

    def fmri_factors(ids, sds):
        evs = [by_id[s]["bold"] for s in ids]
        fs = [spectral("ds004873/bold", "ds004873", evs, ids, Band(0.013, 0.35),
                       kind="bold", floors=floor_grid)]
        obs = {
            "cbf": ([by_id[s]["cbf_control"] for s in ids], ids,
                    lambda th: th["vascular_flow:poiseuille_network"]["baseline_cbf"]),
            "cbv": ([by_id[s]["cbv_control"] for s in ids], ids,
                    lambda th: 100.0 * th["vascular_flow:poiseuille_network"]["baseline_cbv"]),
            "oef": ([by_id[s]["oef_control"] for s in ids], ids,
                    lambda th: th["tissue_exchange:fick_oxygen_limitation"]
                    ["oxygen_extraction_fraction"]),
            "cmro2": ([by_id[s]["cmro2_control"] for s in ids], ids,
                      lambda th: th["metabolism:mass_action_budget"]["cmro2_baseline"]
                      / ML_O2_PER_UMOL)}
        fick_y = np.array([by_id[s]["cmro2_control"] for s in ids])
        fick_x = np.array([by_id[s]["cbf_control"] * by_id[s]["oef_control"] for s in ids])
        n_ids = [s for s in ids if "cmro2_calc" in by_id[s]]
        n_y = np.array([
            ((by_id[s]["cbf_calc"] - by_id[s]["cbf_control"]) / by_id[s]["cbf_control"])
            / ((by_id[s]["cmro2_calc"] - by_id[s]["cmro2_control"])
               / by_id[s]["cmro2_control"]) for s in n_ids]) if n_ids else np.zeros(0)
        keep = (np.isfinite(n_y) & (n_y > 0) & (n_y < 20)) if n_y.size else np.zeros(0, bool)
        out_sds = {}
        for k, (y, sub, pred) in obs.items():
            y = np.asarray(y, float)
            sd = sds[k] if sds else float(max(y.std(ddof=1), 1e-6))
            out_sds[k] = sd
            fs.append(Scalars(f"ds004873/{k}", "ds004873", tuple(sub), y, sd, pred))
        resid = fick_y - fick_x * 0.2 / ML_O2_PER_UMOL
        sd_f = sds["fick"] if sds else float(max(resid.std(ddof=1), 1e-6))
        out_sds["fick"] = sd_f
        fs.append(Scalars("ds004873/fick", "ds004873", tuple(ids), fick_y, sd_f,
                          lambda th, x=fick_x: x * th["tissue_exchange:fick_oxygen_limitation"]
                          ["arterial_oxygen_content"] / ML_O2_PER_UMOL))
        if keep.size and keep.sum() >= 3:
            yy = n_y[keep]
            sd_n = (sds.get("n") if sds else None) or float(max(yy.std(ddof=1), 1e-6))
            out_sds["n"] = sd_n
            fs.append(Scalars("ds004873/coupling_n", "ds004873", tuple(np.array(n_ids)[keep]),
                              yy, sd_n,
                              lambda th: th["tissue_exchange:linearized_exchange_lti"]
                              ["coupling_ratio_n"]))
        return fs, out_sds

    fm_train_f, fm_sds = fmri_factors(fm_tr, None)
    fm_test_f, _ = fmri_factors(fm_te, fm_sds)

    #: which montage weighting each spectral factor is seen through.  a factor and
    #: not a dataset, because ds000117 puts two instruments on one head and a
    #: magnetometer array and a 74-electrode cap weigh cortex differently.
    def make(gradient: bool):
        moves_e = ELECTRO + (GRAD_KEYS if gradient else ())
        ds = {}
        ds["eegmmidb"] = SpatialDataset(
            key="eegmmidb", source="eegmmidb", moves=moves_e,
            train=[spectral("eegmmidb/train", "eegmmidb", [eeg_ev[s] for s in eeg_tr],
                            eeg_tr, eeg_band)],
            test=[spectral("eegmmidb/test", "eegmmidb", [eeg_ev[s] for s in eeg_te],
                           eeg_te, eeg_band)],
            n_train_subjects=len(eeg_tr), n_test_subjects=len(eeg_te),
            band="1-75 Hz, 57-63 excluded", note="resting eyes-closed, posterior montage",
            mix={"eegmmidb/train": mix["eegmmidb"], "eegmmidb/test": mix["eegmmidb"]},
            gradient=gradient)
        ds["sleep-edfx"] = SpatialDataset(
            key="sleep-edfx", source="sleep-edfx", moves=moves_e,
            train=[spectral("sleep/train", "sleep-edfx", [sl_ev[n]["W"] for n in sl_tr],
                            [sleep_subject(n) for n in sl_tr], sleep_band)],
            test=[spectral("sleep/test", "sleep-edfx", [sl_ev[n]["W"] for n in sl_te],
                           [sleep_subject(n) for n in sl_te], sleep_band)],
            n_train_subjects=len(sl_str), n_test_subjects=len(sl_ste),
            band="1-25 Hz", note="scored wake, Fpz-Cz and Pz-Oz",
            mix={"sleep/train": mix["sleep-edfx"], "sleep/test": mix["sleep-edfx"]},
            gradient=gradient)
        ds["ds000117"] = SpatialDataset(
            key="ds000117", source="ds000117", moves=moves_e,
            train=[spectral(f"ds000117/train/{i}", "ds000117", [meg_ev[s][i] for s in meg_tr],
                            meg_tr, meg_band) for i in ("meg", "eeg")],
            test=[spectral(f"ds000117/test/{i}", "ds000117", [meg_ev[s][i] for s in meg_te],
                           meg_te, meg_band) for i in ("meg", "eeg")],
            n_train_subjects=len(meg_tr), n_test_subjects=len(meg_te),
            band="2-45 Hz", note="magnetometers and EEG, same head, same clock",
            mix={f"ds000117/{s}/{i}": mix[f"ds000117/{i}"]
                 for s in ("train", "test") for i in ("meg", "eeg")},
            gradient=gradient)
        ds["ds004873"] = SpatialDataset(
            key="ds004873", source="ds004873", moves=moves_e + VASCULAR,
            train=fm_train_f, test=fm_test_f,
            n_train_subjects=len(fm_tr), n_test_subjects=len(fm_te),
            band="0.013-0.35 Hz spectrum + per-subject baselines",
            note="quantitative physiology and the task-regressed BOLD residual",
            mix={"ds004873/bold": mix["ds004873"]}, gradient=gradient)
        for k, d in ds.items():
            d.weights = calibrated_weights(d, instruments=2 if k == "ds000117" else 1)
        n2 = [n for n in sl_nights if "N2" in sl_ev[n]]
        ctrl = SpatialDataset(
            key="sleep-edfx/N2", source="sleep-edfx", moves=moves_e,
            train=[spectral("sleep/N2", "sleep-edfx", [sl_ev[n]["N2"] for n in n2],
                            [sleep_subject(n) for n in n2], sleep_band)],
            test=[], n_train_subjects=len({sleep_subject(n) for n in n2}), n_test_subjects=0,
            band="1-25 Hz", note="scored N2, the same people",
            mix={"sleep/N2": mix["sleep-edfx"]}, gradient=gradient)
        ctrl.weights = calibrated_weights(ctrl)
        return ds, ctrl

    return make


def run_model(label, space, datasets, n2_ds, args):
    """fit every source alone, jointly, and leave-one-out; then the widths.

    the polish loop is `forge_joint.do_fit`'s, restated here because that one is a
    closure over its own `main`.  it exists because `fit_map` asks L-BFGS-B for a
    tolerance it cannot reach through a numerically differentiated, grid-profiled
    objective and then reports NOT CONVERGED; refitting from the previous answer
    and keeping the best point seen distinguishes "the optimizer gave up" from
    "the optimizer is at the mode", and the gain of the last pass is printed so a
    reader can tell which happened.
    """
    def do_fit(keys_in, tag, *, theta0=None, weighted=True):
        tasks = [datasets[k].task(space, weighted=weighted) for k in keys_in]
        th, gain, lp0 = theta0, float("inf"), None
        best_th, best_rep, best_lp = None, None, -np.inf
        for _ in range(args.polish):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                th_new, rep = fit(space, tasks, method=Method.JOINT, theta0=th,
                                  max_iter=args.max_iter)
            lp0 = rep.log_posterior0 if lp0 is None else lp0
            th = th_new
            if rep.log_posterior <= best_lp:
                break
            gain = rep.log_posterior - rep.log_posterior0
            best_th, best_rep, best_lp = th.copy(), rep, rep.log_posterior
            if gain < 1e-3:
                break
        best_rep.log_posterior0 = lp0
        print(f"  {label}/{tag:34s} log posterior {lp0:.6g} -> {best_lp:.6g}, "
              f"last pass gained {gain:.3g} nats")
        return best_th, best_rep, tasks

    out = {"alone": {}, "alone_tasks": {}, "lodo": {}}
    for k in datasets:
        out["alone"][k], _, out["alone_tasks"][k] = do_fit([k], f"{k} alone")
    out["joint"], out["joint_rep"], out["joint_tasks"] = do_fit(list(datasets), "JOINT")
    rng = np.random.default_rng(args.seed + 100)
    out["restarts"] = [do_fit(list(datasets), f"JOINT restart {i + 1}",
                              theta0=space.sample(rng))[0] for i in range(args.restarts)]
    for k in datasets:
        out["lodo"][k], _, _ = do_fit([x for x in datasets if x != k], f"leave out {k}")
    th_n2 = None
    for _ in range(args.polish):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            th_n2, rep_n2 = fit(space, [n2_ds.task(space)], method=Method.JOINT,
                                theta0=th_n2, max_iter=args.max_iter)
        if rep_n2.converged or rep_n2.log_posterior - rep_n2.log_posterior0 < 1e-3:
            break
    out["n2"] = th_n2
    out["n2_tasks"] = [n2_ds.task(space)]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out["sd"] = {}
        out["pd"] = {}
        for k in datasets:
            out["sd"][k], out["pd"][k] = laplace_sd(space, out["alone_tasks"][k],
                                                    out["alone"][k])
        out["sd"]["sleep-edfx/N2"], out["pd"]["sleep-edfx/N2"] = laplace_sd(
            space, out["n2_tasks"], out["n2"])
        out["sd_joint"], out["pd_joint"] = laplace_sd(space, out["joint_tasks"], out["joint"])
    return out


def conflict_rows(space, res, keys, theta0_ref, opinion=0.25):
    """every pair of sources, on every ELECTRO block both of them moved.

    `forge_joint.conflicts`' statistic exactly: the difference in the
    unconstrained coordinate over the quadrature sum of the two marginal laplace
    widths, and a row is live only when BOTH sides moved the parameter more than a
    quarter of a prior sd off the prior median.  the reason that gate exists is
    the trap it prevents: a source carrying no information leaves a parameter at
    the prior with the prior's width, and its `z` against a source that did move
    it is one dataset disagreeing with the literature wearing a second dataset's
    name.
    """
    us = {k: space.to_unconstrained(res["alone"][k]) for k in keys}
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka, kb = keys[i], keys[j]
            ua, ub, sa, sb = us[ka], us[kb], res["sd"][ka], res["sd"][kb]
            ta, tb = space.to_natural(ua), space.to_natural(ub)
            for b in space.blocks:
                if b.key not in ELECTRO:
                    continue
                s2 = sa[b.slice][0] ** 2 + sb[b.slice][0] ** 2
                if not np.isfinite(s2) or s2 <= 0:
                    continue
                psd = prior_sd_natural(b)
                ma = abs(ta[b.slice][0] - theta0_ref[b.slice][0]) / psd
                mb = abs(tb[b.slice][0] - theta0_ref[b.slice][0]) / psd
                du = float(ua[b.slice][0] - ub[b.slice][0])
                rows.append({"a": ka, "b": kb, "param": b.key, "z": du / math.sqrt(s2),
                             "d_u": du, "sd": float(math.sqrt(s2)),
                             "theta_a": float(ta[b.slice][0]), "theta_b": float(tb[b.slice][0]),
                             "sd_a": float(sa[b.slice][0]), "sd_b": float(sb[b.slice][0]),
                             "both_have_an_opinion": bool(ma > opinion and mb > opinion)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eeg-subjects", type=int, default=60)
    ap.add_argument("--sleep-nights", type=int, default=60)
    ap.add_argument("--meg-subjects", type=int, default=16)
    ap.add_argument("--fmri-subjects", type=int, default=40)
    ap.add_argument("--sleep-epochs", type=int, default=60)
    ap.add_argument("--meg-seconds", type=float, default=240.0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=350)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--polish", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=3000)
    ap.add_argument("--bins", type=int, default=9)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    ibm.load_all(seal=True, strict=True)
    payload = {"argv": sys.argv[1:], "seed": args.seed}

    # -- 1. the cortical field and what each montage weighs -----------------
    rule("1. the forward weighting: what part of cortex each montage actually sees")
    fld = cortical_field(n_nodes=args.nodes, seed=args.seed)
    print(fld.note)
    d117 = None
    from ibm.forge.spectra import local_root
    for p in sorted((local_root("ds000117") / "1.1.0").glob(
            "sub-*/ses-meg/meg/*task-facerecognition_run-01_meg.fif")):
        d117 = p
        break
    table = montage_weight_table(fld, ds000117_raw=d117)
    mix = mixtures(fld, table, k_bins=args.bins)
    print(f"\n{'source':16s} {'mean a':>8s} {'sd a':>7s} {'bins':>5s}  what it is")
    print("-" * 78)
    for k, m in mix.items():
        print(f"{k:16s} {m.mean_a:+8.3f} {m.sd_a:7.3f} {m.k:5d}  {table[k].detail}")
    print("-" * 78)
    print("`a` is the standardized anterior-posterior coordinate of the white surface, so a\n"
          "mean of -0.75 is a montage whose power sensitivity sits three quarters of a sd\n"
          "behind the sheet's centroid.  the SPREAD of these means is what makes a gradient\n"
          "identifiable across sources; the sd within each is what lets one source see it at\n"
          "all, because a wide weighting mixes resonators and broadens the peak.")
    for k, mw in table.items():
        for a in mw.approximations:
            print(f"  ~ {k:16s} {a}")

    means = np.array([[mix[k].mean_a, table[k].mean_ml] for k in mix])
    means = means - means.mean(0)
    sv = np.linalg.svd(means, compute_uv=False)
    print(f"\nconditioning of a two-factor design over these {len(mix)} weightings: singular "
          f"values {sv[0]:.3f} and {sv[1]:.3f},\ncondition number {sv[0] / max(sv[1], 1e-12):.1f}.  "
          "the anterior-posterior spread is "
          f"{means[:, 0].std(ddof=1) / max(means[:, 1].std(ddof=1), 1e-12):.1f}x the "
          "medial-lateral one, and\nwith five weightings of which one is uniform there are "
          "three independent looks at cortex.\nso ONE factor is fitted and the second is "
          "reported and not fitted: nine more coefficients\non a third direction of design "
          "space would be nine more ways for the fit to absorb a\nconflict without having "
          "explained it.")
    payload["montage"] = {k: {"mean_a": mix[k].mean_a, "sd_a": mix[k].sd_a,
                              "mean_ml": table[k].mean_ml, "detail": table[k].detail,
                              "approximations": list(table[k].approximations),
                              "bin_a": mix[k].a.tolist(), "bin_w": mix[k].w.tolist()}
                          for k in mix}
    payload["two_factor_singular_values"] = sv.tolist()

    # -- 2. data ------------------------------------------------------------
    rule("2. data, read once and split once for both models")
    cached = None
    if args.cache is not None and args.cache.is_file():
        import pickle
        with args.cache.open("rb") as fh:
            blob = pickle.load(fh)
        sig = (args.eeg_subjects, args.sleep_nights, args.meg_subjects, args.fmri_subjects,
               args.sleep_epochs, args.meg_seconds)
        if blob.get("signature") == sig:
            cached = blob
            print(f"reusing the extracted spectra in {args.cache}")
    space_old = parameter_space(gradient=False)
    space_new = parameter_space(gradient=True)
    make = build(args, cached, mix)
    ds_old, n2_old = make(False)
    for d in list(ds_old.values()) + [n2_old]:
        d.mix = {k: FLAT for k in d.mix}
    ds_new, n2_new = make(True)

    print(f"\nOLD: {space_old.size} parameters, every weighting flat, every gradient absent.")
    print(f"NEW: {space_new.size} parameters -- the same {space_old.size} plus "
          f"{len(GRAD_KEYS)} anterior-posterior\n     gradient coefficients, one per shared "
          "electrophysiological parameter.")

    # -- 3. the two fits -----------------------------------------------------
    rule("3. forging, twice: the old model as a control and the new one beside it")
    t0_old = space_old.median()
    t0_new = space_new.median()
    res_old = run_model("OLD", space_old, ds_old, n2_old, args)
    print()
    res_new = run_model("NEW", space_new, ds_new, n2_new, args)

    keys = list(ds_old)
    rows_old = conflict_rows(space_old, res_old, keys, t0_old)
    rows_new = conflict_rows(space_new, res_new, keys, t0_new)

    # -- 4. the conflict table, old against new ------------------------------
    rule("4. THE TEST: the same conflicts, before and after the montage factor")
    idx_new = {(r["a"], r["b"], r["param"]): r for r in rows_new}
    live = [r for r in rows_old if r["both_have_an_opinion"]]
    live.sort(key=lambda r: -abs(r["z"]))
    print(f"{'parameter':30s} {'a':11s} {'b':11s} {'z OLD':>8s} {'z NEW':>8s} "
          f"{'|du| OLD':>9s} {'|du| NEW':>9s} {'sd NEW/OLD':>11s}")
    print("-" * 100)
    absorbed = []
    for r in live:
        n = idx_new[(r["a"], r["b"], r["param"])]
        absorbed.append((r, n))
        print(f"{short(r['param']):30s} {r['a'][:11]:11s} {r['b'][:11]:11s} "
              f"{r['z']:+8.2f} {n['z']:+8.2f} {abs(r['d_u']):9.3f} {abs(n['d_u']):9.3f} "
              f"{n['sd'] / max(r['sd'], 1e-30):11.2f}"
              + ("   live" if n["both_have_an_opinion"] else "   no opinion now"))
    print("-" * 100)
    n3o = sum(1 for r in live if abs(r["z"]) > 3)
    n3n = sum(1 for _, n in absorbed if abs(n["z"]) > 3 and n["both_have_an_opinion"])
    du_o = float(np.mean([abs(r["d_u"]) for r, _ in absorbed]))
    du_n = float(np.mean([abs(n["d_u"]) for _, n in absorbed]))
    z_o = float(np.mean([abs(r["z"]) for r, _ in absorbed]))
    z_n = float(np.mean([abs(n["z"]) for _, n in absorbed]))
    print(f"3 sigma or worse: {n3o} of {len(live)} under the OLD model, {n3n} under the NEW.")
    print(f"mean |z|  {z_o:.1f} -> {z_n:.1f}   ({1 - z_n / max(z_o, 1e-9):+.0%})")
    print(f"mean |d_u| (the DISAGREEMENT itself, free of any width) {du_o:.3f} -> {du_n:.3f}   "
          f"({1 - du_n / max(du_o, 1e-9):+.0%})")
    print("\nread `d_u` and not `z` for the verdict.  nine loose gradient coefficients widen\n"
          "every marginal width whether or not they explain anything, and a z that fell only\n"
          "because its denominator grew is a conflict hidden, not a conflict resolved.")
    payload["conflicts_old"] = rows_old
    payload["conflicts_new"] = rows_new
    payload["absorbed"] = {"n_live": len(live), "n3_old": n3o, "n3_new": n3n,
                           "mean_abs_z_old": z_o, "mean_abs_z_new": z_n,
                           "mean_abs_du_old": du_o, "mean_abs_du_new": du_n}

    # the positive control has to keep firing under the new model, or the whole
    # comparison is a detector that was switched off rather than a conflict that
    # was explained.
    def ctrl_rows(space, res, theta0_ref):
        us_w = space.to_unconstrained(res["alone"]["sleep-edfx"])
        us_n = space.to_unconstrained(res["n2"])
        sa, sb = res["sd"]["sleep-edfx"], res["sd"]["sleep-edfx/N2"]
        out = []
        for b in space.blocks:
            if b.key not in ELECTRO:
                continue
            s2 = sa[b.slice][0] ** 2 + sb[b.slice][0] ** 2
            if not np.isfinite(s2) or s2 <= 0:
                continue
            psd = prior_sd_natural(b)
            ta, tb = space.to_natural(us_w), space.to_natural(us_n)
            ma = abs(ta[b.slice][0] - theta0_ref[b.slice][0]) / psd
            mb = abs(tb[b.slice][0] - theta0_ref[b.slice][0]) / psd
            out.append({"param": b.key, "z": float((us_w[b.slice][0] - us_n[b.slice][0])
                                                   / math.sqrt(s2)),
                        "theta_w": float(ta[b.slice][0]), "theta_n2": float(tb[b.slice][0]),
                        "both_have_an_opinion": bool(ma > 0.25 and mb > 0.25)})
        return out

    rule("4b. the positive control, under both models: the same nights, wake against N2")
    co, cn = ctrl_rows(space_old, res_old, t0_old), ctrl_rows(space_new, res_new, t0_new)
    ci = {r["param"]: r for r in cn}
    print(f"{'parameter':30s} {'theta W':>10s} {'theta N2':>10s} {'z OLD':>8s} {'z NEW':>8s}")
    for r in sorted(co, key=lambda r: -abs(r["z"])):
        n = ci[r["param"]]
        mark = " ***" if r["both_have_an_opinion"] and abs(r["z"]) > 3 else ""
        print(f"{short(r['param']):30s} {n['theta_w']:10.4g} {n['theta_n2']:10.4g} "
              f"{r['z']:+8.2f} {n['z']:+8.2f}{mark}")
    lo = [r for r in co if r["both_have_an_opinion"]]
    ln = [r for r in cn if r["both_have_an_opinion"]]
    print(f"OLD: {sum(1 for r in lo if abs(r['z']) > 3)} of {len(lo)} live comparisons above "
          f"3 sigma.  NEW: {sum(1 for r in ln if abs(r['z']) > 3)} of {len(ln)}.")
    print("arousal genuinely changes cortical E/I and both montages are the SAME montage, so\n"
          "the montage factor must NOT flatten this one.  if it does, the new model has lost\n"
          "power rather than gained an explanation.")
    payload["control_old"], payload["control_new"] = co, cn

    # -- 5. are the gradients real, and in the implied direction? ------------
    rule("5. the gradient coefficients: non-zero, and pointing where the montages imply")
    print("g is in log-parameter per sd of anterior-posterior position, anterior POSITIVE.\n"
          "the width is the marginal laplace sd from that fit's own hessian, in the same\n"
          "units, and the prior is normal(0, 1) so a |g| near 1 is a coefficient asking for\n"
          "everything the prior allows.")
    grad = {}
    print(f"\n{'parameter':30s} " + " ".join(f"{k[:15]:>15s}" for k in ["JOINT"] + keys))
    print("-" * 110)
    for key in GRAD_KEYS:
        b = space_new[key]
        cells, rec = [], {}
        for who, th, sd in [("JOINT", res_new["joint"], res_new["sd_joint"])] + \
                [(k, res_new["alone"][k], res_new["sd"][k]) for k in keys]:
            g = float(th[b.slice][0])
            s = float(sd[b.slice][0])
            cells.append(f"{g:+7.3f}+-{s:5.3f}" if np.isfinite(s) else f"{g:+7.3f}   ---")
            rec[who] = {"g": g, "sd": s, "z": (g / s) if (np.isfinite(s) and s > 0) else None}
        grad[key] = rec
        print(f"{short(key):30s} " + " ".join(f"{c:>15s}" for c in cells))
    print("-" * 110)
    sig = [k for k in GRAD_KEYS if (grad[k]["JOINT"]["z"] or 0) and abs(grad[k]["JOINT"]["z"]) > 2]
    print(f"{len(sig)} of {len(GRAD_KEYS)} joint gradients exceed 2 marginal sd: "
          + (", ".join(short(k) for k in sig) if sig else "none"))
    f0 = grad[f"{GRAD}:ap.{short(f'{AR}.f0_hz')}"]["JOINT"]
    print(f"\nthe one with a literature direction is alpha f0.  the alpha rhythm is FASTER\n"
          "posteriorly and slower frontally, which with anterior-positive `a` is g < 0.\n"
          f"fitted jointly: g = {f0['g']:+.3f} +- {f0['sd']:.3f}"
          + (f" ({f0['z']:+.1f} sd), which is "
             + ("the documented direction" if f0["g"] < 0 else "the OPPOSITE direction")
             if f0["z"] is not None else ""))
    payload["gradients"] = grad

    # -- 6. does joint now beat each source's own fit? ------------------------
    rule("6. TRANSFER: does the joint theta now beat each source's own fit, held out?")
    print("scored per held-out SUBJECT, unweighted, at exactly the likelihood each source\n"
          "contributes -- the same scoring `forge_joint` used, so the OLD column here has to\n"
          "reproduce the OLD column there.")
    transfer = {}
    for k in keys:
        row = {}
        for tag, space, res, ds, t0 in (("old", space_old, res_old, ds_old, t0_old),
                                        ("new", space_new, res_new, ds_new, t0_new)):
            d = ds[k]
            sp = d.per_subject(space, t0)
            so = d.per_subject(space, res["alone"][k])
            sj = d.per_subject(space, res["joint"])
            subs = sorted(sp)
            vp = np.array([sp[s] for s in subs])
            vo = np.array([so[s] for s in subs])
            vj = np.array([sj[s] for s in subs])
            row[tag] = {"n": len(subs), "prior": float(vp.mean()), "own": float(vo.mean()),
                        "joint": float(vj.mean()), "joint_vs_own": paired(vo, vj),
                        "joint_vs_prior": paired(vp, vj), "own_vs_prior": paired(vp, vo)}
        transfer[k] = row
        print(f"\n{k}")
        for tag in ("old", "new"):
            r = row[tag]
            jo = r["joint_vs_own"]
            verdict = ("joint BEATS own" if jo["mean"] > 0 and jo["p"] < 0.05 else
                       "joint WORSE than own" if jo["mean"] < 0 and jo["p"] < 0.05 else
                       "no difference")
            print(f"  {tag.upper():3s} prior {r['prior']:12.2f}  own {r['own']:12.2f}  "
                  f"joint {r['joint']:12.2f}   joint-own {jo['mean']:+11.2f} "
                  f"+-{jo['sem']:.2f}, p {jo['p']:.3g}, wins {jo['win_rate']:.0%}  <- {verdict}")
    n_beat_old = sum(1 for k in keys if transfer[k]["old"]["joint_vs_own"]["mean"] > 0
                     and transfer[k]["old"]["joint_vs_own"]["p"] < 0.05)
    n_beat_new = sum(1 for k in keys if transfer[k]["new"]["joint_vs_own"]["mean"] > 0
                     and transfer[k]["new"]["joint_vs_own"]["p"] < 0.05)
    print(f"\njoint beats the source's own fit on {n_beat_old} of {len(keys)} under the OLD "
          f"model and {n_beat_new} of {len(keys)} under the NEW.")
    payload["transfer"] = transfer

    # -- 7. the restart yardstick --------------------------------------------
    rule("7. the restart yardstick, because Method.JOINT on this graph is multimodal")
    yard = {}
    for tag, space, res in (("old", space_old, res_old), ("new", space_new, res_new)):
        u_j = space.to_unconstrained(res["joint"])
        u_r = [space.to_unconstrained(t) for t in res["restarts"]]
        worst = 0.0
        for b in space.blocks:
            s = res["sd_joint"][b.slice][0]
            if not np.isfinite(s) or s <= 0:
                continue
            worst = max(worst, max(abs(u[b.slice][0] - u_j[b.slice][0]) / s for u in u_r))
        yard[tag] = worst
        print(f"{tag.upper():3s} restarting the joint fit from a prior draw moves the "
              f"posterior by up to {worst:.0f} marginal sd;"
              f" hessian pd at the joint mode: {res['pd_joint']}")
    print("no claim above rests on a posterior having moved.  the conflict statistic is a\n"
          "difference between two SEPARATE per-source fits with widths from their own\n"
          "hessians, and the transfer numbers are held-out likelihood.")
    payload["restart_sd"] = yard
    payload["hessian_pd"] = {"old": {k: bool(v) for k, v in res_old["pd"].items()},
                             "new": {k: bool(v) for k, v in res_new["pd"].items()}}

    # -- 8. where the two posteriors landed ----------------------------------
    rule("8. where each source landed, old against new")
    print(f"{'parameter':28s} {'model':5s} " + " ".join(f"{k[:11]:>11s}" for k in keys)
          + f" {'JOINT':>11s}")
    print("-" * 96)
    for b in space_old.blocks:
        if b.key not in ELECTRO:
            continue
        for tag, space, res in (("old", space_old, res_old), ("new", space_new, res_new)):
            bb = space[b.key]
            print(f"{short(b.key):28s} {tag:5s} "
                  + " ".join(f"{res['alone'][k][bb.slice][0]:11.4g}" for k in keys)
                  + f" {res['joint'][bb.slice][0]:11.4g}")
    print("-" * 96)
    print("under the NEW model these are the GLOBAL values -- the field at the sheet's own\n"
          "centroid -- and each source reached them through its own montage's mixture, so\n"
          "they are the numbers that have to agree if one cortical field explains all three.")
    payload["posterior"] = {
        tag: {"joint": {b.key: float(res["joint"][space[b.key].slice][0])
                        for b in space.blocks},
              "alone": {k: {b.key: float(res["alone"][k][space[b.key].slice][0])
                            for b in space.blocks} for k in keys},
              "prior_median": {b.key: float(t0[space[b.key].slice][0]) for b in space.blocks}}
        for tag, space, res, t0 in (("old", space_old, res_old, t0_old),
                                    ("new", space_new, res_new, t0_new))}

    payload["runtime_s"] = time.time() - t0
    out = args.out or (Path(__file__).resolve().parents[1] / "data" / "joint" /
                       "forge_joint_spatial@v1" / "posterior.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwritten to {out}\n{time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
