#!/usr/bin/env python
"""does the declared neurovascular chain survive contact with measured physiology?

the library's `hrf` model names five processes -- `neurovascular_coupling`,
`vascular_flow`, `bold_formation`, `tissue_exchange`, `metabolism` -- and every
parameter of every implementation of those five arrived in `ibm/processes/` from a
paper.  a resting cbf of 50 mL/100g/min, an oxygen extraction fraction of 0.40, a
flow-metabolism coupling ratio of 2.5, davis's alpha and beta at 0.38 and 1.5.
nothing in this repository has ever asked whether those numbers describe a real
cohort, and the reason is that almost no dataset measures them: a bold experiment
sees one function of flow, volume and extraction and cannot separate the three.

ds004873 is the rare deposit that can.  the same forty people, in the same
scanner, in the same hour, with cbf from pCASL, cbv from a dsc bolus, r2' from
separate multi-echo spin-echo and gradient-echo trains, oef derived from r2' and
cbv, cmro2 from the fick relation, and a task bold series -- all resampled onto
one 2 mm MNI grid.  `hrf` already declares ds004873 a fit source.  this script
takes it up on that.

three experiments, in increasing order of how much of the chain they exercise and
decreasing order of how much the data supports them.  saying that ordering out
loud is the point: the last one is the interesting one and it is also the one
whose caveats are longest.

    (a) the four absolute baselines -- cbf, cbv, oef, cmro2 -- against the four
        literature priors that assert them.  fitted on train subjects, scored on
        held-out subjects.  this is the simplest possible use of §4's product
        form and the one most likely to move a prior a long way, because a prior
        read off a review and a cohort measured on one philips scanner have no
        reason to agree.

    (b) the flow-metabolism coupling ratio n, from the calc-versus-control
        contrast in six cortical networks.  n is the single number the whole
        calibrated-bold literature rests on and `linearized_exchange_lti`
        declares it as 2.5 +/- 0.7.

    (c) the davis forward model: predict the measured task bold percent change
        from the measured flow and metabolism changes, with `davis_alpha` and
        `davis_beta` free.  this is the only one of the three that runs a
        prediction through the chain rather than reading a number off one end of
        it.

the split is BY SUBJECT throughout.  splitting by network inside a subject would
be measuring how well a fitted alpha interpolates across one person's cortex,
which is a number that goes up whether or not anything transferred.  the card
says `split: group_by: [participant, session, stimulus]` and this obeys the first
of those.

**what is deliberately not claimed.**  the registered tyings for these
implementations are PER_PARTITION and PER_SITE; this fit collapses each to one
global grey-matter value, because forty subjects with one grey-matter mean each
cannot identify a per-partition field and pretending otherwise would produce a
smooth map that is entirely the prior.  the collapse is printed with the fit.

**what is circular and what is not.**  cbf comes from pCASL and cbv from a dsc
bolus; those two are independent acquisitions and a relation between them is a
real measurement.  oef is *derived* from r2' and cbv, and cmro2 is *derived* from
cbf and oef, so a relation between cmro2 and cbf is not an identity but is not
independent either -- it is determined by how oef moved, and oef's own inputs are
r2' and cbv.  experiment (c) predicts a gradient-echo epi bold series from
quantities fitted to a different pair of acquisitions, which is not circular in
the acquisition sense and *is* circular in the physics sense: both are sensitive
to deoxyhaemoglobin, which is the thing being tested.  that is stated in the
caveats and it is the honest reading.

run:  ./.venv/bin/python scripts/fit_hemodynamic_chain.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ibm
from ibm.forge.fit import Method, Task, fit, fit_map, provenance_after
from ibm.forge.priors import ParameterSpace, assemble
from ibm.forge.spectra import local_root
from ibm.materialize.library import MODELS
from ibm.registry import REGISTRY
from ibm.vocabulary import Provenance, Tying

MODEL_ID = "hrf"
SOURCE = "ds004873"

#: the six registered parameters this script moves, as
#: `process:implementation.name`.  they are pulled out of the registry rather
#: than restated here, so a change to a declared prior changes what this script
#: tests and cannot silently fail to.
FITTED = {
    "baseline_cbf": ("vascular_flow", "poiseuille_network"),
    "baseline_cbv": ("vascular_flow", "poiseuille_network"),
    "oxygen_extraction_fraction": ("tissue_exchange", "fick_oxygen_limitation"),
    "cmro2_baseline": ("metabolism", "mass_action_budget"),
    "coupling_ratio_n": ("tissue_exchange", "linearized_exchange_lti"),
    "davis_alpha": ("bold_formation", "davis_deoxy_constraint"),
    "davis_beta": ("bold_formation", "davis_deoxy_constraint"),
    "grubb_exponent": ("vascular_flow", "windkessel_lti"),
}

#: mL O2 at STP per micromole.  ds004873 reports cmro2 in umol/100g/min and
#: `mass_action_budget.cmro2_baseline` is declared in mL O2/100g/min; the two are
#: the same quantity in different units and converting is not a modelling choice,
#: but getting it wrong would turn a factor of 44.6 into a "finding".
ML_O2_PER_UMOL = 22.414e-3

NETWORKS = ("DAN", "DMN", "FPN", "SMN", "VAN", "VIS")

#: the deposit's own README tells the reader to restrict every quantitative map
#: to T2 < 90 ms, R2' < 9 /s, CBV < 10 % and OEF < 0.9 before analysis, "to avoid
#: areas mainly influenced by CSF, large vessels, and susceptibility-related
#: artifact areas".  obeying the depositor's stated exclusion is not tuning: it is
#: the difference between measuring grey matter and measuring a draining vein.
LIMITS = {"T2map": (0.0, 90.0), "R2prime": (0.0, 9.0), "cbv": (0.2, 10.0),
          "oef": (0.05, 0.9), "cbf": (1.0, 200.0), "cmro2": (1.0, 1e4)}

TR_S = 1.2
LAG_S = 6.0          # hemodynamic settling dropped from the head of every block


# ---------------------------------------------------------------------------
# the parameter space, from the registry
# ---------------------------------------------------------------------------


def parameter_space(names: tuple[str, ...]) -> ParameterSpace:
    """the declared priors for `names`, collapsed to one global value each.

    `assemble` is given explicit implementations rather than a materialized
    model, which is the door `ibm.forge.priors` opens for exactly this: laying
    out a parameter set before -- or without -- a build.  the tying is replaced
    with GLOBAL and that replacement is a narrowing of the declaration, not a
    correction of it.  forty grey-matter means cannot identify a per-partition
    field, and expanding one here would report a smooth cortical map whose every
    entry is the prior.
    """
    impls = []
    for name in names:
        proc, impl_name = FITTED[name]
        src = REGISTRY.implementations[f"{proc}:{impl_name}"]
        impls.append(replace(src, name=f"{impl_name}", process=proc,
                             params={name: src.params[name]}, tying=Tying.GLOBAL))
    return assemble(implementations=impls)


def as_dict(space: ParameterSpace, theta: np.ndarray) -> dict[str, float]:
    return {b.name: float(theta[b.slice][0]) for b in space.blocks}


# ---------------------------------------------------------------------------
# reading ds004873
# ---------------------------------------------------------------------------


def yeo_masks(derivatives: Path, ref, gm: np.ndarray) -> dict[str, np.ndarray]:
    """the six yeo networks resampled onto the deposit's own 2 mm MNI grid.

    the network files ship at 256^3 freesurfer-conformed 1 mm and every
    quantitative map ships at 91x109x91 2 mm, so the label has to be moved.
    nearest-neighbour through the two affines rather than any interpolation: a
    network label is categorical and a trilinear average of `DMN` and `not DMN`
    is not a fractional membership, it is a number with no referent.
    """
    import nibabel as nib

    idx = np.stack(np.meshgrid(*[np.arange(n) for n in ref.shape], indexing="ij"), -1)
    xyz = nib.affines.apply_affine(ref.affine, idx.reshape(-1, 3))
    out: dict[str, np.ndarray] = {}
    root = derivatives
    for nm in NETWORKS:
        y = nib.load(root / f"Yeo2011_7Networks_MNI152_FreeSurferConformed2mm_MNI_{nm}.nii.gz")
        v = np.rint(nib.affines.apply_affine(np.linalg.inv(y.affine), xyz)).astype(int)
        ok = np.all((v >= 0) & (v < np.array(y.shape)), axis=1)
        lab = np.asarray(y.dataobj, dtype=np.uint8)
        m = np.zeros(v.shape[0], bool)
        m[ok] = lab[v[ok, 0], v[ok, 1], v[ok, 2]] > 0
        out[nm] = m.reshape(ref.shape) & gm
    return out


def read_subject(root: Path, sub: str, gm: np.ndarray, nets: dict[str, np.ndarray],
                 want_bold: bool) -> dict | None:
    """one subject's grey-matter and per-network physiology, and its bold contrast.

    the voxel mask is the intersection of the group grey-matter mask with the
    deposit's own advised validity limits applied to *both* conditions.  applying
    them per condition and then comparing would compare two different sets of
    voxels, which for a ratio is the difference between a physiological change
    and a change of denominator.
    """
    import nibabel as nib

    q = root / "derivatives" / sub / "qmri" / sub

    def load(p: Path) -> np.ndarray | None:
        return np.asarray(nib.load(p).dataobj, dtype=np.float32) if p.is_file() else None

    maps: dict[tuple[str, str], np.ndarray] = {}
    for cond in ("control", "calc"):
        for v in ("cbf", "cbv", "oef", "R2prime"):
            a = load(Path(f"{q}_task-{cond}_space-MNI152_{v}.nii.gz"))
            if a is None:
                return None
            maps[(cond, v)] = a
    # cmro2 in CALC is missing for the ten subjects whose calc cbv is a real
    # measurement, and it is NOT reconstructed here from cbf x oef even though
    # that is the relation the depositor used.  a map computed by this script and
    # a map computed by their pipeline are different objects, and silently
    # substituting one for the other is how a corpus acquires numbers nobody made.
    # those subjects are simply absent from every cmro2 experiment, and present in
    # the ones that do not need it.
    for cond in ("control", "calc"):
        a = load(Path(f"{q}_task-{cond}_space-MNI152_desc-orig_cmro2.nii.gz"))
        if a is not None:
            maps[(cond, "cmro2")] = a
    if ("control", "cmro2") not in maps:
        return None
    t2 = load(Path(f"{q}_space-MNI152_T2map.nii.gz"))
    if t2 is None:
        return None

    ok = gm & np.isfinite(t2) & (t2 > LIMITS["T2map"][0]) & (t2 < LIMITS["T2map"][1])
    for (cond, v), a in maps.items():
        lo, hi = LIMITS[v]
        ok &= np.isfinite(a) & (a > lo) & (a < hi)
    if ok.sum() < 5000:
        return None

    has_calc_cmro2 = ("calc", "cmro2") in maps
    rec: dict = {"subject": sub, "n_voxels": int(ok.sum()), "gm": {}, "net": {},
                 "has_calc_cmro2": has_calc_cmro2}
    for (cond, v), a in maps.items():
        rec["gm"][f"{v}_{cond}"] = float(np.median(a[ok]))
    for nm, nmask in nets.items():
        m = ok & nmask
        if m.sum() < 300:
            continue
        rec["net"][nm] = {f"{v}_{cond}": float(np.median(a[m]))
                          for (cond, v), a in maps.items()}

    # the calc cbv map is a measurement for ten subjects and an extrapolation for
    # the other thirty (the README says cbv was acquired in control and, for some
    # subjects, in calc).  the extrapolated ones are detectable without trusting
    # the README: their calc and control maps agree to r > 0.999 with a median
    # ratio of exactly 1.00000, which no bolus measurement does.
    a, b = maps[("control", "cbv")][ok], maps[("calc", "cbv")][ok]
    rec["cbv_calc_r"] = float(np.corrcoef(a, b)[0, 1])
    rec["cbv_measured_in_calc"] = bool(rec["cbv_calc_r"] < 0.99)

    if want_bold:
        try:
            rec["bold"] = bold_contrast(root, sub, ok, nets)
        except Exception as exc:
            # a truncated .nii.gz is a fact about this store's copy, not about the
            # deposit, and it must be reported rather than silently turning into a
            # smaller cohort.  the subject keeps its qmri and loses only (c).
            rec["bold"] = None
            rec["bold_error"] = f"{type(exc).__name__}: {exc}"
    return rec


def bold_contrast(root: Path, sub: str, ok: np.ndarray,
                  nets: dict[str, np.ndarray]) -> dict[str, float] | None:
    """percent bold change, calc against control, per network.

    the events file names calc, mem and rest and is *silent about control*,
    because the depositor treats control as the baseline.  so control is the
    complement -- the 30 s blocks nothing claims -- and a script that read the
    file naively would treat the baseline as missing data.  saying so here is
    cheaper than discovering it in a number.

    the first `LAG_S` seconds of every block are dropped.  the haemodynamic
    response takes about that long to reach plateau, and a mean over the whole
    block would carry the previous condition's tail into this one's average in a
    way that shrinks every contrast by roughly the same factor -- which looks
    like a smaller effect and is actually a smearing.
    """
    import nibabel as nib

    ev = root / sub / "func" / f"{sub}_task-all_events.tsv"
    bo = (root / "derivatives" / sub / "func" /
          f"{sub}task-all_space-MNI152_res-2_desc-preproc_bold.nii.gz")
    if not (ev.is_file() and bo.is_file()):
        return None
    rows = [ln.rstrip("\n").split("\t") for ln in ev.read_text().splitlines()][1:]
    img = nib.load(bo)
    x = np.asarray(img.dataobj, dtype=np.float32)
    n_t = x.shape[-1]
    t = np.arange(n_t) * TR_S
    lab = np.full(n_t, "control", dtype=object)
    for r in rows:
        if len(r) < 3:
            continue
        on, dur, kind = float(r[0]), float(r[1]), r[2]
        lab[(t >= on) & (t < on + dur)] = kind
    keep = np.ones(n_t, bool)
    for a in np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]]):
        keep[a: a + int(round(LAG_S / TR_S))] = False
    mc, ma = (lab == "control") & keep, (lab == "calc") & keep
    if mc.sum() < 20 or ma.sum() < 20:
        return None
    base = x[..., mc].mean(-1)
    act = x[..., ma].mean(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = 100.0 * (act - base) / np.where(base > 0, base, np.nan)
    out = {"n_control_tr": int(mc.sum()), "n_calc_tr": int(ma.sum())}
    for nm, nmask in nets.items():
        m = ok & nmask
        if m.sum() >= 300:
            out[nm] = float(np.nanmedian(pct[m]))
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def paired(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """paired difference y - x with the three numbers that must travel together.

    an effect size with no p, a p with no effect size, and either with no win
    rate are all ways of reporting less than was measured.  the win rate is the
    one that survives a non-gaussian difference distribution intact, which for
    log predictive densities it usually is.
    """
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = y - x
    if d.size < 2:
        return {"mean": float(d.mean()) if d.size else float("nan"), "sd": float("nan"),
                "sem": float("nan"), "t": float("nan"), "p": float("nan"),
                "dz": float("nan"), "win_rate": float("nan"), "n": int(d.size)}
    t, p = stats.ttest_rel(y, x)
    sd = float(d.std(ddof=1))
    return {"mean": float(d.mean()), "sd": sd, "sem": sd / math.sqrt(d.size),
            "t": float(t), "p": float(p), "dz": float(d.mean() / sd) if sd else float("nan"),
            "win_rate": float((d > 0).mean()), "n": int(d.size)}


def gauss_lpd(y: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return -0.5 * (((y - mu) / sd) ** 2 + math.log(2 * math.pi)) - math.log(sd)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=40)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bold", action="store_true",
                    help="skip experiment (c); it is the slow one, ~1 GB of epi per subject")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import nibabel as nib
    t0 = time.time()
    ibm.load_all(seal=True, strict=True)
    model = MODELS[MODEL_ID]

    rule(f"the model: {MODEL_ID}")
    print(f"{SOURCE} is a declared FIT source of `{MODEL_ID}`: "
          f"{SOURCE in model.fit_sources}")
    print("\nwhat the model says evidence should move:")
    for c in model.constrained:
        print(f"  + {c}")
    print("what it says will stay at its prior:")
    for c in model.prior_dominated:
        print(f"  - {c}")
    print("\nnote the mismatch, stated before any number is computed: the model's own "
          "`constrained`\nlist leads with cerebrovascular reactivity from a breath-hold or "
          "co2 challenge, and\nds004873 has no hypercapnia.  so this script cannot touch the "
          "one quantity `hrf` is\nmost confident about, and what it can touch -- absolute "
          "baselines and the coupling\nratio -- the model does not list at all.  that is a "
          "gap in the declaration as much as in\nthe corpus.")

    # -- geometry --------------------------------------------------------
    rule("data")
    root = local_root(SOURCE)
    print(f"{SOURCE} root {root}  (from data/sources/{SOURCE}/raw/.location.yaml)")
    dv = root / "derivatives"
    ref = nib.load(dv / "MNI152_T1_2mm.nii.gz")
    grp = np.asarray(nib.load(
        dv / "N40_cond-control_space-MNI152_median_GMR2pCBVmasked_cbf.nii.gz").dataobj)
    gm = np.isfinite(grp) & (grp != 0)
    print(f"grey-matter mask: the deposit's own N40 group cbf map, which the README states "
          f"is\nalready GM-masked at p(GM) > 0.5 with venous-dominated voxels discarded -- "
          f"{gm.sum():,} voxels\nof {gm.size:,} on the 2 mm MNI grid.  using the depositor's "
          "mask rather than one of ours\nis what keeps this comparable to their own figures.")
    nets = yeo_masks(dv, ref, gm)
    print("networks (yeo 7, resampled to 2 mm by nearest neighbour, intersected with GM): "
          + ", ".join(f"{k} {v.sum():,}" for k, v in nets.items()))

    subs = sorted(p.name for p in dv.glob("sub-p*") if p.is_dir())[: args.subjects]
    recs: list[dict] = []
    for s in subs:
        r = read_subject(root, s, gm, nets, want_bold=not args.no_bold)
        if r is None:
            print(f"  {s}: skipped (a required map is missing or the mask emptied)")
            continue
        if r.get("bold_error"):
            print(f"  {s}: bold series unreadable ({r['bold_error']}); this subject keeps "
                  "its qmri and is absent from (c)")
        recs.append(r)
    print(f"\n{len(recs)} usable subjects, "
          f"{np.median([r['n_voxels'] for r in recs]):,.0f} valid grey-matter voxels each "
          "(median)")
    n_cbv = sum(r["cbv_measured_in_calc"] for r in recs)
    print(f"of those, {n_cbv} have a cbv map in CALC that is actually a measurement; the "
          f"other\n{len(recs) - n_cbv} have a calc cbv that reproduces their control cbv at "
          "r > 0.999 with a median ratio\nof 1.00000, which is an extrapolation wearing a "
          "measurement's filename.  every cbv-change\nnumber below is restricted to the "
          f"{n_cbv}.")

    # -- split -----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    order = [r["subject"] for r in recs]
    rng.shuffle(order)
    n_tr = int(round(args.train_frac * len(order)))
    train_s, test_s = sorted(order[:n_tr]), sorted(order[n_tr:])
    by = {r["subject"]: r for r in recs}

    rule("split")
    print(f"BY SUBJECT, seed {args.seed}.  no subject contributes a network to both sides.  "
          "the\ncard asks for group_by [participant, session, stimulus] and this obeys the "
          "first;\nsession and stimulus are not split because every subject here has one "
          "session and the\nstimulus contrast is the estimand rather than a unit.")
    print(f"train {len(train_s)}: {' '.join(s[4:] for s in train_s)}")
    print(f"test  {len(test_s)}: {' '.join(s[4:] for s in test_s)}")

    results: dict = {}

    # -- (a) the four absolute baselines ---------------------------------
    rule("(a) four absolute baselines against the four literature priors")
    print("each of these is one registered parameter with a literature prior, and ds004873 "
          "measures\nthe same quantity in the same units on forty people.  the fit is the "
          "smallest possible\nuse of §4: p(theta | D) with one gaussian likelihood factor "
          "per train subject.")

    obs = {
        "baseline_cbf": ("cbf_control", 1.0, "mL/100g/min"),
        "baseline_cbv": ("cbv_control", 0.01, "mL/mL"),         # deposit reports per cent
        "oxygen_extraction_fraction": ("oef_control", 1.0, ""),
        "cmro2_baseline": ("cmro2_control", ML_O2_PER_UMOL, "mL O2/100g/min"),
    }
    space_a = parameter_space(tuple(obs))
    theta0 = space_a.median()
    print("\n" + space_a.describe())

    y_tr = {k: np.array([by[s]["gm"][f] * sc for s in train_s]) for k, (f, sc, _) in obs.items()}
    y_te = {k: np.array([by[s]["gm"][f] * sc for s in test_s]) for k, (f, sc, _) in obs.items()}
    # the predictive sd is the BETWEEN-SUBJECT spread on the train split, estimated
    # once and then held fixed for prior and posterior alike.  refitting it per
    # hypothesis would let the posterior widen its own error bars until it won.
    sd_tr = {k: float(v.std(ddof=1)) for k, v in y_tr.items()}

    def make_task(key: str) -> Task:
        b = space_a[key]
        y, s = y_tr[key], sd_tr[key]

        def logp(theta: np.ndarray, _b=b, _y=y, _s=s) -> float:
            return float(gauss_lpd(_y, float(theta[_b.slice][0]), _s).sum())
        return Task(name=f"{SOURCE}.{key}.train", logp=logp, moves=(b.key,),
                    source=SOURCE, kind="fit",
                    note=f"{len(y)} train subjects, grey-matter median")

    tasks_a = [make_task(k) for k in obs]
    theta_a, rep_a = fit(space_a, tasks_a, method=Method.MAP)
    print("\n" + str(rep_a))

    post_a = as_dict(space_a, theta_a)
    prior_a = as_dict(space_a, theta0)
    prov = provenance_after(space_a, rep_a)

    print(f"\n{'parameter':30s} {'prior':>10s} {'prior sd':>9s} {'posterior':>10s} "
          f"{'test mean':>10s} {'shift':>8s}  provenance now")
    res_a = {}
    for k, (fld, sc, unit) in obs.items():
        b = space_a[k]
        psd = abs(b.prior.scale) if b.prior.dist == "normal" else float("nan")
        lp = gauss_lpd(y_te[k], prior_a[k], sd_tr[k])
        lq = gauss_lpd(y_te[k], post_a[k], sd_tr[k])
        st = paired(lp, lq)
        res_a[k] = {"prior": prior_a[k], "posterior": post_a[k], "unit": unit,
                    "test_mean": float(y_te[k].mean()), "test_sd": float(y_te[k].std(ddof=1)),
                    "predictive_sd": sd_tr[k], "lpd_prior": float(lp.mean()),
                    "lpd_post": float(lq.mean()), "delta": st,
                    "provenance": prov[b.key].value}
        print(f"{k:30s} {prior_a[k]:10.4g} {psd:9.3g} {post_a[k]:10.4g} "
              f"{y_te[k].mean():10.4g} {rep_a.shift[b.key]:+8.2f}  {prov[b.key].value}")
    print("(shift is in prior sd, computed by `ibm.forge.fit` and not by this script)")

    print(f"\nheld-out log predictive density, per test subject, prior against posterior at "
          "the SAME\npredictive sd (the train between-subject spread), so the only thing that "
          "differs is where\nthe centre sits:")
    for k in obs:
        r = res_a[k]
        d = r["delta"]
        print(f"  {k:30s} prior {r['lpd_prior']:+8.3f}  posterior {r['lpd_post']:+8.3f}  "
              f"delta {d['mean']:+7.3f} +/- {d['sem']:.3f}  dz {d['dz']:+6.2f}  "
              f"p {d['p']:9.3g}  wins {d['win_rate']:.0%}")
    won = [k for k in obs if res_a[k]["delta"]["mean"] > 0 and res_a[k]["delta"]["p"] < 0.05]
    print(f"\n{len(won)}/{len(obs)} baselines: the posterior beats the untouched literature "
          f"prior on held-out\nsubjects at p < 0.05 ({', '.join(won) if won else 'none'}).")
    lost = [k for k in obs if k not in won]
    if lost:
        print(f"the other {len(lost)} ({', '.join(lost)}) did NOT: the declared prior is "
              "already as good\nas the fitted value on subjects the fit never saw, which is "
              "a result about the\nliterature being right and not a failure of the fit.")
    results["baselines"] = res_a

    # -- (b) flow-metabolism coupling ------------------------------------
    rule("(b) the flow-metabolism coupling ratio n, from calc against control")
    print("`tissue_exchange:linearized_exchange_lti.coupling_ratio_n` is declared 2.5 +/- 0.7:"
          "\nflow is asserted to rise about two and a half times faster than oxygen "
          "consumption\nduring activation, in fractional terms.  that is the number the "
          "entire calibrated-bold\nliterature rests on.  here it is measured per subject and "
          "per network.")

    def coupling_units(subjects: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        f, m, lab = [], [], []
        for s in subjects:
            if not by[s]["has_calc_cmro2"]:
                continue
            for nm, d in by[s]["net"].items():
                fr = d["cbf_calc"] / d["cbf_control"] - 1.0
                mr = d["cmro2_calc"] / d["cmro2_control"] - 1.0
                f.append(fr); m.append(mr); lab.append(f"{s}/{nm}")
        return np.array(f), np.array(m), lab

    n_cm = sum(by[s]["has_calc_cmro2"] for s in train_s + test_s)
    print(f"\nsubjects with a cmro2 map in CALC: {n_cm} of {len(recs)}.  the other "
          f"{len(recs) - n_cm} are the ones\nwhose calc cbv IS measured, and the deposit "
          "ships no calc cmro2 for them; they are absent\nfrom (b) and (c) and present in "
          "falsification 2, which is the opposite division of the\ncohort and worth "
          "noticing -- the ten subjects who could settle grubb's exponent are\nexactly the "
          "ten who cannot speak to the coupling ratio.")
    f_tr, m_tr, _ = coupling_units(train_s)
    f_te, m_te, lab_te = coupling_units(test_s)
    space_b = parameter_space(("coupling_ratio_n",))
    b_n = space_b["coupling_ratio_n"]
    # n is a slope through the origin: dCBF/CBF = n * dCMRO2/CMRO2.  the residual
    # sd is estimated on train at the prior value and then held fixed, for the
    # same reason as above.
    sd_b = float(np.std(f_tr - 2.5 * m_tr, ddof=1))

    def logp_b(theta: np.ndarray) -> float:
        n = float(theta[b_n.slice][0])
        return float(np.sum(gauss_lpd(f_tr - n * m_tr, 0.0, sd_b)))

    theta_b, rep_b = fit_map(space_b, [Task(name=f"{SOURCE}.coupling.train",
                                            logp=logp_b, moves=(b_n.key,),
                                            source=SOURCE, kind="fit")])
    n_post = float(theta_b[b_n.slice][0])
    n_prior = float(space_b.median()[b_n.slice][0])
    lp = -0.5 * (((f_te - n_prior * m_te) / sd_b) ** 2 + math.log(2 * math.pi)) - math.log(sd_b)
    lq = -0.5 * (((f_te - n_post * m_te) / sd_b) ** 2 + math.log(2 * math.pi)) - math.log(sd_b)
    st_b = paired(lp, lq)
    print(f"\nprior n = {n_prior:.3f} +/- {abs(b_n.prior.scale):.3f}   "
          f"posterior n = {n_post:.3f}   shift {rep_b.shift[b_n.key]:+.2f} prior sd")
    print(f"residual sd of dCBF/CBF about the line, on train: {sd_b:.4f}")
    print(f"held-out ({len(f_te)} subject x network units from {len(test_s)} unseen subjects): "
          f"prior lpd {lp.mean():+.4f},\n  posterior lpd {lq.mean():+.4f}, delta "
          f"{st_b['mean']:+.4f} +/- {st_b['sem']:.4f}, dz {st_b['dz']:+.2f}, "
          f"p {st_b['p']:.3g}, wins {st_b['win_rate']:.0%}")
    print("\nCAVEAT, and it is not small: the held-out units here are network ROIs, six per "
          "subject,\nand they are NOT independent -- they share a subject's haematocrit, "
          "labelling efficiency\nand mask.  the paired p-value above is therefore "
          "anticonservative by an amount this\nscript does not estimate.  the subject-level "
          "split is what stops it being meaningless;\nit is not what would make it exact.")
    results["coupling_ratio_n"] = {"prior": n_prior, "posterior": n_post,
                                   "residual_sd": sd_b, "delta": st_b,
                                   "n_units_test": int(f_te.size)}

    # -- (c) the davis forward model -------------------------------------
    have_bold = sum(bool(by[s].get("bold") and by[s]["has_calc_cmro2"])
                    for s in train_s + test_s) >= 10
    if args.no_bold:
        rule("(c) SKIPPED (--no-bold)")
        results["davis"] = None
    elif not have_bold:
        rule("(c) SKIPPED: not every split subject has a preprocessed MNI bold series")
        results["davis"] = None
    else:
        rule("(c) the davis forward model, run forwards on held-out subjects")
        print("dS/S = M [ 1 - (CBF/CBF0)^(alpha-beta) (CMRO2/CMRO2_0)^beta ], with alpha and "
              "beta the\ntwo registered parameters of "
              "`bold_formation:davis_deoxy_constraint` at their literature\npriors 0.38 +/- "
              "0.06 and 1.50 +/- 0.30.  M is the calibration scale and is profiled in\n"
              "closed form ON THE TRAIN SPLIT ONLY, identically for prior and posterior, so "
              "the\ncomparison is about the exponents and not about a free gain.")

        def davis_units(subjects: list[str]):
            y, f, m, lab = [], [], [], []
            for s in subjects:
                bd = by[s]["bold"]
                if not (bd and by[s]["has_calc_cmro2"]):
                    continue
                for nm, d in by[s]["net"].items():
                    if nm not in bd:
                        continue
                    y.append(bd[nm])
                    f.append(d["cbf_calc"] / d["cbf_control"])
                    m.append(d["cmro2_calc"] / d["cmro2_control"])
                    lab.append(f"{s}/{nm}")
            return (np.array(y), np.array(f), np.array(m), lab)

        y_tr_d, f_tr_d, m_tr_d, _ = davis_units(train_s)
        y_te_d, f_te_d, m_te_d, lab_d = davis_units(test_s)

        def g_of(a: float, b: float, f: np.ndarray, m: np.ndarray) -> np.ndarray:
            return 1.0 - np.power(f, a - b) * np.power(m, b)

        def profile_M(a: float, b: float) -> float:
            g = g_of(a, b, f_tr_d, m_tr_d)
            den = float(np.sum(g * g))
            return float(np.sum(y_tr_d * g) / den) if den > 1e-12 else 0.0

        space_c = parameter_space(("davis_alpha", "davis_beta"))
        ba, bb = space_c["davis_alpha"], space_c["davis_beta"]
        sd_c = float(np.std(y_tr_d - profile_M(0.38, 1.5)
                            * g_of(0.38, 1.5, f_tr_d, m_tr_d), ddof=1))

        def logp_c(theta: np.ndarray) -> float:
            a, b = float(theta[ba.slice][0]), float(theta[bb.slice][0])
            if not (np.isfinite(a) and np.isfinite(b)):
                return -1e12
            with np.errstate(over="ignore", invalid="ignore"):
                pred = profile_M(a, b) * g_of(a, b, f_tr_d, m_tr_d)
            v = float(np.sum(gauss_lpd(y_tr_d - pred, 0.0, sd_c)))
            return v if np.isfinite(v) else -1e12

        theta_c, rep_c = fit_map(space_c, [Task(name=f"{SOURCE}.davis.train", logp=logp_c,
                                                moves=(ba.key, bb.key), source=SOURCE,
                                                kind="fit")])
        a_hat, b_hat = float(theta_c[ba.slice][0]), float(theta_c[bb.slice][0])
        a_pri, b_pri = 0.38, 1.5
        M_pri, M_post = profile_M(a_pri, b_pri), profile_M(a_hat, b_hat)
        print(f"\nprior     alpha {a_pri:.3f}  beta {b_pri:.3f}  M {M_pri:+.4f} (profiled on "
              f"train)")
        print(f"posterior alpha {a_hat:.3f}  beta {b_hat:.3f}  M {M_post:+.4f}   "
              f"shift {rep_c.shift[ba.key]:+.2f} / {rep_c.shift[bb.key]:+.2f} prior sd")
        print(f"residual sd of %BOLD about the model on train: {sd_c:.4f} %")

        pred_p = M_pri * g_of(a_pri, b_pri, f_te_d, m_te_d)
        pred_q = M_post * g_of(a_hat, b_hat, f_te_d, m_te_d)
        lp = gauss_lpd(y_te_d - pred_p, 0.0, sd_c)
        lq = gauss_lpd(y_te_d - pred_q, 0.0, sd_c)
        st_c = paired(lp, lq)
        from scipy import stats
        r_p = stats.pearsonr(pred_p, y_te_d)
        r_q = stats.pearsonr(pred_q, y_te_d)
        print(f"\nheld-out ({len(y_te_d)} subject x network units, {len(test_s)} unseen "
              f"subjects):")
        print(f"  lpd  prior {lp.mean():+.4f}   posterior {lq.mean():+.4f}   "
              f"delta {st_c['mean']:+.4f} +/- {st_c['sem']:.4f}, dz {st_c['dz']:+.2f}, "
              f"p {st_c['p']:.3g}, wins {st_c['win_rate']:.0%}")
        print(f"  correlation between predicted and measured %BOLD: "
              f"prior r {r_p.statistic:+.3f} (p {r_p.pvalue:.3g}), "
              f"posterior r {r_q.statistic:+.3f} (p {r_q.pvalue:.3g})")
        results["davis"] = {"alpha_prior": a_pri, "beta_prior": b_pri,
                            "alpha_post": a_hat, "beta_post": b_hat,
                            "M_prior": M_pri, "M_post": M_post, "residual_sd": sd_c,
                            "delta": st_c, "r_prior": float(r_p.statistic),
                            "p_r_prior": float(r_p.pvalue),
                            "r_post": float(r_q.statistic), "p_r_post": float(r_q.pvalue),
                            "n_units_test": int(y_te_d.size)}

        # -- falsification 1 -------------------------------------------
        rule("FALSIFICATION 1: does the chain predict bold at all, on subjects it never saw?")
        print("the declaration says a gradient-echo bold change IS a function of the flow and "
              "metabolism\nchanges, through `bold_formation`.  so the davis prediction built "
              "from ds004873's own\nmeasured cbf and cmro2 must correlate POSITIVELY with the "
              "measured %BOLD on held-out\nsubjects.  if it does not, the chain as declared "
              "does not describe this cohort, and no\nrefit of alpha and beta repairs that -- "
              "the failure would be of the functional form.")
        # negative control: break the pairing between physiology and bold, keeping
        # both marginals.  a pipeline that "predicts" a shuffled target is
        # predicting the network's mean and nothing else.
        rs = np.random.default_rng(args.seed + 1)
        null = []
        for _ in range(2000):
            null.append(stats.pearsonr(pred_q, rs.permutation(y_te_d)).statistic)
        null = np.array(null)
        p_perm = float((np.abs(null) >= abs(r_q.statistic)).mean())
        print(f"\n  measured r = {r_q.statistic:+.3f} over {len(y_te_d)} held-out units")
        print(f"  permutation null (2000 shuffles of the bold values against the "
              f"physiology):\n    mean {null.mean():+.4f}, sd {null.std():.4f}, "
              f"two-sided p = {p_perm:.4g}")
        f1 = bool(r_q.statistic > 0 and p_perm < 0.01)
        print(f"\n  FALSIFICATION 1: {'PASSED' if f1 else 'FAILED'} -- the declared chain "
              f"{'does' if f1 else 'does NOT'} predict held-out bold\n  from held-out "
              "physiology.")
        if not f1:
            print("  that is the result, and it is the more interesting one.  ds004873's own "
                  "headline is\n  that bold changes can OPPOSE oxygen metabolism across "
                  "cortex; a single-signed davis\n  form cannot express that, and a null or "
                  "negative correlation here is what that looks\n  like from inside this "
                  "ontology.  the fix is a different `bold_formation` declaration,\n  not a "
                  "different fitting run.")
        results["falsification_chain_predicts_bold"] = {
            "r": float(r_q.statistic), "p_permutation": p_perm, "passed": f1,
            "null_mean": float(null.mean()), "null_sd": float(null.std())}

    # -- falsification 2: grubb ------------------------------------------
    rule("FALSIFICATION 2: grubb's law, on the ten subjects whose calc cbv is measured")
    print("`vascular_flow:windkessel_lti.grubb_exponent` is declared 0.38 +/- 0.06 and the "
          "declaration\nspells out what it means: cbv = cbf^alpha.  cbf here is pCASL and cbv "
          "is a dsc bolus --\ntwo different sequences, so a relation between them is a "
          "measurement and not an\nidentity.  the prediction that could fail: alpha > 0, i.e. "
          "blood volume rises when flow\nrises.")
    ten = [s for s in (train_s + test_s) if by[s]["cbv_measured_in_calc"]]
    ten_tr = [s for s in train_s if by[s]["cbv_measured_in_calc"]]
    ten_te = [s for s in test_s if by[s]["cbv_measured_in_calc"]]
    print(f"\n  subjects with a measured calc cbv: {len(ten)} "
          f"({len(ten_tr)} train, {len(ten_te)} test) -- {', '.join(s[4:] for s in ten)}")
    if len(ten_tr) < 3 or len(ten_te) < 2:
        print("  NOT ENOUGH to split by subject.  reported unsplit and therefore NOT as a "
              "held-out\n  result; a slope fitted and scored on the same subjects is a "
              "description, not a test.")
    lf, lv, who = [], [], []
    for s in ten:
        for nm, d in by[s]["net"].items():
            lf.append(math.log(d["cbf_calc"] / d["cbf_control"]))
            lv.append(math.log(d["cbv_calc"] / d["cbv_control"]))
            who.append(s)
    lf, lv = np.array(lf), np.array(lv)
    if lf.size >= 6:
        from scipy import stats
        sl = stats.linregress(lf, lv)
        print(f"\n  regression of log(cbv ratio) on log(cbf ratio) over {lf.size} "
              f"subject x network units:")
        print(f"    slope (= grubb alpha) {sl.slope:+.4f} +/- {sl.stderr:.4f}, "
              f"r {sl.rvalue:+.3f}, p {sl.pvalue:.3g}")
        print(f"    declared prior 0.38 +/- 0.06; the fitted slope is "
              f"{(sl.slope - 0.38) / 0.06:+.1f} prior sd from it")
        f2 = bool(sl.slope > 0 and sl.pvalue < 0.05)
        print(f"\n  FALSIFICATION 2: {'PASSED' if f2 else 'FAILED'} -- blood volume "
              f"{'does' if f2 else 'does NOT'} rise with flow in this\n  cohort at "
              "p < 0.05.")
        if not f2:
            print("  read this as underpowered rather than as a refutation.  ten subjects, a "
                  "task whose\n  whole-cortex flow change is a few per cent, and a dsc cbv "
                  "whose absolute scale rests\n  on an arterial input function this deposit "
                  "does not record: the experiment that would\n  settle grubb's exponent is a "
                  "hypercapnia challenge, which is precisely what `hrf`'s\n  own "
                  "`constrained` list asks for and this source does not have.")
        results["falsification_grubb"] = {
            "n_subjects": len(ten), "n_units": int(lf.size), "slope": float(sl.slope),
            "stderr": float(sl.stderr), "r": float(sl.rvalue), "p": float(sl.pvalue),
            "passed": f2}
    else:
        results["falsification_grubb"] = {"n_subjects": len(ten), "skipped": True}

    # -- falsification 3: the oef prior ----------------------------------
    rule("FALSIFICATION 3: the oxygen extraction fraction prior, as an interval claim")
    print("`fick_oxygen_limitation.oxygen_extraction_fraction` is declared normal(0.40, 0.06)."
          "\nthat is not a soft preference; it is a claim that a healthy adult's grey-matter "
          "oef lies\nin 0.28-0.52 with 95% probability.  forty measured subjects either sit "
          "in that interval\nor they do not.")
    oef = np.array([by[s]["gm"]["oef_control"] for s in train_s + test_s])
    inside = float(((oef > 0.40 - 2 * 0.06) & (oef < 0.40 + 2 * 0.06)).mean())
    print(f"\n  measured grey-matter oef: mean {oef.mean():.4f}, sd {oef.std(ddof=1):.4f}, "
          f"range {oef.min():.3f}-{oef.max():.3f}")
    print(f"  fraction inside the prior's central 95% interval [0.28, 0.52]: {inside:.0%} "
          f"of {oef.size}")
    f3 = inside >= 0.9
    print(f"\n  FALSIFICATION 3: {'PASSED' if f3 else 'FAILED'} -- the declared oef prior "
          f"{'contains' if f3 else 'does NOT contain'} this cohort.")
    print("  the deposit's own README warns that oef above 0.9 is not interpretable and that "
          "these\n  maps must be grey-matter masked before use; both were done, so a failure "
          "here would be\n  about the prior and not about the mask.")
    results["falsification_oef_interval"] = {"mean": float(oef.mean()),
                                             "sd": float(oef.std(ddof=1)),
                                             "fraction_inside_95": inside, "passed": f3}

    # -- caveats ---------------------------------------------------------
    rule("where this is weaker than it looks")
    print("""
DERIVATION, NOT MEASUREMENT.  cbf comes from pCASL and cbv from a dsc bolus and
those two are independent.  oef does not: it is computed from r2' and cbv, and
cmro2 is computed from cbf and oef.  so experiment (b)'s coupling ratio is not an
identity -- it is set by how oef moved, and oef's inputs are r2' and cbv -- but it
is not two independent measurements either, and the card says so in its own words:
"CMRO2 is a model output, and different calibration models give different CMRO2
from identical data".  the number above inherits their calibration model.

SHARED PHYSICS IN (c).  the davis test predicts a gradient-echo epi series from
quantities fitted to a *different* pair of acquisitions, which is not circular in
the acquisition sense.  it is circular in the physics sense: both the epi signal
and r2' are sensitive to deoxyhaemoglobin, which is the quantity the chain is
about.  a correlation between them is evidence that the two acquisitions agree,
and only partly evidence that the davis exponents are right.

THE ABSOLUTE SCALES CARRY AN UNRECORDED CALIBRATION.  the pCASL sidecar records
no labelling duration and no post-labelling delay, and the dsc series carries no
arterial input function.  every ratio within a subject divides those out; the four
baselines in (a) do not, and they are the four numbers most likely to be shifted
by a scanner convention rather than by physiology.  a fitted `baseline_cbf` that
disagrees with the literature by 30% is at least as likely to be a philips
kinetic-model default as a fact about these people.

NETWORK UNITS ARE NOT INDEPENDENT.  six yeo networks per subject share a
haematocrit, a labelling efficiency, a mask and a head.  the subject-level split
is what makes the held-out numbers meaningful; it does not make the within-subject
paired statistics exact, and every p-value computed over subject x network units
above is anticonservative.

THE MODEL'S OWN BEST QUANTITY IS UNTOUCHED.  `hrf` lists cerebrovascular
reactivity first under `constrained`, and says plainly why: "a co2 challenge
drives the vasculature without driving neurons, which is the only way in this whole
corpus to separate `neurovascular_coupling` from `vascular_flow`".  ds004873 has no
hypercapnia.  so the two processes remain degenerate here and the hrf is, in this
script's hands, one lumped kernel -- exactly as the declaration warned.""")

    out = args.out or (Path(__file__).resolve().parents[1] / "data" / "sources" / SOURCE /
                       "evidence" / "fit_hemodynamic_chain@v1" / "posterior.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": MODEL_ID, "source": SOURCE, "seed": args.seed,
        "train_subjects": train_s, "test_subjects": test_s,
        "results": results, "runtime_s": time.time() - t0}, indent=2, default=float))

    rule("summary")
    print(f"(a) baselines : {len(won)}/{len(obs)} beat the literature prior on held-out "
          f"subjects (p < 0.05)")
    print(f"(b) coupling n: {n_prior:.2f} -> {n_post:.2f}, held-out delta lpd "
          f"{st_b['mean']:+.4f}, p {st_b['p']:.3g}, wins {st_b['win_rate']:.0%}")
    if results.get("davis"):
        d = results["davis"]
        print(f"(c) davis     : alpha {d['alpha_prior']:.2f} -> {d['alpha_post']:.2f}, "
              f"beta {d['beta_prior']:.2f} -> {d['beta_post']:.2f}, held-out r "
              f"{d['r_prior']:+.3f} -> {d['r_post']:+.3f}")
    for k in ("falsification_chain_predicts_bold", "falsification_grubb",
              "falsification_oef_interval"):
        v = results.get(k)
        if v and "passed" in v:
            print(f"{k:38s} {'PASS' if v['passed'] else 'FAIL'}")
    print(f"\nwritten to {out}\n{time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
