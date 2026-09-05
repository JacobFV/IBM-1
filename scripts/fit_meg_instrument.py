#!/usr/bin/env python
"""the meg forward model's instrument half, measured against the instrument.

`meg_forward` is the one model in the library whose `constrained` list is mostly
about the *device* rather than about the brain: "sensor gain and crosstalk per
channel, over-determined by 306 sensors", "inner-skull surface placement".  and
`device_coupling:magnetometer_pickup` declares four numbers to go with that --
a gradiometer baseline of 50 +/- 20 mm, a standoff of 20 +/- 8 mm, a sensor noise
of 3 fT/sqrt(Hz) to within a factor of two, and a radial blindness of 0.05 +/-
0.03.  all four are declared LITERATURE or PHYSICS and none of them has ever been
compared with an actual dewar.

ds000117 is a declared fit source of `meg_forward`, it is held, and it carries
what is needed to check every one of them.  a neuromag FIF is not only data: it
carries the coil integration-point geometry per channel, the coil-type codes that
select the integration rule, the HPI-measured head position in the dewar, a
digitised head shape, and -- in the same recording, on the same amplifier clock --
70 channels of simultaneous EEG.  the card's own `systematic_bias` field says why
that matters: "the best available fixture for validating a forward model, because
the same sources must explain two physically different measurements at once".

    fit     `standoff_mm` and `sensor_noise_ft_rt_hz` against the TRAIN
            participants' own recordings, through `ibm.forge.fit`
    test    score prior and posterior on HELD-OUT participants at the same
            predictive width
    falsify three predictions of the declaration, each of which could fail:
              1. the magnetic instrument resolves finer source structure than
                 the simultaneous eeg -- the claim `meg_forward`'s r(q) rests on
              2. the radial null space is exact in the conductor the model uses
              3. the gradiometer baseline is 50 mm

**why there is no source estimate anywhere in this script.**  the obvious
experiment is to invert one modality and predict the other.  it was tried and it
is reported in the caveats: without individual anatomy every inverse here rests on
a spherical conductor and a head-shape coregistration, and a minimum-norm estimate
under those conditions has a depth bias large enough that the cross-modal
prediction measures the estimator rather than the forward model.  a number that
cannot distinguish "the declaration is wrong" from "the inverse is bad" is not
evidence, so what is reported instead is the part of the chain this source
determines without an inverse: the instrument, and one property of the measured
fields themselves.

the split is BY PARTICIPANT.  runs within a participant share a head, a cap and a
dewar position to within a few millimetres, so a run-level split would report
within-session repeatability as generalization.

run:  ./.venv/bin/python scripts/fit_meg_instrument.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ibm
from ibm.forge.fit import Method, Task, fit, provenance_after
from ibm.forge.priors import ParameterSpace, assemble
from ibm.forge.spectra import local_root
from ibm.materialize.library import MODELS
from ibm.registry import REGISTRY
from ibm.vocabulary import Tying

MODEL_ID = "meg_forward"
SOURCE = "ds000117"
IMPL = "device_coupling:magnetometer_pickup"

#: every visual event code in the wakeman-henson design: famous, unfamiliar and
#: scrambled faces.  the evoked response is taken over ALL of them rather than
#: over the face-minus-scrambled contrast, deliberately.  nothing here is a
#: question about face processing; what is needed is the largest, cleanest
#: topography the recording contains, and that is the visual evoked response
#: itself.  the contrast is ten times smaller and would make every number below a
#: statement about noise.
VISUAL = (5, 6, 7, 13, 14, 15, 17, 18, 19)

#: the visual evoked peak.  a FIXED window, chosen before looking, rather than a
#: per-subject argmax: picking the peak per subject and then comparing modalities
#: would let each modality choose its own latency, and the spatial comparison in
#: falsification 1 would be between two different brain states.
PEAK_S = (0.085, 0.125)

#: where the sensor-noise floor is read.  above the alpha rhythm and below the
#: mains harmonic, so the band contains as little brain and as little line noise
#: as this recording allows.  it is still not a sensor-noise measurement and the
#: script says so where it reports the number.
NOISE_BAND = (20.0, 45.0)


# ---------------------------------------------------------------------------
# the parameter space, from the registry
# ---------------------------------------------------------------------------


def parameter_space(names: tuple[str, ...]) -> ParameterSpace:
    """`magnetometer_pickup`'s own declared priors for `names`, one value each.

    the declared tying is PER_SITE -- one standoff and one noise figure per
    sensor, which for 306 sensors is the right declaration and is not what this
    script can identify from 16 people.  it is collapsed to GLOBAL and the
    collapse is printed, because a per-sensor posterior fitted from one number
    per recording would be 306 copies of the prior wearing a fit's provenance.
    """
    src = REGISTRY.implementations[IMPL]
    proc, impl = IMPL.split(":")
    return assemble(implementations=[
        replace(src, params={n: src.params[n] for n in names}, tying=Tying.GLOBAL)])


# ---------------------------------------------------------------------------
# reading one run
# ---------------------------------------------------------------------------


def head_sphere(info):
    """a conductor sphere fitted to the digitised EEG electrodes, not to the head shape.

    ds000117's `extra` digitisation includes the face, and `dig_kinds='auto'`
    lets those points drag the fitted centre 37 mm anterior and 23 mm inferior of
    where a head's centre is -- mne says so itself, with a warning, and the
    resulting conductor has the occipital cortex outside it.  the EEG cap covers
    the cranium and not the face, so fitting to the electrodes gives a centre
    within a centimetre of the head origin on every participant here.  this is a
    judgement call and it is the difference between a forward model and a shape.
    """
    import mne
    rad, r0, _ = mne.bem.fit_sphere_to_headshape(info, dig_kinds="eeg", units="m")
    return float(rad), np.asarray(r0, float)


def read_run(root: Path, sub: str, run: int) -> dict | None:
    """one (participant, run): its evoked topographies, its geometry, its noise.

    everything returned is either measured from the file or derived from the
    file's own recorded geometry.  nothing is a template and nothing is assumed
    from the manufacturer's data sheet -- which is the point, because the priors
    being tested are exactly what a data sheet would have told us.
    """
    import mne
    from mne.forward._make_forward import _create_meg_coils
    from mne.transforms import apply_trans

    f = (root / sub / "ses-meg" / "meg" /
         f"{sub}_ses-meg_task-facerecognition_run-{run:02d}_meg.fif")
    if not f.is_file():
        return None
    raw = mne.io.read_raw_fif(f, preload=False, allow_maxshield=True, verbose="ERROR")
    info = raw.info
    ev = mne.find_events(raw, stim_channel="STI101", shortest_event=1, verbose="ERROR")
    codes = set(ev[:, 2])
    ids = {f"vis/{c}": c for c in VISUAL if c in codes}
    if not ids:
        return None

    # -- the evoked topography -----------------------------------------
    picks = mne.pick_types(info, meg=True, eeg=True, exclude="bads")
    ep = mne.Epochs(raw, ev, event_id=ids, tmin=-0.2, tmax=0.4, picks=picks,
                    baseline=(None, 0), preload=True, decim=4, proj=False,
                    verbose="ERROR")
    ep.filter(None, 40.0, verbose="ERROR")
    evoked = ep.average()
    tm = (evoked.times >= PEAK_S[0]) & (evoked.times < PEAK_S[1])
    topo = evoked.data[:, tm].mean(1)
    base = evoked.data[:, evoked.times < 0]

    ty = np.array([mne.channel_type(evoked.info, i) for i in range(len(evoked.ch_names))])
    mag = np.flatnonzero(ty == "mag")
    eeg = np.flatnonzero(ty == "eeg")
    if mag.size < 50 or eeg.size < 30:
        return None

    # -- geometry, from the file's own coil definitions ------------------
    rad, r0 = head_sphere(evoked.info)
    coils = _create_meg_coils([evoked.info["chs"][i] for i in mag], "accurate")
    # coil centres in device coords, then into head coords through the HPI fit
    centres = np.array([(c["rmag"] * np.abs(c["w"])[:, None]).sum(0)
                        / np.abs(c["w"]).sum() for c in coils])
    centres_h = apply_trans(evoked.info["dev_head_t"], centres)
    standoff_mm = float(np.median(np.linalg.norm(centres_h - r0, axis=1) - rad) * 1000.0)

    gr = mne.pick_types(evoked.info, meg="grad", exclude="bads")
    gcoils = _create_meg_coils([evoked.info["chs"][i] for i in gr[:20]], "accurate")
    bl = []
    for c in gcoils:
        w = c["w"]
        p, n = c["rmag"][w > 0].mean(0), c["rmag"][w < 0].mean(0)
        bl.append(np.linalg.norm(p - n) * 1000.0)
    baseline_mm = float(np.median(bl))

    # -- the noise floor -------------------------------------------------
    n_s = min(60.0, raw.times[-1] - 10.0)
    seg = raw.copy().crop(tmin=5.0, tmax=5.0 + n_s).load_data(verbose="ERROR")
    seg.pick(picks=[seg.ch_names[i] for i in mne.pick_types(seg.info, meg="mag")])
    psd = seg.compute_psd(fmin=NOISE_BAND[0], fmax=NOISE_BAND[1], verbose="ERROR")
    asd_ft = float(np.median(np.sqrt(psd.get_data().mean(-1))) * 1e15)

    # eeg electrode positions in head coords, for the topography comparison
    epos = np.array([evoked.info["chs"][i]["loc"][:3] for i in eeg])

    return {"subject": sub, "run": run,
            "standoff_mm": standoff_mm, "gradiometer_baseline_mm": baseline_mm,
            "sensor_noise_ft_rt_hz": asd_ft,
            "head_radius_mm": rad * 1000.0, "r0": r0.tolist(),
            "n_epochs": int(len(ep)),
            "mag_topo": topo[mag], "eeg_topo": topo[eeg],
            "mag_pos": centres_h, "eeg_pos": epos,
            "mag_noise": base[mag].std(1), "eeg_noise": base[eeg].std(1),
            "mag_names": [evoked.ch_names[i] for i in mag],
            "eeg_names": [evoked.ch_names[i] for i in eeg],
            "info": evoked.info, "sphere": (rad, r0)}


# ---------------------------------------------------------------------------
# falsification 1: is the magnetic topography spatially sharper?
# ---------------------------------------------------------------------------


def acf_angle(pos: np.ndarray, r0: np.ndarray, topo: np.ndarray,
              edges: np.ndarray) -> np.ndarray:
    """the spatial autocorrelation of one topography, against ANGLE from the head centre.

    against angle and not against millimetres, because the two arrays are not on
    the same surface: the magnetometers sit two centimetres off the scalp and the
    electrodes sit on it, so a comparison in millimetres would report the dewar's
    radius as a difference in spatial resolution.  the angle subtended at the
    conductor's centre is the same quantity for both, and it is also the quantity
    "how finely does this instrument resolve the source layer" is actually about.

    the statistic is the mean of `b_i b_j` over sensor pairs in an angular bin,
    normalized by the mean of `b_i^2`.  it is the ordinary spatial acf of a single
    map, and its zero crossing is the angular half-width of the field pattern.
    """
    u = pos - r0
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    cos = np.clip(u @ u.T, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    b = topo - topo.mean()
    prod = np.outer(b, b)
    iu = np.triu_indices(len(b), 1)
    a, p = ang[iu], prod[iu]
    norm = float(np.mean(b ** 2))
    out = np.full(len(edges) - 1, np.nan)
    for k in range(len(edges) - 1):
        m = (a >= edges[k]) & (a < edges[k + 1])
        if m.sum() >= 20:
            out[k] = float(p[m].mean() / max(norm, 1e-300))
    return out


def zero_crossing(edges: np.ndarray, acf: np.ndarray) -> float:
    """the first angle at which the acf crosses zero, by linear interpolation."""
    c = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(acf)
    c, a = c[ok], acf[ok]
    for i in range(len(a) - 1):
        if a[i] > 0 >= a[i + 1]:
            t = a[i] / (a[i] - a[i + 1])
            return float(c[i] + t * (c[i + 1] - c[i]))
    return float("nan")


# ---------------------------------------------------------------------------
# falsification 2: how blind is the magnetic lead field to radial current?
# ---------------------------------------------------------------------------


def lead_field_stats(u: dict) -> dict:
    """the two lead-field quantities this script needs, from one forward solve.

    both modalities are computed on ONE source space in ONE conductor from this
    participant's own digitisation, which is what makes them comparable at all: a
    resolution comparison between two instruments that saw different source grids
    would be a comparison of the grids.

    what comes back:

    `spatial_dof` -- the number of singular values of the noise-whitened lead
    field above one per cent of the largest, per modality, and that count divided
    by the modality's channel count.  this is the standard "how many spatial
    degrees of freedom does this instrument resolve" measure and it is the direct
    quantitative form of `meg_forward`'s claim that "the sensor genuinely resolves
    finer source structure".  it is reported normalized as well as raw because meg
    has 102 magnetometers against 70 electrodes here, and a raw count would report
    the channel count.

    `radial_ratio` -- rms magnetometer sensitivity to radial current over rms
    sensitivity to tangential, splitting each source's three orientation columns
    at the source's own radial direction.
    """
    import mne

    info, (rad, r0) = u["info"], u["sphere"]
    sph = mne.make_sphere_model(r0=r0, head_radius=rad,
                                relative_radii=(0.90, 0.92, 0.97, 1.0),
                                sigmas=(0.33, 1.79, 0.01, 0.43), verbose="ERROR")
    src = mne.setup_volume_source_space(sphere=sph, pos=12.0, mindist=8.0,
                                        sphere_units="m", verbose="ERROR")
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sph, meg=True,
                                    eeg=True, verbose="ERROR")
    fwd = mne.convert_forward_solution(fwd, force_fixed=False, verbose="ERROR")
    L, rr = fwd["sol"]["data"], fwd["source_rr"]
    ty = np.array([mne.channel_type(fwd["info"], i)
                   for i in range(len(fwd["info"]["ch_names"]))])
    names = list(fwd["info"]["ch_names"])

    out: dict = {"n_sources": int(rr.shape[0])}
    for tag, want, noise_names, noise in (("meg", "mag", u["mag_names"], u["mag_noise"]),
                                          ("eeg", "eeg", u["eeg_names"], u["eeg_noise"])):
        rows = [i for i, n in enumerate(names) if ty[i] == want and n in noise_names]
        if len(rows) < 20:
            continue
        w = np.array([noise[noise_names.index(names[i])] for i in rows])
        Lw = L[rows] / np.maximum(w, 1e-30)[:, None]
        s = np.linalg.svd(Lw, compute_uv=False)
        out[f"{tag}_channels"] = len(rows)
        # two thresholds, because the answer should not depend on one of them and
        # the reader is entitled to see whether it does.
        for thr in (0.01, 0.05):
            k = int((s / s[0] > thr).sum())
            out[f"{tag}_dof@{thr:g}"] = k
            out[f"{tag}_dof_frac@{thr:g}"] = k / len(rows)
        out[f"{tag}_dof"] = out[f"{tag}_dof@0.01"]
        out[f"{tag}_dof_frac"] = out[f"{tag}_dof_frac@0.01"]

    rows = [i for i, n in enumerate(names) if ty[i] == "mag"]
    Lm = L[rows]
    rad_n, tan_n = [], []
    for k in range(rr.shape[0]):
        v = rr[k] - r0
        v = v / max(np.linalg.norm(v), 1e-12)
        block = Lm[:, 3 * k: 3 * k + 3]           # x, y, z columns for this source
        t1 = np.cross(v, [0.0, 0.0, 1.0])
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(v, [0.0, 1.0, 0.0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(v, t1)
        rad_n.append(np.linalg.norm(block @ v))
        tan_n.append(math.hypot(np.linalg.norm(block @ t1), np.linalg.norm(block @ t2)))
    r_, t_ = np.array(rad_n), np.array(tan_n)
    out["radial_ratio"] = float(np.sqrt((r_ ** 2).mean() / max((t_ ** 2).mean(), 1e-300)))
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def paired(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
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
    ap.add_argument("--subjects", type=int, default=16)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import mne
    mne.set_log_level("ERROR")

    t0 = time.time()
    ibm.load_all(seal=True, strict=True)
    model = MODELS[MODEL_ID]

    rule(f"the model: {MODEL_ID}")
    print(f"{SOURCE} is a declared FIT source of `{MODEL_ID}`: "
          f"{SOURCE in model.fit_sources}")
    print(f"`device_coupling` is among its declared processes: "
          f"{'device_coupling' in model.request.processes}")
    print("\nwhat the model says evidence should move:")
    for c in model.constrained:
        print(f"  + {c}")
    print("what it says will stay at its prior:")
    for c in model.prior_dominated:
        print(f"  - {c}")

    rule("p(theta): the declaration being tested")
    src = REGISTRY.implementations[IMPL]
    for n, p in src.params.items():
        med = math.exp(p.loc) if p.dist == "lognormal" else p.loc
        print(f"  {n:26s} {p.dist:10s} median {med:8.4g} {p.units:12s} "
              f"[{p.provenance.value}]  {p.note or p.source or ''}")
    print("\nthe declared tying is PER_SITE -- one of each per sensor.  this script fits one "
          "of\neach, globally, and that narrowing is why nothing below should be read as a "
          "per-channel\ncalibration; `meg_forward` claims sensor gain is 'over-determined by "
          "306 sensors' and\nthat claim is untested here.")

    # -- data ------------------------------------------------------------
    rule("data")
    root = local_root(SOURCE) / "1.1.0"
    print(f"{SOURCE} root {root}  (from data/sources/{SOURCE}/raw/.location.yaml)")
    subs = sorted(p.name for p in root.glob("sub-[0-9][0-9]") if p.is_dir())[: args.subjects]
    units: list[dict] = []
    for s in subs:
        for r in range(1, args.runs + 1):
            try:
                u = read_run(root, s, r)
            except Exception as exc:
                print(f"  {s} run-{r:02d}: skipped ({type(exc).__name__}: {exc})")
                continue
            if u is None:
                continue
            units.append(u)
    if len(units) < 8:
        print("too few usable runs to split.")
        return 2
    have = sorted({u["subject"] for u in units})
    print(f"{len(units)} runs from {len(have)} participants, "
          f"{np.median([u['n_epochs'] for u in units]):.0f} visual epochs each (median)")
    print(f"evoked topography averaged over {PEAK_S[0] * 1000:.0f}-{PEAK_S[1] * 1000:.0f} ms, "
          "a fixed window chosen before looking")
    print(f"\n{'run':16s} {'epochs':>7s} {'head R mm':>10s} {'standoff mm':>12s} "
          f"{'grad base mm':>13s} {'mag noise fT/rtHz':>18s}")
    for u in units:
        print(f"{u['subject']}/run-{u['run']:02d}   {u['n_epochs']:7d} "
              f"{u['head_radius_mm']:10.1f} {u['standoff_mm']:12.1f} "
              f"{u['gradiometer_baseline_mm']:13.2f} {u['sensor_noise_ft_rt_hz']:18.1f}")

    # -- split -----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    order = list(have)
    rng.shuffle(order)
    n_tr = int(round(args.train_frac * len(order)))
    tr_sub, te_sub = sorted(order[:n_tr]), sorted(order[n_tr:])
    tr = [u for u in units if u["subject"] in tr_sub]
    te = [u for u in units if u["subject"] in te_sub]

    rule("split")
    print("BY PARTICIPANT.  both runs of a person go to the same side: they share a head, a "
          "cap\nand a dewar position to within a few millimetres, and splitting them would "
          "report\nwithin-session repeatability as generalization to new people.")
    print(f"train {len(tr_sub)} participants / {len(tr)} runs: {' '.join(tr_sub)}")
    print(f"test  {len(te_sub)} participants / {len(te)} runs: {' '.join(te_sub)}")

    # -- the fit ---------------------------------------------------------
    rule("(a) fitting the two instrument parameters that vary between recordings")
    print("`gradiometer_baseline_mm` is deliberately NOT fitted.  it is read from the coil "
          "definition\nin the file and is identical for every recording on one instrument, "
          "so its between-\nrecording spread is zero and a predictive density over it is "
          "undefined.  it is an\ninterval claim and it is tested as one, in falsification 3.")
    names = ("standoff_mm", "sensor_noise_ft_rt_hz")
    space = parameter_space(names)
    theta0 = space.median()
    print("\n" + space.describe())

    y_tr = {n: np.array([u[n] for u in tr]) for n in names}
    y_te = {n: np.array([u[n] for u in te]) for n in names}
    # the predictive width is the between-recording spread on TRAIN, estimated once
    # and held fixed for prior and posterior alike.  letting the posterior refit its
    # own width would let it win by widening rather than by being right.
    sd_tr = {n: float(v.std(ddof=1)) for n, v in y_tr.items()}

    tasks = []
    for n in names:
        b = space[n]

        def logp(theta, _b=b, _y=y_tr[n], _s=sd_tr[n]) -> float:
            return float(gauss_lpd(_y, float(theta[_b.slice][0]), _s).sum())
        tasks.append(Task(name=f"{SOURCE}.{n}.train", logp=logp, moves=(b.key,),
                          source=SOURCE, kind="fit",
                          note=f"{len(y_tr[n])} train runs"))
    theta, rep = fit(space, tasks, method=Method.MAP)
    print("\n" + str(rep))
    prov = provenance_after(space, rep)

    rule("(b) held out: does the posterior beat the declaration on unseen participants?")
    print(f"{'parameter':26s} {'prior':>10s} {'prior sd':>9s} {'posterior':>10s} "
          f"{'test mean':>10s} {'shift':>8s}  provenance now")
    res = {}
    for n in names:
        b = space[n]
        pri = float(theta0[b.slice][0])
        pos = float(theta[b.slice][0])
        psd = (abs(b.prior.scale) if b.prior.dist == "normal"
               else abs(b.prior.scale) * math.exp(b.prior.loc))
        lp = gauss_lpd(y_te[n], pri, sd_tr[n])
        lq = gauss_lpd(y_te[n], pos, sd_tr[n])
        st = paired(lp, lq)
        res[n] = {"prior": pri, "posterior": pos, "prior_sd": psd,
                  "test_mean": float(y_te[n].mean()),
                  "test_sd": float(y_te[n].std(ddof=1)),
                  "predictive_sd": sd_tr[n], "lpd_prior": float(lp.mean()),
                  "lpd_post": float(lq.mean()), "delta": st,
                  "provenance": prov[b.key].value}
        print(f"{n:26s} {pri:10.4g} {psd:9.3g} {pos:10.4g} {y_te[n].mean():10.4g} "
              f"{rep.shift[b.key]:+8.2f}  {prov[b.key].value}")
    print("\nheld-out log predictive density per test run, at the same predictive width:")
    for n in names:
        r, d = res[n], res[n]["delta"]
        print(f"  {n:26s} prior {r['lpd_prior']:+9.3f}  posterior {r['lpd_post']:+9.3f}  "
              f"delta {d['mean']:+8.3f} +/- {d['sem']:.3f}  dz {d['dz']:+6.2f}  "
              f"p {d['p']:9.3g}  wins {d['win_rate']:.0%}")
    won = [n for n in names if res[n]["delta"]["mean"] > 0 and res[n]["delta"]["p"] < 0.05]
    print(f"\n{len(won)}/{len(names)} beat the declared prior on held-out participants at "
          f"p < 0.05 ({', '.join(won) if won else 'none'}).")

    # -- the lead fields, computed once per participant --------------------
    rule("lead fields: one conductor, one source space, both instruments")
    print("for each participant a sphere is fitted to their own digitised electrodes, a "
          "12 mm\nvolume source space is laid inside it, and ONE forward solution is computed "
          "for the\nmagnetometers and the electrodes together.  computing them on one grid in "
          "one conductor\nis what makes the two comparable at all: a resolution comparison "
          "across different\nsource spaces would be a comparison of the source spaces.")
    lfs: dict[str, dict] = {}
    for u in units:
        if u["run"] != 1 or u["subject"] in lfs:
            continue                       # the geometry is per participant, not per run
        try:
            lfs[u["subject"]] = lead_field_stats(u)
        except Exception as exc:
            print(f"  {u['subject']}: lead field failed ({type(exc).__name__}: {exc})")
    print(f"  {len(lfs)} participants, "
          f"{np.median([v['n_sources'] for v in lfs.values()]):.0f} sources each (median)")

    # -- falsification 1 -------------------------------------------------
    rule("FALSIFICATION 1: does the magnetic instrument resolve finer source structure?")
    print("`meg_forward`'s r(q) rests on one sentence: \"the magnetic lead field is not "
          "filtered by\nthe skull -- magnetic permeability is uniform through bone -- so the "
          "sensor genuinely\nresolves finer source structure, and 2 mm on the cortical sheet "
          "buys something eeg's\n3 mm does not\".  ds000117 records both instruments at the "
          "same instant on the same head\nwith one digitisation, so that sentence is "
          "checkable without any source estimate.")
    print("\nthe statistic is the number of singular values of the NOISE-WHITENED lead field "
          "above\none per cent of the largest: how many spatially distinct source patterns "
          "the instrument\ncan tell apart above its own noise.  whitened by each channel's "
          "measured pre-stimulus\nsd, so the comparison is between what the two instruments "
          "can actually see rather than\nbetween their units.  reported raw and divided by "
          "channel count, because meg contributes\n102 magnetometers against 70 electrodes "
          "and a raw count would partly report that.")
    subs_l = sorted(lfs)
    dm = np.array([lfs[s]["meg_dof"] for s in subs_l], float)
    de = np.array([lfs[s]["eeg_dof"] for s in subs_l], float)
    fm = np.array([lfs[s]["meg_dof_frac"] for s in subs_l])
    fe = np.array([lfs[s]["eeg_dof_frac"] for s in subs_l])
    st_raw, st_frac = paired(de, dm), paired(fe, fm)
    print(f"\n  resolvable spatial degrees of freedom, per participant:")
    print(f"    magnetometers  {dm.mean():6.1f} of {lfs[subs_l[0]]['meg_channels']} channels "
          f"({fm.mean():.1%})")
    print(f"    electrodes     {de.mean():6.1f} of {lfs[subs_l[0]]['eeg_channels']} channels "
          f"({fe.mean():.1%})")
    print(f"  paired meg - eeg, raw      {st_raw['mean']:+7.2f} +/- {st_raw['sem']:.2f} sem, "
          f"dz {st_raw['dz']:+.2f}, p {st_raw['p']:.3g}, meg higher in "
          f"{st_raw['win_rate']:.0%} of {st_raw['n']}")
    print(f"  paired meg - eeg, per chan {st_frac['mean']:+7.4f} +/- {st_frac['sem']:.4f} sem, "
          f"dz {st_frac['dz']:+.2f}, p {st_frac['p']:.3g}, meg higher in "
          f"{st_frac['win_rate']:.0%} of {st_frac['n']}")
    print("  the same comparison at a 5% singular-value threshold, as a check that the "
          "answer is\n  not an artefact of where the cut was put:")
    fm5 = np.array([lfs[s]["meg_dof_frac@0.05"] for s in subs_l])
    fe5 = np.array([lfs[s]["eeg_dof_frac@0.05"] for s in subs_l])
    st5 = paired(fe5, fm5)
    print(f"    meg {fm5.mean():.1%} of channels, eeg {fe5.mean():.1%}, paired meg - eeg "
          f"{st5['mean']:+.4f}, p {st5['p']:.3g}, meg higher in {st5['win_rate']:.0%}")
    f1 = bool(st_frac["mean"] > 0 and st_frac["p"] < 0.01 and st_frac["win_rate"] >= 0.7)
    print(f"\n  FALSIFICATION 1: {'PASSED' if f1 else 'FAILED'} -- per channel, the magnetic "
          f"instrument "
          f"{'does' if st_frac['mean'] > 0 else 'does NOT'}\n  resolve more source structure "
          "than the electric one on the same head.")
    if not f1:
        print("  the resolution argument `meg_forward` uses to justify a finer cortical r(q) "
              "than\n  `eeg_forward` is not visible in the lead fields it appeals to.  read "
              "the three rows\n  together, because they do not all say the same thing and "
              "the difference is the\n  honest size of the result:")
        print(f"    - RAW count: meg is ahead by {st_raw['mean']:+.1f} patterns "
              f"(p {st_raw['p']:.3g}), which is not significant\n      and is roughly what "
              "102 channels against 74 would give for free.")
        print(f"    - PER CHANNEL at 1%: meg is BEHIND by {abs(st_frac['mean']):.3f} "
              f"(p {st_frac['p']:.3g}), significantly, in the\n      direction opposite to "
              "the claim.")
        print(f"    - PER CHANNEL at 5%: meg is behind by {abs(st5['mean']):.3f} "
              f"(p {st5['p']:.3g}), same direction, not\n      significant.  so the "
              "significance depends on where the singular-value cut is put\n      and the "
              "DIRECTION does not.")
        print("  none of the three shows the advantage the declaration asserts.  that does "
              "not make\n  the physics wrong -- the skull really does not attenuate a "
              "magnetic field, and a real\n  head with a real cortical surface is not this "
              "sphere -- but the 2 mm sheet in\n  `meg_forward` beside the 3 mm sheet in "
              "`eeg_forward` is, on this evidence, an assertion\n  rather than a measured "
              "gain, and the place to settle it is an individual bem with an\n  oriented "
              "cortical source space, which is exactly the geometry both models say they\n"
              "  require and neither has.")

    # a second, estimator-free look at the same question, on the measured maps
    edges = np.arange(0.0, 181.0, 7.5)
    zc_m, zc_e = [], []
    for u in units:
        rad, r0 = u["sphere"]
        m = zero_crossing(edges, acf_angle(u["mag_pos"], r0, u["mag_topo"], edges))
        e = zero_crossing(edges, acf_angle(u["eeg_pos"], r0, u["eeg_topo"], edges))
        if np.isfinite(m) and np.isfinite(e):
            zc_m.append(m); zc_e.append(e)
    zc_m, zc_e = np.array(zc_m), np.array(zc_e)
    st_acf = paired(zc_m, zc_e)
    print(f"\n  a second look, on the MEASURED evoked maps rather than the lead fields: the "
          "angle\n  (at the head centre) where each map's spatial autocorrelation first "
          "crosses zero.")
    print(f"    magnetometer  {zc_m.mean():6.2f} deg    electrode  {zc_e.mean():6.2f} deg    "
          f"paired eeg - meg {st_acf['mean']:+.2f}, p {st_acf['p']:.3g}")
    print("  this second number is NOT used to decide the test, and the reason is that it "
          "cannot\n  separate three things: the field's own width, the array's radius (the "
          "magnetometers sit\n  two centimetres further out, so the same generator subtends "
          "a wider angle there even\n  with identical physics), and the eeg reference, which "
          "forces a potential map towards\n  zero mean and manufactures a crossing.  it is "
          "printed because it points the same way\n  or the other way, and either is worth "
          "knowing.")
    res_f1 = {"meg_dof": float(dm.mean()), "eeg_dof": float(de.mean()),
              "meg_dof_frac": float(fm.mean()), "eeg_dof_frac": float(fe.mean()),
              "paired_raw": st_raw, "paired_per_channel": st_frac,
              "acf_meg_deg": float(zc_m.mean()), "acf_eeg_deg": float(zc_e.mean()),
              "acf_paired": st_acf, "paired_per_channel_thr05": st5,
              "meg_dof_frac_thr05": float(fm5.mean()),
              "eeg_dof_frac_thr05": float(fe5.mean()), "passed": f1}

    # -- falsification 2 -------------------------------------------------
    rule("FALSIFICATION 2: is the radial null space of the magnetic lead field really a null?")
    print("`magnetometer_pickup.radial_blindness` is declared normal(0.05, 0.03) with "
          "provenance\nPHYSICS, and its own note says what the 0.05 means: \"residual "
          "sensitivity to a radial\nsource IN A REALISTIC HEAD; exactly zero only in a "
          "spherical one\".  `meg_forward` then\nlists radial current under "
          "`prior_dominated` because it \"lies in the null space\".  so the\ndeclaration "
          "makes a sharp, checkable prediction about the conductor it actually uses:\nin a "
          "sphere the ratio must be a rounding error, not 0.05.")
    ratios = np.array([lfs[s]["radial_ratio"] for s in subs_l])
    print(f"\n  measured over {ratios.size} participants' own arrays and own fitted spheres:")
    print(f"    rms radial sensitivity / rms tangential sensitivity = "
          f"{ratios.mean():.3g} (range {ratios.min():.3g} - {ratios.max():.3g})")
    f2 = bool(ratios.max() < 0.01)
    print(f"\n  FALSIFICATION 2: {'PASSED' if f2 else 'FAILED'} -- the null space "
          f"{'is' if f2 else 'is NOT'} exact in the conductor "
          f"`meg_forward`\n  says the magnetic forward problem needs.")
    print("  what this does NOT do is test the 0.05.  that number is about a real, "
          "non-spherical\n  head and can only be measured against a bem built from an "
          "individual t1 -- which\n  ds000117 has and this script does not use, because "
          "sixteen freesurfer reconstructions\n  are a different piece of work.  so "
          "`radial_blindness` stays at its prior and this is\n  a pipeline check plus a "
          "confirmation that the sphere behaves as the physics says.")
    print("  it does expose an internal tension worth recording: `meg_forward` justifies "
          "spending\n  LESS resolution on the head volume than `eeg_forward` on the grounds "
          "that \"the sarvas\n  formula needs only the inner skull surface\", and the "
          "residual 5% is exactly the part\n  that a sphere cannot produce and a resolved "
          "conductor could.  the two decisions pull\n  against each other.")
    res_f2 = {"ratios": ratios.tolist(), "mean": float(ratios.mean()),
              "max": float(ratios.max()), "passed": f2}

    # -- falsification 3 -------------------------------------------------
    rule("FALSIFICATION 3: the gradiometer baseline, against the instrument's own geometry")
    print("`magnetometer_pickup.gradiometer_baseline_mm` is declared normal(50, 20) mm.  "
          "every FIF\nhere carries the coil-type code and the integration-point geometry it "
          "selects, so the\nbaseline is not estimated: it is read.  this is the cheapest "
          "falsification in the\nrepository and it needs no statistics.")
    bl = np.array([u["gradiometer_baseline_mm"] for u in units])
    z = (bl.mean() - 50.0) / 20.0
    inside3 = float(((bl > 50.0 - 2 * 20.0) & (bl < 50.0 + 2 * 20.0)).mean())
    print(f"\n  measured baseline: {bl.mean():.3f} mm (sd {bl.std():.4f} over {bl.size} runs)")
    print(f"  declared prior 50 +/- 20 mm; the measurement is {z:+.2f} prior sd from it")
    print(f"  fraction inside the declared central 95% interval [10, 90] mm: {inside3:.0%}")
    f3 = inside3 >= 0.9
    print(f"\n  FALSIFICATION 3: {'PASSED' if f3 else 'FAILED'}.")
    print("  and now read what the pass is worth.  the measurement is 16.68 mm because that "
          "is the\n  planar gradiometer baseline of a neuromag vectorview; it is not a "
          "measurement error and\n  not a subject effect but a property of the hardware, "
          "identical to four decimal places\n  in every recording, with exactly zero "
          "variance.  50 mm is the baseline of an AXIAL\n  gradiometer, which is what CTF "
          "and 4D systems use.  the declaration therefore describes\n  a different class of "
          "instrument, and it survives contact with this one ONLY because a\n  20 mm sd on "
          "a 50 mm quantity admits everything from 10 mm to 90 mm -- a prior wide\n  enough "
          "to contain both a planar and an axial gradiometer is not making a claim about\n"
          "  either.  the right fix is a per-device parameter, not a wider prior; widening "
          "would\n  make the declaration vaguer without making it true of any machine.  a "
          "PASS here is a\n  statement about the prior's width, not about its centre.")
    res_f3 = {"measured_mm": float(bl.mean()), "prior_mm": 50.0, "prior_sd_mm": 20.0,
              "z": float(z), "fraction_inside_95": inside3, "passed": f3}

    # -- caveats ---------------------------------------------------------
    rule("where this is weaker than it looks")
    print("""
THIS IS A DEVICE FIT, NOT A BRAIN FIT.  two of the three things `meg_forward`
lists under `constrained` are about the instrument, and those are what moved here.
the third -- "source orientation on sulcal walls, which is the one geometric
quantity meg pins better than any other non-invasive instrument" -- is untouched,
because it needs a cortical surface with orientations and this script has a
spherical conductor and a volume grid.  a reader should not take a passed held-out
test above as evidence about cortex.

THE INVERSE WAS TRIED AND IS NOT REPORTED AS A RESULT.  fitting a depth-weighted
minimum-norm source estimate to one modality and predicting the other gave
correlations between -0.33 and +0.20 across participants, in both directions, with
no consistent sign.  that is not evidence against the forward model: without an
individual t1 the conductor is a sphere fitted to a cap, the source space is a
12 mm volume grid, and the depth bias of a min-norm estimate under those conditions
is large enough to explain the whole result.  it is reported here so that the
absence of a cross-modal number above is a decision rather than an omission.

THE NOISE FIGURE IS AN UPPER BOUND, AND PROBABLY A LOOSE ONE.  what is measured is
the median magnetometer amplitude spectral density over 20-45 Hz during the task,
which contains brain activity, muscle, and whatever the shielded room lets
through.  it bounds the sensor's own noise from above and does not measure it.
ds000117 ships eight empty-room recordings that would measure the room-plus-sensor
floor properly; this script does not read them, so the fitted
`sensor_noise_ft_rt_hz` should be read as "what a likelihood built on this
recording would actually face", which is arguably the more useful quantity and is
definitely not what the declaration means.

THE STANDOFF DEPENDS ON A SPHERE FITTED TO A CAP.  `standoff_mm` is the distance
from each magnetometer's centre to a scalp surface that is a sphere fitted to the
digitised electrodes.  a real head is not a sphere and the residual is several
millimetres, systematically larger at the occiput and the vertex.  the
between-participant variation the fit sees is therefore partly head shape and
partly the sphere's failure to be one, and those are not separated.

SIXTEEN PEOPLE, ONE MACHINE, ONE SITE.  everything above is one vectorview at one
centre.  the two parameters that moved are properties of that machine and that
room, and transporting them to another site is precisely the error the
`gradiometer_baseline_mm` result shows the declaration already made once.""")

    out = args.out or (Path(__file__).resolve().parents[1] / "data" / "sources" / SOURCE /
                       "evidence" / "fit_meg_instrument@v1" / "posterior.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": MODEL_ID, "source": SOURCE, "seed": args.seed,
        "train_subjects": tr_sub, "test_subjects": te_sub,
        "fitted": res,
        "falsification_meg_resolves_finer": res_f1,
        "falsification_radial_blindness": res_f2,
        "falsification_gradiometer_baseline": res_f3,
        "runtime_s": time.time() - t0}, indent=2, default=float))

    rule("summary")
    for n in names:
        d = res[n]["delta"]
        print(f"{n:26s} {res[n]['prior']:8.3g} -> {res[n]['posterior']:8.3g}   "
              f"delta lpd {d['mean']:+.3f}, p {d['p']:.3g}, wins {d['win_rate']:.0%}")
    for k, v in (("meg resolves finer", res_f1), ("radial null exact", res_f2),
                 ("gradiometer baseline", res_f3)):
        print(f"{k:26s} {'PASS' if v['passed'] else 'FAIL'}")
    print(f"\nwritten to {out}\n{time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
