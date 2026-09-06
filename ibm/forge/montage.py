"""what part of cortex each montage actually weighs.

`scripts/forge_joint.py` fitted one global value per parameter against a posterior
scalp average, a pair of bipolar derivations and a magnetometer array, and found
the three disagree at up to 88 sigma about `alpha_resonator.f0_hz` -- 4.33, 10.4
and 17.4 Hz.  the script's own closing note said what the experiment could not
separate: the collapse to GLOBAL tying, a missing montage factor in the forward
model, and genuine cohort differences.  this module is the missing montage
factor.

the argument for it is not statistical, it is electromagnetic.  a scalp
potential is a weighted sum over cortical current sources, and the weights are
the lead field -- not a distance kernel, because the skull is two orders of
magnitude less conductive than brain and the csf beneath it shunts current
tangentially, which changes the ORDER of sensor sensitivities and not merely
their scale (`ibm.topologies.em` says exactly this).  so a bipolar Fpz-Cz
derivation and an eleven-channel parieto-occipital average are not two noisy
looks at one number.  they are two different linear functionals of one cortical
field, and if that field varies over cortex -- which is what
`Tying.PER_PARTITION` on `ei_loop_lti` and `alpha_resonator` DECLARES -- then
they must report different effective values even when nothing about the model is
wrong.  an anterior-posterior gradient of alpha frequency is one of the
best-documented facts about the rhythm, and the three montages above are ordered
by how posterior they are.

### why a POWER weighting and not the lead field itself

what is fitted is a power spectrum, and `evidence_from_signal` averages the psds
of the channels rather than the traces.  so the quantity a montage reports at
frequency f is

    P_montage(f)  =  (1/C) sum_c sum_q |L(c,q) . n_q|^2 S(f; theta(q))

under the assumption that distinct cortical patches are mutually incoherent.
that assumption is wrong in detail -- cortical sources are spatially correlated
over centimetres -- and it is wrong in a direction that FLATTENS the weighting
towards uniform, so a montage factor built this way understates how differently
two montages see cortex.  the alternative, coherent summation, needs a phase
structure nothing here measures, and inventing one would put a free field in the
forward model where a measured one belongs.

the normal `n_q` is the white surface's, for the reason `from_freesurfer` gives:
the transmembrane current runs along the pyramidal dendrite and the white
surface's normal is that direction, while the pial normal is that direction plus
wherever the cortex happened to bulge.

### what is an approximation here, named as one

three of the four sources did not ship electrode positions with the bytes this
repository holds, and one shipped a different subject's head.  every one of those
is recorded on the returned object rather than smoothed over:

- the head model is always the mne `sample` subject's three-layer BEM.  it is one
  head, and none of the four cohorts is that head.  what a lead field over one
  real head buys is the ORDERING and the relative magnitude of a montage's
  sensitivity across cortex, which is what the mixture weights need; what it
  cannot buy is a cohort's own anatomy.
- eegmmidb and sleep-edfx name 10-10 labels and carry no digitisation, so their
  electrodes are the `standard_1005` template positions carried onto the sample
  head by the similarity transform that matches the two fiducial triples.  the
  residual of that fit is measured and reported.
- ds000117 DID digitise, so its own electrode positions are used, mapped onto the
  sample head by the same fiducial similarity.  its magnetometers are a
  Vectorview 306 array and so is the sample's, so the sample's own magnetometer
  geometry is the array's geometry; what differs is where the head sat in the
  dewar.
- eegmmidb's EDFs do not record their reference.  a monopolar channel's lead
  field is L(c) - L(ref), and with ref unknown the far-reference approximation
  L(c) is used.  a common-average reference would subtract the montage mean and
  is available as `reference="average"` for anyone who wants the sensitivity of
  the result to that choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["CorticalField", "MontageWeights", "cortical_field", "montage_weight_table",
           "standard_positions_on_sample", "SPECS"]


# ---------------------------------------------------------------------------
# the cortical field the montages weigh
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorticalField:
    """a subsample of the white surface, with the coordinates a low-rank field needs.

    subsampled rather than complete, and the count is the whole argument for
    doing it this way.  a per-partition map over 34 areas by 6 layers is 204
    numbers that three montages cannot identify -- each of them integrates the
    whole sheet into one number, so the design matrix has rank 3.  what three
    montages CAN identify is a low-order spatial trend, and for that the sheet
    only has to be sampled densely enough that the weighted integrals converge,
    which a few thousand vertices does to well under a percent.

    `ap` is anterior-positive and standardized by its own area-weighted sd, so a
    gradient coefficient reads as "log parameter change per sd of position along
    the anterior-posterior axis" and is comparable across parameters.
    """

    xyz_mm: np.ndarray            # (N, 3) surface RAS
    normal: np.ndarray            # (N, 3) unit, white-surface outward
    area_mm2: np.ndarray          # (N,) vertex area, the integration measure
    ap: np.ndarray                # (N,) standardized anterior-posterior
    ml: np.ndarray                # (N,) standardized |medial-lateral|
    note: str = ""

    @property
    def n(self) -> int:
        return int(self.xyz_mm.shape[0])


def _vertex_normals_and_areas(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """area-weighted vertex normals and the one-third-of-each-triangle vertex area.

    the cross product is not normalized before accumulation on purpose: its length
    is twice the triangle's area, so summing raw cross products weights each
    incident face by its area, which is the discrete surface normal rather than a
    popularity vote among small slivers.
    """
    tri = v[f]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm = np.zeros_like(v)
    area = np.zeros(len(v))
    for k in range(3):
        np.add.at(nrm, f[:, k], cr)
        np.add.at(area, f[:, k], 0.5 * np.linalg.norm(cr, axis=1) / 3.0)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    return nrm / np.maximum(ln, 1e-30), area


def cortical_field(*, n_nodes: int = 4000, seed: int = 0,
                   subject: str = "sample") -> CorticalField:
    """the sample subject's white surface, thinned to `n_nodes` with its measure kept.

    thinned by an area-weighted draw without replacement rather than by taking
    every k-th vertex: freesurfer's vertex ordering is a traversal of the mesh,
    so a stride is a spatially structured sample and would put a smooth bias in
    exactly the coordinate this module is measuring.  the vertex areas are
    rescaled so the thinned set integrates to the full surface area, which makes
    every weighted mean below an estimate of the same integral at any `n_nodes`.
    """
    from ibm.materialize.geometry import from_freesurfer, sample_paths

    p = sample_paths(subject)
    g = from_freesurfer(p.subject_dir, surface="white")["cortical_surface"]
    v = np.asarray(g.vertices, float)
    f = np.asarray(g.faces, np.int64)
    nrm, area = _vertex_normals_and_areas(v, f)

    rng = np.random.default_rng(seed)
    keep = np.arange(len(v)) if n_nodes >= len(v) else \
        rng.choice(len(v), size=n_nodes, replace=False, p=area / area.sum())
    keep = np.sort(keep)
    a = area[keep] * (area.sum() / area[keep].sum())

    xyz, nn = v[keep], nrm[keep]
    y = xyz[:, 1]
    mu = float(np.average(y, weights=a))
    sd = float(np.sqrt(np.average((y - mu) ** 2, weights=a)))
    ap = (y - mu) / max(sd, 1e-9)
    lat = np.abs(xyz[:, 0])
    mul = float(np.average(lat, weights=a))
    msd = float(np.sqrt(np.average((lat - mul) ** 2, weights=a)))
    ml = (lat - mul) / max(msd, 1e-9)
    return CorticalField(
        xyz, nn, a, ap, ml,
        note=(f"mne {subject} white surface, {len(v):,} vertices thinned to {len(keep):,} by "
              f"an area-weighted draw; total area {area.sum() / 100:.0f} cm^2; the "
              f"anterior-posterior axis has sd {sd:.1f} mm about y = {mu:.1f} mm"))


# ---------------------------------------------------------------------------
# electrodes: template labels carried onto the sample head
# ---------------------------------------------------------------------------


def _fiducials(info) -> np.ndarray:
    import mne
    F = mne.io.constants.FIFF
    got = {d["ident"]: np.asarray(d["r"], float) for d in (info["dig"] or [])
           if d["kind"] == F.FIFFV_POINT_CARDINAL}
    need = (F.FIFFV_POINT_LPA, F.FIFFV_POINT_NASION, F.FIFFV_POINT_RPA)
    if not all(k in got for k in need):
        raise ValueError("this recording carries no cardinal fiducials, so its montage "
                         "cannot be placed on another head")
    return np.stack([got[k] for k in need])


def _similarity_to_sample(src_fid: np.ndarray, tgt_fid: np.ndarray) -> tuple[np.ndarray, float]:
    """rigid plus one uniform scale, from three fiducials to three fiducials.

    scale and not rigid-only, because the template montage is on a standard head
    and the sample subject's is a real one; refusing the scale would put the
    electrodes off the scalp by whatever the two heads differ in size, and the
    forward solver would then either project them back -- silently undoing the
    error in an uncontrolled way -- or solve for electrodes floating in air.  the
    residual is returned because three points determine seven parameters exactly
    only up to the reflection, and a large residual means the two fiducial
    conventions differ and the answer is not usable.
    """
    import mne
    m = mne.coreg.fit_matched_points(src_fid, tgt_fid, scale=True, out="trans")
    m = np.asarray(m, float)
    moved = src_fid @ m[:3, :3].T + m[:3, 3]
    return m, float(np.sqrt(((moved - tgt_fid) ** 2).sum(1)).mean())


def standard_positions_on_sample(labels: Sequence[str], *, montage: str = "standard_1005",
                                 subject: str = "sample") -> tuple[np.ndarray, float]:
    """template 10-05 positions for `labels`, in the sample subject's head frame."""
    import mne
    from ibm.materialize.geometry import sample_paths

    mne.set_log_level("ERROR")
    p = sample_paths(subject)
    info = mne.io.read_raw_fif(p.raw_fif, preload=False, verbose="ERROR").info
    mont = mne.channels.make_standard_montage(montage)
    d = mont.get_positions()
    src = np.stack([d["lpa"], d["nasion"], d["rpa"]])
    m, resid = _similarity_to_sample(src, _fiducials(info))
    xyz = np.stack([np.asarray(d["ch_pos"][l], float) for l in labels])
    return xyz @ m[:3, :3].T + m[:3, 3], resid


def _digitised_positions_on_sample(raw_fif: Path, *, subject: str = "sample"
                                   ) -> tuple[list[str], np.ndarray, float]:
    """another recording's own EEG electrodes, carried onto the sample head."""
    import mne
    from ibm.materialize.geometry import sample_paths

    mne.set_log_level("ERROR")
    src_info = mne.io.read_raw_fif(str(raw_fif), preload=False, allow_maxshield=True,
                                   verbose="ERROR").info
    tgt_info = mne.io.read_raw_fif(sample_paths(subject).raw_fif, preload=False,
                                   verbose="ERROR").info
    m, resid = _similarity_to_sample(_fiducials(src_info), _fiducials(tgt_info))
    picks = mne.pick_types(src_info, meg=False, eeg=True, exclude="bads")
    names = [src_info["ch_names"][i] for i in picks]
    xyz = np.stack([np.asarray(src_info["chs"][i]["loc"][:3], float) for i in picks])
    ok = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1e-4)
    return [n for n, k in zip(names, ok) if k], xyz[ok] @ m[:3, :3].T + m[:3, 3], resid


# ---------------------------------------------------------------------------
# the forward solve
# ---------------------------------------------------------------------------


def _expand(fwd, n: int) -> np.ndarray:
    """put the solved columns back where they belong, and leave the rest at zero.

    a boundary-element solver drops any source outside its innermost boundary, and
    a handful of white-surface vertices always fall there because the bem's
    inner-skull surface and the freesurfer reconstruction do not agree to the
    millimetre.  the returned matrix is over the WHOLE field, so a dropped column
    reads as "this montage cannot see this patch" -- which for a fraction under a
    percent is a rounding error in an integral, and is counted and reported rather
    than assumed to be one.
    """
    import numpy as _np
    sol = _np.asarray(fwd["sol"]["data"], float)
    inuse = _np.asarray(fwd["src"][0]["inuse"], bool)
    out = _np.zeros((sol.shape[0], n))
    out[:, inuse] = sol
    return out


def _eeg_lead_field(xyz_head_m: np.ndarray, field: CorticalField, *,
                    subject: str = "sample") -> np.ndarray:
    """(C, N) potential per unit normal-oriented dipole, this subject's BEM.

    the electrodes are handed to the solver in the HEAD frame together with the
    sample's own `-trans.fif`, which is the same pair `bem_lead_field` uses; the
    only thing that differs is that the electrode set is not the one in the raw
    file, and that is the entire point.  the solver projects eeg electrodes onto
    the outer skin surface itself, so a template cap that lands a few millimetres
    off the scalp is corrected by the head model rather than by a fudge here.
    """
    import mne
    from ibm.materialize.geometry import sample_paths

    mne.set_log_level("ERROR")
    p = sample_paths(subject)
    sols = sorted(p.bem_dir.glob("*-bem-sol.fif"))
    best = max(sols, key=lambda q: q.name.count("5120"))
    names = [f"E{i:03d}" for i in range(len(xyz_head_m))]
    info = mne.create_info(names, 1000.0, "eeg")
    mont = mne.channels.make_dig_montage(
        ch_pos={n: x for n, x in zip(names, xyz_head_m)}, coord_frame="head")
    info.set_montage(mont)
    src = mne.setup_volume_source_space(
        pos=dict(rr=field.xyz_mm / 1000.0, nn=field.normal), verbose="ERROR")
    fwd = mne.make_forward_solution(info, trans=str(p.trans_fif), src=src, bem=str(best),
                                    meg=False, eeg=True, verbose="ERROR")
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=False, verbose="ERROR")
    return _expand(fwd, field.n)


def _meg_lead_field(field: CorticalField, *, subject: str = "sample") -> np.ndarray:
    """(102, N) magnetometer gain for normal-oriented dipoles, same head, same BEM."""
    import mne
    from ibm.materialize.geometry import sample_paths

    mne.set_log_level("ERROR")
    p = sample_paths(subject)
    sols = sorted(p.bem_dir.glob("*-bem-sol.fif"))
    best = max(sols, key=lambda q: q.name.count("5120"))
    info = mne.io.read_raw_fif(p.raw_fif, preload=False, verbose="ERROR").info
    picks = mne.pick_types(info, meg="mag", eeg=False, exclude=())
    info = mne.pick_info(info, picks)
    src = mne.setup_volume_source_space(
        pos=dict(rr=field.xyz_mm / 1000.0, nn=field.normal), verbose="ERROR")
    fwd = mne.make_forward_solution(info, trans=str(p.trans_fif), src=src, bem=str(best),
                                    meg=True, eeg=False, verbose="ERROR")
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=False, verbose="ERROR")
    return _expand(fwd, field.n)


# ---------------------------------------------------------------------------
# the four sources' montages
# ---------------------------------------------------------------------------

#: parieto-occipital labels, exactly the ones `forge_joint.POSTERIOR_CH` picks.
EEGMMIDB_LABELS = ("P3", "P1", "Pz", "P2", "P4", "PO3", "POz", "PO4", "O1", "Oz", "O2")

#: `eval_sleep_state.EEG_CHANNELS`, as electrode pairs.
SLEEP_PAIRS = (("Fpz", "Cz"), ("Pz", "Oz"))

SPECS = ("eegmmidb", "sleep-edfx", "ds000117/eeg", "ds000117/meg", "ds004873")


@dataclass
class MontageWeights:
    """one source's normalized power sensitivity over the cortical field.

    `w` sums to one and multiplies a per-node spectrum; `mean_ap` is the number
    that decides whether an anterior-posterior gradient is identifiable at all,
    because a set of montages that all have the same `mean_ap` cannot separate a
    gradient from a shift of the global value however many of them there are.
    """

    name: str
    w: np.ndarray                 # (N,) sums to 1
    mean_ap: float
    mean_ml: float
    approximations: tuple[str, ...] = ()
    detail: str = ""
    diagnostics: dict = field(default_factory=dict)


def _normalize(power: np.ndarray) -> np.ndarray:
    s = float(power.sum())
    return power / s if s > 0 else np.full(power.size, 1.0 / power.size)


def montage_weight_table(field: CorticalField, *, ds000117_raw: Path | None = None,
                         reference: str = "far") -> dict[str, MontageWeights]:
    """the five weightings the joint fit needs, from one BEM solve per instrument.

    one solve per instrument and not per source: the template electrodes of
    eegmmidb and sleep-edfx come out of the same 10-05 cap and the same forward
    problem, and solving it twice would only give two chances to solve it
    differently.
    """
    labels = tuple(dict.fromkeys(EEGMMIDB_LABELS + tuple(c for p in SLEEP_PAIRS for c in p)))
    xyz, resid = standard_positions_on_sample(labels)
    l_tpl = _eeg_lead_field(xyz, field)                       # (len(labels), N)
    idx = {l: i for i, l in enumerate(labels)}

    out: dict[str, MontageWeights] = {}
    area = field.area_mm2

    def finish(name, w, approx, detail, diag=None):
        w = _normalize(w)
        out[name] = MontageWeights(
            name, w, float(np.dot(w, field.ap)), float(np.dot(w, field.ml)),
            tuple(approx), detail, diag or {})

    # -- eegmmidb: eleven monopolar parieto-occipital channels, psds averaged --
    rows = np.stack([l_tpl[idx[l]] for l in EEGMMIDB_LABELS])
    if reference == "average":
        rows = rows - l_tpl.mean(0, keepdims=True)
    finish("eegmmidb", (rows ** 2).mean(0),
           [f"standard_1005 template electrodes on the sample head; fiducial residual "
            f"{resid * 1000:.1f} mm",
            "the EDFs do not record their reference, so a far reference is assumed"
            if reference == "far" else "referenced to the mean of the 10-05 cap"],
           f"{len(EEGMMIDB_LABELS)} monopolar channels: {', '.join(EEGMMIDB_LABELS)}")

    # -- sleep-edfx: two bipolar derivations, psds averaged ------------------
    rows = np.stack([l_tpl[idx[a]] - l_tpl[idx[b]] for a, b in SLEEP_PAIRS])
    finish("sleep-edfx", (rows ** 2).mean(0),
           [f"standard_1005 template electrodes on the sample head; fiducial residual "
            f"{resid * 1000:.1f} mm"],
           "2 bipolar derivations: " + ", ".join(f"{a}-{b}" for a, b in SLEEP_PAIRS))

    # -- ds000117: its own digitised eeg, and a vectorview magnetometer array --
    if ds000117_raw is not None and Path(ds000117_raw).is_file():
        names, xyz2, r2 = _digitised_positions_on_sample(Path(ds000117_raw))
        l_117 = _eeg_lead_field(xyz2, field)
        finish("ds000117/eeg", (l_117 ** 2).mean(0),
               [f"ds000117's own digitised electrodes carried onto the sample head; "
                f"fiducial residual {r2 * 1000:.1f} mm"],
               f"{len(names)} digitised channels from {Path(ds000117_raw).name}")
    else:
        finish("ds000117/eeg", (l_tpl ** 2).mean(0),
               ["no ds000117 raw file was given, so the whole-head 10-05 template cap "
                "stands in for its 70 digitised electrodes"],
               f"{len(labels)} template channels")

    l_mag = _meg_lead_field(field)
    finish("ds000117/meg", (l_mag ** 2).mean(0),
           ["the sample subject's own Vectorview 306 magnetometers; ds000117 used the same "
            "array, so the geometry is the array's and only the head position differs"],
           f"{l_mag.shape[0]} magnetometers")

    # -- ds004873: a grey-matter median, which weighs grey matter by volume --
    finish("ds004873", area.astype(float),
           ["a whole-grey-matter median over a T2-masked volume weighs cortex by area, "
            "not by any lead field; no forward model enters"],
           "grey-matter median, uniform over the sheet")
    return out
