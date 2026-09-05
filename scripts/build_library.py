#!/usr/bin/env python
"""build every named model in `ibm.materialize.library` against one real head.

    M = materialize(R, r, B, F, A, T, P)

`materialize_eeg_forward.py` takes one model apart in detail.  this takes all
forty and asks the blunter question: given exactly the geometry one mne `sample`
subject has -- a FreeSurfer reconstruction, a three-layer BEM, a 60-channel
digitised montage, a coregistration, and nothing else -- which of the library's
models can be materialized at all, and for the ones that cannot, *why not*.

the point is the triage, because three very different things look identical from
the outside when a build comes back empty:

- **missing data.**  this subject has no diffusion, no fMRI, no vessel
  segmentation, no implanted electrodes, no retina and no body.  a model that
  needs a tractogram is not broken, and neither is the library; the honest
  output is a row saying which file it wanted.
- **a request that mis-states what the model means.**  a region naming a label
  no partitioning system declares, or a placement that puts one quantity on two
  supports covering the same millimetres.  these look like missing data and are
  not: nothing external would fix them.
- **a bug.**  a step that gives up on everything because one thing was absent.

only the first is a fact about the world.  the other two were both present when
this script was first run, and the header printed at the end says what was done
about them.

*on strict.*  every build here runs `strict=False`, which is what makes the
first category a recorded gap rather than an abort -- and each model is then
re-built with `strict=True` if and only if it came back with no gaps, so the
"strict" column is measured rather than inferred.

*on cost.*  the expensive artefacts are the octree, the poisson-disk sheet and
the geodesic graph, not the BEM: a forward solution over 15,000 sources takes a
couple of seconds and a 3 mm sheet takes minutes.  so `ibm.materialize.cache`
is used throughout, models are built at their own declared r(q) rather than at
a common fine one, and a lead field is solved only for the models whose
`electromagnetic` topology actually has both a sensor array and a source table
to relate -- for everything else it is a topology outside the view, which the
build already reports as a note rather than a gap.
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ibm
from ibm.materialize import geometry as geo
from ibm.materialize.build import (
    MaterializationIncomplete, _named_supports, _place_components,
    _supports_to_materialize, build,
)
from ibm.materialize.trace import trace as _trace
from ibm.materialize.cache import Cache
from ibm.materialize.library import MODELS
from ibm.materialize.request import BudgetExceeded, SubjectSpec
from ibm.materialize.sites import MissingData
from ibm.topologies.builders import MissingInput
from ibm.registry import REGISTRY
from ibm.vocabulary import Resolution, ResolutionRule

#: the coarsening ladder.  a model whose declared r(q) does not fit its own
#: declared budget on a real head is not thereby unbuildable -- §1's whole
#: argument is that a coarser materialization is the same model until the
#: coarse-graining stops commuting -- so the request is coarsened and the rung
#: that fit is reported.  it is reported and not hidden: a model built at 4x its
#: declared spacing is a different statement about where resolution earns its
#: cost, and the row says so.
#:
#: the volume rungs come first, and that ordering is the one modelling decision
#: in this script.  `materialize_eeg_forward` makes the argument by hand for one
#: model: an octree's site count goes as the cube of 1/r and a poisson-disk
#: sheet's as the square, so a budget is almost always blown by the volume, and
#: a uniform coarsening pays for the conductor by destroying the sheet -- which
#: for a forward model is exactly backwards, since the sheet is what the lead
#: field integrates over and the conductor is a partial-volume conductivity that
#: 8 mm carries as well as 1.5 mm.  so the volume supports are coarsened alone
#: until they run out of room, and only then is everything coarsened together.
VOLUME_COARSENINGS = (2.0, 4.0, 8.0, 16.0)
COARSENINGS = (2.0, 4.0, 8.0)

#: markers that identify a problem as *external data this subject does not
#: have*, rather than as something wrong with the request or the code.  every
#: one of them is raised by `MissingData` or `MissingInput`, both of which exist
#: precisely to name the file rather than the symptom, so matching on their
#: wording is matching on a deliberate contract and not on a coincidence.
GAP_MARKERS = (
    "cannot run: it needs",                 # MissingData / MissingInput
    "name a support with no site table",    # the consequence of the above
)


def head(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ---------------------------------------------------------------------------
# the anatomy this subject actually has
# ---------------------------------------------------------------------------


class Aparc:
    """`cortical_areas` membership, from this subject's own `aparc+aseg.mgz`.

    worth supplying rather than leaving as a gap, because it is not a gap: the
    desikan-killiany parcellation is written by the same `recon-all` that
    produced the surfaces, its label names are *exactly* the ones
    `ibm.anatomy.systems` declares for `cortical_areas`, and it sits in the
    subject directory next to the aseg the tissue mask is already read from.
    reporting "no atlas" for a system whose atlas is on disk would put a real
    limitation and an unopened file in the same column.

    every other declared system is refused by name.  thalamic nuclei,
    hippocampal subfields, basal-ganglia territories, brainstem nuclei and
    vascular territories all need segmentations this `recon-all` did not produce,
    and a nearest-structure guess from `aseg`'s coarse labels would be a
    plausible-looking answer to a question the data cannot answer -- `Thalamus-
    Proper` is one label and `thalamic_nuclei` declares thirty-one.

    membership is hard 0/1 rather than soft, and that is a property of the
    source: a `recon-all` parcellation is a winner-take-all voxel labelling with
    no per-voxel probability in it.  the region machinery accepts soft weights
    and would use them; there are none to use, and inventing a smooth boundary
    here would be inventing an uncertainty estimate.
    """

    #: cortex in `aparc+aseg` is 1000 + k for the left hemisphere and 2000 + k
    #: for the right, where k indexes the parcellation's colour table.  the
    #: table is committed beside the labels, so the mapping is read rather than
    #: hardcoded and a subject processed with a different atlas version cannot
    #: silently shift by one.
    def __init__(self, subject_dir: Path, frame: str = geo.ANATOMICAL_FRAME) -> None:
        self.frame = frame
        self.path = subject_dir / "mri" / "aparc+aseg.mgz"
        vol = geo.read_mgh(self.path)
        self.data = np.asarray(vol.data)
        self.inv = np.linalg.inv(np.asarray(vol.vox2ras_tkr, float))
        self.names = self._ctab(subject_dir / "label" / "aparc.annot.ctab")
        declared = set(REGISTRY.anatomies["cortical_areas"].labels)
        self.ids: dict[str, tuple[int, ...]] = {}
        for k, name in self.names.items():
            if name in declared:
                self.ids.setdefault(name, ())
                self.ids[name] += (1000 + k, 2000 + k)
        self.unmatched = tuple(sorted(declared - set(self.ids)))

    @staticmethod
    def _ctab(path: Path) -> dict[int, str]:
        out: dict[int, str] = {}
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                out[int(parts[0])] = parts[1]
        return out

    def describe(self) -> str:
        return (f"  cortical_areas    {self.path.name}: {len(self.ids)} of "
                f"{len(REGISTRY.anatomies['cortical_areas'].labels)} declared labels present"
                + (f"; absent: {', '.join(self.unmatched)}" if self.unmatched else ""))

    def __call__(self, system: str, label: str, xyz, frame: str):
        if system != "cortical_areas":
            raise MissingData(
                "region_weights", f"anatomy[{system!r}][{label!r}]",
                f"soft membership in the {label!r} partition of {system!r}",
                f"this subject has aparc+aseg (cortical_areas) and aseg, and no segmentation "
                f"for {system!r}; that atlas is genuinely absent rather than unread")
        if frame != self.frame:
            raise MissingData(
                "region_weights", f"anatomy[{system!r}][{label!r}] in {frame!r}",
                f"the parcellation is in {self.frame!r} and was asked for in {frame!r}",
                "a warp between the two frames, which nothing here has measured")
        ids = self.ids.get(label)
        p = np.asarray(xyz, float).reshape(-1, 3)
        if ids is None or not len(p):
            return np.zeros(len(p))
        h = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ self.inv.T
        v = np.rint(h[:, :3]).astype(np.int64)
        ok = np.all((v >= 0) & (v < np.array(self.data.shape[:3])), axis=1)
        val = np.zeros(len(p), np.int64)
        val[ok] = self.data[v[ok, 0], v[ok, 1], v[ok, 2]]
        return np.isin(val, np.asarray(ids)).astype(float)


# ---------------------------------------------------------------------------
# the frames this subject actually has
# ---------------------------------------------------------------------------


def head_warp(coreg):
    """`eeg_cap -> subject_t1` and `meg_head -> subject_t1` from one -trans.fif.

    the sample dataset's transform is MNE's head->MRI, and MNE's "head" frame is
    defined by the digitised fiducials shared by the eeg cap and the meg helmet;
    one measurement therefore serves both, and saying so explicitly is better
    than handing the eeg coregistration to a meg model and hoping the name check
    does not notice.  every other pair is refused, which is the behaviour
    `Coregistration.warp` already has and the reason it has it.
    """
    def warp(xyz, src, dst):
        if src == "meg_head":
            src = coreg.src_frame
        if dst == "meg_head":
            dst = coreg.src_frame
        return coreg.warp(xyz, src, dst)
    return warp


# ---------------------------------------------------------------------------
# per-model inputs
# ---------------------------------------------------------------------------


def base_inputs(request) -> dict[str, dict]:
    """topology inputs that depend only on the request, not on the site tables.

    the radii are the ones `materialize_eeg_forward` argues for and are repeated
    rather than re-derived: 9 mm on the sheet is six length constants of
    `lateral_surface_transfer`'s 1.5 mm exponential, and 64 long-range partners
    per node is a sampling budget that the association builder records an
    inclusion probability for.  `local` and `microcircuit` take no support
    argument -- both are declared over the sheet and the volume and default to
    every one of those the materialization instantiated -- so a split neural
    field is handled without either being told about the split.
    """
    ti: dict[str, dict] = {
        "cortical_surface": {"radius_mm": 9.0, "k": 18},
        "cortical_association": {"max_degree": 64},
        "local": {},
        "microcircuit": {},
        "laminar": {},
        "mechanical": {},
        # the ascending systems have to be asked for one at a time -- the
        # topology is one declaration and the projections are five different
        # nuclei with different fibre velocities -- and cholinergic basal
        # forebrain is the one every model in this library that names
        # `neuromodulation` actually reads.  it still needs a brainstem and
        # basal-forebrain segmentation to find its sources, which this subject
        # does not have; naming the transmitter is what turns the resulting gap
        # from "nobody said which system" into "the atlas is absent", which is
        # the true one.
        "neuromodulatory_projection": {"transmitter": "acetylcholine"},
    }
    dev = next((d for d in request.devices if d.support == "sensor_array"), None)
    if dev is not None:
        # a scalp montage contacts the conductor at the skin; the builder's own
        # defaults describe an implanted array in parenchyma, which is a
        # different instrument.
        ti["device_coupling"] = {"device_support": "sensor_array",
                                 "medium_support": "head_volume",
                                 "reach_mm": 12.0,
                                 "direction": "stimulate" if dev.role == "stimulate" else "record"}
    return ti


def lead_field_for(model_id: str, sites, paths, montage_ids, kind: str):
    """a BEM solution over the source positions this build actually produced.

    a lead field is indexed by source position and cannot be reindexed onto a
    different one, so it can only be computed after the site table exists --
    which is why every model that needs one is built twice, the first time only
    to find out where its sources landed.  the sheet is preferred over the
    volume when both were materialized, for the reason `materialize_eeg_forward`
    gives: the scalp lead field of a deep source is orders of magnitude below a
    cortical one, the column nodes lie on the white surface by construction, and
    the subcortical block exists so the dynamics have somewhere to live rather
    than because the forward operator needs it.
    """
    src = "cortical_surface" if "cortical_surface" in sites.tables else (
        "tissue" if "tissue" in sites.tables else None)
    if src is None or "sensor_array" not in sites.tables:
        return None
    xyz = np.asarray(sites[src].xyz, float)
    sols = sorted(paths.bem_dir.glob("*-bem-sol.fif"))
    best = max(sols, key=lambda q: q.name.count("5120"))
    lf = geo.bem_lead_field(xyz, raw_fif=paths.raw_fif, trans_fif=paths.trans_fif,
                            bem_sol=best, sensor_ids=montage_ids, kind=kind)
    return {"lead_field": lf.data, "source_support": src}


# ---------------------------------------------------------------------------
# one model
# ---------------------------------------------------------------------------


class Row:
    __slots__ = ("id", "status", "reason", "sites", "variables", "coefficients",
                 "bytes", "edges", "factor", "strict", "gaps", "faults", "seconds",
                 "supports", "split", "out_of_view")

    def __init__(self, mid: str) -> None:
        self.id = mid
        self.status = "refused"
        self.reason = ""
        self.sites = 0
        self.variables = 0
        self.coefficients = 0
        self.bytes = 0
        self.edges: dict[str, int] = {}
        self.factor = "1x"
        self.strict = False
        self.gaps: list[str] = []
        self.faults: list[str] = []
        self.out_of_view: list[str] = []
        self.seconds = 0.0
        self.supports: dict[str, int] = {}
        self.split = 0


def is_gap(problem: str) -> bool:
    return any(m in problem for m in GAP_MARKERS)


def excluded_supports(request) -> frozenset[str]:
    """the supports R puts outside this view, as `build` itself decides it.

    recomputed with the build's own two functions rather than parsed out of its
    prose, because the whole point of the distinction is that it is the request's
    determination and not this script's opinion.  `_supports_to_materialize`
    already answers exactly the question -- which supports can carry allocated
    state, and why the others cannot -- and its answer is what separates "this
    subject has no vessel segmentation" from "this model has no vessels in it".
    """
    tr = _trace(request)
    support_of, _, _ = _place_components(request, tr)
    _, excluded = _supports_to_materialize(request, tr, support_of)
    return frozenset(excluded)


def primary_gap(gaps: Sequence[str], named: frozenset[str]) -> str:
    """which of a model's gaps is the one the row should name.

    the gaps come out in the order the build hit them, which is alphabetical by
    support, so a model stopped dead by a missing thalamic segmentation reports
    "it needs geometry['body']" -- true, irrelevant, and the wrong thing to put
    in a one-line summary.  the binding gap is ranked instead:

    1. a region or resolution rule that could not be evaluated.  this is the one
       that zeroes a model, because r(q) is evaluated per support during
       sampling and a rule naming an atlas the subject lacks takes the whole
       support down with it -- unlike R, which now degrades to an unrestricted
       build and a note.
    2. geometry for a support R actually names.  a bold model that wants a
       vascular tree is missing a venogram; one that merely traces to a retina
       is not missing anything it asked for.
    3. a topology input -- a tractogram, an atlas partition on a built support.
    4. everything else, which is the peripheral supports the trace reaches and
       R never wanted.
    """
    def rank(g: str) -> tuple[int, str]:
        if "'region_weights'" in g:
            return (0, g)
        i = g.find("geometry['")
        if i >= 0:
            return (1 if g[i + 10:].split("'", 1)[0] in named else 3, g)
        if "topology builder" in g:
            return (2, g)
        return (3, g)
    return one_line(min(gaps, key=rank)) if gaps else ""


def is_out_of_view(problem: str, excluded: frozenset[str]) -> bool:
    """is this gap a topology outside the view rather than an absent dataset?

    `_build_edges` skips a topology only when *none* of its declared supports was
    instantiated, which is the right rule for the topology as a whole and too
    weak for the builders: `electromagnetic` is declared over seven supports, so
    a resting-state materialization with a cortex and no sensor array passes that
    test and then fails inside `em_lead_field` asking for the sensor array.  the
    result reads exactly like a missing file and is nothing of the kind -- there
    is no instrument in the model, R said so, and no data would change it.

    so the discrimination is made on what the builder actually asked for.  a
    complaint naming `sites['X']` or `geometry['X']` is out of view when R
    excluded X, and a missing file when it did not: `metabolic_exchange` wanting
    the vascular tree is out of view for `resting_state_fc`, which never named
    one, and a genuine gap for `bold_forward`, which names it in R and this
    subject has no venogram.  a complaint about a non-support input --
    `streamlines`, relay `stages` -- is out of view when the topology it belongs
    to has any declared support that R excluded, which is what makes the afferent
    and efferent pathways (declared over the retina, the cochlea and the
    musculature) out of view for every model here, and leaves `tractometric`,
    declared over `tissue` alone, correctly reported as a missing tractogram.
    """
    for kind in ("sites['", "geometry['"):
        i = problem.find(kind)
        if i >= 0:
            s = problem[i + len(kind):].split("'", 1)[0]
            return s in excluded
    i = problem.find("topology builder '")
    if i >= 0:
        builder = problem[i + len("topology builder '"):].split("'", 1)[0]
        for t in REGISTRY.topologies.values():
            if t.builder == builder:
                return bool(set(t.on) & excluded)
    return False


def one_line(problem: str) -> str:
    """a problem's first line, which is the part that names the thing."""
    return problem.strip().splitlines()[0].strip()


def coarsen_volumes(request, factor: float):
    """the same request with only its volume rules coarsened.

    a rule is taken to govern the volume when the supports it names *by name* are
    volumes, and also when it names none at all: a `Near(...)` shell or an atlas
    label is a set of positions rather than a support, and in this library every
    such rule is written for a parenchyma or conductor octree.  a rule that names
    a surface support keeps its spacing, which is the whole point -- the sheet is
    where a coarsening is most expensive in accuracy and least effective in cost.

    the default clause is coarsened, because the default is what every support
    with no rule of its own falls through to, and those are volumes here.
    """
    vols = frozenset(n for n, s in REGISTRY.supports.items() if s.kind == "volume")
    r = request.resolution
    rules = []
    for x in r.rules:
        named = _named_supports(x.region)
        volumetric = (not named) or bool(named & vols)
        rules.append(ResolutionRule(x.region,
                                    x.spacing_mm * factor if volumetric else x.spacing_mm,
                                    x.band))
    return replace(request, name=f"{request.name}@{factor:g}x-vol",
                   resolution=Resolution(tuple(rules), r.default_mm * factor, r.default_band))


def ladder(request):
    """(label, request) for each rung, declared first, volumes next, all last."""
    yield "1x", request
    for f in VOLUME_COARSENINGS:
        yield f"{f:g}x-vol", coarsen_volumes(request, f)
    for f in COARSENINGS:
        yield f"{f:g}x", request.coarsened(f)


def attempt(request, **kw):
    """build at the declared r(q), coarsening only if the budget refuses.

    the budget is the one failure worth retrying automatically, because it is the
    failure §1 has an answer to: a coarser materialization is the same model
    until the coarse-graining stops commuting with the dynamics.  everything else
    -- a missing file, an overlapping placement -- is not made true or false by
    the spacing, and retrying it would only take longer to say the same thing.

    it is checked twice because it fails in two places.  `build_sites` raises
    `BudgetExceeded` from inside the octree, eagerly, so that a refinement past
    the ceiling does not discover it is out of memory by running out of memory;
    and `Cost.check` reports the same ceilings afterwards, where under
    `strict=False` they are notes rather than an exception.  a retry loop that
    watched only the first would coarsen the models that blow the site cap and
    silently return an unaffordable one for the models that blow the variable
    cap, which is the more common of the two.
    """
    last = ""
    for label, req in ladder(request):
        try:
            m = build(req, **kw)
        except BudgetExceeded as exc:
            last = str(exc)
            continue
        v = m.cost.check(req.budget)
        if not v:
            return m, label, req, ()
        last = "; ".join(v)
    # every rung refused.  return the coarsest attempt if it built at all, so the
    # row can still report what the model would have cost.
    try:
        req = request.coarsened(COARSENINGS[-1])
        return build(req, **kw), f"{COARSENINGS[-1]:g}x", req, tuple(last.split("; "))
    except BudgetExceeded:
        raise BudgetExceeded(last)


def run_model(mid: str, *, montages, warp, anchors, anatomy, paths, cache) -> Row:
    row = Row(mid)
    model = MODELS[mid]
    t0 = time.time()

    # the subject is the one thing every library request has to be told about:
    # they are declared against `template` or against an anonymous `patient`,
    # and this is a named individual with an individual reconstruction.
    subject = SubjectSpec(id="sample", frame=geo.ANATOMICAL_FRAME,
                          surface_frame=geo.ANATOMICAL_FRAME, template=None,
                          note="mne sample: individual T1, watershed BEM, individual coreg")
    # the sensor array is whichever instrument the request declares: this
    # recording carries both a 60-channel eeg montage and a 306-channel meg
    # helmet, they are digitised in different frames, and a meg model handed the
    # eeg positions is not a coarser meg model -- it is a different instrument.
    # the request's own device frame is what picks between them.
    dev = next((d for d in model.request.devices if d.support == "sensor_array"), None)
    kind = "meg" if dev is not None and dev.frame == "meg_head" else "eeg"
    geom, ids = montages[kind]
    devices = tuple(
        replace(d, n_elements=len(ids), element_ids=ids,
                positions=np.asarray(geom.get("sensor_array").xyz, float))
        if d.support == "sensor_array" else d
        for d in model.request.devices)
    request = replace(model.request, subject=subject, devices=devices)

    common = dict(geometry=geom, warp=warp, anchors=anchors, anatomy=anatomy,
                  cache=cache, strict=False)
    over: tuple[str, ...] = ()
    chosen = request
    try:
        m, label, chosen, over = attempt(request, topology_inputs=base_inputs(request),
                                         **common)
        row.factor = label
        # a second pass only where the electromagnetic topology has both ends.
        ti = base_inputs(request)
        if "electromagnetic" in m.trace.topologies:
            lf = lead_field_for(mid, m.sites, paths, ids, kind)
            if lf is not None:
                ti["electromagnetic"] = lf
                m = build(chosen, topology_inputs=ti, **common)
    except BudgetExceeded as exc:
        row.reason = f"budget: {one_line(str(exc))}"
        row.faults.append(str(exc))
        row.seconds = time.time() - t0
        return row
    except (MissingData, MissingInput) as exc:
        # a gap that escaped rather than being recorded: it still names the file
        # or the transform that was absent, so it is triaged as a gap and not as
        # a fault.  the row has no sites, which is the honest consequence.
        row.status = "missing-data"
        row.reason = one_line(str(exc))
        row.gaps.append(str(exc))
        row.seconds = time.time() - t0
        return row
    except Exception as exc:                                        # noqa: BLE001
        row.reason = f"{type(exc).__name__}: {one_line(str(exc))}"
        row.faults.append(f"{type(exc).__name__}: {exc}")
        row.seconds = time.time() - t0
        return row

    row.sites = sum(t.n for t in m.sites.tables.values())
    row.supports = {s: t.n for s, t in sorted(m.sites.tables.items())}
    row.variables = m.cost.state_variables
    row.coefficients = m.cost.spectral_coefficients
    row.bytes = m.cost.n_bytes
    row.edges = {n: (0 if m.edges.get(n) is None else m.edges[n].n_edges)
                 for n in sorted(set(m.trace.topologies))}
    row.split = len(m.layout.split)
    out_of_view = excluded_supports(chosen)
    for p in m.provenance.missing:
        if not is_gap(p):
            row.faults.append(p)
        elif is_out_of_view(p, out_of_view):
            row.out_of_view.append(p)
        else:
            row.gaps.append(p)
    row.faults.extend(over)

    if row.faults:
        row.status = "refused"
        row.reason = one_line(row.faults[0])
    elif row.gaps:
        row.status = "missing-data"
        row.reason = primary_gap(row.gaps, _named_supports(chosen.scope))
    else:
        row.status = "built"
        # the only honest way to fill the strict column is to run it.
        try:
            build(chosen, topology_inputs=ti, geometry=geom, warp=warp, anchors=anchors,
                  anatomy=anatomy, cache=cache, strict=True)
            row.strict = True
        except MaterializationIncomplete as exc:
            # the interesting half is the first problem, not the summary line:
            # `strict=True` promotes to problems exactly the things `strict=False`
            # keeps as notes -- a frame with no supplied warp, a budget note --
            # and naming which one is the whole content of this column.
            row.reason = f"strict refuses: {one_line(exc.problems[0]) if exc.problems else exc}"
        except Exception as exc:                                    # noqa: BLE001
            row.reason = f"strict refuses: {type(exc).__name__}: {one_line(str(exc))}"
    row.seconds = time.time() - t0
    return row


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    ibm.load_all(seal=True, strict=True)
    only = [a for a in argv if not a.startswith("-")]
    ids = only or sorted(MODELS)

    head("1. the subject, and everything it does not have")
    paths = geo.sample_paths()
    print(paths.describe())
    geometry = geo.sample_subject(paths=paths)
    coreg = geo.sample_coregistration(geometry=geometry, paths=paths)
    montage = geometry.get("sensor_array")
    white = np.asarray(geometry.get("cortical_surface").vertices, float)
    aparc = Aparc(paths.subject_dir)
    # two instruments, one head.  `sample_subject` puts the eeg montage on
    # `sensor_array`; the same recording also carries a 306-channel meg helmet,
    # digitised in `meg_head` rather than `eeg_cap`, and a meg model handed eeg
    # positions would be a forward solution for the wrong instrument.  so the
    # helmet is read separately and swapped onto the same support, and
    # `run_model` picks by the frame the request's own device declares.
    meg_montage, meg_ids = geo.from_montage(paths.raw_fif, kind="meg", frame="meg_head")
    montages = {
        "eeg": (geometry, tuple(montage.ids)),
        "meg": (geo.GeometrySet({**geometry.by_support, "sensor_array": meg_montage},
                                subject="sample"), tuple(meg_ids)),
    }
    print()
    print("  supports this reconstruction supplies:")
    for s, g in sorted(geometry.by_support.items()):
        print(f"    {s:18s} {type(g).__name__:17s} frame={g.frame:12s} {g.source[:70]}")
    absent = sorted(set(REGISTRY.supports) - set(geometry.by_support))
    print()
    print(f"  supports it does not, and no substitute is offered for ({len(absent)}):")
    print("    " + ", ".join(absent))
    print()
    print("  anatomy supplied to build(anatomy=...):")
    print(aparc.describe())
    print("    every other declared system is refused by name; see `Aparc`.")
    print()
    print("  landmarks supplied to build(anchors=...):")
    print(f"    cortex   {len(white):,} white-surface vertices, the sheet's own positions.")
    print("             `subcortex()` in the library carves the volume with "
          "Difference(tissue, Near('cortex', 6 mm)),")
    print("             which is what makes a two-support placement of the neural field a "
          "partition rather than")
    print("             a second copy of the cortex.")
    print("    lesion   NOT supplied.  this is a healthy subject and `virtual_lesion` says so "
          "by reporting the")
    print("             missing landmark rather than by silently refining nothing.")
    print()
    print("  instruments on `sensor_array`, one per request frame:")
    for k, (_, mids) in sorted(montages.items()):
        print(f"    {k:4s} {len(mids):>4} channels")
    print()
    print(coreg.describe())

    head("2. building")
    cache = Cache()
    warp = head_warp(coreg)
    anchors = {"cortex": white}
    rows: list[Row] = []
    for mid in ids:
        try:
            r = run_model(mid, montages=montages, warp=warp, anchors=anchors,
                          anatomy=aparc, paths=paths, cache=cache)
        except Exception:                                           # noqa: BLE001
            traceback.print_exc()
            r = Row(mid)
            r.reason = "the script itself failed; see traceback above"
        rows.append(r)
        print(f"  {r.id:26s} {r.status:13s} {r.sites:>8,} sites  {r.variables:>10,} vars  "
              f"{r.seconds:6.1f}s  {r.reason[:60]}")

    head("3. the table")
    hdr = ("model", "status", "r", "sites", "state vars", "spectral coef", "MiB",
           "edges", "strict", "why")
    def fmt(r: Row) -> tuple[str, ...]:
        return (r.id, r.status, r.factor, f"{r.sites:,}", f"{r.variables:,}",
                f"{r.coefficients:,}", f"{r.bytes / 2**20:,.1f}",
                f"{sum(r.edges.values()):,}", "yes" if r.strict else "-",
                r.reason[:74] or "-")
    body = [fmt(r) for r in rows]
    w = [max(len(h), *(len(b[i]) for b in body)) if body else len(h)
         for i, h in enumerate(hdr)]
    print("  " + "  ".join(h.ljust(x) for h, x in zip(hdr, w)))
    print("  " + "  ".join("-" * x for x in w))
    for b in body:
        print("  " + "  ".join(c.ljust(x) for c, x in zip(b, w)))

    head("4. edges per topology, and what each row is resting on")
    for r in rows:
        print()
        print(f"  {r.id}  [{r.status}]  {r.seconds:.1f}s"
              + (f"  built at {r.factor} of its declared r(q)" if r.factor != "1x" else ""))
        if r.supports:
            print("    sites      " + ", ".join(f"{s} {n:,}" for s, n in r.supports.items())
                  + (f"   ({r.split} component(s) split across supports)" if r.split else ""))
        if r.edges:
            built = {k: v for k, v in r.edges.items() if v}
            empty = [k for k, v in r.edges.items() if not v]
            print("    edges      " + (", ".join(f"{k} {v:,}" for k, v in built.items())
                                       or "none built"))
            if empty:
                print("    no edges   " + ", ".join(empty))
        for f in r.faults:
            print("    FAULT      " + one_line(f))
        for g in r.gaps:
            print("    gap        " + one_line(g))
        for g in r.out_of_view[:4]:
            print("    out-of-view " + one_line(g))
        if len(r.out_of_view) > 4:
            print(f"    out-of-view (+{len(r.out_of_view) - 4} more topologies R excludes)")

    head("5. counts")
    built = [r for r in rows if r.status == "built"]
    missing = [r for r in rows if r.status == "missing-data"]
    refused = [r for r in rows if r.status == "refused"]
    strict = [r for r in rows if r.strict]
    print(f"  built          {len(built):3d}   materialized with no gap and no fault")
    print(f"  missing-data   {len(missing):3d}   materialized, with data this subject lacks")
    print(f"  refused        {len(refused):3d}   did not materialize")
    print(f"  strict=True    {len(strict):3d}   would also build with strict=True")
    print()
    for name, group in (("built", built), ("missing-data", missing), ("refused", refused)):
        print(f"  {name:13s} " + (", ".join(r.id for r in group) or "(none)"))
    print()
    print(f"  coarsened      " + (", ".join(f"{r.id}@{r.factor}" for r in rows
                                            if r.factor != "1x") or "(none)"))
    print()
    print("  a `built` row has no gap and no fault.  `missing-data` means it materialized and")
    print("  something external was absent.  `refused` means it did not materialize, or did so")
    print("  with a fault -- a double-counted placement, a budget it could not reach even after")
    print("  8x coarsening, a topology with no builder.  gaps that are only a topology R put")
    print("  outside the view are counted as neither; they are listed per model as out-of-view.")

    head("6. what this subject is missing, aggregated")
    counts: dict[str, list[str]] = {}
    for r in rows:
        for g in r.gaps:
            # the "N traced component(s) name a support with no site table" line
            # differs between models only in N and in which six ids it lists
            # first, and grouping on the raw text turns one gap into fifteen
            # rows.  the count is per model and belongs on the left.
            line = one_line(g)
            if "traced component(s) name a support with no site table" in line:
                line = "traced components name a support with no site table (a "\
                       "consequence of the geometry gaps above)"
            counts.setdefault(line, []).append(r.id)
    for why, ms in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(ms):2d} model(s)  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
