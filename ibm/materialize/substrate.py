"""one structural substrate, in template space, that every materialization draws on.

ARCHITECTURE.md §7 says the forty named models are "materialized views of the same
implicit model rather than independently defined brain models".  until this module
existed they were not.  thirty-four of the forty could not be built against a real
head, and the reason was always the same shape: each one demanded a piece of
SUBJECT-SPECIFIC structure -- this person's tractogram, this person's vessel
segmentation, this person's thalamic nuclei, this person's implanted contacts --
and when the subject did not have it the build recorded a gap and moved on.  forty
models, forty independent demands on one subject's imaging, and no shared floor
underneath any of them.

that is wrong on the architecture's own terms, and §1 says why in one sentence:
"evidence constrains what it constrains; elsewhere the structure remains at its
prior, and is typically smooth or homogeneous.  that is the machinery working, not
failing."  a subject with no diffusion imaging does not thereby have no long-range
connectivity.  they have connectivity we have not measured, and the honest
representation of that is a prior -- a POPULATION prior, since a population is the
only thing we have measured instead.  the failure was never missing data.  it was a
missing rung.

what this module is
-------------------
one object, built once and cached, holding everything that is true of brains in
general rather than of one brain in particular, in a template frame, each piece
carrying the rung of its own ladder that it sits on:

    connectome    a published group consensus structural connectome over the
                  desikan parcels, scored through `tract_prior` at
                  GROUP_CONNECTOME -- somebody else's anatomy, and the module
                  that scores it has measured exactly what that costs (it
                  predicts a held-out subject's edges at AUC 0.82 and their
                  strengths at R^2 0.28).
    vascular      the major arterial tree, skeletonised from a population TOF-MRA
                  atlas at POPULATION_ATLAS, plus a coarse-grained capillary
                  layer whose density, calibre and segment length are the
                  MEASURED microvascular statistics at GENERATIVE_SYNTHESIS.
    microcircuit  the MICrONS connection statistics and laminar priors at
                  SPECIES_ATLAS -- a mouse's cortex, which is the tier and is
                  said out loud.
    partitions    the anatomical partitioning systems that are actually held:
                  desikan `cortical_areas` from a template `aparc+aseg`, and
                  `vascular_territories` from the johns hopkins arterial atlas.
    volumes       template parenchyma, csf and interstitial extents, for a
                  subject who has no segmentation of their own.

everything in it is population-level BY CONSTRUCTION.  that is the point of the
object and it is why the tier travels with every piece: a `TierRecord` is emitted
for each one, `build` records it in provenance, and `provenance.rests_on(...)`
reports it, so a reader can tell whether a prediction rests on this subject's
anatomy or on a population's without reading the code that built it.

what it refuses to hold
-----------------------
the refusals are the load-bearing part.  a substrate that answered every question
would be worse than no substrate, because a population average presented as an
anatomy is exactly the error the tier field exists to prevent.  so:

- **no brainstem, thalamic or hippocampal subnuclei.**  `brainstem_nuclei`
  declares 36 labels and `thalamic_nuclei` 31; the segmentations for them are not
  in the corpus, and a nearest-structure guess from `aseg`'s single `Brain-Stem`
  label would be a plausible-looking answer to a question the data cannot answer.
  eleven models still report a gap for the ascending neuromodulatory sources and
  they are RIGHT to.
- **no retina, cochlea, vestibular organ, viscera, musculature or body.**  no
  measured prior in this repository describes them, and inventing an eye would be
  inventing the afferent pathway's entire boundary condition.
- **no whole-brain capillary bed at capillary resolution.**  the measured
  penetrating-vessel spacing is 129 um, so a whole-brain bed at the density
  `vascular_prior.synthesise` reproduces is order 10^10 nodes.  what is held
  instead is one representative cubic-millimetre bed, synthesised and checked
  against the flow (it delivers 120 ml/100g/min at 45 mmHg against a measured 44),
  and a coarse-grained layer that carries that bed's measured wall area per unit
  volume rather than its geometry.  the coarse-graining is §1's own argument --
  a linear transport process does not distinguish them -- and the note says where
  it stops being true.
- **no venous side.**  the arterial atlas is arterial.  a venogram is a subject
  acquisition, not a population atlas, so the substrate has no veins and BOLD
  models that reach for one will find the arterial tree and a note saying so.

why one object rather than a helper per topology
------------------------------------------------
because the alternative is what was there before: each builder reaching for its
own fallback, in its own frame, with its own idea of what a template is, and
nothing able to say what the whole materialization was standing on.  one object in
one frame, built once, cached by content, and reported as one list of rungs, is
the difference between a model that knows it is resting on a population and forty
models that each quietly decided to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ibm.materialize.provenance import TierRecord
from ibm.materialize.sites import (
    Geometry, MissingData, SurfaceGeometry, TreeGeometry, VolumeGeometry,
)
from ibm.registry import REGISTRY
from ibm.topologies import builders as B
from ibm.topologies import microcircuit_prior as MP
from ibm.topologies import tract_prior as TP
from ibm.topologies import vascular_prior as VP

#: the frame everything in the substrate speaks.  one frame, chosen because the two
#: volumetric atlases held here are distributed in it and because `ibm.frames`
#: already declares a `subject_t1 -> mni152` chain, so a fall-through into a
#: subject's frame is a declared warp rather than an invented one.
TEMPLATE_FRAME = "mni152"

#: cards whose `.location.yaml` says where the bytes are.  named as constants so
#: the error message and the loader cannot drift apart, exactly as
#: `geometry.SAMPLE_CARD` is.
CONNECTOME_CARD = "hansen-many-networks"
ARTERY_CARD = "cerebral-artery-atlas-mouches2019"
TERRITORY_CARD = "arterial-territory-atlas-liu2023"
TEMPLATE_CARD = "mne-sample"          # ships fsaverage beside the sample subject


# ---------------------------------------------------------------------------
# the rung
# ---------------------------------------------------------------------------
#
# `TierRecord` itself lives in `provenance.py`, beside the other records a
# materialization keeps about its own trustworthiness, because that is where it
# gets read.  what lives here is only the sugar for making one out of the three
# prior modules' own `Tier` enums, which share a `.value` and a `.rank` and
# nothing else.


def _rung(what: str, tier: Any, ladder: str, source: str, cost: str, note: str = "") -> TierRecord:
    return TierRecord(what, ladder, tier.value, int(tier.rank), 3, source, cost, note)


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupConnectome:
    """a published parcel connectome, its parcel names, and its centroids.

    the lengths are the honest weakness and are flagged rather than smoothed over.
    `tract_prior.group_connectome` needs a mean streamline length per parcel pair
    and this publication does not report one, so what is carried is the CHORD
    between parcel centroids.  `association.py` argues that the chord is the right
    approximation for an association fibre, which leaves the sheet and crosses
    white matter -- but it is still a LOWER BOUND on arc length, and every
    conduction delay derived from it is therefore too short.  that is the direction
    that makes long-range coupling look faster than it is, so it is stated in the
    tier record's `cost` and not only here.
    """

    matrix: Any                       # (k, k) consensus streamline density
    lengths_mm: Any                   # (k, k) centroid chord, a lower bound on arc length
    labels: tuple[str, ...]           # k parcel names, already mapped onto `cortical_areas`
    centroids_mm: Any                 # (k, 3) in TEMPLATE_FRAME
    n_subjects: int = 0
    source: str = ""

    @property
    def k(self) -> int:
        return int(np.asarray(self.matrix).shape[0])

    def describe(self) -> str:
        A = np.asarray(self.matrix, float)
        return (f"{self.k}-parcel group consensus connectome, {float((A > 0).mean()):.0%} dense, "
                f"lengths are centroid chords ({float(self.lengths_mm[A > 0].mean()):.0f} mm mean)")


@dataclass(frozen=True)
class VascularSubstrate:
    """the template vascular tree: measured arteries, coarse-grained bed.

    two populations of node in one `parent` array, because they are one transport
    network and splitting them would mean two graphs claiming to be the same
    anatomy.  `is_capillary` says which is which, and the radii do the same job
    downstream: `capillary_tissue_exchange` selects on radius <= 5 um and will
    therefore exchange with the bed and not with the arteries, which is correct --
    an artery is muscular and impermeable.
    """

    xyz: Any
    parent: Any
    radius_mm: Any
    segment_length_mm: Any
    is_capillary: Any
    n_artery: int = 0
    n_capillary: int = 0
    #: the report from the one representative bed that was actually synthesised.
    #: it is what licenses the coarse-graining and it is what falsifies it.
    bed: Any = None
    source: str = ""

    def geometry(self, frame: str = TEMPLATE_FRAME) -> TreeGeometry:
        return TreeGeometry("vascular_tree", frame, self.xyz, self.parent, self.radius_mm,
                            source=self.source)

    def columns(self) -> dict:
        return {"segment_length_mm": self.segment_length_mm,
                "is_capillary": self.is_capillary}


@dataclass(frozen=True)
class LabelVolume:
    """one held partitioning system as a hard voxel labelling in a template frame.

    membership is 0/1 and not soft, and that is a property of the source rather
    than a simplification: both atlases held here are winner-take-all labellings
    with no per-voxel probability in them.  the region machinery accepts soft
    weights and would use them; there are none to use, and inventing a smooth
    boundary would be inventing an uncertainty estimate.
    """

    system: str
    frame: str
    data: Any                        # (nx, ny, nz) integer labels
    affine: Any                      # (4, 4) voxel index -> frame, mm
    ids: Mapping[str, tuple[int, ...]]
    source: str = ""

    def weights(self, label: str, xyz: Any) -> np.ndarray:
        ids = self.ids.get(label)
        p = np.asarray(xyz, float).reshape(-1, 3)
        if ids is None or not len(p):
            return np.zeros(len(p))
        inv = np.linalg.inv(np.asarray(self.affine, float))
        h = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ inv.T
        v = np.rint(h[:, :3]).astype(np.int64)
        shape = np.array(self.data.shape[:3])
        ok = np.all((v >= 0) & (v < shape), axis=1)
        val = np.zeros(len(p), np.int64)
        val[ok] = self.data[v[ok, 0], v[ok, 1], v[ok, 2]]
        return np.isin(val, np.asarray(ids, np.int64)).astype(float)


# ---------------------------------------------------------------------------
# the substrate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralSubstrate:
    """everything that is true of brains in general, in one frame, with its tiers.

    read-only and cheap to hold: the volumes are the expensive part and they are
    the same arrays every materialization would otherwise load for itself.  a
    build asks it three questions and no others --

        geometry_for(support)    a support's extent, when the subject has none
        edges_for(topology, ..)  a topology's edge set, when the subject's data is absent
        weights_for(system, ..)  a partition's membership, when no subject atlas has it

    -- and every one of them returns a `TierRecord` alongside the answer, so a
    caller cannot obtain the structure without also obtaining the answer to "where
    did this come from".  keeping those in separate calls is how a population
    average ends up quoted as anatomy: the array is what gets passed around, the
    provenance is what gets dropped.  `tract_prior.edge_support` makes exactly this
    argument about its own return type and this follows it.
    """

    frame: str = TEMPLATE_FRAME
    volumes: Mapping[str, VolumeGeometry] = _field(default_factory=dict)
    surface: SurfaceGeometry | None = None
    vascular: VascularSubstrate | None = None
    connectome: GroupConnectome | None = None
    partitions: Mapping[str, LabelVolume] = _field(default_factory=dict)
    rungs: tuple[TierRecord, ...] = ()
    absent: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    # -- lookup ----------------------------------------------------------

    def rung(self, what: str) -> TierRecord | None:
        for r in self.rungs:
            if r.what == what:
                return r
        return None

    @property
    def supports(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.volumes)
                            | ({"vascular_tree"} if self.vascular is not None else set())
                            | ({"cortical_surface"} if self.surface is not None else set())))

    @property
    def systems(self) -> tuple[str, ...]:
        return tuple(sorted(self.partitions))

    # -- the three questions a build asks --------------------------------

    def geometry_for(self, support: str) -> tuple[Geometry, TierRecord] | None:
        """a template extent for a support the subject has no segmentation of.

        `None` rather than an exception when nothing is held, because the caller's
        next move is to record the subject's own gap -- which already names the
        file and where such a thing comes from -- and replacing that message with
        one of ours would lose the part that is useful.
        """
        if support == "vascular_tree" and self.vascular is not None:
            r = self.rung("vascular_tree")
            return self.vascular.geometry(self.frame), r  # type: ignore[return-value]
        if support == "cortical_surface" and self.surface is not None:
            return self.surface, self.rung("cortical_surface")  # type: ignore[return-value]
        g = self.volumes.get(support)
        if g is None:
            return None
        return g, self.rung(support)                       # type: ignore[return-value]

    def weights_for(self, system: str, label: str, xyz: Any, frame: str
                    ) -> tuple[np.ndarray, TierRecord] | None:
        """membership in one partition of one held system, for positions already in `frame`.

        the frame check is a refusal and not a conversion.  the substrate speaks
        one frame and the caller is the only thing that owns a warp into it; a
        loader that silently accepted subject coordinates and looked them up in a
        template volume would be the silent misregistration `ibm.frames` exists to
        prevent, and it would look exactly like a working atlas.
        """
        lv = self.partitions.get(system)
        if lv is None or frame != lv.frame:
            return None
        r = self.rung(system)
        return lv.weights(label, xyz), r                   # type: ignore[return-value]

    def edges_for(self, topology: str, sites: Any, *, parcels: Any = None,
                  max_degree: int = 64, seed: int = 0, **kw
                  ) -> tuple[B.EdgeSet, TierRecord] | None:
        """a topology's edges, built from the substrate rather than from the subject.

        only the topologies the substrate genuinely has structure for.  a topology
        it does not -- `afferent_pathway`, `neuromodulatory_projection` -- returns
        `None` and keeps its gap, which is the whole discipline of the object: the
        set of questions it declines to answer is what makes its answers worth
        anything.
        """
        if topology == "tractometric":
            return self._tractometric(sites, parcels=parcels, max_degree=max_degree,
                                      seed=seed, **kw)
        return None

    # -- the tier-2 connectome, expanded and capped -----------------------

    def _tractometric(self, sites, *, parcels=None, max_degree: int = 64,
                      max_edges: int = 4_000_000, seed: int = 0,
                      support: str = "tissue", **kw):
        """the group connectome sampled onto this materialization's own sites.

        three things happen here that `tract_prior.group_connectome` does not do.
        the first two are budget decisions with permanent consequences and are
        recorded in the note the way `min_pipelines` and `min_subject_fraction`
        are; the third is a limitation of the corpus.

        **the 68 lateralised parcels are collapsed onto the 34 `cortical_areas`
        labels the registry declares.**  the corpus's parcel memberships are
        bilateral -- `aparc+aseg` gives `superiorfrontal` for both hemispheres and
        the registry declares one label -- so a left-right distinction has nowhere
        to land.  the collapse is a UNION over the four hemisphere blocks, which
        over-includes rather than under-includes, and that direction is the one
        `tract_prior` argues for at length: `T(i, j) = 0` is permanent and
        `theta_ij = 0` is not.  what it costs is that a homotopic callosal pair and
        an intrahemispheric pair become indistinguishable, and their lengths are
        averaged over the two.

        **each site keeps a uniform sample of at most `max_degree` partners.**
        `tractometric_matrix` expands every parcel pair to every site pair, and its
        own docstring says the quadratic cost is "the honest signal that a
        materialization far finer than the connectome is asking for more than the
        connectome has".  that signal is correct and it is not a reason to refuse:
        a 2 mm whole-brain tissue octree puts ten thousand sites in a single
        desikan parcel, so the full expansion is 10^8 edges and 800 MiB of index
        array before anything is scored.  what is built instead is a uniform sample
        of the SAME expansion -- every site keeps `max_degree` partners drawn
        without preference from the union of the sites in the parcels its own
        parcel connects to -- so no site is left with an empty long-range
        neighbourhood, which is the failure that would actually matter, and the
        per-edge inclusion probability is carried so a fit can divide by it.
        sampling uniformly rather than by strength is deliberate: keeping the
        strongest would be a second, unrecorded thresholding of a matrix whose
        publisher has already applied one.

        **every edge carries the population's reproducibility and not its own.**
        this publication reports a consensus matrix and not the fraction of its
        cohort that carried each pair, so `existence_prob` is the measured
        population average 0.32 for every edge alike -- which is exactly what
        `group_connectome` does in the same situation, and it says out loud that
        the publisher's own thresholding is invisible here and cannot be undone.
        """
        if self.connectome is None:
            return None
        s = B.as_sites(sites)
        t = s.get(support)
        if t is None or t.n == 0:
            return None
        lab = parcels
        if lab is None:
            lab = (t.partitions or {}).get("cortical_areas")
        if lab is None:
            return None
        lab = np.asarray(lab)
        if lab.ndim == 2:
            lab = np.argmax(np.where(lab > 0, lab, -1.0), axis=1)
        lab = np.asarray(lab, np.int64)

        A = np.asarray(self.connectome.matrix, float)
        L = np.asarray(self.connectome.lengths_mm, float)
        k = A.shape[0]
        members = [np.flatnonzero(lab == a) for a in range(k)]
        n_in = np.array([len(m) for m in members])
        # how many partner SITES each parcel has, which is what a uniform sample of
        # the full expansion has to be drawn from.
        reach = [np.flatnonzero((A[a] > 0.0) & (n_in > 0)) for a in range(k)]
        n_out = np.array([int(n_in[r].sum()) for r in reach])
        # a parcel only contributes if it has sites of its own AND at least one
        # parcel it connects to also has sites.  checking the two separately is
        # what produced an empty concatenation: a materialization can easily place
        # sites in three parcels none of which the connectome joins to each other.
        placed = int(sum(len(members[a]) for a in range(k) if n_out[a] > 0))
        if placed == 0:
            return B.empty(
                "tractometric", s.n_total,
                ("tract_length_mm", "conduction_delay_s", "distance_mm", "orientation",
                 "existence_prob", "inclusion_prob"), directed=True,
                note=(f"the group connectome has nothing to connect on {support!r}: "
                      f"{placed} of {t.n} sites fall in a cortical parcel.  a request that "
                      f"puts its cortex on the sheet and leaves the subcortical remainder on "
                      f"the volume has no cortico-cortical pairs HERE, and that is the "
                      f"placement rather than a missing connectome")), None

        # the degree cap, tightened if the whole sample would still be too large.
        # reported rather than silently applied: it is a support decision and
        # `T(i, j) = 0` is permanent.
        deg = int(max_degree)
        if placed * deg * 2 > int(max_edges):
            deg = max(int(max_edges) // max(2 * placed, 1), 1)
        rng = np.random.default_rng(int(seed))

        src_l, dst_l, len_l, inc_l = [], [], [], []
        for a in range(k):
            ra = members[a]
            if not len(ra) or n_out[a] == 0:
                continue
            partners = np.concatenate([members[b] for b in reach[a]])
            plabel = np.concatenate([np.full(len(members[b]), b) for b in reach[a]])
            take = min(deg, len(partners))
            pick = rng.integers(0, len(partners), size=(len(ra), take))
            src_l.append(np.repeat(ra, take))
            dst_l.append(partners[pick].ravel())
            len_l.append(L[a, plabel[pick].ravel()])
            inc_l.append(np.full(len(ra) * take, take / float(len(partners))))

        i = np.concatenate(src_l)
        j = np.concatenate(dst_l)
        length = np.concatenate(len_l)
        inc = np.concatenate(inc_l)
        keep = i != j
        i, j, length, inc = i[keep], j[keep], length[keep], inc[keep]
        # the same pair may be drawn twice, and two edges between one pair is two
        # parameters for one connection.
        key = i.astype(np.int64) * t.n + j.astype(np.int64)
        _, uniq = np.unique(key, return_index=True)
        i, j, length, inc = i[uniq], j[uniq], length[uniq], inc[uniq]

        xyz = np.asarray(t.xyz, float)
        chord = np.linalg.norm(xyz[i] - xyz[j], axis=1)
        u = B.unit(xyz[j] - xyz[i], np)
        delay = B.conduction_delay_s(length, 8.0, np)
        p_exist = np.full(len(i), float(TP.GROUP.edge_reproducibility))
        cat = lambda x: np.concatenate([x, x])
        edges = B.EdgeSet(
            "tractometric", np.concatenate([i, j]) + t.offset,
            np.concatenate([j, i]) + t.offset, s.n_total,
            {"tract_length_mm": cat(length), "conduction_delay_s": cat(delay),
             "distance_mm": cat(chord), "orientation": np.concatenate([u, -u]),
             "existence_prob": cat(p_exist), "inclusion_prob": cat(inc)},
            directed=True,
            note=(f"GROUP CONNECTOME over {k} parcels of cortical_areas, sampled onto "
                  f"{placed:,} of {t.n:,} {support!r} sites at {deg} partners each "
                  f"({len(i):,} distinct pairs).  the sample is UNIFORM over the full "
                  f"parcel-pair expansion and its inclusion probability is carried per edge; "
                  f"raising the cap adds edges rather than changing which ones are likely.  "
                  f"tract lengths are parcel-centroid chords, a LOWER BOUND on arc length, so "
                  f"the conduction delays here are too short.  no per-edge subject frequency "
                  f"was published, so every edge carries the population-average "
                  f"reproducibility {TP.GROUP.edge_reproducibility:.2f} and the publisher's own "
                  f"thresholding is invisible.  this is somebody else's anatomy: it explains "
                  f"{TP.GROUP.group_explains_r2:.0%} of a held-out subject's edge strengths and "
                  f"leaves a factor of x{math.exp(TP.GROUP.residual_log_sd):.1f}.  "
                  f"tier {TP.Tier.GROUP_CONNECTOME.value}"))
        return edges, self.rung("tractometric")

    def theta_prior(self, edges: B.EdgeSet, **kw) -> TP.TractPriorSet:
        """the coupling prior that goes with a tier-2 support, at tier 2.

        offered rather than folded into `edges_for` because §3 is explicit that a
        topology is a support for interaction and strength is process parameters.
        what matters is that the tier travels: at GROUP_CONNECTOME the spread is
        dominated by a term no number of pipelines touches -- what a group
        connectome measurably leaves on a subject it was not measured on -- and a
        caller who took the edges here and the prior at tier 1 would report a
        connectome five times more certain than the data supports.
        """
        return TP.theta_prior(edges, tier=TP.Tier.GROUP_CONNECTOME, **kw)

    # -- presentation ----------------------------------------------------

    def describe(self) -> str:
        lines = [f"structural substrate in {self.frame!r} -- population-level by construction"]
        for r in self.rungs:
            lines.append(f"  {r}")
        for s, g in sorted(self.volumes.items()):
            lines.append(f"  volume     {s:14s} {g.source[:88]}")
        if self.surface is not None:
            lines.append(f"  surface    {'cortical_surface':14s} {self.surface.source[:88]}")
        if self.vascular is not None:
            v = self.vascular
            lines.append(f"  tree       {'vascular_tree':14s} {v.n_artery:,} arterial + "
                         f"{v.n_capillary:,} coarse-grained capillary nodes")
            if v.bed is not None:
                lines.append(f"             representative bed: {v.bed.describe()[:110]}")
        if self.connectome is not None:
            lines.append(f"  connectome {'tractometric':14s} {self.connectome.describe()}")
        for s in self.systems:
            lv = self.partitions[s]
            lines.append(f"  partition  {s:14s} {len(lv.ids)} labels, {lv.source[:60]}")
        if self.absent:
            lines.append("  it does NOT hold, and refuses to invent:")
            lines += [f"    - {a}" for a in self.absent]
        for n in self.notes:
            lines.append(f"  ! {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def _root(card: str) -> Path:
    from ibm.forge.spectra import local_root
    return Path(local_root(card))


def _template_subject_dir() -> Path:
    """where `fsaverage` is, resolved from the card that owns the bytes.

    fsaverage ships inside the mne sample dataset, so the card that knows where
    those bytes are is the card that knows where this is.  the path is never
    written down here for the reason `geometry.sample_paths` gives: a hardcoded
    path works on the machine it was written on and reads nothing anywhere else,
    while reporting a result computed over an empty file list.
    """
    from ibm.materialize.geometry import sample_paths
    p = sample_paths().subject_dir.parent / "fsaverage"
    if not (p / "mri" / "aseg.mgz").is_file():
        raise MissingData(
            "substrate", f"{p}/mri/aseg.mgz",
            "a template FreeSurfer reconstruction to carry the population volumes and the "
            "template parcellation",
            f"the {TEMPLATE_CARD} card ships fsaverage beside the sample subject; this "
            "resolved to a directory with no aseg in it")
    return p


def _nifti(path: Path):
    try:
        import nibabel as nib
    except ImportError as exc:                                     # pragma: no cover
        raise MissingData("substrate", "nibabel",
                          "reading the NIfTI atlases the substrate is built from",
                          "pip install nibabel") from exc
    return nib.load(str(path))


#: MNI305 -> MNI152, the affine FreeSurfer publishes for exactly this conversion
#: (its `talairach.xfm` targets MNI305; every volumetric atlas held here is
#: distributed in MNI152).  a LITERATURE constant, and the only one in this module:
#: it is a fixed property of two template spaces rather than of any subject, so
#: there is nothing to measure it against on this machine.  applying it costs about
#: 1 mm and NOT applying it costs the same amount silently, which is worse.
MNI305_TO_MNI152 = np.array([
    [0.9975, -0.0073, 0.0176, -0.0429],
    [0.0146, 1.0009, -0.0024, 1.5496],
    [-0.0130, -0.0093, 0.9971, 1.1840],
    [0.0, 0.0, 0.0, 1.0],
])


def _to_mni152(xyz: Any) -> np.ndarray:
    p = np.asarray(xyz, float).reshape(-1, 3)
    h = np.concatenate([p, np.ones((len(p), 1))], axis=1)
    return (h @ MNI305_TO_MNI152.T)[:, :3]


def _aparc_colour_table(subject_dir: Path) -> dict[int, str]:
    """the desikan colour table: index -> parcel name, read rather than hardcoded.

    `aparc+aseg` codes cortex as 1000 + k on the left and 2000 + k on the right,
    where k indexes the parcellation's colour table, and a subject processed with a
    different atlas version shifts by one.  so the table is read, and it is read
    from the `.annot` that carries it -- fsaverage ships the annotation and not the
    detached `.ctab` that a per-subject `recon-all` writes, and hardcoding the
    thirty-four names in their FreeSurfer order here would be the same silent
    off-by-one wearing a different hat.
    """
    for hemi in ("lh", "rh"):
        p = subject_dir / "label" / f"{hemi}.aparc.annot"
        if p.is_file():
            try:
                from nibabel.freesurfer.io import read_annot
                _, _, names = read_annot(str(p))
                return {i: (n.decode() if isinstance(n, bytes) else str(n))
                        for i, n in enumerate(names)}
            except Exception:                                       # noqa: BLE001
                break
    ctab = subject_dir / "label" / "aparc.annot.ctab"
    out: dict[int, str] = {}
    if ctab.is_file():
        for line in ctab.read_text().splitlines():
            bits = line.split()
            if len(bits) >= 2 and bits[0].isdigit():
                out[int(bits[0])] = bits[1]
    return out


def _template_volumes(subject_dir: Path) -> tuple[dict[str, VolumeGeometry], SurfaceGeometry,
                                                  LabelVolume, np.ndarray]:
    """parenchyma, csf and interstitial extents plus the template sheet and parcellation.

    all four come out of one `recon-all` on the FreeSurfer average, so they are
    consistent with each other by construction -- which matters more here than the
    individual accuracy of any of them, because a materialization that took its
    tissue mask from one template and its parcellation from another would have a
    parcel boundary that does not follow a tissue boundary and no way to notice.

    `interstitial` shares the parenchyma's occupancy and that is not laziness: the
    interstitial space is interdigitated with the cells throughout the parenchyma
    and has the same extent at any resolution a materialization can afford.  what
    distinguishes it is its volume fraction and tortuosity, which are field
    properties rather than extent, and the `interstitial` support's own doc says so.
    """
    from ibm.materialize.geometry import (
        ASEG_CSF, ASEG_PARENCHYMA, _bounds, _label_occupancy, read_freesurfer_surface, read_mgh,
    )

    aseg = read_mgh(subject_dir / "mri" / "aseg.mgz")
    # fsaverage's surface RAS is MNI305; every volume here is put into MNI152 by
    # composing the published affine onto the voxel-to-RAS transform, so the atlas
    # volumes and the template volumes index the same millimetres.
    aff = MNI305_TO_MNI152 @ np.asarray(aseg.vox2ras_tkr, float)

    out: dict[str, VolumeGeometry] = {}
    par = _label_occupancy(aseg.data, ASEG_PARENCHYMA)
    csf = _label_occupancy(aseg.data, ASEG_CSF)
    src = f"fsaverage aseg.mgz -> {TEMPLATE_FRAME}"
    out["tissue"] = VolumeGeometry(
        "tissue", TEMPLATE_FRAME, _bounds(par, aff), par, aff,
        source=f"TEMPLATE: {src}, {int((par > 0).sum()):,} parenchyma voxels")
    out["interstitial"] = VolumeGeometry(
        "interstitial", TEMPLATE_FRAME, _bounds(par, aff), par, aff,
        source=f"TEMPLATE: {src}, the parenchyma's own extent -- the interstitial space is "
               "interdigitated with it and has no separate boundary")
    out["csf_space"] = VolumeGeometry(
        "csf_space", TEMPLATE_FRAME, _bounds(csf, aff), csf, aff,
        source=f"TEMPLATE: {src}, {int((csf > 0).sum()):,} ventricular and cisternal voxels")

    verts, faces, n = [], [], 0
    for hemi in ("lh", "rh"):
        v, f = read_freesurfer_surface(subject_dir / "surf" / f"{hemi}.white")
        verts.append(_to_mni152(v))
        faces.append(f + n)
        n += len(v)
    sheet = SurfaceGeometry(
        "cortical_surface", TEMPLATE_FRAME, np.concatenate(verts), np.concatenate(faces),
        source=f"TEMPLATE: fsaverage lh.white + rh.white ({n:,} vertices) -> {TEMPLATE_FRAME}")

    aparc = read_mgh(subject_dir / "mri" / "aparc+aseg.mgz")
    names = _aparc_colour_table(subject_dir)
    declared = set(REGISTRY.anatomies["cortical_areas"].labels)
    ids: dict[str, tuple[int, ...]] = {}
    for k, name in names.items():
        if name in declared:
            ids[name] = ids.get(name, ()) + (1000 + k, 2000 + k)
    areas = LabelVolume(
        "cortical_areas", TEMPLATE_FRAME, np.asarray(aparc.data),
        MNI305_TO_MNI152 @ np.asarray(aparc.vox2ras_tkr, float), ids,
        source=f"TEMPLATE: fsaverage aparc+aseg.mgz, {len(ids)} of {len(declared)} declared labels")
    return out, sheet, areas, np.asarray(_bounds(par, aff), float)


#: the ICBM152 1 mm grid's voxel-to-world affine.  the arterial territory atlas is
#: distributed on that grid with its origin field zeroed -- `sform_code` is 2 and
#: the translation is (0, 0, 0), so voxel (0, 0, 0) claims to be the AC -- and a
#: volume looked up through that affine returns the wrong label everywhere while
#: looking like a working atlas.  reconstructing it is not a guess: 181x217x181 at
#: 1 mm is one grid, its origin is one voxel, and the reconstruction is CHECKED
#: below against the atlas's own left/right labels rather than asserted.
ICBM152_1MM = np.array([[-1.0, 0.0, 0.0, 90.0],
                        [0.0, 1.0, 0.0, -126.0],
                        [0.0, 0.0, 1.0, -72.0],
                        [0.0, 0.0, 0.0, 1.0]])

#: the atlas's level-1 intensity -> the label `ibm.anatomy.systems` declares.
#: written out rather than name-matched, because the atlas's own label file spells
#: several of them differently and repeats two, and a fuzzy match would silently
#: drop a territory instead of failing.  left and right intensities collapse onto
#: one declared label because `vascular_territories` is not lateralised.
_TERRITORY_IDS: dict[str, tuple[int, ...]] = {
    "aca": (1, 2),
    "lenticulostriate_medial": (3, 4),
    "lenticulostriate_lateral": (5, 6),
    "mca": (7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
    "pca": (17, 18, 19, 20),
    "thalamoperforating": (21, 22),
    "anterior_choroidal": (23, 24),
    "sca": (27, 28),
    # basilar plus "inferior cerebellar", which this atlas does not split into
    # AICA and PICA -- so those two declared labels stay absent rather than both
    # being handed the same voxels.
    "vertebrobasilar": (25, 26, 29, 30),
}


def _territories() -> LabelVolume | None:
    """`vascular_territories` from the johns hopkins arterial atlas, if it is held.

    nine of the fourteen declared labels are in it.  the other five -- AICA and
    PICA, which this atlas merges into one "inferior cerebellar" label, and the
    three watershed zones, which it does not draw at all -- are NOT supplied, and a
    request naming one still gets a gap.  that matters more than the nine: a
    watershed is where two territories fail to overlap, an infarct model asks for
    it precisely because it is the vulnerable tissue, and deriving one here by
    dilating two neighbouring labels until they touch would produce a plausible
    band in a place no measurement put it.
    """
    try:
        root = _root(TERRITORY_CARD)
    except Exception:
        return None
    p = root / "data" / "Atlas" / "ArterialAtlas.nii"
    if not p.is_file():
        return None
    data = np.squeeze(np.asarray(_nifti(p).dataobj)).astype(np.int16)
    if data.shape != (181, 217, 181):
        return None
    declared = set(REGISTRY.anatomies["vascular_territories"].labels)
    ids = {k: v for k, v in _TERRITORY_IDS.items() if k in declared}

    # the check the reconstructed origin has to pass: every label in this atlas is
    # a left/right pair, so the left member must sit on one side of x = 0 and the
    # right member on the other.  if the grid were misplaced they would straddle it
    # together and this refuses rather than returning a shifted atlas.
    ii = np.argwhere(data > 0)
    x = (np.c_[ii.astype(float), np.ones(len(ii))] @ ICBM152_1MM.T)[:, 0]
    v = data[ii[:, 0], ii[:, 1], ii[:, 2]]
    left = np.mean(x[np.isin(v, (1, 7, 9, 11))])          # ACAL, MCAFL, MCAPL, MCATL
    right = np.mean(x[np.isin(v, (2, 8, 10, 12))])        # their right-hand partners
    if not (left < -5.0 < 5.0 < right):
        return None
    return LabelVolume(
        "vascular_territories", TEMPLATE_FRAME, data, ICBM152_1MM, ids,
        source=f"POPULATION ATLAS: {TERRITORY_CARD}/Atlas/ArterialAtlas.nii on the ICBM152 1 mm "
               f"grid (the file's own origin is zeroed; reconstructed and checked, left "
               f"hemisphere centroid {left:.0f} mm vs right {right:.0f} mm), {len(ids)} of "
               f"{len(declared)} declared territories")


def _artery_tree(prob_pct: float = 25.0, bin_mm: float = 1.5
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
    """a centreline tree for the major arteries, from the population TOF-MRA atlas.

    the atlas is a probability volume and a radius volume, not a graph, so a graph
    has to be recovered.  the recovery is deliberately the crudest one that is
    still a tree, and each step is chosen so that being wrong about it is visible
    rather than plausible:

    1.  **threshold at `prob_pct` and bin at `bin_mm`.**  the atlas is 0.5 mm and a
        voxelwise graph of a 2 mm artery zig-zags across its own lumen; binning to
        1.5 mm and taking each cell's centroid is a centreline in the only sense
        this data supports.  the radius is the cell's MAXIMUM, not its mean,
        because a mean over a partially-filled boundary cell reports a vessel
        narrower than the one that is there, and resistance goes as the fourth
        power of that.
    2.  **a minimum spanning forest over neighbours within 3 mm.**  an MST is the
        shortest set of connections that leaves the vessel set connected; it is
        not a claim that blood follows those exact segments, and where the atlas
        has two vessels crossing it will join them.  the alternative -- inferring
        bifurcation order from calibre -- would produce a prettier tree with no
        more information in it.
    3.  **root each component at its widest node.**  the internal carotids and the
        basilar are the widest things in the atlas, so this puts the roots where
        the blood enters without needing a named landmark.

    what comes out is arterial and nothing else.  the atlas is a TOF-MRA average,
    TOF is flow-weighted and sees arteries, and the venous side is a subject
    acquisition rather than a population atlas.  a materialization that needs veins
    will not find them here and will find a note saying so.
    """
    try:
        root = _root(ARTERY_CARD)
    except Exception:
        return None
    pp, rp = root / "vesselProbabilities.nii.gz", root / "vesselRadius.nii.gz"
    if not (pp.is_file() and rp.is_file()):
        return None
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
    from scipy.spatial import cKDTree

    im = _nifti(pp)
    prob = np.asarray(im.dataobj, dtype=np.float32)
    rad = np.asarray(_nifti(rp).dataobj, dtype=np.float32)
    aff = np.asarray(im.affine, float)
    vox = np.argwhere(prob > float(prob_pct))
    if not len(vox):
        return None
    h = np.concatenate([vox.astype(float), np.ones((len(vox), 1))], axis=1)
    xyz = (h @ aff.T)[:, :3]
    r = rad[vox[:, 0], vox[:, 1], vox[:, 2]].astype(float)

    cell = np.floor(xyz / float(bin_mm)).astype(np.int64)
    _, inv = np.unique(cell, axis=0, return_inverse=True)
    k = int(inv.max()) + 1
    cx = np.zeros((k, 3))
    np.add.at(cx, inv, xyz)
    cnt = np.bincount(inv, minlength=k).astype(float)
    cx /= cnt[:, None]
    cr = np.zeros(k)
    np.maximum.at(cr, inv, r)

    tree = cKDTree(cx)
    pairs = np.asarray(sorted(tree.query_pairs(r=2.0 * float(bin_mm))), dtype=np.int64)
    if not len(pairs):
        return None
    w = np.linalg.norm(cx[pairs[:, 0]] - cx[pairs[:, 1]], axis=1)
    g = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(k, k))
    mst = minimum_spanning_tree(g).tocoo()
    sym = coo_matrix((np.concatenate([mst.data, mst.data]),
                      (np.concatenate([mst.row, mst.col]),
                       np.concatenate([mst.col, mst.row]))), shape=(k, k)).tocsr()
    ncomp, comp = connected_components(sym, directed=False)

    parent = np.full(k, -1, np.int64)
    indptr, indices = sym.indptr, sym.indices
    for c in range(ncomp):
        members = np.flatnonzero(comp == c)
        root_i = int(members[np.argmax(cr[members])])
        seen = np.zeros(k, bool)
        seen[root_i] = True
        stack = [root_i]
        while stack:
            u = stack.pop()
            for v in indices[indptr[u]:indptr[u + 1]]:
                if not seen[v]:
                    seen[v] = True
                    parent[v] = u
                    stack.append(int(v))
    src = (f"POPULATION ATLAS: {ARTERY_CARD} vesselProbabilities>{prob_pct:g}% binned at "
           f"{bin_mm:g} mm, {k:,} nodes in {ncomp} component(s), minimum spanning forest rooted "
           f"at each component's widest node; ARTERIAL ONLY -- TOF is flow-weighted and the "
           f"venous side is a subject acquisition, not a population atlas")
    return cx, parent, cr, src


def _capillary_layer(tissue: VolumeGeometry, artery_xyz: np.ndarray, spacing_mm: float,
                     stats: VP.MicrovascularStatistics
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """one coarse-grained capillary node per parenchyma cell, carrying the bed's wall area.

    the argument for this existing is arithmetic.  the measured penetrating-vessel
    spacing is 129 um, so a whole-brain bed at the density `synthesise` reproduces
    is of order 10^10 nodes; no materialization will ever hold one, and a `tissue`
    support with no capillaries at all is a brain with no perfusion, which is worse
    than a coarse one.

    the argument for it being legitimate is §1's: coarse-graining commutes with the
    dynamics where the state is smooth and the processes acting on it are linear.
    oxygen and glucose delivery into a cubic millimetre of cortex is both.  so each
    node stands for the whole bed of its cell and carries, exactly, that bed's
    measured properties -- the median capillary RADIUS (so exchange selects it and
    arteries are excluded, as they should be) and a `segment_length_mm` equal to
    the cell's TOTAL capillary length, which is the measured length density times
    the cell volume.  the quantity `capillary_tissue_exchange` computes from those
    two, 2 pi r L, is then the cell's true capillary wall area rather than one
    segment's.

    where it stops being true is stated rather than hidden: this carries no
    within-cell transit time, no arteriole-to-venule pressure gradient and no
    heterogeneity of flow across the bed, and a process whose nonlinearity lives at
    the capillary scale -- oxygen extraction at low saturation, capillary transit
    time heterogeneity -- is outside the regime where the coarse-graining commutes.
    the representative bed carried alongside is what such a process should be run
    against instead.
    """
    lo, hi = np.asarray(tissue.bounds_mm, float)
    n = np.maximum(np.ceil((hi - lo) / spacing_mm).astype(int), 1)
    grids = [lo[d] + (np.arange(n[d]) + 0.5) * spacing_mm for d in range(3)]
    pts = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
    occ = tissue.occupied(pts)
    pts = pts[occ > 0.5]
    if not len(pts):
        return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0), ""

    # every coarse node is a ROOT, and that is a refusal rather than an oversight.
    # the obvious move is to hang each cell's bed off its nearest atlas artery, and
    # it is wrong twice over: the nearest artery is tens of millimetres away, so the
    # segment carries a Poiseuille resistance for a vessel that does not exist, and
    # the path from a pial artery to a cortical bed runs through penetrating
    # arterioles at a calibre this atlas cannot see.  a tree with a hole in it is
    # honest about the hole; a tree with an invented trunk is not, and the invented
    # trunk would then dominate the flow solve.
    from scipy.spatial import cKDTree
    d_art, _ = cKDTree(artery_xyz).query(pts, k=1)
    nearest = np.full(len(pts), -1, np.int64)

    density = float(np.mean(stats.density_at_depth(np.linspace(0.0, 1.0, 51), np)))
    cell_volume = spacing_mm ** 3
    total_length = density * cell_volume
    r = float(stats.capillary_radius_mm)
    src = (f"GENERATIVE SYNTHESIS: one coarse-grained node per {spacing_mm:g} mm parenchyma "
           f"cell, radius {r * 1e3:.2f} um and segment length {total_length:.0f} mm -- the "
           f"cell's TOTAL capillary length at the measured {density:.0f} mm/mm^3, so the wall "
           f"area it carries is the cell's and not one segment's.  each is a ROOT: the nearest "
           f"atlas artery is a median {float(np.median(d_art)):.0f} mm away and the arterioles "
           f"between them are below what a TOF-MRA average resolves, so the tree has a stated "
           f"hole where it would otherwise have an invented trunk.  no within-cell transit, "
           f"pressure gradient or flow heterogeneity")
    return pts, np.asarray(nearest, np.int64), np.full(len(pts), total_length), src


def _vascular(tissue: VolumeGeometry, *, capillary_spacing_mm: float = 4.0,
              bed_mm: float = 2.0, seed: int = 0) -> tuple[VascularSubstrate | None, list[TierRecord],
                                                           list[str]]:
    """the template vascular tree and the representative bed that licenses it."""
    rungs: list[TierRecord] = []
    absent: list[str] = []
    art = _artery_tree()
    if art is None:
        absent.append(f"the arterial centreline tree: the {ARTERY_CARD} card has no local bytes, "
                      "so nothing here can place a vessel")
        return None, rungs, absent
    axyz, aparent, arad, asrc = art

    # one representative bed, actually synthesised and actually checked.  it is the
    # only thing that entitles the coarse-graining below to claim it carries the
    # measured statistics, and its `feasible` bit is the falsification: a bed that
    # needs several hundred mmHg is wrong however well its histograms match.
    bed = None
    try:
        box = np.array([[0.0, 0.0, 0.0], [bed_mm, bed_mm, bed_mm]])
        bed = VP.synthesise(box_mm=box, tier=VP.Tier.GENERATIVE_SYNTHESIS, seed=seed).report
    except Exception as exc:                                        # noqa: BLE001
        absent.append(f"a representative synthesised bed: {type(exc).__name__}: {exc}")

    cxyz, cparent, clen, csrc = _capillary_layer(tissue, axyz, capillary_spacing_mm, VP.MEASURED)
    n_a = len(axyz)
    xyz = np.vstack([axyz, cxyz]) if len(cxyz) else axyz
    # the capillary nodes' parents index into the arterial block, which sits first
    # in the concatenation, so they need no offset.  stated rather than left
    # implicit because getting it wrong hangs the bed off the wrong vessels and
    # produces a tree that looks entirely reasonable.
    parent = np.concatenate([aparent, cparent]) if len(cxyz) else aparent
    radius = np.concatenate([arad, np.full(len(cxyz), float(VP.MEASURED.capillary_radius_mm))]) \
        if len(cxyz) else arad
    seg = np.concatenate([np.zeros(n_a), clen]) if len(cxyz) else np.zeros(n_a)
    is_cap = np.concatenate([np.zeros(n_a, bool), np.ones(len(cxyz), bool)]) if len(cxyz) \
        else np.zeros(n_a, bool)

    rungs.append(_rung(
        "vascular_tree", VP.Tier.POPULATION_ATLAS, "vascular_prior.Tier", asrc,
        "somebody else's arteries, warped: it places the major trunks to within the atlas's "
        "own between-subject scatter and says nothing about this person's branching pattern, "
        "which is where a watershed territory actually varies",
        note=csrc))
    if bed is not None:
        rungs.append(_rung(
            "vascular_tree.bed", VP.Tier.GENERATIVE_SYNTHESIS, "vascular_prior.Tier",
            f"{bed_mm:g} mm^3 representative bed",
            f"none of the microvasculature is measured at any tier; this one is consistent "
            f"with the flow -- {bed.describe()[:160]}"))
    return VascularSubstrate(xyz, parent, radius, seg, is_cap, n_a, len(cxyz), bed,
                             asrc + "; " + csrc), rungs, absent


def _connectome() -> tuple[GroupConnectome | None, TierRecord | None, list[str]]:
    """the group consensus structural connectome, collapsed onto the declared labels."""
    absent: list[str] = []
    try:
        root = _root(CONNECTOME_CARD)
    except Exception:
        return None, None, [f"a group connectome: the {CONNECTOME_CARD} card has no local bytes"]
    m = root / "data" / "Cammoun033" / "consensusSC_wei.npy"
    coords = root / "data" / "parcellation_files" / "Cammoun033_coords.txt"
    if not (m.is_file() and coords.is_file()):
        return None, None, [f"a group connectome: {m} is not there"]

    A68 = np.load(m).astype(float)
    rows = [ln.split() for ln in coords.read_text().splitlines() if ln.strip()]
    # column 0 is the hemisphere and is dropped: the registry declares 34
    # bilateral labels, so there is nowhere for a left/right distinction to land.
    name = [r[1] for r in rows]
    xyz68 = np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows])
    declared = list(REGISTRY.anatomies["cortical_areas"].labels)
    index = {lab: i for i, lab in enumerate(declared)}
    col = np.array([index.get(n, -1) for n in name], np.int64)
    if (col < 0).any():
        absent.append(f"{int((col < 0).sum())} of {len(col)} published parcels have no declared "
                      f"`cortical_areas` label and were dropped: "
                      f"{sorted({n for n, c in zip(name, col) if c < 0})}")
    keep = col >= 0
    k = len(declared)
    A = np.zeros((k, k))
    L = np.zeros((k, k))
    W = np.zeros((k, k))
    ii = np.flatnonzero(keep)
    for a in ii:
        for b in ii:
            if a == b:
                continue
            v = A68[a, b]
            if v <= 0.0:
                continue
            ca, cb = int(col[a]), int(col[b])
            A[ca, cb] += v
            d = float(np.linalg.norm(xyz68[a] - xyz68[b]))
            L[ca, cb] += d * v
            W[ca, cb] += v
    L = np.where(W > 0, L / np.maximum(W, 1e-12), 0.0)
    cent = np.zeros((k, 3))
    n_of = np.zeros(k)
    for a in ii:
        cent[int(col[a])] += xyz68[a]
        n_of[int(col[a])] += 1
    cent = cent / np.maximum(n_of, 1)[:, None]

    gc = GroupConnectome(
        A, L, tuple(declared), cent, n_subjects=0,
        source=f"{CONNECTOME_CARD}: Cammoun033 consensusSC_wei, 68 lateralised parcels unioned "
               f"onto the {k} declared bilateral `cortical_areas` labels")
    rung = _rung(
        "tractometric", TP.Tier.GROUP_CONNECTOME, "tract_prior.Tier", gc.source,
        f"a published group connectome predicts a held-out subject's edges at AUC "
        f"{TP.GROUP.existence_auc:.2f} and their strengths at R^2 {TP.GROUP.group_explains_r2:.2f}, "
        f"leaving a factor of x{math.exp(TP.GROUP.residual_log_sd):.1f} per edge, so a result "
        f"resting on it is reportable as a POPULATION's connectivity and never as this "
        f"subject's",
        note="tract lengths are parcel-centroid chords, which are a LOWER BOUND on arc length; "
             "every conduction delay derived from them is therefore too short, which is the "
             "direction that makes long-range coupling look faster than it is")
    return gc, rung, absent


#: what the substrate is asked for and refuses, in the words a reader needs.
#: kept as data rather than prose so that `describe()` and a provenance report say
#: the same thing, and so that adding a source is a one-line deletion here.
REFUSALS = (
    "brainstem_nuclei (36 declared labels): no segmentation in the corpus.  `aseg` has one "
    "`Brain-Stem` label and the ascending neuromodulatory sources are not separable from it, "
    "so every model that needs locus coeruleus, raphe or VTA keeps its gap",
    "thalamic_nuclei (31 declared labels): the same, from one `Thalamus-Proper`",
    "hippocampal_subfields: the same, from one `Hippocampus`",
    "retina, cochlea, vestibular_organ, chemosensory_epithelium, viscera, body, body_surface, "
    "motor_units: no measured prior in this repository describes any of them, and an invented "
    "eye is an invented boundary condition for the whole afferent pathway",
    "implanted_array, stimulator, display, scanner_element: an instrument is where it is.  "
    "there is no population prior over where a surgeon put an electrode",
    "the venous vasculature: the arterial atlas is arterial, and a venogram is a subject "
    "acquisition rather than a population atlas",
    "a whole-brain capillary bed at capillary resolution: order 10^10 nodes at the measured "
    "129 um penetrating spacing.  what is held instead is one representative bed plus a "
    "coarse-grained layer carrying its measured wall area per unit volume",
)


def build_substrate(*, capillary_spacing_mm: float = 4.0, bed_mm: float = 2.0,
                    seed: int = 0) -> StructuralSubstrate:
    """assemble the substrate from the cards that are actually held.

    every piece is optional and its absence is recorded rather than raised, for
    the reason `_sample_sites` gives about supports: a corpus missing the arterial
    atlas should lose its vascular tree and nothing else, and an all-or-nothing
    loader turns one absent card into a substrate with nothing in it while
    reporting exactly one problem.
    """
    subject_dir = _template_subject_dir()
    volumes, sheet, areas, _bounds_mm = _template_volumes(subject_dir)
    rungs: list[TierRecord] = []
    absent: list[str] = list(REFUSALS)
    notes: list[str] = []

    for s in sorted(volumes):
        rungs.append(TierRecord(
            s, "template_geometry", "template", 1, 2, volumes[s].source,
            "a template stands in for this subject's segmentation: every position carries a "
            "systematic, head-size-dependent displacement rather than a random one"))
    rungs.append(TierRecord(
        "cortical_surface", "template_geometry", "template", 1, 2, sheet.source,
        "a template sheet has this population's folding and not this subject's; the surface "
        "registration residual is folding-driven and concentrates in tertiary sulci"))
    rungs.append(TierRecord(
        "cortical_areas", "template_atlas", "template", 1, 2, areas.source,
        "desikan parcels on a template: the parcel boundaries follow the average folding, so "
        "a subject whose sulcus runs elsewhere has the wrong label at the boundary"))

    partitions: dict[str, LabelVolume] = {"cortical_areas": areas}
    terr = _territories()
    if terr is not None:
        partitions["vascular_territories"] = terr
        rungs.append(TierRecord(
            "vascular_territories", "population_atlas", "population_atlas", 1, 2, terr.source,
            "arterial territories are a population's; the watershed zones between them are "
            "exactly where individuals differ most, which is also where an infarct model most "
            "needs them to be right"))
        missing = sorted(set(REGISTRY.anatomies["vascular_territories"].labels) - set(terr.ids))
        if missing:
            notes.append("the arterial territory atlas does not distinguish "
                         + ", ".join(missing) + "; those labels keep their gap")
    else:
        absent.append(f"vascular_territories: the {TERRITORY_CARD} card has no local bytes")

    vasc, vrungs, vabsent = _vascular(volumes["tissue"], capillary_spacing_mm=capillary_spacing_mm,
                                      bed_mm=bed_mm, seed=seed)
    rungs.extend(vrungs)
    absent.extend(vabsent)

    conn, crung, cabsent = _connectome()
    if crung is not None:
        rungs.append(crung)
    absent.extend(cabsent)

    rungs.append(TierRecord(
        "microcircuit", "microcircuit_prior.Tier", MP.Tier.SPECIES_ATLAS.value,
        int(MP.Tier.SPECIES_ATLAS.rank), 3, MP.MEASURED.source,
        "a mouse's cortex.  the connection probabilities, the laminar composition and the "
        "gain ratios are measured, and they are measured in the wrong species: human L2/3 is "
        "thicker, its pyramidal cells are larger, and nothing here resolves what that does to "
        "the numbers"))
    notes.append(MP.MEASURED.describe().splitlines()[0])

    return StructuralSubstrate(
        frame=TEMPLATE_FRAME, volumes=volumes, surface=sheet, vascular=vasc, connectome=conn,
        partitions=partitions, rungs=tuple(rungs), absent=tuple(absent), notes=tuple(notes))


def load(cache: Any = None, *, capillary_spacing_mm: float = 4.0, bed_mm: float = 2.0,
         seed: int = 0) -> StructuralSubstrate:
    """the substrate, built once per spec and cached on disk.

    this is the call `build` makes.  the arithmetic behind the object -- a minimum
    spanning forest over 60,000 atlas voxels, a synthesised bed, three template
    volumes -- takes the better part of a minute and is COMPLETELY determined by
    the cards on disk and the three parameters below, which is exactly the
    situation `cache.py` says its key should be a hash of.  forty models in one
    sweep would otherwise pay for it forty times.
    """
    spec = {"substrate": 3, "frame": TEMPLATE_FRAME,
            "capillary_spacing_mm": float(capillary_spacing_mm), "bed_mm": float(bed_mm),
            "seed": int(seed),
            "cards": [CONNECTOME_CARD, ARTERY_CARD, TERRITORY_CARD, TEMPLATE_CARD]}
    make = lambda: build_substrate(capillary_spacing_mm=capillary_spacing_mm, bed_mm=bed_mm,
                                   seed=seed)
    if cache is None:
        return make()
    return cache.memo("substrate", spec, make)


__all__ = ["TEMPLATE_FRAME", "MNI305_TO_MNI152", "TierRecord", "GroupConnectome",
           "VascularSubstrate", "LabelVolume", "StructuralSubstrate", "REFUSALS",
           "build_substrate", "load"]
