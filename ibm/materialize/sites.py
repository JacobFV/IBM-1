"""grid(R, r): where the state variables actually sit.

the request is symbolic; this is the first thing in `ibm.materialize` that costs
anything.  a `Resolution` is a set of rules over symbolic regions, and turning it
into positions means evaluating r(q) *against the sampling itself* -- the spacing
at a point decides whether that point exists, which is why this cannot be a grid
generator with a spacing argument and has to be an adaptive refinement.

three samplers, because the supports genuinely do not share one.  a volume
refines an octree; a surface takes a variable-radius poisson disk under geodesic
distance; a tree walks its segments.  forcing any of them into the others' shape
is the universal-mesh mistake `ibm.fields.supports` exists to refuse: two points a
millimetre apart across a sulcus are centimetres apart along the sheet, and two
capillaries adjacent in space may be far apart in flow.

what a site carries beyond its position is the part downstream code actually
needs and that a bare point cloud silently loses.  per-site volume or area is the
measure a process integrates over -- a conductance per unit area, a metabolic
rate per unit volume, a lead field weighted by source extent -- and with adaptive
sampling it varies by orders of magnitude across one table, so leaving it out
makes every such process quietly wrong by the local refinement factor.  the
resolution level is kept because provenance has to be able to say where a process
ran outside the regime its f was written for, and that question is asked per site.

nothing here invents geometry.  a subject surface, a vascular segmentation and a
head model are data; when they are absent the samplers say which file is missing
and where such a file comes from, because "not implemented" is both false and
useless -- the algorithm is right here, the anatomy is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ibm.materialize.request import Budget, BudgetExceeded, MaterializationRequest
from ibm.registry import REGISTRY
from ibm.topologies.builders import SiteTable, Sites
from ibm.vocabulary import (
    Anat, Ball, Band, Difference, Everywhere, Intersect, Near, OnSupport, Region,
    Resolution, Union,
)


# ---------------------------------------------------------------------------
# missing data
# ---------------------------------------------------------------------------


class MissingData(RuntimeError):
    """a materialization step was asked to run without data that cannot be invented.

    the same argument `ibm.topologies.builders.MissingInput` makes, one layer up.
    a bare NotImplementedError would claim the algorithm is absent, which is a lie
    that costs the caller an afternoon: the octree is implemented, the geodesic
    sampler is implemented, and what is missing is a file with a subject's pial
    surface in it.  so the message names the argument, what it must contain, and
    where such a thing is obtained.
    """

    def __init__(self, step: str, needed: str, what: str, where: str = "") -> None:
        msg = (f"materialization step {step!r} cannot run: it needs {needed!r}.\n"
               f"  what it must contain: {what}")
        if where:
            msg += f"\n  where such a thing comes from: {where}"
        super().__init__(msg)
        self.step, self.needed, self.what, self.where = step, needed, what, where


def _csgraph(step: str):
    """scipy's graph routines, or an error rather than a euclidean substitute.

    the fallback everywhere else in ibm-1 is a slower numpy path, but there is no
    slower numpy path for geodesic distance, and quietly substituting euclidean
    distance on a folded sheet is precisely the error `CORTICAL_SURFACE` is
    declared to prevent.  better to stop.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse import csgraph
        return csr_matrix, csgraph
    except ImportError as exc:                                     # pragma: no cover
        raise MissingData(step, "scipy.sparse.csgraph",
                          "dijkstra over the mesh edge graph; geodesic distance on a folded "
                          "sheet has no euclidean substitute worth making silently",
                          "pip install scipy") from exc


# ---------------------------------------------------------------------------
# geometry: the data ibm-1 does not synthesize
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolumeGeometry:
    """a support's extent as an occupancy field over a frame.

    `occupancy` is soft on purpose.  a tissue mask is a partial volume fraction,
    not a binary, and thresholding it at the sampling stage throws away the one
    piece of information that tells an octree leaf near the pial surface that it
    is half csf.  that fraction becomes the leaf's effective volume.
    """

    support: str
    frame: str
    bounds_mm: Any                      # (2, 3) lower and upper corner
    occupancy: Any = None               # (nx, ny, nz) in [0, 1]
    affine: Any = None                  # (4, 4) voxel index -> frame, mm
    inside: Callable[[Any], Any] | None = None
    source: str = ""
    kind: str = "volume"

    def occupied(self, xyz: Any) -> np.ndarray:
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        lo, hi = np.asarray(self.bounds_mm, float)
        inbox = np.all((xyz >= lo) & (xyz <= hi), axis=1).astype(float)
        if self.inside is not None:
            return np.clip(np.asarray(self.inside(xyz), float).ravel(), 0.0, 1.0) * inbox
        if self.occupancy is None or self.affine is None:
            return inbox
        occ = np.asarray(self.occupancy, float)
        inv = np.linalg.inv(np.asarray(self.affine, float))
        h = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1) @ inv.T
        idx = np.rint(h[:, :3]).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(occ.shape)), axis=1)
        out = np.zeros(len(xyz))
        i = idx[ok]
        out[ok] = occ[i[:, 0], i[:, 1], i[:, 2]]
        return np.clip(out, 0.0, 1.0) * inbox


@dataclass(frozen=True)
class SurfaceGeometry:
    """a triangulated sheet.  vertices are the sampling candidates.

    the mesh is kept on the site table rather than consumed, because the geodesic
    topology builders need it and rebuilding a surface from its samples is both
    expensive and lossy -- the folding is the information.
    """

    support: str
    frame: str
    vertices: Any                       # (n, 3) mm
    faces: Any                          # (m, 3) vertex indices
    source: str = ""
    kind: str = "surface"

    def vertex_area(self) -> np.ndarray:
        """one third of each incident triangle's area: the standard barycentric dual."""
        v = np.asarray(self.vertices, float)
        f = np.asarray(self.faces, np.int64)
        a = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) * 0.5
        out = np.zeros(len(v))
        for c in range(3):
            np.add.at(out, f[:, c], a / 3.0)
        return out

    def edge_graph(self, step: str = "surface_sites"):
        """the mesh as a sparse graph weighted by edge length, both directions."""
        csr, _ = _csgraph(step)
        v = np.asarray(self.vertices, float)
        f = np.asarray(self.faces, np.int64)
        i = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        j = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
        i, j = np.concatenate([i, j]), np.concatenate([j, i])
        w = np.linalg.norm(v[i] - v[j], axis=1)
        return csr((w, (i, j)), shape=(len(v), len(v)))


@dataclass(frozen=True)
class TreeGeometry:
    """a branching transport network: nodes, a parent array, and radii.

    the parent array is the support.  a tree is not a point cloud with edges
    inferred from proximity, and resampling it must not be allowed to merge two
    branches that pass close to each other -- which is why `tree_sites` subdivides
    segments and never re-links by distance.
    """

    support: str
    frame: str
    xyz: Any                            # (n, 3) mm
    parent: Any                         # (n,) int, -1 at a root
    radius_mm: Any = None               # (n,)
    source: str = ""
    kind: str = "tree"

    def roots(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.parent, np.int64) < 0)

    def radii(self) -> np.ndarray:
        if self.radius_mm is None:
            raise MissingData(
                "tree_sites", f"{self.support}.radius_mm",
                "a per-node vessel radius in mm; without it a site has no volume and no "
                "resistance, and every flow process on this support is a shape with no units",
                "vessel segmentation from a TOF/SWI venogram or a 7T angiogram; see "
                "data/sources for the cards that carry one")
        return np.asarray(self.radius_mm, float)


@dataclass(frozen=True)
class DiscreteGeometry:
    """contacts, sensors, motor units: positions that r(q) has no say over.

    an electrode is where it is.  refining around it changes the tissue sampling,
    never the contact count, and a sampler that "coarsened" a 256-channel montage
    to 64 would be inventing a different instrument.
    """

    support: str
    frame: str
    xyz: Any
    ids: tuple[str, ...] = ()
    source: str = ""
    kind: str = "discrete"


Geometry = VolumeGeometry | SurfaceGeometry | TreeGeometry | DiscreteGeometry


@dataclass(frozen=True)
class GeometrySet:
    """every support's geometry for one subject, keyed by support name."""

    by_support: Mapping[str, Geometry] = _field(default_factory=dict)
    subject: str = "template"

    def __contains__(self, support: str) -> bool:
        return support in self.by_support

    def get(self, support: str) -> Geometry | None:
        return self.by_support.get(support)

    def require(self, support: str, step: str = "build_sites") -> Geometry:
        g = self.by_support.get(support)
        if g is None:
            s = REGISTRY.supports.get(support)
            what = (f"the extent of the {support!r} support ({s.doc[:80]}...)" if s
                    else f"the extent of the {support!r} support")
            raise MissingData(
                step, f"geometry[{support!r}]", what,
                "a subject reconstruction: FreeSurfer/fastsurfer surfaces for the cortical "
                "supports, a segmented T1 for the volume supports, a vessel segmentation for "
                "vascular_tree, digitised or CT-localised positions for the device supports.  "
                "pass allow_template_geometry=True on the request to accept a template stand-in, "
                "which is systematic error and is recorded as such")
        return g

    def with_geometry(self, g: Geometry) -> "GeometrySet":
        d = dict(self.by_support)
        d[g.support] = g
        return replace(self, by_support=d)


def template_volume(support: str, frame: str = "subject_t1",
                    semi_axes_mm: tuple[float, float, float] = (72.0, 92.0, 62.0),
                    center_mm: tuple[float, float, float] = (0.0, -18.0, 8.0)) -> VolumeGeometry:
    """an ellipsoid stand-in for a volume support.

    deliberately crude, and deliberately not the default.  it exists so that cost
    accounting, budget checks and the coarsening equivalence argument can be run
    before a subject's anatomy is in hand; a materialization that keeps it has an
    error in every position it reports, and provenance says so in those words
    rather than quietly carrying a plausible-looking number.
    """
    a = np.asarray(semi_axes_mm, float)
    c = np.asarray(center_mm, float)
    return VolumeGeometry(
        support, frame, np.stack([c - a, c + a]),
        inside=lambda xyz: (np.sum(((np.asarray(xyz, float) - c) / a) ** 2, axis=1) <= 1.0).astype(float),
        source="template ellipsoid -- not this subject's anatomy",
        kind="volume")


# ---------------------------------------------------------------------------
# regions -> weights
# ---------------------------------------------------------------------------


@dataclass
class RegionResolver:
    """turns a symbolic `Region` into soft membership over positions.

    membership is in [0, 1] and never thresholded here (ARCHITECTURE.md §2: a
    partition boundary is a gradient, not a wall).  the set algebra uses the
    standard fuzzy operators -- min for intersection, max for union, and
    `a * (1 - b)` for difference -- because those are the ones that agree with the
    crisp case and stay idempotent, which matters when a region expression is
    evaluated twice in one build.

    every region carries the frame it was written in, and this refuses to compare
    positions across frames.  that refusal is the whole point of `ibm.frames`:
    silent misregistration does not announce itself, the numbers stay plausible,
    and a `Ball` written in MNI applied to native positions is off by centimetres
    at the rim while looking entirely reasonable.
    """

    frame: str
    anchors: Mapping[str, Any] = _field(default_factory=dict)
    #: (system, label, xyz, frame) -> (n,) weights in [0, 1].  supplied by
    #: `ibm.anatomy`; absent means atlas-valued regions cannot be evaluated.
    anatomy: Callable[[str, str, Any, str], Any] | None = None
    #: (xyz, src_frame, dst_frame) -> xyz.  supplied by the caller; absent means
    #: cross-frame regions raise instead of being approximated.
    warp: Callable[[Any, str, str], Any] | None = None
    #: (support, xyz) -> (n_sources, n) geodesic distances, for Near(metric="geodesic")
    geodesic: Callable[[str, Any, Any], Any] | None = None
    misses: list[str] = _field(default_factory=list)

    # -- frames ----------------------------------------------------------

    def _to(self, xyz: np.ndarray, src: str, dst: str) -> np.ndarray:
        if src == dst:
            return xyz
        if self.warp is None:
            from ibm import frames as _frames
            chain = _frames.path(src, dst)
            hint = ("the frames are connected in ibm.frames by "
                    + " -> ".join(w.method for w in chain) if chain else
                    "ibm.frames declares no warp chain between them at all, which is a "
                    "modelling gap and not a missing file")
            raise MissingData(
                "region_weights", f"warp({src!r} -> {dst!r})",
                f"a transform carrying positions from {src!r} into {dst!r}; {hint}",
                "the subject's registration outputs -- an affine plus a warp field, a surface "
                "registration, or digitised fiducials for a device frame")
        return np.asarray(self.warp(xyz, src, dst), float).reshape(-1, 3)

    # -- evaluation ------------------------------------------------------

    def weights(self, region: Region, xyz: Any, support: str = "") -> np.ndarray:
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        n = len(xyz)
        if isinstance(region, Everywhere):
            return np.ones(n)
        if isinstance(region, OnSupport):
            return np.ones(n) if region.support == support else np.zeros(n)
        if isinstance(region, Ball):
            p = self._to(xyz, self.frame, region.frame)
            d = np.linalg.norm(p - np.asarray(region.center, float), axis=1)
            return (d <= region.radius_mm).astype(float)
        if isinstance(region, Near):
            return self._near(region, xyz, support)
        if isinstance(region, Anat):
            return self._anat(region, xyz)
        if isinstance(region, Union):
            parts = [self.weights(p, xyz, support) for p in region.parts]
            return np.max(np.stack(parts), axis=0) if parts else np.zeros(n)
        if isinstance(region, Intersect):
            parts = [self.weights(p, xyz, support) for p in region.parts]
            return np.min(np.stack(parts), axis=0) if parts else np.ones(n)
        if isinstance(region, Difference):
            return np.clip(self.weights(region.left, xyz, support)
                           * (1.0 - self.weights(region.right, xyz, support)), 0.0, 1.0)
        raise TypeError(f"no rule for region {type(region).__name__}; regions are a closed "
                        "vocabulary in ibm.vocabulary and a new one needs a case here")

    def _near(self, region: Near, xyz: np.ndarray, support: str) -> np.ndarray:
        pts = self.anchors.get(region.anchor)
        if pts is None:
            raise MissingData(
                "region_weights", f"anchor positions for {region.anchor!r}",
                f"the (m, 3) positions of the device or landmark named {region.anchor!r}, in "
                f"the {self.frame!r} frame; r(q) refines around them and without them the "
                "resolution rule has nothing to be near",
                "DeviceSpec(positions=...) on the request -- digitised electrode positions, a "
                "post-implant CT localisation, or a coil tracking record")
        pts = np.asarray(pts, float).reshape(-1, 3)
        if region.metric == "geodesic":
            if self.geodesic is None:
                raise MissingData(
                    "region_weights", "geodesic distance callback",
                    f"geodesic distance from {region.anchor!r} over the {support!r} support; "
                    "the region asked for distance along the sheet and euclidean distance is "
                    "not a stand-in for it across a sulcus",
                    "pass geodesic=... to RegionResolver, backed by the surface mesh")
            d = np.min(np.asarray(self.geodesic(support, pts, xyz), float), axis=0)
        else:
            d = np.min(np.linalg.norm(xyz[None, :, :] - pts[:, None, :], axis=-1), axis=0)
        return (d <= region.radius_mm).astype(float)

    def _anat(self, region: Anat, xyz: np.ndarray) -> np.ndarray:
        if self.anatomy is None:
            raise MissingData(
                "region_weights", f"anatomy[{region.system!r}][{region.label!r}]",
                f"soft membership a(q) in [0,1] of each position in the {region.label!r} "
                f"partition of the {region.system!r} system",
                "ibm.anatomy: a probabilistic atlas resampled into this materialization's "
                "frame.  pass anatomy=... to the resolver")
        w = np.clip(np.asarray(self.anatomy(region.system, region.label, xyz, self.frame),
                               float).ravel(), 0.0, 1.0)
        if w.shape != (len(xyz),):
            raise ValueError(f"anatomy callback returned {w.shape} for {len(xyz)} positions")
        return np.where(w >= region.threshold, w, 0.0)


# ---------------------------------------------------------------------------
# r(q) and B(q)
# ---------------------------------------------------------------------------


def spacing_at(resolution: Resolution, xyz: Any, resolver: RegionResolver,
               support: str = "") -> np.ndarray:
    """r(q) evaluated at positions.  first matching rule wins.

    "matching" means non-zero membership, not membership above a half.  a rule
    written for a probabilistic partition should refine wherever that partition
    has any weight at all: the alternative is a resolution boundary that cuts
    through the middle of a soft label, which puts the coarse/fine seam exactly
    where the atlas is least certain.
    """
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    out = np.full(len(xyz), float(resolution.default_mm))
    assigned = np.zeros(len(xyz), bool)
    for rule in resolution.rules:
        if assigned.all():
            break
        w = resolver.weights(rule.region, xyz, support)
        hit = (~assigned) & (w > 0.0)
        out[hit] = float(rule.spacing_mm)
        assigned |= hit
    return out


def band_at(resolution: Resolution, xyz: Any, resolver: RegionResolver,
            support: str = "") -> list[Band]:
    """B(q) evaluated at positions, as the per-site band ceiling.

    kept separate from `spacing_at` and returned as a list rather than an array
    because a band is a pair of floats with an intersection operator and not a
    number; collapsing it to two float arrays here would lose the fact that the
    layout allocates one basis per component, not one per site.
    """
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    out = [resolution.default_band] * len(xyz)
    assigned = np.zeros(len(xyz), bool)
    for rule in resolution.rules:
        if assigned.all():
            break
        w = resolver.weights(rule.region, xyz, support)
        hit = (~assigned) & (w > 0.0)
        for i in np.flatnonzero(hit):
            out[int(i)] = rule.band & resolution.default_band
        assigned |= hit
    return out


def widest_band(resolution: Resolution) -> Band:
    """the loosest B(q) any rule asks for: what the layout must allocate.

    a spectral block is one basis over all its sites, so the truncation has to be
    the widest band any site needs.  narrowing per site would save nothing -- the
    array is rectangular -- and would make every process read a ragged spectrum.
    """
    lo, hi = resolution.default_band.lo_hz, resolution.default_band.hi_hz
    for r in resolution.rules:
        b = r.band & resolution.default_band
        lo, hi = min(lo, b.lo_hz), max(hi, b.hi_hz)
    return Band(lo, hi)


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------


def _min_spacing(support: str) -> float:
    s = REGISTRY.supports.get(support)
    return float(s.min_spacing_mm) if s is not None else 0.0


def _frame_of(support: str, fallback: str) -> str:
    s = REGISTRY.supports.get(support)
    return s.frame if s is not None else fallback


def octree_sites(geom: VolumeGeometry, resolution: Resolution, resolver: RegionResolver, *,
                 budget: Budget = Budget(), offset: int = 0) -> SiteTable:
    """adaptive octree over a volume support.

    the loop is level-synchronous, which is not an optimization detail but the
    reason the budget can be enforced *eagerly*.  every cell at a level has the
    same edge length, so the site count after the next subdivision is known before
    any of it is allocated, and a request that would refine past its ceiling is
    refused while it is still eight arrays of centres rather than after it has
    discovered it is out of memory by running out of memory.

    a leaf is kept with its occupancy fraction as a weight and its edge length
    cubed times that fraction as its volume.  a leaf straddling the pial surface
    is genuinely half csf, and pretending otherwise puts a full cell's worth of
    metabolic demand in the subarachnoid space.
    """
    lo, hi = np.asarray(geom.bounds_mm, float)
    root = float(np.max(hi - lo))
    if not np.isfinite(root) or root <= 0:
        raise ValueError(f"{geom.support}: degenerate bounds {geom.bounds_mm!r}")
    centre = (lo + hi) * 0.5
    floor = max(_min_spacing(geom.support), 0.0)

    cells = centre.reshape(1, 3)
    size = root
    level = 0
    leaf_xyz: list[np.ndarray] = []
    leaf_size: list[np.ndarray] = []
    leaf_level: list[np.ndarray] = []
    leaf_occ: list[np.ndarray] = []
    n_kept = 0

    while len(cells):
        occ = _cell_occupancy(geom, cells, size)
        keep = occ > 0.0
        cells, occ = cells[keep], occ[keep]
        if not len(cells):
            break
        target = spacing_at(resolution, cells, resolver, geom.support)
        can_split = (level < budget.max_octree_level) and (size * 0.5 >= floor)
        split = (size > target) & can_split if can_split else np.zeros(len(cells), bool)
        done = ~split
        if done.any():
            leaf_xyz.append(cells[done])
            leaf_size.append(np.full(int(done.sum()), size))
            leaf_level.append(np.full(int(done.sum()), level, dtype=np.int32))
            leaf_occ.append(occ[done])
            n_kept += int(done.sum())
        if not split.any():
            break
        projected = n_kept + int(split.sum()) * 8
        if projected > budget.max_sites_per_support:
            finest = float(np.min(target[split]))
            raise BudgetExceeded(
                f"{geom.support}: refining level {level} ({size:g} mm cells) towards a "
                f"{finest:g} mm target would reach {projected:,} sites, past the "
                f"{budget.max_sites_per_support:,} per-support ceiling.  coarsen the r(q) rule "
                "covering that region, or shrink the region it covers -- refining the whole "
                "volume to a neighbourhood's spacing is the usual cause")
        cells = _subdivide(cells[split], size)
        size *= 0.5
        level += 1

    if not leaf_xyz:
        raise MissingData(
            "octree_sites", f"occupied volume in {geom.support!r}",
            f"the occupancy field for {geom.support!r} is empty everywhere inside its bounds, so "
            "grid(R, r) has nothing to sample.  the mask and the bounds are probably in "
            "different frames, or the mask is all zeros",
            geom.source or "the segmentation this geometry was built from")

    xyz = np.concatenate(leaf_xyz)
    edge = np.concatenate(leaf_size)
    lvl = np.concatenate(leaf_level)
    frac = np.concatenate(leaf_occ)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    xyz, edge, lvl, frac = xyz[order], edge[order], lvl[order], frac[order]
    return SiteTable(
        support=geom.support, frame=geom.frame, xyz=xyz, offset=offset, spacing_mm=edge,
        columns={"volume_mm3": (edge ** 3) * frac, "area_mm2": np.full(len(xyz), np.nan),
                 "level": lvl, "occupancy": frac, "kind": "volume",
                 "metric": "euclidean", "geometry_source": geom.source})


def _cell_occupancy(geom: VolumeGeometry, centres: np.ndarray, size: float) -> np.ndarray:
    """occupancy of a cell, sampled at its centre and its corners.

    the corner test is what stops a thin structure -- a sulcal bank, a vessel, the
    skull's inner table -- from being pruned at a coarse level because the cell's
    centre happened to fall in csf.  once pruned it never comes back, so the test
    is conservative on purpose and the extra eight evaluations are cheap next to
    the alternative of losing the structure entirely.
    """
    h = size * 0.5
    corners = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)])
    best = geom.occupied(centres)
    if size > _min_spacing(geom.support):
        for c in corners:
            best = np.maximum(best, geom.occupied(centres + c))
    return best


def _subdivide(centres: np.ndarray, size: float) -> np.ndarray:
    q = size * 0.25
    off = np.array([[sx, sy, sz] for sx in (-q, q) for sy in (-q, q) for sz in (-q, q)])
    return (centres[:, None, :] + off[None, :, :]).reshape(-1, 3)


def surface_sites(geom: SurfaceGeometry, resolution: Resolution, resolver: RegionResolver, *,
                  budget: Budget = Budget(), seed: int = 0, offset: int = 0) -> SiteTable:
    """variable-radius poisson-disk sampling under geodesic distance.

    a surface is not sampled on a grid because it does not have one, and it is not
    sampled by decimating the mesh because mesh density is a reconstruction
    artefact -- FreeSurfer puts vertices where the curvature is, which is not where
    r(q) wants sites.  poisson-disk gives a blue-noise set whose local density is
    exactly 1/r(q)^2 and whose minimum separation is guaranteed, which is what the
    lateral-propagation topologies assume when they build a neighbourhood graph.

    candidates are visited finest-radius-first.  with variable radii that ordering
    is not cosmetic: a coarse sample accepted early sweeps a large disk and would
    swallow a whole fine neighbourhood, so a request that asks for 50 um near a
    contact and 5 mm elsewhere would silently get 5 mm near the contact.

    the site's area is its geodesic voronoi cell, computed by one multi-source
    dijkstra at the end.  that is the measure every surface process integrates
    over, and with adaptive radii it varies across the sheet by the square of the
    refinement factor.
    """
    csr, cg = _csgraph("surface_sites")
    v = np.asarray(geom.vertices, float)
    if not len(v):
        raise MissingData("surface_sites", f"{geom.support}.vertices",
                          "a non-empty (n, 3) vertex array", geom.source)
    graph = geom.edge_graph("surface_sites")
    r = spacing_at(resolution, v, resolver, geom.support)
    r = np.maximum(r, _min_spacing(geom.support))

    rng = np.random.default_rng(seed)
    jitter = rng.random(len(v))
    order = np.lexsort((jitter, r))                 # finest radius first, ties broken by seed

    covered = np.zeros(len(v), bool)
    chosen: list[int] = []
    for vi in order:
        i = int(vi)
        if covered[i]:
            continue
        chosen.append(i)
        if len(chosen) > budget.max_sites_per_support:
            raise BudgetExceeded(
                f"{geom.support}: poisson-disk sampling at r(q) down to {float(r.min()):g} mm "
                f"passed the {budget.max_sites_per_support:,} per-support ceiling.  a surface "
                "sample count grows as 1/r^2, so halving the finest spacing quadruples this")
        d = cg.dijkstra(graph, directed=False, indices=i, limit=float(r[i]))
        covered |= np.isfinite(d)

    idx = np.array(sorted(chosen), dtype=np.int64)
    _, sources = _voronoi(cg, graph, idx)
    va = geom.vertex_area()
    area = np.zeros(len(idx))
    pos = {int(s): k for k, s in enumerate(idx)}
    valid = sources >= 0
    np.add.at(area, np.array([pos[int(s)] for s in sources[valid]], dtype=np.int64), va[valid])

    return SiteTable(
        support=geom.support, frame=geom.frame, xyz=v[idx], offset=offset, spacing_mm=r[idx],
        columns={"area_mm2": area, "volume_mm3": np.full(len(idx), np.nan),
                 "level": np.rint(np.log2(np.max(r) / np.maximum(r[idx], 1e-9))).astype(np.int32),
                 "vertex_of_site": idx, "mesh_vertices": v, "mesh_faces": np.asarray(geom.faces),
                 "voronoi_source": sources, "kind": "surface", "metric": "geodesic",
                 "geometry_source": geom.source})


def _voronoi(cg, graph, sources_idx: np.ndarray):
    """geodesic voronoi assignment of every mesh vertex to its nearest sample.

    scipy's `min_only` multi-source dijkstra does this in one sweep and returns
    the winning source per vertex, which is exactly the assignment needed for the
    dual areas.  doing it per-sample instead would be N dijkstras over the whole
    mesh rather than one.
    """
    try:
        out = cg.dijkstra(graph, directed=False, indices=sources_idx, min_only=True,
                          return_predecessors=True)
        dist, sources = out[0], np.asarray(out[-1])
    except TypeError:                                              # pragma: no cover
        dist = cg.dijkstra(graph, directed=False, indices=sources_idx, min_only=True)
        sources = np.full(graph.shape[0], -1, dtype=np.int64)
    if np.all(sources < 0):
        # older scipy: recover the assignment by comparing per-source distances in
        # chunks.  correct, slower, and never silently wrong.
        best = np.full(graph.shape[0], np.inf)
        sources = np.full(graph.shape[0], -1, dtype=np.int64)
        for s in sources_idx:
            d = cg.dijkstra(graph, directed=False, indices=int(s))
            better = d < best
            best[better] = d[better]
            sources[better] = int(s)
    return dist, np.asarray(sources, dtype=np.int64)


def tree_sites(geom: TreeGeometry, resolution: Resolution, resolver: RegionResolver, *,
               budget: Budget = Budget(), offset: int = 0) -> SiteTable:
    """traverse a branching network, keeping every junction and subdividing segments.

    r(q) is a *floor* here and not a ceiling, which is the one place in this module
    where a resolution rule does not get the last word.  a bifurcation is topology:
    dropping it because the local spacing rule was coarse would reconnect two
    daughter branches into one vessel and change what the network is, whereas
    dropping an interior point along a straight segment only coarsens a length.
    so nodes always survive and only the interiors are resampled, and the number of
    nodes retained past their spacing rule is recorded rather than hidden.

    per-site volume is pi r^2 L over the half-segments meeting at the site, which
    is the quantity every flow, transit and oxygen-extraction process needs.
    """
    xyz = np.asarray(geom.xyz, float).reshape(-1, 3)
    par = np.asarray(geom.parent, np.int64).ravel()
    rad = geom.radii()
    if len(par) != len(xyz) or len(rad) != len(xyz):
        raise ValueError(f"{geom.support}: xyz, parent and radius must agree in length")

    node_target = spacing_at(resolution, xyz, resolver, geom.support)
    floor = _min_spacing(geom.support)

    pos = [xyz]
    radius = [rad]
    parent = [np.where(par >= 0, par, -1)]
    seg_len = [np.zeros(len(xyz))]
    kind = [np.zeros(len(xyz), np.int8)]            # 0 = original node, 1 = interior sample
    n = len(xyz)
    coarsened = 0

    for child in range(len(xyz)):
        p = int(par[child])
        if p < 0:
            continue
        a, b = xyz[p], xyz[child]
        length = float(np.linalg.norm(b - a))
        mid = (a + b) * 0.5
        target = max(float(spacing_at(resolution, mid.reshape(1, 3), resolver,
                                      geom.support)[0]), floor)
        if length <= target:
            seg_len[0][child] = length
            if target > length and node_target[child] > length:
                coarsened += 1
            continue
        k = int(math.ceil(length / target)) - 1
        if n + k > budget.max_sites_per_support:
            raise BudgetExceeded(
                f"{geom.support}: subdividing the vascular tree at {target:g} mm passed the "
                f"{budget.max_sites_per_support:,} per-support ceiling.  a tree's site count "
                "grows linearly in 1/r, so this is usually an r(q) rule that covers the whole "
                "network rather than the territory of interest")
        t = (np.arange(1, k + 1) / (k + 1.0))[:, None]
        pts = a[None, :] + t * (b - a)[None, :]
        rr = rad[p] + t.ravel() * (rad[child] - rad[p])
        chain = np.arange(n, n + k, dtype=np.int64)
        pos.append(pts)
        radius.append(rr)
        parent.append(np.concatenate([[p], chain[:-1]]))
        seg_len.append(np.full(k, length / (k + 1)))
        kind.append(np.ones(k, np.int8))
        parent[0][child] = chain[-1]
        seg_len[0][child] = length / (k + 1)
        n += k

    xyz_all = np.concatenate(pos)
    rad_all = np.concatenate(radius)
    par_all = np.concatenate(parent)
    len_all = np.concatenate(seg_len)
    kind_all = np.concatenate(kind)
    vol = math.pi * rad_all ** 2 * len_all
    # a node's volume is its own half-segment plus half of each child's
    child_half = np.zeros(len(xyz_all))
    has_parent = par_all >= 0
    np.add.at(child_half, par_all[has_parent], 0.5 * vol[has_parent])
    volume = 0.5 * vol + child_half

    return SiteTable(
        support=geom.support, frame=geom.frame, xyz=xyz_all, offset=offset,
        spacing_mm=np.maximum(len_all, floor),
        columns={"parent": par_all, "radius_mm": rad_all, "length_mm": len_all,
                 "volume_mm3": volume, "area_mm2": 2.0 * math.pi * rad_all * len_all,
                 "level": _depth(par_all), "is_junction": kind_all == 0,
                 "kind": "tree", "metric": "path",
                 "nodes_kept_past_rq": coarsened, "geometry_source": geom.source})


def _depth(parent: np.ndarray) -> np.ndarray:
    """generation index from the root: the tree's natural resolution level.

    swept level by level from the roots rather than in index order, because a
    parent array carries no ordering guarantee -- a segmentation writes nodes in
    whatever order it found them, and a single pass in index order silently gives
    a child a depth before its parent has one.
    """
    parent = np.asarray(parent, np.int64)
    n = len(parent)
    d = np.full(n, -1, np.int32)
    d[parent < 0] = 0
    for _ in range(n):
        unknown = np.flatnonzero(d < 0)
        if not len(unknown):
            break
        p = parent[unknown]
        known = d[p] >= 0
        if not known.any():
            break                       # a cycle, or a parent index out of range
        d[unknown[known]] = d[p[known]] + 1
    return np.maximum(d, 0)


def discrete_sites(geom: DiscreteGeometry, *, offset: int = 0) -> SiteTable:
    """the instrument's own elements, verbatim.

    r(q) does not apply and is not consulted.  a contact has a position and an
    area because it was manufactured with them, and a sampler that had an opinion
    about how many of them there should be would be describing a different device.
    """
    xyz = np.asarray(geom.xyz, float).reshape(-1, 3)
    return SiteTable(
        support=geom.support, frame=geom.frame, xyz=xyz, offset=offset,
        spacing_mm=np.full(len(xyz), np.nan),
        columns={"element_ids": np.asarray(geom.ids, dtype=object) if geom.ids else None,
                 "volume_mm3": np.full(len(xyz), np.nan),
                 "area_mm2": np.full(len(xyz), np.nan),
                 "level": np.zeros(len(xyz), np.int32), "kind": "discrete",
                 "metric": "euclidean", "geometry_source": geom.source})


def sample(geom: Geometry, resolution: Resolution, resolver: RegionResolver, *,
           budget: Budget = Budget(), seed: int = 0, offset: int = 0) -> SiteTable:
    """dispatch on the support's kind.  the four samplers are not interchangeable."""
    if isinstance(geom, VolumeGeometry):
        return octree_sites(geom, resolution, resolver, budget=budget, offset=offset)
    if isinstance(geom, SurfaceGeometry):
        return surface_sites(geom, resolution, resolver, budget=budget, seed=seed, offset=offset)
    if isinstance(geom, TreeGeometry):
        return tree_sites(geom, resolution, resolver, budget=budget, offset=offset)
    if isinstance(geom, DiscreteGeometry):
        return discrete_sites(geom, offset=offset)
    raise TypeError(f"no sampler for geometry {type(geom).__name__}")


# ---------------------------------------------------------------------------
# building every support's table
# ---------------------------------------------------------------------------


def device_geometries(request: MaterializationRequest) -> dict[str, DiscreteGeometry]:
    """a `DiscreteGeometry` per device support named in the request.

    a device with no positions still gets a table, because §1 insists a device is
    ordinary state with a support and not an input: a stimulator whose coil
    position is unknown still has a coil current, and the missing positions become
    an error only when a `Near(...)` rule or a coupling topology actually asks
    where it is.
    """
    out: dict[str, DiscreteGeometry] = {}
    for d in request.devices:
        p = d.anchor_positions()
        if p is None:
            n = max(int(d.n_elements), 1)
            p = np.full((n, 3), np.nan)
        out[d.support] = DiscreteGeometry(
            d.support, d.frame, p, tuple(d.element_ids),
            source=f"DeviceSpec({d.name!r})" + ("" if d.positions is not None
                                                else " -- positions not supplied"))
    return out


def build_sites(request: MaterializationRequest, supports: Iterable[str],
                geometry: GeometrySet, *, resolver: RegionResolver | None = None,
                cache: Any = None) -> Sites:
    """one site table per support the trace reached, with global offsets assigned.

    offsets are assigned in sorted support order rather than in trace order so
    that two materializations of the same supports produce the same global
    indices.  edge sets index globally (`ibm.topologies.builders`), so an unstable
    order would make two runs of the same request produce edge arrays that cannot
    be compared, and a cached geodesic graph would be worse than useless.
    """
    res = resolver or RegionResolver(frame=request.frame, anchors=request.anchors())
    devices = device_geometries(request)
    tables: dict[str, SiteTable] = {}
    offset = 0
    for s in sorted(set(supports)):
        geom = devices.get(s) or geometry.get(s)
        if geom is None:
            if request.allow_template_geometry and _is_volume(s):
                geom = template_volume(s, _frame_of(s, request.frame))
            else:
                geom = geometry.require(s, "build_sites")
        if cache is not None:
            key = {"support": s, "geometry": getattr(geom, "source", ""),
                   "frame": geom.frame, "kind": getattr(geom, "kind", ""),
                   "resolution": request.resolution, "seed": request.seed,
                   "budget": request.budget, "subject": request.subject.id,
                   "anchors": {k: np.asarray(v).round(4).tolist()
                               for k, v in sorted(res.anchors.items())}}
            t = cache.memo("sites", key,
                           lambda g=geom: sample(g, request.resolution, res,
                                                 budget=request.budget, seed=request.seed))
        else:
            t = sample(geom, request.resolution, res, budget=request.budget, seed=request.seed)
        tables[s] = replace(t, offset=offset)
        offset += t.n
    return Sites(tables)


def _is_volume(support: str) -> bool:
    s = REGISTRY.supports.get(support)
    return s is not None and s.kind == "volume"


# ---------------------------------------------------------------------------
# the global flat index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteLayout:
    """(component, site) -> one flat index over the whole materialization.

    named `SiteLayout` and not `Layout` because `ibm.runtime.state.Layout` already
    owns that word for a different job, and the two are genuinely different maps:
    the runtime's layout is *blocked* -- one entry per component carrying that
    component's form of uncertain state, which is the only way blood and neural
    population state can coexist -- while this one is the flat integer addressing
    that a sparse solver, a jacobian and a provenance table need.  keeping both is
    not duplication; collapsing them would force one uncertainty form on the whole
    model, which is what §1 spends its length refusing.

    the ordering is component-major.  a process reads one component over many
    sites far more often than one site over many components, so component-major
    keeps every such read contiguous.
    """

    entries: tuple[tuple[str, str, int, int], ...] = ()   # component, support, offset, n

    @classmethod
    def of(cls, pairs: Sequence[tuple[str, str, int]]) -> "SiteLayout":
        out, off = [], 0
        for component, support, n in pairs:
            out.append((component, support, off, int(n)))
            off += int(n)
        return cls(tuple(out))

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, component: str) -> bool:
        return any(e[0] == component for e in self.entries)

    def _entry(self, component: str) -> tuple[str, str, int, int]:
        for e in self.entries:
            if e[0] == component:
                return e
        raise KeyError(
            f"component {component!r} is not materialized: this layout holds "
            f"{len(self.entries)} components and the trace did not reach that one.  either the "
            "request does not target anything downstream of it or a process names state the "
            "trace missed")

    def index(self, component: str, site: Any = None) -> Any:
        """flat index of one state variable, or of every site of a component."""
        _, _, off, n = self._entry(component)
        if site is None:
            return np.arange(off, off + n)
        s = np.asarray(site, np.int64)
        if np.any((s < 0) | (s >= n)):
            raise IndexError(f"{component!r} has {n} sites; asked for {s.max()}")
        return off + s

    def span(self, component: str) -> slice:
        _, _, off, n = self._entry(component)
        return slice(off, off + n)

    def support_of(self, component: str) -> str:
        return self._entry(component)[1]

    def n_sites(self, component: str) -> int:
        return self._entry(component)[3]

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(e[0] for e in self.entries)

    @property
    def n_state_variables(self) -> int:
        """the §1 count: one component of one field at one position, summed."""
        return sum(e[3] for e in self.entries)

    def by_support(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, support, _, n in self.entries:
            out[support] = out.get(support, 0) + n
        return out

    def describe(self) -> str:
        rows = [(c, s, str(o), str(n)) for c, s, o, n in self.entries]
        head = ("component", "support", "offset", "sites")
        w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
             for i, h in enumerate(head)]
        out = ["  ".join(h.ljust(x) for h, x in zip(head, w)),
               "  ".join("-" * x for x in w)]
        out += ["  ".join(c.ljust(x) for c, x in zip(r, w)) for r in rows]
        out.append(f"{self.n_state_variables:,} state variables over "
                   f"{len(self.entries)} components")
        return "\n".join(out)


__all__ = [
    "MissingData", "VolumeGeometry", "SurfaceGeometry", "TreeGeometry", "DiscreteGeometry",
    "Geometry", "GeometrySet", "template_volume", "RegionResolver", "spacing_at", "band_at",
    "widest_band", "octree_sites", "surface_sites", "tree_sites", "discrete_sites", "sample",
    "device_geometries", "build_sites", "SiteLayout",
]
