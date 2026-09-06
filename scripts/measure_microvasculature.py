"""what a capillary bed actually looks like, and how many mice a mouse is worth.

microvasculature is the one piece of anatomy ibm-1 needs and no in-vivo human
measurement can reach.  a 1 mm voxel of cortex holds on the order of a metre of
capillary in ten thousand segments at five microns calibre and forty microns
spacing; 7T TOF-MRA and QSM venography stop at two or three hundred microns, two
orders of magnitude short.  so the well-posed question is not "derive the
network" -- it is not there to be derived -- but "synthesise a network consistent
with the macro tree that IS measurable, with ex-vivo statistics, and with the
flow it has to carry".  this script measures the second of those three, and
enough of the third to say whether the first two are even compatible.

what is measured, and against what
----------------------------------
two arms, deliberately different in what they can answer, because neither source
can answer both halves.

**arm A -- whole-brain topology (`vesselgraph-mouse`).**  nine whole mouse brains
from three strains, segmented from light-sheet volumes and skeletonised by
Voreen into node and edge lists carrying radius, arc length, chord distance and
curveness.  millions of segments per animal.  this is where the segment length
and radius distributions, the degree distribution, the tortuosity and the
bifurcation statistics come from, and it is the only source here with enough
animals for a between-animal spread that means anything.  what it CANNOT give is
flow: it is fixed, cleared, unperfused tissue.

**arm B -- cortical networks with a flow solution
(`microscopy-microvascular-networks`).**  three ~1.7 mm^3 blocks of mouse
vibrissa cortex reconstructed by Blinder et al. 2013 and distributed by the
Weber lab with a full pressure-and-flow solution on every segment (Zenodo
269650, CC BY 4.0).  vessel type is labelled -- pial artery, pial vein,
descending arteriole, ascending venule, capillary -- and the centreline of each
segment is stored as a polyline, so this arm carries everything that needs a
spatial volume rather than a graph: capillary LENGTH DENSITY, the tissue-to-
capillary distance that sets the Krogh geometry, the depth profile, the distance
from a penetrating vessel, and the physics.

the physics is the point of arm B and the reason this is not texture analysis.
a synthesised network is only worth anything if it can carry the flow the tissue
needs at pressures the circulation can supply, and the only way to know whether
that constraint is satisfiable at all is to check it on a network where somebody
has already solved it.  three things are therefore measured here that no
morphometry paper reports together:

  1. the flow unit, which the release does not document, recovered two
     independent ways that agree;
  2. the effective viscosity implied by each segment's own pressure drop, as a
     function of calibre -- which is the Fahraeus-Lindqvist correction
     `vascular_tree_adjacency` currently declines to apply, measured rather than
     quoted;
  3. the perfusion the network delivers per unit tissue, and the arteriole-to-
     venule pressure drop it does it with.

Murray's law
------------
the constraint a generative network model most wants is Murray's:
r_parent^3 = sum r_child^3, the radius relation that minimises the sum of
pumping power and metabolic cost of blood.  it is quoted constantly and tested
rarely, so it is tested here twice, on two independent sources, and the exponent
is FITTED per bifurcation rather than assumed -- solving

    r_p^k = sum_i r_ci^k

for k at each junction, which has a unique root whenever the parent is the
widest branch.  the distribution of k is what a prior should carry; a single
number would hide that the law is a statement about the mean of a very wide
distribution.

between-animal spread, and what nine animals are worth
------------------------------------------------------
the same argument `scripts/measure_tract_uncertainty.py` makes about pipelines
applies here with nothing changed.  nine mice are not nine independent votes
about capillary geometry: they share a strain background, a fixation protocol, a
clearing protocol, a segmentation network and a skeletonisation algorithm, and
every one of those is a shared failure mode with a known sign -- clearing
shrinks tissue, incomplete labelling drops the thinnest vessels, and light-sheet
attenuation makes both worse with depth.  so each animal's deviation from the
group consensus is computed over a common feature vector, the mean pairwise
correlation of those deviations is `correlated_fraction`, and the effective
count comes out of `ibm.runtime.fuse.TeacherPrecision.effective_constraints` --
the same call, not a reimplementation.

that rho is a LOWER bound and for the same reason the TractoInferno one is: the
reference is the animals' own consensus, so any error every animal makes is
invisible to it.  a shrinkage factor that applies to all nine cannot be seen
here at all.

three honesty problems
----------------------
**every length and radius is an ex-vivo one.**  the tissue was fixed, which
collapses the lumen, and cleared, which shrinks the whole block by a factor that
neither release reports per specimen.  a radius here is a segmentation's estimate
of a fixed vessel and not a perfused lumen.  the direction is knowable -- both
effects make vessels smaller and closer together -- so the densities below are
UPPER bounds and the calibres LOWER bounds, and no correction is applied because
applying one would mean inventing the factor.

**the smallest vessels are the most likely to be missed.**  a segmentation's
false negatives correlate with calibre and capillaries are both the thinnest and
the most numerous, so every density here is also a lower bound in the other
direction from the shrinkage.  the two do not cancel and are not known to be the
same size.

**it is a mouse.**  capillary density scales with metabolic rate and brain size,
and mouse cortical CMRO2 per gram is several times human.  what transfers across
species is the NORMALISED geometry -- the shape of the radius distribution, the
branching exponent, the ratio of intercapillary distance to diffusion length --
and not the absolute density, which is why the numbers below are reported both
raw and normalised and why `ibm/topologies/vascular_prior.py` conditions on a
target CBF rather than on a target density.

usage
-----
    ./.venv/bin/python scripts/measure_microvasculature.py --help
    ./.venv/bin/python scripts/measure_microvasculature.py --out data/sources/...
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# mmHg -> Pa.  the flow solutions are published in mmHg and Poiseuille is in SI.
MMHG_PA = 133.322387415

# vessel-type codes in the Blinder/Weber release.  0 and 1 sit at z ~ 0 and are
# the pial network; 2 and 3 span the full cortical depth and are the penetrating
# arteriole and ascending venule; 4 is the capillary bed and is 70-80% of every
# segment count.  5 exists, carries a median calibre close to 2 and 3, and is
# NOT documented in the release -- so it is counted, named `unlabelled`, and kept
# out of every capillary statistic rather than guessed into one.
KIND = {0: "pial_artery", 1: "pial_vein", 2: "descending_arteriole",
        3: "ascending_venule", 4: "capillary", 5: "unlabelled"}
CAPILLARY_KIND = 4


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def log_stats(x: np.ndarray) -> dict:
    """median and log-sd of a positive quantity, plus the plain percentiles.

    log because every length and every radius here is positive and known to
    within a multiplicative factor -- the same argument `theta_prior` makes for
    using a lognormal -- and because a spread quoted as +- microns is unusable
    for a capillary and a pial artery at the same time.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2:
        return {"n": int(x.size)}
    lg = np.log(x)
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "log_sd": float(lg.std(ddof=1)),
        "spread_factor": float(math.exp(lg.std(ddof=1))),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def between_animal(per_animal: dict[str, float]) -> dict:
    """spread of one scalar across animals, reported as a factor.

    the number a prior widens by, and it is deliberately the spread of the
    ANIMALS rather than the standard error of their mean: a materialization is
    predicting one brain, not the mean of nine.
    """
    v = np.array([x for x in per_animal.values() if np.isfinite(x) and x > 0], float)
    if v.size < 2:
        return {"n_animals": int(v.size)}
    lg = np.log(v)
    return {
        "n_animals": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "between_animal_log_sd": float(lg.std(ddof=1)),
        "between_animal_factor": float(math.exp(lg.std(ddof=1))),
        "cv": float(v.std(ddof=1) / v.mean()),
        "per_animal": {k: float(x) for k, x in per_animal.items()},
    }


def murray_exponent(rp: np.ndarray, rc: list[np.ndarray]) -> np.ndarray:
    """solve r_p^k = sum_i r_ci^k for k, per bifurcation.

    g(k) = sum (rc/rp)^k is strictly decreasing in k whenever every child is
    narrower than the parent, and g(0) = (number of children) > 1, so there is
    exactly one root and bisection finds it without a derivative.  a junction
    where some child is at least as wide as the parent has no root and is
    returned as nan rather than clipped -- those junctions are real (they are
    mostly capillary loops closing, where "parent" is a fiction) and dropping
    them silently would make the law look better than it is.
    """
    rp = np.asarray(rp, float)
    ratios = np.stack([np.asarray(r, float) / np.maximum(rp, 1e-12) for r in rc], axis=1)
    ok = np.all(ratios < 1.0 - 1e-9, axis=1) & np.all(ratios > 0, axis=1)
    out = np.full(rp.shape, np.nan)
    if not ok.any():
        return out
    R = ratios[ok]
    lo = np.full(R.shape[0], 1e-3)
    hi = np.full(R.shape[0], 60.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        g = np.power(R, mid[:, None]).sum(axis=1)
        too_big = g > 1.0            # g decreasing: g > 1 means k must rise
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    out[ok] = 0.5 * (lo + hi)
    return out


#: how far along a segment's own centreline the branching direction is taken.
#: short enough that it is still the direction the vessel leaves the junction in,
#: long enough not to be one skeletonisation voxel of noise.
LOCAL_DIR_UM = 15.0


def local_direction(nw: "Network", edge: int, vtx: int, reach_um: float) -> np.ndarray:
    """unit vector along a segment's centreline, leaving `vtx`.

    the stored polyline runs between the two endpoint vertices in some order, so
    the end nearer `vtx` is found rather than assumed -- assuming it reverses
    roughly half the branching angles, which turns a unimodal distribution into
    a symmetric one centred on 90 degrees and looks entirely plausible.
    """
    p = np.asarray(nw.points[edge], float).reshape(-1, 3)
    if p.shape[0] < 2:
        other = nw.tup[edge][0] if nw.tup[edge][1] == vtx else nw.tup[edge][1]
        d = nw.coords[other] - nw.coords[vtx]
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.zeros(3)
    if np.linalg.norm(p[-1] - nw.coords[vtx]) < np.linalg.norm(p[0] - nw.coords[vtx]):
        p = p[::-1]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    j = int(np.searchsorted(s, min(reach_um, s[-1])))
    j = min(max(j, 1), p.shape[0] - 1)
    d = p[j] - p[0]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.zeros(3)


def polyline_length(pts: np.ndarray) -> float:
    p = np.asarray(pts, float)
    if p.ndim != 2 or p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def resample_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    """points every `step` along a polyline, for a distance transform.

    the tissue-to-capillary distance is a distance to a CURVE, and taking it to
    the nearest stored vertex instead overestimates it by up to half a segment.
    at the 5 um step used here the error is under 2.5 um against a mean distance
    of tens of microns, which is small enough to state and ignore.
    """
    p = np.asarray(pts, float)
    if p.ndim != 2 or p.shape[0] < 2:
        return p.reshape(-1, 3)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = seg > 1e-9
    if not keep.any():
        return p[:1]
    p = np.vstack([p[:-1][keep], p[-1:]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / step) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1)


# ---------------------------------------------------------------------------
# arm B: the Blinder/Weber cortical networks, with flow
# ---------------------------------------------------------------------------


@dataclass
class Network:
    name: str
    coords: np.ndarray          # (nv, 3) um
    pressure: np.ndarray        # (nv,) mmHg
    is_bc: np.ndarray           # (nv,) bool -- pressure boundary condition set
    tup: np.ndarray             # (ne, 2) vertex ids
    diameter: np.ndarray        # (ne,) um
    length: np.ndarray          # (ne,) um, arc length
    flow: np.ndarray            # (ne,) release units
    kind: np.ndarray            # (ne,) vessel-type code
    points: list                # (ne,) polylines, um


def load_network(root: Path, name: str) -> Network:
    e = pickle.loads((root / f"{name}_results" / "edgesDict.pkl").read_bytes(),
                     encoding="latin1")
    v = pickle.loads((root / f"{name}_results" / "verticesDict.pkl").read_bytes(),
                     encoding="latin1")
    return Network(
        name=name,
        coords=np.asarray(v["coords"], float),
        pressure=np.asarray(v["pressure"], float),
        is_bc=np.array([x is not None for x in v["pBC"]], bool),
        tup=np.asarray(e["tuple"], np.int64),
        diameter=np.asarray(e["diameter"], float),
        length=np.asarray(e["length"], float),
        flow=np.asarray(e["flow"], float),
        kind=np.asarray(e["nkind"], np.int64),
        points=list(e["points"]),
    )


def interior_mask(nw: Network, margin_um: float) -> tuple[np.ndarray, float, np.ndarray]:
    """the box the density is computed over, eroded away from the cut faces.

    a reconstruction of a block of cortex is truncated at every face, and a
    segment crossing a face is stored with whatever length survived the cut.
    counting those would understate the length density in a boundary shell whose
    thickness is a segment length.  so a margin is eroded and only segments whose
    MIDPOINT falls inside are counted -- an approximation whose error is bounded
    by half a segment length at the box faces, which at a 100 um margin and a
    50 um median capillary segment is a percent-level effect.
    """
    lo = nw.coords.min(axis=0) + margin_um
    hi = nw.coords.max(axis=0) - margin_um
    mid = 0.5 * (nw.coords[nw.tup[:, 0]] + nw.coords[nw.tup[:, 1]])
    inside = np.all((mid >= lo) & (mid <= hi), axis=1)
    vol_um3 = float(np.prod(hi - lo))
    return inside, vol_um3, np.stack([lo, hi])


def orient_by_pressure(nw: Network) -> tuple[np.ndarray, np.ndarray]:
    """(upstream, downstream) vertex per edge, from the pressure field.

    the graph is stored undirected and blood is not.  direction is taken from
    the solved pressure rather than from the vertex ordering, because the vertex
    ordering means nothing and getting the sign wrong is exactly the error
    `ibm/topologies/vascular.py` warns about -- it produces a perfectly smooth
    and perfectly inverted response.
    """
    a, b = nw.tup[:, 0], nw.tup[:, 1]
    hi = nw.pressure[a] >= nw.pressure[b]
    return np.where(hi, a, b), np.where(hi, b, a)


def bifurcations(nw: Network, up: np.ndarray, dn: np.ndarray) -> dict:
    """junctions with one vessel in and two out, and what happens to the radii.

    degree three and a 1-in-2-out split is the case Murray's law is about.  a
    degree-three junction with 2-in-1-out is a CONFLUENCE and is measured
    separately: the law is not symmetric in practice even though the algebra is,
    because the venous side is more compliant and less regulated, and reporting
    the two together would average a real difference away.
    """
    nv = nw.coords.shape[0]
    deg = np.bincount(nw.tup.ravel(), minlength=nv)
    # for each vertex, the incident edges
    order = np.argsort(nw.tup.ravel(), kind="stable")
    vert = nw.tup.ravel()[order]
    edge = (order // 2)
    starts = np.searchsorted(vert, np.arange(nv))
    ends = np.searchsorted(vert, np.arange(nv), side="right")

    out: dict[str, list] = {"bif_parent_r": [], "bif_child_r": [], "bif_angle_deg": [],
                            "bif_parent_child_angle_deg": [], "bif_kind": [],
                            "con_parent_r": [], "con_child_r": []}
    for vtx in np.nonzero(deg == 3)[0]:
        es = edge[starts[vtx]:ends[vtx]]
        if es.size != 3:
            continue
        incoming = [e for e in es if dn[e] == vtx]
        outgoing = [e for e in es if up[e] == vtx]
        if len(incoming) == 1 and len(outgoing) == 2:
            p, c = incoming[0], outgoing
            rp = 0.5 * nw.diameter[p]
            rc = [0.5 * nw.diameter[c[0]], 0.5 * nw.diameter[c[1]]]
            # direction of each branch AT the junction, taken from its stored
            # centreline rather than from the chord to the far endpoint.  a
            # capillary's chord direction is a 17% tortuosity away from where it
            # actually leaves the junction, and a branching angle built on chords
            # would be measuring the segment's wander rather than the branching.
            d1 = local_direction(nw, c[0], vtx, LOCAL_DIR_UM)
            d2 = local_direction(nw, c[1], vtx, LOCAL_DIR_UM)
            dp = local_direction(nw, p, vtx, LOCAL_DIR_UM)
            ang = math.degrees(math.acos(float(np.clip(d1 @ d2, -1, 1)))) if (
                np.any(d1) and np.any(d2)) else np.nan
            # dp points AWAY from the junction along the parent, so the angle a
            # daughter makes with the parent's incoming direction is against -dp.
            apc = [math.degrees(math.acos(float(np.clip(-dp @ d, -1, 1))))
                   for d in (d1, d2) if np.any(d) and np.any(dp)]
            out["bif_parent_r"].append(rp)
            out["bif_child_r"].append(rc)
            out["bif_angle_deg"].append(ang)
            out["bif_parent_child_angle_deg"].extend(apc)
            out["bif_kind"].append(int(nw.kind[p]))
        elif len(incoming) == 2 and len(outgoing) == 1:
            p = outgoing[0]
            out["con_parent_r"].append(0.5 * nw.diameter[p])
            out["con_child_r"].append([0.5 * nw.diameter[incoming[0]],
                                       0.5 * nw.diameter[incoming[1]]])
    return out


def analyse_flow_network(nw: Network, *, margin_um: float, n_probe: int,
                         depth_bins: int, rng: np.random.Generator) -> dict:
    inside, vol_um3, box = interior_mask(nw, margin_um)
    cap = nw.kind == CAPILLARY_KIND
    up, dn = orient_by_pressure(nw)
    r = 0.5 * nw.diameter

    res: dict = {"name": nw.name,
                 "n_vertices": int(nw.coords.shape[0]),
                 "n_edges": int(nw.tup.shape[0]),
                 "interior_box_um": box.tolist(),
                 "interior_volume_mm3": vol_um3 * 1e-9,
                 "kind_counts": {KIND.get(int(k), str(k)): int((nw.kind == k).sum())
                                 for k in np.unique(nw.kind)}}

    # -- geometry by vessel class -------------------------------------------
    res["by_kind"] = {}
    for k, nm in KIND.items():
        m = nw.kind == k
        if not m.any():
            continue
        res["by_kind"][nm] = {"radius_um": log_stats(r[m]),
                              "length_um": log_stats(nw.length[m])}

    # -- capillary length density -------------------------------------------
    sel = cap & inside
    cap_len_um = float(nw.length[sel].sum())
    all_len_um = float(nw.length[inside].sum())
    res["length_density_mm_per_mm3"] = {
        "capillary": cap_len_um * 1e-3 / (vol_um3 * 1e-9),
        "all_vessels": all_len_um * 1e-3 / (vol_um3 * 1e-9),
        "capillary_fraction_of_length": cap_len_um / max(all_len_um, 1e-9),
    }
    # surface density: the quantity `capillary_tissue_exchange` carries per edge.
    area_um2 = float((2.0 * np.pi * r[sel] * nw.length[sel]).sum())
    res["capillary_surface_density_mm2_per_mm3"] = area_um2 * 1e-6 / (vol_um3 * 1e-9)
    res["capillary_volume_fraction"] = float(
        (np.pi * r[sel] ** 2 * nw.length[sel]).sum()) / vol_um3

    # -- intercapillary distance --------------------------------------------
    # a distance to the capillary CURVE from random points in the interior, which
    # is the Krogh number: how far oxygen has to diffuse from the nearest wall.
    from scipy.spatial import cKDTree
    capillary_pts = [resample_polyline(np.asarray(nw.points[i], float), 5.0)
                     for i in np.nonzero(cap)[0]]
    capillary_pts = [p for p in capillary_pts if p.size]
    tree = cKDTree(np.vstack(capillary_pts)) if capillary_pts else None
    lo, hi = box
    probe = rng.uniform(lo, hi, size=(n_probe, 3))
    if tree is not None:
        d_probe, _ = tree.query(probe, k=1)
        res["tissue_to_capillary_um"] = {
            "mean": float(d_probe.mean()), "median": float(np.median(d_probe)),
            "p90": float(np.percentile(d_probe, 90)),
            "p99": float(np.percentile(d_probe, 99)),
            "max": float(d_probe.max()),
            "note": "distance from a uniform random tissue point to the nearest capillary "
                    "CENTRELINE, minus nothing -- subtract a ~2 um radius for the wall",
        }
        # the naive 1/sqrt(length density) estimate, for comparison
        rho_mm_mm3 = res["length_density_mm_per_mm3"]["capillary"]
        res["tissue_to_capillary_um"]["sqrt_density_estimate"] = float(
            1e3 / math.sqrt(max(rho_mm_mm3, 1e-9)))

    # -- what an angiogram could ever see ------------------------------------
    # the question tier 1 of `ibm/topologies/vascular_prior.py` turns on: 7T
    # TOF-MRA and QSM venography resolve vessels down to roughly 200-300 um, so
    # what fraction of the vasculature is that?  reported as a fraction of LENGTH
    # (which sets exchange surface and transit) and of VOLUME (which sets the
    # susceptibility a BOLD model integrates), because the two answers differ by
    # more than an order of magnitude and quoting only the second is how a
    # venogram comes to be described as showing "the vasculature".
    vol_seg = np.pi * r ** 2 * nw.length
    tot_len, tot_vol = float(nw.length.sum()), float(vol_seg.sum())
    res["resolvable_fraction_above_diameter"] = {
        f"{thr:g}um": {
            "length": float(nw.length[2 * r >= thr].sum()) / max(tot_len, 1e-12),
            "volume": float(vol_seg[2 * r >= thr].sum()) / max(tot_vol, 1e-12),
        }
        for thr in (6, 10, 20, 30, 50, 100, 200, 300)
    }

    # -- tortuosity ----------------------------------------------------------
    chord = np.linalg.norm(nw.coords[nw.tup[:, 0]] - nw.coords[nw.tup[:, 1]], axis=1)
    arc = np.array([polyline_length(np.asarray(p, float)) for p in nw.points])
    good = (chord > 1.0) & (arc > 0)
    res["tortuosity_arc_over_chord"] = {
        "all": log_stats(np.maximum(arc[good] / chord[good], 1.0)),
        "capillary": log_stats(np.maximum(arc[good & cap] / chord[good & cap], 1.0)),
        "stored_length_over_arc_median": float(np.median(
            nw.length[good] / np.maximum(arc[good], 1e-9))),
    }

    # -- depth profile -------------------------------------------------------
    # z is depth from the pial surface: the pial vessels (kinds 0 and 1) sit at
    # z ~ 0 and the penetrating vessels run to z ~ 1.2 mm, which is the cortical
    # thickness of a mouse.  no registration is involved and none is claimed.
    zmid = 0.5 * (nw.coords[nw.tup[:, 0], 2] + nw.coords[nw.tup[:, 1], 2])
    zlo, zhi = float(box[0][2]), float(box[1][2])
    edges_z = np.linspace(zlo, zhi, depth_bins + 1)
    slab_vol = float(np.prod(box[1][:2] - box[0][:2])) * (edges_z[1] - edges_z[0])
    prof = []
    for i in range(depth_bins):
        m = inside & (zmid >= edges_z[i]) & (zmid < edges_z[i + 1])
        mc = m & cap
        prof.append({
            "z_um": 0.5 * (edges_z[i] + edges_z[i + 1]),
            "capillary_length_density_mm_per_mm3":
                float(nw.length[mc].sum()) * 1e-3 / (slab_vol * 1e-9),
            "capillary_radius_um_median":
                float(np.median(r[mc])) if mc.any() else None,
            "capillary_segment_length_um_median":
                float(np.median(nw.length[mc])) if mc.any() else None,
            "n_capillary_segments": int(mc.sum()),
        })
    res["depth_profile"] = prof
    dens = np.array([p["capillary_length_density_mm_per_mm3"] for p in prof])
    ok = dens > 0
    res["depth_variation"] = {
        "log_sd_over_depth": float(np.log(dens[ok]).std(ddof=1)) if ok.sum() > 2 else None,
        "max_over_min": float(dens[ok].max() / dens[ok].min()) if ok.sum() > 2 else None,
    }

    # -- distance from a penetrating vessel ----------------------------------
    pen = np.isin(nw.kind, (2, 3))
    pen_pts = [resample_polyline(np.asarray(nw.points[i], float), 5.0)
               for i in np.nonzero(pen)[0]]
    pen_pts = [p for p in pen_pts if p.size]
    if pen_pts and tree is not None:
        ptree = cKDTree(np.vstack(pen_pts))
        d_pen, _ = ptree.query(probe, k=1)
        # bin the random probes by distance from the nearest penetrating vessel
        # and ask what the capillary geometry looks like in each shell.
        qs = np.percentile(d_pen, [0, 20, 40, 60, 80, 100])
        shells = []
        for i in range(len(qs) - 1):
            m = (d_pen >= qs[i]) & (d_pen <= qs[i + 1])
            if m.sum() < 10:
                continue
            shells.append({
                "d_from_penetrating_um": [float(qs[i]), float(qs[i + 1])],
                "n_probes": int(m.sum()),
                "tissue_to_capillary_um_mean": float(d_probe[m].mean()),
            })
        res["penetrating_vessel_shells"] = shells
        res["penetrating_vessel_spacing_um"] = {
            "mean_distance_to_nearest": float(d_pen.mean()),
            "p90": float(np.percentile(d_pen, 90)),
            "note": "twice this is roughly the spacing of the penetrating vessels, which "
                    "is the length scale a vascular unit is defined on",
        }

    # -- bifurcations and Murray ---------------------------------------------
    bif = bifurcations(nw, up, dn)
    if bif["bif_parent_r"]:
        rp = np.array(bif["bif_parent_r"], float)
        rc = np.array(bif["bif_child_r"], float)
        k = murray_exponent(rp, [rc[:, 0], rc[:, 1]])
        ratio3 = (rc ** 3).sum(axis=1) / np.maximum(rp ** 3, 1e-18)
        res["murray"] = {
            "n_bifurcations": int(rp.size),
            "n_with_a_root": int(np.isfinite(k).sum()),
            "n_child_wider_than_parent": int((~np.isfinite(k)).sum()),
            "exponent": {"median": float(np.nanmedian(k)),
                         "mean": float(np.nanmean(k)),
                         "p10": float(np.nanpercentile(k, 10)),
                         "p90": float(np.nanpercentile(k, 90)),
                         "sd": float(np.nanstd(k, ddof=1))},
            "sum_rc3_over_rp3": {"median": float(np.median(ratio3)),
                                 "log_sd": float(np.log(
                                     np.maximum(ratio3, 1e-12)).std(ddof=1)),
                                 "p10": float(np.percentile(ratio3, 10)),
                                 "p90": float(np.percentile(ratio3, 90))},
            "angle_between_children_deg": {
                "median": float(np.nanmedian(bif["bif_angle_deg"])),
                "p10": float(np.nanpercentile(bif["bif_angle_deg"], 10)),
                "p90": float(np.nanpercentile(bif["bif_angle_deg"], 90))},
            "angle_parent_to_child_deg": {
                "median": float(np.nanmedian(bif["bif_parent_child_angle_deg"])),
                "p10": float(np.nanpercentile(bif["bif_parent_child_angle_deg"], 10)),
                "p90": float(np.nanpercentile(bif["bif_parent_child_angle_deg"], 90))},
        }
        bk = np.array(bif["bif_kind"], int)
        res["murray"]["by_parent_kind"] = {}
        for kk, nm in KIND.items():
            m = bk == kk
            if m.sum() < 20:
                continue
            res["murray"]["by_parent_kind"][nm] = {
                "n": int(m.sum()),
                "exponent_median": float(np.nanmedian(k[m])),
                "sum_rc3_over_rp3_median": float(np.median(ratio3[m])),
            }
    if bif["con_parent_r"]:
        rp = np.array(bif["con_parent_r"], float)
        rc = np.array(bif["con_child_r"], float)
        kk = murray_exponent(rp, [rc[:, 0], rc[:, 1]])
        res["murray_confluences"] = {
            "n": int(rp.size),
            "exponent_median": float(np.nanmedian(kk)),
            "sum_rc3_over_rp3_median": float(
                np.median((rc ** 3).sum(axis=1) / np.maximum(rp ** 3, 1e-18))),
        }

    # -- the physics ---------------------------------------------------------
    # the FULL bounding volume here, not the eroded one.  a descending arteriole
    # runs through the whole block and perfuses the whole block, so dividing its
    # inflow by an eroded interior would inflate the CBF by exactly the erosion
    # factor -- 1.7x at a 100 um margin.  the eroded volume is the right
    # denominator for a LENGTH density, where a truncated segment really is
    # missing length, and the wrong one for a perfusion.
    full_vol_um3 = float(np.prod(nw.coords.max(axis=0) - nw.coords.min(axis=0)))
    res["full_volume_mm3"] = full_vol_um3 * 1e-9
    res["physics"] = physics(nw, inside, full_vol_um3, up, dn, r)
    return res


def physics(nw: Network, inside: np.ndarray, vol_um3: float,
            up: np.ndarray, dn: np.ndarray, r: np.ndarray) -> dict:
    """does the network carry the flow it should, at pressures it could have?

    the whole reason arm B exists.  three questions, in the order they have to be
    answered:

    **what is the flow unit?**  the release does not say.  it is recovered two
    independent ways.  (a) POISEUILLE: every segment has a solved pressure drop, a
    radius and a length, so 8 mu L Q / (pi r^4) = dP determines Q in SI given a
    viscosity, and the ratio of that to the stored number is the unit -- assuming
    only that the effective viscosity is somewhere in the plasma-to-whole-blood
    range.  (b) PERFUSION: the net flow across the pressure boundary divided by
    the tissue volume is a CBF, and cortical CBF in a mouse is known to within a
    factor of two.  the two routes share no arithmetic, so their agreement is
    evidence and their disagreement would have been reportable.

    **what viscosity does the solution imply?**  with the unit fixed, each
    segment's own dP, Q, L and r give an apparent viscosity.  plotting it against
    calibre is the Fahraeus-Lindqvist effect measured rather than quoted, and it
    is the correction `vascular_tree_adjacency` currently declines to apply and
    documents the direction of.

    **what does the tree cost?**  the arteriole-to-venule pressure drop, and how
    it is distributed over the vessel classes.  a synthesised network that
    delivers the right CBF by making every capillary enormous would be caught
    here and nowhere else.
    """
    a, b = nw.tup[:, 0], nw.tup[:, 1]
    dp_mmhg = np.abs(nw.pressure[a] - nw.pressure[b])
    r_m = np.maximum(r, 1e-3) * 1e-6
    L_m = np.maximum(nw.length, 1e-3) * 1e-6
    # Q in m^3/s that a 3.5 mPa s fluid would need to produce the solved dP
    q_ref = dp_mmhg * MMHG_PA * np.pi * r_m ** 4 / (8.0 * 3.5e-3 * L_m)
    ok = (nw.flow > 1e-6) & (dp_mmhg > 1e-4)
    # ratio of that reference (expressed in um^3/s) to the stored flow number
    ratio = (q_ref[ok] * 1e18) / nw.flow[ok]
    unit_from_poiseuille = float(np.median(ratio))

    # perfusion route.  two estimators, and which one is right matters by a
    # factor of two to seven.
    #
    # the flow crossing the PRESSURE BOUNDARY is the obvious one and it is wrong:
    # a reconstructed block is cut through its capillary bed on every lateral
    # face, so a large part of that flow enters through one cut capillary and
    # leaves through another without ever perfusing the block.  it is reported
    # as an upper bound and named as one.
    #
    # the flow entering the DESCENDING ARTERIOLES from the pial network is the
    # right one.  a descending arteriole is the sole route from the pia to the
    # capillary bed underneath it -- that is what makes it a vascular unit -- so
    # summing the flow at the top of every one of them is the blood actually
    # delivered to the tissue in the block.  the two differ by 2-7x here, and
    # the second lands in the physiological range while the first does not.
    nv = nw.coords.shape[0]
    touches_pial_artery = np.zeros(nv, bool)
    touches_pial_artery[nw.tup[nw.kind == 0].ravel()] = True
    top_of_arteriole = (nw.kind == 2) & touches_pial_artery[up]
    arteriolar_units = float(nw.flow[top_of_arteriole].sum())
    inflow_units = float(nw.flow[nw.is_bc[up]].sum())
    outflow_units = float(nw.flow[nw.is_bc[dn]].sum())

    # 1 unit = U um^3/s  =>  CBF [ml/100g/min] = Q*U [um^3/s] * 60 / vol_um3 * 100
    # (blood and tissue both ~1 g/ml, so ml/ml/min * 100 = ml/100g/min)
    def cbf(unit_um3_s: float, q_units: float = None) -> float:
        q = arteriolar_units if q_units is None else q_units
        return q * unit_um3_s * 60.0 / vol_um3 * 100.0

    # the unit is a round power of ten by construction of any sane release, so
    # both routes are reported against the decade they land in rather than being
    # quietly rounded into agreement.
    unit_guess = 10.0 ** round(math.log10(max(unit_from_poiseuille, 1e-30)))
    out = {
        "flow_unit_um3_per_s": {
            "from_poiseuille_at_mu_3.5mPas": unit_from_poiseuille,
            "adopted": unit_guess,
            "cbf_at_adopted_ml_per_100g_per_min": cbf(unit_guess),
            "cbf_one_decade_lower": cbf(unit_guess / 10.0),
            "cbf_one_decade_higher": cbf(unit_guess * 10.0),
            "note": "the release documents no unit.  the Poiseuille route assumes only "
                    "that the effective viscosity is in the plasma-to-whole-blood range; "
                    "the perfusion route assumes only that mouse cortical CBF is of order "
                    "100 ml/100g/min.  a decade either way is ruled out by the second.",
        },
        "perfusion": {
            "n_descending_arterioles_fed_from_the_pia": int(top_of_arteriole.sum()),
            "arteriolar_inflow_units": arteriolar_units,
            "cbf_from_descending_arterioles_ml_per_100g_per_min": cbf(unit_guess),
            "boundary_inflow_units": inflow_units,
            "boundary_outflow_units": outflow_units,
            "cbf_from_pressure_boundary_ml_per_100g_per_min":
                cbf(unit_guess, inflow_units),
            "note": "the boundary figure is an UPPER bound and includes blood that "
                    "enters one cut capillary at a lateral face and leaves by another "
                    "without perfusing anything.  the arteriolar figure is the one to "
                    "read; that in == out at the boundary to machine precision is a "
                    "check that the solution is conservative, not a perfusion estimate",
        },
    }

    # apparent viscosity implied per segment, at the adopted unit
    q_si = nw.flow * unit_guess * 1e-18
    mu_app = np.where(q_si > 0,
                      dp_mmhg * MMHG_PA * np.pi * r_m ** 4 / (8.0 * np.maximum(q_si, 1e-30) * L_m),
                      np.nan)
    d_um = 2.0 * r
    bins = [(0, 4), (4, 6), (6, 9), (9, 15), (15, 30), (30, 1e9)]
    fl = []
    for lo, hi in bins:
        m = ok & (d_um >= lo) & (d_um < hi)
        if m.sum() < 20:
            continue
        fl.append({"diameter_um": [lo, min(hi, 1e6)], "n": int(m.sum()),
                   "apparent_viscosity_mPa_s": float(np.median(mu_app[m]) * 1e3)})
    out["fahraeus_lindqvist"] = {
        "by_diameter": fl,
        "note": "median apparent viscosity implied by each segment's own solved pressure "
                "drop.  this is the correction vascular_tree_adjacency declines to apply "
                "and documents the direction of; the direction is confirmed here",
    }

    # pressure drop over the tree
    out["pressure_mmHg"] = {
        "min": float(nw.pressure.min()), "max": float(nw.pressure.max()),
        "arteriole_to_venule_drop": float(nw.pressure.max() - nw.pressure.min()),
    }
    drop = {}
    for k, nm in KIND.items():
        m = (nw.kind == k) & ok
        if m.sum() < 10:
            continue
        pv = nw.pressure[nw.tup[m].ravel()]
        drop[nm] = {"n": int(m.sum()),
                    "median_segment_drop_mmHg": float(np.median(dp_mmhg[m])),
                    "vertex_pressure_mmHg": {"p10": float(np.percentile(pv, 10)),
                                             "median": float(np.median(pv)),
                                             "p90": float(np.percentile(pv, 90))}}
    out["pressure_drop_by_kind"] = drop

    # capillary transit time straight out of the data: the blood volume in the
    # capillary bed divided by the flow through it.  this is the number a
    # windkessel's `tau_transit_s` is a prior over, and it is measured here rather
    # than quoted -- it is also the quantity that decides whether a synthesised
    # network can deliver oxygen at all, since extraction is a race between
    # transit and diffusion.
    cap_all = nw.kind == CAPILLARY_KIND
    v_cap_um3 = float((np.pi * r[cap_all] ** 2 * nw.length[cap_all]).sum())
    q_art_um3_s = arteriolar_units * unit_guess
    out["capillary_transit_time_s"] = {
        "value": v_cap_um3 / max(q_art_um3_s, 1e-30),
        "capillary_volume_mm3": v_cap_um3 * 1e-9,
        "note": "capillary blood volume over arteriolar inflow.  a mean, and the "
                "heterogeneity around it is what raises effective extraction above what "
                "the mean predicts -- see the validity note on `tissue_exchange`",
    }

    # what fraction of the whole tree's resistance sits in the capillary bed --
    # the number the `vascular` topology's docstring asserts and never measured.
    R = np.where(q_si > 0, dp_mmhg * MMHG_PA / np.maximum(q_si, 1e-30), np.nan)
    cap = nw.kind == CAPILLARY_KIND
    with np.errstate(invalid="ignore"):
        # power dissipated is dP * Q, which is the additive quantity over a
        # network with loops -- a sum of resistances is not, because the
        # capillary bed is massively parallel.
        power = dp_mmhg * MMHG_PA * q_si
        tot = float(np.nansum(power))
        out["dissipation_share"] = {
            nm: float(np.nansum(power[nw.kind == k]) / max(tot, 1e-30))
            for k, nm in KIND.items() if (nw.kind == k).any()
        }
        out["dissipation_share"]["note"] = (
            "share of total viscous dissipation dP*Q, not of series resistance: the "
            "capillary bed is massively parallel and a sum of segment resistances is "
            "not the resistance of anything")
    return out


# ---------------------------------------------------------------------------
# arm A: VesselGraph whole-brain graphs
# ---------------------------------------------------------------------------


def _find_member(z: zipfile.ZipFile, suffix: str) -> str | None:
    for n in z.namelist():
        if n.endswith(suffix) and not n.startswith("__MACOSX"):
            return n
    return None


#: the light-sheet voxel every VesselGraph archive is named for.
VOXEL_UM = 3.0

#: columns actually read out of a 600 MB edge table.  reading the other twelve
#: costs a minute per animal and answers nothing.
EDGE_COLS = ["node1id", "node2id", "length", "distance", "curveness",
             "avgRadiusAvg", "volume", "avgCrossSection", "num_voxels",
             "hasNodeAtSampleBorder"]


def vesselgraph_units(edges) -> dict:
    """which unit each column is in, decided from the data rather than assumed.

    the release documents none of it and the answer is not uniform, which is
    exactly why it has to be settled before any number is quoted.  three
    consistency checks, each of which one reading passes and the other fails:

    1. **length against `num_voxels`.**  a segment's `length` divided by the
       number of centreline voxels it occupies comes out at about 1.4 -- the mean
       step of a 3D digital curve mixing face, edge and corner moves, which are
       1, sqrt(2) and sqrt(3) long.  that ratio is a dimensionless property of a
       voxelised curve, so `length` is counted in VOXELS.  read as microns it
       would make a segment shorter than the voxels it is stored in.

    2. **total vascular length.**  at 3 um per unit the whole-brain length
        density lands within a factor of two of what arm B measures directly in
        cortex; read as microns it is six times too low.

    3. **blood volume fraction.**  with the radius ALSO in voxels, sum(pi r^2 L)
       over the brain is 10-14% of the tissue, which is three times any measured
       cerebral blood volume and is the reading that has to be rejected.  with the
       radius in microns it is 1-2%, which is the right order, and the median
       calibre lands 1.6x above arm B's directly measured capillary radius --
       which is the size of over-estimate a 3 um light-sheet voxel produces on a
       5 um vessel and is in the expected direction.

    so: **length, distance and volume in voxels; avgRadiusAvg in microns.**  a
    mixed convention is strange, and it is reported as a finding rather than
    smoothed over, because every density and every resistance below depends on
    it and a reader who disagrees needs to know exactly what to change.
    """
    L = np.asarray(edges["length"], float)
    nv = np.asarray(edges["num_voxels"], float)
    ok = (L > 0) & (nv > 0)
    return {
        "length_per_centreline_voxel": float(np.median(L[ok] / nv[ok])),
        "expected_for_a_digital_curve": "1.0 to 1.73, mean ~1.4",
        "length_unit": "voxel", "voxel_um": VOXEL_UM,
        "radius_unit": "um",
        "note": vesselgraph_units.__doc__.split("three")[0].strip(),
    }


def load_vesselgraph(zip_path: Path) -> dict | None:
    """node and edge tables out of one VesselGraph raw archive.

    the release ships semicolon-separated csv written by Voreen: a node list with
    positions and degree, and an edge list with, per segment, its two nodes, arc
    length, chord distance, curveness and several radius estimates.
    `avgRadiusAvg` is the one used here -- the mean along the segment of the
    maximal-inscribed-sphere radius -- because it is the estimate the dataset's
    own preprocessing uses and because a min or a max radius over a segment is an
    extremum statistic whose expectation depends on how finely the segment was
    sampled.

    a radius of -1 appears and means "not computed"; those edges are dropped and
    counted rather than clipped to zero, which would put a spike at the bottom of
    every radius distribution.
    """
    import pandas as pd
    with zipfile.ZipFile(zip_path) as z:
        nn = _find_member(z, "_nodes.csv")
        ee = _find_member(z, "_edges.csv")
        if not nn or not ee:
            return None
        edges = pd.read_csv(z.open(ee), sep=";", usecols=EDGE_COLS, engine="c")
        nodes = pd.read_csv(z.open(nn), sep=";",
                            usecols=["pos_x", "pos_y", "pos_z", "degree",
                                     "isAtSampleBorder"], engine="c")
    return {"nodes": nodes, "edges": edges, "node_file": nn, "edge_file": ee}


def analyse_vesselgraph(tbl: dict, name: str, *, capillary_radius_um: float) -> dict:
    nodes, edges = tbl["nodes"], tbl["edges"]
    res: dict = {"name": name, "edge_file": tbl["edge_file"],
                 "n_nodes": int(len(nodes)), "n_edges": int(len(edges))}
    res["units"] = vesselgraph_units(edges)

    rad = np.asarray(edges["avgRadiusAvg"], float)                  # um
    length = np.asarray(edges["length"], float) * VOXEL_UM          # -> um
    dist = np.asarray(edges["distance"], float) * VOXEL_UM
    curve = np.asarray(edges["curveness"], float)
    n1 = np.asarray(edges["node1id"], np.int64)
    n2 = np.asarray(edges["node2id"], np.int64)
    good = np.isfinite(rad) & (rad > 0) & (length > 0)
    res["n_edges_without_a_radius"] = int((~good).sum())

    rad, length, dist, curve = rad[good], length[good], dist[good], curve[good]
    res["radius_um"] = log_stats(rad)
    res["segment_length_um"] = log_stats(length)
    okd = dist > 0
    res["tortuosity_length_over_distance"] = log_stats(
        np.maximum(length[okd] / dist[okd], 1.0))
    res["curveness_as_shipped"] = log_stats(curve[np.isfinite(curve) & (curve > 0)])

    cap = rad <= capillary_radius_um
    res["capillary_fraction_of_segments"] = float(cap.mean())
    res["capillary_fraction_of_length"] = float(length[cap].sum() / length.sum())
    res["capillary"] = {"radius_um": log_stats(rad[cap]),
                        "segment_length_um": log_stats(length[cap])}

    vol_seg = np.pi * rad ** 2 * length
    tot_len, tot_vol = float(length.sum()), float(vol_seg.sum())
    res["resolvable_fraction_above_diameter"] = {
        f"{thr:g}um": {"length": float(length[2 * rad >= thr].sum()) / tot_len,
                       "volume": float(vol_seg[2 * rad >= thr].sum()) / tot_vol}
        for thr in (6, 10, 20, 30, 50, 100, 200, 300)}

    pos = nodes[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_UM
    bbox = np.stack([pos.min(axis=0), pos.max(axis=0)])
    vol_um3 = float(np.prod(bbox[1] - bbox[0]))
    res["bbox_um"] = bbox.tolist()
    res["bounding_volume_mm3"] = vol_um3 * 1e-9
    # a whole-brain bounding box is roughly 30% not brain, so these are LOWER
    # bounds on the tissue density by exactly the fill fraction, which this
    # release does not ship a mask for.  arm B's cortical blocks are where a real
    # tissue density comes from; what this is for is the between-animal spread,
    # which the fill fraction largely divides out of.
    res["length_density_mm_per_mm3_over_bbox"] = {
        "capillary": float(length[cap].sum()) * 1e-3 / (vol_um3 * 1e-9),
        "all_vessels": tot_len * 1e-3 / (vol_um3 * 1e-9),
        "blood_volume_fraction": tot_vol / vol_um3,
        "note": "over the BOUNDING BOX, which a brain fills to maybe 0.7.  a lower "
                "bound on the tissue density by that factor",
    }

    deg = np.asarray(nodes["degree"], np.int64)
    res["degree"] = {str(d): int((deg == d).sum()) for d in range(0, 7)}
    res["degree"]["ge7"] = int((deg >= 7).sum())
    res["degree"]["mean"] = float(deg.mean())

    # -- Murray at degree-3 nodes -------------------------------------------
    # the release carries no flow and no vessel type, so the tree has no
    # orientation and the parent has to be guessed.  the widest of the three is
    # the only defensible guess, and it is biased: taking the max of three noisy
    # radii inflates the parent, which shrinks every rc/rp, which -- since
    # sum (rc/rp)^k is decreasing in k -- drags the fitted exponent DOWN.
    #
    # the exponent nonetheless comes out HIGHER here than in arm B, where the
    # parent is known from the solved pressure field, so that bias does not
    # explain the gap and something else does.  the something else is the
    # resolution floor: a 3 um light-sheet voxel cannot separate a 4 um vessel
    # from a 6 um one, so the three radii at a junction are compressed toward each
    # other, and a junction whose children are nearly as wide as its parent needs
    # a very large k to satisfy the law at all -- as rc/rp -> 1 the root goes to
    # infinity.  arm B is the control and the gap is the measurement error, not a
    # difference in anatomy.
    nmax = int(max(n1.max(), n2.max())) + 1
    ends = np.concatenate([n1[good], n2[good]])
    eid = np.tile(np.arange(rad.size), 2)
    order = np.argsort(ends, kind="stable")
    vs, es = ends[order], eid[order]
    starts = np.searchsorted(vs, np.arange(nmax))
    counts = np.diff(np.append(starts, len(vs)))
    three = np.nonzero(counts == 3)[0]
    if three.size:
        cap_n = 400_000
        if three.size > cap_n:
            three = np.random.default_rng(0).choice(three, cap_n, replace=False)
        trio = np.stack([es[starts[three]], es[starts[three] + 1],
                         es[starts[three] + 2]], axis=1)
        rr = np.sort(rad[trio], axis=1)[:, ::-1]
        k = murray_exponent(rr[:, 0], [rr[:, 1], rr[:, 2]])
        ratio3 = (rr[:, 1] ** 3 + rr[:, 2] ** 3) / np.maximum(rr[:, 0] ** 3, 1e-18)
        res["murray"] = {
            "n_junctions": int(rr.shape[0]),
            "subsampled_from": int((counts == 3).sum()),
            "n_with_a_root": int(np.isfinite(k).sum()),
            "parent_assumed": "widest of the three (this release carries no flow "
                              "direction), which biases the exponent DOWN",
            "exponent": {"median": float(np.nanmedian(k)),
                         "p10": float(np.nanpercentile(k, 10)),
                         "p90": float(np.nanpercentile(k, 90))},
            "sum_rc3_over_rp3": {
                "median": float(np.median(ratio3)),
                "log_sd": float(np.log(np.maximum(ratio3, 1e-12)).std(ddof=1))},
        }
    return res


# ---------------------------------------------------------------------------
# arm C: what a human population angiogram actually pins down
# ---------------------------------------------------------------------------


def analyse_venat(root: Path) -> dict:
    """the VENAT 7T QSM venous atlas, read for its SPREAD rather than its mean.

    the only human source in this measurement, and it is here for one number:
    how much two people's macro venous trees differ.  that is what tier 2 of
    `vascular_prior` costs, and until now it was a judgement.

    VENAT ships a mean and a standard deviation for vessel diameter, curvature
    and torsion over its cohort, on the skeleton of the population venous tree in
    MNI at 0.5 mm.  the ratio of the two, per voxel, is the between-subject
    coefficient of variation of the calibre at a location the atlas is confident
    enough about to have a skeleton at all -- which makes it a LOWER bound twice
    over: the atlas only has a skeleton where subjects agreed, and a
    registration-based atlas blurs any vessel whose position varies into a lower
    mean rather than a higher variance.

    the partial-volume map carries the same warning the source card already
    makes: a diameter here is the output of a threshold somebody chose applied to
    a regularised dipole inversion, not a lumen.  what is used is therefore the
    RATIO, which is insensitive to a multiplicative error in the threshold, and
    not the absolute calibre.
    """
    import nibabel as nib
    def load(name):
        f = root / f"VENAT_{name}.nii.gz"
        return np.asanyarray(nib.load(str(f)).dataobj).astype(float) if f.is_file() else None

    d, ds = load("diameter"), load("diameter_std")
    if d is None or ds is None:
        return {"error": "VENAT diameter maps not found"}
    pv, pvs, dens = load("PartialVolume"), load("PartialVolume_std"), load("DensityMap")
    m = np.isfinite(d) & (d > 0) & np.isfinite(ds) & (ds > 0)
    cv = ds[m] / d[m]
    out = {
        "n_skeleton_voxels": int(m.sum()),
        "voxel_mm": 0.5,
        "diameter_mm": {"median": float(np.median(d[m])), "p10": float(np.percentile(d[m], 10)),
                        "p90": float(np.percentile(d[m], 90)), "max": float(d[m].max())},
        "between_subject_cv_of_diameter": {
            "median": float(np.median(cv)), "mean": float(cv.mean()),
            "p10": float(np.percentile(cv, 10)), "p90": float(np.percentile(cv, 90))},
        "between_subject_log_sd_of_diameter": float(np.median(np.sqrt(np.log1p(cv ** 2)))),
    }
    for nm, a, b in (("curvature", *(load("curvature"), load("curvature_std"))),
                     ("torsion", *(load("torsion"), load("torsion_std")))):
        if a is None or b is None:
            continue
        k = np.isfinite(a) & (a > 0) & np.isfinite(b) & (b > 0)
        out[f"between_subject_cv_of_{nm}"] = float(np.median(b[k] / a[k]))
    if pv is not None:
        vox = 0.5 ** 3
        out["venous_volume_fraction"] = {
            "voxels_with_any_venous_pv": int((pv > 0.01).sum()),
            "mean_pv_where_present": float(pv[pv > 0.01].mean()),
            "total_venous_volume_ml": float(pv[np.isfinite(pv)].sum() * vox / 1000.0),
        }
        if pvs is not None:
            k = (pv > 0.05) & np.isfinite(pvs)
            out["venous_volume_fraction"]["between_subject_cv_of_pv"] = float(
                np.median(pvs[k] / pv[k]))
    if dens is not None:
        k = np.isfinite(dens) & (dens > 0)
        out["density_map"] = {"n_voxels": int(k.sum()),
                              "median": float(np.median(dens[k]))}
    out["read_it_as"] = (
        "a population venous tree, and a LOWER bound on how much two people's "
        "macro vasculature differs: the atlas has a skeleton only where subjects "
        "agreed enough to leave one, and registration turns positional variability "
        "into blur rather than into variance")
    return out


# ---------------------------------------------------------------------------
# between-animal correlation and the effective count
# ---------------------------------------------------------------------------


def effective_animals(features: dict[str, np.ndarray], component: str,
                      groups: dict[str, str] | None = None) -> dict:
    """how many independent animals nine animals are.

    each animal contributes one feature vector; the group consensus is the mean;
    each animal's deviation from it is its error vector; and the mean pairwise
    correlation of those deviations is the fraction of error the animals share.
    that is `correlated_fraction`, it is the same quantity a source card declares
    for a teacher, and it is converted to an effective count by the same call
    into `ibm.runtime.fuse` that `TractUncertainty.effective_pipelines` makes --
    not a reimplementation, because a second copy of this arithmetic is a second
    place to get the sign backwards.

    **the null is not zero and this is not a detail.**  deviations from a sample
    mean of n are linearly dependent by construction: they sum to zero, so their
    expected pairwise correlation is -1/(n-1) even when the animals are perfectly
    independent.  worse, an exchangeable shared component is removed EXACTLY by
    the centring, so a uniform bias across all n animals produces the same
    -1/(n-1) as no bias at all.  the raw correlation is therefore reported
    alongside its null, and the reported `correlated_fraction` is the EXCESS over
    that null rescaled onto [0, 1),

        rho = max(0, (c_raw - c_null) / (1 - c_null)),   c_null = -1/(n-1)

    which at n = 3 means the estimator has almost no dynamic range and says so.
    what the excess actually measures is SUBGROUP structure -- animals that
    resemble each other more than they resemble the consensus, which is what a
    strain or a processing batch produces -- and not the global bias, which is
    invisible here by construction.  that is the same lower-bound caveat the
    TractoInferno measurement carries and it is stated for the same reason.

    `groups` -- a strain or batch label per animal -- buys the measurement its
    dynamic range back, because a between-group variance is NOT removed by
    centring on the grand mean.  when it is supplied the group ICC is computed as
    well, and it is the number to read when the excess correlation is near zero.
    """
    names = sorted(features)
    if len(names) < 3:
        return {"n_animals": len(names), "note": "fewer than three animals; rho undefined"}
    X = np.stack([features[n] for n in names])           # (n_animals, n_features)
    keep = np.all(np.isfinite(X), axis=0)
    X = X[:, keep]
    if X.shape[1] < 3:
        return {"n_animals": len(names), "note": "fewer than three shared features"}
    n = len(names)
    D = X - X.mean(axis=0, keepdims=True)
    sd = D.std(axis=0, ddof=1)
    sd[sd <= 0] = 1.0
    D = D / sd
    C = np.corrcoef(D)
    iu = np.triu_indices(n, 1)
    c_raw = float(np.nanmean(C[iu]))
    c_null = -1.0 / (n - 1)
    rho = float(np.clip((c_raw - c_null) / (1.0 - c_null), 0.0, 1.0 - 1e-9))
    out = {"n_animals": n, "n_features": int(X.shape[1]),
           "mean_pairwise_correlation_of_deviations": c_raw,
           "null_for_independent_animals": c_null,
           "correlated_fraction": rho}

    # -- the group arm.  a between-strain component survives grand-mean centring
    # and is the honest lower bound the deviation correlation cannot see.
    if groups:
        g = np.array([groups.get(nm, "?") for nm in names])
        labs = sorted(set(g))
        if 1 < len(labs) < n:
            Z = (X - X.mean(axis=0, keepdims=True)) / np.maximum(
                X.std(axis=0, ddof=1), 1e-12)
            grand = Z.mean(axis=0)
            gm = {L: Z[g == L].mean(axis=0) for L in labs}
            ss_between = float(sum((g == L).sum() * ((gm[L] - grand) ** 2).sum()
                                   for L in labs))
            ss_within = float(sum(((Z[i] - gm[g[i]]) ** 2).sum() for i in range(n)))
            k = len(labs)
            ms_b = ss_between / max(k - 1, 1)
            ms_w = ss_within / max(n - k, 1)
            n0 = n / k
            icc = (ms_b - ms_w) / max(ms_b + (n0 - 1) * ms_w, 1e-12)
            out["group_icc"] = {
                "groups": {L: int((g == L).sum()) for L in labs},
                "icc": float(np.clip(icc, 0.0, 1.0)),
                "between_group_ms": ms_b, "within_group_ms": ms_w,
                "note": "one-way random-effects ICC over the group label.  unlike the "
                        "deviation correlation above, a between-group term is NOT removed "
                        "by centring on the grand mean, so this is the estimator with "
                        "dynamic range at small n",
            }
            rho = max(rho, float(np.clip(icc, 0.0, 1.0 - 1e-9)))
            out["correlated_fraction_used"] = rho
    try:
        from ibm.runtime.fuse import TeacherPrecision
        tp = TeacherPrecision(r2=0.0, correlated_fraction=rho, error_rank=1,
                              source="microvascular reconstructions")
        ev = tp.evidence(component, np.zeros(n), np.ones(n))
        out["effective_animals"] = float(ev.effective_constraints())
    except Exception as exc:                                    # pragma: no cover
        out["effective_animals"] = n / ((1 - rho) + n * rho)
        out["fuse_error"] = str(exc)
    out["ceiling_at_any_n"] = (1.0 / rho) if rho > 0 else None
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow-root", type=Path,
                    default=Path("/home/brandonin/Documents/IBM-1/data/sources/"
                                 "microscopy-microvascular-networks/extracted"),
                    help="where NW*_results/ live")
    ap.add_argument("--graph-root", type=Path,
                    default=Path("/home/brandonin/Documents/IBM-1/data/sources/"
                                 "vesselgraph-mouse/zips"),
                    help="where the VesselGraph *_raw.zip archives live")
    ap.add_argument("--venat-root", type=Path,
                    default=Path("/home/brandonin/Documents/IBM-1/data/sources/venat/"
                                 "figshare-7205960"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--margin-um", type=float, default=100.0)
    ap.add_argument("--probes", type=int, default=200_000)
    ap.add_argument("--depth-bins", type=int, default=12)
    ap.add_argument("--capillary-radius-um", type=float, default=4.0,
                    help="calibre cut for 'capillary' in arm A, which carries no vessel "
                         "type labels.  4 um is the upper end of a mouse cortical "
                         "capillary and the cut arm B's labelled bed justifies")
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    out: dict = {"measured_at": time.strftime("%Y-%m-%d"), "sources": [], "arms": {}}

    # ---- arm B ------------------------------------------------------------
    flow_nets = {}
    for nm in ("NW1", "NW2", "NW3"):
        p = args.flow_root / f"{nm}_results"
        if not p.is_dir():
            continue
        print(f"[arm B] {nm} ...", file=sys.stderr, flush=True)
        nw = load_network(args.flow_root, nm)
        flow_nets[nm] = analyse_flow_network(
            nw, margin_um=args.margin_um, n_probe=args.probes,
            depth_bins=args.depth_bins, rng=rng)
    if flow_nets:
        out["sources"].append(
            "microscopy-microvascular-networks: Blinder et al. 2013 mouse vibrissa "
            "cortex networks with the Schmid/Weber flow solution (Zenodo 269650, CC BY 4.0)")
        out["arms"]["cortical_networks_with_flow"] = {
            "per_animal": flow_nets,
            "between_animal": {
                "capillary_length_density_mm_per_mm3": between_animal(
                    {k: v["length_density_mm_per_mm3"]["capillary"]
                     for k, v in flow_nets.items()}),
                "capillary_surface_density_mm2_per_mm3": between_animal(
                    {k: v["capillary_surface_density_mm2_per_mm3"]
                     for k, v in flow_nets.items()}),
                "tissue_to_capillary_um_mean": between_animal(
                    {k: v["tissue_to_capillary_um"]["mean"] for k, v in flow_nets.items()
                     if "tissue_to_capillary_um" in v}),
                "capillary_radius_um_median": between_animal(
                    {k: v["by_kind"]["capillary"]["radius_um"]["median"]
                     for k, v in flow_nets.items()}),
                "capillary_segment_length_um_median": between_animal(
                    {k: v["by_kind"]["capillary"]["length_um"]["median"]
                     for k, v in flow_nets.items()}),
                "murray_exponent_median": between_animal(
                    {k: v["murray"]["exponent"]["median"] for k, v in flow_nets.items()
                     if "murray" in v}),
                "cbf_ml_per_100g_per_min": between_animal(
                    {k: v["physics"]["perfusion"]
                     ["cbf_from_descending_arterioles_ml_per_100g_per_min"]
                     for k, v in flow_nets.items()}),
            },
        }
        # feature vector for the correlation: the depth profile of capillary
        # density in log units, which is the thing a prior is actually over.
        feats = {}
        for k, v in flow_nets.items():
            d = np.array([p["capillary_length_density_mm_per_mm3"]
                          for p in v["depth_profile"]], float)
            feats[k] = np.log(np.where(d > 0, d, np.nan))
        out["arms"]["cortical_networks_with_flow"]["effective_animals"] = \
            effective_animals(feats, "structural.capillary_density")

    # ---- arm A ------------------------------------------------------------
    graphs = {}
    zips = sorted(args.graph_root.glob("*_raw.zip")) if args.graph_root.is_dir() else []
    for zp in zips:
        name = zp.name.replace("_raw.zip", "")
        print(f"[arm A] {name} ...", file=sys.stderr, flush=True)
        try:
            tbl = load_vesselgraph(zp)
        except Exception as exc:
            graphs[name] = {"name": name, "error": f"{type(exc).__name__}: {exc}"}
            continue
        if tbl is None:
            graphs[name] = {"name": name, "error": "no node/edge csv in archive"}
            continue
        graphs[name] = analyse_vesselgraph(
            tbl, name, capillary_radius_um=args.capillary_radius_um)
        del tbl
    ok = {k: v for k, v in graphs.items() if "error" not in v and "radius_um" in v}
    if ok:
        out["sources"].append(
            "vesselgraph-mouse: Paetzold et al. 2021 whole-brain mouse vessel graphs, "
            "9 animals from 3 strains (CC BY-NC 4.0)")
        out["arms"]["whole_brain_graphs"] = {
            "per_animal": graphs,
            "between_animal": {
                "radius_um_median": between_animal(
                    {k: v["radius_um"]["median"] for k, v in ok.items()}),
                "segment_length_um_median": between_animal(
                    {k: v["segment_length_um"]["median"] for k, v in ok.items()}),
                "tortuosity_median": between_animal(
                    {k: v["tortuosity_length_over_distance"]["median"]
                     for k, v in ok.items() if "tortuosity_length_over_distance" in v}),
                "murray_exponent_median": between_animal(
                    {k: v["murray"]["exponent"]["median"] for k, v in ok.items()
                     if "murray" in v}),
                "n_edges": between_animal({k: float(v["n_edges"]) for k, v in ok.items()}),
            },
        }
        feats = {}
        for k, v in ok.items():
            f = [v["radius_um"]["median"], v["radius_um"]["p10"], v["radius_um"]["p90"],
                 v["segment_length_um"]["median"], v["segment_length_um"]["p10"],
                 v["segment_length_um"]["p90"], v["radius_um"]["log_sd"],
                 v["segment_length_um"]["log_sd"]]
            if "murray" in v:
                f += [v["murray"]["exponent"]["median"],
                      v["murray"]["sum_rc3_over_rp3"]["median"]]
            if "tortuosity_length_over_distance" in v:
                f += [v["tortuosity_length_over_distance"]["median"],
                      v["tortuosity_length_over_distance"]["p90"]]
            feats[k] = np.log(np.abs(np.array(f, float)) + 1e-12)
        strains = {k: k.rsplit("_no", 1)[0] for k in ok}
        out["arms"]["whole_brain_graphs"]["effective_animals"] = \
            effective_animals(feats, "structural.lumen_radius", groups=strains)

    # ---- arm C ------------------------------------------------------------
    if args.venat_root.is_dir():
        print("[arm C] VENAT ...", file=sys.stderr, flush=True)
        v = analyse_venat(args.venat_root)
        if "error" not in v:
            out["sources"].append(
                "venat: Huck et al. 7T QSM venous atlas over 20 subjects x 5 "
                "measurements, MNI 0.5 mm (figshare 7205960, CC BY 4.0)")
        out["arms"]["human_population_venogram"] = v

    out["runtime_s"] = round(time.time() - t0, 1)
    txt = json.dumps(out, indent=1, sort_keys=False, default=float)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
        print(f"wrote {args.out} ({len(txt)} bytes) in {out['runtime_s']}s", file=sys.stderr)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
