"""how wrong is euclidean distance about this cortex -- locally, and along a fascicle?

Two measurements, one question, because `ibm/topologies/tract.py` opens by
arguing that euclidean, geodesic-on-the-sheet and arc-length-along-a-fascicle are
three different metrics and that using the wrong one produces errors in opposite
directions.  The spherical proxy could not raise the question at all: on a sphere
all three agree up to a chord-vs-arc correction.

**PART 1 -- the local graph.  Is a euclidean k-NN still local on a folded sheet?**

A question the spherical proxy could not raise. On a sphere euclidean and
geodesic agree to within the chord-vs-arc correction, so `knn_edges` — which is
a `torch.cdist` in the volume — was a local sheet by construction. On a real
white surface it is not obviously anything: two points on opposite banks of a
sulcus are millimetres apart in the volume and far apart along the sheet, and
`ibm/topologies/tract.py` opens by insisting that these metrics disagree and that
getting the wrong one produces errors in opposite directions.

So this measures the error rather than assuming it is small. For a sample of
sites it computes the true along-sheet distance to each of that site's k-NN
partners — Dijkstra over the fsaverage mesh, which is the surface's own
connectivity — and reports `geodesic / euclidean` per edge.

**The gate is the whole reason this file is trustworthy.** A pair of
mesh-ADJACENT vertices has geodesic exactly equal to euclidean, so the ratio must
print 1.0000. The first version of this measurement printed **2.0000** for every
adjacent pair: each undirected mesh edge appears once per adjacent triangle, and
`csr_matrix` SUMS duplicate entries, so every edge weight was doubled and every
geodesic with it. Without the gate that would have been reported as "the median
local edge is 2.3x longer along the sheet than through the volume", which is a
different and false conclusion. CLAUDE.md: check a metric against a case whose
answer you know.

Mesh-graph distance still slightly OVERestimates the true geodesic, because a
path must follow edges rather than cut across faces. On fsaverage the mean edge
is 0.72 mm and the sites here are ~4 mm apart, so that bias is a few per cent and
in the conservative direction: it inflates the reported error.

**PART 2 -- the long-range graph.  Is a fascicle's arc length recoverable from
the straight line between its endpoints?**  `--connectome` compares every
declared edge's `fiber_length_mean` against two euclidean quantities.

Against the **centroid-to-centroid chord** the arc comes out SHORTER for 90% of
pairs, which is not a shortcut through the skull: it is a bias in the comparison.
Streamlines terminate at the parts of two parcels that face each other, not at
their centroids, so the centroid chord systematically overstates the endpoint
separation.  The number is reported because seeing it and then explaining it is
the difference between a caveat and an artefact presented as a result.

Against the **minimum border-to-border distance** -- the shortest straight line
between any point of one parcel and any point of the other -- the comparison is
unbiased in the useful direction, because that quantity is a genuine LOWER BOUND
on the length of any path between them.  It is therefore also a gate: a ratio
below 1 would mean the connectome reports a fascicle shorter than the shortest
line its own endpoints admit, i.e. a corrupted length column.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import torch

_SP = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(_SP)
_SP.loader.exec_module(P)

import ibm.cortical_sheet as CS      # noqa: E402


def mesh_graph(hemi: str, surface: str = "white"):
    """(vertices, unique undirected edges, symmetric weighted adjacency)."""
    import nibabel.freesurfer as fsio
    from scipy.sparse import csr_matrix
    v, f = fsio.read_geometry(f"{CS.FSAVERAGE}/surf/{hemi}.{surface}")
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    # DEDUPLICATE.  every interior edge is in two triangles, and csr_matrix adds
    # duplicate entries rather than replacing them -- see the module docstring.
    e = np.unique(np.sort(e, 1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    G = csr_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
        shape=(len(v), len(v)))
    return v, e, G


def gate(v, e, G, rng, n: int = 5) -> float:
    """mesh-adjacent pairs; the ratio must be 1.0."""
    from scipy.sparse.csgraph import dijkstra
    src = rng.choice(len(v), n, replace=False)
    D = dijkstra(G, indices=src, limit=80.0)
    worst = 0.0
    for r, s in enumerate(src):
        nb = e[e[:, 0] == s][:, 1]
        if not len(nb):
            continue
        eu = np.linalg.norm(v[nb] - v[s], axis=1)
        worst = max(worst, float(np.abs(D[r, nb] / eu - 1.0).max()))
    return worst


def connectome_metric_disagreement(threshold: float = 0.5) -> dict:
    """arc length along a fascicle against two euclidean surrogates for it."""
    from scipy.spatial import cKDTree
    import ibm.cortical_tracts as CT
    sheet = CS.load_sheet()
    xyz, lab, area = sheet["xyz"], sheet["region"], sheet["area"]
    n = len(CS.REGIONS)
    pts = [xyz[lab == i] for i in range(n)]
    cent = np.stack([(xyz[lab == i] * area[lab == i, None]).sum(0)
                     / area[lab == i].sum() for i in range(n)])
    c = CT.consensus(threshold)
    A, L = c["adjacency"], c["length_mm"]
    i, j = np.nonzero(np.triu(A))
    trees = {k: cKDTree(pts[k]) for k in set(i.tolist()) | set(j.tolist())}
    border = np.array([trees[b].query(pts[a], k=1, workers=-1)[0].min()
                       for a, b in zip(i, j)])
    arc = L[i, j]
    chord = np.linalg.norm(cent[i] - cent[j], axis=1)
    rb = arc / np.maximum(border, 1e-6)
    # THE GATE: the minimum border-to-border distance lower-bounds any path, so a
    # ratio below 1 is a corrupted length column, not a fast fibre.
    assert (rb >= 1.0).all(), (
        f"{int((rb < 1).sum())} declared edges report an arc SHORTER than the "
        "shortest straight line their own endpoints admit")
    return {
        "threshold": float(threshold), "n_edges": int(len(i)),
        "arc_over_min_border": {f"p{int(100*q):02d}": float(np.quantile(rb, q))
                                for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95)},
        "arc_over_min_border_mean": float(rb.mean()),
        "arc_over_centroid_chord_mean": float((arc / chord).mean()),
        "arc_under_centroid_chord_fraction": float((arc / chord < 1).mean()),
        "spearman_arc_vs_centroid_chord": float(
            __import__("scipy.stats", fromlist=["spearmanr"])
            .spearmanr(arc, chord).statistic),
        "median_min_border_mm": float(np.median(border)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--k", type=int, default=36,
                    help="the LOCAL budget: k - k*long_range, i.e. 36 of 48")
    ap.add_argument("--sources", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--connectome", action="store_true",
                    help="also run part 2: arc length against its euclidean "
                         "surrogates, over the declared tract edges")
    ap.add_argument("--out", default="out/sheet_metric_error.json")
    a = ap.parse_args()
    from scipy.sparse.csgraph import dijkstra

    rng = np.random.default_rng(a.seed)
    sheet = CS.load_sheet()
    pos = P.cortical_sites(a.sites, "cpu", geometry="surface").numpy()
    idx, dist = P.knn_edges(torch.from_numpy(pos), a.k)
    idx, dist = idx.numpy(), dist.numpy()
    hit = CS._kdtree().query(pos, k=1, workers=-1)[1]
    hemi, vert = sheet["hemi"][hit], sheet["vertex"][hit]

    srcs = rng.choice(a.sites, a.sources, replace=False)
    total = cross = 0
    ratios, gates = [], {}
    for hi, h in enumerate(("lh", "rh")):
        v, e, G = mesh_graph(h)
        gates[h] = gate(v, e, G, np.random.default_rng(a.seed))
        assert gates[h] < 1e-6, (
            f"GATE FAILED on {h}: mesh-adjacent geodesic/euclidean is off by "
            f"{gates[h]:.4f}; the mesh graph is wrong and every number below "
            "with it")
        sel = [s for s in srcs if hemi[s] == hi]
        if not sel:
            continue
        D = dijkstra(G, indices=vert[sel], limit=400.0)
        for r, s in enumerate(sel):
            nb = idx[s]
            same = hemi[nb] == hi
            cross += int((~same).sum())
            total += len(nb)
            g = D[r, vert[nb[same]]]
            eu = dist[s][same]
            ok = np.isfinite(g) & (eu > 1e-6)
            ratios.append(g[ok] / eu[ok])
    rat = np.concatenate(ratios)
    res = {
        "config": vars(a),
        "gate_max_abs_error": gates,
        "edges_sampled": int(total),
        "cross_hemisphere_edges": int(cross),
        "cross_hemisphere_fraction": float(cross / max(total, 1)),
        "mean_euclidean_knn_mm": float(dist[srcs].mean()),
        "geodesic_over_euclidean": {
            f"p{int(100 * q):02d}": float(np.quantile(rat, q))
            for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "mean": float(rat.mean()),
        "frac_over_2x": float((rat > 2).mean()),
        "frac_over_4x": float((rat > 4).mean()),
    }
    print(f"GATE  max |geo/eu - 1| on mesh-adjacent pairs: "
          f"{max(gates.values()):.2e}  PASS")
    print(f"{total} local k-NN edges, mean euclidean "
          f"{res['mean_euclidean_knn_mm']:.2f} mm, "
          f"{cross} cross-hemisphere ({100*res['cross_hemisphere_fraction']:.2f}%)")
    for k, v in res["geodesic_over_euclidean"].items():
        print(f"  geodesic/euclidean {k}  {v:.2f}x")
    print(f"  mean {res['mean']:.2f}x   >2x {100*res['frac_over_2x']:.1f}%   "
          f">4x {100*res['frac_over_4x']:.1f}%")
    if a.connectome:
        cm = connectome_metric_disagreement()
        res["connectome"] = cm
        print(f"\nPART 2 -- {cm['n_edges']} declared tract edges")
        print("  GATE arc / min border-to-border >= 1 for every edge  PASS")
        for k, v in cm["arc_over_min_border"].items():
            print(f"  arc / min border  {k}  {v:6.2f}x")
        print(f"  mean {cm['arc_over_min_border_mean']:.2f}x")
        print(f"  arc / CENTROID chord mean "
              f"{cm['arc_over_centroid_chord_mean']:.2f}x, below 1 for "
              f"{100*cm['arc_under_centroid_chord_fraction']:.1f}% -- biased, "
              "streamlines end at parcel borders not centroids")
        print(f"  spearman(arc, centroid chord) "
              f"{cm['spearman_arc_vs_centroid_chord']:.3f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
