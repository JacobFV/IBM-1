"""cortical-surface geodesic adjacency.

the topology that exists because the cortex is a folded sheet and euclidean
distance in the head does not respect the folding.  two points a millimetre apart
across the bank of the central sulcus are, along the sheet, on opposite sides of
a gyrus -- centimetres of cortex apart, in different areas, with different
thalamic input and no direct connection.  every intrinsic horizontal axon,
every patch of layer 2/3 lateral spread, every travelling alpha wave goes *along*
the sheet, so the sheet's own metric is the only one that describes them.  a
radius graph in the volume produces edges that jump the sulcus, and those edges
are not weak connections to be down-weighted, they are connections that do not
exist.

the practical size of the error is worth stating: the ratio of geodesic to
euclidean distance across a sulcal bank routinely exceeds five, and in the insula
and the calcarine it is worse.  no monotone reweighting of euclidean distance can
recover it, because the map is not monotone -- the same euclidean distance
corresponds to a tenth of a millimetre or to eight centimetres depending on which
side of a fold each point is on.

geodesic distance also has to be computed on a mesh that follows the cortex rather
than a smoothed one.  it is measured along the midthickness surface here for a
reason: the pial surface exaggerates gyral crown distances and the white surface
compresses them, and the midthickness is the one that matches the length of a
horizontal axon in layer 3.

the builder emits `geodesic_mm` and `distance_mm` both, and carrying the chord as
well as the path is not redundancy.  their ratio is the local folding, which is
the quantity a forward model of volume conduction needs at the same time as the
lateral-propagation process needs the path -- and it is also the diagnostic that
tells you whether a surface reconstruction has self-intersected.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance

_FACES_WHAT = (
    "an (m, 3) integer array of triangles indexing this table's rows -- the "
    "triangulation of the materialized surface sites.  a set of surface positions "
    "without its faces is a point cloud, and a point cloud has no geodesic metric: "
    "reconstructing one by nearest-neighbour linking is exactly the step that "
    "reintroduces the across-the-sulcus edges this topology exists to exclude")
_FACES_WHERE = (
    "FreeSurfer ?h.midthickness / ?h.white surfaces, or the HCP fs_LR 32k "
    "midthickness GIFTI, read with nibabel; connectome-workbench "
    "-surface-create-sphere and -metric-resample if the materialized sites are a "
    "decimation of the full mesh (data/sources: freesurfer, nibabel, "
    "connectome-workbench, templateflow)")


def _mesh_edges(np, xyz, faces, builder: str):
    """unique undirected mesh edges and their euclidean lengths."""
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise B.MissingInput(builder, "faces", _FACES_WHAT + f" (got shape {f.shape})",
                             _FACES_WHERE)
    if f.size and (f.min() < 0 or f.max() >= len(xyz)):
        raise B.MissingInput(
            builder, "faces",
            "triangles indexing this table's rows; the given faces index vertices "
            f"outside [0, {len(xyz)}), so they belong to a different (probably "
            "undecimated) mesh than the materialized sites", _FACES_WHERE)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.linalg.norm(xyz[e[:, 0]] - xyz[e[:, 1]], axis=1)
    return e, w


def _geodesic_scipy(np, n, e, w, radius, k, chunk):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = csr_matrix((np.concatenate([w, w]), (rows, cols)), shape=(n, n))

    src, dst, geo = [], [], []
    lim = radius if radius is not None else np.inf
    for a in range(0, n, chunk):
        idx = np.arange(a, min(a + chunk, n))
        D = dijkstra(A, directed=False, indices=idx, limit=lim)
        if k is not None:
            kk = int(min(k + 1, D.shape[1]))
            part = np.argpartition(np.where(np.isfinite(D), D, np.inf), kk - 1, axis=1)
            keep = np.zeros(D.shape, dtype=bool)
            np.put_along_axis(keep, part[:, :kk], True, axis=1)
            D = np.where(keep, D, np.inf)
        r, c = np.nonzero(np.isfinite(D) & (D > 0.0))
        src.append(idx[r]); dst.append(c); geo.append(D[r, c])
    return (np.concatenate(src) if src else np.zeros(0, dtype=np.int64),
            np.concatenate(dst) if dst else np.zeros(0, dtype=np.int64),
            np.concatenate(geo) if geo else np.zeros(0, dtype=float))


def _geodesic_python(np, n, e, w, radius, k):
    """dijkstra per source with a heap.

    the fallback when scipy is absent.  it is the same algorithm, an order of
    magnitude slower, and it exists so that the ontology's central metric is not
    conditional on an optional dependency.
    """
    import heapq

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (a, b), ww in zip(e.tolist(), w.tolist()):
        adj[a].append((b, ww)); adj[b].append((a, ww))
    lim = float(radius) if radius is not None else float("inf")
    kk = int(k) if k is not None else None

    src, dst, geo = [], [], []
    for s in range(n):
        dist = {s: 0.0}
        heap = [(0.0, s)]
        out: list[tuple[float, int]] = []
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist.get(v, float("inf")):
                continue
            if v != s:
                out.append((d, v))
                if kk is not None and len(out) >= kk:
                    break
            for u, ww in adj[v]:
                nd = d + ww
                if nd <= lim and nd < dist.get(u, float("inf")):
                    dist[u] = nd
                    heapq.heappush(heap, (nd, u))
        for d, v in out:
            src.append(s); dst.append(v); geo.append(d)
    return (np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64),
            np.asarray(geo, dtype=float))


@B.builder(
    "cortical_geodesic",
    produces=("geodesic_mm", "distance_mm"),
    supports=("cortical_surface",),
    directed=False,
    metric="geodesic distance along the cortical mesh",
    doc="mesh-graph shortest paths within a radius, or the k geodesically nearest")
def cortical_geodesic(sites, *, support: str = "cortical_surface", faces=None,
                      radius_mm: float | None = None, k: int | None = None,
                      chunk: int = 256):
    """geodesic neighbourhoods on the materialized cortical mesh.

    the path length is measured along mesh edges, which overestimates the true
    surface geodesic by a few percent on a triangulation this coarse -- a path
    constrained to edges cannot cut across a face.  exact methods (mmp, heat) fix
    that, and the residual is far below the inter-subject variation in cortical
    geometry, so the edge-graph distance is the right amount of machinery here and
    the bias is recorded rather than corrected.

    give `radius_mm` for a physical neighbourhood, `k` for a fixed degree, or both
    to cap a physical neighbourhood's degree.  neither given means k = 6, the
    degree of a regular triangulation, which is the smallest graph on which a
    surface laplacian is defined at all.
    """
    np = B._numpy("cortical_geodesic")
    t = sites.require(support, "cortical_geodesic",
                      "positions of the materialized cortical surface sites")
    xyz = np.asarray(t.xyz, dtype=float)
    f = faces if faces is not None else t.opt("faces")
    if f is None:
        raise B.MissingInput("cortical_geodesic", f"{support}.columns['faces']",
                             _FACES_WHAT, _FACES_WHERE)
    if radius_mm is None and k is None:
        k = 6

    e, w = _mesh_edges(np, xyz, f, "cortical_geodesic")
    n = t.n
    if n < 2 or len(e) == 0:
        return B.empty("cortical_surface", sites.n_total, ("geodesic_mm", "distance_mm"))

    if B._kdtree() is not None:      # scipy present, so csgraph is too
        s, d, g = _geodesic_scipy(np, n, e, w, radius_mm, k, int(chunk))
    else:                                                     # pragma: no cover
        s, d, g = _geodesic_python(np, n, e, w, radius_mm, k)

    # one fact per pair.  a k-nearest sweep is not symmetric, so the union is
    # taken and the shorter of the two paths kept -- they differ only by the
    # tie-breaking inside dijkstra, and keeping both copies would let them drift.
    lo = np.minimum(s, d); hi = np.maximum(s, d)
    key = lo * np.int64(n) + hi
    order = np.argsort(key, kind="stable")
    key, g = key[order], g[order]
    lo, hi = lo[order], hi[order]
    first = np.ones(len(key), dtype=bool)
    if len(key) > 1:
        first[1:] = key[1:] != key[:-1]
    keep = np.nonzero(first)[0]
    gmin = np.minimum.reduceat(g, keep) if len(keep) else g
    i, j = lo[keep], hi[keep]

    chord = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    return B.EdgeSet(
        "cortical_surface", i + t.offset, j + t.offset, sites.n_total,
        {"geodesic_mm": gmin, "distance_mm": chord}, directed=False,
        note=(f"mesh geodesic, radius={radius_mm}, k={k}; median geodesic/chord ratio "
              f"{float(np.median(gmin / np.maximum(chord, 1e-9))):.2f}"))


CORTICAL_SURFACE = REGISTRY.topology(Topology(
    "cortical_surface",
    "adjacency along the cortical sheet, measured as geodesic distance on the midthickness "
    "mesh.  it is a different topology from `local` and not a refinement of it: the cortex "
    "is folded, so straight-line distance in the head and path length along the sheet are "
    "not monotonically related, and a millimetre across a sulcus is centimetres of cortex.  "
    "intrinsic horizontal connections, lateral inhibitory surrounds and travelling cortical "
    "waves all run along the sheet, so this is the support for every process describing "
    "them.  both the geodesic path and the euclidean chord are carried, because their ratio "
    "is the local folding -- which is what a volume-conduction forward model needs at the "
    "same positions where the propagation process needs the path",
    on=("cortical_surface",),
    edge_features=("geodesic_mm", "distance_mm"),
    directed=False,
    builder="cortical_geodesic",
    provenance=Provenance.PHYSICS))
