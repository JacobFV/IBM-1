"""tractometric adjacency: white-matter connections, with their conduction delays.

the long-range topology, and the one whose metric is neither euclidean nor
geodesic-on-the-sheet but *arc length along a fascicle*.  none of the three agree.
the arcuate connects frontal and temporal cortex over a path of roughly 150 mm
between endpoints 60 mm apart in a straight line and much further than either
along the cortical sheet; the corpus callosum joins homotopic points that are
close in the volume and unreachable along the surface.  a topology built from
either of the other two metrics gets both cases wrong, in opposite directions.

what this topology carries that the others do not is *time*.  cortico-cortical
conduction delays run from under a millisecond to tens of milliseconds, they are
comparable to the periods of the rhythms the model is about, and they are set by
tract length and axonal conduction velocity together -- a long heavily myelinated
callosal fibre can be faster than a short intracortical one.  delay is a fact
about the pair of positions and their anatomy, not about the process, so it lives
here.  in the spectral representation a delay is a phase ramp, so carrying it per
edge costs nothing at run time: no history buffer, no delay-bounded timestep, and
no reason to lump a whole pathway into one mean latency.

direction is the honest weak point.  diffusion MRI reconstructs the fascicle, not
the axon, and cannot say which way a projection runs -- the arcuate is a bundle in
both directions and a streamline has no polarity.  so the builder emits both
directions of every reconstructed connection and marks the topology directed,
because the *anatomy* is directed even though the measurement is not; a process
that has a reason to believe in an asymmetry (a laminar termination pattern, a
tracer prior, a fitted effective connectivity) writes it into its parameters over
these edges rather than into the edge set.

`streamline count` is deliberately not an edge feature.  it is the quantity most
often used as a connection weight and it is not a fact about the anatomy: it
scales with the seeding density, the length of the path, the curvature, and the
distance from the seed region, all of which are properties of the algorithm.  a
topology is a support for interaction, not a weighting of it (ARCHITECTURE.md §3),
and coupling strength is a process parameter with a prior and a posterior.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance

_STREAM_WHAT = (
    "either `streamlines`, a sequence of (p, 3) point arrays in the site frame, or "
    "`endpoints` of shape (m, 2, 3) together with `lengths_mm` of shape (m,).  the "
    "arc length is required separately from the endpoints because a streamline is "
    "not straight: chord length underestimates a u-fibre by a little and the "
    "arcuate by a factor of two, and conduction delay follows the path")
_STREAM_WHERE = (
    "a whole-brain tractogram (.tck/.trk) from mrtrix3 tckgen with an ACT/SIFT2 "
    "pipeline, dsi-studio, or tractoflow; or a precomputed connectome plus "
    "average path lengths, in which case use the `tractometric_matrix` builder "
    "(data/sources: mrtrix3, dsi-studio, tractoflow, tractogram-formats, "
    "braingraph-hcp-connectomes, enigma-hcp-structural-connectome, "
    "iit-human-brain-atlas, histological-tract-atlases, freesurfer-tracts)")


def _arc_lengths(np, streamlines):
    ends = np.zeros((len(streamlines), 2, 3), dtype=float)
    lens = np.zeros(len(streamlines), dtype=float)
    for i, s in enumerate(streamlines):
        p = np.asarray(s, dtype=float)
        ends[i, 0], ends[i, 1] = p[0], p[-1]
        lens[i] = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0
    return ends, lens


@B.builder(
    "tractometric_streamlines",
    produces=("tract_length_mm", "conduction_delay_s", "distance_mm", "orientation"),
    supports=("tissue",),
    requires=("streamlines",),
    directed=True,
    metric="arc length along the reconstructed fascicle",
    doc="aggregate a tractogram onto materialized tissue sites, with delays")
def tractometric_streamlines(sites, *, support: str = "tissue", streamlines=None,
                             endpoints=None, lengths_mm=None,
                             snap_mm: float = 3.0, velocity_m_s: float | None = None,
                             min_length_mm: float = 5.0):
    """map a tractogram's endpoints onto sites and aggregate per site pair.

    `snap_mm` is the radius within which a streamline endpoint is accepted as
    belonging to a site, and its default is not arbitrary: the dwi-to-t1 warp
    residual is about 1.5 mm even with a fieldmap, streamline termination is
    biased toward gyral crowns by a comparable amount, and beyond a few
    millimetres an endpoint is as likely to belong to a different area.
    endpoints that snap to nothing are dropped and counted in the note, because a
    tractogram that mostly fails to reach the materialized sites is a
    registration failure and should be visible as one rather than as a sparse graph.

    velocity comes from the `myelination` site column when there is one, averaged
    over the two endpoints, and from `velocity_m_s` otherwise.  neither is
    measured: the myelin maps ibm-1 can obtain are ordinal proxies, so the delay
    is an ordering with a plausible scale rather than a measurement, and a process
    that depends on it sharply should fit it.
    """
    np = B._numpy("tractometric_streamlines")
    t = sites.require(support, "tractometric_streamlines", "tissue site positions")

    if streamlines is not None:
        ends, lens = _arc_lengths(np, streamlines)
    elif endpoints is not None and lengths_mm is not None:
        ends = np.asarray(endpoints, dtype=float).reshape(-1, 2, 3)
        lens = np.asarray(lengths_mm, dtype=float).reshape(-1)
    else:
        raise B.MissingInput("tractometric_streamlines", "streamlines",
                             _STREAM_WHAT, _STREAM_WHERE)
    if len(ends) != len(lens):
        raise ValueError(f"{len(ends)} endpoint pairs but {len(lens)} lengths")

    xyz = np.asarray(t.xyz, dtype=float)
    ia, da = B.nearest(ends[:, 0, :], xyz, 1, "tractometric_streamlines")
    ib, db = B.nearest(ends[:, 1, :], xyz, 1, "tractometric_streamlines")
    ia, ib = ia[:, 0], ib[:, 0]
    ok = ((da[:, 0] <= snap_mm) & (db[:, 0] <= snap_mm) & (ia != ib)
          & (lens >= min_length_mm))
    dropped = int(len(ok) - ok.sum())
    ia, ib, lens = ia[ok], ib[ok], lens[ok]
    if len(ia) == 0:
        return B.empty("tractometric", sites.n_total,
                       ("tract_length_mm", "conduction_delay_s", "distance_mm", "orientation"),
                       directed=True,
                       note=f"no streamline endpoint pair snapped within {snap_mm} mm")

    lo = np.minimum(ia, ib); hi = np.maximum(ia, ib)
    key = lo.astype(np.int64) * np.int64(t.n) + hi
    order = np.argsort(key, kind="stable")
    key, lens, lo, hi = key[order], lens[order], lo[order], hi[order]
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]
    starts = np.nonzero(first)[0]
    counts = np.diff(np.append(starts, len(key))).astype(float)
    length = np.add.reduceat(lens, starts) / counts          # mean arc length per pair
    i, j = lo[starts], hi[starts]

    if velocity_m_s is not None:
        v = np.full(len(i), float(velocity_m_s))
    else:
        myel = t.opt("myelination")
        if myel is None:
            v = np.full(len(i), 8.0)                          # see the note below
            vnote = "velocity 8 m/s (no myelination column, no velocity_m_s given)"
        else:
            m = np.asarray(myel, dtype=float)
            v = B.velocity_from_myelination(0.5 * (m[i] + m[j]), np)
            vnote = "velocity from the myelination column"
    if velocity_m_s is not None:
        vnote = f"velocity {velocity_m_s} m/s (given)"

    delay = B.conduction_delay_s(length, v, np)
    chord = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    u = B.unit(xyz[j] - xyz[i], np)

    # both directions: diffusion cannot polarize a fascicle, so emitting one
    # direction would be a claim the measurement does not support.
    src = np.concatenate([i, j]) + t.offset
    dst = np.concatenate([j, i]) + t.offset
    cat = lambda a: np.concatenate([a, a])
    return B.EdgeSet(
        "tractometric", src, dst, sites.n_total,
        {"tract_length_mm": cat(length), "conduction_delay_s": cat(delay),
         "distance_mm": cat(chord), "orientation": np.concatenate([u, -u])},
        directed=True,
        note=(f"{len(i)} site pairs from {len(ok)} streamlines, {dropped} dropped "
              f"(snap {snap_mm} mm, min length {min_length_mm} mm); {vnote}; "
              "both directions emitted because tractography has no polarity"))


@B.builder(
    "tractometric_matrix",
    produces=("tract_length_mm", "conduction_delay_s", "distance_mm", "orientation"),
    supports=("tissue",),
    requires=("matrix", "lengths_mm"),
    directed=True,
    metric="mean arc length between parcels",
    doc="expand a published parcel-by-parcel connectome onto materialized sites")
def tractometric_matrix(sites, *, support: str = "tissue", matrix=None, lengths_mm=None,
                        system: str = "cortical_areas", parcels=None,
                        threshold: float = 0.0, velocity_m_s: float | None = None):
    """a connectome given per parcel, expanded to the sites inside those parcels.

    this is the route for a group connectome that ibm-1 did not compute itself.
    the expansion is deliberately crude -- every site in parcel A gets an edge to
    every site in parcel B at the parcel-mean length -- because that is exactly
    what the input says.  a parcel-level connectome contains no within-parcel
    spatial information, and interpolating some would be inventing it.  the cost
    is quadratic in parcel population, which is the honest signal that a
    materialization far finer than the connectome is asking for more than the
    connectome has.
    """
    np = B._numpy("tractometric_matrix")
    t = sites.require(support, "tractometric_matrix", "tissue site positions")
    if matrix is None or lengths_mm is None:
        raise B.MissingInput(
            "tractometric_matrix", "matrix and lengths_mm",
            "a (k, k) structural connectivity matrix and a (k, k) matrix of mean "
            "streamline lengths in mm over the same k parcels, plus the parcel "
            "membership of each site", _STREAM_WHERE)
    A = np.asarray(matrix, dtype=float)
    L = np.asarray(lengths_mm, dtype=float)
    lab = parcels if parcels is not None else t.partitions.get(system)
    if lab is None:
        raise B.MissingInput(
            "tractometric_matrix", f"{support}.partitions[{system!r}]",
            f"an (n,) parcel index per site, or an (n, k) membership over the "
            f"{system!r} partitioning system, so the matrix rows can be placed",
            "ibm.anatomy.systems declares the system; ibm.anatomy.sources says how "
            "the memberships are obtained")
    lab = np.asarray(lab)
    if lab.ndim == 2:
        lab = np.argmax(lab, axis=1)
    lab = lab.astype(np.int64)

    xyz = np.asarray(t.xyz, dtype=float)
    src_l, dst_l, len_l = [], [], []
    for a in range(A.shape[0]):
        ra = np.nonzero(lab == a)[0]
        if not len(ra):
            continue
        for b in range(a + 1, A.shape[1]):
            if A[a, b] <= threshold:
                continue
            rb = np.nonzero(lab == b)[0]
            if not len(rb):
                continue
            g = np.repeat(ra, len(rb)); h = np.tile(rb, len(ra))
            src_l.append(g); dst_l.append(h)
            len_l.append(np.full(len(g), float(L[a, b])))
    if not src_l:
        return B.empty("tractometric", sites.n_total,
                       ("tract_length_mm", "conduction_delay_s", "distance_mm", "orientation"),
                       directed=True, note="no supra-threshold parcel pair was materialized")
    i = np.concatenate(src_l); j = np.concatenate(dst_l)
    length = np.concatenate(len_l)
    v = np.full(len(i), float(velocity_m_s) if velocity_m_s is not None else 8.0)
    delay = B.conduction_delay_s(length, v, np)
    chord = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    u = B.unit(xyz[j] - xyz[i], np)
    cat = lambda x: np.concatenate([x, x])
    return B.EdgeSet(
        "tractometric", np.concatenate([i, j]) + t.offset,
        np.concatenate([j, i]) + t.offset, sites.n_total,
        {"tract_length_mm": cat(length), "conduction_delay_s": cat(delay),
         "distance_mm": cat(chord), "orientation": np.concatenate([u, -u])},
        directed=True,
        note=f"expanded from a {A.shape[0]}-parcel connectome over {system}")


TRACTOMETRIC = REGISTRY.topology(Topology(
    "tractometric",
    "long-range white-matter adjacency, measured in arc length along the fascicle and "
    "carrying the conduction delay that follows from it.  the metric agrees with neither of "
    "the local ones and cannot be recovered from either: the arcuate runs 150 mm between "
    "endpoints 60 mm apart in the volume, and callosal partners are close in the volume and "
    "unreachable along the cortical sheet.  it is the only topology that carries time, "
    "because delay here is set by anatomy -- length over myelination-dependent conduction "
    "velocity -- spans the periods of the rhythms being modelled, and is a phase ramp rather "
    "than a buffer in the spectral representation.  streamline counts are not an edge "
    "feature: they are an artefact of the seeding and tracking algorithm, and coupling "
    "strength belongs to the process's parameters",
    on=("tissue",),
    edge_features=("tract_length_mm", "conduction_delay_s", "distance_mm", "orientation"),
    directed=True,
    builder="tractometric_streamlines",
    provenance=Provenance.LITERATURE))
