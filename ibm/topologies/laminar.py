"""cortical-depth adjacency: the topology inside a cortical column.

the third distinct notion of nearness in cortex, after the volume and the sheet.
its coordinate is not a distance at all -- it is normalized depth, pial to white
-- and that is the point.  cortical thickness ranges from about 1.5 mm in
calcarine cortex to 4.5 mm in precentral, and the layers occupy roughly the same
*proportions* of the ribbon across that range.  a topology that connected depth
samples by their separation in millimetres would make the layers of thin cortex
neighbours and the layers of thick cortex distant, which inverts the anatomy: a
0.5 mm step is a whole layer in V1 and a third of layer 3 in area 4.

so the metric is signed depth difference within a column, and the edges only exist
within a column.  two depth samples in adjacent columns are not laminar
neighbours; they are lateral neighbours, and `cortical_surface` carries that.
what makes a pair a laminar pair is that the same radial bundle of apical
dendrites and the same set of feedforward terminals pass through both.

the topology is *directed* and this is the one place in the local geometry where
direction is anatomy rather than convention.  cortico-cortical feedforward
projections terminate in layer 4 and originate in supragranular layers;
feedback originates infragranularly and terminates outside layer 4; the
translaminar cascade runs 4 -> 2/3 -> 5 -> 6 with a distinct 5 -> 2/3 return.  an
undirected depth graph cannot express any of that, and the asymmetry is precisely
what a laminar model is for -- collapsing it would leave a six-compartment
diffusion equation with no hierarchy in it.  `depth_delta` is therefore signed,
positive toward the white matter, and both directions are emitted so that a
process can give ascending and descending edges different dynamics without
re-deriving which is which.

`distance_mm` is carried alongside `depth_delta` because the two say different
things and both are needed: the delta says which layers are involved and the
millimetres say how far a dendrite actually has to conduct.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance

_DEPTH_WHAT = (
    "a (n,) array of normalized cortical depth in [0, 1], 0 at the pial surface "
    "and 1 at the grey/white boundary, computed equivolumetrically rather than by "
    "equal spacing -- an equidistant depth coordinate assigns the same fraction to "
    "different layers on a gyral crown and in a sulcal fundus, because the ribbon "
    "is compressed on one and stretched on the other")
_DEPTH_WHERE = (
    "FreeSurfer / connectome-workbench equivolumetric surface stack, or "
    "nighres.laminar volumetric layering on a T1 at 0.7 mm or better "
    "(data/sources: freesurfer, connectome-workbench, bigbrainwarp)")
_COLUMN_WHAT = (
    "a (n,) integer array saying which radial column each depth sample belongs to "
    "-- normally the index of the surface vertex the sample was sampled along.  "
    "without it a set of depth values is a histogram, not a set of columns, and "
    "the builder cannot tell which samples are stacked above one another")
_COLUMN_WHERE = (
    "produced by the materializer when it samples a depth stack along each "
    "cortical_surface site; identical to the surface site index by construction")


@B.builder(
    "laminar_adjacent",
    produces=("depth_delta", "distance_mm"),
    supports=("cortical_depth",),
    directed=True,
    metric="signed normalized depth difference within one cortical column",
    doc="edges between depth samples of the same column, ordered pial to white")
def laminar_adjacent(sites, *, support: str = "cortical_depth", span: int = 1):
    """adjacency in depth within each column, both directions.

    `span` is how many depth steps count as adjacent.  the default of 1 gives the
    nearest-neighbour chain, which is what a diffusion-like translaminar process
    wants.  a larger span is for the canonical circuit's skip connections -- layer
    4 to layer 5 is not a nearest-neighbour edge at eight depth samples -- and a
    process that needs a specific pair of layers should ask for the span that
    reaches them and then select on `depth_delta`, rather than the topology
    hard-coding a layer assignment it does not have.
    """
    np = B._numpy("laminar_adjacent")
    t = sites.require(support, "laminar_adjacent", "cortical depth samples")
    depth = np.asarray(t.col("depth", "laminar_adjacent", _DEPTH_WHAT, _DEPTH_WHERE),
                       dtype=float)
    column = np.asarray(t.col("column_id", "laminar_adjacent", _COLUMN_WHAT, _COLUMN_WHERE),
                        dtype=np.int64)
    xyz = np.asarray(t.xyz, dtype=float)
    if t.n < 2:
        return B.empty("laminar", sites.n_total, ("depth_delta", "distance_mm"), directed=True)

    order = np.lexsort((depth, column))
    col_s = column[order]

    a_l, b_l = [], []
    for s in range(1, max(1, int(span)) + 1):
        if s >= t.n:
            break
        same = col_s[:-s] == col_s[s:]
        a_l.append(order[:-s][same])
        b_l.append(order[s:][same])
    if not a_l:
        return B.empty("laminar", sites.n_total, ("depth_delta", "distance_mm"), directed=True)
    a = np.concatenate(a_l); b = np.concatenate(b_l)

    dd = depth[b] - depth[a]                       # positive: a is shallower than b
    dist = np.linalg.norm(xyz[b] - xyz[a], axis=1)

    # both directions, with the sign of the depth step reversed on the ascending
    # copy.  a process reads the sign to know whether it is looking at a
    # descending (supragranular -> infragranular) or an ascending edge.
    src = np.concatenate([a, b]) + t.offset
    dst = np.concatenate([b, a]) + t.offset
    return B.EdgeSet(
        "laminar", src, dst, sites.n_total,
        {"depth_delta": np.concatenate([dd, -dd]),
         "distance_mm": np.concatenate([dist, dist])},
        directed=True,
        note=f"within-column depth chain, span {span}, "
             f"{len(np.unique(column))} columns; depth_delta > 0 runs toward white matter")


LAMINAR = REGISTRY.topology(Topology(
    "laminar",
    "adjacency through the cortical ribbon, within a column, ordered from pial to white.  "
    "its coordinate is normalized depth rather than millimetres, because cortical thickness "
    "varies three-fold across the sheet while the layers keep roughly constant proportions "
    "of it -- so a fixed millimetre step is a whole layer in one place and a third of one in "
    "another, and only the proportion is comparable.  it is directed, and the direction is "
    "anatomy: feedforward input terminates in layer 4 and feedback avoids it, the "
    "translaminar cascade runs granular to supragranular to infragranular, and an "
    "undirected depth graph erases the hierarchy that laminar models exist to represent.  "
    "edges exist only within a column; two depth samples in neighbouring columns are "
    "related through `cortical_surface`, not through this",
    on=("cortical_depth",),
    edge_features=("depth_delta", "distance_mm"),
    directed=True,
    builder="laminar_adjacent",
    provenance=Provenance.LITERATURE))
