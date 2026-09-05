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

the practical size of the error is worth stating, and it is smaller in the median
than the argument sounds.  measured on the mne `sample` white surface (202,437
mm^2, 312,273 vertices), the geodesic/chord ratio over the pairs a euclidean
radius graph would connect has median 1.04 at 2 mm, 1.13 at 5 mm and 1.41 at
10 mm.  the median is not where the damage is.  the p90 reaches 3.5 and the
maximum 11, and at 10 mm spacing 20.4% of the pairs a volume graph connects lie
more than 20 mm apart along the cortex.

no monotone reweighting of euclidean distance recovers it, because the map is not
monotone -- the same chord corresponds to a tenth of a millimetre or to several
centimetres depending on which side of a fold each point is on, and a single
number cannot tell those apart.  that, rather than the median distortion, is the
argument for carrying the sheet's own metric.

(an earlier version of this docstring claimed the ratio "routinely exceeds five".
it did not: that figure came from a geodesic graph assembled from half-edges,
where scipy's duplicate-summing doubled every interior edge weight and so doubled
every path length.  the numbers above are from the deduplicated graph.)

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

    the metric itself lives in `ibm.topologies.builders.geodesic_pairs_within`,
    beside the euclidean `pairs_within`, because it belongs to the *support* and
    not to this topology: more than one relation is measured with the sheet's own
    distance, and `local` on the sheet is the other one.
    """
    np = B._numpy("cortical_geodesic")
    t = sites.require(support, "cortical_geodesic",
                      "positions of the materialized cortical surface sites")
    xyz = np.asarray(t.xyz, dtype=float)
    f = faces if faces is not None else t.opt("faces")
    if f is None:
        raise B.MissingInput("cortical_geodesic", f"{support}.columns['faces']",
                             B.FACES_WHAT, B.FACES_WHERE)
    if radius_mm is None and k is None:
        k = 6

    i, j, gmin = B.geodesic_pairs_within(xyz, f, "cortical_geodesic",
                                         radius_mm=radius_mm, k=k, chunk=int(chunk))
    if not len(i):
        return B.empty("cortical_surface", sites.n_total, ("geodesic_mm", "distance_mm"))

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
