"""metabolic exchange: where blood and tissue actually meet.

the only topology in the inventory that joins two supports across a *barrier*, and
the reason it is not part of either the vascular or the local topology.  the
vascular tree carries substrate along itself; the tissue takes it up; and the
transfer happens at the capillary wall and essentially nowhere else.  arteries and
arterioles are lined by smooth muscle and do not exchange -- an arteriole passing
through a cubic millimetre of cortex delivers nothing to it -- so the topology has
to select capillaries by calibre rather than accept any vessel that happens to be
nearby.  a graph that let every vessel exchange with the tissue it passes through
would deliver oxygen upstream of where it is actually extracted, which is
precisely the error that makes a bold model insensitive to the transit time.

the metric is euclidean and, for the third time in this inventory, for a specific
reason rather than by default: the distance that matters is the radial diffusion
distance from a capillary wall out into tissue, the Krogh geometry.  it is not
path length along the tree, because the molecule has left the tree; it is not
tortuosity-corrected interstitial distance, because the first and dominant barrier
is the endothelium.  the reach is set by the inter-capillary spacing -- capillary
density in human grey matter puts most tissue within about 25 microns of a
capillary, and the oxygen diffusion distance is of that order -- so a
materialization coarser than that has already averaged over the exchange geometry,
and the builder says so rather than pretending the reach is resolved.

`contact_area_mm2` is the exchange surface, taken as the capillary wall area
2 pi r L attributable to the segment.  it is the physically right quantity: flux
is permeability times surface times concentration difference, and capillary
surface density per unit tissue volume is one of the few directly measured
numbers here (roughly 5-10 mm^2 per mm^3 in cortex).  carrying the area rather
than a count is also what makes the topology insensitive to how finely the
vascular tree was sampled.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance


@B.builder(
    "capillary_tissue_exchange",
    produces=("distance_mm", "contact_area_mm2"),
    supports=("vascular_tree", "tissue"),
    directed=False,
    metric="radial diffusion distance from the capillary wall into tissue",
    doc="tissue-to-capillary edges with the exchange surface each one carries")
def capillary_tissue_exchange(sites, *, vascular_support: str = "vascular_tree",
                              tissue_support: str = "tissue",
                              capillary_radius_mm: float = 0.005,
                              reach_mm: float = 0.05, k: int = 4):
    """which capillaries feed which tissue positions, and over how much wall.

    `capillary_radius_mm` defaults to 5 microns, the upper end of human cerebral
    capillary calibre; nodes wider than that are arterioles or venules and are
    excluded from exchange.  if the tree carries no radii the selection cannot be
    made, and the builder says so instead of exchanging with everything -- an
    exchange topology built over the whole tree is worse than none, because it
    looks reasonable and delivers oxygen at the wrong place in the transit.

    `reach_mm` defaults to 50 microns, about twice the mean tissue-to-capillary
    distance in cortex.  when the materialization's tissue spacing is much
    coarser than that -- as any whole-brain request will be -- every tissue site
    contains many capillaries and the reach is doing nothing but linking the site
    to whichever tree nodes were instantiated near it.  that is recorded in the
    note rather than corrected, because the correction is to materialize the
    vascular tree at capillary density, which is a request, not a builder.
    """
    np = B._numpy("capillary_tissue_exchange")
    v = sites.require(vascular_support, "capillary_tissue_exchange",
                      "vascular node positions, including the capillary bed")
    t = sites.require(tissue_support, "capillary_tissue_exchange",
                      "tissue positions that take up substrate")

    rad = v.opt("radius_mm")
    if rad is None:
        raise B.MissingInput(
            "capillary_tissue_exchange", f"{vascular_support}.columns['radius_mm']",
            "a (n,) vessel radius in mm per vascular node.  exchange happens at the "
            "capillary wall and not at arteries or arterioles, which are muscular and "
            "impermeable, so without radii the builder cannot tell which nodes may "
            "exchange -- and letting every node exchange delivers substrate upstream "
            "of where it is extracted",
            "the same centreline extraction the vascular topology needs (data/sources: "
            "microscopy-microvascular-networks, high-resolution-vascular-atlases, "
            "vesselgraph-mouse, fmost-mouse-vasculature-blocks)")
    rad = np.asarray(rad, dtype=float)
    cap = np.nonzero(rad <= float(capillary_radius_mm))[0]
    if len(cap) == 0 or t.n == 0:
        return B.empty("metabolic_exchange", sites.n_total,
                       ("distance_mm", "contact_area_mm2"),
                       note=f"no vascular node has radius <= {capillary_radius_mm} mm")

    cxyz = np.asarray(v.xyz, dtype=float)[cap]
    txyz = np.asarray(t.xyz, dtype=float)
    idx, dist = B.nearest(txyz, cxyz, int(k), "capillary_tissue_exchange")
    ok = dist <= float(reach_mm)
    if not ok.any():
        return B.empty("metabolic_exchange", sites.n_total,
                       ("distance_mm", "contact_area_mm2"),
                       note=(f"no tissue site is within {reach_mm} mm of a capillary; the "
                             "vascular tree is sampled far coarser than the tissue"))
    rows = np.repeat(np.arange(t.n), idx.shape[1])[ok.ravel()]
    cols = cap[idx.ravel()[ok.ravel()]]
    d = dist.ravel()[ok.ravel()]

    # capillary wall area attributable to each edge: 2 pi r L over the node's own
    # segment length, shared among the tissue sites that reach it.
    seg = v.opt("segment_length_mm")
    seg = v.spacing(np) if seg is None else np.asarray(seg, dtype=float)
    wall = 2.0 * np.pi * rad[cols] * seg[cols]
    share = np.bincount(cols, minlength=v.n).astype(float)
    area = wall / np.maximum(share[cols], 1.0)

    return B.EdgeSet(
        "metabolic_exchange", cols + v.offset, rows + t.offset, sites.n_total,
        {"distance_mm": d, "contact_area_mm2": area}, directed=False,
        note=(f"{len(d)} exchange edges over {len(cap)} capillary nodes; reach "
              f"{reach_mm} mm against a tissue spacing of "
              f"{float(np.median(t.spacing(np))):.3g} mm -- when the spacing is the "
              "larger the exchange geometry has already been averaged over"))


METABOLIC_EXCHANGE = REGISTRY.topology(Topology(
    "metabolic_exchange",
    "where blood and tissue trade substrate: the capillary wall, and only there.  it joins "
    "two supports across a barrier, which is why it is neither the vascular topology (which "
    "carries substrate along the tree) nor the local one (which moves it through tissue).  "
    "arteries and arterioles are muscular and do not exchange, so the topology selects "
    "vessels by calibre rather than by proximity -- a graph that let any nearby vessel "
    "exchange would deliver oxygen upstream of where it is extracted and make a bold model "
    "blind to transit time.  the metric is radial euclidean distance from the vessel wall, "
    "the Krogh geometry, because the molecule has left the tree and the endothelium is the "
    "first barrier.  the edge carries the exchange surface rather than a count, since flux "
    "is permeability times area and capillary surface density is one of the few directly "
    "measured quantities in this part of the ontology",
    on=("vascular_tree", "tissue"),
    edge_features=("distance_mm", "contact_area_mm2"),
    directed=False,
    builder="capillary_tissue_exchange",
    provenance=Provenance.PHYSICS))
