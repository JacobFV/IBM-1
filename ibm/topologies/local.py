"""3d local spatial adjacency.

the plain one, and the only one in the inventory for which euclidean distance in
the anatomical volume is simply the right metric.  what travels this way --
extracellular potassium, interstitial glutamate spillover, heat, the local
transmembrane current that a neighbouring position's field sees -- moves through
the parenchyma in whatever direction it happens to be pointing, and does not care
about sheets, trees or fascicles.  a radius graph in millimetres is not an
approximation to something better here; it *is* the physics.

this is worth stating because `local` is the topology most likely to be reached
for out of habit when another one is meant.  two cortical positions a millimetre
apart across a sulcal bank are one millimetre apart for potassium diffusion and
five centimetres apart for lateral cortico-cortical spread; `local` is right for
the first and badly wrong for the second, which is what `cortical_surface` exists
for.  likewise two positions in the same cortical column at different depths are
neighbours here at a distance that says nothing about laminar order, which is what
`laminar` exists for.

the radius is a property of the *process*, not of the topology, but it has to be
chosen when the graph is built, so it defaults to twice the site spacing: the
smallest radius at which a lattice is connected in all directions rather than only
along its axes.  a process with a longer physical length constant asks for a
larger one and gets a denser graph; a process with a shorter one gets nothing from
a finer graph than the materialization instantiated, which is the honest answer.

`orientation` is carried because several of the processes over this topology are
not isotropic even though the graph is.  interstitial diffusion follows the white
matter, thermal conduction follows perfusion, and mechanical stress is a tensor;
all three need the direction of the edge, and recovering it downstream from
positions means the builder's site table has to be carried around with the edge
set.  three extra columns is cheaper.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance


@B.builder(
    "local_radius",
    produces=("distance_mm", "orientation"),
    supports=("tissue",),
    directed=False,
    metric="euclidean distance in the anatomical volume",
    doc="all pairs of sites closer than a radius, with distance and unit direction")
def local_radius(sites, *, support: str = "tissue", radius_mm: float | None = None,
                 max_degree: int | None = None):
    """a radius graph over one volumetric support.

    the degree cap exists because a radius graph on a non-uniform site set is not
    degree-bounded: a materialization that instantiated one region at 0.5 mm and
    another at 4 mm produces, at a radius chosen for the coarse region, a fine
    region in which every site sees hundreds of neighbours.  capping the degree by
    keeping each site's nearest edges is a statement about cost, so it is optional
    and off by default rather than silently applied -- an edge set that quietly
    dropped a third of its edges would be indistinguishable from one that did not.
    """
    np = B._numpy("local_radius")
    t = sites.require(support, "local_radius",
                      f"volumetric site positions on {support!r}")
    xyz = np.asarray(t.xyz, dtype=float)
    r = float(radius_mm) if radius_mm is not None else 2.0 * float(np.max(t.spacing(np)))

    i, j, d = B.pairs_within(xyz, r, "local_radius")

    if max_degree is not None and len(d):
        order = np.argsort(d, kind="stable")
        keep = np.zeros(len(d), dtype=bool)
        deg = np.zeros(t.n, dtype=np.int64)
        cap = int(max_degree)
        # a python sweep over edges in increasing length.  vectorizing it would
        # need a different definition of the cap, and the honest one is "each
        # site keeps its shortest edges", which is inherently sequential.
        for e in order:
            a, b = int(i[e]), int(j[e])
            if deg[a] < cap and deg[b] < cap:
                keep[e] = True
                deg[a] += 1
                deg[b] += 1
        i, j, d = i[keep], j[keep], d[keep]

    u = B.unit(xyz[j] - xyz[i], np)
    return B.EdgeSet(
        "local", i + t.offset, j + t.offset, sites.n_total,
        {"distance_mm": d, "orientation": u}, directed=False,
        note=f"radius {r:.3g} mm over {support}"
             + (f", degree capped at {max_degree}" if max_degree else ""))


LOCAL = REGISTRY.topology(Topology(
    "local",
    "3d local spatial adjacency in the anatomical volume: every pair of materialized "
    "positions closer than a physical radius, with the distance and the unit direction "
    "between them.  the one topology whose metric is straight-line euclidean distance, "
    "because what it supports -- ionic and transmitter diffusion in the interstitium, "
    "thermal conduction, local volume conduction of current -- propagates through tissue "
    "without regard to the sheet, the fascicle or the vascular tree.  it is emphatically "
    "not a substitute for the cortical-surface topology: neighbours across a sulcus are "
    "neighbours here and are far apart along the cortex, and using `local` where lateral "
    "cortical spread is meant is the single most common way a brain model acquires "
    "connections that no axon could make",
    on=("tissue",),
    edge_features=("distance_mm", "orientation"),
    directed=False,
    builder="local_radius",
    provenance=Provenance.PHYSICS))
