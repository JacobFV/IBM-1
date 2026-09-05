"""local spatial adjacency: neighbours in the tissue, on whichever support it is
indexed over.

the plain one, and the relation that says nothing more than "these two positions
are close enough in the parenchyma to exchange something directly".  what travels
this way -- extracellular potassium, interstitial glutamate spillover, heat, the
local transmembrane current that a neighbouring position's field sees, the
capillary bed a patch of tissue draws on -- moves through the parenchyma in
whatever direction it happens to be pointing, and does not care about fascicles
or the vascular tree.

**one relation, two metrics.**  what is fixed is the relation: proximity in the
tissue.  what is *not* fixed is how proximity is measured, because that belongs
to the support the positions were sampled on, and the supports genuinely disagree
(ARCHITECTURE.md §1: "fields do not share a support ... there is no universal
spatial domain, and consequently no universal spatial operator").

- on `tissue`, positions are parenchyma voxels and the metric is straight-line
  euclidean distance in millimetres.  a radius graph here is not an
  approximation to something better; it *is* the physics.
- on `cortical_surface`, positions are column nodes seeded on the folded sheet
  and the metric is geodesic distance along it -- the short-range restriction of
  the same metric `cortical_surface` carries at long range.

it would have been possible to declare a second topology for the second metric.
that would be wrong, and the reason is worth stating because it is the same
argument the registry makes about near-duplicate ids.  a topology names a *class
of process*: ionic exchange, metabolism, neurovascular coupling and recurrent
local excitation are declared over `local` and are the same processes whether
their positions came out of an octree or off a mesh.  two topology names would
mean every one of those processes had to be declared twice, and a materialization
that split the neural field between the sheet and the volume -- which is the
whole point of being able to do so -- would have half its local dynamics under
one name and half under another, with nothing in the ontology saying they were
the same relation.  the relation is one thing; the metric is the support's.

**why euclidean is wrong on the sheet, given that diffusion is euclidean.**  this
looks like a contradiction with the paragraph above and is not.  on `tissue` two
voxels a millimetre apart across a sulcal bank are a millimetre apart for
potassium, and the euclidean graph is right.  on `cortical_surface` the positions
are not voxels: a column node stands for a column of ribbon, and what lies
between two nodes on opposite banks of a sulcus is subarachnoid csf, not
interstitium.  the chord between them does not cross parenchyma at all, so an
edge along it is not a short diffusion path -- it is the across-the-sulcus edge
`cortical_surface` exists to prevent, wearing a different name.  measured on this
subject's own white surface at 10 mm spacing, 20.4% of the pairs a euclidean
radius graph connects are more than 20 mm apart along the sheet; the median
chord/geodesic ratio is a mild 1.41 but the p90 is 3.5 and the maximum 11, and
the map is not monotone, so no reweighting of the chord recovers the path.

what the geodesic restriction genuinely gives up is diffusion *across* the sulcal
cleft, which is real.  it does not belong here: that path runs through csf, and
`csf` and `interstitial` are the topologies for it.  a materialization that
cares should instantiate them rather than let `local` smuggle the coupling in
under a euclidean radius.

the radius is a property of the *process*, not of the topology, but it has to be
chosen when the graph is built, so it defaults to twice the site spacing: the
smallest radius at which a lattice is connected in all directions rather than
only along its axes, and on a mesh the smallest at which a node reaches past its
immediate triangle ring.  a process with a longer physical length constant asks
for a larger one and gets a denser graph; a process with a shorter one gets
nothing from a finer graph than the materialization instantiated, which is the
honest answer.

`local` is still not a substitute for `cortical_surface`, and the two do not
become the same thing on the sheet just because they now share a metric.  they
differ in reach and in what they carry: `local` is two site spacings of tissue
contact, `cortical_surface` is the horizontal axonal arbor, and the processes
declared over each read different edge features.  two positions in the same
cortical column at different depths are likewise neighbours here at a distance
that says nothing about laminar order, which is what `laminar` exists for.

`orientation` is carried because several of the processes over this topology are
not isotropic even though the graph is.  interstitial diffusion follows the white
matter, thermal conduction follows perfusion, and mechanical stress is a tensor;
all three need the direction of the edge, and recovering it downstream from
positions means the builder's site table has to be carried around with the edge
set.  three extra columns is cheaper.  on the sheet the orientation is the chord
direction and the distance is the path length, which are two different facts
about the same pair -- exactly as `cortical_surface` carries both.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance

#: the supports `local` knows how to measure proximity on, and how.  adding one
#: means adding a metric, not adding a case: a support with no notion of
#: short-range adjacency has no business carrying this relation.
METRICS = {"tissue": "euclidean distance in the anatomical volume",
           "cortical_surface": "geodesic distance along the cortical mesh"}


def _degree_cap(np, i, j, d, n, cap: int):
    """keep each site's shortest edges until it has `cap` of them.

    a python sweep over edges in increasing length.  vectorizing it would need a
    different definition of the cap, and the honest one is "each site keeps its
    shortest edges", which is inherently sequential.
    """
    order = np.argsort(d, kind="stable")
    keep = np.zeros(len(d), dtype=bool)
    deg = np.zeros(n, dtype=np.int64)
    for e in order:
        a, b = int(i[e]), int(j[e])
        if deg[a] < cap and deg[b] < cap:
            keep[e] = True
            deg[a] += 1
            deg[b] += 1
    return i[keep], j[keep], d[keep]


@B.builder(
    "local_radius",
    produces=("distance_mm", "orientation"),
    supports=("tissue", "cortical_surface"),
    directed=False,
    metric="proximity in the tissue, measured with the metric of the support it is "
           "built over: euclidean in the volume, geodesic on the sheet",
    doc="all pairs of sites closer than a radius, with distance and unit direction")
def local_radius(sites, *, support=None, radius_mm: float | None = None,
                 max_degree: int | None = None, faces=None, chunk: int = 256):
    """a radius graph over every materialized support this relation is declared on.

    `support` may be one name, several, or None.  None means "every support in
    `METRICS` that this materialization instantiated", which is the case that
    matters: a request that puts cortical population state on the sheet and
    subcortical population state in the volume needs local coupling on both, and
    a builder that took one support would give it dynamics in half its brain.
    the two halves come back as one edge set over global site indices, because
    they are one relation.

    the metric is dispatched on the support and not chosen by the caller.  there
    is no argument for measuring sheet positions with a chord or volume positions
    with a mesh path, and offering one would be offering a way to be wrong.

    the degree cap exists because a radius graph on a non-uniform site set is not
    degree-bounded: a materialization that instantiated one region at 0.5 mm and
    another at 4 mm produces, at a radius chosen for the coarse region, a fine
    region in which every site sees hundreds of neighbours.  capping the degree by
    keeping each site's nearest edges is a statement about cost, so it is optional
    and off by default rather than silently applied -- an edge set that quietly
    dropped a third of its edges would be indistinguishable from one that did not.
    the cap is applied per support, since it is a statement about local density.
    """
    np = B._numpy("local_radius")
    sites = B.as_sites(sites)
    if support is None:
        want = [s for s in METRICS if s in sites.tables]
        if not want:
            raise B.MissingInput(
                "local_radius", f"sites on one of {', '.join(sorted(METRICS))}",
                "materialized positions carrying the neural, interstitial, metabolic or "
                "thermal state that local coupling acts on.  `local` is the relation behind "
                "recurrent local excitation, ionic exchange, transmitter clearance, "
                "metabolism and neurovascular coupling, so a materialization with none of "
                "these supports has traced those processes and given them nothing to run on",
                "name `tissue` or `cortical_surface` in R; ibm.materialize.build places a "
                "component on the support R names")
    else:
        want = [support] if isinstance(support, str) else list(support)

    src, dst, dist, orient, notes = [], [], [], [], []
    for name in want:
        t = sites.require(name, "local_radius",
                          f"materialized site positions on {name!r}")
        if name not in METRICS:
            raise B.MissingInput(
                "local_radius", f"a metric for support {name!r}",
                "proximity in the tissue is measured with the support's own metric, and "
                f"{name!r} declares none here.  ibm.topologies.local.METRICS lists the "
                f"supports this relation knows how to measure on: {', '.join(METRICS)}",
                "declare the metric alongside the others in ibm.topologies.local, or place "
                "the component on a support that has one")
        xyz = np.asarray(t.xyz, dtype=float)
        r = (float(radius_mm) if radius_mm is not None
             else 2.0 * float(np.max(t.spacing(np))))

        if name == "cortical_surface":
            f = faces if faces is not None else t.opt("faces")
            if f is None:
                raise B.MissingInput("local_radius", f"{name}.columns['faces']",
                                     B.FACES_WHAT, B.FACES_WHERE)
            i, j, d = B.geodesic_pairs_within(xyz, f, "local_radius", radius_mm=r,
                                              chunk=int(chunk))
        else:
            i, j, d = B.pairs_within(xyz, r, "local_radius")

        if max_degree is not None and len(d):
            i, j, d = _degree_cap(np, i, j, d, t.n, int(max_degree))

        src.append(i + t.offset)
        dst.append(j + t.offset)
        dist.append(d)
        # the chord direction, on both supports.  on the sheet the *distance* is
        # the path and the *orientation* is the straight line between the two
        # nodes, because what the anisotropic processes over this topology need
        # is a direction in the head -- a fibre orientation, a thermal gradient,
        # a stress axis -- and there is no such thing as a direction along a
        # geodesic that is expressible as one vector.
        orient.append(B.unit(xyz[j] - xyz[i], np))
        notes.append(f"{name}: radius {r:.3g} mm, {METRICS[name].split(' distance')[0]}, "
                     f"{len(d):,} edges")

    z = np.zeros(0, dtype=np.int64)
    return B.EdgeSet(
        "local",
        np.concatenate(src) if src else z,
        np.concatenate(dst) if dst else z,
        sites.n_total,
        {"distance_mm": np.concatenate(dist) if dist else np.zeros(0, dtype=float),
         "orientation": (np.concatenate(orient) if orient
                         else np.zeros((0, 3), dtype=float))},
        directed=False,
        note="; ".join(notes)
             + (f"; degree capped at {max_degree} per support" if max_degree else ""))


LOCAL = REGISTRY.topology(Topology(
    "local",
    "local spatial adjacency in the parenchyma: every pair of materialized positions close "
    "enough to exchange something directly, with the distance and the unit direction between "
    "them.  what it supports -- ionic and transmitter diffusion in the interstitium, thermal "
    "conduction, metabolic draw on the local capillary bed, recurrent excitation within a "
    "patch of tissue, local volume conduction of current -- propagates through tissue without "
    "regard to the fascicle or the vascular tree.  the relation is one thing and its METRIC "
    "belongs to the support: euclidean distance where the positions are parenchyma voxels, "
    "geodesic distance along the sheet where they are cortical column nodes.  that is one "
    "relation with two metrics rather than two relations, which is why it is one topology: "
    "the processes declared over it are the same processes either way, and a materialization "
    "that indexed cortex on the sheet and subcortex in the volume would otherwise have half "
    "its local dynamics under a second name.  a euclidean radius graph over column nodes is "
    "not the cheap version of the geodesic one -- the chord between two nodes on opposite "
    "banks of a sulcus crosses subarachnoid csf rather than parenchyma, so it is the "
    "across-the-sulcus edge `cortical_surface` exists to exclude.  it remains emphatically "
    "not a substitute for `cortical_surface` itself, whose reach is the horizontal axonal "
    "arbor rather than two site spacings of tissue contact",
    on=("tissue", "cortical_surface"),
    edge_features=("distance_mm", "orientation"),
    directed=False,
    builder="local_radius",
    provenance=Provenance.PHYSICS))
