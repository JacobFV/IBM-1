"""the local microcircuit: the canonical wiring among populations at one position.

the odd one in the inventory, because its edges are mostly self-edges, and that is
not a degenerate case -- it is the correct statement of what the microcircuit is.
excitatory, pv, sst and vip populations coexist in the same cubic millimetre of
cortex (ARCHITECTURE.md §2 puts them in the state vector rather than in space
precisely because they do), so exc -> pv, pv -> exc, sst -> exc and vip -> sst are
couplings between *components at one position*.  they have no spatial extent to
carry a distance about.

it would be possible to leave such a coupling with no topology at all and let a
process act pointwise.  declaring it explicitly buys two things.  the registry can
check it -- `seal()` refuses a process whose topology is not registered, and a
pointwise process would be the one hole in that check.  and it makes the local
circuit a *thing that can be varied*: a materialization that instantiates a
column at 100 microns rather than a region at 3 mm should let neighbouring
micro-sites within the same column see each other, which is what `radius_mm`
turns on, and at that scale the sst lateral inhibitory reach of a few hundred
microns becomes a real spatial fact rather than a within-site one.

so the metric is: none by default (identity), and sub-columnar euclidean distance
when the materialization is fine enough for that to mean something.  it is
deliberately *not* the `local` millimetre-scale radius graph -- microcircuit edges
at 3 mm spacing would connect populations in different cortical areas and call it
a canonical circuit.  the radius therefore defaults to zero rather than to a
multiple of the spacing: a coarse materialization gets self-edges only, which is
the honest answer, instead of getting a plausible-looking graph that is wrong.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance


@B.builder(
    "microcircuit_within_site",
    produces=("distance_mm",),
    supports=("tissue", "cortical_surface"),
    directed=False,
    metric="identity, extended to sub-columnar euclidean distance when radius_mm > 0",
    doc="self-edges at every site, plus sub-columnar neighbours if asked for")
def microcircuit_within_site(sites, *, support=None, radius_mm: float = 0.0,
                             max_radius_mm: float = 0.5):
    """the within-position circuit, optionally widened to a real local neighbourhood.

    `support` may be one name, several, or None.  None means "every support this
    topology is declared on that the materialization instantiated", which is the
    case that matters: a request that puts cortical population state on the sheet
    and subcortical population state in the volume has the four populations
    coexisting at both kinds of position, and a builder that took one support
    would give the canonical circuit to half the brain and silently withhold it
    from the other.  it used to default to `tissue`, so a cortical materialization
    got nothing unless the caller thought to say otherwise.

    `max_radius_mm` is a guard rather than a parameter.  the canonical
    microcircuit is a statement about a patch of cortex a few hundred microns
    across; asked for at millimetres it stops being a microcircuit and starts
    being an unlabelled short-range connectivity model, and the refusal is louder
    and more useful than the graph would be.

    the widened neighbourhood is measured with the chord on both supports, and
    that is not the inconsistency it looks like beside `local`.  the guard caps it
    at half a millimetre, which is an order of magnitude below the width of a
    sulcal bank; over that range a column node's chord and its geodesic agree to
    a few percent, so there is no across-the-sulcus edge to be had.  a request
    that raises `max_radius_mm` past the folding scale is asking for `local`
    instead, and the error already says so.
    """
    np = B._numpy("microcircuit_within_site")
    sites = B.as_sites(sites)
    declared = ("tissue", "cortical_surface")
    if support is None:
        want = [s for s in declared if s in sites.tables]
        if not want:
            raise B.MissingInput(
                "microcircuit_within_site", f"sites on one of {', '.join(declared)}",
                "positions whose populations interact locally -- parenchyma voxels, or "
                "cortical column nodes where the neural field was placed on the sheet.  "
                "almost every edge here is a self-edge, so a materialization with neither "
                "support has no positions for the four populations to coexist at",
                "name `tissue` or `cortical_surface` in R")
    else:
        want = [support] if isinstance(support, str) else list(support)

    src_l, dst_l, dist_l, notes = [], [], [], []
    for name in want:
        t = sites.require(name, "microcircuit_within_site",
                          "positions of the sites whose populations interact locally -- "
                          "parenchyma voxels, or cortical column nodes where the neural field "
                          "was placed on the sheet")
        idx = t.gidx(np)
        src, dst = idx, idx
        dist = np.zeros(t.n, dtype=float)

        if radius_mm > 0.0:
            if radius_mm > max_radius_mm:
                raise ValueError(
                    f"microcircuit radius {radius_mm} mm exceeds max_radius_mm "
                    f"{max_radius_mm} mm.  the canonical microcircuit is a few hundred "
                    "microns of cortex; at millimetre range these edges would connect "
                    "populations in different areas.  use the `local` topology for "
                    "millimetre-scale tissue coupling, or raise max_radius_mm "
                    "deliberately if the materialization really is sub-columnar")
            i, j, d = B.pairs_within(np.asarray(t.xyz, dtype=float), float(radius_mm),
                                     "microcircuit_within_site")
            src = np.concatenate([src, i + t.offset, j + t.offset])
            dst = np.concatenate([dst, j + t.offset, i + t.offset])
            dist = np.concatenate([dist, d, d])
        src_l.append(src); dst_l.append(dst); dist_l.append(dist)
        notes.append(f"{name}: {t.n:,} within-site"
                     + ("" if radius_mm <= 0.0
                        else f" plus sub-columnar neighbours within {radius_mm} mm"))

    z = np.zeros(0, dtype=np.int64)
    return B.EdgeSet(
        "microcircuit",
        np.concatenate(src_l) if src_l else z,
        np.concatenate(dst_l) if dst_l else z,
        sites.n_total,
        {"distance_mm": np.concatenate(dist_l) if dist_l else np.zeros(0, dtype=float)},
        directed=False, note="; ".join(notes))


MICROCIRCUIT = REGISTRY.topology(Topology(
    "microcircuit",
    "the canonical local circuit: which populations at one materialized position may act on "
    "one another.  almost every edge is a self-edge, because excitatory, pv, sst and vip "
    "populations occupy the same tissue rather than neighbouring patches of it -- they are "
    "components of the state at a position, so the circuit among them is a relation the "
    "position has with itself and has no distance to report.  it is declared rather than "
    "left implicit so that a pointwise process still names a topology the registry can "
    "check, and so that a sub-columnar materialization can widen it to the few-hundred-"
    "micron lateral reach of sst inhibition without pretending that a millimetre-scale "
    "radius graph is a canonical circuit.  it is declared over the cortical sheet as well as "
    "the parenchyma volume because a relation a position has with itself does not depend on "
    "how the positions were sampled: the four populations coexist at a column node exactly as "
    "they coexist in a voxel, and a materialization that indexes cortex on the sheet would "
    "otherwise lose the canonical circuit entirely",
    on=("tissue", "cortical_surface"),
    edge_features=("distance_mm",),
    directed=False,
    builder="microcircuit_within_site",
    provenance=Provenance.LITERATURE))
