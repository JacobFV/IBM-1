"""csf adjacency: bulk flow in the cavities, and the exchange across their wall.

the cerebrospinal spaces are not a diffusion medium and not a plumbed tree.  they
are cavities -- ventricles, cisterns, the subarachnoid space -- in which fluid
moves by bulk flow driven by a pressure gradient that pulses with the cardiac and
respiratory cycles, and the relevant adjacency is between neighbouring parcels of
fluid in a channel.  the geometry that matters is a cross-sectional area and a
length, because a channel's hydraulic resistance is d / A and the aqueduct's
narrowness is the reason the aqueduct dominates the pressure drop.  distance alone
would make the aqueduct look like any other three millimetres of csf.

the second and less obvious half of this topology is the boundary.  the csf
compartment and the interstitium are separated by a wall that is not sealed:
along penetrating vessels the perivascular spaces carry csf into the parenchyma
and interstitial fluid back out, and solutes -- amyloid, tau, lactate, an
intrathecal drug -- cross there rather than everywhere.  the crossing is what
makes csf part of the state graph at all instead of a boundary condition, and it
is a *different kind of edge* from either compartment's internal adjacency: it
joins two supports, it is limited by a wall area rather than a channel section,
and its resistance is a permeability rather than a geometry.

both edge populations are returned in one set because they are one topology --
`csf_flow` and `csf_interstitial_exchange` are separate processes over it.  they
are told apart by which side of the interstitial table's offset each endpoint
falls on, which is why the site tables carry offsets at all.

what this is not: `vascular`.  the perivascular route runs alongside the arteries
and shares their geometry, but its flow is not the blood's, its driving pressure
is arterial pulsation transmitted to a different compartment, and a molecule in
the perivascular space has not entered the blood.  conflating them would let
anything in csf reach anywhere the blood reaches, instantly.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance


@B.builder(
    "csf_compartment",
    produces=("distance_mm", "contact_area_mm2", "resistance"),
    supports=("csf_space", "interstitial"),
    directed=False,
    metric="channel length over cross-section within csf; wall area across the boundary",
    doc="bulk-flow adjacency inside the csf spaces plus perivascular exchange with tissue")
def csf_compartment(sites, *, support: str = "csf_space",
                    interstitial_support: str = "interstitial",
                    radius_mm: float | None = None, exchange_mm: float = 1.0,
                    exchange_k: int = 4, permeability: float = 1.0):
    """the two edge populations of the csf compartment.

    `exchange_mm` defaults to a millimetre because that is roughly the reach of a
    perivascular space from a pial or ventricular surface at the resolution a
    whole-brain materialization instantiates; it is a proxy for a route whose
    real geometry follows the penetrating vessels, and a materialization that has
    the vascular tree should restrict the exchange to sites near a penetrating
    vessel rather than to sites near the boundary.  that restriction is not made
    here because it needs the vascular table, and the honest failure of the
    proxy -- exchange happening across the whole pial surface instead of along
    vessels -- is a smoothing, not a wrong direction.

    `permeability` scales the exchange resistance.  it is 1.0 by default and is
    not a measurement: net perivascular exchange rates in human are inferred
    rather than measured, they vary several-fold between sleep and waking, and
    treating the number as known is how a glymphatic model becomes untestable.
    """
    np = B._numpy("csf_compartment")
    t = sites.require(support, "csf_compartment", "csf compartment sampling positions")
    xyz = np.asarray(t.xyz, dtype=float)
    sp = t.spacing(np)
    r = float(radius_mm) if radius_mm is not None else 2.0 * float(np.max(sp))

    i, j, d = B.pairs_within(xyz, r, "csf_compartment")
    area = np.minimum(sp[i], sp[j]) ** 2 if len(d) else np.zeros(0, dtype=float)
    # a channel's hydraulic resistance with viscosity factored out: the geometric
    # part is d / A, and a cross-section column overrides the lattice estimate
    # wherever the materialization actually knows the channel width.
    xs = t.opt("cross_section_mm2")
    if xs is not None and len(d):
        x = np.asarray(xs, dtype=float)
        area = np.minimum(x[i], x[j])
    res = d / np.maximum(area, 1e-12) if len(d) else np.zeros(0, dtype=float)

    src = i + t.offset
    dst = j + t.offset
    dist, cont, resis = d, area, res
    n_intra = len(d)

    isf = sites.get(interstitial_support)
    n_exch = 0
    if isf is not None and isf.n and t.n:
        # nearest csf parcels to each interstitial site, kept only where the two
        # compartments actually abut.  the interstitium is the query side because
        # every parenchymal site has a nearest csf parcel while most csf parcels
        # face many parenchymal ones.
        idx, dd = B.nearest(np.asarray(isf.xyz, dtype=float), xyz,
                            int(exchange_k), "csf_compartment")
        ok = dd <= float(exchange_mm)
        if ok.any():
            rows = np.repeat(np.arange(isf.n), idx.shape[1])[ok.ravel()]
            cols = idx.ravel()[ok.ravel()]
            de = dd.ravel()[ok.ravel()]
            isp = isf.spacing(np)
            wall = np.minimum(isp[rows], sp[cols]) ** 2
            rex = de / (np.maximum(wall, 1e-12) * max(float(permeability), 1e-12))
            src = np.concatenate([src, cols + t.offset])
            dst = np.concatenate([dst, rows + isf.offset])
            dist = np.concatenate([dist, de])
            cont = np.concatenate([cont, wall])
            resis = np.concatenate([resis, rex])
            n_exch = len(de)

    return B.EdgeSet(
        "csf", src, dst, sites.n_total,
        {"distance_mm": dist, "contact_area_mm2": cont, "resistance": resis},
        directed=False,
        note=(f"{n_intra} intra-csf channel edges (radius {r:.3g} mm) and {n_exch} "
              f"perivascular exchange edges (within {exchange_mm} mm); an edge is an "
              f"exchange edge when one endpoint is at or above the interstitial table's "
              f"offset"))


CSF = REGISTRY.topology(Topology(
    "csf",
    "adjacency within the cerebrospinal fluid spaces and across their wall into the "
    "interstitium.  inside the cavities the transport is bulk flow driven by a pulsatile "
    "pressure gradient, so the geometry that matters is channel length over cross-section "
    "rather than distance -- the cerebral aqueduct dominates the pressure drop because it is "
    "narrow, and a topology carrying only millimetres could not express that.  the second "
    "edge population crosses the compartment boundary: perivascular spaces along penetrating "
    "vessels exchange csf and interstitial fluid, which is how a solute leaves the "
    "parenchyma and why the csf compartment is part of the state graph rather than a "
    "boundary condition.  it is not the vascular topology -- the perivascular route runs "
    "beside the arteries and shares neither their fluid nor their contents",
    on=("csf_space", "interstitial"),
    edge_features=("distance_mm", "contact_area_mm2", "resistance"),
    directed=False,
    builder="csf_compartment",
    provenance=Provenance.PHYSICS))
