"""vascular adjacency: distance along the tree, not distance in the head.

the clearest case in the inventory of a topology whose metric contradicts
euclidean distance outright.  two capillaries a hundred microns apart in a
cortical column can be at the ends of different penetrating arterioles, and the
path between them along the vasculature runs up one arteriole to a pial artery,
along the surface, and back down another -- tens of millimetres of transport for
a hundred microns of separation.  in the other direction, an arteriole and the
vein draining the same tissue are adjacent in space and are separated by the
entire capillary bed in flow.  anything transported by blood -- oxygen,
deoxyhaemoglobin, glucose, a contrast bolus, a drug, pressure -- moves along the
tree, and a spatial radius graph gets its ordering wrong in both directions.

the tree is also *directed* in a way none of the diffusion-like topologies are.
flow has a sign; upstream pressure sets downstream flow and not the reverse; a
vasodilatory signal at a capillary propagates retrogradely along the endothelium
to the feeding arteriole, which is a different mechanism travelling the other way
along the same edges.  storing the tree with an orientation makes both expressible.

what is carried per edge is the segment's geometry and its hydraulic resistance.
resistance rather than conductance because it is the additive one along a series
path, and because Poiseuille makes it a fourth power of radius: a 10% error in a
segment radius is a 46% error in its resistance, which is why the radius is
carried alongside rather than being folded in and forgotten.  that fourth power
is also why `vascular_prior.theta_prior` widens a conductance prior by FOUR times
the calibre's log spread -- a vein whose diameter is known across a population to
a factor of 1.33 can carry a flow known to a factor of 3.1.

this builder takes a tree and asks nothing about where it came from, which is
right for the macrovasculature and badly wrong below it: no in-vivo human
modality resolves a capillary, so a caller who supplies a capillary bed
synthesised it.  `vascular_prior.py` is where that gets said -- three tiers over
this same topology, a measured statistics table, and a generator whose output is
checked against the flow it has to carry rather than against a histogram.

`interstitial` and `csf` are the other two transport topologies and they are not
this one.  the vascular tree is a plumbed network with walls; the interstitium is
a tortuous connected volume with no channels; the csf spaces are cavities with
bulk flow.  a molecule crossing from one to another crosses a barrier, and those
crossings are the `metabolic_exchange` and `csf` topologies rather than edges here.
"""

from __future__ import annotations

from typing import Any

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance

_PARENT_WHAT = (
    "a (n,) integer array giving each vascular node's parent node index, with -1 "
    "at a root.  it is the tree, and it cannot be inferred from positions: "
    "linking nodes by proximity connects an arteriole to the vein beside it, "
    "which is the one adjacency the vasculature does not have")
_PARENT_WHERE = (
    "a centreline extraction from a segmented angiogram -- TOF-MRA or 7T QSM "
    "venography for the macrovasculature, or a synthetic/measured microvascular "
    "network where capillaries are needed (data/sources: "
    "circle-of-willis-centerline-resources, 7t-qsm-venograms, "
    "high-resolution-vascular-atlases, microscopy-microvascular-networks, "
    "vesselgraph-mouse, fmost-mouse-vasculature-blocks, simvascular, "
    "vascular-model-repository)")
_RADIUS_WHAT = (
    "a (n,) array of vessel radius in mm at each node.  resistance goes as the "
    "inverse fourth power of it, so a tree without radii supports adjacency but "
    "not transport, and substituting a constant radius would make every path's "
    "resistance proportional to its length -- exactly the wrong model, since the "
    "capillary bed is most of the LENGTH (85% of it, measured) and a fifth of the "
    "dissipation, so a length-proportional resistance gets the balance backwards")


@B.builder(
    "vascular_tree_adjacency",
    produces=("distance_mm", "radius_mm", "resistance"),
    supports=("vascular_tree",),
    requires=("parent",),
    directed=True,
    metric="path length along the vascular tree",
    doc="parent-child segments of a vascular tree, with Poiseuille resistance")
def vascular_tree_adjacency(sites, *, support: str = "vascular_tree",
                            parent=None, radius=None,
                            viscosity_pa_s: float | Any = 3.5e-3):
    """segments of the tree, oriented parent to child.

    the resistance is Poiseuille's, R = 8 mu L / (pi r^4), in Pa s m^-3, with mu
    the apparent whole-blood viscosity.

    `viscosity_pa_s` may be a constant or a CALLABLE taking the segment diameter
    in mm and returning a viscosity, which is how the Fahraeus-Lindqvist
    correction enters when a caller wants it.  the default stays constant, and
    the default is wrong in a direction that is now measured rather than merely
    named.  `scripts/measure_microvasculature.py` recovers the apparent viscosity
    implied by every segment's own solved pressure drop in three cortical
    networks, and it is 2.4 mPa s at 4 um, 1.5 mPa s at 6-9 um and 2.2 mPa s
    above 30 um -- so the constant 3.5 mPa s overestimates capillary resistance
    by a factor of 1.5 to 2.4, and the relation is NOT monotone: it has its
    minimum around 7 um and rises again below, which is the Fahraeus-Lindqvist
    inversion.  a caller who applies a monotone "thinner therefore thinner blood"
    correction has the sign right over half the range and wrong over the other
    half.  `ibm.topologies.vascular_prior.MEASURED.viscosity_pa_s` is that
    relation as a callable and is the intended argument.

    the correction is not applied by default because it depends on haematocrit,
    which is state rather than geometry and belongs to a process -- and because
    the measured relation is recovered from a solver's own pressure field, so it
    is what a network NEEDS in order to carry its flow rather than an independent
    rheological measurement.

    the same measurement contradicts this module's own summary of where the
    resistance is.  the capillary bed accounts for 21% of the viscous dissipation
    in those networks, not most of it; the descending arterioles take 37% and the
    pial arteries 32%.  that figure is inflated on the arterial side by the blocks
    being blocks -- their pial vessels are cut and carry the whole block's flow
    through a few segments -- but not by a factor of four, so "the capillary bed
    is most of the resistance" is not right as stated, and the reason it is nearly
    harmless is that the capillary bed IS most of the LENGTH, which is what the
    radius argument below is really about.

    a `flow_sign` column, if present, flips edges whose tree orientation runs
    against the flow.  venous trees are usually rooted at the sinus, so their
    parent-child order is already anti-parallel to flow; getting that wrong sends
    the deoxyhaemoglobin the wrong way, which produces a perfectly smooth and
    perfectly inverted bold response.
    """
    np = B._numpy("vascular_tree_adjacency")
    t = sites.require(support, "vascular_tree_adjacency", "vascular node positions")
    par = parent if parent is not None else t.col(
        "parent", "vascular_tree_adjacency", _PARENT_WHAT, _PARENT_WHERE)
    par = np.asarray(par, dtype=np.int64)
    if len(par) != t.n:
        raise ValueError(f"parent array has {len(par)} entries for {t.n} vascular nodes")

    child = np.nonzero(par >= 0)[0]
    if len(child) == 0:
        return B.empty("vascular", sites.n_total,
                       ("distance_mm", "radius_mm", "resistance"), directed=True,
                       note="every node is a root; the parent array carries no edges")
    up = par[child]

    xyz = np.asarray(t.xyz, dtype=float)
    length = np.linalg.norm(xyz[child] - xyz[up], axis=1)

    rad = radius if radius is not None else t.opt("radius_mm")
    if rad is None:
        raise B.MissingInput("vascular_tree_adjacency",
                             f"{support}.columns['radius_mm']", _RADIUS_WHAT, _PARENT_WHERE)
    rad = np.asarray(rad, dtype=float)
    r_seg = 0.5 * (rad[child] + rad[up])
    r_m = np.maximum(r_seg, 1e-6) * 1e-3
    mu = (viscosity_pa_s(2.0 * r_seg) if callable(viscosity_pa_s)
          else float(viscosity_pa_s))
    resistance = 8.0 * mu * (length * 1e-3) / (np.pi * r_m ** 4)

    src, dst = up, child
    sign = t.opt("flow_sign")
    if sign is not None:
        s = np.asarray(sign, dtype=float)[child] < 0
        src = np.where(s, child, up)
        dst = np.where(s, up, child)

    return B.EdgeSet(
        "vascular", src + t.offset, dst + t.offset, sites.n_total,
        {"distance_mm": length, "radius_mm": r_seg, "resistance": resistance},
        directed=True,
        note=(f"{len(child)} segments, {int((par < 0).sum())} roots, "
              + ("mu from the supplied calibre-dependent relation "
                 "(Fahraeus-Lindqvist applied)" if callable(viscosity_pa_s)
                 else f"mu = {viscosity_pa_s} Pa s constant, which the measurement in "
                      f"data/sources/microscopy-microvascular-networks/evidence/"
                      f"microvasculature@1 puts 1.5-2.4x too high in the capillary bed "
                      f"(Fahraeus-Lindqvist not applied)")))


VASCULAR = REGISTRY.topology(Topology(
    "vascular",
    "adjacency along the cerebral vasculature, from the circle of willis through penetrating "
    "arterioles and capillary beds to venules and sinuses.  its metric is path length along "
    "the tree, which contradicts euclidean distance rather than approximating it: two "
    "capillaries a hundred microns apart may be tens of millimetres apart in flow if they "
    "hang off different penetrating arterioles, and an arteriole and its draining vein touch "
    "in space with the whole capillary bed between them.  everything blood carries -- oxygen, "
    "deoxyhaemoglobin, glucose, a contrast bolus, pressure -- travels this way.  it is "
    "directed because flow has a sign and because retrograde endothelial vasodilatory "
    "signalling runs the other way along the same segments.  resistance is carried per "
    "segment rather than derived downstream because Poiseuille makes it a fourth power of "
    "radius, so it is not recoverable from a length",
    on=("vascular_tree",),
    edge_features=("distance_mm", "radius_mm", "resistance"),
    directed=True,
    builder="vascular_tree_adjacency",
    provenance=Provenance.PHYSICS))
