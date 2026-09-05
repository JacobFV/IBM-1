"""electromagnetic adjacency: who can see whose current, and who can drive whom.

this is the topology with the largest gap between what geometry suggests and what
physics gives.  the quasi-static electric field a transmembrane current produces
at a sensor is not a function of the distance between them.  it is a function of
the whole head: the skull is two orders of magnitude less conductive than brain
and smears and attenuates in a way that depends on its local thickness, the csf
layer beneath it is highly conductive and shunts current tangentially, and the
source's *orientation* relative to the cortical surface decides whether it is
visible at all -- a radial dipole is nearly silent in MEG regardless of how close
it is.  a 1/r^2 falloff over euclidean distance gets the ordering of sensor
sensitivities wrong, not merely the scale.

so the sources-to-sensors half of this topology cannot be built from positions.
its builder takes a lead field and says exactly that, because the fix is to run a
forward solution over a segmented head model, not to pick a better distance
function.  what the builder does do is sparsify: a lead field is dense, most of it
is below the noise floor, and a topology is a statement about which pairs *may*
interact -- so thresholding a computed lead field is a legitimate way to obtain
that statement, and the threshold is relative to each sensor's own maximum so that
a deep source is not deleted for being deep.

the tissue-to-tissue half is different and is buildable.  volume conduction inside
the parenchyma is local, quasi-static and instantaneous: a position's extracellular
potential is influenced by transmembrane currents in its neighbourhood, with the
1/r kernel of a conductive medium, and there is no skull in between.  that is a
radius graph with an orientation, and it is what `em_generation` acts over.  it is
kept in this module rather than folded into `local` because its geometry is the
field's, not the tissue's: it needs the dipole orientation, it is quasi-static
rather than diffusive, and its radius is set by the noise floor of a potential
rather than by a diffusion length.

instantaneity is the other thing that separates this from every other topology
here.  at these frequencies and these distances the quasi-static approximation
holds to well within a microsecond, so em edges carry no delay -- which is exactly
why a tractometric edge carries one and an em edge does not, and why the two
cannot share a topology even where they connect the same positions.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance

_LEADFIELD_WHAT = (
    "an (n_sensor, n_source) or (n_sensor, n_source, 3) array giving the signal "
    "each sensor sees per unit source current, computed for this subject's head.  "
    "it cannot be derived from the positions: it depends on skull thickness and "
    "conductivity, on the csf layer, and on source orientation relative to the "
    "cortical sheet, and those change the ranking of sensor sensitivities and not "
    "just its scale")
_LEADFIELD_WHERE = (
    "a BEM forward solution over a three-layer head model (mne-python "
    "make_bem_model / make_forward_solution), or an FEM solution over a "
    "tissue-segmented mesh where the skull is complicated or a stimulation field "
    "is wanted (simnibs charm, duneuro, fieldtrip) (data/sources: mne-python, "
    "simnibs, duneuro, fieldtrip, k-wave for the acoustic analogue)")


@B.builder(
    "em_lead_field",
    produces=("distance_mm", "orientation"),
    supports=("tissue", "sensor_array"),
    requires=("lead_field",),
    directed=True,
    metric="lead-field sensitivity, which is not a function of distance",
    doc="source-to-sensor edges wherever a computed lead field is above threshold")
def em_lead_field(sites, *, lead_field=None, source_support: str = "tissue",
                  sensor_support: str = "sensor_array", threshold: float = 1e-3):
    """which sources a sensor can see, from a forward solution.

    `threshold` is relative to each sensor's own largest entry, so the sparsity is
    a statement about that sensor's field of view rather than about depth.  an
    absolute threshold would delete every deep source from every sensor, which is
    a true statement about signal-to-noise and a false one about the topology --
    the deep source does reach the sensor, and whether the data can resolve it is
    the inverse problem's business, not the graph's.
    """
    np = B._numpy("em_lead_field")
    s = sites.require(source_support, "em_lead_field",
                      "positions of the current sources (tissue sites)")
    m = sites.require(sensor_support, "em_lead_field",
                      "positions of the sensing elements")
    if lead_field is None:
        raise B.MissingInput("em_lead_field", "lead_field", _LEADFIELD_WHAT, _LEADFIELD_WHERE)
    L = np.asarray(lead_field, dtype=float)
    if L.ndim == 3:
        L = np.linalg.norm(L, axis=2)               # magnitude over the free orientation
    if L.shape != (m.n, s.n):
        raise ValueError(
            f"lead field has shape {L.shape} but the materialization has {m.n} "
            f"{sensor_support} elements and {s.n} {source_support} sources; a lead "
            "field computed for a different source space cannot be reindexed here")

    mag = np.abs(L)
    peak = np.maximum(mag.max(axis=1, keepdims=True), 1e-300)
    keep = mag >= (float(threshold) * peak)
    si, sj = np.nonzero(keep)                        # sensor row, source column
    sxyz = np.asarray(s.xyz, dtype=float)
    mxyz = np.asarray(m.xyz, dtype=float)
    d = np.linalg.norm(mxyz[si] - sxyz[sj], axis=1)
    u = B.unit(mxyz[si] - sxyz[sj], np)

    return B.EdgeSet(
        "electromagnetic", sj + s.offset, si + m.offset, sites.n_total,
        {"distance_mm": d, "orientation": u}, directed=True,
        note=(f"{len(si)} source->sensor edges from a lead field, kept at "
              f"{threshold:g} of each sensor's peak; no delay, the quasi-static "
              "approximation makes these edges instantaneous"))


@B.builder(
    "em_volume_conduction",
    produces=("distance_mm", "orientation"),
    supports=("tissue",),
    directed=False,
    metric="euclidean distance in a conductive medium, with source orientation",
    doc="local quasi-static coupling between tissue positions")
def em_volume_conduction(sites, *, support: str = "tissue", radius_mm: float | None = None):
    """the tissue-internal half: which positions' currents contribute to whose potential.

    inside the parenchyma there is no skull, the medium is close enough to
    homogeneous over a few millimetres, and the field falls off as 1/r from a
    monopole and 1/r^2 from a dipole -- so a radius graph with an orientation is
    the right structure, and the radius is set by where the contribution drops
    below the noise rather than by any transport length.  it defaults to 5 mm,
    which is roughly where a cortical dipole layer's contribution to an
    extracellular potential stops mattering against the local sources.
    """
    np = B._numpy("em_volume_conduction")
    t = sites.require(support, "em_volume_conduction", "tissue site positions")
    r = float(radius_mm) if radius_mm is not None else 5.0
    i, j, d = B.pairs_within(np.asarray(t.xyz, dtype=float), r, "em_volume_conduction")
    if len(d) == 0:
        return B.empty("electromagnetic", sites.n_total, ("distance_mm", "orientation"))
    xyz = np.asarray(t.xyz, dtype=float)
    return B.EdgeSet(
        "electromagnetic", i + t.offset, j + t.offset, sites.n_total,
        {"distance_mm": d, "orientation": B.unit(xyz[j] - xyz[i], np)}, directed=False,
        note=f"quasi-static volume conduction within {r:.3g} mm; instantaneous by construction")


ELECTROMAGNETIC = REGISTRY.topology(Topology(
    "electromagnetic",
    "which currents contribute to which measured or imposed field.  it is the topology whose "
    "geometry least resembles distance: the skull is a hundred times less conductive than "
    "brain and the csf beneath it shunts current tangentially, so a sensor's sensitivity to "
    "a source depends on skull thickness, csf geometry and the source's orientation relative "
    "to the cortical sheet -- a radial dipole is nearly invisible to MEG however close it "
    "is.  no distance function reproduces that ordering, which is why the source-to-sensor "
    "builder takes a computed lead field rather than positions, and only sparsifies it.  "
    "inside the parenchyma there is no skull and the coupling really is a local radius graph "
    "with an orientation.  unlike the tractometric topology these edges carry no delay: at "
    "these frequencies and scales the quasi-static approximation holds, so the coupling is "
    "instantaneous, and that difference alone makes them different topologies over the same "
    "positions",
    on=("tissue", "head_volume", "scalp", "sensor_array", "implanted_array", "stimulator"),
    edge_features=("distance_mm", "orientation"),
    directed=True,
    builder="em_lead_field",
    provenance=Provenance.PHYSICS))
