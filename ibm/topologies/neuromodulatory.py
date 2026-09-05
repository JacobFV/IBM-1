"""neuromodulatory projections: a few thousand cells addressing the whole brain.

structurally the strangest topology here, and it is strange because the anatomy is.
the human locus coeruleus is roughly fifty thousand neurons and it innervates the
entire neocortex, hippocampus, cerebellum and cord; the dorsal raphe and the
dopaminergic midbrain are of the same order.  a single one of these axons branches
over centimetres and releases transmitter from varicosities along its length
rather than only at synapses.  so the graph is not sparse and not reciprocal: it is
a small set of sources times a very large set of targets, one-way, and the
sparsity a normal connectivity topology relies on to stay affordable does not
exist.  that is a property of the anatomy and the builder exposes it as one --
`target_stride` subsamples targets and says in the note that it did, rather than
thresholding edges into a plausible-looking sparse graph that no longer means
"innervates".

the metric is the most compromised in the inventory and the compromise should be
read rather than assumed.  what a neuromodulatory edge would want is the arc
length of the axon from the nucleus to the target, and no measurement of that
exists in human: diffusion tractography cannot follow an unmyelinated axon of
under a micron through the white matter, and the ascending bundles are visible
only as a whole.  so the length is a scaled euclidean distance, with the scale
factor being how much longer a real axonal path is than the straight line -- about
1.5 to 2 for a route that ascends through the medial forebrain bundle and then
turns.  it is a lower bound wearing a correction factor, and it is declared as one.

the delay is correspondingly slow and correspondingly important.  these axons are
thin and largely unmyelinated, so conduction runs around 0.5 to 2 m/s and a
brainstem-to-frontal-cortex transit takes tens of milliseconds -- an order of
magnitude longer than any cortico-cortical delay in the tractometric topology.  a
model that gave neuromodulation cortico-cortical latencies would have arousal
arriving before the event that caused it.

and the reason this is not the tractometric topology, sharing edges and a builder:
these projections write *parameters*, not state.  a neuromodulatory process
changes gain, adaptation and plasticity rate at its targets rather than adding to
their input, which ARCHITECTURE.md §4 handles through `writes="parameters"`.  the
target set is therefore chosen by receptor availability rather than by axonal
density -- a projection that reaches tissue with no receptors for its transmitter
does nothing -- which is why the builder takes a receptor map and why the tract
one does not.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance

#: the ascending sources, by the brainstem_nuclei labels they are declared under,
#: with the conduction velocity of their fibre class.  the velocities are fibre
#: typicals at one significant figure: thin, poorly myelinated ascending axons.
SOURCE_NUCLEI: dict[str, dict] = {
    "noradrenaline": dict(labels=("locus_coeruleus",), velocity_m_s=0.7),
    "serotonin": dict(labels=("dorsal_raphe", "median_raphe"), velocity_m_s=0.7),
    "dopamine": dict(labels=("ventral_tegmental_area", "substantia_nigra_compacta"),
                     velocity_m_s=1.0),
    "acetylcholine": dict(labels=("nucleus_basalis_of_meynert",
                                  "medial_septum_diagonal_band",
                                  "pedunculopontine", "laterodorsal_tegmental"),
                          velocity_m_s=1.5),
    "histamine": dict(labels=("tuberomammillary",), system="hypothalamic_nuclei",
                      velocity_m_s=0.5),
}


@B.builder(
    "neuromodulatory_diffuse",
    produces=("distance_mm", "tract_length_mm", "conduction_delay_s"),
    supports=("tissue",),
    directed=True,
    metric="scaled euclidean distance as a proxy for unmeasurable axonal arc length",
    doc="one-to-many projections from a named nucleus to everything it innervates")
def neuromodulatory_diffuse(sites, *, support: str = "tissue",
                            transmitter: str | None = None,
                            system: str = "brainstem_nuclei", source_labels=None,
                            velocity_m_s: float | None = None,
                            source_threshold: float = 0.2,
                            tortuosity_factor: float = 1.7,
                            receptor_column: str | None = None,
                            receptor_threshold: float = 0.0,
                            target_stride: int = 1, max_edges: int = 20_000_000):
    """edges from a nucleus's sites to every target site it innervates.

    the source set comes from a partitioning system rather than from a mask,
    because that is where "locus coeruleus" is defined in this ontology -- and
    because it makes the localization uncertainty visible: `source_threshold` is a
    cut on a probabilistic membership whose spatial error is the warp residual,
    which for a structure the width of the locus coeruleus is comparable to the
    structure.  a materialization that has neuromelanin imaging should pass its own
    mask through `source_labels=None` and a partition it built itself.

    `receptor_column` restricts targets to tissue that expresses a receptor for
    the transmitter.  it is the honest way to make this graph smaller: a projection
    to tissue with no receptor is not a weak edge, it is not an edge, whereas
    dropping distant targets to save memory would delete real innervation.
    """
    np = B._numpy("neuromodulatory_diffuse")
    t = sites.require(support, "neuromodulatory_diffuse",
                      "tissue positions, including the source nuclei")

    if source_labels is None:
        if transmitter is None:
            raise B.MissingInput(
                "neuromodulatory_diffuse", "transmitter or source_labels",
                "which ascending system to build: one of "
                f"{', '.join(sorted(SOURCE_NUCLEI))}, or an explicit tuple of labels "
                f"from a registered partitioning system",
                "ibm.topologies.neuromodulatory.SOURCE_NUCLEI lists the declared "
                "sources and their fibre velocities")
        if transmitter not in SOURCE_NUCLEI:
            raise KeyError(f"no declared source for {transmitter!r}; "
                           f"known: {', '.join(sorted(SOURCE_NUCLEI))}")
        spec = SOURCE_NUCLEI[transmitter]
        source_labels = spec["labels"]
        system = spec.get("system", system)
        if velocity_m_s is None:
            velocity_m_s = spec["velocity_m_s"]
    if velocity_m_s is None:
        velocity_m_s = 1.0

    part = t.partitions.get(system)
    if part is None:
        raise B.MissingInput(
            "neuromodulatory_diffuse", f"{support}.partitions[{system!r}]",
            f"an (n, k) membership over the {system!r} partitioning system, so the "
            f"source sites of {tuple(source_labels)!r} can be identified.  the "
            "projection originates from a named nucleus and there is no way to find "
            "one from positions alone",
            "ibm.anatomy.systems declares the system; ibm.anatomy.sources records "
            "that the locus coeruleus and raphe are the least reliably localized "
            "structures in it, and that the error is a displacement rather than noise")
    known = REGISTRY.anatomies[system].labels
    cols = [known.index(l) for l in source_labels]
    w = np.asarray(part, dtype=float)[:, cols].sum(axis=1)
    src_rows = np.nonzero(w >= float(source_threshold))[0]
    if len(src_rows) == 0:
        return B.empty("neuromodulatory_projection", sites.n_total,
                       ("distance_mm", "tract_length_mm", "conduction_delay_s"),
                       directed=True,
                       note=(f"no site reaches membership {source_threshold} in "
                             f"{tuple(source_labels)!r}; at a whole-brain spacing these "
                             "nuclei are smaller than a voxel"))

    tgt = np.arange(0, t.n, max(1, int(target_stride)))
    if receptor_column is not None:
        rec = t.col(receptor_column, "neuromodulatory_diffuse",
                    "a (n,) receptor or transporter availability per tissue site for "
                    "this transmitter; targets below threshold are not innervated in "
                    "the sense that matters, since the projection acts through receptors",
                    "PET receptor and transporter density maps (data/sources: "
                    "hansen-receptors, neuromaps)")
        rec = np.asarray(rec, dtype=float)
        tgt = tgt[rec[tgt] > float(receptor_threshold)]
    tgt = tgt[~np.isin(tgt, src_rows)]
    if len(tgt) == 0:
        return B.empty("neuromodulatory_projection", sites.n_total,
                       ("distance_mm", "tract_length_mm", "conduction_delay_s"),
                       directed=True, note="no target site survived the receptor filter")

    n_edges = len(src_rows) * len(tgt)
    if n_edges > int(max_edges):
        raise ValueError(
            f"{len(src_rows)} source sites x {len(tgt)} targets = {n_edges} edges, "
            f"over max_edges={max_edges}.  this topology is genuinely dense -- a few "
            "thousand cells innervate the whole brain -- so the fix is to subsample "
            "targets with target_stride, or to restrict them by receptor availability "
            "with receptor_column, not to threshold edges by distance: a distant "
            "target of a neuromodulatory projection is innervated exactly as much as "
            "a near one")

    xyz = np.asarray(t.xyz, dtype=float)
    s = np.repeat(src_rows, len(tgt))
    d = np.tile(tgt, len(src_rows))
    chord = np.linalg.norm(xyz[d] - xyz[s], axis=1)
    length = float(tortuosity_factor) * chord
    delay = B.conduction_delay_s(length, np.full(len(length), float(velocity_m_s)), np)

    return B.EdgeSet(
        "neuromodulatory_projection", s + t.offset, d + t.offset, sites.n_total,
        {"distance_mm": chord, "tract_length_mm": length, "conduction_delay_s": delay},
        directed=True,
        note=(f"{len(src_rows)} source sites in {tuple(source_labels)} -> {len(tgt)} "
              f"targets (stride {target_stride}) at {velocity_m_s} m/s, path length "
              f"{tortuosity_factor}x euclidean -- a lower bound with a correction "
              f"factor, since no human measurement of these axonal paths exists; "
              f"median delay {float(np.median(delay)) * 1e3:.0f} ms"))


NEUROMODULATORY_PROJECTION = REGISTRY.topology(Topology(
    "neuromodulatory_projection",
    "one-to-many ascending projections from the brainstem and basal forebrain nuclei to "
    "essentially everything.  it is dense where every other long-range topology is sparse: a "
    "few tens of thousands of locus coeruleus neurons innervate the whole cortex, "
    "hippocampus and cerebellum, and a single axon releases transmitter from varicosities "
    "along centimetres of its length rather than at a synapse.  its metric is the weakest "
    "here and is declared as such -- axonal arc length for a sub-micron unmyelinated fibre "
    "is unmeasurable in human, so path length is euclidean distance scaled by a route "
    "factor.  the delays that follow are an order of magnitude longer than any "
    "cortico-cortical one, tens of milliseconds, because these fibres are thin and slow, and "
    "giving neuromodulation tractometric latencies would have arousal arrive before its "
    "cause.  it is separate from the tractometric topology above all because these "
    "projections write parameters rather than state -- they set gain, adaptation and "
    "plasticity rate -- so their targets are selected by receptor availability rather than "
    "by axonal density",
    on=("tissue",),
    edge_features=("distance_mm", "tract_length_mm", "conduction_delay_s"),
    directed=True,
    builder="neuromodulatory_diffuse",
    provenance=Provenance.LITERATURE))
