"""afferent pathways: receptor surface to cortex, through named relays.

the topology that cannot be built from geometry at all, and the reason is
coordinates rather than difficulty.  the retina lives in an eye-centred frame, the
cochlea is parameterized by position along the basilar membrane, the skin is a
two-dimensional surface on the body, and their targets are in the head.  the
euclidean distance between a retinal position and its LGN target is a number about
two frames that were never registered to each other, and it has nothing to do with
the length of the optic nerve.  the adjacency here is *topographic*: a retinal
position connects to the LGN position with the same visual-field coordinate, a
cochlear position to the inferior colliculus position with the same characteristic
frequency, a patch of skin to the VPL position with the same body-surface
coordinate.  the map is the anatomy, and the map is a measurement.

so the metric is a receptotopic one -- distance in visual field, in octaves, in
body-surface coordinates -- and the *physical* length of each stage is a separate
number that comes from anatomy rather than from the site positions.  that split is
why `tract_length_mm` is an explicit per-stage quantity here while the tractometric
topology measures it: there is no image in which the optic nerve is a path between
two materialized sites.

delay dominates this topology in a way it does not dominate the cortical one.  the
periphery is where the model's timing is set: a foot afferent travels a metre at
50 m/s before anything central happens, an auditory brainstem response has
millisecond-precise waves because the stages are short and fast, and an olfactory
receptor axon is unmyelinated and takes tens of milliseconds to go a centimetre.
these are not corrections to a central model, they are the largest latencies in
it, and they are properties of the pathway rather than of any process over it.

the stage table below is anatomy at one significant figure.  it is written down so
that a materialization has something to build with, and it is stated as
order-of-magnitude precisely so that nobody mistakes it for a measurement -- path
lengths vary with body size by a factor of two between adults, and conduction
velocity estimates for the same fibre class differ by a factor of two between
methods.  a process that depends sharply on one of these should fit it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance

_MAP_WHERE = (
    "a receptotopic map measured in the subject or warped from a template: "
    "population receptive field mapping or the Benson template for visual field "
    "coordinates, tonotopic mapping for characteristic frequency, a somatotopy "
    "localizer for body-surface coordinates (data/sources: "
    "benson-neuropythy-retinotopic-atlas, hcp-7t-retinotopy, "
    "hcp-7t-retinotopy-benson-maps, tonotopy-maps, whole-body-somatotopy-fmri)")


def _select(np, table, spec, builder: str, side: str):
    """rows of a site table participating in one stage.

    `spec` is None (all rows), a boolean or index array, or a (system, labels)
    pair read out of the table's partitions -- which is how a stage says "the LGN
    part of the tissue table" without the materializer having had to instantiate
    the LGN as its own support.
    """
    if spec is None:
        return np.arange(table.n)
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        system, labels = spec
        part = table.partitions.get(system)
        if part is None:
            raise B.MissingInput(
                builder, f"{table.support}.partitions[{system!r}]",
                f"memberships over the {system!r} partitioning system, so the {side} "
                f"end of this stage can be restricted to {tuple(labels)!r}",
                "ibm.anatomy.systems declares the system and ibm.anatomy.sources "
                "records which atlas supplies it")
        known = REGISTRY.anatomies[system].labels
        cols = [known.index(l) for l in labels]
        w = np.asarray(part, dtype=float)[:, cols].sum(axis=1)
        return np.nonzero(w > 0.0)[0]
    a = np.asarray(spec)
    return np.nonzero(a)[0] if a.dtype == bool else a.astype(np.int64)


def _coords(np, table, rows, key, builder: str, side: str):
    if key is None:
        return np.asarray(table.xyz, dtype=float)[rows]
    c = table.opt(key)
    if c is None:
        raise B.MissingInput(
            builder, f"{table.support}.columns[{key!r}]",
            f"the receptotopic coordinate of each {side} site -- the quantity the "
            "pathway is organized by (visual field position, characteristic "
            "frequency, body-surface coordinate).  the two ends of an afferent stage "
            "live in different coordinate frames, so they cannot be matched by "
            "position; the map is the connection", _MAP_WHERE)
    c = np.asarray(c, dtype=float)
    return (c[rows] if c.ndim > 1 else c[rows][:, None])


def staged_edges(topology: str, builder: str, sites, stages: Sequence[Mapping[str, Any]],
                 default_k: int = 1):
    """edges for a chain of relay stages, matched by receptotopic coordinate.

    shared by the afferent and efferent topologies because the relay algebra is
    identical and only the direction of travel differs -- keeping one
    implementation is how the two stay consistent rather than drifting into two
    slightly different notions of what a stage is.

    each stage is a mapping with:
        from, to          support names
        map               the coordinate column matched on, on both tables
        length_mm         axonal path length of the stage (a number, not a distance
                          between sites -- the two ends are in different frames)
        velocity_m_s      conduction velocity of the fibre class
        k                 fan-in: how many source sites converge on each target
        select_from/to    None, a mask, or (system, labels) to restrict an end
    """
    np = B._numpy(builder)
    if not stages:
        raise B.MissingInput(
            builder, "stages",
            "a sequence of relay stages, each naming the two supports it joins, the "
            "coordinate column the projection is topographic in, the axonal path "
            "length of the stage in mm and the conduction velocity of its fibres.  "
            "the pathway is not recoverable from the site positions: the receptor "
            "surfaces are in body- and eye-centred frames and their targets are in "
            "the head, so the euclidean distance between them is a statement about "
            "two unregistered frames",
            "ibm.topologies.afferent.HUMAN_AFFERENT_STAGES and "
            "ibm.topologies.efferent.HUMAN_EFFERENT_STAGES carry literature chains at "
            "one significant figure; pass your own to override them")

    src_l, dst_l, len_l, del_l, notes = [], [], [], [], []
    for st in stages:
        a = sites.require(st["from"], builder,
                          f"the source end of the {st.get('name', '?')} stage")
        b = sites.require(st["to"], builder,
                          f"the target end of the {st.get('name', '?')} stage")
        ra = _select(np, a, st.get("select_from"), builder, "source")
        rb = _select(np, b, st.get("select_to"), builder, "target")
        if not len(ra) or not len(rb):
            notes.append(f"{st.get('name', '?')}: no sites materialized")
            continue
        key = st.get("map")
        ca = _coords(np, a, ra, key, builder, "source")
        cb = _coords(np, b, rb, key, builder, "target")
        if ca.shape[1] != cb.shape[1]:
            raise ValueError(
                f"stage {st.get('name', '?')!r} matches on {key!r}, but the source "
                f"coordinate has {ca.shape[1]} dimensions and the target has "
                f"{cb.shape[1]}; a topographic map has to be between the same quantity")
        k = int(st.get("k", default_k))
        idx, _ = B.nearest(cb, ca, k, builder)         # for each target, its k sources
        rows = np.repeat(np.arange(len(rb)), idx.shape[1])
        cols = idx.ravel()
        L = float(st["length_mm"])
        v = float(st["velocity_m_s"])
        src_l.append(ra[cols] + a.offset)
        dst_l.append(rb[rows] + b.offset)
        len_l.append(np.full(len(rows), L))
        del_l.append(np.full(len(rows), L * 1e-3 / max(v, 1e-6)))
        notes.append(f"{st.get('name', '?')}: {len(rows)} edges, {L:g} mm at {v:g} m/s "
                     f"({L * 1e-3 / max(v, 1e-6) * 1e3:.1f} ms)")

    if not src_l:
        return B.empty(topology, sites.n_total,
                       ("tract_length_mm", "conduction_delay_s"), directed=True,
                       note="; ".join(notes) or "no stage produced edges")
    return B.EdgeSet(
        topology, np.concatenate(src_l), np.concatenate(dst_l), sites.n_total,
        {"tract_length_mm": np.concatenate(len_l),
         "conduction_delay_s": np.concatenate(del_l)},
        directed=True, note="; ".join(notes))


#: literature relay chains, at one significant figure.  path lengths are adult
#: anatomy and scale with body size; velocities are fibre-class typicals and
#: differ by a factor of two between estimation methods.  they are here so a
#: materialization has something to build with, not because they are measured.
HUMAN_AFFERENT_STAGES: dict[str, tuple[dict, ...]] = {
    "visual": (
        dict(name="retina->lgn", **{"from": "retina"}, to="tissue", map="visual_field",
             select_to=("thalamic_nuclei", ("lgn",)), length_mm=80.0, velocity_m_s=12.0, k=1),
        dict(name="lgn->v1", **{"from": "tissue"}, to="tissue", map="visual_field",
             select_from=("thalamic_nuclei", ("lgn",)),
             select_to=("cytoarchitecture", ("ba_17",)),
             length_mm=90.0, velocity_m_s=15.0, k=1),
    ),
    "auditory": (
        dict(name="cochlea->cochlear_nuclei", **{"from": "cochlea"}, to="tissue",
             map="tonotopy", select_to=("brainstem_nuclei", ("cochlear_nuclei",)),
             length_mm=25.0, velocity_m_s=20.0, k=1),
        dict(name="cochlear_nuclei->ic", **{"from": "tissue"}, to="tissue", map="tonotopy",
             select_from=("brainstem_nuclei", ("cochlear_nuclei",)),
             select_to=("brainstem_nuclei", ("inferior_colliculus",)),
             length_mm=20.0, velocity_m_s=15.0, k=1),
        dict(name="ic->mgn", **{"from": "tissue"}, to="tissue", map="tonotopy",
             select_from=("brainstem_nuclei", ("inferior_colliculus",)),
             select_to=("thalamic_nuclei", ("mgn",)),
             length_mm=15.0, velocity_m_s=15.0, k=1),
        dict(name="mgn->a1", **{"from": "tissue"}, to="tissue", map="tonotopy",
             select_from=("thalamic_nuclei", ("mgn",)),
             select_to=("cytoarchitecture", ("ba_41",)),
             length_mm=30.0, velocity_m_s=15.0, k=1),
    ),
    "somatosensory": (
        dict(name="skin->dorsal_column_nuclei", **{"from": "body_surface"}, to="tissue",
             map="somatotopy",
             select_to=("brainstem_nuclei", ("cuneate_nucleus", "gracile_nucleus")),
             length_mm=600.0, velocity_m_s=50.0, k=1),
        dict(name="dcn->vpl", **{"from": "tissue"}, to="tissue", map="somatotopy",
             select_from=("brainstem_nuclei", ("cuneate_nucleus", "gracile_nucleus")),
             select_to=("thalamic_nuclei", ("vpl", "vpm")),
             length_mm=120.0, velocity_m_s=40.0, k=1),
        dict(name="vpl->s1", **{"from": "tissue"}, to="tissue", map="somatotopy",
             select_from=("thalamic_nuclei", ("vpl", "vpm")),
             select_to=("cytoarchitecture", ("ba_3", "ba_1", "ba_2")),
             length_mm=40.0, velocity_m_s=20.0, k=1),
    ),
    "vestibular": (
        dict(name="labyrinth->vestibular_nuclei", **{"from": "vestibular_organ"},
             to="tissue", map=None,
             select_to=("brainstem_nuclei", ("vestibular_nuclei",)),
             length_mm=10.0, velocity_m_s=25.0, k=1),
    ),
    "interoceptive": (
        dict(name="viscera->nts", **{"from": "viscera"}, to="tissue", map=None,
             select_to=("brainstem_nuclei", ("nucleus_tractus_solitarius",)),
             length_mm=300.0, velocity_m_s=5.0, k=1),
        dict(name="nts->parabrachial", **{"from": "tissue"}, to="tissue", map=None,
             select_from=("brainstem_nuclei", ("nucleus_tractus_solitarius",)),
             select_to=("brainstem_nuclei", ("parabrachial",)),
             length_mm=15.0, velocity_m_s=5.0, k=1),
    ),
    "olfactory": (
        dict(name="epithelium->bulb", **{"from": "chemosensory_epithelium"}, to="tissue",
             map=None, length_mm=10.0, velocity_m_s=0.5, k=1),
    ),
}


@B.builder(
    "afferent_relay",
    produces=("tract_length_mm", "conduction_delay_s"),
    supports=("retina", "cochlea", "body_surface", "vestibular_organ",
              "chemosensory_epithelium", "viscera", "tissue"),
    requires=("stages",),
    directed=True,
    metric="receptotopic coordinate distance; path length is per-stage anatomy",
    doc="relay-staged sensory pathways, matched by topographic coordinate")
def afferent_relay(sites, *, stages=None, pathway: str | None = None, default_k: int = 1):
    """build one or more afferent chains.

    `pathway` names an entry of HUMAN_AFFERENT_STAGES; `stages` overrides it with
    an explicit chain.  giving neither raises, with the chain table named, because
    a silently empty afferent topology means a model with no input and nothing to
    say about why.
    """
    if stages is None and pathway is not None:
        if pathway not in HUMAN_AFFERENT_STAGES:
            raise KeyError(f"no afferent chain {pathway!r}; "
                           f"known: {', '.join(sorted(HUMAN_AFFERENT_STAGES))}")
        stages = HUMAN_AFFERENT_STAGES[pathway]
    return staged_edges("afferent_pathway", "afferent_relay", sites, stages or (),
                        default_k=default_k)


AFFERENT_PATHWAY = REGISTRY.topology(Topology(
    "afferent_pathway",
    "sensory pathways from a receptor surface through named relays to cortex.  it is the one "
    "topology that cannot be built from geometry at all, because its two ends are in "
    "different coordinate frames: the euclidean distance from a retinal position to its LGN "
    "target is a fact about two unregistered frames and says nothing about the optic nerve.  "
    "what joins them is a topographic map -- same visual field position, same characteristic "
    "frequency, same body-surface coordinate -- so the metric is receptotopic and the "
    "physical path length of each stage is separate anatomy rather than a distance between "
    "sites.  delay matters here more than anywhere else in the model: a foot afferent "
    "travels a metre before anything central happens, and the auditory brainstem's "
    "millisecond precision comes from stages that are short and fast.  these are the largest "
    "latencies in the whole graph and they belong to the pathway, not to any process over it",
    on=("retina", "cochlea", "body_surface", "vestibular_organ", "chemosensory_epithelium",
        "viscera", "body", "tissue"),
    edge_features=("tract_length_mm", "conduction_delay_s"),
    directed=True,
    builder="afferent_relay",
    provenance=Provenance.LITERATURE))
