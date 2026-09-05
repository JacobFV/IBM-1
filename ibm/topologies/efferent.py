"""efferent pathways: cortex and brainstem out to motor units.

the mirror of the afferent topology and, deliberately, the same machinery -- the
relay algebra is identical, only the direction of travel differs, so it is
imported rather than reimplemented.  two implementations of "a topographically
mapped relay stage with a length and a velocity" would drift apart, and the
difference between them would be invisible.

what makes it a *separate topology* rather than the afferent one reversed is that
the maps are different maps and the fibre classes are different classes.  the
corticospinal projection from area 4 to spinal motoneurons is somatotopic in the
same body coordinate the ascending pathway uses, but it is not the inverse of any
ascending stage: it skips the thalamus entirely, it terminates on motoneurons
rather than on relay cells, and in human a substantial fraction of it is
monosynaptic to the motoneuron -- which is the anatomical fact that makes the
descending path fast where the ascending one is slow.  running one topology
backwards would either give the descending path the ascending path's stages or
give it the ascending path's delays, and both are wrong by tens of milliseconds.

the final stage is the one that dominates the timing, and its length is not a
brain measurement.  a corticospinal axon to a lumbar motoneuron runs something
like 450 mm before the peripheral nerve even starts, and the peripheral axon to an
intrinsic foot muscle adds most of another metre.  those lengths scale with the
person, which is why they are stage parameters rather than constants of the
ontology: height is the single largest source of between-subject variance in
motor evoked potential latency, and a model that fixes them makes a systematic
error correlated with a subject variable.

the topology is directed for the obvious reason and a less obvious one.  the
obvious one is that motor commands go one way.  the less obvious one is that
efferent copy -- the collateral of a descending command that reaches sensory
structures -- is a *separate stage in this same topology* rather than an afferent
edge, because it carries the descending signal and inherits the descending delay,
and a model that routed it through the afferent pathway would give it the wrong
latency and the wrong source.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.topologies.afferent import staged_edges
from ibm.vocabulary import Provenance


#: literature chains at one significant figure, as in the afferent table.  path
#: lengths are adult anatomy and scale with height; the velocities are fibre-class
#: typicals.  the corticospinal number is the one most worth overriding per subject.
HUMAN_EFFERENT_STAGES: dict[str, tuple[dict, ...]] = {
    "corticospinal": (
        dict(name="m1->motoneuron", **{"from": "tissue"}, to="body", map="somatotopy",
             select_from=("cytoarchitecture", ("ba_4", "ba_6")),
             length_mm=450.0, velocity_m_s=60.0, k=1),
        dict(name="motoneuron->motor_unit", **{"from": "body"}, to="motor_units",
             map="somatotopy", length_mm=600.0, velocity_m_s=55.0, k=1),
    ),
    "corticobulbar": (
        dict(name="m1->cranial_motor_nuclei", **{"from": "tissue"}, to="tissue",
             map="somatotopy", select_from=("cytoarchitecture", ("ba_4", "ba_6")),
             select_to=("brainstem_nuclei", ("facial_nucleus", "hypoglossal_nucleus",
                                             "trigeminal_motor_nucleus", "nucleus_ambiguus")),
             length_mm=60.0, velocity_m_s=60.0, k=1),
        dict(name="cranial_motor_nuclei->vocal_tract", **{"from": "tissue"},
             to="vocal_tract", map="somatotopy",
             select_from=("brainstem_nuclei", ("facial_nucleus", "hypoglossal_nucleus",
                                               "trigeminal_motor_nucleus",
                                               "nucleus_ambiguus")),
             length_mm=150.0, velocity_m_s=50.0, k=1),
    ),
    "oculomotor": (
        dict(name="sc->oculomotor_nuclei", **{"from": "tissue"}, to="tissue",
             map="visual_field",
             select_from=("brainstem_nuclei", ("superior_colliculus",)),
             select_to=("brainstem_nuclei", ("oculomotor_nucleus", "trochlear_nucleus",
                                             "abducens_nucleus")),
             length_mm=15.0, velocity_m_s=30.0, k=1),
        dict(name="oculomotor_nuclei->extraocular_units", **{"from": "tissue"},
             to="motor_units", map=None,
             select_from=("brainstem_nuclei", ("oculomotor_nucleus", "trochlear_nucleus",
                                               "abducens_nucleus")),
             length_mm=50.0, velocity_m_s=50.0, k=1),
    ),
    "autonomic": (
        dict(name="hypothalamus->preganglionic", **{"from": "tissue"}, to="tissue", map=None,
             select_from=("hypothalamic_nuclei", ("paraventricular",)),
             select_to=("brainstem_nuclei", ("dorsal_motor_vagus", "nucleus_ambiguus")),
             length_mm=40.0, velocity_m_s=2.0, k=1),
        dict(name="preganglionic->viscera", **{"from": "tissue"}, to="viscera", map=None,
             select_from=("brainstem_nuclei", ("dorsal_motor_vagus", "nucleus_ambiguus")),
             length_mm=350.0, velocity_m_s=3.0, k=1),
    ),
}


@B.builder(
    "efferent_relay",
    produces=("tract_length_mm", "conduction_delay_s"),
    supports=("tissue", "body", "motor_units", "vocal_tract", "viscera"),
    requires=("stages",),
    directed=True,
    metric="somatotopic coordinate distance; path length is per-stage anatomy",
    doc="relay-staged motor and autonomic pathways, matched by somatotopic coordinate")
def efferent_relay(sites, *, stages=None, pathway: str | None = None, default_k: int = 1):
    """build one or more descending chains.

    the fan-out convention is worth naming because it is the opposite of the
    afferent one.  `k` here is still the number of source sites converging on each
    target, which for a descending pathway means the number of cortical sites
    driving one motoneuron pool -- and the anatomy is genuinely convergent in that
    direction: many corticospinal neurons across a swathe of motor cortex reach one
    motoneuron pool, which is why a discrete cortical map of individual muscles
    does not exist.
    """
    if stages is None and pathway is not None:
        if pathway not in HUMAN_EFFERENT_STAGES:
            raise KeyError(f"no efferent chain {pathway!r}; "
                           f"known: {', '.join(sorted(HUMAN_EFFERENT_STAGES))}")
        stages = HUMAN_EFFERENT_STAGES[pathway]
    return staged_edges("efferent_pathway", "efferent_relay", sites, stages or (),
                        default_k=default_k)


EFFERENT_PATHWAY = REGISTRY.topology(Topology(
    "efferent_pathway",
    "descending pathways from cortex and brainstem to motor units, the vocal tract and the "
    "viscera.  it shares the afferent topology's relay machinery and is not the afferent "
    "topology reversed: the corticospinal projection skips the thalamus, terminates partly "
    "monosynaptically on motoneurons, and is fast where the ascending path is slow, so "
    "running the sensory chain backwards would give it the wrong stages and the wrong "
    "delays.  its dominant lengths are body lengths rather than brain lengths -- roughly "
    "450 mm of corticospinal axon before the peripheral nerve begins -- and they scale with "
    "height, which is why they are per-stage parameters rather than constants and why height "
    "is the largest between-subject term in motor evoked potential latency.  efference copy "
    "is a stage of this topology rather than an afferent edge, so that it carries the "
    "descending signal and the descending delay",
    on=("tissue", "body", "motor_units", "vocal_tract", "viscera"),
    edge_features=("tract_length_mm", "conduction_delay_s"),
    directed=True,
    builder="efferent_relay",
    provenance=Provenance.LITERATURE))
