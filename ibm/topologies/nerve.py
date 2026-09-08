"""peripheral nerve trunks, and the fibre classes that share them.

the topology `afferent_pathway` answers "which relay does this receptor project
to"; this one answers a question it cannot: **which trunk does the axon run in,
and how many different speeds are in that trunk at once.**

the distinction is not bookkeeping.  a peripheral nerve is not a wire, it is a
cable carrying six or more fibre populations whose conduction velocities span
more than two orders of magnitude -- a group Ia axon runs at 80-120 m/s and an
unmyelinated C fibre in the same trunk at 0.5-2 m/s.  over a one-metre leg that
is 10 ms against 1000 ms.  a topology that gives a nerve one `conduction_delay_s`
has asserted that those arrive together, and every reflex latency, every
first-pain/second-pain separation and every alpha-gamma timing relationship in
the model is then wrong by whatever the lumping chose.

so the edge feature here is a delay *per fibre class*, and the class axis is the
component vector on `ibm.fields.neural` -- `afferent.{ia,ib,ii,abeta,adelta,c}`
and `efferent.{alpha,gamma,b_preganglionic,c_postganglionic}`.  the trunk is the
partition, the classes are the components, and a materialized nerve is the
product.  `ibm.anatomy.systems` states the rule this follows: things that tile
are partitions, things that coexist are components, and fibre classes coexist in
every millimetre of every trunk.

### what a trunk's composition actually says

the composition table below is the fibre content of each trunk, and it carries
real modelling content rather than being a completeness exercise.  a purely
cutaneous nerve (sural, superficial fibular) has no Ia and no alpha, so a lesion
of it costs sensation and no strength.  a purely muscular nerve (deep fibular's
motor portion, anterior interosseous) has no A-beta.  the vagus is ~80% afferent
despite being described as a motor nerve, and that is invisible unless afferent
and efferent are separate axes on the same trunk.  **mixed nerves are the norm
and the mixture is the anatomy.**

### what this topology deliberately does not do

it does not carry the segmental reflex arc.  an edge here is a conduction path,
not a synapse, and the cord circuitry that closes the stretch reflex --
Ia -> alpha monosynaptically, Ib -> inhibitory interneuron, Renshaw recurrent
inhibition -- is a *process* over `spinal_levels`, which does not yet exist
(EMBODIMENT.md §2.2).  this topology is what such a process would be defined
over, and declaring it is the prerequisite rather than the thing itself.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.topologies.afferent import staged_edges
from ibm.vocabulary import Provenance

#: conduction velocity in m/s by fibre class, as (low, typical, high).  the class
#: names are the component suffixes on `ibm.fields.neural`, so a table lookup here
#: and a component id are the same string by construction.
FIBRE_VELOCITY_M_S: dict[str, tuple[float, float, float]] = {
    # -- special sense: cranial afferents, which are not somatic fibres --------
    # the somatic classes above are named by axon diameter in a peripheral nerve.
    # the optic nerve is not a peripheral nerve at all -- it is a CNS tract,
    # myelinated by oligodendrocytes, and its axons are retinal ganglion cells
    # whose classes differ by receptive field and contrast sensitivity rather
    # than by the A-alpha/A-beta scheme.  forcing them into `abeta` would assert
    # a conduction velocity that is not measured and a class that does not exist.
    "retinal_magno":    (15.0,  20.0,  25.0),   # parasol RGC, fast, achromatic
    "retinal_parvo":    ( 8.0,  12.0,  16.0),   # midget RGC, slow, chromatic
    "retinal_konio":    ( 3.0,   6.0,  10.0),   # bistratified, slowest
    "cochlear_type1":   (15.0,  25.0,  35.0),   # spiral ganglion I, 95% of fibres
    "cochlear_type2":   ( 1.0,   3.0,   6.0),   # unmyelinated, outer hair cells
    "vestibular":       (15.0,  25.0,  35.0),
    "olfactory":        ( 0.2,   0.5,   1.0),   # unmyelinated, the slowest tract
    "ia":               (80.0, 100.0, 120.0),   # A-alpha, 12-20 um
    "ib":               (80.0, 100.0, 120.0),   # A-alpha, 12-20 um
    "ii":               (35.0,  55.0,  75.0),   # A-beta,   6-12 um
    "abeta":            (35.0,  55.0,  75.0),   # A-beta,   6-12 um
    "adelta":           ( 5.0,  15.0,  30.0),   # A-delta,  1-5 um
    "c":                ( 0.5,   1.0,   2.0),   # unmyelinated, 0.2-1.5 um
    "alpha":            (80.0, 100.0, 120.0),   # A-alpha motor
    "gamma":            (10.0,  25.0,  45.0),   # A-gamma fusimotor, 2-8 um
    "b_preganglionic":  ( 3.0,   8.0,  15.0),   # B, thinly myelinated
    "c_postganglionic": ( 0.5,   1.0,   2.0),   # unmyelinated
}

#: which classes each named trunk carries.  absence is a claim: a trunk with no
#: "ia" cannot participate in a stretch reflex, and a trunk with no "abeta" costs
#: no touch when it is cut.
SENSORY = ("ia", "ib", "ii", "abeta", "adelta", "c")
CUTANEOUS = ("abeta", "adelta", "c")
MUSCULAR = ("ia", "ib", "ii", "alpha", "gamma")
MIXED = SENSORY + ("alpha", "gamma")
AUTONOMIC = ("b_preganglionic", "c_postganglionic", "c")

TRUNK_COMPOSITION: dict[str, tuple[str, ...]] = {
    # -- cranial nerves ------------------------------------------------------
    # absent until now, which meant the model had no declared route for vision,
    # hearing, balance or visceral afference -- the training loops drove raw
    # index slices named "occipital" and "temporal" by comment only.
    "optic": ("retinal_magno", "retinal_parvo", "retinal_konio"),
    "cochlear": ("cochlear_type1", "cochlear_type2"),
    "vestibular": ("vestibular",),
    "olfactory": ("olfactory",),
    # the vagus is ~80% AFFERENT despite being called a motor nerve, which the
    # module docstring already flags as the case separate axes exist to express.
    # it is the main gut-to-brain route and the one an embodied model needs.
    "vagus": ("abeta", "adelta", "c", "b_preganglionic", "c_postganglionic"),
    "glossopharyngeal": ("abeta", "adelta", "c", "b_preganglionic"),
    "trigeminal": ("abeta", "adelta", "c", "alpha"),
    "facial": ("alpha", "abeta", "b_preganglionic"),
    "hypoglossal": ("alpha",),
    "accessory": ("alpha",),
    "oculomotor": ("alpha", "b_preganglionic"),
    "trochlear": ("alpha",),
    "abducens": ("alpha",),
    # -- mixed limb nerves: the common case ---------------------------------
    "median": MIXED, "ulnar": MIXED, "radial": MIXED, "musculocutaneous": MIXED,
    "axillary": MIXED, "femoral": MIXED, "obturator": MIXED, "sciatic": MIXED,
    "tibial": MIXED, "common_fibular": MIXED, "deep_fibular": MIXED,
    "medial_plantar": MIXED, "lateral_plantar": MIXED, "pudendal": MIXED + AUTONOMIC,
    "intercostal": MIXED, "subcostal": MIXED, "thoracoabdominal": MIXED,
    "iliohypogastric": MIXED, "ilioinguinal": MIXED, "genitofemoral": MIXED,
    # -- purely muscular: no A-beta, so a lesion costs strength and not touch
    "anterior_interosseous": MUSCULAR, "posterior_interosseous": MUSCULAR,
    "dorsal_scapular": MUSCULAR, "long_thoracic": MUSCULAR,
    "suprascapular": MUSCULAR, "thoracodorsal": MUSCULAR,
    "lateral_pectoral": MUSCULAR, "medial_pectoral": MUSCULAR,
    "upper_subscapular": MUSCULAR, "lower_subscapular": MUSCULAR,
    "superior_gluteal": MUSCULAR, "inferior_gluteal": MUSCULAR,
    "phrenic": MUSCULAR, "ansa_cervicalis": MUSCULAR,
    # -- purely cutaneous: no Ia and no alpha, so a lesion costs no strength
    "sural": CUTANEOUS, "saphenous": CUTANEOUS, "superficial_fibular": CUTANEOUS,
    "superficial_radial": CUTANEOUS, "medial_cutaneous_arm": CUTANEOUS,
    "medial_cutaneous_forearm": CUTANEOUS, "lateral_femoral_cutaneous": CUTANEOUS,
    "posterior_femoral_cutaneous": CUTANEOUS, "palmar_digital": CUTANEOUS,
    "dorsal_digital": CUTANEOUS, "lesser_occipital": CUTANEOUS,
    "great_auricular": CUTANEOUS, "transverse_cervical": CUTANEOUS,
    "supraclavicular": CUTANEOUS,
    # -- autonomic trunks ---------------------------------------------------
    "sympathetic_chain": AUTONOMIC, "greater_splanchnic": AUTONOMIC,
    "lesser_splanchnic": AUTONOMIC, "least_splanchnic": AUTONOMIC,
    "lumbar_splanchnic": AUTONOMIC, "pelvic_splanchnic": AUTONOMIC,
    # -- plexuses carry everything their branches do ------------------------
    "cervical_plexus": MIXED, "brachial_plexus": MIXED,
    "lumbar_plexus": MIXED, "sacral_plexus": MIXED,
}

#: typical trunk lengths, mm, one significant figure and stated as such for the
#: same reason `ibm.topologies.afferent` states its stage table that way: adult
#: path lengths vary by a factor of two with body size, and anything that depends
#: sharply on one of these should fit it.
# provenance, adopted from IHM-1's schema.  every one of their records carries
# `evidence_kind`, `geometry_kind` and `measured_axon_geometry: false`, plus an
# explicit `limitations` list saying the centrelines are inferred rather than
# dissected.  the table below was bare floats with no provenance at all -- a
# reader could not tell which came from a source and which I typed from memory,
# and this repo's ledger is 17 rows of numbers that turned out to be compared
# against the wrong thing.
#
# what these lengths ARE: the anatomical length of the named trunk, typed from
# standard references.  what they are NOT: the conduction route from receptor to
# relay, which is longer by the root and cord segment.  IHM-1 measures the route
# over a real body mesh and the two disagree systematically where that segment
# dominates -- lateral_plantar 888 mm measured against 200 typed here (4.4x),
# oculomotor 142 against 45 (3.2x) -- while agreeing within ~20% on limb trunks
# where it does not (median 587 against 700, radial 526 against 650).
#
# FOR A CONDUCTION DELAY THE ROUTE IS THE RIGHT QUANTITY.  prefer
# `ibm.topologies.ihm_bridge.routes()`, which uses IHM's measured length where
# one exists and falls back here with `length_source` saying which was used.
LENGTH_EVIDENCE = "typed_from_reference_anatomy; trunk length, not conduction route"
LENGTH_SUPERSEDED_BY = "ibm.topologies.ihm_bridge.routes"

TRUNK_LENGTH_MM: dict[str, float] = {
    # cranial: tens of millimetres, not hundreds.  the optic nerve is 40-50 mm
    # from globe to chiasm; the whole retinogeniculate path is under 80 mm.
    "optic": 50.0, "cochlear": 25.0, "vestibular": 25.0, "olfactory": 10.0,
    "vagus": 350.0, "glossopharyngeal": 60.0, "trigeminal": 50.0,
    "facial": 60.0, "hypoglossal": 50.0, "accessory": 120.0,
    "oculomotor": 45.0, "trochlear": 60.0, "abducens": 55.0,
    "median": 700.0, "ulnar": 700.0, "radial": 650.0, "musculocutaneous": 300.0,
    "axillary": 150.0, "superficial_radial": 400.0,
    "anterior_interosseous": 350.0, "posterior_interosseous": 300.0,
    "palmar_digital": 120.0, "dorsal_digital": 120.0,
    "medial_cutaneous_arm": 250.0, "medial_cutaneous_forearm": 350.0,
    "femoral": 400.0, "obturator": 250.0, "saphenous": 600.0,
    "lateral_femoral_cutaneous": 300.0, "posterior_femoral_cutaneous": 400.0,
    "sciatic": 550.0, "tibial": 500.0, "common_fibular": 350.0,
    "superficial_fibular": 300.0, "deep_fibular": 350.0, "sural": 400.0,
    "medial_plantar": 200.0, "lateral_plantar": 200.0,
    "phrenic": 350.0, "intercostal": 250.0, "subcostal": 250.0,
    "thoracoabdominal": 250.0, "pudendal": 200.0,
    "superior_gluteal": 100.0, "inferior_gluteal": 100.0,
    "long_thoracic": 250.0, "suprascapular": 120.0, "thoracodorsal": 200.0,
    "dorsal_scapular": 150.0, "sympathetic_chain": 450.0,
}


def fibre_delays_s(trunk: str, length_mm: float | None = None,
                   classes: tuple[str, ...] | None = None) -> dict[str, float]:
    """conduction delay per fibre class along one trunk, seconds.

    the whole point of the module in four lines: one trunk, one length, and a
    delay per class that differs by two orders of magnitude across the classes
    that share it.
    """
    length = TRUNK_LENGTH_MM.get(trunk, 300.0) if length_mm is None else length_mm
    carried = classes if classes is not None else TRUNK_COMPOSITION.get(trunk, ())
    return {c: (length * 1e-3) / FIBRE_VELOCITY_M_S[c][1] for c in carried}


@B.builder(
    "peripheral_nerve_trunk",
    produces=("trunk_length_mm", "conduction_delay_s", "fibre_classes"),
    supports=("body_surface", "motor_units", "viscera", "body", "tissue"),
    requires=("stages",),
    directed=True,
    metric="path length along the trunk; delay is per fibre class, not per edge",
    doc="named peripheral nerve trunks carrying a fibre-class vector")
def peripheral_nerve_trunk(sites, *, stages=None, trunk: str | None = None,
                           default_k: int = 1):
    """build the edges of one named trunk, or an explicit stage chain.

    giving neither raises rather than returning an empty topology, for the reason
    `afferent_relay` gives: a silently empty peripheral topology is a body with no
    nerves and no error explaining why.
    """
    if stages is None and trunk is not None:
        if trunk not in TRUNK_COMPOSITION:
            raise KeyError(f"no trunk {trunk!r}; known: "
                           f"{', '.join(sorted(TRUNK_COMPOSITION))}")
        stages = (dict(name=trunk, **{"from": "body_surface"}, to="tissue",
                       length_mm=TRUNK_LENGTH_MM.get(trunk, 300.0),
                       velocity_m_s=FIBRE_VELOCITY_M_S["abeta"][1], k=default_k),)
    return staged_edges("peripheral_nerve", "peripheral_nerve_trunk", sites,
                        stages or (), default_k=default_k)


PERIPHERAL_NERVE = REGISTRY.topology(Topology(
    "peripheral_nerve",
    "named peripheral nerve trunks, each carrying several fibre classes at once.  it "
    "differs from `afferent_pathway` and `efferent_pathway` in what it is a statement "
    "about: those say which relay a signal reaches, this says which cable it travels in "
    "and how many different speeds are in that cable.  a trunk carrying group Ia at "
    "100 m/s and unmyelinated C at 1 m/s delivers them a second apart over a leg, so a "
    "single conduction_delay_s per edge is not an approximation but a category error -- "
    "it asserts that first and second pain arrive together, that a stretch reflex and a "
    "thermal percept share a latency, and that fusimotor drive reaches the spindle when "
    "alpha drive reaches the muscle.  the composition table is load-bearing too: a "
    "cutaneous trunk carries no Ia and no alpha, so cutting it costs sensation and no "
    "strength, and that asymmetry is anatomy the model can only express if presence and "
    "absence of a class are declared per trunk",
    on=("body_surface", "motor_units", "viscera", "body", "tissue"),
    edge_features=("trunk_length_mm", "conduction_delay_s", "fibre_classes"),
    directed=True,
    builder="peripheral_nerve_trunk",
    provenance=Provenance.LITERATURE))
