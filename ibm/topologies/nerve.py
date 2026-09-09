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
    # -- the posterior primary rami ----------------------------------------
    # every trunk above is a VENTRAL ramus.  the omission was not cosmetic: it
    # left the paravertebral skin strip and the whole deep back with no declared
    # trunk, so a dermatomal partition of the integument had either to leave the
    # back uninnervated or to route it through `intercostal`, which would assert
    # that the back of the trunk is supplied by the nerve that supplies the
    # front.  one declaration covering the series, not 31 separate trunks.
    "dorsal_ramus": MIXED,
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
    "dorsal_ramus": 80.0,
}

#: THE PROXIMAL ENDPOINT OF EVERY TRUNK.  spinal levels for spinal nerves, and
#: the cranial numeral for cranial nerves -- the same two vocabularies
#: `ibm.anatomy.muscles.INNERVATION` already uses for root levels, so a trunk's
#: roots and a muscle's roots are comparable strings by construction.
#:
#: this table is what made the 71 trunks a routing convention rather than
#: anatomy: a trunk had a length and a fibre composition but no ends, so there
#: was no answer to "where does this cable start and what does it reach".  it is
#: declared here and then CHECKED against two independent tables that were built
#: for other reasons -- the union of root levels over the muscles each trunk
#: supplies, and the dermatome-to-trunk map IHM writes into `dermatomes.json`.
#: `tests/test_nerve_endpoints.py` runs both checks; a trunk whose declared roots
#: do not contain the roots of the muscles it supplies is a contradiction inside
#: this repo, not a matter of opinion.
CRANIAL = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii")
SPINAL_LEVELS = tuple(f"c{i}" for i in range(1, 9)) + \
    tuple(f"t{i}" for i in range(1, 13)) + tuple(f"l{i}" for i in range(1, 6)) + \
    tuple(f"s{i}" for i in range(1, 6)) + ("co1",)


def _span(lo: str, hi: str) -> tuple[str, ...]:
    """inclusive run of spinal levels, e.g. _span('t5', 't9')."""
    i, j = SPINAL_LEVELS.index(lo), SPINAL_LEVELS.index(hi)
    return SPINAL_LEVELS[i:j + 1]


TRUNK_ROOTS: dict[str, tuple[str, ...]] = {
    # -- cranial: a nucleus, not a segment ----------------------------------
    "olfactory": ("i",), "optic": ("ii",), "oculomotor": ("iii",),
    "trochlear": ("iv",), "trigeminal": ("v",), "abducens": ("vi",),
    "facial": ("vii",), "cochlear": ("viii",), "vestibular": ("viii",),
    "glossopharyngeal": ("ix",), "vagus": ("x",), "hypoglossal": ("xii",),
    # the accessory nerve is the exception: a cranial nerve with spinal roots
    "accessory": ("xi",) + _span("c1", "c5"),
    # -- plexuses ------------------------------------------------------------
    "cervical_plexus": _span("c1", "c4"),
    "brachial_plexus": _span("c5", "t1"),
    "lumbar_plexus": _span("l1", "l4"),
    "sacral_plexus": _span("l4", "s4"),
    # -- upper limb ----------------------------------------------------------
    "median": _span("c5", "t1"),
    # the ulnar nerve is C8-T1 with a C7 contribution that is present often
    # enough that flexor carpi ulnaris is conventionally given C7-T1
    "ulnar": _span("c7", "t1"),
    "radial": _span("c5", "t1"), "musculocutaneous": _span("c5", "c7"),
    "axillary": ("c5", "c6"),
    "anterior_interosseous": ("c8", "t1"), "posterior_interosseous": _span("c6", "c8"),
    "superficial_radial": ("c6", "c7"),
    "palmar_digital": _span("c6", "c8"), "dorsal_digital": _span("c6", "c8"),
    "medial_cutaneous_arm": ("c8", "t1"), "medial_cutaneous_forearm": ("c8", "t1"),
    "dorsal_scapular": ("c4", "c5"), "long_thoracic": _span("c5", "c7"),
    "suprascapular": ("c5", "c6"), "thoracodorsal": _span("c6", "c8"),
    "lateral_pectoral": _span("c5", "c7"), "medial_pectoral": ("c8", "t1"),
    "upper_subscapular": ("c5", "c6"), "lower_subscapular": ("c5", "c6"),
    # -- neck and trunk ------------------------------------------------------
    "phrenic": _span("c3", "c5"), "ansa_cervicalis": _span("c1", "c3"),
    "lesser_occipital": ("c2",), "great_auricular": ("c2", "c3"),
    "transverse_cervical": ("c2", "c3"), "supraclavicular": ("c3", "c4"),
    "intercostal": _span("t1", "t11"), "subcostal": ("t12",),
    "thoracoabdominal": _span("t7", "t12"),
    "iliohypogastric": ("l1",), "ilioinguinal": ("l1",),
    "genitofemoral": ("l1", "l2"),
    # every spinal nerve has one, so the series is the whole spinal column
    "dorsal_ramus": _span("c1", "s5"),
    # -- lower limb ----------------------------------------------------------
    "femoral": _span("l2", "l4"), "obturator": _span("l2", "l4"),
    "saphenous": ("l3", "l4"), "lateral_femoral_cutaneous": ("l2", "l3"),
    "posterior_femoral_cutaneous": _span("s1", "s3"),
    "sciatic": _span("l4", "s3"), "tibial": _span("l4", "s3"),
    "common_fibular": _span("l4", "s2"), "deep_fibular": _span("l4", "s1"),
    "superficial_fibular": _span("l4", "s1"), "sural": ("s1", "s2"),
    "medial_plantar": ("s1", "s2"), "lateral_plantar": _span("s1", "s3"),
    "superior_gluteal": _span("l4", "s1"), "inferior_gluteal": _span("l5", "s2"),
    "pudendal": _span("s2", "s4"),
    # -- autonomic outflow ---------------------------------------------------
    "sympathetic_chain": _span("t1", "l2"),
    "greater_splanchnic": _span("t5", "t9"), "lesser_splanchnic": ("t10", "t11"),
    "least_splanchnic": ("t12",), "lumbar_splanchnic": ("l1", "l2"),
    "pelvic_splanchnic": _span("s2", "s4"),
}

#: THE DISTAL ENDPOINT.  what the trunk actually reaches, in words, so that a
#: lesion has a describable cost.  a trunk with roots and no target is still a
#: cable running from a segment into nothing.
TRUNK_TARGET: dict[str, str] = {
    "olfactory": "olfactory bulb, from the olfactory epithelium",
    "optic": "lateral geniculate nucleus, from the retinal ganglion cells",
    "oculomotor": "medial, superior and inferior recti, inferior oblique, levator "
                  "palpebrae; ciliary ganglion",
    "trochlear": "superior oblique", "abducens": "lateral rectus",
    "trigeminal": "facial skin (V1-V3), muscles of mastication",
    "facial": "muscles of facial expression, stylohyoid, platysma",
    "cochlear": "cochlear nuclei, from the spiral ganglion",
    "vestibular": "vestibular nuclei, from the vestibular ganglion",
    "glossopharyngeal": "posterior tongue, carotid body and sinus, stylopharyngeus",
    "vagus": "thoracic and abdominal viscera to the left colic flexure; larynx",
    "accessory": "sternocleidomastoid and trapezius",
    "hypoglossal": "intrinsic and extrinsic tongue muscles",
    "cervical_plexus": "neck skin and infrahyoid muscles; the diaphragm via phrenic",
    "brachial_plexus": "the whole upper limb",
    "lumbar_plexus": "anterior and medial thigh, lower abdominal wall",
    "sacral_plexus": "posterior thigh, the whole leg and foot, pelvic floor",
    "median": "forearm flexors, thenar eminence, radial two lumbricals; skin of the "
              "radial three and a half digits",
    "ulnar": "flexor carpi ulnaris, ulnar half of FDP, the intrinsic hand; skin of "
             "the ulnar one and a half digits",
    "radial": "triceps and the whole extensor compartment; dorsoradial hand skin",
    "musculocutaneous": "biceps, brachialis, coracobrachialis; lateral forearm skin",
    "axillary": "deltoid and teres minor; skin over the lateral shoulder",
    "anterior_interosseous": "FPL, radial half of FDP, pronator quadratus; no skin",
    "posterior_interosseous": "the deep extensor compartment; no skin",
    "superficial_radial": "dorsoradial hand and thumb skin; no muscle",
    "palmar_digital": "palmar skin of the digits", "dorsal_digital": "dorsal digital skin",
    "medial_cutaneous_arm": "medial arm skin", "medial_cutaneous_forearm": "medial forearm skin",
    "dorsal_scapular": "rhomboids and levator scapulae", "long_thoracic": "serratus anterior",
    "suprascapular": "supraspinatus and infraspinatus", "thoracodorsal": "latissimus dorsi",
    "lateral_pectoral": "pectoralis major", "medial_pectoral": "pectoralis minor and major",
    "upper_subscapular": "subscapularis", "lower_subscapular": "subscapularis and teres major",
    "phrenic": "the diaphragm", "ansa_cervicalis": "the infrahyoid strap muscles",
    "lesser_occipital": "posterolateral scalp skin", "great_auricular": "skin over the "
                        "parotid and auricle",
    "transverse_cervical": "anterior neck skin", "supraclavicular": "skin over the "
                           "clavicle and upper pectoral region",
    "intercostal": "intercostal and abdominal wall muscles; thoracoabdominal skin",
    "subcostal": "the abdominal wall below the twelfth rib",
    "thoracoabdominal": "rectus abdominis and the flat abdominal muscles",
    "iliohypogastric": "suprapubic skin and the lower transversus abdominis",
    "ilioinguinal": "inguinal and anterior scrotal or labial skin",
    "genitofemoral": "cremaster; skin of the femoral triangle and genitalia",
    "dorsal_ramus": "erector spinae and the paravertebral skin strip, at every level",
    "femoral": "quadriceps, sartorius and iliacus; anterior thigh skin",
    "obturator": "the adductor compartment; medial thigh skin",
    "saphenous": "medial leg and medial foot skin; no muscle",
    "lateral_femoral_cutaneous": "lateral thigh skin; no muscle",
    "posterior_femoral_cutaneous": "posterior thigh and lower buttock skin",
    "sciatic": "the hamstrings and everything below the knee",
    "tibial": "the posterior leg compartment and the plantar foot",
    "common_fibular": "short head of biceps femoris; the anterior and lateral leg",
    "deep_fibular": "the anterior leg compartment; first web space skin",
    "superficial_fibular": "fibularis longus and brevis; dorsal foot skin",
    "sural": "lateral foot and posterior calf skin; no muscle",
    "medial_plantar": "abductor hallucis, FDB, first lumbrical; medial sole skin",
    "lateral_plantar": "the remaining intrinsic foot muscles; lateral sole skin",
    "superior_gluteal": "gluteus medius and minimus, tensor fasciae latae",
    "inferior_gluteal": "gluteus maximus",
    "pudendal": "the pelvic floor and external sphincters; perineal skin",
    "sympathetic_chain": "the paravertebral ganglia, and through them the body wall "
                         "vasculature and sweat glands",
    "greater_splanchnic": "coeliac ganglion, and the foregut",
    "lesser_splanchnic": "aorticorenal ganglion, and the midgut",
    "least_splanchnic": "renal plexus",
    "lumbar_splanchnic": "inferior mesenteric ganglion, and the hindgut",
    "pelvic_splanchnic": "the pelvic viscera, parasympathetic",
}


#: `ibm.anatomy.muscles.INNERVATION` names nine nerves that are BRANCHES of a
#: declared trunk rather than trunks: `facial_vii_somatic_motor` is the somatic
#: motor part of `facial`, and `vagus_x_recurrent_laryngeal` and
#: `vagus_x_pharyngeal` are two branches of the one vagus.  the two tables were
#: written at different times with different naming conventions, so nine muscle
#: entries named a nerve with no trunk, no composition and no conduction delay --
#: not because the nerve was missing but because the join key was.
BRANCH_TRUNK: dict[str, str] = {
    "abducens_vi": "abducens", "accessory_xi": "accessory",
    "facial_vii_somatic_motor": "facial", "hypoglossal_xii": "hypoglossal",
    "oculomotor_iii": "oculomotor", "trigeminal_v3_mandibular": "trigeminal",
    "trochlear_iv": "trochlear", "vagus_x_pharyngeal": "vagus",
    "vagus_x_recurrent_laryngeal": "vagus",
}


def trunk_of(nerve_name: str) -> str:
    """resolve a nerve or branch name to the trunk that carries it."""
    return BRANCH_TRUNK.get(nerve_name, nerve_name)


def trunk_endpoints(trunk: str) -> dict:
    """the two ends of one cable, plus the length that separates them.

    `length_mm` here is the TYPED trunk length and is superseded: for a
    conduction delay use `ibm.topologies.ihm_bridge.routes`, which reads the
    route IHM measured over the body mesh.  the field is kept so that the two can
    be compared, which is the point of `ihm_bridge.length_agreement`.
    """
    if trunk not in TRUNK_COMPOSITION:
        raise KeyError(f"no trunk {trunk!r}")
    return {
        "trunk": trunk,
        "roots": TRUNK_ROOTS.get(trunk, ()),
        "root_kind": "cranial_nucleus" if any(r in CRANIAL for r in
                                              TRUNK_ROOTS.get(trunk, ()))
                     else "spinal_segments",
        "target": TRUNK_TARGET.get(trunk),
        "classes": TRUNK_COMPOSITION[trunk],
        "length_mm": TRUNK_LENGTH_MM.get(trunk),
        "length_evidence": LENGTH_EVIDENCE,
        "length_superseded_by": LENGTH_SUPERSEDED_BY,
    }


#: the length used when a trunk has neither a measured route nor a typed entry.
#: it was an inline literal in three places with three different values -- 300 mm
#: here, 200 mm in `ibm.embodiment`, 50 mm in `scripts/pretrain_video_loop` --
#: which meant the same unnamed trunk got three different conduction delays
#: depending on which module asked.  naming it does not make it right; it makes
#: it findable, and `trunk_length_mm` reports when it was used.
TRUNK_LENGTH_DEFAULT_MM = 300.0


def trunk_length_mm(trunk: str, *, prefer_measured: bool = True
                    ) -> tuple[float, str]:
    """the length to use for a delay, and WHERE IT CAME FROM.

    the order is deliberate.  IHM's measured route over the body mesh first,
    because a conduction delay is a property of the route from receptor to relay
    and not of the named trunk -- the two disagree by 4.4x on the plantar nerves.
    the typed table second.  the bare default last, and it is never silent: the
    caller gets the string `default_no_declared_length` back and can refuse it.

    21 of the 72 declared trunks have no entry in `TRUNK_LENGTH_MM` at all, so
    this is not a corner case.
    """
    if prefer_measured:
        try:
            from ibm.topologies.ihm_bridge import measured_trunk_lengths_mm
            measured = measured_trunk_lengths_mm()
        except Exception:                       # IHM absent: fall through
            measured = {}
        if trunk in measured:
            return measured[trunk], "ihm_measured_route"
    if trunk in TRUNK_LENGTH_MM:
        return TRUNK_LENGTH_MM[trunk], "ibm_typed_trunk"
    return TRUNK_LENGTH_DEFAULT_MM, "default_no_declared_length"


def fibre_delays_s(trunk: str, length_mm: float | None = None,
                   classes: tuple[str, ...] | None = None,
                   prefer_measured: bool = True) -> dict[str, float]:
    """conduction delay per fibre class along one trunk, seconds.

    the whole point of the module in four lines: one trunk, one length, and a
    delay per class that differs by two orders of magnitude across the classes
    that share it.

    the length defaults to the MEASURED route where IHM has one, not to the
    typed trunk table.  the module docstring above has always said the route is
    the right quantity for a delay; until now the function did not read it.
    """
    if length_mm is None:
        length, _ = trunk_length_mm(trunk, prefer_measured=prefer_measured)
    else:
        length = length_mm
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
