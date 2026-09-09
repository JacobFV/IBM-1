"""the named skeletal muscles, and the wiring that reaches them.

`myotomes` says which spinal segments supply a region; `peripheral_nerves` says
which trunks run where.  neither says **which nerve reaches which muscle**, and
that is the fact an embodiment needs: to connect a simulated biceps to this model
you need to know that its drive arrives on the musculocutaneous nerve from C5-C6,
because that determines its conduction delay, which motor pool it shares, and
what a lesion costs.

so this module is the innervation table.  it is the difference between "the model
has a motor system" and "the model has a socket labelled `biceps_brachii`".

three things are recorded per muscle and each is load-bearing:

*the nerve*, which fixes the conduction delay via `ibm.topologies.nerve` -- a
tibialis anterior command travels the deep fibular nerve and arrives later than a
biceps command on the musculocutaneous, and that difference is anatomy rather
than a parameter.

*the root levels*, which are what a segmental lesion acts on and what a reflex
arc closes through.  they are a tuple rather than a single level because nearly
every limb muscle draws from two or more segments -- which is exactly why root
lesions cause weakness and not paralysis.

*whether it has spindles worth modelling*.  a few muscles are effectively
spindle-free in a way that matters: the extraocular muscles are unusual and the
intrinsic laryngeals are sparse, so a controller that assumes uniform
proprioception across the body is wrong in specific, nameable places.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Anatomy
from ibm.vocabulary import Provenance

#: muscle -> (nerve, root levels, spindle_density)
#: spindle_density is relative, 1.0 being a typical limb muscle.  the extremes are
#: real: deep neck and intrinsic hand muscles are spindle-rich, and the large
#: proximal movers are comparatively poor.
INNERVATION: dict[str, tuple[str, tuple[str, ...], float]] = {
    # -- head and neck ------------------------------------------------------
    "masseter": ("trigeminal_v3_mandibular", ("v",), 1.6),
    "temporalis": ("trigeminal_v3_mandibular", ("v",), 1.6),
    "orbicularis_oculi": ("facial_vii_somatic_motor", ("vii",), 0.3),
    "orbicularis_oris": ("facial_vii_somatic_motor", ("vii",), 0.3),
    "zygomaticus_major": ("facial_vii_somatic_motor", ("vii",), 0.3),
    "genioglossus": ("hypoglossal_xii", ("xii",), 0.8),
    "hyoglossus": ("hypoglossal_xii", ("xii",), 0.8),
    "platysma": ("facial_vii_somatic_motor", ("vii",), 0.3),
    "stylohyoid": ("facial_vii_somatic_motor", ("vii",), 0.5),
    "sternocleidomastoid": ("accessory_xi", ("xi", "c2", "c3"), 1.2),
    "trapezius": ("accessory_xi", ("xi", "c3", "c4"), 0.8),
    "posterior_cricoarytenoid": ("vagus_x_recurrent_laryngeal", ("x",), 0.2),
    "thyroarytenoid": ("vagus_x_recurrent_laryngeal", ("x",), 0.2),
    "cricothyroid": ("vagus_x_pharyngeal", ("x",), 0.2),
    # -- extraocular: no conventional spindles, and it is a real exception ---
    "lateral_rectus": ("abducens_vi", ("vi",), 0.0),
    "superior_oblique": ("trochlear_iv", ("iv",), 0.0),
    "medial_rectus": ("oculomotor_iii", ("iii",), 0.0),
    "superior_rectus": ("oculomotor_iii", ("iii",), 0.0),
    "inferior_rectus": ("oculomotor_iii", ("iii",), 0.0),
    "inferior_oblique": ("oculomotor_iii", ("iii",), 0.0),
    "levator_palpebrae_superioris": ("oculomotor_iii", ("iii",), 0.0),
    # -- respiratory --------------------------------------------------------
    "diaphragm": ("phrenic", ("c3", "c4", "c5"), 0.4),
    "external_intercostal": ("intercostal", ("t1", "t11"), 1.0),
    "internal_intercostal": ("intercostal", ("t1", "t11"), 1.0),
    # -- shoulder girdle ----------------------------------------------------
    "rhomboid_major": ("dorsal_scapular", ("c4", "c5"), 0.9),
    # C3-C4 twigs from the cervical plexus also reach levator scapulae; only the
    # dorsal scapular supply is represented, and widening the trunk to C3 would
    # assert that the dorsal scapular nerve itself carries C3.
    "levator_scapulae": ("dorsal_scapular", ("c4", "c5"), 0.9),
    "serratus_anterior": ("long_thoracic", ("c5", "c6", "c7"), 0.8),
    "supraspinatus": ("suprascapular", ("c5", "c6"), 1.0),
    "infraspinatus": ("suprascapular", ("c5", "c6"), 1.0),
    "latissimus_dorsi": ("thoracodorsal", ("c6", "c7", "c8"), 0.7),
    "pectoralis_major": ("lateral_pectoral", ("c5", "c6", "c7"), 0.7),
    "pectoralis_minor": ("medial_pectoral", ("c8", "t1"), 0.7),
    "subscapularis": ("upper_subscapular", ("c5", "c6"), 1.0),
    "teres_major": ("lower_subscapular", ("c5", "c6"), 0.9),
    "deltoid": ("axillary", ("c5", "c6"), 0.8),
    "teres_minor": ("axillary", ("c5", "c6"), 1.0),
    # -- arm ----------------------------------------------------------------
    "biceps_brachii": ("musculocutaneous", ("c5", "c6"), 1.0),
    "brachialis": ("musculocutaneous", ("c5", "c6"), 1.0),
    "coracobrachialis": ("musculocutaneous", ("c5", "c6", "c7"), 1.0),
    "triceps_brachii": ("radial", ("c6", "c7", "c8"), 0.9),
    "brachioradialis": ("radial", ("c5", "c6"), 1.1),
    "anconeus": ("radial", ("c7", "c8"), 1.0),
    # -- forearm ------------------------------------------------------------
    "pronator_teres": ("median", ("c6", "c7"), 1.2),
    "flexor_carpi_radialis": ("median", ("c6", "c7"), 1.2),
    "palmaris_longus": ("median", ("c7", "c8"), 1.2),
    "flexor_digitorum_superficialis": ("median", ("c7", "c8", "t1"), 1.3),
    "flexor_digitorum_profundus_radial": ("anterior_interosseous", ("c8", "t1"), 1.3),
    "flexor_pollicis_longus": ("anterior_interosseous", ("c8", "t1"), 1.3),
    "pronator_quadratus": ("anterior_interosseous", ("c8", "t1"), 1.2),
    "flexor_carpi_ulnaris": ("ulnar", ("c7", "c8", "t1"), 1.2),
    "flexor_digitorum_profundus_ulnar": ("ulnar", ("c8", "t1"), 1.3),
    "extensor_carpi_radialis_longus": ("radial", ("c6", "c7"), 1.2),
    # ECRB is the exception among the wrist extensors: it is supplied by the deep
    # branch before it becomes the posterior interosseous, which is why a PIN
    # palsy spares wrist extension and drops the fingers.
    "extensor_carpi_radialis_brevis": ("radial", ("c7", "c8"), 1.2),
    "extensor_digitorum": ("posterior_interosseous", ("c7", "c8"), 1.2),
    "extensor_digiti_minimi": ("posterior_interosseous", ("c7", "c8"), 1.2),
    "extensor_indicis": ("posterior_interosseous", ("c7", "c8"), 1.2),
    "extensor_carpi_ulnaris": ("posterior_interosseous", ("c7", "c8"), 1.2),
    "supinator": ("posterior_interosseous", ("c6", "c7"), 1.2),
    "abductor_pollicis_longus": ("posterior_interosseous", ("c7", "c8"), 1.2),
    "extensor_pollicis_longus": ("posterior_interosseous", ("c7", "c8"), 1.2),
    "extensor_pollicis_brevis": ("posterior_interosseous", ("c7", "c8"), 1.2),
    # -- hand: spindle-rich, which is why manipulation is what it is --------
    "abductor_pollicis_brevis": ("median", ("c8", "t1"), 2.0),
    "opponens_pollicis": ("median", ("c8", "t1"), 2.0),
    "lumbricals_radial": ("median", ("c8", "t1"), 2.2),
    "adductor_pollicis": ("ulnar", ("c8", "t1"), 2.0),
    "first_dorsal_interosseous": ("ulnar", ("c8", "t1"), 2.2),
    "abductor_digiti_minimi": ("ulnar", ("c8", "t1"), 2.0),
    "lumbricals_ulnar": ("ulnar", ("c8", "t1"), 2.2),
    # -- trunk --------------------------------------------------------------
    "rectus_abdominis": ("thoracoabdominal", ("t7", "t12"), 0.8),
    "external_oblique": ("thoracoabdominal", ("t7", "t12"), 0.8),
    # the erector spinae is supplied by the DORSAL rami at every level, not by
    # the thoracoabdominal (ventral) nerves.  it said "thoracoabdominal" only
    # because no dorsal ramus was declared, and the whole deep back rode on a
    # nerve that supplies the abdominal wall.
    "erector_spinae": ("dorsal_ramus", ("t1", "l5"), 1.4),
    # QL also draws L1-L3 through the lumbar plexus rami directly; that second
    # supply is not a nerve this table can name in one slot, so it is recorded
    # here rather than smuggled into the subcostal root list.
    "quadratus_lumborum": ("subcostal", ("t12",), 1.0),
    # -- hip and thigh ------------------------------------------------------
    # psoas major is supplied by the L1-L3 anterior rami directly, not by the
    # femoral nerve; the iliac half is femoral and now has its own entry.
    "iliopsoas": ("lumbar_plexus", ("l1", "l2", "l3"), 1.0),
    # iliacus separately: the psoas half is supplied by the lumbar plexus rami
    # directly and the iliac half by the femoral nerve, so a body that models the
    # two bellies as distinct actuators cannot share one entry.
    "iliacus": ("femoral", ("l2", "l3"), 1.0),
    "sartorius": ("femoral", ("l2", "l3"), 1.0),
    "rectus_femoris": ("femoral", ("l2", "l3", "l4"), 1.0),
    "vastus_lateralis": ("femoral", ("l2", "l3", "l4"), 0.9),
    "vastus_medialis": ("femoral", ("l2", "l3", "l4"), 0.9),
    "vastus_intermedius": ("femoral", ("l2", "l3", "l4"), 0.9),
    "adductor_longus": ("obturator", ("l2", "l3", "l4"), 0.9),
    "adductor_magnus": ("obturator", ("l2", "l3", "l4"), 0.9),
    "adductor_brevis": ("obturator", ("l2", "l3", "l4"), 0.9),
    "obturator_externus": ("obturator", ("l3", "l4"), 0.9),
    "gracilis": ("obturator", ("l2", "l3"), 1.0),
    # the ischiocondylar head of adductor magnus is a hamstring: tibial division
    # of the sciatic, L4.  Giving it the obturator entry would put a hamstring on
    # the adductor nerve and make an obturator lesion cost hip extension.
    "adductor_magnus_ischiocondylar": ("tibial", ("l4",), 0.9),
    "gluteus_maximus": ("inferior_gluteal", ("l5", "s1", "s2"), 0.6),
    "gluteus_medius": ("superior_gluteal", ("l4", "l5", "s1"), 0.7),
    "gluteus_minimus": ("superior_gluteal", ("l4", "l5", "s1"), 0.7),
    "tensor_fasciae_latae": ("superior_gluteal", ("l4", "l5"), 0.8),
    "biceps_femoris_long": ("tibial", ("l5", "s1", "s2"), 0.9),
    "biceps_femoris_short": ("common_fibular", ("l5", "s1", "s2"), 0.9),
    "semitendinosus": ("tibial", ("l5", "s1", "s2"), 0.9),
    "semimembranosus": ("tibial", ("l5", "s1", "s2"), 0.9),
    # -- leg and foot -------------------------------------------------------
    "tibialis_anterior": ("deep_fibular", ("l4", "l5"), 1.1),
    "extensor_digitorum_longus": ("deep_fibular", ("l5", "s1"), 1.1),
    "extensor_hallucis_longus": ("deep_fibular", ("l5", "s1"), 1.1),
    "fibularis_longus": ("superficial_fibular", ("l5", "s1"), 1.0),
    "fibularis_brevis": ("superficial_fibular", ("l5", "s1"), 1.0),
    "fibularis_tertius": ("deep_fibular", ("l5", "s1"), 1.0),
    "gastrocnemius_medial": ("tibial", ("s1", "s2"), 0.9),
    "gastrocnemius_lateral": ("tibial", ("s1", "s2"), 0.9),
    "soleus": ("tibial", ("s1", "s2"), 1.5),
    "tibialis_posterior": ("tibial", ("l4", "l5"), 1.1),
    "flexor_digitorum_longus": ("tibial", ("l5", "s1"), 1.0),
    "flexor_hallucis_longus": ("tibial", ("s1", "s2"), 1.0),
    "popliteus": ("tibial", ("l4", "l5", "s1"), 1.0),
    "plantaris": ("tibial", ("s1", "s2"), 0.9),
    "extensor_digitorum_brevis": ("deep_fibular", ("l5", "s1"), 1.4),
    "extensor_hallucis_brevis": ("deep_fibular", ("l5", "s1"), 1.4),
    "abductor_hallucis": ("medial_plantar", ("s1", "s2"), 1.6),
    "flexor_digitorum_brevis": ("medial_plantar", ("s1", "s2"), 1.6),
    "abductor_digiti_minimi_foot": ("lateral_plantar", ("s2", "s3"), 1.6),
    "adductor_hallucis": ("lateral_plantar", ("s2", "s3"), 1.6),
    "flexor_accessorius": ("lateral_plantar", ("s1", "s2", "s3"), 1.4),
    "plantar_interossei": ("lateral_plantar", ("s2", "s3"), 1.6),
    "dorsal_interossei_foot": ("lateral_plantar", ("s2", "s3"), 1.6),
    # -- pelvic floor -------------------------------------------------------
    "levator_ani": ("pudendal", ("s2", "s3", "s4"), 0.8),
    "external_anal_sphincter": ("pudendal", ("s2", "s3", "s4"), 0.5),
}

SKELETAL_MUSCLES = REGISTRY.anatomy(Anatomy(
    name="skeletal_muscles",
    doc="named skeletal muscles, each an addressable motor-unit pool.  this is the "
        "system that makes the motor side CONNECTABLE: `myotomes` gives segmental "
        "territory and `peripheral_nerves` gives trunks, but neither answers 'which "
        "nerve reaches biceps', and without that answer there is no socket to plug a "
        "simulated muscle into.  each entry carries its nerve -- which fixes its "
        "conduction delay through `ibm.topologies.nerve` -- its root levels, which "
        "are multiple for nearly every limb muscle and are what a segmental lesion "
        "and a reflex arc act on, and a relative spindle density, because "
        "proprioception is emphatically not uniform: intrinsic hand muscles are "
        "spindle-rich, the large proximal movers are poor, and the extraocular "
        "muscles have no conventional spindles at all.  the list is the major named "
        "muscles rather than all ~640, and that is a stated approximation: the ones "
        "omitted are small, deep, and individually below the resolution any "
        "measurement of this model will have",
    labels=tuple(sorted(INNERVATION)),
    frame="body",
    crisp=True,
    covers="skeletal musculature",
    source="standard innervation tables, Gray's Anatomy 42e; spindle densities from "
           "Banks (2006) counts, which vary by an order of magnitude between muscles "
           "and are known far less precisely than the innervation",
    provenance=Provenance.LITERATURE))


def nerve_of(muscle: str) -> str:
    """which trunk carries this muscle's drive."""
    return INNERVATION[muscle][0]


def roots_of(muscle: str) -> tuple[str, ...]:
    """which spinal segments supply it."""
    return INNERVATION[muscle][1]


def muscles_on(nerve: str) -> tuple[str, ...]:
    """every muscle a trunk supplies -- i.e. what one lesion costs."""
    return tuple(m for m, (n, _, _) in sorted(INNERVATION.items()) if n == nerve)


def muscles_of_root(level: str) -> tuple[str, ...]:
    """every muscle a segment contributes to."""
    return tuple(m for m, (_, roots, _) in sorted(INNERVATION.items()) if level in roots)
