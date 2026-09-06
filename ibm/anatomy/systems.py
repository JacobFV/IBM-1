"""anatomical partitioning systems.

    a(q) in [0,1]^k,  ||a(q)||_1 <= 1

a partitioning system is a soft map from position to named partitions, and the
inequality is the whole design: a system may cover only part of the brain, so
thalamic nuclei say nothing about a cortical position rather than being forced to
say "none of the above".

several systems apply at once and they are not required to nest.  a position in
the putamen is simultaneously in the *matrix* compartment and in the
*sensorimotor* basal-ganglia territory and in the *lenticulostriate* vascular
territory, and none of those three is a refinement of another.  that is why this
is a list of independent systems rather than one label tree: a single hierarchical
atlas would have to choose which of the three is the parent, and every choice is
wrong for some process.

cell populations are *not* here.  excitatory, pv, sst and vip cells coexist at the
same cubic millimetre, so they are components of a field (ARCHITECTURE.md §2) --
a partition would have to make them mutually exclusive in space, which they are
not.  the rule is simple: if two things can be at the same place at the same time,
they are components; if they tile space, they are partitions.

memberships enter processes as weights, never as masks.  a partition boundary in
this ontology is a gradient: the cytoarchitectonic transition between two areas
takes a millimetre or two of cortex and varies between people by more than that,
so `crisp=True` here records what an atlas *file* claims, not what the anatomy
does.  a crisp atlas used as a weight is a probabilistic atlas with all its
uncertainty deleted, and `ibm.anatomy.sources` says which ones those are.

the label lists here are the ones ibm-1 can enumerate honestly.  where the
preferred atlas is larger than the enumerable one -- 360 Glasser areas against 34
FreeSurfer gyri -- the label list is the substitute and `sources.py` records the
preferred system, what it costs to obtain, and what is lost by not having it.  a
wrong label list is worse than a coarse one.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Anatomy
from ibm.vocabulary import Provenance

A = REGISTRY.anatomy


def _a(name: str, doc: str, labels, frame: str, **kw) -> Anatomy:
    kw.setdefault("provenance", Provenance.ATLAS)
    return A(Anatomy(name=name, doc=doc, labels=tuple(labels), frame=frame, **kw))


# ---------------------------------------------------------------------------
# cortex
# ---------------------------------------------------------------------------

#: FreeSurfer `aparc` (Desikan-Killiany).  34 per hemisphere; the hemisphere is
#: not in the label because lateralization is a materialization concern -- a
#: request instantiates left and right sites and both carry the same label set.
DK_GYRI = (
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula")

CORTICAL_AREAS = _a(
    "cortical_areas",
    "the parcellation of the cortical sheet into areas -- units that differ from their "
    "neighbours in more than one property at once (myeloarchitecture, thickness, "
    "connectivity, function).  it is the system almost every other declaration is indexed "
    "through, because 'V1' and 'area 44' are how the literature states nearly every prior "
    "ibm-1 will ever use.  the enumerated labels are the FreeSurfer gyral set, which is "
    "obtainable for any subject with a T1; the preferred system is Glasser HCP-MMP1, whose "
    "360 areas are read from the atlas rather than written here.  the difference is not "
    "cosmetic: gyral labels follow folding, and outside V1 and a few others folding does "
    "not follow areal boundaries",
    DK_GYRI, "fsaverage", crisp=True, hierarchical=False, covers="neocortex",
    source="freesurfer aparc (Desikan et al. 2006); preferred: Glasser et al. 2016")

CORTICAL_LAYERS = _a(
    "cortical_layers",
    "the six-layer depth partition of the cortical ribbon.  it is a partition of *depth*, "
    "not of the sheet: every surface position has all six, and a layer occupies a "
    "proportion of the ribbon rather than a thickness in millimetres, because cortical "
    "thickness varies two-fold across the sheet and three-fold between an agranular and a "
    "granular field.  layer 4 is genuinely absent in agranular cortex, which is why the "
    "membership sums to less than one there rather than being redistributed",
    ("layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6"),
    "fsaverage", crisp=False, hierarchical=False, covers="cortical ribbon",
    source="BigBrain laminar segmentation (Wagstyl et al. 2020) via BigBrainWarp")

CYTOARCHITECTURE = _a(
    "cytoarchitecture",
    "the cell-packing partition: Brodmann's numbering as the shared vocabulary and "
    "Julich-Brain as the actual measurement.  it is *not* a coarser cortical_areas -- the "
    "two systems disagree, and the disagreement is informative.  Brodmann drew borders in "
    "one hemisphere of a handful of brains; Julich-Brain gives an observer-independent "
    "border probability from ten postmortem brains, so a position is 60% area 44 and 40% "
    "area 45 rather than one or the other.  keeping it separate from cortical_areas is what "
    "lets a prior stated in Brodmann numbers and a boundary measured multi-modally "
    "coexist without one silently overwriting the other",
    tuple(f"ba_{n}" for n in (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 48, 52)),
    "mni152", crisp=False, hierarchical=False, covers="cerebral cortex",
    source="Julich-Brain probabilistic cytoarchitectonic maps; Brodmann 1909 numbering")


# ---------------------------------------------------------------------------
# subcortex
# ---------------------------------------------------------------------------

THALAMIC_NUCLEI = _a(
    "thalamic_nuclei",
    "the thalamus as nuclei rather than as a blob.  it exists as its own system because the "
    "thalamus is the one structure where the functional identity of a position is almost "
    "entirely nuclear: LGN, VPL and MD are neighbours in space and have nothing in common "
    "in their inputs, outputs or firing.  the Morel numbering is the vocabulary the "
    "literature uses; standard MRI contrast does not show these borders at all, so every "
    "in-vivo membership here is a warped histological prior or a diffusion-derived cluster, "
    "and the honest form of both is probabilistic",
    ("av", "am", "ad", "ld", "lp", "va", "vamc", "vla", "vlp", "vpl", "vpm", "vm",
     "vpi", "cl", "cem", "cm", "pf", "pc", "pt", "mv_re", "mdm", "mdl",
     "pul_a", "pul_m", "pul_l", "pul_i", "lgn", "mgn", "l_sg", "r", "hb"),
    "mni152", crisp=False, hierarchical=True, covers="thalamus",
    source="Morel stereotactic atlas; THOMAS and FreeSurfer histological segmentations")

HIPPOCAMPAL_SUBFIELDS = _a(
    "hippocampal_subfields",
    "the transverse partition of the hippocampal formation.  it is a separate system from "
    "cortical_areas because the hippocampus is allocortex: it has three layers, not six, so "
    "cortical_layers does not apply to it, and its subfields are a circuit sequence "
    "(dg -> ca3 -> ca1 -> subiculum) rather than a tiling of a sheet.  the fields differ in "
    "connectivity and in vulnerability by more than most cortical areas differ from each "
    "other, which is why a whole-hippocampus label is nearly useless as a prior",
    ("ca1", "ca2", "ca3", "ca4", "dg", "subiculum", "presubiculum", "parasubiculum",
     "molecular_layer", "hata", "fimbria", "hippocampal_fissure", "tail"),
    "subject_t1", crisp=False, hierarchical=False, covers="hippocampal formation",
    source="FreeSurfer 7 subfields (Iglesias et al. 2015); ASHS for T2 subfield segmentation")

STRIOSOME_MATRIX = _a(
    "striosome_matrix",
    "the neurochemical compartments of the striatum.  they are the clearest case in the "
    "ontology of a partition that is real, functionally decisive and invisible to every "
    "in-vivo method: striosomes are 10-15% of striatal volume in patches a few hundred "
    "microns across, they project to the dopaminergic midbrain while the matrix projects to "
    "the pallidum, and no human MRI resolves them.  the system is declared anyway because a "
    "process that models dopaminergic feedback needs a place to put the distinction, and a "
    "membership that is everywhere 0.15/0.85 is an honest statement of what is known",
    ("striosome", "matrix"),
    "mni152", crisp=False, covers="striatum",
    source="mu-opioid / calbindin histochemistry; no human in-vivo source",
    provenance=Provenance.SPECULATIVE)

BG_TERRITORIES = _a(
    "bg_territories",
    "the functional territories of the basal ganglia, defined by which cortex projects "
    "where.  it cuts across striosome_matrix and across the anatomical nuclei both: the "
    "sensorimotor territory spans caudal putamen and the dorsolateral pallidum and stn "
    "together, so it is a partition of the basal-ganglia system rather than of any one "
    "structure.  the seven-territory form follows the cortical parcels that drive the "
    "connectivity clustering; the classical tripartite sensorimotor / associative / limbic "
    "split is these seven merged, and is what most literature priors are stated over",
    ("limbic", "executive", "rostral_motor", "caudal_motor", "parietal", "occipital",
     "temporal"),
    "mni152", crisp=False, hierarchical=True, covers="striatum, pallidum, stn, sn",
    source="diffusion connectivity-based parcellation (Tziortzi et al. 2014); Tian 2020 "
           "for the gradient-based subcortical alternative")

HYPOTHALAMIC_NUCLEI = _a(
    "hypothalamic_nuclei",
    "the hypothalamic nuclei, which matter to ibm-1 out of proportion to their volume: they "
    "are where circadian, autonomic, thermal and arousal state is set, and a model that "
    "carries thermal, metabolic and vascular fields but has no partition for the structure "
    "that regulates them has a hole in it.  the nuclei are 1-3 mm across and are not "
    "separable on a clinical T1, so the enumerated list is the histological vocabulary and "
    "the obtainable membership is a five-subunit segmentation that merges most of them",
    ("preoptic_medial", "preoptic_lateral", "suprachiasmatic", "supraoptic",
     "paraventricular", "anterior_hypothalamic_area", "arcuate", "ventromedial",
     "dorsomedial", "lateral_hypothalamic_area", "tuberomammillary",
     "posterior_hypothalamic_area", "mammillary_medial", "mammillary_lateral"),
    "mni152", crisp=False, covers="hypothalamus",
    source="Allen human brain atlas nuclear delineations; FreeSurfer hypothalamic "
           "subunits (Billot et al. 2020) as the in-vivo substitute")

AMYGDALAR_NUCLEI = _a(
    "amygdalar_nuclei",
    "the amygdalar nuclei.  the amygdala is not one structure: the lateral nucleus is "
    "cortex-like and receives sensory input, the central nucleus is striatum-like and drives "
    "brainstem autonomic output, and treating their union as a region averages an input "
    "stage with an output stage.  the in-vivo segmentation is an ex-vivo prior warped into "
    "the subject, so the borders are as uncertain as the warp",
    ("lateral", "basal", "accessory_basal", "paralaminar", "central", "medial", "cortical",
     "anterior_amygdaloid_area", "corticoamygdaloid_transition"),
    "subject_t1", crisp=False, covers="amygdala",
    source="FreeSurfer amygdalar subnuclei (Saygin et al. 2017)")


# ---------------------------------------------------------------------------
# cerebellum and brainstem
# ---------------------------------------------------------------------------

CEREBELLAR_LOBULES = _a(
    "cerebellar_lobules",
    "the lobular partition of the cerebellum plus the deep nuclei.  it needs its own system "
    "and its own template because cerebellar folia are an order of magnitude finer than "
    "cortical gyri and a whole-brain nonlinear registration smears them together; a "
    "cerebellum-specific normalization is the only way the lobules survive.  the lobules are "
    "anatomy, not function -- the functional boundaries run across them -- so this system "
    "and a functional cerebellar parcellation are both needed and neither replaces the other",
    ("lobule_i_iv", "lobule_v", "lobule_vi", "crus_i", "crus_ii", "lobule_viib",
     "lobule_viiia", "lobule_viiib", "lobule_ix", "lobule_x",
     "vermis_vi", "vermis_crus_i", "vermis_crus_ii", "vermis_viib", "vermis_viiia",
     "vermis_viiib", "vermis_ix", "vermis_x",
     "dentate", "interposed", "fastigial"),
    "mni152", crisp=False, hierarchical=True, covers="cerebellum",
    source="SUIT probabilistic atlas (Diedrichsen et al. 2009) plus its deep-nuclei maps")

CEREBELLAR_MICROZONES = _a(
    "cerebellar_microzones",
    "the parasagittal olivocerebellar zones -- the strips, a few hundred microns wide, that "
    "share a climbing-fibre origin and a deep-nucleus target and are the actual functional "
    "unit of the cerebellar cortex.  they run *orthogonal* to the lobules, which is exactly "
    "why they are a separate system: a lobular label and a zonal label at one position are "
    "independent facts, and merging them into one atlas would force an ordering that does "
    "not exist.  no human in-vivo method resolves a microzone; this is declared as a place "
    "for the structure to live, with memberships that are at best a smooth prior",
    ("zone_a", "zone_b", "zone_c1", "zone_c2", "zone_c3", "zone_d0", "zone_d1", "zone_d2"),
    "mni152", crisp=False, covers="cerebellar cortex",
    source="rodent zebrin II / aldolase C banding and olivocerebellar tracing; "
           "no human source at this scale",
    provenance=Provenance.SPECULATIVE)

BRAINSTEM_NUCLEI = _a(
    "brainstem_nuclei",
    "the brainstem nuclei, and -- deliberately -- the cholinergic basal forebrain with them. "
    "these are the origins of every ascending neuromodulatory projection and every cranial "
    "reflex arc, so the neuromodulatory topology reads its source sites out of this system; "
    "without it there is nowhere in the ontology to say 'locus coeruleus'.  the nuclei are "
    "1-5 mm structures with no contrast on a standard T1, and the ones that matter most "
    "(lc, raphe, vta) are the smallest.  including the basal forebrain here is a compromise "
    "recorded rather than hidden: it is not brainstem, and no other declared system contains it",
    ("locus_coeruleus", "dorsal_raphe", "median_raphe", "raphe_magnus",
     "substantia_nigra_compacta", "substantia_nigra_reticulata", "ventral_tegmental_area",
     "pedunculopontine", "laterodorsal_tegmental", "periaqueductal_gray",
     "superior_colliculus", "inferior_colliculus", "red_nucleus", "subthalamic_nucleus",
     "nucleus_tractus_solitarius", "parabrachial", "inferior_olive", "pontine_nuclei",
     "vestibular_nuclei", "cochlear_nuclei", "oculomotor_nucleus", "trochlear_nucleus",
     "abducens_nucleus", "trigeminal_motor_nucleus", "trigeminal_sensory_nucleus",
     "facial_nucleus", "hypoglossal_nucleus", "nucleus_ambiguus", "dorsal_motor_vagus",
     "cuneate_nucleus", "gracile_nucleus", "mesencephalic_reticular_formation",
     "pontine_reticular_formation", "medullary_reticular_formation",
     "nucleus_basalis_of_meynert", "medial_septum_diagonal_band"),
    "mni152", crisp=False, covers="brainstem and basal forebrain",
    source="Brainstem Navigator (Bianciardi) 7T in-vivo probabilistic maps; "
           "Allen human brain atlas for the nuclei it does not cover")


# ---------------------------------------------------------------------------
# vasculature
# ---------------------------------------------------------------------------

VASCULAR_TERRITORIES = _a(
    "vascular_territories",
    "which artery supplies a position.  it is the one partitioning system here that has "
    "nothing to do with neural identity and everything to do with a shared failure mode: two "
    "positions in the same territory share a perfusion pressure, a co2 response and a "
    "vulnerability, and that is a fact no cytoarchitectonic or connectivity parcellation "
    "expresses.  it cuts across every other system -- the mca territory spans a dozen "
    "cortical areas and half the striatum -- which is the clearest illustration of why "
    "partitioning systems are independent rather than nested.  the watershed labels are "
    "explicitly not one-hot: a watershed position is supplied by both arteries and by "
    "neither reliably, so its membership is genuinely split",
    ("aca", "mca", "pca", "vertebrobasilar", "anterior_choroidal",
     "lenticulostriate_medial", "lenticulostriate_lateral", "thalamoperforating",
     "sca", "aica", "pica",
     "watershed_aca_mca", "watershed_mca_pca", "watershed_deep"),
    "mni152", crisp=False, hierarchical=True, covers="whole brain",
    source="digital arterial territory atlas (Liu et al. 2023); cerebral artery atlas "
           "(Mouches & Forkert 2019) for the vessel geometry the territories derive from")


__all__ = [n for n in dir() if n.isupper() and not n.startswith("_")]


# ---------------------------------------------------------------------------
# the peripheral nervous system
# ---------------------------------------------------------------------------
#
# these are partitions and not components, by this file's own rule: a nerve trunk
# occupies a place and two trunks do not occupy the same place, so they tile.  what
# runs INSIDE a trunk -- Ia, Ib, II, A-beta, A-delta, C, alpha, gamma, B,
# postganglionic C -- coexists at every millimetre of it and is therefore declared
# as components of the neural field (`ibm.fields.neural`, "peripheral traffic,
# resolved by fibre class").
#
# so the pair is: `peripheral_nerves` x fibre-class components = a multidimensional
# nerve.  neither half is meaningful alone.  a trunk with no fibre vector is a wire
# with one number on it, and a fibre vector with no trunk has nowhere to be.
#
# the frame is `body` rather than `mni152` throughout, and that is not a formality:
# there is no image in which the median nerve is a path between two brain positions,
# for exactly the reason `ibm.topologies.afferent` gives about the optic nerve.

CRANIAL_NERVES = _a(
    "cranial_nerves",
    "the twelve cranial nerves, with the trigeminal and facial divisions enumerated "
    "separately because they have different targets and different fibre content.  "
    "their central nuclei are already in `brainstem_nuclei`; this system is the "
    "PERIPHERAL course, which that system has no way to name.  several are mixed in "
    "a way that matters: the facial nerve carries somatic motor, parasympathetic and "
    "special sensory in one trunk, and the vagus is roughly 80% afferent despite "
    "being universally described as a motor nerve -- a fact that is invisible unless "
    "fibre class is a separate axis",
    ("olfactory_i", "optic_ii", "oculomotor_iii", "trochlear_iv",
     "trigeminal_v1_ophthalmic", "trigeminal_v2_maxillary", "trigeminal_v3_mandibular",
     "abducens_vi", "facial_vii_somatic_motor", "facial_vii_parasympathetic",
     "facial_vii_chorda_tympani", "vestibulocochlear_viii_cochlear",
     "vestibulocochlear_viii_vestibular", "glossopharyngeal_ix", "vagus_x_pharyngeal",
     "vagus_x_recurrent_laryngeal", "vagus_x_cardiac", "vagus_x_pulmonary",
     "vagus_x_abdominal", "accessory_xi", "hypoglossal_xii"),
    "body", crisp=False, covers="cranial periphery",
    source="Gray's Anatomy 42e; Standring, cranial nerve courses and fibre content",
    provenance=Provenance.LITERATURE)

SPINAL_LEVELS = _a(
    "spinal_levels",
    "the thirty-one spinal cord segments and their paired roots, C1-Co1.  the "
    "segmental level is the organizing coordinate of the entire body periphery: a "
    "dermatome, a myotome, a sympathetic outflow and a reflex arc are all indexed by "
    "it, and no other declared system can express 'below T6'.  note that segment and "
    "vertebra diverge caudally -- the cord ends near L1-L2, so lumbosacral roots "
    "descend as the cauda equina and a segmental lesion and a vertebral lesion are "
    "different statements",
    ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
     "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12",
     "l1", "l2", "l3", "l4", "l5", "s1", "s2", "s3", "s4", "s5", "co1"),
    "body", crisp=True, covers="spinal cord and roots",
    source="standard segmental anatomy; Gray's Anatomy 42e",
    provenance=Provenance.LITERATURE)

PERIPHERAL_NERVES = _a(
    "peripheral_nerves",
    "named peripheral nerve trunks and the plexuses that form them.  the plexuses are "
    "included as labels rather than being resolved away because a plexus is where "
    "segmental level stops predicting peripheral distribution: the median nerve draws "
    "from C6-T1 and no single level owns it, which is precisely why a dermatome map "
    "and a peripheral-nerve map disagree and why the pattern of a deficit localizes a "
    "lesion.  a model that had only levels could not represent a carpal-tunnel "
    "distribution, and one that had only nerves could not represent a radiculopathy",
    (# plexuses
     "cervical_plexus", "brachial_plexus", "lumbar_plexus", "sacral_plexus",
     # cervical
     "phrenic", "ansa_cervicalis", "lesser_occipital", "great_auricular",
     "transverse_cervical", "supraclavicular",
     # brachial
     "dorsal_scapular", "long_thoracic", "suprascapular", "lateral_pectoral",
     "medial_pectoral", "medial_cutaneous_arm", "medial_cutaneous_forearm",
     "upper_subscapular", "lower_subscapular", "thoracodorsal", "axillary",
     "musculocutaneous", "median", "ulnar", "radial",
     "anterior_interosseous", "posterior_interosseous", "superficial_radial",
     "palmar_digital", "dorsal_digital",
     # thoracic
     "intercostal", "subcostal", "thoracoabdominal",
     # lumbar
     "iliohypogastric", "ilioinguinal", "genitofemoral",
     "lateral_femoral_cutaneous", "femoral", "obturator", "saphenous",
     # sacral
     "superior_gluteal", "inferior_gluteal", "posterior_femoral_cutaneous",
     "sciatic", "tibial", "common_fibular", "superficial_fibular", "deep_fibular",
     "sural", "medial_plantar", "lateral_plantar", "pudendal",
     # autonomic trunks
     "sympathetic_chain", "greater_splanchnic", "lesser_splanchnic",
     "least_splanchnic", "lumbar_splanchnic", "pelvic_splanchnic"),
    "body", crisp=False, covers="peripheral nerve trunks",
    source="Gray's Anatomy 42e; Standring. courses are literature, not imaged",
    provenance=Provenance.LITERATURE)

DERMATOMES = _a(
    "dermatomes",
    "the skin territory of each dorsal root, tiling the body surface.  they tile and "
    "they OVERLAP -- adjacent dermatomes share a wide border, which is why sectioning "
    "one root produces hypaesthesia rather than anaesthesia -- so `crisp=False` here "
    "is anatomy and not atlas uncertainty.  the published maps (Keegan-Garrett, "
    "Foerster, Lee) disagree substantially, and that disagreement is the honest error "
    "bar on any somatotopic claim this model makes about the trunk or limbs",
    tuple(f"{seg}_dermatome" for seg in
          ("c2", "c3", "c4", "c5", "c6", "c7", "c8", "t1", "t2", "t3", "t4", "t5",
           "t6", "t7", "t8", "t9", "t10", "t11", "t12", "l1", "l2", "l3", "l4",
           "l5", "s1", "s2", "s3", "s4", "s5")),
    "body", crisp=False, covers="body surface",
    source="Lee/Foerster/Keegan-Garrett maps; they disagree and the disagreement is "
           "recorded rather than resolved",
    provenance=Provenance.LITERATURE)

MYOTOMES = _a(
    "myotomes",
    "the muscle territory of each ventral root.  almost every limb muscle is "
    "innervated by two or more segments, so this system is emphatically NOT crisp and "
    "a myotome is a weighting rather than a set -- which is exactly why single-root "
    "lesions cause weakness rather than paralysis.  it is the motor counterpart of "
    "`dermatomes` and shares its indexing so that a segmental statement can be made "
    "about both limbs of a reflex arc at once",
    tuple(f"{seg}_myotome" for seg in
          ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "t1", "t2", "t3", "t4",
           "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "l1", "l2", "l3",
           "l4", "l5", "s1", "s2", "s3", "s4")),
    "body", crisp=False, covers="skeletal musculature",
    source="standard segmental innervation tables; Gray's Anatomy 42e",
    provenance=Provenance.LITERATURE)

AUTONOMIC_GANGLIA = _a(
    "autonomic_ganglia",
    "the ganglia where preganglionic autonomic axons synapse.  they are declared "
    "because the synapse is the reason `neural.efferent.b_preganglionic` and "
    "`neural.efferent.c_postganglionic` are separate components: a ganglion is a "
    "place where divergence happens, and sympathetic divergence ratios of 1:10 or "
    "more are why a sparse preganglionic outflow produces a diffuse effector "
    "response.  the parasympathetic ganglia sit near or in their targets and diverge "
    "far less, which is the structural basis of its finer targeting",
    ("superior_cervical", "middle_cervical", "stellate", "thoracic_chain",
     "lumbar_chain", "sacral_chain", "celiac", "superior_mesenteric",
     "inferior_mesenteric", "aorticorenal", "ciliary", "pterygopalatine",
     "submandibular", "otic", "intramural_cardiac", "intramural_enteric",
     "intramural_pelvic", "dorsal_root_ganglion", "trigeminal_ganglion",
     "geniculate_ganglion", "spiral_ganglion", "vestibular_ganglion",
     "nodose_ganglion", "jugular_ganglion"),
    "body", crisp=False, covers="peripheral ganglia",
    source="Gray's Anatomy 42e; Jaenig, The Integrative Action of the Autonomic "
           "Nervous System",
    provenance=Provenance.LITERATURE)
