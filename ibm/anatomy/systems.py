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
