"""where each partitioning system's memberships actually come from.

`systems.py` declares that a partition exists.  this says how a(q) is obtained,
and it is a separate file because the two are separately wrong.  a system can be
correctly declared and completely unavailable; a system can be available in three
atlases that disagree; a system can be available only as a crisp file when the
anatomy is graded.  none of that is visible from a label list.

four things are recorded for each system, and the last two are the ones that get
lost otherwise.

*frame.*  an atlas is a set of positions in a coordinate frame, and every
membership is only as good as the warp that brought it there.  Julich-Brain is in
MNI ICBM152; the FreeSurfer subfield segmentations are computed in the subject's
own volume; SUIT has its own cerebellar template precisely because the whole-brain
warp destroys the folia.  `ibm.frames` carries the residuals, and a 2 mm residual
against a 3 mm nucleus is not a detail.

*crisp vs probabilistic.*  a crisp atlas is a probabilistic one with its
uncertainty deleted, and the deletion is not recoverable.  Glasser's areas are
crisp in the released file because a classifier thresholded a set of continuous
gradients; the gradients are the measurement and the borders are a decision.
where a probabilistic release exists ibm-1 wants it, and where only a crisp one
exists the honest move is to blur it by the known inter-subject border variability
rather than to pretend the border is a wall.

*the substitute.*  most preferred atlases are not obtainable for an arbitrary
subject -- they need held data, a 7T scan, a postmortem series, or a surface
registration ibm-1 may not have run.  the substitute is what actually gets used.

*what the substitute loses.*  this is the field that exists so the loss is stated
once, here, instead of being rediscovered as a puzzling result.  it is written in
terms of what a *process* can no longer distinguish, because that is the level at
which the loss bites.

card ids point at `data/sources/<id>/card.yaml`, which carries licence, access and
the systematic bias of each stream.  a `card` of "" means ibm-1 has no source card
for that atlas -- which is itself worth reading off this table.
"""

from __future__ import annotations

from dataclasses import dataclass

from ibm.registry import REGISTRY
from ibm.anatomy import systems as _systems          # noqa: F401  (registers them)


@dataclass(frozen=True)
class AtlasSource:
    """how one partitioning system's memberships are obtained."""

    system: str
    preferred: str            # the atlas ibm-1 would use given everything
    card: str                 # data/sources card id for the preferred atlas, "" if none
    frame: str                # the frame the preferred atlas is defined in
    crisp: bool               # what the released file claims, not what the anatomy does
    n_labels: int             # how many partitions the preferred atlas actually has
    measured_by: str          # the modality the borders come from
    obtained: str             # the pipeline that produces a(q) for one subject
    substitute: str           # what is used when the preferred atlas is unavailable
    substitute_card: str      # its card id
    lost: str                 # what a process can no longer distinguish
    note: str = ""


SOURCES: dict[str, AtlasSource] = {}


def _s(**kw) -> AtlasSource:
    a = AtlasSource(**kw)
    SOURCES[a.system] = a
    return a


# ---------------------------------------------------------------------------
# cortex
# ---------------------------------------------------------------------------

CORTICAL_AREAS = _s(
    system="cortical_areas",
    preferred="Glasser HCP-MMP1, 180 areas per hemisphere on fs_LR 32k",
    card="glasser2016",
    frame="fslr32k",
    crisp=True,
    n_labels=360,
    measured_by="joint gradients in myelin (T1w/T2w), cortical thickness, resting-state "
                "connectivity and task contrast -- a border is where several of them turn "
                "at once, which is a stronger criterion than any one of them",
    obtained="MSMAll multimodal surface registration of the subject to fs_LR, then the "
             "published areal classifier applied to that subject's own feature maps.  the "
             "result is a classifier output for this person, not a measurement of this "
             "person's areas, and the released files carry no per-vertex confidence",
    substitute="FreeSurfer aparc (Desikan-Killiany, 34 gyral labels per hemisphere) from "
               "recon-all, or Schaefer 2018 at 100-1000 parcels for a resolution-matched "
               "functional tiling",
    substitute_card="freesurfer / desikan2006 / dkt-atlas / schaefer2018",
    lost="areal identity.  a gyral label is a fold, and outside V1, MT, area 4 and the "
         "transverse temporal gyrus the folds do not track the areas -- superiorfrontal "
         "alone spans SMA, pre-SMA and several prefrontal areas with different laminar "
         "structure and different thalamic input.  any prior stated per area (a laminar "
         "profile, a receptor density, an intrinsic timescale) becomes an average over "
         "areas that differ in it.  Schaefer keeps the resolution but throws away the "
         "correspondence: parcel 137 has no name in the literature, so no prior can be "
         "attached to it at all",
    note="the released Glasser file is crisp; the underlying gradient maps are continuous "
         "and are the thing ibm-1 would rather have.  the card records this.")

CORTICAL_LAYERS = _s(
    system="cortical_layers",
    preferred="BigBrain laminar surfaces (Wagstyl et al. 2020): six layer boundaries "
              "traced through a 20 micron histological volume",
    card="bigbrain",
    frame="bigbrain (mapped to fsaverage / fs_LR via BigBrainWarp)",
    crisp=False,
    n_labels=6,
    measured_by="cell-body staining density profiles through the ribbon, in one 65-year-old "
                "postmortem brain",
    obtained="BigBrainWarp transfers the layer-thickness profiles to fsaverage or fs_LR; a "
             "subject's own memberships are then that template profile applied at the "
             "subject's equivolumetric depth coordinate.  equivolumetric rather than "
             "equidistant matters: cortex is folded, and equidistant depth sampling assigns "
             "gyral crowns and sulcal fundi to different layers at the same normalized depth",
    substitute="a fixed proportional depth model -- layer boundaries at constant fractions "
               "of the ribbon everywhere -- or, at 7T, a T1/T2* profile with the layer 4 "
               "stria visible only in primary areas",
    substitute_card="bigbrainwarp / freesurfer",
    lost="regional variation in the laminar profile, which is most of the signal.  a fixed "
         "proportional model makes agranular cingulate cortex have a layer 4, and makes V1 "
         "-- where layer 4 is three sublayers and nearly half the ribbon -- look like "
         "prefrontal cortex.  the feedforward / feedback asymmetry that laminar processes "
         "depend on is defined by exactly that difference",
    note="one brain.  inter-individual variability in laminar thickness is unmeasured by "
         "this source and is not a small effect.")

CYTOARCHITECTURE = _s(
    system="cytoarchitecture",
    preferred="Julich-Brain probabilistic cytoarchitectonic maps",
    card="julich-brain",
    frame="mni152 (also released on fsaverage and fs_LR surfaces)",
    crisp=False,
    n_labels=0,
    measured_by="observer-independent detection of laminar-profile discontinuities in "
                "cell-stained sections of ten postmortem brains, then warped to the "
                "template and averaged -- so a voxel's value is the fraction of those ten "
                "brains in which that area occupied that position",
    obtained="warp the subject to MNI152 (or use the surface release), read the per-area "
             "probability directly as a(q).  this is the one system in the ontology whose "
             "released form is already the soft membership the architecture asks for",
    substitute="Brodmann's 1909 map as digitised on a template, or the FreeSurfer "
               "Brodmann-area exvivo labels (BA1, BA2, BA3a/b, BA4a/p, BA6, BA44, BA45, "
               "V1, V2, MT, entorhinal, perirhinal) -- about a dozen areas, not fifty",
    substitute_card="von-economo / ba-exvivo",
    lost="the border uncertainty, which is the point of the atlas.  Julich-Brain says area "
         "44 occupies a given voxel in 3 of 10 brains; Brodmann says it does or does not.  "
         "a process weighted by a crisp map cannot express that a subject's area 44 might "
         "be a centimetre from the template's, and inter-subject areal displacement of that "
         "size is routine.  the coverage loss is also severe: Brodmann's map is a drawing "
         "of the lateral and medial surfaces and is unreliable inside sulci, where two "
         "thirds of the cortex is",
    note="Julich-Brain is incomplete by design -- it releases an area when the analysis is "
         "finished, so the maps do not tile the cortex.  ||a(q)||_1 < 1 over much of the "
         "brain here is correct, not a bug.")


# ---------------------------------------------------------------------------
# subcortex
# ---------------------------------------------------------------------------

THALAMIC_NUCLEI = _s(
    system="thalamic_nuclei",
    preferred="Morel stereotactic atlas of the human thalamus",
    card="",
    frame="a stereotactic atlas frame, warped to mni152 by the redistributors",
    crisp=True,
    n_labels=0,
    measured_by="myelin- and cell-stained serial sections of a small number of postmortem "
                "brains, drawn by hand at a nuclear level of detail no in-vivo method reaches",
    obtained="nonlinear warp of the subject to the template the atlas was resampled into.  "
             "there is no subject-specific information in the result at all: two people with "
             "the same brain shape get the same nuclei",
    substitute="THOMAS (white-matter-nulled MPRAGE segmentation, 11-12 nuclei), the "
               "FreeSurfer histological thalamic segmentation (26 nuclei from an ex-vivo "
               "prior plus subject T1 intensity), or a diffusion connectivity-based "
               "clustering of thalamus by cortical target",
    substitute_card="thomas-thalamic / freesurfer-thalamic-histological / "
                    "diffusion-thalamic-connectivity-atlases",
    lost="with a warped atlas: everything subject-specific, so a nuclear label is a "
         "positional prior and its error is the warp residual -- 2 mm against nuclei that "
         "are 3-5 mm across.  with the connectivity-based clustering: the nuclear identity "
         "itself.  a diffusion cluster is 'the part of thalamus that reaches motor cortex', "
         "which mixes VLp and VLa and part of VA, and those three differ in whether their "
         "input is cerebellar, pallidal or nigral -- the distinction most thalamic priors "
         "are about",
    note="no source card exists for Morel; the nearest held sources are the two segmentation "
         "tools above, both of which are ex-vivo priors wearing an in-vivo pipeline.")

HIPPOCAMPAL_SUBFIELDS = _s(
    system="hippocampal_subfields",
    preferred="FreeSurfer 7 hippocampal subfields (Iglesias et al. 2015), a probabilistic "
              "ex-vivo atlas from 15 ultra-high-resolution postmortem hippocampi",
    card="freesurfer",
    frame="subject_t1 -- the segmentation runs in the subject's own volume",
    crisp=False,
    n_labels=13,
    measured_by="manual delineation on 0.13 mm ex-vivo MRI, combined into a probabilistic "
                "tetrahedral mesh atlas fitted to the subject's intensities",
    obtained="segmentHA on a T1 (optionally with a T2 or a dedicated hippocampal-slab "
             "acquisition), which fits the mesh to the subject and returns posteriors",
    substitute="ASHS with a hippocampal T2 slab, which is the better choice when such a "
               "scan exists; the unfolded-hippocampus coordinate system where a continuous "
               "long-axis and proximal-distal coordinate is wanted instead of discrete fields",
    substitute_card="ashs / ashs-oap / unfolded-hippocampus",
    lost="on a 1 mm T1 alone, the internal borders are not in the data -- the whole "
         "subfield structure of a 1 mm segmentation is prior, and the subfield volumes it "
         "reports are largely a function of total hippocampal volume.  ca2 and ca3 are not "
         "separable at any in-vivo resolution and are reported merged or by fiat.  the "
         "long-axis gradient, which several literature results are actually about, is "
         "thrown away entirely by a discrete field label unless the tail is kept separate",
    note="the unfolded coordinate is arguably the more honest representation and is not a "
         "partition at all, which is why it is recorded here rather than declared as one.")

STRIOSOME_MATRIX = _s(
    system="striosome_matrix",
    preferred="mu-opioid receptor and calbindin immunohistochemistry on postmortem striatum",
    card="",
    frame="histological sections; no template release",
    crisp=False,
    n_labels=2,
    measured_by="chemoarchitecture -- striosomes are mu-opioid-rich and calbindin-poor "
                "patches 300-600 microns across, occupying 10-15% of striatal volume",
    obtained="not obtained.  there is no human in-vivo method with the resolution or the "
             "contrast, and no released template map",
    substitute="a spatially uniform membership at the histological volume fraction "
               "(striosome 0.15, matrix 0.85) everywhere in the striatum, with a mild "
               "dorsomedial-to-ventrolateral gradient if one wants to encode the known "
               "density trend",
    substitute_card="",
    lost="all spatial structure.  a uniform membership means every striatal position is the "
         "same mixture, so no process can produce a spatially patterned striosomal effect, "
         "and any measurement that would distinguish patch from matrix contributions is "
         "unidentifiable.  what is preserved is only the scalar mixing ratio, which is "
         "enough for a population-averaged dopaminergic feedback term and nothing more",
    note="declared with SPECULATIVE provenance for exactly this reason.  it is a place to "
         "put the structure, not a claim to have measured it.")

BG_TERRITORIES = _s(
    system="bg_territories",
    preferred="connectivity-based parcellation of striatum, pallidum and stn by cortical "
              "target (Tziortzi et al. 2014 for the seven-territory striatal form)",
    card="",
    frame="mni152",
    crisp=True,
    n_labels=7,
    measured_by="probabilistic tractography from each subcortical voxel to a set of cortical "
                "masks, then winner-take-all over the resulting connection probabilities",
    obtained="run the same tractography in the subject, or warp the published template "
             "parcellation.  the subject-specific route is preferable and is the one that "
             "makes the territory boundary mean something for that person",
    substitute="the classical tripartite sensorimotor / associative / limbic split, obtained "
               "by merging the seven; or Tian 2020, a gradient-based subcortical parcellation "
               "at four nested scales derived from resting-state functional connectivity",
    substitute_card="tian2020 / harvard-oxford",
    lost="winner-take-all is where the damage happens, not the merging.  the underlying "
         "connection profile is continuous -- striatal cortico-striatal input changes "
         "smoothly along a rostrocaudal gradient with no border in it -- and thresholding "
         "it to a crisp territory manufactures a wall in the middle of a ramp.  a "
         "membership taken from the normalized connection probabilities rather than from "
         "the argmax is both easier to obtain and closer to the anatomy.  Tian 2020 is "
         "gradient-derived and so avoids this, at the cost of its parcels being functional "
         "rather than anatomical and carrying no cortical-target interpretation",
    note="tractography from a small deep structure through the internal capsule is exactly "
         "the geometry tractography is worst at; the territories are stable across studies "
         "mostly because the cortical masks are.")

HYPOTHALAMIC_NUCLEI = _s(
    system="hypothalamic_nuclei",
    preferred="Allen human brain atlas nuclear delineations on the reference brain",
    card="allen-human-brain-atlas",
    frame="mni152 (the Allen reference brain is registered to it)",
    crisp=True,
    n_labels=0,
    measured_by="Nissl-stained sections of two whole postmortem brains, delineated by "
                "neuroanatomists at the nuclear level",
    obtained="nonlinear warp of the subject to MNI152 and readout of the warped "
             "delineation -- entirely a positional prior",
    substitute="the FreeSurfer hypothalamic subunits (Billot et al. 2020): five subunits "
               "per side -- anterior-inferior, anterior-superior, posterior, "
               "tubular-inferior, tubular-superior -- segmented from a T1 by a CNN",
    substitute_card="freesurfer",
    lost="the nuclei.  the five subunits are geometric thirds of the structure, not nuclei: "
         "'tubular-inferior' contains arcuate and ventromedial and part of lateral "
         "hypothalamus, which respectively drive feeding hormones, satiety and arousal.  "
         "the suprachiasmatic nucleus -- the entire circadian input to the model -- is "
         "under a millimetre across and is inside 'anterior-inferior' together with the "
         "preoptic area.  any circadian or autonomic process is therefore acting on a "
         "compartment several times larger than its actual substrate",
    note="a 2 mm warp residual against a 0.6 mm nucleus means the suprachiasmatic membership "
         "is a smooth bump centred on roughly the right place, and should be declared as one.")

AMYGDALAR_NUCLEI = _s(
    system="amygdalar_nuclei",
    preferred="FreeSurfer amygdalar subnuclei (Saygin et al. 2017)",
    card="freesurfer",
    frame="subject_t1",
    crisp=False,
    n_labels=9,
    measured_by="manual delineation on 0.1-0.15 mm ex-vivo MRI of ten postmortem "
                "amygdalae, built into a probabilistic mesh atlas",
    obtained="segmentHA fits the mesh to the subject's T1 intensities jointly with the "
             "hippocampal subfields and returns posteriors",
    substitute="a whole-amygdala label from aseg, or the Harvard-Oxford subcortical "
               "probabilistic amygdala",
    substitute_card="aseg-subcortical-supports / harvard-oxford",
    lost="the input/output distinction.  the lateral nucleus is the sensory input stage and "
         "the central nucleus is the brainstem output stage; they have opposite roles in "
         "any circuit process, and a whole-amygdala label averages them.  within the "
         "subnuclear segmentation itself the T1 contrast does not show the internal borders "
         "either, so the relative volumes are largely prior-driven -- but the topology "
         "(lateral lateral, central dorsomedial) is right, which is enough to orient a "
         "process even when the borders are not measured",
    note="the paralaminar and corticoamygdaloid-transition labels are thin sheets at the "
         "resolution limit and are the least reliable of the nine.")


# ---------------------------------------------------------------------------
# cerebellum and brainstem
# ---------------------------------------------------------------------------

CEREBELLAR_LOBULES = _s(
    system="cerebellar_lobules",
    preferred="SUIT probabilistic cerebellar atlas (Diedrichsen et al. 2009) with its "
              "deep-nuclei maps",
    card="suit-diedrichsen",
    frame="the SUIT cerebellum-only template, which is the point of it",
    crisp=False,
    n_labels=34,
    measured_by="manual lobular delineation on 20 high-resolution structural scans, "
                "normalized into a cerebellum-specific template and averaged, so the "
                "released maps carry the inter-subject boundary variability",
    obtained="isolate the cerebellum, normalize it to the SUIT template with the "
             "cerebellum-specific nonlinear registration, and read the probabilities back "
             "in subject space",
    substitute="a whole-brain MNI normalization with a lobular atlas resampled into it, or "
               "the MDTB functional cerebellar parcellation when function rather than "
               "anatomy is wanted",
    substitute_card="cerebellar-atlases / mdtb-cerebellar / bigbrain-hippocampal-cerebellar-derivatives",
    lost="with a whole-brain warp: the folia.  cerebellar folia are ~1 mm and a whole-brain "
         "nonlinear registration is not driven by cerebellar features at all, so lobular "
         "boundaries land several millimetres off and neighbouring lobules mix.  the "
         "cerebellar cortical surface area is comparable to a large fraction of the "
         "neocortex packed into a tenth the volume, which is exactly why this warp fails "
         "where the cortical one does not.  substituting the functional parcellation loses "
         "the opposite thing: the functional boundaries genuinely cross the lobules, so a "
         "prior stated per lobule cannot be attached to a functional parcel",
    note="the deep nuclei (dentate, interposed, fastigial) are iron-rich and are better "
         "localized on a susceptibility-weighted or QSM image than on a T1.")

CEREBELLAR_MICROZONES = _s(
    system="cerebellar_microzones",
    preferred="olivocerebellar zonal maps from tract tracing and zebrin II / aldolase C "
              "banding, in rodent",
    card="",
    frame="rodent histology; no human release",
    crisp=False,
    n_labels=8,
    measured_by="climbing-fibre tracing from subdivisions of the inferior olive, and the "
                "parasagittal aldolase C expression bands that align with them",
    obtained="not obtained in human.  the zones are a few hundred microns wide, run "
             "parasagittally across the lobules, and have no MRI contrast",
    substitute="a smooth parasagittal coordinate over the cerebellar cortex -- distance "
               "from the midline in the flattened SUIT surface -- used as a graded "
               "membership over the eight zones in their known mediolateral order",
    substitute_card="suit-diedrichsen",
    lost="the correspondence between a zone and its olivary subnucleus and deep-nucleus "
         "target, which is the only reason the zones matter.  a parasagittal coordinate "
         "preserves that zones are strips and that neighbours in the strip share a climbing "
         "fibre; it does not say which strip is C1.  the number of zones and their widths in "
         "human are also unknown -- the human cerebellum is not a scaled rodent one, and "
         "aldolase C banding in human has been shown only in fragments",
    note="declared SPECULATIVE.  the value of having it declared is that a process about "
         "climbing-fibre-driven plasticity has a topology and a partition to be defined "
         "over, and its parameters simply stay at their prior.")

BRAINSTEM_NUCLEI = _s(
    system="brainstem_nuclei",
    preferred="Brainstem Navigator (Bianciardi et al.): probabilistic in-vivo maps of ~60 "
              "brainstem nuclei from 7T multi-contrast imaging",
    card="brainstem-navigator",
    frame="mni152 (released in ICBM152 space; also 7T-native templates)",
    crisp=False,
    n_labels=0,
    measured_by="7T diffusion-fractional-anisotropy and T2-weighted contrast in a group of "
                "young adults, manually delineated per subject and averaged, so the released "
                "value is the fraction of subjects in whom that voxel was in that nucleus",
    obtained="warp the subject brainstem to the template -- brainstem-specific registration "
             "if possible, since a whole-brain warp is not driven by brainstem features -- "
             "and read the probabilities.  physiological noise correction matters here more "
             "than anywhere else in the brain: the brainstem moves with the cardiac cycle",
    substitute="the Allen human brain atlas delineations for nuclei the Navigator does not "
               "cover; a neuromelanin-sensitive scan for locus coeruleus and substantia "
               "nigra pars compacta specifically, which is the one in-vivo contrast that "
               "shows those two directly",
    substitute_card="allen-human-brain-atlas / freesurfer",
    lost="with a whole-brain warp and a template atlas: the locus coeruleus is a 2 x 15 mm "
         "column of a few thousand cells, so a 2 mm registration residual displaces it by "
         "its own width, and any signal extracted from it is mostly neighbouring pons.  the "
         "raphe nuclei are worse: a midline ribbon whose width is at the voxel size.  this "
         "is the single largest source of error in every neuromodulatory process in the "
         "ontology, and it is a localization error rather than a noise term",
    note="the cholinergic basal forebrain (nucleus basalis, medial septum / diagonal band) "
         "is carried in this system though it is forebrain, because no other declared system "
         "contains it and the neuromodulatory topology needs an acetylcholine source.  its "
         "maps come from a separate stereotactic source (Zaborszky's postmortem "
         "delineations, distributed with SPM's Anatomy toolbox), not from the Navigator.")


# ---------------------------------------------------------------------------
# vasculature
# ---------------------------------------------------------------------------

VASCULAR_TERRITORIES = _s(
    system="vascular_territories",
    preferred="digital arterial territory atlas (Liu et al. 2023)",
    card="arterial-territory-atlas-liu2023",
    frame="mni152",
    crisp=True,
    n_labels=0,
    measured_by="expert consensus over the clinical stroke literature, drawn on a template "
                "-- these are textbook territories rendered as voxels, not a measurement of "
                "anyone's actual vasculature",
    obtained="warp the subject to MNI152 and read the label.  a per-subject alternative "
             "exists and is much better where the data allow it: vessel-encoded or "
             "multi-delay arterial spin labelling measures which artery actually feeds each "
             "voxel in that person",
    substitute="the four major territories (aca, mca, pca, vertebrobasilar) only, or an "
               "artery-centreline model from a TOF angiogram with territories assigned by "
               "proximity along the tree",
    substitute_card="cerebral-artery-atlas-mouches2019 / circle-of-willis-centerline-resources / "
                    "asl-bold-breath-hold-datasets",
    lost="the individual variation, which in this system is the dominant term rather than a "
         "correction.  a fetal pca -- pca supplied from the internal carotid rather than the "
         "basilar -- occurs in something like a fifth of people and moves the entire "
         "occipital territory to the anterior circulation; the anterior communicating and "
         "posterior communicating segments are absent or hypoplastic often enough that the "
         "circle of willis is complete in well under half of brains.  a template territory "
         "map is therefore wrong in a specific, structured way for a large minority of "
         "subjects, and wrong exactly where it matters most, at the watershed.  the "
         "watershed zones themselves are the least reproducible part of any template map "
         "and are the part a perfusion process most needs to be probabilistic about",
    note="venous territories are not covered by any published atlas at this level.  "
         "7t-qsm-venograms gives subject-specific vein geometry but no territory labelling, "
         "so the venous side of the vascular topology has anatomy without a partition; that "
         "gap is real and is recorded here rather than filled with a guess.")


# ---------------------------------------------------------------------------
# the peripheral nervous system
# ---------------------------------------------------------------------------
#
# these six differ in kind from every source above them: there is no probabilistic
# atlas of the median nerve in a template space, because the peripheral nervous
# system has never been mapped the way the brain has.  what exists is dissection
# literature -- consistent, centuries old, and stated as topology and typical
# course rather than as a per-voxel probability.  that is recorded honestly here
# rather than dressed up as an atlas.
#
# the consequence for materialization is the opposite of the cortical case: these
# memberships will NOT be silently substituted, because there is nothing to
# substitute them with.  a request either accepts literature topology or does
# without the periphery.

CRANIAL_NERVES = _s(
    system="cranial_nerves",
    preferred="dissection anatomy: Gray's Anatomy 42e / Standring, cranial nerve course "
              "and fibre content",
    card=None,
    frame="body (no template exists)",
    crisp=False,
    n_labels=21,
    measured_by="cadaveric dissection, replicated for over a century.  fibre content per "
                "nerve is from degeneration and tracing studies; the vagus being ~80% "
                "afferent is a fibre count, not an impression",
    obtained="it is literature.  the in-vivo alternative is high-resolution cranial-nerve "
             "MR neurography, which resolves the proximal segments of V, VII and VIII and "
             "essentially nothing distal",
    substitute="MR neurography for the skull-base segments only",
    substitute_card=None,
    lost="the distal course.  a model that needs to know where the chorda tympani goes "
         "will not learn it from any image, and the literature course is the only "
         "available answer at any price")

SPINAL_LEVELS = _s(
    system="spinal_levels",
    preferred="standard segmental anatomy, 31 segments C1-Co1",
    card=None,
    frame="body",
    crisp=True,
    n_labels=31,
    measured_by="root counting; the least contentious fact in this file",
    obtained="literature.  segment-to-vertebra correspondence is the part that varies "
             "between people and it is a known offset, not a measurement",
    substitute="none needed",
    substitute_card=None,
    lost="nothing at the level of segment identity.  what is lost is the segment-to-"
         "VERTEBRA mapping in an individual, which matters only for localizing a "
         "structural lesion and not for the dynamics")

PERIPHERAL_NERVES = _s(
    system="peripheral_nerves",
    preferred="dissection anatomy: Gray's Anatomy 42e / Standring, named trunks and "
              "plexus formation",
    card=None,
    frame="body (no template exists)",
    crisp=False,
    n_labels=60,
    measured_by="cadaveric dissection.  plexus formation varies between individuals more "
                "than the textbook implies -- prefixed and postfixed brachial plexuses "
                "shift the whole segmental contribution by one level in a few percent of "
                "people",
    obtained="literature.  in vivo, MR neurography and high-resolution ultrasound image "
             "the large trunks (sciatic, median, ulnar) and nothing smaller",
    substitute="MR neurography or ultrasound for the major trunks",
    substitute_card=None,
    lost="everything distal to the named trunks, and the individual plexus variant.  a "
         "model fitted to one person's deficit pattern may be fitting their plexus "
         "variation and calling it a parameter")

DERMATOMES = _s(
    system="dermatomes",
    preferred="Lee et al. evidence-based composite; Foerster and Keegan-Garrett as the "
              "classical alternatives",
    card=None,
    frame="body",
    crisp=False,
    n_labels=29,
    measured_by="three incompatible methods: remaining sensation after multiple root "
                "section (Foerster), hypaesthesia from single-root compression "
                "(Keegan-Garrett), and pooled clinical evidence (Lee).  they disagree "
                "substantially over the trunk and the proximal limb",
    obtained="pick a map and record which one.  the disagreement between them IS the "
             "error bar and should be propagated rather than resolved by preference",
    substitute="any of the three; they are not interchangeable and the choice must be "
               "recorded in provenance",
    substitute_card=None,
    lost="precision that was never there.  adjacent dermatomes overlap widely, so any "
         "crisp map is a fiction -- which is why the declaration sets crisp=False even "
         "though every published figure draws hard borders")

MYOTOMES = _s(
    system="myotomes",
    preferred="standard segmental innervation tables, Gray's Anatomy 42e",
    card=None,
    frame="body",
    crisp=False,
    n_labels=29,
    measured_by="clinical correlation of root lesions with weakness, plus stimulation "
                "studies.  nearly every limb muscle draws from two or more segments, so "
                "the tables are weightings reported as lists",
    obtained="literature",
    substitute="none",
    substitute_card=None,
    lost="the weighting.  the tables say biceps is C5-C6 and do not say in what ratio, "
         "so a materialization must either fit the ratio or assume it uniform, and "
         "assuming it uniform is a real modelling choice that should be visible")

AUTONOMIC_GANGLIA = _s(
    system="autonomic_ganglia",
    preferred="dissection anatomy plus tracing: Gray's Anatomy 42e; Jaenig, The "
              "Integrative Action of the Autonomic Nervous System",
    card=None,
    frame="body",
    crisp=False,
    n_labels=24,
    measured_by="dissection for position; retrograde tracing and ganglion cell counts "
                "for the divergence ratios, which are the functionally important number "
                "and are known only to an order of magnitude",
    obtained="literature.  the sympathetic chain ganglia are visible on high-resolution "
             "CT and MR in some people; the intramural ganglia are not imageable at all",
    substitute="CT/MR for the chain and celiac ganglia only",
    substitute_card=None,
    lost="the divergence ratios, which is the parameter that decides how diffuse an "
         "autonomic response is.  1:10 and 1:100 are both quoted for sympathetic "
         "ganglia and the difference is an order of magnitude in effector recruitment")


# ---------------------------------------------------------------------------
# the invariant
# ---------------------------------------------------------------------------

_declared = set(REGISTRY.anatomies)
_sourced = set(SOURCES)
if _declared - _sourced:
    raise RuntimeError(
        "anatomical partitioning systems declared with no recorded source: "
        f"{sorted(_declared - _sourced)}.  a system whose provenance is not written down "
        "is a system whose memberships will be silently substituted at materialization; "
        "add an entry to ibm.anatomy.sources rather than removing this check.")
if _sourced - _declared:
    raise RuntimeError(
        "sources recorded for unregistered partitioning systems: "
        f"{sorted(_sourced - _declared)}.  ibm.anatomy.systems is the registry of what "
        "exists; this file only says where it comes from.")


def of(system: str) -> AtlasSource:
    if system not in SOURCES:
        raise KeyError(f"no recorded source for {system!r}; "
                       f"known: {', '.join(sorted(SOURCES))}")
    return SOURCES[system]


def unsourced_cards() -> list[str]:
    """systems whose preferred atlas has no card in data/sources.

    these are the ones where ibm-1 knows what it wants and has not written down
    the licence, access and bias of getting it -- which is the state a source is
    in just before someone uses it without noticing that they cannot redistribute it.
    """
    return sorted(a.system for a in SOURCES.values() if not a.card)


def table() -> str:
    """the sources as one printable table, in the spirit of REGISTRY.table()."""
    head = ("system", "preferred", "card", "frame", "crisp")
    rows = [(a.system, a.preferred.split(" (")[0][:44], a.card or "-", a.frame.split(" ")[0],
             "yes" if a.crisp else "prob")
            for a in sorted(SOURCES.values(), key=lambda x: x.system)]
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    line = "  ".join(h.ljust(x) for h, x in zip(head, w))
    rule = "  ".join("-" * x for x in w)
    body = "\n".join("  ".join(c.ljust(x) for c, x in zip(r, w)) for r in rows)
    return f"{line}\n{rule}\n{body}"


__all__ = ["AtlasSource", "SOURCES", "of", "table", "unsourced_cards"]
