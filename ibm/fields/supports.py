"""the domains fields are indexed over.

    F = {x(q) : q in Omega}

Omega is part of the field definition, which is why supports live here and not in
a directory of their own.  fields do not share a support: the cortical surface,
the vascular tree, the interstitial volume, the retina, the musculature and an
instrument's contacts have different notions of adjacency and distance.  that is
the reason ibm-1 has no universal spatial operator and carries spatial structure
in topologies instead.

`min_spacing_mm` is not a resolution setting.  it is the finest sampling at which
a support is still physically meaningful -- below it the continuum description
the field assumes has broken down, and the materializer records a violation
rather than obliging.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Support

S = REGISTRY.support

# -- neural tissue ----------------------------------------------------------
TISSUE = S(Support("tissue", "brain parenchyma as a volume: grey and white matter, "
    "subcortical structures, brainstem and cerebellum", "volume", 3, "subject_t1",
    extent="whole brain", min_spacing_mm=0.01))

CORTICAL_SURFACE = S(Support("cortical_surface", "the folded cortical sheet, where geodesic "
    "distance rather than euclidean distance governs lateral interaction -- two points a "
    "millimetre apart across a sulcus may be centimetres apart along the sheet",
    "surface", 2, "subject_surf", extent="neocortex and allocortex", min_spacing_mm=0.1))

CORTICAL_DEPTH = S(Support("cortical_depth", "normalized depth through the cortical ribbon, "
    "pial to white.  laminar position, not a distance: cortical thickness varies two-fold "
    "across the sheet and layers are defined by proportion", "manifold", 1, "subject_surf",
    extent="cortical ribbon", min_spacing_mm=0.05))

# -- fluid compartments and transport networks ------------------------------
VASCULAR_TREE = S(Support("vascular_tree", "the cerebral vasculature as a branching graph from "
    "the circle of willis to capillary beds and back through venules and sinuses.  a tree, not "
    "a volume: two capillaries adjacent in space may be far apart in flow", "tree", 1,
    "subject_t1", extent="arterial, capillary and venous", min_spacing_mm=0.01))

CSF_SPACE = S(Support("csf_space", "ventricles, cisterns and the subarachnoid space",
    "volume", 3, "subject_t1", extent="csf compartment", min_spacing_mm=0.1))

INTERSTITIAL = S(Support("interstitial", "the extracellular space as a tortuous connected "
    "volume.  its effective diffusivity is the free value divided by tortuosity squared, so "
    "geometry here is a material property rather than a mesh", "volume", 3, "subject_t1",
    extent="brain parenchyma", min_spacing_mm=0.001))

# -- head, body and world ---------------------------------------------------
HEAD_VOLUME = S(Support("head_volume", "the whole head as a conductor and mechanical body: "
    "brain, csf, skull, scalp, air cavities", "volume", 3, "subject_t1",
    extent="head", min_spacing_mm=0.1))

SCALP = S(Support("scalp", "the outer scalp surface, where non-invasive electrodes sit",
    "surface", 2, "subject_t1", extent="scalp", min_spacing_mm=1.0))

BODY = S(Support("body", "body segments, joints and musculature", "manifold", 3, "body",
    extent="whole body", min_spacing_mm=1.0))

VISCERA = S(Support("viscera", "heart, lungs, gut and vasculature outside the head, where "
    "interoceptive afferents originate", "manifold", 3, "body", extent="thorax and abdomen",
    min_spacing_mm=1.0))

# -- receptor surfaces ------------------------------------------------------
RETINA = S(Support("retina", "the retinal surface of both eyes, in eye-centred coordinates. "
    "receptor density spans two orders of magnitude from fovea to periphery, so uniform "
    "sampling here is always wrong", "surface", 2, "eye", extent="both retinae",
    min_spacing_mm=0.002))

COCHLEA = S(Support("cochlea", "the basilar membrane of both ears, parameterized tonotopically. "
    "one-dimensional and logarithmic in frequency", "manifold", 1, "body",
    extent="both cochleae", min_spacing_mm=0.01))

VESTIBULAR_ORGAN = S(Support("vestibular_organ", "semicircular canals and otolith organs, "
    "as a discrete set of directionally tuned sensors", "discrete", 0, "body",
    extent="both labyrinths"))

BODY_SURFACE = S(Support("body_surface", "the skin, where mechano-, thermo- and nociceptors "
    "sit.  innervation density varies by two orders of magnitude across it", "surface", 2,
    "body", extent="whole skin", min_spacing_mm=0.1))

CHEMOSENSORY_EPITHELIUM = S(Support("chemosensory_epithelium", "olfactory epithelium and taste "
    "buds.  its natural coordinate is receptor identity rather than position", "discrete", 0,
    "body", extent="nasal and oral"))

# -- effectors --------------------------------------------------------------
MOTOR_UNITS = S(Support("motor_units", "motor units as a discrete set, grouped by muscle. "
    "recruitment is ordered by size, so the natural index is threshold rather than position",
    "discrete", 0, "body", extent="skeletal musculature"))

VOCAL_TRACT = S(Support("vocal_tract", "articulators and the airway from glottis to lips",
    "manifold", 1, "body", extent="vocal tract", min_spacing_mm=0.5))

# -- instruments ------------------------------------------------------------
SENSOR_ARRAY = S(Support("sensor_array", "non-invasive sensor elements: EEG electrodes, MEG "
    "gradiometers and magnetometers, NIRS optodes", "discrete", 0, "eeg_cap",
    extent="sensor montage"))

IMPLANTED_ARRAY = S(Support("implanted_array", "invasive contacts: subdural grids and strips, "
    "depth electrodes, intracortical arrays", "discrete", 0, "electrode_grid",
    extent="implanted contacts"))

STIMULATOR = S(Support("stimulator", "stimulation devices: TMS coils, tES electrodes, "
    "ultrasound transducers, DBS leads", "discrete", 0, "coil", extent="stimulator elements"))

SCANNER_ELEMENT = S(Support("scanner_element", "MR receive coils, gradient channels and the "
    "pulse sequence as a state-bearing device", "discrete", 0, "scanner", extent="scanner"))

DISPLAY = S(Support("display", "the stimulus display and speakers -- where an experiment's "
    "imposed variables physically live before they reach a receptor surface", "surface", 2,
    "display", extent="stimulus apparatus"))

__all__ = [n for n in dir() if n.isupper() and not n.startswith("_")]
