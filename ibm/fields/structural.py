"""slowly varying tissue structure: what the anatomy is, as state rather than atlas.

scalar, and here the architecture's rule is not even close.  every component in
this field is constant over an experiment -- the band is `STRUCTURAL`, below a
millihertz -- so a belief about it is a belief about a *number*, not about a
trajectory.  the spectral form exists to represent several timescales interacting
inside one variable; these have one timescale, measured in days.

the harder question is why they are state at all rather than parameters of the
processes that use them.  the answer is plasticity.  a synaptic density that is a
process parameter can only ever be fitted; one that is a component can be *written*
by a process, which is exactly what the architecture's plasticity entry does -- it
takes activity history and modulatory state and writes structural state and other
processes' theta.  declaring structure as state is what makes a model of learning
and a model of atrophy the same kind of object as a model of an evoked response,
differing only in timescale.  it also means an atlas is evidence about a state
variable rather than a hard-coded constant, so a subject's own dMRI can move it.

the components divide by what a measurement can separate, which is coarser than
what histology can.  fibre orientation and axonal density are both derived from
diffusion imaging but are not the same claim -- orientation survives demyelination
and density does not.  myelination is separated from axonal density because
myelin, not axon count, sets conduction velocity and therefore every tract delay
in the model, and because T1w/T2w ratio and MT saturation measure it specifically.
synaptic and dendritic density are separated because SV2A PET measures the first
and only histology or high-resolution diffusion models reach the second.  gliosis
is here because it is what changes in disease and after injury while all of the
above stay put, and it changes conductivity and diffusivity in ways nothing else
in the inventory would otherwise explain.

the last four components are vessel geometry and they break this field's one
apparent rule -- that its support is `tissue` -- on purpose.  a lumen radius, a
segment length and a branch order are indexed by position on the VASCULAR TREE,
so they carry a `support` override, which is the mechanism §1 provides exactly
for this: fields do not share a support, and a component may say which one it
means.  they are here rather than in `blood` because the criterion this field is
organised by is timescale, and a vessel's resting calibre moves over days while
`blood` is what moves over seconds.  the fourth, capillary density, stays on
`tissue`, because a length of capillary per cubic millimetre is a property of the
tissue rather than of any one vessel -- and it is the only one of the four that a
living human can be measured for.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Provenance, STRUCTURAL

FIELD = REGISTRY.field(Field(
    "structural",
    "tissue microstructure as state: fibre geometry, axonal and synaptic content, myelin "
    "and glial composition -- constant over an experiment, written by plasticity and "
    "disease over days, and the substrate every conduction delay and every material "
    "property is a function of",
    "tissue", Provenance.ATLAS))

#: days.  a structural component is not slow dynamics; it is a number with a
#: posterior, and `STRUCTURAL` says so in the band.
DAY = 86400.0


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "scalar")
    kw.setdefault("provenance", Provenance.ATLAS)
    kw.setdefault("band", STRUCTURAL)
    kw.setdefault("prior", "structural_dc")
    kw.setdefault("timescale_s", DAY)
    return REGISTRY.component(Component(id=id_, field="structural", doc=doc, units=units, **kw))


FIBER_ORIENTATION = _c(
    "structural.fiber_orientation",
    "the local principal fibre direction, as a unit vector in the subject frame.  it is the "
    "one component in this field that nothing in the process inventory writes -- axons do "
    "not change direction on any timescale ibm-1 models -- so it is exogenous, supplied by "
    "diffusion imaging or an atlas.  it earns its place because conductivity in white "
    "matter is anisotropic along it, which makes the electromagnetic forward problem "
    "depend on it, and because tract geometry and therefore conduction delay is built from "
    "it",
    "dimensionless", bounds=(-1.0, 1.0), exogenous=True,
    tags=frozenset({"geometry", "anisotropy"}))

AXONAL_DENSITY = _c(
    "structural.axonal_density",
    "the fraction of tissue volume occupied by axons at a position.  it sets how many "
    "fibres a tract actually carries and therefore the gain of long-range coupling, and it "
    "is distinct from orientation because a bundle can lose most of its axons without "
    "changing direction -- which is what degeneration looks like on diffusion imaging and "
    "why orientation-only measures miss it",
    "dimensionless", bounds=(0.0, 1.0),
    tags=frozenset({"microstructure"}))

MYELINATION = _c(
    "structural.myelination",
    "myelin volume fraction.  separated from axonal density because myelin and not axon "
    "count sets conduction velocity, so this component is what every tract delay in the "
    "model is a function of -- and delays are phase ramps in the spectral form, which "
    "makes this a structural variable with a direct effect on the frequency at which "
    "long-range synchrony is possible.  it is also the thing myelin-sensitive MR contrasts "
    "measure and the thing demyelinating disease removes",
    "dimensionless", bounds=(0.0, 1.0),
    tags=frozenset({"microstructure", "conduction"}))

SYNAPTIC_DENSITY = _c(
    "structural.synaptic_density",
    "synapses per unit tissue volume.  the component plasticity writes: a change here is a "
    "change in the gain of a local or long-range coupling, expressed once in the ontology "
    "rather than as a per-process weight, so that structural plasticity and synaptic "
    "plasticity are the same statement at different timescales.  SV2A PET binds to it "
    "directly, which makes it one of very few microstructural variables with a live human "
    "measurement",
    "1/um^3", bounds=(0.0, 5.0),
    tags=frozenset({"microstructure", "plasticity", "observable"}))

DENDRITIC_DENSITY = _c(
    "structural.dendritic_density",
    "dendritic length per unit tissue volume.  it is the geometric factor that turns a "
    "population's transmembrane currents into a dipole -- a dense, radially oriented "
    "dendritic field produces an extracellular signal that a tangled one of the same "
    "synaptic density does not -- so it is what the map from `neural.transmembrane_current` "
    "to the electromagnetic field depends on, and the reason cortex is visible to EEG "
    "while most subcortical structures are not",
    "mm/mm^3", bounds=(0.0, 1e5),
    tags=frozenset({"microstructure", "electromagnetic"}))

GLIOSIS = _c(
    "structural.gliosis",
    "reactive glial content: the fraction of tissue given over to activated astrocytes and "
    "microglia.  it is declared because it moves when nothing else in this field does -- "
    "after injury, in epileptic tissue, around an implanted electrode, in "
    "neurodegeneration -- and because it changes the quantities other fields read: it "
    "lowers extracellular volume fraction, raises tortuosity, alters conductivity, and is "
    "the mechanism behind the impedance rise at a chronic implant",
    "dimensionless", bounds=(0.0, 1.0), exogenous=True,
    tags=frozenset({"microstructure", "pathology"}))
# exogenous because its drivers are injury, inflammation and foreign-body response, and
# ibm-1 declares no process for any of them.  plasticity writes the four activity-dependent
# structural components and stops there, which is the honest boundary: activity remodels
# synapses, spines, axons and myelin, and it does not cause astrogliosis.  a materialization
# that needs gliosis -- a chronic implant, an epileptic focus, a lesion -- supplies it as an
# imposed variable from imaging or histology, exactly as it supplies fiber_orientation.
# declaring a process here to close the hole would be inventing pathology dynamics the
# science does not give us.


# ---------------------------------------------------------------------------
# vessel geometry: the same field, a different support
#
# gap #7 of the binding work: `aneurisk` distributes centreline graphs carrying
# radius as a function of arc length -- a tube-chain representation in all but
# name -- and lands nowhere, because the registry had no component for a lumen.
# `blood.volume` is millilitres of blood per hundred grams of TISSUE and
# `blood.flow` is a perfusion; neither is a calibre, and a radius pushed into
# either would be an encoding rather than a physical quantity, which §1 forbids
# outright.
#
# so the four below are declared here rather than in `blood`, and the criterion
# is the one this field is already organised by: TIMESCALE.  `blood` is state
# that moves within an experiment -- flow, pressure, saturation, the balloon
# emptying over seconds.  a vessel's resting calibre, the length of a segment,
# where it sits in the branching order and how much capillary a cubic millimetre
# holds move over DAYS: angiogenesis, remodelling, rarefaction, the capillary
# loss of aging microangiopathy.  putting them in `blood` would give a lumen
# radius the haemodynamic band and invite a fit to move it at 0.1 Hz, which is
# the one thing the geometry must not do -- a vasodilation is a change in
# resistance driven by `neurovascular_coupling`, and it is expressed as flow and
# volume, not as a rewritten anatomy.
#
# three of them carry `support="vascular_tree"`, overriding the field's `tissue`.
# ARCHITECTURE.md §1 is the whole reason that override exists: a lumen radius is
# indexed by position ON THE TREE, and two capillaries a hundred microns apart in
# space can hang off different penetrating arterioles.  the fourth,
# `capillary_density`, is deliberately left on `tissue`, because a length of
# capillary per unit tissue volume is a property of the TISSUE and not of any one
# vessel -- it is what survives when the individual segments are averaged away,
# and it is the only one of the four an in-vivo human measurement can reach.
#
# two things a caller might expect here and will not find, both refused on
# purpose.
#
# FLOW DIRECTION is not declared.  a direction is a property of an EDGE, not of a
# position, and `ibm/topologies/vascular.py` already carries it as the orientation
# of the segment plus an optional `flow_sign` column.  a per-node "direction"
# component would be a second, unreconcilable copy of the same fact, and the two
# would disagree the moment a fit moved one.
#
# ARTERIAL TERRITORY is not declared either.  a territory is a COVER over the
# brain organised by blood supply -- §2's object, not §1's -- and no registered
# component is a region label.  it belongs in `ibm/anatomy` alongside the other
# partitioning systems, which is where `arterial-territory-atlas-liu2023` and
# `high-resolution-vascular-atlases` already say it goes.
# ---------------------------------------------------------------------------

LUMEN_RADIUS = _c(
    "structural.lumen_radius",
    "inner radius of the vessel lumen at a position on the vascular tree.  the component "
    "that makes a tube chain expressible: a centreline plus this is an oriented tube, and "
    "an oriented tube is what a segmented angiogram, a VMTK centreline and a light-sheet "
    "skeletonisation all actually produce.  it earns its place by the fourth power -- "
    "Poiseuille resistance goes as r^-4, so a 10% error here is a 46% error in the segment's "
    "resistance and the capillary bed is where most of the resistance lives.  it is the "
    "RESTING calibre and not the instantaneous one: a vasodilation is a change in resistance "
    "driven by neurovascular coupling and is carried by `blood.flow` and `blood.volume` over "
    "seconds, while this moves over days, as angiogenesis, remodelling and rarefaction",
    "mm", bounds=(0.0, 10.0), exogenous=True, support="vascular_tree",
    tags=frozenset({"geometry", "vascular"}))

SEGMENT_LENGTH = _c(
    "structural.segment_length",
    "arc length of the vessel segment a node on the tree stands for.  declared rather than "
    "computed from the positions of adjacent nodes because the two are not the same number: "
    "a capillary segment wanders, and its arc length exceeds the chord between its endpoints "
    "by a tortuosity that is itself a measured quantity.  taking the chord instead "
    "systematically shortens every path, which lowers resistance, raises predicted flow and "
    "shortens transit time -- three errors all in the same direction.  it is also what "
    "`capillary_tissue_exchange` needs to turn a radius into an exchange surface 2 pi r L, "
    "and the reason that builder already looks for a `segment_length_mm` column",
    "mm", bounds=(0.0, 500.0), exogenous=True, support="vascular_tree",
    tags=frozenset({"geometry", "vascular"}))

BRANCH_ORDER = _c(
    "structural.branch_order",
    "generation of the segment counted from the pial surface inward: 0 on a pial vessel, "
    "rising through the penetrating arteriole into the capillary bed and falling again "
    "through the ascending venule.  an ordinal quantity and not a label -- which is the "
    "reason it is a component at all, where arterial territory is not -- and the variable "
    "every vascular statistic is actually organised by.  Murray's law is a statement about "
    "what happens at one step of it; the pressure drop is distributed over it, with most of "
    "the arterial-side drop across two or three generations of precapillary arteriole; and "
    "which segments may exchange with tissue is decided by it far more sharply than by "
    "euclidean position.  a materialization that carries a tree without it can still compute "
    "transport, but cannot say where in the tree anything happened",
    "dimensionless", bounds=(0.0, 64.0), exogenous=True, support="vascular_tree",
    tags=frozenset({"geometry", "vascular", "topology"}))

CAPILLARY_DENSITY = _c(
    "structural.capillary_density",
    "length of capillary per unit tissue volume at a position in the parenchyma.  the one "
    "component in this group that stays on `tissue`, and the one that generalises: individual "
    "branching topology is not recoverable in a living human and mostly does not need to be, "
    "because oxygen delivery, BOLD contrast and thermal transport all depend on TRANSPORT "
    "STATISTICS -- surface area per volume, diffusion distance, transit-time distribution -- "
    "rather than on which capillary is where.  this is the scalar those statistics reduce to, "
    "it sets the intercapillary distance as roughly 1/sqrt(density), and it is the quantity "
    "stereology, two-photon microscopy and ex-vivo network reconstructions all report.  it is "
    "declared exogenous for the same reason `gliosis` is: ibm-1 registers no angiogenesis "
    "process, and inventing one to close the hole would be inventing vascular growth dynamics "
    "the science does not give us",
    "mm/mm^3", bounds=(0.0, 2000.0), exogenous=True,
    tags=frozenset({"geometry", "vascular", "microstructure"}))
