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
