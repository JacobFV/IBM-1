"""tissue energetics: substrate availability, energy state, products, and heat.

scalar.  the brain's energy budget is buffered on purpose -- creatine kinase holds
atp within a few percent of 2 mM while consumption doubles, glycogen buffers
glucose, and the whole point of that machinery is to remove fast structure from
these variables.  what remains is a slow envelope, and an envelope has one
timescale.  the fast structure that does exist upstream lives in the neural field,
which is spectral, and reaches this field already integrated: metabolism responds
to how much a population fired over the last second, not to when in the gamma
cycle it fired.

the field is on `tissue` rather than on the vascular tree, and the pairing is the
substance of the neurovascular story.  blood delivers on a tree; tissue consumes
in a volume; the exchange between them across the capillary wall is a process with
its own geometry, and keeping the two supports distinct is what makes oxygen
extraction fraction a derived quantity rather than an assumption.  a model that put
both on one support would have already decided that delivery and demand are
matched, which is precisely the thing BOLD contrast exists because they are not.

six components rather than a single "energy" variable because the failure modes
differ.  hypoxia is oxygen falling with glucose intact; hypoglycaemia is the
reverse; ischaemia is both, and is distinguishable from either by lactate, which
rises when glycolysis outruns oxidative phosphorylation and is therefore the
marker of the mismatch rather than of the shortage.  atp is the variable that
actually gates function, and it is nearly constant right up until it is not --
which is why it is declared with a tight prior and a wide bound.  heat is here and
not in the thermal field for the same reason a source is not a field: the thermal
field carries temperature, and this is what drives it.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Band, HEMODYNAMIC, Provenance

FIELD = REGISTRY.field(Field(
    "metabolic",
    "the energy state of neural tissue: oxygen and glucose availability, the adenylate "
    "pool that gates every pump in the model, the glycolytic product that reports "
    "supply-demand mismatch, and the consumption and heat that couple to blood and "
    "temperature",
    "tissue", Provenance.LITERATURE))

#: metabolism follows an activity envelope, not the activity itself.
SLOW = Band(0.0, 2.0)


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "scalar")
    kw.setdefault("provenance", Provenance.LITERATURE)
    kw.setdefault("band", SLOW)
    kw.setdefault("prior", "metabolic_pool")
    return REGISTRY.component(Component(id=id_, field="metabolic", doc=doc, units=units, **kw))


OXYGEN = _c(
    "metabolic.oxygen",
    "tissue oxygen tension: the partial pressure of oxygen in the parenchyma, ~25 mmHg at "
    "rest against ~95 in arterial blood.  a tension rather than a concentration because "
    "diffusion from capillary to mitochondrion is driven by the gradient in partial "
    "pressure, and because oxygen electrodes and phosphorescence quenching both report "
    "this.  the steepness of that gradient is why oxygen and not glucose is the substrate "
    "that runs out first",
    "mmHg", bounds=(0.0, 160.0), timescale_s=1.0,
    tags=frozenset({"substrate"}))

GLUCOSE = _c(
    "metabolic.glucose",
    "tissue glucose concentration, ~1.2 mM against ~5 mM in plasma.  the gap is the "
    "GLUT1/GLUT3 transport step, and it is the reason brain glucose is a state variable "
    "rather than a plasma reading: transport saturates, so during intense activity local "
    "tissue glucose falls even while blood glucose is normal.  FDG-PET measures its uptake "
    "and is bound to this component",
    "mmol/L", bounds=(0.0, 10.0), timescale_s=10.0,
    tags=frozenset({"substrate", "observable"}))

ATP = _c(
    "metabolic.atp",
    "the tissue adenosine-triphosphate pool: the energy currency every ion pump in the "
    "model spends and therefore the variable that connects the metabolic field back to "
    "excitability.  it is held remarkably flat by the creatine-kinase buffer, so the "
    "informative statement about it is not its mean but the fact that when it does move "
    "the tissue is already failing.  its breakdown product is `extracellular.adenosine`, "
    "which is how energy state reaches both neural gain and vascular tone",
    "mmol/L", bounds=(0.0, 5.0), timescale_s=0.5,
    tags=frozenset({"energy"}))

LACTATE = _c(
    "metabolic.lactate",
    "tissue lactate.  it is the mismatch indicator rather than a waste product: it "
    "accumulates when glycolytic flux exceeds what oxidative phosphorylation can consume, "
    "which happens transiently at the onset of activation even in fully oxygenated tissue.  "
    "it is also a substrate in its own right -- the astrocyte-neuron lactate shuttle makes "
    "it an intercompartmental transfer -- and it is directly observable by MR spectroscopy, "
    "which is rare enough in this field to be worth declaring for",
    "mmol/L", bounds=(0.0, 25.0), timescale_s=30.0,
    tags=frozenset({"product", "observable"}))

CONSUMPTION = _c(
    "metabolic.consumption",
    "the local rate of oxidative metabolism, CMRO2: micromoles of oxygen consumed per unit "
    "tissue mass per minute.  a rate, not a pool, and declared separately from the pools "
    "because it is the quantity calibrated BOLD and PET actually estimate and the quantity "
    "the balloon model needs alongside flow.  the BOLD signal is a function of the "
    "*mismatch* between this and `blood.flow`, so a model that derived one from the other "
    "would predict no contrast at all",
    "umol/100g/min", bounds=(0.0, 600.0), timescale_s=2.0,
    tags=frozenset({"rate", "observable"}))

HEAT = _c(
    "metabolic.heat",
    "metabolic heat production per unit tissue mass, ~10-15 W/kg in grey matter.  it is "
    "declared here rather than in the thermal field because a source is not a field: the "
    "thermal field carries temperature, and this is the term that drives it against "
    "perfusion cooling and conduction to the scalp.  it matters beyond bookkeeping for "
    "safety -- specific absorption rate from MR and from focused ultrasound adds to this "
    "same budget, and the tissue cannot tell the two apart",
    "W/kg", bounds=(0.0, 200.0), timescale_s=1.0,
    tags=frozenset({"rate", "thermal"}))
