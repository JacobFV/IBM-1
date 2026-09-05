"""the interstitial compartment: ions, transmitters, modulators and the space itself.

this field is where the two uncertainty forms sit next to each other, and the
split is not a stylistic one.  extracellular potassium after a burst of firing has
a fast rise set by release and diffusion out of the synaptic cleft and a slow
decay set by glial siphoning and the Na/K pump, and the two are not separable
into different variables -- the same [K+] is doing both, and the interaction
between them is what turns a local excess into spreading depolarization instead
of a transient.  that is the architecture's criterion for spectral verbatim, so
the four ions are spectral and carry `ionic_transient`.

everything else here is scalar, and the borderline cases are worth naming rather
than hiding.  glutamate and gaba have millisecond dynamics, so a first reading
says they should be spectral too.  they are not, because the fast component of
transmitter action is already carried, spectrally, by the receptor conductances
in the neural field: `neural.exc.ampa` and `neural.inh.gaba_a` *are* the cleft
transient seen from the postsynaptic side.  what these components add is the
ambient extrasynaptic pool -- the concentration that escapes the cleft, spills
onto neighbouring synapses and tonically occupies extrasynaptic nmda and gaba-a
receptors.  that pool is set by transporter density against volume-averaged
release, has one timescale, and would be double counting if it were also spectral.

the neuromodulators are scalar for a plainer reason: volume transmission.
dopamine, serotonin, acetylcholine and noradrenaline are released from varicosities
without a postsynaptic specialization and act on metabotropic receptors with
second-order latencies, so the concentration a target cell sees is already a
spatial and temporal average.  the fast structure exists in the source neuron's
firing, which is spectral in the neural field, and reaches here smoothed.

pH, osmolarity and volume fraction describe the medium rather than a solute in it.
they are last because they are the ones every other component in this field
implicitly depends on: a diffusion coefficient divided by tortuosity squared, a
concentration measured per unit extracellular volume, and a driving force that is
a log ratio.  `extracellular.volume_fraction` is the dynamic partner of
`material.porosity` -- the material component is the exogenous baseline geometry,
this one is the state that cell swelling actually moves.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Band, Provenance

FIELD = REGISTRY.field(Field(
    "extracellular",
    "the interstitial compartment as a chemical medium: ionic composition, synaptically "
    "released transmitters, volume-transmitted modulators, and the geometry of the space "
    "itself",
    "interstitial", Provenance.LITERATURE))

#: ions live from DC to tens of hertz.  the fast edge is set by cleft clearance
#: and the slow edge by glial buffering, and both must fit in one band.
IONIC = Band(0.0, 50.0)
#: transmitter pools: the ambient level, not the cleft transient.
POOL = Band(0.0, 20.0)
#: volume transmission is intrinsically slow; nothing above a hertz survives it.
MODULATORY = Band(0.0, 2.0)
MEDIUM = Band(0.0, 1.0)


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "scalar")
    kw.setdefault("provenance", Provenance.LITERATURE)
    return REGISTRY.component(
        Component(id=id_, field="extracellular", doc=doc, units=units, **kw))


# -- ions --------------------------------------------------------------------

K = _c(
    "extracellular.k",
    "interstitial potassium concentration.  the single most consequential ion in the "
    "inventory: the space is small enough that the potassium leaving cells during one "
    "second of vigorous firing is a large fraction of what is already there, so [K+]o "
    "rises from ~3 to ~10 mM and the potassium reversal potential -- and with it every "
    "cell's excitability -- moves with it.  it is spectral because its rise and its "
    "clearance are separate mechanisms on separate timescales acting on the same number, "
    "and whether an excess self-limits or runs away is decided by their ratio",
    "mM", uncertainty="spectral", band=IONIC, prior="ionic_transient",
    bounds=(2.0, 15.0), timescale_s=0.5,
    tags=frozenset({"ion", "excitability"}))

NA = _c(
    "extracellular.na",
    "interstitial sodium concentration.  it moves far less than potassium in relative "
    "terms -- 145 mM with excursions of a few millimolar -- but it is declared as state "
    "rather than a constant because those few millimolar are the sodium that entered "
    "cells, and the pump that puts it back is the largest single consumer of brain atp.  "
    "spectral for the same two-pole reason as potassium, with a longer slow pole",
    "mM", uncertainty="spectral", band=IONIC, prior="ionic_transient",
    bounds=(100.0, 165.0), timescale_s=2.0,
    tags=frozenset({"ion"}))

CA = _c(
    "extracellular.ca",
    "interstitial free calcium.  the smallest pool in the field and therefore the most "
    "easily depleted: sustained synaptic activity drops [Ca2+]o from ~1.2 to below 0.8 mM, "
    "which lowers release probability at every synapse in the neighbourhood.  that makes "
    "it a genuine negative-feedback state variable rather than a bath constant, and its "
    "depletion and recovery run on different timescales, so it is spectral",
    "mM", uncertainty="spectral", band=IONIC, prior="ionic_transient",
    bounds=(0.2, 2.0), timescale_s=1.0,
    tags=frozenset({"ion", "release"}))

CL = _c(
    "extracellular.cl",
    "interstitial chloride.  it matters because it sets the gaba-a reversal potential "
    "together with the intracellular concentration, which is the hinge on which inhibition "
    "is hyperpolarizing or depolarizing -- the difference between an immature, injured or "
    "seizing network and a healthy one.  spectral because chloride loading during intense "
    "inhibition is fast and its extrusion by KCC2 is slow",
    "mM", uncertainty="spectral", band=IONIC, prior="ionic_transient",
    bounds=(80.0, 145.0), timescale_s=3.0,
    tags=frozenset({"ion", "inhibition"}))

PH = _c(
    "extracellular.ph",
    "interstitial pH.  scalar rather than spectral despite having a fast alkaline transient "
    "and a slow acid shift, because the alkaline transient is a few hundredths of a unit "
    "and the acid shift that follows sustained activity is tenths -- one of the two "
    "dominates so completely that carrying both costs a spectrum to represent rounding "
    "error.  it is here at all because extracellular acidification gates ASIC channels, "
    "modulates nmda receptors, and is the strongest vasodilatory signal after CO2",
    "pH", band=MEDIUM, prior="concentration", bounds=(6.2, 7.8), timescale_s=10.0,
    tags=frozenset({"medium"}))

# -- fast transmitters, as ambient pools -------------------------------------

GLUTAMATE = _c(
    "extracellular.glutamate",
    "ambient extrasynaptic glutamate: the concentration outside the cleft, held near a "
    "micromolar by EAAT transporters on astrocytes.  deliberately not the cleft transient, "
    "which is already carried by the postsynaptic ampa and nmda conductances -- this is the "
    "spillover pool that tonically occupies extrasynaptic nmda receptors, sets the "
    "excitotoxic threshold, and is what fails when transporters are downregulated",
    "uM", band=POOL, prior="concentration", bounds=(0.0, 500.0), timescale_s=1.0,
    tags=frozenset({"transmitter", "excitatory"}))

GABA = _c(
    "extracellular.gaba",
    "ambient extrasynaptic gaba.  the substrate of tonic inhibition: high-affinity "
    "delta-subunit gaba-a receptors are saturated at concentrations far below what a cleft "
    "reaches, so this pool sets a standing conductance that shifts the whole population's "
    "input-output curve without any spiking inhibitory neuron being involved.  it is the "
    "target of the neurosteroids and of tiagabine, which is why it must be state",
    "uM", band=POOL, prior="concentration", bounds=(0.0, 200.0), timescale_s=2.0,
    tags=frozenset({"transmitter", "inhibitory"}))

# -- neuromodulators ---------------------------------------------------------

DOPAMINE = _c(
    "extracellular.dopamine",
    "interstitial dopamine from midbrain varicosities.  volume-transmitted: released "
    "without a postsynaptic specialization, cleared by DAT over hundreds of milliseconds "
    "to seconds, and acting through G-protein-coupled receptors, so what a target cell "
    "sees is already averaged in space and time.  it is a component rather than a process "
    "parameter because its concentration is a state with its own dynamics -- tonic level "
    "and phasic transient are the same variable at different amplitudes",
    "nM", band=MODULATORY, prior="concentration", bounds=(0.0, 5000.0), timescale_s=1.0,
    tags=frozenset({"modulator"}))

SEROTONIN = _c(
    "extracellular.serotonin",
    "interstitial serotonin from the raphe nuclei.  the slowest and most diffusely "
    "projecting of the modulators, with SERT clearance over seconds and receptor families "
    "that both excite and inhibit the same target depending on subtype.  its level is what "
    "an SSRI moves, which is the only reason a model can connect a chronic pharmacological "
    "intervention to anything measurable",
    "nM", band=MODULATORY, prior="concentration", bounds=(0.0, 1000.0), timescale_s=5.0,
    tags=frozenset({"modulator"}))

ACETYLCHOLINE = _c(
    "extracellular.acetylcholine",
    "interstitial acetylcholine from basal forebrain and brainstem.  it has both a fast "
    "hydrolysed component and a slow ambient one, but the fast component acts on nicotinic "
    "receptors at identified synapses and belongs in the neural field's conductances; what "
    "is carried here is the muscarinic tone that suppresses adaptation, boosts thalamic "
    "transmission and gates cortical plasticity across a whole territory at once",
    "nM", band=MODULATORY, prior="concentration", bounds=(0.0, 5000.0), timescale_s=2.0,
    tags=frozenset({"modulator"}))

NORADRENALINE = _c(
    "extracellular.noradrenaline",
    "interstitial noradrenaline from locus coeruleus.  the modulator most directly tied to "
    "arousal, and the one with the clearest gain-control interpretation: it changes the "
    "slope of the population transfer function rather than its offset.  it also acts on "
    "astrocytes and on smooth muscle, which is why it appears upstream of both the "
    "neuromodulation and the neurovascular processes",
    "nM", band=MODULATORY, prior="concentration", bounds=(0.0, 1000.0), timescale_s=3.0,
    tags=frozenset({"modulator", "arousal"}))

ADENOSINE = _c(
    "extracellular.adenosine",
    "interstitial adenosine: the metabolic byproduct that is also a signal.  it accumulates "
    "in proportion to atp consumption over hours of waking, inhibits transmitter release "
    "through A1 receptors, and dilates arterioles through A2A -- which makes it the single "
    "most direct coupling between the metabolic field and both neural excitability and "
    "blood flow, and the substance caffeine antagonizes",
    "nM", band=MODULATORY, prior="concentration", bounds=(0.0, 20000.0), timescale_s=60.0,
    tags=frozenset({"modulator", "metabolic", "sleep"}))

# -- the medium itself -------------------------------------------------------

OSMOLARITY = _c(
    "extracellular.osmolarity",
    "total interstitial osmolarity.  it is the driving force for water movement between "
    "the intracellular, interstitial and vascular compartments, so it is what connects "
    "ionic redistribution to cell swelling and therefore to the volume fraction below.  "
    "systemic osmolarity is regulated within a couple of percent, which is exactly why "
    "the local excursions that matter are small and need a state variable to be visible",
    "mOsm/L", band=MEDIUM, prior="concentration", bounds=(240.0, 400.0), timescale_s=60.0,
    tags=frozenset({"medium"}))

VOLUME_FRACTION = _c(
    "extracellular.volume_fraction",
    "alpha: the fraction of tissue volume that is extracellular space, nominally ~0.2 and "
    "falling towards 0.05 during ischaemia or spreading depolarization.  every "
    "concentration in this field is per unit of *this* volume, so a shrinking alpha "
    "concentrates all of them at once without any flux -- the mechanism behind the "
    "positive feedback in spreading depolarization.  it is the dynamic partner of "
    "`material.porosity`, which is the exogenous resting geometry; this is the state that "
    "cell swelling moves",
    "dimensionless", band=MEDIUM, prior="concentration", bounds=(0.03, 0.40),
    timescale_s=10.0, tags=frozenset({"medium", "geometry"}))
