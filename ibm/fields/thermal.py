"""temperature of the head volume.

one component, scalar, and the smallest field in the ontology.  it is scalar
because temperature in tissue is the output of a diffusion equation against a
perfusion sink, and diffusion is a low-pass with a single dominant time constant
of tens of seconds to minutes -- there is no second timescale for the first to
interact with, which is the architecture's criterion, and a spectral belief here
would be a spectrum with one non-zero bin.

one component rather than several because the alternatives are not temperature.
metabolic heat production is a source and lives in the metabolic field; thermal
conductivity and specific heat are material properties and live there; perfusion
cooling is `blood.flow` read by the thermal-diffusion process.  what is left is
the scalar field those three act on, and splitting it further would be splitting a
quantity that has one value at a point.

the field exists for two reasons that have nothing to do with each other.  the
physiological one: brain temperature is not core temperature, it runs a few tenths
of a degree above it because local metabolism outpaces local perfusion, that
offset varies across the brain and with activity, and reaction rates -- channel
kinetics, transmitter clearance, enzyme activity -- all depend on it, so a model
that fixes temperature has quietly fixed every rate constant it owns.  the safety
one: every stimulation modality in the inventory deposits energy.  MR gradient and
RF heating, focused ultrasound absorption, tES current through the scalp and the
power dissipated by an implanted device all raise this variable, and the limits
those modalities are operated under are written directly in it.  a model that can
predict a response to stimulation but not the heating that accompanies it can only
answer half the question anyone asks of it.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Provenance, ULTRASLOW

FIELD = REGISTRY.field(Field(
    "thermal",
    "the temperature of head tissue: the state that metabolic heat, perfusion cooling, "
    "conduction to the scalp and every form of deposited stimulation energy all act on, "
    "and that every reaction rate in the model is implicitly a function of",
    "head_volume", Provenance.PHYSICS))

TEMPERATURE = REGISTRY.component(Component(
    id="thermal.temperature",
    field="thermal",
    doc="tissue temperature.  it is a state variable rather than a constant because it is "
        "not uniform and not fixed: brain runs 0.2-0.6 K above core temperature at rest, "
        "grey matter runs warmer than white, and both move with activity, with anaesthesia, "
        "and with anything that deposits energy.  the band is below a tenth of a hertz "
        "because tissue thermal diffusion and perfusion washout are both slow, so even a "
        "millisecond ultrasound burst appears here as an integrated rise rather than as a "
        "transient -- which is the correct physics and also the reason the safety limits "
        "are written as time-averaged quantities",
    units="degC",
    uncertainty="scalar",
    band=ULTRASLOW,
    prior="thermal_dc",
    bounds=(20.0, 50.0),
    timescale_s=60.0,
    provenance=Provenance.LITERATURE,
    tags=frozenset({"safety", "rate_modifier"})))
