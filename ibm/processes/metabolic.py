"""metabolism: what the tissue spends, what it spends it on, and the heat that
falls out of the accounting.

almost all of a brain's energy budget is signalling, and almost all of the
signalling budget is restoring ionic gradients that synaptic and action
currents just dissipated.  so the honest input to this process is not "activity"
in the abstract but the synaptic conductances and the extracellular potassium
that record the pumping still owed -- which is also why the metabolic demand of
an inhibitory population is real rather than negative.

thermodynamically the output side is almost trivial and worth stating plainly:
the brain does no external work, so on any timescale longer than the atp pool
turnover every joule of substrate oxidised leaves as heat.  `metabolic.heat` is
therefore not an independent quantity to be fitted separately from
`metabolic.consumption`; it is the same quantity in different units, and the
only parameter between them is an enthalpy.  `thermal_diffusion` reads it.

this process reads spectral neural and ionic state and writes scalar metabolic
state, so the registered ``spectral -> scalar`` conversion applies.  it keeps
the window mean and discards the temporal structure -- which for a ten-second
process is defensible, and is still the place where the fact that bursty and
tonic activity of equal mean rate have different metabolic costs stops being
representable.
"""

from __future__ import annotations

from ibm.processes.base import implementation, leaky_integrator, low_pass, process, series
from ibm.registry import Form
from ibm.vocabulary import (
    Band,
    HEMODYNAMIC,
    LFP,
    Provenance,
    Tying,
    Validity,
    lognormal,
    normal,
    speculative,
    weak,
    within,
)

#: metabolic demand is read across the full electrophysiological band before it
#: is collapsed onto the scalar metabolic variables.  pump load is set by charge
#: moved, and charge moved lives up in gamma and above.
DEMAND_BAND = Band(0.0, 300.0)


def cmro2_transfer(basis, tau_atp_s: float = 3.0, tau_substrate_s: float = 12.0):
    """demand -> consumption as two lags in series.

    the fast lag is the phosphocreatine-buffered atp pool, which absorbs the
    first seconds of a demand step without the substrate flux changing at all;
    the slow one is the oxidative response that eventually pays for it.  the
    buffer is why cmro2 measured over one second and over one minute are
    different numbers.
    """
    return series(low_pass(basis, tau_atp_s), low_pass(basis, tau_substrate_s))


def heat_accumulation_transfer(basis, tau_s: float = 30.0):
    """consumption -> heat.  an integrator behind a lag, because heat is the
    time integral of dissipated power and the tissue's own heat capacity is the
    lag.  declared separately from the thermal process so that the enthalpy
    conversion and the diffusion stay in different files."""
    return series(low_pass(basis, tau_s), leaky_integrator(basis, tau_s))


METABOLISM = process(
    id="metabolism",
    doc="""substrate consumption, energy state, lactate and heat, driven by the
    ionic work that synaptic and spiking activity leave behind.

    inputs are chosen to be the things that actually cost atp: synaptic
    conductances (glutamate release, receptor currents, and the vesicle cycle),
    transmembrane current, and extracellular potassium as the running record of
    Na/K-ATPase load.  firing rate enters through the neural activity selectors
    but is not the primary driver, which is the correct emphasis -- postsynaptic
    currents dominate the budget, and dendrites cost more than axons.

    where this breaks: it is a bulk-tissue budget with no cell types in it.
    astrocytes and neurons have different substrate preferences and the lactate
    shuttle between them is genuinely contested, so the lactate terms below
    carry speculative priors while the oxygen and glucose terms carry literature
    ones.  it is also a normoxic, normoglycaemic form: under ischaemia the
    substrate terms stop being first-order in availability, anaerobic glycolysis
    takes over, and the atp balance collapses on a timescale this process's ten
    seconds cannot resolve.""",
    inputs=(
        within("neural", "exc.ampa", "exc.nmda", "inh.gaba_a", "inh.gaba_b", band=DEMAND_BAND),
        within("neural", "exc.activity", "inh.activity", band=DEMAND_BAND),
        within("neural", "transmembrane_current", band=DEMAND_BAND),
        within("extracellular", "k", "na", band=DEMAND_BAND),
        within("metabolic", "oxygen", "glucose", "lactate", band=HEMODYNAMIC),
        within("structural", "synaptic_density", band=Band(0.0, 0.001)),
    ),
    outputs=(
        within("metabolic", "atp", "consumption", "lactate", band=HEMODYNAMIC),
        within("metabolic", "heat", band=HEMODYNAMIC),
    ),
    topology="local",
    timescale_s=10.0,
    validity=Validity(
        min_spacing_mm=0.05, max_spacing_mm=10.0, band=HEMODYNAMIC,
        note="a bulk tissue energy budget.  below ~50 um the compartment is a single cell "
             "and neuron/astrocyte separation stops being optional; above ~10 mm grey and "
             "white matter are averaged together, and their cmro2 differs by a factor of three."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"metabolic", "physiological"}),
    notes="spectral neural and ionic inputs, scalar metabolic outputs: the registered "
          "spectral -> scalar conversion applies and appears in provenance.",
)

implementation(
    name="mass_action_budget",
    process="metabolism",
    doc="""an explicit atp budget: pump load in, oxidative and glycolytic supply
    out, energy state as the balance.

    the form to select when the question is about energy failure, because it has
    a state variable that can actually fall.  supply is michaelis-menten in
    oxygen and glucose, demand is proportional to charge moved, and the
    difference accumulates in the atp pool.  nonlinear, evaluated in time, and
    it does not preserve gaussianity -- the projection back onto the scalar form
    is by moment matching and that is where the tail of the distribution, which
    is the part that matters for failure, is thrown away.""",
    form=Form.RATE,
    params={
        "cmro2_baseline": normal(3.4, 0.5, units="mL O2/100g/min",
                                 provenance=Provenance.LITERATURE,
                                 source="pet and arteriovenous difference, resting grey matter",
                                 note="whole-brain average is nearer 3.0; white matter about a third "
                                      "of the grey value"),
        "cmrglc_baseline": normal(0.31, 0.06, units="umol/g/min",
                                  provenance=Provenance.LITERATURE,
                                  source="fdg-pet resting grey matter, ~5.6 mg/100g/min"),
        "oxygen_glucose_index": normal(5.5, 0.4, units="dimensionless",
                                       provenance=Provenance.LITERATURE,
                                       note="below the stoichiometric 6 even at rest; falls further "
                                            "during activation, which is the aerobic glycolysis "
                                            "that the lactate terms are trying to describe"),
        "atp_per_action_potential": lognormal(3.8e8, 3.0, units="molecules",
                                              provenance=Provenance.LITERATURE,
                                              source="attwell & laughlin 2001 energy budget",
                                              note="rodent cortex; the human per-neuron figure is "
                                                   "extrapolated and the spread reflects that"),
        "signalling_fraction": normal(0.80, 0.07, units="dimensionless",
                                      provenance=Provenance.LITERATURE,
                                      note="fraction of the budget spent on signalling rather than "
                                           "housekeeping; the remainder is the floor a silenced "
                                           "cortex still costs"),
        "km_oxygen_mm": lognormal(0.01, 3.0, units="mM", provenance=Provenance.LITERATURE,
                                  note="cytochrome oxidase half-saturation is very low, which is why "
                                       "oxidative rate is flat until it suddenly is not"),
        "km_glucose_mm": lognormal(0.5, 2.0, units="mM", provenance=Provenance.LITERATURE),
        "tau_atp_pool_s": lognormal(3.0, 2.0, units="s", provenance=Provenance.LITERATURE,
                                    note="phosphocreatine buffer; holds atp nearly constant through "
                                         "short demand transients"),
        "lactate_shuttle_gain": speculative(0.3, 10.0, units="dimensionless",
                                            note="the astrocyte-neuron lactate shuttle is contested "
                                                 "in direction as well as magnitude"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="attwell & laughlin 2001; sokoloff; harris et al 2012",
)

implementation(
    name="linear_demand_lti",
    process="metabolism",
    doc="""consumption as a two-lag filter of demand, with heat proportional to it.

    linear, exact at any timestep, and adequate whenever substrate is not
    limiting -- which is to say for every healthy-brain materialization, where
    cmro2 tracks demand and the atp pool never actually moves.  it is precisely
    wrong for the case the nonlinear form exists for: it will happily report
    negative-going atp with no consequence, because in a linear budget nothing
    saturates and nothing fails.""",
    form=Form.LTI,
    transfer=cmro2_transfer,
    params={
        "tau_atp_s": lognormal(3.0, 2.0, units="s", provenance=Provenance.LITERATURE),
        "tau_substrate_s": lognormal(12.0, 2.0, units="s", provenance=Provenance.LITERATURE,
                                     note="oxidative response to a sustained demand step"),
        "demand_gain": weak(1.0, 5.0, units="umol ATP per unit synaptic drive",
                            note="absorbs every unit convention on the neural side; not separately "
                                 "identifiable from neurovascular efficacy without calibrated data"),
        "heat_per_atp_kj": normal(50.0, 8.0, units="kJ/mol",
                                  provenance=Provenance.LITERATURE,
                                  note="in-vivo free energy of atp hydrolysis; the brain does no "
                                       "external work, so on this timescale it is all heat"),
        "volumetric_heat_baseline": normal(10437.0, 2000.0, units="W/m^3",
                                           provenance=Provenance.LITERATURE,
                                           source="pennes-model brain metabolic heat, the value "
                                                  "used throughout the rf and ultrasound safety "
                                                  "literature",
                                           note="about 11 mW/g; roughly ten times resting muscle"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
)

implementation(
    name="learned_energy_budget",
    process="metabolism",
    doc="""a learned map from the retained neural spectrum to substrate flux.

    worth having because the fixed budget above assumes the cost per unit
    activity is a constant, and it is not: cost per spike varies with firing
    regime, with the excitation-inhibition balance, and with how synchronous the
    input is.  a learned f can carry that dependence on the shape of the
    spectrum, which the linear demand gain structurally cannot.  the prior mean
    is the literature budget, so with no data it reproduces it.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "residual_gain": weak(0.25, 5.0, note="learned term is a residual on the analytic budget"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)

implementation(
    name="heat_accumulation_lti",
    process="metabolism",
    doc="""the consumption-to-heat leg on its own, as an integrator behind a lag.

    separated from the budget implementations so that a thermal-safety
    materialization can select a cheap heat source without also committing to a
    particular substrate model, and so that the enthalpy constant appears in
    exactly one place.""",
    form=Form.LTI,
    transfer=heat_accumulation_transfer,
    params={
        "tau_s": lognormal(30.0, 2.0, units="s", provenance=Provenance.PHYSICS,
                           note="tissue heat capacity against local dissipation"),
    },
    tying=Tying.GLOBAL,
    provenance=Provenance.PHYSICS,
)

__all__ = ["METABOLISM"]
