"""cerebrospinal and interstitial fluid: the slow water compartment, what it
carries, and where it trades with the tissue it surrounds.

three processes live here and they share one arithmetic fact worth stating
before any of them: the *net* csf flow is tiny and the *oscillatory* csf flow is
not.  production is about 0.35 mL/min against a total volume of about 150 mL, so
the bulk turnover time is roughly six hours; peak aqueductal velocity over a
cardiac cycle is several cm/s, two orders of magnitude above what the net flux
alone would produce.  a model that carries only the mean therefore gets the
transport approximately right and the measurement -- phase-contrast mri, which
sees velocity -- entirely wrong, and one that carries only the pulsation gets
the clearance wrong.  the selectors below keep both by declaring a band that
reaches the cardiac fundamental rather than stopping at the haemodynamic corner,
and the implementations split into a bulk leg and a pulsatile leg so that the
two are separately identifiable.

the second thing to state plainly is the lossy edge.  extracellular ions are
carried in the spectral form; csf and the bulk interstitial variables are
carried in the scalar one.  `csf_interstitial_exchange` and
`interstitial_transport` both read spectral ionic state and write scalar fluid
state, so the registered ``spectral -> scalar`` conversion is inserted, and it
keeps the window mean and its variance and destroys everything else -- the
rhythmicity of potassium transients, their phase relative to the cardiac cycle,
and every cross-frequency relationship.  for a hundred-second transport process
that is mostly the right trade, but it is also exactly why this graph cannot be
asked whether slow-wave-locked potassium efflux drives clearance differently
from the same efflux delivered tonically, which is a live question in the sleep
literature and not a hypothetical one.  the selectors marginalize to the
ultraslow band explicitly, so that at least the collapse is a declared choice
and not an artefact of whatever window the runtime picked.
"""

from __future__ import annotations

import numpy as np

from ibm.processes.base import implementation, process
from ibm.registry import Form
from ibm.vocabulary import (
    Band,
    OnSupport,
    Provenance,
    STRUCTURAL,
    Tying,
    ULTRASLOW,
    Validity,
    lognormal,
    normal,
    speculative,
    uniform,
    weak,
    within,
)

#: the band csf mechanics is read and written over.  it reaches the cardiac
#: fundamental on purpose: the pulsation is the dominant term in csf *velocity*
#: even though it contributes almost nothing to net transport, and cutting the
#: band at the haemodynamic corner would delete the only part of this process an
#: instrument directly measures.
CSF_PULSATION = Band(0.0, 2.0)
#: solute concentration follows bulk turnover, not the pulse that drives it.
SOLUTE_BAND = Band(0.0, 0.1)

#: the band the slow transport processes read ionic and solute state over before
#: the collapse onto scalar fluid variables.  0.1 Hz is ten seconds, already an
#: order of magnitude faster than the transport being described; anything above
#: it would be averaged away by the process itself even if the conversion kept
#: it.
TRANSPORT_BAND = ULTRASLOW


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def davson_pressure_transfer(basis, outflow_resistance: float = 10.0,
                             compliance_ml_mmhg: float = 0.6):
    """csf formation against absorption and craniospinal compliance.

        H = R / (1 + i omega R C)

    davson's equation says that at steady state ``P = I_f * R + P_ss``: a
    constant production current through an outflow resistance, offset by sagittal
    sinus pressure.  making it dynamic costs one capacitor -- the craniospinal
    space stores volume as pressure -- and the result is a first-order lag with
    dc gain R and time constant RC.  with R around 10 mmHg/(mL/min) and C around
    0.6 mL/mmHg that is a six-minute relaxation, which is why an infusion test
    takes as long as it does.

    the compliance is the dishonest term and the honest place to say so: the
    craniospinal pressure-volume curve is exponential, not linear, so C is a
    local slope at the operating pressure and falls sharply as pressure rises.
    this form is therefore valid at normal icp and progressively wrong through
    the range where the clinical question actually lives.
    """
    omega = basis.omega
    tau_s = outflow_resistance * compliance_ml_mmhg * 60.0
    return outflow_resistance / (1.0 + 1j * omega * tau_s)


def pulsatile_velocity_transfer(basis, tau_compliance_s: float = 0.5,
                                gain: float = 1.0):
    """arterial volume pulsation into csf velocity: a high pass.

        H = g i omega tau / (1 + i omega tau)

    the skull is a closed box, so every millilitre the arteries take up in
    systole displaces a millilitre of csf towards the spinal canal and back.
    velocity is therefore the *derivative* of the intracranial volume the blood
    is carrying, which is why the dc term -- the entire net production flow --
    contributes essentially nothing to what a phase-contrast sequence measures.
    the corner is the craniospinal compliance draining that volume, and above it
    the response flattens.

    where it breaks: the box is not perfectly closed and the spinal compartment
    is much more compliant than the cranial one, so the split between what leaves
    through the foramen magnum and what is absorbed locally is posture-dependent
    and this form has one number for it.
    """
    omega = basis.omega
    s = 1j * omega * tau_compliance_s
    return gain * s / (1.0 + s)


def exchange_transfer(basis, exchange_rate_hz: float = 1.0e-3, gain: float = 1.0):
    """two-compartment first-order exchange, H = g k / (k + i omega).

    the simplest thing that can be said about a fluid interface: the flux across
    it is proportional to the concentration difference, so each compartment
    relaxes towards the other with one rate constant.  at k = 1e-3 Hz that is a
    seventeen-minute equilibration, which is the right order for tracer studies
    of csf-to-interstitium transport.

    it is a *mass-transfer coefficient* model, which means the geometry of the
    perivascular annulus, the astrocytic endfoot sheet and the aquaporin
    distribution have all been folded into one number.  that is defensible while
    the question is how much is cleared; it is not defensible when the question
    is where, because the answer to where is entirely geometry.
    """
    omega = basis.omega
    return gain * exchange_rate_hz / (exchange_rate_hz + 1j * omega)


def tortuous_diffusion_transfer(basis, d_free_m2_s: float = 1.24e-9,
                                tortuosity: float = 1.6,
                                wavelength_mm: float = 1.0,
                                clearance_hz: float = 0.0):
    """one spatial eigenmode of hindered diffusion in the extracellular space.

        D* = D / lambda^2,   H = 1 / (D* k^2 + u + i omega),   k = 2 pi / L

    the extracellular space is a connected film about 40 nm wide occupying a
    fifth of the tissue volume, and a molecule crossing it takes a path longer
    than the straight line by the tortuosity factor lambda.  measured lambda in
    grey matter is close to 1.6 and remarkably stable across regions and species,
    so the effective diffusivity is about 40% of the free one -- a real,
    well-measured, single-number correction, which is unusual in this file.

    the uptake term u is where the difference between species lives.  a small
    ion diffuses and is pumped; amyloid-beta diffuses two orders of magnitude
    more slowly and is cleared by routes this term lumps together.  setting u to
    zero turns this into pure diffusion and makes the clearance timescale scale
    as L^2, which for a centimetre is days -- the argument that diffusion alone
    cannot clear the brain, and the reason the advective implementation exists.
    """
    omega = basis.omega
    k = 2.0 * np.pi / max(wavelength_mm * 1e-3, 1e-30)
    d_eff = d_free_m2_s / max(tortuosity, 1e-30) ** 2
    return 1.0 / (d_eff * k ** 2 + clearance_hz + 1j * omega)


def advection_diffusion_transfer(basis, d_free_m2_s: float = 1.24e-9,
                                 tortuosity: float = 1.6,
                                 wavelength_mm: float = 1.0,
                                 velocity_um_s: float = 0.1,
                                 clearance_hz: float = 0.0):
    """hindered diffusion with a bulk drift, as a diffusion mode behind a
    transport delay.

    a mean velocity v over a path L is a pure delay L/v, and the delay is a phase
    ramp; the diffusive spread around it is the mode above.  the peclet number
    ``v L / D*`` is the whole argument: below 1 the transport is diffusive and
    the delay is decorative, above 1 it is advective and the diffusive corner is.
    at v = 0.1 um/s, L = 1 mm and D* = 5e-10 m^2/s the peclet number is about
    0.2, which is the quantitative core of the objection to bulk interstitial
    flow -- and the reason the velocity prior below is speculative rather than
    literature even though the phenomenon it describes is heavily published.
    """
    omega = basis.omega
    tau_s = (wavelength_mm * 1e-3) / max(velocity_um_s * 1e-6, 1e-30)
    diff = tortuous_diffusion_transfer(basis, d_free_m2_s, tortuosity,
                                       wavelength_mm, clearance_hz)
    return diff * np.exp(-1j * omega * tau_s)


# ---------------------------------------------------------------------------
# csf flow
# ---------------------------------------------------------------------------

CSF_FLOW = process(
    id="csf_flow",
    doc="""bulk and pulsatile movement of csf through the ventricles, cisterns
    and subarachnoid space.

    declared over the csf topology rather than a spatial one for the same reason
    vascular flow is declared over the tree: two points a few millimetres apart
    across the ependyma are not connected, and two points at opposite ends of the
    aqueduct are.  csf transport is a plumbing statement, and euclidean
    neighbourhood is the wrong relation for it.

    the driving terms are read rather than assumed.  arterial volume pulsation
    enters through `blood.volume` and `blood.pressure`, which is what makes this
    process's pulsatile leg a consequence of the vascular state instead of an
    imposed sinusoid, and mechanical pressure enters through
    `mechanical.pressure` because the skull is closed and any tissue volume
    change has to be paid for in csf displacement.  respiration is not in the
    ontology as a driver, and its absence is a real gap: inspiration-driven
    upward csf flow is larger than the cardiac contribution to *net* transport,
    and this process cannot currently produce it.

    where it breaks: a lumped compliant-compartment description assumes csf is
    inviscid and the space is a set of well-mixed reservoirs.  it is adequate for
    pressure and for bulk turnover and inadequate for anything about mixing --
    the flow in the aqueduct is oscillatory with a stokes layer, and the
    dispersion it produces is a boundary-layer effect no compartment model
    contains.""",
    inputs=(
        within("csf", "pressure", "velocity", "solute",
               region=OnSupport("csf_space"), band=CSF_PULSATION),
        within("blood", "volume", "pressure", band=CSF_PULSATION),
        within("mechanical", "pressure", band=CSF_PULSATION),
        within("material", "porosity", band=STRUCTURAL),
    ),
    outputs=(
        # pressure and velocity carry the cardiac fundamental: a csf compartment
        # really does pulse at ~1 Hz, and phase-contrast MRI measures that.
        within("csf", "pressure", "velocity",
               region=OnSupport("csf_space"), band=CSF_PULSATION),
        # solute does not.  the flow pulses, the concentration it carries does
        # not -- oscillatory displacement in the aqueduct is a couple of
        # millimetres against a compartment metres of turnover wide, so
        # concentration integrates the pulsation away.  writing solute at the
        # pulsation band would assert a 1 Hz concentration swing no measurement
        # has ever shown and the component's 600 s time constant forbids.
        within("csf", "solute", region=OnSupport("csf_space"), band=SOLUTE_BAND),
    ),
    topology="csf",
    timescale_s=10.0,
    validity=Validity(
        min_spacing_mm=0.5, max_spacing_mm=100.0, band=CSF_PULSATION,
        note="a compartment model of a plumbing system.  below ~0.5 mm the compartment is a "
             "sulcal film thinner than the cell describing it and the well-mixed assumption "
             "is meaningless; the upper bound is deliberately loose because the whole "
             "ventricular system as one compartment is a standard and defensible model."),
    provenance=Provenance.PHYSICS,
    tags=frozenset({"fluid", "transport", "slow"}),
)

implementation(
    name="davson_lumped_lti",
    process="csf_flow",
    doc="""production through an outflow resistance into a compliant space.

    the clinical model: it is what an infusion study fits, what a shunt is
    specified against, and what normal-pressure hydrocephalus is discussed in
    terms of.  linear, so exact at any timestep and cheap enough that a
    materialization has no reason to omit it.

    it holds at normal icp and progressively fails as pressure rises, because
    compliance is the local slope of an exponential pressure-volume curve rather
    than a constant.  it also has no route for the transependymal and lymphatic
    absorption that becomes dominant when arachnoid granulation outflow is
    obstructed, which is precisely the pathology it would be asked about.""",
    form=Form.LTI,
    transfer=davson_pressure_transfer,
    params={
        "production_ml_min": normal(0.35, 0.08, units="mL/min",
                                    provenance=Provenance.LITERATURE,
                                    source="csf formation rate in adult humans, ~500 mL/day",
                                    note="turns the whole 150 mL pool over roughly three times a "
                                         "day; falls with age and with acetazolamide"),
        "outflow_resistance": lognormal(10.0, 1.6, units="mmHg/(mL/min)",
                                        provenance=Provenance.LITERATURE,
                                        source="marmarou infusion studies; 6-12 in healthy adults",
                                        note="rises above ~13 in communicating hydrocephalus, which is "
                                             "the single measurement shunt selection turns on"),
        "compliance_ml_mmhg": lognormal(0.6, 2.0, units="mL/mmHg",
                                        provenance=Provenance.LITERATURE,
                                        source="craniospinal compliance from pressure-volume index "
                                               "~25 mL",
                                        note="a local slope: it is roughly inversely proportional to "
                                             "pressure, so this number is only the resting one"),
        "baseline_pressure_mmhg": normal(10.0, 3.0, units="mmHg",
                                         provenance=Provenance.LITERATURE,
                                         source="supine adult icp 7-15 mmHg",
                                         note="posture-dependent by more than its own range; upright "
                                              "icp is near zero or negative"),
        "total_volume_ml": normal(150.0, 30.0, units="mL",
                                  provenance=Provenance.LITERATURE,
                                  source="total csf volume 125-150 mL, of which ~25 mL ventricular"),
    },
    tying=Tying.GLOBAL,
    provenance=Provenance.LITERATURE,
    source="davson 1970; marmarou et al 1978",
)

implementation(
    name="cardiac_pulsation_lti",
    process="csf_flow",
    doc="""the oscillatory leg: arterial volume pulsation displacing csf, as a
    high pass on intracranial blood volume.

    separated from the bulk leg because the two have different parameters,
    different evidence and different failure modes, and because a phase-contrast
    velocity measurement constrains this one and says almost nothing about the
    other.  the amplitude is set by how much volume the arteries take up per beat
    and by how compliant the spinal compartment is; the phase relative to the
    cardiac cycle is the part that is actually reproducible across subjects.

    it does not represent respiratory driving, and that is the largest single
    omission in this file.""",
    form=Form.LTI,
    transfer=pulsatile_velocity_transfer,
    params={
        "tau_compliance_s": lognormal(0.5, 2.0, units="s", provenance=Provenance.WEAK,
                                      note="corner between the pulsatile and bulk regimes; not "
                                           "separately measured, inferred from waveform shape"),
        "gain": weak(1.0, 5.0, units="cm/s per mL of arterial volume pulse",
                     note="absorbs the geometry of the aqueduct, which is the actual determinant "
                          "and varies several-fold between subjects"),
        "cardiac_hz": normal(1.15, 0.20, units="Hz", provenance=Provenance.LITERATURE,
                             source="resting adult heart rate ~70 bpm",
                             note="carried as a parameter only so the band restriction can be "
                                  "checked against it; the drive itself comes from blood state"),
        "peak_aqueduct_velocity_cm_s": lognormal(4.0, 1.6, units="cm/s",
                                                 provenance=Provenance.LITERATURE,
                                                 source="phase-contrast mri, cerebral aqueduct 3-6 cm/s",
                                                 note="two orders of magnitude above the mean velocity "
                                                      "the 0.35 mL/min production would give"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="phase-contrast mri aqueductal flow measurements",
)

implementation(
    name="darcy_network",
    process="csf_flow",
    doc="""resistive network flow with an exponential pressure-volume wall law.

    the form to select when the question is about the plumbing: an obstruction,
    a stenosed aqueduct, a shunt with a specified opening pressure, or the
    nonlinear rise of icp as compliance is exhausted.  the exponential elastance
    is the whole reason to pay for it -- it is what makes icp go from tolerable
    to catastrophic over a few millilitres, and no linear model can produce that
    knee.

    costs a nonlinear solve per step and does not preserve gaussianity; the
    projection back onto the scalar form is by moment matching, which for a
    distribution living on the steep part of an exponential is a poor
    description of exactly the tail that matters.""",
    form=Form.RATE,
    params={
        "csf_viscosity_pa_s": lognormal(0.7e-3, 1.2, units="Pa.s",
                                        provenance=Provenance.LITERATURE,
                                        note="essentially water at 37 C; csf protein content is too "
                                             "low to matter outside pathology"),
        "csf_density_kg_m3": normal(1007.0, 3.0, units="kg/m^3",
                                    provenance=Provenance.LITERATURE,
                                    note="very slightly denser than water, which is why the brain "
                                         "floats and weighs ~50 g in situ rather than ~1400 g"),
        "elastance_coefficient": lognormal(0.11, 2.0, units="1/mL",
                                           provenance=Provenance.LITERATURE,
                                           source="pressure-volume index 25 mL, E1 = ln(10)/PVI",
                                           note="the exponent of the pressure-volume curve; the knee "
                                                "this implementation exists to represent"),
        "aqueduct_conductance": weak(1.0, 10.0, units="mL/(min.mmHg)",
                                     note="geometry-dominated and subject-specific; a stenosis is a "
                                          "statement about this number"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.PHYSICS,
)


# ---------------------------------------------------------------------------
# csf / interstitial exchange
# ---------------------------------------------------------------------------

CSF_INTERSTITIAL_EXCHANGE = process(
    id="csf_interstitial_exchange",
    doc="""the interface: solute and water crossing between the csf spaces and
    the interstitium that surrounds every cell.

    the ontology has no interface topology, so this is declared over the csf
    topology, whose edge set is the one that reaches the pial surface and the
    perivascular sheaths -- the places the exchange physically happens.  the
    alternative, declaring it over the interstitial topology, would put the
    edges on the wrong side of the boundary and make the perivascular route
    inexpressible.  this is the one deviation from the inventory in §5, which
    names an `interface` topology docs/CONTRACT.md does not carry, and it is
    recorded here rather than resolved by inventing a topology.

    the mechanism is the contested part and the declaration is written to keep
    the contest visible.  the glymphatic account has csf entering along
    periarterial spaces, crossing the astrocytic endfoot sheet through aquaporin-4,
    mixing convectively with interstitial fluid and leaving along perivenous
    routes; the clearance it produces roughly doubles in sleep and under
    anaesthesia, an effect large enough that it is now the standard explanation
    for why sleep deprivation raises amyloid burden.  the objection is
    quantitative -- the pressure gradients needed to drive convection through a
    porous medium with this permeability are not obviously available -- and it is
    a serious objection, so the convective terms below carry speculative priors
    while the state dependence, which is measured, carries a literature one.

    it reads extracellular potassium in the spectral form and writes scalar csf
    and interstitial state, so the registered ``spectral -> scalar`` conversion
    applies: the window mean of the potassium transient survives and its
    rhythmicity does not.  that is the specific loss which makes the sleep
    question -- whether it is slow-wave *structure* or merely mean ionic state
    that opens the space -- unanswerable in this graph.""",
    inputs=(
        within("csf", "pressure", "solute", "velocity",
               region=OnSupport("csf_space"), band=TRANSPORT_BAND),
        within("extracellular", "k", "na", band=TRANSPORT_BAND),
        within("extracellular", "osmolarity", "volume_fraction", band=TRANSPORT_BAND),
        within("extracellular", "adenosine", "noradrenaline", band=TRANSPORT_BAND),
        within("blood", "volume", band=CSF_PULSATION),
        within("structural", "gliosis", band=STRUCTURAL),
        within("material", "porosity", "tortuosity", band=STRUCTURAL),
    ),
    outputs=(
        within("csf", "solute", region=OnSupport("csf_space"), band=TRANSPORT_BAND),
        within("extracellular", "osmolarity", "volume_fraction", band=TRANSPORT_BAND),
        within("extracellular", "adenosine", band=TRANSPORT_BAND),
    ),
    topology="csf",
    timescale_s=60.0,
    validity=Validity(
        min_spacing_mm=0.05, max_spacing_mm=10.0, band=TRANSPORT_BAND,
        note="a mass-transfer coefficient across a boundary.  the perivascular annulus is "
             "tens of micrometres wide, so below ~50 um the geometry must be resolved and a "
             "lumped coefficient is a category error; above ~10 mm periarterial and "
             "perivenous routes are averaged into one number and the directionality that is "
             "the entire content of the glymphatic hypothesis is lost."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"fluid", "clearance", "slow", "contested"}),
    notes="spectral ionic inputs, scalar fluid outputs: the registered spectral -> scalar "
          "conversion applies and is recorded in provenance.",
)

implementation(
    name="glymphatic_state_dependent",
    process="csf_interstitial_exchange",
    doc="""convective perivascular exchange with an explicitly state-dependent
    exchange coefficient.

    the reason to carry this rather than a fixed rate is that the state
    dependence is the measured part.  extracellular volume fraction rises by
    roughly 60% in natural sleep and tracer influx rises with it; noradrenergic
    tone is the plausible controller, which is why locus-coeruleus transmitter
    concentration is an input here rather than a hidden assumption.  the effect
    size is severalfold, not a few percent, so a model with a constant
    coefficient does not merely lose precision -- it gets the ordering of
    sleep and wake clearance wrong.

    the convective machinery underneath is the speculative half.  cardiac
    pulsation is read as the pump, and whether arterial pulsation can actually
    drive net perivascular flow at the required rate is exactly the open
    question; the prior is wide enough to include zero net convection, in which
    case this implementation degrades to the diffusive one.""",
    form=Form.RATE,
    params={
        "wake_exchange_rate_hz": lognormal(3.0e-4, 3.0, units="1/s",
                                           provenance=Provenance.LITERATURE,
                                           source="csf tracer influx timescales, tens of minutes",
                                           note="an equilibration time of roughly an hour in the "
                                                "awake state"),
        "sleep_exchange_ratio": normal(2.0, 0.6, units="dimensionless",
                                       provenance=Provenance.LITERATURE,
                                       source="xie et al 2013; tracer influx and amyloid-beta "
                                              "clearance roughly double in sleep",
                                       note="the state dependence is the well-supported part of the "
                                            "glymphatic account, independently of the mechanism"),
        "ecs_volume_fraction_awake": normal(0.20, 0.03, units="dimensionless",
                                            provenance=Provenance.LITERATURE,
                                            source="nicholson & sykova, tma+ iontophoresis, alpha ~0.2"),
        "ecs_volume_fraction_sleep_ratio": normal(1.6, 0.3, units="dimensionless",
                                                  provenance=Provenance.LITERATURE,
                                                  source="xie et al 2013, ~60% increase in ecs volume",
                                                  note="a geometric change of this size alters both "
                                                       "diffusive and convective transport, so it is "
                                                       "not evidence for convection by itself"),
        "aquaporin4_conductance": weak(1.0, 10.0, units="dimensionless",
                                       note="endfoot water permeability; aqp4 knockout reduces tracer "
                                            "influx severalfold in some preparations and not in "
                                            "others, and the disagreement is unresolved"),
        "perivascular_pump_gain": speculative(0.3, 30.0,
                                              units="um/s per unit arterial volume pulse",
                                              note="whether arterial pulsation drives net perivascular "
                                                   "convection at a physiologically useful rate is the "
                                                   "central open question; this prior includes zero"),
        "noradrenaline_suppression": weak(0.5, 6.0, units="per uM",
                                          note="the proposed arousal controller of the exchange "
                                               "coefficient; the correlation is solid, the causal "
                                               "chain is not"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="iliff et al 2012; xie et al 2013; mestre et al 2018",
)

implementation(
    name="two_compartment_lti",
    process="csf_interstitial_exchange",
    doc="""first-order relaxation of interstitial solute towards csf, with one
    rate constant.

    the cheap form and the honest floor: it makes no claim about mechanism, only
    that a concentration difference across an interface produces a flux
    proportional to it.  select it when clearance is a boundary condition on
    something else rather than the object of study, which is most of the time.

    it is structurally unable to express the thing the nonlinear form exists for.
    with one constant rate there is no sleep-wake difference, no directionality
    between periarterial entry and perivenous exit, and no way for the exchange
    to depend on the state of the tissue doing it.""",
    form=Form.LTI,
    transfer=exchange_transfer,
    params={
        "exchange_rate_hz": lognormal(3.0e-4, 4.0, units="1/s",
                                      provenance=Provenance.LITERATURE,
                                      note="lumps every route across the boundary into one number; "
                                           "the spread spans the reported sleep-wake range because "
                                           "this form cannot distinguish the two states"),
        "gain": weak(1.0, 5.0, units="dimensionless",
                     note="partition coefficient between the compartments; near 1 for small ions "
                          "and far from it for anything protein-bound"),
    },
    tying=Tying.GLOBAL,
    provenance=Provenance.LITERATURE,
)

implementation(
    name="learned_clearance",
    process="csf_interstitial_exchange",
    doc="""a learned exchange coefficient conditioned on the retained ionic and
    modulatory spectrum, as a residual on the analytic rate.

    worth having for one specific reason: the state dependence above is
    parameterized by a single sleep-wake ratio, and the real dependence is on a
    continuum of arousal states with different ionic and modulatory signatures
    -- and possibly on the temporal structure of those signatures rather than
    only their means.  a learned f reading the spectrum before the collapse can
    carry that; the two-parameter ratio structurally cannot.

    the prior mean is the analytic rate, so with no data it reproduces it.  the
    cost is that uncertainty propagation stops being exact and the ensemble width
    it needs is real.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0, note="weight prior; a learned f is a different p(theta), not a "
                                           "different declaration"),
        "residual_gain": weak(0.3, 5.0, note="learned term is a residual on the analytic exchange "
                                             "rate so the prior stays physical"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


# ---------------------------------------------------------------------------
# interstitial transport
# ---------------------------------------------------------------------------

INTERSTITIAL_TRANSPORT = process(
    id="interstitial_transport",
    doc="""movement of solute and water through the extracellular space between
    neighbouring points in the tissue.

    this is the slow companion to `ionic_diffusion`, and the split between them
    is deliberate rather than redundant.  ionic diffusion is a fast, local,
    uptake-dominated process whose spatial scale is sqrt(D/u) -- tens of
    micrometres -- and whose corner is set by glial pumping.  this process is
    what is left after that: the transport of species with no fast uptake route
    over millimetres and hundreds of seconds, plus the water that moves with
    them.  writing them as one process would force one timescale onto both and
    would put the tortuosity of the extracellular space, which is the parameter
    this one is actually about, into a process where it does no work.

    the geometry is the substance here.  the extracellular space is a connected
    film roughly 40 nm across occupying about a fifth of the tissue volume, with
    a tortuosity near 1.6 that is unusually well measured and unusually stable.
    both numbers are state-dependent in ways that matter: the volume fraction
    falls during intense activity and collapses under ischaemia, which raises
    concentrations and hinders transport exactly when clearance is most needed.
    that coupling is why extracellular potassium is an input to a transport
    process at all -- potassium uptake drives astrocytic water influx, and the
    swelling closes the space the transport happens in.

    it reads spectral ionic state and writes scalar volume and osmolarity, so
    the registered ``spectral -> scalar`` conversion applies; the mean potassium
    load survives it and the burst structure that determines peak swelling does
    not.""",
    inputs=(
        within("extracellular", "k", "na", "cl", band=TRANSPORT_BAND),
        within("extracellular", "glutamate", "gaba", "adenosine", band=TRANSPORT_BAND),
        within("extracellular", "dopamine", "serotonin", "acetylcholine",
               "noradrenaline", band=TRANSPORT_BAND),
        within("extracellular", "osmolarity", "volume_fraction", band=TRANSPORT_BAND),
        within("csf", "pressure", band=TRANSPORT_BAND),
        within("material", "tortuosity", "porosity", band=STRUCTURAL),
    ),
    outputs=(
        within("extracellular", "glutamate", "gaba", "adenosine", band=TRANSPORT_BAND),
        within("extracellular", "dopamine", "serotonin", "acetylcholine",
               "noradrenaline", band=TRANSPORT_BAND),
        within("extracellular", "osmolarity", "volume_fraction", band=TRANSPORT_BAND),
    ),
    topology="interstitial",
    timescale_s=100.0,
    validity=Validity(
        min_spacing_mm=0.05, max_spacing_mm=5.0, band=TRANSPORT_BAND,
        note="a volume-averaged porous-medium description.  below ~50 um the averaging cell "
             "contains too few cells for a porosity and a tortuosity to exist as numbers; "
             "above ~5 mm grey and white matter are averaged together, and white matter "
             "transport is strongly anisotropic along the fibres in a way an isotropic "
             "tortuosity cannot represent."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"fluid", "transport", "slow"}),
    notes="spectral ionic inputs, scalar volume and osmolarity outputs: the registered "
          "spectral -> scalar conversion applies and appears in provenance.",
)

implementation(
    name="tortuous_diffusion_lti",
    process="interstitial_transport",
    doc="""hindered diffusion on the eigenmodes of the interstitial topology.

    the defensible default.  diffusion in the extracellular space is measured,
    the tortuosity correction is a single well-established factor, and the
    resulting operator is linear -- so it is exact at any timestep and costs one
    elementwise multiply per mode.

    its honest limitation is the one the whole glymphatic literature grew out of:
    diffusive clearance time scales as the square of distance, so a millimetre
    takes minutes and a centimetre takes days.  for small molecules over short
    distances that is fine; for a protein over the width of a hemisphere it is
    not, and this implementation will report a clearance time long enough to be
    obviously wrong rather than quietly wrong, which is the preferable failure.""",
    form=Form.LTI,
    transfer=tortuous_diffusion_transfer,
    params={
        "d_free_m2_s": lognormal(1.24e-9, 1.3, units="m^2/s",
                                 provenance=Provenance.LITERATURE,
                                 source="tma+ free diffusion coefficient at 37 C",
                                 note="the reference probe of the iontophoresis literature; a "
                                      "protein is two orders of magnitude slower and this "
                                      "parameter is what carries the difference"),
        "tortuosity": normal(1.6, 0.1, units="dimensionless",
                             provenance=Provenance.LITERATURE,
                             source="nicholson & sykova; grey matter lambda 1.5-1.7",
                             note="remarkably constant across regions and species, which is why it "
                                  "is one of the few parameters in this file with a tight prior"),
        "volume_fraction": normal(0.20, 0.03, units="dimensionless",
                                  provenance=Provenance.LITERATURE,
                                  source="ecs volume fraction alpha ~0.2 in cortex",
                                  note="falls towards 0.15 during intense activity and below 0.05 in "
                                       "ischaemia, which this constant cannot express"),
        "clearance_hz": weak(1.0e-4, 10.0, units="1/s",
                             note="lumped removal by every route other than diffusion; setting it to "
                                  "zero recovers pure diffusion and its L^2 scaling"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="nicholson & sykova 1998; sykova & nicholson 2008",
)

implementation(
    name="advection_diffusion",
    process="interstitial_transport",
    doc="""diffusion with a bulk drift through the extracellular space.

    the implementation that can be wrong in the direction the field is arguing
    about.  it adds one velocity, and the peclet number that velocity implies is
    the entire quantitative dispute: at the velocities usually proposed, bulk
    flow through the interstitial matrix is a small correction to diffusion over
    a millimetre and a large one over a centimetre.

    the prior on the velocity is speculative and deliberately spans zero-effect,
    because the measurements are indirect and the modelling arguments on both
    sides are competent.  selecting this implementation is a hypothesis, and the
    posterior on that one parameter is the thing to look at afterwards.""",
    form=Form.LTI,
    transfer=advection_diffusion_transfer,
    params={
        "d_free_m2_s": lognormal(1.24e-9, 1.3, units="m^2/s",
                                 provenance=Provenance.LITERATURE),
        "tortuosity": normal(1.6, 0.1, units="dimensionless",
                             provenance=Provenance.LITERATURE),
        "velocity_um_s": speculative(0.1, 30.0, units="um/s",
                                     note="proposed interstitial bulk flow velocities span two orders "
                                          "of magnitude and the modelling objections to the upper end "
                                          "are serious; this prior does not take a side"),
        "clearance_hz": weak(1.0e-4, 10.0, units="1/s"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.SPECULATIVE,
    source="iliff et al 2012 against smith et al 2017 and holter et al 2017",
)

implementation(
    name="osmotic_volume_response",
    process="interstitial_transport",
    doc="""transport coupled to a moving extracellular volume fraction.

    the nonlinear form, and the only one that closes the loop the process
    docstring describes: solute accumulation raises osmolarity, water follows
    into the cells, the extracellular space shrinks, and the same solute load is
    now at a higher concentration in a more tortuous space.  that positive
    feedback is real and is what makes spreading depolarization and ischaemic
    swelling behave the way they do.

    it costs a nonlinear solve, it does not preserve gaussianity, and the moment
    matching back onto the scalar form throws away the tail -- which here is the
    part where the feedback runs away.  select it when swelling is the question
    and the linear forms when it is not.""",
    form=Form.RATE,
    params={
        "osmotic_water_permeability": weak(1.0, 10.0, units="um/s per osmol",
                                           note="lumped astrocytic and neuronal aquaporin water flux; "
                                                "the aqp4 contribution is disputed in magnitude"),
        "baseline_osmolarity_mosm": normal(295.0, 8.0, units="mOsm/L",
                                           provenance=Provenance.LITERATURE,
                                           source="plasma and interstitial osmolarity, tightly "
                                                  "regulated 285-300"),
        "volume_fraction_floor": uniform(0.03, 0.10, units="dimensionless",
                                         provenance=Provenance.LITERATURE,
                                         note="the ecs does not close completely even in terminal "
                                              "ischaemia; measured minima are a few percent"),
        "swelling_time_constant_s": lognormal(30.0, 3.0, units="s",
                                              provenance=Provenance.LITERATURE,
                                              note="cell volume response to an osmotic step; seconds "
                                                   "to a minute"),
        "potassium_swelling_gain": weak(1.0, 10.0, units="volume fraction per mM",
                                        note="the k+-uptake-drives-swelling leg; the direction is "
                                             "certain and the magnitude in vivo is not"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.WEAK,
)

__all__ = [
    "CSF_FLOW",
    "CSF_INTERSTITIAL_EXCHANGE",
    "INTERSTITIAL_TRANSPORT",
]
