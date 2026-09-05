"""temperature: where the heat is made, where it goes, and why the brain is
warmer than the blood that cools it.

one process, and it is almost entirely a bookkeeping exercise -- which is the
point.  the brain does no external work, so on any timescale longer than the atp
pool turnover every joule of substrate oxidised leaves as heat; `metabolism`
already computes that quantity and this process only has to move it.  the three
sinks are conduction through the tissue, convection into flowing blood, and
whatever crosses the scalp.  of these the second dominates by roughly an order
of magnitude at normal perfusion, which is the single most important quantitative
fact in the file: the brain is cooled by its own blood supply, not by conduction
to the skull, and a drop in perfusion is a thermal event before it is a metabolic
one.

the characteristic time follows from that.  with tissue heat capacity around
3.8 MJ/(m^3 K) and a perfusion sink around 3.4e4 W/(m^3 K), the perfusion-limited
thermal time constant is close to 110 seconds -- which is why this process is
declared at a hundred-second timescale and why its band stops at 0.02 Hz.
temperature simply cannot follow anything faster; an energy deposition at 1 Hz
is felt by the tissue only through its mean.

two admissions.  first, docs/CONTRACT.md carries no thermal-conductivity or
specific-heat component, so both live here as parameters and `material.mass_density`
is read as the tissue-composition proxy that stands in for them -- the same
substitution `bold_formation` makes for magnetic susceptibility, and the same
reason: inventing components to carry constants would put physics in the state
graph.  second, this process reads `electromagnetic.efield`, which is carried in
the spectral form, and writes scalar temperature, so the registered
``spectral -> scalar`` conversion is inserted.  what it destroys there is
unusually consequential: specific absorption rate goes as the mean square of the
field, so the quantity that heats tissue is a second moment, and a conversion
that keeps the window mean and its variance retains just enough to compute it
and nothing at all about how the deposition is distributed in time.  a duty-cycled
rf sequence and a continuous one of equal mean power are the same object to this
graph, and they are not the same object to a patient.
"""

from __future__ import annotations

import numpy as np

from ibm.processes.base import implementation, process
from ibm.registry import Form
from ibm.vocabulary import (
    Band,
    HEMODYNAMIC,
    OnSupport,
    Provenance,
    STRUCTURAL,
    Tying,
    Validity,
    lognormal,
    normal,
    speculative,
    weak,
    within,
)

#: the band temperature itself is meaningful over.  0.02 Hz is fifty seconds,
#: which is already faster than the perfusion-limited thermal time constant; the
#: tissue is a low-pass filter with a two-minute corner and nothing above this
#: survives it.
THERMAL_BAND = Band(0.0, 0.02)

#: the band electromagnetic energy deposition is read over before it is
#: collapsed onto scalar temperature.  it is enormously wider than anything else
#: in the inventory because the fields that deposit power are device-driven: an
#: mr transmit coil runs at 128 MHz at 3 T, a focused-ultrasound-adjacent rf
#: chain higher still.  the neural model has no power up here at all, so this
#: selector reads exactly zero unless an intervention is driving the field --
#: which is the intended behaviour, not a waste.
DEPOSITION_BAND = Band(0.0, 1.0e9)


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def pennes_transfer(basis, conductivity_w_m_k: float = 0.51,
                    density_kg_m3: float = 1046.0,
                    specific_heat_j_kg_k: float = 3630.0,
                    perfusion_sink_w_m3_k: float = 34000.0,
                    wavelength_mm: float = 10.0):
    """one spatial eigenmode of the pennes bioheat equation.

        rho c dT/dt = -k q^2 T - w_b c_b T + Q,   q = 2 pi / L

        H = 1 / (k q^2 + w_b c_b + i omega rho c)

    the bioheat equation is *linear*, which is worth saying out loud because it
    is the reason this whole process is cheap: conduction, perfusion and source
    all enter additively, so on the eigenbasis of the spatial laplacian it is a
    first-order lag per mode and exact at any timestep.  nothing about brain
    thermodynamics requires a nonlinear solve until the perfusion itself starts
    depending on temperature.

    the ratio of the two sink terms is the physics.  at L = 10 mm, k q^2 is about
    200 W/(m^3 K) while w_b c_b is about 3.4e4 -- perfusion wins by more than two
    orders of magnitude, and the conduction term only becomes comparable below
    about a millimetre.  so a hot spot smaller than a millimetre spreads by
    conduction and one larger than that is washed out by blood, and the crossover
    is a length this expression makes explicit rather than a modelling choice.

    where it breaks: pennes assumes blood arrives at arterial temperature and
    equilibrates completely within the tissue cell, which is true of capillaries
    and false of the vessels that carry most of the flow.  counter-current
    exchange between the carotid and the jugular, and the pre-cooling of blood in
    thermally significant arteries, are both outside this form, and both are
    exactly what selective brain cooling is about.
    """
    omega = basis.omega
    q = 2.0 * np.pi / max(wavelength_mm * 1e-3, 1e-30)
    conduction = conductivity_w_m_k * q ** 2
    heat_capacity = density_kg_m3 * specific_heat_j_kg_k
    return 1.0 / (conduction + perfusion_sink_w_m3_k + 1j * omega * heat_capacity)


def perfusion_washout_transfer(basis, tau_s: float = 110.0, gain: float = 1.0):
    """the perfusion sink alone, as a first-order lag with an explicit time constant.

    the same pole as the full bioheat mode with the conduction term dropped,
    written separately because at every spacing coarser than a millimetre the
    conduction term is negligible and carrying the eigenvalue machinery for it is
    a pretence of precision.  tau = rho c / (w_b c_b), about 110 s at resting
    grey-matter perfusion, and it lengthens in proportion as perfusion falls --
    which is the honest way to say that an ischaemic region also stops being able
    to shed heat.
    """
    omega = basis.omega
    return gain / (1.0 + 1j * omega * tau_s)


# ---------------------------------------------------------------------------
# thermal diffusion
# ---------------------------------------------------------------------------

THERMAL_DIFFUSION = process(
    id="thermal_diffusion",
    doc="""conduction, perfusion and metabolic heating setting the temperature
    field of the head.

    declared over the local 3d spatial topology.  the inventory in
    ARCHITECTURE.md §5 describes this as "spatial + vascular", and the vascular
    half is genuinely present -- blood flow is the dominant sink -- but it enters
    as a *local* volumetric term rather than as transport along the tree: pennes'
    approximation is precisely the claim that blood delivers its temperature
    locally and leaves.  declaring it over the vascular topology would assert
    that heat is advected between connected tree segments, which is the
    counter-current physics this form explicitly does not contain.  the single
    topology field cannot say "local sink, vascular source", so it says local and
    this note says the rest.

    the inputs are the three sources and the one geometry.  `metabolic.heat` is
    the dissipation `metabolism` already computed; `blood.flow` sets the
    convective sink and `blood.oxygenation` is read because arterial blood
    arriving already warmed is a different boundary condition from arterial blood
    arriving at core temperature; `electromagnetic.efield` and
    `material.conductivity` together give the ohmic deposition that makes this
    process a safety model as well as a physiological one.  `material.mass_density`
    stands in for tissue composition, since the contract carries no thermal
    conductivity or heat capacity component and those live as parameters here.

    the temperatures involved are small and consequential.  brain temperature
    sits a few tenths of a degree above arterial, varies by roughly a degree
    across the depth of the cortex, swings with the circadian cycle by about half
    a degree, and rises measurably during intense activity.  a degree is not a
    rounding error at this scale: enzyme rates, channel kinetics and conduction
    velocity all move a few percent per degree, and none of those couplings exist
    in this graph.  temperature is written here and read almost nowhere, and that
    asymmetry is a real gap rather than a simplification.

    where it breaks: everything about the boundary.  the scalp and skull are a
    different thermal problem with a different perfusion and an evaporative
    surface, and this process treats them as more brain with different constants.
    for a deep grey-matter question that is harmless; for scalp heating under an
    rf coil, which is where the safety limit actually binds, it is the wrong
    model of the tissue that matters most.""",
    inputs=(
        within("thermal", "temperature", region=OnSupport("head_volume"), band=THERMAL_BAND),
        within("metabolic", "heat", band=HEMODYNAMIC),
        within("blood", "flow", "oxygenation", band=HEMODYNAMIC),
        within("electromagnetic", "efield", region=OnSupport("head_volume"),
               band=DEPOSITION_BAND),
        within("material", "conductivity", "mass_density", band=STRUCTURAL),
    ),
    outputs=(
        within("thermal", "temperature", region=OnSupport("head_volume"), band=THERMAL_BAND),
    ),
    topology="local",
    timescale_s=100.0,
    validity=Validity(
        min_spacing_mm=0.5, max_spacing_mm=20.0, band=THERMAL_BAND,
        note="a volume-averaged bioheat description.  below ~0.5 mm the averaging cell "
             "contains individual thermally significant vessels and a uniform perfusion sink "
             "is wrong in the specific way counter-current exchange is wrong; above ~20 mm "
             "brain, skull and scalp are averaged into one material whose thermal constants "
             "differ by a factor of two and whose perfusions differ by more."),
    provenance=Provenance.PHYSICS,
    tags=frozenset({"thermal", "physical", "safety", "slow"}),
    notes="spectral electromagnetic input, scalar thermal output: the registered "
          "spectral -> scalar conversion applies.  deposition is a second moment of the "
          "field, so the collapse keeps enough to compute mean power and nothing about how "
          "that power is distributed in time.",
)

implementation(
    name="pennes_bioheat_lti",
    process="thermal_diffusion",
    doc="""the bioheat equation on the eigenmodes of the spatial topology.

    the defensible default and the standard model of the field: it is what every
    rf safety limit, every hyperthermia plan and every transcranial ultrasound
    dosimetry calculation is computed with.  linear, exact at any timestep, one
    multiply per mode, and its parameters are among the best measured in this
    inventory -- brain thermal conductivity and specific heat are known to a few
    percent, which is not something that can be said of much else here.

    its two failures are both about blood.  perfusion is treated as a constant
    when it is the strongest temperature-dependent term in the system, and blood
    is assumed to arrive at a fixed arterial temperature when in fact it has been
    exchanging heat with the vessels it travelled through.  both errors point the
    same way -- the real brain is better at getting rid of heat than this model
    says -- so it is conservative for safety and optimistic for anyone trying to
    heat tissue on purpose.""",
    form=Form.LTI,
    transfer=pennes_transfer,
    params={
        "conductivity_w_m_k": normal(0.51, 0.05, units="W/(m.K)",
                                     provenance=Provenance.LITERATURE,
                                     source="brain thermal conductivity, grey 0.55 white 0.49",
                                     note="one of the tightest priors in the inventory; tissue "
                                          "thermal conductivity is close to water's 0.6 because "
                                          "tissue is mostly water"),
        "specific_heat_j_kg_k": normal(3630.0, 150.0, units="J/(kg.K)",
                                       provenance=Provenance.LITERATURE,
                                       source="brain specific heat capacity ~3600-3700",
                                       note="again close to water's 4180, scaled by water content"),
        "density_kg_m3": normal(1046.0, 20.0, units="kg/m^3",
                                provenance=Provenance.LITERATURE,
                                source="brain tissue density; grey 1045, white 1041"),
        "blood_specific_heat_j_kg_k": normal(3800.0, 150.0, units="J/(kg.K)",
                                             provenance=Provenance.LITERATURE),
        "perfusion_sink_w_m3_k": lognormal(34000.0, 1.5, units="W/(m^3.K)",
                                           provenance=Provenance.LITERATURE,
                                           source="w_b*c_b at 50 mL/100g/min grey matter perfusion",
                                           note="two orders of magnitude above the conduction term at "
                                                "centimetre scale; this single number is why the "
                                                "brain's temperature is set by its blood supply"),
        "arterial_temperature_c": normal(37.0, 0.4, units="degC",
                                         provenance=Provenance.LITERATURE,
                                         note="the boundary condition the whole solution is measured "
                                              "against; circadian variation is about 0.5 C"),
        "brain_arterial_offset_c": normal(0.5, 0.3, units="degC",
                                          provenance=Provenance.LITERATURE,
                                          source="direct brain temperature measurement; brain runs "
                                                 "warmer than arterial blood",
                                          note="the steady-state consequence of metabolic heating "
                                               "against perfusion, and a useful check: it should be "
                                               "predicted by this process, not imposed"),
        "metabolic_heat_w_m3": normal(10437.0, 2000.0, units="W/m^3",
                                      provenance=Provenance.LITERATURE,
                                      source="the pennes-model brain metabolic heat used throughout "
                                             "the rf and ultrasound safety literature",
                                      note="the same quantity metabolism computes; it appears here "
                                           "only as the prior mean the input is expected to match, "
                                           "and a materialization that fits both should tie them"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.PHYSICS,
    source="pennes 1948; iso/iec tissue property databases",
)

implementation(
    name="thermoregulatory_perfusion",
    process="thermal_diffusion",
    doc="""bioheat with perfusion that responds to temperature, and an explicit
    ohmic deposition term.

    the form to select when the question is about heating rather than about
    resting temperature, because it can be wrong in the direction the physiology
    is wrong in: cerebral blood flow rises with temperature, so the sink that
    dominates the energy balance is itself a function of the state being solved
    for.  the feedback is stabilizing and strong, and a linear model that omits
    it overestimates steady-state temperature rise -- conservatively for safety,
    but wrongly.

    it also carries the ohmic term explicitly, ``sar = sigma |E|^2 / (2 rho)``,
    which is where `material.conductivity` and the electromagnetic input do their
    work.  the mean square is computed after the spectral collapse, so this
    implementation inherits the loss the process docstring describes: it can tell
    a materialization how hot the tissue gets under a given mean deposited power
    and cannot tell it whether pulsing that power changes the answer.

    nonlinear, evaluated in time, does not preserve gaussianity.""",
    form=Form.RATE,
    params={
        "perfusion_temperature_gain": lognormal(0.05, 3.0, units="1/K",
                                                provenance=Provenance.LITERATURE,
                                                source="cbf rises several percent per degree of "
                                                       "temperature; the same q10 that governs cmro2",
                                                note="stabilizing feedback: hotter tissue is better "
                                                     "perfused and therefore better cooled"),
        "q10_metabolic": normal(2.3, 0.4, units="dimensionless",
                                provenance=Provenance.LITERATURE,
                                source="q10 of cerebral metabolic rate, 2-3",
                                note="the destabilizing half of the same story -- metabolism also "
                                     "rises with temperature -- and it loses to perfusion, which is "
                                     "why brains do not thermally run away"),
        "tissue_conductivity_s_m": lognormal(0.35, 1.6, units="S/m",
                                             provenance=Provenance.LITERATURE,
                                             source="brain electrical conductivity at rf, ~0.3-0.6 "
                                                    "S/m and strongly frequency dependent",
                                             note="read from material.conductivity where an atlas "
                                                  "supplies it; this prior is the fallback"),
        "sar_limit_w_kg": normal(3.2, 0.3, units="W/kg",
                                 provenance=Provenance.LITERATURE,
                                 source="iec 60601-2-33 head sar limit, normal operating mode",
                                 note="not a physical parameter but the regulatory threshold this "
                                      "implementation exists to be checked against"),
        "scalp_boundary_conductance": weak(10.0, 5.0, units="W/(m^2.K)",
                                           note="lumped convective and evaporative loss at the scalp; "
                                                "matters enormously for surface heating and hardly at "
                                                "all for deep structures"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="pennes 1948; iec 60601-2-33",
)

implementation(
    name="perfusion_washout_lti",
    process="thermal_diffusion",
    doc="""the perfusion sink alone: one lag with a two-minute time constant.

    the cheap form, for materializations coarser than a millimetre where the
    conduction term contributes less than a percent of the sink and carrying the
    spatial eigenvalues for it is decoration.  select it when temperature is a
    boundary condition on something else.

    it cannot represent a spatial gradient at all, which means it cannot answer
    the only questions focal heating poses.  that is the trade and it is a large
    one.""",
    form=Form.LTI,
    transfer=perfusion_washout_transfer,
    params={
        "tau_s": lognormal(110.0, 1.6, units="s", provenance=Provenance.PHYSICS,
                           source="rho*c / (w_b*c_b) at resting grey-matter perfusion",
                           note="lengthens in inverse proportion to perfusion, so an ischaemic "
                               "region is also a thermally isolated one"),
        "gain": weak(1.0, 3.0, units="K per W/m^3",
                     note="steady-state temperature rise per unit volumetric power; the reciprocal "
                          "of the total sink"),
    },
    tying=Tying.GLOBAL,
    provenance=Provenance.PHYSICS,
)

implementation(
    name="quasi_static_thermal_constraint",
    process="thermal_diffusion",
    doc="""temperature as an algebraic function of the instantaneous heat load.

    the stiff limit of the same equation, per ARCHITECTURE.md §4: when the
    materialization's window is much longer than the two-minute thermal time
    constant, temperature has already relaxed and carrying its dynamics is
    carrying a transient no one will observe.  ``T = T_a + Q / (k q^2 + w_b c_b)``
    with no time constant at all.

    it is the right form for a long-duration exposure question -- a whole
    scanning session, a chronic implant -- and exactly the wrong one for a short
    intense one, where the entire answer is that the tissue did not have time to
    reach steady state.  the failure is not subtle: it will report the asymptote
    for a two-second sonication.""",
    form=Form.CONSTRAINT,
    params={
        "steady_state_gain_k_per_w_m3": lognormal(2.9e-5, 2.0, units="K/(W/m^3)",
                                                  provenance=Provenance.PHYSICS,
                                                  source="reciprocal of the perfusion sink",
                                                  note="10 kW/m^3 of metabolic heat times this gain "
                                                       "gives the ~0.3 K brain-arterial offset, which "
                                                       "is the consistency check worth running"),
        "spatial_averaging_mm": speculative(10.0, 3.0, units="mm",
                                            note="the scale over which the steady state is asserted; "
                                                 "below it the constraint form is simply not valid "
                                                 "and the number is a modelling choice, not a "
                                                 "measurement"),
    },
    tying=Tying.GLOBAL,
    provenance=Provenance.PHYSICS,
)

__all__ = ["THERMAL_DIFFUSION"]
