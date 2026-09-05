"""the extracellular ionic milieu: what the membrane takes out of it and puts
back, and how what is left over spreads.

two processes, and the reason they are two is that they have different
topologies and different failure modes.  `ionic_exchange` is a local trade
between a population's membranes and the interstitium it sits in; it is where
the pump load that `metabolism` pays for is generated, and it is a rate law
because pump saturation and reversal-potential shifts are exactly the
nonlinearities that make the ionic environment interesting.  `ionic_diffusion`
is transport through the interstitial space, and it is linear.

the interstitium deserves a sentence of its own because it is where this file's
one piece of real physics lives.  the extracellular space is about a fifth of
tissue volume and is a tortuous connected sheet, so a solute's effective
diffusivity is its free value divided by the square of a tortuosity around 1.6 --
roughly a factor of 2.6 slower than in free solution.  that number is measured,
not assumed, and it means the geometry of the extracellular space enters as a
material property rather than as a mesh.  `material.tortuosity` and
`material.porosity` are therefore inputs here.

the ions are carried in the spectral form and the rest of the extracellular
field is scalar, which is the right split: potassium genuinely has structure
across timescales -- a per-spike transient riding on a seconds-long
accumulation riding on a state-dependent baseline -- while pH and osmolarity do
not, on any timescale a materialization will resolve.  couplings that cross the
split go through the registered conversion and lose the temporal structure,
which is recorded rather than hidden.
"""

from __future__ import annotations

import numpy as np

from ibm.processes.base import (
    diffusion_mode,
    implementation,
    leaky_integrator,
    low_pass,
    process,
)
from ibm.registry import Form
from ibm.vocabulary import (
    Band,
    HEMODYNAMIC,
    LFP,
    STRUCTURAL,
    Provenance,
    Tying,
    Validity,
    lognormal,
    normal,
    speculative,
    weak,
    within,
)

#: the band the *neural* drive is read over.  it has to reach up here: a single
#: population spike moves interstitial potassium measurably, and the
#: accumulation that matters over seconds is the running integral of exactly
#: those transients.
DRIVE = Band(0.0, 300.0)

#: the band the ionic state itself is meaningful over, and it is narrower than
#: the drive on purpose.  the interstitial concentrations are declared
#: meaningful only to 50 Hz, because a lumped extracellular concentration above
#: that is the aliased image of individual release and pump events rather than
#: a field -- so the integration from drive to concentration is also where the
#: bandwidth collapses.
IONIC = Band(0.0, 50.0)

#: the slow half -- accumulation, baseline shift, volume change.  osmolarity
#: and volume fraction live only here, because a cell cannot swell at 40 Hz.
IONIC_SLOW = Band(0.0, 1.0)


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def potassium_accumulation_transfer(basis, tau_clearance_s: float = 2.0,
                                    release_gain: float = 1.0):
    """transmembrane current -> interstitial potassium, as a leaky integrator.

    the reason it is an integrator rather than a low-pass is that the DC gain
    is the whole quantity of interest.  every action potential and every
    excitatory synaptic current leaves potassium outside the cell, and what
    determines whether the tissue's excitability drifts is the *accumulated*
    excess against clearance -- so the steady-state response to a maintained
    drive, which is the clearance time constant, is the number this form
    exists to get right.  a plain low-pass would have unit DC gain and would
    say that sustained firing raises potassium by a fixed amount regardless of
    how fast the glia are pumping, which is backwards.

    where it breaks: clearance is not linear.  Na/K-ATPase saturates, so above
    roughly 10-12 mM the ceiling that normally caps potassium disappears and
    the system runs away -- which is spreading depolarization, and is the
    single most important thing about extracellular potassium that this form
    cannot represent.
    """
    return release_gain * leaky_integrator(basis, tau_clearance_s)


def interstitial_diffusion_transfer(basis, d_free_m2_s: float = 1.96e-9,
                                    tortuosity: float = 1.6,
                                    wavelength_mm: float = 0.5,
                                    uptake_hz: float = 0.5):
    """one spatial eigenmode of interstitial diffusion, with uptake.

    ``D_eff = D_free / lambda^2`` is the measured relation, not a fitted one:
    the extracellular space is a connected tortuous sheet and a solute takes a
    longer path through it than through free solution.  with a tortuosity of
    1.6 the correction is a factor of about 2.6, and it is the same factor for
    every small ion, which is why it belongs in a material property rather than
    in each species' parameters.

    uptake dominates for potassium and is the reason this is not a pure
    diffusion kernel.  glial Na/K-ATPase clears interstitial potassium far
    faster than diffusion redistributes it, so the spatial extent of a hot spot
    is ``sqrt(D_eff / u)`` -- with the values here, a few tens of micrometres.
    that is smaller than any materialization ibm-1 will run, which is the
    honest reason potassium can usually be treated as local: not because
    diffusion is slow, but because uptake wins.

    for calcium the balance is different and worse: it is buffered, actively
    extruded, and depleted rather than accumulated during activity, so applying
    this same form to calcium with a fitted uptake is a convenience rather than
    a mechanism.
    """
    d_eff = d_free_m2_s / max(tortuosity, 1e-6) ** 2
    return diffusion_mode(basis, d_eff, wavelength_mm, uptake_hz)


def volume_osmotic_transfer(basis, tau_water_s: float = 5.0, gain: float = 1.0):
    """osmotic load -> extracellular volume fraction, as a single slow pole.

    water follows solute across an aquaporin-rich membrane on a timescale of
    seconds, so activity that moves ions into cells shrinks the extracellular
    space by a few percent -- which raises every remaining concentration and
    lowers the effective diffusivity at the same time.  a small effect that is
    kept because it is a positive feedback: shrinking the space concentrates
    the potassium that caused the shrinking.
    """
    return gain * low_pass(basis, tau_water_s)


# ---------------------------------------------------------------------------
# nonlinear rate laws
# ---------------------------------------------------------------------------


def pump_limited_exchange(x, theta) -> dict:
    """membrane flux out, saturating Na/K-ATPase back in.

    release is proportional to transmembrane current; clearance is a
    michaelis-menten pump with a finite maximum rate.  the two together give
    the ceiling that the linear form lacks: as long as demand is below vmax the
    potassium concentration settles at a modest elevation, and as soon as it is
    not, nothing stops the rise.  that discontinuity in behaviour, not the
    numbers on either side of it, is why this implementation exists.
    """
    k = np.asarray(x["extracellular.k"], dtype=float)
    j = np.asarray(x["neural.transmembrane_current"], dtype=float)
    k0 = float(theta.get("k_baseline_mm", 3.0))
    vmax = float(theta.get("pump_vmax_mm_s", 2.0))
    km = float(theta.get("pump_km_mm", 3.5))
    release = float(theta.get("release_gain", 1.0)) * np.abs(j)
    uptake = vmax * (k - k0) / (km + np.maximum(k - k0, 0.0))
    return {"extracellular.k": release - uptake}


# ---------------------------------------------------------------------------
# ionic exchange
# ---------------------------------------------------------------------------

IONIC_EXCHANGE = process(
    id="ionic_exchange",
    doc="""the trade across neuronal and glial membranes: potassium and sodium
    out and back, calcium in and depleted, chloride loaded, and the pH shift
    that comes with all of it.

    inputs are the things that actually move charge -- transmembrane current,
    the synaptic conductances that carry most of it, and firing rate as the
    proxy for action-potential sodium and potassium flux -- plus the current
    concentrations, since every flux is down a gradient, plus `metabolic.atp`,
    because the pump that restores the gradients is the brain's largest single
    energy expense and stops when the atp does.

    outputs include `neural.exc.potential`, and that is the load-bearing edge
    in this file.  extracellular potassium sets the potassium reversal
    potential, which sets resting potential and the driving force on every
    potassium conductance, so ionic state is not a passive readout of activity:
    it feeds back on excitability with a time constant of seconds.  a model in
    which activity writes ion concentrations and the concentrations write
    nothing back is a model in which the ionic field is decoration.  the loop
    closed here is what makes it a mechanism -- it is the substrate of
    activity-dependent excitability changes, of the potassium-mediated part of
    neurovascular coupling, and of spreading depolarization.

    where this breaks.  it is a two-compartment description -- inside and
    outside -- with no glia as a separate compartment, when the glial syncytium
    is where spatial buffering happens: potassium enters an astrocyte here and
    leaves it a hundred micrometres away through gap junctions, which is a
    transport mechanism with no representation in this ontology at all.  and
    the intracellular concentrations are treated as constants, which is right
    for sodium and potassium over seconds and simply wrong for chloride, whose
    intracellular accumulation under sustained inhibition is the mechanism by
    which inhibition becomes excitatory.""",
    inputs=(
        within("neural", "transmembrane_current", band=DRIVE),
        within("neural", "exc.activity", "inh.activity", band=LFP),
        within("neural", "exc.ampa", "inh.gaba_a", band=DRIVE),
        within("neural", "exc.potential", "inh.potential", band=DRIVE),
        within("extracellular", "k", "na", "ca", "cl", band=IONIC),
        within("extracellular", "volume_fraction", band=IONIC_SLOW),
        within("metabolic", "atp", band=HEMODYNAMIC),
    ),
    outputs=(
        within("extracellular", "k", "na", "ca", "cl", band=IONIC),
        within("extracellular", "ph", band=IONIC_SLOW),
        within("neural", "exc.potential", "inh.potential", band=DRIVE),
        within("neural", "transmembrane_current", band=DRIVE),
    ),
    topology="local",
    timescale_s=0.1,
    validity=Validity(
        min_spacing_mm=0.01, max_spacing_mm=3.0, band=IONIC,
        note="the interstitium is a genuinely microscopic compartment -- 20 nm sheets -- so "
             "any materialized position is already an enormous average over it, and the "
             "0.01 mm floor is where that average stops containing enough cells to be one.  "
             "above ~3 mm the accumulation is diluted over grey and white matter together, "
             "which have very different pump densities.",
    ),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"ionic", "local", "excitability"}),
    notes="reads scalar metabolic.atp and writes spectral ionic state, so the registered "
          "scalar -> spectral conversion applies on that input and appears in provenance.",
)

implementation(
    name="accumulation_lti",
    process="ionic_exchange",
    doc="""potassium as a leaky integrator of transmembrane current, linearized
    about the resting concentration.

    the cheap and defensible default for ordinary activity.  its one real
    parameter is the clearance time constant, which is measured: potassium
    transients evoked by stimulation decay over a couple of seconds, and the
    steady elevation under sustained drive is a fraction of a millimolar.

    the linearization is about a 3 mM baseline and is good for the fraction of
    a millimolar that normal activity produces.  it has no ceiling and no
    pump saturation, so it will happily report 30 mM potassium under strong
    drive where the real tissue would have depolarized and gone silent.""",
    form=Form.LTI,
    transfer=potassium_accumulation_transfer,
    params={
        "tau_clearance_s": lognormal(2.0, 1.8, units="s", provenance=Provenance.LITERATURE,
                                     source="stimulus-evoked extracellular K+ transients decay "
                                            "over ~1-5 s",
                                     note="dominated by glial Na/K-ATPase rather than by "
                                          "diffusion, which is why it is this fast"),
        "release_gain": weak(1.0, 6.0, units="mM per unit transmembrane current",
                             note="absorbs the extracellular volume fraction, the fraction of "
                                  "membrane current carried by potassium, and every unit "
                                  "convention upstream"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="heinemann & lux 1977 ceiling of stimulus-induced potassium accumulation",
)

implementation(
    name="pump_saturation_rate",
    process="ionic_exchange",
    doc="""explicit Na/K-ATPase kinetics with a finite maximum rate.

    this is the implementation that can be wrong in the direction the
    physiology is wrong in.  below the pump's capacity it behaves almost
    exactly like the linear form -- concentrations settle at a modest
    elevation and the extra parameters buy nothing.  above it, clearance stops
    growing with load and the concentration has no equilibrium, which is
    spreading depolarization: a wave of near-complete ionic collapse that
    propagates at millimetres per minute and is implicated in migraine aura and
    in the expansion of ischaemic injury.

    the ceiling itself is measured and solid -- roughly 10-12 mM in cortex,
    remarkably reproducible.  what is weak is everything about the transition:
    the pump's effective in-vivo vmax at the population scale, how glial and
    neuronal pumps divide the load, and how much of the observed ceiling is
    pump capacity rather than the potassium-dependent changes in excitability
    that shut the tissue down before the pump gives out.

    the stoichiometry is physics: three sodium out for two potassium in per
    atp, so the pump is electrogenic and its own activity hyperpolarizes the
    cell -- which is a negative feedback the linear form has no term for.""",
    form=Form.RATE,
    fn=pump_limited_exchange,
    params={
        "k_baseline_mm": normal(3.0, 0.3, units="mM", provenance=Provenance.LITERATURE,
                                source="resting extracellular potassium in mammalian cortex, "
                                       "~3 mM",
                                note="held far below plasma potassium by continuous pumping; "
                                     "it is a maintained disequilibrium, not an equilibrium"),
        "k_ceiling_mm": normal(11.0, 1.5, units="mM", provenance=Provenance.LITERATURE,
                               source="heinemann & lux 1977; the K+ ceiling, 10-12 mM",
                               note="crossing it is spreading depolarization, not a larger "
                                    "version of ordinary accumulation"),
        "pump_vmax_mm_s": weak(2.0, 6.0, units="mM/s",
                               note="population-scale maximum clearance rate; single-channel "
                                    "and single-cell pump kinetics are well measured and their "
                                    "aggregation to a tissue vmax is not"),
        "pump_km_mm": lognormal(3.5, 1.5, units="mM", provenance=Provenance.LITERATURE,
                                note="Na/K-ATPase half-activation for extracellular K+, near "
                                     "the resting concentration -- so the pump sits on the "
                                     "steep part of its curve at rest, by design"),
        "pump_stoichiometry": normal(1.5, 0.0001, units="Na per K", provenance=Provenance.PHYSICS,
                                     note="3:2, and therefore electrogenic; the pump current "
                                          "hyperpolarizes the cell that runs it"),
        "release_gain": weak(1.0, 6.0, units="mM per unit transmembrane current"),
        "chloride_loading_gain": speculative(0.1, 20.0, units="mM per nS.s",
                                             note="intracellular chloride accumulation under "
                                                  "sustained inhibition, which shifts E_GABA "
                                                  "depolarized.  the mechanism is established "
                                                  "and its mesoscale magnitude is not"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="heinemann & lux 1977; kager, wadman & somjen 2000",
)

implementation(
    name="learned_ionic_exchange",
    process="ionic_exchange",
    doc="""a learned map from the neural spectrum to ionic fluxes.

    the analytic forms above are functions of mean transmembrane current, and
    the real dependence is not on the mean: the same charge moved in a
    synchronous burst and spread evenly over a second loads the pump
    differently, because the pump's response is saturating and saturation does
    not commute with averaging.  a learned f reading the retained spectrum can
    carry that, which is the specific thing it is here for -- not general
    flexibility.  centred on the accumulation form so the prior is physical.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "residual_gain": weak(0.3, 5.0, note="zero recovers the analytic accumulation"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


# ---------------------------------------------------------------------------
# ionic diffusion
# ---------------------------------------------------------------------------

IONIC_DIFFUSION = process(
    id="ionic_diffusion",
    doc="""transport of small ions through the interstitial space, and the
    osmotic volume change that rides along with it.

    declared on the interstitial topology rather than on a 3d local one because
    what diffuses is not what is spatially adjacent: the extracellular space is
    a connected tortuous sheet with a porosity around 0.2, and two points
    separated by a cell body are further apart along it than through it.  the
    topology carries that connectivity; `material.tortuosity` and
    `material.porosity` carry how much longer the path is.

    the useful consequence, and the reason this process is usually small, is
    that potassium's uptake rate beats its diffusion rate by a wide margin.
    the spatial scale over which an ionic hot spot spreads before it is cleared
    is ``sqrt(D_eff / uptake)``, which for potassium is tens of micrometres --
    far below any materialized spacing.  so at ordinary resolution this process
    is nearly a no-op, and that is a *derived* result rather than an assumption,
    which is exactly the kind of thing having the process declared lets a model
    check instead of assume.

    it stops being a no-op in precisely the regime where it matters most.  if
    uptake saturates, the ``sqrt(D/u)`` scale diverges, diffusion becomes the
    only remaining clearance mechanism, and the resulting reaction-diffusion
    system supports a travelling wave -- spreading depolarization, moving at
    2-5 mm per minute.  a linear diffusion form cannot produce that wave; it is
    declared here as the thing this process is missing rather than left
    unmentioned.

    volume fraction and osmolarity are outputs because activity shrinks the
    extracellular space by a few percent, and that is a positive feedback:
    less volume means higher concentration of everything already in it, and a
    lower effective diffusivity to carry it away.""",
    inputs=(
        within("extracellular", "k", "na", "ca", "cl", band=IONIC),
        within("extracellular", "osmolarity", "volume_fraction", band=IONIC_SLOW),
        within("material", "tortuosity", "porosity", band=STRUCTURAL),
    ),
    outputs=(
        within("extracellular", "k", "na", "ca", "cl", band=IONIC),
        within("extracellular", "osmolarity", "volume_fraction", band=IONIC_SLOW),
    ),
    topology="interstitial",
    timescale_s=1.0,
    validity=Validity(
        min_spacing_mm=0.02, max_spacing_mm=2.0, band=Band(0.0, 100.0),
        note="the diffusion length in one clearance time is a few tens of micrometres, so "
             "at spacings above ~0.5 mm this process moves essentially nothing between "
             "cells and is being carried for the regime where uptake fails rather than for "
             "ordinary activity.  below ~0.02 mm the continuum description of a 20 nm "
             "extracellular sheet stops being a continuum.",
    ),
    provenance=Provenance.PHYSICS,
    tags=frozenset({"ionic", "transport", "interstitial"}),
)

implementation(
    name="tortuous_diffusion_lti",
    process="ionic_diffusion",
    doc="""free diffusivity corrected by tortuosity, with first-order uptake, on
    the eigenbasis of the interstitial laplacian.

    linear and therefore exact at any timestep, which matters because an
    explicit diffusion solve is stiff: its stable timestep scales with the
    square of the spacing, so a fine interstitial materialization would
    otherwise set the timestep for the entire model.  diagonalizing it removes
    that coupling entirely.

    the free diffusivities are physics and the tortuosity is a measurement --
    real-time iontophoresis in cortex gives lambda around 1.6 with remarkable
    consistency across regions and species.  what is weak is uptake, which is a
    lumped stand-in for glial Na/K-ATPase, spatial buffering through the
    astrocytic syncytium, and clearance into blood, three mechanisms with
    different geometries collapsed into one rate.""",
    form=Form.LTI,
    transfer=interstitial_diffusion_transfer,
    params={
        "d_free_m2_s": lognormal(1.96e-9, 1.1, units="m^2/s", provenance=Provenance.PHYSICS,
                                 source="free aqueous diffusion coefficient of K+ at 37 C, "
                                        "1.96e-9 m^2/s",
                                 note="Na+ is ~1.33e-9 and Cl- ~2.03e-9; the species share this "
                                      "parameter with a per-species gain rather than each "
                                      "carrying a separate declaration"),
        "tortuosity": normal(1.6, 0.1, units="dimensionless", provenance=Provenance.LITERATURE,
                             source="nicholson & sykova, real-time iontophoresis in cortex; "
                                    "lambda ~ 1.6",
                             note="D_eff = D_free / lambda^2, so 1.6 is a factor of 2.6 "
                                  "slowdown.  it rises during ischaemia and during "
                                  "spreading depolarization as the space collapses"),
        "uptake_hz": lognormal(0.5, 3.0, units="1/s", provenance=Provenance.WEAK,
                               note="lumps glial pumping, syncytial spatial buffering and "
                                    "vascular clearance into one first-order rate; it is the "
                                    "term that makes potassium local, and the term whose "
                                    "failure makes it a travelling wave"),
        "wavelength_mm": weak(0.5, 4.0, units="mm",
                              note="the spatial scale of the mode being evaluated; a "
                                   "materialization holding the interstitial laplacian's "
                                   "eigenvalues supplies them instead"),
    },
    tying=Tying.GLOBAL,
    provenance=Provenance.PHYSICS,
    source="nicholson & sykova 1998 extracellular space structure revealed by diffusion analysis",
)

implementation(
    name="osmotic_volume_lti",
    process="ionic_diffusion",
    doc="""the osmotic limb alone: solute load into extracellular volume fraction.

    separated from the diffusion limb so that a materialization can take the
    transport without committing to the volume feedback, and so that the water
    time constant appears in exactly one place.  small in normal physiology --
    a few percent shrinkage during activity -- and dominant in the two places
    it is not small, cytotoxic oedema and spreading depolarization, where the
    extracellular space collapses by more than half.""",
    form=Form.LTI,
    transfer=volume_osmotic_transfer,
    params={
        "tau_water_s": lognormal(5.0, 2.0, units="s", provenance=Provenance.LITERATURE,
                                 note="aquaporin-mediated water flux across astrocytic "
                                      "membranes; seconds, not milliseconds"),
        "baseline_volume_fraction": normal(0.2, 0.03, units="dimensionless",
                                           provenance=Provenance.LITERATURE,
                                           source="extracellular volume fraction of cortex, "
                                                  "alpha ~ 0.2",
                                           note="falls to ~0.05 during spreading depolarization, "
                                                "which is a fourfold concentration of everything "
                                                "dissolved in it"),
        "gain": weak(1.0, 5.0, units="volume fraction per mOsm"),
    },
    tying=Tying.GLOBAL,
    provenance=Provenance.LITERATURE,
)

implementation(
    name="reaction_diffusion_rate",
    process="ionic_diffusion",
    doc="""diffusion with saturating uptake: the form that can support a wave.

    the point of carrying it is a single qualitative behaviour the linear form
    structurally cannot have.  with first-order uptake the system is stable and
    every perturbation decays; with saturating uptake and a potassium-dependent
    release term the system is excitable, and a large enough local elevation
    propagates as a self-sustaining front at millimetres per minute.  that is
    cortical spreading depolarization, and it is the same mathematics as a
    nerve impulse at a different scale.

    every parameter of the excitable regime is weak, and honestly so.  the
    phenomenon is well documented and its quantitative mesoscale
    parameterization -- in human cortex, at the resolutions ibm-1 materializes,
    with realistic glial buffering -- does not exist in the literature.  nonlinear
    and stiff, so it costs a time-domain solve and gives up the exactness that
    is the whole reason the rest of this file is linear.""",
    form=Form.RATE,
    params={
        "wave_velocity_mm_min": normal(3.5, 1.0, units="mm/min",
                                       provenance=Provenance.LITERATURE,
                                       source="cortical spreading depression propagation, "
                                              "2-5 mm/min",
                                       note="one of the most reproducible numbers in the "
                                            "phenomenon, and a strong constraint on any "
                                            "reaction-diffusion parameterization that claims "
                                            "to produce it"),
        "excitation_threshold_mm": normal(11.0, 2.0, units="mM", provenance=Provenance.LITERATURE,
                                          note="the K+ ceiling, above which uptake stops "
                                               "keeping up"),
        "regenerative_gain": speculative(1.0, 20.0, units="dimensionless",
                                         note="potassium-dependent potassium release; the "
                                              "positive feedback that closes the excitable "
                                              "loop, and the least constrained number here"),
        "refractory_s": lognormal(180.0, 2.0, units="s", provenance=Provenance.LITERATURE,
                                  note="tissue cannot support a second wave for minutes, which "
                                       "is why the phenomenon is a single front rather than an "
                                       "oscillation"),
    },
    tying=Tying.PER_PARTITION,
    differentiable=False,
    provenance=Provenance.SPECULATIVE,
    source="tuckwell & miura 1978; somjen 2001 mechanisms of spreading depression",
)


__all__ = ["IONIC_EXCHANGE", "IONIC_DIFFUSION"]
