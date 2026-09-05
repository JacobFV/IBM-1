"""blood: what activity does to flow, what flow carries downstream, what the
blood trades with the tissue, and what an mr scanner sees of the result.

one admission belongs at the top of this file, because it applies to three of
the four processes in it.  neural and ionic state are carried in the spectral
form; blood state is carried in the scalar form.  every coupling from the first
to the second therefore passes through the registered ``spectral -> scalar``
conversion, which keeps the window mean and its variance and discards
everything else -- rhythms, phase, transients, and every cross-frequency
relationship.  that is not a bug in the conversion; blood genuinely has no
structure above roughly 0.5 Hz and carrying five hundred coefficients for it
would be budget spent on zeros.  but it *is* the exact place where the
multi-timescale structure of the model leaves the model, and it is the reason a
materialization cannot ask this graph why gamma bursts and theta-locked
activity of equal mean power produce different haemodynamic responses.  the
selectors below therefore say which band each process reads before the collapse
happens, so that at least the marginalization is a declared choice rather than
an accident of whatever window the runtime happened to pick.
"""

from __future__ import annotations

from ibm.processes.base import (
    alpha_synapse,
    implementation,
    low_pass,
    process,
    pure_delay,
    series,
)
from ibm.registry import Form
from ibm.vocabulary import (
    Band,
    GAMMA,
    HEMODYNAMIC,
    OnSupport,
    Provenance,
    Tying,
    Validity,
    lognormal,
    normal,
    speculative,
    uniform,
    weak,
    within,
)

#: the band over which synaptic drive is read before it is collapsed onto a
#: scalar blood variable.  wider than the haemodynamic band on purpose: the
#: vasoactive signal is driven by synaptic activity right up into gamma, and
#: restricting the *read* to 0-0.5 Hz would throw the drive away before the
#: conversion ever got to it.
SYNAPTIC_DRIVE = Band(0.0, 120.0)


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def nvc_transfer(basis, efficacy: float = 0.5, tau_signal_s: float = 1.54,
                 tau_feedback_s: float = 2.44, onset_lag_s: float = 1.0):
    """linearized neurovascular transfer: lag, then two first-order lags.

    the balloon model's flow-inducing signal obeys
    ``s' = eps*u - s/tau_s - (f-1)/tau_f`` with ``f' = s``, which after
    linearization about baseline is exactly a cascade of two leaky integrators
    behind an onset delay.  the delay is a phase ramp here, not a history
    buffer, which is the whole reason this stays LTI.
    """
    return efficacy * series(pure_delay(basis, onset_lag_s),
                             low_pass(basis, tau_signal_s),
                             low_pass(basis, tau_feedback_s))


def vasoactive_transfer(basis, tau_rise_s: float = 0.9):
    """astrocyte-mediated vasoactive messenger release as an alpha kernel.

    an alpha function rather than a single lag because the messenger pathway
    (astrocytic Ca, arachidonic-acid derivatives, K+ onto smooth muscle) is a
    two-stage cascade with a genuine rise time -- flow does not begin to change
    the instant activity does.
    """
    return alpha_synapse(basis, tau_rise_s)


def windkessel_transfer(basis, tau_transit_s: float = 3.0,
                        tau_compliance_s: float = 8.0, autoregulation_gain: float = 0.6):
    """flow in -> flow out of a compliant vascular compartment, with myogenic
    autoregulation closed around it.

    the forward path is the balloon's transit-time lag; the loop is the slower
    myogenic response of upstream smooth muscle.  written as ``G / (1 + G K)``
    arithmetically rather than through ``base.feedback`` so that the loop gain
    stays visible as a parameter that evidence can move -- autoregulation gain
    is the single most subject- and pathology-dependent number here.
    """
    g = low_pass(basis, tau_transit_s)
    k = autoregulation_gain * low_pass(basis, tau_compliance_s)
    return g / (1.0 + g * k)


def spm_double_gamma_transfer(basis, peak_s: float = 6.0, undershoot_s: float = 16.0,
                              ratio: float = 6.0, dispersion_s: float = 1.0):
    """the canonical spm haemodynamic response function, in the frequency domain.

    ``h(t) = Gamma(t; 6, 1) - Gamma(t; 16, 1) / 6``.  a gamma density of integer
    shape ``n`` and scale ``d`` is ``n`` identical leaky integrators in series,
    so the whole hrf is two such cascades in parallel with a negative weight on
    the second -- no special kernel machinery, and exact at any timestep.
    """
    peak = series(*[low_pass(basis, dispersion_s)] * int(round(peak_s / dispersion_s)))
    under = series(*[low_pass(basis, dispersion_s)] * int(round(undershoot_s / dispersion_s)))
    return peak - under / ratio


# ---------------------------------------------------------------------------
# neurovascular coupling
# ---------------------------------------------------------------------------

NEUROVASCULAR_COUPLING = process(
    id="neurovascular_coupling",
    doc="""synaptic and astrocytic signalling sets local arteriolar tone.

    the input is deliberately synaptic state and gamma-band population activity
    rather than mean firing rate: the flow response tracks the metabolic cost of
    synaptic transmission and of restoring ionic gradients, which is why gamma
    power is a better single predictor of it than spike output, and why an
    inhibitory population that fires hard can drive flow up rather than down.

    the output is flow alone.  volume and pressure follow from flow through the
    compliant vessel, which is `vascular_flow`'s business; splitting them keeps
    the balloon's two halves in the two processes that actually own them.

    where this breaks: everything above small, activation-driven perturbations
    about baseline.  the coupling saturates, arterioles have a finite dilatory
    reserve, and the sign of the neurovascular relationship is not fixed --
    prolonged stimulation, cortical spreading depression and ischaemic penumbra
    all produce inverted responses this form cannot express.""",
    inputs=(
        within("neural", "exc.ampa", "exc.nmda", "inh.gaba_a", band=SYNAPTIC_DRIVE),
        within("neural", "exc.activity", "inh.activity", band=GAMMA),
        within("extracellular", "k", "ca", band=HEMODYNAMIC),
        within("extracellular", "adenosine", band=HEMODYNAMIC),
        within("metabolic", "consumption", band=HEMODYNAMIC),
    ),
    outputs=(within("blood", "flow", band=HEMODYNAMIC),),
    topology="local",
    timescale_s=1.0,
    validity=Validity(
        min_spacing_mm=0.5, max_spacing_mm=10.0, band=HEMODYNAMIC,
        note="below roughly 0.5 mm the territory of a single penetrating arteriole is "
             "smaller than the cell being modelled and pointwise coupling is meaningless: "
             "flow is regulated over a vascular unit, not over a voxel.  above 10 mm the "
             "voxel spans several arterial territories with different transit times."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"haemodynamic", "physiological"}),
    notes="reads spectral neural state and writes scalar blood state; the registered "
          "spectral -> scalar conversion applies and is recorded in provenance.",
)

implementation(
    name="balloon_lti",
    process="neurovascular_coupling",
    doc="""friston's linearized flow-inducing signal: two lags behind an onset delay.

    valid for the small flow changes evoked activation actually produces -- the
    linearization is taken about resting cbf and the error is second order in
    the fractional flow change.  it is wrong for hypercapnia, for seizure, and
    for direct stimulation-driven hyperaemia, all of which drive flow far enough
    that the neglected quadratic terms dominate and the response ceases to be a
    fixed-shape kernel at all.""",
    form=Form.LTI,
    transfer=nvc_transfer,
    params={
        "efficacy": weak(0.5, 4.0, units="fractional flow change per unit drive",
                         note="neural efficacy eps is the least identifiable parameter in the "
                              "balloon model; it absorbs every unit convention upstream of it"),
        "tau_signal_s": lognormal(1.54, 1.6, units="s", provenance=Provenance.LITERATURE,
                                  source="friston 2000 nonlinear responses in fmri",
                                  note="signal decay 1/kappa, kappa ~ 0.65 1/s"),
        "tau_feedback_s": lognormal(2.44, 1.6, units="s", provenance=Provenance.LITERATURE,
                                    source="friston 2000", note="autoregulatory feedback 1/gamma"),
        "onset_lag_s": lognormal(1.0, 1.5, units="s", provenance=Provenance.LITERATURE,
                                 source="fmri hrf onset latency, 1-2 s across cortex"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="friston et al 2000; buxton et al 1998",
)

implementation(
    name="astrocyte_vasoactive",
    process="neurovascular_coupling",
    doc="""a messenger-cascade form with an explicit saturating dilatory reserve.

    the reason to carry this alongside the lti version is that it can be wrong
    in the direction the physiology is actually wrong in: flow response
    saturates, and the same drive delivered to an already-dilated bed does much
    less.  the mechanism attribution -- astrocytic Ca and K+ siphoning onto
    smooth muscle -- is contested, so the pathway parameters are speculative
    even though the saturation is not.""",
    form=Form.RATE,
    transfer=vasoactive_transfer,
    params={
        "tau_rise_s": lognormal(0.9, 2.0, units="s", provenance=Provenance.LITERATURE,
                                source="astrocytic calcium to arteriolar dilation latency"),
        "dilatory_reserve": normal(1.8, 0.4, units="fraction of baseline cbf",
                                   provenance=Provenance.LITERATURE,
                                   source="cerebrovascular reserve, acetazolamide and co2 challenge",
                                   note="peak achievable cbf is roughly twice baseline"),
        "k_potassium_sensitivity": speculative(1.0, 20.0, units="per mM",
                                               note="the sign flips above ~20 mM extracellular K+; "
                                                    "this form does not represent that"),
        "adenosine_gain": weak(0.3, 8.0, units="fractional flow per uM"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.WEAK,
)

implementation(
    name="learned_nvc",
    process="neurovascular_coupling",
    doc="""a learned map from the retained neural spectrum to the flow response.

    this is the implementation that can express what the lti one structurally
    cannot: that the flow response depends on how activity is distributed across
    frequency and not only on its mean power.  it does so by reading the spectrum
    *before* the scalar collapse, so the conversion still happens -- but on the
    output of a learned function rather than on the input to a fixed kernel.
    the price is that uncertainty propagation is no longer exact, and the
    ensemble width it needs is a real cost.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0, note="weight prior; a learned f is a different p(theta), "
                                           "not a different declaration"),
        "residual_gain": weak(0.3, 5.0, note="learned correction is parameterized as a residual on "
                                             "the balloon prediction so the prior stays physical"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


# ---------------------------------------------------------------------------
# vascular flow
# ---------------------------------------------------------------------------

VASCULAR_FLOW = process(
    id="vascular_flow",
    doc="""transport along the vascular tree: flow, pressure, volume and the
    oxygen the blood is still carrying when it arrives.

    the topology matters more here than the dynamics.  two capillaries a
    hundred microns apart in space may be several centimetres apart along the
    tree, so this process is declared over the vascular topology and nothing
    about it is a spatial neighbourhood operator.  arterial territories are
    watersheds, and a stroke is a statement about this graph rather than about
    euclidean distance.

    it also advects deoxyhaemoglobin and oxygen content downstream, which is
    what closes the loop with `tissue_exchange`: extraction happens at the
    capillary, and the venous compartment an mr scanner is actually sensitive to
    sees the result one transit time later.""",
    inputs=(
        within("blood", "flow", "pressure", "volume", band=HEMODYNAMIC),
        within("blood", "oxygen_content", "deoxyhemoglobin", band=HEMODYNAMIC),
        within("material", "stiffness", band=Band(0.0, 0.001)),
    ),
    outputs=(
        within("blood", "flow", "pressure", "volume", band=HEMODYNAMIC),
        within("blood", "oxygen_content", "deoxyhemoglobin", band=HEMODYNAMIC),
    ),
    topology="vascular",
    timescale_s=1.0,
    validity=Validity(
        min_spacing_mm=0.05, max_spacing_mm=20.0, band=HEMODYNAMIC,
        note="a lumped compartment per tree segment; below ~50 um the segment is a single "
             "capillary and a compliance-and-resistance description of it is a category error."),
    provenance=Provenance.PHYSICS,
    tags=frozenset({"haemodynamic", "transport"}),
)

implementation(
    name="windkessel_lti",
    process="vascular_flow",
    doc="""compliant-compartment transit with myogenic autoregulation closed around it.

    linear, so exact at any timestep and free of the stiffness that makes
    explicit vascular network integration expensive.  it holds while resistance
    is roughly constant, which is to say for ordinary activation.  it fails
    wherever resistance is the thing that changed: severe hypotension below the
    autoregulatory knee, vasospasm, stenosis, and any bed where flow has become
    pressure-passive.""",
    form=Form.LTI,
    transfer=windkessel_transfer,
    params={
        "tau_transit_s": lognormal(3.0, 1.5, units="s", provenance=Provenance.LITERATURE,
                                   source="buxton 1998 balloon model; mean transit time 2-4 s",
                                   note="grey matter capillary-to-venule mean transit time"),
        "tau_compliance_s": lognormal(8.0, 2.0, units="s", provenance=Provenance.LITERATURE,
                                      source="myogenic autoregulatory time constant"),
        "autoregulation_gain": normal(0.6, 0.25, units="dimensionless",
                                      provenance=Provenance.LITERATURE,
                                      note="static autoregulation holds cbf near constant over "
                                           "roughly 60-150 mmHg mean arterial pressure"),
        "grubb_exponent": normal(0.38, 0.06, units="dimensionless",
                                 provenance=Provenance.LITERATURE,
                                 source="grubb et al 1974",
                                 note="cbv = cbf^alpha; measured in hypercapnia, and the venous-only "
                                      "value under activation is nearer 0.2"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="grubb 1974; buxton 1998",
)

implementation(
    name="poiseuille_network",
    process="vascular_flow",
    doc="""resistance-network flow with a nonlinear pressure-volume wall law.

    the form to select when the question is about the vasculature rather than
    about the cortex: resistance scales as the fourth power of radius, so this
    is where a stenosis, a vasospasm or a modelled occlusion has anywhere to
    live.  costs a nonlinear solve per step and does not preserve gaussianity.""",
    form=Form.RATE,
    params={
        "blood_viscosity_pa_s": lognormal(3.5e-3, 1.3, units="Pa.s",
                                          provenance=Provenance.LITERATURE,
                                          note="whole blood at 37 C; falls in small vessels "
                                               "(fahraeus-lindqvist), which this ignores"),
        "baseline_cbf": normal(50.0, 10.0, units="mL/100g/min",
                               provenance=Provenance.LITERATURE,
                               source="grey matter resting cbf; white matter is nearer 20"),
        "baseline_cbv": normal(0.04, 0.01, units="mL/mL tissue",
                               provenance=Provenance.LITERATURE,
                               note="~4% of grey matter volume, roughly a third of it venous"),
        "wall_stiffness_scale": weak(1.0, 5.0, units="dimensionless"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.PHYSICS,
)


# ---------------------------------------------------------------------------
# tissue exchange
# ---------------------------------------------------------------------------

TISSUE_EXCHANGE = process(
    id="tissue_exchange",
    doc="""the capillary trade: oxygen and glucose out of the blood, carbon
    dioxide and deoxyhaemoglobin back into it.

    declared over the metabolic-exchange topology rather than a spatial one
    because what may exchange with what is set by which capillary perfuses which
    tissue, and that incidence is not a distance.

    the central nonlinearity is that extraction is not a constant.  when flow
    rises faster than consumption, the extraction fraction falls and venous
    blood ends up *more* oxygenated than at rest -- which is the entire physical
    basis of positive bold contrast, and the reason a linear exchange model
    quietly gets the sign of the thing it is supposed to explain from luck
    rather than from mechanism.""",
    inputs=(
        within("blood", "flow", "oxygen_content", band=HEMODYNAMIC),
        within("metabolic", "consumption", "oxygen", "glucose", band=HEMODYNAMIC),
        within("extracellular", "ph", band=HEMODYNAMIC),
    ),
    outputs=(
        within("metabolic", "oxygen", "glucose", band=HEMODYNAMIC),
        within("blood", "oxygenation", "deoxyhemoglobin", band=HEMODYNAMIC),
    ),
    topology="metabolic_exchange",
    timescale_s=2.0,
    validity=Validity(
        min_spacing_mm=0.1, max_spacing_mm=10.0, band=HEMODYNAMIC,
        note="single-compartment extraction assumes the capillary bed within a cell is "
             "well mixed; at 10 mm it is not, and heterogeneity of transit time raises "
             "effective extraction above what the mean flow predicts."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"haemodynamic", "metabolic"}),
)

implementation(
    name="fick_oxygen_limitation",
    process="tissue_exchange",
    doc="""buxton-frank oxygen limitation: E = 1 - (1 - E0)^(1/f).

    the nonlinear form, and the one worth defending.  it says that a capillary
    is diffusion-limited rather than flow-limited, so doubling flow does not
    double delivery -- extraction falls to compensate.  it breaks where tissue
    oxygen tension is no longer negligible against plasma, which is exactly the
    ischaemic regime where the answer matters most.""",
    form=Form.RATE,
    params={
        "oxygen_extraction_fraction": normal(0.40, 0.06, units="dimensionless",
                                             provenance=Provenance.LITERATURE,
                                             source="pet oef, resting grey matter 0.3-0.5",
                                             note="remarkably uniform across cortex at rest, which "
                                                  "is itself the evidence for flow-metabolism coupling"),
        "arterial_oxygen_content": normal(0.20, 0.02, units="mL O2 / mL blood",
                                          provenance=Provenance.LITERATURE,
                                          note="~200 mL/L at normal haemoglobin and saturation"),
        "capillary_permeability_surface": weak(1.0, 4.0, units="mL/100g/min",
                                               note="PS product; lumped and poorly identified "
                                                    "separately from transit time"),
        "glucose_transport_km": lognormal(5.0, 1.5, units="mM", provenance=Provenance.LITERATURE,
                                          source="GLUT1 half-saturation at the blood-brain barrier"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="buxton & frank 1997",
)

implementation(
    name="linearized_exchange_lti",
    process="tissue_exchange",
    doc="""first-order washout of deoxyhaemoglobin about the resting operating point.

    the cheap form, and honest about being cheap: it is the derivative of the
    oxygen-limitation law at rest, so it agrees with it for flow changes of a
    few percent and diverges monotonically after that.  select it when the
    exchange step is not the thing under study and its cost is not worth
    paying.""",
    form=Form.LTI,
    transfer=lambda basis, tau_washout_s=3.0: low_pass(basis, tau_washout_s),
    params={
        "tau_washout_s": lognormal(3.0, 1.5, units="s", provenance=Provenance.LITERATURE,
                                   note="venous deoxyhaemoglobin washout, set by transit time"),
        "coupling_ratio_n": normal(2.5, 0.7, units="dimensionless",
                                   provenance=Provenance.LITERATURE,
                                   source="cbf-cmro2 coupling ratio under activation, reported 2-4",
                                   note="flow rises about 2.5 times as much as oxygen consumption; "
                                        "this ratio is the whole of positive bold contrast"),
    },
    tying=Tying.GLOBAL,
    provenance=Provenance.LITERATURE,
)


# ---------------------------------------------------------------------------
# bold formation
# ---------------------------------------------------------------------------

BOLD_FORMATION = process(
    id="bold_formation",
    doc="""deoxyhaemoglobin and blood volume perturb the static magnetic field,
    and that perturbation is what an mr scanner measures.

    this is an algebraic map, not a dynamical one: given the blood state at an
    instant, the field perturbation is determined.  it is declared as
    Form.CONSTRAINT for that reason -- the stiff limit of pressure, per
    ARCHITECTURE.md §4, and a question of how it is solved rather than of what
    it is.  all of the dynamics an fmri time series exhibits were already
    contributed by `neurovascular_coupling`, `vascular_flow` and
    `tissue_exchange` upstream; nothing here has a time constant.

    the observable it writes needs a note.  docs/CONTRACT.md declares no
    mr-signal component, and the physically correct target is the mesoscopic
    static-field perturbation caused by the susceptibility of deoxyhaemoglobin
    -- which *is* a component: `electromagnetic.bfield`.  the scanner's
    T2*-weighted magnitude is the dephasing that perturbation produces.  writing
    it there rather than inventing an `mr.signal` component keeps one
    electromagnetic field in the ontology instead of two, and lets an fmri
    observation attach to the same variable an meg observation attaches to, at a
    different band.  the material read is `material.mass_density` standing in
    for tissue composition and proton density; the contract carries no magnetic
    susceptibility component and this is the nearest honest proxy.""",
    inputs=(
        within("blood", "deoxyhemoglobin", "volume", "oxygenation", band=HEMODYNAMIC),
        within("material", "mass_density", band=Band(0.0, 0.001)),
    ),
    outputs=(
        within("electromagnetic", "bfield", region=OnSupport("head_volume"),
               band=HEMODYNAMIC),
    ),
    topology="local",
    timescale_s=6.0,
    validity=Validity(
        min_spacing_mm=0.4, max_spacing_mm=8.0, band=HEMODYNAMIC,
        note="a voxel-scale susceptibility average.  below ~0.4 mm the intravascular and "
             "extravascular compartments must be separated and the single-pool expression "
             "is wrong; above ~8 mm partial volume with csf and large draining veins "
             "dominates the signal it produces."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"haemodynamic", "observable"}),
    notes="scalar blood inputs, spectral electromagnetic output: the registered "
          "scalar -> spectral conversion applies, lifting the belief into DC.",
)

implementation(
    name="davis_deoxy_constraint",
    process="bold_formation",
    doc="""the algebraic bold expression: signal change as a power law in venous
    blood volume and deoxyhaemoglobin content.

    ``dS/S = V0 * (k1*(1-q) + k2*(1 - q/v) + k3*(1-v))`` in the buxton form, or
    the davis calibrated form with alpha and beta.  the reason to prefer this
    over a fixed kernel is that it retains the separate dependence on volume and
    on deoxyhaemoglobin, so calibrated experiments -- hypercapnia, hyperoxia --
    have parameters to attach to.

    where it breaks: k1, k2 and k3 are field-strength and sequence dependent,
    the values below are for 3 T gradient echo, and the extravascular term
    scales roughly as B0^2 while the intravascular term saturates.  transplanting
    these numbers to 7 T is a mistake this declaration cannot prevent but does
    at least record.""",
    form=Form.CONSTRAINT,
    params={
        "k1": normal(4.2, 0.8, units="dimensionless", provenance=Provenance.LITERATURE,
                     source="3 T gradient echo, k1 = 4.3*nu0*E0*TE with nu0 ~ 40.3 Hz"),
        "k2": normal(1.7, 0.5, units="dimensionless", provenance=Provenance.LITERATURE,
                     note="intravascular term; saturates with field strength"),
        "k3": normal(0.4, 0.3, units="dimensionless", provenance=Provenance.LITERATURE),
        "resting_venous_volume": normal(0.03, 0.01, units="mL/mL",
                                        provenance=Provenance.LITERATURE,
                                        note="V0; the single largest source of between-voxel gain "
                                             "variance in fmri and almost never measured"),
        "davis_alpha": normal(0.38, 0.06, units="dimensionless",
                              provenance=Provenance.LITERATURE, source="grubb et al 1974"),
        "davis_beta": normal(1.5, 0.3, units="dimensionless", provenance=Provenance.LITERATURE,
                             source="davis et al 1998 calibrated bold; 1.3-1.5 at 3 T"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.LITERATURE,
    source="buxton et al 1998; davis et al 1998; obata et al 2004",
)

implementation(
    name="spm_double_gamma_hrf",
    process="bold_formation",
    doc="""the canonical double-gamma hrf, collapsed into one lti kernel.

    peak at 6 s, undershoot at 16 s, peak-to-undershoot amplitude ratio 6, unit
    dispersion -- the spm defaults, and by a wide margin the most-used forward
    model in human neuroimaging.  it is offered here as an alternative f for the
    same (I, O, T): it folds the balloon dynamics, the exchange washout and the
    susceptibility map into a single fixed kernel, so selecting it collapses
    three declared processes into one and the materialization's provenance says
    so.

    it is worth being blunt about the cost.  hrf shape is not a constant.  it
    varies systematically across cortex -- time to peak differs by a second or
    more between regions of the same subject -- across subjects, with age, with
    caffeine, with baseline vascular state, and between grey matter and a
    draining vein within the same voxel.  treating it as fixed while comparing
    groups whose vasculature differs converts a haemodynamic difference into an
    apparent neural one, and that is a documented and recurring source of
    spurious fmri findings rather than a theoretical worry.  the free-shape
    alternative below exists because of this.""",
    form=Form.LTI,
    transfer=spm_double_gamma_transfer,
    params={
        "peak_s": normal(6.0, 1.0, units="s", provenance=Provenance.LITERATURE,
                         source="spm canonical hrf; friston et al 1998",
                         note="the sd is the across-cortex spread, not a measurement error"),
        "undershoot_s": normal(16.0, 2.0, units="s", provenance=Provenance.LITERATURE,
                               source="spm canonical hrf"),
        "ratio": normal(6.0, 1.5, units="dimensionless", provenance=Provenance.LITERATURE,
                        source="spm canonical hrf peak-to-undershoot ratio"),
        "dispersion_s": lognormal(1.0, 1.3, units="s", provenance=Provenance.LITERATURE,
                                  note="gamma scale; spm's dispersion derivative exists precisely "
                                       "because this is not constant either"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="friston et al 1998; spm12 spm_hrf.m defaults",
)

implementation(
    name="learned_free_shape_hrf",
    process="bold_formation",
    doc="""a per-site learned response shape with the canonical hrf as its prior mean.

    the point is not flexibility for its own sake.  it is that hrf variability is
    real and structured, so a model that cannot represent it must attribute it to
    the only thing it can represent, which is neural amplitude.  giving the shape
    parameters somewhere to go makes the confound visible in the posterior
    instead of laundering it into the effect of interest.  the posterior stays
    near the canonical prior wherever the data cannot distinguish shapes, which
    is the intended behaviour rather than a failure.""",
    form=Form.LEARNED,
    params={
        "shape_prior_scale": weak(1.0, 3.0, note="deviation from the canonical basis"),
        "latency_shift_s": normal(0.0, 1.0, units="s", provenance=Provenance.LITERATURE,
                                  note="across-cortex time-to-peak variability, roughly +/- 1 s"),
        "vein_fraction": uniform(0.0, 1.0, units="dimensionless", provenance=Provenance.WEAK,
                                 note="how much of the voxel's signal comes from a draining vein "
                                      "rather than from the tissue under it"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.WEAK,
)

__all__ = [
    "NEUROVASCULAR_COUPLING",
    "VASCULAR_FLOW",
    "TISSUE_EXCHANGE",
    "BOLD_FORMATION",
]
