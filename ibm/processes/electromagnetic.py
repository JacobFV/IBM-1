"""the electromagnetic field: what the tissue's currents make, and what a field
does back to the tissue that made it.

two processes and one physical approximation holding both of them up.

at the frequencies neural tissue produces -- DC to a few kilohertz -- the head
is a quasi-static conductor.  the wavelength at 1 kHz in tissue is on the order
of kilometres, so propagation delay across a head is unmeasurable; inductive
coupling scales with omega and is negligible at these frequencies; and the ratio
of displacement current to conduction current, ``omega * epsilon / sigma``, is
about 1.7% at 1 kHz in grey matter and falls linearly below that.  what is left
is a Poisson equation with no time in it at all: the potential at every point is
an instantaneous linear functional of the current sources.

that is why `em_generation` is declared `Form.CONSTRAINT`.  ARCHITECTURE.md §4
is explicit that an instantaneous algebraic relation ``x_O = g(x_I)`` -- a
quasi-static field determined by its source currents is its named example -- is
the stiff limit of pressure,

    dx_O += -gamma (x_O - g(x_I)),   gamma -> inf

and therefore a question of how it is solved rather than of what it is.
declaring it as a constraint says the potential has no dynamics of its own and
that a solver may treat it algebraically; it does not make it a different kind
of object from every other process in the package.

the same fact stated in the spectral form: the lead field is frequency-flat, so
its transfer function is `base.constant_gain` -- an LTI coupling whose H(omega)
happens to have no frequency dependence.  the constraint and the flat transfer
function are the same statement, which is the point.

where the flatness fails is written out in `quasistatic_lead_field` and is worth
previewing, because it is not hypothetical: capacitive tissue effects become
percent-level in the spike band and are no longer negligible for transcranial
stimulation waveforms with sharp edges, and above roughly 10 kHz the tissue is
frankly dispersive.  a materialization retaining the spike band with a flat lead
field is making an approximation it should be told it is making, which is why
the validity band below stops where it does.
"""

from __future__ import annotations

import numpy as np

from ibm.processes.base import (
    constant_gain,
    implementation,
    low_pass,
    process,
    series,
)
from ibm.registry import Form
from ibm.vocabulary import (
    Band,
    LFP,
    STRUCTURAL,
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

#: the band over which the field is generated and read.  it reaches into the
#: spike band because a microelectrode genuinely records there; the validity
#: note on `em_generation` says what the flat lead field costs up there.
FIELD = Band(0.0, 5000.0)

#: the band over which an applied or endogenous field can plausibly affect a
#: membrane.  the ceiling is the membrane's own low-pass corner, not a
#: modelling convenience: at 15 ms a membrane is already 20 dB down by 100 Hz.
POLARIZING = Band(0.0, 300.0)


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def lead_field_transfer(basis, lead_field_gain: float = 1.0):
    """current source density -> potential, frequency-flat.

    `base.constant_gain`, and the whole physical content of the quasi-static
    approximation is that this is the right answer: no pole, no delay, no
    dependence on omega whatsoever.  the *spatial* structure -- which source
    contributes how much to which sensor -- is entirely in the gain, which the
    electromagnetic topology supplies per source-sensor pair from a head model.

    the practical payoff of writing it this way rather than as a special
    algebraic case is that it composes with everything else in the LTI library
    without any adapter: a lead field followed by an amplifier's bandpass is
    `series(constant_gain, ...)`, and the fact that one of the two has no
    dynamics is not a structural distinction.
    """
    return constant_gain(basis, lead_field_gain)


def admittivity_transfer(basis, lead_field_gain: float = 1.0,
                         conductivity_s_m: float = 0.33,
                         relative_permittivity: float = 1e5):
    """the same lead field with the tissue's displacement current restored.

    the complex admittivity of tissue is ``sigma + i omega epsilon``, so the
    lead field is really ``g / (1 + i omega epsilon / sigma)`` -- a single pole
    at the charge relaxation frequency ``sigma / (2 pi epsilon)``.  with grey
    matter's conductivity of 0.33 S/m and a relative permittivity around 10^5
    in the kilohertz range, that corner sits near 60 kHz, which is why the flat
    approximation is excellent through the whole physiological band and why
    this implementation is a diagnostic rather than a default.

    it is not always a diagnostic.  three cases where the pole matters.  the
    spike band, where the correction is percent-level and phase-shifting and
    where high-density array work is beginning to be precise enough to care.
    transcranial stimulation with sharp-edged waveforms, whose spectral content
    extends far above the band the flat model was validated in.  and skull,
    whose permittivity dispersion is much stronger than brain's and which is
    the dominant unknown in every eeg forward model anyway.
    """
    tau_s = relative_permittivity * 8.854e-12 / max(conductivity_s_m, 1e-9)
    return low_pass(basis, tau_s, lead_field_gain)


def membrane_polarization_transfer(basis, polarization_mv_per_v_m: float = 0.25,
                                   tau_membrane_s: float = 0.015):
    """applied electric field -> membrane potential change.

    a gain and a membrane pole, and both halves are load-bearing.

    the gain is small and measured: a uniform DC field of 1 V/m polarizes a
    pyramidal soma by roughly 0.2 mV, which is under a tenth of the distance
    to threshold.  that is why transcranial stimulation at realistic
    intracranial field strengths -- a fraction of a volt per metre -- cannot
    fire a quiescent neuron and can only bias the timing of one that was going
    to fire anyway.  a model that gets this gain wrong by an order of magnitude
    will happily predict that tACS drives cortex, which is the single most
    common failure mode of stimulation modelling.

    the pole is the membrane's own low-pass, and it is why the effect falls off
    with frequency: with a 15 ms time constant a 100 Hz field polarizes the
    membrane about 20 dB less than a DC field of the same strength.  the
    frequency dependence of stimulation efficacy is therefore predicted rather
    than fitted, and it predicts that high-frequency stimulation must work by
    some other mechanism -- rectification, or an effect on something other than
    somatic membrane potential -- if it works at all.

    where it breaks: the polarization gain depends on the angle between the
    field and the cell's dendritic axis, and it is not one number but a
    distribution over a population whose axes are aligned with the cortical
    normal to varying degrees.  in a gyral crown that alignment is good and in a
    sulcal wall it is not, so a single gain per position is a mean over a
    geometry that a folded cortical surface makes highly heterogeneous.
    """
    return series(
        constant_gain(basis, polarization_mv_per_v_m),
        low_pass(basis, tau_membrane_s),
    )


# ---------------------------------------------------------------------------
# nonlinear rate laws
# ---------------------------------------------------------------------------


def ephaptic_entrainment(x, theta) -> dict:
    """field feedback onto spike timing, as a phase bias rather than a drive.

    the honest form for a sub-threshold effect on a population that is already
    oscillating: a field too weak to fire anything can still shift the phase at
    which cells that were going to fire do so, and the effect on population
    rate is therefore proportional to how steeply the rate depends on phase --
    which is to say, to how synchronous the population already was.

    that state dependence is the whole content, and it is why a linear
    polarization term underestimates the effect during synchronous activity and
    overestimates it during asynchronous activity.  the numbers are weak: the
    existence of ephaptic entrainment is established in slice at field strengths
    around 1 mV/mm, and its magnitude in intact human cortex at endogenous
    field strengths is genuinely unknown.
    """
    e = np.asarray(x["electromagnetic.efield"], dtype=float)
    r = np.asarray(x["neural.exc.activity"], dtype=float)
    gain = float(theta.get("ephaptic_gain_hz_per_mv_mm", 0.5))
    synchrony = float(theta.get("synchrony_scale", 1.0))
    return {"neural.exc.activity": gain * e * (1.0 + synchrony * np.abs(r) / (1.0 + np.abs(r)))}


# ---------------------------------------------------------------------------
# em generation
# ---------------------------------------------------------------------------

EM_GENERATION = process(
    id="em_generation",
    doc="""transmembrane current sources plus the head's material properties
    determine the potential, the electric field, the current density and the
    magnetic field, instantaneously.

    the input is `neural.transmembrane_current` and not firing rate, and that
    is the substantive claim of this declaration.  the extracellular field is
    the volume-conducted sum of transmembrane currents weighted by a geometry
    that depends on dendritic morphology and synapse placement: the same
    population at the same firing rate produces opposite-sign dipoles depending
    on whether its input arrives at apical dendrites or at somata, because the
    sink and source swap ends.  using rate as a proxy for the source is the
    standard way a forward model quietly stops being physics, and putting the
    current in the declaration is how the ontology refuses to allow it.

    material state is an input rather than a constant because the forward model
    is dominated by it.  the skull's conductivity is the single largest source
    of error in eeg source localization and its ratio to brain conductivity is
    quoted anywhere between 1:15 and 1:80 in the literature; csf, at 1.79 S/m,
    shunts current in ways that a model without it gets qualitatively wrong.
    those are uncertain parameters that evidence should be able to move, and
    they can only be if they are declared.

    the magnetic field is here alongside the electric one because it comes from
    the same sources and differs in exactly one useful respect: biological
    tissue has the magnetic permeability of free space, so the skull is
    magnetically transparent and the meg forward model does not depend on the
    conductivity boundaries that dominate the eeg one.  it also cannot see a
    radial dipole in a spherically symmetric conductor at all, which is a
    structural blindness rather than a sensitivity limit -- and stating that in
    the same process as the potential is how a materialization gets to know
    that the two modalities are complementary rather than redundant.

    `Form.CONSTRAINT` on the implementations is the stiff limit of pressure,
    per ARCHITECTURE.md §4: there is no field state to relax, so the field is
    solved for rather than integrated.  it remains a process with inputs,
    outputs, a topology and uncertain parameters like any other.

    where it breaks.  the quasi-static approximation is excellent below about
    1 kHz and degrades smoothly above it; the flat lead field is percent-level
    wrong in the spike band and worse for stimulation waveforms with sharp
    edges.  more importantly the lead field is a *linear* functional of the
    sources with fixed geometry, and the geometry is a subject's own -- skull
    thickness, csf depth, cortical folding -- so a template head model carries
    localization errors of a centimetre or more that no parameter in this
    process can absorb.""",
    inputs=(
        within("neural", "transmembrane_current", band=FIELD),
        within("material", "conductivity", "permittivity", band=STRUCTURAL),
        within("structural", "fiber_orientation", band=STRUCTURAL),
    ),
    outputs=(
        within("electromagnetic", "potential", "efield", band=FIELD),
        within("electromagnetic", "current_density", "bfield", band=FIELD),
    ),
    topology="electromagnetic",
    timescale_s=1e-4,
    validity=Validity(
        min_spacing_mm=0.5, max_spacing_mm=10.0, band=LFP,
        note="the validated band stops at the top of the lfp range because that is where "
             "the frequency-flat lead field is defensible; materializing the spike band "
             "with these implementations is a percent-level approximation and the "
             "materializer should record it.  below ~0.5 mm the source is a single "
             "neuron's dendritic tree and a point current-source-density description of it "
             "is a category error; above ~10 mm cancellation between opposed cortical banks "
             "within one cell dominates the answer, and the cancellation is a geometric "
             "fact the coarse model cannot see.",
    ),
    provenance=Provenance.PHYSICS,
    tags=frozenset({"electromagnetic", "forward", "quasi_static"}),
    notes="declared as a constraint: instantaneous, algebraic, no field state to integrate.  "
          "device-driven fields -- stimulation currents and coil-induced fields -- enter the "
          "same components through device_coupling rather than through this process.",
)

implementation(
    name="quasistatic_lead_field",
    process="em_generation",
    doc="""the frequency-flat lead field: a linear map from sources to sensors with
    no dynamics.

    this is the workhorse and it is physics, not a fit.  the approximation is
    quantitative and worth carrying explicitly: displacement current is
    ``omega * epsilon / sigma`` of conduction current, which in grey matter at
    1 kHz is about 1.7%, at 100 Hz about 0.2%, and negligible below that;
    inductive effects scale with omega and are smaller still; propagation across
    the head is instantaneous on any timescale a brain produces.

    where the flatness fails, in the order it starts to matter.  the spike
    band, where the capacitive correction reaches percent level and shifts
    phase -- relevant for high-density array recordings and irrelevant for eeg.
    transcranial stimulation, whose square and pulsed waveforms carry energy
    far above the band the approximation was validated in, and where the tissue
    is being driven rather than observed.  and the skull specifically, whose
    permittivity dispersion is far stronger than brain's, which is where a
    capacitive correction would first become visible in a scalp recording.

    the conductivities carry literature provenance because they are measured;
    the skull value carries a wide prior because the measurements disagree with
    each other far more than any of them reports as its own uncertainty.""",
    form=Form.CONSTRAINT,
    transfer=lead_field_transfer,
    params={
        "conductivity_grey_s_m": normal(0.33, 0.05, units="S/m", provenance=Provenance.LITERATURE,
                                        source="grey matter conductivity at low frequency, "
                                               "~0.33 S/m",
                                        note="isotropic; white matter is anisotropic by a "
                                             "factor of ~9 along versus across fibres, which "
                                             "is what the fiber_orientation input is for"),
        "conductivity_csf_s_m": normal(1.79, 0.1, units="S/m", provenance=Provenance.LITERATURE,
                                       source="cerebrospinal fluid, 1.79 S/m at body "
                                              "temperature",
                                       note="five times grey matter, so csf shunts current "
                                            "strongly; a forward model without a csf "
                                            "compartment is wrong in a way no conductivity "
                                            "value repairs"),
        "conductivity_skull_s_m": lognormal(0.01, 2.5, units="S/m",
                                            provenance=Provenance.LITERATURE,
                                            source="skull conductivity; reported values span "
                                                   "0.003-0.03 S/m",
                                            note="the brain-to-skull ratio is quoted from 1:15 "
                                                 "to 1:80 and is the single largest source of "
                                                 "eeg localization error.  the wide prior is "
                                                 "the disagreement, not measurement noise"),
        "conductivity_scalp_s_m": normal(0.43, 0.08, units="S/m",
                                         provenance=Provenance.LITERATURE),
        "lead_field_gain": weak(1.0, 5.0, units="uV per nA.mm^-3",
                                note="the per-pair geometric factor; supplied by the head "
                                     "model through the electromagnetic topology rather than "
                                     "fitted, and weak here because a template head model's "
                                     "value for a given subject is an extrapolation"),
        "dipole_orientation_alignment": uniform(0.0, 1.0, units="dimensionless",
                                                provenance=Provenance.WEAK,
                                                note="how well the population's dendritic axes "
                                                     "align with the cortical normal within a "
                                                     "materialized position; cancellation "
                                                     "between opposed sulcal banks lives here"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.PHYSICS,
    source="nunez & srinivasan 2006 electric fields of the brain; "
           "hamalainen et al 1993 magnetoencephalography",
)

implementation(
    name="capacitive_admittivity_lti",
    process="em_generation",
    doc="""the lead field with the displacement-current pole restored.

    carried for exactly one purpose: to make the quasi-static approximation
    checkable rather than assumed.  a materialization that runs both and finds
    no difference has *demonstrated* that the flat lead field is adequate for
    its band and its question, which is a better epistemic position than
    asserting it.  one that finds a difference has found the place where the
    default was wrong.

    declared LTI rather than CONSTRAINT because it genuinely has a pole and
    therefore genuinely has dynamics, unlike the flat version -- the difference
    between the two forms is the difference between the two declarations, which
    is the point of keeping them apart.

    the permittivity prior is enormous and honestly so.  tissue relative
    permittivity at low frequency is dominated by interfacial polarization and
    is reported anywhere from 10^4 to 10^7 depending on frequency, tissue and
    method, with the frequency dependence itself being the phenomenon.  a
    single number is a placeholder for a dispersion relation.""",
    form=Form.LTI,
    transfer=admittivity_transfer,
    params={
        "conductivity_s_m": normal(0.33, 0.05, units="S/m", provenance=Provenance.LITERATURE),
        "relative_permittivity": lognormal(1e5, 10.0, units="dimensionless",
                                           provenance=Provenance.LITERATURE,
                                           source="gabriel et al 1996 dielectric properties of "
                                                  "biological tissues",
                                           note="spans three orders of magnitude across the "
                                                "physiological band; the spread here is the "
                                                "dispersion, not uncertainty about a constant"),
        "lead_field_gain": weak(1.0, 5.0, units="uV per nA.mm^-3"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.PHYSICS,
    source="gabriel, gabriel & corthout 1996",
)

implementation(
    name="learned_lead_field",
    process="em_generation",
    doc="""a learned residual on the analytic lead field, conditioned on individual
    head geometry.

    the argument is not that the physics is wrong -- it is not -- but that the
    *geometry* the physics is applied to is a subject's own and is known only
    approximately.  skull thickness and its regional variation, csf depth,
    cortical folding and the anisotropy of white matter all enter the forward
    map, and a template head model gets them wrong in a spatially structured
    way that produces localization bias rather than noise.  a learned residual
    over a geometric embedding can absorb that structure where individual mri
    is available, and collapses to the analytic lead field where it is not.

    what it must not be allowed to absorb is source amplitude, which is the
    quantity of interest downstream.  parameterizing it as a residual with a
    prior mean of zero and a tight scale is the mechanism for that; the
    posterior on `residual_gain` is the report of how much geometry the
    template was missing.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0, note="prior centred on the analytic lead field"),
        "residual_gain": weak(0.2, 4.0,
                              note="kept tighter than elsewhere in this package on purpose: a "
                                   "flexible forward model absorbs the source amplitudes it "
                                   "exists to estimate"),
        "geometry_sensitivity": speculative(1.0, 8.0,
                                            note="how much of the residual is explained by "
                                                 "measurable head geometry rather than by "
                                                 "unmodelled conductivity"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


# ---------------------------------------------------------------------------
# em coupling
# ---------------------------------------------------------------------------

EM_COUPLING = process(
    id="em_coupling",
    doc="""the electric field acting back on the membranes that made it, and on
    membranes that did not.

    the reverse direction of `em_generation`, and it is a genuinely separate
    process rather than the same one read backwards: the forward map is a
    volume-conduction integral over current sources and the reverse map is a
    polarization of a cell by the local field gradient along its own axis.
    they have different geometry, different units, and wildly different
    magnitudes.

    it covers two things that are usually discussed separately and are the same
    physics.  ephaptic coupling -- endogenous fields, on the order of a
    millivolt per millimetre during slow oscillations and sharp waves, feeding
    back on the population that generated them.  and transcranial stimulation --
    exogenous fields of a fraction of a volt per metre inside the head, written
    into the same `electromagnetic.efield` component by `device_coupling`.  a
    model that treats them as different mechanisms will have two sets of
    parameters for one effect and no way to use evidence about either to
    constrain the other.

    the central quantitative fact, and the reason this process is worth
    declaring carefully, is that the coupling is weak.  one volt per metre
    polarizes a pyramidal soma by roughly 0.2 mV, an order of magnitude below
    the distance to threshold, and realistic transcranial currents produce
    under a volt per metre in cortex.  so the honest prediction is that
    stimulation at these intensities biases the timing of activity that was
    going to happen and does not create activity -- and any implementation
    whose parameters imply otherwise is making a claim the measured
    polarization constant contradicts.

    where it breaks.  polarization depends on the angle between the field and
    the cell's dendritic axis, and cortical folding makes that angle vary over
    a millimetre; a single gain per position is a mean over a distribution
    whose width is comparable to its mean.  the effect is also strongly
    state-dependent in a way a linear form cannot carry: a field too weak to
    fire anything shifts the phase of an ongoing rhythm substantially, so its
    consequence for population rate depends on how synchronous the population
    already was.""",
    inputs=(
        within("electromagnetic", "efield", "potential", band=POLARIZING),
        within("electromagnetic", "current_density", band=POLARIZING),
        within("neural", "exc.potential", "exc.activity", band=POLARIZING),
        within("material", "conductivity", band=STRUCTURAL),
        within("structural", "fiber_orientation", "myelination", band=STRUCTURAL),
    ),
    outputs=(
        within("neural", "exc.potential", "inh.potential", band=POLARIZING),
        within("neural", "exc.activity", band=POLARIZING),
    ),
    topology="electromagnetic",
    timescale_s=1e-3,
    validity=Validity(
        min_spacing_mm=0.5, max_spacing_mm=10.0, band=Band(0.0, 300.0),
        note="the band ceiling is the membrane's own corner: at a 15 ms time constant a "
             "field above ~100 Hz polarizes almost nothing, so an implementation reporting "
             "a large effect at kilohertz stimulation frequencies is reporting a "
             "mechanism this process does not contain.  below ~0.5 mm the polarization of "
             "an individual cell depends on its own morphology rather than on a population "
             "mean, and this form has no morphology in it.",
    ),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"electromagnetic", "ephaptic", "stimulation"}),
)

implementation(
    name="uniform_field_polarization",
    process="em_coupling",
    doc="""a measured polarization constant behind the membrane's own low-pass.

    the default, and it is close to a calibration rather than a model: the
    somatic polarization per unit uniform field is measured in slice at around
    0.2 mV per V/m, and the frequency dependence follows from the membrane time
    constant with no additional parameters.  that makes it falsifiable in a
    useful way -- it predicts that the efficacy of a stimulation waveform falls
    at 20 dB per decade above a few tens of hertz, which is a testable claim
    about a technique people use at 10 Hz, 40 Hz and 5 kHz.

    it is linear and small-signal, which is right for the field strengths in
    question and wrong for the two regimes where fields become large: direct
    cortical stimulation, and the endogenous fields of a seizure, where
    ephaptic effects are strong enough to be part of the propagation
    mechanism rather than a perturbation on it.""",
    form=Form.LTI,
    transfer=membrane_polarization_transfer,
    params={
        "polarization_mv_per_v_m": normal(0.25, 0.1, units="mV per V/m",
                                          provenance=Provenance.LITERATURE,
                                          source="somatic polarization by uniform dc fields in "
                                                 "cortical slice, ~0.15-0.3 mV per V/m",
                                          note="the number that decides whether a stimulation "
                                               "model predicts entrainment or spiking; getting "
                                               "it an order of magnitude wrong is the most "
                                               "common failure in the field"),
        "tau_membrane_s": lognormal(0.015, 1.4, units="s", provenance=Provenance.LITERATURE,
                                    source="cortical pyramidal membrane time constant, 10-20 ms",
                                    note="the same constant as in the neural processes, and it "
                                         "should be the same parameter; that it is written "
                                         "twice is a tying decision the materializer can make"),
        "axis_alignment": uniform(0.0, 1.0, units="dimensionless", provenance=Provenance.WEAK,
                                  note="cosine between the field and the population's mean "
                                       "dendritic axis; on a folded surface this varies over a "
                                       "millimetre and its distribution is as wide as its mean"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.LITERATURE,
    source="radman et al 2009; deans, powell & jefferys 2007 sensitivity of coherent "
           "oscillations to weak electric fields",
)

implementation(
    name="ephaptic_entrainment_rate",
    process="em_coupling",
    doc="""field effect as a phase bias whose consequence scales with synchrony.

    the nonlinear form, and the one that can represent what makes weak-field
    effects interesting: a field far too weak to fire a cell can substantially
    shift when an already-oscillating population fires, and the resulting change
    in population rate depends on the steepness of the rate's dependence on
    phase -- that is, on how synchronous the population already was.

    the state dependence is the entire content, so the linear form is not a
    conservative approximation of this but a qualitatively different claim:
    linear polarization says the effect is the same in synchronous and
    asynchronous cortex, and the measurements say it is not.

    the parameters are weak and stay weak.  entrainment thresholds around
    1 mV/mm are established in slice; the corresponding number for intact human
    cortex at endogenous field strengths, with realistic heterogeneity of
    dendritic orientation, is not measured and is the subject of an active and
    unresolved argument about whether transcranial stimulation does anything at
    all at conventional intensities.""",
    form=Form.RATE,
    fn=ephaptic_entrainment,
    params={
        "ephaptic_gain_hz_per_mv_mm": speculative(0.5, 20.0, units="Hz per mV/mm",
                                                  note="the magnitude in intact human cortex is "
                                                       "genuinely unknown and contested; a "
                                                       "speculative prior is the honest "
                                                       "declaration, not a placeholder"),
        "entrainment_threshold_mv_mm": lognormal(1.0, 3.0, units="mV/mm",
                                                 provenance=Provenance.LITERATURE,
                                                 source="weak-field entrainment of coherent "
                                                        "oscillations in slice, ~0.5-2 mV/mm",
                                                 note="endogenous fields during slow "
                                                      "oscillations reach this range, which is "
                                                      "why ephaptic feedback is plausible at "
                                                      "all"),
        "synchrony_scale": weak(1.0, 5.0, units="dimensionless",
                                note="how strongly the effect grows with population synchrony; "
                                     "the term that makes this form different in kind from the "
                                     "linear one"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.SPECULATIVE,
    source="frohlich & mccormick 2010 endogenous electric fields guide neocortical activity",
)

implementation(
    name="learned_field_response",
    process="em_coupling",
    doc="""a learned map from the applied field spectrum to the population response.

    worth carrying for a specific and unflattering reason: the analytic forms
    above predict that high-frequency stimulation does essentially nothing,
    and several high-frequency stimulation protocols report effects.  either
    those reports are wrong, or the mechanism is something the somatic
    polarization model does not contain -- rectification at a nonlinearity,
    conduction block, an effect on axon terminals rather than somata, or a
    peripheral route entirely.

    a learned f cannot tell those apart, and it is not being offered as an
    explanation.  what it can do is let a materialization fit the observed
    frequency dependence without silently inflating the polarization constant
    to do it -- keeping the discrepancy visible as a large learned residual
    rather than laundering it into a physical parameter that measurement says
    is small.  the prior is centred on the analytic response, so the residual's
    posterior is the measurement of the discrepancy.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "residual_gain": weak(0.3, 5.0,
                              note="zero recovers uniform_field_polarization; a large "
                                   "posterior here is a finding about the mechanism, not a "
                                   "better fit"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


__all__ = ["EM_GENERATION", "EM_COUPLING"]
