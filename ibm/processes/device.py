"""instruments, and the pathway that carries peripheral traffic inward.

two processes with, at first glance, nothing to do with each other.  they share
a file because they share a claim, and the claim is the architecture's central
one about measurement: an electrode voltage is state.  not an output, not an
observation, not a special kind of object -- `device.contact_potential` is a
state variable with a value, an uncertainty, a support and a topology to brain
state, in exactly the way a thalamic membrane potential is.  what makes it
*measured* is that an observation attaches evidence to it (see
`ibm.processes.observation`), and what makes a TMS coil *stimulating* is that an
intervention clamps `device.coil_current` (see `ibm.processes.intervention`).
the physics in between -- the lead field, the induced electric field, the
acoustic beam -- is `device_coupling`, an ordinary process, declared here.

the consequence is worth stating because it is easy to lose.  in ibm-1 there is
no forward model and no inverse problem as separate machinery.  the forward
model *is* `device_coupling` run in the direction from tissue to contact; the
inverse problem *is* inference over the state that produced it; and stimulation
is the same process run the other way.  a TMS-EEG experiment, which is normally
two incompatible pipelines bolted together, is one process graph with a clamp on
one device component and a likelihood on another.

`afferent_propagation` is here for a duller reason: it is the transport leg
between `transduction` and the brain, it is structurally a delay line exactly
like `device_coupling`'s electromagnetic leg, and putting it beside the
instrument processes keeps the two peripheral transport processes from being
scattered across four files.

one thing this file deliberately does not do is invent components.  the contract
declares eight device components and no more.  there is no optical-power
component, no magnetometer-output component, no radiometric component, so
NIRS optodes, MEG channels and optogenetic fibres are each expressed through the
nearest declared thing and the substitution is named where it happens rather
than smoothed over.
"""

from __future__ import annotations

from ibm.processes.base import (
    alpha_synapse,
    constant_gain,
    delay_dispersion,
    implementation,
    low_pass,
    process,
    pure_delay,
    resonator,
    series,
)
from ibm.registry import Form
from ibm.vocabulary import (
    Anat,
    Band,
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

# ---------------------------------------------------------------------------
# bands and regions
# ---------------------------------------------------------------------------

#: what an electrophysiological instrument actually passes.  the ceiling is the
#: anti-alias filter of a typical clinical amplifier; an intracortical system
#: goes to 10 kHz and a scalp system rarely above 500 Hz, and the difference is
#: carried by the observations rather than by this band.
INSTRUMENT = Band(0.0, 10000.0)

#: the band a scalp or subdural contact is coupled over.  the floor is not zero
#: because AC-coupled amplifiers do not pass DC, and that is a property of the
#: coupling rather than of the field.
RECORDING = Band(0.01, 1000.0)

#: stimulation drive.  a TMS pulse is a ~100 us biphasic transient, so its
#: spectral content runs to several kHz and reading it narrower would misstate
#: the induced field entirely.
STIMULATION = Band(0.0, 20000.0)

#: focused ultrasound carrier.  declared honestly rather than conveniently: a
#: 500 kHz carrier is really there, and a materialization will almost always work
#: with the pulse envelope instead.  saying so here means the envelope
#: approximation is a recorded choice rather than an unexamined one.
ULTRASOUND_CARRIER = Band(1.0e5, 2.0e6)

#: the ultrasound envelope, which is what actually couples to anything neural.
#: capped at 500 Hz because that is where `mechanical.pressure` is declared to
#: stop being meaningful, not because pulse repetition frequencies stop there --
#: they reach 1 kHz -- so the top of the envelope's real range is outside what
#: the mechanical field currently carries.
ULTRASOUND_ENVELOPE = Band(0.0, 500.0)

#: the haemodynamic band an optical instrument is sensitive over.
OPTICAL = Band(0.0, 3.0)

#: peripheral afferent traffic.
AFFERENT = Band(0.0, 1000.0)

#: tissue structure over an experiment: constant.
STRUCTURE = Band(0.0, 0.001)

SENSORS = OnSupport("sensor_array")
IMPLANT = OnSupport("implanted_array")
STIMULATOR = OnSupport("stimulator")
SCANNER = OnSupport("scanner_element")
HEAD = OnSupport("head_volume")
BODY = OnSupport("body")

THALAMUS = Anat("thalamic_nuclei", "sensory_relay")
BRAINSTEM = Anat("brainstem_nuclei", "sensory_nuclei")


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def lead_field_transfer(basis, gain: float = 1.0):
    """tissue current source -> contact potential: frequency-flat.

    the quasi-static approximation, and it is worth being precise about why it
    holds rather than treating it as a convention.  at 100 Hz in tissue the
    electromagnetic wavelength is kilometres and the capacitive component of
    tissue admittivity is small against the resistive one, so propagation delay
    and displacement current are both negligible across a head.  the field at a
    contact is therefore an *instantaneous* weighted sum of source currents, and
    the weighting -- the lead field -- has no frequency dependence at all.

    that flatness is the reason a lead field can be a matrix rather than a filter
    bank, which is most of why EEG and MEG forward modelling is tractable.  it
    stops holding above roughly 1 MHz, which no neural signal reaches, and it
    quietly stops holding for a stimulation pulse whose kHz content interacts
    with the frequency-dependent conductivity of tissue.
    """
    return constant_gain(basis, gain)


def electrode_interface_transfer(basis, tau_highpass_s: float = 3.0,
                                 tau_lowpass_s: float = 1.6e-3, gain: float = 1.0):
    """the electrode-electrolyte interface and the amplifier, as a band-pass.

    the high-pass is real hardware, not preprocessing: the half-cell potential at
    a metal-electrolyte interface is hundreds of millivolts against a signal of
    tens of microvolts, so every biopotential amplifier AC-couples, and that
    coupling has a corner between 0.01 and 1 Hz.  the consequence is that the
    slowest components an EEG can carry are set by an engineering decision, and
    a slow cortical potential is attenuated by the instrument before any analysis
    touches it.

    the low-pass is the anti-alias filter, and its corner is where the
    observation's band ceiling comes from.

    both corners belong in the *coupling* and not in the observation's noise
    model, because they act on the state, not on the evidence about it.  putting
    them here means an intervention applied through the same contact sees the
    same filter, which is correct and is not what a preprocessing-shaped model
    would do.
    """
    hp = 1.0 - low_pass(basis, tau_highpass_s)
    return gain * series(hp, low_pass(basis, tau_lowpass_s))


def coil_induction_transfer(basis, ring_hz: float = 3000.0, q: float = 2.0,
                            gain: float = 1.0):
    """coil current -> induced electric field in tissue: a differentiator and a ring.

    faraday's law makes the induced field proportional to dI/dt, and the
    differentiator is why TMS parameters are quoted as rate of change of current
    rather than as current: a hundred amperes changing slowly induces nothing.
    the ringing is the stimulator's own RLC discharge, in the low kHz, and its
    shape is why monophasic and biphasic pulses have different thresholds and
    different directional selectivity from the same peak field.

    the whole path is quasi-static apart from this: the field appears in tissue
    with no propagation delay, so all of the temporal structure of a TMS pulse
    comes from the capacitor and the coil rather than from the head.
    """
    return gain * series(1j * basis.omega, resonator(basis, ring_hz, q))


def current_injection_transfer(basis, gain: float = 1.0):
    """injected contact current -> current density in tissue: frequency-flat.

    the same quasi-static statement as the lead field and, by reciprocity,
    literally the same matrix transposed.  that reciprocity is worth having in
    the ontology: it means a tES montage's field and an EEG montage's
    sensitivity are one object, so a materialization that has solved one has
    solved the other, and an experiment that stimulates and records through the
    same electrodes is not two models.

    frequency-flat holds through tDCS, tACS and tRNS.  it holds less well for
    the kHz components of a pulsed waveform, where tissue conductivity begins to
    disperse.
    """
    return constant_gain(basis, gain)


def acoustic_transfer(basis, time_of_flight_s: float = 8e-5,
                      tau_absorption_s: float = 2e-6, gain: float = 1.0):
    """transducer drive -> acoustic pressure at the focus.

    unlike everything else in this file, ultrasound is *not* quasi-static.  the
    wavelength at 500 kHz in tissue is about 3 mm, which is the same order as the
    structures involved, so there is a genuine propagation delay and a genuine
    diffraction pattern -- the delay term here is a real time of flight through
    the skull and brain, not a formality.

    the absorption pole is the frequency dependence of attenuation, which is why
    transcranial ultrasound uses hundreds of kHz rather than the megahertz of
    imaging: attenuation rises roughly linearly with frequency and the skull is
    by far the worst of it.  the skull also refracts and mode-converts, which
    this one-dimensional form cannot express at all and which is the dominant
    source of focal error in practice.
    """
    return gain * series(pure_delay(basis, time_of_flight_s),
                         low_pass(basis, tau_absorption_s))


def optical_transfer(basis, tau_s: float = 0.05, gain: float = 1.0):
    """chromophore concentration -> detected optical attenuation.

    the modified beer-lambert law is algebraic: attenuation is concentration
    times pathlength times extinction, and the differential pathlength factor
    absorbs the fact that photons scatter and travel much further than the
    source-detector separation.  so the only reason this is a filter at all is
    the detector's own integration, which is fast against everything
    haemodynamic.

    the honest problem is not temporal.  it is that the pathlength is not known
    -- the differential pathlength factor is taken from a population table, varies
    with age and with wavelength, and a systematic error in it becomes a
    systematic error in every concentration reported.  and a large fraction of
    the detected change comes from scalp and skull rather than from brain, which
    no filter can fix.
    """
    return gain * low_pass(basis, tau_s)


def afferent_pathway_transfer(basis, mean_delay_s: float = 0.015,
                              sd_delay_s: float = 0.005, tau_synapse_s: float = 2e-3,
                              gain: float = 1.0):
    """afferent firing -> central synaptic drive: dispersed delay into a synapse.

    the same shape as the corticospinal leg, and the dispersion is even larger
    here because a peripheral nerve is a far more heterogeneous bundle than a
    tract: A-beta fibres conduct at 40-70 m/s and unmyelinated C fibres at under
    2, a difference of nearly two orders of magnitude in the same nerve.  that is
    why first and second pain arrive a second apart, and it is why a single mean
    delay is not merely imprecise but wrong -- the distribution is bimodal, and a
    gaussian dispersion represents it badly.
    """
    return gain * series(delay_dispersion(basis, mean_delay_s, sd_delay_s),
                         alpha_synapse(basis, tau_synapse_s))


# ---------------------------------------------------------------------------
# device coupling
# ---------------------------------------------------------------------------

DEVICE_COUPLING = process(
    id="device_coupling",
    doc="""instrument state and tissue state, coupled in both directions by
    ordinary physics.

    the bidirectionality is the whole design and it is visible in the
    declaration: `device.contact_potential`, `device.coil_current` and
    `device.transducer_drive` appear in both the input and the output sets, as do
    the electromagnetic and mechanical fields.  recording is this process running
    from tissue to contact; stimulating is it running from device to tissue; and
    they are the same process because they are the same physics, related by
    reciprocity.  a model that declared a "forward model" and a "stimulation
    model" separately would have two parameter sets for one lead field, and every
    simultaneous stimulation-recording experiment would be a place where they
    could silently disagree.

    the material reads are load-bearing and not decorative.  conductivity and
    permittivity are what the lead field is made of, and the single largest
    source of error in EEG source localization is skull conductivity, which is
    uncertain by a factor of two or more between individuals and is essentially
    never measured.  declaring it as state read by this process rather than as a
    constant inside an implementation is what allows that uncertainty to
    propagate into the posterior instead of vanishing into a solver.

    the display and speaker components appear as outputs of nothing and inputs to
    `transduction`: they are exogenous, clamped by the stimulus interventions,
    and this process carries them no further because the path from a screen to a
    retina is optics the ontology does not model.  that is a gap, and it is the
    reason the photoreceptor implementation has an `optical_gain` absorbing it.

    where this breaks: it is quasi-static everywhere except the acoustic leg.
    that is correct for every neural frequency and is quietly wrong for the kHz
    content of a stimulation pulse, where tissue conductivity disperses and the
    frequency-flat lead field is no longer exactly the right object.""",
    inputs=(
        # the tissue side.
        within("neural", "transmembrane_current", band=INSTRUMENT),
        within("electromagnetic", "potential", "efield", "current_density", "bfield",
               region=HEAD, band=INSTRUMENT),
        within("mechanical", "pressure", "displacement", region=HEAD,
               band=ULTRASOUND_ENVELOPE),
        within("blood", "deoxyhemoglobin", "oxygenation", "volume", band=OPTICAL),
        within("material", "conductivity", "permittivity", "mass_density", region=HEAD,
               band=STRUCTURE),
        # the device side.
        within("device", "contact_potential", "impedance", region=SENSORS, band=RECORDING),
        within("device", "contact_potential", "impedance", region=IMPLANT, band=INSTRUMENT),
        within("device", "coil_current", region=STIMULATOR, band=STIMULATION),
        within("device", "transducer_drive", region=STIMULATOR, band=ULTRASOUND_CARRIER),
        within("device", "channel_gain", band=STRUCTURE),
        within("device", "sequence_phase", region=SCANNER, band=OPTICAL),
    ),
    outputs=(
        # recording: tissue state appears at a contact.
        within("device", "contact_potential", region=SENSORS, band=RECORDING),
        within("device", "contact_potential", region=IMPLANT, band=INSTRUMENT),
        within("device", "impedance", band=STRUCTURE),
        within("device", "channel_gain", band=STRUCTURE),
        # stimulating: device state appears in tissue.
        within("electromagnetic", "efield", "current_density", region=HEAD,
               band=STIMULATION),
        within("mechanical", "pressure", region=HEAD, band=ULTRASOUND_ENVELOPE),
        within("thermal", "temperature", region=HEAD, band=Band(0.0, 0.1)),
    ),
    topology="device_coupling",
    timescale_s=1e-4,
    validity=Validity(
        min_spacing_mm=0.1, max_spacing_mm=10.0, band=INSTRUMENT,
        note="a quasi-static volume-conduction description.  below ~0.1 mm the "
             "extracellular medium is not a homogeneous conductor and the lead field "
             "stops being the right object; above ~10 mm the source is spread over "
             "tissue whose orientation varies, and the cancellation between opposing "
             "gyral banks -- which is most of why EEG sees so little of what the cortex "
             "does -- is averaged away rather than represented."),
    provenance=Provenance.PHYSICS,
    tags=frozenset({"instrument", "electromagnetic", "observable"}),
    notes="scalar material and blood inputs against spectral electromagnetic and device "
          "state: the registered scalar -> spectral conversion applies to the material "
          "reads, which is harmless because they are genuinely DC.",
)

implementation(
    name="quasistatic_lead_field",
    process="device_coupling",
    doc="""the lead field: a frequency-flat linear map between source currents
    and contact potentials.

    the standard forward model, and Form.LTI with a constant transfer rather than
    Form.CONSTRAINT because the amplifier's own band-pass sits in the same path
    and a constraint would have nowhere to put it.  the gain matrix itself is
    geometry and material, computed by a boundary-element or finite-element
    solve at materialization; what is parameterized here is the handful of
    numbers that solve is most sensitive to.

    the parameter that matters is skull conductivity.  the brain-to-skull
    conductivity ratio has been reported anywhere from 15 to 80, individual
    measurements disagree by more than a factor of two, and localization error
    scales with it directly.  it is given a wide literature prior here rather
    than a point value precisely so that an EEG-only materialization cannot
    quietly report a source location with more confidence than the head model
    supports.

    where it breaks: it assumes a fixed geometry.  a subdural grid deforms the
    cortex it sits on, an electrode drifts over days of chronic recording, and
    scalp electrode positions are digitized with several millimetres of
    residual -- and none of that is in here.""",
    form=Form.LTI,
    transfer=lead_field_transfer,
    params={
        "brain_conductivity_s_m": normal(0.33, 0.05, units="S/m",
                                         provenance=Provenance.LITERATURE,
                                         source="geddes & baker; grey matter at low frequency",
                                         note="white matter is anisotropic, roughly 9:1 along "
                                              "fibres, which an isotropic value discards"),
        "skull_conductivity_ratio": lognormal(40.0, 2.0, units="dimensionless",
                                              provenance=Provenance.LITERATURE,
                                              source="reported brain:skull ratios span 15-80",
                                              note="the single largest source of EEG source "
                                                   "localization error, and essentially never "
                                                   "measured in an individual"),
        "csf_conductivity_s_m": normal(1.79, 0.1, units="S/m",
                                       provenance=Provenance.LITERATURE,
                                       note="five times grey matter; the CSF layer shunts "
                                            "current tangentially and smears the scalp "
                                            "topography more than the skull does"),
        "scalp_conductivity_s_m": normal(0.33, 0.08, units="S/m",
                                         provenance=Provenance.LITERATURE),
        "dipole_orientation_bias": weak(1.0, 3.0, units="dimensionless",
                                        note="how much of a patch's current is radial rather "
                                             "than tangential.  MEG is blind to the radial "
                                             "component and EEG is not, so this parameter is "
                                             "where their disagreement lives"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.PHYSICS,
    source="geddes & baker 1967; hallez et al 2007; vorwerk et al 2014",
)

implementation(
    name="electrode_interface",
    process="device_coupling",
    doc="""the electrode-electrolyte interface and amplifier band-pass, with
    impedance as state.

    declared separately from the lead field so that the geometry and the hardware
    can be selected independently, and so that impedance -- which drifts over a
    recording, differs by two orders of magnitude between a gel scalp electrode
    and a chronic intracortical site, and is routinely measured -- has a place to
    be state rather than a constant.

    the high-pass corner is the part that quietly matters.  a 0.1 Hz corner and a
    1 Hz corner are both defensible engineering choices and they attenuate a slow
    cortical potential very differently, so an observation of a contingent
    negative variation is partly an observation of the amplifier.  putting the
    corner in the coupling rather than in the observation's noise model means it
    is a property of the state, which is what it physically is.

    the impedance model is a constant-phase element in reality, not the RC
    written here; the difference shows up as a fractional-order roll-off that
    matters for microelectrodes and not for scalp gel.""",
    form=Form.LTI,
    transfer=electrode_interface_transfer,
    params={
        "tau_highpass_s": lognormal(3.0, 5.0, units="s", provenance=Provenance.LITERATURE,
                                    note="corner between 0.01 and 1 Hz depending on the "
                                         "amplifier; this is an engineering choice that "
                                         "becomes a property of the measured state"),
        "tau_lowpass_s": lognormal(1.6e-3, 4.0, units="s",
                                   provenance=Provenance.LITERATURE,
                                   note="anti-alias corner: ~100 Hz clinical EEG, ~7 kHz "
                                        "intracortical.  the spread is the range of "
                                        "instruments, not uncertainty about one"),
        "contact_impedance_ohm": lognormal(5000.0, 10.0, units="ohm",
                                           provenance=Provenance.LITERATURE,
                                           note="a few kohm for gelled scalp, hundreds of kohm "
                                                "for a microelectrode, and rising over the "
                                                "life of a chronic implant"),
        "half_cell_drift_v_per_s": speculative(1e-5, 10.0, units="V/s",
                                               note="the drift the high-pass exists to remove; "
                                                    "its magnitude is what sets how aggressive "
                                                    "the corner has to be"),
        "input_noise_v_rt_hz": lognormal(1e-8, 3.0, units="V/sqrt(Hz)",
                                         provenance=Provenance.PHYSICS,
                                         note="amplifier input-referred noise; with the "
                                              "electrode's johnson noise it is the floor every "
                                              "electrophysiological observation sits on"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.PHYSICS,
)

implementation(
    name="magnetometer_pickup",
    process="device_coupling",
    doc="""MEG sensors: a flux pickup, frequency-flat, with a gradiometer
    baseline that shapes the spatial sensitivity rather than the temporal one.

    the substitution this implementation makes is named in the module docstring
    and repeated here because it matters: the contract declares no
    magnetometer-output component, so this writes `device.channel_gain` and the
    observation attaches to `electromagnetic.bfield` directly.  the effect is
    that the sensor's transfer function is folded into the observation's band and
    noise model rather than being carried as device state.  it is the least
    satisfying declaration in this file.

    what is right about the physics: MEG is quasi-static exactly as EEG is, and
    the reason to have both is not temporal but geometric.  the magnetic field is
    almost entirely insensitive to the skull, which removes the single worst
    parameter in the EEG forward model, and almost entirely blind to radial
    sources, which removes gyral crowns from view.  the two instruments are
    therefore not redundant and not interchangeable, and the parameter carrying
    that -- `dipole_orientation_bias` in the lead field implementation -- is
    shared between them on purpose.""",
    form=Form.LTI,
    transfer=lead_field_transfer,
    params={
        "gradiometer_baseline_mm": normal(50.0, 20.0, units="mm",
                                          provenance=Provenance.LITERATURE,
                                          note="axial gradiometer baseline; it suppresses "
                                               "distant environmental noise and also suppresses "
                                               "deep sources, which is a cost paid quietly"),
        "sensor_noise_ft_rt_hz": lognormal(3.0, 2.0, units="fT/sqrt(Hz)",
                                           provenance=Provenance.LITERATURE,
                                           note="SQUID noise floor; optically pumped "
                                                "magnetometers are comparable but sit far "
                                                "closer to the head, which matters more"),
        "standoff_mm": normal(20.0, 8.0, units="mm", provenance=Provenance.LITERATURE,
                              note="scalp-to-sensor distance.  the field falls with distance "
                                   "steeply enough that this single number changes sensitivity "
                                   "by a factor of several, which is the whole argument for "
                                   "on-scalp magnetometry"),
        "radial_blindness": normal(0.05, 0.03, units="dimensionless",
                                   provenance=Provenance.PHYSICS,
                                   note="residual sensitivity to a radial source in a realistic "
                                        "head; exactly zero only in a spherical one"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.PHYSICS,
    source="hamalainen et al 1993",
)

implementation(
    name="coil_induction",
    process="device_coupling",
    doc="""TMS: coil current differentiated into an induced electric field.

    the clean case for the architecture's treatment of intervention.  an
    intervention clamps `device.coil_current`; this implementation carries it
    into `electromagnetic.efield`; `em_coupling` carries the field into
    population state; and everything downstream -- a motor evoked potential, a
    TMS-evoked EEG response, a behavioural effect -- follows from ordinary
    processes with no stimulation-specific machinery anywhere.

    the parameters are unusually well constrained by physics and unusually badly
    constrained where biology enters.  dI/dt and the induced field magnitude are
    measurable and well known: a standard stimulator reaches ~10^8 A/s and
    induces on the order of 100 V/m at the cortical surface.  what is not known
    is what that field does to a neuron, which is `em_coupling`'s problem and not
    this one -- and the division is deliberate, so that the physics can be
    confident while the biology stays weak.

    where it breaks: the field is computed in a quasi-static approximation with
    isotropic conductivity, so the anisotropy of white matter and the sharp
    conductivity boundaries the field is most sensitive to are approximated.  and
    coil position is treated as known, when in practice a few millimetres of
    neuronavigation error changes the stimulated site materially.""",
    form=Form.LTI,
    transfer=coil_induction_transfer,
    params={
        "di_dt_peak_a_per_s": lognormal(1.0e8, 1.5, units="A/s",
                                        provenance=Provenance.LITERATURE,
                                        note="a standard biphasic stimulator at maximum output"),
        "ring_hz": normal(3000.0, 800.0, units="Hz", provenance=Provenance.LITERATURE,
                          note="RLC discharge frequency; a biphasic pulse is ~300 us and a "
                               "monophasic one is a longer, asymmetric discharge"),
        "q": normal(2.0, 0.7, units="dimensionless", provenance=Provenance.PHYSICS,
                    note="the damping is by design: monophasic stimulators dump the reverse "
                         "phase into a diode, and the difference in waveform is why the two "
                         "have different thresholds from the same peak field"),
        "peak_efield_v_per_m": normal(100.0, 30.0, units="V/m",
                                      provenance=Provenance.LITERATURE,
                                      note="at the cortical surface at typical output; it falls "
                                           "steeply with depth, which is why deep TMS is hard "
                                           "for reasons no coil design fully solves"),
        "focality_mm": normal(15.0, 5.0, units="mm", provenance=Provenance.LITERATURE,
                              note="half-maximum extent of a figure-of-eight coil's field; "
                                   "circular coils are far broader"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.PHYSICS,
    source="barker et al 1985; thielscher & kammer 2004; deng et al 2013",
)

implementation(
    name="current_injection",
    process="device_coupling",
    doc="""transcranial and invasive electrical stimulation: injected current
    into a tissue current density.

    the reciprocal of the lead field, and declared as its own implementation so
    that the reciprocity is visible in the source rather than being a remark.
    tDCS, tACS, tRNS, DBS and direct cortical stimulation are all this
    implementation with different waveforms clamped onto the contact -- which is
    the point, because they are all the same physics and differ only in what the
    intervention does.

    the number worth carrying is small and often forgotten: 1 mA of tDCS produces
    on the order of 0.2-0.5 V/m in cortex, one to two orders of magnitude below
    the threshold for driving a neuron to fire.  transcranial electrical
    stimulation is therefore a subthreshold modulation of excitability, not a
    driving stimulus, and any implementation of `em_coupling` that treats it as
    driving is making an error this parameter makes visible.  invasive
    stimulation through a DBS lead is a different regime entirely -- volts across
    millimetres -- and the same implementation covers it because the physics is
    the same and only the magnitude differs.

    where it breaks: the frequency-flat assumption degrades for the kHz content
    of a pulsed waveform, and charge-balance and electrochemical safety limits --
    which are what actually constrain invasive stimulation parameters -- are not
    represented at all.""",
    form=Form.LTI,
    transfer=current_injection_transfer,
    params={
        "efield_per_ma_v_per_m": normal(0.35, 0.15, units="V/m per mA",
                                        provenance=Provenance.LITERATURE,
                                        source="tDCS current-flow modelling and intracranial "
                                               "measurement in humans",
                                        note="two orders of magnitude below firing threshold; "
                                             "tES modulates excitability, it does not drive"),
        "shunt_fraction_scalp": normal(0.5, 0.2, units="dimensionless",
                                       provenance=Provenance.LITERATURE,
                                       note="roughly half the injected current never enters the "
                                            "brain, taking the low-resistance scalp path "
                                            "instead"),
        "invasive_efield_v_per_m": lognormal(100.0, 5.0, units="V/m",
                                             provenance=Provenance.LITERATURE,
                                             note="near a DBS contact at therapeutic amplitude; "
                                                  "suprathreshold within a few millimetres and "
                                                  "subthreshold beyond, which is why the volume "
                                                  "of tissue activated is the quantity clinical "
                                                  "work actually models"),
        "charge_density_limit_uc_cm2": normal(30.0, 10.0, units="uC/cm^2",
                                              provenance=Provenance.LITERATURE,
                                              note="the shannon safety limit for chronic "
                                                   "stimulation.  declared so a materialization "
                                                   "can check it; nothing here enforces it"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.PHYSICS,
    source="datta et al 2009; opitz et al 2016; shannon 1992",
)

implementation(
    name="acoustic_beam",
    process="device_coupling",
    doc="""focused ultrasound: transducer drive into pressure at a focus, with
    the thermal leg alongside.

    the one non-quasi-static implementation in the file.  at a few hundred kHz
    the wavelength in tissue is millimetres, so the coupling has a genuine
    propagation delay and the focus is a diffraction pattern rather than a
    geometric point.

    two output legs, and keeping both is the honest choice.  the mechanical leg
    is the intended one -- radiation force and the mechanical index are the
    presumed mechanism of neuromodulation, though the mechanism is genuinely
    unsettled.  the thermal leg is the one that constrains what an experiment may
    do: absorbed acoustic power becomes heat, the skull absorbs far more than
    brain does, and thermal safety rather than efficacy is what sets duty cycle
    in every human protocol.  a declaration that carried only the mechanical leg
    would make the safety constraint invisible.

    the skull is the problem and this form does not solve it.  it attenuates,
    refracts, defocuses and mode-converts, so the actual focus can be
    millimetres from the intended one with a substantially reduced peak, and
    correcting for it requires a CT-derived skull model that this
    one-dimensional transfer function has nowhere to put.""",
    form=Form.LTI,
    transfer=acoustic_transfer,
    params={
        "carrier_hz": lognormal(5.0e5, 2.0, units="Hz", provenance=Provenance.LITERATURE,
                                note="200-700 kHz for transcranial work: low enough to get "
                                     "through the skull, high enough to focus"),
        "time_of_flight_s": lognormal(8e-5, 1.5, units="s", provenance=Provenance.PHYSICS,
                                      note="~1540 m/s in soft tissue, ~2900 in skull; the "
                                           "difference is what causes the aberration"),
        "skull_attenuation_db": normal(12.0, 5.0, units="dB",
                                       provenance=Provenance.LITERATURE,
                                       note="highly individual, driven by skull thickness and "
                                            "porosity, and the reason CT-based correction "
                                            "exists"),
        "focal_pressure_mpa": normal(0.5, 0.2, units="MPa",
                                     provenance=Provenance.LITERATURE,
                                     note="typical neuromodulation intensity, well below "
                                          "cavitation and well below ablation"),
        "focal_width_mm": normal(4.0, 1.5, units="mm", provenance=Provenance.LITERATURE,
                                 note="lateral half-maximum; axially the focus is several times "
                                      "longer, which is a fact about diffraction and not about "
                                      "the transducer"),
        "thermal_efficiency": weak(0.01, 10.0, units="dimensionless",
                                   note="fraction of acoustic power deposited as heat locally.  "
                                        "small in brain and much larger in skull, which is "
                                        "where the safety limit actually binds"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.PHYSICS,
    source="legon et al 2014; aubry et al 2003",
)

implementation(
    name="optical_beer_lambert",
    process="device_coupling",
    doc="""NIRS: chromophore concentration into detected attenuation, by the
    modified beer-lambert law.

    another named substitution -- there is no optical component in the contract,
    so this writes `device.channel_gain` and the fNIRS observation attaches to
    the blood components directly.  the coupling is therefore expressed as a
    gain relating haemoglobin concentration to a detected quantity, which is
    exactly what the modified beer-lambert law is, so the substitution costs less
    here than it does for MEG.

    what the parameters encode is the instrument's two chronic problems.  the
    differential pathlength factor is not measured per subject -- it comes from an
    age-dependent population table, varies with wavelength, and any error in it
    is a proportional error in every concentration reported.  and the partial
    pathlength in brain is a minority of the total: most of the detected light
    never leaves the scalp, so a large fraction of an fNIRS signal is
    extracerebral haemodynamics.  short-separation regression exists to remove
    it, works partially, and is a preprocessing step whose necessity is a
    statement about this coupling.

    the depth limit follows from the same physics and is hard: with a 3 cm
    source-detector separation the sensitivity peak sits ~1.5 cm below the scalp,
    so anything below superficial cortex is out of reach.""",
    form=Form.LTI,
    transfer=optical_transfer,
    params={
        "differential_pathlength_factor": normal(6.0, 1.0, units="dimensionless",
                                                 provenance=Provenance.LITERATURE,
                                                 source="age- and wavelength-dependent "
                                                        "population tables",
                                                 note="not measured per subject; an error here "
                                                      "is a proportional error in every "
                                                      "reported concentration"),
        "source_detector_mm": normal(30.0, 5.0, units="mm",
                                     provenance=Provenance.LITERATURE,
                                     note="sets penetration: sensitivity peaks around half the "
                                          "separation in depth"),
        "extracerebral_fraction": normal(0.6, 0.15, units="dimensionless",
                                         provenance=Provenance.LITERATURE,
                                         note="most of the detected change is scalp and skull; "
                                              "this parameter is why short-separation channels "
                                              "exist"),
        "extinction_hbo_per_mm_mm": lognormal(0.1, 2.0, units="1/(mM.mm)",
                                              provenance=Provenance.LITERATURE,
                                              note="oxyhaemoglobin extinction at ~850 nm; the "
                                                   "isosbestic point near 800 nm is why two "
                                                   "wavelengths suffice to separate the two "
                                                   "species"),
        "tau_s": lognormal(0.05, 3.0, units="s", provenance=Provenance.PHYSICS,
                           note="detector integration; fast against anything haemodynamic, "
                                "which is why this coupling is effectively algebraic"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.LITERATURE,
    source="delpy et al 1988; scholkmann et al 2014",
)

implementation(
    name="learned_instrument_residual",
    process="device_coupling",
    doc="""a learned correction on the analytic forward model, per instrument.

    the argument here is narrower than elsewhere and correspondingly stronger.
    nobody wants a learned lead field: the physics is right and a fitted
    replacement would be worse.  what is wanted is a correction for the specific
    things the analytic model provably omits -- individual skull conductivity,
    electrode position error, cortical deformation under a subdural grid, the
    scalp shunt path, and the systematic bias of a template head model used in
    place of an individual one.  those have structure across subjects and
    sessions, which is exactly what a learned residual with a per-session
    embedding can carry and a per-site solve cannot.

    it is parameterized as a residual with a small prior gain for the obvious
    reason: with no calibration data it must reduce to the physics, because the
    physics is the trustworthy part.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "residual_gain": weak(0.15, 4.0,
                              note="deliberately small.  the analytic forward model is the "
                                   "best-founded thing in this file and the learned term "
                                   "exists to correct its known omissions, not to replace it"),
        "session_embedding_dim": uniform(2.0, 16.0, units="dimensions",
                                         note="per-session nuisance: cap placement, impedance "
                                              "state, head position.  see ibm.processes.nn"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


# ---------------------------------------------------------------------------
# afferent propagation
# ---------------------------------------------------------------------------

AFFERENT_PROPAGATION = process(
    id="afferent_propagation",
    doc="""primary afferent firing carried into brainstem, thalamic and primary
    sensory populations.

    the transport leg between `transduction` and the brain, and structurally the
    mirror image of `efferent_propagation`: a dispersed delay, a synapse, a gain.
    it is short and it is not simple, because two things happen along it that a
    relay would not do.

    the first is dispersion, and it is extreme.  a peripheral nerve carries
    A-beta fibres at 40-70 m/s beside unmyelinated C fibres at under 2 m/s, so
    the conduction time distribution within one nerve spans nearly two orders of
    magnitude.  that is why first and second pain arrive about a second apart
    from one stimulus, and it means a single mean delay is not an approximation
    but a misrepresentation -- the distribution is bimodal and the gaussian
    dispersion this process's LTI implementation uses represents it poorly.  the
    honest statement is that the fast and slow systems should be separate
    materialized pathways, and the topology permits that even though a single
    parameter set does not express it.

    the second is that the thalamic relay is not a relay.  it has two firing
    modes -- tonic, which transmits input roughly linearly, and burst, which does
    not -- and which mode it is in depends on membrane potential, which depends on
    arousal and on corticothalamic feedback.  the gain of this process is
    therefore state-dependent, and a materialization that treats it as a constant
    has assumed a fixed arousal state without saying so.

    the structural reads are the same as on the efferent side: conduction
    velocity is a function of myelination, and a neuropathy is a change to this
    process's inputs rather than to its theta.""",
    inputs=(
        within("neural", "afferent.activity", region=BODY, band=AFFERENT),
        within("transduction", "photoreceptor", "hair_cell", "mechanoreceptor",
               "thermoreceptor", "nociceptor", "chemoreceptor", "vestibular",
               "baroreceptor", band=AFFERENT),
        within("neural", "exc.potential", region=THALAMUS, band=AFFERENT),
        within("neural", "exc.activity", region=BRAINSTEM, band=AFFERENT),
        within("extracellular", "acetylcholine", "noradrenaline", band=Band(0.0, 1.0)),
        within("structural", "myelination", "axonal_density", band=STRUCTURE),
    ),
    outputs=(
        within("neural", "exc.ampa", region=THALAMUS, band=AFFERENT),
        # nmda separately and narrower: it is declared meaningful only to 50 Hz,
        # which is the honest statement about a conductance with a 100 ms decay --
        # it cannot carry the afferent band and pretending otherwise would be
        # writing structure into a variable that physically low-passes it away.
        within("neural", "exc.nmda", region=THALAMUS, band=Band(0.0, 50.0)),
        within("neural", "exc.activity", "exc.potential", region=THALAMUS, band=AFFERENT),
        within("neural", "exc.activity", region=BRAINSTEM, band=AFFERENT),
        within("neural", "afferent.activity", region=BODY, band=AFFERENT),
    ),
    topology="afferent_pathway",
    timescale_s=5e-3,
    validity=Validity(
        min_spacing_mm=0.5, max_spacing_mm=20.0, band=AFFERENT,
        note="a pathway-level description.  below ~0.5 mm the relay nucleus is resolved "
             "into cells and the burst/tonic distinction is a single-cell property rather "
             "than a population one; above ~20 mm several modalities with different "
             "conduction velocities and different relay nuclei are averaged together."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"sensory", "peripheral", "transport"}),
)

implementation(
    name="peripheral_dispersed_delay",
    process="afferent_propagation",
    doc="""the pathway as a dispersed delay line into a thalamic synapse.

    linear and exact at any timestep, and it carries the latencies that a great
    many datasets in docs/EVIDENCE.md are effectively measurements of: the
    auditory brainstem response wave V at ~6 ms, the somatosensory N20 at 20 ms,
    the visual evoked potential P100 at 100 ms.  those are strong constraints and
    they are why these priors are literature rather than weak.

    the dispersion parameter is doing more work than the mean.  a bundle with a
    wide velocity spread is a low-pass, so a pathway that transmits a click
    faithfully at the brainstem transmits a much smoother thing at cortex purely
    from fibre heterogeneity, before any synapse.  attributing that smoothing to
    synaptic filtering, which a fixed-delay model must, misplaces the mechanism.

    where it breaks: the bimodality described in the process doc.  a gaussian
    dispersion spanning A-beta and C fibres has a mean that corresponds to no
    fibre at all.""",
    form=Form.LTI,
    transfer=afferent_pathway_transfer,
    params={
        "mean_delay_s": lognormal(0.015, 2.5, units="s", provenance=Provenance.LITERATURE,
                                  source="ABR wave V ~6 ms, SEP N20 ~20 ms, VEP P100 ~100 ms",
                                  note="a per-modality quantity; the wide prior is the range "
                                       "across pathways, and per-partition tying is what makes "
                                       "each of them separately identifiable"),
        "sd_delay_s": lognormal(0.005, 3.0, units="s", provenance=Provenance.LITERATURE,
                                note="fibre diameter spread within a nerve; for a mixed nerve "
                                     "carrying both A-beta and C fibres a gaussian is the "
                                     "wrong family and this parameter is a compromise"),
        "conduction_velocity_m_s": lognormal(30.0, 6.0, units="m/s",
                                             provenance=Provenance.LITERATURE,
                                             note="A-beta 40-70, A-delta 5-30, C under 2.  the "
                                                  "log-spread covers all three because they "
                                                  "share a nerve"),
        "tau_synapse_s": lognormal(2e-3, 2.0, units="s", provenance=Provenance.LITERATURE,
                                   note="thalamic relay AMPA EPSP"),
        "gain": weak(1.0, 8.0, units="central Hz per afferent Hz",
                     note="absorbs convergence ratio, which varies enormously by pathway: "
                          "retinal ganglion to LGN is nearly one-to-one, and cutaneous "
                          "afferents to cuneate cells are not"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="chiappa 1997; kandel et al, principles",
)

implementation(
    name="thalamic_relay_gating",
    process="afferent_propagation",
    doc="""the relay with its two firing modes and its state-dependent gain.

    the nonlinear form, and it exists because the most consequential fact about
    the thalamus is one no linear relay can express: the same afferent input
    produces qualitatively different output depending on the relay cell's
    membrane potential.  depolarized, the cell is in tonic mode and transmits
    input approximately linearly.  hyperpolarized, the T-type calcium current
    de-inactivates and the cell fires bursts -- high-frequency packets that are
    excellent at detecting that *something* arrived and poor at reporting what it
    was.  the switch between them is what gates sensory transmission across the
    sleep-wake cycle, and it is set by cholinergic and noradrenergic drive, which
    is why those are read as inputs.

    `state_dependent_weights` is set for this implementation, because the gain is
    a function of state rather than of theta alone, and that is exactly the
    attention-like case the flag exists to mark.

    where it breaks: there is no thalamic reticular nucleus in it, so the
    inhibitory gating that actually implements much of sensory selection is
    absent, and no corticothalamic feedback, so the relay here is driven from
    below only.  both belong to `thalamocortical_coupling` and their absence here
    means this implementation understates how much of thalamic gain is
    top-down.""",
    form=Form.RATE,
    params={
        "burst_threshold_mv": normal(-65.0, 5.0, units="mV",
                                     provenance=Provenance.LITERATURE,
                                     note="below this the T-current de-inactivates and the mode "
                                          "switches; the transition is sharp"),
        "t_current_tau_s": lognormal(0.1, 2.0, units="s", provenance=Provenance.LITERATURE,
                                     note="de-inactivation time; it is why a cell must be held "
                                          "hyperpolarized for ~100 ms before it can burst"),
        "tonic_gain": weak(1.0, 5.0, units="dimensionless"),
        "burst_gain": weak(3.0, 5.0, units="dimensionless",
                           note="a burst delivers more spikes per input event than tonic mode "
                                "does, which is why bursting is a better detector and a worse "
                                "encoder"),
        "acetylcholine_depolarization_mv": normal(5.0, 3.0, units="mV",
                                                  provenance=Provenance.LITERATURE,
                                                  note="cholinergic drive depolarizes relay "
                                                       "cells out of burst mode; this is most "
                                                       "of what waking up does to the thalamus"),
        "adaptation_fraction": speculative(0.3, 5.0, units="dimensionless",
                                           note="rapid adaptation at the relay synapse; "
                                                "reported but not well quantified across "
                                                "pathways"),
    },
    tying=Tying.PER_PARTITION,
    state_dependent_weights=True,
    provenance=Provenance.LITERATURE,
    source="sherman & guillery 2006; steriade et al 1993",
)

implementation(
    name="learned_afferent_pathway",
    process="afferent_propagation",
    doc="""a learned pathway map with the dispersed delay line as its prior mean.

    the case is the same one as on the efferent side and it is a modest one.  the
    latency structure is measured and should not be fitted; what a learned term
    can add is the transformation, because a sensory pathway is not a wire.
    retinal ganglion output is already centre-surround filtered, the cochlear
    nucleus does several distinct things to its input in parallel, and the
    dorsal column nuclei sharpen spatially.  none of that is a delay and none of
    it is a gain.

    the residual parameterization keeps the latency out of reach of the fit for
    the same reason as before: it is the part that is known.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "residual_gain": weak(0.3, 5.0),
        "modality_embedding_dim": uniform(4.0, 32.0, units="dimensions",
                                          note="pathways share structure -- every one of them "
                                               "does some form of contrast enhancement -- so an "
                                               "embedding lets evidence from one inform "
                                               "another.  see ibm.processes.nn"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)

__all__ = ["DEVICE_COUPLING", "AFFERENT_PROPAGATION"]
