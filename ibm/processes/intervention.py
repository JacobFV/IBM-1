"""interventions: what an experimenter clamps, and what the clamp then does
through ordinary processes.

ARCHITECTURE.md §6 gives the whole content of this file in one line: an
intervention is externally constrained state.  a TMS coil does not "stimulate the
brain"; it constrains `device.coil_current` to a waveform the experimenter
chose, and everything after that -- the induced electric field, the population
response, the motor evoked potential, the behavioural effect -- follows from
`device_coupling`, `em_coupling`, `efferent_propagation` and
`effector_activation`, which are the same processes that run when nobody is
stimulating anything.

nothing in this file has a mechanism in it.  that is the test of whether the
design is working: if a declaration below needed a parameter describing how the
stimulation affects the brain, the mechanism would be in the wrong place, and it
would be in the wrong place in a way that made stimulation experiments
incomparable to observational ones.  what is here is a component, a region, a
waveform family and a frame.

three things fall out of this that are worth stating, because each of them is
normally a separate piece of machinery in other modelling frameworks.

*stimulation and recording are the same physics.*  tDCS clamps
`device.contact_potential` and EEG observes it, through one lead field related by
reciprocity.  a montage that stimulates well is a montage that records well, and
in ibm-1 that is arithmetic rather than a coincidence worth remarking on.

*a stimulus and a lesion and a drug are the same kind of object.*  a visual
stimulus clamps `device.display_luminance`; sensory deprivation clamps the same
component to a constant; a pharmacological manipulation clamps an extracellular
concentration.  they are all constraints on ordinary state, and the framework
does not distinguish "input" from "perturbation".

*a clamped component need not be written by any process.*  the registry's own
check knows this: a component that is read but never written is an error
*unless* it is exogenous or clamped by an intervention.  the display and speaker
components exist only because something outside the model sets them, and these
declarations are what makes the graph closed rather than holed.

two limitations to record honestly.

the first is that `Intervention` carries no species field, and two of the
declarations below -- optogenetic stimulation and, in most of its interesting
forms, invasive stimulation -- are not human techniques.  that is noted in the
docstrings and it is not enforced anywhere, so a materialization can construct a
human model with an optogenetic intervention on it and nothing will complain.

the second is deeper and concerns pharmacology.  most drugs do not change state;
they change a *process parameter* -- an anaesthetic changes the GABA-A decay time
constant, not the GABA concentration.  §4 supports exactly this: a process may
apply pressure to another process's theta.  but an `Intervention` constrains
state, not theta, so the pharmacological declarations below are approximations
that clamp a concentration where the honest object would clamp a rate constant.
the gap is named where it occurs rather than papered over.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Intervention
from ibm.vocabulary import Band, Near, OnSupport, sel

I = REGISTRY.intervention

SENSORS = OnSupport("sensor_array")
IMPLANT = OnSupport("implanted_array")
STIMULATOR = OnSupport("stimulator")
DISPLAY = OnSupport("display")
EPITHELIUM = OnSupport("chemosensory_epithelium")
HEAD = OnSupport("head_volume")

#: stimulation waveforms have real bandwidth.  a TMS pulse is a ~100 us biphasic
#: transient whose spectrum runs to several kHz; clamping it over a narrow band
#: would misstate the induced field, since the field is the *derivative* of the
#: current and derivatives are exactly where high-frequency content matters.
PULSE = Band(0.0, 20000.0)

#: slowly varying transcranial waveforms: dc, low-frequency sinusoid, noise.
TES = Band(0.0, 1000.0)

#: what a display or speaker can actually produce.
VISUAL = Band(0.0, 120.0)
AUDIO = Band(20.0, 20000.0)

#: focused ultrasound: the carrier is real and a materialization will normally
#: work with the envelope instead.  clamping the carrier band says which choice
#: was made.
ULTRASOUND = Band(1.0e5, 2.0e6)

#: pharmacological and thermal manipulations, which are slow by construction.
SLOW = Band(0.0, 0.5)


# ---------------------------------------------------------------------------
# non-invasive electromagnetic
# ---------------------------------------------------------------------------

TMS = I(Intervention(
    id="tms",
    doc="""transcranial magnetic stimulation: the coil current is clamped.

    the architecture's own worked example, and the cleanest one.  what an
    experimenter controls is a capacitor discharging through a coil, which is
    `device.coil_current`; everything downstream is physics and biology the model
    already contains.  the induced field is dI/dt, computed by `device_coupling`'s
    coil induction implementation, so the *shape* of the clamped waveform matters
    as much as its amplitude -- which is why monophasic and biphasic pulses have
    different thresholds and different directional selectivity at identical peak
    field, and why the band here runs to 20 kHz rather than to something
    comfortable.

    the paired-pulse and repetitive protocols that most TMS research is built on
    are this same clamp with structured waveforms: two pulses at a chosen
    interstimulus interval, or a train at a chosen frequency.  they need no
    additional declaration, which is the point.  what they *do* need, and what
    ibm-1 does not currently have, is a plasticity process sensitive enough to
    reproduce why 1 Hz repetitive stimulation depresses and 10 Hz facilitates --
    that asymmetry is a claim about `plasticity`, not about this intervention.""",
    constrains=sel("device.coil_current", region=STIMULATOR, band=PULSE),
    modality="tms",
    waveform="monophasic or biphasic pulse, ~100-300 us, optionally in trains",
    frame="coil",
))

TDCS = I(Intervention(
    id="tdcs",
    doc="""transcranial direct current stimulation: contact current clamped to a
    constant.

    the same component EEG observes, clamped instead of measured, and by
    reciprocity through the same lead field.  the number that governs how this
    declaration should be interpreted lives in `device_coupling`: 1 mA produces
    on the order of 0.2-0.5 V/m in cortex, one to two orders of magnitude below
    the field needed to make a neuron fire.  tDCS is therefore a subthreshold
    modulation of excitability rather than a drive, and roughly half the injected
    current never enters the head at all, taking the scalp's low-resistance path.

    both of those facts are in the coupling and not here, which is exactly right:
    they are properties of the head, not of the experimenter's choice.  the
    experimenter's choice is a current and a montage, and that is all this
    declaration contains.

    the band includes DC by construction, and that is the one place where tDCS
    and EEG genuinely differ in what they can touch: an AC-coupled amplifier
    cannot observe the component tDCS most directly constrains.""",
    constrains=sel("device.contact_potential", region=SENSORS, band=Band(0.0, 1.0)),
    modality="tdcs",
    waveform="constant current with ramped on- and offset, typically 1-2 mA",
    frame="eeg_cap",
))

TACS = I(Intervention(
    id="tacs",
    doc="""transcranial alternating current stimulation: a sinusoid at a chosen
    frequency.

    the same clamp as tDCS with a different waveform, and it is worth having a
    separate declaration only because the scientific claim attached to it is
    different in kind.  tACS is used to *entrain* an endogenous rhythm, which is a
    claim about resonance -- that a weak periodic field can phase-lock an
    oscillator whose amplitude it cannot change.

    ibm-1 is unusually well placed to represent that claim and unusually likely
    to expose it as fragile.  entrainment appears in the spectral form as a shift
    in `phase_concentration` at the driven component -- the transition from an
    isotropic eigenplane (ongoing rhythm, uniform phase) to an anisotropic one
    (phase preference) is precisely what entrainment *is* in this
    representation.  so the model can express it exactly, and it can also
    express the alternative: that the observed effect is a stimulation artefact
    of the same frequency, which occupies the same component with a very
    different spatial pattern.  distinguishing them is a real inference problem
    and this declaration lets it be one rather than assuming an answer.""",
    constrains=sel("device.contact_potential", region=SENSORS, band=TES),
    modality="tacs",
    waveform="sinusoid, 1-100 Hz, typically 1-2 mA peak to peak",
    frame="eeg_cap",
))

TRNS = I(Intervention(
    id="trns",
    doc="""transcranial random noise stimulation.

    a band-limited noise waveform on the same contacts.  included because it is a
    genuinely different hypothesis from tACS -- the proposed mechanism is
    stochastic resonance, in which added noise increases the detectability of a
    subthreshold signal in a nonlinear system, rather than entrainment -- and
    because a noise clamp is a different object in the spectral form: it
    constrains the *psd* of the device component across a band while leaving the
    phase unconstrained, where tACS constrains both.

    that difference is representable here with no extra machinery, which is a
    small but real argument for carrying beliefs as distributions over spectra
    rather than as trajectories.  the evidence for the mechanism is thin, and the
    declaration commits to none of it.""",
    constrains=sel("device.contact_potential", region=SENSORS, band=Band(100.0, 640.0)),
    modality="trns",
    waveform="band-limited gaussian noise, commonly 100-640 Hz",
    frame="eeg_cap",
))

TFUS = I(Intervention(
    id="tfus",
    doc="""transcranial focused ultrasound: the transducer drive is clamped.

    the only non-electromagnetic non-invasive intervention with millimetre focal
    resolution at depth, which is why it is interesting, and the one whose
    mechanism is least settled, which is why the declaration is careful to
    contain none.  what is clamped is a drive waveform; what happens next is
    `device_coupling`'s acoustic implementation, which carries the propagation,
    the skull aberration and -- importantly -- the thermal leg alongside the
    mechanical one.

    the thermal leg is not a footnote.  the constraint that actually determines
    what a human ultrasound protocol may do is thermal safety, not efficacy: the
    skull absorbs far more than brain and duty cycle is set by heating.  an
    intervention declaration that constrained only the mechanical path would make
    that invisible, and the fact that `device_coupling` writes
    `thermal.temperature` is what keeps it visible.

    the carrier band is declared honestly at hundreds of kHz.  almost every
    materialization will work with the pulse envelope instead, and doing so is a
    recorded approximation rather than an unexamined default.""",
    constrains=sel("device.transducer_drive", region=STIMULATOR, band=ULTRASOUND),
    modality="tfus",
    waveform="tone burst at a 200-700 kHz carrier, pulsed at 10-1000 Hz",
    frame="transducer",
))


# ---------------------------------------------------------------------------
# invasive
# ---------------------------------------------------------------------------

DBS = I(Intervention(
    id="dbs",
    doc="""deep brain stimulation: charge-balanced pulse trains at an implanted
    lead.

    the same `current_injection` physics as tDCS in a completely different
    regime, and carrying both through one implementation is deliberate: the
    difference between them is amplitude and distance, not mechanism.  near a DBS
    contact the field is on the order of 100 V/m, which is suprathreshold within
    a few millimetres and subthreshold beyond -- which is why the clinically
    modelled quantity is a volume of tissue activated, a statement about where
    the field crosses a threshold.

    the therapeutic frequency band is narrow and its narrowness is a fact worth
    representing: high-frequency stimulation around 130 Hz relieves parkinsonian
    symptoms and low-frequency stimulation around 10 Hz can worsen them.  no
    process in the current inventory explains that, and the declaration does not
    pretend to -- it clamps a waveform and leaves the explanation to whatever f
    the basal ganglia processes eventually carry.

    charge balance is a hardware requirement rather than a scientific one: a net
    DC current at a chronic electrode causes electrochemical damage, which is why
    every clinical waveform is biphasic.  the safety limit is a parameter of
    `device_coupling` and nothing here enforces it.""",
    constrains=sel("device.contact_potential", region=STIMULATOR, band=PULSE),
    modality="dbs",
    waveform="charge-balanced biphasic pulse train, 60-90 us per phase, 130 Hz",
    frame="electrode_grid",
))

INVASIVE_ELECTRICAL = I(Intervention(
    id="invasive_electrical_stimulation",
    doc="""direct cortical and intracranial stimulation through recording
    contacts.

    the same clamp as DBS applied to electrodes that are also observed, which
    makes it the sharpest case for the architecture's insistence that recording
    and stimulation are one process.  a single subdural contact can be clamped in
    one epoch and observed in the next, and the lead field relating it to tissue
    is one object either way -- so a cortico-cortical evoked potential experiment,
    which stimulates one contact and records at all the others, is in ibm-1 a
    clamp on one component and likelihoods on the rest of the same array.

    that class of experiment is unusually valuable to a model like this one,
    because it measures effective connectivity directly rather than inferring it
    from correlation.  the latency and amplitude of a response at a distant
    contact is close to a direct observation of `tract_propagation`'s delay and
    gain, which almost nothing else available in humans is.

    mostly a human technique, unusually -- it happens during epilepsy monitoring
    and during awake mapping -- and the coverage is therefore clinical rather than
    scientific, which is a property of any dataset built from it.""",
    constrains=sel("device.contact_potential", region=IMPLANT, band=PULSE),
    modality="direct_cortical_stimulation",
    waveform="bipolar biphasic pulse train, 0.3-1 ms per phase, 1-50 Hz",
    frame="electrode_grid",
))

OPTOGENETIC = I(Intervention(
    id="optogenetic_stimulation",
    doc="""optogenetic stimulation.  **non-human**: this requires transgenic or
    virally transduced opsin expression and is not a human technique.

    two things about this declaration are unsatisfying and both are recorded
    rather than hidden.

    first, the component.  the contract declares no optical-power component, so
    the clamped state is `device.transducer_drive` at an implanted emitter, which
    is the nearest declared thing to a fibre-coupled LED.  the path from that
    drive to a depolarizing current in an expressing cell -- light propagation and
    scattering in tissue, opsin kinetics, channel conductance -- has no
    implementation in the inventory at all, so this intervention currently clamps
    a component whose downstream coupling is undeclared.  that is a hole, and
    naming it here is better than declaring an optogenetic-specific process that
    would violate the architecture's own rule about not inventing primitives.

    second, and more interesting: optogenetics is cell-type specific, and cell
    type in ibm-1 is a *component* -- `neural.pv.activity`, `neural.sst.activity`,
    `neural.vip.activity` are separate state variables at the same position.  so
    the natural expression of "stimulate PV interneurons" is a clamp on a neural
    component rather than on a device one, and the reason this declaration does
    not do that is that it would skip the physics entirely.  the honest fix is an
    opsin implementation of a coupling process, not a different intervention.

    it is declared despite all of this because a very large fraction of what is
    known about the circuits this model describes comes from optogenetic
    experiments, and a framework that cannot express the manipulation cannot use
    the evidence.""",
    constrains=sel("device.transducer_drive", region=STIMULATOR, band=Band(0.0, 1000.0)),
    modality="optogenetic",
    waveform="light pulse train, 1-100 Hz, at an opsin-matched wavelength",
    frame="array",
))


# ---------------------------------------------------------------------------
# sensory
# ---------------------------------------------------------------------------

VISUAL_STIMULUS = I(Intervention(
    id="visual_stimulus",
    doc="""a visual stimulus: display luminance is clamped.

    the declaration that most directly cashes out §1's claim that a stimulus is
    not an input to the model.  what an experiment controls is a screen, which is
    `device.display_luminance` on the display support -- a state variable with a
    value and a frame -- and the retina reads it through `transduction`.  there is
    no stimulus channel, no input layer, and no boundary condition anywhere in
    the graph.

    the band ceiling is the display's refresh rate rather than the retina's
    bandwidth, deliberately.  the receptor's low-pass belongs to the
    photoreceptor implementation, where evidence can move it; putting it in the
    clamp would assert that the screen cannot show a 90 Hz flicker, which it can,
    and would silently forbid the model from representing what happens when it
    does.

    the gap this declaration sits next to is the one named in
    `ibm.processes.transduction`: the ontology has no radiometric component and
    no optics, so the path from a screen at a viewing distance to an irradiance
    pattern on the retina is absorbed into a gain.  a stimulus's *content* --
    which is what most visual neuroscience is about -- is therefore not
    representable here at all, and belongs to a stimulus encoder used as a
    distillation target rather than to this file.""",
    constrains=sel("device.display_luminance", region=DISPLAY, band=VISUAL),
    modality="visual",
    waveform="arbitrary image sequence at the display refresh rate",
    frame="display",
))

AUDITORY_STIMULUS = I(Intervention(
    id="auditory_stimulus",
    doc="""an auditory stimulus: speaker or headphone pressure is clamped.

    the same shape as the visual clamp, and with a band that is genuinely
    wide -- 20 Hz to 20 kHz -- which makes it the widest-band object anywhere in
    ibm-1 by a factor of four.  that is a real cost and a real fact: a
    materialization that wants to represent an acoustic stimulus faithfully needs
    a temporal basis fine enough for it, and almost none will.  most auditory
    work will clamp an envelope instead, and the wide band declared here is what
    makes that a recorded approximation rather than an assumption.

    the floor at 20 Hz is not a convention.  the middle ear transmits essentially
    nothing below it, so a clamp extending to DC would declare a coupling the
    anatomy does not have.""",
    constrains=sel("device.speaker_pressure", region=DISPLAY, band=AUDIO),
    modality="auditory",
    waveform="arbitrary acoustic waveform",
    frame="audio",
))

TACTILE_STIMULUS = I(Intervention(
    id="tactile_stimulus",
    doc="""a tactile stimulus: a vibrotactile or indenting actuator's drive is
    clamped.

    clamped at the device rather than at the skin, and the reason is a gap worth
    naming: the mechanical field's declared support is the head volume, so there
    is nowhere in the current ontology to put a skin indentation at a fingertip.
    clamping the actuator's drive keeps the declaration honest about where the
    experimenter's control actually is, and leaves the coupling to the skin as a
    piece of missing physics rather than as an assumed one.

    the band runs to 1 kHz because pacinian afferents do, and because the
    classical psychophysics of vibrotaction -- flutter around 30 Hz, vibration
    around 250 Hz, mediated by different afferent classes -- is entirely a
    frequency story.  a narrower clamp would make the two conditions
    indistinguishable at the source.""",
    constrains=sel("device.transducer_drive", region=STIMULATOR, band=Band(0.0, 1000.0)),
    modality="tactile",
    waveform="indentation step or sinusoidal vibration, 5-500 Hz",
    frame="body",
))

VESTIBULAR_STIMULUS = I(Intervention(
    id="vestibular_stimulus",
    doc="""galvanic vestibular stimulation: current at the mastoids.

    clamps `device.contact_potential` at body-surface electrodes, so it is the
    same physics as tDCS applied to a different place -- current polarizes the
    vestibular afferents where they leave the labyrinth, biasing their firing and
    producing a compelling illusion of tilt.  no new machinery, which is the
    point.

    two alternatives exist and are not declared separately because they are not
    interventions on device state at all.  caloric irrigation works by setting up
    a convection current in the endolymph, which is a thermal and mechanical
    manipulation; whole-body rotation on a chair is a mechanical one.  both would
    clamp `thermal.temperature` or `mechanical.velocity` respectively, and both
    are covered by the thermal declaration below and by an ordinary mechanical
    boundary condition.  the fact that three quite different laboratory
    procedures land on three already-declared components, with no vestibular-
    specific object anywhere, is the architecture working as intended.""",
    constrains=sel("device.contact_potential", region=Near("mastoid", 20.0),
                   band=Band(0.0, 50.0)),
    modality="vestibular",
    waveform="bipolar direct current or low-frequency sinusoid, 0.5-3 mA",
    frame="body",
))

OLFACTORY_STIMULUS = I(Intervention(
    id="olfactory_stimulus",
    doc="""an olfactometer delivering an odorant.

    the one sensory intervention clamped at the receptor rather than at a device,
    and the asymmetry is honest rather than convenient.  a display and a speaker
    are declared device components with real state; an olfactometer's delivered
    concentration at the epithelium is not a declared component, and inventing
    one would be adding a primitive to describe an experimental convenience.
    what *is* declared is `transduction.chemoreceptor` -- receptor occupancy --
    and §1 says in as many words that a stimulus is an intervention on
    transduction state, so clamping it is the architecture's own prescription
    rather than a workaround.

    the price is that the clamp skips the delivery dynamics: the rise time of an
    odorant at the epithelium is comparable to a sniff, and sniffing is part of
    smelling rather than incidental to it, so a step clamp on occupancy asserts a
    delivery this instrument cannot achieve.  the `tau_diffusion_s` parameter of
    the chemoreceptor implementation is the thing being bypassed, and a
    materialization that cares about olfactory timing should not use this clamp.

    the band is narrow because nothing chemosensory is fast.""",
    constrains=sel("transduction.chemoreceptor", region=EPITHELIUM, band=Band(0.0, 5.0)),
    modality="olfactory",
    waveform="odorant pulse, typically 0.5-3 s, sniff- or breath-locked",
    frame="body",
))

SENSORY_DEPRIVATION = I(Intervention(
    id="sensory_deprivation",
    doc="""darkness, earplugs, an eyes-closed resting state: the stimulus
    components clamped to a constant.

    included precisely because it looks like it should not need to be.  in a
    framework with stimulus channels, "no stimulus" is the absence of an input
    and there is nothing to declare; in ibm-1 a component that is read but never
    written is an error unless something clamps it, so an eyes-closed resting
    state is an *intervention* -- a clamp to a constant value -- and has to be
    written down.

    that is not bureaucracy.  a resting-state recording is one of the most
    common designs in human neuroscience and the thing that makes it interpretable
    is exactly that the sensory boundary is held fixed; declaring it as a clamp
    means the model knows the boundary is fixed rather than unknown, which is a
    much stronger statement and a much more useful one.  the difference shows up
    immediately in inference: an unclamped display component is a free variable
    the posterior must integrate over, and a clamped one is not.

    it also makes the deprivation *designs* -- monocular deprivation, dark
    rearing, sensory restriction -- the same object as an ordinary stimulus, which
    is right: they differ from a flickering checkerboard only in the value of the
    clamp.""",
    constrains=sel("device.display_luminance", "device.speaker_pressure",
                   region=DISPLAY, band=VISUAL),
    modality="deprivation",
    waveform="constant, typically zero",
    frame="display",
))


# ---------------------------------------------------------------------------
# pharmacological and thermal
# ---------------------------------------------------------------------------

PHARMACOLOGICAL = I(Intervention(
    id="pharmacological",
    doc="""a systemically or locally administered drug acting on a
    neuromodulator system.

    the clamp is on extracellular neuromodulator concentration, and for drugs
    that act by changing how much transmitter is present -- a reuptake inhibitor,
    a releasing agent, a precursor, a degradation blocker -- that is very nearly
    the right object.  L-DOPA raises striatal dopamine; an SSRI raises
    extracellular serotonin; caffeine acts on adenosine receptors and is at least
    adjacent to clamping adenosine's effect.

    for receptor agonists and antagonists it is not the right object at all, and
    the module docstring says why: a drug that occupies a receptor changes the
    *gain* of a coupling, which is a process parameter, and §4's parameter-writing
    mechanism is where it belongs.  clamping a concentration in its place is an
    approximation that gets the sign and the timescale right and the mechanism
    wrong, and will mispredict anything involving competition at the receptor,
    partial agonism, or a drug whose target is not the transmitter its receptor
    binds.

    the timescale is worth carrying: a systemic drug arrives over minutes and
    leaves over hours, so the clamp is slow and long, and the band reflects that.
    a pharmacological experiment is not a trial-locked manipulation and cannot be
    analysed as one.""",
    constrains=sel("extracellular.dopamine", "extracellular.serotonin",
                   "extracellular.acetylcholine", "extracellular.noradrenaline",
                   "extracellular.adenosine", band=SLOW),
    modality="pharmacological",
    waveform="pharmacokinetic time course: minutes to onset, hours to clearance",
    frame="subject_t1",
))

ANAESTHETIC = I(Intervention(
    id="anaesthetic",
    doc="""a general anaesthetic or sedative.

    declared separately from the general pharmacological case because it is the
    clearest instance of the mismatch that case describes, and because it is
    scientifically important enough that the mismatch should be visible rather
    than averaged into a category.

    propofol and the benzodiazepines act at the GABA-A receptor and their
    principal effect is to *prolong the decay* of the inhibitory conductance --
    they change `neural.inh.gaba_a`'s time constant, which is a parameter of the
    local inhibition process, not the amount of GABA present.  the resulting
    change in the excitation-inhibition balance is what produces the
    characteristic slowing and the alpha-band anteriorization that make
    anaesthetic EEG so recognizable, and it is a change in the *dynamics*, which
    no clamp on a concentration can produce.  what this declaration achieves is
    approximately the right direction of effect through the wrong mechanism.

    the correct expression is a process that writes `local_inhibition`'s theta,
    exactly as `neuromodulation` and `plasticity` write parameters.  the schema
    supports it and no such process is declared, so this intervention is a
    placeholder for one.

    it is worth carrying anyway: anaesthesia datasets traverse states of
    consciousness that no other manipulation reaches, and they are among the most
    informative recordings available about what the excitation-inhibition balance
    actually does.""",
    constrains=sel("extracellular.gaba", "extracellular.glutamate", band=SLOW),
    modality="anaesthetic",
    waveform="target-controlled infusion or stepwise concentration levels",
    frame="subject_t1",
))

THERMAL_STIMULUS = I(Intervention(
    id="thermal_stimulus",
    doc="""a thermode, a cooling cap, or caloric irrigation: temperature is
    clamped.

    the same declaration serves three quite different laboratory purposes, which
    is a small piece of evidence that the ontology is carved in the right place.
    a contact thermode delivering a 48 C ramp is a nociceptive stimulus reaching
    `transduction` through the thermoreceptor and nociceptor implementations; a
    cooling cap during a long scan is a safety and comfort measure that also
    changes `thermal_diffusion` and, through temperature dependence, every rate
    constant in the model; caloric irrigation of the ear canal is a *vestibular*
    stimulus that works by convection in the endolymph.

    none of them needs its own object.  they differ in where the clamp is applied
    and to what value.

    the region is the head volume, which is where the contract puts the thermal
    field, and that is a limitation for the thermode case in exactly the way the
    tactile intervention's is -- there is no skin support to clamp.  the caloric
    and cooling cases are properly in the head and are well served.""",
    constrains=sel("thermal.temperature", region=HEAD, band=SLOW),
    modality="thermal",
    waveform="ramp-and-hold, or a sustained offset from baseline",
    frame="subject_t1",
))


__all__ = [
    "TMS", "TDCS", "TACS", "TRNS", "TFUS",
    "DBS", "INVASIVE_ELECTRICAL", "OPTOGENETIC",
    "VISUAL_STIMULUS", "AUDITORY_STIMULUS", "TACTILE_STIMULUS",
    "VESTIBULAR_STIMULUS", "OLFACTORY_STIMULUS", "SENSORY_DEPRIVATION",
    "PHARMACOLOGICAL", "ANAESTHETIC", "THERMAL_STIMULUS",
]

# ---------------------------------------------------------------------------
# manipulations the model library asks for that the inventory above did not cover
# ---------------------------------------------------------------------------

TASK_CUE = I(Intervention(
    id="task_cue",
    doc="""an instruction to the participant: attempt this movement, read this
    character, imagine saying this word, practise on this schedule.

    the awkward one, and worth being explicit about rather than quietly folding
    into a sensory stimulus.  what is physically imposed is the cue -- photons
    from a screen, pressure from a speaker -- and that part is an ordinary
    `visual_stimulus` or `auditory_stimulus`.  what the experiment actually
    manipulates is the participant's compliance with an instruction, and ibm-1
    contains no state variable for an intention.

    so this clamps the cue's delivery and records that the manipulation of
    interest is not the clamped variable.  a decoding materialization fitted
    against it is conditioning on the experimenter's design, which is exactly the
    imposed-versus-measured distinction the data schema exists to keep visible:
    treat a cue as jointly sampled and the model learns the task structure as
    though it were biology.""",
    constrains=sel("device.display_luminance", "device.speaker_pressure"),
    modality="behavioural", waveform="event_sequence", frame="display"))

RESPIRATORY_CHALLENGE = I(Intervention(
    id="respiratory_challenge",
    doc="""breath-hold, hypercapnia, or hyperoxia: the inspired gas or the
    breathing pattern is imposed.

    the standard way to drive the vascular system without driving the neural
    one, which is what makes it the calibration lever for every haemodynamic
    materialization -- it separates the vascular transfer function from the
    neural drive that ordinarily confounds it.

    clamped on blood gas rather than on airway pressure because that is what the
    downstream chemistry reads, and because end-tidal CO2 is what gets recorded.
    the neural system is not untouched -- hypercapnia is mildly sedating and
    changes excitability -- so the assumption that this isolates the vasculature
    is an approximation, and materializations that rely on it should say so.""",
    constrains=sel("blood.oxygenation", "blood.oxygen_content"),
    modality="physiological", waveform="block", frame="body"))

GRAPH_ABLATION = I(Intervention(
    id="graph_ablation",
    doc="""remove a node or a tract from the materialized graph.

    not a physical intervention at all, and declared here because pretending
    otherwise would be worse.  a virtual lesion imposes a value -- zero coupling
    -- on state the model contains, which is formally the same operation as any
    other clamp, but nothing was done to any participant and no measurement
    constrains the result.

    the consequence is stated in `virtual_lesion`'s provenance rather than hidden
    here: its counterfactual is a property of the process graph, not an
    observation, and it is falsifiable only against genuinely interventional
    records such as direct cortical stimulation.  a lesion prediction that has
    never met one of those is a hypothesis the model generated about itself.""",
    constrains=sel("structural.axonal_density", "structural.synaptic_density"),
    modality="model", waveform="step", frame="subject_t1"))
