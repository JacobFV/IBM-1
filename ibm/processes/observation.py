"""observations: what each instrument is evidence *about*, and over what band.

ARCHITECTURE.md §6 is emphatic and the emphasis is the whole design of this
file: an observation is not an ibm-1 primitive.  it is external evidence about
an ordinary state variable that an ordinary process produced.  eeg is not a
modality with its own forward machinery; it is a likelihood attached to
`device.contact_potential`, which `device_coupling` computed from
`electromagnetic.potential`, which `em_generation` computed from
`neural.transmembrane_current`, which the local and laminar processes computed
from population state.  the `via` field on every declaration below names that
chain explicitly, and it is checked at seal time -- a chain naming an
unregistered process is an error, not a comment.

the consequence is that this file contains no physics.  every parameter that
describes how an instrument responds to state lives in an implementation of
`device_coupling` or of the physiological processes upstream, because those are
statements about the state, not about our access to it.  what is left here is
four things per instrument: which component it constrains, what the noise looks
like, which coordinate frame the evidence arrives in, and -- the one that carries
most of the weight -- the **band over which it carries any precision at all**.

that band is the crux, and it is worth being explicit about why.

state in ibm-1 is a belief over temporal laplacian components, and evidence
fusion adds precision component by component (`SpectralGaussian.evidence`).  an
fMRI run therefore contributes enormous precision below about 0.25 Hz and
*literally none* above it -- not a little, not a smoothed amount, none -- because
a 2 s TR cannot constrain a 10 Hz component of anything.  a spike-band recording
contributes nothing below its 300 Hz high-pass, for the symmetric reason.  an
EEG contributes between roughly 0.1 and 100 Hz, bounded below by the amplifier's
AC coupling and above by its anti-alias filter, and bounded further by the skull,
which is a low-pass in *space* whose effect looks like a loss of high-frequency
sensitivity because fast cortical activity is spatially fine-grained.

writing those numbers down per observation is what makes heterogeneous sources
compose without conflict, and it is not a preprocessing detail.  two instruments
that disagree about a state variable can only disagree where their bands
overlap; where they do not overlap there is nothing to reconcile, and no
weighting scheme, no hyperparameter and no special case is needed to say so.  a
simultaneous EEG-fMRI dataset is not a fusion problem in ibm-1.  it is two
likelihood terms on two components of the same graph over two disjoint bands.

three conventions used throughout, and one substitution.

*bands are the instrument's, not the physiology's.*  an fNIRS system samples at
10 Hz and genuinely resolves the cardiac pulsation, so its band runs to a few
hertz even though the neurally driven part of its signal does not.  narrowing
the band to the interesting part would be discarding real evidence about blood
volume.

*the noise field names a family, not a variance.*  the variance is per-dataset
and belongs in the source card, not in the ontology.  what belongs here is
whether the noise is gaussian, poisson (photon and positron counting), or
something with structure the gaussian family cannot carry.

*`via` names the shortest honest chain*, not every process whose state
eventually influences the observed one.  it terminates at the process that
writes the observed component.

one interaction between an observation's band and the ontology's own is worth
naming once rather than repeating.  a component declares the widest band over
which it is *meaningful*, and an observation declares the band over which the
instrument carries precision.  the effective evidence is the intersection, so
where the component is narrower the top of the instrument's band constrains
nothing.  that bites in exactly two places below: surface EMG runs to 500 Hz
while `effector.activation` is declared meaningful only to 100, and a microphone
runs to 20 kHz while `mechanical.pressure` stops at 500.  the instrument's band
is what is written down, because that is a fact about the instrument and will
outlive any particular choice of component ceiling -- but a materialization
should expect no precision above the narrower of the two.

and the substitution: the contract declares no magnetometer-output component and
no optical-power component, so MEG attaches to `electromagnetic.bfield` and
fNIRS attaches to the blood components directly, with the sensor's own transfer
folded into the observation's band.  this is named in `ibm.processes.device` as
well; it is recorded twice because it is the kind of thing that otherwise turns
into an unexamined convention.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Observation
from ibm.vocabulary import (
    Band,
    LFP,
    Near,
    OnSupport,
    SPIKE,
    STRUCTURAL,
    sel,
)

O = REGISTRY.observation

# ---------------------------------------------------------------------------
# bands, named once so that two instruments claiming the same sensitivity say so
# in the same words
# ---------------------------------------------------------------------------

#: scalp EEG.  the floor is the amplifier's AC coupling and the ceiling is its
#: anti-alias filter; between them the skull and scalp act as a spatial low-pass
#: that costs high-frequency sensitivity, because fast cortical activity is
#: spatially fine-grained and therefore cancels across the smoothing.
EEG_BAND = Band(0.1, 100.0)

#: MEG.  the same physiology and a wider usable top end, because the skull is
#: nearly transparent magnetically and does not smooth the field the way it
#: smooths the potential.
MEG_BAND = Band(0.1, 200.0)

#: intracranial contacts.  no skull, so high gamma survives, and the amplifier
#: rather than the head sets both ends.
ICE_BAND = Band(0.1, 500.0)

#: fMRI.  a 2 s TR gives a nyquist of 0.25 Hz, and the haemodynamic response is
#: itself a low-pass well below that.  above this the run contributes nothing.
FMRI_BAND = Band(0.0, 0.25)

#: arterial spin labelling.  slower still: a tag-control pair per measurement
#: halves the effective rate, and the label decays with the blood T1.
ASL_BAND = Band(0.0, 0.06)

#: fNIRS.  the instrument samples fast enough to resolve the cardiac pulse, and
#: that pulse is real evidence about blood volume rather than an artefact to be
#: filtered before the model sees it.
FNIRS_BAND = Band(0.0, 3.0)

#: PET.  frames are 30 s to several minutes, so the constrained band is three
#: orders of magnitude narrower than fMRI's and one below the slowest thing the
#: physiological processes represent.
PET_BAND = Band(0.0, 0.005)

#: surface EMG.  the floor removes movement artefact and the ceiling is where
#: motor unit action potential energy has run out.
EMG_BAND = Band(10.0, 500.0)

#: kinematics: eye position, limb position, chest wall.  saccades are the fastest
#: thing here and they are done in tens of milliseconds.
KINEMATIC_BAND = Band(0.0, 100.0)

#: slow autonomic and respiratory signals.
AUTONOMIC_BAND = Band(0.0, 10.0)

#: produced speech.
AUDIO_BAND = Band(50.0, 8000.0)

# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------

SENSORS = OnSupport("sensor_array")
IMPLANT = OnSupport("implanted_array")
HEAD = OnSupport("head_volume")
BODY = OnSupport("body")
MOTOR_UNITS = OnSupport("motor_units")
VOCAL_TRACT = OnSupport("vocal_tract")
VASCULAR = OnSupport("vascular_tree")

# ---------------------------------------------------------------------------
# chains, named once.  these are the `via` tuples, factored out because several
# observations genuinely share one and writing it twice invites them to drift.
# ---------------------------------------------------------------------------

ELECTRIC_CHAIN = ("local_excitation", "local_inhibition", "laminar_propagation",
                  "em_generation", "device_coupling")
HEMODYNAMIC_CHAIN = ("neurovascular_coupling", "vascular_flow", "tissue_exchange")
MOTOR_CHAIN = ("efferent_propagation", "effector_activation")


# ---------------------------------------------------------------------------
# electromagnetic
# ---------------------------------------------------------------------------

EEG = O(Observation(
    id="eeg",
    doc="""scalp electrode voltage: the workhorse, and the clearest instance of
    the architecture's point.

    what is observed is `device.contact_potential` -- an ordinary state variable
    on the sensor array, produced by `device_coupling` from the head-volume
    potential, which `em_generation` produced from transmembrane current.  the
    observation adds nothing to that chain; it says only that a number was
    recorded at a contact and that the number is informative about the contact's
    state to within the amplifier's noise.

    the band is where the honesty lives.  the floor is the amplifier's AC
    coupling, so a slow cortical potential is attenuated by hardware before any
    analysis; declaring the floor here rather than pretending EEG constrains DC
    is the difference between a model that can be told about contingent negative
    variation and one that cannot.  the ceiling is the anti-alias filter, and
    above it the observation contributes exactly zero precision -- which is what
    makes it composable with a spike-band recording that contributes exactly zero
    below 300 Hz.

    what the band does *not* capture: EEG's real limitation is spatial, not
    temporal.  the skull and CSF smear the potential over centimetres, and
    opposing gyral banks cancel, so a large fraction of cortical activity
    produces no scalp signal at any frequency.  that is carried by the lead field
    in `device_coupling`, where it belongs, and a materialization that reads this
    band as "EEG sees everything below 100 Hz" has misread it.""",
    observes=sel("device.contact_potential", region=SENSORS, band=EEG_BAND),
    modality="eeg",
    noise="gaussian",
    band=EEG_BAND,
    frame="eeg_cap",
    via=ELECTRIC_CHAIN,
))

MEG = O(Observation(
    id="meg",
    doc="""magnetometer and gradiometer output.

    attached to `electromagnetic.bfield` rather than to a device component,
    because the contract declares none for a magnetometer.  the cost of that
    substitution is that the sensor's own transfer function is folded into this
    band and noise model instead of being carried as device state; the benefit is
    that MEG and fMRI attach to the *same* component -- `electromagnetic.bfield`
    is what `bold_formation` writes too -- at bands three orders of magnitude
    apart, which is a rather good demonstration that the band is doing the work
    of separating them and not a modality label.

    MEG's band runs slightly higher than EEG's for a physical reason worth
    stating: the skull is nearly transparent to magnetic fields, so the spatial
    low-pass that costs EEG its high-frequency sensitivity does not apply, and
    high-gamma is measurable at the scalp magnetically where it is marginal
    electrically.  the compensating loss is that MEG is blind to radial sources
    and therefore to gyral crowns, which is a *spatial* selectivity carried in
    the forward model rather than a band.""",
    observes=sel("electromagnetic.bfield", region=HEAD, band=MEG_BAND),
    modality="meg",
    noise="gaussian",
    band=MEG_BAND,
    frame="meg_head",
    via=ELECTRIC_CHAIN,
))

IEEG = O(Observation(
    id="ieeg",
    doc="""stereo-EEG depth contacts.

    the same declaration as scalp EEG with the skull removed, and the removal
    changes the band ceiling by a factor of five: without the volume conductor's
    smoothing, high-gamma activity that is invisible at the scalp is large at a
    depth contact.  the floor is unchanged, because it is the amplifier's and
    amplifiers are the same.

    the thing an ontology can easily get wrong here is coverage.  depth
    electrodes are placed for clinical reasons in patients with epilepsy, so the
    sampling is sparse, idiosyncratic and non-random with respect to the
    pathology -- and no band or noise model expresses that.  it is a property of
    a dataset rather than of the instrument, and it belongs in a source card.""",
    observes=sel("device.contact_potential", region=IMPLANT, band=ICE_BAND),
    modality="ieeg",
    noise="gaussian",
    band=ICE_BAND,
    frame="electrode_grid",
    via=ELECTRIC_CHAIN,
))

ECOG = O(Observation(
    id="ecog",
    doc="""subdural grid and strip electrodes.

    electrically the same object as a depth contact and geometrically a very
    different one: a grid sits on the pial surface and integrates over a few
    millimetres of gyral crown, which makes it excellent for the crowns and
    nearly blind to the sulcal walls that hold most of the cortical surface.
    that is a lead-field statement, carried in `device_coupling`.

    what is worth recording here is that ECoG high-gamma is the closest thing in
    non-invasive-adjacent human neuroscience to a local firing rate proxy, and
    the band ceiling is what makes that claim expressible: the observation
    constrains state up to 500 Hz, where the correlation between broadband high
    frequency power and population spiking is measurable.  a declaration capped
    at 100 Hz would silently forbid the model from using it.""",
    observes=sel("device.contact_potential", region=IMPLANT, band=ICE_BAND),
    modality="ecog",
    noise="gaussian",
    band=ICE_BAND,
    frame="electrode_grid",
    via=ELECTRIC_CHAIN,
))

INTRACORTICAL_SPIKES = O(Observation(
    id="intracortical_spikes",
    doc="""sorted single- and multi-unit activity from an intracortical array.

    the one observation in this file that attaches directly to a neural
    component rather than to a device one, and the reason is that a sorted spike
    train is already the product of an inference -- filtering, thresholding,
    clustering -- whose output is an estimate of a population's firing, not of a
    voltage.  attaching it to `device.contact_potential` would require the model
    to redo spike sorting internally, which is not the division of labour
    anybody wants.  the cost of the shortcut is that sorting errors become
    unmodelled noise, and sorting errors are neither small nor gaussian.

    the band is the sharpest illustration of the composition rule in the whole
    file.  a spike-band recording is high-passed at 300 Hz, so it constrains
    nothing at all below that -- it cannot tell you about the theta rhythm the
    same electrode is sitting in, because the filter that made spike detection
    possible removed it.  a spike observation and an LFP observation from the
    *same physical contact* are therefore two declarations over two disjoint
    bands, and that is not a modelling awkwardness, it is exactly what the
    hardware does.""",
    observes=sel("neural.exc.activity", "neural.inh.activity",
                 region=Near("array", 0.15), band=SPIKE),
    modality="spikes",
    noise="poisson",
    band=SPIKE,
    frame="array",
    via=("local_excitation", "local_inhibition"),
))

LOCAL_FIELD_POTENTIAL = O(Observation(
    id="lfp",
    doc="""local field potential from an intracortical or depth contact.

    the low-passed complement of the spike observation, from the same contact
    and over the disjoint band.  between roughly 0.5 and 300 Hz the LFP is
    dominated by synaptic currents in a volume of a few hundred micrometres to a
    few millimetres, which makes it evidence about `device.contact_potential`
    and, through the lead field, about transmembrane current.

    the interpretive trap this declaration is careful not to fall into: the LFP
    is not a local firing rate.  it is dominated by synaptic input to the region
    rather than by its output, so a region receiving strong input and firing
    nothing produces a large LFP.  nothing here asserts otherwise -- the chain
    runs through `em_generation` from transmembrane current, which is the
    physically correct source -- and the fact that the declaration cannot be read
    as "LFP measures spiking" is a consequence of putting the physics in the
    process rather than in the observation.""",
    observes=sel("device.contact_potential", region=IMPLANT, band=LFP),
    modality="lfp",
    noise="gaussian",
    band=LFP,
    frame="array",
    via=ELECTRIC_CHAIN,
))


# ---------------------------------------------------------------------------
# haemodynamic and metabolic
# ---------------------------------------------------------------------------

BOLD = O(Observation(
    id="bold",
    doc="""blood-oxygenation-level-dependent fMRI.

    the canonical example of why bands are declared per observation.  a 2 s TR
    puts the nyquist frequency at 0.25 Hz, and the haemodynamic response is
    itself a low-pass with a corner well below that, so a whole fMRI session --
    hours of scanner time, gigabytes of data -- contributes precision over a band
    a hundredth as wide as a two-minute EEG recording, and *no* precision at all
    outside it.  fusing it with electrophysiology therefore requires no
    weighting: the two constrain disjoint components and the arithmetic in
    `SpectralGaussian.evidence` handles it with no special case.

    what is observed is `electromagnetic.bfield`, because that is what
    `bold_formation` writes: deoxyhaemoglobin's susceptibility perturbs the
    static field and the T2*-weighted magnitude is the dephasing that
    perturbation causes.  attaching the observation there rather than to an
    invented mr-signal component is what lets fMRI and MEG constrain one field.

    the noise is called gaussian, which is a simplification worth naming: fMRI
    noise is strongly autocorrelated, has a large 1/f component from
    physiological sources, and is heavier-tailed than gaussian because of motion.
    the spectral form can carry the autocorrelation exactly -- coloured noise is
    a psd, not a nuisance -- so the simplification is really only about the
    tails.""",
    observes=sel("electromagnetic.bfield", region=HEAD, band=FMRI_BAND),
    modality="fmri",
    noise="gaussian",
    band=FMRI_BAND,
    frame="subject_bold",
    via=HEMODYNAMIC_CHAIN + ("bold_formation",),
))

ASL_PERFUSION = O(Observation(
    id="asl_perfusion",
    doc="""arterial spin labelling: a quantitative measure of cerebral blood flow.

    slower than BOLD and more valuable per sample, for a reason the band alone
    does not convey.  ASL observes `blood.flow` in physical units --
    mL/100g/min -- rather than a dimensionless signal change, which makes it the
    only routine human measurement that can constrain the *baseline* of the
    haemodynamic chain rather than its perturbation.  everything about BOLD is
    relative; a model fitted to BOLD alone has an unidentifiable baseline flow
    and a compensating gain, and ASL is what breaks that degeneracy.

    the band is narrow because each measurement is a tag-control difference and
    the label decays with the blood T1 of about 1.6 s, so the achievable
    repetition rate is low and the SNR per pair is poor.  that poor SNR is the
    reason ASL is not simply better than BOLD everywhere.""",
    observes=sel("blood.flow", region=VASCULAR, band=ASL_BAND),
    modality="asl",
    noise="gaussian",
    band=ASL_BAND,
    frame="subject_bold",
    via=("neurovascular_coupling", "vascular_flow"),
))

FNIRS = O(Observation(
    id="fnirs",
    doc="""functional near-infrared spectroscopy.

    attached to the blood components directly, because the contract has no
    optical component; the modified beer-lambert coupling lives in
    `device_coupling`.  what fNIRS uniquely offers is that it separates
    oxy- and deoxyhaemoglobin, where BOLD sees a single susceptibility-weighted
    combination -- so it constrains two components of the model where fMRI
    constrains one.

    the band runs to a few hertz, higher than fMRI's, and that is not a mistake.
    fNIRS samples at 10 Hz and resolves the cardiac pulsation directly, which is
    genuine evidence about blood volume; treating it as an artefact to be
    filtered before the model sees it would be discarding data because it is
    inconvenient rather than because it is uninformative.

    the honest limitation is spatial and it is severe: sensitivity peaks about
    1.5 cm below the scalp and a majority of the detected change is
    extracerebral.  that is in the coupling's `extracerebral_fraction`, not
    here.""",
    observes=sel("blood.deoxyhemoglobin", "blood.oxygenation", "blood.volume",
                 region=VASCULAR, band=FNIRS_BAND),
    modality="fnirs",
    noise="gaussian",
    band=FNIRS_BAND,
    frame="nirs_cap",
    via=HEMODYNAMIC_CHAIN,
))

PET = O(Observation(
    id="pet",
    doc="""positron emission tomography: glucose metabolism, oxygen consumption
    or receptor binding, depending on the tracer.

    the narrowest band in the inventory by two orders of magnitude.  frames are
    tens of seconds to minutes, so PET constrains state below about 0.005 Hz and
    nothing above -- which is to say it constrains a *baseline* and not a
    dynamic.  that is exactly what makes it valuable next to fMRI: FDG-PET gives
    absolute metabolic rate in physical units where BOLD gives a relative
    change, so it pins the metabolic chain's operating point in the same way ASL
    pins the vascular one.

    the noise is poisson and saying so matters, because it is one of the few
    places in this file where the gaussian family is genuinely wrong rather than
    merely approximate: a late frame of a decaying tracer has few counts, the
    distribution is skewed, and a gaussian likelihood will systematically
    misplace low-uptake regions.""",
    observes=sel("metabolic.consumption", "metabolic.glucose", "metabolic.oxygen",
                 band=PET_BAND),
    modality="pet",
    noise="poisson",
    band=PET_BAND,
    frame="subject_t1",
    via=("metabolism", "tissue_exchange"),
))


# ---------------------------------------------------------------------------
# structural
# ---------------------------------------------------------------------------

DWI = O(Observation(
    id="dwi_microstructure",
    doc="""diffusion-weighted imaging: fibre orientation and microstructural
    indices.

    band `STRUCTURAL`, which is 0 to 0.001 Hz -- effectively DC.  that is not a
    limitation of the scanner; it is a statement that the state being observed
    does not change on any timescale a session can see.  the same declaration
    would be wrong for a longitudinal study spanning months, where plasticity is
    exactly the process of interest, and the band is what a materialization would
    have to widen to say so.

    what DWI actually constrains is orientation and hindrance, not axons.  fibre
    orientation distributions are well identified; axonal density and myelination
    are inferred through a model whose assumptions -- stick compartments, fixed
    intrinsic diffusivity -- are known to be violated, so the observation is
    listed as constraining all three while the confidence across them is very
    uneven.  crossing fibres, present in the majority of white matter voxels,
    remain the dominant source of error in any tractogram built from this.""",
    observes=sel("structural.fiber_orientation", "structural.axonal_density",
                 "structural.myelination", band=STRUCTURAL),
    modality="dwi",
    noise="gaussian",
    band=STRUCTURAL,
    frame="subject_dwi",
    via=("plasticity",),
))

STRUCTURAL_MRI = O(Observation(
    id="structural_mri",
    doc="""anatomical MRI: tissue composition, geometry and the material state
    the forward models are built on.

    listed as an observation rather than left implicit because it is not only an
    anatomical convenience.  the head model that `device_coupling`'s lead field
    is computed on comes from here, so a structural scan constrains the *material*
    field -- conductivity through tissue segmentation, mass density through
    proton density -- and therefore constrains every electromagnetic observation
    downstream.  treating it as calibration rather than as evidence would hide
    the fact that segmentation error propagates into source localization.

    band `STRUCTURAL` for the same reason as DWI, and with the same caveat about
    longitudinal designs.""",
    observes=sel("material.conductivity", "material.mass_density",
                 "structural.gliosis", "structural.dendritic_density",
                 band=STRUCTURAL),
    modality="mri",
    noise="gaussian",
    band=STRUCTURAL,
    frame="subject_t1",
    via=("plasticity",),
))


# ---------------------------------------------------------------------------
# effector, behaviour and autonomic
# ---------------------------------------------------------------------------

EMG = O(Observation(
    id="emg",
    doc="""surface and intramuscular electromyography.

    evidence about `effector.activation`, and the architecture's claim about
    movement made concrete: a muscle's electrical activity is ordinary state on
    the motor-unit support, reached from cortex by two ordinary processes.  there
    is no "behavioural output" here to be treated differently from a blood
    oxygenation measurement.

    the band's floor is unusually high and is doing real work.  below ~10 Hz
    surface EMG is dominated by movement artefact and electrode-cable motion
    rather than by muscle, so declaring precision down to DC would let a slow
    artefact masquerade as evidence about slow activation -- which is precisely
    the failure mode of EMG envelope analysis.  the ceiling is where motor unit
    action potential energy has run out.

    the caveat worth carrying: surface EMG amplitude depends on how synchronized
    motor units are as much as on how many are active, so it is a better
    observation of `effector.activation` than of `effector.force`, and the
    difference between those two is exactly the `common_drive_fraction` parameter
    in `efferent_propagation`'s recruitment implementation.""",
    observes=sel("effector.activation", region=MOTOR_UNITS, band=EMG_BAND),
    modality="emg",
    noise="gaussian",
    band=EMG_BAND,
    frame="body",
    via=MOTOR_CHAIN,
))

MEP = O(Observation(
    id="mep",
    doc="""motor evoked potential: the EMG response to a single TMS pulse.

    the most complete round trip in the inventory, and the reason it is worth
    declaring separately from ordinary EMG.  the chain runs from an intervention
    clamping `device.coil_current`, through `device_coupling` into the induced
    electric field, through `em_coupling` into cortical population state, through
    `efferent_propagation` down the corticospinal tract, through
    `effector_activation` into muscle -- and this observation attaches at the
    end.  every link is an ordinary process; nothing about stimulation or about
    behaviour required machinery of its own.

    what makes an MEP scientifically valuable is that its amplitude is a readout
    of corticospinal excitability, and what makes it statistically awkward is
    that the amplitude varies several-fold from pulse to pulse at fixed
    stimulator output.  that variability is not measurement noise -- it is
    genuine state variability in the populations being stimulated, so it should
    reduce the posterior over cortical state rather than being absorbed into an
    observation variance.  declaring the observation on `effector.activation`
    with a modest noise term, and letting the variability propagate back through
    the chain, is what makes that possible.""",
    observes=sel("effector.activation", "effector.force", region=MOTOR_UNITS,
                 band=EMG_BAND),
    modality="mep",
    noise="lognormal",
    band=EMG_BAND,
    frame="body",
    via=("device_coupling", "em_coupling", "efferent_propagation", "effector_activation"),
))

BEHAVIOUR = O(Observation(
    id="behaviour",
    doc="""a button press, a lever, a joystick: the discrete behavioural record.

    the plainest statement of §1's claim that a movement is evidence about
    effector state.  a response time is not a special quantity requiring a
    decision model bolted onto the side; it is the time at which
    `effector.force` crossed a switch's threshold, and it constrains the whole
    upstream chain exactly as an EMG trace does, with less bandwidth and more
    convenience.

    the band ceiling is low because a button press is a slow mechanical event and
    a switch closure carries no information about how the force got there.  that
    low bandwidth is the entire cost of behavioural data: it is cheap, abundant,
    and constrains a few tens of milliseconds of timing and one bit of choice per
    trial, which is orders of magnitude less than the neural recording taken
    alongside it.  declaring the band makes that asymmetry arithmetic rather than
    rhetoric.""",
    observes=sel("effector.force", region=MOTOR_UNITS, band=Band(0.0, 20.0)),
    modality="behaviour",
    noise="bernoulli",
    band=Band(0.0, 20.0),
    frame="body",
    via=MOTOR_CHAIN,
))

MOTION_CAPTURE = O(Observation(
    id="motion_capture",
    doc="""optical or inertial motion capture: limb and body kinematics.

    evidence about the mechanical field on the body -- `mechanical.displacement`
    and `mechanical.velocity` -- which `effector_activation` writes.  it is the
    richest behavioural observation available and it constrains the part of the
    motor chain that is otherwise least constrained, because docs/EVIDENCE.md is
    dense on the neural side and thin on the musculoskeletal one.

    two limits.  marker-based capture observes surface markers, not bones, so
    soft-tissue artefact puts several millimetres of correlated error into every
    joint angle, and that error is not white.  and kinematics constrain the
    *output* of the musculoskeletal chain without separating the many
    combinations of muscle activation that produce the same movement -- the
    redundancy problem -- so motion capture alone cannot identify
    `effector_activation`'s parameters, which is why EMG recorded with it is
    worth more than either alone.""",
    observes=sel("mechanical.displacement", "mechanical.velocity", region=BODY,
                 band=KINEMATIC_BAND),
    modality="motion_capture",
    noise="gaussian",
    band=KINEMATIC_BAND,
    frame="body",
    via=MOTOR_CHAIN,
))

EYE_TRACKING = O(Observation(
    id="eye_tracking",
    doc="""gaze position and pupil diameter.

    eye position is limb kinematics with a very small limb: the extraocular
    muscles are effectors, the globe is a mechanical load, and gaze is
    `mechanical.displacement` on the body support.  no separate oculomotor
    machinery is needed, and the same `effector_activation` process that moves an
    arm moves an eye with different parameters.

    the band ceiling is set by saccades, which reach 500 deg/s and complete in
    tens of milliseconds -- fast enough that a 60 Hz tracker aliases them and a
    1 kHz tracker does not, which is a distinction this band makes visible.

    pupil diameter deserves a note because it is routinely and loosely treated as
    an arousal index.  in ibm-1 it would be mechanical state driven by autonomic
    efferents, and the chain from noradrenergic activity to pupil size runs
    through several processes none of which are well constrained -- so reading
    pupil as a direct observation of locus coeruleus firing is an inference this
    declaration deliberately does not license.""",
    observes=sel("mechanical.displacement", "mechanical.velocity", region=BODY,
                 band=KINEMATIC_BAND),
    modality="eye_tracking",
    noise="gaussian",
    band=KINEMATIC_BAND,
    frame="eye",
    via=MOTOR_CHAIN,
))

PRODUCED_AUDIO = O(Observation(
    id="produced_audio",
    doc="""recorded speech and vocalization.

    evidence about `mechanical.pressure` at the vocal tract, which
    `effector_activation` writes.  speech is effector state; a microphone is an
    instrument observing it; and the fact that the content of speech is
    linguistically rich changes nothing about the declaration, which is a claim
    the architecture makes and this file has to honour rather than quietly
    exempt.

    the band runs to 8 kHz, far above anything else on the motor side, and the
    reason is worth stating: vocal fold vibration at 100-300 Hz is not muscle
    contracting at 100-300 Hz -- muscles cannot do that -- it is an aerodynamic
    flutter that muscle merely tunes.  so the observed band and the effector's own
    force band are decoupled, and `effector_activation` declares its vocal tract
    output over the wide band for exactly this reason.

    what this observation does not constrain is anything linguistic.  a speech
    recording is enormously informative about intent, and none of that
    information is expressible through a likelihood on acoustic pressure; it
    would require a model of the mapping, which is a distillation target rather
    than an observation.""",
    observes=sel("mechanical.pressure", region=VOCAL_TRACT, band=AUDIO_BAND),
    modality="audio",
    noise="gaussian",
    band=AUDIO_BAND,
    frame="audio",
    via=MOTOR_CHAIN + ("mechanical_propagation",),
))

ECG = O(Observation(
    id="ecg",
    doc="""electrocardiogram.

    honestly, the most awkward declaration in this file, and the awkwardness is
    informative.  ibm-1 has no cardiac source: the heart is not a declared field,
    so the current dipole that produces the ECG has nowhere to come from within
    the model.  what the observation can legitimately constrain is
    `device.contact_potential` at body electrodes, and through it -- weakly --
    the timing of the cardiac cycle that drives `blood.pressure` and the
    pulsatile component of cerebral blood flow.

    that is not nothing.  cardiac timing is a large and structured contaminant of
    every haemodynamic and electrophysiological measurement, and having it as
    state rather than as a preprocessing step means pulse artefact in EEG and
    cardiac aliasing in fMRI can be *explained* rather than removed.  but the
    declaration is a stand-in for a cardiac field the ontology does not have, and
    it should be replaced rather than elaborated.""",
    observes=sel("device.contact_potential", region=BODY, band=Band(0.05, 150.0)),
    modality="ecg",
    noise="gaussian",
    band=Band(0.05, 150.0),
    frame="body",
    via=("device_coupling",),
))

RESPIRATION = O(Observation(
    id="respiration",
    doc="""respiratory belt or nasal flow.

    evidence about `mechanical.displacement` of the chest wall.  like the ECG it
    is carried mainly because it is a large structured contaminant of everything
    else -- respiration modulates cerebral blood flow through arterial CO2, moves
    the head, and drives a low-frequency component of fMRI signal that sits right
    in the band the haemodynamic response occupies -- and having it as state means
    those effects are mechanism rather than nuisance.

    the CO2 pathway is the interesting one and it is only partly declared: end-
    tidal CO2 changes cerebral blood flow substantially, which is a real
    physiological coupling that would enter through `vascular_flow`'s
    autoregulation, and no process currently reads a respiratory variable.  that
    is a gap this observation makes visible.""",
    observes=sel("mechanical.displacement", region=BODY, band=AUTONOMIC_BAND),
    modality="respiration",
    noise="gaussian",
    band=AUTONOMIC_BAND,
    frame="body",
    via=("mechanical_propagation",),
))

PPG = O(Observation(
    id="ppg",
    doc="""photoplethysmography: peripheral pulse volume.

    evidence about `blood.volume`, observed optically at a fingertip or ear.
    cheap, ubiquitous in wearable datasets, and constrains the vascular chain at
    a place far from the brain -- which is its value and its limitation.  it pins
    heart rate and its variability precisely and says very little about cerebral
    perfusion directly.

    the band extends to 10 Hz, well above the cardiac fundamental, because the
    *shape* of the pulse waveform -- the dicrotic notch, the rise time -- carries
    information about arterial compliance and peripheral resistance that the rate
    alone does not.  narrowing the band to the heart rate would discard it.""",
    observes=sel("blood.volume", "blood.oxygenation", region=VASCULAR,
                 band=AUTONOMIC_BAND),
    modality="ppg",
    noise="gaussian",
    band=AUTONOMIC_BAND,
    frame="body",
    via=("vascular_flow",),
))

POLYSOMNOGRAPHY = O(Observation(
    id="polysomnography",
    doc="""overnight sleep recording: EEG, EOG, EMG and respiration together.

    declared as one observation rather than as four because the thing that makes
    polysomnography scientifically distinctive is not any of its channels but
    their combination over a long recording -- sleep stage is defined by the joint
    pattern, and the datasets in docs/EVIDENCE.md that matter here are scored
    that way.

    the band is narrower than plain EEG at both ends and deliberately so.  the
    ceiling is 35 Hz because clinical polysomnography montages are low-passed
    there and because the features that define staging -- delta, spindles, K
    complexes, sawtooth waves -- all live below it.  the floor is higher than a
    research EEG's because overnight recordings drift and the aggressive
    high-pass that makes them usable removes the slowest components.

    the value of these recordings to a model like this one is specific and
    large: they are hours long, they traverse the full range of arousal states,
    and arousal is the single largest modulator of almost every process in the
    inventory -- thalamic relay mode, neuromodulator tone, neurovascular coupling
    gain.  a model fitted only to task data in awake subjects has seen one
    operating point.""",
    observes=sel("device.contact_potential", region=SENSORS, band=Band(0.3, 35.0)),
    modality="psg",
    noise="gaussian",
    band=Band(0.3, 35.0),
    frame="eeg_cap",
    via=ELECTRIC_CHAIN,
))


__all__ = [
    "EEG", "MEG", "IEEG", "ECOG", "INTRACORTICAL_SPIKES", "LOCAL_FIELD_POTENTIAL",
    "BOLD", "ASL_PERFUSION", "FNIRS", "PET",
    "DWI", "STRUCTURAL_MRI",
    "EMG", "MEP", "BEHAVIOUR", "MOTION_CAPTURE", "EYE_TRACKING", "PRODUCED_AUDIO",
    "ECG", "RESPIRATION", "PPG", "POLYSOMNOGRAPHY",
]

# ---------------------------------------------------------------------------
# modalities the model library asks for that the inventory above did not cover
# ---------------------------------------------------------------------------
# these are not an afterthought: each was discovered because a named
# materialization in ibm/materialize/library named a measurement no observation
# declared, which is the registry catching a gap between what we say we model
# and what we say we can see.

DRUG_CONCENTRATION = O(Observation(
    id="drug_concentration",
    doc="""assayed drug concentration -- plasma, and where available an
    effect-site estimate from a pharmacokinetic model.

    the honest object here is pressure on a process's theta, not a state
    measurement: an anaesthetic does not add current, it changes the parameters
    of synaptic and channel processes.  what is actually assayed, though, is a
    concentration, and concentration is ordinary extracellular state.  so this
    observes the concentration and leaves the concentration-to-theta map to
    `neuromodulation`, rather than pretending the assay measured a gain.

    the band is set by sampling: a plasma draw every few minutes constrains
    nothing above a millihertz, which is exactly why an induction transient is
    unrecoverable from plasma sampling alone and needs the infusion record as an
    intervention instead.""",
    observes=sel("extracellular.adenosine", band=Band(0.0, 0.005)),
    modality="assay", noise="gaussian", band=Band(0.0, 0.005),
    frame="body", via=("transmitter_dynamics",)))

CSF_FLOW_VELOCITY = O(Observation(
    id="csf_flow_velocity",
    doc="""phase-contrast MRI of csf velocity, usually at the aqueduct.

    velocity is signed and directly encoded in phase, so unlike most MR
    observations this one measures the state variable rather than a contrast
    derived from it.  the band reaches the cardiac fundamental because that is
    the point -- the pulsatile component is roughly two orders larger than net
    flow, and a measurement that averaged it away would constrain the thing we
    least care about.

    what it cannot see: respiratory-driven flow, which exceeds the cardiac
    contribution to *net* transport, is aliased by any gated acquisition.  that
    is a limitation of the measurement and separately a gap in `csf_flow`, which
    declares no respiratory driver at all.""",
    observes=sel("csf.velocity", band=Band(0.0, 2.0)),
    modality="mri", noise="gaussian", band=Band(0.0, 2.0),
    frame="subject_t1", via=("csf_flow",)))

TEMPERATURE = O(Observation(
    id="temperature",
    doc="""tissue temperature: MR thermometry non-invasively, a thermocouple or
    fibre-optic probe invasively.

    required by any stimulation-safety materialization, and the reason thermal
    state is in the ontology at all.  proton-resonance-frequency thermometry is
    a phase measurement and therefore drifts with everything else that shifts
    phase -- field drift, motion, susceptibility change -- so its precision is
    far worse than its nominal resolution suggests over the minutes that matter
    for a thermal dose.

    the band is very low by construction: brain thermal time constants are of
    order a hundred seconds, so nothing above about a hundredth of a hertz is
    either measurable or meaningful.""",
    observes=sel("thermal.temperature", band=Band(0.0, 0.02)),
    modality="mri", noise="gaussian", band=Band(0.0, 0.02),
    frame="subject_t1", via=("thermal_diffusion",)))

DC_POTENTIAL = O(Observation(
    id="dc_potential",
    doc="""DC-coupled potential: the slow shift that ordinary AC-coupled
    electrophysiology filters out before anyone sees it.

    declared separately from `ieeg` precisely because the difference is the
    amplifier's high-pass, not the electrode.  spreading depolarization is a
    ten-millivolt negative shift lasting a minute; through a 0.1 Hz high-pass it
    is invisible, which is a large part of why it went unrecognised in humans
    for so long.

    non-polarizable electrodes and a DC-capable amplifier are prerequisites, and
    drift is the dominant error rather than noise -- so this observation carries
    low precision at very low frequency and none at all at true DC.""",
    observes=sel("device.contact_potential", band=Band(0.001, 1.0)),
    modality="ieeg", noise="gaussian", band=Band(0.001, 1.0),
    frame="electrode_grid", via=("em_generation", "em_coupling", "device_coupling")))

TISSUE_DISPLACEMENT = O(Observation(
    id="tissue_displacement",
    doc="""micrometre-scale tissue displacement, from acoustic radiation force
    imaging or MR elastography.

    the only routine non-invasive measurement of the mechanical field, and the
    only thing that constrains where a focused-ultrasound beam actually deposited
    momentum as opposed to where the plan said it would.  displacement rather
    than pressure because displacement is what the imaging encodes; the pressure
    that caused it is inferred through the material state, and that inference is
    where the uncertainty lives.""",
    observes=sel("mechanical.displacement", band=Band(0.0, 500.0)),
    modality="mri", noise="gaussian", band=Band(0.0, 500.0),
    frame="subject_t1", via=("mechanical_propagation",)))
