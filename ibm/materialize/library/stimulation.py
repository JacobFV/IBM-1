"""stimulation as an intervention on device state, propagating through ordinary
processes.

    M = materialize(R, r, B, F, A, T, P)

nothing in this module is special-cased, and that is the architectural claim §6
makes: a coil current, a scalp electrode current, a transducer drive and a dbs
pulse are all `device.*` state that an experimenter is allowed to clamp, and
everything downstream is the same `em_coupling`, `mechanical_propagation`,
`local_excitation` and `thermal_diffusion` that the observation models read.  a
stimulation model differs from a forward model only in which variables are pinned.

r(q) in this group is driven by one thing: **the field gradient**.  a neuron is
depolarised by the derivative of the extracellular potential along its axon, not
by the potential, so the quantity that matters is a spatial derivative and spatial
derivatives are exactly what coarse-graining destroys.  wherever the field changes
fast -- at the grey/white boundary under a tms coil, at a dbs contact, across the
skull under a tes montage, through the skull for ultrasound -- the model refines.
where the field is smooth it does not, no matter how interesting the tissue is.

B splits sharply here between the device and the tissue.  a tms pulse is 100 us of
rise time and needs tens of kilohertz on `device.coil_current`; the population
response it evokes is an lfp-band object.  a focused ultrasound carrier is
hundreds of kilohertz on `mechanical.pressure` and its neural effect is
essentially dc.  keeping those in one materialization with different per-field
bands is the request interface earning its keep.
"""

from __future__ import annotations

from ibm.materialize.library import NamedModel, register, res, rule
from ibm.materialize.request import (
    Budget,
    DeviceSpec,
    MaterializationRequest,
    SubjectSpec,
    Window,)
from ibm.vocabulary import (
    Anat,
    Band,
    Near,
    OnSupport,
    sel,)

#: a biphasic tms pulse is ~250 us end to end with a ~60 us rise; the induced
#: e-field spectrum runs to tens of kHz and a narrower band on the coil would
#: smear the pulse the whole model is about.
PULSE = Band(0.0, 20_000.0)
#: tdcs is dc and tacs runs to a few hundred Hz; a kilohertz ceiling covers both
#: including the transcranial random-noise variants.
TES_DRIVE = Band(0.0, 1_000.0)
#: dbs: 130 Hz pulse trains of 60 us pulses.  the pulse width, not the rate, sets
#: the ceiling.
DBS_DRIVE = Band(0.0, 10_000.0)
#: tfus carrier, 200 kHz - 1.5 MHz.  this is materialized on `mechanical.pressure`
#: and nowhere else; the tissue response is near-dc.
ACOUSTIC = Band(1.0e5, 2.0e6)
#: the evoked population response, in every model here.
RESPONSE = Band(0.5, 300.0)
#: an mep is a compound muscle action potential recorded at 10-1000 Hz.
MEP = Band(10.0, 1_000.0)
#: the response band inside `tfus_response`, whose window is a sub-millisecond
#: acoustic field-solve window.  it starts at dc rather than 0.5 Hz because a
#: 0.4 ms window resolves no finer than a couple of kilohertz: the slow response is
#: near-constant over this trajectory and its dynamics are carried across windows.
SLOW_RESPONSE = Band(0.0, 300.0)


TMS_RESPONSE = register(NamedModel(
    id="tms_response",
    doc="""cortical response to a transcranial magnetic pulse.

    the coil current is clamped (`interventions`), the induced e-field is an
    ordinary `em_coupling` output, and the population response is ordinary
    `local_excitation` and `local_inhibition`.  the only unusual thing about the
    request is r(q).

    **why 0.5 mm within 20 mm of the coil, and specifically at the grey/white
    boundary.**  what activates an axon is the derivative of the induced e-field
    along it -- the activating function -- and that derivative is largest where the
    field's tangential component changes fast, which is where fibres bend into the
    white matter at the gyral crown and at the sulcal lip.  the induced field
    itself is smooth over centimetres; its derivative along a bending fibre is not.
    materializing the field at 2 mm and differentiating gives an activation map
    that is wrong in location by a gyrus, which is the difference between a hand
    response and no response at all.  0.5 mm is where the fibre geometry is locally
    straight.

    **why it coarsens quickly.**  the induced field falls off roughly as the
    inverse cube beyond the coil's footprint, so six centimetres away it is a
    smooth multipole and 4 mm is exact.  the fine shell is a hemisphere a couple of
    centimetres deep, which is what makes 0.5 mm affordable.

    **why the head volume matters as much as the cortex.**  unlike tes, tms is
    largely insensitive to skull conductivity -- but it is very sensitive to the
    csf/grey boundary, where the conductivity jumps by a factor of four and the
    normal component of current density is discontinuous.  the 1 mm head-volume
    rule is spent there.

    **B.**  20 kHz on the coil, 0.5-300 Hz on the neural response.  a single band
    would either alias the pulse or waste four decades of coefficients on cortex.""",
    request=MaterializationRequest(
        name="tms-response",
        targets=(sel("electromagnetic.efield", region=Near("coil", 60.0), band=PULSE),
                 sel("neural.exc.activity", "neural.inh.activity",
                     region=Near("coil", 60.0), band=RESPONSE)),
        regions=(("focus", Near("coil", 20.0)),
                 ("field", Near("coil", 60.0)),
                 ("conductor", OnSupport("head_volume"))),
        resolution=res(
            rule(Near("coil", 20.0), 0.5, PULSE),
            rule(OnSupport("head_volume"), 1.0, PULSE),
            rule(Near("coil", 60.0), 1.5, RESPONSE),
            rule(OnSupport("cortical_surface"), 3.0, RESPONSE),
            default_mm=6.0, default_band=RESPONSE),
        fields=("device", "electromagnetic", "material", "neural", "structural"),
        anatomy=("cortical_areas", "cortical_layers"),
        topologies=("electromagnetic", "device_coupling", "local", "microcircuit",
                    "cortical_surface", "tractometric"),
        processes=("device_coupling", "em_coupling", "local_excitation",
                   "local_inhibition", "lateral_cortical_propagation",
                   "tract_propagation"),
        observations=("mep", "eeg", "bold"),
        interventions=("tms",),
        devices=(DeviceSpec("coil", "stimulator", "coil", "stimulate",
                            n_elements=1,
                            note="figure-of-eight coil; pose is tracked per pulse "
                                 "and is a fitted nuisance, not an assumption"),),
        subject=SubjectSpec(note="individual head geometry is mandatory here -- a "
                            "template skull moves the predicted hot spot by "
                            "more than the effect being studied"),
        window=Window(n=131072, dt=2e-5),  # 2.6 s at 50 kHz: the pulse needs
                                           # the rate, the response needs the length
        bands=(("device", PULSE), ("electromagnetic", PULSE), ("neural", RESPONSE),
               ("effector", MEP)),
        budget=Budget(max_state_variables=1_500_000,
                      max_spectral_coefficients=300_000_000),),
    fit_sources=("ds003037", "gp-tms-hsh", "tms-localization-example-osf-myrqn",
                 "tms-pulsewise-coil-displacement", "motor-threshold-calibration-studies",
                 "manufacturer-coil-geometry", "simnibs"),
    eval_sources=("multi-site-tep-data", "ds005498", "tms-fmri-studies",
                  "tms-e-field-direction-mapping-osf-9f3bc", "ds002094"),
    teachers=("neural-mass-model-families",),
    negative_controls=("ds008037", "tms-artifact-phantoms",
                       "bwin-000-synthetic-scaffold"),
    constrained=(
        "the induced e-field itself, which is a solved electromagnetics problem "
        "given the geometry and which `simnibs` and the osf coil-pose datasets pin "
        "to within the registration error",
        "the coil-position-to-mep mapping, from the per-pulse displacement data -- "
        "this is a genuinely interventional measurement and rare in this corpus",
        "resting motor threshold as a per-subject scalar",),
    prior_dominated=(
        "the cellular target.  which axonal population the pulse actually recruits "
        "-- and therefore the layer and cell type -- is a prior taken from animal "
        "work; the human data constrain only the input-output curve",
        "everything more than about 6 cm from the coil.  the model materializes "
        "propagation over the tractometric topology and nothing here measures it",
        "state dependence: the response depends strongly on ongoing oscillatory "
        "phase and none of these fit sources was phase-triggered",),))


TES_RESPONSE = register(NamedModel(
    id="tes_response",
    doc="""cortical response to transcranial electrical stimulation (tdcs, tacs, trns).

    the model whose honest conclusion is uncomfortable, and it is written to make
    that visible rather than to hide it.

    **the field is well constrained; the response is not.**  a 2 mA montage
    produces at most 0.2-0.5 V/m in cortex, which is one to two orders of magnitude
    below what is needed to fire a neuron.  the intervention is subthreshold and
    acts by biasing membrane potential, so `constrained` below contains the field
    and essentially nothing else, and `prior_dominated` contains the entire
    behavioural claim.  a materialization that reported a confident behavioural
    prediction here would be reporting its prior with the field's error bars.

    **why r(q) is fine in the head volume and coarse in the brain.**  unlike tms,
    tes is dominated by conductivity: the skull shunts, and the csf layer -- four
    times more conductive than brain and 1-3 mm thick -- carries a large fraction
    of the injected current tangentially before it ever reaches cortex.  a 3 mm
    conductor grid does not resolve a 1.5 mm csf layer and gets the cortical field
    wrong by tens of percent.  so 1 mm on `head_volume`, and only 3 mm on the
    cortical sheet, because the field there is smooth at that scale and its
    gradient along fibres is small compared with the tms case.

    **B.**  dc to 1 kHz on the device (tdcs is dc, tacs runs to a few hundred Hz,
    trns is broadband); 0.5-300 Hz on the neural response.  the entrainment claim
    tacs rests on lives in the *phase* relationship between those two, which the
    spectral form represents as anisotropy on the (cos, sin) plane at the driving
    frequency -- the natural way to say 'phase-locked' without a special case.""",
    request=MaterializationRequest(
        name="tes-response",
        targets=(sel("electromagnetic.efield", "electromagnetic.current_density",
                     band=TES_DRIVE),
                 sel("neural.exc.potential", "neural.exc.activity", band=RESPONSE)),
        regions=(("conductor", OnSupport("head_volume")),
                 ("cortex", OnSupport("cortical_surface"))),
        resolution=res(
            rule(Near("pads", 30.0), 1.0, TES_DRIVE),
            rule(OnSupport("head_volume"), 1.0, TES_DRIVE),
            rule(OnSupport("cortical_surface"), 3.0, RESPONSE),
            default_mm=5.0, default_band=RESPONSE),
        fields=("device", "electromagnetic", "material", "neural"),
        anatomy=("cortical_areas",),
        topologies=("electromagnetic", "device_coupling", "local",
                    "cortical_surface"),
        processes=("device_coupling", "em_coupling", "local_excitation",
                   "local_inhibition"),
        observations=("eeg", "bold"),
        interventions=("tdcs",),
        devices=(DeviceSpec("pads", "stimulator", "eeg_cap", "stimulate",
                            n_elements=8,
                            note="scalp electrodes; montage geometry and contact "
                                 "impedance are both fitted"),),
        subject=SubjectSpec(note="skull and csf thickness are the dominant "
                            "uncertainty; template geometry is not acceptable"),
        window=Window(n=8192, dt=5e-4),   # 4 s at 2 kHz: the trns ceiling sets dt,
                                          # the 0.5 Hz response floor sets n
        bands=(("device", TES_DRIVE), ("electromagnetic", TES_DRIVE),
               ("neural", RESPONSE)),
        budget=Budget(max_state_variables=1_500_000),),
    fit_sources=("ds003670", "tes-montage-field-validation-data",
                 "tes-solver-reference-fixtures", "roast", "simnibs",
                 "itis-database"),
    eval_sources=("ds005779", "ds002094", "tfus-esi-resting-eeg",
                  "fem-bem-analytic-reference-problems"),
    teachers=("neural-mass-model-families",),
    negative_controls=("ds008037", "bwin-000-synthetic-scaffold",
                       "example-published-study"),
    constrained=(
        "the intracranial e-field distribution, which `tes-montage-field-validation-data` "
        "measures directly with implanted electrodes -- the only ground truth for a "
        "forward field anywhere in this library",
        "tissue conductivities, from `itis-database` as a literature prior tightened "
        "by the validation recordings",
        "current shunting through scalp and csf as a fraction of injected current",),
    prior_dominated=(
        "the entire neural response.  0.3 V/m is subthreshold and no source here "
        "measures the membrane-potential bias it produces in humans; the coupling "
        "gain sits at a weak prior and any behavioural prediction is that prior",
        "the direction of the effect.  anodal-excitatory/cathodal-inhibitory is a "
        "convenient story, not something these data separate from the null",
        "individual differences in response, which the literature reports as large "
        "and which nothing in this corpus explains",),))


TFUS_RESPONSE = register(NamedModel(
    id="tfus_response",
    doc="""neural response to transcranial focused ultrasound.

    the most expensive r(q) in the library, and the reason is the skull rather than
    the target.

    **why 0.25 mm through the skull and 0.5 mm along the whole propagation path.**
    the acoustic wavelength at 500 kHz is about 3 mm in water and 6 mm in bone, and
    a finite-difference or pseudospectral solution needs several points per
    wavelength -- six is the usual floor -- so 0.5 mm is the *coarsest* grid on
    which the field is a field rather than a numerical artifact.  the skull is
    worse: it is a heterogeneous, dispersive, strongly refracting layer whose
    thickness and porosity vary over millimetres, and the phase error it introduces
    is what displaces and defocuses the beam.  getting the focus in the right place
    is a skull problem, and refinement therefore goes to the skull, not the focus.
    this is the clearest case in the library of r(q) following the physics rather
    than following the target.

    **and the whole path, not a ball around the focus.**  standing waves between
    skull surfaces and reflections off the far side both change the pressure at the
    focus by tens of percent, so truncating the domain is not conservative.

    **B is where this model is strange.**  `mechanical.pressure` carries 0.1-2 MHz,
    which forces a sub-microsecond timestep; the neural response carries 0.5-300 Hz
    and the thermal component is essentially dc.  three bands spanning ten decades
    in one materialization, each on its own field.  this is bandwidth-as-a-laziness-axis
    at its most literal: the acoustic band is unavoidable and enormous, so the
    *window* is short and the model is run as a steady-state field solve plus a
    slow response, rather than as one long trajectory.""",
    request=MaterializationRequest(
        name="tfus-response",
        targets=(sel("mechanical.pressure", "mechanical.displacement",
                     region=Near("transducer", 120.0), band=ACOUSTIC),
                 sel("neural.exc.activity", "neural.inh.activity",
                     region=Near("transducer", 120.0), band=SLOW_RESPONSE),
                 sel("thermal.temperature", band=Band(0.0, 1.0))),
        regions=(("path", Near("transducer", 120.0)),
                 ("skull", OnSupport("head_volume")),
                 ("focus", Near("transducer", 90.0))),
        resolution=res(
            rule(OnSupport("head_volume"), 0.25, ACOUSTIC),
            rule(Near("transducer", 120.0), 0.5, ACOUSTIC),
            rule(OnSupport("cortical_surface"), 2.0, SLOW_RESPONSE),
            # the default band is the *acoustic* one: this request's window is a
            # field-solve window, and the slow response is carried by the band
            # overrides below rather than by this trajectory
            default_mm=4.0, default_band=ACOUSTIC),
        fields=("device", "mechanical", "material", "thermal", "neural"),
        anatomy=("cortical_areas", "thalamic_nuclei", "brainstem_nuclei"),
        topologies=("mechanical", "device_coupling", "local", "microcircuit"),
        processes=("device_coupling", "mechanical_propagation", "thermal_diffusion",
                   "local_excitation", "local_inhibition"),
        observations=("eeg", "bold", "tissue_displacement"),
        interventions=("tfus",),
        devices=(DeviceSpec("transducer", "stimulator", "transducer", "stimulate",
                            n_elements=1,
                            note="single-element or phased array; ct-derived skull "
                                 "map is a hard requirement"),),
        subject=SubjectSpec(template=None,
                            note="skull density and thickness from ct; a template "
                                 "skull produces a focus in the wrong place"),
        window=Window(n=4096, dt=1e-7),
        bands=(("mechanical", ACOUSTIC), ("neural", SLOW_RESPONSE),
               ("thermal", Band(0.0, 1.0)), ("device", ACOUSTIC)),
        budget=Budget(max_state_variables=2_000_000,
                      max_spectral_coefficients=200_000_000,
                      max_octree_level=14),),
    fit_sources=("skull-acoustics-ct-mri-datasets", "transducer-calibration-datasets",
                 "hydrophone-phantoms", "itis-database", "k-wave", "babelbrain",
                 "babelbrain-examples"),
    eval_sources=("kwave-reference-problems", "acoustic-closed-forms",
                  "mr-arfi-displacement-data", "tfus-esi-resting-eeg",
                  "tfus-mvep-speller"),
    teachers=(),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "the acoustic field in water and in skull phantoms, which the hydrophone "
        "and calibration datasets measure directly and which the analytic and "
        "k-wave reference problems verify",
        "skull acoustic properties as a function of ct hounsfield units",
        "focal displacement and defocusing through real skulls, from the mr-arfi "
        "measurements",),
    prior_dominated=(
        "the neuromodulatory mechanism entirely.  whether the effect is "
        "mechanosensitive-channel gating, cavitation, thermal, or an auditory "
        "confound is not settled, and this model's `local_excitation` coupling to "
        "pressure sits at a speculative prior.  any predicted behavioural effect "
        "is that prior",
        "the auditory confound itself: ultrasound pulses are audible through bone "
        "conduction and the eeg sources here do not separate that from a direct "
        "neural effect",
        "in-vivo thermal rise, which is bounded by `thermal_safety` rather than "
        "measured here",),))


DBS_RESPONSE = register(NamedModel(
    id="dbs_response",
    doc="""network response to deep brain stimulation.

    the invasive counterpart of `tms_response`, and the one where the activating
    function argument is sharpest because the electrode is *inside* the tissue.

    **why 100 um within 3 mm of a contact.**  the extracellular potential around a
    dbs contact falls as 1/r, so its second derivative along a passing axon -- the
    activating function -- falls as 1/r^3 and is enormous within a millimetre.  the
    volume of tissue activated is bounded by a surface where that quantity crosses
    a threshold, and the surface's position moves by a millimetre if the potential
    is materialized at 1 mm instead of 0.1 mm.  a millimetre is the difference
    between stimulating the subthalamic nucleus and stimulating the internal
    capsule, which is the difference between benefit and dysarthria.  no other
    model in this library has a resolution requirement with that direct a clinical
    consequence.

    **the encapsulation layer is why the fine shell is 3 mm and not 1 mm.**  a
    fibrotic sheath 100-500 um thick forms around a chronic lead and changes the
    effective impedance by a factor of two; it must be a materialized material
    layer rather than a lumped constant, and it needs several cells across its
    thickness.

    **why 2 mm through connected structures.**  the therapeutic effect is not local:
    it propagates through the basal ganglia-thalamocortical loop, and those edges do
    not survive coarsening to 8 mm.  the loop gets its own rule even though nothing
    there is directly stimulated.

    **B.**  10 kHz on the device (60 us pulses at 130 Hz), 0.5-300 Hz on tissue.""",
    request=MaterializationRequest(
        name="dbs-response",
        targets=(sel("electromagnetic.potential", "electromagnetic.efield",
                     region=Near("lead", 15.0), band=DBS_DRIVE),
                 sel("neural.exc.activity", "neural.inh.activity",
                     region=Near("lead", 15.0), band=RESPONSE)),
        regions=(("contact_shell", Near("lead", 3.0)),
                 ("nucleus", Anat("bg_territories", "subthalamic")),
                 ("loop", Anat("bg_territories", "sensorimotor"))),
        resolution=res(
            rule(Near("lead", 3.0), 0.1, DBS_DRIVE),
            rule(Near("lead", 15.0), 0.5, DBS_DRIVE),
            rule(Anat("bg_territories", "sensorimotor"), 2.0, RESPONSE),
            rule(Anat("thalamic_nuclei", "ventral_lateral"), 2.0, RESPONSE),
            default_mm=8.0, default_band=RESPONSE),
        fields=("device", "electromagnetic", "material", "neural", "structural"),
        anatomy=("bg_territories", "thalamic_nuclei", "striosome_matrix",
                 "cortical_areas"),
        topologies=("electromagnetic", "device_coupling", "local", "microcircuit",
                    "tractometric"),
        processes=("device_coupling", "em_coupling", "local_excitation",
                   "local_inhibition", "tract_propagation",
                   "thalamocortical_coupling", "neuromodulation"),
        observations=("lfp", "mep", "behaviour"),
        interventions=("dbs",),
        devices=(DeviceSpec("lead", "stimulator", "electrode_grid", "both",
                            n_elements=8,
                            note="segmented or ring quadripolar lead; also records"),),
        subject=SubjectSpec(id="patient", template=None,
                            note="lead position from post-op ct; the atlas is for "
                                 "interpretation, not for localisation"),
        window=Window(n=65536, dt=5e-5),  # 3.3 s at 20 kHz: 60 us pulses and a
                                          # 0.5 Hz floor in one window
        bands=(("device", DBS_DRIVE), ("electromagnetic", DBS_DRIVE),
               ("neural", RESPONSE), ("extracellular", Band(0.0, 10.0))),
        budget=Budget(max_state_variables=1_200_000,
                      max_spectral_coefficients=300_000_000),),
    fit_sources=("dbs-ieeg-stimulation-datasets", "distal-lead-dbs",
                 "electrode-impedance-contact-models", "cit168-pauli",
                 "focus-subcortical", "thomas-thalamic"),
    eval_sources=("ram-intracranial", "ds003670", "diffusion-thalamic-connectivity-atlases",
                  "clinical-ieeg-archives"),
    teachers=("neural-mass-model-families",),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "the extracellular potential field around the lead, which is a solved "
        "problem given the geometry and the encapsulation impedance",
        "contact impedance and its post-implant evolution, measured directly by "
        "the device",
        "evoked potentials at neighbouring contacts, which the ieeg stimulation "
        "datasets record and which pin local conduction delays",),
    prior_dominated=(
        "which fibres are recruited.  the model materializes 100 um sites and no "
        "human dataset labels the axons that occupy them; fibre orientation comes "
        "from a diffusion atlas at 1-2 mm and is the binding constraint",
        "the mechanism of therapeutic benefit -- excitation, inhibition, "
        "informational lesion, or antidromic activation.  all reproduce the "
        "clinical data and this corpus does not separate them",
        "the long-latency network response beyond the first synapse, which the "
        "2 mm loop rule materializes and nothing here measures",),))


TMS_EEG_TEP = register(NamedModel(
    id="tms_eeg_tep",
    doc="""the tms-evoked potential: cortical excitability read out at the scalp.

    kept separate from `tms_response` because its honest failure mode is different
    and specific.  a tep is a scalp waveform following a magnetic pulse, and three
    things other than cortical excitability produce scalp waveforms following a
    magnetic pulse: the pulse artifact itself, the click the coil makes, and the
    scalp twitch it causes.  the auditory and somatosensory evoked responses to
    those confounds are larger than the tep in many montages and have similar
    latencies.  this model exists so that the confound is materialized -- as
    ordinary `transduction` and `afferent_propagation` state, driven by the same
    intervention -- rather than subtracted by assumption.

    **B starts at 1 Hz, not 0.5.**  the amplifier recharge transient after a tms
    pulse is a large, slow drift; a materialization carrying 0.5 Hz would be
    fitting the amplifier.

    **and the first 15 ms is excluded from the likelihood**, not from the
    materialization.  the state exists over that interval; the observation simply
    contributes no precision there, which is exactly the distinction §6 draws
    between state and evidence about it.

    **r(q).**  1 mm under the coil, 3 mm on the rest of the sheet, 1.5 mm in the
    conductor.  finer than `eeg_forward` under the coil because the *source* of a
    tep is a small, sharply localised patch whose extent is what the model is
    trying to estimate, and coarser elsewhere because the scalp cannot see it.""",
    request=MaterializationRequest(
        name="tms-eeg-tep",
        targets=(sel("device.contact_potential", band=Band(1.0, 100.0)),
                 sel("neural.exc.activity", "neural.inh.activity",
                     region=Near("coil", 40.0), band=RESPONSE)),
        regions=(("under_coil", Near("coil", 40.0)),
                 ("cortex", OnSupport("cortical_surface")),
                 ("conductor", OnSupport("head_volume"))),
        resolution=res(
            rule(Near("coil", 20.0), 1.0, RESPONSE),
            rule(OnSupport("head_volume"), 1.5, Band(1.0, 100.0)),
            rule(OnSupport("cortical_surface"), 3.0, Band(1.0, 100.0)),
            default_mm=6.0, default_band=Band(1.0, 100.0)),
        fields=("device", "electromagnetic", "neural", "material", "transduction"),
        anatomy=("cortical_areas",),
        topologies=("electromagnetic", "device_coupling", "cortical_surface",
                    "tractometric", "afferent_pathway"),
        processes=("device_coupling", "em_coupling", "em_generation",
                   "local_excitation", "local_inhibition",
                   "lateral_cortical_propagation", "tract_propagation",
                   "transduction", "afferent_propagation"),
        observations=("eeg",),
        interventions=("tms", "auditory_stimulus",),
        devices=(DeviceSpec("coil", "stimulator", "coil", "stimulate"),
                 DeviceSpec("eeg", "sensor_array", "eeg_cap", "observe",
                            n_elements=64,
                            note="tms-compatible amplifier; recharge transient is "
                                 "modelled, not filtered")),
        subject=SubjectSpec(),
        window=Window(n=4096, dt=5e-4),   # 2 s: the 0.5 Hz response floor
        bands=(("device", Band(1.0, 100.0)), ("neural", RESPONSE),
               ("electromagnetic", Band(1.0, 100.0)),
               ("transduction", Band(1.0, 100.0))),
        budget=Budget(max_state_variables=800_000),),
    fit_sources=("ds003037", "multi-site-tep-data", "ds005779", "ds002094",
                 "ds005620", "tesa-reference-data"),
    eval_sources=("fieldtrip-tms-eeg-tutorial-data", "tesa-reference-data",
                  "tms-artifact-phantoms"),
    teachers=("neural-mass-model-families", "eegpt"),
    negative_controls=("ds008037", "tms-artifact-phantoms",
                       "bwin-000-synthetic-scaffold"),
    constrained=(
        "the 15-300 ms tep waveform and its dependence on stimulation intensity, "
        "from the multi-site data which was collected precisely to establish "
        "reproducibility",
        "the artifact and recharge model, from the phantom recordings where there "
        "is no brain at all",
        "the auditory and somatosensory confound magnitude, from the sham condition "
        "in `ds008037`",),
    prior_dominated=(
        "everything in the first 15 ms.  the state is materialized and the "
        "observation contributes no precision there; a reported early component is "
        "the prior",
        "the source of late components.  a tep at 200 ms is spatially broad and the "
        "inverse problem is the same rank-deficient one as `eeg_source`",
        "excitation/inhibition balance, which teps are widely claimed to index and "
        "which nothing in this corpus identifies against the alternatives",),))


MOTOR_MAPPING = register(NamedModel(
    id="motor_mapping",
    doc="""coil position and orientation to motor evoked potential amplitude.

    the smallest and best-posed model in the group: an input-output curve measured
    hundreds of times per subject with a genuinely interventional manipulation.  it
    is separated from `tms_response` because it is where the *nonlinearity* lives.

    **why the nonlinearity forces fine r(q).**  the mep recruitment curve is a
    steep sigmoid in stimulation intensity; near threshold a ten percent change in
    local e-field changes mep amplitude several-fold.  averaging the field over a
    coarse cell and then applying the sigmoid is not the same as applying the
    sigmoid pointwise and averaging -- this is §1's 'a nonlinearity whose average
    is not the average's image', with a measurable consequence.  so 0.5 mm through
    the hand knob, where the field gradient and the fibre bending are both largest.
    2 mm through the rest of precentral cortex, 6 mm elsewhere.

    **B.**  the mep is a compound muscle action potential: 10-1000 Hz on
    `effector.force` and the motor-unit components.  the cortical side is the
    ordinary response band.  the coil keeps the pulse band.

    **why it is worth a named model at all.**  because it is one of very few places
    in this corpus where an intervention is applied, varied systematically, and
    measured with high snr -- so it is the natural place to check whether the
    library's e-field forward model is right, and a failure here invalidates
    `tms_response` and `tms_eeg_tep` both.""",
    request=MaterializationRequest(
        name="motor-mapping",
        targets=(sel("effector.drive", "effector.activation", "effector.force",
                     band=MEP),
                 sel("neural.exc.activity", region=Anat("cortical_areas", "precentral"),
                     band=RESPONSE)),
        regions=(("hand_knob", Anat("cortical_areas", "precentral")),
                 ("focus", Near("coil", 30.0))),
        resolution=res(
            rule(Near("coil", 15.0), 0.5, PULSE),
            rule(Anat("cortical_areas", "precentral"), 2.0, RESPONSE),
            rule(OnSupport("head_volume"), 1.0, PULSE),
            default_mm=6.0, default_band=RESPONSE),
        fields=("device", "electromagnetic", "material", "neural", "effector",
                "structural"),
        anatomy=("cortical_areas",),
        topologies=("electromagnetic", "device_coupling", "efferent_pathway",
                    "tractometric", "local"),
        processes=("device_coupling", "em_coupling", "local_excitation",
                   "efferent_propagation", "effector_activation",
                   "tract_propagation"),
        observations=("mep",),
        interventions=("tms",),
        devices=(DeviceSpec("coil", "stimulator", "coil", "stimulate",
                            note="neuronavigated; per-pulse pose is recorded"),
                 DeviceSpec("emg", "sensor_array", "body", "observe", n_elements=4)),
        subject=SubjectSpec(),
        window=Window(n=131072, dt=2e-5),  # as tms_response: the pulse sets dt,
                                           # the mep and the 0.5 Hz floor set n
        bands=(("device", PULSE), ("electromagnetic", PULSE),
               ("neural", RESPONSE), ("effector", MEP)),
        budget=Budget(max_state_variables=600_000),),
    fit_sources=("gp-tms-hsh", "tms-localization-example-osf-myrqn",
                 "tms-pulsewise-coil-displacement", "motor-threshold-calibration-studies",
                 "ds003037", "manufacturer-coil-geometry"),
    eval_sources=("tms-e-field-direction-mapping-osf-9f3bc", "gp-tms-hsh",
                  "simnibs", "ninapro"),
    teachers=(),
    negative_controls=("ds008037", "tms-artifact-phantoms",
                       "bwin-000-synthetic-scaffold"),
    constrained=(
        "the position and orientation of the cortical hot spot, to a few "
        "millimetres, from per-pulse coil pose regressed against mep amplitude",
        "the recruitment curve's threshold and slope per subject and per muscle",
        "the direction sensitivity of activation, which the e-field-direction "
        "mapping dataset was built to measure",),
    prior_dominated=(
        "the corticospinal conduction chain.  everything between cortical "
        "activation and the muscle is one fitted latency plus a prior; the "
        "spinal segment is not materialized in any detail",
        "trial-to-trial mep variability, which is enormous and which the model "
        "absorbs into noise rather than explaining through ongoing state",
        "the mapping outside the hand representation; almost all of this data is "
        "first dorsal interosseous and generalisation to other muscles is prior",),))
