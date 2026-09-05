"""materializations of slow physiology: clearance, heat, structural change and
depolarising waves.

    M = materialize(R, r, B, F, A, T, P)

what these four share is a band that collapses almost to dc and a cost that is
therefore entirely in the site count.  §1 makes bandwidth co-equal with spatial
resolution as a laziness axis, and this module is the corner where one axis has
gone to nearly zero: `plasticity_learning` carries structural components in the
`STRUCTURAL` band, which is 0-0.001 Hz, so a state variable there is two or three
coefficients over a window measured in weeks.  the whole model costs what its
geometry costs.

that is not the same as being cheap.  `thermal_safety` and
`spreading_depolarization` both need sub-millimetre grids for reasons that have
nothing to do with their bandwidth -- a hot spot at a skull-brain interface, and a
travelling front with a bistable nonlinearity -- so they are expensive on the
spatial axis alone.  this is the clean demonstration that the two budgets do not
substitute for each other.

it is also the module with the least evidence behind it.  glymphatic clearance and
spreading depolarisation are real phenomena with essentially no non-invasive human
measurement in this corpus; both are declared anyway, at weak and speculative
priors, because §5 says a process may exist in the ontology without a
high-confidence f, and a materialization that only ever names well-measured things
would quietly encode the claim that unmeasured things do not happen.
"""

from __future__ import annotations

from ibm.materialize.library import NamedModel, anat, register, res, rule
from ibm.materialize.request import (
    Budget,
    DeviceSpec,
    MaterializationRequest,
    SubjectSpec,
    Window,)
from ibm.vocabulary import (
    Band,
    DC,
    HEMODYNAMIC,
    Near,
    OnSupport,
    ULTRASLOW,
    sel,)

#: clearance runs over hours but is *gated* by cardiac pulsation near 1 Hz,
#: respiration near 0.3 Hz and the sleep slow oscillation.  a single interval has
#: to reach 2 Hz to carry the driver even though the transported quantity is dc.
CLEARANCE = Band(0.0, 2.0)
#: perfused tissue has a thermal time constant of order a minute; nothing above
#: 0.1 Hz survives the bio-heat diffusion.
THERMAL = Band(0.0, 0.1)
#: structural change: days to weeks.  the *field* is declared over `STRUCTURAL`
#: (0-0.001 Hz), but a longitudinal study samples daily at best and the nyquist of
#: daily sampling is 5.8 uHz.  materializing the declared band would be claiming a
#: resolution no human longitudinal design in this corpus can supply, so the
#: materialized band is the sampling's, not the declaration's.
SLOW_STRUCTURAL = Band(0.0, 5.0e-6)
#: a spreading depolarisation is a several-minute dc event; the suppression of
#: normal activity that accompanies it is an lfp-band effect, carried separately.
SD_WAVE = Band(0.0, 0.5)


GLYMPHATIC = register(NamedModel(
    id="glymphatic",
    doc="""perivascular and interstitial clearance of solute.

    the model in this library with the widest gap between what is materialized and
    what is measured, which is why the accounting matters more here than the
    numbers.

    **r(q) is set by the perivascular space, which is 20-100 um across.**  transport
    along it is advective and along the interstitium is diffusive with an effective
    diffusivity divided by tortuosity squared, and the ratio of those two is what
    the whole mechanism turns on.  a materialization coarser than the perivascular
    space cannot represent the ratio at all -- the two compartments merge into one
    and the model becomes pure diffusion.  so 20 um around penetrating arterioles.
    that is finer than anything else in the library, and it is affordable only
    because the band is nearly dc and because the region is restricted to a small
    tissue block rather than a brain.

    **B reaches 2 Hz for a reason that looks wrong at first.**  the *transported*
    quantity changes over hours.  but the transport is driven by cardiac pulsation
    of the arteriole wall at about 1 Hz and modulated by respiration and by the
    sleep slow oscillation, and a band that stopped at 0.01 Hz would exclude the
    driver while keeping the driven.  the coupling is multiplicative -- a
    pulsation-dependent advective velocity -- so it does not close in the spectral
    domain and must be evaluated in time; §1's stated cost of the spectral form,
    paid explicitly.

    **honest status: this is a prior, not a fit.**  no human dataset in this corpus
    measures perivascular flow.  the sleep and vascular sources constrain the
    *gating* variables and the geometry; the transport parameters sit at weak and
    speculative priors and every clearance number the model produces is a
    consequence of them.""",
    request=MaterializationRequest(
        name="glymphatic",
        targets=(sel("csf.solute", "csf.velocity", "csf.pressure", band=CLEARANCE),
                 sel("extracellular.volume_fraction", "extracellular.osmolarity",
                     band=CLEARANCE)),
        regions=(("perivascular", OnSupport("vascular_tree")),
                 ("interstitium", OnSupport("interstitial")),
                 ("csf_space", OnSupport("csf_space"))),
        resolution=res(
            rule(OnSupport("vascular_tree"), 0.02, CLEARANCE),
            rule(OnSupport("interstitial"), 0.1, CLEARANCE),
            rule(OnSupport("csf_space"), 1.0, CLEARANCE),
            default_mm=4.0, default_band=ULTRASLOW),
        fields=("csf", "extracellular", "blood", "material", "neural"),
        anatomy=("vascular_territories", "cortical_areas"),
        topologies=("csf", "interstitial", "vascular", "metabolic_exchange"),
        processes=("csf_flow", "csf_interstitial_exchange", "interstitial_transport",
                   "vascular_flow", "ionic_diffusion"),
        observations=("bold", "csf_flow_velocity", "polysomnography"),
        devices=(DeviceSpec("scanner", "scanner_element", "scanner", "observe",
                            note="fast epi with a fourth-ventricle slice, plus "
                                 "phase-contrast velocity"),),
        subject=SubjectSpec(),
        window=Window(n=1024, dt=0.25),
        bands=(("csf", CLEARANCE), ("extracellular", CLEARANCE),
               ("blood", CLEARANCE), ("neural", CLEARANCE)),
        budget=Budget(max_state_variables=1_500_000,
                      max_spectral_coefficients=20_000_000),),
    fit_sources=("ds003768", "sleep-edfx", "7t-qsm-venograms",
                 "high-resolution-vascular-atlases", "microscopy-microvascular-networks",
                 "capillary-density-statistics"),
    eval_sources=("mpi-leipzig-mind-brain-body", "corr-reliability", "adni",
                  "oasis-3"),
    teachers=(),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study",
                       "adni"),
    constrained=(
        "the vascular geometry the transport runs along, from the venogram and "
        "microscopy sources -- geometry is the constrained part of this model",
        "the sleep-stage gating variable, from the eeg-fmri and polysomnography "
        "sources, which is a real and reproducible modulator",
        "csf bulk motion in the ventricles and aqueduct, which phase-contrast mri "
        "measures directly",),
    prior_dominated=(
        "perivascular advective velocity, the central quantity.  it is a "
        "speculative prior taken from rodent two-photon work; no human measurement "
        "here touches it and every clearance rate the model reports is that prior "
        "propagated forward",
        "interstitial tortuosity and volume fraction in vivo, taken from ex-vivo "
        "and animal literature",
        "the causal direction between sleep and clearance, which is the whole "
        "interest of the model and which nothing here establishes",),))


THERMAL_SAFETY = register(NamedModel(
    id="thermal_safety",
    doc="""tissue temperature under a stimulator: the bound that gates
    `tfus_response` and, at high duty cycles, `tms_response`.

    the cost here is entirely in the site count, and the sites are not where the
    intuition puts them.

    **the hot spot is in the skull, not at the focus.**  cortical bone absorbs
    ultrasound roughly forty to fifty times more strongly than brain, so for a
    transcranial exposure the peak temperature rise is at the bone-brain interface
    and the peak thermal gradient is across a layer a few hundred microns thick.
    a 2 mm grid averages that layer away and returns a temperature rise several
    times too low, which for a safety model is the worst possible direction to be
    wrong in.  0.25 mm through the skull and its interfaces; 0.5 mm through the
    focal region; 4 mm elsewhere, where the bio-heat equation is smooth.

    **B is 0-0.1 Hz and that is generous.**  perfused tissue has a thermal time
    constant of order a minute and the bio-heat equation is diffusive, so the
    temperature field has no structure above a few tens of millihertz.  the window
    is minutes long because thermal dose accumulates and the quantity of interest
    is an integral, not an instantaneous value.

    **the pairing with `tfus_response` is deliberate.**  the acoustic field solve
    is materialized there at MHz bandwidth and its absorbed-power field is the
    source term here at dc bandwidth.  two materializations of one implicit model,
    reading the same `mechanical_propagation` and `thermal_diffusion` declarations,
    at bandwidths seven decades apart.  that is the architecture's central claim
    made concrete.""",
    request=MaterializationRequest(
        name="thermal-safety",
        targets=(sel("thermal.temperature", band=THERMAL),
                 sel("metabolic.heat", band=THERMAL)),
        regions=(("skull", OnSupport("head_volume")),
                 ("focus", Near("transducer", 30.0))),
        resolution=res(
            rule(OnSupport("head_volume"), 0.25, THERMAL),
            rule(Near("transducer", 30.0), 0.5, THERMAL),
            rule(Near("transducer", 90.0), 2.0, THERMAL),
            default_mm=4.0, default_band=THERMAL),
        fields=("thermal", "material", "blood", "metabolic", "mechanical"),
        anatomy=("vascular_territories",),
        topologies=("mechanical", "vascular", "metabolic_exchange",
                    "device_coupling"),
        processes=("thermal_diffusion", "mechanical_propagation", "vascular_flow",
                   "metabolism", "device_coupling"),
        observations=("temperature", "temperature"),
        interventions=("tfus", "tms",),
        devices=(DeviceSpec("transducer", "stimulator", "transducer", "stimulate"),),
        subject=SubjectSpec(template=None,
                            note="ct-derived skull density; a template skull "
                                 "underestimates absorption in thick bone"),
        window=Window(n=512, dt=1.0),
        bands=(("thermal", THERMAL), ("blood", THERMAL), ("metabolic", THERMAL),
               ("material", DC)),
        budget=Budget(max_state_variables=2_000_000,
                      max_spectral_coefficients=10_000_000),),
    fit_sources=("itis-database", "skull-acoustics-ct-mri-datasets",
                 "transducer-calibration-datasets", "hydrophone-phantoms",
                 "mida-head-model", "k-wave", "babelbrain"),
    eval_sources=("kwave-reference-problems", "acoustic-closed-forms",
                  "nist-mri-phantoms", "babelbrain-examples"),
    teachers=(),
    negative_controls=("bwin-000-synthetic-scaffold", "tms-artifact-phantoms"),
    constrained=(
        "tissue thermal conductivity, specific heat and acoustic absorption, which "
        "`itis-database` supplies with real uncertainty ranges -- these are "
        "literature values, not fits, and they are the model's foundation",
        "the absorbed power field, inherited from `tfus_response`'s acoustic solve "
        "and validated against hydrophone and phantom measurements",
        "temperature rise in phantoms, where thermometry ground truth exists",),
    prior_dominated=(
        "the in-vivo perfusion response to heating.  perfusion is the dominant "
        "cooling term and it increases with temperature in living tissue by an "
        "amount no source here measures; a phantom-validated model is "
        "systematically conservative in one direction and no one knows by how much",
        "the thermal-dose threshold for damage, taken from the hyperthermia "
        "literature and not from anything in this corpus",
        "skull heterogeneity below the ct resolution -- diploe porosity varies over "
        "hundreds of microns and the 0.25 mm grid is at the limit of what the "
        "imaging supports",),))


PLASTICITY_LEARNING = register(NamedModel(
    id="plasticity_learning",
    doc="""slow structural change with learning and experience.

    the narrowest band in the library: `STRUCTURAL`, 0-0.001 Hz, meaning a state
    variable here is a handful of coefficients over a window of weeks.  the
    temporal axis has essentially collapsed, and everything the model costs it
    costs in geometry.

    **which makes the r(q) decision the only interesting one, and it is coarse.**
    synaptic density changes at the scale of a synapse; the measurements are
    cortical thickness, diffusion metrics and functional-connectivity change at
    2-3 mm, repeated weeks apart, with registration error of a millimetre or so
    between sessions.  a materialization finer than the registration error is
    materializing misalignment.  6 mm through the parenchyma, 2 mm in the
    hippocampal subfields where the structure is thinner than the error
    everywhere else, and the honest statement is that the *estimand* is a
    parcel-level change and the grid exists to carry the topology.

    **all of it in the volume, and the cortical sheet is not named.**  every
    target of this model -- `structural.synaptic_density`,
    `structural.dendritic_density`, `structural.myelination`,
    `structural.axonal_density` -- has the parenchyma volume as its only support,
    because myelination and axonal density are properties of white matter and
    there is no sheet under them.  the neural field enters only as a *drive* to
    `plasticity`, and over a window of sixty daily samples that drive is a
    time-averaged rate: the average of a week of activity over a 6 mm cell is
    exactly the quantity the learning rule reads, so coarse-graining it commutes
    and the geodesic metric buys nothing.  the model materializes `local` and
    `microcircuit`, both of which are declared over the volume as well as the
    sheet and build a euclidean neighbourhood there, which is the right relation
    for the diffusive and metabolic couplings they carry.

    naming the sheet as well would have split the neural drive across two
    indexings and, worse, would have put cortical structural state on column
    nodes whose positions come from a surface reconstruction with its own
    millimetre-scale error -- registering a longitudinal structural change
    against a reconstruction rather than against the image it was measured in.
    the note on registration drift below is the same argument: a model whose
    whole difficulty is that its signal is the size of its misalignment should
    not add a coordinate transform it does not need.

    the one refinement that is not volumetric-by-default is the 2 mm rule in
    `hippocampal_subfields`, and it is a hard dependency rather than a nicety:
    the subfields are the structure this literature reports changing, they are
    1-2 mm across, and a materialization that cannot place them is materializing
    "hippocampus" as one number.  it means the model needs a subfield
    segmentation -- `recon-all`'s optional module or an equivalent -- and says
    so by refusing to sample rather than by silently falling through to 6 mm.

    **the window is the unusual part of the request.**  n=64 samples at one day
    each: a two-month trajectory.  §1's insistence that a state variable is a
    belief about a trajectory over a window rather than a value at an instant is
    doing real work here, because 'synaptic density in this parcel' is only
    meaningful as a slow trajectory and its interesting structure is entirely in
    the shape of the change.

    **the evidence is thin and the model says so.**  `myconnectome` is one person
    scanned for a year; `midnight-scan-club` is ten people scanned repeatedly but
    not across a learning intervention; the lifespan cohorts are cross-sectional
    and cannot separate change from cohort effects.  the learning-rule parameters
    are weak priors.""",
    request=MaterializationRequest(
        name="plasticity-learning",
        targets=(sel("structural.synaptic_density", "structural.dendritic_density",
                     "structural.myelination", "structural.axonal_density",
                     band=SLOW_STRUCTURAL),),
        regions=(("parenchyma", OnSupport("tissue")),
                 ("hippocampus", anat("hippocampal_subfields", "ca1"))),
        resolution=res(
            rule(anat("hippocampal_subfields", "ca1"), 2.0, SLOW_STRUCTURAL),
            rule(OnSupport("tissue"), 6.0, SLOW_STRUCTURAL),
            default_mm=8.0, default_band=SLOW_STRUCTURAL),
        fields=("structural", "neural", "metabolic", "extracellular"),
        anatomy=("cortical_areas", "hippocampal_subfields", "cortical_layers"),
        topologies=("tractometric", "local", "microcircuit"),
        processes=("plasticity", "local_excitation", "local_inhibition",
                   "neuromodulation", "metabolism"),
        observations=("structural_mri", "dwi_microstructure", "bold",
                      "behaviour"),
        interventions=("task_cue",),
        devices=(DeviceSpec("scanner", "scanner_element", "scanner", "observe"),),
        subject=SubjectSpec(),
        window=Window(n=64, dt=86_400.0),
        bands=(("structural", SLOW_STRUCTURAL), ("neural", SLOW_STRUCTURAL),
               ("metabolic", SLOW_STRUCTURAL)),
        budget=Budget(max_state_variables=400_000,
                      max_spectral_coefficients=2_000_000),),
    fit_sources=("myconnectome", "midnight-scan-club", "hcp-young-adult",
                 "ram-intracranial", "wand-cubric"),
    eval_sources=("supercognition-longitudinal", "cam-can", "sald",
                  "hcp-development", "corr-reliability"),
    teachers=(),
    negative_controls=("corr-reliability", "bwin-000-synthetic-scaffold",
                       "example-published-study"),
    constrained=(
        "test-retest variability of every structural measurement, from "
        "`corr-reliability` -- which is what sets the floor on any claimed change",
        "within-subject longitudinal variance in connectivity and thickness, from "
        "`myconnectome`'s year of scans",
        "microstructural parameters at a single timepoint, from `wand-cubric`'s "
        "ultrastrong-gradient diffusion",),
    prior_dominated=(
        "the learning rule itself.  `plasticity`'s functional form and every rate "
        "constant in it are weak priors; no source here observes a synapse",
        "the mapping from synaptic density to any measured mri quantity, which is "
        "assumed and not calibrated -- cortical thickness change has several "
        "candidate microstructural causes and this corpus separates none of them",
        "anything causal about training.  `training_schedule` is declared as an "
        "intervention and no fit source here randomises it",),
    notes="`corr-reliability` is both eval and negative control: a longitudinal "
          "model that finds structural change in scan-rescan data with no "
          "intervening experience is measuring registration drift.",))


SPREADING_DEPOLARIZATION = register(NamedModel(
    id="spreading_depolarization",
    doc="""the slow depolarising wave of cortical spreading depression.

    two to five millimetres per minute, a front one to three millimetres wide, a
    near-complete collapse of transmembrane ion gradients behind it, and a large
    negative dc shift.  the clearest bistable-front problem in the library and the
    one with the least human data.

    **r(q) is 0.5 mm or the answer is wrong, not merely imprecise.**  front speed in
    a bistable reaction-diffusion system scales as the square root of the
    diffusivity times the reaction rate, and a coarse grid changes the effective
    diffusivity of the discretisation.  materialize this at 3 mm and the wave moves
    at the wrong speed -- and speed is the observable.  the front is 1-3 mm wide, so
    0.5 mm gives a handful of cells across it, which is the minimum for a front to
    propagate correctly rather than lock to the grid.

    **extracellular potassium and the volume fraction are the state, not the
    neurons.**  what travels is an ionic and osmotic disturbance: potassium rises
    an order of magnitude, cells swell, the extracellular volume fraction halves,
    and the effective diffusivity changes as a result -- a nonlinearity feeding
    back on the transport coefficient.  a materialization that treated the neural
    population as primary and the ions as a nuisance would have the causality
    backwards.

    **B.**  0-0.5 Hz.  the dc wave and the *envelope* of the activity suppression
    that follows the front both live there; the suppressed 0.5-300 Hz activity
    itself is `lfp_forward`'s view of the same tissue and is not carried on this
    window, because a half-second sample cannot hold it.  two timescales, one
    materialization each, coupled multiplicatively across windows.

    **evidence.**  human recordings exist -- subdural strips in traumatic brain
    injury and after subarachnoid haemorrhage -- but not in this corpus as such;
    the clinical ieeg archives may contain them and the epilepsy sources contain
    related phenomena.  the model is materialized honestly as mostly prior.""",
    request=MaterializationRequest(
        name="spreading-depolarization",
        targets=(sel("extracellular.k", "extracellular.na", "extracellular.ca",
                     "extracellular.volume_fraction", band=SD_WAVE),
                 sel("neural.exc.potential", band=SD_WAVE),
                 sel("electromagnetic.potential", band=SD_WAVE)),
        regions=(("affected", Near("strip", 40.0)),
                 ("cortex", OnSupport("cortical_surface")),
                 ("interstitium", OnSupport("interstitial"))),
        resolution=res(
            rule(Near("strip", 40.0), 0.5, SD_WAVE),
            rule(OnSupport("cortical_surface"), 1.0, SD_WAVE),
            rule(OnSupport("interstitial"), 0.5, SD_WAVE),
            default_mm=6.0, default_band=ULTRASLOW),
        fields=("extracellular", "neural", "electromagnetic", "metabolic", "blood",
                "material"),
        anatomy=("cortical_areas", "cortical_layers", "vascular_territories"),
        topologies=("interstitial", "local", "cortical_surface", "vascular",
                    "metabolic_exchange", "electromagnetic", "device_coupling"),
        processes=("ionic_exchange", "ionic_diffusion", "interstitial_transport",
                   "local_excitation", "local_inhibition", "metabolism",
                   "neurovascular_coupling", "em_generation", "device_coupling"),
        observations=("ieeg", "dc_potential",
                      "bold"),
        devices=(DeviceSpec("strip", "implanted_array", "electrode_grid", "observe",
                            n_elements=6,
                            note="dc-coupled subdural strip; ac-coupled amplifiers "
                                 "cannot see the event at all"),),
        subject=SubjectSpec(id="patient", template=None),
        window=Window(n=2048, dt=0.5),
        # the suppression of normal activity behind the front is carried here as a
        # slow envelope, not as an lfp band: a 0.5 s sample cannot hold 300 Hz, and
        # the lfp view of the same tissue belongs to `lfp_forward`.
        bands=(("extracellular", SD_WAVE), ("neural.exc.potential", SD_WAVE),
               ("neural.exc.activity", SD_WAVE), ("electromagnetic", SD_WAVE),
               ("blood", HEMODYNAMIC)),
        budget=Budget(max_state_variables=1_500_000,
                      max_spectral_coefficients=20_000_000),),
    fit_sources=("clinical-ieeg-archives", "swec-ethz-ieeg"),
    eval_sources=("epilepsy-ecosystem", "ram-intracranial", "brainweb"),
    teachers=("neural-mass-model-families",),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study",
                       "tuh-eeg"),
    constrained=(
        "almost nothing in humans.  the dc shift's amplitude and duration where "
        "dc-coupled clinical recordings exist, and that is the honest list",
        "the co-occurring suppression of high-frequency activity, which ac-coupled "
        "clinical ieeg does show",),
    prior_dominated=(
        "the entire ionic mechanism.  every rate constant in `ionic_exchange` and "
        "`ionic_diffusion` at these amplitudes comes from rodent slice work and "
        "sits at a weak or speculative prior",
        "front speed and shape, which the 0.5 mm grid is built to compute and which "
        "no source here measures in a human at that resolution",
        "the vascular response, which in humans may be hyperaemic or ischaemic "
        "depending on tissue state -- a sign difference the model cannot resolve "
        "and must report as unresolved rather than picking one",),))
