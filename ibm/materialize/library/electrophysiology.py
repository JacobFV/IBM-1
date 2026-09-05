"""forward and inverse materializations for electromagnetic observation.

    M = materialize(R, r, B, F, A, T, P)

everything here terminates on `electromagnetic.*` or `device.contact_potential`,
and the eight models differ almost entirely in r(q) and B.  that is the point of
the package: they read one declaration of `em_generation` and one of
`device_coupling`, and if two of them disagreed about how transmembrane current
becomes a lead field one of them would simply be wrong.

the recurring physical fact that sets r(q) here is the *smoothing kernel between
the source and the sensor*.  a scalp electrode looks through skull and scalp,
which low-pass the source distribution over centimetres, so no amount of
refinement below about a centimetre changes the prediction -- refining it is the
textbook case of coarse-graining commuting with the dynamics.  a subdural contact
looks through a few hundred microns of csf and pia and sees millimetres.  a
laminar probe sits *inside* the dipole layer, where the second derivative across
300 um of cortical depth is the entire quantity of interest, and averaging over a
millimetre destroys it.  the three r(q)s below differ by two orders of magnitude
for that reason and no other.

B is set by the opposite consideration: what the *amplifier and the tissue*
actually let through.  scalp eeg above roughly 100 Hz is muscle and johnson noise,
so a materialization that carries 300 Hz on a scalp target is carrying noise it
cannot constrain.  intracranial contacts are not skull-limited and high gamma is
real signal there, so the band opens.
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
    Band,
    LFP,
    Near,
    OnSupport,
    SPIKE,
    sel,)

# bands used repeatedly below, named so that the reason survives the number.
#: what a scalp amplifier and the skull jointly admit.  the low edge is set by
#: electrode drift and sweat artifact, the high edge by neck and temporalis emg,
#: which above ~100 Hz exceeds cortical signal at the scalp by an order of
#: magnitude.  carrying 300 Hz here would be carrying muscle.
SCALP = Band(0.5, 100.0)
#: meg has no skull attenuation and squid noise is white, so the ceiling is the
#: recording bandwidth rather than physiology; 200 Hz keeps high gamma honestly.
DEWAR = Band(0.5, 200.0)
#: subdural and depth contacts see the full lfp range plus the high-gamma band
#: that is the workhorse of human ecog.
INTRACRANIAL = Band(0.5, 300.0)


EEG_FORWARD = register(NamedModel(
    id="eeg_forward",
    doc="""scalp potential from cortical population state.  spatially broad,
    temporally wide.

    targets `device.contact_potential` on `sensor_array`, reached through
    `em_generation` (transmembrane current to a primary current density) and
    `em_coupling` (the volume conductor) and finally `device_coupling` (the
    electrode-skin interface).

    **why r(q) is coarse.**  the forward operator from cortical current to scalp
    potential is a spatial low-pass whose kernel is centimetres wide: the skull's
    conductivity is roughly one eightieth of brain and csf, and the resulting
    smearing means a 1 mm source grid and a 3 mm source grid produce scalp fields
    that differ by far less than the measurement noise.  refining below that is
    exactly the waste §1 describes -- the state is smooth at the scale the sensor
    integrates over, and the coupling is linear, so coarse-graining commutes.
    the sheet gets 3 mm because cortical *orientation* flips across a sulcus and
    orientation does not survive coarsening even though amplitude does; that, not
    amplitude, is why we do not go to 6 mm on the surface.

    **where it does not commute, and where r(q) is therefore fine.**  the head
    volume itself.  skull thickness varies threefold across the head and the csf
    layer is 1-3 mm thick and four times more conductive than brain, so a 3 mm
    material grid genuinely changes the lead field.  the fine rule below is on
    `head_volume`, not on tissue -- resolution is spent on the conductor, which is
    what the measurement is actually sensitive to.

    **why B is wide but capped.**  a scalp measurement is millisecond-resolved and
    constrains 0.5-100 Hz honestly.  it constrains nothing above that: temporalis
    and neck emg dominate the scalp spectrum from roughly 20 Hz upward and win
    outright past 100 Hz, so a materialization carrying gamma at the scalp is
    materializing muscle and calling it cortex.""",
    request=MaterializationRequest(
        name="eeg-forward",
        targets=(sel("device.contact_potential", band=SCALP),
                 sel("electromagnetic.potential", band=SCALP)),
        regions=(("cortex", OnSupport("cortical_surface")),
                 ("conductor", OnSupport("head_volume"))),
        resolution=res(
            rule(Near("eeg", 15.0), 1.0, SCALP),
            rule(OnSupport("head_volume"), 1.5, SCALP),
            rule(OnSupport("cortical_surface"), 3.0, SCALP),
            default_mm=6.0, default_band=SCALP),
        fields=("neural", "electromagnetic", "material", "device"),
        anatomy=("cortical_areas",),
        topologies=("cortical_surface", "electromagnetic", "device_coupling"),
        processes=("em_generation", "em_coupling", "device_coupling"),
        observations=("eeg",),
        devices=(DeviceSpec("eeg", "sensor_array", "eeg_cap", "observe",
                            n_elements=64, note="64-256 channel scalp montage"),),
        subject=SubjectSpec(id="template", note="individual t1 strongly preferred; "
                            "a template skull is a systematic lead-field error"),
        window=Window(n=2048, dt=1e-3),   # 2 s: a 0.5 Hz low edge needs it
        bands=(("neural", SCALP), ("electromagnetic", SCALP), ("device", SCALP)),
        budget=Budget(max_state_variables=400_000),),
    fit_sources=("erp-core", "ds000117", "mne-sample", "chbmp"),
    eval_sources=("things-eeg2", "broderick-natural-speech-eeg",
                  "sarvas-closed-forms", "fem-bem-analytic-reference-problems"),
    teachers=("eegpt", "labram"),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "the sensor-referenced gain of `device_coupling` per electrode, since a "
        "scalp montage with a common reference over-determines it",
        "skull and scalp conductivity ratio, weakly, through the ratio of "
        "near-field to far-field topography in erp-core's well-characterised "
        "components",
        "the aperiodic slope and the alpha peak of `neural.exc.activity` in "
        "posterior cortex, which are the two things scalp spectra pin best",),
    prior_dominated=(
        "the radial-versus-tangential decomposition of the source current: the "
        "scalp lead field is nearly blind to depth, so anything the fit reports "
        "about source depth is prior",
        "csf layer thickness where no individual t1 is available -- it enters as "
        "an atlas value and its posterior will not move",
        "laminar structure entirely; nothing at 3 mm distinguishes layer ii/iii "
        "from layer v current",),
    notes="`sarvas-closed-forms` and `fem-bem-analytic-reference-problems` are "
          "analytic references and sit in eval deliberately: they score the "
          "forward operator and must never be allowed to move theta.  "
          "`mne-sample` and `chbmp` supply individual bem geometry (calibration) "
          "as well as data.",))


MEG_FORWARD = register(NamedModel(
    id="meg_forward",
    doc="""magnetic field outside the head from transmembrane current.

    physically the same materialization as `eeg_forward` up to the last step, and
    deliberately written to share it: `em_generation` is the same process and the
    same parameters.  what changes is the coupling, and the change matters for
    r(q) in one specific way.

    **why r(q) is finer on the sheet than eeg.**  the magnetic lead field is not
    filtered by the skull -- magnetic permeability is uniform through bone -- so
    the sensor genuinely resolves finer source structure, and 2 mm on the cortical
    sheet buys something eeg's 3 mm does not.  but the whole gain is spent on
    *tangential* sources: a radial dipole in a spherically symmetric conductor
    produces exactly zero external field, and in a real head it produces very
    little.  so meg sees sulcal walls and is nearly blind to gyral crowns, which
    is why the crown rule is coarser than the wall rule would be if we could write
    one -- and why the null space is listed under prior-dominated rather than
    quietly ignored.

    **why the head volume gets less resolution than in eeg.**  the magnetic
    forward problem barely depends on skull conductivity at all; the sarvas
    formula needs only the inner skull surface.  spending state variables on a
    1.5 mm conductor grid here would be spending them on something the
    measurement cannot see.  this is the clearest contrast in the package: the
    same head, two instruments, and r(q) inverts between them.

    **B.**  no skull attenuation and white squid noise, so the ceiling is the
    recording bandwidth and high gamma survives to the sensor.  200 Hz.""",
    request=MaterializationRequest(
        name="meg-forward",
        targets=(sel("electromagnetic.bfield", band=DEWAR),
                 sel("device.contact_potential", band=DEWAR)),
        regions=(("cortex", OnSupport("cortical_surface")),),
        resolution=res(
            rule(OnSupport("cortical_surface"), 2.0, DEWAR),
            rule(OnSupport("head_volume"), 4.0, DEWAR),
            default_mm=8.0, default_band=DEWAR),
        fields=("neural", "electromagnetic", "material", "device"),
        anatomy=("cortical_areas",),
        topologies=("cortical_surface", "electromagnetic", "device_coupling"),
        processes=("em_generation", "em_coupling", "device_coupling"),
        observations=("meg", "meg"),
        devices=(DeviceSpec("meg", "sensor_array", "meg_head", "observe",
                            n_elements=306,
                            note="magnetometer/planar-gradiometer helmet"),),
        subject=SubjectSpec(note="head position relative to the dewar is a "
                            "per-run nuisance and is fitted, not assumed"),
        window=Window(n=2048, dt=1e-3),
        bands=(("neural", DEWAR), ("electromagnetic", DEWAR)),
        budget=Budget(max_state_variables=600_000),),
    fit_sources=("ds000117", "mne-sample", "hcp-meg", "mne-somato",
                 "vectorview-306-array-geometry"),
    eval_sources=("meg-masc", "sarvas-closed-forms", "omega",
                  "fem-bem-analytic-reference-problems"),
    teachers=("meg-gpt", "brainomni"),
    negative_controls=("bwin-000-synthetic-scaffold",),
    constrained=(
        "source orientation on sulcal walls, which is the one geometric quantity "
        "meg pins better than any other non-invasive instrument",
        "inner-skull surface placement, through the fit of evoked topographies in "
        "`mne-somato` where the generator is known",
        "sensor gain and crosstalk per channel, over-determined by 306 sensors",),
    prior_dominated=(
        "radial source current everywhere -- it lies in the null space of the "
        "magnetic lead field, so its posterior is its prior by construction and "
        "any map showing gyral crown activity is showing the prior",
        "deep and subcortical generators: signal falls off faster than 1/r^2 and "
        "hippocampal or brainstem estimates here are regularisation, not evidence",
        "skull conductivity, which meg simply does not measure",),
    notes="`vectorview-306-array-geometry` is device calibration and fixes sensor "
          "positions; it contributes geometry, not a likelihood on brain state.",))


ECOG_FORWARD = register(NamedModel(
    id="ecog_forward",
    doc="""subdural grid and strip potentials from cortical population state.

    the intermediate case, and the one that shows r(q) is set by the kernel and
    not by the budget.  a subdural contact sits a fraction of a millimetre above
    pia with only csf between it and the source, so the smoothing kernel is
    millimetres rather than centimetres and a 2-3 mm grid materialization would
    visibly blur what the contact reports.

    **why 0.25 mm near a contact.**  the electrode is close enough that the 1/r
    weighting is steep across the cortical ribbon directly beneath it: the top
    300 um and the bottom 300 um of the ribbon contribute with weights differing
    by a factor of several, so the laminar current profile does not average out.
    that is a genuine failure of commutation, not proximity worship -- and it is
    also why the polarity of an ecog deflection depends on which layer the
    synaptic current is in.

    **why it coarsens so fast.**  a centimetre from the contact the same cortical
    patch is a smooth multipole and 1 mm is already exact; three centimetres away
    3 mm is exact.  the whole model is affordable only because the fine shell is a
    few cubic centimetres.

    **B opens to 300 Hz** because there is no skull and no scalp muscle.  high
    gamma (70-200 Hz) is the band ecog actually earns its reputation on, and
    excluding it would remove the reason to implant.""",
    request=MaterializationRequest(
        name="ecog-forward",
        targets=(sel("device.contact_potential", region=Near("grid", 10.0),
                     band=INTRACRANIAL),
                 sel("neural.transmembrane_current", region=Near("grid", 20.0),
                     band=INTRACRANIAL)),
        regions=(("under_grid", Near("grid", 20.0)),
                 ("connected", OnSupport("cortical_surface"))),
        resolution=res(
            rule(Near("grid", 5.0), 0.25, INTRACRANIAL),
            rule(Near("grid", 20.0), 1.0, INTRACRANIAL),
            rule(OnSupport("cortical_surface"), 3.0, LFP),
            default_mm=8.0, default_band=LFP),
        fields=("neural", "electromagnetic", "material", "device"),
        anatomy=("cortical_areas", "cortical_layers"),
        topologies=("cortical_surface", "laminar", "electromagnetic",
                    "device_coupling"),
        processes=("em_generation", "em_coupling", "device_coupling",
                   "laminar_propagation"),
        observations=("ecog",),
        devices=(DeviceSpec("grid", "implanted_array", "electrode_grid", "observe",
                            n_elements=128,
                            note="subdural grid or strip, 3-10 mm pitch"),),
        subject=SubjectSpec(id="patient", template=None,
                            note="contact localisation from post-implant ct is the "
                                 "dominant geometric error, not the atlas"),
        window=Window(n=8192, dt=5e-4),
        bands=(("neural", INTRACRANIAL), ("electromagnetic", INTRACRANIAL)),
        budget=Budget(max_state_variables=800_000),),
    fit_sources=("kai-miller-ecog-library", "ds004194", "chang-lab-ecog-cv-syllables",
                 "ds003688", "mni-open-ieeg-atlas"),
    eval_sources=("ds005574", "brain-treebank", "neurotycho"),
    teachers=("brainbert", "biot"),
    negative_controls=("bwin-000-synthetic-scaffold",),
    constrained=(
        "the local spatial fall-off of the potential, since a 128-contact grid "
        "samples the same patch at several distances and over-determines it",
        "high-gamma amplitude as a function of local population activity, which "
        "is the best-constrained coupling in the whole library",
        "contact impedance and its drift, from `electrode-impedance-contact-models` "
        "priors plus the recordings themselves",),
    prior_dominated=(
        "the laminar decomposition beneath each contact: 0.25 mm sites exist "
        "there, but one contact per patch cannot separate layer ii/iii from "
        "layer v currents, so their split stays at its cytoarchitectural prior",
        "anything more than about 2 cm from the nearest contact -- coverage in "
        "human implants is clinical, not scientific, and the rest of the brain "
        "is unconstrained",
        "the csf layer thickness between pia and contact, which shifts amplitude "
        "multiplicatively and trades off with source strength",),))


LFP_FORWARD = register(NamedModel(
    id="lfp_forward",
    doc="""local field potential at a depth electrode or laminar probe.

    the same physics as `ecog_forward` moved inside the tissue, which changes the
    accounting completely: the contact is now surrounded by sources rather than
    looking at them from outside, and the 1/r kernel has no far side.

    **why 50 um.**  within a few hundred microns of a contact the weighting varies
    by an order of magnitude across a single cortical layer.  the lfp at that
    contact is a weighted sum in which the weights change faster than the state
    does, so the average of the product is not the product of the averages --
    exactly the nonlinearity-in-the-coarse-cell criterion of §1.  50 um is the
    scale at which the weight is flat over a cell.

    **why it is affordable.**  the fine shell is a 4 mm ball, about 250 thousand
    sites at 50 um if it were solid, and it is not solid because tissue occupies
    part of it and the octree stops at the shell.  the model is bought by the
    smallness of the region, not by narrowing the band -- which stays wide.

    **B.**  0.5-300 Hz.  the lfp proper stops there; the spike band above it is a
    different observation on a different variable and belongs to `spike_decode`,
    not here.  keeping them separate is what stops a single materialization
    quietly claiming to explain both with one coupling.""",
    request=MaterializationRequest(
        name="lfp-forward",
        targets=(sel("device.contact_potential", region=Near("probe", 5.0),
                     band=INTRACRANIAL),
                 sel("neural.transmembrane_current", region=Near("probe", 4.0),
                     band=INTRACRANIAL)),
        regions=(("shell", Near("probe", 4.0)),
                 ("neighbourhood", Near("probe", 20.0))),
        resolution=res(
            rule(Near("probe", 1.0), 0.05, INTRACRANIAL),
            rule(Near("probe", 5.0), 0.2, INTRACRANIAL),
            rule(Near("probe", 20.0), 1.0, LFP),
            default_mm=6.0, default_band=LFP),
        fields=("neural", "electromagnetic", "extracellular", "material", "device"),
        anatomy=("cortical_layers", "cytoarchitecture", "thalamic_nuclei"),
        topologies=("local", "laminar", "microcircuit", "electromagnetic",
                    "device_coupling"),
        processes=("em_generation", "em_coupling", "local_excitation",
                   "local_inhibition", "device_coupling"),
        observations=("lfp",),
        devices=(DeviceSpec("probe", "implanted_array", "electrode_grid", "observe",
                            n_elements=384,
                            note="linear silicon probe or seeg depth lead"),),
        subject=SubjectSpec(id="patient"),
        window=Window(n=16384, dt=2e-4),
        bands=(("neural", INTRACRANIAL), ("electromagnetic", INTRACRANIAL),
               ("extracellular", Band(0.0, 10.0))),
        budget=Budget(max_state_variables=1_200_000,
                      max_spectral_coefficients=400_000_000),),
    fit_sources=("steinmetz-neuropixels-2019", "allen-visual-coding-neuropixels",
                 "ibl-brain-wide-map", "electrode-impedance-contact-models"),
    eval_sources=("swec-ethz-ieeg", "single-neuron-ieeg-fmri-movie",
                  "mni-open-ieeg-atlas"),
    teachers=("brainbert",),
    negative_controls=("bwin-000-synthetic-scaffold",),
    constrained=(
        "the effective extracellular conductivity and its frequency dependence, "
        "from the way lfp amplitude falls with contact distance on a linear probe",
        "the ratio of synaptic to spike-related contributions to the 0.5-300 Hz "
        "potential, over-determined when spikes are recorded on the same contacts",
        "population activity in the 1 mm around each contact -- the only place in "
        "this library where the target variable is directly and locally observed",),
    prior_dominated=(
        "everything past 2 cm from the probe; a neuropixels shank samples a "
        "thousandth of a percent of the brain and the rest is the prior",
        "the species transfer.  `steinmetz-neuropixels-2019`, "
        "`allen-visual-coding-neuropixels` and `ibl-brain-wide-map` are mouse, so "
        "human parameters inherit a scaling assumption that no card constrains",
        "cell-type-specific contributions (`neural.pv.activity` and friends), "
        "which the lfp cannot separate without optotagging",),))


CSD_LAMINAR = register(NamedModel(
    id="csd_laminar",
    doc="""current source density through the cortical ribbon.

    the sharpest r(q) in the library, and the easiest to justify.  csd is the
    second spatial derivative of the extracellular potential along the depth axis,
    and a second derivative is precisely the operator that does not survive
    coarse-graining: averaging the potential over a millimetre before
    differentiating it returns something with the wrong sign in the wrong place.
    the human cortical ribbon is 2-4 mm and the dipole layers inside it are
    200-400 um, so 50 um along `cortical_depth` is the coarsest sampling at which
    the target quantity still exists.  this is not resolution bought for accuracy;
    below about 100 um the answer is not merely less precise, it is a different
    function.

    **the region is deliberately tiny.**  one probe track, a few square
    millimetres of sheet, tens of microns through depth.  everything else in the
    materialization is coarse because the depth profile is the only thing that
    needs the resolution -- lateral structure over the same patch is smooth.

    **B.**  0.5-300 Hz for the potential, and the model also reads the spike band
    because unit activity on the same contacts is what identifies which sink
    belongs to which population.  carrying `SPIKE` on `neural.exc.activity` while
    the potential stays at lfp bandwidth is the per-component band override the
    request interface exists for.""",
    request=MaterializationRequest(
        name="csd-laminar",
        targets=(sel("neural.transmembrane_current", region=Near("probe", 2.0),
                     band=INTRACRANIAL),
                 sel("electromagnetic.current_density", region=Near("probe", 2.0),
                     band=INTRACRANIAL)),
        regions=(("column", Near("probe", 2.0)),
                 ("ribbon", OnSupport("cortical_depth"))),
        resolution=res(
            rule(OnSupport("cortical_depth"), 0.05, INTRACRANIAL),
            rule(Near("probe", 2.0), 0.05, INTRACRANIAL),
            rule(Near("probe", 10.0), 0.5, LFP),
            default_mm=6.0, default_band=LFP),
        fields=("neural", "electromagnetic", "extracellular", "structural"),
        anatomy=("cortical_layers", "cytoarchitecture"),
        topologies=("laminar", "microcircuit", "local", "electromagnetic"),
        processes=("laminar_propagation", "em_generation", "local_excitation",
                   "local_inhibition", "thalamocortical_coupling"),
        observations=("lfp", "intracortical_spikes"),
        devices=(DeviceSpec("probe", "implanted_array", "electrode_grid", "observe",
                            n_elements=64,
                            note="laminar probe, 20-100 um contact pitch"),),
        subject=SubjectSpec(id="patient"),
        window=Window(n=32768, dt=1e-4),  # 3.3 s at 10 kHz: spikes and a 0.5 Hz floor
        bands=(("electromagnetic", INTRACRANIAL),
               ("neural.exc.activity", SPIKE),
               ("neural.inh.activity", SPIKE),
               ("neural.transmembrane_current", INTRACRANIAL)),
        budget=Budget(max_state_variables=300_000,
                      max_spectral_coefficients=400_000_000),),
    fit_sources=("allen-visual-coding-neuropixels", "steinmetz-neuropixels-2019",
                 "ibl-brain-wide-map"),
    eval_sources=("bigbrain", "julich-brain", "allen-cell-types-patchseq"),
    teachers=(),
    negative_controls=("bwin-000-synthetic-scaffold",),
    constrained=(
        "the depth of the principal sink relative to the layer iv boundary, which "
        "is what a laminar probe measures and essentially nothing else does",
        "the latency ordering of thalamocortical versus feedback sinks, from "
        "visual-flash responses in the allen and ibl recordings",
        "the sign convention and gain of `laminar_propagation`, over-determined by "
        "having 64 contacts through a 3 mm ribbon",),
    prior_dominated=(
        "human laminar structure entirely: every fit source here is mouse, and "
        "the human cortical ribbon is thicker and layer-proportioned differently.  "
        "the human posterior is the mouse posterior warped by a cytoarchitectural "
        "prior and must be reported that way",
        "per-layer synaptic gains, which trade off exactly against per-layer "
        "population sizes and are jointly unidentifiable from the potential alone",
        "the conductivity anisotropy along the depth axis, taken from literature "
        "and not moved by any of these recordings",),))


EEG_SOURCE = register(NamedModel(
    id="eeg_source",
    doc="""cortical population state from scalp potentials -- the inverse of
    `eeg_forward`, materialized against the same declaration.

    this model exists mainly to make an uncomfortable number explicit.  a 64-channel
    montage supplies 63 independent measurements per time sample.  the source grid
    below has tens of thousands of sites.  the map from sites to sensors therefore
    has a null space of dimension in the tens of thousands, and *every* structure
    the estimate shows inside that null space came from the prior, not from the
    data.  the fine grid is a representation of a smooth answer, not a claim to
    have resolved it.

    **why r(q) is nevertheless not made coarse.**  because a coarse grid would
    hide the problem rather than fix it: 5 mm sites would still exceed the sensor
    count, and orientation would be lost as well.  the honest arrangement is a
    grid fine enough that the geometry is right, plus an explicit statement -- the
    `prior_dominated` list below -- that spatial detail beyond roughly the sensor
    count is regularisation.

    **B.**  same 0.5-100 Hz as the forward model, and for the same reason.  the
    inverse cannot recover a band the forward map never carried.""",
    request=MaterializationRequest(
        name="eeg-source",
        targets=(sel("neural.exc.activity", "neural.inh.activity",
                     region=OnSupport("cortical_surface"), band=SCALP),),
        regions=(("cortex", OnSupport("cortical_surface")),),
        resolution=res(
            rule(OnSupport("cortical_surface"), 3.0, SCALP),
            rule(OnSupport("head_volume"), 2.0, SCALP),
            default_mm=8.0, default_band=SCALP),
        fields=("neural", "electromagnetic", "material", "device"),
        anatomy=("cortical_areas",),
        topologies=("cortical_surface", "tractometric", "electromagnetic",
                    "device_coupling"),
        processes=("em_generation", "em_coupling", "device_coupling",
                   "lateral_cortical_propagation"),
        observations=("eeg",),
        devices=(DeviceSpec("eeg", "sensor_array", "eeg_cap", "observe",
                            n_elements=64),),
        subject=SubjectSpec(),
        window=Window(n=2048, dt=1e-3),   # 2 s: a 0.5 Hz low edge needs it
        bands=(("neural", SCALP), ("electromagnetic", SCALP)),
        budget=Budget(max_state_variables=300_000),),
    fit_sources=("erp-core", "ds000117", "mne-sample", "ds000116", "nm000254-natview"),
    eval_sources=("hcp-meg", "ds003688", "kai-miller-ecog-library",
                  "frequency-resolved-source-association"),
    teachers=("eegpt", "labram", "cbramod"),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "the low-order spatial modes of the source distribution -- roughly as many "
        "as there are sensors, and no more",
        "the coarse anterior-posterior and left-right localisation of well-studied "
        "erp generators, checked against the simultaneous ieeg in `ds003688`",
        "the spectral content of each retained mode, which is genuinely "
        "millisecond-resolved",),
    prior_dominated=(
        "spatial detail beyond about the sensor count.  a 3 mm source map from 64 "
        "channels is a smooth prior painted onto a fine grid and must never be "
        "reported as a resolved map",
        "source depth: the scalp lead field trades depth against amplitude almost "
        "exactly, so any depth estimate is the depth prior",
        "the excitatory/inhibitory split at each site, which no scalp measurement "
        "separates -- both components exist in the materialization and only their "
        "weighted sum is observed",),))


MEG_SOURCE = register(NamedModel(
    id="meg_source",
    doc="""cortical population state from magnetometer and gradiometer data.

    the same inverse honesty as `eeg_source` with a different, sharper null space,
    and this is why both are named rather than one parameterised model: the two
    instruments fail in *different* places, and a combined fit is worth more than
    either because their null spaces do not coincide.

    **the null space is structural, not statistical.**  a radially oriented dipole
    produces no external magnetic field in a spherical conductor.  gyral crowns are
    approximately radial.  so a meg source estimate is systematically blind to
    crown activity in a way no amount of data fixes, and every crown value in the
    output came from the prior.  `prior_dominated` says so in those words.

    **why 2 mm on the sheet.**  306 sensors, three hundred-ish independent modes,
    and a lead field that varies fast across a sulcal wall where orientation flips
    over 1-2 mm.  the grid is fine because orientation needs it; the *rank* is
    still three hundred, and the two facts are not the same fact.

    **B.**  0.5-200 Hz, matching the forward model.""",
    request=MaterializationRequest(
        name="meg-source",
        targets=(sel("neural.exc.activity", "neural.inh.activity",
                     region=OnSupport("cortical_surface"), band=DEWAR),),
        regions=(("cortex", OnSupport("cortical_surface")),),
        resolution=res(
            rule(OnSupport("cortical_surface"), 2.0, DEWAR),
            default_mm=8.0, default_band=DEWAR),
        fields=("neural", "electromagnetic", "material"),
        anatomy=("cortical_areas",),
        topologies=("cortical_surface", "tractometric", "electromagnetic",
                    "device_coupling"),
        processes=("em_generation", "em_coupling", "device_coupling",
                   "lateral_cortical_propagation"),
        observations=("meg", "meg"),
        devices=(DeviceSpec("meg", "sensor_array", "meg_head", "observe",
                            n_elements=306),),
        subject=SubjectSpec(),
        window=Window(n=2048, dt=1e-3),
        bands=(("neural", DEWAR), ("electromagnetic", DEWAR)),
        budget=Budget(max_state_variables=500_000),),
    fit_sources=("hcp-meg", "ds000117", "mne-somato", "mne-spm-face", "cam-can"),
    eval_sources=("hcp-meg-maps", "kymata-soto", "meg-masc",
                  "frequency-resolved-source-association"),
    teachers=("meg-gpt", "brainomni", "brain-harmony"),
    negative_controls=("bwin-000-synthetic-scaffold",),
    constrained=(
        "tangential source amplitude and orientation on sulcal walls, to a few "
        "millimetres in the well-covered lateral convexity",
        "evoked latency structure, which meg measures better than anything else "
        "non-invasive and which `mne-somato` fixes against a known generator",
        "sensor-level noise covariance, from the empty-room and rest segments",),
    prior_dominated=(
        "radial (gyral crown) sources -- structurally invisible.  a crown value in "
        "the output is the prior and nothing else",
        "deep sources: hippocampus, thalamus, brainstem.  claims about them here "
        "are regularisation choices dressed as anatomy",
        "the absolute scale of population activity, which trades off against the "
        "number of synchronously active cells at each site",),))


EEG_PREDICT = register(NamedModel(
    id="eeg_predict",
    doc="""next-window prediction of scalp eeg from its own past.

    the odd one out: its target is the *sensor* variable, not brain state, and its
    r(q) is deliberately the coarsest in this module.  that is a considered choice
    rather than laziness.  predicting the next second of a 64-channel scalp
    recording is a 64-dimensional problem; a fine cortical grid adds state that
    the objective cannot distinguish, and a coarser materialization that predicts
    equally well is the demonstration that the finer one was waste.  this model is
    therefore the natural partner of `MaterializationRequest.coarsened` -- it is
    where the §1 criterion gets *tested* rather than asserted.

    **B.**  0.5-100 Hz, and the interesting structure is almost entirely in how
    the aperiodic background and the alpha rhythm interact.  the spectral form
    earns its keep here: an ongoing rhythm is an isotropic gaussian on the
    (cos, sin) plane at its frequency and an evoked response is an anisotropic
    one, so prediction is a statement about that covariance rather than about a
    waveform.

    **on the teachers.**  this is the most teacher-dependent model in the library
    and therefore the most dangerous.  `eegpt`, `labram`, `cbramod`, `neuro-gpt`,
    `bendr`, `biot` and `neurolm` all predict masked eeg, and every one of them is
    trained on overlapping public corpora.  their agreement is not evidence.  each
    contributes at calibrated precision derived from its own reported accuracy on
    the stream in question, low-rank rather than diagonal because their residuals
    covary across channels, and never at unit precision.""",
    request=MaterializationRequest(
        name="eeg-predict",
        targets=(sel("device.contact_potential", band=SCALP),),
        regions=(("cortex", OnSupport("cortical_surface")),),
        resolution=res(
            rule(OnSupport("cortical_surface"), 6.0, SCALP),
            default_mm=10.0, default_band=SCALP),
        fields=("neural", "electromagnetic", "device"),
        anatomy=("cortical_areas",),
        topologies=("cortical_surface", "electromagnetic", "device_coupling"),
        processes=("em_generation", "em_coupling", "device_coupling",
                   "lateral_cortical_propagation"),
        observations=("eeg",),
        devices=(DeviceSpec("eeg", "sensor_array", "eeg_cap", "observe",
                            n_elements=64),),
        subject=SubjectSpec(),
        window=Window(n=2048, dt=2e-3),
        bands=(("neural", SCALP), ("device", SCALP)),
        budget=Budget(max_state_variables=100_000),),
    fit_sources=("tuh-eeg", "erp-core", "tdbrain", "chbmp", "healthy-brain-network"),
    eval_sources=("eegmmidb", "things-eeg2", "broderick-natural-speech-eeg",
                  "moabb", "braindecode"),
    teachers=("eegpt", "labram", "cbramod", "neuro-gpt", "bendr", "biot",
              "neurolm", "brainlm"),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "the aperiodic exponent and knee per channel, which large corpora pin "
        "tightly and which vary meaningfully across subjects",
        "alpha peak frequency and its occipital-to-frontal gradient",
        "the channel covariance structure, from the sheer volume of `tuh-eeg`",),
    prior_dominated=(
        "everything about the cortical source layer.  the target is a sensor "
        "variable and the fit can trade source geometry against sensor mixing "
        "freely, so the source-side parameters barely move",
        "cross-frequency coupling beyond band-block covariance; the corpora are "
        "large but the estimand is weak and the posterior stays near the prior",
        "any subject-specific structure in the teachers' output, since their "
        "residuals are correlated and they contribute a low-rank increment only",),
    notes="distilled values enter at calibrated precision -- delta-J bounded by "
          "1/((1-r^2) Var[x]) from each teacher's own reported accuracy, inflated "
          "for off-distribution use, and low-rank.  a teacher at unit precision "
          "would make this model hold its biases as firmly as its measurements.",))
