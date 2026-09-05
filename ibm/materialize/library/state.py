"""global state, disorder and counterfactual materializations.

    M = materialize(R, r, B, F, A, T, P)

these six differ from the rest of the library in what they are *for*: not
predicting an instrument's output but characterising a regime the whole brain is
in, or asking what would happen if part of it changed.  that shifts both levers in
the same direction and for the same reason.

**r(q) is coarse almost everywhere here, and that is the honest choice.**  a sleep
model is fitted to two scalp channels; an anaesthesia model to a montage and a drug
concentration; a resting-fc model to a parcellated time series.  the estimand in
each case is a handful of numbers -- a thalamocortical loop gain, a slow-oscillation
frequency, a connectivity matrix -- and materializing them on a fine grid does not
make them better determined, it makes the prior-dominated fraction larger while
hiding that fact behind a pretty map.  §7's requirement that prior-dominated
structure not be presented like constrained structure is easiest to honour by not
materializing it in the first place.

**the exception is `seizure_propagation`, and it is instructive.**  a seizure is a
travelling front with a sharp spatial edge and a bistable local nonlinearity;
coarse-graining a front changes its speed, which is the one quantity the model
exists to predict.  it is also the widest band in the library -- ictal dc shifts
below 0.1 Hz and high-frequency oscillations above 250 Hz are both diagnostic --
so it is expensive on both axes at once, and its region is restricted to the
implanted volume to pay for that.
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
    HEMODYNAMIC,
    LFP,
    Near,
    OnSupport,
    ULTRASLOW,
    sel,)

#: ictal recordings need everything: the dc shift that accompanies the
#: depolarisation block, and the 80-250 Hz ripples and fast ripples that mark the
#: seizure onset zone.  one band, six decades, and it is not padding.
ICTAL = Band(0.0, 600.0)
#: sleep: the slow oscillation at 0.5-1 Hz, spindles at 11-16 Hz, and the beta
#: activity that separates rem from n2.  nothing above 30 Hz is scored.
SLEEP = Band(0.16, 30.0)
#: anaesthesia: alpha anteriorisation, slow-delta power, and burst suppression --
#: all below 45 Hz, and burst suppression pushes the low edge below the sleep band.
SEDATION = Band(0.1, 45.0)
#: neuromodulator concentration changes over minutes.
PHARMACOKINETIC = Band(0.0, 0.02)
#: what the scalp observation in `pharmaco` can carry.  the drug modulates
#: dynamics up to gamma; the *measurement* stops at 100 Hz, so materializing 300
#: would be materializing an unobserved band on the strength of a prior.
MODULATED = Band(0.5, 100.0)
#: the resting-state fc band.  the low edge excludes scanner drift; the high edge
#: excludes aliased cardiac signal at a 1-2 s tr.
RESTING = Band(0.01, 0.1)


SEIZURE_PROPAGATION = register(NamedModel(
    id="seizure_propagation",
    doc="""ictal onset and spread through the implanted volume.

    the only model in this module that earns fine resolution, and it earns it twice.

    **the front.**  a seizure spreads as a travelling wave at centimetres per
    second with a transition zone one to a few millimetres wide, driven by a
    bistable local nonlinearity plus extracellular potassium accumulation.  the
    propagation speed of a bistable front depends on the ratio of the coupling
    strength to the width of the transition, so materializing the front at 5 mm
    does not merely blur it -- it returns the wrong speed and therefore the wrong
    arrival times, which are the model's entire output.  1 mm near contacts and
    3 mm through connected tissue is where the front is resolved.

    **the band.**  ictal dc shifts sit below 0.1 Hz and are one of the more
    reliable onset markers; high-frequency oscillations at 80-250 Hz, and fast
    ripples above that, are the other.  a materialization that dropped either end
    would drop half the evidence.  0-600 Hz is genuinely required and is why this
    model is restricted to the implanted volume: it cannot afford to be whole-brain
    at that bandwidth.

    **extracellular potassium is a first-class state here**, not a nuisance.  the
    accumulation that sustains propagation is a slow (ULTRASLOW-band) variable
    coupled to a fast one, which is exactly the cross-timescale situation §1 says
    a spectral representation of a single variable handles and a scalar one does
    not.

    **what it cannot do.**  human ictal data is clinical: coverage is where the
    surgeon put contacts, seizures are few per patient, and there is no ground truth
    for tissue between electrodes.  the model is materialized at 1 mm in a volume
    sampled at 5-10 mm.""",
    request=MaterializationRequest(
        name="seizure-propagation",
        targets=(sel("neural.exc.activity", "neural.inh.activity",
                     region=Near("contacts", 40.0), band=ICTAL),
                 sel("extracellular.k", region=Near("contacts", 40.0), band=ULTRASLOW),
                 sel("device.contact_potential", band=ICTAL)),
        regions=(("implanted", Near("contacts", 40.0)),
                 ("onset_zone", Near("contacts", 10.0)),
                 ("connected", OnSupport("cortical_surface"))),
        resolution=res(
            rule(Near("contacts", 10.0), 1.0, ICTAL),
            rule(Near("contacts", 40.0), 3.0, LFP),
            rule(OnSupport("cortical_surface"), 5.0, LFP),
            default_mm=8.0, default_band=LFP),
        fields=("neural", "extracellular", "electromagnetic", "metabolic", "device"),
        anatomy=("cortical_areas", "hippocampal_subfields", "cortical_layers"),
        topologies=("local", "cortical_surface", "laminar", "tractometric",
                    "interstitial", "electromagnetic", "device_coupling"),
        processes=("local_excitation", "local_inhibition", "ionic_exchange",
                   "ionic_diffusion", "lateral_cortical_propagation",
                   "tract_propagation", "em_generation", "device_coupling",
                   "metabolism"),
        observations=("ieeg", "behaviour"),
        devices=(DeviceSpec("contacts", "implanted_array", "electrode_grid", "observe",
                            n_elements=128,
                            note="seeg depth leads and/or subdural grids"),),
        subject=SubjectSpec(id="patient", template=None),
        window=Window(n=16384, dt=5e-4),
        bands=(("neural", ICTAL), ("electromagnetic", ICTAL),
               ("extracellular", ULTRASLOW), ("metabolic", HEMODYNAMIC)),
        budget=Budget(max_state_variables=1_000_000,
                      max_spectral_coefficients=500_000_000),),
    fit_sources=("swec-ethz-ieeg", "clinical-ieeg-archives", "epilepsy-ecosystem",
                 "task-ieeg-propagation-fields", "mni-open-ieeg-atlas"),
    eval_sources=("swec-ethz-ieeg", "epilepsy-ecosystem", "ram-intracranial",
                  "meld"),
    teachers=("neural-mass-model-families", "the-virtual-brain", "brainbert"),
    negative_controls=("tuh-eeg", "bwin-000-synthetic-scaffold",
                       "example-published-study"),
    constrained=(
        "arrival times at the recorded contacts, which is the direct observation "
        "and which `swec-ethz-ieeg`'s long continuous recordings supply in quantity",
        "the spectral signature of the onset zone -- the dc shift and the "
        "high-frequency oscillation rate -- as a per-contact quantity",
        "propagation velocity along the recorded path, from arrival-time gradients "
        "across a grid",),
    prior_dominated=(
        "everything between contacts.  a 1 mm materialization interpolated from "
        "5-10 mm sampling is a smooth prior with a fine mesh; the front's exact "
        "path is not measured",
        "extracellular potassium.  it is central to the mechanism and no human "
        "recording here measures it; the coupling comes from animal work at a weak "
        "prior",
        "the tissue that was resected and the tissue nobody implanted -- which is "
        "most of the brain, and which the seizure demonstrably traverses",),
    notes="`tuh-eeg` is the negative control that matters: a propagation model that "
          "flags onset zones in routine scalp recordings of non-epileptic patients "
          "is detecting an artifact, and its other numbers mean nothing until it "
          "produces a null there.",))


SLEEP_DYNAMICS = register(NamedModel(
    id="sleep_dynamics",
    doc="""the sleep cycle as a slow trajectory of thalamocortical state.

    the coarsest r(q) in the library after the surrogates, and the coarseness is
    the honest part of the model rather than a concession.

    **why 8 mm.**  the fit sources are polysomnography: two eeg channels, an eog
    and a chin emg, scored in 30 s epochs.  four channels cannot constrain four
    thousand cortical sites, and the estimands -- slow-oscillation frequency,
    spindle rate and duration, the thalamic gate's gain -- are global or
    thalamus-local scalars.  materializing cortex at 3 mm would multiply the
    prior-dominated fraction by an order of magnitude and change nothing that the
    data touch.  the one refinement is the thalamus at 2 mm, because the reticular
    nucleus is a 1-2 mm thick shell whose separation from the relay nuclei is the
    whole mechanism of spindle generation, and it does not survive an 8 mm cell.

    **B.**  0.16-30 Hz.  the low edge is set by the 30 s epoch's frequency
    resolution and by the slow oscillation at 0.5-1 Hz being the lowest scored
    feature; the high edge by nothing above beta being scored.  the window is long
    -- minutes -- because a sleep cycle is a ninety-minute object and transitions
    between stages are what the model is about.

    **the interesting representational point.**  spindles are transient, phase-
    structured events on a stationary background, which in the spectral form is a
    band-block anisotropy appearing and disappearing within the sigma band of a
    single state variable.  a scalar-gaussian representation of cortical activity
    could not express that; §1's several-interacting-timescales criterion is met
    here about as cleanly as anywhere.""",
    request=MaterializationRequest(
        name="sleep-dynamics",
        targets=(sel("neural.exc.activity", "neural.inh.activity", band=SLEEP),
                 sel("extracellular.acetylcholine", "extracellular.noradrenaline",
                     "extracellular.adenosine", band=PHARMACOKINETIC)),
        regions=(("thalamus", Anat("thalamic_nuclei", "reticular")),
                 ("cortex", OnSupport("cortical_surface")),
                 ("brainstem", Anat("brainstem_nuclei", "locus_coeruleus"))),
        resolution=res(
            rule(Anat("thalamic_nuclei", "reticular"), 2.0, SLEEP),
            rule(Anat("brainstem_nuclei", "locus_coeruleus"), 2.0, SLEEP),
            rule(OnSupport("cortical_surface"), 8.0, SLEEP),
            default_mm=10.0, default_band=SLEEP),
        fields=("neural", "extracellular", "electromagnetic", "device"),
        anatomy=("thalamic_nuclei", "brainstem_nuclei", "cortical_areas"),
        topologies=("cortical_surface", "tractometric", "neuromodulatory_projection",
                    "electromagnetic", "device_coupling"),
        processes=("local_excitation", "local_inhibition", "thalamocortical_coupling",
                   "neuromodulation", "transmitter_dynamics",
                   "lateral_cortical_propagation", "em_generation",
                   "device_coupling"),
        observations=("eeg", "polysomnography", "eye_tracking", "emg"),
        devices=(DeviceSpec("psg", "sensor_array", "eeg_cap", "observe",
                            n_elements=6,
                            note="clinical polysomnography montage: 2 eeg, 2 eog, "
                                 "chin emg"),),
        subject=SubjectSpec(),
        window=Window(n=4096, dt=0.01),   # 41 s: longer than a scoring epoch,
                                          # fast enough for the beta edge
        bands=(("neural", SLEEP), ("electromagnetic", SLEEP),
               ("extracellular", PHARMACOKINETIC)),
        budget=Budget(max_state_variables=80_000,
                      max_spectral_coefficients=20_000_000),),
    fit_sources=("sleep-edfx", "mass", "isruc", "physionet"),
    eval_sources=("shhs", "mesa-sleep", "ds003768", "ds005620"),
    teachers=("neural-mass-model-families", "eegpt", "labram", "cbramod"),
    negative_controls=("bwin-000-synthetic-scaffold", "tuh-eeg"),
    constrained=(
        "slow-oscillation frequency and amplitude, and their change across the "
        "night, which thousands of scored hours determine precisely",
        "spindle density, duration and frequency, from `mass`'s expert spindle "
        "annotations -- the only place in this corpus where the events themselves "
        "are labelled",
        "the stage-transition structure as a slow process, from the staging labels",),
    prior_dominated=(
        "spatial structure of anything.  a two-channel montage constrains no map, "
        "and the 8 mm cortical grid exists to carry the topology, not because it "
        "is resolved",
        "the thalamic mechanism.  the reticular-relay loop is materialized at 2 mm "
        "and every parameter in it comes from animal recordings at a literature "
        "prior; scalp spindles are consistent with it and do not identify it",
        "the neuromodulator trajectories.  acetylcholine and noradrenaline drive "
        "the stage transitions in the model and nothing here measures them in "
        "humans",),
    notes="`ds003768` (simultaneous eeg-fmri during sleep) is in eval rather than "
          "fit deliberately: it is the one source that could falsify the model's "
          "claim about which structures change across stages, and using it to fit "
          "would spend it.",))


ANESTHESIA = register(NamedModel(
    id="anesthesia",
    doc="""loss and recovery of responsiveness under a sedative.

    structurally the same materialization as `sleep_dynamics` -- coarse cortex,
    fine thalamus, a narrow band -- with one important difference: the drug
    concentration is an *intervention*, titrated by the experimenter, so this is
    one of the few models in the library with a real dose-response manipulation
    rather than an observational contrast.

    **why r(q) stays coarse anyway.**  the manipulation is systemic.  propofol
    reaches every gaba-a receptor in the brain, and the observation is a scalp
    montage.  what is identifiable is a small number of loop gains and their
    dependence on concentration.  a fine map would be the receptor-density atlas
    projected through a fitted scalar, which is a legitimate thing to compute and
    an illegitimate thing to present as a measurement -- so the atlas dependence
    is stated under prior-dominated instead.

    **B goes lower than sleep.**  0.1-45 Hz.  burst suppression at deep sedation is
    an alternation on a timescale of seconds, so the low edge has to admit it; the
    high edge covers the alpha anteriorisation and the gamma loss that are the
    other two robust markers.

    **the alpha anteriorisation is the model's real test.**  propofol moves alpha
    power from occipital to frontal cortex, which is a *spatial* prediction that a
    thalamocortical loop-gain model makes and that a global-suppression model does
    not.  `chennu-propofol-sedation-eeg` has the graded sedation levels to check
    it.""",
    request=MaterializationRequest(
        name="anesthesia",
        targets=(sel("neural.exc.activity", "neural.inh.activity", band=SEDATION),
                 sel("neural.inh.gaba_a", "neural.exc.nmda", band=SEDATION),
                 sel("extracellular.gaba", band=PHARMACOKINETIC)),
        regions=(("thalamus", Anat("thalamic_nuclei", "reticular")),
                 ("frontal", Anat("cortical_areas", "prefrontal")),
                 ("cortex", OnSupport("cortical_surface"))),
        resolution=res(
            rule(Anat("thalamic_nuclei", "reticular"), 2.0, SEDATION),
            rule(Anat("cortical_areas", "prefrontal"), 6.0, SEDATION),
            rule(OnSupport("cortical_surface"), 8.0, SEDATION),
            default_mm=10.0, default_band=SEDATION),
        fields=("neural", "extracellular", "electromagnetic", "device"),
        anatomy=("thalamic_nuclei", "cortical_areas", "brainstem_nuclei"),
        topologies=("cortical_surface", "tractometric", "neuromodulatory_projection",
                    "electromagnetic", "device_coupling"),
        processes=("local_excitation", "local_inhibition", "thalamocortical_coupling",
                   "transmitter_dynamics", "neuromodulation",
                   "lateral_cortical_propagation", "em_generation",
                   "device_coupling"),
        observations=("eeg", "behaviour",
                      "drug_concentration"),
        interventions=("anaesthetic",),
        devices=(DeviceSpec("eeg", "sensor_array", "eeg_cap", "observe",
                            n_elements=91),),
        subject=SubjectSpec(),
        window=Window(n=2048, dt=0.01),
        bands=(("neural", SEDATION), ("electromagnetic", SEDATION),
               ("extracellular", PHARMACOKINETIC)),
        budget=Budget(max_state_variables=80_000),),
    fit_sources=("chennu-propofol-sedation-eeg", "neurotycho", "hansen-receptors"),
    eval_sources=("ds005620", "prime-de", "tdbrain", "isruc"),
    teachers=("neural-mass-model-families", "the-virtual-brain", "labram"),
    negative_controls=("bwin-000-synthetic-scaffold", "tuh-eeg",
                       "example-published-study"),
    constrained=(
        "the dose-response curve of frontal alpha power and of the "
        "occipital-to-frontal shift, which the graded sedation levels determine "
        "directly",
        "the loss of high-frequency power with increasing concentration",
        "a small number of effective loop gains as functions of drug level -- "
        "arguably the only genuinely causal parameters in this whole module",),
    prior_dominated=(
        "the receptor-density map.  `hansen-receptors` supplies gaba-a density as "
        "an atlas and it does not move; every regional difference the model "
        "reports is that atlas times a fitted scalar and must be described that way",
        "the mechanism of loss of consciousness -- thalamic gating, cortical "
        "disconnection, or dynamic-repertoire collapse.  all three fit these "
        "spectra",
        "the species and drug transfer: `neurotycho` is macaque and its "
        "anaesthetics differ; human generalisation is prior",),))


PHARMACO = register(NamedModel(
    id="pharmaco",
    doc="""drug action on neuromodulator fields and their downstream effect.

    the most prior-dominated model in the library, and it is declared anyway,
    because §5 is explicit that a process with no defensible implementation still
    gets declared with weak parameters rather than being omitted.  the same applies
    to a materialization.

    **the shape of the problem.**  a drug changes receptor occupancy, occupancy
    changes an effective gain in `neuromodulation`, and the gain changes population
    dynamics.  the first step is well characterised pharmacologically; the third is
    observed at the scalp or in bold; the middle step is the one nobody measures in
    a living human brain, and it is where all the free parameters are.

    **r(q) follows the projection systems, not the cortex.**  dopaminergic,
    serotonergic, cholinergic and noradrenergic projections originate in small
    brainstem and basal-forebrain nuclei -- the locus coeruleus is about 2 mm
    across -- and terminate diffusely.  so 1 mm in the source nuclei, where the
    structure is genuinely small and heterogeneous, and 8 mm in cortex, where the
    innervation is diffuse and a fine grid would be inventing detail.  this is the
    inverse of the usual arrangement and follows directly from where coarse-graining
    actually fails.

    **B is two bands, far apart.**  the concentration field changes over minutes
    (0-0.02 Hz); the population dynamics it modulates are lfp-band.  they are
    coupled multiplicatively, which is a nonlinearity, so the spectral form does not
    close and the coupling has to be evaluated in the time domain -- worth noting
    because it is the cost §1 warns about.""",
    request=MaterializationRequest(
        name="pharmaco",
        targets=(sel("extracellular.dopamine", "extracellular.serotonin",
                     "extracellular.acetylcholine", "extracellular.noradrenaline",
                     band=PHARMACOKINETIC),
                 sel("neural.exc.activity", "neural.inh.activity",
                     band=MODULATED)),
        regions=(("source_nuclei", Anat("brainstem_nuclei", "locus_coeruleus")),
                 ("striatum", Anat("bg_territories", "sensorimotor")),
                 ("cortex", OnSupport("cortical_surface"))),
        resolution=res(
            rule(Anat("brainstem_nuclei", "locus_coeruleus"), 1.0, PHARMACOKINETIC),
            rule(Anat("brainstem_nuclei", "raphe"), 1.0, PHARMACOKINETIC),
            rule(Anat("bg_territories", "sensorimotor"), 2.0, PHARMACOKINETIC),
            rule(OnSupport("cortical_surface"), 8.0, MODULATED),
            default_mm=10.0, default_band=MODULATED),
        fields=("extracellular", "neural", "blood", "metabolic"),
        anatomy=("brainstem_nuclei", "bg_territories", "striosome_matrix",
                 "cortical_areas", "hypothalamic_nuclei"),
        topologies=("neuromodulatory_projection", "interstitial", "local",
                    "vascular", "tractometric"),
        processes=("neuromodulation", "transmitter_dynamics", "interstitial_transport",
                   "local_excitation", "local_inhibition", "vascular_flow"),
        observations=("eeg", "bold", "pet",
                      "drug_concentration"),
        interventions=("pharmacological",),
        devices=(DeviceSpec("eeg", "sensor_array", "eeg_cap", "observe",
                            n_elements=64),),
        subject=SubjectSpec(),
        window=Window(n=8192, dt=2e-3),   # 16 s of fast dynamics; the minutes-long
                                          # concentration trajectory is carried across
                                          # windows, not inside one
        bands=(("extracellular", PHARMACOKINETIC), ("neural", MODULATED),
               ("blood", HEMODYNAMIC)),
        budget=Budget(max_state_variables=200_000),),
    fit_sources=("chennu-propofol-sedation-eeg", "hansen-receptors",
                 "allen-human-brain-atlas", "monash-fdg-pet-fmri"),
    eval_sources=("ppmi", "tdbrain", "psychencode", "neuroquery"),
    teachers=("neural-mass-model-families",),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study",
                       "enigma"),
    constrained=(
        "the receptor-density maps themselves, which `hansen-receptors` and "
        "`allen-human-brain-atlas` supply as measurements with real uncertainty -- "
        "these are the constrained part and they are structural, not dynamic",
        "the scalp spectral change under propofol, the one drug in this corpus "
        "with a graded human dose-response series",
        "regional glucose metabolism as a slow correlate, from the fdg-pet data",),
    prior_dominated=(
        "essentially the entire dynamic model.  the occupancy-to-gain map is a "
        "speculative prior for every transmitter except gaba-a under propofol, and "
        "predictions for any other drug are the prior with a fitted scale",
        "the extracellular concentration field.  volume transmission over a "
        "tortuous interstitial space is materialized and never measured in vivo",
        "regional selectivity of effect, which is receptor atlas times scalar and "
        "must not be reported as though the regional pattern were fitted",),))


VIRTUAL_LESION = register(NamedModel(
    id="virtual_lesion",
    doc="""what changes when a node or a tract is removed.

    a counterfactual, and the model whose epistemic status needs the loudest
    warning in the library: there is no interventional human dataset here.  every
    fit source is observational -- case-control effect sizes, patient cohorts,
    connectome variation -- and a counterfactual computed from observational data
    is a prediction of the model's causal structure, not a measurement of it.  the
    materialization is honest only if it says so, which is what
    `prior_dominated` below is for.

    **r(q) is parcel-scale on purpose.**  lesions in the sources are defined at the
    scale of a resection, an infarct territory or a dysplasia -- centimetres.
    materializing at 2 mm would create a lesion boundary the data cannot place.
    5 mm on the sheet, 3 mm around the lesion mask where the boundary gradient is
    the only place resolution changes the answer, 10 mm elsewhere.

    **B is the resting band.**  the readout is a change in functional connectivity
    and in spectral power, both slow.  0.01-0.1 Hz for the hemodynamic readout and
    a wider lfp band when the readout is electrophysiological.

    **the one thing that would change this model's status** is intracranial
    stimulation data with simultaneous recording -- `ram-intracranial` has direct
    stimulation records -- which is a real intervention on a real network.  it sits
    in eval rather than fit because it is the falsification set.""",
    request=MaterializationRequest(
        name="virtual-lesion",
        targets=(sel("neural.exc.activity", "neural.inh.activity", band=RESTING),
                 sel("structural.axonal_density", "structural.myelination",
                     band=Band(0.0, 0.001))),
        regions=(("lesion", Anat("cortical_areas", "structural_mri")),
                 ("cortex", OnSupport("cortical_surface")),
                 ("tracts", OnSupport("tissue"))),
        resolution=res(
            rule(Anat("cortical_areas", "structural_mri"), 3.0, RESTING),
            rule(OnSupport("cortical_surface"), 5.0, RESTING),
            default_mm=10.0, default_band=RESTING),
        fields=("neural", "structural", "blood", "electromagnetic"),
        anatomy=("cortical_areas", "thalamic_nuclei", "vascular_territories"),
        topologies=("tractometric", "cortical_surface", "local"),
        processes=("tract_propagation", "lateral_cortical_propagation",
                   "local_excitation", "local_inhibition", "plasticity",
                   "neurovascular_coupling"),
        observations=("bold", "eeg", "structural_mri"),
        interventions=("graph_ablation",),
        devices=(),
        subject=SubjectSpec(),
        window=Window(n=512, dt=1.0),
        bands=(("neural", RESTING), ("blood", HEMODYNAMIC),
               ("structural", Band(0.0, 0.001))),
        budget=Budget(max_state_variables=150_000),),
    fit_sources=("enigma", "hansen-many-networks", "braingraph-hcp-connectomes",
                 "netneuro-lausanne-sc", "hcp-functional-connectivity", "meld"),
    eval_sources=("ram-intracranial", "clinical-ieeg-archives", "abide",
                  "corr-reliability"),
    teachers=("the-virtual-brain", "neurolib", "neural-mass-model-families",
              "scwbd-001-004-predecessor-models"),
    negative_controls=("enigma", "example-published-study",
                       "scwbd-001-004-predecessor-models",
                       "bwin-000-synthetic-scaffold"),
    constrained=(
        "the observational association between structural disconnection and "
        "functional-connectivity change, which the connectome and enigma sources "
        "determine across large samples",
        "the group-level topology of structural connectivity itself",
        "direct-stimulation evoked responses at neighbouring contacts, where "
        "`ram-intracranial` provides them -- the only interventional evidence here",),
    prior_dominated=(
        "the causal claim.  a virtual lesion's predicted effect is a property of "
        "the model's process graph, and the corpus contains almost no intervention "
        "that could move it.  this must never be reported with the confidence of "
        "an observed contrast",
        "compensation and plasticity after the lesion, which the `plasticity` "
        "process is materialized to represent and which no source here times",
        "individual variability in effect, which is clinically the whole question",),
    notes="`scwbd-001-004-predecessor-models` appears as both teacher and negative "
          "control by design: the predecessor's demoted assumptions are exactly "
          "what a successor must not silently reproduce.",))


RESTING_STATE_FC = register(NamedModel(
    id="resting_state_fc",
    doc="""the cheapest useful materialization: coarse in space, coarse in band.

    worth a named model because it is the baseline against which every finer model
    in the library argues.  §1's criterion is not a slogan here: `coarsened()` on a
    fine request produces something with this shape, and if the fine materialization
    does not differ from it in a way the data can see, the fine one was waste and
    should be retired.  this model is the thing it gets compared to.

    **r(q).**  5 mm on the sheet, 8 mm elsewhere.  resting bold connectivity is a
    few hundred spatial modes; the parcellation atlases in the eval list encode that
    directly.  the model materializes a grid rather than a parcellation so that a
    finer model can be coarsened onto it, but the honest rank is the parcel count.

    **B.**  0.01-0.1 Hz.  below 0.01 Hz is scanner and physiological drift; above
    0.1 Hz a 1-2 s tr aliases cardiac and respiratory signal into the estimate, and
    the resulting 'connectivity' is a breathing pattern.  the band is not a
    convenience, it is the interval over which the observation has any precision.

    **cost.**  a few tens of thousands of sites at five coefficients each.  it is
    the only model in the library that is cheap on both axes at once, which is
    exactly why it can be run thousands of times.""",
    request=MaterializationRequest(
        name="resting-state-fc",
        targets=(sel("neural.exc.activity", band=RESTING),
                 sel("blood.deoxyhemoglobin", band=RESTING)),
        regions=(("cortex", OnSupport("cortical_surface")),
                 ("subcortex", OnSupport("tissue"))),
        resolution=res(
            rule(OnSupport("cortical_surface"), 5.0, RESTING),
            rule(OnSupport("tissue"), 8.0, RESTING),
            default_mm=10.0, default_band=RESTING),
        fields=("neural", "blood"),
        anatomy=("cortical_areas", "thalamic_nuclei", "cerebellar_lobules"),
        topologies=("cortical_surface", "tractometric", "vascular"),
        processes=("lateral_cortical_propagation", "tract_propagation",
                   "neurovascular_coupling", "bold_formation"),
        observations=("bold",),
        devices=(DeviceSpec("scanner", "scanner_element", "scanner", "observe"),),
        subject=SubjectSpec(),
        window=Window(n=512, dt=1.0),
        bands=(("neural", RESTING), ("blood", RESTING)),
        budget=Budget(max_state_variables=60_000,
                      max_spectral_coefficients=2_000_000),),
    fit_sources=("hcp-functional-connectivity", "hcp-young-adult",
                 "midnight-scan-club", "nki-rockland", "aomic"),
    eval_sources=("corr-reliability", "abide", "adhd-200", "myconnectome",
                  "schaefer2018", "yeo2011"),
    teachers=("brainlm", "brain-jepa", "brain-harmony"),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study",
                       "enigma"),
    constrained=(
        "the group-average connectivity matrix at parcel scale, which is one of "
        "the most reproducible measurements in human neuroimaging",
        "individual deviation from that average, where enough scan time exists -- "
        "`midnight-scan-club` and `myconnectome` are the sources that show how "
        "much time it takes",
        "the low-frequency spectral shape per parcel",),
    prior_dominated=(
        "everything about direction and mechanism.  correlation at 0.05 Hz "
        "orients nothing, and the directed topologies this model materializes "
        "carry their prior straight through",
        "sub-parcel structure.  the 5 mm grid is a representation of a few hundred "
        "modes and its apparent detail is interpolation",
        "the neural-versus-vascular attribution of any connectivity difference, "
        "which needs `hrf` or `asl_perfusion` materialized alongside to address",),))
