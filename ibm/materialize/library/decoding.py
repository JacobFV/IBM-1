"""materializations whose target is a boundary variable read back from brain state.

    M = materialize(R, r, B, F, A, T, P)

decoding is not a separate primitive.  every model here is the same forward
declaration read in the direction that happens to be useful: `afferent_propagation`
and `transduction` run backwards from cortex to the stimulus, or
`efferent_propagation` and `effector_activation` run forwards from cortex to a
device output.  nothing in this module declares a process, and nothing may.

what distinguishes these seven from the electrophysiology group is the *rank of
what is actually being asked for*.  a forward model has to be right about a field;
a decoder only has to be right about a low-dimensional function of it -- a word, a
category, a cursor velocity, an envelope.  that changes r(q) in a specific and
uncomfortable direction: for scalp decoding, fine spatial materialization buys
nothing, because the decodable subspace is bounded by the sensor count long before
it is bounded by the source grid.  the honest r(q) for `eeg_to_image` is coarse and
the honest thing to say about it is that its output is a category posterior, not a
picture of what someone saw.

the invasive models invert that completely.  `invasive_bci` is the §1 worked
example: 50 um within 2 mm of the array, 500 um out to 2 cm, 2 mm through
connected structures and 10 mm everywhere else, with a band five decades wide.
it is affordable because the fine region is a few cubic millimetres, and it is
*necessary* because a 96-channel intracortical array resolves individual units
whose tuning does not survive averaging over a cortical column.
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
    HIGH_GAMMA,
    LFP,
    Near,
    OnSupport,
    sel,)

#: the scalp ceiling, repeated from `electrophysiology` for the same reason: above
#: it a scalp decoder is decoding jaw muscle.
SCALP = Band(0.5, 100.0)
#: meg's ceiling.
DEWAR = Band(0.5, 200.0)
#: intracranial potentials.
INTRACRANIAL = Band(0.5, 300.0)
#: threshold crossings and spike-band power.
SPIKE_BAND = Band(300.0, 5000.0)
#: acoustic envelope tracking.  cortical entrainment to speech lives in delta and
#: theta and is essentially gone by 12 Hz; a wider band adds variance, not signal.
ENVELOPE = Band(0.5, 12.0)
#: the band of a binned neural decoder's own output -- 20 ms bins, so nothing
#: above 25 Hz means anything even though the observation reaches 5 kHz.
DECODER_OUTPUT = Band(0.0, 25.0)


EEG_TO_IMAGE = register(NamedModel(
    id="eeg_to_image",
    doc="""visual category and low-level image structure from scalp eeg.

    named honestly rather than aspirationally.  what a 64-channel scalp recording
    constrains about a seen image is a low-dimensional posterior over category and
    a handful of low-level statistics (luminance, spatial frequency, animacy),
    reached through `transduction`, `afferent_propagation` and the same
    `em_generation`/`em_coupling` chain as `eeg_forward`.  it does not constrain a
    pixel array, and a materialization that produced one would be reporting the
    generative prior of whatever image model was attached.

    **why r(q) is coarse.**  6 mm over occipital and ventral-temporal cortex, 10 mm
    elsewhere.  the decodable subspace is capped by the sensor count; the skull has
    already low-passed the source distribution over centimetres; and the discriminable
    quantity -- an evoked category signature at 100-300 ms -- is a large-scale
    topographic pattern, not a retinotopic one.  every criterion in §1 says
    coarse-graining commutes here, so it is coarse.  the one place it is not is
    early visual cortex, where retinotopic organisation means neighbouring 6 mm
    cells are genuinely heterogeneous in what they respond to; that rule sits at
    3 mm and is the only refinement the model buys.

    **B.**  0.5-100 Hz.  the discriminative information in these datasets is
    concentrated below 30 Hz and in the evoked, phase-locked part of the spectrum
    -- an anisotropic covariance on the (cos, sin) plane in §1's terms -- which is
    exactly why single-trial decoding works at all despite the ongoing alpha being
    an order of magnitude larger.""",
    request=MaterializationRequest(
        name="eeg-to-image",
        targets=(sel("transduction.photoreceptor", band=SCALP),
                 sel("neural.exc.activity", region=Anat("cortical_areas", "occipital"),
                     band=SCALP)),
        regions=(("visual", Anat("cortical_areas", "occipital")),
                 ("ventral", Anat("cortical_areas", "ventral_temporal")),
                 ("cortex", OnSupport("cortical_surface"))),
        resolution=res(
            rule(Anat("cortical_areas", "v1"), 3.0, SCALP),
            rule(Anat("cortical_areas", "occipital"), 6.0, SCALP),
            rule(OnSupport("cortical_surface"), 10.0, SCALP),
            default_mm=12.0, default_band=SCALP),
        fields=("neural", "electromagnetic", "transduction", "device"),
        anatomy=("cortical_areas",),
        topologies=("cortical_surface", "afferent_pathway", "electromagnetic",
                    "device_coupling"),
        processes=("transduction", "afferent_propagation", "em_generation",
                   "em_coupling", "device_coupling"),
        observations=("eeg",),
        devices=(DeviceSpec("eeg", "sensor_array", "eeg_cap", "observe", n_elements=64),
                 DeviceSpec("screen", "display", "display", "stimulate",
                            note="the stimulus is device state, not an input")),
        subject=SubjectSpec(),
        window=Window(n=1024, dt=2e-3),   # 2 s: a 0.5 Hz low edge needs it
        bands=(("neural", SCALP), ("transduction", SCALP)),
        budget=Budget(max_state_variables=80_000),),
    fit_sources=("things-eeg2", "things-eeg", "erp-core"),
    eval_sources=("things-eeg2", "bold5000", "moabb", "braindecode"),
    teachers=("multiscale-visual-encoders", "vjepa2", "video-jepa-family",
              "labram", "eegpt"),
    negative_controls=("salicon", "mit-tuebingen-saliency-benchmark",
                       "bwin-000-synthetic-scaffold"),
    constrained=(
        "the timing and topography of the category-discriminative evoked response, "
        "which `things-eeg2`'s ten thousand image presentations per subject pin well",
        "low-level image statistics (luminance, contrast, spatial frequency), which "
        "dominate the first 150 ms and are the easiest thing on this list",
        "the subject-specific sensor mixing, over-determined by the number of trials",),
    prior_dominated=(
        "retinotopic position of the source.  a 3 mm v1 grid exists in this "
        "materialization and nothing in a scalp recording distinguishes its cells",
        "everything the attached visual teacher supplies beyond category: image "
        "detail in an output here is the teacher's generative prior, entering at "
        "calibrated precision and never at unit precision, and must be labelled as "
        "reconstruction rather than measurement",
        "ventral-stream hierarchy: the model materializes several areas and the "
        "scalp cannot separate their contributions",),
    notes="`salicon` and `mit-tuebingen-saliency-benchmark` are the negative "
          "controls that matter: a decoder that is really reading image salience "
          "rather than the subject's brain will score above chance on them and must "
          "not.  eval on `things-eeg2` is on its held-out split only.",))


MEG_TO_TEXT = register(NamedModel(
    id="meg_to_text",
    doc="""linguistic structure from meg during continuous listening or reading.

    the same caution as `eeg_to_image` with a different failure mode.  meg during
    naturalistic speech constrains *when* words happen, *which* broad lexical and
    acoustic properties they have, and how the superior temporal response scales
    with surprisal.  it does not constrain a word-by-word transcript, and a
    materialization that emitted one would be emitting a language model's
    continuation conditioned on very little.

    **why r(q) refines over temporal cortex only.**  meg is unusually good at
    tangential sources on the superior temporal plane, which is where the auditory
    evoked field and the speech-tracking response are generated, and where
    neighbouring patches have genuinely different latencies -- the temporal
    receptive window grows from tens of milliseconds in heschl's gyrus to seconds
    in anterior temporal and inferior frontal cortex.  that latency heterogeneity
    inside a coarse cell is a real failure of commutation, so 3 mm there.
    elsewhere 8 mm, because nothing in the objective distinguishes finer.

    **B.**  0.5-200 Hz nominally, but the decodable content is concentrated in
    delta and theta (phrase and syllable rate) plus the evoked response.  the
    high end is retained because meg can carry it, not because the decoder uses it.

    **on the teachers.**  `llama-3.2` and `language-models-multiscale` supply word
    surprisal and multiscale linguistic structure that the datasets do not ship.
    this is infill, at precision calibrated from each model's reported accuracy on
    that stream, low-rank across positions.  fitting a brain model to a language
    model's surprisal at unit precision would make the brain model assert the
    language model's idiosyncrasies as neural facts.""",
    request=MaterializationRequest(
        name="meg-to-text",
        targets=(sel("neural.exc.activity",
                     region=Anat("cortical_areas", "superior_temporal"), band=DEWAR),
                 sel("transduction.hair_cell", band=ENVELOPE)),
        regions=(("auditory", Anat("cortical_areas", "superior_temporal")),
                 ("language", Anat("cortical_areas", "inferior_frontal")),
                 ("cortex", OnSupport("cortical_surface"))),
        resolution=res(
            rule(Anat("cortical_areas", "superior_temporal"), 3.0, DEWAR),
            rule(Anat("cortical_areas", "inferior_frontal"), 4.0, DEWAR),
            rule(OnSupport("cortical_surface"), 8.0, SCALP),
            default_mm=12.0, default_band=SCALP),
        fields=("neural", "electromagnetic", "transduction", "device"),
        anatomy=("cortical_areas",),
        topologies=("cortical_surface", "afferent_pathway", "tractometric",
                    "electromagnetic", "device_coupling"),
        processes=("transduction", "afferent_propagation", "em_generation",
                   "em_coupling", "tract_propagation", "device_coupling"),
        observations=("meg", "meg"),
        devices=(DeviceSpec("meg", "sensor_array", "meg_head", "observe",
                            n_elements=306),
                 DeviceSpec("speaker", "display", "display", "stimulate")),
        subject=SubjectSpec(),
        window=Window(n=2048, dt=2e-3),
        bands=(("neural", DEWAR), ("transduction", ENVELOPE)),
        budget=Budget(max_state_variables=200_000),),
    fit_sources=("meg-masc", "libribrain", "armeni-audiobook-meg", "meg-scans",
                 "mous"),
    eval_sources=("libribrain", "kymata-soto", "studyforrest-meg-ds003633",
                  "zuco"),
    teachers=("llama-3.2", "language-models-multiscale", "wav2vec-bert-2.0",
              "audio-spectrotemporal-encoders", "phonetic-acoustic-models",
              "meg-gpt", "tribe"),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "word-onset timing and the latency of the auditory evoked field, which "
        "forced alignments in `meg-masc` and `libribrain` pin to a few milliseconds",
        "the acoustic-to-cortical transfer function in the 1-8 Hz range, "
        "over-determined by hours of continuous speech per subject",
        "the surprisal-amplitude slope in superior temporal cortex, to within the "
        "precision the language-model teacher is allowed to contribute",),
    prior_dominated=(
        "word identity.  nothing here supports a lexical readout; a transcript "
        "produced by this model is the language teacher's continuation and must "
        "be reported as such",
        "the frontal end of the language network -- `meg-masc` and `libribrain` "
        "have few subjects, and inferior frontal sources are deep and weak in meg",
        "syntactic structure beyond what surprisal already encodes; the "
        "annotations exist in `zuco` and `mous` but the brain-side estimand is weak",),))


SPEECH_ENVELOPE = register(NamedModel(
    id="speech_envelope",
    doc="""cortical tracking of the acoustic envelope.

    the narrowest band in the module and the best-constrained model in it.  cortical
    entrainment to the speech envelope is a delta/theta phenomenon: it peaks around
    the syllable rate of 4-5 Hz, is strong down to the phrase rate near 1 Hz, and
    is essentially absent above 12 Hz.  materializing 0.5-12 Hz is not a budget
    compromise, it is the statement that the observation carries no information
    outside that interval -- and because marginalizing a gaussian to a band is
    exact, restricting to it costs nothing in accuracy while costing a factor of
    ten in coefficients.

    **why r(q) is coarse -- 5 mm.**  envelope tracking is a large-scale
    superior-temporal response.  the temporal-receptive-window gradient that
    justified 3 mm in `meg_to_text` is a latency effect and this model does not
    resolve latency finely enough to care.  the auditory core gets 3 mm because
    tonotopy is a real gradient over millimetres; everything else is smooth.

    **the invasive variant is the reason for the high-gamma band override.**  ecog
    high gamma tracks the envelope far better than any scalp measure, so when a
    contact array is present the model carries 70-200 Hz on
    `neural.exc.activity` while the target stays in the envelope band.  those are
    different variables with different bands and the request interface says so
    rather than compromising on one.

    **negative control.**  `broderick-natural-speech-eeg` ships time-reversed
    speech, which has the same envelope statistics and no linguistic content.  a
    tracking model must fit it as well as forward speech; a model that fits forward
    speech better is decoding comprehension it has no right to.""",
    request=MaterializationRequest(
        name="speech-envelope",
        targets=(sel("transduction.hair_cell", band=ENVELOPE),
                 sel("neural.exc.activity",
                     region=Anat("cortical_areas", "superior_temporal"),
                     band=ENVELOPE)),
        regions=(("auditory_core", Anat("cortical_areas", "heschl")),
                 ("belt", Anat("cortical_areas", "superior_temporal"))),
        resolution=res(
            rule(Anat("cortical_areas", "heschl"), 3.0, ENVELOPE),
            rule(Anat("cortical_areas", "superior_temporal"), 5.0, ENVELOPE),
            rule(OnSupport("cortical_surface"), 10.0, ENVELOPE),
            default_mm=12.0, default_band=ENVELOPE),
        fields=("neural", "electromagnetic", "transduction", "device"),
        anatomy=("cortical_areas",),
        topologies=("cortical_surface", "afferent_pathway", "electromagnetic",
                    "device_coupling"),
        processes=("transduction", "afferent_propagation", "em_generation",
                   "em_coupling", "device_coupling"),
        observations=("eeg", "meg",
                      "ecog"),
        devices=(DeviceSpec("eeg", "sensor_array", "eeg_cap", "observe",
                            n_elements=128),
                 DeviceSpec("speaker", "display", "display", "stimulate")),
        subject=SubjectSpec(),
        window=Window(n=8192, dt=1e-3),   # 8 s at 1 kHz: long enough for a 0.5 Hz
                                          # floor, fast enough for the ecog high-gamma
                                          # override below
        bands=(("transduction", ENVELOPE),
               ("neural.exc.activity", HIGH_GAMMA),
               ("electromagnetic", ENVELOPE)),
        budget=Budget(max_state_variables=60_000,
                      max_spectral_coefficients=20_000_000),),
    fit_sources=("broderick-natural-speech-eeg", "meg-masc", "libribrain",
                 "ds005574", "chang-lab-ecog-cv-syllables"),
    eval_sources=("armeni-audiobook-meg", "ds006234", "ds007738",
                  "single-word-production-dutch-ieeg"),
    teachers=("audio-spectrotemporal-encoders", "wav2vec-bert-2.0",
              "phonetic-acoustic-models"),
    negative_controls=("broderick-natural-speech-eeg", "bwin-000-synthetic-scaffold"),
    constrained=(
        "the envelope-to-cortex transfer function in 1-8 Hz, including its latency, "
        "which nineteen subjects of continuous speech determine tightly",
        "the high-gamma envelope response in `ds005574` and the chang-lab ecog, "
        "which is close to a direct measurement of the target variable",
        "the attentional gain difference between attended and ignored streams, "
        "from the cocktail-party condition",),
    prior_dominated=(
        "tonotopic organisation within auditory core: the 3 mm rule materializes "
        "it and no scalp or grid measurement here separates adjacent frequency "
        "bands",
        "the split between `transduction.hair_cell` and subcortical relays -- the "
        "whole afferent chain below cortex is a single fitted delay plus a prior",
        "anything outside superior temporal cortex; frontal envelope tracking is "
        "reported in the literature but not resolvable in these data",),
    notes="the reverse-speech condition of `broderick-natural-speech-eeg` appears "
          "in both fit and negative-control roles by design: forward speech trains, "
          "reversed speech must produce an indistinguishable envelope fit.",))


INVASIVE_BCI = register(NamedModel(
    id="invasive_bci",
    doc="""intracortical array to effector command.  §1's worked example, verbatim.

    spatially tiny, temporally very wide -- the opposite corner of the budget from
    `bold_forward`, and the clearest case in the library where fine resolution is
    not optional.

    **why 50 um within 2 mm.**  a utah array's contacts sit 1.5 mm into cortex on a
    400 um pitch and resolve individual units.  the tuning of one unit is not the
    average of its neighbours' tuning: directional tuning curves rotate over tens of
    microns and a cortical column contains cells with opposing preferred directions.
    averaging over a millimetre returns something close to zero and the decoder built
    on it does not work.  this is heterogeneity inside a coarse cell in the plainest
    possible form, and 50 um is where the tuning is locally flat.

    **why 500 um out to 2 cm, and 2 mm through connected structures.**  the local
    circuit that the array samples is driven by premotor and parietal inputs that
    arrive over the tractometric topology; those edges do not survive coarsening to
    10 mm, so the connected structures get a rule of their own even though nothing
    there is directly observed.  10 mm everywhere else, where the state is smooth
    and the coupling linear.

    **why B is five decades wide.**  the observation is threshold crossings and
    spike-band power, 300 Hz to 5 kHz, but the *decoded* quantity is a binned rate
    at 20 ms and the effector command lives below 25 Hz.  both bands are carried,
    on different components, because collapsing to one would either alias the spikes
    or throw away the command.  the whole thing is affordable purely because the
    fine region is a few cubic millimetres: sites times coefficients is the cost,
    and this model wins on the first factor while losing on the second.""",
    request=MaterializationRequest(
        name="invasive-bci",
        targets=(sel("effector.drive", "effector.activation", band=DECODER_OUTPUT),
                 sel("neural.exc.activity", region=Near("array", 2.0), band=SPIKE_BAND),
                 sel("device.contact_potential", region=Near("array", 5.0),
                     band=Band(0.5, 5000.0))),
        regions=(("array_neighbourhood", Near("array", 2.0)),
                 ("local_circuit", Near("array", 20.0)),
                 ("connected", Anat("cortical_areas", "precentral"))),
        resolution=res(
            rule(Near("array", 2.0), 0.05, Band(0.5, 5000.0)),
            rule(Near("array", 20.0), 0.5, Band(0.5, 5000.0)),
            rule(Anat("cortical_areas", "precentral"), 2.0, LFP),
            rule(Anat("cortical_areas", "posterior_parietal"), 2.0, LFP),
            default_mm=10.0, default_band=Band(0.0, 100.0)),
        fields=("neural", "electromagnetic", "device", "effector", "extracellular"),
        anatomy=("cortical_areas", "cortical_layers"),
        topologies=("local", "microcircuit", "laminar", "tractometric",
                    "efferent_pathway", "device_coupling"),
        processes=("local_excitation", "local_inhibition", "em_generation",
                   "em_coupling", "device_coupling", "tract_propagation",
                   "efferent_propagation", "effector_activation"),
        observations=("intracortical_spikes", "intracortical_spikes",
                      "lfp"),
        interventions=("task_cue", "auditory_stimulus", "visual_stimulus",),
        devices=(DeviceSpec("array", "implanted_array", "electrode_grid", "both",
                            n_elements=192,
                            note="two 96-channel intracortical arrays"),),
        subject=SubjectSpec(id="participant", template=None,
                            note="single-participant models; there is no population "
                                 "here and never will be"),
        window=Window(n=32768, dt=1e-4),  # 3.3 s at 10 kHz: spikes at the top,
                                          # a 0.5 Hz floor at the bottom
        bands=(("neural", Band(0.5, 5000.0)), ("effector", DECODER_OUTPUT),
               ("device", Band(0.5, 5000.0))),
        budget=Budget(max_state_variables=1_500_000,
                      max_spectral_coefficients=600_000_000),),
    fit_sources=("willett-speech-neuroprosthesis", "willett-handwriting-bci",
                 "invasive-human-array-bci", "ram-intracranial"),
    eval_sources=("willett-speech-neuroprosthesis", "invasive-human-array-bci",
                  "braindecode", "riemannian-baselines"),
    teachers=("brainbert", "biot"),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "per-unit tuning within the 4x4 mm array footprint -- directly observed, "
        "thousands of trials, the most tightly constrained state in the library",
        "the day-to-day drift of that tuning, which the diagnostic and tuning "
        "blocks in `willett-speech-neuroprosthesis` were collected to measure",
        "the mapping from binned population rate to attempted-movement kinematics",),
    prior_dominated=(
        "everything more than about 2 cm from the array.  the 2 mm rule over "
        "premotor and parietal cortex materializes state that no contact touches; "
        "its posterior is its prior and predictions that depend on it must say so",
        "the identity of the recorded cell types.  threshold crossings do not "
        "distinguish `neural.exc.activity` from `neural.pv.activity`, and the "
        "split stays at the cytoarchitectural prior",
        "generalisation across participants.  n=1 per array, and the tuning is "
        "individual; a population claim from this model is not evidence",),))


SPIKE_DECODE = register(NamedModel(
    id="spike_decode",
    doc="""stimulus and behaviour from sorted units and multi-unit activity.

    the same corner of the budget as `invasive_bci` but with the target on the
    sensory rather than the effector side, and with the fine region defined by a
    probe track rather than an array footprint.

    **why the band is the spike band and only the spike band for the observation.**
    a threshold crossing is a 1 ms event; representing it below 300 Hz is
    representing something else.  the model carries 300-5000 Hz on the observed
    variable and 0-25 Hz on the decoded one, and the reason those two numbers can
    coexist in one materialization is that bands are per-component, not per-model.

    **why 50 um along the shank and 200 um around it.**  a neuropixels shank has
    contacts every 20 um and the same unit appears on several of them; the spatial
    footprint of a spike is 50-100 um.  materializing coarser than that makes unit
    identity itself ill-defined.  laterally the requirement relaxes fast, because
    two shanks 500 um apart share almost no units.

    **the honest limitation.**  every fit source here is rodent.  the model is a
    good model of mouse visual and sensorimotor cortex and a prior-plus-scaling
    assumption about human, which is why the human transfer sits in
    `prior_dominated` rather than being quietly assumed.""",
    request=MaterializationRequest(
        name="spike-decode",
        targets=(sel("neural.exc.activity", "neural.inh.activity",
                     region=Near("probe", 1.0), band=SPIKE_BAND),
                 sel("transduction.photoreceptor", band=DECODER_OUTPUT)),
        regions=(("shank", Near("probe", 1.0)),
                 ("column", Near("probe", 10.0))),
        resolution=res(
            rule(Near("probe", 1.0), 0.05, Band(0.5, 5000.0)),
            rule(Near("probe", 10.0), 0.2, Band(0.5, 5000.0)),
            rule(Near("probe", 40.0), 1.0, LFP),
            default_mm=8.0, default_band=Band(0.0, 100.0)),
        fields=("neural", "electromagnetic", "device", "transduction"),
        anatomy=("cortical_layers", "cytoarchitecture", "thalamic_nuclei",
                 "hippocampal_subfields"),
        topologies=("local", "microcircuit", "laminar", "afferent_pathway",
                    "device_coupling"),
        processes=("local_excitation", "local_inhibition", "laminar_propagation",
                   "thalamocortical_coupling", "em_generation", "device_coupling",
                   "transduction", "afferent_propagation"),
        observations=("intracortical_spikes", "intracortical_spikes"),
        devices=(DeviceSpec("probe", "implanted_array", "electrode_grid", "observe",
                            n_elements=384),
                 DeviceSpec("screen", "display", "display", "stimulate")),
        subject=SubjectSpec(id="animal", template=None),
        window=Window(n=32768, dt=1e-4),  # as `invasive_bci`
        bands=(("neural", Band(0.5, 5000.0)), ("transduction", DECODER_OUTPUT)),
        budget=Budget(max_state_variables=800_000,
                      max_spectral_coefficients=600_000_000),),
    fit_sources=("steinmetz-neuropixels-2019", "allen-visual-coding-neuropixels",
                 "ibl-brain-wide-map"),
    eval_sources=("ibl-brain-wide-map", "allen-visual-coding-2p",
                  "single-neuron-ieeg-fmri-movie"),
    teachers=(),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "single-unit tuning to visual stimuli and to wheel and running speed, "
        "which is what these three datasets were built to measure",
        "the laminar and areal distribution of that tuning, from probe geometry "
        "plus the allen ccf registration the datasets ship",
        "the relationship between multi-unit rate and lfp gamma at the same site",),
    prior_dominated=(
        "human anything.  all three fit sources are mouse and the transfer to human "
        "cortex is a scaling prior, not a measurement",
        "cell-type identity beyond what optotagging labels in the allen data -- "
        "`neural.sst.activity` and `neural.vip.activity` exist and stay at prior",
        "long-range influences: the brain-wide map covers many areas but never "
        "simultaneously enough to constrain inter-areal coupling",),))


HANDWRITING_BCI = register(NamedModel(
    id="handwriting_bci",
    doc="""attempted handwriting from motor-cortex threshold crossings.

    a deliberately narrow sibling of `invasive_bci`, kept separate because its
    band structure is different in an instructive way and because it is bound to
    one dataset that is one participant.

    **the two bands.**  the observation is spike-band power and threshold crossings
    at 300-5000 Hz.  the decoded quantity is a pen-tip velocity, which for
    handwriting is essentially bandlimited below 10 Hz.  so the same materialization
    carries a five-decade band on the neural components and a one-decade band on
    the effector components, and the cost is dominated entirely by the former.
    writing both down rather than picking one is the point: a model that binned to
    20 ms up front would have thrown away the spike timing that makes the units
    separable, and a model that kept 5 kHz on the effector would be spending
    coefficients on a quantity with no content there.

    **r(q).**  identical in shape to `invasive_bci` -- 50 um in the array
    footprint, 500 um out to 2 cm, 2 mm in precentral cortex -- because it is the
    same instrument in the same place.  the two models share the r(q) rather than
    inventing one each, which is the library's whole discipline.

    **what it cannot do.**  n=1.  everything about this model is about one person's
    hand knob, and its parameters do not transfer.""",
    request=MaterializationRequest(
        name="handwriting-bci",
        targets=(sel("effector.drive", "effector.activation", band=Band(0.0, 10.0)),
                 sel("neural.exc.activity", region=Near("array", 2.0),
                     band=SPIKE_BAND)),
        regions=(("array_neighbourhood", Near("array", 2.0)),
                 ("hand_knob", Anat("cortical_areas", "precentral"))),
        resolution=res(
            rule(Near("array", 2.0), 0.05, Band(0.5, 5000.0)),
            rule(Near("array", 20.0), 0.5, Band(0.5, 5000.0)),
            rule(Anat("cortical_areas", "precentral"), 2.0, LFP),
            default_mm=10.0, default_band=Band(0.0, 50.0)),
        fields=("neural", "device", "effector", "electromagnetic"),
        anatomy=("cortical_areas",),
        topologies=("local", "microcircuit", "efferent_pathway", "device_coupling"),
        processes=("local_excitation", "em_generation", "device_coupling",
                   "efferent_propagation", "effector_activation"),
        observations=("intracortical_spikes", "intracortical_spikes"),
        interventions=("task_cue", "auditory_stimulus", "visual_stimulus",),
        devices=(DeviceSpec("array", "implanted_array", "electrode_grid", "observe",
                            n_elements=192),),
        subject=SubjectSpec(id="participant", template=None),
        window=Window(n=32768, dt=1e-4),  # as `invasive_bci`
        bands=(("neural", Band(0.5, 5000.0)), ("effector", Band(0.0, 10.0))),
        budget=Budget(max_state_variables=1_000_000,
                      max_spectral_coefficients=400_000_000),),
    fit_sources=("willett-handwriting-bci",),
    eval_sources=("willett-handwriting-bci", "invasive-human-array-bci",
                  "braindecode"),
    teachers=("language-models-multiscale", "llama-3.2"),
    negative_controls=("bwin-000-synthetic-scaffold",),
    constrained=(
        "per-channel tuning to attempted pen-stroke direction, which is what the "
        "dataset's character cues are for",
        "the temporal profile of the neural response relative to the cue, to a few "
        "tens of milliseconds",
        "the drift of channel means across sessions, which is explicitly recorded",),
    prior_dominated=(
        "everything outside the array footprint.  the precentral 2 mm rule "
        "materializes cortex that no contact records from",
        "the language teacher's contribution.  a character-level language model "
        "sharply improves decoded text and constrains no neural parameter; it "
        "enters at calibrated precision on the output and its influence must be "
        "reported separately from the neural fit",
        "generalisation to another participant or another array placement -- there "
        "is exactly one of each",),))


INNER_SPEECH = register(NamedModel(
    id="inner_speech",
    doc="""covert speech: the model most at risk of over-claiming, so it is
    written to make its own weakness visible.

    the target is population state in speech-motor and superior-temporal cortex
    during imagined or attempted-but-unvocalised speech.  the difficulty is not
    resolution: it is that no dataset here separates inner speech from articulatory
    preparation, subvocal muscle activity, or auditory imagery.  a scalp
    materialization that reports "decoded inner speech" is almost certainly
    reporting some mixture of those, and `emg` from the jaw and tongue is a real
    confound at exactly the frequencies the decoder uses.

    **r(q).**  4 mm over ventral precentral and superior temporal cortex, where the
    invasive literature places the signal, and 10 mm elsewhere.  it is not finer
    because the scalp datasets cannot use finer, and it is not uniformly coarse
    because the intracranial fit sources can.  the model is deliberately
    materialized at a resolution the *best* of its evidence supports, and the
    `prior_dominated` list says what happens when only scalp data are available.

    **B.**  0.5-100 Hz at the scalp; 70-200 Hz on `neural.exc.activity` where a
    contact array is present, because the ecog and seeg fit sources carry high
    gamma and that is where covert-speech signal actually lives.

    **negative controls do most of the work.**  a rest block and the synthetic
    scaffold must both produce a null.  if they do not, nothing else this model
    reports means anything.""",
    request=MaterializationRequest(
        name="inner-speech",
        targets=(sel("neural.exc.activity",
                     region=Anat("cortical_areas", "ventral_precentral"),
                     band=INTRACRANIAL),
                 sel("effector.drive", band=Band(0.0, 25.0))),
        regions=(("speech_motor", Anat("cortical_areas", "ventral_precentral")),
                 ("auditory", Anat("cortical_areas", "superior_temporal"))),
        resolution=res(
            rule(Near("grid", 5.0), 0.5, INTRACRANIAL),
            rule(Anat("cortical_areas", "ventral_precentral"), 4.0, INTRACRANIAL),
            rule(Anat("cortical_areas", "superior_temporal"), 4.0, INTRACRANIAL),
            default_mm=10.0, default_band=SCALP),
        fields=("neural", "electromagnetic", "effector", "device"),
        anatomy=("cortical_areas",),
        topologies=("cortical_surface", "efferent_pathway", "electromagnetic",
                    "device_coupling"),
        processes=("local_excitation", "em_generation", "em_coupling",
                   "device_coupling", "efferent_propagation", "effector_activation"),
        observations=("eeg", "ecog",
                      "intracortical_spikes", "emg"),
        interventions=("task_cue", "auditory_stimulus", "visual_stimulus",),
        devices=(DeviceSpec("eeg", "sensor_array", "eeg_cap", "observe",
                            n_elements=64),
                 DeviceSpec("grid", "implanted_array", "electrode_grid", "observe",
                            n_elements=128)),
        subject=SubjectSpec(),
        window=Window(n=4096, dt=1e-3),
        bands=(("neural.exc.activity", HIGH_GAMMA),
               ("electromagnetic", SCALP), ("effector", Band(0.0, 25.0))),
        budget=Budget(max_state_variables=200_000),),
    fit_sources=("ds006033", "single-word-production-dutch-ieeg", "ds006234",
                 "willett-speech-neuroprosthesis", "ds007554"),
    eval_sources=("ds005574", "chang-lab-ecog-cv-syllables"),
    teachers=("phonetic-acoustic-models", "language-models-multiscale",
              "wav2vec-bert-2.0"),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study",
                       "tuh-eeg"),
    constrained=(
        "high-gamma amplitude in ventral precentral cortex during attempted speech, "
        "from the invasive fit sources where the contacts sit on the relevant gyrus",
        "the timing of the covert response relative to the cue",
        "the articulatory-emg confound itself, since `ds007554` and the dutch seeg "
        "corpus record it alongside",),
    prior_dominated=(
        "the distinction between imagined speech and articulatory preparation.  no "
        "source here separates them and any claim to have decoded 'inner' speech "
        "rather than suppressed motor output is prior, not evidence",
        "everything at the scalp.  `ds006033` is the only scalp fit source and its "
        "effect sizes are small; a scalp-only materialization of this model is "
        "essentially all prior and must be presented that way",
        "phonological content beyond a small closed vocabulary; the language "
        "teacher supplies the rest at calibrated, low-rank precision",),))
