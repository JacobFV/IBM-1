"""cheap materializations that stand in for expensive ones.

    M = materialize(R, r, B, F, A, T, P)

three models, coarse on both axes, existing to be run thousands of times where a
fine materialization could be run once.  the architectural point they make is the
one §1 ends on: if a coarse materialization and a fine one agree to within what
the data can see, the fine one was waste, and the only way to find that out is to
build both and compare.  `MaterializationRequest.coarsened` produces requests of
roughly this shape, and these three are the named versions with real sources
attached.

a surrogate is not a different model.  `macro_surrogate` reads the same
`tract_propagation` and `lateral_cortical_propagation` declarations as
`resting_state_fc` and `virtual_lesion`; it differs only in r, B and budget.  the
moment a surrogate is allowed to override a process it has stopped being a view of
the implicit model and become an independently defined brain model, which is the
one thing this package exists to prevent.

`encoding_model` is the awkward member.  its dynamics are learned rather than
declared, and the honest way to hold it is as a *teacher-shaped* object: it infills
brain state given a stimulus, it reports an r^2, and everything it contributes
downstream is bounded by that r^2 rather than entering as measurement.  it is
declared here rather than hidden inside another model so that the bound is written
down somewhere.
"""

from __future__ import annotations

from ibm.materialize.library import NamedModel, anat, register, res, rule, subcortex
from ibm.materialize.request import (
    Budget,
    DeviceSpec,
    MaterializationRequest,
    SubjectSpec,
    Window,)
from ibm.vocabulary import (
    Band,
    OnSupport,
    sel,)

#: the surrogate band: slow enough to be cheap, wide enough to carry the
#: functional-connectivity and gradient structure the surrogate is used for.
MACRO = Band(0.01, 0.25)
#: neural-mass surrogates are also run at electrophysiological bandwidth, where
#: they are compared against meg power maps rather than bold connectivity.
MASS = Band(0.5, 100.0)
#: an encoding model predicts a bold or eeg response to a stimulus; the bold arm
#: is the narrow one.
ENCODING = Band(0.0, 0.25)

#: early sensory cortex in the labels `cortical_areas` declares.  `encoding_model`
#: argued for 2 mm in "v1" and "heschl", neither of which is a label of the
#: declared desikan-killiany system, so the refinement never happened.
V1 = anat("cortical_areas", "pericalcarine")
HESCHL = anat("cortical_areas", "transversetemporal")
OCCIPITAL = anat("cortical_areas", "lateraloccipital", "cuneus", "lingual",
                 "pericalcarine")



MACRO_SURROGATE = register(NamedModel(
    id="macro_surrogate",
    doc="""a parcel-level neural-mass materialization: coarse in space, coarse in
    band, and deliberately so.

    **why coarse in space.**  the surrogate exists to be run 10^4 times -- parameter
    sweeps, lesion scans, sensitivity analyses, posterior predictive checks -- and
    at that count the only affordable state variable is one per parcel.  the
    justification is not the budget, though: it is that the couplings this model
    carries are the tract-level ones, whose edges come from a diffusion connectome
    estimated at 1-2 mm and parcellated anyway.  materializing at 2 mm would give
    a fine grid whose edges were all copied from a parcel-level matrix, which is
    resolution with no information behind it.

    **and why, being the coarsest model in the library, it is the one that most
    needs the cortical sheet.**  the two facts are the same fact.  the cost of
    indexing cortex by volume position is that a euclidean neighbourhood joins
    the two banks of a sulcus, and that cost grows with the spacing: on the order
    of a tenth of a percent of cortical pairs at 2 mm, and 58% at 10 mm
    (`Component.alt_supports`).  a 2 mm haemodynamic model can afford the volume
    indexing precisely because it is fine; at 8-10 mm a volume cortical graph is
    mostly wrong, and `lateral_cortical_propagation` -- one of this model's two
    structural couplings, and the one that gives it any spatial organisation
    at all below the tract level -- would be running over connections that do not
    exist.  §1 puts it as "a topology whose edges do not survive coarsening", and
    this is that case: it is not the *state* that fails to coarse-grain here, it
    is the adjacency.

    so cortical population state is on column nodes at 8 mm, and the thalamus,
    basal ganglia, cerebellum and brainstem -- which have no sheet, and which the
    `thalamic_nuclei`, `bg_territories` and `cerebellar_lobules` systems in this
    model's anatomy list exist to partition -- are in the volume at 10 mm.  the
    volume is carved with `subcortex()` so that the two are a partition: a
    surrogate whose cortical nodes were each counted twice, once on the sheet and
    once in the voxel containing it, would have twice the excitatory drive and a
    fitted global coupling that absorbed the error.

    **why coarse in band.**  0.01-0.25 Hz for the bold arm.  a neural mass at
    parcel scale has no meaningful structure above that once it has been passed
    through a hemodynamic filter, and the comparison target -- a connectivity matrix
    -- is a second-order statistic in that band.  the electrophysiological arm
    (`MASS`, 0.5-100 Hz) exists for comparison against meg power maps and costs
    more, which is why it is a separate band override rather than the default.

    **what it is actually for.**  two things.  first, it is the null against which
    every finer materialization in this library argues: if `resting_state_fc` at
    5 mm predicts no better than this at parcel scale, 5 mm was waste.  second, it
    makes uncertainty propagation tractable -- a thousand runs of a cheap model
    give a predictive distribution that one run of an expensive model cannot.

    **the honest limitation.**  a surrogate that is fitted to reproduce a
    connectivity matrix will reproduce a connectivity matrix.  its agreement with
    the data it was fitted to is not evidence about the mechanism, and the
    `prior_dominated` list says which parts of the mechanism are unconstrained.""",
    request=MaterializationRequest(
        name="macro-surrogate",
        targets=(sel("neural.exc.activity", "neural.inh.activity", band=MACRO),
                 sel("blood.deoxyhemoglobin", band=MACRO)),
        regions=(("cortex", OnSupport("cortical_surface")),
                 ("subcortex", subcortex())),
        resolution=res(
            rule(OnSupport("cortical_surface"), 8.0, MACRO),
            rule(OnSupport("tissue"), 10.0, MACRO),
            default_mm=12.0, default_band=MACRO),
        fields=("neural", "blood"),
        anatomy=("cortical_areas", "thalamic_nuclei", "cerebellar_lobules",
                 "bg_territories"),
        topologies=("tractometric", "cortical_surface", "local"),
        processes=("local_excitation", "local_inhibition", "tract_propagation",
                   "lateral_cortical_propagation", "thalamocortical_coupling",
                   "neurovascular_coupling", "bold_formation"),
        observations=("bold", "meg"),
        devices=(),
        subject=SubjectSpec(id="template", template="mni152",
                            note="template geometry is acceptable here and almost "
                                 "nowhere else: the parcellation is the resolution"),
        window=Window(n=512, dt=1.0),
        bands=(("neural", MACRO), ("blood", MACRO)),
        # the unit here is the architecture's own: one component of one field at
        # one position, not one position.  a parcel-scale materialization of this
        # subject is ~4,800 sites and ~20 traced components each, so "one state
        # variable per parcel" is 9.3e4 variables and not 2e4 -- the old ceiling
        # counted sites and was never reachable at the 8 mm this model argues for.
        budget=Budget(max_state_variables=150_000,
                      max_spectral_coefficients=12_000_000,
                      max_bytes=1 << 30),),
    fit_sources=("hcp-functional-connectivity", "braingraph-hcp-connectomes",
                 "netneuro-lausanne-sc", "hansen-many-networks", "hcp-meg-maps",
                 "schaefer2018", "glasser2016"),
    eval_sources=("midnight-scan-club", "corr-reliability", "margulies2016-gradients",
                  "intrinsic-timescale-maps", "hcp-young-adult"),
    teachers=("the-virtual-brain", "neural-mass-model-families", "neurolib",
              "brainpy"),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study",
                       "scwbd-001-004-predecessor-models"),
    constrained=(
        "the global coupling scale and the excitation/inhibition balance parameter, "
        "which the fit to a group connectivity matrix determines to a few percent",
        "conduction delays at the tract level, weakly, through their effect on "
        "band-limited phase relationships in the meg arm",
        "the parcel-level noise amplitude",),
    prior_dominated=(
        "everything below the parcel.  a parcel is a centimetre and the model has "
        "one state variable for it; any within-parcel statement is prior",
        "the neural mass's functional form.  several published forms fit these "
        "targets equally well and the corpus does not separate them, so the choice "
        "of f is a prior and its consequences must be reported as such",
        "the structural connectome's false positives.  tractography-derived edges "
        "carry a large and well-documented false-positive rate and the surrogate "
        "inherits every one of them",),
    notes="the point of comparison is `MaterializationRequest.coarsened`: this "
          "model is what a fine materialization should be coarsened *onto* when "
          "`build.resolution_earns_its_cost` is evaluated.",))


CONNECTOME_GNN = register(NamedModel(
    id="connectome_gnn",
    doc="""a learned surrogate over the parcel graph.

    the same r(q) and B as `macro_surrogate` -- parcel scale, slow band -- with a
    learned rather than a declared coupling.  it is kept as a separate named model
    because its failure mode is different and worth isolating: a graph network fitted
    on tractography-derived edges will happily learn to rely on edges that do not
    exist.

    **why the negative controls are the most important field on this model.**  the
    ismrm 2015 tractography challenge established that a typical whole-brain
    tractogram contains a large fraction of false-positive bundles -- more invalid
    connections than valid ones, for many submitted pipelines.  a surrogate whose
    predictions depend on those edges is fitting an artifact of the reconstruction.
    `ismrm2015-submissions` and `ismrm2015-tractography-challenge` are here so that
    the dependence can be measured: predictions must degrade gracefully when
    known-false bundles are removed, and a model that does not is not a model of
    the brain.

    **r(q) and B.**  8 mm and 0.01-0.25 Hz.  identical to `macro_surrogate` by
    intent -- if a learned coupling needs a finer materialization than a declared
    one to fit the same data, the extra resolution is absorbing model error, not
    representing biology.

    **the cortex is on the sheet because the sheet is what the edges attach to.**
    this model is nothing but a graph, and every claim it makes is a claim about
    which positions are adjacent.  two of its three sources of adjacency are
    surface objects: a streamline terminates at the grey/white interface, which is
    the sheet by construction, and lateral cortical coupling runs under the
    sheet's geodesic metric.  index the cortex by 8 mm voxels instead and a
    majority of cortical pairs acquire a short euclidean neighbour across a sulcus
    that no horizontal axon connects -- and short spurious edges are exactly the
    failure mode `ismrm2015-submissions` is here to detect, so introducing a fresh
    batch of them in the *materialization* would make the negative control
    unreadable.  the model would then be measuring its own octree and reporting
    it as connectivity.

    the volume half carries what has no sheet -- thalamus, striatum, cerebellum,
    brainstem, and the white matter where `structural.axonal_density` and
    `structural.fiber_orientation` live, those being the only support those
    components have.  it is carved with `subcortex()` so the two are a partition:
    the same cortical node present both as a column and as the voxel containing
    it would give a graph network two copies of every cortical feature and let it
    fit the duplication.

    **its relation to the rest of the library.**  it is a surrogate, not a teacher.
    what it produces may be compared against a materialization and may be used to
    propose parameters; it may not supply a likelihood, because its residuals are
    correlated with the connectome it was trained on.""",
    request=MaterializationRequest(
        name="connectome-gnn",
        targets=(sel("neural.exc.activity", band=MACRO),
                 sel("structural.axonal_density", "structural.fiber_orientation",
                     band=Band(0.0, 0.001))),
        regions=(("cortex", OnSupport("cortical_surface")),
                 ("subcortex", subcortex())),
        resolution=res(
            rule(OnSupport("cortical_surface"), 8.0, MACRO),
            rule(OnSupport("tissue"), 10.0, MACRO),
            default_mm=12.0, default_band=MACRO),
        fields=("neural", "structural", "blood"),
        anatomy=("cortical_areas", "thalamic_nuclei", "bg_territories",
                 "cerebellar_lobules"),
        topologies=("tractometric", "cortical_surface"),
        processes=("tract_propagation", "lateral_cortical_propagation",
                   "neurovascular_coupling"),
        observations=("bold", "dwi_microstructure"),
        devices=(),
        subject=SubjectSpec(),
        window=Window(n=512, dt=1.0),
        bands=(("neural", MACRO), ("structural", Band(0.0, 0.001)),
               ("blood", MACRO)),
        # the unit here is the architecture's own: one component of one field at
        # one position, not one position.  a parcel-scale materialization of this
        # subject is ~4,800 sites and ~20 traced components each, so "one state
        # variable per parcel" is 9.3e4 variables and not 2e4 -- the old ceiling
        # counted sites and was never reachable at the 8 mm this model argues for.
        budget=Budget(max_state_variables=150_000,
                      max_spectral_coefficients=12_000_000,
                      max_bytes=1 << 30),),
    fit_sources=("braingraph-hcp-connectomes", "enigma-hcp-structural-connectome",
                 "hansen-lausanne-sc", "hansen-schaefer-sc", "netneuro-lausanne-sc",
                 "hcp-young-adult", "tractoinferno"),
    eval_sources=("tractometer", "hcp-functional-connectivity", "corr-reliability",
                  "fiber-data-hub", "histological-tract-atlases", "cocomac"),
    teachers=("brain-harmony", "brain-jepa", "brainlm"),
    negative_controls=("ismrm2015-submissions", "ismrm2015-tractography-challenge",
                       "bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "the mapping from group structural connectivity to group functional "
        "connectivity, which is reproducible across the fit cohorts",
        "which graph features carry that mapping -- degree, communicability, "
        "geodesic distance -- since the ablations are cheap at this scale",
        "the reliability floor, from `corr-reliability`'s repeat scans",),
    prior_dominated=(
        "the existence of individual edges.  tractography false positives are the "
        "binding limitation and `histological-tract-atlases` and `cocomac` can "
        "falsify edges but are too sparse and too cross-species to fit them",
        "directionality.  diffusion mri is orientation-blind to direction and bold "
        "correlation does not orient; every directed claim is prior",
        "any interpretation of a learned coupling as a mechanism.  the model has "
        "no declared process behind its edges beyond `tract_propagation`, and its "
        "weights are not parameters of that process",),))


ENCODING_MODEL = register(NamedModel(
    id="encoding_model",
    doc="""stimulus to brain state: predict the response a stimulus will produce.

    the teacher-shaped model, declared explicitly so that its precision bound has
    somewhere to live.

    **r(q) is set by where the encoding actually works.**  early visual and
    auditory cortex are retinotopic and tonotopic, and neighbouring millimetres
    genuinely encode different things -- a real heterogeneity-inside-the-cell
    failure -- so 2 mm there.  association cortex is not organised that way at any
    scale these models resolve, and reported encoding accuracy there is low, so
    6 mm.  the resolution follows the evidence gradient rather than being uniform,
    and that gradient is the model's most honest feature.

    **B is the bold band for the fmri arm.**  0-0.25 Hz.  the eeg and meg arms are
    materialized separately by `eeg_to_image` and `meg_to_text`; keeping the
    encoding model narrow here is what makes a whole-brain 2 mm early-sensory
    materialization affordable at all.

    **the precision bound is the whole point.**  an encoding model that explains a
    fraction r^2 of a voxel's variance may contribute at most 1/((1-r^2) Var[x]) of
    precision when its output is used to infill unobserved brain state.  in early
    visual cortex that r^2 is perhaps 0.3-0.5 and the contribution is real; in
    association cortex it is often below 0.1 and the contribution is close to
    nothing.  a single global precision would be wrong in both places.  worse, the
    residuals of a stimulus encoder covary across every position it writes --
    they are all driven by the same stimulus features -- so the increment is
    low-rank, not diagonal: writing 10^5 voxels does not supply 10^5 independent
    constraints.  both corrections are mandatory and neither is optional
    bookkeeping.

    **the teachers here are also the competition.**  `tribe`, `mirage` and the
    stimulus encoders do this task directly, and `algonauts-2025` is the held-out
    benchmark.  they are teachers for infill and evaluation targets for accuracy,
    and those two roles must not be run on the same split.""",
    request=MaterializationRequest(
        name="encoding-model",
        targets=(sel("neural.exc.activity", band=ENCODING),
                 sel("blood.deoxyhemoglobin", band=ENCODING),
                 # the stimulus boundary as a 1 s tr samples it.  the fast
                 # transduction band is real and belongs to `eeg_to_image` and
                 # `speech_envelope`; carrying it here would be aliasing it.
                 sel("transduction.photoreceptor", "transduction.hair_cell",
                     band=ENCODING)),
        regions=(("early_visual", V1),
                 ("early_auditory", HESCHL),
                 ("association", OnSupport("cortical_surface"))),
        resolution=res(
            rule(V1, 2.0, ENCODING),
            rule(HESCHL, 2.0, ENCODING),
            rule(OCCIPITAL, 3.0, ENCODING),
            rule(OnSupport("cortical_surface"), 6.0, ENCODING),
            default_mm=8.0, default_band=ENCODING),
        fields=("neural", "blood", "transduction", "device"),
        anatomy=("cortical_areas", "cerebellar_lobules"),
        topologies=("afferent_pathway", "cortical_surface", "tractometric",
                    "vascular"),
        processes=("transduction", "afferent_propagation",
                   "lateral_cortical_propagation", "tract_propagation",
                   "neurovascular_coupling", "bold_formation"),
        observations=("bold",),
        devices=(DeviceSpec("scanner", "scanner_element", "scanner", "observe"),
                 DeviceSpec("screen", "display", "display", "stimulate"),
                 DeviceSpec("speaker", "display", "display", "stimulate")),
        subject=SubjectSpec(),
        window=Window(n=512, dt=1.0),
        bands=(("neural", ENCODING), ("blood", ENCODING),
               ("transduction", ENCODING)),
        budget=Budget(max_state_variables=600_000,
                      max_spectral_coefficients=10_000_000),),
    fit_sources=("natural-scenes-dataset", "bold5000", "courtois-neuromod",
                 "narratives", "cneuromod-things", "hcp-movie-7t",
                 "studyforrest-ds000113"),
    eval_sources=("algonauts-2025", "cneuromod-friends", "gallant-crcns-natural-movies",
                  "kamitani-perception-and-imagery-fmri", "hcp-7t-retinotopy",
                  "haxby-hyperalignment-corpora"),
    teachers=("tribe", "tribe-v2", "multiscale-visual-encoders", "vjepa2",
              "wav2vec-bert-2.0", "llama-3.2", "audio-spectrotemporal-encoders",
              "multimodal-audiovisual-models", "depth-scene-geometry-models",
              "optical-flow-point-tracking-models", "mirage"),
    negative_controls=("mit-tuebingen-saliency-benchmark", "salicon",
                       "bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "voxelwise receptive fields in early visual cortex, which "
        "`natural-scenes-dataset` and `hcp-7t-retinotopy` determine as well as "
        "anything in this corpus",
        "spectrotemporal tuning in auditory core, from the naturalistic listening "
        "sources",
        "the per-region encoding accuracy itself -- which is the number the "
        "distillation precision bound is computed from, and which therefore has to "
        "be measured on a held-out split rather than assumed",),
    prior_dominated=(
        "association cortex.  reported r^2 there is low and the precision the "
        "teachers may contribute is correspondingly near zero; a map that looks "
        "smooth and confident across frontal cortex is showing the prior",
        "the feature space.  which layer of which network best predicts a region is "
        "an empirical fact about the network, not about the brain, and none of "
        "these sources adjudicates between competing feature spaces mechanistically",
        "everything the teachers write beyond their calibrated precision.  their "
        "residuals covary, so the increment is low-rank and off-distribution use "
        "inflates the variance by a factor that is itself fitted",),
    notes="teachers here are also evaluation baselines; the two roles must be run "
          "on disjoint splits or the model is scored on what trained it.",))
