"""blood, metabolism and their optical and magnetic observation.

    M = materialize(R, r, B, F, A, T, P)

the whole group sits in one corner of the budget: **spatially fine, temporally
narrow**, which is the exact mirror of `invasive_bci`.  the reason is physical and
worth stating once here rather than six times below.

*temporally narrow.*  blood has no meaningful structure above roughly 0.5 Hz.  the
hemodynamic response to a brief neural event is a several-second impulse whose
spectrum is essentially gone by 0.25 Hz, and the measurements that see it are
sampled at 0.5-1.5 Hz anyway.  §1's rule that marginalizing a gaussian to a band
is exact means restricting to `HEMODYNAMIC` costs nothing in accuracy and buys a
factor of several hundred in coefficients over an lfp-bandwidth materialization of
the same sites.  that factor is what makes whole-brain fine spatial sampling
affordable at all.

*spatially fine.*  a 2 mm voxel straddles two cortical areas and both banks of a
sulcus; the vascular tree's edges do not survive coarsening, because two capillary
beds a millimetre apart may be fed by different penetrating arterioles and be far
apart in flow; and the venous drainage that dominates gradient-echo bold is
organised at the scale of pial veins, hundreds of microns across, sitting on the
surface where they contaminate the voxel above them.  every one of those is a §1
failure of commutation.

`asl_perfusion` is the exception that proves the rule: its measurement is a
subtraction with about one percent contrast, so it is coarse in space *and* narrow
in band, and refining it would be materializing noise.
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
    DC,
    HEMODYNAMIC,
    OnSupport,
    Near,
    ULTRASLOW,
    sel,)

#: what a bold time series constrains.  the canonical resting-state band; above
#: 0.25 Hz a 0.72 s tr is aliasing cardiac and respiratory signal into it.
BOLD = Band(0.0, 0.25)
#: fnirs is sampled fast enough to *see* the cardiac pulse, and unlike fmri it
#: cannot separate scalp blood from brain blood, so the pulsatile nuisance must be
#: materialized rather than filtered away.  hence 2 Hz, not 0.5.
OPTICAL = Band(0.0, 2.0)
#: neural drive as the hemodynamic chain sees it.  `neurovascular_coupling` reads a
#: rate *envelope*, not a waveform, and the envelope is all a 0.7-2 s sampled window
#: can carry: the underlying population activity has structure to 100 Hz, and
#: materializing that band here would be materializing an aliased one.  the fast band
#: belongs to `eeg_forward`; a joint eeg-fmri materialization declares two windows
#: rather than one compromise.
DRIVE = Band(0.0, 0.5)

#: the arterial watershed zones, in the labels `vascular_territories` declares.
#: this is where transit delay and reactivity vary fastest and where a single
#: coarse cell is most likely to average two territories, so it is the one place
#: in a hemodynamic model that a finer volume earns its cost.
WATERSHED = anat("vascular_territories", "watershed_aca_mca", "watershed_mca_pca",
                 "watershed_deep")



BOLD_FORWARD = register(NamedModel(
    id="bold_forward",
    doc="""blood and deoxyhemoglobin state from population activity, and the
    gradient-echo signal that follows.

    the canonical spatially-fine, temporally-narrow materialization.

    **why the whole thing is in the volume, and the cortical sheet is not named.**
    the sheet buys exactly one thing that the parenchyma volume cannot supply: a
    metric in which two points across a sulcus are far apart.  that metric is
    worth its cost wherever something *propagates* laterally, because a euclidean
    neighbourhood then invents an axon that does not exist.  nothing in this
    model propagates laterally.  every process here --
    `neurovascular_coupling`, `vascular_flow`, `bold_formation`, `metabolism`,
    `tissue_exchange` -- turns the state of one tissue element into the state of
    that same element or of the vessel feeding it, and the only long-range
    transport is along the vascular tree, whose own topology is volumetric.  so
    re-indexing the cortex by column node rather than by voxel destroys no edge
    and changes no answer, which is §1's condition for the finer object being
    waste.  the sheet is dropped from R for that reason and not to save money.

    **what does fail to commute here is volumetric, and 2 mm is the answer to it.**
    a 4 mm cell contains both banks of a sulcus whose activity is often
    anticorrelated, adjacent cells drain into different pial veins so the
    `vascular` topology's edges do not survive coarsening, and the balloon
    model's volume-to-signal relation is convex, so the signal from a cell with
    one active half and one silent half is not the signal of the average.  all
    three are statements about the size of a *voxel*, and the remedy to all three
    is a finer voxel rather than a different support.  2 mm is where they become
    small; laminar structure would need 0.5 mm and is not materialized here
    because no fit source in this list resolves it.

    the price of the choice is stated rather than hidden: a 2 mm volume graph
    mis-connects on the order of a tenth of a percent of cortical pairs across a
    sulcus (`Component.alt_supports`), and since no process in this model uses
    that graph to propagate anything, a tenth of a percent of nothing is what it
    costs.  a materialization that *does* need lateral spread --
    `resting_state_fc`, `macro_surrogate`, `virtual_lesion` -- names the sheet,
    and pays for it.

    **why the band collapses to 0.25 Hz.**  the hrf is a low-pass filter with a
    several-second impulse response and the measurement is sampled at 1.4 Hz.  the
    neural component keeps a slightly wider band (`DRIVE`) because
    `neurovascular_coupling` reads an activity envelope rather than the bold signal
    itself.  it does *not* keep the 100 Hz band the underlying population activity
    has, because this window cannot represent it and materializing an aliased band
    is worse than not materializing it: that band belongs to `eeg_forward`, and a
    joint materialization declares two windows rather than one compromise.

    **what 2 mm whole-brain actually costs, said out loud.**  a parenchyma octree
    at 2 mm over a real head is ~6e5 leaves, and the trace puts ~25 components on
    `tissue`: 1.5e7 state variables at ~48 retained coefficients each, which is
    22 GiB of state.  the budget below is not a modelling preference -- §1 is
    explicit that `max_bytes` is the one ceiling that is a hardware fact -- so
    this model as declared does not fit on a machine that has 8 GiB to give it,
    and a build coarsens onto 4 mm and says by how much.  that is the
    `coarsened()` argument working rather than failing: the question it forces is
    whether the sulcal-bank and vein effects above are worth 8x the memory, and
    the answer is an empirical one this model is set up to give.

    **the vein problem is the honest limit.**  gradient-echo bold weights draining
    veins heavily, and a vein sits millimetres downstream of the tissue that
    consumed the oxygen.  the model materializes `vascular_tree` and can represent
    it; whether the fit can *identify* it from these sources is another matter and
    is listed under prior-dominated.""",
    request=MaterializationRequest(
        name="bold-forward",
        targets=(sel("blood.deoxyhemoglobin", "blood.volume", "blood.flow",
                     "blood.oxygenation", band=BOLD),
                 sel("metabolic.consumption", "metabolic.oxygen", band=BOLD)),
        regions=(("vasculature", OnSupport("vascular_tree")),
                 ("brain", OnSupport("tissue"))),
        resolution=res(
            rule(OnSupport("vascular_tree"), 0.5, BOLD),
            rule(OnSupport("tissue"), 2.0, BOLD),
            default_mm=4.0, default_band=BOLD),
        fields=("neural", "blood", "metabolic", "extracellular"),
        anatomy=("cortical_areas", "vascular_territories"),
        topologies=("vascular", "metabolic_exchange", "local"),
        processes=("neurovascular_coupling", "vascular_flow", "bold_formation",
                   "metabolism", "tissue_exchange"),
        observations=("bold",),
        devices=(DeviceSpec("scanner", "scanner_element", "scanner", "observe",
                            note="3T or 7T gradient-echo epi"),),
        subject=SubjectSpec(),
        window=Window(n=512, dt=0.72),
        bands=(("blood", BOLD), ("metabolic", BOLD), ("neural", DRIVE),
               ("extracellular", ULTRASLOW)),
        budget=Budget(max_state_variables=2_000_000,
                      max_spectral_coefficients=40_000_000),),
    fit_sources=("hcp-young-adult", "natural-scenes-dataset", "midnight-scan-club",
                 "ds003192", "ds006072", "aomic"),
    eval_sources=("algonauts-2025", "cneuromod-friends", "corr-reliability",
                  "ibc-ds000244", "hcp-task-vision-maps"),
    teachers=("haemodynamic-model-families", "brainlm", "tribe"),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study",
                       "adni"),
    constrained=(
        "the shape and latency of the hemodynamic impulse response in well-sampled "
        "cortex, from the task and breath-hold data together",
        "the neural-drive-to-flow gain per cortical area, which the natural-scenes "
        "and hcp task contrasts pin at the parcel level",
        "the resting oxygen extraction fraction, weakly, through `ds006072`'s "
        "multi-echo data and the calibrated `ds003192` breath-hold task",),
    prior_dominated=(
        "the venous versus tissue decomposition at each voxel.  the model "
        "materializes a vascular tree at 0.5 mm and no fit source here has "
        "vein-resolved data; the split sits at its atlas prior and every claim "
        "about the *location* of neural activity inherits that",
        "laminar and columnar structure, deliberately not materialized -- 1 mm is "
        "the finest these sources support and a finer model would be reporting "
        "the prior at higher resolution",
        "the coupling between metabolic consumption and flow, which is degenerate "
        "with the flow-to-volume exponent from bold data alone",),))


FMRI_INFILL = register(NamedModel(
    id="fmri_infill",
    doc="""predict the unrecorded parts of a bold acquisition: masked runs, missing
    slices, and the response to a stimulus that was shown but not scanned.

    coarser than `bold_forward` on purpose.  infill is a statement about the
    joint distribution over parcels and time, not about the vascular tree, and the
    modes that carry it are large-scale: the principal gradient, the canonical
    networks, and a slow drift.  a 1 mm materialization would spend a hundred times
    the state variables to represent a field whose recoverable rank is a few
    hundred.  3 mm on the sheet is where the objective stops improving, and this
    model is the natural place to run the `coarsened()` equivalence check that §1
    demands: if 6 mm predicts as well, 3 mm was waste and should be dropped.

    **B.**  0.01-0.25 Hz.  the low edge is not zero because scanner drift below
    0.01 Hz is a nuisance the model would otherwise try to explain as brain state.

    **why this one names the sheet when `bold_forward` and `hrf` do not.**  the
    three models sit in the same group and split on one question: does anything
    here *propagate along the cortex*.  in `bold_forward` and `hrf` nothing does
    -- the chain is local plus vascular transport -- so a voxel indexing of the
    cortex destroys no edge and the sheet is not named.  here the two declared
    couplings are `lateral_cortical_propagation`, which runs under the sheet's
    own geodesic metric, and `tract_propagation`, whose endpoints are on the
    grey/white interface; and the modes this model exists to infill are the
    principal gradient and the canonical networks, which are objects on the sheet
    and not in the volume.  a euclidean neighbourhood joins the two banks of a
    sulcus that no horizontal axon connects, and the resulting short-range
    correlation is indistinguishable from the structure being infilled.  so R
    names both supports, and the volume half is carved with `subcortex()` so that
    the two are a partition: cortical population state on column nodes,
    subcortical, cerebellar and brainstem state on parenchyma voxels.

    the volume half is not decoration.  the thalamus and the cerebellum carry a
    large share of resting bold variance and `cerebellar_lobules` is in this
    model's anatomy list; a sheet-only placement would have given them no state
    at all and left the infill predicting cortex from cortex.

    **this is the most teacher-heavy model in the library and it is dangerous.**
    `brainlm`, `brain-jepa`, `brain-harmony`, `tribe`, `tribe-v2` and `neuroworld`
    all do masked or autoregressive infill of exactly this signal, and several of
    them were pretrained on `hcp-young-adult`, which is also a fit source here.
    that overlap means their agreement with each other and with the fit data is not
    independent evidence.  each contributes a low-rank precision increment bounded
    by its own reported r^2 on the stream in question, inflated for
    off-distribution use, and never at unit precision.""",
    request=MaterializationRequest(
        name="fmri-infill",
        targets=(sel("blood.deoxyhemoglobin", "blood.flow", band=BOLD),
                 sel("neural.exc.activity", band=BOLD)),
        regions=(("cortex", OnSupport("cortical_surface")),
                 ("subcortex", subcortex())),
        resolution=res(
            rule(OnSupport("cortical_surface"), 3.0, BOLD),
            rule(OnSupport("tissue"), 4.0, BOLD),
            default_mm=6.0, default_band=Band(0.01, 0.25)),
        fields=("neural", "blood", "metabolic"),
        anatomy=("cortical_areas", "vascular_territories", "cerebellar_lobules"),
        topologies=("cortical_surface", "tractometric", "vascular"),
        processes=("neurovascular_coupling", "bold_formation", "vascular_flow",
                   "tract_propagation", "lateral_cortical_propagation"),
        observations=("bold",),
        devices=(DeviceSpec("scanner", "scanner_element", "scanner", "observe"),),
        subject=SubjectSpec(),
        window=Window(n=1024, dt=1.0),
        bands=(("blood", BOLD), ("neural", BOLD)),
        # 3 mm on the sheet is ~1.6e4 column nodes and 4 mm in the
        # ribbon-excluded parenchyma another ~1.5e4, and the split neural field
        # puts 13 components on each: 5.9e5 variables at a ~180-coefficient
        # window.  the old 3e5 / 2e7 pair was written for a sheet-only placement
        # and refused this model's own r(q) by a factor of two.
        budget=Budget(max_state_variables=800_000,
                      max_spectral_coefficients=150_000_000),),
    fit_sources=("hcp-young-adult", "narratives", "courtois-neuromod",
                 "naturalistic-neuroimaging-database", "aomic-id1000"),
    eval_sources=("algonauts-2025", "sherlock-merlin-princeton",
                  "haxby-hyperalignment-corpora", "corr-reliability",
                  "midnight-scan-club"),
    teachers=("brainlm", "brain-jepa", "brain-harmony", "tribe", "tribe-v2",
              "neuroworld", "mirage"),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study",
                       "enigma"),
    constrained=(
        "the low-dimensional covariance structure across parcels -- the first few "
        "hundred modes, which thousands of subjects determine very well",
        "the temporal autocorrelation of each parcel, including its regional "
        "variation",
        "the stimulus-locked component during naturalistic viewing and listening, "
        "from `narratives` and the neuromod corpora",),
    prior_dominated=(
        "anything below the parcel scale.  the 3 mm grid exists for geometry, not "
        "because the fit resolves it",
        "the direction of influence between parcels.  the topologies are "
        "materialized as directed but resting bold cannot orient them, so directed "
        "claims here are prior",
        "teacher-supplied infill in regions where the teachers' reported accuracy "
        "is low -- association cortex especially.  the precision they contribute "
        "there is near zero by construction and the output is the prior",),))


HRF = register(NamedModel(
    id="hrf",
    doc="""the hemodynamic response function itself, as a spatially varying object.

    the only model in the group whose *target* is a transfer function rather than a
    state trajectory, and it is separated out because the assumption it interrogates
    -- one canonical hrf everywhere -- is wrong in a way that matters for every
    other hemodynamic model in this library.

    **why r(q) is fine where it is fine.**  hrf latency varies by two seconds and
    amplitude by a factor of several across the brain, and the variation is
    organised by the vasculature: tissue near a large draining vein responds later
    and larger, and the arterial transit time differs by hundreds of milliseconds
    between vascular territories.  so 1 mm near mapped veins (which is what
    `7t-qsm-venograms` provides), 2 mm in the watershed zones between arterial
    territories -- where transit time and reactivity vary fastest, and where the
    declared `vascular_territories` system actually has labels for it -- but 4 mm
    across the rest of the parenchyma where the variation is smooth.  the
    refinement follows the *cause* of the heterogeneity rather than following the
    voxel grid, which is the §1 instruction taken literally.

    **and why the cortical sheet is not named at all.**  the same instruction,
    applied to the choice of support rather than to the spacing.  the sheet is
    the right indexing for a quantity organised by cortical topology; an hrf is
    not one.  it is organised by the arterial supply and the venous drainage,
    which cross sulci freely -- a single pial vein drains both banks, and a
    watershed boundary runs through the middle of a gyrus -- so column nodes
    would impose a geometry that the heterogeneity does not follow, at the price
    of splitting a whole-brain vascular model in two.  every process here
    (`neurovascular_coupling`, `vascular_flow`, `bold_formation`,
    `tissue_exchange`, `metabolism`) couples a tissue element to itself or to the
    vessel serving it, so nothing propagates along the sheet and no edge is lost
    by indexing the cortex volumetrically.  the state that matters is in the
    volume and on the vascular tree, and R names exactly those two.

    that also keeps this model usable for what it is *for*: it is the correction
    the other hemodynamic models apply, and every one of them -- `bold_forward`,
    `asl_perfusion`, `quantitative_bold`, `fnirs_forward` -- wants that correction
    voxelwise, over the whole brain including the structures no cortical sheet
    covers.

    **B.**  0-0.5 Hz for blood, but with a long window: an hrf is a five-to-twenty
    second object and a 30 second window resolves no finer than 0.033 Hz, which is
    barely enough to separate the undershoot from the peak.  the window here is
    long for that reason and not because anything is slow.

    **why the breath-hold and hypercapnia sources dominate the fit.**  a co2
    challenge drives the vasculature without driving neurons, which is the only way
    in this whole corpus to separate `neurovascular_coupling` from `vascular_flow`.
    without them the two are degenerate and the hrf is a single lumped kernel.""",
    request=MaterializationRequest(
        name="hrf",
        targets=(sel("blood.flow", "blood.volume", "blood.deoxyhemoglobin",
                     band=HEMODYNAMIC),),
        regions=(("vasculature", OnSupport("vascular_tree")),
                 ("watershed", WATERSHED),
                 ("brain", OnSupport("tissue"))),
        resolution=res(
            rule(OnSupport("vascular_tree"), 1.0, HEMODYNAMIC),
            rule(WATERSHED, 2.0, HEMODYNAMIC),
            rule(OnSupport("tissue"), 4.0, HEMODYNAMIC),
            default_mm=6.0, default_band=HEMODYNAMIC),
        fields=("blood", "metabolic", "neural", "material"),
        anatomy=("vascular_territories", "cortical_areas"),
        topologies=("vascular", "metabolic_exchange"),
        processes=("neurovascular_coupling", "vascular_flow", "bold_formation",
                   "tissue_exchange", "metabolism"),
        observations=("bold", "respiration", "asl_perfusion"),
        interventions=("respiratory_challenge",),
        devices=(DeviceSpec("scanner", "scanner_element", "scanner", "observe"),),
        subject=SubjectSpec(),
        window=Window(n=512, dt=0.5),
        bands=(("blood", HEMODYNAMIC), ("metabolic", HEMODYNAMIC),
               ("neural", DRIVE)),
        budget=Budget(max_state_variables=1_000_000),),
    fit_sources=("ds003192", "breath-hold-hypercapnia-cvr-datasets",
                 "asl-bold-breath-hold-datasets", "ds004873",
                 "physiological-etco2-recordings", "cvrmap"),
    eval_sources=("7t-qsm-venograms", "corr-reliability", "ds006072",
                  "hcp-task-vision-maps"),
    teachers=("haemodynamic-model-families",),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "cerebrovascular reactivity per voxel, which is exactly what a breath-hold "
        "or co2 challenge measures and is the best-determined hemodynamic quantity "
        "in the library",
        "arterial transit delay across vascular territories, from the asl sources",
        "the separation of the vascular and neurovascular gains, which only the "
        "hypercapnia data make identifiable at all",),
    prior_dominated=(
        "the microvascular versus macrovascular split within a voxel.  the venogram "
        "eval source can falsify a bad split but there are too few subjects to fit "
        "one",
        "the post-stimulus undershoot mechanism -- volume recovery versus sustained "
        "oxygen consumption.  both reproduce the data and the corpus does not "
        "separate them, so the mechanism reported is the prior",
        "hrf variation with age and vascular disease; `adni` and `oasis-3` are "
        "negative controls elsewhere in this library and are not fitted here",),))


FNIRS_FORWARD = register(NamedModel(
    id="fnirs_forward",
    doc="""near-infrared optical density at scalp optodes from cortical blood state.

    optically the analogue of `eeg_forward`, and coarse for an even blunter reason:
    photon transport in tissue is diffusive, so the sensitivity profile between a
    source and a detector 30 mm apart is a banana about 15 mm deep and 10-20 mm
    wide.  there is no source configuration that changes that.  a 5 mm cortical
    grid is already finer than the instrument's point spread function, and 2 mm
    would be materializing a distinction the measurement provably averages over.

    **but the head layers get 1 mm, and that is the whole game.**  the light passes
    twice through scalp and skull, and scalp blood flow -- which is large,
    pulsatile, and utterly unrelated to brain activity -- contributes a majority of
    the signal change in a typical channel.  scalp and skull thickness vary by
    several millimetres across the head, so the partial pathlength factor varies
    with them.  spending resolution on the extracerebral layers rather than on
    cortex is the correct allocation here and it inverts the intuition that
    resolution belongs near the target.

    **B goes to 2 Hz, unlike every other model in this group.**  fnirs samples at
    10 Hz and cannot separate brain from scalp by filtering, so the cardiac pulse
    near 1 Hz and the mayer wave near 0.1 Hz are *state to be modelled*, not noise
    to be removed.  a HEMODYNAMIC-band materialization would alias them.""",
    request=MaterializationRequest(
        name="fnirs-forward",
        targets=(sel("blood.oxygenation", "blood.deoxyhemoglobin", "blood.volume",
                     band=OPTICAL),
                 sel("device.channel_gain", band=OPTICAL)),
        regions=(("scalp_layers", OnSupport("head_volume")),
                 ("gyral_crowns", OnSupport("cortical_surface"))),
        resolution=res(
            rule(Near("optodes", 25.0), 1.0, OPTICAL),
            rule(OnSupport("head_volume"), 2.0, OPTICAL),
            rule(OnSupport("cortical_surface"), 5.0, OPTICAL),
            default_mm=10.0, default_band=OPTICAL),
        fields=("blood", "metabolic", "material", "neural", "device"),
        anatomy=("cortical_areas", "vascular_territories"),
        topologies=("vascular", "metabolic_exchange", "device_coupling"),
        processes=("neurovascular_coupling", "vascular_flow", "bold_formation",
                   "device_coupling"),
        observations=("fnirs",),
        devices=(DeviceSpec("optodes", "sensor_array", "eeg_cap", "observe",
                            n_elements=64,
                            note="source-detector pairs, 30 mm nominal separation, "
                                 "plus short-separation channels for scalp regression"),),
        subject=SubjectSpec(),
        window=Window(n=1024, dt=0.1),
        bands=(("blood", OPTICAL), ("device", OPTICAL), ("neural", DRIVE)),
        budget=Budget(max_state_variables=400_000),),
    fit_sources=("ds004514", "ds007554", "ds007738", "duncan1995-forehead-dpf",
                 "prahl-haemoglobin-extinction", "scatterbrains"),
    eval_sources=("ds000116", "nm000254-natview", "corr-reliability"),
    teachers=("haemodynamic-model-families",),
    negative_controls=("bwin-000-synthetic-scaffold", "example-published-study"),
    constrained=(
        "the haemoglobin extinction coefficients and the differential pathlength "
        "factor, which `prahl-haemoglobin-extinction` and the duncan measurements "
        "fix outright -- these are calibration, not fitted brain state",
        "the scalp contribution per channel, when short-separation channels are "
        "present; without them it is not identifiable at all",
        "task-evoked oxy/deoxy amplitude in superficial cortex under the montage",),
    prior_dominated=(
        "cortical depth of the responding tissue.  the diffuse sensitivity profile "
        "cannot localise in depth and the 5 mm surface grid is a representation, "
        "not a resolution",
        "sulcal cortex.  the banana reaches gyral crowns and not much else, so "
        "anything reported in a sulcus is the prior propagated inward",
        "absolute haemoglobin concentration, as opposed to change -- continuous-wave "
        "fnirs measures only the latter and the baseline stays at its prior",),))


ASL_PERFUSION = register(NamedModel(
    id="asl_perfusion",
    doc="""cerebral blood flow in absolute units from arterial spin labelling.

    the one model in this group that is coarse on *both* axes, and the reason is
    signal-to-noise rather than physics.  an asl image is the difference of a
    labelled and a control acquisition, and the difference is about one percent of
    the control.  tens of averages are needed for a usable map, so the effective
    temporal resolution is tens of seconds and the native spatial resolution is
    3-4 mm; label decay with a 1.5 s blood t1 means a smaller voxel would arrive
    with less label, not more detail.  materializing at 1 mm would be materializing
    interpolation noise, and §1's criterion says so directly: the state is smooth at
    the scale the measurement integrates over.

    **where it is not coarse.**  arterial transit time varies by hundreds of
    milliseconds between vascular territories and by more than that in
    steno-occlusive disease, and the territory boundaries are sharp.  so the model
    materializes `vascular_territories` explicitly and puts a 2 mm rule on the
    watershed zones where neighbouring territories meet and transit time changes
    fastest -- the one place where coarse-graining genuinely fails.

    **B is near-DC.**  0-0.05 Hz.  asl measures a baseline, and its use is to
    calibrate `bold_forward`'s degenerate flow-consumption coupling with an
    absolute number rather than to track dynamics.""",
    request=MaterializationRequest(
        name="asl-perfusion",
        targets=(sel("blood.flow", "blood.volume", band=Band(0.0, 0.05)),
                 sel("metabolic.oxygen", band=DC)),
        regions=(("territories", OnSupport("tissue")),
                 ("watershed", WATERSHED)),
        resolution=res(
            rule(WATERSHED, 2.0, Band(0.0, 0.05)),
            rule(OnSupport("tissue"), 4.0, Band(0.0, 0.05)),
            default_mm=6.0, default_band=Band(0.0, 0.05)),
        fields=("blood", "metabolic", "material"),
        anatomy=("vascular_territories", "cortical_areas"),
        topologies=("vascular", "metabolic_exchange"),
        processes=("vascular_flow", "tissue_exchange", "metabolism"),
        observations=("asl_perfusion",),
        devices=(DeviceSpec("scanner", "scanner_element", "scanner", "observe",
                            note="pcasl with multiple post-label delays"),),
        subject=SubjectSpec(),
        window=Window(n=128, dt=4.0),
        bands=(("blood", Band(0.0, 0.05)), ("metabolic", DC)),
        budget=Budget(max_state_variables=200_000,
                      max_spectral_coefficients=2_000_000),),
    fit_sources=("pcasl-datasets", "ds004873", "nki-rockland", "corr-reliability",
                 "arterial-territory-atlas-liu2023"),
    eval_sources=("oasis-3", "adni", "asl-bold-breath-hold-datasets",
                  "monash-fdg-pet-fmri"),
    teachers=("haemodynamic-model-families",),
    negative_controls=("adni", "oasis-3", "bwin-000-synthetic-scaffold"),
    constrained=(
        "absolute grey-matter perfusion in ml/100g/min, which is the point of the "
        "measurement and is well reproduced across `corr-reliability`'s repeats",
        "arterial transit time per territory, when multiple post-label delays are "
        "acquired",
        "the grey-to-white flow ratio, though white-matter asl is close to the "
        "noise floor and this is the weaker half",),
    prior_dominated=(
        "anything at the microvascular scale.  the model materializes a vascular "
        "tree because other models in this library need it, and asl constrains "
        "only its territory-level aggregate",
        "the labelling efficiency, which is assumed from the sequence rather than "
        "measured per subject and multiplies every reported flow value",
        "oxygen extraction: asl gives flow, not consumption, and `metabolic.oxygen` "
        "here rides on the prior unless `quantitative_bold` is materialized "
        "alongside",),
    notes="`adni` and `oasis-3` appear as both eval and negative control: they are "
          "clinical cohorts whose perfusion differences must not be reproduced by a "
          "pipeline artifact, and a null there is a precondition for trusting any "
          "group difference this model reports.",))


QUANTITATIVE_BOLD = register(NamedModel(
    id="quantitative_bold",
    doc="""oxygen extraction fraction and cmro2 from qbold, qsm and calibrated bold.

    the model that tries to close the loop `bold_forward` leaves open.  bold is a
    signal about deoxyhemoglobin, which is flow times extraction times volume; a
    bold experiment alone cannot separate the three, and `bold_forward` says so
    under prior-dominated.  this model adds the measurements that can -- susceptibility
    mapping, reversible-transverse-relaxation-rate fitting, and a hypercapnic
    calibration -- and targets the metabolic components directly.

    **r(q) is set by the vessel-size distribution, not by the voxel.**  the qbold
    signal decay depends on the *distribution* of vessel radii inside a voxel,
    because a field perturbation around a capillary is in the diffusion-narrowing
    regime and one around a venule is in the static-dephasing regime, and those two
    have different signal dependences.  that is a nonlinearity whose average is not
    the average's image, in the most literal possible sense.  the model therefore
    materializes the vascular tree at 0.5 mm and the parenchyma at 2 mm, and its
    honest weakness is that no human dataset resolves the distribution it depends
    on.

    **B is dc.**  extraction fraction and cmro2 are baseline quantities here.
    a long window and a couple of coefficients per variable; the cost is entirely
    in the site count.""",
    request=MaterializationRequest(
        name="quantitative-bold",
        targets=(sel("metabolic.oxygen", "metabolic.consumption", band=DC),
                 sel("blood.oxygenation", "blood.deoxyhemoglobin",
                     "blood.oxygen_content", band=Band(0.0, 0.05)),),
        regions=(("vasculature", OnSupport("vascular_tree")),
                 ("parenchyma", OnSupport("tissue"))),
        resolution=res(
            rule(OnSupport("vascular_tree"), 0.5, Band(0.0, 0.05)),
            rule(OnSupport("cortical_surface"), 1.5, Band(0.0, 0.05)),
            rule(OnSupport("tissue"), 2.0, Band(0.0, 0.05)),
            default_mm=4.0, default_band=Band(0.0, 0.05)),
        fields=("blood", "metabolic", "material", "structural"),
        anatomy=("vascular_territories", "cortical_areas"),
        topologies=("vascular", "metabolic_exchange"),
        processes=("vascular_flow", "tissue_exchange", "metabolism",
                   "bold_formation"),
        observations=("structural_mri", "structural_mri", "bold",
                      "pet"),
        interventions=("respiratory_challenge",),
        devices=(DeviceSpec("scanner", "scanner_element", "scanner", "observe"),),
        subject=SubjectSpec(),
        window=Window(n=128, dt=2.0),
        bands=(("metabolic", DC), ("blood", Band(0.0, 0.05)),
               ("structural", Band(0.0, 0.001))),
        budget=Budget(max_state_variables=1_500_000,
                      max_spectral_coefficients=10_000_000),),
    fit_sources=("qbold-mqbold-datasets", "qsm-oef-datasets", "ds004873",
                 "monash-fdg-pet-fmri", "quantitative-relaxometry-datasets",
                 "raichle-metabolism"),
    eval_sources=("7t-qsm-venograms", "qsm-reconstruction-challenge",
                  "capillary-density-statistics", "microscopy-microvascular-networks"),
    teachers=("haemodynamic-model-families",),
    negative_controls=("bwin-000-synthetic-scaffold", "qsm-reconstruction-challenge"),
    constrained=(
        "whole-brain and grey-matter mean oef, which several independent methods "
        "in this list agree on to within a few percent",
        "venous oxygenation in large vessels, which qsm measures nearly directly",
        "the absolute scale of cmro2 through the fdg-pet cross-calibration",),
    prior_dominated=(
        "the intra-voxel vessel radius distribution, on which the entire qbold "
        "inversion rests.  it is taken from `capillary-density-statistics` and "
        "microscopy of *other* brains and does not move; every regional oef "
        "difference this model reports inherits that assumption",
        "the diffusion regime boundary between capillary and venular dephasing, a "
        "modelling choice with no per-subject evidence",
        "regional cmro2 in white matter, where all of these measurements are at "
        "their noise floor",),))
