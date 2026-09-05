"""neuromodulation: a modulator multiplies a target population's input-output
gain, rather than injecting current into it.

that sentence is the whole design, and it is the reason this is the one process
in the file that sets ``writes="parameters"``.  ARCHITECTURE.md §4 allows a
process to apply pressure to another process's theta,

    dtheta_p' += f(.)

and says it needs no new mechanism because theta is already part of the
process schema.  neuromodulation is the case that motivates it.  a
noradrenergic projection does not drive cortex; it changes what cortical drive
does -- the slope of the population transfer function, the balance between
recurrent and afferent excitation, the strength of adaptation.  a model that
represents that as an added current gets the empirical signature exactly
backwards: added current shifts a response curve, changed gain rotates it, and
the two are distinguishable in any experiment that varies stimulus contrast.

the practical consequence is that this process names *process ids* rather than
only components.  what it modulates is `local_excitation`'s slope, and
`tract_propagation`'s long-range gain, and `plasticity`'s learning rate -- and
those are theta, not state, so the declaration has to be able to point at them.
`targets` is that pointer.

the four ascending systems are declared together rather than one per process
because they share a mechanism and differ only in constants: a small brainstem
or basal-forebrain nucleus projects diffusely, releases into the extracellular
space, and acts through g-protein-coupled receptors on a timescale of hundreds
of milliseconds to tens of seconds.  what differs is where the receptors are
and which way the gain moves, and that is per-partition parameters rather than
separate ontology.

one honest bookkeeping note.  the modulator concentration this process reads is
written by `transmitter_dynamics`, not here.  release and clearance are the same
vesicular-and-transporter rate law for dopamine as for glutamate, so putting
them in one place avoids duplicating the kinetics; what is left here is the
part that is genuinely different, which is that the concentration acts on a
parameter instead of on a state.
"""

from __future__ import annotations

import numpy as np

from ibm.processes.base import (
    double_exponential,
    implementation,
    low_pass,
    parallel,
    process,
)
from ibm.registry import Form
from ibm.vocabulary import (
    Anat,
    Band,
    LFP,
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

#: the modulator concentration band.  narrow on purpose: a dopamine field with
#: beta-band structure would be an artefact of the declaration rather than a
#: fact about volume transmission, which has a diffusion-limited corner well
#: below 1 Hz over any distance a materialization resolves.
MODULATOR = Band(0.0, 2.0)

#: the source nuclei fire in the ordinary electrophysiological band even though
#: what they release does not, and the distinction matters: locus coeruleus
#: phasic bursts are a sub-second event whose *effect* is a seconds-long gain
#: change.  reading the source at LFP bandwidth keeps the burst; the modulator
#: kernel is what turns it into something slow.
SOURCE = LFP

#: the ascending modulatory nuclei.  the cholinergic basal forebrain is not a
#: brainstem structure and has no home in the declared partitioning systems, so
#: it is labelled within `brainstem_nuclei` here with this comment rather than
#: quietly dropped -- an acknowledged deviation, not an anatomical claim.
LOCUS_COERULEUS = Anat("brainstem_nuclei", "locus_coeruleus")
RAPHE = Anat("brainstem_nuclei", "raphe_dorsal")
VTA = Anat("brainstem_nuclei", "vta")
SNC = Anat("brainstem_nuclei", "substantia_nigra_pars_compacta")
BASAL_FOREBRAIN = Anat("brainstem_nuclei", "basal_forebrain")

MODULATORY_SOURCES = LOCUS_COERULEUS | RAPHE | VTA | SNC | BASAL_FOREBRAIN


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def modulator_gain_transfer(basis, tau_receptor_s: float = 0.3,
                            tau_second_messenger_s: float = 3.0, gain: float = 1.0):
    """modulator concentration -> multiplicative gain change on a target's theta.

    two poles, because there are two steps and they differ by an order of
    magnitude.  the receptor is g-protein coupled, so binding to g-protein
    activation takes hundreds of milliseconds rather than the microseconds an
    ionotropic receptor takes; the downstream effect -- channel
    phosphorylation, and for the dopamine and noradrenaline systems a cyclic-amp
    cascade -- takes seconds.  a single pole put anywhere between them
    misrepresents the phase at exactly the frequencies at which modulatory
    effects are measured.

    the output is dimensionless and centred on zero: it is the *change* in a
    log gain, so that composing two modulators multiplies their effects rather
    than adding them.  that is the right composition rule for gain, and it is
    also what keeps a gain from going negative under strong modulation, which a
    linear additive form would cheerfully allow.

    where it breaks: gain modulation saturates, hard.  receptor occupancy is a
    hill function and above its EC50 more transmitter does almost nothing,
    which is why the dose-response curve of essentially every drug acting on
    these systems is sigmoid rather than linear.  this form is the small-signal
    expansion about the ambient concentration and is wrong for anything
    pharmacological.
    """
    return gain * double_exponential(basis, tau_receptor_s, tau_second_messenger_s)


def tonic_phasic_transfer(basis, tau_tonic_s: float = 20.0, tau_phasic_s: float = 0.5,
                          phasic_fraction: float = 0.5, gain: float = 1.0):
    """the same drive split into a fast and a slow modulatory component.

    the locus coeruleus is the clearest case and the reason this exists as a
    separate form.  its tonic discharge rate sets a slow, seconds-to-minutes
    arousal level; its phasic bursts, locked to behaviourally salient events,
    produce a sub-second gain transient on top.  the two have opposite
    relationships to performance -- moderate tonic with strong phasic is the
    good regime, high tonic with weak phasic is distractible -- so a model
    carrying only one of the timescales cannot express the distinction that
    the whole literature on this system is about.

    two parallel paths with very different time constants, weighted.  it is
    still linear, so it cannot produce the *anticorrelation* between tonic and
    phasic that the physiology shows; that lives in the source nucleus's own
    dynamics, which this process reads rather than models.
    """
    tonic = (1.0 - phasic_fraction) * low_pass(basis, tau_tonic_s)
    phasic = phasic_fraction * low_pass(basis, tau_phasic_s)
    return gain * (tonic + phasic)


def cholinergic_routing_transfer(basis, tau_muscarinic_s: float = 1.0,
                                 tau_nicotinic_s: float = 0.05, gain: float = 1.0):
    """acetylcholine's two receptor families, which pull in opposite directions.

    nicotinic receptors are ionotropic and fast, and in cortex they sit
    preferentially on vip interneurons and on thalamocortical terminals --
    so their net effect is to boost afferent drive and to disinhibit.
    muscarinic receptors are metabotropic and slow, and their best-documented
    cortical effect is to *suppress* intracortical synaptic transmission while
    reducing spike-frequency adaptation.

    the combination is not "more excitation": it is a shift in the balance
    between feedforward and recurrent processing, which is the mechanistic
    version of what attention is usually said to do.  a lumped cholinergic gain
    term cannot represent a re-weighting, only a scaling, which is why the two
    receptor paths are kept apart here.
    """
    nicotinic = low_pass(basis, tau_nicotinic_s)
    muscarinic = low_pass(basis, tau_muscarinic_s)
    return gain * parallel(nicotinic, -muscarinic)


# ---------------------------------------------------------------------------
# nonlinear rate laws
# ---------------------------------------------------------------------------


def hill_occupancy_gain(x, theta) -> dict:
    """receptor occupancy as a hill function of concentration.

    the saturating form, and the only one that can be asked a pharmacological
    question.  occupancy is ``c^n / (c^n + ec50^n)``, so the gain change it
    produces is bounded, its sensitivity is maximal near the EC50, and a
    doubling of concentration at high occupancy does essentially nothing --
    all three of which are what dose-response data actually look like.

    it returns pressure on the *activity* of the modulated populations, which
    is the state whose input-output relation the gain change is about; the
    parameter pressure that this process is really for is applied to the
    processes named in `targets` and is carried by the same occupancy.
    """
    out = {}
    n = float(theta.get("hill_coefficient", 1.5))
    for cid, conc_id, ec50_key, sign_key in (
        ("neural.exc.activity", "extracellular.noradrenaline", "na_ec50_nm", "na_sign"),
        ("neural.vip.activity", "extracellular.acetylcholine", "ach_ec50_nm", "ach_sign"),
    ):
        c = np.maximum(np.asarray(x[conc_id], dtype=float), 0.0)
        ec50 = float(theta.get(ec50_key, 100.0))
        sign = float(theta.get(sign_key, 1.0))
        occupancy = c ** n / (c ** n + ec50 ** n)
        out[cid] = sign * float(theta.get("gain_span", 0.5)) * occupancy
    return out


# ---------------------------------------------------------------------------
# neuromodulation
# ---------------------------------------------------------------------------

NEUROMODULATION = process(
    id="neuromodulation",
    doc="""ascending modulatory systems set the gain of cortical and thalamic
    processing rather than driving it.

    inputs are the firing of the source nuclei -- locus coeruleus, dorsal
    raphe, ventral tegmental area and substantia nigra pars compacta, and the
    cholinergic basal forebrain -- together with the extracellular modulator
    concentrations those nuclei produce, and adenosine, which is a modulator
    with no nucleus behind it at all.

    outputs are the state whose input-output relation is being changed, and
    `writes` is `"parameters"`, which is the declaration that matters.  the
    pressure this process applies does not go onto those components; it goes
    onto the theta of the processes in `targets`.  concretely: it multiplies
    the sigmoid slope in `local_excitation:wilson_cowan_adaptive`, the loop
    gain in `local_excitation:ei_loop_lti`, the long-range gain in
    `tract_propagation`, the release probability in `transmitter_dynamics`, and
    the learning rate in `plasticity`.  the output selectors say what the
    modulation is *about*; the target list says where the arithmetic lands.

    why this distinction is worth the extra machinery.  a modulator represented
    as an injected current adds a constant to a population's drive, which
    shifts its response curve horizontally: every stimulus gets the same
    absolute increment, and the effect on a weak stimulus equals the effect on
    a strong one.  a modulator represented as gain multiplies, which rotates
    the curve: weak stimuli are affected little and strong ones a lot.  those
    predictions differ in sign for the interaction term of any contrast-varying
    experiment, and the multiplicative one is what is measured.  the additive
    version can be fitted to a single contrast and will then be wrong at every
    other one, which is the specific kind of error a model with an explicit
    ontology is supposed to make impossible to write down.

    where it breaks.  first, this is a diffuse-projection description: it
    assumes a nucleus's output is one scalar released everywhere it projects,
    and the dopaminergic and cholinergic systems in particular have
    topographically specific projections that carry different signals to
    different targets.  second, receptor subtypes are collapsed -- D1 and D2
    have opposite effects on cortical persistent activity, alpha-1 and alpha-2
    adrenoceptors have different affinities and therefore opposite
    concentration-dependence, and this process carries one gain per modulator
    with a sign.  the inverted-U dose-response that is the signature of the
    dopamine and noradrenaline systems is a *consequence* of two receptor
    subtypes with different EC50s, and a single-receptor form cannot produce it
    at any parameter setting.""",
    inputs=(
        within("neural", "exc.activity", region=MODULATORY_SOURCES, band=SOURCE),
        within("neural", "inh.activity", region=MODULATORY_SOURCES, band=SOURCE),
        within("extracellular", "dopamine", "serotonin", "acetylcholine", "noradrenaline",
               band=MODULATOR),
        within("extracellular", "adenosine", band=MODULATOR),
    ),
    outputs=(
        within("neural", "exc.activity", "inh.activity", band=LFP),
        within("neural", "pv.activity", "sst.activity", "vip.activity", band=LFP),
        within("neural", "exc.adaptation", band=Band(0.0, 50.0)),
    ),
    topology="neuromodulatory_projection",
    writes="parameters",
    targets=(
        "local_excitation",
        "local_inhibition",
        "laminar_propagation",
        "lateral_cortical_propagation",
        "tract_propagation",
        "thalamocortical_coupling",
        "transmitter_dynamics",
        "plasticity",
    ),
    timescale_s=1.0,
    validity=Validity(
        min_spacing_mm=0.5, max_spacing_mm=20.0, band=MODULATOR,
        note="diffuse projections are spatially coarse by construction, so materializing "
             "this below ~0.5 mm asserts a projection specificity no atlas supplies.  above "
             "~20 mm the target region spans areas whose receptor densities differ several "
             "fold, and receptor density is the thing that decides what a uniform modulator "
             "concentration actually does.",
    ),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"neuromodulatory", "gain", "parameters"}),
    notes="writes parameters, not state.  the output selectors name the populations whose "
          "input-output gain is modulated so that the scope of the modulation is declared "
          "and checkable; the pressure itself lands on the theta of the processes in "
          "targets.  the modulator concentrations are written by transmitter_dynamics, "
          "which owns release and clearance for every vesicular transmitter.",
)

implementation(
    name="gain_cascade_lti",
    process="neuromodulation",
    doc="""concentration to log-gain through a receptor pole and a second-messenger
    pole.

    the default.  it is linear in the log of the gain, which is the right
    linearization for a multiplicative effect: two modulators acting together
    compose by multiplying their gains, the gain cannot cross zero, and small
    concentration changes produce proportional rather than absolute effects.

    the time constants are the defensible part -- g-protein coupled receptor
    activation and downstream cascades are measured in hundreds of milliseconds
    and seconds respectively, and they are the reason modulatory effects
    outlast the burst that caused them.  the gains are the weak part: how much
    a given ambient dopamine concentration changes cortical recurrent gain is
    not a measured number in any species, and the prior says so.""",
    form=Form.LTI,
    transfer=modulator_gain_transfer,
    params={
        "tau_receptor_s": lognormal(0.3, 2.0, units="s", provenance=Provenance.LITERATURE,
                                    source="g-protein coupled receptor activation, "
                                           "hundreds of milliseconds",
                                    note="three orders of magnitude slower than an ionotropic "
                                         "receptor, which is the mechanistic reason modulation "
                                         "and transmission are different processes"),
        "tau_second_messenger_s": lognormal(3.0, 2.5, units="s", provenance=Provenance.LITERATURE,
                                            note="cyclic-amp and phosphorylation cascades; this "
                                                 "is why a phasic burst produces a gain change "
                                                 "lasting seconds"),
        "gain": weak(0.3, 6.0, units="log gain per uM",
                     note="the modulatory sensitivity.  no measurement of this exists at "
                          "mesoscale in any species; every number in the literature is a "
                          "fitted parameter of some other model"),
        "noradrenaline_sign": normal(1.0, 0.3, units="dimensionless",
                                     provenance=Provenance.LITERATURE,
                                     note="net gain increase at moderate concentration; the "
                                          "sign genuinely inverts at high concentration through "
                                          "alpha-1 recruitment, which this form cannot express"),
        "acetylcholine_sign": normal(1.0, 0.4, units="dimensionless",
                                     provenance=Provenance.LITERATURE,
                                     note="positive on thalamocortical drive and negative on "
                                          "intracortical transmission; one signed scalar is a "
                                          "compromise between two opposite measured effects"),
        "adenosine_sign": normal(-1.0, 0.3, units="dimensionless",
                                 provenance=Provenance.LITERATURE,
                                 note="A1 receptors presynaptically inhibit release; adenosine "
                                      "accumulating with metabolic expenditure is the best "
                                      "available mechanism for homeostatic sleep pressure, and "
                                      "caffeine's entire pharmacology is blocking this term"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="mcCormick 1992 neurotransmitter actions in thalamus and cortex; "
           "aston-jones & cohen 2005 adaptive gain",
)

implementation(
    name="tonic_phasic_lti",
    process="neuromodulation",
    doc="""the modulatory drive split into a slow tonic and a fast phasic component.

    worth carrying separately because the two components of the same
    projection have different, and in the noradrenergic case opposite,
    behavioural correlates, and a single time constant forces a model to choose
    one.  a materialization asking about arousal or sleep-wake state wants the
    tonic limb; one asking about a trial-locked response to a salient stimulus
    wants the phasic one; and the interesting claims are about their ratio.

    linear, so it cannot produce the tonic-phasic anticorrelation itself --
    that is a property of the source nucleus, which this process reads as an
    input rather than models.""",
    form=Form.LTI,
    transfer=tonic_phasic_transfer,
    params={
        "tau_tonic_s": lognormal(20.0, 3.0, units="s", provenance=Provenance.LITERATURE,
                                 note="arousal-level changes over tens of seconds to minutes; "
                                      "the pupil-linked index of it has roughly this constant"),
        "tau_phasic_s": lognormal(0.5, 2.0, units="s", provenance=Provenance.LITERATURE,
                                  source="locus coeruleus phasic response, ~100 ms burst with a "
                                         "several-hundred-millisecond gain effect"),
        "phasic_fraction": uniform(0.0, 1.0, units="dimensionless", provenance=Provenance.WEAK,
                                   note="the ratio is the quantity of interest and is exactly "
                                        "what varies with task engagement, so a flat prior is "
                                        "the honest one"),
        "tonic_rate_hz": normal(2.0, 1.0, units="Hz", provenance=Provenance.LITERATURE,
                                source="locus coeruleus tonic discharge in waking primates, "
                                       "1-3 Hz; near zero in rem sleep"),
        "gain": weak(0.3, 6.0, units="log gain per uM"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="aston-jones & cohen 2005",
)

implementation(
    name="cholinergic_routing_lti",
    process="neuromodulation",
    doc="""acetylcholine as a re-weighting of feedforward against recurrent drive.

    the nicotinic path is fast and boosts thalamocortical and disinhibitory
    (vip) drive; the muscarinic path is slow and suppresses intracortical
    transmission.  their sum is not a scaling but a rotation of the balance
    between afferent and recurrent processing, which is the specific thing this
    implementation exists to represent and which no single-signed cholinergic
    gain can.

    the mechanism is well established in slice and the mesoscale magnitudes are
    not, so the kinetics carry literature provenance and the weights do not.""",
    form=Form.LTI,
    transfer=cholinergic_routing_transfer,
    params={
        "tau_nicotinic_s": lognormal(0.05, 2.0, units="s", provenance=Provenance.LITERATURE,
                                     note="ionotropic and fast, and desensitizing -- which this "
                                          "linear form does not represent"),
        "tau_muscarinic_s": lognormal(1.0, 2.5, units="s", provenance=Provenance.LITERATURE,
                                      source="m1/m2 receptor mediated effects on cortical "
                                             "synaptic transmission and adaptation"),
        "feedforward_boost": weak(0.5, 5.0, units="log gain",
                                  note="nicotinic enhancement of thalamocortical drive"),
        "recurrent_suppression": weak(0.5, 5.0, units="log gain",
                                      note="muscarinic suppression of intracortical "
                                           "transmission; the two together are the rotation"),
        "gain": weak(0.3, 6.0),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.WEAK,
    source="hasselmo & sarter 2011 modes and models of forebrain cholinergic neuromodulation",
)

implementation(
    name="hill_occupancy_rate",
    process="neuromodulation",
    doc="""saturating receptor occupancy: the form a drug can be asked about.

    the linear implementations are expansions about the ambient concentration
    and are meaningless at pharmacological doses, which is a serious limitation
    because pharmacology is one of the few sources of causal evidence about
    these systems.  a hill function with an EC50 gives a bounded, sigmoid
    dose-response and puts the model's maximum sensitivity where the data say
    it is.

    the hill coefficient above one encodes receptor cooperativity, which is
    real for several of these receptors and is also doing duty here for the
    steepness that receptor-density heterogeneity within a materialized
    position would produce anyway -- an honest conflation, since the two are
    not separable by any measurement at this scale.

    it still carries one receptor per modulator, so it still cannot produce an
    inverted-U dose-response.  getting that requires two receptor subtypes with
    different EC50s and opposite signs, which would be two components and two
    parameter sets, and the evidence that would separate them at mesoscale does
    not exist.""",
    form=Form.RATE,
    fn=hill_occupancy_gain,
    params={
        "hill_coefficient": lognormal(1.5, 1.5, units="dimensionless",
                                      provenance=Provenance.WEAK,
                                      note="cooperativity plus unresolved receptor-density "
                                           "heterogeneity, conflated because nothing at this "
                                           "scale separates them"),
        "na_ec50_nm": lognormal(100.0, 5.0, units="nM", provenance=Provenance.WEAK,
                                note="adrenoceptor affinities differ by two orders of magnitude "
                                     "between subtypes, and this single value straddles them"),
        "ach_ec50_nm": lognormal(300.0, 5.0, units="nM", provenance=Provenance.WEAK),
        "da_ec50_nm": lognormal(50.0, 5.0, units="nM", provenance=Provenance.LITERATURE,
                                note="D2 receptors are high-affinity and largely occupied at "
                                     "ambient concentration; D1 are low-affinity and respond to "
                                     "phasic transients.  that split is what produces the "
                                     "inverted-U this form cannot"),
        "gain_span": uniform(0.0, 2.0, units="log gain", provenance=Provenance.WEAK,
                             note="the full range of gain change between zero and saturating "
                                  "occupancy"),
        "na_sign": normal(1.0, 0.3, units="dimensionless", provenance=Provenance.LITERATURE),
        "ach_sign": normal(1.0, 0.4, units="dimensionless", provenance=Provenance.LITERATURE),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.WEAK,
)

implementation(
    name="learned_modulatory_gain",
    process="neuromodulation",
    doc="""a learned map from the modulator concentrations to the target parameter
    changes.

    the case for it here is stronger than anywhere else in this package,
    because the analytic forms above are almost entirely priors: the kinetics
    are measured and essentially every gain is weak or speculative.  the
    receptor-density maps that would make the gains regional are available from
    pet, so a learned f conditioned on a receptor-density embedding is the
    natural way to bring that evidence in -- which is exactly the
    ARCHITECTURE.md §4 pattern of the topology saying what may interact and the
    learned f saying how strongly it does.

    the parameters it writes are another process's theta, which is worth
    stating because it means the learned module has to be differentiated
    through the target process as well as through itself.  that is a real
    implementation cost and the reason the modulatory pathway is the last part
    of a materialization anyone should try to learn.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "receptor_density_sensitivity": speculative(1.0, 10.0,
                                                    note="how strongly the modulatory gain "
                                                         "tracks pet-measured receptor density; "
                                                         "plausible, and untested at this scale"),
        "residual_gain": weak(0.3, 5.0, note="zero recovers the analytic gain cascade"),
    },
    tying=Tying.EMBEDDING,
    state_dependent_weights=True,
    provenance=Provenance.SPECULATIVE,
)


__all__ = ["NEUROMODULATION"]
