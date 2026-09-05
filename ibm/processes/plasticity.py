"""plasticity: activity history, neuromodulatory context and structural state
writing back onto the structure and onto the parameters of the processes that
produced them.

this is the one process in the inventory whose output is a parameter as much as
a state, and ARCHITECTURE.md §4 says so explicitly.  it needs no new mechanism:
theta is already part of the process schema, pressure on theta composes by
summation exactly as pressure on state does, and so `writes="parameters"` with
`targets` naming `local_excitation` and `tract_propagation` is the whole of the
machinery.  what it buys is that a learned synaptic weight and a measured
synaptic density are the same declaration seen from two sides -- the structural
component is what an anatomical method would measure, the target parameter is
what the dynamics actually use -- instead of two unrelated things that have to
be kept consistent by hand.

the two targets are chosen rather than exhaustive.  `local_excitation` carries
the recurrent cortical weights that hebbian and homeostatic rules move, and
`tract_propagation` carries the long-range gains and conduction delays that
activity-dependent myelination moves -- and the second of those is the reason
this process cannot be purely local.  the inventory in §5 gives plasticity the
topology "local + tractometric" for exactly this reason and a process carries one
topology, so it is declared over `local` and the tractometric half rides on the
parameter write into `tract_propagation`.  that is a real limitation and not a
notational one: the *structural* consequence of myelination is written to a local
component, and the fact that the change belongs to a tract rather than to a point
is recovered only by whichever process reads it.

the timescales are the substance of this file and they span eight decades.
coincidence detection is a twenty-millisecond window; the eligibility trace that
holds a synapse open for a neuromodulatory signal is one to two seconds;
consolidation into a protein-synthesis-dependent form is an hour; homeostatic
scaling is a day or two; myelination and structural remodelling are days to
weeks.  `timescale_s` is set to a thousand seconds -- the consolidation end --
because that is the scale at which this process's *output* moves, and the fast
windows appear as parameters of the rule rather than as the process's own rate.

that choice is where the lossy conversion bites, and it deserves to be stated
sharply.  neural state is spectral and structural state is scalar, so the
registered ``spectral -> scalar`` conversion is inserted here.  it keeps the
window mean and its variance and destroys phase.  but the entire content of a
spike-timing rule is phase: whether the presynaptic component leads or lags the
postsynaptic one by ten milliseconds is the difference between potentiation and
depression, and after the collapse that information is simply gone.  an
implementation of stdp in this graph is therefore only correct if it computes the
pre-post correlation *in the spectral domain*, where the relative phase still
exists, and collapses afterwards.  one that reads the converted scalars and
multiplies them has computed a rate-covariance rule wearing an stdp parameter
name, and will be wrong in the specific, well-documented direction of predicting
potentiation where the timing was reversed.  the selectors below therefore read
across the full electrophysiological band on purpose: narrowing them to the
structural band before the rule runs would destroy the thing the rule is about.
"""

from __future__ import annotations

from ibm.processes.base import implementation, process
from ibm.registry import Form
from ibm.vocabulary import (
    Band,
    HEMODYNAMIC,
    Provenance,
    STRUCTURAL,
    Tying,
    Validity,
    lognormal,
    normal,
    speculative,
    uniform,
    weak,
    within,
)

#: activity history is read across the whole electrophysiological band, before
#: any collapse.  this is not laziness about bandwidth -- it is the opposite.
#: coincidence detection lives in the phase relationships between components up
#: to a few hundred hertz, and a rule that reads a narrower band than this is
#: computing something other than what its parameters are named after.
ACTIVITY_HISTORY = Band(0.0, 300.0)

#: neuromodulatory context is read over the haemodynamic band and no wider.
#: phasic dopamine is a sub-second event, but what reaches a synapse through
#: volume transmission is already low-pass filtered by diffusion and uptake, and
#: the third factor a plasticity rule uses is the envelope rather than the
#: transient.
MODULATORY_CONTEXT = HEMODYNAMIC


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def eligibility_trace_transfer(basis, tau_eligibility_s: float = 1.0,
                               tau_consolidation_s: float = 3600.0):
    """a coincidence signal held open, then consolidated: two lags in series.

        H = 1 / ((1 + i omega tau_e) (1 + i omega tau_c))

    the fast pole is the eligibility trace -- the synapse-local tag that a
    coincidence leaves behind and that decays over a second or two if no
    neuromodulatory signal arrives.  it is what resolves the credit-assignment
    problem that a millisecond rule and a second-scale reward would otherwise
    have; the measured window is narrow and asymmetric, with dopamine arriving
    0.3 to 2 s after spine activation being effective and dopamine arriving
    before it being much less so.

    the slow pole is consolidation into the protein-synthesis-dependent form.
    the separation between the two is four orders of magnitude, which is why this
    is written as an explicit cascade rather than as one effective time constant:
    collapsing them would misplace both the credit-assignment window and the
    consolidation time, and it is the ratio between them that determines what a
    given reinforcement schedule can teach.
    """
    omega = basis.omega
    return 1.0 / ((1.0 + 1j * omega * tau_eligibility_s)
                  * (1.0 + 1j * omega * tau_consolidation_s))


def homeostatic_transfer(basis, tau_s: float = 8.6e4, gain: float = 1.0):
    """negative feedback towards an activity set point, H = -g / (1 + i omega tau).

    synaptic scaling multiplies every synapse onto a cell by a common factor to
    return its firing rate towards a target, over a day or two.  the sign is the
    whole point and is why this is written as an explicit negative: it is the
    only stabilizing term in a file otherwise full of positive feedback, and a
    hebbian rule without it diverges.

    the slowness is not incidental either.  the separation between the hebbian
    timescale and this one is what lets correlations be learned before the
    normalization erases them; making the scaling faster does not merely stabilize
    the network more, it stops it learning.
    """
    omega = basis.omega
    return -gain / (1.0 + 1j * omega * tau_s)


# ---------------------------------------------------------------------------
# plasticity
# ---------------------------------------------------------------------------

PLASTICITY = process(
    id="plasticity",
    doc="""activity, neuromodulation and existing structure changing synaptic,
    dendritic and axonal structure -- and, through it, the parameters of the
    processes that carry the dynamics.

    the inputs are the three factors a modern account of plasticity needs and
    nothing more.  pre- and postsynaptic activity and the synaptic conductances
    give the coincidence; `neural.exc.adaptation` and `extracellular.ca` give the
    slow integrated activity that a sliding threshold or a calcium-dependent rule
    needs, and calcium is read rather than assumed because the direction of the
    weight change is a function of postsynaptic calcium concentration in every
    biophysical account of it -- moderate calcium depresses, high calcium
    potentiates, and that single nonlinearity generates the stdp curve, the bcm
    curve and the frequency dependence of ltp as consequences rather than as
    separate rules.  the neuromodulators give the third factor: dopamine gating
    consolidation, acetylcholine and noradrenaline gating whether the cortex is
    in a state that learns at all.  the structural components are read as well as
    written because plasticity is history-dependent -- a dense synaptic
    population saturates, and metaplasticity is the statement that the rule
    depends on what the rule has already done.

    the outputs are the structural components and the parameters of
    `local_excitation` and `tract_propagation`.  writing both is deliberate and
    is the point of `writes="parameters"`: the synaptic density is what a
    stereological or receptor-autoradiography measurement would see, and the
    target process's gain is what the simulated dynamics use, and they are one
    quantity.  a materialization that fits electrophysiology and one that fits
    post-mortem anatomy are therefore constraining the same posterior, which is
    the main thing this declaration is for.

    where it breaks, in order.  every implementation below is a phenomenological
    rule with a biophysical story attached, and the stories are contested: the
    role of the sliding threshold, the interaction between hebbian and homeostatic
    terms, and whether synaptic scaling is even the mechanism operating in vivo
    are all live disputes.  none of the rules represent structural plasticity
    properly -- spines appear and disappear, and a continuous density is a poor
    description of a discrete process at low density.  the myelination leg is the
    weakest: activity-dependent oligodendrogenesis is established, its timescale
    is roughly known, and the quantitative map from activity to conduction
    velocity change is very nearly unmeasured.  and every rule here is written
    for excitatory synapses; inhibitory plasticity follows different rules, is
    read by nothing in this declaration, and its absence is a real hole given
    that inhibitory tuning is what keeps the recurrent circuit stable.""",
    inputs=(
        within("neural", "exc.activity", "inh.activity", band=ACTIVITY_HISTORY),
        within("neural", "exc.potential", "exc.ampa", "exc.nmda", band=ACTIVITY_HISTORY),
        within("neural", "exc.adaptation", band=ACTIVITY_HISTORY),
        within("neural", "pv.activity", "sst.activity", "vip.activity", band=ACTIVITY_HISTORY),
        within("extracellular", "ca", band=ACTIVITY_HISTORY),
        within("extracellular", "dopamine", "acetylcholine", "noradrenaline", "serotonin",
               band=MODULATORY_CONTEXT),
        within("structural", "synaptic_density", "dendritic_density", "axonal_density",
               band=STRUCTURAL),
        within("structural", "myelination", band=STRUCTURAL),
    ),
    outputs=(
        within("structural", "synaptic_density", "dendritic_density", band=STRUCTURAL),
        within("structural", "axonal_density", "myelination", band=STRUCTURAL),
    ),
    topology="local",
    writes="parameters",
    targets=("local_excitation", "tract_propagation"),
    timescale_s=1.0e3,
    validity=Validity(
        min_spacing_mm=0.05, max_spacing_mm=10.0, band=STRUCTURAL,
        note="a population-averaged rule over many synapses.  below ~50 um the population is "
             "a handful of spines and a continuous density is the wrong description of a "
             "process that is discrete and bimodal at that scale; above ~10 mm the cell "
             "averages over populations with opposite selectivity, and the correlation the "
             "rule is computing is between things that are not connected to each other."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"plasticity", "structural", "slow", "contested"}),
    notes="reads spectral neural state and writes scalar structural state, so the registered "
          "spectral -> scalar conversion applies.  it destroys phase, which is the entire "
          "content of a timing-dependent rule; an implementation must compute its "
          "correlations before the conversion, not after.",
)

implementation(
    name="three_factor_stdp",
    process="plasticity",
    doc="""spike-timing-dependent potentiation and depression, held in an
    eligibility trace and gated by a neuromodulatory third factor.

    the standard account and the one with the most direct evidence behind it: the
    timing window was measured directly in paired recordings, the eligibility
    trace was measured directly by delivering dopamine at controlled delays after
    spine activation, and the two together resolve the credit-assignment problem
    that a millisecond rule and a second-scale outcome otherwise pose.

    it must be evaluated on the spectral state, before the scalar collapse.  the
    pre-post correlation at a lag of ten milliseconds is a phase relationship, and
    the conversion that produces the structural output destroys phase; computing
    the correlation first and converting the result is correct, and converting
    first and multiplying is a rate rule with stdp parameter names on it.  this
    is the single most important implementation note in the file.

    where it breaks: the pairwise timing rule fails at the firing rates cortex
    actually uses.  triplet and voltage-based rules were introduced precisely
    because the pairwise form does not reproduce the frequency dependence of
    plasticity above about 20 Hz, and this form inherits that failure.  it is also
    a rule about pairs of spikes in a model that carries population activity, so
    the correlation it computes is a population one and the mapping between the
    two involves an assumption about synchrony that is nowhere stated.""",
    form=Form.RATE,
    transfer=eligibility_trace_transfer,
    params={
        "tau_potentiation_ms": normal(17.0, 5.0, units="ms",
                                      provenance=Provenance.LITERATURE,
                                      source="bi & poo 1998 hippocampal culture pairing",
                                      note="pre-before-post window; cortical values are broadly "
                                           "similar and the spread here is across preparations"),
        "tau_depression_ms": normal(34.0, 10.0, units="ms",
                                    provenance=Provenance.LITERATURE,
                                    source="bi & poo 1998",
                                    note="the depression window is wider than the potentiation one, "
                                         "which is what makes uncorrelated activity net-depressing"),
        "depression_potentiation_ratio": normal(1.05, 0.30, units="dimensionless",
                                                provenance=Provenance.LITERATURE,
                                                note="amplitude ratio A-/A+; near one, so the net "
                                                     "sign of the rule is set by the window widths "
                                                     "and is therefore fragile"),
        "tau_eligibility_s": lognormal(1.0, 3.0, units="s",
                                       provenance=Provenance.LITERATURE,
                                       source="yagishita et al 2014; dopamine effective 0.3-2 s "
                                              "after spine activation",
                                       note="the window is asymmetric -- dopamine before the "
                                            "coincidence is much less effective -- and this single "
                                            "time constant does not express that"),
        "tau_consolidation_s": lognormal(3600.0, 3.0, units="s",
                                         provenance=Provenance.LITERATURE,
                                         source="protein-synthesis-dependent late ltp; synaptic "
                                                "tagging and capture over 1-2 h",
                                         note="four orders of magnitude above the eligibility trace; "
                                              "the ratio is what a reinforcement schedule has to "
                                              "respect"),
        "dopamine_gain": weak(1.0, 10.0, units="per uM",
                              note="the third factor's efficacy; the gating is established and the "
                                   "concentration-response is not"),
        "acetylcholine_gain": weak(0.5, 10.0, units="per uM",
                                   note="cholinergic gating of cortical plasticity is well "
                                        "demonstrated and quantitatively unpinned"),
        "weight_ceiling": uniform(0.0, 1.0, units="fraction of maximum",
                                  provenance=Provenance.WEAK,
                                  note="soft bound; without one a hebbian rule diverges, and where "
                                       "the bound sits changes the learned distribution's shape "
                                       "more than the learning rate does"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="bi & poo 1998; markram et al 1997; yagishita et al 2014; gerstner et al 2018",
)

implementation(
    name="calcium_bcm_sliding_threshold",
    process="plasticity",
    doc="""weight change as a function of postsynaptic calcium, with a threshold
    that slides with the history of activity.

    the reason to carry this alongside the timing rule is that it derives the
    timing rule rather than competing with it.  a single nonlinear dependence of
    the sign of plasticity on calcium concentration -- moderate depresses, high
    potentiates -- reproduces the stdp window, the frequency dependence of ltp
    and the bcm curve as consequences of the calcium transient's shape, which is
    a considerably stronger position than fitting three phenomenological rules.
    the sliding threshold is the metaplasticity that keeps it stable: the
    crossover point moves with the time-averaged postsynaptic activity, so a cell
    that has been active becomes harder to potentiate.

    the threshold's own timescale is the contested parameter.  it must be slower
    than the hebbian dynamics for stability and faster than the experiment for
    the effect to be visible, and the literature does not pin it within an order
    of magnitude, which is why its prior is wide.  the whole form also depends on
    a calcium concentration this ontology carries as an extracellular component
    when the quantity the rule needs is the postsynaptic intracellular transient
    in a spine -- a substitution that is defensible as a correlate and is not the
    same variable.""",
    form=Form.RATE,
    params={
        "calcium_ltd_threshold": lognormal(0.35, 2.0, units="uM",
                                           provenance=Provenance.LITERATURE,
                                           source="calcium control hypothesis; depression above a "
                                                  "low threshold",
                                           note="the intracellular spine value; what this process "
                                                "actually reads is extracellular calcium, and the "
                                                "map between them is an assumption"),
        "calcium_ltp_threshold": lognormal(1.0, 2.0, units="uM",
                                           provenance=Provenance.LITERATURE,
                                           note="potentiation above it; the ratio of the two "
                                                "thresholds is better constrained than either"),
        "tau_threshold_slide_s": speculative(1.0e4, 10.0, units="s",
                                             note="the metaplastic timescale; must sit between the "
                                                  "hebbian and homeostatic ones for the system to "
                                                  "work, and is not measured to better than an "
                                                  "order of magnitude"),
        "bcm_exponent": normal(2.0, 0.5, units="dimensionless",
                               provenance=Provenance.LITERATURE,
                               source="bienenstock, cooper & munro 1982; threshold moves as the "
                                      "square of mean activity",
                               note="the exponent is what makes the fixed point stable; values "
                                    "below 1 do not stabilize"),
        "nmda_calcium_gain": weak(1.0, 10.0, units="uM per unit nmda conductance",
                                  note="the voltage-dependent magnesium block is the actual "
                                       "coincidence detector and is lumped into this gain"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="bienenstock et al 1982; shouval et al 2002; graupner & brunel 2012",
)

implementation(
    name="homeostatic_scaling_lti",
    process="plasticity",
    doc="""multiplicative synaptic scaling towards an activity set point, as one
    slow negative-feedback lag.

    linear, cheap, exact at any timestep, and the only stabilizing term available
    -- which is why it is worth declaring separately rather than folding into the
    hebbian implementations.  a materialization can select it alongside one of
    them and get a network that learns without diverging, and the two rules'
    parameters stay separately identifiable because their timescales differ by
    two orders of magnitude.

    the two honest caveats are large.  first, whether synaptic scaling as
    measured in culture is what operates in vivo is genuinely disputed, and the
    experiments that would settle it are hard; the timescale below is from
    chronic activity blockade, which is not a manipulation the intact brain
    experiences.  second, multiplicative scaling is defined to preserve relative
    weights and therefore to preserve what hebbian learning stored -- an elegant
    property that is also exactly what makes it unable to explain how a network
    escapes a bad configuration.""",
    form=Form.LTI,
    transfer=homeostatic_transfer,
    params={
        "tau_s": lognormal(8.6e4, 3.0, units="s",
                           provenance=Provenance.LITERATURE,
                           source="turrigiano et al; synaptic scaling over 24-48 h of activity "
                                  "blockade",
                           note="two orders of magnitude slower than consolidation, which is what "
                                "lets correlations be learned before they are normalized away"),
        "target_activity_hz": lognormal(4.0, 3.0, units="Hz",
                                        provenance=Provenance.LITERATURE,
                                        source="cortical set-point firing rates are low and "
                                               "log-normally distributed across cells",
                                        note="a per-cell target in reality; declared per-partition "
                                             "here, which averages over a distribution spanning "
                                             "two orders of magnitude"),
        "gain": weak(1.0, 5.0, units="fractional weight change per Hz of error"),
        "scaling_is_multiplicative": uniform(0.0, 1.0, units="fraction",
                                             provenance=Provenance.WEAK,
                                             note="mixes multiplicative and subtractive scaling; the "
                                                  "two make different predictions about whether "
                                                  "learned structure survives normalization, and the "
                                                  "prior takes no side"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="turrigiano & nelson 2004",
)

implementation(
    name="activity_dependent_myelination",
    process="plasticity",
    doc="""axonal activity driving oligodendrogenesis and myelin thickening, and
    through it the conduction delays of `tract_propagation`.

    the slowest leg and the one with the most interesting consequence: myelin
    sets conduction velocity, conduction velocity sets delay, and delay sets which
    frequencies a recurrent loop can sustain.  so this implementation is the only
    route in the ontology by which experience changes the *timing* of long-range
    coupling rather than its strength -- which, if the oscillatory-synchrony
    accounts of large-scale coordination are right, is a more consequential kind
    of learning than weight change.

    that is also why it is the least defensible.  activity-dependent
    oligodendrogenesis is established, motor learning produces measurable white
    matter change within days, and the quantitative map from activity to a change
    in conduction velocity is essentially unmeasured -- so the gain below is
    speculative while the timescale is literature.  the form also has no way to
    represent that myelination is spatially discrete along an axon and that
    internode length matters as much as sheath thickness, both of which set
    velocity in ways a scalar density cannot express.""",
    form=Form.RATE,
    params={
        "tau_myelination_s": lognormal(6.0e5, 3.0, units="s",
                                       provenance=Provenance.LITERATURE,
                                       source="gibson et al 2014 and mckenzie et al 2014; "
                                              "learning-induced oligodendrogenesis within days",
                                       note="about a week; the diffusion-mri changes reported after "
                                            "a few hours of training are almost certainly a "
                                            "different and probably astrocytic mechanism"),
        "activity_to_velocity_gain": speculative(0.1, 30.0,
                                                 units="fractional velocity change per unit activity",
                                                 note="the quantity that makes this implementation "
                                                      "matter, and the one nobody has measured"),
        "g_ratio": normal(0.77, 0.03, units="dimensionless",
                          provenance=Provenance.LITERATURE,
                          source="axon diameter to fibre diameter ratio, ~0.6-0.8",
                          note="near the theoretical optimum for conduction speed, which is itself "
                               "evidence that this is a regulated quantity"),
        "unmyelinated_velocity_m_s": lognormal(0.5, 2.0, units="m/s",
                                               provenance=Provenance.LITERATURE,
                                               note="the floor the gain works up from; myelinated "
                                                    "cortical fibres run 1-10 m/s and the longest "
                                                    "tracts faster still"),
        "spine_turnover_per_day": normal(0.05, 0.03, units="fraction",
                                         provenance=Provenance.LITERATURE,
                                         source="in vivo two-photon imaging of adult cortex; a few "
                                                "percent of spines turn over daily",
                                         note="carried here because structural remodelling and "
                                              "myelination are the two legs of the same slow "
                                              "process, and this is the better measured one"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.SPECULATIVE,
    source="gibson et al 2014; mckenzie et al 2014; fields 2015",
)

implementation(
    name="learned_plasticity_rule",
    process="plasticity",
    doc="""the update rule itself as a learned function of the retained spectrum
    and the modulatory context.

    the honest response to the state of the field.  there are half a dozen
    plasticity rules above and in the literature, each fits the experiments it
    was designed for, and none generalizes cleanly to the others; that pattern is
    what a misspecified functional form looks like.  a learned rule reading the
    spectral state before the collapse can express dependencies none of the fixed
    forms can -- on the shape of the input spectrum, on the interaction between
    timing and rate, on modulatory state as a continuum rather than a gate -- and
    its prior mean is the three-factor rule, so with no data it reproduces the
    literature.

    the cost is not only computational.  a learned plasticity rule is a
    meta-learning problem, its posterior is under-constrained by any realistic
    amount of data, and it will happily fit a rule that reproduces the training
    observations through a mechanism that is not plasticity at all.  it is
    offered as an alternative f for the same declaration, which is exactly the
    point of keeping (I, O, T) separate from (f, theta), and selecting it is a
    statement about what the materialization is for.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0, note="weight prior; a learned f is a different p(theta), not a "
                                           "different declaration"),
        "residual_gain": weak(0.2, 5.0, note="learned term is a residual on the three-factor rule so "
                                             "the prior stays interpretable"),
        "meta_timescale_s": speculative(1.0e3, 30.0, units="s",
                                        note="the scale over which the learned rule's own parameters "
                                             "are allowed to move; there is no measurement of this "
                                             "because there is no measurement of the object"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.SPECULATIVE,
)

__all__ = ["PLASTICITY"]
