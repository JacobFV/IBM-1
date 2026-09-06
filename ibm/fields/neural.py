"""neural population state.

the field is spectral throughout, and this is the case the architecture's rule
was written for.  a cortical population's membrane state carries slow drift,
theta, alpha, beta and gamma *at the same time and in the same variable*, and
they are not independent: gamma amplitude rides on theta phase, alpha gates the
gain of everything faster, and a slow depolarizing drift changes which of them
survives.  collapse that to a mean and a variance and the interaction -- the
thing the model exists to represent -- is gone before any process runs.  the
spectral form also buys the practical thing that makes a brain-scale graph
tractable: tract conduction delays are phase ramps and every linear coupling is
a diagonal transfer function, so no per-edge history buffer and no delay-bounded
timestep.

populations, not regions.  excitatory, pv, sst and vip cells coexist at the same
millimetre of cortex, so they are components at a position rather than mutually
exclusive spatial labels (ARCHITECTURE.md §2).  pv, sst and vip carry activity
only: their membrane and synaptic detail is not separately identifiable from any
measurement ibm-1 will see, and declaring state no evidence can move is how an
ontology sprawls.  the generic `inh` pair remains, because most mesoscale
literature and most fitted models speak in one inhibitory population, and the
two descriptions coexist rather than one replacing the other.

the three interneuron classes are now a measured decomposition rather than a
declared one, and one of the numbers behind that is a gap in it.  MICrONS's
proofread volume classifies 71,551 typed neurons of mouse visual cortex, and
`scripts/measure_microcircuit.py` counts them: 89.0% excitatory, 4.6% basket
(which is what `pv` is a population of), 3.4% martinotti (`sst`), 2.1% bipolar
(`vip`) and **0.9% neurogliaform, for which this field declares no component at
all**.  neurogliaform cells are not a rounding error in the circuit even at that
abundance -- they are contacted by excitatory cells at 0.056 within 100 um,
comparable to `vip`'s 0.044, and they contact excitatory cells more often than
`vip` does.  the honest position is that the decomposition here is three classes
where the tissue has at least four, and that the fourth is left out because it
carries volume-transmitted gaba-b inhibition this field's `inh.gaba_b` already
lumps rather than because it is absent.  `ibm/topologies/microcircuit_prior.py`
carries neurogliaform connectivity through its measurement and drops it at the
prior, and says so, rather than folding it into `sst` on the grounds that both
target dendrites.

the same census is where the species problem becomes a number.  the inhibitory
fraction is 0.110 in that mouse volume and 0.308 in H01's human temporal cortex,
which is a factor of 2.8 -- part real and part two classifiers disagreeing about
what counts as a neuron.  a field that declares one `inh` population and three
named classes has to be read with that in mind: the classes' *identities* transfer
across species and their *proportions* do not.

separate conductances rather than one "synaptic input" because their time
constants differ by two orders of magnitude and that difference is load-bearing:
ampa and gaba-a set the gamma rhythm, nmda sets the slow recurrent integration
that makes persistent activity possible, and gaba-b sets the slow gating that
scalar summaries of inhibition cannot express.

`transmembrane_current` is declared separately from potential and activity
because it, not they, is what the electromagnetic field integrates.  the
extracellular signal is a weighted sum of transmembrane currents with a spatial
weighting that depends on morphology and material state, so treating firing rate
as a proxy for it is the standard way a forward model quietly stops being
physics.

afferent and efferent traffic sit on `body`.  the architecture is explicit that
peripheral ganglia and nerve are neural population state on a support that is
not brain parenchyma; `body` is the coarsest support that actually contains
retinal ganglion, spiral ganglion, dorsal root and autonomic ganglia together.
it is a compromise -- a retinal ganglion cell would rather be indexed on the
retina -- and it is recorded here rather than papered over with a second
component per pathway.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Band, Provenance

FIELD = REGISTRY.field(Field(
    "neural",
    "population state of neural tissue: membrane and synaptic state, activity, adaptation "
    "and the transmembrane current that sources the electromagnetic field.  its support "
    "extends beyond the brain to peripheral ganglia and nerve",
    "tissue", Provenance.LITERATURE))

#: wide enough to hold the spike band; a materialization almost never asks for
#: all of it, which is the point of bandwidth being a laziness axis.
WIDE = Band(0.0, 5000.0)
SYNAPTIC = Band(0.0, 1000.0)
SLOW = Band(0.0, 50.0)


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "spectral")
    kw.setdefault("provenance", Provenance.LITERATURE)
    return REGISTRY.component(Component(id=id_, field="neural", doc=doc, units=units, **kw))


# -- excitatory population ---------------------------------------------------

EXC_POTENTIAL = _c(
    "neural.exc.potential",
    "mean membrane potential of the excitatory (principal, largely pyramidal) population "
    "at a position.  it is the variable synaptic conductances act on and the variable the "
    "population's transfer function is a function of, so it is the state, not the firing "
    "rate that summarizes it: two populations at the same rate but different distances "
    "from threshold respond differently to identical input",
    "mV", band=WIDE, prior="neural_population", bounds=(-90.0, 50.0),
    timescale_s=1e-2, tags=frozenset({"population", "excitatory"}),
    alt_supports=("cortical_surface",))

EXC_ACTIVITY = _c(
    "neural.exc.activity",
    "population firing rate of excitatory cells: spikes per second averaged over the cells "
    "within a materialized position.  a separate component from potential because the "
    "map between them is a steeply nonlinear, adaptation- and noise-dependent transfer "
    "function, and because rate is what tract propagation transmits while potential is "
    "what stays local",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 500.0),
    timescale_s=5e-3, tags=frozenset({"population", "excitatory"}),
    alt_supports=("cortical_surface",))

EXC_ADAPTATION = _c(
    "neural.exc.adaptation",
    "the slow hyperpolarizing current that accumulates with recent spiking -- calcium- and "
    "sodium-activated potassium conductances, aggregated.  it is the population's memory of "
    "its own recent output on a 0.1-2 s timescale and is why identical input produces "
    "different output at the start and end of a stimulus; without it a population model has "
    "no mechanism for spike-frequency adaptation or for slow oscillation",
    "nA", band=SLOW, prior="adaptation_slow", bounds=(0.0, 10.0),
    timescale_s=0.5, tags=frozenset({"population", "excitatory", "slow"}),
    alt_supports=("cortical_surface",))

EXC_AMPA = _c(
    "neural.exc.ampa",
    "aggregate ampa-receptor conductance onto the excitatory population.  fast (~2-5 ms "
    "decay) and voltage-independent, so it carries the component of recurrent and afferent "
    "excitation that can support gamma-band dynamics.  separate from nmda because the two "
    "are carried by the same synapses but have completely different dynamics",
    "nS", band=SYNAPTIC, prior="synaptic_conductance", bounds=(0.0, 1000.0),
    timescale_s=3e-3, tags=frozenset({"synaptic", "excitatory"}),
    alt_supports=("cortical_surface",))

EXC_NMDA = _c(
    "neural.exc.nmda",
    "aggregate nmda-receptor conductance onto the excitatory population: slow (~100 ms) and "
    "magnesium-blocked, so its effective gain depends on the population's own membrane "
    "potential.  that voltage dependence makes recurrent excitation regenerative and is the "
    "usual substrate for persistent activity, which a single lumped excitatory conductance "
    "cannot produce",
    "nS", band=SLOW, prior="synaptic_conductance", bounds=(0.0, 1000.0),
    timescale_s=0.1, tags=frozenset({"synaptic", "excitatory"}),
    alt_supports=("cortical_surface",))

# -- inhibitory population ---------------------------------------------------

INH_POTENTIAL = _c(
    "neural.inh.potential",
    "mean membrane potential of the lumped inhibitory population.  kept alongside the "
    "interneuron-class activity components rather than replaced by them: most mesoscale "
    "literature, and every neural-mass model ibm-1 will inherit parameters from, is written "
    "in one excitatory and one inhibitory population",
    "mV", band=WIDE, prior="neural_population", bounds=(-90.0, 50.0),
    timescale_s=5e-3, tags=frozenset({"population", "inhibitory"}),
    alt_supports=("cortical_surface",))

INH_ACTIVITY = _c(
    "neural.inh.activity",
    "population firing rate of inhibitory cells.  its bound is higher than the excitatory "
    "one because fast-spiking interneurons sustain rates that principal cells cannot, and "
    "that asymmetry is what sets the frequency of pyramidal-interneuron gamma",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 1000.0),
    timescale_s=3e-3, tags=frozenset({"population", "inhibitory"}),
    alt_supports=("cortical_surface",))

INH_GABA_A = _c(
    "neural.inh.gaba_a",
    "aggregate gaba-a conductance: fast chloride-permeable inhibition, ~6 ms decay.  its "
    "reversal potential sits near rest, so it acts largely by shunting rather than by "
    "hyperpolarizing, and its time constant together with ampa sets the gamma period.  it "
    "is also the target of most anaesthetics, which is why it must be a state variable a "
    "pharmacological process can act on rather than a fixed weight",
    "nS", band=SYNAPTIC, prior="synaptic_conductance", bounds=(0.0, 1000.0),
    timescale_s=6e-3, tags=frozenset({"synaptic", "inhibitory"}),
    alt_supports=("cortical_surface",))

INH_GABA_B = _c(
    "neural.inh.gaba_b",
    "aggregate gaba-b conductance: metabotropic, potassium-mediated, ~150-300 ms, and "
    "recruited only by sustained inhibitory firing.  a slow gate on excitability rather "
    "than a per-event current, and the component that carries inhibition's contribution to "
    "delta and slow-oscillation timescales",
    "nS", band=SLOW, prior="synaptic_conductance", bounds=(0.0, 1000.0),
    timescale_s=0.15, tags=frozenset({"synaptic", "inhibitory", "slow"}),
    alt_supports=("cortical_surface",))

# -- interneuron classes -----------------------------------------------------

PV_ACTIVITY = _c(
    "neural.pv.activity",
    "firing rate of parvalbumin-expressing interneurons: perisomatic, fast-spiking, and "
    "the pacemaker of gamma.  a separate component from the lumped inhibitory rate because "
    "pv cells target the soma while sst cells target distal dendrites, so the two divide "
    "inhibition by target compartment rather than by amount",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 800.0),
    timescale_s=3e-3, tags=frozenset({"population", "inhibitory", "interneuron"}),
    alt_supports=("cortical_surface",))

SST_ACTIVITY = _c(
    "neural.sst.activity",
    "firing rate of somatostatin-expressing interneurons: dendrite-targeting, facilitating "
    "rather than depressing, and therefore recruited by sustained input rather than by "
    "transients.  they gate which inputs a pyramidal dendrite integrates, which is a "
    "different computation from the gain control pv cells implement",
    "Hz", band=SYNAPTIC, prior="neural_spiking", bounds=(0.0, 300.0),
    timescale_s=2e-2, tags=frozenset({"population", "inhibitory", "interneuron"}),
    alt_supports=("cortical_surface",))

VIP_ACTIVITY = _c(
    "neural.vip.activity",
    "firing rate of vasoactive-intestinal-peptide interneurons.  they inhibit sst cells "
    "rather than principal cells, so their effect is disinhibitory, and they are the "
    "principal cortical target of cholinergic and long-range top-down drive.  a sign "
    "inversion in the microcircuit is not something a lumped inhibitory population can "
    "express at all",
    "Hz", band=SYNAPTIC, prior="neural_spiking", bounds=(0.0, 300.0),
    timescale_s=2e-2, tags=frozenset({"population", "inhibitory", "interneuron"}),
    alt_supports=("cortical_surface",))

# -- the electromagnetic source ----------------------------------------------

TRANSMEMBRANE_CURRENT = _c(
    "neural.transmembrane_current",
    "net current per unit tissue volume crossing neuronal membranes at a position, signed: "
    "the current source density that sources every extracellular potential and every "
    "magnetic field the model predicts.  it is declared separately from potential and "
    "activity because the map from population state to it depends on dendritic morphology "
    "and synapse placement -- a synchronous input to apical dendrites and the same input to "
    "somata produce opposite dipoles at identical firing rates",
    "nA/mm^3", band=WIDE, prior="transmembrane_current", timescale_s=1e-3,
    tags=frozenset({"source", "electromagnetic"}),
    alt_supports=("cortical_surface",))

# -- peripheral traffic ------------------------------------------------------

AFFERENT_ACTIVITY = _c(
    "neural.afferent.activity",
    "firing rate of primary afferent neurons -- retinal and spiral ganglion, dorsal root, "
    "cranial nerve, vestibular and visceral afferents -- carried on the body support "
    "because these somata are not brain parenchyma.  it is the variable a transduction "
    "process writes into and an afferent-pathway process reads out of, and keeping it "
    "distinct from the receptor state upstream matters because spike initiation, "
    "refractoriness and rate saturation happen here and not at the receptor",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 1000.0),
    timescale_s=2e-3, support="body",
    tags=frozenset({"population", "peripheral", "afferent"}))

# -- peripheral traffic, resolved by fibre class ------------------------------
#
# `afferent.activity` and `efferent.activity` above are the AGGREGATE traffic in a
# nerve.  the components below are the same traffic resolved by Erlanger-Gasser /
# Lloyd fibre class, and they are components rather than partitions for the reason
# `ibm.anatomy.systems` states in its header: fibre classes coexist in the same
# trunk at the same millimetre, so they cannot tile space.  a peripheral nerve is a
# partition; what runs inside it is a component vector.
#
# the classes are not a taxonomy for its own sake.  each one has a different
# conduction velocity, and in a model where "delay dominates this topology in a way
# it does not dominate the cortical one" (`ibm.topologies.afferent`) a lumped
# afferent rate asserts that a 100 m/s Ia and a 1 m/s C fibre arrive together --
# which is wrong by two orders of magnitude and wrong in the direction that
# destroys every reflex latency the model might otherwise reproduce.

IA_AFFERENT = _c(
    "neural.afferent.ia",
    "primary muscle-spindle afferent: group Ia, the largest and fastest myelinated "
    "sensory fibre (12-20 um, 80-120 m/s).  it reports muscle LENGTH and its RATE OF "
    "CHANGE, is the afferent limb of the monosynaptic stretch reflex, and its dynamic "
    "sensitivity is set by gamma-dynamic drive to the intrafusal fibre rather than "
    "being a fixed property of the receptor.  it is separated from group II because "
    "the velocity sensitivity is the whole functional difference between them",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 500.0),
    timescale_s=2e-3, support="body",
    tags=frozenset({"population", "peripheral", "afferent", "proprioceptive"}))

IB_AFFERENT = _c(
    "neural.afferent.ib",
    "Golgi tendon organ afferent: group Ib, myelinated and nearly as fast as Ia "
    "(12-20 um, 80-120 m/s), but in series with the tendon rather than in parallel "
    "with the muscle, so it reports FORCE where Ia reports length.  the pair is what "
    "lets a controller distinguish a limb that has moved from a limb that is loaded, "
    "and a model with only one of them cannot tell those apart",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 500.0),
    timescale_s=2e-3, support="body",
    tags=frozenset({"population", "peripheral", "afferent", "proprioceptive"}))

II_AFFERENT = _c(
    "neural.afferent.ii",
    "secondary muscle-spindle afferent: group II, medium myelinated (6-12 um, "
    "35-75 m/s), reporting static muscle length with little velocity sensitivity.  "
    "it is the tonic position signal that survives when the dynamic Ia response has "
    "adapted",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 500.0),
    timescale_s=2e-3, support="body",
    tags=frozenset({"population", "peripheral", "afferent", "proprioceptive"}))

ABETA_AFFERENT = _c(
    "neural.afferent.abeta",
    "cutaneous mechanoreceptive afferent: A-beta, myelinated (6-12 um, 35-75 m/s), "
    "carrying touch, vibration and pressure from Merkel, Meissner, Pacinian and "
    "Ruffini endings.  this is the class the dorsal columns are usually pictured as "
    "carrying, and it is the one the model had before proprioception was declared",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 1000.0),
    timescale_s=2e-3, support="body",
    tags=frozenset({"population", "peripheral", "afferent", "cutaneous"}))

ADELTA_AFFERENT = _c(
    "neural.afferent.adelta",
    "thinly myelinated nociceptive and thermoreceptive afferent: A-delta (1-5 um, "
    "5-30 m/s), carrying first pain and cold.  it is an order of magnitude slower "
    "than A-beta, which is why a lumped cutaneous afferent gets the withdrawal-reflex "
    "latency wrong",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 200.0),
    timescale_s=3e-3, support="body",
    tags=frozenset({"population", "peripheral", "afferent", "nociceptive"}))

C_AFFERENT = _c(
    "neural.afferent.c",
    "unmyelinated afferent: C fibre (0.2-1.5 um, 0.5-2 m/s), carrying second pain, "
    "warmth, itch, affective touch and the great majority of visceral afferent "
    "traffic.  two orders of magnitude slower than Ia over the same path -- a foot "
    "C fibre takes most of a second to reach the cord -- and that latency is a "
    "phenomenon, not an error term",
    "Hz", band=SLOW, prior="neural_spiking", bounds=(0.0, 100.0),
    timescale_s=1e-2, support="body",
    tags=frozenset({"population", "peripheral", "afferent", "nociceptive",
                    "visceral"}))

ALPHA_EFFERENT = _c(
    "neural.efferent.alpha",
    "alpha motoneuron axon traffic: the final common path to EXTRAFUSAL muscle "
    "fibres (12-20 um, 80-120 m/s).  this is the component that produces force, and "
    "it is separated from gamma because they are separately controlled and their "
    "co-activation ratio is itself a motor-control variable",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 200.0),
    timescale_s=2e-3, support="body",
    tags=frozenset({"population", "peripheral", "efferent", "somatic"}))

GAMMA_EFFERENT = _c(
    "neural.efferent.gamma",
    "gamma (fusimotor) motoneuron axon traffic to INTRAFUSAL fibres (2-8 um, "
    "10-45 m/s).  it produces no meaningful force; it sets the gain and the dynamic "
    "sensitivity of the spindle that reports back on Ia and II.  **this is the one "
    "efferent whose target is a sensor**, so it is the mechanism by which the "
    "nervous system controls its own proprioceptive gain -- alpha-gamma "
    "co-activation is what keeps the spindle loaded while the muscle shortens, and "
    "without it every spindle falls silent during exactly the movements it is "
    "needed for",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 200.0),
    timescale_s=2e-3, support="body",
    tags=frozenset({"population", "peripheral", "efferent", "fusimotor"}))

B_EFFERENT = _c(
    "neural.efferent.b_preganglionic",
    "preganglionic autonomic axon traffic: B fibres, thinly myelinated (1-3 um, "
    "3-15 m/s), sympathetic and parasympathetic, running from the cord or brainstem "
    "to a ganglion.  the synapse in the ganglion is why this is a separate component "
    "from the postganglionic traffic and not the same signal further along",
    "Hz", band=SLOW, prior="neural_spiking", bounds=(0.0, 50.0),
    timescale_s=5e-3, support="body",
    tags=frozenset({"population", "peripheral", "efferent", "autonomic"}))

C_EFFERENT = _c(
    "neural.efferent.c_postganglionic",
    "postganglionic sympathetic axon traffic: unmyelinated C fibres (0.3-1.3 um, "
    "0.5-2 m/s) from ganglion to effector -- vessels, sweat glands, viscera, "
    "piloerector muscle.  slow, diffuse and volume-transmitted at the target, which "
    "is why autonomic effects have seconds-scale onsets that a myelinated model "
    "cannot produce",
    "Hz", band=SLOW, prior="neural_spiking", bounds=(0.0, 50.0),
    timescale_s=1e-2, support="body",
    tags=frozenset({"population", "peripheral", "efferent", "autonomic"}))

EFFERENT_ACTIVITY = _c(
    "neural.efferent.activity",
    "firing rate of motor and autonomic efferent neurons at their peripheral projection: "
    "alpha and gamma motoneuron axons, preganglionic and postganglionic autonomic fibres.  "
    "on the body support for the same reason the afferents are, and separate from "
    "effector.drive because one motoneuron pool's output is distributed across motor units "
    "by a recruitment rule that is itself state-dependent",
    "Hz", band=WIDE, prior="neural_spiking", bounds=(0.0, 500.0),
    timescale_s=2e-3, support="body",
    tags=frozenset({"population", "peripheral", "efferent"}))
