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
