# What the brain should be

**This is a specification, not a report.** It says what IBM-1's brain is being built to be.
Nothing here is conditional on a measurement; the measurements come after, and they are how
we find out whether we built what we said. Written 18 September 2026, when the programme
stopped asking "is this accurate?" of each part in isolation and started asking "what are we
making?" of the whole.

## The claim this whole thing rests on

**A brain does not work one circuit at a time.** No single loop, run alone, does anything
worth having — and the programme has now measured that six ways: a cortical sheet that
ignites or sleeps, a thalamus whose gating barely gates, a hippocampus whose quiet state is
merely quiet, a cerebellum that learns a little, a basal ganglia that selects in isolation,
a cord that walks with nothing above it. Every one of those is a part behaving correctly and
adding up to nothing.

What we are building is the state where **all of it is active at once and the activity is
coordinated**: many loops running simultaneously, each in its own band, sparse, and
switching together. That is the object. Everything below is what it takes to have one.

## The inventory

Every structure below is part of the brain we are building. None is optional and none is
"later": the ones that were left out were left out because a rhythm metric could not see
them, and that is the mistake this document exists to end.

### Cortex
Excitatory and inhibitory populations per area, over the Desikan–Killiany parcellation, on
the sensorimotor-to-association hierarchy: fast, input-following, low recurrence at the
sensory end; slow, high recurrence, long intrinsic timescales at the transmodal end.
Long-range cortico-cortical projections carry **measured tract delays**. Every population is
**sparse and divisively normalised** — that is not a tuning choice, it is what lets a subset
ignite without the sheet igniting.

### Thalamus
Relay nuclei as separate populations — **LGN, MGN, VPL, VL, VA, MD, pulvinar, centromedian**
— each with its own cortical partner, plus the **reticular nucleus** as a sector per nucleus.
Relay cells have the T-current rebound; the reticular sector inhibits them with GABA-A and
GABA-B. First-order nuclei relay the periphery; higher-order nuclei (pulvinar, MD) relay
cortex to cortex, which is how cortical areas talk without a direct fibre.

### Basal ganglia
**D1 and D2 striatum, GPe, GPi/SNr, STN, SNc**, in parallel channels: motor, oculomotor,
associative, limbic. Direct, indirect and hyperdirect pathways, each with its own delay.
Tonically active output nuclei that *pause* to release a channel. Dopamine from SNc/VTA sets
the D1/D2 balance and is produced by the valuation system, not passed in.

### Hippocampal formation
**Entorhinal cortex, dentate gyrus, CA3, CA1, subiculum**, with the **medial septum** as
theta pacemaker. Separation in the dentate, completion in CA3's recurrent collaterals,
comparison in CA1. Theta with gamma nested inside it during exploration; sharp-wave ripples
when the septum falls quiet.

### Cerebellum
**Pontine nuclei, mossy fibres, granule layer, Golgi cells, Purkinje cells, molecular-layer
interneurons, deep nuclei, inferior olive**. Expansion recoding in the granule layer,
climbing-fibre teaching, and **plasticity in both directions** — long-term depression *and*
potentiation at the parallel-fibre synapse, because a cerebellum that can only depress can
only learn to push one way.

### Valuation and affect
**Basolateral amygdala, central amygdala, nucleus accumbens (core and shell), ventral
tegmental area, lateral habenula, ventromedial prefrontal and orbitofrontal cortex.** This
is the system that makes some states better than others. Without it a body can learn to
avoid and cannot learn to seek, and every "reward" in this programme is otherwise a number
handed in from outside.

### Neuromodulatory nuclei
**Locus coeruleus (noradrenaline), dorsal raphe (serotonin), pedunculopontine and
laterodorsal tegmentum plus nucleus basalis (acetylcholine), VTA and SNc (dopamine),
tuberomammillary (histamine), orexinergic lateral hypothalamus.** Each is a population whose
rate *multiplies* parameters of its targets — gain, threshold, noise, synaptic strength.
**These replace the four hand-set dials** (`m_beta`, `arousal`, `septal_tone`, `dopamine`)
that four separate modules currently use to fake a system that was never built.

### Hypothalamus and the drives
**Suprachiasmatic nucleus** (circadian), **paraventricular** (stress, HPA), **ventrolateral
preoptic** (sleep), **lateral** (orexin, arousal, feeding), **arcuate** (energy state).
These carry the body's needs into the brain as signals that change what the brain wants,
which is where value comes from in a body rather than from a scalar.

### Brainstem
**Nucleus of the solitary tract, parabrachial nucleus, pre-Bötzinger complex, mesencephalic
locomotor region, superior and inferior colliculi, vestibular nuclei, reticular formation.**
Interoception arrives here; breathing is generated here; orienting starts here; locomotion
is commanded from here.

### Spinal cord and the periphery
**Half-centre pattern generators, alpha and gamma motoneurons, Renshaw and Ia interneurons,
muscle spindles, Golgi tendon organs, cutaneous and visceral afferents**, and the muscles
themselves through IHM-1. The cord is where the brain stops being a simulation of itself.

### Olfactory
**Mitral and granule cells of the bulb, piriform cortex** — the one sensory path that does
not go through the thalamus, and the one that is paced by breathing.

## What the assembled thing must do

These are the properties we are building toward. They are stated so that failing them is
unambiguous.

1. **Every structure active at once**, each at its declared sparsity, with no population
   saturated and none silent.
2. **Many loops simultaneously in their own bands** — the thalamo-cortical loop in alpha and
   spindles, the hippocampus in theta with nested gamma, the basal ganglia in beta, the
   cerebellum's fast rhythm, the cortex's gamma — at the same time, in one running network,
   rather than one at a time in separate scripts.
3. **Coordinated switching.** The brain moves between whole-brain states, not region by
   region independently. A state is a pattern across many structures that persists, is left,
   and is returned to.
4. **Ignition as a sparse, global event.** A subset of populations across several structures
   rises together and sustains, while the rest is suppressed rather than dragged along.
   Divisive normalisation is what makes this possible: a subset can go up precisely because
   the normaliser holds the total bounded.
5. **States the brain is in rather than told.** Waking, NREM, REM, alert, drowsy — each a
   place in neuromodulator space the network moves through, with the rhythms changing
   because the state changed.
6. **Value produced inside.** Reward and cost come from the valuation system and the
   hypothalamic drives, reading a body, not from a scalar in a training loop.
7. **A body in the loop.** Motor commands leave through the cord to real muscles; afferents
   return. The brain's rhythms are measurable at the periphery — corticomuscular coherence
   is a prediction of the assembled thing, not a separate experiment.

## How it is built

One engine (`ibm/circuit.py`): declared populations, projections with per-edge conduction
delays, neuromodulatory edges, sparse populations held by divisive normalisation, and
calibration that *measures* each inhibitory gain rather than guessing it.

One declaration per structure under `ibm/brain/`, each exporting the same three things, so
the whole brain is an assembly rather than a set of bespoke interfaces. The loops in
`ibm/rhythms.py` — 37 of them, with their conduction budgets — become projections.

And one instrument that only exists once the thing is assembled: **resonance modes**.
Perturb the running network at each of a set of frequencies and read where it amplifies.
That tells us what bands the built brain actually has, loop by loop, and it is the first
thing in this programme that can say *why* a loop has the band it has.

## The order of the work

Engine (done) → structure declarations, all of them → assembly → calibration of the whole →
resonance modes → the seven properties above, measured, in order, with each failure recorded
as a failure.
