# v3: one circuit, declared populations, and resonance modes you can measure

**Decided 18 September 2026.** This supersedes the module-per-structure approach, and the
reason is not that the modules are wrong — five of them pass gates — but that the *shape* of
them cannot carry what the programme needs next.

## What forced it

Three things, all measured today rather than argued.

**1. Every module has a hand-set dial where a neuromodulatory nucleus should be.**

| module | the dial | what it stands in for |
|---|---|---|
| `substrate.py` | `m_beta`, `m_sigma` | cortical gain and noise — acetylcholine, noradrenaline |
| `thalamus.py` | `arousal` | the ascending arousal system |
| `hippocampus.py` | `septal_tone` | septal cholinergic drive |
| `basal_ganglia.py` | `dopamine` | VTA and substantia nigra pars compacta |

Four floats, passed in by a caller, **produced by nothing**. They cannot interact, cannot be
driven by the body, and cannot produce the state transitions the catalogue declares —
`arousal`, `ultradian_rem`, `cholinergic_desync` are rows waiting on the system these dials
are pretending to be.

**2. The build order was chosen by an instrument that is blind to half the brain.** Modules
were prioritised by "rhythms unlocked per unit of work", computed from `ibm/rhythms.py` — a
catalogue of *oscillations*. A structure whose job is not rhythmic scores near zero on that
metric by construction. The thalamus has a famous band; the amygdala does not. So the
amygdala, the accumbens, the VTA, the hypothalamus, the colliculus and the olfactory bulb
have nothing, and those are exactly the structures that make a body *want* something.
`docs/DEVELOPMENTAL_COMPONENTS.md` already states the consequence: a body with only
nociception can learn to avoid and cannot learn to seek.

**3. Resonance modes cannot be computed, because there is no assembled system.** The
catalogue declares 37 loops with per-edge conduction budgets, and the only thing that has
ever been done with those budgets is the round-trip **bound** — a ceiling, explicitly not a
prediction. A loop's actual resonance depends on every population in it, their time
constants, their gains and their delays *together*. Nothing in the repository assembles
them, so nothing can be asked.

## The v3 object

**One network. Declared populations, declared projections, delays on every edge.**

```
Pop     a named rate population: n units, tau, gain, threshold, optional adaptation,
        optional short-term depression and facilitation, and an optional TARGET SPARSITY
Proj    src -> dst: sign, weight, delay, topology (dense | sparse | topographic | diffuse)
Mod     a modulatory population whose rate MULTIPLIES a declared parameter of a target
        population -- this is what replaces the four dials
```

Three properties are structural rather than optional, because the measurements above say so:

**Sparse coding is inhibition, not a mask.** A population with a declared target sparsity
carries an inhibitory partner driven by that population's own mean rate, scaled so it is
half-activated at the target. `ibm/hippocampus.py` is the existence proof in this repo:
4.4% / 8.9% / 11.1% of three populations active, as a result of the dynamics at every step.
The cortical field has no equivalent, and the operating-point search (0 of 18) plus the
facilitation result (selectivity 0.652) both trace back to that absence.

**Attractors live in weights, not in per-site bistability.** The search proved a single
sigmoidal rate per site cannot be both an attractor and a graded transfer function, because
one loop gain sets both. CA3 holds 6/6 pattern retrieval at 6% active with a shuffled
control at chance, and it does that with a *stored recurrent matrix* and no bistable unit
anywhere. v3 populations are graded; recurrent structure carries the memory.

**Neuromodulators are populations.** `Mod` edges multiply gain, noise, threshold or a
synaptic constant on their targets. A state like waking or NREM is then where the network
sits, not a number a caller passes — which is what the `arousal` row has always declared and
never had.

## Resonance modes, measured rather than asserted

With an assembled network, a loop's resonance is a **measurement**: perturb the network with
a small oscillatory drive at each of a set of frequencies, measure the response amplitude at
each population, and read the peaks. That is a transfer function, it needs no linearisation
algebra, and it gives per loop:

- the frequencies the loop amplifies, with their sharpness;
- what happens to them when an edge is cut, a delay is changed, or a neuromodulator moves;
- a number to put beside the catalogue's declared band for that loop — and beside its
  round-trip bound, which until now has been the only thing the conduction budget was used
  for.

This is the first thing in the programme that can say *why* a loop has the band it has,
rather than that the band is not impossible.

## What happens to v2 and the five modules

They stay, unchanged, with their gate results intact. Nothing is rewritten to match v3 and
no recorded number changes meaning. v3 is built beside them exactly as v2 was built beside
v1, and the modules are the reference each v3 structure is checked against: a v3 thalamus
that does not reproduce `ibm/thalamus.py`'s 13.63 Hz spindle has a bug, and that is a known
answer worth having.

**v2 is left incomplete on purpose.** Its open failures — G2/G3 under a graded prior, gamma,
ripples, gating — are all consequences of properties v3 has structurally. Fixing them in v2
would mean fitting a model class the measurements say cannot hold them.

## Order of work

1. **the engine** — populations, projections, delays, sparse inhibition, modulation, with
   boundedness and idempotence gated as for every other module here;
2. **the resonance-mode instrument** — the frequency sweep, checked on a case whose answer
   is known before it is pointed at anything;
3. **the structures**, declared as population sets: cortex, thalamus, hippocampus, basal
   ganglia, cerebellum, cord, and then the ones that have never been built —
   **neuromodulatory nuclei, amygdala/accumbens/VTA valuation, hypothalamus**, colliculus and
   pulvinar, olfactory bulb;
4. **the loops** from `ibm/rhythms.py`, instantiated as projections, with their declared
   conduction budgets;
5. **the modes measured** against the catalogue's declared bands, loop by loop.

Each step is gated and recorded the way everything else in this repository is, and a failure
at any of them is recorded as a failure rather than routed around.
