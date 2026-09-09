# putting it in a body

> **§1–§3 below were the assessment BEFORE the peripheral nervous system was
> declared. §5 records the periphery and §8 the cord. §2.1 (proprioception),
> the fibre-lumping, and §2.2 (spinal circuitry) are now CLOSED. §2.3 —
> cerebellum and efference copy — still stands.**

**yes, the substrate supports materializing a nerve-based input/output model, and
the interface is better specified than most of the rest of the codebase.** the
realism problem is not the interface. it is that three things which do most of the
work in real sensorimotor control are absent, and their absence is quantifiable.

---

## 1. what is already declared

### afferent — 5 pathways, receptotopic, delay-dominated

`ibm/topologies/afferent.py` is one of the better-argued files in the repo, and
the argument matters here: **the afferent adjacency cannot be built from
geometry.** the retina is in an eye-centred frame, the cochlea is parameterized
along the basilar membrane, the skin is a 2-D body surface, and the euclidean
distance from a retinal position to its LGN target "is a number about two frames
that were never registered to each other, and it has nothing to do with the
length of the optic nerve." so the metric is receptotopic — visual field, octaves,
body-surface coordinates — and physical length is a **separate per-stage
quantity**.

| pathway | stages | total conduction |
|---|---|---|
| visual | retina→LGN→V1 | 12.7 ms |
| auditory | cochlea→CN→IC→MGN→A1 | 5.8 ms |
| somatosensory | skin→dorsal column nuclei→VPL→S1 | 17.0 ms |
| vestibular | labyrinth→vestibular nuclei | 0.4 ms |
| visceral | viscera→NTS→parabrachial | 63.0 ms |
| olfactory | epithelium→bulb | 20.0 ms |

**delay dominates, and the file says why: "these are not corrections to a central
model, they are the largest latencies in it."** that is the right call for
embodiment — the periphery is where the timing is set.

### transduction — 11 receptor implementations
`photoreceptor_cascade`, `hair_cell_cochlear`, `gammatone_cochleagram`,
`luminance_motion_energy`, `mechanoreceptor_rapid`, `mechanoreceptor_slow`,
`thermoreceptor_static_dynamic`, `nociceptor_sensitizing`,
`chemoreceptor_diffusive`, `vestibular_canal_otolith`, `baroreceptor_arterial`,
plus `learned_receptor_bank`. rapid/slow mechanoreceptor adaptation is separated,
which is the distinction that matters for texture and slip.

### efferent — 4 pathways, and a real motor plant
`m1→motoneuron→motor_unit` (18.4 ms), plus cranial/vocal, oculomotor, and
autonomic. the implementations are not placeholders:

- `henneman_recruitment` (RATE) — recruitment order, explicitly non-linear
  because "linear recruitment is the one thing about it that is definitely false"
- `hill_muscle` (RATE) — force-length-velocity
- `twitch_activation_lti`, `fatigue_accumulator`, `limb_load_lti`
- `corticospinal_dispersed_delay` — dispersed rather than fixed, because the
  tract is a fibre-diameter distribution

conduction velocity reads `structural.myelination`, so **a demyelinating lesion is
a change to this process's inputs, not its parameters** — which is the modelling
claim you want if the body sim is ever to include pathology.

---

## 2. what is missing, and what it costs

### 2.1 there is no proprioception at all — the biggest gap

every declared mechanoreceptor is on `body_surface`. **there is no muscle
spindle, no Golgi tendon organ, no joint-angle afferent, and no
`effector.length` or `effector.velocity` state at all.** grep confirms it: the
only "spindle" in the codebase is the thalamocortical sleep spindle.

the somatosensory pathway is `skin→dorsal column nuclei→VPL→S1`, which is
anatomically the dorsal column–medial lemniscus route — and in a real body that
route carries proprioception *and* cutaneous touch. here it carries only touch,
because only touch is transduced.

**consequence: the model has no sense of where its limbs are.** it can feel
contact and it cannot feel posture. every closed-loop limb behaviour — reaching,
balance, load compensation, grip force — is unavailable in principle, not merely
unfitted.

### 2.2 there is no spinal circuitry, and the codebase says so

`effector.py` states it plainly and correctly:

> "where this breaks: there is no spinal circuitry in it. the segmental
> interneurons, reciprocal inhibition, renshaw feedback, and the stretch reflex
> loop are all absent, so the pool here is a relay with a threshold rather than a
> circuit. every reflexive and load-compensating behaviour is therefore missing,
> **and the model will attribute those to descending command because descending
> command is the only thing it has.**"

that last clause is the failure mode to expect: the fitted model will inflate
corticospinal gain to explain behaviour that the spinal cord actually produced.

note also that no anatomy system covers the cord — there is no spinal segment,
dermatome, myotome or dorsal-root partition among the 13 declared systems.

### 2.3 no forward model, no efference copy — and the delays make this decisive

this is where the three gaps compound, and it is computable from the declared
numbers rather than asserted:

| leg | delay |
|---|---|
| cortex → motor unit | 18.4 ms |
| skin → cortex | 17.0 ms |
| **conduction round trip** | **35.4 ms** |
| + electromechanical delay, fast twitch | ≈ 75 ms |
| + electromechanical delay, slow twitch | ≈ 135 ms |

a feedback loop with delay *T* is stable only up to roughly **1/(4T)**:

- 75 ms → **3.3 Hz**
- 135 ms → **1.8 Hz**

real limb control does far better than that, and it does so by two routes the
model does not have: **the monosynaptic stretch reflex (~30 ms, entirely spinal —
§2.2) and cerebellar predictive control (mechanism 18 — declared anatomy,
`cerebellar_lobules` and `cerebellar_microzones`, used by NO process, STATE.md
§4.9).** without either, the only corrective path is the long cortical loop.

so the prediction is specific: **the materialized body model will be limited to
~2–3 Hz of corrective bandwidth, and will therefore be sluggish, or oscillatory
if any gain is fitted up to compensate.** that is not a tuning problem; it is
what a 75–135 ms loop with no predictor does.

mechanism 19 (efference copy) is absent too, so the model also cannot distinguish
self-caused from externally-caused sensory change — no reafference cancellation
during movement.

### 2.4 no musculoskeletal plant
`hill_muscle` gives force per motor unit and `limb_load_lti` a lumped load, but
there is no skeleton, no joints, no segment inertias, no multi-joint coupling.
`learned_musculoskeletal` is a LEARNED placeholder. **the body sim must supply the
plant**; ibm-1 supplies the neural drive to it and the receptors from it.

that is arguably the right split — but it means the interface contract is
`effector.force` out, receptor state in, and the simulator owns everything
between.

### 2.5 the fan-in gain problem applies here too
STATE.md §4.10: `|L(0)| = 7218` because the `weak(1.0, 5.0)` gain prior is
per-edge and does not know the fan-in. the efferent and afferent chains use the
same `gain` priors. **stage 0 blocks embodiment exactly as it blocks everything
else.**

---

## 3. how unrealistic, concretely

| behaviour | expected |
|---|---|
| receptor-level afferent traffic | **plausible** — 11 fitted-able receptor models, correct adaptation classes |
| EMG / motor-unit output | **plausible** — Henneman recruitment, Hill muscle, fatigue, dispersed corticospinal delay |
| conduction timing | **plausible and load-bearing** — per-stage lengths and velocities, one significant figure, honestly flagged |
| reflexes | **absent** |
| load compensation | **absent** |
| limb position sense | **absent** |
| reaching / balance / grip | **not achievable** — no proprioception to close the loop with |
| corrective bandwidth | **~2–3 Hz**, vs human ≫ that |
| reafference cancellation | **absent** |

**summary: it will produce credible nerve traffic and no competent movement.**
the I/O interface is real; the control loop is open.

---

## 4. the shortest path to a usable embodiment

in dependency order. items 1–2 are small and unblock the rest.

1. **declare proprioceptive transduction** — `transduction.spindle_primary`,
   `spindle_secondary`, `golgi_tendon`, reading `effector.length` and
   `effector.velocity` (both of which also need declaring). this is a
   `transduction` implementation plus two components, and it is the single
   highest-value change on this page
2. **route it** — the `somatosensory` stage table already ends at S1 via the
   dorsal columns; proprioception rides the same stages, so this is a selector
   change rather than new topology
3. **spinal circuitry as a process** — a `segmental_reflex` process on a new
   spinal anatomy, giving the monosynaptic loop at ~30 ms, reciprocal inhibition
   and Renshaw feedback. this is what buys load compensation, and it is the same
   shape as the four missing subcortical processes in STATE.md §4.9
4. **cerebellar process** (mechanism 18) — anatomy is already declared and unused;
   this is what buys bandwidth back above 3 Hz
5. **efference copy** (mechanism 19) — the return path for `efferent_propagation`
6. **fan-in gain prior** — stage 0, shared with everything else

1 and 2 alone move it from "cannot close a motor loop in principle" to "closes a
slow one." 3 and 4 are what make it move like a body.

**and this is mechanism 21 — active sensing — which ONTOLOGY.md §5 and
CURRICULUM.md stage 4 already name as the closure the whole curriculum is aimed
at.** a body simulation is not a side quest from the video loop; it is the same
missing loop with a physical world instead of TRIBEv2 standing in for one.

---

## 5. what was built

### 5.1 the nerves, as partitions

six anatomical systems, 192 labels, all on the `body` frame because there is no
image in which the median nerve is a path between two brain positions:

| system | labels | content |
|---|---|---|
| `cranial_nerves` | 21 | I–XII with V and VII divisions enumerated separately, and the vagus split pharyngeal / recurrent laryngeal / cardiac / pulmonary / abdominal |
| `spinal_levels` | 31 | C1–Co1 |
| `peripheral_nerves` | 58 | four plexuses and the named trunks they form, plus the sympathetic chain and splanchnics |
| `dermatomes` | 29 | root skin territory, `crisp=False` because adjacent dermatomes genuinely overlap |
| `myotomes` | 29 | root muscle territory, non-crisp because nearly every limb muscle draws from two or more segments |
| `autonomic_ganglia` | 24 | where preganglionic axons synapse and diverge |

each has an `ibm.anatomy.sources` entry, because the registry refuses a system
with no recorded provenance. those entries say something the cortical ones do
not: **there is no probabilistic atlas of the median nerve.** what exists is
dissection literature, and the consequence is the opposite of the cortical case —
these memberships will not be silently substituted at materialization, because
there is nothing to substitute them with.

### 5.2 the fibre classes, as components

ten new components on `neural`, following `ibm.anatomy.systems`'s own rule —
things that tile are partitions, things that coexist are components, and fibre
classes coexist in every millimetre of every trunk:

- afferent: `ia` (spindle primary, length + velocity), `ib` (Golgi tendon,
  force), `ii` (spindle secondary, static length), `abeta` (cutaneous
  mechanoreception), `adelta` (first pain, cold), `c` (second pain, warmth, itch,
  most visceral traffic)
- efferent: `alpha` (extrafusal, force), `gamma` (intrafusal, **the one efferent
  whose target is a sensor**), `b_preganglionic`, `c_postganglionic`

**so a nerve is the product of a partition and a component vector.** neither half
means anything alone: a trunk with no fibre vector is a wire with one number on
it, and a fibre vector with no trunk has nowhere to be.

### 5.3 the topology that makes the multidimensionality bite

`ibm/topologies/nerve.py` declares `peripheral_nerve` with a delay **per fibre
class**, a composition table per trunk, and literature trunk lengths. measured:

| trunk | Ia | A-beta | A-delta | C |
|---|---|---|---|---|
| sciatic | 6 ms | 10 ms | 37 ms | **550 ms** |
| median | 7 ms | 13 ms | 47 ms | **700 ms** |
| sural | — | 7 ms | 27 ms | 400 ms |

**a hundredfold spread inside a single trunk.** a topology that gives a nerve one
`conduction_delay_s` asserts that first and second pain arrive together, that a
stretch reflex and a thermal percept share a latency, and that fusimotor drive
reaches the spindle when alpha drive reaches the muscle. it is not an
approximation, it is a category error.

the composition table carries content too: `sural` has no Ia and no alpha, so
cutting it costs sensation and no strength; `anterior_interosseous` has no
A-beta, so cutting it costs strength and no sensation. that asymmetry is anatomy
the model can only express because presence and absence are declared per trunk.

### 5.4 proprioception, and the loop it closes

four receptors (`spindle_primary`, `spindle_secondary`, `golgi_tendon`,
`joint_receptor`) and the two plant states they read (`effector.length`,
`effector.velocity` — which `effector.force`'s own docstring already assumed and
which did not exist).

everything is wired, with **zero orphaned components**: every one is written by
some process and read by another. and the result is a genuine cycle —

    effector_activation --length, velocity-->  transduction
    transduction        --spindle Ia/Ib/II-->  afferent_propagation
    afferent_propagation --------------------> efferent_propagation
    efferent_propagation --alpha----------->   effector_activation
                         --gamma----------->   transduction  (spindle gain)

**this is the first closed feedback cycle in the declared process graph.** note
it does not change STATE.md §6.2, which is about what `eeg_forward` materializes
— that request instantiates no periphery, so its graph is still acyclic. what
changed is that a request which *does* include the periphery now has a loop to
materialize.

the gamma edge is the one worth pointing at. it makes spindle gain a controlled
variable, which is what alpha-gamma co-activation needs: without it every spindle
falls silent during exactly the shortening movements it is needed for.

### 5.5 what this does not fix

- **§2.2 stands.** there is still no spinal circuitry. `spinal_levels` and
  `myotomes` now exist as the partitions a `segmental_reflex` process would be
  defined over, and `neural.afferent.ia` is the component its monosynaptic arc
  would read — but the process is not written, so there is still no stretch
  reflex, no reciprocal inhibition and no Renshaw feedback
- **§2.3 stands.** no cerebellar process, no efference copy. the ~75–135 ms loop
  still has no predictor, so the bandwidth estimate of 2–3 Hz is unchanged
- **the implementations are declarations.** `transduction` now has proprioceptive
  outputs, and no implementation computes them yet; the existing eight cover the
  other receptors. an implementation of the spindle — with its gamma-dependent
  gain — is the next concrete piece
- **§2.5 stands.** the fan-in gain problem is upstream of all of this

---

## 6. the wire list

`ibm/embodiment.py` derives the I/O contract from the declarations, so it cannot
drift out of sync the way a hand-written interface document would.
`python -m ibm.embodiment` prints it.

**498 ports over 96 named muscles, 9 receptor surfaces and 5 visceral trunks:**

| group | count | direction | content |
|---|---|---|---|
| `motor_out` | 186 | ibm-1 → simulator | per muscle, an **alpha** drive and a **gamma** drive |
| `plant_in` | 288 | simulator → ibm-1 | per muscle, length (L0), velocity (L0/s), force (N) |
| `sensor_in` | 9 | simulator → ibm-1 | luminance, sound pressure, skin pressure/displacement/temperature, vestibular acceleration, chemical, blood pressure, oxygenation |
| `visceral_in` | 15 | simulator → ibm-1 | gastric and intestinal distension and nutrient load, GI absorption, hepatoportal glucose, pulmonary stretch, aortic baro- and chemoreception, foregut mechano and ischaemia, midgut distension, renal, bladder |

### the visceral group is shaped differently, and that is its whole content

a `sensor_in` port carries one latency per receptor **surface** — 12 ms for skin,
0.4 ms for the vestibular organ, and until this group existed, a flat 60 ms for
"viscera". that flat number is wrong by a factor of 55.

the visceral ports carry a delay per **(trunk, fibre class)** pair, over the
route lengths IHM measured:

| group | route | delay | what arrives |
|---|---|---|---|
| vagus / A-beta | 508 mm | **9.2 ms** | gastric and intestinal volume, lung volume, aortic pressure |
| vagus / A-delta | 508 mm | 33.9 ms | intestinal absorption |
| pelvic splanchnic / C | 128 mm | 128.0 ms | bladder filling |
| greater splanchnic / C | 168 mm | 168.4 ms | high-threshold gastric distension, lactate |
| lesser splanchnic / C | 217 mm | 217.2 ms | high-threshold intestinal distension |
| least splanchnic / C | 232 mm | 232.2 ms | renal |
| vagus / C | 508 mm | **507.8 ms** | nutrient load, portal glucose, aortic chemoreception |

**the stomach reports twice, on two nerves, half a second apart.** the vagal
A-beta channel is a low-threshold volume report at 9 ms; the greater splanchnic C
channel is a high-threshold, nociceptive report of the same organ at 168 ms, and
it stays silent unless the stomach is genuinely overfull. that pair is what the
trunk/fibre-class split exists to express, and a single visceral latency asserts
they are the same event.

it is also why interoception is *late*. a gut feeling is slow and a touch is not,
and the ratio here — 55x within one nerve — is the largest in the model.

**the cortical target is a substitution and it is named.** interoceptive afference
reaches insula and anterior cingulate. `cortical_regions` in
`scripts/pretrain_video_loop.py` is a six-label geometric convention over a
spherical proxy and a sphere has no lateral sulcus, so the insula is not separable
there; the drive enters the `frontal` label, subsampled to the insula+ACC share of
cortical surface (~4.5%). the DK parcellation in `ibm/anatomy/systems.py` does
declare `insula`, `rostralanteriorcingulate` and `caudalanteriorcingulate`, so a
materialization through `ibm/materialize/build.py` has the real target and the
substitution ends there. `ibm.interoception.PORT_SUBSTITUTION` is a constant so it
prints in every report rather than living in a comment.

**one declared gap.** there is no viscera-supported nociceptor component:
`transduction.nociceptor` sits on the transduction field's default support, the
body surface. the splanchnic channels are nociceptive by threshold and by fibre
class and have nowhere in the ontology to say so, so they bind to
`transduction.baroreceptor` — whose own docstring covers "the visceral
mechanoreceptors that report gut and bladder distension" — and are tagged
nociceptive in the row. recorded as `ibm.interoception.ONTOLOGY_GAPS` rather than
fixed by adding a component no process reads.

`ibm/anatomy/muscles.py` is what makes this addressable: 96 muscles, each with its
nerve, its root levels, and a relative spindle density. that table is the
difference between "the model has a motor system" and "the model has a socket
labelled `biceps_brachii`".

**the gamma wire is the one to not omit.** it is not a refinement: without
fusimotor drive the spindle unloads whenever the muscle shortens, so a simulator
that ignores gamma gets a brain whose proprioception switches off during exactly
the movements it is needed for. measured on `spindle_prochazka`:

| state | gamma = 0 | gamma = 50 Hz |
|---|---|---|
| rest, L=1.00 | 10.0 Hz | 25.0 Hz |
| stretched, L=1.10 | 30.0 Hz | 45.0 Hz |
| fast stretch, v=+2 | 121.9 Hz | 228.8 Hz |
| **shortening, v=-2** | **10.0 Hz (floor)** | **25.0 Hz** |

### 6.1 known deficiency in the quoted latencies

the per-port delay is the **trunk** delay from `ibm.topologies.nerve`, not the
full path from the cord. `abductor_hallucis.alpha_drive` reports 2.0 ms because
`medial_plantar` is a 200 mm distal segment; the real cord-to-muscle path is most
of a metre and closer to 10 ms. **the root-to-trunk segment is missing from the
length table**, so every distal port currently understates its latency by roughly
the length of the limb. the fibre-class RATIOS within a trunk are right, and the
absolute values are short. fixing it is a length-table addition, not a structural
change.

## 7. so: is there a wire for everything?

**for muscles and proprioception, yes** — 474 of the 498 ports, derived from the
declarations, with fibre-resolved latencies and working receptor implementations.

**for the viscera, now yes as well** — 15 ports with fibre-class-resolved
latencies and rates computed by IHM-1 from native BioGears state
(`ihm/assembly/interoception.py`), rather than by anything here.

**for the rest, partly.** the 9 sensor ports are one per modality, not one per
receptor field: there is a `skin.pressure` port and not a port per dermatome,
because the dermatome partition exists and no builder instantiates sites on it
yet. a simulator can drive the skin, but not yet drive *a patch of* skin by name.

**and §2.2 still stands.** every port is a conduction path. the cord circuitry
that would close a stretch reflex locally -- Ia → alpha monosynaptic, Ib →
inhibitory interneuron, Renshaw recurrent -- is a process over `spinal_levels`
that does not exist. so the wires are there and the reflexes are not: every
correction has to go all the way to cortex and back.

---

## 8. the cord

`ibm/processes/spinal.py` declares `segmental_reflex` with four arcs, closing
§2.2. it reads the Ia/Ib/II traffic the periphery now produces and writes alpha
and gamma drive directly — the same components `efferent_propagation` writes,
because a motoneuron pool sums descending and segmental input and no experiment
separates them at the axon. **what separates them is latency, which is measurable
and is the entire content of an H-reflex.**

| arc | synapses | sign | what it buys |
|---|---|---|---|
| `monosynaptic_stretch_lti` | 1 | + | length feedback; the only monosynaptic reflex, so its latency is a clean fitting target |
| `reciprocal_inhibition_lti` | 2 | − | makes the reflex joint-level rather than muscle-level; without it a model co-contracts where it should reciprocate |
| `autogenic_inhibition_lti` | 2 | − | force feedback. with the stretch arc it approximates impedance control — length resists displacement, force yields to load, and the gain ratio sets limb stiffness |
| `renshaw_recurrent_lti` | 2 | − | pool gain control; never leaves the cord, so it is the fastest arc |

the inhibitory gains are `normal` and not `weak` priors, because `weak` is
lognormal-backed and cannot express a negative median. the sign is anatomy — these
synapses are glycinergic — so it is not something the prior should leave free to
flip.

**the bandwidth change, computed from the declared nerve table:**

| muscle | nerve | Ia in | alpha out | loop | bandwidth |
|---|---|---|---|---|---|
| biceps brachii | musculocutaneous | 3.0 ms | 3.0 ms | 7.0 ms | 35.7 Hz |
| tibialis anterior | deep fibular | 3.5 ms | 3.5 ms | 8.0 ms | 31.2 Hz |
| soleus | tibial | 5.0 ms | 5.0 ms | 11.0 ms | 22.7 Hz |
| first dorsal interosseous | ulnar | 7.0 ms | 7.0 ms | 15.0 ms | 16.7 Hz |

against **1.8–3.3 Hz** for the long cortical loop (§2.3). these are trunk-only
delays and therefore optimistic by §6.1's admission, but even doubled they leave
the segmental arc most of a decade faster. **it is a different control regime, not
an incremental gain**, and it is why load compensation is possible at all.

the fitting consequence is the one `effector.py` predicted: a model without a cord
"will attribute those to descending command because descending command is the only
thing it has", so corticospinal gain absorbs reflex gain and is then reported as a
property of cortex. that error is now avoidable.

**still open:** the interneurons are implicit — Ia-inhibitory, Ib and Renshaw
cells are real populations folded into transfer functions, which asserts their
only contribution is a delay and a sign. flexor-withdrawal and crossed-extensor
need multi-segmental interneuron chains and are absent entirely.
