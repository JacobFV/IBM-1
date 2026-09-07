# programme state: what exists, what it runs on, and where it goes

the orientation document. ARCHITECTURE/DYNAMICS/ONTOLOGY say what is *declared*;
CURRICULUM and `ibm/curriculum.py` say what the *plan* is; TRAINING says what was
*run*. **this says where the programme stands as a whole** — for a new agent
picking it up, and for anyone who has to describe it to someone outside.

read this first, then `python -m ibm.curriculum` for the live frontier.

---

## 1. the one-paragraph version

IBM-1 declares a brain once, as four primitives — fields, anatomy, topologies,
processes — and materialises task-specific models from it lazily. An EEG forward
model, a hemodynamic model and an audio-visual predictor are different
*projections* of one parameter set, not separate codebases. The bet is that most
of the model is shared across materialisations, so every corpus constrains every
task. **That bet is now testable and, so far, standing**: bypassing the shared
cortical dynamics costs +324% loss, so the substrate is load-bearing rather than
decorative.

---

## 2. what is built

| | count | note |
|---|---|---|
| fields / state components | 13 / 113 | each with units, bounds, band and an uncertainty form |
| anatomical systems | 20 | cortex and thalamus through spinal levels, dermatomes, 96 named muscles |
| interaction topologies | 17 | local, laminar, tractometric, vascular, afferent, efferent, peripheral nerve |
| processes / implementations | 30 / 114 | a process declares (inputs, outputs, topology); implementations compete |
| observations / interventions | 27 / 21 | EEG, MEG, BOLD, iEEG … / TMS, tACS, pharmacological, sensory |
| catalogued sources / held | 601 / 36 | 817 GB in-repo, provenance traced to origin |
| specification / implementation | 5,397 / 66,307 lines | 10 documents |

**the registry validates on every load.** it refuses a process that reads a
variable nothing writes, an anatomical system with no recorded source, a band a
component cannot carry, and a dangling selector. three design errors this period
were caught by that check before they reached a result. an agent that adds a
declaration will find out immediately if it is inconsistent.

---

## 3. compute

two NVIDIA GB10 boxes, aarch64, CUDA 13.0, torch 2.14.0+cu130. **unified memory,
so "GPU memory" and "RAM" are the same pool** — this is why site counts above
~300k get expensive quickly rather than hitting a hard wall.

| | local | `gb10-direct` |
|---|---|---|
| GPU | GB10 | GB10 |
| unified memory | 121 GB | 119 GB |
| cores | 20 | 20 |
| disk free | 637 GB of 3.6 TB | 1.2 TB |
| holds | all corpora + derived arrays | code, and copies of the training arrays |
| role | interactive work, evaluation, smaller runs; **keep under half memory** | long runs, sweeps, dataset-parallel |
| link | — | 200G DAC, ~480 MB/s rsync, `gb10-direct` |

`spark-gb10`, `spark-ec4d` and the RTX Pro 6000 (`supacomputa`) are configured
and were unreachable throughout this period.

### measured throughput
one number matters more than the flop rate: **step time scales badly in site
count**, because the association kernel gathers `(N, k, embed)` per forward.

| configuration | params | s/step | 40k steps |
|---|---|---|---|
| paired, 150k sites, batch 16 | 33.4M | 0.83 | 9.2 h |
| multi (AV + MEG), 300k sites, batch 8 | 93.7M | 2.30 | 25.6 h |
| multi, 570k sites, batch 4 | 171.6M | 13.3 | **148 h — abandoned** |

570k was tried and dropped: 2.3× the parameters for 5× the wall clock is the
wrong point on the curve while the training data is eleven minutes of one film.
**300k is the current operating point.** an `einsum` in `edge_weights` removed a
14 GB intermediate that autograd was keeping; batch size, bf16 and
`torch.compile` are the remaining unclaimed ~6×.

### bandwidth policy
**HuggingFace traffic is unmetered for planning purposes** — checkpoint uploads,
migrations and weight downloads are not budgeted and should not be deferred for
bandwidth reasons.

**everything else carries a 300 GB ceiling.** that is corpus acquisition:
OpenNeuro, OSF, Dryad, archive.org. spend it on PAIRED stimulus-brain data, which
is the scarce resource. unpaired naturalistic video is not worth a byte of it —
it is effectively free and effectively unlimited, and the self-supervised term
that consumes it is the *volume* term, not the constraining one.

spent so far this cycle: ~50 GB (THINGS-EEG2 40.4, THINGS images 5, a discarded
public-domain film experiment ~6). **~250 GB remains**, which covers narratives
(ds002345, 144 GB) with room over.

**do not delete a held corpus to reclaim disk** — 600+ GB is free, and a
re-download costs budget that paired data should get instead.

---

## 4. what has been measured

### results that held
- **the cortex is load-bearing.** bypass +324%, frozen weights +180%, association
  zeroed +180%. the model is not routing around its own dynamics
- **the slow oscillation, fitted to sleep.** SO peak across 8 scored N3
  recordings is 1.000 ± 0.296 Hz; `tau_adaptation_s = 0.12 s` reproduces exactly
  1.000 Hz with a 43.5 mV swing, and satisfies the windowing constraint
- **cross-modal association is learned.** occipito-temporal edge weights reach
  4.4× the random-pair baseline, signed, from a joint audio-visual materialisation
- **the periphery is connectable.** 96 muscles, 58 nerve trunks with
  fibre-class-resolved delay (Ia 6 ms vs C 550 ms down the same sciatic), four
  spinal reflex arcs, a 483-port simulator interface

### claims overturned by measurement
recorded because the pattern matters more than any one of them:
1. **a positive association kernel is a diffusion operator** — all-positive
   weights applied repeatedly average toward the graph mean and destroy rank by
   construction. a signed kernel fixed it
2. **the objective went parasitic on its substrate** — membrane potential drifted
   to 430 mV while loss fell, because a 430 mV cortex predicts frames fine
3. **a free readout substitutes for the cortex** — skill +0.94 at cortical rank
   1.03. constraining the lead field to MEG's real ~64 spatial DOF raised skill
   *and* tripled the rank used
4. **a loss in the wrong units flatters itself by 280×** — the MEG target
   normalisation was loaded and never applied, so training minimised against an
   array with 3e4 less variance, and never beat the zero baseline in either
   coordinate system

**the recurring failure is measuring over the wrong population, or not checking
that a fix landed.** three metric bugs and one dead variable, each producing a
confident wrong number. an agent working here should treat every reported metric
as unverified until it has a baseline beside it.

### not demonstrated
- **no task performance.** stimulus→MEG skill hovers at chance
- **no video generation.** predictions reproduce frame statistics, not content
- **single seed everywhere.** nothing replicated
- **11 minutes of video.** data, not compute, binds the next milestone

---

## 5. trajectory

five of thirteen curriculum stages complete, three running. `python -m
ibm.curriculum` is authoritative; this is the shape.

**near term — weeks.** acquire naturalistic video at scale (bandwidth-bound, not
compute-bound). blob persistence on held-out video is ~3 GB10-hours once the data
exists; two-dimensional motion 15–30 GB10-hours optimised. finish the joint
schedule over both likelihood terms and show one parameter set beating each
single-corpus fit on its own held-out split.

**medium term — months.** re-run the ablation against the corrected long-range
prior; the flat patchy prior is applied and unverified, and the last time
cross-modal weights were large they were also functionally inert. wire the four
subcortical processes whose anatomy is declared and unused — basal ganglia,
hippocampal indexing, replay, cerebellum. these are the cheapest real progress
available because the partitions and supports already exist.

**the gate on everything cognitive.** a stable multi-attractor landscape has to
exist before curriculum, schema or reinforcement mean anything. the substrate now
admits one — multistability and oscillation are both present and nothing diverges
— but θ has to be *fitted* into those regions, which are ~0.05% of the swept
parameter space. that is stage 2 work and it is unblocked.

**what would falsify the programme.** if the joint schedule does not beat
single-corpus fits, the shared-substrate claim is wrong and the architecture's
main advantage evaporates. if severing long-range edges keeps costing nothing
after the prior fix, cross-area communication is not happening and no cognitive
stage is reachable. both are measurable within the current compute budget, and
both should be run before scaling further.

---

## 6. for an agent picking this up

- `python -m ibm.curriculum` — the live DAG and frontier
- `ibm/evaluate.py` — **use it.** every metric reported as skill against explicit
  baselines, splits contiguous with a guard band. a loss without a baseline is not
  a result, and that is not a stylistic preference here, it is the lesson of §4
- `docs/TRAINING.md` — the runs and what they showed
- `docs/RELEASE.md` — checkpoint naming; weights publish to `brandonin/ibm-1`
- corpora are in `data/sources/<id>/raw`, gitignored; `card.yaml` names the origin
- **do not commit weights or caches.** history was rewritten once to remove 2.9 GB
  of them
