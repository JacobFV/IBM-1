# the training trajectory

STATE.md §7b-§7e say what the curriculum must not be (capture), what axes it can
move along, and why it is affordable. **none of that is a schedule.** this is the
schedule: which materialized model, trained by which objective, against which
source, gated on which measurement.

status is measured, not aspirational. **nothing below stage 1 has run.**

---

## 0. the honest position

| question | answer |
|---|---|
| is there a materialized 100M+ parameter model? | **52.5M is TRAINING** — `ASSOCIATION_KERNEL` is now `PER_SITE`, dim 128; 32.0M of those are per-site association embeddings, i.e. the learned cortico-cortical graph |
| has any curriculum stage run? | **stage 4 is running** — `scripts/pretrain_video_loop.py` on `gb10-direct`, 16,500 frames of naturalistic video, 12k steps |
| has theta been moved by evidence? | partially — per-source spectral fits ran; the joint fit is falsified (§4.3b) |
| does the substrate support an attractor landscape? | **yes, measured** (STATE.md §4.10) |

---

## 1. the stages

each stage names its gate. **a stage does not begin until its gate is measured
green**, because every one of them trains against something the previous stage
had to establish.

### stage 0 — fan-in-aware gain prior · NO TRAINING · **BLOCKING**
**RUN, and the blocker is now identified exactly.** the nonlinear materialization
reproduces the divergence to the digit and reports `|L(0)| = 7218` — the DC loop
gain is three orders of magnitude above unity, because the `weak(1.0, 5.0)` gain
prior is **per edge** and a node has 4118 incoming edges. the prior does not know
the fan-in (STATE.md §4.10).

- **do**: scale the lumped per-edge `gain` priors by 1/n_inputs, the same
  variance-preserving scaling any initialization applies. target: total weight
  onto a node below 0.570, i.e. ~1.4e-04 per edge
- **do**: set `tau_adaptation_s` to 0.30 s and raise `eeg_forward`'s overlap from
  512 to 900 ms — one change that clears the windowing refusal AND puts the SO in
  band
- **do**: make window length downstream of implementation selection instead of
  upstream of it
- **gate**: `|L(0)| < 1`; the solve converges; multi-window plan not refused
- **not** a solver problem and **not** a fitting problem — a declaration problem
- **DONE for the training path.** `association_transfer` takes `fan_in` and divides
  by it, and the trainer normalizes the geometric prior by `k`. the dynamics are
  bounded through training (`|v|max` 64.4 mV, flat over thousands of steps).
  the `eeg_forward` materialization still needs the same treatment
- **no longer blocks pretraining.** architecture work and weight training proceed
  in parallel; this is not a release train

### stage 1 — regime selection · SUPERVISED · gated on 0
fit the RATE parameters so the nonlinear model sits in a region that has more
than one invariant set, rather than the single 4.77 Hz fixed point at prior
medians.

- **model**: minimal cortical E/I, `wilson_cowan_adaptive` + `shunting_rate`
- **objective**: measured spectra + the SO band, likelihood on `sleep-edfx` N3
- **known target**: `tau_adaptation_s` = 0.30 s gives 0.533 Hz (STATE.md §4.10)
- **gate**: bounded; >1 invariant set; SO in 0.5–1.0 Hz; oscillatory region
  reachable at `w_ei` > 0
- **measure**: invariant-set count, effective rank — **the ONTOLOGY.md §9
  instruction to build the order parameters alongside the nonlinear work lands
  here**, not later

#### the same parameter sets the SO frequency AND makes the model windowable

`run_eeg_forward` declares the RATE coupling's `memory_s = 3.0 *
tau_adaptation_s`, so selecting the nonlinearity gives the graph a 1.5 s memory
at the prior median — longer than any conduction delay in the model — and the
multi-window plan is **refused twice**: the overlap (512 ms) is shorter than the
memory, and the memory is over half the 2.048 s window so almost nothing in it is
dynamics.

that is a real architectural finding on its own: **the window is chosen against
the target, and the request has no way to know which implementations the target's
trace will select.** window length is downstream of implementation selection and
is currently upstream of it.

but it also means one parameter carries both constraints:

| tau_a | memory (3·tau) | fraction of window | over half? | SO frequency |
|---|---|---|---|---|
| 0.50 (prior median) | 1.50 s | 0.732 | **no — refused** | 0.333 Hz (below band) |
| 0.35 | 1.05 s | 0.513 | **no — refused** | — |
| **0.30** | **0.90 s** | **0.439** | **clears** | **0.533 Hz — in band** |
| 0.20 | 0.60 s | 0.293 | clears | (no oscillation) |

**tau_adaptation_s = 0.30 s is the value that simultaneously puts the slow
oscillation in the cortical SO band and drops the memory under half the window**,
leaving only an overlap raise from 512 ms to 900 ms — which the refusal
explicitly permits. below 0.20 s the oscillation disappears entirely (STATE.md
§4.10), so the viable band is narrow and the target is not arbitrary.

this is the concrete stage-1 objective: fit `tau_adaptation_s` against measured
N3 spectra and expect it to land near 0.30 s, then raise `eeg_forward`'s overlap
to match the fitted memory rather than to a constant.

### stage 2 — per-source spectral fit · SUPERVISED · gated on 1
extend the existing `fit_neural_spectra` work onto the nonlinear model.

- **sources**: eegmmidb, sleep-edfx, ds000117, ds004873
- **known correction owed**: the current values are scalp quantities reported as
  cortical (STATE.md pending list) — redo, do not inherit
- **do NOT** re-attempt the global joint fit: falsified in §4.3b, and the
  residual is cohort/arousal/head, not statistics

### stage 3 — forced sensory alignment · DISTILLATION · **BLOCKED ON DATA**
teach occipital dynamics to evolve with the visual stream and temporal with
audio, by distilling from TRIBEv2.

- **objective**: regression to teacher latents, with the calibrated certainty
  signal; precision `ΔJ = r²/((1-r²)Var[x])`, rank-1 bound only
- **blocker**: **TRIBE v2 weights are not held** — both cards have
  `local_root: null`. this is the single highest-value acquisition
- **fallback if it stays blocked**: `naturalistic-neuroimaging-database`,
  `sherlock-merlin-princeton` are held and give measured film-viewing fMRI/EEG,
  which forces the same pathway with a weaker teacher

### stage 4 — closed loop · SELF-SUPERVISED · gated on 3
frame → TRIBEv2 → EEG → IBM dynamics → EEG′ → TRIBEv2 → next frame.

- **objective**: next-frame likelihood **plus an expansion term** — predictive
  loss alone is a capture curriculum by construction (STATE.md §7d)
- **selection rule**: stimuli where error is high **and declining**
- **gate to advance**: effective rank and invariant-set count RISE while loss
  falls. if loss falls and they do not, the run is capturing and must stop
- **hold out**: at least one expansion measure, or it stops being a diagnostic

### stage 4b — the audio-visual materialization · RUNNING
one cortex, two ports, two predictions. video drives an occipital port, audio a
temporal one, and **the cortical parameters between them are the same tensor** —
so the model must explain both streams with one association kernel, and the only
way to beat two independent models is to use the fact that the streams are
correlated. that is what learning an occipito-temporal association means.

- **the audio must be the video's OWN soundtrack.** pairing these frames with an
  unrelated audiobook presents two independent streams, and the correct thing to
  learn would be that vision and hearing do not interact. so the Koyaanisqatsi
  score is cochleagrammed at exactly 25 fps: 16,498 audio frames against 16,500
  video frames, indexing the same instants
- **long-range association edges are required.** a pure k-NN graph is a local
  sheet on which occipital and temporal sites are not connected at all, so no
  amount of training could associate them. 25% of each node's degree is spent on
  distant partners drawn uniformly — the minimal declaration of patchy
  association fibres — and the learned factor is free to keep or discard them
- **measured by `cross_modal_weight`**, read off the learned kernel rather than
  inferred from behaviour
- **control**: a video-only run at the same horizon. if joint does not beat it on
  video, the shared cortex bought nothing

### stage 5 — multi-materialization schedule · SUPERVISED · **free today**
`∇log p(θ|D) = ∇log p(θ) + Σ_d ∇log p(D_d|θ)`. a curriculum is a schedule over
`d`: spectra → evoked → hemodynamic → behaviour.

- **why free**: `ibm/forge/fit.py` already takes a sequence of `Task`s; only the
  schedule is missing. this is the one stage with no new machinery
- **runs on**: both GB10s, dataset-parallel — the factorization IS the
  parallelization

### stage 6 — developmental · SELF-SUPERVISED · needs mechanism 26
spontaneous structured activity before structured input, then a plastic window,
then a closing critical period.

- **why it matters**: a critical period is a scheduled window of expansion
  followed by consolidation — the one axis that answers §7d structurally rather
  than with a regularizer
- **missing**: maturation-dependent priors. our priors are static

### stage 7 — cognitive schema, then RL · needs task semantics
- **missing**: `task_cue` is declared and carries no content, which blocks six of
  the thirteen intervention primitives (ONTOLOGY.md §8)
- **missing**: mechanisms 5, 7, 8 (basal ganglia, hippocampal indexing, replay).
  the anatomy is declared and no process uses it (STATE.md §4.9)
- **declared path that already exists**: reward → neuromodulator → plasticity → θ
  is a real route, not something to bolt on (§7c)

---

## 2. which axis each stage moves

against STATE.md §7e. conventional curricula only ever have the first column.

| stage | data order | regime | intervention | plasticity schedule | materialization |
|---|---|---|---|---|---|
| 1 regime selection | — | — | — | — | — |
| 2 spectral fit | ✓ | — | — | — | — |
| 3 forced alignment | ✓ | — | — | — | — |
| 4 closed loop | ✓ | — | — | — | — |
| 5 multi-materialization | ✓ | — | — | — | **✓** |
| 6 developmental | ✓ | ✓ | — | **✓** | ✓ |
| 7 schema + RL | ✓ | ✓ | ✓ | ✓ | ✓ |

**stages 1–4 are a conventional curriculum.** that is the honest reading of the
video agenda: it is scaffolding that builds the loop machinery the distinctive
axes need. the axes only an IBM has do not appear until stage 5, and most not
until 6.

---

## 3. what runs where

both machines are now GB10 + CUDA 13.0, ~120 GB unified each.

| | local | `gb10-direct` |
|---|---|---|
| torch | 2.14.0+cu130 | 2.14.0+cu130 ✓ |
| repo | source of truth | synced, seals pass |
| data | 816 GiB held | code only |
| role | interactive, ≤half memory | **long runs, sweeps, dataset-parallel** |

**parameters are cheap; state is expensive.** 73M params is 0.27 GiB; one
materialization's state at 0.68 mm is 6 GiB before psd and workspace. so you
cannot batch materializations — but dataset-parallel is the architecture's own
factorization and needs no new idea.

---

## 4. the 100M question, priced

**there is no 100M-parameter model, and materializing one is not the obstacle —
the declaration is.**

`ASSOCIATION_KERNEL` is declared `embed_dim=8, tying=EMBEDDING, symmetric=True`.
measured `param_count`, at 360 HCP-MMP partitions:

| tying | dim | 3.25 mm (13.5k sites) | 1.0 mm (143k) | 0.5 mm (570k) |
|---|---|---|---|---|
| `GLOBAL` | any | 3 | 3 | 3 |
| `PER_PARTITION` / `EMBEDDING` | 8 | 2,882 | 2,882 | 2,882 |
| `EMBEDDING` | 128 | 46,082 | 46,082 | 46,082 |
| `PER_SITE` | 8 | 108,002 | 1,140,722 | 4,562,882 |
| **`PER_SITE`** | **128** | 1,728,002 | 18,251,522 | **73,006,082** |

**the declared kernel is resolution-independent by construction** — 2,882
parameters whether there are 13.5k column nodes or 570k, because EMBEDDING ties
to areas. that is why refining the mesh has never produced a bigger model.

reaching ~73M is a **two-field change**: `tying=PER_SITE`, `embed_dim=128`, at
0.5 mm. the architecture already prices it and `nn.py` states the consequence
plainly: at 10⁵ positions "no available dataset distinguishes" them, and "the
posterior will stay near the prior — which is exactly the behaviour the
architecture describes and not a bug."

**that is the real decision, and it is not a technical one.** the intent recorded
in STATE.md §6 is that macro structure constrains coarse modular connectivity
while fine cortico-cortical connectivity is LEARNED, with hundreds of millions of
degrees of freedom. `PER_SITE` is how that is expressed. the architecture's
objection — no dataset distinguishes them — is answered by the closed loop, whose
whole point is unbounded data. but that answer only holds **after stage 4 runs**.

so: **do not flip the tying before stage 4.** flipping it now buys 73M parameters
whose posterior equals their prior, which is a larger model and not a better one,
and it would make every measurement in stages 1–3 slower without making one of
them sharper.

---

embodiment -- whether this can be put in a body simulation, and what breaks --
is [EMBODIMENT.md](EMBODIMENT.md). it is mechanism 21 with a physical world in
place of TRIBEv2.

## 5. the critical path

1. **stage 0** — solver convergence. blocks everything
2. **stage 1** — regime selection, and build the order parameters while doing it
3. **acquire TRIBE v2** — in parallel, blocks stage 3, gates the whole video arc
4. **stage 5** — the schedule over `d` is free and needs no gate but stage 1
5. **flip `PER_SITE` only after stage 4 has data to justify it**
6. **give `task_cue` content** — the seam where stage 7 enters

open items not on the path but still owed: implementation selection is
device-blind (§4.4); `fit_neural_spectra` reports scalp as cortical; the measured
microcircuit priors are exposed but not wired into `processes/neural.py`; four
anatomical systems are requested by library models with no process behind them
(§4.9).
