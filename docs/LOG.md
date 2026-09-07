# lab log

chronological, newest first. TRAINING.md holds the runs and PROGRAMME.md the
state; this holds the **sequence** — what was tried, what it measured, and what it
overturned.

it is kept because this programme's most useful output so far is not a result, it
is a list of confident results that turned out to be wrong and the specific check
that caught each one. that list is only legible in order.

---

## the corrections ledger

every claim that was reported and later withdrawn, with what caught it. the
pattern is worth more than any single row.

| # | the claim | what was actually true | caught by |
|---|---|---|---|
| 1 | shunting stiffness causes the divergence | it is quadratic in `g_i` and real, but a one-population reduction held `g_i` near 0.5 where the error is 1.5x. **a reduction that suppresses the variable a bug is quadratic in reports the bug as absent** | giving inhibition its own population: 44% of the grid diverged |
| 2 | the dynamics are bounded, so the gate is a solver problem | boundedness at prior medians was never the question | the same two-population sweep |
| 3 | the divergence is the shunting form | `\|L(0)\| = 7218` — the `gain` prior is per-edge and a node has 4,118 edges, so the prior does not know the fan-in | `run_eeg_forward` printed the diagnosis before I ran three sweeps |
| 4 | effective rank is collapsing | measured over a batch of 4, where the ceiling is 3. the metric was reporting its own ceiling | computing the same quantity across sites |
| 5 | cross-modal association was not learned | it was 4.4x baseline. the metric averaged over 10⁹ **pairs** when only 187,484 are **edges** — diluted 5000:1 | restricting to edges |
| 6 | ...so the association is load-bearing | 4.4x magnitude, **-0.06%** when severed. **magnitude is not contribution** | ablation |
| 7 | paired MEG reaches skill +0.90 | the normalisation was loaded and never applied; training minimised against an array with 3e4 less variance and **never beat the zero baseline in either coordinate system** | an independent eval disagreeing by 280x |
| 8 | the image→EEG target carries no stimulus information | the pairing was **99.94% wrong** — 10 of 16,540. THINGS-EEG2 uses ten images from each of 1,654 concepts; a directory walk takes every image from the first 1,162 | reading the order the dataset declares |
| 9 | contrastive retrieval reaches 18.5% | single-pool, sd 2.8. and the run **checkpointed on best single pool**, selecting for lucky draws | averaging 20 pools |
| 10 | the joint MEG term is reaching skill +0.45 | that is a **training** loss. held-out is -0.003, and the ceiling is +0.036 | the regression control |

**the shape they share:** a quantity computed correctly and then compared against
the wrong thing — the wrong population, the wrong units, the wrong split, the
wrong baseline, or no baseline. Not one was a modelling error. Every one was
caught by a measurement that could have been run first, and several by one I had
already written.

---

## 2026-09-07 — the visual term works

**the cortex beats the encoder it was supposed to merely not obstruct.**
20 held-out pools, concept-disjoint images:

| arm | top-1 | x chance | retained |
|---|---|---|---|
| full | **26.55%** | 53.1 | 100% |
| frozen association | 17.15% | 34.3 | 64.6% |
| bypass dynamics | 10.82% | 21.6 | 40.8% |

a dynamics-free control on the same pairs reaches 21.5%. so the substrate is not
merely costless, it is an advantage — the first time it has paid for itself
against a fair alternative rather than only surviving an ablation.

**the learned association is compounding** across one run: 15% contribution at
step 1,000, 24% at 1,800, **35% at 3,600**. those are the cortico-cortical weights
earning their place, which is what `PER_SITE` tying was declared for and had never
previously shown up.

**the objective was the whole problem, twice.** waveform regression on the same
data peaks at skill +0.011 and then goes negative; retrieval reaches 53x chance.
they are not the same task — retrieval needs enough structure to tell two evoked
responses apart, regression must reproduce an amplitude at every channel and
sample, and those amplitudes are dominated by trial noise no stimulus predicts.
the same applies to speech: the MEG regression ceiling measured **+0.036**, so
26,000 steps of chance was chasing a target worth almost nothing rather than
failing at an achievable one.

**three throughput facts, each learned by hitting it.** sampling interval is not
the integrator step — 10 ms Euler against a 15 ms membrane constant reached
1.9e6 mV. the evoked head costs one dynamics pass per output **sample**, so it was
cropped to the evoked window (94 h → 21 h). and the contrastive head reads one
state, so simulating 100 steps to use the last was 15.8 s/step (35 h → 5 h).

## 2026-09-06 (evening) — publish, migrate, acquire

repos made public after an audit (no tokens, keys or payloads; 11.35 MiB pack).
history had to be rewritten first: **2.9 GB tracked**, of which 2.5 GB was a
regenerable cache and 379 MB were checkpoints, two over GitHub's 100 MB file
limit. safe because nothing had ever been pushed.

both HF repos migrated `brandonin` → `jacob-valdez`, public, copy-verify-delete.
`move_repo` is refused across accounts, so the ordering mattered: **52 checkpoints
existed only on HF and 3 locally**, and a delete-first migration would have
destroyed 49. an identity guard caught `HF_TOKEN` shadowing both sides before
anything was touched.

acquired the image→EEG half: THINGS-EEG2 (10 subjects, 40 GB) plus the THINGS
images. narratives at 40 subjects. ~85 GB of the 300 GB corpus budget.

## 2026-09-06 (afternoon) — the substrate is load-bearing

**the pivotal ablation.** on the trained AV checkpoint: bypassing the dynamics
costs **+324%**, freezing the learned association **+180%**. the cortex is not a
delay line. that was the result that could have invalidated everything else.

**but the long-range half was inert** (+0.1% to sever) because `exp(-d/40mm)` was
applied to long-range edges too, and a uniform partner on a 127 mm sphere sits
~85 mm away — 7x less weight before learning starts. giving them a flat patchy
prior took severing them from +0.1% to **+27%**.

`s1.regime` passed against measurement: the slow-oscillation peak across 8 scored
N3 recordings is **1.000 ± 0.296 Hz**, and `tau_adaptation_s = 0.12 s` reproduces
exactly that with a 43.5 mV swing.

corpora moved in-repo (817 GB) with provenance traced to origin — a peer
programme's cards named the real providers, and 99 cards split into sourced (12),
class (77), authored (5) and aspirational (5).

## 2026-09-06 (morning) — training starts

first runs. `PER_SITE` tying and a fan-in-aware gain prior took the model from
3,369 trainable parameters to 93.7M. the video loop trained; the predictions were
blocky luminance-matched noise, which the clip showed honestly.

## earlier — the declaration

four primitives sealed and validated on load. peripheral nervous system declared:
96 muscles with innervation, 58 nerve trunks with fibre-class-resolved conduction
(Ia at 6 ms against C at 550 ms down the same sciatic), four spinal reflex arcs,
a 483-port embodiment interface. joint forging falsified under controlled placebos.
