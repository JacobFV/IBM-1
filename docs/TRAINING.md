# the training program: infrastructure, runs, and what they found

ARCHITECTURE/DYNAMICS/ONTOLOGY describe what is declared. CURRICULUM.md and
`ibm/curriculum.py` describe the plan. **this describes what was actually run and
what it actually showed**, because until now those results lived in commit
messages and a `note=` field, which is not where a finding survives.

## 1. where the data is

**not in this repo.** `data/sources/` is 14 MB of cards; the payloads are
elsewhere and nothing in the repo said so:

| path | size | what |
|---|---|---|
| `~/Documents/win-data` | **960 G** | every bound corpus. `aomic-id1000` 91G, `ds000117` 88G, `libribrain` 46G, `meg-masc` 43G, `wan2.2-ti2v-5b` 32G, … |
| `~/Documents/win-data/derived` | 141 G | regenerable decoder output. **not evidence**, and its manifests say so |
| `~/Documents/IBM-1/data/sources` | 14 M | 601 cards, 117 bound |

**a card marked `binding: bound` does not mean the bytes are on this machine.**
`things-eeg2`, `broderick-natural-speech-eeg`, `brain-treebank`, `narratives` and
`natural-scenes-dataset` are all carded bound and none is present. that is a
binding-integrity bug, not a documentation nicety: every plan that assumed those
datasets was planning against files that do not exist.

derived arrays built this session, all under `win-data/derived/`:

| file | shape | from |
|---|---|---|
| `koyaanisqatsi-full/frames_64x64.npy` | (16500, 64, 64, 3) | the aomic movie stimulus, all frames |
| `koyaanisqatsi-full/cochleagram_25fps.npy` | (16498, 64) | **that same movie's own soundtrack**, one audio frame per video frame |
| `sherlock-cochleagram.npy` | (416354, 64) | the Sherlock audiobook, 40 ms hop |
| `libribrain-paired/cochleagram_250hz.npy` | (3534623, 64) | audiobook at the MEG rate |
| `libribrain-paired/meg_250hz.npy` | (3534623, 306) | **235.6 minutes of paired MEG** |

the paired build (`scripts/build_paired_meg.py`) recovers an alignment the dataset
does not record: which chapter a run is, is written nowhere, so it is identified by
matching each run's chapter-time span against the 14 wav durations, and the
MEG-to-chapter offset (18–25 s) comes from the events' `timemeg − timechapter`.

## 2. hardware

both boxes are GB10, aarch64, ~120 GB unified, CUDA 13.0, torch 2.14.0+cu130.
`pip install torch --index-url https://download.pytorch.org/whl/cu130` is all it
took; STATE.md §6 called this the gating step for months and it was one command.

- **local** — holds the data, runs interactive work, kept under half memory
- **`gb10-direct`** — 200G DAC, ~480 MB/s rsync, holds code + copies of the arrays,
  runs the long jobs. `spark-gb10`, `spark-ec4d` and the RTX Pro 6000 are down

## 3. the runs

| run | what | params | result |
|---|---|---|---|
| `video_v1` | next-frame, no viability term | 52.5M | **learned a 430 mV cortex.** loss fell the whole time |
| `video_v2/v3` | + viability penalty | 52.5M | bounded; rank collapsed |
| `av_v5` | joint audio-visual, horizon 8, signed kernel | **68.9M** | **completed 20k steps.** recon 1.77 → 0.185, rank recovered 1.42 → 2.97 |
| `paired_meg_v1` | cochleagram → MEG | 33.4M | 91–97% of MEG variance **at rank 1.0–1.2** |
| `multi_v1` | AV + paired, one substrate | 32.0M shared + 12.9M heads | running |

## 4. what the runs actually showed

### 4.1 the cortex is load-bearing — measured, not argued
`scripts/ablate_cortex.py` on `av_v5`, 24 held-out batches:

| arm | loss | vs full | what it removes |
|---|---|---|---|
| full | 0.180 | — | |
| frozen | 0.503 | **+180%** | everything learned about connectivity |
| no_assoc | 0.503 | **+180%** | lateral communication entirely |
| **bypass** | 0.761 | **+324%** | the dynamics themselves |
| local_only | 0.180 | **+0.1%** | long-range edges |

bypassing the cortex triples the loss, so it is not a delay line. **but severing
every long-range edge costs nothing**, while the occipito-temporal weights sit at
4.4× the random baseline — large and functionally inert.

the cause is the geometric prior: `exp(-d/40mm)` was applied to long-range edges
too, and a uniform-random partner on a 127 mm sphere is ~85 mm away, so those
edges start with 7× less weight than local ones and the learned factor never
overcomes it. fixed by giving them a flat, patchy prior instead — **and that fix
is not confirmed until the ablation is re-run.**

### 4.2 a positive association kernel is a diffusion operator
`sigmoid` weights are all positive and fan-in normalized, so eight applications
per forward are eight rounds of averaging toward the graph mean. rank is
destroyed by construction: r_eff 1.57 → 1.01 while recon *diverged* 2.04 → 6.66.
`tanh` holds rank at 3.4 and recon falls. **a kernel that cannot subtract can only
blur.**

### 4.3 the objective was parasitic on its substrate, and the check caught it
run v1's `|v|max` drifted 64 → 430 mV while the loss kept falling, because a
430 mV membrane predicts frames perfectly well. that is ONTOLOGY.md §7's
definition of parasitism — lowering the loss required physiology the substrate
cannot sustain — and the one-sided viability penalty is the regularizer that makes
the objective mutualistic. it is in the checkpoint name for that reason.

### 4.4 three metric bugs, all the same class
each one produced a confident, wrong conclusion about whether the model was
learning:

1. **effective rank over a batch of 4** — ceiling is 3, so "collapse to 1.0" was
   partly the metric reporting its own ceiling
2. **effective rank over samples, not sites** — the quantity wanted is the
   dimensionality of the state across the sheet
3. **cross-modal similarity over PAIRS, not EDGES** — ~10⁹ pairs against 187,484
   edges diluted the signal ~5000:1, and it read the random baseline for 20,000
   steps while the association was in fact 4.4× baseline

**the pattern is measuring over the wrong population.** and the deeper lesson from
§4.1: even with the right population, *magnitude is not contribution*. a gate
should be an ablation.

## 5. checkpoints
`brandonin/ibm-1` on HuggingFace, private, uploaded by the trainer every
`--upload-every` steps. naming schema and its rationale: RELEASE.md.
**known defect**: remote checkpoints are stamped `git-unknown` because the remote
has no git repo — the sha should be passed in, not computed there.

## 6. scripts
| script | does |
|---|---|
| `pretrain_video_loop.py` | the four single-substrate materializations: video, audio, `av`, `paired` |
| `train_multi_materialization.py` | **several materializations at once over one substrate** — the ∇log p factorization |
| `ablate_cortex.py` | is the cortex load-bearing (§4.1) |
| `build_paired_meg.py` | the aligned MEG/cochleagram build (§1) |
| `sweep_nonlinear_attractors.py` | the attractor phase diagram (STATE.md §4.10) |

## 7. what is still not written down
honest list, so it is not mistaken for completeness:

- **no eval harness.** every number here is a training loss on held-out *batches*,
  not a held-out *split*. the movie is 11 minutes and the model has seen all of it
- **no rollout evaluation.** prediction is not generation; nothing has been asked
  to run free and stay stable
- **12 scripts are undocumented by name** in any doc, including several `measure_*`
  from earlier sessions
- **no seed/variance reporting.** every result above is one run
