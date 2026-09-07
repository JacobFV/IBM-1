# IBM-1 — an implicit brain model

A brain declared **once**, as four primitives, from which task-specific models are
materialised on demand.

An EEG forward model, a hemodynamic model and an audio-visual predictor are
different *projections* of one parameter set — not separate codebases. The bet is
that most of the model is shared across materialisations, so every corpus
constrains every task.

```
ibm = (fields, anatomy, topologies, processes)
```

**That bet is testable, and so far it holds.** Bypassing the shared cortical
dynamics in a trained model costs **+324%** loss; resetting the learned
cortico-cortical weights costs **+180%**. The substrate is load-bearing rather
than decorative — which was the failure mode that would have made every other
number here meaningless.

---

## What exists

| | count |
|---|---|
| fields / state components | 13 / 113 |
| anatomical systems | 20 |
| interaction topologies | 17 |
| processes / implementations | 30 / 114 |
| observations / interventions | 27 / 21 |
| catalogued sources / held | 601 / 38 |

The registry **validates on every load**. It refuses a process that reads a
variable nothing writes, an anatomical system with no recorded source, a band a
component cannot carry, and a dangling selector. Several design errors were
caught by that check before they reached a result.

```bash
python -c "import ibm; ibm.load_all(seal=True, strict=True)"   # the whole ontology
python -m ibm.curriculum                                       # the training DAG and its frontier
```

## What has been measured

- **The slow oscillation, fitted to real sleep.** The SO peak across eight scored
  overnight recordings is **1.000 ± 0.296 Hz**; one declared parameter — the
  adaptation time constant — reproduces exactly 1.000 Hz at 0.12 s, with a 43.5 mV
  swing.
- **The cortex beats the encoder it was meant to merely not obstruct.** Image →
  cortical dynamics → embedding, contrastively aligned with measured 64-channel EEG
  (THINGS-EEG2), concept-disjoint held-out split: **26.6% top-1 over 20 pools of
  200 — 53× chance**, against a dynamics-free control reaching 21.5% on the same
  pairs. Bypassing the dynamics retains 41%; freezing the learned association
  retains 65%, and that association's contribution grew 15% → 24% → 35% over one
  run as it trained.
- **A body to inhabit.** 96 named muscles with nerve and root levels, 58 nerve
  trunks with fibre-class-resolved conduction (group Ia at 6 ms against
  unmyelinated C at 550 ms down the same sciatic nerve), four spinal reflex arcs,
  and a **483-port** interface a body simulator connects to.
- **Two claims we could not reproduce.** A shared-parameter fit across four EEG
  corpora was *falsified* under controlled placebos, and the obvious explanation —
  differing electrode montages — was eliminated: a deliberately scrambled montage
  assignment absorbed as much disagreement as the correct one.

## What this is *not* claiming

Stated plainly, because the next report has to be consistent with this one:

- **retrieval, not reconstruction** — the visual result above is discrimination.
  Waveform regression on the same data peaks at skill **+0.011**; speech→MEG peaks
  at **+0.036**. Those are the ceilings, and no head here beats them.
- **no video generation** — predictions reproduce frame statistics, not content
- **single-seed** — nothing here is replicated across seeds
- **11 minutes of video** — data, not compute, binds the self-supervised milestone

**Ten** confident internal results have been overturned by measurement so far, and
[`docs/LOG.md`](docs/LOG.md) keeps the ledger with the check that caught each one.
They share a shape: a quantity computed correctly and then compared against the
wrong thing — the wrong population, the wrong units, the wrong split, or no
baseline. Not one was a modelling error. The worst was a loss computed in the
wrong units that flattered itself by 280× while never beating a zero baseline; the
largest was an image-EEG pairing that was 99.94% wrong and produced three hours of
confident negative results about a corpus that was fine.

**`ibm/evaluate.py` reports every metric as skill against explicit baselines,
including trivial ones, because a trivial baseline is what caught the worst of
them.**

## Layout

```
ibm/            the declaration: fields, anatomy, topologies, processes, registry
  evaluate.py   baselines, contiguous splits, materialisation profiles
  curriculum.py the training DAG, with gates as predicates
scripts/        materialisation, fitting, training, acquisition, evaluation
docs/           ARCHITECTURE · DYNAMICS · ONTOLOGY · EMBODIMENT · CURRICULUM ·
                TRAINING · PROGRAMME · LOG · EVIDENCE · STATE · RELEASE
data/sources/   601 source cards; payloads under raw/ (gitignored)
```

Start with [`docs/PROGRAMME.md`](docs/PROGRAMME.md) — state, compute and
trajectory — then [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the four
primitives.

## Weights

Checkpoints publish to **[jacob-valdez/ibm-1](https://huggingface.co/jacob-valdez/ibm-1)**
with a naming schema that records substrate size, dynamics revision and the exact
commit of the ontology they were trained against, so any result is reproducible
against the declaration that produced it:

```
ibm1.m-av.s250k.e128.k48.dyn-r4.obj-av.vw1.step012000.git-6ed3d27
```

## Data

Corpora are catalogued as cards naming the origin the data actually came from — an
OpenNeuro accession, a DOI, an OSF node. Payloads live under `data/sources/*/raw`
and are never committed. Cards that describe a *class* of resource rather than a
dataset say so; artefacts synthesised by the programme are credited as authored;
entries named in a specification and never obtained are marked aspirational.

Datasets carrying attribution requirements record them in their card —
including [THINGS](https://osf.io/jum2f/) (Hebart et al., PLoS ONE 2019;
Stoinski et al., Behav Res 2024), whose images are used for research under a
non-commercial licence and are neither redistributed nor altered.
