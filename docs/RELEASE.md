# checkpoints: naming, contents, and where they go

`ibm/release.py`. weights are published to **`brandonin/ibm-1`** on HuggingFace
(private), uploaded every `--upload-every` steps by the trainer itself, so a run
that dies has already shipped everything up to its last interval.

## 1. why the name is elaborate

a checkpoint here is not identified by a step number. the substrate changes, the
ontology it was declared against changes, the objective changes, and **a tensor of
32 million association embeddings is meaningless without knowing how many sites
they index and what the dynamics between them were.** every field below exists
because getting it wrong produces a checkpoint that loads cleanly and is wrong.

```
ibm1.m-{modality}.s{sites}.e{embed}.k{degree}.dyn-{rev}.obj-{objective}
    .vw{viability}.step{n}.git-{sha7}
```

| field | meaning | why it must be in the name |
|---|---|---|
| `m` | `v` / `a` / `av` | a joint checkpoint's cortical weights are shared across ports; a single-modality one's are not. **not interchangeable even at identical shape** |
| `s` | site count | leading dimension of the embedding table |
| `e` | embedding dim | the learned rank of the association kernel |
| `k` | association degree | sets the fan-in normalization, so two checkpoints at equal shape and different `k` have differently *scaled* weights |
| `dyn` | dynamics revision | bumped by hand when the E/I equations, shunting form or fan-in convention change. **this is the field that catches "the weights load and the model is wrong"** |
| `obj` | `nf1`, `nfh8`, `av` | what was minimized |
| `vw` | −log₁₀ viability weight | belongs in the name because run v1 learned a **430 mV cortex** under no viability term and looked fine by loss |
| `step`, `git` | when, and against which declaration | |

**the git sha matters most here**, because the ontology is versioned in the same
repo as the weights: a checkpoint trained before `ASSOCIATION_KERNEL` became
`PER_SITE` indexes a different object under the same name.

### dynamics revisions
- `r1` initial E/I + learned association
- `r2` conductance-based shunting (STATE.md §4.11), fan-in normalization (§4.10)
- `r3` viability penalty on membrane potential (ONTOLOGY.md §7)
- `r4` multi-step prediction horizon; cross-modal ports; long-range association

example: `ibm1.m-av.s250k.e128.k48.dyn-r4.obj-av.vw1.step012000.git-6ed3d27`

## 2. the sidecar

each `.pt` ships with a `.json` carrying what the filename cannot: the geometry
the sites were drawn on, parameter counts split into total and association, the
metrics at that step, the full config, and two standing notes — that only the
third factor of `w_ij = M[parcel] × exp(−d/ℓ) × σ(⟨eᵢ,eⱼ⟩)` is trained, and that
effective rank is reported and deliberately **not** in the loss.

## 3. the runs

| run | machine | params | association | objective |
|---|---|---|---|---|
| `av_v1` | `gb10-direct` | **68,876,204** | 32.0M | joint audio-visual, horizon 8 |
| `video_v3` | local | 33,352,838 | 19.2M | video only, horizon 8 — the **control** |

the pair is an experiment, not redundancy: the joint model must explain both
streams with one association kernel, and `cross_modal_weight` measures whether it
did. **if the joint run does not beat the video control on video, the shared
cortex bought nothing.**

## 4. a positive association kernel is a diffusion operator

the single most useful thing the held-out diagnostic has caught, and it is
structural rather than a tuning failure.

with `w_ij = geo * sigmoid(...)` every association weight is **positive**, and the
kernel is fan-in normalized. applying it once is a weighted average; applying it
eight times per forward is eight rounds of averaging, which drives every site
toward the graph mean. **rank is destroyed by construction.** measured, with the
corrected metric (512 sites, ceiling 15):

| kernel | r_eff over 150 steps | recon |
|---|---|---|
| `sigmoid` — positive only | 1.57 → **1.01** | 2.04 → **6.66** (diverging) |
| `tanh` — signed | 3.46 → **3.40**, stable | 1.56 → **1.48**, falling |

a rank-one cortical state means all 512 sampled sites are doing the same thing:
ONTOLOGY.md §4's flat-force-field pathology, arrived at from the opposite
direction. **cortical association is not all-excitatory, and a kernel that cannot
subtract can only blur.** `tanh` lets the learned factor place opposition between
sites, and that is what stops the representation washing out.

this also corrects a claim made earlier in the session: the first report of
"effective rank collapsing" was measured over a batch of 4, where the metric's
ceiling is 3. the *relative* decline was real, the magnitude was not, and the
metric now takes its covariance across sites from a dedicated 16-sample pass.

## 5. throughput, measured

caching the edge weights per forward instead of rebuilding the `(N, k, embed)`
similarity inside every dynamics step gave **4× — 2.76 s/step → 0.67 s/step** at
68.9M parameters, 250k sites, batch 4. it was also the memory ceiling: that
tensor is 6 GB at 250k sites and was being built eight times per forward.

| | measured |
|---|---|
| step time | 0.67 s |
| steps/hour | 5,373 |
| samples/hour | 21,493 |
| one epoch over the 11-minute movie | 4,125 steps = **0.77 GB10-hours** |

**not yet applied**: batch is 4, precision is fp32, there is no `torch.compile`.
batch 32 + bf16 + compile is conservatively 6× on this shape, and the estimates in
CURRICULUM.md assume it is available.
