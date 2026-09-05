# state of the effort

a durable record of where ibm-1 actually stands, what has been established by
measurement, what is open, and which decisions have already been made so they are
not relitigated. written because this effort has produced a lot of hard-won
findings that would otherwise be lost between sessions, and because a wrong
number, once written down, gets copied faster than it gets corrected — that has
already happened twice here.

updated 2026-09-05.

## 1. the goal, as currently set

a working implicit brain model that reaches roughly **baseline industry
performance** on standard tasks. that alone is the innovation, because of what it
bridges: one substrate, many materializations, heterogeneous evidence, and a
posterior that says which parts of a prediction rest on measurement and which on
a prior.

this is a **proving ground**. fitting the model to an encoder during pretraining
is acceptable — it structures the weights, and careful fine-tuning comes later.
what must survive that stage is the bookkeeping: calibrated precision, and
provenance that records forced versus evolved.

## 2. what is built and verified

```
13 fields · 91 components · 22 supports · 13 anatomical systems · 16 topologies
29 processes · 106 implementations · 27 observations · 20 interventions · 40 models
111 python files · 53,154 lines · seals strict, 0 errors
117 of 601 cards bound · 58 held · 816 GiB on disk
464 declared parameters: 283 literature, 140 weak, 31 speculative, 10 physics
```

verified by running, not by inspection:

- **materialization works on a real subject** — mne `sample`, 13,647 cortical
  column nodes at 3.25 mm plus 531 subcortical tissue sites, disjointness
  measured at 0.00% by KD-tree, 389,510 state variables, a real 60-channel BEM
  lead field solved for 13,636 of 13,647 sources.
- **the solver runs.** window 0 converges in 19 damped-Picard sweeps with zero
  Newton solves.
- **delay-as-phase-ramp is exact**: 24 association edges declared 2.17–23.84 ms,
  measured phase slopes 2.17–23.84 ms, pointwise agreement 4.4e-16. this is the
  central bet of the temporal-spectral form and it holds to floating point.
- **the spectrum is made by the solve**, not carried in: output β 1.57 with a
  9.77 Hz peak at 32.7× background, from a drive whose own β is 0.17.
- **band-limited fusion is exact**: real EEG moved the posterior 2.03 prior sd,
  and outside the observation's band the maximum change is exactly 0.
- **58 of 58 LTI transfers** evaluate finite and correctly shaped on a real basis.

## 3. what measurement has overturned

ten declared claims have been put where they could fail. six did. this is the
system working, and the list is the most valuable thing in this document.

| claim | outcome |
|---|---|
| the thalamocortical resonance is alpha (10 Hz prior) | **13.45 Hz** — a spindle. held-out +5503 nats/night, p 7.2e-11, 100% of unseen participants |
| the neurovascular chain predicts BOLD | **fails**, r = −0.11, permutation p 0.38 |
| Grubb's law holds | **fails**, slope 0.066 against a declared 0.38 (−5.2 prior sd) |
| MEG resolves finer source structure than EEG | **fails** per channel: 49.8% vs 63.0%, p 0.0033, MEG higher in 12% of 16 |
| Murray's law in the capillary bed | **fails** — half of degree-3 junctions have a child wider than their parent |
| the capillary bed carries most of the resistance (`vascular.py` docstring) | **wrong** — 6–27% of dissipation; it is 85% of the *length* |
| the canonical laminar cascade | **half** — L4→L2/3 and L5→L6 confirm; L6→L4 is 3× *below* distance |
| tractography is reliable edge-by-edge | **no** — recovery 0.456, FDR 0.725, 91 pipelines worth 3.08 votes |
| consensus across tractography pipelines is evidence | **no** — ρ *rises* with threshold, 0.317 → 0.556 |
| thresholding improves tractography | **no** — FDR flat at ~0.73 across a 100× range while recovery falls 0.456 → 0.150 |

## 4. open concerns, in priority order

### 4.1 uncertainty is not propagated (BEING FIXED)
every written component's psd after three windows is bit-identical to its prior.
`solve_window` moves means only; `SpectralGaussian.apply_transfer` — which
already scales psd by |H|² and relation by H², the exact linear push-forward §4
describes — appears in docstrings and never in the solve path. for an all-LTI
materialization, which is what `build` selects, every psd comes back untouched
**silently**. this undermines everything epistemic: evidence fusion,
prior-dominated versus constrained provenance, distillation precision, the tier
system. we currently have a dynamical model with a decorative covariance.

**consequence that must not be forgotten:** every fit so far bypassed the solver,
going straight from measured spectra to process parameters. if propagation
changes those numbers, the three model evaluations need redoing.

### 4.2 models do not share the substrate (BEING FIXED)
34 of 40 named models were reported as blocked by "missing data". they are not.
`ibm/materialize/*.py` contains **zero references** to `tract_prior`,
`vascular_prior` or `microcircuit_prior` — the three measured, tiered priors that
exist precisely for this. `build` catches `MissingData` and records a gap instead
of falling through to the tier. a model needing a tractogram should fall to the
group connectome and *say so*, not fail. §7's own claim is that models are
materialized views of ONE implicit model.

### 4.3 no parameter has ever been moved by evidence in a materialization
provenance reports 0 of 123 θ entries moved. `local_excitation`'s 22 parameters
are shared by 20 of 40 models. if most parameters stay at their priors regardless
of how much data arrives, the model is an elaborate prior. **joint forging across
heterogeneous datasets has never been run** — `fit()` takes a sequence of `Task`s
and every script has given it exactly one dataset.

### 4.4 implementation selection is blind to the device
all 8 `device_coupling` implementations are `Form.LTI` with `prov=physics`, so
under `prefer_lti` they score identically and fall to an **alphabetical
tiebreak**. it selected `acoustic_beam` — an ultrasound beam pattern — for a
60-channel scalp EEG montage, and an LTI diagnostic over `quasistatic_lead_field`
for `em_generation`. two faults: selection ignores the `DeviceSpec` the request
already declares, and `prefer_lti` is a COST policy being used as a CORRECTNESS
selector when the architecture calls CONSTRAINT the stiff limit.

### 4.5 window is not checked against traced memory at build time
`eeg_forward`'s declared `Window(n=2048, dt=1e-3)` is illegal for the graph
`eeg_forward` itself traces. longest memory is 9000 ms and the culprit is not an
axon — it is the electrode interface's 3 s AC-coupling corner. drop that and
600 ms of GABA-B still exceeds the request's own 25% overlap. nothing catches
this until the runtime refuses.

### 4.6 structural gaps recorded, not patched
- `base.py`'s LTI library returns input→output filters; `step` needs dz/dt.
  nothing converts between them. §4's stiff limit is the conversion.
- the ontology never declares **which input drives which output within a
  process**. `local_excitation` reads 5 and writes 9; the cross product is
  nonsense, so `couplings_of` cannot be written generically.
- `em_generation`'s 818,160 lead-field edges run cortical_surface→sensor_array
  while every `electromagnetic` block sits on `head_volume` — the chain is broken
  at its first link.
- `advance` has no mechanism for advancing an **exogenous** component, so window
  n+1 is driven by window n's input.

### 4.7 twelve ontology gaps the corpus revealed
found by binding 111 cards; none were invented around. no cardiac generator (ECG
cannot bind); no oculomotor component (gaze, saccades, pupil unbindable across
bbbd, studyforrest, NSD); no receptor density, cell density or cortical thickness
(hansen-receptors, von-economo, bigbrain have no target); no vessel geometry
(now partly closed — four components added); `device.impedance` declared on
`implanted_array` only while ds003670's records are scalp contacts; `blood.*`
capped at 0.5 Hz, which sits *at* the cardiac fundamental a plethysmograph
measures, so every pulse-oximetry binding is clipped at its own signal.

## 5. decisions already made — do not relitigate

- **the laplacian is over TIME, for one state variable.** not over space. fields
  do not share a spatial support, so there is no single spatial laplacian to
  take. λ_k is monotone in ω_k, so a distribution over the laplacian spectrum is
  a distribution over the power spectrum; the degenerate 2-planes are amplitude
  and phase; off-diagonal covariance is cross-frequency structure.
- **how uncertainty is carried is part of the field declaration**, not a concept
  alongside the four primitives. blood is scalar, neural is spectral.
- **a learned latent is NOT a state variable.** a latent is an encoding, nothing
  exerts pressure on it, and it lives inside a process's f, invisible to the
  state graph. the stimulus itself IS physical and is already declared.
- **(f, θ) is implementation, not a fifth primitive.** a different f is a
  different p(θ) over a differently shaped θ; (I, O, T) is untouched.
- **a support is a SAMPLING of a domain, not a different place.** cortical
  population activity is the same quantity indexed by volume position or by a
  column node; components may be admissible on several and are instantiated on a
  disjoint partition of them.
- **`local` is one relation with two metrics** — euclidean in the volume,
  geodesic on the sheet. the metric belongs to the support. on `tissue` a
  millimetre across a sulcal bank really is a millimetre for potassium; on the
  sheet what lies between two column nodes on opposite banks is subarachnoid
  csf, not interstitium.
- **be permissive with a topology's support, informative with θ.** T(i,j)=0 is
  permanent within a materialization — no parameter exists, so no evidence can
  restore the edge — while an over-included edge can be driven to zero by a
  fitted θ.
- **distillation precision is diagonal MINUS low-rank**, by Woodbury, because the
  residual COVARIANCE is diag + low-rank. the bound n/((1−ρ)+nρ) is **rank-1
  only**: at ρ=0.9 and n=10⁴, rank 1 gives 1.1 effective constraints and rank 4
  gives 889.
- **`via` has a direction.** measured → the chain ends in a process that WRITES
  the component; imposed or exogenous → it BEGINS with one that reads it. may be
  a per-component mapping when one acquisition makes several claims by different
  routes.
- **tractography constrains COARSE MODULAR connectivity only.** its own measured
  FDR of 0.725 says it cannot be trusted edge-by-edge. fine cortico-cortical
  connectivity is LEARNED. see §6.

## 6. the scaling plan

the association weight factorizes into three factors with three different
epistemic statuses, each recorded separately:

    w_ij = M[π(i),π(j)]  ×  exp(-d_ij/ℓ)  ×  σ(⟨e_i, e_j⟩)
           ^tractography    ^geometry        ^LEARNED
           parcel scale     exact            where the DOF live

sizing, from the measured 202,437 mm² white surface and the measured
poisson-disk jamming ratio of 0.61:

| spacing | column nodes | embedding params (d=128) | assoc edges | state |
|---|---|---|---|---|
| 3.25 mm (current) | 13,500 | 1.7M | 0.9M | 0.3 GiB |
| 1.0 mm | 142,590 | 18.3M | 9.1M | 2.8 GiB |
| 0.68 mm | 308,369 | 39.5M | 19.7M | 6.1 GiB |
| 0.5 mm | 570,360 | **73.0M** | 36.5M | 11.2 GiB |

**parameters are cheap; state is expensive.** 65M params is 0.24 GiB; one
materialization's state at 0.68 mm is 6 GiB before psd and workspace. so standard
data-parallel does not apply — you cannot batch materializations. **dataset-parallel
does, and it is the architecture's own factorization:**

    ∇ log p(θ|D) = ∇ log p(θ) + Σ_d ∇ log p(D_d | θ)

each machine holds a subset of datasets, materializes them, and computes its
gradient on the SHARED θ. the product over d IS the parallelization strategy.

**hardware.** this machine: NVIDIA GB10, 121 GB unified, 20 cores — keep under
half. reachable remote: `promaxgb10-4dfb` (also `gb10-direct`), another GB10,
119 GB, 20 cores, 1.3 T free, currently **bare** — no repo, no data, no torch.
`spark-gb10` and `spark-ec4d` are unreachable. **local torch is CPU-only
(2.14.0+cpu), so the GB10 here is idle.** getting CUDA torch onto aarch64
Blackwell (sm_121) is the gating step for anything at 0.5 mm.

## 7. sensory forcing

continuous naturalistic stimulus→brain pairs are a far higher-bandwidth
constraint than the stationary spectral fits done so far, and unlike a spectrum
they constrain DYNAMICS. held and bound: `libribrain` 45.6 GiB (deep
within-person MEG, audiobook), `meg-masc` 42.5 GiB (MEG, story listening),
`ds003688` 15.3 GiB (iEEG + fMRI during film, with electrode localisation).

the defensible framing: **you are not learning V1; you are using V1 as a driven
boundary condition to learn everything downstream.** that is how the experiments
themselves work — control the input, study the response.

mandatory regardless of pretraining status:
- forcing enters at precision calibrated from r² measured ON BRAIN DATA, never a
  hard clamp and never unit precision
- provenance records FORCED versus EVOLVED per component
- no learned latent becomes a state variable

## 8. corrections already made — do not re-derive the old numbers

| was reported | actually | why it was wrong |
|---|---|---|
| geodesic/euclidean ratio median 2.82, 58% bad edges at 10 mm, worst 21× | **1.41, 20.4%, 11×** | `csr_matrix` SUMS duplicates; a closed mesh's interior edges each belong to two triangles, so assembling from half-edges doubled every path length |
| ~26k column nodes at 3 mm | **15,870** | a maximal poisson-disk set is blue noise, not a lattice; hex packing is an upper bound and the true ratio is 0.61 |
| tractoinferno holds 965 tractograms over 284 subjects | **605 .trk over 30 subjects** | the find counted .nii.gz alongside .trk, and 284 was the card's published cohort, not what is on disk |
| the capillary bed carries most of the resistance | **6–27% of dissipation** | never measured until VesselGraph and Blinder were acquired |

the geodesic figure had to be corrected **twice** — it was overwritten by a later
agent and had already propagated into a new docstring. check for stale copies
before trusting any number in a docstring that is not backed by an evidence file.
