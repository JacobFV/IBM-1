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

### 4.1 uncertainty propagation — LANDED, and it invalidated a headline result

RESOLVED as of 2026-09-05. `ibm/runtime/propagate.py` pushes the second moment
through the solve's OWN operator rather than each coupling's H: at the picard
fixed point z_O = (i omega - A_O)^-1 sum_c M_c z_I, so |inv_O M_c|^2 goes on the
psd and its square on the relation. written spectral components whose psd is
still bit-identical to their prior: **0 of 7**, was 7 of 7.

three things worth keeping:

- **couplings sharing an input compose COHERENTLY.** the association and lateral
  pathways are split into 12 distance bins each, and an incoherent sum is wrong
  by the cross terms — verified, two couplings at +2G and -1.5G give 0.5 coherent
  against 12.5 incoherent, a factor of 25.
- **distinct input components are composed as INDEPENDENT**, which is a real
  approximation in the OVER-SHARPENING direction (ampa and nmda share an
  ancestor). counted and named in every report, not fudged.
- pressure pushed width out along the chain by 10^3-10^5 and evidence pulled it
  back (fusing this subject's EEG multiplies device.contact_potential's psd by
  0.0093 in delta to 0.0409 in gamma). the two compose by different rules, which
  is the architecture's claim, now exercised.
- **the push-forward itself is exact to machine precision.** compared against
  |G|^2 assembled ANALYTICALLY from the couplings (inv = 1/(i omega - A) from
  `_self_operator`, each coupling's own H summed by hand) on a site-local link
  with 885,190 cells: median |psd/(|G|^2 psd_in) - 1| = **2.2e-16**, max 1.0e-15.
- coherent composition earns its keep on the real model too: |sum H|^2 / sum|H|^2
  over 8-13 Hz is **0.9179** on that pair, and far further from 1 on
  `neural.exc.ampa`, whose 25 channels share one input.

#### 4.1d THE MEAN AND THE WIDTH ARE PROPAGATED BY TWO DIFFERENT MAPS

this is the largest remaining item in the runtime, and it was found by tightening
a check until it failed. an earlier claim that "psd ratio and mean-power ratio
agree to <3% on every topology-mediated link" was BAND-AVERAGED and over-read;
per (site, coefficient) the two do not agree at all — **median deviation 98%,
max 1280x**.

that is not a bug in the push-forward, which is exact (above). it is that only
ONE of the two is a push-forward:

- **the mean is a boundary-value solve.** `_match_overlap` pulls the window's
  head towards the previous window's tail every sweep, and the residual settles
  at the floor `StepReport` reports as `limited_by="continuity"`. so the solved
  mean is the least-squares compromise between G times its input and the
  trajectory it has to continue.
- **the width sees none of that.** `propagate_linear` applies |G|^2 and stops.
  `Carry` carries a mean tail and nothing else, so there is no width to inherit
  and no boundary condition to compromise against.

so the mean departs from the analytic |G|^2 by a median of 98% while the psd
matches it to 2e-16 — the gap is exactly the continuity constraint, which acts on
one and not the other. **the fix is a `Carry` that carries a covariance**, which
is a decision about what a window inherits rather than a code change, and it
should be taken deliberately.

the width growth is NOT loss — at prior medians the chain has a power gain of
that size, because the association topology is horvitz-thompson reweighted to
stand for a dense graph sampled under a percent and the `gain` that would hold it
down is `weak(1.0, 5.0)` and **has never been fitted**. what IS lossy is the
frequency SHAPE: every component loses one to two orders at gamma relative to
delta. `neural.exc.potential` is the exception, with a x1116 peak in alpha — the
thalamocortical resonator showing up in the WIDTH for the first time.

**parameter uncertainty is in, with an explicit validity bracket.** dH/dtheta by
central difference, coherent across parameters shared by many couplings, exact to
first order (1.2e-13). but 13 of 28 groups had sigma/median above 0.5 (worst 75x
— a `weak()` lognormal has a natural-units sd 200x its median) and for two
components sigma|dH/dtheta| EXCEEDS |H| in band, because d/dv exp(-i omega d/v)
grows without bound. what physically happens there is phase decoherence and power
redistribution; a derivative reports it as growth. so `parameters=False` is a
lower bound, `parameters=True` an extrapolation, and the right treatment for a
`weak()` prior is sampling over p(theta).

#### 4.1a THE CONSEQUENCE: fit_neural_spectra.py REPORTED SCALP QUANTITIES AS CORTICAL

the model now supplies what all three evaluations assumed was flat by never
mentioning it — the power transfer from `neural.exc.activity` to
`device.contact_potential`. it is 10372 (delta) falling to 231 (gamma), a log-log
slope of **+1.11 over 1-45 Hz**.

`fit_neural_spectra.py` fits `neural_population` — a prior over a CORTICAL
component — to a SCALP psd. deconvolving by that transfer moves the fitted
aperiodic exponent from **1.480 to 0.435** on an exponent-only refit with the
other five held, or to 0.361 on the full six-parameter refit where three
parameters hit a boundary — the exponent-only number is the trustworthy one. the prior it was moved off is 2.000,
so **the chain's bias is 201% of the entire distance the data moved that
parameter.** alpha_gain moves 5.74 -> 2.80.

the SIGN of its held-out delta log-likelihood is safe — both curves are filtered
by the same transfer — so "the posterior beats the prior on unseen subjects,
p 0.0014" stands. **every parameter VALUE it reports is a scalp-level quantity
presented as a cortical one and must be redone.**

`eval_sleep_state.py` is NOT changed, and that is the worse answer. its claim is
directional and the transfer does not depend on sleep stage: deconvolving source
exponents 1.20 / 1.60 / 2.00 gives 0.133 / 0.563 / 1.015 — **ordering strictly
preserved**, and stage differences inflated by only x1.075-1.130, far inside the
sems that test already reports. **a linear time-invariant head is transparent to
a comparison between two spectra recorded through it, so the dynamics are
irrelevant to that test.** it would stop being transparent only if something
between cortex and scalp were stage-dependent or nonlinear, and nothing in this
model is.

`fit_hemodynamic_chain.py` and `fit_meg_instrument.py` cannot be affected —
neither ever instantiates a State. note the hemodynamic components are declared
SCALAR and `_advance_scalar` already carries the exact push-forward for that
form; it has never run, because no assembled coupling in any script writes a
scalar block.

#### 4.1e the continuity floor, measured (coarse; full-resolution pending)

from `run_eeg_forward.py` section 7 (drive advanced, the meaningful case):

    window 0   converged, residual 3.9e-09 of initial, joint gap 0
    window 1   limited_by=continuity, residual 9.23e-02, joint gap 1.818e+02
    window 2   limited_by=continuity, residual 9.26e-02, joint gap 1.754e+02
    joint gap / state rms  0.0702, 0.0678

**ANSWERED, and it is a clean negative: the floor is NOT set by the hop against
the graph's memory.** r(q) held fixed at coarse (4,318 sites on the merged neural
block, longest memory 600 ms, 2048 ms window), hop varied — the only other thing
the floor could depend on:

    overlap  hop_ms  hop/memory  gap/rms  resid/init  limited_by
      0.586     848        1.41   0.0690      0.0925  continuity
      0.700     614        1.02   0.0615      0.0939  continuity
      0.800     410        0.68   0.0582      0.0987  continuity

the 0.586 row reproduces the `run_eeg_forward` baseline exactly, so the harness
agrees with itself. **halving the hop — from 1.4x the graph's longest memory to
0.68x it — buys 16% on the joint gap and makes the residual floor MONOTONICALLY
WORSE.**

**the floor is structural, not a tradeoff.** a driven cyclic window has a unique
solution, and the carried tail is a second condition it cannot satisfy. the size
of the mismatch is set by how much the dynamics DISAGREE with the tail, not by
how much of the window is pinned — so widening the overlap pins more of a window
whose own dynamics already determined it, adding constraint without adding
agreement. that is why the gap falls while the residual rises. it is the
alternating projection onto two non-intersecting sets that
`WindowPlan.match_weight`'s docstring already describes, now with a number on it.

so a covariance-carrying `Carry` (4.1d) is a **once-and-for-all** fix, not an
r(q)-dependent one: the hop is the only temporal knob and the floor does not
respond to it, which leaves the disagreement between the dynamics and the imposed
tail — a property of the transfer functions and `match_weight`, neither of which
is r(q). the r(q) axis is formally open (only the hop was varied) with a strong
prior that it does not matter.

**side finding: `WindowPlan.for_memory` gives advice that does not do what a
reader will assume.** it exists to widen the overlap until a plan is causally
legal, and it is correct about that — it is what makes the run legal at all. but
"widen the overlap" reads as also buying a better stitch, and it does not: 0.80
overlap costs **2.5x the time per window** (287 s vs 114 s for the same three
windows) for 16% better gap and 7% worse residual. the docstring should say so.

#### 4.1f do not recover central moments from raw sums at scale
`reproject_nonlinear` was briefly changed to stream members by accumulating RAW
moments and recovering central ones at the end. on a diverged ensemble — the
wilson-cowan run drives members to 1e30 — `E[x^4] - 4 mu E[x^3] + ...` cancels
away every digit: it reported skew 8e83 and excess kurtosis -4e120 where the
two-pass code gave +0.00 and -0.68. replaced with Pebay online recurrences
(running mean, M2, M3, M4, every term a deviation from the running mean) plus a
complex Welford for the width.

**audited the rest of the repo for the same pattern: none found.**
`ensemble._diagnose` standardizes first (`d = x - mu`, `z = d/sd`) and is
two-pass; `SpectralGaussian.moment_match` centres before squaring;
`ScalarGaussian.moment_match` uses `np.var(ddof=1)`. the bug existed only in the
streaming path and only while it was raw.

#### 4.1b what still does not propagate
cross-component covariance (independent composition across distinct inputs — a
component reading two inputs with a common ancestor gets a width that is too
small; the declared form cannot carry it and the fix is a block covariance, not
an approximation); the boundary condition's own width (Carry carries the previous
window's MEAN tail only, so width is re-derived each window — wrong for any
component whose memory exceeds the hop); the low-rank `factor` term on write-back;
p(theta) beyond first order; and the scalar form's dynamics, still never
exercised by anything.

### 4.1-OLD uncertainty is not propagated (SUPERSEDED, kept for the record)
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

### 4.1c RESOLVED: the eeg_forward build regression
cleared 2026-09-05, verified independently. cause: the new tier fall-through in
`_build_edges` computed `parcels("cortical_areas","tissue")` EAGERLY and let its
exceptions escape, so a caller supplying no subject `anatomy=` fell through to the
substrate's template parcellation and demanded a `subject_t1 -> mni152` transform
nobody had bound. `eeg_forward` needs no connectome at all — it was only passing
through that branch on its way to reporting it has no tractogram.

the fix is the general lesson, not the specific guard: **a fall-through that
cannot get its inputs must degrade to the ORIGINAL gap** — the one naming the
subject file a caller could supply — rather than replacing it with a complaint
about a template registration nobody asked for. the parcel lookup now runs in the
subject frame and declines instead of raising.

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
