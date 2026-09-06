# state of the effort

a durable record of where ibm-1 actually stands, what has been established by
measurement, what is open, and which decisions have already been made so they are
not relitigated. written because this effort has produced a lot of hard-won
findings that would otherwise be lost between sessions, and because a wrong
number, once written down, gets copied faster than it gets corrected — that has
already happened twice here.

updated 2026-09-05.

## 1. the goal, as currently set

**this project sits at the bridge between neurophysiology and the cognitive
structures that emerge from attractor dynamics.** that is the thesis, and it
decides what counts as progress at every stage.

the two ends are different kinds of claim and the whole design exists to hold
them together:

- **the neurophysiology end is measured and falsifiable.** declared time
  constants, conduction delays, transfer functions, connection probabilities —
  each one a number that can meet data and lose. six have (§3). the machinery
  for that is what most of this repository is.
- **the cognitive end is emergent and cannot be declared.** a schema, a working
  memory, a stable percept is not a parameter you set — it is a property of the
  *attractor landscape* the dynamics produce once the parameters are right.

the bridge is the claim that the second follows from the first: that if the
physiology is measured well enough and the dynamics are carried faithfully
enough, cognitive structure appears as a consequence rather than as an
architectural addition.

**where we actually are on that bridge, measured:** the neurophysiology end is
under construction and yielding real corrections. the cognitive end has not
started, and we know precisely why — **the materialized graph currently has NO
CYCLE.** `build` selects LTI for all 29 processes, and the one edge that would
close the cortical loop is potential->rate, whose only f is a sigmoid. a linear
system has exactly one fixed point. **there is no attractor landscape yet, so
there is nothing for a cognitive structure to be.** that is not a failure; it is
the precise statement of the remaining distance, and §7c gives the ordering that
closes it.

the near-term target is a working implicit brain model at roughly **baseline
industry performance** on standard tasks. that alone is the innovation, because
of what it bridges: one substrate, many materializations, heterogeneous evidence,
and a posterior that says which parts of a prediction rest on measurement and
which on a prior.

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
prior that it does not matter — **now corroborated**: a partial full-resolution
run at 13,647 column nodes, 3.4x the coarse sheet, gives the same iteration
counts and the same residual ratio to two figures (window 0: 19 picard sweeps at
8.4e-09 of initial against coarse's 20 at 3.9e-09; window 1 drive-frozen: 12
picard at 1.3e+00 in both). the raw joint gap scales with amplitude (3.4x,
matching the denser fan-in), which is why the NORMALIZED ratio is the only number
that means anything there — and that one did not land before the run was stopped
under machine contention. corroboration, not proof.

the full-resolution propagation report did land and confirms the machinery at
that r(q): the same 28 theta groups, the same 13 capped at sigma/median > 0.5
(worst 74.99), the same two components flagged first-order-invalid (523x and
32.5x |H| against coarse's 515x and 32.5x). posterior psds are ~10x larger
throughout, which is the denser fan-in rather than a propagation artefact.

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

### 4.2 RESOLVED: the shared substrate and tier fall-through

`ibm/materialize/substrate.py` — one cached object in `mni152`, ten rungs,
reachable from the subject through `talairach.xfm` composed with the published
MNI305->MNI152 affine (92% of the warped white surface lands inside the template
parenchyma, 97% inside an arterial territory).

**built 6 -> 7, missing-data 34 -> 33, refused 0.** nine of forty models now
stand on population structure and RECORD WHICH.

**my hypothesis was wrong and the correction is worth more than the claim.** I
said "most of the 34 were never missing data, they were missing a fall-through".
false as stated: **1** model was only a missing fall-through (glymphatic), **32**
had a fall-throughable gap PLUS a real one, **1** had no fall-throughable gap at
all. counting gap INSTANCES rather than models, 101 were a missing rung:

    geometry['vascular_tree']              28 -> 0   substrate
    geometry['interstitial']               26 -> 0   the subject's own aseg
    geometry['csf_space']                  26 -> 0   the subject's own aseg
    tractometric_streamlines               11 -> 0   group connectome
    anatomy['vascular_territories']         4 -> 0   Liu arterial atlas

two of those were **an unopened file, not a gap**.

and some previously-invisible gaps got WORSE, which is the machinery working: a
model that died on the first alphabetical missing geometry never reached its
later checks. `anchor positions for 'coil'` 7->16, `'probe'` 6->15,
`thalamic_nuclei` 6->12 — **+63 instances now visible**.

**what actually blocks the library is instruments and periphery, not structure.**
retina / cochlea / vestibular_organ / viscera / body / motor_units at 25 models
each, scanner_element 21, display 20, stimulator 19, implanted_array 16. none is
a fall-through; none has a measured prior in the corpus. that is a different
problem from the one I diagnosed.

the fall-through does real work where it applies: `bold_forward` gains 440,396
tractometric edges and 43,213 capillary-exchange edges it had none of;
`glymphatic` gains a 33,983-node vascular tree.

#### where a population substitute is most dangerous
1. **a group connectome under a per-subject causal claim.** it predicts a
   held-out subject's edges at AUC 0.82 and their STRENGTHS at R^2 0.28, leaving
   x7.4 per edge. a seizure map or virtual lesion computed on it looks exactly
   like the real thing and is about nobody. `seizure_propagation` now declares
   `tier_ceilings=(("tractometric", SUBJECT_ONLY),)` and raises
   `TierCeilingExceeded` rather than silently resecting somebody else's fascicle.
2. **arterial watersheds** are exactly where individuals differ most and exactly
   what an infarct model needs — and they are the labels the atlas does not
   supply, so the danger is deriving them by dilating neighbours.
3. **tract lengths use parcel-centroid chords**, which are a LOWER bound on arc
   length, so every conduction delay derived from them is TOO SHORT — the
   direction that makes long-range coupling look faster than it is. flagged in
   the tier record, not fixed.
4. the coarse capillary layer carries the right wall area per unit volume and no
   within-cell transit heterogeneity, so anything whose nonlinearity lives at
   capillary scale (oxygen extraction at low saturation) is outside the regime
   where the coarse-graining commutes.

#### mechanism
`TierRecord` (ladder, rung, source, and a `cost` string that must never be
empty); `Provenance.tiers` / `tier_of` / `rests_on_population`;
`Basis.trustworthy` now FAILS on population structure;
`MaterializationRequest.max_tier` plus per-piece `tier_ceilings`. the substrate is
consulted ONLY from inside an `except (MissingData, MissingInput)`, so subject
data wins by control flow, and `_is_geometry_gap` keeps a missing coil anchor
from being answered with a template brain.

### 4.2-OLD models do not share the substrate (SUPERSEDED)
34 of 40 named models were reported as blocked by "missing data". they are not.
`ibm/materialize/*.py` contains **zero references** to `tract_prior`,
`vascular_prior` or `microcircuit_prior` — the three measured, tiered priors that
exist precisely for this. `build` catches `MissingData` and records a gap instead
of falling through to the tier. a model needing a tractogram should fall to the
group connectome and *say so*, not fail. §7's own claim is that models are
materialized views of ONE implicit model.

### 4.3 JOINT FORGING RAN. the strong claim fails; the scope of the failure matters

`scripts/forge_joint.py`, 4 sources (60 eegmmidb, 60 sleep-edfx nights scored
WAKE, 16 ds000117 with 102 magnetometers + 70 EEG on the same head, 37 ds004873),
one `ParameterSpace`, `Method.JOINT`, split by subject 60/40. weighting by
MEASURED overdispersion phi rather than by size — phi is 3.3 / 31.5 / 7.1 / 1.7,
and a naive w=1 would have handed sleep-edfx 81% of the corpus's curvature from
19 heads purely because its recordings are long.

**three findings, and they have different scopes.**

#### (i) across a band gap the coupling is EXACTLY NOMINAL — structural, stands
ds004873 names 9 `local_excitation`/`thalamocortical_coupling` blocks, because the
process graph says its signal passes through them. it carries **4e-4 nats** about
`tau_membrane_s` against eegmmidb's **2729** — a ratio of 6e-7. fitting
ds004873 ALONE leaves all 9 electro blocks at exactly their prior median. an
`onset_lag_s` control returns 9e-13, confirming the sweep machinery.

this is not a fitting failure. |H_ei| is flat below 1 Hz to within 1e-3, so once
the per-recording gain is profiled the BOLD likelihood is CONSTANT in every fast
time constant. **two processes wired in series in the graph is not the same as one
measurement constraining the other's parameters — what matters is whether the
BANDS OVERLAP, and here they do not overlap at all.** the precision-addition story
is arithmetically correct and empirically vacuous across this gap. this finding is
independent of tying and stands unconditionally.

#### (ii) `Method.JOINT` on this graph is MULTIMODAL — methodological, stands
restarting the same joint fit from a prior draw moves the posterior by up to
**209 marginal sd** — as much as deleting an entire dataset (208 sd). the joint
hessian is not positive definite at the mode. **so "the posterior moved when I
added a source" is NOT usable evidence from this machinery; only swept likelihood
in nats is.** any future joint posterior needs the restart yardstick beside it.

#### (iii) the conflict table falsifies GLOBAL tying, which the architecture does not claim
18 of 19 live comparisons exceed 3 sigma, to 88 sigma: `alpha_resonator.f0_hz`
comes out at **4.33 / 10.4 / 17.4 Hz** from three montages; `loop_gain` at
4.92 vs 1.21; `synaptic_lag_s` differs by an order of magnitude. joint beats each
source's OWN fit on **0 of 4** and is significantly worse on 2. a positive control
(same nights, wake vs N2) gives 4 of 6 above 3 sigma including tau_membrane
8.99 vs 1.58 ms at z=+19, so **the detector has power and the nulls are
interpretable.**

BUT: the experiment collapsed all 21 blocks to **GLOBAL**, and
`ei_loop_lti` and `alpha_resonator` both declare **`tying=per_partition`**
(62 of 108 implementations are per_partition; only 10 are global). so what is
falsified is "one global parameter set describes a bipolar Fpz-Cz derivation, a
posterior scalp average and a magnetometer array simultaneously" — a claim the
architecture explicitly does not make. the agent named this itself as the first of
three causes it could not separate; the other two are a missing instrument/montage
factor in the forward model, and genuine cohort differences.

**the open question, now sharp and testable:** re-run with the DECLARED
per_partition tying and a montage factor. if the conflicts survive that, the
shared-parameter architecture is falsified. if they collapse, the architecture is
intact and the GLOBAL collapse was the error. until that is run, (iii) is not
evidence against the architecture.

#### what does survive
the joint theta beats the untouched literature prior on all four sources
(p <= 0.003), with r2 of log psd rising 0.56->0.80, -1.21->0.80, 0.53->0.65,
0.55->0.92. so the WEAK form holds and the strong form -- that pooling pays for
itself -- does not, on these four sources under a global collapse.

### 4.3-OLD no parameter has ever been moved by evidence in a materialization
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

## 4.8 SENSORY FORCING RAN — delta band only, and it corrected the §4 formula

`ibm/processes/forcing.py` + `scripts/force_early_sensory.py`. `AUDITORY_CHAIN` is
nine links validated at runtime by `check_chain()` against the sealed registry —
which **caught two links naming a process that does not write the component
claimed**. new `Intervention` `naturalistic_stream` (continuous, unrepeated, both
modalities clamped jointly: no baseline and no repeat, so what is constrained is
the MAP rather than a mean evoked response). no new component, no new process —
the front ends are two more `f` for the already-declared `transduction`, so no
learned latent enters the state graph.

front end actually run: `gammatone_cochleagram`, 28 filters at Greenwood-spaced
CFs, ERB bandwidths, compression 0.3, 100 Hz. **no pretrained teacher** —
`tribe` and `tribe-v2` both have `local_root: null`. **no visual arm** —
ds003688's `stimuli/` holds annotation tables and zero media; the film is not in
the deposit, and `ForcingCalibration.unmeasured` refuses the retinal front end a
precision rather than inventing one.

### the numbers (meg-masc, 4 participants AND 1 story held out, 208 ch, 903k samples)

    band        r^2
    0.5-4 Hz   +0.0033
    4-8 Hz     +0.0008
    8-13 Hz    -0.0001
    13-30 Hz   -0.0000
    overall    +0.00103   (top decile 0.0038, best channel 0.0052)

    split                                  unforced   driven   teacher-direct  ratio
    seen participants, held-out story       0.00000   0.00152     0.00197       0.77
    held-out participants, val story        0.00000   0.00105     0.00151       0.70
    held-out participants x held-out story  0.00000   0.00129     0.00180       0.72

**vs teacher-direct: no.** the declared chain reaches 72% of a free TRF's r^2
with 112 free numbers per sensor against 1148 — buying constraint but NOT data
efficiency, and the gap does not close with less adaptation data (0.59 at 5 min,
0.70 at 20, 0.72 at 44). out-of-distribution transfer to libribrain (different
person, scanner, 306-ch array, different book) degrades delta r^2 by **1.4x**,
comfortably inside the declared 3x lognormal inflation prior — **the first
empirical check of that prior in this registry.**

### what it constrains, and in which band
**delta, 0.5-4 Hz, and essentially nothing above.** the joint-forging lesson
applied in mirror, and made structural: `ForcingCalibration` is band-resolved and
`teacher()` takes a band, because a chain existing does not mean it carries
information. **a claim about what forcing constrains that does not name a band is
not a claim.**

within that band it constrains the phase-resolved transfer rather than a marginal
second moment. ten chain time constants moved off their literature priors, ALL IN
THE SAME DIRECTION and **all hitting the top of their grid** — afferent delay
15->45 ms, thalamic relay 12->40, NMDA 100->200, TRN GABA-B 150->300. so the
constraint is ONE-SIDED: the data ask for a slower chain than the priors and this
experiment cannot say how much slower. `loop_gain` and `tau_cortex_s` did not
move at all, consistent with the loop's resonance living at 10 Hz where the
measured r^2 is zero.

### THE FORMULA IN §4 WAS WRONG AND IS NOW FIXED
`dJ = 1/((1-r^2)Var[x])` does not vanish as r^2 -> 0: it tends to
`1/Var[x]`, one prior-equivalent, so **a teacher explaining NOTHING takes half
the posterior and halves the variable's variance.** corrected in ARCHITECTURE.md
and EVIDENCE.md to

    dJ = r^2 / ((1 - r^2) Var[x])

which gives the teacher a posterior weight of exactly r^2. the error is invisible
where teachers are good and dominant where they are weak — exactly the regime a
first forcing experiment lives in. at the measured r^2 = 0.0033 the old form
credits a shrinkage of 0.25 where the correct one gives 0.003.

### three measurement errors, each of which produced a confident wrong number first
1. a case mismatch between events (`The_Black_Willow_3.wav`) and audio files
   (lowercase) silently dropped the largest story — **half the corpus**.
   `load_segments` now refuses rather than skipping.
2. a lead field pooled across people, or across one participant's two sessions,
   **averages to zero** — a channel index is not an anatomical label, and the
   montage moves between visits.
3. six saturated sensors turned a +0.0009 median into a **-0.073 mean**.

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

## 7b. THE CLOSED LOOP — next-frame prediction through brain state

**this was asked for early and I recorded the wrong thing.** §7 above records
stimulus-driven forcing fitted against MEASURED brain data, which is bounded by
the ~90 GiB of paired recordings we hold. the actual proposal is a closed loop
that needs no paired recordings for its training signal:

    frame_t --[encoder]--> eeg_t --[IBM dynamics]--> eeg_{t+dt} --[decoder]--> frame_{t+dt}

trained on RAW NATURALISTIC VIDEO with next-frame prediction as the objective.
likewise audio and human speech; likewise any other modality with a stimulus
stream. paired brain data is needed only to build and validate the encoder and
decoder — not to generate the optimization signal. **that is the difference
between a 90 GiB corpus and an unbounded one, and it is what justifies learning
10^8 parameters.**

what makes it more than an autoencoder: the latent is BRAIN STATE, and the
transition is not free. the encoder and decoder may be learned, but the step from
eeg_t to eeg_{t+dt} is the declared process graph with its declared transfer
functions and its fitted theta. so next-frame prediction supervises THETA through
the transition — a dense, abundant signal on the one part of the model that is
otherwise stuck at prior medians (see 4.3: zero theta have ever been moved by
evidence in a materialization).

### the degenerate solution is the whole design problem
if encoder and decoder are both free, they will learn to route information AROUND
the brain state and the IBM dynamics collapse to an identity map. next-frame
prediction would be perfectly satisfied with the brain contributing nothing, and
the result would look like it worked. two guards, and neither is optional:

1. **the transition stays in its declared form.** LTI transfers with priors, not
   a free network. this is the entire reason to use the IBM as the bottleneck
   rather than any convenient latent — a free transition makes the brain
   decorative.
2. **a joint objective.** next-frame prediction on unbounded video PLUS the
   measured-EEG likelihood on the 90 GiB we hold, fitted together. the second
   anchors the first: theta must simultaneously predict the next frame AND
   reproduce real recorded brain state. this is exactly the multi-source product
   `ibm/forge/fit.py` already implements, with video as one more Task.

### gating dependencies, in order
- **TRIBE v2 weights are NOT held.** `data/sources/tribe/` and `tribe-v2/` both
  have `local_root: null`. this is the first blocker and nothing starts without
  it or a substitute encoder.
- **the decoder is a different and harder model than the encoder.** TRIBE maps
  stimulus -> brain. the loop needs brain -> stimulus, which is decoding, which
  is `eeg_to_image` in the library and is not free. it must be trained, and its
  own r^2 calibrates how much the loop's output can be believed.
- **bandwidth.** the measured forcing r^2 was 0.0033 in delta and ~0 above
  8 Hz (§4.8). a loop whose transition only carries delta cannot predict a video
  frame 33 ms ahead. either the encoder must target a band the dynamics actually
  carry, or dt must be chosen to match — and that is a measurement, not a choice.

### why the bottleneck is the point
64 channels of EEG over a declared band is a far narrower channel than a video
frame. that narrowness is not a limitation to engineer around — it is what forces
the model to learn what the brain would have to represent, which is the whole
claim. a wide bottleneck would let the loop succeed without ever using the
dynamics.

## 7c. WHY "BOTH" IS FREE, AND THE CURRICULUM / RL STAGE BEYOND IT

**there is no standard materialization, and most of the model is used in all of
them.** 15 processes appear in 5+ of the 40 named models carrying 271 parameters;
`local_excitation` alone is in 20 of 40. so a video-loop materialization, an
audio-loop one, a text one and a measured-EEG one are not competing training
regimes — they are **more terms in the same product**:

    p(theta | D)  ∝  p(theta) · Π_d p(D_d | theta)

`ibm/forge/fit.py` already takes a sequence of `Task`s. a closed loop is one more
Task with a next-frame likelihood; a measured recording is one more Task with a
spectral likelihood. they share theta by construction, not by arrangement. that
is the whole reason the implicit/explicit split was worth building, and it means
the answer to "measured forcing or closed loop" is both, at no architectural
cost.

### the stage beyond: curriculum, schema, reinforcement
the intended arc after the modality loops is: ordered curriculum -> a stable
cognitive schema -> reinforcement learning on cognitive tasks.

**the architecture already declares the path RL needs**, which is worth stating
because it was not designed for it:

- `neuromodulation` has `writes="parameters"` and targets `local_excitation`,
  `local_inhibition`, `laminar_propagation`, `lateral_cortical_propagation`,
  `tract_propagation` and more. a reward signal reaching theta THROUGH a
  modulator is the biological story and it is already the declared mechanism.
- `plasticity` also writes parameters, targeting `local_excitation` and
  `tract_propagation`.
- actions have a route out: `efferent_propagation` -> `effector.drive` and
  `neural.efferent.activity`; `effector_activation` -> `effector.{activation,
  force, fatigue}`.

so reward -> neuromodulator -> plasticity -> theta is a declared path, not
something to bolt on.

### the sequencing constraint that decides when this can start
**a stable cognitive schema requires nonlinear dynamics, and ours are not yet
stable.** three measured facts, all in this document:

1. the LTI graph HAS NO CYCLE (§ run_eeg_forward). the one edge that would close
   the cortical loop is potential->rate, whose only f is a sigmoid, and `build`
   selects LTI for all 29 processes. a linear system has one fixed point — it
   cannot hold multiple attractors, so it cannot have a schema, a working memory,
   or a persistent state to reinforce.
2. 27 `Form.RATE` implementations exist, so the nonlinearity is declared and
   available. but the one nonlinear run attempted DIVERGED (7.5e15 mV) at prior
   parameters, and `ei_loop_lti`'s return path has min|1+L| = 0.327 at 38.6 Hz —
   marginal, with the minimum landing on PING gamma from time constants alone.
3. theta is still at prior medians everywhere (§4.3).

so the order is forced: **fit theta through the modality loops FIRST, then select
the RATE implementations, then check the nonlinear dynamics are bounded and have
the attractor structure a schema requires, and only then curriculum and RL.**
attempting RL on a linear graph would train a reward model with no state to
condition on; attempting it on an unfitted nonlinear one would train against
divergence.

what makes the ordering testable rather than a guess: boundedness, stability
margin and attractor count are all measurable on a materialized model, and
`run_eeg_forward` already measures the first two.

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
