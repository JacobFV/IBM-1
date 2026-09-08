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
| 12 | the cortex beats the dynamics-free encoder on vision | it wins on the training-split holdout (30.12% vs 26.81%) and **loses on the corpus's designated test set** (63.50% vs 66.50%), where the two are indistinguishable anyway -- 0.89 sd, six images out of 200. the quoted 21.5% control was also a single pool | building `images_test.npy`, which had never existed, and selecting the control the same way the model was selected |
| 10 | the joint MEG term is reaching skill +0.45 | that is a **training** loss. held-out is -0.003, and the ceiling is +0.036 | the regression control |
| 11 | ~~the LibriBrain arrays carry no envelope tracking~~ **and, one entry later, that speech→MEG is hard at all** | the builder assumed `timemeg - timechapter` was CONSTANT; the clocks differ by **4,300-5,300 ppm**, which is ±3.4 s of drift across a chapter and smears a 1-8 Hz effect across 3-27 cycles. resampling onto the fitted line takes the corpus from p=0.171/0.463/0.902 to **p=0.024 in all three windows**, peak at 140 ms. the effect was averaged away by the builder, and four negative results are suspended with it | fitting a LINE where a constant was assumed, after the gate's own sensitivity floor was measured |

**the shape they share:** a quantity computed correctly and then compared against
the wrong thing — the wrong population, the wrong units, the wrong split, the
wrong baseline, or no baseline. Not one was a modelling error. Every one was
caught by a measurement that could have been run first, and several by one I had
already written.

---

## 2026-09-08 — the auditory cortex term matches its own ceiling, and no more

the first auditory term with the cortex in the path finished 8,000 steps on the
repaired corpus. best held-out **6.75% ± 2.18** (13.5x chance) at step 6,600,
against the dynamics-free control's **7.12% ± 1.71** (14.2x).

the gap is 0.37 points against a combined sd of 2.07 — **0.18 sd**. the two are
indistinguishable. the cortex neither beats its control nor loses to it.

that is the same verdict the visual branch received yesterday on the designated
test set, arrived at independently on a different modality: the substrate matches
a purpose-built encoder rather than beating it. two modalities, two controls, the
same answer. the pattern is now the finding, and any claim that the substrate
*outperforms* a task-specific baseline should be treated as withdrawn until
something measures otherwise.

what is NOT withdrawn is the ablation result, which asks a different question and
still says the dynamics do the work *inside* a model that has them.

the term is worth keeping for the reason the architecture exists: it is one
parameter set shared with the visual term, not a second encoder. matching a
specialist while sharing a substrate is the actual bet. but it is a weaker claim
than the one this branch was set up to make, and the run does not yet test it --
the shared-substrate version is the next thing to build.

one caveat on the number: 200 ms windows reached 4.62% and 1 s reached 7.12% for
the control, so window length matters more than anything else measured here. the
cortex term used 1 s. a longer window has not been tried and is the cheapest
remaining lever.

## 2026-09-08 — the dynamics contribute a constant amount; the bypass is learned

evaluating four checkpoints on the designated test set — two scales and three
training stages — separates two things that had been reported as one.

| model | step | full | frozen | bypass | dynamics pts | dyn share | assoc pts |
|---|---|---|---|---|---|---|---|
| 30k | 1800 | 40.5 | 43.0 | 6.5 | 34.0 | 84% | **-2.5** |
| 30k | 3600 | 47.0 | 32.0 | 10.5 | 36.5 | 78% | 15.0 |
| 150k | 2500 | 45.5 | 20.5 | 8.5 | 37.0 | 81% | 25.0 |
| 30k | 5800 | 63.5 | 28.0 | 26.0 | 37.5 | 59% | 35.5 |

**the dynamics contribute 34-37.5 points in every row.** across a 3x range of
training and a 5x range of substrate size, what the cortex adds over its own
bypass barely moves. that is a far more stable quantity than anything else here.

**what moves is the bypass, and it is LEARNED**: 6.5 -> 10.5 -> 26.0 as the 30k
model trains, a factor of four. so the "dynamics share" falling from 84% to 59%,
which was recorded earlier as a point against the substrate, is not the dynamics
weakening. it is the encoder progressively learning a shortcut around them while
their own contribution stays flat. those are different mechanisms and only the
second is happening.

**at step 1800 the learned association is worth nothing measurable** — frozen
(43.0) is 2.5 points ABOVE full (40.5), which at n = 200 is 0.72 sd and therefore
indistinguishable, but it is certainly not the "largest single contributor" the
README claimed on the strength of the 5800 checkpoint. that contribution grows
with training: -2.5, then 15.0, then 35.5. the compounding story holds; the claim
that it is a property of the architecture rather than of a training stage does
not.

on scale: the 150k model at step 2500 (45.5%) sits between the 30k model at 1800
and 3600, so **5x the sites buys nothing on accuracy at matched steps** while
costing 11x per step. one thing does favour it: its learned association carries
25.0 points against 15.0 for 30k at a comparable stage, and its bypass is weaker
(8.5 against 10.5). more substrate does put more of the work into the substrate.
that is the scaling argument, and it is about where the computation sits rather
than about the score.

the run is left going, because it is the only model in the 10-100M band and it is
still climbing (20.44% on the trainsplit holdout at 2500, best so far). but it is
not yet earning its compute on accuracy, and that should be said plainly rather
than discovered later.

## 2026-09-07 (night, later) — on the designated test set the cortex does not beat the control

**the README's headline claim does not survive the corpus's own test set.**

`images_test.npy` had never been built. the evoked test responses were there --
200 concepts at **80 repetitions** each against training's 4, a factor of sqrt(20)
in target SNR -- and their images were not, so the designated set had never been
usable. it is built now, in the order `image_metadata.npy` declares.

it also removes the last sampling artefact: 200 images means a pool of 200 IS the
whole set, chance is 1/200 by construction, and the measurement is deterministic.
ledger row 9 cannot recur there.

the trained checkpoint scores **63.50%** on it — 127x chance, top-5 89.0%, median
rank 1. that looked like a large improvement on the 30.12% we report, and it is
not: it is the same model on a cleaner target. the two numbers are not
comparable and neither should be quoted as progress over the other.

then the control, selected the same way the cortex checkpoint was — best on the
training-split holdout, then scored on the designated set:

| | trainsplit (selection) | designated test |
|---|---|---|
| dynamics-free control | 26.81% | **66.50%** (133x) |
| cortex model | 30.12% | 63.50% (127x) |

**the control wins on the designated set.** it loses on the split we had been
reporting and wins on the one the corpus was built to be scored on.

the honest statement is not "the control beats the cortex" either: at p ~ 0.65 and
n = 200 the standard deviation of a proportion is 3.37 points, so a 3-point gap is
**0.89 sd — six images**. the two are *indistinguishable* on this set. what dies is
the directional claim, which the README states as a headline: "the cortex beats
the encoder it was meant to merely not obstruct."

three things follow, and none of them is that the substrate is worthless:

- the **ablation** result is untouched and was always the stronger evidence.
  bypassing the dynamics inside the trained model costs it 37.5 points on this set
  (63.50 -> 26.00), and a randomly initialised association is worth nothing beyond
  no association at all. the cortex is load-bearing *within* the model that has
  one. that is a different claim from beating a separately-trained encoder, and
  only the second one just failed.
- the 21.5% control figure we have been quoting was itself a **single pool**.
  measured over 8, the control reaches 26.81% on the trainsplit — so the visual
  margin was always narrower than reported.
- the control **peaks at step 500 and decays to 35% by 3000** on the designated
  set. quoting its peak selects a checkpoint on the evaluation set; quoting its
  final value flatters the model it is a control for. matched selection is the
  only defensible comparison and it is what the table above uses.

## 2026-09-07 (night) — the auditory ceiling, measured honestly

the dynamics-free speech->MEG control on the v3 corpus, averaged over 8 held-out
pools of 200 rather than one:

| window | top-1 | vs chance |
|---|---|---|
| 200 ms | 4.62% ± 0.99 | **9.2x** |
| 1 s | 7.12% ± 1.71 | **14.2x** |

stable across the last six evaluations in both cases (200 ms spans 3.81-5.31, 1 s
spans 6.37-7.37), so this is a level and not a lucky draw. on v1 the same control
sat at 0.0-1.0% across 13 evaluations — 0-2x chance — while its training loss fell
to 2.83. that is the difference the alignment fix made, and it is not subtle.

the single-pool version of this run printed 8x at one step and 17x at another,
which is ledger row 9 recurring for the third time in one day: recorded in the
ledger, fixed in the visual trainer, and left standing in this script until the
numbers it produced were about to be quoted. the control now averages pools and
prints the sd.

**this is a CEILING, not a result.** it is what a pair of convnets extract with no
dynamics in the path, and it is the number any cortex-in-the-path auditory term
has to beat. for calibration, the visual side's dynamics-free control reached
21.5% (43x) and the cortex model then reached 30.1% (60x). the auditory ceiling
is roughly a third of the visual one — real, and much weaker.

the longer window is genuinely better (7.12 against 4.62, sds ~1-1.7), which is
what the coupling geometry predicted: the tracking sits at a 140 ms lag, so a
200 ms window barely contains one response.

## 2026-09-07 (later) — the residual was signal, and it was measurable

the linear fit left 43-160 ms of residual and that was treated as noise to
tolerate. it is not. across the 12 runs, **residual predicts coupling at
r = -0.697** (R2 = 0.49 with run length in the model), while run length on its own
explains nothing (+0.004) and the clock rate itself nothing (-0.089). the
best-aligned run couples at r = 0.100, the worst at 0.029. the slope is **-0.54
of coupling per second of residual** — so the leftover was signal being discarded,
and its size said how much.

that is a prediction, so it was used as one. a piecewise-linear map through ~30 s
knots takes the residual from 43-160 ms to **11-21 ms**. at -0.54/s that predicts
about +0.044 of coupling. measured, on five disjoint windows:

| window | v2 (linear) | v3 (piecewise) |
|---|---|---|
| 0-20 min | r=0.0703 PASS | r=0.1094 PASS |
| 25-45 min | r=0.0853 PASS | r=0.1265 PASS |
| 50-70 min | r=0.0734 PASS | r=0.1264 PASS |
| 75-95 min | r=0.0394 PASS | r=0.1379 PASS |
| 100-120 min | r=0.0340 fail | r=0.0412 PASS |

**5 of 5**, mean coupling 0.0605 -> 0.1083, a factor of 1.79. predicted +0.044,
observed +0.048. the window v2 could not pass now passes.

the three builds in order: v1 (median offset) 0 of 5 windows, mean r ~ 0.027;
v2 (one line) 4 of 5, 0.0605; v3 (piecewise) 5 of 5, **0.1083** — which is inside
the published range for cortical speech tracking rather than at the edge of
detection.

the per-run table (`logs/perrun_gate.log`) is what made this findable: 10 of 12
runs passed individually, and the two that did not were the shortest run (12.7
min, underpowered at r=0.051) and the run with the worst linear residual (0.160 s)
— which is the relationship above, visible before it was fitted. both previously
ambiguous chapters pass, 07 at r=0.100, confirming that choosing them by coupling
was right.

**this does not yet say the term trains.** the contrastive control on v2 reached
2.0% against 0.5% chance. v3 is a better corpus, not a result. controls at 200 ms
and 1 s windows are running.

## 2026-09-07 (late) — the auditory corpus was fine; the clock was not

**the previous entry is overturned, and the branch it wrote off is alive.**

it concluded the LibriBrain arrays carry no speech tracking. that was measured
correctly and read too far. two corrections, in order.

*first, the gate has a floor, and it had never been checked against a case whose
answer was known* — CLAUDE.md's own rule, skipped when the gate was written.
injecting a known coupling into one channel of real MEG: it returns PASS at
r ~ 0.058, finding the right channel and the right lag, and FAIL at r ~ 0.023. so
a FAIL means "nothing above ~0.05", not "nothing". that floor is now measured, in
the docstring, and in the failure message.

*second, and this is the finding:* `build_paired_meg.py` asserted that
`timemeg - timechapter` "is constant within a run" and sliced the MEG at its
median. it is not constant. its standard deviation is **1.0-2.1 seconds** in
every run, because the clocks run at different RATES — fitting
`tmeg = a*tchap + b` gives **a - 1 of 4,300-5,300 ppm, consistently across all 12
runs**, and drops the residual to 0.04-0.16 s, a factor of 15-20.

0.48% over a 1,400 s chapter is ~6.7 s of accumulated drift, so a median offset
is correct in the middle of a run and off by ±3.4 s at its ends. speech tracking
is a 1-8 Hz effect; ±3.4 s smears it across 3-27 cycles. **the effect was never
absent — it was averaged away by the builder.**

the cochleagram is now RESAMPLED onto the MEG clock through the fitted line. the
same gate, same nulls, five disjoint windows:

| window | v1 (median offset) | v2 (drift corrected) |
|---|---|---|
| 0-20 min | r=0.0312 p=0.171 | r=0.0703 **p=0.024** |
| 25-45 min | r=0.0274 p=0.463 | r=0.0853 **p=0.024** |
| 50-70 min | r=0.0234 p=0.902 | r=0.0734 **p=0.024** |
| 75-95 min | r=0.0255 p=0.707 | r=0.0394 **p=0.024** |
| 100-120 min | r=0.0241 p=0.902 | r=0.0340 p=0.098 |

**4 of 5**, and the peak sits at **140 ms** — the auditory M100/M150 range. v1
fails all five; v2 passes four with 0 of 40 shifts reaching the observed value.

the honest qualifier is the trend down that column: 0.070, 0.085, 0.073, 0.039,
0.034. the effect halves across the file. the concatenation is ordered by run, so
a window blends runs and that decay is more likely to be about WHICH runs than
about time — the per-run breakdown is in `logs/perrun_gate.log`. the correction
is real and replicated; it is **not** uniform, and the run-level variation is not
yet explained.

a second false assertion in the same docstring, also now measured: "14 chapters
with distinct lengths, so the match is unambiguous". two chapters differ by 1.5 s
and two runs sit between them. those are chosen by which candidate actually
couples — well-posed only once the drift is out — and both resolve to the chapter
matching their session index.

**what this suspends is larger than what it settles.** the +0.036 regression
ceiling, `paired_v5`'s 10,000 steps below the zero baseline, the joint MEG head's
-0.003 held-out skill, and this morning's contrastive control at chance were all
measured against arrays whose audio and neural streams drift up to 3.4 s apart.
none of them is evidence about the modality, the objective, or the model.

what is NOT yet claimed: that the term now trains. the contrastive control rerun
on v2 sits at 1-2.5% against 0.5% chance — better than v1's flat chance, and
notably it no longer memorises (loss holds at 4.43 against ln(128) = 4.85, where
v1 fell to 2.83 while held-out stayed at chance). the corpus demonstrably carries
the signal now; extracting it at 200 ms windows is a separate open question.

a note on instruments: a forward TRF on the same v2 arrays reports p = 0.286 and
looks like a contradiction. it is not — it is the weaker test here. it scores only
the 20% held-out block, and bandpassing to 1-8 Hz leaves ~3,600 effective samples
there against ~19,000 for the gate's full-window correlation, which is why its
null is 2.4x wider. **the more elaborate method is not automatically the more
powerful one.**

## 2026-09-07 (evening) — the auditory corpus does not contain the effect

the auditory branch has produced three negative results: waveform regression
peaking at skill +0.036, `paired_v5` below the zero baseline for 10,000 steps,
and the joint MEG head at held-out -0.003 after 29,000. the obvious reading was
that speech -> MEG is simply hard.

the contrastive control was the remaining hope, because changing the OBJECTIVE
is exactly what rescued the visual branch (+0.011 regression -> 53x chance
retrieval). it was never run. it is now, and it sits at **chance**: training loss
falls 5.04 -> 2.83 against ln(128) = 4.85, while held-out top-1 never leaves
0.0-1.0% against 0.5% chance across 13 evaluations. no peak, at any point.

**so the objective was not the problem, and the next question was whether the
arrays are.** ledger row 8 is a negative result produced by a broken pairing, so
before recording a fourth, the corpus was asked whether it carries the one effect
it must: 1-8 Hz speech-envelope tracking in MEG, among the most reproduced
findings in the field.

it does not. max |r| over 306 channels x 10 lags is 0.0312, against a
circular-shift null whose mean is 0.0278 — **7 of 50 shifts reach the observed
value, p = 0.157**. broadband and GFP versions are worse (p = 0.51).

that does not merely add a negative result, it **suspends three**. the +0.036
ceiling, the 10,000 steps below baseline and the -0.003 held-out skill were all
measured against arrays that do not demonstrably contain the signal. none of them
is evidence about the modality, and `s7.joint`'s stated route — rebuild the
auditory term contrastively — is withdrawn as written, four hours after it was
written here.

getting to that took two wrong readings of the same number, both recorded because
the reasoning matters more than the answer:

- the first check declared "no coupling" against a threshold of |r| < 0.02
  **chosen out of the air**. right conclusion, no baseline — the ledger's own
  shape.
- the correction called r = 0.0106 a 6-sigma effect using the i.i.d. standard
  error 1/sqrt(N). both signals are heavily autocorrelated; the measured
  circular-shift null is sd 0.0145, **eight times wider** than that formula
  claims. the effective N is nowhere near the nominal one.

the arbiter in both cases was a null built by the same procedure as the
statistic. that is now `scripts/check_pairing.py`, and it is a **gate**: run it
on a paired corpus before building a term, because it costs under a minute and
row 8 cost three hours. it fixes the band (the effect is defined in 1-8 Hz;
broadband is the right quantity in the wrong units), the null (circular shifts,
not a formula), and the statistic (a grid maximum needs a grid-maximum null).

what is NOT claimed: that LibriBrain is unusable. the failure could be the
pairing order, the resampling, or the derivation, and the raw corpus is untouched
by this measurement. what is claimed is that **the derived arrays under
`data/derived/libribrain-paired` should not carry another training term until
they pass the gate.**

## 2026-09-07 (afternoon) — what the retrieval is actually made of

the 30k-site visual run finished 8,000 steps and its checkpoint was stamped
**35.00%**. that number is a single held-out pool of 200, and the evaluations
around it sat between 24.0% and 31.5%. measured honestly over 20 pools the
checkpoint is **30.12% ± 3.14** — 60x chance. the stamp was a lucky draw, which
is ledger row 9 recurring in the same run that recorded it, so the gate was
changed rather than noted: it now reads the mean over 8 pools, seeded on the
step so runs stay comparable, with the sd carried into the sidecar.

the four-arm ablation separates who earns the 30.12%:

| arm | top-1 | retained | what it removes |
|---|---|---|---|
| full | 30.12% ± 3.14 | 100% | — |
| frozen | 17.15% ± 2.04 | 56.9% | the learned association, kept at init |
| no_assoc | 17.07% ± 1.98 | 56.7% | association entirely |
| bypass | 15.02% ± 2.82 | 49.9% | the dynamics |

read down the column rather than across. dynamics with no association are worth
**2.05 points** over no dynamics at all. a randomly initialised association is
worth **0.08 points** beyond none — nothing. *learning* it is worth **12.97
points**. so the learned cortico-cortical connectivity is not a contributor
among several, it is the largest single one, and its share has now compounded
15% -> 24% -> 35% -> **43.1%** across successive runs.

one number moved against us and is recorded as such: bypass retention rose from
41% to 49.9%, so the dynamics' own share fell from 59% to 50.1% as the
association's grew. the substrate is still load-bearing by a wide margin, but
half the retrieval now survives skipping it.

s7.joint was still marked RUNNING in `ibm/curriculum.py` while the process
holding it had been killed hours earlier — the first thing another agent reads
to find the frontier pointed at a run that did not exist. it is FAILED against
its own gate now, with the structural claim (32.0M shared substrate against
9.9M and 3.0M heads) kept and the performance claim withdrawn.

the 150k-site run on the remote box briefly looked like a collapse: effective
rank fell 5.52 -> 1.83 by step 250, which is the documented stop condition. it
was not stopped, because the 30k run did the same thing (4.62 -> 2.43 at step
200) and went on to 26.5% by step 1,800 and r_eff 8.5 by the end. against that
trajectory the 150k run is ahead — 3.75% ± 0.71 at step 250 against 2.00% at
200. **the check is the one CLAUDE.md already prescribes: compare against a case
whose answer you know, not against the quantity's own earlier value.**

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
