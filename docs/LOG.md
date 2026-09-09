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
| 13 | the video loop is learning continuation (recon 0.540 -> 0.011, a 50x fall, and visibly sharper clips) | **25% WORSE than persistence** on held-out frames: model 0.01235 against persistence 0.00985, skill -0.2535. it looked right because at horizon 8 on 25 fps film, frame t+8 resembles frame t. skill against the ZERO baseline was +0.9787 -- the flattering comparison, and the meaningless one | computing the persistence baseline, which the loop had never logged |
| 14 | the residual objective beats persistence (skill +0.0003, printed as "BEATS PERSISTENCE") | the model emits a residual **3.9% of the true magnitude with cosine -0.0016** -- it learned to output ZERO, which *is* persistence. it was reproducing the baseline, not beating it | measuring the predicted residual's size and direction, which the loss cannot distinguish |
| 15 | MSE cannot do video at either parameterisation; video continuation does not work in any form tried | three of the four failures read `s[1][:, -n//8:]`, the eighth of the sheet the drive never reaches -- across-batch state sd there is **0.000004 against 0.003099** in the driven region, 775x less, and a readout probe scores **chance from N=1 onward** where the drive itself scores 99.80%. reading the whole sheet takes the same objective from -1.08 to **-0.149** | asking whether the readout can identify the frame that drove it -- a question with a known answer |
| 16 | next-frame video training learns cortical wiring that transfers to EEG (91.5% recovery) | pairing each frame with a **RANDOM** frame from the same film -- temporal structure destroyed, everything else identical -- still transfers **83-86%**. only ~5 of the 35.5 points depend on prediction. the mechanism is gradient flow through the dynamics, not next-frame learning | running the shuffled-target control, which the saturation-by-step-2,500 should have prompted immediately |
| 17 | the shared kernel is carrying the ten per-subject terms -- their mean doubled from 4.97% to 11.09% while each got only 3% of the budget | a SOLO run of one subject for the same 135 own steps, same warm start, same kernel, reaches **17.37%** where the shared run reaches 15.44%. s03 is a wash. sharing is neutral to slightly negative; the doubling is what 135 steps buys either way | running the matched-step solo control instead of reading the trajectory |
| 18 | warm-starting does not cross resolutions -- the 2k run opened at 0.44% despite "warm-started (20 tensors)", so the restart cost 3,000 steps and bought nothing | it reaches **25.87% by step 1,000** where the cold run was at 4.31% by 2,000 -- **4x at matched steps**. only the port projection fails to transfer, and it is re-learned inside 1,000 steps. I read step 0, the one moment the untransferable tensor dominates, and generalised | letting the run continue and reading the trajectory instead of its first point |
| 10 | the joint MEG term is reaching skill +0.45 | that is a **training** loss. held-out is -0.003, and the ceiling is +0.036 | the regression control |
| 11 | ~~the LibriBrain arrays carry no envelope tracking~~ **and, one entry later, that speech→MEG is hard at all** | the builder assumed `timemeg - timechapter` was CONSTANT; the clocks differ by **4,300-5,300 ppm**, which is ±3.4 s of drift across a chapter and smears a 1-8 Hz effect across 3-27 cycles. resampling onto the fitted line takes the corpus from p=0.171/0.463/0.902 to **p=0.024 in all three windows**, peak at 140 ms. the effect was averaged away by the builder, and four negative results are suspended with it | fitting a LINE where a constant was assumed, after the gate's own sensitivity floor was measured |

**the shape they share:** a quantity computed correctly and then compared against
the wrong thing — the wrong population, the wrong units, the wrong split, the
wrong baseline, or no baseline. Not one was a modelling error. Every one was
caught by a measurement that could have been run first, and several by one I had
already written.

---

## 2026-09-09 — 128 sites is below the plateau, and IHM-1 is running 128

the sweep now has the arm that matters for the embodiment.  one term, one task,
identical settings, resolution the only variable:

| step | 128 sites | 512 sites | 2,048 sites |
|---|---|---|---|
| 2,000 | 17.44% | 21.62% | 20.12% |
| 3,000 | 17.37% | 20.50% | 22.62% |
| 4,000 | **19.44%** | 23.00% | 23.00% |

512 and 2,048 are indistinguishable -- identical at 4,000 -- and **128 is below
both at every measured step.**

taken alone, 19.44% against 23.00% is 3.56 points at a combined sd of 3.38, which
is 1.05 sd and settles nothing.  what settles it is the consistency: 128 is lower
in **6 of 6** paired comparisons against the two larger arms across three
independent evaluations.  a sign test gives p = 0.031, and the mean deficit is
**3.73 points**.  a single marginal gap repeated six times in the same direction
is a different quantity from one marginal gap, and this project has been burned
enough by reading a single comparison that the distinction is worth stating.

so the plateau has a floor between 128 and 512, and **IHM-1 is running underneath
it.**  the cost is real but small -- roughly 3.7 points of top-1, 16% relative --
and it is the difference between 38.9x and 46.0x chance, not between working and
not working.  moving to 512 sites would recover it at 4x the substrate, which on
a 262,144-parameter kernel is 65,536 parameters against 262,144: still trivial
inside a physics loop.

that is now measured rather than assumed, which is what the last entry said it
should be.

## 2026-09-09 — the controlled resolution sweep says 512 and 2,048 are equal

the previous entry compared 2,048 against 30,000 sites and concluded that a
smaller substrate performs better on the nerve pathway.  it matched on own-steps,
which was the right correction, but the two runs still differed in objective mix
and schedule.  so the clean version: ONE term, one task, identical settings,
resolution the only variable, 4,000 steps each.

| step | 512 sites | 2,048 sites |
|---|---|---|
| 2,500 | 20.69% | 21.37% |
| 3,000 | 20.50% | 22.62% |
| 3,500 | 20.56% | 22.25% |
| 4,000 | **23.00%** | **23.00%** |

**identical at the end, and 2,048 slightly ahead through the middle.**  a
four-fold difference in substrate size produces no difference in outcome on this
task, which is a narrower and more useful statement than "smaller is better".

so the earlier claim needs qualifying rather than withdrawing.  what holds:
2,048 beats 30,000, and 30,000 beats 150,000.  what does not hold: a monotone
"smaller is better" -- between 512 and 2,048 the curve is flat.  the honest shape
is a plateau at the small end with degradation above it, not a gradient, and the
useful consequence is that anything in the 512-2,048 range costs nothing.

that is good news for the embodiment specifically: IHM-1 runs 128 sites, and the
flat region extends at least down to 512, so their resolution is close to a range
where nothing is lost.  whether 128 itself is inside the plateau is not measured
-- the sweep starts at 512 -- and that is the arm worth adding rather than
assuming.

the 8,192-site arm is still producing its first evaluations.  I launched it twice
by accident, so two processes shared a GPU and a log file for seventeen minutes;
the duplicate is killed and the survivor is running alone.

## 2026-09-09 — a 2,048-site substrate beats a 30,000-site one on the nerve pathway

`optic_nerve` reaches **23.56% (47.1x chance)** on the 2,048-site run against
**14.69%** on the 30,000-site one.  the raw comparison is confounded -- the small
run gives that term a third of its steps and the large run a tenth -- so it was
matched on OWN steps, the ones the term actually took:

| own steps | 30,000 sites | 2,048 sites |
|---|---|---|
| 300 | 3.75% | 5.75% |
| 600 | 8.37% | 10.06% |
| 900 | 9.87% | 11.37% |
| 1,200 | 12.06% | **18.00%** |

**the gap narrows under matching but does not close, and it widens with
training.**  a substrate one-fifteenth the size is not merely competitive on the
anatomically-routed visual term, it is ahead at every matched point.

that is the third independent measurement pointing the same way.  the 150,000-site
model finished 6.5 points BEHIND the 30,000-site one on the designated test set
after 28 hours; its kernel transferred at -32%, below the random floor; and now
2,048 beats 30,000 on the nerve term.  **on these corpora, more substrate has not
once bought accuracy.**  what it bought was where the computation sits -- the
150k model's bypass path was much weaker -- and that is a real property, but it
is not the same thing as being better.

the practical consequence is good news for the embodiment: the kernel the body
sim wants is small, and small is what performs.  the 2,048-site run exists
because IHM-1 runs a 128-site cortex and resampling down is lossy in one
direction; it turns out to also be the better model.

what this does NOT say: that 2,048 is optimal, or that the ordering holds on
other tasks.  visual_eeg is roughly equal on both (28.50% against 27.75%), so the
advantage is specific to the nerve-routed term so far.  a resolution sweep on one
task with matched own-steps is the clean experiment and has not been run.

## 2026-09-08 (overnight) — warm-starting DOES cross resolutions; I read step 0 and called it

**this entry is a correction of itself.**  what follows the rule is the measured
trajectory; what precedes it is what I claimed an hour earlier from a single
evaluation, and it was wrong.

the native 2,048-site run was restarted with warm-start checkpoints shipped to
the remote.  step 0 opened at **0.44%** -- chance -- and I concluded the warm
start had done nothing, because `to_cortex` maps the encoder onto a port whose
size is a fraction of the site count and therefore cannot transfer across
resolutions.  that mechanism is real.  the conclusion drawn from it was not.

the trajectories, same objective, same corpus, same resolution:

| step | cold (first run) | warm-started |
|---|---|---|
| 0 | — | 0.44% |
| 1,000 | — | **25.87%** |
| 2,000 | 4.31% | 26.87% |
| 3,000 | 6.56% | **25.81%** |

**four times the performance at the same step count.**  the warm start transfers
the encoder and the readout, and the model then has only the port projection left
to learn -- which it does within 1,000 steps, going 0.44% -> 25.87%.  what I
measured at step 0 was the one moment where the untransferable tensor dominates,
and I generalised from it.

so: "warm-starting is within-resolution only" is withdrawn.  the correct
statement is that the port projection does not transfer and is re-learned
quickly, while the encoder and readout do transfer and are worth carrying.  the
restart cost nothing; it gained 19 points at step 3,000.

the original claim, kept for the ledger:

## 2026-09-08 (overnight) — [WITHDRAWN] warm-starting does not cross resolutions

the native 2,048-site run for the body sim was going well -- at step 3,000,
optic_nerve **13.31%**, higher than the 30,000-site run's 11.37% at four times
the steps, which is the first sign that the anatomically-routed term may prefer a
smaller substrate.  its `audio_meg` was collapsed because the remote had no
warm-start checkpoints, so I shipped them over and restarted the run.

**that was a mistake and it cost the 3,000 steps.**  the restart opens at chance:

    visual_eeg  0.44% at step 0, after "warm-started (20 tensors)"

the warm start reports success and does nothing, because the tensors it cannot
carry are the ones that matter:

    to_cortex.weight   (3750, 256) at 30k sites -> (256, 256) at 2k
    to_cortex.bias     (3750,)                  -> (256,)

`to_cortex` maps the encoder's hidden vector onto the cortical PORT, and the port
is a fraction of the site count, so its shape is resolution-specific.  the 20
tensors that do transfer are the encoder and the readout; **the entire cortical
input path is re-randomised**, which is exactly the part a warm start is for.

so warm-starting is within-resolution only, and the log line saying "warm-started
from X (20 tensors)" is misleading when 2 of 22 are the load-bearing ones.  the
count should be reported against what the transfer needs rather than against what
happens to match -- a check that names the tensor rather than counting it.

the 2k run is training from scratch now, which is what it should have been left
doing.  no result was lost, only time, and the time was mine to waste rather than
a corpus or a claim.

## 2026-09-08 (overnight) — the kernel survives downsampling; a round trip does not

IHM-1 is running a **128-site** IBM cortex in their embodied loop, and this
repo's published kernels are 30,000 sites.  the first thing measured was a round
trip -- 30k -> 128 -> 30k -- which returned cosine **+0.34** against the original,
and that looked like the kernel not surviving.

**it was the wrong question.**  a round trip asks whether the original can be
RECOVERED; what matters for their sim is whether the downsampled kernel carries
structure.  the kernel acts through <e_i, e_j>, so the test is the site-to-site
similarity structure against a random kernel put through the same downsample:

| n sites | trained | random | ratio |
|---|---|---|---|
| 128 | 0.4269 | 0.0882 | **4.84x** |
| 512 | 0.4448 | 0.0888 | 5.01x |
| 2048 | 0.4634 | 0.0886 | 5.23x |

so a 128-site resample retains nearly **5x** the structure of random.  their
cortex is carrying trained connectivity, and the alarm was mine, from measuring
recoverability when the question was retention.  going DOWN is fine; going back
UP is what loses information, which is consistent with the earlier finding that
k=4 interpolation destroyed a kernel on a 30k -> 150k -> 30k round trip.

a native 2,048-site curriculum is now training on the remote (visual_eeg,
audio_meg, optic_nerve; 262,144-parameter kernel) so the body sim can have a
kernel at its own resolution rather than a resampled one -- resampling is
demonstrably lossy in one direction and there is no reason to spend that when the
run is cheap.

**one hazard fixed, and it matters more now that another project reads these
files.**  `torch.save` streams a zip, so a reader opening a checkpoint mid-write
gets "failed finding central directory" -- a CORRUPT file, not a partial one.
that happened to me reading `ibm1_curriculum16.pt` while the trainer was saving
it.  checkpoints are written to a temp path and renamed now, so a reader sees
either the old file or the new one.

## 2026-09-08 (late) — 16 objectives, 10,000 steps, and the kernel is not worse

the curriculum run has gone far enough to answer the question it was built to
ask.  step 10,000, all 16 materializations alive, none collapsed:

| term | now |
|---|---|
| visual_eeg | 28.25% (56.5x) |
| optic_nerve | 10.56% (21.1x) |
| audio_meg | 5.25% (ceiling 7.12%) |
| cochlear_nerve | 2.00% (4.0x) |
| ten per-subject, mean | 12.64% |
| video / audio_visual | -0.69 / -2.28 vs persistence |

transplanted into the trained EEG model on the designated test set, the
consolidated kernel scores **60.00% (90.1% recovered)** against the fused
predecessor's 59.50% (88.7%).

**that difference is within noise on a 200-image set and is not an improvement.**
what it does establish is the thing that was actually at risk: 16 objectives
pulling on one 3.84M parameter set, consolidated every 500 steps, did not
DEGRADE it.  a kernel serving sixteen materializations could easily have ended up
worse than any single-task kernel, and it did not.

published as `implicit/ibm1.implicit.s30k.e128.curriculum16.step010000` with that
caveat in the sidecar rather than in a footnote.

one bug found by running the evaluation: `transfer_sweep.py` read only
`{"model": sd}` checkpoints and skipped anything else on a KeyError -- silently,
because a skip is not an error.  the curriculum checkpoints store `dyn.embed` at
the top level, so **the run being evaluated was absent from its own evaluation**
and the sweep printed a clean table without it.  three checkpoint shapes are
handled now.

## 2026-09-08 (night) — the port placement carries the result, and what that does and does not mean

`optic_nerve` trains.  solo on the remote it reaches **18.12% ± 1.95 at step
2,000 — 36.2x chance**; inside the 16-way curriculum it is at 9.00% (18.0x) on a
tenth of the budget.  the earlier collapse was starvation, not the architecture.

comparing its curve against `visual_eeg` answers nothing, and that is worth
saying because it was the obvious thing to do: `visual_eeg` was warm-started at
26.75% and has been flat since, `optic_nerve` started from scratch and climbed to
9.00%.  different starting points, so the levels are not comparable.

the question the anatomy actually poses is whether it matters WHERE the drive
enters, and that admits a direct test: take one trained model and move its port,
changing nothing else.  8 pools of 200, chance 0.50%:

| port | top-1 | vs chance |
|---|---|---|
| as-trained (occipital) | **8.69% ± 1.48** | 17.4x |
| shifted (contiguous, elsewhere) | 0.50% ± 0.00 | 1.0x |
| scattered (uniform random) | 0.44% ± 0.17 | 0.9x |
| antipodal | 0.50% ± 0.00 | 1.0x |

every displaced port collapses to chance.  a scattered port of **identical size**
-- 3,775 of 30,000 sites — carries nothing.

**what this establishes:** the model's performance depends on the drive entering
the specific sites it was trained on.  the encoder and readout have not learned
something port-independent; the geometry is load-bearing.

**what it does not establish, and the distinction matters:** that OCCIPITAL is
the right place.  this model was trained with an occipital port, so of course it
fails when the port moves — any trained port would.  the honest reading is
"placement matters", not "anatomical placement is correct".  testing the second
claim needs models trained at different ports and compared on equal footing, and
that has not been run.  the result is real and its scope is narrower than the
framing invites.

`cochlear_nerve` finally escaped collapse — 1.44% solo, 0.87% in the curriculum,
against a 7.12% ceiling — after 5,000 steps of sitting at exactly chance.  the
cause was never hearing: `audio_meg` uses the same corpus, batch and objective
and sits at 5.81%, because it was warm-started and the cochlear term had nothing
to start from.

## 2026-09-08 — fourteen materializations train, and sharing does not pay for itself

fourteen terms against one kernel: the group-mean visual term, speech->MEG, video
continuation, the audio-visual loop, and ten per-subject visual terms built from
`evoked_training_persubject` (10, 16540, 64, 100). at step 4,500, **none
collapsed**, and twelve improved:

| term | step 0 | step 4,500 |
|---|---|---|
| visual_eeg (group mean) | 26.75% | 27.50% |
| audio_meg | 5.37% | 5.25% |
| video | -0.495 | -0.297 |
| audio_visual (cold start) | -13.16 | **-1.25** |
| ten per-subject terms, mean | 4.97% | **11.09%** |

the per-subject mean more than doubled while each term received about 3% of the
step budget, and `audio_visual` -- the only term with no warm start -- improved
tenfold. that reads like the shared substrate carrying the load, and an earlier
status note said exactly that.

**the control says otherwise.** each subject term took roughly 135 of its own
gradient steps by step 4,500. running one subject SOLO for exactly 135 steps, from
the same warm start and the same fused kernel, with nothing else touching it:

| subject | solo, 135 own steps | shared, ~135 own steps + 4,365 from others |
|---|---|---|
| s07 | **17.37% ± 1.78** | 15.44% |
| s03 | 6.44% ± 1.67 | 6.69% |

s07 does **better alone** — 17.37% against 15.44%, about one sd — despite the
shared run having 4,365 additional kernel updates from the other thirteen terms.
s03 is a wash (6.44 vs 6.69). so on two subjects, sharing is **neutral to
slightly negative**, not the multiplier the trajectory suggested.

what the trajectory actually showed was warm-starting plus each term's own steps.
attributing it to the shared kernel required the comparison that was not run
until now, and the honest reading is that **the doubling is what 135 steps of
this task buys, with or without sharing.**

that does not sink the architecture: fourteen materializations do train
simultaneously without collapse, on one 3.84M kernel, and the storage and
publication argument for that is unaffected. but the *performance* argument --
that a corpus constraining the substrate helps every other materialisation -- is
not supported by this measurement, and it is the third time a shared-substrate
claim has failed a control (joint fitting, cross-modal binding, and now
multi-subject sharing).

## 2026-09-08 — the transfer is not about next-frame prediction

**the control I should have run before reporting it.** yesterday's entry claimed
next-frame video training learns cortical wiring that transfers to the EEG task,
on the strength of a 91.5% recovery. destroying the temporal correspondence
entirely -- pairing each frame with a RANDOM frame from the same film, leaving
image statistics, encoder inputs and gradient magnitudes untouched -- costs
almost nothing:

| step | real next-frame task | shuffled targets |
|---|---|---|
| 2,500 | 88.7% | **83.1%** |
| 5,000 | 90.1% | **81.7%** |
| 7,500 | 91.5% | **85.9%** |

so **~5 points of a 35.5-point range** is what actually depends on predicting the
next frame. the other 83% comes from something that never needed temporal
structure at all.

that also explains why the curve saturated by step 2,500 and why the run's own
skill did not predict transfer: the mechanism was never the prediction task. the
fine-grained curve shows it arriving fast and stopping — 35.2% at 250 steps,
60.6% at 500, 78.9% at 1,000, 88.7% by 2,000 and flat.

what survives: gradient descent through the dynamics on natural images *does*
produce a kernel that transfers, and a random kernel recovers 0%. that is still a
real effect and still cheap. it is just not the claim that was made, and the
claim as written -- "next-frame training pushes cortical wiring into shape" --
overstated the mechanism by an order of magnitude.

the next control is running: **noise inputs**, gaussian at matched moments with
no image structure whatsoever. if that also transfers, the effect is not about
vision at all but about any gradient flow organising the kernel, and what is
being measured is closer to a property of the optimiser than of the corpus.

## 2026-09-08 — transfer is decided by the readout, not by task performance

sweeping every video checkpoint on disk through the transplant test. EEG-trained
kernel is the ceiling (63.50%), a random kernel the floor (28.00%), designated
test set, chance 0.50%:

| video run | corpus | readout | own task | top-1 | recovered |
|---|---|---|---|---|---|
| video_fixedread | 37.7 h | whole sheet, learned | **-0.149** | 60.50% | **91.5%** |
| video_via_eeg_learned | 37.7 h | whole sheet, learned | **-5.80** | 60.00% | **90.1%** |
| video_via_eeg_random | 37.7 h | whole sheet, **frozen** | -4.75 | 28.50% | 1.4% |
| video_multifilm | 37.7 h | anterior | -0.66 | 24.00% | -11.3% |
| video_contrastive | 37.7 h | anterior | 0.26 ratio | 28.00% | 0.0% |
| video_residual | 37.7 h | anterior | degenerate | 28.00% | 0.0% |
| video_v6 | 11 min | anterior | -0.25 | 20.00% | -22.5% |

**the corpus is controlled** — six of these trained on the same 37.7 hours. the
variable that separates them is the readout, and it is a clean 2x2:

- whole sheet **and learned** -> transfers ~90%
- whole sheet, **frozen** readout -> nothing (1.4%)
- anterior readout -> **worse than random** (-11% to -22%)

two consequences, and neither was expected.

**task performance does not predict transfer.** `video_via_eeg_learned` failed
its own task about as badly as anything here — skill **-5.80** against
persistence, which is why the previous entry recorded it as a clean negative —
and it produced the second-best cortical wiring in the programme. meanwhile
`video_multifilm` did four times better on video (-0.66) and its wiring is
*worse than random*. so the volume term does not have to succeed at its own task
to be worth running, and its own score is the wrong thing to select it on.

**a broken readout is worse than no training at all.** the anterior-readout runs
do not land near the random floor, they land 11-22 points below it. training
through a readout that cannot see the drive does not leave the wiring
uninformative — it actively organises it around a signal that is not there.

what this revises: the via-EEG result was written up as "negative on both
questions it was built to ask". that stands for its own questions -- the learned
lead field still did not beat a frozen random projection *on video prediction*.
but on the architecture's actual claim the two arms separate completely, 90.1%
against 1.4%, and the learned readout is what made the difference. the run was
not a dead end; it was measured against the wrong objective.

## 2026-09-08 — next-frame video training learns wiring that transfers to EEG

**the strongest evidence the shared-substrate bet has yet received, and it comes
from the cheapest corpus.**

`dyn.embed` is the cortical wiring -- w_ij = M[parcel] x exp(-d/l) x
sigma(<e_i,e_j>), only the third factor trains. both the video run and the
EEG-aligned run move it essentially all the way from initialisation (cosine
+0.001 in each case), so magnitude says both reshaped it. magnitude has misled
this project twice, so the question was asked by TRANSFER: take the kernel the
**next-frame video run** learned -- 40 hours of public-domain film, no EEG
anywhere in it -- and drop it into the visual-contrastive model, changing nothing
else. designated THINGS-EEG2 test set, 200 images, chance 0.50%:

| kernel | top-1 |
|---|---|
| own (EEG-trained) | 63.50% |
| **video-trained, transplanted** | **60.50%** |
| random | 28.00% |
| zeroed | 28.00% |

the video kernel recovers **32.5 of the 35.5 points** the EEG-trained kernel is
worth — **92%** — having never seen an EEG recording.

and it is the structure, not the statistics. three matched controls:

| control | top-1 |
|---|---|
| video, as trained | 60.50% |
| same values, shuffled across sites | 28.50% |
| site-vectors intact, assigned to wrong sites | **15.50%** |
| fresh noise at matched RMS | 28.00% |

shuffling returns it to random, and permuting *which site gets which embedding*
is **worse than random** — the site-to-site correspondence is the thing that
carries. so this is learned cortical structure, not a magnitude effect.

what this changes: the self-supervised video term has been judged by whether it
beats persistence on its own task, where it still loses by 1.2x. by the
architecture's actual claim — that a corpus constraining the substrate
constrains every materialisation — it is the most productive term measured. the
video branch's value is not its own prediction quality.

what this does not show: that the transfer runs the other way, or that video
wiring plus more video keeps improving EEG retrieval. both are cheap and neither
has been run.

## 2026-09-08 — the readout fix is worth 0.93 of skill, and three negatives were about one line

reading the whole sheet instead of the anterior eighth, same objective, same
corpus, same split:

| steps | skill vs persistence | residual ratio | cosine |
|---|---|---|---|
| 0-5,000 | -1.073 | 1.257 | +0.246 |
| 5,000-10,000 | -0.417 | 1.027 | +0.265 |
| 10,000-15,000 | -0.388 | 0.993 | +0.260 |
| 15,000-20,000 | -0.400 | 0.994 | +0.260 |
| **broken readout** | **-0.66 best** | **0.019** | **+0.002** |

best **-0.149** at step 13,000 against the broken version's -0.66, so the fix is
worth **0.93 of skill** on the same task. and the diagnostic numbers move
together with it: the model now emits a residual **0.99x the true magnitude with
cosine +0.26** where the broken one emitted 1.9% of the magnitude with no
direction at all. that is the difference between predicting motion badly and
predicting nothing.

it is still 1.2x behind persistence, so the branch has not turned. but the
failures it was diagnosed with were mostly not about objectives:

| failure | readout | now |
|---|---|---|
| direct L2, -1.08 | anterior | **-0.149** with the fix |
| residual L2, degenerate | anterior | untested, but the degeneracy was the same erasure |
| contrastive, 0.26 of baseline | anterior | untested |
| via-EEG, both arms | whole sheet | **stands** — that one was never affected |

so "MSE cannot do this task at either parameterisation" and "video continuation
does not work in any form tried" were both written against a model that could not
see its own input. the objective diagnosis may still be right; it has not been
tested on a working readout, and that is now the cheap experiment rather than the
conclusion.

what survives untouched: the via-EEG result, which used `linspace(0, n-1)` from
the start — its learned lead field was no better than a frozen random projection
of the same width, and that comparison had nothing to do with this bug.

## 2026-09-08 — the video branch fails under three objectives; and 5x the sites costs 6.5 points

**video, third objective, same answer.** rebuilt as discrimination — frame t
drives the cortex, the state aligns contrastively with an embedding of frame
t+50, negatives from other positions in the same film — and it never once beat
its baseline:

| steps | model top-1 | persistence top-1 | ratio | r_eff |
|---|---|---|---|---|
| 0-1,000 | 1.56% | 43.99% | 0.04 | 4.28 |
| 1,000-2,500 | 3.64% | 31.80% | 0.11 | 1.33 |
| 2,500-4,000 | 5.30% | 21.75% | 0.24 | 1.42 |
| 4,000-6,000 | 6.71% | 25.42% | 0.26 | 1.57 |

**0 of 24 evaluations above persistence**, best ratio 0.26, and effective rank
collapsed from 4.28 to ~1.5. so the cortical state after 8 dynamics steps carries
*less* about the future than the input frame's own embedding does — the dynamics
are destroying information rather than propagating it.

that is now three objectives on the same branch: direct L2 (blur, -1.08 skill),
residual L2 (degenerates to zero, ratio 0.019), contrastive (0.26 of baseline).
the honest summary is that **video continuation does not work in any form tried
here**, and the common factor is the readout, not the loss. one caveat kept for
the record: the contrastive run trained its target encoder jointly, which lets
the baseline drift down (44% -> 25%) — the model still never approached even the
degraded baseline, but a fixed target encoder is the cleaner design.

**scaling, measured on the designated test set.** the 150k-site model at step
5,500 against the 30k model at 5,800, same evaluation, same arms:

| | 30k @ 5,800 | 150k @ 5,500 |
|---|---|---|
| full | **63.50%** | 57.00% |
| frozen | 28.00% | 22.50% |
| no_assoc | 28.00% | 22.50% |
| bypass | 26.00% | 16.50% |
| bypass retained | 40.9% | **28.9%** |
| dynamics contribute | 37.5 pts | **40.5 pts** |

5x the substrate, 18 hours of GPU, and it is **6.5 points worse**. what improved
is where the computation sits: the bypass path is much weaker (28.9% retained
against 40.9%) and the dynamics contribute 40.5 points against 37.5. so more
substrate does move work into the substrate — it just does not buy accuracy.

that is the scaling answer as it stands, and it should be stated as such rather
than left implied: on this task, at this data scale, parameters are not the
binding constraint.

## 2026-09-08 — MSE cannot do this task at either parameterisation

the ablation left the objective as the only suspect, so the decoder was changed
to emit the RESIDUAL and add it to frame t. a zero output is then exactly
persistence, so the model starts level with the trivial baseline and can only
improve by spending capacity on what moves. it cannot buy a lower loss by
blurring toward the mean.

it worked structurally and failed substantively. skill opened at -0.32 against
the direct objective's -11.13, reached -0.002 by step 250, and printed
**"BEATS PERSISTENCE +0.0003"** at step 750.

that line was false, and the check that caught it is the point of this entry.
skill at ±0.002 of persistence is consistent with two very different models — one
that learned the motion, and one that learned to emit **zero**, which *is*
persistence. the loss cannot separate them. the residual's size and direction
can:

    predicted residual RMS   0.00768
    true      residual RMS   0.19937
    ratio                    0.039
    cosine(pred, true)      -0.0016

**3.9% of the true magnitude, and exactly no correlation in direction.** the
model emits nothing. "matching persistence" here means *reproducing* it, not
competing with it, and a run left alone would have reported a win.

so both parameterisations fail, and they fail the same way for the same reason.
MSE asks for the conditional mean of the target. for direct prediction that mean
is a blur; for residual prediction it is zero. **neither is motion**, and no
amount of data or substrate changes what an L2 loss is minimised by — 15.2 h did
not, and the substrate is the part that works (bypass costs 4.41x).

the trainer now prints `res` and `cos` beside skill every eval, refuses to stamp
BEATS PERSISTENCE unless the residual has real size and direction, and **will not
checkpoint a degenerate model** — the gate reads ratio > 0.15 and cos > 0.1
alongside skill.

what this leaves: pixel-space L2 is the wrong question, the third time this
programme has reached that conclusion on a different branch. the visual term went
from skill +0.011 to 53x chance by changing the question from reconstruction to
discrimination. video needs the same move.

## 2026-09-08 — the cortex is not what loses to persistence

the ablation the previous entry said had to run, on held-out films with the same
persistence baseline computed on the same batches:

| arm | MSE | skill vs persistence | vs full |
|---|---|---|---|
| full | 0.09908 ± 0.00521 | -1.08 | 1.00x |
| frozen | 0.25807 ± 0.01684 | -4.42 | 2.60x |
| no_assoc | 0.25709 ± 0.01678 | -4.40 | 2.59x |
| **bypass** | 0.43666 ± 0.04434 | **-8.17** | **4.41x** |

**bypassing the dynamics costs 4.41x.** the substrate is carrying this task more
heavily than any other measured here — retrieval retains 41-50% under bypass,
video retains 23%. and the learned association is doing real work too: freezing
it at initialisation costs 2.6x.

so the two facts sit together and both are true. **the dynamics do the work, and
the whole thing still loses to copying the previous frame.** the failure is not
the cortex; removing the cortex makes it four times worse.

that leaves the objective. MSE frame regression at horizon 8 is precisely the
shape of failure this programme has already diagnosed once: on the visual branch,
waveform regression peaked at skill +0.011 while contrastive retrieval on the
same pairs reached 53x chance — the same encoder, the same data, a different
question. an L2 loss over 64x64 frames is minimised by blur, and blur loses to
persistence by construction, because persistence at least keeps the edges.

the next thing to try on video is therefore an objective change, not more data
(15.2 h did not help) and not more substrate (the substrate is already the part
that works). that is the third time the answer has been "you asked the wrong
question of the data", and it is worth stating as a pattern rather than as three
incidents.

## 2026-09-08 — 42x the data removed the overfitting and not the failure

the video corpus went from 11 minutes of one film to **15.2 hours across 13
public-domain features**, and the split became by FILM rather than by frame, so
the test asks about footage the model has never seen instead of about frames
adjacent to its training set.

that fixed exactly what it should have, and nothing else:

| steps | skill vs persistence | train | held | overfit |
|---|---|---|---|---|
| 0-2,000 | -2.82 ± 3.17 | 0.194 | 0.155 | 0.80x |
| 2,000-6,000 | -1.53 ± 0.21 | 0.133 | 0.115 | 0.87x |
| 6,000-10,000 | -1.34 ± 0.19 | 0.117 | 0.108 | 0.93x |
| 10,000-14,000 | -1.37 ± 0.21 | 0.114 | 0.102 | 0.89x |
| 14,000-20,000 | -1.20 ± 0.28 | 0.107 | 0.099 | 0.93x |

**the overfitting is gone** — held-out error now sits *below* training error
(0.93x), against the 9x gap the via-EEG arms showed on the old corpus. the
acquisition did its job and the data argument is settled: this term is no longer
data-bound.

**the failure survived it.** the model is still **2.20x worse than persistence**
on held-out films, improving about 0.03 of skill per 1,000 steps and flattening.
best ever -0.81 at step 14,000. so the honest reading of ledger row 13 changes:
losing to persistence was not an artefact of eleven minutes of unusually static
film. it is what this materialisation does.

one supporting number: persistence MSE is 0.044 on these films against 0.0086 on
Koyaanisqatsi, so the trivial baseline here is five times weaker and the model
still cannot clear it.

what this does not say: that the substrate is at fault. the same VideoLoop
architecture, the same encoder and decoder, would have to be run without the
dynamics before any of this attaches to the cortex rather than to the
frame-prediction setup around it. that ablation is the next thing to run, and
until it does, "video continuation does not work" is the claim, not "the cortex
cannot do video".

## 2026-09-08 — video through the sensor projection: negative, twice over

the materialisation the user asked for: frame -> cortex -> sensor projection ->
next frame, with the decoder seeing **only** the projection and never the
cortical state, so the neural readout is load-bearing by construction rather than
a head hanging beside a video branch. that construction is right — it is exactly
what m-multi lacked, where the MEG head sat at chance for 29,000 steps while the
video branch trained happily around it.

the result is negative on both questions it was built to ask, over steps
8,000-13,250:

| arm | held MSE | skill vs persistence | train | overfit |
|---|---|---|---|---|
| learned lead field | 0.05871 ± 0.00359 | **-5.80** | 0.00682 | 8.6x |
| frozen random projection | 0.05542 ± 0.00232 | **-4.75** | 0.00562 | 9.9x |

*first*, both are catastrophically worse than persistence (MSE 0.00864). routing
continuation through a 64 x 8 sensor bottleneck costs roughly 6x the trivial
baseline.

*second, and this is the one the control existed for*: the learned lead field is
**not better than a frozen random projection of identical width** — it is 0.77 sd
*worse*, i.e. indistinguishable. whatever the bottleneck contributes is a fact
about its width, not about the readout being a lead field. without that arm a
reader would have taken the first result as "the projection is too narrow" when
the projection being *learned* buys nothing at all.

the honest qualifier: both arms overfit ~9x (train 0.006 against held 0.055) on
11 minutes of film. so this is not yet evidence that the architecture cannot
work — it is evidence that it cannot be evaluated on this corpus. the binding
constraint is data, which `docs/PROGRAMME.md` has said since the beginning and
which every video result here keeps re-demonstrating.

both runs were stopped at 13,250 of 20,000 rather than left to reconfirm a flat
held loss for another hour.

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
