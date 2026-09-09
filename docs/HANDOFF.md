# handoff — 2026-09-09

State at the end of the overnight session, for picking this up cold. Covers both
repos. `docs/LOG.md` holds the full chronology and the corrections ledger; this
is the shorter thing you need to resume.

---

## What is running right now

**Superseded — see `docs/LOG.md` for the current session.** As of the following
morning four jobs are live: the interoception 30k repeat below, a contrastive
video-continuation run (`logs/video_cc_w45.log`), an end-to-end disjoint-transport
run on `gb10-direct` (`logs/e2e.log`), and gait search J (`IHM-1/logs/gait_J.log`).
The two MSE video runs were stopped as structurally degenerate.

```
.venv/bin/python -u scripts/ablate_interoception.py \
    --sites 30000 --steps 400 --out out/intero_ablation_30k.json
    > logs/intero_30k.log
```

Repeats the interoception ablation at full scale. The published numbers were
measured at **8,000 sites** under load 92 with three other agents on the GPU, and
the agent that produced them asked for this repeat before anything is quoted.
Expect ~10 h. It holds ~2 GB, not the ~87 GB an earlier launch did — that launch
predated `997eccb`, which resolves `--checkpoint-every` from `--sites` (0 below
12k, 8 above) instead of defaulting to a value that is only safe at the site
count someone happened to test.

**What to do with the result:** compare `cortex_lumped_delays` against
`cortex_trained`. At 8k that gap was **−0.124** and it is the most novel positive
result of the session (below). The dynamics-free controls have already reproduced
at 30k (`ridge_no_dynamics` +0.4956, identical as it must be — closed form and
site-independent), so the cortical arms are the part that was worth repeating.

Two GB10s. `gb10-direct` was running a visual 150k job — check
`ssh gb10-direct 'tail ~/ibm-1/logs/vc_s150k.log'`; it was at step 8,750,
26.12% ±2.00 top-1, r_eff 6.20, and healthy.

---

## The one thing to read first

**The cortical sheet was never information-blocked. It is amplitude-blocked, in
both directions.**

Forward, ~1/300 of a signal survives each hop. Backward, the encoder loses
**175×** of its gradient crossing to a disjoint region. Those two facts together
resolve what looked contradictory all session: a head fit on *frozen* features
reads **48.4× chance** from a precentral-only readout, while every *end-to-end*
attempt to train through the sheet has failed. Training was starved of gradient,
not of data and not of a correct objective.

| | base | concentrated |
|---|---|---|
| encoder gradient, disjoint readout | 1.883e-03 | 4.035e-02 |
| encoder gradient, readout at the port | 3.292e-01 | 3.033e-01 |
| **gradient lost crossing the sheet** | **175×** | **7.5×** |

**Measured at INITIALISATION, and that may be the whole story.** A random head
backpropagates a random error. The end-to-end run testing this reports encoder
gradient as it trains, and the base arm goes 5.453e-05 → 2.248e-01 in 250 steps
while rising to 2.6× chance. If it keeps rising, the 175× is an initialisation
artifact and transport is not what blocked end-to-end training.

---

## Results that stand

- **The dynamics are necessary.** Severing the association kernel takes disjoint
  retrieval from 48.37× chance to 1.00×, with across-image variance exactly
  `0.000e+00` and effective rank 1.0.
- **What they learned is not.** A permuted kernel reads 50.75× — indistinguishable
  from trained, with amplitude, rank, train loss and train top-1 all preserved.
  Now confirmed on **three independent pathways**: motor, vision, viscera.
- **Peripheral anatomy IS load-bearing, and so are the delays.** Dropping the
  vagal C group (507.8 ms) costs **−0.378**; dropping vagal A-beta (9.2 ms) costs
  −0.122; **lumping every delay to step 0 while keeping all 15 channels costs
  −0.124** — as much as deleting a whole fibre group. Noise band measured, not
  assumed: ±0.04. *(8k sites; the 30k repeat is running.)*
- **Concentration beats gain.** Raising gain is counterproductive — it jumps the
  resting rate past the sigmoid's midpoint and transport *falls* 400×. Transport
  is unimodal in the operating point, peaking near r_max/2, which retro-explains
  the old near-critical result (1.254 at 42 Hz, collapse at 75 Hz) as one curve
  sampled either side of its peak.
- **The fix is a better wire, quantified.** 35.4× amplitude → 23.4× noise
  tolerance, **zero improvement at zero noise**. No regression on the sheet's own
  task and 4× better under noise.
- **Effective rank 12.9 of 2,012.** Whatever crosses the sheet is ~13-dimensional
  in every arm that conducts. Probably the tightest real constraint on any
  somato-motor design, and more useful than any amplitude number.
- **TCT loop** oscillates with no stimulus at 6.92 Hz; severing the
  thalamo-cortical projection takes the cortical swing 39.49 Hz → exactly 0. Two
  honest negatives with it: it is a driven relay with feedback rather than a
  closed loop, and it lands in theta, not the spindle band the declaration claims.

## Results that do not

- **It does not walk.** One genuine step — foot airborne at 0.14% body weight,
  1.8 cm clearance, 17.1 cm advance, landing on measured contact — then a forward
  pitch at 1.49 s. 1.88 cm of travel. **The controller can step or travel, never
  both**: stage cost on `pelvis_tx/speed` moves travel 2.24 mm → 119 mm (53×,
  corroborated across twelve parameter sets), and every travelling arm loses its
  step and falls.
- **No video number is citable.** `video_multifilm.pt` reads MSE 0.857 at horizon
  8 against the **0.102 its own log recorded at that step**, with persistence
  matching to 9% and neither output interpretation explaining the gap. Until a
  checkpoint reproduces its own run's held-out loss, nothing from it — including
  the −0.66 skill it reported as best — can be quoted. `video_v6` is separately
  and solidly **−35.43** skill against persistence.
- **The cortex adds nothing to interoception.** Severing costs −0.0058 (slightly
  *better* severed); a permuted kernel retrained beats the trained one; a
  dynamics-free MLP beats both.
- **Sight, hearing and touch do not integrate** on the unmodified sheet: driving
  all three gives precentral variance 7.170e-05 against a linear-sum null of
  7.169e-05. Ratio 1.0002 — pure superposition.

---

## Pick up here

Ranked by value per hour.

1. **Run the end-to-end training the gradient result implies.** This is the
   obvious next experiment and nobody has done it. The gradient measurement says
   there is 21× more gradient to train on with the concentrated kernel; it does
   **not** show that it trains. Only a run does. If it trains where the
   unmodified sheet did not, that closes the whole line and unblocks the
   somato-motor materialization.
2. **Finish the 30k interoception repeat** and quote the delay-lumping number
   from it, not from 8k.
3. **Resolve the video checkpoint discrepancy** — or retire the video term. It
   has consumed a lot and currently supports no claim.
4. **Gait:** two parameters sit pinned at their bounds (`a_stance_hipext` 0.500,
   `b_kneeext` 0.800) — the search wants more of both and cannot have it, and
   widening them is the cheapest untried move. Forward pitch is now the binding
   failure and **the lumbar actuators are in the catalog receiving no command at
   all.** Do not spend more CMA on the current parameterisation; searches G and H
   converged to identical best reports.
5. **App:** 7 browser tests fail, all on the same symptom — `#scene-status`
   never emptying, i.e. the 2,229-structure load not finishing. **None is an
   assertion about behaviour.** An earlier claim that the ring caused this was
   WITHDRAWN: a paired A/B on the same build, alternating minutes, gives ring-off
   32.3 s against ring-on 35.0 s with fully overlapping ranges (~8%, not the
   doubling first reported). The 15.5-vs-30.1 min gap was machine load measured at
   different times on a shared box. No behavioural regression is indicated; a
   clean pass/fail needs both suites run back to back when the machine is quiet.

---

## Traps that cost real time today

Read `CLAUDE.md` — the **Randomness** section is new and was written from two
instances in one day. Beyond it:

- **A checkpoint that cannot open its own weights.** `read_sites` was hardcoded
  4096 while `video_v6` was trained at `dyn.n // 8` = 3,750. Size from the
  artifact, never from today's default. Same shape as the association graph:
  `ckpt/ibm1_implicit.pt` carries **no graph at all**, so every materialization
  from the flagship fused artifact ran a quarter of its connectivity at random.
- **`cfg.get("target", "residual")` cannot rescue a key that is present and
  null.** The checkpoint records `target: None`; the default never fired.
- **Reconstructing a split.** The film corpus grew from 5 to 30 between runs, so
  `sorted(glob(...))[-2:]` returned films the run never designated. The run
  writes its holdout *by name* — read it, and cross-check the run's own logged
  persistence before believing anything below it.
- **Measuring under contention.** A baseline browser run taken *while another
  full suite was still running* was used to wave off seven failures as
  environmental. They were not. If a measurement competes with another job for
  the same resource, it is not a baseline.
- **92 of 144 bridge routes** used typed trunk lengths instead of
  `nerves[].path_length_m`, the field IHM's own contract marks
  `never silently substitute`. Vagus is 508 mm measured against 350 typed.

---

## Artifacts worth looking at

- `IHM-1/artifacts/gait_one_step.mp4` — the real gait result, rendered from the
  actual trajectory. One step, then the pitch.
- `IBM-1/out/clips/video_0652.gif` — hourly video clip. **Visual assurance, not
  an eval**: 11 minutes of one film is memorisation, and the model is 36× worse
  than copying the previous frame.
- `IHM-1` app — 135 unit tests pass; the ring labels every unmeasured edge
  `not measured` and a unit test enforces it. Do not "fix" that by filling in
  plausible values; the emptiness is the honest content.

## Ledger

Four new withdrawals today (rows 21–24), three of them mine. The pattern held
without exception: **a quantity computed correctly and compared against the wrong
thing.** Two were caught only because someone went back and ran a control that
had already been identified and skipped. The `relabel` arm — permute something
that *cannot* change the answer, and confirm it scores what intact scores — is
now permanent in the ablation's default mode set, and it is the single cheapest
guard added all day.
