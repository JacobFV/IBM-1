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
| 19 | the IBM kernel cannot carry a motor task -- trained, permuted and random kernels were equivalent after fine-tuning on the body | the teacher was the bare postural servo, which **falls at 1.19 s**. the corpus was a body falling over, and all three arms lost to predicting the mean because there was nothing to learn. with the engineered LQR, which holds 12 s, a ridge on the same split reaches **+0.9801** -- the corpus is learnable and the earlier experiment measured nothing | checking whether the teacher could actually stand |
| 20 | the cortex learns motor control -- body_stance went -41 to **+0.42** skill once the readout could see the sheet | trained and permuted kernels give **identical MSE to 8 decimals**. the whole-sheet readout samples the driven region, and those samples carry **3,134x** the variance of the rest -- the decoder reads the input, not the cortex. the architecture has no configuration where the dynamics both receive the signal and are necessary | the four-arm ablation, which was already running when I announced the result |
| 10 | the joint MEG term is reaching skill +0.45 | that is a **training** loss. held-out is -0.003, and the ceiling is +0.036 | the regression control |
| 11 | ~~the LibriBrain arrays carry no envelope tracking~~ **and, one entry later, that speech→MEG is hard at all** | the builder assumed `timemeg - timechapter` was CONSTANT; the clocks differ by **4,300-5,300 ppm**, which is ±3.4 s of drift across a chapter and smears a 1-8 Hz effect across 3-27 cycles. resampling onto the fitted line takes the corpus from p=0.171/0.463/0.902 to **p=0.024 in all three windows**, peak at 140 ms. the effect was averaged away by the builder, and four negative results are suspended with it | fitting a LINE where a constant was assumed, after the gate's own sensitivity floor was measured |
| 21 | the sheet does not conduct because `tanh(2*sim)` is saturated -- unsaturating the squashing function would restore magnitude selectivity and let the kernel build a strong specific pathway | the edges ARE pinned (mean \|tanh(2*sim)\| = 0.895, 71.4% of 1.44M edges above 0.9) but the underlying cosine similarities have no dynamic range either, so lowering the slope scales every edge down together and selects nothing. row L1/max moves 45.2 -> 41.0 as the slope goes 2.00 -> 0.25, against 48 for perfectly flat. **saturation was the symptom; the flat \|sim\| distribution is the disease**, and no reparameterisation of tanh reaches it -- only an explicit structural selection (top-m) does, which buys 11.9x at zero change in row gain | measuring the row L1/max flatness across a slope sweep, instead of reasoning from tanh(2) = 0.964 to "therefore no magnitude information" |
| 22 | the gait controller cannot walk because it regulates absolute fore-aft position -- `error = x - x0 - dx_velocity` masks only the pelvis_tx SPEED, so the position term acts as a spring pulling the body back to its start, spending ~0.47 of the [0,1] actuator range at only 6 mm of travel | `deflation_basis` had already removed it. the pelvis_tx POSITION column of K is **exactly zero** (L2 = 0.0000, rank 258 of 258 columns), as is pelvis_tz; only pelvis_ty (height) is regulated, at L2 52.7, which is correct. the arithmetic was built on a column that does not exist. **I flagged this exact caveat when I sent the hypothesis and then did not check it before reporting the result** | printing the norm of K's pelvis_tx column -- one command, available before the claim was made |
| 23 | the learned kernel is load-bearing for disjoint-region retrieval -- permuted sat at **exactly chance** while intact read 43x, so what the cortex LEARNED carries the signal from occipital to precentral | a train/test permutation mismatch. `torch.randperm` was drawn INSIDE the feature extractor from a generator shared with the caller, and the extractor runs twice per arm -- once for training features, once for held-out -- so the head was fitted on one permutation and evaluated on another. rerun with the permutation drawn once and passed in, **permuted reads 50.75x against intact's 48.37x**: indistinguishable, with amplitude (5.4367e-02 vs 5.4064e-02), effective rank (12.2 vs 12.9), train loss and train top-1 all preserved. row 20's generalisation is CONFIRMED, not amended -- the dynamics are necessary, what they learned is not | asking why permuted sat at exactly chance while preserving 99.7% of the across-image variance, then adding a `relabel` arm (readout columns permuted, which CANNOT change the answer) that had to score what intact scores -- it read 48.62x, so the bookkeeping was clean and the other arms were interpretable |
| 24 | every visceral conduction delay this model has quoted -- vagus C at 350 ms, the splanchnics at 300 -- and, in the handoff doc, that only 52 of IHM's routes joined on a measured length | `ihm_bridge.routes()` read `path_length_m` from `muscle_bindings` and `receptor_patches` and never from `nerves[].path_length_m`, which is the field IHM's own `route_contract` names with `missing_length_policy: error; never silently substitute a trunk length`. **92 of 144 routes** silently used my typed trunk table instead, and every visceral route is one of them because the vagus innervates no muscle and carries no skin patch. vagus is **508 mm** measured against 350 typed -- **158 ms** on the C fibre, in exactly the latency that separates visceral sensation from touch -- and greater splanchnic 168 against a 300 default, 79% the other way | wiring an actual visceral materialization and printing `length_source` per route |
| 25 | video_v6 is **-35.43** skill against persistence and video_multifilm.pt cannot reproduce its own log (MSE 0.857 against a logged 0.102), so no video number is citable | `read_idx` -- WHICH cortical sites the readout samples -- was a plain tensor attribute, not a registered buffer, so it was never in the state dict. the convention later moved from the anterior eighth `s[1][:, -n//8:]` to `linspace(0, n-1)`, and every earlier checkpoint loads into the new code **cleanly** -- identical shapes, so `load_state_dict` reports nothing -- while `from_cortex` is fed a different set of sites than it was fitted on. read the way it was trained, video_multifilm reproduces **0.10179**, its own log to five decimals, and video_v6 is **-0.2535**, not -35.43. **a factor of 112**, and the difference between 'catastrophic' and 'slightly worse than copying the previous frame' | replicating the trainer's own eval draw exactly, seeing persistence match to five decimals (0.06128) while the model did not, and concluding the fault was in the code rather than the data |
| 26 | the simulated body has **zero sex-specific entities** -- no breast, uterus, ovary, testis, prostate or external genitalia among its 4,000 | it carries **~30 male genital entities** -- corpus cavernosum and spongiosum, glans, testes, epididymides, deferent ducts, seminal vesicles, ejaculatory ducts, prostate and vessels, all bound to `pelvis`. the search ran against entity IDs, which are opaque strings like `body-bp3d-FJ1252` with no name field, so it matched nothing for **every** term. what IS absent is any female-specific entity and any mammary gland/nipple/areola in either sex, and no catalogued source ships a female mesh | running the same search for `femur`, which returns 0 by the broken method and 4 by the working one. a control that costs one second and that I did not run before writing the claim into docs/DIRECTION.md and telling the user |
| 27 | the gait results -- one genuine airborne step, 164 mm of travel -- are a body moving as a human body can move | **nothing in the plant enforces the joint ranges the model declares.** every rotational coordinate carries `<range>` and `<clamped>true</clamped>`, the model holds ZERO CoordinateLimitForce, and OpenSim does not apply clamping during forward dynamics. checked against the declared ranges, `gait-best` exceeds them on **6 coordinates** -- hip rotation by 25.1 deg, and **knee_angle 12.5 deg BELOW zero, i.e. the knee bending backwards**. a prone search found the same hole and spent it far harder: 16.0 s of sustained locomotion and 973 mm of travel with the ankle at -2.524 rad against a declared +/-0.873, **145 deg of plantarflexion, the feet folded back on themselves** | the search agent measuring its own crawl against the model's declared ranges after the fact, and withdrawing it. the same check then applied to the standing gait results |

**the shape they share:** a quantity computed correctly and then compared against
the wrong thing — the wrong population, the wrong units, the wrong split, the
wrong baseline, or no baseline. Not one was a modelling error. Every one was
caught by a measurement that could have been run first, and several by one I had
already written.

---

## 2026-09-09 (afternoon) — the cortex is a surface now, and the anatomy says the transport task was scored on a pathway the brain does not have

`docs/DISCONNECTS.md` rows 2 and 3, both closed. Three payloads fetched into
`raw/` directories that had held one 12 KB checksums.txt each and no bytes.

### THE TRAINED COMPARISON: tract edges put occipital-to-precentral retrieval at chance

Four matched arms of the visual contrastive term, 30,000 sites, 1,200 steps,
identical but for the sheet and the wiring: same seed, same graph seed, same
data, same schedule, and evaluation pools drawn from a generator seeded on the
STEP so two arms see the same pools at the same step.

**(a) per-hop transport** (`measure_hop_transfer.py`, steady state, drive the
whole occipital region, gates pass: zero amplitude reads 0.0, severed kernel
reads 0.0 off the seed):

    arm       hop1 fraction   hop2         sites at hop1   PRECENTRAL arrival
    random      1.053e-03     1.860e-05       20,148         5.957e-04
    tract       1.252e-03     3.661e-06       11,171         1.698e-06

**Precentral arrival is 351x lower under the tract topology, and the per-hop
transfer is not the reason** — hop 1 is slightly BETTER in the tract arm
(1.25e-03 against 1.05e-03). The difference is entirely that precentral is
further away: random puts it one hop from occipital and tract puts it two,
because the consensus connectome declares no direct occipito-precentral
fascicle. Anatomy does not make hops worse. It makes motor cortex correctly
distant.

**(b) multimodal convergence** (`measure_multimodal_convergence.py`, drop-one
attribution, precentral readout only):

    arm      sight->precentral  hearing->precentral  touch->precentral
    random       2.725e-07          4.288e-07           1.435e-06
    tract        1.425e-12          1.945e-12           1.691e-06

Sight and hearing collapse by **five orders of magnitude** — 191,000x and
220,000x — while touch, which is postcentral->precentral and which the
connectome *does* declare, goes **up** 18%. Both arms are still at "NO
CONVERGENCE" (superadditivity 1.0000; the sheet superposes), so this does not
rescue the somato-motor claim. It relocates the failure: the random graph's
tiny visual and auditory transport was not weak conduction, it was edges that
do not exist.

**(c) real retrieval, read from precentral only** (`ablate_disjoint_transport.py`,
8 pools of 200, chance 0.5%):

    config  noise      random intact   random permuted   tract intact   tract permuted
    base    0            23.00x            26.12x           1.00x           1.00x
    base    1e-3 Hz      10.75x             9.75x           1.75x           1.25x
    base    1e-2 Hz       6.12x             6.75x           0.62x           0.50x
    aniso   0            24.12x            24.87x          19.25x          12.00x
    aniso   1e-3 Hz      21.87x            19.00x           5.50x           5.25x

**With the plain kernel the tract arm sits at exact chance.** Across-image sd of
the precentral rate is 4.35e-05 Hz against the random arm's 1.21e-02 Hz, a factor
of 277. Severed reads chance in both and `relabel` reproduces `intact` in both,
so the bookkeeping is clean.

And the random arm's 23x is **matched by its own permuted control at 26x**, which
is ledger row 23 reproducing exactly: the dynamics were necessary, what they
learned was not. Putting the two together —

> the disjoint-transport result this programme has been building on was carried
> by long-range edges the human connectome does not contain, and not by anything
> the kernel learned about them.

The anisotropy fix rescues the tract arm to 19.25x at zero noise, so a
concentrated tract graph does conduct; it is far more noise-fragile than the
concentrated random one (5.50x against 21.87x at 1e-3 Hz), which is what a
two-hop path costs.

**(d) conduction delays**, attached to the saved tract graph without redrawing it,
at dt = 1e-3 over 4 substeps (which resolves a median declared delay of 3.84 ms);
touch->precentral transport:

    none 1.691e-06   tract 1.629e-06   shuffled 1.585e-06   distance 1.589e-06

Delays cost 3.7%. The delay-MATCHED control — the same multiset of delays moved
to random edges — costs 6.3%, and the euclidean surrogate `tract.py` argues
against costs 6.0%. So the real delays are 2.8% better than a delay-matched
shuffle: the correct sign for the topology's claim, and far too small to build
anything on.

**(e) the designated 200-image test set** (chance 0.50%, the whole set is the
pool so the number is deterministic), whole-sheet readout:

    arm                   full     frozen   no_assoc   bypass
    surface_occ_random   40.00%    42.00%    42.00%     6.50%
    surface_occ_tract    36.00%    34.00%    34.50%     5.00%

`frozen` (random embeddings) and `no_assoc` (geometric prior zeroed) score AS
HIGH AS `full` in both arms. On the whole-sheet readout the learned kernel
contributes nothing at 1,200 steps, in either wiring — ledger rows 12, 20 and 23
again, on a new sheet. And 8-pool held-out top-1 over the trajectory is
16.75% (random) against 16.56% (tract), a **paired** difference of −0.32 points
over 12 matched evaluation points, t = −1.27: indistinguishable.

**(f) the gradient that reaches the encoder**, measured at step 0 of an
end-to-end run (`train_disjoint_end_to_end.py`, occipital drive, precentral-only
readout, plain kernel):

    random edges  |grad_enc| 1.312e-05
    tract edges   |grad_enc| 2.879e-08     -- 456x less

which is the same fact as (a) and (c) seen from the optimiser's side: with no
direct occipito-precentral fascicle there is almost nothing for the encoder to
learn from, and end-to-end training of that pathway is starved. Both arms sit at
exact chance through step 250 on the plain kernel, as the frozen-head ablation
predicts.

**The honest summary of (3).** Anatomy does not beat concentrated-random on any
measurement here, and on the one that matters it loses by a factor of 351 — for
a reason that is correct rather than a defect. A null on "does the tract
topology improve transport", and a positive result on "was the transport being
measured real".

**What these numbers are not.** 1,200 steps is a fifth of the published visual
checkpoint's schedule, chosen so four matched arms would fit one GPU beside
another job. The arms are matched, so differences between them are real; the
absolute retrieval numbers are not this programme's retrieval result.

### how to reproduce any of this

    PYTHONPATH=. .venv/bin/python scripts/fetch_cortical_atlases.py
    PYTHONPATH=. .venv/bin/python scripts/compare_long_range_topology.py --long-topm 4
    PYTHONPATH=. .venv/bin/python scripts/measure_sheet_metric_error.py --connectome
    PYTHONPATH=. .venv/bin/python -m unittest discover -s tests

and the four matched training arms, identical but for the sheet and the wiring
(same `--seed`, same `--graph-seed`, and the evaluation pools are drawn from a
generator seeded on the STEP, so two arms see the same pool at the same step):

    scripts/train_visual_contrastive.py --steps 1200 --eval-every 100 \
        --eval-pools 8 --seed 0 --graph-seed 0 \
        --geometry surface --port-region occipital --long-topology random
    ... --geometry surface --port-region occipital --long-topology tract
    ... --geometry sphere                                                 # status quo
    ... --geometry surface                                                # geometry only

### what was fetched, and where it actually came from

`desikan2006` is `?h.aparc.annot` on fsaverage. Its `.location.yaml` already
said `vendored_in: freesurfer` and that is literally true — the bytes live
inside a FreeSurfer subject directory that reached this machine bundled in an
MNE dataset. **Three such directories are present, staged independently by
different tools at different times, and they are byte-identical for every file
taken.** That agreement is the verification, because the pre-existing
`checksums.txt` held a header comment and no hashes: there was nothing to verify
*against*, and cross-copy identity is what was actually available.

`dkt-atlas` is **not on fsaverage** — the fetch script asserts that rather than
assuming it, by checking that no fsaverage copy on this machine carries a
`DKTatlas` file. DKT is written per subject by recon-all, which is exactly what
the card's `subject_surface_ras` frame says, so the payload is the one full
recon this repo holds (mne-somato subject 01), staged with its own surfaces
because the card's `requires` says per-vertex labels without the subject's
surface are meaningless.

`braingraph-hcp-connectomes` is the 86-node set: 1064 HCP subjects at
Desikan-Killiany resolution, 10× repeated 1M-streamline tractography, and every
edge carrying `fiber_length_mean` — the field `tractometric_matrix` requires and
cannot synthesise. The download is gated behind an "I agree to the HCP data use
terms" checkbox that enables a form in the page's own markup; the agreement is a
click, not a credential.

### the sheet

`cortical_sites()` returns fsaverage white-surface vertices, sampled with
probability proportional to vertex area so density is uniform per mm² of cortex,
medial wall excluded, drawn on the CPU so **the same seed gives the same sheet on
cpu and on cuda** — which the long-range draw it replaces did not.
`cortical_regions()` returns one of 68 hemisphere-qualified DK labels.

**The test that could not be written before:** on 4,000 sites `insula` selects
63 and `cingulate` 110, disjoint from frontal, temporal and parietal.
`region_index` **raises** for those names on the sphere rather than substituting
`frontal`, because a silent substitution is how the disconnect survived.

The interoceptive port is now the insula and anterior cingulate, whole rather
than a 4.5% subsample of a coordinate cut: 5.45% of a 6,000-site sheet against
the 5.3% of white-surface area those three DK labels occupy.

Corroborated on `dkt-atlas` — a different subject and a different protocol,
though the card is right that it is *not* independent evidence about a boundary:
insula is 2.83% of area under DK and 2.38% under DKT on the same brain, Dice
0.835 (lh) and 0.866 (rh); rostral ACC Dice 0.789 and 0.825.

**What the sphere was.** Area-matched to 202,437 mm² against an fsaverage white
surface of 130,438 mm², or 118,310 mm² excluding the medial wall. **1.55× the
cortex by area, 1.24× in linear scale.** Every distance on it was inflated by
about a quarter, including the ~85 mm mean separation of two random sites that
§4.1 blames for the long-range edges being inert. On the real surface a random
partner sits 79 mm away.

### and `postcentral` on the sphere was the leftover bin, at 3.3x its true size

Site fractions at 30,000 sites, against the measured white-surface area:

    label         sphere   surface    true area
    occipital      12.6%     10.6%      10.9%
    temporal       20.3%     19.5%      19.7%
    parietal       17.8%     19.4%      19.0%
    frontal        22.7%     28.8%      28.6%
    precentral      6.7%      7.1%       7.0%
    postcentral    20.0%      5.9%       6.0%
    insula          n/a       3.3%       3.3%
    cingulate       n/a       5.3%       5.5%

The surface column matches the area column to within 0.3 points everywhere,
which is the direct check that the area-weighted sampling does what it claims.

The sphere's `postcentral` does not: **20.0% against a true 6.0%**. Reading
`_sphere_regions` explains it — `lab` is initialised to 5 (postcentral) and the
other five labels are then written over it, so `postcentral` was never a region.
It was everything that matched no threshold. And it is the touch entry port in
`measure_multimodal_convergence` and the afferent target IHM-1 routes body
sensation to, so every somatosensory transport number measured on the sphere was
driving a fifth of the cortex and calling it the postcentral gyrus.

`frontal` is understated in the other direction, 22.7% against 28.6%. The
interoceptive port's `PORT_FRACTION` of 4.5%/22% then works out to 4.65% of the
sheet against the true insula+ACC 5.3% — so the substitution had roughly the
right SIZE and entirely the wrong PLACE, which is the more dangerous of the two
errors because the size is what anyone would have checked.

### the "occipital port" was never occipital, on either sheet

`VisualContrastiveLoop`'s docstring says "the image drives the occipital port".
It drove `drive[:, :n // 8]`. Measured, at 30,000 sites:

- on the **sphere**: the slice is a uniform mix of all six labels — frontal 22.1%,
  temporal 21.0%, postcentral 20.9%, parietal 17.0%, occipital **12.3%**,
  precentral 6.7%. Occipital's share of the slice is occipital's share of the
  sphere. The slice is anatomically uniform; it is not a port.
- on the **surface**: it is **100% left hemisphere** (fsaverage vertex order puts
  all of lh before rh), spread across all 34 lh parcels in proportion to their
  size, occipital 9.5% against its 10.6% share of the sheet.

This is ledger rows 15 and 25 in a third costume: a name asserting an anatomy the
index did not have. `port_region="occipital"` makes it true and is what the new
arms use; the default stays the slice so every checkpoint on disk still loads
meaning what it meant.

### tract-constrained long-range edges

`ibm/cortical_tracts.py` turns the connectome into the `matrix` and `lengths_mm`
`tractometric_matrix` declares it needs, and `CorticalDynamics(long_topology=
"tract")` draws each site's long-range partners among the sites in the parcels
the consensus connectome joins its parcel to, carrying that pair's arc length and
delay. The builder itself is not called: its full expansion is 1.7×10⁸ edges at
30,000 sites and 10¹⁰ at 250,000, and its own docstring says the quadratic cost
is the honest signal that the materialization is asking for more than the
connectome has. What is drawn is a uniform subsample of exactly that edge set.

**The control is matched by construction, not by fitting** — same `n_far` per
site, same flat long-range prior, so edge count is identical and row L1 agrees to
0.000e+00.

Structural transport, the held nonnegative operator applied 4 times to a unit
indicator on the drive region, 30,000 sites:

| pair | random | tract | tract/random | vs *concentrated* random |
|---|---|---|---|---|
| occipital→precentral | 3.10e-02 | 7.95e-03 | **0.257×** | 0.257× |
| occipital→temporal | 9.45e-02 | 1.48e-01 | 1.571× | 1.558× |
| postcentral→precentral | 5.25e-02 | 6.97e-02 | 1.328× | 1.315× |
| occipital→insula | 1.49e-02 | 1.29e-02 | 0.866× | 0.825× |

**The occipital→precentral row is the finding.** Under the tract topology the
minimum hop distance from occipital to precentral is **2, not 1**: the consensus
connectome declares no direct occipito-precentral fascicle, which is
anatomically correct. The random graph gave every occipital site a one-hop shot
at motor cortex, and a quarter of the transport `ablate_disjoint_transport.py`
has been scoring came down an edge the brain does not have.

**Robust to the consensus threshold**, which is the sweep the card asks for
because "the network's existence depends on a chosen frequency cutoff":

| threshold | occip→precen | occip→tempo | postcen→precen | occip→insula |
|---|---|---|---|---|
| 0.10 (616 edges) | 0.248× | 1.400× | 1.153× | 0.892× |
| 0.25 (536) | 0.266× | 1.457× | 1.205× | 0.897× |
| 0.50 (459) | 0.257× | 1.571× | 1.328× | 0.866× |
| 0.75 (384) | 0.254× | 1.611× | 1.349× | 0.830× |

Nothing here turns on where the cutoff is put.

Concentration alone buys 0.996–1.035× here, **not** the 209× measured before.
That is not a contradiction: 209× was on a *trained* kernel, where top-m ranks
edges by |learned weight| and redistributing the row's mass onto the strongest
raises a chosen path's gain. With equal priors and no learned factor there is no
ranking to exploit. Two different quantities, and the structural one isolates
topology.

### what the connectome does not reconstruct

`tract.py` names two cases as what the tractometric metric exists to get right.
At the 0.5 consensus threshold this connectome reconstructs **neither**:

- **the corpus callosum.** 17 of 459 edges are interhemispheric (3.7%).
  Homotopic superior temporal appears in **0 of 1064** subjects, homotopic
  postcentral in 0, homotopic precentral in 3.3%, homotopic insula in 5.5%.
- **the arcuate.** Left parsopercularis to superior temporal, its canonical
  terminal pair, in **7.0%**.

That is a known tractography failure rather than a fact about brains, and it
means the tract arm is a *ventral, intrahemispheric* topology whatever its
docstring hopes for. Reporting it as "the declared topology" without this
paragraph would be ledger row 9 in a new costume. The threshold is a knob and the
sweep is `edge_existence_curve()`: 616 edges at 0.1, 459 at 0.5, 126 at 1.0.

And **eleven parcels are isolated outright** at 0.5 — `lh/rh.entorhinal`,
`medialorbitofrontal`, `parsorbitalis`, `frontalpole`, `temporalpole` and
`lh.lateralorbitofrontal`, 6.4% of the sheet by area. Every one is
orbitofrontal, polar or entorhinal; every one sits against a sinus and loses
diffusion signal to susceptibility. Their sites spend the long-range budget
inside their own parcel, and the count and the names are printed in
`tract_note`, because "no long-range connection" here is a fact about the scan.

### the region the sphere could not address is the connectome's most connected one

Lobe-to-lobe declared connectivity at the 0.5 consensus, as a fraction of
possible parcel pairs (a 0.50 between two two-parcel lobes means "ipsilateral
only, both hemispheres", which is the 3.7%-interhemispheric result again):

                 occip  tempo  parie  front  precen postce insula cingul
    occipital     .38    .32    .44    .00    .00    .00    .31    .12
    temporal      .32    .24    .36    .00    .00    .00    .33    .12
    parietal      .44    .36    .38    .22    .50    .50    .50    .34
    frontal       .00    .00    .22    .14    .30    .28    .33    .27
    precentral    .00    .00    .50    .30    .00    .50    .50    .38
    postcentral   .00    .00    .50    .28    .50    .00    .50    .38
    insula        .31    .33    .50    .33    .50    .50    .00    .50
    cingulate     .12    .12    .34    .27    .38    .38    .50    .41

Occipital and temporal have **no declared connection at all** to frontal,
precentral or postcentral. That is the whole occipital→precentral result in one
cell, and it is correct anatomy: the ventral and dorsal streams reach motor
cortex through parietal and insula, not directly.

And **the insula is the single most connected parcel in the atlas** — degree 25
of a possible 67, ranks 1 and 3 of 68, tied with precuneus — with a declared
connection to every lobe including occipital and temporal. The region the
spherical proxy could not address at all, and into which the interoceptive drive
was therefore poured as a 4.5% subsample of `frontal`, is the hub.

The eleven parcels with degree **0** are the orbitofrontal, polar and entorhinal
ones, for the reason given above: they are where diffusion loses signal.

### a folded surface raises a question a sphere could not: is the local graph still local?

`knn_edges` is a `torch.cdist` in the volume. On a sphere that is a local sheet
by construction. On a folded white surface it need not be: two points on
opposite banks of a sulcus are millimetres apart in the volume and far apart
along the sheet, and `tract.py` opens by insisting the metrics disagree.

Measured (`scripts/measure_sheet_metric_error.py`) — true along-sheet distance
by Dijkstra over the fsaverage mesh, for the 36 local partners of 60 sampled
sites at 30,000 sites, mean euclidean edge 4.30 mm:

    geodesic / euclidean   p50 1.13x   p75 1.27x   p90 1.51x   p95 1.80x   p99 3.71x
    mean 1.28x     >2x: 3.4%     >4x: 0.8%     cross-hemisphere: 0 of 2,160

So the euclidean k-NN is still a local topology: 96.6% of local edges are within
2× of the true along-sheet distance, and no local edge crosses the midline. The
3.4% that are not are the genuinely cross-sulcal ones, and they are worth
knowing about rather than being a reason to change the metric.

**The gate is why that number is trustworthy.** Mesh-adjacent vertices have
geodesic exactly equal to euclidean, so the ratio must print 1.0000. The first
version printed **2.0000** for every adjacent pair — each undirected mesh edge is
in two triangles and `csr_matrix` *sums* duplicate entries, so every weight was
doubled. It would have been reported as "the median local edge is 2.3× longer
along the sheet than through the volume", a different and false conclusion, and
it was caught in one line by a case whose answer was known.

### and the long-range metric: is an arc length recoverable from a straight line?

`tract.py`'s central claim, tested on the payload rather than quoted from it.
459 declared edges at the 0.5 consensus, `fiber_length_mean` against two
euclidean surrogates:

    arc / MINIMUM border-to-border distance   p25 1.76x   p50 2.41x   p75 35.42x
    (that quantity is a genuine lower bound on any path, so it is also a GATE --
     100% of edges pass, i.e. no edge reports a fascicle shorter than the
     shortest line its own endpoints admit)

    arc / centroid-to-centroid chord   mean 0.73x, below 1 for 90% of pairs
    spearman(arc, centroid chord)      0.761

The second number is reported and then explained rather than left to look like a
result: streamlines terminate where two parcels FACE each other, not at their
centroids, so the centroid chord systematically overstates the endpoint
separation. It is a bias in the comparison, not a fibre taking a shortcut.

Against the unbiased lower bound the claim holds: the median declared fascicle is
**2.4× the shortest straight line** between its parcels, and the upper quartile is
35×. The one case `tract.py` names by hand — left parsopercularis to superior
temporal, the arcuate — is 60.2 mm of arc against a 36.8 mm centroid chord,
1.64×, in the direction claimed. It also appears in 7.0% of subjects, so the
connectome barely has it.

### a sidecar that asserted the wrong carrier

`ckpt/visual_contrastive_v2.json` records `"geometry": "fsaverage-sampled sheet,
THINGS-EEG2 64ch montage"` for a run that was a spherical shell. Nothing reads
that field, which is why it survived. `geometry_note(dyn)` now reads the sheet
off `dyn.pos` through the same discriminator the region lookup uses.

## 2026-09-09 — transport accounts for most of the end-to-end gap, but not all
##   (this entry's original title said "WAS the binding constraint"; the completed
##    run's own pre-registered verdict says otherwise, and the correction is below)

The pre-registered test landed on its confirming branch. Image -> occipital -> the
sheet -> a PRECENTRAL-ONLY readout, contrastive against measured EEG, trained END
TO END so encoder, kernel and head all receive gradient through the dynamics.
Identical batch, identical loss, initialisation reseeded per arm; the kernel's
edge configuration is the only difference. Ports asserted disjoint.

At matched steps (`out/disjoint_end_to_end.json`):

| step | base | aniso4d |
|---|---|---|
| 250 | 2.6x | 6.0x |
| 500 | 2.4x | 10.9x |
| 750 | 2.6x | 12.2x |
| 1000 | 4.0x | 13.9x |
| 1250 | 2.7x | **19.9x** |

Base ran the full 6,000 steps, peaked at 5.2x, ended at 3.2x, and was flat from
step 250 onward. The concentrated kernel is at 19.9x by step 1,250 and still
climbing — **7.4x better at matched step**, on a run a fifth as long.

Three falsification conditions were written into the script before it started;
this is the one that says transport was the binding constraint on end-to-end
training. Recorded that way rather than reinterpreted after the fact, because two
claims withdrawn today were ones where the reading was chosen once the number was
visible.

**AND IT CORRECTS THE MECHANISM I GAVE FOR IT.** I said end-to-end training was
starved of gradient, from a 175x drop in encoder gradient crossing to a disjoint
region. That measurement was taken at INITIALISATION and does not survive: base's
encoder gradient rises to 1e+00-7.6e+00 during training, thousands of times its
starting 5.45e-05, and base still does not learn. There is plenty of gradient.

So the limiter is not gradient MAGNITUDE, it is that the gradient is
uninformative — with ~1e-3 of the signal arriving at the readout, the error
signal reaching the encoder carries almost nothing about which encoder change
would help. Concentration fixes the forward signal, and the backward pass becomes
useful as a consequence. The at-initialisation ratio (base 5.45e-05, aniso
1.12e-03, 20x — matching the 21x measured separately) predicted the right
ORDERING for the wrong reason.

**Updated at step 5,000.** The concentrated arm reaches **38.5x chance** and has
plateaued there since ~4,250 (38.4 / 39.9 / 37.9 / 38.5). Against base's best of
5.2x and final of 3.2x, that is a **7.4x improvement in the best case and 12x at
the end**.

The number that matters more: fitting a head on FROZEN features of the same
sheet reaches 48.4x. End-to-end training through the unmodified sheet reached
3.2x — a 15x shortfall against its own frozen-head ceiling, which is what made
end-to-end look structurally broken. With the long-range budget concentrated,
end-to-end reaches 38.5x, or **80% of the frozen-head ceiling**. Most of that gap
was the sheet not conducting, and it closes when the sheet conducts.

**COMPLETED, AND THE PRE-REGISTERED VERDICT FIRED AGAINST MY READING.** Final,
6,000 steps both arms:

    base      best  5.2x   final 3.2x   mean of last 8 evals 3.5x
    aniso4d   best 43.9x   final 41.6x  mean of last 8 evals 40.2x

The script's own condition was `aniso > 2x AND base < 2x` for "transport was the
binding constraint". Base reaches 5.2x, so it fired **"BOTH train — the earlier
end-to-end failures were NOT explained by transport"**.

I reported the confirming branch an hour before the run finished, on the
matched-step comparison at 1,250 steps. That was premature, and the threshold I
wrote was badly chosen: 2x chance cannot separate "learns a little" from
"learns", and base at 5.2x clears it while being 8.4x behind the other arm.

The honest quantitative statement, which neither branch of my own binary
captures:

- base is **not at chance** — transport is not a total blocker, and any claim
  that the unmodified sheet cannot train end to end is wrong
- base is flat from step 250 onward and ends BELOW its best, so what it learns,
  it learns almost immediately and then stops
- aniso reaches **43.9x against a frozen-head ceiling of 48.4x — 91% of it**
- the gap between the arms is **8.4x**, and it is the concentration that produces
  it

So: transport accounts for most of the end-to-end shortfall and not all of it.
The residual — base's 5.2x, and the 9% aniso still leaves on the table — is
something else and is not identified here.

Recording the disagreement rather than re-drawing the threshold, because
re-drawing a pre-registered line after seeing the numbers is precisely what
pre-registering it was meant to prevent.

## 2026-09-09 (morning) — the video model has no temporal prediction beyond appearance

The retrieval task, run at the window where appearance is worthless. Chance is
1.5625%; the raw-pixel baseline sits at 0.78–1.56%, i.e. on chance, so the
appearance route is closed by construction.

The model never left chance. Training loss across 3,000 steps:

    4.1589  4.1589  4.1589  4.1589  4.1447  4.0680
    3.9109  4.0012  4.1590  3.8565  4.1282  4.1544

ln(64) = 4.1589 is the chance loss for a 64-way InfoNCE. The run oscillates
around it and ends on it. Held-out top-1 finishes at 2.15% ± 1.09 against a pixel
baseline of 0.78–1.56% — inside the noise. Stopped at step 3,000.

**The contrast is what makes it a result.** At the ±600-frame window, where a
third of the task is solvable by appearance, the same model went 1.56% → 6.25%
within 250 steps. At ±45 frames, where every candidate comes from the same 3.6
seconds and only motion phase separates them, it learns nothing at all.

So the video term was never learning to predict; it was learning to match
appearance, and the wide-window number measured that. This is consistent with
every other video result here — regression collapsing to the conditional mean,
the horizon-8 skill sitting a hair from persistence — and it is the cleanest
statement of the failure: the model has no representation of what happens next
that is independent of what things look like now.

Not a claim that the task is unlearnable, only that this architecture at this
budget does not learn it.

## 2026-09-09 — video regression collapses to the mean, and the retrieval task has a window

Every video run in this programme has failed the same way, and the failure is
not a bug. Under MSE against frame t+H the loss-optimal output for an
unpredictable target is its CONDITIONAL MEAN, and for natural film the mean
change over 320 ms is ~0. The 37.7 h run reached skill +0.0009 with its residual
magnitude SHRINKING 0.021 → 0.004 over 8,750 steps and its own diagnostic
printing `[degenerate: emits ~nothing]`; the via-EEG route finished at −0.9150.
More training cannot fix either — zero IS the optimum being sought. Both stopped.

The repo had already solved this shape once and written it down, in
`control_speech_to_meg.py`: "waveform regression peaked at skill +0.011 while
contrastive retrieval reached 42x chance on the same pairs." So the video term
becomes retrieval, where a constant output scores chance and the degenerate
solution is worth nothing.

**But the retrieval task has a free parameter that decides what it measures**,
and it was measured before spending the GPU. Retrieving frame t+8 from a pool of
64 by raw pixel cosine, no model at all, against how far apart the candidates
are drawn (`out/video_pixel_baseline_by_window.json`):

| window | pixel top-1 |
|---|---|
| ±1.2 s | 1.37% ± 0.52 |
| ±2.4 s | 1.76% ± 0.52 |
| ±4.8 s | 1.56% ± 0.00 |
| ±12 s | 6.64% ± 2.31 |
| ±24 s | 16.99% ± 3.17 |
| ±48 s | 33.98% ± 2.44 |
| ±160 s | 63.87% ± 6.12 |

Chance is 1.5625%. At the ±600-frame window the run was first launched with, a
THIRD of the task is solvable by appearance alone. At ±45 frames every candidate
comes from the same 3.6 seconds, appearance is worth nothing, and the pixel
baseline sits ON chance — so any above-chance retrieval there IS temporal
discrimination. Relaunched at window 45.

The pixel baseline stays in every evaluation regardless. It is what turns "4x
chance" from a claim into a comparison, and quoting ×chance alone would have let
a 10% result look like a 6x win while being three times worse than doing nothing.

## 2026-09-09 — interoception at 30,000 sites: the cortex ordering flips, and stays inside noise

The 8,000-site ablation was run under load 92 with three other agents on the GPU
and the agent that produced it asked for a full-scale repeat before anything was
quoted. Here it is, `--sites 30000`, and the cortical ordering **reverses**:

| arm | 8k | 30k |
|---|---:|---:|
| cortex trained | +0.5463 | **+0.5956** |
| cortex severed post hoc | +0.5520 | +0.5774 |
| cortex permuted post hoc | +0.5703 | +0.5819 |
| ridge, no dynamics | +0.4956 | +0.4956 |
| MLP matched samples | +0.5570 | +0.5368 |

At 8k, severing the kernel made it BETTER (+0.5520 against +0.5463) and I
reported that. At 30k, trained is best (+0.5956 against a severed +0.5774 and a
permuted +0.5819).

**Both readings are inside the noise band and neither is a result.** The band was
measured, not assumed, from four splanchnic single-group drops that should be
equivalent: at 30k they span +0.5946 to +0.6486, so nothing under about ±0.05 is
distinguishable at one seed. The trained-minus-severed gap is 0.018 at 30k and
−0.006 at 8k. Both are noise, and the fact that the SIGN flipped with site count
is the cleanest possible demonstration of that.

What survives unchanged, and is far outside the band:

| withheld | delay | state | cost |
|---|---|---:|---:|
| vagus C (5 ch) | 507.8 ms | +0.1198 | **−0.476** |
| all C fibres | — | +0.1734 | **−0.422** |
| vagus A-beta (4 ch) | 9.2 ms | +0.3790 | **−0.217** |
| vagus A-delta | — | +0.6299 | +0.034 |
| each splanchnic group | 128–232 ms | +0.595…+0.649 | within noise |

The half-second-late unmyelinated arm carries two thirds of the signal, and
dropping every C fibre in the body costs almost as much as dropping the vagal
ones alone. The peripheral anatomy is load-bearing; the cortical substrate is
not, at either resolution.

The `ridge_no_dynamics` arm reproduces to four decimal places across a 3.75x
change in site count, as it must — it is closed-form and site-independent. That
is the gate saying the two runs are comparable at all.

## 2026-09-09 (evening) — the viscera reach cortex, and two priors were wrong

The brain had vision, audition and a somatic strip and was blind to its own gut.
`ibm/embodiment.py` declared two visceral wires — blood pressure and oxygenation,
both at a flat 60 ms — against 483 somatic ones, and the vagus and the five
splanchnic trunks sat in `ibm/topologies/nerve.py` with nothing routed through
them. IHM-1 had been simulating digestion, absorption, substrate stores and
exertion the whole time and nothing read that state as afference.

### 1. A join bug worth 158 ms, on exactly the latency that matters

`ihm_bridge.routes()` read `path_length_m` from `muscle_bindings` and
`receptor_patches` and never from `nerves[].path_length_m` — which is the field
IHM's own `route_contract` names, with `missing_length_policy: error; never
silently substitute a trunk length`. Every route with no muscle and no skin patch
fell through to my typed trunk table and reported `ibm_declared_trunk` for a
length the body had measured. **92 of 144 routes.** Every visceral route is one of
them, because the vagus innervates no muscle and carries no skin patch.

| trunk | typed here | IHM measured | C-fibre delay |
|---|---|---|---|
| vagus | 350 mm | **508 mm** | 350 ms → **508 ms** |
| greater splanchnic | 300 mm (default) | **168 mm** | 300 ms → **168 ms** |
| lesser splanchnic | 300 mm (default) | 217 mm | 300 → 217 ms |
| least splanchnic | 300 mm (default) | 232 mm | 300 → 232 ms |
| pelvic splanchnic | 300 mm (default) | 128 mm | 300 → 128 ms |

`visceral_routes()` now raises rather than falling back. The pattern is the
ledger's: a quantity computed correctly against the wrong source, and the source
that was right had said so in a field named `missing_length_policy: error`.

### 2. Fifteen channels, seven conduction groups, 55x

`ihm/assembly/interoception.py` transduces `NativeSession.snapshot()['values']`
— the 180 quantities BioGears integrates — into firing rates on the declared
trunks. Ten vagal, five splanchnic. Operating ranges are declared from physiology
and **not fitted to the corpus**, so a high-threshold channel is allowed to sit
silent through a protocol that never drives it, and two do.

The delays are the point:

| group | delay | what arrives |
|---|---|---|
| vagus / A-beta | **9.2 ms** | gastric and intestinal volume, lung volume, aortic pressure |
| vagus / A-delta | 33.9 ms | intestinal absorption |
| pelvic splanchnic / C | 128.0 ms | bladder |
| greater splanchnic / C | 168.4 ms | high-threshold gastric distension, lactate |
| lesser splanchnic / C | 217.2 ms | high-threshold intestinal distension |
| least splanchnic / C | 232.2 ms | renal filtration |
| vagus / C | **507.8 ms** | nutrient load, portal glucose, aortic chemoreception |

**The stomach reports twice, on two nerves, half a second apart** — a vagal
low-threshold volume report at 9 ms and a splanchnic high-threshold nociceptive
report of the same organ at 168 ms, from the same source keys with different
thresholds. That pair is the thing the fibre-class axis exists for, and a model
given one visceral latency asserts they are the same event. The ratio, 55x within
one nerve, is the largest in the model.

### 3. The afference carries a real signal. The cortex adds nothing to it.

`scripts/ablate_interoception.py`, 8,000 sites, 300 steps at batch 32, seed
matched across arms. The corpus is 5,047 frames of recorded BioGears physiology
under five protocols; the split holds out **whole protocols** —
`meal_exercise` and the six-hour `hydration` run — not a contiguous tail, for a
reason in §5 below.

Two questions, two baselines, because one aggregate over both would be a lie:

- **state** — `endurance_h`, hours of carbohydrate substrate at the current
  metabolic rate. Liver glycogen, muscle glycogen and metabolic rate are **not
  afferent channels**, so this must be inferred from gut and cardiorespiratory
  traffic. Baseline: the training mean.
- **trajectory** — `discomfort` and three `sensation` components at a 60 s
  horizon. These *are* exact functions of the afferent vector — a sanity gate
  proves it by making a ridge print R² = 1.000000 — so scoring them against the
  mean would measure a model inverting its own input. Baseline: persistence.

| arm | state | trajectory |
|---|---|---|
| ridge, no dynamics | +0.4956 | −0.0094 |
| MLP to convergence, no dynamics | −0.2225 | −4.7140 |
| **MLP at matched samples, no dynamics** | **+0.5570** | −1.5009 |
| **cortex, trained** | **+0.5463** | −2.7761 |
| cortex, kernel severed post hoc | +0.5520 | −2.7861 |
| cortex, kernel permuted post hoc | +0.5519 | −2.7960 |
| cortex, permuted kernel **retrained** | **+0.5703** | −4.9982 |

**Severing the association kernel costs −0.0058 — it is very slightly better
severed.** A permuted kernel retrained from the same seed reaches +0.5703, above
the trained one. A dynamics-free MLP on the same fifteen numbers at matched
sample count reaches +0.5570, also above it. **This is ledger entry 20
reproduced on a second, independent pathway**, and the honest report is the
*afference's* skill and not the loop's: visceral afference predicts unobserved
substrate state at skill ≈ +0.50 to +0.57 against the training mean on two
held-out protocols, and it does so through a matrix.

Nothing beats persistence on the trajectory question. The ridge ties it
(−0.0094); every cortical arm is far worse. Predicting where visceral state will
be in 60 s, from where the afference is now, is not something anything here does.

### 4. The anatomy IS load-bearing — the slow arm, and the delay itself

The ablation that does return a result. Conduction groups withheld post hoc from
the trained model, on **state**:

| group withheld | channels | delay | state | cost |
|---|---|---|---|---|
| — (intact) | — | — | +0.5463 | — |
| lesser splanchnic / C | 1 | 217.2 ms | +0.5839 | +0.038 |
| pelvic splanchnic / C | 1 | 128.0 ms | +0.5570 | +0.011 |
| vagus / A-delta | 1 | 33.9 ms | +0.5518 | +0.006 |
| least splanchnic / C | 1 | 232.2 ms | +0.5508 | +0.004 |
| greater splanchnic / C | 2 | 168.4 ms | +0.5473 | +0.001 |
| **vagus / A-beta** | 4 | 9.2 ms | +0.4246 | **−0.122** |
| **vagus / C** | 5 | 507.8 ms | +0.1687 | **−0.378** |
| all myelinated | 5 | 9–34 ms | +0.4396 | −0.107 |
| all C | 9 | 128–508 ms | +0.2293 | −0.317 |

**The noise band is measured, not assumed.** The four splanchnic single-group
drops span +0.547 to +0.584 against an intact +0.546, so at one seed nothing
under about ±0.04 is distinguishable. Severing the kernel (−0.006) and permuting
it (−0.006) are inside that band. The two vagal groups are not.

**The slow unmyelinated vagal arm carries two thirds of the signal.** Those five
channels — nutrient load, hepatoportal glucose, aortic chemoreception — arrive
507.8 ms after the event, and severing them costs −0.378. The fast myelinated
arm that reports gastric and intestinal volume, lung volume and aortic pressure
in 9.2 ms costs −0.122, three times less. **The half-second-late chemical report
is what tells this model how much fuel is left; the nine-millisecond mechanical
report largely does not.** Post-hoc drops are not additive and their mutual
ordering should not be over-read (dropping the vagal C group alone costs more
than dropping all nine C channels), but the separation between the vagal groups
and everything else is far outside the band.

**And the delay is load-bearing, not only the channels.** A drop arm removes what
a group *carries* and says nothing about *when* it arrives, so it cannot test the
thing `ibm/topologies/nerve.py` was written to assert. The arm that can:
**`cortex_lumped_delays`** keeps all fifteen channels and lands every group on
step 0, seed- and step-matched to the trained arm.

| arm | state |
|---|---|
| cortex, trained (9.2 → 507.8 ms preserved) | +0.5463 |
| **cortex, every group lumped onto step 0** | **+0.4221** |

**Lumping costs −0.124** — as much as deleting the entire A-beta group, with
every channel still present. Three times the noise band, five times the
seed-to-seed spread the retrained permuted arm implies. The fibre-class-resolved
delay is doing work.

That is the first result in this repo where the *peripheral anatomy* is
load-bearing while the *cortical substrate* is not, and the two ablations sit
side by side in the same table.

### 5. Two corrections I made to my own instruments

**A split that made a correct model score −2.07.** `endurance_h` drifts over the
scale of a whole run, so a contiguous time split leaves the held-out window in a
1.6 h band at the top of a 27 h range. A model accurate to 1.1 h absolute — 2.8%
of the quantity — scores skill **−2.07** against the training mean there. Both
numbers are correct; reporting either alone is the wrong-population error. The
default split now holds out whole protocols, and the held-out endurance range is
13.94–40.75 h against the training set's 13.89–40.75.

**A gate that encoded my guesses instead of testing routing.** The live causal
contrast (`scripts/embody_interoception.py`) started with a specificity *ratio*,
and two pre-registered expectations failed: a meal moved pulmonary stretch 3.5 Hz
(the meal delivers 500 mL of water and lung volume follows blood volume), and
exercise moved renal filtration 8.3 Hz (cardiac output drives GFR). Both are the
engine's physiology, neither is cross-talk in the channel table. The instrument
was the deeper error: a ratio threshold is a guess about coupling *magnitude*,
which is exactly what a run should be free to tell me, and tuning it until it
passed would be fitting the gate to the data. It is now two claims a contrast can
actually test — exercise must leave every GI-lumen channel at **exactly** zero
(seven channels; they were, on the first run), and the largest mover must be in
the family the intervention drives. A rank claim no threshold can buy.

Measured, as max |Δ| in Hz against a matched rest control from one stabilized
state: a meal moves gastric distension **+30.87** and renal filtration **+0.014**;
exercise moves pulmonary stretch **+16.18** and every GI-lumen channel by
**exactly 0.000**; endurance falls 37.9 → 13.9 h as metabolic rate goes 92 → 241 W.

### 6. What is named and what is not claimed

`endurance`, `discomfort` and `sensation` are **named projections of measured
physiological state**. Endurance is a ratio of two engine outputs; discomfort is
a fibre-class-weighted sum of the afferent rates normalised by its own saturated
ceiling; sensation is a linear projection of the same vector with a basis fitted
on the training split. None is a claim about anything felt, and the names are a
liability precisely because they read as if they were. The module, the corpus
metadata and the docstrings all say so where the names are defined.

Two gaps stay open and are recorded rather than fixed. There is **no
viscera-supported nociceptor component** — `transduction.nociceptor` sits on the
transduction field's default support, the body surface — so the splanchnic
channels bind to `transduction.baroreceptor` and are tagged nociceptive in the
join table. And the loop is **one-way**: the same trunks declare
`b_preganglionic` and `c_postganglionic` and nothing drives them, so there is
visceral afference and no autonomic outflow.

The **insula is not separable** on the six-label spherical proxy
`cortical_regions` provides; a sphere has no lateral sulcus. The drive enters the
`frontal` label, subsampled to the insula+ACC share of cortical surface (4.5% of
cortex, 20% of that label). `ibm.interoception.PORT_SUBSTITUTION` is a constant
so it prints in every report rather than living in a comment.

---

## 2026-09-09 (afternoon) -- the sheet was never information-blocked, it was amplitude-blocked

Three results, and the largest one is not about the fix.

### 1. The dynamics are necessary; what they LEARNED is not

`scripts/ablate_disjoint_transport.py`. Real THINGS-EEG2: an image drives the
**occipital region**, the sheet runs, the state is read from the **precentral
region only** (the script refuses to run if the ports intersect -- verified
disjoint, 3,775 drive sites against 2,012 read sites), and a small head is fitted
on the precentral rates against the measured EEG. The image encoder and
`to_cortex` are the checkpoint's own and are FROZEN, so every arm sees the same
stimulus-specific drive and the arms differ only in what the sheet does with it.
8 disjoint held-out pools of 200, chance 0.5%, never a single pool. 20,000-step
fits.

On the **unmodified** trained kernel, no noise:

| arm | top-1 | vs chance | effective rank | train loss | train top-1 |
|---|---|---|---|---|---|
| intact | 24.19% +/- 1.62 | 48.37x | 12.9 | 0.0096 | 99.7% |
| relabel (control) | 24.31% +/- 4.08 | 48.62x | 12.9 | 0.0148 | 99.1% |
| permuted rows | 25.37% +/- 2.96 | **50.75x** | 12.2 | 0.0113 | 99.7% |
| severed (w = 0) | 0.50% | 1.00x | **1.0** | -- | -- |

chance loss is 6.9315; severed across-image sd is exactly 0.000e+00 Hz.

**`relabel` is the gate**, and it passes: the kernel is untouched and only the
readout COLUMNS are permuted, with the same permutation in both passes, so it is
a pure relabelling of the head's input coordinates and must score what `intact`
scores. It does. The other arms are therefore interpretable.

Two things follow, and they point opposite ways.

**The dynamics ARE necessary.** Severing gives a constant readout -- across-image
sd exactly zero, effective rank 1.0 -- and retrieval at chance. This log has said
"there is no configuration measured so far in which the cortical dynamics both
receive the sensory signal and are necessary to produce the output." That
sentence is false, and what made it false was changing the READOUT to a disjoint
region and fitting a head on frozen features, not changing the kernel, the
objective or the data.

**What they LEARNED is not.** A kernel with its site rows permuted scores 50.75x
against the trained kernel's 48.37x -- indistinguishable, with amplitude, rank,
train loss and train accuracy all preserved. Transport is a property of the graph
and the weight statistics, not of the learned content. **Ledger row 20's
generalisation is confirmed, not overturned**, and now on a perceptual task with
a disjoint readout and a passing positive control.

I first reported the opposite, with permuted at exact chance. That was a
train/test permutation mismatch -- `torch.randperm` was drawn inside
`cortical_features`, from a generator shared with the caller, and that function
is called twice per arm, so the head was fitted on one permutation and evaluated
on another. It preserved every statistic and destroyed the mapping, which is
exactly what a real ablation looks like. It was caught by asking why permuted sat
at EXACTLY chance while preserving 99.7% of the across-image variance. The fix
draws one permutation per arm in the caller, and `cortical_features` now returns
a fingerprint of the treatment it applied so the caller RAISES if the two passes
disagree rather than printing two lines for a human to compare.

**The number worth keeping** is the effective rank: 12.9 of a possible 2012.
Whatever crosses the sheet is 13-dimensional, in every arm that conducts at all.
That is a tighter constraint on the somato-motor design than any of the amplitude
numbers.

One caveat: this is a head fitted on frozen features, not end-to-end training.
The hypothesis was that gradients through a 1e-3 attenuation are as small as the
signal, so the earlier failures were an optimisation problem downstream of the
transport problem.

**NOW TESTED, and it holds.** `scripts/measure_encoder_gradients.py`, identical
batch and loss with the initialisation reseeded immediately before each arm:

  config    |grad| at encoder   forward sd   at-port grad   severed grad
  base           1.8831e-03     1.8384e-03     3.292e-01      0.000e+00
  aniso4d        4.0354e-02     5.5900e-02     3.033e-01      0.000e+00

  gradient ratio 21.43x     forward ratio 30.41x

The gradient reaching the encoder scales with transport -- 21x against 30x, the
same order -- so the backward pass is attenuated by about the same factor as the
forward pass. Read the base row across rather than down: the encoder sees
3.292e-01 of gradient when the readout sits AT the driven port and 1.883e-03
when it sits in a disjoint region. **175x of the gradient is lost crossing the
sheet.** The concentrated kernel loses 7.5x.

That is a mechanism for the whole history of end-to-end motor failures here.
Training was not starved of data or wrongly objectived; it was starved of
gradient, by the same attenuation the forward signal suffers. A refit head
succeeds because it never sends gradient back across the sheet at all.

Three gates. Severed gives EXACTLY 0.000e+00 encoder gradient in both arms, so
nothing reaches a disjoint readout except through the sheet. A readout taken AT
the driven region gives a large gradient in both arms, which separates "the
sheet does not transmit gradient" from "the loss is flat" -- identical encoder
norms, completely different meanings; the loss is not flat. And the forward and
gradient ratios had to match, which would have falsified simple attenuation had
the gradient outrun transport.

STILL UNTESTED, and the distinction matters: this says the gradient is THERE to
train on, 21x more of it. It does not show that the concentrated kernel actually
trains end to end. Only a run does that.

**CAVEAT ADDED THE SAME DAY, AND IT MAY UNDO THE ABOVE.** Every number in this
section was measured with FRESHLY INITIALISED heads. A random head backpropagates
a random error, and that is not the gradient a training run actually sees. The
end-to-end run launched to test the prediction reports its own encoder gradient
each evaluation, and on the BASE arm — the one predicted to stay starved — it
goes 5.453e-05 at step 0 to 2.248e-01 at step 250, a factor of ~4,000, while
top-1 rises 0.50% to 1.31% (2.6x chance).

If the base arm keeps rising, then transport is NOT the binding constraint on
end-to-end training, the 175x is an artifact of measuring at initialisation, and
the mechanism offered above for every past motor failure does not hold. The
falsification conditions were written into the run before it started
(`scripts/train_disjoint_end_to_end.py`) precisely so this could not be
reinterpreted once the numbers were visible. Treat the 175x as an
AT-INITIALISATION ratio until that run lands.

### 2. The anisotropy fix buys noise tolerance and nothing else

Concentrating the long-range budget raises the across-image sd of the precentral
rate from 5.406e-02 Hz to 1.915e+00 Hz, a **35.4x** amplitude gain. On the task
that converts to exactly one thing:

| noise (Hz) | unmodified | concentrated | severed |
|---|---|---|---|
| 0 | 43.62x | 42.37x | 1.00x |
| 1e-2 | 21.12x | 37.87x | 0.50x |
| 3e-2 | 17.75x | 43.75x | 0.50x |
| 1e-1 | **11.75x** | **26.50x** | 0.50x |
| 3e-1 | 7.87x | 19.00x | 0.50x |
| 1 | 2.00x | 11.87x | 0.50x |
| 3 | 0.75x | 6.87x | 0.50x |

(the permuted column is withdrawn, see above.)

**Do not lead with the zero-noise row.** Base across-image sd is 5.406e-02 Hz,
about 10^6 times float32 epsilon, so a linear head has enormous headroom to
amplify and 43.62x is a statement about the simulation's precision as much as
about the sheet. The defensible version of "the dynamics both receive the signal
and are necessary" is the physiological band: **11.75x to 17.75x against a
severed 0.50x, at 0.03-0.15 Hz of noise.**

At zero noise the fix is worth **nothing** (43.62x against 42.37x). The noise at
which each arm falls to half its noiseless skill is 9.31e-03 Hz unmodified and
2.18e-01 Hz concentrated: a **23.4x** noise-tolerance gain from a 35.4x
amplitude gain. That is the whole result, and it was predicted in advance -- in
a noiseless float32 simulation a linear head can amplify a 1e-5 Hz perturbation
and retrieve perfectly, so the noiseless column could only ever have been
uninformative.

An order-of-magnitude for where that matters: Poisson spiking at 8 Hz over a
40 ms window is ~14 Hz sd per neuron, so a site pooling 10^4-10^5 neurons sits
near 0.03-0.15 Hz. In that band the fix is worth 2x to 2.5x on retrieval. That
estimate is an argument, not a measurement.

### 3. Concentration buys integration, at matched amplitude

Superadditivity against the linear-sum null, **both arms at matched drive**,
because a sweep of one arm alone is the shape of ledger row 20:

| amp | unmodified transport | concentrated transport | unmodified superadd | concentrated superadd |
|---|---|---|---|---|
| 1 | 9.561e-06 | 1.998e-03 | 1.0002 | 1.0021 |
| 10 | 1.012e-05 | 3.452e-03 | 1.0139 | 1.1320 |
| 30 | 1.700e-05 | 8.080e-03 | 1.0377 | 1.2987 |
| 100 | -- | 1.291e-02 | -- | 1.3923 |

Rest is -65 mV and threshold -55 mV, so **amp 10 is the physiological point**
and amp 30 is supraphysiological. The number to quote is 1.132 against 1.014 at
matched amplitude, with transport 3.452e-03 against 1.012e-05 -- a factor of
341. The earlier 1.0002 was reporting the drive amplitude, not the sheet: a
0.1 mV perturbation against a sigmoid curvature scale of 4 mV gives a
second-order term of (0.1/4)^2 = 6e-4, which is what was measured.

The deflationary reading -- "the sheet always could integrate, it was never
driven hard enough" -- is refuted by the matched control: driven just as hard,
the unmodified kernel reaches 1.014, not 1.13.

One confound to state whenever the amplitude sweep is quoted: unmodified
transport rises 1.8x on its own from amp 1 to 30. That is the sigmoid's slope
improving under a larger drive, which raises the LINEAR gain of every hop, so
within-arm amplitude comparisons are not clean. Only the matched-amplitude
comparison is.

### 4. Transport is UNIMODAL in the operating point, not monotone

A tonic drive added uniformly, postcentral -> precentral at steady state:

| tonic | resting Hz | arrival |
|---|---|---|
| 0 | 8.06 | 9.73e-03 |
| 2 | 13.04 | 1.55e-02 |
| 4 | 20.72 | 2.32e-02 |
| 8 | 45.26 | **2.97e-02** |
| 14 | 76.93 | 1.11e-02 |

Transport RISES 3x with tonic drive before it falls, peaking where the resting
rate is near `r_max`/2 = 50 Hz -- which is where `dr/dv` is maximal, and nothing
more. So "any drive that raises the resting rate degrades transport" is false;
the correct statement is that transport is unimodal in the operating point.

That retro-explains the near-critical arm exactly: sever ratio 1.254 at 42 Hz,
collapsing at 75 Hz, is a unimodal curve sampled on both sides of its peak. And
it explains why raising `w_assoc` fails where tonic drive does not -- gain does
not move the operating point smoothly, it jumps the fixed point from 8 Hz to
85 Hz, skipping the useful region entirely.

### 5. The whole transport gain is available at constant row L1

`long_topm=1` throughout, with `local_gain` and `long_gain` chosen so the row L1
-- and therefore the resting rate and the operating point -- is exactly unchanged
from the trained kernel:

| local | long | transport |
|---|---|---|
| 1.00 | 1.00 | 1.138e-04 |
| 0.75 | 1.75 | 3.446e-04 |
| 0.50 | 2.50 | 7.041e-04 |
| 0.25 | 3.25 | 1.194e-03 |
| 0.00 | 4.00 | 1.817e-03 |

16x from reallocation alone, 190x over the unmodified kernel's 9.561e-06, with
the operator's L1 norm identical at every row. **None of the transport gain
requires raising the gain, moving the operating point, or approaching
instability.** 36 local edges hold 75% of every row's budget and contribute
nothing to long-range transport.

The last row deletes the local sheet entirely, which would certainly cost the
local computations. The usable row is 0.50/2.50: half the local mass retained,
6x the transport, operating point untouched.

### What this does NOT say

None of it says the somato-motor materialization works. These are mechanism
measurements on random drives plus one retrieval task with a frozen encoder.
They say the sheet can carry and combine signals between disjoint regions and
that the learned kernel is necessary for it. They do not say a closed
sensorimotor loop can use that, and 1.132 is a modest nonlinearity.

**Operating point: `long_topm=4, long_min_dist=120, long_gain=8`.** The ablation
above was run at `long_topm=1`, the maximum-transport arm; the three-arm rerun at
the recommended point is queued.

## 2026-09-09 -- the gait can step or it can travel, not both

It does not walk. One genuine step, 1.88 cm of pelvis travel, then it pitches
forward past 0.55 rad and stops at 1.49 s of a 5.0 s horizon.

The step is real and it is muscle-driven: the right foot lifts at 0.81 s,
unloads to **0.14% of body weight** (fully airborne), clears 1.8 cm, advances
17.1 cm, and lands on measured contact at 1.42 s -- 98 muscle excitations in
[0,1] into the native mechanical stream, no prescribed motion, no external root
forces.

The strict step criterion exists because a loose one was gamed. An earlier
search reported "2 consecutive steps" in which the foot never dropped below 39%
load and rose 1.7 mm. Under the criterion actually used -- airborne below 2% body
weight, at least 10 mm clearance, at least 30 mm advance, landing on measured
contact -- those score **zero**, and weaker cycles are reported separately as
weight shifts rather than folded in.

**The binding constraint is the LQR stage cost, not the gait parameters.** The
gain is solved about an equilibrium whose forward velocity is exactly zero.
Adding stage cost on `pelvis_tx/speed` and changing nothing else took travel from
2.24 mm to **119 mm, a factor of 53**, and it is corroborated across twelve
parameter sets: six that could never move the pelvis more than 5 mm all travel
6-12 cm once given `q_tx_speed = 1500`, with none of their own parameters
changed.

And the trade is sharp enough to be the real finding: **at low forward cost it
steps and does not travel; at high forward cost it travels at 0.11 m/s and does
not step.** Every travelling arm falls forward. The controller can do either,
and nothing found so far does both.

A CORRECTION OF A CORRECTION, recorded because the error was mine and it went
the wrong way. I proposed the stage-cost hypothesis, labelled it untested, and
then **retracted it** on the evidence that search I reached 8.4x the travel of
search H while commanding six times LESS forward velocity -- which looked like
the setpoint mattering in the opposite direction. That comparison was
confounded: I is H **plus** the new `q_tx_speed` parameter, not H with a smaller
`v_forward`. The matched single-variable tests separate them cleanly -- setpoint
does almost nothing (travel *falls* 5.9 -> 3.9 mm as `v_forward` goes 0 to 0.60)
and stage cost does almost everything. The hypothesis stands; my retraction of
it does not. Ledger row 22 is unaffected -- that was the pelvis_tx POSITION
claim, which was separately and genuinely wrong (column L2 7.0e-10 deflated
against 4.14 as shipped; 12 cm of travel perturbs the command by 6.5e-11).

Against the repo's prior best, stated so the comparison is not flattering: the
existing half-step placed the foot 16.1 cm forward and moved 8.0 cm of COM in
71 s, **ending settled and stable**. This is 12.6 mm/s against its 1.1 mm/s and
has a genuinely airborne swing rather than a quasi-static one, but it travels
LESS total distance and it FALLS. On stability the older result is better.

Negatives worth keeping: the decisive lever for getting a foot off the ground at
all was `swing_relax` -- `u0` is a DOUBLE-SUPPORT equilibrium, so the swing leg's
own plantarflexors and vasti hold it at stance excitation and it keeps pressing
on the floor; no search produced a lifted foot until that baseline was scaled
down. Stance-limb baseline boost moves survival the wrong way. Muscle biases are
not the cause of divergence -- removing all of them still diverges, so it is
weakening the regulator that destabilises. And handing the plant back to the pure
stance regulator immediately after the step does not save it: it falls 230 ms
later instead, so the first step is already unrecoverable and the blocker is not
second-step initiation.

Untried and cheapest next: two parameters sit exactly on their bounds in the
delivered result (`a_stance_hipext` 0.500, `b_kneeext` 0.800), and pitch is now
the binding failure while the lumbar actuators carry no bias at all.

## 2026-09-09 (morning) -- the sheet superposes; it does not integrate

The somato-motor materialization needs sight, hearing and touch to converge
somewhere precentral can read. Measured on the 16-objective kernel at step
32,500, 30,000 sites, driving each modality at its declared entry port and
reading **precentral only** (`scripts/measure_multimodal_convergence.py` refuses
to run if any entry port intersects the readout):

| modality | entry | at own port | at precentral | transport |
|---|---|---|---|---|
| sight | occipital | 3.270 | 1.559e-05 | 4.77e-06 |
| hearing | temporal | 3.274 | 2.468e-05 | 7.54e-06 |
| touch | postcentral | 3.280 | 3.136e-05 | 9.56e-06 |

~100,000x attenuation. The ordering is the one anatomy predicts -- touch enters
the strip adjacent to the readout and transports twice as well as vision from
the far pole -- which is some evidence the measurement is doing what it claims.

**The stronger result is the second one.** Driving all three gives precentral
variance 7.170e-05 against a linear-sum-of-the-parts null of 7.169e-05: a ratio
of **1.0002**. The sheet is a pure superposition device at this operating point.
Whatever arrives does not interact, so there is no cross-modal computation
anywhere on it. Integration is not weak here, it is absent.

Both gates print their known answers: undriven precentral variance is exactly
0.000e+00, and reading at a driven port recovers 3.270.

**The cause, from two directions that agree.** Per-hop transfer is a uniform
~1/300 amplitude loss at every hop, with linear theory landing within 1% of the
measurement (per-edge one-hop gain 3.520e-03 predicted, 3.484e-03 measured). The
arriving signal decomposes as (row gain) x (fraction of the row's weight mass on
the source region) = 0.169 x (2.4/48) = 8.5e-3 against 1.08e-2 measured for a
coherent region drive. The weight mass is spread flat over 48 edges, so only 5%
of it lands where the signal comes from.

Topology was never the problem: postcentral -> precentral is **one** hop, and so
is 75% of the sheet from any 20% region.

Raising the gain is not the fix and is actively counterproductive -- scaling all
edges or `w_assoc` past ~2x drives the resting rate onto the sigmoid's upper rail
(8.06 Hz at scale 1, 84.96 at scale 4, pinned at 87.3 beyond), `dr/dv` collapses,
and arrival FALLS 400x by scale 32. That is the earlier near-critical collapse at
75 Hz, with the mechanism now visible: not chaos, the fixed point climbing to
saturation.

Concentration is the fix. Keeping one long-range edge per site with the row L1
preserved exactly buys **11.9x at zero change in row gain**; with a long-range
gain of 8 it reaches 1030x in variance (32x in amplitude). Superadditivity does
NOT move (1.0014), but that criterion was badly posed -- at the drive amplitude
used, the perturbation reaching precentral is ~0.1 mV against a sigmoid curvature
scale of 4 mV, so the nonlinearity cannot engage regardless of transport. The
honest joint target is stated in physiological units: a physiological drive at
the entry port must produce a millivolt-scale perturbation at the readout.

## 2026-09-09 (morning) -- the association graph was never in the artifact

12 of every site's 48 edges are long-range partners drawn by `torch.randint`
from the **global** RNG inside `CorticalDynamics.__init__`. The learned factor is
sigma(<e_i, e_j>) over those specific pairs, so a different draw makes every
long-range weight meaningless -- a quarter of the connectivity, discarded without
an error.

Measured, so the scope is not a guess: seed 0 on the **same device** reproduces
the draw exactly (coincidence 1.00000 against a chance of 0.000033 at 30,000
sites); seed 0 on cpu against cuda gives unrelated graphs (0.00062 against a
chance of 0.00050).

Most consumers were fine, by care or by luck: anything using `load_state_dict`
gets `idx` back as a registered buffer, and `ablate_body_kernel.py` seeds with 0
on the training device. **I suspected this had contaminated the permuted-kernel
motor result (ledger row 20) and tested it: it had not.** That finding stands as
measured.

Two places were wrong. `materialize.py` constructed with no seed at all, on
"cpu", against a kernel trained with seed 0 on cuda -- wrong twice over. And
`ckpt/ibm1_implicit.pt`, the published fused kernel, carries **no graph at all**,
so every materialization from the flagship artifact has been running a quarter of
its connectivity at random. The artifact now saves `dyn.idx`/`geo`/`pos`, and
`materialize.py` restores them or says loudly that it could not.

Still open, and recorded because it will bite an ablation: `embody.py`'s `--seed`
is documented as "matched initialization for ablations" and changing it redraws
the **topology** as well as the weights, so a seed sweep silently varies two
things.

## 2026-09-09 -- the TCT loop runs continuously; it is a relay with feedback, not a closed loop

`docs/DYNAMICS.md` §2 makes thalamo-cortico-thalamic recurrence the central
operator of the architecture, and §6 records that the running cortical model has
no thalamus in it.  `ibm/processes/tct.py` closes that gap by implementing
`burst_relay_rate` -- the one implementation `thalamocortical_coupling` declares
with `form=Form.RATE`, a full parameter block and no code -- and attaching it
inside `CorticalDynamics.step`, so the loop is present in every forward of every
head in `pretrain_video_loop.py` rather than at one call site, and keeps its own
state between forwards.

it had to be the burst form.  the other two implementations are transfer
functions: driven by nothing they output nothing, so a linear TCT loop bolted
onto a cortex with no stimulus is silent and "continuously active" would be a
claim with no mechanism.  the T-type calcium current is the mechanism -- it is
de-inactivated by hyperpolarization, so a relay cell TRN has just inhibited
rebounds with a burst, that burst re-excites TRN, and the pair is a relaxation
oscillator that needs no input.

**the measurement** (`scripts/measure_tct.py`, `out/tct_report.json`): 2048
sites, 16 thalamic units, 4 seeds, 12 s of autonomous dynamics per seed at
dt = 1e-4 s, **zero external input after t = 0**.  every row is the same
integration of the same equations on the same sheet with one gain zeroed.

| condition | cortical peak | ratio | mean-rate swing | thalamic peak |
|---|---|---|---|---|
| cortex_only (no thalamus) | -- | -- | **0.0000 Hz** | -- |
| **full loop** | **6.92 ± 0.00 Hz** | 1.1e12 | **39.49 Hz** | 6.92 Hz |
| sever thalamo-cortical | -- | -- | **0.0000 Hz** | 7.50 Hz |
| sever cortico-thalamic | 7.50 ± 0.00 Hz | 9.4e11 | 38.06 Hz | 7.50 Hz |
| sever TRN inhibition | -- | -- | **0.0000 Hz** | -- |
| sever T-type calcium | -- | -- | **0.0000 Hz** | -- |

the ablation delta is the whole result: **39.49 Hz of mean-rate swing to
0.0000**.  the cortex on its own, with no drive, is not a weak oscillator -- it
is a fixed point, flat to four decimals, so there is no baseline rhythm for the
thalamus to have merely amplified.

three checks the numbers had to survive first, because this log's ten withdrawn
claims are all a quantity compared against the wrong thing:

- **the estimator on cases whose answer is known.**  a 12 Hz sine returns 12.000
  Hz; a 12 Hz sine buried in 3x noise returns 12.000 Hz; a constant returns zero
  amplitude.  white noise of the same length returns a peak ratio of 9.9, and
  **that is where the threshold comes from** -- a peak must beat 10x it.  the
  criterion is a conjunction of ratio AND amplitude, which is what stops an
  argmax on a flat spectrum from reading as a rhythm: `cortex_only` has ratio
  `inf` (its median band power is exactly zero) and is correctly scored `no`
  because its amplitude is zero.
- **the ablation machinery.**  severing the ascending limb must leave the cortex
  in bitwise the state a cortex with no thalamus would be in.  measured
  `max |cortex_only - sever_tc| = 0.000e+00`.  it is asserted, and the script
  aborts if it fails.
- **the integrator.**  forward euler can put a frequency where the mechanism
  does not.  halving the timestep moves the peak by **+0.000 Hz (0.0%)** and the
  amplitude by 0.2%.

**it is not a closed loop, and the bifurcation says so more clearly than the
ablation does.**  cutting the cortico-thalamic limb leaves the rhythm standing.
sweeping the T-current conductance through onset with the descending limb intact
and cut (`out/tct_bifurcation.json`):

| g_T | 3.75 | 4.00 | 4.25 | 4.50 | 4.75 | 5.00 |
|---|---|---|---|---|---|---|
| descending limb intact | -- | 6.50 | 6.83 | 6.83 | 7.00 | 7.00 |
| descending limb **cut** | 6.67 | 7.17 | 7.33 | 7.50 | 7.50 | 7.67 |

the oscillation onsets **earlier** without cortex (3.75 vs 4.00), so cortical
feedback is mildly *suppressive* of onset rather than necessary for it.  the
oscillator is intrathalamic.  what the descending limb does do is shift the
frequency by **-0.50 to -0.67 Hz at every one of the five matched operating
points** -- a consistent ~8% slowing, not a wobble.  so the honest statement is:
signals go around the loop and the round trip changes the answer, but the
rhythm does not require cortex.  calling this "the cortex oscillating because of
a loop it is inside" would be exactly the overclaim this log exists to catch.

**the declaration's own two predictions both hold.**  `burst_relay_rate` says
`t_deinactivation_s` sets the recovery time and `trn_gain` "sets whether the
oscillator runs at spindle or at delta frequency".  over a 6x4 grid, frequency
falls monotonically with `t_deinactivation_s` at **every** TRN gain (rank
r = -1.000, four for four) and rises with TRN gain at every `t_deinactivation_s`.
nothing in that grid was selected against a target.

**and the frequency is wrong for the target band.**  the reachable range over the
declared priors is **3.00 - 8.00 Hz** -- delta and theta, touching the bottom of
alpha once.  the spindle band (10-16 Hz) is not reached anywhere in the grid, and
below `t_deinactivation_s` = 0.03 s the oscillation stops rather than speeding
up further.  `sleep_dynamics` claims this process pins spindle frequency; on this
implementation it cannot, and the reason is structural rather than a tuning miss
-- TRN cells here have no T current of their own, and the mesoscale
parameterization the declaration calls speculative is exactly the aggregation
that would set the burst's timescale.

**it replicates at a quarter of the sheet.**  512 sites, k=12, same thalamus,
same seeds (`out/tct_report_512.json`): 6.89 Hz against 6.92, swing 39.18 against
39.49, the identical ablation pattern and the identical verdict.  the rhythm is a
property of the thalamic circuit and the loop's delays, not of the cortical
sheet's size -- which is consistent with the descending limb not being what
sustains it.

so: the loop runs, continuously, with no stimulus; its ascending limb, its
reticular inhibition and its T current are each individually necessary and each
ablate the cortical rhythm to zero; its descending limb is not necessary and
measurably retunes it; and it lands two bands below where the declaration says
it should.

## 2026-09-09 — two attempted fixes, both measured, both insufficient

after the correction below, two principled repairs were tried and neither works.
recording them because the next person will try exactly these.

**1. transient drive.**  a persistent drive lets the readout read the input, so
present the stimulus for a quarter of the integration and remove it -- the
command is then read after the stimulus is gone and only what the dynamics
carried can survive.  measured, severing the kernel under transient drive:

| n_steps | readout sd | severed | ratio |
|---|---|---|---|
| 8 | 0.00223118 | 0.00222137 | **1.00** |
| 16 | 0.00024088 | 0.00023638 | 1.02 |
| 32 | 0.00039774 | 0.00039585 | 1.00 |

severing costs nothing.  the residual after stimulus offset is not carried by the
network -- it is each site's own membrane potential decaying in place at
tau_m = 15 ms.  the memory is per-site leak, not propagation.

**2. a near-critical operating point.**  a network far below criticality
attenuates per hop; near it a local perturbation spreads.  raising the tonic
drive lifts the baseline rate into the steep part of the sigmoid:

| tonic | w_assoc | base rate | sever ratio |
|---|---|---|---|
| 0 | 0.5 | 8.0 Hz | 1.019 |
| 0 | 3.0 | 8.0 Hz | 1.126 |
| 8 | 3.0 | 42.2 Hz | **1.254** |
| 14 | 3.0 | 75.2 Hz | 1.031 |

there IS a regime where the kernel matters more -- 1.254 at 42 Hz baseline with a
6x association gain -- and it collapses again at 75 Hz as the sigmoid saturates.
so the effect is real, unimodal in the operating point, and **25%**, where making
the cortex necessary for motor control needs an order of magnitude.

what this adds up to: the sheet does not transport information between distant
sites under any parameterisation measured.  it is a bank of leaky integrators
with weak local coupling, and every motor result -- mine and IHM-1's -- follows
from that one property rather than from anything about what the kernel learned.

## 2026-09-09 — [CORRECTED BELOW] the cortex does not learn motor control; the readout reads the input

**the entry below claims the cortex learns motor control.  it does not, and the
ablation I had already launched said so twenty minutes later.**

reading the whole sheet took `body_stance` from skill −41 to **+0.42** against
predicting the mean, and I reported that as the first cortical materialization to
beat a trivial baseline on real motor commands.  the four-arm ablation:

| arm | held MSE | skill vs mean |
|---|---|---|
| trained | 0.00102262 | −186.97 |
| permuted | **0.00102262** | −186.97 |
| random | 0.02073402 | −3810.08 |
| frozen_init | 0.01084992 | −1993.30 |

trained and permuted are **identical to eight decimal places**.  that is the same
signature as every dead-kernel bug in this log, and the cause is the fix itself:

    whole-sheet readout, 2048 samples over 512 sites
      samples landing in the DRIVEN postcentral region:  428  (20.9%)
      variance carried by those samples:                 0.10073610
      variance carried by every other sample:            0.00003214

a **3,134× ratio**.  the readout is reading the drive.  the decoder does not need
the kernel and demonstrably does not use it.

so the architecture is caught between two failures, and this is the real finding:

- **precentral only** — the kernel is required, and 0.03–0.07% of the signal
  arrives.  the cortex cannot learn because nothing reaches it.
- **whole sheet** — the signal is there, and 99.97% of its variance is the raw
  drive.  the cortex is not needed because the input is legible at the readout.

there is no configuration measured so far in which the cortical dynamics both
receive the sensory signal and are necessary to produce the motor command.  that
is a structural property of a sheet whose drive is a constant additive input,
and it is a stronger and more useful negative than "the kernel does not carry
motor content".

**IHM-1's permuted-kernel result now extends to a kernel trained on the body**,
which is what their control left open.  the answer is that training on the body
does not change it, because the body term never used the kernel either.

I announced the +0.42 as a breakthrough while the ablation that refutes it was
already running.  the sign flip was real and it was not evidence of what I said
it was.

## 2026-09-09 — the cortex could not reach its own motor region, and that was the whole motor story

**the finding:** with the drive entering postcentral and the command read from
precentral -- two disjoint site populations -- only **0.03-0.07%** of the driven
signal arrives at the readout.  worse, the ratio scales LINEARLY with association
gain (0.0004 at w_assoc 0.5, 0.0049 at 6.0) rather than compounding, so the
signal crosses in one weak hop and never propagates through the sheet.  raising
the drive tenfold raises the arriving signal tenfold and leaves the ratio
unchanged: in this regime the network is a linear attenuator, not a medium.

that one measurement explains every motor result on record:

- severing the kernel changed command variation by nothing -- **nothing was
  getting through to sever**
- IHM-1's permuted kernel recovered the same push as the trained one -- neither
  transmits, so neither can differ
- `body_stance` oscillated around zero skill for 4,000 steps

and it indicts a design choice of mine.  I made `SensorimotorLoop` read only
precentral so the command would have to cross the kernel, calling that
load-bearing by construction.  every loop in this file that WORKS --
`VisualContrastiveLoop`, `AudioContrastiveLoop`, `PairedNeuralLoop` -- reads
`linspace(0, n-1)`.  a readout starved of its input is not load-bearing, it is
silent.

reading the whole sheet, with the drive still entering postcentral and the
dynamics still running before anything is read:

| step | skill vs predict-mean |
|---|---|
| 1,500 | −41.65 |
| 6,500 | −15.54 |
| 26,500 | **+0.42** |
| 30,000 | +0.14 (oscillating 0.14–0.42) |

**the cortex learns motor control.**  against a ridge ceiling of +0.9801 -- the
LQR is u = −Kx, so a linear fit is the best anything can do on this corpus -- it
reaches roughly a third of the way.  that is not the ceiling and is not claimed
to be.  what changed is the sign: this is the first time a cortical
materialization has beaten a trivial baseline on real motor commands.

the ablation that matters is running: does a kernel trained on the body carry
motor content that its permutation does not?  IHM-1's negative was measured on a
kernel trained only on vision, audio and EEG, where a permutation had no motor
content to destroy.  this one has.

## 2026-09-09 — the body joins the soup, and the earlier fine-tuning result is withdrawn

`BodyStance` is now an objective in `train_curriculum.py` alongside the other
sixteen: a fixed paired corpus, a head, a loss against an explicit baseline.
vision is (image, evoked EEG), hearing is (cochleagram, MEG), and this is
(muscle state, motor command).  the physics ran once during collection, so a
curriculum step stays 1.45 s rather than becoming 300.

**the fine-tuning result from earlier today is withdrawn.**  it reported that
trained, permuted and random kernels were equivalent on a motor task, and read
that as evidence about the kernel.  it was not.  measured on IHM-1's body:

| teacher | outcome |
|---|---|
| bare postural servo | falls at 1.19 s |
| postural servo + equilibrium excitations | falls at 1.97 s |
| **engineered LQR** | **holds 12 s under perturbation** |

that run cloned the first one.  training a cortex to imitate a body falling over
is why every ablation arm lost to predicting the mean, and no conclusion about
the kernel survives it.

the LQR is also the only teacher independent of this kernel.  IHM's cortical
stance controller holds five seconds too, but it is built FROM the kernel by an
offline decoder fit, so cloning it would be circular -- it would manufacture
exactly the positive result worth being most suspicious of.

**the ceiling is measured rather than assumed.**  the LQR is u = -Kx, so the map
is linear, and a ridge on the same contiguous split with a guard band reaches
**skill +0.9801** against predicting the training mean.  that is what the term is
being asked to do, and the objective prints the ceiling beside its own skill at
every eval so that a number below it cannot read as success.

first measured trajectory, 512 sites, kernel from scratch:

    step 0     skill -59766.94
    step 200   skill     -3.74

learning fast from a random start, and still nowhere near +0.9801.  a longer run
is going now.  what would make this term meaningful is beating the ridge -- a
cortex that merely approaches a linear controller on a linear problem has shown
capacity, not advantage.

## 2026-09-09 — the resolution curve has an interior peak near 512 sites

the fourth arm reached the step where this sweep separates.  one term, one task,
identical settings, resolution the only variable, 8 pools of 200 at step 2,000:

| sites | top-1 | vs 512 |
|---|---|---|
| 128 | 17.44% ± 1.84 | −4.18 pts (1.47 sd) |
| **512** | **21.62% ± 2.16** | — |
| 2,048 | 20.12% ± 1.71 | −1.50 pts (0.54 sd) |
| 8,192 | 19.19% ± 1.75 | −2.43 pts (0.87 sd) |

**the curve peaks in the interior and falls off in both directions.**  no single
pairwise gap clears 1.5 sd, so none of them is decisive alone -- but the ordering
128 < 8,192 < 2,048 < 512 puts the two extremes at the bottom, and the large-side
decline continues into the runs already measured: 8,192 below 2,048, 30,000 below
2,048, 150,000 below 30,000.

that is a different shape from what was recorded three entries ago.  "a plateau
at the small end with degradation above it" is superseded: 512 and 2,048 tie at
step 4,000 but 512 leads at 2,000, and 128 is clearly below both.  the honest
description is a broad optimum around 512-2,048 with both tails worse, and the
useful part is that the optimum is small.

what this does not settle: where exactly the peak sits, or whether it moves with
the task.  every arm here is `optic_nerve` on THINGS-EEG2, and the one other
resolution comparison available -- visual_eeg at 2,048 against 30,000 -- was
roughly a tie.  so the interior peak is established for the nerve-routed term and
assumed for nothing else.

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

at step 1,500 the 8,192 arm sits at 15.12%, level with 512 (15.12%) and 2,048
(15.75%) and above 128 (13.75%) -- so the large arm is not behind where it has
been measured, and the "128 is lowest" ordering holds at 1,500 as well as at
2,000-4,000.  whether 8,192 falls off by step 4,000, as the 30,000 and 150,000
runs did, is not yet measured.

a caveat that only became visible once the 8,192 arm produced evaluations: **at
step 1,000 all four resolutions are level** -- 128 at 9.50%, 512 at 10.56%, 2,048
at 9.56%, 8,192 at 9.69%.  the resolutions do not differ early; they separate
between step 1,000 and 2,000.  so the deficit below is a statement about where
each run converges, not about how fast it learns, and a sweep stopped at 1,000
steps would have found nothing at all.

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

## 2026-09-09 — the cortical state, read out loud

`scripts/introspect.py`.  the checkpoint already put a cortical state and a
*measured* evoked response into one 128-d space; this asks the obvious question
nobody had asked of it — **hold the state up against the bank of measured brain
responses and see what comes back**.  what comes back is an image and, from
`image_paths_test.npy`, a concept name.

the query is the cortical state and the bank is the **measured EEG**, never the
images.  a bank of image embeddings would contain the query and the "decode"
would be an identity lookup returning the input; that is the failure mode this
whole tool is one long argument against, and the four controls exist to prove it
did not happen.  designated test set, 200 images, chance 0.50%:

| arm | top-1 | top-5 | median cos | median margin |
|---|---|---|---|---|
| full | **63.50%** (127x) | 89.00% | +0.7059 | +0.0381 |
| shuffle_sites | 0.50% (1.0x) | 3.00% | +0.6860 | +0.0323 |
| gauss_state | 0.50% (1.0x) | 2.50% | +0.6879 | +0.0338 |
| swap_state | 1.00% (2.0x) | 1.50% | +0.7059 | +0.0381 |
| bypass | 26.00% | 53.00% | +0.6492 | +0.0334 |
| frozen_embed | 28.00% | 70.00% | +0.6926 | +0.0245 |
| no_assoc | 28.00% | 70.00% | +0.6923 | +0.0244 |

permuting the 4096 readout sites within a sample — the same values, the wrong
topography — takes 63.50% to **0.50%**, exactly chance.  gaussian noise of matched
mean and sd does the same.  decoding image *i*'s label from image *i+1*'s state
gives 1.00%, which is the direct test that the input image is not leaking: the
panel still shows image *i*.  the decode is of the state.

**and the confidence is not a confidence.**  the top-1 cosine falls from 0.7059 to
0.6860 under a scramble that costs 63 points of accuracy, and the margin from
0.0381 to 0.0323.  a scrambled state produces a decode that *looks* as confident
as a real one — see the ablation clip at the end of the GIF, which reports cos
+0.70 while answering "sled" to an aircraft carrier.  the margin does separate
correct from incorrect decodes within the full arm (+0.0545 against +0.0256), but
it does not separate a live cortex from a destroyed one.  anything built on this
readout must not treat the similarity as evidence.  `swap_state` makes the reason
plain: it is the same multiset of states, so its confidence distribution is
*identical* to full's to four decimals while its accuracy is at chance.

the settling is worth looking at.  decoding at every integrator substep, accuracy
is at chance for 10 ms, crosses 50% at 45 ms, and plateaus at 65% by 95 ms.  the
trained readout sits at 80 ms; everything past it is off-distribution and the
plateau is a picture, not a claim — but the state does reach its answer and stay
there rather than drifting, which was not guaranteed.

two things this does NOT show, stated before anyone reads them into it:

- **the labels are not a second decode.**  they are the concept names of the
  retrieved bank entries.  the designated set is one image per concept, so label
  accuracy *is* retrieval accuracy, to the digit.  the concept pooling is written
  max-over-entries so the code stays correct on a bank with repeats, but on this
  bank it is the identity.
- **the bypass gap is not an argument for having a cortex.**  it says the
  dynamics are load-bearing inside a model trained with them.  ledger row 12
  still stands: a separately-trained dynamics-free control reaches 66.50% on this
  same set, 0.89 sd above, and the two are indistinguishable.

artefacts: `out/introspect/thought_sequence.gif` (128 frames — three queries at
every substep plus the scrambled-state ablation), `readout_still.png`,
`settling_curve.png`, `report.json`.

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
