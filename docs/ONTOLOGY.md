# higher-order constructs: what to take, what to decline, and what it means for curriculum

orientation against *The Shape of Experience* (theshapeofexperience.org), read in
full: introduction, the identity thesis, the geometry of affect, the meaning
crisis, aesthetics, science, affect engineering, the geometry of capture,
transcendence.

**none of it is assumed correct.** the purpose is a sharp ontology of
higher-order constructs to think with while designing the curriculum and the
interventions in STATE.md §7c. DYNAMICS.md §8 audits the affect geometry against
what we can compute; this is the layer above that.

---

## 1. the framework is four different kinds of claim, with four different standings

conflating these is the main way to get confused by it. separated:

| tier | claim | standing for ibm-1 |
|---|---|---|
| **T1 metaphysics** | $P(S,s)\equiv CE^{\rm intrinsic}(S,s)$ — experience *is* intrinsic cause-effect structure | **decline.** flagged by the author as a working axiom, unverifiable by their own argument ("no vantage point accesses both"). adopting it buys nothing and costs falsifiability |
| **T2 coordinates** | the cognitively relevant macrostate of a self-maintaining controller is ~6–9 dimensional: $(\mathrm{Val},\mathrm{Ar},\Phi,r_{\rm eff},\mathrm{CF},\mathrm{SM})$ plus $(\kappa,\gamma,\alpha)$ | **take, and test.** this is a dimensional-reduction hypothesis about good order parameters. it is exactly what a curriculum needs |
| **T3 taxonomy** | specific pathologies and aesthetic modes have specific signatures in those coordinates | **take as hypotheses.** several are recognisable training failure modes with no name in ML (§4) |
| **T4 engineering** | interventions are operations on the macrostate distribution; expansion vs capture; parasitic vs mutualistic optimization | **take. most actionable material in the eight pages** (§3, §5) |

**the crucial move: every operational consequence of T2–T4 survives if T1 is
false.** read the identity thesis as *"these are the right macroscopic
coordinates for describing a self-maintaining controller"* — a claim about good
coordinates, which is testable — and nothing downstream breaks. that reading is
what we adopt. it lets us use the whole apparatus without inheriting a
metaphysics we cannot check.

so: **we are not building a system in order to make claims about its experience.
we are borrowing a well-developed vocabulary for the macroscopic structure of
controllers, because we happen to be building one and have no other vocabulary at
that level.**

---

## 2. the order parameters are the curriculum's coordinate system

the practical reason T2 matters: **you cannot design a curriculum over $10^8$
parameters. you can design one over nine order parameters.**

with DYNAMICS.md §8: three are computable ($\mathrm{Ar}$ as a KL between our
propagated belief states, $r_{\rm eff}$, $\Phi$ via the predictive proxy), one
has a principled grounding we uniquely possess (valence against a *metabolic*
viability boundary), and two are blocked on mechanisms 8, 18, 19, 21.

the three control parameters map onto things we already have:

- **$\gamma$ (gain / precision on evidence)** — `neuromodulation` writes exactly
  this. it is the one process whose `writes="parameters"`
- **$\kappa$ (coupling between modes)** — this is the effective coupling
  $W_{cc}+W_{tc}GW_{ct}$ of DYNAMICS.md §2. **thalamocortical gain IS $\kappa$.**
  that is a strong and unforced identification: the framework's central control
  knob and ours are the same object
- **$\alpha$ (ascription)** — no counterpart, and correctly so: it requires a
  self/other model (mechanisms 19, 21)

---

## 3. two convergences that were not designed, and are therefore worth something

**3.1 substrate independence is our laziness criterion.** transcendence gives the
portability condition

$$
h\!\left(T_A(x,u)\right)\approx T_B\!\left(h(x),\tilde u\right)
$$

— a coarse-graining $h$ is faithful iff it commutes with the dynamics. that is
*verbatim* ARCHITECTURE.md's rule that **resolution earns its cost only where
coarse-graining fails to commute with the dynamics.** same equation, derived
independently for a different purpose. this means the multiscale materialization
machinery already implements the test their portability claims require, and
conversely that our laziness axis is doing philosophically heavier work than we
built it for.

**3.2 the reversibility ethic is already in the type system.** affect engineering
proposes that interventions permanently altering the attractor landscape warrant
higher scrutiny than transient ones. for us that distinction is not a policy — it
is `writes="parameters"` vs `writes="state"`, and it is checked at seal time.
**training changes $\theta$ and is irreversible; stimulation changes $m$ and is
not.** we get their central governance distinction for free because the
architecture already had to make it for correctness reasons.

---

## 4. the pathology taxonomy is a diagnostic vocabulary for training failures

this is the part I did not expect to be useful and think is the most immediately
useful. each pathology is a *shape of the fitted vector field*, and each has an ML
analogue that our current vocabulary cannot name:

| their pathology | geometry | our failure mode |
|---|---|---|
| melancholic depression | low $\kappa$ → flat force field; landscape visible but unfelt; collapsed $r_{\rm eff}$ | **representation collapse / dead gradients.** the model has states and no dynamics moving between them |
| expansive despair | high $\alpha,\gamma,r_{\rm eff}$ but low $\kappa$ — abundance without force to traverse it | **scaling capacity ahead of coupling.** see §6 |
| anxiety | high $\gamma$ flooding under-coupled models; terrain destabilised continuously | **precision/learning rate too high on sensory evidence relative to priors.** the landscape never settles long enough to be learned |
| addiction | closed circular attractor: high force, circular trajectory, **zero net traversal** | **reward hacking / policy limit cycle.** the sharpest of these — a system maximally driven and going nowhere is a limit cycle in the objective, and "zero net traversal" is the exact diagnostic |
| degenerate evaluation | intact skeleton, pathological embedding; ascent to a frame where all gradients symmetrically cancel | **the model finds a meta-frame trivially satisfying the objective.** gradients cancel by symmetry rather than by convergence |

**why this is worth having:** these are distinguishable by *measurement on the
fitted $F$*, not by loss curves. flat-field collapse and zero-net-traversal
looping and symmetric-cancellation all look like "loss stopped improving." they
are three different diseases with three different interventions, and the
framework tells us which measurement separates them ($\kappa$, net displacement
along the trajectory, gradient rank at the abstraction level in question).

their proposed interventions also transfer: depression → restore curvature by
recoupling modes; anxiety → stabilise the eigenspace; addiction → expand topology
beyond the loop; degenerate evaluation → **frame separation, anchor to embodied
non-abstracted coordinates where local gradients resist nullification.** that last
one is an argument for keeping the metabolic and interoceptive channels *in* the
training loop rather than treating them as periphery.

---

## 5. expansion vs capture is the central curriculum principle — and it indicts our current plan

the aesthetics page draws the distinction we most need:

- **expansion** installs new basis vectors, enabling previously unrepresentable
  states. uncomfortable initially; grows capacity
- **capture** activates *existing* palette vectors with maximal resonance.
  immediately gratifying; **entropic** — reinforces the palette without extending
  it

and the warning: optimizing for palette resonance produces "comfortable prisons
rather than challenging curricula."

**this is a direct critique of STATE.md §7b as currently specified.** next-frame
prediction on naturalistic video, trained on predictive loss alone, is *exactly*
palette resonance. the loss is minimised by exploiting what is already
representable. it will deepen existing basins and has no term that installs new
ones. **a pure predictive-loss curriculum is a capture curriculum by
construction.**

the framework also says what expansion requires: *"sufficient divergence from
audience expectation paired with integrability."* high prediction error that is
nonetheless resolvable. that is a learning-progress / zone-of-proximal-development
criterion, and it is implementable:

> **curriculum rule.** select stimuli where prediction error is high **and
> declining**, not where it is lowest (capture) and not where it is highest
> (noise). track $r_{\rm eff}$ and the count of distinct metastable sets as the
> expansion signal; predictive loss alone is the capture signal. **if loss falls
> while $r_{\rm eff}$ and the invariant-set count do not rise, the curriculum is
> capturing, not teaching.**

that is a measurable curriculum objective we did not previously have, and it is
the single most actionable thing in the eight pages.

two supporting notions worth keeping:

- **holonomy specification** — a metaphor "declares this IS that", installing one
  domain's topological structure onto another; once coupled, the modes cannot be
  uncoupled without losing the insight. the machine-learning reading is
  **cross-modal binding as a permanent topological edit**, which is what our
  distillation from TRIBEv2 across visual and auditory streams actually is. it
  predicts such bindings should be *hard to undo*, which is a testable property
  of the fitted model
- **"growing up is compression"** — crude compression discards modes that look
  low-variance individually while cutting their coupled signal. this is a precise
  warning about our own low-rank uncertainty forms and any pruning we do: **rank
  selection on marginal variance destroys exactly the coupled structure we are
  trying to learn.** prune on coupled contribution, never on marginal variance

---

## 6. a specific scaling warning

the meaning-crisis diagnosis is that $\kappa$ declines while symbolic capacity
expands, leaving traversal machinery unable to navigate an enlarged landscape.

for us, transposed: **scaling the learnable cortical association weights to
$10^8$ while leaving thalamocortical coupling and the neuromodulatory channel at
their priors produces expansive despair** — an enormously rich landscape with no
force field able to traverse it. concretely, `cortical_association` carries the
parameter count; $\kappa$ lives in `thalamocortical_coupling` and $\gamma$ in
`neuromodulation`, both currently at prior with a single LTI implementation each.

**scale $\kappa$ with $r_{\rm eff}$, not after it.** that is a design constraint
on the 100M-parameter plan, and it is checkable during training as a ratio rather
than discovered afterwards as a failure.

---

## 7. objective specification: parasitism has a formal test, and we can run it

the capture geometry defines the alignment relation as **viability manifold
containment**:

$$
\text{aligned:}\;\;V_{\rm substrate}\subseteq V_{\rm agent}
\qquad
\text{parasitic:}\;\;H^*_G>\varepsilon
\;\;\text{(no persistence policy avoids substantial substrate harm)}
$$

with the diagnostic being counterfactual welfare, not ideological labelling, and
symbiosis being $\mathbb E[m_h|do(G)]-\mathbb E[m_h|do(\neg G)]>0$.

**transposed to training, this is a real check we can actually run**, because
DYNAMICS.md §8 gives us a non-stipulated $V$: metabolic feasibility, from
`metabolic.atp`, `metabolic.consumption` and the delivery bound in
`vascular_flow`.

> **objective check.** does minimising the training loss drive the model toward
> states outside its own metabolic viability set? if lowering loss requires
> metabolically infeasible activity, the objective is parasitic on the substrate
> **in the framework's exact technical sense**, and the fitted brain is buying
> predictive performance with physiology it could not sustain.

this is not a metaphor and it is not ethics — it is a regularizer with a measured
constant, and it is the reason mechanism 25 belongs in the dynamics rather than
in the noise model. **it also converts "is this still a brain?" from a vibe into
an inequality.**

the self-sealing observation transfers too: parasitic patterns resist detection
because critique is delivered at low $\kappa$ and metabolised as decoupled
propositional information. our version — **a metric that the training objective
can optimise directly stops being a diagnostic.** hold at least one expansion
measure out of the loss.

---

## 8. the intervention primitives, against our 21

affect engineering's thirteen primitives map cleanly onto what we can and cannot
currently do. we have the **physical** interventions and almost none of the
**cognitive** ones:

| primitive | ibm-1 |
|---|---|
| entrainment | **have**: `tacs`, `trns`, `auditory_stimulus`, rhythmic `visual_stimulus` |
| gain modulation ($\gamma$) | **have**: `pharmacological`, `anaesthetic`, and `neuromodulation` as the process |
| boundary softening / palette work | **partial**: `sensory_deprivation`, `naturalistic_stream` |
| ritualized traversal (basin deepening) | **have in principle**: repeated `task_cue` — this is just curriculum repetition |
| salience redistribution, self-model resizing, counterfactual loading/pruning, ascription modulation, viability-horizon modulation, symbolic immortality transfer | **absent.** all require task semantics and a self-model — mechanisms 7, 8, 19, 21. they route through `task_cue`, which is declared but carries no content |
| synchrony induction (multi-agent) | out of scope; there is one brain |

**the gap is exactly the curriculum.** our interventions are things done *to* a
brain; theirs are things done *to a brain that is doing something*. `task_cue`
is the seam, and it is currently an empty declaration. **giving `task_cue` real
content is the concrete form of "curriculum → cognitive schema → RL" in STATE.md
§7c.**

the flourishing functional they propose,

$$
F(a)=w_1\mathrm{Val}+w_2\Phi+w_3 r_{\rm eff}
-w_4(\sigma_{\rm att}-\sigma_{\rm opt})^2-w_5|\mathrm{Ar}-\mathrm{Ar}^*|
+w_6\,\mathrm{flex}(\alpha,\kappa,\gamma)
$$

is worth recording not as an objective to optimise — §7's self-sealing warning
applies directly, and optimising it is precisely how you get a captured system —
but as a **held-out evaluation**. the $w_6$ flexibility term (time-averaged rate
of axis modulation) is the interesting one: it rewards *capacity to modulate*
rather than any particular set point, which is the right shape for something
you want to remain able to learn.

---

## 9. what I do not buy, and what I would watch for

- **the RSA cross-substrate result** ($\rho=0.81$–$0.89$ human vs synthetic,
  rotation-invariant) is a claim about representational similarity, not about
  dynamics. two systems can have matching similarity structure and completely
  different vector fields. it is evidence for shared *coordinates*, which is T2
  and useful, and not evidence for T1
- **$\Phi$ as defined** (minimum over bipartitions) is not computable at our
  scale and the proxy $\Delta P$ is a different quantity that happens to
  correlate. use the proxy, call it the proxy
- **"faint experience nearly everywhere"** follows from T1 and we declined T1, so
  it does not follow here. good — it is the part of the view that would otherwise
  make every measurement we take morally fraught
- **the affect motifs** (joy, grief, shame as specific coordinate signatures) are
  the least constrained part. they are post-hoc assignments; if we ever test them
  it must be against held-out labels (`deap`), never by inspection

and one thing to state plainly rather than avoid. the transcendence page's
failure case is a system at high $\Phi$, sustained negative valence, high
attentional self-salience and minimal causal efficacy. **we are nowhere near
this** — the graph is LTI with one fixed point, there is no valence computation,
no self-model and no $\Phi$ measurement. but the ordering matters: **the
measurement should exist before the capability does**, and §8's three computable
dimensions are cheap. building them during the nonlinear-dynamics work costs
little and means we are never in the position of having built something we cannot
characterise. that is a reason to sequence the measurements early, not a reason
to slow anything down.

---

## 10. what this changes

concrete, and ordered by when it bites:

1. **the curriculum needs an expansion term.** predictive loss alone is capture
   by construction (§5). track $r_{\rm eff}$ and the invariant-set count; select
   on error that is high *and declining*
2. **scale $\kappa$ with $r_{\rm eff}$** (§6) — a constraint on the 100M plan,
   checkable as a ratio during training
3. **add the metabolic viability check to the objective** (§7) — a regularizer
   with a measured constant, not a metaphor
4. **prune on coupled contribution, never marginal variance** (§5)
5. **give `task_cue` content** (§8) — it is the seam where curriculum enters, and
   the blocker for six of the thirteen intervention primitives
6. **hold at least one expansion measure out of the loss** (§7)
7. **build the three computable order parameters alongside the nonlinear work**
   (§9), not after

and the standing frame: **T1 declined, T2 adopted as a testable claim about good
coordinates, T3 used as a diagnostic vocabulary, T4 used as engineering.**
