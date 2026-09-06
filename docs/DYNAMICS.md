# cognition as the geometry of a vector field

what ibm-1 is ultimately for, and the standard against which its parts are
judged. ARCHITECTURE.md says what exists; STATE.md says where we are; this says
why any of it matters.

## 1. the formulation

take brain state at time $t$ as a vector $s$ over cortical, thalamic,
hippocampal, basal-ganglia, cerebellar, neuromodulatory, metabolic and structural
fields:

$$
s=(x_c,\;x_t,\;x_{bg},\;x_h,\;x_{cb},\;n,\;\theta,\;e,\ldots),
\qquad
\dot s=F(s,u)
$$

cognition is **not** symbols being passed around. it is the trajectory generated
by a nonlinear vector field

$$
\dot x=f(x,u;\theta,m)
$$

where $u$ is sensory and interoceptive forcing, $\theta$ is slow
synaptic/structural state, and $m$ is fast modulatory state — thalamic mode,
neuromodulators, oscillatory phase, arousal, task context.

**the cognitive objects are geometric objects of $f$**: fixed points, limit
cycles, slow manifolds, metastable sets, heteroclinic channels, continuous
attractors, and their basins. a concept or a working-memory item need not be one
neural pattern; it can be a region $A_i$ onto which many microscopic states
converge to approximately the same macroscopic trajectory. basin geometry is what
supplies categorical robustness: perturb $x$, and if it stays inside $B(A_i)$,
the dynamics reconstruct the cognitive state.

**this is not a gradient system.** biological networks are generally
non-gradient, so $\dot x \neq -\nabla E(x)$. rotational flows, limit cycles,
travelling waves, asymmetric transition paths and metastability are all
available. "landscape" here is shorthand for the topology of the vector field and
its invariant and metastable structures — never literally a hopfield energy.

## 2. the thalamo-cortico-thalamic loop is landscape control

the reason TCT is special is that thalamus modifies the **effective vector
field** rather than merely contributing another state variable:

$$
\dot x_c=f_c(x_c)+W_{tc}(m)\,x_t+W_{cc}(m)\,x_c,
\qquad
\dot x_t=f_t(x_t)+W_{ct}(m)\,x_c
$$

eliminating the fast thalamic variables locally induces an effective cortical
operator

$$
\dot x_c\approx f_{\mathrm{eff}}\!\left(x_c;\;W_{cc}+W_{tc}\,G\,W_{ct},\;m\right)
$$

so changing thalamic gain $G$, inhibition, synchrony or nucleus-specific routing
changes the **effective recurrent coupling of cortex**, and therefore the
jacobian

$$
J(x)=\frac{\partial f_{\mathrm{eff}}}{\partial x}
$$

— which modes are stable, which perturbations grow, where basin boundaries lie,
transition probabilities, and the dimensionality of the active manifold. that is
a far stronger claim than "thalamus routes information".

**cognitive control is then landscape control**: $f(x;\theta,m_1)\to
f(x;\theta,m_2)$. the same learned brain, different $m$, different effective
attractor topology — one context deepens basin $A$, suppresses $B$, opens
$A\to C$ and closes $A\to D$. attention, working memory, perceptual
interpretation and much of executive control become controlled changes in the
stability and accessibility of distributed trajectories, rather than tokens moved
through a pipeline.

## 3. learning deforms $F$

learning acts on the slow variable. an experience traces
$\gamma: x_0\to x_1\to\cdots\to x_T$, and plasticity changes $\theta$ so later
dynamics are biased toward reconstructing relevant parts of it:

$$
\theta_{t+1}=\theta_t+\eta\,\Phi(\gamma,m)
$$

geometrically: create a new attractor, enlarge a basin, carve a preferred
transition channel, or bind previously separate manifolds. repeated experience
does not store a literal sequence — it progressively changes $f$ until the
sequence becomes a natural trajectory of the system.

**sleep is offline replay and systems consolidation** — not reconsolidation,
which is the distinct phenomenon of a consolidated memory becoming labile after
retrieval and then restabilising. during NREM the nesting is

$$
\text{slow oscillation}\supset\text{thalamocortical spindle}\supset\text{hippocampal ripple}
$$

and plasticity is **phase dependent**:

$$
\Delta\theta=\Phi\!\left(x,\;\phi_{\mathrm{SO}},\;\phi_{\mathrm{spindle}},\;\phi_{\mathrm{ripple}}\right)
$$

so identical activity at a different phase has a different learning consequence.
consolidation is not copying bytes hippocampus→cortex; it is repeatedly
perturbing cortex along correlated trajectories until $\gamma$ lies closer to an
intrinsic high-probability trajectory of cortical dynamics — deeper basin,
stronger transition channel, less dependence on whatever originally scaffolded
it.

the honest abstraction for motor learning avoids "hippocampus records footsteps
and writes them to cortex":

$$
\text{online trajectory}\to\text{temporary distributed trace}\to
\text{state-gated offline reactivation}\to\Delta\theta\to
\text{changed attractor/transition geometry}
$$

## 4. multi-timescale eligibility ties it together

events leave decaying hidden traces

$$
e_k(t)=\int_{-\infty}^{t}K_k(t-\tau)\,\psi_k(x(\tau))\,d\tau
$$

on which later dopamine, replay, spindle phase, dendritic spikes or novelty can
act. this is what lets causally related information separated by milliseconds,
seconds or hours **meet at the plasticity rule**. and the timescales are
genuinely separate:

$$
x:\;\text{ms–s}
\qquad
m:\;\text{s–hours}
\qquad
\theta:\;\text{min–months}
\qquad
\mathcal G:\;\text{days–years}
$$

## 5. the bridge

$$
\boxed{
\text{biophysics}\to f(x;\theta,m)\to\text{attractor/manifold geometry}
\to\text{cognitive dynamics}\to\text{experience-dependent deformation of }f
}
$$

TCT sits near the middle because it operates on both timescales: fast modulation
of the currently accessible dynamical subspace, and — with appropriately timed
plasticity — participation in changing the longer-lived geometry itself. sleep is
the legible case because the system deliberately enters a different dynamical
regime in which external forcing falls, oscillatory timing becomes highly
structured, and internally generated trajectories can train the network.

**cognition is not the contents of individual states; it is the structured flow
between metastable regions of brain-state space. learning changes that flow
field. sleep performs offline trajectory-dependent edits to it. TCT dynamics
rapidly reparameterise which portions of that learned flow field are currently
stable, communicative and plastic.**

## 6. mechanism inventory, against what ibm-1 declares today

audited against the sealed registry. this is the gap list, not an aspiration.

### declared and materialized
| # | mechanism | where |
|---|---|---|
| 1 | local recurrent dynamics | `local_excitation`, `local_inhibition`, `microcircuit` |
| 2 | laminar computation | `laminar_propagation`, `cortical_layers`, `cortical_depth` |
| 3 | E/I interneuron classes | `neural.{pv,sst,vip}.activity` |
| 4 | cortico-thalamo-cortical | `thalamocortical_coupling`, `alpha_resonator`, `corticothalamic_loop_lti` |
| 11 | synaptic homeostasis | `homeostatic_scaling_lti` |
| 14 | oscillatory coordination | **the temporal-spectral form itself** — phase-dependent effective coupling is native here, not bolted on |
| 15 | neuromodulatory state | `neuromodulation`, `writes="parameters"` |
| 16 | intrinsic dynamics | `neural.exc.adaptation`, `wilson_cowan_adaptive` |
| 23 | structural plasticity | `plasticity`, `structural.*`, `activity_dependent_myelination` |
| 24 | astrocytic/extracellular | `extracellular.*`, `structural.gliosis` |
| 25 | metabolic/vascular constraint | `metabolism`, `neurovascular_coupling`, `tissue_exchange` |
| 28 | multi-timescale eligibility | `three_factor_stdp`, `eligibility_trace_transfer` |

### absent — and four have their anatomy already declared with NO process using it
| # | mechanism | status |
|---|---|---|
| 5 | basal-ganglia selection | `bg_territories` declared, **used by nothing** |
| 7 | hippocampal fast indexing | `hippocampal_subfields` declared, **used by nothing** |
| 8 | hippocampal replay / sequences | absent |
| 18 | cerebellar predictive control | `cerebellar_lobules`, `cerebellar_microzones` declared, **used by nothing** |
| 2b | apical/basal dendritic nonlinearity | absent — `y=f(x_{basal},x_{apical},m)`, not $f(\sum w_ix_i)$ |
| 9 | systems consolidation | absent |
| 10 | NREM phase-gated plasticity | absent — plasticity does not read oscillatory phase |
| 12 | reconsolidation | absent |
| 17 | dendritic credit assignment | absent |
| 19 | efference copy / corollary discharge | absent — `efferent_propagation` exists, the return path does not |
| 21 | active sensing (closed brain–body–world loop) | absent |
| 26 | developmental priors | absent |
| 27 | offline generative regimes beyond NREM | absent |

**the four anatomy-without-process entries are the cheapest real progress
available**: the partitioning systems are declared, the supports exist, and what
is missing is the process declaration and an implementation.

## 7. why this can be bridged *en training*

these mechanisms do not all have to be hand-specified, and that is the point of
the closed-loop regime in STATE.md §7b. the declared form CONSTRAINS and the data
FILLS IN:

- a process declares $(I,O,T)$ and a family of $f$ with priors; training moves
  $\theta$ within that family
- an unbounded stimulus stream supplies the trajectories $\gamma$ that plasticity
  needs, at a volume no paired-recording corpus can reach
- the measured-recording likelihood anchors the result so the learned $f$ stays a
  brain rather than becoming whatever fits the loop

so the arc is: declare the mechanism, give it a weak prior, and let the
trajectories deform it — rather than specifying a hippocampus and hoping.

**but nothing cognitive can emerge until the dynamics are nonlinear and
bounded.** the materialized graph currently has NO CYCLE — `build` selects LTI
for all 29 processes and the one edge closing the cortical loop is
`potential→rate`, whose only $f$ is a sigmoid. a linear system has exactly one
fixed point: no metastable sets, no basins, no transitions, nothing for a
cognitive object to be. 27 `Form.RATE` implementations are declared and the one
nonlinear run attempted diverged at prior parameters. that ordering constraint is
STATE.md §7c and it gates everything in this document.
