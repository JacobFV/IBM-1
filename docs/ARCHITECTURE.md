# implicit brain model (ibm-1)

the implicit brain model (ibm) is a multiscale probabilistic dynamical substrate from which task-specific explicit brain models are lazily materialized.

$$
\boxed{
\mathrm{ibm}
=
(\text{fields},\ \text{anatomy},\ \text{topologies},\ \text{processes})
}
$$

resolution, bandwidth, datasets, observations, interventions, and explicit models are expressed through these four primitives rather than introduced as additional primitives.

the implicit structure exists independently of what any instrument can observe or any materialization can afford. evidence constrains what it constrains; elsewhere the structure remains at its prior, and is typically smooth or homogeneous. that is the machinery working, not failing.

## 1. fields

a field is an indexed family of state variables:

$$
F=\{x(q):q\in\Omega\},
\qquad
x(q)\in\mathbb R^d
$$

where:

- $\Omega$ is the support of the field
- $q$ is a position on that support
- $x(q)$ is the state associated with that position
- $d$ is the number of state components at each position

the ontology declares **components**. a materialization instantiates **state variables**: one component of one field at one position on its support. a component is a kind of physical quantity — membrane potential, $K^+$ concentration, flow rate. it is what processes exert pressure on, and it is never an encoding of something else.

fields do not share a support. the cortical surface, the vascular tree, the interstitial volume and a sensor array are different domains with different notions of adjacency and distance. a support carries whatever coordinate frame its positions are expressed in, and every atlas, tractogram and dataset declares the frame it speaks. there is no universal spatial domain, and consequently no universal spatial operator.

### materialization of a field

a materialization chooses a region $R\subseteq\Omega$ and spatial resolution $r$:

$$
F[R,r]
=
\{x(q):q\in\operatorname{grid}(R,r)\}
$$

the same field can therefore simultaneously exist at different resolutions in different regions:

$$
F[\text{whole brain},1\text{ mm}]
$$

and

$$
F[\text{electrode neighborhood},50\ \mu\text m]
$$

without instantiating $50\ \mu\text m$ state across the whole brain.

more generally, resolution may itself vary over space:

$$
r=r(q)
$$

for example:

$$
r(q)=
\begin{cases}
50\ \mu\text m & d(q,e)<2\text{ mm}\\
500\ \mu\text m & d(q,e)<20\text{ mm}\\
2\text{ mm} & q\in\text{relevant connected structures}\\
10\text{ mm} & \text{otherwise, if required}
\end{cases}
$$

where $e$ is an invasive electrode location.

this is the basis of spatial laziness.

### when resolution earns its cost

if the state over a region is smooth and every process acting there is linear, coarse-graining commutes with the dynamics: $F[R,r_{\text{fine}}]$ and $F[R,r_{\text{coarse}}]$ produce the same answer, and materializing the finer one is waste.

resolution earns its cost only where that commutation breaks — a nonlinearity whose average is not the average's image, heterogeneity within a coarse cell, or a topology whose edges do not survive coarsening.

$r(q)$ should be derived from this, not from proximity to whatever we happen to be measuring.

### uncertain state

a state variable is not a value at an instant. it is a belief about its trajectory over a window.

*how* that uncertainty is carried is part of the field declaration, not a separate concept:

| form | belief | appropriate when |
|---|---|---|
| scalar gaussian | $x\sim\mathcal N(\mu,\sigma^2)$ over the window | the component has one relevant timescale |
| spectral gaussian | gaussian over temporal laplacian coefficients | several timescales interact within the component |

blood has no meaningful structure above roughly $0.5\ \mathrm{Hz}$; a scalar gaussian carries its uncertainty adequately. neural population state is not like this: slow drift, theta, alpha and gamma interact within the same component, and collapsing them loses the interaction.

learned latent embeddings are not a form of uncertain state. a latent is an encoding, not a physical quantity, and nothing exerts pressure on it. latents live inside a process's $f$ and are invisible to the state graph.

gaussianity describes ibm-1's epistemic representation of uncertain state, not a claim that the underlying biological state is gaussian.

### the spectral form

where overlapping timescales interact within a component, take the laplacian of the **time** axis of that single state variable over a window of $N$ samples:

$$
L_t\varphi_k=\lambda_k\varphi_k,
\qquad
\lambda_k=4\sin^2\!\left(\frac{\pi k}{N}\right)
$$

and carry the belief in that basis:

$$
x=\sum_k z_k\varphi_k,
\qquad
z\sim\mathcal N(\mu,\Sigma)
$$

$L_t$ is a laplacian over time, for one state variable. it is not a laplacian over space; there is no shared spatial domain to take one over. spatial structure is carried entirely by interaction topologies (§3).

three properties follow.

**the laplacian spectrum is the power spectrum.** $\lambda_k$ is monotone in $\omega_k$, so a distribution over the laplacian spectrum is a distribution over the power spectral density. a diagonal $\Sigma$ is a stationary gaussian process; aperiodic $1/f^\beta$ background is a one-line prior.

**amplitude and phase are separately representable.** every eigenvalue away from DC and nyquist is doubly degenerate, and its eigenspace is the $(\cos,\sin)$ pair at that frequency. an isotropic gaussian on that plane is known amplitude with uniform phase — the honest belief about an ongoing rhythm. an anisotropic one is phase preference — an evoked, phase-locked response. the same state variable expresses both, and the transition between them across a trial is what an evoked response is.

**off-diagonal $\Sigma$ is cross-frequency structure.** diagonal is stationary; band-block correlation is phase–amplitude coupling; dense is an arbitrary transient. the covariance structure a model materializes is the hypothesis class it entertains.

a process interacts with only the components of the spectrum that concern it. marginalizing a gaussian to a band is exact, so band-limited coupling costs nothing in accuracy: an fMRI observation contributes precision below $0.25\ \mathrm{Hz}$ and none above it, and no special case is required to say so. conduction delay and any linear time-invariant coupling are likewise diagonal — a phase ramp and a transfer function — rather than requiring per-edge history and a delay-bounded timestep.

it costs, in exchange: a pointwise nonlinearity is local in time and dense in frequency, so a nonlinear $f$ must be evaluated in the time domain and re-analyzed, and does not preserve gaussianity. the window is finite, so continuity between windows is imposed rather than inherited.

### bandwidth as a laziness axis

spatial resolution $r(q)$ is spatial laziness. bandwidth is its temporal twin:

$$
F[R,r,B]
$$

where $B$ is the band materialized for the field over region $R$.

both are budgeted, and the two budgets multiply. a whole-brain hemodynamic materialization is spatially broad and temporally narrow; a single-electrode spike materialization is spatially tiny and temporally wide.

### canonical fields

| field | representative state components |
|---|---|
| neural population | membrane-potential state, firing/activity, excitatory and inhibitory population state, adaptation, synaptic state, population-specific channels |
| extracellular / interstitial | pressure, flow, $K^+$, $Na^+$, $Ca^{2+}$, $Cl^-$, pH, neurotransmitters, neuromodulators, metabolites and other solutes |
| electromagnetic | electric potential, electric field, current density, magnetic field |
| blood | flow, volume, pressure, oxygenation, deoxyhemoglobin, composition |
| csf | pressure, velocity, flow, composition |
| metabolic | oxygen availability, glucose availability, energy state, production and consumption |
| structural | fiber orientation, axonal density, myelination, synaptic density and other slowly varying tissue structure |
| material | conductivity, permittivity, density and mechanical properties |
| thermal | temperature |
| mechanical | displacement, velocity, stress, strain, pressure |
| sensory transduction | photoreceptor and bipolar state, hair-cell displacement and transduction current, vestibular afferent drive, mechanoreceptor, thermoreceptor and nociceptor state, olfactory and gustatory receptor occupancy, visceral and baroreceptor state, adaptation |
| effector | motor-unit recruitment and rate, muscle activation, contractile force, fatigue; extraocular, articulatory and autonomic effector state |
| device | electrode contact voltage and impedance, coil current, array channel state, transducer drive, scanner sequence state, display and speaker output |

additional fields should only be introduced when they represent state that cannot cleanly be expressed as components of an existing field.

### sensorimotor, body and device state

the last three fields are not a different kind of thing. a photoreceptor potential, a
muscle activation and an eeg electrode voltage are state variables with values,
uncertainty and interaction topologies to brain state, exactly as a thalamic membrane
potential is. they are separate fields only because their supports are separate — the
retina, the musculature and an instrument's contacts are not brain tissue — which is
the same reason the vascular tree and the cortical surface are separate.

three things that look like they need new fields do not:

- **afferent and efferent traffic** is neural population state on peripheral supports:
  retinal ganglion, spiral ganglion, dorsal root and cranial nerve, motor neuron pools,
  autonomic ganglia. the neural field's support extends beyond the brain.
- **limb, eye and articulator kinematics** is the mechanical field on body supports.
- **visceral state** is the mechanical, blood and metabolic fields on visceral supports;
  interoception is transduction from those.

it follows that a stimulus, a movement and a measurement are not special:

$$
\text{stimulus}=\text{intervention on transduction state}
$$

$$
\text{movement}=\text{evidence about effector state}
$$

$$
\text{electrode voltage}=\text{device state}
$$

none of them is an input or an output of the model. they are ordinary state, coupled to
the rest by ordinary processes, and a materialization that predicts a movement from a
stimulus is doing the same thing as one that predicts blood oxygenation from population
activity.

## 2. anatomical partitioning systems

an anatomical partitioning system maps spatial positions into memberships over a set of named anatomical partitions.

$$
a(q)\in[0,1]^k
$$

with

$$
\|a(q)\|_1\le1
$$

the inequality permits a partitioning system to cover only part of the brain.

a crisp anatomical atlas is the special case where $a(q)$ is one-hot or zero.

an anatomical partitioning system may be:

- hierarchical
- probabilistic
- incomplete
- defined only over one brain structure
- independently overlapping with other partitioning systems

a position may therefore simultaneously satisfy:

$$
q\in
\text{putamen}
\cap
\text{matrix}
\cap
\text{sensorimotor territory}
$$

provided those labels belong to different partitioning systems.

candidate systems include:

- cortical areas
- cortical layers
- cortical cytoarchitecture
- thalamic nuclei and subdivisions
- hippocampal subfields
- striosome / matrix
- basal-ganglia functional territories
- hypothalamic nuclei
- amygdalar nuclei
- cerebellar lobules
- cerebellar microzones
- brainstem nuclei
- vascular territories

cell populations are generally represented as components of fields rather than anatomical partitions, because several populations may coexist at the same spatial location.

for example:

$$
x(q)=
\begin{bmatrix}
x_{\mathrm{exc}}\\
x_{\mathrm{pv}}\\
x_{\mathrm{sst}}\\
x_{\mathrm{vip}}
\end{bmatrix}
$$

rather than treating excitatory, pv, sst, and vip populations as mutually exclusive spatial regions.

anatomical partitioning systems provide the biological interpretation of multiresolution spatial fields without requiring a universal mesostructural unit.

for example, ibm-1 can represent the subcortex with a $500\ \mu\text m$ neural-population field while simultaneously mapping those state variables through thalamic nuclei, basal-ganglia compartments, hippocampal subfields, or other relevant anatomical systems.

memberships enter processes as weights. a partition boundary is a gradient, not a wall.

## 3. interaction topologies

an interaction topology specifies which materialized state variables may interact through some class of process.

write:

$$
T(i,j)
$$

for the relation between state variables $i$ and $j$.

at minimum:

$$
T(i,j)\neq0
\iff
i\text{ and }j\text{ may interact through }T
$$

a topology may additionally contain geometric information such as:

- distance
- tract length
- orientation
- direction
- contact area
- transport path

a topology is a *support* for interaction, not a weighting of it. interaction strength and dynamics belong to processes.

canonical topologies include:

- 3d local spatial
- cortical-surface geodesic
- cortical-depth / laminar
- local microcircuit
- tractometric
- vascular
- csf
- interstitial
- electromagnetic spatial
- mechanical
- metabolic exchange
- afferent pathway
- efferent pathway
- device coupling

there is no universal interaction graph:

$$
T_{\text{surface}}
\neq
T_{\text{tract}}
\neq
T_{\text{vascular}}
\neq
T_{\text{em}}
$$

even when these topologies operate over overlapping state variables.

the state graph is simply the set of currently materialized state variables. interaction graphs are induced when a topology is applied to that materialized state.

learned effective connectivity is represented as learned process parameters over an interaction topology rather than as a separate fundamental topology.

## 4. processes

a process defines dynamics over state variables connected by one or more interaction topologies.

the canonical process schema is:

$$
\boxed{
P=(I,O,T,f,\theta)
}
$$

where:

- $I$ specifies input field components and spatial/anatomical regions
- $O$ specifies output field components and spatial/anatomical regions
- $T$ specifies the interaction topology
- $f$ specifies the dynamics
- $\theta$ contains uncertain process parameters

process definitions should avoid generic statements such as:

`neural → neural`

and instead identify precise state.

for example:

$$
I=
\{\text{layer iv excitatory activity in v1}\}
$$

$$
O=
\{\text{local layer ii/iii excitatory and inhibitory state}\}
$$

$$
T=T_{\text{laminar}}
$$

with dynamics:

$$
\dot x_O
\mathrel{+}=
f(x_I,x_O;T,\theta)
$$

multiple processes may simultaneously apply pressure to the same state:

$$
\dot{\mathbf x}
=
\sum_p f_p(\mathbf x)
$$

### what a process may write

a process applies pressure to state, and may also apply pressure to the parameters of another process:

$$
\dot\theta_{p'}
\mathrel{+}=
f(\cdot)
$$

neuromodulation does this — a modulator multiplies a target population's input–output gain rather than injecting current into it — and so does plasticity, whose output is a parameter as much as a state. this requires no additional mechanism: $\theta$ is already part of the process schema, and pressure on it composes by summation exactly as pressure on state does.

an instantaneous algebraic relation $x_O=g(x_I)$ — a quasi-static electromagnetic field determined by its source currents, an mr-observable signal determined by blood state — is the stiff limit of pressure,

$$
\dot x_O
\mathrel{+}=
-\gamma\left(x_O-g(x_I)\right),
\qquad
\gamma\to\infty
$$

and is a question of how it is solved, not of what it is.

### dynamics and their uncertainty

$(I,O,T)$ is ontology. $(f,\theta)$ is what we know about the dynamics, and it is uncertain:

$$
\theta\sim p(\theta)
$$

different initialization methods simply produce different priors.

physics or literature initialization:

$$
p(\theta)
\leftarrow
p_{\text{physics/literature}}(\theta)
$$

data-fitted initialization:

$$
\theta
\leftarrow
\operatorname{fit}(D)
$$

unknown dynamics:

$$
p(\theta)
\leftarrow
p_{\text{weak}}(\theta)
$$

a process may carry more than one candidate form of $f$ — an analytic transfer function, a mass-action or conductance-based rate law, a table or atlas lookup, a learned module — and a materialization selects among them. this is not a separate primitive: a different $f$ is a different $p(\theta)$ over a differently shaped $\theta$, and $(I,O,T)$ is untouched. nothing about the process graph presumes $f$ is linear, differentiable, closed-form, or interpretable.

most of the inventory in §5 will begin analytic and become learned as data accumulates. that transition changes $p(\theta)$ and no declaration.

$f$ may also use state to set its own effective interaction weights, since $f$ depends on state by definition:

$$
w_{ij}
=
\underbrace{\exp(-d_{ij}/\ell)}_{\text{from }T}
\cdot
\underbrace{\sigma\!\left(\langle e_i,e_j\rangle\right)}_{\text{from }f}
$$

where $e_i$ is a learned embedding of the state at $i$. the topology still specifies which state variables *may* interact; $f$ decides how strongly they *do*, which is where interaction strength was always said to live.

$\theta$ may be shared globally, per anatomical partition, as a function of a learned embedding, or held per position. this determines how many effective parameters exist and therefore what data could move them off their prior. it does not determine what exists: a per-position parameterization over $10^5$ cortical positions is a legitimate declaration even when no available dataset can distinguish its entries. the posterior simply stays near the prior, and the structure comes out smooth.

### forging

forging over heterogeneous datasets has the semantics of posterior updating:

$$
p(\theta\mid D)
\propto
p(\theta)
\prod_d p(D_d\mid\theta)
$$

the implementation may approximate this update using gradient optimization, variational inference, ensembles, distillation, or other techniques.

this common mechanism covers:

- hand-engineered initialization
- atlas and literature initialization
- simple statistical fitting
- pretrained learned processes
- heterogeneous supervised forging
- distillation
- subject-specific adaptation

### distillation precision

a teacher supplies a value where no measurement exists. that value enters as evidence
and therefore carries a precision, and the precision is not free: it is calibrated from
the teacher's own reported accuracy on the variable it is writing.

if a teacher explains a fraction $r^2$ of the variance of a state variable, the residual
variance is $(1-r^2)\operatorname{Var}[x]$ and the precision it may contribute is

$$
\Delta J_{\text{distilled}}
=
\frac{r^2}{(1-r^2)\operatorname{Var}[x]}
$$

which gives the teacher a posterior weight of exactly $r^2$: a teacher explaining a tenth
of a variable's variance moves that variable's posterior a tenth of the way. distilling
at unit precision instead is the fastest way to make a model hold a teacher's biases as
firmly as its own measurements.

**the numerator is not decoration, and an earlier version of this document omitted it.**
without it $\Delta J \to 1/\operatorname{Var}[x]$ as $r^2 \to 0$ — one prior-equivalent
of precision — so a teacher explaining *nothing* takes half the posterior and halves the
variable's variance. the error is invisible where teachers are good and dominant where
they are weak, which is exactly the regime a first forcing experiment lives in: an
auditory chain measured at $r^2 = 0.0033$ would have been credited with a shrinkage of
0.25 rather than 0.003.

two corrections are mandatory beyond the nominal figure.

**reported accuracy holds on the benchmark distribution.** used off that distribution the
precision must be inflated, and by how much is itself uncertain — so the inflation is a
process parameter with a prior, not a constant.

**a teacher's errors are correlated across everything it writes.** a model writing $10^4$
cortical positions does not supply $10^4$ independent constraints; its residuals share
structure. model the teacher's residual covariance as a diagonal part plus a low-rank
part capturing the shared error directions,

$$
R
=
\operatorname{diag}\!\left((1-\rho)\,v\right)
+
BB^{\top},
\qquad
B\in\mathbb R^{n\times q}
$$

where $\rho$ is the fraction of the teacher's error variance that is shared and $q$ is
the rank of that sharing. the precision it contributes is the inverse of that, which by
the woodbury identity is a diagonal **minus** a rank-$q$ correction:

$$
\Delta J
=
R^{-1}
=
D^{-1}-D^{-1}B\left(I+B^{\top}D^{-1}B\right)^{-1}B^{\top}D^{-1}
$$

it is worth being exact about the sign, because the intuition points the wrong way: a
teacher's shared error does not *add* a low-rank constraint, it *subtracts* the
confidence that correlated values would otherwise appear to supply.

for $q=1$ the discount has a closed form,

$$
n_{\text{eff}}
=
\frac{n}{(1-\rho)+n\rho}
\;\xrightarrow[n\to\infty]{}\;
\frac{1}{\rho}
$$

so at $\rho=0.9$ ten thousand distilled values are worth $1.11$ independent
measurements about anything they have in common, however many more of them arrive.

**this bound is rank-one and does not survive $q>1$.** a rank-$q$ basis encodes a
correlation *length* rather than a fraction shared across everything a teacher writes,
and the discount weakens sharply: at $\rho=0.9$ and $n=10^4$, rank 1 gives $1.1$
effective constraints and rank 4 gives $889$. a card that declares a higher rank hands
its teacher orders of magnitude more influence, so $q$ is a claim about the structure of
a teacher's error and not a solver setting.

the discount is also *directional*, not a blanket discount. conditional precision rises
as $1/(1-\rho)$: a correlated teacher is less informative about levels and more
informative about differences. there is therefore no safe direction in which to round
$\rho$, and a card that has not measured it must say so rather than choosing a value
that feels conservative.

a distilled value is therefore always distinguishable from a measured one in the
materialized model's provenance, and the two never carry the same weight by default.

### state uncertainty propagation

processes operate over uncertain state and uncertain parameters:

$$
\mathbf x\sim p(\mathbf x),
\qquad
\theta\sim p(\theta)
$$

and induce a new state distribution:

$$
p(\mathbf x_{t+\Delta t})
=
P_f\!\left(
p(\mathbf x_t),
p(\theta)
\right)
$$

ibm-1 projects the result back onto each component's declared form of uncertain state.

for a linear $f$ this projection is exact. for a nonlinear or learned $f$ it is not, and the general mechanism is sampling followed by moment matching. the projection is where non-gaussian structure is discarded — deliberately and visibly — and the ensemble width it requires is a real cost that scales with how much of a materialized graph is learned rather than analytic.

### gaussian evidence fusion

when external evidence contributes an approximately gaussian constraint on state, define:

$$
J=\Sigma^{-1},
\qquad
h=J\mu
$$

where $J$ is precision.

independent gaussian evidence composes as:

$$
J'
=
J+\Delta J
$$

$$
h'
=
h+\Delta h
$$

high-confidence evidence therefore contributes greater precision than low-confidence evidence.

this is distinct from the dynamical pressure applied by processes: pressure moves state, evidence constrains it, and the two compose by different rules.

## 5. initial process inventory

| process | input → output | topology |
|---|---|---|
| local excitation | local excitatory + synaptic state → nearby excitatory/inhibitory population state | local |
| local inhibition | inhibitory + excitatory state → local population state | local |
| laminar propagation | layer-specific cortical population state → other cortical layers | laminar |
| lateral cortical propagation | cortical population state → nearby cortical population state | cortical surface |
| tract propagation | source population activity + tract structure → target synaptic-input state | tractometric |
| thalamo-cortical coupling | specified thalamic populations ↔ specified cortical layers/populations | tractometric |
| ionic exchange | membrane/population state ↔ extracellular ionic state | local |
| ionic diffusion | extracellular ionic state → nearby extracellular ionic state | spatial/interstitial |
| transmitter dynamics | presynaptic population state → extracellular transmitter state → postsynaptic state | local |
| neuromodulation | source population state → extracellular modulator state → target population gain/state | projection + spatial |
| em generation | transmembrane/current state + material state → electromagnetic field | em |
| em coupling | electromagnetic field + population state → population state | em |
| neurovascular coupling | population activity + metabolic state → local blood state | local + vascular |
| vascular flow | upstream blood state → downstream blood state | vascular |
| tissue exchange | blood + metabolic/interstitial state ↔ oxygen/glucose state | vascular + local |
| bold formation | blood + material state → mr-observable state | spatial |
| metabolism | population activity + substrates → metabolic state + heat | local |
| csf flow | csf state → neighboring csf state | csf |
| csf/interstitial exchange | csf + interstitial state ↔ both | interface |
| interstitial transport | interstitial state → neighboring interstitial state | spatial |
| thermal diffusion | temperature + metabolic heat + blood → temperature | spatial + vascular |
| mechanical propagation | mechanical + material + fluid state → mechanical state | spatial |
| transduction | physical field state at a receptor surface → afferent population drive | receptor |
| afferent propagation | afferent population activity → brainstem, thalamic and primary sensory population state | afferent pathway |
| efferent propagation | motor and autonomic population activity → motor-unit and effector drive | efferent pathway |
| effector activation | motor-unit drive → muscle activation, force, and the mechanical state of the body | effector |
| device coupling | brain, body or device field state ↔ device element state | device coupling |
| plasticity | activity history + modulatory + structural state → structural state and process parameters | local + tractometric |

a process may exist in the ibm ontology without having a high-confidence $f$.

uncertainty in $\theta$ represents uncertainty in the corresponding dynamics rather than requiring ibm-1 to invent precision where the science does not provide it.

## 6. observations and interventions

observations and interventions are not additional ibm primitives.

### observation

an observation is privileged external access to a state variable.

$$
\boxed{
\text{observation}
=
\text{external evidence about state}
}
$$

for example, eeg may be represented as:

$$
\text{brain electromagnetic field}
\xrightarrow{\text{electrode coupling}}
\text{electrode voltage}
$$

followed by an externally observed measurement:

$$
p(y\mid x_{\text{electrode}})
$$

the electrode voltage remains ordinary ibm state. the coupling is an ordinary process. the observation is simply evidence constraining it.

### intervention

an intervention is externally constrained state or a process input.

$$
\boxed{
\text{intervention}
=
\text{externally constrained state/process input}
}
$$

for example, tms may constrain coil-current state:

$$
x_{\text{coil}}(t)=u(t)
$$

which then influences electromagnetic brain state through ordinary ibm processes.

the same formulation applies to:

- electrical stimulation
- sensory inputs
- pharmacological inputs
- mechanical perturbations
- invasive stimulation

## 7. explicit-model materialization

an explicit model is a lazy materialization of the ibm:

$$
M
=
\operatorname{materialize}
(
R,r,B,F,A,T,P
)
$$

where:

- $R$ specifies spatial regions
- $r$ specifies resolution over those regions
- $B$ specifies bandwidth over those regions
- $F$ specifies fields
- $A$ specifies anatomical partitioning systems
- $T$ specifies interaction topologies
- $P$ specifies processes

dependency tracing determines which state variables and processes must actually be instantiated. every process reachable from a target is materialized. the process graph is finite, so the core clique is shared across most materializations, and $R$, $r$ and $B$ are the levers by which materializations actually differ.

a materialized model carries provenance: which $f$ was selected per process, which parameters were moved off their prior by evidence and which were not, and where a process was run outside the resolution or bandwidth regime in which its $f$ is meaningful. a prediction resting on prior-dominated structure must not be presented with the confidence of one resting on constrained structure.

candidate explicit models include:

| group | models |
|---|---|
| electrophysiological forward | `eeg-forward`, `meg-forward`, `ecog-forward`, `lfp-forward`, `csd-laminar` |
| electrophysiological inverse | `eeg-source`, `meg-source`, `eeg-predict` |
| decoding | `eeg-to-image`, `meg-to-text`, `speech-envelope`, `invasive-bci`, `spike-decode` |
| hemodynamic | `bold-forward`, `fmri-infill`, `hrf`, `fnirs-forward` |
| stimulation response | `tms-response`, `tes-response`, `tfus-response`, `dbs-response` |
| state and disorder | `seizure-propagation`, `sleep-dynamics`, `anesthesia`, `pharmaco`, `virtual-lesion` |
| slow and physiological | `glymphatic`, `thermal-safety`, `plasticity` |
| surrogate | `macro-surrogate`, `resting-state-fc` |

all are materialized views of the same implicit model rather than independently defined brain models.

the training trajectory that puts all of it to work is
[CURRICULUM.md](CURRICULUM.md).

what the four primitives are ultimately FOR -- cognition as the geometry of a
vector field, and the mechanism inventory ibm-1 is measured against -- is in
[DYNAMICS.md](DYNAMICS.md).

the sources these are fitted and evaluated against are inventoried in
[EVIDENCE.md](EVIDENCE.md), and their cards live in `data/sources/`.

where the effort actually stands -- what has been verified by running, which declared claims have been overturned by measurement, and which concerns are open -- is recorded in [STATE.md](STATE.md).

## 8. organization

the codebase mirrors the four primitives. nothing else may become a peer of them.

```
ibm/
  registry.py          one namespace: register, resolve, validate, print
  vocabulary.py        controlled ids, synonym table, collision refusal
  frames.py            coordinate frames

  fields/              F = {x(q) : q in Omega}
    supports.py          the domains fields are indexed over
    uncertainty/         how a belief about a component is carried
      scalar.py
      spectral.py        the temporal laplacian
    neural.py  extracellular.py  electromagnetic.py  blood.py  csf.py
    metabolic.py  structural.py  material.py  thermal.py  mechanical.py
    transduction.py  effector.py  device.py

  anatomy/             a(q) in [0,1]^k
    systems.py  sources.py

  topologies/          T(i,j)
    builders.py
    local.py  surface.py  laminar.py  tract.py  vascular.py
    interstitial.py  csf.py  em.py  mechanical.py  metabolic.py
    afferent.py  efferent.py  device.py

  processes/           P = (I, O, T, f, theta)
    neural.py  ionic.py  transmitter.py  neuromodulation.py  electromagnetic.py
    vascular.py  metabolic.py  csf.py  thermal.py  mechanical.py  plasticity.py
    transduction.py  effector.py  device.py
    observation.py  intervention.py
    nn.py                reusable learned building blocks for f

  materialize/         M = materialize(R, r, B, F, A, T, P)
    request.py  trace.py  build.py  model.py  provenance.py  cache.py
    library/             the named explicit models

  runtime/             executing a materialized model
    state.py  step.py  fuse.py

  forge/               p(theta | D)
    fit.py  priors.py  bind.py     # binds data/sources cards to registered components
```

four placements follow from the architecture rather than from convenience:

- **there is no `observe/` directory.** §6 states that observations and interventions are not primitives; giving them peer status would contradict that. electrode coupling is an ordinary process, and the measurement likelihood is evidence.
- **there is no `spatial/` directory.** $\Omega$ belongs to fields, atlas sources belong to anatomy, and $\operatorname{grid}(R,r)$ belongs to materialization. nothing coherent remains.
- **forms of uncertain state live under `fields/`.** how a belief is carried is part of the field declaration, not a concept alongside the four. the temporal laplacian is one file, two levels down; if it were removed the ontology would not notice.
- **candidate forms of $f$ live beside their process declaration.** they are alternative dynamics for one process, not a parallel tree.

### the registry

the way an ontology of this size normally fails is not that it is wrong. it is that after two years there are hundreds of components and processes declared across dozens of files, several pairs of which are the same thing under different names and several more of which almost are, until no one can say which processes are actually coupled.

so nothing exists unless it is registered:

- every component and every process is declared exactly once, in one registry
- ids are checked against a grammar and a controlled vocabulary
- two ids that normalize to the same token multiset are refused, not warned about
- $I$ and $O$ name registered components; a typo is a declaration error, not a silently empty coupling
- sealing the registry rejects unregistered topologies, dead components, writes above a component's declared band, and any process coupling two forms of uncertain state with no declared conversion between them
- every lossy conversion declares what it destroys, and appears in materialization provenance

the whole ontology must be printable as one table. if it is not, it has already begun to sprawl.

## 9. core schema

$$
\boxed{
\begin{aligned}
\text{field} &= \text{indexed uncertain state}\\
\text{anatomy} &= \text{spatial partitions}\\
\text{topology} &= \text{possible interactions}\\
\text{process} &= \text{dynamics over interactions}
\end{aligned}
}
$$

and:

$$
\boxed{
\text{explicit model}
=
\text{lazy materialization of the ibm}
}
$$
