```markdown
# implicit brain model (ibm-1)

the implicit brain model (ibm) is a multiscale probabilistic dynamical substrate from which task-specific explicit brain models are lazily materialized.

$$
\boxed{
\mathrm{ibm}
=
(\text{fields},\ \text{anatomy},\ \text{topologies},\ \text{processes})
}
$$

resolution, datasets, observations, interventions, and explicit models are expressed through these four primitives rather than introduced as additional primitives.

## 1. fields

a field is an indexed family of state variables:

$$
F=\{x(q):q\in\Omega\},
\qquad
x(q)\in\mathbb R^d
$$

where:

- $\Omega$ is the spatial support of the field
- $q$ is a position on that support
- $x(q)$ is the state associated with that position
- $d$ is the number of state components at each position

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

### canonical uncertain state

for a materialized field, concatenate its state variables into $\mathbf x$ and represent them in a laplacian basis:

$$
\mathbf x=Uz
$$

$$
z\sim\mathcal N(\mu,\Sigma)
$$

with

$$
LU=U\Lambda
$$

where $L$ is the laplacian associated with the materialized spatial support or interaction topology.

gaussianity describes ibm-1's epistemic representation of uncertain state, not a claim that the underlying biological state is gaussian.

the spectral representation gives ibm-1 explicit control over:

- spatial bandwidth
- uncertainty
- interpolation
- multiresolution truncation
- heterogeneous evidence fusion
- lazy computation

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

additional fields should only be introduced when they represent state that cannot cleanly be expressed as components of an existing field.

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

interaction strength and dynamics generally belong to processes rather than to the topology itself.

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

the state graph is simply the set of currently materialized state variables.

interaction graphs are induced when a topology is applied to that materialized state.

learned effective connectivity is generally represented as learned process parameters over an interaction topology rather than as a separate fundamental topology.

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

### process uncertainty

process parameters are uncertain:

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

ibm-1 approximates the resulting state using its canonical laplacian-spectral gaussian representation.

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

this is distinct from the dynamical pressure applied by processes.

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
| plasticity | activity history + modulatory + structural state → structural state and process parameters | local + tractometric |

a process may exist in the ibm ontology without having a high-confidence implementation.

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

the electrode voltage remains ordinary ibm state.

the observation is simply evidence constraining it.

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
R,r,F,A,T,P
)
$$

where:

- $R$ specifies spatial regions
- $r$ specifies resolution over those regions
- $F$ specifies fields
- $A$ specifies anatomical partitioning systems
- $T$ specifies interaction topologies
- $P$ specifies processes

dependency tracing determines which state variables and processes must actually be instantiated.

an explicit model therefore materializes only the field components, anatomical information, interactions, and dynamics required to produce or constrain its target state.

examples include:

- `ibm-1-eeg-predict`
- `ibm-1-eeg-to-image`
- `ibm-1-meg-to-text`
- `ibm-1-fmri-infill`
- `ibm-1-invasive-bci`
- `ibm-1-tms-response`
- `ibm-1-macro-surrogate`

all are materialized views of the same implicit model rather than independently defined brain models.

## 8. core schema

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
```
