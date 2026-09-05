# build contract

the exact names every module must use. this exists so that modules written
independently agree. deviating from it produces a registry collision or a dangling
selector at seal time, not a silent bug — but the collision is the *symptom*; the
contract is the fix.

ids follow `ibm.vocabulary`: dotted lowercase snake_case, and two ids that
normalize to the same token multiset are **refused** at registration.

## fields → components

### neural — support `tissue` (overridable), uncertainty `spectral`
```
neural.exc.potential        neural.exc.activity        neural.exc.adaptation
neural.exc.ampa             neural.exc.nmda
neural.inh.potential        neural.inh.activity
neural.inh.gaba_a           neural.inh.gaba_b
neural.pv.activity          neural.sst.activity        neural.vip.activity
neural.transmembrane_current
neural.afferent.activity    neural.efferent.activity
```

### extracellular — support `interstitial`, uncertainty `scalar` (ions `spectral`)
```
extracellular.k             extracellular.na           extracellular.ca
extracellular.cl            extracellular.ph
extracellular.glutamate     extracellular.gaba
extracellular.dopamine      extracellular.serotonin    extracellular.acetylcholine
extracellular.noradrenaline extracellular.adenosine
extracellular.osmolarity    extracellular.volume_fraction
```

### electromagnetic — support `head_volume`, uncertainty `spectral`
```
electromagnetic.potential   electromagnetic.efield
electromagnetic.current_density  electromagnetic.bfield
```

### blood — support `vascular_tree`, uncertainty `scalar`
```
blood.flow  blood.volume  blood.pressure  blood.oxygenation
blood.deoxyhemoglobin  blood.oxygen_content
```

### csf — support `csf_space`, uncertainty `scalar`
```
csf.pressure  csf.velocity  csf.solute
```

### metabolic — support `tissue`, uncertainty `scalar`
```
metabolic.oxygen  metabolic.glucose  metabolic.atp
metabolic.lactate  metabolic.consumption  metabolic.heat
```

### structural — support `tissue`, uncertainty `scalar`, band `STRUCTURAL`
```
structural.fiber_orientation  structural.axonal_density  structural.myelination
structural.synaptic_density   structural.dendritic_density  structural.gliosis
```

### material — support `head_volume`, uncertainty `scalar`, exogenous
```
material.conductivity  material.permittivity  material.mass_density
material.stiffness     material.tortuosity    material.porosity
```

### thermal — support `head_volume`, uncertainty `scalar`
```
thermal.temperature
```

### mechanical — support `head_volume`, uncertainty `scalar`
```
mechanical.displacement  mechanical.velocity  mechanical.stress
mechanical.strain        mechanical.pressure
```

### transduction — receptor supports, uncertainty `spectral`
```
transduction.photoreceptor      transduction.hair_cell
transduction.mechanoreceptor    transduction.thermoreceptor
transduction.nociceptor         transduction.chemoreceptor
transduction.baroreceptor       transduction.vestibular
transduction.adaptation
```

### effector — support `motor_units`, uncertainty `spectral`
```
effector.drive  effector.activation  effector.force  effector.fatigue
```

### device — instrument supports, uncertainty `spectral`
```
device.contact_potential  device.impedance      device.coil_current
device.transducer_drive   device.channel_gain   device.sequence_phase
device.display_luminance  device.speaker_pressure
```

## topologies
```
local              cortical_surface   laminar          microcircuit
tractometric       vascular           csf              interstitial
electromagnetic    mechanical         metabolic_exchange
afferent_pathway   efferent_pathway   device_coupling
neuromodulatory_projection
```

## processes
```
local_excitation           local_inhibition          laminar_propagation
lateral_cortical_propagation  tract_propagation      thalamocortical_coupling
ionic_exchange             ionic_diffusion           transmitter_dynamics
neuromodulation            em_generation             em_coupling
neurovascular_coupling     vascular_flow             tissue_exchange
bold_formation             metabolism                csf_flow
csf_interstitial_exchange  interstitial_transport    thermal_diffusion
mechanical_propagation     plasticity                transduction
afferent_propagation       efferent_propagation      effector_activation
device_coupling
```

## anatomical partitioning systems
```
cortical_areas      cortical_layers      cytoarchitecture
thalamic_nuclei     hippocampal_subfields  striosome_matrix
bg_territories      hypothalamic_nuclei  amygdalar_nuclei
cerebellar_lobules  cerebellar_microzones  brainstem_nuclei
vascular_territories
```

## house style

- lowercase prose in docstrings; explain **why** a choice was made, not what the
  code does
- priors carry real literature values where they exist and `weak()` / `speculative()`
  where they do not — never invent precision the science has not provided
- `Provenance` is honest: `PHYSICS`, `LITERATURE`, `ATLAS` only when true
- a process with no defensible implementation still gets declared, with `weak()`
  parameters. ARCHITECTURE.md §5: a process may exist in the ontology without a
  high-confidence f
- no tests, no test files

## observations (27)

an observation is external evidence about ordinary state (ARCHITECTURE.md §6), not a
primitive. these ids are the ones a materialization may name. **this list was missing
from the first version of this contract, and the omission cost real work**: the model
library invented 39 descriptive names (`eeg_scalp_potential`, `bold_signal`,
`threshold_crossings`) against 22 registered short ones, and every one had to be
reconciled by hand afterwards. fixing component ids here but not observation ids was an
arbitrary line.

```
asl_perfusion               behaviour                   bold                        csf_flow_velocity
dc_potential                drug_concentration          dwi_microstructure          ecg
ecog                        eeg                         emg                         eye_tracking
fnirs                       ieeg                        intracortical_spikes        lfp
meg                         mep                         motion_capture              pet
polysomnography             ppg                         produced_audio              respiration
structural_mri              temperature                 tissue_displacement
```

## interventions (20)

```
anaesthetic                 auditory_stimulus           dbs                         graph_ablation
invasive_electrical_stimulationolfactory_stimulus          optogenetic_stimulation     pharmacological
respiratory_challenge       sensory_deprivation         tacs                        tactile_stimulus
task_cue                    tdcs                        tfus                        thermal_stimulus
tms                         trns                        vestibular_stimulus         visual_stimulus
```

two of these deserve their names read carefully. `task_cue` clamps the delivery of an
instruction and records that the manipulation of interest — the participant's compliance
— is *not* the clamped variable, because ibm-1 has no state for an intention.
`graph_ablation` is not a physical intervention at all: a virtual lesion imposes zero
coupling on state the model contains, and nothing was done to any participant.
