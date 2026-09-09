# progress against the objectives

Milestones with the measurement that earned them, and what is still open under
each objective. A milestone goes on this page when a number exists; a plan does
not.

Objectives are those in `docs/DIRECTION.md`. Component-level status is in
`docs/DEVELOPMENTAL_COMPONENTS.md`. Where the running model and the declared
model differ is `docs/DISCONNECTS.md`.

---

## A. The body is real

**Reached**

- **The anatomy is bound to the driven skeleton.** 3,995 of 4,000 entities
  (99.88%). Gates that are not the assignment rule restated: held-out bones 1.6%
  misassignment, 80 native muscles against the kinematic chain 2.5%, hand-checked
  21/21 including organs, arteries and tendons, independent second opinion 93.5%.
- **The real anatomy moves.** Up to **2,234 mm** of entity travel, median 1,659
  mm, against 6.35 mm across 30 s for the body's own canonical run — three orders
  of magnitude. Rendered: `artifacts/anatomical_crawl.mp4`.
- **Height is a gated parameter, and it now reaches the whole body.** It scaled
  only the 22-segment scaffold: joint frames, meshes, all 80 fitted path
  polynomials, fibre and tendon slack, contact spheres, mass (s³) and inertia
  (s⁵). It now also scales the **4,000 anatomical entities**, the **146 nerve
  routes** and their **1,743 conduction delays**, the **1,326 skin patches**, the
  98 actuators' PCSA/force/mass and the 117 ligament force elements — **48 gates
  at 2.03 m, 52 at the identity**, and identity reproduces the base *exactly*,
  not to a tolerance. No exponent in `ihm/body_scaling.py` is typed: each is
  computed from a dimensional formula over primitives that are 1 (a length) or 0
  (a material property), and the table is checked against answers fixed outside
  it and against itself. Ranges from NHANES (4,883 adults), validated against
  published CDC means to 0.1 cm.
- **A taller body has a slower periphery — the first physiological consequence of
  a body parameter.** Conduction velocity is set by axon diameter and internodal
  myelin; route length is anatomy. So every conduction delay lengthens in exact
  proportion to stature: at 2.03 m the **vagal C fibre goes 507.84 → 573.60 ms**,
  the optic nerve's three retinal populations 3.28/5.47/10.95 → 3.71/6.18/12.36
  ms, and their spread — one nerve, three velocities — separates 7.66 → 8.65 ms.
  All named in advance. This is the result that says the parametrization reached
  the body rather than a scale factor in a config, and it matters because the
  delay-lumping ablation under C already showed delays are load-bearing.
- **Geometric similarity is measurably false, and the six places are written
  down.** Over 4,822 survey-weighted NHANES adults, mass goes as stature^2.034 ±
  0.083, not ^3 — isometry rejected at z = −11.6; BMI as stature^0.034 where
  isometry demands 1.0. Reported and *not* applied. Five more in `ALLOMETRY`,
  each with the size of its error: vascular flow 9.6% over-perfused at 2.03 m
  (Poiseuille's r⁴ beats the length), brain volume 1.44× with no catalogued
  source to correct it, characteristic time s^0.5 under equal Froude number,
  passive joint stops unscaled, and receptor density s^−2 by choice.
- **Real bone meshes can carry contact, and concavity survives.** `ContactMesh`
  over Simbody `TriangleMesh`; the rib cage encloses **4.3%** of its convex
  hull's volume and is not hulled by the solver. Cost 1.15× spheres; mesh
  geometry is nearly free until it touches.

**Open**

- **A deformable solver.** Nothing in the stack has one — every compliant element
  is a 1-D spring law on a rigid carrier. Blocks the participant mode's
  soft-tissue clause. **Build-or-defer decision, not work.**
- **Skin registration.** Neither available map encloses the bones (0.273 / 0.445
  of bone vertices inside their own skin; 0/20 segments enclosing). Hidden until
  now because the supine foundation sets its support plane to the skin's own
  minimum, so the floor follows the skin and the error is invisible by
  construction. Needs per-segment geometric transformation.
- **Tissue structures as mechanics — 117 of 645 now carry force, and the reason
  the rest do not is different for each class.** Counted in the bound body, with
  what the plant carries after `scripts/build_tissue_force_elements.py`:

  | | entities | inside ONE scaffold body | force elements |
  |---|---:|---:|---:|
  | ligament | 300 | 195 | **105** |
  | bursa / synovium | 80 | 68 | 0 |
  | meniscus / intervertebral disc | 69 | 59 | 0 |
  | cartilage | 51 | 49 | 0 |
  | tendon (36 are sheaths) | 44 | 8 | via muscle only |
  | fascia / aponeurosis | 44 | 19 | 0 |
  | joint capsule | 36 | 24 | **12** |
  | retinaculum | 18 | 8 | 0 |
  | adipose | 2 | 0 | 0 |
  | skin | 1 | 0 | rigid quadrature only |
  | periosteum | 0 | — | — |
  | **total** | **645** | **430** | **117** |

  Ligaments were recorded as blocked structurally, and they were not: the binding
  gives each entity one segment, but the per-vertex vote it is built from
  partitions the surface between two, and each side's tip centroid is an
  attachment. Gated 30/30 on named bone pairs including three negative controls
  (sacrotuberous, sacrospinous and inguinal must come out ONE segment because
  both their bones are `pelvis` here, and they do), six published lengths at
  0.70–1.04x, three published Blankevoort stiffnesses at 0.44–2.03x, and two
  independent implementations of the path length agreeing to 3.3e-16 m.

  Standing weight stays 761.3757 N in every arm and the momentum residual stays
  at its relative floor, because these are internal forces. What they do to the
  plant, on **three** prone drops rather than one — worst excursion past the
  model's declared ranges:

  | | prone | high | rolled | mean |
  |---|---:|---:|---:|---:|
  | bare | 17.43° | 30.69° | 30.27° | 26.13° |
  | joint stops at 30 N·m/rad | 5.30° | 6.62° | 5.96° | 5.96° |
  | all 105 ligaments | 32.78° | 35.33° | 32.88° | 33.66° |
  | the 66 admissible | 6.08° | 30.47° | 18.50° | 18.35° |
  | stops + the 66 | **4.05°** | **6.39°** | **5.35°** | **5.27°** |

  **The full set makes the plant worse, three drops out of three.** A straight
  line between two attachment centroids is not a ligament's path: the derived ACL
  reads 77% strain at 90° of knee flexion against a 17.1% ultimate, because a
  real cruciate is near-isometric only because it wraps. The 51 that fail that
  check are the cruciates, the collaterals and the ankle ligaments, and closing
  them needs a wrap surface per joint.

  **The 66 that pass it do not replace the joint stops** — ledger row 28, which
  is what one drop said and three drops withdrew. What holds: they are never
  worse than the bare plant where the unfiltered set is always worse, and added
  to the stops they improve the worst excursion on every drop, 5.96° → 5.27°.

  **430 of the 645 are inside one rigid body** — 23 intervertebral discs and 23
  nuclei in `torso`, 29 ligaments per hand — so the scaffold has no joint where
  they act. **Cartilage is a DATA gap**: 51 cartilage entities and not one is a
  joint surface (34 costal, 8 laryngotracheal, 7 nasal, 2 growth plate), and
  bone-on-bone contact cannot substitute because a synovial joint's bone surfaces
  overlap by construction — the hips interpenetrate in 19/21 and 21/21 sampled
  configurations of their own declared range, and 11 of 21 joints interpenetrate
  somewhere in theirs. **Retinacula are blocked on the muscle path**: 80 of 98
  muscles run as fitted polynomials with no geometry, so not one of the 18 can
  constrain anything. **Adipose is 1.26 mL of geometry in the whole body and
  periosteum is 0 entities** — acquisition, not modelling, and the same gap as
  the declared skin needing ~22 mm of indentation to hold 761 N against its own
  6.6 mm thickness.
- Sex parametrization: no catalogued source ships a female mesh; blocked behind a
  path-refitting tool.

## B. The brain touches the body only through nerves

**Reached**

- **214 of 214** contractile channels innervated; 192 under spinal reflex arcs;
  22 cranial with no spinal segment, enumerated by nerve; **0 uninnervated**.
- **1,326 skin patches** covering **100.00%** of the 1.7805 m² exterior, 0
  triangles unassigned; all innervated; 1,199 with spinal roots, 127 trigeminal
  with a stated reason. 29 of 30 dermatome levels — C1 absent because it has no
  cutaneous territory in life.
- **72 of 72 nerve trunks** with a proximal end, a distal end and a measured
  route length. Three modules had carried three different silent default lengths.

**Open** — no peripheral nerve mesh outside the orbit; dermatome assignment is an
authored prior over measured geometry, not a registered atlas; patch density is
by area, not receptor density — and stature scaling deliberately holds the patch
COUNT at 1,326 and lets area go as s², so density falls 21.6% at 2.03 m. That
keeps the brain's afferent channel count invariant under body size, which is the
reason for the choice; it does not make the underlying density measured.

## C. The brain is mesoscale and real

**Reached**

- **Signal crosses the sheet.** Concentrating the long-range budget with
  task-blind spatial diversity: **209×** transport, and end-to-end training goes
  3.2× → **43.9×** chance, which is **91% of the frozen-head ceiling** (48.4×).
  This was the central blocker of the session.
- **The dynamics are necessary and their learned content is not.** Severing takes
  disjoint retrieval to 1.00× chance with effective rank 1.0; a permuted kernel
  reads 50.75× against intact's 48.37×. Confirmed on three independent pathways:
  motor, vision, viscera.
- **Transported signal has effective rank 12.9 of 2,012** — about thirteen
  dimensions. The tightest real constraint on any somato-motor design.
- **Peripheral conduction delays are load-bearing.** Lumping every delay to step
  0 while keeping all 15 visceral channels costs −0.124, as much as deleting an
  entire fibre group. First result where the declared peripheral anatomy matters.
- **A nociceptor exists.** Silent through the innocuous range (exactly 0 Hz at
  5 N), 51.2 Hz at a 40 N crush, and sensitisation makes the innocuous painful.

**Open** — the cortex is a sphere and the real atlases were never fetched; the
declared tract topology is unused by the kernel; the TCT loop is a driven relay
rather than a closed loop.

## D. The body moves

**Reached**

- **16.0 seconds of sustained locomotion, 0.920 m**, full horizon, no stall. Peak
  fibre velocity 9.69 of 10 — **zero time outside the muscle model's domain** —
  worst joint excursion **11.5°** against 145° for the withdrawn attempt and
  25.1° for the standing gait.
- **The engine no longer collapses.** 26 declared `CoordinateActuator` ports had
  no controller connected, leaving the arm chain a rag doll at 23.9 optimal fibre
  lengths/s against a maximum of 10. Ports held: 180 s wall → 18.5 s.
- **Joint stops exist and are free.** At 30 N·m/rad they are *half* the wall clock
  of no stops at all and hold the plant 6.6× closer to its declared range.
- **And the body's own tissue now takes a little of that load off the constant.**
  66 derived elements, stiffness from the declared ligament modulus and
  attachments from the structures' own surfaces, added to the stops: 5.96° → 5.27°
  mean worst excursion over three drops, same sign 3/3. They do not replace the
  stops — one drop said they did and three drops withdrew it (ledger row 28).
- **68 forced-pose motions, 17,622 frames** of what the real muscles experience.

**Open** — nothing the brain produced; this is the crude controller and the
anatomy posed from it.

## E. The world

**Reached**

- **The body can feel an object.** Ball dropped: lands on the chest (47.2 N),
  rolls to the pelvis (**218.1 N**), settles between the thighs. Pushed: ends
  **0.761 m** away. Control with the ball out of reach: **0 contact frames, 0.000
  m displacement.**

**Open** — the ball touches an inertia-inscribed proxy, not a chest; the rendered
surround reaches the physics not at all; the engine ground is a flat half-space
so rooms cannot be an environment and terrain cannot be uneven.

## F. The full materialization

Sensory in works on corpora (optic 43×, cochlear 7.5× chance) and not on a
rendered world. Body state in: 15 visceral channels reaching cortex. Motor out:
the cord exists, the brain's content does not drive it. **Speech out: not
started.**

## G. Agentic training

Not started, correctly gated. Pain-guided RL now has its receptor; it still needs
pleasure, and a body with only nociception can learn to avoid and cannot learn to
seek.

---

## What the ledger says about all of this

**28 claims withdrawn**, and they share one shape: a quantity computed correctly
and compared against the wrong thing. Several on this page exist because a gate
caught something that looked like a result — the 973 mm crawl with the ankles
folded 145°, the video model that was matching appearance rather than predicting,
the checkpoint that could not reproduce its own log. The withdrawal rate is not a
problem with the programme; it is the programme working.
