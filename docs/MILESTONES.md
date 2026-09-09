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
- **Height is a gated parameter.** Scales joint frames, meshes, all 80 fitted
  path polynomials, fibre and tendon slack, contact spheres, mass (s³) and
  inertia (s⁵). Identity at s=1.0 reproduces the base exactly; five scales
  0.80–1.13 monotone. Ranges from NHANES (4,883 adults), validated against
  published CDC means to 0.1 cm.
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
- **Tissue structures as mechanics.** They exist as geometry and not as force
  elements. Counted in the bound body:

  | | entities | force elements in the plant |
  |---|---:|---:|
  | ligament | 311 | **0** |
  | bursa / synovium | 80 | 0 |
  | cartilage | 51 | 0 |
  | fascia / aponeurosis | 46 | 0 |
  | tendon | 44 | via muscle only |
  | joint capsule | 36 | 0 |
  | meniscus / intervertebral disc | 33 | 0 |
  | retinaculum | 18 | 0 |
  | skin | 4 | rigid quadrature only |
  | adipose | 2 | 0 |
  | periosteum | 0 | — |
  | **total** | **625 of 4,000** | |

  Ligaments are blocked structurally: the binding gives each entity exactly one
  segment and a ligament spans two — 123 of 328 report a runner-up segment
  different from their assigned one, which is the binding itself saying they
  straddle a joint. Cartilage, menisci and discs need contact and compliance
  between bones, which is the same solver question. Adipose at 2 entities is a
  data gap, not just a mechanics gap, and the declared skin needs ~22 mm of
  indentation to hold 761 N against its own 6.6 mm thickness — **the skin alone
  cannot hold the body up and nothing carries a thickness for the fat and muscle
  that would.**
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
by area, not receptor density.

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

**27 claims withdrawn**, and they share one shape: a quantity computed correctly
and compared against the wrong thing. Several on this page exist because a gate
caught something that looked like a result — the 973 mm crawl with the ankles
folded 145°, the video model that was matching appearance rather than predicting,
the checkpoint that could not reproduce its own log. The withdrawal rate is not a
problem with the programme; it is the programme working.
