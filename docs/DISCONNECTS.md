# where a declared model and the running model are different objects

Measured, not surveyed. Each row is a thing this programme declares, catalogues
or renders, and which the code that actually produces results does not use. They
are not bugs; they are gaps between the ontology and the running system, and each
one is a place where a result could be quietly about the wrong object.

## 1. The skeleton the brain drives is not the body you see

| | |
|---|---|
| declared / rendered | 3,816 BodyParts3D entities in the simulated body; 2,229 surfaces in the display atlas |
| actually driven | 22 OpenSim bodies, 80 muscles |
| mapping between them | **none** |

The anatomical entities take their transforms from the native run, not from the
OpenSim skeleton, so brain-driven motion cannot reach them. Every embodied video
so far is the 22-body skeleton, which is why it looks like capsules.

Also: the anatomical body's own canonical trajectory moves a maximum of **6.35 mm**
across 30 s. It is breathing and perfusing, not moving.

## 2. The cortical sheet is a sphere, and the real atlases are not held

`cortical_sites()` places sites on a **spherical shell** area-matched to the
measured white surface, and `cortical_regions()` says so plainly — "a geometric
convention on the spherical proxy, NOT an atlas". Six lobe labels are cut by
coordinate thresholds.

The consequence is not cosmetic: **the insula is not separable on a sphere**,
because a sphere has no lateral sulcus, so the interoceptive port enters a 4.5%
subsample of the `frontal` label instead.

Real parcellations are catalogued — `desikan2006`, `dkt-atlas` — and both hold
**one file, 12 KB, a checksums.txt**. The payloads were never fetched.

## 3. The white-matter tracts are declared and unused — and this one bit today

`ibm/topologies/tract.py` declares tractometric adjacency: which cortical regions
are joined by which fascicle, and the **conduction delay** of each. Its docstring
argues at length that this metric is the right one for long-range cortical
connectivity and that euclidean and geodesic both get it wrong.

The trained kernel draws its long-range partners with `torch.randint` —
**uniformly at random over all sites**. No training or evaluation script imports
the tract topology. The HCP connectome source backing it is also checksums-only.

This is the sharpest one, because a whole session went into the long-range edges:
measuring that they carry ~1/300 per hop, that concentrating them onto 4 partners
per site buys 209x transport, and that spatial diversity among those partners
raises multimodal convergence. All of that was spent choosing among **random**
partners, while a declared topology specifies which partners should exist at all.

## 4. Seventy-two nerve trunks — now with two ends each — CLOSED as declared

*Was: 71 trunks with a fibre composition and a length, and no ends. A trunk had
no proximal root and no distal target, so it was a routing convention over names
rather than a cable running from somewhere to somewhere.*

`TRUNK_ROOTS` and `TRUNK_TARGET` now give every one of the 72 trunks a proximal
end (spinal levels, or a cranial numeral) and a distal end. Writing them down
surfaced **eleven contradictions** against `INNERVATION`, all of which are fixed
and all of which are now guarded by `tests/test_innervation_coverage.py`: erector
spinae was hanging off the thoracoabdominal (ventral) nerves when it is supplied
by dorsal rami, psoas major was on the femoral nerve when it is supplied by the
L1–L3 rami directly, and nine root lists disagreed on one side or the other.

A 72nd trunk was added: **`dorsal_ramus`**. Every one of the original 71 was a
*ventral* ramus, so the paravertebral skin strip and the whole deep back had no
declared trunk at all.

`INNERVATION` also named nine nerves that were *branches* — `facial_vii_somatic_
motor`, `vagus_x_recurrent_laryngeal` — which were keys in neither trunk table.
`trunk_of()` resolves them; before it, every extraocular, laryngeal, tongue and
facial muscle in the model shared one **invented 200 mm** conduction delay.

**Still open:** the body carries no peripheral nerve *mesh* outside the orbit.
Of 146 entities with role `nerve` in `anatomy.json`, 34 are real BodyParts3D
orbital branches and 112 are CNS structures mis-roled. There is no median or
sciatic geometry; the routes are authored schematic centrelines with
`measured_axon_geometry: false`, which IHM says plainly on every record.

## 5. Skin is three whole-body layers — CLOSED

*Was: three entities — epidermis, dermis, hypodermis — for the entire
integument, so a cutaneous afferent had nothing to innervate at a location.*

IHM-1's `scripts/build_dermatome_patches.py` cuts the exterior component of the
real skin mesh into **1,326 patches**, each with a position on the surface, an
area, a normal, a dermatome, a dorsal root and a named trunk.

| | |
|---|---|
| area covered | 1.7805 m² of the 1.7805 m² exterior component — **100.00%**, 0 triangles unassigned |
| denominator | the exterior component. The raw mesh is 3.5026 m² and includes interior and orifice surfaces |
| with a spinal root | **1,199 of 1,326** |
| trigeminal, no spinal root | **127 of 1,326** (0.0688 m²), reported apart and never given one |
| dermatomes present | 29 of 30 spinal levels. C1 has no cutaneous territory in life |

**Still a prior, and flagged as one.** `measured_dermatome_atlas` is `False`.
Region, T2–T12 on the trunk wall (from the nearest rib), the dorsal/ventral ramus
split (from vertebra vs rib) and C6/C7/C8 on the hand (from the named ray) are
measured off this body's bones; every limb sector and the facial bands are
authored. Each patch records which rule produced it. Dermatomes overlap by about
one segment in life and this partition is exclusive.

## 6. Muscles moving without spinal arcs — CLOSED to 192 of 214

*Was: "of 98 catalogued muscles, 66 map to a declared nerve and root level".*
That denominator was stale. The body's binding list now holds **249 channels**,
and the honest split is three-way rather than "mapped/unmapped":

| | |
|---|---|
| channels in the body's binding list | 249 |
| not muscles at all — tendon sheaths, tendons, check ligaments | 35, excluded by name |
| **contractile channels** | **214** |
| under spinal reflex arcs | **192 of 214** |
| cranial nerve, innervated, correctly no segment | **22 of 214** (extraocular, tongue, platysma, stylohyoid) |
| **no declared innervation at all** | **0 of 214** |

Closing it needed 22 new entries in `INNERVATION` (96 → 118 muscles) and one
real bug fixed: `scripts/embody.py` called `SegmentalCord` **without**
`muscle_bindings`, so the cord could only match the 72 OpenSim channels by name
and gave arcs to 66 of 249. The loop ran either way and never said which.

**Still open:** the intercostal and erector-spinae entries write their root
ranges as endpoints — `("t1", "t11")` — and the cord reads them as two literal
levels rather than a span, so those muscles recruit 2 segments where the anatomy
says 11.

## 7. The gait reference is two different people

`ihm/native/gait_reference.py` says it in its own docstring: the Rajagopal
coordinates and the Gait2392 CMC excitations are **different subjects and
different trials**. Any corpus using both carries that seam.

## 8. The thalamo-cortical loop is not a loop

Measured: it oscillates at 6.92 Hz with no stimulus, and severing the
thalamo-cortical projection takes the cortical swing 39.49 Hz to exactly 0. But
cutting the *descending* limb leaves the rhythm standing — it is a driven relay
with feedback, not a closed loop — and it lands in theta, not the spindle band
the declaration claims for it.

## 9. The body cannot feel the world it is rendered in

**Surfaced by the user, who noticed the app clearly has environments while the
body clearly does not experience them.**

The app declares an environment catalogue — `studio`, `floor`, `bed` — with
tiles, thumbnails and scene objects, and renders the body inside them.

The body's **entire** contact experience, read from a gait trajectory frame:

    foot_load_fraction     {r, l}
    foot_contact_force_n   {r, l}
    foot_centre_m          {r, l}
    fall_support_force_n   scalar

Two feet and a fall-catch plane. No objects, no surfaces, no contact anywhere
else on the body. `NativeMechanicalStream` takes `environment` as a three-valued
enum (`free`/`supine`/`upright`) plus an optional bed material — the rendered
scene reaches the physics not at all.

Two consequences, and the second is the sharp one:

- Nothing in a rendered environment can be touched, pushed, sat on or bumped
  into. The world is a backdrop.
- **Prone locomotion is unrepresentable.** A crawling body has no contact on
  hands, knees, forearms, shins or torso, so it would pass through the floor
  everywhere except its soles. Crawling is not untuned here, it is impossible,
  and `fall_support_force_n` is a surface that ENDS a run rather than one that
  can bear weight.

There is machinery that is not being used: `surface_contact_manifest` and
`surface_sensor_indices` are constructor arguments on the stream, and
`ihm/assembly/surface_binding.py` exists. Whether extending contact is a
configuration change or a build is not yet known.

---

## What this list is for

Two of these (1, 3) are load-bearing for what the programme is trying to
demonstrate next, and 3 is the one that would change how the current cortical
work is done rather than merely extending it. The rest are honest scope: things
declared ahead of being built, which is fine as long as no result quietly claims
them.

`scripts/measure_innervation_coverage.py` prints 4, 5 and 6 with their
denominators and is the thing to re-run rather than re-reading the numbers above.

One more disconnect closed along the way, and it belongs here because it is the
same shape. **Three different modules carried three different silent defaults for
a trunk with no declared length** — 300 mm in `nerve.py`, 200 mm in
`embodiment.py`, 50 mm in `pretrain_video_loop.py` — so the same unnamed trunk
got three different conduction delays depending on which module asked. **21 of
the 72 trunks have no typed length at all**, so this was not a corner case.
`trunk_length_mm()` now reads IHM's measured route first and returns the source
string with the number; `assert_measured_lengths()` guards all 146 routes, where
`visceral_routes` had guarded only the 16 visceral ones.
