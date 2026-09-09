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

## 4. Seventy-one nerve trunks, zero nerves in the body

`ibm/topologies/nerve.py` declares 71 trunks with fibre-class composition and
length-resolved conduction. The simulated body contains **0 entities that are
nerves**. The trunks are a routing convention over muscle and region names; there
is no nerve in the mechanical model for them to be.

## 5. Skin is three whole-body layers, not patches

The body carries `body-skin-epidermis`, `body-skin-dermis`, `body-skin-hypodermis`
— **three** entities for the entire integument. There are no discrete skin
patches, so there is nothing for a cutaneous afferent to innervate at a location,
and dermatomal organisation cannot be expressed.

## 6. Thirty-two muscles move without spinal arcs

Of 98 catalogued muscles, **66** map to a declared nerve and root level. The other
32 pass descending drive straight through and receive no reflex arcs. They move,
so they look innervated. A count of "muscles driven" that includes them is not a
count of muscles under reflex control.

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
