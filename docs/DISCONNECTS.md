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

## 2. The cortical sheet was a sphere — CLOSED

*Was: `cortical_sites()` placed sites on a **spherical shell** area-matched to
the measured white surface, and `cortical_regions()` said so plainly — "a
geometric convention on the spherical proxy, NOT an atlas", six lobe labels cut
by coordinate thresholds. The consequence was not cosmetic: **the insula is not
separable on a sphere**, because a sphere has no lateral sulcus, so the
interoceptive port entered a 4.5% subsample of the `frontal` label instead.
`desikan2006` and `dkt-atlas` each held one file, 12 KB, a checksums.txt.*

The payloads are fetched (`scripts/fetch_cortical_atlases.py`) and the sheet is
a surface. `cortical_sites()` returns **fsaverage white-surface vertices**,
sampled with probability proportional to vertex area so density is uniform per
mm² of cortex, medial wall excluded. `cortical_regions()` returns one of **68
hemisphere-qualified Desikan-Killiany labels** read from `?h.aparc.annot`.

**The test this had to pass.** On 4,000 sites `insula` selects 63 and
`cingulate` 110, disjoint from frontal, temporal and parietal, and `lh.insula`
names one hemisphere's worth. None of those was an addressable target before.
`region_index` **raises** for them on the sphere rather than substituting
`frontal` — a silent substitution is how this survived.

The interoceptive port is now the insula and the anterior cingulate, the whole
label rather than a fraction of a larger one: 5.45% of a 6,000-site sheet
against the 5.3% of white-surface area those three labels occupy.

Corroborated on `dkt-atlas`, which is a different subject and a different
labelling protocol (though the card is right that it is *not* independent
evidence about a boundary — DKT is a deliberate revision of DK): insula 2.83% of
area under DK and 2.38% under DKT on the same brain, per-vertex Dice 0.835 and
0.866. Both hold the label; they disagree at the boundary, which is where the
revision was written to act.

**What the sphere was, now that the real number is held:** it was area-matched
to 202,437 mm², and the fsaverage white surface is 130,438 mm² in total and
118,310 mm² excluding the medial wall. The proxy was **1.55× the cortex by area
and 1.24× in linear scale**, so every distance on it — and the ~85 mm mean
separation of two random sites that made the long-range edges inert — was
inflated by about a quarter.

**Still open:** the sphere is kept and selectable (`--geometry sphere`), because
`dyn.pos` is a saved buffer and every checkpoint on disk restores its own sheet.
Those checkpoints' region labels still come from the coordinate convention, and
their published numbers are about that object. The sidecar of
`ckpt/visual_contrastive_v2.pt` says `"geometry": "fsaverage-sampled sheet"`,
which was **not true when it was written** — that run was a sphere.

## 3. The white-matter tracts are imported, and they change the answer

*Was: `ibm/topologies/tract.py` declared tractometric adjacency — which cortical
regions a fascicle joins and the conduction delay of each — and argued at length
that euclidean and geodesic metrics both get long-range cortical connectivity
wrong. The trained kernel drew its long-range partners with `torch.randint`,
uniformly at random over all sites, and no script imported the topology.*

`ibm/cortical_tracts.py` turns the `braingraph-hcp-connectomes` payload — 1064
HCP subjects, 86 Desikan-Killiany nodes, `fiber_length_mean` per edge — into the
`matrix` and `lengths_mm` that `tractometric_matrix` declares it needs.
`CorticalDynamics(long_topology="tract")` then draws each site's long-range
partners among the sites in the parcels the consensus connectome joins its own
parcel to, carrying that pair's arc length and conduction delay.

The full expansion `tractometric_matrix` would emit is 1.7×10⁸ edges at 30,000
sites and 10¹⁰ at 250,000 — its own docstring says the quadratic cost is "the
honest signal that a materialization far finer than the connectome is asking for
more than the connectome has" — so what is drawn is a **uniform subsample of
exactly that edge set**, which is what a fixed per-site budget can hold of it.

**The control is matched by construction, not by fitting:** the same `n_far`
edges per site and the same flat long-range prior, so edge count and row L1 are
identical to 0.000e+00 and the arms differ only in where the edges go.

Structural transport (`scripts/compare_long_range_topology.py`), the held
nonnegative operator applied 4 times to a unit indicator, 30,000 sites:

| pair | random | tract | tract/random | vs *concentrated* random |
|---|---|---|---|---|
| occipital→precentral | 3.10e-02 | 7.95e-03 | **0.257×** | 0.257× |
| occipital→temporal | 9.45e-02 | 1.48e-01 | 1.571× | 1.558× |
| postcentral→precentral | 5.25e-02 | 6.97e-02 | 1.328× | 1.315× |
| occipital→insula | 1.49e-02 | 1.29e-02 | 0.866× | 0.825× |

**The occipital→precentral row is the one that matters**, because it is the pair
`ablate_disjoint_transport.py` scores and the pair the whole disjoint-transport
enterprise has been about. Under the tract topology the minimum hop distance
from occipital to precentral is **2, not 1**: the consensus connectome declares
no direct occipito-precentral fascicle, which is anatomically correct. The
random graph gave every occipital site a one-hop shot at motor cortex, and a
quarter of the transport it was scored on came from an edge the brain does not
have.

Anatomy wins where anatomy has a pathway — the ventral stream and across the
central sulcus — and its partners sit 53 mm away in 7.6 distinct parcels against
79 mm in 10.7 for random ones.

**What the connectome does not reconstruct, which is the honest limit.**
`tract.py` names two cases as what the metric exists to get right: the arcuate,
which "connects frontal and temporal cortex over a path of roughly 150 mm", and
the corpus callosum, which "joins homotopic points that are close in the volume
and unreachable along the surface". At the 0.5 consensus threshold this
connectome reconstructs **neither**. Only 17 of 459 edges are interhemispheric
(3.7%); homotopic superior temporal appears in 0 of 1064 subjects, homotopic
postcentral in 0, homotopic precentral in 3.3%; and left parsopercularis to
superior temporal — the arcuate's canonical terminal pair — in 7.0%. That is a
known tractography failure, not a fact about brains, and it means the tract arm
is a **ventral, intrahemispheric** topology whatever its docstring hopes for.
The threshold is a knob and the sweep is in `edge_existence_curve()`: 616 edges
at 0.1, 459 at 0.5, 126 at 1.0.

**Still open:** conduction delays are carried per edge and saved
(`dyn.delay_s`), but they only bite at a timestep that resolves them — the
median declared delay is 3.84 ms and the maximum 10.2 ms, so at the visual
loop's `dt/substeps = 5 ms` they quantise to 0-2 steps. And the connectome is a
group average of one pipeline's tractography, with the seeding, response
function and thresholds discarded; the card says so, and `use: prior` is the
only role it is allowed.

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

Row 1 is what remains load-bearing for what the programme is trying to
demonstrate next. Rows 2 and 3 are closed, and closing 3 did what it was
predicted to do — it changed how the current cortical work is done rather than
merely extending it, and not in the direction that was hoped: the anatomy says
there is no direct occipito-precentral fascicle, so the disjoint-transport task
the last several sessions were scored on was being scored on a pathway the brain
does not have. The rest are honest scope: things declared ahead of being built,
which is fine as long as no result quietly claims them.

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
