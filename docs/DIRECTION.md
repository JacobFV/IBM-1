# programme direction

The standing target, and the corrections that produced it. Read this before
choosing what to work on. Every item here came from the user redirecting work
that had drifted; they are recorded so the redirection compounds instead of
being re-litigated.

## The target

**A real naked human body crawling around and experiencing a real 3D simulated
world, through a mesoscale-fidelity brain.**

With basic behaviours and modality prediction pretrained — the hope being that
pretraining gives the body a **lower-dimensional structure to build from** once
agentic-level training begins: curiosity, pleasure/pain, social identity, and
the rest.

The materialization this needs is explicitly:

    sensory in  ·  body state in (digestive, blood, breath, reproductive,
    hormonal)  ·  other interoceptive in  ·  motor out  ·  speech out

## How the body comes to move

The end state, and the reason every stage below is a stage:

> **The brain directly controls individual muscles. Those muscles' contractions
> control the skeleton. The whole thing is a highly complex rigid AND soft body
> system, actuated by the dynamic elasticity of the muscles, which is determined
> by the brain's outputs.**

The route, recorded because the middle of it is easy to mistake for the end:

1. **Now.** The crude 22-segment body actuates the real body's muscles and bones
   and determines rough contact with floor and objects. That motion and its
   afference are collected as training data for teaching the brain to actuate the
   real body. Contact is rough on purpose; it only has to generate honest load.
2. **Possibly.** The brain learns to actuate the crude body too, as a
   lower-dimensional curriculum stage — justified only by whether it transfers.
3. **Gradually.** The crude body is thrown away, job by job. Contact first — real
   potentially-concave bone meshes replacing inertia-inscribed spheres — then
   actuation.
4. **The end state above.**

**The crude body has always been a scaffold.** Not a simplification settled for,
a stage. Making it better *as a scaffold* is worth doing; deepening dependence on
it is not; and what it did must never be reported as what the body did. The full
version with the code pointers is `docs/ACTUATION_STAGES.md` in the IHM-1 repo.

## Corrections that define the work

**No simplifications of the body.** Not a stick figure, not capsules, not a
22-body skeleton standing in for a person. The actual anatomical human: its
bones, its muscles, its integument. A demo of a capsule figure is not a demo of
this programme, however good the numbers beside it are.

**The brain touches the body only through nerve fibres.** Not through index
slices, not through a port that means "some sites". Fibres connected to *real*
muscles and *real* skin patches — and every one of them has to actually be
innervated. Coverage is a thing to measure, not assume.

**Muscles pull on real rigid bodies.** Motor output is force through tendon on
bone, not an activation vector consumed by an abstraction.

**The skeleton is real meshes, not proxies.** No inertia-inscribed spheres at
segment centres of mass. A potentially-concave rigid-body skeleton of 3D meshes,
one per bone, is what the body contacts the world with. Anything that replaces a
bone with a sphere is a simplification of exactly the kind this programme is not
allowed to make — and a solver that silently takes the convex hull of a concave
mesh is the same simplification wearing a better name, so it has to be measured
rather than assumed.

**Muscles, tendons and ligaments anchor at real points** on those meshes.

**The tissue that holds it together is part of the model.** Not only elastic
integument, ligaments, tendons, lymph and vasculature — also **cartilage,
menisci and intervertebral discs, joint capsules, bursae and synovium, fascia and
aponeurosis, retinacula, periosteum, and adipose**, and the integument bound to
what lies under it rather than floating over it.

Counted in the bound body: 300 ligament entities, 80 bursa/synovium, 69
meniscus/disc, 51 cartilage, 44 tendon, 44 fascia, 36 joint capsule, 18
retinaculum, 2 adipose, 1 skin — **645 of 4,000 entities are tissue structures.**
They existed as geometry and not as mechanics.

**117 of them now carry force**: 105 ligaments and 12 joint capsules, as
`Blankevoort1991Ligament` elements over two-ended attachments derived from each
structure's own surface, in `docs/TISSUE_MECHANICS.md` in IHM-1. Standing weight
is unchanged at 761.3757 N because these are internal forces. Over three prone
drops, the **full** set makes the plant worse every time — worst excursion past
the model's declared ranges goes 17.4/30.7/30.3 deg bare to 32.8/35.3/32.9 —
because a straight line between two attachment centroids is not a ligament's path
and a real cruciate is near-isometric only because it wraps. The **66** that
never pass ligament ultimate strain inside a spanned joint's declared range are
never worse than bare, and added to the joint stops they improve every drop,
5.96 -> 5.27 deg mean. They do NOT replace those stops; one drop said they did
and is ledger row 28. Closing the other 51 — the cruciates, the collaterals, the
ankle ligaments — needs a wrap surface per joint.

The rest is blocked for three different reasons, and only one is a missing
solver. **430 of the 645 are inside ONE scaffold rigid body** — 23
intervertebral discs and 23 nuclei in `torso`, 29 ligaments per hand — so the
scaffold has no joint where they act. **Cartilage is a DATA gap**: 51 cartilage
entities and not one of them is a joint surface, all costal, laryngeal, nasal or
growth plate; and bone-on-bone contact cannot substitute, because a synovial
joint's two bone surfaces overlap by construction (the hips interpenetrate in
19/21 and 21/21 sampled configurations of their own declared range). **Fascia and
retinacula are blocked on the muscle path**: 80 of the 98 muscles run as fitted
polynomials with no geometry, so not one of the 18 retinacula can constrain
anything. Adipose is 1.26 mL of geometry in the whole body, and periosteum is 0
entities — acquisition, not modelling.

Progress against this and every other objective is tracked in
`docs/MILESTONES.md`. Elastic integument,
ligaments, tendons, lymph vessels, blood vessels. The body is not a skeleton
with a skin texture.

**Forced motion is scaffolding, never a result.** Driving the body through a
gait by external force is legitimate *only* to generate the rough shape of what
real muscles and skin experience during that motion. That corpus exists to
bootstrap the brain into what embryonic reflex circuits are supposed to
condition. Forced motion is not locomotion and must never be reported as such.

**Speed over fidelity in motor learning, for now.** Low-fidelity motor
bootstrapping is acceptable and wanted; the higher-fidelity curriculum comes
later. Do not spend the budget perfecting gait search.

## Parametrization (added after the direction above)

The body must be **parametrized**, not a single fixed subject. Ask for a tall
person or a short person and get one. Sex is a parameter too, and it pulls a
large dependent cluster with it: breasts, genitalia and hair as anatomy;
hormones and anatomical proportions as parameter clusters.

Not everything needs a knob. Some things already are knobs and only need
surfacing into one body-parameter input. Others need bones and muscles taken from
the source to become geometrically parametrized, transformed entities.

Measured starting point, so the scope is not guessed: `target_mass_kg` and
`instance_mass_variant` are already knobs; the body is a scaled Rajagopal subject
registered against BodyParts3D; and the body **already carries male genital
anatomy** — corpus cavernosum and spongiosum, glans, testes, epididymides,
deferent ducts, seminal vesicles, ejaculatory ducts, prostate and their vessels,
~30 entities, all bound to `pelvis`. What is genuinely absent is **any
female-specific entity, and any mammary gland, nipple or areola in either sex**;
"mammary region" is a patch of chest skin. No catalogued source ships a female
mesh — BodyParts3D is male-only, Z-Anatomy is male and not an independent
subject, Rajagopal is male.

(This paragraph first said the body had ZERO sex-specific entities. That was
wrong, and is ledger row 26: the search ran against entity IDs, which are opaque
strings carrying no name, so it returned 0 for every term — including `femur`,
which was never checked.) The reproductive model exists but its own limitation note says
E2/P4/inhibin are prescribed time functions, not an autonomous cycle.

The standing honesty requirement applies with force here: a scaled male mesh with
surfaces bolted onto it is not a female body, and must not be labelled one.

**Where it stands now.** `sex` stopped raising, because it stopped meaning
nothing. It sets four measured proportions — stature, mass, hip separation over
stature (+12.6461%, to 1e-13 of the NHANES target), leg length over stature
(−2.5526%) — and its resolved record carries a `sex_realisation` block naming
what did NOT move: still 0 female-specific entities, still 0 mammary glands in
either sex, male genital tract still bound to `pelvis`. `--sex female` builds
**male anatomy at female-typical proportions** and the registration says so in
those words.

What unblocked it was the path refitter, and the interesting part is that it was
never missing. `docs/BODY_PARAMETERS.md` in IHM-1 said it needed OpenSim in the
environment; `libosimActuators.so` in the install this repository already builds
exports 97 `PolynomialPathFitter` symbols. It needed a driver. A refit of the
unchanged model is **16.7× closer** to the model's own `GeometryPath`s than the
`FunctionBasedPathSet` the engine currently runs (3.01e-4 m against 5.02e-3 m RMS
on held-out frames), and it covers **98** muscles against the shipped set's 80.

That last number bears on the paragraph above about retinacula: the 18 muscles
that still had explicit geometry were the `arm26` and `gait2392` trunk muscles,
and a full refit gives them fitted moment arms while taking their `GeometryPath`
away. Both directions are recorded in each variant's registration rather than
chosen silently.

## The developmental components

Pain is one component of the developmental learning process, not the list.
`docs/DEVELOPMENTAL_COMPONENTS.md` enumerates ~40 of them across seven groups —
homeostatic drives, affect and reward, self-supervised prediction, motor
development, body schema, social, and regulatory — each with what it is for and
whether it is **built / partial / declared / absent** here.

A handful are built, rather more are partial, and most are absent. That is the
expected shape at this stage; the point of the list is that it stays visible.

The four highest-leverage absent items, in unblocking order rather than
importance: **efference copy** (prediction, sensory attenuation, body ownership
and agency all sit on it), **a unified prediction-error signal**, **spontaneous
motor activity** (how a body with 214 innervated muscles finds out what they do),
and **pleasure** — because a body with only nociception can learn to avoid and
cannot learn to seek, and half of development is seeking.

## What this rules out

Work that improves a number on an object that is not the real body, or that
reaches the body through anything other than declared innervation, is off-target
however well it measures. `docs/DISCONNECTS.md` is the standing list of places
where the running model and the declared model are different objects; closing
those is on the critical path, not a cleanup task.

## Why this file exists

The programme repeatedly drifted toward whatever was measurable on the objects
already wired up — a spherical cortex with random long-range edges (both now
replaced: fsaverage surface, Desikan-Killiany labels, and long-range partners
drawn from a 1064-subject HCP connectome), a 22-body skeleton, a video term that
turned out to be matching appearance — and each time
the user had to redirect it back to the real body and the real brain. The drift
is not random: measurable-on-the-proxy is always cheaper than correct-on-the-real
-thing, so it wins by default unless something holds the target fixed. This file
is that thing.
