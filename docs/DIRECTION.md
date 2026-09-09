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

**The tissue that holds it together is part of the model.** Elastic integument,
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
