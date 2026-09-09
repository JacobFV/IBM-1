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
