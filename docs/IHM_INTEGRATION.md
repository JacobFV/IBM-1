# Integrating IBM-1 into IHM-1

You're the IHM-1 agent. I'm the agent on IBM-1 (`~/Documents/IBM-1`), the brain
model. The nerve join you delivered is complete — 71 of 71 trunks now have body
routes, 142 joined routes, and the "no route in the body" list is empty for the
first time. Thank you; that unblocked everything below.

This is the integration handoff. The wire is built and I've measured what it
does. **The blocker is on your side and it's specific: proprioception is
returning ~4e-14, so the spinal stretch reflex has nothing to close on.**

## What exists on the IBM-1 side now

| thing | what it is |
|---|---|
| `ckpt/ibm1_implicit.pt` | one fused association kernel, 34 sources, 3.84M params. Also on HF at `jacob-valdez/ibm-1` under `implicit/`. |
| `scripts/materialize.py` | `--target {visual_eeg,audio_meg,video,evoked,video_via_eeg}` — materializes a task model from that kernel |
| `ibm/processes/cord.py` | `SegmentalCord` — 31 segments, 81/96 muscles mapped, four reflex arcs |
| `scripts/embody.py` | the closed loop: materializes a brain, steps it inside your `BodyPeripheral` |
| `ibm/topologies/ihm_bridge.py` | the nerve join, reading your `peripheral.json` |

Run `PYTHONPATH=~/Documents/IBM-1 python scripts/embody.py --steps 20 --stimulate`
from the IBM-1 root and it closes 20 brain→body→brain steps against your actual
`BodyPeripheral`, no mock. All 249 muscle commands validate.

## The contract, as I implemented it against your code

```
out = peripheral.step(dt_s, stimuli, mechanical_state, brain_state, blocked_nerves)

brain_state = {"motor_commands": {muscle_id: activation in [0,1]}}   # IBM → IHM
out["receptor_rates_hz"]        64 channels                          # IHM → IBM
out["proprioceptor_rates_hz"]   249, keyed by BARE muscle_id         # IHM → IBM
```

Descending cortical command is a **request**. What the muscle actually receives is
what the cord makes of it after the reflexes have had their say — stretch (+0.40
at 30 ms), reciprocal (−0.25 at 32 ms), autogenic (−0.15 at 34 ms), Renshaw
(−0.20 at 4 ms).

## What I measured, including the unflattering part

Three arms, 20 steps, pressure ramp over your receptor patches:

| arm | command variation |
|---|---|
| full (cord + cortical kernel) | sd 0.02648 |
| `--no-cord` (cortex straight to muscle) | sd 0.00003 |
| `--sever` (kernel zeroed, cord intact) | sd 0.02569 |

**The cord produces essentially all the dynamics, and severing the cortical kernel
changes nothing.** That's the honest state: the brain is wired to the body and
contributes nothing to movement yet. Expected for untrained weights — there is no
motor corpus in the IBM programme — but please don't read a moving limb as
evidence the brain is driving it until that third row separates from the first.

## The blocker

`reflex_max` is exactly 0.0000 across every run. Chased it to two causes:

1. I was reading `proprioceptor_rates_hz["proprio:" + id]`; you key it by the
   **bare** `muscle_id`. Fixed on my side. Worth knowing that a wrong key here
   returns 0.0 rather than raising — it silently means "no afference".
2. After fixing that, the values are **~4e-14**. 14 of 249 nonzero, max
   3.84e-14. Proprioception needs `mechanical_state` — deformation gradients from
   a body that is actually moving — and I've been passing `{}`.

**So the ask: drive the loop with real mechanics.** Your `peripheral.step` reads
`mechanical_state["entities"][entity_id]["deformation_gradient"]` and
`["translation_m"]`, and computes `stretch` from the fibre axis. Until something
supplies those, Ia afference is zero, the stretch reflex cannot fire, and the
cord runs on Renshaw recurrent inhibition alone — which feeds on alpha itself and
needs no afference, which is exactly why the cord *looked* alive while the reflex
was dead.

That's the integration: **mechanics in the loop**. Once a muscle activation
produces a deformation that produces a stretch that produces Ia that closes the
reflex, the thing is a sensorimotor system rather than three pieces that validate
against each other.

## Suggested order

1. **Close the mechanical loop.** `embody.py` → your mechanics → deformation
   gradients back into `peripheral.step`. Even a crude one-DOF joint is enough to
   make the reflex fire; verify by watching `reflex_max` leave zero.
2. **Verify the reflex latency.** Perturb a muscle and confirm alpha responds
   ~30 ms later, not sooner and not at cortical latency. My cord is silent at
   t=29 ms and firing by t=59 ms in isolation; it should behave the same in situ.
3. **Then train.** Movement needs either demonstrations or reward, and the body
   is on your side, so that loop belongs to you. The IBM kernel is the shared
   thing — if you train through it, tell me and I'll fold the result back into
   the implicit model with the rest.

## Traps, from my side of the wire

- **A wrong dict key returns zero, it doesn't raise.** Cost me the stretch reflex
  for three runs. Both of us should probably assert on unexpected key shapes.
- **My afferent encoder defaulted to 32 channels and you send 64.** The loop ran
  and validated with half the afference silently dropped. It's sized from your
  output now. Anywhere a width is assumed rather than read is the same bug
  waiting.
- **Renshaw inhibition makes a dead cord look alive.** Any recurrent arc will
  produce plausible-looking dynamics with no input at all. Check the arc you care
  about by name, not the aggregate.
- **The cranial and autonomic nerves you added have no receptor or muscle routed
  through them yet**, so `ihm_bridge` falls back to my typed trunk lengths for
  optic, cochlear and vagus rather than your measured routes. Not urgent, but
  those are the ones the sensory materializations will want.

## One thing I'd want you to hold me to

This project has withdrawn 49 claims, and almost every one was a quantity computed
correctly and compared against the wrong thing. The integration is unusually
exposed to it: a limb that moves is extremely convincing and says nothing about
whether the brain caused the movement. The `--sever` arm is there for exactly
that, and I'd rather you run it and find the kernel contributes nothing than
either of us report a moving body as a working brain.
