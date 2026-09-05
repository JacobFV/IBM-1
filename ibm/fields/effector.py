"""motor units and muscle: where neural activity becomes force.

spectral, and the evidence for it is unusually direct.  the common drive to a
motor-unit pool has four separable components at rest and during steady
contraction -- a 1/f drift below 5 Hz that sets slow force wander, a ~10 Hz
physiological tremor, a beta band that is demonstrably corticospinal in origin and
is the coherence peak every EEG-EMG study measures, and a piper component near
40 Hz that appears only in strong contractions.  they are simultaneously present
in the same drive and they interact: beta drive suppresses during movement while
the low-frequency component grows, and the ratio is what distinguishes a postural
hold from a movement.  a scalar belief would report a mean drive and lose all of
it, and with it the single most-used non-invasive measure of corticospinal
transmission.  `motor_drive` is that four-band spectrum written as a prior.

the muscle in front of that drive is a low-pass with a time constant of tens of
milliseconds, so force does not simply follow drive -- it follows a filtered
version of it, and that filtering is why tremor at 10 Hz is visible in force while
beta at 20 Hz mostly is not.  keeping drive, activation and force as three
components rather than one is what lets that filter be a process with a fitted
time constant instead of an assumption baked into a single variable.

the support is `motor_units`: a discrete set indexed by recruitment threshold
rather than by position, because Henneman's size principle makes threshold the
coordinate along which motor units are actually ordered and along which anything
generalizes.  extraocular, articulatory and autonomic effectors are components of
this same field on the same discrete support, which is what the architecture means
when it declines to give them fields of their own.

a movement is evidence about this field, not an output of the model.  kinematics
-- where the limb actually went -- is the mechanical field on body supports, and
the map from force to it is an ordinary process.  that separation is what makes
force plate, EMG and motion capture three different observations of three
different components rather than three views of one opaque "motor output".
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Band, Provenance

FIELD = REGISTRY.field(Field(
    "effector",
    "motor-unit and muscle state: the common drive a pool receives, the activation it "
    "produces, the force that follows, and the fatigue that slowly changes the map "
    "between them",
    "motor_units", Provenance.LITERATURE))

DRIVE = Band(0.0, 100.0)
MECHANICAL = Band(0.0, 50.0)


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "spectral")
    kw.setdefault("provenance", Provenance.LITERATURE)
    return REGISTRY.component(Component(id=id_, field="effector", doc=doc, units=units, **kw))


EFFECTOR_DRIVE = _c(
    "effector.drive",
    "the common synaptic drive reaching a motor-unit pool, expressed as the discharge rate "
    "it would produce in a recruited unit.  distinct from `neural.efferent.activity` "
    "because one motoneuron pool's output is distributed over its units by a recruitment "
    "rule ordered by size, so the same descending command produces different drive in a "
    "low- and a high-threshold unit -- and recruitment order, not drive amplitude, is what "
    "changes between a slow ramp and a ballistic contraction",
    "Hz", band=DRIVE, prior="motor_drive", bounds=(0.0, 100.0), timescale_s=5e-3,
    tags=frozenset({"motor", "drive"}))

ACTIVATION = _c(
    "effector.activation",
    "muscle activation: the fraction of maximal calcium-mediated cross-bridge availability, "
    "which is the state the surface EMG envelope estimates.  it sits between drive and "
    "force because excitation-contraction coupling is a low-pass with a time constant of "
    "tens of milliseconds -- the reason a 20 Hz beta drive is clearly present in the EMG "
    "and barely present in the force, and the reason those two measurements constrain "
    "different things",
    "dimensionless", band=DRIVE, prior="motor_drive", bounds=(0.0, 1.0), timescale_s=3e-2,
    tags=frozenset({"motor", "observable"}))

FORCE = _c(
    "effector.force",
    "the contractile force produced at the tendon.  a separate component from activation "
    "because the map between them is neither instantaneous nor fixed: it depends on muscle "
    "length and shortening velocity through the force-length and force-velocity relations, "
    "both of which are functions of the mechanical field's state on the body support.  it "
    "is the variable a force transducer measures and the variable that becomes a boundary "
    "condition for limb kinematics",
    "N", band=MECHANICAL, prior="motor_drive", bounds=(0.0, 5000.0), timescale_s=5e-2,
    tags=frozenset({"motor", "observable"}))

FATIGUE = _c(
    "effector.fatigue",
    "peripheral fatigue: the slowly accumulating loss of force-generating capacity at a "
    "given activation, from metabolite accumulation and impaired excitation-contraction "
    "coupling rather than from anything central.  it is state and not a parameter because "
    "it has its own dynamics -- it accumulates over tens of seconds and recovers over "
    "minutes -- and because holding it separate is what lets the model say that a drop in "
    "force with drive held constant is peripheral, which is the whole content of a "
    "twitch-interpolation experiment.  it carries the aperiodic prior rather than the motor "
    "one: it has no rhythms, only a spectrum that is all at the bottom",
    "dimensionless", band=Band(0.0, 1.0), prior="aperiodic", bounds=(0.0, 1.0),
    timescale_s=60.0, tags=frozenset({"motor", "slow"}))
