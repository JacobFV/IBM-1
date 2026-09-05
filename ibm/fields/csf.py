"""cerebrospinal fluid in the ventricles, cisterns and subarachnoid space.

scalar, for the blood reason and one more.  csf motion is driven almost entirely
by cardiac and respiratory pulsation and by the slow secretion-absorption balance,
so its content sits below a few hertz; and unlike blood it has no measurement in
the inventory that resolves anything faster -- phase-contrast MR velocimetry is
gated to the cardiac cycle and reports a cycle-averaged waveform, which is a
scalar statement about a window by construction.  declaring it spectral would be
declaring a belief no evidence can ever move.

three components, and the omission is deliberate: the architecture's field table
lists "pressure, velocity, flow, composition" and this field carries pressure,
velocity and composition without a separate flow.  flow is velocity times the
cross-sectional area of the space, and area on the csf support is geometry -- it
belongs to the support and the materialization, not to the state.  a separate
flow component would be the same number twice and would collide on normalization,
which is the registry catching a modelling error rather than a naming one.

the field exists at all because csf is not a passive filler.  it is the mechanical
coupling that makes the brain nearly neutrally buoyant and that transmits cardiac
pulsation into tissue; it is the compliance that turns a small volume change into
a large pressure change once it is exhausted, which is the Monro-Kellie doctrine
and the whole of intracranial-pressure physiology; and its exchange with the
interstitium along perivascular spaces is the glymphatic route by which solutes
including amyloid leave the brain, which is why `csf.solute` is a component rather
than a boundary condition.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Band, ULTRASLOW, Provenance

FIELD = REGISTRY.field(Field(
    "csf",
    "cerebrospinal fluid state: the pressure it transmits, the bulk and pulsatile motion "
    "it undergoes, and the solutes it clears from the interstitium",
    "csf_space", Provenance.LITERATURE))

#: cardiac at ~1 Hz, respiratory at ~0.3 Hz, and the harmonics of both.
PULSATILE = Band(0.0, 5.0)


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "scalar")
    kw.setdefault("provenance", Provenance.LITERATURE)
    return REGISTRY.component(Component(id=id_, field="csf", doc=doc, units=units, **kw))


PRESSURE = _c(
    "csf.pressure",
    "csf pressure, which is intracranial pressure wherever the space is continuous.  it is "
    "the shared variable of the Monro-Kellie constraint: brain, blood and csf occupy a rigid "
    "box, so any volume added by one must be taken from another and this pressure is what "
    "reports the failure to do so.  it enters perfusion as the downstream pressure -- "
    "cerebral perfusion pressure is arterial minus this -- which is the only route by which "
    "a purely mechanical event changes blood flow in the model",
    "mmHg", band=PULSATILE, prior="mechanical_quasistatic", bounds=(-15.0, 80.0),
    timescale_s=1.0, tags=frozenset({"pressure", "constraint"}))

VELOCITY = _c(
    "csf.velocity",
    "bulk csf velocity through the space at a position, signed along the local channel "
    "axis.  it is oscillatory rather than unidirectional over a cardiac cycle -- aqueductal "
    "flow reverses twice per beat with peaks of a few centimetres per second and a net "
    "displacement close to zero -- so a model that carried only a mean would report almost "
    "nothing.  phase-contrast MR measures exactly this component, which is why it is "
    "declared in the units that measurement reports",
    "mm/s", band=PULSATILE, prior="mechanical_quasistatic", bounds=(-200.0, 200.0),
    timescale_s=0.3, tags=frozenset({"transport"}))

SOLUTE = _c(
    "csf.solute",
    "the concentration of a transported solute in csf: the generic composition component, "
    "instantiated once per species a materialization actually needs rather than once per "
    "species that exists.  it is the sink side of glymphatic clearance -- the reason "
    "perivascular exchange is a process in the inventory at all -- and the compartment "
    "every csf biomarker is measured in, so it is what binds a lumbar-puncture measurement "
    "to interstitial state that nothing can sample directly",
    "mmol/L", band=ULTRASLOW, prior="concentration", bounds=(0.0, 500.0),
    timescale_s=600.0, tags=frozenset({"transport", "clearance"}))
