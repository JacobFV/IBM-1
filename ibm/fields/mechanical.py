"""tissue motion and load on the head volume.

scalar, and this is the field where the choice most needs defending, because
mechanical phenomena obviously span timescales: cardiac pulsation at ~1 Hz,
respiratory motion at ~0.3 Hz, elastography drive at 50-100 Hz, an impact at
kilohertz.  the reason it is still scalar is that they do not occur together in
one materialization.  a glymphatic model materializes pulsation and nothing else;
an elastography model materializes a single drive frequency and reports a complex
modulus at it; an ultrasound model materializes a carrier and its envelope.  the
architecture's criterion is not "does the variable ever move fast" but "do several
timescales interact *within* this variable", and mechanically they do not -- each
regime is quasi-static with respect to the ones above it and averaged over the
ones below.  the band is declared wide enough to hold whichever regime is chosen,
and the belief within it is one number and one width.

the field is on `head_volume` rather than on tissue because the mechanically
interesting boundaries are precisely the ones brain parenchyma excludes: the skull
is the rigid shell, the csf is the coupling layer, and the falx and tentorium are
where shear concentrates.  limb, eye and articulator kinematics are this same
field on body supports, which is what the architecture means when it says
kinematics needs no field of its own.

five components because motion, rate of motion and load are separately observable
and separately meaningful.  displacement is what tagged and phase-contrast MR
measure and what a brain-shift correction needs; velocity is separate because at a
fixed frequency the two are related by a factor of omega and it is velocity that
enters viscous dissipation and the acoustic radiation force; stress and strain are
separate from each other because their ratio is `material.stiffness` and the whole
of elastography is estimating that ratio, so a model carrying only one of them has
assumed the answer; and pressure is separate from stress because it is the
isotropic part, the part a nearly incompressible tissue supports without deforming
at all, and the part that couples to `csf.pressure` and through it to perfusion.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Band, Provenance

FIELD = REGISTRY.field(Field(
    "mechanical",
    "the mechanical state of the head as a soft body in a rigid shell: displacement, "
    "velocity, stress, strain and pressure, from cardiac pulsation through elastography "
    "drive to focused ultrasound",
    "head_volume", Provenance.PHYSICS))

#: wide enough for an elastography drive and the harmonics of pulsation; an
#: ultrasound carrier is materialized as an envelope, not in this band.
MOTION = Band(0.0, 500.0)


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "scalar")
    kw.setdefault("provenance", Provenance.PHYSICS)
    kw.setdefault("band", MOTION)
    kw.setdefault("prior", "mechanical_quasistatic")
    kw.setdefault("timescale_s", 0.05)
    return REGISTRY.component(Component(id=id_, field="mechanical", doc=doc, units=units, **kw))


DISPLACEMENT = _c(
    "mechanical.displacement",
    "tissue displacement from a reference configuration.  measured from a reference and "
    "not from an origin, which is why its prior is centred at zero: there is no absolute "
    "position of a piece of brain, only how far it has moved.  brain tissue displaces by "
    "tens of micrometres per cardiac cycle and by millimetres after a craniotomy, and both "
    "are this component -- the first is what drives perivascular pumping, the second is "
    "what invalidates a neuronavigation plan",
    "mm", bounds=(-50.0, 50.0),
    tags=frozenset({"kinematic"}))

VELOCITY = _c(
    "mechanical.velocity",
    "the rate of tissue displacement.  not redundant with displacement even at a single "
    "frequency, because it is velocity and not position that enters viscous dissipation, "
    "acoustic radiation force and the momentum term of the wave equation -- and because "
    "phase-contrast MR measures this directly while tagged MR measures the other, so the "
    "two components are where two different measurements attach",
    "mm/s", bounds=(-1e4, 1e4),
    tags=frozenset({"kinematic", "observable"}))

STRESS = _c(
    "mechanical.stress",
    "internal force per unit area in the tissue, as the deviatoric part that actually "
    "deforms it.  declared separately from strain because their ratio is the stiffness the "
    "material field carries, and a model that derived one from the other with an assumed "
    "modulus could not then estimate that modulus from data.  shear stress concentrates at "
    "the falx, the tentorium and the grey-white boundary, which is why injury models care "
    "about where it is rather than how large it is on average",
    "Pa", bounds=(-1e7, 1e7),
    tags=frozenset({"load"}))

STRAIN = _c(
    "mechanical.strain",
    "the dimensionless deformation of the tissue.  it is the quantity injury thresholds are "
    "written in -- axonal damage correlates with strain and strain rate far better than "
    "with acceleration -- and the quantity elastography inverts for.  it is bounded well "
    "inside unity for anything short of trauma, and a materialization that produces larger "
    "values has left the linear regime the processes reading it were written for",
    "dimensionless", bounds=(-2.0, 2.0),
    tags=frozenset({"load"}))

PRESSURE = _c(
    "mechanical.pressure",
    "the isotropic part of the stress: hydrostatic pressure in the tissue.  separated from "
    "stress because brain is nearly incompressible, so it supports large pressures with "
    "almost no deformation and the two parts of the stress tensor behave completely "
    "differently.  it is the component that couples to `csf.pressure` and therefore to "
    "cerebral perfusion pressure, and it is the acoustic pressure a focused ultrasound "
    "beam delivers -- the same variable at two very different amplitudes",
    "Pa", bounds=(-1e7, 1e7),
    tags=frozenset({"load", "coupling"}))
