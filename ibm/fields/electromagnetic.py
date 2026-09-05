"""the head as a volume conductor: potential, electric field, current density, flux.

spectral throughout, and for a reason that is different from the neural field's.
neural state is spectral because several rhythms genuinely interact inside one
variable.  the electromagnetic field is spectral because it is the meeting point
of sources that live on wildly different timescales and must share one component:
the same `electromagnetic.potential` at a scalp position holds the millihertz drift
of a DC-coupled recording, the alpha rhythm, the gamma a grid picks up, the
kilohertz ringing of a TMS pulse, and the 1 kHz burst of a DBS lead.  a scalar
belief over that window would report a variance and nothing else, and every
observation the model has -- EEG, MEG, ECoG, LFP -- is an observation of a
*spectrum* of this field.  the spectral form is also what makes the forward
problem cheap: in the quasi-static regime the head is resistive to a very good
approximation up to tens of kilohertz, so the map from source to sensor is a real
matrix applied per frequency component with no filtering and no history.

quasi-static means something specific and load-bearing here.  at physiological
frequencies the wavelength in tissue is kilometres, capacitive currents are small
against ohmic ones, and inductive coupling within the head is negligible; Maxwell
therefore collapses to Poisson's equation with a source term, and potential is
determined *instantaneously* by `neural.transmembrane_current` and
`material.conductivity`.  that is why the em generation process is an algebraic
constraint rather than a dynamics -- the stiff limit of pressure, in the
architecture's terms -- and why all four components below carry the same timescale.

the four are not redundant.  potential is what an electrode measures and is
defined only up to a reference; the electric field is its negative gradient and is
reference-free, which is why it and not potential is what stimulation dosimetry
and membrane polarization depend on; current density is the field times local
conductivity and is the quantity safety limits are written in; and the magnetic
flux density is sourced by current density through Biot-Savart but is famously
*not* determined by potential, since radial and tangential sources contribute to
the two measurements differently.  collapsing any pair of them would make one of
MEG, EEG or tES dosimetry inexpressible.

units are SI here rather than the practical units electrophysiology uses, because
this is the one field where the same component must span thirteen orders of
magnitude: 10^-13 T at a MEG sensor and 2 T inside a TMS coil are the same
physical quantity, and picking femtotesla as the unit would make the stimulation
case unreadable.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Band, Provenance

FIELD = REGISTRY.field(Field(
    "electromagnetic",
    "the quasi-static electromagnetic state of the head volume: the potential, field, "
    "current density and magnetic flux produced by neural current sources and by "
    "stimulation devices, in the same variables",
    "head_volume", Provenance.PHYSICS))

#: wide enough for a stimulator.  neural sources occupy the bottom 300 Hz of it;
#: TMS ringdown and DBS pulse trains occupy the top, and both are the same field.
WIDE = Band(0.0, 20000.0)


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "spectral")
    kw.setdefault("provenance", Provenance.PHYSICS)
    kw.setdefault("band", WIDE)
    kw.setdefault("prior", "em_field")
    kw.setdefault("timescale_s", 1e-4)
    return REGISTRY.component(
        Component(id=id_, field="electromagnetic", doc=doc, units=units, **kw))


POTENTIAL = _c(
    "electromagnetic.potential",
    "electric potential in the head volume, relative to a stated reference.  it is what "
    "every electrode in the inventory actually measures -- scalp, subdural, depth and "
    "intracortical alike differ in where they sample this one field, not in what they "
    "measure -- and it is declared separately from the electric field because it is "
    "reference-dependent and the field is not.  the reference is not a nuisance to be "
    "removed: an average-referenced EEG and a bipolar depth recording are different "
    "linear functionals of the same component, and the model should say so",
    "V", bounds=(-100.0, 100.0),
    tags=frozenset({"observable", "quasistatic"}))

EFIELD = _c(
    "electromagnetic.efield",
    "the electric field, E = -grad(V).  reference-free, and the quantity that actually "
    "does anything to a neuron: membrane polarization depends on the component of E along "
    "the dendritic axis, so a uniform potential offset does nothing while a gradient of a "
    "volt per metre measurably biases spike timing.  every tES and TMS dose is specified "
    "in these units, and deriving it from potential at materialization time rather than "
    "declaring it would make the dosimetry depend on the mesh",
    "V/m", bounds=(-1000.0, 1000.0),
    tags=frozenset({"stimulation", "quasistatic"}))

CURRENT_DENSITY = _c(
    "electromagnetic.current_density",
    "ohmic current density, J = sigma E.  it carries the conductivity dependence that the "
    "field alone does not, which is why it and not E is where anisotropy shows up: in "
    "white matter sigma is a tensor aligned with `structural.fiber_orientation`, so J and "
    "E point in different directions there.  it is also the quantity safety limits and "
    "tissue-heating calculations are written in, and the source term the magnetic field "
    "integrates",
    "A/m^2", bounds=(-1000.0, 1000.0),
    tags=frozenset({"stimulation", "source"}))

BFIELD = _c(
    "electromagnetic.bfield",
    "magnetic flux density.  a separate component from the electric quantities because it "
    "is not recoverable from them: MEG is blind to radial sources that EEG sees well and "
    "insensitive to skull conductivity that EEG depends on entirely, and that "
    "complementarity is the reason for recording both.  its dynamic range is the widest in "
    "the ontology -- 10^-13 T at a magnetometer, 2 T inside a TMS coil -- and both ends "
    "are this component, which is why the unit is the tesla and not the femtotesla",
    "T", bounds=(-10.0, 10.0),
    tags=frozenset({"observable", "stimulation"}))
