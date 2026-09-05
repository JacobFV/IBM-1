"""tissue as a material: the constants every physical process is written against.

scalar and exogenous, both by construction.  scalar because a material constant
has no dynamics at all on any timescale ibm-1 runs -- the band is DC, and the
belief is about a number.  exogenous because nothing in the process inventory
writes it: conductivity is not produced by anything the brain does over an
experiment, it is a property the tissue has, and the registry's rule that a
component read but never written must be declared exogenous is exactly what
forces that admission to be visible rather than implicit.

being exogenous does not make these constants.  they are uncertain, and the width
of that uncertainty is the point.  reported skull conductivity spans a factor of
five across studies, grey and white matter differ by a factor of two from each
other and by less than the disagreement between measurement techniques, and
mechanical stiffness reported by elastography and by ex-vivo indentation differ by
an order of magnitude.  a narrow prior over any of these is the single most
reliable way to build a forward model that is confidently wrong -- an EEG source
localization is more sensitive to assumed skull conductivity than to almost any
other choice -- so `material_constant` widens rather than picking a favourite.

the support is `head_volume` rather than `tissue`, and that is deliberate: skull,
scalp, air cavities and csf are not brain parenchyma but they are exactly where
the interesting conductivity and stiffness contrasts are.  a material field
defined only on the brain would have nothing to say about the two tissues that
dominate every forward model.

tortuosity and porosity sit here rather than in the extracellular field because
they are geometry rather than chemistry, and because they are what the interstitial
diffusion process reads.  their dynamic partner is
`extracellular.volume_fraction`: this field carries the resting geometry, and that
component carries what swelling does to it.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import DC, Provenance

FIELD = REGISTRY.field(Field(
    "material",
    "the electrical, mechanical and transport properties of head tissue: what the "
    "electromagnetic, mechanical, thermal and diffusion processes are written against, "
    "supplied from outside the model and uncertain by factors rather than by percentages",
    "head_volume", Provenance.LITERATURE))

#: a material property does not vary over an experiment.  the timescale is the
#: subject's lifetime; the band is DC.
LIFETIME = 1e7


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "scalar")
    kw.setdefault("provenance", Provenance.LITERATURE)
    kw.setdefault("band", DC)
    kw.setdefault("prior", "material_constant")
    kw.setdefault("timescale_s", LIFETIME)
    kw.setdefault("exogenous", True)
    return REGISTRY.component(Component(id=id_, field="material", doc=doc, units=units, **kw))


CONDUCTIVITY = _c(
    "material.conductivity",
    "ohmic conductivity of tissue.  the quantity Poisson's equation is solved with, so it "
    "is what stands between `neural.transmembrane_current` and any predicted potential -- "
    "and it is anisotropic in white matter, a tensor aligned with "
    "`structural.fiber_orientation`, which is why the two fields have to be able to see "
    "each other.  the prior is deliberately wide: skull conductivity is reported between "
    "roughly 0.003 and 0.015 S/m and that spread alone moves an EEG source estimate by "
    "more than most experimental effects",
    "S/m", bounds=(0.0, 5.0),
    tags=frozenset({"electromagnetic"}))

PERMITTIVITY = _c(
    "material.permittivity",
    "relative permittivity of tissue.  it is what justifies the quasi-static approximation "
    "rather than what breaks it: at physiological frequencies the displacement current it "
    "carries is small against the ohmic one, so the electromagnetic field is instantaneous "
    "in its sources.  it is declared anyway because the approximation must be checkable, "
    "and because at the kilohertz-and-above end of the band a stimulator occupies, the "
    "ratio is no longer negligible.  its value is enormous and strongly dispersive at low "
    "frequency, which is a real property of tissue and not an error",
    "dimensionless", bounds=(1.0, 1e8),
    tags=frozenset({"electromagnetic"}))

MASS_DENSITY = _c(
    "material.mass_density",
    "tissue mass density, near 1040 kg/m^3 for brain and around twice that for cortical "
    "bone.  it enters the mechanical wave equation together with stiffness -- their ratio "
    "is the square of the wave speed -- and it sets the acoustic impedance mismatch at the "
    "skull that makes transcranial ultrasound hard.  it also converts the metabolic field's "
    "per-mass heat production into a per-volume source term for thermal diffusion",
    "kg/m^3", bounds=(0.0, 3000.0),
    tags=frozenset({"mechanical", "thermal"}))

STIFFNESS = _c(
    "material.stiffness",
    "the tissue shear modulus.  brain is one of the softest solids in the body at a few "
    "kilopascals, and skull is six orders of magnitude stiffer, which is the whole reason "
    "the head behaves mechanically as a soft body in a rigid shell.  it is the component "
    "MR elastography measures, and it changes measurably with age, oedema and tumour, so "
    "declaring it as a field rather than a constant is what lets that measurement enter",
    "kPa", bounds=(0.0, 1e8),
    tags=frozenset({"mechanical", "observable"}))

TORTUOSITY = _c(
    "material.tortuosity",
    "lambda: the factor by which the extracellular path length exceeds the straight-line "
    "distance, ~1.6 in healthy cortex.  effective diffusivity in the interstitium is the "
    "free value divided by lambda squared, so this single number is most of what "
    "distinguishes interstitial transport from free diffusion and is what the glymphatic "
    "and neuromodulator-spread processes read.  geometry, and therefore material, rather "
    "than a property of any solute",
    "dimensionless", bounds=(1.0, 4.0),
    tags=frozenset({"transport", "geometry"}))

POROSITY = _c(
    "material.porosity",
    "the resting extracellular volume fraction of the tissue, ~0.2.  the exogenous baseline "
    "geometry of the interstitial space, in contrast with `extracellular.volume_fraction`, "
    "which is the state that cell swelling moves away from it.  the pair is deliberate: a "
    "materialization that does not model swelling can read this and ignore the state "
    "variable, and one that does has somewhere to put the deviation without overwriting "
    "the subject's anatomy",
    "dimensionless", bounds=(0.0, 1.0),
    tags=frozenset({"transport", "geometry"}))
