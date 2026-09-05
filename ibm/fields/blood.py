"""cerebral blood on the vascular tree.

scalar, and this field is the architecture's own worked example of why the two
forms exist.  blood has essentially no structure above half a hertz that any
measurement in the inventory resolves: the haemodynamic response to a brief event
is a five-second curve, the resting fluctuations that resting-state fMRI is built
on live below 0.1 Hz, and everything faster is aliased cardiac and respiratory
pulsation that fMRI treats as a nuisance regressor rather than as signal.  a
spectral belief over a component whose whole content lies in two or three
low-frequency bins would spend the temporal budget on zeros, and the two budgets
multiply, so that waste is paid for at every vascular position.

the support is a tree rather than a volume, and that is the structural claim this
field makes: two capillaries a hundred microns apart in space can be fed by
different pial arterioles and are, for flow purposes, far apart.  a volume
representation of perfusion silently asserts the opposite and is how a model
acquires vascular smoothing it was never given.

six components rather than one "perfusion" variable, because the BOLD signal is a
*ratio* and the things that enter it move independently.  flow and volume are
related by Grubb's law only in steady state and diverge transiently -- the venous
balloon empties more slowly than it fills, which is the whole of the post-stimulus
undershoot.  oxygenation and deoxyhaemoglobin content are not the same variable
either: deoxyhaemoglobin is what dephases spins and therefore what an MR sequence
sees, and it is the product of oxygenation with volume, so a model that carries
only a saturation fraction cannot produce a BOLD signal at all.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Band, HEMODYNAMIC, Provenance

FIELD = REGISTRY.field(Field(
    "blood",
    "cerebral blood state on the vascular tree: perfusion, compartment volume, driving "
    "pressure, and the oxygen carriage that makes the vasculature visible to MR",
    "vascular_tree", Provenance.LITERATURE))

#: pulsation is real content in pressure, and only in pressure.
PULSATILE = Band(0.0, 20.0)


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "scalar")
    kw.setdefault("provenance", Provenance.LITERATURE)
    kw.setdefault("band", HEMODYNAMIC)
    kw.setdefault("prior", "hemodynamic")
    return REGISTRY.component(Component(id=id_, field="blood", doc=doc, units=units, **kw))


FLOW = _c(
    "blood.flow",
    "cerebral blood flow: volume of blood delivered per unit tissue mass per minute.  the "
    "variable neurovascular coupling writes and the one arterial spin labelling measures "
    "directly, in these units and without a calibration constant.  it is separate from "
    "volume because the two are related by a power law only at steady state: a step in "
    "flow produces a volume change that lags it by seconds on the arterial side and by "
    "much longer on the venous side, and that lag is not a detail",
    "ml/100g/min", bounds=(0.0, 250.0), timescale_s=2.0,
    tags=frozenset({"perfusion"}))

VOLUME = _c(
    "blood.volume",
    "cerebral blood volume per unit tissue mass, summed over the compartment at a position "
    "on the tree.  it is what VASO and contrast-based methods measure, it is the "
    "multiplier that turns an oxygenation fraction into an amount of deoxyhaemoglobin, and "
    "its slow venous relaxation is the standard explanation for the post-stimulus "
    "undershoot -- a phenomenon a flow-only model has to invent a neural cause for",
    "ml/100g", bounds=(0.0, 20.0), timescale_s=6.0,
    tags=frozenset({"perfusion"}))

PRESSURE = _c(
    "blood.pressure",
    "intravascular pressure at a position on the tree.  it is the boundary condition "
    "perfusion is computed against, so a model that omits it cannot express autoregulation "
    "failure, orthostatic change, or the raised intracranial pressure that reduces "
    "perfusion pressure without changing anything neural.  the one component in this field "
    "whose band extends past the haemodynamic range, because cardiac pulsation is genuinely "
    "there -- it is carried scalar anyway, because within ibm-1 pressure enters as a slowly "
    "varying perfusion boundary condition and the pulsatility that matters mechanically is "
    "carried by the mechanical field instead",
    "mmHg", band=PULSATILE, bounds=(0.0, 250.0), timescale_s=1.0,
    tags=frozenset({"perfusion", "boundary"}))

OXYGENATION = _c(
    "blood.oxygenation",
    "haemoglobin oxygen saturation: the fraction of binding sites occupied, ~0.98 arterial "
    "and ~0.60-0.65 venous at rest.  an intensive quantity, which is why it is not "
    "sufficient on its own -- an MR signal responds to how much deoxyhaemoglobin is "
    "present, not to what fraction of the haemoglobin present is deoxygenated, and the two "
    "differ exactly by the blood volume",
    "dimensionless", bounds=(0.0, 1.0), timescale_s=2.0,
    tags=frozenset({"oxygen"}))

DEOXYHEMOGLOBIN = _c(
    "blood.deoxyhemoglobin",
    "deoxyhaemoglobin concentration in the vascular compartment at a position.  the "
    "paramagnetic species: it is the source of the susceptibility difference between blood "
    "and tissue, therefore of every extravascular BOLD contrast the model predicts, and "
    "the reason the BOLD signal goes *up* when oxygen consumption goes up more slowly than "
    "flow does.  declared separately from oxygenation because it is the product of "
    "oxygenation with volume and the model must be able to move either factor alone",
    "umol/L", bounds=(0.0, 250.0), timescale_s=3.0,
    tags=frozenset({"oxygen", "observable"}))

OXYGEN_CONTENT = _c(
    "blood.oxygen_content",
    "total oxygen carried per unit volume of blood, bound plus dissolved.  it is what the "
    "tissue-exchange process draws on and what the arteriovenous difference is taken of, "
    "and it is not interchangeable with saturation because anaemia, altitude and "
    "hyperoxia all move content while leaving saturation near normal.  keeping it "
    "separate is what lets a physiological manipulation enter the model at the right place",
    "mmol/L", bounds=(0.0, 12.0), timescale_s=2.0,
    tags=frozenset({"oxygen"}))
