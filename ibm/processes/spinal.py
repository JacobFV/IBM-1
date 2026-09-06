"""the segmental cord: the reflexes that close inside the spinal cord.

`effector.py` states the gap this file fills, in its own words:

    "where this breaks: there is no spinal circuitry in it.  the segmental
    interneurons, reciprocal inhibition, renshaw feedback, and the stretch reflex
    loop are all absent, so the pool here is a relay with a threshold rather than
    a circuit.  every reflexive and load-compensating behaviour is therefore
    missing, and the model will attribute those to descending command because
    descending command is the only thing it has."

that last clause is the reason this matters for FITTING and not only for
behaviour: a model without a cord will inflate corticospinal gain to explain
force that the cord actually produced, and the inflated gain is then reported as
a property of cortex.

### why it changes what the model can do at all

EMBODIMENT.md measures the long loop: 18.4 ms out, 17.0 ms back, and 75-135 ms
once electromechanical delay is counted, which caps corrective bandwidth near
1/(4T) = 2-3 Hz.  the monosynaptic arc here is Ia -> alpha at ONE synapse inside
the cord, so its loop is the trunk delay twice plus a synaptic delay -- on the
order of 30 ms for a leg and under 15 ms for a hand.  that is not an incremental
improvement, it is a different control regime, and it is why load compensation is
possible at all.

### the four arcs, and why they are separate processes' worth of distinction

*stretch (myotatic)*: Ia -> homonymous alpha, monosynaptic and excitatory.  the
only monosynaptic reflex in the body, and the reason its latency is diagnostic.

*reciprocal inhibition*: Ia -> Ia-inhibitory interneuron -> ANTAGONIST alpha.
disynaptic, so ~1 ms slower, and it is what makes the stretch reflex a joint-level
response rather than a muscle-level one.

*autogenic inhibition*: Ib -> Ib interneuron -> homonymous alpha, inhibitory.
force feedback rather than length feedback, and with the opposite sign to the
stretch reflex, so the pair implements something closer to impedance control than
either does alone.

*recurrent (Renshaw)*: alpha collateral -> Renshaw cell -> the same alpha pool.
a gain control on the motoneuron pool itself, and the reason a pool does not
saturate under strong descending drive.

they share a topology and a support and differ in sign, in synapse count and in
whether the target is homonymous or antagonist -- which is exactly the pattern
ARCHITECTURE.md 4 says should be separate declarations rather than one lumped
coupling with a sign parameter.
"""

from __future__ import annotations

from ibm.processes.base import (
    alpha_synapse,
    implementation,
    low_pass,
    process,
    pure_delay,
    series,
)
from ibm.registry import Form
from ibm.vocabulary import (
    Band,
    OnSupport,
    Provenance,
    Tying,
    lognormal,
    normal,
    weak,
    within,
)

BODY = OnSupport("body")
MOTOR_UNITS = OnSupport("motor_units")
AFFERENT = Band(0.0, 1000.0)
MOTOR = Band(0.0, 200.0)
STRUCTURE = Band(0.0, 0.001)


SEGMENTAL_REFLEX = process(
    id="segmental_reflex",
    doc="""proprioceptive afferent traffic onto motoneurons, inside the cord.

    the arc that does not pass through the brain.  it reads the Ia, Ib and II
    traffic that `transduction` produces and `afferent_propagation` carries, and
    writes alpha and gamma motoneuron drive directly -- so a load applied to a
    muscle changes its own drive within tens of milliseconds and without cortical
    involvement.

    it writes the SAME components `efferent_propagation` writes, and that overlap
    is deliberate and is the physiology: a motoneuron pool sums descending command
    and segmental input, and no experiment can separate them at the axon.  what
    separates them here is latency, which is measurable and is the whole content
    of an H-reflex.

    it also reads descending command, because the cord is not autonomous: the
    gain of every arc below is under supranuclear control, and a stretch reflex
    whose gain does not change between standing and sitting is not a model of a
    stretch reflex.  that is the presynaptic-inhibition story, lumped here into a
    gain rather than given its own interneuron.

    where this breaks: the interneurons are implicit.  Ia-inhibitory, Ib and
    Renshaw cells are real populations with their own dynamics, convergence from
    several afferent classes, and their own descending control, and folding each
    into a transfer function asserts that their only contribution is a delay and
    a sign.  that is defensible at the scale this model runs at and it is not
    true, and the flexor-withdrawal and crossed-extensor reflexes -- which need
    multi-segmental interneuron chains -- are absent entirely for the same
    reason.""",
    inputs=(
        within("neural", "afferent.ia", "afferent.ii", region=BODY, band=AFFERENT),
        within("neural", "afferent.ib", region=BODY, band=AFFERENT),
        within("neural", "afferent.adelta", region=BODY, band=AFFERENT),
        within("neural", "efferent.alpha", region=BODY, band=MOTOR),
        within("effector", "drive", region=MOTOR_UNITS, band=MOTOR),
        within("structural", "myelination", "axonal_density", band=STRUCTURE),
    ),
    outputs=(
        within("neural", "efferent.alpha", region=BODY, band=MOTOR),
        within("neural", "efferent.gamma", region=BODY, band=MOTOR),
    ),
    topology="efferent_pathway",
    validity=None,
    provenance=Provenance.LITERATURE,
)


def _arc(basis, delay_s: float, tau_s: float, gain: float):
    """one reflex arc: a conduction and synaptic delay into a synaptic filter."""
    return gain * series(pure_delay(basis, delay_s), alpha_synapse(basis, tau_s))


def monosynaptic_stretch(basis, loop_delay_s: float = 0.030,
                         tau_synapse_s: float = 0.005, gain: float = 0.4,
                         descending_gain: float = 1.0):
    """Ia onto homonymous alpha, one synapse.

    the loop delay is the dominant parameter and it is anatomy: afferent conduction
    from spindle to cord, a synaptic delay under a millisecond, and efferent
    conduction back.  `ibm.topologies.nerve` supplies both conduction legs per
    fibre class, so this parameter should be FIT against an H-reflex latency and
    not assumed -- 30 ms is a soleus value and a hand muscle is half of it.
    """
    return descending_gain * _arc(basis, loop_delay_s, tau_synapse_s, gain)


def reciprocal_inhibition(basis, loop_delay_s: float = 0.032,
                          tau_synapse_s: float = 0.006, gain: float = -0.25,
                          descending_gain: float = 1.0):
    """Ia onto the ANTAGONIST alpha through one interneuron: disynaptic, inhibitory.

    the extra ~2 ms over the monosynaptic arc is the interneuron, and the negative
    gain is the whole point: stretching a muscle excites it and releases its
    antagonist, which makes the reflex a statement about a joint rather than about
    a muscle.
    """
    return descending_gain * _arc(basis, loop_delay_s, tau_synapse_s, gain)


def autogenic_inhibition(basis, loop_delay_s: float = 0.034,
                         tau_synapse_s: float = 0.006, gain: float = -0.15,
                         descending_gain: float = 1.0):
    """Ib onto homonymous alpha through one interneuron: force feedback, negative.

    the sign is opposite to the stretch reflex and the input is force rather than
    length, so the two together approximate impedance control: length feedback
    resists displacement, force feedback yields to load, and the ratio between
    their gains is what sets limb stiffness.  a model with only the stretch reflex
    has a limb that cannot yield.
    """
    return descending_gain * _arc(basis, loop_delay_s, tau_synapse_s, gain)


def renshaw_recurrent(basis, loop_delay_s: float = 0.004,
                      tau_synapse_s: float = 0.010, gain: float = -0.2,
                      descending_gain: float = 1.0):
    """alpha collateral onto its own pool through a Renshaw cell: recurrent inhibition.

    the fastest arc here because it never leaves the cord -- the delay is a
    collateral and two synapses.  it is automatic gain control on the pool, and it
    is why a motoneuron pool under strong descending drive does not simply
    saturate.
    """
    return descending_gain * _arc(basis, loop_delay_s, tau_synapse_s, gain)


_ARC_PARAMS = {
    "tau_synapse_s": lognormal(0.005, 2.0, units="s", provenance=Provenance.LITERATURE),
    "descending_gain": weak(1.0, 3.0, units="dimensionless",
                            note="supranuclear control of reflex gain, lumping "
                                 "presynaptic inhibition.  a stretch reflex whose gain "
                                 "does not change with posture or task is not a model "
                                 "of one, so this is the parameter a task manipulation "
                                 "should move"),
}

implementation(
    name="monosynaptic_stretch_lti",
    process="segmental_reflex",
    doc="""the myotatic reflex: Ia to homonymous alpha, one synapse.

    the only monosynaptic reflex in the body, which is why its latency is clean
    enough to be a clinical measurement and therefore a fitting target: the
    H-reflex is this arc and nothing else.  fit `loop_delay_s` against it rather
    than assuming the 30 ms soleus value, since a hand muscle is half that and
    `ibm.topologies.nerve` already supplies both conduction legs.""",
    form=Form.LTI,
    transfer=monosynaptic_stretch,
    params={
        "loop_delay_s": normal(0.030, 0.008, units="s",
                               provenance=Provenance.LITERATURE,
                               source="soleus H-reflex latency ~30 ms; hand muscles ~15 ms"),
        "gain": weak(0.4, 4.0, units="dimensionless",
                     note="reflex gain, and the quantity a cordless model silently "
                          "adds to corticospinal gain instead"),
        **_ARC_PARAMS,
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="Matthews, Mammalian Muscle Receptors; Pierrot-Deseilligny & Burke")

implementation(
    name="reciprocal_inhibition_lti",
    process="segmental_reflex",
    doc="""Ia to antagonist alpha through the Ia-inhibitory interneuron: disynaptic.

    negative by construction.  it is what turns a muscle-level reflex into a
    joint-level one, and its absence is why a cordless model co-contracts when it
    should reciprocate.""",
    form=Form.LTI,
    transfer=reciprocal_inhibition,
    params={
        "loop_delay_s": normal(0.032, 0.008, units="s", provenance=Provenance.LITERATURE,
                               note="one interneuron slower than the monosynaptic arc"),
        # normal, not weak: `weak` is lognormal-backed and cannot express a
        # negative median, and an inhibitory arc's gain IS negative.  the sign is
        # anatomy -- this synapse is glycinergic -- so it is not a parameter the
        # prior should leave free to flip.
        "gain": normal(-0.25, 0.15, units="dimensionless",
                       provenance=Provenance.LITERATURE),
        **_ARC_PARAMS,
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="Pierrot-Deseilligny & Burke, The Circuitry of the Human Spinal Cord")

implementation(
    name="autogenic_inhibition_lti",
    process="segmental_reflex",
    doc="""Ib to homonymous alpha through the Ib interneuron: force feedback, negative.

    the counterpart to the stretch reflex and the reason the pair is not
    redundant: length feedback resists displacement and force feedback yields to
    load, so their gain ratio sets limb stiffness.  a limb with only a stretch
    reflex cannot yield to an obstacle.""",
    form=Form.LTI,
    transfer=autogenic_inhibition,
    params={
        "loop_delay_s": normal(0.034, 0.008, units="s", provenance=Provenance.LITERATURE),
        "gain": normal(-0.15, 0.10, units="dimensionless",
                       provenance=Provenance.LITERATURE),
        **_ARC_PARAMS,
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="Houk & Henneman; Jami, Golgi tendon organs")

implementation(
    name="renshaw_recurrent_lti",
    process="segmental_reflex",
    doc="""alpha collateral to Renshaw cell to the same pool: recurrent inhibition.

    the fastest arc in the file because it never leaves the cord.  automatic gain
    control on the motoneuron pool, and the reason strong descending drive does
    not simply saturate it.""",
    form=Form.LTI,
    transfer=renshaw_recurrent,
    params={
        "loop_delay_s": normal(0.004, 0.002, units="s", provenance=Provenance.LITERATURE,
                               note="a collateral and two synapses; it never leaves the cord"),
        "gain": normal(-0.2, 0.12, units="dimensionless",
                       provenance=Provenance.LITERATURE),
        **_ARC_PARAMS,
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="Renshaw 1946; Hultborn et al. on recurrent inhibition")

__all__ = ["SEGMENTAL_REFLEX"]
