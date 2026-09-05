"""release, clearance and receptor occupancy: the chemical step between a
presynaptic rate and a postsynaptic conductance, written out instead of lumped.

`local_excitation` and `local_inhibition` collapse this whole chain into a
receptor kernel -- a rate goes in, a conductance comes out, and the transmitter
never appears.  that is the right reduction almost always, because the cleft
transient is over in a millisecond and nothing at mesoscale can see it.  this
process exists for the cases where the reduction is what breaks.

three of them, and they are the reason the file is worth its length.

*the pool runs out.*  release depletes a readily-releasable vesicle pool that
refills over hundreds of milliseconds, so the conductance produced by the tenth
spike of a train is not the conductance produced by the first.  short-term
depression and facilitation are the most reliably measured synaptic
nonlinearity there is, and a fixed rate-to-conductance gain has no term for
them.

*clearance saturates.*  glutamate is cleared from the cleft in about a
millisecond, almost entirely by astrocytic transporters.  when those
transporters are saturated or downregulated -- ischaemia, seizure, a glial
scar -- glutamate spills out of the cleft, reaches extrasynaptic nmda
receptors, and becomes excitotoxic.  that is a different regime, not a larger
number.

*volume transmission is not synaptic transmission at all.*  the monoamines and
acetylcholine are released from varicosities into the extracellular space and
diffuse to receptors that are not opposite any release site.  their
concentration is a genuine field variable with its own slow dynamics, and
`neuromodulation` reads exactly that field.  they are declared here rather than
there because release-and-clearance is the same rate law for dopamine as for
glutamate with different constants, and putting the two in different files
would duplicate it.

the fast transmitters are read as spectral neural state and written as scalar
extracellular state, so the registered ``spectral -> scalar`` conversion
applies and the sub-millisecond structure of the cleft transient leaves the
model there.  that is not a loss worth fighting: the transient's *shape* is
already inside the receptor kernel, and what survives -- how much transmitter
is around on a timescale of tens of milliseconds and up -- is what spillover
and volume transmission actually depend on.
"""

from __future__ import annotations

import numpy as np

from ibm.processes.base import (
    alpha_synapse,
    double_exponential,
    implementation,
    low_pass,
    parallel,
    process,
    series,
)
from ibm.registry import Form
from ibm.vocabulary import (
    Band,
    HEMODYNAMIC,
    LFP,
    STRUCTURAL,
    Provenance,
    Tying,
    Validity,
    lognormal,
    normal,
    speculative,
    uniform,
    weak,
    within,
)

#: presynaptic rate is read across the electrophysiological band; release is a
#: per-spike event and the fastest thing about it that a population variable
#: can carry is the rate's own bandwidth.
RELEASE = LFP

#: fast transmitter concentration.  narrow, because a lumped extracellular
#: glutamate or gaba concentration above ~20 Hz is the aliased image of
#: individual release events rather than a field: the cleft transient's shape
#: is already inside the receptor kernel, and what survives at field bandwidth
#: is the ambient level that spillover and volume transmission depend on.
FAST_TRANSMITTER = Band(0.0, 20.0)

#: extracellular calcium, read as the presynaptic release gate.  wider than the
#: transmitter it gates, because calcium is an ion and carries the ionic
#: field's bandwidth.
CALCIUM = Band(0.0, 50.0)

#: volume-transmitted modulators.  seconds to minutes, and deliberately narrow:
#: a dopamine concentration field with gamma-band structure would be an artefact
#: of the declaration rather than a fact about the tissue.
VOLUME_TRANSMITTER = Band(0.0, 2.0)


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def fast_transmitter_transfer(basis, tau_clearance_s: float = 1e-3,
                              release_gain: float = 1.0):
    """presynaptic rate -> cleft transmitter concentration, one fast pole.

    a millisecond time constant, which is barely a filter at all on the bands
    a materialization retains -- and that is the finding rather than a
    limitation.  glutamate clearance is so fast that the cleft concentration
    tracks the release rate almost instantaneously, which is exactly why the
    lumped receptor kernel in `local_excitation` is a good reduction and why
    the transmitter concentration does not usually need to be a state variable.

    where it stops being fast: clearance is transporter-mediated, and
    transporters saturate.  under heavy release, or with astrocytic transporter
    capacity reduced, the effective time constant lengthens by orders of
    magnitude and transmitter escapes the cleft.  this linear form represents
    that as a longer tau and cannot represent the fact that the *destination*
    changes too -- extrasynaptic receptors are a different receptor population
    with different kinetics, not more of the same synapse.
    """
    return release_gain * low_pass(basis, tau_clearance_s)


def volume_transmitter_transfer(basis, tau_release_s: float = 0.05,
                                tau_clearance_s: float = 1.0,
                                release_gain: float = 1.0):
    """varicosity release -> extracellular modulator concentration.

    two poles with very different constants, because volume transmission has
    two genuinely separate steps: release and diffusion out of the varicosity
    on tens of milliseconds, then reuptake and enzymatic degradation over
    something between a hundred milliseconds and tens of seconds depending on
    the transmitter and the region.

    the regional dependence is not a detail.  striatal dopamine is cleared in
    a couple of hundred milliseconds by a dense transporter population; cortical
    dopamine, where that transporter is sparse, persists for seconds and is
    cleared partly by the noradrenaline transporter instead.  so this
    parameter is per-partition, and treating it as global is one of the
    easier ways to get a cortical dopamine model badly wrong.
    """
    return release_gain * double_exponential(basis, tau_release_s, tau_clearance_s)


def receptor_occupancy_transfer(basis, tau_receptor_s: float = 3e-3,
                                affinity_gain: float = 1.0):
    """cleft concentration -> postsynaptic conductance.

    the receptor's own kinetics as an alpha kernel, times an affinity gain.
    written separately from the transmitter kernel so that the two halves of
    the chain carry their own parameters: a drug that blocks a receptor changes
    `affinity_gain`, a drug that blocks a transporter changes the clearance
    constant in `fast_transmitter_transfer`, and the model can tell them apart
    only if they are different numbers.
    """
    return affinity_gain * alpha_synapse(basis, tau_receptor_s)


def clearance_receptor_transfer(basis, tau_clearance_s: float = 1e-3,
                                release_gain: float = 1.0, tau_receptor_s: float = 3e-3,
                                affinity_gain: float = 1.0):
    """the whole fast chain: release, clearance, receptor, in series.

    kept as the composition of the two halves rather than as one fitted kernel
    so that the transporter parameter and the receptor parameter stay
    separately addressable.  the composition is a product in the frequency
    domain, which is the only reason writing it this way costs nothing.
    """
    return series(
        fast_transmitter_transfer(basis, tau_clearance_s, release_gain),
        receptor_occupancy_transfer(basis, tau_receptor_s, affinity_gain),
    )


def spillover_transfer(basis, tau_cleft_s: float = 1e-3, tau_spillover_s: float = 0.03,
                       spillover_fraction: float = 0.05, tau_receptor_s: float = 3e-3):
    """cleft plus extrasynaptic paths in parallel, then the receptor.

    a small fraction of released transmitter escapes the cleft and reaches
    receptors tens of nanometres to micrometres away, on a much slower
    timescale set by tortuous diffusion rather than by transporters.  the
    fraction is small and the timescale is long, so the escaped transmitter
    contributes little at any instant and dominates the *sustained* component
    during high-frequency release -- which is why tonic extrasynaptic nmda and
    gaba-a currents exist at all and why they scale super-linearly with
    activity.
    """
    cleft = (1.0 - spillover_fraction) * low_pass(basis, tau_cleft_s)
    escaped = spillover_fraction * low_pass(basis, tau_spillover_s)
    return series(parallel(cleft, escaped), alpha_synapse(basis, tau_receptor_s))


# ---------------------------------------------------------------------------
# nonlinear rate laws
# ---------------------------------------------------------------------------


def depressing_release(x, theta) -> dict:
    """vesicle-pool depletion: release proportional to rate times available pool.

    the pool is depleted by release and refills with a fixed time constant, so
    the transmitter produced by a sustained rate is not proportional to that
    rate -- it approaches a ceiling set by the refill rate alone.  a depressing
    synapse therefore transmits *changes* in rate rather than rate, which is a
    qualitatively different computation from the linear one and is the standard
    account of why cortical responses adapt to sustained input without any
    cellular adaptation current being involved.
    """
    r_e = np.asarray(x["neural.exc.activity"], dtype=float)
    g = np.asarray(x["extracellular.glutamate"], dtype=float)
    p = float(theta.get("release_probability", 0.3))
    tau_recover = float(theta.get("tau_recovery_s", 0.7))
    tau_clear = float(theta.get("tau_clearance_s", 1e-3))
    gain = float(theta.get("release_gain", 1.0))
    # steady-state available fraction under rate r: 1 / (1 + p r tau_recover)
    available = 1.0 / (1.0 + p * np.maximum(r_e, 0.0) * tau_recover)
    return {"extracellular.glutamate": gain * p * available * np.maximum(r_e, 0.0) - g / tau_clear}


def michaelis_menten_clearance(x, theta) -> dict:
    """transporter-limited clearance with a finite maximum velocity.

    the same shape as the Na/K-ATPase term in `ionic_exchange` and for the same
    reason: a transporter is an enzyme, its rate saturates, and the interesting
    behaviour is at the saturation point rather than on either side of it.
    below vmax the transmitter concentration is pinned near its resting value
    and the linear form is fine; at vmax clearance stops responding to load and
    concentration climbs until something else stops it.  glutamate excitotoxicity
    is that regime.
    """
    out = {}
    for cid, vmax_key, km_key in (
        ("extracellular.glutamate", "glu_vmax_um_s", "glu_km_um"),
        ("extracellular.dopamine", "da_vmax_um_s", "da_km_um"),
    ):
        c = np.asarray(x[cid], dtype=float)
        vmax = float(theta.get(vmax_key, 1.0))
        km = float(theta.get(km_key, 1.0))
        out[cid] = -vmax * c / (km + np.maximum(c, 0.0))
    return out


# ---------------------------------------------------------------------------
# transmitter dynamics
# ---------------------------------------------------------------------------

TRANSMITTER_DYNAMICS = process(
    id="transmitter_dynamics",
    doc="""presynaptic activity releases transmitter into the extracellular space;
    transporters and enzymes clear it; receptors read what is left.

    the process the architecture's inventory writes as a three-step chain --
    presynaptic population state, to extracellular transmitter state, to
    postsynaptic state -- and it is declared as one process rather than three
    because the middle term is not separately observable at mesoscale.  what a
    microdialysis probe or a voltammetry electrode measures is the ambient
    concentration this process writes; what an evoked potential measures is the
    conductance it writes; the release step between them is inferred from both.

    both fast transmitters and volume-transmitted modulators are here.  the
    grouping is by mechanism, not by function: glutamate, gaba, dopamine,
    serotonin, acetylcholine and noradrenaline are all vesicular release
    followed by transporter or enzymatic clearance, and the same rate law
    covers all six with time constants spanning four orders of magnitude.
    what the modulators *do* once they are in the space is a different process
    entirely, and `neuromodulation` reads these concentrations to do it.

    adenosine is the odd one and is included deliberately.  its dominant source
    is not vesicular release but the extracellular breakdown of atp, so this
    process takes `metabolic.atp` as an input and writes adenosine as its
    consequence -- which is what makes adenosine an integrator of metabolic
    expenditure rather than of firing, and therefore the best available
    mechanistic account of homeostatic sleep pressure.

    calcium is an input because release is steeply calcium-dependent -- roughly
    a fourth-power relation at the presynaptic terminal -- so the extracellular
    calcium that `ionic_exchange` depletes during activity feeds back on release
    probability.  it is a real and often-ignored negative feedback.

    where this breaks.  it is a single well-mixed extracellular compartment per
    materialized position, and the cleft is not the extracellular space: the
    concentration in a 20 nm cleft during a release event is three orders of
    magnitude above the ambient concentration a probe measures, and this
    process carries one number for both.  the `spillover_lti` implementation is
    the partial repair -- two compartments instead of one -- and it is still not
    a cleft.""",
    inputs=(
        within("neural", "exc.activity", band=RELEASE),
        within("neural", "inh.activity", "pv.activity", "sst.activity", "vip.activity",
               band=RELEASE),
        within("extracellular", "ca", band=CALCIUM),
        within("extracellular", "glutamate", "gaba", band=FAST_TRANSMITTER),
        within("extracellular", "dopamine", "serotonin", "acetylcholine", "noradrenaline",
               band=VOLUME_TRANSMITTER),
        within("extracellular", "adenosine", band=VOLUME_TRANSMITTER),
        within("extracellular", "volume_fraction", band=Band(0.0, 1.0)),
        within("material", "tortuosity", band=STRUCTURAL),
        within("metabolic", "atp", band=HEMODYNAMIC),
        within("structural", "synaptic_density", band=STRUCTURAL),
    ),
    outputs=(
        within("extracellular", "glutamate", "gaba", band=FAST_TRANSMITTER),
        within("extracellular", "dopamine", "serotonin", "acetylcholine", "noradrenaline",
               band=VOLUME_TRANSMITTER),
        within("extracellular", "adenosine", band=VOLUME_TRANSMITTER),
        within("neural", "exc.ampa", band=Band(0.0, 1000.0)),
        within("neural", "exc.nmda", band=Band(0.0, 50.0)),
        within("neural", "inh.gaba_a", band=Band(0.0, 1000.0)),
        within("neural", "inh.gaba_b", band=Band(0.0, 50.0)),
    ),
    topology="local",
    timescale_s=1e-2,
    validity=Validity(
        min_spacing_mm=0.01, max_spacing_mm=3.0, band=Band(0.0, 300.0),
        note="a well-mixed extracellular compartment per position.  volume transmission "
             "genuinely has a spatial scale -- dopamine reaches a few micrometres from a "
             "varicosity in striatum and further in cortex -- so below ~0.01 mm the "
             "well-mixed assumption is the whole model and above ~3 mm the compartment "
             "averages over regions with different transporter densities, which is the "
             "specific way cortical and striatal dopamine differ.",
    ),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"transmitter", "local", "chemical"}),
    notes="this process resolves the release-clearance-receptor chain that "
          "local_excitation and local_inhibition lump into a receptor kernel.  a "
          "materialization selecting both and the lumped forms together double-counts "
          "the synaptic drive; the two are alternatives at different resolutions, and "
          "the choice belongs to the materializer.  spectral neural inputs and scalar "
          "extracellular outputs, so both registered conversions appear in provenance.",
)

implementation(
    name="clearance_receptor_lti",
    process="transmitter_dynamics",
    doc="""a clearance pole and a receptor kernel in series, per transmitter.

    the default, and cheap.  its value is not that it is more accurate than the
    lumped receptor kernel -- for the fast transmitters it is almost exactly
    the same filter -- but that it separates two parameters the lumped form
    conflates.  a transporter blocker and a receptor antagonist have different
    signatures here and identical signatures there, and pharmacological
    evidence is one of the few kinds of data that can constrain a mesoscale
    model at all.

    the volume transmitters are where the separation earns real money, because
    their clearance constant is seconds and their receptor kinetics are a
    g-protein cascade: the two timescales are genuinely different and no single
    kernel represents both.""",
    form=Form.LTI,
    transfer=clearance_receptor_transfer,
    params={
        "tau_clearance_s": lognormal(1e-3, 2.0, units="s", provenance=Provenance.LITERATURE,
                                     source="glutamate clearance from the synaptic cleft, ~1 ms, "
                                            "dominated by astrocytic EAAT1/EAAT2",
                                     note="gaba is a few times slower via GAT-1; the volume "
                                          "transmitters are three to four orders of magnitude "
                                          "slower, which is why this parameter is per-partition "
                                          "rather than global"),
        "release_gain": weak(1.0, 8.0, units="uM per Hz",
                             note="release probability times vesicle content times synapse "
                                  "count per unit volume; not separately identifiable from "
                                  "the structural synaptic-density input"),
        "tau_receptor_s": lognormal(3e-3, 1.5, units="s", provenance=Provenance.LITERATURE,
                                    source="ampa-receptor kinetics; the postsynaptic half of "
                                           "the chain"),
        "affinity_gain": weak(1.0, 5.0, units="nS per uM",
                              note="the parameter a receptor antagonist moves, kept separate "
                                   "from the one a transporter blocker moves"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="clements et al 1992 time course of glutamate in the synaptic cleft; "
           "danbolt 2001 glutamate uptake",
)

implementation(
    name="volume_transmission_lti",
    process="transmitter_dynamics",
    doc="""the slow limb: varicosity release and reuptake for the monoamines and
    acetylcholine.

    declared as its own implementation rather than as different parameters of
    the fast one, because the geometry is different in a way that matters.
    volume transmission has no cleft: transmitter is released from varicosities
    that are not apposed to any postsynaptic density, diffuses through
    interstitial space, and acts on receptors that may be micrometres away.
    the resulting concentration is therefore a genuine field variable whose
    spatial extent is set by the same tortuosity that governs `ionic_diffusion`,
    and whose time course is set by transporter density rather than by
    diffusion at all.

    acetylcholine is the exception within the exception: acetylcholinesterase
    is one of the fastest enzymes known and hydrolyses it in well under a
    millisecond, so cholinergic volume transmission has a much tighter spatial
    envelope than the monoamines.  the prior below is wide enough to hold both,
    which is a fair statement of the disagreement in the literature about how
    much cholinergic signalling is volumetric at all.""",
    form=Form.LTI,
    transfer=volume_transmitter_transfer,
    params={
        "tau_release_s": lognormal(0.05, 2.0, units="s", provenance=Provenance.LITERATURE,
                                   note="release plus escape from the varicosity; tens of "
                                        "milliseconds"),
        "tau_clearance_s": lognormal(1.0, 5.0, units="s", provenance=Provenance.LITERATURE,
                                     source="dopamine clearance is ~0.1-0.2 s in striatum and "
                                            "seconds in cortex, where DAT is sparse",
                                     note="the spread of this prior is real regional variation, "
                                          "not measurement uncertainty; a global value is one "
                                          "of the easier ways to get cortical dopamine wrong"),
        "release_gain": weak(1.0, 10.0, units="nM per Hz"),
        "resting_dopamine_nm": lognormal(20.0, 3.0, units="nM", provenance=Provenance.LITERATURE,
                                         note="ambient striatal dopamine; estimates from "
                                              "microdialysis and voltammetry disagree by an "
                                              "order of magnitude and the prior says so"),
        "resting_acetylcholine_nm": weak(30.0, 10.0, units="nM",
                                         note="acetylcholinesterase is fast enough that ambient "
                                              "levels are poorly defined and probe-dependent"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="cragg & rice 2004 dancing past the DAT; zoli et al 1999 volume transmission",
)

implementation(
    name="spillover_lti",
    process="transmitter_dynamics",
    doc="""two extracellular compartments -- cleft and extrasynaptic -- in parallel.

    the minimum repair to the well-mixed assumption, and it buys one specific
    thing: a sustained, super-linear extrasynaptic component that appears only
    at high release rates.  tonic extrasynaptic gaba-a current and
    extrasynaptic nmda activation are both real, both matter (the first is the
    principal target of several anaesthetics, the second is the excitotoxic
    pathway), and neither exists in a one-compartment model at any parameter
    setting.

    still not a cleft.  the ratio of cleft to ambient concentration during a
    release event is around a thousand, and no two-compartment model with
    fitted volumes recovers the geometry that produces it.""",
    form=Form.LTI,
    transfer=spillover_transfer,
    params={
        "tau_cleft_s": lognormal(1e-3, 2.0, units="s", provenance=Provenance.LITERATURE,
                                 source="glutamate clearance from the cleft, ~1 ms"),
        "tau_spillover_s": lognormal(0.03, 3.0, units="s", provenance=Provenance.WEAK,
                                     note="tortuous diffusion over a micrometre-scale "
                                          "extrasynaptic distance; the distance is the guess"),
        "spillover_fraction": uniform(0.0, 0.2, units="dimensionless", provenance=Provenance.WEAK,
                                      note="rises sharply when transporters are saturated or "
                                           "downregulated, which this constant cannot express"),
        "tau_receptor_s": lognormal(3e-3, 1.5, units="s", provenance=Provenance.LITERATURE),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.WEAK,
)

implementation(
    name="depressing_release_rate",
    process="transmitter_dynamics",
    doc="""vesicle-pool depletion and transporter saturation, in the time domain.

    the nonlinear form, and the one that changes what the synapse computes
    rather than how fast it does it.  with depletion, a synapse's steady-state
    output saturates at a value set by the pool refill rate, so it transmits
    rate changes rather than rate -- a high-pass, and one whose corner
    frequency moves with the background rate.  no linear time-invariant kernel
    has that property, which is the point.

    the release probability is the parameter that decides which regime a
    synapse is in: high probability gives strong depression and a
    differentiating synapse, low probability gives facilitation and an
    integrating one, and cortical synapses come in both kinds onto the same
    postsynaptic cell from different sources.  a global value is therefore
    known in advance to be wrong; the tying is per-partition and even that is
    optimistic.""",
    form=Form.RATE,
    fn=depressing_release,
    params={
        "release_probability": uniform(0.05, 0.6, units="dimensionless",
                                       provenance=Provenance.LITERATURE,
                                       note="cortical synapses span this whole range, and which "
                                            "end a synapse sits at determines whether it "
                                            "depresses or facilitates"),
        "tau_recovery_s": lognormal(0.7, 2.0, units="s", provenance=Provenance.LITERATURE,
                                    source="readily-releasable pool refilling, 0.3-1.5 s",
                                    note="the time constant of short-term depression recovery, "
                                         "and therefore the corner of the synapse's high-pass"),
        "tau_facilitation_s": lognormal(0.1, 2.0, units="s", provenance=Provenance.LITERATURE,
                                        source="residual presynaptic calcium, ~100 ms"),
        "tau_clearance_s": lognormal(1e-3, 2.0, units="s", provenance=Provenance.LITERATURE),
        "release_gain": weak(1.0, 8.0, units="uM per Hz"),
        "calcium_exponent": normal(3.8, 0.6, units="dimensionless",
                                   provenance=Provenance.LITERATURE,
                                   source="release probability scales as roughly the fourth "
                                          "power of presynaptic calcium",
                                   note="why extracellular calcium depletion during activity is "
                                        "a real negative feedback on release rather than a "
                                        "rounding error"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="tsodyks & markram 1997; abbott et al 1997 synaptic depression and cortical gain",
)

implementation(
    name="transporter_saturation_rate",
    process="transmitter_dynamics",
    doc="""michaelis-menten clearance for glutamate and dopamine.

    carried because the clearance side has a saturation point too, and crossing
    it is the mechanism of a named pathology rather than a parameter regime.
    with transporters below vmax the ambient glutamate concentration is pinned
    at a micromolar or two and nothing interesting happens; with them saturated
    or lost, ambient glutamate climbs, extrasynaptic nmda receptors activate,
    and calcium-mediated excitotoxicity follows.  the vmax values are measured
    in preparations whose relationship to intact human cortex is a genuine
    extrapolation, so they carry literature provenance with wide spreads
    rather than tight ones.""",
    form=Form.RATE,
    fn=michaelis_menten_clearance,
    params={
        "glu_vmax_um_s": lognormal(500.0, 4.0, units="uM/s", provenance=Provenance.LITERATURE,
                                   source="astrocytic EAAT2 transport capacity; enormous, which "
                                          "is why cleft clearance is a millisecond"),
        "glu_km_um": lognormal(20.0, 2.0, units="uM", provenance=Provenance.LITERATURE,
                               note="far above the ambient concentration and far below the "
                                    "cleft peak, so the transporter is linear at rest and "
                                    "saturated during release -- both, in the same synapse"),
        "resting_glutamate_um": lognormal(2.0, 5.0, units="uM", provenance=Provenance.LITERATURE,
                                          note="ambient extracellular glutamate; estimates span "
                                               "an order of magnitude depending on method, and "
                                               "the prior is wide because the disagreement is "
                                               "real rather than resolvable"),
        "da_vmax_um_s": lognormal(4.0, 3.0, units="uM/s", provenance=Provenance.LITERATURE,
                                  source="striatal DAT vmax; roughly two orders of magnitude "
                                         "lower in cortex"),
        "da_km_um": lognormal(0.2, 2.0, units="uM", provenance=Provenance.LITERATURE),
        "adenosine_from_atp_gain": speculative(1.0, 20.0, units="nM per uM ATP",
                                               note="ecto-nucleotidase conversion of released "
                                                    "atp; the pathway is established and its "
                                                    "mesoscale rate is not"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
)

implementation(
    name="learned_release_dynamics",
    process="transmitter_dynamics",
    doc="""a learned map from the presynaptic spectrum to transmitter concentration.

    short-term plasticity makes a synapse's output a functional of its input
    history, and the analytic forms above capture two terms of that functional
    -- depression and facilitation -- with time constants fitted from paired
    pulses.  real synapses have more structure than that, and the structure is
    frequency-dependent in a way that reading the retained spectrum directly
    can express and a two-state kinetic scheme cannot.

    the prior is centred on the depressing-release form, so a materialization
    with no data gets tsodyks-markram and nothing else.  what the posterior on
    `residual_gain` reports afterwards is how much the data actually needed
    beyond it, which is the honest way to run this comparison.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "residual_gain": weak(0.3, 5.0, note="zero recovers the analytic release model"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


__all__ = ["TRANSMITTER_DYNAMICS"]
