"""neural traffic: what a cortical population does to its neighbours, to the
layer above it, to the patch three millimetres away, and to the area at the far
end of a fascicle.

six processes, and the thing that separates them is not the dynamics -- most of
them are a synapse behind a delay behind a membrane -- but the topology and the
selectors.  a lumped ``neural -> neural`` coupling would be all six at once and
would be identifiable from nothing, because every distinguishing fact about
cortical signal traffic is a fact about *which* population at *which* depth
reaches *which* other one over *what* delay.  ARCHITECTURE.md §4 makes that a
rule; this file is what obeying it costs and buys.

three commitments run through the whole file.

*the conductances are the state, not the input.*  a process here writes
``neural.exc.ampa`` rather than "excitatory input", because ampa and nmda have
time constants two orders of magnitude apart and the difference is what decides
whether recurrent excitation supports gamma or supports persistent activity.
collapsing them into one drive term deletes that distinction before any
implementation gets to have an opinion about it.

*delay is a phase ramp.*  every propagation process below is written as an LTI
transfer function on the spectral form, so a conduction delay is
``exp(-i omega tau)`` and costs one complex multiply per retained component.
`tract_propagation` says at length why this is the single largest practical
reason the spectral form earns its cost.

*the nonlinearity is in exactly one place.*  the map from membrane potential to
population firing rate is steeply nonlinear, and it is the only thing in this
file that genuinely cannot be written as a transfer function.  so the LTI
implementations are all linearizations about a background rate and say so, and
the RATE implementations exist alongside them for the regimes -- saturation,
seizure, burst suppression, up/down transitions -- where the linearization is
not merely imprecise but structurally the wrong f.
"""

from __future__ import annotations

import numpy as np

from ibm.processes.base import (
    alpha_synapse,
    delay_dispersion,
    double_exponential,
    feedback,
    implementation,
    low_pass,
    parallel,
    process,
    pure_delay,
    resonator,
    series,
)
from ibm.registry import Form
from ibm.vocabulary import (
    Anat,
    Band,
    GAMMA,
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

#: the band over which population activity is read as a *drive* to something
#: else.  it stops at 300 Hz rather than at the spike band because a population
#: rate above a few hundred hertz is the aliased image of individual spikes, not
#: a rate, and no coupling in this file is meant to transmit it.
DRIVE = LFP

#: synaptic conductance state.  wider than DRIVE because the conductance is
#: what the fastest membrane events act on, and narrower than the spike band
#: because a lumped population conductance has no meaning at 3 kHz.
SYNAPTIC = Band(0.0, 1000.0)

#: adaptation, nmda and gaba-b: the slow half of the neural field.
SLOW = Band(0.0, 50.0)

#: cortical layers, as regions.  written out rather than inlined so that the
#: laminar edges below read as a circuit diagram.
L1 = Anat("cortical_layers", "L1")
L2_3 = Anat("cortical_layers", "L2_3")
L4 = Anat("cortical_layers", "L4")
L5 = Anat("cortical_layers", "L5")
L6 = Anat("cortical_layers", "L6")

#: first-order thalamic relay nuclei and the reticular sheet that gates them.
#: named individually rather than as "thalamus" because the corticothalamic
#: loop's dynamics depend on which of them is in it: trn is inhibitory and the
#: others are not, and a lumped thalamus cannot oscillate for the right reason.
RELAY = (
    Anat("thalamic_nuclei", "lgn")
    | Anat("thalamic_nuclei", "mgn")
    | Anat("thalamic_nuclei", "vpl")
    | Anat("thalamic_nuclei", "vpm")
    | Anat("thalamic_nuclei", "pulvinar")
    | Anat("thalamic_nuclei", "medial_dorsal")
)
TRN = Anat("thalamic_nuclei", "trn")


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def synaptic_drive_transfer(basis, tau_ampa_s: float = 3e-3, tau_nmda_rise_s: float = 2e-3,
                            tau_nmda_decay_s: float = 0.1, nmda_fraction: float = 0.3,
                            gain: float = 1.0):
    """presynaptic rate -> postsynaptic conductance, as two parallel receptor paths.

    a glutamatergic synapse is not one filter.  the same release event opens
    ampa channels that shut in a few milliseconds and nmda channels that stay
    open for a hundred, so the conductance it produces is the *sum* of a fast
    and a slow kernel and its shape depends on how the two are weighted.  that
    weighting is `nmda_fraction`, and it is not a nuisance parameter: it is the
    single number that decides whether a recurrent population is a gamma
    oscillator or an integrator.

    where it breaks: nmda's magnesium block makes its effective gain a function
    of the postsynaptic membrane potential, so `nmda_fraction` is really a state
    variable and this form freezes it at its value around the background
    potential.  a population that depolarizes far enough to unblock nmda
    becomes regeneratively excitable, and this transfer function will report a
    perfectly ordinary linear response while the real circuit ignites.
    """
    fast = (1.0 - nmda_fraction) * alpha_synapse(basis, tau_ampa_s)
    slow = nmda_fraction * double_exponential(basis, tau_nmda_rise_s, tau_nmda_decay_s)
    return gain * parallel(fast, slow)


def ei_loop_transfer(basis, tau_membrane_s: float = 0.015, tau_ampa_s: float = 3e-3,
                     tau_inh_membrane_s: float = 0.008, tau_gaba_a_s: float = 6e-3,
                     loop_gain: float = 8.0, synaptic_lag_s: float = 1.5e-3,
                     gain: float = 1.0):
    """the local excitatory-inhibitory loop, closed.

    forward path: excitatory synapse into an excitatory membrane.  return path:
    the inhibitory population's synapse and membrane, plus the small axonal and
    release lag that a two-synapse round trip actually costs.  ``G / (1 + G L)``
    with negative feedback, which is `base.feedback`.

    this is where gamma comes from, and the derivation is worth stating because
    it is also where the form's limits are.  the loop gain crosses -1 in phase
    at the frequency where the round trip is half a cycle; with ~5 ms of
    combined synaptic and membrane lag that is a resonance in the 40-80 Hz
    range, which is pyramidal-interneuron gamma, arrived at from time constants
    rather than fitted.  raise `loop_gain` and the denominator gets small there
    and the peak sharpens; raise it far enough and the peak becomes a pole and
    the linearization has stopped being the right f entirely.
    `base.stability_margin` on the loop term is the honest diagnostic.
    """
    forward = series(alpha_synapse(basis, tau_ampa_s), low_pass(basis, tau_membrane_s))
    loop = loop_gain * series(
        alpha_synapse(basis, tau_gaba_a_s),
        low_pass(basis, tau_inh_membrane_s),
        pure_delay(basis, synaptic_lag_s),
    )
    return gain * feedback(forward, loop, negative=True)


def gaba_transfer(basis, tau_gaba_a_s: float = 6e-3, tau_gaba_b_rise_s: float = 0.05,
                  tau_gaba_b_decay_s: float = 0.15, gaba_b_fraction: float = 0.15,
                  gain: float = 1.0):
    """inhibitory rate -> inhibitory conductance, fast and slow paths in parallel.

    gaba-a is a channel and gets one alpha kernel.  gaba-b is a g-protein
    cascade onto a potassium conductance, so it has a genuine rise time of tens
    of milliseconds and gets a two-pole kernel with distinct rise and decay --
    collapsing it onto one pole puts its phase in the wrong place at exactly
    the delta and slow-oscillation frequencies it is responsible for.

    the fraction is small and strongly nonlinear in reality: gaba-b is
    recruited only by sustained, high-frequency inhibitory firing, because it
    needs spillover out of the cleft.  a fixed `gaba_b_fraction` is the
    linearization of that recruitment about the background rate, and it will
    understate slow inhibition during a burst and overstate it at rest.
    """
    fast = (1.0 - gaba_b_fraction) * alpha_synapse(basis, tau_gaba_a_s)
    slow = gaba_b_fraction * double_exponential(basis, tau_gaba_b_rise_s, tau_gaba_b_decay_s)
    return gain * parallel(fast, slow)


def laminar_edge_transfer(basis, delay_s: float = 1.2e-3, tau_ampa_s: float = 3e-3,
                          tau_membrane_s: float = 0.015, gain: float = 1.0):
    """one directed laminar edge: a short delay, a synapse, a membrane.

    the delay is genuinely short -- a few hundred microns of mostly
    unmyelinated vertical axon -- so it contributes almost no phase below beta
    and is kept anyway because the *ordering* of laminar responses is one of
    the few things laminar recordings actually measure, and a model with no
    delay at all predicts simultaneity.
    """
    return gain * series(
        pure_delay(basis, delay_s),
        alpha_synapse(basis, tau_ampa_s),
        low_pass(basis, tau_membrane_s),
    )


def lateral_surface_transfer(basis, distance_mm: float = 1.0, velocity_m_s: float = 0.25,
                             velocity_cv: float = 0.5, length_constant_mm: float = 1.5,
                             tau_ampa_s: float = 3e-3, gain: float = 1.0):
    """horizontal intracortical propagation over a geodesic distance.

    two things scale with distance and both are supplied by the topology's edge
    geometry rather than fitted: an exponentially decaying weight with a
    millimetre-scale length constant, and a delay of distance over velocity.
    the velocity here is the slow one -- horizontal layer 2/3 axons are largely
    unmyelinated and run at a few tenths of a metre per second -- so a five
    millimetre hop already costs tens of milliseconds, which is why lateral
    cortical coupling is a theta- and alpha-band phenomenon and long-range
    gamma synchrony has to be explained some other way.

    dispersion is included because the fibre calibres are heterogeneous, and a
    dispersed delay is a low-pass filter on top of the phase ramp
    (`base.delay_dispersion`).  it matters more here than for myelinated tracts:
    the coefficient of variation of unmyelinated conduction speed is large.

    where it breaks: an exponential in geodesic distance is isotropic, and
    horizontal connectivity is not -- patchy, clustered, and in some areas
    aligned to a functional map.  this form averages over that structure and
    will systematically overestimate coupling between unlike columns and
    underestimate it between like ones.
    """
    mean_s = (distance_mm * 1e-3) / max(velocity_m_s, 1e-6)
    weight = gain * float(np.exp(-distance_mm / max(length_constant_mm, 1e-6)))
    return weight * series(
        delay_dispersion(basis, mean_s, velocity_cv * mean_s),
        alpha_synapse(basis, tau_ampa_s),
    )


def tract_transfer(basis, tract_length_mm: float = 80.0, velocity_m_s: float = 3.0,
                   velocity_cv: float = 0.35, tau_ampa_s: float = 3e-3,
                   tau_nmda_decay_s: float = 0.1, nmda_fraction: float = 0.25,
                   gain: float = 1.0):
    """a whole white-matter fascicle as one complex number per frequency.

    this function is the argument for the spectral form, so it is worth
    spelling out.  a tract with an 80 mm length and a 3 m/s mean velocity has a
    ~27 ms delay.  a conventional simulator represents that with a ring buffer
    per edge and a timestep bounded below the shortest delay in the graph; a
    tractogram with 10^4 edges therefore costs 10^4 buffers and a global
    timestep set by its fastest fibre.  here the same delay is
    ``exp(-i omega tau)`` -- one multiply per retained frequency, exact at any
    window length, with no buffer and no coupling between one edge's delay and
    another edge's timestep.  heterogeneity is free: every edge simply gets its
    own tau.  that is the single largest practical payoff of carrying neural
    state as a spectrum, and it is what makes a whole-brain delayed network
    cheaper than the mean-field approximation to it.

    dispersion is not decoration.  axon diameters within a fascicle span two
    orders of magnitude, so the tract's kernel is the delay distribution's
    characteristic function: a phase ramp at the mean delay times a real
    gaussian low-pass from the spread.  a 27 ms delay with a 35% coefficient of
    variation is ~9 ms of jitter, which is already 3 dB down near 25 Hz -- long
    range coupling is low-pass for a purely geometric reason, before any
    synapse is involved.

    two failures.  the window is cyclic, so a delay approaching
    ``basis.duration_s`` wraps rather than truncates; a materialization
    carrying long tracts must either lengthen the window or overlap them.  and
    the delay distribution is really lognormal, not gaussian, so the tail is
    wrong even though the corner frequency is right.
    """
    mean_s = (tract_length_mm * 1e-3) / max(velocity_m_s, 1e-6)
    return gain * series(
        delay_dispersion(basis, mean_s, velocity_cv * mean_s),
        synaptic_drive_transfer(basis, tau_ampa_s=tau_ampa_s,
                                tau_nmda_decay_s=tau_nmda_decay_s,
                                nmda_fraction=nmda_fraction),
    )


def thalamocortical_loop_transfer(basis, tc_delay_s: float = 5e-3, ct_delay_s: float = 8e-3,
                                  tau_relay_s: float = 0.012, tau_trn_gaba_b_s: float = 0.15,
                                  loop_gain: float = 2.0, tau_ampa_s: float = 3e-3,
                                  gain: float = 1.0):
    """the corticothalamic loop, closed around its own delay.

    forward: thalamic relay through its ascending delay into a cortical
    synapse.  return: cortical layer 6 back down through a longer descending
    delay, through the reticular nucleus's slow inhibition, onto the relay
    cell.  the total round trip is 15-25 ms, which puts the half-cycle phase
    crossing squarely in the alpha and spindle range -- ~10 Hz awake, ~12-14 Hz
    in spindles when trn inhibition is stronger and the relay cells are
    hyperpolarized.

    the model therefore predicts the frequency from two delays and one
    inhibitory time constant rather than fitting it, which is the whole reason
    to declare this process separately from `tract_propagation` even though
    both live on the tractometric topology.

    where it breaks, and it breaks hard: relay cells have a t-type calcium
    current that makes them burst when hyperpolarized, and burst mode is not a
    perturbation of tonic mode -- it is a different dynamical regime with a
    different input-output function.  spindles, delta oscillation and absence
    seizures are all burst-mode phenomena, so this linear form gets their
    frequency approximately right for the wrong reason and their waveform and
    amplitude entirely wrong.  the `burst_relay_rate` implementation exists for
    exactly that regime.
    """
    forward = series(pure_delay(basis, tc_delay_s), alpha_synapse(basis, tau_ampa_s),
                     low_pass(basis, tau_relay_s))
    loop = loop_gain * series(pure_delay(basis, ct_delay_s),
                              low_pass(basis, tau_trn_gaba_b_s))
    return gain * feedback(forward, loop, negative=True)


def alpha_resonance_transfer(basis, f0_hz: float = 10.0, q: float = 4.0, gain: float = 1.0):
    """the thalamocortical loop reduced to its resonant peak and nothing else.

    a second-order band-pass fitted at the alpha frequency, offered as a
    deliberately cheap alternative to the delay-and-feedback form.  it has two
    parameters instead of six and no delays, so it cannot say *why* the peak is
    at 10 Hz -- but for a materialization whose target is a scalp spectrum
    rather than a mechanism, the extra five parameters buy nothing that any
    available evidence could move.
    """
    return resonator(basis, f0_hz, q, gain)


# ---------------------------------------------------------------------------
# nonlinear rate laws
# ---------------------------------------------------------------------------


def _sigmoid(v: np.ndarray, v_half: float, slope_mv: float, r_max: float) -> np.ndarray:
    """population activation: the fraction of a heterogeneous population above
    threshold, which is a cumulative distribution and therefore sigmoidal.  the
    slope is the spread of thresholds across cells plus the noise each of them
    sees, not a property of any single neuron."""
    return r_max / (1.0 + np.exp(-(v - v_half) / max(slope_mv, 1e-6)))


def wilson_cowan_excitatory(x, theta) -> dict:
    """excitatory population with adaptation, in the time domain.

    three terms and no more: conductance-weighted drive relaxing the membrane,
    a sigmoid from membrane to rate, and an adaptation current that integrates
    the rate and subtracts from the drive.  the adaptation term is what makes
    this worth carrying alongside the LTI form -- it is a slow negative
    feedback whose gain depends on recent output, so it produces spike-
    frequency adaptation, up/down alternation and the slow oscillation, none of
    which a fixed transfer function has any mechanism for.
    """
    v = np.asarray(x["neural.exc.potential"], dtype=float)
    a = np.asarray(x["neural.exc.adaptation"], dtype=float)
    g_e = np.asarray(x["neural.exc.ampa"], dtype=float) + np.asarray(
        x["neural.exc.nmda"], dtype=float)
    tau_m = float(theta.get("tau_membrane_s", 0.015))
    e_rest = float(theta.get("v_rest_mv", -65.0))
    drive = float(theta.get("drive_gain", 1.0))
    r = _sigmoid(v, float(theta.get("v_half_mv", -55.0)),
                 float(theta.get("slope_mv", 4.0)), float(theta.get("r_max_hz", 100.0)))
    tau_a = float(theta.get("tau_adaptation_s", 0.5))
    a_gain = float(theta.get("adaptation_gain", 0.05))
    return {
        "neural.exc.potential": (-(v - e_rest) + drive * g_e - a) / tau_m,
        "neural.exc.activity": (r - np.asarray(x["neural.exc.activity"], dtype=float)) / 5e-3,
        "neural.exc.adaptation": (a_gain * r - a) / tau_a,
    }


def shunting_inhibition_rate(x, theta) -> dict:
    """inhibition as a divisive gain change rather than a subtraction.

    gaba-a's reversal potential sits within a few millivolts of rest, so
    opening it mostly increases the membrane conductance instead of pushing the
    potential anywhere -- the effect is to shorten tau and divide the gain.
    that is a genuinely different operation from subtracting a current, and it
    is the reason a linear ``-g_i`` term gets inhibition qualitatively wrong at
    high inhibitory rates: subtraction can drive a rate negative, division
    cannot.
    """
    v = np.asarray(x["neural.exc.potential"], dtype=float)
    g_i = np.asarray(x["neural.inh.gaba_a"], dtype=float)
    e_rev = float(theta.get("e_gaba_a_mv", -70.0))
    g_leak = float(theta.get("g_leak", 1.0))
    tau_m = float(theta.get("tau_membrane_s", 0.015))
    tau_eff = tau_m * g_leak / (g_leak + np.maximum(g_i, 0.0))
    return {"neural.exc.potential": -(v - e_rev) * np.maximum(g_i, 0.0) / np.maximum(tau_eff, 1e-6)}


# ---------------------------------------------------------------------------
# local excitation
# ---------------------------------------------------------------------------

LOCAL_EXCITATION = process(
    id="local_excitation",
    doc="""recurrent glutamatergic drive within one materialized patch of tissue:
    excitatory cells onto themselves and onto the interneurons that will inhibit
    them a few milliseconds later.

    the selectors are the content of this declaration.  it reads excitatory
    firing rate, the two receptor conductances it produced last, the membrane
    potential those conductances act on, and the adaptation current that has
    accumulated -- and it writes the conductances back, the potential, the rate,
    the adaptation, and the depolarization it delivers to pv and sst cells.
    that closed set is what makes the process identifiable at all: an
    ``excitation -> activity`` coupling with no conductance state in it has one
    timescale, and every measurement that distinguishes cortical regimes from
    each other distinguishes them by having several.

    pv and sst appear as outputs and vip does not, and the asymmetry is
    deliberate.  local excitatory axons contact perisomatic pv cells and
    dendrite-targeting sst cells densely; vip cells are driven by long-range
    and neuromodulatory input instead, so `tract_propagation` writes them.  a
    process that drove all three interneuron classes locally would delete the
    disinhibitory motif, which is the one thing the three-class decomposition
    was introduced to represent.

    it also writes `neural.transmembrane_current`, because synaptic current is
    the dominant contribution to the extracellular field and the current has to
    be sourced somewhere.  putting it here rather than in a separate
    "current generation" process keeps the morphological weighting -- apical
    versus perisomatic input produce opposite dipoles at identical rates --
    next to the synapse that determines it.

    where this breaks: it is a mean-field over the cells within a materialized
    position, so it assumes their inputs are shared and their spiking is
    asynchronous.  both fail in the same regime, synchrony, and they fail in
    opposite directions -- shared input makes correlations larger than the
    mean-field allows, while the mean-field's own instability makes them
    smaller.  at fine spacing the assumption fails for the boring reason that
    there are not enough cells in the cell to average over.""",
    inputs=(
        within("neural", "exc.activity", band=DRIVE),
        within("neural", "exc.potential", band=SYNAPTIC),
        within("neural", "exc.ampa", band=SYNAPTIC),
        within("neural", "exc.nmda", band=SLOW),
        within("neural", "exc.adaptation", band=SLOW),
        within("structural", "synaptic_density", "dendritic_density", band=STRUCTURAL),
    ),
    outputs=(
        within("neural", "exc.ampa", band=SYNAPTIC),
        within("neural", "exc.nmda", band=SLOW),
        within("neural", "exc.potential", "exc.activity", band=SYNAPTIC),
        within("neural", "exc.adaptation", band=SLOW),
        within("neural", "inh.potential", band=SYNAPTIC),
        within("neural", "pv.activity", "sst.activity", band=DRIVE),
        within("neural", "transmembrane_current", band=SYNAPTIC),
    ),
    topology="local",
    timescale_s=5e-3,
    validity=Validity(
        min_spacing_mm=0.05, max_spacing_mm=5.0, band=Band(0.0, 300.0),
        note="a population mean field.  below ~50 um a materialized position holds a "
             "handful of cells and the law of large numbers this form rests on is simply "
             "false; above ~5 mm the patch spans several columns with different preferred "
             "stimuli and averaging them makes the recurrent gain look smaller than it is."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"neural", "local", "excitatory"}),
)

implementation(
    name="conductance_lti",
    process="local_excitation",
    doc="""ampa and nmda as parallel kernels, linearized about a background rate.

    the defensible default.  its parameters are receptor time constants, which
    are among the best-measured numbers in the whole inventory, and it is exact
    at any timestep because it never leaves the frequency domain.

    the linearization is the cost, and it is specific rather than vague: the
    gain from rate to conductance is treated as constant, when the real map is
    a sigmoid whose slope falls to zero at both ends.  so this form is right for
    ordinary evoked and ongoing activity around a background rate of a few
    hertz, and wrong wherever the population is near silence or near
    saturation.  it also freezes the nmda magnesium block, which means it
    cannot produce the regenerative depolarization that up-states and
    persistent activity consist of.""",
    form=Form.LTI,
    transfer=synaptic_drive_transfer,
    params={
        "tau_ampa_s": lognormal(3e-3, 1.5, units="s", provenance=Provenance.LITERATURE,
                                source="ampa-receptor epsc decay in cortical pyramidal cells",
                                note="2-5 ms at 35 C; roughly twice that at room temperature, "
                                     "which is how slice and in-vivo values disagree"),
        "tau_nmda_rise_s": lognormal(2e-3, 1.6, units="s", provenance=Provenance.LITERATURE,
                                     note="fast relative to decay, and kept separate because a "
                                          "single pole puts the phase in the wrong place"),
        "tau_nmda_decay_s": lognormal(0.1, 1.5, units="s", provenance=Provenance.LITERATURE,
                                      source="nmda-receptor epsc decay, ~100 ms",
                                      note="nr2b-rich synapses are slower still; the subunit "
                                           "composition shifts through development"),
        "nmda_fraction": uniform(0.05, 0.6, units="dimensionless", provenance=Provenance.WEAK,
                                 note="the ratio is measured in slice but is voltage- and "
                                      "pathway-dependent in vivo, so the prior stays wide"),
        "gain": weak(1.0, 5.0, units="nS per Hz",
                     note="absorbs synapse count, release probability and every unit convention "
                          "between rate and conductance; not separately identifiable from the "
                          "structural synaptic-density term without calibrated data"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="destexhe et al 1998 kinetic models of synaptic transmission",
)

implementation(
    name="ei_loop_lti",
    process="local_excitation",
    doc="""the excitatory-inhibitory loop closed, so that the local gamma
    resonance falls out of time constants rather than being fitted.

    worth carrying separately from `conductance_lti` because it makes a
    prediction the open-loop form cannot: the peak frequency moves when
    inhibitory kinetics move.  that is testable -- benzodiazepines lengthen the
    gaba-a decay and lower the gamma peak, and this form gets the direction and
    roughly the magnitude of that shift from the same tau the pharmacology
    acts on.

    it is also the form whose failure is easiest to detect.  `loop_gain` above
    the value where `base.stability_margin` approaches zero is not a strong
    oscillation, it is a linearization that has stopped applying, and the
    resulting spectrum will look plausible while being meaningless.""",
    form=Form.LTI,
    transfer=ei_loop_transfer,
    params={
        "tau_membrane_s": lognormal(0.015, 1.4, units="s", provenance=Provenance.LITERATURE,
                                    source="cortical pyramidal membrane time constant, 10-20 ms",
                                    note="at rest.  in the high-conductance state of active "
                                         "cortex the effective value is several times shorter, "
                                         "and it is the effective one this parameter means"),
        "tau_ampa_s": lognormal(3e-3, 1.5, units="s", provenance=Provenance.LITERATURE),
        "tau_inh_membrane_s": lognormal(8e-3, 1.4, units="s", provenance=Provenance.LITERATURE,
                                        source="fast-spiking interneuron membrane time constant",
                                        note="shorter than pyramidal, which is part of why the "
                                             "loop resonates where it does"),
        "tau_gaba_a_s": lognormal(6e-3, 1.4, units="s", provenance=Provenance.LITERATURE,
                                  source="gaba-a ipsc decay, 5-10 ms",
                                  note="benzodiazepines and most volatile anaesthetics prolong "
                                       "this, which is the cleanest causal test of the form"),
        "synaptic_lag_s": lognormal(1.5e-3, 1.5, units="s", provenance=Provenance.LITERATURE,
                                    note="release plus local axonal conduction, per synapse"),
        "loop_gain": lognormal(8.0, 2.0, units="dimensionless", provenance=Provenance.WEAK,
                               note="the least identifiable and most consequential parameter "
                                    "here; it sets both the sharpness of the gamma peak and "
                                    "the distance to instability"),
        "gain": weak(1.0, 5.0),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="brunel & wang 2003; wilson & cowan 1972",
)

implementation(
    name="wilson_cowan_adaptive",
    process="local_excitation",
    doc="""the nonlinear form: sigmoidal rate transfer plus an adaptation current.

    this exists for the regimes the linearizations structurally cannot enter.
    a sigmoid saturates, so drive cannot produce unbounded rate; adaptation is a
    slow negative feedback proportional to recent output, so a population that
    has been firing hard becomes harder to drive.  together they produce
    up/down alternation, the slow oscillation, and the response decrement to a
    sustained stimulus -- three phenomena that any linear form must attribute to
    changing input because it has nowhere else to put them.

    the price is real and is paid in the uncertainty machinery, not in flops: a
    pointwise nonlinearity is dense in frequency, so this f has to round-trip
    through the time domain and its output is not gaussian.  ARCHITECTURE.md's
    projection back onto the spectral form is where that non-gaussianity gets
    discarded, and the ensemble width needed to estimate the projection is the
    actual cost of selecting this implementation.""",
    form=Form.RATE,
    fn=wilson_cowan_excitatory,
    params={
        "tau_membrane_s": lognormal(0.015, 1.4, units="s", provenance=Provenance.LITERATURE,
                                    source="cortical pyramidal membrane time constant, 10-20 ms"),
        "v_rest_mv": normal(-65.0, 4.0, units="mV", provenance=Provenance.LITERATURE),
        "v_half_mv": normal(-55.0, 3.0, units="mV", provenance=Provenance.LITERATURE,
                            note="the population's half-activation, not any cell's threshold"),
        "slope_mv": lognormal(4.0, 1.6, units="mV", provenance=Provenance.WEAK,
                              note="threshold heterogeneity plus membrane noise; it is the "
                                   "noise that makes the population curve smooth, so this "
                                   "parameter is a statement about the input statistics"),
        "r_max_hz": lognormal(100.0, 1.8, units="Hz", provenance=Provenance.LITERATURE,
                              note="refractory-limited ceiling for principal cells"),
        "tau_adaptation_s": lognormal(0.5, 2.0, units="s", provenance=Provenance.LITERATURE,
                                      source="calcium- and sodium-activated potassium currents, "
                                             "0.1-2 s"),
        "adaptation_gain": weak(0.05, 6.0, units="mV per Hz"),
        "drive_gain": weak(1.0, 5.0, units="mV per nS"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="wilson & cowan 1972; benda & herz 2003",
)

implementation(
    name="learned_local_transfer",
    process="local_excitation",
    doc="""a learned map from the retained local spectrum to the conductances and
    rate it produces.

    the reason to carry it is not that the analytic forms are bad, it is that
    they are all functions of the *mean* of their inputs and the real local
    circuit is not.  the same mean rate delivered synchronously and
    asynchronously produces different conductance transients, different nmda
    recruitment and different adaptation, and a learned f reading the spectrum
    before any collapse can represent that dependence on shape.

    parameterized as a residual on the conductance prediction so that with no
    data it reproduces `conductance_lti` exactly, which is what keeps the prior
    physical.  a learned f is a different p(theta) over a differently shaped
    theta and touches nothing in the declaration above.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0, note="weight prior on the residual network"),
        "residual_gain": weak(0.3, 5.0,
                              note="zero recovers the analytic form; the posterior on this is "
                                   "the honest report of how much the data needed beyond it"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


# ---------------------------------------------------------------------------
# local inhibition
# ---------------------------------------------------------------------------

LOCAL_INHIBITION = process(
    id="local_inhibition",
    doc="""gabaergic drive within a patch: the lumped inhibitory population and
    the three interneuron classes acting on excitatory cells, on each other, and
    on themselves.

    two facts justify this being a separate process rather than a negative
    branch of `local_excitation`.  first, its kinetics are different in kind:
    gaba-a is a channel with a 5-10 ms decay and gaba-b is a metabotropic
    cascade an order of magnitude slower, and no reweighting of the excitatory
    kernels reproduces that pair.  second, its arithmetic is different:
    inhibition near its reversal potential divides rather than subtracts, and
    the two operations differ in exactly the regime -- high inhibitory rate --
    where inhibition matters most.

    the disinhibitory motif is the reason vip appears as an input here and
    nowhere as a local output.  vip cells inhibit sst cells, so this process
    reads `neural.vip.activity` and applies *negative* pressure to
    `neural.sst.activity`; the net effect on pyramidal dendrites is
    excitatory, delivered entirely through two inhibitory synapses.  a lumped
    inhibitory population cannot express a sign inversion, which is the whole
    argument for carrying the classes.

    where this breaks: pv and sst inhibition are distinguished here by their
    time constants and their targets, but their real distinction is
    compartmental -- perisomatic versus distal dendritic -- and this model has
    no dendrite.  so the model can represent that sst inhibition is slower and
    differently recruited, and cannot represent that it gates the integration
    of a specific input pathway while leaving others untouched.""",
    inputs=(
        within("neural", "inh.activity", band=DRIVE),
        within("neural", "pv.activity", "sst.activity", "vip.activity", band=DRIVE),
        within("neural", "inh.potential", band=SYNAPTIC),
        within("neural", "exc.potential", "exc.activity", band=SYNAPTIC),
        within("neural", "inh.gaba_a", band=SYNAPTIC),
        within("neural", "inh.gaba_b", band=SLOW),
        within("structural", "synaptic_density", band=STRUCTURAL),
    ),
    outputs=(
        within("neural", "inh.gaba_a", band=SYNAPTIC),
        within("neural", "inh.gaba_b", band=SLOW),
        within("neural", "inh.potential", "inh.activity", band=SYNAPTIC),
        within("neural", "exc.potential", band=SYNAPTIC),
        within("neural", "sst.activity", band=DRIVE),
        within("neural", "transmembrane_current", band=SYNAPTIC),
    ),
    topology="microcircuit",
    timescale_s=5e-3,
    validity=Validity(
        min_spacing_mm=0.05, max_spacing_mm=3.0, band=Band(0.0, 300.0),
        note="declared on the microcircuit topology because interneuron class targeting is "
             "a statement about cell types within a column, not about distance.  above ~3 mm "
             "the column structure the topology encodes has been averaged away and the class "
             "decomposition stops buying anything."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"neural", "local", "inhibitory"}),
)

implementation(
    name="gaba_conductance_lti",
    process="local_inhibition",
    doc="""fast and slow gaba kernels in parallel, linearized about background.

    the same bargain as `local_excitation:conductance_lti` and the same
    limitation, with one addition specific to inhibition: treating gaba-b's
    recruitment as a fixed fraction of gaba-a's is wrong in a direction that
    matters.  gaba-b needs spillover, which needs sustained high-frequency
    inhibitory firing, so its real fraction rises steeply with inhibitory rate.
    this form will therefore understate slow inhibition during bursts, which is
    precisely when slow inhibition terminates them.""",
    form=Form.LTI,
    transfer=gaba_transfer,
    params={
        "tau_gaba_a_s": lognormal(6e-3, 1.4, units="s", provenance=Provenance.LITERATURE,
                                  source="gaba-a ipsc decay in cortex, 5-10 ms"),
        "tau_gaba_b_rise_s": lognormal(0.05, 1.6, units="s", provenance=Provenance.LITERATURE,
                                       note="g-protein cascade, not a channel opening; this is "
                                            "why gaba-b needs a two-pole kernel"),
        "tau_gaba_b_decay_s": lognormal(0.15, 1.6, units="s", provenance=Provenance.LITERATURE,
                                        source="gaba-b / girk ipsc decay, 150-300 ms"),
        "gaba_b_fraction": uniform(0.0, 0.4, units="dimensionless", provenance=Provenance.WEAK,
                                   note="rate-dependent in reality; a constant here is the "
                                        "linearization of spillover recruitment"),
        "gain": weak(1.0, 5.0, units="nS per Hz"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="destexhe et al 1998; connors & gutnick 1990",
)

implementation(
    name="shunting_rate",
    process="local_inhibition",
    doc="""divisive inhibition: gaba-a as a conductance change rather than a current.

    the form to select whenever the question is about gain control.  a
    subtractive inhibitory term shifts the input-output curve sideways; a
    conductance term shortens the membrane time constant and flattens its
    slope, and those are different computations with different signatures.
    contrast normalization in visual cortex is a divisive story and cannot be
    fitted by a subtractive model without the fitted "input" absorbing the
    normalization -- which is how a confound becomes a result.

    nonlinear, so it costs a time-domain round trip and does not preserve
    gaussianity.  it also assumes the inhibitory reversal potential is fixed,
    and it is not: chloride accumulates under heavy inhibitory load, the
    reversal drifts depolarized, and inhibition can become frankly excitatory.
    that failure is real and is the mechanism behind several seizure models;
    `ionic_exchange` carries the chloride this form treats as a constant.""",
    form=Form.RATE,
    fn=shunting_inhibition_rate,
    params={
        "e_gaba_a_mv": normal(-70.0, 5.0, units="mV", provenance=Provenance.LITERATURE,
                              note="within a few mV of rest, which is what makes the effect "
                                   "shunting rather than hyperpolarizing"),
        "g_leak": weak(1.0, 3.0, units="nS", note="resting membrane conductance the synaptic "
                                                  "conductance is compared against"),
        "tau_membrane_s": lognormal(0.015, 1.4, units="s", provenance=Provenance.LITERATURE),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="chance, abbott & reyes 2002",
)

implementation(
    name="learned_interneuron_routing",
    process="local_inhibition",
    doc="""a learned map from the four interneuron rates to their effect on the
    excitatory population.

    the analytic forms above weight pv, sst and vip by fixed gains, and the
    honest position is that those weights are not known: the connectivity
    matrix among interneuron classes is measured in mouse, in slice, in a
    handful of areas, and its translation to human cortex at mesoscale is a
    guess.  a learned routing with a weak prior says that plainly and lets data
    move it, instead of hard-coding a mouse connectome as if it were physics.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "class_coupling_scale": speculative(1.0, 10.0,
                                            note="the interneuron-class connectivity matrix is "
                                                 "the speculative part; the kinetics are not"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.SPECULATIVE,
)


# ---------------------------------------------------------------------------
# laminar propagation
# ---------------------------------------------------------------------------

LAMINAR_PROPAGATION = process(
    id="laminar_propagation",
    doc="""the canonical cortical microcircuit as a set of directed edges between
    depths: layer 4 to layer 2/3, layer 2/3 to layer 5, layer 5 to layer 6, and
    layer 6 back to layer 4.

    this is the process the architecture's "avoid ``neural -> neural``" rule was
    written to produce.  the feedforward edge is
    ``I = {L4 excitatory activity}``, ``O = {L2/3 excitatory and inhibitory
    state}``, over the laminar topology -- exactly §4's worked example -- and
    the reason it must be written that way is that the *asymmetry* between the
    edges is the entire content.  the same cortical patch, coarse-grained over
    depth, has none of it: feedforward and feedback traffic are distinguished
    by which layer they leave from and arrive at, and a model without layers
    cannot represent the distinction that most of systems neuroscience is
    stated in.

    the band split is part of the claim, not decoration.  superficial layers
    carry gamma and deep layers carry alpha and beta, and the ascending and
    descending edges are therefore declared over different bands: the L4 to
    L2/3 and L2/3 to L5 edges up into gamma, the L5 to L6 to L4 return
    restricted to alpha and beta.  that makes the frequency asymmetry a
    declared, falsifiable property of the ontology rather than something an
    implementation happens to produce.

    where it breaks: layer 1 is in the inputs and almost nothing else here
    touches it, because layer 1 is mostly apical dendrite and long-range
    afferent axon rather than cell bodies -- which means its contribution is
    compartmental, and this model has no dendrite to put it in.  the honest
    statement is that feedback arriving in layer 1 is represented as arriving
    at the layer 5 and layer 2/3 populations whose apical tufts are there, with
    no ability to say that it modulates their integration rather than driving
    them.""",
    inputs=(
        within("neural", "exc.activity", region=L4, band=GAMMA),
        within("neural", "exc.activity", region=L2_3, band=GAMMA),
        within("neural", "exc.activity", region=L5, band=Band(0.5, 30.0)),
        within("neural", "exc.activity", region=L6, band=Band(0.5, 30.0)),
        within("neural", "exc.ampa", region=L1, band=SYNAPTIC),
        within("structural", "axonal_density", "dendritic_density", band=STRUCTURAL),
    ),
    outputs=(
        within("neural", "exc.ampa", region=L2_3, band=SYNAPTIC),
        within("neural", "exc.nmda", region=L2_3, band=SLOW),
        within("neural", "pv.activity", region=L2_3, band=DRIVE),
        within("neural", "exc.ampa", region=L5, band=SYNAPTIC),
        within("neural", "exc.nmda", region=L5, band=SLOW),
        within("neural", "inh.potential", region=L5, band=SYNAPTIC),
        within("neural", "exc.ampa", region=L6, band=SYNAPTIC),
        within("neural", "exc.ampa", region=L4, band=Band(0.0, 30.0)),
    ),
    topology="laminar",
    timescale_s=2e-3,
    validity=Validity(
        min_spacing_mm=0.1, max_spacing_mm=1.0, band=Band(0.0, 200.0),
        note="cortical thickness is 2-4 mm and layers are defined by proportion of it, so a "
             "materialization coarser than ~1 mm has fewer than three samples through the "
             "ribbon and its laminar labels are mixtures.  below ~0.1 mm the labels are "
             "sharper than any in-vivo atlas that supplies them, and the soft weights this "
             "process multiplies are atlas uncertainty rather than anatomy."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"neural", "laminar", "cortical"}),
    notes="the four directed edges share one transfer function shape and differ in their "
          "per-edge gain and delay, which the laminar topology supplies; the declaration "
          "names the union of sources and the union of targets.",
)

implementation(
    name="canonical_microcircuit_lti",
    process="laminar_propagation",
    doc="""each laminar edge as a short delay into a synapse into a membrane, with
    per-edge gains carried on the topology.

    linear and cheap, which is what makes a laminar materialization affordable
    at all: multiplying the number of positions by five is only tolerable if
    the per-edge cost is a complex multiply.  the delays are sub-millisecond to
    a couple of milliseconds -- vertical axons are short -- so they contribute
    little phase below beta and are retained because laminar response *order*
    is the measurable thing.

    the gains are the weak part and are labelled as such.  laminar connection
    strengths come from rodent slice and from a small number of primate tracing
    studies, and their transfer to a human mesoscale model is an extrapolation
    across species and across four orders of magnitude of spatial scale.""",
    form=Form.LTI,
    transfer=laminar_edge_transfer,
    params={
        "delay_s": lognormal(1.2e-3, 1.8, units="s", provenance=Provenance.LITERATURE,
                             note="a few hundred microns of largely unmyelinated vertical "
                                  "axon, plus one synaptic delay"),
        "tau_ampa_s": lognormal(3e-3, 1.5, units="s", provenance=Provenance.LITERATURE),
        "tau_membrane_s": lognormal(0.015, 1.4, units="s", provenance=Provenance.LITERATURE),
        "gain": weak(1.0, 6.0, units="dimensionless",
                     note="per-edge laminar gain.  the strongest evidence for it is rodent "
                          "slice; treating it as literature for human cortex would be "
                          "inventing precision the science has not provided"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="douglas & martin 2004 neuronal circuits of the neocortex; "
           "bastos et al 2012 canonical microcircuits for predictive coding",
)

implementation(
    name="learned_laminar_gains",
    process="laminar_propagation",
    doc="""the same kernel shape with the edge gain matrix learned per area.

    the case for it is that laminar connectivity is genuinely not uniform
    across cortex -- the granular layer that the feedforward edge terminates in
    is thick in primary sensory areas, thin in association cortex and absent in
    agranular cortex -- so a single global gain matrix is known in advance to
    be wrong somewhere.  making the gains a function of a learned areal
    embedding lets laminar structure vary with cytoarchitecture, which is what
    the `cytoarchitecture` partitioning system exists to condition on.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "granularity_sensitivity": speculative(1.0, 10.0,
                                               note="how strongly the feedforward gain tracks "
                                                    "granular-layer thickness; a plausible "
                                                    "hypothesis, not a measurement"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


# ---------------------------------------------------------------------------
# lateral cortical propagation
# ---------------------------------------------------------------------------

LATERAL_CORTICAL_PROPAGATION = process(
    id="lateral_cortical_propagation",
    doc="""horizontal spread within the cortical sheet, over geodesic distance.

    declared on the cortical-surface topology and not on a 3d local one, and
    the difference is the point.  two points a millimetre apart across a sulcal
    bank are centimetres apart along the sheet and are connected by no
    horizontal axon at all; a euclidean neighbourhood would couple them
    strongly.  every folded-cortex artefact in source-space connectivity
    analysis is that mistake, and putting the topology in the declaration is
    how the ontology refuses to make it.

    restricted to layer 2/3, where the long horizontal axon collaterals
    actually run.  layer 5 has its own, sparser, longer-range horizontal
    system, folded in here rather than declared separately because at
    mesoscale the two are not separable by any measurement ibm-1 will see.

    the input band stops at gamma and the output band does not extend past
    beta, which is a prediction rather than a convenience: horizontal
    conduction at a few tenths of a metre per second gives a millimetre-scale
    delay of several milliseconds and a large dispersion, so lateral coupling
    is low-pass and cannot synchronize gamma across more than a millimetre or
    two.  gamma coherence over longer distances therefore has to come from
    tracts or from common drive, and this declaration says so.""",
    inputs=(
        within("neural", "exc.activity", region=L2_3, band=Band(0.5, 90.0)),
        within("neural", "exc.activity", region=L5, band=Band(0.5, 30.0)),
        within("structural", "axonal_density", band=STRUCTURAL),
    ),
    outputs=(
        within("neural", "exc.ampa", "exc.nmda", region=L2_3, band=Band(0.0, 30.0)),
        within("neural", "pv.activity", region=L2_3, band=Band(0.0, 30.0)),
        within("neural", "exc.ampa", region=L5, band=Band(0.0, 30.0)),
    ),
    topology="cortical_surface",
    timescale_s=1e-2,
    validity=Validity(
        min_spacing_mm=0.2, max_spacing_mm=5.0, band=Band(0.0, 90.0),
        note="an exponential kernel in geodesic distance with a ~1-2 mm length constant.  "
             "at 5 mm spacing the kernel is narrower than one cell and the process becomes "
             "a self-coupling with a fitted gain, which is not wrong but is no longer "
             "propagation.  below ~0.2 mm the surface mesh is finer than the atlas that "
             "defines it and the geodesics are interpolation."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"neural", "cortical", "lateral"}),
)

implementation(
    name="geodesic_exponential_lti",
    process="lateral_cortical_propagation",
    doc="""exponential decay in geodesic distance, dispersed delay, ampa kernel.

    every distance-dependent quantity is read from the topology's edge
    geometry rather than fitted, so the only free numbers are a length
    constant, a velocity, a dispersion and a gain.  that is deliberately few:
    horizontal connectivity is one of the places where a flexible model would
    happily absorb the effects of tracts, of common thalamic drive and of
    volume conduction, all of which have their own processes.""",
    form=Form.LTI,
    transfer=lateral_surface_transfer,
    params={
        "velocity_m_s": lognormal(0.25, 2.0, units="m/s", provenance=Provenance.LITERATURE,
                                  source="unmyelinated horizontal axon conduction in cortex, "
                                         "0.1-0.5 m/s",
                                  note="two orders of magnitude below myelinated tract velocity, "
                                       "which is why lateral and long-range coupling occupy "
                                       "different bands"),
        "velocity_cv": uniform(0.2, 0.8, units="dimensionless", provenance=Provenance.WEAK,
                               note="calibre spread among unmyelinated collaterals is large and "
                                    "poorly characterized in human cortex"),
        "length_constant_mm": lognormal(1.5, 1.8, units="mm", provenance=Provenance.LITERATURE,
                                        source="horizontal connection density falls off over "
                                               "1-3 mm, with patchy longer-range collaterals"),
        "tau_ampa_s": lognormal(3e-3, 1.5, units="s", provenance=Provenance.LITERATURE),
        "gain": weak(1.0, 6.0),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
)

implementation(
    name="patchy_learned_kernel",
    process="lateral_cortical_propagation",
    doc="""a learned, anisotropic horizontal kernel conditioned on local structure.

    the exponential above is isotropic and horizontal connectivity is not: it
    is patchy, it links columns of like preference, and in some areas it is
    elongated along a map axis.  averaging over that structure biases coupling
    upward between unlike columns and downward between like ones, which is a
    systematic error rather than noise, and no setting of a length constant
    fixes it.  a learned kernel over a structural embedding can carry the
    anisotropy; with no data it collapses back to the isotropic prior.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "anisotropy_scale": speculative(1.0, 10.0,
                                        note="the existence of patchy anisotropy is well "
                                             "established; its mesoscale magnitude in human "
                                             "cortex is not"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


# ---------------------------------------------------------------------------
# tract propagation
# ---------------------------------------------------------------------------

TRACT_PROPAGATION = process(
    id="tract_propagation",
    doc="""long-range cortico-cortical and cortico-subcortical transmission along
    white-matter fascicles.

    inputs are excitatory firing rate at the source and the tract's own
    structure -- length, axonal density, myelination, fibre orientation;
    outputs are postsynaptic conductance at the target, plus vip activity,
    which is where long-range feedback's disinhibitory effect enters the local
    circuit.  the structural terms are inputs rather than parameters on
    purpose: myelination changes with development, with learning and with
    disease, and `plasticity` writes it, so conduction velocity in this model
    is state rather than a constant.

    the reason this process is the clearest argument for the whole spectral
    design is arithmetic, and `tract_transfer` gives it in full.  briefly: a
    conduction delay in the temporal laplacian basis is
    ``exp(-i omega tau)``, one complex multiply per retained frequency.  a
    conventional delayed network carries a history buffer per edge and a global
    timestep bounded by its shortest delay, so a 10^4-edge tractogram costs
    10^4 buffers and pays for its fastest fibre everywhere.  here delays are
    exact at any window length, heterogeneity is free, and a whole-brain
    delayed network is cheaper than the delay-free approximation to it would be
    in the time domain.  every other advantage of the spectral form is a
    modelling advantage; this one is a factor on the bill.

    where it breaks.  the basis is cyclic, so a delay approaching the window
    duration wraps around instead of leaving the window -- a materialization
    with 100 ms tracts and a 128 ms window is producing nonsense that looks
    like a phase relationship.  and the input is a firing rate: this process
    transmits rate, not spikes, so it cannot carry the millisecond-precise
    spike timing that some long-range synchrony accounts depend on.""",
    inputs=(
        within("neural", "exc.activity", band=DRIVE),
        within("structural", "axonal_density", "myelination", "fiber_orientation",
               band=STRUCTURAL),
    ),
    outputs=(
        within("neural", "exc.ampa", band=SYNAPTIC),
        within("neural", "exc.nmda", band=SLOW),
        within("neural", "inh.potential", band=SYNAPTIC),
        within("neural", "vip.activity", band=DRIVE),
    ),
    topology="tractometric",
    timescale_s=2e-2,
    validity=Validity(
        min_spacing_mm=1.0, max_spacing_mm=20.0, band=Band(0.0, 200.0),
        note="tract endpoints come from tractography, whose spatial precision is a few "
             "millimetres at best and which is systematically blind to crossing fibres and "
             "to gyral-crown terminations.  materializing below ~1 mm spacing asserts an "
             "endpoint precision the data does not have.  the band ceiling is set by "
             "dispersion, not by the declaration: above ~200 Hz a realistic delay spread "
             "has attenuated everything anyway."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"neural", "long_range", "delayed"}),
)

implementation(
    name="dispersed_delay_lti",
    process="tract_propagation",
    doc="""phase ramp at the mean delay, gaussian low-pass from the delay spread,
    then the glutamatergic receptor kernels at the target.

    the default, and the one the cost argument above is about.  the mean delay
    is tract length over conduction velocity, both supplied per edge by the
    topology and the structural inputs; the spread is a coefficient of
    variation on that velocity.

    two honest limits.  the delay distribution is lognormal in reality and
    gaussian here, so the second moment -- which sets the low-pass corner -- is
    right and the tail is not.  and the velocity is derived from myelination
    through a crude monotone relation, which is fine for the direction of an
    effect and poor for its magnitude: g-ratio, calibre and internodal length
    all matter and diffusion mri measures none of them directly.""",
    form=Form.LTI,
    transfer=tract_transfer,
    params={
        "velocity_m_s": lognormal(3.0, 2.0, units="m/s", provenance=Provenance.LITERATURE,
                                  source="cortico-cortical conduction velocity, 1-10 m/s",
                                  note="the spread is genuinely this wide within a single "
                                       "fascicle; the median is a fascicle-level summary"),
        "velocity_cv": uniform(0.15, 0.6, units="dimensionless", provenance=Provenance.LITERATURE,
                               note="axon calibre within a fascicle spans two orders of "
                                    "magnitude; this is the resulting delay dispersion"),
        "tau_ampa_s": lognormal(3e-3, 1.5, units="s", provenance=Provenance.LITERATURE),
        "tau_nmda_decay_s": lognormal(0.1, 1.5, units="s", provenance=Provenance.LITERATURE),
        "nmda_fraction": uniform(0.05, 0.6, units="dimensionless", provenance=Provenance.WEAK,
                                 note="feedback pathways are reported to be relatively "
                                      "nmda-rich, which if true makes long-range feedback "
                                      "intrinsically slower than feedforward drive"),
        "gain": weak(1.0, 8.0, units="nS per Hz",
                     note="per-edge connection strength.  tractography streamline counts are a "
                          "weak proxy for it -- they are biased by length, curvature and "
                          "termination geometry -- so this stays weak even where a connectome "
                          "supplies a number"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.LITERATURE,
    source="ringo et al 1994; caminiti et al 2013 conduction velocity in primate callosum",
)

implementation(
    name="learned_effective_connectivity",
    process="tract_propagation",
    doc="""per-edge gains and delays learned rather than derived from structure.

    ARCHITECTURE.md §3 is explicit that learned effective connectivity is
    process parameters over an interaction topology and not a new topology, and
    this is that declaration.  the topology still says which pairs *may*
    interact -- it comes from tractography and stays anatomical -- and the
    learned f decides how strongly they *do*.

    worth carrying because structural connectivity and effective connectivity
    are known to differ: a tract's existence does not fix the sign, the gain,
    or the state-dependence of what travels along it, and the same fascicle
    carries different effective coupling in different brain states.  the
    parameterization is per-site over 10^4-ish edges, which no dataset can
    fully determine; the posterior staying near its structural prior wherever
    the data are silent is the intended behaviour.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0, note="prior centred on the structure-derived gain"),
        "delay_scale": lognormal(1.0, 1.3, units="dimensionless", provenance=Provenance.WEAK,
                                 note="a multiplicative correction on the structural delay, kept "
                                      "tight because the geometry is better known than the gain"),
        "state_dependence": speculative(0.3, 10.0,
                                        note="how much the effective weight is allowed to depend "
                                             "on state; ARCHITECTURE.md §4's w_ij = topology "
                                             "times a learned state term"),
    },
    tying=Tying.PER_SITE,
    state_dependent_weights=True,
    provenance=Provenance.WEAK,
)


# ---------------------------------------------------------------------------
# thalamocortical coupling
# ---------------------------------------------------------------------------

THALAMOCORTICAL_COUPLING = process(
    id="thalamocortical_coupling",
    doc="""the reciprocal loop between thalamic relay nuclei, the reticular
    nucleus, and specified cortical layers.

    ascending: relay cell activity in lgn, mgn, vpl/vpm, pulvinar and medial
    dorsal onto layer 4 and, more weakly, layers 5 and 6.  descending: layer 6
    corticothalamic axons back onto the same relay cells and onto the reticular
    nucleus, which inhibits them.  matrix-type projections terminating in layer
    1 are folded into the layer 5 target with the same caveat laminar
    propagation carries: this model has no dendrite, so a projection onto
    apical tufts is represented as arriving at the cell.

    it is declared separately from `tract_propagation` even though both live on
    the tractometric topology, and the justification is that the loop, not the
    edge, is the object of interest.  the round-trip delay plus the reticular
    nucleus's inhibitory time constant *predict* the alpha and spindle
    frequencies rather than fitting them, and that prediction is only available
    to a process that owns both directions at once.  splitting it into two
    one-way tract couplings would leave the resonance as an emergent property
    of two independently fitted gains, which is the same number with none of
    the constraint.

    where it breaks, and it is the biggest structural failure in this file:
    relay neurons have a t-type calcium current, so they fire tonically when
    depolarized and in bursts when hyperpolarized, and burst mode is a
    different regime rather than a large-signal version of the same one.  every
    phenomenon this loop is famous for at night -- spindles, delta, absence
    seizure -- is burst-mode, so the linear implementations get the frequency
    approximately right by accident and the waveform wrong on principle.""",
    inputs=(
        within("neural", "exc.activity", region=RELAY, band=DRIVE),
        within("neural", "inh.activity", region=TRN, band=DRIVE),
        within("neural", "exc.potential", region=RELAY, band=SYNAPTIC),
        within("neural", "exc.activity", region=L6, band=Band(0.5, 30.0)),
        within("neural", "exc.activity", region=L5, band=Band(0.5, 30.0)),
        within("structural", "myelination", band=STRUCTURAL),
    ),
    outputs=(
        within("neural", "exc.ampa", region=L4, band=SYNAPTIC),
        within("neural", "exc.nmda", region=L4, band=SLOW),
        within("neural", "pv.activity", region=L4, band=DRIVE),
        within("neural", "exc.ampa", region=L5, band=SYNAPTIC),
        within("neural", "exc.ampa", region=RELAY, band=SYNAPTIC),
        within("neural", "exc.potential", region=RELAY, band=SYNAPTIC),
        within("neural", "inh.gaba_a", region=RELAY, band=SYNAPTIC),
        within("neural", "inh.gaba_b", region=RELAY, band=SLOW),
        within("neural", "exc.ampa", region=TRN, band=SYNAPTIC),
        within("neural", "inh.activity", region=TRN, band=DRIVE),
    ),
    topology="tractometric",
    timescale_s=1e-2,
    validity=Validity(
        min_spacing_mm=0.5, max_spacing_mm=10.0, band=Band(0.0, 100.0),
        note="thalamic nuclei are millimetre-scale and the reticular nucleus is a sheet "
             "under a millimetre thick, so above ~10 mm spacing trn and relay occupy the "
             "same cell and the loop's inhibitory limb disappears into its excitatory one -- "
             "which removes the negative feedback the resonance depends on and leaves a "
             "form that cannot oscillate at all."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"neural", "thalamic", "loop", "rhythm"}),
)

implementation(
    name="corticothalamic_loop_lti",
    process="thalamocortical_coupling",
    doc="""the loop closed around two delays and the reticular nucleus's slow
    inhibition, so alpha and spindle frequency come out of the kinetics.

    the implementation that earns the process its separate declaration.  it
    makes a falsifiable prediction -- lengthening the corticothalamic delay
    lowers the resonant frequency, deepening trn inhibition raises the peak and
    shifts it toward the spindle range -- from parameters that are separately
    measurable.

    linear, so exact at any timestep, and wrong in the same place the process
    docstring says: it is a small-signal description around tonic firing.""",
    form=Form.LTI,
    transfer=thalamocortical_loop_transfer,
    params={
        "tc_delay_s": lognormal(5e-3, 1.5, units="s", provenance=Provenance.LITERATURE,
                                source="thalamocortical axon conduction, ~3-8 ms to cortex",
                                note="myelinated and fast relative to cortico-cortical fibres"),
        "ct_delay_s": lognormal(8e-3, 1.5, units="s", provenance=Provenance.LITERATURE,
                                source="layer 6 corticothalamic axons are thinner and slower "
                                       "than the ascending limb"),
        "tau_relay_s": lognormal(0.012, 1.5, units="s", provenance=Provenance.LITERATURE,
                                 note="relay cell membrane time constant"),
        "tau_trn_gaba_b_s": lognormal(0.15, 1.6, units="s", provenance=Provenance.LITERATURE,
                                      source="gaba-b mediated inhibition from trn onto relay "
                                             "cells, ~150 ms",
                                      note="this is the slow limb that puts the resonance in "
                                           "alpha rather than in gamma"),
        "tau_ampa_s": lognormal(3e-3, 1.5, units="s", provenance=Provenance.LITERATURE),
        "loop_gain": lognormal(2.0, 2.5, units="dimensionless", provenance=Provenance.WEAK,
                               note="corticothalamic gain is state-dependent -- it is a large "
                                    "part of what changes between waking and sleep -- so a "
                                    "single value is a snapshot of one state"),
        "gain": weak(1.0, 5.0),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="steriade et al 1993; lopes da silva 1991 alpha as thalamocortical feedback",
)

implementation(
    name="alpha_resonator",
    process="thalamocortical_coupling",
    doc="""the loop reduced to a fitted second-order resonance and nothing more.

    two parameters, no delays, no mechanism.  the case for keeping it is that
    for a materialization whose target is a scalp spectrum, the six-parameter
    loop form is not identifiable from the data -- an eeg alpha peak constrains
    a centre frequency and a width, and nothing else -- so the extra parameters
    would simply sit at their priors while costing ensemble width.  selecting
    this f is an honest statement that the question being asked cannot see the
    mechanism.""",
    form=Form.LTI,
    transfer=alpha_resonance_transfer,
    params={
        "f0_hz": normal(10.0, 1.2, units="Hz", provenance=Provenance.LITERATURE,
                        source="individual alpha peak frequency, 8-13 Hz",
                        note="drops with age and with drowsiness; the sd is across-subject "
                             "spread, not measurement error"),
        "q": lognormal(4.0, 1.8, units="dimensionless", provenance=Provenance.LITERATURE,
                       note="alpha peaks are narrow but not ringing; q above ~10 would be a "
                            "pathological loop"),
        "gain": weak(1.0, 5.0),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
)

implementation(
    name="burst_relay_rate",
    process="thalamocortical_coupling",
    doc="""relay cells with a t-type calcium current, in the time domain.

    the implementation for the regime the linear ones cannot enter.  the t
    current is de-inactivated by hyperpolarization, so a relay cell that has
    been inhibited by trn rebounds with a burst -- which excites trn, which
    inhibits the relay cell again.  that is a relaxation oscillator, not a
    resonator, and it is what a sleep spindle actually is.  the same mechanism
    at a longer inhibitory timescale is the delta oscillation, and the same
    mechanism with the loop gain raised is the 3 Hz spike-and-wave of absence
    seizure.

    no defensible mesoscale parameterization of a population-level t current
    exists -- the single-cell biophysics is well characterized and its
    aggregation over a population that is heterogeneously hyperpolarized is
    not -- so every parameter here is weak or speculative.  ARCHITECTURE.md §5
    permits exactly this: the process exists in the ontology with an honest
    prior rather than being omitted because its f is uncertain.""",
    form=Form.RATE,
    params={
        "t_current_gain": speculative(1.0, 20.0, units="nA",
                                      note="population-aggregated low-threshold calcium "
                                           "current; the aggregation is the speculative part"),
        "t_deinactivation_s": lognormal(0.1, 2.0, units="s", provenance=Provenance.LITERATURE,
                                        source="t-type calcium channel de-inactivation, "
                                               "~50-200 ms of hyperpolarization"),
        "v_deinactivation_mv": normal(-75.0, 5.0, units="mV", provenance=Provenance.LITERATURE,
                                      note="relay cells must be held below roughly this to "
                                           "burst, which is why burst mode is a sleep and "
                                           "not a waking phenomenon"),
        "burst_spikes": uniform(2.0, 8.0, units="spikes", provenance=Provenance.LITERATURE),
        "trn_gain": weak(1.0, 8.0, note="reticular inhibition strength; sets whether the "
                                        "oscillator runs at spindle or at delta frequency"),
    },
    tying=Tying.PER_PARTITION,
    differentiable=False,
    provenance=Provenance.SPECULATIVE,
    source="destexhe & sejnowski 2003 interactions between membrane conductances",
)


__all__ = [
    "LOCAL_EXCITATION",
    "LOCAL_INHIBITION",
    "LAMINAR_PROPAGATION",
    "LATERAL_CORTICAL_PROPAGATION",
    "TRACT_PROPAGATION",
    "THALAMOCORTICAL_COUPLING",
]
