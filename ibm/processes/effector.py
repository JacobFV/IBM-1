"""the motor side: cortical command becoming motoneuron traffic, and motoneuron
traffic becoming force and movement.

the architecture's claim about this side is the mirror of the claim about the
sensory side, and it is the same claim.  a movement is not an output of the
model.  it is *evidence about effector state* -- `effector.activation`,
`effector.force`, and the mechanical state of the body -- which are ordinary
state variables coupled to the rest of the model by ordinary processes.  a
materialization that predicts a button press from a visual stimulus is doing
precisely the same kind of thing as one that predicts blood oxygenation from
population activity: pushing pressure through a chain of declared couplings and
comparing the result to a likelihood.  there is no special "action" machinery in
ibm-1 and this file is where the absence of it is cashed out.

two processes live here.

`efferent_propagation` carries descending command out of the brain: cortical and
brainstem population activity through the corticospinal and extrapyramidal
tracts and the segmental motoneuron pool, ending in motoneuron firing and the
drive delivered to individual motor units.  it is a transport process, and like
every transport process in the inventory its content is mostly delay,
dispersion and gain.

`effector_activation` turns that drive into activation, force and motion.  it is
the one place in ibm-1 where the model touches the physical world in the
direction of causation rather than of measurement, and it is decidedly the least
constrained by the evidence base -- the neuroscience datasets in docs/EVIDENCE.md
constrain the neural side densely and the musculoskeletal side hardly at all.

the split between the two is not arbitrary.  it is placed exactly where the
representation changes: above it, the state is a firing rate on a peripheral
neural support; below it, the state is a mechanical quantity with units of force
and metres.  recruitment -- which motor units a given descending drive actually
engages -- is the hinge, and it is declared in `efferent_propagation` because it
is a property of the motoneuron pool, not of the muscle.

one representational admission.  `effector.*` is spectral and `mechanical.*` is
scalar, so `effector_activation` crosses the registered spectral -> scalar
conversion and the temporal structure of force is collapsed to a window mean
before it reaches the body's mechanics.  for a reaching movement over a second
that is defensible.  for tremor, for the 8-12 Hz physiological oscillation in
steady force, or for the 15-30 Hz piper rhythm in strong contraction it is
exactly wrong -- those *are* structure in the band the conversion discards, and
they are among the more interesting things a brain-body model could say.
"""

from __future__ import annotations

from ibm.processes.base import (
    alpha_synapse,
    delay_dispersion,
    implementation,
    leaky_integrator,
    low_pass,
    process,
    resonator,
    series,
)
from ibm.registry import Form
from ibm.vocabulary import (
    Anat,
    Band,
    OnSupport,
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

# ---------------------------------------------------------------------------
# bands and regions
# ---------------------------------------------------------------------------

#: descending command.  wide, because corticomuscular coherence in the beta band
#: is a real and measurable property of the corticospinal pathway and reading the
#: command only up to a few hertz would throw it away before the coupling ever
#: saw it.
COMMAND = Band(0.0, 200.0)

#: motoneuron and motor-unit traffic.  motor units discharge between roughly 8
#: and 40 Hz, but the *pool's* aggregate output carries content well above that
#: because units are not synchronized.
MOTOR = Band(0.0, 200.0)

#: the drive actually delivered to a motor unit.  narrower than MOTOR because
#: `effector.drive` is declared meaningful only to 100 Hz -- past the point where
#: a motor unit could act on it differently from a smoothed version of itself.
DRIVE = Band(0.0, 100.0)

#: activation and force.  a muscle is a low-pass filter with a corner near a few
#: hertz, so force above ~50 Hz is not something a muscle can produce however
#: the drive is structured -- this ceiling is mechanical, not conventional.
FORCE = Band(0.0, 50.0)

#: fatigue, and every other quantity that accumulates over a contraction.
FATIGUE = Band(0.0, 0.5)

#: tissue structure over an experiment: constant.
STRUCTURE = Band(0.0, 0.001)

MOTOR_CORTEX = Anat("cortical_areas", "m1")
BRAINSTEM = Anat("brainstem_nuclei", "reticular_formation")
MOTOR_UNITS = OnSupport("motor_units")
BODY = OnSupport("body")
HEAD = OnSupport("head_volume")
VOCAL_TRACT = OnSupport("vocal_tract")


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def corticospinal_transfer(basis, mean_delay_s: float = 0.012, sd_delay_s: float = 0.004,
                           tau_synapse_s: float = 3e-3, gain: float = 1.0):
    """descending command -> motoneuron drive: a dispersed delay into a synapse.

    the delay is dispersed rather than fixed because the corticospinal tract is a
    bundle with a wide diameter spectrum -- a few large, fast fibres and a great
    many small, slow ones -- so the conduction time to a motoneuron pool is a
    distribution, not a number.  the dispersion is not a nuisance: it is a
    low-pass whose corner sits in the beta band, and it is a large part of why
    corticomuscular coherence falls away above ~30 Hz for reasons that have
    nothing to do with synapses.

    the ~12 ms mean is the arm; the leg is nearer 20 ms and the face nearer 8,
    which is a per-target quantity and therefore a per-partition parameter.
    """
    return gain * series(delay_dispersion(basis, mean_delay_s, sd_delay_s),
                         alpha_synapse(basis, tau_synapse_s))


def twitch_transfer(basis, contraction_time_s: float = 0.05, gain: float = 1.0):
    """motor unit drive -> force, as a critically damped second-order twitch.

    a single motor unit twitch rises to peak in 30-100 ms and decays over roughly
    twice that, which is exactly an alpha function; the double pole is why muscle
    rolls off at 40 dB/decade and why a muscle cannot follow a drive that a
    motoneuron can easily produce.  a 30 Hz drive to a slow unit is fused force,
    not 30 Hz force, and that fusion is the filter rather than a separate
    mechanism.

    fast and slow units differ threefold in contraction time and are recruited in
    order of increasing speed, so the effective time constant of a *pool* changes
    with contraction level.  this form holds it fixed, which is a linearization
    about one recruitment level.
    """
    return gain * alpha_synapse(basis, contraction_time_s)


def activation_dynamics_transfer(basis, tau_rise_s: float = 0.03,
                                 tau_fall_s: float = 0.06, gain: float = 1.0):
    """neural drive -> muscle activation: calcium release and reuptake as two poles.

    excitation-contraction coupling is genuinely asymmetric -- the sarcoplasmic
    reticulum releases calcium faster than it pumps it back -- so rise and fall
    are separate constants and collapsing them onto one pole misplaces the phase
    of every fast movement.  the asymmetry is roughly two-fold and is the reason
    relaxation is the rate-limiting step in rapid alternating movement.
    """
    return gain * series(low_pass(basis, tau_rise_s), low_pass(basis, tau_fall_s))


def limb_mechanics_transfer(basis, f0_hz: float = 2.0, q: float = 0.6, gain: float = 1.0):
    """force -> limb displacement, as a damped second-order mechanical load.

    inertia, stiffness and damping in one resonator.  the resonance is low --
    around 1-3 Hz for a human arm at typical co-contraction -- and heavily damped,
    which is why voluntary movement above a few hertz is impossible however fast
    the neural command is.  it is also where physiological tremor comes from: at
    higher co-contraction the resonance stiffens and rises, and the 8-12 Hz peak
    in steady force is this loop and the stretch reflex together.

    stiffness is under neural control through co-contraction, so f0 is not
    really a constant; treating it as one is a linearization about a posture and
    a background contraction level.
    """
    return gain * resonator(basis, f0_hz, q)


def fatigue_transfer(basis, tau_s: float = 60.0, gain: float = 1.0):
    """activation -> fatigue, as a leaky accumulator.

    fatigue is the running integral of activation against its own recovery, so a
    leaky integrator with a DC gain of tau is the right shape: a maintained
    contraction accumulates fatigue linearly at first and saturates, and rest
    unwinds it on the same constant.

    the single constant is a fiction of convenience.  real fatigue has at least
    two timescales an order of magnitude apart -- metabolite accumulation over
    tens of seconds, and central and structural components over minutes to hours
    -- and recovery is slower than accumulation, which no linear form can express.
    """
    return gain * leaky_integrator(basis, tau_s)


# ---------------------------------------------------------------------------
# efferent propagation
# ---------------------------------------------------------------------------

EFFERENT_PROPAGATION = process(
    id="efferent_propagation",
    doc="""descending command from cortex and brainstem to motoneurons, and
    motoneuron output distributed across motor units.

    the inputs are motor-cortical and brainstem population activity read across a
    wide band, and the wide band is a deliberate choice rather than a default:
    the beta-band component of corticospinal drive is measurable at the muscle as
    corticomuscular coherence, so restricting the read to the movement's own
    bandwidth would discard a signal that several datasets in docs/EVIDENCE.md
    exist to constrain.

    the outputs are motoneuron firing (`neural.efferent.activity`, on the body
    support) and the drive delivered to motor units (`effector.drive`, on the
    motor-unit support).  keeping those separate is the point of putting
    recruitment in this process: one motoneuron pool's output is distributed
    across its units by a threshold ordering, so the same pool firing rate
    delivers very different drive depending on how much of the pool is already
    recruited.  a declaration that wrote pool rate straight into muscle
    activation would have asserted that recruitment is linear, which is the one
    thing about it that is definitely false.

    structural inputs are read because conduction velocity is a function of
    myelination and the tract's delay is most of this process's content.  a
    demyelinating lesion is a change to this process's *inputs*, not to its
    parameters, and that is a modelling claim worth being explicit about.

    where this breaks: there is no spinal circuitry in it.  the segmental
    interneurons, reciprocal inhibition, renshaw feedback, and the stretch reflex
    loop are all absent, so the pool here is a relay with a threshold rather than
    a circuit.  every reflexive and load-compensating behaviour is therefore
    missing, and the model will attribute those to descending command because
    descending command is the only thing it has.""",
    inputs=(
        within("neural", "exc.activity", region=MOTOR_CORTEX, band=COMMAND),
        within("neural", "exc.potential", region=MOTOR_CORTEX, band=COMMAND),
        within("neural", "exc.activity", region=BRAINSTEM, band=COMMAND),
        within("neural", "efferent.activity", region=BODY, band=MOTOR),
        within("neural", "efferent.alpha", "efferent.gamma", region=BODY, band=MOTOR),
        within("neural", "efferent.b_preganglionic", region=BODY, band=MOTOR),
        within("effector", "drive", "fatigue", region=MOTOR_UNITS, band=DRIVE),
        within("structural", "myelination", "axonal_density", band=STRUCTURE),
    ),
    outputs=(
        within("neural", "efferent.activity", region=BODY, band=MOTOR),
        # alpha and gamma are separate outputs because they are separately
        # controlled and their RATIO is a motor-control variable: alpha-gamma
        # co-activation is what keeps the spindle loaded while the muscle shortens.
        # a lumped efferent rate cannot express the ratio, so it cannot express
        # fusimotor set at all, and every spindle in the model would then fall
        # silent during voluntary movement.
        within("neural", "efferent.alpha", region=BODY, band=MOTOR),
        within("neural", "efferent.gamma", region=BODY, band=MOTOR),
        # autonomic outflow, pre- and postganglionic, separated by the ganglionic
        # synapse where divergence happens.
        within("neural", "efferent.b_preganglionic", region=BODY, band=MOTOR),
        within("neural", "efferent.c_postganglionic", region=BODY, band=MOTOR),
        within("effector", "drive", region=MOTOR_UNITS, band=DRIVE),
    ),
    topology="efferent_pathway",
    timescale_s=5e-3,
    validity=Validity(
        min_spacing_mm=0.5, max_spacing_mm=20.0, band=COMMAND,
        note="a pool-level description.  below ~0.5 mm the motoneuron pool is resolved "
             "into individual cells and a rate description of it stops being the right "
             "object; above ~20 mm the cortical source spans several body representations "
             "of the motor homunculus, whose descending delays differ by a factor of two."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"motor", "peripheral", "transport"}),
)

implementation(
    name="corticospinal_dispersed_delay",
    process="efferent_propagation",
    doc="""the tract as a dispersed delay line into a motoneuron synapse.

    linear, so exact at any timestep, and it carries the one thing about
    descending transmission that is quantitatively solid: the latency
    distribution.  the mean is measured directly by TMS -- the motor evoked
    potential latency to a hand muscle is about 20 ms from stimulus, of which
    roughly 12 is conduction -- and the dispersion is measured indirectly, by how
    fast corticomuscular coherence falls with frequency.

    the reason to prefer a dispersed delay over a fixed one is not accuracy in
    the mean; it is that a bundle with 12 ms mean and 4 ms spread is already
    several dB down at 40 Hz from dispersion alone, before any synapse.  a fixed
    delay predicts coherence flat to the synaptic corner and would attribute the
    observed roll-off to the wrong mechanism.

    where it breaks: it is a pure relay.  no threshold, no saturation, no
    recruitment, so it will happily report negative motoneuron firing for a
    negative command.""",
    form=Form.LTI,
    transfer=corticospinal_transfer,
    params={
        "mean_delay_s": lognormal(0.012, 1.4, units="s", provenance=Provenance.LITERATURE,
                                  source="TMS motor evoked potential latency minus segmental "
                                         "and peripheral conduction",
                                  note="hand ~12 ms, leg ~20 ms, face ~8 ms; this is a "
                                       "per-target quantity and the tying reflects that"),
        "sd_delay_s": lognormal(0.004, 2.0, units="s", provenance=Provenance.LITERATURE,
                                note="corticospinal fibre diameters span an order of "
                                     "magnitude; the dispersion low-pass sits in the beta band "
                                     "and is a large part of why corticomuscular coherence "
                                     "dies above ~30 Hz"),
        "tau_synapse_s": lognormal(3e-3, 2.0, units="s", provenance=Provenance.LITERATURE,
                                   note="monosynaptic corticomotoneuronal EPSP; the direct "
                                        "projection is a primate specialisation and most of "
                                        "the descending drive in other species is not this"),
        "gain": weak(1.0, 10.0, units="motoneuron Hz per cortical Hz",
                     note="absorbs the number of corticomotoneuronal cells contacting the "
                          "pool and every unit convention upstream; not identifiable without "
                          "a calibrated stimulation dataset"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="rothwell 1997; conway et al 1995; baker et al 1997",
)

implementation(
    name="henneman_recruitment",
    process="efferent_propagation",
    doc="""the size principle: motor units recruited in order of threshold, then
    rate-coded.

    the nonlinear form, and the one that earns its cost.  a motoneuron pool does
    not scale its output; it recruits.  units are engaged in a fixed order from
    smallest and slowest to largest and fastest, and only once a unit is
    recruited does its firing rate rise with drive.  this produces a force-drive
    relation that is smooth but definitely not linear, and it produces a
    *systematically changing* force resolution -- fine at low force where small
    units are being added, coarse at high force where large ones are.

    the ordering is the most robust fact in motor physiology and holds across
    muscles, species and tasks, which is why the priors here are literature
    rather than weak.  the recruitment range is not so robust: small hand muscles
    complete recruitment by ~50% of maximum force and rely on rate coding above
    that, while large limb muscles recruit until ~85%, and this parameter is what
    distinguishes them.

    where it breaks: recruitment order can be partially violated by task, and
    rate and recruitment are not independent -- the pool's common drive
    synchronizes units to a degree that this form, which treats units as
    conditionally independent given drive, cannot express.  that synchronization
    is precisely what surface EMG measures, so this implementation predicts force
    better than it predicts the signal used to observe force.""",
    form=Form.RATE,
    params={
        "recruitment_range_fraction": normal(0.6, 0.2, units="fraction of max force",
                                             provenance=Provenance.LITERATURE,
                                             note="hand muscles complete recruitment near 0.5, "
                                                  "large limb muscles near 0.85"),
        "threshold_spread_decades": normal(2.0, 0.5, units="log10 units",
                                           provenance=Provenance.LITERATURE,
                                           note="motor unit twitch forces span roughly two "
                                                "orders of magnitude within one muscle"),
        "min_firing_rate_hz": normal(8.0, 2.0, units="Hz", provenance=Provenance.LITERATURE,
                                     note="units start firing at a nonzero rate, not from "
                                          "zero; below it they drop out entirely"),
        "max_firing_rate_hz": normal(35.0, 8.0, units="Hz", provenance=Provenance.LITERATURE,
                                     note="in sustained voluntary contraction; brief ballistic "
                                          "efforts reach far higher"),
        "rate_gain_hz_per_drive": weak(20.0, 4.0, units="Hz per unit drive"),
        "common_drive_fraction": speculative(0.3, 5.0, units="dimensionless",
                                             note="how much of the pool's input is shared "
                                                  "across units.  it sets synchronization and "
                                                  "therefore the EMG amplitude at a given "
                                                  "force, and it is not well pinned down"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="henneman et al 1965; de luca & hostage 2010; fuglevand et al 1993",
)

implementation(
    name="learned_descending_map",
    process="efferent_propagation",
    doc="""a learned map from the cortical population spectrum to pool drive,
    with the dispersed delay line as its prior mean.

    the case for it is specific.  the analytic forms above treat descending
    command as a scalar intensity travelling down a pipe, and the evidence that
    it is not is strong: motor cortical population activity has a low-dimensional
    rotational structure whose relation to muscle activity is a linear readout in
    a learned basis, not a copy.  a learned f with access to the population
    spectrum can express that readout; a delay line with a gain cannot express it
    at all, and will attribute all of its structure to the cortical input.

    the residual parameterization keeps the delay physical.  what the learned
    term is allowed to change is the mixing across populations and the shape of
    the gain, not the latency, because latency is the one quantity here that is
    measured directly and there is nothing to gain from letting a fit move it.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0, note="weight prior over the learned readout"),
        "residual_gain": weak(0.3, 5.0,
                              note="the learned term is a correction on the delay line; with "
                                   "no data the process reduces to the literature form"),
        "readout_rank": uniform(2.0, 20.0, units="dimensions",
                                note="dimensionality of the cortical-to-muscle readout; the "
                                     "reported intrinsic dimensionality of motor cortical "
                                     "population activity during reaching is in this range"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)


# ---------------------------------------------------------------------------
# effector activation
# ---------------------------------------------------------------------------

EFFECTOR_ACTIVATION = process(
    id="effector_activation",
    doc="""motor-unit drive becomes muscle activation, activation becomes force,
    and force moves the body.

    this is where the model leaves neural units behind, and the change of
    representation is the reason it is a separate process from
    `efferent_propagation` rather than the tail of it.

    three outputs and they are genuinely different things.  `effector.activation`
    is the calcium-mediated internal state of the muscle and lags the drive by
    tens of milliseconds asymmetrically.  `effector.force` is what the muscle
    produces given that activation *and* its current length and shortening
    velocity, which is why force is not a function of activation alone.
    `effector.fatigue` is the slow accumulator that reduces the force a given
    activation produces, on a timescale three orders of magnitude longer than the
    twitch.  and the mechanical outputs on the body are the movement itself.

    the vocal tract is included in the output region set, without ceremony,
    because speech production is effector activation and articulator kinematics
    are mechanical state on a body support.  a produced-audio observation
    therefore attaches to the mechanical field the same way a motion-capture
    observation does, which is the intended consequence of the architecture's
    refusal to treat behaviour as special.

    where this breaks, and it breaks in more places than any other process in
    this file: there is no skeleton.  moment arms, joint limits, multi-joint
    coupling, the fact that most muscles cross more than one joint, and the
    redundancy of having more muscles than degrees of freedom are all absent.
    what is here is a lumped activation-force-motion chain per effector, which is
    adequate for isometric force and for a single-joint movement and is not a
    musculoskeletal model.  the evidence base reflects this: docs/EVIDENCE.md is
    dense on the neural side and thin here, so most of these parameters will stay
    near their priors and should.""",
    inputs=(
        within("effector", "drive", region=MOTOR_UNITS, band=DRIVE),
        # alpha drive specifically: gamma produces no meaningful force, so a plant
        # driven by the lumped efferent rate would generate force from fusimotor
        # traffic, which is exactly backwards.
        within("neural", "efferent.alpha", region=BODY, band=DRIVE),
        within("effector", "length", "velocity", region=MOTOR_UNITS, band=FORCE),
        within("effector", "activation", "force", "fatigue", region=MOTOR_UNITS,
               band=FORCE),
        within("mechanical", "displacement", "velocity", region=HEAD, band=FORCE),
        within("metabolic", "atp", "lactate", band=FATIGUE),
        within("structural", "axonal_density", band=STRUCTURE),
    ),
    outputs=(
        within("effector", "activation", "force", region=MOTOR_UNITS, band=FORCE),
        # the muscle-tendon unit's own kinematic state.  it is an OUTPUT of this
        # process because the plant is what knows it, and it closes the
        # proprioceptive loop: `transduction` reads these to drive the spindle and
        # tendon-organ receptors, which drive Ia/Ib/II, which reach cortex.  before
        # they existed the force-length and force-velocity relations this process
        # claims to implement had to assume a constant length.
        within("effector", "length", "velocity", region=MOTOR_UNITS, band=FORCE),
        within("effector", "fatigue", region=MOTOR_UNITS, band=FATIGUE),
        within("mechanical", "displacement", "velocity", "stress", region=HEAD,
               band=FORCE),
        # phonation.  the band is far above the FORCE ceiling on purpose: vocal
        # fold vibration at 100-300 Hz is not muscle-driven oscillation, it is an
        # aerodynamic-myoelastic flutter that muscle only *tunes* by setting
        # tension and adduction.  declaring it inside the muscle's own bandwidth
        # would assert that a muscle contracts at 200 Hz, which it cannot.  the
        # ceiling is the mechanical field's own declared limit of 500 Hz, not the
        # acoustics': a microphone hears to 20 kHz and this component does not
        # carry it, so speech above the first formant is outside what the current
        # ontology can represent at all.
        within("mechanical", "pressure", region=VOCAL_TRACT, band=Band(0.0, 500.0)),
    ),
    topology="efferent_pathway",
    timescale_s=0.03,
    validity=Validity(
        min_spacing_mm=1.0, max_spacing_mm=100.0, band=FORCE,
        note="a lumped muscle.  below ~1 mm the compartment is a fascicle and the "
             "activation-force relation is a fibre property rather than a muscle one; "
             "above ~100 mm several muscles with opposing actions are averaged and the "
             "sign of the produced motion is arbitrary."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"motor", "peripheral", "mechanical"}),
    notes="spectral effector inputs, scalar mechanical outputs: the registered "
          "spectral -> scalar conversion applies.  tremor, the 8-12 Hz physiological "
          "oscillation and the piper rhythm are structure in the band that conversion "
          "discards, which is a real and specific loss rather than a formality.",
)

implementation(
    name="hill_muscle",
    process="effector_activation",
    doc="""the hill-type muscle: activation dynamics, then force-length and
    force-velocity scaling.

    the standard model of muscle for fifty years, and standard for a good reason
    -- the two nonlinearities it carries are both large and both unavoidable.
    force falls off either side of optimal length by tens of percent over a
    physiological range, and force falls with shortening velocity so steeply that
    a muscle shortening at a third of its maximum velocity produces well under
    half its isometric force.  a model without those does not merely
    mis-scale movement; it gets the direction of the errors wrong, because it
    predicts that fast movements are as forceful as slow ones.

    the parameters split cleanly by provenance.  the shape constants -- the Hill
    a/F0 ratio, maximum shortening velocity in fibre lengths per second, the
    width of the force-length curve -- are measured, transfer between muscles and
    species, and carry literature priors.  the scale constants -- maximum
    isometric force, optimal fibre length, tendon slack length -- are
    subject- and muscle-specific, vary by an order of magnitude, and are almost
    never measured in a neuroscience experiment, so they carry weak priors and
    will not move.

    where it breaks: it is a lumped, rate-independent, history-free description.
    residual force enhancement after stretch, force depression after shortening,
    and short-range stiffness are all real, all outside it, and all matter for
    exactly the perturbation experiments a brain-body model would want to run.""",
    form=Form.RATE,
    params={
        "tau_activation_rise_s": lognormal(0.03, 1.6, units="s",
                                           provenance=Provenance.LITERATURE,
                                           note="calcium release; 10-30 ms fast fibres, longer "
                                                "in slow"),
        "tau_activation_fall_s": lognormal(0.06, 1.6, units="s",
                                           provenance=Provenance.LITERATURE,
                                           note="reuptake is roughly twice as slow as release, "
                                                "and that asymmetry rate-limits fast "
                                                "alternating movement"),
        "hill_a_over_f0": normal(0.25, 0.08, units="dimensionless",
                                 provenance=Provenance.LITERATURE,
                                 source="hill 1938 force-velocity constant",
                                 note="lower in slow muscle; the curvature of the "
                                      "force-velocity relation"),
        "vmax_lengths_per_s": normal(10.0, 3.0, units="L0/s",
                                     provenance=Provenance.LITERATURE,
                                     note="maximum shortening velocity, ~5 L0/s slow and "
                                          "~15 fast; the recruitment order means the pool's "
                                          "effective vmax rises with contraction level"),
        "force_length_width": normal(0.45, 0.1, units="fraction of L0",
                                     provenance=Provenance.LITERATURE,
                                     note="half-width of the active force-length curve"),
        "max_isometric_force_n": lognormal(500.0, 8.0, units="N",
                                           provenance=Provenance.WEAK,
                                           note="spans two orders of magnitude between an "
                                                "intrinsic hand muscle and quadriceps, and is "
                                                "essentially never measured in the datasets "
                                                "this model will see"),
        "optimal_fibre_length_m": lognormal(0.08, 3.0, units="m",
                                            provenance=Provenance.WEAK),
        "tendon_compliance": weak(0.03, 3.0, units="strain at max force",
                                  note="tendon strain at maximum force is ~3-5%; it decouples "
                                       "fibre length from joint angle and is why a muscle can "
                                       "be nearly isometric during a movement"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="hill 1938; zajac 1989; millard et al 2013",
)

implementation(
    name="twitch_activation_lti",
    process="effector_activation",
    doc="""the linear chain: drive through activation dynamics, through a twitch,
    into a damped mechanical load.

    the cheap form and honest about being cheap.  it is the linearization of the
    Hill model about one operating point -- one length, one velocity, one
    recruitment level -- and within a few percent of that point it agrees with it.
    it is exact at any timestep and carries no state, which matters because a
    materialization that only needs a plausible motor output alongside a
    detailed cortical model should not be paying for a nonlinear muscle solve.

    it is also the form that says something the nonlinear one obscures: the
    entire drive-to-force path is a cascade of low-passes with a corner around
    2-5 Hz, so voluntary force simply cannot follow a command faster than that.
    the bandwidth of movement is set by the muscle, not by the brain, and this
    filter is the statement of that fact.

    where it is precisely wrong: it has no length or velocity dependence, so it
    predicts the same force whatever the limb is doing, and no saturation, so it
    will report forces the muscle cannot produce.""",
    form=Form.LTI,
    transfer=lambda basis, tau_rise_s=0.03, tau_fall_s=0.06, contraction_time_s=0.05: series(
        activation_dynamics_transfer(basis, tau_rise_s, tau_fall_s),
        twitch_transfer(basis, contraction_time_s)),
    params={
        "tau_rise_s": lognormal(0.03, 1.6, units="s", provenance=Provenance.LITERATURE),
        "tau_fall_s": lognormal(0.06, 1.6, units="s", provenance=Provenance.LITERATURE),
        "contraction_time_s": lognormal(0.05, 1.8, units="s",
                                        provenance=Provenance.LITERATURE,
                                        note="single twitch time to peak: ~30 ms fast units, "
                                             "~100 ms slow.  the pool's effective value shifts "
                                             "with recruitment and is not a constant"),
        "gain": weak(1.0, 10.0, units="N per unit drive"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
)

implementation(
    name="limb_load_lti",
    process="effector_activation",
    doc="""force into a damped second-order mechanical load: the body as a filter.

    declared separately from the twitch so that a materialization can select a
    muscle model and a load model independently, and so that limb inertia and
    stiffness -- which are posture-dependent and under neural control through
    co-contraction -- have a place to live that is not inside the muscle's
    parameters.

    the resonance is low and heavily damped, around 1-3 Hz for a human arm, and
    that number does most of the work: it is why reaching movements have the
    duration they do, and why the 8-12 Hz physiological tremor is a property of
    this loop closed through the stretch reflex rather than a property of the
    motor command.  the tremor itself is not produced here, because the reflex
    loop is not declared anywhere in the inventory -- which is the gap that
    matters most on this side of the model.""",
    form=Form.LTI,
    transfer=limb_mechanics_transfer,
    params={
        "f0_hz": lognormal(2.0, 2.0, units="Hz", provenance=Provenance.LITERATURE,
                           note="rises with co-contraction, so it is under neural control and "
                                "not really a constant; the fixed value is a linearization "
                                "about one background stiffness"),
        "q": normal(0.6, 0.2, units="dimensionless", provenance=Provenance.LITERATURE,
                    note="below 0.5 is overdamped; a relaxed limb is close to critically "
                         "damped, which is why voluntary movement does not ring"),
        "gain": weak(1.0, 10.0, units="m per N"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
)

implementation(
    name="fatigue_accumulator",
    process="effector_activation",
    doc="""fatigue as a leaky integral of activation.

    separated from the force implementations so that a long-recording
    materialization can carry fatigue cheaply without committing to a muscle
    model, and so that the accumulation constant appears in exactly one place.

    what it gets right is the shape: fatigue accumulates roughly linearly at
    first, saturates, and unwinds when the contraction stops.  what it gets wrong
    is that there is only one constant.  metabolite accumulation acts over tens of
    seconds, central fatigue over minutes, and low-frequency fatigue over hours,
    and recovery is slower than accumulation in every one of them -- an asymmetry
    a linear filter cannot have.  the parameter below is therefore a compromise
    between processes that differ by two orders of magnitude, and the wide prior
    is that compromise made visible.""",
    form=Form.LTI,
    transfer=fatigue_transfer,
    params={
        "tau_s": lognormal(60.0, 5.0, units="s", provenance=Provenance.LITERATURE,
                           note="a compromise across timescales that differ by two orders of "
                                "magnitude; the spread is the disagreement, not measurement "
                                "error"),
        "gain": weak(1.0, 5.0, units="fatigue per unit activation-second"),
        "recovery_asymmetry": speculative(2.0, 4.0, units="dimensionless",
                                          note="recovery is slower than accumulation.  declared "
                                               "so a nonlinear f has somewhere to put it; this "
                                               "linear form cannot use it"),
    },
    tying=Tying.GLOBAL,
    provenance=Provenance.WEAK,
)

implementation(
    name="learned_musculoskeletal",
    process="effector_activation",
    doc="""a learned drive-to-kinematics map with the Hill chain as its prior mean.

    the honest argument for this one is different from the argument elsewhere in
    the inventory.  it is not that a learned f can express a subtlety the
    analytic form misses; it is that the analytic form is missing an entire
    skeleton, and the data that exists -- motion capture and EMG recorded
    together, of which docs/EVIDENCE.md lists a good deal -- constrains the
    drive-to-kinematics map directly while constraining almost none of the
    parameters of a biomechanical model.  fitting what the data can see is a
    better use of it than fitting a mechanism it cannot.

    the cost is that the fitted map is subject- and task-specific in ways a
    biomechanical model would not be, and it will not extrapolate to a posture or
    a load outside its training set.  the residual parameterization limits the
    damage: outside the fitted regime the learned term shrinks towards zero and
    the Hill chain is what remains.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0),
        "residual_gain": weak(0.4, 5.0,
                              note="larger than elsewhere in the inventory because the "
                                   "analytic prior is weaker here than anywhere else"),
        "state_rank": uniform(4.0, 64.0, units="dimensions",
                              note="latent body-state dimension; see ibm.processes.nn for the "
                                   "parameter count as a function of tying"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.WEAK,
)

__all__ = ["EFFERENT_PROPAGATION", "EFFECTOR_ACTIVATION"]
