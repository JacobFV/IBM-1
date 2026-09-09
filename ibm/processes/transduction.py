"""transduction: the physical world becoming population state, at the surfaces
where that actually happens.

the architecture is blunt about this and the bluntness is the point.  a
stimulus is not an input to the model.  it is an intervention on transduction
state -- ordinary state, with a value, an uncertainty and an interaction
topology, sitting on a support that happens not to be brain parenchyma.  a
photoreceptor potential is a state variable in exactly the sense a thalamic
membrane potential is; the retina and the cortical sheet are separate supports
for the same reason the vascular tree and the cortical sheet are separate
supports, and for no deeper reason than that.  nothing in this file is a
boundary condition.

so there is **one** process here, `transduction`, and eight implementations of
it.  that ratio is deliberate and it is the file's main claim.  a
photoreceptor, a hair cell, a merkel disc and a carotid baroreceptor are not
eight kinds of process; they are one process -- a physical quantity at a
receptor surface driving a receptor state variable and, through it, primary
afferent firing -- with eight different f and eight different theta.  (I, O, T)
is ontology and is shared.  writing them as eight processes would assert a
structural difference that is not there and would forbid the thing the schema
exists to allow, which is swapping f without touching a declaration.

what genuinely differs between them is the *filter*, and it differs by five
orders of magnitude in time constant.  hair-cell mechanoelectrical transduction
gates in tens of microseconds; a rod integrates for two hundred milliseconds; a
carotid chemoreceptor answers a step change in arterial pH over tens of
seconds.  every implementation below is therefore mostly a statement about a
timescale and an adaptation rule, and the region restrictions matter as much as
the parameters -- an f written for the cochlea evaluated on the retina is not
approximately right, it is meaningless.

three honest admissions about what the declared ontology can and cannot carry.

*there is no radiometric component.*  docs/CONTRACT.md declares no photon flux
or irradiance anywhere.  the only declared carrier of photic drive is
`device.display_luminance`, which lives on the display support, so the
photoreceptor implementation reads that and folds the eye's optics -- pupil
area, transmittance, the point spread of the cornea and lens, and the
retinotopic mapping from screen to retina -- into a single gain.  that is a
real loss: it means an accommodation change and a pupil change are not
representable here, only absorbed.  it is recorded rather than hidden, and the
correct fix is a radiometric component, not a cleverer gain.

*the mechanical, thermal and material fields live on `head_volume`.*  the skin,
the basilar membrane and the carotid sinus are not in the head volume, and the
contract gives them nowhere else to be.  the mechanical and thermal reads below
are therefore restricted to `head_volume` explicitly, and a materialization that
wants skin indentation at the fingertip is relying on a support the contract
does not currently provide.  saying so in a selector is better than reading
`Everywhere()` and letting the materializer decide what that meant.

*adaptation is one component for eight receptor classes.*  `transduction.adaptation`
is shared, so a materialization that runs the photoreceptor and the
mechanoreceptor implementations at overlapping positions has them writing the
same adaptation variable.  they do not overlap in practice, because their
regions are disjoint supports, and that disjointness is doing load-bearing work
the type system does not check.
"""

from __future__ import annotations

import numpy as np

from ibm.processes.base import (
    constant_gain,
    double_exponential,
    implementation,
    low_pass,
    parallel,
    process,
    resonator,
    series,
)
from ibm.registry import Form
from ibm.vocabulary import (
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
# bands, and the regions each implementation is restricted to
# ---------------------------------------------------------------------------

#: photic drive.  the ceiling is the display's, not the retina's: a photoreceptor
#: cannot follow anything near 100 Hz, and the point of reading this wide is that
#: the low-pass which destroys the fast content is *in the implementation*, where
#: evidence can move it, rather than assumed in the selector.
PHOTIC = Band(0.0, 100.0)

#: acoustic drive at the ear.  the floor is not zero: below ~20 Hz the middle ear
#: transmits essentially nothing, so declaring DC drive to the cochlea would be
#: declaring a coupling that the anatomy does not have.
ACOUSTIC = Band(20.0, 20000.0)

#: hair-cell output.  phase locking in the auditory nerve degrades above ~1 kHz
#: and is gone by ~4-5 kHz, above which the afferent carries envelope and place
#: information only.  5 kHz is where the spectral form stops buying anything.
HAIR_CELL_OUT = Band(0.0, 5000.0)

#: cutaneous mechanoreception.  pacinian corpuscles respond to vibration up to
#: roughly 1 kHz, which is two orders of magnitude above every other receptor
#: class in this file and the reason the band is not shared.
TACTILE = Band(0.0, 1000.0)

#: thermo-, chemo- and baroreception.  all three are limited by diffusion or by
#: a second-messenger cascade rather than by a channel, and none of them carries
#: usable information above a few tens of hertz.  this is the band the *drive* to
#: them is read over; each receptor's own output band is narrower still, and the
#: three constants below are those, matched to what the components declare.
SLOW_RECEPTOR = Band(0.0, 50.0)

#: thermoreceptor and nociceptor output.  their transduction is limited by the
#: thermal time constant of the tissue between the surface and the ending, which
#: is hundreds of milliseconds -- there is nothing above 20 Hz to carry.
THERMO_NOCI = Band(0.0, 20.0)

#: chemoreceptor output.  the slowest receptor class in the inventory: a
#: molecule has to arrive and a G-protein cascade has to answer.
CHEMO = Band(0.0, 10.0)

#: adaptation.  it is by construction the slow envelope of the drive, so writing
#: it over a wider band than this would be writing a variable that adapts as fast
#: as the thing it is adapting to -- which is not adaptation.
ADAPTATION = Band(0.0, 5.0)

#: vestibular.  head movement has essentially no power above ~20 Hz, and the
#: canals' own high-pass corner sits near 0.03 Hz.
VESTIBULAR = Band(0.0, 20.0)

#: primary afferent firing.  wide because the auditory nerve is in here with the
#: unmyelinated visceral afferents.
AFFERENT = Band(0.0, 1000.0)

#: tissue structure over an experiment: constant.
STRUCTURE = Band(0.0, 0.001)

RETINA = OnSupport("retina")
COCHLEA = OnSupport("cochlea")
SKIN = OnSupport("body_surface")
LABYRINTH = OnSupport("vestibular_organ")
EPITHELIUM = OnSupport("chemosensory_epithelium")
VISCERA = OnSupport("viscera")
DISPLAY = OnSupport("display")
HEAD = OnSupport("head_volume")
BODY = OnSupport("body")
#: the muscle-tendon unit: where spindles and Golgi tendon organs sit.  it is the
#: same support the motor units are on, because a spindle is IN the muscle it
#: reports on -- which is the whole reason fusimotor drive can change what it says.
MUSCLE = OnSupport("motor_units")


# ---------------------------------------------------------------------------
# transfer functions
#
# every one of these is a statement about a timescale.  they are written
# separately rather than as one parameterized filter because the *shape*
# differs -- a cascade, a resonance, a differentiator, a leak -- and collapsing
# them onto one form with a free order would hide the fact that the shapes are
# known and only the constants are uncertain.
# ---------------------------------------------------------------------------


def photoreceptor_transfer(basis, tau_s: float = 0.05, n_stages: int = 4,
                           gain: float = 1.0):
    """photon capture -> photoreceptor membrane potential, as a cascade of lags.

    the phototransduction cascade is literally a chain of first-order chemical
    steps -- rhodopsin to transducin to phosphodiesterase to cGMP hydrolysis to
    channel closure -- so a cascade of identical leaks is the right shape and not
    just a convenient one.  the order matters more than any single tau: a
    four-stage cascade has a much steeper roll-off and a much longer latency to
    peak than one leak of the same total integration time, which is why a
    photoreceptor has a genuine delay and not merely a smoothing.

    the impulse response peaks near ``(n-1) * tau`` and the integration time is
    ``n * tau``, so cone and rod are the same form with tau differing by a factor
    of four.  it is linear, which is a small-signal statement: photoreceptor gain
    falls with background luminance (weber adaptation) and the linearization is
    taken about one adapting level.
    """
    return gain * series(*[low_pass(basis, tau_s)] * max(int(n_stages), 1))


def hair_cell_transfer(basis, cf_hz: float = 1000.0, q: float = 7.0,
                       tau_transduction_s: float = 4e-5, gain: float = 1.0):
    """basilar membrane resonance in series with mechanoelectrical transduction.

    two very different things in one filter, and separating them in the source
    is the point.  the resonance is *mechanical* -- a place on the basilar
    membrane tuned to a characteristic frequency with a quality factor the
    outer hair cells actively set -- and the transduction is *electrical*, the
    tip-link-gated channel plus the hair cell's own RC.

    the transduction pole is deliberately far above the audible range.  channel
    gating is a few tens of microseconds, which is what makes phase locking to a
    kilohertz tone possible at all; if this pole sat where a neuronal membrane's
    does, interaural time differences of ten microseconds could not be
    represented and binaural hearing would not work.  it is the fastest time
    constant anywhere in the ibm-1 inventory.
    """
    return gain * series(resonator(basis, cf_hz, q), low_pass(basis, tau_transduction_s))


def rapid_adaptation_transfer(basis, tau_adapt_s: float = 0.03,
                              tau_membrane_s: float = 2e-3, gain: float = 1.0):
    """a rapidly adapting receptor: a differentiator with a leak, then a membrane.

    ``H = g * (i w t_a / (1 + i w t_a)) * (1 / (1 + i w t_m))``.  the first
    factor is what "rapidly adapting" means as a filter -- zero DC gain, so a
    maintained indentation produces no maintained response, and the receptor
    reports change rather than level.  the second is the receptor's own membrane.

    band-pass with a peak near ``1 / sqrt(t_a t_m)``, which is how meissner
    (flutter, ~30 Hz) and pacinian (vibration, ~250 Hz) corpuscles end up as the
    same declaration with two parameter sets.  it is also the vestibular canal's
    shape, for an entirely different mechanical reason, which is why
    `canal_transfer` below is a thin alias rather than a new form.
    """
    return gain * series(1.0 - low_pass(basis, tau_adapt_s), low_pass(basis, tau_membrane_s))


def slow_adaptation_transfer(basis, static_fraction: float = 0.4,
                             tau_adapt_s: float = 2.0, tau_membrane_s: float = 0.01,
                             gain: float = 1.0):
    """a slowly adapting receptor: a maintained term in parallel with an adapting one.

    ``H = g * (s + (1-s) * i w t_a/(1 + i w t_a)) / (1 + i w t_m)``.  the whole
    content is the parameter ``s``: it is the fraction of the response that
    survives to DC, and therefore the answer to whether this receptor can report
    a constant.  merkel discs and muscle spindles have s well above zero and can
    signal a held posture; a rapidly adapting receptor is this form at s = 0.

    the same structure serves the thermoreceptors and the baroreceptors, where
    the static/dynamic split is the classical description of the physiology and
    not a modelling convenience.
    """
    s = float(static_fraction)
    dynamic = (1.0 - s) * (1.0 - low_pass(basis, tau_adapt_s))
    return gain * series(parallel(constant_gain(basis, s), dynamic),
                         low_pass(basis, tau_membrane_s))


def canal_transfer(basis, tau_long_s: float = 5.7, tau_short_s: float = 3e-3,
                   gain: float = 1.0):
    """the semicircular canal as a torsion pendulum.

    ``H = g * i w T_l / ((1 + i w T_l)(1 + i w T_s))`` from head *angular
    velocity* to cupula deflection.  the long constant is the endolymph's
    viscous relaxation against cupular stiffness; the short one is the fluid's
    inertia.  between them -- roughly 0.03 to 30 Hz -- the canal is a velocity
    transducer with flat gain, which is precisely the band natural head movement
    occupies, and outside it the canal is wrong in a way that is directly
    perceptible: a sustained constant-velocity rotation feels like it stops,
    because the response decays with T_l.

    the same band-pass shape as a rapidly adapting mechanoreceptor, arrived at
    from fluid mechanics rather than from channel adaptation.  the coincidence is
    worth noticing and is not worth unifying: the parameters that move under
    evidence are different quantities.
    """
    return rapid_adaptation_transfer(basis, tau_long_s, tau_short_s, gain)


def diffusive_receptor_transfer(basis, tau_diffusion_s: float = 1.0,
                                tau_cascade_s: float = 8.0, gain: float = 1.0):
    """a chemically driven receptor: diffusion to the sensor, then a cascade.

    two lags with an order of magnitude between them, for the same reason
    `double_exponential` exists on the synaptic side.  the fast one is the
    stimulus reaching the receptor -- odorant through mucus, CO2 across the
    glomus cell membrane -- and the slow one is the second-messenger cascade
    that actually changes the receptor potential.  collapsing them onto one pole
    misplaces the latency, and latency is most of what a chemosensory
    measurement can distinguish.
    """
    return gain * double_exponential(basis, tau_diffusion_s, tau_cascade_s)


# ---------------------------------------------------------------------------
# the process
# ---------------------------------------------------------------------------

TRANSDUCTION = process(
    id="transduction",
    doc="""a physical quantity at a receptor surface drives receptor state, and
    receptor state drives primary afferent firing.

    one process, eight implementations.  the shared ontology is real: in every
    case a field the model already carries -- luminance at a display, pressure in
    the head volume, temperature, arterial pH, blood pressure -- is read at a
    receptor surface, passed through a filter with an adaptation rule, and
    written into a receptor component and into `neural.afferent.activity`.  what
    varies is f, and f varies enormously.

    the inputs are region-restricted per receptor class rather than declared as a
    generic "physical state -> receptor state", which is exactly what §4 asks
    for.  the restrictions are not decoration: they are what makes it a type
    error rather than a silent nonsense for the cochlear implementation to be
    evaluated on retinal positions.

    two structural notes.  first, the receptor components and the afferent rate
    are separate outputs on purpose -- spike initiation, refractoriness and rate
    saturation happen at the ganglion cell and not at the receptor, so a model
    that writes receptor potential straight into a firing rate has silently
    asserted that the two saturate together, and they do not.  second,
    `transduction.adaptation` is both read and written: adaptation is the
    receptor's memory of its own recent drive, and without it in the input set
    the process would be memoryless and no implementation could express weber
    scaling, light adaptation, or the difference between the first second of a
    stimulus and the tenth.

    where the whole declaration breaks: it is a receptor-local map with no
    lateral interaction in it.  retinal centre-surround, cochlear two-tone
    suppression, and cutaneous surround inhibition are all lateral, all real, and
    all absent here -- they belong to the local and afferent-pathway processes
    downstream.  reading a photoreceptor implementation as a model of retinal
    output is therefore a mistake this process cannot prevent.""",
    inputs=(
        # photic: no radiometric component exists, so the display's luminance is
        # the only declared carrier of photon flux.  see the module docstring.
        within("device", "display_luminance", region=DISPLAY, band=PHOTIC),
        # acoustic and mechanical drive.  restricted to head_volume because that
        # is where the contract puts the mechanical field, not because the skin
        # is in the head.
        within("device", "speaker_pressure", region=DISPLAY, band=ACOUSTIC),
        within("mechanical", "pressure", "displacement", region=HEAD, band=ACOUSTIC),
        within("mechanical", "displacement", "velocity", "strain", region=HEAD, band=TACTILE),
        within("thermal", "temperature", region=HEAD, band=SLOW_RECEPTOR),
        # chemical and visceral drive.
        within("extracellular", "ph", "k", region=HEAD, band=SLOW_RECEPTOR),
        within("blood", "pressure", "oxygenation", band=SLOW_RECEPTOR),
        # the receptor's memory of its own drive.
        # the proprioceptive inputs.  `effector.length` and `velocity` are what a
        # spindle and a tendon organ actually transduce, and `neural.efferent.gamma`
        # is here because this is the one receptor in the inventory whose GAIN is
        # under central control: fusimotor drive reloads the intrafusal fibre, so
        # without reading gamma the spindle would fall silent during exactly the
        # shortening movements it is needed for.
        within("effector", "length", "velocity", region=MUSCLE, band=TACTILE),
        within("effector", "force", region=MUSCLE, band=TACTILE),
        within("neural", "efferent.gamma", region=BODY, band=AFFERENT),
        within("transduction", "adaptation", band=ADAPTATION),
        # innervation density, which sets how much afferent traffic a given
        # receptor drive actually produces.
        within("structural", "axonal_density", band=STRUCTURE),
    ),
    outputs=(
        within("transduction", "photoreceptor", region=RETINA, band=PHOTIC),
        within("transduction", "hair_cell", region=COCHLEA, band=HAIR_CELL_OUT),
        within("transduction", "mechanoreceptor", region=SKIN, band=TACTILE),
        within("transduction", "thermoreceptor", region=SKIN, band=THERMO_NOCI),
        within("transduction", "nociceptor", region=SKIN, band=THERMO_NOCI),
        within("transduction", "chemoreceptor", region=EPITHELIUM, band=CHEMO),
        within("transduction", "vestibular", region=LABYRINTH, band=VESTIBULAR),
        within("transduction", "baroreceptor", region=VISCERA, band=SLOW_RECEPTOR),
        within("transduction", "spindle_primary", region=MUSCLE, band=TACTILE),
        within("transduction", "spindle_secondary", region=MUSCLE, band=SLOW_RECEPTOR),
        within("transduction", "golgi_tendon", region=MUSCLE, band=TACTILE),
        within("transduction", "joint_receptor", region=SKIN, band=SLOW_RECEPTOR),
        within("transduction", "adaptation", band=ADAPTATION),
        within("neural", "afferent.activity", region=BODY, band=AFFERENT),
        # the same traffic resolved by fibre class.  writing both the aggregate and
        # the classes is deliberate: an implementation that has no opinion about
        # fibre class still writes `activity`, and one that does writes the vector,
        # so selecting a coarse implementation costs resolution rather than failing.
        within("neural", "afferent.ia", "afferent.ib", "afferent.ii",
               region=BODY, band=AFFERENT),
        within("neural", "afferent.abeta", region=BODY, band=AFFERENT),
        within("neural", "afferent.adelta", "afferent.c", region=BODY, band=AFFERENT),
    ),
    topology="local",
    timescale_s=1e-3,
    validity=Validity(
        min_spacing_mm=0.002, max_spacing_mm=5.0, band=Band(0.0, 20000.0),
        note="the floor is the foveal cone spacing, below which a position is a fraction "
             "of one receptor and a population description is a category error.  the "
             "ceiling is a compromise: 5 mm of skin holds receptors of four classes with "
             "different adaptation, and averaging them produces a filter that matches "
             "none of them.  on the cochlea, 5 mm is a third of the whole basilar "
             "membrane and the tonotopy is gone entirely, so the acoustic implementations "
             "have a much tighter effective ceiling than this number suggests."),
    provenance=Provenance.LITERATURE,
    tags=frozenset({"sensory", "peripheral", "boundary"}),
    notes="reads scalar mechanical, thermal, blood and extracellular state and writes "
          "spectral transduction and neural state; the registered scalar -> spectral "
          "conversion applies, which lifts those beliefs into DC and thereby asserts they "
          "have no within-window structure.  for a 100 Hz vibrotactile stimulus that "
          "assertion is false, and it is the sharpest known limitation of this declaration.",
)


# ---------------------------------------------------------------------------
# implementations: eight receptor classes, one process
# ---------------------------------------------------------------------------

def spindle_afferent(x, theta) -> dict:
    """muscle spindle: length and velocity into Ia and II firing, gated by gamma.

    the form is Prochazka's, which is the one that has actually been fitted to
    recorded Ia traffic: a fractional-power velocity term plus a linear length
    term plus an offset.  the fractional power matters -- a spindle's velocity
    response is strongly compressive, so a linear velocity term overestimates the
    afferent burst during fast stretch by an order of magnitude.

    what this implementation adds over `mechanoreceptor_slow`, whose docstring
    names the gap: **fusimotor drive is an input here.**  gamma-dynamic traffic
    contracts the intrafusal poles and reloads the ending, so it multiplies the
    velocity sensitivity and adds a bias.  without it the spindle unloads whenever
    the muscle shortens and falls silent during exactly the movements it is needed
    for, which is why alpha-gamma co-activation exists.
    """
    length = np.asarray(x["effector.length"], dtype=float)
    vel = np.asarray(x["effector.velocity"], dtype=float)
    gamma = np.asarray(x.get("neural.efferent.gamma", 0.0), dtype=float)

    l0 = float(theta.get("rest_length_l0", 1.0))
    k_p = float(theta.get("position_gain_hz", 200.0))
    k_v = float(theta.get("velocity_gain_hz", 65.0))
    power = float(theta.get("velocity_power", 0.5))
    bias = float(theta.get("bias_hz", 10.0))
    g_dyn = float(theta.get("gamma_dynamic_gain", 0.02))
    g_bias = float(theta.get("gamma_bias_hz", 0.3))
    r_max = float(theta.get("r_max_hz", 300.0))

    # fusimotor set: gamma multiplies velocity sensitivity and adds tonic drive
    dyn = 1.0 + g_dyn * gamma
    stretch = np.maximum(length - l0, 0.0)
    # signed fractional power: only lengthening excites, shortening unloads
    lengthening = np.maximum(vel, 0.0)
    r_ia = bias + g_bias * gamma + k_p * stretch + dyn * k_v * lengthening ** power
    # the secondary ending is static: position, essentially no velocity term
    r_ii = bias + g_bias * gamma + k_p * stretch
    return {
        "transduction.spindle_primary": np.clip(-80.0 + 0.2 * r_ia, -80.0, 0.0),
        "transduction.spindle_secondary": np.clip(-80.0 + 0.2 * r_ii, -80.0, 0.0),
        "neural.afferent.ia": np.clip(r_ia, 0.0, r_max),
        "neural.afferent.ii": np.clip(r_ii, 0.0, r_max),
    }


def golgi_tendon_afferent(x, theta) -> dict:
    """Golgi tendon organ: force into Ib firing, logarithmically compressed.

    in SERIES with the muscle rather than in parallel with it, and that is the
    whole functional content.  a spindle unloads when the muscle shortens against
    no load; a tendon organ fires harder the harder the muscle pulls.  so this
    reports force where the spindle reports length, and a controller with only one
    of them cannot tell a limb that moved from a limb that met resistance.

    the compression is real and large: a tendon organ's dynamic range spans three
    orders of magnitude of force, which a linear map cannot cover without either
    saturating at rest or being useless at load.
    """
    force = np.asarray(x["effector.force"], dtype=float)
    k = float(theta.get("force_gain_hz", 40.0))
    f0 = float(theta.get("force_scale_n", 1.0))
    bias = float(theta.get("bias_hz", 5.0))
    r_max = float(theta.get("r_max_hz", 200.0))
    r_ib = bias + k * np.log1p(np.maximum(force, 0.0) / max(f0, 1e-9))
    return {
        "transduction.golgi_tendon": np.clip(-80.0 + 0.3 * r_ib, -80.0, 0.0),
        "neural.afferent.ib": np.clip(r_ib, 0.0, r_max),
    }


implementation(
    name="photoreceptor_cascade",
    process="transduction",
    doc="""rod and cone phototransduction as a four-stage lag cascade.

    restricted to `OnSupport("retina")` -- the `transduction.photoreceptor`
    output selector above -- and driven by `device.display_luminance`, with the
    eye's optics absorbed into `optical_gain`.

    the integration time is the parameter that matters and it is well measured:
    a primate cone integrates over roughly 50 ms and a rod over roughly 200 ms,
    which is the whole reason scotopic vision is temporally sluggish and why
    critical flicker fusion falls from ~60 Hz photopic to ~15 Hz scotopic.  the
    prior below is centred between them, wide enough to cover both, and a
    materialization that knows which receptor class it is modelling should
    narrow it rather than rely on the centre.

    where it breaks: it is linear, and the single most important fact about a
    photoreceptor is that it is not.  gain falls roughly inversely with
    background luminance over ten log units of light -- that is what allows one
    receptor with a ~2 log unit response range to operate from starlight to
    noon -- and here that adaptation appears only as the separate
    `weber_exponent` parameter modulating gain through the adaptation component,
    not as an intrinsic property of the filter.  a step from dark to light is
    predicted badly.""",
    form=Form.LTI,
    transfer=photoreceptor_transfer,
    params={
        "tau_s": lognormal(0.03, 2.2, units="s", provenance=Provenance.LITERATURE,
                           source="primate cone and rod flash responses; baylor et al, "
                                  "schnapf et al",
                           note="per-stage constant; four stages give 50-200 ms total "
                                "integration, cone at the fast end and rod at the slow"),
        "n_stages": uniform(2.0, 6.0, units="stages", provenance=Provenance.LITERATURE,
                            note="the cascade has more chemical steps than this; four is "
                                 "where the fitted impulse response stops improving"),
        "optical_gain": weak(1.0, 30.0, units="receptor mV per cd/m^2",
                             note="absorbs pupil area, ocular transmittance, the point spread "
                                  "of the optics and the screen-to-retina mapping.  it is a "
                                  "stand-in for a radiometric component the contract lacks, "
                                  "and it is not separately identifiable from anything"),
        "weber_exponent": normal(0.9, 0.15, units="dimensionless",
                                 provenance=Provenance.LITERATURE,
                                 note="gain ~ background^-k; k near 1 is weber's law, and the "
                                      "measured value falls slightly short of it"),
        "dark_noise_hz": lognormal(0.006, 3.0, units="isomerizations/s/rod",
                                   provenance=Provenance.LITERATURE,
                                   note="thermal isomerization; the physical floor on absolute "
                                        "visual threshold and the reason the floor is not zero"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="baylor et al 1984; schnapf et al 1990; rieke & rudd 2009",
)

implementation(
    name="hair_cell_cochlear",
    process="transduction",
    doc="""cochlear hair cell: a tuned mechanical resonance into a
    sub-millisecond transduction channel.

    restricted to `OnSupport("cochlea")`.  the two parameters that carry the
    physiology are the quality factor and the transduction time constant, and
    they are uncertain in opposite ways.

    Q is measured and contested in its interpretation rather than its value:
    auditory nerve tuning curves give Q10 of roughly 5-10 across most of the
    human range, rising with characteristic frequency, and the reason it is that
    high rather than the ~1 a passive membrane would give is outer hair cell
    electromotility -- an active, metabolically powered, saturating amplifier.
    modelling it as a passive Q is therefore correct only at moderate levels: at
    high levels the cochlear amplifier saturates, tuning broadens, and the
    resonator's Q should fall with input amplitude.  it does not here.

    the transduction constant is not contested and is remarkable: tip-link-gated
    channels open in tens of microseconds.  it is set that fast because it must
    be -- phase locking up to a kilohertz and the ten-microsecond interaural
    time differences that azimuthal localization depends on both require it.

    where it breaks: linear and passive, so no two-tone suppression, no
    distortion products, no compressive input-output function, and no
    level-dependent tuning.  every one of those is a first-order feature of real
    cochlear mechanics.""",
    form=Form.LTI,
    transfer=hair_cell_transfer,
    params={
        "cf_hz": lognormal(1000.0, 8.0, units="Hz", provenance=Provenance.LITERATURE,
                           source="greenwood tonotopic map",
                           note="a per-position quantity: the tonotopic map is the topology "
                                "here, and the wide prior is the range across the membrane, "
                                "not uncertainty at one place"),
        "q": normal(7.0, 2.0, units="dimensionless", provenance=Provenance.LITERATURE,
                    source="auditory nerve tuning curves, Q10 5-10",
                    note="active; it falls towards ~1 with outer hair cell loss, which is "
                         "why sensorineural hearing loss broadens tuning as well as "
                         "raising threshold"),
        "tau_transduction_s": lognormal(4e-5, 3.0, units="s",
                                        provenance=Provenance.LITERATURE,
                                        note="mechanoelectrical channel gating, tens of "
                                             "microseconds; the fastest constant in the "
                                             "inventory by two orders of magnitude"),
        "tau_hair_cell_rc_s": lognormal(3e-4, 2.5, units="s",
                                        provenance=Provenance.LITERATURE,
                                        note="the cell's own membrane; it, not the channel, is "
                                             "what finally limits phase locking"),
        "compression_exponent": normal(0.3, 0.1, units="dimensionless",
                                       provenance=Provenance.LITERATURE,
                                       note="basilar membrane input-output slope at CF, roughly "
                                            "0.2-0.4.  declared but unused by this linear "
                                            "form, so that selecting a nonlinear f later has "
                                            "somewhere to put it"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.LITERATURE,
    source="greenwood 1990; ruggero et al 1997; corey & hudspeth 1979",
)

implementation(
    name="mechanoreceptor_rapid",
    process="transduction",
    doc="""rapidly adapting cutaneous mechanoreceptors: meissner and pacinian.

    restricted to `OnSupport("body_surface")`.  a differentiator with a leak, so
    DC gain is exactly zero and a maintained indentation produces no maintained
    response -- which is not an approximation, it is the defining property of the
    class and the reason you stop feeling your clothes.

    the two afferent types are one declaration with two parameter sets.  meissner
    (RA1) adapts over tens of milliseconds and peaks in the flutter range around
    30 Hz; pacinian (RA2) is wrapped in a lamellated capsule that mechanically
    high-passes its own input, adapts in milliseconds, and peaks near 250 Hz with
    a displacement threshold below a micrometre.  the capsule is the reason a
    pacinian corpuscle is the most sensitive mechanical transducer in the body
    and also the reason it can tell you nothing about pressure.

    the prior below is centred between the two and is correspondingly wide.  a
    materialization that knows which afferent class it wants should narrow it;
    running the wide prior means the posterior over vibrotactile tuning stays
    uninformative, which is honest but not useful.""",
    form=Form.LTI,
    transfer=rapid_adaptation_transfer,
    params={
        "tau_adapt_s": lognormal(0.02, 6.0, units="s", provenance=Provenance.LITERATURE,
                                 source="microneurography, human glabrous skin",
                                 note="RA1 tens of ms; RA2 a few ms.  the spread spans both"),
        "tau_membrane_s": lognormal(2e-3, 3.0, units="s", provenance=Provenance.LITERATURE,
                                    note="receptor membrane; with tau_adapt it sets the best "
                                         "frequency, ~30 Hz for RA1 and ~250 Hz for RA2"),
        "gain": weak(1.0, 20.0, units="mV per um indentation",
                     note="varies by two orders of magnitude across the body; the fingertip "
                          "and the back are not the same receptor population"),
        "threshold_um": lognormal(1.0, 10.0, units="um", provenance=Provenance.LITERATURE,
                                  note="RA2 threshold is below 1 um at best frequency; RA1 is "
                                       "nearer 10 um.  declared but unused by the linear form"),
        "innervation_density_per_cm2": lognormal(50.0, 6.0, units="units/cm^2",
                                                 provenance=Provenance.LITERATURE,
                                                 source="johansson & vallbo 1979",
                                                 note="~140/cm^2 RA1 at the fingertip, under "
                                                      "10/cm^2 on the forearm; this is what "
                                                      "sets two-point discrimination"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="johansson & vallbo 1979; talbot et al 1968",
)

implementation(
    name="mechanoreceptor_slow",
    process="transduction",
    doc="""slowly adapting cutaneous mechanoreceptors and muscle spindles: merkel
    and ruffini, and the spindle's static and dynamic response.

    restricted to `OnSupport("body_surface")`.  the same filter as the rapid form
    with one parameter changed -- `static_fraction` above zero -- and that single
    parameter is the whole physiological distinction: a receptor that can report
    a constant, and therefore posture, indentation depth and joint angle, versus
    one that can only report change.

    the muscle spindle is included here rather than given its own implementation
    because its Ia/II split is the same static/dynamic decomposition under
    another name, and because the state it writes is the same
    `transduction.mechanoreceptor` component read by the same afferents.  the
    thing that makes a spindle genuinely different -- fusimotor drive, which lets
    the nervous system set the sensitivity of its own sensor -- is *not*
    representable here, since it is efferent pressure on this process's theta
    rather than on its inputs.  the schema supports that (a process may write
    another's parameters, §4) and no process in the inventory currently does it,
    which is a real gap and is recorded as one.""",
    form=Form.LTI,
    transfer=slow_adaptation_transfer,
    params={
        "static_fraction": normal(0.4, 0.2, units="dimensionless",
                                  provenance=Provenance.LITERATURE,
                                  note="the fraction surviving to DC.  SA1 sustains roughly "
                                       "half its peak over seconds; this parameter *is* the "
                                       "difference between the SA and RA classes"),
        "tau_adapt_s": lognormal(2.0, 4.0, units="s", provenance=Provenance.LITERATURE,
                                 note="slow adaptation over seconds, not milliseconds"),
        "tau_membrane_s": lognormal(0.01, 3.0, units="s", provenance=Provenance.LITERATURE),
        "gain": weak(1.0, 20.0, units="mV per um indentation"),
        "spindle_dynamic_index": normal(0.5, 0.25, units="dimensionless",
                                        provenance=Provenance.LITERATURE,
                                        note="Ia velocity sensitivity relative to II length "
                                             "sensitivity; under fusimotor control in vivo, "
                                             "and therefore not really a constant"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="johansson & vallbo 1979; matthews 1972",
)

implementation(
    name="thermoreceptor_static_dynamic",
    process="transduction",
    doc="""warm and cold thermoreceptors: a maintained rate that encodes
    temperature plus a dynamic overshoot that encodes its rate of change.

    restricted to `OnSupport("body_surface")`, reading `thermal.temperature`.
    the same static/dynamic filter as the slowly adapting mechanoreceptor, and
    with a high static fraction, because a thermoreceptor's maintained discharge
    genuinely does encode absolute skin temperature -- one of the few places in
    the sensory periphery where that is true.

    the physiology that makes this hard is that the static response is not
    monotone.  cold fibres peak near 25-30 C and warm fibres near 40-45 C, so the
    map from temperature to rate is a pair of overlapping bells, and no linear
    gain can represent it.  the `peak_temperature_c` and `tuning_width_c`
    parameters below record where those bells sit; the linear transfer function
    above does not use them, and a materialization that cares about absolute
    temperature coding rather than about transients needs a nonlinear f that
    does.  the paradoxical cold sensation at high temperatures -- cold fibres
    firing again above ~45 C -- is not representable at all.

    the dynamic term is why a thermal step feels far stronger than the steady
    state it settles to, and why the same 32 C surface feels warm to a cold hand
    and cold to a warm one.""",
    form=Form.LTI,
    transfer=slow_adaptation_transfer,
    params={
        "static_fraction": normal(0.7, 0.2, units="dimensionless",
                                  provenance=Provenance.LITERATURE,
                                  note="high, because maintained discharge encodes absolute "
                                       "temperature; the dynamic overshoot is the smaller part"),
        "tau_adapt_s": lognormal(10.0, 3.0, units="s", provenance=Provenance.LITERATURE,
                                 note="adaptation to a temperature step takes tens of seconds"),
        "tau_membrane_s": lognormal(0.3, 3.0, units="s", provenance=Provenance.LITERATURE,
                                    note="dominated by the thermal time constant of the tissue "
                                         "between the surface and the ending, not by the "
                                         "membrane; the ending sits 0.2-0.6 mm deep"),
        "peak_temperature_c": normal(33.0, 8.0, units="degC",
                                     provenance=Provenance.LITERATURE,
                                     note="cold fibres ~27 C, warm fibres ~42 C.  the wide "
                                          "prior is the gap between two populations, not "
                                          "uncertainty about either"),
        "tuning_width_c": normal(8.0, 3.0, units="degC", provenance=Provenance.LITERATURE),
        "gain": weak(1.0, 20.0, units="Hz per degC"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="hensel 1981; darian-smith et al 1979",
)

implementation(
    name="nociceptor_sensitizing",
    process="transduction",
    doc="""nociceptors: a high-threshold, non-adapting, *sensitizing* receptor.

    restricted to `OnSupport("body_surface")`, reading mechanical, thermal and
    extracellular pH.  the only implementation in this file that is not LTI, and
    it is not LTI for a reason that would be wrong to smooth over: a nociceptor's
    gain goes *up* with use.  every other receptor here adapts -- repeated or
    maintained stimulation reduces the response -- and a nociceptor sensitizes,
    lowering threshold and raising gain, on a timescale of minutes to hours,
    through inflammatory mediators acting on the terminal.  a linear filter with
    a fixed gain has the sign of the most clinically important property of the
    system backwards.

    the threshold is the second non-linear essential.  a nociceptor is silent,
    not merely small, below roughly 43 C for heat and below a mechanical
    intensity that is frankly damaging.  a linearization about the resting point
    predicts a proportional response to gentle touch, which is not a small error;
    it is the difference between touch and pain.

    the priors are a mixture.  the TRPV1 heat threshold and the proton gating are
    well measured.  the sensitization dynamics are not -- they are a whole
    signalling cascade compressed into two constants -- so those carry
    speculative priors.  C-fibre conduction is slow enough that the second pain
    arrives about a second after the first, and that latency lives in
    `afferent_propagation` rather than here.""",
    form=Form.RATE,
    params={
        "heat_threshold_c": normal(43.0, 2.0, units="degC",
                                   provenance=Provenance.LITERATURE,
                                   source="TRPV1 activation threshold",
                                   note="falls by several degrees under inflammation, which is "
                                        "what makes sunburnt skin hurt in a warm shower"),
        "proton_threshold_ph": normal(6.5, 0.3, units="pH",
                                      provenance=Provenance.LITERATURE,
                                      note="ASIC and TRPV1 proton gating; tissue acidosis is a "
                                           "direct nociceptor stimulus, not an indirect one"),
        "mechanical_threshold_mn": lognormal(20.0, 4.0, units="mN",
                                             provenance=Provenance.LITERATURE,
                                             note="A-delta high-threshold mechanoreceptors; two "
                                                  "orders above the RA1 threshold"),
        "tau_sensitization_s": speculative(300.0, 10.0, units="s",
                                           note="peripheral sensitization builds over minutes; "
                                                "the constant is a compression of an entire "
                                                "inflammatory cascade into one number"),
        "sensitization_gain": speculative(2.0, 5.0, units="dimensionless",
                                          note="how far gain rises at full sensitization.  sign "
                                               "is known, magnitude is not"),
        "silent_fraction": uniform(0.0, 0.5, units="dimensionless",
                                   provenance=Provenance.LITERATURE,
                                   note="mechanically insensitive afferents that only become "
                                        "responsive after inflammation; a substantial minority "
                                        "of the C-fibre population"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.WEAK,
    source="caterina et al 1997; schmidt et al 1995",
)

implementation(
    name="chemoreceptor_diffusive",
    process="transduction",
    doc="""olfactory, gustatory, and arterial chemoreceptors as a two-lag
    diffusive cascade.

    restricted to `OnSupport("chemosensory_epithelium")` for the olfactory and
    gustatory cases, and reading `extracellular.ph` and `blood.oxygenation` for
    the arterial ones.  grouped into one implementation because the rate-limiting
    physics is the same in all of them -- a molecule has to get to a receptor,
    and then a G-protein cascade has to respond -- and that shared bottleneck is
    why no chemosensory system anywhere has a bandwidth above a few hertz.

    the natural coordinate here is receptor identity rather than position, which
    the `chemosensory_epithelium` support already says.  that matters: olfactory
    coding is combinatorial across ~400 human receptor types, so a materialized
    "position" on this support is a receptor type, and the distance between two
    of them has nothing to do with anatomy.  the topology, not the filter, is
    where olfaction's structure lives, and this implementation carries none of it.

    the priors split sharply.  the sniff-locked timing of olfactory transduction
    and the carotid body's response latency are measured.  receptor-ligand
    affinities are not -- there is no general theory that predicts which odorant
    binds which receptor -- so `affinity_scale` is speculative and will stay that
    way.""",
    form=Form.LTI,
    transfer=diffusive_receptor_transfer,
    params={
        "tau_diffusion_s": lognormal(0.15, 3.0, units="s",
                                     provenance=Provenance.LITERATURE,
                                     note="odorant through the mucus layer; comparable to a "
                                          "sniff, which is why sniffing is part of smelling "
                                          "rather than incidental to it"),
        "tau_cascade_s": lognormal(1.0, 4.0, units="s", provenance=Provenance.LITERATURE,
                                   note="second-messenger transduction and its adaptation.  "
                                        "the carotid body's response to a step change in "
                                        "arterial PO2 is at the slow end, ~10 s"),
        "affinity_scale": speculative(1.0, 30.0, units="dimensionless",
                                      note="receptor-ligand affinity.  no theory predicts it "
                                           "and the space is ~400 receptors by an unbounded "
                                           "set of ligands; this will not become literature"),
        "po2_half_response_mmhg": normal(60.0, 15.0, units="mmHg",
                                         provenance=Provenance.LITERATURE,
                                         note="carotid body output rises steeply only below "
                                              "~60 mmHg, which is why mild hypoxia is not felt"),
        "pco2_gain": weak(1.0, 5.0, units="Hz per mmHg",
                          note="central chemoreception dominates the CO2 response and acts on "
                               "brainstem pH, not on this receptor"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
    source="firestein 2001; kumar & prabhakar 2012",
)

implementation(
    name="vestibular_canal_otolith",
    process="transduction",
    doc="""semicircular canals and otolith organs: the torsion pendulum, and a
    static gravitoinertial sensor next to it.

    restricted to `OnSupport("vestibular_organ")`.  the canals are the cleanest
    band-pass in the whole inventory and the cleanest example of a receptor whose
    errors are directly perceptible.  between roughly 0.03 and 30 Hz the canal
    reports angular velocity with flat gain.  outside that band it reports
    something else, and the something else has names: a sustained constant-
    velocity rotation feels like it decays to nothing over about six seconds
    (the long time constant), and the illusory counter-rotation on stopping is
    the same filter's release.  the central velocity-storage mechanism partly
    compensates for this downstream, and that compensation is not here.

    the otoliths are the parallel path, and the parameters below carry both
    because they are the same organ.  an otolith cannot distinguish linear
    acceleration from a tilt relative to gravity -- einstein's equivalence
    principle applies to the utricle -- and resolving that ambiguity requires
    combining canal and otolith signals centrally.  this implementation therefore
    cannot be read as a model of perceived tilt; it produces the ambiguous
    afferent signal that a downstream process would have to disambiguate.""",
    form=Form.LTI,
    transfer=canal_transfer,
    params={
        "tau_long_s": lognormal(5.7, 1.4, units="s", provenance=Provenance.LITERATURE,
                                source="human cupular time constant from post-rotatory "
                                       "nystagmus, 4-7 s",
                                note="the dominant time constant of the peripheral canal; "
                                     "central velocity storage extends the *perceptual* one "
                                     "to 15-20 s, which is a different quantity"),
        "tau_short_s": lognormal(3e-3, 2.0, units="s", provenance=Provenance.LITERATURE,
                                 note="endolymph inertia; puts the upper corner well above "
                                      "any natural head movement"),
        "canal_gain": weak(1.0, 5.0, units="afferent Hz per deg/s"),
        "otolith_gain": weak(1.0, 5.0, units="afferent Hz per g",
                             note="the parallel static path; a separate transfer function "
                                  "that this LTI form folds into a gain rather than "
                                  "representing properly"),
        "resting_discharge_hz": normal(90.0, 20.0, units="Hz",
                                       provenance=Provenance.LITERATURE,
                                       note="vestibular afferents fire ~90 Hz at rest so that "
                                            "they can encode both directions of rotation in "
                                            "one rate; this high tonic rate is why unilateral "
                                            "loss is catastrophic rather than merely halving "
                                            "sensitivity"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="goldberg & fernandez 1971; van egmond et al 1949",
)

implementation(
    name="baroreceptor_arterial",
    process="transduction",
    doc="""carotid sinus and aortic arch baroreceptors: an adapting stretch
    sensor with a sigmoid operating range.

    restricted to `OnSupport("viscera")`, reading `blood.pressure`.  included in
    this file, alongside the retina and the cochlea, because the architecture's
    claim is that interoception is transduction from visceral state and needs no
    machinery of its own -- and the claim holds: the same static/dynamic filter
    that describes a merkel disc describes a baroreceptor, because both are
    stretch sensors.

    the two facts a materialization needs from this are the operating range and
    the resetting.  baroreceptors are near-silent below ~60 mmHg and saturate
    above ~180, so the sigmoid's steep segment sits exactly where mean arterial
    pressure normally lives -- the sensor is placed to maximise sensitivity where
    it matters, and outside that range the reflex has essentially no gain.  and
    they *reset*: over hours to days the operating point tracks the prevailing
    pressure, which is why baroreceptors do not oppose the development of chronic
    hypertension and why they cannot be read as a long-term pressure reference at
    all.  the resetting timescale is far outside any window this model runs in,
    so it appears as a parameter and not as dynamics.

    the dynamic term is large here: baroreceptor firing tracks dP/dt strongly, so
    the same mean pressure with a wider pulse produces more afferent traffic.""",
    form=Form.LTI,
    transfer=slow_adaptation_transfer,
    params={
        "static_fraction": normal(0.5, 0.2, units="dimensionless",
                                  provenance=Provenance.LITERATURE,
                                  note="roughly half the response survives to DC; the rest "
                                       "tracks the rate of pressure change"),
        "tau_adapt_s": lognormal(3.0, 3.0, units="s", provenance=Provenance.LITERATURE),
        "tau_membrane_s": lognormal(0.05, 3.0, units="s", provenance=Provenance.LITERATURE,
                                    note="fast enough to follow the arterial pulse, which it "
                                         "must, since pulsatility is part of the signal"),
        "threshold_mmhg": normal(60.0, 10.0, units="mmHg", provenance=Provenance.LITERATURE,
                                 note="below this the carotid baroreceptor is silent and the "
                                      "baroreflex has no gain at all"),
        "saturation_mmhg": normal(180.0, 20.0, units="mmHg",
                                  provenance=Provenance.LITERATURE),
        "reset_tau_hours": speculative(24.0, 5.0, units="h",
                                       note="operating-point resetting; far outside any window "
                                            "this model runs, so it is a parameter rather than "
                                            "dynamics.  it is also why baroreceptors do not "
                                            "correct chronic hypertension"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="chapleau & abboud 1987; kirchheim 1976",
)

implementation(
    name="learned_receptor_bank",
    process="transduction",
    doc="""one learned filter bank standing in for all eight receptor classes,
    with the analytic forms as its prior mean.

    worth having for a reason specific to this process rather than as a
    reflex.  every implementation above is linear about one operating point, and
    the single largest systematic error in all of them is the same: real
    receptors adapt their *gain* to the recent stimulus distribution, and none of
    these filters can.  that failure has a shape -- it is a slow multiplicative
    modulation driven by the same input -- and a learned residual with access to
    `transduction.adaptation` can represent it, which no fixed transfer function
    can.

    it is also the right place to carry the nonlinearity that matters most in
    each modality: photoreceptor light adaptation, cochlear compression, and
    nociceptor sensitization are three instances of one computational motif, and
    a learned form can share structure across them that eight separately fitted
    analytic filters cannot.

    the prior mean is the analytic bank, so with no data it reproduces it
    exactly.  what moves it is psychophysical and microneurographic data, and
    there is not much of either per receptor class, so the honest expectation is
    that most of these parameters stay near their prior.  `Tying.EMBEDDING` is
    the choice that makes that survivable: receptor class is an embedding rather
    than a per-site parameter, so evidence from one cochlear position informs
    every other one.""",
    form=Form.LEARNED,
    params={
        "prior_scale": weak(1.0, 3.0, note="weight prior; a learned f is a different p(theta), "
                                           "not a different declaration"),
        "residual_gain": weak(0.25, 5.0,
                              note="the learned term is a correction on the analytic filter, so "
                                   "the prior stays physical and a data-starved posterior "
                                   "degrades to the literature form rather than to noise"),
        "adaptation_coupling": speculative(1.0, 10.0,
                                           note="how strongly the learned gain is modulated by "
                                                "transduction.adaptation.  this is the term "
                                                "the analytic bank structurally cannot have"),
        "embedding_dim": uniform(4.0, 32.0, units="dimensions",
                                 note="receptor-class embedding width; see ibm.processes.nn "
                                      "for what the count of free parameters is a function of"),
    },
    tying=Tying.EMBEDDING,
    provenance=Provenance.WEAK,
)

__all__ = ["TRANSDUCTION"]


implementation(
    name="spindle_prochazka",
    process="transduction",
    doc="""the muscle spindle with fusimotor drive as an input.

    `mechanoreceptor_slow` folds the spindle into the cutaneous slowly-adapting
    class and says why that is wrong: "the thing that makes a spindle genuinely
    different -- fusimotor drive, which lets the nervous system set the
    sensitivity of its own sensor -- is *not* representable here".  this
    implementation is that gap closed, by the simpler of the two available routes:
    gamma is read as an INPUT rather than as efferent pressure on theta.  the
    schema permits either; reading it is cheaper and keeps the process graph
    acyclic in parameters while still closing the loop in state.

    it writes both the receptor potentials and the Ia/II rates directly, because
    the spindle is one of the few receptors where the afferent code is better
    characterized than the receptor potential.""",
    form=Form.RATE,
    fn=spindle_afferent,
    params={
        "rest_length_l0": normal(1.0, 0.05, units="L0", provenance=Provenance.LITERATURE,
                                 note="the length at which the ending is just loaded"),
        "position_gain_hz": lognormal(200.0, 2.0, units="Hz per L0",
                                      provenance=Provenance.LITERATURE,
                                      source="Prochazka & Gorassini, cat Ia recordings"),
        "velocity_gain_hz": lognormal(65.0, 2.0, units="Hz per (L0/s)^p",
                                      provenance=Provenance.LITERATURE,
                                      source="Prochazka, fractional-power velocity term"),
        "velocity_power": normal(0.5, 0.1, units="dimensionless",
                                 provenance=Provenance.LITERATURE,
                                 note="strongly compressive; a linear term overestimates "
                                      "the stretch burst by an order of magnitude"),
        "bias_hz": lognormal(10.0, 2.0, units="Hz", provenance=Provenance.LITERATURE),
        "gamma_dynamic_gain": weak(0.02, 5.0, units="per Hz",
                                   note="how much fusimotor drive multiplies velocity "
                                        "sensitivity.  this parameter IS fusimotor set"),
        "gamma_bias_hz": weak(0.3, 5.0, units="Hz per Hz"),
        "r_max_hz": lognormal(300.0, 1.5, units="Hz", provenance=Provenance.LITERATURE),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="Prochazka 1999; Mileusnic & Loeb 2006 for the intrafusal mechanics this "
           "lumps")


def nociceptor_afferent(x, theta) -> dict:
    """noxious stimulus into A-delta and C firing, with the threshold that defines it.

    a nociceptor is not a sensitive mechanoreceptor.  the defining property is a
    HIGH THRESHOLD -- it is silent through the whole innocuous range and begins to
    fire only where tissue is threatened -- and that is what makes its rate usable
    as a reward signal rather than as another channel of touch.  a receptor that
    reported every contact would say nothing about damage.

    the two fibre classes are separated because they carry different information
    and the model already declares both with resolved conduction velocities.
    A-delta is myelinated and fast: first pain, sharp, well localised, and it
    arrives in time to withdraw from the thing that caused it.  C is unmyelinated
    and slow: second pain, burning, poorly localised, and it outlasts the
    stimulus.  48 of 72 declared trunks carry one or both, so a materialization
    that lumps them is discarding a distinction the periphery already makes -- and
    the delay between them is not a detail, it is the reason withdrawal happens
    before suffering does.

    three modalities converge on the same ending because polymodal nociceptors are
    the common case: mechanical load beyond the damage threshold, temperature
    outside the innocuous band in either direction, and chemical irritants
    including the ones inflamed tissue produces itself.

    SENSITISATION is included and is the part most easily left out.  injured
    tissue lowers its own threshold, so a stimulus that was innocuous becomes
    painful and one that was painful becomes worse.  Without it a body cannot
    learn to protect an injury, which is most of what pain is for.  It is carried
    as a state the caller advances, not inferred here.

    what this is NOT: an account of pain.  It is the receptor.  Everything that
    makes pain an experience -- affect, attention, expectation, the descending
    control that can abolish it -- happens far past this function, and naming the
    output `nociceptor` rather than `pain` is deliberate.
    """
    load = np.asarray(x["effector.force"], dtype=float)
    temp = np.asarray(x.get("thermal.tissue_temperature_c", 37.0), dtype=float)
    chem = np.asarray(x.get("transduction.irritant_concentration", 0.0), dtype=float)

    thr_n = float(theta.get("mechanical_threshold_n", 8.0))
    hot_c = float(theta.get("heat_threshold_c", 43.0))
    cold_c = float(theta.get("cold_threshold_c", 15.0))
    k_mech = float(theta.get("mechanical_gain_hz_per_n", 1.6))
    k_heat = float(theta.get("thermal_gain_hz_per_c", 3.5))
    k_chem = float(theta.get("chemical_gain_hz", 25.0))
    sens = float(theta.get("sensitisation", 1.0))
    r_max = float(theta.get("r_max_hz", 100.0))
    c_frac = float(theta.get("c_fibre_fraction", 0.7))
    c_tau = float(theta.get("c_persistence", 0.85))

    # sensitisation lowers the thresholds; it does not add a baseline, because a
    # nociceptor that fires at rest is a pathology and not the normal case.
    thr_n /= max(sens, 1e-6)
    hot_c -= (sens - 1.0) * float(theta.get("sensitisation_shift_c", 4.0))

    # HIGH THRESHOLD: exactly zero drive through the innocuous range.
    d_mech = k_mech * np.maximum(load - thr_n, 0.0)
    d_heat = k_heat * np.maximum(temp - hot_c, 0.0)
    d_cold = k_heat * np.maximum(cold_c - temp, 0.0)
    d_chem = k_chem * np.maximum(chem, 0.0)
    drive = d_mech + d_heat + d_cold + d_chem

    # A-delta reports the stimulus; C reports it lower, later and for longer.  the
    # persistence is applied by the caller across steps -- here C simply carries
    # the fraction and the compression that make second pain what it is.
    r_adelta = np.clip(drive, 0.0, r_max)
    r_c = np.clip(c_frac * r_max * np.tanh(drive / max(r_max, 1e-9)) / max(c_tau, 1e-6),
                  0.0, r_max)
    return {
        "transduction.nociceptor": np.clip(-70.0 + 0.5 * r_adelta, -70.0, 0.0),
        "neural.afferent.adelta": r_adelta,
        "neural.afferent.c": r_c,
    }


implementation(
    name="golgi_tendon_log",
    process="transduction",
    doc="""the Golgi tendon organ: force, logarithmically compressed.

    in series with the muscle rather than in parallel with it, which is the entire
    difference from a spindle.  the pair is what lets a controller separate
    kinematics from kinetics, and a model with only spindles will attribute every
    load change to a length change.""",
    form=Form.RATE,
    fn=golgi_tendon_afferent,
    params={
        "force_gain_hz": lognormal(40.0, 2.0, units="Hz per log unit",
                                   provenance=Provenance.LITERATURE),
        "force_scale_n": lognormal(1.0, 5.0, units="N",
                                   provenance=Provenance.WEAK,
                                   note="the force at which the log becomes linear; it "
                                        "scales with muscle size and is not one number "
                                        "for the body"),
        "bias_hz": lognormal(5.0, 2.0, units="Hz", provenance=Provenance.LITERATURE),
        "r_max_hz": lognormal(200.0, 1.5, units="Hz", provenance=Provenance.LITERATURE),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="Houk & Henneman; Crago et al. on the logarithmic force relation")

implementation(
    name="nociceptor_polymodal",
    process="transduction",
    doc="""the polymodal nociceptor: high threshold, two fibre classes, sensitising.

    `transduction.nociceptor` was declared on the field and had no component, so
    every high-threshold channel in the body was bound to
    `transduction.baroreceptor` and merely TAGGED nociceptive in its row.  A tag
    is not a transducer: nothing computed a rate from a noxious stimulus, and a
    reward signal grounded in damage had nothing to read.  This is that gap
    closed.

    The threshold is the physiology.  Below it the output is exactly zero, which
    is what separates a nociceptor from a sensitive mechanoreceptor and what makes
    its rate meaningful as a cost rather than as another touch channel.

    A-delta and C are written separately because the periphery already
    distinguishes them -- both are declared fibre classes with resolved
    velocities, and 48 of 72 trunks carry one or both -- and because the delay
    between fast and slow pain is functional, not incidental.

    Where it is weak: the thresholds are single numbers for the whole body, and
    real ones vary by tissue over a wide range; the chemical arm reads one lumped
    irritant concentration rather than the several species inflamed tissue
    actually produces; and sensitisation is a caller-advanced scalar rather than a
    model of the inflammatory cascade that drives it.""",
    form=Form.RATE,
    fn=nociceptor_afferent,
    params={
        "mechanical_threshold_n": lognormal(8.0, 2.0, units="N",
                                            provenance=Provenance.WEAK,
                                            note="tissue-damage threshold; varies "
                                                 "widely by tissue and is not one "
                                                 "number for the body"),
        "heat_threshold_c": normal(43.0, 1.5, units="degC",
                                   provenance=Provenance.LITERATURE,
                                   note="the classical heat-pain threshold, and "
                                        "close to the TRPV1 activation point"),
        "cold_threshold_c": normal(15.0, 3.0, units="degC",
                                   provenance=Provenance.LITERATURE),
        "mechanical_gain_hz_per_n": lognormal(1.6, 2.0, units="Hz per N",
                                              provenance=Provenance.WEAK),
        "thermal_gain_hz_per_c": lognormal(3.5, 2.0, units="Hz per degC",
                                           provenance=Provenance.LITERATURE),
        "chemical_gain_hz": lognormal(25.0, 3.0, units="Hz",
                                      provenance=Provenance.WEAK),
        "sensitisation": lognormal(1.0, 1.5, units="1",
                                   provenance=Provenance.WEAK,
                                   note="1.0 is naive tissue; above 1 lowers the "
                                        "thresholds, which is how an injury comes "
                                        "to be protected"),
        "sensitisation_shift_c": normal(4.0, 2.0, units="degC",
                                        provenance=Provenance.WEAK),
        "c_fibre_fraction": normal(0.7, 0.1, units="1",
                                   provenance=Provenance.LITERATURE),
        "c_persistence": normal(0.85, 0.1, units="1", provenance=Provenance.WEAK),
        "r_max_hz": lognormal(100.0, 1.5, units="Hz",
                              provenance=Provenance.LITERATURE),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="Sherrington on the high-threshold definition; Bessou & Perl on "
           "polymodal C endings; LaMotte & Campbell on first and second pain")
