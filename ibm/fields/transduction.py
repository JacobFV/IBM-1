"""receptor state at the sensory epithelia: where a physical field becomes neural.

spectral throughout, and unlike the neural field the reason is not several rhythms
in one variable -- it is that every receptor in the inventory is explicitly a
band-pass device and its band is what it computes.  a photoreceptor low-passes at
tens of milliseconds and adapts over seconds, and the ratio of those two poles is
what makes vision a contrast sense rather than a luminance sense; a hair cell
transduces to twenty kilohertz and adapts in milliseconds, and that ratio is what
lets the cochlea hold sensitivity across 120 dB.  the transduction low-pass and
the adaptation high-pass live in the *same* variable and their product is the
receptor's transfer function, which is exactly the architecture's criterion for
spectral state.  the `receptor_transduction` prior is that product written down.

it is also the practical form.  a stimulus is an intervention on this field, and
almost every stimulus an experiment uses is specified spectrally -- a drifting
grating, a tone, a click train, a vibrotactile carrier, a flicker frequency.
clamping a spectral component with a spectral specification is a direct
substitution; clamping a scalar one would require simulating the waveform in the
time domain first, which is the cost the spectral form exists to avoid.

each component sits on its own receptor surface, because the supports genuinely
differ -- the retina is a surface in eye coordinates whose receptor density spans
two orders of magnitude, the cochlea is one-dimensional and logarithmic in
frequency, and the vestibular organ is six directionally tuned sensors and not a
surface at all.  a materialization asking for foveal vision and one asking for
tonotopy are asking for different geometry, not for different resolutions of the
same geometry.

one honest compromise, recorded here rather than papered over.  `adaptation` is a
single component on the field's default support, the skin, and every receptor
class in fact adapts.  the alternative -- one adaptation component per receptor
surface -- is eight components carrying the same physics, which is how an ontology
sprawls; and the fast, receptor-intrinsic adaptation of each class is already
inside that class's own spectral prior as the high-pass.  what this component adds
is the slow adaptation state that is separately identifiable on the body surface,
where rapidly- and slowly-adapting mechanoreceptors are distinguished by it and
where psychophysics can measure it.  photoreceptor light adaptation over ten log
units is not representable this way, and that is a known limitation of the current
inventory rather than a claim that it does not exist.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Band, Provenance

FIELD = REGISTRY.field(Field(
    "transduction",
    "receptor state at the sensory epithelia: the membrane potentials, transduction "
    "currents and occupancies by which light, sound, motion, deformation, temperature, "
    "chemistry and visceral state become drive to primary afferents",
    "body_surface", Provenance.LITERATURE))

VISUAL = Band(0.0, 100.0)
AUDITORY = Band(0.0, 20000.0)
TACTILE = Band(0.0, 1000.0)
SLOW = Band(0.0, 20.0)


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "spectral")
    kw.setdefault("provenance", Provenance.LITERATURE)
    kw.setdefault("prior", "receptor_transduction")
    return REGISTRY.component(
        Component(id=id_, field="transduction", doc=doc, units=units, **kw))


PHOTORECEPTOR = _c(
    "transduction.photoreceptor",
    "the graded membrane potential of rod and cone photoreceptors, taken to include the "
    "outer-plexiform processing that immediately follows it.  it hyperpolarizes with light "
    "-- the sign is inverted relative to every other receptor here, which matters because "
    "the process that reads it must not assume otherwise -- and its band-pass, a "
    "~30 ms integration against seconds of adaptation, is what turns an absolute luminance "
    "into the contrast signal the retina actually transmits.  on the retina because "
    "receptor density there varies a hundredfold from fovea to periphery and uniform "
    "sampling of it is always wrong",
    "mV", support="retina", band=VISUAL, bounds=(-80.0, -20.0), timescale_s=3e-2,
    tags=frozenset({"receptor", "visual"}))

HAIR_CELL = _c(
    "transduction.hair_cell",
    "the mechanoelectrical transduction current of cochlear inner and outer hair cells, "
    "driven by stereociliary deflection.  the widest band of any component in the ontology "
    "because it must phase-lock to the stimulus itself up to a few kilohertz and follow its "
    "envelope above that, and both are the same current.  on the cochlea, whose natural "
    "coordinate is tonotopic position rather than distance, so the basilar membrane's own "
    "frequency decomposition is carried by the support and not repeated in the state",
    "pA", support="cochlea", band=AUDITORY, bounds=(-2000.0, 2000.0), timescale_s=2e-5,
    tags=frozenset({"receptor", "auditory"}))

MECHANORECEPTOR = _c(
    "transduction.mechanoreceptor",
    "the receptor potential of cutaneous and proprioceptive mechanoreceptors: Merkel, "
    "Meissner, Pacinian and Ruffini endings on the skin, and muscle spindles and Golgi "
    "organs read on the same support.  one component rather than four because their "
    "difference is a transfer function -- Pacinian corpuscles are high-pass to a few "
    "hundred hertz, Merkel cells are nearly static -- and a transfer function is precisely "
    "what the spectral form parameterizes rather than what it needs separate variables for",
    "mV", band=TACTILE, bounds=(-80.0, 40.0), timescale_s=5e-3,
    tags=frozenset({"receptor", "somatosensory"}))

SPINDLE_PRIMARY = _c(
    "transduction.spindle_primary",
    "receptor potential of the muscle-spindle PRIMARY (annulospiral) ending, on "
    "intrafusal nuclear-bag and nuclear-chain fibres.  its response is dominated by "
    "the RATE OF CHANGE of muscle length, with a static component on top -- which is "
    "why it is separated from the secondary ending rather than scaled from it.  its "
    "gain is not a constant: fusimotor drive (`neural.efferent.gamma`) contracts the "
    "intrafusal poles and reloads the ending, so this receptor's sensitivity is under "
    "central control and the model must read gamma to compute it",
    "mV", band=TACTILE, bounds=(-80.0, 0.0), timescale_s=2e-3,
    support="motor_units",
    tags=frozenset({"receptor", "proprioceptive", "mechanical"}))

SPINDLE_SECONDARY = _c(
    "transduction.spindle_secondary",
    "receptor potential of the muscle-spindle SECONDARY (flower-spray) ending, "
    "chiefly on nuclear-chain fibres: a largely static, position-proportional signal "
    "with little velocity sensitivity.  it is the tonic limb-position signal that "
    "persists after the primary ending's dynamic response has adapted",
    "mV", band=SLOW, bounds=(-80.0, 0.0), timescale_s=1e-2,
    support="motor_units",
    tags=frozenset({"receptor", "proprioceptive", "mechanical"}))

GOLGI_TENDON = _c(
    "transduction.golgi_tendon",
    "receptor potential of the Golgi tendon organ, in SERIES with the muscle at the "
    "musculotendinous junction rather than in parallel with it.  the series/parallel "
    "distinction is the entire functional content: a spindle unloads when the muscle "
    "shortens against no load, while a tendon organ fires harder the harder the "
    "muscle pulls.  so this reports FORCE and the spindle reports LENGTH, and a "
    "controller with only one of them cannot separate a limb that moved from a limb "
    "that met resistance",
    "mV", band=TACTILE, bounds=(-80.0, 0.0), timescale_s=3e-3,
    support="motor_units",
    tags=frozenset({"receptor", "proprioceptive", "mechanical"}))

JOINT_RECEPTOR = _c(
    "transduction.joint_receptor",
    "receptor potential of joint-capsule and ligament afferents -- Ruffini-like "
    "endings, Golgi-type endings and free nerve endings in the capsule.  they signal "
    "extreme-of-range and capsular stress rather than mid-range position, which is "
    "why they do not substitute for spindles: a model that used joint receptors as "
    "its position sense would be blind through most of the working range",
    "mV", band=SLOW, bounds=(-80.0, 0.0), timescale_s=1e-2,
    support="body_surface",
    tags=frozenset({"receptor", "proprioceptive", "mechanical"}))

THERMORECEPTOR = _c(
    "transduction.thermoreceptor",
    "the receptor state of warm and cool cutaneous thermoreceptors, TRP-channel mediated.  "
    "it reads `thermal.temperature` at the skin, which is the only place in the inventory "
    "where the thermal field is sensed rather than merely constrained, and it is separate "
    "from the nociceptor because innocuous and noxious thermal signalling use different "
    "channels, different fibre classes and different central pathways -- and because the "
    "boundary between them is exactly what a heat-pain threshold measures",
    "mV", band=SLOW, bounds=(-80.0, 40.0), timescale_s=0.2,
    tags=frozenset({"receptor", "thermal"}))

NOCICEPTOR = _c(
    "transduction.nociceptor",
    "the state of polymodal nociceptive endings: high-threshold, slowly adapting, and "
    "sensitized rather than adapted by repeated stimulation.  that sign reversal is the "
    "reason it is a component and not a threshold applied to the mechanoreceptor -- a "
    "receptor whose gain goes up with use cannot be expressed as a nonlinearity on one "
    "whose gain goes down.  its slow band reflects the unmyelinated conduction and second- "
    "messenger sensitization that make pain a seconds-to-minutes signal",
    "mV", band=SLOW, bounds=(-80.0, 40.0), timescale_s=0.5,
    tags=frozenset({"receptor", "nociceptive"}))

CHEMORECEPTOR = _c(
    "transduction.chemoreceptor",
    "fractional receptor occupancy at olfactory and gustatory epithelia.  an occupancy "
    "rather than a potential because the natural coordinate here is receptor identity, not "
    "position -- a smell is a pattern across four hundred receptor types, and the "
    "epithelium's spatial arrangement is close to irrelevant to it -- which is why its "
    "support is declared discrete.  bounded in [0,1] because it is a fraction, and slow "
    "because binding and unbinding at these affinities take hundreds of milliseconds",
    "dimensionless", support="chemosensory_epithelium", band=Band(0.0, 10.0),
    bounds=(0.0, 1.0), timescale_s=0.3,
    tags=frozenset({"receptor", "chemosensory"}))

BARORECEPTOR = _c(
    "transduction.baroreceptor",
    "the state of arterial baroreceptors and, by extension, the visceral mechanoreceptors "
    "that report gut and bladder distension.  on the viscera support because that is where "
    "these endings are, and it is the field's interoceptive entry point: it reads "
    "`blood.pressure` and `mechanical.displacement` on visceral positions and is the reason "
    "the architecture can claim interoception is ordinary transduction rather than a "
    "special mechanism.  its band holds the cardiac cycle it is phase-locked to",
    "mV", support="viscera", band=Band(0.0, 50.0), bounds=(-80.0, 40.0), timescale_s=2e-2,
    tags=frozenset({"receptor", "interoceptive"}))

VESTIBULAR = _c(
    "transduction.vestibular",
    "the transduction state of semicircular-canal and otolith hair cells.  separate from "
    "`transduction.hair_cell` despite sharing a transduction mechanism because the "
    "mechanics in front of it are completely different: a canal integrates angular "
    "acceleration into cupular deflection with a several-second time constant, and an "
    "otolith responds to linear acceleration including gravity, which is why the organ is "
    "six directionally tuned sensors rather than a surface.  its bidirectional resting "
    "discharge is why the variable is signed",
    "mV", support="vestibular_organ", band=Band(0.0, 50.0), bounds=(-80.0, 40.0),
    timescale_s=5e-3, tags=frozenset({"receptor", "vestibular"}))

ADAPTATION = _c(
    "transduction.adaptation",
    "the slow adaptation state of a receptor population: the multiplicative gain that "
    "falls with sustained stimulation and recovers over seconds, declared once on the "
    "field's default support.  it is what keeps a receptor from saturating on the mean of "
    "its input, and it is a separate component rather than a parameter because it is state "
    "-- it carries the history that makes the response to a step different at its onset "
    "and at its end, and an aftereffect is nothing but this variable failing to have "
    "recovered yet",
    "dimensionless", band=Band(0.0, 5.0), bounds=(0.0, 1.0), timescale_s=2.0,
    tags=frozenset({"receptor", "adaptation", "slow"}))
