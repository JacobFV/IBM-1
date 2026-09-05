"""instruments as state: electrodes, coils, transducers, scanners, displays.

the architecture is explicit that a measurement is not a special kind of thing.
an electrode contact voltage is a state variable with a value, an uncertainty and
a topology to brain state, exactly as a membrane potential is; it is a separate
field only because its support is separate -- a contact is not tissue.  the
consequence is that there is no `observe/` directory anywhere in ibm-1: electrode
coupling is an ordinary process, and the measurement likelihood is evidence
attached to the component it produces.

spectral, for two reasons that both matter.  first, an instrument *is* a filter:
an amplifier has a passband, an electrode-electrolyte interface is a constant-phase
element whose impedance falls as a power of frequency, a coil has a resonant
ringdown, and a display has a refresh rate.  a device component that could not
carry a frequency dependence would have to model each of those as a separate
scalar variable per band.  second, a device component's prior is a *noise model* --
a Johnson-Nyquist white floor plus the 1/f drift of a double layer -- and a noise
model is a spectrum by definition.  stating it here is what lets an observation
contribute honest precision at each frequency instead of one fitted fudge factor,
and it is why EEG evidence is informative about gamma and nearly useless about the
drift below 0.1 Hz without anyone having to special-case that.

the supports are the instrument classes, and they are separate because their
geometry is: a scalp montage, an implanted grid, a TMS coil, a receive coil and a
monitor have nothing in common as domains.  the field's default is `sensor_array`
-- the non-invasive case, the most common one -- and everything else overrides.

exogeneity is declared component by component and not by fiat.  a contact potential
is produced by a coupling process and is not exogenous; a coil current, a
transducer drive, a display luminance and a speaker pressure are set by the
experiment and nothing in the model writes them, so they are.  the two ambiguous
ones are marked exogenous and the reason is recorded on each: nothing in the
current process inventory writes electrode encapsulation or amplifier gain drift,
and declaring them exogenous makes that gap visible in the registry rather than
letting them be read from nowhere.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Component, Field
from ibm.vocabulary import Band, Provenance

FIELD = REGISTRY.field(Field(
    "device",
    "instrument state: the contact voltages and impedances a recording system carries, the "
    "currents and drives a stimulator delivers, the sequence state of a scanner, and the "
    "luminance and pressure a display and speaker put into the world",
    "sensor_array", Provenance.PHYSICS))

#: recording chain: DC-coupled at the bottom, above the spike band at the top.
RECORDING = Band(0.0, 20000.0)
#: stimulator transients: a TMS pulse rings at a few kilohertz, a DBS train at
#: hundreds of hertz with much faster edges.
STIMULATION = Band(0.0, 100000.0)


def _c(id_: str, doc: str, units: str, **kw) -> Component:
    kw.setdefault("uncertainty", "spectral")
    kw.setdefault("provenance", Provenance.PHYSICS)
    return REGISTRY.component(Component(id=id_, field="device", doc=doc, units=units, **kw))


CONTACT_POTENTIAL = _c(
    "device.contact_potential",
    "the voltage at a recording contact relative to the system reference: what the "
    "amplifier actually digitizes.  it is not `electromagnetic.potential` at the contact's "
    "position -- it is that field seen through the contact's spatial sensitivity, its "
    "impedance divider against the amplifier input, and the reference the montage uses, "
    "and every one of those is a device property rather than a head property.  keeping "
    "them separate is what makes a re-referencing choice a declared transformation rather "
    "than a silent one",
    "uV", band=RECORDING, prior="device_broadband", bounds=(-1e7, 1e7), timescale_s=1e-4,
    tags=frozenset({"recording", "observable"}))

IMPEDANCE = _c(
    "device.impedance",
    "the magnitude of the electrode-tissue interface impedance.  spectral in the most "
    "literal sense available: the interface is a constant-phase element, so |Z| falls "
    "roughly as f^-alpha over four decades and quoting it at one frequency is quoting a "
    "point on a curve.  it rises over weeks as glial encapsulation forms around a chronic "
    "implant, which makes it the most direct observable of `structural.gliosis`, and it "
    "sets both the thermal noise floor and the divider that attenuates the recorded "
    "signal.  on the implanted support, where the encapsulation story is; declared "
    "exogenous because no process in the current inventory writes it",
    "kohm", support="implanted_array", band=RECORDING, prior="aperiodic",
    bounds=(0.0, 1e5), timescale_s=86400.0, exogenous=True,
    tags=frozenset({"recording", "interface"}))

COIL_CURRENT = _c(
    "device.coil_current",
    "the current in a stimulation coil or the current delivered by a tES or DBS source.  "
    "the driving term of the whole stimulation chain: it induces the electric field that "
    "polarizes membranes, and it is the one quantity in that chain an experimenter sets "
    "directly and reports exactly.  its band reaches a hundred kilohertz because a "
    "biphasic TMS pulse is tens of microseconds wide, and the induced field depends on its "
    "time derivative -- so the pulse shape, not just its amplitude, is what the model needs",
    "A", support="stimulator", band=STIMULATION, prior="device_drive",
    bounds=(-2e4, 2e4), timescale_s=1e-6, exogenous=True,
    tags=frozenset({"stimulation", "drive"}))

TRANSDUCER_DRIVE = _c(
    "device.transducer_drive",
    "the acoustic pressure a focused-ultrasound transducer emits at its face, before any "
    "propagation.  separate from `mechanical.pressure` in the head because everything "
    "interesting happens between the two: skull attenuation and aberration can cost 90 per "
    "cent of the amplitude and displace the focus by millimetres, and a model that "
    "conflated the drive with the delivered dose would have assumed that away.  carried at "
    "the envelope rather than the carrier, which is why its band is the pulse-repetition "
    "band and not the megahertz one",
    "MPa", support="stimulator", band=Band(0.0, 10000.0), prior="device_drive",
    bounds=(-20.0, 20.0), timescale_s=1e-4, exogenous=True,
    tags=frozenset({"stimulation", "drive"}))

CHANNEL_GAIN = _c(
    "device.channel_gain",
    "the transfer of one recording channel from contact to sample: nominally a constant, "
    "actually a passband with a drifting gain.  it is declared because an observation's "
    "likelihood is written against the state the amplifier is in, not against an idealized "
    "one, and because gain drift and filter settings are the two most common reasons two "
    "recordings of the same phenomenon disagree.  its prior is the instrument-noise one "
    "with a baseline of unity: white measurement noise on a 1/f drift.  exogenous, because "
    "nothing in the model changes an amplifier's settings",
    "dimensionless", band=RECORDING, prior="device_broadband", bounds=(0.0, 1e6),
    timescale_s=3600.0, exogenous=True,
    tags=frozenset({"recording", "calibration"}))

SEQUENCE_PHASE = _c(
    "device.sequence_phase",
    "the phase of an MR pulse sequence: where in the TR the scanner currently is.  it is "
    "state and not a schedule because things depend on it that the model must be able to "
    "express -- slice timing, which decides when each voxel was actually sampled; the "
    "gradient switching that is the loudest sound in the experiment and an auditory "
    "stimulus in its own right; and the RF power deposition that enters "
    "`metabolic.heat`.  on the scanner support, exogenous, and the reason a simultaneous "
    "EEG-fMRI artifact is a predictable coupling rather than noise",
    "rad", support="scanner_element", band=STIMULATION, prior="device_drive",
    timescale_s=1e-5, exogenous=True,
    tags=frozenset({"scanner", "timing"}))

DISPLAY_LUMINANCE = _c(
    "device.display_luminance",
    "the luminance a display emits at a position on its surface.  this is where a visual "
    "stimulus physically is: not an input to the model, but state on the display support "
    "that an intervention clamps and that an ordinary transduction process couples to "
    "`transduction.photoreceptor`.  its band stops at a few hundred hertz because a "
    "display is sampled -- the refresh rate is a real property of the stimulus and is why "
    "a nominally steady grating drives a measurable response at the frame rate",
    "cd/m^2", support="display", band=Band(0.0, 500.0), prior="device_drive",
    bounds=(0.0, 1e4), timescale_s=8e-3, exogenous=True,
    tags=frozenset({"stimulus", "drive"}))

SPEAKER_PRESSURE = _c(
    "device.speaker_pressure",
    "the sound pressure a speaker or earphone produces.  the auditory counterpart of "
    "display luminance and, like it, exogenous state rather than an input: an experiment "
    "clamps it, and the path from it to `transduction.hair_cell` -- head-related transfer "
    "function, ear canal resonance, middle-ear transmission -- is a chain of ordinary "
    "processes.  its band runs to twenty kilohertz because the cochlea's does, and "
    "truncating it here would silently make high-frequency hearing unmodellable",
    "Pa", support="display", band=Band(0.0, 20000.0), prior="device_drive",
    bounds=(-200.0, 200.0), timescale_s=2.5e-5, exogenous=True,
    tags=frozenset({"stimulus", "drive"}))
