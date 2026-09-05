"""driving early sensory state from a continuous naturalistic stimulus.

this file exists because a stationary spectrum is a weak constraint on dynamics.
a resting power spectrum pins the *marginal* second moment of a component and
says nothing about its response to anything; two models with completely different
transfer functions produce the same 1/f slope and the same alpha bump.  a
continuous stimulus-brain pair is a different kind of evidence: the stimulus is
known at every instant, it is not stationary, and the brain's answer to it is
measured at every instant, so what is constrained is the *map* rather than the
marginal.  that is the bandwidth argument for this whole module, and it is the
only reason to pay its cost.

**nothing here is a new primitive, and nothing here is a new state variable.**
that is the first thing to check about a file like this one, because the obvious
way to write it is the wrong way: take a pretrained stimulus encoder, call its
hidden activations "early sensory state", and let them into the state graph.
ARCHITECTURE.md §1 forbids it in as many words -- a latent is an encoding, not a
physical quantity, nothing exerts pressure on it, and latents live inside a
process's f and are invisible to the state graph.  so what is declared below is
two more candidate forms of f for the *already declared* `transduction` process,
whose outputs are the already declared `transduction.hair_cell` and
`transduction.photoreceptor`.  the filterbank is inside f.  it writes a
mechanoelectrical transduction current at a tonotopic place, which is a physical
quantity with units, and if the front end were swapped for a better one the
declaration would not change -- which is the test §4 sets for whether an f is in
the right place.

**the stimulus is already state.**  `device.speaker_pressure` and
`device.display_luminance` are declared, exogenous, on the display support, and
`ibm.processes.intervention` clamps them.  so the path from a wav file to cortex
does not enter the model through a side channel; it enters through a clamp on
declared state and then through `transduction`, `afferent_propagation` and
`thalamocortical_coupling`, which are the same processes that run when nobody is
playing anything.  `FORCING_CHAIN` below writes that path down and checks every
link against the registry at import, so a typo in it is a declaration error
rather than a coupling that silently does nothing.

**forcing is evidence, not a clamp.**  this is the part that is easy to get wrong
and expensive to get wrong.  a front end that writes `transduction.hair_cell`
is a teacher: it supplies a value where no measurement exists, and §4 says a
teacher's value carries a precision calibrated from its measured accuracy on the
variable it is writing.  nobody has ever measured a hair cell current in a MEG
scanner, so the accuracy that *can* be measured is downstream: how much of the
variance of measured brain state the forced chain accounts for, on held-out data.
`ForcingCalibration` measures exactly that, per band, and hands it to
`ibm.runtime.fuse.TeacherPrecision`, which applies the two mandatory corrections
-- off-distribution inflation with a prior, and a rank-1 shared-error discount --
and returns an `Evidence` whose low-rank part is *negative*, because a teacher's
correlated error subtracts confidence rather than adding it.

a consequence worth stating up front, because it is the least intuitive thing
here.  at a *linear* readout, a calibrated precision does not change the
posterior mean's shape by a scalar -- the shrinkage factor is absorbed by any
fitted downstream gain.  what it changes is (a) the relative weighting between
frequency bands, since the measured r^2 is not flat in frequency and the
shrinkage therefore is not either, and (b) the predictive variance, which is the
whole point: a driven model that reports its own uncertainty as though the
forcing were a measurement is exactly the failure §4's distillation section
describes.  so the band-resolved calibration is doing real work on the mean and
the low-rank discount is doing real work on the variance, and neither is
decoration.

**provenance has to survive the whole thing.**  a component whose value came from
a stimulus front end and a component whose value came from the model's own
dynamics are not the same kind of claim, and a result reached by driving early
cortex must never be reported as emergent.  `ForcingRecord` records FORCED versus
EVOLVED per component, together with the r^2 that earned the forcing its
precision, and renders as notes that attach to
`ibm.materialize.provenance.Provenance` without that module needing to know this
one exists.

what is *not* here, recorded rather than glossed:

- **no pretrained teacher is used.**  `data/sources/tribe/card.yaml` and
  `tribe-v2` are declared and their `raw/.location.yaml` sets `local_root: null`
  -- no weights are held on this machine, so no pretrained stimulus encoder was
  available and none is pretended.  what runs is the analytic front end below.
- **the visual front end is declared and has not been run on brain data.**  the
  only held source with a visual naturalistic stream and localized electrodes is
  `ds003688`, and its `stimuli/` directory holds annotation tables only -- the
  film itself is not in the deposit.  so `RetinalFrontEnd` is written, is
  physically argued, and carries no measured r^2.  its calibration is therefore
  refused rather than guessed, which is what `ForcingCalibration.unmeasured`
  exists to express.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Iterable, Mapping, Sequence

import numpy as np

from ibm.processes.base import (
    alpha_synapse,
    delay_dispersion,
    implementation,
    low_pass,
    series,
)
from ibm.registry import REGISTRY, Form
from ibm.vocabulary import Band, Prior, Provenance, Tying, lognormal, normal, uniform, weak

_EPS = 1e-30


# ---------------------------------------------------------------------------
# the chain, written down
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Link:
    """one step of the path from a clamped stimulus to a brain state variable.

    a link names a registered thing and the component it moves the signal into.
    it is a declaration and not a computation: the arithmetic lives in the
    processes' own implementations, and the only thing this object adds is that
    the path is written in one place and checked, so "the stimulus reaches cortex
    through the declared graph" is a statement with a test attached rather than a
    claim in a docstring.
    """

    kind: str                       # intervention | process
    id: str
    into: str                       # the registered component this link writes
    note: str = ""

    def __str__(self) -> str:
        return f"{self.kind} {self.id} -> {self.into}"


#: the auditory forcing path.  every entry is checked against the registry by
#: `check_chain()` at import.
AUDITORY_CHAIN: tuple[Link, ...] = (
    Link("intervention", "naturalistic_stream", "device.speaker_pressure",
         "the experimenter clamps a speaker.  this is where a wav file physically is"),
    Link("process", "transduction", "transduction.hair_cell",
         "the cochlea: a tonotopic resonance into a sub-millisecond transduction channel.  "
         "`gammatone_cochleagram` below is the f that gets evaluated on a real waveform"),
    Link("process", "transduction", "neural.afferent.activity",
         "spiral ganglion firing.  a separate output from the receptor potential because "
         "spike initiation and rate saturation happen at the ganglion cell, not the hair cell"),
    Link("process", "afferent_propagation", "neural.exc.ampa",
         "the dispersed peripheral delay into a thalamic relay synapse.  the dispersion is "
         "the part that matters: a velocity spread is a low-pass, and it smooths a click "
         "before any synapse does"),
    Link("process", "afferent_propagation", "neural.exc.activity",
         "medial geniculate relay output"),
    Link("process", "thalamocortical_coupling", "neural.exc.ampa",
         "granular-layer synaptic drive in primary auditory cortex, through a loop closed "
         "around its own delay -- which is what puts a resonance in the band an evoked "
         "response rides on.  this process writes a *conductance* and not a firing rate, "
         "and the distinction is why the next link is needed rather than optional"),
    Link("process", "local_excitation", "neural.exc.activity",
         "the cortical population's own answer to that drive.  the first link whose "
         "dynamics are recurrent rather than feed-forward, and therefore the first place "
         "the forcing stops determining the answer by itself"),
    Link("process", "local_excitation", "neural.transmembrane_current",
         "the current that actually sources a field.  a firing rate does not; a "
         "transmembrane current does, and conflating them is how a forward model acquires "
         "an unexplained gain"),
    Link("process", "em_generation", "electromagnetic.bfield",
         "oriented cortical current into the field a SQUID sees.  the last link, and the "
         "one whose parameters are least known here: no subject head model is held for "
         "meg-masc, so the lead field is fitted rather than computed"),
)

#: the visual forcing path.  declared, and not exercised on brain data -- see the
#: module docstring on ds003688's missing stimulus.
VISUAL_CHAIN: tuple[Link, ...] = (
    Link("intervention", "naturalistic_stream", "device.display_luminance",
         "a screen.  the only declared carrier of photic drive in the contract"),
    Link("process", "transduction", "transduction.photoreceptor",
         "the phototransduction cascade with the eye's optics absorbed into a gain.  the "
         "absorbed part is a real loss and is named in `ibm.processes.transduction`"),
    Link("process", "transduction", "neural.afferent.activity",
         "retinal ganglion firing"),
    Link("process", "afferent_propagation", "neural.exc.activity",
         "the optic pathway into LGN"),
    Link("process", "thalamocortical_coupling", "neural.exc.ampa",
         "geniculostriate drive into layer IV of V1"),
    Link("process", "local_excitation", "neural.exc.activity",
         "V1's own answer to that drive"),
    Link("process", "em_generation", "electromagnetic.bfield",
         "or `device_coupling` for an intracranial contact, which is the modality "
         "ds003688 would have supplied had the film been deposited with it"),
)

FORCING_CHAIN: tuple[Link, ...] = AUDITORY_CHAIN + VISUAL_CHAIN


def check_chain(chain: Sequence[Link] = FORCING_CHAIN) -> list[str]:
    """every link names a registered thing, and writes a component that thing writes.

    the check is deliberately stricter than "the id exists".  a link that named a
    real process but a component that process does not write would describe a
    path the graph does not have, and would do so convincingly.  so the process
    case verifies the component appears in the process's own output selectors,
    which is the registry's own statement of what it writes.
    """
    problems: list[str] = []
    for link in chain:
        if link.into not in REGISTRY.components and link.into not in REGISTRY._alias:
            problems.append(f"{link}: writes unregistered component {link.into!r}")
        if link.kind == "process":
            p = REGISTRY.processes.get(link.id)
            if p is None:
                problems.append(f"{link}: unregistered process {link.id!r}")
            elif not any(link.into in s.vars for s in p.outputs):
                problems.append(f"{link}: process {link.id!r} does not write {link.into!r}")
        elif link.kind == "intervention":
            iv = REGISTRY.interventions.get(link.id)
            if iv is None:
                problems.append(f"{link}: unregistered intervention {link.id!r}")
            elif link.into not in iv.constrains.vars:
                problems.append(f"{link}: intervention {link.id!r} does not clamp {link.into!r}")
        else:
            problems.append(f"{link}: unknown link kind {link.kind!r}")
    return problems


def describe_chain(chain: Sequence[Link] = AUDITORY_CHAIN) -> str:
    return "\n".join(f"  {i}. {l}\n       {l.note}" for i, l in enumerate(chain, 1))


# ---------------------------------------------------------------------------
# front ends: f for `transduction`, evaluated on a real stimulus stream
# ---------------------------------------------------------------------------


def erb_hz(f_hz: np.ndarray | float) -> np.ndarray:
    """glasberg & moore's equivalent rectangular bandwidth of the auditory filter.

    the width of the cochlear filter at a given place, measured psychophysically
    and by notched-noise masking.  it is what makes a filterbank a *cochlear*
    filterbank rather than a constant-Q one: the bandwidth is neither constant nor
    proportional to frequency, and using either instead changes how much spectral
    detail survives to the auditory nerve.
    """
    return 24.7 * (4.37 * np.asarray(f_hz, float) / 1000.0 + 1.0)


def greenwood_places(n: int, lo_hz: float = 80.0, hi_hz: float = 8000.0) -> np.ndarray:
    """characteristic frequencies of `n` equally spaced positions on the cochlea.

    the spacing is uniform in *position along the basilar membrane*, not in
    frequency and not in log frequency, because position is what the support is.
    greenwood's map for the human cochlea is `f = A (10^(a x) - k)` with the
    constants below, so equal steps in x give the cf list this returns; a
    log-spaced bank would over-sample the apex and under-sample the base, which
    is the same error as sampling the retina uniformly.
    """
    a, k = 165.4, 0.88

    def x_of(f: float) -> float:
        return float(np.log10(f / a + k) / 2.1)

    xs = np.linspace(x_of(lo_hz), x_of(hi_hz), int(n))
    return a * (10.0 ** (2.1 * xs) - k)


@dataclass(frozen=True)
class CochlearFrontEnd:
    """a gammatone bank with rectification and compression: `transduction.hair_cell`.

    the physical claim, stated so it can be disagreed with.  a place on the
    basilar membrane is a resonator whose bandwidth is that place's ERB; the
    stereociliary channel rectifies, because a tip link can be pulled open and
    cannot be pushed further shut; and the transduction current is a compressive
    function of deflection, because the cochlear amplifier's gain falls with
    level.  those three facts are the whole front end, and each of them is
    already named in `hair_cell_cochlear`'s parameter block -- `q`,
    `tau_transduction_s`, `compression_exponent` -- which is why this is a second
    f for that same declaration rather than something new.

    what it adds over the declared LTI form is precisely the two nonlinearities
    that form's docstring admits it cannot carry.  rectification is what makes
    the output an envelope with a real DC term, which is what a cortical response
    to speech actually follows; compression is what keeps a 60 dB range of
    natural speech inside the ~20 dB range of a receptor.  a linear resonator
    bank produces neither, and a downstream fit against it is fitting around the
    front end's missing nonlinearity.

    what it still does not have, and this is not a small list: no two-tone
    suppression, no distortion products, no level-dependent tuning, no efferent
    (medial olivocochlear) gain control, and no adaptation -- the last of which is
    the one that most plausibly matters for continuous speech, since
    `transduction.adaptation` is a declared component this f does not write.

    the output units are declared pA and are honestly arbitrary in scale: the
    absolute current depends on stapes gain, ear canal resonance and the
    head-related transfer function, none of which is held for these datasets, so
    the level is normalized and the missing factor is exactly the kind of thing
    `optical_gain` is on the visual side.  scale is not identifiable here and is
    absorbed downstream.
    """

    n_places: int = 28
    lo_hz: float = 80.0
    hi_hz: float = 8000.0
    #: gammatone order.  four is the standard fit to the human auditory filter
    #: shape; the order sets the skirt slope, and the skirts are what decide how
    #: much of a neighbouring formant leaks into a place.
    order: int = 4
    #: bandwidth scale on the ERB.  1.019 is patterson's value for order 4.
    b: float = 1.019
    #: `compression_exponent` from `hair_cell_cochlear`: basilar membrane
    #: input-output slope at CF, measured at 0.2-0.4.
    compression: float = 0.3
    out_fs_hz: float = 100.0

    @property
    def cf_hz(self) -> np.ndarray:
        return greenwood_places(self.n_places, self.lo_hz, self.hi_hz)

    def __call__(self, x: np.ndarray, fs_hz: float) -> np.ndarray:
        """a pressure waveform to `(n_places, T)` transduction current at `out_fs_hz`.

        the filter is run as a cascade of `order` identical one-pole *complex*
        sections, which is the analytic gammatone: the pole is at
        `exp(-2 pi b ERB / fs + 2 pi i cf / fs)`, so the real part of the output is
        the gammatone response and its modulus is the envelope, with no separate
        rectifier or smoother needed.  taking the modulus is the rectification --
        it is the half-wave-rectified response's envelope up to a factor of two --
        and doing it this way costs one complex filter instead of a filter, a
        rectifier and a low-pass, which matters when the bank is run over hours of
        audio.
        """
        from scipy.signal import lfilter, resample_poly

        x = np.asarray(x, float)
        if x.ndim > 1:
            x = x.mean(axis=tuple(range(x.ndim - 1)))       # mono: both ears hear it
        peak = float(np.max(np.abs(x))) or 1.0
        x = x / peak
        up, down = _ratio(self.out_fs_hz, fs_hz)
        out = []
        for cf in self.cf_hz:
            bw = 2.0 * np.pi * self.b * float(erb_hz(cf)) / fs_hz
            p = np.exp(-bw + 2j * np.pi * cf / fs_hz)
            y = x.astype(np.complex128)
            for _ in range(max(int(self.order), 1)):
                y = lfilter([1.0 - abs(p)], [1.0, -p], y)
            env = resample_poly(np.abs(y), up, down)
            out.append(np.maximum(env, 0.0) ** self.compression)
        m = min(len(o) for o in out)
        return np.stack([o[:m] for o in out]).astype(np.float32)


@dataclass(frozen=True)
class RetinalFrontEnd:
    """luminance, contrast and motion energy: `transduction.photoreceptor`.

    declared and, on the data held here, not run against brain state -- the one
    source with a naturalistic film and localized electrodes deposits its
    annotations and not its film.  it is written anyway because the auditory arm
    would otherwise be indistinguishable from a speech-envelope encoding model,
    and the claim this whole module makes is about the *shape* of the path rather
    than about audio.

    three stages, and each of them is a physical quantity rather than a feature.

    *luminance to receptor potential.*  a cascade of first-order lags, exactly
    `photoreceptor_transfer`, applied per retinotopic position after pooling the
    display over that position's receptive aperture.  the aperture grows with
    eccentricity because receptor density falls by two orders of magnitude from
    fovea to periphery, and pooling uniformly instead is the same error as a
    log-spaced cochlear filterbank.

    *contrast.*  the photoreceptor's band-pass -- a ~30 ms integration against
    seconds of adaptation -- is what makes vision a contrast sense rather than a
    luminance sense, and it is already in the declared component's own docstring.
    here it appears as a divisive normalization by a slow local mean, which is the
    honest form: the receptor's gain falls roughly as the inverse of the
    background, and a subtractive high-pass gets the sign right and the scaling
    wrong across ten log units of light.

    *motion energy.*  a quadrature pair of spatiotemporal filters whose squared
    sum is direction-selective power.  this one is the least defensible as
    *receptor* state: motion energy is computed downstream of the photoreceptor,
    in the retina and LGN and V1, not in the outer segment.  it is included as a
    separate output channel with that caveat attached, because the alternative --
    handing a downstream model a static luminance sequence -- would understate the
    front end by more than this overstates it.  a materialization that cares
    should route it through `afferent_propagation`'s learned residual instead,
    which is where a non-trivial pathway transformation belongs.
    """

    n_positions: int = 16                 # retinotopic aperture count per axis
    tau_s: float = 0.03
    n_stages: int = 4
    adaptation_tau_s: float = 2.0
    out_fs_hz: float = 100.0
    directions: int = 4

    def __call__(self, frames: np.ndarray, fs_hz: float) -> np.ndarray:
        """`(T, H, W)` luminance to `(channels, T)` receptor state at `out_fs_hz`.

        returns luminance, contrast and motion-energy channels stacked, in that
        order, so a caller that wants only the first two can slice rather than
        re-run.  no brain data held on this machine can be used to calibrate it;
        `ForcingCalibration.unmeasured` is what that fact turns into.
        """
        from scipy.signal import lfilter, resample_poly

        f = np.asarray(frames, float)
        if f.ndim != 3:
            raise ValueError(f"expected (T, H, W) luminance, got shape {f.shape}")
        t, h, w = f.shape
        n = int(self.n_positions)
        # pool onto retinotopic apertures.  a plain block mean, because without a
        # gaze trace there is no fixation point and a foveated pooling would be
        # asserting a fixation nobody measured.
        hs, ws = max(h // n, 1), max(w // n, 1)
        pooled = f[:, : hs * n, : ws * n].reshape(t, n, hs, n, ws).mean((2, 4))
        pooled = pooled.reshape(t, n * n)

        a = np.exp(-1.0 / (self.tau_s * fs_hz))
        y = pooled
        for _ in range(max(int(self.n_stages), 1)):
            y = lfilter([1.0 - a], [1.0, -a], y, axis=0)
        b = np.exp(-1.0 / (self.adaptation_tau_s * fs_hz))
        background = lfilter([1.0 - b], [1.0, -b], pooled, axis=0)
        contrast = (y - background) / np.maximum(background, 1e-3)

        # motion energy: a quadrature pair in time against a spatial gradient in
        # each of `directions` directions.  crude, and named as crude.
        grad = np.diff(y, axis=0, prepend=y[:1])
        motion = []
        for d in range(max(int(self.directions), 1)):
            phase = 2.0 * np.pi * d / max(int(self.directions), 1)
            q = np.cos(phase) * grad + np.sin(phase) * (y - background)
            motion.append((q ** 2).reshape(t, n, n).mean((1, 2)))
        motion_arr = np.stack(motion, axis=1)

        stack = np.concatenate([y, contrast, motion_arr], axis=1).T
        up, down = _ratio(self.out_fs_hz, fs_hz)
        return np.stack([resample_poly(c, up, down) for c in stack]).astype(np.float32)


def _ratio(out_fs: float, in_fs: float) -> tuple[int, int]:
    """integer up/down factors for a resample, without a float sample rate anywhere.

    `resample_poly` with a rounded ratio silently changes the output's sample rate
    by a fraction of a percent, which over a 400 s recording is several hundred
    milliseconds of drift against a stimulus -- larger than every latency this
    module is trying to estimate.  so the ratio is exact or it is an error.
    """
    from math import gcd

    a, b = int(round(out_fs * 100)), int(round(in_fs * 100))
    if abs(out_fs * 100 - a) > 1e-6 or abs(in_fs * 100 - b) > 1e-6:
        raise ValueError(f"sample rates {in_fs} -> {out_fs} are not expressible as an exact "
                         "integer ratio at 0.01 Hz resolution; resampling would drift")
    g = gcd(a, b)
    return a // g, b // g


# ---------------------------------------------------------------------------
# the two implementations these front ends are
# ---------------------------------------------------------------------------

implementation(
    name="gammatone_cochleagram",
    process="transduction",
    doc="""the cochlea as a rectifying, compressive gammatone bank, evaluated on a
    real acoustic waveform.

    the same (I, O, T) as `hair_cell_cochlear` and a different f, which is exactly
    the swap §4 says a declaration should survive.  what it buys is the two things
    the LTI form's own docstring says it cannot do: rectification, so the output
    has an envelope a cortical response can follow, and compression, so a 60 dB
    speech range fits a receptor's 20 dB one.  what it costs is that it is no
    longer diagonal in the temporal laplacian basis -- a pointwise nonlinearity is
    dense in frequency -- so it must be evaluated in the time domain and
    re-analyzed, at the price §1 names.

    `Form.RATE` and not `Form.LEARNED` on purpose.  every parameter below is a
    measured property of the auditory periphery with a literature prior, and
    nothing in it is fitted to brain data; the fitting happens downstream, where
    the unknowns actually are.  a front end that had free parameters fitted
    against MEG would be an encoder wearing a cochlea's name, and the distinction
    between the two is the whole subject of this module.

    where it breaks, beyond the list in `CochlearFrontEnd`: it does not write
    `transduction.adaptation`, which the process declares as both an input and an
    output.  continuous speech has a strongly non-stationary level and a receptor
    bank with no gain control will over-represent loud passages and
    under-represent quiet ones.  that is a known, directional error and it is the
    single most likely reason a measured r^2 here understates what the periphery
    actually delivers.""",
    form=Form.RATE,
    params={
        "n_places": uniform(16.0, 64.0, units="positions",
                            provenance=Provenance.LITERATURE,
                            note="tonotopic sampling of the basilar membrane.  the auditory "
                                 "nerve has ~3000 inner hair cells; 28 places is a coarse "
                                 "sampling chosen so a downstream fit has fewer predictors "
                                 "than a recording has independent samples, and it is a "
                                 "materialization choice rather than a physiological one"),
        "erb_scale": normal(1.019, 0.1, units="dimensionless",
                            provenance=Provenance.LITERATURE,
                            source="patterson et al 1987, gammatone order-4 fit",
                            note="bandwidth as a multiple of the ERB.  it sets how much of a "
                                 "neighbouring formant reaches a place, and it is the "
                                 "parameter a hearing loss would move"),
        "gammatone_order": uniform(2.0, 6.0, units="stages",
                                   provenance=Provenance.LITERATURE,
                                   note="skirt slope of the auditory filter; four is the "
                                        "standard fit and higher orders stop improving it"),
        "compression_exponent": normal(0.3, 0.1, units="dimensionless",
                                       provenance=Provenance.LITERATURE,
                                       source="ruggero et al 1997 basilar membrane i/o slope",
                                       note="the same parameter `hair_cell_cochlear` declares "
                                            "and does not use.  here it is used, which is most "
                                            "of the reason this f exists"),
        "level_gain": weak(1.0, 30.0, units="pA per unit normalized pressure",
                           note="absorbs ear canal resonance, middle ear transmission and the "
                                "head-related transfer function, none of which is held for any "
                                "source in data/sources with a naturalistic audio stream.  it "
                                "is not separately identifiable from anything downstream and "
                                "is the auditory counterpart of `optical_gain`"),
    },
    tying=Tying.PER_SITE,
    provenance=Provenance.LITERATURE,
    source="patterson et al 1987; greenwood 1990; ruggero et al 1997",
)

implementation(
    name="luminance_motion_energy",
    process="transduction",
    doc="""a photoreceptor cascade with divisive light adaptation, plus a motion
    energy channel, evaluated on a real image sequence.

    the visual counterpart of `gammatone_cochleagram` and, unlike it, uncalibrated
    -- no source held on this machine pairs a naturalistic film with brain state
    *and* deposits the film.  ds003688 is the one that would: iEEG with electrode
    localization during an audiovisual film, 63 participants.  its `stimuli/`
    directory contains thirteen sound annotation tables and thirty video
    annotation tables and no media, because the film is copyrighted.  an
    annotation is a description of the stimulus, not the stimulus, and binding one
    as though it were `device.display_luminance` would hand the model an event
    time a human typed.

    so this declaration is honest about being a declaration.  the r^2 that would
    earn it a precision has not been measured, `ForcingCalibration.unmeasured`
    refuses to invent one, and anything downstream of it stays at its prior.

    the divisive adaptation is the one genuine improvement over
    `photoreceptor_cascade`.  that form's own docstring says its worst error is
    that gain does not fall with background luminance; here it does, as a division
    by a slow local mean, which is weber's law written as an operation instead of
    as an unused `weber_exponent` parameter.  the motion-energy channel is flagged
    as anatomically misplaced in `RetinalFrontEnd` and that flag is not a
    formality -- motion energy is computed downstream of the receptor, and putting
    it here overstates what an outer segment does.""",
    form=Form.RATE,
    params={
        "tau_s": lognormal(0.03, 2.2, units="s", provenance=Provenance.LITERATURE,
                           note="per-stage integration; the same prior "
                                "`photoreceptor_cascade` carries, and for the same reason"),
        "n_stages": uniform(2.0, 6.0, units="stages", provenance=Provenance.LITERATURE),
        "adaptation_tau_s": lognormal(2.0, 3.0, units="s", provenance=Provenance.LITERATURE,
                                      source="light adaptation recovery over seconds",
                                      note="the divisor's time constant.  it is what makes this "
                                           "a contrast front end rather than a luminance one"),
        "aperture_count": uniform(8.0, 64.0, units="positions per axis",
                                  note="retinotopic pooling.  uniform, because no gaze trace is "
                                       "held for any candidate source, and a foveated pooling "
                                       "without a fixation point asserts a fixation nobody "
                                       "measured"),
        "motion_directions": uniform(2.0, 8.0, units="directions",
                                     provenance=Provenance.WEAK,
                                     note="direction-selective channels.  declared at a weak "
                                          "prior because the channel itself is anatomically "
                                          "misplaced -- see the docstring"),
        "optical_gain": weak(1.0, 30.0, units="receptor mV per cd/m^2",
                             note="the same unidentifiable optics gain "
                                  "`photoreceptor_cascade` carries"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.WEAK,
    source="baylor et al 1984; adelson & bergen 1985",
)


# ---------------------------------------------------------------------------
# the downstream chain, as declared transfer functions
# ---------------------------------------------------------------------------


def chain_taps(basis, *, mean_delay_s: float = 0.015, sd_delay_s: float = 0.005,
               tau_synapse_s: float = 2e-3, tau_nmda_s: float = 0.1,
               tau_relay_s: float = 0.012, tc_delay_s: float = 5e-3,
               ct_delay_s: float = 8e-3, tau_trn_gaba_b_s: float = 0.15,
               loop_gain: float = 2.0, tau_cortex_s: float = 0.02) -> dict[str, np.ndarray]:
    """the declared components the forcing reaches, each as its own H(omega).

    this is the whole downstream model, and its shape is the claim being tested.
    a stimulus-to-brain encoder learns one free filter per sensor per stimulus
    feature; what is here instead is *four* filters, shared across every sensor
    and every participant, whose form comes from the process declarations and
    whose parameters are the time constants those declarations already carry.
    everything sensor-specific is left to the last link -- the lead field -- which
    is instantaneous, because a quasi-static field is an algebraic function of its
    source currents (§4).

    the four taps are not a basis chosen for expressiveness.  they are the four
    components the chain in `AUDITORY_CHAIN` actually writes, with the bands the
    declarations give them:

    - `neural.exc.ampa` at thalamus: the dispersed peripheral delay into a fast
      relay synapse.  `afferent_pathway_transfer`, unchanged.
    - `neural.exc.nmda` at thalamus: the same drive through a ~100 ms conductance.
      `afferent_propagation` declares this output only to 50 Hz, and the extra
      low-pass here is that declaration rather than a fitted smoother.
    - `neural.exc.ampa` at cortical layer IV: the relay output through
      `thalamocortical_loop_transfer` -- the *declared* form, feedback and all,
      imported rather than re-derived.  it is the only tap with a resonance in it,
      and the resonance is not decoration: the loop is closed around a 15-25 ms
      round trip, which is what makes a cortical response to an acoustic
      transient biphasic rather than a smoothed copy of the input.  a chain
      assembled from low-passes alone can only ever produce single-sided kernels,
      and a single-sided kernel cannot express a polarity reversal at a fixed
      latency, which is most of what an auditory evoked field is.
    - `neural.exc.activity` at cortex: the same, through one more cortical
      membrane, which is what a laminar step adds and is why a cortical population
      response is broader than the synaptic drive that caused it.

    a fifth thing that could be here and deliberately is not: a per-tap gain.  the
    gains are the lead field's and belong to the fit, so putting one here would
    make the same number appear twice and neither of them identifiable.
    """
    from ibm.processes.neural import thalamocortical_loop_transfer

    afferent = series(delay_dispersion(basis, mean_delay_s, sd_delay_s),
                      alpha_synapse(basis, tau_synapse_s))
    loop = thalamocortical_loop_transfer(
        basis, tc_delay_s=tc_delay_s, ct_delay_s=ct_delay_s, tau_relay_s=tau_relay_s,
        tau_trn_gaba_b_s=tau_trn_gaba_b_s, loop_gain=loop_gain, tau_ampa_s=tau_synapse_s)
    cortical = series(afferent, loop)
    return {
        "neural.exc.ampa@thalamus": afferent,
        "neural.exc.nmda@thalamus": series(afferent, low_pass(basis, tau_nmda_s)),
        "neural.exc.ampa@cortex_L4": cortical,
        "neural.exc.activity@cortex": series(cortical, low_pass(basis, tau_cortex_s)),
    }


# ---------------------------------------------------------------------------
# calibration: what the forcing is worth, measured on brain data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandFit:
    """the fraction of measured brain variance the forced chain explains in one band.

    `r2` is out-of-sample by construction of the caller and is stored beside the
    thing that makes it interpretable: `n_targets` (how many measured channels it
    is an average over), `n_samples` (how much held-out data), and `held_out`
    (what was held out -- segments, participants, or both).  an r^2 without those
    three is not a calibration, it is a number.
    """

    band: Band
    r2: float
    n_targets: int = 0
    n_samples: int = 0
    held_out: str = ""

    def __str__(self) -> str:
        return (f"{self.band}: r2={self.r2:+.4f} over {self.n_targets} channels, "
                f"{self.n_samples:,} held-out samples ({self.held_out or 'unspecified split'})")


@dataclass(frozen=True)
class ForcingCalibration:
    """measured r^2 per band, and the precision it earns the forcing.

    the object that keeps §4's rule enforceable rather than aspirational.  a
    forcing may not be applied unless this exists, and this may not exist unless
    somebody measured an r^2 on brain data -- which is why `unmeasured` is a
    constructor and not a default.

    the r^2 is measured *downstream*.  that needs saying plainly, because it is a
    weaker statement than the formula in §4 assumes.  §4's `dJ = 1/((1-r2)Var[x])`
    wants the r^2 of the teacher on the variable it is writing, and here the
    variable it is writing is a hair cell current nobody has ever measured in a
    scanner.  what is measurable is how much of the *measured* brain state the
    forced chain accounts for, and that is an r^2 on a different variable, several
    processes downstream, through an unknown lead field.  it is a lower bound on
    the front end's own fidelity and an upper bound on what the chain as a whole
    delivers, and neither bound is tight.  the gap is exactly what
    `ood_inflation` is for and it is set wide here for that reason.
    """

    component: str
    bands: tuple[BandFit, ...] = ()
    #: fraction of the teacher's residual variance treated as shared.  a front end
    #: writing every tonotopic place at once has *highly* correlated errors -- if
    #: its compression exponent is wrong it is wrong at every place in the same
    #: direction -- so the default here is higher than `TeacherPrecision`'s.
    correlated_fraction: float = 0.9
    #: rank 1 and stated as a choice.  §4 is explicit that the closed-form
    #: discount is a rank-one statement and that a higher rank hands a teacher
    #: orders of magnitude more influence; nothing here has measured a correlation
    #: length across the cochlea, so 1 is the only defensible declaration.
    error_rank: int = 1
    measured_on: str = ""
    front_end: str = ""
    note: str = ""

    @classmethod
    def unmeasured(cls, component: str, why: str, front_end: str = "") -> "ForcingCalibration":
        """a calibration that refuses to supply a precision, and says why.

        the visual arm's state.  it is a constructor rather than an absence
        because an absent calibration reads as an oversight and this one is a
        finding: the film is not in the deposit, so the number cannot be measured,
        so the forcing may not be applied.  `teacher()` raises with `why` attached
        rather than returning a default, because a default here is precisely the
        unit-precision failure §4 warns about wearing a smaller hat.
        """
        return cls(component=component, bands=(), note=why, front_end=front_end)

    @property
    def measured(self) -> bool:
        return bool(self.bands)

    @property
    def overall_r2(self) -> float:
        """the sample-weighted mean r^2 across bands.

        weighted by held-out samples and not by bandwidth, because what makes one
        band's figure more trustworthy than another's is how much data went into
        it, not how wide it is.
        """
        if not self.bands:
            return 0.0
        w = np.array([max(b.n_samples, 1) for b in self.bands], float)
        r = np.array([b.r2 for b in self.bands], float)
        return float(np.sum(w * r) / np.sum(w))

    def teacher(self, band: Band | None = None):
        """the `TeacherPrecision` this calibration earns, for one band or overall.

        the returned object carries `verified=True`, which in `TeacherPrecision`
        means "this r^2 was measured rather than read off a paper".  that flag is
        the difference between a calibrated forcing and a cited one, and it is set
        here only because `ForcingCalibration` cannot be constructed with bands
        that nobody measured.
        """
        from ibm.runtime.fuse import TeacherPrecision

        if not self.measured:
            raise ValueError(
                f"forcing on {self.component!r} has no measured r2 and therefore no precision: "
                f"{self.note or 'no reason recorded'}.  a forcing may not be applied at a "
                "guessed precision -- that is the unit-precision failure ARCHITECTURE.md §4 "
                "describes, and it is worse here because it would be invisible")
        r2 = self._r2_for(band)
        return TeacherPrecision(
            r2=min(max(r2, 0.0), 0.999),
            ood_inflation=Prior("lognormal", float(np.log(3.0)), float(np.log(3.0)),
                                provenance=Provenance.WEAK,
                                note="the r2 was measured downstream of the component being "
                                     "written, through several processes and an unknown lead "
                                     "field, so it bounds rather than states the front end's "
                                     "fidelity; and it was measured on read speech, which is "
                                     "not the distribution any other use will be on"),
            error_rank=self.error_rank,
            correlated_fraction=self.correlated_fraction,
            on_benchmark=self.measured_on,
            verified=True,
            source=self.front_end or "ibm.processes.forcing",
        )

    def _r2_for(self, band: Band | None) -> float:
        if band is None:
            return self.overall_r2
        best, hit = self.overall_r2, False
        for b in self.bands:
            if b.band.lo_hz <= band.lo_hz and band.hi_hz <= b.band.hi_hz + 1e-9:
                best, hit = b.r2, True
        if not hit:
            overlap = [b for b in self.bands
                       if b.band.hi_hz > band.lo_hz and b.band.lo_hz < band.hi_hz]
            if overlap:
                w = np.array([max(b.n_samples, 1) for b in overlap], float)
                best = float(np.sum(w * [b.r2 for b in overlap]) / np.sum(w))
        return best

    def shrinkage(self, band: Band, prior_var: float = 1.0) -> float:
        """how far towards the prior the forced value is pulled, in one band.

            w = dJ / (J_prior + dJ),   dJ = 1 / ((1 - r2) Var[x] * inflation)

        the number that makes "evidence, not a clamp" arithmetic.  a clamp is
        `w = 1` and it is what a forcing pipeline does by default; a calibrated
        forcing at an r^2 of 0.02, an inflation of 3 and a unit prior variance is
        `w` near 0.25, which is a very different object.

        at a linear readout with a fitted gain, a band-independent `w` is absorbed
        and changes nothing -- so the honest statement is that this matters
        because it is *not* band-independent.  the r^2 of a speech front end
        against MEG is several times larger in the delta-theta range than above
        it, so the shrinkage reweights the bands, and that reweighting is a
        Wiener filter derived from measurement rather than a smoothing constant
        somebody chose.
        """
        t = self.teacher(band)
        dj = 1.0 / max((1.0 - t.r2) * float(prior_var) * t.inflation(), _EPS)
        return float(dj / (1.0 / max(float(prior_var), _EPS) + dj))

    def describe(self) -> str:
        head = (f"forcing calibration for {self.component}"
                + (f" via {self.front_end}" if self.front_end else ""))
        if not self.measured:
            return f"{head}: UNMEASURED -- {self.note}"
        lines = [f"{head}  (measured on {self.measured_on or 'unrecorded data'})"]
        lines += [f"  {b}" for b in self.bands]
        lines.append(f"  overall r2 {self.overall_r2:+.4f}; rank {self.error_rank}, "
                     f"{self.correlated_fraction:.0%} of residual variance shared")
        if self.note:
            lines.append(f"  ! {self.note}")
        return "\n".join(lines)


def force(component: str, value: np.ndarray, calibration: ForcingCalibration, *,
          prior_var: np.ndarray | float = 1.0, sites: np.ndarray | None = None,
          band: Band | None = None, rng: np.random.Generator | None = None):
    """turn a front end's output into `Evidence` at the precision it earned.

    one line of arithmetic and three refusals, and the refusals are the content.

    it refuses an uncalibrated forcing, because a forcing at a guessed precision
    is indistinguishable in the state from a measurement.  it refuses to call the
    result `measured`, because `TeacherPrecision.evidence` stamps it `distilled`
    and that stamp is what `ibm.materialize.provenance` reads.  and it never
    clamps: the returned increment is finite, low-rank-discounted, and composes
    with whatever else constrains the component by `J' = J + dJ` like any other
    evidence.
    """
    t = calibration.teacher(band)
    b = band or Band(0.0, float("inf"))
    return t.evidence(component, np.asarray(value), prior_var, sites=sites, band=b, rng=rng)


# ---------------------------------------------------------------------------
# provenance: forced or evolved
# ---------------------------------------------------------------------------


FORCED = "forced"
EVOLVED = "evolved"
CLAMPED = "clamped"
MEASURED = "measured"


@dataclass(frozen=True)
class ComponentFate:
    """how one component got its value in a driven run.

    the distinction this module exists to keep.  a component that was written by
    a stimulus front end and a component that the model's own dynamics produced
    are different claims, and a report that presents them alike is claiming an
    emergent result for something that was put there by hand.  §7 asks for the
    same thing about parameters; this is its counterpart for state, and it is
    here rather than in `ibm.materialize.provenance` only because that module is
    about what a *build* knows and this is about what a *run* did.
    """

    component: str
    status: str                       # forced | evolved | clamped | measured
    via: str = ""                     # front end or process that wrote it
    r2: float | None = None
    shrinkage: float | None = None
    band: Band | None = None
    note: str = ""

    def __str__(self) -> str:
        bits = [f"{self.component}: {self.status.upper()}"]
        if self.via:
            bits.append(f"via {self.via}")
        if self.r2 is not None:
            bits.append(f"r2={self.r2:+.4f}")
        if self.shrinkage is not None:
            bits.append(f"shrinkage={self.shrinkage:.3f}")
        if self.band is not None:
            bits.append(str(self.band))
        s = ", ".join(bits)
        return s + (f"  ({self.note})" if self.note else "")


@dataclass
class ForcingRecord:
    """every component a driven run touched, and how.

    mutable, unlike `ibm.materialize.provenance.Provenance`, and the difference is
    deliberate: that record is built once at build time and frozen so it cannot
    drift from what was materialized, whereas this one accumulates over a run as
    each component is written.  what keeps it honest is that it is written by the
    code doing the writing rather than by a summariser afterwards.

    `as_notes()` renders it into strings that can be handed to
    `Provenance(notes=...)`.  that direction is deliberate too -- this module
    knows about `materialize` and `materialize` does not know about this one, so
    forcing can be added to a pipeline without any edit to the provenance record
    it feeds.
    """

    run: str = ""
    fates: list[ComponentFate] = _field(default_factory=list)

    def forced(self, component: str, calibration: ForcingCalibration, band: Band,
               *, via: str = "", prior_var: float = 1.0) -> "ForcingRecord":
        self.fates.append(ComponentFate(
            component, FORCED, via or calibration.front_end,
            r2=calibration._r2_for(band) if calibration.measured else None,
            shrinkage=calibration.shrinkage(band, prior_var) if calibration.measured else None,
            band=band,
            note="" if calibration.measured else calibration.note))
        return self

    def evolved(self, component: str, via: str = "", note: str = "") -> "ForcingRecord":
        self.fates.append(ComponentFate(component, EVOLVED, via, note=note))
        return self

    def clamped(self, component: str, via: str = "", note: str = "") -> "ForcingRecord":
        self.fates.append(ComponentFate(component, CLAMPED, via, note=note))
        return self

    def measured(self, component: str, via: str = "", note: str = "") -> "ForcingRecord":
        self.fates.append(ComponentFate(component, MEASURED, via, note=note))
        return self

    def status_of(self, component: str) -> str:
        for f in self.fates:
            if f.component == component:
                return f.status
        return EVOLVED

    def is_downstream_of_forcing(self, component: str) -> bool:
        """true if this component is at or below a forced one in the declared chain.

        the question a reader actually has: "was this result driven or did it come
        out?".  it is answered by walking `FORCING_CHAIN` rather than by a flag,
        so a component that is downstream of a forced one inherits the caveat even
        though nothing wrote it directly -- which is the case that a per-component
        flag alone would get wrong.
        """
        forced_at = {f.component for f in self.fates if f.status == FORCED}
        if component in forced_at:
            return True
        seen = False
        for link in FORCING_CHAIN:
            if link.into in forced_at:
                seen = True
            if link.into == component:
                return seen
        return False

    def as_notes(self) -> tuple[str, ...]:
        """the record as provenance notes, worst first.

        forced components lead, because a reader skimming a provenance block needs
        to know what was put there before they read what came out.
        """
        order = {FORCED: 0, CLAMPED: 1, MEASURED: 2, EVOLVED: 3}
        rows = sorted(self.fates, key=lambda f: (order.get(f.status, 9), f.component))
        n_forced = sum(1 for f in rows if f.status == FORCED)
        head = (f"driven run{' ' + self.run if self.run else ''}: {n_forced} of {len(rows)} "
                "components FORCED from a stimulus front end rather than evolved.  a result "
                "downstream of a forced component is not an emergent one")
        return (head,) + tuple(f"  {f}" for f in rows)

    def describe(self) -> str:
        return "\n".join(self.as_notes())


# ---------------------------------------------------------------------------
# the import-time check
# ---------------------------------------------------------------------------

#: problems with the declared chain, computed lazily by `check_chain()`.  it is
#: not run at import because this module is imported *by* the process package and
#: the interventions it names are registered later in the same package's
#: `__init__`; running it here would report a hole that closes a few lines after.
#: `ibm.load_all()` seals the registry, and `check_chain()` is meant to be called
#: after that -- `scripts/force_early_sensory.py` does exactly that and refuses to
#: run if it returns anything.
__all__ = [
    "AUDITORY_CHAIN", "VISUAL_CHAIN", "FORCING_CHAIN", "Link", "check_chain", "describe_chain",
    "CochlearFrontEnd", "RetinalFrontEnd", "erb_hz", "greenwood_places", "chain_taps",
    "BandFit", "ForcingCalibration", "force",
    "ComponentFate", "ForcingRecord", "FORCED", "EVOLVED", "CLAMPED", "MEASURED",
]
