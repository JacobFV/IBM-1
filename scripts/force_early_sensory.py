"""drive early auditory state from a continuous naturalistic stimulus, and ask
what it bought.

the experiment ARCHITECTURE.md's §4 and §6 imply and nothing in this repository
had yet run.  a stationary spectral fit constrains a component's marginal second
moment; a continuous stimulus-brain pair constrains the map from a known input to
a measured output at every instant, which is a far higher-bandwidth statement
about dynamics.  this script measures how much higher.

three models are compared on the same held-out data, and the comparison is set up
so that it can come out badly:

*unforced.*  the declared graph with no stimulus evidence.  its stimulus-locked
prediction is the mean, so its r^2 is zero by construction -- not a contest, and
reported anyway because it is the number the whole exercise is measured against.

*driven.*  `device.speaker_pressure` clamped by `naturalistic_stream`, through
`transduction`'s `gammatone_cochleagram` into `transduction.hair_cell`, then
through `afferent_propagation` and `thalamocortical_coupling` as *declared
transfer functions with literature time constants*, then through one fitted
instantaneous lead field.  the whole temporal model is four filters shared across
every sensor and every participant.

*teacher-direct.*  the same cochleagram regressed straight onto the same sensors
with a free 41-lag kernel per sensor per tonotopic place.  this is the standard
encoding model and it is strictly more expressive than the driven model -- it can
represent any kernel the driven model can, and 1148 predictors per sensor rather
than 112.  if the declared chain is adding structure rather than only removing
freedom, the gap between them should be small; if the declaration is wrong, the
gap is where it shows.

the arithmetic that makes the search cheap is worth naming because it also makes
the comparison exact.  every model here is linear in the lagged cochleagram, so
the driven model is the teacher-direct model restricted to the span of four basis
kernels: `K_driven = W B` with `B` the taps' impulse responses.  so one pass over
the corpus accumulates `X'X` and `X'Y` in the full lagged basis, and every
chain-parameter setting after that is a projection `P' X'X P`.  the two models are
then fitted on identical sufficient statistics and differ only in the constraint,
which is the only way the comparison means anything.

splits.  `naturalistic_stream`'s own declaration says why: continuous material is
heavily autocorrelated, so a cut inside a story leaks a speaker, a topic and a
spectral envelope into the test set.  every split here is therefore by whole
story and by whole participant, and the headline number is the intersection --
held-out participants listening to a held-out story.

what is not here, and why.  no visual arm: the only held source pairing a
naturalistic film with localized electrodes is ds003688, whose `stimuli/`
directory deposits thirty video annotation tables and no film.  no pretrained
teacher: `data/sources/tribe` and `tribe-v2` set `local_root: null`, so no weights
are on this machine and none are pretended.  what runs is the analytic front end
in `ibm.processes.forcing`.

    ./.venv/bin/python scripts/force_early_sensory.py cache
    ./.venv/bin/python scripts/force_early_sensory.py fit
    ./.venv/bin/python scripts/force_early_sensory.py eval
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ibm                                                   # noqa: E402
from ibm.forge.spectra import local_root                     # noqa: E402
from ibm.processes.forcing import (                          # noqa: E402
    BandFit,
    CochlearFrontEnd,
    ForcingCalibration,
    ForcingRecord,
    chain_taps,
    check_chain,
    describe_chain,
    force,
)
from ibm.fields.uncertainty.spectral import TemporalBasis     # noqa: E402
from ibm.vocabulary import Band                               # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# the choices, in one place
# ---------------------------------------------------------------------------

#: analysis rate.  100 Hz is the ceiling the whole design is set by: the cortical
#: response to continuous speech is a low-frequency phenomenon, MEG above 30 Hz in
#: a listening task is dominated by muscle, and every extra hertz multiplies the
#: lag basis the teacher-direct model has to fit.  a higher rate would flatter
#: neither model and would cost both.
FS = 100.0

#: passband.  the floor is above the 0.1 Hz drift every MEG amplifier has and
#: above the slow head-position term; the ceiling is where speech tracking has
#: stopped and myogenic power has started.
LO_HZ, HI_HZ = 0.5, 30.0

#: TRF support, in samples.  0 to 400 ms covers every auditory latency anybody
#: reports for continuous speech -- the M50, the M100, and the long negative tail
#: that carries most of the envelope response -- and stops before the window
#: starts absorbing the stimulus's own autocorrelation.
N_LAGS = 41

#: tonotopic places.  28 is a materialization choice and not a physiological one:
#: it is the coarsest sampling that still resolves the first two formants, and it
#: keeps the teacher-direct model at 1148 predictors rather than at several
#: thousand, which is the number that decides whether the comparison is about
#: model structure or about regularization.
N_PLACES = 28

#: bands the calibration is resolved over.  chosen because the r^2 of a speech
#: front end against MEG is emphatically not flat across them, and a single
#: overall figure would hand the delta band's precision to the beta band.
CAL_BANDS = (Band(0.5, 4.0), Band(4.0, 8.0), Band(8.0, 13.0), Band(13.0, 30.0))

#: the four stories, and what each is for.  every split here is by whole story
#: and by whole participant, for the reason `naturalistic_stream` gives: a cut
#: inside a stream leaks a speaker, a topic and a spectral envelope.
#:
#: `easy_money` is the test material and is touched by no fit anywhere.
#: `cable_spool_fort` selects the chain's time constants and the ridge, on
#: training participants only.  `lw1` and `the_black_willow` are what the lead
#: fields are fitted on.  the roles were assigned by story length -- the largest
#: two train, the second-smallest validates, the middle one tests -- before any
#: number was computed, rather than by which assignment looked best.
TEST_STORY = "easy_money"
VAL_STORY = "cable_spool_fort"
TRAIN_STORIES = ("lw1", "the_black_willow")
HELD_OUT_SUBJECTS = ("sub-08", "sub-09", "sub-10", "sub-11")

CACHE = Path(os.environ.get(
    "IBM_FORCING_CACHE",
    "/tmp/claude-1000/-home-brandonin-Documents-IBM-1/"
    "d6587b2e-c347-48cf-bb14-8df80b8c7706/scratchpad/forcing"))

#: ridge grid, as a multiple of the design's mean diagonal so it is scale free.
#: it runs three decades below where the first version started, because the first
#: version's floor of 1 was above the optimum: a global ridge chosen by the mean
#: over 208 channels lands on whatever protects the ~150 channels that carry no
#: auditory response at all, and that is a very large amount of smoothing.
_RIDGES = np.array([1e-2, 1e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 1e3, 1e4, 1e5], float)


# ---------------------------------------------------------------------------
# stage 1: cache
# ---------------------------------------------------------------------------


def _exact_ratio(out_fs: float, in_fs: float) -> tuple[int, int]:
    """integer up/down factors, or an error.

    the same refusal `ibm.processes.forcing._ratio` makes, for the same reason: a
    rounded resampling ratio drifts the recording against the stimulus by an
    amount that grows with the recording's length, and over a half-hour run that
    drift is larger than every latency in the chain being fitted.
    """
    from math import gcd

    a, b = int(round(out_fs * 100)), int(round(in_fs * 100))
    if abs(out_fs * 100 - a) > 1e-6 or abs(in_fs * 100 - b) > 1e-6:
        raise ValueError(f"sample rates {in_fs} -> {out_fs} have no exact integer ratio")
    g = gcd(a, b)
    return a // g, b // g


def _sound_events(path: Path) -> list[tuple[float, str, str]]:
    """(onset seconds, story, wav filename) for every sound this run played.

    the `trial_type` column is a python dict literal rather than a label, and its
    `sound` field carries a filename with a float index -- `lw1_0.0.wav` for a
    file called `lw1_0.wav`.

    it also carries a *case* that does not match the deposit.  the events name
    `The_Black_Willow_3.wav`; the file on disk is `the_black_willow_3.wav`.  that
    one detail silently removed the largest story in the corpus -- twelve of the
    thirty segments per recording, about half the data -- from an earlier run of
    this script, because the lookup returned nothing and the loop moved on.  the
    fix is to fold the case here, and the real fix is that `load_segments` now
    counts what it could not pair and refuses rather than continuing, since a
    quietly halved corpus is far worse than a crash.
    """
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    out = []
    for _, row in df.iterrows():
        try:
            d = ast.literal_eval(row["trial_type"])
        except (ValueError, SyntaxError):
            continue
        if not isinstance(d, dict) or d.get("kind") != "sound":
            continue
        name = str(d.get("sound", "")).split("/")[-1]
        stem = name[:-4] if name.endswith(".wav") else name
        if stem.endswith(".0"):
            stem = stem[:-2]
        out.append((float(row["onset"]), str(d.get("story", "")).lower(),
                    stem.lower() + ".wav"))
    return out


def cache_cochleagrams(root: Path, out: Path) -> dict[str, np.ndarray]:
    """every stimulus wav through the cochlear front end, once.

    cached because the bank costs about five seconds per ninety seconds of audio
    and the same thirty-one files are read by every participant.  the cache is
    keyed by filename only: the front end has no fitted parameter, so the same
    file always produces the same cochleagram, which is a property worth relying
    on and worth stating -- if it did not hold, this cache would be a subtle way
    of fitting the front end to whichever run happened to run first.
    """
    from scipy.io import wavfile

    out.mkdir(parents=True, exist_ok=True)
    front = CochlearFrontEnd(n_places=N_PLACES, out_fs_hz=FS)
    cochs: dict[str, np.ndarray] = {}
    audio = root / "stimuli" / "audio"
    for wav in sorted(audio.glob("*.wav")):
        dst = out / f"{wav.stem}.npy"
        if dst.is_file():
            cochs[wav.name] = np.load(dst)
            continue
        sr, x = wavfile.read(wav)
        c = front(x.astype(float), float(sr))
        np.save(dst, c)
        cochs[wav.name] = c
        print(f"  cochleagram {wav.name}: {c.shape[1] / FS:6.1f} s at {sr} Hz", flush=True)
    return cochs


def cache_meg(root: Path, out: Path) -> None:
    """every run's MEG, band-limited, decimated, robustly scaled, sliced by sound.

    three preprocessing choices carry weight and are recorded rather than left in
    a filter call.

    *the scale is a median absolute deviation, per channel, per run.*  a KIT axial
    gradiometer array has real channel-to-channel gain differences and real
    session-to-session ones, and an r^2 pooled over channels without them removed
    is dominated by whichever channel is loudest.  MAD rather than standard
    deviation because a single SQUID jump moves the second moment by orders of
    magnitude and the median by nothing.

    *nothing is rejected.*  a blink or a jump stays in, and it costs both models
    equally.  removing artifacts would raise every number here and would make the
    comparison depend on a rejection threshold nobody can calibrate against a
    continuous recording with no trials to drop.

    *only the sound-aligned stretches are kept.*  the ~8 s gaps between segments
    are not silence in any useful sense -- they are a participant waiting -- and
    including them would let both models score on predicting that nothing is
    happening.
    """
    import mne

    mne.set_log_level("ERROR")
    out.mkdir(parents=True, exist_ok=True)
    # sharding by index, set from the environment, so the eighty-odd runs can be
    # decoded by several processes at once.  each shard writes disjoint files and
    # skips what already exists, so a shard that dies can simply be re-run.
    shard, n_shards = (int(v) for v in os.environ.get("IBM_SHARD", "0,1").split(","))
    for i, con in enumerate(sorted(root.glob("sub-*/ses-*/meg/*_meg.con"))):
        if i % n_shards != shard:
            continue
        stem = con.name.replace("_meg.con", "")
        dst = out / f"{stem}.npz"
        if dst.is_file():
            continue
        ev = con.parent / f"{stem}_events.tsv"
        if not ev.is_file():
            print(f"  {stem}: no events file, skipped", flush=True)
            continue
        # picked before the data are loaded and filtered with a zero-phase IIR
        # rather than an FIR.  both are speed and neither is free: an IIR of this
        # order has a slightly less brick-wall response than the default firwin,
        # which matters not at all here because the passband is chosen by
        # physiology rather than by a stopband requirement, and reading 49 unused
        # MISC channels off disk for every run is pure cost.
        raw = mne.io.read_raw_kit(str(con), preload=False, verbose="ERROR")
        raw.pick(mne.pick_types(raw.info, meg=True))
        raw.load_data(verbose="ERROR")
        raw.filter(LO_HZ, HI_HZ, method="iir",
                   iir_params=dict(order=4, ftype="butter", output="sos"), verbose="ERROR")
        # decimated with `resample_poly` rather than `Raw.resample`.  the reason
        # is memory and it is not marginal: MNE's resampler is FFT-based and pads
        # the whole recording to a fast length, so a half-hour run at 1 kHz over
        # 208 channels transiently needs tens of gigabytes, and six of those in
        # parallel took this machine to its limit.  the signal has already been
        # low-passed at 30 Hz by the line above, so a polyphase decimation to
        # 100 Hz is exact for everything that survives that filter.
        from scipy.signal import resample_poly
        fs_in = float(raw.info["sfreq"])
        up, down = _exact_ratio(FS, fs_in)
        x = resample_poly(raw.get_data(), up, down, axis=-1).astype(np.float32)
        mad = np.median(np.abs(x - np.median(x, 1, keepdims=True)), 1, keepdims=True)
        x = x / np.maximum(mad * 1.4826, 1e-30)
        loc = np.array([raw.info["chs"][i]["loc"][:3] for i in range(x.shape[0])], float)

        segs, meta = [], []
        for onset, story, wav in _sound_events(ev):
            i0 = int(round(onset * FS))
            if i0 < 0 or i0 >= x.shape[1]:
                continue
            segs.append(i0)
            meta.append((story, wav))
        np.savez(
            dst, data=x, loc=loc, starts=np.array(segs, int),
            stories=np.array([m[0] for m in meta]), wavs=np.array([m[1] for m in meta]),
            ch_names=np.array(raw.ch_names), fs=FS)
        print(f"  {stem}: {x.shape[0]} ch x {x.shape[1] / FS:.0f} s, "
              f"{len(segs)} sounds", flush=True)


# ---------------------------------------------------------------------------
# the corpus, as segments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    subject: str
    session: str
    task: str
    story: str
    wav: str
    y: np.ndarray            # (T, n_ch) measured MEG
    c: np.ndarray            # (T, n_places) forced hair-cell state
    loc: np.ndarray          # (n_ch, 3) sensor positions, device frame

    @property
    def recording(self) -> str:
        """the unit a lead field belongs to: one person, one time in the dewar.

        not the participant.  a MEG participant is repositioned in the helmet
        between sessions, so the map from cortical current to channel is a
        property of (person, session) and not of the person; a lead field fitted
        in one session and applied to the next is applying a rotated montage.
        measured here on sub-02: a kernel fitted on session 0 scores +0.0013 mean
        r^2 on a held-out story of the *same* session and -0.0010 on the same
        story in session 1, which is the sign of a montage that has moved.
        """
        return f"{self.subject}/{self.session}"


def load_segments(cache: Path, cochs: dict[str, np.ndarray],
                  subjects: tuple[str, ...] | None = None,
                  stories: tuple[str, ...] | None = None,
                  exclude_stories: tuple[str, ...] = ()) -> list[Segment]:
    """pair every cached run with the cochleagram of what was playing.

    the pairing is by the events file's own onset, which is the trigger channel's
    sample index converted to seconds by the deposit's own conversion.  no
    cross-correlation is used to refine it: a refinement fitted per run would
    absorb exactly the latency this experiment is trying to estimate, and would do
    so invisibly because it would improve every model's r^2 at once.
    """
    segs: list[Segment] = []
    unpaired: set[str] = set()
    for f in sorted(cache.glob("sub-*.npz")):
        parts = f.stem.split("_")
        sub, ses, task = parts[0], parts[1], parts[2]
        if subjects is not None and sub not in subjects:
            continue
        d = np.load(f, allow_pickle=False)
        x, starts = d["data"], d["starts"]
        for i, i0 in enumerate(starts):
            story, wav = str(d["stories"][i]).lower(), str(d["wavs"][i]).lower()
            if stories is not None and story not in stories:
                continue
            if story in exclude_stories:
                continue
            c = cochs.get(wav)
            if c is None:
                unpaired.add(wav)
                continue
            n = min(c.shape[1], x.shape[1] - int(i0))
            if n < int(20 * FS):
                continue
            segs.append(Segment(sub, ses, task, story, wav,
                                x[:, int(i0): int(i0) + n].T.astype(np.float32).copy(),
                                c[:, :n].T.astype(np.float32).copy(),
                                d["loc"]))
    if unpaired:
        raise SystemExit(
            f"{len(unpaired)} stimulus file(s) named in the events could not be paired with "
            f"a cochleagram: {sorted(unpaired)[:6]}.  this is refused rather than skipped "
            "because skipping it is what silently removed the largest story in this corpus "
            "from an earlier run, and a quietly halved training set produces a quietly wrong "
            "answer.")
    return segs


def channel_mask(segs: list[Segment], factor: float = 20.0) -> dict[str, np.ndarray]:
    """which channels of each recording are usable, from their variance alone.

    a plain bad-channel rule, and it is here because leaving it out produced the
    single most misleading number in an earlier run of this script: a mean r^2 of
    -0.073 on held-out participants whose *median* channel was +0.0009.  the mean
    was six broken sensors.

    the mechanism is worth writing down because it is not obvious.  each run is
    scaled by its own per-channel median absolute deviation, which is the right
    scale for an amplifier whose gain drifts and is deliberately blind to a SQUID
    jump.  a channel that is saturated in one run therefore comes out with an
    ordinary MAD and a variance five million times the array median -- and since
    the lead field is fitted on one story and scored on another, its weight is
    fitted at one scale and applied at a completely different one.  the resulting
    r^2 is not small, it is enormous and negative, and it swamps two hundred
    healthy channels.

    the rule uses no target and no prediction: a channel is bad in a recording if
    its variance in any segment of that recording exceeds `factor` times the
    array's median variance.  measured on sub-08, that separates cleanly -- the
    healthy session's worst channel is at 10x the median and the broken session's
    worst is at 4,500,000x -- so the threshold is not doing delicate work.  the
    count of what it drops is printed, because a rejection rule that is not
    reported is a rejection rule that can be tuned.
    """
    out: dict[str, np.ndarray] = {}
    for rec in sorted({s.recording for s in segs}):
        mine = [s for s in segs if s.recording == rec]
        v = np.stack([s.y.var(0) for s in mine])              # (n_seg, n_ch)
        out[rec] = v.max(0) <= factor * float(np.median(v))
    return out


# ---------------------------------------------------------------------------
# sufficient statistics in the lagged basis
# ---------------------------------------------------------------------------


#: rows of the lagged design built at once.  the segments here run to nearly half
#: an hour, and a whole one in the lagged basis is most of a gigabyte before the
#: float64 cast doubles it.  chunking with a lookback of `n_lags - 1` samples is
#: exactly equivalent -- every row is built from the same history it would have
#: had -- and caps the working set at a few hundred megabytes regardless of how
#: long a recording is.
CHUNK = 60000


def _chunks(t: int, n_lags: int):
    for a in range(0, t, CHUNK):
        b = min(a + CHUNK, t)
        lo = max(0, a - n_lags + 1)
        yield lo, a, b


def _lagged(c: np.ndarray, n_lags: int = N_LAGS) -> np.ndarray:
    """(T, places) -> (T, places * lags), causal, zero-padded at the start.

    causal and *only* causal.  a lag basis that included negative lags would let
    every model here predict the brain before the stimulus, which improves an r^2
    -- an acausal filter has more freedom -- and is silent in every summary
    statistic.  the padding at the start is zeros rather than a reflection because
    a reflection invents stimulus that was not played.
    """
    t, p = c.shape
    out = np.zeros((t, p * n_lags), np.float32)
    for l in range(n_lags):
        out[l:, l * p:(l + 1) * p] = c[: t - l]
    return out


def accumulate(segs: list[Segment], n_lags: int = N_LAGS,
               mask: dict[str, np.ndarray] | None = None) -> dict:
    """one pass over the corpus for `X'X`, `X'Y`, `Y'Y` and the counts.

    the whole reason the chain-parameter search below is affordable.  every model
    compared here is linear in the same lagged cochleagram, so the corpus enters
    the fit only through these three matrices, and everything after this function
    is dense linear algebra on a 1148-dimensional space.  it also means the driven
    and teacher-direct models are fitted on *identical* sufficient statistics,
    which is what makes the comparison about the constraint rather than about two
    slightly different training sets.

    the cochleagram is centred per segment and the MEG is centred per segment,
    which removes a per-run DC offset that neither model should be scored on
    predicting.  the scaling is left alone: a global cochleagram scale is absorbed
    by the fitted gains, and rescaling per segment would remove the level
    differences between passages, which is real stimulus structure.
    """
    p = segs[0].c.shape[1]
    d = p * n_lags
    n_ch = segs[0].y.shape[1]
    xtx = np.zeros((d, d), np.float64)
    xty = np.zeros((d, n_ch), np.float64)
    yty = np.zeros(n_ch, np.float64)
    n = 0
    good = np.ones(n_ch, bool)
    for s in segs:
        if mask is not None and s.recording in mask:
            good &= mask[s.recording]
        c = s.c - s.c.mean(0, keepdims=True)
        y = s.y - s.y.mean(0, keepdims=True)
        for lo, a, b in _chunks(c.shape[0], n_lags):
            x = _lagged(c[lo:b], n_lags)[a - lo:]
            xd = x.astype(np.float64)
            xtx += xd.T @ xd
            xty += xd.T @ y[a:b]
            n += x.shape[0]
        yty += np.einsum("tc,tc->c", y, y, dtype=np.float64)
    return {"xtx": xtx, "xty": xty, "yty": yty, "n": n, "p": p, "n_lags": n_lags,
            "n_ch": n_ch, "good": good}


# ---------------------------------------------------------------------------
# the two model families, as projections of the same basis
# ---------------------------------------------------------------------------


def tap_basis(theta: dict, n_lags: int = N_LAGS) -> np.ndarray:
    """the declared chain's impulse responses, `(n_taps, n_lags)`.

    this is where the ontology actually enters the arithmetic.  `chain_taps`
    returns each tap as `H(omega)` on a `TemporalBasis`, because that is the form
    every process in ibm-1 declares its dynamics in; here they are brought back
    into the time domain and truncated to the TRF's support, which is the
    conversion §1 says a nonlinear or windowed use has to pay for.

    the basis length is four times the TRF support so the truncation lands on a
    tail that has already decayed.  a tap whose response is still large at 400 ms
    would be truncated into a discontinuity, and the resulting kernel would carry
    a spurious edge; the assertion below is what makes that a failure rather than
    a quiet distortion.
    """
    n = 4 * n_lags
    basis = TemporalBasis(n, 1.0 / FS)
    taps = chain_taps(basis, **theta)
    rows = []
    for h in taps.values():
        full = np.zeros(n // 2 + 1, complex)
        full[: len(h)] = h
        ir = np.fft.irfft(full, n)[:n_lags]
        rows.append(ir)
    b = np.stack(rows)
    tail = np.abs(b[:, -3:]).max(1) / np.maximum(np.abs(b).max(1), 1e-30)
    if float(tail.max()) > 0.25:
        raise ValueError(
            "a chain tap is still at more than a quarter of its peak at the end of the "
            "TRF support, so truncating it here would put an artificial edge in the "
            "kernel.  widen N_LAGS or shorten the offending time constant rather than "
            f"accepting it: tail fractions {np.round(tail, 3).tolist()}")
    return b


def _projection(b: np.ndarray, p: int) -> np.ndarray:
    """`P` mapping (place, tap) coefficients into the (place, lag) basis.

    the driven model's kernel is `K[place, lag] = sum_tap W[place, tap] B[tap,
    lag]`, so `X_driven = X P` with this `P`, and every second-moment matrix the
    driven fit needs is `P' X'X P`.  the ordering of the columns has to match
    `_lagged`'s -- place-fastest within a lag -- and getting it wrong produces a
    model that is silently fitting a permuted tonotopy, which trains fine and
    generalizes to nothing.
    """
    n_taps, n_lags = b.shape
    proj = np.zeros((p * n_lags, p * n_taps))
    for l in range(n_lags):
        for k in range(n_taps):
            proj[l * p:(l + 1) * p, k * p:(k + 1) * p] = np.eye(p) * b[k, l]
    return proj


def ridge_solve(a: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    d = a.shape[0]
    scale = float(np.trace(a)) / max(d, 1)
    return np.linalg.solve(a + lam * scale * np.eye(d), b)


def fitted_r2(stats: dict, w: np.ndarray, proj: np.ndarray | None = None) -> np.ndarray:
    """in-sample r^2 per channel from sufficient statistics alone.

    used only for the ridge sweep on a *validation* corpus whose statistics were
    accumulated separately; calling it on the training statistics would report the
    training fit and choose the smallest ridge every time, which is the failure
    mode this signature makes visible by taking `stats` as an argument rather than
    holding one.
    """
    xtx, xty, yty = stats["xtx"], stats["xty"], stats["yty"]
    if proj is not None:
        xtx = proj.T @ xtx @ proj
        xty = proj.T @ xty
    # `optimize=True` is not a micro-optimization here: the naive contraction of
    # w' X'X w over a 1148-dimensional design and 208 channels is quartic and
    # takes minutes, and the ridge sweep evaluates it dozens of times.
    resid = (yty - 2.0 * np.einsum("dc,dc->c", w, xty)
             + np.einsum("dc,de,ec->c", w, xtx, w, optimize=True))
    r = 1.0 - resid / np.maximum(yty, 1e-30)
    good = stats.get("good")
    return np.where(good, r, np.nan) if good is not None else r


def accumulate_by_recording(segs: list[Segment], n_lags: int = N_LAGS,
                            mask: dict[str, np.ndarray] | None = None) -> dict[str, dict]:
    """the same statistics, kept per participant rather than pooled.

    the correction that makes this experiment mean anything, and it was found the
    expensive way: a first version pooled every participant into one design and
    both models scored an r^2 of 3e-5, a hundredfold below what a single
    participant's cross-correlation with the speech envelope plainly shows
    (r = 0.10 at 160 ms on the best lateral sensors).  pooling was the whole
    error.  a MEG sensor index is not an anatomical label -- channel 71 sits over
    different cortex in two people, and the field it sees can be of opposite sign
    -- so one lead field shared across participants averages the response towards
    nothing.

    the fix is not a trick, it is what the ontology already says.  the *chain* is
    physiology and its time constants are declared `Tying.PER_PARTITION` or
    global; the *lead field* is one person's head geometry and is
    `Tying.PER_SITE` on that person's array.  so the chain is fitted across
    participants and the lead field within one, and the held-out-participant test
    is a test of whether the shared part transfers -- which is the only part that
    could.
    """
    out: dict[str, dict] = {}
    if mask is None:
        mask = channel_mask(segs)
    for key in sorted({s.recording for s in segs}):
        out[key] = accumulate([s for s in segs if s.recording == key], n_lags, mask)
    return out


def cached_stats(tag: str, segs_fn, n_lags: int = N_LAGS,
                 mask: dict[str, np.ndarray] | None = None) -> dict[str, dict]:
    """`accumulate_by_recording`, memoised on disk under `tag`.

    the pass over the corpus costs about ten minutes and the statistics are a
    deterministic function of (segments, n_lags), so recomputing them for every
    experiment on the same split is pure waste.  `segs_fn` is a thunk rather than
    a list so that a cache hit never touches the six gigabytes of MEG on disk at
    all -- which is most of the ten minutes.
    """
    f = CACHE / f"stats_{tag}_{n_lags}.npz"
    if f.is_file():
        d = np.load(f, allow_pickle=False)
        keys = [str(k) for k in d["keys"]]
        return {k: {"xtx": d[f"xtx_{i}"], "xty": d[f"xty_{i}"], "yty": d[f"yty_{i}"],
                    "good": d[f"good_{i}"], "n": int(d["n"][i]), "p": int(d["p"]),
                    "n_lags": n_lags, "n_ch": int(d["n_ch"])}
                for i, k in enumerate(keys)}
    st = accumulate_by_recording(segs_fn(), n_lags, mask)
    keys = list(st)
    payload = {"keys": np.array(keys), "n": np.array([st[k]["n"] for k in keys]),
               "p": np.array(st[keys[0]]["p"]), "n_ch": np.array(st[keys[0]]["n_ch"])}
    for i, k in enumerate(keys):
        payload[f"xtx_{i}"] = st[k]["xtx"]
        payload[f"xty_{i}"] = st[k]["xty"]
        payload[f"yty_{i}"] = st[k]["yty"]
        payload[f"good_{i}"] = st[k]["good"]
    np.savez(f, **payload)
    return st


def summarize(r: np.ndarray) -> str:
    """a per-channel r^2 vector as the four numbers that are worth reporting.

    the mean over all 208 channels is the headline and it is also the least
    informative of the four, because most of a whole-head MEG array is nowhere
    near auditory cortex and contributes noise to it.  the top-decile mean is what
    a reader who wants to know whether the model found the response should look
    at, and the best channel is what makes the two comparable to a published
    figure.  reporting only the mean understates every model here equally;
    reporting only the best overstates them equally.
    """
    v = r[np.isfinite(r)]
    o = np.sort(v)[::-1]
    k = max(len(o) // 10, 1)
    return (f"n {v.size:3d}  mean {float(np.mean(v)):+.5f}  "
            f"median {float(np.median(v)):+.5f}  "
            f"top10% {float(np.mean(o[:k])):+.5f}  best {float(o[0]):+.5f}")


def _predict(c: np.ndarray, w: np.ndarray, proj: np.ndarray | None,
             n_lags: int = N_LAGS) -> np.ndarray:
    """the model's prediction for one segment, built in chunks.

    the kernel is collapsed into the lagged basis once -- `proj @ w` is the
    driven model's TRF written out in the same coordinates as the teacher-direct
    model's -- so the two prediction paths are literally the same matrix multiply
    and cannot diverge through a bookkeeping error in only one of them.
    """
    k = (proj @ w) if proj is not None else w
    out = np.empty((c.shape[0], k.shape[1]), np.float32)
    for lo, a, b in _chunks(c.shape[0], n_lags):
        out[a:b] = _lagged(c[lo:b], n_lags)[a - lo:] @ k
    return out


def predict_r2(segs: list[Segment], w: np.ndarray | dict[str, np.ndarray],
               proj: np.ndarray | None = None, n_lags: int = N_LAGS,
               mask: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """out-of-sample r^2 per channel, from actual predictions on held-out segments.

    `w` is either one lead field or a map from participant to lead field; the
    second form is the one that is used, and the first is kept because the
    libribrain arm has exactly one participant and does not need the indirection.

    deliberately not computed from sufficient statistics.  the two agree
    algebraically -- there is an assertion of that in the synthetic check -- and
    keeping a path that builds the prediction explicitly is what catches an error
    in the projection or the lag ordering, both of which are invisible in a
    quadratic form and obvious in a waveform.
    """
    if mask is None:
        mask = channel_mask(segs)
    num = den = None
    for s in segs:
        wi = w[s.recording] if isinstance(w, dict) else w
        if wi is None:
            continue
        m = mask.get(s.recording, np.ones(s.y.shape[1], bool)).astype(float)
        c = s.c - s.c.mean(0, keepdims=True)
        y = s.y - s.y.mean(0, keepdims=True)
        r = y - _predict(c, wi, proj, n_lags)
        a, b = (r * r).sum(0) * m, (y * y).sum(0) * m
        num = a if num is None else num + a
        den = b if den is None else den + b
    return np.where(den > 0, 1.0 - num / np.maximum(den, 1e-30), np.nan)


def band_r2(segs: list[Segment], w: np.ndarray | dict[str, np.ndarray], band: Band,
            proj: np.ndarray | None = None, n_lags: int = N_LAGS,
            mask: dict[str, np.ndarray] | None = None) -> tuple[float, int]:
    """out-of-sample r^2 inside one band, and how many samples it rests on.

    the band restriction is applied to the residual and to the target with the
    same filter, which is the only version of this that is a variance ratio:
    filtering only the prediction would compare a band-limited estimate with a
    broadband target and report a number that cannot reach one however good the
    model is.

    the ratio is formed **per channel and then averaged**, not pooled over the
    array.  pooling looks equivalent and is not: the channels are scaled by their
    median absolute deviation, which is deliberately insensitive to a SQUID jump,
    so a channel carrying two artefacts has an ordinary MAD and an enormous
    variance.  a pooled sum is then dominated by exactly the channels that carry
    no signal, and it reported a delta-band r^2 of 0.0001 for a model whose mean
    per-channel figure over the same data was 0.0013 -- a factor of thirteen, all
    of it metric and none of it model.  averaging the ratios weights every sensor
    once, which is what the headline number does, so the two are comparable.
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [max(band.lo_hz, 0.1), min(band.hi_hz, 0.49 * FS)],
                 btype="band", fs=FS, output="sos")
    if mask is None:
        mask = channel_mask(segs)
    num = den = None
    n = 0
    for s in segs:
        wi = w[s.recording] if isinstance(w, dict) else w
        if wi is None:
            continue
        m = mask.get(s.recording, np.ones(s.y.shape[1], bool)).astype(float)
        c = s.c - s.c.mean(0, keepdims=True)
        y = s.y - s.y.mean(0, keepdims=True)
        pred = _predict(c, wi, proj, n_lags)
        yb = sosfiltfilt(sos, y, axis=0)
        rb = sosfiltfilt(sos, y - pred, axis=0)
        a, b = (rb * rb).sum(0) * m, (yb * yb).sum(0) * m
        num = a if num is None else num + a
        den = b if den is None else den + b
        n += y.shape[0]
    if num is None:
        return 0.0, 0
    r = np.where(den > 0, 1.0 - num / np.maximum(den, 1e-30), np.nan)
    return float(np.nanmean(r)), n


# ---------------------------------------------------------------------------
# chain parameters
# ---------------------------------------------------------------------------

#: the declared time constants, at their literature priors' medians.  these are
#: the starting point and, for four of the seven, the ending point: the search
#: below moves only what a naturalistic listening recording could plausibly
#: identify, and leaves the rest where `ibm.processes.device` and
#: `ibm.processes.neural` put them.
THETA0 = dict(mean_delay_s=0.015, sd_delay_s=0.005, tau_synapse_s=2e-3,
              tau_nmda_s=0.1, tau_relay_s=0.012, tc_delay_s=5e-3, ct_delay_s=8e-3,
              tau_trn_gaba_b_s=0.15, loop_gain=2.0, tau_cortex_s=0.02)

#: what the fit is allowed to move, and over what range.  each grid is inside the
#: declared prior's support, so a fitted value is a posterior mode within
#: p(theta) rather than an unconstrained optimum -- which is the difference
#: between forging and fitting a filterbank to MEG.  `tau_synapse_s` is not in
#: here: a thalamic relay AMPA EPSP is measured to within a factor of two and a
#: 400 ms MEG kernel at 100 Hz cannot see a 2 ms constant, so letting the fit
#: touch it would move a literature parameter on no evidence.
GRIDS = {
    "mean_delay_s": [0.005, 0.010, 0.015, 0.020, 0.030, 0.045],
    "sd_delay_s": [0.002, 0.005, 0.010, 0.020],
    "tau_nmda_s": [0.05, 0.1, 0.2],
    "tau_relay_s": [0.006, 0.012, 0.025, 0.04],
    "tc_delay_s": [0.003, 0.005, 0.010, 0.020],
    "ct_delay_s": [0.005, 0.008, 0.015, 0.030],
    "tau_trn_gaba_b_s": [0.05, 0.1, 0.15, 0.3],
    "loop_gain": [0.5, 1.0, 2.0, 4.0],
    "tau_cortex_s": [0.01, 0.02, 0.05, 0.1],
}


def search_chain(train: dict[str, dict], val: dict[str, dict],
                 n_lags: int = N_LAGS) -> tuple[dict, float, float]:
    """coordinate descent over the chain's time constants, scored on validation.

    the score is the mean over training participants of that participant's
    validation r^2, with that participant's own lead field.  the chain is one
    object across all of them, which is the tying the declarations state; the lead
    fields are seven separate objects, which is also what they state.  the search
    therefore moves only the shared part, and moves it towards whatever makes all
    seven heads fit at once rather than towards whatever suits the loudest one.

    coordinate descent and not a gradient because the objective is a ridge
    solution's held-out r^2, which is cheap but not differentiable through the
    ridge choice, and because the grids are small enough that two sweeps visit
    most of the space.  the ridge is re-selected inside every evaluation, so a
    theta is never rewarded for happening to suit the previous theta's
    regularization.
    """
    theta = dict(THETA0)
    best, lam = _score(theta, train, val, n_lags)
    for _ in range(2):
        for name, grid in GRIDS.items():
            for v in grid:
                if v == theta[name]:
                    continue
                cand = dict(theta, **{name: v})
                try:
                    s, l = _score(cand, train, val, n_lags)
                except ValueError:
                    continue
                if s > best:
                    theta, best, lam = cand, s, l
    return theta, best, lam


def _score(theta: dict, train: dict[str, dict], val: dict[str, dict],
           n_lags: int) -> tuple[float, dict[str, float]]:
    proj = _projection(tap_basis(theta, n_lags), N_PLACES)
    r, lams = select_ridges(train, val, proj)
    return float(np.mean(list(r.values()))), lams


def select_ridges(train: dict[str, dict], val: dict[str, dict],
                  proj: np.ndarray | None = None
                  ) -> tuple[dict[str, float], dict[str, float]]:
    """the best ridge for each recording, chosen on the validation story.

    per recording and not one global value, which is standard for a temporal
    response function and is here for a measured reason.  a single ridge picked
    by the mean over 208 channels is picked by the ~150 of them that carry no
    auditory response, because those are what the mean is made of; it lands three
    decades above the optimum for the sensors that do carry one, and it cost a
    factor of four in r^2 in an earlier run of this script.  the validation story
    is never the test story, so nothing here is selected against the number that
    gets reported.

    both models get exactly this treatment.  giving the flexible one a tuned
    ridge and the constrained one a fixed one would be the sort of asymmetry that
    makes a comparison worthless in the direction the author was hoping for.
    """
    keys = [k for k in train if k in val]
    # the projections are done once rather than once per ridge.  `P' X'X P` over a
    # 1148-dimensional design is the dominant cost of the whole search.
    prepared = {}
    for k in keys:
        if proj is None:
            prepared[k] = (train[k]["xtx"], train[k]["xty"], val[k])
        else:
            prepared[k] = (proj.T @ train[k]["xtx"] @ proj, proj.T @ train[k]["xty"],
                           dict(val[k], xtx=proj.T @ val[k]["xtx"] @ proj,
                                xty=proj.T @ val[k]["xty"]))
    best_r, best_lam = {}, {}
    for k, (a, b, v) in prepared.items():
        scored = [(float(np.nanmean(fitted_r2(v, ridge_solve(a, b, lam)))), float(lam))
                  for lam in _RIDGES]
        best_r[k], best_lam[k] = max(scored)
    return best_r, best_lam


def fit_lead_fields(stats: dict[str, dict], lam: float | dict[str, float],
                    proj: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """one lead field per participant, at a ridge chosen elsewhere.

    the ridge is passed in rather than swept here, and that is the whole point of
    the signature: a ridge selected against the participant's own test data is a
    hyperparameter fitted on the test set, and it would favour the more flexible
    model, which is the one this experiment is trying to be sceptical about.
    """
    out = {}
    default = float(np.median(list(lam.values()))) if isinstance(lam, dict) else float(lam)
    for s, st in stats.items():
        l = lam.get(s, default) if isinstance(lam, dict) else float(lam)
        if proj is None:
            out[s] = ridge_solve(st["xtx"], st["xty"], l)
        else:
            out[s] = ridge_solve(proj.T @ st["xtx"] @ proj, proj.T @ st["xty"], l)
    return out


def _sweep_direct(train: dict[str, dict], val: dict[str, dict]):
    """the ridges for the teacher-direct model, chosen exactly as the driven one's."""
    r, lams = select_ridges(train, val, None)
    return float(np.mean(list(r.values()))), lams


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_cache() -> None:
    root = local_root("meg-masc")
    print(f"meg-masc at {root}")
    cache_cochleagrams(root, CACHE / "coch")
    cache_meg(root, CACHE / "meg")


def _corpus():
    cochs = {p.stem + ".wav": np.load(p) for p in (CACHE / "coch").glob("*.npy")}
    return cochs


def stage_fit() -> None:
    ibm.load_all(seal=True, strict=True)
    problems = check_chain()
    if problems:
        raise SystemExit("the declared forcing chain does not resolve:\n  "
                         + "\n  ".join(problems))
    print("forcing chain:")
    print(describe_chain())

    cochs = _corpus()
    meg = CACHE / "meg"
    train_subj = tuple(sorted({p.stem.split("_")[0] for p in meg.glob("sub-*.npz")}
                              - set(HELD_OUT_SUBJECTS)))
    print(f"\ntrain participants: {', '.join(train_subj)}")
    print(f"held-out participants: {', '.join(HELD_OUT_SUBJECTS)}")
    print(f"stories: fit on {', '.join(TRAIN_STORIES)}; select on {VAL_STORY}; "
          f"test on {TEST_STORY}")

    #: the split is by story and the lead field is per *recording* -- one person,
    #: one time in the dewar.  both choices were forced by measurement rather than
    #: chosen: a lead field shared across people scores nothing at all, and one
    #: shared across a participant's two sessions loses most of what it had,
    #: because the participant is repositioned in the helmet between them.
    # the bad-channel mask is computed once over ALL of the training
    # participants' material, not per split.  a channel that is saturated in the
    # story the lead field is fitted on and healthy in the story it is scored on
    # is the case a per-split mask misses, and it is the case that matters: the
    # weight is fitted at one scale and applied at another.
    all_train_segs = load_segments(meg, cochs, subjects=train_subj)
    mask = channel_mask(all_train_segs)
    dropped = {k: int((~v).sum()) for k, v in mask.items() if (~v).any()}
    print(f"\nbad channels dropped, by recording: {dropped or 'none'} "
          f"(variance above 20x the array median in some segment)")
    del all_train_segs

    train = cached_stats("train_seen", lambda: load_segments(
        meg, cochs, subjects=train_subj, stories=TRAIN_STORIES), mask=mask)
    val = cached_stats("val_seen", lambda: load_segments(
        meg, cochs, subjects=train_subj, stories=(VAL_STORY,)), mask=mask)
    print(f"\naccumulated {sum(t['n'] for t in train.values()):,} training samples "
          f"({sum(t['n'] for t in train.values()) / FS / 60:.0f} min) over "
          f"{len(train)} recordings x {next(iter(train.values()))['n_ch']} channels, "
          f"design dimension {next(iter(train.values()))['xtx'].shape[0]}")

    theta, r_val, lam = search_chain(train, val)
    print(f"\nchain parameters after forging against {VAL_STORY} in the training "
          "participants.  the chain is ONE object across every recording; only the lead "
          "field is per recording.")
    for k in sorted(THETA0):
        mark = "  <- moved" if theta[k] != THETA0[k] else ""
        print(f"  {k:16s} {THETA0[k]:8.4f} -> {theta[k]:8.4f}{mark}")
    print(f"  driven: validation r2 {r_val:+.5f}, per-recording ridges "
          f"{sorted(set(lam.values()))}")
    r_val_d, lam_d = _sweep_direct(train, val)
    print(f"  teacher-direct: validation r2 {r_val_d:+.5f}, per-recording ridges "
          f"{sorted(set(lam_d.values()))}")

    proj = _projection(tap_basis(theta), N_PLACES)
    w_driven = fit_lead_fields(train, lam, proj)
    val_segs = load_segments(meg, cochs, subjects=train_subj, stories=(VAL_STORY,))
    val_mask = mask

    # the calibration used *during* fitting: measured on the validation story,
    # which is held out of the statistics above.  the figure quoted as the
    # forcing's accuracy is the stricter one computed in `eval`, on participants
    # and material that were both unseen.
    fits = []
    print(f"\ncalibration r2, per band, on {VAL_STORY} in the training participants:")
    for b in CAL_BANDS:
        r, n = band_r2(val_segs, w_driven, b, proj, mask=val_mask)
        fits.append(BandFit(b, float(r), n_targets=next(iter(train.values()))["n_ch"],
                            n_samples=int(n),
                            held_out=f"{VAL_STORY}, training participants"))
        print(f"  {fits[-1]}")
    cal = ForcingCalibration(
        component="transduction.hair_cell", bands=tuple(fits),
        measured_on=f"meg-masc, 7 participants, held-out story {VAL_STORY}",
        front_end="gammatone_cochleagram",
        note="the r2 is measured on MEG, several processes and one unknown lead field "
             "downstream of the component being written; it bounds rather than states the "
             "front end's own fidelity")

    np.savez(CACHE / "fit.npz", theta=json.dumps(theta), lam=json.dumps(lam),
             lam_d=json.dumps(lam_d),
             cal=json.dumps([[b.band.lo_hz, b.band.hi_hz, b.r2, b.n_samples] for b in fits]))
    print("\n" + cal.describe())
    for b in CAL_BANDS:
        print(f"  shrinkage in {b}: {cal.shrinkage(b):.4f}   (a hard clamp would be 1.0000)")


def _fit_and_test(fit_segs: list[Segment], test_segs: list[Segment], proj: np.ndarray,
                  lam, lam_d, minutes: float | None = None
                  ) -> tuple[np.ndarray, np.ndarray, float]:
    """fit both models' lead fields on one set of segments and score them on another.

    `minutes` truncates the fitting set, per participant, to the first N minutes of
    it.  truncating from the front rather than sampling at random is deliberate:
    an adaptation session in a real experiment is the beginning of the recording,
    not a stratified sample of it, and a random sample would spread the training
    data across every story and quietly remove the material-transfer part of the
    problem.
    """
    if minutes is not None:
        cut, kept, used = minutes * 60.0 * FS, [], {}
        for s in fit_segs:
            have = used.get(s.recording, 0.0)
            if have >= cut:
                continue
            take = int(min(s.y.shape[0], cut - have))
            used[s.recording] = have + take
            kept.append(Segment(s.subject, s.session, s.task, s.story, s.wav,
                                s.y[:take], s.c[:take], s.loc))
        fit_segs = kept
    # one bad-channel mask over the union of the fitting and test material, and
    # the union is the point.  a channel saturated in the story the lead field was
    # fitted on but healthy in the story it is scored on passes a mask computed
    # from the test set alone, and then contributes a weight fitted at one scale
    # applied at another -- which is exactly the -0.07 mean r^2 an earlier run
    # reported for a set of channels whose median was +0.0009.
    mask = channel_mask(fit_segs + test_segs)
    st = accumulate_by_recording(fit_segs, mask=mask)
    w_dr = fit_lead_fields(st, lam, proj)
    w_te = fit_lead_fields(st, lam_d, None)
    for s in {x.recording for x in test_segs}:
        w_dr.setdefault(s, None)
        w_te.setdefault(s, None)
    minutes_used = sum(x.y.shape[0] for x in fit_segs) / FS / 60.0 / max(len(st), 1)
    return (predict_r2(test_segs, w_dr, proj, mask=mask),
            predict_r2(test_segs, w_te, mask=mask), minutes_used)


def stage_eval() -> None:
    ibm.load_all(seal=True, strict=True)
    cochs = _corpus()
    meg = CACHE / "meg"
    d = np.load(CACHE / "fit.npz", allow_pickle=False)
    theta = json.loads(str(d["theta"]))
    lam = {k: float(v) for k, v in json.loads(str(d["lam"])).items()}
    lam_d = {k: float(v) for k, v in json.loads(str(d["lam_d"])).items()}
    proj = _projection(tap_basis(theta), N_PLACES)
    cal_rows = json.loads(str(d["cal"]))
    cal = ForcingCalibration(
        component="transduction.hair_cell",
        bands=tuple(BandFit(Band(lo, hi), r2, n_samples=int(n)) for lo, hi, r2, n in cal_rows),
        measured_on=f"meg-masc, 7 participants, held-out story {VAL_STORY}",
        front_end="gammatone_cochleagram")

    all_subj = sorted({p.stem.split("_")[0] for p in meg.glob("sub-*.npz")})
    train_subj = tuple(s for s in all_subj if s not in HELD_OUT_SUBJECTS)

    print("\nheld-out r2, mean over the 208 MEG channels.\n"
          "the chain's time constants are frozen at the values forged on the seven "
          "training participants.\nthe lead field is fitted within each test participant "
          "on material that excludes the test story,\nwhich is the only thing a sensor "
          "array permits: a channel index is not an anatomical label.\n"
          "the unforced model carries no stimulus information at all, so its "
          "stimulus-locked prediction\nis the mean and its r2 is exactly zero by "
          "construction.  it is the baseline, not a contestant.\n")

    header = (f"{'split':44s} {'ch':>4s} {'unforced':>9s} {'driven':>9s} {'teacher':>9s} "
              f"{'driven/teacher':>15s}")
    print(header)
    print("-" * len(header))

    results = {}
    per_channel = {}
    rows = [
        ("seen participants, held-out story",
         dict(subjects=train_subj, stories=TRAIN_STORIES),
         dict(subjects=train_subj, stories=(TEST_STORY,))),
        ("held-out participants, chain-selection story",
         dict(subjects=HELD_OUT_SUBJECTS, stories=TRAIN_STORIES),
         dict(subjects=HELD_OUT_SUBJECTS, stories=(VAL_STORY,))),
        ("held-out participants x held-out story",
         dict(subjects=HELD_OUT_SUBJECTS, stories=TRAIN_STORIES + (VAL_STORY,)),
         dict(subjects=HELD_OUT_SUBJECTS, stories=(TEST_STORY,))),
    ]
    for name, fit_kw, test_kw in rows:
        fit_segs = load_segments(meg, cochs, **fit_kw)
        test_segs = load_segments(meg, cochs, **test_kw)
        if not fit_segs or not test_segs:
            print(f"{name:44s}  no segments")
            continue
        r_dr, r_te, _ = _fit_and_test(fit_segs, test_segs, proj, lam, lam_d)
        per_channel[name] = (r_dr, r_te, len(test_segs))
        top = max(r_dr.size // 10, 1)
        results[name] = dict(driven=float(np.nanmean(r_dr)), teacher=float(np.nanmean(r_te)),
                             driven_top10=float(np.mean(np.sort(r_dr[np.isfinite(r_dr)])[::-1][:top])),
                             teacher_top10=float(np.mean(np.sort(r_te[np.isfinite(r_te)])[::-1][:top])),
                             driven_best=float(np.nanmax(r_dr)),
                             teacher_best=float(np.nanmax(r_te)),
                             n_test_segments=len(test_segs))
        ratio = float(np.nanmean(r_dr)) / max(float(np.nanmean(r_te)), 1e-12)
        print(f"{name:44s} {r_dr.size:4d} {0.0:9.5f} {float(np.nanmean(r_dr)):9.5f} "
              f"{float(np.nanmean(r_te)):9.5f} {ratio:15.2f}")
        print(f"{'  driven  ':44s} {summarize(r_dr)}")
        print(f"{'  teacher ':44s} {summarize(r_te)}")
    print(f"\nbest single channel, held-out participants x held-out story: "
          f"driven r2 {results.get('held-out participants x held-out story', {}).get('driven_best', 0):.4f}, "
          f"teacher r2 {results.get('held-out participants x held-out story', {}).get('teacher_best', 0):.4f}")

    # how much of a new person's data each model needs.  this is where a
    # constrained model can genuinely win and where the claim that the
    # declaration is carrying structure is actually testable: the driven model
    # brings a chain forged on seven other people and has 112 free numbers per
    # sensor to fit; the teacher-direct model brings nothing and has 1148.
    print("\nadaptation curve on the four held-out participants, tested on the held-out "
          "story.\nminutes are per participant, taken from the start of their recordings.\n")
    print(f"{'minutes of the new person':>26s} {'driven':>9s} {'teacher':>9s} "
          f"{'driven/teacher':>15s}")
    fit_segs = load_segments(meg, cochs, subjects=HELD_OUT_SUBJECTS,
                             stories=TRAIN_STORIES + (VAL_STORY,))
    test_segs = load_segments(meg, cochs, subjects=HELD_OUT_SUBJECTS,
                              stories=(TEST_STORY,))
    curve = []
    for minutes in (2.0, 5.0, 10.0, 20.0, 40.0, None):
        r_dr, r_te, used = _fit_and_test(fit_segs, test_segs, proj, lam, lam_d, minutes)
        label = f"{used:.1f}" if minutes is not None else f"{used:.1f} (all)"
        ratio = float(np.nanmean(r_dr)) / max(float(np.nanmean(r_te)), 1e-12)
        curve.append((used, float(np.nanmean(r_dr)), float(np.nanmean(r_te))))
        print(f"{label:>26s} {float(np.nanmean(r_dr)):9.5f} {float(np.nanmean(r_te)):9.5f} "
              f"{ratio:15.2f}")

    # the same comparison restricted to sensors that are downstream of the forced
    # region rather than over it.  "downstream" is defined on TRAINING
    # participants only -- the sensors whose own direct kernel explains the least
    # -- and applied unchanged to held-out participants, so the definition cannot
    # have been tuned on the answer.
    tr_fit = load_segments(meg, cochs, subjects=train_subj, stories=TRAIN_STORIES)
    tr_test = load_segments(meg, cochs, subjects=train_subj, stories=(TEST_STORY,))
    _, r_train, _ = _fit_and_test(tr_fit, tr_test, proj, lam, lam_d)
    order = np.argsort(np.nan_to_num(r_train, nan=-1e9))
    downstream = order[: int(0.6 * len(order))]
    early = order[int(0.9 * len(order)):]
    loc = tr_fit[0].loc
    print(f"\nsensor split from training participants only: {len(early)} 'early' sensors "
          f"(top decile of direct-kernel r2), {len(downstream)} 'downstream' sensors "
          f"(bottom 60%).  device-frame centroids in mm:")
    print(f"  early:      |x| {np.abs(loc[early, 0]).mean() * 1e3:5.1f}  "
          f"y {loc[early, 1].mean() * 1e3:+6.1f}  z {loc[early, 2].mean() * 1e3:+6.1f}")
    print(f"  downstream: |x| {np.abs(loc[downstream, 0]).mean() * 1e3:5.1f}  "
          f"y {loc[downstream, 1].mean() * 1e3:+6.1f}  "
          f"z {loc[downstream, 2].mean() * 1e3:+6.1f}")

    for label, idx in (("early sensors", early), ("downstream sensors", downstream)):
        print(f"\n{label:44s} {'ch':>4s} {'unforced':>9s} {'driven':>9s} {'teacher':>9s}")
        for name, (r_dr, r_te, n_seg) in per_channel.items():
            print(f"{name:44s} {idx.size:4d} {0.0:9.5f} "
                  f"{float(np.nanmean(r_dr[idx])):9.5f} {float(np.nanmean(r_te[idx])):9.5f}")

    # the calibration the forcing actually earns, on the strictest split available.
    strict_fit = load_segments(meg, cochs, subjects=HELD_OUT_SUBJECTS,
                               stories=TRAIN_STORIES + (VAL_STORY,))
    strict_test = load_segments(meg, cochs, subjects=HELD_OUT_SUBJECTS,
                                stories=(TEST_STORY,))
    strict_mask = channel_mask(strict_fit + strict_test)
    w_strict = fit_lead_fields(accumulate_by_recording(strict_fit, mask=strict_mask),
                               lam, proj)
    fits = []
    for band in CAL_BANDS:
        r, n = band_r2(strict_test, w_strict, band, proj, mask=strict_mask)
        fits.append(BandFit(band, float(r), n_targets=strict_test[0].y.shape[1],
                            n_samples=int(n),
                            held_out="held-out participants x held-out story"))
    strict_cal = ForcingCalibration(
        component="transduction.hair_cell", bands=tuple(fits),
        measured_on="meg-masc, 4 unseen participants, 1 unseen story",
        front_end="gammatone_cochleagram",
        note="measured on MEG, several processes and one unknown lead field downstream of "
             "the component being written")
    print("\n" + strict_cal.describe())
    print("\n  shrinkage the calibrated forcing applies, per band "
          "(a hard clamp would be 1.0000 everywhere):")
    for band in CAL_BANDS:
        print(f"    {band}: {strict_cal.shrinkage(band):.4f}   "
              f"(fitting-time figure {cal.shrinkage(band):.4f})")

    # the increment the forcing is actually allowed to contribute, with both of
    # §4's corrections applied.  what matters is not the diagonal precision but
    # what survives the rank-1 shared-error discount.
    ev = force("transduction.hair_cell", np.zeros(N_PLACES), strict_cal,
               prior_var=1.0, band=Band(0.5, 30.0))
    print(f"\n  {ev.describe()}")
    print(f"  {N_PLACES} forced tonotopic places are worth "
          f"{ev.effective_constraints():.2f} independent measurements about anything "
          f"they have in common")

    # provenance: what was forced and what evolved.
    rec = ForcingRecord(run="meg-masc auditory forcing")
    rec.clamped("device.speaker_pressure", via="naturalistic_stream",
                note="the audiobook waveform, exogenous state on the display support")
    rec.forced("transduction.hair_cell", strict_cal, Band(0.5, 30.0),
               via="gammatone_cochleagram")
    for c in ("neural.afferent.activity", "neural.exc.ampa", "neural.exc.activity",
              "neural.transmembrane_current"):
        rec.evolved(c, via="afferent_propagation / thalamocortical_coupling / "
                          "local_excitation at declared literature time constants")
    rec.measured("electromagnetic.bfield", via="meg-masc",
                 note="the only measured component in the chain; everything above it is "
                      "forced or evolved")
    print("\nprovenance of the driven run:")
    print(rec.describe())
    print("\nis a result at neural.exc.activity emergent?  "
          + ("NO -- it is downstream of a forced component"
             if rec.is_downstream_of_forcing("neural.exc.activity") else "yes"))

    with (CACHE / "results.json").open("w") as fh:
        json.dump({"theta": theta, "results": results, "adaptation_curve": curve,
                   "calibration_fit": cal_rows,
                   "calibration_strict": [[b.band.lo_hz, b.band.hi_hz, b.r2, b.n_samples]
                                          for b in strict_cal.bands],
                   "effective_constraints": float(ev.effective_constraints())},
                  fh, indent=2)


def stage_libribrain() -> None:
    """the same chain, unchanged, on a different person, scanner and book.

    the test §4 asks for and that almost nothing in this literature runs: a
    teacher's reported accuracy holds on the benchmark distribution, and used off
    it the precision must be inflated by an amount that is itself uncertain.
    `ForcingCalibration` declares a lognormal inflation prior centred at 3x for
    exactly that reason, and this stage is the only way to find out whether 3 is
    anywhere near right.

    everything transfers except the lead field, which cannot: the sensor arrays
    are different instruments (208 KIT axial gradiometers against 306 Elekta
    magnetometers and planar gradiometers) with different geometry and different
    units, so a per-sensor gain fitted on one is meaningless on the other.  the
    chain's *time constants* transfer unchanged, and they are the part the
    declaration is a claim about.  so the comparison here is: with the chain
    frozen at the meg-masc posterior and only the lead field refitted, how much of
    the r^2 survives?

    the split is by session, which is the only split n=1 allows.  it tests
    generalisation across sessions and material and says nothing whatever about
    generalisation across people -- the libribrain card's own `held_out_splits`
    stream carries that warning and it is repeated here because it is the exact
    confusion this stage could otherwise invite.
    """
    ibm.load_all(seal=True, strict=True)
    root = local_root("libribrain") / "Sherlock1"
    out = CACHE / "libribrain"
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(CACHE / "fit.npz", allow_pickle=False)
    theta = json.loads(str(d["theta"]))
    proj = _projection(tap_basis(theta), N_PLACES)

    front = CochlearFrontEnd(n_places=N_PLACES, out_fs_hz=FS)
    segs = _libribrain_segments(root, out, front)
    if not segs:
        raise SystemExit("no libribrain sessions could be paired with a chapter")
    print(f"\n{len(segs)} sessions, "
          f"{sum(s.y.shape[0] for s in segs) / FS / 60:.0f} min of within-person MEG")

    # the split is *within* each held-out session: the first 70 per cent fits the
    # montage, the last 30 per cent is scored.  it has to be, and the reason is
    # the same one that governs the meg-masc arm.  a MEG participant is
    # repositioned in the helmet at every visit, and these are twelve separate
    # visits; a lead field pooled over sessions 1-7 and applied to session 9
    # scores exactly zero, for both models, which is what an averaged montage
    # looks like rather than what a weak model looks like.  a within-session
    # split is the only thing sensor space permits, and it is still a real test
    # of the part that transferred: the chain's ten time constants are frozen at
    # the meg-masc posterior and nothing about them is refitted here, on a
    # different person, a different scanner, a different array and a different
    # book.
    #
    # the ridge is chosen on sessions 1-8 and never on the tested tail.
    early = [s for s in segs if int(s.session.split("-")[1]) <= 8]
    test_sessions = [s for s in segs if int(s.session.split("-")[1]) > 8]
    print(f"  {len(early)} sessions choosing the ridge, {len(test_sessions)} held out")

    def _split(seg, frac=0.7):
        k = int(frac * seg.y.shape[0])
        return (Segment(seg.subject, seg.session, seg.task, seg.story, seg.wav,
                        seg.y[:k], seg.c[:k], seg.loc),
                Segment(seg.subject, seg.session, seg.task, seg.story, seg.wav,
                        seg.y[k:], seg.c[k:], seg.loc))

    # the same bad-channel rule the meg-masc arm uses, and it is not optional
    # here either: this array mixes 102 magnetometers with 204 planar
    # gradiometers, and without it the mean over 306 channels is maximised by the
    # largest ridge in the grid -- which is to say by predicting nothing.
    mask = channel_mask(segs)

    def _score_sessions(sessions, proj_or_none, lam):
        rs = []
        for seg in sessions:
            a, b = _split(seg)
            st = accumulate([a], mask=mask)
            if proj_or_none is None:
                w = ridge_solve(st["xtx"], st["xty"], lam)
            else:
                w = ridge_solve(proj_or_none.T @ st["xtx"] @ proj_or_none,
                                proj_or_none.T @ st["xty"], lam)
            rs.append(predict_r2([b], w, proj_or_none, mask=mask))
        r = np.concatenate(rs)
        # selected on the *top decile*, not the mean.  two thirds of a whole-head
        # array is nowhere near auditory cortex, so the mean over 306 channels is
        # a statement about the sensors that carry nothing, and maximising it
        # chooses the ridge that predicts nothing.  the decile is the same
        # statistic `summarize` reports, so the selection and the report agree.
        v = np.sort(r[np.isfinite(r)])[::-1]
        k = max(v.size // 10, 1)
        return float(np.mean(v[:k])), r

    lam_dr = max((_score_sessions(early, proj, l)[0], l) for l in _RIDGES)[1]
    lam_te = max((_score_sessions(early, None, l)[0], l) for l in _RIDGES)[1]
    r_driven, rd = _score_sessions(test_sessions, proj, lam_dr)
    r_direct, rt = _score_sessions(test_sessions, None, lam_te)
    print("\nheld-out sessions 9-12, chain frozen at the meg-masc posterior, montage "
          "refitted within each session:")
    print(f"  driven         ridge {lam_dr:g}  {summarize(rd)}")
    print(f"  teacher-direct ridge {lam_te:g}  {summarize(rt)}")

    test = [b for seg in test_sessions for b in (_split(seg)[1],)]
    w_best = {}
    for seg in test_sessions:
        a, _ = _split(seg)
        st = accumulate([a], mask=mask)
        w_best[seg.recording] = ridge_solve(proj.T @ st["xtx"] @ proj,
                                            proj.T @ st["xty"], lam_dr)

    fits = []
    for band in CAL_BANDS:
        r, n = band_r2(test, w_best, band, proj)
        fits.append(BandFit(band, float(r), n_targets=test[0].y.shape[1], n_samples=int(n),
                            held_out="last 30% of sessions 9-12 of one participant"))
        print(f"  {fits[-1]}")
    cal = ForcingCalibration(
        component="transduction.hair_cell", bands=tuple(fits),
        measured_on="libribrain, 1 participant, held-out sessions",
        front_end="gammatone_cochleagram",
        note="a within-person figure.  it tests generalisation across sessions and "
             "material and says nothing about generalisation across people")
    print("\n" + cal.describe())


def _libribrain_segments(root: Path, out: Path, front: CochlearFrontEnd) -> list[Segment]:
    """one segment per session, MEG paired with that session's chapter.

    the alignment is a *fitted line*, not an offset.  the events table carries
    `timemeg` and `timechapter` for twelve thousand phoneme onsets per session,
    and regressing one on the other recovers both the start offset and the clock
    ratio between the presentation machine and the MEG digitizer.  taking the
    first event's difference as a constant offset instead leaves a drift of tens
    of milliseconds by the end of an eighteen-minute chapter, which is larger than
    every latency in `chain_taps` and would be absorbed as a smearing of the
    kernel rather than showing up as an error.
    """
    import h5py
    import pandas as pd

    segs: list[Segment] = []
    for h5 in sorted((root / "derivatives" / "serialised").glob("*.h5")):
        ses = [p for p in h5.stem.split("_") if p.startswith("ses-")][0]
        n = int(ses.split("-")[1])
        wav = root / "stimuli" / "audio" / f"studyinscarlet_{n:02d}_doyle_64kb.wav"
        ev = next((root / "derivatives" / "events").glob(f"*_{ses}_*_events.tsv"), None)
        if not wav.is_file() or ev is None:
            continue
        cpath = out / f"coch_{n:02d}.npy"
        if cpath.is_file():
            c = np.load(cpath)
        else:
            from scipy.io import wavfile
            sr, x = wavfile.read(wav)
            c = front(x.astype(float), float(sr))
            np.save(cpath, c)
            print(f"  cochleagram chapter {n}: {c.shape[1] / FS:.0f} s", flush=True)

        e = pd.read_csv(ev, sep="\t").dropna(subset=["timemeg", "timechapter"])
        a, b = np.polyfit(e["timechapter"].to_numpy(float), e["timemeg"].to_numpy(float), 1)
        if abs(a - 1.0) > 0.05:
            print(f"  {ses}: clock ratio {a:.5f} is more than 5% off unity, which means the "
                  "pairing is wrong rather than merely drifting; skipped")
            continue

        mpath = out / f"meg_{n:02d}.npy"
        if mpath.is_file():
            y = np.load(mpath)
        else:
            from scipy.signal import butter, sosfiltfilt
            with h5py.File(h5, "r") as fh:
                raw = np.asarray(fh["data"], np.float32)
                fs_in = float(fh.attrs["sample_frequency"])
            sos = butter(4, [LO_HZ, HI_HZ], btype="band", fs=fs_in, output="sos")
            raw = sosfiltfilt(sos, raw, axis=-1)
            up, down = int(round(FS * 10)), int(round(fs_in * 10))
            from math import gcd
            g = gcd(up, down)
            from scipy.signal import resample_poly
            y = resample_poly(raw, up // g, down // g, axis=-1).astype(np.float32)
            mad = np.median(np.abs(y - np.median(y, 1, keepdims=True)), 1, keepdims=True)
            y = (y / np.maximum(mad * 1.4826, 1e-30)).astype(np.float32)
            np.save(mpath, y)
            print(f"  meg session {n}: {y.shape[0]} ch x {y.shape[1] / FS:.0f} s", flush=True)

        # the cochleagram is *resampled* onto the MEG clock, not merely shifted.
        # the fitted ratio is 1.0048 -- the chapter time base runs half a percent
        # fast against the digitizer -- which is five seconds of drift by the end
        # of an eighteen-minute chapter.  an offset-only alignment therefore has
        # the stimulus and the brain further apart at the end of a session than
        # any latency in the chain, and it scored an r^2 of exactly zero for both
        # models, which is what a destroyed alignment looks like: not a weak
        # result, an absent one.
        t_meg = np.arange(y.shape[1]) / FS
        t_chap = (t_meg - float(b)) / float(a)
        ok = (t_chap >= 0.0) & (t_chap <= (c.shape[1] - 1) / FS)
        if int(ok.sum()) < int(60 * FS):
            continue
        i0, i1 = int(np.argmax(ok)), int(len(ok) - np.argmax(ok[::-1]))
        src = t_chap[i0:i1] * FS
        grid = np.arange(c.shape[1], dtype=float)
        cr = np.stack([np.interp(src, grid, row) for row in c]).astype(np.float32)
        segs.append(Segment("sub-0", ses, "Sherlock1", f"chapter{n:02d}", wav.name,
                            y[:, i0:i1].T.copy(), cr.T.copy(),
                            np.zeros((y.shape[0], 3))))
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("cache", "fit", "eval", "libribrain"))
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    {"cache": stage_cache, "fit": stage_fit, "eval": stage_eval,
     "libribrain": stage_libribrain}[a.stage]()


if __name__ == "__main__":
    main()
