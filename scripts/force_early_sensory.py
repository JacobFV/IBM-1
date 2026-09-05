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

#: held out entirely from every fit.  one whole story and four whole participants,
#: chosen by name before anything was fitted rather than by which split looked
#: best afterwards.
HELD_OUT_STORY = "the_black_willow"
HELD_OUT_SUBJECTS = ("sub-08", "sub-09", "sub-10", "sub-11")

CACHE = Path(os.environ.get(
    "IBM_FORCING_CACHE",
    "/tmp/claude-1000/-home-brandonin-Documents-IBM-1/"
    "d6587b2e-c347-48cf-bb14-8df80b8c7706/scratchpad/forcing"))

_RIDGES = np.array([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], float)


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
    file called `lw1_0.wav`.  fixing that here rather than at read time is the
    difference between silently pairing MEG with the wrong story and failing
    loudly, since a missing file raises and a wrong one does not.
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
        out.append((float(row["onset"]), str(d.get("story", "")), stem + ".wav"))
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
    for f in sorted(cache.glob("sub-*.npz")):
        parts = f.stem.split("_")
        sub, ses, task = parts[0], parts[1], parts[2]
        if subjects is not None and sub not in subjects:
            continue
        d = np.load(f, allow_pickle=False)
        x, starts = d["data"], d["starts"]
        for i, i0 in enumerate(starts):
            story, wav = str(d["stories"][i]), str(d["wavs"][i])
            if stories is not None and story not in stories:
                continue
            if story in exclude_stories:
                continue
            c = cochs.get(wav)
            if c is None:
                continue
            n = min(c.shape[1], x.shape[1] - int(i0))
            if n < int(20 * FS):
                continue
            segs.append(Segment(sub, ses, task, story, wav,
                                x[:, int(i0): int(i0) + n].T.astype(np.float32).copy(),
                                c[:, :n].T.astype(np.float32).copy(),
                                d["loc"]))
    return segs


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


def accumulate(segs: list[Segment], n_lags: int = N_LAGS) -> dict:
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
    for s in segs:
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
            "n_ch": n_ch}


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
    return 1.0 - resid / np.maximum(yty, 1e-30)


def accumulate_by_subject(segs: list[Segment], n_lags: int = N_LAGS) -> dict[str, dict]:
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
    for sub in sorted({s.subject for s in segs}):
        out[sub] = accumulate([s for s in segs if s.subject == sub], n_lags)
    return out


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
               proj: np.ndarray | None = None, n_lags: int = N_LAGS) -> np.ndarray:
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
    num = None
    den = None
    for s in segs:
        wi = w[s.subject] if isinstance(w, dict) else w
        if wi is None:
            continue
        c = s.c - s.c.mean(0, keepdims=True)
        y = s.y - s.y.mean(0, keepdims=True)
        r = y - _predict(c, wi, proj, n_lags)
        num = (r * r).sum(0) if num is None else num + (r * r).sum(0)
        den = (y * y).sum(0) if den is None else den + (y * y).sum(0)
    return 1.0 - num / np.maximum(den, 1e-30)


def band_r2(segs: list[Segment], w: np.ndarray | dict[str, np.ndarray], band: Band,
            proj: np.ndarray | None = None, n_lags: int = N_LAGS) -> tuple[float, int]:
    """out-of-sample r^2 inside one band, and how many samples it rests on.

    the band restriction is applied to the residual and to the target with the
    same filter, which is the only version of this that is a variance ratio:
    filtering only the prediction would compare a band-limited estimate with a
    broadband target and report a number that cannot reach one however good the
    model is.
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [max(band.lo_hz, 0.1), min(band.hi_hz, 0.49 * FS)],
                 btype="band", fs=FS, output="sos")
    num = den = 0.0
    n = 0
    for s in segs:
        wi = w[s.subject] if isinstance(w, dict) else w
        if wi is None:
            continue
        c = s.c - s.c.mean(0, keepdims=True)
        y = s.y - s.y.mean(0, keepdims=True)
        pred = _predict(c, wi, proj, n_lags)
        yb = sosfiltfilt(sos, y, axis=0)
        rb = sosfiltfilt(sos, y - pred, axis=0)
        num += float((rb * rb).sum())
        den += float((yb * yb).sum())
        n += y.shape[0]
    return 1.0 - num / max(den, 1e-30), n


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
           n_lags: int) -> tuple[float, float]:
    proj = _projection(tap_basis(theta, n_lags), N_PLACES)
    subs = [s for s in train if s in val]
    # the projections are done once per theta rather than once per (theta, ridge).
    # `P' X'X P` over a 1148-dimensional design is the dominant cost of the whole
    # search, and recomputing it inside the ridge loop made the sweep eight times
    # slower for no change in the answer.
    proj_stats = {s: (proj.T @ train[s]["xtx"] @ proj, proj.T @ train[s]["xty"],
                      dict(val[s], xtx=proj.T @ val[s]["xtx"] @ proj,
                           xty=proj.T @ val[s]["xty"]))
                  for s in subs}
    per_lam = []
    for lam in _RIDGES:
        rs = [float(np.mean(fitted_r2(v, ridge_solve(a, b, lam))))
              for a, b, v in proj_stats.values()]
        per_lam.append((float(np.mean(rs)), lam))
    best, lam = max(per_lam)
    return best, lam


def fit_lead_fields(stats: dict[str, dict], lam: float,
                    proj: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """one lead field per participant, at a ridge chosen elsewhere.

    the ridge is passed in rather than swept here, and that is the whole point of
    the signature: a ridge selected against the participant's own test data is a
    hyperparameter fitted on the test set, and it would favour the more flexible
    model, which is the one this experiment is trying to be sceptical about.
    """
    out = {}
    for s, st in stats.items():
        if proj is None:
            out[s] = ridge_solve(st["xtx"], st["xty"], lam)
        else:
            out[s] = ridge_solve(proj.T @ st["xtx"] @ proj, proj.T @ st["xty"], lam)
    return out


def _sweep_direct(train: dict[str, dict], val: dict[str, dict]) -> float:
    """the ridge for the teacher-direct model, chosen the same way as the driven one."""
    subs = [s for s in train if s in val]
    scored = []
    for lam in _RIDGES:
        rs = [float(np.mean(fitted_r2(val[s], ridge_solve(train[s]["xtx"],
                                                          train[s]["xty"], lam))))
              for s in subs]
        scored.append((float(np.mean(rs)), lam))
    return max(scored)


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
    print(f"held-out story: {HELD_OUT_STORY}")

    segs = load_segments(meg, cochs, subjects=train_subj,
                         exclude_stories=(HELD_OUT_STORY,))
    #: the validation split is by *session*, not by time: session 1 of every
    #: training participant is held out of the statistics the chain is fitted
    #: from, and is what the chain, the ridge and the calibration r^2 are chosen
    #: against.  a split by time inside a story would leak, for the reason
    #: `naturalistic_stream` gives.
    fit_segs = [s for s in segs if s.session != "ses-1"]
    val_segs = [s for s in segs if s.session == "ses-1"]
    print(f"\n{len(fit_segs)} fitting segments, {len(val_segs)} validation segments, "
          f"{sum(s.y.shape[0] for s in fit_segs) / FS / 60:.1f} min of fitting data")

    train = accumulate_by_subject(fit_segs)
    val = accumulate_by_subject(val_segs)
    print(f"accumulated {sum(t['n'] for t in train.values()):,} training samples over "
          f"{len(train)} participants x {next(iter(train.values()))['n_ch']} channels, "
          f"design dimension {next(iter(train.values()))['xtx'].shape[0]}")

    theta, r_val, lam = search_chain(train, val)
    print("\nchain parameters after forging against the validation sessions.  the chain "
          "is ONE object across all seven participants; only the lead field is per person.")
    for k in sorted(THETA0):
        mark = "  <- moved" if theta[k] != THETA0[k] else ""
        print(f"  {k:16s} {THETA0[k]:8.4f} -> {theta[k]:8.4f}{mark}")
    print(f"  driven: validation r2 {r_val:+.5f} at ridge {lam:g}")
    r_val_d, lam_d = _sweep_direct(train, val)
    print(f"  teacher-direct: validation r2 {r_val_d:+.5f} at ridge {lam_d:g}")

    proj = _projection(tap_basis(theta), N_PLACES)
    w_driven = fit_lead_fields(train, lam, proj)

    # the calibration used *during* fitting: measured on the validation sessions,
    # which are held out of the statistics above.  the figure quoted as the
    # forcing's accuracy is the stricter one computed in `eval`, on participants
    # and material that were both unseen.
    fits = []
    print("\ncalibration r2, per band, on the validation sessions:")
    for b in CAL_BANDS:
        r, n = band_r2(val_segs, w_driven, b, proj)
        fits.append(BandFit(b, float(r), n_targets=next(iter(train.values()))["n_ch"],
                            n_samples=int(n),
                            held_out="validation sessions of training participants"))
        print(f"  {fits[-1]}")
    cal = ForcingCalibration(
        component="transduction.hair_cell", bands=tuple(fits),
        measured_on="meg-masc, 7 participants, held-out sessions",
        front_end="gammatone_cochleagram",
        note="the r2 is measured on MEG, several processes and one unknown lead field "
             "downstream of the component being written; it bounds rather than states the "
             "front end's own fidelity")

    np.savez(CACHE / "fit.npz", theta=json.dumps(theta), lam=lam, lam_d=lam_d,
             cal=json.dumps([[b.band.lo_hz, b.band.hi_hz, b.r2, b.n_samples] for b in fits]))
    print("\n" + cal.describe())
    for b in CAL_BANDS:
        print(f"  shrinkage in {b}: {cal.shrinkage(b):.4f}   (a hard clamp would be 1.0000)")


def _fit_and_test(fit_segs: list[Segment], test_segs: list[Segment], proj: np.ndarray,
                  lam: float, lam_d: float, minutes: float | None = None
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
            have = used.get(s.subject, 0.0)
            if have >= cut:
                continue
            take = int(min(s.y.shape[0], cut - have))
            used[s.subject] = have + take
            kept.append(Segment(s.subject, s.session, s.task, s.story, s.wav,
                                s.y[:take], s.c[:take], s.loc))
        fit_segs = kept
    st = accumulate_by_subject(fit_segs)
    w_dr = fit_lead_fields(st, lam, proj)
    w_te = fit_lead_fields(st, lam_d, None)
    for s in {x.subject for x in test_segs}:
        w_dr.setdefault(s, None)
        w_te.setdefault(s, None)
    minutes_used = sum(x.y.shape[0] for x in fit_segs) / FS / 60.0 / max(len(st), 1)
    return (predict_r2(test_segs, w_dr, proj), predict_r2(test_segs, w_te),
            minutes_used)


def stage_eval() -> None:
    ibm.load_all(seal=True, strict=True)
    cochs = _corpus()
    meg = CACHE / "meg"
    d = np.load(CACHE / "fit.npz", allow_pickle=False)
    theta = json.loads(str(d["theta"]))
    lam, lam_d = float(d["lam"]), float(d["lam_d"])
    proj = _projection(tap_basis(theta), N_PLACES)
    cal_rows = json.loads(str(d["cal"]))
    cal = ForcingCalibration(
        component="transduction.hair_cell",
        bands=tuple(BandFit(Band(lo, hi), r2, n_samples=int(n)) for lo, hi, r2, n in cal_rows),
        measured_on="meg-masc, 7 participants, held-out sessions",
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
         dict(subjects=train_subj, exclude_stories=(HELD_OUT_STORY,)),
         dict(subjects=train_subj, stories=(HELD_OUT_STORY,))),
        ("held-out participants, seen stories",
         dict(subjects=HELD_OUT_SUBJECTS, exclude_stories=("lw1", HELD_OUT_STORY)),
         dict(subjects=HELD_OUT_SUBJECTS, stories=("lw1",))),
        ("held-out participants x held-out story",
         dict(subjects=HELD_OUT_SUBJECTS, exclude_stories=(HELD_OUT_STORY,)),
         dict(subjects=HELD_OUT_SUBJECTS, stories=(HELD_OUT_STORY,))),
    ]
    for name, fit_kw, test_kw in rows:
        fit_segs = load_segments(meg, cochs, **fit_kw)
        test_segs = load_segments(meg, cochs, **test_kw)
        if not fit_segs or not test_segs:
            print(f"{name:44s}  no segments")
            continue
        r_dr, r_te, _ = _fit_and_test(fit_segs, test_segs, proj, lam, lam_d)
        per_channel[name] = (r_dr, r_te, len(test_segs))
        results[name] = dict(driven=float(np.mean(r_dr)), teacher=float(np.mean(r_te)),
                             driven_best=float(np.max(r_dr)),
                             teacher_best=float(np.max(r_te)),
                             n_test_segments=len(test_segs))
        ratio = float(np.mean(r_dr)) / max(float(np.mean(r_te)), 1e-12)
        print(f"{name:44s} {r_dr.size:4d} {0.0:9.5f} {float(np.mean(r_dr)):9.5f} "
              f"{float(np.mean(r_te)):9.5f} {ratio:15.2f}")
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
                             exclude_stories=(HELD_OUT_STORY,))
    test_segs = load_segments(meg, cochs, subjects=HELD_OUT_SUBJECTS,
                              stories=(HELD_OUT_STORY,))
    curve = []
    for minutes in (2.0, 5.0, 10.0, 20.0, 40.0, None):
        r_dr, r_te, used = _fit_and_test(fit_segs, test_segs, proj, lam, lam_d, minutes)
        label = f"{used:.1f}" if minutes is not None else f"{used:.1f} (all)"
        ratio = float(np.mean(r_dr)) / max(float(np.mean(r_te)), 1e-12)
        curve.append((used, float(np.mean(r_dr)), float(np.mean(r_te))))
        print(f"{label:>26s} {float(np.mean(r_dr)):9.5f} {float(np.mean(r_te)):9.5f} "
              f"{ratio:15.2f}")

    # the same comparison restricted to sensors that are downstream of the forced
    # region rather than over it.  "downstream" is defined on TRAINING
    # participants only -- the sensors whose own direct kernel explains the least
    # -- and applied unchanged to held-out participants, so the definition cannot
    # have been tuned on the answer.
    tr_fit = load_segments(meg, cochs, subjects=train_subj,
                           exclude_stories=(HELD_OUT_STORY,))
    tr_test = load_segments(meg, cochs, subjects=train_subj, stories=(HELD_OUT_STORY,))
    _, r_train, _ = _fit_and_test(tr_fit, tr_test, proj, lam, lam_d)
    order = np.argsort(r_train)
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
                  f"{float(np.mean(r_dr[idx])):9.5f} {float(np.mean(r_te[idx])):9.5f}")

    # the calibration the forcing actually earns, on the strictest split available.
    strict_fit = load_segments(meg, cochs, subjects=HELD_OUT_SUBJECTS,
                               exclude_stories=(HELD_OUT_STORY,))
    strict_test = load_segments(meg, cochs, subjects=HELD_OUT_SUBJECTS,
                                stories=(HELD_OUT_STORY,))
    w_strict = fit_lead_fields(accumulate_by_subject(strict_fit), lam, proj)
    fits = []
    for band in CAL_BANDS:
        r, n = band_r2(strict_test, w_strict, band, proj)
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

    #: sessions 1-7 fit the lead field, 8 selects the ridge, 9-12 are held out.
    #: chosen by number before anything was run.  the middle session exists so
    #: that no ridge is ever selected on the test sessions -- the shortcut that
    #: would flatter both models and flatter the more flexible one more.
    fit = [s for s in segs if int(s.session.split("-")[1]) <= 7]
    val_segs = [s for s in segs if int(s.session.split("-")[1]) == 8]
    test = [s for s in segs if int(s.session.split("-")[1]) > 8]
    print(f"  {len(fit)} sessions fitting, {len(val_segs)} selecting the ridge, "
          f"{len(test)} held out")

    tr = accumulate(fit)
    val = accumulate(val_segs)
    a_ = proj.T @ tr["xtx"] @ proj
    b_ = proj.T @ tr["xty"]
    best, w_best, lam_best = -np.inf, None, None
    for lam in _RIDGES:
        w = ridge_solve(a_, b_, lam)
        r = float(np.mean(fitted_r2(val, w, proj)))
        if r > best:
            best, w_best, lam_best = r, w, lam
    r_driven = float(np.mean(predict_r2(test, w_best, proj)))
    print(f"\nheld-out sessions, chain frozen at the meg-masc posterior, lead field "
          f"refitted: r2 = {r_driven:+.5f} (ridge {lam_best:g})")

    best_d, w_direct = -np.inf, None
    for lam in _RIDGES:
        w = ridge_solve(tr["xtx"], tr["xty"], lam)
        r = float(np.mean(fitted_r2(val, w)))
        if r > best_d:
            best_d, w_direct = r, w
    print(f"teacher-direct on the same split: r2 = "
          f"{float(np.mean(predict_r2(test, w_direct))):+.5f}")

    fits = []
    for band in CAL_BANDS:
        r, n = band_r2(test, w_best, band, proj)
        fits.append(BandFit(band, float(r), n_targets=tr["n_ch"], n_samples=int(n),
                            held_out="sessions 9-12 of one participant"))
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
        if abs(a - 1.0) > 0.01:
            print(f"  {ses}: clock ratio {a:.5f} is more than 1% off unity; skipped rather "
                  "than resampled, because a ratio that large means the pairing is wrong")
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

        i0 = int(round(float(b) * FS))
        m = min(c.shape[1], y.shape[1] - i0)
        if m < int(60 * FS):
            continue
        segs.append(Segment("sub-0", ses, "Sherlock1", f"chapter{n:02d}", wav.name,
                            y[:, i0:i0 + m].T.copy(), c[:, :m].T.copy(),
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
