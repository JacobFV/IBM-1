"""measure the catalogue's scorable rows on ds008037 resting EEG, as distributions over subjects.

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        .venv/bin/python scripts/measure_rhythms_eeg.py --limit 40

WHAT THIS IS FOR
----------------
`ibm/rhythms.py` declares 53 rhythms.  Most carry `evidence="literature"`: a band and an
interval copied from published physiology, with nothing local behind them.  This script
takes the five rows that a resting scalp recording can actually speak to -- occipital
alpha, sensorimotor mu, the 1/f background, frontal-midline theta, posterior beta -- and
measures them on 121 subjects, so the catalogue can either keep its declared interval
with a local number behind it or be told plainly that the data does not support it.

The model side and the data side go through the SAME instrument: every number below
comes out of `ibm/spectral.py` (`welch_psd`, `aperiodic_fit`, `peak_frequency`,
`peak_prominence`, `relative_band_power`).  `mne` is used to READ the .set files and for
nothing else -- no scipy, no mne spectral function touches a measurement.  The one
piece of signal processing written here is the high-pass (`_highpass_fft`), which is
preprocessing, not measurement, and which carries its own known-answer check.

THE FIVE MEASUREMENTS, and the catalogue row each speaks to
-----------------------------------------------------------
1. occipital alpha  peak_frequency + peak_prominence, 8-13 Hz, over O/PO  -> `alpha_occipital`
2. sensorimotor mu  peak_frequency + peak_prominence, 8-13 Hz, over C/CP  -> `mu`
3. aperiodic exponent, 1-45 Hz, 8-13 and 45-55 excluded, global and per group -> `BACKGROUND`
4. frontal-midline theta prominence, 4-8 Hz, over Fz/F1/F2/FC1/FC2 -> `frontal_midline_theta`
5. posterior beta prominence, 13-30 Hz, over parieto-occipital -> NO ROW EXISTS (see below)

METHOD DECISIONS, each defended in one line
-------------------------------------------
* **RE-REFERENCE: average reference over the retained channels.**  The two artefacts
  disagree about the reference and the disagreement is exactly CLAUDE.md's "check what
  they each declare about the shared quantity": `*_channels.tsv` says every channel is
  referenced to FCz, but FCz is PRESENT in the data with a full-amplitude signal
  (sub-001: 51.7 uV rms, r = 0.99 with Cz) and the EEGLAB header says `ref = 'Common'`
  with a `loadcurry` history and no re-referencing step -- a channel referenced to
  itself would be exactly zero, so the sidecar's reference column cannot be read
  literally.  An average reference makes the answer not depend on which of the two is
  right, and, decisively for row 2, a fronto-central reference sits ON the sensorimotor
  strip: keeping FCz would subtract mu from C3/C4/Cz, which is the one row this script
  exists to measure.  The cost is stated, not hidden: an average reference over a
  62-channel cap with no inferior coverage is not a physical zero, it removes whatever
  is common to all channels, and it makes every channel's value depend weakly on every
  other -- so a number here is comparable to another number here and not to a
  mastoid-referenced paper.
* **WELCH: 4 s Hann segments, 50% overlap (df = 0.25 Hz).**  The alpha peak is a ~1 Hz-wide
  feature, and a Hann taper's effective noise bandwidth is 1.5*df, so 4 s gives 0.375 Hz
  of smoothing and 4-5 independent-ish bins across the peak; the 2 s segment
  `spectral_loss` defaults to gives 0.75 Hz of smoothing and would flatten the very bump
  `peak_prominence` is asked about.  ~247 segments per subject keeps the chi-square
  variance of each bin small.
* **ARTEFACT REJECTION: a 4 s segment is dropped, for ALL channels at once, if any
  retained channel's peak-to-peak within it exceeds `--ptp-k` (default 3.0) times that
  CHANNEL's own median segment peak-to-peak.**  One criterion, one number, reported per
  subject as `rejected_fraction`.  Relative and not absolute, because an absolute
  threshold measures the amplifier: probed before the run, a fixed 250 uV kept 0.4% of
  sub-001's segments and 56% of sub-005's, which is a criterion that rejects loud
  SUBJECTS rather than bad segments.  The relative form keeps 96%/99%/62%/67% on
  sub-001/005/007/008 -- it fires differentially on the subjects that have artefact,
  which is what it is for.  Its own cost is stated: it is stricter in microvolts for a
  quiet subject, so `median_segment_ptp_uv` is reported per subject and known answer (g)
  re-measures one subject with rejection OFF so the size of the effect is on the record.
  It is applied jointly across channels rather than per channel so that every group's PSD
  is averaged over the SAME segments -- a per-channel mask would make the occipital and
  sensorimotor numbers averages over different stretches of the recording and their
  difference would be partly a difference of time windows.  A subject whose kept fraction
  falls below `--min-kept` (0.5) is EXCLUDED with that reason recorded, never silently
  trimmed.
* **HIGH-PASS: yes, zero-phase FFT high-pass, stop 0.25 Hz, pass 0.5 Hz.**  These files
  are raw Curry exports (`SoftwareFilters: n/a`, `low_cutoff: n/a`) and carry ~1e3 uV^2/Hz
  at 0.2-0.5 Hz against ~7 uV^2/Hz at 10 Hz -- five orders of magnitude.  That drift is
  not in the 1-45 Hz fit band, but it makes the peak-to-peak artefact criterion fire on
  slow drift instead of on artefact, and it leaks through the Hann skirts.  The corner is
  put BELOW the catalogue's own 1 Hz fit floor so the filter cannot shape any band that
  is measured.  Zero-phase and even-gain, so it is idempotent and adds no phase.

THINGS THAT WOULD MAKE A NUMBER LOOK RIGHT FOR THE WRONG REASON -- read before quoting
--------------------------------------------------------------------------------------
* **THIS CORPUS IS EYES-OPEN.**  All 121 `*_task-rest_eeg.json` sidecars say
  `"EyesCondition": "eyes open"`.  `alpha_occipital` is declared in state
  `wake-eyes-closed` and its target interval (0.4, 1.5) decades is an eyes-CLOSED number.
  Measuring eyes-open alpha against an eyes-closed declaration is the ledger's recurring
  shape -- a quantity computed correctly and compared against the wrong population -- so
  this script does NOT propose to confirm that row.  `mu` is declared in `wake-rest`,
  which is what this corpus is, so mu is the row that can honestly move.
* **Marker files are not uniform.**  114 of 121 rest recordings carry one 200001 and one
  200002; sub-105/-107 carry two starts, sub-128 two ends, sub-013/-043/-047/-085 a start
  and no end, and sub-147/-148/-149 have NO events file at all.  The rule is fixed in
  `measure_subject` (last start before the first end; first end; fall back to the end of
  the recording when there is no end mark or no file) and each subject's
  `marker_rule` is recorded, so a subject analysed on a fallback is visible rather than
  pooled in silently.
* **The events file marks only the recording's start and end** (values 200001, 200002).
  There is no eyes-open/eyes-closed marker, no task marker, no trial structure.  Every
  `state_contrast`, `load_slope` and `evoked_band` row in the catalogue is therefore OUT
  OF SCOPE here, including `frontal_midline_theta`'s declared `load_slope` measure: what
  is measured below is resting 4-8 Hz prominence, which is a different quantity from the
  theta-against-set-size slope that row declares, and it cannot confirm it.
* **A peak frequency without a prominence beside it is the band you chose.**
  `peak_frequency` is a soft-argmax that returns the band CENTRE (10.5 Hz for 8-13) on a
  flat spectrum -- known answer (d) below prints exactly that.  Never quote an
  `*_peak_hz` from this file without its `*_prominence`.
* **`aperiodic_fit` weights LINEAR bins uniformly**, so with df = 0.25 Hz the 13-45 Hz
  stretch contributes 128 of the 176 fitted bins and the exponent is mostly the
  high-frequency slope.  That is what FOOOF does with the same input and it is fine for
  comparing this spectrum against the model's on the same grid; it is NOT the same number
  as a paper that fitted log-spaced bins.  Say which you did.
* **`45-55 Hz excluded` removes exactly one bin** when the fit stops at 45 Hz.  It is kept
  because the task named it and because it documents the intent, but the thing that
  actually keeps the 50 Hz line out of the fit is `hi = 45`.  Known answer (f) confirms
  the line is at 50 Hz and not 60 -- a known answer about the DATA, not the instrument.
* **MEASURED, and it is the most important caveat in this file: the occipital and
  sensorimotor 8-13 Hz numbers are NOT two independent measurements.**  Over 119
  subjects, r(occipital alpha prominence, sensorimotor mu prominence) = **+0.836**
  [+0.784, +0.878] and r of the two peak frequencies = **+0.732** [+0.625, +0.820] --
  70% of the variance is shared.  The paired difference goes the wrong way for two
  separate generators: central prominence is HIGHER than occipital by 0.063 +/- 0.020
  decades (95% CI [-0.102, -0.024] for occ - smr, excluding zero).  At the scalp, with
  an average reference, a posterior alpha source spreads onto C3/C4/Cz and the average
  reference subtracts the dominant source from everyone, so this is what volume
  conduction predicts.  A scalp 8-13 Hz prominence over the central electrodes is
  therefore NOT evidence of a separate sensorimotor mu generator, and the `mu` row must
  not be moved to `measured-here` on it.  Separating mu from alpha needs a spatial filter
  (CSD/Laplacian, ICA or a source model) or the movement contrast the row's sibling
  `mu_erd` declares -- neither of which this script does and neither of which this corpus
  supports.
* **Prominence is a log10 ratio against a fitted background**, so it is insensitive to
  amplifier gain and to the average reference's overall scale -- but it is NOT insensitive
  to the background's shape, and a subject whose 1/f is bent by a broad artefact will show
  a prominence that is about the artefact.  `exponent_*` is reported beside every
  prominence for that reason.

DISCIPLINE
----------
* **Split.**  One seeded `numpy.random.Generator` (`--split-seed`, default 20260918) draws
  one permutation of ALL 121 subjects that have a rest recording, in `participants.tsv`
  order -- the ordering the dataset itself defines, not a directory walk.  First half is
  DECLARATION, second half HELD-OUT.  Every proposed interval is the 10th-90th percentile
  of the DECLARATION half only; the held-out half is then scored for coverage, whose
  pre-registered expectation under exchangeability is 80%, with binomial SE
  sqrt(.8*.2/60) = 5.2%.  Because the split is drawn over all 121 ids up front, a partial
  run's split is a strict subset of the full run's and does not move as subjects are added.
* **Bootstrap over SUBJECTS.**  10,000 percentile-bootstrap resamples of the subject list,
  from one generator drawn at the top and passed in (`_bootstrap` asserts it received
  one).  Segments are not items: 247 segments of one subject are one subject.
* **Known answers, printed with their numbers.**  (a) an 11.3 Hz sinusoid injected into a
  real recording at a power set to 1x and 4x its own alpha band power, predicting
  peak_frequency 11.3 and prominence +log10(2) and +log10(5); (b) a white-noise surrogate
  of the same length through the same pipeline, predicting exponent 0 and prominence 0;
  (c) IDEMPOTENCE -- one subject measured twice, compared for bit-identity; (d) a flat
  spectrum, predicting peak_frequency = band centre; (e) Parseval, psd.sum()*df = var(x);
  (f) the line frequency read off the data, predicting 50 Hz; (g) the same subject with
  artefact rejection switched OFF, so the size of the preprocessing choice is a number.
* **The summary JSON is rewritten after EVERY subject**, with `default=` coercing numpy
  types, so a crash leaves the record of everything measured so far.  A subject that
  cannot be processed goes in `exclusions` WITH ITS REASON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ibm.spectral import (  # noqa: E402
    aperiodic_fit,
    band_power,
    peak_frequency,
    peak_prominence,
    relative_band_power,
    welch_psd,
)

CORPUS = REPO / "data" / "sources" / "ds008037" / "raw"
OUT = REPO / "out" / "rhythms_eeg_ds008037.json"

# channel names as they appear in these files: UPPERCASE.  The requested sets, intersected
# with the montage at run time and reported as `channels_used` so a missing electrode is
# visible rather than silently dropping out of the average.
GROUPS: Dict[str, Tuple[str, ...]] = {
    # 1. occipital alpha.  Oz/POz are spelled OZ/POZ here.
    "occipital": ("O1", "O2", "OZ", "PO3", "PO4", "POZ", "PO7", "PO8"),
    # 2. sensorimotor mu.
    "sensorimotor": ("C3", "C4", "CZ", "CP3", "CP4"),
    # 4. frontal midline theta.  FCz is the nominal reference and AFz is ground, so
    #    neither is a data channel in the BIDS sense; Fz/F1/F2/FC1/FC2 is the requested set.
    "frontal_midline": ("FZ", "F1", "F2", "FC1", "FC2"),
    # 5. posterior beta.  "posterior" is defined here as the occipital set plus the
    #    parietal row, and it is defined HERE because no catalogue row declares it.
    "posterior": ("O1", "O2", "OZ", "PO3", "PO4", "POZ", "PO7", "PO8",
                  "P1", "P2", "P3", "P4", "PZ"),
}

ALPHA = (8.0, 13.0)
THETA = (4.0, 8.0)
BETA = (13.0, 30.0)
FIT = (1.0, 45.0)
LINE = (45.0, 55.0)
# what the catalogue's BACKGROUND row itself excludes, measured alongside the instructed
# exclusion so the patch below compares like with like.
CATALOGUE_EXCLUDE = ((8.0, 13.0), (0.5, 1.5))
INSTRUCTED_EXCLUDE = ((8.0, 13.0), (45.0, 55.0))


# ======================================================================================
# preprocessing.  the only signal processing written in this file.
# ======================================================================================
def _highpass_fft(x: torch.Tensor, fs: float, f_stop: float, f_pass: float,
                  chunk: int = 16) -> torch.Tensor:
    """zero-phase FFT high-pass with a raised-cosine transition from `f_stop` to `f_pass`.

    Even, real gain applied to the rFFT, so the filter has exactly zero phase and applying
    it twice is the same map applied twice (it is idempotent on its own output only up to
    the square of the gain -- the idempotence that matters here is that `measure_subject`
    called twice on the same file returns the same bits, which known answer (c) checks).

    Chunked over channels because the rFFT of 62 x 5e5 float64 is 246 MB of complex128 and
    this machine has a training job on it.
    """
    T = x.shape[-1]
    f = torch.fft.rfftfreq(T, d=1.0 / float(fs), dtype=x.dtype, device=x.device)
    ramp = ((f - f_stop) / (f_pass - f_stop)).clamp(0.0, 1.0)
    gain = 0.5 * (1.0 - torch.cos(math.pi * ramp))
    out = torch.empty_like(x)
    for i in range(0, x.shape[0], chunk):
        X = torch.fft.rfft(x[i:i + chunk], dim=-1)
        X = X * gain
        out[i:i + chunk] = torch.fft.irfft(X, n=T, dim=-1)
        del X
    return out


def _robust_sd(x: torch.Tensor) -> torch.Tensor:
    """median absolute deviation * 1.4826, per channel.  Robust to the very artefacts the
    segment criterion is about to reject, which a plain sd is not."""
    med = x.median(dim=-1, keepdim=True).values
    return 1.4826 * (x - med).abs().median(dim=-1).values


# ======================================================================================
# reading.  mne reads the bytes; nothing else.
# ======================================================================================
def _read_channels_tsv(path: Path) -> List[Tuple[str, str]]:
    rows = path.read_text().strip().split("\n")
    head = rows[0].split("\t")
    i_name, i_status = head.index("name"), head.index("status")
    return [(r.split("\t")[i_name], r.split("\t")[i_status]) for r in rows[1:]]


def _read_events_tsv(path: Path) -> List[Tuple[float, str]]:
    rows = path.read_text().strip().split("\n")
    head = rows[0].split("\t")
    i_on, i_val = head.index("onset"), head.index("value")
    out = []
    for r in rows[1:]:
        c = r.split("\t")
        out.append((float(c[i_on]), c[i_val].strip()))
    return out


def subject_ids(corpus: Path) -> Tuple[List[str], Dict[str, str]]:
    """the subject list, in `participants.tsv` order -- the ordering the DATASET defines.

    CLAUDE.md: take an ordering from the dataset that defines it, never from a
    reconstruction that happens to be the right length.  A directory walk would give the
    same 121 ids here and there is no reason to trust that it always would.  Ids in
    participants.tsv with no rest recording, and rest recordings with no participants row,
    are both returned as exclusions with their reason.
    """
    rows = (corpus / "participants.tsv").read_text().strip().split("\n")
    head = rows[0].split("\t")
    i_id = head.index("participant_id")
    listed = [r.split("\t")[i_id] for r in rows[1:]]
    ids, excl = [], {}
    for s in listed:
        p = corpus / s / "eeg" / f"{s}_task-rest_eeg.set"
        if p.exists():
            ids.append(s)
        else:
            excl[s] = "listed in participants.tsv but has no *_task-rest_eeg.set on disk"
    on_disk = sorted(p.parent.parent.name for p in corpus.glob("sub-*/eeg/*_task-rest_eeg.set"))
    for s in on_disk:
        if s not in listed:
            excl[s] = "has a *_task-rest_eeg.set on disk but no participants.tsv row"
    return ids, excl


# ======================================================================================
# the measurement.  a pure function of its arguments -- known answer (c) calls it twice.
# ======================================================================================
def measure_subject(
    sid: str,
    corpus: Path,
    nperseg_s: float = 4.0,
    ptp_k: float = 3.0,
    min_kept: float = 0.5,
    edge_s: float = 2.0,
    hp_stop: float = 0.25,
    hp_pass: float = 0.5,
    reject: bool = True,
    inject: Optional[Tuple[str, float, float]] = None,
    surrogate_seed: Optional[int] = None,
    surrogate_sd: float = 10.0,
) -> Dict:
    """measure one subject.  Returns a dict of plain floats, or raises with a reason.

    `inject` is `(group, freq_hz, amplitude_uv)` -- a sinusoid added to that group's
    channels AFTER re-referencing.  After, not before: a signal added to every channel is
    removed EXACTLY by an average reference, so a before-reference injection into all
    channels would test nothing, and an injection into a subset before the reference would
    be attenuated by a factor that depends on the subset size.
    `surrogate_seed` replaces the data with white noise of the SAME shape, drawn from a
    generator built from that integer and nothing else.  It is an int rather than a tensor
    so this stays a pure function of its arguments -- passing a live generator in would
    advance its state and the second call would not return the first call's answer, which
    is exactly what known answer (c) is there to catch.  Both default to None and the real
    path never draws at all.
    """
    import mne  # imported here so the module can be read without mne installed
    mne.set_log_level("ERROR")

    eeg = corpus / sid / "eeg"
    setf = eeg / f"{sid}_task-rest_eeg.set"
    if not setf.exists():
        raise FileNotFoundError("no *_task-rest_eeg.set")

    status = dict(_read_channels_tsv(eeg / f"{sid}_task-rest_channels.tsv"))
    evf = eeg / f"{sid}_task-rest_events.tsv"
    # sub-147, sub-148 and sub-149 have a rest recording and NO events file at all.  This
    # branch was added after the first full pass found them; it extends the missing-end-mark
    # fallback below to a missing FILE and it cannot change any already-measured value,
    # because it only fires where there was no measurement.  Recorded here rather than
    # applied quietly.
    ev = _read_events_tsv(evf) if evf.exists() else []
    starts = sorted(float(t) for t, v in ev if v == "200001")
    ends = sorted(float(t) for t, v in ev if v == "200002")
    no_marks = not starts
    # The markers are not always a clean pair.  Across the 121 rest files: 114 are one
    # start and one end; sub-105 and sub-107 have TWO starts before one end (34.45/39.05 s
    # and 12.04/16.39 s -- a restarted mark); sub-128 has two ends 0.15 s apart; sub-013,
    # -043, -047 and -085 have a start and NO end at all.  The rule, fixed here rather
    # than per subject: take the LAST start that precedes the first end (the conservative
    # one, which drops the ambiguous lead-in), take the FIRST end, and when there is no end
    # mark fall back to the end of the recording -- these are task-rest files with nothing
    # after the rest period, and dropping four subjects for a missing mark would be a
    # selection on the marker file rather than on the data.
    t1 = min(ends) if ends else None
    t0 = max([t for t in starts if t1 is None or t < t1], default=(starts[0] if starts else 0.0))
    marker_rule = ("no events file: the whole recording is analysed" if not evf.exists() else
                   "no 200001 mark: the whole recording is analysed" if no_marks else
                   "start+end" if len(starts) == 1 and len(ends) == 1 else
                   "no-end-mark: trimmed to the end of the recording" if not ends else
                   f"{len(starts)} start / {len(ends)} end marks: last start before the "
                   f"first end, first end")

    raw = mne.io.read_raw_eeglab(str(setf), preload=True)
    fs = float(raw.info["sfreq"])
    if t1 is None:
        t1 = raw.n_times / fs
    if t1 - t0 < 60.0:
        raise ValueError(f"rest period is only {t1 - t0:.1f} s between the start and end marks")
    names = [c.upper() for c in raw.ch_names]
    i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
    i1 = min(i1, raw.n_times)
    x = torch.from_numpy(np.ascontiguousarray(raw.get_data()[:, i0:i1] * 1e6))  # volts -> uV
    del raw
    if surrogate_seed is not None:
        g = torch.Generator().manual_seed(int(surrogate_seed))
        x = torch.randn(x.shape, generator=g, dtype=x.dtype) * float(surrogate_sd)

    # -- channels declared bad by the dataset ------------------------------------------
    tsv_bad = [n for n in names if status.get(n, "good") != "good"]
    keep = [i for i, n in enumerate(names) if status.get(n, "good") == "good"]
    if len(keep) < 20:
        raise ValueError(f"only {len(keep)} channels marked good")
    x = x[keep]
    names = [names[i] for i in keep]

    # -- high-pass, then discard the circular-filter edges ------------------------------
    x = x - x.mean(dim=-1, keepdim=True)
    x = _highpass_fft(x, fs, hp_stop, hp_pass)
    e = int(round(edge_s * fs))
    x = x[:, e:x.shape[-1] - e] if x.shape[-1] > 4 * e else x
    T = x.shape[-1]

    # -- channels the DATA says are bad.  The tsv marks nothing bad anywhere in this
    #    corpus (7,502 of 7,502 channels "good"), so a status-only filter is a control
    #    that cannot fail; this is the one that can.
    sd = _robust_sd(x)
    med = float(sd.median())
    dead = (sd < 1e-3) | (sd > 5.0 * med) | (sd < 0.2 * med)
    data_bad = [names[i] for i in range(len(names)) if bool(dead[i])]
    live = [i for i in range(len(names)) if not bool(dead[i])]
    if len(live) < 20:
        raise ValueError(f"only {len(live)} channels survive the robust-sd check")
    x = x[live]
    names = [names[i] for i in live]

    # -- average reference --------------------------------------------------------------
    x = x - x.mean(dim=0, keepdim=True)

    # -- segment, and decide which segments survive, jointly over channels ---------------
    #    the keep mask is computed on the UNINJECTED data, so the injection known answer
    #    compares two runs over the same segments and cannot move the answer by moving the
    #    mask.
    nperseg = int(round(nperseg_s * fs))
    step = nperseg // 2
    if T < 4 * nperseg:
        raise ValueError(f"only {T / fs:.1f} s survive trimming; need at least {4 * nperseg / fs:.0f} s")
    seg = x.unfold(-1, nperseg, step)                     # (C, M, nperseg), a view
    M = seg.shape[1]
    ptp = torch.empty(len(names), M, dtype=x.dtype)
    for i in range(0, len(names), 8):
        s = seg[i:i + 8]
        ptp[i:i + 8] = s.amax(dim=-1) - s.amin(dim=-1)
    med_ptp = ptp.median(dim=1, keepdim=True).values
    if reject:
        good_seg = ~((ptp > float(ptp_k) * med_ptp).any(dim=0))
    else:
        good_seg = torch.ones(M, dtype=torch.bool)
    n_keep = int(good_seg.sum())
    rejected = 1.0 - n_keep / float(M)
    if n_keep / float(M) < min_kept:
        raise ValueError(f"artefact rejection kept only {n_keep}/{M} segments "
                         f"({100 * n_keep / M:.1f}%, below {100 * min_kept:.0f}%)")
    kidx = torch.nonzero(good_seg, as_tuple=False).flatten()

    if inject is not None:
        gname, f_hz, amp = inject
        idx = [names.index(c) for c in GROUPS[gname] if c in names]
        t = torch.arange(T, dtype=x.dtype) / fs
        x[idx] = x[idx] + amp * torch.sin(2.0 * math.pi * float(f_hz) * t)
        seg = x.unfold(-1, nperseg, step)                 # rebuilt after the in-place add

    # -- Welch over the kept segments, chunked over channels ----------------------------
    psd_rows = []
    freqs = None
    for i in range(0, len(names), 8):
        s = seg[i:i + 8].index_select(1, kidx).contiguous()
        f, p = welch_psd(s, fs, nperseg=nperseg, noverlap=0)
        psd_rows.append(p.mean(dim=1))                     # average over kept segments
        freqs = f
        del s, p
    psd = torch.cat(psd_rows, dim=0)                       # (C, F)
    del psd_rows, seg, x

    # -- group spectra.  Average POWER across a group's channels, not the signals: two
    #    channels carrying the same rhythm at opposite polarity cancel in the signal
    #    average and add in the power average.
    def group_psd(g: str) -> Tuple[torch.Tensor, List[str]]:
        ch = [c for c in GROUPS[g] if c in names]
        if not ch:
            raise ValueError(f"no channel of group {g} survives")
        return psd[[names.index(c) for c in ch]].mean(dim=0), ch

    occ, occ_ch = group_psd("occipital")
    smr, smr_ch = group_psd("sensorimotor")
    fmt, fmt_ch = group_psd("frontal_midline")
    post, post_ch = group_psd("posterior")
    glob = psd.mean(dim=0)

    def prom(p: torch.Tensor, band: Tuple[float, float], extra=()) -> float:
        """prominence of `band` against a 1-45 Hz background fitted with `band`, the alpha
        band and the line band all left out -- so neither the peak under test nor the alpha
        peak nor the mains line can bend the background they are measured against."""
        ex = [band, ALPHA, LINE] + list(extra)
        ex = [b for i, b in enumerate(ex) if b not in ex[:i]]
        return float(peak_prominence(p, freqs, band[0], band[1],
                                     exclude_for_background=ex, fit_lo=FIT[0], fit_hi=FIT[1]))

    def expo(p: torch.Tensor, exclude, lo: float = FIT[0], hi: float = FIT[1]) -> float:
        return float(aperiodic_fit(p, freqs, lo, hi, exclude=list(exclude))[1])

    r = {
        "subject": sid,
        # --- 1. occipital alpha
        "occ_alpha_peak_hz": float(peak_frequency(occ, freqs, *ALPHA)),
        "occ_alpha_prominence": prom(occ, ALPHA),
        "occ_alpha_rel_power": float(relative_band_power(occ, freqs, *ALPHA)),
        # --- 2. sensorimotor mu
        "smr_mu_peak_hz": float(peak_frequency(smr, freqs, *ALPHA)),
        "smr_mu_prominence": prom(smr, ALPHA),
        "smr_mu_rel_power": float(relative_band_power(smr, freqs, *ALPHA)),
        # --- 3. the 1/f background
        "exponent_global": expo(glob, INSTRUCTED_EXCLUDE),
        "exponent_occipital": expo(occ, INSTRUCTED_EXCLUDE),
        "exponent_sensorimotor": expo(smr, INSTRUCTED_EXCLUDE),
        "exponent_frontal_midline": expo(fmt, INSTRUCTED_EXCLUDE),
        "exponent_global_catalogue_exclusions": expo(glob, CATALOGUE_EXCLUDE),
        # sensitivity, NOT the headline: `aperiodic_fit` weights linear bins uniformly, so
        # 13-45 Hz supplies 128 of the 176 bins in the 1-45 fit and the headline exponent is
        # mostly the high-frequency slope -- which in eyes-open scalp EEG is where broadband
        # muscle sits.  Splitting the range says whether a flat headline number is the
        # spectrum or the EMG.
        "exponent_global_1_20": expo(glob, INSTRUCTED_EXCLUDE, 1.0, 20.0),
        "exponent_global_20_45": expo(glob, INSTRUCTED_EXCLUDE, 20.0, 45.0),
        "exponent_occipital_1_20": expo(occ, INSTRUCTED_EXCLUDE, 1.0, 20.0),
        "exponent_sensorimotor_1_20": expo(smr, INSTRUCTED_EXCLUDE, 1.0, 20.0),
        "exponent_frontal_midline_1_20": expo(fmt, INSTRUCTED_EXCLUDE, 1.0, 20.0),
        # --- 4. frontal midline theta
        "fm_theta_prominence": prom(fmt, THETA),
        "fm_theta_peak_hz": float(peak_frequency(fmt, freqs, *THETA)),
        # --- 5. posterior beta
        "post_beta_prominence": prom(post, BETA),
        "post_beta_peak_hz": float(peak_frequency(post, freqs, *BETA)),
        "post_beta_rel_power": float(relative_band_power(post, freqs, *BETA)),
        # --- diagnostics, all of which can make a number above wrong
        "line_peak_hz": float(peak_frequency(glob, freqs, *LINE)),
        "line_prominence": float(peak_prominence(glob, freqs, LINE[0], LINE[1],
                                                 exclude_for_background=[LINE, ALPHA],
                                                 fit_lo=FIT[0], fit_hi=FIT[1])),
        "rest_s": round(t1 - t0, 3),
        "marker_rule": marker_rule,
        "analysed_s": round(T / fs, 3),
        "n_segments": int(M),
        "n_segments_kept": n_keep,
        "rejected_fraction": round(rejected, 6),
        "median_segment_ptp_uv": float(med_ptp.median()),
        "rejection_on": bool(reject),
        "n_channels_kept": len(names),
        "channels_bad_tsv": tsv_bad,
        "channels_bad_robust_sd": data_bad,
        "channels_occipital": occ_ch,
        "channels_sensorimotor": smr_ch,
        "channels_frontal_midline": fmt_ch,
        "channels_posterior": post_ch,
        "alpha_band_power_uv2": float(band_power(occ, freqs, *ALPHA)),
        "df_hz": float(freqs[1] - freqs[0]),
    }
    return r


# ======================================================================================
# known answers.  every one prints its prediction beside its measurement.
# ======================================================================================
def known_answers(corpus: Path, sid: str, gen: torch.Generator, cfg: Dict) -> Dict:
    assert isinstance(gen, torch.Generator), "pass a generator in; do not draw inside"
    ka: Dict = {"subject_used": sid}
    fs = 1000.0

    # -- (e) Parseval: the units check that the ledger keeps recording ------------------
    n = 8000
    z = torch.randn(4, n, generator=gen, dtype=torch.float64)
    f, p = welch_psd(z, fs, nperseg=2000, noverlap=1000)
    df = float(f[1] - f[0])
    got = float((p.sum(-1) * df).mean())
    want = float(z.var(dim=-1, unbiased=True).mean())
    ka["e_parseval"] = {"what": "psd.sum()*df vs var(x) on white noise",
                        "predicted": want, "measured": got,
                        "ratio": got / want, "pass": bool(abs(got / want - 1.0) < 0.05)}

    # -- (d) a flat spectrum: peak_frequency returns the band centre --------------------
    flat = torch.ones(1, f.numel(), dtype=torch.float64)
    ka["d_flat_spectrum"] = {
        "what": "peak_frequency on a perfectly flat psd, 8-13 Hz",
        "predicted": 0.5 * (ALPHA[0] + ALPHA[1]),
        "measured": float(peak_frequency(flat, f, *ALPHA)),
        "pass": bool(abs(float(peak_frequency(flat, f, *ALPHA)) - 10.5) < 1e-6),
        "note": "this is why a peak_hz is never quoted without its prominence",
    }

    # -- (b) white-noise surrogate through the WHOLE pipeline ---------------------------
    #    the length is taken from the real subject so the segmenting is identical.
    base = measure_subject(sid, corpus, **cfg)
    sur_seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen).item())
    wn = measure_subject(sid, corpus, **cfg, surrogate_seed=sur_seed, surrogate_sd=10.0)
    ka["b_white_noise"] = {
        "what": "a white-noise surrogate of the same length through the same pipeline",
        "predicted_exponent": 0.0, "measured_exponent_global": wn["exponent_global"],
        "predicted_alpha_prominence": 0.0,
        "measured_occ_alpha_prominence": wn["occ_alpha_prominence"],
        "measured_smr_mu_prominence": wn["smr_mu_prominence"],
        "measured_fm_theta_prominence": wn["fm_theta_prominence"],
        "measured_post_beta_prominence": wn["post_beta_prominence"],
        "measured_occ_alpha_peak_hz": wn["occ_alpha_peak_hz"],
        "pass": bool(abs(wn["exponent_global"]) < 0.05
                     and abs(wn["occ_alpha_prominence"]) < 0.05),
        "note": f"same shape and same pipeline as {sid}; sd 10 uV; surrogate_seed="
                f"{sur_seed}, itself drawn from the generator passed in",
    }

    # -- (a) an 11.3 Hz sinusoid injected at a KNOWN power ------------------------------
    #    P_tone = A^2/2.  Set A so P_tone = k * P_alpha(measured), and the prominence must
    #    rise by log10(1 + k) because the background fit excludes 8-13 and so barely moves.
    inj = []
    p_alpha = base["alpha_band_power_uv2"]
    for k in (1.0, 4.0):
        amp = math.sqrt(2.0 * k * p_alpha)
        m = measure_subject(sid, corpus, **cfg, inject=("occipital", 11.3, amp))
        inj.append({
            "k_times_alpha_power": k, "amplitude_uv": amp,
            "predicted_peak_hz": 11.3, "measured_peak_hz": m["occ_alpha_peak_hz"],
            "predicted_prominence_rise": math.log10(1.0 + k),
            "measured_prominence_rise": m["occ_alpha_prominence"] - base["occ_alpha_prominence"],
            "baseline_prominence": base["occ_alpha_prominence"],
            "injected_prominence": m["occ_alpha_prominence"],
            "pass": bool(abs(m["occ_alpha_peak_hz"] - 11.3) <= 0.1),
        })
    ka["a_injected_sinusoid"] = {
        "what": "11.3 Hz added to the occipital channels after re-referencing",
        "levels": inj,
        "pass": all(t["pass"] for t in inj),
    }

    # -- (f) the line frequency, read off the data -------------------------------------
    ka["f_line_frequency"] = {
        "what": "peak_frequency in 45-55 Hz on the global psd; the corpus was recorded in Australia",
        "predicted": 50.0, "measured": base["line_peak_hz"],
        "prominence_decades": base["line_prominence"],
        "pass": bool(abs(base["line_peak_hz"] - 50.0) < 0.5),
    }

    # -- (g) how much the artefact rejection moved the answer ---------------------------
    #    a preprocessing choice that changes the result is a result about the choice.
    noreject = measure_subject(sid, corpus, **{**cfg, "min_kept": 0.0}, reject=False)
    ka["g_rejection_off"] = {
        "what": "the same subject with artefact rejection switched off",
        "rejected_fraction_when_on": base["rejected_fraction"],
        "delta": {k: noreject[k] - base[k] for k in
                  ("occ_alpha_peak_hz", "occ_alpha_prominence", "smr_mu_peak_hz",
                   "smr_mu_prominence", "exponent_global", "fm_theta_prominence",
                   "post_beta_prominence")},
        "with_rejection": {k: base[k] for k in
                           ("occ_alpha_prominence", "smr_mu_prominence", "exponent_global",
                            "fm_theta_prominence", "post_beta_prominence")},
        "note": "no pass/fail: this is a sensitivity number, not a gate",
    }

    # -- (c) IDEMPOTENCE.  not a known-answer check: it tests whether the thing is a
    #    function at all.  CLAUDE.md, "call it twice at the same input".
    a = measure_subject(sid, corpus, **cfg)
    b = measure_subject(sid, corpus, **cfg)
    diffs = {}
    for k in a:
        if isinstance(a[k], float) and a[k] != b[k]:
            diffs[k] = [a[k], b[k]]
        elif not isinstance(a[k], float) and a[k] != b[k]:
            diffs[k] = [a[k], b[k]]
    ka["c_idempotence"] = {
        "what": "measure_subject called twice on the same file, compared for bit-identity",
        "bit_identical": not diffs, "differing_keys": diffs,
        "pass": not diffs,
    }
    ka["all_pass"] = all(v.get("pass", True) for v in ka.values() if isinstance(v, dict))
    return ka


# ======================================================================================
# split, bootstrap, intervals
# ======================================================================================
def draw_split(ids: Sequence[str], seed: int) -> Dict:
    """one permutation from one seeded generator, over ALL ids that have a recording.

    Drawn over the full 121 up front so that a partial run's split is a subset of the full
    run's: adding subjects must not move which half an already-measured subject is in.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    half = len(ids) // 2
    dec = sorted(ids[i] for i in order[:half])
    hel = sorted(ids[i] for i in order[half:])
    return {"seed": int(seed), "n_all": len(ids), "declaration": dec, "held_out": hel,
            "rule": "numpy default_rng(seed).permutation over participants.tsv order; "
                    "first floor(n/2) declares, the rest are held out"}


def _bootstrap(v: np.ndarray, gen: np.random.Generator, n: int = 10000) -> Dict:
    """percentile bootstrap over SUBJECTS.  `gen` is drawn once by the caller and passed
    in; this asserts it received one rather than making its own (CLAUDE.md, randomness)."""
    assert isinstance(gen, np.random.Generator), "pass a generator in; do not draw inside"
    v = np.asarray(v, dtype=float)
    k = v.size
    if k < 2:
        return {"n": int(k), "mean": float(v.mean()) if k else None, "se": None, "ci95": None}
    idx = gen.integers(0, k, size=(n, k))
    means = v[idx].mean(axis=1)
    return {
        "n": int(k),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),
        "se": float(v.std(ddof=1) / math.sqrt(k)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "median": float(np.median(v)),
        "q10": float(np.percentile(v, 10)),
        "q90": float(np.percentile(v, 90)),
        "q05": float(np.percentile(v, 5)),
        "q95": float(np.percentile(v, 95)),
        "min": float(v.min()), "max": float(v.max()),
    }


HEADLINE = ("occ_alpha_prominence", "occ_alpha_peak_hz", "smr_mu_prominence",
            "smr_mu_peak_hz", "exponent_global", "fm_theta_prominence",
            "post_beta_prominence")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def _corr_boot(a: np.ndarray, b: np.ndarray, gen: np.random.Generator, n: int = 4000) -> Dict:
    """Pearson r with a bootstrap over SUBJECTS.  Resampling the subjects is the only
    resampling that is allowed to widen this interval; resampling segments would not."""
    assert isinstance(gen, np.random.Generator), "pass a generator in; do not draw inside"
    k = a.size
    if k < 5:
        return {"n": int(k), "r": None, "ci95": None}
    idx = gen.integers(0, k, size=(n, k))
    rs = np.array([_pearson(a[i], b[i]) for i in idx])
    rs = rs[np.isfinite(rs)]
    return {"n": int(k), "r": _pearson(a, b),
            "ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]}


QUANTITIES = [
    "occ_alpha_peak_hz", "occ_alpha_prominence", "occ_alpha_rel_power",
    "smr_mu_peak_hz", "smr_mu_prominence", "smr_mu_rel_power",
    "exponent_global", "exponent_occipital", "exponent_sensorimotor",
    "exponent_frontal_midline", "exponent_global_catalogue_exclusions",
    "exponent_global_1_20", "exponent_global_20_45", "exponent_occipital_1_20",
    "exponent_sensorimotor_1_20", "exponent_frontal_midline_1_20",
    "fm_theta_prominence", "fm_theta_peak_hz",
    "post_beta_prominence", "post_beta_peak_hz", "post_beta_rel_power",
    "line_peak_hz", "rejected_fraction",
]


def summarise(per_subject: Dict[str, Dict], split: Dict, gen: np.random.Generator,
              diagnostics: bool = False, n_boot: int = 10000) -> Dict:
    dec_ids = [s for s in split["declaration"] if s in per_subject]
    hel_ids = [s for s in split["held_out"] if s in per_subject]
    out = {"n_measured": len(per_subject),
           "n_declaration_measured": len(dec_ids), "n_held_out_measured": len(hel_ids),
           "interval_rule": "the proposed interval is the 10th-90th percentile of the "
                            "DECLARATION half only; under exchangeability the held-out "
                            "coverage is 80%, binomial SE sqrt(.8*.2/n_heldout)",
           "quantities": {}}
    for q in QUANTITIES:
        allv = np.array([per_subject[s][q] for s in sorted(per_subject) if q in per_subject[s]])
        dv = np.array([per_subject[s][q] for s in dec_ids if q in per_subject[s]])
        hv = np.array([per_subject[s][q] for s in hel_ids if q in per_subject[s]])
        rec = {"all": _bootstrap(allv, gen, n_boot), "declaration": _bootstrap(dv, gen, n_boot),
               "held_out": _bootstrap(hv, gen, n_boot)}
        if dv.size >= 5 and hv.size >= 1:
            lo, hi = float(np.percentile(dv, 10)), float(np.percentile(dv, 90))
            inside = ((hv >= lo) & (hv <= hi)).astype(float)
            cov = float(inside.mean())
            rec["proposed_interval_10_90"] = [lo, hi]
            rec["held_out_coverage"] = cov
            rec["held_out_coverage_se"] = float(math.sqrt(0.8 * 0.2 / hv.size))
            rec["held_out_coverage_expected"] = 0.8
            # CLAUDE.md: a margin over a MEASURED baseline must clear sampling error on
            # BOTH sides.  The 80% expectation is not exact -- the interval's two edges are
            # order statistics of the declaration half, so the coverage of a FIXED held-out
            # population is itself ~Beta and carries sqrt(.8*.2/(n_dec+1)).  Judge a
            # coverage against this combined figure, never against the binomial alone.
            rec["held_out_coverage_se_combined"] = float(math.sqrt(
                0.8 * 0.2 / hv.size + 0.8 * 0.2 / (dv.size + 1)))
            rec["held_out_coverage_boot"] = _bootstrap(inside, gen, n_boot)
            lo5, hi5 = float(np.percentile(dv, 5)), float(np.percentile(dv, 95))
            in5 = ((hv >= lo5) & (hv <= hi5)).astype(float)
            rec["proposed_interval_05_95"] = [lo5, hi5]
            rec["held_out_coverage_05_95"] = float(in5.mean())
            rec["held_out_coverage_05_95_expected"] = 0.9
            # paired difference in the two halves' means, bootstrapped over subjects on
            # each side -- the check that the split itself did not split on something.
            rec["halves_mean_difference"] = float(dv.mean() - hv.mean())
        out["quantities"][q] = rec

    # -- diagnostics: the ways a number above could be right for the wrong reason -------
    sids = sorted(per_subject)
    def col(k):
        return np.array([per_subject[s][k] for s in sids if k in per_subject[s]], dtype=float)
    if diagnostics and len(sids) >= 5:
        rej = col("rejected_fraction")
        ptp = col("median_segment_ptp_uv")
        d = {"note": "Pearson r over SUBJECTS with a percentile bootstrap over subjects. "
                     "These do not falsify a measurement; they say what else it moves with.",
             "rejected_fraction_vs": {q: _corr_boot(rej, col(q), gen) for q in HEADLINE},
             "median_segment_ptp_uv_vs": {q: _corr_boot(ptp, col(q), gen) for q in HEADLINE}}
        # the big one: under an average reference, occipital alpha volume-conducts onto the
        # central electrodes, so a "mu" prominence can be occipital alpha wearing a hat.  A
        # very high correlation here would mean rows 1 and 2 are one measurement, not two.
        oa, mu = col("occ_alpha_prominence"), col("smr_mu_prominence")
        d["occ_alpha_vs_mu_prominence"] = _corr_boot(oa, mu, gen)
        diff = oa - mu
        d["occ_minus_mu_prominence_paired"] = _bootstrap(diff, gen)
        d["occ_minus_mu_prominence_paired"]["note"] = (
            "the PAIRED difference, bootstrapped as a difference -- the shared per-subject "
            "sampling error cancels, which it would not if the two means were compared")
        opk, mpk = col("occ_alpha_peak_hz"), col("smr_mu_peak_hz")
        d["occ_alpha_vs_mu_peak_hz"] = _corr_boot(opk, mpk, gen)
        d["occ_minus_mu_peak_hz_paired"] = _bootstrap(opk - mpk, gen)
        out["diagnostics"] = d
    return out


# ======================================================================================
# the suggested catalogue patch.  PRINTED, never written: ibm/rhythms.py is not touched.
# ======================================================================================
PATCH_ROWS = [
    dict(row="alpha_occipital", quantity="occ_alpha_prominence",
         declared_kind="peak_prominence", declared_band=(8.0, 13.0),
         declared_target=(0.4, 1.5), declared_states=("wake-eyes-closed",),
         state_match=False,
         blocker="every ds008037 rest sidecar declares EyesCondition = 'eyes open', and the "
                 "events file has only start/end marks, so this corpus contains no "
                 "eyes-closed data at all.  The row's declared state is wake-eyes-closed."),
    dict(row="alpha_occipital.peak", quantity="occ_alpha_peak_hz",
         declared_kind="peak", declared_band=(8.0, 13.0), declared_target=None,
         declared_point=10.0,
         declared_states=("wake-eyes-closed",), state_match=False,
         blocker="same state mismatch.  The declared peak is a POINT, 10.0 Hz, and a point "
                 "has no width, so 'fraction of subjects inside it' is meaningless -- what "
                 "is reported instead is whether 10.0 lies inside the measured interval."),
    dict(row="mu", quantity="smr_mu_prominence",
         declared_kind="peak_prominence", declared_band=(8.0, 13.0),
         declared_target=(0.2, 1.0), declared_states=("wake-rest",), state_match=True,
         blocker=""),
    dict(row="mu.peak", quantity="smr_mu_peak_hz",
         declared_kind="peak", declared_band=(8.0, 13.0), declared_target=None,
         declared_point=10.0,
         declared_states=("wake-rest",), state_match=True, blocker=""),
    dict(row="BACKGROUND (aperiodic)", quantity="exponent_global",
         declared_kind="aperiodic_exponent", declared_band=(1.0, 45.0),
         declared_target=(0.8, 2.0), declared_states=("wake-rest",), state_match=True,
         blocker=""),
    dict(row="frontal_midline_theta", quantity="fm_theta_prominence",
         declared_kind="load_slope", declared_band=(4.0, 8.0),
         declared_target=(0.02, 0.50), declared_states=("wake-task", "working-memory"),
         state_match=False,
         blocker="the row declares kind=load_slope: theta power AGAINST SET SIZE on the "
                 "working-memory task.  This corpus is rest with no task markers, so the "
                 "quantity measured here (resting 4-8 Hz prominence) is a DIFFERENT "
                 "quantity in different units and cannot confirm or refute that target."),
    dict(row="(no row: posterior scalp beta at rest)", quantity="post_beta_prominence",
         declared_kind="peak_prominence", declared_band=(13.0, 30.0),
         declared_target=None, declared_states=("wake-rest",), state_match=True,
         blocker="the catalogue's only 13-30 Hz rest row is `beta_bg`, declared at the STN "
                 "and GPe (substrate: needs-basal-ganglia).  A scalp parieto-occipital beta "
                 "measurement is not that row and must not be used to confirm it."),
]


def catalogue_patch(summary: Dict) -> List[Dict]:
    out = []
    for spec in PATCH_ROWS:
        q = summary["quantities"].get(spec["quantity"], {})
        iv = q.get("proposed_interval_10_90")
        cov = q.get("held_out_coverage")
        dt = spec["declared_target"]
        rec = {k: spec[k] for k in
               ("row", "quantity", "declared_kind", "declared_band", "declared_target",
                "declared_states", "state_match", "blocker")}
        rec["declared_point"] = spec.get("declared_point")
        rec.update({
            "measured_mean": q.get("all", {}).get("mean"),
            "measured_se": q.get("all", {}).get("se"),
            "measured_ci95": q.get("all", {}).get("ci95"),
            "declaration_interval_10_90": iv,
            "held_out_coverage": cov,
            "held_out_coverage_expected": q.get("held_out_coverage_expected"),
            "held_out_coverage_se": q.get("held_out_coverage_se"),
            "held_out_coverage_se_combined": q.get("held_out_coverage_se_combined"),
            "declaration_interval_05_95": q.get("proposed_interval_05_95"),
            "held_out_coverage_05_95": q.get("held_out_coverage_05_95"),
        })
        # does the DECLARED interval contain the measured population?
        if dt is not None and q.get("all") and q["all"].get("ci95"):
            a = q["all"]
            rec["fraction_of_all_subjects_inside_declared_target"] = None
            rec["declared_target_contains_mean"] = bool(dt[0] <= a["mean"] <= dt[1])
            rec["declared_target_contains_ci95"] = bool(
                dt[0] <= a["ci95"][0] and a["ci95"][1] <= dt[1])
            rec["measured_interval_below_declared"] = bool(a["q90"] < dt[0])
            rec["measured_interval_above_declared"] = bool(a["q10"] > dt[1])
        pt = spec.get("declared_point")
        if pt is not None and q.get("all") and q["all"].get("ci95"):
            a = q["all"]
            rec["declared_point_inside_declaration_interval"] = bool(
                iv is not None and iv[0] <= pt <= iv[1])
            rec["declared_point_inside_mean_ci95"] = bool(a["ci95"][0] <= pt <= a["ci95"][1])
            rec["mean_minus_declared_point"] = float(a["mean"] - pt)
        out.append(rec)
    return out


def _fraction_inside(per_subject, quantity, target):
    if target is None:
        return None
    v = np.array([r[quantity] for r in per_subject.values() if quantity in r])
    if not v.size:
        return None
    return float(((v >= target[0]) & (v <= target[1])).mean())


# ======================================================================================
# io
# ======================================================================================
def _default(o):
    """coerce anything json cannot take, so a stray array degrades to its shape instead of
    destroying the record.  CLAUDE.md: never hand numpy straight to json."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return {"__ndarray__": list(o.shape), "dtype": str(o.dtype)}
    if isinstance(o, torch.Tensor):
        return {"__tensor__": list(o.shape), "dtype": str(o.dtype)}
    if isinstance(o, Path):
        return str(o)
    return {"__repr__": repr(o)[:200]}


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_default, sort_keys=False))
    tmp.replace(path)


def git_sha() -> str:
    sha = os.environ.get("IBM_GIT_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(REPO), text=True).strip()
    except Exception:
        return "git-unknown"


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--start", type=int, default=0, help="index into the participants.tsv order")
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--split-seed", type=int, default=20260918)
    ap.add_argument("--boot-seed", type=int, default=20260919)
    ap.add_argument("--known-seed", type=int, default=20260920)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--nperseg-s", type=float, default=4.0)
    ap.add_argument("--ptp-k", type=float, default=3.0,
                    help="a segment dies if any channel's peak-to-peak exceeds k x that "
                         "channel's median segment peak-to-peak")
    ap.add_argument("--min-kept", type=float, default=0.5)
    ap.add_argument("--edge-s", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--redo", action="store_true", help="re-measure subjects already in the file")
    ap.add_argument("--redo-known", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    cfg = dict(nperseg_s=a.nperseg_s, ptp_k=a.ptp_k, min_kept=a.min_kept,
               edge_s=a.edge_s)

    ids, missing = subject_ids(a.corpus)
    split = draw_split(ids, a.split_seed)

    payload: Dict = {}
    if a.out.exists():
        try:
            payload = json.loads(a.out.read_text())
        except Exception:
            payload = {}
    per_subject: Dict[str, Dict] = payload.get("per_subject", {})
    exclusions: Dict[str, str] = payload.get("exclusions", {})
    exclusions.update(missing)
    known: Dict = payload.get("known_answers", {})
    timings: Dict = payload.get("timings", {})

    # the cheap summary is assembled and written BEFORE anything that can raise
    boot_gen = np.random.default_rng(a.boot_seed)
    payload = {
        "script": "scripts/measure_rhythms_eeg.py",
        "git_sha": git_sha(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": str(a.corpus),
        "corpus_facts": {
            "task": "task-rest", "sfreq_hz": 1000.0, "n_eeg_channels": 62,
            "line_frequency_hz": 50.0,
            "eyes_condition": "eyes open, in all 121 rest sidecars -- there is NO "
                              "eyes-closed data and no eyes-open/eyes-closed contrast "
                              "is in scope",
            "events": "only 200001 (start) and 200002 (end); no task or state markers",
            "channels_tsv_status": "7502 of 7502 channel rows across the corpus are "
                                   "'good', so the status column excludes nothing; a "
                                   "robust-sd check is applied on top and reported per subject",
            "declared_reference": "channels.tsv says FCz; the data contain a full-amplitude "
                                  "FCZ channel and the EEGLAB header says ref='Common', so "
                                  "the sidecar cannot be read literally -- see the module "
                                  "docstring",
        },
        "config": {**cfg, "welch_window": "hann", "welch_overlap": 0.5,
                   "highpass_stop_hz": 0.25, "highpass_pass_hz": 0.5,
                   "reference": "average over retained channels",
                   "fit_band_hz": list(FIT),
                   "aperiodic_exclude_instructed": [list(b) for b in INSTRUCTED_EXCLUDE],
                   "aperiodic_exclude_catalogue": [list(b) for b in CATALOGUE_EXCLUDE],
                   "bootstrap_resamples": a.boot,
                   "bootstrap_unit": "SUBJECT (never segment, never window)",
                   "boot_seed": a.boot_seed, "known_seed": a.known_seed},
        "channel_groups": {k: list(v) for k, v in GROUPS.items()},
        "subjects_all": ids,
        "split": split,
        "known_answers": known,
        "exclusions": exclusions,
        "per_subject": per_subject,
        "timings": timings,
    }
    payload["summary"] = summarise(per_subject, split, boot_gen, n_boot=1000) if per_subject else {}
    payload["catalogue_patch"] = catalogue_patch(payload["summary"]) if per_subject else []
    write_json(a.out, payload)

    if not a.summary_only:
        # -- known answers -------------------------------------------------------------
        if a.redo_known or not known:
            gen = torch.Generator().manual_seed(a.known_seed)
            t = time.time()
            print("known answers ...", flush=True)
            payload["known_answers"] = known_answers(a.corpus, ids[0], gen, cfg)
            timings["known_answers_s"] = round(time.time() - t, 2)
            payload["timings"] = timings
            write_json(a.out, payload)
            _print_known(payload["known_answers"])

        # -- the subjects ---------------------------------------------------------------
        todo = ids[a.start:] if not a.limit else ids[a.start:a.start + a.limit]
        for i, sid in enumerate(todo):
            if sid in per_subject and not a.redo:
                continue
            t = time.time()
            try:
                per_subject[sid] = measure_subject(sid, a.corpus, **cfg)
                exclusions.pop(sid, None)
                dt = time.time() - t
                timings[sid] = round(dt, 2)
                r = per_subject[sid]
                print(f"[{i + 1}/{len(todo)}] {sid}  {dt:5.1f}s  "
                      f"alpha {r['occ_alpha_peak_hz']:5.2f} Hz / {r['occ_alpha_prominence']:+.3f}  "
                      f"mu {r['smr_mu_peak_hz']:5.2f} Hz / {r['smr_mu_prominence']:+.3f}  "
                      f"exp {r['exponent_global']:.2f}  "
                      f"theta {r['fm_theta_prominence']:+.3f}  "
                      f"beta {r['post_beta_prominence']:+.3f}  "
                      f"rej {100 * r['rejected_fraction']:4.1f}%", flush=True)
            except Exception as exc:  # recorded WITH ITS REASON, never silently skipped
                exclusions[sid] = f"{type(exc).__name__}: {exc}"
                timings[sid] = round(time.time() - t, 2)
                print(f"[{i + 1}/{len(todo)}] {sid}  EXCLUDED  {exclusions[sid]}", flush=True)
            # written after EVERY subject, so a crash leaves the record
            payload["per_subject"] = per_subject
            payload["exclusions"] = exclusions
            payload["timings"] = timings
            # cheap in-loop summary (1,000 resamples) so the per-subject write stays
            # cheap; the FINAL summary below uses the full --boot count.
            payload["summary"] = summarise(per_subject, split,
                                           np.random.default_rng(a.boot_seed), n_boot=1000)
            payload["catalogue_patch"] = catalogue_patch(payload["summary"])
            write_json(a.out, payload)

    # -- final summary, from a generator drawn once here -------------------------------
    payload["summary"] = summarise(per_subject, split, np.random.default_rng(a.boot_seed),
                                   diagnostics=True, n_boot=a.boot)
    for spec in PATCH_ROWS:
        q = payload["summary"]["quantities"].get(spec["quantity"])
        if q is not None and spec["declared_target"] is not None:
            q["fraction_inside_declared_target"] = _fraction_inside(
                per_subject, spec["quantity"], spec["declared_target"])
    payload["catalogue_patch"] = catalogue_patch(payload["summary"])
    for rec in payload["catalogue_patch"]:
        rec["fraction_of_all_subjects_inside_declared_target"] = _fraction_inside(
            per_subject, rec["quantity"], rec["declared_target"])
    payload["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(a.out, payload)

    _report(payload)
    return 0


def _print_known(ka: Dict) -> None:
    print("\n" + "=" * 78)
    print("KNOWN ANSWERS  (prediction -> measurement)")
    print("=" * 78)
    e = ka["e_parseval"]
    print(f"(e) Parseval          var(x) = {e['predicted']:.6f}   psd.sum*df = "
          f"{e['measured']:.6f}   ratio {e['ratio']:.6f}   {'PASS' if e['pass'] else 'FAIL'}")
    d = ka["d_flat_spectrum"]
    print(f"(d) flat spectrum     peak_frequency(8-13) predicted {d['predicted']:.2f} Hz "
          f"-> {d['measured']:.6f} Hz   {'PASS' if d['pass'] else 'FAIL'}")
    b = ka["b_white_noise"]
    print(f"(b) white surrogate   exponent predicted 0.0 -> {b['measured_exponent_global']:+.4f}   "
          f"alpha prominence predicted 0.0 -> {b['measured_occ_alpha_prominence']:+.4f}   "
          f"{'PASS' if b['pass'] else 'FAIL'}")
    print(f"                      (mu {b['measured_smr_mu_prominence']:+.4f}, theta "
          f"{b['measured_fm_theta_prominence']:+.4f}, beta "
          f"{b['measured_post_beta_prominence']:+.4f}, peak_hz "
          f"{b['measured_occ_alpha_peak_hz']:.3f} = the band centre, as (d) predicts)")
    for t in ka["a_injected_sinusoid"]["levels"]:
        print(f"(a) inject 11.3 Hz    at {t['k_times_alpha_power']:.0f}x alpha power "
              f"(A = {t['amplitude_uv']:.2f} uV): peak predicted 11.30 -> "
              f"{t['measured_peak_hz']:.4f} Hz   prominence rise predicted "
              f"{t['predicted_prominence_rise']:+.4f} -> {t['measured_prominence_rise']:+.4f}"
              f"   {'PASS' if t['pass'] else 'FAIL'}")
    f = ka["f_line_frequency"]
    print(f"(f) line frequency    predicted 50.0 Hz -> {f['measured']:.4f} Hz "
          f"({f['prominence_decades']:+.2f} decades above background)   "
          f"{'PASS' if f['pass'] else 'FAIL'}")
    g = ka["g_rejection_off"]
    print(f"(g) rejection OFF     (sensitivity, not a gate; {100 * g['rejected_fraction_when_on']:.1f}% "
          f"of segments are dropped when it is on)")
    print("                      " + "  ".join(f"{k.split('_')[0]}{k.split('_')[-1][:4]} {v:+.4f}"
                                               for k, v in g["delta"].items()))
    c = ka["c_idempotence"]
    print(f"(c) IDEMPOTENCE       measure_subject called twice: bit-identical = "
          f"{c['bit_identical']}   {'PASS' if c['pass'] else 'FAIL'}")
    if c["differing_keys"]:
        print(f"    DIFFERING: {c['differing_keys']}")
    print("=" * 78 + "\n", flush=True)


LABELS = {
    "occ_alpha_peak_hz": "1. occipital alpha peak (Hz)",
    "occ_alpha_prominence": "1. occipital alpha prominence (decades)",
    "occ_alpha_rel_power": "1. occipital alpha relative power",
    "smr_mu_peak_hz": "2. sensorimotor mu peak (Hz)",
    "smr_mu_prominence": "2. sensorimotor mu prominence (decades)",
    "smr_mu_rel_power": "2. sensorimotor mu relative power",
    "exponent_global": "3. aperiodic exponent, global",
    "exponent_occipital": "3. aperiodic exponent, occipital",
    "exponent_sensorimotor": "3. aperiodic exponent, sensorimotor",
    "exponent_frontal_midline": "3. aperiodic exponent, frontal midline",
    "exponent_global_catalogue_exclusions": "3. exponent, catalogue's own exclusions",
    "exponent_global_1_20": "3s. exponent, global, 1-20 Hz only",
    "exponent_global_20_45": "3s. exponent, global, 20-45 Hz only",
    "exponent_occipital_1_20": "3s. exponent, occipital, 1-20 Hz only",
    "exponent_sensorimotor_1_20": "3s. exponent, sensorimotor, 1-20 Hz only",
    "exponent_frontal_midline_1_20": "3s. exponent, frontal midline, 1-20 Hz only",
    "fm_theta_prominence": "4. frontal-midline theta prominence (decades)",
    "fm_theta_peak_hz": "4. frontal-midline theta peak (Hz)",
    "post_beta_prominence": "5. posterior beta prominence (decades)",
    "post_beta_peak_hz": "5. posterior beta peak (Hz)",
    "post_beta_rel_power": "5. posterior beta relative power",
    "line_peak_hz": "d. 50 Hz line peak (Hz)",
    "rejected_fraction": "d. segments rejected (fraction)",
}


def _report(payload: Dict) -> None:
    s = payload["summary"]
    print("\n" + "=" * 100)
    print(f"ds008037 resting EEG -- {s['n_measured']} subjects measured "
          f"({s['n_declaration_measured']} declaration, {s['n_held_out_measured']} held out) "
          f"of {len(payload['subjects_all'])}")
    print("=" * 100)
    print(f"{'quantity':45s} {'mean':>9s} {'SE':>8s} {'95% CI (bootstrap over SUBJECTS)':>28s}"
          f" {'10-90 decl':>16s} {'cover':>7s}")
    for q in QUANTITIES:
        r = s["quantities"].get(q)
        if not r or r["all"]["n"] < 2:
            continue
        a = r["all"]
        iv = r.get("proposed_interval_10_90")
        cov = r.get("held_out_coverage")
        ivs = f"[{iv[0]:.3f},{iv[1]:.3f}]" if iv else "-"
        cs = f"{100 * cov:.0f}%" if cov is not None else "-"
        print(f"{LABELS.get(q, q):45s} {a['mean']:9.4f} {a['se']:8.4f} "
              f"   [{a['ci95'][0]:9.4f}, {a['ci95'][1]:9.4f}] {ivs:>16s} {cs:>7s}")
    dg = s.get("diagnostics")
    if dg:
        print("\nDIAGNOSTICS -- what else these numbers move with (Pearson r over subjects, "
              "95% CI bootstrapped over subjects)")
        for k, lbl in (("rejected_fraction_vs", "segments rejected"),
                       ("median_segment_ptp_uv_vs", "subject amplitude (median seg ptp)")):
            for q, v in dg[k].items():
                if v["r"] is None:
                    continue
                print(f"  r({lbl:38s}, {q:24s}) = {v['r']:+.3f}  "
                      f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        v = dg["occ_alpha_vs_mu_prominence"]
        print(f"  r(occipital alpha prominence, sensorimotor mu prominence) = {v['r']:+.3f} "
              f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]   <- if this were ~1 the two rows "
              f"would be one measurement")
        v = dg["occ_alpha_vs_mu_peak_hz"]
        print(f"  r(occipital alpha peak Hz,    sensorimotor mu peak Hz)    = {v['r']:+.3f} "
              f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        v = dg["occ_minus_mu_prominence_paired"]
        print(f"  PAIRED occipital - sensorimotor prominence = {v['mean']:+.4f} +/- {v['se']:.4f} "
              f"SE, 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        v = dg["occ_minus_mu_peak_hz_paired"]
        print(f"  PAIRED occipital - sensorimotor peak (Hz)  = {v['mean']:+.4f} +/- {v['se']:.4f} "
              f"SE, 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print("\nexclusions:")
    for k, v in sorted(payload["exclusions"].items()):
        print(f"  {k}: {v}")
    if not payload["exclusions"]:
        print("  (none)")

    print("\n" + "=" * 100)
    print("SUGGESTED PATCH TO ibm/rhythms.py  -- printed only; this script does not edit it")
    print("=" * 100)
    for rec in payload["catalogue_patch"]:
        print(f"\n--- {rec['row']}   [{rec['quantity']}]")
        dt = rec["declared_target"]
        print(f"    declared   kind={rec['declared_kind']} band={rec['declared_band']} "
              f"target={dt} states={rec['declared_states']}")
        if rec["measured_mean"] is None or rec["measured_ci95"] is None:
            print("    measured   (too few subjects yet)")
            continue
        print(f"    measured   mean {rec['measured_mean']:+.4f} +/- {rec['measured_se']:.4f} SE, "
              f"95% CI [{rec['measured_ci95'][0]:+.4f}, {rec['measured_ci95'][1]:+.4f}]")
        if rec["declaration_interval_10_90"] is None or rec["held_out_coverage"] is None:
            print("    declaration half supports  (too few subjects yet)")
            continue
        iv = rec["declaration_interval_10_90"]
        print(f"    declaration half supports  [{iv[0]:+.4f}, {iv[1]:+.4f}] (10-90 pct of the "
              f"declaration half)")
        sc = rec.get("held_out_coverage_se_combined") or rec["held_out_coverage_se"]
        print(f"    held-out coverage          {100 * rec['held_out_coverage']:.1f}% "
              f"(expected {100 * rec['held_out_coverage_expected']:.0f}% +/- "
              f"{100 * sc:.1f}%, combining the held-out binomial error with the "
              f"declaration half's own)")
        if rec.get("declared_point") is not None:
            print(f"    declared point {rec['declared_point']:.2f}: inside the declaration "
                  f"interval = {rec['declared_point_inside_declaration_interval']}, inside "
                  f"the 95% CI of the mean = {rec['declared_point_inside_mean_ci95']} "
                  f"(mean - point = {rec['mean_minus_declared_point']:+.4f})")
        iv5 = rec.get("declaration_interval_05_95")
        if iv5:
            print(f"    wider 5-95 alternative     [{iv5[0]:+.4f}, {iv5[1]:+.4f}]  held-out "
                  f"coverage {100 * rec['held_out_coverage_05_95']:.1f}% (expected 90%)")
        fi = rec.get("fraction_of_all_subjects_inside_declared_target")
        if fi is not None:
            print(f"    fraction of ALL subjects inside the DECLARED target: {100 * fi:.1f}%")
        if rec["blocker"]:
            print(f"    evidence tag: CANNOT move to measured-here.")
            print(f"      why: {rec['blocker']}")
        elif dt is not None and rec.get("measured_interval_below_declared"):
            print(f"    evidence tag: the declared target is NOT supported -- 90% of "
                  f"subjects fall BELOW its lower edge {dt[0]}.")
        elif dt is not None and rec.get("measured_interval_above_declared"):
            print(f"    evidence tag: the declared target is NOT supported -- 90% of "
                  f"subjects fall ABOVE its upper edge {dt[1]}.")
        else:
            print("    evidence tag: see the numbers above; the verdict is written in the report.")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
