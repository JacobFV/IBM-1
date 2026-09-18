"""spectral instruments, differentiable, in pure torch.

why this module exists
----------------------
the programme is shaping a cortical field (`ibm/substrate.py`: a Wilson-Cowan style
E/I field with per-site heterogeneous time constants) so that its simulated regional
activity carries the brain's real rhythms -- a slow oscillation near 1 Hz, sleep
spindles at a measured 13.45 Hz, occipital alpha, sensorimotor mu, basal-ganglia
beta, hippocampal theta with gamma nested inside it, and so on.  those rhythms are
a *specification*, not a score to be searched for, and the way to build toward a
specification is to write it down as a pressure the model feels.  so a power-spectral
term has to sit in the training objective beside the anatomical priors, which means
every quantity below must be differentiable with respect to the simulated trace.

the same functions are the measurement-side instrument: when the claim is "the model
has a spindle at 13.45 Hz like the recording does", the number for the model and the
number for the recording must come out of the *same code path*, or the comparison is
between two instruments rather than between two brains.  (CLAUDE.md, "a gate that
compares like with like is blind to a difference between the two likes" -- here the
risk runs the other way: two different instruments make a difference that is not
there.)  there is exactly one implementation here and both sides call it.

what this is NOT
----------------
`ibm/forge/spectra.py` is a different instrument for a different job: it turns a
recording into gamma-distributed *evidence* with an exactly known precision, and to
keep that precision knowable it refuses overlapping windows.  this module is the
loss side.  it overlaps windows by default (lower-variance estimate, better-behaved
gradient) and it does not pretend to know its own sampling distribution.  do not use
these numbers as a likelihood, and do not use that module inside a backward pass.

conventions that hold everywhere below
--------------------------------------
* time is the LAST axis.  any leading batch axes `(..., T)` are carried through.
* `fs` is in Hz, frequencies are in Hz, and a psd is a DENSITY in units of x^2/Hz,
  so `psd.sum(-1) * df` is an estimate of the variance of x (Parseval).  a "power"
  returned by `band_power` is therefore in units of x^2.
* float32 and float64 both work; the output dtype follows the input.
* **nothing here touches the global RNG.**  there is no `torch.randn`/`torch.rand`
  in this file at all.  CLAUDE.md's randomness section is about instruments that
  quietly redraw between calls; an instrument that cannot draw cannot do that.
* **every public function is idempotent**: called twice on the same input it returns
  bit-identical tensors.  the only cache is `_WINDOW_CACHE`, keyed purely on
  (window name, length, dtype, device), holding values that are never mutated.
  `tests/test_spectral.py` checks this by calling everything twice, because
  idempotence is not a known-answer check -- it tests whether the thing is a
  *function* at all, and a non-idempotent instrument produces confident wrong
  conclusions from reasoning that is itself sound.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

__all__ = [
    "welch_psd",
    "band_power",
    "relative_band_power",
    "aperiodic_fit",
    "peak_frequency",
    "peak_prominence",
    "pac_mi",
    "band_coherence",
    "phase_locking_value",
    "spectral_loss",
    "bandpass_analytic",
]

# a floor under every logarithm and every division.  it is deliberately much smaller
# than any psd this programme will see in real units (volts^2/Hz on scalp EEG is
# ~1e-12 and up) and much larger than float32's denormal range.
_EPS = 1e-20


# --------------------------------------------------------------------------------
# windows and segmenting
# --------------------------------------------------------------------------------

# keyed on (name, length, dtype-string, device-string) only -- no mutable state can
# reach this key, so a hit and a miss return the same numbers.  the cached tensors
# are read-only by convention: nothing in this module writes into one.
_WINDOW_CACHE: Dict[Tuple[str, int, str, str], Tensor] = {}


def _window(spec: Union[str, Tensor], n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """window coefficients of length `n`.

    the string windows are PERIODIC (`periodic=True`), not symmetric, which is what
    spectral estimation wants and what `scipy.signal.get_window(..., fftbins=True)`
    returns -- `tests/test_spectral.py` cross-checks against scipy and would fail on
    the symmetric variant.
    """
    if isinstance(spec, Tensor):
        w = spec.to(dtype=dtype, device=device)
        if w.shape != (n,):
            raise ValueError(f"window tensor has shape {tuple(w.shape)}, expected ({n},)")
        return w
    key = (str(spec).lower(), int(n), str(dtype), str(device))
    w = _WINDOW_CACHE.get(key)
    if w is None:
        name = key[0]
        if name == "hann":
            w = torch.hann_window(n, periodic=True, dtype=dtype, device=device)
        elif name == "hamming":
            w = torch.hamming_window(n, periodic=True, dtype=dtype, device=device)
        elif name in ("boxcar", "rect", "rectangular", "none"):
            w = torch.ones(n, dtype=dtype, device=device)
        else:
            raise ValueError(f"unknown window {spec!r}; use 'hann', 'hamming', 'boxcar' or a tensor")
        _WINDOW_CACHE[key] = w
    return w


def _detrend(seg: Tensor, mode: str) -> Tensor:
    """remove a per-segment constant or line.  differentiable; both are linear maps.

    `"constant"` is the default everywhere because a segment mean is a DC coefficient
    that would otherwise sit in bin 0 and, through window leakage, in the bins next to
    it -- which is exactly where the slow oscillation near 1 Hz lives.
    """
    if mode in (None, "none", False):
        return seg
    if mode == "constant":
        return seg - seg.mean(dim=-1, keepdim=True)
    if mode == "linear":
        n = seg.shape[-1]
        t = torch.arange(n, dtype=seg.dtype, device=seg.device)
        t = t - t.mean()
        denom = (t * t).sum()
        slope = (seg * t).sum(dim=-1, keepdim=True) / denom
        return seg - seg.mean(dim=-1, keepdim=True) - slope * t
    raise ValueError(f"unknown detrend {mode!r}; use 'constant', 'linear' or 'none'")


def _segment_fft(
    x: Tensor,
    fs: float,
    nperseg: Optional[int],
    noverlap: Optional[int],
    window: Union[str, Tensor],
    detrend: str,
    onesided: bool,
) -> Tuple[Tensor, Tensor, Tensor, int]:
    """the one segmenting path.  `welch_psd` and `band_coherence` both go through it,
    so a cross-spectrum and an auto-spectrum can never be computed on different grids.

    returns `(freqs, X, winpow, n_segments)` where `X` is `(..., M, F)` complex,
    `winpow = sum(w**2)` is the window power used for the density normalisation, and
    `freqs` is `(F,)` in Hz.
    """
    if x.ndim < 1:
        raise ValueError("x needs at least a time axis")
    T = x.shape[-1]
    if nperseg is None:
        nperseg = min(256, T)  # scipy.signal.welch's default, so the cross-check is unsurprising
    nperseg = int(nperseg)
    if nperseg < 2 or nperseg > T:
        raise ValueError(f"nperseg={nperseg} must be in [2, T={T}]")
    if noverlap is None:
        noverlap = nperseg // 2  # welch's 50%: lowest variance for a hann taper
    noverlap = int(noverlap)
    if not (0 <= noverlap < nperseg):
        raise ValueError(f"noverlap={noverlap} must be in [0, nperseg={nperseg})")

    step = nperseg - noverlap
    seg = x.unfold(-1, nperseg, step)                       # (..., M, nperseg), a view
    M = seg.shape[-2]
    seg = _detrend(seg, detrend)
    w = _window(window, nperseg, seg.dtype, seg.device)
    seg = seg * w
    winpow = (w * w).sum()

    if onesided:
        X = torch.fft.rfft(seg, dim=-1)
        freqs = torch.fft.rfftfreq(nperseg, d=1.0 / float(fs), dtype=seg.dtype, device=seg.device)
    else:
        X = torch.fft.fft(seg, dim=-1)
        freqs = torch.fft.fftfreq(nperseg, d=1.0 / float(fs), dtype=seg.dtype, device=seg.device)
    return freqs, X, winpow, M


def welch_psd(
    x: Tensor,
    fs: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: Union[str, Tensor] = "hann",
    detrend: str = "constant",
    onesided: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Welch's averaged periodogram, differentiable with respect to `x`.

    segment -> detrend -> taper -> rFFT -> square -> average over segments.

    args
        x          `(..., T)`, time last.  any leading axes are carried through.
        fs         sampling rate in Hz.
        nperseg    samples per segment.  default `min(256, T)` (scipy's default).
                   the frequency resolution is `fs / nperseg`; to resolve the slow
                   oscillation near 1 Hz you want at least a 2 s segment, so pass
                   `nperseg=int(2*fs)` rather than accepting the default there.
        noverlap   default `nperseg // 2`.  overlap lowers the variance of the
                   estimate and, more to the point here, smooths the gradient,
                   because every sample then contributes to two segments.
        window     "hann" (default), "hamming", "boxcar", or a `(nperseg,)` tensor.
        detrend    "constant" (default, per-segment mean), "linear", or "none".
        onesided   True returns `rfftfreq` bins with the standard doubling of every
                   bin except DC and (for even `nperseg`) Nyquist.  False returns the
                   full two-sided spectrum with no doubling.

    returns
        `freqs` `(F,)` in Hz, and `psd` `(..., F)` a DENSITY in x^2/Hz.

    the normalisation is `|X_k|^2 / (fs * sum(w**2))`, which is the one that makes
    `psd.sum(-1) * df` an estimate of the variance of `x` -- test 1 pins that, because
    a psd whose units are wrong is the exact shape of error CLAUDE.md's ledger keeps
    recording (a quantity computed correctly and compared against the wrong thing).

    the magnitude is taken as `re**2 + im**2` rather than `abs(X)**2`: they are the
    same number, but `abs` has a non-finite gradient at zero and a psd bin genuinely
    reaches zero (a perfectly detrended constant segment puts an exact 0 in bin 0),
    so `abs` would put a NaN in the backward pass of an otherwise healthy run.
    """
    freqs, X, winpow, _ = _segment_fft(x, fs, nperseg, noverlap, window, detrend, onesided)
    p = (X.real * X.real + X.imag * X.imag) / (float(fs) * winpow)
    if onesided:
        scale = torch.full_like(freqs, 2.0)
        scale[0] = 1.0                       # DC has no mirror image
        if _rfft_even(freqs, fs):
            scale[-1] = 1.0                  # nor does Nyquist, when the segment is even
        p = p * scale
    return freqs, p.mean(dim=-2)


def _rfft_even(freqs: Tensor, fs: float) -> bool:
    """did the rfft come from an even-length segment?  true iff the last bin is Nyquist."""
    return bool(abs(float(freqs[-1]) - float(fs) / 2.0) < 1e-9 * max(1.0, float(fs)))


# --------------------------------------------------------------------------------
# band quantities
# --------------------------------------------------------------------------------

def _band_mask(freqs: Tensor, lo: float, hi: float) -> Tensor:
    """a 0/1 float mask over `freqs`, inclusive at both ends.

    it is a *fixed* mask, not a soft one.  `freqs` is a constant grid -- nothing in a
    training run differentiates with respect to a frequency axis -- so the mask carries
    no gradient of its own and multiplying by it leaves `psd`'s gradient untouched.
    a soft mask would only blur the band edges without buying any gradient that a hard
    one loses.
    """
    return ((freqs >= lo) & (freqs <= hi)).to(freqs.dtype)


def _df(freqs: Tensor) -> Tensor:
    if freqs.numel() < 2:
        raise ValueError("need at least two frequency bins to know df")
    return freqs[1] - freqs[0]


def band_power(psd: Tensor, freqs: Tensor, lo: float, hi: float) -> Tensor:
    """absolute integrated power in `[lo, hi]` Hz, in units of x^2.

    integration is a **rectangle (df-sum)**, `sum(psd[in band]) * df`, not a
    trapezoid.  the reason is consistency with Parseval: the df-sum over the whole
    axis is exactly the variance estimate the density normalisation was built to give,
    while a trapezoid halves the two end bins and would make `band_power` over the
    full band disagree with `welch_psd`'s own units by a bin and a half.  on a uniform
    grid the two differ by O(df) at the band edges and nothing else.

    the mask is inclusive at both ends, so `band_power(..., 8, 13)` and
    `band_power(..., 13, 30)` double-count the 13 Hz bin.  that is the convention
    clinical band tables use and it is one bin; if you need a partition, pass
    `(13 + df, 30)`.

    returns `(...)` -- the leading axes of `psd`.
    """
    m = _band_mask(freqs, lo, hi)
    if float(m.sum()) == 0.0:
        raise ValueError(f"no frequency bin falls in [{lo}, {hi}] Hz (df={float(_df(freqs)):.4g})")
    return (psd * m).sum(dim=-1) * _df(freqs)


def relative_band_power(
    psd: Tensor,
    freqs: Tensor,
    lo: float,
    hi: float,
    total: Tuple[float, float] = (0.5, 45.0),
) -> Tensor:
    """power in `[lo, hi]` divided by power in `total`.  dimensionless, in `[0, 1]`
    when the band is inside `total`.

    `total` defaults to 0.5-45 Hz rather than 0-Nyquist on purpose.  below ~0.5 Hz a
    recording is drift and a simulation is whatever its slowest time constant happens
    to be doing; above ~45 Hz there is line noise and, in a simulation, whatever the
    integrator's step size leaves behind.  neither is a rhythm, and letting either into
    the denominator makes a relative power a statement about the artefact.

    the denominator is clamped away from zero, so a silent trace returns 0 rather than
    a NaN that would poison a whole training step.
    """
    num = band_power(psd, freqs, lo, hi)
    den = band_power(psd, freqs, total[0], total[1])
    return num / den.clamp_min(_EPS)


def aperiodic_fit(
    psd: Tensor,
    freqs: Tensor,
    lo: float = 1.0,
    hi: float = 45.0,
    exclude: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[Tensor, Tensor]:
    """closed-form least squares of `log10(psd)` on `log10(f)`, giving the 1/f
    background `psd ~= 10**offset * f**(-exponent)`.

    args
        lo, hi    the fit range in Hz.  bins at f <= 0 are always dropped.
        exclude   a list of `(lo, hi)` bands to leave OUT of the fit, so that an alpha
                  peak does not bend the background it is supposed to be measured
                  against.  for `peak_prominence` this is the whole point.

    returns `(offset, exponent)`, each `(...)` matching `psd`'s leading axes.
    `exponent` is the POSITIVE slope convention (a 1/f^2 process gives ~2.0, white
    noise gives ~0.0), because that is how the literature quotes it; the fitted slope
    of log10(psd) on log10(f) is `-exponent`.

    it is ordinary least squares in one pass -- no iterative solver, no peak model, so
    the whole thing is three sums and a division and differentiates cleanly.  two
    things to know about what that costs:

    * the fit is uniform over the LINEAR frequency bins it is handed, so an octave
      near 40 Hz carries ten times the weight of an octave near 4 Hz.  this is what
      FOOOF does with the same input and it is fine for comparing two spectra on the
      same grid; it is not fine for quoting an exponent against a paper that fitted
      on log-spaced bins.  say which you did.
    * `log10` of a chi-square-distributed periodogram bin is biased low by 0.25 in
      log10 units when M = 1 segment.  Welch averaging kills it: the bias is
      `(psi(M) - ln M)/ln 10`, about -0.007 at M = 31.  it lands entirely in `offset`
      and not in `exponent`, but if you ever fit a single-segment periodogram, the
      offset you get is not the offset you want.
    """
    m = _band_mask(freqs, lo, hi) * (freqs > 0).to(freqs.dtype)
    if exclude:
        for (elo, ehi) in exclude:
            m = m * (1.0 - _band_mask(freqs, elo, ehi))
    n = m.sum()
    if float(n) < 3.0:
        raise ValueError(
            f"aperiodic_fit has {float(n):.0f} usable bins in [{lo}, {hi}] Hz after exclusions; need >= 3"
        )
    lf = torch.log10(freqs.clamp_min(_EPS))
    ly = torch.log10(psd.clamp_min(_EPS))

    xbar = (lf * m).sum() / n
    ybar = (ly * m).sum(dim=-1) / n
    dx = (lf - xbar) * m                       # zero outside the mask, so the sums below are masked
    sxx = (dx * (lf - xbar)).sum()
    sxy = (ly * dx).sum(dim=-1)                # (ly - ybar) * dx summed == ly * dx summed, since dx sums to 0
    slope = sxy / sxx.clamp_min(_EPS)
    offset = ybar - slope * xbar
    return offset, -slope


def peak_frequency(
    psd: Tensor,
    freqs: Tensor,
    lo: float,
    hi: float,
    sharpness: float = 4.0,
) -> Tensor:
    """a differentiable soft-argmax: the power-weighted centroid of `psd**sharpness`
    inside `[lo, hi]`, in Hz.

    `f_hat = sum_k f_k * p_k**s / sum_k p_k**s`, over the band's bins only.

    it is not the argmax and it does not pretend to be.  two properties to hold on to:

    * **it is biased toward the band centre when the spectrum is flat.**  with a
      perfectly flat psd it returns exactly `(lo + hi) / 2` whatever `sharpness` is.
      so a peak frequency read off a spectrum with no peak in it is not a measurement,
      it is the band you chose -- always read `peak_prominence` beside it before
      believing a peak frequency, which is why both exist.
    * **`sharpness` trades bias against gradient smoothness.**  s -> infinity
      approaches the true argmax and its gradient approaches zero almost everywhere
      (an argmax is piecewise constant, so it cannot train); s = 1 is the plain
      spectral centroid, smooth but pulled hard by the 1/f background.  4 is the
      default because it puts ~250x more weight on a bin twice its neighbour's height
      while still passing useful gradient to the bins around the peak.

    numerically, the psd is divided by its in-band maximum before being raised to
    `sharpness`.  the result is mathematically identical (the ratio is scale
    invariant) and the scale is `detach`ed so no gradient flows through the max --
    it is purely there to stop `psd**4` overflowing float32 on a psd of 1e12.

    a uniform floor of 1e-30 is added to the weights.  the largest weight is 1.0 by
    construction, so for any real spectrum the floor changes nothing; for a band that
    is numerically EMPTY (a signal with no power there at all) it makes the answer the
    band centre, which is the documented flat-spectrum answer, instead of whatever
    denormal happened to be biggest.  without it an empty band returns something near
    0 Hz -- a number outside the band that was asked about, which reads as a bug in
    whatever consumed it rather than as "there is nothing here".
    """
    if sharpness <= 0:
        raise ValueError("sharpness must be positive")
    m = _band_mask(freqs, lo, hi)
    if float(m.sum()) == 0.0:
        raise ValueError(f"no frequency bin falls in [{lo}, {hi}] Hz")
    pb = psd * m
    scale = pb.amax(dim=-1, keepdim=True).detach().clamp_min(_EPS)
    w = ((pb / scale).clamp_min(0.0) ** float(sharpness) + 1e-30) * m
    den = w.sum(dim=-1)
    return (w * freqs).sum(dim=-1) / den


def peak_prominence(
    psd: Tensor,
    freqs: Tensor,
    lo: float,
    hi: float,
    exclude_for_background: Optional[Sequence[Tuple[float, float]]] = None,
    fit_lo: float = 1.0,
    fit_hi: float = 45.0,
) -> Tensor:
    """how far the power in `[lo, hi]` rises above the fitted 1/f background, in
    log10 units (decades).  0 means "exactly what the background predicts", +0.3 means
    twice the background, -0.3 means half.

    `log10( band_power(measured) / band_power(10**offset * f**-exponent) )`

    this is the quantity that separates "there is a peak at 10 Hz" from "there is more
    low-frequency power".  a raw alpha `band_power` rises when the whole 1/f background
    rises, and a raw `relative_band_power` rises when the background STEEPENS even with
    no alpha peak at all -- both of which have been read as "more alpha" in the
    literature and in this programme.  the background is fitted and divided out here so
    the number is about the bump.

    `exclude_for_background` defaults to `[(lo, hi)]`: the band under test is left out
    of its own background fit.  leaving it in lets a genuine peak drag the fitted line
    up through it, which shrinks the prominence of exactly the peaks you care about
    most.  pass an explicit list to exclude other bands too (e.g. a line-noise notch),
    or pass `[]` to fit through everything.

    `fit_lo`/`fit_hi` are the background's fit range, 1-45 Hz by default -- see
    `aperiodic_fit` for why the ends are where they are.
    """
    excl: Sequence[Tuple[float, float]]
    excl = [(lo, hi)] if exclude_for_background is None else exclude_for_background
    offset, exponent = aperiodic_fit(psd, freqs, lo=fit_lo, hi=fit_hi, exclude=excl)
    # the DC bin is f = 0 and f**-exponent there is inf, which would become inf*0 = NaN
    # the moment band_power masked it out.  substitute 1 Hz outside the band; the mask
    # discards it anyway and nothing infinite ever enters the graph.
    m = _band_mask(freqs, lo, hi)
    f = torch.where(m > 0, freqs, torch.ones_like(freqs))
    pred = torch.pow(10.0, offset.unsqueeze(-1)) * torch.pow(f, -exponent.unsqueeze(-1))
    meas = band_power(psd, freqs, lo, hi)
    base = band_power(pred, freqs, lo, hi)
    return torch.log10(meas.clamp_min(_EPS)) - torch.log10(base.clamp_min(_EPS))


# --------------------------------------------------------------------------------
# band-pass, analytic signal, and the phase quantities
# --------------------------------------------------------------------------------

def _analytic_band_transfer(
    n: int, fs: float, lo: float, hi: float, transition: Optional[float],
    dtype: torch.dtype, device: torch.device,
) -> Tensor:
    """the FFT-domain transfer that band-passes AND takes the analytic signal in one
    multiply.

    the band is a brick wall with a raised-cosine (Tukey) transition on each side.  a
    true brick wall is a sinc in time, and its ringing is not a cosmetic problem here:
    it is a *periodic* ripple at the band edge, and a periodic ripple is precisely what
    a phase-locking or phase-amplitude measure reports as coupling.  the transition
    costs some band selectivity and buys a filter that does not manufacture its own
    answer.

    the analytic part is Hilbert's: keep DC and Nyquist, double the positive
    frequencies, zero the negatives.  doing it in the same multiply means there is one
    FFT and one iFFT per band, and no chance of the two grids disagreeing.
    """
    if not (0.0 <= lo < hi):
        raise ValueError(f"need 0 <= lo < hi, got ({lo}, {hi})")
    if transition is None:
        transition = 0.25 * (hi - lo)
    transition = float(transition)
    if transition <= 0:
        raise ValueError("transition must be positive; a hard edge rings and the ringing reads as coupling")
    # the lower ramp is not allowed to reach DC.  for a narrow slow band -- the 0.5-1.5 Hz
    # slow oscillation is the case -- a quarter-width ramp would otherwise start below
    # zero and leave the drift the detrender just removed back in the band.
    trans_lo = min(transition, lo) if lo > 0.0 else transition
    trans_hi = transition

    f = torch.fft.fftfreq(n, d=1.0 / float(fs), dtype=dtype, device=device)
    fa = f.abs()
    lo_t, hi_t = lo - trans_lo, hi + trans_hi
    pi = math.pi
    if trans_lo > 0.0:
        rise = 0.5 * (1.0 - torch.cos(pi * (fa - lo_t).clamp(0.0, trans_lo) / trans_lo))
    else:
        rise = torch.ones_like(fa)
    fall = 0.5 * (1.0 + torch.cos(pi * (fa - hi).clamp(0.0, trans_hi) / trans_hi))
    g = torch.where(fa < lo_t, torch.zeros_like(fa), rise)
    g = torch.where(fa > hi, fall, g)
    g = torch.where(fa > hi_t, torch.zeros_like(fa), g)

    h = torch.zeros(n, dtype=dtype, device=device)
    if n % 2 == 0:
        h[0] = 1.0
        h[1 : n // 2] = 2.0
        h[n // 2] = 1.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return (g * h).to(torch.complex128 if dtype == torch.float64 else torch.complex64)


def bandpass_analytic(
    x: Tensor, fs: float, lo: float, hi: float, transition: Optional[float] = None
) -> Tensor:
    """band-pass `x` to `[lo, hi]` Hz and return its complex analytic signal `(..., T)`.

    `abs()` of the result is the instantaneous amplitude envelope of the band and
    `angle()` is its instantaneous phase in radians.  differentiable throughout.

    the filter is circular (it is one FFT over the whole trace), so the first and last
    few cycles wrap into each other.  for a training trace of a few seconds that is a
    real effect at the ends; discard a cycle of the slowest band at each end if the
    edges matter.  it is done this way rather than with an IIR filter because an IIR
    recursion of length T is a sequential graph of depth T in the backward pass, which
    is both slow and where the gradient goes to die.
    """
    n = x.shape[-1]
    h = _analytic_band_transfer(n, fs, lo, hi, transition, x.dtype if x.is_floating_point() else torch.float32, x.device)
    return torch.fft.ifft(torch.fft.fft(x.to(h.real.dtype), dim=-1) * h, dim=-1)


def _amp_phase(z: Tensor) -> Tuple[Tensor, Tensor]:
    """amplitude and phase of an analytic signal, with a finite gradient at zero.

    `abs()` and `angle()` both have a non-finite gradient where the signal vanishes, and
    a vanishing analytic signal is not hypothetical: an all-zero trace (a dead channel,
    or step 0 of a simulation) makes every sample exactly zero.

    the envelope uses `sqrt(re^2 + im^2 + eps)`.  the phase substitutes the direction
    `(1, 0)` wherever the magnitude is below the floor -- a `torch.where` routes the
    backward pass to that CONSTANT for those samples, so they contribute a phase of 0
    and a gradient of 0 instead of a NaN that would spread to every parameter.
    """
    re, im = z.real, z.imag
    p2 = re * re + im * im
    amp = torch.sqrt(p2 + _EPS)
    dead = p2 < _EPS
    re_s = torch.where(dead, torch.ones_like(re), re)
    im_s = torch.where(dead, torch.zeros_like(im), im)
    return amp, torch.atan2(im_s, re_s)


def pac_mi(
    x: Tensor,
    fs: float,
    phase_band: Tuple[float, float],
    amp_band: Tuple[float, float],
    n_bins: int = 18,
    kappa: float = 8.0,
) -> Tensor:
    """Tort's modulation index for phase-amplitude coupling, made differentiable.

    "is the amplitude of the fast band systematically bigger at some phases of the slow
    band than at others?"  hippocampal theta with gamma nested inside it is the case
    this programme is shaping toward, and it is a coupling, not a co-occurrence: both
    rhythms can be present with no nesting at all and this is the number that tells
    them apart.

    args
        phase_band  `(lo, hi)` Hz for the slow rhythm whose phase is read (e.g. 4-8).
        amp_band    `(lo, hi)` Hz for the fast rhythm whose envelope is read (e.g. 30-80).
                    it should be wide enough to contain the slow rhythm's sidebands,
                    i.e. at least `2 * phase_hi` wide, or the modulation is filtered
                    out before it can be measured.
        n_bins      phase bins.  18 is Tort's.
        kappa       concentration of the soft binning kernel (see below).

    returns `(...)`, in `[0, 1)`.  0 is a flat amplitude-by-phase distribution (no
    coupling); it rises toward 1 as the amplitude concentrates at one phase.  an
    exactly uniform distribution lands a few times 1e-16 BELOW zero, because
    `(log n - H)/log n` is a difference of two nearly equal floats; it is not clamped,
    so that a reader can tell "exactly uniform" from "clamped from something".  do not
    write a one-sided `mi < tol` test against it.

    **soft von Mises binning instead of a histogram.**  Tort assigns each sample to one
    of `n_bins` phase bins, which is a step function of phase and has zero gradient
    almost everywhere -- useless in a loss.  here sample `t` contributes to bin `b` with
    weight proportional to `exp(kappa * cos(phase_t - centre_b))`, normalised over `b`
    so each sample still contributes a total weight of 1.  the amplitude distribution is
        `P_b = sum_t w_tb * amp_t / sum_t amp_t`
    and the index is Tort's normalised KL divergence from uniform,
        `MI = (log(n_bins) - H(P)) / log(n_bins)`.

    two consequences of the softening, both of which matter when quoting a number:

    * **MI is exactly 0 for a uniform distribution, as in Tort.**  smoothing a uniform
      distribution leaves it uniform, so the zero point is not moved and "no coupling"
      still reads 0.
    * **MI is systematically LOWER than hard-binned Tort MI for real coupling**, because
      the kernel is a circular smoother and smoothing can only reduce the KL from
      uniform.  at kappa = 8 the kernel has a half-width of about 30 degrees, near the
      20-degree width of an 18-bin histogram, so the two are close -- but they are not
      the same number.  do not compare a value from here against a published Tort MI;
      compare it against another value from here.  lower kappa smooths harder and passes
      smoother gradient.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    zp = bandpass_analytic(x, fs, phase_band[0], phase_band[1])
    za = bandpass_analytic(x, fs, amp_band[0], amp_band[1])
    _, phase = _amp_phase(zp)
    amp, _ = _amp_phase(za)

    dtype = amp.dtype
    centres = (
        -math.pi
        + (torch.arange(n_bins, dtype=dtype, device=amp.device) + 0.5) * (2.0 * math.pi / n_bins)
    )
    # (..., T, B): softmax over bins of kappa*cos(phase - centre) -- a von Mises kernel,
    # normalised per sample so total weight per sample is exactly 1 whatever kappa is.
    d = phase.unsqueeze(-1) - centres
    w = torch.softmax(float(kappa) * torch.cos(d), dim=-1)

    num = (w * amp.unsqueeze(-1)).sum(dim=-2)              # (..., B)
    P = num / num.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    H = -(P * torch.log(P.clamp_min(_EPS))).sum(dim=-1)
    logn = math.log(n_bins)
    return (logn - H) / logn


def band_coherence(
    x: Tensor,
    y: Tensor,
    fs: float,
    lo: float,
    hi: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: Union[str, Tensor] = "hann",
) -> Tensor:
    """magnitude-squared coherence between `x` and `y`, averaged over `[lo, hi]` Hz.

    `C_xy(f) = |<X* Y>|^2 / (<|X|^2> <|Y|^2>)`, each `<.>` an average over the same
    segments `welch_psd` would use -- the cross- and auto-spectra come out of the one
    `_segment_fft` call path, so they cannot be computed on different grids or with
    different tapers.

    returns `(...)` in `[0, 1]`, the UNWEIGHTED mean of `C_xy(f)` over the band's bins.
    unweighted, not power-weighted, because coherence is already normalised: weighting
    by power would make the answer a statement about whichever frequency in the band is
    loudest, which for an EEG band is almost always its low edge.

    **the floor is not zero.**  with `M` independent segments, two unrelated signals
    give `E[C] = 1/M`, not 0 -- the estimate is a ratio of noisy quantities and its
    numerator does not know the two are unrelated.  `M = 1` gives `C = 1` identically,
    which is why fewer than two segments raises.  overlapping segments are not
    independent, so with the default 50% overlap the real floor sits somewhat above
    `1/M`; pass `noverlap=0` when you need the floor to be a number you can state.
    always report the floor beside the coherence.

    the denominator is clamped, so a silent channel gives 0 rather than a NaN.
    """
    fx, X, _, M = _segment_fft(x, fs, nperseg, noverlap, window, "constant", True)
    fy, Y, _, My = _segment_fft(y, fs, nperseg, noverlap, window, "constant", True)
    if M < 2 or My < 2:
        raise ValueError(
            f"coherence needs at least 2 segments (got {min(M, My)}); with one segment it is identically 1"
        )
    if X.shape[-1] != Y.shape[-1]:
        raise ValueError("x and y gave different frequency grids")

    # <X* Y> averaged over segments.  real/imag kept apart so no complex abs() is taken.
    cr = (X.real * Y.real + X.imag * Y.imag).mean(dim=-2)
    ci = (X.real * Y.imag - X.imag * Y.real).mean(dim=-2)
    pxx = (X.real * X.real + X.imag * X.imag).mean(dim=-2)
    pyy = (Y.real * Y.real + Y.imag * Y.imag).mean(dim=-2)
    coh = (cr * cr + ci * ci) / (pxx * pyy).clamp_min(_EPS)

    m = _band_mask(fx, lo, hi)
    if float(m.sum()) == 0.0:
        raise ValueError(f"no frequency bin falls in [{lo}, {hi}] Hz")
    return (coh * m).sum(dim=-1) / m.sum()


def phase_locking_value(x: Tensor, y: Tensor, fs: float, lo: float, hi: float) -> Tensor:
    """PLV between the band-passed analytic signals of `x` and `y`, in `[0, 1]`.

    `PLV = | mean_t exp(i (phase_x(t) - phase_y(t))) |`.  1 means the phase difference
    is constant over the whole trace whatever it is; 0 means it wanders uniformly.

    PLV and `band_coherence` answer different questions and disagree on purpose: PLV
    throws the amplitudes away, so two signals that lock only during the moments when
    both are strong score lower here and higher there.  quote which one you used.

    the same `1/M`-style floor logic applies: with `T` effectively independent samples
    the null PLV is about `sqrt(pi)/2 / sqrt(T_eff)`, and band-passing to a narrow band
    makes `T_eff` far smaller than `T`.  a PLV of 0.2 on a 4 Hz-wide band over 10 s is
    not evidence of anything.
    """
    zx = bandpass_analytic(x, fs, lo, hi)
    zy = bandpass_analytic(y, fs, lo, hi)
    _, px = _amp_phase(zx)
    _, py = _amp_phase(zy)
    d = px - py
    c = torch.cos(d).mean(dim=-1)
    s = torch.sin(d).mean(dim=-1)
    return torch.sqrt(c * c + s * s + _EPS)


# --------------------------------------------------------------------------------
# the loss
# --------------------------------------------------------------------------------

_KINDS = (
    "relative_power",
    "peak_frequency",
    "peak_prominence",
    "pac",
    "coherence",
    "aperiodic_exponent",
)


def _channel_index(spec: Any, device: torch.device) -> Tensor:
    if isinstance(spec, Tensor):
        return spec.to(device=device, dtype=torch.long)
    if isinstance(spec, int):
        return torch.tensor([spec], device=device, dtype=torch.long)
    return torch.tensor(list(spec), device=device, dtype=torch.long)


def _penalty(measured: Tensor, target: Union[float, Sequence[float]], scale: float) -> Tensor:
    """squared distance to a point target, or a two-sided hinge on an interval target.

    an interval target costs EXACTLY zero inside the interval -- `clamp(min=0)` returns
    a true 0.0, not a small number -- so a term that is satisfied contributes nothing
    to the gradient and stops pushing.  that is the point of it.
    """
    if isinstance(target, (tuple, list)) and len(target) == 2:
        lo, hi = float(target[0]), float(target[1])
        if lo > hi:
            raise ValueError(f"interval target ({lo}, {hi}) is empty")
        d = (lo - measured).clamp_min(0.0) + (measured - hi).clamp_min(0.0)
    else:
        d = measured - float(target)  # type: ignore[arg-type]
    return (d / float(scale)) ** 2


def spectral_loss(
    traces: Tensor,
    fs: float,
    targets: Sequence[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Tensor, Dict[str, Dict[str, Tensor]]]:
    """a smooth penalty on the distance between a batch of traces and a list of
    declared spectral facts.  differentiable end to end.

    args
        traces   `(B, C, T)` -- batch, channel/region, time.
        fs       Hz.
        targets  a list of specifications.  each is a dict:

            kind      one of "relative_power", "peak_frequency", "peak_prominence",
                      "pac", "coherence", "aperiodic_exponent".
            channels  channel indices (int, list, or LongTensor).  for "coherence" it
                      is a pair `(a, b)` or a `(P, 2)` list of pairs, and the term is
                      the mean over pairs.
            target    a float, OR a `(lo, hi)` tuple meaning "anywhere in this interval
                      costs nothing".
            band      `(lo, hi)` Hz -- needed by every kind except "pac".
            (kind-specific, all optional)
            total     "relative_power": the denominator band, default (0.5, 45).
            sharpness "peak_frequency", default 4.0.
            exclude_for_background, fit_lo, fit_hi   "peak_prominence".
            exclude   "aperiodic_exponent": bands to leave out of the fit.
            phase_band, amp_band, n_bins, kappa      "pac".
            nperseg, noverlap, window                the Welch settings for this term.
            weight    float, default 1.0.
            scale     float, default 1.0 -- see the units warning below.
            name      string for `parts`; defaults to "<kind>#<i>".

        weights  optional `{name: float}`, which OVERRIDES the spec's own `weight`.
                 it is there so a caller can sweep the weighting of a fixed target
                 list without rewriting the list.

    returns
        `total` -- a scalar, `sum(weight * penalty)`.
        `parts` -- `{name: {"value", "penalty", "weight", "weighted"}}`, where
                   **`value` is the UNWEIGHTED measured quantity**: the relative alpha
                   power itself, the peak in Hz, the exponent.  log that, not just the
                   loss.  a loss of 0.04 does not say whether alpha is at 0.4 or 0.8,
                   and "the spectrum we actually got" is the thing this programme is
                   trying to state.  `penalty` is unweighted too.

    **an interval target is the right default.**  a declared band IS an interval.
    "occipital alpha is 8-13 Hz" does not mean "the peak must be at 10.5 Hz", and a
    point target there spends gradient pushing a perfectly good 9.2 Hz peak toward a
    number nobody measured.  give a point target only where a point was measured --
    the 13.45 Hz spindle is one, and even that has a width.

    **penalties in different kinds are in different units.**  a peak frequency error is
    in Hz and squares to Hz^2, a relative power error is dimensionless and squares to
    ~1e-2, so a naive `weight=1.0` on both makes the frequency term about ten thousand
    times louder.  `scale` is the unit converter: set it to the tolerance you would
    accept for that quantity (1.0 Hz for a peak, 0.05 for a relative power) and the
    weights then mean what they look like they mean.

    the penalty is computed PER (batch, channel) and then averaged, not computed on the
    average.  averaging first would let one channel's excess alpha pay for another's
    deficit and report a satisfied term over two wrong regions.

    the Welch default here is a 2 s segment (`nperseg = min(T, 2*fs)`), NOT
    `welch_psd`'s scipy-compatible `min(256, T)`: the slow oscillation near 1 Hz needs
    at least two seconds to be a bin rather than a trend.
    """
    if traces.ndim != 3:
        raise ValueError(f"traces must be (B, C, T), got {tuple(traces.shape)}")
    B, C, T = traces.shape
    dev = traces.device
    weights = weights or {}

    total = traces.new_zeros(())
    parts: Dict[str, Dict[str, Tensor]] = {}

    for i, spec in enumerate(targets):
        kind = spec["kind"]
        if kind not in _KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {_KINDS}")
        name = spec.get("name") or f"{kind}#{i}"
        if name in parts:
            raise ValueError(f"duplicate target name {name!r}")
        if "target" not in spec:
            raise ValueError(f"target {name!r} has no 'target'")

        nperseg = spec.get("nperseg", min(T, max(2, int(round(2.0 * fs)))))
        noverlap = spec.get("noverlap", None)
        window = spec.get("window", "hann")
        idx = _channel_index(spec.get("channels", list(range(C))), dev)

        if kind == "coherence":
            pairs = idx.reshape(-1, 2)
            xs = traces[:, pairs[:, 0], :]
            ys = traces[:, pairs[:, 1], :]
            lo, hi = spec["band"]
            measured = band_coherence(xs, ys, fs, lo, hi, nperseg, noverlap, window)
        elif kind == "pac":
            xs = traces[:, idx, :]
            measured = pac_mi(
                xs, fs, tuple(spec["phase_band"]), tuple(spec["amp_band"]),
                n_bins=int(spec.get("n_bins", 18)), kappa=float(spec.get("kappa", 8.0)),
            )
        else:
            xs = traces[:, idx, :]
            freqs, psd = welch_psd(xs, fs, nperseg, noverlap, window)
            if kind == "relative_power":
                lo, hi = spec["band"]
                measured = relative_band_power(psd, freqs, lo, hi, tuple(spec.get("total", (0.5, 45.0))))
            elif kind == "peak_frequency":
                lo, hi = spec["band"]
                measured = peak_frequency(psd, freqs, lo, hi, float(spec.get("sharpness", 4.0)))
            elif kind == "peak_prominence":
                lo, hi = spec["band"]
                measured = peak_prominence(
                    psd, freqs, lo, hi,
                    exclude_for_background=spec.get("exclude_for_background", None),
                    fit_lo=float(spec.get("fit_lo", 1.0)), fit_hi=float(spec.get("fit_hi", 45.0)),
                )
            else:  # aperiodic_exponent
                lo, hi = spec.get("band", (1.0, 45.0))
                _, measured = aperiodic_fit(psd, freqs, lo, hi, exclude=spec.get("exclude", None))

        pen = _penalty(measured, spec["target"], float(spec.get("scale", 1.0))).mean()
        w = float(weights.get(name, spec.get("weight", 1.0)))
        total = total + w * pen
        parts[name] = {
            "value": measured.detach().mean(),
            "penalty": pen.detach(),
            "weight": torch.as_tensor(w, dtype=traces.dtype, device=dev),
            "weighted": (w * pen).detach(),
        }

    return total, parts
