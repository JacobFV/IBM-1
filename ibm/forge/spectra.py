"""turning a recording into the evidence the spectral form consumes.

`ibm.fields.uncertainty.spectral` says what a belief about a spectral component
*is*.  `ibm.forge.fit` says how a likelihood moves theta.  neither of them knows
how to get from a file of sampled volts to a number that a likelihood can be
computed from, and that gap is where a fitting pipeline usually acquires its
worst habit: it computes a power spectrum, subtracts a model, and calls the
squared difference a log-likelihood with an invented weight.

the weight is not free and it is not a nuisance.  a periodogram averaged over M
independent windows is a gamma-distributed estimate of the true spectrum with
shape M -- the sampling distribution of a spectral estimator is one of the few
things in this whole system that is known exactly rather than assumed -- so the
precision of the evidence follows from M and from nothing else.  a recording
twice as long, split the same way, is worth twice the precision.  a recording
whose windows were mostly rejected as artefact is worth less, automatically,
because there are fewer of them.  that is the entire reason this module exists
and it is why `SpectralEvidence` carries `n_windows` beside the psd rather than a
scalar "noise level" someone chose.

three consequences shape the code.

**windowing is onto a `TemporalBasis`, not onto an arbitrary segment length.**
the basis is the object the rest of ibm-1 states beliefs over; if the evidence
were computed on a different grid, every comparison would carry an interpolation
nobody declared.  so the caller supplies the basis and the windows are its `n`.

**windows do not overlap.**  welch's 50% overlap buys a lower-variance estimate
and destroys the one thing this module needs, which is that M is the number of
*independent* observations.  overlapped segments share samples, the effective
degrees of freedom fall below 2M by a factor that depends on the taper, and the
resulting likelihood is overconfident by an amount that is invisible.  paying a
slightly noisier estimate for an exactly known precision is the right trade here.

**degrees of freedom come from the basis's own multiplicity.**  a bin away from
DC and nyquist is a (cos, sin) pair and contributes 2 dof per window; DC and
nyquist are single real coefficients and contribute 1.  the spectral form already
declares that degeneracy, for its own reasons, and reusing it here means the two
cannot drift apart.

what this module does *not* do is decide what the psd should look like.  the
model spectrum comes from a registered prior builder in `ibm.fields.priors`, and
the likelihood below takes it as an argument.  keeping those apart is what lets
the same evidence score a fitted posterior and an untouched prior on identical
terms, which is the only comparison that means anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln, polygamma

from ibm.fields.uncertainty.spectral import SpectralGaussian, TemporalBasis
from ibm.vocabulary import Band, FULL

_EPS = 1e-300


# ---------------------------------------------------------------------------
# where the bytes are
# ---------------------------------------------------------------------------


def local_root(source_id: str, root: Path | None = None) -> Path:
    """the directory holding a source's raw bytes, per its own card.

    `data/sources/<id>/raw/.location.yaml` is committed and the bytes are not, so
    this is the only supported way to find them.  a hardcoded path in a script is
    the failure mode this prevents: it works on one machine, silently reads
    nothing on another, and the experiment reports a result computed over an
    empty file list.  so an unset or missing `local_root` raises and names the
    card, because the fix is to edit that card and not this code.
    """
    from ibm.forge.bind import repo_root

    import yaml

    base = Path(root) if root is not None else repo_root()
    loc = base / "data" / "sources" / source_id / "raw" / ".location.yaml"
    if not loc.is_file():
        raise FileNotFoundError(
            f"source {source_id!r} has no {loc.relative_to(base)}; the card has never been "
            "acquired, and nothing here may invent a path for it")
    with loc.open() as fh:
        d = yaml.safe_load(fh) or {}
    lr = d.get("local_root")
    if not lr:
        raise FileNotFoundError(
            f"source {source_id!r}: {loc.relative_to(base)} sets local_root: {lr!r}.  the card "
            "is declared but the bytes have not been acquired on this machine; set local_root "
            "there rather than passing a path in.")
    p = Path(str(lr)).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(
            f"source {source_id!r}: local_root {p} in {loc.relative_to(base)} does not exist")
    return p


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------


def window(x: np.ndarray, basis: TemporalBasis, *, detrend: bool = True) -> np.ndarray:
    """cut a recording into non-overlapping windows of the basis's length.

    returns `(..., M, n)`.  the trailing remainder is dropped rather than
    zero-padded: a padded window is not an independent observation of the
    spectrum, and counting it as one is exactly the overconfidence this module is
    built to avoid.

    detrending removes each window's mean and linear trend.  it costs two degrees
    of freedom at the very bottom of the band, which is why every fit here starts
    above the first few bins, and it stops slow electrode drift -- which the
    eegmmidb card names as a confound and which no amplifier removes -- from
    appearing as aperiodic power.
    """
    x = np.asarray(x, float)
    n = basis.n
    m = x.shape[-1] // n
    if m < 1:
        raise ValueError(f"recording of {x.shape[-1]} samples is shorter than one "
                         f"{n}-sample window")
    w = x[..., : m * n].reshape(x.shape[:-1] + (m, n))
    if detrend:
        t = np.linspace(-1.0, 1.0, n)
        t = t - t.mean()
        w = w - w.mean(-1, keepdims=True)
        w = w - (w @ t / (t @ t))[..., None] * t
    return w


def clean_windows(w: np.ndarray, *, max_mad: float = 6.0) -> np.ndarray:
    """boolean mask over windows, dropping the ones a robust amplitude rule rejects.

    a blink, a swallow or an electrode pop is orders of magnitude above cortical
    signal and would otherwise set the broadband level of the whole recording.
    the rule is deliberately crude and deliberately *reported*: the number of
    surviving windows is the precision of the evidence, so rejecting more is not
    free, and a recording that loses most of its windows loses most of its vote
    rather than quietly contributing a cleaned-up spectrum at full confidence.

    the statistic is the median absolute deviation of per-window log rms, taken
    across the whole recording, so the threshold adapts to the subject's own
    amplitude rather than to an absolute microvolt figure that would depend on
    the reference montage.
    """
    rms = np.sqrt(np.mean(w ** 2, -1))                     # (..., M)
    lr = np.log(np.maximum(rms, _EPS))
    if lr.ndim > 1:                                         # collapse channels
        lr = lr.mean(tuple(range(lr.ndim - 1)))
    med = np.median(lr)
    mad = np.median(np.abs(lr - med)) * 1.4826
    if mad <= 0:
        return np.ones(lr.shape, bool)
    return np.abs(lr - med) <= max_mad * mad


# ---------------------------------------------------------------------------
# the estimate
# ---------------------------------------------------------------------------


def _tapers(n: int, method: str, nw: float, n_tapers: int) -> np.ndarray:
    if method == "welch":
        return np.hanning(n + 1)[:n][None, :]
    if method == "multitaper":
        from scipy.signal.windows import dpss
        return np.atleast_2d(dpss(n, nw, Kmax=n_tapers))
    raise ValueError(f"unknown psd method {method!r}; have 'welch' and 'multitaper'")


def periodogram(w: np.ndarray, basis: TemporalBasis, *, method: str = "welch",
                nw: float = 3.0, n_tapers: int = 5) -> np.ndarray:
    """one-sided power spectral density of each window, averaged over tapers.

    scaled as a density -- power per hertz -- so that windows of different length
    or recordings at different sampling rates are on the same axis.  the factor
    is the basis's multiplicity over the taper energy, which is the standard
    one-sided density normalization written in terms of the degeneracy the
    spectral form already declares.

    multitaper is offered because two channels of a whole-night recording want a
    lower-variance estimate than a hann window gives, and it is *not* the default
    because its K tapers are only approximately independent, so the honest
    degrees of freedom are a little below 2MK -- a small overconfidence that
    welch does not have.
    """
    tap = _tapers(basis.n, method, nw, n_tapers)
    fs = 1.0 / basis.dt
    out = np.zeros(w.shape[:-1] + (basis.k,))
    for t in tap:
        z = np.fft.rfft(w * t, axis=-1)[..., : basis.k]
        out += np.abs(z) ** 2
    return out * (basis.multiplicity / (len(tap) * fs * float(np.sum(tap[0] ** 2))))


@dataclass(frozen=True)
class SpectralEvidence:
    """an observed psd and the precision it actually earned.

    `psd` is the averaged one-sided density on `basis.freqs_hz`.  `n_windows` is
    how many independent windows went into that average and `n_tapers` how many
    per window; together with the basis's multiplicity they give the degrees of
    freedom of the estimate, and the degrees of freedom are the precision.  there
    is no separate noise parameter and there is nowhere to put one.

    the sampling distribution is exact rather than assumed: an averaged
    periodogram of a stationary gaussian process is `S * Gamma(a, 1/a)` with
    `a = dof/2`, so `log_likelihood` is a real density and not a least-squares
    objective wearing a likelihood's name.  that matters here because the two
    disagree in a specific direction -- a squared error on the raw psd is
    dominated entirely by the low-frequency bins, where the power is, and a
    squared error on the log psd throws away the fact that the estimate's
    variance is known.

    `band` records the range the evidence is claimed over.  it is not cosmetic:
    the eegmmidb card declares 0-80 Hz and its own notes say the amplifier drifts
    below ~0.1 Hz, and a fit that reads precision there is fitting a confound the
    card already warned about.
    """

    basis: TemporalBasis
    psd: np.ndarray                  # (..., k), one-sided density
    n_windows: int
    n_tapers: int = 1
    band: Band = FULL
    channels: tuple[str, ...] = ()
    source: str = ""
    note: str = ""

    # -- precision -------------------------------------------------------

    @property
    def dof(self) -> np.ndarray:
        """chi-square degrees of freedom per bin: multiplicity x windows x tapers.

        the (cos, sin) pair away from DC contributes two real degrees of freedom
        per window and DC and nyquist contribute one, which is the same
        degeneracy `TemporalBasis.multiplicity` exists to express.  reusing it
        rather than writing `2 * M` keeps the evidence and the belief describing
        the same object.
        """
        return self.basis.multiplicity.astype(float) * self.n_windows * self.n_tapers

    @property
    def log_precision(self) -> np.ndarray:
        """1 / var[log psd_hat]: the gaussian-in-logs precision this evidence carries.

        `J = 1 / trigamma(dof/2)`, which is the exact variance of the log of a
        gamma variate, and it is what §4's `J' = J + dJ` wants when this evidence
        is fused with something else rather than optimized against.  it rises
        with the window count, roughly as `dof/2`, which is the quantitative
        version of "more windows, more precision".
        """
        return 1.0 / polygamma(1, 0.5 * self.dof)

    @property
    def log_bias(self) -> np.ndarray:
        """E[log psd_hat] - log psd: `digamma(a) - log a`, always negative.

        a small point with a large consequence.  the log of an averaged
        periodogram is biased low, by about `-1/dof`, so a model fitted by least
        squares on log psd comes out systematically under-powered -- and the bias
        is larger where fewer windows survived, which is to say larger for
        exactly the noisiest recordings.  the gamma likelihood below has no such
        bias, and this property exists so the log-domain path can correct for it.
        """
        a = 0.5 * self.dof
        return polygamma(0, a) - np.log(a)

    # -- selection -------------------------------------------------------

    def in_band(self, band: Band) -> "SpectralEvidence":
        """drop the bins above `band`'s ceiling, keeping the frequency axis intact.

        deliberately a truncation from above rather than a slice out of the
        middle.  a `TemporalBasis` derives its frequencies from `n` and `dt`, so
        the only band restriction it can express without lying about what
        frequency each bin is, is a `kmax`.  a low-edge restriction is a
        selection over bins, which is what `log_likelihood(band=...)` does
        directly -- it never has to rebuild a basis, and so it never has to
        misrepresent one.
        """
        i = self.basis.band_indices(band)
        if i.size == 0:
            raise ValueError(f"{band!r} contains no bin of a basis with "
                             f"{self.basis.k} bins up to {self.basis.nyquist_hz:g} Hz")
        b = TemporalBasis(self.basis.n, self.basis.dt, kmax=int(i[-1]) + 1)
        return SpectralEvidence(b, self.psd[..., : b.k], self.n_windows, self.n_tapers,
                                band & self.band, self.channels, self.source, self.note)

    def mask(self, band: Band) -> np.ndarray:
        return self.basis.band_mask(band)

    # -- likelihood ------------------------------------------------------

    def log_likelihood(self, model_psd: np.ndarray, *, band: Band | None = None,
                       gain: float | np.ndarray | None = None) -> float:
        """log p(psd_hat | model), summed over the bins of `band`.

        `gain` is an overall multiplicative scale on the model.  passing `None`
        profiles it out at its exact maximum-likelihood value, which is what
        should almost always happen: absolute eeg power is set by skull
        conductivity, electrode impedance and the reference montage, none of
        which the neural prior is a claim about, so leaving the scale in would
        make the fit a fit of the amplifier.  what remains after profiling is the
        *shape* of the spectrum, which is what the prior actually asserts.

        one degree of freedom is spent on the profile.  over the ~180 bins a fit
        here uses that is negligible, and it is the same one degree for the prior
        and for the posterior, so the comparison between them is not affected.
        """
        i = self._idx(band)
        s = np.asarray(self.psd, float)[..., i]
        g = np.maximum(np.asarray(model_psd, float)[..., i], _EPS)
        a = 0.5 * self.dof[i]
        if gain is None:
            gain = self.profile_gain(model_psd, band=band)
        g = g * np.asarray(gain, float)[..., None] if np.ndim(gain) else g * float(gain)
        return float(np.sum(a * np.log(a) - gammaln(a) + (a - 1.0) * np.log(np.maximum(s, _EPS))
                            - a * np.log(g) - a * s / g))

    def profile_gain(self, model_psd: np.ndarray, *, band: Band | None = None) -> float:
        """the maximum-likelihood overall scale of `model_psd` against this evidence.

        for `S = c g` the gamma likelihood is maximized by the dof-weighted
        arithmetic mean of `psd_hat / g`, in closed form.  closed form matters:
        the alternative is another optimizer dimension per recording, and with
        one nuisance scale per recording that is most of the parameter vector.
        """
        i = self._idx(band)
        s = np.asarray(self.psd, float)[..., i]
        g = np.maximum(np.asarray(model_psd, float)[..., i], _EPS)
        a = 0.5 * self.dof[i]
        return float(np.sum(a * s / g) / np.sum(a))

    def r2_log(self, model_psd: np.ndarray, *, band: Band | None = None,
               gain: float | None = None) -> float:
        """variance of log10 psd explained by the model, at the likelihood's own gain.

        reported beside the log-likelihood because a likelihood in nats is
        unreadable and because the two can disagree: r^2 on logs weights every
        decade equally, the likelihood weights every bin by its degrees of
        freedom.  the gain is *not* re-fitted to maximize r^2 -- it is the one the
        likelihood chose -- so this is a conservative figure rather than a
        flattering one.
        """
        i = self._idx(band)
        if gain is None:
            gain = self.profile_gain(model_psd, band=band)
        y = np.log10(np.maximum(np.asarray(self.psd, float)[..., i], _EPS))
        f = np.log10(np.maximum(np.asarray(model_psd, float)[..., i] * gain, _EPS))
        ss = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - float(np.sum((y - f) ** 2)) / max(ss, _EPS)

    def _idx(self, band: Band | None) -> np.ndarray:
        if band is None:
            return np.arange(self.basis.k)
        i = self.basis.band_indices(band)
        if i.size == 0:
            raise ValueError(f"{band!r} contains no bin of this evidence")
        return i

    # -- as a belief -----------------------------------------------------

    def as_spectral_gaussian(self) -> SpectralGaussian:
        """the measured psd read as a belief about the state, zero mean, no phase.

        worth being precise about what is and is not carried across.  a
        `SpectralGaussian`'s `psd` is the *variance of the state's coefficients*;
        `n_windows` is the confidence in the *estimate of that variance*.  they
        are different objects and the spectral form deliberately holds only the
        first, so this conversion drops the precision and everything downstream
        of it treats the measured spectrum as exact.

        that is the right thing for fusing evidence about state, which is what
        §4's `J' = J + dJ` composes, and the wrong thing for fitting a parameter
        of the spectrum's shape, which is what `log_likelihood` is for.  the
        relation is left `None` because an averaged periodogram has discarded
        phase by construction: it is amplitude with uniform phase, which is
        precisely the resting-rhythm belief and never an evoked one.
        """
        return SpectralGaussian.from_psd(self.basis, self.psd)

    def __str__(self) -> str:
        lo, hi = self.band.lo_hz, self.band.hi_hz
        return (f"{self.source or 'evidence'}: {self.n_windows} x "
                f"{self.basis.duration_s:g} s windows"
                + (f" x {self.n_tapers} tapers" if self.n_tapers > 1 else "")
                + f", {self.basis.k} bins to {self.basis.nyquist_hz:g} Hz, claimed over "
                  f"{lo:g}-{hi:g} Hz, dof {self.dof.max():.0f}"
                + (f"  [{self.note}]" if self.note else ""))


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def evidence_from_signal(x: np.ndarray, fs: float, *, window_s: float = 4.0,
                         band: Band = FULL, method: str = "welch", nw: float = 3.0,
                         n_tapers: int = 5, max_mad: float = 6.0,
                         average_channels: bool = True, channels: tuple[str, ...] = (),
                         source: str = "", note: str = "") -> SpectralEvidence:
    """a `(channels, samples)` array to one `SpectralEvidence`.

    `window_s` sets the frequency resolution and, against a fixed recording
    length, trades it against precision: 4 s gives 0.25 Hz bins, which resolves
    an alpha peak whose width is 2 Hz, and leaves enough windows in a one-minute
    run for the estimate to mean something.  the two are the same budget and the
    caller has to spend it.

    `average_channels` averages the *psds*, never the traces -- averaging traces
    across a montage cancels whatever is not common to them, which for cortical
    rhythms is most of it.  and the window count is not multiplied by the channel
    count when it does so: scalp channels are strongly correlated, so C channels
    are worth appreciably less than C independent observations, and inflating the
    degrees of freedom by C would be the exact fabrication this module exists to
    refuse.  the estimate gets quieter; the stated precision does not rise.
    """
    x = np.atleast_2d(np.asarray(x, float))
    basis = TemporalBasis(int(round(window_s * fs)), 1.0 / float(fs))
    w = window(x, basis)
    keep = clean_windows(w, max_mad=max_mad)
    if not keep.any():
        raise ValueError(f"{source or 'recording'}: every window was rejected as artefact")
    w = w[..., keep, :]
    p = periodogram(w, basis, method=method, nw=nw, n_tapers=n_tapers)
    p = p.mean(-2)                                        # over windows
    if average_channels and p.ndim > 1:
        p = p.mean(tuple(range(p.ndim - 1)))
    return SpectralEvidence(basis, p, int(keep.sum()),
                            n_tapers=(n_tapers if method == "multitaper" else 1),
                            band=band, channels=tuple(channels), source=source, note=note)


__all__ = ["SpectralEvidence", "clean_windows", "evidence_from_signal", "local_root",
           "periodogram", "window"]
