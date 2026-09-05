"""spectral gaussian: a belief over the temporal laplacian spectrum.

for a component in which several timescales interact, take the laplacian of the
**time** axis of that single state variable over a window of N samples.  taking
the cyclic path graph,

    lambda_k = 4 sin^2(pi k / N)

three properties carry the design.

*the laplacian spectrum is the power spectrum.*  lambda_k is monotone in omega_k,
so a distribution over it is a distribution over the psd; a diagonal covariance
is a stationary gaussian process and 1/f^beta background is a one-line prior.

*amplitude and phase are separately representable.*  every eigenvalue away from
DC and nyquist is doubly degenerate and its eigenspace is the (cos, sin) pair.
an isotropic gaussian there is known amplitude with uniform phase -- an ongoing
rhythm.  an anisotropic one is phase preference -- an evoked response.  the
transition between them across a trial is what an evoked response *is*.

*off-diagonal covariance is cross-frequency structure.*  diagonal is stationary,
band-block correlation is phase-amplitude coupling, dense is an arbitrary
transient.  what a model materializes is the hypothesis class it entertains.

and the practical payoff: conduction delay and every linear time-invariant
coupling are diagonal here -- a phase ramp and a transfer function -- so the
stiffness and the per-edge history buffers that dominate conventional brain
simulators do not arise.

the cost: a pointwise nonlinearity is local in time and dense in frequency, so a
nonlinear f must round-trip through the time domain and does not preserve
gaussianity.  and the window is finite, so continuity between windows is imposed
rather than inherited.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from ibm.fields.uncertainty.base import Conversion, UncertaintyForm, conversion, register
from ibm.fields.uncertainty.scalar import ScalarGaussian
from ibm.vocabulary import Band

_EPS = 1e-30


@dataclass(frozen=True)
class TemporalBasis:
    """a windowed temporal laplacian eigenbasis.

    `n` is set by the fastest *nonlinear* process, not the fastest process: linear
    and delayed couplings are exact at any n, but a pointwise nonlinearity must be
    evaluated in time and aliases if the window undersamples its own bandwidth.
    `kmax` is the temporal analogue of coarsening r(q).
    """

    n: int
    dt: float
    kmax: int | None = None

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("window needs at least two samples")

    @property
    def duration_s(self) -> float: return self.n * self.dt
    @property
    def k(self) -> int: return self.kmax if self.kmax is not None else self.n // 2 + 1
    @property
    def nyquist_hz(self) -> float: return 0.5 / self.dt

    @cached_property
    def freqs_hz(self) -> np.ndarray:
        return np.fft.rfftfreq(self.n, d=self.dt)[: self.k]

    @cached_property
    def omega(self) -> np.ndarray:
        return 2.0 * np.pi * self.freqs_hz

    @cached_property
    def eigenvalues(self) -> np.ndarray:
        """lambda_k = 4 sin^2(omega_k dt / 2); tends to (omega_k dt)^2."""
        return 4.0 * np.sin(np.pi * np.arange(self.k) / self.n) ** 2

    @cached_property
    def multiplicity(self) -> np.ndarray:
        """1 at DC and nyquist, else 2.  the 2-planes are amplitude/phase pairs."""
        m = np.full(self.k, 2, dtype=np.int8)
        m[0] = 1
        if self.n % 2 == 0 and self.k == self.n // 2 + 1:
            m[-1] = 1
        return m

    def band_mask(self, band: Band) -> np.ndarray:
        f = self.freqs_hz
        return (f >= band.lo_hz) & (f < band.hi_hz)

    def band_indices(self, band: Band) -> np.ndarray:
        return np.flatnonzero(self.band_mask(band))

    def truncated(self, band: Band) -> "TemporalBasis":
        if not np.isfinite(band.hi_hz):
            return self
        keep = int(np.searchsorted(np.fft.rfftfreq(self.n, d=self.dt), band.hi_hz, "right"))
        return TemporalBasis(self.n, self.dt, kmax=max(1, min(keep, self.n // 2 + 1)))

    def analyze(self, x: np.ndarray) -> np.ndarray:
        return np.fft.rfft(x, axis=-1, norm="ortho")[..., : self.k]

    def synthesize(self, z: np.ndarray) -> np.ndarray:
        full = self.n // 2 + 1
        if z.shape[-1] < full:
            z = np.concatenate([z, np.zeros(z.shape[:-1] + (full - z.shape[-1],), dtype=z.dtype)], -1)
        return np.fft.irfft(z, n=self.n, axis=-1, norm="ortho")

    def roundtrip(self, f, z: np.ndarray) -> np.ndarray:
        """apply a pointwise-in-time nonlinearity to spectral state.

        the pseudo-spectral escape hatch, and where the cost of this form lives.
        a product in time is a convolution in frequency, so a nonlinear process
        cannot read only the components that concern it -- it must see the whole
        retained band.  O(K log K), and it does not preserve gaussianity; the
        caller reprojects by moment matching.
        """
        return self.analyze(f(self.synthesize(z)))


@dataclass(frozen=True)
class SpectralGaussian:
    """belief over the spectral coefficients of a batch of state variables.

    mean      mu_k = E[z_k]                 complex
    psd       p_k  = E[|z_k - mu_k|^2]      real, >= 0   -- circular part
    relation  r_k  = E[(z_k - mu_k)^2]      complex, |r| <= p -- non-circularity

    psd alone is a circle on the eigenplane: known amplitude, uniform phase.
    relation breaks the circle and is exactly phase preference.  factor is an
    optional low-rank term contributing F F^H across frequencies, where every
    non-stationarity lives.
    """

    basis: TemporalBasis
    mean: np.ndarray
    psd: np.ndarray
    relation: np.ndarray | None = None
    factor: np.ndarray | None = None

    # -- construction ---------------------------------------------------

    @classmethod
    def zeros(cls, basis: TemporalBasis, shape: tuple[int, ...] = ()) -> "SpectralGaussian":
        z = np.zeros(shape + (basis.k,), dtype=np.complex128)
        return cls(basis, z, np.zeros(shape + (basis.k,)))

    @classmethod
    def from_psd(cls, basis: TemporalBasis, psd: np.ndarray,
                 mean: np.ndarray | None = None, shape: tuple[int, ...] = ()) -> "SpectralGaussian":
        psd = np.broadcast_to(np.asarray(psd, float), shape + (basis.k,)).copy()
        if mean is None:
            mean = np.zeros(shape + (basis.k,), dtype=np.complex128)
        return cls(basis, np.asarray(mean, np.complex128), psd)

    @classmethod
    def moment_match(cls, basis: TemporalBasis, members: np.ndarray) -> "SpectralGaussian":
        """project an ensemble of time-domain trajectories back onto this form.

        the only place a nonlinearity's non-gaussian output is discarded --
        deliberately and visibly.
        """
        z = basis.analyze(members)
        m = z.shape[0]
        mu = z.mean(0)
        d = z - mu
        c = m / max(m - 1, 1)
        return cls(basis, mu, (np.abs(d) ** 2).mean(0).real * c, (d * d).mean(0) * c)

    # -- structure ------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]: return self.mean.shape[:-1]
    @property
    def rank(self) -> int: return 0 if self.factor is None else self.factor.shape[-1]

    def total_psd(self) -> np.ndarray:
        p = self.psd
        if self.factor is not None:
            p = p + np.sum(np.abs(self.factor) ** 2, -1)
        return p

    def phase_concentration(self) -> np.ndarray:
        """|relation| / psd in [0, 1]: 0 uniform phase, 1 phase-determined.

        the most diagnostic scalar in this form.  an ongoing rhythm sits near 0,
        an evoked response near 1.
        """
        p = self.total_psd()
        if self.relation is None:
            return np.zeros_like(p)
        return np.abs(self.relation) / np.maximum(p, _EPS)

    def _canon_relation(self) -> np.ndarray:
        r = (self.relation if self.relation is not None else np.zeros_like(self.mean)).copy()
        one = self.basis.multiplicity == 1
        r[..., one] = self.total_psd()[..., one].astype(np.complex128)
        return r

    # -- linear time-invariant action -----------------------------------

    def apply_transfer(self, H: np.ndarray) -> "SpectralGaussian":
        """push through an LTI filter H(omega).  exact and diagonal.

        every LTI and pure-delay process reduces to this call.  the mean rotates
        and scales, power scales by |H|^2, and the relation scales by H^2 -- so a
        filter with phase creates phase preference exactly as it should.
        """
        H = np.asarray(H)
        return replace(self,
            mean=self.mean * H,
            psd=self.psd * (np.abs(H) ** 2),
            relation=None if self.relation is None else self.relation * (H ** 2),
            factor=None if self.factor is None else self.factor * H[..., None])

    def differentiate(self) -> "SpectralGaussian":
        return self.apply_transfer(1j * self.basis.omega)

    def delay(self, tau_s) -> "SpectralGaussian":
        """conduction delay as a phase ramp.

        heterogeneous tract delays applied in one operation rather than through
        per-edge history buffers.  this is the single largest practical reason
        the spectral form earns its cost.
        """
        tau = np.asarray(tau_s)
        H = np.exp(-1j * (tau[..., None] * self.basis.omega if tau.ndim else tau * self.basis.omega))
        return self.apply_transfer(H)

    def gain(self, g) -> "SpectralGaussian":
        g = np.asarray(g)
        if g.ndim and g.shape[-1] != self.basis.k:
            g = g[..., None]
        return self.apply_transfer(g.astype(np.complex128))

    # -- band restriction ------------------------------------------------

    def marginal(self, band: Band) -> "SpectralGaussian":
        """restrict to the components inside `band`.  exact.

        this is what "a process interacts with only the components that concern
        it" means operationally, and why band-limited coupling costs nothing in
        accuracy.
        """
        i = self.basis.band_indices(band)
        return SpectralGaussian(
            TemporalBasis(self.basis.n, self.basis.dt, kmax=len(i)),
            self.mean[..., i], self.psd[..., i],
            None if self.relation is None else self.relation[..., i],
            None if self.factor is None else self.factor[..., i, :])

    def truncate(self, k: int) -> "SpectralGaussian":
        k = min(k, self.basis.k)
        return SpectralGaussian(TemporalBasis(self.basis.n, self.basis.dt, kmax=k),
            self.mean[..., :k], self.psd[..., :k],
            None if self.relation is None else self.relation[..., :k],
            None if self.factor is None else self.factor[..., :k, :])

    # -- pressure and evidence -------------------------------------------

    def pressure(self, dz: np.ndarray, dpsd: np.ndarray | float = 0.0) -> "SpectralGaussian":
        return replace(self, mean=self.mean + dz, psd=self.psd + dpsd)

    def evidence(self, other: "SpectralGaussian") -> "SpectralGaussian":
        """J' = J + dJ, h' = h + dh.

        the mechanism by which sources at incommensurate sampling rates combine:
        an fMRI run contributes enormous precision below 0.25 Hz and literally
        none above it, and no special case is needed to say so.

        exact for the circular diagonal part; a non-zero relation or factor makes
        it approximate, and the exact form lives in ibm.runtime.fuse where the
        batching is known.
        """
        ja = 1.0 / np.maximum(self.total_psd(), _EPS)
        jb = 1.0 / np.maximum(other.total_psd(), _EPS)
        j = ja + jb
        return SpectralGaussian(self.basis, (ja * self.mean + jb * other.mean) / j, 1.0 / j)

    def inflate(self, factor: float = 1.0, floor: float = 0.0) -> "SpectralGaussian":
        return replace(self, psd=np.maximum(self.psd * factor, floor))

    # -- sampling ---------------------------------------------------------

    def sample(self, m: int, rng: np.random.Generator) -> np.ndarray:
        p, r = self.total_psd(), self._canon_relation()
        cxx, cyy, cxy = 0.5 * (p + r.real), 0.5 * (p - r.real), 0.5 * r.imag
        lxx = np.sqrt(np.maximum(cxx, 0.0))
        lyx = np.where(lxx > _EPS, cxy / np.maximum(lxx, _EPS), 0.0)
        lyy = np.sqrt(np.maximum(cyy - lyx ** 2, 0.0))
        a = rng.standard_normal((m,) + p.shape)
        b = rng.standard_normal((m,) + p.shape)
        z = self.mean + (lxx * a + 1j * (lyx * a + lyy * b))
        if self.factor is not None:
            sh = (m,) + self.shape + (self.rank,)
            eta = (rng.standard_normal(sh) + 1j * rng.standard_normal(sh)) / np.sqrt(2.0)
            z = z + np.einsum("...kr,m...r->m...k", self.factor, eta)
        return z

    def sample_time(self, m: int, rng: np.random.Generator) -> np.ndarray:
        return self.basis.synthesize(self.sample(m, rng))

    def mean_time(self) -> np.ndarray:
        return self.basis.synthesize(self.mean)

    def band_power(self, band: Band) -> np.ndarray:
        i = self.basis.band_indices(band)
        return (np.abs(self.mean[..., i]) ** 2 + self.total_psd()[..., i]).sum(-1)


FORM = register(UncertaintyForm(
    name="spectral",
    doc=SpectralGaussian.__doc__ or "",
    banded=True,
    zero=SpectralGaussian.zeros,
    pressure=SpectralGaussian.pressure,
    evidence=SpectralGaussian.evidence,
    cost=lambda band: 4,          # per retained component; the materializer multiplies by k
    sample=SpectralGaussian.sample,
    moment_match=SpectralGaussian.moment_match,
))


# -- conversions -------------------------------------------------------------


def _spectral_to_scalar(b: SpectralGaussian) -> ScalarGaussian:
    """collapse to the window mean and its variance.

    this discards exactly what the spectral form exists to carry.  it is the
    right conversion when the target genuinely has one timescale -- neurovascular
    coupling reading neural activity into blood state -- and the wrong one
    everywhere else, so it is recorded in provenance every time it is inserted.
    """
    return ScalarGaussian(b.mean[..., 0].real, b.total_psd()[..., 0])


def _scalar_to_spectral(b: ScalarGaussian, basis: TemporalBasis) -> SpectralGaussian:
    """lift a scalar belief into the DC component of a spectral one.

    lossless in the sense that nothing is discarded, but it asserts that the
    variable has no power above DC within the window -- which is a claim, not a
    neutral embedding, and is only true for variables whose form was chosen
    scalar for exactly that reason.
    """
    mean = np.zeros(b.mean.shape + (basis.k,), dtype=np.complex128)
    psd = np.zeros(b.mean.shape + (basis.k,))
    mean[..., 0] = b.mean
    psd[..., 0] = b.var
    return SpectralGaussian(basis, mean, psd)


conversion(Conversion("spectral", "scalar", _spectral_to_scalar, lossy=True,
    destroys="all temporal structure above DC: rhythms, phase, transients, and every "
             "cross-frequency relationship the spectral form exists to carry",
    doc=_spectral_to_scalar.__doc__ or ""))

conversion(Conversion("scalar", "spectral", _scalar_to_spectral, lossy=False,
    destroys="", doc=_scalar_to_spectral.__doc__ or ""))
