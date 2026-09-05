"""scalar gaussian: one number and its variance, over a window.

the right form wherever a component has a single relevant timescale.  blood
volume, temperature, interstitial concentration, myelination and most structural
state gain nothing from a spectrum -- they have essentially no power above a few
tenths of a hertz, and carrying five hundred coefficients for them would be pure
budget spent on zeros.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ibm.fields.uncertainty.base import UncertaintyForm, register
from ibm.vocabulary import Band

_EPS = 1e-30


@dataclass(frozen=True)
class ScalarGaussian:
    """belief over a batch of scalar state variables.  arrays are (...,)."""

    mean: np.ndarray
    var: np.ndarray

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> "ScalarGaussian":
        return cls(np.zeros(shape), np.zeros(shape))

    @classmethod
    def prior(cls, shape: tuple[int, ...], mean: float = 0.0, sd: float = 1.0) -> "ScalarGaussian":
        return cls(np.full(shape, mean), np.full(shape, sd * sd))

    @property
    def precision(self) -> np.ndarray:
        return 1.0 / np.maximum(self.var, _EPS)

    def pressure(self, dmean: np.ndarray, dvar: np.ndarray | float = 0.0) -> "ScalarGaussian":
        """additive drift.  process contributions sum, per ARCHITECTURE.md §4."""
        return ScalarGaussian(self.mean + dmean, self.var + dvar)

    def evidence(self, mean: np.ndarray, var: np.ndarray) -> "ScalarGaussian":
        """independent gaussian constraint: J' = J + dJ, h' = h + dh."""
        ja, jb = self.precision, 1.0 / np.maximum(var, _EPS)
        j = ja + jb
        return ScalarGaussian((ja * self.mean + jb * mean) / j, 1.0 / j)

    def gain(self, g: np.ndarray | float) -> "ScalarGaussian":
        g = np.asarray(g)
        return ScalarGaussian(self.mean * g, self.var * g * g)

    def inflate(self, factor: float = 1.0, floor: float = 0.0) -> "ScalarGaussian":
        return replace(self, var=np.maximum(self.var * factor, floor))

    def sample(self, m: int, rng: np.random.Generator) -> np.ndarray:
        return self.mean + np.sqrt(np.maximum(self.var, 0.0)) * rng.standard_normal((m,) + self.mean.shape)

    @classmethod
    def moment_match(cls, ensemble: np.ndarray) -> "ScalarGaussian":
        return cls(ensemble.mean(0), ensemble.var(0, ddof=1))


FORM = register(UncertaintyForm(
    name="scalar",
    doc=ScalarGaussian.__doc__ or "",
    banded=False,
    zero=ScalarGaussian.zeros,
    pressure=ScalarGaussian.pressure,
    evidence=ScalarGaussian.evidence,
    cost=lambda band: 2,
    sample=ScalarGaussian.sample,
    moment_match=ScalarGaussian.moment_match,
))
