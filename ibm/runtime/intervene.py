"""clamps: externally constrained state, applied after contributions accumulate.

an intervention is not a primitive and not an input (ARCHITECTURE.md §6).  a TMS
coil is state on a device support, and constraining its current is an ordinary
statement about an ordinary state variable:

    x_coil(t) = u(t)

everything downstream -- induced field, membrane polarization, evoked response --
happens through ordinary processes that have no idea an experimenter is involved.
that is the whole reason there is no `observe/` or `stimulate/` directory: giving
interventions their own machinery would have made them a fifth primitive by the
back door.

so what is left here is small and mechanical, and it is exactly two things.

**when the clamp applies.**  after contribution accumulation, before the state is
read on.  pressure composes by summation and every process writing a clamped
variable has already had its say; the clamp then says that none of it moved the
variable, which is what "externally constrained" means.  applying it before
accumulation would let the processes overwrite it; applying it outside the solve
entirely would let the rest of the window equilibrate against a value about to be
replaced, which is why `ibm.runtime.step` calls this inside its iteration.

**which clamp wins.**  a session routinely carries several: a global baseline
holding a device off, a protocol imposing a waveform on it, a per-trial override.
they overlap in sites and in band, and silently letting the last one registered
win is how a control condition quietly becomes a stimulation condition.  so
precedence is explicit, ties are refused rather than resolved, and the resolution
is reported.

### hard and soft

a hard clamp asserts a value and destroys the belief's width: the variable is
what the experimenter set it to.  a soft clamp is evidence with finite precision
and goes through `ibm.runtime.fuse` unchanged -- which is the honest form for
every "intervention" whose delivered value is itself uncertain.  a TMS pulse
whose realized coil current has 5% run-to-run variation is a soft clamp, and
treating it as hard imports the experimenter's *intent* as though it were a
measurement of the coil.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field, replace
from typing import Any, Iterable, Sequence

import numpy as np

from ibm.fields.uncertainty.scalar import ScalarGaussian
from ibm.fields.uncertainty.spectral import SpectralGaussian, TemporalBasis
from ibm.registry import REGISTRY
from ibm.vocabulary import FULL, Band, Everywhere, Region, Sel

_EPS = 1e-30


class ClampConflict(Exception):
    """two clamps of equal precedence over the same state, and no way to choose.

    refused rather than resolved.  the two plausible tie-breaks -- declaration
    order and "the more specific one" -- are both wrong often enough that a model
    which picks one will eventually run the wrong condition and report it as the
    right one.
    """


@dataclass(frozen=True)
class Clamp:
    """one externally imposed constraint on state.

    `values` is in whatever the target block carries: a time-domain trajectory
    `(sites, n)` or `(n,)` for a spectral block, which is analyzed here, or plain
    numbers for a scalar one.  a waveform is the natural way to write a
    stimulation protocol and the natural way to write it *wrongly* is in
    coefficients, so the time domain is the default and the spectral path is
    opt-in via `spectral=True`.
    """

    sel: Sel
    values: Any
    #: hard destroys the belief's width; soft fuses at `precision`.
    hard: bool = True
    #: dJ for a soft clamp.  ignored when hard.
    precision: float | np.ndarray = 1.0
    #: residual variance a hard clamp leaves behind.  not zero by default: an
    #: imposed value is delivered by an instrument, and an instrument that
    #: reports its setting to nine digits is still not exact.  set it to 0.0 to
    #: assert an idealized clamp and accept a singular precision downstream.
    residual_var: float = 1e-12
    precedence: int = 0
    spectral: bool = False
    source: str = ""
    note: str = ""

    @property
    def components(self) -> tuple[str, ...]:
        return self.sel.vars

    def describe(self) -> str:
        return (f"{'hard' if self.hard else 'soft'} clamp on "
                f"{', '.join(self.components)} in {self.sel.band}, precedence "
                f"{self.precedence}" + (f" ({self.source})" if self.source else ""))


@dataclass
class ClampReport:
    applied: list[str] = _field(default_factory=list)
    masked: list[str] = _field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(self.applied) or "no clamps applied"


# ---------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------


def resolve(clamps: Sequence[Clamp]) -> tuple[Clamp, ...]:
    """order clamps so that the winner is applied last, and refuse real ties.

    applied last rather than filtered, because two clamps that overlap only
    partly are both legitimate: a baseline over the whole session and an override
    over one band should compose, with the override on top.  what is refused is
    two clamps of the same precedence and the same hardness whose selectors may
    overlap -- `Sel.may_overlap` is deliberately conservative, so this errs
    towards complaining, which is the right direction for a stimulation protocol.
    """
    ordered = sorted(clamps, key=lambda c: (c.precedence, 0 if c.hard else 1))
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if a.precedence != b.precedence or a.hard != b.hard:
                continue
            if any(x.may_overlap(y) for x in (a.sel,) for y in (b.sel,)):
                raise ClampConflict(
                    f"{a.describe()} and {b.describe()} have equal precedence and may cover "
                    "the same state.  give one of them a higher precedence -- guessing which "
                    "condition was intended is how a control run becomes a stimulation run")
    return tuple(ordered)


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------


def apply_clamps(state, clamps: Clamp | Iterable[Clamp], basis: TemporalBasis | None = None,
                 *, report: ClampReport | None = None) -> ClampReport:
    """impose every clamp on `state`, in place, in precedence order.

    in place and not on a copy, unlike `fuse`: this is called from inside the
    window solve once per picard sweep, and copying a whole materialized state
    tens of times per window would dominate the cost of the step.  callers who
    want isolation copy first, which is one line and visible.
    """
    rep = report or ClampReport()
    if isinstance(clamps, Clamp):
        clamps = (clamps,)
    items = tuple(clamps)
    if not items:
        return rep
    for c in resolve(items):
        for cid in c.components:
            if cid not in state:
                continue
            block = state.layout[cid]
            sites = _sites(state, cid, c.sel.region)
            if sites.size == 0:
                continue
            if block.uncertainty == "spectral":
                b = basis or block.basis
                if b is None:
                    continue
                _clamp_spectral(state, cid, c, sites, b)
            else:
                _clamp_scalar(state, cid, c, sites)
            rep.applied.append(f"{cid}[{sites.size} sites]: {c.describe()}")
            state.note(cid, c.describe() + (f" -- {c.note}" if c.note else ""))
    return rep


def _sites(state, cid: str, region: Region) -> np.ndarray:
    w = state.layout.weights(cid, region)
    return np.flatnonzero(w > 0.0)


def _clamp_spectral(state, cid: str, c: Clamp, sites: np.ndarray, basis: TemporalBasis) -> None:
    belief = state[cid]
    k = basis.band_indices(c.sel.band & state.layout[cid].band)
    if k.size == 0:
        return
    z = _as_coefficients(c, basis, sites.size, k.size)
    mean = belief.mean.copy()
    psd = belief.psd.copy()
    sl = np.ix_(sites, k)
    if c.hard:
        mean[sl] = z
        psd[sl] = c.residual_var
        # the relation is zeroed with the psd: a clamped variable has no residual
        # phase preference left to express, because it has no residual at all.
        rel = None if belief.relation is None else belief.relation.copy()
        if rel is not None:
            rel[sl] = 0.0
        state.beliefs[cid] = SpectralGaussian(belief.basis, mean, psd, rel, belief.factor)
        return
    j0 = 1.0 / np.maximum(belief.total_psd()[sl], _EPS)
    j1 = np.broadcast_to(np.asarray(c.precision, float), j0.shape)
    j = j0 + j1
    mean[sl] = (j0 * belief.mean[sl] + j1 * z) / j
    psd[sl] = 1.0 / j
    state.beliefs[cid] = SpectralGaussian(belief.basis, mean, psd, belief.relation, belief.factor)


def _clamp_scalar(state, cid: str, c: Clamp, sites: np.ndarray) -> None:
    belief = state[cid]
    # a scalar block has no window, so a clamp written as a waveform is imposed
    # by its window mean -- which is the registered spectral -> scalar reading of
    # it, and is exactly right for a variable whose form was declared scalar
    # because it has one relevant timescale.
    v = np.asarray(c.values, float)
    v = v.mean(-1) if v.ndim > 1 else v
    v = np.broadcast_to(np.atleast_1d(v), (sites.size,))
    mean, var = belief.mean.copy(), belief.var.copy()
    if c.hard:
        mean[sites] = v
        var[sites] = c.residual_var
    else:
        j0 = 1.0 / np.maximum(var[sites], _EPS)
        j1 = np.broadcast_to(np.asarray(c.precision, float), j0.shape)
        j = j0 + j1
        mean[sites] = (j0 * mean[sites] + j1 * v) / j
        var[sites] = 1.0 / j
    state.beliefs[cid] = ScalarGaussian(mean, var)


def _as_coefficients(c: Clamp, basis: TemporalBasis, n_sites: int, n_k: int) -> np.ndarray:
    """turn a clamp's values into coefficients on the selected band.

    a waveform is analyzed here rather than by the caller so that a protocol can
    be written as `u(t)` -- which is how §6 writes it, and how anybody who has
    ever specified a stimulation train thinks about it.
    """
    v = np.asarray(c.values)
    if c.spectral:
        z = v.astype(np.complex128)
    else:
        if v.ndim == 0 or v.shape[-1] == 1:
            # a constant is a legitimate waveform and the common one: a coil
            # carries no current, a display shows a grey field.  it is held over
            # the whole window rather than resampled, so there is nothing to
            # decide about its ends.
            v = np.broadcast_to(np.asarray(v, float).reshape(-1)[..., None],
                                (max(np.asarray(v).size, 1), basis.n))
        x = np.atleast_2d(np.asarray(v, float))
        if x.shape[0] == 1 and n_sites > 1:
            x = np.broadcast_to(x, (n_sites, x.shape[1]))
        if x.shape[-1] != basis.n:
            raise ValueError(
                f"clamp waveform has {x.shape[-1]} samples but the window is {basis.n}; a "
                "clamp is a statement about this window's trajectory and cannot be resampled "
                "here without deciding what happens at its ends")
        z = basis.analyze(x)[..., : basis.k]
    z = np.atleast_2d(z)
    if z.shape[-1] > n_k:
        z = z[..., :n_k]
    return np.broadcast_to(z, (n_sites, n_k))


# ---------------------------------------------------------------------------
# from the registry
# ---------------------------------------------------------------------------


def from_intervention(iv_id: str, values: Any, *, hard: bool = True,
                      precedence: int = 0, precision: float = 1.0,
                      region: Region | None = None) -> Clamp:
    """build a clamp from a registered intervention.

    the intervention already names the state it constrains, so the caller
    supplies only the waveform.  going through the registry is what makes an
    experimental protocol checkable: a protocol naming a variable no intervention
    was declared for is a declaration error, caught here, rather than a clamp
    silently landing on nothing.
    """
    iv = REGISTRY.interventions.get(iv_id)
    if iv is None:
        raise KeyError(f"no registered intervention {iv_id!r}")
    sel = iv.constrains if region is None else iv.constrains.restricted(region)
    return Clamp(sel=sel, values=values, hard=hard, precedence=precedence,
                 precision=precision, source=iv_id, note=iv.doc.strip().splitlines()[0]
                 if iv.doc else "")


def baseline(cid: str, value: float = 0.0, *, precedence: int = -100,
             band: Band = FULL) -> Clamp:
    """hold a variable at a resting value unless something outranks it.

    the common case for device state: a coil carries no current, a display shows
    a grey field, a stimulator is off.  low precedence by construction, so any
    protocol overrides it without a conflict.
    """
    return Clamp(Sel((cid,), Everywhere(), band), value, hard=True,
                 precedence=precedence, source="baseline")


__all__ = ["Clamp", "ClampConflict", "ClampReport", "apply_clamps", "baseline",
           "from_intervention", "resolve"]
