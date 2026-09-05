"""the uncertainty-form protocol and the conversion table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ibm.registry import REGISTRY
from ibm.vocabulary import Band


@dataclass(frozen=True)
class UncertaintyForm:
    """a registered way of carrying a belief about one component over a window.

    `cost` is what the materializer budgets: numbers stored per state variable.
    scalar costs two; spectral costs its retained component count.  the two
    budgets -- spatial sites and spectral components -- multiply, which is the
    whole reason bandwidth is a laziness axis.
    """
    name: str
    doc: str
    banded: bool
    zero: Callable[..., Any]
    pressure: Callable[..., Any]      # fold additive drift onto a belief
    evidence: Callable[..., Any]      # fold a gaussian constraint (precision addition)
    cost: Callable[[Band], int]
    sample: Callable[..., Any] | None = None
    moment_match: Callable[..., Any] | None = None


@dataclass(frozen=True)
class Conversion:
    """a declared map between forms.

    every conversion declares what it destroys.  spectral -> scalar discards
    precisely the information the spectral form exists to carry, so it is
    recorded in the materialized model's provenance and shows up in the audit as
    a place where multi-timescale structure leaves the model.  nothing is ever
    converted implicitly.
    """
    src: str
    dst: str
    fn: Callable[..., Any]
    lossy: bool
    destroys: str = ""
    doc: str = ""


def register(form: UncertaintyForm) -> UncertaintyForm:
    return REGISTRY.uncertainty_form(form)


def conversion(c: Conversion) -> Conversion:
    return REGISTRY.conversion(c)


def convert(belief: Any, src: str, dst: str) -> Any:
    if src == dst:
        return belief
    c = REGISTRY.conversions.get((src, dst))
    if c is None:
        raise KeyError(f"no registered conversion {src} -> {dst}")
    return c.fn(belief)
