"""the global state vector over a materialized model.

the ontology declares components; a materialization instantiates state
variables, one component at one position, and this is where those live.  the
vector is deliberately *blocked* rather than flat: a block is one component over
its materialized sites, carried in that component's declared form of uncertain
state.  a single flat array would force one form on everything, and the whole
point of ARCHITECTURE.md §1 is that blood and neural population state are not
carried the same way.

three consequences of that choice show up throughout this module.

*allocation is per block, not global.*  the cost of a block is
`sites x form.cost(band)`, and the two budgets multiply -- spatial sites and
retained spectral components -- which is why bandwidth is a laziness axis
co-equal with resolution and why a whole-brain haemodynamic block is cheap while
a single-electrode spike block is not.

*get/set go through selectors, never through indices.*  a process names a `Sel`
-- variables, region, band -- and never learns where in memory anything sits.
that is what lets the materializer re-lay-out a model, coarsen a region or drop
a band without a single process changing.  a region is symbolic until
materialization, so the site weights come from the model, not from here.

*flattening exists only for solvers.*  `pack_means` produces the one real vector
`scipy.sparse.linalg` needs and nothing else uses it.  keeping that confined to
two functions is what makes a jax backend a small change: the packing is the
only place that assumes numpy's memory model, and `jax.flatten_util.ravel_pytree`
replaces it wholesale.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ibm.fields.uncertainty.scalar import ScalarGaussian
from ibm.fields.uncertainty.spectral import SpectralGaussian, TemporalBasis
from ibm.registry import REGISTRY
from ibm.vocabulary import FULL, Band, Everywhere, Region, Sel


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """read the first attribute that exists.

    `ibm.materialize` owns the layout and is written independently of this
    module.  rather than couple to one spelling of its field names, the runtime
    accepts any of the plausible ones and fails loudly only when none is present.
    this is an adaptor, not a guess: the moment materialize exposes a `Layout`
    directly, `Layout.of` returns it untouched and none of this runs.
    """
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


@dataclass(frozen=True)
class Block:
    """one component over its materialized sites *on one support*.

    `basis` is present only for banded forms.  a scalar block has no window
    structure to speak of -- its belief is one number and one variance over the
    whole window -- and giving it a basis anyway would invite code that quietly
    assumes every block has a spectrum.

    the identity of a block is `(component, support)` and not the component
    alone.  a support is a *sampling* of a domain, not a different quantity
    (`Component.alt_supports`), and a component admissible on two of them can be
    instantiated on both at once so long as they cover disjoint positions:
    cortical population state indexed by column nodes on the folded sheet,
    subcortical population state indexed by parenchyma voxels, one component and
    two blocks.  the alternative -- one block per component -- forced a
    materialization to choose, and choosing the sheet is what left the thalamus,
    the brainstem and the cerebellum with no state at all.

    what is *not* permitted is two blocks over the same positions.  that is the
    same quantity entered twice, and it is checked geometrically in
    `ibm.materialize.build._overlap_check` rather than assumed here, because
    whether two samplings actually cover the same millimetres is a fact about one
    subject's anatomy and not about the declaration.
    """

    component: str
    uncertainty: str
    n_sites: int
    band: Band = FULL
    basis: TemporalBasis | None = None
    sites: np.ndarray | None = None          # ids into the model's site table
    support: str = ""
    spacing_mm: float = float("nan")

    @property
    def key(self) -> tuple[str, str]:
        """what a block is identified by: one component on one support."""
        return (self.component, self.support)

    @property
    def label(self) -> str:
        return f"{self.component}@{self.support}" if self.support else self.component

    @property
    def k(self) -> int:
        return 0 if self.basis is None else self.basis.k

    @property
    def cost(self) -> int:
        """numbers stored.  what the materializer budgets against."""
        form = REGISTRY.uncertainty.get(self.uncertainty)
        per = form.cost(self.band) if form is not None else 2
        return self.n_sites * per * max(self.k, 1)

    def zero(self) -> Any:
        if self.uncertainty == "spectral":
            if self.basis is None:
                raise ValueError(f"spectral block {self.component!r} has no temporal basis")
            return SpectralGaussian.zeros(self.basis, (self.n_sites,))
        return ScalarGaussian.zeros((self.n_sites,))


@dataclass(frozen=True)
class Layout:
    """the allocation plan for one materialized model.

    ordered, because provenance and cost reports read better when the order is
    stable, and because the flat packing used by the solvers has to be
    deterministic.  region resolution is delegated: regions stay symbolic until
    materialization (see `ibm.vocabulary.Region`), so the only thing that can
    turn `Anat("cortical_layers", "iv")` into weights over sites is the model
    that built the site table.

    blocks are keyed by `(component, support)` rather than by component.  the
    two are the same thing for almost every materialization, and differ exactly
    where one has to: a component admissible on several supports may be
    instantiated on more than one of them at once, cortex on the sheet and
    subcortex in the volume, and the allocation plan has to be able to say so.
    what the plan may *not* contain is the same component on the same support
    twice -- that is one block written by two processes, which is a bookkeeping
    error rather than a modelling decision, and it is still refused here.

    blocks of one component are contiguous and in a stable order, so the
    component's sites are the concatenation of its blocks' sites in that order.
    that is what lets the rest of the runtime keep holding one belief per
    component: the split is a fact about where the positions came from, and a
    process reading `neural.exc.activity` is reading one physical quantity
    whether its positions were sampled off a sheet or out of a volume.
    """

    blocks: tuple[Block, ...]
    basis: TemporalBasis | None = None
    region_weights: Any = None               # (component, Region) -> weights in [0,1]

    def __post_init__(self) -> None:
        seen: set[tuple[str, str]] = set()
        order: list[str] = []
        for b in self.blocks:
            if b.key in seen:
                raise ValueError(
                    f"component {b.component!r} allocated twice on support {b.support!r}; a "
                    "materialized model holds one block per (component, support), not one per "
                    "process.  two samplings of one domain are a legitimate split and get two "
                    "blocks; the same sampling twice is a double count")
            if b.component in order and order[-1] != b.component:
                raise ValueError(
                    f"the blocks of {b.component!r} are not contiguous in this layout.  a "
                    "component's sites are the concatenation of its blocks' sites in layout "
                    "order, so interleaving them would make that concatenation mean nothing")
            if b.component not in order or order[-1] != b.component:
                order.append(b.component)
            seen.add(b.key)

    # -- construction ----------------------------------------------------

    @classmethod
    def of(cls, model: Any) -> "Layout":
        """derive a layout from a materialized model, tolerating its spelling."""
        lay = _attr(model, "layout")
        if isinstance(lay, Layout):
            return lay
        raw = _attr(lay, "blocks", "entries", default=None) or _attr(
            model, "blocks", "state_blocks", "variables", default=None)
        if raw is None:
            raise TypeError(
                f"{type(model).__name__} exposes no state layout: expected `.layout`, "
                "`.blocks` or `.state_blocks`.  ibm.runtime cannot allocate state for a "
                "model that has not said what it materialized.")
        basis = _attr(model, "basis", "temporal_basis") or _attr(lay, "basis")
        blocks = tuple(b if isinstance(b, Block) else cls._adapt(b, basis) for b in raw)
        return cls(blocks, basis, _attr(model, "region_weights", "weights_for"))

    @staticmethod
    def _adapt(b: Any, default_basis: TemporalBasis | None) -> Block:
        cid = _attr(b, "component", "id", "cid")
        if cid is None:
            raise TypeError(f"layout entry {b!r} names no component")
        comp = REGISTRY.components.get(cid)
        unc = _attr(b, "uncertainty", "form") or (comp.uncertainty if comp else "scalar")
        sites = _attr(b, "sites", "site_ids", "positions")
        n = _attr(b, "n_sites", "n", "size")
        if n is None:
            n = len(sites) if sites is not None else 1
        band = _attr(b, "band") or (comp.band if comp else FULL)
        basis = _attr(b, "basis") or (default_basis if unc == "spectral" else None)
        if basis is not None and unc == "spectral":
            basis = basis.truncated(band)
        return Block(cid, unc, int(n), band, basis,
                     None if sites is None else np.asarray(sites),
                     _attr(b, "support", default="") or "",
                     float(_attr(b, "spacing_mm", default=float("nan"))))

    # -- lookup ----------------------------------------------------------

    def __iter__(self) -> Iterator[Block]:
        return iter(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def __contains__(self, cid: str) -> bool:
        return any(b.component == cid for b in self.blocks)

    def __getitem__(self, key: str | tuple[str, str]) -> Block:
        """one block by `(component, support)`, or a component's blocks merged.

        indexing by a bare component id is the common path and returns exactly
        the block when there is one.  where a component was split across two
        supports it returns the *merged* block -- same form, same band, same
        basis, sites concatenated in layout order -- because that is what the
        rest of the runtime is holding a belief about: one physical quantity over
        every position at which it was instantiated.  the merged block reports no
        support and no spacing, deliberately, so that nothing downstream can read
        a single support or a single r(q) off a component that has two.

        index by the pair to reach one half of a split.
        """
        if isinstance(key, tuple):
            cid, support = key
            for b in self.blocks:
                if b.key == (cid, support):
                    return b
            raise KeyError(
                f"component {cid!r} is not materialized on support {support!r}; this model has "
                f"it on {', '.join(self.supports_of(cid)) or 'no support at all'}")
        parts = self.blocks_of(key)
        if not parts:
            raise KeyError(f"component {key!r} is not materialized in this model; either the "
                           "request did not reach it or a process names state the trace missed")
        if len(parts) == 1:
            return parts[0]
        return self._merged(parts)

    @staticmethod
    def _merged(parts: Sequence[Block]) -> Block:
        first = parts[0]
        sites = None
        if all(p.sites is not None for p in parts):
            sites = np.concatenate([np.asarray(p.sites) for p in parts])
        return Block(first.component, first.uncertainty, sum(p.n_sites for p in parts),
                     first.band, first.basis, sites, "", float("nan"))

    def blocks_of(self, cid: str) -> tuple[Block, ...]:
        """every block of one component, in layout order.  usually one."""
        return tuple(b for b in self.blocks if b.component == cid)

    def supports_of(self, cid: str) -> tuple[str, ...]:
        return tuple(b.support for b in self.blocks_of(cid))

    def is_split(self, cid: str) -> bool:
        """was this component instantiated on more than one support at once?"""
        return len(self.blocks_of(cid)) > 1

    @property
    def components(self) -> tuple[str, ...]:
        """the distinct components, in layout order.  shorter than `blocks`."""
        out: list[str] = []
        for b in self.blocks:
            if not out or out[-1] != b.component:
                out.append(b.component)
        return tuple(out)

    @property
    def split(self) -> tuple[str, ...]:
        return tuple(c for c in self.components if self.is_split(c))

    def weights(self, component: str, region: Region) -> np.ndarray:
        """soft membership of each site of a component in a region.

        a partition boundary is a gradient, not a wall (ARCHITECTURE.md §2), so
        this returns weights in [0,1] and callers multiply by them rather than
        thresholding.  with no resolver the whole component is selected, which is
        the honest reading of `Everywhere` and the only safe default for a model
        that has not built an atlas.

        for a split component the answer spans both supports, concatenated in
        layout order.  that is the one shape a process can use: a region is a set
        of positions and does not know which sampling produced them, so
        `Anat("thalamic_nuclei", "vpl")` selects the parenchyma half and none of
        the sheet half without either side having to be named.
        """
        b = self[component]
        if isinstance(region, Everywhere) or self.region_weights is None:
            return np.ones(b.n_sites)
        w = np.asarray(self.region_weights(component, region), float)
        if w.shape != (b.n_sites,):
            raise ValueError(f"region weights for {component!r} have shape {w.shape}, "
                             f"expected {(b.n_sites,)}")
        return np.clip(w, 0.0, 1.0)

    def cost(self) -> dict[tuple[str, str], int]:
        """numbers stored per block.  keyed by (component, support), like the plan."""
        return {b.key: b.cost for b in self.blocks}

    def cost_by_component(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for b in self.blocks:
            out[b.component] = out.get(b.component, 0) + b.cost
        return out

    def describe(self) -> str:
        rows = [(b.component, b.support or "-", b.uncertainty, str(b.n_sites), str(b.k),
                 f"{b.band.lo_hz:g}-{b.band.hi_hz:g}", f"{b.cost:,}") for b in self.blocks]
        head = ("component", "support", "form", "sites", "k", "band Hz", "numbers")
        w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
             for i, h in enumerate(head)]
        out = ["  ".join(h.ljust(x) for h, x in zip(head, w)),
               "  ".join("-" * x for x in w)]
        out += ["  ".join(c.ljust(x) for c, x in zip(r, w)) for r in rows]
        n_split = len(self.split)
        out.append(f"total {sum(b.cost for b in self.blocks):,} numbers over "
                   f"{len(self.blocks)} blocks / {len(self.components)} components"
                   + (f", {n_split} of them split across supports" if n_split else ""))
        return "\n".join(out)


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class View:
    """what a process actually sees: one component, some sites, one band.

    the belief carried here is already restricted -- band-marginalized and
    gathered onto the selected sites -- because marginalizing a gaussian to a
    band is exact and doing it at the boundary is what makes band-limited
    coupling free.  `weights` is kept beside the belief rather than folded into
    it: a soft membership scales a process's *gain*, and folding it into the
    belief would corrupt the state itself.
    """

    component: str
    sites: np.ndarray
    weights: np.ndarray
    band: Band
    belief: Any

    @property
    def n(self) -> int:
        return int(self.sites.size)


# ---------------------------------------------------------------------------
# the state vector
# ---------------------------------------------------------------------------


class State:
    """the materialized state of one model over one window.

    a state variable is not a value at an instant; it is a belief about a
    trajectory over a window (ARCHITECTURE.md §1).  so `State` holds beliefs, and
    there is no method anywhere that returns "the value" of anything -- the
    closest is `mean_time`, which is explicitly a mean and drops the uncertainty
    the rest of the system exists to carry.

    mutation is in place and deliberate.  a step over a window touches every
    block, contributions from many processes accumulate onto the same state
    (`dot x = sum_p f_p`), and copying the whole vector per contribution would
    dominate the cost of a large materialization.  `copy()` is available where a
    caller needs isolation -- the ensemble machinery uses it.
    """

    __slots__ = ("layout", "beliefs", "provenance")

    def __init__(self, layout: Layout, beliefs: dict[str, Any] | None = None) -> None:
        self.layout = layout
        self.beliefs: dict[str, Any] = dict(beliefs or {})
        #: per-component notes about how the belief got where it is: which
        #: conversions were inserted, where a nonlinearity was moment-matched,
        #: which values are distilled rather than measured.  a prediction resting
        #: on prior-dominated structure must not be presented with the confidence
        #: of one resting on constrained structure, and this is the record of
        #: which is which.
        self.provenance: dict[str, list[str]] = {}

    # -- allocation ------------------------------------------------------

    @classmethod
    def zeros(cls, layout: Layout) -> "State":
        return cls(layout, {c: layout[c].zero() for c in layout.components})

    @classmethod
    def prior(cls, layout: Layout, sd: float = 1.0, beta: float = 1.0,
              declared: bool = True) -> "State":
        """allocate at the ontology's prior rather than at zero.

        "the ontology's prior" means the one the component actually declares.
        every component carries a `prior` naming a registered builder in
        `ibm.fields.priors` -- `neural_population` is 1/f with theta, alpha and
        beta bumps on it, `device_broadband` is a johnson floor plus mains, and
        the numbers in each are cited -- and allocating a generic `1/f^beta`
        instead threw all of that away at the last possible moment.  a resting
        cortical block and a resting electrode block are not the same belief, and
        a run initialised as though they were cannot show an alpha peak it was
        never given.

        `declared=False` falls back to the flat `sd`/`beta` shape, which is what
        a caller wants when it is testing the solver rather than the ontology.
        """
        from ibm.fields import priors as _priors

        out: dict[str, Any] = {}
        for c in layout.components:
            b = layout[c]
            comp = REGISTRY.components.get(c)
            builder = (_priors.PRIORS.get(comp.prior)
                       if declared and comp is not None and comp.prior else None)
            if builder is not None and builder.form == b.uncertainty:
                out[c] = (builder(b.basis, shape=(b.n_sites,))
                          if b.uncertainty == "spectral" and b.basis is not None
                          else builder(shape=(b.n_sites,)))
                continue
            if b.uncertainty == "spectral" and b.basis is not None:
                f = np.maximum(b.basis.freqs_hz, b.basis.freqs_hz[1] if b.basis.k > 1 else 1.0)
                psd = (sd * sd) * f ** (-beta)
                out[c] = SpectralGaussian.from_psd(b.basis, psd, shape=(b.n_sites,))
            else:
                out[c] = ScalarGaussian.prior((b.n_sites,), 0.0, sd)
        return cls(layout, out)

    def copy(self) -> "State":
        s = State(self.layout, dict(self.beliefs))
        s.provenance = {k: list(v) for k, v in self.provenance.items()}
        return s

    def note(self, component: str, message: str) -> None:
        self.provenance.setdefault(component, []).append(message)

    # -- direct access ---------------------------------------------------

    def __contains__(self, cid: str) -> bool:
        return cid in self.beliefs

    def __getitem__(self, cid: str) -> Any:
        try:
            return self.beliefs[cid]
        except KeyError:
            raise KeyError(f"{cid!r} is not materialized in this state") from None

    def __setitem__(self, cid: str, belief: Any) -> None:
        b = self.layout[cid]
        self._check(b, belief)
        self.beliefs[cid] = belief

    def items(self) -> Iterable[tuple[str, Any]]:
        return self.beliefs.items()

    @staticmethod
    def _check(block: Block, belief: Any) -> None:
        want = SpectralGaussian if block.uncertainty == "spectral" else ScalarGaussian
        if not isinstance(belief, want):
            raise TypeError(
                f"{block.component!r} is declared {block.uncertainty!r} but was assigned a "
                f"{type(belief).__name__}.  forms are never converted implicitly -- every "
                "conversion declares what it destroys; use ibm.fields.uncertainty.convert.")
        n = belief.mean.shape[0] if belief.mean.ndim else 1
        if n != block.n_sites:
            raise ValueError(f"{block.component!r} has {block.n_sites} sites but the belief "
                             f"carries {n}")

    # -- selector access -------------------------------------------------

    def get(self, s: Sel) -> tuple[View, ...]:
        """read a slice of state.

        the only read path a process has.  everything a process is allowed to
        know about where its inputs live is in the `Sel` it declared, which is
        also the thing the registry checked at seal time -- so a typo here is a
        declaration error caught before anything runs, not a silently empty
        coupling.
        """
        out: list[View] = []
        for cid in s.vars:
            if cid not in self.beliefs:
                continue
            b = self.layout[cid]
            w = self.layout.weights(cid, s.region)
            idx = np.flatnonzero(w > 0.0)
            band = s.band & b.band
            belief = self.beliefs[cid]
            if b.uncertainty == "spectral":
                belief = belief.marginal(band)
                gathered = _gather_spectral(belief, idx)
            else:
                gathered = ScalarGaussian(belief.mean[idx], belief.var[idx])
            out.append(View(cid, idx, w[idx], band, gathered))
        return tuple(out)

    def one(self, cid: str, region: Region | None = None, band: Band = FULL) -> View:
        v = self.get(Sel((cid,), region or Everywhere(), band))
        if not v:
            raise KeyError(f"{cid!r} is not materialized in this state")
        return v[0]

    def set(self, s: Sel, beliefs: Sequence[Any] | Any) -> None:
        """write a slice of state back.

        scatter, not assign: the selector named a band and a region, so only
        those coefficients at those sites move.  writing a band-restricted belief
        over the whole block would silently zero every component the process was
        never entitled to touch.
        """
        vals = list(beliefs) if isinstance(beliefs, (list, tuple)) else [beliefs] * len(s.vars)
        for cid, val in zip(s.vars, vals):
            if cid not in self.beliefs:
                continue
            b = self.layout[cid]
            w = self.layout.weights(cid, s.region)
            idx = np.flatnonzero(w > 0.0)
            if b.uncertainty == "spectral":
                self.beliefs[cid] = _scatter_spectral(
                    self.beliefs[cid], val, idx, b.basis.band_indices(s.band & b.band))
            else:
                cur = self.beliefs[cid]
                mean, var = cur.mean.copy(), cur.var.copy()
                mean[idx], var[idx] = val.mean, val.var
                self.beliefs[cid] = ScalarGaussian(mean, var)

    def add(self, s: Sel, dmean: np.ndarray, dvar: np.ndarray | float = 0.0) -> None:
        """accumulate additive pressure.  `dot x += f`, per ARCHITECTURE.md §4.

        pressure composes by summation and evidence composes by precision
        addition; those are different rules and this is the one for pressure.
        keeping them in separate methods is the whole reason `ibm.runtime.fuse`
        is a separate module.
        """
        for cid in s.vars:
            if cid not in self.beliefs:
                continue
            b = self.layout[cid]
            w = self.layout.weights(cid, s.region)
            cur = self.beliefs[cid]
            if b.uncertainty == "spectral":
                k = b.basis.band_indices(s.band & b.band)
                dz = np.zeros_like(cur.mean)
                dz[:, k] = np.asarray(dmean)
                self.beliefs[cid] = cur.pressure(dz * w[:, None], dvar)
            else:
                self.beliefs[cid] = cur.pressure(np.asarray(dmean) * w, dvar)

    # -- flat packing, for solvers only ----------------------------------

    def pack_means(self, components: Sequence[str] | None = None) -> np.ndarray:
        """the one real vector `scipy.sparse.linalg` will accept.

        complex spectral coefficients are split into real and imaginary halves
        rather than passed through as complex: gmres over a complex field is
        available but the matrix-free jvp below is a real directional derivative,
        and mixing the two is the classic way to get a silently wrong krylov
        space.  a jax backend deletes this function -- `ravel_pytree` does it
        without the runtime knowing the shapes.
        """
        parts: list[np.ndarray] = []
        for b in self._selected(components):
            v = self.beliefs[b.component].mean
            parts.append(np.concatenate([v.real.ravel(), v.imag.ravel()])
                         if np.iscomplexobj(v) else np.asarray(v, float).ravel())
        return np.concatenate(parts) if parts else np.zeros(0)

    def unpack_means(self, vec: np.ndarray, components: Sequence[str] | None = None) -> "State":
        out = self.copy()
        i = 0
        for b in self._selected(components):
            cur = out.beliefs[b.component]
            shape = cur.mean.shape
            n = int(np.prod(shape))
            if np.iscomplexobj(cur.mean):
                re = vec[i:i + n].reshape(shape); i += n
                im = vec[i:i + n].reshape(shape); i += n
                out.beliefs[b.component] = replace(cur, mean=re + 1j * im)
            else:
                out.beliefs[b.component] = replace(cur, mean=vec[i:i + n].reshape(shape))
                i += n
        return out

    def _selected(self, components: Sequence[str] | None) -> list[Block]:
        """one entry per component -- merged, where a component was split.

        the packing is over beliefs and there is one belief per component, so
        iterating `self.layout` here would visit a split component once per
        support and pack it twice.
        """
        if components is None:
            return [self.layout[c] for c in self.layout.components if c in self.beliefs]
        return [self.layout[c] for c in components if c in self.beliefs]

    # -- reporting -------------------------------------------------------

    def mean_time(self, cid: str) -> np.ndarray:
        """the window trajectory of the mean.  drops everything else, visibly."""
        b = self.layout[cid]
        belief = self.beliefs[cid]
        if b.uncertainty == "spectral":
            return belief.mean_time()
        n = b.basis.n if b.basis is not None else 1
        return np.repeat(belief.mean[:, None], n, axis=1)

    def total_power(self, cid: str, band: Band = FULL) -> np.ndarray:
        b = self.layout[cid]
        belief = self.beliefs[cid]
        if b.uncertainty == "spectral":
            return belief.band_power(band)
        return belief.mean ** 2 + belief.var

    def describe(self) -> str:
        rows = []
        for cid in self.layout.components:
            if cid not in self.beliefs:
                continue
            b = self.layout[cid]
            belief = self.beliefs[cid]
            if b.uncertainty == "spectral":
                sd = float(np.sqrt(belief.total_psd().sum(-1).mean()))
                extra = f"phase {float(belief.phase_concentration().mean()):.2f}"
            else:
                sd = float(np.sqrt(np.mean(belief.var)))
                extra = "-"
            rows.append((cid, "+".join(self.layout.supports_of(cid)) or "-", b.uncertainty,
                         str(b.n_sites),
                         f"{float(np.mean(np.abs(belief.mean))):.3g}", f"{sd:.3g}", extra,
                         ";".join(self.provenance.get(cid, []))[:48] or "-"))
        head = ("component", "support", "form", "sites", "|mean|", "sd", "note", "provenance")
        w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
             for i, h in enumerate(head)]
        out = ["  ".join(h.ljust(x) for h, x in zip(head, w)),
               "  ".join("-" * x for x in w)]
        out += ["  ".join(c.ljust(x) for c, x in zip(r, w)) for r in rows]
        return "\n".join(out)


#: the architecture calls this the global state vector; the class is `State`
#: because it is not flat and pretending otherwise invites indexing into it.
StateVector = State


# ---------------------------------------------------------------------------
# gather / scatter for the spectral form
# ---------------------------------------------------------------------------


def _gather_spectral(b: SpectralGaussian, idx: np.ndarray) -> SpectralGaussian:
    return SpectralGaussian(
        b.basis, b.mean[idx], b.psd[idx],
        None if b.relation is None else b.relation[idx],
        None if b.factor is None else b.factor[idx])


def _scatter_spectral(full: SpectralGaussian, part: SpectralGaussian,
                      sites: np.ndarray, k: np.ndarray) -> SpectralGaussian:
    """write a band- and site-restricted belief back into the full block.

    the relation term is scattered with the rest.  it is not decoration: it is
    exactly the phase preference an evoked response consists of, and dropping it
    on write-back would turn every evoked response the model produces back into
    an ongoing rhythm.
    """
    mean, psd = full.mean.copy(), full.psd.copy()
    mean[np.ix_(sites, k)] = part.mean
    psd[np.ix_(sites, k)] = part.psd
    rel = full.relation
    if part.relation is not None:
        rel = np.zeros_like(full.mean) if rel is None else rel.copy()
        rel[np.ix_(sites, k)] = part.relation
    return SpectralGaussian(full.basis, mean, psd, rel, full.factor)


__all__ = ["Block", "Layout", "View", "State", "StateVector"]
