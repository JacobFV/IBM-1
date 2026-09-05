"""binding source cards to the registry.

a card in `data/sources/` is not a description of a dataset.  it is a statement
of *which ibm state variables the dataset realizes, and how* -- `constrains` as
registered component ids, `via` as a registered process id, `band_hz` as the
temporal range it can constrain at all.  those three fields are the entire reason
the schema differs from a conventional dataset registry, and they are worth
nothing until something checks them.  this module is that check.

    unbound   -> the card records what the source's own documentation said, and
                 nothing has been resolved.  usable for planning, never for fitting.
    partial   -> some streams resolve and some do not.  the resolved ones may
                 contribute a likelihood; the rest are still prose.
    bound     -> every stream's `constrains` and `via` resolve, and every band is
                 inside the band the component is declared meaningful over.

the direction of the check matters.  a card cannot claim to constrain state that
does not exist, so an unregistered component id is an error against the *card*,
not a request to register something.  the registry is the ontology; the corpus
adapts to it.  the opposite arrangement -- registering a component because a
dataset mentioned one -- is exactly how an ontology acquires four names for the
same quantity, which is what `ibm.vocabulary`'s collision refusal exists to stop.

the band check is the one that catches real errors rather than typos.  a stream
sampled at 2 s TR cannot constrain anything above 0.25 Hz, and a component
declared meaningful only below 0.5 Hz cannot be constrained by a 1 kHz recording
above that -- not because the recording is bad but because the model has said
there is nothing up there for it to bear on.  ARCHITECTURE.md §1 makes bandwidth
a laziness axis co-equal with resolution precisely so that heterogeneous sources
stop appearing to conflict where they simply do not overlap, and this is where
that stops being a slogan.

reading is always safe; writing back is opt-in.  `bind(write=True)` rewrites only
the `binding:` line of a card, in place, because a card is a hand-written
document with comments and ordering that a yaml round-trip would destroy.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ibm.registry import REGISTRY
from ibm.vocabulary import Band, FULL

_BINDING_LINE = re.compile(r"^binding:\s*\S+\s*$", re.MULTILINE)


def repo_root(start: Path | None = None) -> Path:
    """the directory holding `data/` and `ibm/`.

    walked rather than configured, because the alternative is an environment
    variable that is wrong on exactly one machine and produces an empty corpus
    with no error.
    """
    p = (start or Path(__file__)).resolve()
    for q in (p, *p.parents):
        if (q / "data" / "sources").is_dir() and (q / "ibm").is_dir():
            return q
    raise FileNotFoundError(
        "cannot find the repository root: no ancestor of "
        f"{p} contains both data/sources and ibm/")


def sources_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "sources"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def load_card(path: Path) -> dict[str, Any]:
    import yaml
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_cards(root: Path | None = None) -> dict[str, tuple[Path, dict[str, Any]]]:
    """every card under `data/sources/`, keyed by its directory name.

    keyed by directory rather than by the card's `id` field so that a card whose
    id disagrees with its directory is *visible* as a mismatch instead of
    silently shadowing another card.
    """
    out: dict[str, tuple[Path, dict[str, Any]]] = {}
    for p in sorted(sources_dir(root).glob("*/card.yaml")):
        try:
            out[p.parent.name] = (p, load_card(p))
        except Exception as e:                    # a malformed card is a finding
            out[p.parent.name] = (p, {"_error": f"{type(e).__name__}: {e}"})
    return out


def band_of(stream: dict[str, Any]) -> Band:
    """the band a stream can constrain, from `band_hz` or, failing that, `rate_hz`.

    nyquist is the fallback and it is generous: a sampling rate bounds the band
    from above and says nothing about the anti-alias filter, the amplifier's
    high-pass, or whatever the provider already filtered out.  a card that
    declares `band_hz` is making a stronger and more useful claim, and the
    fallback is reported as a fallback so that the difference is visible.
    """
    b = stream.get("band_hz")
    if isinstance(b, (list, tuple)) and len(b) == 2:
        lo, hi = b
        if lo is not None or hi is not None:
            return Band(float(lo or 0.0), float(hi) if hi is not None else math.inf)
    rate = stream.get("rate_hz")
    if rate:
        return Band(0.0, float(rate) / 2.0)
    return FULL


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------


@dataclass
class StreamBinding:
    name: str
    kind: str = "measured"
    constrains: tuple[str, ...] = ()
    resolved: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    via: str | None = None
    via_ok: bool = False
    band: Band = FULL
    band_problems: tuple[str, ...] = ()

    @property
    def bound(self) -> bool:
        # a context stream is bound by deliberately constraining nothing: an expert
        # hypnogram is a human annotation of brain state, not a measurement of a
        # component, and a geometry file supplies a support rather than evidence.
        # forcing either to name a component would be a false claim.  the note is
        # what separates a decision from an omission, and it is checked above.
        if self.kind == "context":
            return not self.constrains and not self.band_problems
        return (bool(self.resolved) and not self.unknown
                and self.via is not None and self.via_ok and not self.band_problems)

    @property
    def missing(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.constrains:
            out.append("constrains: empty -- the card has not said which state this bears on")
        if self.unknown:
            out.append("constrains: unregistered " + ", ".join(self.unknown))
        if self.via is None:
            out.append("via: null -- §6 says an observation is evidence about state an "
                       "ordinary process produces; naming that process is what makes the "
                       "claim checkable")
        elif not self.via_ok:
            out.append(f"via: {self.via!r} is not a registered process")
        out += list(self.band_problems)
        return tuple(out)


@dataclass
class CardBinding:
    id: str
    path: Path
    title: str = ""
    access: str = ""
    was: str = "unbound"
    now: str = "unbound"
    streams: list[StreamBinding] = _field(default_factory=list)
    errors: list[str] = _field(default_factory=list)
    #: geometry the card says a stream needs before it can be attributed to a
    #: support at all.  a blocking requirement means the stream cannot be bound
    #: even when its ids resolve -- there is nowhere for the constraint to attach.
    blocking: tuple[str, ...] = ()

    @property
    def n_bound(self) -> int:
        return sum(1 for s in self.streams if s.bound)

    def summary(self) -> str:
        return (f"{self.id:28s} {self.access:10s} {self.was:8s} -> {self.now:8s}  "
                f"{self.n_bound}/{len(self.streams)} streams")

    def detail(self) -> str:
        lines = [self.summary()]
        for e in self.errors:
            lines.append(f"    ! {e}")
        for s in self.streams:
            if s.bound:
                lines.append(f"    ok   {s.name}: {', '.join(s.resolved)} via {s.via} "
                             f"over {s.band}")
                continue
            lines.append(f"    --   {s.name} [{s.kind}]")
            for m in s.missing:
                lines.append(f"           {m}")
        for b in self.blocking:
            lines.append(f"    !!   blocked: {b}")
        return "\n".join(lines)


@dataclass
class BindReport:
    cards: list[CardBinding] = _field(default_factory=list)
    written: list[str] = _field(default_factory=list)

    def by_status(self, status: str) -> list[CardBinding]:
        return [c for c in self.cards if c.now == status]

    def __str__(self) -> str:
        b, p, u = (len(self.by_status(s)) for s in ("bound", "partial", "unbound"))
        head = (f"{len(self.cards)} cards: {b} bound, {p} partial, {u} unbound"
                + (f"; {len(self.written)} rewritten" if self.written else ""))
        return "\n".join([head, ""] + [c.detail() for c in self.cards])

    def table(self) -> str:
        return "\n".join([c.summary() for c in self.cards])


# ---------------------------------------------------------------------------
# binding
# ---------------------------------------------------------------------------


def bind_card(card: dict[str, Any], path: Path, cid: str) -> CardBinding:
    """resolve one card against the registry.

    the registry must already be populated -- `ibm.load_all()` -- or every id
    fails to resolve and the report says the corpus is unbound when what is
    actually unbound is the import.  `bind()` handles that; this function
    assumes it.
    """
    out = CardBinding(id=cid, path=path,
                      title=str(card.get("title", "")),
                      access=str(card.get("access", "")),
                      was=str(card.get("binding", "unbound")))
    if "_error" in card:
        out.errors.append(str(card["_error"]))
        out.now = "unbound"
        return out
    if card.get("id") and card["id"] != cid:
        out.errors.append(f"card declares id {card['id']!r} but lives in {cid}/")

    blocking = []
    for r in card.get("requires") or []:
        if r.get("severity") == "blocking":
            blocking.append(f"{r.get('geometry', '?')}: "
                            f"{(r.get('without_it') or '').strip().split('.')[0]}")
    out.blocking = tuple(blocking)
    blocked_streams = {s for r in (card.get("requires") or [])
                       if r.get("severity") == "blocking"
                       for s in (r.get("for_streams") or [])}

    for st in card.get("streams") or []:
        name = str(st.get("name", "?"))
        constrains = tuple(str(x) for x in (st.get("constrains") or []))
        resolved, unknown = [], []
        band = band_of(st)
        problems: list[str] = []
        for c in constrains:
            comp = REGISTRY.components.get(c) or (
                REGISTRY.components.get(REGISTRY._alias.get(c, "")) if c in REGISTRY._alias
                else None)
            if comp is None:
                unknown.append(c)
                continue
            resolved.append(comp.id)
            if not band.intersects(comp.band):
                problems.append(
                    f"band: stream covers {band} but {comp.id} is declared meaningful only "
                    f"over {comp.band}; the two do not overlap, so this stream can contribute "
                    "no precision to it at all")
            elif band.hi_hz > comp.band.hi_hz:
                problems.append(
                    f"band: stream reaches {band.hi_hz:g} Hz but {comp.id} is meaningful only "
                    f"to {comp.band.hi_hz:g} Hz; the evidence above that is about something "
                    "the model does not represent and must be band-limited before fusion")
        # `via` is a chain, not one id: an observation is evidence about state that an
        # ordinary process chain produced (ARCHITECTURE.md §6).  scalp EEG is
        # em_generation -> em_coupling -> device_coupling, and collapsing that to a
        # single id would lose the step where the electrode interface enters.  a bare
        # string is read as a one-element chain.
        raw = st.get("via")
        chain = ([raw] if isinstance(raw, str) else list(raw or []))
        chain = [str(x) for x in chain]
        missing_via = [x for x in chain if x not in REGISTRY.processes]
        via = " -> ".join(chain) if chain else None
        via_ok = bool(chain) and not missing_via
        for x in missing_via:
            problems.append(f"via: {x!r} is not a registered process")
        if name in blocked_streams:
            problems.append("geometry: a blocking requirement is unmet, so there is no support "
                            "for this constraint to attach to even once its ids resolve")
        kind = str(st.get("kind", "measured"))
        if kind == "context":
            # a context stream is bound when it explicitly constrains nothing AND says why.
            # silence is not the same as a decision, so the note is required.
            if constrains or not str(st.get("notes", "")).strip():
                problems.append("context: a context stream must declare `constrains: []` and a "
                                "`notes` saying why no component is the right one; without the "
                                "note this is an undecided stream, not a decided one")
        out.streams.append(StreamBinding(
            name=name, kind=kind, constrains=constrains,
            resolved=tuple(resolved), unknown=tuple(unknown), via=via, via_ok=via_ok,
            band=band, band_problems=tuple(problems)))

    n = len(out.streams)
    ok = out.n_bound
    out.now = "bound" if (n and ok == n) else ("partial" if ok else "unbound")
    return out


def bind(root: Path | None = None, *, write: bool = False, only: Sequence[str] | None = None,
         load: bool = True) -> BindReport:
    """bind every card, and optionally flip the ones whose status changed.

    `load` imports the ontology first, because an empty registry makes every card
    look unbound for the wrong reason -- and a corpus report that is wrong in the
    pessimistic direction is exactly as useless as one that is wrong in the
    optimistic direction.
    """
    if load:
        try:
            import ibm
            ibm.load_all(seal=False)
        except Exception as e:                    # a half-written ontology is normal here
            pass
    rep = BindReport()
    for cid, (path, card) in load_cards(root).items():
        if only and cid not in only:
            continue
        cb = bind_card(card, path, cid)
        rep.cards.append(cb)
        if write and cb.now != cb.was:
            if _rewrite_binding(path, cb.now):
                rep.written.append(cid)
    return rep


def _rewrite_binding(path: Path, status: str) -> bool:
    """rewrite the `binding:` line and nothing else.

    a full yaml round-trip would reflow every block scalar, drop the schema
    comment at the top and reorder nothing predictably, turning a one-word status
    change into an unreviewable diff.  a card is a document people read.
    """
    text = path.read_text()
    new, n = _BINDING_LINE.subn(f"binding: {status}", text, count=1)
    if n == 0:
        return False
    path.write_text(new)
    return True


# ---------------------------------------------------------------------------
# what the corpus is missing
# ---------------------------------------------------------------------------


def unmet(rep: BindReport) -> dict[str, list[str]]:
    """the corpus's gaps, grouped by what would fix them.

    the useful shape for deciding what to do next: a hundred cards all missing
    `via` for an EEG stream is one declaration away from being fixed, and looking
    at it card by card hides that.
    """
    out: dict[str, list[str]] = {}
    for c in rep.cards:
        for s in c.streams:
            for m in s.missing:
                key = m.split(":", 1)[0]
                out.setdefault(key, []).append(f"{c.id}/{s.name}")
    return {k: sorted(v) for k, v in sorted(out.items())}


def components_wanted(rep: BindReport) -> dict[str, list[str]]:
    """unregistered component ids the corpus asked for, and who asked.

    not a to-do list for the registry.  it is a list of places where a card's
    author and the ontology disagree about what a quantity is called, and the fix
    is a synonym or a rewritten card far more often than a new component.
    """
    out: dict[str, list[str]] = {}
    for c in rep.cards:
        for s in c.streams:
            for u in s.unknown:
                out.setdefault(u, []).append(f"{c.id}/{s.name}")
    return {k: sorted(v) for k, v in sorted(out.items())}


__all__ = [
    "BindReport", "CardBinding", "StreamBinding", "band_of", "bind", "bind_card",
    "components_wanted", "load_card", "load_cards", "repo_root", "sources_dir", "unmet",
]
