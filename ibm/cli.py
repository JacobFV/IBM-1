"""the `ibm` console entry point.

    ibm ontology              print the registry as tables, and the summary
    ibm check                 run REGISTRY.check() and print every problem
    ibm sources               list data/sources cards with access and binding
    ibm bind                  resolve cards against the registry and report
    ibm materialize <model>   build a named model and print its describe()
    ibm plan [<model>]        cost and budget for a materialization

the design brief for this file is one line from ARCHITECTURE.md §8: *the whole
ontology must be printable as one table; if it is not, it has already begun to
sprawl.*  `ibm ontology` is that table, and it exists so that the sprawl is
visible early and cheaply rather than discovered in year two when several pairs
of components turn out to be the same thing under different names.

everything here is a thin shell over machinery that lives elsewhere.  the CLI
formats and exits; it does not decide anything.  the one place it has an opinion
is the exit code: `check` and `bind` return non-zero when they found errors, so
they are usable in a pre-commit hook without anybody having to grep the output.

the ontology is imported *unsealed* by default.  the package is written by many
hands and is routinely half-declared; sealing on every invocation would make the
inspection tools unusable at exactly the moment they are most useful.  `check`
prints what sealing would have refused, which is the same information without the
exception.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _load(strict: bool = False, seal: bool = False) -> tuple[Any, list[str]]:
    """import the ontology, tolerating modules that are not written yet.

    a partially declared package is the normal state of this repository, and a
    tool that refuses to print anything until every module imports is a tool
    nobody runs.  so import failures are collected and reported beside the
    tables, and the tables are printed anyway.
    """
    import importlib
    from ibm.registry import REGISTRY

    problems: list[str] = []
    for m in ("ibm.frames", "ibm.fields", "ibm.anatomy", "ibm.topologies", "ibm.processes"):
        try:
            importlib.import_module(m)
        except Exception as e:
            problems.append(f"{m}: {type(e).__name__}: {e}")
    if seal:
        try:
            REGISTRY.seal(strict=strict)
        except Exception as e:
            problems.append(f"seal: {e}")
    return REGISTRY, problems


def _warn(problems: Sequence[str]) -> None:
    for p in problems:
        print(f"  ! {p}", file=sys.stderr)


# ---------------------------------------------------------------------------
# ontology
# ---------------------------------------------------------------------------


def cmd_ontology(args: argparse.Namespace) -> int:
    reg, problems = _load()
    if problems:
        print("modules that did not import (their declarations are missing below):",
              file=sys.stderr)
        _warn(problems)
        print(file=sys.stderr)

    kinds = [args.table] if args.table else ["components", "processes", "topologies",
                                             "observations"]
    for kind in kinds:
        print(f"== {kind} ==")
        try:
            print(reg.table(kind))
        except Exception as e:
            print(f"  (no {kind}: {e})")
        print()
    print("== summary ==")
    print(reg.summary())
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """print everything sealing would refuse, grouped by severity.

    errors first and warnings after, because the two mean different things: an
    error is a graph that cannot run -- a dangling selector, a component read but
    never written, a coupling between two forms with no declared conversion --
    while a warning is usually a process that exists in the ontology without a
    high-confidence f, which §5 explicitly permits.
    """
    reg, problems = _load()
    if problems:
        print("modules that did not import:", file=sys.stderr)
        _warn(problems)
        print(file=sys.stderr)

    probs = reg.check()
    errs = [p for p in probs if p.severity == "error"]
    warns = [p for p in probs if p.severity != "error"]
    for p in errs:
        print(p)
    if warns and not args.errors_only:
        if errs:
            print()
        for p in warns:
            print(p)
    print()
    print(f"{len(errs)} errors, {len(warns)} warnings over {len(reg.components)} components "
          f"and {len(reg.processes)} processes")
    if not errs and not problems:
        print("the ontology seals")
    return 1 if (errs or problems) else 0


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def cmd_sources(args: argparse.Namespace) -> int:
    """list the corpus as it stands: access, binding, streams, permitted uses.

    `access` and `binding` are independent and both matter.  a card may be
    `access: held` -- the bytes are on disk -- and still `binding: unbound`,
    which means it can be planned against and not fitted against.  the opposite
    also happens: a fully bound card whose licence is unresolved is a card that
    must not be used yet for reasons that have nothing to do with the ontology.
    """
    from ibm.forge.bind import load_cards

    cards = load_cards()
    rows = []
    for cid, (path, card) in cards.items():
        if "_error" in card:
            rows.append((cid, "?", "?", "!", "!", str(card["_error"])[:40]))
            continue
        streams = card.get("streams") or []
        use = ",".join(card.get("use") or []) or "-"
        lic = (card.get("licence") or {}).get("redistribution", "-")
        if args.access and card.get("access") != args.access:
            continue
        if args.binding and card.get("binding") != args.binding:
            continue
        rows.append((cid, str(card.get("access", "-")), str(card.get("binding", "-")),
                     str(len(streams)), use, lic))

    head = ("id", "access", "binding", "streams", "use", "redist")
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
         for i, h in enumerate(head)]
    print("  ".join(h.ljust(x) for h, x in zip(head, w)))
    print("  ".join("-" * x for x in w))
    for r in sorted(rows):
        print("  ".join(c.ljust(x) for c, x in zip(r, w)))

    by: dict[str, int] = {}
    for _, (_, card) in cards.items():
        by[str(card.get("binding", "?"))] = by.get(str(card.get("binding", "?")), 0) + 1
    print()
    print(f"{len(cards)} cards: " + ", ".join(f"{n} {k}" for k, n in sorted(by.items())))
    print("an unbound card is usable for planning and not for fitting")
    return 0


# ---------------------------------------------------------------------------
# bind
# ---------------------------------------------------------------------------


def cmd_bind(args: argparse.Namespace) -> int:
    # NB: ibm.forge re-exports a *function* named `bind`, which overwrites the
    # package attribute pointing at the submodule of the same name.  neither
    # `from ibm.forge import bind` nor `import ibm.forge.bind as m` gets the
    # module back; a from-import of the submodule's members resolves through
    # sys.modules and does.
    from ibm.forge.bind import bind as run_bind, unmet, components_wanted

    rep = run_bind(write=args.write, only=tuple(args.only) if args.only else None)
    print(rep if args.verbose else rep.table())
    print()

    gaps = unmet(rep)
    if gaps:
        print("== what is missing, grouped by the fix ==")
        for key, where in gaps.items():
            print(f"{key}: {len(where)} streams")
            for w in where[: args.limit]:
                print(f"    {w}")
            if len(where) > args.limit:
                print(f"    ... and {len(where) - args.limit} more")
        print()

    wanted = components_wanted(rep)
    if wanted:
        print("== component ids the corpus asked for that are not registered ==")
        print("(a synonym or a rewritten card fixes most of these; a new component "
              "fixes few)")
        for cid, where in wanted.items():
            print(f"  {cid:44s} {len(where)} streams, e.g. {where[0]}")
        print()

    b = len(rep.by_status("bound"))
    p = len(rep.by_status("partial"))
    u = len(rep.by_status("unbound"))
    print(f"{b} bound, {p} partial, {u} unbound"
          + (f"; wrote {len(rep.written)} cards" if rep.written else
             ("; run with --write to flip the ones that changed" if not args.write else "")))
    return 0 if u == 0 else 1


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------


def _library() -> dict[str, Any]:
    from ibm.materialize import library
    return library.MODELS


def cmd_materialize(args: argparse.Namespace) -> int:
    """build a named model from `ibm/materialize/library` and describe it.

    the description is the deliverable, not a side effect.  §7 requires a
    materialized model to carry provenance -- which f was selected per process,
    which parameters evidence moved, and where a process is being run outside the
    regime its form is valid in -- and printing that is how a person decides
    whether to believe anything the model then says.
    """
    _load()
    try:
        models = _library()
    except Exception as e:
        print(f"the model library did not import: {type(e).__name__}: {e}", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return 2

    if not args.model:
        print("named models:")
        for mid, m in sorted(models.items()):
            print(f"  {mid:28s} {m.doc.strip().splitlines()[0][:70] if m.doc else ''}")
        return 0

    m = models.get(args.model)
    if m is None:
        print(f"no model {args.model!r}; known: {', '.join(sorted(models))}", file=sys.stderr)
        return 2

    print(f"== {m.id} ==")
    if m.doc:
        print(m.doc.strip())
        print()
    print(m.request.describe())
    print()
    print(f"fit sources    {', '.join(m.fit_sources) or '(none)'}")
    print(f"eval sources   {', '.join(m.eval_sources) or '(none)'}")
    print(f"teachers       {', '.join(m.teachers) or '(none)'}")
    print(f"controls       {', '.join(m.negative_controls) or '(none)'}")
    print()
    print("constrained by the listed sources:")
    for c in m.constrained:
        print(f"  + {c}")
    print("prior-dominated -- it will come out smooth because nothing here distinguishes it:")
    for c in m.prior_dominated:
        print(f"  ~ {c}")

    built = _build(m, args)
    if built is None:
        print()
        print("(ibm.materialize.build is not available yet, so this is the request and its "
              "accounting rather than an allocated model)")
        return 0

    print()
    try:
        from ibm.runtime.state import Layout
        layout = Layout.of(built)
        print("== state layout ==")
        print(layout.describe())
    except Exception as e:
        print(f"(no state layout: {type(e).__name__}: {e})")
    for attr in ("describe", "provenance"):
        v = getattr(built, attr, None)
        if callable(v):
            print()
            print(v())
    return 0


def _build(m: Any, args: argparse.Namespace) -> Any:
    """call whatever `ibm.materialize` exposes as its builder, or give up quietly.

    the builder is being written independently of this file, so this tries the
    plausible entry points and returns None rather than failing: `ibm materialize`
    is useful for reading a request's accounting long before anything can allocate
    a model from it.
    """
    try:
        import importlib
        mod = importlib.import_module("ibm.materialize.build")
    except Exception:
        return None
    for name in ("materialize", "build", "build_model"):
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                return fn(m.request)
            except Exception as e:
                print(f"build failed: {type(e).__name__}: {e}", file=sys.stderr)
                if args.traceback:
                    traceback.print_exc()
                return None
    return None


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    """cost and budget for a materialization, before anything is allocated.

    the two budgets multiply.  sites times retained spectral components is the
    whole accounting, and either axis alone tells you almost nothing: a
    whole-brain haemodynamic model is spatially broad and temporally narrow, a
    single-electrode spike model is the reverse, and both are affordable while
    their product is not.  so this prints both and their product, and it prints
    the window's own limits beside them, because a request whose band exceeds its
    nyquist is not expensive -- it is wrong.
    """
    _load()
    try:
        models = _library()
    except Exception as e:
        print(f"the model library did not import: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    chosen = ([models[args.model]] if args.model in models
              else sorted(models.values(), key=lambda m: m.id))
    if args.model and args.model not in models:
        print(f"no model {args.model!r}", file=sys.stderr)
        return 2

    rows = []
    for m in chosen:
        r = m.request
        w = r.window
        band = r.resolution.default_band
        k = w.k_for(band)
        ok, why = w.covers(band)
        sites = args.sites
        state_vars = sites * max(len(r.target_components), 1)
        coeffs = state_vars * k
        over = r.budget.check(state_variables=state_vars, spectral_coefficients=coeffs,
                              n_bytes=coeffs * 16)
        rows.append((m.id, f"{w.n}", f"{w.dt:g}", f"{w.duration_s:g}", f"{k}",
                     f"{state_vars:,}", f"{coeffs:,}", f"{coeffs * 16 / 2**30:.2f}",
                     "ok" if not over and ok else "OVER" if over else "BAND"))
        if not ok:
            rows[-1] = rows[-1][:-1] + (f"BAND: {why[:40]}",)

    head = ("model", "n", "dt s", "win s", "k", "state vars", "coefficients", "GiB", "")
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
         for i, h in enumerate(head)]
    print("  ".join(h.ljust(x) for h, x in zip(head, w)))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(c.ljust(x) for c, x in zip(r, w)))
    print()
    print(f"state variables assume {args.sites:,} sites per component -- pass --sites to "
          "vary it.  the real count comes from r(q) at build time; this is the shape of "
          "the cost, not its value")

    if args.parameters:
        from ibm.forge.priors import assemble
        print()
        print("== p(theta) over every declared implementation ==")
        print(assemble().describe())
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ibm",
        description="implicit brain model: inspect the ontology, the corpus, and what a "
                    "materialization would cost")
    p.add_argument("--traceback", action="store_true",
                   help="print full tracebacks instead of one-line failures")
    sub = p.add_subparsers(dest="command")

    o = sub.add_parser("ontology", help="print the registry tables and the summary")
    o.add_argument("--table", choices=["components", "processes", "topologies", "observations"],
                   help="print one table instead of all of them")
    o.set_defaults(fn=cmd_ontology)

    c = sub.add_parser("check", help="run REGISTRY.check() and print problems")
    c.add_argument("--errors-only", action="store_true")
    c.set_defaults(fn=cmd_check)

    s = sub.add_parser("sources", help="list data/sources cards")
    s.add_argument("--access", help="filter by access: held, priority, restricted, watch, "
                                    "reference")
    s.add_argument("--binding", help="filter by binding: bound, partial, unbound")
    s.set_defaults(fn=cmd_sources)

    b = sub.add_parser("bind", help="resolve cards against the registry")
    b.add_argument("--write", action="store_true",
                   help="rewrite each card's binding: line where the status changed")
    b.add_argument("--only", nargs="*", help="bind only these card ids")
    b.add_argument("-v", "--verbose", action="store_true", help="per-stream detail")
    b.add_argument("--limit", type=int, default=8, help="examples printed per gap")
    b.set_defaults(fn=cmd_bind)

    m = sub.add_parser("materialize", help="build a named model and describe it")
    m.add_argument("model", nargs="?", help="omit to list the library")
    m.set_defaults(fn=cmd_materialize)

    pl = sub.add_parser("plan", help="cost and budget for a materialization")
    pl.add_argument("model", nargs="?")
    pl.add_argument("--sites", type=int, default=10_000,
                    help="sites per component to assume (default 10000)")
    pl.add_argument("--parameters", action="store_true",
                    help="also print p(theta) over every declared implementation")
    pl.set_defaults(fn=cmd_plan)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 0
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        if getattr(args, "traceback", False):
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
