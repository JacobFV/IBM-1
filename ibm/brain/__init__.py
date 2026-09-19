"""the brain, declared structure by structure, assembled into one circuit.

Read `docs/BRAIN_SPEC.md` first: it says what this is being built to be.  This package is
the declaration of it.  `ibm/circuit.py` is the engine that runs it.

THE CONTRACT
------------
Every structure module in this package exports exactly these, and nothing else is required
of it:

    STRUCTURE : str              the prefix every population id in this module starts with
    pops()    -> [Pop]           its populations
    internal()-> [Proj]          projections whose source and target are both inside it
    external()-> [Proj]          projections that leave or enter it, naming populations in
                                 other structures by their full id.  Declared by whichever
                                 structure the pathway is named for -- the corticostriatal
                                 projection belongs to the basal ganglia, the thalamocortical
                                 one to the thalamus -- and `assemble()` reports any edge
                                 whose other end does not exist rather than dropping it.
    mods()    -> [Mod]           neuromodulatory edges this structure originates
    drives()  -> {pop_id: float} the tonic drive each population sits at, used for
                                 calibration.  A structure that needs no tonic drive returns
                                 {} and says so in its docstring.
    targets() -> {...}           what this structure is FOR, in numbers: the sparsity each
                                 population should run at, the bands it should carry, and any
                                 known answer a gate can check.  This is a specification, not
                                 a measurement, and it is what the gates are written against.

Population ids are `structure.part.class`: `ctx.v1.E`, `thal.lgn.relay`, `bg.str.d1`,
`nm.lc`.  The structure prefix is checked at assembly, so a typo is a failure rather than a
silently orphaned population.

WHY THE SHAPE IS THIS
---------------------
Because the previous shape -- a bespoke module per structure, each with its own state dict,
its own interfaces and its own hand-set dials -- cannot be assembled.  Five such modules
exist, they pass their gates, and there is no way to run them together, which is why nothing
in this programme has ever measured more than two structures at once.  `docs/BRAIN_SPEC.md`
says the object we want is the state where all of it is active together; this package exists
so that object can be built at all.
"""
from __future__ import annotations

import importlib
import pkgutil

from ibm.circuit import Circuit, Mod, Pop, Proj  # noqa: F401  (re-exported for modules)

#: every structure module in this package, in the order they are assembled.  A module not in
#: this list is not in the brain -- there is no discovery by accident.
STRUCTURES = (
    "cortex", "thalamus", "basal_ganglia", "hippocampus", "cerebellum",
    "valuation", "neuromodulators", "hypothalamus", "brainstem", "cord", "olfactory",
)


def load(name: str):
    return importlib.import_module(f"ibm.brain.{name}")


def available() -> list:
    """the structure modules that actually exist on disk, in STRUCTURES order."""
    here = {m.name for m in pkgutil.iter_modules(__path__)}
    return [s for s in STRUCTURES if s in here]


def missing() -> list:
    """declared in STRUCTURES and not yet written.  Printed by `assemble`, never ignored."""
    here = {m.name for m in pkgutil.iter_modules(__path__)}
    return [s for s in STRUCTURES if s not in here]


def collect(names=None):
    """(pops, projs, mods, drives, targets, report) over the named structures.

    Cross-structure edges whose other end is missing are NOT dropped: they are returned in
    `report["orphans"]` with the structure that declared them, because an edge that quietly
    disappears is a loop that quietly stops existing.
    """
    names = list(names or available())
    pops, internal, external, mods, drives, targets = [], [], [], [], {}, {}
    for n in names:
        m = load(n)
        ps = list(m.pops())
        bad = [p.id for p in ps if not p.id.startswith(m.STRUCTURE + ".")
               and p.id != m.STRUCTURE]
        assert not bad, f"{n}: population ids outside structure '{m.STRUCTURE}': {bad}"
        pops += ps
        internal += list(m.internal())
        external += list(m.external())
        mods += list(m.mods())
        drives.update(m.drives())
        targets[n] = m.targets()
    have = {p.id for p in pops}
    kept = [e for e in external if e.src in have and e.dst in have]
    orphans = [{"edge": e.key, "declared_by": e.src.split(".", 1)[0]}
               for e in external if e.src not in have or e.dst not in have]
    report = {"structures": names, "missing_structures": missing(),
              "n_pops": len(pops), "n_internal": len(internal),
              "n_external_kept": len(kept), "orphans": orphans,
              "units": sum(p.n for p in pops)}
    return pops, internal + kept, mods, drives, targets, report


def assemble(names=None, seed: int = 0, device="cpu"):
    """build the Circuit.  Returns (circuit, drives, targets, report)."""
    pops, projs, mods, drives, targets, report = collect(names)
    c = Circuit(pops, projs, mods, seed=seed, device=device)
    return c, drives, targets, report
