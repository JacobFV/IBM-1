"""executing a materialized model.

the ontology declares what exists and `ibm.materialize` decides what to
instantiate; nothing in either of them computes.  this package is where a
materialized model is actually advanced, constrained and clamped, and it is four
modules because the architecture draws exactly four lines here.

    state.py       the global state vector: beliefs, blocked by component
    step.py        dot x = sum_p f_p, solved over a window
    fuse.py        J' = J + dJ: evidence, which is not pressure
    intervene.py   externally constrained state, applied after accumulation
    propagate.py   the push-forward wherever f IS linear -- exactly, in closed form
    ensemble.py    the push-forward wherever f is not

the separation between `step` and `fuse` is the one that carries weight.
ARCHITECTURE.md §4 ends by saying that dynamical pressure and gaussian evidence
"compose by different rules" -- pressure moves state, evidence constrains it --
and the cheapest way to violate that is to have one function that takes both.  so
there is no such function.  a process contributes through `State.add` and
`step`; a measurement contributes through `fuse`; a clamp overrides both through
`intervene`; and the ensemble exists only because the first of those stops being
exact the moment f is nonlinear.

`propagate` and `ensemble` are two halves of one sentence in §4: the induced
distribution `P_f(p(x), p(theta))` is exact and closed-form for a linear f and
has to be sampled otherwise.  they are separate modules because the exact half is
the one that runs on essentially every materialization `build` selects, and it
must not be reachable only through a code path whose entry condition is "there is
a nonlinearity here".  that was the arrangement until `propagate` existed, and
its consequence was that an all-LTI model returned every psd bit-identical to its
prior, silently, for the entire life of the runtime.

what the runtime deliberately does not do: it does not choose an implementation
for a process, resolve a region, build a topology, or decide a resolution.  all
of that happened at materialization and is recorded in the model's provenance.
by the time anything here runs, the only remaining questions are numerical.
"""

from ibm.runtime.state import Block, Layout, State, StateVector, View
#: `step.step` is deliberately NOT re-exported.  binding the name `step` in this
#: package would shadow the `ibm.runtime.step` module, so `import
#: ibm.runtime.step as s` would hand back a function -- a confusing failure that
#: only appears at the call site.  the whole-run entry point is `advance`; the
#: convenience wrapper lives at `ibm.runtime.step.step` for anyone who wants it.
from ibm.runtime.step import (
    Carry, CausalityViolation, Coupling, Solve, StepReport, WindowPlan,
    advance, couplings_of, longest_memory, solve_window,
)
from ibm.runtime.fuse import (
    Evidence, TeacherPrecision, distillation_precision, fuse, fuse_scalar, fuse_spectral,
)
from ibm.runtime.intervene import Clamp, ClampConflict, apply_clamps, baseline, from_intervention
from ibm.runtime.ensemble import Ensemble, NonGaussianity, advise, propagate, reproject_nonlinear
from ibm.runtime.propagate import (
    Propagate, PropagationReport, TargetReport, band_table, propagate_linear,
)

__all__ = [
    "Block", "Carry", "CausalityViolation", "Clamp", "ClampConflict", "Coupling",
    "Ensemble", "Evidence", "Layout", "NonGaussianity", "Propagate", "PropagationReport",
    "Solve", "State", "StateVector", "StepReport", "TargetReport", "TeacherPrecision",
    "View", "WindowPlan", "advance", "advise", "apply_clamps", "band_table", "baseline",
    "couplings_of", "distillation_precision", "from_intervention",
    "fuse", "fuse_scalar", "fuse_spectral", "longest_memory", "propagate",
    "propagate_linear", "reproject_nonlinear", "solve_window",
]
