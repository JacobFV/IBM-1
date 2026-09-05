"""p(theta | D): forging over heterogeneous datasets.

    p(theta | D)  proportional to  p(theta) prod_d p(D_d | theta)

ARCHITECTURE.md §4 says forging *has the semantics of posterior updating* and
that one mechanism covers hand-engineered initialization, atlas and literature
initialization, statistical fitting, pretrained learned processes, heterogeneous
supervised forging, distillation and subject-specific adaptation.  the whole
point of this package is that those are not seven pipelines.  they are one
product, and they differ only in which factor a source is permitted to be:

    priors.py   p(theta): what every implementation declared, expanded by tying
    fit.py      the update itself, per process wherever the data allow
    bind.py     which factor a source card is even allowed to contribute

`bind.py` is load-bearing and is easy to mistake for plumbing.  a source card
names registered component ids and a registered process id; until those resolve,
the card records what a provider's documentation said and cannot contribute a
likelihood to anything.  that is the mechanism behind "an unbound card is usable
for planning and not for fitting", and it is also what stops a teacher's output
being folded in as though it were a measurement -- the card declares `use: distil`
and a distillation precision, and the fit refuses to treat the two alike.

what is deliberately absent: there is no data loader here.  a likelihood is a
callable, and where its number comes from is the caller's business.  that keeps
this package about the update and not about zarr.
"""

#: the functions `fit.fit` and `bind.bind` are deliberately NOT re-exported here.
#: binding those names in this package would shadow the `ibm.forge.fit` and
#: `ibm.forge.bind` *modules*, so `import ibm.forge.bind as b` would hand back a
#: function -- a failure that only shows up at the call site and reads as a
#: mystery.  call them as `from ibm.forge.fit import fit`.
from ibm.forge.priors import (
    ParameterBlock, ParameterSpace, assemble, expand, log_prior, median_of, sample_prior,
)
from ibm.forge.fit import FitReport, Method, Task, provenance_after
from ibm.forge.bind import BindReport, CardBinding, StreamBinding, load_cards

__all__ = [
    "BindReport", "CardBinding", "FitReport", "Method", "ParameterBlock", "ParameterSpace",
    "StreamBinding", "Task", "assemble", "expand", "load_cards", "log_prior",
    "median_of", "provenance_after", "sample_prior",
]
