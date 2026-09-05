"""F = {x(q) : q in Omega} -- the fields, and everything a field declaration needs.

importing this package *is* the declaration.  supports, uncertainty forms and
prior builders register on import, then every field module registers its own
components, and afterwards `ibm.fields` has populated the registry with the whole
state graph of ibm-1.  nothing here computes; `REGISTRY.table("components")`
prints what was declared.

the import order is not incidental.  supports first, because a field names one and
the registry checks it.  uncertainty forms second, because a component names one
and a process that couples two of them needs the conversion between them declared.
priors third, because a component names a builder and the check below is only
possible once every builder exists.  the field modules last, in no particular
order among themselves -- they do not reference each other, which is the point of
a registry.

the check at the bottom is deliberately load-time.  a component pointing at a
prior builder that does not exist, or at one of the wrong uncertainty form, is a
declaration error, and a declaration error should surface when the ontology is
loaded rather than when a materialization three layers down tries to build a
belief and finds a KeyError.  it costs one pass over the component table.
"""

from __future__ import annotations

from ibm.fields import supports          # noqa: F401  the domains fields are indexed over
from ibm.fields import uncertainty       # noqa: F401  how a belief is carried, and conversions
from ibm.fields import priors            # noqa: F401  named resting beliefs

from ibm.fields import neural            # noqa: F401
from ibm.fields import extracellular     # noqa: F401
from ibm.fields import electromagnetic   # noqa: F401
from ibm.fields import blood             # noqa: F401
from ibm.fields import csf               # noqa: F401
from ibm.fields import metabolic         # noqa: F401
from ibm.fields import structural        # noqa: F401
from ibm.fields import material          # noqa: F401
from ibm.fields import thermal           # noqa: F401
from ibm.fields import mechanical        # noqa: F401
from ibm.fields import transduction      # noqa: F401
from ibm.fields import effector          # noqa: F401
from ibm.fields import device            # noqa: F401

from ibm.registry import REGISTRY

_bad = priors.check_components(REGISTRY)
if _bad:
    raise ValueError("field declarations do not validate:\n  " + "\n  ".join(_bad))
del _bad

#: the fields, in the order the architecture's table lists them.
FIELDS = (
    "neural", "extracellular", "electromagnetic", "blood", "csf", "metabolic",
    "structural", "material", "thermal", "mechanical", "transduction", "effector",
    "device",
)

__all__ = [
    "FIELDS", "supports", "uncertainty", "priors",
    "neural", "extracellular", "electromagnetic", "blood", "csf", "metabolic",
    "structural", "material", "thermal", "mechanical", "transduction", "effector",
    "device",
]
