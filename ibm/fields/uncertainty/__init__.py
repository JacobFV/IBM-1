"""how a belief about a component is carried.

not a concept alongside the four primitives -- it is part of the field
declaration.  a component whose dynamics have one relevant timescale is a scalar
gaussian; one in which several timescales interact is spectral.  processes are
written against selectors and never name a form, which is what lets the neural
field be spectral and the blood field scalar without any process knowing.

a learned latent is not a form.  a latent is an encoding, not a physical
quantity, and nothing exerts pressure on it; latents live inside a process's f
and are invisible to the state graph.
"""

from ibm.fields.uncertainty.base import UncertaintyForm, Conversion, convert
from ibm.fields.uncertainty import scalar, spectral

__all__ = ["UncertaintyForm", "Conversion", "convert", "scalar", "spectral"]
