"""anatomical partitioning systems (ARCHITECTURE.md §2).

    a(q) in [0,1]^k,  ||a(q)||_1 <= 1

importing this package registers every system.  `systems` declares what the
partitions are; `sources` records how a(q) is actually obtained for one subject,
which atlas it comes from, in which frame, crisply or probabilistically, and what
a process can no longer distinguish when the substitute is used instead.

they are two files because they fail separately.  a system can be perfectly
declared and unobtainable, and an atlas can be obtainable and mean something other
than what the declaration says.
"""

from __future__ import annotations

from ibm.anatomy import systems, sources
from ibm.anatomy.sources import SOURCES, AtlasSource
from ibm.anatomy.systems import (
    AMYGDALAR_NUCLEI, BG_TERRITORIES, BRAINSTEM_NUCLEI, CEREBELLAR_LOBULES,
    CEREBELLAR_MICROZONES, CORTICAL_AREAS, CORTICAL_LAYERS, CYTOARCHITECTURE,
    HIPPOCAMPAL_SUBFIELDS, HYPOTHALAMIC_NUCLEI, STRIOSOME_MATRIX,
    THALAMIC_NUCLEI, VASCULAR_TERRITORIES,
)

__all__ = [
    "systems", "sources", "SOURCES", "AtlasSource",
    "CORTICAL_AREAS", "CORTICAL_LAYERS", "CYTOARCHITECTURE", "THALAMIC_NUCLEI",
    "HIPPOCAMPAL_SUBFIELDS", "STRIOSOME_MATRIX", "BG_TERRITORIES",
    "HYPOTHALAMIC_NUCLEI", "AMYGDALAR_NUCLEI", "CEREBELLAR_LOBULES",
    "CEREBELLAR_MICROZONES", "BRAINSTEM_NUCLEI", "VASCULAR_TERRITORIES",
]
