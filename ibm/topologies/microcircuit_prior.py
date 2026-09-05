"""the microcircuit as a calibrated prior: a measured circuit, an honest spread.

`microcircuit.py` declares WHICH populations at a position may act on one another
and says almost nothing about how strongly, because a topology is a support and
strength belongs to the process (ARCHITECTURE.md §3).  `ibm/processes/neural.py`
then declares the strengths -- and every one of them that is not a receptor time
constant is `weak()` or `speculative()`, with notes that say so plainly: "the
connectivity matrix among interneuron classes is measured in mouse, in slice, in
a handful of areas, and its translation to human cortex at mesoscale is a guess".

that note was right when it was written and is no longer the best available.
dense electron microscopy with PROOFREAD AXONS measures the same matrix directly,
and this module is what that measurement turns into.  it is the microcircuit's
counterpart to `tract_prior.py` and it makes the same argument in a different
place, so the two are worth reading together: **be generous with the support,
informative with the prior**, and record which tier the numbers came from,
because a generative guess and a measurement have the same type and must never
have the same authority.

what proofreading buys, and why nothing else would do
-----------------------------------------------------
a connection probability is a ratio and both halves of it have to be complete.
an automatically segmented axon breaks every few tens of microns, so a
connectome built from one reports the segmentation's reach rather than the cell's
-- and no amount of averaging repairs it, because the error is topological in
exactly the sense H01's own source card records for its cell segmentation.  what
makes the measurement possible is MICrONS's 564 hand-traced axons, of which 77
are `axon_fully_extended`: within the volume their output is complete, so the
numerator (cells this axon contacts) and the denominator (cells whose soma is in
the volume) refer to the same thing.

the size of what that buys is itself measured.  the same quantity computed from
`axon_partially_extended` cells of the same class is smaller by

    E -> E    x1.37     E -> PV   x1.15     E -> SST  x1.18     E -> VIP  x1.64

so a published non-proofread connection probability for excitatory recurrence is
low by about a third, and by more for the sparser pathways where a missing branch
costs a whole connection.  those factors are in the evidence file, and they are
the conversion a reader needs before comparing any of this with the literature.

the numbers, and what they overturn
-----------------------------------
measured by `scripts/measure_microcircuit.py` on MICrONS materialization 1078,
71,551 typed neurons and 564 proofread axons, and recorded in
`data/sources/microns/evidence/microcircuit_statistics@1/measured.json`.
connection probability is within 100 um of tangential intersomatic distance;
`lambda` is the space constant of an exponential fit to p(d) out to 1 mm.

    pre -> post     p(<100 um)   lambda      synapses/connection
    E   -> E          0.033      220 um            1.15
    E   -> PV         0.121      207 um            1.37
    E   -> SST        0.118      162 um            1.37
    E   -> VIP        0.044      233 um            1.18
    PV  -> E          0.131      111 um            2.26
    SST -> E          0.195       47 um            2.10
    VIP -> E          0.015       68 um            1.37
    VIP -> SST        0.181      118 um            2.60

four things in that table are worth more than their decimal places.

**inhibition is denser and more local than excitation, by a lot.**  an SST cell
contacts 20% of the pyramidal cells within 100 um and an excitatory cell contacts
3%.  the space constants differ by a factor of five, 47 um against 220 um.  a
model in which E and I couple over the same neighbourhood -- which is what a
single `radius_mm` on `microcircuit_within_site` produces -- has the geometry of
inhibition wrong by that factor, and the direction of the error is the one that
matters: it makes inhibition look more diffuse than it is, which is the
assumption under which lateral inhibition can substitute for local gain control.

**inhibitory connections carry twice the synapses.**  E -> E is 1.15 synapses per
connection and PV -> E is 2.26.  the ontology has nowhere to put that at present;
the process parameter `gain` absorbs it silently, and it means the E and I gains
are not comparable numbers even before any electrophysiology enters.

**the disinhibitory motif is real and is a factor of twenty.**  VIP -> SST is
0.181 and VIP -> E is 0.015, and weighting each by its synapse count makes the
ratio 22.6.  `local_inhibition`'s docstring asserts that vip cells inhibit sst
cells "rather than principal cells"; that assertion is now measured, and its size
is the number `class_coupling_scale` was standing in for.

**the canonical laminar cascade is half confirmed and half refuted.**  see below.

the laminar motif, distance divided out
---------------------------------------
`laminar_propagation` declares four directed edges -- L4 -> L2/3 -> L5 -> L6 with
an L6 -> L4 return -- and says the *asymmetry* between them is the entire content.
the raw connection probabilities cannot test that, because layers are separated
in depth and connection probability falls with distance; a "laminar" effect and a
distance effect are the same number until they are separated.  so each candidate
pair is given the probability the excitatory p(d) curve predicts at its own
intersomatic distance, and the laminar specificity is observed over expected:

    L4  -> L2/3   1.44      L2/3 -> L4    0.67      ratio  2.15
    L2/3-> L5     3.41      L5   -> L2/3  4.56      ratio  0.75
    L5  -> L6     1.30      L6   -> L5    0.71      ratio  1.84
    L6  -> L4     0.30      L4   -> L6    0.74      ratio  0.41

the first and third confirm the declaration: the feedforward L4 -> L2/3 edge is
twice as likely as its reverse once distance is accounted for, and L5 -> L6 runs
1.8 times more strongly downward than upward.

the second is a correction rather than a refutation.  the L2/3 <-> L5 coupling is
the strongest laminar preference in the circuit -- both directions are three to
four times what distance alone predicts -- but it is very nearly SYMMETRIC, and
what asymmetry there is runs the wrong way, favouring the L5 -> L2/3 return by
1.3.  declaring L2/3 -> L5 as a step of a cascade is therefore right about the
edge and wrong about its character: it is a reciprocal loop, and the `laminar`
topology's docstring already knows this ("a distinct 5 -> 2/3 return") while the
process's four-edge cascade does not.

the fourth is a negative result and is reported as one.  the L6 -> L4 feedback
edge, which is the edge that closes the canonical loop, is measured at 0.30 --
three times LESS likely than distance alone predicts, and weaker than the
L4 -> L6 direction nobody declares.  read it with its sample size: only 10 layer
6 pyramidal cells in this volume have proofread axons, and MICrONS's 6P-CT cells
send most of their axon to thalamus, which leaves the cortex before it can be
counted.  the honest statement is that this volume does not support the L6 -> L4
edge and cannot rule it out either; what it does rule out is treating it as
comparable in strength to L4 -> L2/3.

the lateral extent, and what a column is worth
----------------------------------------------
this is the number the whole exercise was aimed at, because it decides whether
`microcircuit`'s within-site self-edge is a description or an approximation.
tangential distance from a proofread cell to the cells it contacts:

                    median    p90     within 200 um   within 300 um
    excitatory      146 um   337 um       70%              88%
    PV              113 um   214 um       87%              98%
    SST             107 um   206 um       89%              99%
    VIP              99 um   194 um       91%              99%

**a column of radius 200 um contains 70% of a pyramidal cell's local output and
about 90% of every interneuron's.**  that asymmetry is the finding.  a
materialization at 200-300 um spacing represents inhibition almost exactly as a
within-site relation and excitation only approximately, so `microcircuit_within_
site` with `radius_mm = 0` is a good model of the inhibitory circuit and a
truncation of the excitatory one -- and the third of excitatory connectivity it
drops is precisely the part that carries information between columns.

it also puts a number on the guard `microcircuit_within_site` already enforces.
`max_radius_mm = 0.5` was declared as a judgement; at 500 um the measurement says
95.5% of excitatory and effectively 100% of inhibitory local output is enclosed,
so the guard is in the right place and can now say why.

**what this cannot measure is the tail.**  the volume is 1.38 x 0.80 mm in the
tangential plane, so a connection longer than about a millimetre has nowhere to
land, and the excitatory p(d) curve stops falling at around 0.002 from 300 um out
to 700 um -- a floor, not a decay.  that floor is the beginning of
`cortical_association`'s territory and this module says nothing about it.

species, and the part that does not travel
------------------------------------------
MICrONS is one mouse.  H01 is one human, one cubic millimetre of temporal cortex,
resected from a person with epilepsy, and its axons are not proofread -- so it
cannot produce a connection probability at all and this module does not pretend
it can.  what it produces is a census, and the census disagrees with the mouse:

    fraction of classified neurons that are inhibitory
        MICrONS (mouse visual)        0.110
        H01 (human temporal)          0.308

that is a factor of 2.8 and it is not a rounding difference.  part of it is real
-- human cortex has a higher interneuron fraction than mouse, and this is one of
the few places the literature and both volumes agree -- and part of it is the two
classifiers disagreeing about what counts as a neuron; MICrONS's own automatic
labels agree with its manual ones only 84% of the time, on the same cells.  the
figure is reported and not reconciled.

two more human numbers came out of it and both are worth having.

**40% of H01's synapses are inhibitory**, over 96.5 million sampled ones, and
47.7% of the input to a pyramidal segment is.  that is two to three times the
mouse literature's 15-20% and it is high enough to be worth doubting, so the
synapse class was not taken on trust from the release note: the code called
inhibitory here targets somata and axon initial segments 6.5 times as often as
the other code, which is what inhibition does and excitation does not, so the
label is settled from the tissue.  the number stands as H01's own classifier's
answer and is reported without being reconciled with the rodent literature.

**human cortex is 2.24 mm from pia to white in this block and the mouse volume's
somata span 0.80 mm of depth.**  that ratio is the cleanest single reason a
lateral extent in absolute microns cannot be carried across: a 220 um space
constant is 28% of the mouse ribbon's thickness and 10% of the human one, and
nothing measured anywhere says which of those two fractions is the invariant.

and one comparison was attempted and failed, which is worth recording because it
is the obvious one.  the fraction of a cell's INPUT that is inhibitory is the only
quantity both volumes measure in the same way, so it was computed on MICrONS's
2.48 million input synapses to proofread cells -- and only 8.1% of their
presynaptic partners have a soma in the volume AND a cell-type call.  inhibitory
axons are locally complete and excitatory ones mostly arrive from outside the
cubic millimetre, so the 8% that can be attributed is enriched in inhibition by
construction and the resulting 0.53 is an artefact of attribution rather than a
measurement of the tissue.  it is in the evidence file, labelled, and it is not
compared with H01's 0.48; two numbers that look alike for different reasons are
worse than no number.

so, explicitly, what transfers and what does not.

    TRANSFERS (shape, ratios, motifs)
      the shape of p(d): exponential with a floor, on both species' anatomy
      the ORDERING of connection probabilities: E->PV and E->SST about four times
        E->E, inhibition denser than excitation, VIP->SST >> VIP->E
      the fact that inhibitory connections use about twice the synapses
      the laminar motif's sign where this measured it (L4 -> L2/3, L5 -> L6)
      cellular kinetics as ratios: pv membrane time constants are a third of
        pyramidal ones in mouse and the ratio holds in the human recordings

    DOES NOT TRANSFER
      absolute connection probability.  it is a density, human cortex is thicker,
        sparser in cell bodies and larger in dendritic field, and no measurement
        anywhere converts one into the other
      lambda in absolute microns.  a space constant is a length and human cortex
        is not a scaled mouse -- the 220 um is a mouse number and applying it to
        a human column is the error this paragraph exists to prevent
      anything about human ASSOCIATION cortex.  MICrONS is visual, H01 is
        temporal, and neither is prefrontal or parietal; the granular layer the
        L4 -> L2/3 edge terminates in is thick in V1, thin in association cortex
        and absent in agranular cortex, which `learned_laminar_gains` already
        says and this measurement cannot improve on
      the interneuron fraction, as above
      anything at all about the fourth interneuron class.  neurogliaform cells
        are 0.9% of the volume's neurons and have measurable connectivity here,
        and `ibm.fields.neural` declares no component for them; they are carried
        through the measurement and dropped at the prior rather than folded into
        sst because they are also dendrite-targeting

the effective sample count, which is the reason the spread stays wide
---------------------------------------------------------------------
`tract_prior.py` had 96 tractograms of one phantom and had to say how many
independent votes that was.  the same question here has a worse answer and it is
the single most important number in this module.

within the volume, the per-cell spread of log connection probability is measured
at 0.60 in natural-log units for excitatory cells -- a factor of 1.8 between
otherwise identical cells -- and the intraclass correlation of that spread across
nine spatial blocks of the volume is 0.070.  routed through
`ibm.runtime.fuse.TeacherPrecision.effective_constraints`, 53 proofread
excitatory axons are worth 11.4 independent ones, and 80 PV axons are worth 19.
that is the honest within-volume figure and it is not small.

**it is also not the figure that matters.**  every one of those cells shares one
animal, one cortical area, one fixation, one segmentation model and one
proofreading team, and an intraclass correlation computed inside the volume is
blind to all five by construction -- they have zero between-block variance.  the
between-animal term is not measured here because it cannot be: there is one
volume.  so the prior below adds a declared cross-volume widening, marked as a
declaration rather than a measurement, in exactly the way `tract_prior`'s tier-2
term was declared before TractoInferno made it measurable.  the day a second
proofread cubic millimetre exists, that term becomes a measurement and this
docstring should be rewritten around it.

the tiers
---------
    SUBJECT_MICROCIRCUIT   this subject's own synapse-level reconstruction.
                           **never available in vivo and never will be.**  it is
                           declared so that the ontology has a name for what it
                           is approximating, and so that a materialization can
                           never silently claim it.
    SPECIES_ATLAS          a dense EM volume of the right species, or the wrong
                           one.  everything below.
    GENERATIVE_PRIOR       depth and cell-type composition, run through the
                           measured p(d) shape.  a guess with a measured shape,
                           which is a better guess than `weak()` and not a
                           measurement.

the first tier being permanently empty is not a defect of this module.  it is the
statement `association.py` makes about tiers made true in the one place where the
top tier is unreachable rather than merely expensive, and a model that reports
tier 2 as though it were tier 1 is making a claim no instrument can support.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field as _field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ibm.topologies import builders as B
from ibm.vocabulary import Prior, Provenance, Tying, lognormal, normal

_MICRONS = ("data/sources/microns/evidence/microcircuit_statistics@1/measured.json")
_H01 = ("data/sources/h01/evidence/human_microcircuit_census@1/measured.json")
_ALLEN = ("data/sources/allen-cell-types-patchseq/evidence/"
          "population_kinetics@1/measured.json")

#: the population components the four measured classes are evidence about.  NGC
#: has no entry on purpose -- see the docstring.
COMPONENT_OF = {"E": "neural.exc.activity", "PV": "neural.pv.activity",
                "SST": "neural.sst.activity", "VIP": "neural.vip.activity"}


class Tier(str, Enum):
    """which of the three sources of microcircuit structure a build actually used.

    an ordering, and unusually one whose top element is empty.  a materialization
    records which tier it reached and provenance reports it, for the same reason
    `tract_prior.Tier` exists: the edges are what gets passed around and the
    provenance is what gets dropped.
    """

    SUBJECT_MICROCIRCUIT = "subject_microcircuit"
    SPECIES_ATLAS = "species_atlas"
    GENERATIVE_PRIOR = "generative_prior"

    @property
    def rank(self) -> int:
        return {"subject_microcircuit": 0, "species_atlas": 1,
                "generative_prior": 2}[self.value]

    def demoted_to(self, other: "Tier") -> "Tier":
        """the weaker of two tiers.  weakness wins, always -- as in `tract_prior`.

        a caller may cap a tier downward (an EM atlas of the wrong species is not
        this species' microcircuit however it was computed) and may never raise
        one, because an intention is not evidence about where data came from.
        """
        return self if self.rank >= other.rank else other


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrocircuitStatistics:
    """what the cortical microcircuit actually is, measured rather than assumed.

    the defaults are the MICrONS figures and they are compiled in so that this
    module works with no data on disk -- but none of them is a literature value or
    a round number.  each is reproduced by `scripts/measure_microcircuit.py` and
    recorded with its provenance in
    `data/sources/microns/evidence/microcircuit_statistics@1/`.

    read every connection probability as **conditional on one mouse and on the
    volume's own extent**.  a connection whose two cells would be a millimetre
    and a half apart has nowhere to be observed, so the tail is truncated and
    every space constant below is shortened by an unknown amount; and the whole
    table is visual cortex.
    """

    #: p(at least one synapse | the two somata are within 100 um tangentially),
    #: keyed 'PRE->POST'.  the pairs marked lower_bound below came from
    #: partially-extended axons and are low by the factors in `proofreading`.
    p_within_100um: Mapping[str, float] = _field(default_factory=lambda: {
        "E->E": 0.03254, "E->PV": 0.12149, "E->SST": 0.11824, "E->VIP": 0.04395,
        "E->NGC": 0.05558,
        "PV->E": 0.13052, "PV->PV": 0.19273, "PV->SST": 0.11762, "PV->VIP": 0.06845,
        "SST->E": 0.19477, "SST->PV": 0.19677, "SST->SST": 0.15869, "SST->VIP": 0.16795,
        "VIP->E": 0.01521, "VIP->PV": 0.13668, "VIP->SST": 0.18115, "VIP->VIP": 0.08614,
    })
    #: space constant of p(d) = p0 exp(-d / lambda), microns, tangential.
    lambda_um: Mapping[str, float] = _field(default_factory=lambda: {
        "E->E": 220.3, "E->PV": 207.0, "E->SST": 161.7, "E->VIP": 232.6,
        "E->NGC": 162.5,
        "PV->E": 111.0, "PV->PV": 109.7, "PV->SST": 121.2, "PV->VIP": 94.8,
        "SST->E": 47.0, "SST->PV": 54.1, "SST->SST": 54.0, "SST->VIP": 44.9,
        "VIP->E": 68.3, "VIP->PV": 86.6, "VIP->SST": 117.7, "VIP->VIP": 198.5,
    })
    #: mean number of anatomical synapses in a connection that exists.  the
    #: excitatory entries sit just above 1 and the inhibitory ones just above 2,
    #: and that factor of two is invisible everywhere in the current ontology.
    synapses_per_connection: Mapping[str, float] = _field(default_factory=lambda: {
        "E->E": 1.145, "E->PV": 1.366, "E->SST": 1.372, "E->VIP": 1.180,
        "E->NGC": 1.226,
        "PV->E": 2.256, "PV->PV": 2.433, "PV->SST": 1.962, "PV->VIP": 1.699,
        "SST->E": 2.097, "SST->PV": 2.058, "SST->SST": 1.959, "SST->VIP": 2.220,
        "VIP->E": 1.370, "VIP->PV": 1.949, "VIP->SST": 2.595, "VIP->VIP": 1.582,
    })
    #: pairs whose presynaptic cells were only partially proofread.  their
    #: probabilities are LOWER BOUNDS and the note on every prior built from them
    #: says so.
    lower_bound_pairs: tuple[str, ...] = (
        "PV->E", "PV->PV", "PV->SST", "PV->VIP", "PV->NGC",
        "VIP->E", "VIP->PV", "VIP->SST", "VIP->VIP", "VIP->NGC")
    #: p measured on fully-extended axons over p measured on partially-extended
    #: ones, for the pairs where both exist.  the conversion from a non-proofread
    #: published figure into this table's units.
    proofreading: Mapping[str, float] = _field(default_factory=lambda: {
        "E->E": 1.365, "E->PV": 1.145, "E->SST": 1.181, "E->VIP": 1.641,
        "E->NGC": 2.085, "SST->E": 1.021, "SST->PV": 1.026, "SST->SST": 1.425,
        "SST->VIP": 0.936})
    #: observed over distance-predicted connections, per ordered layer pair of
    #: excitatory cells.  1.0 is "exactly as often as distance alone predicts".
    laminar_specificity: Mapping[str, float] = _field(default_factory=lambda: {
        "L2/3->L2/3": 1.153, "L2/3->L4": 0.669, "L2/3->L5": 3.414, "L2/3->L6": 0.855,
        "L4->L2/3": 1.437, "L4->L4": 1.151, "L4->L5": 1.766, "L4->L6": 0.741,
        "L5->L2/3": 4.561, "L5->L4": 0.866, "L5->L5": 1.890, "L5->L6": 1.304,
        "L6->L2/3": 0.406, "L6->L4": 0.302, "L6->L5": 0.709, "L6->L6": 0.772})
    #: how many proofread presynaptic cells stand behind each layer's row above.
    #: L6 is 10, which is why the L6 -> L4 refutation is reported as a failure to
    #: support rather than as a refutation.
    laminar_n_pre: Mapping[str, int] = _field(default_factory=lambda: {
        "L2/3": 61, "L4": 50, "L5": 60, "L6": 10})
    #: fraction of a cell's within-volume output landing inside a disc of the
    #: given tangential radius, per presynaptic class.
    enclosed_by_radius: Mapping[str, Mapping[str, float]] = _field(
        default_factory=lambda: {
            "E": {"100": 0.293, "200": 0.703, "300": 0.876, "500": 0.955},
            "PV": {"100": 0.417, "200": 0.867, "300": 0.981, "500": 0.997},
            "SST": {"100": 0.451, "200": 0.889, "300": 0.987, "500": 1.000},
            "VIP": {"100": 0.505, "200": 0.912, "300": 0.986, "500": 0.998}})
    #: median tangential distance to a contacted cell, microns.
    lateral_median_um: Mapping[str, float] = _field(default_factory=lambda: {
        "E": 145.6, "PV": 113.3, "SST": 107.4, "VIP": 99.1})
    #: fraction of typed neurons in each class.  a mouse visual cortex census.
    class_fraction: Mapping[str, float] = _field(default_factory=lambda: {
        "E": 0.8903, "PV": 0.0462, "SST": 0.0340, "VIP": 0.0207, "NGC": 0.0088})
    inhibitory_fraction: float = 0.1097
    #: natural-log sd of connection probability BETWEEN CELLS of one class in one
    #: volume.  the spread a per-cell prior deserves before any cross-volume term.
    per_cell_log_sd: Mapping[str, float] = _field(default_factory=lambda: {
        "E": 0.596, "PV": 0.577, "SST": 0.521, "VIP": 0.673})
    #: fraction of that spread shared by cells in the same spatial block of the
    #: volume.  a LOWER bound on what a second animal would contribute, and blind
    #: by construction to everything constant across the volume.
    icc_within_volume: Mapping[str, float] = _field(default_factory=lambda: {
        "E": 0.0701, "PV": 0.0407, "SST": 0.0, "VIP": 0.0})
    n_proofread_axons: Mapping[str, int] = _field(default_factory=lambda: {
        "E": 53, "PV": 80, "SST": 17, "VIP": 33})
    #: how many animals, and therefore the ceiling on everything above.  one.
    n_volumes: int = 1
    n_animals: int = 1
    #: fraction of the classifier's automatic labels that match a hand label on
    #: the same cell.  the stratification of this whole table is only this good.
    classifier_agreement: float = 0.843
    #: fraction of a proofread axon's synapses landing on a fragment with no soma
    #: in the volume.  those targets are real cells that were cut, so every
    #: connection probability here is conditioned on the target being resolvable.
    orphan_target_fraction: float = 0.2434
    volume_tangential_mm: tuple[float, float] = (1.381, 0.502)
    volume_depth_mm: float = 0.796
    #: the DECLARED cross-volume widening, in natural-log units, and the one
    #: number in this dataclass that is not a measurement.  a factor of three
    #: between two animals' microcircuits is a guess; it is set at the same
    #: magnitude `tract_prior` used for its declared tier-3 term, and it is
    #: labelled here rather than hidden inside `theta_prior` so that a reader can
    #: find it and a second volume can delete it.
    cross_volume_log_sd: float = 1.0986        # log(3.0)
    species: str = "Mus musculus, visual cortex"
    source: str = ("MICrONS minnie65 public release, materialization 1078; 71,551 typed "
                   "neurons, 564 proofread axons of which 77 fully extended")
    measured_at: str = "2026-09-05"
    from_evidence: bool = False

    @classmethod
    def load(cls, root: Path | None = None) -> "MicrocircuitStatistics":
        """read the measurement if it is on disk, else use the compiled-in copy.

        the same contract as `tract_prior.TractUncertainty.load` and for the same
        reason: the defaults ARE the file's contents, so a checkout with no
        evidence directory behaves identically, and `from_evidence` is what lets a
        provenance report distinguish "measured here" from "measured once and
        carried in the source".
        """
        try:
            from ibm.forge.bind import repo_root
            base = root or repo_root(Path(__file__))
            data = json.loads((base / _MICRONS).read_text())
        except Exception:
            return cls()
        try:
            p100: dict[str, float] = {}
            lam: dict[str, float] = {}
            lo: list[str] = []
            per_cell: dict[str, float] = {}
            icc: dict[str, float] = {}
            n_ax: dict[str, int] = {}
            cp = data["connection_probability"]
            for strat in ("axon_partially_extended", "axon_fully_extended"):
                # fully-extended second so that it OVERWRITES: where both exist the
                # proofread number is the measurement and the other is its bound.
                for src, e in (cp.get(strat) or {}).items():
                    for post, v in e["by_post_class"].items():
                        key = f"{src}->{post}"
                        p100[key] = float(v["p_within_100um"])
                        if v["fit"].get("lambda_um"):
                            lam[key] = float(v["fit"]["lambda_um"])
                        if strat == "axon_partially_extended":
                            lo.append(key)
                        else:
                            lo = [k for k in lo if k != key]
                    pc = e["per_cell_log_p"]
                    # no max() here: the second pass OVERWRITES, so a class whose
                    # probabilities came from fully-extended axons also reports the
                    # fully-extended cell count.  taking the larger of the two would
                    # pair 53 cells' measurement with 128 cells' worth of confidence,
                    # which is the exact arithmetic `effective_axons` exists to stop.
                    per_cell[src] = float(pc["sd"] or 0.0)
                    if pc.get("icc_across_blocks") is not None:
                        icc[src] = float(pc["icc_across_blocks"])
                    n_ax[src] = int(pc["n_cells"])
            spc = {k: float(v["mean"]) for k, v in data["synapses_per_connection"].items()}
            lam_spec = {f"{a}->{b}": float(v["ratio"])
                        for a, row in data["laminar"]["distance_controlled_specificity"].items()
                        for b, v in row.items() if v.get("ratio")}
            enc, med = {}, {}
            for k, v in data["lateral_extent"].items():
                strat, cls_ = k.split(":")
                if cls_ in enc and strat != "axon_fully_extended":
                    continue
                enc[cls_] = {r: float(v[f"frac_within_{r}um"]) for r in
                             ("100", "200", "300", "500")}
                med[cls_] = float(v["median_um"])
            cen = data["census"]
            n = float(cen["n_neurons"])
            vol = data["volume_um"]
            return cls(
                p_within_100um=p100, lambda_um=lam, synapses_per_connection=spc,
                lower_bound_pairs=tuple(sorted(set(lo))),
                proofreading={k: float(v) for k, v in
                              data["proofreading_correction_p_fully_over_partially"].items()},
                laminar_specificity=lam_spec,
                laminar_n_pre={k: int(v) for k, v in
                               data["laminar"]["n_pre_by_layer"].items()},
                enclosed_by_radius=enc, lateral_median_um=med,
                class_fraction={k: v / n for k, v in cen["by_class"].items()},
                inhibitory_fraction=float(cen["inhibitory_fraction"]),
                per_cell_log_sd=per_cell, icc_within_volume=icc, n_proofread_axons=n_ax,
                classifier_agreement=float(cen["classifier_agreement_with_manual"]),
                orphan_target_fraction=float(data["orphan_target_fraction"]),
                volume_tangential_mm=((vol["x_um"][1] - vol["x_um"][0]) / 1000.0,
                                      (vol["z_um"][1] - vol["z_um"][0]) / 1000.0),
                volume_depth_mm=(vol["y_um"][1] - vol["y_um"][0]) / 1000.0,
                species=str(data.get("species", "")),
                source=str((data.get("sources") or ["microns"])[0]),
                measured_at=str(data.get("measured_at", "")), from_evidence=True)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return cls()

    # -- derived quantities ------------------------------------------------

    def coupling_weight(self, pre: str, post: str) -> float | None:
        """p x synapses per connection: the structural half of a synaptic gain.

        the quantity a process's `gain` parameter multiplies before any
        electrophysiology enters -- how many synapses one presynaptic cell places
        on a typical postsynaptic one at short range.  it is not a conductance and
        this module never pretends it is: the quantal conductance and the release
        probability are not in an electron micrograph, so an ABSOLUTE gain cannot
        come from here.  what can, and what `gain_ratio` returns, is the ratio
        between two pathways, in which the missing factors cancel to the extent
        that they are the same at both synapse types -- which for two glutamatergic
        synapses is a fair assumption and for a glutamatergic against a gabaergic
        one is not.
        """
        k = f"{pre}->{post}"
        p, s = self.p_within_100um.get(k), self.synapses_per_connection.get(k)
        return None if p is None or s is None else p * s

    def gain_ratio(self, pre: str, post: str, ref: tuple[str, str] = ("E", "E")) -> float | None:
        """`coupling_weight` relative to a reference pathway, default E -> E."""
        a, b = self.coupling_weight(pre, post), self.coupling_weight(*ref)
        return None if not a or not b else a / b

    def connection_probability(self, pre: str, post: str, distance_um: Any, np=None) -> Any:
        """p(d) for one ordered class pair, from the measured p0 and lambda.

        an exponential, because that is what fits: the weighted fits behind the
        table run r^2 of 0.73 to 0.99 in log space, and the worst of them are the
        pairs with the fewest connections rather than a pair whose shape is wrong.

        it is extrapolated freely beyond the volume and should not be trusted
        there.  the measured curves flatten to a floor of about 0.002 for
        excitatory pairs past 300 um and this form does not reproduce that; a
        materialization that needs the floor wants `cortical_association`, whose
        whole subject it is.
        """
        np = np or B._numpy("microcircuit_prior")
        k = f"{pre}->{post}"
        p100, lam = self.p_within_100um.get(k), self.lambda_um.get(k)
        if p100 is None or not lam:
            return np.zeros_like(np.asarray(distance_um, float))
        # p_within_100um is the mean over a disc, not the value at the origin.  the
        # mean of exp(-d/l) over 0..100 um weighted by 2 pi d is what the measured
        # number is, so p0 is recovered by dividing that out rather than by taking
        # p100 as p(0) -- which would understate the peak by about a third at
        # lambda = 50 um and hardly at all at 220 um.
        d = np.asarray(distance_um, float)
        r = 100.0
        u = r / lam
        mean_kernel = 2.0 * (1.0 - (1.0 + u) * math.exp(-u)) / (u * u)
        p0 = p100 / max(mean_kernel, 1e-9)
        return np.clip(p0 * np.exp(-d / lam), 0.0, 1.0)

    def effective_axons(self, klass: str, n: int | None = None) -> float:
        """`n` proofread axons of one class are worth how many independent ones.

        the same call into `ibm.runtime.fuse` that `tract_prior` makes, on a
        different rho, and separate for the same reason: the two rhos are not the
        same kind of number and a caller who could pass either would eventually
        pass the wrong one.

        the answer is WITHIN ONE VOLUME.  it is the right discount for "how much
        do these 53 cells constrain a parameter shared across the volume" and the
        wrong one for "how much do they constrain a parameter shared across
        cortex", for which the answer is bounded above by `n_animals`, which is 1.
        """
        n = int(n if n is not None else self.n_proofread_axons.get(klass, 0))
        if n <= 0:
            return 0.0
        rho = min(max(self.icc_within_volume.get(klass, 0.0), 0.0), 1.0 - 1e-9)
        try:
            import numpy as np

            from ibm.runtime.fuse import TeacherPrecision
            tp = TeacherPrecision(r2=0.0, correlated_fraction=rho, error_rank=1,
                                  source=self.source)
            ev = tp.evidence("structural.synaptic_density", np.zeros(n), np.ones(n))
            return float(ev.effective_constraints())
        except Exception:
            return n / ((1.0 - rho) + n * rho)

    def log_sd(self, klass: str, *, tier: "Tier" = Tier.SPECIES_ATLAS,
               same_species: bool = False) -> float:
        """the spread a prior on this class's coupling deserves, in log units.

        three terms and they do different jobs.  the first is the measured
        between-cell spread, reduced by the EFFECTIVE number of proofread axons
        rather than by their raw number -- the same correction `tract_prior`
        makes and for the same reason, that averaging correlated observations
        stops buying certainty at 1/rho.

        the second is the declared cross-volume term, and it is added OUTSIDE the
        reduction because no number of cells in one animal averages away the fact
        that it is one animal.  it dominates: the measured part of an excitatory
        prior is 0.60/sqrt(11.4) = 0.18 and the declared part is 1.10, so a
        species-atlas prior is wide however many axons were proofread.  that is
        the correct shape, and it is the same shape `tract_prior` found at tier 2
        -- what is being approximated is not this brain measured badly, it is
        another brain measured well.

        `same_species=False` adds it a second time, in quadrature, for a mouse
        atlas applied to human cortex.  there is no measurement behind that either
        and the docstring at the top of this module lists what it is standing in
        for.  it is the term a human proofread volume would replace.
        """
        base = self.per_cell_log_sd.get(klass, 0.6)
        n_eff = max(self.effective_axons(klass), 1.0)
        v = base ** 2 / n_eff
        if tier is not Tier.SUBJECT_MICROCIRCUIT:
            v += self.cross_volume_log_sd ** 2
        if not same_species:
            v += self.cross_volume_log_sd ** 2
        if tier is Tier.GENERATIVE_PRIOR:
            v += math.log(2.0) ** 2
        return math.sqrt(v)

    def describe(self) -> str:
        return (f"microcircuit [{self.source}, {self.measured_at}, "
                f"{'from evidence' if self.from_evidence else 'compiled in'}]: "
                f"E->E p(<100um) {self.p_within_100um.get('E->E', 0):.3f} at lambda "
                f"{self.lambda_um.get('E->E', 0):.0f} um, E->PV "
                f"{self.p_within_100um.get('E->PV', 0):.3f}, SST->E "
                f"{self.p_within_100um.get('SST->E', 0):.3f} at lambda "
                f"{self.lambda_um.get('SST->E', 0):.0f} um; inhibitory connections carry "
                f"{self.synapses_per_connection.get('PV->E', 0)/max(self.synapses_per_connection.get('E->E', 1), 1e-9):.1f}x "
                f"the synapses of excitatory ones; a 200 um column holds "
                f"{self.enclosed_by_radius.get('E', {}).get('200', 0):.0%} of excitatory and "
                f"{self.enclosed_by_radius.get('SST', {}).get('200', 0):.0%} of sst output; "
                f"{self.n_proofread_axons.get('E', 0)} proofread excitatory axons are worth "
                f"{self.effective_axons('E'):.1f} within the volume and the volume count is "
                f"{self.n_volumes}")


@dataclass(frozen=True)
class HumanCensus:
    """what H01 supports, which is a census and not a circuit.

    kept as a separate type from `MicrocircuitStatistics` deliberately, and for
    the same reason `tract_prior` keeps `GroupUncertainty` apart from
    `TractUncertainty`: the two answer different questions and a single type
    would let a caller read a human connection probability off an object that
    does not have one.  H01's axons are not proofread, so it HAS no connection
    probability, and the absence is structural rather than a gap to be filled in
    later from the same bytes.
    """

    #: inhibitory neurons over classified neurons.  2.8 times the mouse figure,
    #: and see the module docstring for how much of that is real.
    inhibitory_fraction: float = 0.3080
    n_somas: int = 49379
    n_excitatory: int = 10531
    n_inhibitory: int = 4688
    #: fraction of a pyramidal segment's input synapses that are inhibitory,
    #: averaged over segments, from the released per-segment counts -- complete
    #: rather than sampled, so this is a real human number with no proofreading
    #: caveat.  it is very high against the mouse literature's 15-20% and it is
    #: reported rather than reconciled; it rests on H01's own synapse classifier.
    inhibitory_input_fraction: float | None = 0.4768
    #: fraction of ALL synapses in the volume that are inhibitory, from 96.5
    #: million sampled synapses.  the class code behind it was settled from the
    #: tissue and not from the release note: the code called inhibitory here
    #: targets somata and axon initial segments 6.5 times as often as the other,
    #: which is what inhibition does and excitation does not.
    inhibitory_synapse_fraction: float | None = 0.4013
    #: pia-to-white distance between the layer-1 and white-matter soma centroids
    #: along the fitted depth axis.  2.24 mm against the mouse volume's 0.80 mm of
    #: soma depth range: the single cleanest reason a lateral extent in absolute
    #: microns cannot be carried from one species to the other.
    cortical_depth_span_mm: float | None = 2.243
    #: what fraction of the volume's synapses the held shards actually contain,
    #: MEASURED against the released whole-volume per-segment counts rather than
    #: assumed to be 100/166 = 0.602.  the two agreeing to 6% is the evidence that
    #: the shards are a random partition and a prefix of them is a uniform sample.
    sampling_fraction: float = 0.568
    n_synapses_sampled: int = 96474370
    species: str = "Homo sapiens, temporal cortex, one person, epilepsy surgery"
    source: str = "H01 release 20210601, c3 soma table and segment properties"
    measured_at: str = "2026-09-05"
    from_evidence: bool = False

    @classmethod
    def load(cls, root: Path | None = None) -> "HumanCensus":
        try:
            from ibm.forge.bind import repo_root
            base = root or repo_root(Path(__file__))
            d = json.loads((base / _H01).read_text())
            c = d["census"]
            return cls(
                inhibitory_fraction=float(c["inhibitory_fraction_of_classified_neurons"]),
                n_somas=int(c["n_somas"]), n_excitatory=int(c["n_excitatory"]),
                n_inhibitory=int(c["n_inhibitory"]),
                inhibitory_input_fraction=(
                    d.get("ei_input", {}).get("inhibitory_input_fraction_pyramidal")),
                inhibitory_synapse_fraction=(
                    d.get("synapse_sample", {}).get("inhibitory_synapse_fraction")),
                sampling_fraction=float(
                    d.get("synapse_sample", {}).get("sampling_fraction_measured") or 0.0),
                n_synapses_sampled=int(
                    d.get("synapse_sample", {}).get("n_synapses_sampled") or 0),
                cortical_depth_span_mm=float(c["cortical_depth_span_um"]) / 1000.0,
                species=str(d.get("species", "")),
                source=str((d.get("sources") or ["h01"])[0]),
                measured_at=str(d.get("measured_at", "")), from_evidence=True)
        except Exception:
            return cls()

    def describe(self) -> str:
        return (f"human census [{self.source}, "
                f"{'from evidence' if self.from_evidence else 'compiled in'}]: "
                f"{self.n_somas:,} somata, inhibitory fraction "
                f"{self.inhibitory_fraction:.3f} of classified neurons, "
                f"{(self.inhibitory_synapse_fraction or 0):.3f} of "
                f"{self.n_synapses_sampled / 1e6:.0f}M sampled synapses inhibitory, "
                f"{(self.cortical_depth_span_mm or 0):.2f} mm pia to white -- NO "
                f"connection probability, because the axons are not proofread")


@dataclass(frozen=True)
class CellKinetics:
    """membrane kinetics of identified cells, both species, as distributions.

    not connectivity, and here because three of the parameters the three target
    processes declare are cellular rather than circuit quantities and are measured
    in hundreds of identified cells.  there is nothing to infer: the Allen Cell
    Types Database reports a membrane time constant per cell and the measurement
    is a grouping and a spread.

    the caveat that governs all of it: these are whole-cell recordings in a quiet
    slice at rest.  the *effective* membrane time constant in the high-conductance
    state of active cortex is several times shorter, and that effective value is
    what `tau_membrane_s` actually means -- as the parameter's own note in
    `ibm/processes/neural.py` already says.  so these numbers pin the SPREAD and
    the RATIOS between cell classes exactly, and the median only up to the
    high-conductance correction, which nothing here measures.
    """

    #: median membrane time constant in ms, and the natural-log sd around it.
    tau_ms: Mapping[str, tuple[float, float]] = _field(default_factory=lambda: {
        "mouse_spiny": (20.43, 0.315), "human_spiny": (26.37, 0.345),
        "mouse_aspiny": (10.27, 0.566), "human_aspiny": (13.85, 0.540),
        "mouse_pvalb": (7.02, 0.335), "mouse_sst": (23.68, 0.568),
        "mouse_vip": (13.24, 0.448)})
    #: resting potential, mV, mean and sd.  liquid-junction corrected, which is
    #: most of why they sit below the -65 mV the ontology carries.
    vrest_mv: Mapping[str, tuple[float, float]] = _field(default_factory=lambda: {
        "mouse_spiny": (-73.0, 5.6), "human_spiny": (-70.7, 4.4),
        "mouse_pvalb": (-73.4, 4.6), "mouse_sst": (-70.2, 5.4),
        "mouse_vip": (-71.6, 5.3)})
    #: input resistance, MOhm, median and log sd.  1000/Ri is the resting membrane
    #: conductance `shunting_rate`'s `g_leak` compares a synaptic conductance to.
    ri_mohm: Mapping[str, tuple[float, float]] = _field(default_factory=lambda: {
        "mouse_spiny": (163.4, 0.364), "human_spiny": (103.0, 0.633),
        "mouse_pvalb": (109.0, 0.359), "mouse_sst": (232.1, 0.401),
        "mouse_vip": (208.6, 0.385)})
    n_mouse: int = 1920
    n_human: int = 413
    source: str = "Allen Cell Types Database, 2333 patch-clamp specimens"
    measured_at: str = "2026-09-05"
    from_evidence: bool = False

    @classmethod
    def load(cls, root: Path | None = None) -> "CellKinetics":
        try:
            from ibm.forge.bind import repo_root
            base = root or repo_root(Path(__file__))
            d = json.loads((base / _ALLEN).read_text())
            tau, vr, ri = {}, {}, {}
            for g, v in d["by_group"].items():
                if v.get("tau_ms"):
                    tau[g] = (float(v["tau_ms"]["median"]),
                              float(v["tau_ms"]["log_sd"] or 0.4))
                if v.get("vrest_mv"):
                    vr[g] = (float(v["vrest_mv"]["mean"]), float(v["vrest_mv"]["sd"]))
                if v.get("ri_mohm"):
                    ri[g] = (float(v["ri_mohm"]["median"]),
                             float(v["ri_mohm"]["log_sd"] or 0.4))
            return cls(tau_ms=tau, vrest_mv=vr, ri_mohm=ri,
                       n_mouse=int(d["n_mouse"]), n_human=int(d["n_human"]),
                       source=str((d.get("sources") or ["allen"])[0]),
                       measured_at=str(d.get("measured_at", "")), from_evidence=True)
        except Exception:
            return cls()

    def g_leak_ns(self, group: str = "mouse_spiny") -> tuple[float, float]:
        """resting membrane conductance in nS, as (median, log sd).

        the reciprocal of input resistance, which is a measurement, and the log sd
        carries across unchanged because inverting a lognormal negates its median
        in log space and leaves its width alone.
        """
        r, s = self.ri_mohm.get(group, (150.0, 0.4))
        return 1000.0 / r, s

    def describe(self) -> str:
        t = self.tau_ms
        return (f"cell kinetics [{self.source}, "
                f"{'from evidence' if self.from_evidence else 'compiled in'}]: "
                f"pyramidal tau {t.get('mouse_spiny', (0, 0))[0]:.1f} ms in mouse and "
                f"{t.get('human_spiny', (0, 0))[0]:.1f} ms in human, pv "
                f"{t.get('mouse_pvalb', (0, 0))[0]:.1f} ms; the pv/pyramidal ratio is "
                f"{t.get('mouse_pvalb', (1, 0))[0] / max(t.get('mouse_spiny', (1, 0))[0], 1e-9):.2f} "
                f"and it is the part that transfers")


MEASURED = MicrocircuitStatistics.load()
HUMAN = HumanCensus.load()
KINETICS = CellKinetics.load()


# ---------------------------------------------------------------------------
# what this replaces, and what it does not
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterGrounding:
    """one parameter of one implementation, and what the measurement does to it.

    the point of the type is `status`.  a report that listed only the parameters
    a measurement improved would be a sales document; what a reader needs is the
    complete list of the parameters this exercise was AIMED at, with the ones it
    could not move marked as such and the reason given.  of the thirteen below,
    six are `measured`, two are `partly_measured` -- the structural half of a gain
    is measured and the physiological half is not, so what changes is the RATIO
    between two gains and not either one -- and five could not be moved at all:
    three `still_literature`, one `still_weak` and one `still_speculative`.  those
    five are the honest yield of the negative half and each says why the data
    cannot reach it.

    nothing here edits `ibm/processes/neural.py`.  the priors are exposed for a
    human to wire, because replacing a declared prior changes what every fit in
    the repository is regularized toward and that is not a change a measurement
    script gets to make on its own.
    """

    process: str
    implementation: str
    param: str
    #: what the declaration currently says, in words.
    current: str
    #: what it would say if this measurement were wired in.  None when the
    #: measurement cannot support a replacement.
    proposed: Prior | None
    #: measured | partly_measured | still_literature | still_weak
    status: str
    why: str

    def describe(self) -> str:
        head = f"{self.process}:{self.implementation}.{self.param}  [{self.status}]"
        if self.proposed is None:
            return f"{head}\n    stays {self.current}\n    {self.why}"
        p = self.proposed
        shown = (f"lognormal median {math.exp(p.loc):.4g} x{math.exp(p.scale):.2f}"
                 if p.dist == "lognormal" else
                 f"normal {p.loc:.4g} +- {p.scale:.3g}")
        return (f"{head}\n    was  {self.current}\n    now  {shown} "
                f"[{p.provenance.value}] {p.units}\n    {self.why}")


def parameter_priors(stats: MicrocircuitStatistics | None = None,
                     kinetics: CellKinetics | None = None,
                     *, species: str = "mouse") -> tuple[ParameterGrounding, ...]:
    """the measured replacements for what the three processes currently declare.

    returned rather than applied.  `ibm/processes/neural.py` is not edited by this
    module and must not be: a prior is a statement about what the model believes
    before it sees data, changing one silently changes the meaning of every
    posterior that was ever reported against it, and the provenance field would
    then be the only record that anything had happened.  what this function does
    is put the measurement and the declaration side by side with a status on each
    pair, so that the wiring is a decision somebody takes rather than a side
    effect of running a script.

    `species` selects which of the Allen groups the cellular parameters come from.
    it defaults to mouse, not because ibm-1 is a mouse model but because the
    circuit half of this measurement is mouse and mixing a human membrane time
    constant into a mouse connectivity matrix would produce a chimera that is
    neither -- and the human numbers are one line away for a caller who wants a
    human cellular prior with the species caveat stated.
    """
    s = stats or MEASURED
    k = kinetics or KINETICS
    pyr = f"{species}_spiny"
    out: list[ParameterGrounding] = []

    # -- local_excitation ------------------------------------------------
    tau, tau_sd = k.tau_ms.get(pyr, (20.4, 0.32))
    out.append(ParameterGrounding(
        "local_excitation", "ei_loop_lti", "tau_membrane_s",
        "lognormal(0.015 s, x1.4), LITERATURE, 'cortical pyramidal membrane time "
        "constant, 10-20 ms'",
        lognormal(tau / 1000.0, math.exp(tau_sd), units="s",
                  provenance=Provenance.ATLAS,
                  source=k.source,
                  note=f"median of {k.n_mouse if species == 'mouse' else k.n_human} "
                       f"whole-cell recordings of spiny cells; the declared 15 ms sits "
                       f"at the 25th percentile of the measured distribution and the "
                       f"declared x1.4 is very close to the measured x"
                       f"{math.exp(tau_sd):.2f}.  it is still a RESTING value: the "
                       f"high-conductance-state correction the parameter's own note "
                       f"asks for is not measured here and is the reason this is atlas "
                       f"and not fit"),
        "measured",
        "the declared spread was right and the declared centre was 25% low.  the same "
        "prior appears verbatim in `conductance_lti` and `wilson_cowan_adaptive` and "
        "the replacement applies to all three."))
    tau_pv, tau_pv_sd = k.tau_ms.get("mouse_pvalb", (7.0, 0.34))
    out.append(ParameterGrounding(
        "local_excitation", "ei_loop_lti", "tau_inh_membrane_s",
        "lognormal(0.008 s, x1.4), LITERATURE, 'fast-spiking interneuron membrane "
        "time constant'",
        lognormal(tau_pv / 1000.0, math.exp(tau_pv_sd), units="s",
                  provenance=Provenance.ATLAS, source=k.source,
                  note="242 Pvalb-IRES-Cre cells.  the declared value was within 14% of "
                       "the measured median and the declared spread within 1% of the "
                       "measured one -- a literature value that survives contact with "
                       "the distribution, which is worth recording as such"),
        "measured",
        "confirmation rather than correction, and the ratio to the pyramidal value "
        f"({tau_pv / max(tau, 1e-9):.2f}) is the part that transfers across species."))
    out.append(ParameterGrounding(
        "local_excitation", "ei_loop_lti", "loop_gain",
        "lognormal(8.0, x2.0), WEAK, 'the least identifiable and most consequential "
        "parameter here'",
        None, "still_weak",
        "the E -> I -> E loop's STRUCTURAL gain is now measured -- E -> PV coupling is "
        f"{s.gain_ratio('E', 'PV'):.1f} times E -> E and PV -> E is "
        f"{s.gain_ratio('PV', 'E'):.1f} times it -- but `loop_gain` is a product of "
        "that with two quantities an electron micrograph does not contain: the quantal "
        "conductance and the postsynaptic input-output slope.  a measured structural "
        "factor multiplied by two unmeasured ones is not a measured parameter, and "
        "declaring it one would be the failure this whole exercise is against."))
    out.append(ParameterGrounding(
        "local_excitation", "conductance_lti", "gain",
        "weak(1.0, x5.0) nS per Hz, 'absorbs synapse count, release probability and "
        "every unit convention between rate and conductance'",
        None, "partly_measured",
        "one of the three factors the note names is now measured.  synapse count per "
        f"connection is {s.synapses_per_connection.get('E->E', 0):.2f} for E -> E and "
        f"{s.synapses_per_connection.get('PV->E', 0):.2f} for PV -> E, so the E and I "
        "gains differ by a factor of two before any physiology -- which means the two "
        "`gain` parameters in this ontology are not in comparable units and never were. "
        "release probability and the unit convention remain unmeasured, so the prior "
        "stays weak and what changes is that its RATIO to the inhibitory gain is now "
        "constrained.  see `gain_ratio`."))
    out.append(ParameterGrounding(
        "local_excitation", "wilson_cowan_adaptive", "v_rest_mv",
        "normal(-65.0, 4.0) mV, LITERATURE",
        normal(*k.vrest_mv.get(pyr, (-73.8, 5.6)), units="mV",
               provenance=Provenance.ATLAS, source=k.source,
               note="mean over spiny cells, liquid-junction-potential corrected.  the "
                    "8 mV gap from the declared -65 is mostly that correction: many "
                    "older recordings do not apply it and read about 10 mV "
                    "depolarized, so this is a disagreement about a convention as much "
                    "as about a membrane"),
        "measured",
        "the declared sd of 4 mV was 30% narrow against a measured 5.6, and the centre "
        "is 9 mV off.  a resting potential 9 mV from where the model puts it changes "
        "the distance to threshold, which is what the sigmoid in this implementation "
        "is a function of."))
    out.append(ParameterGrounding(
        "local_excitation", "wilson_cowan_adaptive", "adaptation_gain",
        "weak(0.05, x6.0) mV per Hz",
        None, "still_literature",
        "the Allen recordings give an adaptation INDEX (0.034 for mouse spiny cells, "
        "0.002 for pv) which orders the classes correctly and is not in mV per Hz.  "
        "converting one to the other needs the adaptation current's reversal and the "
        "input resistance during the train, neither of which the summary table "
        "carries.  the index is in the evidence file for whoever wants to do that "
        "properly."))

    # -- local_inhibition -------------------------------------------------
    g_med, g_sd = k.g_leak_ns(pyr)
    out.append(ParameterGrounding(
        "local_inhibition", "shunting_rate", "g_leak",
        "weak(1.0, x3.0) nS, 'resting membrane conductance the synaptic conductance "
        "is compared against'",
        lognormal(g_med, math.exp(g_sd), units="nS", provenance=Provenance.ATLAS,
                  source=k.source,
                  note="1000 / input resistance over spiny cells.  it is a RESTING "
                       "conductance and this implementation's whole subject is what "
                       "happens when synaptic conductance becomes comparable to it, so "
                       "a materialization in the high-conductance state should expect "
                       "the effective value several times higher -- which is exactly "
                       "the regime the parameter exists to describe"),
        "measured",
        "the declared median of 1 nS is six times below the measured 6.1, which matters "
        "because this parameter is a denominator: too small a leak makes every "
        "inhibitory conductance look more divisive than it is."))
    out.append(ParameterGrounding(
        "local_inhibition", "learned_interneuron_routing", "class_coupling_scale",
        "speculative(1.0, x10.0), 'the interneuron-class connectivity matrix is the "
        "speculative part; the kinetics are not'",
        lognormal(1.0, math.exp(s.log_sd("SST")), units="dimensionless",
                  provenance=Provenance.ATLAS, source=s.source,
                  note="the matrix is no longer speculative: every entry of it is in "
                       "`MicrocircuitStatistics.p_within_100um` and "
                       "`synapses_per_connection`, measured on proofread axons.  the "
                       "spread is what remains -- the measured between-cell spread "
                       "reduced by the effective axon count, plus a DECLARED "
                       "cross-volume and cross-species term, because it is one mouse"),
        "measured",
        "the parameter this exercise was most pointed at.  the class matrix that was "
        f"speculative is now measured, including the disinhibitory motif at "
        f"{s.gain_ratio('VIP', 'SST') / max(s.gain_ratio('VIP', 'E') or 1e-9, 1e-9):.0f}:1 "
        "for VIP -> SST over VIP -> E.  what keeps the prior wide is not the matrix any "
        "more, it is that there is exactly one volume."))
    out.append(ParameterGrounding(
        "local_inhibition", "gaba_conductance_lti", "gaba_b_fraction",
        "uniform(0.0, 0.4), WEAK, 'rate-dependent in reality'",
        None, "still_literature",
        "gaba-b is metabotropic and leaves no structure an electron micrograph can "
        "see.  nothing in a connectome distinguishes a synapse whose spillover "
        "recruits gaba-b from one that does not, and no amount of proofreading will "
        "change that.  this is a parameter for pharmacology, not for anatomy."))
    out.append(ParameterGrounding(
        "local_inhibition", "gaba_conductance_lti", "gain",
        "weak(1.0, x5.0) nS per Hz",
        None, "partly_measured",
        "the same position as the excitatory `gain`, with one extra number attached: "
        f"PV -> E carries {s.synapses_per_connection.get('PV->E', 0):.2f} synapses per "
        f"connection against E -> E's {s.synapses_per_connection.get('E->E', 0):.2f}, "
        "and PV -> E reaches "
        f"{s.p_within_100um.get('PV->E', 0) / max(s.p_within_100um.get('E->E', 1e-9), 1e-9):.1f} "
        "times as many nearby cells.  their product is the structural ratio between "
        "the two gains and it is about eight; the absolute values remain unmeasured."))

    # -- laminar_propagation ----------------------------------------------
    fw = s.laminar_specificity.get("L4->L2/3")
    out.append(ParameterGrounding(
        "laminar_propagation", "canonical_microcircuit_lti", "gain",
        "weak(1.0, x6.0), 'per-edge laminar gain.  the strongest evidence for it is "
        "rodent slice; treating it as literature for human cortex would be inventing "
        "precision the science has not provided'",
        lognormal(float(fw or 1.0), math.exp(s.log_sd("E")), units="dimensionless",
                  provenance=Provenance.ATLAS, source=s.source,
                  note="the median is the L4 -> L2/3 distance-controlled laminar "
                       "specificity, which is the only one of the four declared edges "
                       "the measurement both confirms and pins.  the OTHER THREE EDGES "
                       "SHOULD NOT SHARE THIS PRIOR: L2/3 -> L5 measures 3.41, L5 -> L6 "
                       "1.30 and L6 -> L4 0.30, and giving four edges one prior is what "
                       "the per-edge parameterization exists to avoid.  see "
                       "`laminar_gain_priors` for the four"),
        "measured",
        "still rodent, as the current note says, and no longer slice: this is a "
        "distance-controlled probability over an intact volume, which removes the "
        "slicing artefact that truncates the very translaminar axons the parameter is "
        "about.  the caution about human cortex stands unchanged and is why the "
        "provenance is ATLAS rather than LITERATURE."))
    out.append(ParameterGrounding(
        "laminar_propagation", "canonical_microcircuit_lti", "delay_s",
        "lognormal(0.0012 s, x1.8), LITERATURE, 'a few hundred microns of largely "
        "unmyelinated vertical axon, plus one synaptic delay'",
        None, "still_literature",
        "the geometry half is now measurable -- the laminar edges in this volume span "
        "150 to 450 um of depth -- and the conduction velocity half is not.  an "
        "electron micrograph shows the axon's calibre and myelination but not its "
        "conduction velocity, and the map between them is a model.  a delay prior "
        "built from measured length and assumed velocity would look measured and be "
        "half assumption."))
    out.append(ParameterGrounding(
        "laminar_propagation", "learned_laminar_gains", "granularity_sensitivity",
        "speculative(1.0, x10.0), 'how strongly the feedforward gain tracks "
        "granular-layer thickness'",
        None, "still_speculative",
        "this needs two areas of different granularity measured the same way.  MICrONS "
        "is one visual area's worth of volume and H01 has no proofread axons at all, "
        "so the comparison that would move this parameter does not exist yet in any "
        "dataset.  a clearly-reported nothing."))
    return tuple(out)


def laminar_gain_priors(stats: MicrocircuitStatistics | None = None,
                        *, same_species: bool = False) -> dict[str, Prior]:
    """one prior per declared laminar edge, because they are not one number.

    `canonical_microcircuit_lti` ties `gain` per partition, which means the four
    declared edges can carry four values -- and the measurement says they should,
    by a factor of eleven from end to end.  handing back a dict keyed by the edge
    is the smallest thing that makes that usable without editing the process.

    the L6 -> L4 entry is the interesting one and it is deliberately not omitted.
    its median is 0.30, three times below what distance alone predicts, and its
    note says the sample is ten cells.  a prior centred there with a wide spread
    is the correct representation of "this volume does not support the edge and
    cannot rule it out"; dropping the edge would be a support decision, and
    `tract_prior`'s whole argument is that support decisions are permanent while
    priors are not.
    """
    s = stats or MEASURED
    sd = math.exp(s.log_sd("E", same_species=same_species))
    n = s.laminar_n_pre
    out = {}
    for edge, npre_layer in (("L4->L2/3", "L4"), ("L2/3->L5", "L2/3"),
                             ("L5->L6", "L5"), ("L6->L4", "L6")):
        v = s.laminar_specificity.get(edge)
        if v is None:
            continue
        rev = "->".join(reversed(edge.split("->")))
        back = s.laminar_specificity.get(rev)
        note = (f"distance-controlled laminar specificity: {v:.2f} times what "
                f"intersomatic distance alone predicts, from {n.get(npre_layer, 0)} "
                f"proofread presynaptic cells in {npre_layer}")
        if back:
            note += (f".  the reverse edge measures {back:.2f}, so the declared "
                     f"asymmetry is {v / back:.2f}"
                     + ("" if v / back > 1.2 else
                        " -- which does NOT favour the declared direction"))
        if n.get(npre_layer, 0) < 20:
            note += (f".  {n.get(npre_layer, 0)} presynaptic cells is thin, and this "
                     "prior is a failure to support the edge rather than a refutation "
                     "of it")
        out[edge] = lognormal(max(v, 1e-3), sd, units="dimensionless",
                              provenance=Provenance.ATLAS, source=s.source, note=note)
    return out


# ---------------------------------------------------------------------------
# theta over an edge set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrocircuitPriorSet:
    """priors over the coupling on a microcircuit edge set, with their provenance.

    the same shape as `tract_prior.TractPriorSet` and for the same reason: the
    tier travels with the priors, because ARCHITECTURE.md §7 requires a prediction
    resting on prior-dominated structure to be reportable as one, and a set at
    `GENERATIVE_PRIOR` and one at `SPECIES_ATLAS` have the same type.
    """

    tier: Tier
    tying: Tying
    priors: tuple[Prior, ...]
    #: 'PRE->POST' for each entry of `priors`, in order.
    pairs: tuple[str, ...] = ()
    group_of: Any = None
    stats: MicrocircuitStatistics = _field(default_factory=lambda: MEASURED)
    note: str = ""

    def __len__(self) -> int:
        return len(self.priors)

    def describe(self) -> str:
        meds = [math.exp(p.loc) for p in self.priors]
        return (f"{len(self.priors)} {self.tying.value} priors at tier "
                f"{self.tier.value}, median {min(meds) if meds else 0:.3g}-"
                f"{max(meds) if meds else 0:.3g}, spread x"
                f"{math.exp(self.priors[0].scale) if self.priors else 0:.1f}"
                + (f"  ({self.note})" if self.note else ""))


def theta_prior(edges: B.EdgeSet, *, tier: Tier | str = Tier.SPECIES_ATLAS,
                tying: Tying = Tying.PER_PARTITION,
                same_species: bool = False,
                reference: tuple[str, str] = ("E", "E"),
                stats: MicrocircuitStatistics | None = None) -> MicrocircuitPriorSet:
    """a lognormal prior on each class-pair coupling, widened by the measurement.

    lognormal for the reason every rate constant in the inventory is: the quantity
    is positive and known to within a multiplicative factor.  here it is more than
    conventional -- the measured between-cell spread IS a factor (1.8 for
    excitatory cells) and the cross-volume term is a factor too.

    **the median is a RATIO and not a conductance.**  `coupling_weight` is
    connection probability times synapses per connection, which is the structural
    half of a synaptic gain and has no units of conductance; dividing by the
    reference pathway removes the quantal conductance and the unit convention,
    both of which are the same unknown at both ends for two synapses of the same
    transmitter.  so a prior on E -> PV relative to E -> E is a measurement, and a
    prior on PV -> E relative to E -> E carries an extra assumption -- that the
    gabaergic and glutamatergic quantal sizes are comparable -- which they are not
    known to be, and the note on those priors says so.

    **the spread does not shrink with the number of cells.**  `log_sd` reduces the
    measured between-cell term by the EFFECTIVE axon count and then adds the
    declared cross-volume and cross-species terms outside that reduction, because
    no number of cells in one animal averages away being one animal.  at the
    measured figures the first term is 0.18 in log units and the others are 1.10
    each, so a species-atlas prior for human cortex is a factor of about 4.7 wide
    whatever else happens.  that is the correct shape and it is the same one
    `tract_prior` found: the thing being approximated is another brain measured
    well, not this brain measured badly.
    """
    s = stats or MEASURED
    tier = Tier(tier)
    pairs = tuple(sorted(s.p_within_100um))
    priors = []
    for key in pairs:
        pre, post = key.split("->")
        r = s.gain_ratio(pre, post, reference)
        if r is None:
            continue
        sd = math.exp(s.log_sd(pre, tier=tier, same_species=same_species))
        note = (f"p(<100 um) {s.p_within_100um[key]:.4f} x "
                f"{s.synapses_per_connection.get(key, float('nan')):.2f} synapses per "
                f"connection, relative to {reference[0]}->{reference[1]}; "
                f"{s.n_proofread_axons.get(pre, 0)} proofread {pre} axons worth "
                f"{s.effective_axons(pre):.1f}")
        if key in s.lower_bound_pairs:
            note += ("; LOWER BOUND -- the presynaptic axons were only partially "
                     "proofread, and the measured correction on comparable pathways "
                     "is 1.0 to 2.1")
        if pre != post and (pre == "E") != (post == "E"):
            note += ("; this ratio crosses transmitters, so the quantal conductance "
                     "does NOT cancel and the number is structural only")
        if not same_species:
            note += ("; mouse visual cortex, and the absolute values do not transfer "
                     "-- see the module docstring")
        priors.append(lognormal(max(r, 1e-4), sd, units="dimensionless",
                                provenance=(Provenance.ATLAS
                                            if tier is Tier.SPECIES_ATLAS
                                            else Provenance.WEAK),
                                source=f"{s.source}; tier {tier.value}", note=note))
    return MicrocircuitPriorSet(
        tier, tying, tuple(priors), pairs, None, s,
        f"{len(priors)} class-pair couplings relative to "
        f"{reference[0]}->{reference[1]}, on an edge set of {edges.n_edges} edges")


# ---------------------------------------------------------------------------
# the generative microcircuit
# ---------------------------------------------------------------------------

_DEPTH_WHAT = (
    "a (n,) array of normalized cortical depth in [0, 1], 0 pial and 1 at the "
    "grey/white boundary, one value per site.  without it every site gets the "
    "volume-average cell-type composition, which is the composition of no layer")
_DEPTH_WHERE = (
    "the same equivolumetric depth `laminar_adjacent` reads (freesurfer / "
    "connectome-workbench surface stack, or nighres layering); or, on a "
    "cortical_depth support, the `depth` column directly")


@B.builder(
    "microcircuit_generative",
    produces=("distance_mm", "connection_probability", "synapses_per_connection",
              "coupling_ratio", "pre_class", "post_class"),
    supports=("cortical_surface", "cortical_depth", "tissue"),
    directed=True,
    metric="tangential distance between column nodes, with the class pair carried "
           "as an edge feature rather than as separate site indices",
    doc="the measured class-by-class circuit, laid out over depth and composition")
def microcircuit_generative(sites, *, support=None, depth=None,
                            radius_mm: float = 0.0, max_radius_mm: float = 0.5,
                            classes: tuple[str, ...] = ("E", "PV", "SST", "VIP"),
                            composition: Mapping[str, Any] | None = None,
                            min_probability: float = 1e-4,
                            stats: MicrocircuitStatistics | None = None) -> B.EdgeSet:
    """generate the microcircuit conditioned on cortical depth and composition.

    the tier-3 builder, and the one a materialization reaches when it has no EM
    volume of its own -- which is every materialization, since tier 1 does not
    exist for a living person.  it produces the SAME edges
    `microcircuit_within_site` produces, at the same sites, and adds three
    features that builder has no way to compute: the measured connection
    probability for the class pair at the site's separation, the measured number
    of synapses in such a connection, and the coupling ratio a process's `gain`
    would multiply.

    **it is generative in the composition and measured in the circuit.**  the two
    halves are worth keeping apart.  what varies with depth is how many cells of
    each class are present -- `composition`, either supplied per site or taken
    from the measured depth profile -- and that is an interpolation over one
    volume's census, which is a guess with a measured shape.  what does not vary
    is p(A -> B | distance), which is measured directly.  a materialization that
    wants to vary the circuit with cytoarchitecture is asking for
    `learned_laminar_gains`, and the parameter it would need is the one this
    exercise could not ground.

    every edge is emitted with `connection_probability` rather than being
    thresholded away, and the default `min_probability` of 1e-4 is deliberately
    four orders of magnitude below the largest.  that is `tract_prior`'s argument
    in its own words: T(i, j) = 0 is permanent and theta = 0 is ordinary, so an
    edge the measurement makes unlikely costs one parameter and a prior centred
    low, while an edge the support refuses is a hypothesis the model can never
    entertain.  the sixteen ordered class pairs at one site are sixteen edges and
    there is no reason to carry fewer.

    `radius_mm` widens the circuit to sub-columnar neighbours exactly as
    `microcircuit_within_site` does, with the same guard and for the same reason
    -- and here the guard can finally say why it is at 0.5 mm: 95.5% of an
    excitatory cell's local output and effectively all of an interneuron's falls
    inside a 500 um tangential radius.
    """
    np = B._numpy("microcircuit_generative")
    s = stats or MEASURED
    sites = B.as_sites(sites)
    declared = ("cortical_surface", "cortical_depth", "tissue")
    if support is None:
        want = [x for x in declared if x in sites.tables]
        if not want:
            raise B.MissingInput(
                "microcircuit_generative", f"sites on one of {', '.join(declared)}",
                "positions carrying neural population state -- column nodes on the "
                "sheet, depth samples through the ribbon, or parenchyma voxels.  the "
                "circuit this builder lays out is a relation among the four population "
                "components at a position, so a materialization with none of these "
                "supports has no positions for them to coexist at",
                "name `cortical_surface`, `cortical_depth` or `tissue` in R")
    else:
        want = [support] if isinstance(support, str) else list(support)
    if radius_mm > max_radius_mm:
        raise ValueError(
            f"microcircuit radius {radius_mm} mm exceeds max_radius_mm {max_radius_mm} "
            "mm.  the measurement behind this builder puts 95% of a proofread cell's "
            "local output inside 500 um and 100% of it inside the volume's 1.4 mm; "
            "past that the numbers are extrapolations of a fit and not measurements. "
            "use `local` or `cortical_surface` for millimetre-scale coupling")

    pairs = [(a, b) for a in classes for b in classes]
    src_l, dst_l, dist_l = [], [], []
    pc_l, qc_l, p_l, n_l, r_l, notes = [], [], [], [], [], []
    for name in want:
        t = sites.require(name, "microcircuit_generative",
                          "positions whose populations interact locally")
        idx = t.gidx(np)
        xyz = np.asarray(t.xyz, dtype=float)
        # depth is optional and its absence is a flat composition, not an error.
        # a materialization on `tissue` has no laminar coordinate at all and is
        # entitled to the volume-average circuit; one on `cortical_depth` carries
        # `depth` as a column and gets a per-layer composition for free.
        got = depth if depth is not None else (
            t.opt("depth") if hasattr(t, "opt") else None)
        d_local = None if got is None else np.asarray(got, dtype=float)

        # the within-site block: every ordered class pair at every position.
        for a, b in pairs:
            key = f"{a}->{b}"
            p0 = s.connection_probability(a, b, np.zeros(t.n), np)
            w = s.gain_ratio(a, b) or 0.0
            src_l.append(idx); dst_l.append(idx)
            dist_l.append(np.zeros(t.n))
            pc_l.append(np.full(t.n, classes.index(a)))
            qc_l.append(np.full(t.n, classes.index(b)))
            p_l.append(_composition_scale(np, s, b, d_local, composition, t.n) * p0)
            n_l.append(np.full(t.n, s.synapses_per_connection.get(key, 1.0)))
            r_l.append(np.full(t.n, w))

        if radius_mm > 0.0:
            i, j, dmm = B.pairs_within(xyz, float(radius_mm), "microcircuit_generative")
            for a, b in pairs:
                key = f"{a}->{b}"
                p = s.connection_probability(a, b, dmm * 1000.0, np)
                keep = p >= min_probability
                if not np.any(keep):
                    continue
                ii, jj, dd, pp = i[keep], j[keep], dmm[keep], p[keep]
                # both directions, with the class pair reversed on the return copy,
                # because A -> B at distance d and B -> A at distance d are two
                # different measured numbers and an undirected edge would have to
                # average them.
                src_l.append(ii + t.offset); dst_l.append(jj + t.offset)
                dist_l.append(dd)
                pc_l.append(np.full(dd.size, classes.index(a)))
                qc_l.append(np.full(dd.size, classes.index(b)))
                p_l.append(pp * _composition_scale(np, s, b, d_local, composition,
                                                   dd.size, at=jj))
                n_l.append(np.full(dd.size, s.synapses_per_connection.get(key, 1.0)))
                r_l.append(np.full(dd.size, s.gain_ratio(a, b) or 0.0))
        notes.append(f"{name}: {t.n:,} sites x {len(pairs)} class pairs"
                     + ("" if radius_mm <= 0 else
                        f" plus sub-columnar neighbours within {radius_mm} mm"))

    z = np.zeros(0, dtype=np.int64)
    cat = lambda xs, dt=float: (np.concatenate(xs) if xs else np.zeros(0, dtype=dt))
    return B.EdgeSet(
        "microcircuit", cat(src_l, np.int64), cat(dst_l, np.int64), sites.n_total,
        {"distance_mm": cat(dist_l), "connection_probability": cat(p_l),
         "synapses_per_connection": cat(n_l), "coupling_ratio": cat(r_l),
         "pre_class": cat(pc_l, np.int64), "post_class": cat(qc_l, np.int64)},
        directed=True,
        note="; ".join(notes) + f".  classes {list(classes)}; probabilities from "
             f"{s.source}; tier {Tier.GENERATIVE_PRIOR.value} -- the circuit is "
             f"measured and the per-site composition is interpolated")


def _composition_scale(np, s: MicrocircuitStatistics, post: str, depth,
                       composition: Mapping[str, Any] | None, n: int, at=None):
    """how much of the postsynaptic class is present, relative to the census mean.

    connection probability was measured against the cells that WERE there, so it
    is already conditioned on the volume's composition.  a site whose layer holds
    twice the volume-average density of pv cells offers twice as many targets, and
    the coupling a population process sees scales with that -- so the probability
    per target is measured and the number of targets is what depth changes.

    with no composition supplied the scale is 1 everywhere, which is the volume
    average and is the honest default: a cortical depth profile of interneuron
    density interpolated from one mouse volume onto a human column would be three
    guesses stacked, and this returns a flat 1.0 and says so in the note rather
    than doing that quietly.
    """
    if composition is not None and post in composition:
        v = np.asarray(composition[post], dtype=float)
        v = v if at is None else v[at]
        base = s.class_fraction.get(post, 1.0) or 1.0
        return np.broadcast_to(v / base, (n,))
    return np.ones(n)


def describe() -> str:
    """three lines a provenance report can print, and never fewer than three.

    the circuit line alone reads as though a mouse microcircuit were a human one,
    which is what the census line is there to contradict, and the kinetics line is
    the only one of the three whose parameters this measurement can hand a model
    directly.
    """
    return MEASURED.describe() + "\n" + HUMAN.describe() + "\n" + KINETICS.describe()


__all__ = ["Tier", "MicrocircuitStatistics", "HumanCensus", "CellKinetics",
           "ParameterGrounding", "MicrocircuitPriorSet", "MEASURED", "HUMAN",
           "KINETICS", "COMPONENT_OF", "parameter_priors", "laminar_gain_priors",
           "theta_prior", "microcircuit_generative", "describe"]
