"""tractography as a calibrated prior: permissive support, informative theta.

`tract.py` builds a tractometric edge set from a tractogram.  it does not say how
much to believe it, and until now nothing did.  this module is that missing
half, and it exists because the asymmetry between the two halves of a
materialization is total and almost always got backwards.

    T(i, j) = 0  is PERMANENT within a materialization.  no parameter exists for
    that pair, so no amount of evidence can ever restore the edge.  a fit cannot
    discover a connection the topology refused to carry; it will simply explain
    the data with whatever else is available and report a good fit.

    theta_ij = 0 is ORDINARY.  the parameter exists, its prior is centred low,
    and data that wants the coupling can move it -- or leave it where it started,
    which is also a reportable outcome.

so the two decisions have opposite failure modes and must be taken with opposite
temperaments.  **be generous with the support, informative with the prior.**  an
over-included edge costs memory and one parameter and can be driven to zero; an
excluded edge is a hypothesis the model can never entertain.

this is not a licence to carry everything.  it is the reason the threshold on
`edge_support` is deliberately at the *permissive* end -- an edge is carried if
ANY plausible pipeline found it -- while `theta_prior` is where the pipeline
disagreement is spent.

what makes it calibrated rather than asserted
---------------------------------------------
every number below came out of `scripts/measure_tract_uncertainty.py` run against
the ISMRM 2015 phantom and its 96 submitted tractograms, and the module reads
`data/sources/ismrm2015-submissions/evidence/tract_uncertainty@1/measured.json`
when it is there.  the four that matter:

    sensitivity   0.456   a pipeline recovers 46% of the true edges
    specificity   0.836   a pipeline asserts 16% of the pairs that are NOT edges
    strength      x3.9    pipelines disagree about an edge's strength by a factor
                          of about four, one sigma, after normalizing out how many
                          streamlines each emitted
    rho           0.317   the fraction of a pipeline's error variance shared with
                          the other pipelines

the last is the one this repository had been assuming.  it is the same quantity a
source card calls `correlated_fraction` for a teacher, and it converts through
`ibm.runtime.fuse` to an effective number of independent pipelines:

    91 tractograms of one brain  ->  3.1 independent votes
    20 different teams           ->  3.1 independent votes

the two agreeing is the finding.  it means the redundancy is not one team
submitting the same algorithm at twenty settings -- it survives across genuinely
different algorithms, because they share the failure modes the physics gives
them: crossing fibres, gyral bias, a preference for short paths, and a bottleneck
at the same deep white-matter chokepoints.  a consensus over more pipelines does
not converge on the truth; it converges on the shared error.

that is why `existence_probability` below counts effective pipelines and not
pipelines.  ten pipelines agreeing an edge exists is worth about three, and
saying so is the difference between a prior and a vote count.

what this module refuses to do
------------------------------
it does not build a topology of its own.  `tractometric` already exists and this
is a prior *over* it -- registering a second one would mean two graphs claiming
to be the same anatomy, which is exactly what ARCHITECTURE.md §3's "there is no
universal interaction graph" is not an invitation to.

it does not decide coupling strength.  §3 again: a topology is a support for
interaction and strength is process parameters.  `theta_prior` returns
`ibm.vocabulary.Prior` objects for `ibm.forge.priors` to lay out, and a fit is
expected to move them.  a prior that could not be overruled would be worse than
none.

the tiers, and why the field exists
-----------------------------------
`association.py` names three tiers and says a model that falls through should say
so.  `Tier` is where that becomes machine-readable rather than a docstring
promise:

    SUBJECT_TRACTOGRAM   this subject's own diffusion data.  a measurement, with
                         the error rate measured above.
    GROUP_CONNECTOME     a published matrix over somebody else's subjects.  it
                         adds between-subject variance to the pipeline error and
                         removes the subject's own anatomy entirely.
    DISTANCE_PRIOR       no tractography at all -- `cortical_association`'s
                         exp(-d/l).  a guess with a principled shape.

a materialization records which one it actually used, and provenance reports it.
the number that makes this worth carrying: at tier 1 the prior's spread is a
factor of 3.9 and at tier 3 it is whatever `weak()` says, which is a factor of
10.  presenting the third as though it were the first is how a geometric prior
comes to be quoted as anatomy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field as _field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ibm.topologies import builders as B
from ibm.vocabulary import Prior, Provenance, Tying, lognormal

_EVIDENCE = ("data/sources/ismrm2015-submissions/evidence/tract_uncertainty@1/"
             "measured.json")


class Tier(str, Enum):
    """which of the three sources of long-range connectivity a build actually used.

    an ordering, not a taxonomy: each tier is strictly less informative than the
    one above it, and a materialization that fell through to the third has a
    different kind of answer, not a worse-scored version of the same one.
    """

    SUBJECT_TRACTOGRAM = "subject_tractogram"
    GROUP_CONNECTOME = "group_connectome"
    DISTANCE_PRIOR = "distance_prior"

    @property
    def rank(self) -> int:
        return {"subject_tractogram": 0, "group_connectome": 1,
                "distance_prior": 2}[self.value]

    def demoted_to(self, other: "Tier") -> "Tier":
        """the weaker of two tiers.  weakness wins, always.

        used to combine what a caller declared with what the inputs actually
        are.  the asymmetry is the point: a caller may cap a tier downward -- a
        tractogram warped from a template subject is not this subject's anatomy
        however it was computed -- and may never raise one, because an intention
        is not evidence about where the data came from.
        """
        return self if self.rank >= other.rank else other


@dataclass(frozen=True)
class TractUncertainty:
    """what tractography's error actually is, measured rather than assumed.

    the defaults are the figures from the ISMRM 2015 phantom, and they are
    compiled in so that this module works with no data on disk -- but they are
    not literature values or round numbers.  each one is reproduced by
    `scripts/measure_tract_uncertainty.py` and recorded with its provenance in
    `data/sources/ismrm2015-submissions/evidence/tract_uncertainty@1/`.

    read every one of them as a LOWER bound on the error for real diffusion data.
    the phantom is simulated, its crossings are simpler than tissue's, and the
    20 mm spatial bins the measurement used merge nearby endpoints -- all three
    push the measured error down.
    """

    #: P(a pipeline asserts an edge | the edge is real).
    sensitivity: float = 0.456
    #: P(a pipeline asserts an edge | the edge is NOT real), over every bin pair
    #: that could have been asserted.  this is the number that makes tractography
    #: a weak vote: it is only about three times smaller than the sensitivity, so
    #: one pipeline finding an edge is worth a likelihood ratio of 2.8, not 100.
    false_positive_rate: float = 0.164
    #: natural-log sd, across pipelines, of an edge's streamline count normalized
    #: by the pipeline's own total.  exp of it is a factor of 3.9.
    strength_log_sd: float = 1.359
    #: fraction of a pipeline's error variance shared with the others.  the same
    #: quantity a source card calls `correlated_fraction`.
    correlated_fraction: float = 0.317
    #: rank of the shared-error subspace.  1 means "a bias affecting everything",
    #: which is the only value that can be claimed without measuring a correlation
    #: length -- and the discount weakens by orders of magnitude above it, so this
    #: is a claim about structure and not a solver setting.
    error_rank: int = 1
    #: fraction of possible bin pairs that were real edges in the phantom.  used
    #: as the prior odds in `existence_probability`, and the shakiest number here:
    #: it depends on the parcellation and on a phantom with 25 bundles, so it is a
    #: parameter with a default rather than a constant.
    base_rate: float = 0.0875
    source: str = "ISMRM 2015 tractography challenge, 91 of 96 submissions, 20 teams"
    measured_at: str = "2026-09-04"
    #: True when these came off disk, False when they are the compiled-in copy.
    from_evidence: bool = False

    @classmethod
    def load(cls, root: Path | None = None) -> "TractUncertainty":
        """read the measurement if it is on disk, else use the compiled-in copy.

        deliberately not an error when the file is missing.  the figures below
        ARE the file's contents at the time of writing, so a checkout with no
        evidence directory behaves identically -- and `from_evidence` says which
        happened, so a provenance report can distinguish "measured here" from
        "measured once, carried in the source".
        """
        try:
            from ibm.forge.bind import repo_root
            base = root or repo_root(Path(__file__))
            data = json.loads((base / _EVIDENCE).read_text())
        except Exception:
            return cls()
        try:
            ex, fp = data["existence"], data["false_positives"]
            st, ec = data["strength"], data["error_correlation"]
            n_pairs = float(data["counts"]["possible_bin_pairs"])
            return cls(
                sensitivity=float(ex["mean_recovery_of_true_edges"]),
                false_positive_rate=float(fp["per_pipeline_false_positive_rate"]),
                strength_log_sd=float(st["log_sd_across_pipelines"]),
                correlated_fraction=float(ec["correlated_fraction"]),
                error_rank=int(ec.get("error_rank_assumed", 1)),
                base_rate=float(data["counts"]["gt_edges"]) / max(n_pairs, 1.0),
                source=data.get("sources", ["ismrm2015"])[-1],
                measured_at=str(data.get("measured_at", "")),
                from_evidence=True,
            )
        except (KeyError, TypeError, ValueError):
            return cls()

    def effective_pipelines(self, n: int) -> float:
        """`n` pipelines whose errors share a fraction rho are worth how many.

        routed through `ibm.runtime.fuse` rather than reimplemented, because that
        module is where this arithmetic lives and because a second copy of it is
        a second place for the sign to be got wrong -- a correlated source does
        not ADD a low-rank constraint, it SUBTRACTS the confidence correlated
        values appear to supply.

        the ceiling is 1/rho however many pipelines arrive, which is why a
        consensus connectome built from a hundred subjects is not a hundred times
        more certain than one built from three.
        """
        if n <= 0:
            return 0.0
        rho = min(max(self.correlated_fraction, 0.0), 1.0 - 1e-9)
        try:
            import numpy as np

            from ibm.runtime.fuse import TeacherPrecision
            tp = TeacherPrecision(r2=0.0, correlated_fraction=rho,
                                  error_rank=max(self.error_rank, 1), source=self.source)
            ev = tp.evidence("structural.axonal_density", np.zeros(n), np.ones(n))
            return float(ev.effective_constraints())
        except Exception:
            return n / ((1.0 - rho) + n * rho)

    def existence_probability(self, found: Any, n_pipelines: int, np=None) -> Any:
        """P(the edge is real | `found` of `n_pipelines` asserted it).

        an ordinary bayes update with one correction that changes the answer by
        an order of magnitude: the counts are scaled to EFFECTIVE pipelines
        first.  ten agreeing pipelines are about three agreeing pipelines, so the
        likelihood ratio is cubed rather than raised to the tenth, and an edge
        that eight of ten found comes out around 0.8 instead of 0.999.

        the naive version is the failure this whole module is about.  it is what
        produces a group connectome in which every edge is either certain or
        absent, from data whose per-pipeline sensitivity is 46%.
        """
        np = np or B._numpy("tract_prior")
        n = max(int(n_pipelines), 1)
        k = np.asarray(found, dtype=float)
        scale = self.effective_pipelines(n) / n
        k_eff, n_eff = k * scale, n * scale
        se = min(max(self.sensitivity, 1e-6), 1 - 1e-6)
        fp = min(max(self.false_positive_rate, 1e-6), 1 - 1e-6)
        base = min(max(self.base_rate, 1e-9), 1 - 1e-9)
        log_odds = (math.log(base / (1.0 - base))
                    + k_eff * math.log(se / fp)
                    + (n_eff - k_eff) * math.log((1.0 - se) / (1.0 - fp)))
        return 1.0 / (1.0 + np.exp(-log_odds))

    def describe(self) -> str:
        return (f"tract uncertainty [{self.source}, {self.measured_at}, "
                f"{'from evidence' if self.from_evidence else 'compiled in'}]: "
                f"sensitivity {self.sensitivity:.3f}, false-positive rate "
                f"{self.false_positive_rate:.3f}, strength spread x"
                f"{math.exp(self.strength_log_sd):.2f}, rho "
                f"{self.correlated_fraction:.3f} (rank {self.error_rank})")


MEASURED = TractUncertainty.load()


# ---------------------------------------------------------------------------
# support
# ---------------------------------------------------------------------------


@B.builder(
    "tractometric_consensus",
    produces=("tract_length_mm", "conduction_delay_s", "distance_mm", "orientation",
              "pipelines_found", "existence_prob"),
    supports=("tissue",),
    requires=("tractograms",),
    directed=True,
    metric="arc length along the reconstructed fascicle, over several reconstructions",
    doc="union the edges several tractography pipelines found, and score each one")
def tractometric_consensus(sites, *, support: str = "tissue", tractograms=None,
                           min_pipelines: int = 1, snap_mm: float = 3.0,
                           velocity_m_s: float | None = None,
                           min_length_mm: float = 5.0,
                           uncertainty: TractUncertainty | None = None) -> B.EdgeSet:
    """several reconstructions of one brain, unioned permissively and scored.

    `tractograms` is a mapping from a pipeline name to its streamlines -- the
    same input `tractometric_streamlines` takes, once per pipeline.  each is run
    through that builder unchanged, so the geometry, the arc lengths and the
    delays are computed exactly as they are for a single tractogram and nothing
    here reimplements them.

    **the union, not the intersection, and `min_pipelines` defaults to 1.**  this
    is the module's whole argument made operational.  an intersection would drop
    every edge one pipeline missed, and the measurement says a pipeline misses
    54% of the real ones -- so the intersection of three pipelines retains under
    a tenth of the true connectivity and the model can never recover any of it.
    the union costs one parameter per spurious edge and a prior that starts it
    near zero.

    raising `min_pipelines` above 1 is legitimate under a memory budget and it is
    a *support* decision with permanent consequences, so the note records it.
    what it must never be used for is cleaning up false positives; that is what
    `existence_prob` and `theta_prior` are for, and they are reversible.

    `tract_length_mm` is averaged over the pipelines that found the edge rather
    than over all of them, because a pipeline that did not find the edge has no
    opinion about its length -- and averaging in a zero would shorten every
    contested pathway, which is the direction that makes conduction delays too
    small and long-range coupling look faster than it is.
    """
    np = B._numpy("tract_prior")
    unc = uncertainty or MEASURED
    if not tractograms:
        raise B.MissingInput(
            "tractometric_consensus", "tractograms",
            "a mapping {pipeline name: streamlines}, each a sequence of (p, 3) point "
            "arrays in the site frame -- the SAME subject reconstructed several ways, "
            "which is what makes the disagreement measurable",
            "run tractoflow, mrtrix3 tckgen and dsi-studio over one subject's dwi; or "
            "take the several algorithms fiber-data-hub keeps attached to one parent "
            "dataset (data/sources/fiber-data-hub); a single tractogram belongs in "
            "tractometric_streamlines instead, and this builder will accept one and say "
            "in its note that a consensus of one is not a consensus")

    from ibm.topologies.tract import tractometric_streamlines

    names = list(tractograms)
    per: list[B.EdgeSet] = []
    for name in names:
        per.append(tractometric_streamlines(
            sites, support=support, streamlines=tractograms[name], snap_mm=snap_mm,
            velocity_m_s=velocity_m_s, min_length_mm=min_length_mm))

    n_sites = sites.n_total if hasattr(sites, "n_total") else per[0].n_sites
    key_of = lambda e: e.src.astype(np.int64) * np.int64(n_sites) + e.dst.astype(np.int64)
    keys = [key_of(e) for e in per]
    union = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    if union.size == 0:
        return B.empty("tractometric", n_sites,
                       ("tract_length_mm", "conduction_delay_s", "distance_mm",
                        "orientation", "pipelines_found", "existence_prob"),
                       directed=True,
                       note=f"no pipeline of {len(names)} produced an edge")

    found = np.zeros(union.size, dtype=np.int64)
    acc = {"tract_length_mm": np.zeros(union.size),
           "conduction_delay_s": np.zeros(union.size),
           "distance_mm": np.zeros(union.size)}
    orient = np.zeros((union.size, 3))
    for e, k in zip(per, keys):
        pos = np.searchsorted(union, k)
        found[pos] += 1
        for c in acc:
            acc[c][pos] += np.asarray(e.features[c], float)
        orient[pos] += np.asarray(e.features["orientation"], float)

    keep = found >= max(int(min_pipelines), 1)
    union, found = union[keep], found[keep]
    denom = found.astype(float)
    feats = {c: acc[c][keep] / denom for c in acc}
    feats["orientation"] = B.unit(orient[keep], np)
    feats["pipelines_found"] = found
    feats["existence_prob"] = unc.existence_probability(found, len(names), np)

    n_eff = unc.effective_pipelines(len(names))
    return B.EdgeSet(
        "tractometric", (union // n_sites).astype(np.int64),
        (union % n_sites).astype(np.int64), n_sites, feats, directed=True,
        note=(f"UNION of {len(names)} pipelines at min_pipelines={min_pipelines}: "
              f"{int(keep.sum())} edges, {int((found == len(names)).sum())} found by all "
              f"and {int((found == 1).sum())} by exactly one.  those {len(names)} "
              f"pipelines are worth {n_eff:.1f} independent ones "
              f"(rho = {unc.correlated_fraction:.3f}), which is what existence_prob "
              f"counts.  lengths averaged over the pipelines that FOUND each edge.  "
              + ("consensus of one -- existence_prob is a single pipeline's likelihood "
                 "ratio and nothing more.  " if len(names) == 1 else "")
              + f"tier {Tier.SUBJECT_TRACTOGRAM.value}"))


def edge_support(sites, *, tier: Tier | str = Tier.SUBJECT_TRACTOGRAM,
                 tractograms: Mapping[str, Any] | None = None,
                 streamlines=None, matrix=None, lengths_mm=None,
                 support: str = "tissue", min_pipelines: int = 1,
                 uncertainty: TractUncertainty | None = None,
                 **kw) -> tuple[B.EdgeSet, Tier]:
    """the tractometric support, built at whichever tier the inputs allow.

    one entry point for the three tiers, returning the tier alongside the edges,
    so that a materialization cannot obtain the support without also obtaining
    the answer to "where did this come from".  keeping them in separate calls is
    how a distance prior ends up quoted as anatomy: the edges are what gets
    passed around, the provenance is what gets dropped.

    `tier` is a CEILING and not a declaration.  the returned tier is the weaker of
    what the caller asked for and what the inputs actually are, so asking for
    `SUBJECT_TRACTOGRAM` with only a group matrix in hand returns
    `GROUP_CONNECTOME` -- an intention is not evidence about where data came from
    -- while asking for `GROUP_CONNECTOME` with a subject tractogram in hand
    honours the cap, because a caller may know something the inputs do not show.
    a tractogram warped from a template subject is the case that matters: it
    arrives as streamlines and is not this subject's anatomy, and nothing in the
    file would say so.

    the third tier is not implemented here and that is deliberate: it is
    `cortical_association`, it already exists, and it lives on the cortical
    surface support rather than on tissue.  what this function does at that tier
    is refuse, and name the builder to use instead -- a `tractometric` edge set
    fabricated from distance would be indistinguishable downstream from one built
    from a tractogram, which is precisely the confusion the tier field exists to
    prevent.
    """
    unc = uncertainty or MEASURED
    tier = Tier(tier)

    if tractograms:
        return (tractometric_consensus(
            sites, support=support, tractograms=tractograms,
            min_pipelines=min_pipelines, uncertainty=unc, **kw),
            tier.demoted_to(Tier.SUBJECT_TRACTOGRAM))
    if streamlines is not None:
        return (tractometric_consensus(
            sites, support=support, tractograms={"only": streamlines},
            min_pipelines=1, uncertainty=unc, **kw),
            tier.demoted_to(Tier.SUBJECT_TRACTOGRAM))
    if matrix is not None and lengths_mm is not None:
        from ibm.topologies.tract import tractometric_matrix
        edges = tractometric_matrix(sites, support=support, matrix=matrix,
                                    lengths_mm=lengths_mm, **kw)
        # threshold 0 on purpose, and the note says so: a group connectome that
        # has already been thresholded by its publisher has had this decision
        # taken for it once, and taking it a second time here compounds two
        # unrecorded cutoffs into one number nobody can reconstruct.
        return edges, tier.demoted_to(Tier.GROUP_CONNECTOME)

    raise B.MissingInput(
        "edge_support", "tractograms, streamlines, or matrix + lengths_mm",
        "one of: several reconstructions of this subject (tier 1, and the only one "
        "whose spread can be measured on this subject); one reconstruction (tier 1 with "
        "no measurable spread, so the phantom's is used); or a published parcel-by-parcel "
        "connectome with its mean path lengths (tier 2)",
        "for tier 3 -- no tractography at all -- do NOT call this.  use the "
        "`cortical_association` builder in ibm/topologies/association.py, which carries "
        "exp(-d/l) over the cortical surface and is honest about being a geometric prior. "
        "returning tractometric edges built from distance would make a guess "
        "indistinguishable from a measurement everywhere downstream")


# ---------------------------------------------------------------------------
# theta
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TractPriorSet:
    """priors over the coupling on a tractometric edge set, with their provenance.

    `priors` is one `Prior` per entry of `groups` -- per edge when the tying is
    per-site, per group otherwise -- and `tier` travels with them because §7
    requires a prediction resting on prior-dominated structure to be reportable
    as one.  a `TractPriorSet` at `DISTANCE_PRIOR` and one at
    `SUBJECT_TRACTOGRAM` have the same type and must never have the same
    authority.
    """

    tier: Tier
    tying: Tying
    priors: tuple[Prior, ...]
    #: index into `priors` for each edge.  identity when tying is per-site.
    group_of: Any = None
    uncertainty: TractUncertainty = _field(default_factory=lambda: MEASURED)
    note: str = ""

    def __len__(self) -> int:
        return len(self.priors)

    def describe(self) -> str:
        meds = [math.exp(p.loc) for p in self.priors]
        spreads = [math.exp(p.scale) for p in self.priors]
        lo, hi = (min(meds), max(meds)) if meds else (0.0, 0.0)
        return (f"{len(self.priors)} {self.tying.value} priors at tier "
                f"{self.tier.value}, median {lo:.3g}-{hi:.3g}, spread x"
                f"{(sum(spreads) / len(spreads)) if spreads else 0:.2f}"
                + (f"  ({self.note})" if self.note else ""))


def theta_prior(edges: B.EdgeSet, *, tier: Tier | str = Tier.SUBJECT_TRACTOGRAM,
                n_pipelines: int | None = None,
                strengths: Any = None,
                tying: Tying = Tying.PER_SITE,
                groups: Any = None,
                median_strength: float = 1.0,
                uncertainty: TractUncertainty | None = None) -> TractPriorSet:
    """a lognormal prior on coupling strength per edge, widened by the measurement.

    lognormal because the parameter is positive and known to within a
    multiplicative factor, which is the house prior for every rate constant and
    conductance in the inventory -- and here it is not merely conventional: the
    quantity a tractogram reports is a streamline count, its errors are
    multiplicative in the seeding density and the path length, and the measured
    spread across pipelines is a FACTOR (3.9) rather than an increment.

    two things go into each prior and they do different jobs.

    **the median is scaled by the edge's existence probability.**  an edge only
    one pipeline in ten found is carried -- the support is permissive by design
    -- but it starts at a tenth of the coupling of an edge every pipeline found.
    that is what "generous with the support, informative with the prior" means
    arithmetically, and it is the mechanism by which a false positive costs a
    parameter rather than a wrong prediction.  the floor is 1e-3 rather than 0
    because a lognormal at zero has no log.

    **the spread is measured, and it does not shrink the way an average would.**
    the naive figure is the cross-pipeline spread divided by sqrt(n_pipelines).
    that is wrong by construction here: a fraction rho of the pipelines' error is
    shared, so the variance of their consensus is

        v * ((1 - rho) / n + rho)  =  v / n_eff

    which stops falling at v * rho however many pipelines are averaged.  at the
    measured rho = 0.317, 91 pipelines buy a factor of 1.8 off the spread and not
    a factor of 9.5.  a materialization that used the naive form would report a
    connectome five times more certain than the data supports, and would do it
    more confidently the more pipelines it was given.

    `strengths`, when supplied, is a per-edge consensus strength in whatever units
    the process's parameter is in; the default of 1.0 with the existence
    probability doing the work is the right choice when the process has no scale
    of its own yet.  `tying` at anything other than PER_SITE collapses the priors
    over `groups`, which is how a bundle segmentation earns its keep -- one
    parameter per named fascicle rather than per edge -- and the collapse takes
    the geometric mean of the medians and the WIDEST spread in the group, because
    a group is no better known than its least certain member.
    """
    np = B._numpy("tract_prior")
    unc = uncertainty or MEASURED
    tier = Tier(tier)
    m = edges.n_edges

    if "existence_prob" in edges.features:
        p_exist = np.asarray(edges.features["existence_prob"], float)
    elif "pipelines_found" in edges.features and n_pipelines:
        p_exist = unc.existence_probability(edges.features["pipelines_found"],
                                            n_pipelines, np)
    else:
        # no per-edge evidence: every edge gets the probability a single pipeline
        # asserting it implies.  at the measured rates that is 0.21, not 1.0, and
        # the difference between those two numbers is the entire point of the
        # module.
        p_exist = np.full(m, float(unc.existence_probability(np.ones(1), 1, np)[0]))

    if tier is Tier.GROUP_CONNECTOME:
        # a group matrix has already averaged over subjects, so its edges carry
        # BETWEEN-SUBJECT variance on top of the pipeline error -- and it is not
        # this subject's anatomy at all.  the widening is a declared judgement,
        # not a measurement, and it is small on purpose: the honest response to an
        # unmeasured extra error is to say it is unmeasured, and the alternative
        # (leaving it out) would rank a group connectome equal to the subject's
        # own scan.
        extra = math.log(1.5)
    elif tier is Tier.DISTANCE_PRIOR:
        extra = math.log(3.0)
    else:
        extra = 0.0

    n = max(int(n_pipelines or 1), 1)
    n_eff = unc.effective_pipelines(n)
    sd = math.sqrt(unc.strength_log_sd ** 2 / max(n_eff, 1.0) + extra ** 2)
    spread = math.exp(sd)

    base = (np.asarray(strengths, float) if strengths is not None
            else np.full(m, float(median_strength)))
    med = np.maximum(base * p_exist, 1e-3 * max(float(median_strength), 1e-9))

    # FIT rather than LITERATURE at tier 1: the median came from this subject's own
    # tractogram, which is `theta <- fit(D)` in §4's vocabulary and not a number
    # read out of a paper.  ATLAS at tier 2 says the opposite -- somebody else's
    # subjects, averaged -- and WEAK at tier 3 is what a geometric guess is worth.
    prov = (Provenance.FIT if tier is Tier.SUBJECT_TRACTOGRAM
            else Provenance.ATLAS if tier is Tier.GROUP_CONNECTOME
            else Provenance.WEAK)
    src = (f"{unc.source} (calibration); tier {tier.value}")
    note = (f"median = consensus strength x P(edge is real | {n} pipeline(s), "
            f"{n_eff:.1f} effective); spread x{spread:.2f} from a measured "
            f"cross-pipeline log sd of {unc.strength_log_sd:.3f} reduced by "
            f"sqrt(n_eff) and NOT by sqrt(n)")

    if tying is Tying.PER_SITE or groups is None:
        priors = tuple(lognormal(float(x), spread, units="dimensionless",
                                 provenance=prov, source=src, note=note) for x in med)
        return TractPriorSet(tier, Tying.PER_SITE, priors, None, unc, note)

    g = np.asarray(groups).astype(np.int64)
    k = int(g.max()) + 1 if g.size else 0
    priors = []
    for j in range(k):
        sel = g == j
        # geometric mean of the medians: these are multiplicative quantities and
        # an arithmetic mean would be dominated by whichever edge one pipeline
        # over-seeded.
        mu = float(np.exp(np.mean(np.log(med[sel])))) if sel.any() else 1e-3
        priors.append(lognormal(mu, spread, units="dimensionless", provenance=prov,
                                source=src, note=note + f"; tied over {int(sel.sum())} edges"))
    return TractPriorSet(tier, tying, tuple(priors), g, unc,
                         note + f"; {k} tying groups over {m} edges")


def describe() -> str:
    """one line a provenance report can print without importing anything else."""
    return MEASURED.describe()


__all__ = ["Tier", "TractUncertainty", "TractPriorSet", "MEASURED",
           "tractometric_consensus", "edge_support", "theta_prior", "describe"]
