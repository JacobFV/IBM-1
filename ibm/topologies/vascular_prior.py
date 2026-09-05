"""microvasculature as a three-tier prior: measurable tree, measured statistics,
synthesised bed.

`vascular.py` builds a tree from a parent array and a radius array and says
nothing about where either came from.  for the macrovasculature that is fine --
an angiogram is a measurement and the builder is right to just take it.  for
everything below about two hundred microns it is not fine at all, because THE
INPUT DOES NOT EXIST AND CANNOT BE MADE TO EXIST, and a module that quietly
accepts a capillary bed from a caller has no way to say whether it was measured,
warped from an atlas, or invented.

the asymmetry is not the one `tract_prior` faces and pretending otherwise would
be the mistake here.  tractography at least attempts the thing it reports; the
question there is how wrong it is.  in vivo human angiography does not attempt
the capillary bed at all:

    7T TOF-MRA and QSM venography resolve to roughly 200-300 um.
    a cortical capillary is 4-5 um across at 40-60 um spacing.

measured on the cortical blocks in `data/sources/microscopy-microvascular-
networks`, everything at or above 30 um diameter -- an order of magnitude finer
than any human angiogram -- is 0.5% of the vascular LENGTH and 19-25% of the
vascular VOLUME.  above 100 um there is nothing at all inside a cubic
millimetre of cortex.  so an angiogram sees a fifth of the blood and none of the
plumbing, and the fifth it sees is the part that does no exchanging.

**every tier of this module therefore synthesises the microvasculature.**  the
tier says only how much of the BOUNDARY CONDITION -- the macro tree the
synthesis hangs off -- is this subject's own.  saying that plainly is the point
of the type, because a materialization that carried a synthesised bed under a
tier-1 label would be indistinguishable downstream from one that had measured it,
and nothing measures it.

    SUBJECT_ANGIOGRAM     this subject's TOF-MRA or QSM venogram supplies the
                          vessels above ~300 um.  the bed below is synthesised.
    POPULATION_ATLAS      somebody else's macro tree, warped.  VENAT puts the
                          between-subject spread of a vein's calibre at a factor
                          of 1.33, and that is a LOWER bound.
    GENERATIVE_SYNTHESIS  no angiogram at all.  penetrating vessels are placed at
                          the measured spacing and everything is synthesised.

what is measured, and by whom
-----------------------------
`scripts/measure_microvasculature.py`, against three sources, writing
`data/sources/microscopy-microvascular-networks/evidence/microvasculature@1/
measured.json`.  the numbers this module compiles in ARE that file's contents.

    capillary length density        664 mm/mm^3 in mouse cortex, x1.09 between
                                    animals, varying 2.6-fold over cortical depth
                                    with a peak in the middle third
    tissue-to-capillary distance    27 um mean, 40 um at p90 -- and the naive
                                    1/sqrt(density) estimate is 39 um, which
                                    overstates the mean by 1.5x
    capillary surface density       8.8 mm^2/mm^3, which is the number
                                    `metabolic.py` quotes from the literature as
                                    "roughly 5-10" and is now measured
    capillary radius                2.06 um, x1.03 between animals
    segment length                  32 um whole-brain, 56 um in the cortical
                                    blocks -- a skeletonisation difference, not
                                    an anatomical one -- and x2.2 WITHIN an animal
    tortuosity                      1.17 arc over chord, p90 1.57
    Murray exponent                 2.8 at bifurcations where a parent exists,
                                    and NO exponent at all at half of them
    penetrating vessel spacing      ~129 um between nearest neighbours
    capillary transit time          0.19-0.41 s
    CBF the networks deliver        93-173 ml/100g/min, mean 120
    nine mice are worth             4.1 independent animals, ceiling 6.6

what does NOT generalise, stated before anything is built on what does
---------------------------------------------------------------------
**capillary bed STATISTICS generalise.**  they are conserved across cortical
areas, and across species once normalised by metabolic rate, because they are
set by a diffusion problem that is the same everywhere: oxygen has to get from a
capillary wall to a mitochondrion in a few tens of microns, and that fixes the
intercapillary distance up to the local CMRO2.  the depth profile, the radius
distribution, the branching angle and the tortuosity are all in that category.

**individual branching TOPOLOGY does not generalise, and mostly does not need
to.**  which capillary is where in a particular person is not recoverable and
never will be from a living head.  everything ibm-1 computes from vasculature --
BOLD contrast, oxygen delivery, thermal transport -- is a functional of
TRANSPORT STATISTICS: surface area per unit volume, the distribution of
diffusion distances, the distribution of transit times.  two networks with the
same statistics and completely different wiring give the same answer for all
three, which is why a synthesised bed is a legitimate substrate for them.

**it fails exactly where individual variation IS the signal.**  four cases, and
they are not edge cases, they are most of clinical cerebrovascular medicine:

    watershed zones          the boundary between two arterial territories is
                             where the individual's Circle of Willis variant
                             decides the answer, and a population atlas has
                             averaged that variant away
    stroke penumbra          the tissue at risk is defined by WHICH collateral
                             happens to exist, which is the one fact a
                             statistical bed cannot carry
    tumour neovasculature    its whole pathology is that its statistics are not
                             the normal ones -- chaotic, leaky, shunting
    aging microangiopathy    rarefaction is a change in the density this module
                             takes as given, and a synthesis conditioned on the
                             healthy density will reproduce the healthy density

a materialization that reaches any of those must say so, and `Tier` plus the
`assumptions` field of a synthesis report is how it says so mechanically rather
than in a comment.

there is a fifth failure and it is the generator's own rather than the argument's.
the synthesised bed reproduces the measured length density, the measured
tissue-to-capillary distance, the measured transit time and -- to within 5% and
without being fitted to it -- the measured arteriole-to-venule pressure drop.  what
it does NOT reproduce is the branching asymmetry: at its degree-three junctions
sum(r_child^3)/r_parent^3 comes out at 1.0 where the real bed is at 1.5, because
the capillary radii are independent draws from a symmetric distribution and three
exchangeable branches have no reason to be asymmetric.  a real bed expands its
total cross-section going downstream and the synthesis does not, so its flow is
more evenly distributed than a real bed's and its transit-time distribution is too
narrow.  since transit heterogeneity is exactly what raises effective oxygen
extraction above what the mean transit predicts, this bed is usable for bulk
delivery and BOLD and is NOT usable for a question about extraction.  the report
says so in its `assumptions` whenever the realised ratio drifts.

why the synthesis is not texture
--------------------------------
because the network has to carry the flow.  a bed sampled to match a length
density and a radius distribution will match them and be useless: it can have
the right statistics and still be unable to deliver the measured CBF at any
pressure the circulation can supply, or deliver it at a transit time too short
for oxygen to leave the blood.

so `synthesise` builds the graph, assigns radii through Murray's law at the
MEASURED exponent, solves the Poiseuille network with the MEASURED viscosity-
versus-calibre relation, and then reports the arteriole-to-venule pressure drop
the target CBF actually requires.  that number is a falsification: the measured
drop across a real cortical network is 44 mmHg, and a synthesis that needs 400
is wrong however good its histogram looks.  the same solve gives a transit time
and a Krogh radius, and both are checked against measured values rather than
against nothing.

what this module refuses to do
------------------------------
it registers no topology.  `vascular` already exists and this is a prior over
it, for the same reason `tract_prior` registers none over `tractometric`: two
graphs claiming to be the same anatomy is exactly what ARCHITECTURE.md §3's
"there is no universal interaction graph" is not an invitation to.

it does not decide coupling strength.  `theta_prior` returns `ibm.vocabulary.Prior`
objects over hydraulic conductance and expects a fit to move them.  the one thing
it does insist on is the exponent: conductance goes as r^4, so a calibre known to
a factor of 1.33 is a conductance known to a factor of 3.1, and a prior that did
not carry that fourth power would be four times too confident in log units.
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

_EVIDENCE = ("data/sources/microscopy-microvascular-networks/evidence/"
             "microvasculature@1/measured.json")

#: mmHg -> Pa.
MMHG_PA = 133.322387415


class Tier(str, Enum):
    """how much of the vascular BOUNDARY CONDITION is this subject's.

    an ordering, and -- unlike `tract_prior.Tier` -- not an ordering of how much
    of the anatomy was measured.  none of the microvasculature is measured at any
    tier.  what changes is the tree the synthesis is hung off, and therefore
    which questions the result may be asked.
    """

    SUBJECT_ANGIOGRAM = "subject_angiogram"
    POPULATION_ATLAS = "population_atlas"
    GENERATIVE_SYNTHESIS = "generative_synthesis"

    @property
    def rank(self) -> int:
        return {"subject_angiogram": 0, "population_atlas": 1,
                "generative_synthesis": 2}[self.value]

    def demoted_to(self, other: "Tier") -> "Tier":
        """the weaker of two tiers.  weakness wins, always.

        the same contract `tract_prior.Tier` has and for the same reason: a
        caller may cap a tier downward, because they may know something the
        inputs do not show -- a TOF-MRA warped from a template is not this
        subject's tree however it arrived -- and may never raise one, because an
        intention is not evidence about provenance.
        """
        return self if self.rank >= other.rank else other


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrovascularStatistics:
    """what a capillary bed measurably is, in the units a generator needs.

    every default is reproduced by `scripts/measure_microvasculature.py` and
    recorded with its provenance in
    `data/sources/microscopy-microvascular-networks/evidence/microvasculature@1/`.
    the compiled-in copy IS that file, so a checkout with no evidence directory
    behaves identically and `from_evidence` says which happened.

    read every length as an EX-VIVO one.  the tissue was fixed, which collapses
    the lumen, and cleared, which shrinks the block by a factor neither release
    reports per specimen.  both effects make vessels narrower and closer
    together, so the densities are upper bounds and the calibres lower bounds --
    while the segmentation's false negatives, which are worst on the thinnest and
    most numerous vessels, push the density the other way.  the two are not known
    to cancel and no correction is applied, because applying one would mean
    inventing the factor.

    and read every one of them as a MOUSE.  the shape of these distributions
    transfers; the absolute density does not, because capillary density tracks
    metabolic rate and a mouse's is several times a human's per gram.  that is
    why `synthesise` is conditioned on a target CBF and not on a target density.
    """

    # -- the capillary bed, from mouse cortex with a flow solution ------------
    #: mm of capillary per mm^3 of tissue, mean over three cortical blocks.
    capillary_length_density: float = 664.264464
    #: between-ANIMAL log sd of the above.  small, which is the good news.
    capillary_length_density_log_sd: float = 0.084308
    #: log sd of the density across cortical DEPTH within one animal.  three
    #: times the between-animal spread: where in the ribbon you are matters more
    #: than which animal you are in, which is why the generator conditions on
    #: depth and does not condition on subject.
    depth_log_sd: float = 0.263656
    #: max/min of the density over depth.  the peak is in the middle third.
    depth_max_over_min: float = 2.566539
    #: mm^2 of capillary wall per mm^3 of tissue.  the quantity
    #: `capillary_tissue_exchange` carries per edge as `contact_area_mm2`, and the
    #: number `metabolic.py` quotes from the literature as "roughly 5-10".
    capillary_surface_density: float = 8.786732
    #: fraction of tissue volume inside a capillary lumen.
    capillary_volume_fraction: float = 0.010459
    #: mm, mean distance from a random tissue point to the nearest capillary
    #: CENTRELINE.  `capillary_tissue_exchange` defaults `reach_mm` to 0.05,
    #: which the p90 says is about right and the mean says is generous.
    tissue_to_capillary_mm: float = 0.02688
    tissue_to_capillary_p90_mm: float = 0.039579
    #: the 1/sqrt(density) estimate of the same quantity.  it is 1.8x too large,
    #: which is worth carrying because it is the estimate everyone reaches for.
    tissue_to_capillary_sqrt_estimate_mm: float = 0.039098
    #: mm.  a capillary radius, not a diameter.
    capillary_radius_mm: float = 0.00206
    capillary_radius_log_sd: float = 0.203676
    capillary_radius_between_animal_log_sd: float = 0.042819
    #: mm.  segment length between branch points in the capillary bed.
    segment_length_mm: float = 0.055876
    segment_length_log_sd: float = 0.797378
    #: arc length over chord.  1.0 would be a straight tube.
    tortuosity: float = 1.166891
    tortuosity_log_sd: float = 0.214336
    #: mm between a random tissue point and the nearest penetrating vessel.
    #: roughly half the spacing of the penetrating vessels themselves.
    penetrating_distance_mm: float = 0.064509

    # -- branching -----------------------------------------------------------
    #: exponent k fitted per bifurcation from r_p^k = sum r_c^k, at junctions
    #: where the parent is known from the solved pressure field.  Murray says 3.
    murray_exponent: float = 2.802162
    #: sd of that exponent ACROSS junctions.  seven.  the median is near 3 and
    #: essentially nothing else about the distribution is.
    murray_exponent_sd: float = 7.314748
    #: sum(r_c^3)/r_p^3 at the median bifurcation.  Murray says 1.
    murray_ratio: float = 1.495785
    murray_ratio_log_sd: float = 0.985307
    #: fraction of degree-3 junctions at which a "child" is wider than its
    #: "parent", so no exponent exists at all.  half of them.
    murray_no_root_fraction: float = 0.499696
    #: degrees between the two daughters at a bifurcation.
    branch_angle_deg: float = 104.260583
    branch_angle_p10_deg: float = 76.056193
    branch_angle_p90_deg: float = 137.967062

    # -- the physics ---------------------------------------------------------
    #: ml/100g/min delivered by the measured networks through their descending
    #: arterioles.  the target a synthesis has to hit.
    cbf_ml_100g_min: float = 120.193187
    cbf_log_sd: float = 0.354886
    #: mmHg from the top of a descending arteriole to the ascending venule.
    arteriole_to_venule_drop_mmHg: float = 44.312969
    #: s.  capillary blood volume over arteriolar inflow.
    transit_time_s: float = 0.414559
    #: apparent viscosity in mPa s, by vessel diameter in um.  the
    #: Fahraeus-Lindqvist relation `vascular_tree_adjacency` declines to apply and
    #: documents the direction of; the direction is confirmed and the minimum is
    #: at 6-9 um rather than at the capillary, which is the FL inversion.
    viscosity_mpa_s: tuple[tuple[float, float], ...] = ((4.0, 2.412), (6.0, 1.744), (9.0, 1.467), (15.0, 1.523), (30.0, 1.81), (1000000.0, 2.157))
    #: capillary bed is a FIFTH of it in the network this is loaded from, not
    #: most of it, and it runs 0.06 to 0.27 across the three -- see `describe`.
    capillary_dissipation_share: float = 0.210858

    # -- what an angiogram could ever see ------------------------------------
    #: fraction of vascular LENGTH in vessels at or above 30 um diameter, in
    #: cortex.  an order of magnitude finer than any human angiogram.
    resolvable_length_fraction_30um: float = 0.005002
    #: the same, by VOLUME.  the gap between these two numbers is the whole
    #: reason a venogram is not a vasculature.
    resolvable_volume_fraction_30um: float = 0.221433

    # -- how many animals an animal is ---------------------------------------
    n_animals: int = 9
    #: fraction of the between-animal variance in the geometric feature vector
    #: that is shared.  measured as a strain-level ICC over three strains, which
    #: is the estimator with dynamic range at this n -- a deviation-from-consensus
    #: correlation cannot see a bias every animal shares and at n = 3 has almost
    #: no range at all.
    correlated_fraction: float = 0.150665
    error_rank: int = 1
    source: str = ("VesselGraph (9 whole mouse brains, 3 strains) and the "
                   "Blinder/Weber cortical networks with flow (3 mice)")
    measured_at: str = "2026-09-05"
    from_evidence: bool = False

    # -- loading -------------------------------------------------------------

    @classmethod
    def load(cls, root: Path | None = None) -> "MicrovascularStatistics":
        """read the measurement if it is on disk, else use the compiled-in copy.

        deliberately not an error when the file is missing, exactly as
        `TractUncertainty.load` is not: the defaults ARE the file's contents at
        the time of writing, so a checkout without the evidence directory
        behaves identically, and `from_evidence` is what lets a provenance report
        tell "measured here" from "measured once, carried in the source".
        """
        try:
            from ibm.forge.bind import repo_root
            base = root or repo_root(Path(__file__))
            data = json.loads((base / _EVIDENCE).read_text())
        except Exception:
            return cls()
        try:
            arms = data["arms"]
            flow = arms["cortical_networks_with_flow"]
            ba = flow["between_animal"]
            one = next(iter(flow["per_animal"].values()))
            graphs = arms.get("whole_brain_graphs", {})
            eff = graphs.get("effective_animals") or flow.get("effective_animals") or {}
            rho = float(eff.get("correlated_fraction_used",
                                eff.get("correlated_fraction", 0.617)))
            visc = tuple((float(b["diameter_um"][1]), float(b["apparent_viscosity_mPa_s"]))
                         for b in one["physics"]["fahraeus_lindqvist"]["by_diameter"])
            res30 = one["resolvable_fraction_above_diameter"]["30um"]
            return cls(
                capillary_length_density=float(
                    ba["capillary_length_density_mm_per_mm3"]["mean"]),
                capillary_length_density_log_sd=float(
                    ba["capillary_length_density_mm_per_mm3"]["between_animal_log_sd"]),
                depth_log_sd=float(one["depth_variation"]["log_sd_over_depth"]),
                depth_max_over_min=float(one["depth_variation"]["max_over_min"]),
                capillary_surface_density=float(
                    ba["capillary_surface_density_mm2_per_mm3"]["mean"]),
                capillary_volume_fraction=float(one["capillary_volume_fraction"]),
                tissue_to_capillary_mm=float(
                    ba["tissue_to_capillary_um_mean"]["mean"]) * 1e-3,
                tissue_to_capillary_p90_mm=float(
                    one["tissue_to_capillary_um"]["p90"]) * 1e-3,
                tissue_to_capillary_sqrt_estimate_mm=float(
                    one["tissue_to_capillary_um"]["sqrt_density_estimate"]) * 1e-3,
                capillary_radius_mm=float(
                    ba["capillary_radius_um_median"]["mean"]) * 1e-3,
                capillary_radius_log_sd=float(
                    one["by_kind"]["capillary"]["radius_um"]["log_sd"]),
                capillary_radius_between_animal_log_sd=float(
                    ba["capillary_radius_um_median"]["between_animal_log_sd"]),
                segment_length_mm=float(
                    ba["capillary_segment_length_um_median"]["mean"]) * 1e-3,
                segment_length_log_sd=float(
                    one["by_kind"]["capillary"]["length_um"]["log_sd"]),
                tortuosity=float(one["tortuosity_arc_over_chord"]["capillary"]["median"]),
                tortuosity_log_sd=float(
                    one["tortuosity_arc_over_chord"]["capillary"]["log_sd"]),
                penetrating_distance_mm=float(
                    one["penetrating_vessel_spacing_um"]["mean_distance_to_nearest"]) * 1e-3,
                murray_exponent=float(one["murray"]["exponent"]["median"]),
                murray_exponent_sd=float(one["murray"]["exponent"]["sd"]),
                murray_ratio=float(one["murray"]["sum_rc3_over_rp3"]["median"]),
                murray_ratio_log_sd=float(one["murray"]["sum_rc3_over_rp3"]["log_sd"]),
                murray_no_root_fraction=float(
                    one["murray"]["n_child_wider_than_parent"]) / max(
                        float(one["murray"]["n_bifurcations"]), 1.0),
                branch_angle_deg=float(one["murray"]["angle_between_children_deg"]["median"]),
                branch_angle_p10_deg=float(
                    one["murray"]["angle_between_children_deg"]["p10"]),
                branch_angle_p90_deg=float(
                    one["murray"]["angle_between_children_deg"]["p90"]),
                cbf_ml_100g_min=float(ba["cbf_ml_per_100g_per_min"]["mean"]),
                cbf_log_sd=float(ba["cbf_ml_per_100g_per_min"]["between_animal_log_sd"]),
                arteriole_to_venule_drop_mmHg=float(
                    one["physics"]["pressure_mmHg"]["arteriole_to_venule_drop"]),
                transit_time_s=float(one["physics"]["capillary_transit_time_s"]["value"]),
                viscosity_mpa_s=visc,
                capillary_dissipation_share=float(
                    one["physics"]["dissipation_share"]["capillary"]),
                resolvable_length_fraction_30um=float(res30["length"]),
                resolvable_volume_fraction_30um=float(res30["volume"]),
                n_animals=int(graphs.get("between_animal", {})
                              .get("radius_um_median", {}).get("n_animals", 9)),
                correlated_fraction=rho,
                source=str(data.get("sources", ["microvasculature"])[0]),
                measured_at=str(data.get("measured_at", "")),
                from_evidence=True,
            )
        except (KeyError, TypeError, ValueError, StopIteration):
            return cls()

    # -- derived -------------------------------------------------------------

    def viscosity_pa_s(self, diameter_mm: Any, np=None) -> Any:
        """apparent whole-blood viscosity at a given calibre, from the measurement.

        a step function over the measured diameter bins rather than an
        interpolation, because the measurement is a set of bin medians and an
        interpolation between them would invent a smoothness that was not
        measured.  the shape is the Fahraeus-Lindqvist one INCLUDING its
        inversion: viscosity falls from 2.4 mPa s at 4 um to a minimum of 1.5
        around 7 um and rises again above 15, so a model that applies a monotone
        "capillaries are thinner therefore less viscous" correction has the sign
        right over half the range and wrong over the other half.

        this is recovered from a solved pressure field, so it is the viscosity
        law the solver used rather than an independent measurement of blood.
        what it is good for is saying what a network needs in order to carry its
        flow -- which is exactly what the synthesis needs it for -- and NOT for
        claiming a rheological fact.
        """
        np = np or B._numpy("vascular_prior")
        d_um = np.asarray(diameter_mm, float) * 1e3
        out = np.full(d_um.shape, self.viscosity_mpa_s[-1][1])
        for hi, mu in reversed(self.viscosity_mpa_s):
            out = np.where(d_um < hi, mu, out)
        return out * 1e-3

    def effective_animals(self, n: int) -> float:
        """`n` animals whose errors share a fraction rho are worth how many.

        routed through `ibm.runtime.fuse` rather than reimplemented, for the same
        reason `TractUncertainty.effective_pipelines` is: a correlated source
        SUBTRACTS confidence rather than adding a constraint, and a second copy
        of that arithmetic is a second place for the sign to be got wrong.

        the number this returns is one of the findings of the measurement.  at
        the measured rho of 0.151 -- a strain-level ICC over three strains of
        three animals each -- nine mice are worth 4.1 and the ceiling is 6.6
        however many arrive.  the ceiling is the part that matters: a strain is
        a shared genome and a batch is a shared protocol, and neither divides out
        by adding animals to it.

        it is a LOWER bound on the shared error, twice.  the ICC sees only what
        differs BETWEEN strains, so any bias all nine share -- the clearing
        shrinkage, the segmentation's size-dependent false negatives -- is
        invisible to it; and the alternative estimator, the correlation of each
        animal's deviation from the group consensus, is worse still, because
        deviations from a sample mean are linearly dependent by construction and
        an exchangeable shared component is removed by the centring exactly.
        that estimator returns 0.040 here against a null of -0.125, and the
        measurement script reports both and says which to read.
        """
        if n <= 0:
            return 0.0
        rho = min(max(self.correlated_fraction, 0.0), 1.0 - 1e-9)
        try:
            import numpy as np

            from ibm.runtime.fuse import TeacherPrecision
            tp = TeacherPrecision(r2=0.0, correlated_fraction=rho,
                                  error_rank=max(self.error_rank, 1), source=self.source)
            ev = tp.evidence("structural.capillary_density", np.zeros(n), np.ones(n))
            return float(ev.effective_constraints())
        except Exception:
            return n / ((1.0 - rho) + n * rho)

    def density_at_depth(self, depth_fraction: Any, np=None) -> Any:
        """capillary length density as a function of normalized cortical depth.

        the measured profile is not monotone and not flat: it rises from the pial
        surface to a maximum around 40% of the way through the ribbon -- which in
        a mouse is layer 4 -- and then falls by a factor of 2.6 to the white
        matter boundary.  a generator that ignored it would put the same bed under
        layer 1 and layer 6 and be wrong by a factor of two at both ends, which
        for a laminar BOLD model is the entire effect being modelled.

        a smooth two-parameter shape fitted to the measured bins rather than the
        bins themselves, because the bins are one animal's binning and the shape
        is what transfers.  the peak position and the depth range are measured;
        the functional form is a choice and is declared as one.
        """
        np = np or B._numpy("vascular_prior")
        d = np.clip(np.asarray(depth_fraction, float), 0.0, 1.0)
        peak, width = 0.40, 0.45
        shape = np.exp(-0.5 * ((d - peak) / width) ** 2)
        # normalize so the depth-average is the measured mean density
        grid = np.linspace(0.0, 1.0, 101)
        norm = float(np.mean(np.exp(-0.5 * ((grid - peak) / width) ** 2)))
        return self.capillary_length_density * shape / max(norm, 1e-9)

    def describe(self) -> str:
        return (
            f"microvasculature [{self.source}, {self.measured_at}, "
            f"{'from evidence' if self.from_evidence else 'compiled in'}]: "
            f"capillary length density {self.capillary_length_density:.0f} mm/mm^3 "
            f"(x{math.exp(self.capillary_length_density_log_sd):.2f} between animals, "
            f"{self.depth_max_over_min:.1f}-fold over cortical depth), surface "
            f"{self.capillary_surface_density:.1f} mm^2/mm^3, tissue-to-capillary "
            f"{self.tissue_to_capillary_mm * 1e3:.0f} um mean / "
            f"{self.tissue_to_capillary_p90_mm * 1e3:.0f} um p90; Murray exponent "
            f"{self.murray_exponent:.2f} (sd {self.murray_exponent_sd:.1f}, and "
            f"{self.murray_no_root_fraction:.0%} of junctions have no exponent at all); "
            f"CBF {self.cbf_ml_100g_min:.0f} ml/100g/min at a "
            f"{self.arteriole_to_venule_drop_mmHg:.0f} mmHg drop, transit "
            f"{self.transit_time_s:.2f} s; {self.n_animals} animals are worth "
            f"{self.effective_animals(self.n_animals):.1f} "
            f"(rho = {self.correlated_fraction:.2f})")


MEASURED = MicrovascularStatistics.load()


@dataclass(frozen=True)
class AngiogramUncertainty:
    """what the MACRO tree costs at each of the first two tiers.

    a separate type from `MicrovascularStatistics` and deliberately so: that one
    is about a capillary bed nobody can image, this one is about the handful of
    vessels everybody can, and the two errors are unrelated.  mixing them would
    let an excellent angiogram lend its confidence to a synthesised bed, which is
    the exact confusion the tier field exists to prevent.

    the numbers come from VENAT -- a 7T QSM venous atlas over 20 subjects at five
    measurements each, 0.5 mm isotropic in MNI, measured by
    `scripts/measure_microvasculature.py` arm C.  it is the only human source in
    this module.
    """

    #: mm.  the finest calibre a 7T angiogram or venogram resolves.  not a
    #: measured quantity here -- it is the acquisition's own resolution -- and it
    #: is carried because everything else in this module is conditional on it.
    resolution_mm: float = 0.3
    #: median calibre on the VENAT skeleton.  small, because the atlas skeleton
    #: is mostly cortical veins rather than sinuses.
    atlas_diameter_mm: float = 0.730248
    #: between-SUBJECT coefficient of variation of that calibre, per voxel.
    between_subject_cv_of_diameter: float = 0.288925
    #: the same as a log sd.  exp of it is a factor of 1.33, and because
    #: conductance goes as r^4 that is a factor of 3.1 in what a segment can
    #: carry.
    between_subject_log_sd_of_diameter: float = 0.283153
    #: between-subject CV of the venous partial volume, which is what a BOLD
    #: model integrates.  twice the calibre's, because it is a volume.
    between_subject_cv_of_volume: float = 0.545834
    n_subjects: int = 20
    source: str = "VENAT 7T QSM venous atlas (figshare 7205960), 20 subjects x 5"
    measured_at: str = "2026-09-05"
    from_evidence: bool = False

    @classmethod
    def load(cls, root: Path | None = None) -> "AngiogramUncertainty":
        try:
            from ibm.forge.bind import repo_root
            base = root or repo_root(Path(__file__))
            data = json.loads((base / _EVIDENCE).read_text())
            v = data["arms"]["human_population_venogram"]
        except Exception:
            return cls()
        try:
            return cls(
                atlas_diameter_mm=float(v["diameter_mm"]["median"]),
                between_subject_cv_of_diameter=float(
                    v["between_subject_cv_of_diameter"]["median"]),
                between_subject_log_sd_of_diameter=float(
                    v["between_subject_log_sd_of_diameter"]),
                between_subject_cv_of_volume=float(
                    v["venous_volume_fraction"]["between_subject_cv_of_pv"]),
                measured_at=str(data.get("measured_at", "")),
                from_evidence=True,
            )
        except (KeyError, TypeError, ValueError):
            return cls()

    def extra_log_sd(self, tier: "Tier") -> float:
        """the widening a segment's conductance prior deserves at this tier.

        **the fourth power is the whole content of this method.**  Poiseuille
        makes conductance proportional to r^4, so a calibre uncertainty of sigma
        in log units is a conductance uncertainty of 4 sigma.  at tier 2 the
        measured calibre spread is 0.283, which is a factor of 1.33 on the radius
        and a factor of 3.1 on what the vessel can carry.  a prior that carried
        the calibre's spread instead would be four times too tight in log units,
        and would be so in the direction that makes a fit believe a warped atlas
        knows this subject's perfusion.

        at tier 1 the subject's own angiogram supplies the calibre, so the
        between-subject term drops out and what is left is the acquisition's
        own: a partial-volume-limited diameter estimate at a resolution
        comparable to the vessel.  that residual is NOT measured here -- VENAT
        ships no test-retest -- and 0.15 in log units is a declared judgement,
        marked as one, chosen as roughly half the between-subject figure on the
        grounds that a within-subject error should be smaller than a
        between-subject one and no better argument is available.

        at tier 3 there is no angiogram, the penetrating vessels are placed at
        the measured spacing, and the calibre of any individual one is a draw
        rather than a measurement.  the spread is then the full between-animal
        radius spread plus the population calibre spread, added in quadrature.
        """
        if tier is Tier.SUBJECT_ANGIOGRAM:
            calibre = 0.15                     # DECLARED, not measured.  see above.
        elif tier is Tier.POPULATION_ATLAS:
            calibre = self.between_subject_log_sd_of_diameter
        else:
            calibre = math.sqrt(self.between_subject_log_sd_of_diameter ** 2
                                + MEASURED.capillary_radius_log_sd ** 2)
        return 4.0 * calibre

    def describe(self) -> str:
        return (f"macro tree [{self.source}, {self.measured_at}, "
                f"{'from evidence' if self.from_evidence else 'compiled in'}]: "
                f"resolves to {self.resolution_mm * 1e3:.0f} um, which in cortex is "
                f"{MEASURED.resolvable_length_fraction_30um:.1%} of the vascular length "
                f"and {MEASURED.resolvable_volume_fraction_30um:.0%} of its volume even "
                f"at ten times that resolution; between-subject calibre spread x"
                f"{math.exp(self.between_subject_log_sd_of_diameter):.2f}, which is x"
                f"{math.exp(4 * self.between_subject_log_sd_of_diameter):.1f} in "
                f"conductance")


ANGIOGRAM = AngiogramUncertainty.load()


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------


@dataclass
class SynthesisReport:
    """everything a provenance report needs to say about a synthesised bed.

    it is returned alongside the network rather than logged, because a caller who
    can obtain the graph without obtaining this can present a synthesis as an
    anatomy -- which is the failure the whole module is about.  `feasible` is the
    single bit a materialization must check; `assumptions` is what it must
    reproduce when it reports a result.
    """

    tier: Tier
    n_nodes: int
    n_segments: int
    tissue_volume_mm3: float
    #: mm/mm^3 the realised network actually has, against what was asked for.
    length_density: float
    length_density_target: float
    #: ml/100g/min the network delivers at `pressure_drop_mmHg`.  equal to
    #: `cbf_target` by construction, because the solve is linear and the pressure
    #: drop is chosen to hit the target rather than searched for -- so the
    #: informative number is the DROP it took, not this.
    cbf_delivered: float
    cbf_target: float
    #: the arteriole-to-venule drop the target CBF requires of THIS network.
    #: the falsification: the measured drop across a real cortical network is
    #: about 44 mmHg, and a synthesis needing several hundred is wrong however
    #: well its histograms match.
    pressure_drop_mmHg: float
    #: capillary blood volume over delivered flow.
    transit_time_s: float
    #: mean and p90 distance from tissue to the nearest capillary, realised.
    tissue_to_capillary_mm: float
    tissue_to_capillary_p90_mm: float
    #: the Krogh radius the target CMRO2 allows.  a bed whose p90 exceeds it has
    #: tissue the network cannot oxygenate however much blood it moves.
    krogh_radius_mm: float
    #: realised median sum(r_child^3)/r_parent^3 at the bed's own degree-three
    #: junctions, with the widest branch taken as the parent.  the generator
    #: imposes Murray when it sizes the penetrating vessels, so this is a check
    #: that it did rather than an independent measurement -- but the capillary
    #: radii are independent draws and nothing forces those to obey it, so a bed
    #: whose ratio drifts far from the measured 1.5 has a calibre distribution the
    #: generator did not intend.
    murray_ratio: float
    feasible: bool
    reasons: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def describe(self) -> str:
        ok = "FEASIBLE" if self.feasible else "INFEASIBLE"
        return (f"synthesised bed at tier {self.tier.value} [{ok}]: {self.n_segments} "
                f"segments over {self.tissue_volume_mm3:.3g} mm^3, length density "
                f"{self.length_density:.0f} against a target of "
                f"{self.length_density_target:.0f} mm/mm^3; delivers "
                f"{self.cbf_delivered:.0f} ml/100g/min at "
                f"{self.pressure_drop_mmHg:.0f} mmHg with a transit of "
                f"{self.transit_time_s:.2f} s; tissue-to-capillary "
                f"{self.tissue_to_capillary_mm * 1e3:.0f} um mean / "
                f"{self.tissue_to_capillary_p90_mm * 1e3:.0f} um p90 against a Krogh "
                f"radius of {self.krogh_radius_mm * 1e3:.0f} um"
                + ("  REASONS: " + "; ".join(self.reasons) if self.reasons else ""))


@dataclass
class SynthesisedNetwork:
    """a vascular tree in the shape `vascular_tree_adjacency` already takes.

    `xyz`, `parent`, `radius_mm` and `segment_length_mm` are exactly the columns
    a `SiteTable` on the `vascular_tree` support needs, so the output of this
    generator drops into the existing builder with nothing in between.  that is
    deliberate: a synthesis that needed its own topology would be a second graph
    claiming to be the same anatomy.
    """

    xyz: Any
    parent: Any
    radius_mm: Any
    segment_length_mm: Any
    branch_order: Any
    #: 1 for the arterial side, -1 for the venous, 0 for the capillary bed.
    flow_sign: Any
    pressure_mmHg: Any
    flow_mm3_s: Any
    report: SynthesisReport

    def site_columns(self) -> dict:
        """the `columns` dict for a `vascular_tree` SiteTable.

        `parent`, `radius_mm`, `segment_length_mm` and `flow_sign` are the four
        `vascular_tree_adjacency` and `capillary_tissue_exchange` between them
        look for, and `branch_order` is the fifth this module adds as
        `structural.branch_order`.
        """
        return {"parent": self.parent, "radius_mm": self.radius_mm,
                "segment_length_mm": self.segment_length_mm,
                "flow_sign": self.flow_sign, "branch_order": self.branch_order}


def _krogh_radius_mm(cmro2_ml_100g_min: float, *, d_o2_m2_s: float = 1.7e-9,
                     c_capillary_mol_m3: float = 0.055) -> float:
    """the tissue radius one capillary can oxygenate, from Krogh's cylinder.

    R^2 = 4 D C / M, dropping the logarithmic term, which is the standard
    order-of-magnitude form and is used as one: this is a check on whether the
    synthesised spacing is in the right regime, not a computation of tissue PO2.

    the two constants are LITERATURE and are the only literature numbers in this
    module.  D is oxygen's diffusivity in brain tissue and C is the free oxygen
    concentration at the capillary wall at a venous-end PO2 of about 40 mmHg.
    they are declared here rather than folded into a magic number so that a
    caller who disagrees can see exactly what to change, and so that the result
    can be reported as depending on them.
    """
    # ml O2 / 100 g / min -> mol / m^3 / s, at 22.4 l/mol and a tissue density of 1
    m = (cmro2_ml_100g_min / 100.0) / 22.4 / 60.0 * 1000.0
    if m <= 0:
        return float("inf")
    return math.sqrt(4.0 * d_o2_m2_s * c_capillary_mol_m3 / m) * 1e3


def synthesise(*, box_mm: Any, depth_of: Any = None,
               tier: Tier | str = Tier.GENERATIVE_SYNTHESIS,
               penetrating_xyz: Any = None,
               cbf_ml_100g_min: float | None = None,
               cmro2_ml_100g_min: float = 5.0,
               grey: bool = True,
               stats: MicrovascularStatistics | None = None,
               seed: int = 0,
               max_nodes: int = 200_000) -> SynthesisedNetwork:
    """a capillary bed consistent with the measured statistics AND with the flow.

    the argument for this function existing is in the module docstring: the bed
    is not measurable in a living human, and a materialization needs one anyway
    for BOLD, for oxygen delivery and for thermal transport.  the argument for it
    being shaped this way is that the alternative -- sample points, connect
    neighbours, match a histogram -- produces a network that looks right and
    cannot carry blood.

    the procedure, and what each step is constrained by.

    1.  **penetrating vessels.**  descending arterioles and ascending venules on
        a jittered lattice at the MEASURED nearest-neighbour spacing, running
        from the pial surface to the depth of the box.  `penetrating_xyz`
        overrides the lattice and is how a subject's angiogram enters at tier 1:
        the vessels a 7T acquisition can actually see are placed where it saw
        them, and only the bed below is invented.

    2.  **capillary nodes.**  sampled at a depth-dependent rate so that the
        realised LENGTH density follows `density_at_depth`, using the identity
        that a graph of n nodes with mean degree 3 and mean segment length L has
        length density 1.5 n L / V.  the depth profile is measured and is not a
        detail: the density varies 2.6-fold through the ribbon, three times more
        than it varies between animals.

    3.  **edges.**  each node joined to its nearest neighbours until the measured
        degree distribution is reached -- dominated by 3, which is what a real
        capillary bed is.  nodes near a penetrating vessel are joined to it,
        which is what makes the bed a bed and not a disconnected cloud.

    4.  **radii.**  capillaries drawn from the measured lognormal; every vessel
        above them sized by MURRAY at the measured exponent, aggregating up the
        arterial tree from the leaves.  this is the step that makes the geometry
        physical rather than decorative -- and note that the exponent used is
        2.80 rather than Murray's 3, because that is what the measurement says.

    5.  **the flow solve.**  a Poiseuille network with the MEASURED
        viscosity-versus-calibre relation, pressures fixed at the arteriolar
        inlets and venular outlets, solved as a sparse Laplacian.  the solve is
        linear, so the delivered flow is exactly proportional to the imposed
        drop, and the drop the TARGET CBF requires can be read off in one step
        rather than searched for.

    6.  **the verdict.**  `feasible` is false when the required pressure drop is
        outside the range a cortical circulation can supply, when the transit
        time is too short for oxygen to leave the blood, or when tissue sits
        further from a capillary than the Krogh radius the target CMRO2 allows.
        a network that fails is returned WITH its report rather than raised on,
        because knowing which constraint it failed is the useful part.

    `cbf_ml_100g_min` defaults to the value measured on the source networks,
    which is a MOUSE's.  a human cortical materialization should pass its own --
    from ASL, which measures exactly this quantity in exactly these units without
    a calibration constant -- and that is the intended route: the geometry
    transfers, the perfusion is measured on the subject, and the generator
    reconciles them.
    """
    np = B._numpy("vascular_prior.synthesise")
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree

    st = stats or MEASURED
    tier = Tier(tier)
    rng = np.random.default_rng(seed)
    box = np.asarray(box_mm, float).reshape(2, 3)
    lo, hi = box[0], box[1]
    extent = hi - lo
    volume = float(np.prod(extent))
    cbf_target = float(cbf_ml_100g_min if cbf_ml_100g_min is not None
                       else st.cbf_ml_100g_min)
    assumptions: list[str] = []

    # -- 1. penetrating vessels ---------------------------------------------
    spacing = 2.0 * st.penetrating_distance_mm
    if penetrating_xyz is not None:
        pen_xy = np.asarray(penetrating_xyz, float)[:, :2]
        assumptions.append("penetrating vessels taken from the caller (an angiogram or "
                           "a two-photon map), not placed on a lattice")
    else:
        nx = max(int(extent[0] / spacing), 2)
        ny = max(int(extent[1] / spacing), 2)
        gx = lo[0] + (np.arange(nx) + 0.5) * extent[0] / nx
        gy = lo[1] + (np.arange(ny) + 0.5) * extent[1] / ny
        pen_xy = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
        pen_xy = pen_xy + rng.normal(0.0, 0.25 * spacing, pen_xy.shape)
        assumptions.append(
            f"penetrating vessels on a jittered lattice at the measured "
            f"{spacing * 1e3:.0f} um spacing -- a REGULARITY the real cortex does not "
            f"have, and the one place this synthesis is visibly not tissue")
    # alternate arteriole / venule, which is what a cortical surface does
    is_artery = (np.arange(pen_xy.shape[0]) % 2) == 0
    n_pen = pen_xy.shape[0]

    step = st.segment_length_mm
    n_depth = max(int(extent[2] / step), 2)
    z = lo[2] + np.linspace(0.0, extent[2], n_depth)

    xyz = [np.column_stack([np.repeat(pen_xy[:, 0], n_depth),
                            np.repeat(pen_xy[:, 1], n_depth),
                            np.tile(z, n_pen)])]
    pen_index = np.arange(n_pen * n_depth).reshape(n_pen, n_depth)
    n_pen_nodes = n_pen * n_depth

    # -- 2. capillary nodes at the measured depth-dependent density ----------
    depth_frac_of = (depth_of if depth_of is not None
                     else (lambda p: (p[:, 2] - lo[2]) / max(extent[2], 1e-9)))
    # mean degree 3 => 1.5 edges per node; length density = 1.5 n L / V
    target_density = float(np.mean(st.density_at_depth(np.linspace(0, 1, 51), np)))
    if not grey:
        # white matter carries roughly half the capillary of cortex.  a DECLARED
        # factor: the sources here are cortical blocks and a whole-brain graph
        # with no tissue mask, so nothing in this measurement resolves it.
        target_density *= 0.5
        assumptions.append("white matter density taken as half of grey -- DECLARED, "
                           "not measured; neither source here separates the two")
    n_cap = int(np.clip(target_density * volume / (1.5 * step), 100, max_nodes))
    cand = rng.uniform(lo, hi, size=(int(n_cap * 2.5), 3))
    w = st.density_at_depth(depth_frac_of(cand), np)
    keep = rng.random(cand.shape[0]) < (w / max(float(np.max(w)), 1e-9))
    cap_xyz = cand[keep][:n_cap]
    xyz.append(cap_xyz)
    xyz = np.vstack(xyz)
    n = xyz.shape[0]
    cap_index = np.arange(n_pen_nodes, n)

    # -- 3. edges ------------------------------------------------------------
    src = [pen_index[:, :-1].ravel()]
    dst = [pen_index[:, 1:].ravel()]
    tree = cKDTree(xyz)
    k = 5
    _, nb = tree.query(cap_xyz, k=k + 1)
    for j in range(1, k + 1):
        a = cap_index
        b = nb[:, j].astype(np.int64)
        take = rng.random(a.size) < (0.75 if j <= 3 else 0.10)
        src.append(a[take])
        dst.append(b[take])
    src = np.concatenate(src).astype(np.int64)
    dst = np.concatenate(dst).astype(np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    key = np.minimum(src, dst) * n + np.maximum(src, dst)
    _, uniq = np.unique(key, return_index=True)
    src, dst = src[uniq], dst[uniq]

    chord = np.linalg.norm(xyz[src] - xyz[dst], axis=1)
    seg_len = chord * st.tortuosity
    ok = seg_len > 1e-9
    src, dst, seg_len = src[ok], dst[ok], seg_len[ok]

    # a random bed has isolated fragments, and a fragment is not merely useless:
    # it makes the network Laplacian singular and the flow solve returns nan for
    # every node in the graph.  so the component containing the penetrating
    # vessels is kept and everything else is discarded and COUNTED -- discarding
    # it silently would quietly lower the length density that the same function
    # then reports as matching its target.
    from scipy.sparse.csgraph import connected_components
    adj = coo_matrix((np.ones(src.size), (src, dst)), shape=(n, n))
    ncomp, label = connected_components(adj, directed=False)
    main = int(np.bincount(label[:n_pen_nodes]).argmax())
    alive = label == main
    dropped_nodes = int((~alive).sum())
    dropped_len = float(seg_len[~(alive[src] & alive[dst])].sum())
    keep_e = alive[src] & alive[dst]
    src, dst, seg_len = src[keep_e], dst[keep_e], seg_len[keep_e]
    remap = np.full(n, -1, np.int64)
    remap[alive] = np.arange(int(alive.sum()))
    xyz = xyz[alive]
    src, dst = remap[src], remap[dst]
    pen_index = remap[pen_index]
    n = xyz.shape[0]
    n_pen_nodes = int(alive[:n_pen_nodes].sum())
    is_pen_full = np.zeros(len(alive), bool)
    is_pen_full[:len(alive)] = False

    # -- 4. radii, by Murray from the leaves upward --------------------------
    is_pen = np.zeros(n, bool)
    is_pen[pen_index[pen_index >= 0]] = True
    r_edge = np.exp(rng.normal(math.log(st.capillary_radius_mm),
                               st.capillary_radius_log_sd, size=src.size))
    # a penetrating vessel is sized by Murray from everything hanging off it:
    # r_pen = (sum over its capillaries of r^k)^(1/k).  that is the aggregation
    # Murray's law actually licenses and it is what makes the arteriole's calibre
    # a consequence of the bed rather than a free parameter.
    kmur = max(st.murray_exponent, 1.5)
    pen_of = np.full(n, -1, np.int64)
    for i in range(n_pen):
        ok_i = pen_index[i] >= 0
        pen_of[pen_index[i][ok_i]] = i
    touches = np.where(is_pen[src], pen_of[src], np.where(is_pen[dst], pen_of[dst], -1))
    on_pen = (is_pen[src] & is_pen[dst])
    feed = (~on_pen) & (touches >= 0)
    agg = np.zeros(n_pen)
    np.add.at(agg, touches[feed], r_edge[feed] ** kmur)
    r_pen = np.maximum(agg, (st.capillary_radius_mm * 2) ** kmur) ** (1.0 / kmur)
    r_edge = np.where(on_pen, r_pen[np.maximum(touches, 0)], r_edge)
    # taper: a descending arteriole gives up its cross-section on the way down,
    # so the segment at depth d carries only what is still below it.
    depth_i = np.zeros(src.size)
    depth_i[on_pen] = np.minimum(xyz[src[on_pen], 2], xyz[dst[on_pen], 2])
    frac = 1.0 - (depth_i - lo[2]) / max(extent[2], 1e-9)
    r_edge = np.where(on_pen, r_edge * np.maximum(frac, 0.15) ** (1.0 / kmur), r_edge)

    # -- 5. the flow solve ---------------------------------------------------
    mu = st.viscosity_pa_s(2.0 * r_edge, np)
    g = (np.pi * (r_edge * 1e-3) ** 4) / (8.0 * mu * (seg_len * 1e-3))   # m^3 / (s Pa)
    inlets = pen_index[is_artery, 0]
    outlets = pen_index[~is_artery, 0]
    inlets = inlets[inlets >= 0]
    outlets = outlets[outlets >= 0]
    if inlets.size == 0 or outlets.size == 0:
        raise RuntimeError(
            "the synthesised bed has no arteriolar inlet or no venular outlet inside "
            "its largest connected component -- the box is smaller than the measured "
            f"{spacing * 1e3:.0f} um penetrating-vessel spacing, so there is nothing "
            "for the bed to hang off.  ask for a larger box")
    fixed = np.zeros(n, bool)
    fixed[inlets] = True
    fixed[outlets] = True
    p_fixed = np.zeros(n)
    p_fixed[inlets] = 1.0                      # unit drop; the solve is linear
    free = np.nonzero(~fixed)[0]
    idx = np.full(n, -1, np.int64)
    idx[free] = np.arange(free.size)

    rows, cols, vals = [], [], []
    rhs = np.zeros(free.size)
    for a, b in ((src, dst), (dst, src)):
        m = ~fixed[a]
        rows.append(idx[a[m]]); cols.append(idx[a[m]]); vals.append(g[m])
        inner = m & ~fixed[b]
        rows.append(idx[a[inner]]); cols.append(idx[b[inner]]); vals.append(-g[inner])
        edge_bc = m & fixed[b]
        np.add.at(rhs, idx[a[edge_bc]], g[edge_bc] * p_fixed[b[edge_bc]])
    L = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(free.size, free.size)).tocsr()
    p = np.zeros(n)
    p[fixed] = p_fixed[fixed]
    if free.size:
        p[free] = spsolve(L.tocsc(), rhs)
    q_unit = g * (p[src] - p[dst])                      # m^3/s per Pa of drop

    # flow into the tissue = what leaves the arteriolar inlets
    inflow_unit = float(np.abs(q_unit[np.isin(src, inlets) | np.isin(dst, inlets)]).sum()
                        / 2.0)
    # CBF [ml/100g/min] for a drop of dP Pa: Q [m^3/s] * dP * 60 / V[m^3] * 100
    v_m3 = volume * 1e-9
    cbf_per_pa = inflow_unit * 60.0 / max(v_m3, 1e-30) * 100.0
    dp_pa = cbf_target / max(cbf_per_pa, 1e-30)
    dp_mmhg = dp_pa / MMHG_PA
    q = q_unit * dp_pa
    p_mmhg = p * dp_pa / MMHG_PA

    # -- 6. the verdict ------------------------------------------------------
    realised_density = float(seg_len.sum() / volume)
    if dropped_nodes:
        assumptions.append(
            f"{dropped_nodes} of {dropped_nodes + n} sampled nodes ({dropped_len:.0f} mm "
            f"of vessel) fell in fragments disconnected from the penetrating vessels and "
            f"were discarded; the density above is what SURVIVED, not what was sampled")
    cap_edge = r_edge <= 2.5 * st.capillary_radius_mm
    v_cap_mm3 = float((np.pi * r_edge[cap_edge] ** 2 * seg_len[cap_edge]).sum())
    q_mm3_s = inflow_unit * dp_pa * 1e9
    transit = v_cap_mm3 / max(q_mm3_s, 1e-30)

    probe = rng.uniform(lo, hi, size=(20_000, 3))
    mid = 0.5 * (xyz[src[cap_edge]] + xyz[dst[cap_edge]])
    d_probe, _ = cKDTree(mid).query(probe, k=1) if mid.size else (np.array([np.inf]), None)
    krogh = _krogh_radius_mm(cmro2_ml_100g_min)

    # the realised Murray ratio at the bed's own degree-three junctions.  the
    # generator IMPOSES Murray when it sizes the penetrating vessels, so this is a
    # check that it did and not a measurement -- but it is a check worth having,
    # because the capillary radii are independent draws and nothing forces THEM to
    # satisfy the law, so a bed whose ratio drifts far from the measured 1.5 has a
    # calibre distribution the generator did not intend.
    deg3 = np.bincount(np.concatenate([src, dst]), minlength=n)
    trio_nodes = np.nonzero(deg3 == 3)[0]
    murray_realised = float("nan")
    if trio_nodes.size:
        ends = np.concatenate([src, dst])
        eid = np.tile(np.arange(src.size), 2)
        o = np.argsort(ends, kind="stable")
        vs2, es2 = ends[o], eid[o]
        st2 = np.searchsorted(vs2, trio_nodes)
        tri = np.stack([es2[st2], es2[st2 + 1], es2[st2 + 2]], axis=1)
        rr = np.sort(r_edge[tri], axis=1)[:, ::-1]
        murray_realised = float(np.median(
            (rr[:, 1] ** 3 + rr[:, 2] ** 3) / np.maximum(rr[:, 0] ** 3, 1e-18)))

    reasons = []
    # the measured drop across a real cortical network is 44 mmHg.  a factor of
    # three either way is the tolerance, because the measured networks are blocks
    # with truncated pial vessels and the true drop across a whole cortical
    # column is not pinned to better than that.
    lo_dp, hi_dp = st.arteriole_to_venule_drop_mmHg / 3.0, st.arteriole_to_venule_drop_mmHg * 3.0
    if not (lo_dp <= dp_mmhg <= hi_dp):
        reasons.append(
            f"the target CBF needs {dp_mmhg:.0f} mmHg across the bed, outside the "
            f"{lo_dp:.0f}-{hi_dp:.0f} mmHg a cortical circulation supplies")
    if transit < 0.05:
        reasons.append(f"capillary transit is {transit:.3f} s, too short for oxygen to "
                       f"leave the blood (measured: {st.transit_time_s:.2f} s)")
    p90 = float(np.percentile(d_probe, 90))
    if p90 > krogh:
        reasons.append(f"tissue sits {p90 * 1e3:.0f} um from the nearest capillary at "
                       f"p90, past the {krogh * 1e3:.0f} um Krogh radius that "
                       f"{cmro2_ml_100g_min:g} ml/100g/min of CMRO2 allows")

    # branch order: 0 at the top of a penetrating vessel, rising with depth along
    # it, and 2 in the bed.  a coarse proxy for a Strahler order, which a network
    # with loops does not have -- and saying that is better than computing a
    # Strahler number on a spanning tree and calling it the network's.
    order = np.full(n, 2.0)
    for i in range(n_pen):
        ok_i = pen_index[i] >= 0
        order[pen_index[i][ok_i]] = np.arange(n_depth)[ok_i] / max(n_depth - 1, 1)
    sign = np.zeros(n)
    a_idx = pen_index[is_artery].ravel(); a_idx = a_idx[a_idx >= 0]
    v_idx = pen_index[~is_artery].ravel(); v_idx = v_idx[v_idx >= 0]
    sign[a_idx] = 1.0
    sign[v_idx] = -1.0

    # a parent array in the shape `vascular_tree_adjacency` wants: the spanning
    # tree of the solved network oriented DOWN the pressure gradient, which is
    # the only orientation that is not a guess.  the bed has loops and a tree
    # cannot carry them; that loss is recorded rather than hidden, because a
    # capillary bed's loops are exactly what makes its transit-time distribution
    # what it is.
    parent = np.full(n, -1, np.int64)
    hi_p = np.where(p[src] >= p[dst], src, dst)
    lo_p = np.where(p[src] >= p[dst], dst, src)
    best = np.full(n, -np.inf)
    for a, b in zip(lo_p, hi_p):
        if p[b] > best[a]:
            best[a] = p[b]
            parent[a] = b
    parent[inlets] = -1

    node_r = np.zeros(n)
    node_len = np.zeros(n)
    cnt = np.zeros(n)
    for arr in (src, dst):
        np.add.at(node_r, arr, r_edge)
        np.add.at(node_len, arr, seg_len)
        np.add.at(cnt, arr, 1.0)
    node_r /= np.maximum(cnt, 1.0)
    node_len /= np.maximum(cnt, 1.0)

    node_q = np.zeros(n)
    np.add.at(node_q, src, np.abs(q) * 1e9)
    node_q /= np.maximum(cnt, 1.0)

    if np.isfinite(murray_realised) and abs(
            math.log(max(murray_realised, 1e-9) / max(st.murray_ratio, 1e-9))) > 0.2:
        assumptions.append(
            f"the bed's junctions come out at sum(rc^3)/rp^3 = {murray_realised:.2f} "
            f"where the measured bed is at {st.murray_ratio:.2f}.  this is a REAL and "
            f"unreproduced feature: capillary radii here are independent draws from a "
            f"symmetric lognormal, so a junction's three branches are exchangeable and "
            f"the ratio comes out near 1 by symmetry, while a real bed systematically "
            f"expands its total cross-section going downstream.  the consequence is that "
            f"the synthesised bed's flow is more evenly distributed than a real one's, "
            f"which UNDERSTATES the heterogeneity of capillary transit time -- and "
            f"transit heterogeneity is precisely what raises effective oxygen extraction "
            f"above what the mean transit predicts.  a materialization whose question is "
            f"oxygen extraction rather than bulk delivery should not use this bed")

    assumptions += [
        f"capillary bed SYNTHESISED, not measured -- no in-vivo human modality "
        f"resolves it; tier {tier.value} describes the macro tree only",
        f"statistics are MOUSE cortex ({st.source}), transferred on the argument that "
        f"normalised capillary geometry is conserved and absolute density is not",
        f"radii assigned by Murray at the MEASURED exponent {kmur:.2f}, not at 3",
        "the bed has loops; the `parent` array is a spanning tree oriented down the "
        "solved pressure gradient and does NOT carry them",
    ]
    report = SynthesisReport(
        tier=tier, n_nodes=int(n), n_segments=int(src.size),
        tissue_volume_mm3=volume,
        length_density=realised_density, length_density_target=target_density,
        cbf_delivered=cbf_target, cbf_target=cbf_target,
        pressure_drop_mmHg=float(dp_mmhg), transit_time_s=float(transit),
        tissue_to_capillary_mm=float(np.mean(d_probe)),
        tissue_to_capillary_p90_mm=p90,
        krogh_radius_mm=krogh,
        murray_ratio=murray_realised,
        feasible=not reasons, reasons=tuple(reasons), assumptions=tuple(assumptions))
    return SynthesisedNetwork(
        xyz=xyz, parent=parent, radius_mm=node_r, segment_length_mm=node_len,
        branch_order=order, flow_sign=sign, pressure_mmHg=p_mmhg,
        flow_mm3_s=node_q, report=report)


# ---------------------------------------------------------------------------
# support
# ---------------------------------------------------------------------------


def vascular_support(sites=None, *, tier: Tier | str = Tier.GENERATIVE_SYNTHESIS,
                     parent=None, radius_mm=None, box_mm=None,
                     penetrating_xyz=None, support: str = "vascular_tree",
                     stats: MicrovascularStatistics | None = None,
                     **kw) -> tuple[Any, Tier]:
    """the vascular support, built at whichever tier the inputs allow.

    one entry point returning the tier alongside the graph, so that a
    materialization cannot obtain the edges without also obtaining the answer to
    "where did this come from" -- the same contract `tract_prior.edge_support`
    has and for the same reason.

    `tier` is a CEILING.  the returned tier is the weaker of what the caller
    declared and what the inputs are, so asking for `SUBJECT_ANGIOGRAM` with only
    a box returns `GENERATIVE_SYNTHESIS`.

    the three cases and what each actually gives you:

    - a `parent` and `radius_mm` in hand: this is a real tree from somewhere and
      it goes straight to `vascular_tree_adjacency`.  the tier is whatever the
      caller declares, capped, and this function has no way to check it -- an
      angiogram and an atlas warp arrive as the same two arrays.  that is why the
      declaration is a ceiling and why the note says which was claimed.

    - a `box_mm` and nothing else: `synthesise` runs, and the returned edge set
      carries the synthesis report in its note.

    - neither: an error naming what to supply, rather than an empty graph.
    """
    tier = Tier(tier)
    if parent is not None and radius_mm is not None:
        from ibm.topologies.vascular import vascular_tree_adjacency
        edges = vascular_tree_adjacency(sites, support=support, parent=parent,
                                        radius=radius_mm, **kw)
        return edges.with_features(), tier
    if box_mm is not None:
        net = synthesise(box_mm=box_mm, tier=tier, penetrating_xyz=penetrating_xyz,
                         stats=stats, **kw)
        return net, tier.demoted_to(
            Tier.SUBJECT_ANGIOGRAM if penetrating_xyz is not None
            else Tier.GENERATIVE_SYNTHESIS)
    raise B.MissingInput(
        "vascular_support", "parent + radius_mm, or box_mm",
        "either a centreline tree with radii -- from a TOF-MRA or QSM venogram "
        "segmentation, which reaches ~300 um and no further -- or the bounding box of "
        "the tissue whose bed is to be synthesised",
        "data/sources: 7t-qsm-venograms and venat for the venous side, "
        "circle-of-willis-centerline-resources and cerebral-artery-atlas-mouches2019 for "
        "the arterial, aneurisk for a worked centreline-with-radius representation.  "
        "for the microvasculature there is no such input for a living human and there "
        "will not be one; call this with `box_mm` and read the synthesis report")


# ---------------------------------------------------------------------------
# theta
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VascularPriorSet:
    """priors over per-segment hydraulic conductance, with their provenance.

    the tier travels with them because §7 requires a prediction resting on
    prior-dominated structure to be reportable as one, and because in this module
    the tier is doing more work than it does in `tract_prior`: at every tier the
    capillary bed is synthesised, so a `VascularPriorSet` never licenses a claim
    about an individual capillary at any tier at all.
    """

    tier: Tier
    tying: Tying
    priors: tuple[Prior, ...]
    group_of: Any = None
    stats: MicrovascularStatistics = _field(default_factory=lambda: MEASURED)
    note: str = ""

    def __len__(self) -> int:
        return len(self.priors)

    def describe(self) -> str:
        meds = [math.exp(p.loc) for p in self.priors]
        spreads = [math.exp(p.scale) for p in self.priors]
        lo, hi = (min(meds), max(meds)) if meds else (0.0, 0.0)
        return (f"{len(self.priors)} {self.tying.value} conductance priors at tier "
                f"{self.tier.value}, median {lo:.3g}-{hi:.3g} m^3/(s Pa), spread x"
                f"{(sum(spreads) / len(spreads)) if spreads else 0:.2f}"
                + (f"  ({self.note})" if self.note else ""))


def theta_prior(edges, *, tier: Tier | str = Tier.GENERATIVE_SYNTHESIS,
                radius_mm=None, length_mm=None,
                tying: Tying = Tying.PER_SITE,
                groups=None,
                stats: MicrovascularStatistics | None = None,
                angiogram: AngiogramUncertainty | None = None) -> VascularPriorSet:
    """a lognormal prior on each segment's hydraulic conductance.

    lognormal because conductance is positive and known to within a factor, and
    here that is not merely conventional: it is a fourth power of a radius whose
    own error is multiplicative, so the induced distribution is lognormal by
    construction rather than by convenience.

    **the fourth power is the whole content of the spread.**  G = pi r^4 / (8 mu
    L), so

        sd(log G)^2 = 16 sd(log r)^2 + sd(log L)^2

    and the two terms are not remotely comparable.  at tier 2 the measured
    between-subject calibre spread is 0.283 in log units; sixteen times its square
    is 1.28, against the segment length's 0.66^2 = 0.44.  a prior that propagated
    the radius error linearly would be four times too tight in log units -- a
    factor of three in conductance -- and would be tight in the direction that
    lets a fit believe a warped population atlas knows this subject's perfusion.

    at tier 3, where the individual calibre is a draw from a distribution rather
    than any kind of measurement, the WITHIN-bed radius spread enters as well and
    the conductance prior becomes very wide indeed.  that is the correct shape: a
    synthesised capillary's conductance is not an estimate of anything, it is a
    sample from a population, and the prior should say so rather than pretend the
    sample is the mean.

    the median is the Poiseuille conductance of the segment as built, using the
    MEASURED viscosity-versus-calibre relation rather than a constant -- which is
    a factor of two in the capillary bed and, at the fourth power, the difference
    between a bed that carries the measured CBF and one that does not.
    """
    np = B._numpy("vascular_prior.theta")
    st = stats or MEASURED
    ang = angiogram or ANGIOGRAM
    tier = Tier(tier)

    r = radius_mm if radius_mm is not None else (
        edges.features.get("radius_mm") if hasattr(edges, "features") else None)
    L = length_mm if length_mm is not None else (
        edges.features.get("distance_mm") if hasattr(edges, "features") else None)
    if r is None or L is None:
        raise B.MissingInput(
            "vascular_prior.theta_prior", "radius_mm and length_mm",
            "a per-segment radius and length; an EdgeSet from "
            "`vascular_tree_adjacency` already carries both as edge features",
            "build the support first with `vascular_support`, which returns them")
    r = np.asarray(r, float)
    L = np.asarray(L, float)
    mu = st.viscosity_pa_s(2.0 * r, np)
    g = np.pi * (np.maximum(r, 1e-6) * 1e-3) ** 4 / (
        8.0 * mu * np.maximum(L, 1e-6) * 1e-3)

    calibre_sd = ang.extra_log_sd(tier) / 4.0        # back out the radius term
    var = 16.0 * calibre_sd ** 2 + st.segment_length_log_sd ** 2
    if tier is Tier.GENERATIVE_SYNTHESIS:
        # the bed is a sample, so the within-bed spread is part of the prior and
        # not something a better measurement removes.
        var += 16.0 * st.capillary_radius_log_sd ** 2
    sd = math.sqrt(var)
    spread = math.exp(sd)

    prov = (Provenance.ATLAS if tier is Tier.POPULATION_ATLAS
            else Provenance.FIT if tier is Tier.SUBJECT_ANGIOGRAM
            else Provenance.WEAK)
    src = f"{st.source}; {ang.source}; tier {tier.value}"
    note = (f"median = Poiseuille conductance at the measured viscosity for the "
            f"segment's own calibre; spread x{spread:.1f} from a calibre log sd of "
            f"{calibre_sd:.3f} entering at the FOURTH power plus a segment length log "
            f"sd of {st.segment_length_log_sd:.3f}")
    if tier is Tier.GENERATIVE_SYNTHESIS:
        note += (f"; PLUS the within-bed radius spread {st.capillary_radius_log_sd:.3f}, "
                 f"because a synthesised capillary's calibre is a draw and not an "
                 f"estimate")

    if tying is Tying.PER_SITE or groups is None:
        priors = tuple(lognormal(max(float(x), 1e-30), spread,
                                 units="m^3/(s Pa)", provenance=prov,
                                 source=src, note=note) for x in g)
        return VascularPriorSet(tier, Tying.PER_SITE, priors, None, st, note)

    grp = np.asarray(groups).astype(np.int64)
    k = int(grp.max()) + 1 if grp.size else 0
    priors = []
    for j in range(k):
        sel = grp == j
        # geometric mean, because these are multiplicative and an arithmetic mean
        # over a fourth power is dominated by whichever segment came out widest.
        m = float(np.exp(np.mean(np.log(np.maximum(g[sel], 1e-30))))) if sel.any() else 1e-30
        priors.append(lognormal(m, spread, units="m^3/(s Pa)", provenance=prov,
                                source=src, note=note + f"; tied over {int(sel.sum())}"))
    return VascularPriorSet(tier, tying, tuple(priors), grp, st,
                            note + f"; {k} tying groups over {g.size} segments")


def describe() -> str:
    """three lines a provenance report can print without importing anything else.

    all three, always.  the first alone reads as though the capillary bed were
    known; the second says how little of it any angiogram sees; and the third is
    the one a reader most needs, because it names the four situations in which
    everything above is the wrong tool.
    """
    return "\n".join([
        MEASURED.describe(),
        ANGIOGRAM.describe(),
        ("this generalises for BOLD, oxygen delivery and thermal transport, which are "
         "functionals of transport statistics rather than of which capillary is where.  "
         "it does NOT generalise to watershed zones, stroke penumbra, tumour "
         "neovasculature or aging microangiopathy, where the individual network IS the "
         "signal and a synthesis conditioned on healthy statistics will reproduce "
         "healthy statistics"),
    ])


__all__ = ["Tier", "MicrovascularStatistics", "AngiogramUncertainty",
           "SynthesisReport", "SynthesisedNetwork", "VascularPriorSet",
           "MEASURED", "ANGIOGRAM", "synthesise", "vascular_support",
           "theta_prior", "describe"]
