"""large-scale pretraining: raw video -> cortical dynamics -> next frame.

STATE.md 7b. a frame is encoded into cortical drive, the IBM dynamics are evolved
for one frame interval, and the resulting cortical state is decoded to the next
frame.  the brain is in the middle of a self-supervised video model, so raw
naturalistic video becomes an optimization source for cortical parameters -- which
is the only regime in which 10^5 PER_SITE embeddings are identifiable at all.

**what is being learned is the fine cortico-cortical connectivity.**
ARCHITECTURE.md's division of labour is that tractography constrains the coarse
modular graph and the fine graph is learned; `ibm.processes.neural`'s association
weight is the declared factorization of exactly that

    w_ij  =  M[pi(i), pi(j)]  x  exp(-d_ij / l)  x  sigma(<e_i, e_j>)
             ^ tractography      ^ geometry        ^ LEARNED, per site

and this script optimizes the third factor.  the embeddings are the parameters;
the geometry and the parcel-scale prior are held.

three things are deliberately NOT the loss:

*effective rank* and *the count of distinct metastable sets* are tracked and
reported and never optimized.  STATE.md 7d: predictive loss alone is a capture
curriculum -- it is minimized by exploiting what the model can already represent,
so it deepens existing basins and installs no new ones.  if loss falls while these
do not rise, the run is capturing rather than teaching, and holding them out of
the loss is what keeps them diagnostic (ONTOLOGY.md 7).

*the fan-in normalization is not a tunable.*  `|L(0)| = 7218` was measured before
it, three orders of magnitude above unity, so without it the dynamics diverge
before any gradient is meaningful (STATE.md 4.10).

the encoder and decoder are learned jointly rather than being a held TRIBEv2.
that is fitting the brain to an encoder, and it is legitimate HERE because this
stage is a pretraining structuring of the weights, not the final fit -- the
measured-recording likelihoods in CURRICULUM.md stages 2 and 5 are what anchor
the result to a brain.
"""
from __future__ import annotations

import argparse
import os
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# the cortical sheet
# ---------------------------------------------------------------------------

#: the spherical proxy this file used to run on.  KEPT, not deleted: every
#: checkpoint on disk was trained on it, `dyn.pos` is a saved buffer so those
#: checkpoints restore their own sphere whatever the default is, and a change
#: that cannot be measured against what it replaces is a change that cannot be
#: defended.  `--geometry sphere` selects it; `IBM_CORTEX_GEOMETRY=sphere`
#: selects it for a script that has no flag.
#:
#: the area it is matched to, 202,437 mm^2, is worth stating now that the real
#: number is held: the fsaverage white surface is 130,438 mm^2 in total and
#: 118,310 mm^2 once the medial wall is removed.  so the proxy is 1.55x the
#: cortex it stands in for by area and 1.24x in linear scale -- which means
#: every distance on it, and therefore the exp(-d/l) prior and the 85 mm mean
#: separation of two random sites that made the long-range edges inert, was
#: inflated by about a quarter.
SPHERE_AREA_MM2 = 202437.0
SPHERE_RADIUS_MM = math.sqrt(SPHERE_AREA_MM2 / (4.0 * math.pi))

#: which sheet `cortical_sites` builds when the caller does not say.
GEOMETRY = os.environ.get("IBM_CORTEX_GEOMETRY", "surface")


def _sphere_sites(n: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.rand(n, generator=g) * 2 - 1
    theta = torch.rand(n, generator=g) * 2 * math.pi
    r = torch.sqrt(1 - z * z)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], 1) \
        * SPHERE_RADIUS_MM


def cortical_sites(n: int, device, seed: int = 0, geometry: str | None = None):
    """positions of `n` cortical sites, in millimetres, reproducibly.

    **This used to be a sphere.**  `docs/DISCONNECTS.md` row 2: sites were placed
    on a spherical shell area-matched to the measured white surface, "so that
    distances and therefore the exp(-d/l) prior are in millimetres and not
    arbitrary units", and the substitution was recorded rather than hidden.  It
    is now replaced rather than recorded: sites are **fsaverage white-surface
    vertices**, drawn with probability proportional to vertex area so that site
    density is uniform per mm^2 of cortex, and the medial wall is excluded.

    What that buys is not distance -- the sphere already had millimetres -- it is
    that a site now has an ATLAS LABEL, because it is at a place on a real
    surface.  On a sphere the insula is not separable at all (there is no lateral
    sulcus for it to be buried in), so the interoceptive port entered a 4.5%
    subsample of a `frontal` label that was itself a coordinate cut.  See
    `ibm/cortical_sheet.py`.

    The signature is unchanged and the draw is still a pure function of `seed`.
    It is now a pure function of the seed on EVERY DEVICE: the sampling runs on
    numpy and is moved to `device` afterwards, where the sphere ran a CPU torch
    generator and the long-range partner draw -- which CLAUDE.md records as
    having cost a factor of 112 elsewhere -- ran a device generator whose seed 0
    meant a different graph on cpu than on cuda.
    """
    geometry = GEOMETRY if geometry is None else geometry
    if geometry == "sphere":
        return _sphere_sites(n, seed).to(device)
    if geometry != "surface":
        raise ValueError(f"geometry must be 'surface' or 'sphere', not {geometry!r}")
    import ibm.cortical_sheet as CS
    xyz, _ = CS.sample_sites(n, seed=seed)
    return torch.from_numpy(xyz).to(device)


#: the six labels the spherical proxy cut out of coordinates.  they are the first
#: six of the atlas lobes below, in the same order, so `LOBES.index(name)` means
#: the same thing on both sheets for every name the sphere could express.
SPHERE_LOBES = ("occipital", "temporal", "parietal", "frontal",
                "precentral", "postcentral")


def _lobe_names():
    try:
        import ibm.cortical_sheet as CS
        return CS.LOBES
    except Exception:                                     # payload not staged
        return SPHERE_LOBES


LOBES = _lobe_names()


def is_spherical_proxy(pos) -> bool:
    """is this the old sphere?

    a positional test rather than a flag, because `pos` is a SAVED BUFFER: every
    checkpoint on disk restores its own site positions, and the sheet a stored
    graph was trained on is a property of that graph and not of whatever the
    current default is.  a flag would have to be threaded through every script
    that rebuilds a model from a state dict, and the one that was not threaded
    would silently read a sphere's coordinates through an atlas.

    the test is exact enough to be safe: on the sphere every site is at radius
    126.9 mm to floating-point, and on a cortical surface the radius has a
    standard deviation of about 12 mm.
    """
    r = pos.norm(dim=1)
    return bool((r.std() < 1e-2) and ((r.mean() - SPHERE_RADIUS_MM).abs() < 1e-1))


def _sphere_regions(pos):
    """the old six-label geometric convention, unchanged, for old checkpoints.

    NOT an atlas.  the axes are radiological -- x left-right, y posterior-
    anterior, z inferior-superior -- and the thresholds are set to published lobe
    fractions.  it is kept verbatim so that a sphere-vs-surface comparison is
    against what was actually run, not against a cleaned-up version of it.
    """
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = pos.norm(dim=1).clamp_min(1e-6)
    yn, zn, xn = y / r, z / r, x.abs() / r
    lab = torch.full((len(pos),), 5, dtype=torch.long, device=pos.device)
    lab[yn < -0.75] = 0                                    # occipital  ~12%
    lab[(zn < -0.10) & (xn > 0.40) & (yn >= -0.75)] = 1    # temporal   ~20%
    lab[(zn > 0.35) & (yn < 0.20) & (yn >= -0.75)] = 2     # parietal
    lab[yn > 0.55] = 3                                     # frontal    ~22%
    lab[(zn > 0.35) & (yn >= 0.20) & (yn <= 0.55)] = 4     # precentral (motor)
    return lab


def cortical_regions(pos):
    """the ATLAS label of every site: an index into `ibm.cortical_sheet.REGIONS`.

    68 hemisphere-qualified Desikan-Killiany gyri, read from
    `?h.aparc.annot` on fsaverage -- `data/sources/desikan2006`, whose `raw/`
    held one 12 KB checksums.txt and no bytes until
    `scripts/fetch_cortical_atlases.py` staged it.

    This function used to return one of six labels cut out of spherical
    coordinates, and said so in its own docstring: "a geometric convention on the
    spherical proxy, NOT an atlas".  The concrete thing that changes is that
    `insula` and the four cingulate divisions are now addressable, because they
    are labels on a surface that has a lateral sulcus and a corpus callosum
    rather than thresholds on a ball that has neither.

    **On a sphere it still returns the six-label convention**, and callers must
    not mix the two: use `region_index`, which knows which sheet it is looking
    at, rather than comparing this return value against a hard-coded integer.
    """
    if is_spherical_proxy(pos):
        return _sphere_regions(pos)
    import ibm.cortical_sheet as CS
    lab = CS.regions_at(pos.detach().cpu().numpy())
    return torch.from_numpy(np.ascontiguousarray(lab)).to(pos.device)


def region_index(pos, name: str):
    """the site indices belonging to a lobe, a gyrus, or a hemisphere's gyrus.

    on the atlas sheet `name` may be a lobe (`"occipital"` ... plus `"insula"`
    and `"cingulate"`, which the sphere could not express), a bare DK gyrus
    (`"rostralanteriorcingulate"`, both hemispheres), or a qualified label
    (`"lh.insula"`).

    on the spherical proxy only the six coordinate labels exist, and asking for
    one of the others raises rather than silently returning the enclosing lobe.
    THAT is the disconnect this change is about: `ibm/interoception.py` records
    that the interoceptive drive entered `frontal` because "a sphere has no
    lateral sulcus", and a substitution that raises when it is no longer needed
    is a substitution that cannot be left in by accident.
    """
    if is_spherical_proxy(pos):
        if name not in SPHERE_LOBES:
            raise KeyError(
                f"{name!r} is not separable on the spherical proxy; it has only "
                f"{SPHERE_LOBES}.  run with --geometry surface (or "
                "IBM_CORTEX_GEOMETRY=surface) to get the atlas, and see "
                "docs/DISCONNECTS.md row 2 for why this raises rather than "
                "substituting.")
        return (_sphere_regions(pos) == SPHERE_LOBES.index(name)) \
            .nonzero(as_tuple=True)[0]
    import ibm.cortical_sheet as CS
    want = torch.tensor(CS.resolve(name), device=pos.device)
    lab = cortical_regions(pos)
    idx = torch.isin(lab, want).nonzero(as_tuple=True)[0]
    if not len(idx):
        raise ValueError(
            f"{name!r} selects 0 of {len(pos)} sites.  a port with no sites is a "
            "drive that goes nowhere; materialize more sites or name a larger "
            "region.")
    return idx


def knn_edges(pos, k: int, chunk: int = 4096):
    """k nearest neighbours, chunked so the N x N distance matrix is never formed."""
    n = pos.shape[0]
    idx = torch.empty(n, k, dtype=torch.long, device=pos.device)
    dist = torch.empty(n, k, device=pos.device)
    for i in range(0, n, chunk):
        d = torch.cdist(pos[i:i + chunk], pos)
        dd, ii = torch.topk(d, k + 1, largest=False)
        idx[i:i + chunk] = ii[:, 1:]
        dist[i:i + chunk] = dd[:, 1:]
    return idx, dist


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class CorticalDynamics(nn.Module):
    """E/I population dynamics with a learned per-site association kernel."""

    def __init__(self, n_sites: int, embed_dim: int, k: int, device,
                 length_scale_mm: float = 40.0, long_range: float = 0.25,
                 graph_seed: int = 0, tanh_slope: float = 2.0,
                 long_gain: float = 1.0, long_topm: int = 0,
                 local_gain: float = 1.0, long_min_dist: float = 0.0,
                 geometry: str | None = None,
                 long_topology: str = "random", tract_threshold: float = 0.5,
                 tract_delays: bool = False,
                 tract_velocity_m_s: float = 8.0,
                 delay_shuffle: bool = False):
        super().__init__()
        self.n, self.k = n_sites, k
        pos = cortical_sites(n_sites, device, geometry=geometry)
        n_far = int(k * long_range)
        idx, dist = knn_edges(pos, k - n_far)
        delay_s = None
        if n_far and long_topology == "tract":
            # ------------------------------------------------------------------
            # LONG-RANGE PARTNERS FROM THE DECLARED TRACT TOPOLOGY.
            #
            # docs/DISCONNECTS.md row 3: `ibm/topologies/tract.py` declares which
            # cortical regions a fascicle joins and with what conduction delay,
            # argues that euclidean and geodesic metrics both get long-range
            # connectivity wrong, and was imported by nothing.  this is the
            # import.  a site in DK parcel `a` draws its partners uniformly among
            # the sites lying in the parcels a group connectome of 1064 HCP
            # subjects joins `a` to -- a uniform subsample of exactly the edge
            # set `tractometric_matrix` would emit, which is what a fixed
            # per-site budget can hold of a topology whose full expansion is
            # 10^10 edges at this resolution.
            #
            # the CONTROL is `long_topology="random"`, and it is matched by
            # construction rather than by fitting: the same `n_far` edges per
            # site, and the same flat long-range prior below, so the edge count
            # and the row L1 are identical and only the DESTINATIONS differ.
            # ------------------------------------------------------------------
            import ibm.cortical_tracts as CT
            if is_spherical_proxy(pos):
                raise ValueError(
                    "tract-constrained long-range edges need atlas labels, and "
                    "the spherical proxy has none.  use geometry='surface'.")
            reg = cortical_regions(pos).cpu().numpy()
            far_np, dly, _len, note = CT.draw_partners(
                reg, n_far, seed=graph_seed, threshold=tract_threshold,
                velocity_m_s=tract_velocity_m_s)
            self.tract_note = note
            far = torch.from_numpy(far_np).to(device)
            far_d = (pos[far] - pos[:, None, :]).norm(dim=-1)
            idx = torch.cat([idx, far], 1)
            dist = torch.cat([dist, far_d], 1)
            if tract_delays:
                delay_s = torch.from_numpy(dly).to(device)
                if delay_shuffle:
                    # DELAY-MATCHED CONTROL.  the same multiset of delays, moved
                    # to edges chosen at random, so "the tract arm is slower"
                    # cannot be confused with "the tract arm is wired
                    # differently".  a flat permutation of the whole (N, n_far)
                    # table, drawn from the graph seed.
                    g = np.random.default_rng(graph_seed + 1)
                    flat = delay_s.flatten()
                    delay_s = flat[torch.from_numpy(
                        g.permutation(flat.numel())).to(device)].view_as(delay_s)
        elif n_far and long_topology != "random":
            raise ValueError(
                f"long_topology must be 'random' or 'tract', not {long_topology!r}")
        elif n_far:
            # patchy long-range association fibres.  a pure k-NN graph is a local
            # sheet, and on a local sheet occipital and temporal sites are simply
            # not connected -- so no amount of training could associate them.
            # cortical association fibres are long-range and patchy, and this is
            # the minimal declaration of that: a fraction of each node's budget
            # spent on distant partners drawn uniformly, whose weight the learned
            # factor is then free to keep or discard.
            #
            # drawn from a DEDICATED generator, not the global RNG.  the global
            # draw made the topology a function of whatever seed the caller set
            # for weight initialisation, so `scripts/embody.py --seed`, which is
            # documented as "matched initialization for ablations", silently
            # redrew the association graph as well: an ablation meaning to vary
            # the init varied the wiring too and its arms were not comparable.
            # `cortical_sites` already takes this precaution for the positions,
            # which is why the local k-NN was reproducible while these were not.
            #
            # graph_seed=0 on cuda reproduces the graph every existing
            # checkpoint was trained against EXACTLY -- verified against
            # ckpt/ibm1_curriculum16.pt at coincidence 1.00000 against a chance
            # of 1/30000 -- so this is a no-op for every checkpoint on disk.
            # the draw is still device-dependent (a cuda generator and a cpu
            # generator with the same seed give different streams); making it
            # device-independent would change every stored graph and so needs a
            # schema bump rather than a silent fix.
            g = torch.Generator(device=device).manual_seed(graph_seed)
            far = torch.randint(0, n_sites, (n_sites, n_far), device=device,
                                generator=g)
            far_d = (pos[far] - pos[:, None, :]).norm(dim=-1)
            idx = torch.cat([idx, far], 1)
            dist = torch.cat([dist, far_d], 1)
        self.register_buffer("pos", pos)
        self.n_far = n_far
        self.long_topology = long_topology
        self.register_buffer("idx", idx)
        if delay_s is not None:
            # PART OF THE TRAINED OBJECT, so it is a persistent buffer and it is
            # saved.  CLAUDE.md's `read_idx` entry is the reason: that was a
            # plain attribute, was not saved, and cost a factor of 112 on every
            # video number.  anything that selects or times an edge is part of
            # the artifact.
            self.register_buffer("delay_s", delay_s.to(torch.float32))
            self.tract_delays = True
        else:
            self.tract_delays = False
        self._lag_cache: dict = {}
        # the geometric prior, held: exp(-d/l), normalized by fan-in so that the
        # total drive onto a node is O(1) rather than O(k).
        geo = torch.exp(-dist / length_scale_mm) / k
        if long_range:
            # a PATCHY, distance-INDEPENDENT prior on the long-range edges.
            #
            # applying exp(-d/40mm) to them too was measured to make them inert: a
            # uniform-random partner on a 127 mm sphere is ~85 mm away, so a
            # long-range edge starts with 7x less weight than a local one and the
            # learned factor never overcomes it.  the ablation is unambiguous --
            # severing EVERY long-range edge cost +0.1% loss while the occipito-
            # temporal weights sat at 4.4x the random baseline.  large and inert.
            #
            # real association fibres are long-range AND strong, and
            # `ibm.topologies.association` already says so: the connection is
            # patchy and specific, not a decaying function of distance.  so these
            # edges get a flat prior and the LEARNED factor decides which survive,
            # which is the division of labour ARCHITECTURE.md asks for.
            n_loc = k - n_far
            geo[:, n_loc:] = float(torch.exp(-dist[:, :n_loc].median() /
                                             length_scale_mm)) / k
        self.register_buffer("geo", geo)

        # THE PARAMETERS.  one embedding per site; the learned factor of w_ij.
        self.embed = nn.Parameter(torch.randn(n_sites, embed_dim) * 0.02)
        self.log_len = nn.Parameter(torch.tensor(math.log(length_scale_mm)))

        # E/I population parameters, initialized at the declared priors
        self.w_ee = nn.Parameter(torch.tensor(0.30))
        self.w_ei = nn.Parameter(torch.tensor(0.20))
        self.w_assoc = nn.Parameter(torch.tensor(0.50))
        self.tau_m = 0.015
        self.tau_a = 0.30          # STATE.md 4.10: puts the SO at 0.533 Hz, in band
        self.a_gain = nn.Parameter(torch.tensor(0.10))
        self.v_half, self.slope, self.r_max = -55.0, 4.0, 100.0
        self.e_rest, self.e_rev = -65.0, -70.0

        # ANISOTROPY.  defaults reproduce the original kernel exactly.
        #
        # measured (scripts/measure_hop_transfer.py): the sheet loses ~99% of a
        # coherent perturbation per hop, and the arriving fraction factors as
        #     arrival  =  (row gain)  x  (fraction of the row's weight MASS that
        #                                 lands on the source region).
        # the second term is pinned at ~5% because `tanh(2*sim)` with `sim` a
        # cosine in [-1, 1] has an argument in [-2, 2] and tanh(2) = 0.964: every
        # edge is saturated to the same magnitude, mean |w| = 0.77 of the
        # ceiling.  the learned factor can therefore express only a SIGN, and
        # 48 roughly-equal edges cannot form a pathway.  raising the row gain
        # instead does not work -- swept, it drives the resting rate onto the
        # sigmoid's upper rail at 87 Hz and transfer FALLS.
        #
        # so these knobs concentrate rather than amplify:
        #   tanh_slope  the saturation slope, previously the hard-coded 2.0.
        #   long_topm   keep only the m strongest long-range edges per site and
        #               redistribute the row's long-range L1 mass onto them, so
        #               the row gain -- which is what governs stability -- is
        #               UNCHANGED while a chosen path's gain rises by ~n_far/m.
        #   long_gain   a separate multiplier on the long-range population, kept
        #               distinct from w_assoc so the local sheet's fixed point
        #               is not moved when the association fibres are.
        #   local_gain  the matching multiplier on the local k-NN population.
        #               with long_gain it turns the fix into a pure
        #               REALLOCATION: 36 local edges carry 75% of every row's
        #               weight and contribute nothing to long-range transport,
        #               so moving that mass onto the association fibres raises
        #               arrival while leaving the row L1 -- and therefore the
        #               resting rate and the operating point -- exactly where
        #               they were.  that matters because the gain sweep showed
        #               every attempt to raise the row gain instead moves the
        #               fixed point onto the sigmoid's upper rail and makes
        #               transport WORSE.
        self.tanh_slope = float(tanh_slope)
        self.long_gain = float(long_gain)
        self.long_topm = int(long_topm)
        self.local_gain = float(local_gain)
        #   long_min_dist  when > 0, long_topm selects GREEDILY FOR SPATIAL
        #               DIVERSITY instead of by magnitude alone: keep the
        #               strongest surviving long-range edge, then the strongest
        #               remaining edge whose partner is at least this far (mm)
        #               from every partner already kept, and repeat.
        #
        #               the point is that plain top-m can keep four edges that
        #               all land in the same patch, and with the |sim|
        #               distribution as flat as it is measured to be, which four
        #               it keeps is close to arbitrary.  spreading the budget is
        #               also the better anatomy: a column's association fibres
        #               project to several distinct areas, not several times into
        #               one.
        #
        #               the rule is deliberately TASK-BLIND.  it never mentions
        #               occipital, temporal, postcentral or precentral -- keying
        #               the selection on the regions the convergence metric
        #               drives and reads would tune the connectivity using the
        #               evaluation's own definition, and the capacity table would
        #               then be reporting how well the rule had been told the
        #               answer.  it says only "spread your long-range budget",
        #               so any gain in multimodal convergence is a consequence
        #               rather than a restatement.
        self.long_min_dist = float(long_min_dist)
        #: the thalamus, when one is attached.  see `attach_thalamus`.
        self.tct = None

    def edge_weights(self):
        """the learned association weights, one per edge.  (N, k).

        computed ONCE per forward and reused across the dynamics steps.  building
        it inside the step loop rebuilt an (N, k, embed) tensor -- 6 GB at 250k
        sites -- eight times per forward, which was both the memory ceiling and
        most of the runtime.  the weights are constant within a forward because
        the embeddings are, so recomputing them was pure waste.
        """
        e = F.normalize(self.embed, dim=-1)
        sim = (e.unsqueeze(1) * e[self.idx]).sum(-1)          # (N, k)
        # TANH, not sigmoid: the weights must be able to be NEGATIVE.
        #
        # a strictly positive, fan-in-normalized kernel is a diffusion operator.
        # applying it eight times per forward is eight rounds of weighted
        # averaging, which drives every site toward the graph's mean and destroys
        # rank by construction -- measured: effective rank fell 1.57 -> 1.01 with
        # 512 sites and a ceiling of 15, i.e. the cortical state became genuinely
        # rank-one while reconstruction got WORSE.
        #
        # that is ONTOLOGY.md 4's flat-force-field pathology, and its cause here
        # is structural rather than a tuning failure: cortical association is not
        # all-excitatory, and a kernel that cannot subtract can only blur.  tanh
        # lets the learned factor place opposition between sites, which is what
        # keeps a representation from washing out.
        w = self.geo * torch.tanh(self.tanh_slope * sim)
        if self.n_far and (self.long_topm or self.long_gain != 1.0
                           or self.local_gain != 1.0):
            n_loc = self.k - self.n_far
            lw = w[:, n_loc:]
            if self.long_topm and self.long_topm < self.n_far:
                mass = lw.abs().sum(1, keepdim=True)
                if self.long_min_dist > 0:
                    keep = self._diverse_keep(lw)
                else:
                    keep = torch.zeros_like(lw, dtype=torch.bool).scatter_(
                        1, lw.abs().topk(self.long_topm, dim=1).indices, True)
                lw = lw * keep
                # renormalise to the mass that was there before the pruning, so
                # this is a redistribution and NOT a gain change.  without this
                # top-m would simply delete 10 of 12 edges and the comparison
                # against the unpruned kernel would confound concentration with
                # attenuation -- the exact "wrong thing to compare against"
                # failure docs/LOG.md keeps a ledger of.
                lw = lw * (mass / lw.abs().sum(1, keepdim=True).clamp_min(1e-30))
            w = torch.cat([w[:, :n_loc] * self.local_gain,
                           lw * self.long_gain], 1)
        return w

    def _diverse_keep(self, lw):
        """greedy spatially-diverse selection of `long_topm` long-range edges.

        returns a (N, n_far) bool mask.  a row may keep FEWER than m edges when
        the separation constraint cannot be satisfied, which is intended: such a
        row spends its long-range budget on fewer, better-separated partners
        rather than on near-duplicates.
        """
        n_loc = self.k - self.n_far
        part = self.idx[:, n_loc:]                                # (N, n_far)
        p = self.pos[part]                                        # (N, n_far, 3)
        score = lw.abs()
        avail = torch.ones_like(score, dtype=torch.bool)
        keep = torch.zeros_like(score, dtype=torch.bool)
        rows = torch.arange(score.shape[0], device=score.device)
        for _ in range(self.long_topm):
            s = score.masked_fill(~avail, -1.0)
            best = s.argmax(1)
            ok = s[rows, best] >= 0
            keep[rows, best] |= ok
            avail[rows, best] = False
            d = (p - p[rows, best].unsqueeze(1)).norm(dim=-1)      # (N, n_far)
            # rows that could not pick anything this round are left untouched
            avail &= (d >= self.long_min_dist) | ~ok.unsqueeze(1)
        return keep

    def association(self, r, w, hist=None, ptr: int = 0, dt: float | None = None):
        """message passing on the cached kernel.  r: (B, N), w: (N, k).

        with `hist` supplied, the LONG-RANGE columns read a delayed rate instead
        of the current one.  `ibm/topologies/tract.py` is emphatic that this is
        the one thing the tractometric topology carries which the local ones do
        not: "cortico-cortical conduction delays run from under a millisecond to
        tens of milliseconds, they are comparable to the periods of the rhythms
        the model is about".  the local k-NN columns are not delayed, which is
        the same claim in the other direction -- a u-fibre at 8 m/s over the
        median 4 mm k-NN edge is 0.5 ms, below any timestep this file runs at.

        `hist` is (B, depth, N), a ring buffer of past rates, and `ptr` is the
        slot the CURRENT rate was just written to.  the per-edge lag is
        `round(delay_s / dt)`, so a delay below half a timestep is zero lag and
        the edge is instantaneous -- stated rather than hidden, because it means
        the delays only bite at a `dt` that resolves them.
        """
        if hist is None or not getattr(self, "tract_delays", False):
            return (r[:, self.idx] * w).sum(-1)               # (B, N)
        n_loc = self.k - self.n_far
        out = (r[:, self.idx[:, :n_loc]] * w[:, :n_loc]).sum(-1)
        lag, depth = self._lags(dt)
        slot = torch.remainder(ptr - lag, depth)              # (N, n_far)
        flat = (slot * self.n + self.idx[:, n_loc:]).reshape(-1)
        far = hist.reshape(hist.shape[0], depth * self.n)[:, flat] \
                  .view(r.shape[0], self.n, self.n_far)
        return out + (far * w[:, n_loc:]).sum(-1)

    def _lags(self, dt: float):
        """(per-edge lag in steps, ring depth) for a timestep, cached.

        cached per `dt` and not per call: the table is (N, n_far) int64 and
        rebuilding it inside the step loop was the shape of the waste
        `edge_weights` already documents.
        """
        key = round(float(dt), 12)
        got = self._lag_cache.get(key)
        if got is None:
            lag = torch.round(self.delay_s / float(dt)).long().clamp_min(0)
            got = (lag, int(lag.max().item()) + 1)
            self._lag_cache[key] = got
        return got

    def rate(self, v):
        return self.r_max * torch.sigmoid((v - self.v_half) / self.slope)

    def attach_thalamus(self, tct):
        """close the thalamo-cortico-thalamic loop around this sheet.

        `ibm/processes/tct.py` holds the thalamus.  attaching it here rather than
        at each call site is deliberate: there are nine heads in this file and
        every one of them calls `step`, so the loop has to be inside `step` for
        it to be CONTINUOUSLY active -- present in every forward of every head,
        and running between the frames as well as during them, which is what
        `docs/DYNAMICS.md` §2 means by the thalamus setting the effective
        cortical operator rather than being another input port.

        the thalamus keeps its own state ACROSS forwards.  `init_state` resets
        the cortex every forward (each clip starts from rest); the thalamus does
        not, because a loop that is re-initialized every 8 steps has no history
        to oscillate with.  `ThalamoCortical.ensure` re-initializes it only when
        the batch shape or the timestep changes.

        the thalamus needs a timestep that resolves its 5/8 ms delay lines.  this
        file's default `--dt` is 5e-3, at which a 5 ms delay is ONE STEP, so
        `ensure` will raise rather than quietly integrate a loop with no delay in
        it.  run with `--dt 1e-4` and more `--dyn-steps`, or leave the thalamus
        detached.
        """
        self.tct = tct
        return self

    def step(self, s, drive, dt, w):
        """one Euler step.  `s` is (v, r, a, gi), plus (hist, ptr) with delays.

        the state grew a fifth and sixth element rather than the delay history
        living on the module, and that is deliberate: a per-batch buffer held as
        an attribute is shared between two heads that run concurrently and
        silently wrong under gradient checkpointing.  every existing caller
        either indexes `s[1]` or passes `s` straight back, so the longer tuple is
        transparent to all of them.
        """
        v, r, a, gi = s[0], s[1], s[2], s[3]
        hist = ptr = None
        if getattr(self, "tract_delays", False):
            _, depth = self._lags(dt)
            if len(s) >= 6 and s[4] is not None:
                hist, ptr = s[4].clone(), int(s[5])
            else:
                hist, ptr = torch.zeros(v.shape[0], depth, self.n,
                                        device=v.device, dtype=v.dtype), 0
            hist[:, ptr] = r
        tct = getattr(self, "tct", None)
        if tct is not None:
            # the ascending limb, added to whatever sensory drive the head
            # supplies.  when the head supplies none this is the ONLY drive, and
            # the sheet is running on the loop alone.
            tct.ensure(v.shape[0], dt, device=v.device)
            drive = drive + tct.step(r, dt)
        r_inf = self.rate(v)
        assoc = self.association(r, w, hist=hist, ptr=ptr, dt=dt)
        g_e = self.w_ee * r / self.r_max + self.w_assoc * assoc / self.r_max
        dv = (-(v - self.e_rest) + 20.0 * g_e - a + drive) / self.tau_m
        # conductance-based shunting, LINEAR in g_i (STATE.md 4.11)
        dv = dv - (v - self.e_rev) * gi.clamp_min(0.0) / self.tau_m
        dr = (r_inf - r) / 5e-3
        da = (self.a_gain * r_inf - a) / self.tau_a
        dgi = (self.w_ei * r / self.r_max - gi) / 8e-3
        out = (v + dt * dv, r + dt * dr, a + dt * da, gi + dt * dgi)
        if hist is not None:
            out = out + (hist, (ptr + 1) % hist.shape[1])
        return out

    def init_state(self, b, device):
        z = torch.zeros(b, self.n, device=device)
        return (torch.full_like(z, self.e_rest), z, z.clone(), z.clone())


def dynamics_from_state_dict(sd, device, prefix: str = "dyn.", **kw):
    """rebuild a `CorticalDynamics` whose graph MATCHES a saved one, then load it.

    every script in this repo does the same three lines by hand -- read
    `dyn.embed`'s shape for `n` and `embed_dim`, read `dyn.idx`'s for `k`,
    construct, `load_state_dict(strict=False)` -- and every one of them would
    break silently the moment a checkpoint carried a buffer they did not expect.
    `delay_s` is such a buffer.

    the graph is RESTORED, never redrawn: `pos`, `idx`, `geo` and `delay_s` are
    all persistent buffers, so what the constructor draws is thrown away by the
    load.  that is the point -- CLAUDE.md: "the graph is part of the trained
    object; `dyn.idx`/`geo`/`pos` must be saved and restored, never redrawn" --
    and this function asserts it rather than trusting it.
    """
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)} \
        if any(k.startswith(prefix) for k in sd) else dict(sd)
    n, embed_dim = sub["embed"].shape
    k = sub["idx"].shape[1]
    kw.setdefault("tract_delays", "delay_s" in sub)
    if kw["tract_delays"]:
        kw.setdefault("long_topology", "tract")
    dyn = CorticalDynamics(int(n), int(embed_dim), int(k), device, **kw)
    miss = dyn.load_state_dict({kk: v.to(device) for kk, v in sub.items()},
                               strict=False)
    if miss.missing_keys or miss.unexpected_keys:
        raise KeyError(f"state dict does not match: missing {miss.missing_keys}, "
                       f"unexpected {miss.unexpected_keys}")
    for name in ("pos", "idx", "geo"):
        assert torch.equal(getattr(dyn, name), sub[name].to(device)), \
            f"{name} did not survive the load -- the graph is not the saved graph"
    dyn._lag_cache = {}
    return dyn


class AudioLoop(nn.Module):
    """cochleagram frame -> cortical drive -> dynamics -> next cochleagram frame.

    the same loop as `VideoLoop` with a different port.  the input is the output
    of `transduction:gammatone_cochleagram` -- ERB-spaced bands, phase-blind --
    so the pretraining signal enters through the representation the model's own
    cochlea produces rather than through a spectrogram chosen for convenience.
    drive reaches a temporal-lobe subset instead of an occipital one; the cortical
    parameters in between are SHARED with the video loop, which is the whole point
    of there being no standard materialization (STATE.md 7c).
    """

    def __init__(self, dyn: CorticalDynamics, n_bands: int = 64, ctx: int = 8,
                 hidden: int = 256):
        super().__init__()
        self.dyn, self.n_bands, self.ctx = dyn, n_bands, ctx
        self.enc = nn.Sequential(
            nn.Flatten(), nn.Linear(n_bands * ctx, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.n_in = dyn.n // 8
        self.to_cortex = nn.Linear(hidden, self.n_in)
        self.read_sites = 4096
        self.read_idx = torch.linspace(0, dyn.n - 1, self.read_sites).long()
        self.from_cortex = nn.Linear(self.read_sites, hidden)
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_bands))

    def forward(self, ctx, n_steps: int, dt: float):
        b = ctx.shape[0]
        h = self.enc(ctx)
        drive = torch.zeros(b, self.dyn.n, device=ctx.device)
        # the auditory port sits in the middle of the sheet, not the posterior pole
        off = self.dyn.n // 3
        drive[:, off:off + self.n_in] = self.to_cortex(h)
        s = self.dyn.init_state(b, ctx.device)
        w = self.dyn.edge_weights()
        for _ in range(n_steps):
            s = self.dyn.step(s, drive, dt, w)
        read = s[1][:, -self.dyn.n // 8:]
        return self.dec(self.from_cortex(read)), s


class PairedNeuralLoop(nn.Module):
    """stimulus -> cortex -> MEASURED neural activity.

    the materialization that stops the cortex being decorative.  in the
    self-supervised loops both the encoder and the decoder are learned, so nothing
    prevents the model routing information around the dynamics and using them as a
    delay line -- and a video generator with a decorative brain would still show a
    falling loss.  here the target is MEG a real head produced while hearing this
    stimulus, so the cortical state cannot be arbitrary: it has to be the state
    that generates this field.

    the readout is a linear map from cortical rate to sensors, which is what a
    lead field IS.  it is learned rather than computed from a head model because
    the sites here are a spherical stand-in rather than a subject's cortex -- so
    this stage learns the instrument along with the dynamics, and CURRICULUM.md
    stage 2 is where a measured forward model replaces it.
    """

    def __init__(self, dyn: CorticalDynamics, n_bands: int = 64, n_sensors: int = 306,
                 ctx: int = 125, hidden: int = 256, read_sites: int = 4096,
                 lead_rank: int = 64):
        super().__init__()
        self.dyn, self.ctx, self.n_sensors = dyn, ctx, n_sensors
        self.port = dyn.n // 8
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(n_bands * ctx, hidden),
                                 nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.to_cortex = nn.Linear(hidden, self.port)
        # auditory port: temporal, not occipital
        self.off = dyn.n // 3
        self.read_idx = torch.linspace(0, dyn.n - 1, read_sites).long()
        # LOW-RANK lead field, and the rank is a physiological fact rather than a
        # regularization guess.
        #
        # a free 4096 -> 306 linear map is 1.25M parameters and can express almost
        # any mapping, which is why the paired head reached skill +0.94 with an
        # effective cortical rank of 1.03: the readout was doing the work and the
        # cortex was a scalar.  a real MEG lead field cannot do that.  the spatial
        # degrees of freedom a 306-channel array can resolve is ~60-80 -- the field
        # is smooth, the sensors are far from the sources, and the operator is
        # severely ill-conditioned by physics.  factorizing through `lead_rank`
        # imposes exactly that ceiling, so the readout can no longer substitute for
        # the dynamics and any variance explained has to come through the cortex.
        self.lead_rank = lead_rank
        self.lead_u = nn.Linear(read_sites, lead_rank, bias=False)
        self.lead_v = nn.Linear(lead_rank, n_sensors, bias=False)

    def forward(self, coch_ctx, n_steps: int, dt: float):
        b = coch_ctx.shape[0]
        drive = torch.zeros(b, self.dyn.n, device=coch_ctx.device)
        drive[:, self.off:self.off + self.port] = self.to_cortex(self.enc(coch_ctx))
        s = self.dyn.init_state(b, coch_ctx.device)
        w = self.dyn.edge_weights()
        for _ in range(n_steps):
            s = self.dyn.step(s, drive, dt, w)
        idx = self.read_idx.to(s[1].device)
        return self.lead_v(self.lead_u(s[1][:, idx])), s


class VisualContrastiveLoop(nn.Module):
    """image -> cortex -> embedding, aligned contrastively with the measured EEG.

    the objective the data actually supports.  measured on correctly paired
    THINGS-EEG2, same encoder, same split:

        waveform regression   peak skill +0.011, then negative
        contrastive retrieval held-out top-1 21.5% against 0.5% chance

    43x chance for discrimination and essentially nothing for reconstruction.  the
    two are not the same problem: retrieval needs only enough structure to tell one
    evoked response from another, while regression must reproduce an amplitude at
    every channel and every sample -- and those amplitudes are dominated by trial
    and subject noise that no stimulus can predict.  fitting MSE against them
    spends the whole model on the unpredictable part.

    the cortex stays in the path.  the image drives the occipital port, the
    dynamics run, and the cortical state is read into the embedding -- so the
    alignment is only achievable if the dynamics carry stimulus-specific
    structure.  that keeps the term a test of the substrate rather than of an
    encoder bolted beside it.
    """

    def __init__(self, dyn: CorticalDynamics, n_sensors: int = 64, n_times: int = 25,
                 dim: int = 128, hidden: int = 256, read_sites: int = 4096):
        super().__init__()
        self.dyn, self.n_times = dyn, n_times
        self.port = dyn.n // 8
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden), nn.GELU())
        self.to_cortex = nn.Linear(hidden, self.port)
        self.read_idx = torch.linspace(0, dyn.n - 1, read_sites).long()
        self.cortex_head = nn.Sequential(nn.Linear(read_sites, 512), nn.GELU(),
                                         nn.Linear(512, dim))
        self.eeg_head = nn.Sequential(
            nn.Conv1d(n_sensors, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Flatten(), nn.Linear(128 * ((n_times + 1) // 2), 512), nn.GELU(),
            nn.Linear(512, dim))

    def embed_image(self, img, substeps: int = 4, dt: float = 2e-2,
                    n_steps: int | None = None):
        """run the dynamics, then read the FINAL state into the embedding.

        `n_steps` is decoupled from `n_times` on purpose.  the evoked head needed
        one integrator pass per output sample because its target was the whole
        trajectory; this head reads one state, so simulating the full 500 ms costs
        100 dynamics steps to use exactly one of them.  measured, that was 15.8
        s/step -- 35 hours for the run -- against a dynamics-free control that
        reached its answer in minutes.  what the embedding needs is enough
        propagation for the drive to reach the readout sites, not a physiological
        epoch length.
        """
        b = img.shape[0]
        drive = torch.zeros(b, self.dyn.n, device=img.device)
        drive[:, :self.port] = self.to_cortex(self.enc(img))
        s = self.dyn.init_state(b, img.device)
        w = self.dyn.edge_weights()
        h = dt / substeps
        for _ in range((n_steps if n_steps is not None else self.n_times) * substeps):
            s = self.dyn.step(s, drive, h, w)
        z = self.cortex_head(s[1][:, self.read_idx.to(img.device)])
        return F.normalize(z, dim=-1), s

    def embed_eeg(self, eeg):
        return F.normalize(self.eeg_head(eeg), dim=-1)


class AudioContrastiveLoop(nn.Module):
    """cochleagram -> cortex -> embedding, aligned contrastively with measured MEG.

    the auditory twin of `VisualContrastiveLoop`, and it exists now because the
    corpus it needs was only fixed on 2026-09-07.  before that the LibriBrain
    arrays were misaligned by up to +/-3.4 s -- the builder assumed the audio and
    MEG clocks shared a rate when they differ by ~4,800 ppm -- and every auditory
    result measured against them, four in total, was void.

    the ceiling this has to beat is measured, not assumed.  two convnets with no
    dynamics in the path, on the same corpus and split, averaged over 8 held-out
    pools of 200:

        200 ms window   4.62% +/- 0.99    9.2x chance
        1 s    window   7.12% +/- 1.71   14.2x chance

    so the bar is **14.2x**, and the window matters: the coupling sits at a 140 ms
    lag, and 200 ms barely contains one response.  the default window here is 1 s
    for that reason rather than by analogy with the visual head.

    two things differ from the visual loop, both forced by the data:

    *the port is temporal, not occipital*.  auditory cortex is a different patch of
    the sheet, so the drive enters a different slice of the site index.

    *the input is a window, not a frame*.  an image is one drive vector; a
    cochleagram window is a time series, and it is encoded to one drive vector
    before the dynamics run.  that discards the stimulus's own temporal structure
    inside the window, which is a real limitation and the first thing to revisit if
    this underperforms the ceiling -- driving the dynamics continuously would test
    the substrate harder, at a cost the visual head measured as prohibitive
    (15.8 s/step when one integrator pass was spent per output sample).
    """

    def __init__(self, dyn: CorticalDynamics, n_sensors: int = 306,
                 n_bands: int = 64, stim_len: int = 250, n_times: int = 25,
                 dim: int = 128, hidden: int = 256, read_sites: int = 4096):
        super().__init__()
        self.dyn, self.n_times = dyn, n_times
        # auditory cortex is not the occipital port the visual head drives; take a
        # different slice of the sheet so the two terms do not collide when a
        # single substrate carries both.
        self.port = dyn.n // 8
        self.port_lo = dyn.n // 4
        self.enc = nn.Sequential(
            nn.Conv1d(n_bands, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Flatten(), nn.Linear(128 * ((stim_len + 7) // 8), hidden), nn.GELU())
        self.to_cortex = nn.Linear(hidden, self.port)
        self.read_idx = torch.linspace(0, dyn.n - 1, read_sites).long()
        self.cortex_head = nn.Sequential(nn.Linear(read_sites, 512), nn.GELU(),
                                         nn.Linear(512, dim))
        self.meg_head = nn.Sequential(
            nn.Conv1d(n_sensors, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Flatten(), nn.Linear(128 * ((n_times + 1) // 2), 512), nn.GELU(),
            nn.Linear(512, dim))

    def embed_audio(self, coch, substeps: int = 4, dt: float = 2e-2,
                    n_steps: int | None = None):
        """encode the window, drive the temporal port, run, read the final state."""
        b = coch.shape[0]
        drive = torch.zeros(b, self.dyn.n, device=coch.device)
        drive[:, self.port_lo:self.port_lo + self.port] = self.to_cortex(self.enc(coch))
        s = self.dyn.init_state(b, coch.device)
        w = self.dyn.edge_weights()
        h = dt / substeps
        for _ in range((n_steps if n_steps is not None else self.n_times) * substeps):
            s = self.dyn.step(s, drive, h, w)
        z = self.cortex_head(s[1][:, self.read_idx.to(coch.device)])
        return F.normalize(z, dim=-1), s

    def embed_meg(self, meg):
        return F.normalize(self.meg_head(meg), dim=-1)


class CranialNerveLoop(nn.Module):
    """receptor -> named cranial nerve -> the lobe it projects to -> measured EEG.

    the first materialisation in this file whose input path is DECLARED rather
    than asserted.  every other loop drives `drive[:, :dyn.n // 8]` and calls it
    the occipital port in a comment; the site order is a seeded RNG over a sphere,
    so that slice is an arbitrary eighth of a random point cloud.  here the drive
    enters the sites `cortical_regions` labels occipital (or temporal), through a
    nerve that exists in `ibm/topologies/nerve.py` with a measured length and a
    measured per-fibre-class conduction velocity.

    **the fibre classes are the point, not decoration.**  the optic nerve carries
    three retinal ganglion populations that differ in speed by a factor of three
    -- magnocellular at 20 m/s, parvocellular at 12, koniocellular at 6 -- so over
    50 mm they arrive 2.5, 4.2 and 8.3 ms apart.  a model given one conduction
    delay asserts they arrive together, and then every latency it predicts is
    wrong by whatever the lumping chose.  each class gets its own encoder and its
    own arrival step, and the drive is the sum of what has arrived by then.

    the split is functional as well as temporal: magno is achromatic and
    high-contrast, parvo is chromatic and fine, konio carries blue-yellow.  they
    are encoded from different transforms of the same image rather than from
    three copies of it, so severing one costs a specific thing.

    for hearing the same structure runs over the cochlear nerve into temporal
    cortex, with type I fibres (95% of the nerve, myelinated, 25 m/s) against
    type II (unmyelinated, 3 m/s).

    the target is the measured evoked response, so this is supervised on real
    neural data rather than self-supervised -- which is what makes it a test of
    the pathway rather than of an encoder.
    """

    def __init__(self, dyn: CorticalDynamics, nerve: str = "optic",
                 lobe: str = "occipital", n_sensors: int = 64, n_times: int = 25,
                 dim: int = 128, hidden: int = 256, read_sites: int = 4096,
                 dt: float = 1e-3):
        super().__init__()
        self.dyn, self.nerve, self.lobe, self.n_times = dyn, nerve, lobe, n_times
        from ibm.topologies.nerve import (TRUNK_COMPOSITION, FIBRE_VELOCITY_M_S,
                                          trunk_length_mm)
        if nerve not in TRUNK_COMPOSITION:
            raise ValueError(f"{nerve} is not a declared trunk")
        self.classes = list(TRUNK_COMPOSITION[nerve])
        # the MEASURED route over the body mesh, with the source recorded, rather
        # than the typed trunk length behind a silent 50 mm default.  it moves the
        # optic nerve from 50 mm to 65.7 mm, so the three retinal populations now
        # arrive 3.3, 5.5 and 11.0 ms apart instead of 2.5, 4.2 and 8.3 -- the
        # separation this loop exists to resolve, off by a third.
        length_mm, self.length_source = trunk_length_mm(nerve)
        length_m = length_mm * 1e-3
        # arrival step per class, from the declaration -- not a hyperparameter
        self.delays_s = {c: length_m / FIBRE_VELOCITY_M_S[c][1] for c in self.classes}
        self.arrive = {c: max(0, int(round(d / dt))) for c, d in self.delays_s.items()}
        # the integration step must RESOLVE the delays or the fibre classes are
        # decoration.  the optic nerve spans 2.5-8.3 ms across its three
        # populations; at the dt=2e-2 the other loops use, all three round to
        # step 0 and arrive together -- precisely the lumping this class exists
        # to avoid.  dt=1e-3 separates them into steps 2, 4 and 8.
        span = max(self.delays_s.values()) - min(self.delays_s.values())
        if len(set(self.arrive.values())) < len(self.classes) and span > 0:
            raise ValueError(
                f"dt={dt:g}s cannot resolve {nerve}'s fibre delays "
                f"({', '.join(f'{c}={1000*d:.1f}ms' for c, d in self.delays_s.items())}"
                f") -- they collapse to steps {sorted(set(self.arrive.values()))}. "
                f"use dt <= {span/2:.4g}s or state that the classes are lumped.")
        self.dt = dt
        self.port = region_index(dyn.pos, lobe)
        self.n_port = len(self.port)

        vis = nerve == "optic"
        self.enc = nn.ModuleDict()
        for c in self.classes:
            self.enc[c] = (nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1),
                nn.GELU(), nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(),
                nn.Linear(128 * 8 * 8, hidden), nn.GELU()) if vis else
                nn.Sequential(
                nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.GELU(), nn.Flatten(),
                nn.Linear(128 * 63, hidden), nn.GELU()))
        self.to_cortex = nn.ModuleDict(
            {c: nn.Linear(hidden, self.n_port) for c in self.classes})
        self.read_idx = torch.linspace(0, dyn.n - 1, read_sites).long()
        self.head = nn.Sequential(nn.Linear(read_sites, 512), nn.GELU(),
                                  nn.Linear(512, dim))
        self.eeg_head = nn.Sequential(
            nn.Conv1d(n_sensors, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Flatten(), nn.Linear(128 * ((n_times + 1) // 2), 512), nn.GELU(),
            nn.Linear(512, dim))

    def channels(self, x):
        """split the stimulus into what each fibre class actually carries."""
        if self.nerve != "optic":
            return {c: x for c in self.classes}
        grey = x.mean(1, keepdim=True).expand_as(x)
        out = {}
        for c in self.classes:
            if c == "retinal_magno":
                out[c] = grey                                   # achromatic
            elif c == "retinal_parvo":
                out[c] = x - grey                               # chromatic detail
            else:
                out[c] = torch.stack([x[:, 2] - x[:, :2].mean(1)] * 3, 1)  # blue-yellow
        return out

    def embed_stimulus(self, x, substeps: int = 4, dt: float | None = None,
                       n_steps: int | None = None, drop: str = ""):
        dt = self.dt if dt is None else dt
        # run at least until the slowest class has arrived and propagated
        n_steps = (max(self.arrive.values()) + 8) if n_steps is None else n_steps
        b = x.shape[0]
        ch = self.channels(x)
        pend = {c: self.to_cortex[c](self.enc[c](ch[c]))
                for c in self.classes if c != drop}
        s = self.dyn.init_state(b, x.device)
        w = self.dyn.edge_weights()
        idx = self.port.to(x.device)
        h = dt / substeps
        for step in range(n_steps):
            drive = torch.zeros(b, self.dyn.n, device=x.device)
            # a class contributes only once its conduction delay has elapsed
            arrived = [c for c in pend if self.arrive[c] <= step]
            if arrived:
                drive[:, idx] = sum(pend[c] for c in arrived)
            for _ in range(substeps):
                s = self.dyn.step(s, drive, h, w)
        z = self.head(s[1][:, self.read_idx.to(x.device)])
        return F.normalize(z, dim=-1), s

    def embed_eeg(self, eeg):
        return F.normalize(self.eeg_head(eeg), dim=-1)


class InteroceptiveLoop(nn.Module):
    """visceral afference -> vagus and splanchnics -> cortex -> a bodily readout.

    `CranialNerveLoop` with the body on the input side instead of a stimulus, and
    with several trunks at once instead of one.  The structure it inherits is the
    part that matters: the drive enters the sites `cortical_regions` labels, each
    fibre class gets its own encoder, and a class contributes only once its
    conduction delay has elapsed.

    **This is the loop where the delay is largest and least optional.**  The
    optic nerve's three retinal populations span 2.5-8.3 ms.  Here, over IHM's
    measured 508 mm route from gastric wall to the solitary nucleus, the vagal
    A-beta channel reporting gastric volume arrives at 9 ms and the vagal C
    channel reporting the same meal's nutrient content at 508 ms -- 55x, on one
    nerve -- with the splanchnic report of the same stomach at 168 ms in between.
    Interoception being late relative to touch is not an artifact to smooth away;
    it is why a gut feeling is slow, and a model handed one visceral latency has
    asserted otherwise.  So `dt` here defaults to 1e-2 rather than the 1e-3 the
    cranial loop uses: at 1e-3 the same seven groups would need 508 integration
    steps to let the slowest arrive, and at 1e-2 they land on steps 1, 3, 13, 17,
    22, 23 and 51 -- still all distinct, which is the condition, checked below.

    **The encoder is sized from the body, per group, and never from a default.**
    `SensorimotorLoop`'s `afferent_channels=32` default once silently dropped half
    of IHM's 64 channels: the loop ran, validated, and threw away the input.  Here
    the channel list comes from the corpus, every channel must resolve to a
    declared port, and each group's encoder input width is the count of channels
    that actually landed in that group.  A channel that resolves nowhere raises.

    **The port is a substitution and it is named.**  `ibm.interoception`'s
    `PORT_SUBSTITUTION` says it: insula and anterior cingulate are not separable
    on the six-label spherical proxy, so the drive enters `frontal`.  That is
    printed by `describe()` rather than left in a comment, because the one thing
    this file's history says loudest is that a port named in a comment is not a
    port.

    **The drive is TONIC, and that has a known consequence.**  Visceral afference
    is tonic -- the gut does not stop reporting -- so unlike `SensorimotorLoop`
    this does not present and withdraw the stimulus.  The consequence, measured
    on the motor path and recorded as ledger entry 20, is that a whole-sheet
    readout then samples the driven region, and the decoder can read the input
    rather than the cortex.  That is why `forward` carries `kernel="permuted"` as
    a first-class arm and why the ablation script runs it: with a tonic drive,
    trained-vs-permuted is the only measurement that separates a cortical result
    from a readout result, and it is expected to be unfavourable.
    """

    #: insula plus anterior cingulate as a fraction of the `frontal` label.
    #: the DK parcellation puts insula at ~2.0% of cortical surface and the
    #: rostral and caudal anterior cingulate together at ~2.5%, so the
    #: interoceptive target is ~4.5% of cortex.  `cortical_regions` gives
    #: `frontal` 22% by construction (the cut at yn > 0.55 on a uniform sphere).
    #: driving all of it would put visceral afference onto five times the
    #: cortex the target occupies -- and would give this head 1.6M projection
    #: parameters against a 3,555-sample corpus, which is a second reason.
    PORT_FRACTION = 0.045 / 0.22

    def __init__(self, dyn: CorticalDynamics, channels: list[str], n_out: int,
                 lobe: str | None = None, hidden: int = 64,
                 read_sites: int = 1024, dt: float = 1e-2, substeps: int = 4,
                 port_frac: float | None = None):
        super().__init__()
        import ibm.interoception as IO
        self.dyn, self.dt, self.substeps = dyn, dt, substeps
        self.lobe = IO.PORT_LOBE if lobe is None else lobe
        self.port_substitution = IO.PORT_SUBSTITUTION
        self.channels = list(channels)

        # -- resolve every channel to a declared port, or raise ---------------
        by_key = {p.key: p for p in IO.PORTS}
        unknown = [c for c in self.channels if c not in by_key]
        if unknown:
            raise ValueError(
                f"{len(unknown)} afferent channel(s) resolve to no declared "
                f"visceral port: {unknown}.  known ports: {sorted(by_key)}")
        unsent = [k for k in by_key if k not in self.channels]
        if unsent:
            raise ValueError(
                f"{len(unsent)} declared port(s) receive no channel from the "
                f"body: {unsent}.  a port with no rate is a wire the encoder "
                f"would size for and never see.")

        # -- group by (trunk, fibre class); the group is the arrival unit -----
        self.group_keys: list[tuple[str, str]] = []
        cols: dict[tuple[str, str], list[int]] = {}
        for i, c in enumerate(self.channels):
            p = by_key[c]
            g = (p.trunk, p.fibre)
            if g not in cols:
                cols[g] = []
                self.group_keys.append(g)
            cols[g].append(i)
        if sum(len(v) for v in cols.values()) != len(self.channels):
            raise AssertionError("grouping lost or duplicated a channel")

        self.delays_s = IO.group_delays_s()
        self.arrive = {g: max(0, int(round(self.delays_s[g] / dt)))
                       for g in self.group_keys}
        # the integration step must RESOLVE the delays, or the fibre classes and
        # the trunks are decoration.  same check, same reason, as
        # CranialNerveLoop -- the failure it prevents is a model that asserts
        # gastric touch and gastric chemistry reach cortex together.
        span = max(self.delays_s[g] for g in self.group_keys) - \
            min(self.delays_s[g] for g in self.group_keys)
        if len(set(self.arrive.values())) < len(self.group_keys) and span > 0:
            raise ValueError(
                f"dt={dt:g}s cannot resolve the visceral conduction groups "
                f"({', '.join(f'{t}/{f}={1000*self.delays_s[(t,f)]:.1f}ms' for t, f in self.group_keys)}"
                f") -- they collapse to steps {sorted(set(self.arrive.values()))}. "
                f"use a smaller dt or state that the groups are lumped.")

        for g, idx in cols.items():
            self.register_buffer(f"cols_{g[0]}_{g[1]}",
                                 torch.tensor(idx, dtype=torch.long))
        full = region_index(dyn.pos, self.lobe)
        frac = self.PORT_FRACTION if port_frac is None else port_frac
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"port_frac {frac} must be in (0, 1]")
        want = max(1, int(round(len(full) * frac)))
        # evenly spaced through the lobe's site list rather than a random draw:
        # the site order is a seeded RNG over a sphere, so spacing samples the
        # lobe uniformly and is reproducible without carrying a second seed.
        self.port = full[torch.linspace(0, len(full) - 1, want).long()]
        self.n_port = len(self.port)
        self.port_frac, self.n_lobe = frac, len(full)

        # ONE ENCODER PER GROUP, input width = how many channels landed in it.
        self.enc = nn.ModuleDict()
        self.to_cortex = nn.ModuleDict()
        for g in self.group_keys:
            name = f"{g[0]}_{g[1]}"
            n_in = len(cols[g])
            self.enc[name] = nn.Sequential(
                nn.Linear(n_in, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU())
            self.to_cortex[name] = nn.Linear(hidden, self.n_port)
        self.group_width = {g: len(cols[g]) for g in self.group_keys}

        self.read_idx = torch.linspace(0, dyn.n - 1, read_sites).long()
        self.head = nn.Sequential(nn.Linear(read_sites, 256), nn.GELU(),
                                  nn.Linear(256, n_out))
        self.n_out = n_out
        # ONE FIXED permutation, held as a buffer.  drawing a fresh one every
        # forward would make the permuted arm a different random kernel at every
        # step, which is a weaker and different control: a model can neither
        # exploit nor be hurt by structure that is resampled under it.  ledger
        # entry 20's permuted arm was a fixed shuffle of the trained weights and
        # matched the trained arm to eight decimals, and matching that setup is
        # the point of running it again here.
        self.register_buffer("perm", torch.randperm(dyn.n))

    def n_steps(self) -> int:
        return max(self.arrive.values()) + 8

    def forward(self, x, kernel: str = "trained", drop: tuple = (),
                n_steps: int | None = None, generator=None,
                checkpoint_every: int = 8, lump: bool = False):
        """afferent rates (B, n_channels) -> readout (B, n_out).

        `kernel`:
          trained   the learned association weights
          severed   zeroed -- the dynamics carry nothing between sites
          permuted  the same weights, rows shuffled -- the control that separates
                    "the cortex computed this" from "the decoder read the drive",
                    which on the motor path was the difference between a result
                    and ledger entry 20
        `lump`: every group arrives at step 0.  THE CONTROL FOR THE DELAY
        ITSELF, as distinct from the channels.  dropping a conduction group
        removes what it carries, so a cost there says the channels matter and
        says nothing about when they arrive.  this arm keeps all fifteen
        channels and destroys only the 9.2-508 ms ordering -- which is the
        lumping `ibm/topologies/nerve.py` was written to refuse, run as an
        experiment instead of asserted against.

        `drop`: (trunk, fibre) groups whose contribution is withheld.  dropping
        every C group is the specific ablation this anatomy makes possible: it
        severs the slow unmyelinated arm of visceral afference and keeps the fast
        myelinated one, which is a vagotomy of the chemical report alone.
        """
        if x.shape[1] != len(self.channels):
            raise ValueError(f"body sent {x.shape[1]} channels, the encoder was "
                             f"built for {len(self.channels)}")
        b, dev = x.shape[0], x.device
        n_steps = self.n_steps() if n_steps is None else n_steps
        drop = set(tuple(g) for g in drop)

        pend = {}
        for g in self.group_keys:
            if g in drop:
                continue
            name = f"{g[0]}_{g[1]}"
            idx = getattr(self, f"cols_{name}").to(dev)
            pend[g] = self.to_cortex[name](self.enc[name](x[:, idx]))

        w = self.dyn.edge_weights()
        if kernel == "severed":
            w = torch.zeros_like(w)
        elif kernel == "permuted":
            w = w[self.perm.to(w.device)]
        elif kernel != "trained":
            raise ValueError(f"unknown kernel arm {kernel!r}")

        s = self.dyn.init_state(b, dev)
        port = self.port.to(dev)
        h = self.dt / self.substeps

        def run(lo, hi, *state_and_w):
            st, ww = state_and_w[:4], state_and_w[4]
            for step in range(lo, hi):
                drive = torch.zeros(b, self.dyn.n, device=dev)
                arrived = list(pend) if lump else \
                    [g for g in pend if self.arrive[g] <= step]
                if arrived:
                    # TONIC: an arrived group keeps contributing.  the viscera do
                    # not stop reporting, and a transient visceral drive would be
                    # a different organ.
                    drive = drive.index_copy(
                        1, port, sum(pend[g] for g in arrived))
                for _ in range(self.substeps):
                    st = self.dyn.step(st, drive, h, ww)
            return st

        # GRADIENT CHECKPOINTING, and it is not an optimisation -- without it
        # this loop does not fit.  the association gather is (B, N, k) per
        # substep, 368 MB at 30k sites and batch 64, and this head runs 236
        # substeps against the cranial loop's 64 because the vagal C fibre takes
        # 508 ms to arrive.  retaining all of them is 87 GB and OOMs.  storing
        # one state per chunk and recomputing the chunk in backward trades ~2x
        # compute for ~`checkpoint_every`x memory, which is the only way the
        # slowest fibre class in the body stays in the graph at all.
        if checkpoint_every and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            for lo in range(0, n_steps, checkpoint_every):
                hi = min(lo + checkpoint_every, n_steps)
                s = checkpoint(run, lo, hi, *s, w, use_reentrant=False)
        else:
            s = run(0, n_steps, *s, w)
        return self.head(s[1][:, self.read_idx.to(dev)]), s

    def describe(self) -> str:
        lines = [f"InteroceptiveLoop: {len(self.channels)} channels in "
                 f"{len(self.group_keys)} conduction groups -> {self.n_port:,} "
                 f"sites ({100*self.port_frac:.0f}% of the {self.n_lobe:,} "
                 f"{self.lobe} sites, the insula+ACC share of cortex) -> "
                 f"{self.n_out} outputs",
                 f"  dt {self.dt:g}s x {self.substeps} substeps, "
                 f"{self.n_steps()} steps "
                 f"({1000*self.dt*self.n_steps():.0f} ms of afference)"]
        for g in self.group_keys:
            lines.append(f"  {g[0]:20s} {g[1]:7s} {self.group_width[g]:2d} ch  "
                         f"{1000*self.delays_s[g]:7.1f} ms -> step "
                         f"{self.arrive[g]:3d}")
        lines.append("  port: " + self.port_substitution)
        return "\n".join(lines)


class SensorimotorLoop(nn.Module):
    """afferent rates -> cortex -> descending motor commands.  the body interface.

    every other materialization in this file ends at a measurement -- an EEG
    trace, an embedding, a predicted frame.  this one ends at an ACTION, and it
    is the one a body simulator can close a loop around.

    the contract is IHM-1's, read from `ihm/assembly/peripheral.py` rather than
    invented:

        in   afferent rates keyed by brain target, e.g. brain-rh-postcentral
        out  {'motor_commands': {muscle_id: activation in [0, 1]}}

    IHM validates that dict -- an unknown muscle id or an activation outside
    [0, 1] raises -- so the output layer is sized from its 215 muscle bindings
    and squashed, rather than emitting whatever the decoder feels like.

    **the two strips are different populations of sites.**  afference enters the
    postcentral sites and the command is read from the precentral ones, because
    IHM routes to `brain-{l,r}h-postcentral` and reads from
    `brain-{l,r}h-precentral`.  a single "central" label could not express that,
    which is why `cortical_regions` splits at the sulcus.  the signal has to
    cross from one strip to the other through the association kernel -- which
    means the kernel is load-bearing for movement in the same way the ablations
    have been testing it for perception, and severing it should cost the same way.

    what this does NOT do, stated because it is the obvious thing to assume: it
    has no motor babbling, no reward, and no learned controller.  the weights are
    initialised, not trained -- there is no motor corpus here, and the only way
    to train this is in a closed loop with a body, which is the point of handing
    it to IHM-1.  a fresh materialization emits small activations around 0.5 and
    means nothing by them.
    """

    def __init__(self, dyn: CorticalDynamics, muscles: list[str],
                 afferent_channels: int = 32, hidden: int = 256,
                 read_sites: int = 2048):
        super().__init__()
        self.dyn = dyn
        self.muscles = list(muscles)
        self.n_muscle = len(self.muscles)
        self.sense_idx = region_index(dyn.pos, "postcentral")
        self.motor_idx = region_index(dyn.pos, "precentral")
        self.enc = nn.Sequential(
            nn.Linear(afferent_channels, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.to_cortex = nn.Linear(hidden, len(self.sense_idx))
        # read ONLY precentral: a motor command that could see the whole sheet
        # would not have to route through the kernel, and the kernel is the thing
        # under test.
        # READ THE WHOLE SHEET.  reading only precentral was meant to force the
        # command to cross the kernel from postcentral, making the substrate
        # load-bearing by construction.  measured, that crossing does not happen:
        # only 0.03-0.07% of the driven-region signal reaches precentral, and the
        # ratio scales LINEARLY with association gain (0.0004 at w_assoc 0.5,
        # 0.0049 at 6.0) rather than compounding -- so signal arrives in one weak
        # hop and never propagates.  a readout starved of its input cannot be
        # load-bearing, it can only be silent, which is exactly what every motor
        # ablation here has reported.
        #
        # every loop in this file that WORKS -- VisualContrastiveLoop,
        # AudioContrastiveLoop, PairedNeuralLoop -- reads linspace(0, n-1).  the
        # disjoint-region readout was my design choice and it is what starved
        # this term.  the kernel is still in the path: the drive enters
        # postcentral and the dynamics run before anything is read.
        self.read_idx_full = torch.linspace(0, dyn.n - 1, read_sites).long()
        n_read = read_sites
        self.motor_read = None
        self.dec = nn.Sequential(
            nn.Linear(n_read, hidden), nn.GELU(),
            nn.Linear(hidden, self.n_muscle))

    def forward(self, afferent, n_steps: int = 8, dt: float = 5e-3,
                substeps: int = 2, sever: bool = False, hold: int | None = None):
        """drive is TRANSIENT, and that is what makes the kernel necessary.

        with a persistent drive the readout can simply read it: measured on the
        whole-sheet readout, samples landing in the driven region carry 3,134x
        the variance of the rest, the decoder learns to read the input, and a
        permuted kernel gives MSE identical to the trained one to eight decimals.
        making the drive constant hands the decoder the answer.

        so the stimulus is presented for `hold` substeps and then removed, and the
        command is read after it is gone.  the only route from stimulus to readout
        is then what the dynamics carried forward -- which is what a cortex is
        for, and what the persistent-drive version could never test.  a severed
        kernel now has nothing to propagate and must fail.
        """
        b = afferent.shape[0]
        drive = torch.zeros(b, self.dyn.n, device=afferent.device)
        drive[:, self.sense_idx.to(afferent.device)] = self.to_cortex(self.enc(afferent))
        s = self.dyn.init_state(b, afferent.device)
        w = torch.zeros_like(self.dyn.edge_weights()) if sever else self.dyn.edge_weights()
        h = dt / substeps
        total = n_steps * substeps
        hold = max(1, total // 4) if hold is None else hold
        zero = torch.zeros_like(drive)
        for i in range(total):
            s = self.dyn.step(s, drive if i < hold else zero, h, w)
        m = s[1][:, self.read_idx_full.to(afferent.device)]
        # IHM refuses an activation outside [0, 1], so squash rather than clamp:
        # a clamp hides saturation, a sigmoid reports it as a gradient.
        return torch.sigmoid(self.dec(m)), s

    @torch.no_grad()
    def act(self, afferent_by_region: dict, device="cpu", **kw) -> dict:
        """one step in IHM's protocol: its afferent dict in, its brain_state out."""
        v = torch.zeros(1, self.enc[0].in_features, device=device)
        for i, (_, val) in enumerate(sorted(afferent_by_region.items())):
            if i < v.shape[1]:
                v[0, i] = float(val)
        cmd, _ = self.forward(v.to(device), **kw)
        return {"motor_commands":
                {m: float(cmd[0, i]) for i, m in enumerate(self.muscles)}}


class VisualEvokedLoop(nn.Module):
    """image -> occipital drive -> dynamics -> the EVOKED RESPONSE as it unfolds.

    the most physiologically direct materialisation in the file.  every other head
    reads one cortical state and maps it to one output; here the target is a
    64-channel time series over -0.2 to +0.79 s, and the model produces it by
    running the dynamics and reading the lead field OUT AT EVERY STEP.  the evoked
    response is not something the cortex emits at the end -- it IS the trajectory,
    sampled by the sensors as it happens.

    that makes it the sharpest test of the dynamics available: a model that gets
    the P1/N1 timing right has to have the right time constants, not merely the
    right steady state.  and the target is unusually clean, because THINGS-EEG2
    retains repetitions -- averaging buys sqrt(80) on the test split.

    the lead field is rank-limited for the reason `PairedNeuralLoop` documents: a
    free readout substitutes for the cortex and reaches high accuracy at rank 1.
    64 channels resolve fewer spatial degrees of freedom than 306, so the ceiling
    here is lower still.
    """

    def __init__(self, dyn: CorticalDynamics, n_sensors: int = 64, n_times: int = 100,
                 img: int = 64, hidden: int = 256, read_sites: int = 4096,
                 lead_rank: int = 32):
        super().__init__()
        self.dyn, self.n_sensors, self.n_times = dyn, n_sensors, n_times
        self.port = dyn.n // 8
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden), nn.GELU())
        self.to_cortex = nn.Linear(hidden, self.port)
        self.read_idx = torch.linspace(0, dyn.n - 1, read_sites).long()
        self.lead_u = nn.Linear(read_sites, lead_rank, bias=False)
        self.lead_v = nn.Linear(lead_rank, n_sensors, bias=False)

    def forward(self, img, n_steps: int, dt: float, substeps: int = 8):
        """one output sample per EEG sample, several integrator steps per output.

        the sampling interval and the integrator step are different quantities and
        conflating them diverges.  the EEG is sampled at 100 Hz, so an output is
        due every 10 ms -- but the membrane constant is 15 ms, and an explicit
        Euler step of 10 ms against a 15 ms constant is unstable: measured, |v|
        reached 1.9e6 mV within one forward.  so `dt` is the OUTPUT interval and it
        is integrated in `substeps` pieces.
        """
        b = img.shape[0]
        drive = torch.zeros(b, self.dyn.n, device=img.device)
        drive[:, :self.port] = self.to_cortex(self.enc(img))   # occipital port
        s = self.dyn.init_state(b, img.device)
        w = self.dyn.edge_weights()
        idx = self.read_idx.to(img.device)
        zero = torch.zeros_like(drive)
        h = dt / substeps
        out = []
        # the stimulus arrives at t=0; the first samples are pre-stimulus baseline,
        # so no drive is delivered until the onset index.
        # the target is cropped to begin at stimulus onset, so drive is present
        # from the first sample; there is no baseline phase to withhold it for.
        for t in range(self.n_times):
            d = drive
            for _ in range(substeps):
                s = self.dyn.step(s, d, h, w)
            out.append(self.lead_v(self.lead_u(s[1][:, idx])))
        return torch.stack(out, -1), s        # (B, sensors, time)


class AudioVisualLoop(nn.Module):
    """one cortex, two ports, two predictions -- and the association between them.

    the video loop drives an occipital port and the audio loop a temporal one.
    THE CORTICAL PARAMETERS BETWEEN THEM ARE THE SAME TENSOR.  so a model trained
    here has to explain both streams with one association kernel, and the only way
    to do that better than two independent models is to use the fact that the
    streams are correlated -- which is what learning an occipito-temporal
    association means.

    this is why the audio has to be the movie's OWN soundtrack.  pairing these
    frames with an unrelated audiobook would present two independent streams and
    the correct thing to learn would be that vision and hearing do not interact.

    `cross_modal_weight` reports the mean learned association between the two
    ports, and it is the number that says whether anything was actually learned
    across modalities rather than in each separately.
    """

    def __init__(self, dyn: CorticalDynamics, img: int = 64, n_bands: int = 64,
                 ctx: int = 8, hidden: int = 256):
        super().__init__()
        self.dyn, self.n_bands, self.ctx = dyn, n_bands, ctx
        n = dyn.n
        self.port = n // 8
        self.occ = (0, self.port)                       # occipital: visual drive
        self.tmp = (n // 3, n // 3 + self.port)         # temporal: auditory drive
        self.v_read = (n - self.port, n)                # visual readout
        self.a_read = (n // 2, n // 2 + self.port)      # auditory readout

        self.v_enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden), nn.GELU())
        self.a_enc = nn.Sequential(nn.Flatten(), nn.Linear(n_bands * ctx, hidden),
                                   nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.v_in = nn.Linear(hidden, self.port)
        self.a_in = nn.Linear(hidden, self.port)
        self.v_out = nn.Linear(self.port, hidden)
        self.a_out = nn.Linear(self.port, hidden)
        self.v_dec = nn.Sequential(
            nn.Linear(hidden, 128 * 8 * 8), nn.GELU(), nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))
        self.a_dec = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Linear(hidden, n_bands))

    def forward(self, frame, coch_ctx, n_steps: int, dt: float, drop: str = ""):
        b = frame.shape[0]
        drive = torch.zeros(b, self.dyn.n, device=frame.device)
        if drop != "video":
            drive[:, self.occ[0]:self.occ[1]] = self.v_in(self.v_enc(frame))
        if drop != "audio":
            drive[:, self.tmp[0]:self.tmp[1]] = self.a_in(self.a_enc(coch_ctx))
        s = self.dyn.init_state(b, frame.device)
        w = self.dyn.edge_weights()
        for _ in range(n_steps):
            s = self.dyn.step(s, drive, dt, w)
        r = s[1]
        v = self.v_dec(self.v_out(r[:, self.v_read[0]:self.v_read[1]]))
        a = self.a_dec(self.a_out(r[:, self.a_read[0]:self.a_read[1]]))
        return v, a, s

    @torch.no_grad()
    def cross_modal_weight(self):
        """mean |learned weight| on the occipito-temporal EDGES.

        it must be edges and not pairs.  the first version of this averaged over
        every occipital x temporal PAIR -- 31,250 x 31,250 is a billion of them,
        of which only 187,484 are actual edges -- so the measured signal was
        diluted about five thousand to one and the metric sat at the random
        baseline for 20,000 steps while reporting that nothing had been learned.
        measured on that same checkpoint, restricted to edges:

            occ->tmp edges   |w| = 0.618   mean +0.450   sd 0.556
            random pairs     |w| = 0.139

        4.4x the baseline, spanning [-0.964, +0.964].  the association was there
        the whole time and the metric could not see it.

        magnitude rather than signed mean, because an inhibitory occipito-temporal
        projection is coupling too and averaging signed weights would report a
        strongly coupled push-pull pair as zero.
        """
        e = F.normalize(self.dyn.embed, dim=-1)
        src = torch.arange(self.occ[0], self.occ[1], device=e.device)
        nb = self.dyn.idx[src]                                   # (port, k)
        is_tmp = (nb >= self.tmp[0]) & (nb < self.tmp[1])
        if not bool(is_tmp.any()):
            return float("nan")
        si = torch.arange(src.shape[0], device=e.device).unsqueeze(1).expand_as(nb)[is_tmp]
        w = torch.tanh(2.0 * (e[src[si]] * e[nb[is_tmp]]).sum(-1))
        return float(w.abs().mean())


class VideoLoop(nn.Module):
    """frame -> cortical drive -> dynamics -> cortical state -> next frame."""

    def __init__(self, dyn: CorticalDynamics, img: int = 64, hidden: int = 256,
                 read_sites: int = 4096, readout: str = "linspace"):
        super().__init__()
        self.dyn, self.img = dyn, img
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),      # 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),     # 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),    # 8
            nn.Flatten(), nn.Linear(128 * 8 * 8, hidden), nn.GELU())
        # drive reaches a posterior subset -- the occipital port
        self.n_in = dyn.n // 8
        self.to_cortex = nn.Linear(hidden, self.n_in)
        # READ_IDX IS A BUFFER because WHICH sites are read is part of the
        # trained object and was silently not being saved.
        #
        # it was a plain tensor attribute, so it never entered the state dict.
        # the readout convention then changed from the anterior eighth
        # (`s[1][:, -n//8:]`) to `linspace(0, n-1)`, and every checkpoint trained
        # before that change loads into the new code CLEANLY -- same shape, so
        # load_state_dict reports nothing -- while `from_cortex` is fed an
        # entirely different set of sites than it was fitted on.
        #
        # measured: video_multifilm.pt reads held-out MSE 0.54581 under the
        # current convention and 0.10179 under the one it was trained with, which
        # is its own log's number to five decimals.  video_v6 reads skill -36.50
        # against persistence under the wrong readout and -0.3255 under the right
        # one -- a factor of 112, and the difference between "catastrophic" and
        # "slightly worse than copying the previous frame".
        #
        # READ_SITES IS AN ARGUMENT because it has not always been 4096.  the
        # v6 checkpoint was trained when the readout was `dyn.n // 8`, so it
        # carries a (256, 3750) from_cortex against this default's (256, 4096)
        # and load_state_dict refuses it.  a renderer that hardcodes the default
        # cannot open its own run's checkpoint -- size it from the artifact.
        self.read_sites = read_sites
        # and READ_IDX IS A BUFFER, because WHICH sites are read is part of the
        # trained object and was silently not being saved.  it was a plain tensor
        # attribute, so it never entered the state dict; the convention then moved
        # from the anterior eighth (`s[1][:, -n//8:]`) to `linspace(0, n-1)`, and
        # every checkpoint from before that change loads into the new code
        # CLEANLY -- same shape, so load_state_dict reports nothing -- while
        # from_cortex is fed a completely different set of sites than it was
        # fitted on.  measured: video_multifilm.pt reads held-out MSE 0.54581
        # under the current convention and 0.10179 under the one it was trained
        # with, which is its own log's number to five decimals; video_v6 reads
        # skill -36.50 against persistence the wrong way and -0.3255 the right
        # way, a factor of 112 and the difference between "catastrophic" and
        # "slightly worse than copying the previous frame".
        self.register_buffer(
            "read_idx",
            (torch.arange(dyn.n - read_sites, dyn.n) if readout == "anterior"
             else torch.linspace(0, dyn.n - 1, read_sites)).long())
        self.from_cortex = nn.Linear(self.read_sites, hidden)
        self.dec = nn.Sequential(
            nn.Linear(hidden, 128 * 8 * 8), nn.GELU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))

    def forward(self, frame, n_steps: int, dt: float):
        b = frame.shape[0]
        h = self.enc(frame)
        drive = torch.zeros(b, self.dyn.n, device=frame.device)
        drive[:, :self.n_in] = self.to_cortex(h)
        s = self.dyn.init_state(b, frame.device)
        # edge weights are built ONCE and reused across the integration, which is
        # what `edge_weights` exists for.  this call was left passing dt as the
        # last positional argument when `step` gained `w`, so the video loop had
        # not run since that refactor.
        w = self.dyn.edge_weights()
        for _ in range(n_steps):
            s = self.dyn.step(s, drive, dt, w)
        # READ THE WHOLE SHEET, not the anterior eighth.  measured on an untrained
        # sheet: with the drive entering the first eighth, the across-batch
        # standard deviation of the state at N=8 is 0.003099 in the driven region
        # and 0.000004 in the anterior eighth -- 775x less.  the sheet is active
        # there (|r| ~ 8) but that activity is drive-INDEPENDENT, so the anterior
        # readout was returning intrinsic dynamics uncorrelated with the stimulus.
        # every loop in this file that WORKS reads linspace(0, n-1); every loop
        # that failed read the anterior eighth.
        read = s[1][:, self.read_idx.to(s[1].device)]
        return self.dec(self.from_cortex(read)), s


class VideoViaEEGLoop(nn.Module):
    """frame -> cortex -> SENSOR PROJECTION -> next frame.

    every other head in this file reads the cortical state directly.  this one
    does not: the decoder sees only what a sensor array can see of the cortex --
    `n_sensors` channels through a rank-limited lead field -- and has to
    reconstruct frame t+H from that alone.  the cortical state is never handed to
    it.

    that makes the neural readout **load-bearing for the video task** rather than
    a second head hanging off a shared trunk.  if the projection carries nothing
    about the stimulus, the video prediction fails, and no amount of decoder
    capacity rescues it.  in the ordinary paired setup the neural head can sit at
    chance for 26,000 steps while the video branch trains happily around it, which
    is exactly what m-multi did.

    **what this can and cannot claim.**  nothing here supervises the projection
    against measured EEG -- there is no corpus with simultaneous film and
    recording -- so the 64 channels are not "EEG" in any measured sense.  what is
    load-bearing is the *lead-field readout*: the claim under test is that what
    the sensors can observe of the cortex suffices to continue the film.  calling
    it an EEG prediction without that qualifier would be the kind of claim this
    project keeps having to withdraw.

    two comparisons make the number mean something, and both are cheap:

    *the direct loop* (`VideoLoop`, recon 0.011 at 40k steps) reads the cortical
    state with no bottleneck at all.  it is the upper bound.

    *a random projection of identical width* (`--control random`) replaces the
    learned lead field with a frozen random matrix.  without it, a good result
    would only show that a 64 x T bottleneck is wide enough -- which is a fact
    about the width, not about the readout.  this separates the two.
    """

    def __init__(self, dyn: CorticalDynamics, img: int = 64, n_sensors: int = 64,
                 n_times: int = 8, hidden: int = 256, read_sites: int = 4096,
                 lead_rank: int = 32, control: str = "learned"):
        super().__init__()
        self.dyn, self.n_sensors, self.n_times = dyn, n_sensors, n_times
        self.n_in = dyn.n // 8
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden), nn.GELU())
        self.to_cortex = nn.Linear(hidden, self.n_in)
        self.read_idx = torch.linspace(0, dyn.n - 1, read_sites).long()
        # the lead field is factorised through `lead_rank` for the reason
        # PairedNeuralLoop documents: a free sites->sensors map is ill-conditioned
        # by physics, and an unconstrained one lets the readout substitute for the
        # cortex.  MEG resolves ~60-80 spatial degrees of freedom; 64-channel EEG
        # fewer still.
        self.lead_u = nn.Linear(read_sites, lead_rank, bias=False)
        self.lead_v = nn.Linear(lead_rank, n_sensors, bias=False)
        if control == "random":
            for m in (self.lead_u, self.lead_v):
                m.weight.requires_grad_(False)
        self.control = control
        # the decoder sees ONLY the sensor trace.  this is the whole point of the
        # materialisation and the one thing not to relax if it underperforms.
        self.dec = nn.Sequential(
            nn.Flatten(), nn.Linear(n_sensors * n_times, hidden), nn.GELU(),
            nn.Linear(hidden, 128 * 8 * 8), nn.GELU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))

    def forward(self, frame, n_steps: int, dt: float):
        """run the dynamics, sample the sensors AS THE STATE EVOLVES, decode.

        the sensor trace is read at every step rather than once at the end,
        because an evoked response is not something the cortex emits when it
        finishes -- it IS the trajectory, sampled while it happens.  that also
        gives the decoder a time series rather than a single vector, which is what
        a real recording would hand it.
        """
        b = frame.shape[0]
        drive = torch.zeros(b, self.dyn.n, device=frame.device)
        drive[:, :self.n_in] = self.to_cortex(self.enc(frame))
        s = self.dyn.init_state(b, frame.device)
        w = self.dyn.edge_weights()
        idx = self.read_idx.to(frame.device)
        every = max(1, n_steps // self.n_times)
        trace = []
        for i in range(n_steps):
            s = self.dyn.step(s, drive, dt, w)
            if i % every == 0 and len(trace) < self.n_times:
                trace.append(self.lead_v(self.lead_u(s[1][:, idx])))
        while len(trace) < self.n_times:
            trace.append(trace[-1])
        sensors = torch.stack(trace, -1)                 # (b, n_sensors, n_times)
        return self.dec(sensors), sensors, s


# ---------------------------------------------------------------------------
# the diagnostics that are NOT the loss
# ---------------------------------------------------------------------------

def viability_penalty(v, lo=-90.0, hi=50.0):
    """how far the membrane potential is outside the range a neuron can occupy.

    ONTOLOGY.md §7 defines the alignment relation as viability-manifold
    containment and asks whether minimising the loss drives the model outside its
    own viability set.  measured on run v1: it does.  |v|max drifted 64 -> 430 mV
    over 375 steps while the loss fell, because nothing stopped the encoder from
    driving cortex arbitrarily hard and a 430 mV membrane predicts frames just
    fine.  it is not a brain, so the association weights learned under it mean
    nothing.

    this is the check working, not a surprise: an objective is parasitic on its
    substrate exactly when lowering the loss requires physiology the substrate
    could not sustain.  the penalty is the regularizer that makes the objective
    mutualistic instead, and it is one-sided -- zero cost anywhere inside the
    range, so it constrains nothing the model is entitled to do.
    """
    return (F.relu(v - hi) ** 2 + F.relu(lo - v) ** 2).mean()


def effective_rank(x):
    """(tr C)^2 / tr(C^2) -- ONTOLOGY.md's expansion signal.

    computed over SITES, not over the batch.  the first version of this took the
    covariance across samples, which caps the answer at batch_size - 1: at batch 4
    it could never report more than 3, and "collapse to 1.0" was partly the metric
    reporting its own ceiling.  the quantity we actually want is the dimensionality
    of the cortical state ACROSS THE SHEET -- how many independent directions the
    population is using -- which is a covariance over sites accumulated over
    whatever samples are to hand, and is bounded by the number of sites rather
    than by the batch.
    """
    x = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
    x = x - x.mean(0, keepdim=True)
    n = x.shape[0]
    if n < 2:
        return float("nan")
    c = (x.T @ x) / (n - 1)
    t1 = torch.diagonal(c).sum()
    t2 = (c * c).sum()
    return float((t1 * t1 / t2.clamp_min(1e-12)).item())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", type=int, default=100_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dyn-steps", type=int, default=8)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--frames", default="/home/brandonin/Documents/IBM-1/data/derived/"
                                        "koyaanisqatsi-full/frames_64x64.npy")
    ap.add_argument("--viability-weight", type=float, default=1e-1,
                    help="ONTOLOGY.md §7: cost of leaving the physiological range")
    ap.add_argument("--modality", choices=("video", "audio", "av", "paired"),
                    default="video")
    ap.add_argument("--neural", default="", help="measured neural target, (T, sensors)")
    ap.add_argument("--horizon", type=int, default=8,
                    help="predict t+H, not t+1.  at 25 fps consecutive frames barely "
                         "differ, so a 1-step target is close to an identity map and "
                         "is minimized by COLLAPSING the representation -- measured: "
                         "effective rank fell 2.45 -> 1.04 while loss fell.  a longer "
                         "horizon forces the dynamics to do work (STATE.md 7d)")
    ap.add_argument("--audio-frames", default="")
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--upload-every", type=int, default=1000)
    ap.add_argument("--ckpt", default="", help="where to save weights")
    ap.add_argument("--out", default="out/pretrain_video.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    frames = np.load(a.frames, mmap_mode="r")
    print(f"frames {frames.shape} from {a.frames}", flush=True)

    coch = np.load(a.audio_frames, mmap_mode="r") if a.audio_frames else None
    dyn = CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    neural = np.load(a.neural, mmap_mode="r") if a.neural else None
    # robust rescaling, applied at load so the stored array stays untouched.
    # the array was standardized by a per-run std that MEG artifacts dominate, so
    # 99.9% of values sat under 0.1 and predicting zero scored 0.0065 -- an MSE of
    # 0.05 looked like 95% variance explained and was in fact 8x WORSE than zero.
    megsc = (np.load(a.neural.replace("meg_250hz", "meg_scale"))
             if a.neural and os.path.exists(a.neural.replace("meg_250hz", "meg_scale"))
             else None)
    if a.modality == "paired":
        model = PairedNeuralLoop(dyn, n_bands=frames.shape[-1],
                                 n_sensors=neural.shape[-1]).to(dev)
    elif a.modality == "av":
        model = AudioVisualLoop(dyn, n_bands=coch.shape[-1]).to(dev)
    elif a.modality == "audio":
        model = AudioLoop(dyn, n_bands=frames.shape[-1]).to(dev)
    else:
        model = VideoLoop(dyn).to(dev)
    n_assoc = dyn.embed.numel()
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"association embeddings: {n_assoc:,}   total trainable: {n_tot:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n_frames = frames.shape[0]
    log = {"config": vars(a), "n_params": n_tot, "n_assoc": n_assoc, "steps": []}
    t0 = time.time()

    for step in range(a.steps):
        H, ctx = a.horizon, 8
        if a.modality == "paired":
            ctx = 125                     # 500 ms at 250 Hz: a speech-tracking
                                          # response is spread over 0-400 ms of lag,
                                          # so a 100 ms context cannot carry it
            lim = min(n_frames, neural.shape[0]) - H - 2
            i = np.random.randint(ctx, lim, size=a.batch)
            x = torch.from_numpy(np.stack([frames[j - ctx:j] for j in i])).float().to(dev)
            # APPLY the scale that was loaded 30 lines up.  it was loaded and then
            # never referenced -- a dead variable -- so training minimized MSE
            # against the RAW MEG array, whose mean-square is 4.7e-5.  that made
            # the printed loss look like 0.005 while the model was 160x WORSE than
            # predicting zero in its own units.  the comment at the load site
            # describes exactly this trap and the fix was never wired in.
            yr = np.ascontiguousarray(neural[i + H])
            if megsc is not None:
                yr = np.clip((yr - megsc[0]) / megsc[1], -6.0, 6.0)
            y = torch.from_numpy(yr.astype(np.float32)).to(dev)
            pred, s = model(x, a.dyn_steps, a.dt)
            recon = F.mse_loss(pred, y)
        elif a.modality == "av":
            lim = min(n_frames, coch.shape[0]) - H - 2
            i = np.random.randint(ctx, lim, size=a.batch)
            xv = torch.from_numpy(np.ascontiguousarray(frames[i])).to(dev)
            yv = torch.from_numpy(np.ascontiguousarray(frames[i + H])).to(dev)
            xv = (xv.permute(0, 3, 1, 2).float() / 127.5) - 1.0
            yv = (yv.permute(0, 3, 1, 2).float() / 127.5) - 1.0
            xa = torch.from_numpy(np.stack([coch[j - ctx:j] for j in i])).float().to(dev)
            ya = torch.from_numpy(np.ascontiguousarray(coch[i + H])).float().to(dev)
            pv, pa, s = model(xv, xa, a.dyn_steps, a.dt)
            recon = F.mse_loss(pv, yv) + 0.5 * F.mse_loss(pa, ya)
        elif a.modality == "audio":
            i = np.random.randint(ctx, n_frames - H - 2, size=a.batch)
            x = torch.from_numpy(np.stack([frames[j - ctx:j] for j in i])).float().to(dev)
            y = torch.from_numpy(np.ascontiguousarray(frames[i + H])).float().to(dev)
            pred, s = model(x, a.dyn_steps, a.dt)
            recon = F.mse_loss(pred, y)
        else:
            i = np.random.randint(0, n_frames - H - 2, size=a.batch)
            x = torch.from_numpy(np.ascontiguousarray(frames[i])).to(dev)
            y = torch.from_numpy(np.ascontiguousarray(frames[i + H])).to(dev)
            x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1.0
            y = (y.permute(0, 3, 1, 2).float() / 127.5) - 1.0
            pred, s = model(x, a.dyn_steps, a.dt)
            recon = F.mse_loss(pred, y)
        viab = viability_penalty(s[0])
        loss = recon + a.viability_weight * viab
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 25 == 0 or step == a.steps - 1:
            with torch.no_grad():
                # (batch x subsampled sites) flattened: the covariance is across
                # the 512 retained sites, so the ceiling is 512 rather than batch-1
                # a dedicated measurement pass at a larger batch.  the training
                # batch is 4, and an effective rank estimated from 4 samples has a
                # ceiling of 3 whatever the model is doing -- so measuring on the
                # training batch reports the metric's own ceiling and calls it
                # collapse.  64 samples cost one extra forward per 25 steps and
                # make the number mean something.
                mb = 16
                if a.modality == "paired":
                    jj = np.random.randint(125, min(n_frames, neural.shape[0]) - H - 2, size=mb)
                    mx = torch.from_numpy(np.stack([frames[j-125:j] for j in jj])).float().to(dev)
                    _, ms = model(mx, a.dyn_steps, a.dt)
                elif a.modality == "av":
                    lim = min(n_frames, coch.shape[0]) - H - 2
                    jj = np.random.randint(ctx, lim, size=mb)
                    mv = torch.from_numpy(np.ascontiguousarray(frames[jj])).to(dev)
                    mv = (mv.permute(0, 3, 1, 2).float() / 127.5) - 1.0
                    ma = torch.from_numpy(np.stack([coch[j-ctx:j] for j in jj])).float().to(dev)
                    _, _, ms = model(mv, ma, a.dyn_steps, a.dt)
                elif a.modality == "audio":
                    jj = np.random.randint(ctx, n_frames - H - 2, size=mb)
                    mx = torch.from_numpy(np.stack([frames[j-ctx:j] for j in jj])).float().to(dev)
                    _, ms = model(mx, a.dyn_steps, a.dt)
                else:
                    jj = np.random.randint(0, n_frames - H - 2, size=mb)
                    mx = torch.from_numpy(np.ascontiguousarray(frames[jj])).to(dev)
                    mx = (mx.permute(0, 3, 1, 2).float() / 127.5) - 1.0
                    _, ms = model(mx, a.dyn_steps, a.dt)
                sub = ms[1][:, ::max(dyn.n // 512, 1)].float()
                r_eff = effective_rank(sub)
                vmax = float(s[0].abs().max())
            # a recon loss is not a result.  for CONTINUATION the baseline that
            # matters is persistence -- emitting frame t for frame t+H, which is
            # free -- and on a 25 fps film it is very strong.  measured on the
            # 40k-step video run: recon 0.01235 against persistence 0.00985, i.e.
            # skill -0.25.  the model was 25% WORSE than doing nothing while its
            # loss fell 50x and its clips looked increasingly sharp, because at
            # horizon 8 frame t+H resembles frame t.  reporting recon alone hid
            # that for the whole run, so the baseline is computed here now.
            persist = float("nan")
            if a.modality in ("video", "av"):
                with torch.no_grad():
                    kk = np.random.randint(0, n_frames - H - 2, size=64)
                    pa = torch.from_numpy(np.ascontiguousarray(frames[kk])).to(dev)
                    pb = torch.from_numpy(np.ascontiguousarray(frames[kk + H])).to(dev)
                    f_ = lambda t: (t.permute(0, 3, 1, 2).float() / 127.5) - 1.0
                    persist = float(((f_(pa) - f_(pb)) ** 2).mean())
            rec = {"step": step, "loss": float(loss.detach()),
                   "recon": float(recon.detach()), "viability": float(viab.detach()),
                   "persistence_mse": persist,
                   "skill_vs_persistence": (1 - float(recon.detach()) / persist
                                            if persist == persist else float("nan")),
                   "r_eff": r_eff, "v_absmax": vmax, "grad_norm": float(gn),
                   "sec": round(time.time() - t0, 1)}
            log["steps"].append(rec)
            xm = (model.cross_modal_weight() if a.modality == "av" else float("nan"))
            rec["cross_modal"] = xm
            sk = rec["skill_vs_persistence"]
            print(f"{step:5d}  recon {float(recon):.5f}  SKILLvsPERSIST {sk:+.4f}  "
                  f"viab {float(viab):8.3f}  r_eff {r_eff:7.2f}  "
                  f"|v|max {vmax:7.1f}  xmod {xm:.4f}  "
                  f"{time.time()-t0:6.0f}s", flush=True)
            if not math.isfinite(float(loss)):
                print("DIVERGED", flush=True); break
        if a.ckpt and a.upload_every and step % a.upload_every == 0 and step:
            # SAVE FIRST.  this used to sit after the `ibm.release` import, so a
            # missing PYTHONPATH threw before torch.save ran and a 2000-step run
            # kept nothing at all.  the weights are the artefact; publishing them
            # is a convenience, and a convenience must never be able to destroy
            # the artefact.
            torch.save({"model": model.state_dict(), "step": step,
                        "config": vars(a)}, a.ckpt)
            # the artefact is saved above and must not be endangered by what
            # follows.  the earlier fix moved torch.save ABOVE this import so a
            # throw could not destroy the weights; it still killed the RUN, which
            # is the same lesson one level out -- publishing is a convenience and
            # a convenience must not be able to stop the training either.
            try:
                from ibm.release import CheckpointName, sidecar, upload
            except Exception as e:
                print(f"  [{step}] saved {a.ckpt}; publish unavailable ({e})",
                      flush=True)
                continue
            obj = {"av": "av", "paired": "meg"}.get(a.modality, f"nfh{a.horizon}")
            nm = CheckpointName(modality={"video": "v", "audio": "a", "av": "av",
                                          "paired": "p"}[a.modality],
                                sites=a.sites, embed=a.embed, degree=a.k,
                                objective=obj, viability_weight=a.viability_weight,
                                step=step)
            # this rewrite used to DROP `config`, which the first save above wrote.
            # every loader needs it -- render_predictions.py reads `modality` from
            # it and, finding nothing, defaulted to "av", built an AudioVisualLoop
            # with no cochleagram and threw.  a checkpoint that cannot be loaded is
            # not a checkpoint.
            torch.save({"model": model.state_dict(), "step": step,
                        "name": str(nm), "config": vars(a)}, a.ckpt)
            meta = sidecar(nm, geometry="spherical shell, area-matched to the measured "
                                        "202,437 mm^2 white surface",
                           n_params=n_tot, n_assoc=n_assoc,
                           metrics=log["steps"][-1], config=vars(a))
            try:
                dest = upload(a.ckpt, meta)
                print(f"  uploaded {dest}", flush=True)
            except Exception as e:
                print(f"  upload failed: {e}", flush=True)

    if a.ckpt:
        os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "config": vars(a)}, a.ckpt)
        print(f"wrote {a.ckpt}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
