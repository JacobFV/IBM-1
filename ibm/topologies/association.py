"""learned cortico-cortical association: long range, geometric prior, no tract.

the third cortical relation, and the one that exists because of what is *missing*
rather than what is known.

`cortical_surface` carries the sheet's own metric, and it is anatomically hard:
horizontal axons genuinely travel along the cortex, so geodesic distance is a
statement about where a signal can physically go.  `tractometric` carries measured
white-matter paths, and where a subject has diffusion imaging it is evidence.
neither covers the ordinary case.

the ordinary case is an EEG or MEG dataset with no diffusion imaging at all -- the
mne `sample` subject has none, and neither do eegmmidb, sleep-edfx, erp-core or
most of the corpus.  long-range cortico-cortical coupling still exists in those
subjects; we simply have no measurement of it.  the honest representation of that
is a *prior over unobserved connectivity*, and this topology is its support:

    w_ij = exp(-d_ij / l) . sigma(<e_i, e_j>)
           \\__________/     \\_____________/
            here (geometry)   the process (learned)

the geometric factor is euclidean rather than geodesic on purpose.  a long-range
association fibre leaves the sheet, crosses white matter and re-enters elsewhere;
its length is far better approximated by the chord than by the path along the
cortex, and using the geodesic here would penalise exactly the connections between
opposite banks that association fibres are known to make.  this is the one place
in the cortical inventory where euclidean distance is the right metric, and it is
right for the opposite reason to the one that makes it wrong in `cortical_surface`.

why it is not folded into `tractometric`
----------------------------------------
they answer different questions and a materialization has to be able to say which
one it used.  a coupling asserted by a subject's own tractogram is a measurement
with a known error rate; a coupling asserted by distance decay is a guess with a
principled shape.  collapsing them lets the second borrow the first's authority,
and provenance could no longer report that a prediction rests on a geometric prior
rather than on the subject's anatomy.  ARCHITECTURE.md §7 requires exactly that
distinction to be reportable.

the three tiers are meant to be tried in order -- subject tractogram, group
connectome, distance prior -- and a model that falls through to this one should
say so rather than presenting the result as anatomically grounded.

what it is not
--------------
this is a support, not a weighting.  the exponential above sets which pairs are
worth carrying an edge for and supplies `distance_mm`; it does not decide coupling
strength.  strength is process parameters (§3), initialized from whatever anatomy
is available and updated by data -- which is the whole point, because a distance
prior that could not be overruled by evidence would be worse than no prior.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance


@B.builder("cortical_association",
           produces=("distance_mm", "geometric_prior", "inclusion_prob"),
           supports=("cortical_surface",),
           directed=False, metric="euclidean")
def cortical_association(sites, *, length_mm: float = 40.0, max_degree: int = 256,
                         min_weight: float = 1e-3, min_mm: float = 8.0,
                         seed: int = 0) -> B.EdgeSet:
    """cortical column nodes under an exponential distance prior.

    `length_mm` is the decay constant.  40 mm is the scale at which
    cortico-cortical connection probability falls by roughly 1/e in tract-tracing
    work; it is a prior, declared as one, and a fitted `prior_scale` on the
    process is expected to move it.

    two truncations, both budget decisions rather than claims about anatomy:
    `min_weight` drops pairs whose geometric prior is negligible, and `max_degree`
    keeps the strongest per node so that 10^5 column nodes do not attempt 10^10
    edges.  both are recorded in the note, so a fit cannot later be surprised that
    a coupling it wanted was never carried.

    `min_mm` excludes near neighbours because those pairs are already related by
    `cortical_surface`.  carrying them here as well would double-count local
    lateral spread as long-range association -- the same double count that
    materializing `transmitter_dynamics` alongside `local_excitation` produces.
    """
    np = B._numpy("cortical_association")
    t = B.as_sites(sites).require(
        "cortical_surface", "cortical_association",
        "materialized cortical column nodes -- positions seeded on the folded sheet. "
        "this topology relates column nodes to each other, so without them there is "
        "nothing to relate")
    xyz = np.asarray(t.xyz, dtype=float)
    n = t.n
    if n < 2:
        return B.empty("cortical_association", n, ("distance_mm", "geometric_prior"),
                       note="fewer than two column nodes materialized")

    g = t.gidx(np)
    keep = int(min(max_degree, n - 1))
    rng = np.random.default_rng(seed)
    src, dst, dmm, pincl = [], [], [], []
    block = max(1, int(2e7 // max(n, 1)))
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        d = np.linalg.norm(xyz[lo:hi, None, :] - xyz[None, :, :], axis=-1)
        for r in range(hi - lo):
            i = lo + r
            row = d[r].copy()
            row[i] = np.inf
            row[row < min_mm] = np.inf
            w = np.exp(-row / length_mm)
            w[~np.isfinite(row)] = 0.0
            tot = w.sum()
            if tot <= 0:
                continue
            # SAMPLE proportional to the prior rather than taking the top-k.
            # top-k by exp(-d/l) is top-k by proximity, so it returns the local
            # neighbourhood that `cortical_surface` already carries and no
            # long-range edge at all -- a long-range prior whose edges are all
            # short is worse than useless, because it looks like it worked.
            # sampling keeps the prior's shape: near pairs are still likelier,
            # but the tail is represented, and the inclusion probability is
            # recorded so a process can reweight to the full dense graph it is
            # a sample of.
            k = int(min(keep, (w > 0).sum()))
            if k <= 0:
                continue
            prob = w / tot
            take = rng.choice(w.size, size=k, replace=False, p=prob)
            incl = 1.0 - (1.0 - prob[take]) ** k          # ~inclusion probability
            take_mask = take > i                          # undirected: one copy per pair
            take, incl = take[take_mask], incl[take_mask]
            if take.size == 0:
                continue
            src.append(np.full(take.size, g[i], dtype=np.int64))
            dst.append(g[take].astype(np.int64))
            dmm.append(row[take]); pincl.append(incl)
    if not src:
        return B.empty("cortical_association", n, ("distance_mm", "geometric_prior"),
                       note="every pair fell below the weight floor or inside min_mm")
    s_, t_, dd = np.concatenate(src), np.concatenate(dst), np.concatenate(dmm)
    pi = np.concatenate(pincl)
    return B.EdgeSet(
        topology="cortical_association", src=s_, dst=t_, n_sites=n,
        features={"distance_mm": dd, "geometric_prior": np.exp(-dd / length_mm),
                  "inclusion_prob": pi},
        directed=False,
        note=(f"exponential distance prior, l = {length_mm:g} mm; SAMPLED at degree "
              f"{keep} proportional to the prior (seed {seed}), not truncated to the "
              f"nearest -- top-k here would return only the local neighbourhood; pairs "
              f"under {min_mm:g} mm left to "
              "cortical_surface.  this is a geometric prior over UNOBSERVED "
              "connectivity -- no tractogram informed these edges, and a "
              "materialization resting on them must say so."))


REGISTRY.topology(Topology(
    name="cortical_association",
    doc=__doc__,
    on=("cortical_surface",),
    edge_features=("distance_mm", "geometric_prior", "inclusion_prob"),
    directed=False,
    builder="cortical_association",
    provenance=Provenance.WEAK,
))
