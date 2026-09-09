"""the declared tract topology, made into something a training loop can use.

`docs/DISCONNECTS.md` row 3.  `ibm/topologies/tract.py` declares tractometric
adjacency -- which cortical regions a fascicle joins, and the conduction delay
that follows from its arc length -- and argues at length that euclidean and
geodesic metrics both get long-range cortical connectivity wrong.  **No training
or evaluation script imported it.**  `CorticalDynamics.__init__` drew its
long-range partners with `torch.randint`, uniformly over all sites.

This module is the missing link between the two: it turns
`data/sources/braingraph-hcp-connectomes` into the `matrix` and `lengths_mm`
arguments that `ibm.topologies.tract.tractometric_matrix` declares it needs, and
then draws long-range partners from them.

**Why the builder is not called directly.**  `tractometric_matrix` expands a
parcel-level connectome to EVERY site pair across every supra-threshold parcel
pair, and its own docstring says the quadratic cost "is the honest signal that a
materialization far finer than the connectome is asking for more than the
connectome has".  At 30,000 sites over 68 parcels that is ~440 sites per parcel
and ~900 connected parcel pairs -- 1.7×10^8 edges, and 10^10 at 250,000 sites.
The pretraining kernel has a fixed per-site long-range budget (`k * long_range`,
typically 12 of 48), so what it needs is a **uniform subsample of exactly that
edge set**, carrying exactly the same per-edge features.  That is what
`draw_partners` returns, and it is a subsample of the declared topology rather
than a different topology.

**What is a weight and what is not.**  `tract.py` refuses `streamline count` as
an edge feature, in terms: it scales with seeding density, path length and
curvature, all properties of the algorithm rather than of the anatomy.  So the
count is used for exactly one thing here -- deciding whether an edge EXISTS in a
subject, which is what the consensus threshold is a threshold on -- and never as
a coupling strength.  Coupling strength stays where ARCHITECTURE.md §3 puts it:
in the process's learned parameters, which for this sheet is the per-site
embedding.

**The consensus threshold is exposed, because the card asks for it.**
`braingraph-hcp-connectomes/card.yaml` singles this out as the source's one
unusually honest property: making the threshold a parameter "makes visible that
the network's existence depends on a chosen frequency cutoff", and sweeping it is
a direct measurement of edge-existence uncertainty.  `consensus()` takes it, and
`edge_existence_curve()` sweeps it.
"""
from __future__ import annotations

import functools
import os
import xml.etree.ElementTree as ET

import numpy as np

from ibm.cortical_sheet import REGIONS, REGION_INDEX

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data/sources/braingraph-hcp-connectomes/raw/"
                         "repeated_10_scale_33")
CACHE = os.path.join(ROOT, "data/derived/braingraph-dk/subject_stack.npz")

_NS = "{http://graphml.graphdrawing.org/xmlns}"

#: the velocity used when no myelination column is available.  it is
#: `ibm/topologies/tract.py`'s own default, quoted rather than re-chosen, and it
#: is not a measurement -- that file says so: "the myelin maps ibm-1 can obtain
#: are ordinal proxies, so the delay is an ordering with a plausible scale rather
#: than a measurement".
DEFAULT_VELOCITY_M_S = 8.0


class TractsMissing(RuntimeError):
    pass


def _parse_one(path: str):
    """(node names in file order, region region matrix of lengths, counts).

    returns `(names, length_mm, n_fibers)` where the matrices are (86, 86) over
    the file's own node order; the caller maps to REGIONS.
    """
    root = ET.parse(path).getroot()
    keys = {k.get("id"): k.get("attr.name")
            for k in root.findall(f"{_NS}key")}
    g = root.find(f"{_NS}graph")
    ids, names, region = [], [], []
    for nd in g.findall(f"{_NS}node"):
        d = {keys[x.get("key")]: x.text for x in nd.findall(f"{_NS}data")}
        ids.append(nd.get("id"))
        names.append(d.get("dn_name"))
        region.append(d.get("dn_region"))
    pos = {i: k for k, i in enumerate(ids)}
    n = len(ids)
    L = np.zeros((n, n), dtype=np.float32)
    C = np.zeros((n, n), dtype=np.float32)
    for ed in g.findall(f"{_NS}edge"):
        a, b = pos[ed.get("source")], pos[ed.get("target")]
        d = {keys[x.get("key")]: x.text for x in ed.findall(f"{_NS}data")}
        length = float(d.get("fiber_length_mean", 0.0) or 0.0)
        cnt = float(d.get("number_of_fibers", 0.0) or 0.0)
        L[a, b] = L[b, a] = length
        C[a, b] = C[b, a] = cnt
    return names, region, L, C


def build_cache(limit: int | None = None, verbose: bool = True) -> str:
    """parse every subject graph once into a (S, 68, 68) stack of the two fields.

    68, not 86: the subcortical nodes are dropped.  they are real and they matter
    for a thalamo-cortical claim, but this sheet has no subcortical sites for
    them to land on -- `ibm/processes/tct.py` holds the thalamus separately -- and
    silently routing a cortico-thalamic fascicle into a cortical site would be
    inventing a connection the connectome does not assert.
    """
    if not os.path.isdir(RAW):
        raise TractsMissing(
            f"{RAW} is missing.  run\n\n"
            "    PYTHONPATH=. .venv/bin/python scripts/fetch_cortical_atlases.py"
            " --only braingraph-hcp-connectomes\n")
    files = sorted(f for f in os.listdir(RAW) if f.endswith(".graphml"))
    if limit:
        files = files[:limit]
    if not files:
        raise TractsMissing(f"no .graphml under {RAW}")

    n = len(REGIONS)
    Ls = np.zeros((len(files), n, n), dtype=np.float32)
    Cs = np.zeros((len(files), n, n), dtype=np.float32)
    order = None
    for s, f in enumerate(files):
        names, region, L, C = _parse_one(os.path.join(RAW, f))
        if order is None:
            # the node order is identical across braingraph files, but that is
            # checked rather than trusted: a permuted node order would silently
            # transpose the connectome onto the wrong parcels, which is exactly
            # ledger row 8's failure (an ordering taken from a reconstruction
            # that happened to be the right length).
            order = [REGION_INDEX.get(nm, -1) for nm in names]
            first_names = names
            missing = sorted({nm for nm, r in zip(names, region)
                              if r == "cortical" and nm not in REGION_INDEX})
            if missing:
                raise ValueError(
                    f"cortical braingraph nodes with no DK label: {missing}")
            order = np.asarray(order)
            keep = order >= 0
            if int(keep.sum()) != n:
                raise ValueError(
                    f"{int(keep.sum())} of {n} DK regions present in {f}")
        elif names != first_names:
            raise ValueError(f"{f} has a different node order from {files[0]}")
        src = np.nonzero(keep)[0]
        dst = order[src]
        Ls[s][np.ix_(dst, dst)] = L[np.ix_(src, src)]
        Cs[s][np.ix_(dst, dst)] = C[np.ix_(src, src)]
        if verbose and (s + 1) % 100 == 0:
            print(f"  {s + 1}/{len(files)}", flush=True)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, length_mm=Ls, n_fibers=Cs,
                        regions=np.asarray(REGIONS), files=np.asarray(files))
    if verbose:
        print(f"  wrote {CACHE}  ({len(files)} subjects)", flush=True)
    return CACHE


@functools.lru_cache(maxsize=1)
def _stack():
    if not os.path.exists(CACHE):
        build_cache()
    d = np.load(CACHE, allow_pickle=False)
    regs = [str(x) for x in d["regions"]]
    if tuple(regs) != REGIONS:
        raise ValueError(
            f"{CACHE} was built against a different region order; delete it "
            "and rebuild")
    return d["length_mm"], d["n_fibers"], len(d["files"])


@functools.lru_cache(maxsize=8)
def consensus(threshold: float = 0.5, velocity_m_s: float = DEFAULT_VELOCITY_M_S):
    """the group connectome at one edge-existence threshold.

    returns a dict with

      `frequency`   (68, 68) fraction of the 1064 subjects in which the edge was
                    reconstructed at all.  this is the quantity the card calls
                    the source's honest property.
      `adjacency`   (68, 68) bool, `frequency >= threshold`.  the topology.
      `length_mm`   (68, 68) mean arc length over the subjects that HAVE the
                    edge -- not over all subjects, which would divide a real
                    length by a count of people who do not have the connection.
      `delay_s`     length / velocity.  the field that makes this topology the
                    only one carrying time.
    """
    L, C, S = _stack()
    have = C > 0
    freq = have.mean(0)
    n_have = have.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        length = np.where(n_have > 0, (L * have).sum(0) / np.maximum(n_have, 1), 0.0)
    A = freq >= threshold
    np.fill_diagonal(A, False)          # a parcel is not a fascicle to itself
    delay = np.where(A, length / (1000.0 * velocity_m_s), 0.0)
    return {
        "n_subjects": int(S), "threshold": float(threshold),
        "velocity_m_s": float(velocity_m_s),
        "frequency": freq.astype(np.float64),
        "adjacency": A,
        "length_mm": length.astype(np.float64),
        "delay_s": delay.astype(np.float64),
        "n_edges": int(A.sum() // 2),
        "density": float(A.sum() / (len(REGIONS) * (len(REGIONS) - 1))),
    }


def edge_existence_curve(thresholds=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) -> list[dict]:
    """density and mean length as the consensus threshold is swept.

    the card asks for exactly this: "sweeping that parameter is a direct
    measurement of edge-existence uncertainty, which is what §39 wants and what a
    single fixed matrix hides".
    """
    out = []
    for t in thresholds:
        c = consensus(float(t))
        A = c["adjacency"]
        out.append({"threshold": float(t), "n_edges": c["n_edges"],
                    "density": c["density"],
                    "mean_length_mm": float(c["length_mm"][A].mean()) if A.any() else 0.0,
                    "mean_delay_ms": float(1e3 * c["delay_s"][A].mean()) if A.any() else 0.0})
    return out


def draw_partners(region_of_site: np.ndarray, n_far: int, seed: int,
                  threshold: float = 0.5,
                  velocity_m_s: float = DEFAULT_VELOCITY_M_S):
    """`n_far` long-range partners per site, constrained to the declared fascicles.

    `region_of_site` is the REGIONS index of every site.  a site in parcel `a`
    draws its partners uniformly among the sites lying in the parcels the
    consensus connectome joins `a` to -- which is a uniform subsample of the edge
    set `tractometric_matrix` would emit for the same sites, at the same
    threshold.  the edge's arc length and delay come from the parcel pair, which
    is all a parcel-level connectome can say and exactly what `tract.py` says it
    should be crude about.

    returns `(partner (N, n_far) int64, delay_s (N, n_far) float32,
              length_mm (N, n_far) float32)`.

    the draw is a numpy `default_rng(seed)` on the CPU.  the `torch.randint` it
    replaces was a device generator, and CLAUDE.md records the consequence: seed
    0 reproduced on the same device only, so a cpu run and a cuda run of the same
    configuration were different graphs.
    """
    c = consensus(threshold, velocity_m_s)
    A, L, D = c["adjacency"], c["length_mm"], c["delay_s"]
    reg = np.asarray(region_of_site, dtype=np.int64)
    n = len(reg)
    rng = np.random.default_rng(seed)

    order = np.argsort(reg, kind="stable")
    sorted_reg = reg[order]
    start = np.searchsorted(sorted_reg, np.arange(len(REGIONS)), "left")
    stop = np.searchsorted(sorted_reg, np.arange(len(REGIONS)), "right")

    partner = np.zeros((n, n_far), dtype=np.int64)
    delay = np.zeros((n, n_far), dtype=np.float32)
    length = np.zeros((n, n_far), dtype=np.float32)
    orphan = 0
    for a in range(len(REGIONS)):
        rows = order[start[a]:stop[a]]
        if not len(rows):
            continue
        # the parcels a's fascicles reach, restricted to those that actually
        # HOLD sites: a target parcel with no site in this materialization is not
        # a partner that exists, and pretending otherwise would put the drive
        # nowhere.
        targets = np.nonzero(A[a])[0]
        targets = targets[(stop[targets] - start[targets]) > 0]
        if not len(targets):
            # no declared partner: the row keeps a SELF-LOOP-FREE fallback of its
            # own parcel's other sites, and it is counted.  silently leaving the
            # row at index 0 would make site 0 a hub with no anatomy behind it.
            orphan += len(rows)
            pool = rows
            pick = pool[rng.integers(0, len(pool), size=(len(rows), n_far))]
            partner[rows] = pick
            continue
        # choose a target parcel with probability proportional to how many sites
        # it holds, then a site uniformly inside it -- which together is a
        # uniform draw over the sites of the union, i.e. over the edges
        # tractometric_matrix would emit.
        sizes = (stop[targets] - start[targets]).astype(np.float64)
        p = sizes / sizes.sum()
        tsel = rng.choice(len(targets), size=(len(rows), n_far), p=p)
        tpar = targets[tsel]
        off = (rng.random((len(rows), n_far)) * sizes[tsel]).astype(np.int64)
        partner[rows] = order[start[tpar] + np.minimum(off, sizes[tsel].astype(np.int64) - 1)]
        delay[rows] = D[a, tpar]
        length[rows] = L[a, tpar]
    return partner, delay, length, {"orphan_sites": int(orphan),
                                    "threshold": float(threshold),
                                    "n_subjects": c["n_subjects"],
                                    "velocity_m_s": float(velocity_m_s)}
