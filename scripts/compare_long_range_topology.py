"""tract-declared long-range edges against random ones, on the graph alone.

`docs/DISCONNECTS.md` row 3.  The long-range partners of every site were drawn
with `torch.randint` while `ibm/topologies/tract.py` declared which regions a
fascicle actually joins.  This script compares the two **as supports**, before
any weight is learned on them, which is the only place the comparison is clean:

  the association weight factorises as
      w_ij  =  M[pi(i), pi(j)]  x  exp(-d_ij / l)  x  sigma(<e_i, e_j>)
               ^ tractography      ^ geometry        ^ learned
  and this script varies ONLY the first factor.  the second is the same held
  `geo` in both arms -- the long-range prior is flat by construction, so the row
  L1 is identical -- and the third is left out entirely.

**The control is matched by construction, not by fitting.**  Both arms give every
site exactly `n_far` long-range edges and give every one of them the same flat
prior, so edge count and row L1 agree to floating point.  The arms differ in
where the edges GO and in nothing else.  `--long-topm` additionally reproduces
the concentration the previous session measured as worth 209x, because CLAUDE.md
is right that a strong control is the one that matters: anatomy has to beat
concentrated-random, not merely plain-random.

**What is measured.**  Three things, none of which needs a trained kernel:

*reach* -- BFS hop distance on the real message-passing graph, from a named
    region to a named region.  `association` computes `(r[:, idx] * w).sum(-1)`,
    so information flows j -> i exactly when j appears in ROW i; the frontier of
    a source set is the set of rows containing it, which is not `idx[source]`.
    This is the same direction convention `measure_hop_transfer.py` uses and
    gates.

*linear transport* -- the held nonnegative operator `geo` applied h times to an
    indicator of the drive region, reporting the mass that lands in the read
    region.  This is an upper bound on what any learned kernel can transport
    through this support with these magnitudes, and it is the quantity that
    a topology can change while the row L1 stays fixed.

*spatial diversity* -- how many distinct DK parcels a site's long-range budget
    lands in.  The previous session measured that spreading the budget raises
    multimodal convergence, and the tract topology either spreads it or does not;
    that is a property of the anatomy and worth reading off rather than assuming.

Two gates, per CLAUDE.md's "check a metric against a case whose answer you know":

  a severed operator must transport exactly 0 at every hop >= 1;
  hop 0 must return the whole indicator, i.e. transport 1.0.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import torch

_SP = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(_SP)
_SP.loader.exec_module(P)


def bfs_hops(idx, source, n, max_hop, device):
    """hop distance from `source`, expanding ROWS THAT CONTAIN the frontier.

    returns an (n,) int tensor, `max_hop + 1` for unreached.
    """
    hop = torch.full((n,), max_hop + 1, dtype=torch.long, device=device)
    seen = torch.zeros(n, dtype=torch.bool, device=device)
    seen[source] = True
    hop[source] = 0
    frontier = seen.clone()
    for h in range(1, max_hop + 1):
        # a row is reached if any of its k listed neighbours is in the frontier
        reached = frontier[idx].any(1) & ~seen
        if not reached.any():
            break
        hop[reached] = h
        seen |= reached
        frontier = reached
    return hop


def linear_transport(geo, idx, drive, read, n, hops, device, sever=False):
    """mass arriving in `read` after h applications of the nonnegative support.

    `x <- (x[idx] * geo).sum(-1)` is exactly `association` with the learned
    factor replaced by 1, i.e. the operator's own gain.  the input is a unit-mass
    indicator on `drive`, so the reported number is a fraction and needs no
    external scale.
    """
    w = torch.zeros_like(geo) if sever else geo
    x = torch.zeros(n, device=device)
    x[drive] = 1.0 / len(drive)
    out = [float(x[read].sum())]
    for _ in range(hops):
        x = (x[idx] * w).sum(-1)
        out.append(float(x[read].sum()))
    return out


def arm(name, n, k, long_range, device, seed, **kw):
    dyn = P.CorticalDynamics(n, 8, k, device, long_range=long_range,
                             graph_seed=seed, **kw)
    return name, dyn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--tract-threshold", type=float, default=0.5)
    ap.add_argument("--long-topm", type=int, default=4,
                    help="the concentration arm: keep the m strongest long-range "
                         "edges per row and redistribute the row's long-range L1 "
                         "onto them.  0 disables it.  this is the STRONG control")
    ap.add_argument("--pairs", default="occipital>precentral,occipital>temporal,"
                                       "postcentral>precentral,occipital>insula")
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--max-hop", type=int, default=6)
    ap.add_argument("--out", default="out/long_range_topology.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arms = [
        ("random", dict(long_topology="random"), 0),
        ("tract", dict(long_topology="tract", tract_threshold=a.tract_threshold), 0),
    ]
    if a.long_topm:
        # THE STRONG CONTROL.  the previous session measured that concentrating a
        # row's long-range budget onto m partners buys 209x transport, and
        # CLAUDE.md is right that this is what anatomy has to beat: a comparison
        # against plain-random alone would credit the topology with a gain that
        # concentration supplies on its own.
        #
        # the model's own `long_topm` ranks the surviving edges by |learned
        # weight|, and this script has no learned weights -- ranking noise would
        # be a ranking of nothing.  so concentration here is structural: keep the
        # FIRST m long-range columns and scale their prior by n_far/m.  every
        # long-range prior entry is equal by construction, so that preserves the
        # row L1 exactly, which is the only property the comparison needs.
        arms += [("random_topm", dict(long_topology="random"), a.long_topm),
                 ("tract_topm", dict(long_topology="tract",
                                     tract_threshold=a.tract_threshold), a.long_topm)]

    res = {"config": vars(a), "device": dev, "arms": {}}
    ref = None
    for name, kw, topm in arms:
        print(f"=== {name}", flush=True)
        dyn = P.CorticalDynamics(a.sites, 8, a.k, dev, long_range=a.long_range,
                                 graph_seed=a.graph_seed, geometry="surface",
                                 **kw).to(dev)
        n_loc = a.k - dyn.n_far
        if topm:
            with torch.no_grad():
                keep = n_loc + topm
                g = dyn.geo.clone()
                g[:, keep:] = 0.0
                g[:, n_loc:keep] *= dyn.n_far / topm
                dyn.geo.copy_(g)
        far = dyn.idx[:, n_loc:n_loc + topm] if topm else dyn.idx[:, n_loc:]
        d_far = (dyn.pos[far] - dyn.pos[:, None, :]).norm(dim=-1)
        reg = P.cortical_regions(dyn.pos)
        far_reg = reg[far]
        n_parcels = torch.tensor(
            [len(torch.unique(r)) for r in far_reg[:2000].cpu()]).float()

        row_l1 = dyn.geo.abs().sum(1)
        entry = {
            "n_edges": int(dyn.idx.numel()),
            "n_effective_long_edges_per_site": int(topm or dyn.n_far),
            "n_far_per_site": int(dyn.n_far),
            "row_l1_mean": float(row_l1.mean()),
            "row_l1_sd": float(row_l1.std()),
            "long_range_row_l1_mean": float(dyn.geo[:, n_loc:].abs().sum(1).mean()),
            "partner_distance_mm": {
                "mean": float(d_far.mean()), "median": float(d_far.median()),
                "p10": float(d_far.flatten().quantile(0.10)),
                "p90": float(d_far.flatten().quantile(0.90))},
            "distinct_parcels_per_site": {
                "mean": float(n_parcels.mean()),
                "max_possible": int(topm or dyn.n_far)},
            "reach": {}, "transport": {},
        }
        if hasattr(dyn, "tract_note"):
            entry["tract_note"] = dyn.tract_note

        for spec in a.pairs.split(","):
            src_name, dst_name = spec.split(">")
            src = P.region_index(dyn.pos, src_name).to(dev)
            dst = P.region_index(dyn.pos, dst_name).to(dev)
            hop = bfs_hops(dyn.idx, src, a.sites, a.max_hop, dev)
            h_dst = hop[dst]
            entry["reach"][spec] = {
                "n_source": int(len(src)), "n_read": int(len(dst)),
                "min_hop": int(h_dst.min()), "median_hop": float(h_dst.float().median()),
                "frac_unreached": float((h_dst > a.max_hop).float().mean()),
                "hop_histogram": {str(h): int((h_dst == h).sum())
                                  for h in range(a.max_hop + 2)},
            }
            t = linear_transport(dyn.geo, dyn.idx, src, dst, a.sites, a.hops, dev)
            t0 = linear_transport(dyn.geo, dyn.idx, src, dst, a.sites, a.hops, dev,
                                  sever=True)
            # GATE 1: read the DRIVE region at hop 0 and the answer must be
            # exactly 1.0 -- the whole indicator is there and nothing has moved.
            # (reading the DESTINATION at hop 0 is 0 whenever the two regions are
            # disjoint, which is the point of the pair, so it is not the gate.)
            self0 = linear_transport(dyn.geo, dyn.idx, src, src, a.sites, 0, dev)
            assert abs(self0[0] - 1.0) < 1e-5, f"hop-0 gate failed: {self0[0]}"
            # GATE 2: a severed operator must move exactly nothing.
            assert all(abs(x) < 1e-12 for x in t0[1:]), f"severed gate failed: {t0}"
            entry["transport"][spec] = {"by_hop": t, "severed": t0}
            print(f"  {spec:32s} min hop {int(h_dst.min())}  "
                  f"transport {['%.3e' % x for x in t[1:]]}", flush=True)
        res["arms"][name] = entry
        if ref is None:
            ref = entry
        else:
            # MATCHING CHECK.  the comparison is only about destinations if the
            # edge count and the row L1 agree; if they do not, the arms differ in
            # gain and the whole measurement is the wrong-thing-compared failure
            # docs/LOG.md keeps a ledger of.
            assert entry["n_edges"] == ref["n_edges"], "edge count not matched"
            assert abs(entry["row_l1_mean"] - ref["row_l1_mean"]) < 1e-6, \
                (entry["row_l1_mean"], ref["row_l1_mean"])
        del dyn
        if dev == "cuda":
            torch.cuda.empty_cache()

    def ratio(num, den):
        return {k: (res["arms"][num]["transport"][k]["by_hop"][-1] /
                    max(res["arms"][den]["transport"][k]["by_hop"][-1], 1e-30))
                for k in res["arms"][den]["transport"]}

    r, t = res["arms"]["random"], res["arms"]["tract"]
    res["summary"] = {
        "edge_count_matched": all(v["n_edges"] == r["n_edges"]
                                  for v in res["arms"].values()),
        "row_l1_matched_to": max(abs(v["row_l1_mean"] - r["row_l1_mean"])
                                 for v in res["arms"].values()),
        "ratios_tract_over_random": ratio("tract", "random"),
    }
    if "tract_topm" in res["arms"]:
        res["summary"]["ratios_concentration_random"] = ratio("random_topm", "random")
        res["summary"]["ratios_tract_over_concentrated_random"] = \
            ratio("tract_topm", "random_topm")
        res["summary"]["ratios_concentrated_tract_over_random"] = \
            ratio("tract_topm", "random")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nmatched: edges {res['summary']['edge_count_matched']}, "
          f"row L1 to {res['summary']['row_l1_matched_to']:.3e}")
    for label, key in (("tract / random", "ratios_tract_over_random"),
                       ("concentration alone (random_topm / random)",
                        "ratios_concentration_random"),
                       ("tract / CONCENTRATED random",
                        "ratios_tract_over_concentrated_random")):
        if key not in res["summary"]:
            continue
        print(f"\n{label}, at hop {a.hops}:")
        for k, v in res["summary"][key].items():
            print(f"  {k:32s} {v:8.3f}x")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
