"""how many distinct senses can reach ONE precentral site in one hop?

transport and integration are different requirements and they pull the kernel in
opposite directions.  `measure_hop_transfer.py` shows arrival is (row gain) x
(fraction of the row's weight mass on the source region), so the way to raise
arrival without touching the row gain is to CONCENTRATE the mass -- and the
limit of concentration, one surviving long-range edge per site, is a wire.  a
site with one long-range input can receive one modality intact and can never
combine two.

so this counts the thing concentration destroys.  for every readout site it asks
how many of the entry regions it holds a direct long-range edge from, under a
given `long_topm`, and reports the distribution.  no dynamics are integrated --
this is a property of the GRAPH and the surviving weights, and it is the ceiling
on what any amount of gain could deliver in one hop.

read it against the transport numbers, not instead of them: the pair is a
trade-off curve, and the right operating point is the one that clears the
transport threshold with the smallest loss of convergence.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import torch

_SP = importlib.util.spec_from_file_location(
    "hoptx", os.path.join(os.path.dirname(__file__), "measure_hop_transfer.py"))
H = importlib.util.module_from_spec(_SP)
_SP.loader.exec_module(H)
P = H.P

ENTRY = {"sight": "occipital", "hearing": "temporal", "touch": "postcentral"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/ibm1_curriculum16.pt")
    ap.add_argument("--head", default="visual_eeg")
    ap.add_argument("--read-region", default="precentral")
    ap.add_argument("--topms", default="0,1,2,3,4,6")
    ap.add_argument("--min-dist", type=float, default=0.0,
                    help="mm; > 0 switches top-m to the greedy spatially-"
                         "diverse rule.  task-blind: it never names a region.")
    ap.add_argument("--out", default="out/convergence_capacity.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dyn, ck = H.build(a.ckpt, dev, a.head)
    read = P.region_index(dyn.pos, a.read_region).to(dev)
    n_loc = dyn.k - dyn.n_far
    ports = {m: P.region_index(dyn.pos, r).to(dev) for m, r in ENTRY.items()}
    member = {}
    for m, p in ports.items():
        v = torch.zeros(dyn.n, dtype=torch.bool, device=dev)
        v[p] = True
        member[m] = v

    print(f"{a.ckpt}: {dyn.n} sites, {dyn.n_far} long-range edges per site, "
          f"readout {a.read_region} ({len(read)} sites); "
          f"min_dist={a.min_dist} mm")
    print(f"\n{'topm':>5s} {'live long edges':>16s} "
          f"{'>=1 modality':>13s} {'>=2':>8s} {'all 3':>8s} "
          f"{'mean modalities':>16s}")
    res = {"ckpt": a.ckpt, "read": a.read_region, "n_read": len(read),
           "min_dist": a.min_dist, "rows": {}}
    for tm in [int(x) for x in a.topms.split(",")]:
        dyn.long_topm = tm
        dyn.long_min_dist = a.min_dist
        w = dyn.edge_weights().detach()[read, n_loc:]          # (R, n_far)
        live = w != 0
        src = dyn.idx[read, n_loc:]                            # (R, n_far)
        cnt = torch.zeros(len(read), device=dev)
        for m in ENTRY:
            hit = (member[m][src] & live).any(dim=1)
            cnt += hit.float()
        row = {"live_edges": float(live.sum(1).float().mean()),
               "ge1": float((cnt >= 1).float().mean()),
               "ge2": float((cnt >= 2).float().mean()),
               "ge3": float((cnt >= 3).float().mean()),
               "mean_modalities": float(cnt.mean())}
        print(f"{tm:5d} {row['live_edges']:16.2f} {100*row['ge1']:12.1f}% "
              f"{100*row['ge2']:7.1f}% {100*row['ge3']:7.1f}% "
              f"{row['mean_modalities']:16.3f}")
        res["rows"][str(tm)] = row
    print("\ntopm=0 is the unmodified kernel.  a readout site needs >= 2 to "
          "combine two senses in ONE hop, and >= 3 for all three; anything less "
          "has to route through a second hop, which costs another factor of the "
          "per-hop transfer.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
