"""collect the four matched arms into one table, with a PAIRED comparison.

The arms differ in the sheet and in the long-range wiring and in nothing else:
same seed, same graph seed, same steps, same data, same schedule. Two questions:

    sphere_slice        vs  surface_slice        -- does the geometry change cost?
    surface_occ_random  vs  surface_occ_tract    -- does the anatomy help?

**Why paired, and why not just the endpoint.** A single held-out pool of 200 has
sd ~2.8 points in this repo's experience, which is docs/LOG.md ledger row 9. The
trainer already averages 8 pools per evaluation, and it draws them from a
generator seeded on the STEP -- so two arms see *the same pools at the same
step*, and the 12 evaluation points of a 1,200-step run are 12 matched pairs.
Comparing the two trajectories point by point uses all of them; comparing only
the endpoints throws eleven twelfths of the evidence away and lands on whichever
step happened to be lucky.

The endpoint is still reported, and so is the DESIGNATED test set number, which
is a different and better-conditioned measurement: 200 images is the whole
designated set, so the pool is not a sample, chance is exactly 0.5% by
construction, and the number is deterministic rather than an estimate.

**What a difference here does and does not mean.** These are 1,200-step runs, a
fifth of the published visual checkpoint's schedule, chosen because four matched
arms had to fit one GPU alongside another job. They are matched, so a difference
between them is real; they are short, so the absolute numbers are not the
programme's retrieval result and must not be quoted as one.
"""
from __future__ import annotations

import argparse
import json
import math
import os

ARMS = ("sphere_slice", "surface_slice", "surface_occ_random", "surface_occ_tract")
PAIRS = (("sphere_slice", "surface_slice", "geometry: sphere -> surface"),
         ("surface_occ_random", "surface_occ_tract",
          "wiring: random -> tract-constrained"))


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def paired(a, b):
    """mean difference over matched evaluation points, with its own sd.

    `a` and `b` are the trainers' step traces.  points are matched on `step`
    because that is what the pools were seeded on; a step present in one trace
    and not the other is dropped rather than aligned by position.
    """
    ma = {r["step"]: r for r in a["steps"]}
    mb = {r["step"]: r for r in b["steps"]}
    steps = sorted(set(ma) & set(mb))
    steps = [s for s in steps if s > 0]        # step 0 is chance in both arms
    d = [mb[s]["top1"] - ma[s]["top1"] for s in steps]
    n = len(d)
    if n < 2:
        return None
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    return {"n_points": n, "steps": steps, "mean_diff": mean,
            "sd_diff": sd, "se": sd / math.sqrt(n),
            "t": mean / (sd / math.sqrt(n)) if sd > 0 else float("inf"),
            "per_step": {str(s): (ma[s]["top1"], mb[s]["top1"]) for s in steps}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/wiring_comparison.json")
    a = ap.parse_args()

    res = {"arms": {}, "pairs": {}}
    print(f"{'arm':22s} {'step':>6s} {'best 8-pool':>12s} {'designated 200':>15s} "
          f"{'frozen':>8s} {'bypass':>8s}")
    for name in ARMS:
        tr = load(f"out/wiring_{name}.json")
        te = load(f"out/things_test_{name}.json")
        if tr is None:
            print(f"{name:22s} {'-- not run --':>44s}")
            continue
        last = tr["steps"][-1]
        row = {"best_top1_8pools": tr.get("best_top1"),
               "final_step": last["step"], "final_top1": last["top1"],
               "final_sd": last.get("top1_sd"), "chance": tr["chance"],
               "geometry": tr["config"].get("geometry"),
               "long_topology": tr["config"].get("long_topology"),
               "port_region": tr["config"].get("port_region")}
        if te:
            row.update({"designated_top1": te["full"]["top1"],
                        "designated_chance": te["chance"],
                        "designated_step": te["step"],
                        "designated_frozen": te["frozen"]["top1"],
                        "designated_no_assoc": te["no_assoc"]["top1"],
                        "designated_bypass": te["bypass"]["top1"]})
        res["arms"][name] = row
        print(f"{name:22s} {row['final_step']:6d} "
              f"{100*(row['best_top1_8pools'] or 0):11.2f}% "
              f"{(100*row['designated_top1']) if te else float('nan'):14.2f}% "
              f"{(100*row['designated_frozen']) if te else float('nan'):7.2f}% "
              f"{(100*row['designated_bypass']) if te else float('nan'):7.2f}%")

    print()
    for x, y, label in PAIRS:
        ta, tb = load(f"out/wiring_{x}.json"), load(f"out/wiring_{y}.json")
        if not (ta and tb):
            print(f"{label}: one arm missing")
            continue
        p = paired(ta, tb)
        res["pairs"][label] = p
        if p is None:
            continue
        print(f"{label}")
        print(f"  paired over {p['n_points']} matched evaluation points "
              f"(same pools, same step)")
        print(f"  mean difference {100*p['mean_diff']:+.2f} points, "
              f"sd {100*p['sd_diff']:.2f}, se {100*p['se']:.2f}, "
              f"t = {p['t']:+.2f}")
        # the honest reading, spelled out rather than left to a t value
        if abs(p["t"]) < 2.0:
            print("  -> indistinguishable at this run length.  a null, and the "
                  "run length is the reason to be careful about calling it one.")
        else:
            print(f"  -> {y} is {'better' if p['mean_diff'] > 0 else 'worse'}, "
                  "consistently across the trajectory")
        ea = load(f"out/things_test_{x}.json")
        eb = load(f"out/things_test_{y}.json")
        if ea and eb:
            n = ea["n"]
            da, db = ea["full"]["top1"], eb["full"]["top1"]
            print(f"  designated test set: {100*da:.2f}% -> {100*db:.2f}% "
                  f"({round(da*n)} -> {round(db*n)} of {n} images, "
                  f"chance {100*ea['chance']:.2f}%)")

    # the transport measurements, if they have been run
    for name in ("surface_occ_random", "surface_occ_tract"):
        for kind in ("hop", "convergence", "disjoint"):
            d = load(f"out/{kind}_{name}.json")
            if d is not None:
                res.setdefault("transport", {}).setdefault(name, {})[kind] = d
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
