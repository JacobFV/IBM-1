#!/usr/bin/env python3
"""How much of the body the brain can actually reach, with the denominators.

The standing correction this answers: **the brain touches the body only through
nerve fibres connected to real muscles and real skin patches, and every one has
to actually be innervated.**  Coverage is a thing to measure.  So this prints
"N of M", never a bare count, for three surfaces at once:

    skin      -- patches of the real integument, and how much of its area
    muscle    -- channels the body exposes, and which ones get spinal arcs
    trunks    -- declared nerves, and which have a route in the body

and it separates three things that a single "innervated" number would merge:

  * a channel that is not a muscle.  IHM's muscle binding list carries tendon
    sheaths, tendons and check ligaments.  A tendon sheath is not a weak motor
    pool; counting it as an uninnervated muscle inflates the gap with things
    that were never muscles.
  * a structure innervated by a CRANIAL nerve.  An extraocular muscle and a
    forehead patch are innervated, and correctly have no spinal segment.  They
    are not gaps; reporting them as such would be as wrong as hiding them.
  * a structure with no declared innervation at all.  This is the only category
    that is a gap, and it is the one to drive to zero.

Run with no arguments.  It reads IHM-1's canonical peripheral.json and
dermatomes.json; if either is missing it says so rather than reporting a partial
number as a whole one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from ibm.anatomy.muscles import INNERVATION                       # noqa: E402
from ibm.processes.cord import SegmentalCord                      # noqa: E402
from ibm.topologies import dermatome as D                         # noqa: E402
from ibm.topologies import ihm_bridge as BR                       # noqa: E402
from ibm.topologies.nerve import (TRUNK_COMPOSITION, TRUNK_ROOTS,  # noqa: E402
                                  TRUNK_TARGET, TRUNK_LENGTH_MM)


def skin(out: dict) -> None:
    c = D.coverage()
    if not c["ihm_available"]:
        print("SKIN: dermatomes.json absent; run IHM-1 "
              "scripts/build_dermatome_patches.py")
        out["skin"] = {"available": False}
        return
    ps = list(D.patches())
    print("SKIN")
    print(f"  {c['patches']} patches over {c['patch_area_m2']:.4f} m2 of the "
          f"{c['skin_exterior_area_m2']:.4f} m2 exterior component "
          f"= {100 * c['area_fraction_of_exterior']:.2f}%")
    print(f"    (the raw skin mesh is {c['skin_raw_mesh_area_m2']:.4f} m2 and "
          f"includes interior and orifice surfaces; it is not the denominator)")
    print(f"  innervated by a declared trunk : {c['patches'] - c['patches_on_undeclared_trunk']} "
          f"of {c['patches']}")
    print(f"  carrying a spinal root level   : {c['patches_with_spinal_root']} "
          f"of {c['patches']}")
    print(f"  cranial territory, no root     : {c['patches_without_spinal_root']} "
          f"of {c['patches']}  ({c['patches_without_spinal_root_area_m2']:.4f} m2, "
          f"trigeminal V1-V3)")
    levels = set(c["root_levels_covered"])
    all_spinal = [f"{a}{i}" for a, n in (("c", 8), ("t", 12), ("l", 5), ("s", 5))
                  for i in range(1, n + 1)]
    absent = [l for l in all_spinal if l not in levels]
    print(f"  dermatomes present             : {len(levels)} of {len(all_spinal)} "
          f"spinal levels; absent {absent or '-'}")
    print("    C1 has no cutaneous territory in life, so its absence is anatomy, "
          "not a gap.")
    fastest = min(ps, key=lambda p: p["path_length_m"])
    slowest = max(ps, key=lambda p: p["path_length_m"])
    print(f"  route length spans {1000 * fastest['path_length_m']:.0f} mm "
          f"({fastest['region']}) to {1000 * slowest['path_length_m']:.0f} mm "
          f"({slowest['region']}); C-fibre latency "
          f"{1000 * fastest['delays_s']['c']:.0f}-"
          f"{1000 * slowest['delays_s']['c']:.0f} ms")
    print(f"  measured dermatome atlas: {c['measured_dermatome_atlas']} "
          f"(assignment is an authored prior over measured geometry)")
    out["skin"] = c


def muscle(out: dict) -> None:
    path = os.path.expanduser(
        "~/Documents/IHM-1/data/derived/canonical/peripheral.json")
    if not os.path.exists(path):
        print("MUSCLE: peripheral.json absent")
        out["muscle"] = {"available": False}
        return
    with open(path) as fh:
        spec = json.load(fh)
    mb = sorted(spec["muscle_bindings"], key=lambda b: b["muscle_id"])
    cord = SegmentalCord(muscles=[b["muscle_id"] for b in mb], muscle_bindings=mb)
    names = {m: n for m, n in zip(cord.muscles, cord.channel_names)}
    n_contractile = cord.n - len(cord.non_contractile)
    arced = n_contractile - len(cord.cranial_no_segment) - len(cord.no_innervation_entry)
    print("\nMUSCLE")
    print(f"  {cord.n} channels in the body's binding list, of which "
          f"{n_contractile} are contractile")
    print(f"    {len(cord.non_contractile)} are not muscles at all "
          f"(tendon sheaths, tendons, check ligaments) and are excluded by name")
    print(f"  under spinal reflex arcs      : {arced} of {n_contractile} "
          f"contractile channels")
    print(f"  cranial, innervated, no segment: {len(cord.cranial_no_segment)} of "
          f"{n_contractile}")
    print(f"  NO DECLARED INNERVATION        : {len(cord.no_innervation_entry)} of "
          f"{n_contractile}")
    if cord.no_innervation_entry:
        for m in cord.no_innervation_entry:
            print(f"      {m}  ({names.get(m) or 'no catalog name'})")
    else:
        print("      -- none; every contractile channel names a trunk and roots")
    cranial = defaultdict(list)
    for m in cord.cranial_no_segment:
        cranial[INNERVATION[cord.mapping_keys[cord.muscles.index(m)]][0]].append(m)
    for nerve, ms in sorted(cranial.items()):
        bare = sorted({re.sub(r"^(left|right) ", "", names.get(m) or m) for m in ms})
        print(f"      {nerve}: {len(ms)} channels -- {', '.join(bare)}")
    seg = int((cord.seg.sum(0) > 0).sum())
    print(f"  segments recruited            : {seg} of 31 declared cord segments")
    print(f"  INNERVATION table             : {len(INNERVATION)} named muscles")
    out["muscle"] = dict(channels=cord.n, contractile=n_contractile, arced=arced,
                         cranial=len(cord.cranial_no_segment),
                         no_entry=len(cord.no_innervation_entry),
                         non_contractile=len(cord.non_contractile))


def trunks(out: dict) -> None:
    cov = BR.coverage()
    print("\nNERVE TRUNKS")
    print(f"  declared here                 : {len(TRUNK_COMPOSITION)}")
    if not cov["ihm_available"]:
        print("  peripheral.json absent; no routes to join")
        out["trunks"] = {"available": False}
        return
    print(f"  with a route in the body      : {len(cov['joined'])} of "
          f"{len(TRUNK_COMPOSITION)}")
    print(f"    declared here with no route : {cov['ibm_only'] or '-'}")
    print(f"    routed but not declared here: {cov['ihm_only'] or '-'}")
    with_roots = [t for t in TRUNK_COMPOSITION if TRUNK_ROOTS.get(t)]
    with_target = [t for t in TRUNK_COMPOSITION if TRUNK_TARGET.get(t)]
    print(f"  with a declared proximal end  : {len(with_roots)} of "
          f"{len(TRUNK_COMPOSITION)} (root levels or a cranial nucleus)")
    print(f"  with a declared distal end    : {len(with_target)} of "
          f"{len(TRUNK_COMPOSITION)}")
    n = BR.assert_measured_lengths()
    rs = list(BR.routes())
    src = Counter(r["length_source"] for r in rs)
    print(f"  routes reading a measured length: {n} of {len(rs)} "
          f"({dict(src)})")
    la = BR.length_agreement()
    print(f"  typed length vs measured route: {la['agree_within_25_percent']} of "
          f"{la['trunks_typed']} typed trunks agree within 25%")
    print(f"    {la['trunks_without_typed_length']} of "
          f"{len(TRUNK_COMPOSITION)} trunks have NO typed length; before this "
          f"they fell through to a bare default")
    worst = [r for r in la["rows"] if r["ratio"]][:5]
    for r in worst:
        print(f"      {r['trunk']:24s} typed {r['typed_mm']:6.0f} mm  measured "
              f"{r['measured_mm']:6.0f} mm  x{r['ratio']:.2f}")
    print(f"    the two are different quantities -- IHM measures receptor to "
          f"relay, {len(TRUNK_LENGTH_MM)} typed entries measure the named trunk "
          f"alone -- so a large ratio means the root and cord segment dominate.")
    out["trunks"] = dict(declared=len(TRUNK_COMPOSITION), joined=len(cov["joined"]),
                         with_roots=len(with_roots), with_target=len(with_target),
                         routes=len(rs), measured=n, agreement=la)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also write the numbers here")
    a = ap.parse_args()
    out: dict = {}
    skin(out)
    muscle(out)
    trunks(out)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
