"""join IHM-1's measured nerve routes to IBM-1's fibre-class axis.

the two projects declare nerves for different reasons and each has what the other
lacks.  this is the join, not a migration.

**what IHM-1 has that this repo does not.**

*sidedness*.  52 nerve ids, which is 26 nerves x 2 sides.  `TRUNK_COMPOSITION`
here has one "median", and a body has two.  every delay this repo computes has
been implicitly bilateral-symmetric because there was no way to say otherwise.

*measured geometry*.  `path_length_m` comes from routing over an actual body
mesh with BodyParts3D entity ids, against the hand-typed millimetres in
`TRUNK_LENGTH_MM`.  compared on the 25 nerves both declare, the limb trunks agree
within about 20% -- median 587 mm against 700, radial 526 against 650, ulnar 609
against 700 -- which is reassuring for both.  the disagreements are systematic
and instructive: plantar nerves differ 4.4x (888 mm against 200) and the ocular
nerves 2.5-3.2x (oculomotor 142 against 45).  that is not one side being wrong.
**IHM measures the whole conduction route from receptor to relay; this repo typed
the length of the named trunk alone.**  for a delay the route is the right
quantity, and this repo's numbers understate it by whatever the root and cord
segment contribute.

*evidence flags on every record*.  `evidence_kind`, `geometry_kind:
schematic_route`, `measured_axon_geometry: false`, and a `limitations` list
saying the centrelines are inferred and not dissected.  `TRUNK_LENGTH_MM` is bare
floats with no provenance -- a reader cannot tell which are measured and which I
typed from memory.  this repo's whole corrections ledger is a list of numbers
that turned out to be compared against the wrong thing, and IHM's schema makes
that class of error visible at the point of use.

**what this repo has that IHM-1 does not.**

one delay per route.  a binding carries `motor_delay_s` and `afferent_delay_s`,
each a single number -- which is exactly the lumping `nerve.py`'s docstring
argues makes every reflex latency wrong: a trunk carries populations whose
velocities span two orders of magnitude, and a group Ia axon and a C fibre in the
same sciatic nerve arrive 5.5 ms and 550 ms after the same event.  IHM's 0.0153 s
motor delay for the obturator is a reasonable alpha-motor number and says nothing
about gamma.

**so the join is: IHM's identity, geometry and evidence as the spine; this
repo's fibre classes as the axis on it.**  one route becomes N delays.  neither
schema is flattened into the other, because neither is a superset -- the product
is what a materialization actually needs.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterator

from ibm.topologies.nerve import (FIBRE_VELOCITY_M_S, TRUNK_COMPOSITION,
                                  TRUNK_LENGTH_MM)

IHM_PERIPHERAL = os.path.expanduser(
    "~/Documents/IHM-1/data/derived/canonical/peripheral.json")

# IHM names a route that this repo splits, or vice versa.  recorded rather than
# silently dropped: `sciatic_tibial` is IHM's fused route for what this repo
# declares as `sciatic` continuing into `tibial`.
ALIAS = {"sciatic_tibial": "sciatic", "sciatic_fibular": "common_fibular"}


def _bare(nerve_id: str) -> tuple[str, str]:
    """'peripheral-nerve-left-median' -> ('median', 'left')"""
    m = re.match(r"peripheral-nerve-(left|right)-(.+)$", nerve_id)
    if not m:
        return nerve_id, "unknown"
    side, name = m.group(1), m.group(2)
    return ALIAS.get(name, name), side


def load_ihm(path: str = IHM_PERIPHERAL) -> dict | None:
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def routes(path: str = IHM_PERIPHERAL) -> Iterator[dict]:
    """every IHM route, carrying its measured length and IBM's fibre classes.

    yields one record per (nerve, side), with:
        name, side, ihm_id, relay_id, path_length_m  -- from IHM
        classes, delays_s                            -- from this repo, per class
        length_source                                -- which length was used
        measured_axon_geometry, evidence_kind        -- IHM's own honesty flags
    """
    d = load_ihm(path)
    if d is None:
        return
    lengths: dict[tuple[str, str], list[float]] = {}
    for rec in d.get("muscle_bindings", []) + d.get("receptor_patches", []):
        key = _bare(rec["nerve_id"])
        lengths.setdefault(key, []).append(float(rec["path_length_m"]))
    for n in d.get("nerves", []):
        name, side = _bare(n["id"])
        classes = TRUNK_COMPOSITION.get(name)
        if not classes:
            continue                     # IHM route this repo has not declared
        seen = sorted(lengths.get((name, side), []))
        if seen:
            L = seen[len(seen) // 2]     # median route, in metres
            src = "ihm_measured_route"
        elif n.get("path_length_m") is not None:
            # THE NERVE RECORD ITSELF CARRIES A ROUTE LENGTH, and it is the
            # field IHM's own `route_contract` names:
            #
            #     length_field:          nerves[].path_length_m
            #     missing_length_policy: error; never silently substitute a
            #                            trunk length
            #
            # this branch did not exist.  every route with no muscle binding and
            # no receptor patch -- which is every VISCERAL route, since the vagus
            # and the splanchnics innervate no muscle and carry no skin patch --
            # fell through to the typed trunk table and reported
            # `ibm_declared_trunk` for a route IHM had measured.  the cost is
            # exactly the quantity this module exists to get right: the vagus is
            # 508 mm end to relay and 350 mm in the typed table, so the C-fibre
            # delay was 350 ms where the route says 508 -- a 158 ms error in the
            # latency that separates visceral sensation from touch.  the greater
            # splanchnic went the other way, 300 mm typed against 168 measured,
            # 79% long.
            L = float(n["path_length_m"])
            src = "ihm_nerve_route"
        else:
            L = TRUNK_LENGTH_MM.get(name, 300.0) * 1e-3
            src = "ibm_declared_trunk"
        yield {
            "name": name, "side": side, "ihm_id": n["id"],
            "relay_id": n.get("relay_id"), "path_length_m": L,
            "length_source": src,
            "classes": list(classes),
            # the product: one route, one delay PER FIBRE CLASS
            "delays_s": {c: L / FIBRE_VELOCITY_M_S[c][1] for c in classes
                         if c in FIBRE_VELOCITY_M_S},
            "measured_axon_geometry": n.get("measured_axon_geometry", False),
            "evidence_kind": n.get("evidence_kind"),
            "geometry_kind": n.get("geometry_kind"),
            "route_kind": n.get("route_kind"),
            "endpoint_label": n.get("endpoint_label"),
            "relay_id": n.get("relay_id"),
        }


def visceral_routes(path: str = IHM_PERIPHERAL) -> dict[str, dict]:
    """The visceral routes only, one record per trunk, sides collapsed.

    IHM declares each visceral trunk twice, once per side, with identical route
    lengths -- the schematic centrelines are mirror images.  A materialization
    that wants one delay per trunk should not have to decide which side to read,
    and should not silently take whichever came first, so this collapses them
    and RAISES if the two sides disagree, which would mean the mirror assumption
    had stopped holding.

    Every record here has `length_source == "ihm_nerve_route"`.  If one does
    not, the join has regressed to the typed trunk table for a route the body
    measured, and the assertion says so rather than letting a 158 ms latency
    error through as a plausible number.
    """
    by_name: dict[str, dict] = {}
    for r in routes(path):
        if r.get("route_kind") != "visceral":
            continue
        prev = by_name.get(r["name"])
        if prev is not None:
            if abs(prev["path_length_m"] - r["path_length_m"]) > 1e-9:
                raise ValueError(
                    f"{r['name']}: left and right route lengths differ "
                    f"({prev['path_length_m']} vs {r['path_length_m']}); the "
                    f"sides can no longer be collapsed")
            continue
        if r["length_source"] != "ihm_nerve_route":
            raise ValueError(
                f"{r['name']}: visceral route length came from "
                f"{r['length_source']!r}, not from IHM's nerves[].path_length_m. "
                f"IHM's route_contract says 'error; never silently substitute a "
                f"trunk length'.")
        by_name[r["name"]] = r
    return by_name


def coverage(path: str = IHM_PERIPHERAL) -> dict:
    """what each side declares that the other does not -- the work list."""
    d = load_ihm(path)
    if d is None:
        return {"ihm_available": False}
    ihm = {_bare(n["id"])[0] for n in d.get("nerves", [])}
    ibm = set(TRUNK_COMPOSITION)
    return {"ihm_available": True, "ihm_nerves": len(ihm), "ibm_trunks": len(ibm),
            "joined": sorted(ihm & ibm),
            "ihm_only": sorted(ihm - ibm),
            "ibm_only": sorted(ibm - ihm)}


if __name__ == "__main__":
    c = coverage()
    if not c["ihm_available"]:
        raise SystemExit(f"IHM peripheral.json not found at {IHM_PERIPHERAL}")
    print(f"IHM nerves {c['ihm_nerves']}  IBM trunks {c['ibm_trunks']}  "
          f"joined {len(c['joined'])}")
    print(f"  IHM-only (no fibre classes here): {', '.join(c['ihm_only']) or '-'}")
    print(f"  IBM-only (no route in the body) : {', '.join(c['ibm_only'])}")
    rs = list(routes())
    print(f"\n{len(rs)} joined routes (nerve x side)")
    print(f"{'nerve':22s} {'side':6s} {'len mm':>7s} {'src':>18s} "
          f"{'fastest':>9s} {'slowest':>9s}")
    for r in sorted(rs, key=lambda x: (x["name"], x["side"]))[:14]:
        ds = r["delays_s"]
        print(f"{r['name']:22s} {r['side']:6s} {1000*r['path_length_m']:7.0f} "
              f"{r['length_source']:>18s} {1000*min(ds.values()):8.1f}ms "
              f"{1000*max(ds.values()):8.1f}ms")
