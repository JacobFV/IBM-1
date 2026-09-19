"""Gate the nociceptive receptor and route joins on the brain side.

The receptor itself is transduced in IHM-1 (`ihm/assembly/nociception.py`, gated
by IHM-1's `scripts/gate_nociception.py`).  This gates what IBM-1 owns: the
visceral nociceptor component and the splanchnic rows bound to it, and the fast
and slow pain routes from each of IHM's 1,326 skin patches.

  V1  the component is real        transduction.visceral_nociceptor is registered on
                                   `viscera`, tagged nociceptive, and the registry
                                   reports nothing about it (not dead, not unread)
  V2  the rows are honest          interoception.check() passes: every nociceptive
                                   row binds a nociceptive receptor and vice versa
  V3  V2 can fail                  the same check, handed a splanchnic row bound back
                                   to the baroreceptor, and a row that binds the new
                                   component but is not flagged, must raise both times
  C1  every patch trunk has both   all 1,326 patch trunks carry adelta AND c
  C2  C1 can fail                  a composition with `c` removed from one used trunk
                                   must raise
  C3  A-delta first, everywhere    delay_adelta < delay_c on every patch, at the
                                   typical velocities AND at the worst case (slowest
                                   A-delta 5 m/s against fastest C 2 m/s)
  C4  the two repos agree          IHM's velocity snapshot == nerve.FIBRE_VELOCITY_M_S
                                   for adelta and c; IHM's delay line would otherwise
                                   route on numbers this repo no longer declares
  C5  idempotent                   cutaneous_nociceptive_routes() twice -> equal
  C6  scope on every record        each record carries the schematic-lower-bound scope

Every delay printed is a SCHEMATIC LOWER BOUND (ibm/topologies/ihm_bridge.py).
"""
from __future__ import annotations

import dataclasses
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ibm.fields  # noqa: E402,F401
import ibm.processes  # noqa: E402,F401
from ibm import interoception as IO  # noqa: E402
from ibm.registry import REGISTRY  # noqa: E402
from ibm.topologies import ihm_bridge as B  # noqa: E402
from ibm.topologies import nerve as NV  # noqa: E402

RESULTS: list[dict] = []
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out", "gate_nociception_routes.json")


def record(key, ok, detail, **values):
    RESULTS.append(dict(check=key, verdict="PASS" if ok else "FAIL", detail=detail,
                        **values))
    print(f"  {key:3s} {'PASS' if ok else 'FAIL'}  {detail}")


def main() -> int:
    # -- visceral --------------------------------------------------------------
    cid = IO.VISCERAL_NOCICEPTOR
    comp = REGISTRY.components.get(cid)
    mentions = [str(p) for p in REGISTRY.check() if cid in str(p)]
    ok = (comp is not None and comp.support == "viscera"
          and "nociceptive" in comp.tags and not mentions)
    record("V1", ok, f"{cid}: support {getattr(comp, 'support', None)}, tags "
           f"{sorted(getattr(comp, 'tags', ()))}, registry problems {mentions or 'none'}")

    try:
        summary = IO.check()
        rows = [p for p in IO.PORTS if p.nociceptive]
        bound = all(p.receptor == cid for p in rows)
        record("V2", bound, f"{summary}; {len(rows)} nociceptive rows, all bind {cid}: "
               f"{bound}", rows=[p.channel for p in rows])
    except ValueError as e:
        record("V2", False, f"check() raised: {e}")

    orig = IO.PORTS
    raised = []
    for mutate in (lambda p: dataclasses.replace(p, receptor="transduction.baroreceptor"),
                   lambda p: dataclasses.replace(p, nociceptive=False)):
        IO.PORTS = tuple(mutate(p) if p.channel == "foregut_mechano" else p for p in orig)
        try:
            IO.check()
            raised.append(False)
        except ValueError:
            raised.append(True)
        finally:
            IO.PORTS = orig
    record("V3", all(raised), f"corrupted rows raised: {raised}")

    # -- cutaneous routes ----------------------------------------------------------
    routes = B.cutaneous_nociceptive_routes()
    trunks = sorted({r["trunk"] for r in routes})
    record("C1", len(routes) == 1326,
           f"{len(routes)} patch routes over {len(trunks)} trunks, all carrying "
           f"adelta and c", trunks=trunks)

    victim = trunks[0]
    saved = NV.TRUNK_COMPOSITION[victim]
    B.TRUNK_COMPOSITION[victim] = tuple(c for c in saved if c != "c")
    try:
        B.cutaneous_nociceptive_routes()
        c2 = False
    except ValueError:
        c2 = True
    finally:
        B.TRUNK_COMPOSITION[victim] = saved
    record("C2", c2, f"removing 'c' from {victim!r} made the join raise: {c2}")

    V = NV.FIBRE_VELOCITY_M_S
    typical = all(r["delays_s"]["adelta"] < r["delays_s"]["c"] for r in routes)
    worst = all(r["path_length_m"] / V["adelta"][0] < r["path_length_m"] / V["c"][2]
                for r in routes)
    lead = [r["delays_s"]["c"] - r["delays_s"]["adelta"] for r in routes]
    record("C3", typical and worst,
           f"typical: {typical}; worst case (A-delta {V['adelta'][0]} vs C {V['c'][2]} m/s): "
           f"{worst}; C-minus-A-delta lead min {1e3*min(lead):.1f} ms, median "
           f"{1e3*statistics.median(lead):.1f}, max {1e3*max(lead):.1f} ms")

    ihm = B.load_ihm()
    snap = ihm.get("fibre_velocity_m_s", {}) if ihm else {}
    agree = {c: (snap.get(c, {}).get("low"), snap.get(c, {}).get("typical"),
                 snap.get(c, {}).get("high")) == V[c] for c in B.NOCICEPTIVE_CLASSES}
    record("C4", all(agree.values()), f"IHM snapshot vs nerve.py: {agree}")

    record("C5", routes == B.cutaneous_nociceptive_routes(),
           "cutaneous_nociceptive_routes() called twice")
    record("C6", all(r["delay_scope"] == B.NOCICEPTIVE_DELAY_SCOPE for r in routes),
           "every record carries the schematic-lower-bound scope")

    delays = {c: {"min_ms": 1e3 * min(r["delays_s"][c] for r in routes),
                  "median_ms": 1e3 * statistics.median(r["delays_s"][c] for r in routes),
                  "max_ms": 1e3 * max(r["delays_s"][c] for r in routes)}
              for c in B.NOCICEPTIVE_CLASSES}
    for c, d in delays.items():
        print(f"  {c:6s} delay min {d['min_ms']:7.1f}  median {d['median_ms']:7.1f}  "
              f"max {d['max_ms']:7.1f} ms   (schematic lower bound)")
    failed = [r["check"] for r in RESULTS if r["verdict"] == "FAIL"]
    report = dict(gate="nociceptive receptor and route joins (IBM side)",
                  verdict="FAILED" if failed else "PASSED", failed=failed,
                  checks=RESULTS, delays_ms=delays, delay_scope=B.NOCICEPTIVE_DELAY_SCOPE,
                  closed_gaps=IO.CLOSED_GAPS, open_gaps=IO.ONTOLOGY_GAPS)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
        fh.write("\n")
    print(f"\n{report['verdict']}: {len(RESULTS) - len(failed)} of {len(RESULTS)} "
          f"checks pass; written to out/gate_nociception_routes.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
