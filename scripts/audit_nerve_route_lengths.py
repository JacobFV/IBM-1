"""audit: are IHM-1's nerve route lengths measured in the wrong frame?  and are they right?

    PYTHONPATH=. .venv/bin/python scripts/audit_nerve_route_lengths.py

**why.**  IHM-1 has two body atlases in two frames -- the Z-Anatomy display export in a
normalised +-1 box, the BodyParts3D canonical body in metres -- and on 17 September a
LOG line claiming they "share a frame" was withdrawn at a 15.7% scale error.  every
conduction delay this repo computes is `path_length_m / velocity`, with the length read
from IHM's `peripheral.json` (`ihm_bridge.routes`, `interoception.visceral_routes`) or
`dermatomes.json` (`dermatome.patches`).  if any of those lengths were measured on the
normalised atlas, every delay built on it is off by a uniform ~1.157x.

**what it checks, in order.**

1. the frame each file DECLARES, and the frame its builder actually reads geometry from
   (`IHM-1/scripts/build_body_peripheral.py`, `enrich_peripheral_routes.py`,
   `build_dermatome_patches.py` -- all read `canonical/anatomy.json`, frame
   `bodyparts3d-display-m`).  the frame is then tested, not trusted: skin height and
   long-bone lengths must come out adult-sized in metres.
2. an INSTRUMENT check: IHM's own length method is re-implemented here and must
   reproduce every declared left-side nerve length exactly, before anything it says
   about a changed relay is believed.
3. a LANDMARK test: each route's endpoint against the Z-Anatomy nerve centreline that
   serves it, and each AUTHORED coordinate (relays, visceral endpoints) against the
   structure it names.  a frame error is a common factor (~0.865 or ~1.156) across all
   of them; anything else is not a frame error.
4. KNOWN ANSWERS: same-scope lengths along the Z-Anatomy centrelines (which the site
   export verified lands on the canonical femur) and literature values.

the Z-Anatomy vagus is not in the site's extraction (it is filed under cranial nerves,
which `blender_zanatomy_nerves.py` excludes for licence scope), so this file ALSO runs
inside blender to pull the vagus, sciatic, tibial, plantar and median centrelines for
measurement only; nothing it extracts is shipped.

findings: docs/LOG.md 18 September 2026, and IHM-1 docs/BODY_PERIPHERAL.md.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CACHE = HERE.parents[1] / ".cache" / "zanatomy_nerve_audit.json"
OUT = HERE.parents[1] / "out" / "nerve_route_audit.json"


# --------------------------------------------------------------------- blender half
def _extract(out_path: str) -> None:
    import bpy
    pat = re.compile(r"vagus|sciatic|tibial nerve|plantar nerve|median nerve|sympathetic trunk", re.I)
    out = {"frame": "z-anatomy-blender-world", "curves": [], "meshes": {}}
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.data.objects:
        if o.type == "MESH" and o.name in ("Femur.l",):
            vs = [o.matrix_world @ v.co for v in o.data.vertices]
            out["meshes"][o.name] = [[min(v[i] for v in vs) for i in range(3)],
                                     [max(v[i] for v in vs) for i in range(3)]]
        if o.type != "CURVE" or not pat.search(o.name) or "artery" in o.name.lower() \
                or "vein" in o.name.lower():
            continue
        cu = o.data
        keep = (cu.bevel_depth, cu.extrude, cu.bevel_object)
        cu.bevel_depth, cu.extrude, cu.bevel_object = 0.0, 0.0, None
        dg.update()
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        out["curves"].append({"name": o.name, "v": [list(o.matrix_world @ v.co) for v in me.vertices],
                              "e": [list(e.vertices) for e in me.edges]})
        ev.to_mesh_clear()
        cu.bevel_depth, cu.extrude, cu.bevel_object = keep
    Path(out_path).write_text(json.dumps(out))
    print("audit extraction:", len(out["curves"]), "curves")


try:
    import bpy  # noqa: F401
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# --------------------------------------------------------------------- analysis half
def main() -> int:
    import numpy as np
    sys.path.insert(0, str(HERE.parent))
    import export_site_body as X          # constants and helpers; its main() is not run

    IHM = X.IHM
    CAN = IHM / "data/derived/canonical"
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("extracting Z-Anatomy centrelines with blender (one-off, cached, CPU)...", flush=True)
        subprocess.run(["blender", "-b", str(X.BLEND), "--python", str(HERE), "--", str(CACHE)],
                       check=True, capture_output=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    ex = json.loads(CACHE.read_text())
    rec: dict = {"frame": {}, "instrument": {}, "landmarks": [], "authored": [],
                 "known_answers": [], "relays_on_cord": []}

    def dump():                            # the cheap summary goes out first, and often
        OUT.write_text(json.dumps(rec, indent=1, default=lambda o: getattr(o, "tolist", str)()))

    anat = json.loads((CAN / "anatomy.json").read_text())
    E = {e["id"]: e for e in anat["entities"]}
    per = json.loads((CAN / "peripheral.json").read_text())
    der = json.loads((CAN / "dermatomes.json").read_text())
    N = {n["id"]: n for n in per["nerves"]}
    R = {r["id"]: np.asarray(r["position_m"], float) for r in per["relays"]}

    def geo(eid):
        g = json.loads(gzip.decompress((IHM / E[eid]["reference_geometry"]["path"]).read_bytes()))
        return np.asarray(g["positions"], float).reshape(-1, 3)

    def bp(name):
        return next(e for e in anat["entities"]
                    if e["name"].lower() == name and e["id"].startswith("body-bp3d"))

    def ycent(name):
        return bp(name)["centroid_m"][1]

    # ---- 1. declared frames, then the frame tested on anatomy of known size
    print("== 1. frames ==")
    for label, d in (("peripheral.json", per), ("dermatomes.json", der), ("anatomy.json", anat)):
        fr = d["frame"]
        print(f"  {label:16s} declares {fr['id']} ({fr['units']}), source scale "
              f"{fr['source_transform']['scale']} (BodyParts3D mm -> m)")
        rec["frame"][label] = fr["id"]
    sk = geo("body-bp3d-FJ2810")
    height = float(sk[:, 1].max() - sk[:, 1].min())
    print(f"  skin FJ2810 height {1000 * height:.0f} mm  (a +-1 box would read ~1990)")
    rec["frame"]["skin_height_m"] = height
    for nm in ("left femur", "left tibia", "left humerus", "left radius"):
        v = geo(bp(nm)["id"]); c = v - v.mean(0)
        t = c @ np.linalg.svd(c, full_matrices=False)[2][0]
        rec["frame"][nm] = float(t.max() - t.min())
        print(f"  {nm:13s} {1000 * (t.max() - t.min()):4.0f} mm")
    za_eps = sorted({n.get("endpoint_id") for n in per["nerves"]
                     if str(n.get("endpoint_id", "")).startswith("body-za-")})
    print(f"  {len(za_eps)} route endpoints are Z-Anatomy entities; their geometry frame: "
          f"{sorted({E[e]['reference_geometry']['frame'] for e in za_eps})} (registered by "
          f"landmark affine+TPS, held-out RMS "
          f"{1000 * anat['registrations']['z_anatomy']['held_out_rms_m']:.1f} mm)")
    dump()

    # ---- Z-Anatomy centrelines into metres, exactly as the site exporter does
    ZS = X.za_to_metres()
    dtf = json.loads((X.ATLAS / "manifest_fragment.json").read_text())["models"][0]["display_transform"]
    Rz, sz, tz = np.asarray(dtf["rotation"]), float(dtf["scale"]), np.asarray(dtf["translation"])
    to_m = lambda q: (sz * (np.asarray(q, float) @ Rz.T) + tz) * ZS          # noqa: E731
    print(f"  net blender-world -> metres scale {sz * ZS:.4f} (canonical landmark affine diag "
          f"{[round(anat['registrations']['z_anatomy']['affine_4x3'][i][i], 3) for i in range(3)]})")

    def arclen(p):
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def longest(nm):
        c = next(c for c in ex["curves"] if c["name"] == nm)
        cs = [X._resample(ch, 0.005) for ch in X._chains(to_m(c["v"]), c["e"])]
        m = max([c for c in cs if c is not None], key=arclen)
        return m if m[0, 1] >= m[-1, 1] else m[::-1]              # top first

    s = open(X.OUT).read()
    site = json.loads(s[s.index("{"): s.rindex("}") + 1])["nerves"]
    P = np.asarray(site["p"]).reshape(-1, 3)
    owner = np.zeros(len(P), int)
    for i, (o, n, *_) in enumerate(site["chains"]):
        owner[o:o + n] = i
    paths = {p["name"]: np.asarray(p["idx"]) for p in site["paths"]}
    c0 = site["chains"][0]
    assert site["names"][0] == "Spinal cord"
    cord = P[c0[0]: c0[0] + c0[1]]

    def path_parts(nm):
        idx = paths[nm]; pts = P[idx]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        kc = int(np.argmax([site["names"][owner[i]] == "Spinal cord" for i in idx]))
        return pts, seg, kc

    # ---- 2. instrument: IHM's own method must reproduce IHM's own numbers
    L1 = ycent("first lumbar vertebra")
    med = E["body-bp3d-FJ1769"]["centroid_m"]
    level = {"cranial": med[1], "solitary": med[1], "cervical": ycent("fifth cervical vertebra"),
             "thoracic": ycent("sixth thoracic vertebra"), "lumbar": ycent("eleventh thoracic vertebra"),
             "sacral": L1}

    def on_cord(rid, pos):
        side = "left" if "-left-" in rid else "right"
        lvl = rid.split(f"-{side}-")[1]
        if lvl not in level:
            return pos
        if lvl in ("cranial", "solitary"):
            return np.array([pos[0], med[1], med[2]])
        c = cord[int(np.argmin(np.abs(cord[:, 1] - level[lvl])))].copy()
        c[0] = pos[0]
        return c

    newR = {rid: on_cord(rid, p) for rid, p in R.items()}

    def lengths(relays):
        per_nerve: dict = {}
        for b in per["muscle_bindings"]:
            pos = np.asarray(E[b["canonical_entity_id"]]["centroid_m"])
            per_nerve.setdefault(b["nerve_id"], []).append(
                (b["path_length_m"], max(.03, 1.15 * float(np.linalg.norm(pos - relays[b["relay_id"]])))))
        for p in per["receptor_patches"]:
            per_nerve.setdefault(p["nerve_id"], []).append(
                (p["path_length_m"], max(.03, 1.15 * float(np.linalg.norm(np.asarray(p["position_m"]) - relays[p["relay_id"]])))))
        out = {}
        for n in per["nerves"]:
            if n["id"] in per_nerve:
                c = sorted(per_nerve[n["id"]], key=lambda t: t[0])      # IHM's own ordering
                out[n["id"]] = c[len(c) // 2][1]
            else:
                pts = np.asarray(n["points_m"]); r = relays[n["relay_id"]]
                sgn = 1.0 if n["side"] == "left" else -1.0
                q = [pts[0], r] if n.get("route_kind") == "special_sense" else \
                    [pts[0], np.array([sgn * .04, r[1], r[2]]), r]
                out[n["id"]] = float(sum(np.linalg.norm(a - b) for a, b in zip(q, q[1:])))
        return out

    base = lengths(R)
    worst = max(abs(base[k] - N[k]["path_length_m"]) for k in base)
    ok = worst < 1e-9
    print(f"\n== 2. instrument: IHM's method reproduces {sum(abs(base[k] - N[k]['path_length_m']) < 1e-9 for k in base)}"
          f"/{len(base)} declared nerve lengths (worst {worst:.1e} m) ==")
    rec["instrument"] = {"reproduced": ok, "n": len(base), "worst_abs_m": worst}
    dump()
    if not ok:
        print("  !! the re-implementation does not reproduce IHM; nothing below is trustworthy",
              file=sys.stderr)
        return 2

    # ---- 3a. endpoint vs the Z-Anatomy nerve that serves it
    print("\n== 3a. route endpoint vs Z-Anatomy centreline (frame error at the foot ~ 0.12 m) ==")
    for ihm, sp in (("tibial", "Tibial nerve"), ("deep_fibular", "Deep fibular nerve"),
                    ("superficial_fibular", "Superficial fibular nerve"), ("femoral", "Femoral nerve"),
                    ("median", "Median nerve"), ("radial", "Radial nerve")):
        n = N[f"peripheral-nerve-left-{ihm}"]
        pos = np.asarray(E[n["endpoint_id"]]["centroid_m"])
        pts, seg, kc = path_parts(f"{sp}.l")
        k = int(np.argmin(np.linalg.norm(pts - pos, axis=1)))
        row = dict(route=ihm, endpoint=E[n["endpoint_id"]]["name"],
                   gap_m=float(np.linalg.norm(pts[k] - pos)), ihm_m=n["path_length_m"],
                   z_endpoint_to_root_m=float(seg[kc] - seg[k]), ihm_on_cord_m=lengths(newR)[n["id"]])
        row["ratio_ihm_over_z"] = row["ihm_m"] / row["z_endpoint_to_root_m"]
        rec["landmarks"].append(row)
        print(f"  {ihm:20s} gap {1000 * row['gap_m']:4.0f} mm   IHM {1000 * row['ihm_m']:4.0f}   "
              f"Z endpoint->root entry {1000 * row['z_endpoint_to_root_m']:4.0f}   "
              f"IHM/Z {row['ratio_ihm_over_z']:.3f}")
    dump()

    # ---- 3b. authored coordinates against what they name
    print("\n== 3b. authored y vs the structure it names (frame error: ratio ~0.865 on every row) ==")
    for lab, ya, yt in (("solitary relay | medulla", R["peripheral-relay-left-solitary"][1], med[1]),
                        ("cranial relay | medulla", R["peripheral-relay-left-cranial"][1], med[1]),
                        ("cervical relay | C5 vertebra", R["peripheral-relay-left-cervical"][1], level["cervical"]),
                        ("thoracic relay | T6 vertebra", R["peripheral-relay-left-thoracic"][1], level["thoracic"]),
                        ("lumbar relay | T11 vertebra (L cord)", R["peripheral-relay-left-lumbar"][1], level["lumbar"]),
                        ("sacral relay | L1 vertebra (conus)", R["peripheral-relay-left-sacral"][1], L1),
                        ("vagus endpoint | stomach centroid", N["peripheral-nerve-left-vagus"]["points_m"][0][1],
                         E["body-bp3d-FJ2564"]["centroid_m"][1]),
                        ("phrenic endpoint | diaphragm centroid", N["peripheral-nerve-left-phrenic"]["points_m"][0][1],
                         E["body-bp3d-FJ3131"]["centroid_m"][1]),
                        ("lateral knee endpoint | patella", N["peripheral-nerve-left-common_fibular"]["points_m"][0][1],
                         ycent("left patella"))):
        rec["authored"].append(dict(what=lab, authored_y=float(ya), anatomy_y=float(yt)))
        print(f"  {lab:40s} authored {ya:+.3f}  anatomy {yt:+.3f}  off {1000 * (ya - yt):+5.0f} mm  "
              f"ratio {yt / ya:5.2f}")
    dump()

    # ---- 4. known answers, same scope
    print("\n== 4. known answers ==")
    pts, seg, kc = path_parts("Tibial nerve.l")
    kL1 = kc + int(np.argmin(np.abs(pts[kc:, 1] - L1)))
    mp = longest("Medial plantar nerve.l")
    z_leg = float(seg[kL1] + arclen(mp))
    md = longest("Median nerve.l")
    ms = np.r_[0, np.cumsum(np.linalg.norm(np.diff(md, axis=0), axis=1))]
    wrist = geo(bp("left radius")["id"])[:, 1].min() + 0.01
    z_median = float(ms[int(np.argmin(np.abs(md[:, 1] - wrist)))])
    vg = longest("Vagus nerve (X).r")
    vs = np.r_[0, np.cumsum(np.linalg.norm(np.diff(vg, axis=0), axis=1))]
    stom = geo("body-bp3d-FJ2564")
    near = np.array([np.linalg.norm(stom - q, axis=1).min() for q in vg])
    k_st = int(np.argmax(near < 0.01))
    newL = lengths(newR)
    L = lambda k: N[f"peripheral-nerve-left-{k}"]["path_length_m"]                 # noqa: E731
    Ln = lambda k: newL[f"peripheral-nerve-left-{k}"]                             # noqa: E731
    table = [
        ("sciatic-tibial, conus -> sole", "~1.0 m or more ('can exceed one metre')",
         f"{1000 * z_leg:.0f}", f"medial_plantar {1000 * L('medial_plantar'):.0f}",
         f"{1000 * Ln('medial_plantar'):.0f}"),
        ("median, axilla -> wrist", "no single published figure found; skeleton humerus+radius ~ 0.54 m",
         f"{1000 * z_median:.0f}", f"median {1000 * L('median'):.0f} (pronator quadratus -> relay)",
         f"{1000 * Ln('median'):.0f}"),
        ("vagus, brainstem -> stomach", "no total length found in open literature",
         f"{1000 * vs[k_st]:.0f} (first stomach contact) - {1000 * vs[-1]:.0f} (lowest)",
         f"vagus {1000 * L('vagus'):.0f}", f"{1000 * Ln('vagus'):.0f}"),
        ("phrenic, root -> diaphragm", "right 246 +-17 / left 306 +-18 mm; right 300 / left 330 mm",
         "not in the atlas", f"phrenic {1000 * L('phrenic'):.0f}", f"{1000 * Ln('phrenic'):.0f}"),
    ]
    for row in table:
        rec["known_answers"].append(dict(zip(("route", "literature", "z_anatomy_mm", "ihm_mm", "ihm_relays_on_cord_mm"), row)))
        print("  " + " | ".join(row))
    for k in sorted(newL, key=lambda k: newL[k] / N[k]["path_length_m"]):
        if N[k]["side"] == "left":
            rec["relays_on_cord"].append(dict(route=k[21:], ihm_m=N[k]["path_length_m"], on_cord_m=newL[k]))
    dump()
    ratios = np.array([r["on_cord_m"] / r["ihm_m"] for r in rec["relays_on_cord"]])
    print(f"\n  relays moved onto their cord segment: routes change x{ratios.min():.2f} .. x{ratios.max():.2f} "
          f"(median x{np.median(ratios):.2f}); full list in {OUT.relative_to(HERE.parents[1])}")
    return 0


if __name__ == "__main__":
    if _IN_BLENDER:
        _extract(sys.argv[sys.argv.index("--") + 1])
    else:
        sys.exit(main())
