"""run INSIDE blender: dump Z-Anatomy's spinal nerves and spinal dura, for the site's body figure.

    blender -b ~/Documents/IHM-1/data/raw/anatomy/extended/extracted/Z-Anatomy/Startup.blend \
        --python scripts/blender_zanatomy_nerves.py -- .cache/zanatomy_nerves.json

scripts/export_site_body.py calls this for you when its cache is missing.

why this exists at all.  IHM-1's display export of the Z-Anatomy atlas (data/derived/anatomy/
extended) EXCLUDES the whole "7: Nervous system & Sense organs" collection, with the reason
"sense-organ/nervous collection excluded: third-party license scope".  the licence file names
exactly which components carry that scope: "Anatomy of the Inner Ear" (University of Dundee,
CC-BY-NC-SA 4.0) and "Kidney" (Lissie Cowley, CC-BY-NC 4.0).  the spinal nerves and the spinal
dura are Z-Anatomy/BodyParts3D, CC-BY-SA like the rest of the atlas the site already ships.
so this takes the peripheral nervous system and the dura, and takes NOTHING filed under an ear
or a cranial-nerve collection -- which keeps it out of the NC component by construction rather
than by trusting a name.

the nerves in the source are CURVE objects: a centreline with a 0.5 mm bevel.  the bevel is
switched off before evaluation so what comes out is the centreline itself, resampled at the
curve's own resolution, in blender world coordinates.  no geometry is invented here.
"""
import json
import sys

import bpy


def all_collections(obj):
    names = set()

    def up(c):
        names.add(c.name)
        for p in bpy.data.collections:
            if c.name in p.children:
                up(p)

    for c in obj.users_collection:
        up(c)
    return names


def excluded(cols):
    for c in cols:
        words = c.lower().replace("(", " ").replace(")", " ").split()
        if "ear" in words or "inner ear" in c.lower() or "cranial nerves" in c.lower():
            return True
    return False


def main(out_path):
    out = {"source": bpy.data.filepath, "frame": "z-anatomy-blender-world",
           "nerves": [], "dura": None, "check": {}}
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.data.objects:
        cols = all_collections(o)
        # a bone the exporter ALSO ships from IHM-1's derived geometry: the known answer that
        # the blender->display transform is checked against before any nerve is trusted.
        if o.type == "MESH" and o.name in ("Femur.l", "Humerus.r", "Sacrum"):
            mw = o.matrix_world
            vs = [mw @ v.co for v in o.data.vertices]
            out["check"][o.name] = [[min(v[i] for v in vs) for i in range(3)],
                                    [max(v[i] for v in vs) for i in range(3)]]
        if o.type == "MESH" and o.name == "Spinal dura":
            mw = o.matrix_world
            out["dura"] = [list(mw @ v.co) for v in o.data.vertices]
        if o.type != "CURVE":
            continue
        if not ("Peripheral nervous system" in cols or o.name == "Cauda equina"):
            continue
        if excluded(cols):
            continue
        cu = o.data
        keep = (cu.bevel_depth, cu.extrude, cu.bevel_object)
        cu.bevel_depth, cu.extrude, cu.bevel_object = 0.0, 0.0, None
        dg.update()
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        mw = o.matrix_world
        verts = [list(mw @ v.co) for v in me.vertices]
        edges = [list(e.vertices) for e in me.edges]
        ev.to_mesh_clear()
        cu.bevel_depth, cu.extrude, cu.bevel_object = keep
        out["nerves"].append({"name": o.name, "bevel": keep[0], "collections": sorted(cols),
                              "v": verts, "e": edges})
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"blender_zanatomy_nerves: {len(out['nerves'])} nerve objects, "
          f"dura {'present' if out['dura'] else 'MISSING'}, check {sorted(out['check'])}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1])
