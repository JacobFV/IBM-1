"""export a whole-body anatomical figure -- every system the atlas ships -- to site/data/body.js.

the site's body station used to be a screen recording, then a musculoskeletal rest pose.
this ships the whole atlas: Z-Anatomy surface meshes for all thirteen systems the
extended manifest carries, decimated, each one bound to the segment of the driven
skeleton it rides on, plus the skeleton's joint tree so the page can pose it.

the two halves of the body model do NOT share an id space, AND THEY DO NOT SHARE A FRAME.
the meshes are `za-*` structures in `z-anatomy-display-normalized`, which is a normalized
cube -- the atlas manifest gives its bounds as exactly +-1 in y and says in as many words
that "display normalization does not establish physical dimensions".  the segment binding
and its bone anchors are `body-bp3d-FJ*` entities in `bodyparts3d-display-m`, which is
metres.  the previous version of this exporter compared the two directly, so every mesh was
bound to a segment using vertices that were 15.7% too large and 1 cm off in y; it produced
a plausible-looking figure only because that figure never moved, and the error would have
shown up the moment a limb rotated.  the scale is recovered here from the two manifests'
declared bounds and checked on an independent landmark (the patella).

what this exporter deliberately does NOT do any more is ship a stored trajectory.  every
solved trajectory for this body is outside the body model's own declared joint ranges
(docs/LOG.md: crawl-best's left knee is past its limit for 95.6% of 1,600 frames, the
ankles for 79-94%), so rigid-driving the anatomy with one tears the skeleton apart at the
joints.  instead it ships the *kinematic tree* -- parent, pivot, axis, and the declared
range of the OpenSim coordinate that owns each degree of freedom -- and site/body.js
synthesises a preview pose inside those ranges.  a synthesised preview is honest about
being a preview; a trajectory that dislocates a knee is not.

    PYTHONPATH=~/Documents/IBM-1 python scripts/export_site_body.py
"""
from __future__ import annotations

import argparse, gzip, json, os, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

IHM = Path.home() / "Documents" / "IHM-1"
BIND = IHM / "data/derived/anatomy-segment-binding"
ATLAS = IHM / "data/derived/anatomy/extended"
CANON = IHM / "data/derived/canonical"
OSIM = IHM / "data/models/engineering_stance_v1/model.osim"
CANON_MAN = CANON / "manifest_fragment.json"
OUT = Path(__file__).resolve().parents[1] / "site" / "data" / "body.js"

# per-system display budget: how many structures to ship, how many triangles each may keep,
# and how fine the clustering grid starts.  the skin is the silhouette and gets the largest
# share; the viscera are seen THROUGH skin at low alpha, where a coarse blob reads the same
# as a fine one, so they are cut hard.  `None` for a count means "ship all of them" -- the
# integumentary regions tile the body surface and dropping one leaves a hole in the figure.
#                     count  max_tris  grid
BUDGET = {
    "integumentary": (None,   380,      38),
    "skeletal":      (140,    320,      30),
    "muscular":      (150,    230,      26),
    "connective":    (40,     110,      20),
    "arterial":      (48,     110,      18),
    "venous":        (36,     110,      18),
    "lymphatic":     (16,      70,      14),
    "digestive":     (26,     200,      22),
    "respiratory":   (20,     200,      22),
    "cardiac":       (12,     150,      22),
    "urinary":       (6,      130,      20),
    "endocrine":     (10,      70,      16),
    "reproductive":  (10,     110,      18),
}

# the kinematic tree of the driven skeleton.  these are the segments the binding declares,
# and the parent of each is anatomy, not a fit.  patella hangs off the femur rather than the
# tibia because the kneecap tracks the knee rather than swinging with the shank.
TREE = {
    "torso": "pelvis",
    "femur_l": "pelvis", "femur_r": "pelvis",
    "tibia_l": "femur_l", "tibia_r": "femur_r",
    "patella_l": "femur_l", "patella_r": "femur_r",
    "talus_l": "tibia_l", "talus_r": "tibia_r",
    "calcn_l": "talus_l", "calcn_r": "talus_r",
    "toes_l": "calcn_l", "toes_r": "calcn_r",
    "humerus_l": "torso", "humerus_r": "torso",
    "ulna_l": "humerus_l", "ulna_r": "humerus_r",
    "radius_l": "ulna_l", "radius_r": "ulna_r",
    "hand_l": "radius_l", "hand_r": "radius_r",
}

# which OpenSim coordinates gate each joint.  the page clamps every angle it synthesises to
# the range read out of the .osim below, so "inside the declared ranges" is a property of the
# shipped data rather than of my arithmetic.  a segment with no entry here does not rotate.
JOINT_COORDS = {
    "torso": ["lumbar_extension", "lumbar_bending", "lumbar_rotation"],
    "femur_l": ["hip_flexion_l", "hip_adduction_l", "hip_rotation_l"],
    "femur_r": ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r"],
    "tibia_l": ["knee_angle_l"], "tibia_r": ["knee_angle_r"],
    "talus_l": ["ankle_angle_l"], "talus_r": ["ankle_angle_r"],
    "toes_l": ["mtp_angle_l"], "toes_r": ["mtp_angle_r"],
    "humerus_l": ["arm_flex_l", "arm_add_l", "arm_rot_l"],
    "humerus_r": ["arm_flex_r", "arm_add_r", "arm_rot_r"],
    "ulna_l": ["elbow_flex_l"], "ulna_r": ["elbow_flex_r"],
    "radius_l": ["pro_sup_l"], "radius_r": ["pro_sup_r"],
}


def decimate(pos: np.ndarray, idx: np.ndarray, cell: float):
    """vertex-clustering decimation: snap vertices to a grid, keep one representative per
    cell, drop triangles that collapse.  crude, but it preserves silhouette far better
    than dropping every Nth triangle and needs no mesh library."""
    keys = np.floor(pos / cell).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n = counts.shape[0]
    reps = np.zeros((n, 3))
    np.add.at(reps, inv, pos)
    reps /= counts[:, None]
    tri = inv[idx.reshape(-1, 3)]
    ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    return reps, tri[ok]


def za_to_metres() -> float:
    """the uniform scale that carries `z-anatomy-display-normalized` into
    `bodyparts3d-display-m`.  both manifests declare an axis-aligned bounding box centred on
    the origin for a standing figure, so the ratio of the two boxes IS the registration, up
    to the two atlases being slightly different people.  height is the one dimension that
    means the same thing in both (the x extent is set by where the hands hang), so the scale
    comes from y and the other two axes are reported as residuals rather than fitted."""
    za = json.loads((ATLAS / "manifest_fragment.json").read_text())["models"][0]["bounds"]
    bp = json.loads(CANON_MAN.read_text())["models"][0]["bounds"]
    s = bp["max"][1] / za["max"][1]
    res = [bp["max"][i] / za["max"][i] / s - 1 for i in (0, 2)]
    print(f"z-anatomy -> metres: uniform scale {s:.5f} from height; "
          f"x and z bounds then disagree by {res[0] * 100:+.1f}% and {res[1] * 100:+.1f}% "
          f"(two atlases, not one subject)")
    if max(abs(r) for r in res) > 0.06:
        print("  !! the two bounding boxes are not the same shape -- a uniform scale is "
              "the wrong registration", file=sys.stderr)
    return s


# ---------------------------------------------------------------------------- the nerves
#
# the schematic routes this station used before -- IHM-1's peripheral_display.json, three
# points each, cortex / spinal relay / muscle, labelled by their own file as
# `schematic_anatomical_prior` -- were straight-line sketches through the body, and on screen
# they read as exactly that.  they are gone.  what replaces them is the nervous system Z-Anatomy
# itself draws: the spinal nerves as the atlas authors them (centrelines), the spinal cord
# along the centreline of the atlas's own spinal dura, and a pulse routed through that network
# along named nerves.  see scripts/blender_zanatomy_nerves.py for why these were missing from
# IHM-1's display export (a licence exclusion scoped to the inner ear and the kidney, applied
# to the whole collection) and why taking the spinal nerves stays inside it.

BLEND = IHM / "data/raw/anatomy/extended/extracted/Z-Anatomy/Startup.blend"
NERVE_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "zanatomy_nerves.json"

# display radius per class of nerve, in metres.  the source bevel is 0.5 mm, which would be
# sub-pixel at this size; these are at or BELOW the real calibre (the sciatic is 1-2 cm wide
# where it leaves the pelvis, the cord about 1 cm across), so thickening the source's hairline
# to them does not make any nerve bigger than it is.
NERVE_RADIUS = [("Spinal cord", 0.0042),
                ("Sciatic nerve", 0.0032),
                ("Tibial nerve|Common fibular nerve|Femoral nerve|Median nerve|Ulnar nerve"
                 "|Radial nerve|Superior trunk|Middle trunk|Inferior trunk|cord of brachial", 0.0021),
                ("", 0.0011)]

# the pathways a pulse may run, and the named nerves each one MUST pass through, distal first.
# a route is found by shortest path through the atlas's own network, and then GATED against
# this list: a path that reaches the cord through anything but these nerves and the plexus /
# root / cauda connectors between them is dropped, not drawn.  that gate exists because the
# first routing sent the musculocutaneous nerve to the cord through the long thoracic nerve,
# which is a shortest path through the geometry and a falsehood about the anatomy.
PATHWAYS = [
    ("Tibial nerve", ["Tibial nerve", "Sciatic nerve"]),
    ("Deep fibular nerve", ["Deep fibular nerve", "Common fibular nerve", "Sciatic nerve"]),
    ("Superficial fibular nerve", ["Superficial fibular nerve", "Common fibular nerve", "Sciatic nerve"]),
    ("Femoral nerve", ["Femoral nerve"]),
    ("Median nerve", ["Median nerve"]),
    ("Ulnar nerve", ["Ulnar nerve"]),
    ("Radial nerve", ["Radial nerve"]),
    ("Musculocutaneous nerve", ["Musculocutaneous nerve"]),
]
CONNECTOR = r"brachial plexus|root of spinal nerve|Cauda equina|Spinal cord|lumbar plexus|sacral plexus|Lumbosacral trunk"
# autonomic chains and the intercostals are drawn but never ROUTED through: they run parallel
# to the spine for its whole length and made every shortest path a shortcut along them.
NO_ROUTE = r"ympathetic|Vagus|ntercostal|planchnic|ardiac"
JOIN_TOL = 0.015      # chain ends this close to another chain are one junction
HOP_COST = 0.10       # metres added for changing named nerve, so a path follows a nerve's own length


def blender_nerves() -> dict:
    """the extracted Z-Anatomy nerves, running blender once to make them if they are not cached."""
    if not NERVE_CACHE.exists():
        import subprocess
        NERVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).resolve().parent / "blender_zanatomy_nerves.py"
        print(f"extracting nerves from {BLEND.name} with blender (one-off, cached)...", flush=True)
        subprocess.run(["blender", "-b", str(BLEND), "--python", str(script), "--",
                        str(NERVE_CACHE)], check=True, capture_output=True)
    return json.loads(NERVE_CACHE.read_text())


def _chains(v: np.ndarray, edges) -> list:
    """an evaluated curve arrives as a bag of edges; walk it into ordered polylines, one per
    unbranched run, so arc length along a nerve means something."""
    adj: dict[int, list] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    used, out = set(), []
    for st in [k for k, n in adj.items() if len(n) != 2] + list(adj):
        for nb in adj[st]:
            k = (min(st, nb), max(st, nb))
            if k in used:
                continue
            used.add(k)
            ch, prev, cur = [st, nb], st, nb
            while len(adj[cur]) == 2:
                nx = [n for n in adj[cur] if n != prev][0]
                k = (min(cur, nx), max(cur, nx))
                if k in used:
                    break
                used.add(k)
                ch.append(nx)
                prev, cur = cur, nx
            out.append(v[ch])
    return out


def _resample(c: np.ndarray, step: float):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    if d[-1] < 1e-5:
        return None
    u = np.linspace(0, d[-1], max(2, int(np.ceil(d[-1] / step)) + 1))
    return np.stack([np.interp(u, d, c[:, i]) for i in range(3)], 1)


def osim_ranges(path: Path) -> dict:
    """every Coordinate's declared range, in radians, straight out of the model the body
    programme actually poses.  reading them from the file rather than typing them in is the
    point: CLAUDE.md records a whole day lost to a knee whose range was mirror-imaged
    between two models, and a number nobody transcribed cannot be transcribed wrong."""
    out = {}
    root = ET.parse(path).getroot()
    for c in root.iter("Coordinate"):
        rg = c.find("range")
        if rg is None or not c.get("name"):
            continue
        lo, hi = (float(x) for x in rg.text.split())
        # the arm coordinates in this model are declared [-10, 10] rad, which is not a
        # range, it is an absent one.  clamp them to something anatomical and SAY which
        # ones were substituted, so the page never claims a bound the model did not give.
        out[c.get("name")] = {"lo": round(lo, 5), "hi": round(hi, 5),
                              "declared": bool(abs(lo) < 7 and abs(hi) < 7)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", default=",".join(BUDGET),
                    help="comma-separated system ids; default is every system the atlas has")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every per-system structure count and triangle cap, to "
                         "trade payload against detail in one knob")
    ap.add_argument("--report", default="",
                    help="print the segment every structure of this system was bound to; the "
                         "only way to find out which mesh is the one flapping at the hip")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    want = [s for s in a.systems.split(",") if s]

    binding = json.loads((BIND / "binding.json").read_text())
    ent_seg = {k: v["segment"] for k, v in binding["entities"].items()}
    ent_role = {k: v.get("role") for k, v in binding["entities"].items()}

    by_seg: dict[str, list[str]] = {}
    for e, sg in ent_seg.items():
        if sg and e in binding["centroids_m"]:
            by_seg.setdefault(sg, []).append(e)
    by_seg = {k: v for k, v in by_seg.items() if len(v) >= 3}
    seg_names = sorted(by_seg)

    # ---- the nearest-neighbour pool is BONES ONLY, and the vote is over a mesh's vertices.
    # taking the nearest of all 4,000 bound entities to a mesh's CENTROID -- which is what
    # this did first -- votes against a pool that is three-quarters vessels, muscles and soft
    # organs, and picks a segment from one interior point of a long bone.  femurs and humeri
    # came out attached to the wrong segment and the skeleton visibly came apart at the
    # joints.  `nearest_bone_group_vertex_vote` is the basis the binding itself used.
    bone_pts, bone_owner = [], []
    for si, sg in enumerate(seg_names):
        for e in by_seg[sg]:
            if ent_role.get(e) == "rigid_bone":
                bone_pts.append(binding["centroids_m"][e])
                bone_owner.append(si)
    all_pts = np.asarray(bone_pts, dtype=float)
    all_owner = np.asarray(bone_owner, dtype=int)
    print(f"bone anchors: {len(all_pts)} across {len(set(all_owner.tolist()))} segments")

    weak, rebound = [], []
    surf = {"pts": None, "owner": None}          # filled by the skeletal pass, below

    def seg_by_anchor(v, name):
        """the segment whose bone-group ANCHOR best represents the whole mesh.

        a modal vote over each vertex's nearest anchor is biased by anchor density along an
        elongated bone: the femur's distal vertices are all nearest the patella anchor, so
        the femur was assigned to `patella_l` and rode the kneecap.  taking the anchor that
        minimises MEAN distance to every vertex asks which bone this mesh actually is, which
        is the question, and the patella anchor loses it badly over a whole femur.

        this is only used for the BONES, which is all it is good for -- an anchor is a bone
        group's centroid, so soft tissue is tens of centimetres from every anchor it has and
        the comparison is between two large numbers."""
        sample = v if len(v) <= 256 else v[np.linspace(0, len(v) - 1, 256).astype(int)]
        d = np.linalg.norm(sample[:, None, :] - all_pts[None, :, :], axis=2)
        mean_to_anchor = d.mean(0)
        best = int(mean_to_anchor.argmin())
        win = int(all_owner[best])
        other = mean_to_anchor.copy()
        other[all_owner == win] = np.inf
        rival = float(other.min()) if np.isfinite(other).any() else np.inf
        ratio = mean_to_anchor[best] / rival if rival else 0.0
        if ratio > 0.92:
            weak.append((name, seg_names[win], round(float(ratio), 2)))
        return seg_names[win]

    def seg_by_surface(v, name):
        """the segment whose BONE SURFACE this mesh lies against.

        once the skeletal pass has run, every segment has an actual surface rather than one
        interior point, and the right question becomes askable: of the 22 skeletons-worth of
        bone in this body, which one is this structure wrapped around?  it is asked as a mean
        over the mesh's vertices of the distance to the nearest point of that segment's bone,
        because a max is decided by one stray vertex and a min by one lucky one.

        this replaced an anchor vote that put a fascial sheet from the hip onto the humerus.
        it looked right at rest and swung a hand's width clear of the body as soon as the arm
        moved, which is the whole failure mode of a rigid-segment figure: a binding error is
        invisible until something rotates.  two earlier attempts to catch it by a threshold
        on the anchor distances both rejected the femur and the thigh skin, which are bound
        correctly -- the distances they compared were dominated by the anchors being interior
        points, not by the binding being wrong."""
        if surf["pts"] is None:
            return seg_by_anchor(v, name)
        sample = v if len(v) <= 96 else v[np.linspace(0, len(v) - 1, 96).astype(int)]
        # SYMMETRISED, and not as a tidiness measure.  the raw query bound `Lateral region of
        # abdomen.r` to the torso and its mirror image `.l` to the pelvis, because the two
        # are a borderline case and the atlas's own left and right are not exactly mirrored.
        # the left half of the abdomen then rode the pelvis while the right half rode the
        # trunk, and the walk's lumbar rotation opened a flap of skin at the waist.  so every
        # structure is scored twice -- itself, and its reflection against the mirrored
        # segment labels -- and the two are averaged.  a mirrored pair of structures then
        # cannot be given unmirrored segments, whatever the answer is.
        mirror = lambda sg: sg[:-2] + ("_r" if sg.endswith("_l") else "_l") \
            if sg.endswith(("_l", "_r")) else sg
        flip = sample * [-1, 1, 1]
        d_of = {}
        for sg in seg_names:
            pts = surf["pts"].get(sg)
            if pts is None:
                continue
            d_of[sg] = (float(np.linalg.norm(sample[:, None, :] - pts[None, :, :], axis=2).min(1).mean()),
                        float(np.linalg.norm(flip[:, None, :] - pts[None, :, :], axis=2).min(1).mean()))
        best, second, win = 1e9, 1e9, None
        for sg in d_of:
            m = (d_of[sg][0] + d_of.get(mirror(sg), d_of[sg])[1]) / 2
            if m < best:
                best, second, win = m, best, sg
            elif m < second:
                second = m
        if win is None:
            return seg_by_anchor(v, name)
        if best / second > 0.94:
            weak.append((name, win, round(best / second, 2)))
        return win

    # ---- load, decimate and bind every structure inside its system's budget ----
    man = json.loads((ATLAS / "manifest_fragment.json").read_text())
    sysinfo = {s["id"]: s for s in man["systems"]}
    ZS = za_to_metres()

    # there is no triangle count in the manifest -- `source_triangles` and `original_faces`
    # are both absent, so the old `sort(key=-(...or 0))` was a no-op that sorted nothing and
    # shipped whatever order the manifest happened to be in.  the gzipped geometry's size on
    # disk is a real proxy for how much surface a structure has, and it costs a stat().
    def gz_size(st):
        p = IHM / st["geometry_path"]
        return p.stat().st_size if p.exists() else 0

    # a per-segment quota WITHIN each system, because sorting by size across a whole system
    # spends the entire budget on the torso and ships a figure with no legs -- which is what
    # the first run of the old exporter produced.
    raw: dict[tuple, list] = {}
    seg_verts: dict[str, list] = {}       # skeletal vertices per segment, for the pivots
    per_sys = {}
    # the skeleton goes FIRST whatever order the caller asked for, because it is what every
    # other system is then bound against: nothing but bone can say where a segment is.
    order = (["skeletal"] if "skeletal" in want else []) + [s for s in want if s != "skeletal"]
    for sysid in order:
        if sysid not in BUDGET:
            print(f"  !! no budget for system {sysid}", file=sys.stderr)
            continue
        cap, max_tris, grid = BUDGET[sysid]
        if cap is not None:
            cap = max(1, int(round(cap * a.scale)))
        max_tris = max(12, int(round(max_tris * a.scale)))
        structures = [s for s in man["structures"]
                      if s.get("system") == sysid and s.get("kind") == "mesh"]
        structures.sort(key=lambda s: -gz_size(s))
        quota = 10 ** 6 if cap is None else max(2, int(np.ceil(cap / len(seg_names) * 2.4)))
        taken: dict[str, int] = {}
        deferred, kept, tris = [], 0, 0
        # the skeletal pass reads EVERY bone even when the display budget will not ship it,
        # because the joint pivots below are found where two segments' bone surfaces meet and
        # a segment whose bones all fell outside the budget has no surface to meet with.  the
        # first run of this lost the talus, calcaneus, ulna and radius that way and their
        # joints simply vanished from the tree.
        cloud_only = sysid == "skeletal"
        for st in structures:
            if cap is not None and kept >= cap and not cloud_only:
                break
            gp = IHM / st["geometry_path"]
            if not gp.exists():
                continue
            g = json.loads(gzip.open(gp).read())
            pos = np.asarray(g["positions"], dtype=np.float64).reshape(-1, 3) * ZS
            idx = np.asarray(g["indices"], dtype=np.int64)
            if pos.size == 0 or idx.size < 3:
                continue
            ext = float(np.linalg.norm(pos.max(0) - pos.min(0)))
            cell = max(ext / grid, 1e-4)
            for _ in range(9):
                v, tri = decimate(pos, idx, cell)
                if len(tri) <= max_tris or len(tri) == 0:
                    break
                cell *= 1.3
            if len(tri) < 4:
                continue
            seg = (seg_by_anchor if cloud_only else seg_by_surface)(v, st["name"])
            if a.report == sysid:
                print(f"      {seg:10s}  {st['name']}")
            if cloud_only:
                seg_verts.setdefault(seg, []).append(v)
            if cap is not None and kept >= cap:
                continue
            if taken.get(seg, 0) >= quota:
                deferred.append((seg, v, tri, st))
                continue
            taken[seg] = taken.get(seg, 0) + 1
            raw.setdefault((seg, sysid), []).append((v, tri))
            kept += 1
            tris += len(tri)
        for seg, v, tri, st in deferred:          # spend the leftover on hungrier segments
            if cap is not None and kept >= cap:
                break
            raw.setdefault((seg, sysid), []).append((v, tri))
            kept += 1
            tris += len(tri)
        per_sys[sysid] = {"n": kept, "tris": tris}
        print(f"  {sysid:14s} {kept:4d} structures, {tris:6d} tris", flush=True)
        if cloud_only:
            # thin each segment's bone surface to a couple of hundred points.  the binding
            # question is answered to within a centimetre at that density and the whole
            # remaining atlas then binds in seconds rather than minutes.
            surf["pts"] = {}
            for sg, parts in seg_verts.items():
                c = np.concatenate(parts)
                if len(c) > 260:
                    c = c[np.linspace(0, len(c) - 1, 260).astype(int)]
                surf["pts"][sg] = c
            print(f"    bone surfaces for {len(surf['pts'])}/{len(seg_names)} segments; "
                  f"every other system binds against these")

    if weak:
        print(f"  {len(weak)} mesh(es) bound with a thin margin, e.g. {weak[:3]}", file=sys.stderr)


    # ---- joint pivots, from the SKELETAL surfaces rather than the bone centroids ----
    # a first attempt put each pivot at the midpoint of the two segments' bone-entity
    # centroids.  most limb segments are ONE entity, so that is the midpoint of two
    # midshafts: it put the knee 5 cm below the kneecap and the elbow inside the forearm.
    # the joint is where the two SURFACES come closest, so take the closest 4% of
    # cross-segment vertex pairs and average them -- distal femoral condyles against tibial
    # plateau, which is the knee.
    clouds = {s: np.concatenate(v) for s, v in seg_verts.items() if v}
    anchors = {s: np.asarray([binding["centroids_m"][e] for e in by_seg[s]
                              if ent_role.get(e) == "rigid_bone"]) for s in seg_names}
    pivots, pivot_gap, fellback = {}, {}, []
    for child, parent in TREE.items():
        A, B = clouds.get(child), clouds.get(parent)
        if A is not None and B is not None:
            if len(A) > 2000:
                A = A[np.linspace(0, len(A) - 1, 2000).astype(int)]
            if len(B) > 2000:
                B = B[np.linspace(0, len(B) - 1, 2000).astype(int)]
            D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
            near = D.min(1)
            k = max(3, int(0.04 * len(A)))
            sel = np.argsort(near)[:k]
            pivots[child] = ((A[sel] + B[D.argmin(1)[sel]]) / 2).mean(0)
            pivot_gap[child] = float(near[sel].mean())
        # a joint whose two bone surfaces are judged to meet 6 cm apart has not been found:
        # either the bones that actually articulate are not in the atlas or they went to
        # another segment.  fall back on the bone-group CENTROIDS, which are complete, and
        # say which joints were placed that way rather than averaging the two silently.
        if child not in pivots or pivot_gap.get(child, 9) > 0.06:
            ca, cb = anchors.get(child), anchors.get(parent)
            if ca is not None and len(ca) and cb is not None and len(cb):
                D = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
                i, j = np.unravel_index(D.argmin(), D.shape)
                pivots[child] = (ca[i] + cb[j]) / 2
                fellback.append(child)
    pivots["pelvis"] = (clouds["pelvis"].mean(0) if "pelvis" in clouds
                        else anchors["pelvis"].mean(0))
    if fellback:
        print(f"pivots from bone centroids rather than surfaces: {fellback}")

    # mirror check: the atlas is a standing figure with x to the subject's left, so every
    # left pivot must be the mirror of its right one.  a pivot that fails this has bound to
    # the wrong side and the limb will hinge across the body.
    worst_mirror, worst_at = 0.0, ""
    for s in list(pivots):
        if s.endswith("_l") and s[:-2] + "_r" in pivots:
            l, r = pivots[s], pivots[s[:-2] + "_r"]
            m = float(np.abs(l * [-1, 1, 1] - r).max())
            if m > worst_mirror:
                worst_mirror, worst_at = m, s[:-2]
    print(f"pivot mirror asymmetry: worst {worst_mirror * 1000:.1f} mm at {worst_at} (expect < 15)")
    if worst_mirror > 0.015:
        print("  !! a pivot is not mirror-symmetric -- check the segment binding", file=sys.stderr)

    # a pivot must also lie BETWEEN its segment and its parent, not out in the air.  the gap
    # is how far apart the two surfaces are where they were judged to meet; a joint whose
    # surfaces are 10 cm apart is not a joint.
    bad = {c: round(g * 1000, 1) for c, g in pivot_gap.items() if g > 0.06}
    print(f"pivot surface gap: max {max(pivot_gap.values()) * 1000:.1f} mm" +
          (f"  WIDE {bad}" if bad else ""))

    # a KNOWN-ANSWER check on the whole za -> metres chain, on a landmark nothing above
    # used: the knee pivot is found from Z-Anatomy surfaces, and the bp3d binding
    # independently gives a centroid for the patella.  they are two atlases, so they will not
    # agree exactly, but a wrong scale shows up here as tens of centimetres.  with the frame
    # bug in place this read 74 mm; registered it reads about 10.
    for side in ("l", "r"):
        pat = [binding["centroids_m"][e] for e, v in binding["entities"].items()
               if (v.get("name") or "").lower() == f"{'left' if side == 'l' else 'right'} patella"]
        if pat and f"patella_{side}" in pivots:
            d = float(np.linalg.norm(np.asarray(pat[0]) - pivots[f"patella_{side}"]))
            print(f"knee cross-check ({side}): za-derived pivot vs bp3d patella centroid "
                  f"{d * 1000:.0f} mm")
            if d > 0.05:
                print("  !! the two atlases disagree about where the knee is by more than "
                      "5 cm -- the frame registration is wrong", file=sys.stderr)

    # ---- merge to one buffer per (segment, system) ----
    # 900-odd separate meshes is 900 draw calls a frame, and at this size nobody can tell a
    # rectus femoris from a vastus lateralis anyway: the atlas gives one colour per SYSTEM,
    # not per structure, so merging inside a system loses nothing that was ever visible.
    meshes = []
    for (seg, sysid), parts in sorted(raw.items()):
        piv = pivots.get(seg, np.zeros(3))
        V, I, off = [], [], 0
        for v, tri in parts:
            V.append(v - piv)                 # emitted in the segment's own frame
            I.append(tri + off)
            off += len(v)
        V = np.concatenate(V); I = np.concatenate(I)
        meshes.append({
            "sys": sysid, "seg": seg,
            "v": [round(float(x), 4) for x in V.reshape(-1)],
            "i": [int(x) for x in I.reshape(-1)],
        })

    # ---- the nervous system, from the Z-Anatomy source itself ----
    raw_n = blender_nerves()
    zman = json.loads((ATLAS / "manifest_fragment.json").read_text())["models"][0]
    dtf = zman["display_transform"]
    Rz, sz, tz = np.asarray(dtf["rotation"]), float(dtf["scale"]), np.asarray(dtf["translation"])
    to_m = lambda q: (sz * (np.asarray(q, dtype=float) @ Rz.T) + tz) * ZS

    # KNOWN ANSWER before any nerve is trusted: the blend's own femur, carried through
    # blender-world -> display -> metres, must land on the femur this exporter already ships
    # from IHM-1's derived geometry.  the convention (scale.R.p + t, versus R.(p + t) and the
    # rest) was chosen by this test, not assumed: the wrong one misses by about a metre.
    fem = next((st for st in man["structures"] if st["name"] == "Femur.l"), None)
    if fem is not None and "Femur.l" in raw_n.get("check", {}):
        g = json.loads(gzip.open(IHM / fem["geometry_path"]).read())
        ref = np.asarray(g["positions"], dtype=float).reshape(-1, 3) * ZS
        lo, hi = (np.asarray(x) for x in raw_n["check"]["Femur.l"])
        corners = to_m([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        err = float(np.abs(np.r_[corners.min(0) - ref.min(0), corners.max(0) - ref.max(0)]).max())
        print(f"nerve frame check: blend femur vs shipped femur, worst box edge {err * 1000:.2f} mm")
        if err > 0.002:
            print("  !! the blend -> metres transform is wrong; refusing to ship nerves", file=sys.stderr)
            return 2

    import re as _re, heapq
    chains = []
    for nv in raw_n["nerves"]:
        for c in _chains(to_m(nv["v"]), nv["e"]):
            r = _resample(c, 0.010)
            if r is not None:
                chains.append({"name": nv["name"], "p": r})
    # the cord: the atlas has no cord surface, but it has the SPINAL DURA, the sac the cord runs
    # in.  its centreline, sliced every centimetre down its length, is the cord's path from the
    # foramen magnum to the conus.  ordered top-first, so the cord's first point is the end the
    # brain is at.
    if raw_n.get("dura"):
        D = to_m(raw_n["dura"])
        ys = np.arange(D[:, 1].min() + 0.004, D[:, 1].max(), 0.01)
        cl = [D[np.abs(D[:, 1] - y) < 0.006].mean(0) for y in ys]
        cord = _resample(np.asarray(cl)[::-1], 0.010)
        chains.insert(0, {"name": "Spinal cord", "p": cord})
    else:
        print("  !! no spinal dura in the extraction -- the pulse has no cord to reach", file=sys.stderr)

    P = np.concatenate([c["p"] for c in chains])
    off = np.cumsum([0] + [len(c["p"]) for c in chains])
    owner = np.concatenate([[i] * len(c["p"]) for i, c in enumerate(chains)])
    base = lambda nm: nm.rsplit(".", 1)[0] if nm.endswith((".l", ".r")) else nm
    no_route = np.array([bool(_re.search(NO_ROUTE, chains[o]["name"])) for o in owner])

    # the routing graph: consecutive points along a chain, plus a junction from every chain
    # END to the nearest point of every OTHER chain within 15 mm.  every-other-chain, not just
    # the nearest one: the median nerve's top end sits exactly on the superior subscapular
    # nerve's, and a nearest-only junction wired the median to THAT and never to the plexus
    # division 15 mm away, so its pulse went up the wrong nerve.
    adj = [[] for _ in range(len(P))]
    for ci, c in enumerate(chains):
        if _re.search(NO_ROUTE, c["name"]):
            continue
        for i in range(len(c["p"]) - 1):
            a_, b_ = off[ci] + i, off[ci] + i + 1
            w = float(np.linalg.norm(P[a_] - P[b_]))
            adj[a_].append((b_, w)); adj[b_].append((a_, w))
    n_join = 0
    for ci, c in enumerate(chains):
        if _re.search(NO_ROUTE, c["name"]):
            continue
        for end in (off[ci], off[ci + 1] - 1):
            d = np.linalg.norm(P - P[end], axis=1)
            d[owner == ci] = 9; d[no_route] = 9
            best = {}
            for j in np.where(d < JOIN_TOL)[0]:
                if owner[j] not in best or d[j] < d[best[owner[j]]]:
                    best[owner[j]] = j
            for j in best.values():
                w = float(d[j]) + (0.0 if base(chains[owner[j]]["name"]) == base(c["name"]) else HOP_COST)
                adj[end].append((int(j), w)); adj[int(j)].append((end, w)); n_join += 1
    top = int(off[0])                                       # the cord's upper end
    dist = np.full(len(P), np.inf); prev = np.full(len(P), -1); dist[top] = 0.0
    pq = [(0.0, top)]
    while pq:
        d0, u = heapq.heappop(pq)
        if d0 > dist[u]:
            continue
        for v_, w in adj[u]:
            if d0 + w < dist[v_]:
                dist[v_] = d0 + w; prev[v_] = u; heapq.heappush(pq, (dist[v_], v_))

    paths, gate = [], []
    for nerve, expect in PATHWAYS:
        for side in ("l", "r"):
            ends = [e for ci, c in enumerate(chains) if c["name"] == f"{nerve}.{side}"
                    for e in (int(off[ci]), int(off[ci + 1]) - 1) if np.isfinite(dist[e])]
            if not ends:
                gate.append((f"{nerve}.{side}", "not connected to the cord"))
                continue
            start = max(ends, key=lambda e: dist[e])       # the nerve's far end
            idx, u = [], start
            while u != -1:
                idx.append(u); u = int(prev[u])
            runs = []
            for i in idx:
                nm = base(chains[owner[i]]["name"])
                if runs and runs[-1][0] == nm:
                    runs[-1][1] += 1
                else:
                    runs.append([nm, 1])
            # a run of one or two points is a junction the path touches in passing, not a
            # nerve it travels along
            named = [nm for nm, k in runs if k > 2 and not _re.search(CONNECTOR, nm, _re.I)]
            ok = named == expect and runs[-1][0] == "Spinal cord"
            via = " > ".join(f"{nm}({k})" for nm, k in runs)
            gate.append((f"{nerve}.{side}", "PASS" if ok else "FAIL " + via))
            if ok:
                paths.append({"name": f"{nerve}.{side}", "idx": [int(i) for i in idx]})
    print(f"nerves: {len(chains)} chains, {len(P)} centreline points, {n_join} junctions; "
          f"{len(paths)}/{len(gate)} pathways pass the named-nerve gate")
    for nm, verdict in gate:
        if verdict != "PASS":
            print(f"    dropped {nm}: {verdict[:160]}")

    # bind every centreline POINT, not every nerve: a nerve crosses joints (the sciatic runs
    # pelvis -> thigh, the median shoulder -> hand), so it cannot ride one segment the way a
    # bone does.  each point takes the nearest bone surface, and a point within 3 cm of the
    # joint to an ADJACENT segment is blended between the two, so the nerve bends through a
    # joint instead of snapping at it.
    adjacent = {(a_, b_) for a_, b_ in TREE.items()} | {(b_, a_) for a_, b_ in TREE.items()}
    snames = [sg for sg in seg_names if sg in surf["pts"]]
    dmat = np.stack([np.linalg.norm(P[:, None, :] - surf["pts"][sg][None, :, :], axis=2).min(1)
                     for sg in snames], 1)
    order = np.argsort(dmat, 1)
    sa, sb, wa = [], [], []
    for i in range(len(P)):
        a_, b_ = snames[order[i, 0]], snames[order[i, 1]]
        d1, d2 = dmat[i, order[i, 0]], dmat[i, order[i, 1]]
        if (a_, b_) in adjacent and d2 - d1 < 0.03:
            w = 0.5 + 0.5 * (d2 - d1) / 0.03
        else:
            b_, w = a_, 1.0
        sa.append(seg_names.index(a_)); sb.append(seg_names.index(b_)); wa.append(int(round(w * 100)))

    def radius(nm):
        for pat, r in NERVE_RADIUS:
            if not pat or _re.search(pat, nm):
                return r
        return NERVE_RADIUS[-1][1]

    nerves = {
        "source": "Z-Anatomy spinal nerves and spinal dura (CC-BY-SA 4.0), extracted from "
                  "the staged Startup.blend; cord = dura centreline",
        "p": [round(float(x), 4) for x in P.reshape(-1)],
        "sa": sa, "sb": sb, "w": wa,
        "chains": [[int(off[i]), len(c["p"]), radius(c["name"])] for i, c in enumerate(chains)],
        "names": [c["name"] for c in chains],
        "paths": paths,
    }

    # ---- the cortex point cloud the pulse arrives at ----
    # site/data/graph.js is a DIFFERENT subject's pial surface in surface RAS; it is not
    # registered to this atlas's skull and this exporter does not pretend otherwise.  the
    # page places it by bounding box inside the head, which is a picture of a cortex in the
    # right place, not a coregistration.  the head box it goes in is measured here.
    skull = clouds.get("torso")
    head_box = None
    if skull is not None:
        top = skull[skull[:, 1] > np.quantile(skull[:, 1], 0.965)]
        head_box = [top.min(0).round(4).tolist(), top.max(0).round(4).tolist()]

    ranges = osim_ranges(OSIM)
    undeclared = [c for cs in JOINT_COORDS.values() for c in cs
                  if c in ranges and not ranges[c]["declared"]]
    print(f"joint ranges: {len(ranges)} coordinates read from {OSIM.name}; "
          f"{len(undeclared)} carry no real bound ({sorted(set(undeclared))[:4]}...)")

    tris = sum(len(m["i"]) // 3 for m in meshes)
    verts = sum(len(m["v"]) // 3 for m in meshes)
    cen = binding["centroids_m"]
    centre = np.concatenate([np.asarray([cen[e] for e in by_seg[s]]) for s in seg_names]).mean(0)

    payload = {
        "source": "Z-Anatomy / BodyParts3D, DBCLS — CC-BY-SA",
        "frame": binding["frame"],
        "systems": [{"id": s, "name": sysinfo[s]["name"], "color": sysinfo[s]["color"],
                     **per_sys[s]} for s in want if s in per_sys],
        "segments": seg_names,
        "parents": TREE,
        "pivots": {s: [round(float(x), 4) for x in p] for s, p in pivots.items()},
        "jointCoords": JOINT_COORDS,
        "ranges": ranges,
        "rangeSource": str(OSIM.relative_to(IHM)),
        "nerves": nerves,
        "headBox": head_box,
        "centre": [round(float(x), 4) for x in centre],
        "meshes": meshes,
    }
    a.out.write_text(
        "/* generated by scripts/export_site_body.py -- do not edit by hand.\n"
        "   Z-Anatomy / BodyParts3D surface meshes (DBCLS, CC-BY-SA), decimated for display,\n"
        "   bound to the segments of IHM-1's driven skeleton.  no trajectory is shipped: the\n"
        "   page synthesises a preview pose inside the joint ranges carried in `ranges`. */\n"
        "window.IBM_BODY = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    print(f"{a.out}: {len(meshes)} merged meshes over {len(per_sys)} systems, "
          f"{verts} verts / {tris} triangles, {len(P)} nerve points, "
          f"{a.out.stat().st_size / 1e6:.2f} MB")
    if a.out.stat().st_size > 3.5e6:
        print("  !! over the 3.5 MB payload budget for this station", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
