"""export a down-res musculoskeletal body, animated by a solved trajectory, to site/data/body.js.

the site's body station used to be a screen recording.  this instead ships the real
anatomy: Z-Anatomy surface meshes (CC-BY, attributed in the output), decimated, rigidly
driven per segment by a trajectory the body model actually solved.

the two halves of the body model do NOT share an id space.  the meshes are `za-*`
structures; the trajectory and the segment binding are over `body-bp3d-FJ*` entities.
they are bridged geometrically: each mesh is assigned the segment of the nearest bound
bp3d centroid, and each segment's per-frame rigid motion is recovered from its own
member centroids by Kabsch.  a mesh therefore moves because the segment it sits on
moved, which is what "the anatomy is bound to the driven skeleton" means.

    python scripts/export_site_body.py --motion gait-best --structures 150
"""
from __future__ import annotations

import argparse, gzip, json, sys
from pathlib import Path
import numpy as np

IHM = Path.home() / "Documents" / "IHM-1"
BIND = IHM / "data/derived/anatomy-segment-binding"
ATLAS = IHM / "data/derived/anatomy/extended"
OUT = Path(__file__).resolve().parents[1] / "site" / "data" / "body.js"


# the trajectory already carries the rigid transform per entity -- rotation_matrix and
# translation_m -- and its own projection note says every entity bound to a segment moves
# EXACTLY rigidly with it.  so one member per segment is the segment's motion, and fitting
# it back out of the centroids with Kabsch would only add error to a number already given.


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", default="gait-best", choices=["gait-best", "cortex-in-loop"])
    ap.add_argument("--structures", type=int, default=150, help="meshes to ship")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--rest", action="store_true",
                    help="ship the atlas rest pose only and no motion.  the stored "
                         "trajectories are all outside the body model's own declared "
                         "joint ranges (docs/LOG.md), so animating them tears the "
                         "skeleton apart at the joints -- the geometry is right and the "
                         "motion is not.")
    ap.add_argument("--max-tris", type=int, default=340, help="per mesh, after decimation")
    ap.add_argument("--systems", default="skeletal,muscular")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    want = set(a.systems.split(","))

    binding = json.loads((BIND / "binding.json").read_text())
    ent_seg = {k: v["segment"] for k, v in binding["entities"].items()}
    ent_role = {k: v.get("role") for k, v in binding["entities"].items()}
    segments = sorted(binding["segments"])

    if a.rest:
        # no trajectory at all: segments come from the binding, and the meshes ship in the
        # atlas rest pose, which IS a correctly articulated standing figure.
        by_seg = {}
        for e, sg in ent_seg.items():
            if e in binding["centroids_m"]:
                by_seg.setdefault(sg, []).append(e)
        by_seg = {k: v for k, v in by_seg.items() if len(v) >= 3}
        eids = [e for m in by_seg.values() for e in m]
        motion, times, pick, frames = {}, [], [], []
        print("rest pose only: no trajectory read")
    traj_path = BIND / f"trajectory-{a.motion}.json"
    if not a.rest:
      print(f"reading {traj_path.name} ({traj_path.stat().st_size / 1e6:.0f} MB)...", flush=True)
    if not a.rest:
      traj = json.loads(traj_path.read_text())
      frames = traj["frames"]
      pick = np.linspace(0, len(frames) - 1, min(a.frames, len(frames))).round().astype(int)
      eids = [e for e in frames[0]["entities"] if e in ent_seg]

      # per-segment member lists, and the rest pose those members sit in at frame 0
      by_seg: dict[str, list[str]] = {}
      for e in eids:
          by_seg.setdefault(ent_seg[e], []).append(e)
      # ---- per-segment rigid motion, frame by frame, read straight off one member ----
      motion = {s: [] for s in by_seg}
      for fi in pick:
          fr = frames[int(fi)]["entities"]
          for s, members in by_seg.items():
              e = fr[members[0]]
              R = np.asarray(e["rotation_matrix"], dtype=float)
              t = np.asarray(e["translation_m"], dtype=float)
              motion[s].append([round(float(v), 5) for v in R.reshape(-1)] +
                               [round(float(v), 5) for v in t])

      # a segment is rigid by construction, so every member must agree with the one we read.
      # check it rather than trust the note: a binding error would show up here as a spread.
      worst = 0.0
      for s, members in by_seg.items():
          fr = frames[int(pick[len(pick) // 2])]["entities"]
          R0 = np.asarray(fr[members[0]]["rotation_matrix"], dtype=float)
          for e in members[1:40]:
              worst = max(worst, float(np.abs(np.asarray(fr[e]["rotation_matrix"], dtype=float) - R0).max()))
      print(f"segment rigidity: worst member disagreement {worst:.2e} (expect ~0)")
      if worst > 1e-6:
          print("  !! members of a segment disagree -- the binding is not rigid", file=sys.stderr)
      times = [round(float(frames[int(fi)]["time_s"]), 4) for fi in pick]

      # sanity: apply the transforms to the rest centroids and check the body is a body.
      # a wrong convention (about the origin vs about the centroid) shows up instantly as a
      # figure several metres across or flung away from the floor.
      c0 = np.array([binding["centroids_m"][e] for e in eids])
      segi = [ent_seg[e] for e in eids]
      mid = len(pick) // 2
      moved = np.zeros_like(c0)
      for i, (c, sg) in enumerate(zip(c0, segi)):
          f = motion.get(sg, [None] * len(pick))[mid]
          if f is None:
              moved[i] = c
              continue
          R = np.array(f[:9]).reshape(3, 3)
          moved[i] = R @ c + np.array(f[9:])
      ext = moved.max(0) - moved.min(0)
      print(f"posed body extent {ext.round(3)} m, centroid {moved.mean(0).round(3)}")
      if not (1.2 < ext.max() < 2.4):
          print(f"  !! extent {ext.max():.2f} m is not a human -- transform convention is wrong",
                file=sys.stderr)
          return 2

    # ---- the meshes, and which segment each rides on ----
    man = json.loads((ATLAS / "manifest_fragment.json").read_text())
    sysname = {s["id"]: s for s in man["systems"]}
    structures = [s for s in man["structures"] if s.get("system") in want and s.get("kind") == "mesh"]

    seg_centroids = {s: np.array([binding["centroids_m"][e] for e in m]) for s, m in by_seg.items()}
    seg_names = list(seg_centroids)

    # the nearest-neighbour pool is BONES ONLY, and the vote is over a mesh's VERTICES.
    # taking the nearest of all 4,000 bound entities to a mesh's CENTROID -- which is what
    # this did first -- votes against a pool that is three-quarters vessels, muscles and
    # soft organs, and picks a segment from one interior point of a long bone.  femurs and
    # humeri came out attached to the wrong segment and the skeleton visibly came apart at
    # the joints.  `nearest_bone_group_vertex_vote` is the basis the binding itself used.
    bone_pts, bone_owner = [], []
    for si, sg in enumerate(seg_names):
        for e in by_seg[sg]:
            if ent_role.get(e) == "rigid_bone":
                bone_pts.append(binding["centroids_m"][e])
                bone_owner.append(si)
    if len(bone_pts) < 20:
        print(f"  !! only {len(bone_pts)} bone anchors -- falling back to all entities", file=sys.stderr)
        bone_pts = [binding["centroids_m"][e] for sg in seg_names for e in by_seg[sg]]
        bone_owner = [i for i, sg in enumerate(seg_names) for _ in by_seg[sg]]
    all_pts = np.asarray(bone_pts, dtype=float)
    all_owner = np.asarray(bone_owner, dtype=int)
    print(f"bone anchors: {len(all_pts)} across {len(set(all_owner))} segments")

    # ship the biggest structures first WITHIN EACH SEGMENT, not globally.  sorting by size
    # across the whole body spends the whole budget on the torso and ships a figure with no
    # legs -- which is what the first run produced.  a per-segment quota keeps the silhouette.
    structures.sort(key=lambda s: -(s.get("source_triangles") or s.get("original_faces") or 0))

    weak = []

    def seg_of_mesh(v, name):
        """the segment whose bone anchor best represents the WHOLE mesh.

        a modal vote over each vertex's nearest anchor is biased by anchor density along an
        elongated bone: the femur's distal vertices are all nearest the patella anchor, so
        the femur was assigned to `patella_l` and rode the kneecap.  taking the anchor that
        minimises MEAN distance to every vertex asks which bone this mesh actually is,
        which is the question, and the patella anchor loses it badly over a whole femur.
        """
        sample = v if len(v) <= 256 else v[np.linspace(0, len(v) - 1, 256).astype(int)]
        d = np.linalg.norm(sample[:, None, :] - all_pts[None, :, :], axis=2)
        mean_to_anchor = d.mean(0)
        best = int(mean_to_anchor.argmin())
        win = int(all_owner[best])
        # margin against the best anchor of any OTHER segment
        other = mean_to_anchor.copy()
        other[all_owner == win] = np.inf
        rival = float(other.min()) if np.isfinite(other).any() else np.inf
        ratio = mean_to_anchor[best] / rival if rival else 0.0
        if ratio > 0.92:
            weak.append((name, seg_names[win], round(float(ratio), 2)))
        return seg_names[win]

    quota = max(2, a.structures // max(1, len(seg_names)))
    taken: dict[str, int] = {}
    out_meshes, skipped, deferred = [], 0, []
    for st in structures:
        if len(out_meshes) >= a.structures:
            break
        gp = IHM / st["geometry_path"]
        if not gp.exists():
            skipped += 1
            continue
        g = json.loads(gzip.open(gp).read())
        pos = np.asarray(g["positions"], dtype=np.float64).reshape(-1, 3)
        idx = np.asarray(g["indices"], dtype=np.int64)
        if pos.size == 0 or idx.size == 0:
            skipped += 1
            continue
        ext = float(np.linalg.norm(pos.max(0) - pos.min(0)))
        cell = max(ext / 26.0, 1e-4)
        for _ in range(7):                       # tighten until under budget
            v, tri = decimate(pos, idx, cell)
            if len(tri) <= a.max_tris or len(tri) == 0:
                break
            cell *= 1.32
        if len(tri) < 4:
            skipped += 1
            continue
        seg = seg_of_mesh(v, st["name"])
        if taken.get(seg, 0) >= quota:
            deferred.append((seg, v, tri, st))      # comes back only if budget is left over
            continue
        taken[seg] = taken.get(seg, 0) + 1
        # positions are emitted in the segment's REST frame, so the renderer only ever
        # applies that segment's matrix -- no per-vertex work at 60fps
        out_meshes.append({
            "n": st["name"], "sys": st["system"], "seg": seg,
            "col": st.get("color") or sysname.get(st["system"], {}).get("color", "#999"),
            "v": [round(float(x), 4) for x in v.reshape(-1)],
            "i": [int(x) for x in tri.reshape(-1)],
        })

    # spend anything left over on the segments that had more to give
    for seg, v, tri, st in deferred:
        if len(out_meshes) >= a.structures:
            break
        out_meshes.append({
            "n": st["name"], "sys": st["system"], "seg": seg,
            "col": st.get("color") or sysname.get(st["system"], {}).get("color", "#999"),
            "v": [round(float(x), 4) for x in v.reshape(-1)],
            "i": [int(x) for x in tri.reshape(-1)],
        })
    if weak:
        print(f"  {len(weak)} mesh(es) assigned with a thin margin, e.g. {weak[:4]}", file=sys.stderr)
    import collections as _c
    cov = _c.Counter(m["seg"] for m in out_meshes)
    missing = [s for s in seg_names if s not in cov]
    print(f"segment coverage: {len(cov)}/{len(seg_names)} segments" +
          (f"  MISSING {missing}" if missing else ""))

    tris = sum(len(m["i"]) // 3 for m in out_meshes)
    centre = np.concatenate([seg_centroids[s] for s in seg_names]).mean(0)

    payload = {
        "attribution": man.get("limitations", {}) if isinstance(man.get("limitations"), dict) else {},
        "source": "Z-Anatomy / BodyParts3D, DBCLS — CC-BY",
        "motion": a.motion,
        "duration_s": times[-1] if times else 0,
        "times": times,
        "centre": [round(float(x), 4) for x in centre],
        "segments": sorted(motion),
        "motionBySegment": motion,
        "meshes": out_meshes,
    }
    a.out.write_text(
        "/* generated by scripts/export_site_body.py -- do not edit by hand.\n"
        "   Z-Anatomy / BodyParts3D surface meshes (DBCLS, CC-BY), decimated for display,\n"
        f"   driven by the {a.motion} trajectory the body model solved. */\n"
        "window.IBM_BODY = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    print(f"{a.out}: {len(out_meshes)} meshes, {tris} triangles, "
          f"{len(times)} frames over {payload['duration_s']}s, "
          f"{a.out.stat().st_size / 1e6:.1f} MB  (skipped {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
