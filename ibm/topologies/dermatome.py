"""the cutaneous afferent surface: which patch of real skin each fibre enters.

`ibm.topologies.nerve` says which cable an axon runs in.  `ibm.topologies.
afferent` says which relay it reaches.  Neither says **where on the body it
starts**, and until the body had discrete skin patches there was nowhere for it
to start: the integument was three whole-body shell entities -- epidermis,
dermis, hypodermis -- so "the skin" was one object and a cutaneous afferent could
only be attached to all of it at once.

That is not a resolution complaint.  A model whose skin is one object cannot have
a dermatome, cannot lose sensation over a territory when a root is cut, cannot
localise touch, and cannot say that the sole and the fingertip are innervated
differently -- and every one of those is a thing this programme's brain is
supposed to learn from.

IHM-1 now discretises the exterior component of the real skin mesh into patches,
each with a position on the surface, an area, a dermatome, a dorsal root and a
named trunk, and writes them to `dermatomes.json`.  This module is the read side:
it joins those patches to this repo's fibre-class axis, so one patch becomes a
delay per cutaneous class along its own measured route.

**Three things here are load-bearing and easy to get wrong.**

*The delay comes from the patch's own route, not from the trunk.*  Two patches on
the same trunk are different distances from the relay -- a fingertip and the
mid-forearm are both median territory -- so reading `TRUNK_LENGTH_MM` for a patch
would give every median patch the same latency.  `path_length_m` per patch is
what IHM measured, and it is what is used.

*Only the cutaneous classes are carried.*  Skin has A-beta, A-delta and C.  It
has no Ia and no Ib, because it has no spindles and no Golgi organs, and a patch
that reported an `ia` delay would be asserting a receptor that is not there.  The
trunk composition is intersected with the cutaneous set rather than taken whole,
which is exactly the asymmetry `TRUNK_COMPOSITION` exists to express.

*The face is counted apart.*  Trigeminal V1/V2/V3 are cranial territories with no
spinal root.  `coverage()` reports patches with and without a root level
separately and never merges them, because a single "N patches innervated" figure
would hide the fact that a fifth of the head has no segment to lesion.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Iterator

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.topologies.afferent import staged_edges
from ibm.topologies.nerve import FIBRE_VELOCITY_M_S, TRUNK_COMPOSITION
from ibm.vocabulary import Provenance

IHM_DERMATOMES = os.path.expanduser(
    "~/Documents/IHM-1/data/derived/canonical/dermatomes.json")

#: the fibre classes skin actually has.  a cutaneous patch has no muscle spindle
#: and no Golgi tendon organ, so `ia` and `ib` are absent by anatomy and not by
#: omission -- which is why this is an intersection and not a copy.
CUTANEOUS_CLASSES = ("abeta", "adelta", "c")


def load(path: str = IHM_DERMATOMES) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def patches(path: str = IHM_DERMATOMES) -> Iterator[dict]:
    """every skin patch, with a delay per cutaneous fibre class along its route.

    yields, per patch:
        id, side, dermatome, root_level, region, nerve_name, relay_id  -- IHM
        position_m, area_m2, path_length_m                             -- measured
        classes, delays_s                                              -- this repo
        undeclared_trunk    -- True if IHM routes it through a trunk this repo
                               does not declare, which is a join failure and not
                               a patch to quietly drop
    """
    d = load(path)
    if d is None:
        return
    for p in d.get("patches", []):
        trunk = p["nerve_name"]
        declared = TRUNK_COMPOSITION.get(trunk)
        classes = tuple(c for c in (declared or ()) if c in CUTANEOUS_CLASSES)
        L = float(p["path_length_m"])
        yield {
            "id": p["id"], "side": p["side"], "dermatome": p["dermatome"],
            "root_level": p["root_level"], "root_gap_reason": p.get("root_gap_reason"),
            "region": p["region"], "nerve_name": trunk, "nerve_id": p["nerve_id"],
            "relay_id": p["relay_id"], "brain_target_id": p.get("brain_target_id"),
            "position_m": p["position_m"], "area_m2": float(p["area_m2"]),
            "path_length_m": L,
            "classes": list(classes),
            # per PATCH, not per trunk: the fingertip and the mid-forearm are
            # both median territory and are not the same distance from the relay
            "delays_s": {c: L / FIBRE_VELOCITY_M_S[c][1] for c in classes},
            "undeclared_trunk": declared is None,
            "measured_dermatome_atlas": p.get("measured_dermatome_atlas", False),
            "assignment_rule": p.get("assignment_rule"),
        }


def coverage(path: str = IHM_DERMATOMES) -> dict:
    """how much of the real skin is innervated, with the denominators named."""
    d = load(path)
    if d is None:
        return {"ihm_available": False, "path": path}
    ps = list(patches(path))
    c = d["coverage"]
    rooted = [p for p in ps if p["root_level"]]
    cranial = [p for p in ps if not p["root_level"]]
    undeclared = [p for p in ps if p["undeclared_trunk"]]
    no_classes = [p for p in ps if not p["classes"]]
    return {
        "ihm_available": True,
        "skin_exterior_area_m2": c["exterior_component_area_m2"],
        "skin_raw_mesh_area_m2": c["raw_source_area_m2"],
        "area_denominator": c["area_denominator"],
        "patches": len(ps),
        "patch_area_m2": c["patch_area_m2"],
        "area_fraction_of_exterior": c["patch_area_fraction_of_exterior"],
        "patches_with_spinal_root": len(rooted),
        "patches_without_spinal_root": len(cranial),
        "patches_without_spinal_root_area_m2":
            c["patches_without_spinal_root_area_m2"],
        "patches_on_undeclared_trunk": len(undeclared),
        "patches_with_no_cutaneous_class": len(no_classes),
        "root_levels_covered": sorted({p["root_level"] for p in rooted}),
        "trunks_used": sorted({p["nerve_name"] for p in ps}),
        "patches_per_region": dict(sorted(Counter(p["region"] for p in ps).items())),
        "measured_dermatome_atlas": d["measured_dermatome_atlas"],
    }


def stages_for(patch: dict, relay_support: str = "body_surface",
               target_support: str = "tissue") -> tuple[dict, ...]:
    """one stage per cutaneous class, each with the patch's own route length."""
    return tuple(
        dict(name=f"{patch['id']}:{c}", **{"from": relay_support}, to=target_support,
             length_mm=1000.0 * patch["path_length_m"],
             velocity_m_s=FIBRE_VELOCITY_M_S[c][1], k=1)
        for c in patch["classes"])


@B.builder(
    "cutaneous_dermatome",
    produces=("patch_area_m2", "dermatome", "root_level", "conduction_delay_s"),
    supports=("body_surface", "tissue"),
    requires=("stages",),
    directed=True,
    metric="measured route from the skin patch to its segmental relay; one delay "
           "per cutaneous fibre class",
    doc="afferent edges from discrete dermatomal skin patches")
def cutaneous_dermatome(sites, *, stages=None, patch: dict | None = None,
                        default_k: int = 1):
    """edges from one named skin patch, or an explicit stage chain.

    giving neither raises rather than returning an empty topology, for the same
    reason `afferent_relay` does: a silently empty cutaneous topology is a body
    that cannot feel, with no error saying why.
    """
    if stages is None and patch is not None:
        stages = stages_for(patch)
    return staged_edges("cutaneous_dermatome", "cutaneous_dermatome", sites,
                        stages or (), default_k=default_k)


CUTANEOUS_DERMATOME = REGISTRY.topology(Topology(
    "cutaneous_dermatome",
    "discrete patches of the real integument, each with a position on the skin surface, "
    "an area, a dermatome and therefore a dorsal root, and a named cutaneous trunk.  it "
    "is the thing a cutaneous afferent innervates, and before it existed there was none: "
    "the body carried three whole-body shell entities for the entire integument, so "
    "'the skin' was one object and an afferent could only attach to all of it at once.  "
    "the consequences of that are not resolution: a model whose skin is one object "
    "cannot express a dermatome, cannot lose a territory when a root is cut, cannot "
    "localise touch, and cannot say that the sole and the fingertip are innervated "
    "differently.  the delay on an edge here comes from the patch's OWN measured route "
    "to its relay rather than from the trunk's length, because a fingertip and the "
    "mid-forearm are both median territory and are not the same distance away; and the "
    "fibre classes are the trunk's composition intersected with A-beta, A-delta and C, "
    "because skin has no spindles and no Golgi organs and a patch reporting a group Ia "
    "delay would be asserting a receptor that is not there",
    on=("body_surface", "tissue"),
    edge_features=("patch_area_m2", "dermatome", "root_level", "conduction_delay_s"),
    directed=True,
    builder="cutaneous_dermatome",
    provenance=Provenance.LITERATURE))


if __name__ == "__main__":
    c = coverage()
    if not c["ihm_available"]:
        raise SystemExit(f"IHM dermatomes.json not found at {c['path']}; run "
                         f"IHM-1 scripts/build_dermatome_patches.py")
    print(f"skin exterior {c['skin_exterior_area_m2']:.4f} m2 "
          f"(raw mesh {c['skin_raw_mesh_area_m2']:.4f} m2)")
    print(f"{c['patches']} patches covering {c['patch_area_m2']:.4f} m2 = "
          f"{100 * c['area_fraction_of_exterior']:.2f}% of the exterior")
    print(f"  with a spinal root   : {c['patches_with_spinal_root']} of {c['patches']}")
    print(f"  without (trigeminal) : {c['patches_without_spinal_root']} of "
          f"{c['patches']}  ({c['patches_without_spinal_root_area_m2']:.4f} m2)")
    print(f"  on a trunk this repo does not declare: "
          f"{c['patches_on_undeclared_trunk']}")
    print(f"  with no cutaneous fibre class        : "
          f"{c['patches_with_no_cutaneous_class']}")
    print(f"  root levels covered: {' '.join(c['root_levels_covered'])}")
    print(f"  regions: {c['patches_per_region']}")
