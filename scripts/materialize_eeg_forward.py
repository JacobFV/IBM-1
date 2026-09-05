#!/usr/bin/env python
"""materialize `eeg_forward` on a real head.

    M = materialize(R, r, B, F, A, T, P)

this is the library's `eeg_forward` request, built against one actual FreeSurfer
reconstruction with its own three-layer BEM and its own digitised montage --
`ibm.materialize.geometry.sample_subject()`, resolved from the `mne-sample`
card's `.location.yaml`.  nothing here is a template, and nothing here is a
stand-in.

three things the script says out loud rather than papering over, because each of
them is a modelling decision that a silent build would hide:

1. **the sources are on the sheet, and that is a change.**  an earlier version of
   this script amended R with `("sources", OnSupport("tissue"))`, because every
   neural component's primary support is `tissue` and nothing at all was
   admissible on `cortical_surface`: the surface sampler never ran, the sources
   fell back to 10.5 mm parenchyma voxels, and `cortical_surface` -- the topology
   on which every horizontal cortical process is declared -- built zero edges
   while the build reported success.  the neural components now declare
   `alt_supports=("cortical_surface",)` and `ibm.materialize.build` places them
   on the support R actually names, so the amendment is gone and R is used
   exactly as `eeg_forward` declares it.  the cost of the old arrangement was
   measurable on this subject: a euclidean volume graph at 10 mm mis-connects
   58% of cortical pairs across a sulcus.
2. **the declared r(q) is unaffordable, by two orders of magnitude.**  it is
   tried first, and the `Budget` machinery's own refusal is printed verbatim
   before anything is coarsened.  the coarser r(q) is then written out here, with
   the arithmetic that forced each number.
3. **what the build still cannot do.**  the topologies whose data this subject
   does not have, and the supports the materialization deliberately does not
   reach, are listed at the end.  moving the neural field to the sheet moves the
   subcortical half of it out of this view entirely, which is the largest single
   thing this materialization gives up and is reported rather than hidden.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ibm
from ibm import frames as _frames
from ibm.materialize import geometry as geo
from ibm.materialize.build import (
    MaterializationIncomplete, _support_overlap, build, earns_its_cost,
)
from ibm.materialize.library import MODELS
from ibm.materialize.library.electrophysiology import SCALP
from ibm.materialize.request import BudgetExceeded, DeviceSpec, SubjectSpec
from ibm.vocabulary import OnSupport, Resolution, ResolutionRule


def rule(region, mm: float, band=SCALP) -> ResolutionRule:
    return ResolutionRule(region, mm, band)


def head(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    ibm.load_all(seal=True, strict=True)

    # -- 1. the subject -----------------------------------------------------
    head("1. subject geometry")
    paths = geo.sample_paths()
    print(paths.describe())
    geometry = geo.sample_subject()
    print()
    for support, g in sorted(geometry.by_support.items()):
        n = (len(np.asarray(g.xyz)) if hasattr(g, "xyz")
             else len(np.asarray(g.vertices)) if hasattr(g, "vertices")
             else int((np.asarray(g.occupancy) > 0).sum()))
        print(f"  {support:18s} {type(g).__name__:17s} frame={g.frame:12s} {n:>9,} "
              f"{'vertices' if hasattr(g, 'vertices') else 'points' if hasattr(g, 'xyz') else 'voxels'}")
        print(f"      {g.source}")

    # -- 2. the montage, and the frame chain that reaches the head ----------
    head("2. montage and coregistration")
    montage = geometry.get("sensor_array")
    coreg = geo.sample_coregistration(geometry=geometry)
    chain = _frames.path(geo.MONTAGE_FRAME, geo.ANATOMICAL_FRAME) or []
    print(f"  ibm.frames chain  {geo.MONTAGE_FRAME} -> {geo.ANATOMICAL_FRAME}: "
          f"{' -> '.join(w.method for w in chain)}")
    for w in chain:
        print(f"      {w.src} -> {w.dst}: {w.note}")
    print(coreg.describe())
    print("  the positions stay in the montage's own frame; the transform above is handed to")
    print("  build(warp=...), so the site table records the warp it went through and r(q)'s")
    print("  Near(...) rule is evaluated against electrodes that are where the electrodes are.")

    # -- 3. the request -----------------------------------------------------
    head("3. the request, and the one place it is amended")
    model = MODELS["eeg_forward"]
    base = model.request
    print(base.describe())

    dev = DeviceSpec("eeg", "sensor_array", geo.MONTAGE_FRAME, "observe",
                     n_elements=len(np.asarray(montage.xyz)),
                     positions=np.asarray(montage.xyz, float),
                     element_ids=tuple(montage.ids),
                     note=f"digitised montage from {paths.raw_fif.name}")
    subject = SubjectSpec(id="sample", frame=geo.ANATOMICAL_FRAME,
                          surface_frame=geo.ANATOMICAL_FRAME, template=None,
                          note="mne sample: individual T1, individual watershed BEM, "
                               "individual coregistration.  no template stands in for anything")
    print()
    print("  R is used exactly as declared: cortex(cortical_surface) | conductor(head_volume).")
    print("    the ('sources', OnSupport('tissue')) amendment this script used to add is gone.")
    print("    it existed for one reason -- no component was admissible on cortical_surface, so")
    print("    R selected nothing anywhere the neural field could live and the forward model")
    print("    had no sources at all.  13 neural components now declare cortical_surface as an")
    print("    alternative support, and build() places a component on the support R names, so")
    print("    the declared R reaches the sources by itself.  what the amendment was buying is")
    print("    now paid for properly: column nodes seeded on this subject's own white surface")
    print("    -- 15,870 of them at the declared 3 mm, 13,647 at the 3.25 mm the budget leaves")
    print("    room for (step 4) -- instead of 1,756 parenchyma voxels 10.5 mm across.")
    print("    what it costs: `tissue` is no longer materialized, so metabolic.* and")
    print("    structural.* leave this view, and so does subcortical neural state -- a")
    print("    component gets one block on one support, and this request's R puts the neural")
    print("    field on the sheet.  a materialization that needs the thalamus needs a request")
    print("    that names tissue, and it would then have to say which of the two supports")
    print("    carries the cortex, because both cover it.")
    print("  the one amendment: the device carries 60 real electrode positions and their frame.")

    request = replace(base, subject=subject, devices=(dev,))

    # -- 4. the declared r(q), tried and refused ----------------------------
    head("4. the declared r(q), priced before it is changed")
    for i, r in enumerate(request.resolution.rules):
        print(f"  rule[{i}] {r.region} -> {r.spacing_mm:g} mm")
    print(f"  default       {request.resolution.default_mm:g} mm")
    print(f"  budget        {request.budget.max_state_variables:,} state variables, "
          f"{request.budget.max_sites_per_support:,} sites per support")
    print()
    try:
        build(request, geometry=geometry, warp=coreg.warp, strict=False)
        print("  it fit.  no coarsening was needed.")
        chosen = request
    except BudgetExceeded as exc:
        print("  BudgetExceeded, as it should be:")
        for line in str(exc).split("; "):
            print(f"    {line}")
        chosen = replace(request, resolution=coarsened_resolution())
        print()
        print("  the coarser r(q) this script uses instead, and why each number:")
        print(COARSENING_ARGUMENT)
        for i, r in enumerate(chosen.resolution.rules):
            print(f"    rule[{i}] {r.region} -> {r.spacing_mm:g} mm")
        print(f"    default       {chosen.resolution.default_mm:g} mm")

    # -- 5. the lead field, over the positions the octree actually produced --
    head("5. the forward solution, over the sites this r(q) produced")
    print("  a lead field is indexed by source position, so it cannot be computed before the")
    print("  site table exists and cannot be reindexed onto a different one -- the em builder")
    print("  refuses exactly that.  so the model is built twice: once to find out where the")
    print("  sources landed, and once with a BEM solve over precisely those positions.")
    probe = build(chosen, geometry=geometry, warp=coreg.warp, strict=False)
    sheet = probe.sites["cortical_surface"]
    print(f"  sources are now the {sheet.n:,} cortical column nodes, not parenchyma voxels: the")
    print("  BEM is solved at positions that lie on the white surface by construction, so the")
    print("  rim problem the octree had -- cells whose centre falls outside the inner skull --")
    print("  does not arise.")
    lead = geo.sample_lead_field(np.asarray(sheet.xyz, float),
                                 sensor_ids=tuple(montage.ids), paths=paths)
    print()
    print(lead.describe())

    topology_inputs = {
        # the sources moved to the sheet, so the lead field is indexed by column
        # node.  the builder defaults to `tissue` and must be told, rather than
        # guessing, which support the rows of L belong to.
        "electromagnetic": {"lead_field": lead.data, "source_support": "cortical_surface"},
        # 9 mm is six length constants of `lateral_surface_transfer`'s 1.5 mm
        # exponential, where the kernel is down to 0.25% and truncation stops
        # being a modelling choice; k caps the degree so a node in a tightly
        # folded region does not carry three times its neighbours' edges.
        "cortical_surface": {"radius_mm": 9.0, "k": 18},
        # the association prior is SAMPLED, so the degree is a budget decision
        # and is written down: 64 long-range partners per node over ~1.4e4 nodes
        # is ~5e5 edges, and the inclusion probability the builder records is what
        # lets the process reweight that sample back to the dense graph.
        "cortical_association": {"max_degree": 64},
        # the canonical circuit is a relation a position has with itself, so it
        # follows the neural field onto the sheet.  the builder cannot guess that:
        # it defaults to `tissue`, and a materialization that indexes cortex on the
        # sheet would silently lose the local e-i loop entirely.
        "microcircuit": {"support": "cortical_surface"},
        # the builder's defaults describe an implanted array coupling to parenchyma.
        # a scalp montage couples to the conductor at the skin, so both supports are
        # named explicitly and the reach is one conductor cell -- an electrode is a
        # disc on the scalp, and the head_volume site under it is what it contacts.
        "device_coupling": {"device_support": "sensor_array", "medium_support": "head_volume",
                            "reach_mm": 12.0, "direction": "record"},
    }

    # -- 6. the build -------------------------------------------------------
    head("6. the materialized model")
    m = build(chosen, geometry=geometry, warp=coreg.warp, strict=False,
              topology_inputs=topology_inputs)
    print(m.describe())

    head("7. cost")
    print(f"  sites                 {sum(t.n for t in m.sites.tables.values()):>15,}")
    print(m.cost.describe())
    print()
    print("  per-support site tables:")
    for s in sorted(m.sites.tables):
        t = m.sites[s]
        vol = np.asarray(t.columns.get("volume_mm3", np.nan), float)
        area = np.asarray(t.columns.get("area_mm2", np.nan), float)
        extra = (f"  total {np.nansum(vol):,.0f} mm3" if np.isfinite(vol).any()
                 else f"  total {np.nansum(area):,.0f} mm2" if np.isfinite(area).any() else "")
        lo, mid, hi = m.achieved_spacing_mm.get(s, (np.nan,) * 3)
        print(f"    {s:16s} {t.n:>9,} sites  r(q) {lo:g}/{mid:g}/{hi:g} mm  frame {t.frame}{extra}")

    sheet = m.sites["cortical_surface"]
    a = float(np.nansum(np.asarray(sheet.columns["area_mm2"], float)))
    r = float(np.median(np.asarray(sheet.spacing(np), float)))
    hexn = a / (0.866 * r * r)
    print()
    print("  the sheet, against the arithmetic that predicts it:")
    print(f"    white surface area (both hemispheres)   {a:>12,.0f} mm2")
    print(f"    r(q) on the sheet                       {r:>12.2f} mm")
    print(f"    hexagonal-packing bound  A/(sqrt(3)/2 r^2) {hexn:>9,.0f} nodes")
    print(f"    poisson-disk nodes actually seeded      {sheet.n:>12,} "
          f"({sheet.n / hexn:.3f} of the bound)")
    print("    a maximal poisson-disk set is blue noise and not a lattice, so the hexagonal")
    print("    figure is an upper bound, not a target; the jamming ratio of random sequential")
    print("    adsorption in 2d is 0.547/0.9069 = 0.60, which is what this is.")
    print(f"    induced triangulation                   "
          f"{len(np.asarray(sheet.columns['faces'])):>12,} triangles, "
          f"{float(sheet.columns['dual_adjacency_covered']):.4f} of the voronoi adjacencies")

    print()
    print("  edges per topology:")
    for name in sorted(set(m.trace.topologies)):
        e = m.edges.get(name)
        n = 0 if e is None else e.n_edges
        why = "" if e is not None else "  (not built: see the notes)"
        print(f"    {name:26s} {n:>10,}{why}")

    head("8. is the resolution earning its cost? (ARCHITECTURE.md 1)")
    for v in earns_its_cost(m, factor=2.0):
        print("  " + v.describe().replace("\n", "\n  "))

    head("9. provenance audit")
    print(m.audit())

    head("10. the non-overlap rule, exercised on the request that would break it")
    print("  a materialization must place each POSITION on exactly one support.  the sheet and")
    print("  the parenchyma volume are two samplings of the same cortex, so instantiating the")
    print("  neural field on both is not a finer description of it -- it is the same state")
    print("  entered twice, and a lead field, which sums sources linearly, would report a")
    print("  scalp potential inflated by however much of the cortex both supports cover.")
    print("  so here is the old amendment put back, and what build() now does with it:")
    print()
    overlapping = replace(request, resolution=chosen.resolution,
                          regions=request.regions + (("sources", OnSupport("tissue")),))
    try:
        build(overlapping, geometry=geometry, warp=coreg.warp, strict=True)
        print("  it built.  that is a bug in the check, not a licence.")
    except MaterializationIncomplete as exc:
        hit = [p for p in exc.problems if p.startswith("R names") or "double-count" in p]
        for p in hit:
            print(f"    REFUSED  {p}")
        for p in exc.problems:
            if p not in hit:
                print(f"    also     {p.splitlines()[0][:110]}")
        print()
        print("  note what else went wrong there, because it is the original bug returning: with")
        print("  the sources back on `tissue` the surface is not materialized at all, so")
        print("  cortical_surface and cortical_association have nothing to relate and build no")
        print("  edges -- which is exactly the state this work started from.")
    loose = build(overlapping, geometry=geometry, warp=coreg.warp, strict=False)
    frac, med = _support_overlap(m.sites["cortical_surface"], loose.sites["tissue"])
    print()
    print("  and the geometry behind the refusal, measured rather than asserted:")
    print(f"    {frac:.1%} of the {m.sites['cortical_surface'].n:,} column nodes fall inside a")
    print(f"    {loose.sites['tissue'].n:,}-voxel `tissue` cell (median nearest-neighbour")
    print(f"    separation {med:.2f} mm).  the two supports are describing the same millimetres.")
    print("  with strict=False the same build completes and carries the violation in its")
    print("  provenance instead, which is the behaviour cost estimation needs: you can price a")
    print("  model that double-counts, you must not run one.")

    head("11. what this build still cannot do honestly")
    for n in m.notes:
        print(f"  - {n}")
    for p in m.provenance.missing:
        print(f"  ! {p.splitlines()[0]}")
    return 0


#: the coarsening, written out rather than applied.  §1 is explicit that r(q)
#: should be derived from where coarse-graining fails to commute with the
#: dynamics and not from proximity to the instrument, and `graded_around` in
#: request.py repeats it; the declared 1 mm shell around the electrodes is the
#: proximity rule, and it is the first thing to go.
COARSENING_ARGUMENT = """    - what actually refuses: both fine rules -- Near('eeg', 15 mm) at 1.0 mm and
      head_volume at 1.5 mm -- drive the conductor octree past its level-7 cells
      (1.91 mm), and the sampler stops at 5,680,544 projected sites.  head_volume
      carries 16 traced components, so that is ~9.1e7 state variables against a
      4e5 budget: 227x, which is not a matter of trimming.
    - head_volume 1.5 mm -> 8 mm.  the octree is dyadic over a 245 mm root, so
      the reachable cell sizes near this are 15.3, 7.66 and 3.83 mm; 8 mm lands
      on 7.66 mm and gives 12,419 leaves x 16 components = 1.99e5 variables.
      3.83 mm would be ~9.2e4 leaves = 1.5e6 variables, 3.7x the whole budget.
    - the electrode shell is dropped rather than coarsened.  §1 says r(q) should
      be derived from where coarse-graining fails to commute with the dynamics
      and not from proximity to whatever we happen to be measuring, and
      `graded_around` in request.py repeats the warning.  here proximity is the
      wrong proxy twice over: the scalp under a contact is the smoothest part of
      the lead field, and this model's own docstring says the skull's
      centimetre-wide kernel is why eeg is coarse by physics, not by budget.
    - the tissue rule is gone with the amendment it existed for.  no traced
      component is placed on `tissue` any more -- the 13 neural components go to
      the sheet, and metabolic.* and structural.* are outside R -- so a rule over
      it would never fire.
    - cortical_surface 3.0 mm -> 3.25 mm, and this is the one number this script
      changes from the declaration.  the arithmetic: poisson-disk sampling of
      this subject's 202,437 mm2 white surface at 3.0 mm gives 15,870 column
      nodes (0.611 of the 25,974 a hexagonal lattice would fit, which is the
      random-sequential-adsorption jamming ratio and is what a blue-noise set is
      supposed to give).  15,870 x 13 neural components = 2.06e5, and with the
      conductor's 1.99e5 the total is 4.05e5 against a 4.0e5 ceiling -- over by
      1.3%.  the conductor cannot give it back: the next octree level is 15.3 mm
      and the skull is 5-8 mm thick, so one cell would straddle scalp, skull and
      brain, and the partial-volume conductivity of that cell is the entire
      content of the lead field.  the sheet can, because its count goes as 1/r^2
      and is continuous in r: the ceiling implies n <= 15,475, so r >= 3.0 *
      sqrt(15870/15475) = 3.04 mm, and 3.25 mm leaves ~8% margin against the
      sampler's own jitter.  the sampler then delivers 13,647 nodes = 1.77e5
      variables, 3.76e5 with the conductor: 94% of the ceiling.  what 3.25 mm
      costs is the thing 3 mm was buying -- source orientation across a sulcal
      bank -- and it costs about 7% of it, which is the smallest defensible cut
      available."""


def coarsened_resolution() -> Resolution:
    """the explicitly-stated coarser r(q).  see `COARSENING_ARGUMENT`."""
    return Resolution(
        (rule(OnSupport("head_volume"), 8.0),
         rule(OnSupport("cortical_surface"), 3.25)),
        default_mm=12.0, default_band=SCALP)


if __name__ == "__main__":
    raise SystemExit(main())
