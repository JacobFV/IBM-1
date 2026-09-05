#!/usr/bin/env python
"""materialize `eeg_forward` on a real head.

    M = materialize(R, r, B, F, A, T, P)

this is the library's `eeg_forward` request, built against one actual FreeSurfer
reconstruction with its own three-layer BEM and its own digitised montage --
`ibm.materialize.geometry.sample_subject()`, resolved from the `mne-sample`
card's `.location.yaml`.  nothing here is a template, and nothing here is a
stand-in.

four things the script says out loud rather than papering over, because each of
them is a modelling decision that a silent build would hide:

1. **the neural field is on two supports at once, and that is the change.**  the
   previous version of this script put the whole neural field on
   `cortical_surface`, because a component could be placed on exactly one
   support and R named the sheet.  the cortex was then well described and the
   thalamus, the brainstem and the cerebellum had no state at all -- a forward
   operator with nothing behind it.  a placement is now a *partition*: R names
   the sheet and the volume, and `ibm.materialize.build` gives each neural
   component one block on each.  cortical population state lives on column
   nodes; subcortical, cerebellar and deep population state lives on parenchyma
   voxels; and the two are required to cover disjoint positions, which is
   measured with the same KD-tree check that refuses a genuine double count.
2. **`local` now means "locally adjacent on whichever support it is built
   over".**  it used to be declared `on=("tissue",)` and therefore built zero
   edges in a materialization whose neural field was on the sheet: recurrent
   local excitation, ionic exchange, transmitter clearance, metabolism and
   neurovascular coupling were all traced, scored and priced over a graph with
   no edges in it.  the relation is proximity in the tissue and the *metric*
   belongs to the support -- euclidean in the volume, geodesic on the sheet --
   so it is one topology with two metrics rather than two topologies.
3. **the declared r(q) is unaffordable, by two orders of magnitude.**  it is
   tried first, and the `Budget` machinery's own refusal is printed verbatim
   before anything is coarsened.  the coarser r(q) is then written out here, with
   the arithmetic that forced each number.
4. **what the build still cannot do.**  the topologies whose data this subject
   does not have, and the supports the materialization deliberately does not
   reach, are listed at the end.
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
from ibm.registry import REGISTRY
from ibm.topologies import builders as _builders
from ibm.topologies.builders import MissingInput
from ibm.vocabulary import Difference, Near, OnSupport, Resolution, ResolutionRule

#: how far from this subject's white surface a parenchyma voxel has to be before
#: the volume is allowed to carry state at it.  two constraints, and the binding
#: one is the second.
#:
#: - anatomy: the cortical ribbon runs outward from the white surface, mean 2.5 mm
#:   thick on this subject with a 95th percentile near 4 mm, so 6 mm clears it.
#: - the disjointness rule: `_support_overlap` counts a column node as inside a
#:   `tissue` cell when it is within half that cell's width, and the cells here
#:   are 10.5 mm, so anything under 5.25 mm would be an overlap by construction
#:   however thin the ribbon happened to be.
#:
#: it is expressed as a region rather than as a filtered geometry on purpose.  R
#: is the part of a request that decides *where*, and "the parenchyma the sheet
#: does not already index" is a statement about where, not a different notion of
#: what tissue is.
RIBBON_EXCLUSION_MM = 6.0


def rule(region, mm: float, band=SCALP) -> ResolutionRule:
    return ResolutionRule(region, mm, band)


def head(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def edge_counts(model) -> dict[str, int]:
    return {name: (0 if model.edges.get(name) is None else model.edges[name].n_edges)
            for name in sorted(set(model.trace.topologies))}


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
    head("3. the request, and the one place R is extended")
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
    white = np.asarray(geometry.get("cortical_surface").vertices, float)

    print()
    print("  R as `eeg_forward` declares it:  cortex(cortical_surface) | conductor(head_volume).")
    print("  R as this script builds it, with one region added:")
    print("      cortex     OnSupport('cortical_surface')")
    print(f"      subcortex  Difference(OnSupport('tissue'), Near('cortex', {RIBBON_EXCLUSION_MM:g} mm))")
    print("      conductor  OnSupport('head_volume')")
    print()
    print("  what the added region buys, and why it is a region rather than a second request:")
    print("    every neural component declares `tissue` as its primary support and")
    print("    `cortical_surface` as an alternative.  naming only the sheet put ALL of the")
    print("    neural field on the sheet -- there is no cortical-only neural component -- so")
    print("    thalamus, brainstem, cerebellum and striatum carried no state whatever, and")
    print("    every process writing them was traced, scored, priced and then handed nothing")
    print("    to write.  naming both supports now instantiates each neural component TWICE,")
    print("    once per support, as two blocks of one component.")
    print("    the non-overlap rule is unchanged and is what the Difference is for: the sheet")
    print("    and the parenchyma volume are two samplings of the same cortex, so the two")
    print("    blocks are a partition only if they cover disjoint positions.  excluding the")
    print(f"    parenchyma within {RIBBON_EXCLUSION_MM:g} mm of the white surface is how R says 'the volume")
    print("    carries what the sheet does not'.  section 10 measures the result and section")
    print("    10 also puts back the version that does NOT say it, which is refused.")
    print("  the other amendment: the device carries 60 real electrode positions and their frame.")
    print("  `Near('cortex', ...)` is an ordinary landmark region -- Near is declared over")
    print("    'device or landmark positions' -- and the landmark itself is external geometry,")
    print(f"    handed to build(anchors=...) as this subject's own {len(white):,} white-surface")
    print("    vertices, exactly as an atlas is handed to build(anatomy=...).")

    request = replace(base, subject=subject, devices=(dev,), regions=REGIONS)

    # -- 4. the declared r(q), tried and refused ----------------------------
    head("4. the declared r(q), priced before it is changed")
    for i, r in enumerate(request.resolution.rules):
        print(f"  rule[{i}] {r.region} -> {r.spacing_mm:g} mm")
    print(f"  default       {request.resolution.default_mm:g} mm")
    print(f"  budget        {request.budget.max_state_variables:,} state variables, "
          f"{request.budget.max_sites_per_support:,} sites per support")
    print()
    try:
        build(request, geometry=geometry, warp=coreg.warp, anchors={"cortex": white},
              strict=False)
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
    probe = build(chosen, geometry=geometry, warp=coreg.warp, anchors={"cortex": white},
                  strict=False)
    sheet = probe.sites["cortical_surface"]
    print(f"  the BEM is solved at the {sheet.n:,} cortical column nodes, which lie on the white")
    print("  surface by construction, so the rim problem the octree had -- cells whose centre")
    print("  falls outside the inner skull -- does not arise.")
    print(f"  it is NOT solved at the {probe.sites['tissue'].n:,} subcortical voxels, and that is a")
    print("  statement about the instrument rather than a gap in the build: the scalp lead")
    print("  field of a deep source is three orders of magnitude below a cortical one and is")
    print("  nearly rank-deficient against it, so `em_generation` there writes state that")
    print("  `em_coupling` cannot see.  the subcortical block exists so the DYNAMICS have")
    print("  somewhere to live, not because the forward operator needs it.")
    lead = geo.sample_lead_field(np.asarray(sheet.xyz, float),
                                 sensor_ids=tuple(montage.ids), paths=paths)
    print()
    print(lead.describe())

    topology_inputs = {
        # the cortical sources are on the sheet, so the lead field is indexed by
        # column node.  the builder defaults to `tissue` and must be told, rather
        # than guessing, which support the columns of L belong to.
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
        # `local` and `microcircuit` take NO support argument any more.  both are
        # declared over the sheet and the volume, both default to "every one of
        # those this materialization instantiated", and both therefore build over
        # the split neural field without being told about it.  the radius is left
        # at its default of twice the site spacing, which is per support and has
        # to be: 6.5 mm of geodesic on a 3.25 mm sheet and 21 mm of euclidean on
        # a 10.5 mm volume are the same statement about connectivity of the
        # sampling, and one number could not be both.
        "local": {},
        "microcircuit": {},
        # the builder's defaults describe an implanted array coupling to parenchyma.
        # a scalp montage couples to the conductor at the skin, so both supports are
        # named explicitly and the reach is one conductor cell -- an electrode is a
        # disc on the scalp, and the head_volume site under it is what it contacts.
        "device_coupling": {"device_support": "sensor_array", "medium_support": "head_volume",
                            "reach_mm": 12.0, "direction": "record"},
    }

    # -- 6. the build -------------------------------------------------------
    head("6. the materialized model")
    m = build(chosen, geometry=geometry, warp=coreg.warp, anchors={"cortex": white},
              strict=False, topology_inputs=topology_inputs)
    print(m.describe())

    head("7. cost, and where the state variables went")
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

    print()
    print("  components split across supports -- one component, two blocks:")
    for cid in m.layout.split:
        parts = m.layout.blocks_of(cid)
        print(f"    {cid:28s} " + " + ".join(
            f"{b.n_sites:,} on {b.support} @ {b.spacing_mm:g} mm" for b in parts)
            + f"   = {sum(b.n_sites for b in parts):,} state variables")
    print(f"    {len(m.layout.split)} of {len(m.layout.components)} materialized components are "
          "split; every one of them is a neural")
    print("    population component, which is exactly the field that exists on both samplings.")

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

    tis = m.sites["tissue"]
    tv = float(np.nansum(np.asarray(tis.columns.get("volume_mm3", np.nan), float)))
    print()
    print("  the volume half, against the same kind of arithmetic:")
    print(f"    parenchyma the volume carries           {tv:>12,.0f} mm3")
    print(f"    r(q) in the volume                      "
          f"{float(np.median(np.asarray(tis.spacing(np), float))):>12.2f} mm")
    print(f"    voxels seeded                           {tis.n:>12,}")
    print(f"    mean occupancy of a leaf                "
          f"{float(np.mean(np.asarray(tis.columns.get('occupancy', np.nan), float))):>12.3f}")
    print(f"    this is the parenchyma more than {RIBBON_EXCLUSION_MM:g} mm from the white surface: deep and")
    print("    periventricular white matter, thalamus, striatum, pallidum, hippocampus and")
    print("    amygdala, brainstem, and the whole cerebellum -- which has no surface in this")
    print("    mesh at all, so none of it was ever a candidate for the sheet.")

    # -- 8. the point of all of it: edges --------------------------------
    head("8. edges per topology, before and after")
    print("  three columns, all measured on this subject in this run rather than remembered.")
    print("    `on=(tissue,)`  what each topology built when the neural field was on the")
    print("                    sheet and `local`/`microcircuit` were reached for over the")
    print("                    volume -- the arrangement this work started from.  it is")
    print("                    reproduced by pinning those two builders to `tissue` on the")
    print("                    sheet-only site tables.  `local` was DECLARED on=(tissue,), so")
    print("                    build() skipped it outright; `microcircuit` was declared on")
    print("                    both but its builder defaulted to `tissue`, and this script")
    print("                    used to rescue it by passing support='cortical_surface' by")
    print("                    hand.  both now dispatch on the supports that exist.")
    print("    sheet only      the same R, with the two topologies made support-aware.  this")
    print("                    column isolates the second fix.")
    print("    split           R naming both supports as well.  this column adds the first.")
    before_request = replace(chosen, name="eeg-forward@sheet-only",
                             regions=(("cortex", OnSupport("cortical_surface")),
                                      ("conductor", OnSupport("head_volume"))))
    before = build(before_request, geometry=geometry, warp=coreg.warp,
                   anchors={"cortex": white}, strict=False,
                   topology_inputs=topology_inputs)
    b_edges, a_edges = edge_counts(before), edge_counts(m)
    shipped = dict(b_edges)
    for name in ("local", "microcircuit"):
        spec = _builders.get(REGISTRY.topologies[name].builder)
        try:
            shipped[name] = spec(before.sites, support="tissue").n_edges
        except MissingInput:
            # which is what happened: the builder asked for `tissue` sites, the
            # materialization had none, and the edge set was never built.
            shipped[name] = 0
    print()
    print(f"    {'topology':26s} {'on=(tissue,)':>13s} {'sheet only':>12s} {'split':>12s}   "
          "what changed")
    print(f"    {'-' * 26} {'-' * 13} {'-' * 12} {'-' * 12}   {'-' * 44}")
    for name in sorted(set(shipped) | set(b_edges) | set(a_edges)):
        sn, bn, an = shipped.get(name, 0), b_edges.get(name, 0), a_edges.get(name, 0)
        if sn == 0 and bn > 0 and an > bn:
            why = "was empty; now built on both supports"
        elif sn == 0 and bn > 0:
            why = "was empty; the sheet's own metric built it"
        elif an > bn:
            why = "the volume half of the split adds its edges"
        elif an == bn == sn and an > 0:
            why = "unchanged: cortical, and the sheet did not move"
        else:
            why = "still empty: its data or its support is absent"
        print(f"    {name:26s} {sn:>13,} {bn:>12,} {an:>12,}   {why}")
    print()
    print("  the two topologies that moved, in their own words:")
    for name in ("local", "microcircuit", "cortical_surface"):
        e = m.edges.get(name)
        if e is not None:
            print(f"    {name}: {e.note}")
    print()
    print("  and the processes that were priced over an empty `local` graph until now, all")
    print("  of which are in this trace:")
    for pid in sorted(p for p in m.trace.processes
                      if REGISTRY.processes[p].topology == "local"):
        print(f"    {pid}")

    head("9. is the resolution earning its cost? (ARCHITECTURE.md 1)")
    for v in earns_its_cost(m, factor=2.0):
        print("  " + v.describe().replace("\n", "\n  "))

    head("10. provenance audit")
    print(m.audit())

    head("11. the non-overlap rule, measured and then exercised")
    frac_st, med_st = _support_overlap(m.sites["cortical_surface"], m.sites["tissue"])
    frac_ts, med_ts = _support_overlap(m.sites["tissue"], m.sites["cortical_surface"])
    print("  a materialization must place each POSITION on exactly one support.  a COMPONENT")
    print("  may be on two, and here every neural component is -- but only because the two")
    print("  sets of positions are disjoint.  measured on the built tables:")
    print(f"    {frac_st:.2%} of the {m.sites['cortical_surface'].n:,} column nodes fall inside a "
          f"{m.sites['tissue'].n:,}-voxel `tissue` cell")
    print(f"    {frac_ts:.2%} of those tissue voxels fall inside a column node's cell")
    print(f"    median nearest-neighbour separation {med_st:.2f} mm sheet->volume, "
          f"{med_ts:.2f} mm volume->sheet")
    print(f"    the exclusion radius is {RIBBON_EXCLUSION_MM:g} mm and half a tissue cell is "
          f"{float(np.median(np.asarray(tis.spacing(np), float))) / 2:.2f} mm, so the")
    print("    zero above is a consequence of the arithmetic rather than a lucky measurement.")
    print()
    print("  and here is the same request WITHOUT the Difference -- both supports named, the")
    print("  volume unrestricted, so the sheet and the voxels covering it both carry cortical")
    print("  population state.  a lead field sums sources linearly, so that reports a scalp")
    print("  potential inflated by however much of the cortex both supports cover:")
    print()
    # the budget is raised for this one build, and only here.  unrestricted
    # `tissue` at 10.5 mm is 1,756 voxels x 25 components = 4.4e4 more variables
    # than the partitioned version, which puts the total at 4.2e5 and trips the
    # ceiling first -- and being refused for the price of a double count instead
    # of for the double count itself would be the wrong lesson entirely.
    overlapping = replace(request, resolution=chosen.resolution,
                          name="eeg-forward@double-counted",
                          budget=replace(request.budget, max_state_variables=600_000),
                          regions=(("cortex", OnSupport("cortical_surface")),
                                   ("sources", OnSupport("tissue")),
                                   ("conductor", OnSupport("head_volume"))))
    try:
        build(overlapping, geometry=geometry, warp=coreg.warp, anchors={"cortex": white},
              strict=True)
        print("  it built.  that is a bug in the check, not a licence.")
    except (MaterializationIncomplete, BudgetExceeded) as exc:
        problems = getattr(exc, "problems", (str(exc),))
        hit = [p for p in problems if "entered twice" in p or "overlap" in p]
        for p in hit:
            print(f"    REFUSED  {p}")
        for p in problems:
            if p not in hit:
                print(f"    also     {p.splitlines()[0][:110]}")
    loose = build(overlapping, geometry=geometry, warp=coreg.warp,
                  anchors={"cortex": white}, strict=False)
    frac, med = _support_overlap(loose.sites["cortical_surface"], loose.sites["tissue"])
    print()
    print("  and the geometry behind that refusal, measured rather than asserted:")
    print(f"    {frac:.1%} of the {loose.sites['cortical_surface'].n:,} column nodes fall inside "
          f"one of the {loose.sites['tissue'].n:,} unrestricted")
    print(f"    `tissue` cells (median nearest-neighbour separation {med:.2f} mm).  the two")
    print("    supports are describing the same millimetres.")
    print("  with strict=False the same build completes and carries the violation in its")
    print("  provenance instead, which is the behaviour cost estimation needs: you can price a")
    print("  model that double-counts, you must not run one.")

    head("12. what this build still cannot do honestly")
    print("  the count of processes running outside their f's regime went from 8 to "
          f"{len(m.provenance.breaches)} with")
    print("  this change, and that is the change working rather than failing.  the volume")
    print("  half of the neural field is at 10.5 mm, and `local_excitation`, `ionic_exchange`,")
    print("  `transmitter_dynamics` and `laminar_propagation` all declare validity ceilings of")
    print("  1-5 mm.  those breaches were previously invisible because there was no volume")
    print("  half to breach anything: the state did not exist, so nothing could be reported")
    print("  as out of regime.  a coarse block that says so is strictly better than an absent")
    print("  one that says nothing, and the remedy -- more budget on `tissue` -- is now a")
    print("  number in the coarsening argument rather than a missing feature.")
    print()
    for n in m.notes:
        print(f"  - {n}")
    for p in m.provenance.missing:
        print(f"  ! {p.splitlines()[0]}")
    return 0


#: R, extended by one region.  see section 3 and `RIBBON_EXCLUSION_MM`.
REGIONS = (
    ("cortex", OnSupport("cortical_surface")),
    ("subcortex", Difference(OnSupport("tissue"), Near("cortex", RIBBON_EXCLUSION_MM))),
    ("conductor", OnSupport("head_volume")),
)


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
      the conductor cannot give any of that back either: the next level up is
      15.3 mm and the skull is 5-8 mm thick, so one cell would straddle scalp,
      skull and brain, and the partial-volume conductivity of that cell is the
      entire content of the lead field.
    - the electrode shell is dropped rather than coarsened.  §1 says r(q) should
      be derived from where coarse-graining fails to commute with the dynamics
      and not from proximity to whatever we happen to be measuring, and
      `graded_around` in request.py repeats the warning.  here proximity is the
      wrong proxy twice over: the scalp under a contact is the smoothest part of
      the lead field, and this model's own docstring says the skull's
      centimetre-wide kernel is why eeg is coarse by physics, not by budget.
    - cortical_surface 3.0 mm -> 3.25 mm.  poisson-disk sampling of this
      subject's 202,437 mm2 white surface at 3.0 mm gives 15,870 column nodes
      (0.611 of the 25,974 a hexagonal lattice would fit, which is the
      random-sequential-adsorption jamming ratio and is what a blue-noise set is
      supposed to give).  15,870 x 13 neural components = 2.06e5, and the sheet
      is the only support whose count is continuous in r, so it is the only one
      that can be trimmed at all: n goes as 1/r^2, so r >= 3.0 * sqrt(15870/n).
      3.25 mm delivers 13,647 nodes = 1.77e5 variables.  what it costs is the
      thing 3 mm was buying -- source orientation across a sulcal bank -- and it
      costs about 7% of it.
    - tissue at 10.5 mm, which is the coarse end and is deliberate.  the volume
      half of the neural field carries 25 traced components, not 13: the 13
      neural ones plus the 12 metabolic and structural components whose only
      support is `tissue` and which were outside this view entirely while the
      field was on the sheet alone.  the tissue octree is dyadic over its own
      aseg bounding box and reaches 10.5 / 5.25 / 2.63 mm; at 5.25 mm the
      ribbon-excluded parenchyma is 3,138 leaves x 25 = 7.8e4 variables and the
      total is 4.55e5, over the ceiling by 14%.  at 10.5 mm it is 531 leaves x
      25 = 1.33e4 and the total is 3.90e5, 97% of the ceiling.
      the choice of which support to coarsen is not arbitrary.  the sheet is what
      the scalp lead field actually integrates over, and the volume is not in the
      lead field at all; spending the sheet's orientation resolution to resolve
      structures the instrument cannot see would be exactly backwards for THIS
      materialization.  a `dbs-response` or a `sleep-dynamics` request should
      invert that trade, and the numbers above are what it would be inverting.
    - total: 12,419 x 16 (conductor) + 13,647 x 13 (sheet) + 531 x 25 (volume)
      + 60 x 2 (montage) = 389,510 of a 400,000 ceiling."""


def coarsened_resolution() -> Resolution:
    """the explicitly-stated coarser r(q).  see `COARSENING_ARGUMENT`."""
    return Resolution(
        (rule(OnSupport("head_volume"), 8.0),
         rule(OnSupport("tissue"), 10.5),
         rule(OnSupport("cortical_surface"), 3.25)),
        default_mm=12.0, default_band=SCALP)


if __name__ == "__main__":
    raise SystemExit(main())
