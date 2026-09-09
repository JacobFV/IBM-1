"""the cortical sheet is a surface with an atlas, and it has to stay one.

`docs/DISCONNECTS.md` rows 2 and 3.  Four things can go wrong silently here and
each has a shape this repo's corrections ledger already knows.

*the sheet reverts to a sphere.*  `cortical_sites` used to place sites on a
spherical shell, and everything downstream still runs on it -- `dyn.pos` is a
saved buffer, so an old checkpoint restores its own sphere and must keep meaning
what it meant.  What must NOT happen is a surface run silently getting spherical
coordinates or a spherical run being read through an atlas.  `is_spherical_proxy`
is the discriminator and it is tested in both directions.

*the insula stops being separable.*  That is the single concrete thing the change
had to buy: on a sphere there is no lateral sulcus, so `insula` was not an
addressable target and the interoceptive port entered a subsample of `frontal`
instead.  The test asserts the port is the insula and the anterior cingulate, and
that asking for them on the sphere RAISES rather than substituting -- a silent
substitution is how the disconnect survived in the first place.

*the graph stops being reproducible, or stops being saved.*  CLAUDE.md: "the
graph is part of the trained object; `dyn.idx`/`geo`/`pos` must be saved and
restored, never redrawn", and `read_idx` -- an attribute that was not saved --
cost a factor of 112.  The tract wiring adds `delay_s`, which is exactly such an
object, so a round trip through a state dict is tested.

*the tract arm stops being matched to its control.*  A comparison between two
wirings is only about the wiring if the edge count and the row L1 agree.  If they
drift the arms differ in gain, and docs/LOG.md's ledger is 25 rows of a quantity
computed correctly and compared against the wrong thing.
"""
from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SP = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(ROOT, "scripts/pretrain_video_loop.py"))
P = importlib.util.module_from_spec(_SP)
_SP.loader.exec_module(P)

try:
    import ibm.cortical_sheet as CS
    HAVE_SURFACE = os.path.exists(f"{CS.FSAVERAGE}/surf/lh.white")
except Exception:                                              # pragma: no cover
    CS, HAVE_SURFACE = None, False

try:
    import ibm.cortical_tracts as CT
    HAVE_TRACTS = os.path.isdir(CT.RAW) or os.path.exists(CT.CACHE)
except Exception:                                              # pragma: no cover
    CT, HAVE_TRACTS = None, False

SKIP = "payload not staged; run scripts/fetch_cortical_atlases.py"


@unittest.skipUnless(HAVE_SURFACE, SKIP)
class TestSheetIsASurface(unittest.TestCase):

    def test_lobes_partition_the_atlas(self):
        # every DK gyrus in exactly one lobe.  a lobe that is a subset with an
        # unstated remainder makes "the drive enters frontal" a claim about
        # something smaller than frontal.
        from ibm.anatomy.systems import DK_GYRI
        seen = [g for v in CS.LOBE_MEMBERS.values() for g in v]
        self.assertEqual(sorted(seen), sorted(DK_GYRI))
        self.assertEqual(len(set(seen)), len(seen))

    def test_sites_are_on_the_surface_and_reproducible(self):
        a = P.cortical_sites(3000, "cpu", seed=0, geometry="surface")
        b = P.cortical_sites(3000, "cpu", seed=0, geometry="surface")
        self.assertTrue(torch.equal(a, b))
        c = P.cortical_sites(3000, "cpu", seed=1, geometry="surface")
        self.assertFalse(torch.equal(a, c))
        # no two sites at the same point: sampling is WITHOUT replacement, and a
        # duplicated vertex is not a finer sheet.
        self.assertEqual(len({tuple(r) for r in a.numpy().tolist()}), 3000)

    def test_the_discriminator_works_in_both_directions(self):
        s = P.cortical_sites(2000, "cpu", geometry="sphere")
        f = P.cortical_sites(2000, "cpu", geometry="surface")
        self.assertTrue(P.is_spherical_proxy(s))
        self.assertFalse(P.is_spherical_proxy(f))

    def test_insula_and_cingulate_are_separable(self):
        """THE test for disconnect 2.  it was not possible to write this before."""
        pos = P.cortical_sites(4000, "cpu", geometry="surface")
        ins = P.region_index(pos, "insula")
        cing = P.region_index(pos, "cingulate")
        self.assertGreater(len(ins), 50)
        self.assertGreater(len(cing), 50)
        # disjoint from each other and from the lobes they used to be buried in
        for other in ("frontal", "temporal", "parietal"):
            o = P.region_index(pos, other)
            self.assertEqual(len(np.intersect1d(ins.numpy(), o.numpy())), 0)
            self.assertEqual(len(np.intersect1d(cing.numpy(), o.numpy())), 0)
        # and a hemisphere can be named
        self.assertLess(len(P.region_index(pos, "lh.insula")), len(ins))

    def test_the_sphere_refuses_rather_than_substituting(self):
        pos = P.cortical_sites(2000, "cpu", geometry="sphere")
        for name in ("insula", "cingulate", "rostralanteriorcingulate"):
            with self.assertRaises(KeyError):
                P.region_index(pos, name)
        # the six it DOES have still work, and still mean what they meant
        self.assertGreater(len(P.region_index(pos, "occipital")), 0)

    def test_interoceptive_port_is_the_insula_now(self):
        import ibm.interoception as IO
        dyn = P.CorticalDynamics(4000, 8, 6, "cpu", long_range=0.0,
                                 geometry="surface")
        loop = P.InteroceptiveLoop(dyn, [p.key for p in IO.PORTS], n_out=2)
        self.assertEqual(tuple(loop.lobe), IO.PORT_REGIONS)
        self.assertEqual(loop.port_frac, 1.0)          # the whole label, not 4.5%
        want = P.region_index(dyn.pos, IO.PORT_REGIONS)
        self.assertEqual(len(loop.port), len(want))

    def test_lobe_area_fractions_are_plausible(self):
        # a case whose answer is roughly known: occipital cortex is about a tenth
        # of the sheet, and the spherical proxy's thresholds were set to imitate
        # exactly these numbers.  this is the measurement they were guessing at.
        f = CS.area_fractions()["lobe"]
        self.assertAlmostEqual(sum(f.values()), 1.0, places=6)
        self.assertTrue(0.05 < f["occipital"] < 0.16, f["occipital"])
        self.assertTrue(0.02 < f["insula"] < 0.05, f["insula"])


@unittest.skipUnless(HAVE_SURFACE and HAVE_TRACTS, SKIP)
class TestTractWiring(unittest.TestCase):

    N, K = 3000, 24

    def _arm(self, **kw):
        return P.CorticalDynamics(self.N, 8, self.K, "cpu", long_range=0.25,
                                  geometry="surface", graph_seed=0, **kw)

    def test_arms_are_matched_on_edge_count_and_row_l1(self):
        r = self._arm(long_topology="random")
        t = self._arm(long_topology="tract")
        self.assertEqual(r.idx.shape, t.idx.shape)
        self.assertEqual(r.n_far, t.n_far)
        # row L1 of the HELD prior: this is what makes the comparison about
        # destinations rather than about gain.
        self.assertLess(float((r.geo.sum(1) - t.geo.sum(1)).abs().max()), 1e-9)

    def test_tract_partners_land_where_the_connectome_says(self):
        t = self._arm(long_topology="tract", tract_threshold=0.5)
        A = CT.consensus(0.5)["adjacency"]
        reg = P.cortical_regions(t.pos).numpy()
        n_loc = self.K - t.n_far
        far = t.idx[:, n_loc:].numpy()
        src = np.repeat(reg[:, None], t.n_far, 1).reshape(-1)
        dst = reg[far].reshape(-1)
        # every long-range edge is a declared parcel pair.  (a site in a parcel
        # with no declared partner falls back to its own parcel and is counted in
        # `tract_note`; allow exactly that.)
        bad = ~(A[src, dst] | (src == dst))
        self.assertEqual(int(bad.sum()), 0)

    def test_random_partners_are_not_constrained(self):
        # the control must actually be a control: if random happened to satisfy
        # the tract constraint the comparison would be measuring nothing.
        r = self._arm(long_topology="random")
        A = CT.consensus(0.5)["adjacency"]
        reg = P.cortical_regions(r.pos).numpy()
        n_loc = self.K - r.n_far
        far = r.idx[:, n_loc:].numpy()
        src = np.repeat(reg[:, None], r.n_far, 1).reshape(-1)
        dst = reg[far].reshape(-1)
        self.assertGreater(float((~(A[src, dst] | (src == dst))).mean()), 0.5)

    def test_delays_survive_a_state_dict_round_trip(self):
        # CLAUDE.md's read_idx entry: anything that selects or times an edge is
        # part of the artifact and must be saved.
        t = self._arm(long_topology="tract", tract_delays=True)
        self.assertIn("delay_s", t.state_dict())
        back = P.dynamics_from_state_dict(
            {f"dyn.{k}": v for k, v in t.state_dict().items()}, "cpu")
        for name in ("pos", "idx", "geo", "delay_s"):
            self.assertTrue(torch.equal(getattr(t, name), getattr(back, name)), name)

    def test_delays_change_the_trajectory_and_only_at_a_dt_that_resolves_them(self):
        t = self._arm(long_topology="tract", tract_delays=True)
        u = self._arm(long_topology="tract", tract_delays=False)
        with torch.no_grad():
            u.embed.copy_(t.embed)
        drive = torch.zeros(1, self.N)
        drive[:, :64] = 5.0

        def run(dyn, dt):
            s, w = dyn.init_state(1, "cpu"), dyn.edge_weights()
            for _ in range(12):
                s = dyn.step(s, drive, dt, w)
            return s[1]

        # at 1 ms the declared delays (median ~3.8 ms) are several steps
        self.assertGreater(float((run(t, 1e-3) - run(u, 1e-3)).abs().max()), 0.0)
        # and the delayed run carries the ring buffer in its state
        s = t.init_state(1, "cpu")
        s = t.step(s, drive, 1e-3, t.edge_weights())
        self.assertEqual(len(s), 6)
        self.assertEqual(len(u.step(u.init_state(1, "cpu"), drive, 1e-3,
                                    u.edge_weights())), 4)

    def test_the_declared_builder_actually_runs_and_agrees(self):
        """DISCONNECTS row 3 is that nothing imports `ibm.topologies.tract`.

        The training loop uses `draw_partners`, a uniform subsample of the
        builder's edge set, because the builder's full expansion is quadratic in
        parcel population -- 607k edges from 1,500 sites here, 10^10 at the
        resolution the sheet runs at.  That is a good reason not to call it in a
        training loop and a bad reason never to call it at all, so it is called
        here, and the subsample is checked to be a SUBSET of what it emits.  If
        the two ever disagree, one of them is wired to a different connectome.
        """
        from ibm.topologies.tract import tractometric_matrix
        n = 1200
        sites, kw = CT.builder_inputs(n_sites=n, seed=0)
        edges = tractometric_matrix(sites, **kw)
        self.assertGreater(len(edges.src), 1000)
        for f in ("tract_length_mm", "conduction_delay_s", "distance_mm"):
            self.assertIn(f, edges.features)
        declared = set(zip(edges.src.tolist(), edges.dst.tolist()))

        region = np.asarray(sites["tissue"].partitions["cortical_areas"])
        n_far = 6
        part, dly, ln, note = CT.draw_partners(region, n_far, seed=0)
        src = np.repeat(np.arange(n)[:, None], n_far, 1)
        same_parcel = region[part] == region[src]
        bad = [(int(a), int(b)) for a, b in zip(src[~same_parcel],
                                                part[~same_parcel])
               if (a, b) not in declared]
        self.assertEqual(bad[:5], [],
                         f"{len(bad)} drawn partners are not edges of the "
                         "declared topology")

        # the ONLY drawn edges that are not builder edges are the same-parcel
        # ones, and those are exactly the orphan fallback: a parcel the
        # connectome leaves with no partner spends its budget internally, and the
        # builder emits no self-parcel edge for it to match.  the two counts must
        # agree, or something else is producing same-parcel draws.
        self.assertAlmostEqual(float(same_parcel.mean()),
                               note["orphan_fraction"], places=6)
        self.assertGreater(note["orphan_fraction"], 0.0)
        self.assertTrue(all(r.startswith(("lh.", "rh."))
                            for r in note["orphan_parcels"]))
        # and the delay a drawn edge carries must be the delay the builder gives
        feat = {(int(a), int(b)): float(d) for a, b, d in
                zip(edges.src, edges.dst, edges.features["conduction_delay_s"])}
        checked = 0
        for i in range(0, n, 37):
            for c, j in enumerate(part[i]):
                if int(j) == i or (i, int(j)) not in feat:
                    continue
                self.assertAlmostEqual(feat[(i, int(j))], float(dly[i, c]),
                                       places=9)
                checked += 1
        self.assertGreater(checked, 20)

    def test_consensus_threshold_is_monotone(self):
        # the card singles out the parameterisable threshold as this source's
        # one honest property; a sweep that is not monotone in edge count would
        # mean the sweep is not doing what it says.
        curve = CT.edge_existence_curve((0.25, 0.5, 0.9))
        self.assertGreater(curve[0]["n_edges"], curve[1]["n_edges"])
        self.assertGreater(curve[1]["n_edges"], curve[2]["n_edges"])


if __name__ == "__main__":
    unittest.main()
