"""Interoceptive join regression tests (unittest, no fixtures).

Three things that can go wrong silently here, all of which have equivalents in
this repo's corrections ledger, and one that already did.

*a route length that falls back to the typed trunk table.*  IHM's route contract
says `missing_length_policy: error; never silently substitute a trunk length`,
and `ihm_bridge.routes()` read the length from `muscle_bindings` and
`receptor_patches` only.  Every visceral route has neither, so all of them fell
through and reported a typed number for a measured one -- vagus 350 mm against
508, which is 158 ms on the C fibre.  That is what `test_visceral_routes_are_*`
exists to keep fixed.

*a fibre class that is not on the trunk, or is efferent.*  The velocity table has
an entry for every class regardless of which trunk carries it, so a wrong class
still produces a plausible delay.  And the autonomic trunks are mostly EFFERENT
classes -- a loop that encoded onto `b_preganglionic` would be sending a stimulus
out along the outflow.

*a channel set that has drifted between the two repositories.*  A channel renamed
in IHM leaves a port here wired to a rate nobody sends, and the encoder still
runs.
"""
import unittest

from ibm.topologies.nerve import FIBRE_VELOCITY_M_S, TRUNK_COMPOSITION
import ibm.interoception as IO


class InteroceptiveJoinTests(unittest.TestCase):

    def test_join_validates_against_registry(self):
        r = IO.check()
        self.assertEqual(r["ports"], len(IO.PORTS))
        self.assertEqual(r["trunks"], len(IO.TRUNKS))

    def test_every_port_uses_an_afferent_class_its_trunk_carries(self):
        for p in IO.PORTS:
            self.assertIn(p.fibre, TRUNK_COMPOSITION[p.trunk],
                          f"{p.channel}: {p.trunk} does not carry {p.fibre}")
            self.assertIn(p.fibre, IO.afferent_classes(p.trunk),
                          f"{p.channel}: {p.fibre} is not afferent")
            self.assertNotIn(p.fibre, IO.EFFERENT_CLASSES)

    def test_autonomic_trunks_have_exactly_one_afferent_class(self):
        # AUTONOMIC = (b_preganglionic, c_postganglionic, c).  Two of the three
        # are efferent, so a splanchnic afferent is a C fibre and nothing else.
        # That is not a simplification of the model, it is what the trunk
        # composition says, and a splanchnic port on any other class is a bug.
        for t in ("greater_splanchnic", "lesser_splanchnic", "least_splanchnic",
                  "lumbar_splanchnic", "pelvic_splanchnic"):
            self.assertEqual(IO.afferent_classes(t), ("c",))

    def test_vagus_afferent_classes_exclude_the_outflow(self):
        self.assertEqual(IO.afferent_classes("vagus"), ("abeta", "adelta", "c"))
        # and the practical reason it matters beyond correctness: `c` and
        # `c_postganglionic` share a velocity, so a loop carrying both would find
        # two classes on the same arrival step and could not tell physiology from
        # a collapsed timestep.
        self.assertEqual(FIBRE_VELOCITY_M_S["c"][1],
                         FIBRE_VELOCITY_M_S["c_postganglionic"][1])

    def test_visceral_routes_come_from_ihm_and_never_from_the_trunk_table(self):
        try:
            from ibm.topologies.ihm_bridge import visceral_routes
            vr = visceral_routes()
        except (KeyError, ValueError) as e:      # raises rather than substituting
            self.skipTest(f"IHM peripheral.json unavailable or inconsistent: {e}")
        if not vr:
            self.skipTest("IHM peripheral.json not present")
        for name, r in vr.items():
            self.assertEqual(r["length_source"], "ihm_nerve_route", name)

    def test_the_vagal_delay_spread_is_the_physiology_and_not_a_rounding_budget(self):
        try:
            d = IO.group_delays_s()
        except KeyError:
            self.skipTest("IHM peripheral.json not present")
        fast, slow = d[("vagus", "abeta")], d[("vagus", "c")]
        # one nerve, one length, two orders of magnitude apart in velocity.
        self.assertLess(fast, 0.02)
        self.assertGreater(slow, 0.4)
        self.assertGreater(slow / fast, 40.0)
        # and the splanchnic report of the same stomach lands between them.
        self.assertLess(fast, d[("greater_splanchnic", "c")])
        self.assertLess(d[("greater_splanchnic", "c")], slow)

    def test_every_conduction_group_resolves_at_the_loop_default_dt(self):
        # the check `InteroceptiveLoop` makes at construction, made here too so a
        # change to a route length or a velocity fails a test rather than a run.
        try:
            d = IO.group_delays_s()
        except KeyError:
            self.skipTest("IHM peripheral.json not present")
        steps = {g: round(v / 1e-2) for g, v in d.items()}
        self.assertEqual(len(set(steps.values())), len(steps),
                         f"groups collapse onto shared arrival steps: {steps}")

    def test_channel_set_matches_the_corpus_when_one_is_present(self):
        meta = IO.load_corpus_meta()
        if meta is None:
            self.skipTest("no intero corpus built")
        IO.check(meta)                     # raises on any drift
        self.assertEqual(sorted(p.key for p in IO.PORTS),
                         sorted(meta["channels"]))

    def test_the_port_substitution_is_declared_and_not_silent(self):
        # the insula is not separable on the spherical proxy.  the point of the
        # constant is that it is printed rather than assumed, so the test is that
        # it exists and says which region was substituted.
        self.assertIn("insula", IO.PORT_SUBSTITUTION)
        self.assertIn(IO.PORT_LOBE, IO.PORT_SUBSTITUTION)
        self.assertTrue(IO.ONTOLOGY_GAPS)


class VisceralPortTests(unittest.TestCase):

    def test_embodiment_manifest_carries_the_visceral_group(self):
        from ibm.embodiment import manifest
        try:
            m = manifest()
        except KeyError:
            self.skipTest("IHM peripheral.json not present")
        self.assertIn("visceral_in", m)
        self.assertEqual(len(m["visceral_in"]), len(IO.PORTS))
        # every visceral port is an input, and no two share a delay unless they
        # share a conduction group.
        for p in m["visceral_in"]:
            self.assertEqual(p.direction, "from_body")
            self.assertGreater(p.delay_s, 0.0)

    def test_a_channel_the_body_does_not_send_raises(self):
        # the 32-vs-64 truncation: an encoder sized for more ports than the body
        # supplies ran and validated with the afference silently halved.
        import torch
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "ptrain", os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "scripts", "pretrain_video_loop.py"))
        P = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(P)
        try:
            IO.group_delays_s()
        except KeyError:
            self.skipTest("IHM peripheral.json not present")
        torch.manual_seed(0)
        dyn = P.CorticalDynamics(600, 8, 6, "cpu")
        keys = [p.key for p in IO.PORTS]
        with self.assertRaises(ValueError):
            P.InteroceptiveLoop(dyn, keys[:-1], n_out=2)      # a port with no rate
        with self.assertRaises(ValueError):
            P.InteroceptiveLoop(dyn, keys + ["vagus/not_a_channel"], n_out=2)


if __name__ == "__main__":
    unittest.main()
