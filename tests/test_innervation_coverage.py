"""The innervation tables must agree with each other and with the body.

Every check here is a cross-table one.  A table can be internally tidy and still
be about a different object than the table beside it, and that is the failure
this repo's ledger is made of -- a quantity computed correctly and compared
against the wrong thing.  So: trunk roots against muscle roots, trunk names
against the body's routes, dermatome roots against trunk roots, and the route
length against the one the body measured.
"""
import os
import unittest

from ibm.anatomy.muscles import INNERVATION
from ibm.topologies import dermatome as D
from ibm.topologies import ihm_bridge as BR
from ibm.topologies.nerve import (BRANCH_TRUNK, CRANIAL, FIBRE_VELOCITY_M_S,
                                  TRUNK_COMPOSITION, TRUNK_LENGTH_MM,
                                  TRUNK_ROOTS, TRUNK_TARGET,
                                  fibre_delays_s, trunk_endpoints, trunk_length_mm,
                                  trunk_of)

IHM = os.path.exists(BR.IHM_PERIPHERAL)
DERM = os.path.exists(D.IHM_DERMATOMES)


class TrunkEndpoints(unittest.TestCase):
    def test_every_trunk_has_two_ends(self):
        """A trunk with a length and no ends is a routing convention, not a nerve."""
        for t in TRUNK_COMPOSITION:
            with self.subTest(trunk=t):
                self.assertTrue(TRUNK_ROOTS.get(t), f"{t} has no proximal end")
                self.assertTrue(TRUNK_TARGET.get(t), f"{t} has no distal end")
        self.assertEqual(set(TRUNK_ROOTS) - set(TRUNK_COMPOSITION), set())
        self.assertEqual(set(TRUNK_TARGET) - set(TRUNK_COMPOSITION), set())

    def test_root_kinds_do_not_mix_cranial_and_spinal_except_the_accessory(self):
        """CN XI is the one nerve with both, and it must be the only one."""
        mixed = [t for t, r in TRUNK_ROOTS.items()
                 if any(x in CRANIAL for x in r) and any(x not in CRANIAL for x in r)]
        self.assertEqual(mixed, ["accessory"])

    def test_every_muscle_nerve_resolves_to_a_declared_trunk(self):
        for m, (nerve, _, _) in INNERVATION.items():
            with self.subTest(muscle=m):
                self.assertIn(trunk_of(nerve), TRUNK_COMPOSITION,
                              f"{m} names {nerve!r}, which resolves to no trunk")
        self.assertEqual(set(BRANCH_TRUNK.values()) - set(TRUNK_COMPOSITION), set())

    def test_muscle_roots_are_contained_in_their_trunk_roots(self):
        """A muscle cannot draw a segment its own nerve does not carry.

        This is the check that caught eleven real contradictions when the trunk
        endpoint table was first written -- erector spinae hanging off the
        thoracoabdominal (ventral) nerves when it is supplied by dorsal rami,
        psoas major on the femoral nerve when it is supplied by the L1-L3 rami
        directly, and nine looser root lists on both sides.
        """
        for m, (nerve, roots, _) in sorted(INNERVATION.items()):
            declared = TRUNK_ROOTS[trunk_of(nerve)]
            for r in roots:
                with self.subTest(muscle=m, root=r):
                    self.assertIn(r, declared,
                                  f"{m} claims {r} but {trunk_of(nerve)} carries "
                                  f"{declared}")

    def test_a_purely_cutaneous_trunk_carries_no_motor_fibre(self):
        """The composition table's absences are claims and must stay true."""
        for t in ("sural", "saphenous", "superficial_radial",
                  "lateral_femoral_cutaneous", "supraclavicular"):
            with self.subTest(trunk=t):
                self.assertNotIn("alpha", TRUNK_COMPOSITION[t])
                self.assertNotIn("ia", TRUNK_COMPOSITION[t])
                self.assertEqual(
                    [m for m, (n, _, _) in INNERVATION.items() if trunk_of(n) == t],
                    [], f"{t} carries no alpha fibre and cannot supply a muscle")

    def test_endpoints_record_that_the_typed_length_is_superseded(self):
        e = trunk_endpoints("vagus")
        self.assertEqual(e["length_superseded_by"], "ibm.topologies.ihm_bridge.routes")
        self.assertIn("not conduction route", e["length_evidence"])
        self.assertEqual(e["root_kind"], "cranial_nucleus")
        self.assertEqual(e["roots"], ("x",))
        self.assertTrue(e["target"])


class MeasuredLengths(unittest.TestCase):
    @unittest.skipUnless(IHM, "IHM peripheral.json absent")
    def test_no_route_falls_back_to_the_typed_trunk_table(self):
        """The regression that cost the vagus 158 ms, guarded on ALL routes.

        `visceral_routes` guarded the 16 visceral ones because that is where it
        was caught; the visceral routes were not special, they were merely the
        ones with no muscle binding to mask the fall-through.
        """
        n = BR.assert_measured_lengths()
        self.assertEqual(n, len(list(BR.routes())))
        self.assertGreater(n, 0)

    @unittest.skipUnless(IHM, "IHM peripheral.json absent")
    def test_delays_read_the_measured_route_not_the_typed_length(self):
        length, src = trunk_length_mm("vagus")
        self.assertEqual(src, "ihm_measured_route")
        typed = TRUNK_LENGTH_MM["vagus"]
        self.assertNotAlmostEqual(length, typed, places=3)
        d = fibre_delays_s("vagus")
        self.assertAlmostEqual(d["c"], (length * 1e-3) / FIBRE_VELOCITY_M_S["c"][1])

    @unittest.skipUnless(IHM, "IHM peripheral.json absent")
    def test_a_trunk_with_no_typed_length_never_gets_a_silent_default(self):
        """21 of 72 trunks have no typed entry, so this is not a corner case."""
        untyped = sorted(set(TRUNK_COMPOSITION) - set(TRUNK_LENGTH_MM))
        self.assertTrue(untyped)
        for t in untyped:
            with self.subTest(trunk=t):
                _, src = trunk_length_mm(t)
                self.assertEqual(src, "ihm_measured_route")
                self.assertEqual(trunk_length_mm(t, prefer_measured=False)[1],
                                 "default_no_declared_length")

    @unittest.skipUnless(IHM, "IHM peripheral.json absent")
    def test_every_declared_trunk_has_a_route_in_the_body(self):
        c = BR.coverage()
        self.assertEqual(c["ibm_only"], [],
                         "a trunk declared here with no route in the body")
        self.assertEqual(c["ihm_only"], [],
                         "a route in the body with no fibre classes here")


class SkinPatches(unittest.TestCase):
    @unittest.skipUnless(DERM, "IHM dermatomes.json absent")
    def test_every_patch_routes_through_a_declared_trunk(self):
        ps = list(D.patches())
        self.assertTrue(ps)
        self.assertEqual([p["id"] for p in ps if p["undeclared_trunk"]], [])

    @unittest.skipUnless(DERM, "IHM dermatomes.json absent")
    def test_a_skin_patch_carries_no_spindle_or_tendon_organ_fibre(self):
        """Skin has A-beta, A-delta and C.  It has no Ia and no Ib."""
        for p in D.patches():
            with self.subTest(patch=p["id"]):
                self.assertTrue(p["classes"])
                self.assertNotIn("ia", p["classes"])
                self.assertNotIn("ib", p["classes"])
                self.assertNotIn("alpha", p["classes"])

    @unittest.skipUnless(DERM, "IHM dermatomes.json absent")
    def test_patch_root_level_is_carried_by_its_own_trunk(self):
        """A patch's dermatome must be a segment its nerve actually carries."""
        for p in D.patches():
            if p["root_level"] is None:
                continue
            with self.subTest(patch=p["id"]):
                self.assertIn(p["root_level"], TRUNK_ROOTS[p["nerve_name"]],
                              f"{p['nerve_name']} does not carry "
                              f"{p['root_level']}")

    @unittest.skipUnless(DERM, "IHM dermatomes.json absent")
    def test_delay_varies_with_the_patch_and_not_only_with_the_trunk(self):
        """Two patches on one trunk at different distances must differ.

        Reading the trunk length for a patch would give every median-territory
        patch one latency, which is the lumping this whole module exists to
        avoid, one level down.
        """
        by_trunk = {}
        for p in D.patches():
            by_trunk.setdefault(p["nerve_name"], []).append(p["delays_s"]["abeta"])
        spread = {k: max(v) - min(v) for k, v in by_trunk.items() if len(v) > 4}
        self.assertTrue(spread)
        self.assertTrue(all(s > 0 for s in spread.values()),
                        f"a trunk whose patches all share one delay: "
                        f"{[k for k, s in spread.items() if s == 0]}")

    @unittest.skipUnless(DERM, "IHM dermatomes.json absent")
    def test_the_face_is_reported_apart_from_the_dermatomes(self):
        c = D.coverage()
        self.assertGreater(c["patches_without_spinal_root"], 0)
        self.assertEqual(c["patches"],
                         c["patches_with_spinal_root"] +
                         c["patches_without_spinal_root"])
        self.assertFalse(c["measured_dermatome_atlas"])

    @unittest.skipUnless(DERM, "IHM dermatomes.json absent")
    def test_coverage_states_its_denominator_and_uses_the_exterior_area(self):
        c = D.coverage()
        self.assertGreater(c["skin_raw_mesh_area_m2"], c["skin_exterior_area_m2"])
        self.assertAlmostEqual(c["area_fraction_of_exterior"], 1.0, places=6)
        self.assertIn("exterior", c["area_denominator"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
