"""Cord boundary and anatomical identity regression tests (unittest, no fixtures)."""
import unittest

import numpy as np

from ibm.processes.cord import SegmentalCord


class CordContractTests(unittest.TestCase):
    def test_named_component_mapping_and_exclusions(self):
        ids = ["body-muscle-opensim-glmax2_l", "body-muscle-prior-body-bp3d-arm",
               "body-muscle-prior-body-bp3d-eye", "body-connective-body-bp3d-tendon",
               "body-muscle-opensim-addmagIsch_r", "body-muscle-prior-unknown"]
        bindings = [dict(muscle_id=ids[1], name="long head of right biceps brachii"),
                    dict(muscle_id=ids[2], name="left lateral rectus"),
                    dict(muscle_id=ids[3], name="right biceps brachii"),
                    dict(muscle_id=ids[4], name="right adductor magnus")]
        cord = SegmentalCord(ids, muscle_bindings=bindings)
        self.assertEqual(cord.mapping_keys[:3], ["gluteus_maximus", "biceps_brachii", "lateral_rectus"])
        self.assertEqual(cord.unmapped, ids[2:])
        np.testing.assert_allclose(cord.seg.sum(axis=1), [1, 1, 0, 0, 0, 0])

    def test_stretch_latency_and_cranial_silence(self):
        cord = SegmentalCord(["soleus", "genioglossus"], dt=.001)
        for step in range(31):
            result = cord.step(np.array([.2, .2]), stretch=np.ones(2), force=np.ones(2),
                               antagonist=np.array([1, 0]))
            if step < 30:
                self.assertEqual(float(result["stretch"][0]), 0.)
            else:
                self.assertGreater(float(result["stretch"][0]), 0.)
            for arc in ("stretch", "reciprocal", "autogenic", "renshaw"):
                self.assertEqual(float(result[arc][1]), 0.)

    def test_wrong_width_and_nonfinite_rejected_before_state_changes(self):
        cord = SegmentalCord(["soleus", "tibialis_anterior"])
        for name in ("descending", "stretch", "force"):
            for invalid in (0., [0.], [[0., 0.]], [np.nan, 0.], [0., np.inf]):
                args = dict(descending=np.zeros(2))
                args[name] = invalid
                with self.assertRaises(ValueError):
                    cord.step(**args)
                self.assertEqual(cord._delay_buf, {})
                np.testing.assert_array_equal(cord.gamma, 0.)
        for invalid in ([0], [0., 1.], [-2, 0], [0, 2]):
            with self.assertRaises(ValueError):
                cord.step(np.zeros(2), antagonist=invalid)

    def test_unpaired_antagonist_is_silent(self):
        cord = SegmentalCord(['soleus', 'tibialis_anterior'])
        for _ in range(40):
            result = cord.step(np.zeros(2), stretch=np.ones(2), antagonist=np.array([-1, 0]))
        self.assertEqual(result['reciprocal'][0], 0.)
        self.assertLess(result['reciprocal'][1], 0.)

    def test_native_catalog_id_and_source_name(self):
        cord = SegmentalCord(["native-opaque"], muscle_bindings=[
            {"id": "native-opaque", "name": "right biceps brachii"}])
        self.assertFalse(cord.unmapped)
        cord = SegmentalCord(["native-opaque"], muscle_bindings=[
            {"id": "native-opaque", "source_name": "glmax2_r"}])
        self.assertEqual(cord.mapping_keys, ["gluteus_maximus"])

    def test_constructor_contract(self):
        for dt in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                SegmentalCord(dt=dt)
        for ids in (["soleus", "soleus"], ["proprio:soleus"], [""]):
            with self.assertRaises(ValueError):
                SegmentalCord(ids)
        self.assertEqual(SegmentalCord([]).n, 0)


if __name__ == "__main__":
    unittest.main()
