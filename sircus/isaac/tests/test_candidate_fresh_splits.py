"""Synthetic geometry-only split tests; no model, solver or shot execution."""

from collections import Counter
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_fresh_splits as fresh
import shot_benchmark_contract as original


def resign(value):
    return fresh.signed({k:v for k,v in value.items() if k != "fingerprint"})


class FreshSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = original.make_splits(np.empty((0,2)))
        cls.old_before = copy.deepcopy(cls.old)
        cls.history = fresh.build_history_receipt(
            [cls.old["validation"][0]["bucket_xy_m"], [2.,2.], [.75,-.5]],
            {"synthetic-coordinate-file.json":"a"*64}, scope="Synthetic coordinate manifests only")
        cls.manifest = fresh.make_fresh_splits(cls.history,cls.old)

    def test_fixed_35_70_no_train_tiles_and_old_contract_unchanged(self):
        value = self.manifest
        self.assertEqual(value["seed"],20261126)
        self.assertNotIn("train",value)
        self.assertEqual([len(value[r]) for r in ("validation","test")],[35,70])
        self.assertEqual(Counter(c["tile_index"] for c in value["validation"]),{i:1 for i in range(35)})
        self.assertEqual(Counter(c["tile_index"] for c in value["test"]),{i:2 for i in range(35)})
        self.assertEqual([c["tile_index"] for c in value["test"][:35]],list(range(35)))
        self.assertEqual([c["tile_index"] for c in value["test"][35:]],list(range(35)))
        self.assertEqual(self.old,self.old_before)
        self.assertEqual(original.validate_splits(self.old),self.old["fingerprint"])

    def test_determinism_and_local_rng_do_not_change_global_state(self):
        before = np.random.get_state()
        again = fresh.make_fresh_splits(self.history,self.old)
        self.assertEqual(again,self.manifest)
        after = np.random.get_state()
        self.assertEqual(before[0],after[0]); np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])

    def test_history_receipt_normalizes_duplicates_and_binds_coordinate_sources(self):
        points = [[2.,2.],[0.,0.],[2.,2.]]
        receipt = fresh.build_history_receipt(points,{"synthetic":"b"*64},scope="Coordinates only")
        self.assertEqual(receipt["coordinates"],[[0.,0.],[2.,2.]])
        self.assertEqual(receipt["coordinate_count"],2)
        self.assertEqual(receipt,fresh.build_history_receipt(list(reversed(points)),{"synthetic":"b"*64},scope="Coordinates only"))
        points[0][0] = 8
        self.assertEqual(receipt["coordinates"],[[0.,0.],[2.,2.]])
        changed = copy.deepcopy(receipt); changed["source_sha256"]["synthetic"] = "c"*64
        with self.assertRaisesRegex(ValueError,"fingerprint"): fresh.validate_history(changed)
        for bad in ([[True,0]],[[float("nan"),0]],[[0,0,0]]):
            with self.assertRaises(ValueError): fresh.build_history_receipt(bad,{"synthetic":"b"*64},scope="Coordinates only")
        for sources,scope in (({},"x"),({"x":"no"},"x"),({"x":"b"*64},"")):
            with self.assertRaises(ValueError): fresh.build_history_receipt([] ,sources,scope=scope)

    def test_all_140_original_positions_are_always_in_exclusion_union(self):
        audit = fresh.audit_fresh_splits(self.manifest,self.history,self.old)
        self.assertEqual(audit["status"],"passed")
        self.assertTrue(audit["exact_draw_replay"])
        self.assertEqual(audit["exclusion_counts"],dict(discovered_history=3,original_benchmark=140,
                                                     excluded_union=142,shared_history_original=1))
        points = np.asarray([c["bucket_xy_m"] for c in fresh.cells_from_splits(self.manifest)])
        excluded = np.asarray([c["bucket_xy_m"] for role in ("train","validation","test") for c in self.old[role]])
        self.assertGreaterEqual(np.linalg.norm(points[:,None]-excluded[None,:],axis=2).min(),.025)
        self.assertGreaterEqual(audit["minimum_exclusion_distance_m"],.025)
        self.assertGreaterEqual(audit["minimum_new_pair_distance_m"],.025)
        self.assertEqual(audit["new_train_positions"],0)

    def test_empty_discovered_history_still_excludes_all_original_positions(self):
        history = fresh.build_history_receipt([],{"synthetic":"d"*64},scope="Empty discovered pool")
        value = fresh.make_fresh_splits(history,self.old)
        self.assertEqual(value["exclusion_counts"]["excluded_union"],140)
        self.assertEqual(fresh.validate_fresh_splits(value,history,self.old),value["fingerprint"])

    def test_changed_source_or_old_manifest_fails_even_with_new_fingerprints(self):
        history = copy.deepcopy(self.history)
        history["source_sha256"]["synthetic-coordinate-file.json"] = "e"*64
        history = resign(history)
        with self.assertRaisesRegex(ValueError,"exclusion pool"): fresh.validate_fresh_splits(self.manifest,history,self.old)
        other = original.make_splits([[2.,2.]])
        with self.assertRaisesRegex(ValueError,"exclusion pool"): fresh.validate_fresh_splits(self.manifest,self.history,other)

    def test_outcome_fields_wrong_roles_seed_spacing_and_draw_count_fail_closed(self):
        mutations = [lambda v:v.update(seed=20261127),lambda v:v.update(minimum_separation_m=.024),
            lambda v:v.update(optimizer_steps=1),lambda v:v.update(native_requests=1),
            lambda v:v.update(outcome_files_read=True),lambda v:v.update(train=[]),
            lambda v:v.update(draws=v["draws"]+1),lambda v:v["test"][0].update(capture=True),
            lambda v:v["validation"][0].update(id="train-000"),lambda v:v["test"].pop(),
            lambda v:v.update(release_authorized=True)]
        for mutation in mutations:
            changed = copy.deepcopy(self.manifest); mutation(changed); changed = resign(changed)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                fresh.validate_fresh_splits(changed,self.history,self.old)

    def test_standalone_cells_are_detached_but_do_not_certify_history_exclusion(self):
        cells = fresh.cells_from_splits(self.manifest)
        cells[0]["bucket_xy_m"][0] = 99
        self.assertNotEqual(self.manifest["validation"][0]["bucket_xy_m"][0],99)
        changed = copy.deepcopy(self.manifest)
        # The old TRAIN coordinate belongs to tile0; standalone validation can
        # check geometry shape, while the full audit additionally binds history.
        changed["validation"][0]["bucket_xy_m"] = self.old["train"][0]["bucket_xy_m"]
        changed = resign(changed)
        with self.assertRaises(ValueError): fresh.validate_fresh_splits(changed,self.history,self.old)

    def test_pair_overlap_or_tile_changes_fail_even_if_resigned(self):
        for mutation in (lambda v:v["test"][0].update(bucket_xy_m=v["validation"][0]["bucket_xy_m"]),
                         lambda v:v["test"][0].update(grid_column=1),
                         lambda v:v["test"][0].update(bucket_xy_m=[1.4,.2])):
            changed = copy.deepcopy(self.manifest); mutation(changed)
            with self.assertRaises(ValueError): fresh.cells_from_splits(resign(changed))

    def test_seeded_draw_replay_rejects_geometry_valid_manual_shift(self):
        changed = copy.deepcopy(self.manifest)
        changed["validation"][0]["bucket_xy_m"][0] += .000001
        changed = resign(changed)
        self.assertEqual(len(fresh.cells_from_splits(changed)),105)
        with self.assertRaisesRegex(ValueError,"draw sequence"):
            fresh.audit_fresh_splits(changed,self.history,self.old)
        report = fresh.audit_fresh_splits(changed,self.history,self.old,replay=False)
        self.assertFalse(report["exact_draw_replay"])

    def test_packing_failure_has_no_spacing_relaxation_or_tile_replacement(self):
        # A 10mm grid covers tile0 with exclusion disks. This synthetic history
        # blocks the first tile entirely; the producer must stop at its cap.
        lower = np.asarray(fresh.BOUNDS[0]); size = (np.asarray(fresh.BOUNDS[1])-lower)/[7,5]
        points = [[float(x),float(y)] for x in np.arange(lower[0]-.01,lower[0]+size[0]+.02,.01)
                  for y in np.arange(lower[1]-.01,lower[1]+size[1]+.02,.01)]
        history = fresh.build_history_receipt(points,{"synthetic-dense-grid":"f"*64},scope="Synthetic packing failure")
        with self.assertRaisesRegex(ValueError,"no constraint was relaxed"):
            fresh.make_fresh_splits(history,self.old,max_draws=105)
        for cap in (104,1_000_001,True):
            with self.assertRaises(ValueError): fresh.make_fresh_splits(self.history,self.old,max_draws=cap)

    def test_no_model_solver_outcomes_or_execution_in_preparation_surface(self):
        self.assertEqual((self.manifest["optimizer_steps"],self.manifest["native_requests"],self.manifest["outcome_files_read"]),(0,0,0))
        self.assertFalse(self.manifest["execution_authorized"])
        self.assertFalse(self.manifest["release_authorized"])
        receipt = copy.deepcopy(self.history); receipt["captures"] = 3
        with self.assertRaises(ValueError): fresh.validate_history(resign(receipt))
        for key in ("outcome_files_read", "optimizer_steps", "native_requests"):
            receipt = copy.deepcopy(self.history); receipt[key] = False
            with self.assertRaises(ValueError): fresh.validate_history(resign(receipt))


if __name__ == "__main__":
    unittest.main()
