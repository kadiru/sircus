"""Synthetic-only comparison preparation tests; no policy or simulator loads."""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from collections import Counter
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import location_comparison as comparison
import prepare_location_comparison as preparer


def fixture():
    """Invented coordinates and enclosures, not measured/native evidence."""
    train = [dict(id=f"train-{i:03d}", split="train", tile_index=i,
                  bucket_xy_m=[-.10 + (i % 7 + .5) * 1.5 / 7,
                               -.70 + (i // 7 + .5) * .9 / 5]) for i in range(35)]
    # Exactly these four placements intersect the synthetic base enclosures.
    for i, point in ((0, [.05, -.60]), (1, [.065, -.60]),
                     (5, [1.285, -.60]), (6, [1.30, -.60])):
        train[i]["bucket_xy_m"] = point
    placement = dict(table_bounds_xy_m=[[-.2, -.8], [1.5, .3]],
        bucket_radius_m=.03, clearance_m=.005,
        bases=[dict(id="left", xy_m=[.05, -.60], radius_m=.025),
               dict(id="right", xy_m=[1.30, -.60], radius_m=.025)],
        source_sha256={"synthetic_geometry": "a" * 64},
        method="Synthetic enclosing circles for mock tests only")
    history = [[.67, -.27], [.95, -.18]]
    return train, history, placement


def bind(train, history, placement):
    points = comparison.coordinates(history + [x["bucket_xy_m"] for x in train])
    return dict(parent_checkpoint=comparison.PARENT_SHA,
        original_split_file=comparison.SPLIT_SHA,
        continuous_geometry_layer=comparison.GEOMETRY_SHA,
        original_train=comparison.digest(train),
        historical_coordinates=comparison.digest([list(x) for x in points]),
        placement_geometry=comparison.digest(placement), synthetic_source="b" * 64)


def resign(value):
    return comparison.signed({k: v for k, v in value.items() if k != "fingerprint"})


class PlacementTests(unittest.TestCase):
    def test_known_synthetic_four_withheld_and_thirty_one_admitted(self):
        train, _, placement = fixture()
        withheld = [c["id"] for c in train
                    if not comparison.placement_check(c["bucket_xy_m"], placement)["admitted"]]
        self.assertEqual(withheld, ["train-000", "train-001", "train-005", "train-006"])
        self.assertEqual(35 - len(withheld), 31)

    def test_base_clearance_boundary_uses_whole_bucket_radius(self):
        _, _, placement = fixture()
        # left base x + base radius + bucket radius + required clearance
        threshold = .05 + .025 + .03 + .005
        inside = comparison.placement_check([threshold - 1e-9, -.60], placement)
        outside = comparison.placement_check([threshold + 1e-9, -.60], placement)
        self.assertEqual(inside["reasons"], ["base_enclosure_clearance:left"])
        self.assertTrue(outside["admitted"])
        self.assertAlmostEqual(outside["base_clearance_lower_bounds_m"]["left"], .005 + 1e-9)

    def test_bucket_table_footprint_and_original_goal_domain_are_separate(self):
        _, _, placement = fixture()
        placement["table_bounds_xy_m"][0][0] = -.05
        inside = comparison.placement_check([-.02 + 1e-9, -.3], placement)
        outside = comparison.placement_check([-.02 - 1e-9, -.3], placement)
        self.assertTrue(inside["admitted"])
        self.assertEqual(outside["reasons"], ["bucket_footprint_outside_table"])
        domain = comparison.placement_check([1.4, -.3], placement)
        self.assertEqual(domain["reasons"], ["outside_original_goal_domain"])

    def test_invalid_or_unbound_placement_fields_fail(self):
        _, _, original = fixture()
        for mutate in (lambda p: p.update(clearance_m=0),
                       lambda p: p.update(bucket_radius_m=float("nan")),
                       lambda p: p.update(source_sha256={"source": "not-a-sha"}),
                       lambda p: p["bases"][1].update(id="left"),
                       lambda p: p.update(extra="unrecognized")):
            with self.subTest(mutate=mutate):
                placement = copy.deepcopy(original)
                mutate(placement)
                with self.assertRaises(ValueError):
                    comparison.validate_placement(placement)

    def test_separation_index_checks_neighboring_cells_and_negative_coordinates(self):
        index = comparison._SeparationIndex([[-.001, -.001]])
        self.assertFalse(index.clear([.001, .001]))
        self.assertTrue(index.clear([.024000001, -.001]))
        self.assertFalse(index.clear([.023999999, -.001]))


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.history, cls.placement = fixture()
        cls.bindings = bind(cls.train, cls.history, cls.placement)
        cls.plan = comparison.build_comparison(cls.train, cls.history, cls.placement, cls.bindings)

    def test_equal_paid_requests_balanced_locations_and_exact_updates(self):
        a, b = (self.plan["schedules"][name] for name in ("original_usable", "broader"))
        self.assertEqual(len(a), 8960)
        self.assertEqual(len(b), 8960)
        counts = Counter(row["original_id"] for row in a)
        self.assertEqual(len(counts), 31)
        self.assertEqual(Counter(counts.values()), {289: 30, 290: 1})
        # Global valid-domain sampling, not an impossible quota in occupied tiles.
        tile_counts = Counter(row["tile_index"] for row in b)
        self.assertEqual(set(tile_counts), set(range(35)))
        self.assertNotEqual(set(tile_counts.values()), {256})
        for schedule in (a, b):
            self.assertEqual(Counter(row["update_index"] for row in schedule), {i: 140 for i in range(64)})
            self.assertEqual(Counter(row["batch_index"] for row in schedule), {i: 35 for i in range(256)})
            self.assertEqual([r["request_index"] for r in schedule], list(range(8960)))
            self.assertTrue(all(r["physical_slot"] == i % 35 for i, r in enumerate(schedule)))

    def test_monitor_and_final_assessment_budgets_match_cohorts(self):
        protocol = self.plan["protocol"]
        self.assertEqual(protocol["monitor_milestones"], [0, 2240, 4480, 6720, 8960])
        self.assertTrue(all(n % 140 == 0 for n in protocol["monitor_milestones"]))
        self.assertEqual(protocol["test_positions"], {"broad": 70, "local": 31})
        self.assertEqual(protocol["monitor_requests"], 315)
        self.assertEqual(protocol["final_assessment_requests"], 303)
        self.assertEqual(protocol["total_assessment_requests"], 618)
        self.assertEqual(protocol["proposed_total_train_requests"], 17920)

    def test_fresh_heldouts_and_train_are_disjoint_with_required_spacing(self):
        groups = self.plan["cohorts"]
        self.assertEqual({key: len(value) for key, value in groups.items()},
                         {"monitor": 35, "test_broad": 70, "test_local": 31})
        heldouts = [row["bucket_xy_m"] for group in groups.values() for row in group]
        historical = self.plan["historical_coordinates"]
        for i, point in enumerate(heldouts):
            self.assertTrue(comparison.placement_check(point, self.placement)["admitted"])
            self.assertTrue(all(math.dist(point, old) >= .025 for old in historical + heldouts[:i]))
        broad = [tuple(row["bucket_xy_m"]) for row in self.plan["schedules"]["broader"]]
        self.assertEqual(len(set(broad)), 8960)
        self.assertFalse(set(broad).intersection(map(tuple, historical)))
        for point in broad:
            self.assertTrue(comparison.placement_check(point, self.placement)["admitted"])
            self.assertTrue(all(math.dist(point, target) >= .025 for target in heldouts))

    def test_local_offsets_remain_fifty_mm_from_their_named_original_anchor(self):
        original = {c["id"]: c["bucket_xy_m"] for c in self.train}
        for row in self.plan["cohorts"]["test_local"]:
            self.assertAlmostEqual(math.dist(row["bucket_xy_m"], original[row["anchor_id"]]), .05)
            self.assertEqual(row["offset_m"], .05)

    def test_dense_old_history_records_missing_local_anchor_without_relaxing_freshness(self):
        history = list(self.history)
        anchor = self.train[10]
        x, y = anchor["bucket_xy_m"]
        history.extend([[x + .05 * math.cos(i * math.tau / 64),
                         y + .05 * math.sin(i * math.tau / 64)] for i in range(64)])
        bindings = bind(self.train, history, self.placement)
        with mock.patch.object(comparison, "MAX_DRAWS_PER_POINT", 1000):
            plan = comparison.build_comparison(self.train, history, self.placement, bindings)
        self.assertIn(anchor["id"], [row["anchor_id"] for row in plan["unavailable_local_anchors"]])
        self.assertNotIn(anchor["id"], [row["anchor_id"] for row in plan["cohorts"]["test_local"]])
        self.assertEqual(plan["protocol"]["final_assessment_requests"],
                         3 * (70 + len(plan["cohorts"]["test_local"])))

    def test_preparation_is_deterministic_and_does_not_consume_global_rng_or_mutate_inputs(self):
        inputs = copy.deepcopy((self.train, self.history, self.placement, self.bindings))
        before = random.getstate()
        result = comparison.build_comparison(*inputs)
        self.assertEqual(before, random.getstate())
        self.assertEqual(result, self.plan)
        self.assertEqual(inputs, (self.train, self.history, self.placement, self.bindings))
        self.assertEqual(len(set(result["seeds"].values())), len(result["seeds"]))

    def test_external_parent_split_checker_and_content_bindings_are_required(self):
        for key in ("parent_checkpoint", "original_split_file", "continuous_geometry_layer",
                    "original_train", "historical_coordinates", "placement_geometry"):
            with self.subTest(key=key):
                bindings = dict(self.bindings, **{key: "f" * 64})
                with self.assertRaises(ValueError):
                    comparison.build_comparison(self.train, self.history, self.placement, bindings)
        for bindings in ({}, dict(self.bindings, synthetic_source="not-a-sha")):
            with self.assertRaises(ValueError):
                comparison.build_comparison(self.train, self.history, self.placement, bindings)

    def test_mutated_source_inputs_fail_original_bindings(self):
        for item in (0, 1, 2):
            inputs = copy.deepcopy([self.train, self.history, self.placement])
            if item == 0:
                inputs[0][4]["bucket_xy_m"][0] += .001
            elif item == 1:
                inputs[1].append([.56, -.14])
            else:
                inputs[2]["bucket_radius_m"] += .001
            with self.assertRaises(ValueError):
                comparison.build_comparison(*inputs, self.bindings)

    def test_resealing_an_edited_plan_cannot_override_external_regeneration(self):
        for mutate in (lambda p: p["protocol"].update(execution_authorized=True),
                       lambda p: p["schedules"]["broader"][0]["bucket_xy_m"].__setitem__(0, .5),
                       lambda p: p["bindings"].update(synthetic_source="c" * 64)):
            changed = copy.deepcopy(self.plan)
            mutate(changed)
            with self.assertRaises(ValueError):
                comparison.validate_comparison(resign(changed), original_train=self.train,
                    history=self.history, placement=self.placement, bindings=self.bindings)

    def test_plan_is_not_launch_authorization_and_contains_no_measured_outcomes(self):
        protocol = self.plan["protocol"]
        self.assertFalse(protocol["execution_authorized"])
        self.assertFalse(protocol["release_authorized"])
        self.assertEqual(protocol["status"], "prepared_only_not_launchable")
        self.assertEqual(self.plan["native_requests"], 0)
        self.assertEqual(self.plan["optimizer_steps"], 0)
        self.assertFalse(self.plan["outcome_fields_used"])
        self.assertTrue(protocol["shared_learning"]["restore_weights_and_adam"])
        self.assertFalse(protocol["shared_learning"]["bank"])
        self.assertFalse(protocol["shared_learning"]["old_shot_replay"])
        self.assertIn("proposals", protocol["budget_status"])
        self.assertEqual(comparison.summarize(self.plan)["original_requests_per_location_range"], [289, 290])


class PreparationWriterTests(unittest.TestCase):
    def write_fixture(self, root):
        train, history, placement = fixture()
        split = root / "original.json"
        split.write_text(json.dumps({"train": train}, sort_keys=True))
        split_sha = hashlib.sha256(split.read_bytes()).hexdigest()
        history_path = root / "history.json"
        history_path.write_text(json.dumps(comparison.signed({"coordinates": history})))
        receipt = dict(schema="sircus-seed22-location-comparison-placement-source-v1",
            original_split_sha256=split_sha,
            base_mesh=dict(sha256="871d501d23e8164440656331ee0e005d859dbdb7fab1c3725bef423adac9cf0e",
                collider_type="convex_hull", collision_from_visuals=False,
                conservative_rounded_radius_m=.025),
            tabletop_xy_bounds_m=placement["table_bounds_xy_m"],
            bucket=dict(outer_footprint_radius_m=.03),
            base_poses=[dict(name=base["id"], position_m=base["xy_m"] + [0.]) for base in placement["bases"]],
            comparison_minimum_base_clearance_m=.005,
            source_files_sha256={"synthetic_geometry": "a" * 64}, asset_files_sha256={},
            train=[dict(id=c["id"], admitted=i not in (0, 1, 5, 6)) for i, c in enumerate(train)])
        receipt_path = root / "placement.json"
        receipt_path.write_text(json.dumps(comparison.signed(receipt)))
        return split, history_path, receipt_path, split_sha

    def test_writer_only_creates_new_package_and_preserves_all_inputs_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, history, receipt, split_sha = self.write_fixture(root)
            before = {path: path.read_bytes() for path in (split, history, receipt)}
            output = root / "new-preparation"
            with mock.patch.object(comparison, "SPLIT_SHA", split_sha), \
                    mock.patch.object(preparer, "PLACEMENT_SOURCE_FINGERPRINT",
                                      json.loads(receipt.read_text())["fingerprint"]):
                result = preparer.prepare_files(split, history, receipt, output)
                self.assertEqual(result["admitted_original_locations"], 31)
                self.assertEqual(result["broader_unique_locations"], 8960)
                created = {path.name: path.read_bytes() for path in output.iterdir()}
                self.assertEqual(set(created), {"comparison.json", "summary.json"})
                with self.assertRaises(ValueError):
                    preparer.prepare_files(split, history, receipt, output)
                self.assertEqual(created, {path.name: path.read_bytes() for path in output.iterdir()})
            self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_invalid_original_or_receipt_fails_before_creating_any_output(self):
        for defect in ("split", "fingerprint", "mesh", "classification", "resealed_radius"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                split, history, receipt, split_sha = self.write_fixture(root)
                pinned_fingerprint = json.loads(receipt.read_text())["fingerprint"]
                if defect == "split":
                    split.write_text(split.read_text() + " ")
                else:
                    changed = json.loads(receipt.read_text())
                    if defect == "fingerprint":
                        changed["bucket"]["outer_footprint_radius_m"] = .02
                    elif defect == "mesh":
                        changed["base_mesh"]["sha256"] = "d" * 64
                        changed = resign(changed)
                    elif defect == "classification":
                        changed["train"][0]["admitted"] = True
                        changed = resign(changed)
                    else:
                        changed["base_mesh"]["conservative_rounded_radius_m"] = .02
                        changed = resign(changed)
                    receipt.write_text(json.dumps(changed))
                output = root / "must-not-exist"
                with mock.patch.object(comparison, "SPLIT_SHA", split_sha), \
                        mock.patch.object(preparer, "PLACEMENT_SOURCE_FINGERPRINT", pinned_fingerprint):
                    with self.assertRaises(ValueError):
                        preparer.prepare_files(split, history, receipt, output)
                self.assertFalse(output.exists())

    def test_preparation_imports_only_stdlib_and_its_pure_contract(self):
        allowed = {"argparse", "copy", "hashlib", "json", "math", "random", "re",
                   "collections", "pathlib", "location_comparison"}
        for module in (comparison, preparer):
            tree = ast.parse(Path(module.__file__).read_text())
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add(node.module.split(".")[0])
            self.assertTrue(imports <= allowed, imports - allowed)


if __name__ == "__main__":
    unittest.main()
