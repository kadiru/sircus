"""Independent synthetic CPU fixtures; none is evidence of a native shot."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab import xy650_verify as audit
from sircus.mjlab.benchmark_verify import Checks, EvidenceError
from sircus.mjlab.tests.test_calibrated_score import config, resign


def row(ident="source", **changes):
    value = dict(id=ident, role="discovery", phase="coarse", values=[.47, -.27, .38, 28., .3868],
                 bucket_xy_m=[1.12, -.27], translation_m=[0., 0.], query_kind="original_reference",
                 source_record_id="historical", source_record_sha256="a"*64, query_fingerprint="b"*64,
                 q10=[0.]*6, q14=[0.]*6, native_initial_q10=[0.]*6, native_initial_q14=[0.]*6,
                 native_release=True, invalid=None, invalid_physics=False, route_success=False,
                 route_quality=0, first_flight=dict(available=True, first_crossing=None, mouth_clearance_m=None,
                                                 closest_opening_distance_m=.1))
    value.update(changes)
    return value


class XYAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        cls.scene = json.loads((root/"sircus/isaac/scene.json").read_text())
        cls.cfg = config(cls.scene)
        cls.xy = np.asarray(cls.scene["bucket"]["center_xy_m"])
        cls.rim = cls.scene["bucket"]["assumed_bottom_thickness_m"]+cls.scene["bucket"]["assumed_interior_depth_m"]

    def trace(self):
        points = np.tile([*self.xy, 0.], (7, 1))
        points[:, 2] = [.30, .27, .25, .24, .21, .17, .13]
        velocity = np.zeros_like(points);velocity[2, 2] = .1
        force = np.zeros((6, 6));force[0, 0] = 1.
        return dict(position_m=points, velocity_m_s=velocity, forces_N=force, dt_s=.01)

    def metric(self, arrays, **changes):
        return audit.first_flight(self.scene, self.xy, [0., 0., 1.], self.cfg, arrays, **changes)

    def test_literal253_queries_shift_board_and_bucket_together(self):
        from sircus.mjlab import xy650_contract as producer
        seed = row()
        targets = audit.initial_inputs(seed)
        self.assertEqual(len(targets), 253)
        self.assertEqual(len({tuple(t["values"]+t["bucket_xy_m"]) for t in targets}), 253)
        np.testing.assert_allclose(targets[0]["values"], [.46, -.23, .36, 26., .39], rtol=0, atol=1e-15)
        np.testing.assert_allclose(targets[242]["values"], [.48, -.19, .40, 30., .43], rtol=0, atol=1e-15)
        self.assertEqual(targets[243]["values"][2:], [.38, 28., .3868])
        self.assertEqual(targets[-1]["values"], seed["values"])
        for target, actual in zip(targets, producer.coarse(seed)):
            self.assertEqual(target["values"], actual["values"])
            self.assertEqual(target["bucket_xy_m"], actual["bucket_xy_m"])
            self.assertEqual(target["translation_xy_m"], actual["translation_m"])
            np.testing.assert_allclose(np.asarray(target["bucket_xy_m"])-target["values"][:2], [.65, 0.], rtol=0, atol=1e-15)

    def test_settings_freeze_budget_both_unique_stages_and_new_ranking(self):
        from sircus.mjlab import xy650_contract as producer
        audit.audit_settings(producer.settings(), Checks())
        for key in ("refinement_dedup_decimals", "max_requests", "coarse_requests", "first_flight_version"):
            settings = producer.settings();settings[key] = -1
            with self.subTest(key=key), self.assertRaises(EvidenceError):
                audit.audit_settings(settings, Checks())

    def test_initial_groups_retain_unshifted_controls_and_exact_source(self):
        from sircus.mjlab import xy650_contract as producer
        seed = row(role="candidate")
        controls = [row("positive", role="positive_control"), row("negative", role="negative_control")]
        groups = producer.controls(controls)+producer.coarse(seed)
        for group in groups:
            group.update(eligible=False, q10=[0.]*6, q14=[0.]*6)
        audit.audit_initial(groups, seed, controls, Checks())
        changed = copy.deepcopy(groups);changed[3]["bucket_xy_m"][0] += .001
        with self.assertRaisesRegex(EvidenceError, "bucket translated"):
            audit.audit_initial(changed, seed, controls, Checks())
        changed = copy.deepcopy(groups);changed[0]["q14"][0] = .001
        with self.assertRaisesRegex(EvidenceError, "control joint"):
            audit.audit_initial(changed, seed, controls, Checks())

    def test_first_flight_analytic_center_aperture_and_interval_cutoff(self):
        arrays = self.trace();before = copy.deepcopy(arrays)
        metric = self.metric(arrays)
        self.assertEqual((metric["start_index"], metric["end_index"], metric["sample_count"]), (2, 6, 5))
        self.assertEqual(metric["crossing_plane"], "ball_center_at_rim")
        self.assertEqual(metric["first_crossing"]["interval_index"], 4)
        self.assertAlmostEqual(metric["first_crossing"]["fraction"], (.21-self.rim)/.04)
        radius = self.scene["ball"]["radius_m"]
        expected = self.scene["bucket"]["inner_top_diameter_m"]/2*math.cos(math.pi/self.scene["bucket"]["segments"])-radius
        self.assertAlmostEqual(metric["mouth_clearance_m"], expected, places=12)
        self.assertAlmostEqual(metric["closest_opening_distance_m"], abs(.17-self.rim))
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], before[key])
        arrays["forces_N"][4, 2] = 1.
        arrays["position_m"][5:] = [*self.xy, self.rim]
        metric = self.metric(arrays)
        self.assertEqual(metric["cutoff_contact_channels"], ["table"])
        self.assertEqual(metric["end_index"], 4)
        self.assertIsNone(metric["first_crossing"])
        self.assertAlmostEqual(metric["closest_opening_distance_m"], .21-self.rim)

    def test_every_force_channel_cuts_before_deflection_exact_threshold_does_not(self):
        for column, name in enumerate(audit.FORCE_NAMES):
            arrays = self.trace();arrays["forces_N"][3, column] = .010001
            with self.subTest(channel=name):
                metric = self.metric(arrays)
                self.assertEqual(metric["end_index"], 3)
                self.assertEqual(metric["cutoff_contact_channels"], [name])
                self.assertIsNone(metric["first_crossing"])
                arrays["forces_N"][3, column] = .01
                self.assertEqual(self.metric(arrays)["end_index"], 6)

    def test_first_flight_matches_producer_without_importing_it_in_auditor(self):
        from sircus.mjlab import xy650_metrics as producer
        for column in range(6):
            arrays = self.trace();arrays["forces_N"][4, column] = 1.
            arrays["position_m"][:, 0] += .03
            with self.subTest(column=column):
                actual = producer.first_flight(self.scene, self.xy, [0, 0, 1], self.cfg, arrays)
                audit.same(actual, self.metric(arrays), Checks(), "independent diagnostic")
        cfg = resign(dict(self.cfg, compiled_rim_m=self.rim+4e-7))
        actual = producer.first_flight(self.scene, self.xy, [0, 0, 1], cfg, self.trace())
        expected = audit.first_flight(self.scene, self.xy, [0, 0, 1], cfg, self.trace())
        audit.same(actual, expected, Checks(), "compiled-rim diagnostic")

    def test_malformed_unavailable_and_contact_distance_semantics(self):
        self.assertEqual(self.metric(None)["reason"], "missing_trace")
        self.assertEqual(self.metric(None, valid=False)["reason"], "invalid_physics")
        arrays = self.trace();arrays["forces_N"][0, 0] = 0.
        self.assertEqual(self.metric(arrays)["reason"], "no_board_rebound")
        for key in ("position_m", "velocity_m_s", "forces_N"):
            arrays = self.trace();arrays[key].flat[0] = np.nan
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.metric(arrays)
        with self.assertRaises(ValueError):
            self.metric(self.trace(), valid=1)
        arrays = self.trace();arrays["contact_distance_m"] = np.full((6, 2), np.inf)
        arrays["contact_distance_m"][2, 1] = -.003
        self.assertEqual(self.metric(arrays)["end_index"], 6)
        arrays["contact_distance_m"][2, 1] = np.nan
        with self.assertRaises(ValueError):
            self.metric(arrays)

    def test_ranking_prefers_clean_capture_then_first_flight_not_table_roll(self):
        near = row("table-roll", closest_opening_distance_m=.000001)
        near["first_flight"]["closest_opening_distance_m"] = .2
        center = row("center")
        center["first_flight"].update(first_crossing=dict(time_s=.2), mouth_clearance_m=.02, closest_opening_distance_m=.03)
        rim = row("rim", route_success=True, route_quality=1)
        clean = row("clean", route_success=True, route_quality=2)
        invalid = row("invalid", route_success=True, route_quality=2, invalid_physics=True)
        self.assertEqual([r["id"] for r in audit.select([near, center, rim, clean, invalid], 10)], ["clean", "rim", "center", "table-roll"])
        from sircus.mjlab import xy650_contract as producer
        rows = [near, center, rim, clean, invalid]
        self.assertEqual([r["id"] for r in audit.select(rows, 10)], [r["id"] for r in producer.ranked(rows, 10)])

    def test_both_selection_stages_deduplicate_geometries_without_rounding_actions(self):
        source = row("a");duplicate = row("b");duplicate["values"][2] += 1e-12
        distinct = row("c");distinct["values"][2] += .001
        for count in (3, 5):
            selected = audit.select([source, duplicate, distinct], count, unique=True)
            self.assertEqual([r["id"] for r in selected], ["a", "c"])
        self.assertEqual(duplicate["values"][2], .38+1e-12)

    def test_refinement_keeps27_requests_per_unique_seed_and_source_query(self):
        from sircus.mjlab import xy650_contract as producer
        source = row();groups = producer.refinement([source])
        audit.audit_refinement(groups, [source], Checks())
        self.assertEqual(groups[0]["values"][2:], [.375, 27.5, .3818])
        self.assertEqual(groups[13]["values"], source["values"])
        groups[0]["source_query_fingerprint"] = "c"*64
        with self.assertRaisesRegex(EvidenceError, "query ancestry"):
            audit.audit_refinement(groups, [source], Checks())
        with self.assertRaisesRegex(EvidenceError, "distinct geometries"):
            audit.audit_refinement([], [source, copy.deepcopy(source)], Checks())

    def test_scene_query_and_execution_identities_have_separate_meanings(self):
        from sircus.mjlab import xy650_contract as producer
        group = row()
        scene_fp = audit.scene_fingerprint(self.scene, dict(score="fixed"), {"asset": "a"*64})
        self.assertEqual(scene_fp, producer.scene_fingerprint(self.scene, dict(score="fixed"), {"asset": "a"*64}))
        query_fp = audit.query_fingerprint(group, scene_fp)
        self.assertEqual(query_fp, audit.digest(producer.query_payload(group, scene_fp)))
        changed = copy.deepcopy(group);changed["bucket_xy_m"][0] += .001
        self.assertNotEqual(audit.query_fingerprint(changed, scene_fp), query_fp)
        changed = dict(group, source_record_id="another-observation")
        self.assertNotEqual(audit.query_fingerprint(changed, scene_fp), query_fp)
        group.update(query_fingerprint=query_fp, substeps=64, repeat_index=0)
        execution = audit.execution_fingerprint(group)
        self.assertEqual(execution, producer.execution_fingerprint(group))
        for field in ("native_initial_q10", "q14"):
            changed = copy.deepcopy(group);changed[field][0] = 1e-7
            self.assertNotEqual(audit.execution_fingerprint(changed), execution)
        self.assertNotEqual(audit.execution_fingerprint(dict(group, substeps=128)), execution)
        self.assertNotEqual(audit.execution_fingerprint(dict(group, repeat_index=1)), execution)

    def test_verification_preserves_selected_branch_and_query_ancestry(self):
        from sircus.mjlab import xy650_contract as producer
        source = row();groups = producer.verification([source])
        audit.audit_verification(groups, [source], Checks())
        groups[0]["q14"][0] = 1e-7
        with self.assertRaisesRegex(EvidenceError, "exact source field"):
            audit.audit_verification(groups, [source], Checks())

    def test_saved_native_column_is_the_only_source_for_diagnostic(self):
        arrays = self.trace()
        native = row(native_directory="native/64", native_cache_file="evidence.npz", native_cache_row=0,
                     bucket_xy_m=self.xy.tolist(), board_normal=[0., 0., 1.], first_flight=self.metric(arrays))
        rejected = row("reject", native_release=False, first_flight=None)
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory);cache = study/"native/64";cache.mkdir(parents=True)
            np.savez_compressed(cache/"evidence.npz", **{key: value[:, None] if isinstance(value, np.ndarray) else value
                                                     for key, value in arrays.items()})
            audit.audit_first_flight(study, [native, rejected], self.scene, self.cfg, Checks())
            changed = copy.deepcopy(native);changed["first_flight"]["closest_opening_distance_m"] = 0.
            with self.assertRaisesRegex(EvidenceError, "independent first flight"):
                audit.audit_first_flight(study, [changed], self.scene, self.cfg, Checks())
            changed = dict(rejected, first_flight=self.metric(None))
            with self.assertRaisesRegex(EvidenceError, "no observed first flight"):
                audit.audit_first_flight(study, [changed], self.scene, self.cfg, Checks())

    def test_fingerprint_audit_catches_actual_joint_and_bearing_tampering(self):
        spec = dict(scene=self.scene, calibration={"frozen": "calibration"}, asset_sha256={"collision": "a"*64})
        spec["scene_fingerprint"] = audit.scene_fingerprint(spec["scene"], spec["calibration"], spec["asset_sha256"])
        report = dict(scene_fingerprint=spec["scene_fingerprint"])
        seed = row();group = copy.deepcopy(seed)
        group["query_fingerprint"] = audit.query_fingerprint(group, spec["scene_fingerprint"])
        executed = dict(copy.deepcopy(group), substeps=64, repeat_index=0)
        executed["execution_fingerprint"] = audit.execution_fingerprint(executed)
        audit.audit_fingerprints(spec, report, [group], [executed], seed, Checks())
        changed = copy.deepcopy(executed);changed["native_initial_q10"][0] += .001
        with self.assertRaisesRegex(EvidenceError, "measured execution identity"):
            audit.audit_fingerprints(spec, report, [group], [changed], seed, Checks())
        changed = copy.deepcopy(group);changed["bucket_xy_m"] = [.47, .38]
        changed["query_fingerprint"] = audit.query_fingerprint(changed, spec["scene_fingerprint"])
        with self.assertRaisesRegex(EvidenceError, "fixed bearing"):
            audit.audit_fingerprints(spec, report, [changed], [], seed, Checks())

    def test_complete_plan_repeats_rejections_without_duplicate_request_ids(self):
        from sircus.mjlab import xy650_contract as producer
        groups = producer.bind([row("one", eligible=True, geometry_row=0), row("two", eligible=False, geometry_row=1)], "c"*64)
        actual = audit.make_plan(groups, "verification", steps=(64, 128), repeats=3)
        self.assertEqual(actual, producer.plan(groups, "verification", (64, 128), 3))
        self.assertEqual(len(actual), 12)
        self.assertEqual(sum(not r["eligible"] for r in actual), 6)
        self.assertEqual(len({r["id"] for r in actual}), 12)
        self.assertEqual(len({r["pair_id"] for r in actual}), 6)


if __name__ == "__main__":
    unittest.main()
