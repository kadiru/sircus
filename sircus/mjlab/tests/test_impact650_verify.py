"""Synthetic CPU controls for the independent impact audit; never native evidence."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import impact650_verify as audit
from sircus.mjlab.benchmark_verify import Checks, EvidenceError, sha
from sircus.mjlab.tests.test_calibrated_score import config
from sircus.mjlab.tests.test_impact650_contract import sources
from sircus.mjlab.tests.test_tilt650_verify import fresh_rows


ROOT = Path(__file__).resolve().parents[3]
SCENE = json.loads((ROOT/"sircus/isaac/scene.json").read_text())


def geometry_fields(group):
    result = copy.deepcopy(group)
    g = audit.nominal_geometry(SCENE, group["bucket_xy_m"], group["values"], group["impact_offset_m"])
    for key in ("impact_point_world_m", "release_center_m", "release_center_height_m", "impact_to_bucket_horizontal_m"):
        result[key] = g[key]
    return result


class ImpactAuditTests(unittest.TestCase):
    def test_surface_basis_center_plane_vertical_gap_and_zero_history(self):
        from sircus.mjlab.learning_ik import shot_targets
        values, bucket = [.47, -.21, .38, 30., .41], [-.03, -.63]
        zero = audit.nominal_geometry(SCENE, bucket, values, [0., 0.])
        self.assertEqual(zero["release_center_m"], [values[0], values[1], values[2]+(SCENE["ball"]["radius_m"]+values[4])])
        old = shot_targets(SCENE, np.asarray([bucket]), np.asarray([values[:3]]), [30.], [.41])
        np.testing.assert_allclose(audit.endpoint_targets(SCENE, bucket, values, [0., 0.])["dropper_flange_world"], old["dropper_flange_world"][0], rtol=0, atol=1e-15)
        offset = audit.nominal_geometry(SCENE, bucket, values, [.04, .02])
        self.assertAlmostEqual(offset["impact_point_world_m"][2], .36)
        self.assertAlmostEqual(offset["release_center_height_m"], .36+.41+.0375)
        self.assertNotAlmostEqual(offset["impact_to_bucket_horizontal_m"], zero["impact_to_bucket_horizontal_m"])
        normal, forward, lateral = (np.asarray(offset[key]) for key in ("board_normal_world", "forward_world", "lateral_world"))
        np.testing.assert_allclose(np.cross(normal, forward), lateral, atol=1e-15, rtol=0)
        self.assertAlmostEqual(float(normal@forward), 0.)

    def test_independent_geometry_matches_actual_producer_targets_without_importing_it(self):
        from sircus.mjlab.impact650_geometry import impact_targets
        for source in sources():
            for offset in ([0., 0.], [-.04, .02], [.04, -.02]):
                expected = audit.endpoint_targets(SCENE, source["bucket_xy_m"], source["values"], offset)
                actual = impact_targets(SCENE, np.asarray([source["bucket_xy_m"]]), np.asarray([source["values"]]), np.asarray([offset]))
                for key in expected:
                    np.testing.assert_allclose(expected[key], actual[key][0], rtol=0, atol=1e-15)

    def test_literal135_requests_freeze_each_source_board_and_zero_offset_dropper(self):
        from sircus.mjlab import impact650_contract as producer
        from sircus.mjlab.tests.test_tilt650_verify import row
        selected = sources()
        controls = producer.controls([row("positive-control", role="positive_control"), row("negative-control", role="negative_control")])
        groups = controls+producer.coarse(selected)
        audit.audit_initial(groups, selected, controls, Checks())
        self.assertEqual(len(groups), 137)
        self.assertEqual(len({audit.target_key(group) for group in groups[2:]}), 135)
        changed = copy.deepcopy(groups);changed[2+22]["q10"][0] += .001
        with self.assertRaisesRegex(EvidenceError, "zero-offset dropper"):
            audit.audit_initial(changed, selected, controls, Checks())
        changed = copy.deepcopy(groups);changed[3]["q14"][0] += .001
        with self.assertRaisesRegex(EvidenceError, "fixed source board"):
            audit.audit_initial(changed, selected, controls, Checks())

    def test_selection_preserves_family_quota_then_global_rank_and_offset_uniqueness(self):
        from sircus.mjlab import impact650_contract as producer
        candidates = producer.coarse(sources())
        rows = []
        for i in (0, 1, 2, 45, 90):
            group = candidates[i]
            rows.append(dict(group, id=f"observed-{i:03}", native_release=True, invalid=None, invalid_physics=False,
                             route_success=i<90, route_quality=2 if i<45 else 1 if i<90 else 0, first_flight=None))
        actual = audit.selected_verification(rows)
        self.assertEqual([row["family_id"] for row in actual[:3]], list(audit.FAMILIES))
        self.assertEqual([row["id"] for row in actual], [row["id"] for row in producer.select(rows)])
        self.assertEqual(len(actual), 5)
        self.assertEqual(len({audit.target_key(row) for row in actual}), 5)
        duplicate = copy.deepcopy(rows[0]);duplicate["id"] = "duplicate";duplicate["impact_offset_m"][0] += 1e-12
        self.assertEqual(len(audit.selected_verification([rows[0], duplicate])), 1)
        self.assertEqual(len(audit.selected_verification(rows[:2])), 2)

    def test_new_action_and_actual_release_are_bound_without_relabeling_physical_scene(self):
        from sircus.mjlab import impact650_contract as producer
        group = producer.coarse(sources())[0]
        fp = audit.scene_fingerprint("a"*64)
        self.assertEqual(fp, producer.scene_fingerprint(dict(scene_fingerprint="a"*64)))
        self.assertNotEqual(fp, "a"*64)
        group = producer.bind([group], fp)[0]
        self.assertEqual(audit.query_fingerprint(group, fp), group["query_fingerprint"])
        changed = copy.deepcopy(group);changed["impact_offset_m"][0] += .001
        self.assertNotEqual(audit.query_fingerprint(changed, fp), group["query_fingerprint"])
        group.update(substeps=64, repeat_index=0, native_initial_q10=[0.]*6, native_initial_q14=[0.]*6, native_initial_ball_center_m=[.4, -.2, .8])
        original = audit.execution_fingerprint(group)
        self.assertEqual(original, producer.execution_fingerprint(group))
        changed = copy.deepcopy(group);changed["native_initial_ball_center_m"][0] += .001
        self.assertNotEqual(audit.execution_fingerprint(changed), original)

    def test_verified_source_offset_and_joints_cannot_change_across_steps(self):
        from sircus.mjlab import impact650_contract as producer
        rows = producer.bind(producer.coarse(sources())[:5], "a"*64)
        for i, row in enumerate(rows):row.update(route_quality=i%3)
        groups = producer.verification(rows)
        audit.audit_verification(groups, rows, Checks())
        changed = copy.deepcopy(groups);changed[0]["impact_offset_m"][0] += .001
        with self.assertRaisesRegex(EvidenceError, "frozen target"):
            audit.audit_verification(changed, rows, Checks())
        changed = copy.deepcopy(groups);changed[0]["q14"][0] += .001
        with self.assertRaisesRegex(EvidenceError, "frozen target"):
            audit.audit_verification(changed, rows, Checks())

    def test_inclusive189_plan_keeps_rejections_and_geometry_fields(self):
        from sircus.mjlab import impact650_contract as producer
        from sircus.mjlab.tests.test_tilt650_verify import row
        controls = producer.controls([row("positive-control", role="positive_control"), row("negative-control", role="negative_control")])
        groups = producer.bind([geometry_fields(group) for group in controls+producer.coarse(sources())], "b"*64)
        for i, group in enumerate(groups):group.update(geometry_row=i, eligible=i != 3)
        verified = [dict(groups[i], id=f"impact-verify-{i}", role="verification") for i in range(2, 7)]
        plans = [audit.make_plan(g, phase, steps=steps, repeats=repeats) for g, phase, steps, repeats in
                 ((groups[:2], "controls_pre", (64, 128), 3), (groups[2:], "coarse", (64,), 1),
                  (verified, "verification", (64, 128), 3), (groups[:2], "controls_post", (64, 128), 3))]
        self.assertEqual([len(rows) for rows in plans], [12, 135, 30, 12])
        self.assertEqual(len(sum(plans, [])), 189)
        self.assertEqual(len({row["id"] for row in sum(plans, [])}), 189)
        self.assertEqual(plans[1], producer.plan(groups[2:], "coarse"))

    def test_source_capture_is_never_invented_and_clean_is_a_fresh_tier(self):
        rows = fresh_rows(2);positions = {row["id"]: [0., 0., .3] for row in rows}
        self.assertTrue(audit.qualification(rows, positions, controls_passed=True, source_capture=True)["clean_qualified"])
        blocked = audit.qualification(rows, positions, controls_passed=True, source_capture=False)
        self.assertFalse(blocked["capture_qualified"] or blocked["clean_qualified"])
        self.assertFalse(blocked["source_capture"])
        self.assertNotIn("parent_source_capture", blocked)

    def test_settings_preserve_request_and_hour_caps(self):
        from sircus.mjlab import impact650_contract as producer
        audit.audit_settings(producer.settings(), Checks())
        self.assertEqual(audit.VERSION, producer.AUDIT_VERSION)
        for key in ("max_requests", "user_request_cap", "worker_wall_limit_s", "forward_m"):
            changed = producer.settings();changed[key] = 201
            with self.subTest(key=key), self.assertRaises(EvidenceError):audit.audit_settings(changed, Checks())

    def test_custom_dense_checker_rejects_center_only_ball_reset(self):
        from sircus.mjlab.calibrated_score_oracle import score_trace
        cfg = config(SCENE);values = [.47, -.21, .38, 30., .41];bucket = [-.03, -.63];offset = [.04, .01]
        geometry = audit.nominal_geometry(SCENE, bucket, values, offset)
        points = np.tile(geometry["release_center_m"], (7, 1));normal = geometry["board_normal_world"]
        arrays = dict(position_m=points[:, None], velocity_m_s=np.zeros((7, 1, 3)), forces_N=np.zeros((6, 1, 6)),
                      time_s=np.arange(7)*.01, native_time_s=(np.arange(7)*.01)[:, None],
                      contact_distance_m=np.full((6, 1, 2), np.inf), contact_geom_ids=np.full((6, 1, 2, 2), -1, dtype=int),
                      contact_position_m=np.zeros((6, 1, 2, 3)), contact_normal=np.zeros((6, 1, 2, 3)), record_ids=np.asarray(["shifted"]))
        trace = {key: value[:, 0] if value.ndim >= 2 else value for key, value in arrays.items() if key != "record_ids"}
        metrics = score_trace(SCENE, bucket, normal, cfg, trace)
        row = dict(metrics, id="shifted", native_release=True, native_cache_file="native.npz", native_cache_row=0,
                   values=values, bucket_xy_m=bucket, impact_offset_m=offset, board_center_m=values[:3], board_normal=normal,
                   release_ball_center_m=geometry["release_center_m"], native_initial_ball_center_m=geometry["release_center_m"],
                   native_initial_bucket_center_m=[*bucket, 0.], native_clock_drift_s=0., bucket_collider_error_m=0.,
                   pose_gate=dict(board_position_error_mm=0., board_normal_error_deg=0., dropper_error_mm=0.))
        spec = dict(version="sircus-synthetic-interval-trace-fixture-v2", native_dt_s=.01, horizon_s=.06)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path = root/"native.npz"
            np.savez_compressed(path, **arrays);row["native_cache_sha256"] = sha(path)
            audit.rescore_impact_rows(root, [row], SCENE, cfg, Checks(), "synthetic shifted reset", spec)
            center = audit.nominal_geometry(SCENE, bucket, values, [0., 0.])["release_center_m"]
            arrays["position_m"][:] = center
            row["native_initial_ball_center_m"] = center
            np.savez_compressed(path, **arrays);row["native_cache_sha256"] = sha(path)
            with self.assertRaisesRegex(EvidenceError, "dense time-zero release pose"):
                audit.rescore_impact_rows(root, [row], SCENE, cfg, Checks(), "synthetic wrong center reset", spec)


if __name__ == "__main__":unittest.main()
