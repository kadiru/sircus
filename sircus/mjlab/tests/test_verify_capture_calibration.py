"""CPU-only guard tests for independent native calibration evidence auditing."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sircus.mjlab import verify_capture_calibration as audit
from sircus.mjlab.tests.test_calibrated_score import config as synthetic_config

ROOT = Path(__file__).resolve().parents[3]
SCENE = json.loads((ROOT / "sircus/isaac/scene.json").read_text())


def manifest(role="fit", reverse=False):
    cases = audit.expected_cases(SCENE, role)
    if reverse:
        cases.reverse()
    return dict(version=audit.PRODUCER_VERSION, role=role, substeps=64,
                fit_rule=audit.FIT_RULE, duration_s=4, input_speed_limit_m_s=6.,
                verified_speed_limit_m_s=6.25, cases=cases,
                fingerprint=audit.digest(cases), fixture_order="reversed" if reverse else "canonical")


def stream():
    case = audit.expected_cases(SCENE, "fit")[0]
    ticks, dt = 4 * 240 * 32, 1 / (240 * 32)
    p = np.tile(np.asarray(case["position_local_m"], np.float32), (ticks + 1, 1, 1))
    return [case], dict(position_m=p, velocity_m_s=np.zeros_like(p),
        forces_N=np.zeros((ticks, 1, 6), np.float32),
        native_time_s=(np.arange(ticks + 1) * dt).astype(np.float32)[:, None],
        contact_distance_m=np.full((ticks, 1, 2), np.inf, np.float32),
        contact_geom_ids=np.full((ticks, 1, 2, 2), -1, np.int32),
        contact_position_m=np.zeros((ticks, 1, 2, 3), np.float32),
        contact_normal=np.zeros((ticks, 1, 2, 3), np.float32),
        time_s=np.arange(ticks + 1) * dt, dt_s=np.asarray(dt),
        record_ids=np.asarray([case["id"]]))


class CalibrationAuditGuards(unittest.TestCase):
    def test_prespecified_grids_and_order(self):
        self.assertEqual(len(audit.expected_cases(SCENE, "fit")), 57)
        self.assertEqual(len(audit.expected_cases(SCENE, "verification")), 72)
        for role in ("fit", "verification"):
            for reverse in (False, True):
                value = manifest(role, reverse)
                cases, substeps, order = audit.check_fixture_manifest(value, SCENE, role, audit.Checks())
                self.assertEqual(order, "reversed" if reverse else "canonical")
                self.assertEqual(substeps, 64)
                self.assertEqual(cases, value["cases"])

    def test_resigned_fixture_change_is_still_rejected(self):
        value = manifest()
        value["cases"][0]["position_local_m"][2] += .001
        value["fingerprint"] = audit.digest(value["cases"])
        with self.assertRaisesRegex(audit.CalibrationAuditError, "initial condition"):
            audit.check_fixture_manifest(value, SCENE, "fit", audit.Checks())

    def test_duplicate_fixture_rejected(self):
        value = manifest()
        value["cases"][1] = copy.deepcopy(value["cases"][0])
        value["fingerprint"] = audit.digest(value["cases"])
        with self.assertRaisesRegex(audit.CalibrationAuditError, "repeated"):
            audit.check_fixture_manifest(value, SCENE, "fit", audit.Checks())

    def test_order_declaration_checked(self):
        value = manifest(reverse=True)
        value["fixture_order"] = "canonical"
        with self.assertRaisesRegex(audit.CalibrationAuditError, "order"):
            audit.check_fixture_manifest(value, SCENE, "fit", audit.Checks())

    def test_dense_native_schema_passes(self):
        cases, arrays = stream()
        audit.validate_arrays(arrays, cases, 32, audit.Checks())

    def test_initial_velocity_requires_exact_float32_quantization(self):
        cases, arrays = stream()
        cases[0]["velocity_m_s"][0] = 4.487807055054106
        arrays["velocity_m_s"][0, 0] = np.asarray(cases[0]["velocity_m_s"], np.float32)
        audit.validate_arrays(arrays, cases, 32, audit.Checks())
        arrays["velocity_m_s"][0, 0, 0] = np.nextafter(arrays["velocity_m_s"][0, 0, 0], np.float32(np.inf))
        with self.assertRaisesRegex(audit.CalibrationAuditError, "velocity"):
            audit.validate_arrays(arrays, cases, 32, audit.Checks())

    def test_dense_time_is_exact_tick_convention(self):
        cases, arrays = stream()
        arrays["time_s"][100] += 1e-10
        with self.assertRaisesRegex(audit.CalibrationAuditError, "tick-derived"):
            audit.validate_arrays(arrays, cases, 32, audit.Checks())

    def test_dense_clock_is_checked_separately(self):
        cases, arrays = stream()
        arrays["native_time_s"][100] = arrays["native_time_s"][99]
        with self.assertRaisesRegex(audit.CalibrationAuditError, "clock"):
            audit.validate_arrays(arrays, cases, 32, audit.Checks())

    def test_dense_wrong_force_category_rejected(self):
        cases, arrays = stream()
        arrays["forces_N"][1, 0, 0] = .01
        with self.assertRaisesRegex(audit.CalibrationAuditError, "force categories"):
            audit.validate_arrays(arrays, cases, 32, audit.Checks())

    def test_fit_rule_uses_both_contact_and_center_overlap(self):
        rows = [dict(contact_wall_overlap_m=.001, cavity_wall_overlap_m=.002,
                     contact_floor_overlap_m=.003, cavity_floor_overlap_m=.0001),
                dict(contact_wall_overlap_m=.0025, cavity_wall_overlap_m=.0001,
                     contact_floor_overlap_m=.001, cavity_floor_overlap_m=.002)]
        self.assertEqual(audit.fit_bounds(rows), (.0025, .003, 1.25*.0025+8e-6, 1.25*.003+8e-6))

    def test_zero_contact_frames_require_precision_bounded_tangency(self):
        arrays = dict(contact_distance_m=np.array([[[-2e-8, 0]]], np.float32),
                      contact_geom_ids=np.array([[[[65, 0], [65, 64]]]], np.int32),
                      contact_position_m=np.array([[[[0, 0, .0125], [0, 0, .0125]]]], np.float32),
                      contact_normal=np.array([[[[0, 0, 0], [0, 0, 1]]]], np.float32),
                      forces_N=np.zeros((1, 1, 6), np.float32),
                      position_m=np.array([[[0, 0, .05]], [[0, 0, .05]]], np.float32))
        geometry = dict(wall_ids=[0], base_id=64, ball_id=65, radius=.0375)
        audit.validate_contact_observations(dict(arrays=arrays), geometry, audit.Checks())
        arrays["contact_distance_m"][0, 0, 0] = -1e-4
        with self.assertRaisesRegex(audit.CalibrationAuditError, "frame"):
            audit.validate_contact_observations(dict(arrays=arrays), geometry, audit.Checks())

    def test_verification_never_refits_envelope(self):
        cfg = synthetic_config(SCENE)
        cfg.update(compiled_floor_m=.008, compiled_rim_m=.183,
                   wall_tolerance_cap_m=.0038, floor_tolerance_cap_m=.004,
                   geometry_epsilon_m=audit.GEOMETRY_EPS_M)
        cfg.pop("fingerprint")
        cfg["fingerprint"] = audit.digest(cfg)
        geometry = {key:cfg[key] for key in ("compiled_planes", "compiled_floor_m", "compiled_rim_m", "wall_tolerance_cap_m", "floor_tolerance_cap_m")}
        rows = [dict(contact_wall_overlap_m=.0036, cavity_wall_overlap_m=.001,
                     contact_floor_overlap_m=.001, cavity_floor_overlap_m=.001,
                     verified_pre_post_bucket_speed_m_s=6.0)]
        before = copy.deepcopy(cfg)
        bounds = audit.check_score_config(cfg, geometry, rows, audit.Checks(), fitted=False)
        self.assertFalse(bounds["wall_inside_envelope"])
        self.assertEqual(cfg, before)
        with self.assertRaisesRegex(audit.CalibrationAuditError, "fit rule"):
            audit.check_score_config(cfg, geometry, rows, audit.Checks(), fitted=True)

    def test_unsafe_source_journal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "producer"
            repo.mkdir()
            members = {name:"a"*64 for name in audit.REQUIRED_PRODUCER_FILES}
            members["../secret"] = "a"*64
            # Reject traversal before attempting any target read. Required files
            # are independently mocked as exact only to reach the traversal guard.
            with mock.patch.object(audit, "sha", return_value="a"*64):
                with self.assertRaisesRegex(audit.CalibrationAuditError, "Unsafe"):
                    audit.source_bindings(repo, {"snapshots":{"producer":members}}, {"source_hashes":{}}, audit.Checks())

    def test_symlink_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actual.json"
            path.write_text("{}")
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(audit.CalibrationAuditError):
                audit.read(link)


if __name__ == "__main__":
    unittest.main()
