"""Independent temporal/contact controls; no engine, GPU or old score imports."""

import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from sircus.mjlab import calibrated_score as primary
from sircus.mjlab import calibrated_score_oracle as oracle

ROOT = Path(__file__).resolve().parents[3]


def config(scene):
    b = scene["bucket"]
    angle = (np.arange(b["segments"]) + 0.5) * 2 * math.pi / b["segments"]
    cosine = math.cos(math.pi / b["segments"])
    slope = (b["inner_top_diameter_m"] - b["inner_bottom_diameter_m"]) / (2 * b["assumed_interior_depth_m"])
    scale = math.sqrt(1 + (slope * cosine) ** 2)
    planes = np.column_stack([-np.cos(angle), -np.sin(angle), np.full(len(angle), slope * cosine), np.full(len(angle), cosine * (b["inner_bottom_diameter_m"] / 2 - slope * b["assumed_bottom_thickness_m"]))]) / scale
    value = dict(version="sircus-contact-capture-calibration-v1", wall_tolerance_m=0.0035, floor_tolerance_m=0.003, geometry_epsilon_m=1e-6, max_speed_m_s=6.25, retention_s=1.0, compiled_planes=planes.tolist(), fixture="synthetic logic geometry; not measured calibration")
    return resign(value)


def resign(value):
    value = copy.deepcopy(value)
    value.pop("fingerprint", None)
    value["fingerprint"] = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return value


def trace(scene, end=1.6, dt=0.001):
    times = np.arange(round(end / dt) + 1) * dt
    b, r = scene["bucket"], scene["ball"]["radius_m"]
    top = b["assumed_bottom_thickness_m"] + b["assumed_interior_depth_m"]
    critical = top - r
    xyz = np.tile([*b["center_xy_m"], 0.0], (len(times), 1))
    xyz[:, 2] = np.maximum(critical + 0.1 - 0.5 * times, b["assumed_bottom_thickness_m"] + r)
    velocity = np.gradient(xyz, times, axis=0)
    force = np.zeros((len(times) - 1, 6))
    contact = min(round(0.05 / dt), len(force) - 2)
    force[contact, 0] = 1.0
    velocity[contact + 2, 2] = 0.1
    return dict(position_m=xyz, velocity_m_s=velocity, forces_N=force, time_s=times)


def truncate(arrays, end):
    out = copy.deepcopy(arrays)
    count = np.searchsorted(out["time_s"], end + 1e-12, side="right")
    for key in ("position_m", "velocity_m_s", "time_s"):
        out[key] = out[key][:count]
    out["forces_N"] = out["forces_N"][:count - 1]
    if "contact_distance_m" in out:
        out["contact_distance_m"] = out["contact_distance_m"][:count - 1]
    return out


class CaptureControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = json.loads((ROOT / "sircus/isaac/scene.json").read_text())
        cls.cfg = config(cls.scene)
        cls.xy = cls.scene["bucket"]["center_xy_m"]
        cls.normal = [0.0, 0.0, 1.0]

    def score(self, arrays, calibration=None, xy=None):
        a = primary.score_trace(self.scene, self.xy if xy is None else xy, self.normal, self.cfg if calibration is None else calibration, arrays)
        b = oracle.score_trace(self.scene, self.xy if xy is None else xy, self.normal, self.cfg if calibration is None else calibration, arrays)
        self.assertEqual(set(a), set(b))
        def compare(x, y, path):
            if isinstance(x, dict):
                self.assertEqual(set(x), set(y), path)
                for key in x:
                    compare(x[key], y[key], path + "/" + key)
            elif isinstance(x, float):
                self.assertAlmostEqual(x, y, delta=1e-9, msg=path)
            else:
                self.assertEqual(x, y, path)
        compare(a, b, "row")
        self.assertAlmostEqual(a["reward"], oracle.reward(a), delta=1e-12)
        return a

    def test_clear_board_then_whole_entry_and_one_second(self):
        out = self.score(trace(self.scene))
        self.assertTrue(out["route_success"])
        self.assertTrue(out["single_bounce_success"])
        self.assertEqual(out["route_quality"], 2)
        self.assertAlmostEqual(out["whole_ball_entry_time_s"], 0.2)
        self.assertAlmostEqual(out["capture_time_s"], 1.2)
        self.assertAlmostEqual(out["max_retention_s"], 1)

    def test_first_inside_sample_has_no_preceding_interval_credit(self):
        out = self.score(truncate(trace(self.scene), 0.2))
        self.assertTrue(out["whole_ball_entry"])
        self.assertFalse(out["success"])
        self.assertEqual(out["max_retention_s"], 0)

    def test_just_short_of_second_and_exact_second(self):
        short = self.score(truncate(trace(self.scene), 1.199))
        exact = self.score(truncate(trace(self.scene), 1.2))
        self.assertFalse(short["route_success"])
        self.assertAlmostEqual(short["max_retention_s"], 0.999)
        self.assertTrue(exact["route_success"])

    def test_initial_inside_without_new_open_mouth_entry_never_captures(self):
        a = trace(self.scene)
        a["position_m"][:, 2] = self.scene["bucket"]["assumed_bottom_thickness_m"] + self.scene["ball"]["radius_m"]
        out = self.score(a)
        self.assertTrue(out["board_bounce"])
        self.assertFalse(out["whole_ball_entry"])
        self.assertFalse(out["route_success"])

    def test_previous_above_state_beside_bucket_cannot_enable_sidewall_entry(self):
        a = trace(self.scene)
        a["position_m"][a["time_s"] < 0.5, 0] += 0.3
        out = self.score(a)
        self.assertFalse(out["whole_ball_entry"])
        self.assertFalse(out["route_success"])
        self.assertFalse(out["invalid_physics"])

    def test_rim_assisted_capture_remains_success(self):
        a = trace(self.scene)
        t = a["time_s"]
        a["position_m"][:, 0] += np.where(t <= 0.2, 0.05, np.maximum(0, 0.05 - (t - 0.2) * 0.25))
        a["forces_N"][199, 1] = 2.0
        out = self.score(a)
        self.assertTrue(out["route_success"])
        self.assertEqual(out["route_quality"], 1)
        self.assertTrue(out["rim_band_contact"])

    def test_calibrated_floor_contact_does_not_reset_retention(self):
        a = trace(self.scene)
        mask = a["time_s"] >= 0.4
        a["position_m"][mask, 2] -= 0.002
        a["forces_N"][400:, 1] = 2.0
        out = self.score(a)
        self.assertTrue(out["route_success"])
        self.assertAlmostEqual(out["max_nominal_floor_overlap_m"], 0.002)

    def test_calibrated_wall_contact_does_not_reset_retention(self):
        a = trace(self.scene)
        b, r = self.scene["bucket"], self.scene["ball"]["radius_m"]
        angle = math.pi / b["segments"]
        cosine = math.cos(angle)
        slope = (b["inner_top_diameter_m"] - b["inner_bottom_diameter_m"]) / (2 * b["assumed_interior_depth_m"])
        scale = math.sqrt(1 + (slope * cosine) ** 2)
        height = b["assumed_bottom_thickness_m"] + r
        radial = (b["inner_bottom_diameter_m"] / 2 + slope * (height - b["assumed_bottom_thickness_m"])) * cosine - (r - 0.0025) * scale
        weight = np.clip((a["time_s"] - 0.3) / 0.1, 0, 1)
        a["position_m"][:, 0] += weight * radial * math.cos(angle)
        a["position_m"][:, 1] += weight * radial * math.sin(angle)
        out = self.score(a)
        self.assertTrue(out["route_success"])
        self.assertAlmostEqual(out["max_nominal_wall_overlap_m"], 0.0025)

    def test_excess_floor_overlap_is_invalid_not_a_capture(self):
        a = trace(self.scene)
        a["position_m"][a["time_s"] >= 0.4, 2] -= 0.004
        out = self.score(a)
        self.assertTrue(out["invalid_physics"])
        self.assertFalse(out["route_success"])
        self.assertEqual(out["reward"], -1)

    def test_partial_exit_resets_and_reentry_earns_fresh_second(self):
        a = trace(self.scene, end=2.2)
        top = self.scene["bucket"]["assumed_bottom_thickness_m"] + self.scene["bucket"]["assumed_interior_depth_m"]
        r = self.scene["ball"]["radius_m"]
        a["position_m"][(a["time_s"] >= 0.9) & (a["time_s"] < 0.95), 2] = top - r + 0.001
        out = self.score(a)
        self.assertTrue(out["route_success"])
        self.assertAlmostEqual(out["capture_time_s"], 1.95)
        self.assertTrue(out["exited_after_entry"])
        self.assertGreaterEqual(out["retention_resets"], 1)

    def test_disjoint_short_inside_periods_do_not_accumulate(self):
        a = trace(self.scene, end=1.5)
        b = self.scene["bucket"]
        a["position_m"][(a["time_s"] >= 0.8) & (a["time_s"] < 0.85), 2] = b["assumed_bottom_thickness_m"] + b["assumed_interior_depth_m"] - self.scene["ball"]["radius_m"] + 0.001
        out = self.score(a)
        self.assertFalse(out["route_success"])
        self.assertAlmostEqual(out["max_retention_s"], 0.65)

    def test_success_terminal_ignores_later_escape_and_bad_physics(self):
        a = trace(self.scene)
        a["position_m"][a["time_s"] >= 1.3, 2] = 1.0
        a["velocity_m_s"][a["time_s"] >= 1.3] = 100.0
        a["forces_N"][1300:, 3] = 10.0
        out = self.score(a)
        self.assertTrue(out["route_success"])
        self.assertFalse(out["invalid_physics"])
        self.assertEqual(out["external_contact_steps"], 0)
        self.assertEqual(out["physics_steps"], 1200)

    def test_external_assistance_disqualifies_route_and_reward_bonuses(self):
        a = trace(self.scene)
        a["forces_N"][500, 2] = 1.0
        out = self.score(a)
        self.assertTrue(out["success"])
        self.assertFalse(out["route_success"])
        self.assertEqual(out["route_quality"], 0)
        self.assertEqual(out["external_contact_steps"], 1)
        self.assertLess(out["reward"], 0.2)

    def test_no_board_rebound_does_not_complete_shot(self):
        a = trace(self.scene)
        a["forces_N"][:, 0] = 0
        out = self.score(a)
        self.assertFalse(out["board_bounce"])
        self.assertFalse(out["whole_ball_entry"])
        self.assertFalse(out["route_success"])

    def test_bucket_domain_speed_guard_and_ordinary_fast_freefall(self):
        a = trace(self.scene)
        a["velocity_m_s"][600, 0] = 6.3
        out = self.score(a)
        self.assertTrue(out["invalid_physics"])
        self.assertFalse(out["route_success"])
        b = trace(self.scene)
        b["position_m"][:, 0] += 0.3
        b["position_m"][:, 2] = np.linspace(0.6, -0.8, len(b["time_s"]))
        b["velocity_m_s"][:] = [0, 0, -10]
        miss = self.score(b)
        self.assertFalse(miss["invalid_physics"])
        self.assertFalse(miss["route_success"])
        self.assertGreater(miss["max_speed_m_s"], self.cfg["max_speed_m_s"])

    def test_contact_distance_guard_uses_actual_bucket_contacts(self):
        a = trace(self.scene)
        a["position_m"][:, 0] += 0.3
        a["contact_distance_m"] = np.full((len(a["time_s"]) - 1, 2), np.inf)
        a["contact_distance_m"][100, 0] = -0.005
        out = self.score(a)
        self.assertTrue(out["invalid_physics"])
        self.assertIn("contact_wall_penetration_outside_calibration", out["invalid_physics_reasons"])

    def test_contact_absence_inf_is_allowed_and_recorded(self):
        a = trace(self.scene)
        a["contact_distance_m"] = np.full((len(a["time_s"]) - 1, 2), np.inf)
        out = self.score(a)
        self.assertTrue(out["route_success"])
        self.assertTrue(out["contact_distance_evidence"])
        self.assertEqual(out["max_contact_wall_penetration_m"], 0)

    def test_multiple_board_episodes_are_not_single_bounce(self):
        a = trace(self.scene)
        a["forces_N"][120, 0] = 1.0
        out = self.score(a)
        self.assertTrue(out["route_success"])
        self.assertEqual(out["board_contact_episodes"], 2)
        self.assertFalse(out["single_bounce_success"])

    def test_plane_order_and_bucket_translation_invariance(self):
        a = trace(self.scene)
        expected = self.score(a)
        cfg = copy.deepcopy(self.cfg)
        cfg["compiled_planes"] = cfg["compiled_planes"][::-1]
        cfg = resign(cfg)
        moved = copy.deepcopy(a)
        shift = np.array([0.017, -0.023])
        moved["position_m"][:, :2] += shift
        actual = self.score(moved, cfg, np.asarray(self.xy) + shift)
        for name in ("route_success", "whole_ball_entry", "capture_time_s", "board_contact_episodes", "external_contact_steps"):
            self.assertEqual(expected[name], actual[name])

    def test_explicit_nonuniform_times_and_uniform_dt(self):
        a = trace(self.scene)
        a["dt_s"] = 0.001
        self.assertTrue(self.score(a)["route_success"])
        a.pop("dt_s")
        a["time_s"][300] += 0.0001
        self.assertTrue(self.score(a)["route_success"])

    def test_single_native_tick_rim_exit_resets_full_second(self):
        a = trace(self.scene, end=2.0, dt=1 / 15360)
        b = self.scene["bucket"]
        r = self.scene["ball"]["radius_m"]
        top = b["assumed_bottom_thickness_m"] + b["assumed_interior_depth_m"]
        index = round(0.8 * 15360)
        a["position_m"][index, 2] = top - r + 0.0001
        out = self.score(a)
        self.assertTrue(out["route_success"])
        self.assertAlmostEqual(out["capture_time_s"], a["time_s"][index + 1] + 1)
        self.assertTrue(out["exited_after_entry"])

    def test_force_interval_ending_at_capture_counts_but_later_interval_does_not(self):
        before = trace(self.scene)
        before["forces_N"][1199, 2] = 1
        out = self.score(before)
        self.assertTrue(out["success"])
        self.assertFalse(out["route_success"])
        self.assertEqual(out["external_contact_steps"], 1)
        after = trace(self.scene)
        after["forces_N"][1200, 2] = 1
        out = self.score(after)
        self.assertTrue(out["route_success"])
        self.assertEqual(out["external_contact_steps"], 0)

    def test_active_bucket_contact_high_speed_is_invalid_even_above_rim(self):
        a = trace(self.scene)
        a["position_m"][:, 0] += 0.3
        a["forces_N"][10, 1] = 1
        a["velocity_m_s"][10, 0] = 6.3
        out = self.score(a)
        self.assertTrue(out["invalid_physics"])
        self.assertIn("bucket_contact_speed_outside_calibration", out["invalid_physics_reasons"])

    def test_no_unrequested_low_speed_capture_requirement(self):
        a = trace(self.scene)
        a["velocity_m_s"][a["time_s"] > 0.4, 0] = 5.0
        out = self.score(a)
        self.assertTrue(out["route_success"])
        self.assertAlmostEqual(out["max_bucket_domain_speed_m_s"], 5.0)

    def test_excessive_wall_overlap_in_cavity_is_invalid(self):
        a = trace(self.scene)
        b = self.scene["bucket"]
        r = self.scene["ball"]["radius_m"]
        plane = np.asarray(self.cfg["compiled_planes"])[0]
        z = b["assumed_bottom_thickness_m"] + r
        radial = (plane[2] * z + plane[3] - (r - 0.0045)) / math.hypot(*plane[:2])
        direction = -plane[:2] / np.linalg.norm(plane[:2])
        a["position_m"][a["time_s"] >= 0.4, :2] = np.asarray(self.xy) + radial * direction
        out = self.score(a)
        self.assertTrue(out["invalid_physics"])
        self.assertFalse(out["route_success"])

    def test_corrupt_inputs_and_calibration_are_rejected(self):
        for implementation in (primary, oracle):
            for key, value in (("wall_tolerance_m", -1), ("retention_s", 0.9), ("max_speed_m_s", True)):
                with self.subTest(implementation=implementation.__name__, key=key):
                    cfg = copy.deepcopy(self.cfg)
                    cfg[key] = value
                    with self.assertRaises(ValueError):
                        implementation.score_trace(self.scene, self.xy, self.normal, resign(cfg), trace(self.scene))
            bad = copy.deepcopy(self.cfg)
            bad["wall_tolerance_m"] += 0.001
            with self.assertRaises(ValueError):
                implementation.validate_calibration(bad)
            for kind in ("shape", "nan", "force", "time", "dt", "distance"):
                a = trace(self.scene)
                if kind == "shape":a["position_m"] = a["position_m"][:-1]
                elif kind == "nan":a["velocity_m_s"][0, 0] = np.nan
                elif kind == "force":a["forces_N"][0, 0] = -1
                elif kind == "time":a["time_s"][5] = a["time_s"][4]
                elif kind == "dt":a["dt_s"] = 0.002
                else:a["contact_distance_m"] = np.full((len(a["time_s"]) - 1, 2), np.nan)
                with self.subTest(implementation=implementation.__name__, kind=kind):
                    with self.assertRaises(ValueError):
                        implementation.score_trace(self.scene, self.xy, self.normal, self.cfg, a)

    def test_compiled_geometry_changed_even_with_new_digest_is_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["compiled_planes"][0][3] += 0.001
        for implementation in (primary, oracle):
            with self.assertRaises(ValueError):
                implementation.score_trace(self.scene, self.xy, self.normal, resign(cfg), trace(self.scene))

    def test_repeated_compiled_plane_cannot_replace_a_closed_bucket(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["compiled_planes"] = [cfg["compiled_planes"][0]] * len(cfg["compiled_planes"])
        for implementation in (primary, oracle):
            with self.assertRaises(ValueError):
                implementation.score_trace(self.scene, self.xy, self.normal, resign(cfg), trace(self.scene))

    def test_invalid_physics_at_capture_boundary_wins_over_success(self):
        a = trace(self.scene)
        a["velocity_m_s"][1200, 0] = 6.3
        out = self.score(a)
        self.assertTrue(out["invalid_physics"])
        self.assertFalse(out["route_success"])
        self.assertIsNone(out["capture_time_s"])

    def test_dense_native_cadence_is_supported_without_python_tick_scan(self):
        a = trace(self.scene, end=1.6, dt=1 / 15360)
        out = self.score(a)
        self.assertTrue(out["route_success"])
        self.assertAlmostEqual(out["capture_time_s"] - out["whole_ball_entry_time_s"], 1, delta=1 / 15360)


if __name__ == "__main__":
    unittest.main()
