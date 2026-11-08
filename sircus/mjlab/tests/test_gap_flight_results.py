"""Synthetic unit controls for bound native contact-result derivation."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import gap_flight_results as results
from sircus.mjlab.tests.test_prepare_gap_flight_report import correction_fixture


def packet():
    ticks, n, dt = 6, 2, 1/15360
    arrays = dict(record_ids=np.array(["one", "two"]), position_m=np.zeros((ticks+1, n, 3)),
        velocity_m_s=np.zeros((ticks+1, n, 3)), forces_N=np.zeros((ticks, n, 6)),
        time_s=np.arange(ticks+1)*dt, dt_s=np.array(dt), native_time_s=np.repeat((np.arange(ticks+1)*dt)[:, None], n, axis=1),
        contact_distance_m=np.full((ticks, n, 2), np.inf), contact_geom_ids=np.full((ticks, n, 2, 2), -1, dtype=int),
        contact_position_m=np.zeros((ticks, n, 2, 3)), contact_normal=np.zeros((ticks, n, 2, 3)))
    rows = [dict(id="one", native_cache_row=0), dict(id="two", native_cache_row=1)]
    return arrays, rows, dict(native_dt_s=dt, horizon_s=ticks*dt)


def contact_fixture():
    arrays, _, _ = packet()
    trace = {k: v[:, 0] if v.ndim >= 2 and k != "time_s" else v for k, v in arrays.items() if k != "record_ids"}
    row = dict(id="one", repeat=0, native_release=True, route_success=False, route_quality=0,
        invalid=None, invalid_physics=False, whole_ball_entry=True, max_retention_s=1., physics_steps=4)
    metric = dict(available=True, start_index=1, end_index=2, start_time_s=trace["time_s"][1],
        end_time_s=trace["time_s"][2], cutoff_reason="contact", cutoff_contact_channels=["robot"])
    return row, trace, metric


class ContactResultTests(unittest.TestCase):
    def test_valid_packet_binds_every_column_and_all_numeric_arrays(self):
        arrays, rows, spec = packet(); checks = results.Checks()
        results.validate_packet(arrays, rows, spec, checks)
        self.assertGreater(checks.count, 20)

    def test_wrong_duplicate_missing_or_reassigned_columns_rejected(self):
        for mutation in ("duplicate_ids", "swapped_ids", "duplicate_columns", "missing_column", "fractional_column"):
            with self.subTest(mutation=mutation):
                arrays, rows, spec = packet()
                if mutation == "duplicate_ids": arrays["record_ids"] = np.array(["one", "one"])
                elif mutation == "swapped_ids": arrays["record_ids"] = arrays["record_ids"][::-1]
                elif mutation == "duplicate_columns": rows[-1]["native_cache_row"] = 0
                elif mutation == "missing_column": rows.pop()
                else: rows[-1]["native_cache_row"] = .5
                with self.assertRaises(ValueError): results.validate_packet(arrays, rows, spec, results.Checks())

    def test_malformed_or_unbound_numeric_observations_rejected(self):
        for mutation in ("missing", "shape", "nan", "negative", "time", "dt", "contact_ids", "contact_distance", "contact_normal"):
            with self.subTest(mutation=mutation):
                arrays, rows, spec = packet()
                if mutation == "missing": arrays.pop("native_time_s")
                elif mutation == "shape": arrays["forces_N"] = arrays["forces_N"][:-1]
                elif mutation == "nan": arrays["position_m"][1, 0, 1] = np.nan
                elif mutation == "negative": arrays["forces_N"][1, 0, 4] = -.01
                elif mutation == "time": arrays["time_s"][2] += 1e-8
                elif mutation == "dt": arrays["dt_s"] = np.array(1/30720)
                elif mutation == "contact_ids": arrays["contact_geom_ids"][0, 0, 0] = [1, 2]
                elif mutation == "contact_distance": arrays["contact_distance_m"][0, 0, 0] = -np.inf
                else:
                    arrays["contact_distance_m"][0, 0, 0] = -.001
                    arrays["contact_geom_ids"][0, 0, 0] = [1, 2]
                    arrays["contact_normal"][0, 0, 0] = [.5, 0, 0]
                with self.assertRaises(ValueError): results.validate_packet(arrays, rows, spec, results.Checks())

    def test_strict_existing_threshold_and_terminal_split(self):
        row, trace, metric = contact_fixture()
        trace["forces_N"][0, 4] = 99.  # Before the first rebound: separate, excluded here.
        trace["forces_N"][1, 4] = .01  # Equal to threshold is not active.
        trace["forces_N"][2, 4] = np.nextafter(.01, 1.)
        trace["forces_N"][4, 4] = 3.   # Starts at scorer terminal: tail only.
        measured = results.scan_contact(row, trace, metric)
        self.assertEqual(measured["robot_contact_intervals_before_terminal"], 1)
        self.assertEqual(measured["robot_contact_intervals_after_terminal"], 1)
        self.assertEqual(measured["robot_contact_intervals_post_rebound"], 2)
        self.assertEqual(measured["first_robot_contact_time_s"], trace["time_s"][2])
        self.assertEqual(measured["last_robot_contact_time_s"], trace["time_s"][5])
        self.assertEqual(measured["peak_robot_force_N_after_terminal"], 3.)
        self.assertEqual(measured["first_post_bounce_contact_channels"], ["robot"])
        self.assertEqual(measured["terminal_time_s"], trace["time_s"][4])
        self.assertEqual(measured["trace_end_time_s"], trace["time_s"][-1])

    def test_missing_rebound_or_rejected_request_never_becomes_zero_contacts(self):
        row, trace, metric = contact_fixture()
        for native in (True, False):
            changed = dict(row, native_release=native)
            measured = results.scan_contact(changed, trace, dict(available=False))
            self.assertIsNone(measured["robot_contact_intervals_before_terminal"])
            self.assertIsNone(measured["first_post_bounce_contact_channels"])
            summary = results.summarize([measured])
            self.assertEqual(summary["contact_unknown"], 1)
            self.assertIsNone(summary["robot_contact_trials_before_terminal"])
            self.assertIsNone(summary["peak_robot_force_N_before_terminal"])
        measured = results.scan_contact(row, trace, metric)
        summary = results.summarize([measured, results.scan_contact(dict(row, native_release=False), {}, {})])
        self.assertIsNone(summary["robot_contact_trials_before_terminal"])
        self.assertEqual(summary["contact_observations"], 1)

    def test_observed_zero_and_bucket_contact_stay_distinct_from_unknown_and_rim(self):
        row, trace, metric = contact_fixture()
        metric["cutoff_contact_channels"] = ["bucket"]
        observed = results.scan_contact(dict(row, route_success=True, route_quality=1), trace, metric)
        summary = results.summarize([observed]*3)
        self.assertEqual(summary["robot_contact_trials_before_terminal"], 0)
        self.assertEqual(summary["peak_robot_force_N_before_terminal"], 0.)
        self.assertEqual(summary["first_post_bounce_contact_counts"], {"bucket": 3})
        self.assertEqual(summary["captures"], 3)
        self.assertEqual(summary["clean"], 0)
        self.assertEqual(summary["retention_s"], [1., 1., 1.])
        failed_route = results.scan_contact(row, trace, metric)
        self.assertEqual(results.summarize([failed_route])["captures"], 0)

    def test_scope_indices_and_threshold_changes_fail_closed(self):
        row, trace, metric = contact_fixture()
        for start, terminal in ((5, 4), (-1, 4), (1, 8), (.5, 4)):
            with self.assertRaises(ValueError): results.scan_contact(dict(row, physics_steps=terminal), trace, dict(metric, start_index=start))
        with patch.object(results, "FORCE_THRESHOLD_N", .1):
            with self.assertRaises(ValueError): results.scan_contact(row, trace, metric)

    def test_derivative_requires_full_audit_and_bound_native_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            study, audit, _ = correction_fixture(Path(temp), audit=False)
            with self.assertRaisesRegex(ValueError, "fully verified"): results.build(study, audit)
        with tempfile.TemporaryDirectory() as temp:
            study, audit, _ = correction_fixture(Path(temp))
            target = study/"raw-fixture.dat"; target.write_bytes(b"changed after audit")
            with self.assertRaisesRegex(ValueError, "artifact changed"): results.build(study, audit)

    def test_changed_oracle_helper_rejected_against_frozen_native_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            study, audit, _ = correction_fixture(Path(temp))
            real_read, real_sha = results.read, results.sha
            names = ("calibrated_score.py", "calibrated_score_oracle.py", "xy650_verify.py")
            pinned = {"sircus/mjlab/"+name: real_sha(Path(results.__file__).with_name(name)) for name in names}
            def read_fixture(path):
                value = real_read(path)
                if Path(path).name == "contract.json": value["native_source_sha256"] = pinned
                return value
            def altered_hash(path):
                return "0"*64 if Path(path).name == "calibrated_score_oracle.py" else real_sha(path)
            with patch.object(results, "read", side_effect=read_fixture), patch.object(results, "sha", side_effect=altered_hash):
                with self.assertRaisesRegex(ValueError, "Exact original contact/scorer helper"):
                    results.build(study, audit)


if __name__ == "__main__": unittest.main()
