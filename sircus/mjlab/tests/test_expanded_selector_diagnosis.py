"""Observed-flight geometry, source authentication and proposal-feature checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab.benchmark_verify import Checks, sha
from sircus.mjlab.expanded_selector_diagnosis import (
    findings, rim_crossings, selection_details, setting_delta, verify_parent,
)


class FlightGeometryTests(unittest.TestCase):
    def setUp(self):
        self.geometry = dict(rim_m=.2, ball_radius_m=.025,
            inward_rim_halfplanes=[[1, 0, .1], [-1, 0, .1], [0, 1, .1], [0, -1, .1]])

    def test_observed_rim_interpolation_and_signed_short_lateral(self):
        crossings = rim_crossings([0, .2], [[.8, .02, .3], [1., .02, .1]],
                                 [1., 0.], [0., 0., .4], self.geometry, after_s=0.)
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0]["time_s"], .1)
        self.assertAlmostEqual(crossings[0]["along_error_m"], -.1)
        self.assertAlmostEqual(crossings[0]["lateral_error_m"], .02)
        self.assertAlmostEqual(crossings[0]["aperture_clearance_m"], -.025)

    def test_upward_crossing_ignored_but_later_table_return_retained(self):
        t = [0, .1, .2, .3, .4]
        p = [[1., 0., z] for z in (.3, .1, .4, .3, .1)]
        crossings = rim_crossings(t, p, [1., 0.], [0., 0., .4], self.geometry, after_s=.1)
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0]["time_s"], .35)
        self.assertAlmostEqual(crossings[0]["aperture_clearance_m"], .075)

    def test_missing_crossing_is_not_extrapolated(self):
        self.assertEqual(rim_crossings([0, .1], [[.1, 0., .1], [.2, 0., .15]],
                         [1., 0.], [0., 0., .4], self.geometry, after_s=0.), [])

    def test_bad_time_and_undefined_axis_refused(self):
        with self.assertRaises(ValueError):
            rim_crossings([0, 0], [[0., 0., .3], [0., 0., .1]],
                          [1., 0.], [0., 0., .4], self.geometry, after_s=0.)
        with self.assertRaises(ValueError):
            rim_crossings([0, 1], [[0., 0., .3], [0., 0., .1]],
                          [0., 0.], [0., 0., .4], self.geometry, after_s=0.)

    def test_joint_delta_preserves_distinct_wrapped_branch(self):
        a = dict(command_board_center_m=[0, 0, .3], board_center_m=[0, 0, .3],
                 release_ball_center_m=[0, 0, .6], q10_rad=[0]*6, q14_rad=[0]*6,
                 board_height_m=.3, tilt_deg=20, drop_gap_m=.3, heading_offset_deg=0,
                 twist_deg=0, board_bucket_distance_m=.5)
        b = copy.deepcopy(a)
        b["q14_rad"][5] = 2*np.pi
        b["tilt_deg"] += 3
        delta = setting_delta(a, b)
        self.assertEqual(delta["q14_rad"][5], 2*np.pi)
        self.assertEqual(delta["tilt_deg"], 3)

    def test_capture_with_table_rebound_is_not_described_as_short_miss(self):
        first = dict(along_error_m=-.3, lateral_error_m=0., previous_nonboard_contact=None)
        later = dict(along_error_m=.005, lateral_error_m=0., previous_nonboard_contact="table")
        shot = dict(outcome=dict(capture=True, route_class="table_assisted", capture_time_s=2.,
                        retention_resets=0, board_contact_episodes=2),
                    flight_measurements=dict(first_robot_contact=None, first_bucket_contact_geometry=None,
                        first_descending_center_rim=first, first_descending_center_rim_after_table=later))
        facts, _ = findings({"expanded":shot})
        self.assertTrue(any("captured" in text for text in facts))
        self.assertTrue(any("5.0 mm beyond" in text for text in facts))
        self.assertFalse(any("300.0" in text or "entered but" in text for text in facts))


class ProvenanceTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir(exist_ok=True)
        for name in ("contract.json", "choices.json"):
            (root/name).write_text("{}\n")
        manifest = {name: sha(root/name) for name in ("contract.json", "choices.json")}
        (root/"evidence-manifest.json").write_text(json.dumps(manifest))
        report = dict(status="completed", evidence_manifest_sha256=sha(root/"evidence-manifest.json"))
        (root/"report.json").write_text(json.dumps(report))
        audit = dict(status="passed", version="sircus-expanded-selector-independent-audit-v1",
                     report_sha256=sha(root/"report.json"), evidence_manifest_sha256=sha(root/"evidence-manifest.json"),
                     contract_sha256=sha(root/"contract.json"), choices_sha256=sha(root/"choices.json"))
        audit_path = root/"audit.json"
        audit_path.write_text(json.dumps(audit))
        return audit_path

    def test_parent_and_consumed_artifact_binding(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            audit = self.fixture(root)
            _, _, bound, verified = verify_parent(root, audit, Checks())
            self.assertEqual(bound("choices.json"), (root/"choices.json").resolve())
            self.assertEqual(len(verified), 2)

    def test_mutated_choices_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            audit = self.fixture(root)
            (root/"choices.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verify_parent(root, audit, Checks())

    def test_mutated_report_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            audit = self.fixture(root)
            (root/"report.json").write_text('{"status":"completed","changed":true}')
            with self.assertRaises(ValueError):
                verify_parent(root, audit, Checks())

    def test_probabilities_use_original_proposal_not_solved_query(self):
        proposal = dict(id="proposal", seed_q10=[1]*6)
        request = dict(proposal_id="proposal", seed_q10=[2]*6,
                       selected_q10=[3]*6, intended_route="direct")
        site = dict(plan={"neural": {"proposals": [proposal]}}, decision_latency_s=2,
                    request_wall_s=3, return_reason="finished", stalled_reference={"selected_adapter":"reference"})
        slot = dict(ranked_proposal_ids=["proposal"], checked_proposals=1, candidate_pool="neural_checked",
                    prediction={}, predicted_flight_risk=False)
        with patch("sircus.mjlab.neural_shot_model.predict", return_value={"capture_probability":.8}) as mocked:
            result = selection_details(site, slot, request, {}, Checks())
            mocked.assert_called_once_with({}, proposal)
        self.assertEqual(result["proposal_rank"], 1)


if __name__ == "__main__":
    unittest.main()
