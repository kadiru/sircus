"""Read-only replay guards with temporary JSON fixtures, without sockets/GPU."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from sircus.mjlab import grid_study_contract as contract
from sircus.mjlab import grid_diverse_study as diverse
from sircus.mjlab import grid_study_server as server
from sircus.mjlab.interface import RequestError
from sircus.mjlab.learning_contract import digest


class GridStudyServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.study = self.directory / "study"
        (self.study / "results").mkdir(parents=True)
        self.layout = contract.grid_layouts()[0]
        self.identity = dict(
            contract_fingerprint="a" * 64,
            scene_sha256="b" * 64,
            asset_sha256={"mesh": "c" * 64},
            source_sha256={"worker": "d" * 64},
            physics=copy.deepcopy(contract.PHYSICS),
        )
        self.result = dict(
            request_id="recorded",
            bucket_xy_m=self.layout["bucket_xy_m"],
            success=True,
            qualified=True,
            status_native="verified_shot",
            method="sweep_planner",
            configuration=dict(
                board_center=[0.6, -0.3, 0.4],
                release_center=[0.6, -0.3, 0.7],
                tilt_deg=32.0,
                gap_m=0.2625,
            ),
            metrics=dict(
                nominal_captures=3,
                nominal_attempts=3,
                fresh_captures=20,
                fresh_attempts=20,
                scalar_score_audit_passed=True,
                invalid=None,
            ),
            provenance=dict(self.identity, policy_used=False, saved_evidence=False),
            motion=dict(passed=True),
            trajectory=[dict(time=0.0, ball=[0.6, -0.3, 0.7])],
        )
        attempt = dict(
            stage="baseline",
            request=dict(
                request_id="recorded",
                mode="search",
                bucket_xy=self.layout["bucket_xy_m"],
            ),
            result=self.result,
            result_url="/api/planner-grid/result/" + self.layout["id"],
        )
        self.report = contract.make_report(
            {self.layout["id"]: [attempt]},
            identity=self.identity,
            protocol=dict(mock=True),
        )
        self.report_path = self.study / "report.json"
        self.result_path = self.study / "results" / (self.layout["id"] + ".json")
        self.write(self.report_path, self.report)
        self.write(self.result_path, self.result)
        self.base_path = self.directory / "base.json"
        self.write(
            self.base_path,
            dict(
                status="not_started",
                planner_only=True,
                planner=dict(ready=True, version="sircus-corrected-sweep-planner-v1"),
            ),
        )
        self.solver = Mock(
            side_effect=AssertionError("Read-only access cannot invoke native solver")
        )
        with patch.object(server, "study_identity", return_value=self.identity):
            self.state = server.GridState(
                {}, self.study, self.base_path, None, self.solver
            )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path, payload):
        path.write_text(json.dumps(payload))

    def resign(self, report):
        raw = dict(report)
        raw.pop("fingerprint", None)
        report["fingerprint"] = digest(raw)

    def assert_status(self, status, action):
        with self.assertRaises(RequestError) as caught:
            action()
        self.assertEqual(caught.exception.status, status)

    def blocked_report(self, homes, *, state=None):
        attempts = []
        for stage, home in homes:
            path = self.study / "cells" / self.layout["id"] / stage / "home.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.write(path, home)
            request_id = "blocked-" + stage
            result = dict(
                request_id=request_id,
                bucket_xy_m=self.layout["bucket_xy_m"],
                success=False,
                status_native="common_home_unresolved",
                method="sweep_planner",
                metrics={},
                provenance=dict(self.identity, policy_used=False, saved_evidence=False),
            )
            attempts.append(
                dict(
                    stage=stage,
                    request=dict(
                        request_id=request_id,
                        mode="search",
                        bucket_xy=self.layout["bucket_xy_m"],
                    ),
                    result=result,
                    evidence_sha256={
                        "home.json": hashlib.sha256(path.read_bytes()).hexdigest()
                    },
                )
            )
        value = contract.make_report(
            {self.layout["id"]: attempts},
            identity=self.identity,
            protocol=dict(mock=True),
            states={self.layout["id"]: state} if state else None,
        )
        self.write(self.report_path, value)
        return value

    @staticmethod
    def home(pair=None, distance=-0.001):
        return dict(
            passed=False,
            reason="clearance",
            margin_contacts=[
                dict(
                    pair=pair or ["Arm10/link_base/collision/0", "bucket/wall00"],
                    distance_m=distance,
                )
            ],
        )

    def followup_state(self):
        parent = self.study / "parent"
        parent.mkdir()
        request = dict(
            request_id=self.result["request_id"],
            mode="search",
            bucket_xy=self.layout["bucket_xy_m"],
        )
        attempt = dict(stage="baseline", request=request, result=self.result)
        states = {
            layout["id"]: "unresolved"
            for layout in contract.grid_layouts()
            if layout["id"] != self.layout["id"]
        }
        validation = dict(
            status="completed",
            configuration_sha256=digest(self.result["configuration"]),
            nominal=dict(captures=3, attempts=3),
            fresh=dict(captures=29, attempts=30),
        )
        parent_contract = dict(
            identity=self.identity,
            protocol=dict(fixture=True),
            packages=dict(fixture="1"),
        )
        parent_contract["fingerprint"] = digest(parent_contract)
        parent_report = contract.make_report(
            {self.layout["id"]: [attempt]},
            identity=self.identity,
            protocol=parent_contract["protocol"],
            status="completed",
            states=states,
            validation_by_cell={self.layout["id"]: validation},
        )
        self.write(parent / "contract.json", parent_contract)
        self.write(parent / "report.json", parent_report)
        self.write(
            parent / "frozen-selection.json",
            {
                row["id"]: row["selected"]
                for row in parent_report["cells"]
                if row.get("selected")
            },
        )
        manifest = {
            name: server.sha(parent / name)
            for name in ("contract.json", "report.json", "frozen-selection.json")
        }
        self.write(parent / "manifest.json", manifest)
        audit = dict(
            status="passed",
            failures=[],
            checks=3,
            contract_fingerprint=parent_contract["fingerprint"],
            study_manifest_sha256=server.sha(parent / "manifest.json"),
            report_sha256=server.sha(parent / "report.json"),
            frozen_selection_sha256=server.sha(parent / "frozen-selection.json"),
            packages=parent_contract["packages"],
            **{
                key: self.identity[key]
                for key in ("source_sha256", "asset_sha256", "physics")
            }
        )
        self.write(parent / "audit.json", audit)
        metadata = diverse.load_parent_snapshot(parent)["metadata"]
        identity = copy.deepcopy(self.identity)
        identity["source_sha256"]["followup"] = "e" * 64
        protocol = diverse.protocol(
            metadata, [layout["id"] for layout in contract.grid_layouts()[1:]]
        )
        result = copy.deepcopy(self.result)
        result.update(
            method="catalog_lookup",
            provenance=dict(identity, policy_used=False, saved_evidence=True),
        )
        current_attempt = dict(
            stage="baseline",
            request=request,
            result=result,
            result_url="/api/planner-grid/result/" + self.layout["id"],
        )
        current_report = contract.make_report(
            {self.layout["id"]: [current_attempt]}, identity=identity, protocol=protocol
        )
        self.write(self.report_path, current_report)
        self.write(self.result_path, result)
        current_contract = dict(identity=identity, protocol=protocol, parent=metadata)
        current_contract["fingerprint"] = digest(current_contract)
        self.write(self.study / "contract.json", current_contract)
        with patch.object(
            server, "study_identity", return_value=self.identity
        ), patch.object(
            server,
            "followup_helpers",
            return_value=(lambda _ref: identity, diverse.load_parent_snapshot),
        ):
            state = server.GridState({}, self.study, self.base_path, None, self.solver)
        return state, parent_report, current_report

    def test_verified_report_and_recorded_shot_preserve_native_identity_without_solver_work(
        self,
    ):
        report = self.state.report()
        self.assertEqual(len(report["planner_grid_study"]["cells"]), 35)
        self.assertTrue(report["search_ready"])
        self.assertFalse(report["policy_ready"])
        self.assertEqual(self.state.recorded_result(self.layout["id"]), self.result)
        self.assertEqual(self.state.snapshot()["status"], "idle")
        self.solver.assert_not_called()

    def test_changed_report_hash_and_source_identity_are_not_served_as_results(self):
        changed = copy.deepcopy(self.report)
        changed["cells"][0]["fresh"]["captures"] = 999
        self.write(self.report_path, changed)
        self.assert_status(503, self.state.grid_report)
        error_report = self.state.report()
        self.assertIn("planner_grid_error", error_report)
        self.assertNotIn("planner_grid_study", error_report)
        other = copy.deepcopy(self.report)
        other["identity"]["source_sha256"]["worker"] = "e" * 64
        self.resign(other)
        self.write(self.report_path, other)
        self.assert_status(409, self.state.grid_report)
        self.solver.assert_not_called()

    def test_result_payload_configuration_bucket_and_request_are_bound_to_selected_recipe(
        self,
    ):
        changes = [
            lambda value: value.update(message="changed after publication"),
            lambda value: value["configuration"].update(tilt_deg=33.0),
            lambda value: value.update(bucket_xy_m=[0.2, -0.4]),
            lambda value: value.update(request_id="different"),
        ]
        for change in changes:
            value = copy.deepcopy(self.result)
            change(value)
            self.write(self.result_path, value)
            self.assert_status(
                409, lambda: self.state.recorded_result(self.layout["id"])
            )
        self.solver.assert_not_called()

    def test_missing_unresolved_and_malformed_cells_fail_without_file_traversal(self):
        for cell_id in (
            "../result",
            "dev-grid-00-00/../../private",
            "/absolute",
            "dev-grid-99-99",
            "dev-grid-0-0",
        ):
            self.assert_status(404, lambda: self.state.recorded_result(cell_id))
        self.assert_status(
            404, lambda: self.state.recorded_result(contract.grid_layouts()[1]["id"])
        )
        self.result_path.unlink()
        self.assert_status(503, lambda: self.state.recorded_result(self.layout["id"]))
        self.solver.assert_not_called()

    def test_get_endpoint_returns_exact_native_result_and_errors_without_submit(self):
        state = Mock()
        state.recorded_result.return_value = self.result
        handler = object.__new__(server.grid_handler(state))
        handler.path = "/api/planner-grid/result/" + self.layout["id"]
        handler._json = Mock()
        handler.do_GET()
        state.recorded_result.assert_called_once_with(self.layout["id"])
        handler._json.assert_called_once_with(200, self.result)
        state.submit.assert_not_called()
        state.recorded_result.side_effect = RequestError(
            "No qualified recorded shot", 404
        )
        handler._json.reset_mock()
        handler.do_GET()
        handler._json.assert_called_once_with(
            404, dict(error="No qualified recorded shot")
        )
        state.submit.assert_not_called()

    def test_hash_checked_home_notes_classify_only_actual_negative_base_contacts(self):
        raw = self.blocked_report([("baseline", self.home())])
        envelope = self.state.report()
        self.assertEqual(envelope["planner_grid_study"], raw)
        self.assertEqual(
            envelope["planner_grid_study"]["fingerprint"], raw["fingerprint"]
        )
        self.assertEqual(
            envelope["planner_grid_blockers"],
            {
                self.layout[
                    "id"
                ]: "Bucket overlaps an arm base; another placement is needed."
            },
        )
        for home in (
            self.home(distance=0.0),
            self.home(distance=0.001),
            self.home(pair=["Arm14/link4/collision/0", "bucket/wall00"]),
        ):
            self.blocked_report([("baseline", home)])
            note = self.state.report()["planner_grid_blockers"][self.layout["id"]]
            self.assertEqual(
                note,
                "Prescribed starting pose is blocked; another staging strategy needs validation.",
            )
        self.solver.assert_not_called()

    def test_notes_use_latest_stage_without_falling_back_to_stale_overlap_evidence(
        self,
    ):
        self.blocked_report(
            [
                ("baseline", self.home()),
                (
                    "expanded",
                    self.home(pair=["Arm14/link4/collision/0", "bucket/wall00"]),
                ),
            ]
        )
        self.assertIn(
            "starting pose",
            self.state.report()["planner_grid_blockers"][self.layout["id"]],
        )
        path = self.study / "cells" / self.layout["id"] / "expanded" / "home.json"
        path.unlink()
        self.assertEqual(self.state.report()["planner_grid_blockers"], {})
        self.solver.assert_not_called()

    def test_unresolved_fixed_base_clearance_is_distinct_from_penetration_and_staging(
        self,
    ):
        expected = "Clearance between the bucket and a fixed arm base is unproven; placement or geometry needs validation."
        for role in ("Arm10", "Arm14"):
            home = dict(
                passed=False,
                reason="unproven_clearance",
                margin_contacts=[],
                minimum_clearance_m=0.0,
                closest_pair=[role + "/link_base/collision/0", "bucket/wall00"],
                unresolved_clearance_pairs=[
                    dict(
                        pair=[role + "/link_base/collision/0", "bucket/wall00"],
                        lower_bound_m=0.0,
                    )
                ],
            )
            raw = self.blocked_report([("baseline", home)])
            envelope = self.state.report()
            self.assertEqual(
                envelope["planner_grid_blockers"][self.layout["id"]], expected
            )
            self.assertEqual(envelope["planner_grid_study"], raw)
        for pair in (
            ["Arm14/link4/collision/0", "bucket/wall00"],
            ["Arm10/link_base/collision/0", "table"],
        ):
            home["unresolved_clearance_pairs"] = [dict(pair=pair, lower_bound_m=0.0)]
            self.blocked_report([("baseline", home)])
            self.assertIn(
                "starting pose",
                self.state.report()["planner_grid_blockers"][self.layout["id"]],
            )
        home = self.home()
        home["unresolved_clearance_pairs"] = [
            dict(
                pair=["Arm14/link_base/collision/0", "bucket/wall00"], lower_bound_m=0.0
            )
        ]
        self.blocked_report([("baseline", home)])
        self.assertEqual(
            self.state.report()["planner_grid_blockers"][self.layout["id"]],
            "Bucket overlaps an arm base; another placement is needed.",
        )
        self.solver.assert_not_called()

    def test_missing_changed_unbound_or_escaped_home_does_not_classify_a_blocker(self):
        self.blocked_report([("baseline", self.home())])
        path = self.study / "cells" / self.layout["id"] / "baseline" / "home.json"
        self.write(path, self.home(distance=0.001))
        self.assertEqual(self.state.report()["planner_grid_blockers"], {})
        report = self.blocked_report([("baseline", self.home())])
        report["cells"][0]["attempts"][0]["evidence_sha256"] = {}
        self.resign(report)
        self.write(self.report_path, report)
        self.assertEqual(self.state.report()["planner_grid_blockers"], {})
        self.blocked_report([("baseline", self.home())])
        outside = self.directory / "outside-home.json"
        self.write(outside, self.home())
        path.unlink()
        path.symlink_to(outside)
        self.assertEqual(self.state.report()["planner_grid_blockers"], {})
        self.solver.assert_not_called()

    def test_unresolved_search_and_invalid_home_do_not_gain_verified_blocker_notes(
        self,
    ):
        self.blocked_report([("baseline", self.home())], state="unresolved")
        self.assertEqual(self.state.report()["planner_grid_blockers"], {})
        for home in (dict(passed=True), dict(passed=False, margin_contacts="unknown")):
            self.blocked_report([("baseline", home)])
            self.assertEqual(self.state.report()["planner_grid_blockers"], {})
        self.solver.assert_not_called()

    def test_followup_envelope_preserves_bound_audited_parent_and_current_reports_separately(
        self,
    ):
        state, parent, current = self.followup_state()
        envelope = state.report()
        self.assertEqual(envelope["planner_grid_parent"], parent)
        self.assertEqual(envelope["planner_grid_study"], current)
        self.assertEqual(
            envelope["planner_grid_parent"]["cells"][0]["frozen_validation"]["fresh"][
                "captures"
            ],
            29,
        )
        self.assertIsNone(
            envelope["planner_grid_study"]["cells"][0]["frozen_validation"]
        )
        self.assertEqual(
            state.recorded_result(self.layout["id"])["method"], "catalog_lookup"
        )
        self.assertEqual(json.loads(self.report_path.read_text()), current)
        self.solver.assert_not_called()

    def test_parent_audit_tampering_never_serves_stale_parent_rates(self):
        state, _parent, _current = self.followup_state()
        state.report()
        audit_path = self.study / "parent" / "audit.json"
        audit = json.loads(audit_path.read_text())
        audit["checks"] += 1
        self.write(audit_path, audit)
        self.assert_status(409, state.grid_report)
        envelope = state.report()
        self.assertIn("planner_grid_error", envelope)
        self.assertNotIn("planner_grid_parent", envelope)
        self.assertNotIn("planner_grid_study", envelope)
        self.solver.assert_not_called()

    def test_current_followup_contract_protocol_and_parent_binding_are_pinned(self):
        state, _parent, current = self.followup_state()
        changed = copy.deepcopy(current)
        changed["protocol"]["parent_manifest_sha256"] = "f" * 64
        self.resign(changed)
        self.write(self.report_path, changed)
        self.assert_status(409, state.grid_report)
        self.write(self.report_path, current)
        path = self.study / "contract.json"
        original = json.loads(path.read_text())
        original["parent"]["audit_sha256"] = "f" * 64
        original.pop("fingerprint")
        original["fingerprint"] = digest(original)
        self.write(path, original)
        self.assert_status(409, state.grid_report)
        self.solver.assert_not_called()

    def test_v1_cannot_claim_parent_evidence_or_an_inherited_baseline_without_a_bound_contract(
        self,
    ):
        with patch.object(server, "study_identity", return_value=self.identity):
            self.assert_status(
                409,
                lambda: server.GridState(
                    {},
                    self.study,
                    self.base_path,
                    None,
                    self.solver,
                    parent_study=self.directory,
                ),
            )
        report = copy.deepcopy(self.report)
        report["protocol"]["inherited_baseline"] = True
        self.resign(report)
        self.write(self.report_path, report)
        self.assert_status(409, self.state.grid_report)
        self.solver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
