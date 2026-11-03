"""Source/contract binding and public projection, without sockets or GPU work."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from sircus.mjlab import benchmark_contract as contract
from sircus.mjlab import benchmark_server as server
from sircus.mjlab.interface import RequestError
from sircus.mjlab.learning_contract import digest


class BenchmarkServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.benchmark = self.directory / "benchmark"
        self.benchmark.mkdir()
        self.identity = dict(scene_sha256="a" * 64, source_sha256={"native": "b" * 64})
        self.protocol = contract.protocol(seeds=[7], budget=32, milestones=[0, 32])
        self.protocol["prior"] = dict(recipes=17, full_historical_cost_known=False)
        self.contract = dict(
            version=contract.VERSION, identity=self.identity, protocol=self.protocol
        )
        self.sign(self.contract)
        self.contract_path = self.benchmark / "contract.json"
        self.report_path = self.benchmark / "report.json"
        self.audit_path = self.directory / "independent-audit.json"
        self.raw = dict(
            version=contract.VERSION,
            status="running",
            phase="training",
            protocol=self.protocol,
            contract_fingerprint=self.contract["fingerprint"],
            curves=[],
            runs=[],
            final_test=[],
        )
        self.sign(self.raw)
        self.write(self.contract_path, self.contract)
        self.write(self.report_path, self.raw)
        self.solver = Mock(
            side_effect=AssertionError("Display must not invoke native solver")
        )
        self.base = dict(
            planner_only=True,
            planner_grid_study=dict(status="completed"),
            unrelated=dict(preserved=True),
        )
        self.patches = [
            patch.object(server.GridState, "__init__", return_value=None),
            patch.object(
                server.GridState,
                "report",
                autospec=True,
                side_effect=lambda state: copy.deepcopy(self.base),
            ),
            patch.object(
                server,
                "benchmark_helpers",
                return_value=(lambda ref: self.identity, contract.validate_report),
            ),
        ]
        for p in self.patches:
            p.start()
        self.state = self.make_state()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    @staticmethod
    def sign(value):
        value.pop("fingerprint", None)
        value["fingerprint"] = digest(value)
        return value

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def make_state(self):
        return server.BenchmarkState(
            {},
            self.directory / "old-grid",
            self.directory / "display.json",
            None,
            self.solver,
            self.benchmark,
            benchmark_audit=self.audit_path,
        )

    def assert_unavailable(self):
        value = self.state.report()
        self.assertIn("sweep_rl_error", value)
        self.assertNotIn("sweep_rl_benchmark", value)
        self.assertEqual(value["unrelated"], self.base["unrelated"])
        self.solver.assert_not_called()

    def test_preserves_existing_study_and_never_invokes_solver(self):
        result = self.state.report()
        self.assertEqual(result["planner_grid_study"], self.base["planner_grid_study"])
        self.assertEqual(result["sweep_rl_benchmark"]["status"], "running")
        self.assertEqual(result["sweep_rl_benchmark"]["curves"], [])
        self.assertNotIn("audit", result["sweep_rl_benchmark"])
        self.solver.assert_not_called()

    def test_contract_source_mismatch_rejected_at_startup(self):
        self.contract["identity"] = dict(scene_sha256="f" * 64)
        self.sign(self.contract)
        self.write(self.contract_path, self.contract)
        with self.assertRaises(RequestError):
            self.make_state()

    def test_contract_bytes_cannot_change_after_startup(self):
        self.contract_path.write_text(self.contract_path.read_text() + "\n")
        self.assert_unavailable()

    def test_wrong_contract_binding_rejected_even_with_resigned_report(self):
        self.raw["contract_fingerprint"] = "c" * 64
        self.write(self.report_path, self.sign(self.raw))
        self.assert_unavailable()

    def test_changed_protocol_rejected_even_with_resigned_report(self):
        self.raw["protocol"] = dict(self.protocol, sample_budget=64, milestones=[0, 64])
        self.write(self.report_path, self.sign(self.raw))
        self.assert_unavailable()

    def test_bad_report_fingerprint_does_not_display_stale_counts(self):
        self.raw["phase"] = "modified without fingerprint"
        self.write(self.report_path, self.raw)
        self.assert_unavailable()

    def test_missing_report_is_explicit_and_does_not_invent_zeros(self):
        self.report_path.unlink()
        self.assert_unavailable()

    def test_public_projection_removes_host_paths_and_raw_records(self):
        raw = copy.deepcopy(self.raw)
        raw.update(
            command=["private-command"],
            checkpoint_path="/private/checkpoint.pt",
            raw_rows=[dict(private=True)],
        )
        raw["protocol"].update(local_paths={"assets": "/private/assets"})
        raw["protocol"]["prior"].update(local_path="/private/prior")
        raw["curves"] = [
            dict(
                method="ppo",
                seed=7,
                samples=0,
                artifact_sha256="a" * 64,
                validation=dict(
                    attempts=36,
                    captures=2,
                    native_releases=34,
                    invalid=2,
                    clean=1,
                    raw_rows=[dict(private=True)],
                ),
                resources=dict(wall_s=2, peak_rss_mb=100, private_path="/private/host"),
                checkpoint_path="/private/checkpoint.pt",
            )
        ]
        raw["runs"] = [
            dict(
                method="ppo",
                seed=7,
                status="running",
                checkpoint_path="/private/checkpoint.pt",
                resources={},
            )
        ]
        value = server.public_benchmark(raw)
        serialized = json.dumps(value)
        self.assertNotIn("/private", serialized)
        self.assertNotIn("raw_rows", serialized)
        self.assertNotIn("private-command", serialized)
        self.assertEqual(value["curves"][0]["validation"]["attempts"], 36)

    def test_partial_test_feedback_hidden_until_completed(self):
        raw = dict(
            self.raw,
            final_test=[
                dict(
                    method="ppo",
                    seed=7,
                    counts=dict(attempts=72, captures=70),
                    frozen_artifact_sha256="a" * 64,
                    query_s=0.1,
                )
            ],
        )
        self.assertEqual(server.public_benchmark(raw)["final_test"], [])
        raw["status"] = "completed"
        self.assertEqual(
            server.public_benchmark(raw)["final_test"][0]["counts"]["captures"], 70
        )

    def test_independent_audit_requires_exact_contract_and_report_bytes(self):
        audit = dict(
            status="passed",
            checks=25,
            contract_fingerprint=self.contract["fingerprint"],
            report_sha256=server.sha(self.report_path),
            report_fingerprint=self.raw["fingerprint"],
        )
        self.write(self.audit_path, audit)
        value = self.state.report()["sweep_rl_benchmark"]
        self.assertEqual(value["audit"], dict(status="passed", checks=25))
        self.raw["phase"] = "next phase"
        self.write(self.report_path, self.sign(self.raw))
        self.assert_unavailable()

    def test_wrong_independent_audit_fingerprint_rejected(self):
        audit = dict(
            status="passed",
            checks=25,
            contract_fingerprint=self.contract["fingerprint"],
            report_sha256=server.sha(self.report_path),
            report_fingerprint="f" * 64,
        )
        self.write(self.audit_path, audit)
        self.assert_unavailable()

    def test_audit_cannot_claim_pass_without_checks_or_with_recorded_failures(self):
        audit = dict(
            status="passed",
            contract_fingerprint=self.contract["fingerprint"],
            report_sha256=server.sha(self.report_path),
        )
        for value in (None, 0, True, "25"):
            self.write(self.audit_path, dict(audit, checks=value))
            self.assert_unavailable()
        self.write(
            self.audit_path, dict(audit, checks=25, failures=["rescore mismatch"])
        )
        self.assert_unavailable()


if __name__ == "__main__":
    unittest.main()
