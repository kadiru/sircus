"""Source-pinned diagnostic reader; no GPU, hardware or server restart."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from sircus.mjlab import impact_pose_server as server
from sircus.mjlab.interface import RequestError


class ImpactPoseServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "replays").mkdir()
        for name in ("plan.json", "manifest.json"):
            (self.root / name).write_text("{}")
        self.state = object.__new__(server.ImpactPoseState)
        self.state.impact_dir = self.root
        self.state.diagnostic_sha256 = "d" * 64
        diagnostic = dict(source=dict(report_sha256="a" * 64, study_manifest_sha256="b" * 64,
            audit_sha256="c" * 64, contract_fingerprint="e" * 64))
        self.state.diagnostic_report = Mock(return_value=diagnostic)
        self.source = dict(benchmark_report_sha256="a" * 64, benchmark_manifest_sha256="b" * 64,
            benchmark_audit_sha256="c" * 64, validation_report_sha256="d" * 64,
            contract_fingerprint="e" * 64)
        self.replay_id = "impact-0123456789abcdef"
        self.replay = dict(version="sircus-impact-pose-replay-v1", id=self.replay_id,
            role="validation_diagnostic", source=self.source, source_row_id="original-validation-05",
            result=dict(request_id="sweep-case-01", method="impact_sweep_recorded_replay", bucket_xy_m=[.9, -.4]))
        self.replay_path = self.root / "replays" / (self.replay_id + ".json")
        self.replay_path.write_text(json.dumps(self.replay))
        self.value = dict(version=server.VERSION, status="completed", synthetic_fixture=False, source=self.source,
            layouts=[dict(id="case-01", bucket_xy_m=[.9, -.4], source_row_id="original-validation-05", trials=[
                dict(id="sweep-case-01", native_release=True, replay_id=self.replay_id,
                     replay_sha256=server.sha(self.replay_path)),
                dict(id="reject-01", native_release=False)])])
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.value.pop("fingerprint", None)
        self.value["fingerprint"] = server.digest(self.value)
        (self.root / "report.json").write_text(json.dumps(self.value))
        self.state.impact_sha256 = server.sha(self.root / "report.json")
        self.audit = dict(version="sircus-impact-pose-sweep-independent-audit-v1",
            synthetic_fixture=False, status="passed", checks=100, failures=[], source=self.source,
            report_sha256=self.state.impact_sha256, plan_sha256=server.sha(self.root / "plan.json"),
            manifest_sha256=server.sha(self.root / "manifest.json"))
        self.write_audit()

    def write_audit(self):
        (self.root / "audit.json").write_text(json.dumps(self.audit))
        self.state.impact_audit_sha256 = server.sha(self.root / "audit.json")

    def test_serves_only_source_bound_report_and_replay(self):
        value = self.state.impact_report()
        self.assertEqual(value["fingerprint"], self.value["fingerprint"])
        self.assertEqual(value["audit"]["status"], "passed")
        self.assertEqual(self.state.impact_replay(self.replay_id), self.replay["result"])

    def test_changed_bytes_fail_closed(self):
        for name in ("report.json", "plan.json", "manifest.json", "audit.json"):
            path = self.root / name
            before = path.read_bytes()
            path.write_bytes(before + b" ")
            with self.subTest(name=name), self.assertRaises(RequestError):
                self.state.impact_report()
            path.write_bytes(before)

    def test_incomplete_synthetic_or_changed_contract_refused(self):
        original = copy.deepcopy(self.value)
        for change in (dict(status="running"), dict(synthetic_fixture=True), dict(version="old")):
            self.value = dict(copy.deepcopy(original), **change); self.write()
            with self.subTest(change=change), self.assertRaises(RequestError):
                self.state.impact_report()
        for key in self.source:
            self.value = copy.deepcopy(original); self.value["source"][key] = "f" * 64; self.write()
            with self.subTest(key=key), self.assertRaises(RequestError):
                self.state.impact_report()

    def test_original_diagnostic_failure_propagates(self):
        self.state.diagnostic_report.side_effect = RequestError("Source changed", 409)
        with self.assertRaises(RequestError): self.state.impact_report()

    def test_failed_or_mismatched_audit_refused(self):
        original = copy.deepcopy(self.audit)
        for change in (dict(status="failed"), dict(checks=0), dict(checks=True), dict(failures=["bad"]),
                       dict(synthetic_fixture=True), dict(version="old"),
                       dict(report_sha256="f" * 64), dict(plan_sha256="f" * 64),
                       dict(manifest_sha256="f" * 64), dict(source={})):
            self.audit = dict(copy.deepcopy(original), **change); self.write_audit()
            with self.subTest(change=change), self.assertRaises(RequestError): self.state.impact_report()

    def test_unknown_rejected_or_path_traversal_replays_refused(self):
        for replay_id in ("unknown-123456789", "../../report", "reject-01", self.replay_id + "/other"):
            with self.subTest(replay_id=replay_id), self.assertRaises(RequestError): self.state.impact_replay(replay_id)
        self.value["layouts"][0]["trials"][1]["replay_id"] = "reject-12345678"
        self.write()
        with self.assertRaises(RequestError): self.state.impact_report()

    def test_replay_identity_and_role_must_match_hash_bound_metadata(self):
        for change in (dict(role="test"), dict(source_row_id="other"), dict(source={}), dict(result={})):
            original = self.replay_path.read_text()
            self.replay_path.write_text(json.dumps(dict(self.replay, **change)))
            self.value["layouts"][0]["trials"][0]["replay_sha256"] = server.sha(self.replay_path)
            self.write()
            with self.subTest(change=change), self.assertRaises(RequestError): self.state.impact_replay(self.replay_id)
            self.replay_path.write_text(original)

    def test_duplicate_native_replay_refused(self):
        rows = self.value["layouts"][0]["trials"]
        rows.append(copy.deepcopy(rows[0])); self.write()
        with self.assertRaises(RequestError): self.state.impact_report()

    def test_all_post_commands_are_rejected_without_state_calls(self):
        state = Mock()
        handler = object.__new__(server.impact_handler(state)); handler._json = Mock()
        handler.do_POST()
        self.assertEqual(handler._json.call_args.args[0], 405)
        self.assertEqual(state.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
