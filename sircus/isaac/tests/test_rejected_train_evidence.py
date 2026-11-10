"""Bound synthetic receipts; no original geometry, policy or simulator work."""

import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_train_evidence as original
import rejected_train_evidence as collector
from test_candidate_train_evidence import fixture, write


class RejectedCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expected = fixture(self.root)
        self.run = self.root/"runs/ppo-seed20261122"

    def collect(self):
        return collector.collect(self.root, **self.expected)

    def rebind_packet(self, mutate, relative="trials/train-00000.json.gz"):
        path = self.run/relative
        packet = json.loads(gzip.decompress(path.read_bytes()))
        mutate(packet)
        new_sha = write(path, packet)
        progress = json.loads((self.run/"progress.json").read_text())
        progress["source_trial_sha256"][relative] = new_sha
        self.expected["progress_sha"] = write(self.run/"progress.json", progress)
        audit_path = Path(str(self.run)+".audit.json")
        audit = json.loads(audit_path.read_text())
        audit["source_trial_sha256"][relative] = new_sha
        audit["source_progress_sha256"] = self.expected["progress_sha"]
        self.expected["audit_sha"] = write(audit_path, audit)

    def test_complete_rejections_without_inactive_capture_or_released_miss(self):
        result = self.collect()
        self.assertEqual(result["schema"], collector.VERSION)
        self.assertEqual(result["rejected_count"], result["training_totals"]["rejected"])
        self.assertEqual(len(result["rejections"]), result["rejected_count"])
        self.assertEqual(len({r["rejection_key"] for r in result["rejections"]}), result["rejected_count"])
        self.assertEqual(len(result["train_trial_sha256"]), 289)
        self.assertEqual(result["training_totals"]["attempts"], 10000)
        self.assertTrue(all(r["cell"]["split"] == "train" and r["original_reason"] == "outer_link_length_bound"
                            for r in result["rejections"]))
        self.assertNotIn("inactive_split", result["rejection_reasons"])
        self.assertEqual(set(result["rejection_stages"]), {"geometry_or_path"})
        self.assertNotIn("NEVER EXPORTED", json.dumps(result))
        self.assertNotIn("successes", result)
        self.assertEqual(result["new_native_shots"], 0)
        self.assertFalse(result["release_authorized"])
        unsigned = dict(result); fingerprint = unsigned.pop("fingerprint")
        self.assertEqual(original.canonical_hash(unsigned), fingerprint)

    def test_measured_start_is_distinct_from_saved_endpoint_witness(self):
        result = self.collect()
        row = next(r for r in result["rejections"] if r["slot"] == 10)
        self.assertEqual(row["recorded_starts_rad"]["Arm10"], [.020]*6)
        self.assertEqual(row["qualification"]["ik"]["Arm10"]["q"], [.010]*6)
        self.assertFalse(result["historical_endpoint_witnesses_for_solver"])
        self.assertIs(row["home_path_eligible"], True)

    def test_original_full_verifier_runs_and_no_heldout_packet_is_read(self):
        calls, real_read = [], original.read
        def read(path, expected):
            self.assertNotIn("eval-", str(path))
            calls.append(Path(path))
            return real_read(path, expected)
        with patch.object(original, "read", side_effect=read), patch.object(
                original, "collect", wraps=original.collect) as verified:
            result = self.collect()
        verified.assert_called_once()
        self.assertEqual(sum(p.name == "train-00288.json.gz" for p in calls), 2)
        self.assertEqual(result["evaluated_trial_files_read"], 0)
        self.assertEqual(result["optimizer_updates"], 0)

    def test_native_rejection_stage_does_not_claim_geometric_failure(self):
        self.assertEqual(collector.NATIVE_REASONS,
            {"endpoints_not_settled", "setup_motion_not_quiet", "setup_transit_incomplete"})
        # A short controlled verified result permits a native-reason test while
        # the full-budget verifier itself is tested without substitution above.
        verified = original.collect(self.root, **self.expected)
        packet = json.loads(gzip.decompress((self.run/"trials/train-00000.json.gz").read_bytes()))
        slot = 10
        packet["enabled"] = [i == slot for i in range(35)]
        packet["rows"][slot]["invalid"] = "setup_transit_incomplete"
        verified["train_trial_sha256"] = {"trials/train-00000.json.gz": "f"*64}
        verified["training_totals"]["rejected"] = 1
        verified["training_totals"]["failures"] = {"setup_transit_incomplete": 1}
        with patch.object(original, "collect", return_value=verified), patch.object(original, "read", return_value=packet):
            result = self.collect()
        row = result["rejections"][0]
        self.assertEqual(row["rejection_stage"], "native_motion_or_settling")
        self.assertTrue(row["qualification"]["accepted"])

    def test_original_packet_corruption_and_changed_model_fail_closed(self):
        path = self.run/"trials/train-00000.json.gz"
        old = path.read_bytes(); path.write_bytes(old+b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.collect()
        path.write_bytes(old)
        (self.run/"checkpoints/milestone-10000.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
            self.collect()

    def test_rejected_row_action_mismatch_is_not_exported(self):
        self.rebind_packet(lambda p: p["rows"][10].update(gap_m=.4))
        with self.assertRaisesRegex(ValueError, "row/action settings"):
            self.collect()

    def test_unexpected_action_metadata_is_not_exported_as_geometry(self):
        self.rebind_packet(lambda p: p["actions"][10].update(reward="not a geometry input"))
        with self.assertRaisesRegex(ValueError, "Missing rejected continuous action"):
            self.collect()

    def test_malformed_measured_home_flag_fails(self):
        self.rebind_packet(lambda p: p["home"]["path_eligible"].__setitem__(10, 1))
        with self.assertRaisesRegex(ValueError, "home path eligibility"):
            self.collect()

    def test_bound_rejection_set_cannot_be_dropped_on_second_read(self):
        verified = original.collect(self.root, **self.expected)
        real_read = original.read
        def read(path, expected):
            result = real_read(path, expected)
            if Path(path).name == "train-00000.json.gz":
                result["rows"][10]["invalid"] = None
            return result
        with patch.object(original, "collect", return_value=verified), patch.object(original, "read", side_effect=read):
            with self.assertRaisesRegex(ValueError, "rejection population"):
                self.collect()

    def test_collector_cli_rejects_output_inside_frozen_input(self):
        args = [sys.executable, str(SOURCE/"rejected_train_evidence.py"), "--benchmark", str(self.root)]
        for key, value in self.expected.items():
            args += ["--"+key.replace("_", "-"), value]
        target = self.root/"forbidden.json"
        args += ["--output", str(target)]
        result = subprocess.run(args, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("separate from frozen", result.stderr)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
