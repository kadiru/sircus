"""Synthetic posthoc diagnostics and corruption rejection; not native evidence."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
import feasibility_guided_diagnostics as diagnostic
from continuous_feasibility import decode_action
from continuous_pilot import make_study
from test_feasibility_guided_audit import fixture


class SupportExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root/"trials").mkdir()
        support, _, spec = fixture()
        self.support_path = self.root/"support.json"
        self.write("support.json", support)
        study = make_study(dict(spec=spec, source=dict(camera_sha256="a"*64)), 4)
        self.write("contract.json", spec)
        self.write("study.json", study)
        (self.root/"policy-final.pt").write_bytes(b"Synthetic model fixture; not a policy artifact")
        checkpoint = dict(file="policy-final.pt", sha256=diagnostic.digest(self.root/"policy-final.pt"))
        summaries = []
        for repeat in range(4):
            rows = []
            for i, cell in enumerate(study["cells"]):
                values = [0., .4, .50 if i == 2 else .35, .24, .5061454830783556]
                action = decode_action(cell, values, f"frozen-{i}")
                rows.append(dict(action, cell_id=cell["id"], split=cell["split"], route_success=i in (1, 2)))
            name = f"trials/eval-{repeat:04d}.json"
            trial = dict(mode="eval", checkpoint=checkpoint, deterministic=True, rows=rows)
            self.write(name, trial)
            summaries.append(dict(file=name, sha256=diagnostic.digest(self.root/name)))
        self.result = dict(schema=study["schema"], status="completed", study_fingerprint=study["fingerprint"],
                           geometry_fingerprint=spec["fingerprint"], final=checkpoint, assessments=summaries)
        self.write("result.json", self.result)
        final_hashes = {s["file"]: s["sha256"] for s in summaries[2:]}
        self.native = dict(schema="sircus-continuous-native-outcome-audit-v1", status="passed",
                           source_result_sha256=diagnostic.digest(self.root/"result.json"), source_trial_sha256=final_hashes)
        self.policy_audit = dict(schema="sircus-continuous-pilot-checkpoint-audit-v1", status="passed",
                               source_result_sha256=diagnostic.digest(self.root/"result.json"), trials=final_hashes)
        self.write("native-audit.json", self.native)
        self.write("checkpoint-audit.json", self.policy_audit)
        self.result_sha = diagnostic.digest(self.root/"result.json")
        self.support_sha = diagnostic.digest(self.support_path)

    def write(self, name, value):
        (self.root/name).write_text(json.dumps(value, allow_nan=False))

    def check(self):
        return diagnostic.support_exclusion(self.root, self.support_path,
            expected_result_sha256=self.result_sha, expected_support_sha256=self.support_sha)

    def test_successful_settings_retained_and_excluded_have_exact_per_axis_bounds(self):
        result = self.check()
        self.assertEqual(result["old_successful_sites"], 2)
        self.assertEqual(result["old_successful_settings_retained"], 1)
        self.assertEqual(result["old_successful_settings_excluded"], 1)
        self.assertEqual(result["excluded_repeated_successes"], 2)
        self.assertEqual(result["excluded_site_counts_by_axis"], {"height_m": 1})
        self.assertEqual(result["records"][1]["excluded_axes"][0]["side"], "above_maximum")
        self.assertIn("not input", result["scope"])

    def test_support_or_original_result_replacement_is_rejected(self):
        self.write("support.json", dict(changed="not the frozen support"))
        with self.assertRaisesRegex(ValueError, "Guided support checksum changed"):
            self.check()

    def test_trial_must_match_both_independent_audits(self):
        self.native["source_trial_sha256"] = dict.fromkeys(self.native["source_trial_sha256"], "f"*64)
        self.write("native-audit.json", self.native)
        with self.assertRaisesRegex(ValueError, "differs from independent audits"):
            self.check()

    def test_different_native_audit_source_result_is_rejected(self):
        self.native["source_result_sha256"] = "f"*64
        self.write("native-audit.json", self.native)
        with self.assertRaisesRegex(ValueError, "audit binding changed"):
            self.check()

    def test_final_checkpoint_file_change_is_rejected(self):
        (self.root/"policy-final.pt").write_bytes(b"Different checkpoint")
        with self.assertRaisesRegex(ValueError, "final checkpoint changed"):
            self.check()


if __name__ == "__main__":
    unittest.main()
