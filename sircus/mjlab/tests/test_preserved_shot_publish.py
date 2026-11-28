import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import preserved_shot_publish as pub


class PublicationTest(unittest.TestCase):
    def fixture(self, root, cohort):
        repeats, count = (3, 9) if cohort == "development" else (1, 50)
        layouts, slots, plans, selections, rows = [], [], [], [], []
        for i in range(count):
            identity = (f"approach-comparison-test-{(1,16,28,32,33,42,43,44,46)[i]:02d}"
                        if cohort == "development" else f"case-{i:02d}")
            layouts.append(dict(id=identity, bucket_xy_m=[i*.01, -.3]))
            plans.append(dict(variant="current", site=dict(layout=layouts[-1], request_wall_s=2.)))
            selections.append(dict(context_id=identity, fallback_used=i%2 == 0,
                timing=dict(pair_wall_s=3., reserved_worker_window_s=4.)))
            for method in pub.METHODS:
                for repeat in range(repeats):
                    name = f"{identity}-{method}-{repeat}"
                    declined = method == "current" and i == 0
                    slots.append(dict(id=name, context_id=identity, method=method, request=None if declined else {}))
                    if not declined:
                        rows.append(dict(id=name, adapter_directory="native-current", native_release=True,
                            board_first_scalar_audit_passed=True, board_first_score=dict(
                                scoring_version="sircus-board-first-native-capture-v4",
                                invalid_physics=i == 1, route_success=i not in (1, 2))))
        rows.extend(dict(id=f"control-{i}", adapter_directory="native-controls") for i in range(18))
        for name, data in (("layouts.json",dict(layouts=layouts)),("choices.json",dict(slots=slots)),
                           ("plans.json",plans),("selection.json",selections)):
            (root/name).write_text(json.dumps(data))
        (root/"rows.jsonl").write_text("\n".join(map(json.dumps,rows))+"\n")
        report = dict(gpu_wait_s=0., gpu_lease_s=20.)
        contract = dict(attempts_per_method_per_position=repeats)
        provenance = {key:"a"*64 for key in ("audit_sha256","manifest_sha256","report_sha256","contract_sha256")}
        return root, report, dict(checks_count=100), contract, provenance

    @unittest.skipUnless(shutil.which("node"), "Node is required for publisher/display contract validation")
    def test_published_cohorts_are_accepted_by_ui_validators(self):
        module = (Path(__file__).resolve().parents[1]/"web"/"interactive-physics.mjs").as_uri()
        for cohort, validator in (("development","preservedDevelopmentReady"),("fresh","preservedFreshReady")):
            with self.subTest(cohort=cohort), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with patch.object(pub,"verified",return_value=self.fixture(root,cohort)):
                    result = pub.compose_cohort(root,root/"audit.json",cohort)
                script = f"import {{readFileSync}} from 'node:fs'; import {{{validator}}} from {json.dumps(module)}; if (!{validator}(JSON.parse(readFileSync(0,'utf8')))) process.exit(1);"
                subprocess.run(["node","--input-type=module","-e",script],input=json.dumps(result),text=True,check=True,capture_output=True)

    def test_two_declines_do_not_claim_a_current_shot_was_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root,"fresh")
            choices = json.loads((root/"choices.json").read_text())
            for slot in choices["slots"]:
                if slot["context_id"] == "case-00":
                    slot["request"] = None
            (root/"choices.json").write_text(json.dumps(choices))
            rows = [json.loads(line) for line in (root/"rows.jsonl").read_text().splitlines()]
            rows = [row for row in rows if row["id"] != "case-00-protected-0"]
            (root/"rows.jsonl").write_text("\n".join(map(json.dumps,rows))+"\n")
            with patch.object(pub,"verified",return_value=fixture):
                result = pub.compose_cohort(root,root/"audit.json","fresh")
            self.assertEqual(result["results"][0]["preserved"]["outcome"],"setup refused")
            self.assertIs(result["results"][0]["preserved"]["used_fallback"],False)

    def test_fresh_setup_declines_and_invalid_remain_in_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(pub,"verified",return_value=self.fixture(root,"fresh")):
                result = pub.compose_cohort(root,root/"audit.json","fresh")
            current, protected = result["methods"]
            self.assertEqual((current["attempts"], current["captures"], current["setup_rejections"],
                              current["released_misses"], current["invalid_physics"]),(50,47,1,1,1))
            self.assertEqual(protected["captures"],48)
            self.assertEqual(protected["request_s"]["mean"],3.)
            self.assertEqual(sum(r["preserved"]["used_fallback"] for r in result["results"]),25)

    def test_known_repeats_stay_separate_from_fresh_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(pub,"verified",return_value=self.fixture(root,"development")):
                result = pub.compose_cohort(root,root/"audit.json","development")
            self.assertEqual(len(result["positions"]),9)
            self.assertEqual(result["positions"][0]["current"]["releases"],0)
            self.assertEqual(result["positions"][0]["preserved"]["captures"],3)
            self.assertEqual(result["positions"][0]["preserved"]["used_fallback"],3)
            self.assertNotIn("results",result)

    def test_incomplete_native_inventory_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root,"fresh")
            lines = (root/"rows.jsonl").read_text().splitlines()
            (root/"rows.jsonl").write_text("\n".join(lines[1:])+"\n")
            with patch.object(pub,"verified",return_value=fixture), self.assertRaisesRegex(ValueError,"fully accounted"):
                pub.compose_cohort(root,root/"audit.json","fresh")

    def test_failed_or_stale_audit_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"report.json").write_text(json.dumps(dict(status="completed",native_completed=True)))
            (root/"contract.json").write_text("{}")
            audit = root/"audit.json"
            audit.write_text(json.dumps(dict(version="sircus-preserved-shot-independent-audit-v1",status="failed")))
            with self.assertRaisesRegex(ValueError,"Independent"):
                pub.verified(root,audit,"fresh")
            audit.write_text(json.dumps(dict(version="sircus-preserved-shot-independent-audit-v1",status="passed",frozen_before_test=True)))
            with self.assertRaisesRegex(ValueError,"no longer binds"):
                pub.verified(root,audit,"fresh")


if __name__ == "__main__":
    unittest.main()
