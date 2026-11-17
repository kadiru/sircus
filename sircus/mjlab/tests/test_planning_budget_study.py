"""The budget comparison cannot silently replace its model or reuse old targets."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab.benchmark_verify import sha
from sircus.mjlab.planning_budget_study import COPIED_INPUTS, audited_neural, fresh_layouts


class BudgetStudyTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        for name in (*COPIED_INPUTS, "layouts.json", "contract.json"):
            (root / name).write_text('{}')
        (root / "report.json").write_text(json.dumps(dict(status="completed", comparison={})))
        manifest = {p.name: sha(p) for p in root.glob('*.json')}
        (root / "evidence-manifest.json").write_text(json.dumps(manifest))
        receipt = root / "audit.json"
        receipt.write_text(json.dumps(dict(status="passed", comparison={},
            report_sha256=sha(root / "report.json"), evidence_manifest_sha256=sha(root / "evidence-manifest.json"))))
        return SimpleNamespace(neural_study=root, neural_audit=receipt)

    def test_unchanged_audited_model_is_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(tmp)
            self.assertEqual(audited_neural(args)["checkpoint_sha256"], sha(Path(tmp) / "neural-checkpoint.json"))

    def test_modified_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(tmp)
            (Path(tmp) / "neural-checkpoint.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "input changed"):
                audited_neural(args)

    def test_failed_or_modified_audit_cannot_authorize(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(tmp)
            receipt = json.loads(args.neural_audit.read_text())
            for changes in (dict(status="failed"), dict(report_sha256="bad"), dict(comparison={"modified":1})):
                args.neural_audit.write_text(json.dumps(dict(receipt, **changes)))
                with self.assertRaisesRegex(ValueError, "passed comparison"):
                    audited_neural(args)

    def test_second_batch_excludes_first_and_both_have_new_ids(self):
        seen = []
        def sample(excluded, geometry, grid, *, seed, include_train):
            seen.append(list(excluded))
            return {"test":[dict(bucket_xy_m=[float(seed),float(i)],stratum="interior") for i in range(50)]}
        with patch("sircus.mjlab.planning_budget_study.sample_splits", side_effect=sample):
            rows = fresh_layouts([[0.,0.]], {}, [], seed=11)
        self.assertEqual(len(rows),100)
        self.assertEqual(len({r['id'] for r in rows}),100)
        self.assertEqual(len(seen[1]),51)
        self.assertTrue(all(tuple(r['bucket_xy_m']) in seen[1] for r in rows[:50]))
        self.assertTrue(all(r['split']=='test' and r['context_id']==r['id'] for r in rows))


if __name__ == '__main__':
    unittest.main()
