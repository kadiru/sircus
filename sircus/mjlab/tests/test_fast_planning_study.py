"""Fresh placements and unchanged lineage for the planning-speed comparison."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from sircus.mjlab.benchmark_verify import sha
    from sircus.mjlab.fast_planning_study import (
        COPIED_INPUTS, audited_budget, fresh_layouts, processing_order,
    )


@unittest.skipUnless(HAS_TORCH, "Existing study runtime imports require PyTorch")
class FastStudyTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        archive, neural = root / "budget", root / "neural"
        archive.mkdir()
        neural.mkdir()
        for name in (*COPIED_INPUTS, "layouts.json", "contract.json"):
            (archive / name).write_text('{}')
            (neural / name).write_text('{}')
        (archive / "report.json").write_text(json.dumps(dict(status="completed", comparison={})))
        manifest = {p.name: sha(p) for p in archive.glob('*.json')}
        (archive / "evidence-manifest.json").write_text(json.dumps(manifest))
        receipt = root / "audit.json"
        receipt.write_text(json.dumps(dict(status="passed", comparison={},
            report_sha256=sha(archive / "report.json"),
            evidence_manifest_sha256=sha(archive / "evidence-manifest.json"))))
        return SimpleNamespace(budget_study=archive, budget_audit=receipt, neural_study=neural)

    def test_prior_inputs_and_receipt_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(tmp)
            value = audited_budget(args)
            self.assertEqual(value["layouts_sha256"], sha(args.budget_study / "layouts.json"))
            (args.budget_study / "layouts.json").write_text('{"modified":true}')
            with self.assertRaisesRegex(ValueError, "budget input changed"):
                audited_budget(args)

    def test_cannot_silently_replace_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(tmp)
            (args.neural_study / "neural-checkpoint.json").write_text('{"replacement":true}')
            with self.assertRaisesRegex(ValueError, "same frozen inputs"):
                audited_budget(args)

    def test_failed_or_changed_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(tmp)
            receipt = json.loads(args.budget_audit.read_text())
            for changes in (dict(status="failed"), dict(report_sha256="bad"), dict(comparison={"changed":1})):
                args.budget_audit.write_text(json.dumps(dict(receipt, **changes)))
                with self.assertRaisesRegex(ValueError, "passed planning-budget"):
                    audited_budget(args)

    def test_one_fresh_batch_retains_exclusion_and_strata(self):
        with patch("sircus.mjlab.fast_planning_study.sample_splits", return_value={
            "test": [dict(bucket_xy_m=[.01*i,.1], stratum="edge") for i in range(50)]
        }) as sample:
            rows = fresh_layouts([[8,9]], {}, [], seed=11)
        self.assertEqual(sample.call_args.args[0], [(8,9)])
        self.assertEqual(len(rows),50)
        self.assertEqual(rows[49]["id"], "fast-planning-test-049")
        self.assertTrue(all(r["stratum"] == "edge" and r["split"] == "test" for r in rows))
        self.assertTrue(all(r["context_id"] == r["id"] for r in rows))

    def test_duplicate_coordinates_cannot_warm_cache(self):
        with patch("sircus.mjlab.fast_planning_study.sample_splits", return_value={
            "test": [dict(bucket_xy_m=[.1,.1]) for _ in range(50)]
        }):
            with self.assertRaisesRegex(ValueError,"50 distinct"):
                fresh_layouts([],{},[])

    def test_processing_order_reproducible_and_complete(self):
        rows = [dict(id=f"case-{i}") for i in range(50)]
        order = processing_order(rows)
        self.assertEqual(order, processing_order(rows))
        self.assertCountEqual(order,[r["id"] for r in rows])
        self.assertNotEqual(order,[r["id"] for r in rows])
        self.assertNotEqual(order,processing_order(rows,seed=999))


if __name__ == "__main__":
    unittest.main()
