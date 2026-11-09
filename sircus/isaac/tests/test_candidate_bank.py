"""Retention reporting and write boundaries; synthetic scalar cases only."""

import copy
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_bank as bank


class CandidateBankTests(unittest.TestCase):
    def setUp(self):
        self.evidence = dict(successful_shots=2, clean_successes=1, successes=[
            dict(success_key="old-a", cell=dict(id="train-000"), clean=True),
            dict(success_key="old-b", cell=dict(id="train-001"), clean=False)])
        self.result = dict(history_survival=[
            dict(case_id="old-a", candidate_id="a", cell_id="train-000", generator_retained=True,
                 accepted=True, status="admitted", reason=None),
            dict(case_id="old-b", candidate_id="b", cell_id="train-001", generator_retained=True,
                 accepted=False, status="unresolved", reason="endpoint_ik_unresolved_Arm14")])
        self.audit = dict(successful_train_shots=2, clean_train_shots=1,
                          successful_train_survive=1, clean_train_survive=1)

    def test_retention_preserves_success_denominator_and_clean_subset(self):
        actual = bank.retention_summary(self.evidence, self.result, self.audit)
        self.assertEqual(actual["historical_successes"], dict(requested=2, included_exactly=2,
            qualified=1, excluded=1, reasons=dict(endpoint_ik_unresolved_Arm14=1)))
        self.assertEqual(actual["clean_historical_successes"], dict(requested=1, qualified=1, excluded=0))
        self.assertEqual(actual["historical_exclusions"][0]["case_id"], "old-b")
        self.assertFalse(actual["historical_exclusions"][0]["clean"])

    def test_independent_clean_accounting_cannot_be_omitted_or_changed(self):
        for key in ("clean_train_shots", "clean_train_survive", "successful_train_survive"):
            audit = dict(self.audit); audit[key] = 0
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "accounting"):
                bank.retention_summary(self.evidence, self.result, audit)

    def test_missing_duplicated_or_changed_historical_case_fails_closed(self):
        cases = []
        result = copy.deepcopy(self.result); result["history_survival"].pop(); cases.append(result)
        result = copy.deepcopy(self.result); result["history_survival"][1] = result["history_survival"][0]; cases.append(result)
        result = copy.deepcopy(self.result); result["history_survival"][0]["generator_retained"] = False; cases.append(result)
        result = copy.deepcopy(self.result); result["history_survival"][0]["cell_id"] = "test-001"; cases.append(result)
        result = copy.deepcopy(self.result); result["history_survival"][1]["reason"] = None; cases.append(result)
        for result in cases:
            with self.subTest(result=result), self.assertRaises(ValueError):
                bank.retention_summary(self.evidence, result, self.audit)

    def test_csv_includes_each_success_and_its_explicit_exclusion_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"retention.csv"
            bank.export_retention(path, self.evidence, self.result)
            with path.open() as file:
                rows = list(csv.DictReader(file))
        self.assertEqual([r["case_id"] for r in rows], ["old-a", "old-b"])
        self.assertEqual([r["qualified"] for r in rows], ["1", "0"])
        self.assertEqual(rows[0]["clean"], "1")
        self.assertEqual(rows[0]["exclusion_reason"], "")
        self.assertEqual(rows[1]["exclusion_reason"], "endpoint_ik_unresolved_Arm14")

    def test_existing_output_or_frozen_descendant_is_rejected_before_read_or_write(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bank, "load_inputs") as load:
            root = Path(directory); frozen = root/"frozen"; frozen.mkdir()
            for output in (frozen, frozen/"new-artifact"):
                with self.assertRaises(ValueError):
                    bank.build(None, None, frozen, None, None, output)
            load.assert_not_called()
            self.assertFalse((frozen/"new-artifact").exists())

    def test_import_starts_no_training_or_gpu_runtime(self):
        script = ("import sys; sys.path.insert(0,%r); import candidate_bank; "
                  "assert 'torch' not in sys.modules; "
                  "assert not any(n.startswith(('isaacsim','isaaclab')) for n in sys.modules)") % str(SOURCE)
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
