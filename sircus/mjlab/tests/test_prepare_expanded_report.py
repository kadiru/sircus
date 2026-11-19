"""Display aggregation never changes experiment evidence or solver deployment."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sircus.mjlab.prepare_expanded_report import ACTIVE_POLICY, prepare, watch


class ExpandedReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base.json"
        self.search = self.root / "search"
        self.train = self.root / "training"
        self.output = self.root / "derived.json"
        self.base_value = dict(
            status="completed",
            frozen_evaluation={"captures": 43, "trials": 100},
            bank_evaluation={
                "status": "completed",
                "comparison": {"bank_selected": {"captures": 63}},
            },
        )
        self.base.write_text(json.dumps(self.base_value))

    def tearDown(self):
        self.temp.cleanup()

    def write(self, directory, report, contract=None):
        directory.mkdir(exist_ok=True)
        (directory / "report.json").write_text(json.dumps(report))
        if contract is not None:
            (directory / "contract.json").write_text(json.dumps(contract))

    def prepare(self):
        return prepare(self.base, self.search, self.train, self.output)

    def test_missing_reports_preserve_original_evidence_and_route(self):
        before = self.base.read_bytes()
        result = self.prepare()
        self.assertEqual(result["expansion_evaluation"]["status"], "not_started")
        self.assertIsNone(result["expansion_evaluation"]["training"])
        self.assertEqual(result["expansion_evaluation"]["active_policy"], ACTIVE_POLICY)
        self.assertEqual({k: result[k] for k in self.base_value}, self.base_value)
        self.assertEqual(self.base.read_bytes(), before)
        self.assertFalse((self.search / "report.json").exists())

    def test_compact_reports_keep_denominators_and_promotion_provenance(self):
        search = dict(
            status="completed",
            native_search_attempts=24,
            confirmation_attempts=10,
            test_rows=[{"trajectory": [1] * 100}],
            cells=[
                dict(
                    id="old-test-01",
                    bucket_xy_m=[0.3, -0.5],
                    search=dict(attempts=24, captures=1, invalid=2, rows=["raw"]),
                    promotion=dict(
                        source_trial_id="expand-shot-17",
                        source_phase="targeted_development",
                        development_phase="wide",
                        confirmation=dict(
                            attempts=10, captures=8, invalid=1, chunks=["raw"]
                        ),
                    ),
                )
            ],
        )
        training = dict(
            status="completed",
            train_episodes=4096,
            updates=32,
            test_rows=["raw"],
            geometry={
                "expanded-test": dict(
                    queries=100, eligible_candidates=75, chunks=["large"]
                )
            },
            frozen_evaluation=dict(
                attempts=300,
                captures=160,
                comparison={
                    "expanded_learned": dict(
                        attempts=300,
                        captures=160,
                        eligible_attempts=270,
                        conditional_capture_rate=160 / 270,
                        layouts=100,
                        repeats=3,
                        rows=["raw"],
                    )
                },
            ),
            learning_curve=[
                dict(
                    update=1,
                    attempts=128,
                    captures=12,
                    weight_change={"epochs": ["raw"]},
                )
            ],
        )
        contract = dict(
            train_layouts=[[1, 2]] * 175,
            validation_layouts=[[1, 2]] * 30,
            test_layouts=[[1, 2]] * 100,
            development_layouts=[[1, 2]] * 12,
            ppo_updates=32,
            test_repeats=3,
        )
        self.write(self.search, search)
        self.write(self.train, training, contract)
        source_bytes = [(p, p.read_bytes()) for p in self.root.glob("*/*.json")]
        result = self.prepare()["expansion_evaluation"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["training"]["study_plan"]["test_layouts"], 100)
        self.assertEqual(
            result["search"]["cells"][0]["promotion"]["confirmation"],
            dict(attempts=10, captures=8, invalid=1),
        )
        self.assertEqual(
            result["training"]["frozen_evaluation"]["comparison"]["expanded_learned"][
                "attempts"
            ],
            300,
        )
        compact = json.dumps(result)
        for raw in (
            '"test_rows"',
            '"chunks"',
            '"rows"',
            '"weight_change"',
            '"trajectory"',
        ):
            self.assertNotIn(raw, compact)
        self.assertEqual(result["active_policy"], ACTIVE_POLICY)
        for path, before in source_bytes:
            self.assertEqual(path.read_bytes(), before)

    def test_progress_and_queued_gpu_are_read_only(self):
        self.write(
            self.search,
            dict(
                status="completed",
                development_layouts=[dict(id="a", bucket_xy_m=[0.3, -0.5])],
            ),
        )
        self.assertEqual(
            self.prepare()["expansion_evaluation"]["status"], "awaiting_training"
        )
        self.write(
            self.train,
            dict(
                status="running",
                phase="waiting-for-4090",
                gpu_queue=dict(status="waiting_for_4090", wait_s=30, reason=[123]),
            ),
        )
        result = self.prepare()["expansion_evaluation"]
        self.assertEqual(result["status"], "training_running")
        self.assertEqual(result["training"]["gpu_queue"]["reason"], [123])
        (self.train / "report.json").write_text("incomplete JSON")
        self.assertIsNone(self.prepare()["expansion_evaluation"]["training"])

    def test_source_files_cannot_be_used_as_output(self):
        self.write(self.search, {})
        self.write(self.train, {}, {})
        for destination in (
            self.base,
            self.search / "report.json",
            self.train / "contract.json",
            self.train / "selected.pt",
            self.search / "new-display.json",
        ):
            with self.assertRaises(ValueError):
                prepare(self.base, self.search, self.train, destination)

    def test_watch_is_bounded_and_exits_when_both_studies_complete(self):
        initial = dict(expansion_evaluation={"status": "search_running"})
        completed = dict(expansion_evaluation={"status": "completed"})
        with mock.patch(
            "sircus.mjlab.prepare_expanded_report.prepare",
            side_effect=[initial, completed],
        ) as build, mock.patch(
            "sircus.mjlab.prepare_expanded_report.time.monotonic", side_effect=[0, 0]
        ), mock.patch(
            "sircus.mjlab.prepare_expanded_report.time.sleep"
        ) as sleep:
            self.assertEqual(
                watch(self.base, self.search, self.train, self.output, 10), completed
            )
            self.assertEqual(build.call_count, 2)
            sleep.assert_called_once_with(5)
        for duration in (-1, 1801, float("nan")):
            with self.assertRaises(ValueError):
                watch(self.base, self.search, self.train, self.output, duration)


if __name__ == "__main__":
    unittest.main()
