import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_report import LABELS, finalize


class PilotReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.write(
            "splits.json",
            {
                "train": [{"id": "train"}],
                "validation": [{"id": "validation"}],
                "test": [{"id": str(i)} for i in range(36)],
            },
        )
        self.write("contract.json", {})
        (self.run / "selected.pt").write_bytes(b"fixture")
        self.write(
            "summary.json",
            {
                "phase": "completed",
                "episodes": 3,
                "budget": 3,
                "selected_sha256": hashlib.sha256(b"fixture").hexdigest(),
                "evaluation": {
                    k: {"captures": 36 if k == "pilot_selected" else 0} for k in LABELS
                },
            },
        )
        for k in LABELS:
            self.write(
                f"test-{k}.json",
                [
                    {"scene_id": str(i), "success": k == "pilot_selected"}
                    for i in range(36)
                ],
            )
        (self.run / "episodes.jsonl").write_text(
            "\n".join(json.dumps({"difficulty": k}) for k in ["near", "medium", "far"])
        )

    def write(self, name, data):
        (self.run / name).write_text(json.dumps(data))

    def test_paired_counts_and_hashes(self):
        r = finalize(self.run)
        self.assertEqual(r["paired_comparison"]["ppo_100"]["pilot_only_captures"], 36)
        self.assertEqual(
            r["verified_train_balance"], {"near": 1, "medium": 1, "far": 1}
        )
        self.assertEqual(r["selected_sha256"], r["artifact_sha256"]["selected.pt"])

    def test_unmatched_layout_rejected(self):
        self.write(
            "test-ppo_100.json",
            [{"scene_id": "wrong", "success": False} for _ in range(36)],
        )
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            finalize(self.run)

    def test_truncated_training_rejected(self):
        (self.run / "episodes.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "Truncated"):
            finalize(self.run)

    def test_checkpoint_mismatch_rejected(self):
        (self.run / "selected.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            finalize(self.run)
