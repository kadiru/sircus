import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distance_route_audit import audit, low_external_rebound


class RouteAuditTests(unittest.TestCase):
    def test_external_low_reversal_differs_from_bucket_floor_and_high_board(self):
        path = [[0.67, -0.40, 0.08], [0.67, -0.42, 0.05], [0.67, -0.44, 0.10]]
        self.assertTrue(low_external_rebound(path, [0.67, -0.65], 0.0375, 0.1))
        self.assertTrue(
            low_external_rebound(
                [[x, y, z + 1] for x, y, z in path], [0.67, -0.65], 0.0375, 0.1, 1.0
            )
        )
        self.assertFalse(low_external_rebound(path, [0.67, -0.42], 0.0375, 0.1))
        self.assertFalse(
            low_external_rebound(
                [[x, y, z + 0.3] for x, y, z in path], [0.67, -0.65], 0.0375, 0.1
            )
        )
        self.assertFalse(low_external_rebound(path[:2], [0.67, -0.65], 0.0375, 0.1))

    def test_hash_ids_missing_paths_and_only_successful_capture_denominator(self):
        scene = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )
        report = {
            "contract": {
                "parent_quality": {
                    "parent": {"scene": scene, "board_xy_m": [0.67, -0.1]}
                }
            },
            "distances": [
                {
                    "distance_m": 0.55,
                    "confirmation": [
                        {"trial_id": str(i), "success": i != 2} for i in range(3)
                    ],
                }
            ],
        }
        rows = [
            {
                "stage": "confirmation",
                "trial_id": str(i),
                "distance_m": 0.55,
                "result": {"success": i != 2},
                "trajectory_m": [
                    [0.67, -0.40, 0.08],
                    [0.67, -0.42, 0.05],
                    [0.67, -0.44, 0.10],
                ]
                if i != 1
                else [],
            }
            for i in range(3)
        ]
        raw = "\n".join(json.dumps(r) for r in rows).encode()
        report["artifact_sha256"] = {"trials.jsonl": hashlib.sha256(raw).hexdigest()}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trials.jsonl"
            p.write_bytes(raw)
            result = audit(report, p)["distances"][0]
            self.assertEqual(result["confirmed_captures"], 2)
            self.assertEqual(result["low_external_rebound_flags"], 1)
            self.assertEqual(result["missing_capture_paths"], 1)
            self.assertEqual(result["flagged_trial_ids"], ["0"])
            p.write_bytes(raw + b" ")
            with self.assertRaisesRegex(AssertionError, "hash"):
                audit(report, p)
