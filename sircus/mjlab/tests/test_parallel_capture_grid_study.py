"""Exercise mixed native packet scheduling without starting a simulator."""
from collections import Counter, deque
from contextlib import ExitStack
import importlib.util
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "Full study imports require PyTorch")
class ParallelCaptureGridTests(unittest.TestCase):
    def test_round_robin_skips_empty_queues_and_carries_starting_cell(self):
        from sircus.mjlab.parallel_capture_grid_study import take_packet

        pending = {"a": deque(("a1", "a2", "a3")), "b": deque(), "c": deque(("c1", "c2", "c3"))}
        order = deque(pending)
        self.assertEqual(take_packet(pending, order, 3), ["a1", "c1", "a2"])
        self.assertEqual(take_packet(pending, order, 3), ["c2", "a3", "c3"])
        self.assertEqual(take_packet(pending, order, 3), [])

    def test_mixed_packets_fill_all_quotas_even_with_a_single_layout_tail(self):
        from sircus.mjlab import cell1050_heading_native, parallel_capture_grid_plan
        from sircus.mjlab import parallel_capture_grid_study as study
        from sircus.mjlab import table_correction_native

        quota = 65
        cells = {
            f"cell-{index:02d}-00": {
                "bucket_xy_m": [index / 10, 0.0],
                "prior_count": 1,
                "excluded_geometry": {study.geometry_key({"values": [0.2, 0.0, 0.4, 15.0, 0.15]})},
            }
            for index in range(13)
        }
        slow_cell = next(reversed(cells))
        search_packets = []
        preparation_methods = []
        control_phases = []
        publications = 0

        def candidates(cell, *, seed):
            for index in count(1):
                yield {
                    "bucket_xy_m": cell["bucket_xy_m"],
                    "values": [0.2, 0.0, 0.4, 15.0, 0.15 + index * 0.0001],
                }

        def atomic(path, value):
            nonlocal publications
            if path.name == "report.json":
                publications += 1
                if publications > 100:
                    raise AssertionError("Scheduler stalled with a partially filled packet")
            original_atomic(path, value)

        class Engine:
            def __init__(self, *args, **kwargs):
                env = SimpleNamespace(effective_options={}, compiled_geometry={})
                self.adapter = SimpleNamespace(env=env)
                self.engine = self

            def prepare(self, proposals, *, one_per_setting):
                admitted, records = [], []
                for row in proposals:
                    cell = row["context_id"]
                    position = int(row["id"].rsplit("-", 1)[-1])
                    requests = []
                    if cell == slow_cell:
                        preparation_methods.append(one_per_setting)
                    if cell != slow_cell or position >= 120:
                        branches = 1 if one_per_setting else 2
                        requests = [
                            dict(row, id=f"{row['id']}-b{slot}", proposal_id=row["id"])
                            for slot in range(branches)
                        ]
                    admitted.extend(requests)
                    records.append({
                        "proposal_id": row["id"],
                        "proposal_sha256": study.digest(row),
                        "admitted_request_ids": [request["id"] for request in requests],
                    })
                return admitted, records

            def evaluate(self, requests, phase):
                if phase == "search":
                    search_packets.append(requests)
                else:
                    control_phases.append(phase)
                return [dict(row, native_release=True) for row in requests], {}

        def annotate(rows, directory):
            for row in rows:
                row["experimental_route"] = {"route_class": "standard_capture"}

        with TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            parent = root / "parent"
            (parent / "native-yaw").mkdir(parents=True)
            parent_files = (
                "calibration.json", "verification.json", "setup-home.json", "prior.json",
                "old-prior.json", "control-replays.json", "report.json", "evidence-manifest.json",
                "native-yaw/effective-options.json", "native-yaw/native-compiled-geometry.json",
            )
            for name in parent_files:
                (parent / name).write_text("{}")
            parent_audit = root / "audit.json"
            parent_audit.write_text("{}")
            binding = {
                "report": study.sha(parent / "report.json"),
                "audit": study.sha(parent_audit),
                "manifest": study.sha(parent / "evidence-manifest.json"),
            }
            spec = dict(scene={}, identity=dict(scene_sha256=study.digest({}), asset_sha256={}), packages={})
            args = SimpleNamespace(
                output=root / "output", parent_study=parent, parent_audit=parent_audit,
                reference=root / "reference", assets=root / "assets", history_root=root,
                target_per_cell=quota, prepare_batch=8, seed=1,
            )
            original_atomic = study.atomic
            patches = (
                patch.object(study, "parent_inputs", return_value=(spec, binding, {})),
                patch.object(study, "load_runtime_reference", return_value=dict(scene={}, asset_hashes={})),
                patch.object(parallel_capture_grid_plan, "load_grid_plan", return_value=(cells, {})),
                patch.object(parallel_capture_grid_plan, "cell_candidates", side_effect=candidates),
                patch.object(cell1050_heading_native, "HeadingEngine", Engine),
                patch.object(table_correction_native, "MatchedEngine", Engine),
                patch.object(study, "source_manifest", return_value={}),
                patch.object(study, "control_requests", side_effect=lambda _, phase: [{"id": phase}]),
                patch.object(study, "control_gate", return_value={"status": "passed"}),
                patch.object(study, "mirror_control_proofs"),
                patch.object(study, "share_native_environment"),
                patch.object(study, "foreign_processes", return_value=[]),
                patch.object(study.shutil, "disk_usage", return_value=SimpleNamespace(free=50 * 1024**3)),
                patch.object(study, "annotate", side_effect=annotate),
                patch.object(study, "atomic", side_effect=atomic),
                patch("builtins.print"),
            )
            for item in patches:
                stack.enter_context(item)
            study.run(args)
            report = study.read(args.output / "report.json")

        self.assertEqual(report["status"], "completed")
        self.assertEqual(control_phases, ["controls_pre", "controls_post"])
        self.assertEqual(report["search_releases"], 13 * quota)
        self.assertGreaterEqual(report["max_cells_in_packet"], 12)
        self.assertGreater(report["mixed_packets"], 0)
        self.assertIn(False, preparation_methods)
        self.assertEqual(report["cells"][slow_cell]["preparation_declines"], 120)
        self.assertEqual(
            Counter(row["context_id"] for packet in search_packets for row in packet),
            dict.fromkeys(cells, quota),
        )
        for packet in search_packets:
            self.assertLessEqual(len(packet), study.NATIVE_BATCH)
            self.assertTrue(all(row["id"].endswith("-b0") for row in packet))
        self.assertEqual({row["context_id"] for row in search_packets[-1]}, {slow_cell})
        self.assertEqual(len(search_packets[-2]), study.NATIVE_BATCH)
        self.assertEqual({row["context_id"] for row in search_packets[-2]}, {slow_cell})
