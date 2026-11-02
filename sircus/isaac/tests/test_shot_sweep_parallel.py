"""Worker lineage, assignment and task guards; no subprocess/GPU startup."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from expanded_farm import make_spec
from expanded_pose_menu import canonical_hash, make_manifest
from shot_sweep_parallel import assignments, merge_worker_manifests


class ParallelSweepTests(unittest.TestCase):
    def fixture(self, directory):
        spec = make_spec(json.loads((SOURCE / "scene.json").read_text()), 10)
        groups = assignments(4)
        paths = []
        for worker, group in enumerate(groups):
            menus = [[dict(id=f"worker-{worker}-cell-{i}")] for i in range(35)]
            fits = [[dict(Arm10={}, Arm14={})] for _ in range(35)]
            value = make_manifest(
                spec,
                menus,
                fits=fits,
                checks=[[] for _ in range(35)],
                seed_manifest_sha256="seed-hash",
                proposal_protocol=dict(version="fixture", workspace="table"),
                summary=dict(requested_cells=group),
            )
            path = Path(directory) / f"worker-{worker}.json"
            path.write_text(json.dumps(value))
            paths.append(path)
        return spec, groups, paths

    def test_partition_is_exactly_original_cells_and_merge_uses_assigned_worker(self):
        for n in range(1, 5):
            self.assertEqual(
                sorted(i for group in assignments(n) for i in group), list(range(35))
            )
        for n in (0, 5, 1.5, True):
            with self.assertRaises(ValueError):
                assignments(n)
        with tempfile.TemporaryDirectory() as directory:
            spec, groups, paths = self.fixture(directory)
            value = merge_worker_manifests(spec, paths, groups)
            for i, group in enumerate(value["menus"]):
                self.assertEqual(group[0]["id"], f"worker-{i%4}-cell-{i}")
            self.assertEqual(value["qualification"], "offline_endpoint_ik")
            self.assertEqual(value["spec"], spec)
            self.assertEqual(value["parallel_protocol"]["threads_per_worker"], 1)
            self.assertEqual(len(value["parallel_protocol"]["worker_artifacts"]), 4)
            self.assertEqual(
                value["fingerprint"],
                canonical_hash({k: v for k, v in value.items() if k != "fingerprint"}),
            )

    def test_rejects_changed_assignments_worker_sources_and_missing_fit(self):
        for mutation in ("assignment", "seed", "spec", "checksum", "fits"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as directory:
                spec, groups, paths = self.fixture(directory)
                value = json.loads(paths[1].read_text())
                if mutation == "assignment":
                    value["summary"]["requested_cells"] = groups[0]
                if mutation == "seed":
                    value["seed_manifest_sha256"] = "other-seed"
                if mutation == "spec":
                    value["spec"]["cells"][0]["bucket_xy_m"][0] += 0.01
                if mutation == "fits":
                    value["fits"][groups[1][0]] = []
                if mutation != "checksum":
                    value["fingerprint"] = canonical_hash(
                        {k: v for k, v in value.items() if k != "fingerprint"}
                    )
                else:
                    value["summary"]["requested_cells"] = [99]
                paths[1].write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    merge_worker_manifests(spec, paths, groups)
        with tempfile.TemporaryDirectory() as directory:
            spec, groups, paths = self.fixture(directory)
            bad = copy.deepcopy(groups)
            bad[1][0] = bad[0][0]
            with self.assertRaises(ValueError):
                merge_worker_manifests(spec, paths, bad)
            with self.assertRaises(ValueError):
                merge_worker_manifests(spec, paths[:-1], groups)


if __name__ == "__main__":
    unittest.main()
