"""End-to-end worker orchestration with explicitly synthetic physics fixtures.

No GPU, mesh model, hardware, NVIDIA query or native simulation is constructed.
When Torch is installed, the PPO actor/critic and CPU Adam updates are real.
Synthetic trial labels here are test fixtures, never benchmark evidence.
"""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sircus.mjlab.benchmark_contract import (
    load_prior,
    make_contract,
    perturbation,
    schedule,
)
from sircus.mjlab.corrected_contract import sha
from sircus.mjlab.learning_contract import digest


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from sircus.mjlab import benchmark_run as worker
else:
    worker = None


ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + "\n")


class SyntheticAdapter:
    """Ordered finite trial fixture; deliberately rejects every fourth slot."""

    def __init__(
        self, _ref, _assets, directory, *, batch, fail_phase=None, tamper=False
    ):
        if batch != 32:
            raise AssertionError("Worker changed the shared native batch size")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.calls, self.ids, self.test_selections = [], set(), []
        self.fail_phase, self.tamper = fail_phase, tamper

    def evaluate(self, requests, phase):
        copied = copy.deepcopy(requests)
        self.calls.append((phase, copied))
        if phase == self.fail_phase:
            raise RuntimeError("deliberate synthetic adapter failure")
        if phase == "test":
            path = self.directory.parent / "selected.json"
            if not path.is_file():
                raise AssertionError(
                    "Test was accessed before selected.json was frozen"
                )
            selected = read(path)
            artifact = self.directory.parent / selected["artifact"]["path"]
            if sha(artifact) != selected["artifact"]["sha256"]:
                raise AssertionError("Selected artifact changed before test")
            self.test_selections.append(copy.deepcopy(selected))
        if self.tamper and phase == "validation":
            item = self.directory.parent / "sweep/milestone-0000.json"
            item.write_bytes(item.read_bytes() + b" ")
            self.tamper = False
        rows = []
        for request in requests:
            if request["id"] in self.ids:
                raise AssertionError("Worker reused a trial ID")
            self.ids.add(request["id"])
            if phase == "train":
                index = request["train_index"]
                released = index % 4 != 0
                captured = released and index % 3 == 1
            else:
                context = int(request["context_id"].rsplit("-", 1)[-1])
                index = 3 * context + request["repeat"]
                released = index % 4 != 0
                # Different fixed milestone labels make checkpoint selection
                # testable; they are explicitly synthetic and not a simulator.
                captured = released and (request["samples"] > 0 or index % 3 == 1)
            row = dict(
                id=request["id"],
                bucket_xy_m=request["bucket_xy_m"],
                values=request["values"],
                native_release=released,
                invalid=None if released else "synthetic_fixture_reject",
                success=captured,
                route_success=captured,
                route_quality=(2 if index % 2 == 0 else 1) if captured else 0,
                complete_success=captured,
                board_bounce=released,
                whole_ball_entry=captured,
                max_retention_s=1.1 if captured else 0.1,
                closest_opening_distance_m=0.02 if captured else 0.3,
                external_contact_steps=0,
                board_contact_episodes=int(released),
                single_bounce_success=captured,
                scalar_score_audit_passed=released,
                synthetic_test_fixture=True,
            )
            rows.append(row)
        with (self.directory / "shots.jsonl").open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        return rows, dict(
            requested=len(rows),
            native_releases=sum(row["native_release"] for row in rows),
            geometry_s=0.01,
            path_s=0.01,
            gpu_s=0.0,
            native_init_s=0.0,
            wall_s=0.02,
            scope="Synthetic CPU test fixture; no native physics executed",
        )


class SyntheticMeter:
    def __init__(self, *, gpu_uuid, log_path):
        self.gpu_uuid, self.log_path = gpu_uuid, Path(log_path)
        self.closed, self.close_calls = False, 0
        self.adapter_calls, self.snapshot_closed = [], []
        self.wall = dict(setup=0.0, train=0.0, evaluation=0.0)
        write(self.log_path, dict(synthetic_test_fixture=True))

    @contextlib.contextmanager
    def phase(self, phase):
        try:
            yield
        finally:
            self.wall[phase] += 0.1

    def add_adapter(self, timing, *, training):
        self.adapter_calls.append((copy.deepcopy(timing), training))

    def snapshot(self):
        self.snapshot_closed.append(self.closed)
        return dict(
            wall_s=self.wall["setup"] + self.wall["train"],
            cpu_s=0.0,
            geometry_s=0.0,
            gpu_s=0.0,
            evaluation_wall_s=self.wall["evaluation"],
            peak_rss_mb=1.0,
            peak_vram_mb=None,
            vram_samples=0,
        )

    def close(self):
        self.closed = True
        self.close_calls += 1


@unittest.skipUnless(HAS_TORCH, "CPU Torch is required for worker/PPO integration")
class WorkerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.study = self.root / "study"
        self.study.mkdir()
        self.prior = load_prior(ROOT / "baselines/mjlab-grid35-expanded-20261103")
        self.ref = dict(
            scene={},
            contract=dict(fingerprint="a" * 64),
            source_hashes={},
            asset_hashes={},
            packages={},
        )
        self.contract = make_contract(
            self.ref, self.prior, seeds=[7], budget=32, milestones=[0, 32]
        )
        write(self.study / "contract.json", self.contract)
        write(self.study / "prior.json", self.prior)
        self.adapters, self.meters = [], []

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, method, *, fail_phase=None, tamper=False):
        def adapter(*args, **kwargs):
            value = SyntheticAdapter(
                *args, fail_phase=fail_phase, tamper=tamper, **kwargs
            )
            self.adapters.append(value)
            return value

        def meter(**kwargs):
            value = SyntheticMeter(**kwargs)
            self.meters.append(value)
            return value

        args = SimpleNamespace(
            study=self.study,
            reference=self.root / "mock-reference",
            assets=self.root / "mock-assets",
            gpu_lock=self.root / "mock-lock",
            method=method,
            seed=7,
            queue_timeout=1.0,
            wall_time_limit=60.0,
        )
        with (
            patch.object(worker, "load_reference", return_value=self.ref),
            patch.object(
                worker, "study_identity", return_value=self.contract["identity"]
            ),
            patch.object(worker, "BenchmarkAdapter", side_effect=adapter),
            patch.object(worker, "ResourceMeter", side_effect=meter),
            patch.object(
                worker,
                "gpu_slot",
                side_effect=lambda *_a, **_k: contextlib.nullcontext(),
            ) as slot,
            patch.object(worker, "foreign_processes", return_value=[]) as foreign,
            patch.object(
                worker.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=10 * 1024**3),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            worker.run(args)
        self.assertEqual(slot.call_count, 1)
        self.assertGreater(foreign.call_count, 0)
        return self.study / "runs" / f"{method}-seed7"

    def check_completed(self, method):
        directory = self.execute(method)
        report = read(directory / "report.json")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["train_requests"], 32)
        self.assertEqual(report["train_native_releases"], 24)
        self.assertEqual([point["samples"] for point in report["curves"]], [0, 32])
        self.assertEqual(
            [point["validation"]["attempts"] for point in report["curves"]], [36, 36]
        )
        self.assertEqual(report["final_test"]["counts"]["attempts"], 72)
        self.assertEqual(report["selected_samples"], 32)
        self.assertFalse(report["final_test"]["search_at_inference"])
        rows = lines(directory / "rows.jsonl")
        raw = lines(directory / "native/shots.jsonl")
        self.assertEqual(len(rows), 176)
        self.assertEqual(len({row["id"] for row in rows}), 176)
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in raw])
        self.assertTrue(all(row["synthetic_test_fixture"] for row in rows))
        train = [row for row in rows if row["role"] == "train"]
        self.assertEqual(len(train), 32)
        self.assertEqual([row["train_index"] for row in train], list(range(32)))
        expected = schedule(7, 32)
        self.assertEqual(
            [row["context_id"] for row in train],
            [self.contract["splits"]["train"][i]["id"] for i in expected],
        )
        self.assertEqual(sum(bool(row["invalid"]) for row in train), 8)
        train_ids = {row["id"] for row in train}
        for role, samples in (("validation", 0), ("validation", 32), ("test", 32)):
            predicted = read(directory / f"predictions/{role}-{samples:04d}.json")
            wanted = 12 if role == "validation" else 24
            self.assertEqual(len(predicted["rows"]), wanted)
            self.assertEqual(len({p["context_id"] for p in predicted["rows"]}), wanted)
            self.assertFalse(predicted["selection_or_learning"])
            artifact = predicted["artifact"]
            self.assertEqual(sha(directory / artifact["path"]), artifact["sha256"])
            for context in predicted["rows"]:
                cohort = [
                    r
                    for r in rows
                    if r["role"] == role
                    and r["samples"] == samples
                    and r["context_id"] == context["context_id"]
                ]
                self.assertEqual(len(cohort), 3)
                self.assertEqual([r["repeat"] for r in cohort], [0, 1, 2])
                for trial in cohort:
                    np.testing.assert_array_equal(
                        trial["command_values"], context["command_values"]
                    )
                    noise = perturbation(context["context_id"], trial["repeat"])
                    np.testing.assert_array_equal(
                        trial["perturbation_mixed_units"], noise
                    )
                    np.testing.assert_allclose(
                        trial["values"],
                        np.asarray(trial["command_values"]) + noise,
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(
                        trial["frozen_artifact_sha256"], artifact["sha256"]
                    )
                    self.assertTrue(
                        set(trial.get("source_training_ids", [])) <= train_ids
                    )
        adapter = self.adapters[0]
        self.assertTrue(adapter.test_selections)
        self.assertTrue(
            all(
                s["frozen_before_test"] and s["selected_samples"] == 32
                for s in adapter.test_selections
            )
        )
        selection = read(directory / "selected.json")
        self.assertEqual(
            selection["artifact"]["sha256"],
            report["final_test"]["frozen_artifact_sha256"],
        )
        meter = self.meters[0]
        self.assertTrue(meter.closed)
        self.assertEqual(meter.close_calls, 1)
        self.assertTrue(meter.snapshot_closed[-1])
        self.assertEqual(sum(training for _timing, training in meter.adapter_calls), 1)
        self.assertEqual(
            sum(
                timing["requested"]
                for timing, training in meter.adapter_calls
                if training
            ),
            32,
        )
        manifest = read(directory / "manifest.json")
        for name, expected_sha in manifest.items():
            self.assertEqual(sha(directory / name), expected_sha)
        self.assertIn("native/shots.jsonl", manifest)
        self.assertIn("rows.jsonl", manifest)
        return directory

    def test_guided_sweep_complete_worker_uses_full_paired_cohorts_and_frozen_transfer(
        self,
    ):
        directory = self.check_completed("sweep")
        self.assertEqual(read(directory / "report.json")["updates"], 0)
        initial = read(directory / "sweep/milestone-0000.json")
        final = read(directory / "sweep/milestone-0032.json")
        self.assertEqual(initial["state"]["best"], {})
        self.assertEqual(len(final["state"]["counts"]), 32)
        self.assertTrue(final["state"]["best"])

    def test_residual_ppo_complete_worker_performs_real_cpu_adam_update(self):
        directory = self.check_completed("ppo")
        updates = lines(directory / "updates.jsonl")
        self.assertEqual(len(updates), 1)
        self.assertNotEqual(updates[0]["actor_before"], updates[0]["actor_after"])
        self.assertEqual(len(updates[0]["log"]["epochs"]), 6)
        initial = torch.load(
            directory / "checkpoints/milestone-0000.pt",
            map_location="cpu",
            weights_only=True,
        )
        final = torch.load(
            directory / "checkpoints/milestone-0032.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.assertNotEqual(initial["actor_sha256"], final["actor_sha256"])
        with np.load(directory / "buffers/update-0000.npz") as buffer:
            self.assertEqual(buffer["observation"].shape, (32, 9))
            self.assertEqual(buffer["latent"].shape, (32, 5))
            self.assertEqual(buffer["old_logp"].shape, (32,))
            self.assertEqual(len(buffer["ids"]), 32)
        self.assertEqual(read(directory / "report.json")["updates"], 1)

    def test_worker_propagates_adapter_failure_and_preserves_partial_evidence(self):
        with self.assertRaisesRegex(
            RuntimeError, "deliberate synthetic adapter failure"
        ):
            self.execute("ppo", fail_phase="train")
        directory = self.study / "runs/ppo-seed7"
        report = read(directory / "report.json")
        self.assertEqual(report["status"], "failed")
        self.assertIn("deliberate synthetic adapter failure", report["error"])
        self.assertEqual(len(lines(directory / "rows.jsonl")), 36)
        self.assertFalse((directory / "manifest.json").exists())
        self.assertFalse((directory / "selected.json").exists())
        self.assertTrue(self.meters[0].closed)

    def test_evaluation_rejects_changed_frozen_artifact(self):
        with self.assertRaisesRegex(RuntimeError, "Frozen inference changed"):
            self.execute("sweep", tamper=True)
        directory = self.study / "runs/sweep-seed7"
        self.assertEqual(read(directory / "report.json")["status"], "failed")
        self.assertFalse((directory / "manifest.json").exists())
        self.assertTrue(self.meters[0].closed)

    def test_changed_contract_prior_is_rejected_before_gpu_slot_or_adapter(self):
        prior = copy.deepcopy(self.prior)
        prior["records"][0]["values"][2] += 0.001
        write(self.study / "prior.json", prior)
        with self.assertRaisesRegex(ValueError, "contract/shared prior changed"):
            self.execute("sweep")
        self.assertEqual(self.adapters, [])
        self.assertEqual(self.meters, [])
        self.assertFalse((self.study / "runs").exists())

    def test_existing_worker_directory_is_preserved(self):
        directory = self.study / "runs/sweep-seed7"
        directory.mkdir(parents=True)
        marker = directory / "preserve.txt"
        marker.write_text("existing work")
        with self.assertRaises(FileExistsError):
            self.execute("sweep")
        self.assertEqual(marker.read_text(), "existing work")
        self.assertEqual(self.adapters, [])
        self.assertEqual(self.meters, [])


if __name__ == "__main__":
    unittest.main()
