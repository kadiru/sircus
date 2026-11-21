"""Recovery ownership/budget/runtime safeguards; every external action is mocked."""
import copy
import gzip
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_benchmark_recover as recover
from shot_benchmark_contract import make_protocol
from shot_benchmark_supervisor import sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Fixture:
    def __init__(self, folder):
        self.root = Path(folder) / "workspace"
        self.source = self.root / "source-C"
        isaac = self.source / "sircus/isaac"
        isaac.mkdir(parents=True)
        for name in ("shot_benchmark_contract.py", "shot_benchmark_supervisor.py"):
            shutil.copyfile(Path(recover.__file__).with_name(name), isaac / name)
        (isaac / "shot_benchmark_audit.py").write_text("# mocked frozen auditor\n")
        (isaac / "python.sh").write_text("# mocked wrapper\n")
        self.commit = "a" * 40
        self.archive = self.root / "source-C.tar.gz"
        with tarfile.open(self.archive, "w:gz", format=tarfile.PAX_FORMAT,
                          pax_headers={"comment": self.commit}) as tar:
            for path in sorted(isaac.iterdir()):
                tar.add(path, arcname=str(path.relative_to(self.source)))
        self.exp = self.root / "output/study"
        self.exp.mkdir(parents=True)
        self.camera = self.root / "camera.json"
        self.camera.write_text("{}")
        self.split = self.exp / "splits.json"
        self.split.write_text("{}")
        self.source_identity = dict(commit=self.commit, archive_sha256=sha(self.archive),
            camera_sha256=sha(self.camera), geometry_fingerprint="fixture-geometry",
            scoring_version="fixture-scorer")
        protocol = make_protocol()
        self.experiment = dict(schema="sircus-shot-benchmark-experiment-v1", protocol=protocol,
            source=self.source_identity, split_manifest_file="splits.json", split_manifest_sha256=sha(self.split),
            runs=[dict(**r, path=f"runs/{r['method']}-seed{r['seed']}") for r in protocol["run_order"]])
        self.plan = self.exp / "experiment.json"
        write(self.plan, self.experiment)
        self.failed = dict(status="failed", pid=999999999, source_commit=self.commit,
            total_authorized_train_attempts=60000, elapsed_s=100,
            updated_utc=datetime.now(timezone.utc).isoformat(), error="original audit failure")
        write(self.exp / "coordinator.json", self.failed)
        (self.exp / "coordinator.log").write_text("Original traceback retained\n")
        self.args = SimpleNamespace(experiment=self.plan, source=self.source, archive=self.archive,
            camera=self.camera, source_run=self.root / "old-source-run", workspace=self.root,
            source_commit=self.commit, adopt_audit=self.exp / "first-original-cpu-audit.json")
        first = protocol["run_order"][0]
        self.first = self.exp / self.experiment["runs"][0]["path"]
        self.native(self.first, first["method"], first["seed"])
        self.audit(self.first, first["method"], first["seed"], self.args.adopt_audit)

    def native(self, dest, method, seed):
        dest.mkdir(parents=True)
        (dest / "trials").mkdir()
        (dest / "checkpoints").mkdir()
        (dest / "physics").mkdir()
        (dest / "checkpoints/final.json").write_text("mock checkpoint")
        (dest / "physics/trace.npz").write_bytes(b"mock full-rate trace")
        raw = json.dumps(dict(physics=dict(file="physics/trace.npz", sha256=sha(dest / "physics/trace.npz")))).encode()
        compressed = dest / "trials/train-00000.json.gz"
        compressed.write_bytes(gzip.compress(raw))
        write(dest / "trials/train-00000.json.gz.receipt.json", dict(file=compressed.name,
            sha256=sha(compressed), bytes=compressed.stat().st_size,
            uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
        write(dest / "result.json", dict(status="completed", source=self.source_identity))
        write(dest / "progress.json", dict(status="completed", source=self.source_identity,
            method=method, seed=seed, smoke=False, train=dict(attempts=10000)))
        write(dest / "experiment-binding.json", {"mock": "binding"})
        write(dest / "contract.json", {"mock": "contract"})
        write(dest.with_name(dest.name + ".launch.json"), dict(source_commit=self.commit, session="owned-session",
            container=recover.supervisor.CONTAINER, gpu_uuid=recover.supervisor.GPU_UUID,
            source_host=str(self.source.resolve()), integrity=dict(archive_sha256=sha(self.archive))))
        write(dest.with_name(dest.name + ".native-exec.json"), dict(source_commit=self.commit, session="owned-session"))
        write(dest.with_name(dest.name + ".supervisor.json"), dict(source_commit=self.commit, session="owned-session",
            status="completed", returncode=0))
        dest.with_name(dest.name + ".native.log").write_text("Mock native completed\n")

    def audit(self, dest, method, seed, output):
        value = dict(schema=recover.AUDIT_SCHEMA, status="passed", smoke_only=False, method=method,
            seed=seed, paid_train_attempts=10000, evaluation_attempts=420, source=self.source_identity,
            experiment_sha256=sha(self.plan), split_manifest_sha256=sha(self.split),
            source_result_sha256=sha(dest / "result.json"), source_progress_sha256=sha(dest / "progress.json"),
            source_binding_sha256=sha(dest / "experiment-binding.json"), source_contract_sha256=sha(dest / "contract.json"),
            source_sha256={p.name: sha(p) for p in sorted((self.source / "sircus/isaac").glob("*.py"))},
            audit_source_sha256=sha(self.source / "sircus/isaac/shot_benchmark_audit.py"),
            source_trial_sha256={"trials/train-00000.json.gz": sha(dest / "trials/train-00000.json.gz")},
            checkpoint_sha256={"checkpoints/final.json": sha(dest / "checkpoints/final.json")},
            trials=[dict(trial_file="trials/train-00000.json.gz", file="physics/trace.npz",
                sha256=sha(dest / "trials/train-00000.json.gz"),
                bytes=(dest / "trials/train-00000.json.gz").stat().st_size,
                receipt_sha256=sha(dest / "trials/train-00000.json.gz.receipt.json"),
                uncompressed_sha256=json.loads((dest / "trials/train-00000.json.gz.receipt.json").read_text())["uncompressed_sha256"])])
        write(output, value)
        return value


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def verify(self):
        return recover.verify_audit(self.f.args.adopt_audit, self.f.first, self.f.experiment,
            self.f.args, "sweep", 20261121)

    def test_original_audit_rebound_without_weakening_any_check(self):
        self.assertEqual(self.verify()["status"], "passed")
        with patch.object(recover, "require_dead"):
            root, plan, indexed, failed, deadline = recover.verify_inputs(self.f.args)
        self.assertEqual(root, self.f.exp.resolve())
        self.assertEqual(len(indexed), 6)
        self.assertAlmostEqual(deadline, datetime.fromisoformat(failed["updated_utc"]).timestamp() - 100 + 8*86400)

    def test_original_summary_sha_is_compressed_trial_not_physics(self):
        receipt = self.verify()
        node = receipt["trials"][0]
        self.assertEqual(node["sha256"], sha(self.f.first / node["trial_file"]))
        self.assertNotEqual(node["sha256"], sha(self.f.first / node["file"]))
        original = self.f.args.adopt_audit.read_bytes()
        node["file"] = "physics/another-trace.npz"
        write(self.f.args.adopt_audit, receipt)
        with self.assertRaisesRegex(ValueError, "full-rate physics"):
            self.verify()
        self.f.args.adopt_audit.write_bytes(original)

    def test_wrong_receipt_identity_and_budget_refused(self):
        original = self.f.args.adopt_audit.read_bytes()
        for field, value in [("smoke_only", True), ("status", "failed"), ("method", "ppo"),
                ("seed", 20261122), ("paid_train_attempts", 9999), ("evaluation_attempts", 419),
                ("experiment_sha256", "bad"), ("split_manifest_sha256", "bad"),
                ("audit_source_sha256", "bad"), ("source_sha256", {}), ("source", {})]:
            with self.subTest(field=field):
                receipt = json.loads(original)
                receipt[field] = value
                write(self.f.args.adopt_audit, receipt)
                with self.assertRaises(ValueError):
                    self.verify()
        self.f.args.adopt_audit.write_bytes(original)

    def test_postaudit_source_physics_checkpoint_and_trial_mutations_refused(self):
        for path in (self.f.first / "progress.json", self.f.first / "trials/train-00000.json.gz",
                     self.f.first / "checkpoints/final.json", self.f.first / "physics/trace.npz",
                     self.f.first / "trials/train-00000.json.gz.receipt.json",
                     self.f.source / "sircus/isaac/shot_benchmark_audit.py"):
            original = path.read_bytes()
            with self.subTest(path=path.name):
                path.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    self.verify()
                path.write_bytes(original)

    def test_native_completion_and_owner_must_match(self):
        path = self.f.first.with_name(self.f.first.name + ".supervisor.json")
        original = json.loads(path.read_text())
        for field, value in [("status", "failed"), ("returncode", 1), ("session", "another-owner")]:
            changed = dict(original, **{field: value})
            write(path, changed)
            with self.assertRaises(ValueError):
                self.verify()
        write(path, original)

    def test_cpu_audit_is_original_wrapper_inside_existing_4090_container(self):
        argv = recover.cpu_audit_command(self.f.args, self.f.first, self.f.exp / "new.audit.json")
        self.assertEqual(argv[:4], ["docker", "exec", "-u", "0"])
        self.assertIn("CUDA_VISIBLE_DEVICES=", argv)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=0", argv)
        self.assertIn(recover.supervisor.CONTAINER, argv)
        self.assertIn("/workspace/source-C/sircus/isaac/python.sh", argv)
        self.assertIn("/workspace/source-C/sircus/isaac/shot_benchmark_audit.py", argv)
        self.assertIn(str(recover.AUDIT_SECONDS), argv)
        self.assertNotIn("restart", argv)

    def test_active_owner_and_expired_or_invalid_deadline_refused(self):
        with self.assertRaises(ValueError):
            recover.require_dead(123, kill=lambda *a: None)
        def gone(*a):
            raise ProcessLookupError()
        recover.require_dead(123, kill=gone)
        def unknown(*a):
            raise PermissionError()
        with self.assertRaises(PermissionError):
            recover.require_dead(123, kill=unknown)
        for change in [dict(elapsed_s=-1), dict(elapsed_s=float("nan")), dict(status="running"),
                       dict(updated_utc="2026-01-01T00:00:00")]:
            with self.assertRaises(ValueError):
                recover.finite_deadline(dict(self.f.failed, **change))
        old = dict(self.f.failed, updated_utc="2020-01-01T00:00:00+00:00")
        write(self.f.exp / "coordinator.json", old)
        with patch.object(recover, "require_dead"), self.assertRaisesRegex(ValueError, "deadline expired"):
            recover.verify_inputs(self.f.args)

    def test_any_partial_next_run_refuses_before_adoption_or_launch(self):
        next_dest = self.f.exp / self.f.experiment["runs"][1]["path"]
        for suffix in ("", ".launch.json", ".native.log", ".native-exec.json", ".supervisor.json"):
            path = next_dest if not suffix else next_dest.with_name(next_dest.name + suffix)
            path.mkdir(parents=True) if not suffix else path.write_text("partial")
            calls = []
            with patch.object(recover, "require_dead"), self.assertRaisesRegex(ValueError, "partial run"):
                recover.run(self.f.args, execute=lambda *a, **k: calls.append(a))
            self.assertEqual(calls, [])
            self.assertFalse(self.f.first.with_name(self.f.first.name + ".audit.json").exists())
            path.rmdir() if not suffix else path.unlink()

    def test_storage_guard_never_deletes_evidence(self):
        calls = []
        with patch.object(recover, "require_dead"), patch.object(recover.shutil, "disk_usage",
                return_value=SimpleNamespace(free=recover.FREE_BYTES - 1)), self.assertRaisesRegex(ValueError, "6 GiB"):
            recover.run(self.f.args, execute=lambda *a, **k: calls.append(a))
        self.assertFalse(calls)
        self.assertTrue(self.f.args.adopt_audit.exists())
        self.assertFalse((self.f.exp / "control/recovery-launch.json").exists())

    def test_success_runs_only_five_untouched_runs_in_original_order(self):
        original_status = (self.f.exp / "coordinator.json").read_bytes()
        original_log = (self.f.exp / "coordinator.log").read_bytes()
        calls, launches, audits = [], [], []
        def execute(argv, **kw):
            calls.append(argv)
            self.assertTrue(kw["check"])
            if "--output-host" in argv:
                dest = Path(argv[argv.index("--output-host")+1])
                args_file = Path(argv[argv.index("--native-args-json")+1])
                native = json.loads(args_file.read_text())
                method, seed = native[native.index("--shot-benchmark-method")+1], int(native[-1])
                self.assertEqual(argv[argv.index("--batches")+1], "289")
                self.assertEqual(Path(argv[1]), self.f.source / "sircus/isaac/shot_benchmark_supervisor.py")
                launches.append((method, seed))
                self.f.native(dest, method, seed)
            else:
                self.assertIn("CUDA_VISIBLE_DEVICES=", argv)
                mapped_run = argv[argv.index("--run")+1]
                dest = self.f.root / Path(mapped_run).relative_to("/workspace")
                out = self.f.root / Path(argv[-1]).relative_to("/workspace")
                progress = json.loads((dest / "progress.json").read_text())
                self.f.audit(dest, progress["method"], progress["seed"], out)
                audits.append((progress["method"], progress["seed"]))
            return SimpleNamespace(returncode=0)
        with patch.object(recover, "require_dead"):
            recover.run(self.f.args, execute=execute, sleep=lambda t: self.fail("No wait expected in mock"))
        expected = [(x["method"], x["seed"]) for x in self.f.experiment["protocol"]["run_order"][1:]]
        self.assertEqual(launches, expected)
        self.assertEqual(audits, expected)
        self.assertEqual(len(calls), 10)
        status = json.loads((self.f.exp / "recovery-coordinator.json").read_text())
        receipt_path = self.f.exp / "control/recovery-launch.json"
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["recovery_receipt_sha256"], sha(receipt_path))
        self.assertEqual(receipt["adopted_audit"]["source_sha256"], receipt["adopted_audit"]["canonical_sha256"])
        self.assertEqual(original_status, (self.f.exp / "coordinator.json").read_bytes())
        self.assertEqual(original_log, (self.f.exp / "coordinator.log").read_bytes())
        with patch.object(recover, "require_dead"), self.assertRaisesRegex(ValueError, "recovery already exists"):
            recover.run(self.f.args, execute=execute)

    def test_nonzero_native_or_audit_stops_without_next_paid_run(self):
        for failure in ("native", "audit"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                fixture = Fixture(temp)
                launches = []
                def execute(argv, **kw):
                    if "--output-host" in argv:
                        dest = Path(argv[argv.index("--output-host")+1])
                        launches.append(dest.name)
                        fixture.native(dest, "ppo", 20261121)
                        if failure == "native":
                            path = dest.with_name(dest.name + ".supervisor.json")
                            write(path, dict(status="failed", returncode=1))
                        return SimpleNamespace(returncode=0)
                    raise subprocess.CalledProcessError(1, argv)
                with patch.object(recover, "require_dead"), self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    recover.run(fixture.args, execute=execute, sleep=lambda t: None)
                self.assertEqual(launches, ["ppo-seed20261121"])
                status = json.loads((fixture.exp / "recovery-coordinator.json").read_text())
                self.assertEqual(status["status"], "failed")
                self.assertTrue((fixture.exp / "coordinator.json").exists())

    def test_changed_archive_or_plan_or_helper_refused(self):
        for target in (self.f.archive, self.f.camera, self.f.source / "sircus/isaac/shot_benchmark_supervisor.py"):
            original = target.read_bytes()
            target.write_bytes(original + b"changed")
            with patch.object(recover, "require_dead"), self.assertRaises(ValueError):
                recover.verify_inputs(self.f.args)
            target.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
