"""CPU/mock guards for explicit sweep evidence; never import or launch a GPU."""

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import sweep_interface as adapter
from sircus.mjlab import sweep_worker as worker
from sircus.mjlab.corrected_contract import PHYSICS, sha
from sircus.mjlab.learning_contract import digest


ROOT = Path(__file__).resolve().parents[3]


def request(**extra):
    return dict(
        request_id="request-a",
        mode="search",
        bucket_xy=[0.67, -0.5],
        constraints=dict(board_position=None, dropper_position=None),
        **extra
    )


def shot(**extra):
    row = dict(
        id="shot-a",
        bucket_xy_m=[0.67, -0.5],
        values=[0.67, -0.1, 0.32, 28.0, 0.45],
        board_center_m=[0.67, -0.1, 0.32],
        board_normal=[0.0, -0.5, 0.866],
        board_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        release_ball_center_m=[0.67, -0.1, 0.8075],
        q10=[0.0] * 6,
        q14=[0.0] * 6,
        link_positions10=[[0.0, 0.0, 0.0]] * 6,
        link_positions14=[[1.33731, 0.0, 0.0]] * 6,
        native_release=True,
        route_success=True,
        invalid=None,
        route_quality=2,
        board_contact_episodes=1,
        max_retention_s=1.0,
        external_contact_steps=0,
        scalar_score_audit_passed=True,
        native_cache_file="native-0000.npz",
        native_cache_row=0,
    )
    row.update(extra)
    return row


class SweepWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.scene = json.loads((ROOT / "sircus/isaac/scene.json").read_text())

    def tearDown(self):
        self.temp.cleanup()

    def ref(self):
        return dict(
            root=self.directory,
            scene=self.scene,
            contract=dict(fingerprint="contract-a"),
            source_hashes={"worker": "a" * 64},
            asset_hashes={"mesh": "b" * 64},
            report=dict(evidence_sha256={}),
            catalogue=dict(sources=[]),
        )

    def item(self, **extra):
        value = dict(
            qualified=True,
            source=shot(),
            transit_file="path.json",
            nominal_attempts=3,
            nominal_captures=3,
            fresh_attempts=20,
            fresh_captures=20,
        )
        value.update(extra)
        return value

    def write_evidence(self, ref, item):
        path = self.directory / "native-0000.npz"
        positions = np.arange(48, dtype=float).reshape(16, 1, 3) / 1000
        np.savez_compressed(
            path, position_m=positions, record_ids=np.asarray([item["source"]["id"]])
        )
        motion = self.directory / "path.json"
        motion.write_text(json.dumps(dict(passed=True, frames=[])))
        ref["report"]["evidence_sha256"] = {
            path.name: sha(path),
            motion.name: sha(motion),
        }
        return positions

    def test_exact_source_never_rounds_or_uses_position_locks(self):
        ref = self.ref()
        weak = self.item(fresh_captures=18)
        strong = self.item()
        unqualified = self.item(qualified=False, fresh_captures=999)
        ref["catalogue"]["sources"] = [weak, strong, unqualified]
        self.assertIs(worker.exact_source(ref, request()), strong)
        changed = request()
        changed["bucket_xy"][0] += 1e-12
        self.assertIsNone(worker.exact_source(ref, changed))
        for name in ("board_position", "dropper_position"):
            locked = request()
            locked["constraints"][name] = [0.67, -0.1, 0.32]
            self.assertIsNone(worker.exact_source(ref, locked))

    def test_cached_trajectory_requires_hash_row_identity_and_admitted_path(self):
        ref, item = self.ref(), self.item()
        positions = self.write_evidence(ref, item)
        result = worker.cached_result(ref, request(), item)
        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "catalog_lookup")
        self.assertTrue(result["provenance"]["saved_evidence"])
        self.assertFalse(result["provenance"]["search_at_inference"])
        self.assertEqual(result["metrics"]["search_requests"], 0)
        np.testing.assert_array_equal(
            [row["ball"] for row in result["trajectory"]], positions[3::4, 0]
        )
        item["source"]["id"] = "another-shot"
        with self.assertRaisesRegex(ValueError, "another shot"):
            worker.cached_result(ref, request(), item)
        item["source"]["id"] = "shot-a"
        (self.directory / "path.json").write_text(json.dumps(dict(passed=False)))
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            worker.cached_result(ref, request(), item)
        ref["report"]["evidence_sha256"]["path.json"] = sha(
            self.directory / "path.json"
        )
        with self.assertRaisesRegex(ValueError, "not admitted"):
            worker.cached_result(ref, request(), item)

    def test_evidence_cannot_escape_reference_or_ignore_hash(self):
        ref = self.ref()
        for name in ("../escape.npz", str(self.directory / "absolute.npz")):
            with self.assertRaisesRegex(ValueError, "inside"):
                worker.evidence_file(ref, name)
        path = self.directory / "trajectory.npz"
        path.write_bytes(b"not native evidence")
        for expected in (None, "a" * 64):
            ref["report"]["evidence_sha256"] = (
                {} if expected is None else {path.name: expected}
            )
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                worker.evidence_file(ref, path.name)

    def test_query_cache_binds_result_identity_and_all_saved_evidence(self):
        directory = self.directory / "query-cache"
        directory.mkdir()
        hashes = {"worker": "a" * 64}
        self.assertIsNone(worker.read_query_cache(directory, "query-a", hashes))
        result = dict(
            success=True,
            provenance=dict(request_sha256="query-a", source_sha256=hashes),
        )
        trace = directory / "trace.npz"
        trace.write_bytes(b"fixture native trace")
        saved = directory / "result.json"

        def save(value, include_result=True):
            saved.write_text(json.dumps(value))
            manifest = {trace.name: sha(trace)}
            if include_result:
                manifest[saved.name] = sha(saved)
            (directory / "manifest.json").write_text(json.dumps(manifest))

        save(result)
        self.assertEqual(worker.read_query_cache(directory, "query-a", hashes), result)
        for identity in ("another-query",):
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                worker.read_query_cache(directory, identity, hashes)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            worker.read_query_cache(directory, "query-a", {"worker": "b" * 64})
        trace.write_bytes(b"changed trace")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            worker.read_query_cache(directory, "query-a", hashes)
        save(result, include_result=False)
        with self.assertRaisesRegex(ValueError, "omits its result"):
            worker.read_query_cache(directory, "query-a", hashes)
        save(dict(result, success=False))
        self.assertIsNone(worker.read_query_cache(directory, "query-a", hashes))

    def test_query_cache_manifest_path_cannot_escape(self):
        directory = self.directory / "query-cache"
        directory.mkdir()
        (directory / "result.json").write_text("{}")
        (directory / "manifest.json").write_text(json.dumps({"../outside": "a" * 64}))
        with self.assertRaisesRegex(ValueError, "inside"):
            worker.read_query_cache(directory, "query-a", {"worker": "a" * 64})

    def test_success_requires_verified_status_qualification_and_native_evidence(self):
        ref, row = self.ref(), shot()
        motion = dict(passed=True)
        valid = worker.result_for(
            ref, request(), row, status="verified_shot", qualified=True, motion=motion
        )
        self.assertTrue(valid["success"])
        for status in (
            "busy",
            "capture_found_confirmation_unresolved",
            "no_verified_solution_within_budget",
        ):
            self.assertFalse(
                worker.result_for(
                    ref, request(), row, status=status, qualified=True, motion=motion
                )["success"]
            )
        self.assertFalse(
            worker.result_for(
                ref,
                request(),
                row,
                status="verified_shot",
                qualified=False,
                motion=motion,
            )["success"]
        )
        for path in (None, {}, dict(passed=False)):
            self.assertFalse(
                worker.result_for(
                    ref,
                    request(),
                    row,
                    status="verified_shot",
                    qualified=True,
                    motion=path,
                )["success"]
            )
        for update in (
            dict(native_release=False),
            dict(scalar_score_audit_passed=False),
            dict(invalid="strict_geometry"),
            dict(route_success=False),
        ):
            self.assertFalse(
                worker.result_for(
                    ref,
                    request(),
                    dict(row, **update),
                    status="verified_shot",
                    qualified=True,
                    motion=motion,
                )["success"]
            )

    def reference_fixture(self):
        repo, reference, assets = (
            self.directory / "repo",
            self.directory / "reference",
            self.directory / "assets",
        )
        reference.mkdir()
        assets.mkdir()
        repo.mkdir()
        critical = "sircus/mjlab/contact_env.py"
        for name in (
            critical,
            "sircus/mjlab/sweep_worker.py",
            "sircus/mjlab/sweep_contract.py",
        ):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n")
        (assets / "mesh.stl").write_bytes(b"fixture mesh")
        spec = dict(
            physics=PHYSICS,
            scene=self.scene,
            source_sha256={critical: sha(repo / critical)},
            asset_sha256={"assets/mesh.stl": sha(assets / "mesh.stl")},
        )
        spec["fingerprint"] = digest(spec)
        report = dict(status="completed", evidence_sha256={})
        catalogue = dict(
            physics=PHYSICS, contract_fingerprint=spec["fingerprint"], sources=[]
        )
        catalogue["fingerprint"] = digest(catalogue)
        for name, value in (
            ("contract.json", spec),
            ("report.json", report),
            ("confirmed-catalogue.json", catalogue),
        ):
            (reference / name).write_text(json.dumps(value))
        public = dict(
            status="completed",
            physics=PHYSICS,
            contract_fingerprint=spec["fingerprint"],
            artifacts_sha256={
                name: sha(reference / name)
                for name in ("contract.json", "report.json", "confirmed-catalogue.json")
            },
        )
        path = repo / "baselines/mjlab-corrected-grid-20261102/native-results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(public))
        return repo, reference, assets, critical

    def test_reference_changed_source_or_asset_rejected_before_gpu(self):
        repo, reference, assets, critical = self.reference_fixture()
        with ExitStack() as stack:
            stack.enter_context(patch.object(worker, "CRITICAL", (critical,)))
            stack.enter_context(
                patch.object(
                    worker.importlib.metadata, "version", return_value="fixture"
                )
            )
            packages = stack.enter_context(
                patch.object(worker, "validate_packages", side_effect=lambda x: x)
            )
            stack.enter_context(
                patch(
                    "sircus.mjlab.gpu_slot.gpu_slot",
                    side_effect=AssertionError("GPU must never be touched"),
                )
            )
            result = worker.load_reference(reference, assets, repository=repo)
            self.assertEqual(result["scene"], self.scene)
            packages.reset_mock()
            (repo / critical).write_text("# changed physics\n")
            with self.assertRaisesRegex(ValueError, "source differs"):
                worker.load_reference(reference, assets, repository=repo)
            packages.assert_not_called()
            (repo / critical).write_text("# fixture\n")
            (assets / "mesh.stl").write_bytes(b"changed mesh")
            with self.assertRaisesRegex(ValueError, "asset changed"):
                worker.load_reference(reference, assets, repository=repo)

    def test_reference_artifact_hash_failure_stops_worker_before_gpu(self):
        repo, reference, assets, _ = self.reference_fixture()
        (reference / "report.json").write_text(json.dumps(dict(status="tampered")))
        source = self.directory / "request.json"
        source.write_text(json.dumps(request()))
        args = SimpleNamespace(
            reference=reference,
            assets=assets,
            request=source,
            cache=self.directory / "cache",
            output=self.directory / "result.json",
            gpu_lock=self.directory / "lock",
            queue_timeout=0,
        )
        real_loader = worker.load_reference
        with patch.object(
            worker,
            "load_reference",
            side_effect=lambda r, a: real_loader(r, a, repository=repo),
        ), patch("sircus.mjlab.gpu_slot.gpu_slot") as gpu, patch.object(
            worker, "NativePlanner"
        ) as native:
            with self.assertRaisesRegex(ValueError, "differs from published"):
                worker.run(args)
            gpu.assert_not_called()
            native.assert_not_called()

    def test_busy_worker_is_a_no_trial_result_and_never_constructs_native(self):
        ref = self.ref()
        source = self.directory / "request.json"
        source.write_text(json.dumps(request()))
        args = SimpleNamespace(
            reference=self.directory,
            assets=self.directory,
            request=source,
            cache=self.directory / "cache",
            output=self.directory / "result.json",
            gpu_lock=self.directory / "lock",
            queue_timeout=0,
        )
        with patch.object(worker, "load_reference", return_value=ref), patch.object(
            worker, "exact_source", return_value=None
        ), patch(
            "sircus.mjlab.gpu_slot.gpu_slot", side_effect=TimeoutError("busy")
        ), patch.object(
            worker, "NativePlanner"
        ) as native:
            worker.run(args)
        native.assert_not_called()
        result = json.loads(args.output.read_text())
        self.assertEqual(result["status_native"], "busy")
        self.assertFalse(result["success"])
        self.assertEqual(result["trajectory"], [])
        self.assertIn("left untouched", result["message"])

    def test_adapter_rejects_policy_before_launching_child(self):
        with patch.object(adapter.subprocess, "Popen") as process:
            with self.assertRaisesRegex(ValueError, "search only"):
                adapter.solve(dict(request(), mode="policy"))
            process.assert_not_called()

    def test_adapter_preserves_busy_result_and_selects_only_the_4090(self):
        environment = {
            "SIRCUS_SWEEP_" + key: str(self.directory / key.lower())
            for key in ("REFERENCE", "ASSETS", "LOCK", "CACHE")
        }
        result = dict(
            request_id="request-a",
            status_native="busy",
            success=False,
            method="sweep_planner",
            trajectory=[],
        )
        proposal_counts = []

        def child(command, **options):
            self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], adapter.GPU_UUID)
            self.assertTrue(options["start_new_session"])
            submitted = json.loads(
                Path(command[command.index("--request") + 1]).read_text()
            )
            proposal_counts.append(submitted["cpu_candidates"])
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps(result))
            return SimpleNamespace(
                returncode=0, communicate=lambda **kwargs: ("", None), poll=lambda: 0
            )

        with patch.dict(adapter.os.environ, environment), patch.object(
            adapter.subprocess, "Popen", side_effect=child
        ), patch.object(adapter, "_terminate_group"):
            self.assertEqual(adapter.solve(request()), result)
            self.assertEqual(adapter.solve(request(search_budget=16)), result)
            self.assertEqual(adapter.solve(request(search_budget=128)), result)
        self.assertEqual(proposal_counts, [256, 64, 512])

    def test_adapter_timeout_cancels_only_its_mock_child(self):
        environment = {
            "SIRCUS_SWEEP_" + key: str(self.directory / key.lower())
            for key in ("REFERENCE", "ASSETS", "LOCK", "CACHE")
        }

        class Child:
            def communicate(self, **kwargs):
                raise adapter.subprocess.TimeoutExpired("mock-child", adapter.TIMEOUT_S)

        child = Child()
        with patch.dict(adapter.os.environ, environment), patch.object(
            adapter.subprocess, "Popen", return_value=child
        ), patch.object(adapter, "_terminate_group") as terminate:
            with self.assertRaisesRegex(RuntimeError, "600-second request budget"):
                adapter.solve(request())
        self.assertTrue(terminate.called)
        self.assertTrue(all(call.args == (child,) for call in terminate.call_args_list))


if __name__ == "__main__":
    unittest.main()
