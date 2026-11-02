"""CPU/mocked guards for the post-study native interface adapter."""

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sircus.mjlab import learning_interface as bridge
from sircus.mjlab import learning_solve as native
from sircus.mjlab.learning_contract import WARM, constraints, digest, project


class AdapterArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.repo, self.study, self.assets = (
            root / "repo",
            root / "study",
            root / "tree/assets",
        )
        self.study.mkdir()
        self.assets.mkdir(parents=True)
        for name in native.CRITICAL_MODULES:
            target = self.repo / "sircus/mjlab" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# frozen fixture " + name)
        shared = {}
        for name in native.SHARED_SOURCES:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# shared frozen fixture " + name)
            shared[name] = native.sha256(target)
        mesh = self.assets.parent / "vendor/fixture.stl"
        mesh.parent.mkdir()
        mesh.write_text("fixture collision mesh")
        urdf = '<robot><link><collision><geometry><mesh filename="/workspace/vendor/fixture.stl"/></geometry></collision></link></robot>'
        for name in ("dropper.urdf", "board_arm.urdf"):
            (self.assets / name).write_text(urdf)
        (self.assets / "poses.json").write_text("{}")
        scene = dict(ball=dict(radius_m=0.0375))
        contract = dict(scene=scene, version="fixture")
        contract["fingerprint"] = digest(contract)
        self.checkpoint = dict(
            contract=contract, state={}, weight_sha256="fixture-policy-weights"
        )
        (self.study / "selected.pt").write_bytes(b"fixture checkpoint")
        (self.study / "contract.json").write_text(json.dumps(contract))
        self.report = dict(
            status="completed",
            contract_fingerprint=contract["fingerprint"],
            checkpoint_sha256=native.sha256(self.study / "selected.pt"),
            frozen_evaluation=dict(
                captures=1, policy_weight_sha256="fixture-policy-weights"
            ),
            source_sha256={
                name: native.sha256(self.repo / "sircus/mjlab" / name)
                for name in native.CRITICAL_MODULES
            },
            versions={
                name: "fixture-version"
                for name in ("mjlab", "mujoco", "mujoco-warp", "torch", "warp-lang")
            },
        )
        self.write_report()
        provenance = dict(
            version="sircus-mjlab-artifacts-v1",
            contract_fingerprint=contract["fingerprint"],
            checkpoint_sha256=self.report["checkpoint_sha256"],
            source_sha256=shared,
            asset_sha256={
                name: native.sha256(self.assets.parent / name)
                for name in native.required_asset_paths(self.assets)
            },
        )
        (self.study / "artifact_provenance.json").write_text(json.dumps(provenance))
        benchmark = dict(
            scene=scene,
            scorer_sha256={
                name: shared[f"sircus/isaac/{name}.py"]
                for name in ("rl_contract", "entry_quality", "board_route")
            },
        )
        benchmark["fingerprint"] = digest(benchmark)
        path = self.repo / "baselines/mjlab-comparison-20261102/benchmark.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(benchmark))

    def tearDown(self):
        self.directory.cleanup()

    def write_report(self):
        (self.study / "report.json").write_text(json.dumps(self.report))

    def validate(self):
        with patch.object(
            native.importlib.metadata, "version", return_value="fixture-version"
        ):
            return native.validate_artifacts(
                self.study,
                self.assets,
                repository=self.repo,
                load_checkpoint=lambda _: copy.deepcopy(self.checkpoint),
            )

    def test_complete_matching_artifacts_pass_without_native_import(self):
        with patch.dict(sys.modules, {"sircus.mjlab.learning_env": None}):
            report, checkpoint, sha, fingerprint = self.validate()
        self.assertEqual(sha, self.report["checkpoint_sha256"])
        self.assertEqual(fingerprint, self.report["contract_fingerprint"])
        self.assertEqual(checkpoint, self.checkpoint)
        self.assertEqual(report["status"], "completed")

    def test_unfinished_or_missing_provenance_blocks_before_checkpoint_load(self):
        self.report["status"] = "running"
        self.write_report()
        with self.assertRaisesRegex(RuntimeError, "complete first"):
            self.validate()
        self.report["status"] = "completed"
        self.write_report()
        (self.study / "artifact_provenance.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "artifact_provenance"):
            self.validate()

    def test_changed_checkpoint_source_shared_geometry_or_mesh_is_rejected(self):
        targets = (
            self.study / "selected.pt",
            self.repo / "sircus/mjlab/learning_env.py",
            self.repo / "sircus/isaac/geometry.py",
            self.assets.parent / "vendor/fixture.stl",
        )
        for target in targets:
            before = target.read_bytes()
            target.write_bytes(before + b" tampered")
            with self.assertRaises(RuntimeError, msg=str(target)):
                self.validate()
            target.write_bytes(before)

    def test_wrong_scene_tensor_identity_or_package_version_is_rejected(self):
        self.checkpoint["contract"]["scene"]["ball"]["radius_m"] = 9
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            self.validate()
        self.checkpoint["contract"]["scene"]["ball"]["radius_m"] = 0.0375
        self.report["frozen_evaluation"]["policy_weight_sha256"] = "other-policy"
        self.write_report()
        with self.assertRaisesRegex(RuntimeError, "weights"):
            self.validate()
        self.report["frozen_evaluation"][
            "policy_weight_sha256"
        ] = "fixture-policy-weights"
        self.write_report()
        self.report["versions"]["mujoco"] = "other-version"
        self.write_report()
        with self.assertRaisesRegex(RuntimeError, "dependency"):
            self.validate()

    def test_manifest_cannot_escape_root_or_omit_collision_mesh(self):
        path = self.study / "artifact_provenance.json"
        original = json.loads(path.read_text())
        value = copy.deepcopy(original)
        value["asset_sha256"].pop("vendor/fixture.stl")
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "omits"):
            self.validate()
        value = copy.deepcopy(original)
        value["asset_sha256"]["../escape"] = "wrong"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "declared root"):
            self.validate()


class AdapterConstraintTests(unittest.TestCase):
    def test_invalid_policy_projection_does_not_reject_feasible_search(self):
        xy = np.array([[0.65, -0.5]])
        locks = constraints(1)
        locks["dropper"][0] = [0.65, -0.04, 0.5]
        predicted = np.asarray([WARM.copy()])
        predicted[0, 2] = 0.53  # Above the locked release; this policy proposal fails.
        _, valid = project(predicted, locks)
        self.assertFalse(valid[0])
        self.assertIsNone(native.structural_constraint_error(xy, locks, 0.0375))
        bucket, candidates, lk = native.search_candidates(
            xy, locks, np.random.default_rng(42), policy_values=predicted
        )
        projected, valid = project(candidates, lk)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(projected[:, 2] + 0.0375 + projected[:, 4], 0.5)
        np.testing.assert_allclose(bucket, np.repeat(xy, len(bucket), axis=0))

    def test_structural_conflicts_and_impossible_gap_are_rejected(self):
        xy = np.array([[0.65, -0.5]])
        locks = constraints(1)
        locks["board"][0] = [0.65, -0.04, 0.36]
        locks["dropper"][0] = [0.7, -0.04, 0.7]
        self.assertIn("conflict", native.structural_constraint_error(xy, locks, 0.0375))
        locks["dropper"][0] = [0.65, -0.04, 0.1]
        self.assertIn("height", native.structural_constraint_error(xy, locks, 0.0375))
        locks = constraints(1)
        locks["board"][0] = [0.65, -0.2, 0.36]
        self.assertIn(
            "separation",
            native.structural_constraint_error(np.array([[0.65, -0.2]]), locks, 0.0375),
        )

    def test_other_gpu_workload_is_ignored_but_foreign_4090_is_not(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=native.GPU_UUID):
            with patch.object(
                native.subprocess, "check_output", return_value="GPU-4070, 123\n"
            ):
                native._assert_gpu_free()
            with patch.object(
                native.subprocess,
                "check_output",
                return_value=native.GPU_UUID + ", 999999\n",
            ):
                with self.assertRaisesRegex(RuntimeError, "another workload"):
                    native._assert_gpu_free()
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0,1"):
            with self.assertRaisesRegex(RuntimeError, "only the RTX4090"):
                native._assert_gpu_free()


class AdapterProcessTests(unittest.TestCase):
    def test_bank_routes_only_unlocked_policy_requests(self):
        for mode, locks, bank in (("policy", {}, True), ("policy", {"board_position":[.6,0,.35]},False), ("search",{},False)):
            request=dict(request_id="route-case",mode=mode,constraints=locks)
            process=MagicMock();process.poll.return_value=0;process.returncode=0
            process.communicate.return_value=("",None)
            def popen(args, **kwargs):
                Path(args[args.index("--output")+1]).write_text(json.dumps(dict(request_id="route-case")))
                return process
            with self.settings(),patch.dict(os.environ,SIRCUS_MJLAB_BANK_STUDY="/tmp/bank"),patch.object(bridge.subprocess,"Popen",side_effect=popen) as spawn:
                bridge.solve(request)
            command=spawn.call_args.args[0]
            self.assertEqual(command[2],"sircus.mjlab.bank_solve" if bank else "sircus.mjlab.learning_solve")
            self.assertEqual("--parent-study" in command,bank)

    def settings(self):
        return patch.dict(
            os.environ,
            SIRCUS_MJLAB_STUDY="/tmp/study",
            SIRCUS_MJLAB_ASSETS="/tmp/assets",
            SIRCUS_MJLAB_GPU_LOCK="/tmp/gpu.lock",
        )

    def test_timeout_cancels_entire_process_group_and_reports_error(self):
        process = MagicMock()
        process.pid = 9123
        process.poll.side_effect = [None, -signal.SIGTERM]
        process.communicate.side_effect = [
            subprocess.TimeoutExpired("fixture", 240),
            ("stopped", None),
        ]
        with self.settings(), patch.object(
            bridge.subprocess, "Popen", return_value=process
        ) as popen, patch.object(bridge.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                bridge.solve(dict(request_id="timeout-test"))
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(
                popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], native.GPU_UUID
            )
            kill.assert_any_call(9123, signal.SIGTERM)

    def test_stubborn_process_is_killed_after_term_grace(self):
        process = MagicMock()
        process.pid = 9124
        process.poll.return_value = None
        process.communicate.side_effect = [
            subprocess.TimeoutExpired("fixture", 5),
            ("killed", None),
        ]
        with patch.object(bridge.os, "killpg") as kill:
            bridge._terminate_group(process)
        self.assertEqual(kill.call_args_list[0].args, (9124, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (9124, signal.SIGKILL))

    def test_success_requires_matching_request_result(self):
        request = dict(request_id="request-12")
        process = MagicMock()
        process.poll.return_value = 0
        process.returncode = 0

        def popen(args, **kwargs):
            output = Path(args[args.index("--output") + 1])
            output.write_text(json.dumps(dict(request_id="another-request")))
            return process

        process.communicate.return_value = ("", None)
        with self.settings(), patch.object(
            bridge.subprocess, "Popen", side_effect=popen
        ):
            with self.assertRaisesRegex(RuntimeError, "mismatched"):
                bridge.solve(request)

    def test_native_child_has_deadline_and_dies_with_linux_parent(self):
        libc = MagicMock()
        libc.prctl.return_value = 0
        with patch.object(native.sys, "platform", "linux"), patch.object(
            native.os, "getppid", side_effect=[42, 42]
        ), patch("ctypes.CDLL", return_value=libc), patch.object(
            native.signal, "alarm"
        ) as alarm:
            native._bound_child_lifetime()
        libc.prctl.assert_called_once_with(1, signal.SIGTERM, 0, 0, 0)
        alarm.assert_called_once_with(240)

    def test_parent_exit_during_native_startup_is_rejected(self):
        libc = MagicMock()
        libc.prctl.return_value = 0
        with patch.object(native.sys, "platform", "linux"), patch.object(
            native.os, "getppid", side_effect=[42, 1]
        ), patch("ctypes.CDLL", return_value=libc), patch.object(
            native.signal, "alarm"
        ) as alarm:
            with self.assertRaisesRegex(RuntimeError, "parent exited"):
                native._bound_child_lifetime()
        alarm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
