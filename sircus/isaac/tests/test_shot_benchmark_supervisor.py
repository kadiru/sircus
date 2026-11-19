"""Ownership/source safeguards with mocked process queries; launches nothing."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import shot_benchmark_supervisor as supervisor
from shot_benchmark_study import native_arguments


class SupervisorTests(unittest.TestCase):
    def test_other_gpu_is_not_a_conflict_or_launch_target(self):
        def query(argv,**kwargs):
            return SimpleNamespace(stdout="GPU-other, 71\n"+supervisor.GPU_UUID+", 82\n")
        self.assertEqual(supervisor.foreign_processes(query),[82])
        self.assertEqual(supervisor.CONTAINER,"sircus_isaac_4090")

    def test_existing_container_must_have_exact_4090_assignment(self):
        calls=[]
        def inspect(argv,**kwargs):
            calls.append(argv)
            return SimpleNamespace(stdout=json.dumps([dict(State=dict(Running=True),
                HostConfig=dict(DeviceRequests=[dict(DeviceIDs=[supervisor.GPU_UUID])]))]))
        supervisor.require_running_container(inspect)
        self.assertEqual(len(calls),2)
        def wrong(argv,**kwargs):
            return SimpleNamespace(stdout=json.dumps([dict(State=dict(Running=True),
                HostConfig=dict(DeviceRequests=[dict(DeviceIDs=["GPU-other"])]))]))
        with self.assertRaises(RuntimeError):supervisor.require_running_container(wrong)

    def test_native_arguments_cannot_override_device_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"args.json"
            for value in [["--device=cuda:1"],["--output"],["{unknown}"]]:
                p.write_text(json.dumps(value))
                with self.assertRaises(ValueError):supervisor.load_native_args(p,{})
        args=native_arguments("/workspace/new/experiment.json","ppo",20261121)
        self.assertIn("--shot-benchmark-plan",args)
        self.assertNotIn("--shot-benchmark-smoke",args)
        self.assertIn("--shot-benchmark-smoke",native_arguments("plan","sweep",20261121,smoke=True))

    def test_paths_remain_inside_existing_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertEqual(supervisor.container_path(root,root/"run"),"/workspace/run")
            with self.assertRaises(ValueError):supervisor.container_path(root,root.parent/"outside")


if __name__=="__main__":unittest.main()
