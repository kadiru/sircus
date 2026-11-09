"""Mocked ownership and mutable-input tests; no Docker, GPU or native launch."""

import copy
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import candidate_comparison_supervisor as wrapper
import candidate_selector as selector
import shot_benchmark_supervisor as supervisor
import test_candidate_selector as selector_fixture


class CandidateSupervisorTests(unittest.TestCase):
    def setUp(self):
        helper=selector_fixture.SelectorTests()
        helper.setUp();self.addCleanup(helper.doCleanups)
        self.root=helper.root
        self.plan=helper.build()
        self.checkpoint=self.root/"frozen/seed22.pt"
        self.checkpoint.parent.mkdir();self.checkpoint.write_bytes(b"synthetic checkpoint")
        self.archive=self.root/"source.tar";self.archive.write_bytes(b"synthetic archive")
        self.plan["native_binding"]=dict(original_source=dict(commit="1"*40,geometry_fingerprint="2"*64),
            source_commit="3"*40,source_archive_sha256=supervisor.sha(self.archive),
            checkpoint_path="/workspace/frozen/seed22.pt",checkpoint_sha256=selector.PARENT_SHA256,
            geometry_fingerprint="2"*64,source_scene_sha256="5"*64,bank_preprocessing={},selection_wall_s=.1)
        # Selector enforces the real parent hash; this synthetic I/O fixture uses
        # its own verified checkpoint bytes behind the validator mock only.
        self.plan["native_binding"]["checkpoint_sha256"]=supervisor.sha(self.checkpoint)
        self.comparison=self.root/"comparison.json"
        self.write_comparison()
        self.native_args=self.root/"args.json"
        self.native_args.write_text(json.dumps(["--candidate-comparison-plan","/workspace/comparison.json"]))
        self.lock=self.root/"scene.lock";self.lock.write_text("")
        self.args=SimpleNamespace(workspace_host=self.root,source_commit="3"*40,
            archive_host=self.archive,native_args_json=self.native_args,helper_host=Path(wrapper.__file__),
            batches=2,entrypoint="film_cylinder_farm.py",supervise=True,session="owned",
            output_host=self.root/"new-result",plan_only=False)
        self.base=dict(session="owned",input_sha256={str(self.native_args):supervisor.sha(self.native_args)},
                       integrity=dict(archive_sha256=supervisor.sha(self.archive)),
                       lock_host=str(self.lock),source_commit="3"*40,
                       native_exec_receipt=str(self.root/"native-exec.json"),command=["never-executed"])

    def write_comparison(self):
        value=copy.deepcopy(self.plan);value.pop("fingerprint",None)
        self.plan=selector.signed(value)
        self.comparison.write_text(json.dumps(self.plan,indent=2)+"\n")

    def mocked_base(self,*args):
        return copy.deepcopy(self.base)

    def make_plan(self):
        with patch.object(wrapper,"_BASE_MAKE_PLAN",side_effect=self.mocked_base), \
             patch.object(selector,"validate_comparison"):
            return wrapper.make_plan(self.args,"owned")

    def test_external_comparison_and_parent_bytes_are_both_frozen_inputs(self):
        plan=self.make_plan()
        self.assertEqual(plan["input_sha256"][str(self.comparison.resolve())],supervisor.sha(self.comparison))
        self.assertEqual(plan["input_sha256"][str(self.checkpoint.resolve())],supervisor.sha(self.checkpoint))
        self.assertEqual(plan["comparison_fingerprint"],self.plan["fingerprint"])

    def test_changed_parent_model_or_producer_archive_fails_before_any_launch(self):
        self.checkpoint.write_bytes(b"mutated")
        with self.assertRaisesRegex(ValueError,"checkpoint bytes changed"):self.make_plan()
        self.checkpoint.write_bytes(b"synthetic checkpoint")
        self.archive.write_bytes(b"mutated archive")
        with self.assertRaisesRegex(ValueError,"producer/archive"):self.make_plan()

    def test_missing_duplicate_or_outside_plan_argument_is_rejected(self):
        for values in ([],["--candidate-comparison-plan"],
                       ["--candidate-comparison-plan","/workspace/comparison.json","--candidate-comparison-plan=/workspace/comparison.json"],
                       ["--candidate-comparison-plan","/outside/comparison.json"],
                       ["--candidate-comparison-plan","/workspace/../comparison.json"]):
            self.native_args.write_text(json.dumps(values))
            with self.subTest(values=values),self.assertRaises(ValueError):self.make_plan()

    def test_detached_launch_expected_hash_cannot_be_changed_before_supervision(self):
        expected=supervisor.sha(self.comparison)
        self.comparison.write_text(self.comparison.read_text()+" ")
        with patch.dict(wrapper.os.environ,{wrapper.EXPECTED_PLAN_ENV:expected}):
            with self.assertRaisesRegex(ValueError,"detached launch receipt"):self.make_plan()

    def test_missing_native_binding_is_rejected_before_queue(self):
        del self.plan["native_binding"];self.write_comparison()
        with self.assertRaisesRegex(ValueError,"complete producer binding"):self.make_plan()

    def test_actual_supervise_rechecks_external_plan_after_queue_without_launching(self):
        calls=[]
        def base(*args):
            calls.append(None);return copy.deepcopy(self.base)
        def queue(*args):
            self.comparison.write_text(self.comparison.read_text()+" ")
        original_planner=supervisor.make_plan
        previous_deadlines=supervisor.WAIT_SECONDS,supervisor.NATIVE_SECONDS
        with patch.object(wrapper,"_BASE_MAKE_PLAN",side_effect=base), \
             patch.object(selector,"validate_comparison"), \
             patch.object(supervisor,"parse_args",return_value=self.args), \
             patch.object(supervisor,"require_running_container"), \
             patch.object(supervisor,"wait_for_slot",side_effect=queue), \
             patch.object(supervisor.subprocess,"Popen") as launch:
            with self.assertRaisesRegex(RuntimeError,"Frozen inputs changed while waiting"):
                wrapper.main()
            launch.assert_not_called()
        self.assertEqual(len(calls),2)
        self.assertIs(supervisor.make_plan,original_planner)
        self.assertEqual((supervisor.WAIT_SECONDS,supervisor.NATIVE_SECONDS),previous_deadlines)

    def test_unchanged_queued_plan_rechecks_then_launches_only_owned_command(self):
        calls=[]
        def base(*args):calls.append(None);return copy.deepcopy(self.base)
        child=Mock();child.wait.return_value=0;child.pid=4321
        with patch.object(wrapper,"_BASE_MAKE_PLAN",side_effect=base), \
             patch.object(selector,"validate_comparison"), \
             patch.object(supervisor,"parse_args",return_value=self.args), \
             patch.object(supervisor,"require_running_container"), \
             patch.object(supervisor,"wait_for_slot"), \
             patch.object(supervisor.signal,"signal"), \
             patch.object(supervisor.subprocess,"Popen",return_value=child) as launch:
            self.assertEqual(wrapper.main(),0)
        self.assertEqual(len(calls),2)
        self.assertEqual(launch.call_args.args[0],["never-executed"])

    def test_minimal_python_helper_routes_need_no_numpy_scipy_or_torch(self):
        for option, name in (("--container-exec", "container_exec"),
                             ("--container-terminate", "container_terminate")):
            script = (
                "import sys; sys.path.insert(0,%r); "
                "import candidate_comparison_supervisor as w; "
                "assert not any(x in sys.modules for x in ('numpy','scipy','torch','candidate_selector')); "
                "setattr(w.supervisor,%r,lambda raw: print('owned-helper:'+raw)); "
                "sys.argv=['candidate_comparison_supervisor.py',%r,'{}']; "
                "w.main(); "
                "assert not any(x in sys.modules for x in ('numpy','scipy','torch','candidate_selector'))"
            ) % (str(SOURCE), name, option)
            with self.subTest(option=option):
                result=subprocess.run([sys.executable,"-S","-c",script],check=True,
                                      capture_output=True,text=True)
                self.assertEqual(result.stdout.strip(),"owned-helper:{}")


if __name__=="__main__":unittest.main()
