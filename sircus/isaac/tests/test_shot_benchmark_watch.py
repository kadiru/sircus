"""Finite CPU publisher safeguards; mock encoding and no network/services."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import shot_benchmark_watch as module


class WatchTests(unittest.TestCase):
    def test_requires_explicit_cpu_and_finite_deadline(self):
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}):
            with self.assertRaisesRegex(ValueError,"CPU publication"):
                module.watch("unused","unused-output")
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}):
            with self.assertRaisesRegex(ValueError,"Unbounded"):
                module.watch("unused","unused-output",duration_s=0)

    def test_completed_coordinator_publishes_and_exits_without_sleep(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}):
            root=Path(folder)/"native";out=Path(folder)/"public";root.mkdir();out.mkdir()
            (root/"experiment.json").write_text(json.dumps(dict(runs=[])))
            (root/"coordinator.json").write_text(json.dumps(dict(status="completed")))
            (out/"report.json").write_text("{}")
            with patch.object(module,"publish",return_value={"observed_train_attempts":60000,"status":"completed","coordinator":{"status":"completed"}}) as publish,patch.object(module.time,"sleep") as sleep:
                result=module.watch(root,out)
            self.assertEqual(result["observed_train_attempts"],60000)
            self.assertEqual(json.loads((root/"publication-watch.json").read_text())["status"],"completed")
            publish.assert_called_once();sleep.assert_not_called()

    def test_publication_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}):
            root=Path(folder)/"native";root.mkdir()
            (root/"experiment.json").write_text(json.dumps(dict(runs=[])))
            with patch.object(module,"publish",side_effect=ValueError("changed receipt")):
                with self.assertRaisesRegex(ValueError,"changed receipt"):module.watch(root,Path(folder)/"public")
            self.assertEqual(json.loads((root/"publication-watch.json").read_text())["status"],"failed")

    def test_unbound_raw_completion_does_not_stop_publication_before_audits(self):
        for value in ({"status":"awaiting_audit","coordinator":{"status":"completed"}},
                      {"status":"attention_required","coordinator":{"status":"failed"}}):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}):
                root=Path(folder)/"native";out=Path(folder)/"public";root.mkdir();out.mkdir()
                (root/"experiment.json").write_text(json.dumps(dict(runs=[])))
                # The raw receipt must not override the report's validated gate.
                (root/"coordinator.json").write_text(json.dumps(dict(status="completed")))
                (out/"report.json").write_text("{}")
                with patch.object(module,"publish",return_value=dict(value,observed_train_attempts=10000)),patch.object(module.time,"monotonic",side_effect=[0,0,0,61,61]),patch.object(module.time,"sleep") as sleep,patch.object(module,"atomic",wraps=module.atomic) as atomic:
                    with self.assertRaisesRegex(TimeoutError,"expired"):module.watch(root,out,duration_s=60)
                sleep.assert_called_once();self.assertEqual(atomic.call_args_list[0].args[1]["status"],"watching")

    def test_media_failure_keeps_verified_progress_publishing_and_retries(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}):
            root=Path(folder)/"native";out=Path(folder)/"public";root.mkdir();out.mkdir()
            run=root/"run";run.mkdir()
            (run/"clips.json").write_text("{}");(run/"progress.json").write_text("{}")
            (root/"experiment.json").write_text(json.dumps(dict(runs=[dict(path="run")])))
            (root/"coordinator.json").write_text(json.dumps(dict(status="completed")))
            (out/"report.json").write_text("{}")
            with patch.object(module,"encode_run",side_effect=[ValueError("missing frame"),{}]) as encode,patch.object(module,"publish",return_value={"observed_train_attempts":10000,"status":"completed","coordinator":{"status":"completed","origin":"recovery-coordinator.json"}}) as publish,patch.object(module.time,"sleep") as sleep,patch.object(module,"atomic",wraps=module.atomic) as atomic:
                result=module.watch(root,out)
            self.assertEqual(result["observed_train_attempts"],10000)
            self.assertEqual(publish.call_count,2);self.assertEqual(encode.call_count,2);sleep.assert_called_once()
            first=atomic.call_args_list[0].args[1]
            self.assertEqual(first["status"],"watching_with_media_errors")
            self.assertEqual(first["media_errors"],[dict(run="run",error="missing frame")])
            self.assertEqual(json.loads((root/"publication-watch.json").read_text())["status"],"completed")


if __name__=="__main__":unittest.main()
