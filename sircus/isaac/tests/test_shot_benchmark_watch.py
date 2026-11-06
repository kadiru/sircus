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
            with patch.object(module,"publish",return_value={"observed_train_attempts":60000}) as publish,patch.object(module.time,"sleep") as sleep:
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


if __name__=="__main__":unittest.main()
