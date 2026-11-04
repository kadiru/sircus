"""File-only publisher tests: no services, GPUs or physical clients."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from sircus.mjlab import shallow_pipeline as module
from sircus.mjlab.prepare_shallow_report import prepare as actual_prepare


class ShallowPublisherTests(unittest.TestCase):
    def fixture(self, root):
        study=root/'study';display=root/'display.json'
        display.write_text(json.dumps(dict(distance_evaluation={'preserved':'parent'},old_policy={'active':'unchanged'})))
        study.with_suffix('.pipeline.json').write_text(json.dumps(dict(status='running',started_unix_s=time.time())))
        raw=dict(version='sircus-shallow-board-candidate-v1',settings={},targets=[],contexts=[],selected_recipes=[],trace_examples=[],counts={})
        return study,display,raw

    def test_queue_and_completed_full_audit_publish_separate_key_without_history_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            study,display,raw=self.fixture(Path(directory));module.publish(study,display,raw)
            value=json.loads(display.read_text());self.assertEqual(value['distance_evaluation'],{'preserved':'parent'})
            self.assertEqual(value['shallow_evaluation']['status'],'waiting_for_4090')
            study.mkdir();raw.update(status='completed',phase='completed',contract_fingerprint='synthetic')
            (study/'report.json').write_text(json.dumps(raw))
            (study/'audit-shallow.json').write_text(json.dumps(dict(status='passed',contract_fingerprint='synthetic',checks_count=1,checks=[{'passed':True}],failures=[])))
            module.publish(study,display,raw);value=json.loads(display.read_text())
            self.assertTrue(value['shallow_evaluation']['audit']['full_checks_verified'])
            self.assertEqual(value['old_policy'],{'active':'unchanged'})
            self.assertEqual(value['distance_evaluation'],{'preserved':'parent'})

    def test_concurrent_report_update_is_retried_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            study,display,raw=self.fixture(Path(directory));study.mkdir()
            raw.update(status='running',phase='coarse-64');(study/'report.json').write_text(json.dumps(raw))
            calls=[]
            def prepare(base,source,output):
                result=actual_prepare(base,source,output)
                if not calls:
                    value=json.loads(display.read_text());value['unrelated_writer']='preserved';display.write_text(json.dumps(value))
                calls.append(1);return result
            with patch('sircus.mjlab.prepare_shallow_report.prepare',prepare):module.publish(study,display,raw)
            value=json.loads(display.read_text());self.assertEqual(len(calls),2)
            self.assertEqual(value['unrelated_writer'],'preserved');self.assertEqual(value['shallow_evaluation']['phase'],'coarse-64')


if __name__=='__main__':unittest.main()
