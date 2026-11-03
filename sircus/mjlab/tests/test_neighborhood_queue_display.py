import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab.neighborhood_queue_display import pending_report,publish_pending


class QueueDisplayTests(unittest.TestCase):
    def test_queued_state_has_no_invented_observations_or_evidence_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);study=root/'new-study'
            study.with_suffix('.pipeline.json').write_text(json.dumps(dict(status='running',started_unix_s=100)))
            report=pending_report(study,now=125)
            self.assertEqual(report['status'],'waiting_for_4090');self.assertEqual(report['queue']['wait_s'],25)
            self.assertEqual(report['counts']['native_releases'],0)
            self.assertTrue(all(c['board_admitted'] is None for c in report['contexts']))
            display=root/'display.json';display.write_text(json.dumps(dict(active_policy='old',prior={'captures':23})))
            self.assertTrue(publish_pending(study,display));self.assertFalse(study.exists())
            value=json.loads(display.read_text());self.assertEqual(value['active_policy'],'old');self.assertEqual(value['prior'],{'captures':23})

    def test_real_evidence_takes_over_without_overwriting_display(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);study=root/'study';study.mkdir();(study/'report.json').write_text('{}')
            display=root/'display.json';display.write_text('{"current":3}')
            self.assertIsNone(pending_report(study));self.assertFalse(publish_pending(study,display))
            self.assertEqual(display.read_text(),'{"current":3}')
