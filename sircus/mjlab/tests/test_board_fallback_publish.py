import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab.board_fallback_publish import publish


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.study=self.root/'study';self.study.mkdir()
        self.report=self.study/'report.json';self.manifest=self.study/'evidence-manifest.json';self.audit=self.root/'audit.json';self.display=self.root/'display.json'
        self.report.write_text(json.dumps(dict(version='sircus-board-fallback-study-v1',status='completed',physics_substeps=64)))
        self.manifest.write_text('{}')
        self.bind()
        self.display.write_text(json.dumps({'historical':{'captures':17}}))

    def bind(self):
        self.audit.write_text(json.dumps(dict(status='passed',report_sha256=hashlib.sha256(self.report.read_bytes()).hexdigest(),evidence_manifest_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest())))

    def test_publishes_only_derived_results_and_preserves_history(self):
        before=self.report.read_bytes();publish(self.study,self.audit,self.display)
        result=json.loads(self.display.read_text())
        self.assertEqual(result['historical'],{'captures':17})
        self.assertEqual(result['board_fallback_study']['audit_status'],'passed')
        self.assertEqual(before,self.report.read_bytes())

    def test_changed_report_or_manifest_cannot_publish(self):
        for path in (self.report,self.manifest):
            original=path.read_bytes();path.write_bytes(original+b' ')
            with self.assertRaises(ValueError):publish(self.study,self.audit,self.display)
            self.assertNotIn('board_fallback_study',json.loads(self.display.read_text()))
            path.write_bytes(original)

    def test_refuses_overwriting_evidence(self):
        for target in (self.report,self.manifest,self.audit):
            with self.assertRaises(ValueError):publish(self.study,self.audit,target)

    def test_pending_audit_cannot_publish(self):
        a=json.loads(self.audit.read_text());a['status']='pending';self.audit.write_text(json.dumps(a))
        with self.assertRaises(ValueError):publish(self.study,self.audit,self.display)


if __name__ == '__main__':unittest.main()
