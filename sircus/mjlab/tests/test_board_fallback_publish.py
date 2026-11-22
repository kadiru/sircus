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

    def test_refuses_every_file_inside_either_raw_study(self):
        heading=self.root/'heading';heading.mkdir()
        heading_audit=self.root/'heading-audit.json';heading_audit.write_text('{}')
        for evidence in (self.study, heading):
            existing=evidence/'choices.json';existing.write_text('{"raw": "preserve"}')
            nested=evidence/'native'/'new-display.json';nested.parent.mkdir()
            for target in (existing,nested):
                before=existing.read_bytes()
                with self.assertRaisesRegex(ValueError,'separate from original evidence'):
                    publish(self.study,self.audit,target,heading,heading_audit)
                self.assertEqual(existing.read_bytes(),before)
                self.assertFalse(nested.exists())

    def test_resolved_directory_and_file_aliases_cannot_reach_evidence(self):
        choices=self.study/'choices.json';choices.write_text('{"raw": "preserve"}')
        directory_alias=self.root/'raw-alias';directory_alias.symlink_to(self.study,target_is_directory=True)
        file_alias=self.root/'display-alias.json';file_alias.symlink_to(choices)
        audit_alias=self.root/'audit-alias.json';audit_alias.symlink_to(self.audit)
        before=choices.read_bytes();audit_before=self.audit.read_bytes()
        for target in (directory_alias/'choices.json',file_alias,audit_alias):
            with self.assertRaisesRegex(ValueError,'separate from original evidence'):
                publish(self.study,self.audit,target)
        self.assertEqual(choices.read_bytes(),before)
        self.assertEqual(self.audit.read_bytes(),audit_before)
        self.assertTrue(file_alias.is_symlink())

    def test_temporary_output_alias_cannot_overwrite_raw_evidence_or_audit(self):
        raw=self.study/'choices.json';raw.write_text('{"raw": "preserve"}')
        temporary=self.display.with_suffix(self.display.suffix+'.tmp')
        display_before=self.display.read_bytes()
        for target in (raw,self.audit):
            before=target.read_bytes();temporary.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'separate from original evidence'):
                publish(self.study,self.audit,self.display)
            self.assertEqual(target.read_bytes(),before)
            self.assertEqual(self.display.read_bytes(),display_before)
            temporary.unlink()

    def test_pending_audit_cannot_publish(self):
        a=json.loads(self.audit.read_text());a['status']='pending';self.audit.write_text(json.dumps(a))
        with self.assertRaises(ValueError):publish(self.study,self.audit,self.display)


if __name__ == '__main__':unittest.main()
