import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab.board_fallback_publish import sha
from sircus.mjlab.guarded_board_fallback_publish import publish


class GuardedPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / 'study'
        self.study.mkdir()
        self.audit = self.root / 'audit.json'
        self.display = self.root / 'display.json'
        self.report = dict(version='sircus-guarded-board-fallback-study-v1',
                           status='completed', audit_status='pending', physics_substeps=64,
                           budgets=dict(positions=20),
                           controls={key: dict(status='passed') for key in ('controls_pre', 'controls_post')})
        self.display.write_text(json.dumps({'historical': {'captures': 41}}))
        (self.study / 'evidence-manifest.json').write_text('{}')
        self.bind()

    def bind(self):
        (self.study / 'report.json').write_text(json.dumps(self.report))
        self.audit.write_text(json.dumps(dict(status='passed',
            report_sha256=sha(self.study / 'report.json'),
            evidence_manifest_sha256=sha(self.study / 'evidence-manifest.json'))))

    def test_publishes_only_derived_copy_and_preserves_history(self):
        original = (self.study / 'report.json').read_bytes()
        publish(self.study, self.audit, self.display)
        result = json.loads(self.display.read_text())
        self.assertEqual(result['historical'], {'captures': 41})
        self.assertEqual(result['guarded_board_fallback_study']['audit_status'], 'passed')
        self.assertEqual((self.study / 'report.json').read_bytes(), original)

    def test_modified_report_and_pending_audit_cannot_publish(self):
        before = self.display.read_bytes()
        (self.study / 'report.json').write_text(json.dumps(dict(self.report, extra='unverified')))
        with self.assertRaises(ValueError):
            publish(self.study, self.audit, self.display)
        self.bind()
        receipt = json.loads(self.audit.read_text())
        receipt['status'] = 'pending'
        self.audit.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            publish(self.study, self.audit, self.display)
        self.assertEqual(self.display.read_bytes(), before)

    def test_wrong_protocol_or_failed_control_is_not_displayed(self):
        for key, value in [('physics_substeps', 128), ('budgets', {'positions': 50}),
                           ('controls', {'controls_pre': {'status': 'passed'}})]:
            saved = self.report[key]
            self.report[key] = value
            self.bind()
            with self.assertRaises(ValueError):
                publish(self.study, self.audit, self.display)
            self.report[key] = saved

    def test_raw_destination_and_symlink_temporary_are_protected(self):
        raw = self.study / 'choices.json'
        raw.write_text('{"preserve": true}')
        before = raw.read_bytes()
        with self.assertRaises(ValueError):
            publish(self.study, self.audit, raw)
        self.display.with_suffix('.json.tmp').symlink_to(raw)
        with self.assertRaises(ValueError):
            publish(self.study, self.audit, self.display)
        self.assertEqual(raw.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
