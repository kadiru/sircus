import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab.parallel_board_publish import publish
from sircus.mjlab.benchmark_verify import sha


class ParallelPublishTests(unittest.TestCase):
    def fixture(self,path):
        study=path/'study';source=path/'source';parent=path/'parent'
        for p in (study,source,parent):p.mkdir()
        audit=path/'audit.json';old_audit=path/'source-audit.json';parent_audit=path/'parent-audit.json'
        for p in (old_audit,parent_audit):p.write_text('{}')
        (study/'contract.json').write_text(json.dumps(dict(source_paths=dict(study=str(source),audit=str(old_audit),parent_study=str(parent),parent_audit=str(parent_audit)))))
        report=dict(version='sircus-parallel-board-study-v1',status='completed',phase='completed',scope='development',physics_substeps=64,
                    budgets=dict(positions=6,repeats=3,planning_s=30.),controls={k:dict(status='passed') for k in ('controls_pre','controls_post')})
        (study/'report.json').write_text(json.dumps(report));(study/'evidence-manifest.json').write_text('{}')
        audit.write_text(json.dumps(dict(status='passed',report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'))))
        return study,source,parent,audit

    def test_publish_preserves_raw_bytes_and_other_display_sections(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);s,source,parent,a=self.fixture(p);before=(s/'report.json').read_bytes()
            target=p/'display.json';target.write_text('{"older": 1}')
            publish(s,a,target);value=json.loads(target.read_text())
            self.assertEqual(value['older'],1);self.assertEqual(value['parallel_board_study']['audit_status'],'passed')
            self.assertEqual((s/'report.json').read_bytes(),before)

    def test_all_raw_roots_audits_and_write_through_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);s,source,parent,a=self.fixture(p)
            for target in (s/'new.json',source/'new.json',parent/'new.json',a,p/'source-audit.json',p/'parent-audit.json'):
                with self.assertRaises(ValueError):publish(s,a,target)
            target=p/'display.json';target.with_suffix('.json.tmp').symlink_to(source/'new.json')
            with self.assertRaises(ValueError):publish(s,a,target)

    def test_changed_report_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);s,source,parent,a=self.fixture(p);(s/'report.json').write_text('{}')
            with self.assertRaises(ValueError):publish(s,a,p/'display.json')
