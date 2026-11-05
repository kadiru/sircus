"""Synthetic controller/export fixtures; no native audit or physics is claimed."""

import copy
import csv
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from sircus.mjlab import curate_xy650 as export
from sircus.mjlab.tests import test_xy650_study as controller_fixture


class XYExportTests(unittest.TestCase):
    def fixture(self, *, blocked=False, private_xy=False):
        controller=controller_fixture.CleanControllerTests()
        self.addCleanup(controller.doCleanups)
        original_sources=controller_fixture.sources

        def source_rows():
            rows=original_sources()
            for row in rows:
                row['values']=[.4708757793012148,-.2778934014509746,.38,28.,.3868]
                row['bucket_xy_m']=[-.030306598129941298,-.6917970430784122]
            return rows

        class FixtureExecutor(controller_fixture.ExecutorFixture):
            def execute(current, requests):
                before=len(current.rows)
                super().execute(requests)
                for row in current.rows[before:]:
                    row['native_initial_q10']=copy.deepcopy(row['q10'])
                    row['native_initial_q14']=copy.deepcopy(row['q14'])
                    row['execution_fingerprint']=controller_fixture.protocol.execution_fingerprint(row)
                    row['first_flight']=dict(version=export.METRIC_VERSION,crossing_plane='ball_center_at_rim',
                        available=True,reason=None,start_index=2,end_index=10,start_time_s=.02,end_time_s=.1,
                        cutoff_reason='horizon',cutoff_contact_channels=[],sample_count=9,
                        first_crossing=dict(time_s=.05,interval_index=4,fraction=1.,
                            xy_m=['PRIVATE-MEASURED-XY'] if private_xy else copy.deepcopy(row['bucket_xy_m'])),
                        mouth_clearance_m=.01,closest_opening_distance_m=.02)
                (current.output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in current.rows))

        with patch.object(controller_fixture,'sources',source_rows),patch.object(controller_fixture,'ExecutorFixture',FixtureExecutor):
            study,report,_,_=controller.run_fixture(fail_initial=blocked)
        self.study,self.report=study,report
        self.audit_path=study/'audit-xy650.json'
        self.audit=dict(version=export.AUDIT_VERSION,status='passed',checks_count=1,
            checks=[dict(label='Synthetic export fixture; not a native independent audit',passed=True)],failures=[],
            auditor_sha256='a'*64,auditor_identity=dict(dependencies_sha256={'synthetic-checker.py':'b'*64}))
        self.bind()
        return study

    def bind(self):
        spec=export.read(self.study/'contract.json')
        manifest={str(p.relative_to(self.study)):export.sha(p) for p in self.study.rglob('*') if p.is_file()
                  and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
        export.write_json(self.study/'evidence-manifest.json',manifest)
        self.report['evidence_sha256']=manifest
        export.write_json(self.study/'report.json',self.report)
        self.audit.update(report_sha256=export.sha(self.study/'report.json'),evidence_manifest_sha256=export.sha(self.study/'evidence-manifest.json'),
            contract_fingerprint=spec['fingerprint'],scene_fingerprint=spec['scene_fingerprint'],requested_trials=self.report['counts']['attempts'],
            native_releases=self.report['counts']['native_releases'],gate=copy.deepcopy(self.report['gate']))
        export.write_json(self.audit_path,self.audit)

    def records(self):return [json.loads(line) for line in (self.study/'records.jsonl').read_text().splitlines()]
    def write_records(self,rows): (self.study/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def run_export(self,name='publication'):return export.curate(self.study,self.audit_path,self.study.parent/name)

    def test_full388_preserves_configured_translations_hashes_lf_and_denominators(self):
        self.fixture();before={str(p.relative_to(self.study)):p.read_bytes() for p in self.study.rglob('*') if p.is_file()}
        summary=self.run_export();out=self.study.parent/'publication'
        self.assertEqual(summary['study_version'],export.STUDY_VERSION)
        self.assertEqual(summary['counts']['attempts'],388)
        self.assertEqual(summary['counts']['phase_counts'],dict(controls_pre=12,coarse=253,refinement=81,verification=30,controls_post=12))
        self.assertEqual(summary['settings']['translations'],dict(dx_m=[-.01,0.,.01],dy_m=[.04,.06,.08],rule=export.DISTANCE_RULE))
        self.assertEqual(summary['gate']['qualified_settings'],5)
        self.assertEqual(summary['provenance']['scene_fingerprint'],self.report['scene_fingerprint'])
        content=(out/'requested-shots.csv').read_bytes();self.assertNotIn(b'\r',content)
        rows=list(csv.DictReader(content.decode().splitlines()));self.assertEqual(len(rows),388)
        self.assertEqual(rows[-1]['phase'],'controls_post')
        self.assertEqual(len({r['execution_fingerprint'] for r in rows}),388)
        self.assertTrue(all(len(r['query_fingerprint'])==64 for r in rows))
        translated=[r for r in rows if r['query_kind']=='translated_grid']
        self.assertEqual(len(translated),243)
        self.assertEqual({r['translation_x_m'] for r in translated},{'-0.01','0.0','0.01'})
        self.assertEqual(summary['settings']['max_requests'],388)
        self.assertEqual(summary['settings']['verification_dedup_decimals'],9)
        for name,h in export.read(out/'checksums.json').items():self.assertEqual(export.sha(out/name),h)
        self.assertEqual(before,{str(p.relative_to(self.study)):p.read_bytes() for p in self.study.rglob('*') if p.is_file()})

    def test_blocked_controls_export_only12_and_keep_all_declared_layouts(self):
        self.fixture(blocked=True);summary=self.run_export()
        self.assertEqual(summary['counts']['attempts'],12)
        self.assertEqual(summary['gate']['status'],'blocked_controls')
        self.assertEqual(summary['gate']['qualified_settings'],0)
        self.assertEqual(summary['declared_groups'],255)
        self.assertEqual(summary['executed_groups'],2)
        self.assertEqual(summary['verification'],[])

    def test_privacy_excludes_measured_crossing_xy_raw_arrays_and_extra_secrets(self):
        self.fixture(private_xy=True)
        secret='NEVER-PUBLISH-host-secret'
        self.report.update(token=secret,host_path='/Users/private/'+secret,position_m=[[secret]])
        for row in self.report['groups']+self.report['top_settings']+self.report['verification']:
            row.update(q10=[secret],link_positions14=[[secret]],native_pose={'raw':secret})
        spec=export.read(self.study/'contract.json');spec['settings']['extra']={'secret':secret}
        unsigned=dict(spec);unsigned.pop('fingerprint');spec['fingerprint']=export.digest(unsigned)
        export.write_json(self.study/'contract.json',spec)
        self.report['settings']=spec['settings'];self.report['contract_fingerprint']=spec['fingerprint'];self.bind()
        summary=self.run_export()
        for path in (self.study.parent/'publication').iterdir():
            text=path.read_text();self.assertNotIn(secret,text);self.assertNotIn('PRIVATE-MEASURED-XY',text)
            self.assertNotIn('"xy_m"',text);self.assertNotIn('"position_m"',text);self.assertNotIn('"q10"',text)
        self.assertEqual(set(summary['top_settings'][0]['first_flight']['first_crossing']),{'time_s','interval_index','fraction'})
        self.assertIn('board_xy_m',summary['top_settings'][0]['configuration'])

    def test_actual_execution_hashes_required_and_rejects_explicitly_have_none(self):
        self.fixture(blocked=True);original=self.records()
        for change in (lambda r:r.pop('execution_fingerprint'),lambda r:r.update(execution_fingerprint=None),
                       lambda r:r.update(execution_fingerprint='not-a-hash')):
            records=copy.deepcopy(original);change(records[0]);self.write_records(records);self.bind()
            with self.assertRaises(ValueError):self.run_export()
        records=copy.deepcopy(original)
        for row in records:
            if row['role']!='negative_control':continue
            row.update(eligible=False,native_release=False,invalid='strict_endpoint_rejected',scalar_score_audit_passed=False,
                       execution_fingerprint=None,first_flight=None)
            row.pop('rim_band_contact');row.pop('minimum_mouth_clearance_mm')
        self.write_records(records)
        for g in self.report['groups']:
            if g['role']=='negative_control':g['eligible']=False
        self.report['counts'].update(export.statistics(records))
        for c in self.report['controls']:
            selected=[r for r in records if r['phase']==c['phase'] and r['group_id']==c['group_id'] and r['substeps']==c['substeps']]
            c['stats']=export.statistics(selected)
        self.report['control_gates']['controls_pre']=controller_fixture.protocol.control_gate(records)
        self.bind();summary=self.run_export()
        self.assertEqual(summary['counts']['attempts'],12);self.assertEqual(summary['counts']['geometry_rejections'],6)
        rows=list(csv.DictReader((self.study.parent/'publication/requested-shots.csv').read_text().splitlines()))
        rejected=[r for r in rows if r['native_release']=='False'];self.assertEqual(len(rejected),6)
        self.assertTrue(all(r['execution_fingerprint']==r['first_flight_version']==r['rim_band_contact']=='' for r in rejected))
        next(r for r in records if not r['native_release'])['execution_fingerprint']='d'*64;self.write_records(records);self.bind()
        with self.assertRaisesRegex(ValueError,'Rejected request invents'):self.run_export('invented')

    def test_audit_structure_exact_pair_report_bytes_and_all_manifest_members_required(self):
        self.fixture(blocked=True);original=copy.deepcopy(self.audit)
        for edit in (dict(version='sircus-xy650-independent-audit-v1'),dict(version='sircus-clean650-independent-audit-v3'),dict(status='failed'),dict(checks=[],checks_count=0),
                     dict(checks_count=2),dict(checks=[{'label':'false','passed':1}]),dict(failures=['failure'])):
            self.audit=dict(original,**edit);export.write_json(self.audit_path,self.audit)
            with self.assertRaises(ValueError):self.run_export()
        self.audit=original;export.write_json(self.audit_path,self.audit)
        original_report=(self.study/'report.json').read_bytes();(self.study/'report.json').write_bytes(original_report+b'\n')
        with self.assertRaisesRegex(ValueError,'Audit/report bytes'):self.run_export()
        (self.study/'report.json').write_bytes(original_report)
        (self.study/'calibration.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'Changed evidence bytes'):self.run_export()

    def test_explicit_external_v2_audit_can_publish_while_failed_v1_is_preserved(self):
        self.fixture(blocked=True)
        external=self.study.parent/'independent-v2-proof.json'
        external.write_bytes(self.audit_path.read_bytes())
        failed=dict(self.audit,version='sircus-xy650-independent-audit-v1',status='failed',failures=['Historical checker assumption'])
        export.write_json(self.audit_path,failed);before=self.audit_path.read_bytes()
        with self.assertRaisesRegex(ValueError,'Full independent audit'):self.run_export()
        summary=export.curate(self.study,external,self.study.parent/'external-publication')
        self.assertEqual(summary['provenance']['independent_audit_version'],'sircus-xy650-independent-audit-v2')
        self.assertEqual(summary['provenance']['independent_audit_sha256'],export.sha(external))
        self.assertEqual(self.audit_path.read_bytes(),before)

    def test_literal_settings_scene_query_and_metric_guards_fail_closed(self):
        self.fixture(blocked=True);spec=export.read(self.study/'contract.json')
        for edit in (dict(max_requests=389),dict(coarse_requests=252),dict(translations=dict(dx_m=[0.],dy_m=[.06],rule=export.DISTANCE_RULE))):
            with self.assertRaises(ValueError):export.settings(dict(spec['settings'],**edit))
        self.audit['scene_fingerprint']='f'*64;export.write_json(self.audit_path,self.audit)
        with self.assertRaisesRegex(ValueError,'Static scene fingerprint'):self.run_export()
        self.bind();self.report['groups'][0]['query_fingerprint']='f'*64;self.bind()
        with self.assertRaisesRegex(ValueError,'Configured query fingerprint'):self.run_export()

    def test_first_flight_whitelist_checks_intervals_finite_values_and_unavailable_schema(self):
        self.fixture(blocked=True);original=self.records()
        for field,value in [('mouth_clearance_m',{'private':'raw'}),('closest_opening_distance_m',-1),('sample_count',100),
                            ('cutoff_contact_channels',['secret-channel'])]:
            records=copy.deepcopy(original);records[0]['first_flight'][field]=value
            self.write_records(records);self.bind()
            with self.assertRaises(ValueError):self.run_export()
        unavailable=dict(version=export.METRIC_VERSION,crossing_plane='ball_center_at_rim',available=False,reason='invalid_physics',
            start_index=None,end_index=None,start_time_s=None,end_time_s=None,cutoff_reason=None,cutoff_contact_channels=[],first_crossing=None,
            mouth_clearance_m=None,closest_opening_distance_m=None,sample_count=0)
        self.assertEqual(export.flight(unavailable,native=True),unavailable)
        for changed in [dict(unavailable,closest_opening_distance_m=.01),dict(unavailable,first_crossing={'xy_m':[0,0]})]:
            with self.assertRaises(ValueError):export.flight(changed,native=True)

    def test_output_reuse_incomplete_state_traversal_and_finite_request_budget_guards(self):
        self.fixture(blocked=True)
        with self.assertRaisesRegex(ValueError,'outside evidence'):export.curate(self.study,self.audit_path,self.study/'publication')
        self.report['status']='running';self.bind()
        with self.assertRaisesRegex(ValueError,'not completed'):self.run_export()
        self.report['status']='completed';self.bind();self.run_export()
        with self.assertRaisesRegex(ValueError,'New publication directory'):self.run_export()
        manifest=export.read(self.study/'evidence-manifest.json');manifest['../private']='a'*64
        export.write_json(self.study/'evidence-manifest.json',manifest);self.report['evidence_sha256']=manifest
        export.write_json(self.study/'report.json',self.report)
        self.audit.update(report_sha256=export.sha(self.study/'report.json'),evidence_manifest_sha256=export.sha(self.study/'evidence-manifest.json'))
        export.write_json(self.audit_path,self.audit)
        with self.assertRaisesRegex(ValueError,'Unsafe provenance filename'):self.run_export('escaped')

    def test_requested_counts_duplicate_hashes_and_adaptive_provenance_cannot_be_relabelled(self):
        self.fixture();original_count=self.report['counts']['attempts']
        self.report['counts']['attempts']=389;self.bind()
        with self.assertRaisesRegex(ValueError,'trial accounting'):self.run_export()
        self.report['counts']['attempts']=original_count
        rows=self.records();original=copy.deepcopy(rows)
        rows[1]['execution_fingerprint']=rows[0]['execution_fingerprint'];self.write_records(rows);self.bind()
        with self.assertRaisesRegex(ValueError,'Duplicate actual execution'):self.run_export()
        self.write_records(original)
        group=next(g for g in self.report['groups'] if g['query_kind']=='local_refinement')
        group['source_query_fingerprint']='f'*64;self.bind()
        with self.assertRaisesRegex(ValueError,'Requested query provenance|Adaptive source provenance'):self.run_export()


if __name__=='__main__':unittest.main()
