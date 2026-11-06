"""Synthetic CPU-only export fixtures, never native or independent evidence."""
import copy
import csv
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import curate_impact650 as export
from sircus.mjlab import impact650_contract as protocol,impact650_study as study
from sircus.mjlab.tests.test_impact650_study import ImpactFixture
from sircus.mjlab.tests.test_xy650_study import SCENE,ExecutorFixture,sources as old_sources


class ImpactExportTests(unittest.TestCase):
    def fixture(self,*,mode='rim',blocked=False,reject_one=False,private_xy=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        parent=root/'parent';parent.mkdir();assets=root/'assets';assets.mkdir();(root/'asset.txt').write_text('synthetic')
        for name in ('calibration.json','verification.json'):(parent/name).write_text('{}')
        calibration=dict(calibration_sha256='a'*64,verification_sha256='b'*64,score_config_sha256='c'*64,score_config={})
        asset_sha={'asset.txt':export.sha(root/'asset.txt')}
        physical=export.digest(dict(scene=SCENE,calibration=calibration,asset_sha256=asset_sha,capture_version='sircus-calibrated-native-capture-v2'))
        parent_spec=dict(scene=SCENE,calibration=calibration,packages={'synthetic':'fixture'},asset_sha256=asset_sha,scene_fingerprint=physical,fingerprint=export.PARENT_FP)
        sources=[]
        for i,(ident,tilt,gap) in enumerate(zip(export.SOURCE_IDS,[27.5,30.,30.],[.3918,.41,.39])):
            sources.append(dict(id=ident,phase='coarse',substeps=64,values=[.47,-.21,.38,tilt,gap],bucket_xy_m=[-.03,-.21-math.sqrt(.65**2-.5**2)],
                translation_m=[0.,.06],q10=[i/10]*6,q14=[i/20]*6,query_fingerprint=export.digest({'parent_query':ident}),
                native_release=True,invalid=None,invalid_physics=False,route_success=True,route_quality=1,rim_band_contact=True))
        bindings=dict(parent={n:'a'*64 for n in ('contract.json','report.json','evidence-manifest.json','records.jsonl','groups.json','initial-groups.json')},parent_audit_sha256=export.PARENT_AUDIT_SHA)
        controls=protocol.xy.controls(old_sources());geometry=ImpactFixture()

        class Executor(ExecutorFixture):
            def execute(current,requests):
                before=len(current.rows);super().execute(requests)
                for row in current.rows[before:]:
                    if row['phase'] in ('coarse','verification'):
                        success=not(mode=='failed' and row['phase']=='verification' and row['substeps']==128) and not(mode=='uncaptured' and row['phase']=='coarse')
                        rejected=reject_one and row['phase']=='coarse' and row['group_id']=='impact-f00-u0v0'
                        success=success and not rejected
                        row.update(native_release=not rejected,route_success=success,route_quality=2 if mode=='clean' and success else 1 if success else 0,
                            whole_ball_entry=success,max_retention_s=1. if success else 0.,rim_band_contact=success and mode!='clean',
                            invalid='strict_endpoint_rejected' if rejected else None,minimum_mouth_clearance_mm=2. if success else None)
                    native=row['native_release']
                    row.update(native_initial_q10=copy.deepcopy(row['q10']),native_initial_q14=copy.deepcopy(row['q14']),native_initial_ball_center_m=copy.deepcopy(row['release_center_m']))
                    row['execution_fingerprint']=protocol.execution_fingerprint(row) if native else None
                    row['first_flight']=dict(version='sircus-xy650-first-flight-aperture-v1',crossing_plane='ball_center_at_rim',available=True,reason=None,
                        start_index=2,end_index=10,start_time_s=.02,end_time_s=.1,cutoff_reason='horizon',cutoff_contact_channels=[],sample_count=9,
                        first_crossing=dict(time_s=.05,interval_index=4,fraction=1.,xy_m=['PRIVATE-MEASURED-XY'] if private_xy else row['bucket_xy_m']),mouth_clearance_m=.01,closest_opening_distance_m=.02) if native else None
                (current.output/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in current.rows))

        def executor(*args):return Executor(*args,fail_initial=blocked)
        args=SimpleNamespace(parent_study=parent,parent_audit=root/'parent-proof.json',assets=assets,output=root/'evidence',batch=32,wall_time_limit=3600)
        with patch.object(protocol,'load_inputs',return_value=(parent_spec,sources,controls,bindings)), \
             patch.object(study.importlib.metadata,'version',return_value='fixture'), \
             patch('sircus.mjlab.learning_solve.required_asset_paths',return_value=['asset.txt']), \
             patch.object(study,'source_manifest',return_value={'synthetic.py':'f'*64}), \
             patch.object(study,'storage_check',return_value={'synthetic':True}), \
             patch('sircus.mjlab.impact650_geometry.ImpactGeometry',return_value=geometry), \
             patch('sircus.mjlab.impact650_native.NativeExecutor',side_effect=executor),patch('builtins.print'):
            report=study.run(args)
        self.study,self.report=args.output,report;self.audit_path=self.study/'audit-impact650.json'
        self.audit=dict(version=export.AUDIT_VERSION,status='passed',checks_count=1,
            checks=[dict(label='Synthetic exporter fixture, not native independent evidence',passed=True)],failures=[],auditor_sha256='a'*64,
            auditor_identity=dict(dependencies_sha256={'synthetic-checker.py':'b'*64}))
        self.bind();return self.study

    def bind(self):
        spec=export.read(self.study/'contract.json')
        manifest={str(p.relative_to(self.study)):export.sha(p) for p in self.study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
        export.write_json(self.study/'evidence-manifest.json',manifest);self.report['evidence_sha256']=manifest
        export.write_json(self.study/'report.json',self.report)
        self.audit.update(report_sha256=export.sha(self.study/'report.json'),evidence_manifest_sha256=export.sha(self.study/'evidence-manifest.json'),
            contract_fingerprint=spec['fingerprint'],scene_fingerprint=spec['scene_fingerprint'],physical_scene_fingerprint=spec['physical_scene_fingerprint'],
            requested_trials=self.report['counts']['attempts'],native_releases=self.report['counts']['native_releases'],gate=copy.deepcopy(self.report['gate']))
        export.write_json(self.audit_path,self.audit)

    def run_export(self,name='publication'):
        return export.curate(self.study,self.audit_path,self.study.parent/name)

    def test189_rows_separate_coarse_verification_and_clean_control_counts(self):
        self.fixture();before={str(p.relative_to(self.study)):p.read_bytes() for p in self.study.rglob('*') if p.is_file()}
        summary=self.run_export();out=self.study.parent/'publication'
        self.assertEqual(summary['counts']['attempts'],189);self.assertEqual(summary['candidate_counts']['attempts'],165)
        self.assertEqual(summary['discovery_counts']['attempts'],135);self.assertEqual(summary['verification_counts']['attempts'],30)
        self.assertEqual(summary['candidate_counts']['captures'],165);self.assertEqual(summary['candidate_counts']['clean'],0)
        self.assertEqual(summary['control_counts']['attempts'],24);self.assertEqual(summary['control_counts']['clean'],12)
        self.assertEqual(summary['gate']['status'],'capture_only');self.assertEqual(summary['gate']['capture_qualified_settings'],5)
        self.assertEqual(summary['gate']['qualified_settings'],0)
        content=(out/'requested-shots.csv').read_bytes();self.assertNotIn(b'\r',content)
        rows=list(csv.DictReader(content.decode().splitlines()));self.assertEqual(len(rows),189)
        self.assertEqual(len({r['execution_fingerprint'] for r in rows}),189)
        self.assertEqual({r['board_center_distance_m'] for r in rows if r['phase']=='coarse'},{'0.65'})
        self.assertGreater(len({r['nominal_point_to_bucket_horizontal_m'] for r in rows if r['phase']=='coarse'}),5)
        self.assertGreater(len({r['nominal_release_center_z_m'] for r in rows if r['phase']=='coarse'}),5)
        self.assertEqual(tuple(p['record_id'] for p in summary['parent_sources']),export.SOURCE_IDS)
        for name,h in export.read(out/'checksums.json').items():self.assertEqual(export.sha(out/name),h)
        self.assertEqual(before,{str(p.relative_to(self.study)):p.read_bytes() for p in self.study.rglob('*') if p.is_file()})

    def test_fresh_clean_and_failed128_gates_do_not_relabel_source_history(self):
        self.fixture(mode='clean');summary=self.run_export();self.assertEqual(summary['gate']['qualified_settings'],5)
        self.assertEqual(summary['candidate_counts']['clean'],165)
        self.assertTrue(all(p['historical_route_quality']==1 for p in summary['parent_sources']))
        self.fixture(mode='failed');summary=self.run_export();self.assertEqual(summary['verification_counts']['captures'],15)
        self.assertEqual(summary['gate']['status'],'failed');self.assertEqual(summary['gate']['capture_qualified_settings'],0)

    def test_control_failure_and_geometry_rejection_preserve_denominators(self):
        self.fixture(blocked=True);summary=self.run_export();self.assertEqual(summary['counts']['attempts'],12)
        self.assertEqual(summary['candidate_counts']['attempts'],0);self.assertEqual(summary['verification'],[])
        self.assertEqual(summary['gate']['status'],'blocked_controls')
        self.fixture(reject_one=True);summary=self.run_export();self.assertEqual(summary['counts']['attempts'],189)
        self.assertEqual(summary['discovery_counts']['attempts'],135);self.assertEqual(summary['discovery_counts']['native_releases'],134)
        self.assertEqual(summary['discovery_counts']['geometry_rejections'],1)

    def test_fresh_capture_cannot_bypass_an_uncaptured_coarse_source(self):
        self.fixture(mode='uncaptured');summary=self.run_export()
        self.assertEqual(summary['discovery_counts']['captures'],0)
        self.assertEqual(summary['verification_counts']['captures'],30)
        self.assertEqual(summary['gate']['capture_qualified_settings'],0)
        self.assertTrue(all(not row['qualification']['source_capture'] for row in summary['verification']))

    def test_privacy_excludes_raw_points_contacts_joints_and_extra_secrets(self):
        self.fixture(private_xy=True);secret='DO-NOT-PUBLISH-secret-host-path'
        self.report.update(credentials={'token':secret},host_path='/private/'+secret,position_m=[[secret]])
        for row in self.report['groups']+self.report['verification']:row.update(contact_position_m=[[secret]],native_pose={'raw':secret},link_positions10=[[secret]])
        self.bind();summary=self.run_export()
        for path in (self.study.parent/'publication').iterdir():
            text=path.read_text();self.assertNotIn(secret,text);self.assertNotIn('PRIVATE-MEASURED-XY',text)
            for field in ('"q10"','"q14"','"position_m"','"contact_position_m"','"xy_m"'):self.assertNotIn(field,text)
        self.assertEqual(set(summary['verification'][0]['executions'][0]['first_flight']['first_crossing']),{'time_s','interval_index','fraction'})

    def test_exact_external_full_audit_bytes_and_complete_manifest_required(self):
        self.fixture();original=copy.deepcopy(self.audit)
        for edit in (dict(version='sircus-tilt30-650-independent-audit-v1'),dict(checks=[],checks_count=0),dict(checks_count=True),dict(checks_count=2),dict(checks=[{'passed':1}]),dict(status='failed')):
            self.audit=dict(original,**edit);export.write_json(self.audit_path,self.audit)
            with self.assertRaises(ValueError):self.run_export()
        self.audit=original;export.write_json(self.audit_path,self.audit)
        external=self.study.parent/'external-proof.json';external.write_bytes(self.audit_path.read_bytes())
        summary=export.curate(self.study,external,self.study.parent/'external-publication')
        self.assertEqual(summary['provenance']['independent_audit_sha256'],export.sha(external))
        report=(self.study/'report.json').read_bytes();(self.study/'report.json').write_bytes(report+b'\n')
        with self.assertRaisesRegex(ValueError,'Audit/report bytes'):self.run_export()
        (self.study/'report.json').write_bytes(report);(self.study/'calibration.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'Changed evidence bytes'):self.run_export()

    def test_wrong_budget_nominal_point_source_plan_and_gate_fail_closed(self):
        self.fixture();spec=export.read(self.study/'contract.json')
        for change in (dict(max_requests=201),dict(worker_wall_limit_s=7200),dict(forward_m=[0.]),dict(source_ids=list(reversed(export.SOURCE_IDS)))):
            with self.assertRaises(ValueError):export.settings(dict(spec['settings'],**change))
        self.report['elapsed_s']=3600.001;self.bind()
        with self.assertRaisesRegex(ValueError,'Worker-hour limit'):self.run_export()
        self.report['elapsed_s']=1.;self.bind()
        self.report['groups'][2]['impact_point_world_m'][2]+=.001;self.bind()
        with self.assertRaisesRegex(ValueError,'Nominal center-plane point'):self.run_export()
        self.fixture();self.report['verification'][0]['qualification'].update(qualified=True,clean_qualified=True,status='qualified');self.bind()
        with self.assertRaisesRegex(ValueError,'Capture and clean gates'):self.run_export()
        self.fixture();plan=export.read(self.study/'initial-plan.json');plan['coarse'].pop();export.write_json(self.study/'initial-plan.json',plan);self.bind()
        with self.assertRaisesRegex(ValueError,'Literal initial159'):self.run_export()

    def test_output_guards_finite_counts_and_execution_provenance(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError,'outside evidence'):export.curate(self.study,self.audit_path,self.study/'publication')
        records=[json.loads(line) for line in (self.study/'records.jsonl').read_text().splitlines()]
        records[0]['execution_fingerprint']=None;(self.study/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));self.bind()
        with self.assertRaises(ValueError):self.run_export()
        self.fixture();self.run_export()
        with self.assertRaisesRegex(ValueError,'New publication directory'):self.run_export()


if __name__=='__main__':unittest.main()
