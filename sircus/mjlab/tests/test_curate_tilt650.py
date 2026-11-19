"""Synthetic fixed-plan export controls, never native or independent evidence."""

import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import curate_tilt650 as export
from sircus.mjlab import tilt650_contract as protocol,tilt650_study as study,xy650_contract as xy
from sircus.mjlab.tests import test_xy650_study as fixture


class TiltExportTests(unittest.TestCase):
    def fixture(self,*,mode='rim',blocked=False,private_xy=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        parent=root/'parent';parent.mkdir();assets=root/'assets';assets.mkdir();(root/'asset.txt').write_text('synthetic')
        for name in ('calibration.json','verification.json'):(parent/name).write_text('{}')
        calibration=dict(calibration_sha256='a'*64,verification_sha256='b'*64,score_config_sha256='c'*64,score_config={})
        identities={'asset.txt':export.sha(root/'asset.txt')}
        spec=dict(scene=fixture.SCENE,calibration=calibration,packages={'synthetic':'fixture'},asset_sha256=identities,
            scene_fingerprint=xy.scene_fingerprint(fixture.SCENE,calibration,identities),fingerprint=export.PARENT_FP)
        selected=[]
        for ident,(translation,gap) in zip(export.SOURCE_IDS,export.RECIPES):
            row=copy.deepcopy(fixture.sources()[1]);dx,dy=translation
            row.update(id=ident,phase='coarse',substeps=64,values=[.4708757793012148+dx,-.2778934014509746+dy,.38,30.,gap],
                bucket_xy_m=[-.030306598129941298+dx,-.6917970430784122+dy],translation_m=translation,
                query_fingerprint=export.digest({'parent_query':ident}),execution_fingerprint=export.digest({'parent_execution':ident}),
                route_quality=1,route_success=True,rim_band_contact=True,native_release=True,invalid=None,invalid_physics=False)
            selected.append(row)
        controls=xy.controls(fixture.sources())
        bindings=dict(parent={n:'a'*64 for n in ('contract.json','report.json','evidence-manifest.json','records.jsonl','groups.json','initial-groups.json')},parent_audit_sha256=export.PARENT_AUDIT_SHA)
        geometry=fixture.GeometryFixture()

        class Executor(fixture.ExecutorFixture):
            def execute(current,requests):
                before=len(current.rows);super().execute(requests)
                for r in current.rows[before:]:
                    if r['phase']=='verification':
                        success=not(mode=='failed' and r['substeps']==128)
                        r.update(route_success=success,route_quality=2 if mode=='clean' and success else 1 if success else 0,
                            whole_ball_entry=success,max_retention_s=1. if success else 0.,rim_band_contact=mode!='clean' and success,
                            minimum_mouth_clearance_mm=2. if success else None)
                    r['native_initial_q10']=copy.deepcopy(r['q10']);r['native_initial_q14']=copy.deepcopy(r['q14'])
                    r['execution_fingerprint']=xy.execution_fingerprint(r)
                    r['first_flight']=dict(version='sircus-xy650-first-flight-aperture-v1',crossing_plane='ball_center_at_rim',available=True,reason=None,
                        start_index=2,end_index=10,start_time_s=.02,end_time_s=.1,cutoff_reason='horizon',cutoff_contact_channels=[],sample_count=9,
                        first_crossing=dict(time_s=.05,interval_index=4,fraction=1.,xy_m=['PRIVATE-MEASURED-XY'] if private_xy else r['bucket_xy_m']),
                        mouth_clearance_m=.01,closest_opening_distance_m=.02)
                (current.output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in current.rows))

        def executor(*args):return Executor(*args,fail_initial=blocked)
        args=SimpleNamespace(parent_study=parent,parent_audit=root/'parent-proof.json',assets=assets,output=root/'evidence',batch=32,wall_time_limit=1800)
        with patch.object(protocol,'load_parent',return_value=(spec,selected,controls,bindings)), \
             patch.object(study.importlib.metadata,'version',return_value='fixture'), \
             patch('sircus.mjlab.learning_solve.required_asset_paths',return_value=['asset.txt']), \
             patch.object(study,'source_manifest',return_value={'synthetic.py':'f'*64}), \
             patch.object(study,'storage_check',return_value={'synthetic':True}), \
             patch('sircus.mjlab.clean650_geometry.CleanGeometry',return_value=geometry), \
             patch('sircus.mjlab.xy650_native.NativeExecutor',side_effect=executor),patch('builtins.print'):
            report=study.run(args)
        self.study,self.report=args.output,report;self.audit_path=self.study/'audit-tilt650.json'
        self.audit=dict(version=export.AUDIT_VERSION,status='passed',checks_count=1,
            checks=[dict(label='Synthetic exporter fixture; not a native independent audit',passed=True)],failures=[],auditor_sha256='a'*64,
            auditor_identity=dict(dependencies_sha256={'synthetic-checker.py':'b'*64}))
        self.bind();return self.study

    def bind(self):
        spec=export.read(self.study/'contract.json')
        manifest={str(p.relative_to(self.study)):export.sha(p) for p in self.study.rglob('*') if p.is_file()
            and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
        export.write_json(self.study/'evidence-manifest.json',manifest);self.report['evidence_sha256']=manifest
        export.write_json(self.study/'report.json',self.report)
        self.audit.update(report_sha256=export.sha(self.study/'report.json'),evidence_manifest_sha256=export.sha(self.study/'evidence-manifest.json'),
            contract_fingerprint=spec['fingerprint'],scene_fingerprint=spec['scene_fingerprint'],requested_trials=self.report['counts']['attempts'],
            native_releases=self.report['counts']['native_releases'],gate=copy.deepcopy(self.report['gate']))
        export.write_json(self.audit_path,self.audit)

    def run_export(self,name='publication'):return export.curate(self.study,self.audit_path,self.study.parent/name)

    def test_all54_rows_separate_candidate_rim_captures_from_clean_controls(self):
        self.fixture();before={str(p.relative_to(self.study)):p.read_bytes() for p in self.study.rglob('*') if p.is_file()}
        summary=self.run_export();out=self.study.parent/'publication'
        self.assertEqual(summary['counts']['attempts'],54);self.assertEqual(summary['candidate_counts']['attempts'],30)
        self.assertEqual(summary['candidate_counts']['captures'],30);self.assertEqual(summary['candidate_counts']['clean'],0)
        self.assertEqual(summary['control_counts']['attempts'],24);self.assertEqual(summary['control_counts']['clean'],12)
        self.assertEqual(summary['gate']['status'],'capture_only');self.assertEqual(summary['gate']['capture_qualified_settings'],5)
        self.assertEqual(summary['gate']['qualified_settings'],0)
        self.assertTrue(all(v['parent_route_quality']==1 and v['qualification']['capture_qualified'] and not v['qualification']['clean_qualified'] for v in summary['verification']))
        content=(out/'requested-shots.csv').read_bytes();self.assertNotIn(b'\r',content)
        rows=list(csv.DictReader(content.decode().splitlines()));self.assertEqual(len(rows),54)
        self.assertEqual(len({r['execution_fingerprint'] for r in rows}),54)
        self.assertEqual({r['parent_route_quality'] for r in rows if r['phase']=='verification'},{'1'})
        self.assertEqual(tuple(p['record_id'] for p in summary['parent_sources']),export.SOURCE_IDS)
        for name,h in export.read(out/'checksums.json').items():self.assertEqual(export.sha(out/name),h)
        self.assertEqual(before,{str(p.relative_to(self.study)):p.read_bytes() for p in self.study.rglob('*') if p.is_file()})

    def test_fresh_clean_qualification_does_not_relabel_historical_parent(self):
        self.fixture(mode='clean');summary=self.run_export()
        self.assertEqual(summary['gate']['qualified_settings'],5)
        self.assertEqual(summary['candidate_counts']['clean'],30)
        self.assertTrue(all(p['parent_route_quality']==1 for p in summary['parent_sources']))

    def test_128_misses_fail_both_capture_and_clean_and_control_failure_keeps_candidates_closed(self):
        self.fixture(mode='failed');summary=self.run_export()
        self.assertEqual(summary['candidate_counts']['captures'],15);self.assertEqual(summary['gate']['status'],'failed')
        self.assertTrue(all(not v['qualification']['capture_qualified'] for v in summary['verification']))
        self.fixture(blocked=True);summary=self.run_export()
        self.assertEqual(summary['counts']['attempts'],12);self.assertEqual(summary['candidate_counts']['attempts'],0)
        self.assertEqual(summary['gate']['status'],'blocked_controls')
        self.assertEqual(len(summary['verification']),5)
        self.assertTrue(all(v['qualification']['status']=='pending' for v in summary['verification']))

    def test_privacy_drops_measured_crossing_xy_raw_joints_poses_and_extra_fields(self):
        self.fixture(private_xy=True);secret='DO-NOT-PUBLISH-private-path'
        self.report.update(credentials={'token':secret},host_path='/home/private/'+secret,position_m=[[secret]])
        for r in self.report['groups']+self.report['verification']:r.update(link_positions10=[[secret]],native_pose={'raw':secret})
        self.bind();summary=self.run_export()
        for p in (self.study.parent/'publication').iterdir():
            text=p.read_text();self.assertNotIn(secret,text);self.assertNotIn('PRIVATE-MEASURED-XY',text)
            self.assertNotIn('"xy_m"',text);self.assertNotIn('"q10"',text);self.assertNotIn('"position_m"',text)
        self.assertEqual(set(summary['verification'][0]['executions'][0]['first_flight']['first_crossing']),{'time_s','interval_index','fraction'})

    def test_old_checker_partial_audit_byte_tampering_and_external_v1_proof(self):
        self.fixture();original=copy.deepcopy(self.audit)
        for edit in (dict(version='sircus-xy650-independent-audit-v2'),dict(checks=[],checks_count=0),dict(checks_count=2),dict(checks=[{'passed':1}]),dict(status='failed')):
            self.audit=dict(original,**edit);export.write_json(self.audit_path,self.audit)
            with self.assertRaises(ValueError):self.run_export()
        self.audit=original;export.write_json(self.audit_path,self.audit)
        external=self.study.parent/'passed-proof.json';external.write_bytes(self.audit_path.read_bytes())
        self.run_export('external')
        summary=export.curate(self.study,external,self.study.parent/'external-explicit')
        self.assertEqual(summary['provenance']['independent_audit_sha256'],export.sha(external))
        report=(self.study/'report.json').read_bytes();(self.study/'report.json').write_bytes(report+b'\n')
        with self.assertRaisesRegex(ValueError,'Audit/report bytes'):self.run_export()
        (self.study/'report.json').write_bytes(report);(self.study/'calibration.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'Changed evidence bytes'):self.run_export()

    def test_literal54_source_geometry_plan_parent_proof_and_gate_guards(self):
        self.fixture();spec=export.read(self.study/'contract.json')
        for edit in (dict(max_requests=60),dict(source_ids=list(reversed(export.SOURCE_IDS))),dict(verification_requests=29)):
            with self.assertRaises(ValueError):export.settings(dict(spec['settings'],**edit))
        source=export.read(self.study/'source-selection.json');source[0]['route_quality']=2
        export.write_json(self.study/'source-selection.json',source);self.bind()
        with self.assertRaisesRegex(ValueError,'Historical source rim capture'):self.run_export()
        source[0]['route_quality']=1;export.write_json(self.study/'source-selection.json',source)
        plan=export.read(self.study/'initial-plan.json');plan['verification'].pop()
        export.write_json(self.study/'initial-plan.json',plan);self.bind()
        with self.assertRaisesRegex(ValueError,'Literal54-request plan'):self.run_export()

    def test_missing_execution_hashes_or_forged_fresh_clean_labels_fail(self):
        self.fixture();records=[json.loads(s) for s in (self.study/'records.jsonl').read_text().splitlines()]
        original=copy.deepcopy(records);records[0]['execution_fingerprint']=None
        (self.study/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));self.bind()
        with self.assertRaises(ValueError):self.run_export()
        (self.study/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in original))
        self.report['verification'][0]['qualification'].update(clean_qualified=True,qualified=True,status='qualified');self.bind()
        with self.assertRaisesRegex(ValueError,'Capture and clean gates'):self.run_export()

    def test_output_cannot_overwrite_evidence_or_existing_publication_and_counts_stay_finite(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError,'outside evidence'):export.curate(self.study,self.audit_path,self.study/'publication')
        self.report['counts']['attempts']=55;self.bind()
        with self.assertRaisesRegex(ValueError,'trial accounting'):self.run_export()
        self.report['counts']['attempts']=54;self.bind();self.run_export()
        with self.assertRaisesRegex(ValueError,'New publication directory'):self.run_export()


if __name__=='__main__':unittest.main()
