"""Synthetic byte/denominator/privacy tests; no native physics is executed."""
import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import curate_physics_ensemble as c


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


class PhysicsEnsembleCurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.study.mkdir();self.audit=self.root/'proof.json';self.output=self.root/'public';self.patch=None
        self.addCleanup(lambda:self.patch.stop() if self.patch else None)
        self.models={}
        for method in c.METHODS:
            path=self.study/'model'/(method+'.pt');path.parent.mkdir(exist_ok=True);path.write_bytes((method+' synthetic checkpoint, never loaded').encode());self.models[method]=c.sha(path)
        self.splits={role:[dict(id=f'fresh-{role}-{i:02d}',split=role,bucket_xy_m=[.7+i*.001,-.4+(role=='test')*.1]) for i in range(size)] for role,size in (('validation',24),('test',60))}
        self.rows=[]
        for phase in c.PHASES:
            if phase.startswith('controls_'):
                for role in ('positive_control','negative_control'):
                    for repeat in range(3):self.rows.append(self.row(phase,role,'control',f'{phase}-{role}-{repeat}',[.8,-.2],True,role=='positive_control',True))
            else:
                for method in c.METHODS:
                    for i,context in enumerate(self.splits[phase]):
                        native=i%10!=0;capture=native and i<(20+c.METHODS.index(method)*3)
                        self.rows.append(self.row(phase,phase,method,context['id'],context['bucket_xy_m'],native,capture,i<4))
        self.bind()

    def row(self,phase,role,method,context,bucket,native,capture,clean):
        prediction=dict(admission_probability=.9,board_bounce_probability=.98,whole_entry_probability=.8,capture_probability=.7,
            conditional_capture_probability=.78,uncertainty=.03,retention_fraction=.7,aperture_distance_m=.04,ranking_score=.69,
            rebound_velocity_m_s=[1.,.1,2.],rebound_basis='board_to_bucket_forward_lateral_vertical',member_raw_output=['PRIVATE'])
        return dict(id=f'trial-{phase}-{method}-{context}',phase=phase,role=role,method=method,context_id=context,bucket_xy_m=bucket,
            values=[.5,-.2,.3,28.,.4],native_release=native,route_success=capture,route_quality=2 if capture and clean else 1 if capture else 0,
            invalid=None if native else 'finite_ik_unresolved',invalid_physics=False,whole_ball_entry=capture,max_retention_s=1.1 if capture else 0.,
            scalar_score_audit_passed=True,contact_protocol={'substeps':64},learning_reward=1. if capture else -.1,
            selection_or_learning=False,ball_flight_search_at_inference=False,action_substituted=False,perturbation_mixed_units=[0.]*5,
            frozen_artifact_sha256=self.models.get(method,'d'*64),model_prediction=prediction,q10=['PRIVATE'],q14=['PRIVATE'],position_m=['PRIVATE'],contact_arrays=['PRIVATE'])

    def bind(self):
        if self.patch:self.patch.stop();self.patch=None
        bindings=dict(parent_contract_fingerprint=c.PARENT['contract_fingerprint'],parent_report_sha256=c.PARENT['report_sha256'],parent_manifest_sha256=c.PARENT['manifest_sha256'],parent_audit_sha256=c.PARENT['original_audit_sha256'])
        save(self.study/'source-bindings.json',bindings);save(self.study/'splits.json',self.splits)
        curves=[dict(member=m,step=s,train_loss=.1,full_train_loss=.2) for m in range(5) for s in range(0,601,100)]
        artifacts={method:dict(path=f'model/{method}.pt',sha256=self.models[method],samples=0 if method=='bc' else 1024) for method in c.METHODS}
        artifacts['ensemble'].update(version=c.MODEL_VERSION,ensemble_members=5,fit_device='cpu',fit_steps=600,curves=curves)
        plan=[{k:copy.deepcopy(v) for k,v in r.items() if k not in ('native_release','route_success','route_quality','invalid','invalid_physics','whole_ball_entry','max_retention_s','learning_reward','position_m','contact_arrays','q10','q14','action_substituted','scalar_score_audit_passed','contact_protocol')} for r in self.rows]
        packet=dict(requests=plan,ensemble_artifact=artifacts['ensemble'],frozen_before_any_native_outcome=True,validation_selects_model=False,native_planning_queries=0)
        save(self.study/'prediction-plan.json',packet)
        protocol=dict(substeps=64,methods=list(c.METHODS),historical_current_train_requests=1024,new_train_requests=0,validation_layouts=24,test_layouts=60,validation_requests=72,test_requests=180,control_requests=12,maximum_fresh_requests=264,noise=False,no_test_flight_search=True,fixed_model_no_validation_selection=True,fit_steps=600,fit_device='cpu',ensemble_members=5,candidates=16,maximum_geometry_checks=4)
        spec=dict(version=c.STUDY_VERSION,protocol=protocol,parent_contract_fingerprint=c.PARENT['contract_fingerprint'],parent_bindings=bindings,source_sha256={'sircus/mjlab/producer.py':'a'*64},native_source_sha256={'sircus/mjlab/native.py':'b'*64},training_native_requests=0,validation_selects_model=False,splits=self.splits)
        spec['fingerprint']=c.digest(spec);save(self.study/'contract.json',spec)
        (self.study/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        manifest={str(p.relative_to(self.study)):c.sha(p) for p in self.study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')};save(self.study/'evidence-manifest.json',manifest)
        phases={phase:c.statistics([r for r in self.rows if r['phase']==phase]) for phase in c.PHASES}
        gates={key:c.control_gate([r for r in self.rows if r['phase']==phase]) for key,phase in (('pre','controls_pre'),('post','controls_post'))}
        comparisons=[]
        for role in ('validation','test'):
            for method in c.METHODS:
                cohort=[r for r in self.rows if r['role']==role and r['method']==method]
                comparisons.append(dict(role=role,method=method,counts=c.statistics(cohort),artifact_sha256=self.models[method],mean_reward=sum(r['learning_reward'] for r in cohort)/len(cohort)))
        positions=[]
        for row in self.rows:
            if row['role']!='test':continue
            cap=c.captured(row)
            positions.append(dict(record_id=row['id'],context_id=row['context_id'],method=row['method'],bucket_xy_m=row['bucket_xy_m'],values=row['values'],artifact_sha256=row['frozen_artifact_sha256'],model_prediction=row['model_prediction'],board_bucket_distance_m=math.dist(row['values'][:2],row['bucket_xy_m']),actual=dict(native_release=row['native_release'],capture=cap,clean=cap and row['route_quality']==2,invalid=row['invalid'],invalid_physics=row['invalid_physics'],whole_entry=row['whole_ball_entry'],retention_s=row['max_retention_s'],entry_class='clean_capture' if cap and row['route_quality']==2 else 'captured_nonclean' if cap else 'no_capture')))
        self.raw=dict(version=c.STUDY_VERSION,status='completed',phase='completed',contract_fingerprint=spec['fingerprint'],protocol=protocol,evidence_sha256=manifest,counts=c.statistics(self.rows),phase_counts=phases,source_sha256=spec['source_sha256'],artifacts=artifacts,control_gates=gates,comparisons=comparisons,test_comparison=[r for r in comparisons if r['role']=='test'],test_positions=positions,prediction_plan_sha256=c.sha(self.study/'prediction-plan.json'),
            dataset=dict(train_requests=1024,train_labels=1024,train_native_releases=887,train_layouts=35,excluded_prior_layouts=169,target_support=[1024]+[886]*8,source_contract_fingerprint=c.PARENT['contract_fingerprint'],source_report_sha256=c.PARENT['report_sha256'],source_audit_sha256=c.PARENT['original_audit_sha256'],split_hashes={'splits.json':c.sha(self.study/'splits.json')}),
            training=dict(status='completed',ensemble_members=5,fit_device='cpu',fit_steps=600,input_features=list(c.INPUTS),target_features=list(c.TARGETS),artifact=artifacts['ensemble'],curves=curves),prediction_metrics=[dict(split='test',target='requested_capture_uncalibrated_score',support=180,brier=.2,mae=.3,rmse=.4,unit='fraction',raw_pose=['PRIVATE'])],geometry_planning_checks=200,cpu_preparation_elapsed_s=100.,acquired_elapsed_s=500.,gate=dict(status='passed',complete=True,controls_passed=True,predictions_frozen_before_any_evaluation=True,no_test_flight_search=True),credentials={'token':'PRIVATE'},trace_examples=[dict(position_m=['PRIVATE'])])
        save(self.study/'report.json',self.raw)
        self.proof=dict(version=c.AUDIT_VERSION,status='passed',contract_fingerprint=spec['fingerprint'],checks_count=1,checks=[dict(passed=True,label='Synthetic fixture, not physics')],failures=[],report_sha256=c.sha(self.study/'report.json'),evidence_manifest_sha256=c.sha(self.study/'evidence-manifest.json'),source_parent=c.PARENT,producer_source_sha256=spec['source_sha256'],native_source_sha256=spec['native_source_sha256'],auditor_sha256='e'*64,requested_trials=264,native_releases=c.statistics(self.rows)['native_releases'],training_requests=1024,new_training_requests=0,training_masks=[1024]+[886]*8,control_gates=gates,comparisons=comparisons)
        save(self.audit,self.proof)
        self.patch=patch.multiple(c,SOURCE_FP=spec['fingerprint'],REPORT_SHA=c.sha(self.study/'report.json'),MANIFEST_SHA=c.sha(self.study/'evidence-manifest.json'),AUDIT_SHA=c.sha(self.audit),ARTIFACT_SHA=self.models);self.patch.start()

    def run_curate(self):return c.curate(self.study,self.audit,self.output)

    def test_full_sanitized_exports_exact180_rows_and_lf_with_immutable_inputs(self):
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()};result=self.run_curate()
        self.assertEqual(result['counts']['requested'],264);self.assertEqual(result['phase_counts']['test']['requested'],180);self.assertEqual(result['controls']['pre']['counts']['requested'],6)
        self.assertEqual(len(result['training']['curves']),35);self.assertEqual(result['training']['target_support']['native_execution'],1024)
        raw=(self.output/'test-settings.csv').read_bytes();self.assertNotIn(b'\r',raw);rows=list(csv.DictReader(raw.decode().splitlines()));self.assertEqual(len(rows),180)
        self.assertEqual(len({(r['context_id'],r['method']) for r in rows}),180);self.assertEqual(rows[0]['predicted_aperture_distance_m'],'0.04')
        for name,expected in c.read(self.output/'checksums.json').items():self.assertEqual(c.sha(self.output/name),expected)
        text='\n'.join(p.read_text() for p in self.output.iterdir())
        for forbidden in ('PRIVATE','q10','q14','position_m','contact_arrays','member_raw_output','/home/','/Users/'):
            self.assertNotIn(forbidden,text)
        self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_strict_v1_complete_checklist_and_exact_identity_pins(self):
        original=copy.deepcopy(self.proof)
        for change in (dict(version='old'),dict(checks=[],checks_count=0),dict(checks_count=True),dict(checks_count=2),dict(checks=[dict(passed=1,label='not bool')]),dict(failures=['bad'])):
            save(self.audit,dict(original,**change))
            with patch.object(c,'AUDIT_SHA',c.sha(self.audit)):
                with self.assertRaises(ValueError):self.run_curate()
        save(self.audit,original)
        for key in ('SOURCE_FP','REPORT_SHA','MANIFEST_SHA','AUDIT_SHA'):
            with patch.object(c,key,'f'*64):
                with self.assertRaises(ValueError):self.run_curate()

    def test_all_evidence_bytes_and_full_manifest_coverage_are_required(self):
        for file in (self.study/'report.json',self.study/'records.jsonl',self.study/'contract.json',self.study/'prediction-plan.json',self.study/'model/ensemble.pt',self.study/'model/bc.pt',self.study/'evidence-manifest.json',self.audit):
            old=file.read_bytes();file.write_bytes(old+b'\n')
            with self.assertRaises(ValueError):self.run_curate()
            file.write_bytes(old)
        (self.study/'unmanifested.bin').write_bytes(b'new')
        with self.assertRaisesRegex(ValueError,'every frozen'):self.run_curate()

    def test_ledger_and_same60_layout_denominators_cannot_be_replaced(self):
        original=copy.deepcopy(self.rows);self.rows[-1]['id']=self.rows[-2]['id'];self.bind()
        with self.assertRaisesRegex(ValueError,'264 unique'):self.run_curate()
        self.rows=copy.deepcopy(original);row=next(r for r in self.rows if r['role']=='test' and r['method']=='ppo');row['context_id']='unseen';self.bind()
        with self.assertRaisesRegex(ValueError,'Same layouts'):self.run_curate()
        self.rows=original;self.rows[0]['route_success']=False;self.rows[0]['route_quality']=0;self.bind()
        with self.assertRaisesRegex(ValueError,'control gates'):self.run_curate()

    def test_no_overwrite_and_source_artifact_or_frozen_decision_substitution(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):self.run_curate()
        for output in (self.study/'public',self.root):
            with self.assertRaises(ValueError):c.curate(self.study,self.audit,output)
        self.output.rmdir()
        with patch.object(c,'ARTIFACT_SHA',dict(self.models,ensemble='f'*64)):
            with self.assertRaisesRegex(ValueError,'checkpoint'):self.run_curate()


if __name__=='__main__':unittest.main()
