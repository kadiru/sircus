import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import gap_coverage_study as study
from sircus.mjlab import physics_ensemble_study as old
from sircus.mjlab.prepare_gap_coverage_report import compact,prepare,ACTOR_COMPARISON_REVISION

class FrozenStudyTests(unittest.TestCase):
    def test_split_repeatability_excludes_used_positions(self):
        previous=dict(excluded=[[.1,-.65],[1.24,.25]],splits=dict(test=[dict(bucket_xy_m=[.55,-.2])]))
        a,excluded=study.fresh_splits(previous);b,_=study.fresh_splits(previous)
        self.assertEqual(a,b);self.assertEqual(len(a['test']),50)
        pts=np.array([c['bucket_xy_m'] for c in a['test']])
        self.assertGreaterEqual(np.linalg.norm(pts[:,None]-np.array(excluded),axis=-1).min(),.01)
        self.assertGreaterEqual((np.linalg.norm(pts[:,None]-pts[None,:],axis=-1)+np.eye(50)).min(),.01)

    def execute_mock(self,foreign=False,tamper=False):
        tmp=TemporaryDirectory();self.addCleanup(tmp.cleanup);out=Path(tmp.name)
        (out/'prediction-plan.json').write_text('{}');frozen=study.sha(out/'prediction-plan.json')
        (out/'model.pt').write_bytes(b'model');artifact=dict(path='model.pt',sha256=study.sha(out/'model.pt'))
        requests=[]
        for phase in ('controls_pre','test','controls_post'):
            if phase=='test':
                for m in study.METHODS:
                    for i in range(50):requests.append(dict(id=f'{m}{i}',role='test',phase=phase,method=m,context_id=f'c{i}',bucket_xy_m=[.5,-.2],values=[.6,-.3,.3,30,.2]))
            else:
                for role in ('positive_control','negative_control'):
                    for i in range(3):requests.append(dict(id=f'{phase}{role}{i}',role=role,phase=phase,method='control'))
        calls=[]
        class Engine:
            def evaluate(self,rs,phase):
                calls.append((phase,[r['id'] for r in rs]));rows=[]
                for r in rs:
                    success=r['role']!='negative_control'
                    rows.append(dict(r,native_release=True,route_success=success,route_quality=2 if success else 0,
                        invalid=None,invalid_physics=False,scalar_score_audit_passed=True,whole_ball_entry=success,max_retention_s=1.2 if success else 0))
                return rows,dict(wall_s=0)
        state=dict(output=out,engine=Engine(),report=dict(artifacts={m:artifact for m in study.METHODS},control_gates={},comparisons=[],test_positions=[]),requests=requests,prediction_sha256=frozen)
        if tamper:(out/'prediction-plan.json').write_text('{"changed":true}')
        api=SimpleNamespace(gpu=SimpleNamespace(foreign_processes=lambda:[123] if foreign else []),reward=lambda r:int(r['route_success']))
        with patch.object(study.shutil,'disk_usage',return_value=SimpleNamespace(free=20*1024**3)):
            if foreign or tamper:
                with self.assertRaises((RuntimeError,ValueError)):study.execute(state,api,old,SimpleNamespace(wall_time_limit=3600))
                self.assertEqual(calls,[])
            else:
                study.execute(state,api,old,SimpleNamespace(wall_time_limit=3600))
                self.assertEqual(sum(len(ids) for _,ids in calls),112)
                self.assertEqual(state['report']['status'],'completed')
                self.assertEqual(len(state['report']['test_positions']),100)
                self.assertEqual([r['counts']['requested'] for r in state['report']['comparisons']],[50]*2)
    def test_bounded_mock_execution(self):self.execute_mock()
    def test_foreign_job_no_execution(self):self.execute_mock(foreign=True)
    def test_tampered_frozen_plan_no_execution(self):self.execute_mock(tamper=True)

class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.study=self.root/'study';self.study.mkdir()
        self.base=self.root/'base.json';self.base.write_text('{"old_result":{"capture":false}}')
        self.spec=dict(version=study.VERSION,protocol=study.protocol(),maximum_requests=112,new_training_requests=0,
            source_sha256={'sircus/mjlab/gap_coverage_candidates.py':'d'*64},native_source_sha256={'engine':'e'*64})
        self.spec['fingerprint']=study.digest(self.spec);study.atomic(self.study/'contract.json',self.spec)
        self.raw=dict(version=study.VERSION,status='queued',phase='waiting_for_4090',scope=study.SCOPE,protocol=study.protocol(),
            contract_fingerprint=self.spec['fingerprint'],counts=old.counts([]),comparisons=[],source_report_sha256='a'*64,
            selector_sha256='b'*64,artifacts=dict(ensemble=dict(sha256='c'*64)),queue=dict(wait_s=3))
        study.atomic(self.study/'report.json',self.raw)
    def test_queue_preserves_old_data_and_missing_results(self):
        result=prepare(self.base,self.study,self.root/'display.json')
        self.assertEqual(result['old_result'],dict(capture=False));self.assertEqual(result['gap_coverage_study']['comparisons'],[])
        self.assertIsNone(result['gap_coverage_study']['audit']['passed'])
    def test_no_in_place_evidence_or_display_write(self):
        for dest in (self.base,self.study/'new.json'):
            with self.assertRaises(ValueError):prepare(self.base,self.study,dest)
    def completed_fixture(self):
        counts=lambda requested,native,capture,clean:dict(requested=requested,native_releases=native,
            captures=capture,clean=clean,rejected=requested-native,native_misses=native-capture,invalid=0)
        comparisons=[dict(method='consistent',counts=counts(50,48,30,12)),
                     dict(method='gap_coverage',counts=counts(50,47,32,13))]
        gates={k:dict(status='passed',counts=counts(6,6,3,3)) for k in ('pre','post')}
        groups=[c['counts'] for c in comparisons]+[g['counts'] for g in gates.values()]
        aggregate={k:sum(c[k] for c in groups) for k in groups[0]}
        positions=[]
        for comparison in comparisons:
            method=comparison['method'];count=comparison['counts']
            for i in range(50):
                captured=i<count['captures'];native=i<count['native_releases']
                positions.append(dict(context_id=f'gap-coverage-test-{i:02d}',method=method,
                    record_id=f'gap-coverage-{method}-gap-coverage-test-{i:02d}',bucket_xy_m=[.1+.01*i,-.2],
                    values=[.5,-.2,.3,28.,.2],capture=captured,native_release=native,
                    clean=i<count['clean'],whole_ball_entry=captured,retention_s=1.1 if captured else 0.))
        self.raw.update(status='completed',phase='completed',counts=aggregate,comparisons=comparisons,
                        control_gates=gates,test_positions=positions)
        study.atomic(self.study/'report.json',self.raw);study.atomic(self.study/'evidence-manifest.json',{})
        labels=[f'gap-coverage-test-{i:02d}/{method}/{name}' for i in range(50) for method in ('bc','ppo')
            for name in ('independent actor forward','independent actor residual','saved tanh derivation',
                         'saved action derivation','normalized independent action')]
        proof=dict(version='sircus-gap-coverage-audit-v2',actor_comparison_revision=copy.deepcopy(ACTOR_COMPARISON_REVISION),
            status='passed',checks_count=len(labels),checks=[dict(passed=True,label=label) for label in labels],failures=[],
            auditor_sha256=study.sha(Path(study.__file__).with_name('gap_coverage_verify.py')),
            report_sha256=study.sha(self.study/'report.json'),evidence_manifest_sha256=study.sha(self.study/'evidence-manifest.json'),
            contract_fingerprint=self.spec['fingerprint'],requested_trials=112,native_releases=aggregate['native_releases'],
            training_requests=0,comparisons=copy.deepcopy(comparisons),control_gates=copy.deepcopy(gates),
            producer_source_sha256=self.spec['source_sha256'],native_source_sha256=self.spec['native_source_sha256'])
        return proof

    def test_completed_audit_binding_and_tamper(self):
        proof=self.completed_fixture();audit=self.root/'audit.json';study.atomic(audit,proof)
        self.assertEqual(compact(self.study,audit)['status'],'verified')
        for key,bad in [('checks_count',0),('report_sha256','bad'),('requested_trials',111),('training_requests',1),
                        ('native_source_sha256',{}),('native_releases',106),('control_gates',{}),
                        ('checks',[dict(passed=True)]),('checks',[dict(passed=True,label='')]),
                        ('actor_comparison_revision',{}),('version','sircus-gap-coverage-audit-v1'),('auditor_sha256','bad')]:
            changed=dict(proof);changed[key]=bad;study.atomic(audit,changed)
            with self.subTest(key=key),self.assertRaises(ValueError):compact(self.study,audit)

    def test_incomplete_or_inconsistent_report_cannot_be_verified_by_matching_proof(self):
        proof=self.completed_fixture();baseline=copy.deepcopy(self.raw);audit=self.root/'audit.json'
        for kind in ('empty_comparisons','wrong_method','missing_method','unequal_budget','partial_counts',
                     'invalid_counts','bool_count','pre_failed','post_missing','wrong_control_counts',
                     'aggregate_mismatch','wrong_phase','missing_native_proof'):
            raw=copy.deepcopy(baseline);changed=copy.deepcopy(proof)
            if kind=='empty_comparisons':raw['comparisons']=[]
            elif kind=='wrong_method':raw['comparisons'][1]['method']='ppo'
            elif kind=='missing_method':raw['comparisons'].pop()
            elif kind=='unequal_budget':
                raw['comparisons'][0]['counts']['requested']=49;raw['comparisons'][0]['counts']['rejected']=1
            elif kind=='partial_counts':raw['counts']=old.counts([])
            elif kind=='invalid_counts':raw['counts']['rejected']=-1
            elif kind=='bool_count':raw['counts']['invalid']=False
            elif kind=='pre_failed':raw['control_gates']['pre']['status']='failed'
            elif kind=='post_missing':raw['control_gates'].pop('post')
            elif kind=='wrong_control_counts':
                raw['control_gates']['pre']['counts']['captures']=2
                raw['control_gates']['pre']['counts']['native_misses']=4
            elif kind=='aggregate_mismatch':
                raw['counts']['captures']+=1;raw['counts']['native_misses']-=1
            elif kind=='wrong_phase':raw['phase']='test'
            else:changed.pop('native_releases')
            study.atomic(self.study/'report.json',raw)
            changed.update(report_sha256=study.sha(self.study/'report.json'),
                comparisons=copy.deepcopy(raw['comparisons']),control_gates=copy.deepcopy(raw['control_gates']))
            study.atomic(audit,changed)
            with self.subTest(kind=kind),self.assertRaises(ValueError):compact(self.study,audit)

    def test_paired_rows_are_copied_exactly_only_after_successful_audit(self):
        proof=self.completed_fixture();audit=self.root/'audit-v2.json'
        self.assertNotIn('test_positions',compact(self.study))
        study.atomic(audit,dict(version='sircus-gap-coverage-audit-v1',status='failed',failures=['retained failure']))
        self.assertNotIn('test_positions',compact(self.study,audit))
        study.atomic(audit,proof)
        result=compact(self.study,audit)
        self.assertEqual(result['test_positions'],self.raw['test_positions'])
        self.assertEqual(len(result['test_positions']),100)
        result['test_positions'][0]['values'][0]=999
        self.assertNotEqual(self.raw['test_positions'][0]['values'][0],999)

    def test_numerical_revision_and_tolerances_are_explicit_and_bound(self):
        from sircus.mjlab.gap_coverage_verify import ACTOR_COMPARISON_REVISION as independent_contract
        self.assertEqual(ACTOR_COMPARISON_REVISION,independent_contract)
        proof=self.completed_fixture();audit=self.root/'audit.json'
        for key,bad in [('action_scales',[.04,.04,.05,10.,.12]),('independent_latent_atol',1e-5),
                        ('normalized_action_atol',1e-5),('saved_tanh_atol',2e-6),('saved_action_atol',2e-6)]:
            changed=copy.deepcopy(proof);changed['actor_comparison_revision'][key]=bad
            study.atomic(audit,changed)
            with self.subTest(key=key),self.assertRaises(ValueError):compact(self.study,audit)

    def test_each_actor_derivation_is_required_at_every_layout(self):
        proof=self.completed_fixture();audit=self.root/'audit.json'
        self.assertEqual(len(proof['checks']),500)
        for index in (0,19,49):
            for method in ('bc','ppo'):
                for name in ('independent actor forward','independent actor residual','saved tanh derivation',
                             'saved action derivation','normalized independent action'):
                    label=f'gap-coverage-test-{index:02d}/{method}/{name}'
                    changed=copy.deepcopy(proof)
                    changed['checks']=[c for c in changed['checks'] if c['label']!=label]
                    changed['checks_count']=len(changed['checks'])
                    study.atomic(audit,changed)
                    with self.subTest(label=label),self.assertRaises(ValueError):compact(self.study,audit)

    def test_duplicate_label_cannot_replace_a_missing_actor_check(self):
        proof=self.completed_fixture();proof['checks'][-1]=copy.deepcopy(proof['checks'][0])
        audit=self.root/'audit.json';study.atomic(audit,proof)
        with self.assertRaises(ValueError):compact(self.study,audit)

    def test_metadata_only_v2_relabel_cannot_publish_verified(self):
        proof=self.completed_fixture();proof.update(checks=[dict(passed=True,label='metadata only')],checks_count=1)
        audit=self.root/'audit.json';study.atomic(audit,proof)
        with self.assertRaises(ValueError):compact(self.study,audit)

    def test_original_failed_v1_proof_remains_failed(self):
        self.completed_fixture();audit=self.root/'audit-v1.json'
        proof=dict(version='sircus-gap-coverage-audit-v1',status='failed',failures=['original mixed-unit comparison'])
        study.atomic(audit,proof);before=audit.read_bytes()
        self.assertIs(compact(self.study,audit)['audit']['passed'],False)
        self.assertEqual(audit.read_bytes(),before)

    def test_empty_proof_and_protocol_changes_refused(self):
        audit=self.root/'audit.json';study.atomic(audit,dict(status='passed',version='sircus-gap-coverage-audit-v1',checks=[]))
        with self.assertRaises(ValueError):compact(self.study,audit)
        self.raw['protocol']['candidates']=17;study.atomic(self.study/'report.json',self.raw)
        with self.assertRaises(ValueError):compact(self.study)

if __name__=='__main__':unittest.main()
