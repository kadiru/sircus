"""Stdlib synthetic export fixtures; no actual model, physics or audit is run."""
import copy
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import curate_policy64 as export


def digest_text(value):return hashlib.sha256(value.encode()).hexdigest()


class Policy64ExportTests(unittest.TestCase):
    def fixture(self,*,select_zero=False,zero_test=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.study=self.root/'study';self.study.mkdir();self.audit_path=self.root/'study-audit.json'
        for name in ('bootstrap','checkpoints'):(self.study/name).mkdir()
        splits={}
        for role,n in (('train',35),('validation',12),('test',50)):
            splits[role]=[]
            for i in range(n):
                xy=([.1+.05*(i%7),-.65+.05*(i//7)] if role=='train' else [.65+.03*(i%4),-.6+.03*(i//4)] if role=='validation' else [.1+.1*(i%10),-.3+.1*(i//10)])
                splits[role].append(dict(id=f'{role}-{i:02d}',split=role,bucket_xy_m=xy))
        self.splits=splits;export.write_json(self.study/'splits.json',splits)
        data=[]
        for i in range(6144):
            context=splits['train'][i%35];success=i<22
            data.append(dict(id=f'historical-{i:05d}',role='train',context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],native_release=True,route_success=success,route_quality=2 if i<11 else 1 if success else 0,invalid=None,invalid_physics=False,learning_reward=1. if success else 0.,q10=['RAW-PRIVATE-JOINTS']*6))
        teachers=data[:22];export.write_json(self.study/'bootstrap/dataset.json',data);export.write_json(self.study/'bootstrap/teachers.json',teachers)
        self.artifacts={}
        for samples in export.MILESTONES:
            path=f'checkpoints/milestone-{samples:04d}.pt'
            (self.study/path).write_bytes(f'SYNTHETIC-CHECKPOINT-BYTES-NOT-A-MODEL:{samples}'.encode())
            self.artifacts[samples]=dict(path=path,sha256=export.sha(self.study/path),weight_sha256=digest_text(f'weights-{samples}'),samples=samples)
        selected_samples=0 if select_zero else 512;selected=self.artifacts[selected_samples]
        definitions={}
        for method in ('prior','nearest'):
            path=f'checkpoints/{method}.json';export.write_json(self.study/path,{'method':method,'private_value':'DO-NOT-EXPORT-PRIVATE-METADATA'})
            definitions[method]=dict(path=path,sha256=export.sha(self.study/path),samples=0)
        p=dict(substeps=64,seed=20261511,train_layouts=35,validation_layouts=12,test_layouts=50,train_requests=1024,batch_size=32,updates=32,milestones=list(export.MILESTONES),test_methods=list(export.METHODS),test_repeats=1,validation_requests=48,test_requests=200,control_requests=12,maximum_fresh_requests=1284,noise=False,actor_steps=800,critic_steps=400,bootstrap_lr=.001,ppo_lr=.0003,initial_logstd=-3.,ppo_epochs=6,clip=.2,entropy=.001,gradient_norm=1.)
        boot=dict(requests=6144,native_releases=6144,captures=22,clean=11,layouts=35,successful_layouts=22,teacher_count=22,labels_scope='Historical TRAIN only')
        bindings=dict(parent_contract_fingerprint=export.SOURCE_FP,report_sha256=export.SOURCE_REPORT_SHA,parent_audit_sha256=export.SOURCE_AUDIT_SHA,parent_manifest_sha256=export.SOURCE_MANIFEST_SHA,
            train_files_sha256={f'runs/{m}-seed{s}/rows.jsonl':digest_text(f'{m}-{s}') for m in ('ppo','sweep') for s in (20261411,20261412,20261413)},teacher_ids=[r['id'] for r in teachers])
        scene=dict(ball=dict(radius_m=.0375),raw_private_joint_bank=['DO-NOT-EXPORT-PRIVATE-METADATA'])
        self.spec=dict(version=export.STUDY_VERSION,protocol=p,evidence_kind='native_fixed64_success_bootstrap_ppo',scene=scene,packages={'torch':'2.7.0+cu128','numpy':'2.2.0'},splits=splits,bootstrap_summary=boot,source_bindings=bindings,
            source_sha256={'sircus/mjlab/policy64_study.py':'a'*64},identity=dict(asset_sha256={'robot.urdf':'b'*64},scene_sha256=export.digest(scene),setup=dict(home=['DO-NOT-EXPORT-PRIVATE-METADATA'])),
            bootstrap_dataset_sha256=export.sha(self.study/'bootstrap/dataset.json'),teachers_sha256=export.sha(self.study/'bootstrap/teachers.json'))
        self.spec['fingerprint']=export.digest(self.spec);export.write_json(self.study/'contract.json',self.spec)
        self.rows=[]
        def row(phase,role,method,index,*,samples=0,success=False,clean=False,rejected=False):
            context=splits[role][index%len(splits[role])] if role in splits else dict(id='control-'+role,bucket_xy_m=[.3,-.7])
            a=selected if method=='ppo' and role=='test' else self.artifacts[samples] if method in ('ppo','bc') else definitions.get(method)
            result=dict(id=f'{phase}-{method}-{samples}-{role}-{index:05d}',phase=phase,role=role,method=method,samples=samples,context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],
                values=[.45,-.1,.38,28.,.4],command_values=[.45,-.1,.38,28.,.4],native_release=not rejected,route_success=success and not rejected,route_quality=2 if clean and success else 1 if success else 0,
                invalid='strict_endpoint_rejected' if rejected else None,invalid_physics=False,learning_reward=1. if success else -1. if rejected else 0.,whole_ball_entry=success,
                action_substituted=False,perturbation_mixed_units=[0.]*5,scalar_score_audit_passed=True,q10=['RAW-PRIVATE-JOINTS']*6,trajectory_m=[['DO-NOT-EXPORT-PRIVATE-METADATA']])
            if role=='train':result['train_index']=index
            if role in ('validation','test'):result.update(ball_flight_search_at_inference=False,selection_or_learning=False,artifact_sha256=a['sha256'],frozen_artifact_sha256=a['sha256'])
            self.rows.append(result)
            return result
        def controls(phase):
            for role in ('positive_control','negative_control'):
                for i in range(3):row(phase,role,'control',i,success=role=='positive_control',clean=role=='positive_control')
        validation_captures={0:7 if select_zero else 2,256:4,512:6,1024:5}
        def validation(samples):
            for i in range(12):row('validation','validation','ppo',i,samples=samples,success=i<validation_captures[samples],clean=i==0)
        controls('controls_pre');validation(0)
        for offset in range(0,1024,32):
            for i in range(offset,offset+32):row('train','train','ppo',i,success=i%5==0)
            if offset+32 in export.MILESTONES:validation(offset+32)
        for method,n in zip(export.METHODS,(5,8,12,15)):
            for i in range(50):row('test','test',method,i,samples=selected_samples if method=='ppo' else 0,success=not zero_test and i<n,clean=not zero_test and i<3,rejected=method=='ppo' and i==49)
        controls('controls_post')
        train=[r for r in self.rows if r['role']=='train'];self.updates=[dict(index=i,train_ids=[r['id'] for r in train[32*i:32*i+32]]) for i in range(32)]
        (self.study/'updates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.updates))
        curves=[dict(samples=s,validation=export.statistics([r for r in self.rows if r['role']=='validation' and r['samples']==s]),artifact=self.artifacts[s]) for s in export.MILESTONES]
        comparisons=[]
        for method in export.METHODS:
            a=selected if method=='ppo' else self.artifacts[0] if method=='bc' else definitions[method]
            comparisons.append(dict(method=method,counts=export.statistics([r for r in self.rows if r['role']=='test' and r['method']==method]),samples=a['samples'],artifact_sha256=a['sha256']))
        gates={}
        for phase,key in (('controls_pre','pre'),('controls_post','post')):
            gates[key]=dict(status='passed',counts=export.statistics([r for r in self.rows if r['phase']==phase]),positive_captures=3,negative_captures=0)
        self.report=dict(version=export.STUDY_VERSION,status='completed',phase='completed',contract_fingerprint=self.spec['fingerprint'],protocol=p,bootstrap_summary=boot,train_requests=1024,updates=32,elapsed_s=10.,
            counts=export.statistics(self.rows),curves=curves,selected=selected,artifacts=dict(bc=self.artifacts[0],selected=selected,**definitions),test_comparison=comparisons,test_positions=[],control_gates=gates,
            gate=dict(status='passed',controls_passed=True,complete=True,substeps=64,scope='Protocol/controls completion, not a128convergence or minimum success-rate gate'),private_path='/home/DO-NOT-EXPORT-PRIVATE-METADATA')
        self.positions()
        self.audit=dict(version=export.AUDIT_VERSION,status='passed',checks_count=1,checks=[{'passed':True,'label':'Synthetic exporter fixture, not a native independent audit'}],failures=[],auditor_sha256='c'*64,auditor_identity=dict(dependencies_sha256={'synthetic-checker.py':'d'*64}),source_contract_fingerprint=export.SOURCE_FP)
        self.bind()

    def positions(self):
        self.report['test_positions']=[dict(context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],method=r['method'],values=r['values'],board_bucket_distance_m=math.dist(r['values'][:2],r['bucket_xy_m']),capture=export.capture(r),clean=export.capture(r) and r['route_quality']==2,native_release=r['native_release'],invalid=r['invalid'],invalid_physics=r['invalid_physics']) for r in self.rows if r['role']=='test']

    def bind(self):
        (self.study/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        export.write_json(self.study/'selected.json',dict(selected=self.report['selected']))
        export.write_json(self.study/'test-plan.json',dict(selected=self.report['selected'],frozen_before_outcomes=True))
        manifest={str(p.relative_to(self.study)):export.sha(p) for p in self.study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
        export.write_json(self.study/'evidence-manifest.json',manifest);self.report['evidence_sha256']=manifest;export.write_json(self.study/'report.json',self.report)
        self.audit.update(report_sha256=export.sha(self.study/'report.json'),evidence_manifest_sha256=export.sha(self.study/'evidence-manifest.json'),contract_fingerprint=self.spec['fingerprint'],producer_source_sha256=self.spec['source_sha256'],requested_trials=len(self.rows),native_releases=sum(r['native_release'] for r in self.rows),bootstrap_summary={k:v for k,v in self.spec['bootstrap_summary'].items() if k!='labels_scope'},selected=copy.deepcopy(self.report['selected']),test_comparison=copy.deepcopy(self.report['test_comparison']),control_gates=copy.deepcopy(self.report['control_gates']),role_counts={role:export.statistics([r for r in self.rows if r['role']==role]) for role in ('train','validation','test','positive_control','negative_control')})
        export.write_json(self.audit_path,self.audit)

    def run_export(self,name='publication'):return export.curate(self.study,self.audit_path,self.root/name)

    def test_complete1284_accounting_and_two_exact_checkpoint_copies(self):
        self.fixture();before={str(p.relative_to(self.study)):p.read_bytes() for p in self.study.rglob('*') if p.is_file()};s=self.run_export();out=self.root/'publication'
        self.assertEqual(s['counts']['attempts'],1284);self.assertEqual(s['counts_by_role']['train']['attempts'],1024);self.assertEqual(s['counts_by_role']['validation']['attempts'],48);self.assertEqual(s['counts_by_role']['test']['attempts'],200)
        self.assertEqual(s['bootstrap_summary']['requests'],6144);self.assertEqual(s['bootstrap_summary']['teacher_count'],22);self.assertEqual(s['selected']['samples'],512)
        self.assertEqual(s['portable_policy_contract']['inputs']['size'],9);self.assertEqual(s['portable_policy_contract']['action']['size'],5);self.assertEqual(s['portable_policy_contract']['action']['residual_scale'],[.04,.04,.05,8.,.12])
        self.assertEqual(s['portable_policy_contract']['actor']['state_prefix'],'actor.');self.assertIn('not a complete arm setup resolver',s['portable_policy_contract']['required_external_geometry'])
        self.assertIn('strip the actor. prefix',s['portable_policy_contract']['actor']['loader']);self.assertEqual(s['portable_policy_contract']['full_training_state']['critic_layers'],[9,64,64,1])
        self.assertEqual({p.name for p in out.iterdir()},{'summary.json','test-positions.csv','bc-policy.pt','selected-policy.pt','checksums.json'})
        for role in ('bc','selected'):
            a=s['checkpoints'][role];self.assertEqual(export.sha(out/a['file']),a['sha256']);self.assertEqual((out/a['file']).read_bytes(),(self.study/self.report['artifacts'][role]['path']).read_bytes())
        with (out/'test-positions.csv').open() as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),200);self.assertEqual(Counter(r['method'] for r in rows),Counter({m:50 for m in export.METHODS}))
        self.assertEqual(sum(r['native_release']=='False' for r in rows),1)
        for name,value in export.read(out/'checksums.json').items():self.assertEqual(export.sha(out/name),value)
        self.assertEqual(before,{str(p.relative_to(self.study)):p.read_bytes() for p in self.study.rglob('*') if p.is_file()})

    def test_selected_zero_checkpoint_and_zero_test_captures_are_valid_completion(self):
        self.fixture(select_zero=True,zero_test=True);s=self.run_export();out=self.root/'publication'
        self.assertEqual(s['selected']['samples'],0);self.assertEqual((out/'bc-policy.pt').read_bytes(),(out/'selected-policy.pt').read_bytes())
        self.assertTrue(all(r['counts']['captures']==0 for r in s['test_comparison']));self.assertEqual(s['gate']['status'],'completed')

    def test_privacy_strips_metadata_joints_poses_trajectories_and_host_paths(self):
        self.fixture();s=self.run_export()
        for name in ('summary.json','test-positions.csv','checksums.json'):
            text=(self.root/'publication'/name).read_text()
            for forbidden in ('DO-NOT-EXPORT-PRIVATE-METADATA','RAW-PRIVATE-JOINTS','"q10"','"q14"','"trajectory_m"','"board_center_m"','/home/','/Users/'):
                self.assertNotIn(forbidden,text)
        self.assertIn('6,144 historical TRAIN',s['scope']);self.assertIn('Not equal-acquisition-budget',s['scope'])

    def test_full_exact_external_audit_and_evidence_bytes_required(self):
        self.fixture();old=copy.deepcopy(self.audit)
        for change in (dict(version='sircus-impact650-independent-audit-v1'),dict(checks=[],checks_count=0),dict(checks_count=True),dict(checks_count=2),dict(checks=[{'passed':1}]),dict(status='failed')):
            self.audit=dict(old,**change);export.write_json(self.audit_path,self.audit)
            with self.assertRaises(ValueError):self.run_export()
        self.audit=old;export.write_json(self.audit_path,self.audit)
        report=(self.study/'report.json').read_bytes();(self.study/'report.json').write_bytes(report+b'\n')
        with self.assertRaisesRegex(ValueError,'Audit/report bytes'):self.run_export()
        (self.study/'report.json').write_bytes(report);(self.study/'checkpoints/milestone-0512.pt').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'Changed evidence bytes'):self.run_export()

    def test_cohort_missing_duplicate_and_forbidden_test_search_fail_closed(self):
        self.fixture();self.rows.pop();self.bind()
        with self.assertRaisesRegex(ValueError,'1284 unique'):self.run_export()
        self.fixture();r=next(r for r in self.rows if r['role']=='test');r['ball_flight_search_at_inference']=True;self.bind()
        with self.assertRaisesRegex(ValueError,'Flight search'):self.run_export()
        self.fixture();cohort=[r for r in self.rows if r['role']=='test' and r['method']=='ppo'];cohort[1]['context_id']=cohort[0]['context_id'];cohort[1]['bucket_xy_m']=cohort[0]['bucket_xy_m'];self.positions();self.bind()
        with self.assertRaisesRegex(ValueError,'One requested shot per TEST'):self.run_export()

    def test_validation_only_selection_and_predicted_action_must_be_frozen(self):
        self.fixture();self.report['selected']=self.artifacts[1024];self.report['artifacts']['selected']=self.artifacts[1024];self.bind()
        with self.assertRaisesRegex(ValueError,'VAL only'):self.run_export()
        self.fixture();r=next(r for r in self.rows if r['role']=='test');r['command_values']=[.46,-.1,.38,28.,.4];self.bind()
        with self.assertRaisesRegex(ValueError,'Predicted action was replaced'):self.run_export()
        self.fixture();r=next(r for r in self.rows if r['role']=='test');r['artifact_sha256']='f'*64;self.bind()
        with self.assertRaisesRegex(ValueError,'another artifact'):self.run_export()

    def test_no_overwrite_checkpoint_path_escape_or_missing_teacher(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError,'outside evidence'):export.curate(self.study,self.audit_path,self.study/'publication')
        self.report['artifacts']['bc']=dict(self.report['artifacts']['bc'],path='../private-credentials');self.bind()
        with self.assertRaisesRegex(ValueError,'publication source'):self.run_export()
        self.fixture();teachers=export.read(self.study/'bootstrap/teachers.json');teachers.pop();export.write_json(self.study/'bootstrap/teachers.json',teachers);self.bind()
        with self.assertRaisesRegex(ValueError,'bootstrap denominator'):self.run_export()
        self.fixture();self.run_export()
        with self.assertRaisesRegex(ValueError,'New publication directory'):self.run_export()


if __name__=='__main__':unittest.main()
