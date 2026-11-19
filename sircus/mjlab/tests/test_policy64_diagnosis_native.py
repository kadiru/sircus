"""CPU-only paired-plan, raw byte bindings and full96-controller mocks."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import policy64_diagnosis_native as worker


def source_row(identity,*,context='control',method='control',role='positive_control',native=True,capture=True):
    hint=dict(id='hint',values=[.6,-.2,.3,30.,.2],bucket_xy_m=[.4,-.3],q10=[.1]*6,q14=[.2]*6,
              source_record_sha256='a'*64)
    row=dict(id=identity,phase='controls_pre' if method=='control' else 'test',role=role,repeat=0,
        method=method,context_id=context,bucket_xy_m=[.4,-.3],values=[.6,-.2,.3,30.,.2],observation=[0.]*9,
        native_release=native,route_success=capture,route_quality=2 if capture else 0,
        invalid=None if native else 'finite_ik_unresolved',invalid_physics=False,
        source_teacher_id='hint',source_hint=hint,effective_source_hint=copy.deepcopy(hint),
        context_resolution_file='resolutions/common.json',context_resolution_sha256='b'*64,
        context_home_sha256='c'*64,context_layout_eligible=True,context_anchor_admitted=True,context_hard_reject=False,
        connection_file='connections/common.json',connection_sha256='d'*64,
        q10=[.1]*6,q14=[.2]*6,native_cache_file='original.npz',native_cache_sha256='e'*64,
        native_cache_row=0,whole_ball_entry=capture)
    return row


def parent_rows():
    rows=[source_row('controls_pre-positive_control-r0'),
          source_row('controls_pre-negative_control-r0',role='negative_control',capture=False)]
    for code in worker.CONTEXTS:
        context='policy64-test-'+code
        bc=source_row('test-bc-0000-'+context,context=context,method='bc',role='test')
        ppo=source_row('test-ppo-1024-'+context,context=context,method='ppo',role='test',
                       native=code in ('07','17'),capture=False)
        ppo['values']=[.601,-.202,.295,30.4,.21]
        rows.extend([bc,ppo])
    return rows


class PlanTests(unittest.TestCase):
    def test_literal96_branches_swaps_controls_and_no_repair_of_historical_rejects(self):
        rows=parent_rows();before=copy.deepcopy(rows);plan=worker.make_plan(rows)
        self.assertEqual(rows,before);self.assertEqual(len(plan),96);self.assertEqual(len({r['id'] for r in plan}),96)
        self.assertEqual([r['role'] for r in plan[:6]],['positive_control']*3+['negative_control']*3)
        self.assertEqual([r['phase'] for r in plan[-6:]],['controls_post']*6)
        byid={r['id']:r for r in rows}
        for index,code in enumerate(worker.CONTEXTS):
            cohort=plan[6+index*12:18+index*12]
            self.assertEqual([r['variant'] for r in cohort],['bc_exact']*3+['ppo_exact']*3+[v for v,_ in worker.SWAPS])
            bc,ppo=(byid['test-'+method+'-'+('0000' if method=='bc' else '1024')+'-policy64-test-'+code]
                    for method in ('bc','ppo'))
            for r in cohort[:6]:
                source=bc if r['variant']=='bc_exact' else ppo
                self.assertEqual(r['values'],source['values']);self.assertEqual(r['parent_record_sha256'],worker.digest(source))
                if source['native_release']:
                    self.assertEqual(r['q10'],source['q10']);self.assertEqual(r['q14'],source['q14'])
                else:
                    self.assertEqual(r['kind'],'historical_rejection');self.assertNotIn('q10',r);self.assertNotIn('q14',r)
            for r,(_,indices) in zip(cohort[6:],worker.SWAPS):
                expected=ppo['values'].copy()
                for i in indices:expected[i]=bc['values'][i]
                self.assertEqual(r['values'],expected);self.assertEqual(r['context_resolution_sha256'],ppo['context_resolution_sha256'])
                self.assertEqual(r['kind'],'coordinate_swap');self.assertNotIn('q10',r)
        self.assertEqual(sum(r['kind']=='historical_rejection' for r in plan),15)
        self.assertEqual(sum(r['kind']=='coordinate_swap' for r in plan),42)

    def test_pairing_source_or_context_drift_cannot_change_frozen_diagnosis(self):
        for edit in (lambda r:r[2].update(route_success=False),
                     lambda r:r[3].update(bucket_xy_m=[.41,-.3]),
                     lambda r:r[3].update(context_resolution_sha256='f'*64),
                     lambda r:r.append(copy.deepcopy(r[2]))):
            rows=parent_rows();edit(rows)
            with self.assertRaises(ValueError):worker.make_plan(rows)


class BindingsTests(unittest.TestCase):
    def fixture(self,root):
        study=root/'parent';study.mkdir();(study/'native').mkdir()
        (study/'native/proof.json').write_text('{"passed":true}\n')
        (study/'native/cache.npz').write_bytes(b'mock-dense-cache')
        proof_sha=worker.sha(study/'native/proof.json');cache_sha=worker.sha(study/'native/cache.npz')
        rows=[dict(source_row(f'parent-{i}'),context_resolution_file='proof.json',context_resolution_sha256=proof_sha,
                   connection_file='proof.json',connection_sha256=proof_sha,native_cache_file='cache.npz',native_cache_sha256=cache_sha)
              for i in range(1284)]
        (study/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        spec=dict(version=worker.PARENT_VERSION,fixture=True);spec['fingerprint']=worker.digest(spec)
        worker.atomic(study/'contract.json',spec)
        manifest={str(p.relative_to(study)):worker.sha(p) for p in study.rglob('*') if p.is_file()}
        worker.atomic(study/'evidence-manifest.json',manifest)
        report=dict(version=worker.PARENT_VERSION,status='completed',phase='completed',
                    contract_fingerprint=spec['fingerprint'],evidence_sha256=manifest)
        worker.atomic(study/'report.json',report)
        proof=dict(version=worker.PARENT_AUDIT,status='passed',failures=[],checks_count=346485,
                   checks=[dict(passed=True)]*346485,contract_fingerprint=spec['fingerprint'],
                   report_sha256=worker.sha(study/'report.json'),evidence_manifest_sha256=worker.sha(study/'evidence-manifest.json'))
        audit=root/'audit.json';worker.atomic(audit,proof)
        published=root/'published.json'
        worker.atomic(published,dict(provenance=dict(report_sha256=worker.sha(study/'report.json'),
            contract_sha256=worker.sha(study/'contract.json'),evidence_manifest_sha256=worker.sha(study/'evidence-manifest.json'),
            independent_audit_sha256=worker.sha(audit),contract_fingerprint=spec['fingerprint'])))
        return study,audit,published

    def test_parent_authentication_binds_every_manifest_artifact_and_exact_passed_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,audit,published=self.fixture(Path(tmp))
            checked=worker.authenticate_parent(study,audit,published)
            self.assertEqual(len(checked['rows']),1284);self.assertEqual(checked['bindings']['parent_checks'],346485)
            (study/'native/cache.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'artifact bytes'):worker.authenticate_parent(study,audit,published)

    def test_failed_audit_or_changed_row_proof_hash_cannot_enter_new_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,audit,published=self.fixture(Path(tmp))
            proof=worker.read(audit);proof['checks'][0]['passed']=False;worker.atomic(audit,proof)
            public=worker.read(published);public['provenance']['independent_audit_sha256']=worker.sha(audit);worker.atomic(published,public)
            with self.assertRaisesRegex(ValueError,'complete passed'):worker.authenticate_parent(study,audit,published)

    def test_copy_proof_is_hash_identical_and_cannot_escape_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';native=root/'new';(parent/'native').mkdir(parents=True);native.mkdir()
            original=parent/'native/p.json';original.write_text('{"fixture":true}\n');fingerprint=worker.sha(original)
            result=worker.copy_proof(parent,native,'p.json',fingerprint)
            self.assertEqual((native/'p.json').read_bytes(),original.read_bytes());self.assertEqual(result['sha256'],fingerprint)
            worker.copy_proof(parent,native,'p.json',fingerprint)
            (native/'p.json').write_text('{}')
            with self.assertRaises(ValueError):worker.copy_proof(parent,native,'p.json',fingerprint)
            with self.assertRaises(ValueError):worker.copy_proof(parent,native,'../outside.json',fingerprint)


class RuntimeTests(unittest.TestCase):
    def fixture(self,root,*,fail_controls=False,wrong_q=False):
        parentdir=root/'parent';(parentdir/'native/resolutions').mkdir(parents=True);(parentdir/'native/connections').mkdir()
        resolution=parentdir/'native/resolutions/common.json';worker.atomic(resolution,dict(fixture=True))
        connection=parentdir/'native/connections/common.json';worker.atomic(connection,dict(fixture=True))
        rows=parent_rows()
        for r in rows:r.update(context_resolution_sha256=worker.sha(resolution),connection_sha256=worker.sha(connection))
        for name in ('calibration.json','verification.json','setup-home.json'):worker.atomic(parentdir/name,{})
        original_options=dict(version='sircus-calibrated-native-warp-options-v1',substeps=64,native={'mock':1},warp={'mock':1})
        worker.atomic(parentdir/'native/effective-options.json',original_options)
        worker.atomic(parentdir/'native/native-compiled-geometry.json',{'fixture':True})
        worker.atomic(parentdir/'prior.json',dict(records=[rows[0]['source_hint']]))
        analysis=root/'analysis.json'
        spec=dict(scene=dict(ball=dict(radius_m=.0375)),identity=dict(setup=dict(home={'sha256':'c'*64}),
            calibration=dict(score_config={})),source_sha256={},packages={})
        bindings=dict(parent_contract_fingerprint='f'*64,parent_report_sha256='a'*64,
            parent_manifest_sha256='b'*64,parent_audit_sha256='c'*64,parent_rows_sha256='d'*64)
        parent=dict(spec=spec,rows=rows,bindings=bindings)
        worker.atomic(analysis,dict(version=worker.ANALYSIS_VERSION,status='completed',fixture=True,
            source=dict(contract_fingerprint='f'*64,report_sha256='a'*64,manifest_sha256='b'*64,
                        original_audit_sha256='c'*64,rows_sha256='d'*64,substeps=64)))
        args=SimpleNamespace(parent_study=parentdir,parent_audit=root/'audit.json',native_source=root/'native-source',
                             assets=root/'assets',analysis=analysis,output=root/'study',wall_time_limit=3600)
        self.connections=[];self.runs=[]
        owner=self
        class Resolver:
            def __init__(self,ref,assets,directory,sources,home):self.directory=Path(directory);self.sources=sources;self.home=home
            def connect(self,resolution,values):
                owner.connections.append(copy.deepcopy(values));index=len(owner.connections)
                c=dict(passed=index!=1,reason=None if index!=1 else 'strict_clearance_reject',
                       endpoint_eligible=index!=1,q10=[.31]*6,q14=[.41]*6)
                filename=f'connections/new-{index}.json';worker.atomic(self.directory/filename,c)
                return dict(c,file=filename,sha256=worker.sha(self.directory/filename))
        class Environment:
            def __init__(self,*a,**kw):
                owner.plan_frozen_before_native=len(worker.read(root/'study/contract.json')['plan'])==96
                self.effective_options=copy.deepcopy(original_options)
                self.compiled_geometry={'fixture':True};self.fixed=None
            def run(self,bucket,values,locks,record):
                owner.runs.append((bucket.copy(),values.copy(),self.fixed[0].copy(),self.fixed[1].copy()))
                n=len(bucket);p=np.zeros((3,n,3));p[0,:,:2]=values[:,:2];p[0,:,2]=values[:,2]+values[:,4]+.0375
                self.last_arrays=dict(position_m=p,velocity_m_s=np.zeros_like(p),forces_N=np.zeros((2,n,6)),time_s=np.array([0.,.1,.2]))
                result=[]
                for i in range(n):
                    # Positive/negative original controls use the same nominal
                    # recipe in this fixture; alternate the first six slots.
                    capture=i<3 and (len(owner.runs) in (1,9)) and not fail_controls
                    q10,q14=self.fixed
                    result.append(dict(native_release=True,values=values[i].tolist(),bucket_xy_m=bucket[i].tolist(),
                        q10=(q10[i]+(.001 if wrong_q else 0)).tolist(),q14=q14[i].tolist(),
                        native_initial_q10=q10[i].astype(np.float32).astype(float).tolist(),native_initial_q14=q14[i].astype(np.float32).astype(float).tolist(),
                        board_normal=[0.,0.,1.],route_success=capture,route_quality=2 if capture else 0,
                        whole_ball_entry=capture,invalid=None,invalid_physics=False,scalar_score_audit_passed=True,
                        contact_protocol=dict(substeps=64),board_bounce=True,max_retention_s=1. if capture else 0.))
                return result,dict(mock=True)
        @contextmanager
        def fixed(env,bucket,values,q10,q14,valid):env.fixed=(q10,q14);yield
        runtime=SimpleNamespace(GeometryResolver=Resolver,verify_resolution=lambda *a:True,verify_connection=lambda *a:True,
            source_geometry=copy.deepcopy,Environment=Environment,fixed_endpoints=fixed,bounds={},
            constraints=lambda n:None,first_flight=lambda *a,**kw:dict(available=kw.get('valid',True),fixture=True),
            gpu=SimpleNamespace(foreign_processes=lambda:[]))
        return args,parent,runtime

    def test_full96_flow_keeps_original_rejects_new_rejects_exact_branches_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,parent,runtime=self.fixture(Path(tmp))
            with patch('builtins.print'):report=worker.run(args,parent=parent,runtime=runtime)
            records=[json.loads(x) for x in (args.output/'records.jsonl').read_text().splitlines()]
            self.assertEqual(report['status'],'completed');self.assertEqual(len(records),96)
            self.assertTrue(self.plan_frozen_before_native)
            self.assertEqual(len(self.connections),42);self.assertEqual(len(self.runs),9)
            self.assertEqual(sum(r['kind']=='historical_rejection' for r in records),15)
            self.assertEqual(report['counts']['rejected'],16)
            self.assertTrue(all(not r['native_release'] and 'q10' not in r for r in records if r['kind']=='historical_rejection'))
            exact=[r for r in records if r['kind']=='exact_replay']
            self.assertTrue(all(r['q10']==[.1]*6 and r['q14']==[.2]*6 for r in exact))
            new=[r for r in records if r['kind']=='coordinate_swap' and r['native_release']]
            self.assertTrue(all(r['q10']==[.31]*6 and r['q14']==[.41]*6 for r in new))
            self.assertEqual((args.output/'analysis.json').read_bytes(),args.analysis.read_bytes())
            manifest=worker.read(args.output/'evidence-manifest.json')
            self.assertIn('native/effective-options.json',manifest);self.assertIn('native/native-compiled-geometry.json',manifest)
            self.assertEqual(manifest,report['evidence_sha256'])
            for name,fingerprint in manifest.items():self.assertEqual(worker.sha(args.output/name),fingerprint)
            self.assertEqual(len(worker.read(args.output/'contract.json')['plan']),96)

    def test_control_failure_stops_before_swaps_and_native_branch_substitution_fails(self):
        for mode in ('control','branch'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                args,parent,runtime=self.fixture(Path(tmp),fail_controls=mode=='control',wrong_q=mode=='branch')
                with patch('builtins.print'),self.assertRaises(RuntimeError):worker.run(args,parent=parent,runtime=runtime)
                self.assertEqual(len(self.connections),0);self.assertEqual(len(self.runs),1)
                self.assertEqual(worker.read(args.output/'report.json')['status'],'failed')
                self.assertFalse((args.output/'evidence-manifest.json').exists())


if __name__=='__main__':unittest.main()
