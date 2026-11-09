"""No-GPU correction contract, branch preservation and tamper checks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from sircus.mjlab import table_correction_native as N
from sircus.mjlab import table_correction_verify as V
from sircus.mjlab import table_correction_study as S
from sircus.mjlab.benchmark_verify import digest, sha


def fixture():
    rows=[]
    for i,context in enumerate((*N.CONTROL_ROLES,*N.CONTEXTS)):
        row={k:None for k in N.META}
        row.update(id='original-'+context,context_id=context,
            phase='controls_pre' if i<3 else 'test',role=context if i<3 else 'test',
            method='control' if i<3 else 'ppo',repeat=0,
            values=[.3,-.1,.35,20.,.25],bucket_xy_m=[.95,-.1],
            native_release=True,route_success=i<2,invalid_physics=False,
            q10=[.1]*6,q14=[.2]*6,connection_file='connection/'+str(i)+'.json',connection_sha256=str(i)*64)
        rows.append(row)
    rule=dict(parameter='tilt_deg',parameter_index=3,target_value=32,contexts=list(N.CONTEXTS),
        description='One post-hoc tilt-only hypothesis; no training.')
    return rows,rule


class PlanTests(unittest.TestCase):
    def test_exact_42_matching_plan_and_independent_oracle(self):
        rows,rule=fixture();original=copy.deepcopy(rows);plan=N.make_plan(rows,rule)
        self.assertEqual(rows,original);self.assertEqual(len(plan),42)
        self.assertEqual([r['phase'] for r in plan],['controls_pre']*9+['diagnostic']*24+['controls_post']*9)
        V.audit_plan(plan,rows,rule,V.replay.AuditChecks())
        self.assertTrue(all(set(r).isdisjoint({'route_success','q10','q14','native_cache_file'}) for r in plan))
        for row in plan:
            source=next(s for s in rows if s['id']==row['source_record_id'])
            changed=[i for i,(a,b) in enumerate(zip(source['values'],row['values'])) if a!=b]
            self.assertEqual(changed,[3] if row['variant']=='corrected' else [])

    def test_wrong_rule_fail_closed(self):
        rows,rule=fixture()
        for change in ({'parameter_index':4},{'parameter_index':True},{'parameter':'drop_gap_m'},
                       {'target_value':31},{'contexts':list(N.CONTEXTS)+['new']},{'description':''}):
            with self.subTest(change=change),self.assertRaises(ValueError):N.make_plan(rows,dict(rule,**change))

    def test_no_success_replacement_near_shot_or_extra_source(self):
        rows,rule=fixture()
        for key,value in [('native_release',False),('route_success',True),('invalid_physics',True),
                          ('values',[.3,-.1,.35,22.,.25]),('bucket_xy_m',[.4,-.1])]:
            bad=copy.deepcopy(rows);bad[3][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):N.make_plan(bad,rule)
        with self.assertRaises(ValueError):N.make_plan(rows+[rows[3]],rule)

    def test_independent_checker_rejects_tampering(self):
        rows,rule=fixture();plan=N.make_plan(rows,rule)
        for change in ('height','bucket','baseline','missing','duplicate','source'):
            altered=copy.deepcopy(plan)
            if change=='height':altered[12]['values'][2]+=.02
            elif change=='bucket':altered[12]['bucket_xy_m'][0]+=.1
            elif change=='baseline':altered[9]['values'][3]=32
            elif change=='missing':altered.pop()
            elif change=='duplicate':altered[2]=copy.deepcopy(altered[1])
            elif change=='source':altered[0]['source_record_sha256']='fake'
            with self.subTest(change=change),self.assertRaises(V.EvidenceError):
                V.audit_plan(altered,rows,rule,V.replay.AuditChecks())


class CopyTests(unittest.TestCase):
    def test_parent_proofs_are_immutable_hash_bound_and_contained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'src';dst=root/'dst';src.mkdir();dst.mkdir()
            (src/'p.json').write_text('{"key":"proof"}\n');expected=sha(src/'p.json')
            value=N.copy_bound(src,dst,'p.json',expected)
            self.assertEqual(value['sha256'],expected);self.assertEqual(sha(dst/'p.json'),expected)
            self.assertTrue(N.copy_bound(src,dst,'p.json',expected)['cache_hit'])
            (dst/'p.json').write_text('{}')
            with self.assertRaises(ValueError):N.copy_bound(src,dst,'p.json',expected)
            with self.assertRaises(ValueError):N.copy_bound(src,dst,'p.json','incorrect')
            with self.assertRaises(Exception):N.copy_bound(src,dst,'../p.json',expected)


class MatchedTests(unittest.TestCase):
    def engine(self,variant='baseline',mode='ok'):
        sources,rule=fixture();request=next(r for r in N.make_plan(sources,rule) if r['variant']==variant)
        parent=next(r for r in sources if r['id']==request['source_record_id'])
        wrapper=N.MatchedEngine.__new__(N.MatchedEngine);wrapper.sources={parent['id']:parent}
        wrapper.parent_options={'exact':'options'};wrapper.parent_geometry={'exact':'geometry'}
        env=SimpleNamespace(effective_options={'exact':'options'},compiled_geometry={'exact':'geometry'})
        def evaluate(requests,phase):
            result=[]
            for req in requests:
                row=copy.deepcopy(req)
                for key in ('q10','q14','native_release','connection_file','connection_sha256'):row[key]=copy.deepcopy(parent[key])
                if req['variant']=='corrected':row['connection_sha256']='fresh'
                for key in ('q10','q14'):row['native_initial_'+key]=np.asarray(row[key],dtype=np.float32).astype(float).tolist()
                if mode=='branch':row['q10'][0]+=.1
                if mode=='initial':row['native_initial_q10'][0]+=.1
                if mode=='reuse':row['connection_sha256']=parent['connection_sha256']
                if mode=='no_proof':row['connection_sha256']=None
                if mode=='options':env.effective_options['changed']=True
                if mode=='geometry':env.compiled_geometry['changed']=True
                result.append(row)
            return ([] if mode=='missing' else result),{}
        wrapper.engine=SimpleNamespace(adapter=SimpleNamespace(env=env),evaluate=evaluate)
        return wrapper,request

    def test_baseline_branch_exact_and_correction_new_proof(self):
        for variant in ('baseline','corrected','control_exact'):
            engine,request=self.engine(variant);rows,_=engine.evaluate([request],'test')
            self.assertEqual(len(rows),1)

    def test_native_options_joint_branch_and_proof_tamper_fail(self):
        for variant,mode in [('baseline','branch'),('baseline','initial'),('baseline','options'),
                             ('baseline','geometry'),('baseline','missing'),('corrected','reuse'),('corrected','no_proof')]:
            engine,request=self.engine(variant,mode)
            with self.subTest(variant=variant,mode=mode),self.assertRaises(RuntimeError):engine.evaluate([request],'test')

    def test_changed_parent_hash_or_second_parameter_fails(self):
        for variant in ('baseline','corrected'):
            for mode in ('hash','bucket','height'):
                engine,request=self.engine(variant)
                if mode=='hash':request['source_record_sha256']='bad'
                elif mode=='bucket':request['bucket_xy_m'][1]+=.03
                else:request['values'][2]+=.02
                with self.subTest(variant=variant,mode=mode),self.assertRaises(ValueError):engine.evaluate([request],'test')

    def test_independent_observation_branch_check(self):
        sources,rule=fixture();plan=N.make_plan(sources,rule);lookup={r['id']:r for r in sources};rows=[]
        for request in plan:
            row=copy.deepcopy(request);parent=lookup[row['source_record_id']]
            for key in ('connection_file','connection_sha256','native_release','q10','q14'):row[key]=parent[key]
            if row['variant']=='corrected':row['connection_sha256']='fresh'
            rows.append(row)
        V.audit_observations(rows,plan,sources,V.replay.AuditChecks())
        rows[9]['q10']=[0]*6
        with self.assertRaises(V.EvidenceError):V.audit_observations(rows,plan,sources,V.replay.AuditChecks())


class ContactTests(unittest.TestCase):
    def test_robot_channel_does_not_count_floor_table_or_post_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);forces=np.zeros((5,2,6));forces[0,0,2]=10;forces[0,0,3]=10
            forces[1,0,4]=.02;forces[3,0,4]=10;forces[0,1,5]=10
            np.savez(path/'raw.npz',forces_N=forces)
            rows=[dict(native_release=True,native_cache_file='raw.npz',native_cache_row=i,physics_steps=3) for i in range(2)]
            rows.append(dict(native_release=False));S.attach_robot_counts(rows,path)
            self.assertEqual([r['diagnostic_robot_contact_steps'] for r in rows],[1,0,0])


if __name__=='__main__':unittest.main()
