"""CPU-only63-slot plan, immutable branches, refusals and intervention tamper tests."""
import copy
from types import SimpleNamespace
from pathlib import Path
import unittest
import numpy as np
from sircus.mjlab import recovery_regression_native as N
from sircus.mjlab import recovery_regression_verify as V
from sircus.mjlab import recovery_regression_study as S
from sircus.mjlab import recovery_regression_pipeline as P
from sircus.mjlab.benchmark_verify import read


def fixture():
    controls=[];sources=[]
    for i,context in enumerate((*N.ROLES,*N.CASES)):
        methods=('control',) if i<3 else ('parent_bc','recovery_bc')
        for method in methods:
            row={key:None for key in N.META}
            row.update(id=f'original-{context}-{method}',context_id=context,role=context if i<3 else 'test',
                phase='controls_pre' if i<3 else 'test',method=method,repeat=0,
                values=[.3,-.1,.35 if method!='recovery_bc' else .37,20.,.25],bucket_xy_m=[.95,-.1],
                native_release=not (method=='recovery_bc' and i>=5),route_success=(i<2 or method=='parent_bc'),
                invalid=None if not (method=='recovery_bc' and i>=5) else 'finite_ik_unresolved',invalid_physics=False,
                q10=[.1]*6,q14=[.2]*6,connection_file=f'connection/{i}-{method}.json',connection_sha256=f'{i}-{method}')
            (controls if i<3 else sources).append(row)
    rule=dict(parameter='board_height_m',parameter_index=2,mode='restore_matched_parent_coordinate',
        contexts=list(N.CASES),description='One saved-evidence height restoration; fixed gap also shifts release height.',diagnosis_sha256=N.DIAGNOSIS_SHA256)
    return sources,controls,rule


class PlanTests(unittest.TestCase):
    def test_literal_rule_and_budget_contract_import_without_training(self):
        rule=read(Path(N.__file__).with_name('recovery-regression-height-correction.json'))
        self.assertEqual(N.validate_rule(rule),rule)
        self.assertEqual((S.VERSION,V.STUDY_VERSION),(N.VERSION,)*2)
        self.assertEqual(S.INPUT_KEYS,P.INPUT_KEYS)
        p=S.protocol()
        self.assertEqual((p['maximum_requests'],p['diagnostic_requests'],p['control_requests']),(63,45,18))
        self.assertEqual((p['maximum_acquired_s'],p['maximum_queue_s'],p['maximum_pipeline_s'],p['audit_reserve_s']),(3600,900,6300,1800))
        self.assertEqual(p['storage_reserve_bytes'],8*1024**3)

    def test_exact63_plan_independently_reconstructed(self):
        sources,controls,rule=fixture();before=copy.deepcopy((sources,controls));plan=N.make_plan(sources,controls,rule)
        self.assertEqual((sources,controls),before)
        self.assertEqual([r['phase'] for r in plan],['controls_pre']*9+['diagnostic']*45+['controls_post']*9)
        V.audit_plan(plan,sources,controls,rule,V.replay.AuditChecks())
        self.assertEqual(sum(r['admission_mode']=='cached_parent_refusal' for r in plan),9)
        self.assertEqual(sum(r['admission_mode']=='new_action_strict_proof' for r in plan),15)
        self.assertTrue(all(set(r).isdisjoint({'q10','q14','route_success','native_cache_file'}) for r in plan))
        lookup={r['id']:r for r in sources+controls}
        for row in plan:
            changes=[i for i,(a,b) in enumerate(zip(row['values'],lookup[row['source_record_id']]['values'])) if a!=b]
            self.assertEqual(changes,[2] if row['variant']=='corrected' else [])
            self.assertEqual(row['values'][4],lookup[row['source_record_id']]['values'][4])

    def test_no_rule_or_case_mutation(self):
        sources,controls,rule=fixture()
        for change in ({'parameter_index':4},{'parameter_index':True},{'parameter':'drop_gap_m'},
            {'mode':'absolute_height'},{'contexts':list(reversed(N.CASES))},{'description':''},{'diagnosis_sha256':'bad'}):
            with self.subTest(change=change),self.assertRaises(ValueError):N.make_plan(sources,controls,dict(rule,**change))
        for key,value in (('route_success',True),('invalid_physics',True),('bucket_xy_m',[.5,.1]),('observation',[1]*9)):
            bad=copy.deepcopy(sources);bad[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):N.make_plan(bad,controls,rule)
        with self.assertRaises(ValueError):N.make_plan(sources+[sources[0]],controls,rule)
        with self.assertRaises(ValueError):N.make_plan(sources,controls[:2],rule)

    def test_independent_checker_rejects_action_count_and_namespace_tamper(self):
        sources,controls,rule=fixture();plan=N.make_plan(sources,controls,rule)
        for change in ('gap','bucket','parent','height','missing','duplicate','namespace','donor','refusal'):
            bad=copy.deepcopy(plan)
            if change=='gap':bad[15]['values'][4]+=.01
            elif change=='bucket':bad[15]['bucket_xy_m'][0]+=.1
            elif change=='parent':bad[9]['values'][2]+=.01
            elif change=='height':bad[15]['values'][2]+=.01
            elif change=='missing':bad.pop()
            elif change=='duplicate':bad[1]=copy.deepcopy(bad[0])
            elif change=='namespace':bad[0]['adapter_directory']='native'
            elif change=='donor':bad[15]['donor_record_sha256']='wrong'
            else:bad[30]['admission_mode']='new_action_strict_proof'
            with self.subTest(change=change),self.assertRaises(V.EvidenceError):V.audit_plan(bad,sources,controls,rule,V.replay.AuditChecks())


class MatchedTests(unittest.TestCase):
    def test_control_namespaces_share_only_one_lazy_environment(self):
        calls=[]
        def factory():calls.append(1);return object()
        first=SimpleNamespace(env=None,environment_factory=factory)
        second=SimpleNamespace(env=None,environment_factory=lambda:object())
        S.share_native_environment(SimpleNamespace(adapter=first),SimpleNamespace(engine=SimpleNamespace(adapter=second)))
        self.assertEqual(calls,[])
        self.assertIs(first.environment_factory(),second.environment_factory())
        self.assertEqual(calls,[1])
        first.env=first.environment_factory()
        with self.assertRaises(ValueError):S.share_native_environment(SimpleNamespace(adapter=first),SimpleNamespace(engine=SimpleNamespace(adapter=second)))

    def engine(self,variant='parent_exact',mode='ok',refusal=False):
        sources,controls,rule=fixture();plan=N.make_plan(sources,controls,rule)
        request=next(r for r in plan if r['variant']==variant and (not refusal or r['admission_mode']=='cached_parent_refusal'))
        lookup={r['id']:copy.deepcopy(r) for r in sources+controls};parent=lookup[request['source_record_id']]
        wrapper=N.MatchedEngine.__new__(N.MatchedEngine);wrapper.rule=rule;wrapper.sources=lookup
        wrapper.parent_options={'exact':'options'};wrapper.parent_geometry={'exact':'geometry'}
        env=SimpleNamespace(effective_options={'exact':'options'},compiled_geometry={'exact':'geometry'})
        def evaluate(requests,phase):
            result=[]
            for req in requests:
                row=copy.deepcopy(req)
                for key in ('q10','q14','native_release','invalid','connection_file','connection_sha256'):row[key]=copy.deepcopy(parent[key])
                if req['variant']=='corrected':row['connection_sha256']='fresh'
                if row['native_release']:
                    for key in ('q10','q14'):row['native_initial_'+key]=np.asarray(row[key],np.float32).astype(float).tolist()
                if mode=='branch':row['q14'][0]+=.2
                if mode=='initial':row['native_initial_q10'][0]+=.1
                if mode=='reuse':row['connection_sha256']=parent['connection_sha256']
                if mode=='no_proof':row['connection_sha256']=None
                if mode=='options':env.effective_options['changed']=True
                if mode=='geometry':env.compiled_geometry['changed']=True
                if mode=='release':row['native_release']=True
                result.append(row)
            return ([] if mode=='missing' else result),{}
        wrapper.engine=SimpleNamespace(adapter=SimpleNamespace(env=env),evaluate=evaluate)
        return wrapper,request

    def test_original_parent_recovery_control_and_fresh_correction(self):
        for variant in (*N.VARIANTS,'control_exact'):
            engine,request=self.engine(variant);self.assertEqual(len(engine.evaluate([request],'diagnostic')[0]),1)
        engine,request=self.engine('recovery_exact',refusal=True)
        self.assertFalse(engine.evaluate([request],'diagnostic')[0][0]['native_release'])

    def test_branch_reset_options_missing_proof_or_placeholder_release_fails(self):
        for variant,mode,refusal in [('parent_exact','branch',False),('parent_exact','initial',False),('parent_exact','options',False),
            ('parent_exact','geometry',False),('parent_exact','missing',False),('corrected','reuse',False),('corrected','no_proof',False),('recovery_exact','release',True)]:
            engine,request=self.engine(variant,mode,refusal)
            with self.subTest(variant=variant,mode=mode),self.assertRaises(RuntimeError):engine.evaluate([request],'diagnostic')

    def test_parent_donor_second_coordinate_and_command_tamper_fail(self):
        for mode in ('source','donor','gap','command','bucket'):
            engine,request=self.engine('corrected')
            if mode=='source':request['source_record_sha256']='bad'
            elif mode=='donor':request['donor_record_sha256']='bad'
            elif mode=='gap':request['values'][4]+=.01
            elif mode=='command':request['command_values'][2]+=.01
            else:request['bucket_xy_m'][0]+=.01
            with self.subTest(mode=mode),self.assertRaises(ValueError):engine.evaluate([request],'diagnostic')

    def test_independent_observation_audit_preserves_refusal_branch_and_fresh_proof(self):
        sources,controls,rule=fixture();plan=N.make_plan(sources,controls,rule);lookup={r['id']:r for r in sources+controls};rows=[]
        for request in plan:
            row=copy.deepcopy(request);parent=lookup[row['source_record_id']]
            for key in ('connection_file','connection_sha256','native_release','q10','q14','invalid'):row[key]=copy.deepcopy(parent[key])
            if row['variant']=='corrected':row['connection_sha256']='new-'+row['context_id']
            rows.append(row)
        V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())
        for mode in ('branch','release','placeholder','reuse','repeated_proof'):
            bad=copy.deepcopy(rows)
            if mode=='branch':bad[9]['q14'][0]+=.3
            elif mode=='release':bad[30]['native_release']=True
            elif mode=='placeholder':bad[30]['native_initial_q10']=[0]*6
            elif mode=='reuse':bad[15]['connection_sha256']=lookup[bad[15]['source_record_id']]['connection_sha256']
            else:bad[15]['connection_sha256']='different-new-proof'
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):V.audit_observations(bad,plan,sources,controls,V.replay.AuditChecks())


if __name__=='__main__':unittest.main()
