"""CPU-only frozen drop-gap plan, branch preservation and mutation guards."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from sircus.mjlab import regression_drop_gap_native as N
from sircus.mjlab import regression_drop_gap_study as S
from sircus.mjlab import regression_drop_gap_pipeline as P
from sircus.mjlab.benchmark_verify import read


def fixture():
    controls=[];sources=[]
    for i,context in enumerate((*N.ROLES,*N.CASES)):
        for method in (('control',) if i<3 else ('parent_bc','recovery_bc')):
            row={key:None for key in N.META}
            row.update(id=f'original-{context}-{method}',context_id=context,role=context if i<3 else 'test',
                phase='controls_pre' if i<3 else 'test',method=method,repeat=0,
                values=[.3,-.1,.35 if method!='recovery_bc' else .37,20.,.25 if method!='recovery_bc' else .29],
                bucket_xy_m=[.95,-.1],native_release=True,route_success=(i<2 or method=='parent_bc'),
                invalid=None,invalid_physics=False,q10=[.1]*6,q14=[.2]*6,
                board_center_m=[.3,-.1,.37],board_normal=[0.,0.,1.],
                connection_file=f'connection/{i}-{method}.json',connection_sha256=f'{i}-{method}')
            (controls if i<3 else sources).append(row)
    rule=read(Path(N.__file__).with_name('regression-drop-gap-correction.json'))
    return sources,controls,rule


class PlanTests(unittest.TestCase):
    def test_literal_rule_and_budget(self):
        _,_,rule=fixture();self.assertEqual(N.validate_rule(rule),rule)
        self.assertEqual(N.VERSION,'sircus-regression-drop-gap-correction-v1')
        self.assertEqual(S.VERSION,N.VERSION);self.assertEqual(S.INPUT_KEYS,P.INPUT_KEYS)
        p=S.protocol()
        self.assertEqual((p['substeps'],p['maximum_requests'],p['diagnostic_requests'],p['control_requests']),(64,36,18,18))
        self.assertEqual((p['changed_parameter'],p['parameter_index']),('drop_gap_m',4))
        self.assertEqual((p['maximum_acquired_s'],p['maximum_queue_s'],p['maximum_pipeline_s'],p['audit_reserve_s']),(3600,900,6300,1800))
        self.assertEqual(p['storage_reserve_bytes'],8*1024**3)

    def test36_plan_only_original_recovery_gap_changes(self):
        sources,controls,rule=fixture();before=copy.deepcopy((sources,controls));plan=N.make_plan(sources,controls,rule)
        self.assertEqual((sources,controls),before)
        self.assertEqual([r['phase'] for r in plan],['controls_pre']*9+['diagnostic']*18+['controls_post']*9)
        self.assertEqual(len(set(r['id'] for r in plan)),36)
        self.assertEqual(sum(r['admission_mode']=='cached_parent_refusal' for r in plan),0)
        self.assertEqual(sum(r['admission_mode']=='new_action_strict_proof' for r in plan),6)
        self.assertTrue(all(set(r).isdisjoint({'q10','q14','route_success','native_cache_file'}) for r in plan))
        lookup={r['id']:r for r in sources+controls}
        for row in plan:
            source=lookup[row['source_record_id']]
            changes=[i for i,(a,b) in enumerate(zip(row['values'],source['values'])) if a!=b]
            self.assertEqual(changes,[4] if row['variant']=='corrected' else [])
            self.assertEqual(row['values'][:4],source['values'][:4])
            if row['variant']=='corrected':
                self.assertEqual(row['values'][2],.37) # Earlier parent-height correction is NOT combined.
                self.assertEqual(row['values'][4],lookup[row['donor_record_id']]['values'][4])

    def test_rule_source_and_admission_mutation_rejected(self):
        sources,controls,rule=fixture()
        for change in ({'parameter_index':2},{'parameter_index':True},{'parameter':'board_height_m'},
            {'mode':'absolute_height'},{'contexts':list(reversed(N.CASES))},{'description':''},{'diagnosis_sha256':'bad'}):
            with self.subTest(change=change),self.assertRaises(ValueError):N.make_plan(sources,controls,dict(rule,**change))
        for key,value in (('route_success',True),('invalid_physics',True),('native_release',False),('bucket_xy_m',[.5,.1]),('observation',[1]*9)):
            bad=copy.deepcopy(sources);bad[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):N.make_plan(bad,controls,rule)
        with self.assertRaises(ValueError):N.make_plan(sources+[sources[0]],controls,rule)
        with self.assertRaises(ValueError):N.make_plan(sources,controls[:2],rule)
        bad=copy.deepcopy(sources);bad[1]['values'][4]=bad[0]['values'][4]
        with self.assertRaises(ValueError):N.make_plan(bad,controls,rule)


class MatchedTests(unittest.TestCase):
    def engine(self,variant='parent_exact',mode='ok'):
        sources,controls,rule=fixture();plan=N.make_plan(sources,controls,rule)
        request=next(r for r in plan if r['variant']==variant)
        lookup={r['id']:copy.deepcopy(r) for r in sources+controls};parent=lookup[request['source_record_id']]
        wrapper=N.MatchedEngine.__new__(N.MatchedEngine);wrapper.rule=rule;wrapper.sources=lookup
        wrapper.parent_options={'exact':'options'};wrapper.parent_geometry={'exact':'geometry'}
        env=SimpleNamespace(effective_options={'exact':'options'},compiled_geometry={'exact':'geometry'})
        def evaluate(requests,phase):
            result=[]
            for req in requests:
                row=copy.deepcopy(req)
                for key in ('q10','q14','native_release','invalid','connection_file','connection_sha256','board_center_m','board_normal'):row[key]=copy.deepcopy(parent[key])
                if req['variant']=='corrected':row['connection_sha256']='fresh'
                if mode=='new_branch':row['q14'][0]+=.3;row['board_center_m'][2]+=.001;row['board_normal']=[0.,.1,1.]
                if row['native_release']:
                    for key in ('q10','q14'):row['native_initial_'+key]=np.asarray(row[key],np.float32).astype(float).tolist()
                if mode=='branch':row['q14'][0]+=.2
                if mode=='initial':row['native_initial_q10'][0]+=.1
                if mode=='reuse':row['connection_sha256']=parent['connection_sha256']
                if mode=='no_proof':row['connection_sha256']=None
                if mode=='options':env.effective_options['changed']=True
                if mode=='geometry':env.compiled_geometry['changed']=True
                if mode=='decline':
                    row.update(native_release=False,q10=None,q14=None,invalid='finite_ik_unresolved')
                    for key in ('native_initial_q10','native_initial_q14','board_center_m','board_normal'):row.pop(key,None)
                result.append(row)
            return ([] if mode=='missing' else result),{}
        wrapper.engine=SimpleNamespace(adapter=SimpleNamespace(env=env),evaluate=evaluate)
        return wrapper,request

    def test_bracketing_controls_and_variants(self):
        for variant in (*N.VARIANTS,'control_exact'):
            engine,request=self.engine(variant);rows,_=engine.evaluate([request],'diagnostic')
            self.assertEqual(len(rows),1)
            self.assertTrue(rows[0]['realization_comparison']['q14_equal_to_source'])
            self.assertEqual(rows[0]['realization_comparison']['board_center_delta_m'],[0.,0.,0.])

    def test_changed_branch_reported_not_hidden_or_forced(self):
        engine,request=self.engine('corrected','new_branch');rows,_=engine.evaluate([request],'diagnostic')
        comparison=rows[0]['realization_comparison']
        self.assertFalse(comparison['q14_equal_to_source']);self.assertAlmostEqual(comparison['q14_max_abs_delta_rad'],.3)
        self.assertAlmostEqual(comparison['board_center_delta_m'][2],.001)
        self.assertAlmostEqual(comparison['board_normal_angle_deg'],np.degrees(np.arctan(.1)))
        engine,request=self.engine('corrected','decline');rows,_=engine.evaluate([request],'diagnostic')
        self.assertIsNone(rows[0]['realization_comparison']);self.assertFalse(rows[0]['native_release'])

    def test_original_branch_reset_or_options_mutation_fails(self):
        for variant,mode in [('parent_exact','branch'),('parent_exact','initial'),('parent_exact','options'),
            ('parent_exact','geometry'),('parent_exact','missing'),('parent_exact','decline'),('corrected','reuse'),('corrected','no_proof')]:
            engine,request=self.engine(variant,mode)
            with self.subTest(variant=variant,mode=mode),self.assertRaises(RuntimeError):engine.evaluate([request],'diagnostic')

    def test_source_donor_extra_coordinate_or_command_tamper_fails(self):
        for mode in ('source','donor','height','gap','command','bucket'):
            engine,request=self.engine('corrected')
            if mode=='source':request['source_record_sha256']='bad'
            elif mode=='donor':request['donor_record_sha256']='bad'
            elif mode=='height':request['values'][2]+=.01
            elif mode=='gap':request['values'][4]+=.01
            elif mode=='command':request['command_values'][4]+=.01
            else:request['bucket_xy_m'][0]+=.01
            with self.subTest(mode=mode),self.assertRaises(ValueError):engine.evaluate([request],'diagnostic')

    def test_shared_environment_is_lazy_and_single(self):
        calls=[]
        def factory():calls.append(1);return object()
        first=SimpleNamespace(env=None,environment_factory=factory);second=SimpleNamespace(env=None,environment_factory=lambda:object())
        S.share_native_environment(SimpleNamespace(adapter=first),SimpleNamespace(engine=SimpleNamespace(adapter=second)))
        self.assertEqual(calls,[]);self.assertIs(first.environment_factory(),second.environment_factory());self.assertEqual(calls,[1])
        first.env=first.environment_factory()
        with self.assertRaises(ValueError):S.share_native_environment(SimpleNamespace(adapter=first),SimpleNamespace(engine=SimpleNamespace(adapter=second)))


if __name__=='__main__':unittest.main()
