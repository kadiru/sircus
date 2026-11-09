"""CPU-only tests for bounded IK continuation, preserved recipes and exact admission."""
import copy
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import regression_ik_native as N
from sircus.mjlab import regression_ik_study as S
from sircus.mjlab import regression_ik_pipeline as P
from sircus.mjlab.benchmark_verify import read


def fixture():
    controls=[];sources=[]
    for i,context in enumerate((*N.ROLES,*N.CASES)):
        for method in (('control',) if i<3 else ('parent_bc','recovery_bc')):
            declined=method=='recovery_bc'
            row={key:None for key in N.META}
            row.update(id=f'original-{context}-{method}',context_id=context,role=context if i<3 else 'test',
                phase='controls_pre' if i<3 else 'test',method=method,repeat=0,
                values=[.33,-.13,.37,21.,.29] if declined else [.3,-.1,.35,20.,.25],
                bucket_xy_m=[.95,-.1],native_release=not declined,route_success=(i<2 or method=='parent_bc'),
                invalid='finite_ik_unresolved' if declined else None,invalid_physics=False,
                q10=[0.]*6 if declined else [.1]*6,q14=[0.]*6 if declined else [.2]*6,
                connection_file=f'connections/{i}-{method}.json',connection_sha256=f'{i}-{method}',
                context_resolution_file='resolutions/context.json',context_resolution_sha256='resolution')
            if not declined:row.update(board_center_m=[.3,-.1,.35],board_normal=[0.,0.,1.])
            (controls if i<3 else sources).append(row)
    rule=read(Path(N.__file__).with_name('regression-ik-correction.json'))
    return sources,controls,rule


class PlanTests(unittest.TestCase):
    def test_literal_rule_and_budget(self):
        _,_,rule=fixture();self.assertEqual(N.validate_rule(rule),rule)
        self.assertEqual(N.VERSION,'sircus-regression-ik-feasibility-v1')
        self.assertEqual(S.VERSION,N.VERSION);self.assertEqual(S.INPUT_KEYS,P.INPUT_KEYS)
        p=S.protocol()
        self.assertEqual((p['substeps'],p['maximum_requests'],p['diagnostic_requests'],p['control_requests']),(64,66,48,18))
        self.assertEqual((p['maximum_acquired_s'],p['maximum_queue_s'],p['maximum_pipeline_s'],p['audit_reserve_s']),(3600,900,6300,1800))
        self.assertEqual(p['storage_reserve_bytes'],8*1024**3)

    def test66_plan_coordinate_blocks_and_full_interpolation(self):
        sources,controls,rule=fixture();before=copy.deepcopy((sources,controls));plan=N.make_plan(sources,controls,rule)
        self.assertEqual((sources,controls),before)
        self.assertEqual([r['phase'] for r in plan],['controls_pre']*9+['diagnostic']*48+['controls_post']*9)
        self.assertEqual(len(set(r['id'] for r in plan)),66)
        self.assertEqual(sum(r['admission_mode']=='cached_parent_refusal' for r in plan),6)
        self.assertEqual(sum(r['admission_mode']=='new_action_strict_proof' for r in plan),36)
        self.assertTrue(all(set(r).isdisjoint({'q10','q14','route_success','native_cache_file'}) for r in plan))
        lookup={r['id']:r for r in sources+controls}
        for row in plan:
            source=lookup[row['source_record_id']]
            changed=[i for i,(a,b) in enumerate(zip(row['values'],source['values'])) if a!=b]
            self.assertEqual(changed,row['parameter_indices'])
            if row['variant'] in N.NEW_VARIANTS:
                parent=lookup[row['donor_record_id']]
                for i in range(5):
                    if row['variant'].startswith('blend'):
                        alpha=row['interpolation_alpha']
                        self.assertEqual(row['values'][i],source['values'][i]+alpha*(parent['values'][i]-source['values'][i]))
                    elif i in row['parameter_indices']:self.assertEqual(row['values'][i],parent['values'][i])
                    else:self.assertEqual(row['values'][i],source['values'][i])
            reference=lookup[row['realization_reference_record_id']]
            self.assertTrue(reference['native_release'])
            if row['phase']=='diagnostic':self.assertEqual(reference['method'],'parent_bc')
            if row['variant']=='parent_height_gap':
                self.assertEqual(sum(row['values'][i] for i in (2,4)),sum(lookup[row['donor_record_id']]['values'][i] for i in (2,4)))

    def test_rule_and_original_source_tampering_rejected(self):
        sources,controls,rule=fixture()
        for change in ({'mode':'generic_search'},{'contexts':list(reversed(N.CASES))},{'description':''},
                       {'diagnosis_sha256':'bad'},{'blend_fractions':dict(N.BLEND_FRACTIONS,blend25=.3)},
                       {'coordinate_groups':dict(N.COORDINATE_GROUPS,parent_xy=[False,True])},{'extra_budget':10}):
            with self.subTest(change=change),self.assertRaises(ValueError):N.make_plan(sources,controls,dict(rule,**change))
        for key,value in (('route_success',True),('invalid_physics',True),('native_release',True),('invalid','collision'),('bucket_xy_m',[.5,.1]),('observation',[1]*9)):
            bad=copy.deepcopy(sources);bad[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):N.make_plan(bad,controls,rule)
        with self.assertRaises(ValueError):N.make_plan(sources+[sources[0]],controls,rule)
        with self.assertRaises(ValueError):N.make_plan(sources,controls[:2],rule)


class MatchedTests(unittest.TestCase):
    def engine(self,variant='parent_exact',mode='ok'):
        sources,controls,rule=fixture();plan=N.make_plan(sources,controls,rule)
        request=next(r for r in plan if r['variant']==variant)
        lookup={r['id']:copy.deepcopy(r) for r in sources+controls}
        source=lookup[request['source_record_id']];reference=lookup[request['realization_reference_record_id']]
        wrapper=N.MatchedEngine.__new__(N.MatchedEngine);wrapper.rule=rule;wrapper.sources=lookup;wrapper.directory=Path('/unused-proof-test')
        wrapper.parent_options={'exact':'options'};wrapper.parent_geometry={'exact':'geometry'}
        env=SimpleNamespace(effective_options={'exact':'options'},compiled_geometry={'exact':'geometry'})
        hashes={str(wrapper.directory/s['connection_file']):s['connection_sha256'] for s in sources+controls}
        hashes[str(wrapper.directory/'resolutions/context.json')]='resolution'
        hashes[str(wrapper.directory/'connections/new.json')]='fresh'
        def evaluate(requests,phase):
            result=[]
            for req in requests:
                row=copy.deepcopy(req)
                for key in ('q10','q14','native_release','invalid','connection_file','connection_sha256'):
                    row[key]=copy.deepcopy(source[key])
                if req['variant'] in N.NEW_VARIANTS:
                    row.update(native_release=True,invalid=None,q10=[.13]*6,q14=[.23]*6,connection_file='connections/new.json',connection_sha256='fresh')
                if row['native_release']:
                    row.update(board_center_m=[.31,-.11,.36],board_normal=[0.,.1,1.])
                    for key in ('q10','q14'):row['native_initial_'+key]=np.asarray(row[key],np.float32).astype(float).tolist()
                if mode=='branch':row['q14'][0]+=.2
                if mode=='initial':row['native_initial_q10'][0]+=.1
                if mode=='reuse':row['connection_sha256']=source['connection_sha256']
                if mode=='no_proof':row['connection_sha256']=None
                if mode=='options':env.effective_options['changed']=True
                if mode=='geometry':env.compiled_geometry['changed']=True
                if mode=='proofbytes':row['connection_sha256']='wrong'
                if mode=='decline':
                    row.update(native_release=False,q10=[0.]*6,q14=[0.]*6,invalid='finite_ik_unresolved')
                    for key in ('native_initial_q10','native_initial_q14','board_center_m','board_normal'):row.pop(key,None)
                result.append(row)
            return ([] if mode=='missing' else result),{}
        wrapper.engine=SimpleNamespace(adapter=SimpleNamespace(env=env),evaluate=evaluate)
        return wrapper,request,hashes

    def evaluate(self,engine,request,hashes):
        with ExitStack() as stack:
            stack.enter_context(patch.object(N,'contained',side_effect=lambda root,name:root/name))
            stack.enter_context(patch.object(N,'sha',side_effect=lambda p:hashes[str(p)]))
            stack.enter_context(patch.object(N,'read',return_value={}))
            stack.enter_context(patch.object(N,'setup_evidence',return_value={'passed':'stub'}))
            return engine.evaluate([request],request['phase'])

    def test_exact_variants_and_new_branches_use_admitted_parent_reference(self):
        for variant in (*N.VARIANTS,'control_exact'):
            engine,request,hashes=self.engine(variant);rows,_=self.evaluate(engine,request,hashes)
            compare=rows[0]['realization_comparison']
            if variant=='recovery_exact':
                self.assertIsNone(compare);self.assertFalse(rows[0]['native_release'])
            else:
                self.assertEqual(compare['source_record_id'],request['realization_reference_record_id'])
                self.assertEqual(compare['basis'],'exact_control_source' if variant=='control_exact' else 'matched_successful_parent')
                if variant in N.NEW_VARIANTS:
                    self.assertFalse(compare['q14_equal_to_source']);self.assertAlmostEqual(compare['q14_max_abs_delta_rad'],.03)
                    self.assertAlmostEqual(compare['board_center_delta_m'][2],.01)
            self.assertEqual(rows[0]['setup_proof'],{'passed':'stub'})

    def test_new_refusal_has_no_native_pose(self):
        engine,request,hashes=self.engine('blend50','decline');rows,_=self.evaluate(engine,request,hashes)
        self.assertIsNone(rows[0]['realization_comparison']);self.assertNotIn('native_initial_q10',rows[0])

    def test_original_branch_native_reset_options_and_bound_proofs(self):
        for variant,mode in [('parent_exact','branch'),('parent_exact','initial'),('parent_exact','options'),
            ('parent_exact','geometry'),('parent_exact','missing'),('parent_exact','decline'),
            ('blend25','reuse'),('blend25','no_proof'),('blend25','proofbytes')]:
            engine,request,hashes=self.engine(variant,mode)
            with self.subTest(variant=variant,mode=mode),self.assertRaises(RuntimeError):self.evaluate(engine,request,hashes)

    def test_request_recipe_donor_and_basis_mutation_rejected(self):
        for mode in ('source','donor','value','command','bucket','indices','alpha','reference','wrong_reference','phase'):
            engine,request,hashes=self.engine('blend25')
            if mode=='source':request['source_record_sha256']='bad'
            elif mode=='donor':request['donor_record_sha256']='bad'
            elif mode=='value':request['values'][2]+=.01
            elif mode=='command':request['command_values'][2]+=.01
            elif mode=='bucket':request['bucket_xy_m'][0]+=.01
            elif mode=='indices':request['parameter_indices']=[2]
            elif mode=='alpha':request['interpolation_alpha']=.5
            elif mode=='reference':request['realization_reference_record_sha256']='bad'
            elif mode=='wrong_reference':
                request['realization_reference_record_id']=request['source_record_id'];request['realization_reference_record_sha256']=request['source_record_sha256']
            else:request['phase']='controls_post'
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.evaluate(engine,request,hashes)

    def test_shared_environment_is_lazy_and_single(self):
        calls=[]
        def factory():calls.append(1);return object()
        first=SimpleNamespace(env=None,environment_factory=factory);second=SimpleNamespace(env=None,environment_factory=lambda:object())
        S.share_native_environment(SimpleNamespace(adapter=first),SimpleNamespace(engine=SimpleNamespace(adapter=second)))
        self.assertEqual(calls,[]);self.assertIs(first.environment_factory(),second.environment_factory());self.assertEqual(calls,[1])
        first.env=first.environment_factory()
        with self.assertRaises(ValueError):S.share_native_environment(SimpleNamespace(adapter=first),SimpleNamespace(engine=SimpleNamespace(adapter=second)))

if __name__=='__main__':unittest.main()
