"""Alternative joint branches, immutable replay and bounded geometry-only pools."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sircus.mjlab import fixed35_six_native as N
from sircus.mjlab.tests import test_fixed35_imitation_native as original_fixture
from sircus.mjlab.tests.test_fixed35_imitation_native import Array
from sircus.mjlab.tests import test_calibrated_admission as path_fixture
from sircus.mjlab.tests.test_fixed35_recovery_contract import fixture as recovery_fixture


class BranchTests(unittest.TestCase):
    def fixture(self):
        legacy,hint,arms,endpoint = original_fixture.NativeTests().fixture()
        legacy.base.sources=[];legacy.base.home={'sha256':'home'}
        legacy.base._check_resolution_bytes=Mock()
        legacy.base._save=Mock(side_effect=lambda folder,key,record:dict(file=folder+'/'+key+'.json',sha256=N.digest(record)))
        legacy.base.planner=SimpleNamespace(plan=Mock(return_value=dict(passed=True,waypoints=[[0.]*12])))
        resolver=N.BranchResolver(legacy)
        anchor=dict(bucket_xy_m=hint['bucket_xy_m'],sha256='anchor',file='anchor.json',home_passed=True,
            hard_reject=False,passed=False,anchor_values=hint['values'],anchor_derivation={},effective_source_hint=hint)
        return resolver,hint,arms,endpoint,anchor

    def solve(self,resolver,hint):
        with patch.object(N,'outer_geometry_mask',return_value=(np.ones((1,1),bool),np.ones((1,1),bool))),\
             patch.object(N,'native_pose_error',return_value=dict(position_m=0.,rotation_rad=0.)):
            return resolver.solve(hint['bucket_xy_m'],hint['values'],hint)

    def test_full_pose_geometry_only_multi_seed_contract(self):
        resolver,hint,arms,endpoint,_=self.fixture()
        result=self.solve(resolver,hint)
        self.assertTrue(result['passed']);self.assertFalse(result['selection_uses_flight_labels'])
        self.assertEqual(result['geometry_hint_sha256'],N.digest(N.source_geometry(hint)))
        self.assertEqual(endpoint.check.call_count,1)
        for arm in arms:
            self.assertEqual(arm.solve_candidates.call_count,2)
            self.assertEqual([c.kwargs['seed'] for c in arm.solve_candidates.call_args_list],N.BRANCH_RULE['seeds'])
            for c in arm.solve_candidates.call_args_list:
                self.assertEqual((c.kwargs['starts'],c.kwargs['iterations']),(32,360))
                self.assertFalse(c.kwargs['normal_only']);self.assertFalse(c.kwargs['preserve_known'][0])

    def test_wrapped_and_tiny_joint_variations_do_not_make_new_posture(self):
        resolver,hint,arms,endpoint,_=self.fixture()
        q=np.zeros((1,4,6));q[0,1,5]=2*np.pi;q[0,2,0]=.02;q[0,3,0]=.3
        for arm in arms:
            arm.solve_candidates.side_effect=lambda *a,**kw:{k:Array(v) for k,v in dict(q=q,valid=np.ones((1,4),bool),residual=np.zeros((1,4))).items()}
        result=self.solve(resolver,hint)
        self.assertEqual(len(result['candidates']),4)
        for i,first in enumerate(result['candidates']):
            for second in result['candidates'][i+1:]:
                self.assertFalse(N.same_branch(first['q10']+first['q14'],second['q10']+second['q14']))
        self.assertTrue(N.same_branch([0.]*12,[2*np.pi]*12))
        self.assertTrue(N.same_branch([0.]*12,[.049]*12))
        self.assertFalse(N.same_branch([0.]*12,[.051]+[0.]*11))

    def test_endpoint_and_unique_pair_budgets_are_bounded(self):
        for pass_endpoint,expected in ((True,12),(False,64)):
            resolver,hint,arms,endpoint,_=self.fixture()
            q=np.zeros((1,16,6));q[0,:,0]=np.arange(16)*.15
            for arm in arms:
                arm.solve_candidates.side_effect=lambda *a,**kw:{k:Array(v) for k,v in dict(q=q,valid=np.ones((1,16),bool),residual=np.zeros((1,16))).items()}
            endpoint.check.return_value=dict(reason=None if pass_endpoint else 'blocked',independent_reject_arms=[])
            result=self.solve(resolver,hint)
            self.assertEqual(endpoint.check.call_count,expected)
            self.assertEqual(len(result['candidates']),12 if pass_endpoint else 0)

    def test_pose_error_and_bounds_remain_hard_rejections(self):
        resolver,hint,arms,_,_=self.fixture()
        with patch.object(N,'outer_geometry_mask',return_value=(np.ones((1,1),bool),np.ones((1,1),bool))),\
             patch.object(N,'native_pose_error',return_value=dict(position_m=.001,rotation_rad=0.)):
            result=resolver.solve(hint['bucket_xy_m'],hint['values'],hint)
        self.assertFalse(result['passed']);self.assertTrue(result['pair_attempts'][0]['pose_rejected'])
        for arm in arms:arm.solve_candidates.reset_mock()
        hint['values'][4]=-1
        result=self.solve(resolver,hint)
        self.assertEqual(result['reason'],'action_bounds_rejected')
        for arm in arms:arm.solve_candidates.assert_not_called()

    def endpoint(self,hint,count=3):
        return dict(passed=bool(count),reason=None,candidates=[dict(values=hint['values'],bucket_xy_m=hint['bucket_xy_m'],q10=[i*.2]*6,q14=[0.]*6,endpoint={'reason':None}) for i in range(count)],pair_attempts=[])

    def test_two_slots_share_ik_path_pool_but_have_distinct_exact_proofs(self):
        resolver,hint,_,_,anchor=self.fixture()
        with patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(resolver,'solve',return_value=self.endpoint(hint)) as solve:
            resolver.active_slot=0;first=resolver.connect(anchor,hint['values'])
            resolver.active_slot=1;second=resolver.connect(anchor,hint['values'])
            again=resolver.connect(anchor,hint['values'])
        self.assertEqual(solve.call_count,1);self.assertEqual(resolver.planner.plan.call_count,2)
        self.assertTrue(first['passed']);self.assertTrue(second['passed']);self.assertTrue(again['cache_hit'])
        self.assertNotEqual(first['q10'],second['q10']);self.assertNotEqual(first['file'],second['file'])
        self.assertEqual(first['behavior_key'],second['behavior_key'])
        self.assertEqual(first['branch_pool_sha256'],second['branch_pool_sha256'])
        self.assertFalse(first['branch_pool_cache_hit']);self.assertTrue(second['branch_pool_cache_hit'])
        self.assertEqual(second['timing']['endpoint_s'],0.)

    def test_absent_second_branch_cannot_silently_repeat_first(self):
        resolver,hint,_,_,anchor=self.fixture()
        with patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(resolver,'solve',return_value=self.endpoint(hint,1)):
            resolver.active_slot=0;first=resolver.connect(anchor,hint['values'])
            resolver.active_slot=1;second=resolver.connect(anchor,hint['values'])
        self.assertTrue(first['passed']);self.assertFalse(second['passed'])
        self.assertEqual(second['reason'],'requested_distinct_branch_unavailable')
        self.assertIsNone(second['proof']);self.assertEqual(second['q10'],[0.]*6)

    def test_fixed_base_and_all_path_declines_remain_declines(self):
        resolver,hint,_,_,anchor=self.fixture();resolver.active_slot=0
        anchor.update(home_passed=False,hard_reject=True)
        with patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(resolver,'solve') as solve:
            result=resolver.connect(anchor,hint['values'])
        solve.assert_not_called();self.assertEqual(result['reason'],'fixed_base_clearance_rejected')
        resolver,hint,_,_,anchor=self.fixture();resolver.active_slot=0
        resolver.planner.plan.return_value={'passed':False,'reason':'blocked'}
        with patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(resolver,'solve',return_value=self.endpoint(hint,12)):
            result=resolver.connect(anchor,hint['values'])
        self.assertFalse(result['passed']);self.assertEqual(result['reason'],'finite_requested_setup_path_unresolved')
        self.assertEqual(resolver.planner.plan.call_count,12)

    def test_legacy_exact_parent_cache_is_separate_and_unchanged(self):
        resolver,hint,_,_,anchor=self.fixture()
        legacy={'file':'legacy.json','sha256':'unchanged','q10':[1.]*6}
        with patch.object(resolver.legacy,'connect',return_value=legacy) as old,patch.object(resolver,'_pool') as pool:
            self.assertIs(resolver.connect(anchor,hint['values']),legacy)
        old.assert_called_once();pool.assert_not_called()

    def test_two_admitted_branches_keep_global_suffix_fallback_indices(self):
        resolver,hint,_,_,anchor=self.fixture()
        home=dict(sha256='home',q10=[0.]*6,q14=[0.]*6)
        resolver.base.home=home
        aq=np.asarray([.1]*12)
        anchor.update(passed=True,anchor_admitted=True,home_sha256='home',
            resolved=dict(q10=aq[:6].tolist(),q14=aq[6:].tolist(),values=hint['values'],bucket_xy_m=hint['bucket_xy_m']),
            proof=path_fixture.proof(np.zeros(12),aq))
        def plan(target,bucket,*,start=None):
            return dict(passed=False,reason='suffix blocked') if start is not None else path_fixture.proof(np.zeros(12),target)
        resolver.base.planner.plan.side_effect=plan
        with patch.object(N,'verify_resolution'),patch.object(resolver,'solve',return_value=self.endpoint(hint)):
            resolver.active_slot=0;first=resolver.connect(anchor,hint['values'])
            resolver.active_slot=1;second=resolver.connect(anchor,hint['values'])
        N.verify_connection(first,anchor,home,hint['values'])
        N.verify_connection(second,anchor,home,hint['values'])
        self.assertEqual(first['proof']['connection_route']['suffix_attempt_index'],0)
        self.assertEqual(second['proof']['connection_route']['suffix_attempt_index'],2)
        self.assertEqual(len(second['path_attempts']),4)


class EngineTests(unittest.TestCase):
    def test_source_roots_preserve_both_original_provenances(self):
        original=recovery_fixture()['preserved'][0];new=copy.deepcopy(original);new['id']='new-confirmed'
        sources=[original,new];geometry=[{'id':'geometry-anchor'}];home={'sha256':'home'}
        legacy=SimpleNamespace(sources=geometry,home=home,bounds={'mock':True},base=SimpleNamespace(_resolved={}),_connected={})
        resolution=dict(sha256=original['context_resolution_sha256'],bucket_xy_m=original['bucket_xy_m'])
        resolution['key']=N.digest(dict(version=N.VERSION,bucket_xy_m=original['bucket_xy_m'],home_sha256='home',sources=geometry,bounds=legacy.bounds,anchor_search=N.anchor_search_spec()))
        behavior=dict(version=N.BEHAVIOR_VERSION,yaw_rad=0.)
        key=N.digest(dict(version=N.BEHAVIOR_VERSION,connection_policy=N.CONNECTION_POLICY_VERSION,resolution_sha256=resolution['sha256'],values=original['values'],bucket_xy_m=original['bucket_xy_m'],release_behavior=behavior))
        for source in sources:source['connection_file']='fixed-yaw-connections/'+key+'.json'
        connection=dict(behavior_key=key,release_behavior=behavior,sha256=original['connection_sha256'])
        def initialize(engine,*args,**kwargs):
            engine.sources=geometry;engine.adapter=SimpleNamespace(resolver=legacy)
        def copied(origin,directory,relative,expected):
            return dict(copy.deepcopy(resolution if relative==original['context_resolution_file'] else connection),file=relative)
        roots={original['id']:Path('original/native-yaw'),new['id']:Path('recovery/native-yaw')}
        with patch.object(N.FixedYawEngine,'__init__',initialize),patch.object(N,'copy_bound',side_effect=copied) as copies,\
             patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(N,'read',return_value={}):
            engine=N.SixEngine({},'assets','target',parent='recovery',sources=sources,source_roots=roots)
        self.assertEqual([c.args[0] for c in copies.call_args_list],[roots[original['id']]]*2+[roots[new['id']]]*2)
        self.assertEqual(engine.sources,geometry)
        self.assertEqual([r['source_id'] for r in engine.source_receipts],[s['id'] for s in sources])
        self.assertEqual(legacy._connected[key]['sha256'],original['connection_sha256'])
        self.assertIs(engine.adapter.resolver.legacy,legacy)

    def test_slot_grouping_keeps_request_order_and_adds_exact_pool_binding(self):
        engine=N.SixEngine.__new__(N.SixEngine);engine.directory=Path('native')
        engine.adapter=SimpleNamespace(resolver=SimpleNamespace(active_slot=None))
        requests=[dict(id='one',ik_branch_slot=1),dict(id='zero',ik_branch_slot=0),dict(id='old',exact_parent_action=True),dict(id='two',ik_branch_slot=1)]
        calls=[]
        def native(group,phase):
            slot=engine.adapter.resolver.active_slot;calls.append((slot,[r['id'] for r in group]))
            return [dict(r,connection_file='fixed-yaw-connections/key'+str(slot)+'.json') for r in group],dict(requested=len(group),native_releases=len(group),gpu_s=2.)
        def read(path):
            slot=int(path.stem[-1]);return dict(ik_branch_slot=slot,branch_rule_version=N.BRANCH_VERSION,
                branch_key=path.stem,branch_pool_file='branch-pools/pool.json',branch_pool_sha256='pool-sha',branch_pool_key='pool')
        with patch.object(N.RecoveryEngine,'evaluate',side_effect=native),patch.object(N,'read',side_effect=read):
            rows,timing=engine.evaluate(requests,'search')
        self.assertEqual([r['id'] for r in rows],[r['id'] for r in requests])
        self.assertEqual(calls,[(None,['old']),(0,['zero']),(1,['one','two'])])
        self.assertEqual(timing['requested'],4);self.assertEqual(timing['gpu_s'],6.)
        self.assertIsNone(engine.adapter.resolver.active_slot)
        self.assertEqual(rows[0]['branch_pool_sha256'],'pool-sha')

    def test_new_requests_require_slot_and_preservation_forbids_slot(self):
        engine=N.SixEngine.__new__(N.SixEngine)
        for request in (dict(id='x'),dict(id='x',ik_branch_slot=True),dict(id='x',ik_branch_slot=2),dict(id='x',exact_parent_action=True,ik_branch_slot=0)):
            with patch.object(N.RecoveryEngine,'evaluate') as native:
                with self.assertRaises(ValueError):engine.evaluate([request],'search')
                native.assert_not_called()

    def test_failed_native_call_clears_active_branch(self):
        engine=N.SixEngine.__new__(N.SixEngine);engine.adapter=SimpleNamespace(resolver=SimpleNamespace(active_slot=None))
        with patch.object(N.RecoveryEngine,'evaluate',side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):engine.evaluate([dict(id='x',ik_branch_slot=1)],'search')
        self.assertIsNone(engine.adapter.resolver.active_slot)


if __name__=='__main__':unittest.main()
