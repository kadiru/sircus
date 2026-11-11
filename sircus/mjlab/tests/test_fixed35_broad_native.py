"""Reach filtering and bouncer-diverse CPU preparation never substitute physics."""
import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from collections import deque
import json
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation
from sircus.mjlab import fixed35_broad_native as N
from sircus.mjlab.tests import test_fixed35_imitation_native as original_fixture
from sircus.mjlab.tests import test_calibrated_admission as path_fixture
from sircus.mjlab.tests.test_fixed35_imitation_native import Array, SCENE


class ReachTests(unittest.TestCase):
    def model(self):
        return dict(shoulder_m=[0,0,.364],terminal_axial_m=.09,outer_radius_m=.8416370223974)

    def test_conditional_gap_is_exact_downward_sphere_intersection(self):
        model=self.model(); models=dict(dropper=model,bouncer=dict(model,outer_radius_m=20.))
        with patch.object(N,'reach_models',return_value=models):
            interval=N.drop_gap_interval('assets',SCENE,[.2,-.6],[.8,0,.2],28)
            expected=.364-.2-.0375-.095-.09+math.sqrt(model['outer_radius_m']**2-.8**2)
            self.assertAlmostEqual(interval[0],.05);self.assertAlmostEqual(interval[1],expected,places=13)
            self.assertIsNone(N.drop_gap_interval('assets',SCENE,[.2,-.6],[.8,0,.4],28))
            for gap,wanted in ((expected-1e-7,True),(expected+1e-7,False)):
                self.assertEqual(N.reach_bound('assets',SCENE,[.2,-.6],[.8,0,.2,28,gap])['passed'],wanted)
            # The interval can impose a lower bound too, rather than silently accepting it.
            low_model=dict(model,shoulder_m=[0,0,1.4],outer_radius_m=.4)
            with patch.object(N,'reach_models',return_value=dict(dropper=low_model,bouncer=models['bouncer'])):
                lower=N.drop_gap_interval('assets',SCENE,[.7,-.6],[.1,0,.2],28)
                self.assertGreater(lower[0],.05)

    def test_bouncer_and_horizontal_unreachability_reject_before_ik(self):
        models=dict(dropper=self.model(),bouncer=dict(self.model(),outer_radius_m=.01))
        with patch.object(N,'reach_models',return_value=models):
            self.assertIsNone(N.drop_gap_interval('assets',SCENE,[.2,-.6],[.8,0,.2],28))
        models['bouncer']['outer_radius_m']=20.
        with patch.object(N,'reach_models',return_value=models):
            self.assertIsNone(N.drop_gap_interval('assets',SCENE,[.2,-.6],[.9,0,.2],28))

    def test_urdf_terms_tolerance_and_asset_reads_cached(self):
        origins=np.tile(np.eye(4),(6,1,1));origins[0,:3,3]=[0,0,.364]
        origins[1,:3,3]=[0,0,.39];origins[2,:3,3]=[.15,0,.426]
        origins[5,:3,:3]=Rotation.from_euler('x',math.pi/2).as_matrix();origins[5,:3,3]=[0,-.09,0]
        axes=np.tile([0,0,1.],(6,1));N.reach_models.cache_clear()
        with patch.object(N,'read_chain',return_value=(origins,axes,None,None)) as chain,patch.object(N,'sha',return_value='asset'):
            first=N.reach_models('assets');again=N.reach_models('assets')
        self.assertIs(first,again);self.assertEqual(chain.call_count,2)
        m=first['dropper'];self.assertAlmostEqual(m['upstream_length_upper_m'],.8416370223974)
        self.assertAlmostEqual(m['terminal_axial_m'],.09)
        self.assertAlmostEqual(m['accepted_pose_allowance_m'],.0001+.18*math.sin(.0002))
        self.assertGreater(m['outer_radius_m'],m['upstream_length_upper_m'])
        N.reach_models.cache_clear()


class BroadTests(unittest.TestCase):
    def fixture(self):
        legacy,hint,arms,endpoint=original_fixture.NativeTests().fixture()
        legacy.base.sources=[];legacy.base.home={'sha256':'home'}
        legacy.base._check_resolution_bytes=Mock()
        legacy.base._save=Mock(side_effect=lambda folder,key,record:dict(file=folder+'/'+key+'.json',sha256=N.digest(record)))
        legacy.base.planner=SimpleNamespace(plan=Mock(return_value=dict(passed=True,waypoints=[[0.]*12])))
        resolver=N.BroadResolver(legacy,{}, {}, 'assets')
        anchor=dict(bucket_xy_m=hint['bucket_xy_m'],sha256='anchor',file='anchor.json',home_passed=True,
            hard_reject=False,passed=False,anchor_values=hint['values'],anchor_derivation={},effective_source_hint=hint)
        return resolver,hint,arms,endpoint,anchor

    def solve(self,resolver,hint):
        with patch.object(N,'reach_bound',return_value={'passed':True}),patch.object(N,'native_pose_error',return_value=dict(position_m=0.,rotation_rad=0.)):
            return resolver.solve(hint['bucket_xy_m'],hint['values'],hint)

    def test_full_pose_and_fixed_seed_with_no_flight_labels(self):
        resolver,hint,arms,endpoint,_=self.fixture();result=self.solve(resolver,hint)
        self.assertTrue(result['passed']);self.assertFalse(result['selection_uses_flight_labels'])
        self.assertEqual(result['geometry_hint_sha256'],N.digest(N.source_geometry(hint)))
        for arm in arms:
            self.assertEqual(arm.solve_candidates.call_count,1)
            kw=arm.solve_candidates.call_args.kwargs
            self.assertEqual((kw['starts'],kw['iterations']),(32,360));self.assertFalse(kw['normal_only'])
            self.assertFalse(kw['preserve_known'][0]);self.assertEqual(kw['seed'],N.action_seed(hint['bucket_xy_m'],hint['values']))
        other=list(hint['values']);other[2]+=.001
        self.assertNotEqual(N.action_seed(hint['bucket_xy_m'],hint['values']),N.action_seed(hint['bucket_xy_m'],other))

    def test_q14_proximal_diversity_precedes_wrist_and_dropper_flip(self):
        resolver,hint,arms,_,_=self.fixture()
        q10=np.zeros((1,2,6));q10[0,1,5]=math.pi
        q14=np.zeros((1,4,6));q14[0,1,5]=math.pi;q14[0,2,0]=.7;q14[0,3,0]=.71
        for arm,q in zip(arms,(q10,q14)):
            arm.solve_candidates.side_effect=lambda *a,q=q,**kw:{k:Array(v) for k,v in dict(q=q,valid=np.ones(q.shape[:2],bool),residual=np.zeros(q.shape[:2])).items()}
        result=self.solve(resolver,hint)
        self.assertEqual(result['distinct_ik_branches'],[2,3])
        self.assertEqual(result['bouncer_branch_order'][0],2)
        self.assertEqual([r['branch10'] for r in result['candidates'][:3]],[0,0,0])
        self.assertEqual(set(r['branch14'] for r in result['candidates'][:3]),{0,1,2})

    def test_bounds_reach_and_strict_fk_errors_remain_rejections(self):
        resolver,hint,arms,_,_=self.fixture()
        with patch.object(N,'reach_bound',return_value={'passed':False}):
            self.assertEqual(resolver.solve(hint['bucket_xy_m'],hint['values'],hint)['reason'],'proven_full_pose_wrist_reach_rejected')
        for arm in arms:arm.solve_candidates.assert_not_called()
        with patch.object(N,'reach_bound',return_value={'passed':True}),patch.object(N,'native_pose_error',return_value=dict(position_m=0.,rotation_rad=.0005)):
            result=resolver.solve(hint['bucket_xy_m'],hint['values'],hint)
        self.assertFalse(result['passed']);self.assertTrue(result['pair_attempts'][0]['pose_rejected'])

    def test_pool_reuses_cpu_work_and_never_admits_q10_only_alternative(self):
        resolver,hint,_,_,anchor=self.fixture()
        candidates=[]
        for q10,q14 in ((0,0),(.5,0),(0,.3)):
            candidates.append(dict(values=hint['values'],bucket_xy_m=hint['bucket_xy_m'],q10=[q10]*6,q14=[q14]*6,endpoint={'reason':None}))
        with patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(resolver,'solve',return_value=dict(candidates=candidates,passed=True,reason=None)) as solve:
            pool=resolver.pool(anchor,hint['values']);again=resolver.pool(anchor,hint['values'])
            a=resolver.prepared_connection(anchor,pool,0);b=resolver.prepared_connection(anchor,pool,1)
        self.assertEqual(solve.call_count,1);self.assertEqual(resolver.planner.plan.call_count,2)
        self.assertEqual([p['candidate_index'] for p in pool['admitted']],[0,2]);self.assertEqual(again,pool)
        self.assertEqual(a['behavior_key'],b['behavior_key']);self.assertNotEqual(a['broad_key'],b['broad_key'])
        self.assertNotEqual(a['q14'],b['q14']);self.assertEqual(a['q10'],b['q10'])

    def test_bounded_endpoint_and_pair_checks(self):
        for accepted,budget in ((True,16),(False,64)):
            resolver,hint,arms,endpoint,_=self.fixture();q=np.zeros((1,16,6));q[0,:,0]=np.arange(16)*.1
            for arm in arms:arm.solve_candidates.side_effect=lambda *a,**kw:{k:Array(v) for k,v in dict(q=q,valid=np.ones(q.shape[:2],bool),residual=np.zeros(q.shape[:2])).items()}
            endpoint.check.return_value=dict(reason=None if accepted else 'blocked',independent_reject_arms=[])
            result=self.solve(resolver,hint);self.assertEqual(endpoint.check.call_count,budget)
            self.assertEqual(len(result['candidates']),16 if accepted else 0)

    def test_prepared_connect_reads_exact_bytes_and_does_not_solve(self):
        resolver,hint,_,_,anchor=self.fixture();source=dict(hint,id='parent');resolver.parent_rows={'parent':source}
        connection=dict(passed=True,q10=[0.]*6,q14=[.2]*6,**dict(zip(N.BROAD_FIELDS,[N.BROAD_VERSION,0,'key','pool.json','pool-sha','pool-key'])))
        with TemporaryDirectory() as tmp:
            resolver.base.directory=Path(tmp);path=Path(tmp)/'connection.json';path.write_text(json.dumps(connection));checksum=N.sha(path)
            request=dict(parent_row_id='parent',parent_row_sha256=N.digest(source),bucket_xy_m=hint['bucket_xy_m'],values=hint['values'],native_prepared=True,prepared_connection_file='connection.json',prepared_connection_sha256=checksum,selected_q10=connection['q10'],selected_q14=connection['q14'],**{k:connection[k] for k in N.BROAD_FIELDS})
            with patch.object(N,'verify_connection'),patch.object(resolver,'solve') as solve,patch.object(resolver,'pool') as pool:
                resolver.pending.append(request);result=resolver.connect(anchor,hint['values'])
                self.assertEqual(result['sha256'],checksum);solve.assert_not_called();pool.assert_not_called()
                bad=dict(request,selected_q14=[0.]*6);resolver.pending.append(bad)
                with self.assertRaisesRegex(ValueError,'joint branch'):resolver.connect(anchor,hint['values'])
                path.write_text('{}');resolver.pending.append(request)
                with self.assertRaisesRegex(ValueError,'bytes'):resolver.connect(anchor,hint['values'])

    def test_parent_proof_is_preserved_without_new_ik_or_path(self):
        resolver,hint,_,_,anchor=self.fixture();source=dict(hint,id='parent');resolver.parent_rows={'parent':source}
        original=dict(file='old.json',sha256='old-sha',q10=[0.]*6,q14=[.4]*6)
        resolver.parent_connections={'parent':copy.deepcopy(original)}
        request=dict(parent_row_id='parent',parent_row_sha256=N.digest(source),bucket_xy_m=hint['bucket_xy_m'],values=hint['values'],exact_parent_action=True)
        with patch.object(resolver,'pool') as pool,patch.object(resolver,'solve') as solve:
            resolver.pending.append(request);result=resolver.connect(anchor,hint['values'])
        self.assertEqual({k:result[k] for k in original},original);self.assertTrue(result['cache_hit'])
        pool.assert_not_called();solve.assert_not_called();resolver.planner.plan.assert_not_called()

    def test_no_new_path_accepted_after_all_strict_path_failures(self):
        resolver,hint,_,_,anchor=self.fixture();resolver.planner.plan.return_value=dict(passed=False,reason='sampled_collision')
        rows=[dict(values=hint['values'],bucket_xy_m=hint['bucket_xy_m'],q10=[0.]*6,q14=[i*.1]*6,endpoint={'reason':None}) for i in range(16)]
        with patch.object(N,'verify_resolution'),patch.object(resolver,'solve',return_value=dict(candidates=rows,passed=True,reason=None)):
            pool=resolver.pool(anchor,hint['values'])
        self.assertEqual(pool['admitted'],[]);self.assertEqual(pool['reason'],'finite_requested_setup_path_unresolved')
        self.assertEqual(resolver.planner.plan.call_count,16)

    def test_anchor_suffix_fallback_indices_remain_global_across_branches(self):
        resolver,hint,_,_,anchor=self.fixture()
        home=dict(sha256='home',q10=[0.]*6,q14=[0.]*6);resolver.base.home=home
        aq=np.asarray([.1]*12)
        anchor.update(passed=True,anchor_admitted=True,home_sha256='home',
            resolved=dict(q10=aq[:6].tolist(),q14=aq[6:].tolist(),values=hint['values'],bucket_xy_m=hint['bucket_xy_m']),
            proof=path_fixture.proof(np.zeros(12),aq))
        def plan(target,bucket,*,start=None):
            return dict(passed=False,reason='suffix blocked') if start is not None else path_fixture.proof(np.zeros(12),target)
        resolver.planner.plan.side_effect=plan
        rows=[dict(values=hint['values'],bucket_xy_m=hint['bucket_xy_m'],q10=[.2]*6,q14=[i*.3]*6,endpoint={'reason':None}) for i in range(2)]
        with patch.object(N,'verify_resolution'),patch.object(resolver,'solve',return_value=dict(candidates=rows,passed=True,reason=None)):
            pool=resolver.pool(anchor,hint['values'])
            first=resolver.prepared_connection(anchor,pool,0);second=resolver.prepared_connection(anchor,pool,1)
        self.assertEqual(len(pool['path_attempts']),4)
        self.assertEqual(first['proof']['connection_route']['suffix_attempt_index'],0)
        self.assertEqual(second['proof']['connection_route']['suffix_attempt_index'],2)
        N.verify_connection(first,anchor,home,hint['values']);N.verify_connection(second,anchor,home,hint['values'])


class PreparationTests(unittest.TestCase):
    def fixture(self):
        engine=N.BroadEngine.__new__(N.BroadEngine);engine._prepared_ids=set()
        source=dict(id='parent',bucket_xy_m=[1.05,.025],values=[.6,0,.3,28,.2]);engine.parent_rows={'parent':source}
        proposal=dict(id='p0',parent_row_id='parent',parent_row_sha256=N.digest(source),bucket_xy_m=source['bucket_xy_m'],values=source['values'],phase='search',context_id='03-05',exact_parent_action=False)
        anchor=dict(file='anchor.json',sha256='anchor',source_hint=dict(source,q10=[0.]*6,q14=[0.]*6),effective_source_hint=dict(source,q10=[0.]*6,q14=[0.]*6),layout_eligible=True,anchor_admitted=False,hard_reject=False)
        pool=dict(file='pool.json',sha256='pool-sha',pool_key='key',admitted=[0,1],reason=None,timing={})
        def connection(anchor,pool,slot):
            return dict(file='connection'+str(slot)+'.json',sha256='connection-sha'+str(slot),q10=[0.]*6,q14=[slot*.5]*6,**dict(zip(N.BROAD_FIELDS,[N.BROAD_VERSION,slot,'key'+str(slot),'pool.json','pool-sha','key'])))
        resolver=SimpleNamespace(pending=deque(),home={'sha256':'home'},pool=Mock(return_value=pool),prepared_connection=Mock(side_effect=connection))
        engine.adapter=SimpleNamespace(resolver=resolver,env=None,resolve_context=Mock(return_value=(anchor,{})),_environment=Mock(side_effect=AssertionError('GPU forbidden')))
        return engine,proposal,pool

    def test_prepare_is_cpu_only_binds_both_packets_and_rejects_duplicate_ids(self):
        engine,proposal,_=self.fixture();packets,records=engine.prepare([proposal])
        self.assertEqual([r['id'] for r in packets],['p0-b0','p0-b1']);self.assertEqual(records[0]['native_releases'],0)
        self.assertEqual(records[0]['admitted_request_sha256'],[N.digest(r) for r in packets])
        self.assertTrue(all(r['native_prepared'] for r in packets));engine.adapter._environment.assert_not_called()
        with self.assertRaisesRegex(ValueError,'Unique'):engine.prepare([proposal])
        self.assertEqual(engine.adapter.resolver.pool.call_count,1)

    def test_decline_has_no_native_packet_and_no_physics(self):
        engine,proposal,pool=self.fixture();pool.update(admitted=[],reason='strict_endpoint_clearance_or_pose_rejected')
        packets,records=engine.prepare([proposal]);self.assertEqual(packets,[])
        self.assertEqual(records[0]['admitted'],0);self.assertEqual(records[0]['reason'],pool['reason'])
        engine.adapter._environment.assert_not_called();engine.adapter.resolver.prepared_connection.assert_not_called()

    def test_native_queue_cleared_on_error(self):
        engine,proposal,_=self.fixture()
        with patch.object(N.RecoveryEngine,'evaluate',side_effect=RuntimeError('native failed')):
            with self.assertRaisesRegex(RuntimeError,'native failed'):engine.evaluate([proposal],'search')
        self.assertEqual(list(engine.adapter.resolver.pending),[])


if __name__=='__main__':unittest.main()
