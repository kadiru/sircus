"""Saved bouncer-only candidates never re-solve IK or bypass path admission."""
import copy
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np

from sircus.mjlab import fixed35_bouncer_native as N
from sircus.mjlab.tests import test_fixed35_six_native as six_fixture
from sircus.mjlab.tests import test_calibrated_admission as path_fixture


class ResolverTests(unittest.TestCase):
    def fixture(self):
        branch,hint,arms,endpoint,anchor=six_fixture.BranchTests().fixture()
        branch.base.home=dict(sha256='home',q10=[0.]*6,q14=[0.]*6)
        source=dict(id='source',values=hint['values'],bucket_xy_m=hint['bucket_xy_m'],q10=[0.]*6,q14=[0.]*6,
            context_resolution_sha256=anchor['sha256'],connection_sha256='original-sha',connection_file='fixed-yaw-connections/original.json')
        original=dict(file=source['connection_file'],sha256=source['connection_sha256'],release_behavior={'version':N.BEHAVIOR_VERSION},behavior_key='behavior',passed=True)
        candidate=dict(values=hint['values'],bucket_xy_m=hint['bucket_xy_m'],q10=[0.]*6,q14=[.2]*6,
            endpoint={'reason':0},branch10=0,branch14=1,geometry_eligible=True)
        pool=dict(file='branch-pools/pool.json',sha256='pool-sha',endpoint_attempts={'candidates':[candidate]})
        resolver=N.CandidateResolver(branch.legacy,{'source':source},{'source':original},{'source':pool})
        request=dict(id='request',parent_row_id='source',parent_row_sha256=N.digest(source),values=source['values'],
            candidate_index=0,candidate_pool_file=pool['file'],candidate_pool_sha256=pool['sha256'],
            candidate_sha256=N.digest(candidate),selected_q10=candidate['q10'],selected_q14=candidate['q14'])
        resolver.base.planner.plan.side_effect=lambda target,bucket,**kw:path_fixture.proof(np.asarray(kw.get('start',[0.]*12)),target)
        return resolver,source,request,anchor,candidate,arms,endpoint

    def connect(self,resolver,request,anchor):
        resolver.pending.append(copy.deepcopy(request))
        with patch.object(N,'verify_resolution'),patch.object(N,'native_pose_error',return_value=dict(position_m=0.,rotation_rad=0.)):
            return resolver.connect(anchor,request['values'])

    def test_exact_saved_joints_recheck_gate_without_any_ik(self):
        resolver,source,request,anchor,candidate,arms,endpoint=self.fixture()
        result=self.connect(resolver,request,anchor)
        self.assertTrue(result['passed']);self.assertEqual(result['q10'],source['q10']);self.assertEqual(result['q14'],candidate['q14'])
        self.assertEqual(result['endpoint_attempts']['source_candidate'],candidate)
        self.assertEqual(result['endpoint_attempts']['source_candidate_sha256'],N.digest(candidate))
        self.assertFalse(result['ik_solve_performed']);self.assertEqual(endpoint.check.call_count,1)
        for arm in arms:arm.solve_candidates.assert_not_called()
        N.verify_connection(result,anchor,resolver.home,source['values'])
        again=self.connect(resolver,request,anchor)
        self.assertTrue(again['cache_hit']);self.assertEqual(endpoint.check.call_count,1)
        self.assertEqual(again['sha256'],result['sha256'])

    def test_source_candidate_is_not_overwritten_by_fresh_gate_evidence(self):
        resolver,_,request,anchor,candidate,_,endpoint=self.fixture()
        endpoint.check.return_value={'reason':0,'minimum_clearance_lower_bound_m':.007}
        result=self.connect(resolver,request,anchor)
        self.assertEqual(result['endpoint_attempts']['source_candidate']['endpoint'],{'reason':0})
        self.assertEqual(result['endpoint_attempts']['candidates'][0]['endpoint']['minimum_clearance_lower_bound_m'],.007)
        self.assertEqual(result['candidate_sha256'],N.digest(candidate))

    def test_source_hash_q10_and_nearly_same_bouncer_are_rejected(self):
        for mutation in ('hash','selectedq','dropper','equivalent','values'):
            resolver,source,request,anchor,candidate,arms,endpoint=self.fixture()
            if mutation=='hash':request['candidate_sha256']='bad'
            if mutation=='selectedq':request['selected_q14']=[.3]*6
            if mutation=='values':request['values']=list(source['values']);request['values'][2]+=.01
            if mutation in ('dropper','equivalent'):
                if mutation=='dropper':candidate['q10']=[.1]*6
                else:candidate['q14']=[2*np.pi+.01]*6
                request.update(candidate_sha256=N.digest(candidate),selected_q10=candidate['q10'],selected_q14=candidate['q14'])
            with self.assertRaises(ValueError):self.connect(resolver,request,anchor)
            endpoint.check.assert_not_called()
            for arm in arms:arm.solve_candidates.assert_not_called()

    def test_endpoint_pose_or_path_rejection_never_admits(self):
        for kind in ('endpoint','pose','path','home'):
            resolver,_,request,anchor,_,_,endpoint=self.fixture()
            if kind=='endpoint':endpoint.check.return_value={'reason':7}
            if kind=='path':resolver.planner.plan.side_effect=None;resolver.planner.plan.return_value={'passed':False,'reason':'blocked'}
            if kind=='home':anchor.update(home_passed=False,hard_reject=True)
            resolver.pending.append(request)
            error=dict(position_m=.001 if kind=='pose' else 0.,rotation_rad=0.)
            with patch.object(N,'verify_resolution'),patch.object(N,'native_pose_error',return_value=error):
                result=resolver.connect(anchor,request['values'])
            self.assertFalse(result['passed']);self.assertIsNone(result['proof'])
            if kind!='path':resolver.planner.plan.assert_not_called()

    def test_exact_preservation_uses_source_id_not_behavior_alias(self):
        resolver,source,request,anchor,_,arms,endpoint=self.fixture()
        request['exact_parent_action']=True
        result=self.connect(resolver,request,anchor)
        self.assertEqual(result['file'],source['connection_file']);self.assertEqual(result['sha256'],source['connection_sha256'])
        endpoint.check.assert_not_called();resolver.planner.plan.assert_not_called()
        for arm in arms:arm.solve_candidates.assert_not_called()

    def test_exact_candidate_anchor_suffix_fallback_passes_existing_verifier(self):
        resolver,_,request,anchor,_,_,_=self.fixture();aq=np.full(12,.1)
        anchor.update(passed=True,anchor_admitted=True,home_sha256='home',resolved=dict(q10=aq[:6].tolist(),q14=aq[6:].tolist()),
            proof=path_fixture.proof(np.zeros(12),aq))
        resolver.planner.plan.side_effect=lambda target,bucket,**kw:dict(passed=False,reason='blocked suffix') if kw.get('start') is not None else path_fixture.proof(np.zeros(12),target)
        result=self.connect(resolver,request,anchor)
        self.assertTrue(result['passed']);self.assertEqual(result['proof']['connection_route']['kind'],'full_home_after_anchor_suffix')
        N.verify_connection(result,anchor,resolver.home,request['values'])

    def test_unbound_connect_and_reordered_request_refuse(self):
        resolver,_,request,anchor,_,_,_=self.fixture()
        with self.assertRaisesRegex(ValueError,'identity'):resolver.connect(anchor,request['values'])
        resolver.pending.append(request)
        wrong=list(request['values']);wrong[3]+=.01
        with patch.object(N,'verify_resolution'):
            with self.assertRaisesRegex(ValueError,'changed'):resolver.connect(anchor,wrong)


class EngineTests(unittest.TestCase):
    def test_constructor_keeps_legacy_and_six_proofs_for_same_behavior_separate(self):
        fixture,old,_,anchor,_,_,_=ResolverTests().fixture()
        legacy=fixture.legacy;legacy.base._resolved={};legacy.base.bounds={'mock':True}
        old['context_resolution_file']='resolutions/anchor.json'
        resolution=dict(file=old['context_resolution_file'],sha256=old['context_resolution_sha256'],bucket_xy_m=old['bucket_xy_m'])
        resolution['key']=N.digest(dict(version=N.VERSION,bucket_xy_m=old['bucket_xy_m'],home_sha256=legacy.home['sha256'],
            sources=legacy.sources,bounds=legacy.bounds,anchor_search=N.anchor_search_spec()))
        behavior={'version':N.BEHAVIOR_VERSION,'yaw_rad':0.}
        behavior_key=N.digest(dict(version=N.BEHAVIOR_VERSION,connection_policy=N.CONNECTION_POLICY_VERSION,
            resolution_sha256=resolution['sha256'],values=old['values'],bucket_xy_m=old['bucket_xy_m'],release_behavior=behavior))
        old['connection_file']='fixed-yaw-connections/'+behavior_key+'.json'
        previous=dict(passed=True,release_behavior=behavior,behavior_key=behavior_key,q10=old['q10'],q14=old['q14'])
        fresh=copy.deepcopy(old);fresh.update(id='six-parent',connection_sha256='six-sha')
        poolkey=N.digest(dict(version=N.BRANCH_VERSION,branch_rule=N.BRANCH_RULE,behavior_key=behavior_key,resolution_sha256=resolution['sha256']))
        pool=dict(branch_rule=N.BRANCH_RULE,behavior_key=behavior_key,pool_key=poolkey,
            admitted=[dict(q10=old['q10'],q14=old['q14'],endpoint={'reason':0})])
        branchkey=N.digest(dict(version=N.BRANCH_VERSION,pool_key=poolkey,branch_pool_sha256='pool-sha',ik_branch_slot=0))
        fresh['connection_file']='fixed-yaw-connections/'+branchkey+'.json'
        connection=dict(previous,branch_rule_version=N.BRANCH_VERSION,branch_key=branchkey,ik_branch_slot=0,
            branch_pool_file='branch-pools/'+poolkey+'.json',branch_pool_sha256='pool-sha',endpoint={'reason':0})
        def initialize(engine,*args,**kwargs):
            engine.sources=legacy.sources;engine.adapter=SimpleNamespace(resolver=legacy)
        def copied(origin,directory,relative,expected):
            if relative==old['context_resolution_file']:value=resolution
            elif relative==old['connection_file']:value=previous
            elif relative==fresh['connection_file']:value=connection
            else:value=pool
            return dict(copy.deepcopy(value),file=relative,sha256=expected)
        with patch.object(N.FixedYawEngine,'__init__',initialize),patch.object(N,'copy_bound',side_effect=copied),\
             patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(N,'read',return_value={}):
            engine=N.BouncerEngine({},'assets','out',parent='six',sources=[old,fresh],source_roots={old['id']:'original/native-yaw',fresh['id']:'six/native-yaw'})
        resolver=engine.adapter.resolver
        for source in (old,fresh):
            resolver.pending.append(dict(id='replay-'+source['id'],parent_row_id=source['id'],parent_row_sha256=N.digest(source),values=source['values'],exact_parent_action=True))
            with patch.object(N,'verify_resolution'):
                got=resolver.connect(resolution,source['values'])
            self.assertEqual(got['file'],source['connection_file']);self.assertEqual(got['sha256'],source['connection_sha256'])
        self.assertEqual(len(engine.source_receipts),2)

    def test_queue_preserves_mixed_candidate_order_and_clears_on_error(self):
        engine=N.BouncerEngine.__new__(N.BouncerEngine);engine.directory=Path('native')
        engine.adapter=SimpleNamespace(resolver=SimpleNamespace(pending=deque()))
        requests=[{'id':'new'},{'id':'preserved','exact_parent_action':True},{'id':'other'}]
        def native(group,phase):
            got=[engine.adapter.resolver.pending.popleft()['id'] for _ in group]
            self.assertEqual(got,[r['id'] for r in group])
            return [dict(r,native_release=False,connection_file='fixed-yaw-connections/'+r['id']+'.json') for r in group],{}
        with patch.object(N.RecoveryEngine,'evaluate',side_effect=native),patch.object(N,'read',side_effect=lambda p:dict(bouncer_rule_version=N.BOUNCER_VERSION,bouncer_key=p.stem)):
            rows,_=engine.evaluate(requests,'search')
        self.assertEqual([r['id'] for r in rows],['new','preserved','other']);self.assertFalse(engine.adapter.resolver.pending)
        with patch.object(N.RecoveryEngine,'evaluate',side_effect=RuntimeError('stop')):
            with self.assertRaises(RuntimeError):engine.evaluate(requests,'search')
        self.assertFalse(engine.adapter.resolver.pending)


if __name__=='__main__':unittest.main()
