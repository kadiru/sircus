"""CPU/mock admission controls; no GPU, ball-flight labels or robot connection."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab.calibrated_admission import (
    CLEARANCE_M, HOME_VERSION, SAMPLE_STEP_RAD, SCOPE, GeometryResolver,
    SerializedPlanner, StrictSolver, _signed, rank_sources, source_geometry,
    validate_home, verify_connection, verify_path, verify_resolution, anchor_candidates, anchor_search_spec,
    CONNECTION_POLICY_VERSION,
)
from sircus.mjlab.learning_contract import digest


def source(name, bucket=(.4, -.4)):
    return dict(id=name, bucket_xy_m=list(bucket), values=[.6, -.2, .3, 30., .2], q10=[0.]*6, q14=[0.]*6,
                route_success=True, capture_rate=1.)


def home(scene):
    return _signed(dict(version=HOME_VERSION, passed=True, scene_sha256=digest(scene), q10=[0.]*6, q14=[0.]*6))


def proof(start, target):
    middle = np.asarray(start).copy(); middle[6:] = target[6:]
    points = [np.asarray(start), middle, np.asarray(target)]
    edges = []
    for a,b in zip(points,points[1:]):
        n=max(1,int(np.ceil(np.abs(b-a).max()/SAMPLE_STEP_RAD)))
        edges.append(dict(passed=True, minimum_clearance_m=.006, total_samples=n+1,
                          maximum_actual_joint_sample_step_rad=float(np.abs(b-a).max()/n)))
    return dict(passed=True,scope=SCOPE,waypoints=[q.tolist() for q in points],edges=edges,
                reverse_edges=copy.deepcopy(edges[::-1]),minimum_clearance_m=.006)


class FakeSolver:
    def __init__(self, reject_nominal=False): self.calls=[];self.reject_nominal=reject_nominal
    def solve(self,bucket,values,hint):
        self.calls.append((copy.deepcopy(bucket),copy.deepcopy(values),hint['id']))
        if self.reject_nominal and not np.isclose(values[0],.61):
            return dict(passed=False,reason='finite_ik_unresolved',candidates=[])
        row=dict(bucket_xy_m=list(bucket),values=list(values),q10=[values[0]]+[0.]*5,
                 q14=[values[1]]+[0.]*5,endpoint={'reason':0})
        return dict(passed=True,reason=None,candidates=[row])


class FakePlanner:
    def __init__(self,h,blocked=False): self.home=h;self.blocked=blocked;self.calls=[]
    def check_home(self,bucket): return dict(passed=not self.blocked,hard_reject=self.blocked,reason='clearance' if self.blocked else None)
    def plan(self,target,bucket,*,start=None):
        self.calls.append((np.asarray(target).tolist(),list(bucket),None if start is None else np.asarray(start).tolist()))
        return proof(np.r_[self.home['q10'],self.home['q14']] if start is None else np.asarray(start),np.asarray(target))


class ResolverTests(unittest.TestCase):
    def build(self,directory,*,reject_nominal=False,blocked=False):
        scene={};h=home(scene);solver=FakeSolver(reject_nominal);planner=FakePlanner(h,blocked)
        resolver=GeometryResolver({'scene':scene},directory,Path(directory)/'out',[source('z'),source('a')],
            home=h,solver=solver,planner=planner)
        return resolver,solver,planner

    def test_sources_strip_flight_labels_and_ties_are_by_id(self):
        stripped=source_geometry(source('a'))
        self.assertNotIn('route_success',stripped);self.assertNotIn('capture_rate',stripped)
        self.assertEqual([s['id'] for s in rank_sources([source('z'),source('a')],[.5,-.4])],['a','z'])

    def test_resolve_is_hash_bound_cached_and_chooses_first_geometry_source(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver,solver,planner=self.build(directory)
            row=resolver.resolve([.5,-.4]);self.assertTrue(row['passed']);self.assertEqual(row['source_hint']['id'],'a')
            self.assertEqual(len(row['attempts']),1);verify_resolution(row,resolver.sources,resolver.home)
            path=resolver.directory/row['file'];before=path.read_bytes()
            self.assertEqual(hashlib.sha256(before).hexdigest(),row['sha256'])
            again=resolver.resolve([.5,-.4]);self.assertTrue(again['cache_hit']);self.assertEqual(len(solver.calls),1)
            self.assertEqual(path.read_bytes(),before)

    def test_failed_nominal_anchor_does_not_veto_actual_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver,solver,planner=self.build(directory,reject_nominal=True)
            row=resolver.resolve([.5,-.4]);self.assertFalse(row['passed']);self.assertTrue(row['home_passed'])
            self.assertEqual(len(row['attempts']),34);verify_resolution(row,resolver.sources,resolver.home)
            requested=[.61,-.2,.3,30.,.2]
            result=resolver.connect(row,requested)
            self.assertTrue(result['passed']);self.assertEqual(result['values'],requested)
            self.assertFalse(result['action_substituted']);verify_connection(result,row,resolver.home,requested)
            self.assertIsNone(planner.calls[-1][2]);self.assertNotIn('connection_route',result['proof'])

    def test_exact_anchor_joints_reuse_hash_bound_prefix_without_planner_call(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver,solver,planner=self.build(directory)
            anchor=resolver.resolve([.5,-.4]);calls=len(planner.calls)
            result=resolver.connect(anchor,anchor['anchor_values'])
            self.assertTrue(result['passed']);self.assertEqual(len(planner.calls),calls)
            self.assertEqual(result['connection_policy'],CONNECTION_POLICY_VERSION)
            route=result['proof']['connection_route']
            self.assertTrue(route['exact_anchor_endpoint']);self.assertIsNone(route['suffix_proof'])
            self.assertEqual(route['prefix']['prefix_sha256'],digest(anchor['proof']))
            self.assertEqual(result['proof']['waypoints'],anchor['proof']['waypoints'])
            self.assertEqual(result['proof']['edges'],anchor['proof']['edges'])
            verify_connection(result,anchor,resolver.home,anchor['anchor_values'])

    def test_changed_exact_action_checks_only_suffix_and_composes_both_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver,solver,planner=self.build(directory)
            anchor=resolver.resolve([.5,-.4]);calls=len(planner.calls)
            requested=[.61,-.2,.3,30.,.2];result=resolver.connect(anchor,requested)
            self.assertTrue(result['passed']);self.assertEqual(len(planner.calls),calls+1)
            aq=np.r_[anchor['resolved']['q10'],anchor['resolved']['q14']].tolist()
            self.assertEqual(planner.calls[-1][2],aq)
            joined=result['proof'];suffix=joined['connection_route']['suffix_proof']
            self.assertEqual(joined['waypoints'],anchor['proof']['waypoints']+suffix['waypoints'][1:])
            self.assertEqual(joined['edges'],anchor['proof']['edges']+suffix['edges'])
            self.assertEqual(joined['reverse_edges'],suffix['reverse_edges']+anchor['proof']['reverse_edges'])
            self.assertEqual(result['values'],requested);verify_connection(result,anchor,resolver.home,requested)
            for change in ('hash','edge_count','prefix_waypoint','suffix_target','reverse'):
                bad=copy.deepcopy(result);route=bad['proof']['connection_route']
                if change=='hash':route['prefix']['prefix_sha256']='f'*64
                if change=='edge_count':route['prefix']['prefix_edge_count']+=1
                if change=='prefix_waypoint':bad['proof']['waypoints'][0][0]=.01
                if change=='suffix_target':route['suffix_proof']['waypoints'][-1][0]+=.01
                if change=='reverse':bad['proof']['reverse_edges']=bad['proof']['reverse_edges'][::-1]
                with self.subTest(change=change),self.assertRaises(ValueError):verify_connection(bad,anchor,resolver.home,requested)

    def test_prefix_requires_exact_saved_resolution_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver,solver,planner=self.build(directory)
            anchor=resolver.resolve([.5,-.4]);bad=copy.deepcopy(anchor)
            bad['proof']['minimum_clearance_m']=.007
            bad['attempts'][-1]['paths'][-1]=bad['proof']
            verify_resolution(bad,resolver.sources,resolver.home)
            with self.assertRaisesRegex(ValueError,'bytes/prefix'):resolver.connect(bad,bad['anchor_values'])
            path=resolver.directory/anchor['file'];path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'bytes/prefix'):resolver.connect(anchor,anchor['anchor_values'])

    def test_blocked_suffix_falls_back_to_full_home_for_same_requested_endpoint(self):
        class SuffixBlocked(FakePlanner):
            def plan(self,target,bucket,*,start=None):
                result=super().plan(target,bucket,start=start)
                return dict(passed=False,reason='strict_suffix_clearance_rejected') if start is not None else result
        with tempfile.TemporaryDirectory() as directory:
            h=home({});planner=SuffixBlocked(h);solver=FakeSolver()
            resolver=GeometryResolver({'scene':{}},directory,Path(directory)/'out',[source('a')],home=h,solver=solver,planner=planner)
            anchor=resolver.resolve([.5,-.4]);calls=len(planner.calls)
            requested=[.61,-.2,.3,30.,.2];result=resolver.connect(anchor,requested)
            self.assertTrue(result['passed']);self.assertEqual(len(planner.calls),calls+2)
            self.assertIsNotNone(planner.calls[-2][2]);self.assertIsNone(planner.calls[-1][2])
            self.assertEqual(planner.calls[-2][0],planner.calls[-1][0]);self.assertEqual(result['values'],requested)
            self.assertFalse(result['path_attempts'][0]['passed'])
            self.assertEqual(result['proof']['connection_route']['kind'],'full_home_after_anchor_suffix')
            verify_connection(result,anchor,h,requested)
            bad=copy.deepcopy(result);bad['proof']['connection_route']['suffix_attempt_index']=1
            with self.assertRaisesRegex(ValueError,'preceding'):verify_connection(bad,anchor,h,requested)

    def test_derived_geometry_anchor_is_selected_before_action_and_audited(self):
        class DerivedSolver(FakeSolver):
            def solve(self,bucket,values,hint):
                if values==hint['values']: return dict(passed=False,reason='finite_ik_unresolved',candidates=[])
                return super().solve(bucket,values,hint)
        with tempfile.TemporaryDirectory() as directory:
            h=home({});solver=DerivedSolver();planner=FakePlanner(h)
            resolver=GeometryResolver({'scene':{}},directory,Path(directory)/'out',[source('z'),source('a')],home=h,solver=solver,planner=planner)
            r=resolver.resolve([.5,-.4]);self.assertTrue(r['passed']);self.assertEqual(len(r['attempts']),3)
            self.assertEqual(r['source_hint']['values'],[.6,-.2,.3,30.,.2])
            self.assertEqual(r['anchor_values'],[.6,-.2,.325,26.,.2])
            self.assertEqual(r['effective_source_hint']['values'],r['anchor_values'])
            self.assertEqual(r['effective_source_hint']['original_source_id'],'a')
            verify_resolution(r,resolver.sources,h)
            forged=copy.deepcopy(r);forged['anchor_derivation']['offset'][2]=.03
            with self.assertRaisesRegex(ValueError,'derivation|order'):verify_resolution(forged,resolver.sources,h)
        plan=anchor_candidates([source(str(i),(i/100.,0)) for i in range(17)],[.5,-.4])
        self.assertEqual(len(plan),64);self.assertTrue(all(x['derivation']['offset_id']=='nominal' for x in plan[:17]))
        self.assertEqual(plan[17]['derivation']['offset_id'],'height25_tilt-minus4')
        self.assertFalse(anchor_search_spec()['flight_labels_used'])

    def test_actual_fixed_base_collision_remains_hard_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver,solver,planner=self.build(directory,blocked=True)
            row=resolver.resolve([.5,-.4]);self.assertTrue(row['hard_reject']);self.assertFalse(row['home_passed'])
            result=resolver.connect(row,[.61,-.2,.3,30.,.2]);self.assertFalse(result['passed'])
            self.assertEqual(solver.calls,[]);self.assertEqual(planner.calls,[])

    def test_duplicate_sources_and_changed_home_are_rejected(self):
        h=home({});h['q10'][0]=.1
        with self.assertRaisesRegex(ValueError,'identity'):validate_home(h,{})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'Unique'):
                GeometryResolver({'scene':{}},directory,directory,[source('a'),source('a')],home=home({}),solver=FakeSolver(),planner=FakePlanner(home({})))

    def test_path_requires_reverse_clearance_sampling_and_serialization(self):
        a=np.zeros(12);b=a.copy();b[0]=.1;p=proof(a,b);self.assertTrue(verify_path(p,a,b))
        for change in ['reverse','clearance','resolution','endpoint','both_arms']:
            bad=copy.deepcopy(p)
            if change=='reverse':bad['reverse_edges']=[]
            if change=='clearance':bad['edges'][0]['minimum_clearance_m']=.00499
            if change=='resolution':bad['edges'][0]['total_samples']=1
            if change=='endpoint':bad['waypoints'][-1][0]=.2
            if change=='both_arms':bad['waypoints'][1][0]=.02
            with self.subTest(change=change),self.assertRaises(ValueError):verify_path(bad,a,b)

    def test_mismatched_action_or_source_order_fails_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver,solver,planner=self.build(directory)
            row=resolver.resolve([.5,-.4]);bad=copy.deepcopy(row);bad['attempts'][0]['source_id']='z'
            with self.assertRaisesRegex(ValueError,'order'):verify_resolution(bad,resolver.sources,resolver.home)
            requested=[.61,-.2,.3,30.,.2];result=resolver.connect(row,requested)
            with self.assertRaisesRegex(ValueError,'binding'):verify_connection(result,row,resolver.home,[.62,-.2,.3,30.,.2])


class SimpleGate:
    def __init__(self, blocked=False):
        self.bucket_xy=np.zeros(2);self.calls=[];self.blocked=blocked
        self.names=[]
        limits=np.array([[-2.,2.]]*6)
        self.native=type("Limits",(),{"limits10":limits,"limits14":limits})()
    def check(self,q,**kwargs):
        q=np.asarray(q);self.calls.append(q.tolist())
        collision=(q[6]>.2 and q[0]<.2) if not self.blocked else (np.any((q>.01)&(q<.59)))
        return dict(passed=not collision,reason="clearance" if collision else None,
                    minimum_clearance_m=.0049 if collision else .006)


class PlannerTests(unittest.TestCase):
    def test_exact_anchor_reuse_does_not_rerun_actual_sampled_collision_gate(self):
        gate=SimpleGate();h=home({});pathsolver=type('Solver',(),{'gate':gate,'ik':object()})()
        planner=SerializedPlanner({},pathsolver,h,rrt_nodes=0)
        with tempfile.TemporaryDirectory() as directory:
            resolver=GeometryResolver({'scene':{}},directory,Path(directory)/'out',[source('a')],
                home=h,solver=FakeSolver(),planner=planner)
            anchor=resolver.resolve([.5,-.4]);self.assertTrue(anchor['passed']);self.assertGreater(len(gate.calls),100)
            gate.calls.clear();result=resolver.connect(anchor,anchor['anchor_values'])
            self.assertTrue(result['passed']);self.assertEqual(gate.calls,[])
            verify_connection(result,anchor,h,anchor['anchor_values'])

    def test_actual_suffix_samples_stay_at_anchor_instead_of_rechecking_home_prefix(self):
        gate=SimpleGate();h=home({});pathsolver=type('Solver',(),{'gate':gate,'ik':object()})()
        planner=SerializedPlanner({},pathsolver,h,rrt_nodes=0)
        with tempfile.TemporaryDirectory() as directory:
            resolver=GeometryResolver({'scene':{}},directory,Path(directory)/'out',[source('a')],
                home=h,solver=FakeSolver(),planner=planner)
            anchor=resolver.resolve([.5,-.4]);gate.calls.clear()
            requested=[.61,-.2,.3,30.,.2];result=resolver.connect(anchor,requested)
            self.assertTrue(result['passed']);self.assertGreater(len(gate.calls),0);self.assertLess(len(gate.calls),20)
            self.assertTrue(all(.6-1e-12<=q[0]<=.61+1e-12 and q[6]==-.2 for q in gate.calls))
            verify_connection(result,anchor,h,requested)

    def test_serialized_order_fallback_preserves_strict_edges_and_reverse(self):
        gate=SimpleGate();solver=type("Solver",(),{"gate":gate,"ik":object()})()
        planner=SerializedPlanner({},solver,home({}),rrt_nodes=0)
        target=np.zeros(12);target[0]=target[6]=.6
        result=planner.plan(target,[.5,-.4])
        self.assertTrue(result['passed']);self.assertFalse(result['attempts'][0]['proof']['passed'])
        self.assertEqual(result['attempts'][1]['order'],[10,14])
        self.assertTrue(verify_path(result,np.zeros(12),target))
        self.assertGreater(len(gate.calls),100)
        for edge in result['edges']+result['reverse_edges']:
            self.assertGreaterEqual(edge['minimum_clearance_m'],CLEARANCE_M)
            self.assertLessEqual(edge['maximum_actual_joint_sample_step_rad'],SAMPLE_STEP_RAD+1e-12)

    def test_blocked_edges_are_not_relaxed_and_rrt_stays_bounded(self):
        gate=SimpleGate(blocked=True);solver=type("Solver",(),{"gate":gate,"ik":object()})()
        planner=SerializedPlanner({},solver,home({}),rrt_nodes=4)
        target=np.zeros(12);target[0]=target[6]=.6
        with patch.object(planner,'_cartesian',return_value=(None,[])):
            result=planner.plan(target,[.5,-.4])
        self.assertFalse(result['passed'])
        for attempt in result['attempts']:
            if attempt['kind']=='bounded_serialized_rrt':
                for arm in attempt['arms']:
                    self.assertLessEqual(sum(arm['nodes']),6)
                    self.assertLessEqual(len(arm['attempts']),16)
        self.assertTrue(any(q[0]>.01 or q[6]>.01 for q in gate.calls))


class Array:
    def __init__(self,value):self.value=np.asarray(value)
    def __getitem__(self,key):return Array(self.value[key])
    def cpu(self):return self
    def numpy(self):return self.value


class ActualIKControls(unittest.TestCase):
    def test_existing_distance_task_constraint_is_preserved_without_projection(self):
        class NeverIK:
            def candidates(self,*args,**kwargs): raise AssertionError("IK must not run outside task bound")
        class Gate:
            strict=object()
        with tempfile.TemporaryDirectory() as d:
            Path(d,'poses.json').write_text(json.dumps(dict(dropper_joint_rad=[0.]*6,board_joint_rad=[0.]*6)))
            solver=StrictSolver({'arms':{'base_spacing_m':1.33731,'board_base_yaw_rad':np.pi},'board':{'tool_offset_m':.1},'ball':{'radius_m':.0375}},d,ik=NeverIK(),gate=Gate())
            values=[.6,-.2,.3,30.,.2]
            result=solver.solve([.6,-.1],values,source('a'))
            self.assertFalse(result['passed']);self.assertEqual(result['reason'],'board_bucket_distance_constraint_rejected')
            self.assertEqual(values,[.6,-.2,.3,30.,.2])
    def test_cheap_workspace_hint_cannot_suppress_actual_ik(self):
        class IK:
            def __init__(self):self.calls=0
            def candidates(self,*args,**kwargs):
                self.calls+=1
                pool={k:Array(v) for k,v in dict(q=np.zeros((1,1,6)),valid=np.ones((1,1),bool),residual=np.zeros((1,1))).items()}
                return dict(dropper=pool,board=pool,targets={'dropper_flange_world':np.eye(4)[None], 'board_flange_local':np.eye(4)[None]})
        class Gate:
            strict=object()
            def check(self,*args):return dict(reason=0,independent_reject_arms=[])
        with tempfile.TemporaryDirectory() as d:
            Path(d,'poses.json').write_text(json.dumps(dict(dropper_joint_rad=[0.]*6,board_joint_rad=[0.]*6)))
            ik=IK();solver=StrictSolver({'arms':{'base_spacing_m':1.33731,'board_base_yaw_rad':np.pi}},d,ik=ik,gate=Gate())
            with patch('sircus.mjlab.calibrated_admission.outer_geometry_mask',return_value=(np.ones((1,1),bool),np.zeros((1,1),bool))),patch('sircus.mjlab.calibrated_admission.native_pose_error',return_value=dict(position_m=0.,rotation_rad=0.)):
                result=solver.solve([.5,-.4],[.6,-.2,.3,30.,.2],source('a'))
            self.assertTrue(result['passed']);self.assertEqual(ik.calls,1)
            self.assertFalse(result['analytic_hint']['old_workspace']);self.assertFalse(result['analytic_hint']['hard_veto'])


if __name__=='__main__':unittest.main()
