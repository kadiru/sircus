"""CPU tests for isolated yaw intervention, exact shot preservation and clear labels."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch,Mock
import numpy as np
from sircus.mjlab import gripper_yaw_geometry as G,gripper_yaw_native as N,gripper_yaw_contract as C
from sircus.mjlab import gripper_yaw_pipeline as P,gripper_yaw_study as S
from sircus.mjlab.benchmark_verify import read,digest

SCENE={'board':{'mount_yaw_rad':math.pi/2,'tool_offset_m':.12},'arms':{'base_spacing_m':1.2},'ball':{'radius_m':.0375}}

def fixture():
    sources=[];controls=[]
    for i,context in enumerate((*N.ROLES,*N.CASES)):
        for method in (('control',) if i<3 else ('current_bc','corrected_bc')):
            row={k:None for k in N.META};changed=method=='corrected_bc'
            row.update(id=f'original-{context}-{method}',context_id=context,role=context if i<3 else 'test',phase='controls_pre' if i<3 else 'test',
                method=method,repeat=0,values=[.6,-.1,.45,30.,.08] if changed else [.59,-.11,.452,30.4,.1],bucket_xy_m=[1.1,-.54],
                native_release=True,route_success=i<2 or method=='current_bc',invalid=None,invalid_physics=False,
                forbidden_contact_steps=3 if changed or i==2 else 0,q10=[.1]*6,q14=[.2]*6,
                connection_file=f'connections/{i}-{method}.json',connection_sha256=f'{i}-{method}',
                context_resolution_file='resolutions/context.json',context_resolution_sha256='resolution',
                board_center_m=[.6,-.1,.45],board_normal=[0,0,1],release_ball_center_m=[.6,-.1,.5675])
            (controls if i<3 else sources).append(row)
    return sources,controls,read(Path(N.__file__).with_name('gripper-yaw-correction.json'))

class PlanTests(unittest.TestCase):
    def test36_slots_no_other_parameter_changes(self):
        sources,controls,rule=fixture();before=copy.deepcopy((sources,controls));plan=N.make_plan(sources,controls,rule)
        self.assertEqual((sources,controls),before);self.assertEqual(len(plan),36)
        self.assertEqual([r['phase'] for r in plan],['controls_pre']*9+['diagnostic']*18+['controls_post']*9)
        lookup={r['id']:r for r in sources+controls}
        for r in plan:
            s=lookup[r['source_record_id']];self.assertEqual(r['values'],s['values']);self.assertEqual(r['command_values'],s['values']);self.assertEqual(r['preserved_q14'],s['q14'])
            self.assertEqual(r['bucket_xy_m'],s['bucket_xy_m']);self.assertEqual(r['parameter_indices'],[])
            self.assertEqual(r['source_record_sha256'],digest(s));self.assertEqual(r['substeps'],64)
            if r['variant']=='bucket_facing':
                self.assertEqual(r['adapter_directory'],'native-yaw');self.assertEqual(s['method'],'corrected_bc')
                self.assertAlmostEqual(r['requested_gripper_yaw_rad'],math.atan2(-.44,.5));self.assertEqual(r['admission_mode'],'fixed_board_yaw_strict_proof')
            else:self.assertEqual(r['requested_gripper_yaw_rad'],0.)
        self.assertEqual(sum(r['variant']=='bucket_facing' for r in plan),6)

    def test_literal_rule_and_parent_mutations_rejected(self):
        sources,controls,rule=fixture()
        for change in ({'mode':'search'},{'contexts':list(reversed(N.CASES))},{'preserve_board_joints':False},{'diagnosis_sha256':'bad'},{'rotation':'yaw+pi'},{'extra':1}):
            with self.subTest(change=change),self.assertRaises(ValueError):N.make_plan(sources,controls,dict(rule,**change))
        for key,value in (('route_success',True),('invalid_physics',True),('native_release',False),('forbidden_contact_steps',0),('bucket_xy_m',[.3,.4])):
            bad=copy.deepcopy(sources);bad[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):N.make_plan(bad,controls,rule)

    def test_protocol_limits_and_namespace(self):
        p=C.protocol();self.assertEqual(p['maximum_requests'],36);self.assertEqual(p['diagnostic_requests'],18)
        self.assertEqual(p['control_requests'],18);self.assertEqual(p['maximum_acquired_s'],3600)
        self.assertEqual(p['maximum_queue_s'],900);self.assertEqual(p['maximum_pipeline_s'],6300)
        self.assertEqual(C.INPUT_KEYS,P.INPUT_KEYS);self.assertEqual(S.VERSION,N.VERSION)

class RefusalTests(unittest.TestCase):
    def test_refusal_propagates_fixed_board_without_allocating_native_environment(self):
        from sircus.mjlab import calibrated_adapter as A
        sources,controls,rule=fixture();request=next(r for r in N.make_plan(sources,controls,rule) if r['variant']=='bucket_facing')
        source=next(s for s in sources if s['id']==request['source_record_id'])
        resolution=dict(file='context.json',sha256='context-hash',source_hint=request['source_hint'],effective_source_hint=request['effective_source_hint'],
            layout_eligible=True,anchor_admitted=True,hard_reject=False)
        request.update(context_resolution_file=resolution['file'],context_resolution_sha256=resolution['sha256'],context_home_sha256='home-hash',
            source_teacher_id='teacher',source_hint={'id':'teacher'},effective_source_hint={'id':'teacher'},context_layout_eligible=True,context_anchor_admitted=True,context_hard_reject=False)
        resolution.update(source_hint=request['source_hint'],effective_source_hint=request['effective_source_hint'])
        connection=dict(reason='finite_ik_unresolved',passed=False,endpoint_eligible=False,q10=[0.]*6,q14=source['q14'],file='yaw-connections/refusal.json',sha256='new-yaw-proof')
        adapter=object.__new__(A.CalibratedAdapter);adapter._ids=set();adapter.storage_check=lambda:None
        adapter.resolver=SimpleNamespace(resolve=lambda b:resolution,connect=lambda a,v:connection);adapter.sources=[];adapter.home={'sha256':'home-hash'};adapter.batch=32
        adapter._native=Mock(side_effect=AssertionError('Refusal must never allocate or execute native dynamics'))
        with tempfile.TemporaryDirectory() as tmp,patch.object(A,'verify_resolution'),patch.object(A,'verify_connection'),patch.object(A,'source_geometry',side_effect=lambda x:x):
            adapter.directory=Path(tmp);rows,timing=adapter.evaluate([request],'diagnostic')
        self.assertEqual(rows[0]['q14'],source['q14']);self.assertFalse(rows[0]['native_release']);self.assertEqual(timing['native_releases'],0)
        self.assertNotIn('native_initial_q14',rows[0]);adapter._native.assert_not_called()

    def test_request_orientation_mutation_rejected_before_native_work(self):
        sources,controls,rule=fixture();request=next(r for r in N.make_plan(sources,controls,rule) if r['variant']=='bucket_facing')
        worker=object.__new__(N.MatchedEngine);worker.sources={s['id']:s for s in sources};worker.yaw=True;worker.engine=SimpleNamespace(evaluate=Mock())
        for key,value in (('requested_gripper_yaw_rad',0.),('substeps',128),('parameter_indices',[4]),('selection_or_learning',True),('donor_record_id','alternate')):
            with self.subTest(key=key),self.assertRaises(ValueError):worker.evaluate([dict(request,**{key:value})],'diagnostic')
        worker.engine.evaluate.assert_not_called()

class GeometryTests(unittest.TestCase):
    def test_passage_faces_bucket_and_tool_stays_down(self):
        for bucket in ([1.1,-.54],[.2,-.7],[.6,.8]):
            values=[.6,-.1,.45,30.,.08];yaw,targets=G.yaw_target(SCENE,bucket,values)
            rot=targets['dropper_flange_world'][:3,:3];direction=np.r_[np.asarray(bucket)-values[:2],0.];direction/=np.linalg.norm(direction)
            np.testing.assert_allclose(rot[:,0],direction,atol=1e-15);np.testing.assert_allclose(rot[:,2],[0,0,-1],atol=1e-15)
            np.testing.assert_allclose(rot.T@rot,np.eye(3),atol=1e-15);self.assertAlmostEqual(np.linalg.det(rot),1.)
            np.testing.assert_allclose(targets['release_center_m'],[.6,-.1,.5675]);np.testing.assert_allclose(targets['dropper_flange_world'][:3,3],[.6,-.1,.6625])

    def test_orientation_changes_only_dropper_rotation(self):
        from sircus.mjlab.learning_ik import shot_targets
        values=np.array([.6,-.1,.45,30.,.08]);bucket=np.array([1.1,-.54]);_,new=G.yaw_target(SCENE,bucket,values)
        old={k:v[0] for k,v in shot_targets(SCENE,bucket[None],values[None,:3],[values[3]],[values[4]]).items()}
        for key in old:
            if key!='dropper_flange_world':np.testing.assert_array_equal(old[key],new[key])
        np.testing.assert_array_equal(old['dropper_flange_world'][:3,3],new['dropper_flange_world'][:3,3])
        self.assertFalse(np.array_equal(old['dropper_flange_world'][:3,:3],new['dropper_flange_world'][:3,:3]))

    def test_full_rotation_binding_changes_digest(self):
        sources,_,_=fixture();s=sources[1];ext=G.intervention(SCENE,s)
        self.assertEqual(ext['fixed_q14'],s['q14']);self.assertEqual(ext['source_record_sha256'],digest(s))
        altered=copy.deepcopy(ext);altered['requested_rotation_matrix'][0][0]+=.01;self.assertNotEqual(digest(ext),digest(altered))
        altered=copy.deepcopy(ext);altered['fixed_q14'][0]+=.01;self.assertNotEqual(digest(ext),digest(altered))

    def test_invalid_target_refused(self):
        for b,v in (([.6,-.1],[.6,-.1,.45,30,.08]),([1.,float('nan')],[.6,-.1,.45,30,.08]),([1.,0.],[.6,.1,.4])):
            with self.assertRaises(ValueError):G.yaw_target(SCENE,b,v)

    def test_measured_yaw_requires_full_rotation_and_preserved_board(self):
        sources,_,_=fixture();source=sources[1];_,targets=G.yaw_target(SCENE,source['bucket_xy_m'],source['values'])
        row=dict(source,variant='bucket_facing',native_initial_q10=np.asarray(source['q10'],np.float32).astype(float).tolist(),native_initial_q14=np.asarray(source['q14'],np.float32).astype(float).tolist())
        with patch.object(G,'body_transform',return_value=targets['dropper_flange_world']):
            measured=G.yaw_evidence(SCENE,None,row,source);self.assertLess(measured['rotation_error_rad'],1e-10)
            self.assertTrue(measured['board_q14_unchanged']);self.assertTrue(measured['release_unchanged'])
            changed=copy.deepcopy(row);changed['q14'][0]+=.1
            with self.assertRaises(RuntimeError):G.yaw_evidence(SCENE,None,changed,source)
        wrong=targets['dropper_flange_world'].copy();wrong[:3,:3]=G.DOWNWARD
        with patch.object(G,'body_transform',return_value=wrong),self.assertRaises(RuntimeError):G.yaw_evidence(SCENE,None,row,source)

    def test_refusal_has_no_invented_measured_rotation(self):
        sources,_,_=fixture();source=sources[1];row=dict(source,variant='bucket_facing',native_release=False)
        measured=G.yaw_evidence(SCENE,None,row,source)
        self.assertNotIn('measured_initial_rotation_matrix',measured);self.assertIsNone(measured['release_unchanged'])

if __name__=='__main__':unittest.main()
