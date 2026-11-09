"""CPU-only independent yaw proof, frozen-plan and evidence-tamper tests."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sircus.mjlab import gripper_yaw_verify as V


def fixtures():
    sources=[];controls=[]
    for i,context in enumerate((*V.ROLES,*V.CASES)):
        for method in (('control',) if i<3 else ('current_bc','corrected_bc')):
            changed=method=='corrected_bc';values=[.6,-.1,.45,30.,.08] if changed else [.59,-.11,.452,30.4,.1]
            row={k:None for k in V.META}
            row.update(id=f'original-{context}-{method}',context_id=context,role=context if i<3 else 'test',phase='controls_pre' if i<3 else 'test',
                method=method,repeat=0,values=values,bucket_xy_m=[1.1,-.54],native_release=True,route_success=i<2 or method=='current_bc',
                invalid=None,invalid_physics=False,forbidden_contact_steps=3 if changed or i==2 else 0,q10=[.1]*6,q14=[.2]*6,
                connection_file=f'connections/{i}-{method}.json',connection_sha256=f'{i}-{method}',
                context_resolution_file='resolutions/context.json',context_resolution_sha256='resolution',
                board_center_m=[.6,-.1,.45],board_normal=[0.,0.,1.],release_ball_center_m=[values[0],values[1],values[2]+.0375+values[4]],pose_gate={})
            (controls if i<3 else sources).append(row)
    rule=dict(mode=V.MODE,contexts=list(V.CASES),diagnosis_sha256=V.DIAGNOSIS_SHA256,description='Change only dropper yaw toward bucket.',
        rotation='Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)',preserve_board_joints=True)
    return sources,controls,rule


def planned():
    from sircus.mjlab import gripper_yaw_native as N
    sources,controls,rule=fixtures()
    return sources,controls,rule,N.make_plan(sources,controls,rule)


def observed():
    sources,controls,rule,plan=planned();lookup={r['id']:r for r in sources+controls};rows=[]
    for request in plan:
        row=copy.deepcopy(request);source=lookup[row['source_record_id']]
        for key in ('connection_file','connection_sha256','native_release','q10','q14','invalid','board_center_m','board_normal','pose_gate','release_ball_center_m'):
            row[key]=copy.deepcopy(source[key])
        if row['variant']=='bucket_facing':row.update(connection_sha256='new-'+row['context_id'],q10=[.13]*6)
        for arm in ('q10','q14'):row['native_initial_'+arm]=np.asarray(row[arm],np.float32).astype(float).tolist()
        row['realization_comparison']=V.realized_comparison(row,source,V.replay.AuditChecks())
        row.update(scoring_version=V.oracle.VERSION,route_version=V.oracle.ROUTE_VERSION,reward_version=V.oracle.REWARD_VERSION,
            closest_opening_distance_m=.1,board_bounce=True,forbidden_contact_steps=0,max_retention_s=0.,route_success=False,
            route_quality=0,single_board_bounce_success=False)
        rows.append(row)
    return sources,controls,plan,rows


class PlanTests(unittest.TestCase):
    def test_exact_plan_three_variants_and_original_values(self):
        sources,controls,rule,plan=planned();V.audit_plan(plan,sources,controls,rule,V.replay.AuditChecks())
        self.assertEqual(len(plan),36);self.assertEqual([r['phase'] for r in plan],['controls_pre']*9+['diagnostic']*18+['controls_post']*9)
        lookup={s['id']:s for s in sources+controls}
        for r in plan:
            self.assertEqual(r['values'],lookup[r['source_record_id']]['values']);self.assertEqual(r['preserved_q14'],lookup[r['source_record_id']]['q14'])

    def test_request_tampering_fails(self):
        sources,controls,rule,plan=planned()
        for key,value in [('values',[.6,-.1,.46,30.,.08]),('bucket_xy_m',[1.2,-.54]),('requested_gripper_yaw_rad',0.),
                ('preserved_q14',[0.]*6),('substeps',128),('adapter_directory','native'),('admission_mode','exact_parent_branch'),('parameter_indices',[4])]:
            bad=copy.deepcopy(plan);bad[15][key]=value
            with self.subTest(key=key),self.assertRaises(V.EvidenceError):V.audit_plan(bad,sources,controls,rule,V.replay.AuditChecks())
        with self.assertRaises(V.EvidenceError):V.audit_plan(plan[:-1],sources,controls,rule,V.replay.AuditChecks())

    def test_parent_provenance_or_rule_change_fails(self):
        sources,controls,rule,plan=planned()
        for key,value in [('route_success',True),('native_release',False),('forbidden_contact_steps',0),('invalid_physics',True)]:
            bad=copy.deepcopy(sources);bad[1][key]=value
            with self.subTest(key=key),self.assertRaises(V.EvidenceError):V.audit_plan(plan,bad,controls,rule,V.replay.AuditChecks())
        for key,value in [('preserve_board_joints',False),('rotation','yaw+pi'),('description','')]:
            with self.subTest(key=key),self.assertRaises(V.EvidenceError):V.audit_plan(plan,sources,controls,dict(rule,**{key:value}),V.replay.AuditChecks())

    def test_no_import_of_producer_or_flight_scorer(self):
        tree=ast.parse(Path(V.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                names=[node.module or '']+[a.name for a in node.names]
                self.assertFalse(any(n in {'gripper_yaw_geometry','gripper_yaw_native','gripper_yaw_study','gripper_yaw_contract','table_assisted_score'} for n in names))
        self.assertNotIn('make_plan',[n.id for n in ast.walk(tree) if isinstance(n,ast.Name)])


class ObservationTests(unittest.TestCase):
    def test_all_observations_and_branch_change_preserve_board(self):
        sources,controls,plan,rows=observed();V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())
        self.assertFalse(rows[15]['realization_comparison']['q10_equal_to_source']);self.assertTrue(rows[15]['realization_comparison']['q14_equal_to_source'])

    def test_fixed_board_and_independent_proofs_cannot_be_forged(self):
        sources,controls,plan,rows=observed()
        for mode in ('board','reset','proof','repeat','original','comparison','missing'):
            bad=copy.deepcopy(rows)
            if mode=='board':bad[15]['q14'][0]+=.01
            elif mode=='reset':bad[15]['native_initial_q10'][0]+=.1
            elif mode=='proof':bad[15]['connection_sha256']=sources[1]['connection_sha256']
            elif mode=='repeat':bad[15]['connection_sha256']='different'
            elif mode=='original':bad[9]['q10'][0]+=.1
            elif mode=='comparison':bad[15]['realization_comparison']['q10_equal_to_source']=True
            else:bad.pop()
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):V.audit_observations(bad,plan,sources,controls,V.replay.AuditChecks())

    def test_finite_decline_has_no_fake_flight(self):
        sources,controls,plan,rows=observed()
        for row in rows[15:18]:
            row.update(native_release=False,q10=[0.]*6,invalid='finite_ik_unresolved',realization_comparison=None)
            row.pop('native_initial_q10');row.pop('native_initial_q14')
        V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())
        self.assertEqual(V.independent_cases(plan,rows)[0]['variants']['bucket_facing']['setup_rejections'],3)
        rows[15]['native_cache_file']='fake.npz'
        with self.assertRaises(V.EvidenceError):V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())


class GeometryTests(unittest.TestCase):
    def test_full_target_has_bucket_facing_passage_and_down_axis(self):
        for bucket in ([1.1,-.54],[.2,-.7],[.6,.8]):
            t=V.expected_yaw([.6,-.1,.45,30.,.08],bucket,.0375);r=np.asarray(t['rotation_matrix'])
            direction=np.r_[np.asarray(bucket)-[.6,-.1],0.];direction/=np.linalg.norm(direction)
            np.testing.assert_allclose(r[:,0],direction,atol=1e-15);np.testing.assert_array_equal(r[:,2],[0,0,-1])
            np.testing.assert_allclose(np.asarray(t['dropper_flange_world'])[:3,3],[.6,-.1,.6625],atol=1e-15)
            self.assertAlmostEqual(np.linalg.det(r),1.)
        with self.assertRaises(V.EvidenceError):V.expected_yaw([.6,-.1,.45,30.,.08],[.6,-.1],.0375)
        with self.assertRaises(V.EvidenceError):V.rotation_error(np.zeros((3,3)),np.eye(3))

    def proof(self):
        # Cross-check producer shape only in the fixture; checker imports none of it.
        from sircus.mjlab.gripper_yaw_geometry import intervention
        from sircus.mjlab.calibrated_admission import CONNECTION_POLICY_VERSION
        source=fixtures()[0][1];scene=dict(ball=dict(radius_m=.0375),board=dict(mount_yaw_rad=math.pi/2,tool_offset_m=.12),arms=dict(base_spacing_m=1.2))
        ext=intervention(scene,source)
        key=V.digest(dict(version=ext['version'],connection_policy=CONNECTION_POLICY_VERSION,resolution_sha256='original-res',values=source['values'],bucket_xy_m=source['bucket_xy_m'],yaw_intervention=ext))
        c=dict(yaw_intervention=ext,yaw_key=key,connection_policy=CONNECTION_POLICY_VERSION,resolution_sha256='original-res',values=source['values'],bucket_xy_m=source['bucket_xy_m'],q14=source['q14'])
        return c,source

    def test_extension_independently_binds_matrix_source_and_key(self):
        c,s=self.proof();t,key=V.audit_intervention(c,s,.0375,V.replay.AuditChecks(),'proof');self.assertEqual(key,c['yaw_key'])
        for mode in ('angle','matrix','translation','board','values','key','release','source'):
            bad=copy.deepcopy(c);e=bad['yaw_intervention']
            if mode=='angle':e['requested_yaw_rad']+=math.pi
            elif mode=='matrix':e['requested_rotation_matrix']=np.diag([1.,-1.,-1.]).tolist()
            elif mode=='translation':e['requested_dropper_flange_world'][2][3]+=.01
            elif mode=='board':e['fixed_q14'][0]+=.01
            elif mode=='values':e['source_values'][4]+=.01
            elif mode=='key':bad['yaw_key']='old-five-value-key'
            elif mode=='release':e['release_center_m'][2]+=.01
            else:e['source_record_sha256']='forged'
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):V.audit_intervention(bad,s,.0375,V.replay.AuditChecks(),'tamper')

    def test_scalar_fk_and_limits(self):
        origins=np.tile(np.eye(4),(6,1,1));axes=np.tile([0.,0.,1.],(6,1));chain=(origins,axes,np.full(6,-4.),np.full(6,4.))
        pose=V.forward_chain(chain,[math.pi/2,0,0,0,0,0]);np.testing.assert_allclose(pose[:3,:3],[[0,-1,0],[1,0,0],[0,0,1]],atol=1e-15)
        self.assertAlmostEqual(V.rotation_error(pose[:3,:3],np.eye(3)),math.pi/2)
        with self.assertRaises(V.EvidenceError):V.forward_chain(chain,[5,0,0,0,0,0])


class SchemaAndTimingTests(unittest.TestCase):
    def test_all_case_outcomes_match_producer_schema(self):
        from sircus.mjlab import gripper_yaw_study as S
        sources,controls,plan,rows=observed();self.assertEqual(S.case_results(plan,rows),V.independent_cases(plan,rows));self.assertEqual(S.summarize(rows),V.statistics(rows))

    def test_six_frozen_namespace_batches_and_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'study';root.mkdir();sources,controls,plan,rows=observed()
            def write(path,value):path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value))
            write(root/'plan.json',plan);times=[]
            for index,(start,end) in enumerate(((0,9),(9,15),(15,18),(18,24),(24,27),(27,36))):
                name=f'requests/batch-{index:05d}.json';phase=plan[start]['phase'];write(root/name,dict(phase=phase,rows=plan[start:end],pre_outcome=True))
                for row in rows[start:end]:row.update(request_packet=name,request_packet_sha256=V.sha(root/name))
                times.append(dict(phase=phase,request_packet=name,request_packet_sha256=V.sha(root/name),acquired_start_s=index*2+.5,acquired_end_s=index*2+1.5))
            raw=[(json.dumps(r)+'\n').encode() for r in rows];(root/'rows.jsonl').write_bytes(b''.join(raw));(root/'timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in times))
            boundaries=[]
            for phase,count,t in [('plan-frozen',0,0.),('controls_pre-complete',9,2.),('diagnostic-complete',27,10.),('controls_post-complete',36,12.)]:
                boundaries.append(dict(phase=phase,rows_count=count,acquired_elapsed_s=t,plan_sha256=V.sha(root/'plan.json'),rows_sha256=hashlib.sha256(b''.join(raw[:count])).hexdigest() if count else None))
            (root/'phase-boundaries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in boundaries))
            budget=dict(acquired_unix_s=100.,limit_s=3600.,effective_worker_limit_s=3600.,global_deadline_unix_s=6390.,audit_reserve_s=1800,gpu_uuid='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4')
            write(root.with_name('study-budget.json'),budget);write(root.with_suffix('.pipeline.json'),dict(started_unix_s=90.,global_deadline_unix_s=6390.,global_limit_s=6300.))
            report=dict(elapsed_s=13.,deadlines=dict(worker_limit_s=3600.,effective_worker_limit_s=3600.,global_deadline_unix_s=6390.))
            packets,_=V.audit_timing(root,rows,report,V.replay.AuditChecks());self.assertEqual(packets['batches'],6)
            times[1]['acquired_end_s']=3600.;(root/'timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in times))
            with self.assertRaises(V.EvidenceError):V.audit_timing(root,rows,report,V.replay.AuditChecks())

if __name__=='__main__':unittest.main()
