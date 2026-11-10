"""Tamper tests for fixed-grid evaluation, yaw behavior and teacher provenance."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
from sircus.mjlab import fixed35_imitation_verify as A
from sircus.mjlab import fixed35_imitation_contract as C
from sircus.mjlab.benchmark_verify import Checks, EvidenceError
from sircus.mjlab.tests.test_corrected_imitation_contract import fixture as one_correction
from sircus.mjlab.tests.test_coverage_training_verify import outcome


def write(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def native_sequence(root):
    rows=[]
    for phase in ('controls_pre','test','controls_post'):
        groups=[(role,'control',3) for role in A.ROLES] if phase!='test' else [('test',m,35) for m in A.METHODS]
        for role,method,n in groups:
            for i in range(n):rows.append(dict(id=str(len(rows)),phase=phase,role=role,method=method))
    raw=[(json.dumps(r)+'\n').encode() for r in rows];(root/'rows.jsonl').write_bytes(b''.join(raw))
    boundaries=[]
    for phase,count,t in [('inputs-frozen',0,0),('controls_pre-complete',9,1),('fit-complete',9,2),('test-plan-frozen',9,3),('test-complete',79,10),('controls_post-complete',88,12)]:
        boundaries.append(dict(phase=phase,rows_count=count,rows_sha256=hashlib.sha256(b''.join(raw[:count])).hexdigest() if count else None,acquired_elapsed_s=t))
    (root/'phase-boundaries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in boundaries))
    timings=[dict(phase=p,acquired_start_s=start,acquired_end_s=end) for p,start,end in [('controls_pre',0,1),*[( 'test',4+i,4.5+i) for i in range(4)],('controls_post',11,12)]]
    (root/'timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in timings))
    return rows,dict(elapsed_s=13),boundaries


def inference_fixture(root):
    test=A.fixed_grid();states={};rows=[];positions=[];packets=[];report=dict(artifacts={},test_comparison=[])
    source,_=one_correction();anchor=source['effective_source_hint']
    for mi,method in enumerate(A.METHODS):
        policy={}
        for layer,shape in ((0,(64,9)),(2,(64,64)),(4,(5,64))):
            policy[f'actor.{layer}.weight']=np.zeros(shape,np.float32);policy[f'actor.{layer}.bias']=np.zeros(shape[0],np.float32)
        states[method]=dict(policy=policy);artifact=dict(path=f'checkpoints/{method}.pt',sha256=str(mi)*64,weight_sha256='f'*64,samples=99 if mi==0 else 92,seed=1)
        report['artifacts'][method]=artifact;predictions=[];group=[]
        for i,context in enumerate(test):
            meta={k:copy.deepcopy(source[k]) for k in A.META};meta['observation']=A.calibrated.expected_observation(context['bucket_xy_m'],anchor).tolist()
            prediction=dict(meta,context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],method=method,command_values=anchor['values'],normalized_residual=[0.]*5,
                latent_mean=[0.]*5,source_training_ids=[],artifact_sha256=artifact['sha256'],ball_flight_search_at_inference=False,geometry_search_at_inference=True,selection_or_learning=False)
            predictions.append(prediction)
            row=outcome(f'fixed35-imitation-test-{method}-{context["id"]}',capture=(i+mi)%3!=0)
            row.update(prediction,phase='test',role='test',values=anchor['values'],frozen_artifact_sha256=artifact['sha256'],perturbation_mixed_units=[0.]*5)
            group.append(row);rows.append(row)
            positions.append(dict(context_id=row['context_id'],bucket_xy_m=row['bucket_xy_m'],method=method,capture=A.native.captured(row),clean=A.native.captured(row) and row['route_quality']==2,
                native_release=True,invalid=None,invalid_physics=False,direct_capture=row['direct_capture'],table_assisted_capture=False,forbidden_contact_steps=0,values=row['values'],record_id=row['id']))
        packet=dict(method=method,role='test',artifact=artifact,rows=predictions,one_prediction_per_layout=True,selection_or_learning=False)
        packets.append(packet);write(root,f'predictions/test-{method}.json',packet)
        report['test_comparison'].append(dict(method=method,counts=A.native.statistics(group),artifact_sha256=artifact['sha256']))
    write(root,'test-plan.json',dict(methods=packets,frozen_before_outcomes=True))
    for row in rows:row['test_plan_sha256']=A.sha(root/'test-plan.json')
    report['test_positions']=positions
    captures=[[A.native.captured(rows[j*35+i]) for j in range(2)] for i in range(35)]
    gains=sum(a and not b for a,b in captures);losses=sum(b and not a for a,b in captures)
    report['paired_comparison']=dict(paired_positions=35,both_capture=sum(a and b for a,b in captures),fixed35_only=gains,current_only=losses,
        both_fail=sum(not a and not b for a,b in captures),net_captures=gains-losses)
    return rows,report,dict(splits=dict(test=test)),states


def teacher_sources(root):
    old=[]
    for i in range(97):
        row=outcome('old-'+str(i));row.update(context_id='old-context-'+str(i),bucket_xy_m=[.2+i*.001,-.3],
            observation=[0.]*9,normalized_residual=[0.]*5)
        old.append(row)
    write(root,'bootstrap/teachers.json',old)
    corrected=dict(rows=[],bound=lambda name:root/name);source=dict(rows=[])
    for name,context,phase,variant,index,xy in (
        ('far',A.FAR,'selected_repeats','selected',22,[1.1,-.5]),
        ('near',A.NEAR,'references','near_success_reference',None,[.5,-.4])):
        parent,_=one_correction();values=[.65,-.2,.45,30.33068546648235,.09564961424222355]
        if name=='near':values=[.65,-.3,.29,27.3,.0758]
        hint=copy.deepcopy(parent['effective_source_hint']);hint['values']=values[:3]+[values[3],values[4]-.01]
        obs=A.calibrated.expected_observation(xy,hint).tolist()
        for repeat in range(3):
            row=outcome(f'{name}-{repeat}');row.update({k:copy.deepcopy(parent[k]) for k in A.META})
            row.update(id=f'{name}-{repeat}',context_id=context,phase=phase,variant=variant,grid_index=index,repeat=repeat,
                values=values,bucket_xy_m=xy,q10=[0.]*6,q14=[0.]*6,connection_sha256='a',adapter_directory='native-yaw',
                diagnostic_robot_contact_steps=0,invalid_physics=False,effective_source_hint=hint,observation=obs,
                requested_gripper_yaw_rad=math.atan2(xy[1]-values[1],xy[0]-values[0]))
            source['rows'].append(row)
    return corrected,source,old


class FixedGridTests(unittest.TestCase):
    def test_literal35_points_and_no_random_generator(self):
        self.assertEqual(A.fixed_grid(),C.fixed_grid())
        points=[r['bucket_xy_m'] for r in A.fixed_grid()]
        self.assertEqual((len(points),len(set(map(tuple,points)))),(35,35))
        np.testing.assert_allclose(points[0],[.1,-.65],atol=1e-16)
        np.testing.assert_allclose(points[-1],[1.24,.25],atol=1e-16)

    def test_grid_cannot_jitter_rename_drop_or_repeat_a_position(self):
        for mode in ('jitter','drop','duplicate','rename'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);grid=A.fixed_grid();good=dict(train=[],test=grid)
                write(root,'splits.json',good);A.audit_grid(root,dict(splits=good),[],Checks())
                if mode=='jitter':grid[0]['bucket_xy_m'][0]+=.0001
                elif mode=='drop':grid.pop()
                elif mode=='duplicate':grid[1]=copy.deepcopy(grid[0])
                else:grid[0]['id']='new'
                bad=dict(train=[],test=grid);write(root,'splits.json',bad)
                with self.assertRaises(EvidenceError):A.audit_grid(root,dict(splits=bad),[],Checks())

    def test_protocol_declines_different_budget_steps_perturbations_or_orientation(self):
        A.protocol_checks(C.protocol(),Checks())
        for key,value in [('substeps',128),('test_layouts',70),('maximum_requests',89),('actor_steps',801),
                          ('teacher_count',107),('bucket_perturbations',True),('noise',True),('rotation','yaw0')]:
            contract=C.protocol();contract[key]=value
            with self.subTest(key=key),self.assertRaises(EvidenceError):A.protocol_checks(contract,Checks())


class TeacherTests(unittest.TestCase):
    def test_reconstructs99_distinct_layouts_without_repeat_weighting(self):
        with tempfile.TemporaryDirectory() as tmp:
            corrected,source,old=teacher_sources(Path(tmp));before=copy.deepcopy(source)
            with patch.object(A,'authenticate_row'):
                original,new,teachers,lineage=A.reconstruct_teachers(corrected,source,Checks())
            self.assertEqual(original,old);self.assertEqual(source,before)
            self.assertEqual((len(teachers),len(new),len(lineage)),(99,2,2))
            for name,teacher in zip(('far','near'),new):
                self.assertEqual(teacher,C.yaw_teacher([r for r in source['rows'] if r['id'].startswith(name+'-')],name))

    def test_failed_repeats_other_setting_and_out_of_support_target_fail(self):
        for mode in ('miss','repeat','variant','contact','values','observation','outside'):
            with tempfile.TemporaryDirectory() as tmp:
                corrected,source,_=teacher_sources(Path(tmp));row=source['rows'][0]
                if mode=='miss':row['route_success']=False
                elif mode=='repeat':source['rows'][1]['repeat']=0
                elif mode=='variant':row['grid_index']=23
                elif mode=='contact':row['diagnostic_robot_contact_steps']=1
                elif mode=='values':row['values']=list(row['values']);row['values'][3]+=1
                elif mode=='observation':row['observation']=[0.]*9
                else:
                    for r in source['rows'][:3]:
                        r['effective_source_hint']['values'][0]=.01
                        r['observation']=A.calibrated.expected_observation(r['bucket_xy_m'],r['effective_source_hint']).tolist()
                with patch.object(A,'authenticate_row'),self.subTest(mode=mode),self.assertRaises(EvidenceError):
                    A.reconstruct_teachers(corrected,source,Checks())

    def test_duplicate_coordinates_under_new_ids_fail_layout_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corrected,source,old=teacher_sources(root);old[1]['bucket_xy_m']=old[0]['bucket_xy_m']
            write(root,'bootstrap/teachers.json',old)
            with patch.object(A,'authenticate_row'),self.assertRaises(EvidenceError):A.reconstruct_teachers(corrected,source,Checks())


class FreezeTests(unittest.TestCase):
    def test_complete88_sequence_and_early_freeze_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,report,boundaries=native_sequence(root)
            self.assertEqual(A.audit_sequence(root,rows,report,Checks())['total_requests'],88)
            with self.assertRaises(EvidenceError):A.audit_sequence(root,rows[:-1],report,Checks())
            boundaries[3]['acquired_elapsed_s']=5
            (root/'phase-boundaries.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in boundaries))
            with self.assertRaises(EvidenceError):A.audit_sequence(root,rows,report,Checks())

    def test_changed_saved_outcome_prefix_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,report,_=native_sequence(root)
            (root/'rows.jsonl').write_text((root/'rows.jsonl').read_text().replace('"id": "0"','"id": "x"'))
            with self.assertRaises(EvidenceError):A.audit_sequence(root,rows,report,Checks())

    def test_all70_predictions_and_paired_gains_reconstructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,report,spec,states=inference_fixture(root)
            A.audit_test(root,rows,report,spec,states,Checks())
            report['paired_comparison']['net_captures']+=1
            with self.assertRaises(EvidenceError):A.audit_test(root,rows,report,spec,states,Checks())

    def test_actor_latent_command_or_adaptive_test_change_fails(self):
        for key,value in [('selection_or_learning',True),('latent_mean',[.1]*5),('command_values',[0.]*5)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);rows,report,spec,states=inference_fixture(root)
                path='predictions/test-fixed35_bc.json';packet=A.read(root/path)
                packet['rows'][0][key]=value;write(root,path,packet);rows[0][key]=value
                with self.subTest(key=key),self.assertRaises(EvidenceError):A.audit_test(root,rows,report,spec,states,Checks())

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Pinned Torch cross-schema check runs on Ubuntu CPU')
    def test_actual_mocked_producer_matches_independent_grid_prediction_sequence(self):
        from sircus.mjlab.tests.test_fixed35_imitation_study import StudyTests
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            W,args,engines,mocks=StudyTests().fixture(Path(tmp))
            with ExitStack() as stack:
                for mock in mocks:stack.enter_context(mock)
                report=W.run(args)
            rows=A.common.lines(args.output/'rows.jsonl');spec=A.read(args.output/'contract.json')
            states={m:torch.load(args.output/a['path'],weights_only=True,map_location='cpu') for m,a in report['artifacts'].items()}
            A.protocol_checks(spec['protocol'],Checks())
            A.audit_grid(args.output,spec,A.read(args.output/'bootstrap/teachers.json'),Checks())
            A.audit_sequence(args.output,rows,report,Checks())
            A.geometry.audit_request_packets(args.output,rows,Checks())
            A.audit_test(args.output,rows,report,spec,states,Checks())


class BehaviorTests(unittest.TestCase):
    def fixture(self):
        from sircus.mjlab.fixed35_imitation_native import yaw_target,BEHAVIOR_VERSION
        from sircus.mjlab.calibrated_admission import CONNECTION_POLICY_VERSION
        scene=dict(ball=dict(radius_m=.0375),board=dict(mount_yaw_rad=math.pi/2,tool_offset_m=.12),arms=dict(base_spacing_m=1.33731))
        values=[.65,-.2,.45,30.,.1];bucket=[1.1,-.5];angle,targets=yaw_target(scene,bucket,values)
        ext=dict(version=BEHAVIOR_VERSION,yaw_rad=angle,rotation_matrix=targets['dropper_flange_world'][:3,:3].tolist(),
            target_dropper_flange_world=targets['dropper_flange_world'].tolist(),geometry_hint_only=True)
        connection=dict(values=values,bucket_xy_m=bucket,resolution_sha256='a',release_behavior=ext,
            connection_policy=CONNECTION_POLICY_VERSION,action_substituted=False)
        connection['behavior_key']=A.digest(dict(version=BEHAVIOR_VERSION,connection_policy=CONNECTION_POLICY_VERSION,
            resolution_sha256='a',values=values,bucket_xy_m=bucket,release_behavior=ext))
        return scene,connection,targets

    def test_scalar_targets_match_producer_full_orientation(self):
        scene,connection,targets=self.fixture();independent=A.expected_targets(scene,connection['values'],connection['bucket_xy_m'])
        for key in ('dropper_flange_world','board_flange_local','release_center_m'):
            np.testing.assert_allclose(independent[key],targets[key],atol=1e-13,rtol=0)
        A.audit_behavior(connection,scene,Checks(),'test')

    def test_zero_yaw_half_turn_changed_height_or_stale_key_fail(self):
        for mode in ('zero','flip','height','key'):
            scene,connection,_=self.fixture()
            if mode=='zero':connection['release_behavior']['yaw_rad']=0.
            elif mode=='flip':connection['release_behavior']['rotation_matrix']=np.diag([-1.,1.,-1.]).tolist()
            elif mode=='height':connection['values'][4]+=.001
            else:connection['behavior_key']='stale'
            with self.subTest(mode=mode),self.assertRaises(EvidenceError):A.audit_behavior(connection,scene,Checks(),'test')

    def test_inadmissible_close_proposal_has_target_without_fabricated_flight(self):
        scene,_,_=self.fixture();targets=A.expected_targets(scene,[.6,-.2,.4,28.,.1],[.6,-.2])
        self.assertEqual(targets['yaw_rad'],0.)

    def test_inadmissible_negative_gap_requires_null_target_and_cannot_release(self):
        scene,connection,_=self.fixture();connection['values'][4]=-.01
        connection['release_behavior']['target_dropper_flange_world']=None;connection['passed']=False
        def rekey():
            connection['behavior_key']=A.digest(dict(version=connection['release_behavior']['version'],
                connection_policy=connection['connection_policy'],resolution_sha256=connection['resolution_sha256'],
                values=connection['values'],bucket_xy_m=connection['bucket_xy_m'],release_behavior=connection['release_behavior']))
        rekey();A.audit_behavior(connection,scene,Checks(),'invalid')
        connection['passed']=True
        with self.assertRaises(EvidenceError):A.audit_behavior(connection,scene,Checks(),'invalid')

    def test_independent_auditor_does_not_import_new_producer_or_online_score(self):
        forbidden={'fixed35_imitation_contract','fixed35_imitation_study','fixed35_imitation_native','table_assisted_score'}
        for node in ast.walk(ast.parse(Path(A.__file__).read_text())):
            if isinstance(node,ast.ImportFrom):
                self.assertFalse(any(x in forbidden for x in [node.module or '']+[n.name for n in node.names]))


if __name__=='__main__':unittest.main()
