"""Independent corrected-imitation provenance, split, inference and timing checks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
from sircus.mjlab import corrected_imitation_verify as A
from sircus.mjlab import corrected_imitation_contract as C
from sircus.mjlab.benchmark_verify import Checks, EvidenceError
from sircus.mjlab.tests.test_corrected_imitation_contract import fixture as one_correction
from sircus.mjlab.tests.test_coverage_training_verify import outcome


def write(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def teacher_sources(root):
    sources={k:dict(rows=[]) for k in ('parent','height','gap','ik')}
    for i,(kind,context,variant) in enumerate(A.SELECTION):
        original,group=one_correction();original.update(id='parent-'+context,context_id=context)
        for j,row in enumerate(group):
            scored=outcome(row['id']);scored.update(row);row.clear();row.update(scored)
            row.update(context_id=context,id=f'{kind}-{context}-{j}',variant=variant,source_record_id=original['id'],source_record_sha256=A.digest(original))
        sources['parent']['rows'].append(original);sources[kind]['rows'].extend(group)
    old=[]
    for i in range(92):
        row=outcome('old-'+str(i));row.update(context_id='old-context-'+str(i),bucket_xy_m=[.4,-.3],observation=[0.]*9,normalized_residual=[0.]*5)
        old.append(row)
    write(root,'bootstrap/teachers.json',old)
    sources['parent']['bound']=lambda name:root/name
    return sources,old


def native_sequence(root):
    rows=[]
    for phase in ('controls_pre','test','controls_post'):
        groups=[(role,'control',3) for role in A.ROLES] if phase!='test' else [('test',m,70) for m in A.METHODS]
        for role,method,n in groups:
            for i in range(n):rows.append(dict(id=str(len(rows)),phase=phase,role=role,method=method))
    raw=[(json.dumps(r)+'\n').encode() for r in rows];(root/'rows.jsonl').write_bytes(b''.join(raw))
    boundaries=[]
    for phase,count,t in [('inputs-frozen',0,0),('controls_pre-complete',9,1),('fit-complete',9,2),('test-plan-frozen',9,3),('test-complete',149,10),('controls_post-complete',158,12)]:
        boundaries.append(dict(phase=phase,rows_count=count,rows_sha256=hashlib.sha256(b''.join(raw[:count])).hexdigest() if count else None,acquired_elapsed_s=t))
    (root/'phase-boundaries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in boundaries))
    timings=[dict(phase=p,acquired_start_s=start,acquired_end_s=end) for p,start,end in [('controls_pre',0,1),*[( 'test',4+i,4.5+i) for i in range(6)],('controls_post',11,12)]]
    (root/'timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in timings))
    return rows,dict(elapsed_s=13),boundaries


def inference_fixture(root):
    test=A.regenerated_test([[.4,-.3]]);states={};rows=[];positions=[];packets=[];report=dict(artifacts={},test_comparison=[])
    source,_=one_correction();anchor=source['effective_source_hint']
    for mi,method in enumerate(A.METHODS):
        policy={}
        for layer,shape in ((0,(64,9)),(2,(64,64)),(4,(5,64))):
            policy[f'actor.{layer}.weight']=np.zeros(shape,np.float32);policy[f'actor.{layer}.bias']=np.zeros(shape[0],np.float32)
        states[method]=dict(policy=policy);artifact=dict(path=f'checkpoints/{method}.pt',sha256=str(mi)*64,weight_sha256='f'*64,samples=97 if mi==0 else 92,seed=1)
        report['artifacts'][method]=artifact;predictions=[];group=[]
        for i,context in enumerate(test):
            meta={k:copy.deepcopy(source[k]) for k in A.META};meta['observation']=A.calibrated.expected_observation(context['bucket_xy_m'],anchor).tolist()
            prediction=dict(meta,context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],method=method,command_values=anchor['values'],normalized_residual=[0.]*5,
                latent_mean=[0.]*5,source_training_ids=[],artifact_sha256=artifact['sha256'],ball_flight_search_at_inference=False,geometry_search_at_inference=True,selection_or_learning=False)
            predictions.append(prediction)
            row=outcome(f'corrected-imitation-test-{method}-{context["id"]}',capture=(i+mi)%3!=0)
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
    captures=[[A.native.captured(rows[j*70+i]) for j in range(2)] for i in range(70)]
    gains=sum(a and not b for a,b in captures);losses=sum(b and not a for a,b in captures)
    report['paired_comparison']=dict(paired_positions=70,both_capture=sum(a and b for a,b in captures),corrected_only=gains,current_only=losses,
        both_fail=sum(not a and not b for a,b in captures),net_captures=gains-losses)
    return rows,report,dict(splits=dict(test=test)),states


class AuditTests(unittest.TestCase):
    def test_literal_protocol_refuses_budget_or_method_changes(self):
        A.protocol_checks(C.protocol(),Checks())
        for key,value in [('substeps',128),('methods',['corrected_bc','old_bc']),('actor_steps',801),('teacher_count',107),('maximum_requests',159),('noise',True),('test_seed',1)]:
            p=C.protocol();p[key]=value
            with self.assertRaises(EvidenceError):A.protocol_checks(p,Checks())

    def test_independent_seeded_split_matches_and_excludes_every_old_coordinate(self):
        excluded=[[.1,-.65],[.4,-.3],[.8,-.2]];split=A.regenerated_test(excluded)
        self.assertEqual(split,C.test_split(excluded));self.assertEqual(len(split),70)
        points=np.asarray([p['bucket_xy_m'] for p in split]);distance=np.linalg.norm(points[:,None]-points[None,:],axis=2)+np.eye(70)
        self.assertGreaterEqual(distance.min(),.005)
        self.assertNotEqual(split,A.regenerated_test(excluded+[split[0]['bucket_xy_m']]))
        for bad in ([],[[float('nan'),0.]]):
            with self.assertRaises(EvidenceError):A.regenerated_test(bad)

    def test_reconstructs92unchanged_plus5_without_repeat_weighting(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources,old=teacher_sources(Path(tmp));before=copy.deepcopy(sources['height']['rows'])
            with patch.object(A,'authenticate_row'):
                copied,corrected,union,lineage=A.reconstructed_teachers(sources,Checks())
            self.assertEqual(copied,old);self.assertEqual(len(union),97);self.assertEqual(len(lineage),5)
            self.assertEqual(sources['height']['rows'],before)
            for kind,context,variant in A.SELECTION:
                original=next(r for r in sources['parent']['rows'] if r['context_id']==context)
                expected,_=C.corrected_teacher(sources[kind]['rows'],context,variant,original)
                actual=next(r for r in corrected if r['context_id']==context)
                self.assertEqual(actual,expected);self.assertEqual(actual['original_role'],'test')
                self.assertEqual(actual['source_correction_role'],'diagnostic')
                np.testing.assert_allclose(actual['normalized_residual'],[.25,-.25,.2,.125,1/12])

    def test_bad_confirmations_anchors_or_teacher_membership_rejected(self):
        mutations=[lambda s:s['height']['rows'][2].update(route_success=False),lambda s:s['height']['rows'].pop(),
            lambda s:s['height']['rows'][2].update(repeat=1),lambda s:s['height']['rows'][2].update(q14=[.7]*6),
            lambda s:s['height']['rows'][0].update(source_record_sha256='wrong'),lambda s:s['height']['rows'][0].update(observation=[0.]*9),
            lambda s:[r.update(values=[.9,-.2,.3,28.,.2]) for r in s['height']['rows']]]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as tmp:
                sources,_=teacher_sources(Path(tmp));mutate(sources)
                with patch.object(A,'authenticate_row'),self.assertRaises(EvidenceError):A.reconstructed_teachers(sources,Checks())

    def test_exact_native_proof_and_cache_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row=dict(id='r',native_release=True,adapter_directory='native')
            for key in ('connection','context_resolution','native_cache'):
                path=write(root,'native/'+key+'.json',{'proof':key});row[key+'_file']=key+'.json';row[key+'_sha256']=A.sha(path)
            source=dict(rows=[row],bound=lambda name:root/name)
            A.authenticate_row(source,row,Checks())
            (root/'native/native_cache.json').write_text('{}')
            with self.assertRaises(EvidenceError):A.authenticate_row(source,row,Checks())

    def test_full_sequence_and_pretest_freeze_are_independently_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,report,boundaries=native_sequence(root)
            self.assertEqual(A.audit_sequence(root,rows,report,Checks())['total_requests'],158)
            with self.assertRaises(EvidenceError):A.audit_sequence(root,rows[:-1],report,Checks())
            boundaries[3]['acquired_elapsed_s']=5
            (root/'phase-boundaries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in boundaries))
            with self.assertRaises(EvidenceError):A.audit_sequence(root,rows,report,Checks())

    def test_serialized_outcome_prefix_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,report,_=native_sequence(root)
            (root/'rows.jsonl').write_text((root/'rows.jsonl').read_text().replace('"id": "0"','"id": "x"'))
            with self.assertRaises(EvidenceError):A.audit_sequence(root,rows,report,Checks())

    def test_all140_predictions_and_paired_gains_reconstructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,report,spec,states=inference_fixture(root)
            A.audit_test(root,rows,report,spec,states,Checks())
            report['paired_comparison']['net_captures']+=1
            with self.assertRaises(EvidenceError):A.audit_test(root,rows,report,spec,states,Checks())

    def test_frozen_packet_tamper_or_adaptive_selection_is_detected(self):
        for field,value in [('selection_or_learning',True),('latent_mean',[.1]*5),('command_values',[0.]*5)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);rows,report,spec,states=inference_fixture(root)
                name='predictions/test-corrected_bc.json';packet=A.read(root/name);packet['rows'][0][field]=value;write(root,name,packet)
                rows[0][field]=value
                with self.assertRaises(EvidenceError):A.audit_test(root,rows,report,spec,states,Checks())

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch cross-schema runs on Ubuntu CPU')
    def test_actual_mocked_producer_matches_independent_full_inference_and_sequence(self):
        from sircus.mjlab.tests.test_corrected_imitation_study import StudyTests
        with tempfile.TemporaryDirectory() as tmp:
            W,args,engines,mocks=StudyTests().fixture(Path(tmp))
            with ExitStack() as stack:
                for m in mocks:stack.enter_context(m)
                report=W.run(args)
            rows=A.common.lines(args.output/'rows.jsonl');spec=A.read(args.output/'contract.json')
            import torch
            states={m:torch.load(args.output/a['path'],weights_only=True,map_location='cpu') for m,a in report['artifacts'].items()}
            A.audit_sequence(args.output,rows,report,Checks())
            A.geometry.audit_request_packets(args.output,rows,Checks())
            A.audit_test(args.output,rows,report,spec,states,Checks())

if __name__=='__main__':unittest.main()
