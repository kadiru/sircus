"""CPU adversarial checks for the independent coverage experiment audit."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import coverage_training_verify as A
from sircus.mjlab import coverage_training_contract as C
from sircus.mjlab.benchmark_verify import Checks, EvidenceError


def original():
    return {'train':[dict(id=f'old-{i}',bucket_xy_m=[.1+(i%7)*.19,-.65+(i//7)*.225]) for i in range(35)]}


def outcome(ident='r',capture=True):
    row=dict(id=ident,role='train',phase='search',context_id='train-a',native_release=True,invalid=None,invalid_physics=False,
        route_success=capture,route_quality=2 if capture else 0,board_bounce=True,whole_ball_entry=capture,single_board_bounce_success=capture,
        max_retention_s=1. if capture else 0.,closest_opening_distance_m=.02,forbidden_contact_steps=0,
        direct_capture=capture,table_assisted_capture=False,scoring_version=A.oracle.VERSION,
        route_version=A.oracle.ROUTE_VERSION,reward_version=A.oracle.REWARD_VERSION)
    row['learning_reward']=A.oracle.reward(row)
    return row


def sequence(search=1260,per_seed=512):
    rows=[]
    def add(phase,role,count,**fields):
        for _ in range(count):rows.append(dict(id=str(len(rows)),phase=phase,role=role,**fields))
    for role in A.CONTROL_ROLES:add('controls_pre',role,3)
    add('search','train',search)
    add('validation','validation',35,seed=A.BC_SEED,samples=0)
    for offset in range(0,per_seed,32):
        for seed in A.SEEDS:
            add('train','train',32,seed=seed)
            if offset+32 in (256,512):add('validation','validation',35,seed=seed,samples=offset+32)
    for method in A.METHODS:add('test','test',70,method=method)
    for role in A.CONTROL_ROLES:add('controls_post',role,3)
    reasons=[]
    if search<1260:reasons.append(dict(stage='search',reason='search_or_training_deadline',elapsed_s=7300,
        search_elapsed_s=7200,round_index=search//105,completed_in_round=search%105))
    if per_seed<512:reasons.append(dict(stage='train',reason='training_deadline_reserve_test',elapsed_s=16200,samples_per_seed=per_seed))
    report=dict(search_requests=search,search_completed_rounds=search//105,search_partial_round_requests=search%105,
        training_budget_complete=search==1260 and per_seed==512,budget_reasons=reasons,
        deadlines=dict(worker_limit_s=19800,training_cut_s=16200,search_limit_s=7200),search_started_elapsed_s=100,elapsed_s=18000)
    return rows,report


class CoverageAudit(unittest.TestCase):
    def test_literal_protocol_detects_changed_budgets_or_scoring(self):
        A.protocol_checks(C.protocol(),Checks())
        for key,value in [('substeps',128),('test_layouts',50),('seeds',[A.SEEDS[0]]),('noise',True),
                          ('search_requests',1400),('maximum_fresh_requests',2756),('residual_scale',[.1]*5)]:
            changed=C.protocol();changed[key]=value
            with self.assertRaises(EvidenceError):A.protocol_checks(changed,Checks())

    def test_independent_stratified_split_and_full_novelty(self):
        parent=original();ex=[r['bucket_xy_m'] for r in parent['train']]+[[.8,-.333]]
        expected,excluded=C.make_splits(parent,ex)
        actual=A.regenerated_splits(parent,excluded)
        self.assertEqual(actual,expected)
        fresh=actual['train'][35:]+actual['validation']+actual['test'];seen=np.asarray(excluded)
        for r in fresh:
            self.assertGreaterEqual(np.linalg.norm(seen-r['bucket_xy_m'],axis=1).min(),.005)
            seen=np.vstack((seen,r['bucket_xy_m']))
        self.assertEqual([len(actual[k]) for k in ('train','validation','test')],[105,35,70])

    def test_inventory_is_frozen_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);history=root/'output';source=root/'src';study=root/'new'
            history.mkdir();(source/'baselines').mkdir(parents=True);study.mkdir()
            (history/'exclusions.json').write_text(json.dumps([[.8,-.333]]))
            inventory=C.collect_exclusions(history,source);parent=original()
            split,excluded=C.make_splits(parent,inventory['coordinates'])
            for name,obj in [('historical-coordinate-bindings.json',inventory),('splits.json',split),('exclusions.json',excluded)]:
                (study/name).write_text(json.dumps(obj))
            spec=dict(historical_coordinate_bindings_sha256=A.sha(study/'historical-coordinate-bindings.json'),splits=split,exclusions=excluded)
            result=A.audit_inventory(study,source,history,parent,spec,Checks())
            self.assertEqual(result['historical_coordinates'],36)
            # A new unrelated output after freezing must not change the split.
            (history/'later.json').write_text(json.dumps({'bucket_xy_m':[.234,-.444]}))
            A.audit_inventory(study,source,history,parent,spec,Checks())
            (history/'exclusions.json').write_text('[]')
            with self.assertRaises(EvidenceError):A.audit_inventory(study,source,history,parent,spec,Checks())

    def test_search_literal_axes_and_train_only_refinement(self):
        context=dict(id='train-a',split='train')
        best=dict(outcome(),normalized_residual=[.3,-.2,0,.25,.1])
        for index in range(12):
            x,m=A.search_proposal(context,index,best)
            y,n=C.search_residual(context,index,best)
            np.testing.assert_array_equal(x,y);self.assertEqual(m,n)
        self.assertEqual(A.search_proposal(context,10,best)[1]['source_training_ids'],['r'])
        for seed in A.SEEDS:self.assertEqual(A.seeded_schedule(seed),C.schedule(seed))

    def test_search_audit_rejects_heldout_context_and_modified_action(self):
        contexts=[dict(id=f'train-{i}',split='train',bucket_xy_m=[.3+i*.001,-.4]) for i in range(105)]
        rows=[];best={}
        for index in range(1260):
            c=contexts[index%105];r,meta=C.search_residual(c,index//105,best.get(c['id']))
            anchor=[.3,-.2,.46,28.,.2518];v=(np.asarray(anchor)+r*A.RESIDUAL).tolist()
            row=dict(outcome(f'coverage-search-{index:05d}'),context_id=c['id'],bucket_xy_m=c['bucket_xy_m'],method='guided_search',
                round_index=index//105,train_index=index,normalized_residual=r.tolist(),values=v,
                effective_source_hint=dict(values=anchor),search_metadata=meta,source_training_ids=meta['source_training_ids'])
            rows.append(row)
            if c['id'] not in best or row['id']<best[c['id']]['id']:best[c['id']]=row
        A.audit_search(rows,contexts,Checks())
        changed=copy.deepcopy(rows);changed[0]['context_id']='test-layout'
        with self.assertRaises(EvidenceError):A.audit_search(changed,contexts,Checks())
        changed=copy.deepcopy(rows);changed[-1]['values'][4]+=.001
        with self.assertRaises(EvidenceError):A.audit_search(changed,contexts,Checks())

    def test_bootstrap_excludes_historical_and_validation_labels(self):
        rows=[outcome('b'),outcome('a')]
        self.assertEqual(A.teachers_from_search(rows)[0]['id'],'a')
        for change in ({'role':'validation'},{'phase':'train'},{'phase':'historical'}):
            with self.assertRaises(EvidenceError):A.teachers_from_search([dict(rows[0],**change)])

    def test_checkpoint_choice_allows_bc_and_never_reads_test(self):
        curves=[dict(seed=A.BC_SEED,samples=0,validation=dict(captures=8,mean_reward=2)),
                dict(seed=A.SEEDS[0],samples=256,validation=dict(captures=8,mean_reward=2))]
        self.assertEqual(A.checkpoint_choice(curves)['samples'],0)
        curves[1]['validation']['captures']=9
        self.assertEqual(A.checkpoint_choice(curves)['samples'],256)
        curves.append(dict(seed=A.SEEDS[1],samples=256,validation=dict(captures=9,mean_reward=2)))
        self.assertEqual(A.checkpoint_choice(curves)['seed'],A.SEEDS[0])

    def test_complete_and_deadline_truncated_prefixes(self):
        for search,train in [(1260,512),(105+64,256),(32,0)]:
            rows,report=sequence(search,train)
            result=A.audit_sequence(rows,report,Checks())
            self.assertEqual(result['total_requests'],len(rows))
            self.assertEqual(result['full_training_budget'],search==1260 and train==512)
        rows,report=sequence(1260,512);self.assertEqual(len(rows),2757)
        with self.assertRaises(EvidenceError):A.audit_sequence(rows[:-1],report,Checks())
        rows,report=sequence(32,0);report['budget_reasons'][0]['search_elapsed_s']=100
        with self.assertRaises(EvidenceError):A.audit_sequence(rows,report,Checks())

    def test_no_unbalanced_seed_prefix_or_hidden_test_retry(self):
        rows,report=sequence(1260,512)
        first=next(i for i,r in enumerate(rows) if r['phase']=='train')
        rows[first]['seed']=A.SEEDS[1]
        with self.assertRaises(EvidenceError):A.audit_sequence(rows,report,Checks())
        rows,report=sequence(1260,512)
        rows.insert(-9,dict(rows[-10],id='extra-test-retry'))
        with self.assertRaises(EvidenceError):A.audit_sequence(rows,report,Checks())

    def test_preoutcome_packet_rejects_action_and_hash_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'requests').mkdir();name='requests/batch-00000.json'
            request=dict(id='r',phase='search',values=[1.,2.,3.,4.,5.])
            (p/name).write_text(json.dumps(dict(phase='search',rows=[request],pre_outcome=True)))
            digest=A.sha(p/name);row=dict(request,request_packet=name,request_packet_sha256=digest)
            timing=dict(phase='search',request_packet=name,request_packet_sha256=digest,acquired_start_s=1.,acquired_end_s=2.)
            (p/'timing.jsonl').write_text(json.dumps(timing)+'\n')
            A.audit_request_packets(p,[row],Checks())
            with self.assertRaises(EvidenceError):A.audit_request_packets(p,[dict(row,values=[0]*5)],Checks())
            with self.assertRaises(EvidenceError):A.audit_request_packets(p,[dict(row,request_packet_sha256='wrong')],Checks())

    def test_boundaries_freeze_all_test_predictions_before_test_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rows,_=sequence(32,0)
            raw=[(json.dumps(r)+'\n').encode() for r in rows];(p/'rows.jsonl').write_bytes(b''.join(raw))
            before_test=next(i for i,r in enumerate(rows) if r['phase']=='test')
            boundaries=[]
            for index,(phase,n) in enumerate([('controls_pre-complete',9),('search-complete',41),('bootstrap-complete',41),
                                             ('test-plan-frozen',before_test),('controls_post-complete',len(rows))]):
                boundaries.append(dict(phase=phase,rows_count=n,acquired_elapsed_s=index*10.,
                    rows_sha256=hashlib.sha256(b''.join(raw[:n])).hexdigest()))
            (p/'phase-boundaries.jsonl').write_text(''.join(json.dumps(b)+'\n' for b in boundaries))
            (p/'timing.jsonl').write_text(json.dumps(dict(phase='test',acquired_start_s=31.))+'\n')
            A.audit_boundaries(p,rows,Checks())
            boundaries[3]['acquired_elapsed_s']=32.
            (p/'phase-boundaries.jsonl').write_text(''.join(json.dumps(b)+'\n' for b in boundaries))
            with self.assertRaises(EvidenceError):A.audit_boundaries(p,rows,Checks())

    def test_effective_global_budget_reserves_cpu_audit_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            study=Path(tmp)/'run';study.mkdir()
            budget=dict(acquired_unix_s=3000.,limit_s=19800,training_cut_s=16200,search_limit_s=7200,
                effective_worker_limit_s=19600.,global_deadline_unix_s=26200.,audit_reserve_s=3600)
            pipeline=dict(started_unix_s=1000.,global_deadline_unix_s=26200.,global_limit_s=25200)
            study.with_name('run-budget.json').write_text(json.dumps(budget))
            study.with_suffix('.pipeline.json').write_text(json.dumps(pipeline))
            report=dict(deadlines=dict(effective_worker_limit_s=19600.,global_deadline_unix_s=26200.),elapsed_s=19000.)
            A.audit_budget(study,report,Checks())
            with self.assertRaises(EvidenceError):A.audit_budget(study,dict(report,elapsed_s=19700.),Checks())
            budget['effective_worker_limit_s']=19800.
            study.with_name('run-budget.json').write_text(json.dumps(budget))
            with self.assertRaises(EvidenceError):A.audit_budget(study,report,Checks())

    def test_statistics_count_rejections_and_table_capture(self):
        data=[outcome(),dict(outcome('table'),direct_capture=False,table_assisted_capture=True),
              outcome('miss',False),dict(id='reject',native_release=False,invalid='ik',route_success=False)]
        self.assertEqual(A.statistics(data),C.statistics(data))
        self.assertEqual(A.statistics(data)['attempts'],4)
        self.assertEqual(A.statistics(data)['captures'],2)

    def test_matched_controls_reject_wrist_branch_change(self):
        sources=[];replays=[];rows=[]
        fields=('source_hint','effective_source_hint','source_teacher_id','context_resolution_file',
                'context_resolution_sha256','context_home_sha256','context_layout_eligible','context_anchor_admitted',
                'context_hard_reject','observation','connection_file','connection_sha256')
        for i,role in enumerate(A.CONTROL_ROLES):
            source=dict(id='teacher-'+role,values=[.4,-.2,.3,28.,.25],bucket_xy_m=[.8,-.2]);sources.append(source)
            replay=dict(source,id='parent-'+role,q10=[.1]*6,q14=[0.,0.,0.,3.279484,1.9104921,-4.1933761])
            replay.update({f:None for f in fields});replays.append(replay)
        for phase in ('controls_pre','controls_post'):
            for source,replay,role in zip(sources,replays,A.CONTROL_ROLES):
                for repeat in range(3):
                    rows.append(dict(replay,id=f'coverage-{phase}-{role}-r{repeat}',phase=phase,role=role,method='control',repeat=repeat,
                        control_source_id=source['id'],control_source_sha256=A.digest(source),source_record_id=replay['id'],
                        source_record_sha256=A.digest(replay),variant='control_exact',adapter_directory='native-controls',
                        native_initial_q10=np.asarray(replay['q10'],np.float32).astype(float).tolist(),
                        native_initial_q14=np.asarray(replay['q14'],np.float32).astype(float).tolist()))
        with patch.object(A.native,'control_gate',return_value=dict(status='passed')):
            A.audit_controls(rows,sources,replays,Checks())
            changed=copy.deepcopy(rows);changed[3]['q14'][3]=.1378813
            with self.assertRaises(EvidenceError):A.audit_controls(changed,sources,replays,Checks())
            changed=copy.deepcopy(rows);changed[3]['adapter_directory']='native'
            with self.assertRaises(EvidenceError):A.audit_controls(changed,sources,replays,Checks())

    def test_namespace_routing_keeps_policies19_and_matched_controls17(self):
        with tempfile.TemporaryDirectory() as tmp:
            study=Path(tmp)
            for name in ('native','native-controls'):
                (study/name).mkdir()
                for file in ('effective-options.json','native-compiled-geometry.json'):(study/name/file).write_text('{}')
            rows=[dict(id='policy',phase='test',adapter_directory='native',native_release=True,effective_options_file='effective-options.json',
                       effective_options_sha256=A.sha(study/'native/effective-options.json')),
                  dict(id='control',phase='controls_pre',adapter_directory='native-controls',native_release=True,effective_options_file='effective-options.json',
                       effective_options_sha256=A.sha(study/'native-controls/effective-options.json'))]
            bank19=dict(records=list(range(19)));bank17=dict(records=list(range(17)))
            with patch.object(A.calibrated,'check_setup_evidence',return_value={}) as setup,patch.object(A.old,'audit_native_options'),\
                 patch.object(A.native,'rescore_rows',side_effect=lambda path,group,*args:{r['id']:{} for r in group}):
                result,count=A.audit_native_namespaces(study,rows,bank19,bank17,{},dict(scene={},identity=dict(trace_schema={})),{},study,Checks())
                self.assertEqual(count,2)
                self.assertEqual(set(result),{'native','native-controls'})
                self.assertEqual(setup.call_args_list[0].args[2],bank19)
                self.assertEqual(setup.call_args_list[1].args[2],bank17)
                self.assertEqual(setup.call_args_list[1].args[0]/'native',study/'native-controls')
                bad=copy.deepcopy(rows);bad[0]['adapter_directory']='native-controls'
                with self.assertRaises(EvidenceError):A.audit_native_namespaces(study,bad,bank19,bank17,{},dict(scene={},identity=dict(trace_schema={})),{},study,Checks())

    def test_inference_packet_rejects_online_search_or_changed_coordinates(self):
        context=dict(id='test-0',bucket_xy_m=[.5,-.2]);record=dict(sha256='a'*64,samples=0,seed=A.BC_SEED)
        prediction=dict(context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],method='bc',samples=0,seed=A.BC_SEED,
            artifact_sha256=record['sha256'],source_training_ids=[],observation=[0.]*9,latent_mean=[0.]*5,normalized_residual=[0.]*5,
            command_values=[.4,-.2,.3,28.,.25],effective_source_hint=dict(values=[.4,-.2,.3,28.,.25]),
            ball_flight_search_at_inference=False,geometry_search_at_inference=True,selection_or_learning=False)
        packet=dict(role='test',method='bc',samples=0,seed=A.BC_SEED,artifact=record,rows=[prediction],
                    selection_or_learning=False,one_prediction_per_layout=True)
        row=dict(prediction,id=f'coverage-test-bc-seed{A.BC_SEED}-0000-test-0',role='test',phase='test',
                 values=prediction['command_values'],frozen_artifact_sha256=record['sha256'])
        with patch.object(A.common,'network_forward',return_value=np.zeros(5)):
            A.audit_packet(packet,[row],[context],record,dict(policy={}),Checks())
            changed=copy.deepcopy(packet);changed['rows'][0]['ball_flight_search_at_inference']=True
            with self.assertRaises(EvidenceError):A.audit_packet(changed,[dict(row,ball_flight_search_at_inference=True)],[context],record,dict(policy={}),Checks())
            with self.assertRaises(EvidenceError):A.audit_packet(packet,[row],[dict(context,bucket_xy_m=[.6,-.2])],record,dict(policy={}),Checks())

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Pinned CPU Torch needed for exact gradient replay')
    def test_seeded_bootstrap_replays_actual_gradient_updates(self):
        from sircus.mjlab.policy64_model import initialized,bootstrap
        rows=[]
        for i in range(4):
            r=outcome(f'search-{i}');r.update(context_id=f'context-{i}',observation=[.02*i]*9,
                normalized_residual=[0.,0.,.1,-.25,.08])
            rows.append(r)
        model=initialized(A.BC_SEED);initial=copy.deepcopy(model.state_dict())
        fit=bootstrap(model,rows,rows)
        A.bootstrap_replay(initial,model.state_dict(),rows,rows,fit,Checks(),A.BC_SEED)
        altered=copy.deepcopy(model.state_dict());altered['actor.4.bias'][0]+=.01
        with self.assertRaises(EvidenceError):A.bootstrap_replay(initial,altered,rows,rows,fit,Checks(),A.BC_SEED)


if __name__=='__main__':unittest.main()
