"""Complete controller-flow fixture with no native physics, GPU, SSH or services."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'),'Pinned Torch runner checks run on Ubuntu CPU')
class RunnerTests(unittest.TestCase):
    def fixture(self, directory, *, fail_controls=False, clock_jump=None, no_teachers=False):
        from sircus.mjlab import coverage_training_study as worker
        source=dict(id='shared',bucket_xy_m=[.4,-.3],values=[.5,-.2,.3,30.,.4],
                    q10=[0.]*6,q14=[0.]*6,source_record_sha256='a'*64)
        root=Path(__file__).resolve().parents[3]
        parent_splits=json.loads((root/'baselines/mjlab-calibrated-benchmark-20261104/splits.json').read_text())
        scene={'fixture':'mock-controller-only'};assets={'mock-asset':'b'*64}
        parent=dict(identity=dict(scene_sha256=worker.digest(scene),asset_sha256=assets,
                                  calibration={'mock':'checked'},setup=dict(home={'mock':'home'})),
                    packages={},splits=parent_splits)
        teacher=dict(id='teacher',role='train',context_id=parent_splits['train'][0]['id'],
                     bucket_xy_m=parent_splits['train'][0]['bucket_xy_m'],observation=[0.]*9,
                     normalized_residual=[.1]*5,learning_reward=3.5,values=source['values'])
        positive=dict(teacher,id='source-positive',values=[.4,-.2,.3,30.,.4])
        table=dict(teacher,id='source-table',values=[.5,-.2,.3,30.,.4])
        negative=dict(teacher,id='source-negative',values=[.6,-.2,.3,30.,.4])
        parent_path=directory/'parent';parent_path.mkdir()
        for name in ('calibration.json','verification.json','setup-home.json'):
            (parent_path/name).write_text('{}\n')
        args=SimpleNamespace(current_study=parent_path,current_audit=directory/'audit.json',
                             case00_study=directory/'case00',case00_audit=directory/'case00-audit.json',
                             policy_study=directory/'policy',policy_audit=directory/'policy-audit.json',
                             rescore_dir=directory/'rescore',history_root=directory/'history',
                             reference=directory/'reference.json',assets=directory/'assets',
                             output=directory/'study',wall_time_limit=19800,training_cut=16200,search_limit=7200,acquired_monotonic=0.)
        clock=SimpleNamespace(now=0.)

        import torch
        old_artifacts={}
        for method in ('current_bc','current_ppo'):
            model=worker.initialized(23); path=directory/(method+'.pt')
            torch.save(dict(policy=model.state_dict(),samples=0,weight_sha256=worker.weight_hash(model)),path)
            old_artifacts[method]=dict(path=str(path),relative_path='checkpoints/'+method+'.pt',
                sha256=worker.sha(path),weight_sha256=worker.weight_hash(model),samples=0)

        class FakeEngine:
            instances=[]
            def __init__(self,*a,**kw):
                self.prior=kw["prior"];self.calls=[];self.resolved=[];self.directory=Path(a[2]);self.directory.mkdir()
                self.instances.append(self)
            def resolve(self,context):
                self.resolved.append(copy.deepcopy(context))
                anchor=copy.deepcopy(source)
                meta=dict(source_hint=copy.deepcopy(source),effective_source_hint=copy.deepcopy(source),
                          context_resolution_file='fixture.json',context_resolution_sha256='c'*64,
                          context_home_sha256='d'*64,context_layout_eligible=True,
                          context_anchor_admitted=True,context_hard_reject=False)
                return anchor,meta,dict(wall_s=.001)
            def evaluate(self,requests,phase):
                self.calls.append((copy.deepcopy(requests),phase));result=[]
                if phase=='test':
                    plan=json.loads((self.directory.parent/'test-plan.json').read_text())
                    if len(plan['methods'])!=4:raise AssertionError('All test predictions must precede first outcome')
                    if sum(len(p['rows']) for p in plan['methods'])!=280:raise AssertionError('All280 predictions must precede TEST')
                    if any(r.get('test_plan_sha256')!=worker.sha(self.directory.parent/'test-plan.json') for r in requests):raise AssertionError('TEST must bind frozen plan hash')
                for r in requests:
                    capture=(r['role'] in ('direct_control','table_control') and not fail_controls) or (phase=='search' and not no_teachers and r['train_index']%11!=0)
                    table_hit=capture and r['role']=='table_control'
                    row=dict(r,native_release=True,invalid=None,invalid_physics=False,
                             route_success=capture,route_quality=2 if capture else 0,
                             board_bounce=True,whole_ball_entry=capture,max_retention_s=1. if capture else 0.,
                             single_bounce_success=capture,single_board_bounce_success=capture,closest_opening_distance_m=.01 if capture else .3,
                             external_contact_steps=int(table_hit),forbidden_contact_steps=0,table_contact_steps=int(table_hit),
                             table_assisted_capture=table_hit,direct_capture=capture and not table_hit,
                             route_class='table_assisted' if table_hit else 'direct' if capture else 'not_captured',
                             historical_v2_score=dict(scoring_version='sircus-calibrated-native-capture-v2',route_success=capture and not table_hit),
                             scalar_score_audit_passed=True,contact_protocol=dict(substeps=64),
                             scoring_version='sircus-table-assisted-native-capture-v3',
                             route_version='sircus-table-assisted-board-route-v3',reward_version='sircus-table-assisted-shot-reward-v3',
                             native_cache_sha256='e'*64,native_cache_file='fixture.npz',native_cache_row=0,
                             board_center_m=[.5,-.2,.3],board_normal=[0.,0.,1.],
                             trajectory_m=[[.5,-.2,.7],[.5,-.2,.3]],trajectory_dt_s=1/60)
                    if r['role']=='train' and r['train_index'] % 11 == 0:
                        row.update(native_release=False, invalid='mock_exact_setup_rejected')
                    result.append(row)
                if clock_jump is not None:clock.now=clock_jump(self.calls,clock.now)
                return result,dict(requested=len(requests),native_releases=len(requests),wall_s=.001)

        historical=json.loads((root/'baselines/mjlab-sweep-rl-benchmark-20261103/splits.json').read_text())
        exclusion=[c['bucket_xy_m'] for groups in (parent_splits,historical) for group in groups.values() for c in group]
        patches=[patch.object(worker.C,'load_inputs',return_value=dict(parent=parent,prior={'records':[dict(source,id=f'prior-{i}') for i in range(19)]},
                    old_prior={'records':[dict(source,id=f'prior-{i}') for i in range(17)]},train_splits=parent_splits,
                    bindings={'mock':'binding'},controls=[positive,table,negative],old_artifacts=old_artifacts,
                    calibration=parent['identity']['calibration'],home=parent['identity']['setup']['home'])),
                 patch.object(worker.C,'collect_exclusions',return_value=dict(coordinates=exclusion,files_sha256={},scope='mock')),
                 patch.object(worker,'load_runtime_reference',return_value=dict(scene=scene,asset_hashes=assets,
                     runtime_reference=dict(version='mock-current-reference',source_sha256={'fixture.py':'f'*64}))),
                 patch.object(worker,'source_manifest',return_value={'fixture.py':'f'*64}),
                 patch.object(worker,'Engine',FakeEngine),patch.object(worker.time,'monotonic',side_effect=lambda:clock.now),patch.object(worker,'foreign_processes',return_value=[]),
                 patch.object(worker.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)),
                 patch.object(worker,'bootstrap',return_value=dict(mock=True)),
                 patch.object(worker,'update',return_value=dict(mock_update=True)),patch('builtins.print')]
        return worker,args,FakeEngine,patches

    def run_fixture(self,directory,**kw):
        import torch
        from contextlib import ExitStack
        torch.set_num_threads(1)
        worker,args,engine,patches=self.fixture(directory,**kw)
        with ExitStack() as stack:
            mocks=[stack.enter_context(item) for item in patches]
            report=worker.run(args)
        rows=[json.loads(x) for x in (args.output/'rows.jsonl').read_text().splitlines()]
        return worker,args,engine,report,rows,mocks

    def test_complete_budget_fresh_teachers_interleaved_seeds_frozen_common_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker,args,engine,report,rows,_=self.run_fixture(Path(tmp))
            self.assertEqual(report['status'],'completed');self.assertEqual(len(rows),2757)
            expected=dict(search=1260,train=1024,validation=175,test=280,controls_pre=9,controls_post=9)
            self.assertEqual({p:sum(r['phase']==p for r in rows) for p in expected},expected)
            self.assertEqual(report['updates'],32);self.assertTrue(report['training_budget_complete'])
            self.assertEqual(len(engine.instances),1);self.assertEqual(len(engine.instances[0].prior['records']),19)
            self.assertEqual(report['search_completed_rounds'],12);self.assertEqual(report['search_partial_round_requests'],0)
            train=json.loads((args.output/'splits.json').read_text())['train']
            search=[r for r in rows if r['phase']=='search']
            self.assertEqual([r['context_id'] for r in search],[c['id'] for _ in range(12) for c in train])
            dataset=json.loads((args.output/'bootstrap/dataset.json').read_text())
            self.assertEqual(dataset,search)
            teachers=json.loads((args.output/'bootstrap/teachers.json').read_text())
            self.assertTrue(teachers);self.assertTrue(all(r['phase']=='search' and r['role']=='train' for r in teachers))
            self.assertEqual(len(report['search_coverage']),105)
            self.assertEqual(report['selected']['samples'],0) # Matched misses: earliest validation milestone wins.
            self.assertEqual([(x['seed'],x['samples']) for x in report['curves']],
                [(worker.C.BC_SEED,0),(worker.C.SEEDS[0],256),(worker.C.SEEDS[1],256),
                 (worker.C.SEEDS[0],512),(worker.C.SEEDS[1],512)])
            updates=[json.loads(x) for x in (args.output/'updates.jsonl').read_text().splitlines()]
            self.assertEqual([u['seed'] for u in updates],list(worker.C.SEEDS)*16)
            self.assertEqual(len({i for u in updates for i in u['train_ids']}),1024)
            rejected=[r for r in rows if r['phase']=='train' and not r['native_release']]
            self.assertEqual(len(rejected),94);self.assertTrue(all(r['learning_reward']==-1 for r in rejected))
            self.assertTrue({r['id'] for r in rejected}.issubset({i for u in updates for i in u['train_ids']}))
            plan=json.loads((args.output/'test-plan.json').read_text())
            self.assertEqual([p['method'] for p in plan['methods']],list(worker.C.METHODS))
            for method in worker.C.METHODS:
                actual=[r for r in rows if r['phase']=='test' and r['method']==method]
                self.assertEqual(len(actual),70);self.assertEqual(len({r['context_id'] for r in actual}),70)
                promised=next(p['rows'] for p in plan['methods'] if p['method']==method)
                self.assertEqual([r['values'] for r in actual],[r['command_values'] for r in promised])
                self.assertTrue(all(r['test_plan_sha256']==worker.sha(args.output/'test-plan.json') for r in actual))
                self.assertTrue(all(not r['ball_flight_search_at_inference'] for r in actual))
            for method in ('current_bc','current_ppo'):
                artifact=report['artifacts'][method]
                self.assertEqual(worker.sha(args.output/artifact['path']),artifact['sha256'])
                self.assertEqual(worker.sha(Path(tmp)/(method+'.pt')),artifact['sha256'])
            timing=[json.loads(x) for x in (args.output/'timing.jsonl').read_text().splitlines()]
            self.assertTrue(all('acquired_start_s' in t and 'acquired_end_s' in t and t['request_packet_sha256'] for t in timing))
            boundaries=[json.loads(x) for x in (args.output/'phase-boundaries.jsonl').read_text().splitlines()]
            freeze=next(x for x in boundaries if x['phase']=='test-plan-frozen')
            self.assertEqual(freeze['rows_count'],2468)
            self.assertEqual(engine.instances[0].calls[-1][1],'controls_post')

    def test_failed_initial_controls_never_search_train_or_test(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp:
            worker,args,engine,patches=self.fixture(Path(tmp),fail_controls=True)
            with ExitStack() as stack:
                for item in patches:stack.enter_context(item)
                with self.assertRaisesRegex(RuntimeError,'v3 control block failed'):worker.run(args)
            self.assertEqual([phase for _,phase in engine.instances[0].calls],['controls_pre'])
            self.assertFalse((args.output/'bootstrap/dataset.json').exists())
            self.assertFalse((args.output/'test-plan.json').exists())

    def test_no_fresh_teacher_never_substitutes_historical_outcomes(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp:
            worker,args,engine,patches=self.fixture(Path(tmp),no_teachers=True,
                clock_jump=lambda calls,now:7201. if calls[-1][1]=='search' else now)
            with ExitStack() as stack:
                for item in patches:stack.enter_context(item)
                with self.assertRaisesRegex(RuntimeError,'No fresh TRAIN capture teacher'):worker.run(args)
            self.assertFalse((args.output/'bootstrap/fit.json').exists())
            self.assertFalse((args.output/'test-plan.json').exists())

    def test_search_deadline_retains_literal_partial_prefix_and_full_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker,args,engine,report,rows,_=self.run_fixture(Path(tmp),
                clock_jump=lambda calls,now:7201. if calls[-1][1]=='search' else now)
            self.assertEqual(report['search_requests'],32);self.assertEqual(report['search_completed_rounds'],0)
            self.assertEqual(report['search_partial_round_requests'],32);self.assertFalse(report['training_budget_complete'])
            self.assertEqual(report['train_requests'],1024)
            self.assertEqual(sum(r['phase']=='test' for r in rows),280)
            reason=report['budget_reasons'][0]
            self.assertEqual(reason['stage'],'search');self.assertEqual(reason['limit_s'],7200)
            self.assertEqual(reason['search_elapsed_s'],7201.)

    def test_training_deadline_finishes_seed_pair_then_selects_completed_bc(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker,args,engine,report,rows,_=self.run_fixture(Path(tmp),
                clock_jump=lambda calls,now:16201. if calls[-1][1]=='train' else now)
            self.assertEqual(report['train_requests'],64);self.assertEqual(report['updates'],2)
            self.assertEqual([p['samples'] for p in report['curves']],[0]);self.assertEqual(report['selected']['samples'],0)
            self.assertEqual([report['per_seed'][str(seed)]['samples'] for seed in worker.C.SEEDS],[32,32])
            self.assertEqual(sum(r['phase']=='test' for r in rows),280)
            self.assertTrue(any(r['stage']=='train' and r['samples_per_seed']==32 for r in report['budget_reasons']))


if __name__=='__main__':unittest.main()
