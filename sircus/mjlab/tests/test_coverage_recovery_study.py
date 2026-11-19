"""Whole recovery flow with mocked admission/physics; Ubuntu CPU Torch only."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from sircus.mjlab.tests.test_coverage_recovery_contract import fixture_cases,score_fields


@unittest.skipUnless(importlib.util.find_spec('torch'),'Pinned Torch controller tests run on Ubuntu CPU')
class ControllerTests(unittest.TestCase):
    def fixture(self,directory,*,no_captures=False,fail_control=False,jump=None):
        from sircus.mjlab import coverage_recovery_study as W
        import torch
        cases=fixture_cases();anchor=cases[0]['source_hint'];source=directory/'parent';source.mkdir()
        for name in ('calibration.json','verification.json','setup-home.json'):(source/name).write_text('{}\n')
        args=SimpleNamespace(parent_study=source,parent_audit=directory/'audit',history_root=directory/'history',
            reference=directory/'reference',assets=directory/'assets',output=directory/'study',wall_time_limit=7200,
            search_limit=3600,repeat_cut=4500,acquired_monotonic=0.)
        controls=[dict(cases[0],id='control-'+role,context_id='control-'+role,role=role,phase='controls_pre',repeat=0)
            for role in ('direct_control','table_control','negative_control')]
        teachers=[dict(cases[0],**score_fields(),id=f'teacher-{i}',context_id=f'train-{i}',role='train',phase='search',learning_reward=4.) for i in range(68)]
        artifacts={}
        for method in ('parent_bc','parent_ppo','previous_ppo'):
            torch.manual_seed(7);model=W.ResidualPolicy();path=source/(method+'.pt')
            torch.save(dict(policy=model.state_dict(),samples=0,seed=7,weight_sha256=W.weight_hash(model)),path)
            artifacts[method]=dict(source_path=str(path),sha256=W.sha(path),weight_sha256=W.weight_hash(model),samples=0,seed=7)
        scene={'mock':'scene'};assets={'mock':'a'*64};clock=SimpleNamespace(now=0.)
        parent=dict(identity=dict(scene_sha256=W.digest(scene),asset_sha256=assets),packages={})
        loaded=dict(parent=parent,prior=dict(records=[dict(anchor,id=f'prior-{i}') for i in range(19)]),
            old_prior=dict(records=[dict(anchor,id=f'prior-{i}') for i in range(17)]),
            controls=controls,control_replays=controls,calibration={},home={},teachers=teachers,cases=cases,
            blocked=[dict(cases[0],id=f'blocked-{i}',context_id=f'blocked-{i}',invalid='fixed_base_clearance_rejected') for i in range(3)],
            artifacts=artifacts,rows=cases,bindings={'mock':'bindings'})
        class Engine:
            def __init__(self,directory,prior):
                self.directory=Path(directory);self.directory.mkdir();self.prior=prior
                self.adapter=SimpleNamespace(env=None,environment_factory=lambda:SimpleNamespace(effective_options={},compiled_geometry={}))
                self.calls=[]
            def resolve(self,context):
                return copy.deepcopy(anchor),{k:copy.deepcopy(cases[0][k]) for k in W.META if k not in ('source_teacher_id','observation')},{}
            def evaluate(self,requests,phase):
                self.calls.append((phase,copy.deepcopy(requests)));out=[]
                if phase=='test':
                    plan=json.loads((args.output/'test-plan.json').read_text())
                    if sum(len(m['rows']) for m in plan['methods'])!=280:raise AssertionError('All predictions must be frozen')
                    if any(r['test_plan_sha256']!=W.sha(args.output/'test-plan.json') for r in requests):raise AssertionError('TEST needs frozen plan hash')
                for request in requests:
                    capture=(request['role'] in ('direct_control','table_control') and not fail_control) or (not no_captures and (phase=='repeat' or phase=='search' and request['proposal_index']==1))
                    row=dict(request,**score_fields(capture));row.update(q10=[0.]*6,q14=[0.]*6,connection_file='mock.json',connection_sha256='e'*64,
                        invalid_physics=False,scalar_score_audit_passed=True,contact_protocol=dict(substeps=64),
                        historical_v2_score=dict(route_success=capture and request['role']!='table_control'),whole_ball_entry=capture,table_contact_steps=int(capture and request['role']=='table_control'),
                        table_assisted_capture=capture and request['role']=='table_control',direct_capture=capture and request['role']!='table_control')
                    if phase=='search' and request['proposal_index']==0 and int(request['context_id'].split('-')[-1])<19:
                        row.update(native_release=False,invalid='finite_ik_unresolved')
                    out.append(row)
                if jump is not None:clock.now=jump(phase,clock.now)
                return out,dict(requested=len(out),wall_s=.01)
        class Matched:
            instances=[]
            def __init__(self,ref,assets,directory,**kwargs):
                self.engine=Engine(directory,kwargs['prior']);self.parent_options={};self.parent_geometry={}
                self.instances.append(self)
            def evaluate(self,requests,phase):return self.engine.evaluate(requests,phase)
        mocks=[patch.object(W.C,'load_inputs',return_value=loaded),patch.object(W.C,'collect_exclusions',return_value=dict(coordinates=[[.1,-.65]],files_sha256={})),
            patch.object(W,'load_runtime_reference',return_value=dict(scene=scene,asset_hashes=assets,runtime_reference={})),
            patch.object(W,'source_manifest',return_value={'fixture.py':'f'*64}),patch.object(W,'MatchedEngine',Matched),
            patch.object(W,'mirror_control_proofs',return_value=source),patch.object(W,'foreign_processes',return_value=[]),
            patch.object(W.shutil,'disk_usage',return_value=SimpleNamespace(free=50*1024**3)),
            patch.object(W.time,'monotonic',side_effect=lambda:clock.now),patch.object(W,'bootstrap',return_value=dict(actor_steps=800,critic_steps=0)),patch('builtins.print')]
        return W,args,Matched,mocks

    def invoke(self,directory,**kw):
        W,args,engine,mocks=self.fixture(directory,**kw)
        with ExitStack() as stack:
            for mock in mocks:stack.enter_context(mock)
            report=W.run(args)
        rows=[json.loads(x) for x in (args.output/'rows.jsonl').read_text().splitlines()]
        return W,args,engine,report,rows

    def test_full1000_accounting_round_robin_repeats_actor_only_and_four_frozen_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            W,args,engine,report,rows=self.invoke(Path(tmp))
            self.assertEqual(report['status'],'completed');self.assertEqual(len(rows),1000)
            self.assertEqual({p:sum(r['phase']==p for r in rows) for p in ('search','repeat','test','controls_pre','controls_post')},
                dict(search=624,repeat=78,test=280,controls_pre=9,controls_post=9))
            self.assertEqual(report['confirmed_teachers'],39);self.assertEqual(report['fit']['new_teacher_count'],39)
            self.assertEqual(report['fit']['critic_steps'],0);self.assertEqual(report['fit']['critic_before_sha256'],report['fit']['critic_after_sha256'])
            self.assertEqual(len(json.loads((args.output/'bootstrap/teachers.json').read_text())),107)
            self.assertEqual([len(x.engine.prior['records']) for x in engine.instances],[19,17])
            self.assertIs(engine.instances[0].engine.adapter.environment_factory(),engine.instances[1].engine.adapter.environment_factory())
            self.assertEqual([r['method'] for r in rows if r['phase']=='test'],[m for m in W.C.METHODS for _ in range(70)])
            self.assertTrue(all(r['adapter_directory']==('native-controls' if r['phase'].startswith('controls_') else 'native') for r in rows))
            for method in ('parent_bc','parent_ppo','previous_ppo'):
                self.assertEqual(W.sha(args.parent_study/(method+'.pt')),report['artifacts'][method]['sha256'])

    def test_no_new_capture_copies_parent_weights_instead_of_claiming_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            W,args,engine,report,rows=self.invoke(Path(tmp),no_captures=True)
            self.assertEqual(len(rows),922);self.assertEqual(report['confirmed_teachers'],0)
            self.assertEqual(report['fit']['status'],'no_confirmed_new_teacher');self.assertEqual(report['fit']['actor_steps'],0)
            self.assertEqual(report['artifacts']['recovery_bc']['sha256'],report['artifacts']['parent_bc']['sha256'])

    def test_search_timecap_preserves_prefix_and_still_finishes_full_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            W,args,engine,report,rows=self.invoke(Path(tmp),jump=lambda phase,now:3601. if phase=='search' else now)
            self.assertEqual(report['search_requests'],32);self.assertFalse(report['search_budget_complete'])
            self.assertEqual(report['budget_reasons'][0]['completed'],32)
            self.assertEqual(sum(r['phase']=='test' for r in rows),280)

    def test_repeat_cut_excludes_unconfirmed_teachers_and_preserves_requested_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            W,args,engine,report,rows=self.invoke(Path(tmp),jump=lambda phase,now:4501. if phase=='repeat' else now)
            self.assertEqual(report['repeat_requests'],2);self.assertEqual(report['confirmed_teachers'],1)
            self.assertFalse(report['repeat_budget_complete']);self.assertEqual(sum(r['phase']=='test' for r in rows),280)

    def test_failed_controls_prevent_search_and_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            W,args,engine,mocks=self.fixture(Path(tmp),fail_control=True)
            with ExitStack() as stack:
                for mock in mocks:stack.enter_context(mock)
                with self.assertRaisesRegex(RuntimeError,'control block failed'):W.run(args)
            self.assertEqual(engine.instances[0].engine.calls,[])
            self.assertFalse((args.output/'fit.json').exists());self.assertFalse((args.output/'test-plan.json').exists())


if __name__=='__main__':unittest.main()
