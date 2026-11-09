"""Whole fixed comparison under mocked geometry/physics, never native GPU calls."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from sircus.mjlab.tests.test_corrected_imitation_contract import fixture, score_fields


@unittest.skipUnless(importlib.util.find_spec('torch'),'Pinned Torch tests run on Ubuntu CPU')
class StudyTests(unittest.TestCase):
    def fixture(self,directory,*,fail_control=False):
        from sircus.mjlab import corrected_imitation_study as W
        import torch
        src,corrected=fixture();anchor=src['effective_source_hint'];source=directory/'parent';source.mkdir()
        for name in ('calibration.json','verification.json','setup-home.json'):(source/name).write_text('{}\n')
        args=SimpleNamespace(**{key.replace('-','_'):source for key in W.INPUT_KEYS},output=directory/'study',wall_time_limit=3600,effective_worker_limit=3600,acquired_monotonic=0.)
        controls=[dict(src,id='control-'+role,context_id='control-'+role,role=role,phase='controls_pre',repeat=0,values=anchor['values']) for role in ('direct_control','table_control','negative_control')]
        def teacher(i):return dict(copy.deepcopy(corrected[0]),id=f'teacher-{i}',context_id=f'train-{i}',role='train',observation=src['observation'],normalized_residual=[.1]*5,learning_reward=4.)
        teachers=[teacher(i) for i in range(92)];new=[teacher(i) for i in range(92,97)]
        for row in new:row.update(source_correction_record_id='corrected',source_correction_record_sha256='c'*64)
        torch.manual_seed(7);model=W.ResidualPolicy();path=source/'current.pt';torch.save(dict(policy=model.state_dict(),samples=92,seed=7,weight_sha256=W.weight_hash(model),actor_sha256=W.weight_hash(model.actor)),path)
        artifact=dict(source_path=str(path),sha256=W.sha(path),weight_sha256=W.weight_hash(model),samples=92,seed=7)
        scene={'mock':'scene'};assets={'mock':'a'*64}
        loaded=dict(parent=dict(identity=dict(scene_sha256=W.digest(scene),asset_sha256=assets),packages={}),prior=dict(records=[anchor]*19),old_prior=dict(records=[anchor]*17),controls=controls,control_replays=controls,
            calibration={},home={},teachers=teachers,corrected_teachers=new,all_teachers=teachers+new,teacher_lineage=[],sources=[],artifacts={'current_bc':artifact},
            bindings=dict(corrections={k:dict(report_sha256='r'*64) for k in ('height','gap','ik')}))
        class Engine:
            def __init__(self,directory,prior):
                Path(directory).mkdir();self.prior=prior;self.adapter=SimpleNamespace(env=None,environment_factory=lambda:SimpleNamespace(effective_options={},compiled_geometry={}))
            def resolve(self,context):return anchor,{k:copy.deepcopy(src[k]) for k in W.META if k not in ('source_teacher_id','observation')},{}
            def evaluate(self,requests,phase):
                if phase=='test':
                    plan=json.loads((args.output/'test-plan.json').read_text());assert sum(len(m['rows']) for m in plan['methods'])==140
                    assert all(r['test_plan_sha256']==W.sha(args.output/'test-plan.json') for r in requests)
                out=[]
                for r in requests:
                    capture=r['role']!='negative_control' and not (phase.startswith('controls_') and fail_control)
                    out.append(dict(r,**dict(score_fields(capture),invalid_physics=False,scalar_score_audit_passed=True,
                        contact_protocol=dict(substeps=64),historical_v2_score=dict(route_success=capture and r['role']!='table_control'),
                        direct_capture=capture and r['role']!='table_control',table_assisted_capture=capture and r['role']=='table_control',
                        route_quality=1,forbidden_contact_steps=int(not capture),whole_ball_entry=capture,table_contact_steps=int(r['role']=='table_control'),reward=4. if capture else -1.)))
                return out,dict(requested=len(out),wall_s=.01)
        class Matched:
            instances=[]
            def __init__(self,ref,assets,directory,**kw):
                self.engine=Engine(directory,kw['prior']);self.parent_options={};self.parent_geometry={};self.instances.append(self)
            def evaluate(self,r,p):return self.engine.evaluate(r,p)
        mocks=[patch.object(W.C,'load_inputs',return_value=loaded),patch.object(W.C,'collect_exclusions',return_value=dict(coordinates=[src['bucket_xy_m']],files_sha256={})),
            patch.object(W,'load_runtime_reference',return_value=dict(scene=scene,asset_hashes=assets,runtime_reference={})),patch.object(W,'source_manifest',return_value={}),
            patch.object(W,'MatchedEngine',Matched),patch.object(W,'mirror_control_proofs',return_value=source),patch.object(W,'foreign_processes',return_value=[]),
            patch.object(W.shutil,'disk_usage',return_value=SimpleNamespace(free=50*1024**3)),patch.object(W.time,'monotonic',return_value=0.),
            patch.object(W,'bootstrap',return_value=dict(actor_steps=800,critic_steps=0)),patch('builtins.print')]
        return W,args,Matched,mocks

    def test_full158_accounting97teachers_and_two_frozen_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            W,args,engines,mocks=self.fixture(Path(tmp))
            with ExitStack() as stack:
                for m in mocks:stack.enter_context(m)
                report=W.run(args)
            rows=[json.loads(x) for x in (args.output/'rows.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),158);self.assertEqual(report['status'],'completed')
            self.assertEqual([r['method'] for r in rows if r['phase']=='test'],[m for m in W.C.METHODS for _ in range(70)])
            self.assertEqual(report['fit']['parent_teacher_count'],92);self.assertEqual(report['fit']['new_teacher_count'],5)
            self.assertEqual(len(json.loads((args.output/'bootstrap/teachers.json').read_text())),97)
            self.assertEqual(W.sha(args.parent_study/'current.pt'),report['artifacts']['current_bc']['sha256'])
            self.assertEqual(report['paired_comparison'],dict(paired_positions=70,both_capture=70,corrected_only=0,current_only=0,both_fail=0,net_captures=0))
            self.assertEqual([len(e.engine.prior['records']) for e in engines.instances],[19,17])
            self.assertIs(engines.instances[0].engine.adapter.environment_factory(),engines.instances[1].engine.adapter.environment_factory())
            self.assertEqual(len((args.output/'timing.jsonl').read_text().splitlines()),8)

    def test_failed_precontrol_stops_before_fit_or_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            W,args,engines,mocks=self.fixture(Path(tmp),fail_control=True)
            with ExitStack() as stack:
                for m in mocks:stack.enter_context(m)
                with self.assertRaises(ValueError):W.run(args)
            self.assertFalse((args.output/'fit.json').exists());self.assertFalse((args.output/'test-plan.json').exists())
            self.assertEqual(len((args.output/'rows.jsonl').read_text().splitlines()),9)

if __name__=='__main__':unittest.main()
