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
    def fixture(self, directory, *, fail_controls=False):
        from sircus.mjlab import policy64_study as worker
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
        negative=dict(teacher,id='source-negative',values=[.6,-.2,.3,30.,.4])
        parent_path=directory/'parent';parent_path.mkdir()
        for name in ('calibration.json','verification.json','setup-home.json'):
            (parent_path/name).write_text('{}\n')
        args=SimpleNamespace(parent_study=parent_path,parent_audit=directory/'audit.json',
                             reference=directory/'reference.json',assets=directory/'assets',
                             output=directory/'study',wall_time_limit=7200)

        class FakeEngine:
            instances=[]
            def __init__(self,*a,**kw):
                self.calls=[];self.resolved=[];self.directory=Path(a[2]);self.directory.mkdir()
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
                for r in requests:
                    capture=r['role']=='positive_control' and not fail_controls
                    row=dict(r,native_release=True,invalid=None,invalid_physics=False,
                             route_success=capture,route_quality=2 if capture else 0,
                             board_bounce=True,whole_ball_entry=capture,max_retention_s=1. if capture else 0.,
                             single_bounce_success=capture,closest_opening_distance_m=.01 if capture else .3,
                             external_contact_steps=0,scalar_score_audit_passed=True,
                             contact_protocol=dict(substeps=64),scoring_version='sircus-calibrated-native-capture-v2',
                             native_cache_sha256='e'*64,native_cache_file='fixture.npz',native_cache_row=0,
                             board_center_m=[.5,-.2,.3],board_normal=[0.,0.,1.],
                             trajectory_m=[[.5,-.2,.7],[.5,-.2,.3]],trajectory_dt_s=1/60)
                    result.append(row)
                return result,dict(requested=len(requests),native_releases=len(requests),wall_s=.001)

        patches=[patch.object(worker.C,'load_parent',return_value=(parent,{'records':[source]},[teacher],[teacher],
                    dict(requests=1,teacher_count=1),{'mock':'binding'},[positive,negative])),
                 patch.object(worker.C,'load_runtime_reference',return_value=dict(scene=scene,asset_hashes=assets,
                     runtime_reference=dict(version='mock-current-reference',source_sha256={'fixture.py':'f'*64}))),
                 patch.object(worker,'source_manifest',return_value={'fixture.py':'f'*64}),
                 patch.object(worker,'Engine',FakeEngine),patch.object(worker,'foreign_processes',return_value=[]),
                 patch.object(worker.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)),
                 patch.object(worker,'bootstrap',return_value=dict(mock=True)),
                 patch.object(worker,'update',return_value=dict(mock_update=True)),patch('builtins.print')]
        return worker,args,FakeEngine,patches

    def test_complete_literal_budget_and_frozen_test_decisions_with_pre_post_controls(self):
        import torch
        from contextlib import ExitStack
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            worker,args,engine,patches=self.fixture(Path(tmp))
            with ExitStack() as stack:
                for item in patches:stack.enter_context(item)
                report=worker.run(args)
            rows=[json.loads(line) for line in (args.output/'rows.jsonl').read_text().splitlines()]
            self.assertEqual(report['status'],'completed');self.assertEqual(len(rows),1284)
            self.assertEqual(report['train_requests'],1024);self.assertEqual(report['updates'],32)
            self.assertEqual(len([r for r in rows if r['role']=='train']),1024)
            self.assertEqual(len([r for r in rows if r['role']=='validation']),48)
            self.assertEqual(len([r for r in rows if r['role']=='test']),200)
            self.assertEqual(len([r for r in rows if r['role'].endswith('_control')]),12)
            self.assertTrue(all(g['status']=='passed' for g in report['control_gates'].values()))
            self.assertEqual([p['samples'] for p in report['curves']],[0,256,512,1024])
            self.assertEqual(report['selected']['samples'],0) # Tied validation chooses earliest checkpoint.
            plan=json.loads((args.output/'test-plan.json').read_text())
            self.assertTrue(plan['frozen_before_outcomes'])
            self.assertEqual([p['method'] for p in plan['methods']],['prior','nearest','bc','ppo'])
            for method in ('prior','nearest','bc','ppo'):
                actual=[r for r in rows if r['role']=='test' and r['method']==method]
                self.assertEqual(len(actual),50);self.assertEqual(len({r['context_id'] for r in actual}),50)
                self.assertTrue(all(r['ball_flight_search_at_inference'] is False for r in actual))
                promised=next(p['rows'] for p in plan['methods'] if p['method']==method)
                self.assertEqual([r['values'] for r in actual],[p['command_values'] for p in promised])
            self.assertEqual(engine.instances[0].calls[-1][1],'controls_post')
            self.assertEqual(len((args.output/'updates.jsonl').read_text().splitlines()),32)

    def test_failed_initial_control_block_never_bootstraps_trains_or_evaluates_test(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp:
            worker,args,engine,patches=self.fixture(Path(tmp),fail_controls=True)
            with ExitStack() as stack:
                for item in patches:stack.enter_context(item)
                with self.assertRaisesRegex(RuntimeError,'control block failed'):
                    worker.run(args)
            rows=[json.loads(line) for line in (args.output/'rows.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),6)
            self.assertTrue(all(r['role'] in ('positive_control','negative_control') for r in rows))
            self.assertEqual([phase for _,phase in engine.instances[0].calls],['controls_pre'])
            self.assertFalse((args.output/'bootstrap/fit.json').exists())
            self.assertFalse((args.output/'selected.json').exists())
            self.assertFalse((args.output/'test-plan.json').exists())


if __name__=='__main__':unittest.main()
