"""Full finite recovery scheduling against mocks, with no GPU or services."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from sircus.mjlab.tests.test_fixed35_recovery_contract import fixture,score_fields


@unittest.skipUnless(importlib.util.find_spec('torch'),'Pinned Torch import dependencies tested on Ubuntu CPU')
class StudyTests(unittest.TestCase):
    def run_mock(self,root,*,search_capture=True,repeat_capture=True,fail_control=False,lose_preserved=False,cutoff=False):
        from sircus.mjlab import fixed35_recovery_study as W
        data=fixture();source=root/'parent';source.mkdir()
        for name in ('calibration.json','verification.json','setup-home.json'):(source/name).write_text('{}\n')
        (source/'native-yaw').mkdir()
        for name in ('effective-options.json','native-compiled-geometry.json'):(source/'native-yaw'/name).write_text('{}\n')
        args=SimpleNamespace(parent_study=source,parent_audit=root/'audit.json',reference=root/'ref.json',assets=root/'assets',
            output=root/'study',wall_time_limit=7200,effective_worker_limit=7200,acquired_monotonic=0.)
        seed=data['preserved'][0];controls=[dict(copy.deepcopy(seed),id='control-'+role,context_id='control-'+role,role=role,
            phase='controls_pre',repeat=0) for role in ('direct_control','table_control','negative_control')]
        scene={'mock':'scene'};assets={'mock':'a'*64};clock=[0.]
        loaded=dict(data,parent=dict(identity=dict(scene_sha256=W.digest(scene),asset_sha256=assets),packages={}),
            parent_report={},prior=dict(records=[seed]*19),old_prior=dict(records=[seed]*17),controls=controls,control_replays=controls,
            calibration={},home={},bindings={},artifacts={},diagnoses=[])
        calls=[]
        class Engine:
            def __init__(self,directory):
                Path(directory).mkdir();self.adapter=SimpleNamespace(env=None,environment_factory=lambda:None)
            def evaluate(self,requests,phase):
                # Every flight sees an already frozen full plan, including controls.
                assert len(json.loads((args.output/'search-plan.json').read_text())['rows'])==720
                calls.append((phase,copy.deepcopy(requests)));result=[]
                for r in requests:
                    capture=(r['role']!='negative_control') if phase.startswith('controls_') else search_capture if phase=='search' else repeat_capture if phase=='repeat' else True
                    if fail_control and phase.startswith('controls_'):capture=False
                    if lose_preserved and phase=='preserved' and r['context_id']==data['preserved'][0]['context_id']:capture=False
                    if phase=='repeat':
                        p=json.loads((args.output/r['repeat_plan_file']).read_text());assert p['frozen_before_repeat_outcomes'] is True
                        assert W.sha(args.output/r['repeat_plan_file'])==r['repeat_plan_sha256']
                    row=dict(r,**dict(score_fields(capture),invalid_physics=False,scalar_score_audit_passed=True,
                        contact_protocol=dict(substeps=64),historical_v2_score=dict(route_success=capture and r['role']!='table_control'),
                        direct_capture=capture and r['role']!='table_control',table_assisted_capture=capture and r['role']=='table_control',
                        route_quality=1,forbidden_contact_steps=int(not capture),whole_ball_entry=capture,
                        table_contact_steps=int(r['role']=='table_control'),reward=4. if capture else -1.),
                        connection_sha256='d'*64,q10=[.1]*6,q14=[.2]*6)
                    result.append(row)
                if cutoff and phase=='search':clock[0]=5401.
                return result,dict(requested=len(result),wall_s=.01)
        class Matched:
            def __init__(self,ref,assets,directory,**kw):self.engine=Engine(directory)
            def evaluate(self,r,p):return self.engine.evaluate(r,p)
        mocks=[patch.object(W.C,'load_inputs',return_value=loaded),patch.object(W,'load_runtime_reference',return_value=dict(scene=scene,asset_hashes=assets,runtime_reference={})),
            patch.object(W,'source_manifest',return_value={}),patch.object(W,'MatchedEngine',Matched),
            patch.object(W,'RecoveryEngine',side_effect=lambda ref,assets,directory,**kw:Engine(directory)),
            patch.object(W,'mirror_control_proofs',return_value=source),patch.object(W,'foreign_processes',return_value=[]),
            patch.object(W.shutil,'disk_usage',return_value=SimpleNamespace(free=50*1024**3)),patch.object(W.time,'monotonic',side_effect=lambda:clock[0]),patch('builtins.print')]
        with ExitStack() as stack:
            for m in mocks:stack.enter_context(m)
            if fail_control:
                with self.assertRaises(ValueError):W.run(args)
                report=json.loads((args.output/'report.json').read_text())
            else:report=W.run(args)
        rows=[json.loads(x) for x in (args.output/'rows.jsonl').read_text().splitlines()]
        return W,report,rows,calls,args

    def test_all_nine_confirmed_then_stop_and_preserve24(self):
        with tempfile.TemporaryDirectory() as t:
            W,report,rows,calls,a=self.run_mock(Path(t))
            self.assertEqual(len(rows),78);self.assertEqual(report['recovered_layouts'],9)
            self.assertEqual(report['demonstrated_recipe_coverage']['fresh_total'],33)
            self.assertTrue(report['preservation']['all_recaptured']);self.assertFalse(report['training'])
            self.assertEqual(report['completion_reason'],'all_searchable_cases_confirmed')
            self.assertEqual([p for p,r in calls],['controls_pre','preserved','search','repeat','controls_post'])
            self.assertEqual(len(calls[3][1]),27);self.assertEqual(len(list((a.output/'repeat-plans').glob('*.json'))),1)
            self.assertEqual(sum(r['status']=='fixed_base_blocked' for r in report['cases']),2)
            self.assertFalse((a.output/'checkpoints').exists());self.assertIn('recipes.json',report['evidence_sha256'])

    def test_three_failed_confirmation_groups_stop_and_never_count_discovery_as_reliable(self):
        with tempfile.TemporaryDirectory() as t:
            _,report,rows,calls,_=self.run_mock(Path(t),repeat_capture=False)
            self.assertEqual(len(rows),150);self.assertEqual(report['stage_counts']['search']['attempts'],27)
            self.assertEqual(report['stage_counts']['repeat']['attempts'],81);self.assertEqual(report['recovered_layouts'],0)
            self.assertEqual(sum(r['status']=='confirmation_limit' for r in report['cases']),9)
            self.assertEqual(report['completion_reason'],'finite_candidate_or_confirmation_caps')
            self.assertFalse(report['budget_truncated'])

    def test_no_captures_exhausts720_and_does_not_repeat_failures(self):
        with tempfile.TemporaryDirectory() as t:
            _,report,rows,_,_=self.run_mock(Path(t),search_capture=False)
            self.assertEqual(len(rows),762);self.assertEqual(report['stage_counts']['search']['attempts'],720)
            self.assertEqual(report['stage_counts']['repeat']['attempts'],0)
            self.assertEqual(sum(r['status']=='search_exhausted' for r in report['cases']),9)

    def test_cutoff_finishes_controls_but_does_not_claim_exhaustive_completion(self):
        with tempfile.TemporaryDirectory() as t:
            _,report,rows,_,_=self.run_mock(Path(t),search_capture=False,cutoff=True)
            self.assertEqual(len(rows),51);self.assertTrue(report['budget_truncated'])
            self.assertEqual(report['completion_reason'],'search_time_limit');self.assertFalse(report['gate']['search_cap_completed'])
            self.assertEqual(sum(r['status']=='budget_truncated' for r in report['cases']),9)
            self.assertEqual(report['control_gates']['post']['status'],'passed')

    def test_lost_preserved_capture_is_visible_not_replaced_or_hidden(self):
        with tempfile.TemporaryDirectory() as t:
            _,report,_,_,_=self.run_mock(Path(t),lose_preserved=True)
            self.assertEqual(report['preservation']['captures'],23);self.assertFalse(report['preservation']['all_recaptured'])
            self.assertEqual(report['demonstrated_recipe_coverage']['fresh_total'],32)

    def test_failed_precontrol_stops_before_preservation_or_search(self):
        with tempfile.TemporaryDirectory() as t:
            _,report,rows,calls,_=self.run_mock(Path(t),fail_control=True)
            self.assertEqual(len(rows),9);self.assertEqual(len(calls),1)
            self.assertNotEqual(report['status'],'completed')

if __name__=='__main__':unittest.main()
