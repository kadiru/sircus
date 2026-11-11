"""Worker quota, evidence commits and fail-closed control tests without a GPU."""
import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from sircus.mjlab.tests.test_fixed35_bouncer_contract import fixture, score_fields


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Full import dependencies tested on Ubuntu CPU')
class StudyTests(unittest.TestCase):
    def run_mock(self, root, *, capture=True, decline=False, cutoff=False, window=False, window_build_cutoff=False, fail_control=False, lose_preserved=False):
        from sircus.mjlab import fixed35_sixhour_study as W
        d=fixture(); parent=root/'parent'; parent.mkdir(); (parent/'native-yaw').mkdir()
        for name in ('calibration.json','verification.json','setup-home.json','geometry-audit.json',
                     'native-yaw/effective-options.json','native-yaw/native-compiled-geometry.json'):
            (parent/name).write_text('{}\n')
        roles=('direct_control','table_control','negative_control'); seed=d['preserved'][0]
        controls=[dict(copy.deepcopy(seed),id='control-'+r,role=r) for r in roles]
        scene={}; assets={}; clock=[0.]; calls=[]
        loaded=dict(d,parent=dict(identity=dict(scene_sha256=W.digest(scene),asset_sha256=assets),packages={}),
            broad_sources=[],broad_source_roots={},broad_rows=[],prior={},old_prior={},controls=controls,control_replays=controls,calibration={},home={},bindings={},artifacts={})
        plan=[]
        for c in (W.C.TARGET,):
            source=next(r for r in d['numeric_sources'] if r['context_id']==c)
            for i in range(600):
                values=list(source['values']);values[4]+=.00001*i
                plan.append(W.C.parent.request(source,identity=f'proposal-{c}-{i}',phase='search',method='broad_search',
                    values=values,proposal_index=i,exact_parent_action=False))
        a=SimpleNamespace(parent_study=parent,original_study=parent,recovery_study=parent,reference=root/'ref',assets=root/'assets',
            output=root/'out',broad_study=parent,wall_time_limit=18000,search_limit=16200,effective_worker_limit=18000,acquired_monotonic=0.)
        class Engine:
            def __init__(self,directory):
                self.directory=Path(directory);self.directory.mkdir();self.source_receipts=[];self.number=0
                self.adapter=SimpleNamespace(env=None,environment_factory=lambda:None);self.engine=self
            def prepare(self,requests):
                prepared=[];records=[]
                for r in requests:
                    ids=[]
                    if not decline:
                        for slot in range(2):
                            p=dict(r,id=r['id']+f'-b{slot}',native_prepared=True,selected_q10=[.1]*6,selected_q14=[.2+slot]*6)
                            prepared.append(p);ids.append(p['id'])
                    records.append(dict(proposal_id=r['id'],admitted_request_ids=ids))
                return prepared,records
            def evaluate(self,requests,phase):
                assert (a.output/'search-plan.json').exists()
                calls.append(phase);result=[];cache=self.directory/f'native-{self.number:04d}.npz';self.number+=1
                cache.write_bytes(b'fixture-only-native-archive')
                for r in requests:
                    yes=(r['role']!='negative_control') if phase.startswith('controls') else capture if phase in ('search','repeat') else True
                    if fail_control and phase.startswith('controls'):yes=False
                    if lose_preserved and phase=='preserved':yes=False
                    table=r.get('role')=='table_control'
                    result.append(dict(r,**dict(score_fields(yes),invalid_physics=False,scalar_score_audit_passed=True,
                        contact_protocol=dict(substeps=64),historical_v2_score=dict(route_success=yes and not table),
                        direct_capture=yes and not table,table_assisted_capture=yes and table,route_quality=1,
                        forbidden_contact_steps=int(not yes),whole_ball_entry=yes,table_contact_steps=int(table),reward=4. if yes else -1.),
                        q10=r.get('selected_q10',[.1]*6),q14=r.get('selected_q14',[.2]*6),connection_sha256='d'*64,
                        native_cache_file=cache.name,native_cache_sha256=W.sha(cache)))
                if cutoff and phase=='search':clock[0]=16201.
                if window and phase=='repeat':clock[0]=10801.
                return result,dict(requested=len(result),wall_s=.01)
        def make_window(*args,**kwargs):
            if window_build_cutoff:
                clock[0]=16201.
                return dict(rows=[dict(plan[0],id='fresh-window-proposal')])
            return dict(rows=[])
        mocks=[patch.object(W.C,'load_inputs',return_value=loaded),patch.object(W.C,'make_plan',return_value=dict(rows=plan)),
            patch.object(W.C,'make_window_plan',side_effect=make_window),
            patch.object(W,'load_runtime_reference',return_value=dict(scene=scene,asset_hashes=assets,runtime_reference={})),
            patch.object(W,'source_manifest',return_value={}),patch.object(W,'SixHourEngine',side_effect=lambda r,a,p,**kw:Engine(p)),
            patch.object(W,'MatchedEngine',side_effect=lambda r,a,p,**kw:Engine(p)),patch.object(W,'mirror_control_proofs',return_value=parent),
            patch.object(W,'foreign_processes',return_value=[]),patch.object(W.shutil,'disk_usage',return_value=SimpleNamespace(free=50*1024**3)),
            patch.object(W.time,'monotonic',side_effect=lambda:clock[0]),patch('builtins.print')]
        with ExitStack() as stack:
            for m in mocks:stack.enter_context(m)
            if fail_control or lose_preserved:
                with self.assertRaises(ValueError):W.run(a)
                report=json.loads((a.output/'report.json').read_text())
            else:report=W.run(a)
        rows=[json.loads(x) for x in (a.output/'rows.jsonl').read_text().splitlines()]
        return W,report,rows,calls,a

    def test_full_limit_continues_after_captures(self):
        with TemporaryDirectory() as t:
            W,r,rows,calls,a=self.run_mock(Path(t))
            self.assertEqual(r['status'],'completed');self.assertEqual(sum(r['fresh_search_releases'].values()),1200)
            self.assertEqual(len(rows),1257);self.assertEqual(r['stage_counts']['repeat']['attempts'],9)
            self.assertTrue(r['preservation']['all_recaptured']);self.assertEqual(calls[-1],'controls_post')
            commits=[json.loads(x) for x in (a.output/'packet-commits.jsonl').read_text().splitlines()]
            self.assertEqual(commits[-1]['rows_sha256'],W.sha(a.output/'rows.jsonl'))
    def test_declines_never_count_as_drops(self):
        with TemporaryDirectory() as t:
            _,r,rows,_,_=self.run_mock(Path(t),decline=True)
            self.assertEqual(sum(r['fresh_search_releases'].values()),0);self.assertEqual(len(rows),48)
            self.assertEqual(r['completion_reason'],'finite_reachable_proposal_pool_exhausted')
    def test_bounded_time_pass_keeps_postcontrols(self):
        with TemporaryDirectory() as t:
            _,r,rows,calls,_=self.run_mock(Path(t),capture=False,cutoff=True)
            self.assertTrue(r['budget_truncated']);self.assertEqual(r['status'],'completed');self.assertEqual(calls[-1],'controls_post')
    def test_no_search_after_control_or_preservation_loss(self):
        for kwargs in ({'fail_control':True},{'lose_preserved':True}):
            with self.subTest(**kwargs),TemporaryDirectory() as t:
                _,_,_,calls,_=self.run_mock(Path(t),**kwargs);self.assertNotIn('search',calls)
    def test_window_switch_requires_confirmed_primary_and_three_hours(self):
        with TemporaryDirectory() as t:
            _,r,rows,calls,a=self.run_mock(Path(t),window=True)
            p=json.loads((a.output/'window-plans/plan.json').read_text())
            self.assertEqual(p['acquired_elapsed_s'],10801.)
            self.assertEqual(p['selected_after_rows_count'],len([x for x in rows if x['phase']!='controls_post']))
    def test_duplicate_commands_and_release_cap(self):
        from sircus.mjlab import fixed35_sixhour_study as W
        a=dict(id='a',context_id=W.C.TARGET,values=[1]*5,selected_q10=[0]*6,selected_q14=[0]*6)
        b=dict(a,id='b');c=dict(a,id='c',selected_q14=[1]*6);d=dict(a,id='d',selected_q14=[2]*6)
        counts=dict.fromkeys(W.C.CASE_IDS,0);counts[W.C.TARGET]=1199
        keep,skip=W.select_fresh([a,b,c,d],counts,{W.C.command_key(a)})
        self.assertEqual([x['id'] for x in keep],['c'])
        self.assertEqual([x['reason'] for x in skip],['previously_flown_exact_command','previously_flown_exact_command','native_release_limit'])

    def test_window_construction_crossing_cutoff_never_prepares_or_flies_window(self):
        with TemporaryDirectory() as t:
            _,r,rows,calls,a=self.run_mock(Path(t),window=True,window_build_cutoff=True)
            self.assertEqual(r['completion_reason'],'search_time_limit')
            self.assertTrue(r['budget_truncated'])
            self.assertEqual(calls.count('search'),1)
            self.assertEqual(calls[-1],'controls_post')
            self.assertEqual(len(list((a.output/'preparation-requests').glob('*.json'))),1)
            self.assertTrue((a.output/'window-plans/plan.json').exists())
            self.assertFalse(any(r['id'].startswith('fresh-window') for r in rows))

if __name__=='__main__':unittest.main()
