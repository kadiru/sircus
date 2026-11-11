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
    def run_mock(self, root, *, capture=True, decline=False, cutoff=False, fail_control=False, lose_preserved=False):
        from sircus.mjlab import fixed35_broad_study as W
        d=fixture(); parent=root/'parent'; parent.mkdir(); (parent/'native-yaw').mkdir()
        for name in ('calibration.json','verification.json','setup-home.json','geometry-audit.json',
                     'native-yaw/effective-options.json','native-yaw/native-compiled-geometry.json'):
            (parent/name).write_text('{}\n')
        roles=('direct_control','table_control','negative_control'); seed=d['preserved'][0]
        controls=[dict(copy.deepcopy(seed),id='control-'+r,role=r) for r in roles]
        scene={}; assets={}; clock=[0.]; calls=[]
        loaded=dict(d,parent=dict(identity=dict(scene_sha256=W.digest(scene),asset_sha256=assets),packages={}),
            prior={},old_prior={},controls=controls,control_replays=controls,calibration={},home={},bindings={},artifacts={})
        plan=[]
        for c in W.C.CASE_IDS:
            source=next(r for r in d['numeric_sources'] if r['context_id']==c)
            for i in range(180):
                values=list(source['values']);values[4]+=.00001*i
                plan.append(W.C.parent.request(source,identity=f'proposal-{c}-{i}',phase='search',method='broad_search',
                    values=values,proposal_index=i,exact_parent_action=False))
        a=SimpleNamespace(parent_study=parent,original_study=parent,recovery_study=parent,reference=root/'ref',assets=root/'assets',
            output=root/'out',wall_time_limit=43200,search_limit=36000,effective_worker_limit=43200,acquired_monotonic=0.)
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
                if cutoff and phase=='search':clock[0]=36001.
                return result,dict(requested=len(result),wall_s=.01)
        mocks=[patch.object(W.C,'load_inputs',return_value=loaded),patch.object(W.C,'make_plan',return_value=dict(rows=plan)),
            patch.object(W,'load_runtime_reference',return_value=dict(scene=scene,asset_hashes=assets,runtime_reference={})),
            patch.object(W,'source_manifest',return_value={}),patch.object(W,'BroadEngine',side_effect=lambda r,a,p,**kw:Engine(p)),
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

    def test_full_quota_continues_after_captures_and_preserves_exact_repeats(self):
        with TemporaryDirectory() as t:
            W,r,rows,calls,a=self.run_mock(Path(t))
            self.assertEqual(r['status'],'completed');self.assertEqual(r['fresh_search_releases'],W.C.QUOTAS)
            self.assertEqual(len(rows),1075);self.assertEqual(r['stage_counts']['repeat']['attempts'],27)
            self.assertEqual(sum(len(c['confirmed_recipes']) for c in r['cases']),9)
            self.assertTrue(r['preservation']['all_recaptured']);self.assertEqual(calls[-1],'controls_post')
            commits=[json.loads(x) for x in (a.output/'packet-commits.jsonl').read_text().splitlines()]
            self.assertEqual(commits[-1]['rows_count'],1075);self.assertEqual(commits[-1]['rows_sha256'],W.sha(a.output/'rows.jsonl'))

    def test_declines_never_top_up_actual_drop_counts(self):
        with TemporaryDirectory() as t:
            _,r,rows,_,_=self.run_mock(Path(t),decline=True)
            self.assertEqual(r['status'],'incomplete');self.assertEqual(sum(r['fresh_search_releases'].values()),0)
            self.assertEqual(len(rows),48);self.assertEqual(r['preparation_declines'],540)

    def test_time_shortfall_is_honest_and_postcontrols_still_run(self):
        with TemporaryDirectory() as t:
            _,r,rows,calls,_=self.run_mock(Path(t),capture=False,cutoff=True)
            self.assertEqual(r['status'],'incomplete');self.assertTrue(r['budget_truncated'])
            self.assertEqual(sum(r['fresh_search_releases'].values()),24);self.assertEqual(calls[-1],'controls_post')

    def test_failed_controls_or_preservation_prevent_search(self):
        for kw in (dict(fail_control=True),dict(lose_preserved=True)):
            with TemporaryDirectory() as t:
                _,r,rows,calls,_=self.run_mock(Path(t),**kw)
                self.assertNotIn('search',calls);self.assertNotEqual(r['status'],'completed')


if __name__=='__main__':unittest.main()
