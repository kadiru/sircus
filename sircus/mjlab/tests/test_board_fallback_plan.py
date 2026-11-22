import copy
from contextlib import nullcontext
import hashlib
import json
import math
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from sircus.mjlab.board_fallback_plan import (
    RULE, PrefixGuard, PrefixTimeExpired, arm_margin, better_prediction,
    fallback_menu, needs_board_fallback,
)


def choice(margin=.02,score=.01,request=True):
    return dict(request={'id':'checked'} if request else None,prediction=dict(
        score_m=score,predicted_route='direct',direct=dict(arm_clearance_m=margin)))


def source():
    return dict(id='known-checked-command',values=[.6,0.,.45,24.,.2],bucket_xy_m=[1.,0.],
        selected_q10=[0.1]*6,selected_q14=[0.2]*6,intended_route='direct',
        board_heading_offset_deg=0.,board_twist_deg=0.)


class BoardFallbackRuleTests(unittest.TestCase):
    def test_only_risky_checked_shots_trigger(self):
        self.assertFalse(needs_board_fallback(choice(request=False)))
        self.assertFalse(needs_board_fallback(choice(margin=.005)))
        self.assertTrue(needs_board_fallback(choice(margin=.004999)))
        self.assertTrue(needs_board_fallback(choice(margin=-.01)))
        for value in (None,math.nan,math.inf,True):
            self.assertIsNone(arm_margin(choice(margin=value)))
            self.assertTrue(needs_board_fallback(choice(margin=value)))
        table=choice();table['prediction'].update(predicted_route='table',table=dict(arm_clearance_m=-.01))
        self.assertTrue(needs_board_fallback(table))

    def test_bad_rim_alone_does_not_claim_arm_obstruction(self):
        self.assertFalse(needs_board_fallback(choice(margin=.1,score=-.2)))

    def test_strict_improvement_keeps_parent_on_ties_or_invalid_prediction(self):
        old={'score_m':.02}
        for value in (.01,.02,None,math.nan,math.inf,True):
            self.assertFalse(better_prediction({'score_m':value},old))
        self.assertTrue(better_prediction({'score_m':.02001},old))
        self.assertTrue(better_prediction({'score_m':-.1},{'score_m':None}))
        self.assertFalse(better_prediction(None,old))

    def test_menu_is_deterministic_bounded_and_preserves_source(self):
        layout=dict(id='fresh-42',split='test',bucket_xy_m=[1.,0.]);original=source();saved=copy.deepcopy(original)
        low=[.1,-.65,.14,12.,.08];high=[1.3,.25,.65,44.,.55]
        rows=fallback_menu(layout,original,low,high)
        self.assertEqual(rows,fallback_menu(layout,original,low,high))
        self.assertEqual(saved,original)
        keys=set()
        for r in rows:
            v=r['values'];key=(*v,r['board_heading_offset_deg'],r['board_twist_deg'])
            self.assertNotIn(key,keys);keys.add(key)
            self.assertTrue(all(lo-1e-9<=x<=hi+1e-9 for x,lo,hi in zip(v,low,high)))
            self.assertLessEqual(abs(v[0]-.6),.150000001)
            self.assertLessEqual(abs(v[1]),.150000001)
            self.assertGreater(math.dist(v[:2],layout['bucket_xy_m']),.15)
            self.assertLessEqual(abs(r['board_heading_offset_deg']),30.)
            self.assertIn(r['board_twist_deg'],(-180.,0.,180.))
            self.assertEqual(r['seed_q10'],original['selected_q10'])
            self.assertEqual(r['seed_q14'],original['selected_q14'])
        self.assertGreater(len(rows),600)
        self.assertEqual({r['group'] for r in rows},{'position_heading','position_height_tilt_gap'})

    def test_menu_is_target_relative_not_a_known_target_lookup(self):
        a=source();b=source();b['values'][0]+=.1;b['bucket_xy_m'][0]+=.1
        first=fallback_menu(dict(id='a',split='test',bucket_xy_m=a['bucket_xy_m']),a,[-2.]*5,[2.,2.,2.,90.,2.])
        second=fallback_menu(dict(id='b',split='test',bucket_xy_m=b['bucket_xy_m']),b,[-2.]*5,[2.,2.,2.,90.,2.])
        self.assertEqual(len(first),len(second))
        for left,right in zip(first,second):
            self.assertAlmostEqual(right['values'][0]-left['values'][0],.1)
            self.assertEqual(left['values'][1:],right['values'][1:])
            self.assertEqual(left['board_heading_offset_deg'],right['board_heading_offset_deg'])

    def test_no_checked_pose_means_no_geometry_anchor(self):
        row=source();row['values'][:2]=row['bucket_xy_m']
        with self.assertRaises(ValueError):
            fallback_menu(dict(id='x',split='test',bucket_xy_m=row['bucket_xy_m']),row,[-2.]*5,[90.]*5)


class PrefixGuardTests(unittest.TestCase):
    def test_soft_boundary_only_after_risky_snapshot_and_does_not_extend_clock(self):
        now=[10.];g=PrefixGuard(10.,clock=lambda:now[0])
        now[0]=100.;g.check()  # No checked source: original planner owns full deadline.
        g.choice=choice(margin=.1);g.check()
        g.choice=choice(margin=-.1);now[0]=21.999;g.check()
        now[0]=22.
        with self.assertRaises(PrefixTimeExpired):g.check()
        g.close();g.check()

    def test_gate_checks_preserve_rejection_and_are_restored(self):
        class Gate:
            def __init__(self):self.calls=0
            def check(self,*args,**kwargs):self.calls+=1;return dict(passed=False,reason='collision')
        gate=Gate();engine=SimpleNamespace(adapter=SimpleNamespace(resolver=SimpleNamespace(solver=SimpleNamespace(gate=gate))))
        now=[0.];guard=PrefixGuard(0.,clock=lambda:now[0]);old=gate.check
        guard.install(engine);guard.install(engine)
        self.assertEqual(len(guard.installed),1)
        self.assertEqual(gate.check('original-input'),dict(passed=False,reason='collision'))
        guard.choice=choice(margin=-.1);now[0]=12.
        with self.assertRaises(PrefixTimeExpired):gate.check('same-input')
        self.assertEqual(gate.calls,1)
        guard.close();self.assertEqual(gate.check,old)
        self.assertEqual(gate.check(),dict(passed=False,reason='collision'))


class IncrementalFallbackTests(unittest.TestCase):
    def exercise(self,new_score=None,new_margin=.02,source_margin=-.02,clock_after_prefix=1.,
                 setup_finish_time=None,soft_handoff=False):
        import sircus.mjlab as package
        from sircus.mjlab import board_fallback_plan as module
        now=[0.];published=[];checked=[]
        layout=dict(id='fresh-test',split='test',stratum='test',bucket_xy_m=[1.,0.])
        original=source();base=choice(margin=source_margin,score=-.02)
        base.update(id='fresh-test-warm_early',context_id=layout['id'],bucket_xy_m=layout['bucket_xy_m'],
                    method='warm_early',intended_route='direct',available_routes=['direct'],request=original)
        proposal=dict(original,id='fresh-test-fallback-0',context_id=layout['id'],group='position_heading')
        gate=SimpleNamespace(check=lambda *a,**k:dict(passed=True))
        solver=SimpleNamespace(gate=gate,low=[-2.]*5,high=[90.]*5)
        resolver=SimpleNamespace(solver=solver,models={})
        def prepare(rows,**kwargs):
            checked.append(rows[0]['id'])
            if setup_finish_time is not None:now[0]=setup_finish_time
            row=dict(rows[0],id=rows[0]['id']+'-b0',proposal_id=rows[0]['id'])
            record=dict(proposal_id=rows[0]['id'],admitted=1 if new_score is not None else 0)
            return ([row] if new_score is not None else []),[record]
        engine=SimpleNamespace(adapter=SimpleNamespace(resolver=resolver,env=None),prepare=prepare)
        workers=SimpleNamespace(_engine=engine)
        def hybrid_call(layout,method,*args,publish,deadline,profile):
            now[0]=clock_after_prefix
            publish(dict(layout=layout,method=method,active_stage='finished',choice=copy.deepcopy(base),
                completed_proposal_ids=['base-0'],ranked_proposal_ids=['base-0'],generated={'menu':[]},
                plan={k:dict(proposals=[],admitted=[],records=[]) for k in ('base','fallback','neural','branches')},
                stalled_reference_snapshot={'engine_mode':'reference'}))
            if soft_handoff:
                now[0]=12.1
                gate.check()
                raise AssertionError('Soft boundary must interrupt the next exact gate check')
            return 'completed'
        hybrid=SimpleNamespace(_reference_engine=engine,_direct_engine=None,incremental_stalled_reference_plan=hybrid_call)
        recovery=SimpleNamespace(wrist_screen=lambda *a:dict(passed=True),
            nominal_prediction=lambda *a:dict(score_m=.02),diversified=lambda rows,preds,limit:rows[:limit])
        ranker=SimpleNamespace(predict=lambda r:dict(score_m=new_score,predicted_route='direct',direct={'arm_clearance_m':new_margin}))
        clearance=SimpleNamespace(_ranker=lambda config:ranker,
            needs_expansion=lambda c:c.get('prediction',{}).get('score_m',-1)<=0)
        wrapper=SimpleNamespace(ranker=ranker,predict=ranker.predict)
        prefix='sircus.mjlab.'
        modules={prefix+'parallel_capture_grid_workers':workers,prefix+'stalled_reference_plan':hybrid,
                 prefix+'board_pose_recovery_study':recovery,prefix+'clearance_expansion_plan':clearance,
                 prefix+'benchmark_verify':SimpleNamespace(digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest())}
        profile=SimpleNamespace(started=0.,stage=lambda name:nullcontext())
        def publish(row):published.append(copy.deepcopy(row));return True
        guard_class=module.PrefixGuard
        with mock.patch.dict(sys.modules,modules), \
             mock.patch.object(package,'parallel_capture_grid_workers',workers,create=True), \
             mock.patch.object(package,'stalled_reference_plan',hybrid,create=True), \
             mock.patch.object(module.time,'perf_counter',lambda:now[0]), \
             mock.patch.object(module,'PrefixGuard',lambda started:guard_class(started,clock=lambda:now[0])), \
             mock.patch.object(module,'fallback_menu',lambda *a:[copy.deepcopy(proposal)]), \
             mock.patch.object(module,'CachedRanker',lambda r:wrapper):
            reason=module.incremental_board_fallback_plan(layout,'warm_early',{},None,None,{'scene':{}},None,
                publish=publish,deadline=30.,profile=profile)
        return reason,published,checked,base

    def test_clear_primary_bypasses_geometry_search(self):
        reason,rows,checked,base=self.exercise(source_margin=.02)
        self.assertEqual(reason,'completed');self.assertEqual(checked,[])
        self.assertFalse(rows[-1]['board_fallback']['triggered'])
        self.assertEqual(rows[-1]['choice'],base)

    def test_failed_geometry_checks_preserve_complete_parent_choice(self):
        reason,rows,checked,base=self.exercise()
        self.assertEqual(reason,'completed');self.assertTrue(checked)
        self.assertEqual(rows[-1]['active_stage'],'finished')
        self.assertEqual(rows[-1]['choice'],base)
        self.assertTrue(rows[-1]['board_fallback']['triggered'])
        self.assertIsNone(rows[-1]['board_fallback']['selected_fallback_candidate_id'])
        self.assertEqual(rows[-1]['completed_proposal_ids'][0],'base-0')

    def test_better_checked_geometry_publishes_and_returns_early(self):
        reason,rows,checked,base=self.exercise(new_score=.02)
        self.assertEqual(reason,'early_good_shot')
        self.assertEqual(rows[-1]['active_stage'],'early_good_shot')
        self.assertEqual(rows[-1]['choice']['prediction']['score_m'],.02)
        self.assertEqual(rows[-1]['board_fallback']['baseline_choice'],base)
        self.assertEqual(len(rows[-1]['board_fallback']['admitted']),1)

    def test_insufficient_time_keeps_primary_without_second_clock(self):
        reason,rows,checked,base=self.exercise(new_score=.02,clock_after_prefix=29.9)
        self.assertEqual(checked,[]);self.assertEqual(rows[-1]['choice'],base)
        self.assertFalse(rows[-1]['board_fallback']['triggered'])

    def test_setup_finishing_after_absolute_deadline_cannot_replace_published_parent(self):
        _,rows,checked,base=self.exercise(new_score=.03,setup_finish_time=31.)
        self.assertTrue(checked)
        # The owned worker will report deadline instead of accepting a late
        # done message. Its last immutable choice is still the checked parent.
        self.assertEqual(rows[-1]['active_stage'],'board_fallback_menu')
        self.assertEqual(rows[-1]['choice'],base)
        self.assertEqual(rows[-1]['completed_proposal_ids'],['base-0'])
        self.assertEqual(rows[-1]['board_fallback']['admitted'],[])
        self.assertEqual(rows[-1]['board_fallback']['records'],[])
        self.assertIsNone(rows[-1]['board_fallback']['selected_fallback_candidate_id'])

    def test_soft_prefix_deadline_hands_off_without_resetting_absolute_budget(self):
        reason,rows,checked,base=self.exercise(new_score=.02,soft_handoff=True)
        self.assertEqual(reason,'early_good_shot')
        self.assertTrue(checked)
        handoff=next(r for r in rows if r['board_fallback']['triggered'])
        self.assertEqual(handoff['board_fallback']['trigger']['reason'],'risky_prefix_soft_boundary')
        self.assertEqual(handoff['board_fallback']['trigger']['elapsed_s'],12.1)
        self.assertEqual(handoff['board_fallback']['baseline_choice'],base)
        self.assertEqual(rows[0]['choice'],base)
        self.assertEqual(rows[-1]['choice']['prediction']['score_m'],.02)


if __name__=='__main__':unittest.main()
