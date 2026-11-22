"""Frozen development search and exact-repeat invariants."""
import copy
import unittest
import numpy as np
from sircus.mjlab.board_pose_recovery_study import (
    MAX_CANDIDATES,SEARCH_RULE,diversified,geometry_key,make_slot,proposal_menu,
    repeat_schedule,schedule,successful,summarize,
)


def request(identity='old'):
    return dict(id=identity,values=[.60,-.04,.40,28.,.25],bucket_xy_m=[1.05,.025],
                board_heading_offset_deg=0.,board_twist_deg=0.,selected_q10=[0.]*6,selected_q14=[0.]*6,
                intended_route='direct',prepared_connection_file='proof.json',prepared_connection_sha256='a'*64)


def case():
    old=request(); candidate=request('new');candidate['values'][0]+=.1
    return dict(id='stalled-reference-test-03',original=old,candidates=[candidate],directory='native-03',
                proposals=[candidate],records=[{'proposal_id':'new'}],admitted=[candidate],predictions={'new':{'score_m':.01}})


def observed(candidate='new',variant='search',repeat=0,success=True,invalid=False):
    return dict(id=f'{candidate}-{variant}-{repeat}',candidate_id=candidate,context_id='stalled-reference-test-03',
                variant=variant,repeat=repeat,native_release=True,invalid_physics=invalid,route_success=success,
                shot_route='direct' if success else None,experimental_route=dict(robot_contact_steps=0,robot_assisted_capture=False))


class RecoveryPlanTests(unittest.TestCase):
    def test_menu_is_deterministic_unique_bounded_and_preserves_bucket(self):
        c=case();before=copy.deepcopy(c)
        low=np.array([.32,-.3,.18,12.,.065]);high=np.array([1.01,.25,.53,48.,.52])
        a=proposal_menu(c,low,high);b=proposal_menu(c,low,high)
        self.assertEqual(a,b);self.assertEqual(c,before)
        self.assertEqual(len(a),len({geometry_key(r) for r in a}))
        self.assertLessEqual(len(a),686+4096)
        for r in a:
            self.assertTrue(np.all(np.asarray(r['values'])>=low-1e-9))
            self.assertTrue(np.all(np.asarray(r['values'])<=high+1e-9))
            self.assertEqual(r['bucket_xy_m'],c['original']['bucket_xy_m'])
            self.assertGreater(np.linalg.norm(np.asarray(r['bucket_xy_m'])-r['values'][:2]),.15)
        self.assertEqual({r['group'] for r in a},{'position_heading','position_height_tilt_gap'})
        self.assertEqual({r['board_twist_deg'] for r in a},{0.,180.})

    def test_initial_schedule_exact_commands_unique_ids_and_three_originals(self):
        c=case();before=copy.deepcopy(c);slots=schedule({'cases':[c]})
        self.assertEqual(c,before);self.assertEqual(len(slots),4)
        self.assertEqual([s['variant'] for s in slots],['original']*3+['search'])
        self.assertEqual(len({s['request']['id'] for s in slots}),4)
        for s in slots:
            src=c['original'] if s['variant']=='original' else c['candidates'][0]
            for k in ('values','bucket_xy_m','selected_q10','selected_q14','prepared_connection_file','prepared_connection_sha256'):
                self.assertEqual(s['request'][k],src[k])

    def test_repeat_schedule_requires_initial_valid_standard_capture(self):
        plan={'cases':[case()]}
        for row in (observed(success=False),observed(invalid=True),observed(variant='repeat')):
            self.assertEqual(repeat_schedule(plan,[row]),[])
        robot=observed(success=False);robot['experimental_route']['robot_assisted_capture']=True
        self.assertEqual(repeat_schedule(plan,[robot]),[])
        slots=repeat_schedule(plan,[observed()])
        self.assertEqual([s['repeat'] for s in slots],[1,2,3])
        self.assertTrue(all(s['variant']=='repeat' for s in slots))
        self.assertEqual(len({s['request']['id'] for s in slots}),3)

    def test_repeat_keeps_exact_request_and_is_not_affected_by_later_outcome(self):
        plan={'cases':[case()]};rows=[observed()]
        first=repeat_schedule(plan,rows)
        self.assertEqual(first,repeat_schedule(plan,rows+[observed(variant='repeat',repeat=1,success=False)]))
        source=plan['cases'][0]['candidates'][0]
        for slot in first:
            for key in source:
                if key!='id':self.assertEqual(slot['request'][key],source[key])

    def test_diversification_prefers_clearance_but_retains_distinct_geometry(self):
        a=request('a');b=request('b');c=request('c');b['values'][0]+=.0001;c['values'][0]+=.1
        predictions={'a':{'score_m':.1},'b':{'score_m':.09},'c':{'score_m':.08}}
        self.assertEqual([r['id'] for r in diversified([b,c,a],predictions,2)],['a','c'])
        self.assertEqual([r['id'] for r in diversified([b,c,a],predictions,3)],['a','c','b'])

    def test_identical_geometry_uses_best_scored_joint_branch_once(self):
        a=request('a');b=request('b');b['selected_q14'][2]=.5
        self.assertEqual([r['id'] for r in diversified([a,b],{'a':{'score_m':-.02},'b':{'score_m':.01}},40)],['b'])

    def test_confirmation_is_three_fresh_successes_not_one_search_capture(self):
        plan={'cases':[case()]};rows=[observed()]
        report=summarize(plan,rows)[0]
        self.assertEqual(report['confirmed_settings'],[])
        self.assertEqual(report['results']['repeat']['requested'],3)
        rows.extend(observed(variant='repeat',repeat=i) for i in range(1,4))
        report=summarize(plan,rows)[0]
        self.assertEqual(len(report['confirmed_settings']),1)
        self.assertEqual(report['results']['search']['captures'],1)
        self.assertEqual(report['results']['repeat']['captures'],3)
        rows[-1]['route_success']=False;rows[-1]['shot_route']=None
        self.assertEqual(summarize(plan,rows)[0]['confirmed_settings'],[])

    def test_worst_case_budget_is_bounded(self):
        self.assertEqual(MAX_CANDIDATES,40)
        self.assertEqual(3*3+3*MAX_CANDIDATES+3*MAX_CANDIDATES*3+18,507)
        self.assertEqual(SEARCH_RULE['repeats_per_success'],3)


if __name__=='__main__':unittest.main()
