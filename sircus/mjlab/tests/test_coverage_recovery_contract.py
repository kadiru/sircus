import copy
import unittest
import numpy as np
from sircus.mjlab import coverage_recovery_contract as C


def score_fields(capture=True):
    return dict(scoring_version='sircus-table-assisted-native-capture-v3',route_version='sircus-table-assisted-board-route-v3',
        reward_version='sircus-table-assisted-shot-reward-v3',board_bounce=True,closest_opening_distance_m=.01,
        forbidden_contact_steps=0,max_retention_s=1. if capture else 0.,route_success=capture,route_quality=2 if capture else 0,
        single_board_bounce_success=capture,native_release=True,invalid=None)


def fixture_cases():
    anchor=dict(id='anchor',bucket_xy_m=[.7,-.2],values=[.5,-.2,.3,30.,.2],q10=[0.]*6,q14=[0.]*6,source_record_sha256='a'*64)
    meta=dict(source_hint=anchor,effective_source_hint=anchor,source_teacher_id='anchor',context_resolution_file='r.json',
        context_resolution_sha256='b'*64,context_home_sha256='c'*64,context_layout_eligible=True,context_anchor_admitted=True,context_hard_reject=False,observation=[0.]*9)
    return [dict(meta,id=f'parent-{i:02d}',context_id=f'case-{i:02d}',bucket_xy_m=[.7+i*.001,-.2],
        values=[.504,-.196,.305,30.8,.212],normalized_residual=[.1]*5,native_release=i>=19,role='test',phase='test',method='selected_ppo',
        route_success=False,invalid=None if i>=19 else 'finite_ik_unresolved') for i in range(39)]


class ProtocolTests(unittest.TestCase):
    def test_exact_budget_and_round_robin(self):
        cases=fixture_cases();plan=C.make_plan(cases,cases)
        self.assertEqual(len(plan),624);self.assertEqual([r['context_id'] for r in plan],[c['context_id'] for _ in range(16) for c in cases])
        self.assertEqual([r['proposal_index'] for r in plan],[i for i in range(16) for _ in cases])
        self.assertEqual(C.protocol()['maximum_requests'],624+78+18+280)
        self.assertEqual(C.protocol()['critic_steps'],0)
        self.assertTrue(all(r['role']=='train' and r['original_role']=='test' and r['development_reclassified'] for r in plan))
        self.assertTrue(all(np.max(np.abs(r['normalized_residual']))<1 for r in plan))

    def test_literal_setup_height_tilt_grid_and_flight_gap_grid(self):
        setup,flight=fixture_cases()[0],fixture_cases()[-1]
        got=[C.proposal(setup,i,[])[0][2:4].tolist() for i in range(2,11)]
        self.assertEqual(got,[[h,t] for h in (-.8,0.,.8) for t in (-.75,0.,.75)])
        self.assertEqual([C.proposal(flight,i,[])[0][4] for i in range(2,8)],[-.9,-.6,-.3,.3,.6,.9])
        self.assertEqual(C.proposal(flight,15,[])[0].tolist(),[.1,.1,0.,-.5,.75])

    def test_alternate_capture_requires_same_original_anchor_and_observation(self):
        source=fixture_cases()[0];other=copy.deepcopy(source)
        other.update(score_fields());other.update(id='other',normalized_residual=[.2]*5)
        r,meta=C.proposal(source,1,[other]);self.assertEqual(r.tolist(),[.2]*5);self.assertEqual(meta['parent_source_ids'],['other'])
        other['observation']=[1.]*9
        with self.assertRaisesRegex(ValueError,'same|exact original'):C.proposal(source,1,[other])

    def test_no_action_projection_at_absolute_bounds(self):
        case=fixture_cases()[0];case=copy.deepcopy(case);case['effective_source_hint']['values'][4]=.055
        r,_=C.proposal(case,0,[]);self.assertEqual(r.tolist(),[.1]*5)
        # A within-policy-support request may still fail native action bounds.
        flight=dict(case,native_release=True);r,_=C.proposal(flight,2,[])
        self.assertLess(C.action_from_residual(flight['effective_source_hint']['values'],r)[4],.05)
        self.assertEqual(r[4],-.9)

    def test_only_two_fresh_captures_qualify_teacher(self):
        cases=fixture_cases();plan=C.make_plan(cases,cases);row=dict(plan[0],**score_fields())
        selected,repeats=C.repeat_plan([row],cases);self.assertEqual(len(repeats),2)
        self.assertEqual(C.confirmed_teachers([row],[],selected),[])
        actual=[dict(r,native_release=True,route_success=True,route_quality=2) for r in repeats]
        self.assertEqual(C.confirmed_teachers([row],actual[:1],selected),[])
        self.assertEqual(len(C.confirmed_teachers([row],actual,selected)),1)
        actual[1]['route_success']=False;self.assertEqual(C.confirmed_teachers([row],actual,selected),[])

    def test_fresh_seventy_positions_stratified_disjoint_and_deterministic(self):
        excluded=[[.1,-.65],[.4,-.3],[1.2,.2]];first=C.test_split(excluded);self.assertEqual(first,C.test_split(excluded))
        self.assertEqual(len(first),70);self.assertEqual(len({tuple(r['bucket_xy_m']) for r in first}),70)
        self.assertEqual(sorted(sum(r['grid_cell']==[x,y] for r in first) for y in range(5) for x in range(7)),[2]*35)
        allxy=np.asarray(excluded+[r['bucket_xy_m'] for r in first]);d=np.linalg.norm(allxy[:,None]-allxy[None,:],axis=-1);np.fill_diagonal(d,np.inf)
        self.assertGreaterEqual(d.min(),.005)


if __name__=='__main__':unittest.main()
