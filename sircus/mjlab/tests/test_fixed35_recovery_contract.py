"""Bounded posthoc plans, immutable originals and three fresh confirmations."""
import copy
from pathlib import Path
import unittest
import numpy as np
from sircus.mjlab import fixed35_recovery_contract as C
from sircus.mjlab.tests.test_corrected_imitation_contract import fixture as old_fixture, score_fields


def fixture():
    source,observed=old_fixture();source.update(copy.deepcopy(observed[0]))
    failures={(0,0),(1,5),(1,6),(2,3),(2,6),(3,0),(3,4),(3,5),(3,6),(4,4),(4,6)}
    current=[];refit=[]
    for grid in C.fixed_grid():
        col,row=grid['grid_cell'];success=(row,col) not in failures
        r=dict(copy.deepcopy(source),id='parent-'+grid['id'],context_id=grid['id'],bucket_xy_m=grid['bucket_xy_m'],
            method='current_bc',phase='test',role='test',values=[.6,-.12,.40,28.,.24],q10=[.1]*6,q14=[.2]*6,
            connection_file='fixed-yaw-connections/'+grid['id']+'.json',connection_sha256='c'*64,**score_fields(success))
        r.update(route_success=success,native_release=(row,col) not in {(3,0),(3,6),(4,6)},invalid=None)
        if not r['native_release']:r['invalid']='finite_ik_unresolved' if row==4 else 'fixed_base_clearance_rejected'
        current.append(r)
        a=copy.deepcopy(r);a.update(id='refit-'+grid['id'],method='fixed35_bc')
        a['values'][3]=28.1
        if (row,col)==(1,6):a.update(route_success=True,native_release=True,invalid=None)
        refit.append(a)
    preserved=[r for r in current if C.captured(r)];failed=[r for r in current if not C.captured(r)]
    blocked=[r for r in failed if r.get('invalid')=='fixed_base_clearance_rejected'];cases=[r for r in failed if r not in blocked]
    return dict(current=current,refit=refit,preserved=preserved,cases=cases,blocked=blocked,grid=C.fixed_grid())


class ContractTests(unittest.TestCase):
    def test_literal_counts_and_unchanged_bucket_grid(self):
        data=fixture();before=copy.deepcopy(data)
        plan=C.make_plan(data['cases'],data['refit'],data['preserved'])
        self.assertEqual(len(plan),720);self.assertEqual(len({r['id'] for r in plan}),720)
        self.assertEqual(data,before)
        for index in range(80):
            group=plan[index*9:(index+1)*9]
            self.assertEqual([r['context_id'] for r in group],[r['context_id'] for r in data['cases']])
            self.assertTrue(all(r['proposal_index']==index for r in group))
            for r,p in zip(group,data['cases']):
                self.assertEqual(r['bucket_xy_m'],p['bucket_xy_m']);self.assertEqual(r['parent_row_sha256'],C.digest(p))
                self.assertEqual(r['values'],r['command_values']);self.assertFalse(r['selection_or_learning'])
                self.assertEqual(r['perturbation_mixed_units'],[0.]*5)
                self.assertTrue((np.asarray(r['values'])>=C.LOW).all() and (np.asarray(r['values'])<=C.HIGH).all())
        self.assertEqual(C.protocol()['maximum_requests'],720+81+24+18)

    def test_known_other_policy_capture_is_first_alternative(self):
        data=fixture();parent=next(r for r in data['cases'] if r['context_id'].endswith('01-06'))
        alternatives=C.proposals(parent,data['refit'],data['preserved'])
        wanted=next(r for r in data['refit'] if r['context_id']==parent['context_id'])
        self.assertEqual(alternatives[1]['values'],wanted['values'])
        self.assertEqual(alternatives[1]['kind'],'exact_historical_refit_capture')
        self.assertEqual(alternatives[1]['parent_source_ids'],[wanted['id']])
        self.assertEqual(alternatives[0]['values'],parent['values'])

    def test_all_axes_and_coupled_support_are_present_and_clipping_is_explicit(self):
        d=fixture();source=copy.deepcopy(d['cases'][0]);source['values']=[.31,-.49,.13,9.,.06]
        p=C.proposals(source,d['refit'],d['preserved'])
        self.assertEqual([r['kind'] for r in p].count('gap_axis'),12)
        self.assertEqual([r['kind'] for r in p].count('tilt_axis'),12)
        self.assertEqual([r['kind'] for r in p].count('height_axis'),10)
        self.assertEqual([r['kind'] for r in p].count('coupled_tilt_gap'),16)
        self.assertEqual([r['kind'] for r in p].count('bounce_plane_xy'),12)
        self.assertEqual([r['kind'] for r in p].count('coarse_height_tilt_gap'),12)
        clipped=[r for r in p if r['clipped']];self.assertTrue(clipped)
        self.assertTrue(all(r['raw_values']!=r['values'] for r in clipped))

    def test_preserved_replay_exact_and_source_immutable(self):
        data=fixture();plan=C.preserved_plan(data['preserved'])
        self.assertEqual(len(plan),24)
        for r,p in zip(plan,data['preserved']):
            self.assertEqual(r['values'],p['values']);self.assertTrue(r['exact_parent_action'])
            self.assertEqual(r['parent_row_sha256'],C.digest(p));self.assertNotIn('route_success',r)
        plan[0]['values'][0]=999
        self.assertNotEqual(data['preserved'][0]['values'][0],999)

    def test_confirmation_requires_three_fresh_passes_and_retains_lineage(self):
        d=fixture();p=C.make_plan(d['cases'],d['refit'],d['preserved'])[0]
        discovery=dict(p,**score_fields(True),connection_sha256='d'*64,q10=[1.]*6,q14=[2.]*6)
        repeats=C.repeat_plan(discovery,0)
        self.assertEqual(len(repeats),3)
        for i,r in enumerate(repeats):
            self.assertEqual(r['repeat'],i);self.assertEqual(r['selected_from_sha256'],C.digest(discovery))
            self.assertEqual(r['parent_row_id'],p['parent_row_id']);self.assertNotIn('route_success',r)
            self.assertEqual(r['selected_q10'],discovery['q10'])
        observed=[dict(r,**score_fields(True)) for r in repeats]
        self.assertEqual(C.confirmation_groups(observed,discovery['context_id'])[0]['status'],'confirmed')
        observed[-1]['route_success']=False
        self.assertEqual(C.confirmation_groups(observed,discovery['context_id'])[0]['status'],'failed')
        self.assertEqual(C.confirmation_groups(observed[:2],discovery['context_id'])[0]['status'],'incomplete')
        with self.assertRaises(ValueError):C.repeat_plan(dict(discovery,route_success=False),0)
        with self.assertRaises(ValueError):C.repeat_plan(discovery,3)

    def test_failed_repeats_do_not_enter_confirmed_recipe_and_budget_remains_explicit(self):
        d=fixture();p=C.make_plan(d['cases'],d['refit'],d['preserved'])[0]
        discovery=dict(p,**score_fields(True),connection_sha256='d'*64,q10=[1.]*6,q14=[2.]*6)
        repeats=[dict(r,**score_fields(False)) for r in C.repeat_plan(discovery,0)]
        results=C.case_results([discovery]+repeats,d['cases'],d['blocked'],finished=True,budget_truncated=True)
        first=next(r for r in results if r['context_id']==p['context_id'])
        self.assertIsNone(first['confirmed_recipe']);self.assertEqual(first['status'],'budget_truncated')
        self.assertEqual(sum(r['status']=='fixed_base_blocked' for r in results),2)

    def test_checked_in_rule_matches_contract(self):
        rule=C.read(Path(C.__file__).with_name('fixed35_recovery_rule.json'))
        self.assertEqual(rule['protocol'],C.protocol());self.assertFalse(rule['protocol']['training'])
        self.assertFalse(rule['protocol']['bucket_perturbations'])

if __name__=='__main__':unittest.main()
