"""Independent audit rejects changed geometry, omitted checks and branch leakage."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch
from sircus.mjlab.benchmark_verify import Checks,EvidenceError,digest
from sircus.mjlab.clear_shot_range_audit import (
    verify_grid,verify_records,verify_admissions,independent_summary,
    board_joint_difference,release_accounting,verify_request_keys,verify_preservation_slots)
from sircus.mjlab.clear_shot_range_plan import proposal_menu
from sircus.mjlab.clear_shot_range_study import summarize,schedule,preservation_schedule,repeat_schedule
from sircus.mjlab.tests.test_clear_shot_range_study import case,prepared,good


def fixture():
    c=case();c['proposals']=proposal_menu(c)
    c.update(bounds=dict(low=[.3,-.5,.12,8.,.05],high=[1.04,.35,.6,60.,.65]),
        preparation_timed_out=False,proposal_timeouts=0,engine_rebuilds=0,
        admitted=[],candidates=[],rejected_branches=[])
    c['records']=[dict(proposal_id=p['id'],proposal_sha256=digest(p),reason='finite_ik_unresolved',
        attempted=True,admitted=0,admitted_request_ids=[]) for p in c['proposals']]
    return c


def admitted(c,index=0):
    p=c['proposals'][index];r=prepared(p)
    r.update(native_prepared=True,command_values=p['values'])
    c['records'][index].update(admitted=1,admitted_request_ids=[r['id']],reason=None)
    c['admitted'].append(r);c['candidates'].append(copy.deepcopy(r))
    return r


class ClearRangeAuditTests(unittest.TestCase):
    def test_independent_grid_detects_position_orientation_seed_or_label_edit(self):
        c=fixture();verify_grid(c,Checks())
        mutations=(lambda p:p['values'].__setitem__(0,p['values'][0]+.01),
                   lambda p:p.__setitem__('board_twist_deg',-135.),
                   lambda p:p['seed_q14'].__setitem__(0,.4),
                   lambda p:p.__setitem__('gap_delta_m',.24),
                   lambda p:p.__setitem__('source_request_id','different'),
                   lambda p:p.__setitem__('proposal_family','tilt_and_gap'))
        for mutation in mutations:
            wrong=copy.deepcopy(c);mutation(wrong['proposals'][0])
            with self.subTest(mutation=mutation),self.assertRaises(EvidenceError):verify_grid(wrong,Checks())
        wrong=copy.deepcopy(c);wrong['proposals'][0],wrong['proposals'][1]=wrong['proposals'][1],wrong['proposals'][0]
        with self.assertRaises(EvidenceError):verify_grid(wrong,Checks())

    def test_all49_records_required_and_budget_skips_cannot_resume(self):
        c=fixture();verify_records(c,Checks())
        c['preparation_timed_out']=True
        for r in c['records'][10:]:r.update(attempted=False,reason='case_budget_not_attempted')
        verify_records(c,Checks())
        wrong=copy.deepcopy(c);wrong['records'].pop()
        with self.assertRaises(EvidenceError):verify_records(wrong,Checks())
        wrong=copy.deepcopy(c);wrong['records'][-1].update(attempted=True,reason='finite_ik_unresolved')
        with self.assertRaises(EvidenceError):verify_records(wrong,Checks())
        wrong=copy.deepcopy(c);wrong['records'][0]['proposal_sha256']='0'*64
        with self.assertRaises(EvidenceError):verify_records(wrong,Checks())

    def test_timeouts_cannot_admit_and_rebuild_count_must_match(self):
        c=fixture();r=c['records'][0]
        r.update(reason='proposal_preparation_time_limit',timing=dict(budget_s=30,wall_s=30.1))
        c.update(proposal_timeouts=1,engine_rebuilds=1)
        verify_records(c,Checks())
        for mutation in (lambda x:x['records'][0].update(admitted=1,admitted_request_ids=['x']),
                         lambda x:x.update(engine_rebuilds=0),
                         lambda x:x['records'][0]['timing'].update(budget_s=31)):
            wrong=copy.deepcopy(c);mutation(wrong)
            with self.assertRaises(EvidenceError):verify_records(wrong,Checks())

    def test_gap_only_branch_guard_and_rejected_proofs_are_independent(self):
        c=fixture();r=admitted(c)
        r['selected_q14'][0]+=2*math.pi;c['candidates']=copy.deepcopy(c['admitted'])
        with patch('sircus.mjlab.direct_path_first_audit._proof',return_value={}) as proof,patch('sircus.mjlab.direct_path_first_audit.verify_seed'):
            verify_admissions(c,Path('.'),{}, {},{},Checks());self.assertEqual(proof.call_count,1)
            wrong=copy.deepcopy(c);wrong['admitted'][0]['selected_q14'][0]+=.01;wrong['candidates']=copy.deepcopy(wrong['admitted'])
            with self.assertRaises(EvidenceError):verify_admissions(wrong,Path('.'),{},{},{},Checks())
            changed=wrong['admitted'].pop();wrong['candidates']=[]
            wrong['rejected_branches']=[dict(request=changed,board_joint_change_rad=board_joint_difference(changed,wrong['original']),reason='fixed_board_joint_branch_changed')]
            wrong['records'][0].update(admitted=0,admitted_request_ids=[],rejected_request_ids=[changed['id']],reason='fixed_board_joint_branch_changed')
            verify_admissions(wrong,Path('.'),{},{},{},Checks())
            wrong['rejected_branches'][0]['board_joint_change_rad']=0
            with self.assertRaises(EvidenceError):verify_admissions(wrong,Path('.'),{},{},{},Checks())

    def test_tilt_change_may_change_branch_but_second_branch_is_not_allowed(self):
        c=fixture();r=admitted(c,9);r['selected_q14'][0]+=.1;c['candidates']=copy.deepcopy(c['admitted'])
        with patch('sircus.mjlab.direct_path_first_audit._proof',return_value={}),patch('sircus.mjlab.direct_path_first_audit.verify_seed'):
            verify_admissions(c,Path('.'),{},{},{},Checks())
            extra=copy.deepcopy(r);extra['id']=extra['id'][:-1]+'1';c['admitted'].append(extra);c['candidates'].append(extra)
            c['records'][9].update(admitted=2,admitted_request_ids=[r['id'],extra['id']])
            with self.assertRaises(EvidenceError):verify_admissions(c,Path('.'),{},{},{},Checks())

    def test_grid_summary_separates_direct_table_robot_capture_and_unattempted(self):
        c=fixture();a=admitted(c,0);b=admitted(c,9);robot=admitted(c,10)
        for r in c['records'][20:]:r.update(attempted=False,reason='case_budget_not_attempted')
        rows=[good(a),dict(good(b),shot_route='table'),dict(good(robot),route_success=False,shot_route=None,
            experimental_route=dict(robot_contact_steps=2,robot_assisted_capture=True))]
        rows += [good(a,'repeat',i) for i in (1,2,3)]
        rows += [dict(good(b,'repeat',i),shot_route='table',route_success=i!=3) for i in (1,2,3)]
        plan=dict(cases=[c]);out=independent_summary(plan,rows)[0]
        self.assertEqual(out,summarize(plan,rows)[0])
        self.assertEqual(out['results']['search']['captures'],2)
        self.assertEqual(out['results']['search']['robot_contacts'],1)
        self.assertEqual(out['results']['repeat']['captures'],5)
        self.assertEqual(len(out['confirmed_settings']),1)
        self.assertEqual(out['grid'][10]['initial_outcome'],'arm_assisted_capture')
        self.assertFalse(out['grid'][10]['initial_capture'])
        self.assertEqual(out['grid'][20]['setup_status'],'not_checked')

    def test_full_budget_preservation_and_no_repeating_failed_search(self):
        c=fixture()
        for i in range(49):admitted(c,i)
        keep=[dict(id='saved-'+str(i),directory='preserved-'+str(i),original=copy.deepcopy(c['original'])) for i in range(3)]
        for i,item in enumerate(keep):item['original']['id']='saved-source-'+str(i)
        plan=dict(cases=[c],preserved_cases=keep)
        initial=schedule(plan);preservation=preservation_schedule(plan)
        verify_request_keys(plan,initial,Checks());verify_preservation_slots(plan,preservation,Checks())
        repeats=repeat_schedule(plan,[good(r) for r in c['candidates']])
        verify_request_keys(plan,repeats,Checks(),confirmations=True,eligible={r['id'] for r in c['candidates']})
        self.assertEqual(release_accounting(len(initial),len(repeats),len(preservation)),dict(progress=dict(completed=199,planned=199),native_releases=226))
        with self.assertRaises(EvidenceError):verify_request_keys(plan,repeats,Checks(),confirmations=True,eligible=set())
        wrong=copy.deepcopy(preservation);wrong[0]['request']['values'][4]+=.001
        with self.assertRaises(EvidenceError):verify_preservation_slots(plan,wrong,Checks())


if __name__=='__main__':unittest.main()
