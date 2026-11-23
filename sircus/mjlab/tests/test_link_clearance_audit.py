"""Evidence-auditor regression guards: branch identity, preservation, denominators."""
import copy
import math
import unittest

from sircus.mjlab.benchmark_verify import Checks
from sircus.mjlab.link_clearance_audit import (branch_change,equivalent_commands,
    independent_candidates,independent_order,release_accounting,verify_preservation_slots,verify_preparation_records)


def command(name='shot'):
    return dict(id=name,values=[.6,0.,.4,25.,.2],bucket_xy_m=[1.1,0.],
        board_heading_offset_deg=0.,board_twist_deg=0.,selected_q10=[0.]*6,selected_q14=[0.]*6,
        prepared_connection_file='proof.json',prepared_connection_sha256='hash')


class AuditTests(unittest.TestCase):
    def test_angular_wrap_is_not_new_arm_configuration(self):
        source=command();other=copy.deepcopy(source);other['selected_q14'][0]=2*math.pi
        self.assertAlmostEqual(branch_change(other,source)['maximum_rad'],0.)
        self.assertTrue(equivalent_commands(source,other))
        other['selected_q14'][0]+=.011
        self.assertFalse(equivalent_commands(source,other))

    def test_fixed_geometry_candidates_need_real_joint_change(self):
        source=command();row=dict(command('a'),source_request_id='shot',source_bank_id='shot',fixed_cartesian=True,proposal_family='fixed_cartesian')
        case=dict(sources=[dict(id='shot',source_request=source)],admitted=[row],predictions={'a':dict(score_m=.05)})
        self.assertEqual(independent_candidates(case),[])
        row['selected_q14'][1]=.15
        selected=independent_candidates(case)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['joint_change']['proximal_rad'],.15)

    def test_fixed_branch_preference_is_bouncer_proximal_not_large_wrist_flip(self):
        source=command();rows=[]
        for name,arm,joint,value in [('wrist','q14',4,3.),('dropper','q10',1,2.),('bouncer','q14',1,.5)]:
            row=dict(command(name),source_request_id='shot',source_bank_id='shot',fixed_cartesian=True,proposal_family='fixed_cartesian')
            row['selected_'+arm][joint]=value;rows.append(row)
        case=dict(sources=[dict(id='shot',source_request=source)],admitted=rows,predictions={r['id']:dict(score_m=.01) for r in rows})
        self.assertEqual([r['id'] for r in independent_candidates(case)],['bouncer','dropper','wrist'])

    def test_preserved_recipe_cannot_change_or_skip_repeats(self):
        cases=[dict(id=str(i),original=command(str(i)),directory='preserved-'+str(i)) for i in range(3)]
        slots=[]
        for case in cases:
            for repeat in range(3):
                request=dict(copy.deepcopy(case['original']),id=f"{case['id']}-{repeat}",context_id=case['id'],repeat=repeat,
                    candidate_id=case['original']['id'],variant='preserved',phase='preservation')
                slots.append(dict(request=request,directory=case['directory']))
        verify_preservation_slots(dict(preserved_cases=cases),slots,Checks())
        with self.assertRaises(ValueError):verify_preservation_slots(dict(preserved_cases=cases),slots[:-1],Checks())
        changed=copy.deepcopy(slots);changed[0]['request']['values'][4]+=.01
        with self.assertRaises(ValueError):verify_preservation_slots(dict(preserved_cases=cases),changed,Checks())
        duplicate=copy.deepcopy(slots);duplicate[-1]=duplicate[0]
        with self.assertRaises(ValueError):verify_preservation_slots(dict(preserved_cases=cases),duplicate,Checks())

    def test_native_budget_includes_preservation_and_controls(self):
        self.assertEqual(release_accounting(70,192,9),dict(progress=dict(completed=262,planned=262),native_releases=289))
        self.assertEqual(release_accounting(70,0,9)['native_releases'],97)
        with self.assertRaises(ValueError):release_accounting(70,0,0)
        with self.assertRaises(ValueError):release_accounting(70,-1,9)

    def test_geometry_families_are_checked_before_second_fixed_branch(self):
        proposals=[]
        for seed in range(2):
            for family in ('fixed_cartesian','radial_shift','lateral_shift','board_twist'):
                for sample in range(2):
                    proposals.append(dict(id=f'{seed}-{family}-{sample}',seed_index=seed,proposal_family=family,fixed_cartesian=family=='fixed_cartesian'))
        case=dict(proposals=proposals,wrist_reach={p['id']:dict(passed=True) for p in proposals},
            nominal_predictions={p['id']:dict(score_m=0.) for p in proposals})
        order=independent_order(case)
        self.assertEqual(order[:4],['0-fixed_cartesian-0','0-radial_shift-0','0-lateral_shift-0','0-board_twist-0'])
        self.assertEqual(order[8],'0-fixed_cartesian-1')

    def test_proposal_timeout_can_continue_but_case_timeout_is_terminal(self):
        def record(name,reason):return dict(proposal_id=name,reason=reason,admitted=0,admitted_request_ids=[])
        case=dict(records=[record('a','proposal_preparation_time_limit'),record('b','finite_ik_unresolved')],preparation_timed_out=False)
        verify_preparation_records(case,['a','b'],Checks())
        case['records'][0]['admitted']=1
        with self.assertRaises(ValueError):verify_preparation_records(case,['a','b'],Checks())
        case=dict(records=[record('a','case_preparation_time_limit'),record('b','finite_ik_unresolved')],preparation_timed_out=True)
        with self.assertRaises(ValueError):verify_preparation_records(case,['a','b'],Checks())
        case['records']=case['records'][:1]
        verify_preparation_records(case,['a','b'],Checks())


if __name__=='__main__':unittest.main()
