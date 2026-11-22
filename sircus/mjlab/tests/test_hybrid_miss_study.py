"""Causal-study invariants: fixed targets, one frozen choice, matched requests."""
import copy
import unittest
from sircus.mjlab.hybrid_miss_study import schedule, validate_change, predicted_arm_margin, choose_branch


def request(identity='old'):
    return dict(id=identity, values=[.5,-.3,.4,28.,.3], bucket_xy_m=[1.,0.],
                board_heading_offset_deg=0.,board_twist_deg=0.,
                selected_q10=[0.]*6,selected_q14=[0.]*6)


class CausalPlanTests(unittest.TestCase):
    def test_schedule_keeps_exact_commands_and_interleaves_repeats(self):
        old=request(); new=request('new'); new['selected_q14'][1]=.3
        plan={'cases':[dict(id='case',directory='native-case',original=old,corrected=new)]}
        before=copy.deepcopy(plan); slots=schedule(plan)
        self.assertEqual(plan,before)
        self.assertEqual([s['variant'] for s in slots],['original','corrected','corrected','original','original','corrected'])
        self.assertEqual(len({s['request']['id'] for s in slots}),6)
        for s in slots:
            for key in ('values','bucket_xy_m','selected_q10','selected_q14'):
                self.assertEqual(s['request'][key],plan['cases'][0][s['variant']][key])

    def test_unavailable_correction_stays_in_request_denominator(self):
        slots=schedule({'cases':[dict(id='case',directory='native-case',original=request(),corrected=None)]})
        self.assertEqual(len(slots),6)
        self.assertEqual(sum(s['request'] is None for s in slots),3)

    def test_moving_release_or_dropper_is_not_the_selected_intervention(self):
        for field in ('values','selected_q10','bucket_xy_m'):
            old=request(); new=copy.deepcopy(old); new[field][0]+=.1
            with self.assertRaisesRegex(ValueError,'Only board-arm'):
                validate_change(old,new)

    def test_ranking_uses_arm_clearance_not_capture_labels(self):
        old=request(); a=request('a'); b=request('b')
        a['selected_q14'][0]=.2; b['selected_q14'][0]=.3
        predictions={'a':{'direct':{'arm_clearance_m':.001}},'b':{'direct':{'arm_clearance_m':.02}}}
        a['success']=True; b['success']=False
        self.assertEqual(choose_branch(old,[a,b],predictions,'direct')['id'],'b')
        self.assertEqual(predicted_arm_margin({'direct':{'arm_clearance_m':float('nan')}},'direct'),float('-inf'))


if __name__=='__main__':unittest.main()
