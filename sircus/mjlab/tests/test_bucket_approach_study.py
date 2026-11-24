"""Bounded selection and matched accounting do not use native outcomes."""
import copy
import math
import unittest

from sircus.mjlab.bucket_approach_study import (choose_correction,ordered_proposals,
    schedule,summarize,SEARCH_RULE)
from sircus.mjlab.bucket_approach_audit import expected_slots,independent_summary


def request(identity):
    return dict(id=identity,values=[.6,0.,.4,24.,.2],bucket_xy_m=[1.,0.],
                selected_q10=[0.]*6,selected_q14=[1.]*6)


def plan():
    return dict(cases=[dict(id=f'case{i}',directory=f'native-{i}',original=request(f'old{i}'),corrected=request(f'new{i}')) for i in range(3)],
        preserved_cases=[dict(id=f'keep{i}',directory=f'preserved-{i}',original=request(f'keep{i}')) for i in range(4)],
        diagnostic_cases=[dict(id='robot47',directory='diagnostic-47',original=request('old47'))])


def observed(slot,*,capture=False,invalid=False,robot=0):
    return dict(slot['request'],board_first_score=dict(route_success=capture,invalid_physics=invalid,
        route_class='arm_assisted' if robot else 'table_assisted',robot_contact_steps=robot,bucket_contact_before_entry=True))


class BucketApproachStudyTests(unittest.TestCase):
    def test_matched_three_repeats_preserve_commands(self):
        source=plan();before=copy.deepcopy(source);slots=schedule(source)
        self.assertEqual(slots,expected_slots(source))
        self.assertEqual(source,before)
        self.assertEqual(len(slots),33)
        self.assertEqual(sum(s['variant']=='preserved' for s in slots),12)
        self.assertEqual(sum(s['variant']=='diagnostic_control' for s in slots),3)
        target=[s for s in slots if s['case_id']=='case0']
        self.assertEqual([(s['variant'],s['repeat']) for s in target],
            [('original',0),('corrected',0),('corrected',1),('original',1),('original',2),('corrected',2)])
        target[0]['request']['values'][0]=99
        self.assertEqual(source,before)

    def test_decline_retained_in_requested_denominator(self):
        source=plan();source['cases'][1]['corrected']=None
        slots=schedule(source)
        self.assertEqual(len(slots),33)
        self.assertEqual(sum(s['request'] is not None for s in slots),30)
        rows=[observed(s) for s in slots if s['request'] is not None]
        result=summarize(source,rows)
        self.assertEqual(result,independent_summary(source,rows))
        group=result['cases'][1]['results']['corrected']
        self.assertEqual((group['requested'],group['released'],group['setup_rejected']),(3,0,3))

    def test_risk_ranking_is_not_contact_ban(self):
        candidates=[request('b'),request('a')]
        approach={'a':dict(predicted_score_m=-.002),'b':dict(predicted_score_m=-.002)}
        self.assertEqual(choose_correction(dict(predicted_score_m=-.01),candidates,approach)['id'],'a')
        self.assertIsNone(choose_correction(dict(predicted_score_m=-.001),candidates,approach))
        self.assertIsNone(choose_correction(dict(predicted_score_m=-.002),candidates,approach))

    def test_missing_nonfinite_candidate_score_cannot_win(self):
        rows=[request(x) for x in ('nan','none','inf')]
        scores={x:dict(predicted_score_m=v) for x,v in [('nan',math.nan),('none',None),('inf',math.inf)]}
        self.assertIsNone(choose_correction(dict(predicted_score_m=.01),rows,scores))
        self.assertIsNone(choose_correction({},[],{}))

    def test_checks_are_bounded_and_groups_interleaved(self):
        rows=[dict(id=f'{group}-{i:02d}',group=group) for group in SEARCH_RULE['groups'] for i in range(30)]
        predictions={r['id']:dict(score_m=float(r['id'][-2:])) for r in rows}
        wrist={r['id']:dict(passed=not r['id'].endswith('29')) for r in rows}
        ordered=ordered_proposals(rows,predictions,wrist)
        self.assertEqual(len(ordered),40)
        self.assertEqual([r['group'] for r in ordered[:4]],SEARCH_RULE['groups']*2)
        self.assertTrue(all(not r['id'].endswith('29') for r in ordered))
        self.assertEqual(ordered_proposals([],{},{}),[])

    def test_allowed_robot_capture_and_preservation_failure_reported(self):
        source=plan();slots=schedule(source)
        rows=[observed(s,capture=s['variant']=='corrected',robot=3 if s['variant']=='corrected' else 0) for s in slots]
        result=summarize(source,rows)
        self.assertEqual(result,independent_summary(source,rows))
        self.assertEqual(result['cases'][0]['results']['corrected']['captures'],3)
        self.assertEqual(result['cases'][0]['results']['corrected']['robot_contacts'],3)
        self.assertEqual(result['preservation'][0]['captures'],0)
        self.assertEqual(result['diagnostic_controls'][0]['captures'],0)


if __name__=='__main__':unittest.main()
