import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from sircus.mjlab import guarded_board_fallback_audit as audit
from sircus.mjlab.guarded_board_fallback_plan import RULE,VERSION
from sircus.mjlab.benchmark_verify import Checks,EvidenceError,digest


def base():
    return dict(request=dict(id='checked',values=[.6,0.,.4,28.,.2]),prediction=dict(
        predicted_route='direct',score_m=.002,direct=dict(arm_clearance_m=.003)))


def state():
    return dict(version=VERSION,protocol=copy.deepcopy(RULE),triggered=False,trigger=None,original_completion=None,
        original_interrupted_by_fallback=False,baseline_choice=None,bounds=None,proposals=[],nominal_predictions={},
        wrist_screen={},check_order=[],records=[],admitted=[],predictions={},replacement_decisions=[],
        selected_fallback_candidate_id=None,completed=False)


def completion():
    return dict(returned_naturally=True,return_reason='completed',elapsed_s=1.2,interrupted_by_fallback=False,
                choice_sha256=digest(base()))


class GuardedAuditTests(unittest.TestCase):
    def test_independent_replacement_requires_clearance_and_two_finite_scores(self):
        old=dict(score_m=.002)
        candidate=dict(score_m=.02,predicted_route='direct',direct=dict(arm_clearance_m=.005))
        self.assertTrue(audit.independent_replacement_decision(candidate,old)['accepted'])
        for margin in (.0049,None,math.nan,True):
            value=copy.deepcopy(candidate);value['direct']['arm_clearance_m']=margin
            self.assertFalse(audit.independent_replacement_decision(value,old)['accepted'])
        for score in (.002,None,math.inf,True):
            value=dict(candidate,score_m=score)
            self.assertFalse(audit.independent_replacement_decision(value,old)['accepted'])
        self.assertFalse(audit.independent_replacement_decision(candidate,{'score_m':None})['accepted'])
        value=dict(candidate,predicted_route='table',table=dict(arm_clearance_m=.007))
        self.assertTrue(audit.independent_replacement_decision(value,old)['accepted'])

    def test_natural_completion_is_bound_to_original_choice_and_clock(self):
        value=state();value['original_completion']=completion()
        audit.verify_original_completion(value,base(),{'planning_budget_s':30},Checks(),'test')
        for key,replacement in [('returned_naturally',False),('interrupted_by_fallback',True),('elapsed_s',30.1),('choice_sha256','bad')]:
            changed=copy.deepcopy(value);changed['original_completion'][key]=replacement
            with self.assertRaises(EvidenceError):
                audit.verify_original_completion(changed,base(),{'planning_budget_s':30},Checks(),'test')

    def test_completion_annotation_does_not_inflate_original_publication_count(self):
        first=dict(plan={},generated={},ranked_proposal_ids=[],stalled_reference_snapshot={},choice=base(),
                   active_stage='finished',board_fallback=state())
        second=copy.deepcopy(first);second['board_fallback'].update(original_completion=completion(),completed=True)
        site=dict(board_fallback=second['board_fallback'],accepted_receipts=[{'elapsed_s':1.},{'elapsed_s':1.3}],
                  stalled_reference={'events':[dict(event='query_completed',published_snapshots=1,elapsed_s=1.1)]})
        with patch.object(audit,'_verify_hybrid_chain') as verifier:
            audit.verify_hybrid_events(site,[first,second],Path('.'),{},Checks(),{'planning_budget_s':30},[],[])
            self.assertEqual(verifier.call_args.kwargs['snapshots'],[first])
            for change in ({'event':'reference_path_finished'},{'published_snapshots':2}):
                bad=copy.deepcopy(site);bad['stalled_reference']['events'][-1].update(change)
                with self.assertRaises(EvidenceError):
                    audit.verify_hybrid_events(bad,[first,second],Path('.'),{},Checks(),{'planning_budget_s':30},[],[])

    def test_repeat_ids_and_proofs_cannot_be_changed(self):
        from sircus.mjlab.guarded_board_fallback_study import repeated_requests
        from sircus.mjlab.planning_budget_plan import _wrap_choice
        from sircus.mjlab.tests.test_stalled_reference_study import choice
        original=choice(dict(id='known',bucket_xy_m=[.948,-.098]))
        slot=_wrap_choice(original,'guarded');actual={'guarded':repeated_requests(original,'guarded')}
        audit.verify_repeat_requests([slot],actual,Checks())
        for field,value in [('id','another-shot'),('prepared_connection_sha256','wrong-proof'),('values',[0.]*5)]:
            changed=copy.deepcopy(actual);changed['guarded'][1][field]=value
            with self.assertRaises(EvidenceError):audit.verify_repeat_requests([slot],changed,Checks())

    def test_nontriggered_receipt_never_smuggles_in_geometry_work(self):
        def verify(value):
            return audit.verify_fallback_snapshot({'board_fallback':value,'active_stage':'finished'},base(),
                {'id':'x','bucket_xy_m':[1.,0.]},Path('.'),{}, {}, {},SimpleNamespace(),Checks(),
                {'planning_budget_s':30,'fallback_rule':RULE})
        value=state();self.assertEqual(verify(value),base())
        value.update(original_completion=completion(),completed=True)
        self.assertEqual(verify(value),base())
        value['admitted'].append({'id':'not-checked'})
        with self.assertRaises(EvidenceError):verify(value)


if __name__=='__main__':unittest.main()
