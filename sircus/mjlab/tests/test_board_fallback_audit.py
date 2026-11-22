"""Independent prefix and outcome-blind geometry takeover receipt checks."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import board_fallback_audit as audit
from sircus.mjlab.board_fallback_plan import RULE,VERSION
from sircus.mjlab.benchmark_verify import Checks,EvidenceError,digest


def base():
    return dict(request=dict(id='checked',values=[.6,0.,.4,28.,.2]),
                prediction=dict(predicted_route='direct',score_m=-.02,direct=dict(arm_clearance_m=-.02)))


def empty():
    return dict(version=VERSION,protocol=copy.deepcopy(RULE),triggered=False,trigger=None,baseline_choice=None,bounds=None,
        proposals=[],nominal_predictions={},wrist_screen={},check_order=[],records=[],admitted=[],predictions={},
        selected_fallback_candidate_id=None,completed=False)


class BoardFallbackAuditTests(unittest.TestCase):
    def verify(self,state,original=None):
        original=base() if original is None else original
        return audit.verify_fallback_snapshot(dict(board_fallback=state,active_stage='board_fallback_triggered'),original,
            dict(id='test',bucket_xy_m=[1.,0.]),Path('.'),{}, {}, {},SimpleNamespace(),Checks(),
            dict(fallback_rule=RULE,planning_budget_s=30.,scene={}))

    def triggered(self):
        state=empty();original=base()
        state.update(triggered=True,baseline_choice=original,trigger=dict(reason='risky_prefix_soft_boundary',elapsed_s=12.1,
            arm_clearance_m=-.02,baseline_choice_sha256=digest(original),selection_uses_flight_labels=False))
        return state

    def test_pure_trigger_preserves_checked_prefix(self):
        self.assertEqual(self.verify(empty()),base())
        self.assertEqual(self.verify(self.triggered()),base())

    def test_trigger_cannot_use_future_outcomes_or_extra_time(self):
        for field,value in [('selection_uses_flight_labels',True),('elapsed_s',30.),('elapsed_s',11.9)]:
            state=self.triggered();state['trigger'][field]=value
            with self.assertRaises(EvidenceError):self.verify(state)

    def test_beneficial_prefix_cannot_be_replaced_by_unchecked_geometry(self):
        state=empty();state['selected_fallback_candidate_id']='unproved'
        with self.assertRaises(EvidenceError):self.verify(state)
        state=self.triggered();state['baseline_choice']['request']['values'][2]+=.01
        with self.assertRaises(EvidenceError):self.verify(state)

    def test_injected_geometry_before_frozen_bounds_is_rejected(self):
        state=self.triggered();state['proposals']=[{'id':'undeclared'}]
        with self.assertRaises(EvidenceError):self.verify(state)
        state=self.triggered();state['bounds']=dict(low=[-999.]*5,high=[999.]*5)
        with self.assertRaises(EvidenceError):self.verify(state)

    def test_hybrid_prefix_cannot_be_rewritten_after_takeover(self):
        initial=dict(plan={'frozen':'stage'},generated={'menu':[]},ranked_proposal_ids=[],choice=base(),
            stalled_reference_snapshot={'mode':'reference'},board_fallback=empty())
        later=copy.deepcopy(initial);later['board_fallback']=self.triggered()
        site=dict(board_fallback=later['board_fallback'])
        with patch.object(audit,'_verify_hybrid_chain') as checker:
            audit.verify_hybrid_events(site,[initial,later],Path('.'),{},Checks(),{},[],[])
            self.assertEqual(checker.call_args.kwargs['snapshots'],[initial])
            self.assertTrue(checker.call_args.kwargs['geometry_triggered'])
            later['generated']['menu'].append({'id':'rewritten'})
            with self.assertRaises(EvidenceError):
                audit.verify_hybrid_events(site,[initial,later],Path('.'),{},Checks(),{},[],[])


if __name__=='__main__':unittest.main()
