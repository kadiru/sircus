import copy
import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab import parallel_board_audit as audit
from sircus.mjlab import parallel_board_study as study
from sircus.mjlab.benchmark_verify import Checks,sha
from sircus.mjlab.planning_budget_plan import _wrap_choice
from sircus.mjlab.tests.test_stalled_reference_study import choice


class ParallelAuditTests(unittest.TestCase):
    def fixture(self,path):
        layout=dict(id='p',bucket_xy_m=[.4,-.1]);wrappers=[]
        for method in study.METHODS:
            wrappers.append(dict(variant=method,site=dict(layout=layout,choice=choice(layout),selected_adapter_directory='alternative')))
        def write(name,value):(path/name).write_text(json.dumps(value))
        write('layouts.json',dict(layouts=[layout]));write('plans.json',wrappers)
        write('contract.json',dict(frozen_inputs={'layouts.json':sha(path/'layouts.json')}))
        slots=[_wrap_choice(w['site']['choice'],w['variant']) for w in wrappers]
        value=dict(slots=slots,requests={v:[r for w in wrappers if w['variant']==v for r in study.repeated_requests(w['site']['choice'],v)] for v in study.METHODS},
            channels=study.native_channels(wrappers),bindings={n:sha(path/n) for n in ('layouts.json','contract.json','plans.json')})
        write('choices.json',value)
        manifest={p.name:sha(p) for p in path.iterdir()}
        return [layout],wrappers,manifest,value

    def test_repeat_proof_and_adapter_namespace_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);layouts,wrappers,manifest,value=self.fixture(p)
            audit.verify_slots(p,layouts,wrappers,manifest,Checks(),native_started=False)
            for key in ('proof','id','lane'):
                altered=copy.deepcopy(value)
                if key=='proof':altered['requests']['parallel'][0]['prepared_connection_sha256']='bad'
                elif key=='id':altered['requests']['parallel'][0]['id']='different-repeat'
                else:altered['channels']['parallel/original']=altered['channels'].pop('parallel/alternative')
                (p/'choices.json').write_text(json.dumps(altered))
                with self.assertRaises((ValueError,AssertionError)):
                    audit.verify_slots(p,layouts,wrappers,manifest,Checks(),native_started=False)

    def test_final_audit_requires_native_start_binding(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);layouts,wrappers,manifest,value=self.fixture(p)
            with self.assertRaises(FileNotFoundError):
                audit.verify_slots(p,layouts,wrappers,manifest,Checks(),native_started=True)


class ConcurrentSelectionTests(unittest.TestCase):
    def shot(self,score,margin=.01):
        return dict(request={'id':'checked'},prediction=dict(predicted_route='direct',direct=dict(arm_clearance_m=margin),score_m=score))

    def test_stronger_final_original_beats_alternative_that_improved_stale_trigger(self):
        trigger=self.shot(.001);alternative=self.shot(.008);final=self.shot(.012)
        self.assertEqual(audit.independent_final_selection(trigger,alternative)['selected_lane'],'alternative')
        self.assertEqual(audit.independent_final_selection(final,alternative)['selected_lane'],'original')
        self.assertEqual(audit.independent_final_selection(final,self.shot(.02,.004))['selected_lane'],'original')
        self.assertEqual(audit.independent_final_selection(final,self.shot(.02),True)['selected_lane'],'original')

    def test_equal_and_unknown_scores_preserve_original(self):
        original=self.shot(.01)
        for other in (self.shot(.01),self.shot(None),None):
            self.assertEqual(audit.independent_final_selection(original,other)['selected_lane'],'original')
        self.assertEqual(audit.independent_final_selection(self.shot(None),self.shot(.02))['selected_lane'],'original')

    def timing(self):
        return dict(decision_latency_s=30.,trigger=dict(elapsed_s=8.),
            shared_clock=dict(started=100.,deadline=130.,cutoff_elapsed_s=30.,original_interrupted_by_alternative=False),
            lanes=dict(original=dict(budget_s=30.,decision_latency_s=30.),alternative=dict(budget_s=30.,decision_latency_s=30.,
                launch_offset_s=8.1,readiness_offset_s=9.,available_budget_at_launch_s=21.9)))

    def test_second_lane_does_not_receive_a_new_thirty_second_budget(self):
        value=self.timing();audit.verify_shared_timing(value,Checks())
        for key,replacement in (('available_budget_at_launch_s',30.),('readiness_offset_s',31.),('decision_latency_s',31.)):
            bad=copy.deepcopy(value);bad['lanes']['alternative'][key]=replacement
            with self.assertRaises((ValueError,AssertionError)):audit.verify_shared_timing(bad,Checks())
        bad=copy.deepcopy(value);bad['shared_clock']['original_interrupted_by_alternative']=True
        with self.assertRaises((ValueError,AssertionError)):audit.verify_shared_timing(bad,Checks())

    def test_late_hash_validation_cannot_enter_final_selection(self):
        value=self.timing();value['lanes']['original']['accepted_receipts']=[dict(received_elapsed_s=29.9,validated_elapsed_s=30.01)]
        with self.assertRaises((ValueError,AssertionError)):audit.verify_shared_timing(value,Checks())
        value['lanes']['original']['accepted_receipts'][0]['validated_elapsed_s']=29.99
        audit.verify_shared_timing(value,Checks())
