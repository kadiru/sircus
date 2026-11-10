"""Synthetic rehearsal only: no policy, simulator, optimizer or real shots."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import candidate_v4_evaluation_runner as subject
from candidate_continuation_journal import read_events
from candidate_continuation_ledger import canonical_hash


class Clock:
    def __init__(self): self.value=0.; self.cpu=0.
    def time(self): return self.value
    def tick(self,value=1.): self.value+=value; self.cpu+=max(value,0)/2
    def usage(self): return dict(cpu_s=self.cpu,lifetime_peak_rss_mib=42.)


class MockRuntime:
    synthetic=True
    mock_only=True
    def __init__(self,output,plan,clock):
        self.output,self.clock=output,clock
        self.state=dict(final_checkpoint_sha256=plan["final_checkpoint_sha256"],
            parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
            bindings=copy.deepcopy(plan["bindings"]),model_hashes=copy.deepcopy(plan["original_plan"]["model_hashes"]))
        self.events=[];self.initializations=self.closes=0;self.cohorts=[]
        self.all_decline=False;self.partial=None;self.fail_at=None;self.exception_prefix=None
        self.row_change=None;self.after_execute=None;self.after_audit=None;self.bad_audit=False
        self.init_seconds=1.;self.close_error=False
    def verify_frozen(self,plan): return copy.deepcopy(self.state)
    def initialize(self,plan):
        self.initializations+=1;self.events.append("initialize");self.clock.tick(self.init_seconds)
    def execute(self,cohort,choices,reservation):
        events=read_events(self.output/"journal")
        assert events[-1]["kind"]=="reserve"
        assert events[-1]["fingerprint"]==reservation["journal_head_sha256"]
        assert events[-1]["payload"]["requested"]==reservation["requested"]
        assert events[-1]["payload"]["choices"]==choices
        assert reservation["request_ids"]==[c["request_id"] for c in choices]
        self.cohorts.append(copy.deepcopy(cohort));self.events.append("execute");self.clock.tick()
        rows=[]
        for choice in choices:
            released=choice["enabled"] and not self.all_decline
            captured=released and choice["request_index"]%4==0
            clean=captured and choice["request_index"]%8==0
            rows.append(dict(request_id=choice["request_id"],released=released,
                captured=captured,clean=clean,decline_reason=None if released else
                choice["decline_reason"] or "mock_guard_decline",source_sha256="e"*64,
                **{"return":-1. if not released else 1.6 if clean else 1.5 if captured else -.5}))
        if self.row_change: self.row_change(choices,rows)
        if self.after_execute: self.after_execute(self)
        if cohort["index"]==self.fail_at:
            error=RuntimeError("fabricated interrupted callback")
            if self.exception_prefix is not None: error.terminal_prefix=rows[:self.exception_prefix]
            raise error
        return rows if self.partial is None else rows[:self.partial]
    def close(self):
        self.closes+=1;self.events.append("close")
        if self.close_error: raise RuntimeError("fabricated close failure")
    def audit(self,directory,plan,summary):
        from candidate_v4_evaluation_audit import audit_mock_evaluation
        assert self.closes==1
        self.events.append("audit");self.clock.tick()
        value=audit_mock_evaluation(directory,plan,summary)
        if self.bad_audit: value["full_native_audit"]=True
        if self.after_audit: self.after_audit(self)
        return value


class EvaluationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_candidate_v4_evaluation import fixture
        cls.prepared=fixture()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.protected=self.root/"old-evidence"
        self.protected.mkdir();self.old=self.protected/"old-partial.json"
        self.old.write_text('{"charged":140,"terminal":105,"unresolved":35}')
        self.old_bytes=self.old.read_bytes()
        self.plan=copy.deepcopy(self.prepared)
        self.output=self.root/"mock-evaluation"
        self.clock=Clock();self.free=10*1024**3
        self.runtime=MockRuntime(self.output,self.plan,self.clock)
    def runner(self,**kwargs):
        options=dict(protected_roots=[str(self.protected)],clock=self.clock.time,
            usage=self.clock.usage,free_bytes=lambda:self.free)
        options.update(kwargs)
        return subject.MockEvaluation(self.output,self.plan,self.runtime,**options)
    def failure(self): return json.loads((self.output/"evaluation-failure.json").read_bytes())
    def assert_failure(self,requested,terminal):
        failure=self.failure()
        self.assertEqual(failure["status"],"mock_incomplete_no_retry")
        self.assertEqual(failure["counts"]["requested"],requested)
        self.assertEqual(failure["counts"]["terminal"],terminal)
        self.assertEqual(failure["counts"]["unresolved"],requested-terminal)
        self.assertEqual(self.old.read_bytes(),self.old_bytes)
        self.assertFalse((self.output/"evaluation-result.json").exists())
        self.assertEqual(self.runtime.closes,1)

    def test_complete210_uses_frozen_order_and_preserves_old_partial(self):
        result=self.runner().run()
        self.assertEqual([c["method"] for c in self.runtime.cohorts],
            ["continued","parent","candidate_bank","parent","candidate_bank","continued"])
        self.assertEqual([c["original_cohort"] for c in self.runtime.cohorts],list(range(6,12)))
        self.assertEqual([c["choices"] for c in self.runtime.cohorts],[c["choices"] for c in self.plan["cohorts"]])
        self.assertEqual(result["counts"]["requested"],210)
        self.assertEqual(result["counts"]["terminal"],210)
        self.assertEqual(result["counts"]["unresolved"],0)
        self.assertEqual(result["status"],"mock_evaluation_complete_audit_pending")
        self.assertEqual(result["audit"]["status"],"passed_mock_only")
        self.assertFalse(result["audit"]["full_native_audit"])
        self.assertEqual((result["training_requests"],result["optimizer_steps"],result["native_requests_executed"]),(0,0,0))
        self.assertEqual((result["native_application_starts"],result["mock_initializations"]),(0,1))
        self.assertFalse(result["model_promoted"]);self.assertFalse(result["execution_authorized"])
        self.assertEqual(set(result["resources"]["phases"]),{"initialization","evaluation","audit"})
        self.assertEqual(self.runtime.events[-2:],["close","audit"])
        self.assertEqual(self.old.read_bytes(),self.old_bytes)

    def test_all210_rejections_are_failures_not_replaced(self):
        self.runtime.all_decline=True
        result=self.runner().run()
        self.assertEqual(result["counts"],dict(requested=210,terminal=210,released=0,rejected=210,
            captures=0,clean_captures=0,unresolved=0))
        self.assertEqual(len(self.runtime.cohorts),6)

    def test_partial_prefix_charges35_and_closes_without_retry(self):
        self.runtime.partial=7
        run=self.runner()
        with self.assertRaisesRegex(ValueError,"Partial cohort"):run.run()
        self.assert_failure(35,7)
        self.assertEqual(len(self.runtime.cohorts),1)
        with self.assertRaisesRegex(ValueError,"already used"):run.run()
        self.assertEqual(self.runtime.initializations,1)

    def test_callback_failure_without_prefix_preserves35_unresolved(self):
        self.runtime.fail_at=0
        with self.assertRaisesRegex(RuntimeError,"interrupted"):self.runner().run()
        self.assert_failure(35,0)
        self.assertEqual(read_events(self.output/"journal")[-2]["kind"],"reserve")

    def test_callback_failure_with_prefix_preserves_committed_terminal_count(self):
        self.runtime.fail_at=0;self.runtime.exception_prefix=9
        with self.assertRaisesRegex(RuntimeError,"interrupted"):self.runner().run()
        self.assert_failure(35,9)

    def test_second_cohort_failure_keeps_first35_and_next35_charge(self):
        self.runtime.fail_at=1
        with self.assertRaises(RuntimeError):self.runner().run()
        self.assert_failure(70,35)

    def test_existing_directory_cannot_be_reused_or_erased(self):
        self.output.mkdir();witness=self.output/"keep";witness.write_text("preserve")
        with self.assertRaisesRegex(ValueError,"already exists"):self.runner()
        self.assertEqual(witness.read_text(),"preserve");self.assertEqual(self.runtime.initializations,0)

    def test_protected_ancestor_descendant_and_links_rejected_before_init(self):
        for output in (self.protected/"new",self.root):
            self.output=output
            with self.assertRaisesRegex(ValueError,"overlaps"):self.runner()
        link=self.root/"link";link.symlink_to(self.protected,target_is_directory=True)
        self.output=link/"new"
        with self.assertRaises(ValueError):self.runner()
        self.assertEqual(self.runtime.initializations,0)

    def test_native_plan_and_runtime_rejected_before_output_or_initialize(self):
        self.plan["synthetic"]=False
        self.plan["fingerprint"]=canonical_hash({k:v for k,v in self.plan.items() if k!="fingerprint"})
        with self.assertRaises(ValueError):self.runner()
        self.assertFalse(self.output.exists());self.assertEqual(self.runtime.initializations,0)
        self.plan["synthetic"]=True
        self.plan["fingerprint"]=canonical_hash({k:v for k,v in self.plan.items() if k!="fingerprint"})
        self.runtime.mock_only=False
        with self.assertRaisesRegex(ValueError,"mock-only"):self.runner()

    def test_learning_api_is_not_accepted_by_evaluation_runtime(self):
        self.runtime.train=lambda:None
        with self.assertRaisesRegex(ValueError,"training or optimizer"):self.runner()
        self.assertFalse(self.output.exists())

    def test_checkpoint_or_source_state_tamper_rejected_before_initialize(self):
        self.runtime.state["final_checkpoint_sha256"]="f"*64
        with self.assertRaisesRegex(ValueError,"Frozen checkpoint"):self.runner()
        self.assertFalse(self.output.exists());self.assertEqual(self.runtime.initializations,0)

    def test_inflight_model_tamper_keeps_charge_but_never_final_result(self):
        self.runtime.after_execute=lambda r:r.state["model_hashes"].update(continued_weight_hash="1"*64)
        with self.assertRaisesRegex(ValueError,"Frozen checkpoint"):self.runner().run()
        self.assert_failure(35,35)

    def test_out_of_order_outcomes_do_not_make_terminal_receipts(self):
        self.runtime.row_change=lambda choices,rows:rows.reverse()
        with self.assertRaisesRegex(ValueError,"identity or order"):self.runner().run()
        self.assert_failure(35,0)

    def test_empty_support_cannot_be_released_or_fabricated(self):
        def alter(choices,rows):
            for choice,row in zip(choices,rows):
                if not choice["enabled"]:
                    row.update(released=True,decline_reason=None,**{"return":-.5});return
        self.runtime.row_change=alter
        with self.assertRaisesRegex(ValueError,"Empty candidate"):self.runner().run()
        self.assert_failure(35,0)

    def test_initialization_timeout_closes_once_before_evaluation(self):
        self.runtime.init_seconds=601.
        with self.assertRaisesRegex(ValueError,"wall or phase"):self.runner().run()
        self.assert_failure(0,0);self.assertFalse(self.runtime.cohorts)

    def test_storage_floor_and_invalid_clock_fail_before_dispatch(self):
        self.runtime.after_execute=lambda r:setattr(self,"free",6*1024**3-1)
        with self.assertRaisesRegex(ValueError,"6 GiB"):self.runner().run()
        self.assert_failure(35,35)

    def test_backwards_clock_preserves_charged_work(self):
        self.runtime.after_execute=lambda r:setattr(self.clock,"value",-1.)
        with self.assertRaisesRegex(ValueError,"Monotonic clock"):self.runner().run()
        self.assert_failure(35,35)

    def test_native_audit_claim_is_rejected_and_no_completion_file_written(self):
        self.runtime.bad_audit=True
        with self.assertRaisesRegex(ValueError,"mock-only audit"):self.runner().run()
        self.assert_failure(210,210)
        self.assertFalse((self.output/"mock-audit.json").exists())

    def test_model_tamper_in_audit_is_rejected_before_result(self):
        self.runtime.after_audit=lambda r:r.state.update(final_checkpoint_sha256="2"*64)
        with self.assertRaisesRegex(ValueError,"Frozen checkpoint"):self.runner().run()
        self.assert_failure(210,210)

    def test_close_error_is_not_retried_and_preserves_complete_charges(self):
        self.runtime.close_error=True
        with self.assertRaisesRegex(RuntimeError,"close failure"):self.runner().run()
        self.assert_failure(210,210)
        self.assertNotIn("audit",self.runtime.events)


if __name__=="__main__": unittest.main()
