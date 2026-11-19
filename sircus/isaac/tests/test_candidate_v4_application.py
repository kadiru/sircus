"""Synthetic orchestration only; no App, checkpoint load or native shot."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_v4_application as subject
from candidate_continuation_journal import read_events
from candidate_continuation_ledger import canonical_hash
from test_candidate_v4_evaluation import fixture


class Clock:
    def __init__(self):self.value=0.
    def time(self):return self.value
    def tick(self,seconds=1.):self.value+=seconds
    def usage(self):return dict(cpu_s=self.value/2,lifetime_peak_rss_mib=42.,largest_child_lifetime_peak_rss_mib=7.)


class Publisher:
    def __init__(self):self.rows=[]
    def publish(self,value):self.rows.append(copy.deepcopy(value))


class Components:
    synthetic=True
    def __init__(self,root,plan,clock):
        self.root,self.plan,self.clock=root,plan,clock;self.closed=False
        self.events=[];self.starts=self.close_effects=0;self.partial=None;self.failure=None;self.fail_after=None
        self.bad_audit=False;self.init_seconds=1.;self.change_model=False;self.called=[]
        self.frozen=dict(final_checkpoint_sha256=plan["final_checkpoint_sha256"],parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
            bindings=copy.deepcopy(plan["bindings"]),model_hashes=copy.deepcopy(plan["model_hashes"]))
    def verify_frozen(self):
        v=copy.deepcopy(self.frozen)
        if self.change_model:v["final_checkpoint_sha256"]="d"*64
        return v
    def device_snapshot(self,reset):return None
    def initialize(self,callback):
        self.events.append("initialize");self.starts+=1;self.clock.tick(self.init_seconds)
        if self.failure=="initialization":raise RuntimeError("mock initialization failed")
        callback([dict(synthetic=True)],1)
        return dict(records=[dict(synthetic=True)],parking=[{}])
    def execute_reserved(self,cohort,choices,reservation,*,journal_directory):
        events=read_events(journal_directory)
        assert events[-1]["kind"]=="reserve" and events[-1]["fingerprint"]==reservation["journal_head_sha256"]
        assert events[-1]["payload"]["requested"]==(cohort["index"]+1)*35
        self.called.append(cohort["index"]);self.events.append("execute");self.clock.tick()
        rows=[]
        for choice in choices:
            released=choice["enabled"]
            rows.append(dict(request_id=choice["request_id"],released=released,captured=False,clean=False,
                decline_reason=None if released else "no_qualified_candidate_support",source_sha256="a"*64,
                **{"return":-.5 if released else -1.}))
        if self.failure=="evaluation":
            error=RuntimeError("mock shot failed")
            if self.partial is not None:error.terminal_prefix=rows[:self.partial]
            raise error
        if self.fail_after:self.fail_after()
        return rows if self.partial is None else rows[:self.partial]
    def close_native(self):
        if not self.closed:self.closed=True;self.close_effects+=1;self.events.append("close")
    def audit(self,summary):
        assert self.closed
        assert json.loads((self.root/"evaluation-result.json").read_bytes())==summary
        self.events.append("audit");self.clock.tick()
        if self.failure=="audit":raise RuntimeError("mock audit failed")
        v=dict(status="passed_synthetic",all_components_passed=not self.bad_audit,synthetic=True,
            plan_fingerprint=self.plan["fingerprint"],journal_head_sha256=summary["journal_head_sha256"])
        v["fingerprint"]=canonical_hash(v);return v


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prepared=fixture()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.plan=copy.deepcopy(self.prepared)
        self.clock=Clock();self.publisher=Publisher();self.components=Components(self.root,self.plan,self.clock)
        self.free=10*1024**3
    def run_app(self,**options):
        return subject.run_application(self.root,self.plan,self.components,self.publisher,clock=self.clock.time,
            free_bytes=lambda:self.free,costs=subject.PhaseCosts(clock=self.clock.time,measure=self.clock.usage),**options)
    def failure(self):return json.loads((self.root/"application-failure.json").read_bytes())
    def test_complete210_has_only_three_phases_no_training_and_cpu_audit_after_close(self):
        v=self.run_app()
        self.assertEqual(v["status"],"synthetic_application_complete");self.assertTrue(v["synthetic"])
        self.assertEqual(v["counts"]["requested"],210);self.assertEqual(v["counts"]["terminal"],210)
        self.assertEqual(v["counts"]["rejected"],4);self.assertEqual(v["native_application_starts"],0)
        self.assertEqual(self.components.called,list(range(6)))
        self.assertEqual(self.components.starts,1);self.assertEqual(self.components.close_effects,1)
        self.assertEqual(self.components.events[-2:],["close","audit"])
        self.assertEqual(set(v["resources"]["phases"]),{"initialization","evaluation","audit"})
        self.assertFalse(v["original_assessment_completed"]);self.assertFalse(v["model_promoted"])
        phases=[]
        for row in self.publisher.rows:
            if not phases or phases[-1]!=row["phase"]:phases.append(row["phase"])
            self.assertEqual(row["train_requests"],0);self.assertEqual(row["optimizer_steps"],0)
            self.assertEqual(row["audit_cpu_only"],row["phase"]=="audit")
        self.assertEqual(phases,["initialization","evaluation","audit"])

    def test_progress_charges_before_dispatch_and_matches_watchdog_fields(self):
        self.run_app()
        first=next(r for r in self.publisher.rows if r["evaluation_requests"]==35)
        self.assertEqual(first["evaluation_terminal"],0)
        self.assertEqual(set(first),{"phase","diagnostics","parking_resets","evaluation_requests","evaluation_terminal",
            "evaluation_releases","train_requests","optimizer_steps","running","exit_code","audit_cpu_only"})

    def test_returned_partial_prefix_is_preserved_and_never_retried(self):
        self.components.partial=7
        with self.assertRaisesRegex(ValueError,"Partial TEST"):self.run_app()
        counts=self.failure()["counts"]
        self.assertEqual((counts["requested"],counts["terminal"],counts["unresolved"]),(35,7,28))
        self.assertEqual(self.components.called,[0]);self.assertEqual(self.components.close_effects,1)
        with self.assertRaisesRegex(ValueError,"already used"):self.run_app()

    def test_callback_exception_preserves_zero_or_explicit_terminal_prefix(self):
        self.components.failure="evaluation";self.components.partial=9
        with self.assertRaisesRegex(RuntimeError,"mock shot"):self.run_app()
        counts=self.failure()["counts"]
        self.assertEqual((counts["requested"],counts["terminal"],counts["unresolved"]),(35,9,26))
        self.assertFalse((self.root/"evaluation-result.json").exists())

    def test_every_phase_failure_has_closed_evidence_not_audited_completion(self):
        for phase in ("initialization","evaluation","audit"):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();clock=Clock();c=Components(root,self.plan,clock);c.failure=phase
                with self.assertRaises(RuntimeError):
                    subject.run_application(root,self.plan,c,Publisher(),clock=clock.time,free_bytes=lambda:10*1024**3,
                        costs=subject.PhaseCosts(clock=clock.time,measure=clock.usage))
                f=json.loads((root/"application-failure.json").read_bytes())
                self.assertEqual(f["status"],"incomplete_no_retry");self.assertFalse(f["original_assessment_completed"])
                self.assertFalse((root/"application-result.json").exists());self.assertEqual(c.close_effects,1)

    def test_bad_independent_audit_cannot_generate_completion(self):
        self.components.bad_audit=True
        with self.assertRaisesRegex(ValueError,"audit did not pass"):self.run_app()
        self.assertEqual(self.failure()["counts"]["terminal"],210)
        self.assertTrue((self.root/"evaluation-result.json").exists())
        self.assertFalse((self.root/"application-result.json").exists())

    def test_initialization_timeout_stops_before_any_test_request(self):
        self.components.init_seconds=601
        with self.assertRaisesRegex(ValueError,"deadline"):self.run_app()
        self.assertEqual(self.failure()["counts"]["requested"],0);self.assertFalse(self.components.called)

    def test_storage_and_frozen_model_failure_keep_charged_prefix(self):
        self.components.fail_after=lambda:setattr(self,"free",6*1024**3-1)
        with self.assertRaisesRegex(ValueError,"6 GiB"):self.run_app()
        self.assertEqual(self.failure()["counts"]["terminal"],35)

    def test_source_or_model_change_prevents_any_app_start(self):
        self.components.change_model=True
        with self.assertRaisesRegex(ValueError,"Frozen model"):self.run_app()
        self.assertEqual(self.components.starts,0);self.assertFalse((self.root/"journal").exists())

    def test_training_api_is_rejected_before_any_application_evidence(self):
        self.components.optimizer=object()
        with self.assertRaisesRegex(ValueError,"training API"):self.run_app()
        self.assertFalse((self.root/"journal").exists())

    def test_real_provenance_requires_separate_approval(self):
        self.components.synthetic=False
        self.plan["synthetic"]=False
        self.plan["fingerprint"]=canonical_hash({k:v for k,v in self.plan.items() if k!="fingerprint"})
        with self.assertRaises(ValueError):self.run_app()
        self.assertEqual(self.components.starts,0)


if __name__=="__main__":unittest.main()
