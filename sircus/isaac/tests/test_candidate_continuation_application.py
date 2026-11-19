"""Application wiring through fake components only; no models or processes."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_continuation_application as subject
from candidate_continuation_ledger import ASSESSMENT_COHORTS


class FakeClock:
    def __init__(self):self.wall=10.;self.cpu=3.
    def tick(self):self.wall+=.01;self.cpu+=.02
    def time(self):return self.wall
    def usage(self):return dict(cpu_s=self.cpu,lifetime_peak_rss_mib=42.,largest_child_lifetime_peak_rss_mib=7.)


class FakeLedger:
    """Fast accounting interface; the real immutable ledger is tested separately."""
    def __init__(self,phase,request_cap=1750,release_cap=1400,train=None):
        self.phase,self.request_cap,self.release_cap=phase,request_cap,release_cap
        self.requested=self.released=0;self.pending=None;self.train=train
    def next_batch(self):
        if self.pending is not None:raise AssertionError("Unresolved fake reservation")
        return None if self.requested==self.request_cap or self.released==self.release_cap else object()
    def reserve(self):
        if self.next_batch() is None:raise AssertionError("Cap reached")
        size=min(35,self.request_cap-self.requested,self.release_cap-self.released)
        begin=self.requested;self.requested+=size
        self.pending=[dict(request_id=f"{self.phase}-{i:06d}") for i in range(begin,begin+size)]
        return self.pending
    def record(self,rows):
        assert self.pending is not None and len(rows)==len(self.pending)
        assert [r['request_id'] for r in rows]==[r['request_id'] for r in self.pending]
        self.released+=sum(r["released"] for r in rows);self.pending=None
    def summary(self):
        train=self if self.phase=="train" else self.train
        assessment=self if self.phase=="assessment" else None
        return dict(train=dict(requested=train.requested,released=train.released),
            assessment=dict(requested=assessment.requested if assessment else 0,released=assessment.released if assessment else 0))


class FakeRunner:
    def __init__(self,components):
        self.owner=components;self.ledger=FakeLedger("train")
        self.journal=types.SimpleNamespace(head="0"*64);self.imitation_calls=0;self.start_calls=0;self.finish_calls=0
    def imitation_step(self):
        self.owner.tick("imitation");self.imitation_calls+=1
    def start_ppo(self):
        assert self.imitation_calls==subject.IMITATION_MAX_STEPS
        self.start_calls+=1;self.owner.events.append("start_ppo")
    def reserve_train_batch(self):
        assert self.start_calls==1;self.owner.tick("train")
        rows=self.ledger.reserve();self.journal.head=f"{self.ledger.requested:064x}"
        self.owner.events.append("reserve_train");return rows
    def record_results(self,rows):self.ledger.record(rows);self.owner.events.append("record_train")
    def finish(self):
        assert self.ledger.next_batch() is None
        self.finish_calls+=1;self.owner.events.append("freeze_final")
        return dict(checkpoint_sha256="f"*64)


class FakeProducer:
    def __init__(self,components,native_instance,assessment=False):
        self.owner,self.native_instance,self.assessment=components,native_instance,assessment
        self.calls=0
    def execute_reserved(self,runner,choices,*,expected_journal_head):
        assert expected_journal_head==runner.journal.head and runner.ledger.pending==choices
        phase="assessment" if self.assessment else "train"
        self.owner.tick(phase);self.calls+=1;self.owner.events.append("execute_"+phase)
        return [dict(request_id=c["request_id"],released=self.owner.releases) for c in choices]


class FakeAssessment:
    def __init__(self,components,train):
        self.owner=components;self.directory=components.output/"assessment";self.directory.mkdir()
        self.ledger=FakeLedger("assessment",420,420,train)
        self.journal=types.SimpleNamespace(head="0"*64);self.cohorts=[];self.finish_calls=0
    def reserve_batch(self):
        self.owner.tick("assessment");cohort=self.ledger.requested//35
        self.cohorts.append(ASSESSMENT_COHORTS[cohort])
        rows=self.ledger.reserve();self.journal.head=f"{self.ledger.requested:064x}"
        self.owner.events.append("reserve_assessment");return rows
    def record_results(self,rows):self.ledger.record(rows);self.owner.events.append("record_assessment")
    def finish(self):
        assert self.ledger.next_batch() is None and self.ledger.requested==420
        self.finish_calls+=1;self.owner.events.append("finish_assessment")
        path=self.directory/"assessment-result.json";path.write_text('{"synthetic":true}')
        return dict(status="assessment_complete_audit_pending",synthetic=True)


class FakeComponents:
    synthetic=True
    def __init__(self,output,clock,*,releases=True,fail=None,audit=None,abort_raises=False):
        self.output,self.clock,self.releases,self.fail=output,clock,releases,fail
        self.events=[];self.abort_errors=[];self.close_calls=self.close_effects=self.native_starts=0
        self.native_instance=None;self.closed=False;self.audit_result=audit
        self.abort_raises=abort_raises;self.device_calls=[];self.runner=self.assessment=None
    def tick(self,phase):
        self.clock.tick()
        if self.fail==phase:raise RuntimeError("mock failure in "+phase)
    def device_snapshot(self,reset):
        self.device_calls.append(reset)
        return None if reset else dict(torch_peak_allocated_mib=17.,sampled_device_used_mib=1234.)
    def initialize(self,callback):
        self.tick("initialization");self.native_instance=object();self.native_starts+=1
        self.events.append("initialize")
        callback([dict(kind="mock_diagnostic",passed=True)],1)
        self.runner=FakeRunner(self);self.producer=FakeProducer(self,self.native_instance)
        return self.runner,self.producer,dict(records=[dict(kind="mock_diagnostic",passed=True)],parking=[{}])
    def begin_assessment(self,runner,trained):
        assert runner is self.runner and runner.finish_calls==1 and trained["checkpoint_sha256"]=="f"*64
        self.events.append("begin_assessment");self.tick("assessment")
        self.assessment=FakeAssessment(self,runner.ledger)
        self.assessor=FakeProducer(self,self.native_instance,True)
        return self.assessment,self.assessor
    def audit(self,trained,assessed):
        assert self.closed and self.assessment.finish_calls==1
        self.events.append("audit");self.tick("audit")
        return self.audit_result if self.audit_result is not None else dict(status="passed",all_components_passed=True,synthetic=True)
    def abort(self,error):
        self.events.append("abort");self.abort_errors.append(error)
        if self.abort_raises:raise RuntimeError("mock abort failure")
    def close_native(self):
        self.close_calls+=1
        if self.native_instance is not None and not self.closed:
            self.close_effects+=1;self.closed=True;self.events.append("close_native")


class FakePublisher:
    def __init__(self):self.rows=[]
    def publish(self,row):self.rows.append(copy.deepcopy(row))


class ApplicationWiringTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name);self.clock=FakeClock();self.publisher=FakePublisher()
    def run_application(self,**changes):
        self.components=FakeComponents(self.output,self.clock,**changes)
        costs=subject.PhaseCosts(clock=self.clock.time,usage=self.clock.usage,device=self.components.device_snapshot)
        return subject.run_pipeline(self.output,"1"*32,self.components,self.publisher,costs=costs)

    def test_complete_mock_release_cap_reuses_one_native_instance_and_freezes_before_assessment(self):
        result=self.run_application()
        c=self.components
        self.assertEqual(c.runner.imitation_calls,240);self.assertEqual(c.runner.start_calls,1)
        self.assertEqual(c.runner.finish_calls,1);self.assertEqual(c.native_starts,1)
        self.assertIs(c.producer.native_instance,c.assessor.native_instance)
        self.assertEqual((c.producer.calls,c.assessor.calls),(40,12))
        self.assertEqual(c.assessment.cohorts,list(ASSESSMENT_COHORTS))
        self.assertLess(c.events.index("freeze_final"),c.events.index("begin_assessment"))
        self.assertLess(c.events.index("close_native"),c.events.index("audit"))
        self.assertEqual(c.close_effects,1);self.assertEqual(c.close_calls,2)
        self.assertEqual(result["status"],"completed_audited");self.assertTrue(result["synthetic"])
        self.assertEqual(result["counters"],dict(train_requests=1400,train_releases=1400,assessment_requests=420,assessment_releases=420))
        self.assertFalse(result["model_promoted"]);self.assertFalse(result["execution_authorized"])
        self.assertEqual(self.publisher.rows[-1]["exit_code"],0)

    def test_all_decline_request_cap_keeps1750_train_and420_assessment_requests(self):
        result=self.run_application(releases=False)
        self.assertEqual(self.components.producer.calls,50)
        self.assertEqual(result["counters"],dict(train_requests=1750,train_releases=0,assessment_requests=420,assessment_releases=0))

    def test_progress_records_reservations_before_execution_and_all_phase_boundaries(self):
        self.run_application()
        rows=self.publisher.rows
        phases=[r["phase"] for i,r in enumerate(rows) if i==0 or r["phase"]!=rows[i-1]["phase"]]
        self.assertEqual(phases,list(subject.PHASES))
        self.assertTrue(all(r["assessment_requests"]==0 for r in rows if r["phase"] in ("initialization","imitation","train")))
        first=next(r for r in rows if r["phase"]=="train" and r["train_requests"]==35)
        self.assertEqual(first["train_releases"],0)
        self.assertTrue(all(r["audit_cpu_only"]==(r["phase"]=="audit") for r in rows))
        self.assertEqual(rows[-1]["parking_resets"],1)
        self.assertEqual(rows[-1]["diagnostics"],[dict(kind="mock_diagnostic",passed=True)])

    def test_every_phase_failure_preserves_partial_new_receipt_and_does_not_retry(self):
        for phase in subject.PHASES:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
                out=Path(directory);clock=FakeClock();publisher=FakePublisher();components=FakeComponents(out,clock,fail=phase)
                costs=subject.PhaseCosts(clock=clock.time,usage=clock.usage,device=components.device_snapshot)
                with self.assertRaisesRegex(RuntimeError,"mock failure"):
                    subject.run_pipeline(out,"2"*32,components,publisher,costs=costs)
                failure=json.loads((out/"application-failure.json").read_bytes())
                self.assertEqual(failure["status"],"incomplete_no_retry");self.assertFalse(failure["model_promoted"])
                self.assertEqual(len(components.abort_errors),1)
                self.assertFalse((out/"application-result.json").exists())
                self.assertLessEqual(components.native_starts,1);self.assertLessEqual(components.close_effects,1)
                if phase in ("initialization","imitation","train"):
                    self.assertIsNone(components.assessment)

    def test_failed_independent_audit_never_emits_audited_completion(self):
        for audit in (dict(status="failed",all_components_passed=False),dict(status="passed",all_components_passed=False)):
            with self.subTest(audit=audit),tempfile.TemporaryDirectory() as directory:
                out=Path(directory);clock=FakeClock();components=FakeComponents(out,clock,audit=audit)
                costs=subject.PhaseCosts(clock=clock.time,usage=clock.usage,device=components.device_snapshot)
                with self.assertRaisesRegex(ValueError,"Combined independent audit"):
                    subject.run_pipeline(out,"2"*32,components,FakePublisher(),costs=costs)
                self.assertFalse((out/"application-result.json").exists())
                self.assertEqual(components.close_effects,1)

    def test_abort_failure_is_preserved_without_hiding_original_failure(self):
        with self.assertRaisesRegex(RuntimeError,"mock failure in train"):
            self.run_application(fail="train",abort_raises=True)
        row=json.loads((self.output/"application-failure.json").read_bytes())
        self.assertIn("mock abort failure",row["abort_error"])
        self.assertIn("mock failure in train",row["error"])
        self.assertEqual(self.components.close_effects,1)

    def test_actual_phase_cost_fields_have_distinct_overlap_scopes(self):
        result=self.run_application()
        receipt=result["resources"]
        self.assertEqual(set(receipt["phases"]),set(subject.PHASES))
        for row in receipt["phases"].values():
            self.assertGreaterEqual(row["wall_s"],0);self.assertGreaterEqual(row["aggregate_cpu_s"],0)
            self.assertEqual(row["lifetime_peak_rss_mib"],42.)
            self.assertEqual(row["largest_child_lifetime_peak_rss_mib"],7.)
            self.assertEqual(row["device"]["sampled_device_used_mib"],1234.)
        self.assertIn("not a summed phase peak",receipt["scope"])
        self.assertEqual(self.components.device_calls,[True,False]*5)


class PhaseCostTests(unittest.TestCase):
    def test_missing_device_samples_stay_null_not_zero(self):
        clock=FakeClock();costs=subject.PhaseCosts(clock=clock.time,usage=clock.usage)
        costs.begin("initialization");clock.tick();row=costs.finish()
        self.assertIsNone(row["device"]);self.assertAlmostEqual(row["wall_s"],.01)
        self.assertAlmostEqual(row["aggregate_cpu_s"],.02)

    def test_repeated_overlapping_and_invalid_phases_are_rejected(self):
        clock=FakeClock();costs=subject.PhaseCosts(clock=clock.time,usage=clock.usage)
        with self.assertRaises(ValueError):costs.finish()
        with self.assertRaises(ValueError):costs.begin("unknown")
        costs.begin("train")
        with self.assertRaises(ValueError):costs.begin("assessment")
        costs.finish()
        with self.assertRaises(ValueError):costs.begin("train")

    def test_negative_or_nonfinite_cpu_wall_rss_device_cannot_be_reported(self):
        for kind in ("cpu","wall","rss","device"):
            with self.subTest(kind=kind):
                clock=FakeClock()
                def usage():
                    value=clock.usage()
                    if kind=="rss":value["lifetime_peak_rss_mib"]=-1.
                    return value
                device=lambda reset:None if reset else ({"memory_mib":float("nan")} if kind=="device" else None)
                costs=subject.PhaseCosts(clock=clock.time,usage=usage,device=device)
                costs.begin("initialization")
                if kind=="cpu":clock.cpu-=1.
                if kind=="wall":clock.wall-=1.
                with self.assertRaisesRegex(ValueError,"Invalid phase"):costs.finish()


if __name__ == "__main__":unittest.main()
