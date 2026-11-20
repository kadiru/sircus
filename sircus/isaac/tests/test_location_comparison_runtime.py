"""Fake results exercise the complete runner; no model, simulator or learning."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_location_comparison import fixture, bind
import location_comparison as preparation
from location_comparison_bounds import Bounds
from location_comparison_mock import FakeClock, MockEngine
from location_comparison_runtime import ComparisonRunner, Journal


def synthetic_plan():
    train, history, geometry = fixture()
    plan = preparation.build_comparison(train, history, geometry, bind(train,history,geometry))
    # Clearly synthetic 25-local fixture matches the actual plan's600 cap.
    plan["cohorts"]["test_local"] = plan["cohorts"]["test_local"][:25]
    plan["protocol"]["test_positions"]["local"] = 25
    plan["protocol"]["final_assessment_requests"] = 285
    plan["protocol"]["total_assessment_requests"] = 600
    plan.pop("fingerprint")
    return preparation.signed(plan)


def runner(plan, engine_class=MockEngine, journal=None):
    clock = FakeClock()
    engine = engine_class(clock)
    bounds = Bounds(plan, clock, lambda: dict(free_storage_bytes=10*1024**3))
    return ComparisonRunner(plan, engine, bounds, journal or Journal(), clock=clock), engine


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = synthetic_plan()
        cls.full_runner, cls.engine = runner(cls.plan)
        cls.trace = cls.full_runner.run()

    def test_complete_budgets_and_exactly_one_update_per_140_fresh_requests(self):
        finish = self.trace["events"][-1]["payload"]
        self.assertEqual(finish["status"], "complete_audit_pending", finish["reason"])
        self.assertEqual(finish["counts"]["assessment"]["completed"], 600)
        for arm in ("original_usable", "broader"):
            counts = finish["counts"]["train"][arm]
            self.assertEqual((counts["requested"],counts["completed"]), (8960,8960))
            self.assertEqual(counts["captures"]+counts["released_misses"]+counts["rejected"], 8960)
            self.assertGreater(counts["rejected"], 0)
            self.assertEqual(self.engine.state(arm)["updates"], 137)
            self.assertEqual(self.engine.state(arm)["optimizer_steps"], 548)
        self.assertEqual(len(self.engine.learned), 128)
        learned = [request for update in self.engine.learned for request in update["request_ids"]]
        self.assertEqual(len(learned), len(set(learned)))
        self.assertTrue(all(request.startswith("train:") for request in learned))
        with self.assertRaisesRegex(ValueError, "one-use"):
            self.full_runner.run()

    def test_final_tests_use_frozen_models_and_rotate_order_per_target(self):
        events = self.trace["events"]
        freeze = next(e["seq"] for e in events if e["kind"] == "frozen")
        by_target = {}
        for event in events:
            if event["kind"] != "reserve" or not event["payload"]["phase"].startswith("test_"):
                continue
            self.assertGreater(event["seq"],freeze)
            for cell in event["payload"]["cells"]:
                by_target.setdefault(cell["target_id"],[]).append(event["payload"]["model"])
        models = ("parent","original_usable","broader")
        for role in ("test_broad","test_local"):
            for index,row in enumerate(self.plan["cohorts"][role]):
                self.assertEqual(by_target[row["id"]], [models[(index+r)%3] for r in range(3)])
        self.assertEqual(self.engine.state("parent")["updates"],73)
        self.assertFalse(any(e["kind"] == "update" for e in events[freeze:]))

    def test_changed_plan_and_parent_state_fail_before_requests(self):
        changed = copy.deepcopy(self.plan)
        changed["schedules"]["broader"][0]["bucket_xy_m"][0] += .001
        with self.assertRaisesRegex(ValueError,"Changed prepared"):
            runner(changed)

        class WrongParent(MockEngine):
            def restore(self,name,*args):
                state=super().restore(name,*args)
                if name == "broader":
                    state["optimizer_hash"] = "f"*64
                    self.models[name] = state
                return state
        value,engine = runner(self.plan,WrongParent)
        trace=value.run()
        self.assertEqual(trace["events"][-1]["payload"]["status"],"incomplete_no_retry")
        self.assertEqual(engine.executed,[])

    def test_partial_native_failure_preserves_charges_without_retry_or_final_test(self):
        class FailsFirstTrain(MockEngine):
            def execute(self,cells,*args,**kwargs):
                if cells[0]["split"] == "train":
                    raise RuntimeError("invented native failure")
                return super().execute(cells,*args,**kwargs)
        value,engine=runner(self.plan,FailsFirstTrain)
        trace=value.run(); final=trace["events"][-1]["payload"]
        self.assertEqual(final["status"],"incomplete_no_retry")
        self.assertEqual(final["counts"]["train"]["original_usable"]["requested"],35)
        self.assertEqual(final["counts"]["train"]["original_usable"]["completed"],0)
        self.assertIsNotNone(final["bounds"]["pending"])
        self.assertEqual(engine.learned,[])
        self.assertFalse(any(e["kind"] == "frozen" for e in trace["events"]))

    def test_terminal_work_after_deadline_is_counted_but_next_request_never_runs(self):
        class Late(MockEngine):
            def execute(self,*args,**kwargs):
                result=super().execute(*args,**kwargs)
                self.clock.advance(3600)
                return result
        value,engine=runner(self.plan,Late)
        trace=value.run(); final=trace["events"][-1]["payload"]
        self.assertEqual(final["status"],"incomplete_no_retry")
        self.assertEqual(final["counts"]["assessment"]["completed"],35)
        self.assertEqual(final["bounds"]["assessment_completed"],35)
        self.assertEqual(len(engine.executed),1)
        self.assertEqual(engine.learned,[])

    def test_omitted_result_cannot_be_counted_as_completed(self):
        class Missing(MockEngine):
            def execute(self,*args,**kwargs):
                result=super().execute(*args,**kwargs); result["rows"].pop()
                return result
        value,_=runner(self.plan,Missing)
        final=value.run()["events"][-1]["payload"]
        self.assertEqual(final["counts"]["assessment"]["requested"],35)
        self.assertEqual(final["counts"]["assessment"]["completed"],0)
        self.assertEqual(final["status"],"incomplete_no_retry")

    def test_native_engine_requires_external_owner_and_durable_journal(self):
        value,engine=runner(self.plan)
        engine.synthetic=False
        with mock.patch("location_comparison_runtime.PREPARED_FINGERPRINT",self.plan["fingerprint"]):
            with self.assertRaisesRegex(ValueError,"explicit authorization"):
                ComparisonRunner(self.plan,engine,value.bounds,Journal(),clock=value.clock)
        self.assertEqual(engine.models,{})

    def test_restored_rng_pairing_and_assessment_rng_preservation_fail_closed(self):
        class RngEngine(MockEngine):
            def restore(self,name,*args):
                super().restore(name,*args)
                self.models[name].update(action_rng_hash="a"*64,optimizer_rng_hash="b"*64)
                return self.state(name)
        class Unpaired(RngEngine):
            def restore(self,name,*args):
                super().restore(name,*args)
                if name == "broader": self.models[name]["action_rng_hash"]="c"*64
                return self.state(name)
        value,engine=runner(self.plan,Unpaired)
        self.assertIn("not paired",value.run()["events"][-1]["payload"]["reason"])
        self.assertEqual(engine.executed,[])
        class Consumed(RngEngine):
            def propose(self,name,cells,deterministic):
                packet=super().propose(name,cells,deterministic)
                if deterministic: self.models[name]["action_rng_hash"]="d"*64
                return packet
        value,engine=runner(self.plan,Consumed)
        self.assertIn("Assessment consumed",value.run()["events"][-1]["payload"]["reason"])
        self.assertEqual(engine.learned,[])

    def test_reservation_write_failure_prevents_any_dispatch(self):
        class BrokenJournal(Journal):
            def append(self,kind,payload):
                if kind == "reserve":
                    self.failed=True
                    raise OSError("invented fsync failure")
                return super().append(kind,payload)
        value,engine=runner(self.plan,journal=BrokenJournal())
        with self.assertRaisesRegex(ValueError,"Journal write failed"):
            value.run()
        self.assertEqual(engine.executed,[])
        self.assertEqual(engine.learned,[])
        self.assertEqual(value.bounds.assessment_requested,35)

    def test_journal_refuses_reuse_and_keeps_contiguous_hash_bound_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"events"
            with mock.patch("location_comparison_runtime._sync_directory") as sync:
                journal=Journal(path)
                sync.assert_called_once_with(path.parent)
            first=journal.append("example",dict(synthetic=True))
            second=journal.append("example",dict(synthetic=True))
            self.assertEqual(second["previous"],first["fingerprint"])
            self.assertEqual(json.loads((path/"event-000001.json").read_text()),second)
            with self.assertRaises(FileExistsError):
                Journal(path)


if __name__ == "__main__":
    unittest.main()
