import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from sircus.mjlab.parallel_board_plan import _Lane, select_final, supervise_parallel


def _choice(score, margin, identifier="original"):
    return dict(id=identifier,request=dict(id=identifier),prediction=dict(score_m=score,
        predicted_route="direct",direct=dict(arm_clearance_m=margin)))


def _snapshot(layout,method,choice,stage="finished",alternate=False):
    value = dict(layout=layout,method=method,choice=choice,active_stage=stage,
        completed_proposal_ids=[choice["id"]],ranked_proposal_ids=[choice["id"]],generated={},
        plan={key:dict(proposals=[],admitted=[],records=[]) for key in ("base","fallback","neural","branches")})
    if alternate:
        value["board_alternative"] = dict(selected_candidate_id=choice["id"])
    return value


def _original(layout,method,script,*,publish,deadline,profile):
    for offset,score,margin in script:
        time.sleep(max(0.,profile.started+offset-time.perf_counter()))
        with profile.stage("arm_setup_checks"):
            publish(_snapshot(layout,method,_choice(score,margin,f"original-{offset}")))
    return "completed"


def _initialize_sleep(delay):
    time.sleep(delay)


def _alternate(layout,method,source,config,*,publish,deadline,profile):
    time.sleep(max(0.,profile.started+config.get("at",.3)-time.perf_counter()))
    if config.get("error"):
        raise RuntimeError("Deliberate alternate worker failure")
    with profile.stage("arm_setup_checks"):
        publish(_snapshot(layout,method,_choice(config.get("score",.03),config.get("margin",.02),"alternate"),alternate=True))
    return "completed"


class SelectionTests(unittest.TestCase):
    def test_recheck_against_improved_original_not_trigger(self):
        trigger = _choice(-.02,-.01)
        original = _choice(.04,.01)
        alternate = _choice(.03,.02,"alternate")
        self.assertEqual(select_final(trigger,alternate)["selected_lane"],"alternative")
        self.assertEqual(select_final(original,alternate)["selected_lane"],"original")

    def test_unknown_negative_tied_or_errored_alternate_keeps_original(self):
        original = _choice(.01,.01)
        for candidate in (None,_choice(.02,-.01),_choice(.02,None),_choice(.01,.02)):
            self.assertEqual(select_final(original,candidate)["selected_lane"],"original")
        self.assertEqual(select_final(original,_choice(.1,.1),worker_error=True)["selected_lane"],"original")
        empty = dict(request=None,prediction=None)
        self.assertEqual(select_final(empty,_choice(.1,.1))["selected_lane"],"original")


class SpawnedLifecycleTests(unittest.TestCase):
    def run_case(self,script,config=None,*,budget=1.5,init_delay=0.,factory_delay=0.):
        config = {} if config is None else config
        with tempfile.TemporaryDirectory(prefix="parallel-board-test-") as temporary:
            root = Path(temporary)
            layout = dict(id="cpu-development",bucket_xy_m=[1.,0.],stratum="test",split="test")
            def factory(source,original_dir,alternate_dir):
                time.sleep(factory_delay)
                return dict(function=_alternate,common_args=(source,config),
                            initializer=_initialize_sleep,initargs=(init_delay,))
            result = supervise_parallel(layout,"warm_early",(script,),directory=root,
                original_function=_original,alternative_factory=factory,budget_s=budget,startup_timeout_s=10.)
            for lane in result["lanes"].values():
                if lane is None:
                    continue
                self.assertFalse(lane["worker_reused"])
                self.assertEqual(lane["lifecycle"],"fresh_worker_per_request")
                for receipt in lane["accepted_receipts"]:
                    self.assertLess(receipt["validated_elapsed_s"],budget)
                    raw=(Path(lane["journal_directory"])/receipt["file"]).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(),receipt["sha256"])
                self.assertIsNotNone(lane["worker_exitcode"])
            return result

    def test_clear_original_never_starts_second_process(self):
        result = self.run_case([(.02,.02,.02)])
        self.assertIsNone(result["trigger"])
        self.assertIsNone(result["lanes"]["alternative"])
        self.assertEqual(result["selected_lane"],"original")

    def test_two_lanes_overlap_and_original_improvement_is_preserved(self):
        result = self.run_case([(.03,-.02,-.01),(.5,.04,.02)],dict(at=.35,score=.03))
        primary,secondary = result["lanes"]["original"],result["lanes"]["alternative"]
        self.assertNotEqual(primary["worker_pid"],secondary["worker_pid"])
        self.assertLess(secondary["launch_offset_s"],primary["decision_latency_s"])
        self.assertGreaterEqual(secondary["launch_offset_s"],result["trigger"]["elapsed_s"])
        self.assertEqual(result["selected_lane"],"original")
        self.assertEqual(result["choice"]["prediction"]["score_m"],.04)
        self.assertFalse(result["shared_clock"]["original_interrupted_by_alternative"])

    def test_qualified_alternate_can_win_after_original_completes(self):
        result = self.run_case([(.03,-.02,-.01)],dict(at=.3,score=.03))
        self.assertEqual(result["selected_lane"],"alternative")
        self.assertEqual(result["choice"]["id"],"alternate")
        self.assertEqual(result["selected_adapter_directory"],"alternative")
        self.assertGreater(result["lanes"]["alternative"]["startup_s"],0.)
        self.assertGreater(result["cpu_s_last_receipts"],0.)

    def test_alternate_initialization_uses_remaining_deadline(self):
        result = self.run_case([(.02,-.02,-.01)],budget=.35,init_delay=1.)
        self.assertEqual(result["selected_lane"],"original")
        self.assertEqual(result["decision_latency_s"],.35)
        alternate = result["lanes"]["alternative"]
        self.assertEqual(alternate["status"],"deadline")
        self.assertIsNone(alternate["readiness_receipt"])
        self.assertEqual(alternate["accepted_receipts"],[])

    def test_late_alternate_snapshot_never_enters_selection(self):
        result = self.run_case([(.02,-.02,-.01)],dict(at=1.,score=.9),budget=.4)
        self.assertEqual(result["selected_lane"],"original")
        self.assertEqual(result["lanes"]["alternative"]["accepted_receipts"],[])
        self.assertIsNone(result["selection"]["alternative_choice_sha256"])

    def test_slow_factory_does_not_launch_after_cutoff(self):
        result = self.run_case([(.02,-.02,-.01)],budget=.2,factory_delay=.3)
        self.assertIsNotNone(result["trigger"])
        self.assertIsNone(result["lanes"]["alternative"])
        self.assertEqual(result["selected_lane"],"original")

    def test_original_deadline_keeps_last_checked_original(self):
        result = self.run_case([(.02,-.02,-.01),(1.,.9,.1)],dict(at=.25,margin=-.01),budget=.4)
        self.assertEqual(result["lanes"]["original"]["status"],"deadline")
        self.assertEqual(result["selected_lane"],"original")
        self.assertEqual(result["choice"]["prediction"]["score_m"],-.02)

    def test_alternate_error_preserves_original_without_restarting(self):
        result = self.run_case([(.02,-.02,-.01)],dict(at=.2,error=True))
        self.assertEqual(result["selected_lane"],"original")
        self.assertTrue(result["selection"]["worker_error"])
        self.assertEqual(result["lanes"]["alternative"]["status"],"error")
        self.assertEqual(result["lanes"]["original"]["worker_generation"],0)


class ReceiptIntegrityTests(unittest.TestCase):
    def test_changed_snapshot_fails_closed(self):
        from sircus.mjlab.planning_budget_plan import _json_bytes,_sha
        with tempfile.TemporaryDirectory() as directory:
            lane = _Lane.__new__(_Lane)
            lane.journal = Path(directory)
            lane.job_id = "cpu-job"
            lane.ready = {"kind":"ready"}
            lane.receipts = []
            lane.cpu_s = lane.verification_s = 0.
            lane.snapshot = None
            candidate = dict(choice=_choice(.02,.02),completed_proposal_ids=["original"])
            valid = _json_bytes(candidate)
            (lane.journal/"snapshot-00000.json").write_text("{}")
            message = dict(kind="snapshot",job_id=lane.job_id,sequence=0,file="snapshot-00000.json",
                sha256=hashlib.sha256(valid).hexdigest(),choice_sha256=_sha(candidate["choice"]),
                completed_proposals=1,cpu_s=.01,elapsed_s=.01)
            lane.connection = SimpleNamespace(recv_bytes=lambda limit:_json_bytes(message))
            with self.assertRaises(ValueError):
                lane.receive(time.perf_counter()-.1,time.perf_counter()+1.)
            self.assertEqual(lane.receipts,[])
            self.assertIsNone(lane.snapshot)


if __name__ == "__main__":
    unittest.main()
