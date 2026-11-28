"""The preservation rule changes ownership, never physical commands or proofs."""
from copy import deepcopy
import unittest

from sircus.mjlab import preserved_shot_plan as plan
from sircus.mjlab.benchmark_verify import digest


def fixture(identity="p0", current=True, expanded=False, source="reference"):
    layout=dict(id=identity,context_id=identity,bucket_xy_m=[.5,-.2],stratum="center")
    namespace="resolutions" if source=="reference" else "direct-contexts"
    def site(admitted,offset):
        request=dict(id=identity+"-warm_early",method="warm_early",bucket_xy_m=[.5,-.2],values=[.1,-.2,.3,28.,.4+offset],
            board_heading_offset_deg=0.,board_twist_deg=135.,selected_q10=[0.]*6,selected_q14=[1.]*6,
            context_resolution_file=namespace+"/proof.json",context_resolution_sha256="a"*64,
            prepared_connection_file="connections/connection.json",prepared_connection_sha256="b"*64,
            release_provenance=dict(untouched=True),arm_gripper_angle=1.5)
        choice=dict(id=identity+"-warm_early",method="warm_early",context_id=identity,bucket_xy_m=[.5,-.2],
            request=request if admitted else None,prediction=dict(score_m=.01))
        return dict(layout=deepcopy(layout),status="completed",choice=choice,decision_latency_s=10.,request_wall_s=11.)
    timing=dict(pair_started_s=100.,pair_finished_s=112.,lanes={
        "current":dict(started_s=100.1,finished_s=111.1),"expanded":dict(started_s=100.2,finished_s=111.2)})
    return site(current,0.),site(expanded,.05),timing


class PreservedShotPlanTests(unittest.TestCase):
    def test_checked_expanded_wins_without_outcomes(self):
        current,expanded,timing=fixture(expanded=True)
        result=plan.select_pair(current,expanded,timing)
        self.assertEqual(result["selected_source"],"expanded")
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["timing"]["pair_wall_s"],12.)
        self.assertEqual(result["timing"]["reserved_worker_window_s"],20.)

    def test_decline_preserves_every_request_field_except_release_identity(self):
        current,expanded,timing=fixture(source="direct_first")
        before=deepcopy((current,expanded,timing))
        result=plan.select_pair(current,expanded,timing)
        slots=plan.slots_for_pair(current,expanded,result,3)
        self.assertEqual((current,expanded,timing),before)
        self.assertEqual(len(slots),6)
        self.assertEqual(len({s["id"] for s in slots}),6)
        for slot in slots:
            wanted=deepcopy(current["choice"]["request"])
            wanted.update(id=slot["id"],method=slot["method"])
            self.assertEqual(slot["request"],wanted)
            self.assertEqual(slot["source_method"],"current")
            self.assertEqual(slot["source_request_sha256"],digest(current["choice"]["request"]))
            self.assertEqual(slot["fallback_used"],slot["method"]=="protected")

    def test_both_decline_remains_counted_slot_without_release(self):
        current,expanded,timing=fixture(current=False)
        result=plan.select_pair(current,expanded,timing)
        slots=plan.slots_for_pair(current,expanded,result)
        self.assertTrue(all(s["request"] is None for s in slots))
        self.assertEqual(result["source_has_request"],dict(current=False,expanded=False))

    def test_request_or_source_tampering_is_rejected(self):
        current,expanded,timing=fixture()
        result=plan.select_pair(current,expanded,timing)
        changed=deepcopy(current)
        changed["choice"]["request"]["arm_gripper_angle"]+=.1
        with self.assertRaises(ValueError):plan.slots_for_pair(changed,expanded,result)
        result["selected_source"]="expanded"
        with self.assertRaises(ValueError):plan.slots_for_pair(current,expanded,result)

    def test_worker_errors_are_not_silent_fallback(self):
        current,expanded,timing=fixture()
        expanded["status"]="error"
        with self.assertRaises(ValueError):plan.select_pair(current,expanded,timing)

    def test_missing_proof_or_mismatched_layout_is_rejected(self):
        current,expanded,timing=fixture()
        current["choice"]["request"].pop("prepared_connection_sha256")
        with self.assertRaises(ValueError):plan.select_pair(current,expanded,timing)
        current,expanded,timing=fixture()
        expanded["layout"]["bucket_xy_m"]=[.7,-.2]
        with self.assertRaises(ValueError):plan.select_pair(current,expanded,timing)

    def test_pair_wall_cannot_hide_later_lane(self):
        current,expanded,timing=fixture()
        timing["pair_finished_s"]=111.
        with self.assertRaises(ValueError):plan.select_pair(current,expanded,timing)

    def test_repeats_must_be_positive_integer(self):
        current,expanded,timing=fixture()
        selected=plan.select_pair(current,expanded,timing)
        for repeats in (0,-1,True,1.5):
            with self.assertRaises(ValueError):plan.slots_for_pair(current,expanded,selected,repeats)


if __name__=="__main__":unittest.main()
