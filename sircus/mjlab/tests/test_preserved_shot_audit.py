"""Fail-closed evidence checks for preservation and matched native releases."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from sircus.mjlab import preserved_shot_audit as audit
from sircus.mjlab.preserved_shot_plan import select_pair,slots_for_pair
from sircus.mjlab.benchmark_verify import Checks, EvidenceError, sha, digest
from sircus.mjlab.sweep_worker import atomic


def fixture(expanded_has_request=False,repeats=3):
    layout=dict(id="p00",bucket_xy_m=[.55,-.2],stratum="cell",split="test")
    def site(source):
        request=dict(id="p00-warm_early",method="warm_early",context_id="p00",bucket_xy_m=layout["bucket_xy_m"],
            values=[.4,-.1,.25,30.,.3],selected_q10=[1.,2.],selected_q14=[3.,4.],
            context_resolution_file="direct-contexts/anchor.json",context_resolution_sha256="a"*64,
            prepared_connection_file="connections/connection.json",prepared_connection_sha256="b"*64,
            native_prepared=True)
        if source=="expanded" and not expanded_has_request:request=None
        choice=dict(id="p00-warm_early",method="warm_early",context_id="p00",bucket_xy_m=layout["bucket_xy_m"],
                    stratum="cell",request=request,prediction={"risk":source})
        return dict(layout=layout,status="completed",method="warm_early",choice=choice,
                    request_wall_s=1.5 if source=="current" else 2.0,decision_latency_s=1.4 if source=="current" else 1.9)
    current,expanded=site("current"),site("expanded")
    timing=dict(pair_started_s=10.,pair_finished_s=12.5,
                lanes=dict(current=dict(started_s=10.1,finished_s=11.7),expanded=dict(started_s=10.2,finished_s=12.3)))
    selection=select_pair(current,expanded,timing)
    plans=[dict(variant="current",site=current),dict(variant="expanded",site=expanded)]
    slots=slots_for_pair(current,expanded,selection,repeats)
    return [layout],plans,[selection],slots


class PreservedShotAuditTests(unittest.TestCase):
    def test_current_checked_fallback_retains_every_physical_field(self):
        layouts,plans,selections,slots=fixture()
        checks=Checks()
        audit.verify_slots(slots,plans,selections,layouts,3,checks)
        self.assertGreater(checks.count,10)
        self.assertEqual([s["source_method"] for s in slots],["current"]*6)
        self.assertEqual([s["repeat_index"] for s in slots],[0,1,2,0,1,2])
        self.assertEqual(audit.physical_request(slots[0]["request"]),audit.physical_request(slots[3]["request"]))

    def test_expanded_admission_is_used_without_consulting_capture_flags(self):
        layouts,plans,selections,slots=fixture(True)
        audit.verify_slots(slots,plans,selections,layouts,3,Checks())
        self.assertEqual([s["source_method"] for s in slots],["current"]*3+["expanded"]*3)
        for item in plans:item["site"]["later_capture"]=item["variant"]=="current"
        audit.verify_slots(slots,plans,selections,layouts,3,Checks())

    def test_changed_gap_joint_proof_or_arbitrary_request_field_fails(self):
        for field in ("values","selected_q10","prepared_connection_sha256","native_prepared","extra"):
            with self.subTest(field=field):
                layouts,plans,selections,slots=fixture()
                slots[3]["request"][field]="changed"
                with self.assertRaises(EvidenceError):audit.verify_slots(slots,plans,selections,layouts,3,Checks())

    def test_wrong_source_even_with_same_five_values_fails(self):
        layouts,plans,selections,slots=fixture(True)
        slots[3]["source_method"]="current"
        with self.assertRaises(EvidenceError):audit.verify_slots(slots,plans,selections,layouts,3,Checks())

    def test_missing_repeat_and_duplicate_identity_fail(self):
        for modify in (lambda s:s.pop(),lambda s:s[3].update(id=s[0]["id"])):
            layouts,plans,selections,slots=fixture()
            modify(slots)
            with self.assertRaises(EvidenceError):audit.verify_slots(slots,plans,selections,layouts,3,Checks())

    def test_invented_zero_fallback_latency_fails(self):
        layouts,plans,selections,slots=fixture()
        selections[0]["timing"]["pair_wall_s"]=0.
        with self.assertRaises(EvidenceError):audit.verify_slots(slots,plans,selections,layouts,3,Checks())

    def test_shortened_lane_interval_that_hides_work_fails(self):
        layouts,plans,selections,slots=fixture()
        selections[0]["timing"]["lanes"]["current"]["finished_s"]=10.2
        with self.assertRaises(EvidenceError):audit.verify_slots(slots,plans,selections,layouts,3,Checks())

    def test_worker_error_is_not_an_ordinary_decline(self):
        layouts,plans,selections,slots=fixture()
        plans[1]["site"]["status"]="error"
        with self.assertRaises(EvidenceError):audit.verify_slots(slots,plans,selections,layouts,3,Checks())

    def test_schedule_keeps_source_and_result_lanes_distinct(self):
        _,_,_,slots=fixture(True)
        contract=dict(processing_order=["p00"],attempts_per_method_per_position=3)
        packets=audit.expected_schedule(contract,slots)
        self.assertEqual([(p["method"],p["source_method"],p["adapter"],len(p["request_ids"])) for p in packets],
                         [("current","current","direct_first",3),("protected","expanded","direct_first",3)])
        slots[-1]=deepcopy(slots[0])
        with self.assertRaises(ValueError):audit.expected_schedule(contract,slots)

    def test_source_proof_namespace_cannot_be_silently_swapped(self):
        _,_,_,slots=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            proofs={}
            for prefix in ("context_resolution","prepared_connection"):
                relative="native-current/"+slots[0]["request"][prefix+"_file"]
                atomic(root/relative,dict(checked=True,source="current"))
                checksum=sha(root/relative)
                proofs[relative]=checksum
                for slot in slots:slot["request"][prefix+"_sha256"]=checksum
            atomic(root/"admission-proof-bindings.json",proofs)
            audit.verify_admission_proofs(root,slots,Checks())
            # An expanded proof at the same relative path is never an adequate
            # substitute for the source selected by the portfolio.
            slots[3]["source_method"]="expanded"
            with self.assertRaises((EvidenceError,FileNotFoundError)):
                audit.verify_admission_proofs(root,slots,Checks())


    def test_independent_counts_keep_declines_and_invalid_repeats_in_denominator(self):
        from sircus.mjlab.preserved_shot_study import summarize
        layouts,plans,selections,slots=fixture(True)
        slots[0]["request"]=None
        rows=[]
        for i,slot in enumerate(slots):
            if slot["request"] is not None:
                rows.append(dict(id=slot["id"],adapter_directory="native-"+slot["source_method"],
                    board_first_score=dict(route_success=i in (1,3,4),invalid_physics=i==2)))
        result=audit.independent_summary(layouts,slots,rows,plans,selections)
        self.assertEqual(result,summarize(layouts,slots,rows,plans,selections))
        self.assertEqual(result["methods"]["current"]["capture_rate"],1/3)
        self.assertEqual(result["methods"]["current"]["setup_rejections"],1)
        self.assertEqual(result["methods"]["current"]["invalid_physics"],1)
        self.assertEqual(result["methods"]["protected"]["flight_misses"],1)
        self.assertEqual(result["paired_first_drop"],dict(both=0,current_only=0,protected_only=1,neither=0))
        with self.assertRaises(ValueError):audit.independent_summary(layouts,slots,rows+rows[:1],plans,selections)

    def test_preflight_cannot_accept_a_final_audit_or_modified_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/"study";study.mkdir()
            for name in ("contract.json","current-checkpoint.json","expanded-checkpoint.json","proof.json"):
                atomic(study/name,dict(name=name))
            bindings={p.name:sha(p) for p in study.iterdir()}
            atomic(study/"choices.json",dict(slots=[],bindings=bindings))
            expected=dict(bindings,**{"choices.json":sha(study/"choices.json")})
            receipt=dict(version=audit.VERSION,stage="preflight",status="passed",
                contract_sha256=sha(study/"contract.json"),choices_sha256=sha(study/"choices.json"),
                preparation_manifest_sha256=digest(expected),
                checkpoint_sha256={m:sha(study/(m+"-checkpoint.json")) for m in audit.RAW_METHODS})
            path=root/"preflight.json";atomic(path,receipt)
            args=SimpleNamespace(output=study,preflight_output=path)
            self.assertEqual(audit.require_preflight(args),receipt)
            atomic(path,dict(receipt,stage="audit"))
            with self.assertRaises(ValueError):audit.require_preflight(args)
            atomic(path,receipt)
            atomic(study/"proof.json",dict(changed=True))
            with self.assertRaises(ValueError):audit.require_preflight(args)

    def test_schedule_declines_remain_slots_but_never_produce_hidden_drops(self):
        _,_,_,slots=fixture(True)
        slots[3]["request"]=None
        packets=audit.expected_schedule(dict(processing_order=["p00"],attempts_per_method_per_position=3),slots)
        released=[identity for packet in packets for identity in packet["request_ids"]]
        self.assertEqual(len(released),5)
        self.assertNotIn(slots[3]["id"],released)
        self.assertEqual(len(slots),6)


if __name__=="__main__":unittest.main()
