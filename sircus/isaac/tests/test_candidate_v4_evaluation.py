"""Synthetic-only V4 TEST preparation, exposure and non-replacement checks."""

import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_v4_evaluation as subject
import candidate_continuation_assessment as old
from candidate_continuation_contract import JOURNAL_SCHEMA
from candidate_continuation_ledger import Outcome, canonical_hash
from test_candidate_continuation_assessment import complete_train_ledger, assessment_inputs


def fixture():
    """Fabricated metadata/actions/accounting. No real checkpoint is loaded."""
    train=complete_train_ledger()
    cells,supports,records,parent,continued=assessment_inputs(train)
    with patch.object(old,"source_fingerprint",return_value="a"*64):
        original=old.build_assessment_plan(train,final_checkpoint_sha256=subject.V4_CHECKPOINT_SHA256,
            cells_by_split=cells,supports=supports,prepared_records=records,parent_distributions=parent,
            continued_distributions=continued,model_hashes=dict(parent_weight_hash="c"*64,continued_weight_hash="d"*64),
            bindings=dict(source_sha256="a"*64,producer_source_sha256="b"*64,native_binding_sha256="e"*64))
    ledger=train.begin_assessment(subject.V4_CHECKPOINT_SHA256)
    events=[]
    def append(kind,payload):
        v=dict(schema=JOURNAL_SCHEMA,seq=len(events),previous_sha256=events[-1]["fingerprint"] if events else "0"*64,
            kind=kind,payload=copy.deepcopy(payload));v["fingerprint"]=canonical_hash(v);events.append(v)
    append("start",dict(plan_fingerprint=original["fingerprint"],ledger=ledger.to_dict(),
        final_checkpoint_sha256=subject.V4_CHECKPOINT_SHA256,model_hashes=original["model_hashes"]))
    for cohort in range(4):
        batch=ledger.next_batch();ledger=ledger.reserve_batch(batch)
        append("reserve",dict(ledger=ledger.to_dict(),choices=original["choices"][cohort*35:(cohort+1)*35]))
        if cohort < 3:
            rows=[dict(request_id=r.request_id,released=i>=3,decline_reason=None if i>=3 else "synthetic_decline",
                source_sha256="f"*64) for i,r in enumerate(batch.requests)]
            ledger=ledger.record_outcomes(tuple(Outcome(**r) for r in rows))
            append("outcomes",dict(ledger=ledger.to_dict(),results=rows))
    ledger=ledger.halt("synthetic closed failure")
    append("failure",dict(ledger=ledger.to_dict(),reason="synthetic closed failure"))
    names={"assessment-plan.json",*(f"journal/event-{i:06d}.json" for i in range(9))}
    for i in range(4):
        names.update(f"native/assessment-{i:06d}"+s for s in
            ((".dispatch.json",".json.gz",".receipt.json",".npz") if i<3 else (".dispatch.json",".failure.json",".npz")))
    inventory={n:dict(sha256="a"*64,bytes=1) for n in sorted(names)}
    evidence=dict(events=events,inventory=inventory,inventory_sha256=canonical_hash(inventory),
        original_plan_sha256="a"*64,journal_head_sha256=events[-1]["fingerprint"],
        failure_sha256="d"*64,component_summary_sha256="e"*64)
    return subject.prepare_evaluation(original,evidence,evaluation_source_sha256="f"*64,synthetic=True)


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=fixture()

    def test_exact_untouched_test_suffix_and_four_empty_support_declines(self):
        p=subject.validate_prepared(self.plan)
        self.assertEqual(p["requests"],210)
        self.assertEqual([c["original_cohort"] for c in p["cohorts"]],list(range(6,12)))
        self.assertEqual(len({(c["context_id"],c["method"]) for c in p["choices"]}),210)
        self.assertEqual(sum(not c["enabled"] for c in p["choices"]),4)
        for method in ("parent","candidate_bank","continued"):
            self.assertEqual(sum(c["method"]==method for c in p["choices"]),70)
        for row,source in zip(p["choices"],p["original_plan"]["choices"][210:]):
            changed={k:v for k,v in row.items() if k not in ("original_request_id","original_request_index","original_cohort")}
            changed.update(request_id=row["original_request_id"],request_index=row["original_request_index"],cohort=row["original_cohort"])
            self.assertEqual(changed,source)

    def test_all_bounds_and_real_execution_gates_are_explicit(self):
        p=self.plan
        self.assertEqual(p["protocol"]["phases"],dict(initialization=600,evaluation=1800,audit=600))
        self.assertEqual(p["protocol"]["wall_limit_s"],3000)
        for k in ("execution_authorized","release_authorized","native_execution_ready","original_assessment_completed","model_promoted"):
            self.assertIs(p[k],False)
        self.assertEqual(p["prior_assessment"]["unresolved"],35)
        self.assertEqual(p["independent_inference_status"],"pending_pinned_cpu_forward_reconstruction")

    def test_old_failure_is_preserved_not_reclassified_as_completed(self):
        p=subject.validate_prepared(self.plan)
        self.assertEqual(p["prior_assessment"],dict(requested=140,terminal=105,unresolved=35,
            recorded_releases=96,recorded_declines=9,uncommitted_physical_work=True,test_requests=0,status="incomplete_no_retry"))

    def test_inputs_and_validated_result_do_not_alias(self):
        p=subject.validate_prepared(self.plan)
        p["choices"][0]["action"]["board_x_m"]+=1
        self.assertNotEqual(p,self.plan)
        self.assertEqual(subject.validate_prepared(self.plan),self.plan)

    def test_modified_action_even_with_rehashed_outer_plan_rejected(self):
        p=copy.deepcopy(self.plan);p["choices"][0]["action"]["board_x_m"]+=.01
        p["fingerprint"]=canonical_hash({k:v for k,v in p.items() if k!="fingerprint"})
        with self.assertRaisesRegex(ValueError,"preserved original TEST suffix"):subject.validate_prepared(p)

    def test_different_checkpoint_is_rejected_before_any_work(self):
        p=copy.deepcopy(self.plan["original_plan"]);p["final_checkpoint_sha256"]="0"*64
        p["fingerprint"]=canonical_hash({k:v for k,v in p.items() if k!="fingerprint"})
        with self.assertRaisesRegex(ValueError,"saved V4"):
            subject.prepare_evaluation(p,self.plan["prior_evidence"],evaluation_source_sha256="f"*64,synthetic=True)

    def test_missing_or_additional_native_test_evidence_is_rejected(self):
        for mode in ("missing","test"):
            e=copy.deepcopy(self.plan["prior_evidence"])
            if mode=="missing":e["inventory"].pop("native/assessment-000003.npz")
            else:e["inventory"]["native/assessment-000006.dispatch.json"]=dict(sha256="f"*64,bytes=1)
            e["inventory_sha256"]=canonical_hash(e["inventory"])
            with self.assertRaisesRegex(ValueError,"inventory"):
                subject.prepare_evaluation(self.plan["original_plan"],e,evaluation_source_sha256="f"*64,synthetic=True)

    def test_pending_reservation_is_charged_and_cannot_be_removed(self):
        e=copy.deepcopy(self.plan["prior_evidence"]);e["events"]=e["events"][:-2]
        with self.assertRaisesRegex(ValueError,"closed 140-request prefix"):
            subject.prepare_evaluation(self.plan["original_plan"],e,evaluation_source_sha256="f"*64,synthetic=True)

    def test_real_preparation_cannot_accept_synthetic_history(self):
        with self.assertRaisesRegex(ValueError,"Actual V4 closed evidence"):
            subject.prepare_evaluation(self.plan["original_plan"],self.plan["prior_evidence"],
                evaluation_source_sha256="f"*64,synthetic=False)

    def test_guard_clearance_or_drop_gap_changes_cannot_enter_plan(self):
        for key,value in (("protocol",dict(self.plan["protocol"],retries=1)),("native_execution_ready",True)):
            p=copy.deepcopy(self.plan);p[key]=value
            p["fingerprint"]=canonical_hash({k:v for k,v in p.items() if k!="fingerprint"})
            with self.assertRaises(ValueError):subject.validate_prepared(p)


if __name__=="__main__":unittest.main()
