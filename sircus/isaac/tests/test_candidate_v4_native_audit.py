"""Fabricated native packets and readouts only; no simulation or model loads."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_v4_native_audit as subject
import candidate_continuation_assessment_audit as original
from candidate_continuation_journal import Journal,write_new_json
from candidate_continuation_ledger import canonical_hash
from settled_transit import protocol as setup_protocol
from test_candidate_v4_evaluation import fixture
from test_candidate_v4_model_check import model_receipt


class NativeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prepared=fixture()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.plan=copy.deepcopy(self.prepared)
        self.native=self.root/"native";self.native.mkdir()
        self.scene={"synthetic":True};self.cells=self.plan["original_plan"]["cells_by_split"]["train"]
        old=dict(schema="sircus-candidate-continuation-native-binding-v1",source_sha256=self.plan["bindings"]["producer_source_sha256"],
            guard_source_sha256={"unchanged.py":"e"*64},scene_sha256=canonical_hash(self.scene),cells_sha256=canonical_hash(self.cells),
            startup_control_sha256={"probes.json":"1"*64,"rim-rays.json":"2"*64},execution_authorized=False,release_authorized=False)
        old["fingerprint"]=canonical_hash(old);self.binding=copy.deepcopy(old)
        self.binding.update(source_sha256=self.plan["bindings"]["evaluation_source_sha256"],startup_control_sha256={"probes.json":"3"*64,"rim-rays.json":"4"*64})
        self.binding.pop("fingerprint");self.binding["fingerprint"]=canonical_hash(self.binding)
        self.model=model_receipt(self.plan,original_binding=old)
        self.model_path=self.root/"model-check.json";write_new_json(self.model_path,self.model)
        write_new_json(self.root/"evaluation-plan.json",self.plan)
        self.controls=dict(mock_controls=True);self.packet_calls=0
        self.journal=Journal(self.root/"journal")
        self.frozen=dict(final_checkpoint_sha256=self.plan["final_checkpoint_sha256"],parent_checkpoint_sha256=self.plan["parent_checkpoint_sha256"],
            bindings=self.plan["bindings"],model_hashes=self.plan["model_hashes"])
        self.journal.append("start",dict(plan_fingerprint=self.plan["fingerprint"],synthetic=True,mock_only=True,frozen=self.frozen))
        self.rows=[];by_id={c["id"]:c for c in self.plan["original_plan"]["cells_by_split"]["test"]}
        for cohort in range(6):
            choices=self.plan["cohorts"][cohort]["choices"];old_choices=self.plan["original_plan"]["choices"][(cohort+6)*35:(cohort+7)*35]
            event=self.journal.append("reserve",dict(cohort=cohort,original_cohort=cohort+6,choices=choices,
                requested=(cohort+1)*35,terminal=len(self.rows),plan_fingerprint=self.plan["fingerprint"]))
            stem=f"evaluation-{cohort:06d}";physics=self.native/(stem+".npz");physics.write_bytes(b"fabricated-physics-"+str(cohort).encode())
            rows=[];returns=[]
            for i,choice in enumerate(old_choices):
                released=choice["enabled"] and i%5!=0;capture=released and i%3==0;clean=capture and i%2==0
                invalid=None if released else choice["decline_reason"] or "synthetic_guard_decline"
                rows.append(dict(split="test",invalid=invalid,native_invalid=invalid if choice["enabled"] else "no_selected_setting",
                    success=capture,route_success=capture,route_quality=2 if clean else 1 if capture else 0,
                    qualification=dict(accepted=released) if choice["enabled"] else dict(accepted=False,reason="inactive_split"),path_proxy=None,path_rejection=None))
                returns.append(1.6 if clean else 1.5 if capture else -.5 if released else -1.)
            cells=[dict(copy.deepcopy(self.cells[i]),**copy.deepcopy(by_id[c["context_id"]])) for i,c in enumerate(old_choices)]
            packet=dict(schema=original.PACKET_SCHEMA,mode="assessment",batch=cohort+6,cohort=cohort+6,split="test",method=choices[0]["method"],synthetic=True,
                reservation_sha256=event["fingerprint"],assessment_plan_fingerprint=self.plan["original_plan"]["fingerprint"],native_binding=self.binding,
                model_hashes=self.plan["model_hashes"],final_checkpoint_sha256=self.plan["final_checkpoint_sha256"],cells=cells,choices=old_choices,
                request_ids=[c["request_id"] for c in old_choices],actions=[c["action"] for c in old_choices],enabled=[c["enabled"] for c in old_choices],
                decline_reasons=[c["decline_reason"] for c in old_choices],proposal_ik_seeds=[c["ik_seeds"] for c in old_choices],
                rows=rows,rewards=returns,reward_features=[{} if not r["invalid"] else None for r in rows],preflight_records=[None]*35,
                physics=dict(file=physics.name,sha256=subject.digest(physics)),setup_controller=setup_protocol(),startup_controls=self.controls,
                optimizer_updates=0,wall_s=1.,process_cpu_s=2.)
            self.write_packet(cohort,packet)
            packet_path=self.native/(stem+".packet.json.gz")
            envelope=dict(schema=subject.ENVELOPE_SCHEMA,plan_fingerprint=self.plan["fingerprint"],reservation_sha256=event["fingerprint"],
                new_cohort=cohort,original_cohort=cohort+6,new_choices=choices,original_choices=old_choices,
                packet=dict(file=packet_path.name,sha256=subject.digest(packet_path)),native_binding_fingerprint=self.binding["fingerprint"],
                final_checkpoint_sha256=self.plan["final_checkpoint_sha256"],synthetic=True)
            envelope["fingerprint"]=canonical_hash(envelope);envelope_path=self.native/(stem+".envelope.json");write_new_json(envelope_path,envelope)
            write_new_json(self.native/(stem+".dispatch.json"),dict(schema="sircus-v4-test-native-dispatch-v1",plan_fingerprint=self.plan["fingerprint"],
                reservation_sha256=event["fingerprint"],new_cohort=cohort,original_cohort=cohort+6,new_request_ids=[c["request_id"] for c in choices],
                original_request_ids=[c["request_id"] for c in old_choices],native_binding_fingerprint=self.binding["fingerprint"],status="claimed_before_native_call"))
            terminal=self.fake_score(None,packet,self.scene)["terminal_results"]
            terminal=[dict(row,request_id=c["request_id"],source_sha256=subject.digest(envelope_path)) for c,row in zip(choices,terminal)]
            self.rows.extend(terminal);self.journal.append("outcomes",dict(cohort=cohort,results=terminal,requested=(cohort+1)*35,terminal=len(self.rows)))
        self.counts=dict(requested=210,terminal=210,released=sum(r["released"] for r in self.rows),rejected=sum(not r["released"] for r in self.rows),
            captures=sum(r["captured"] for r in self.rows),clean_captures=sum(r["clean"] for r in self.rows),unresolved=0)
        final=dict(plan_fingerprint=self.plan["fingerprint"],frozen=self.frozen,counts=self.counts,synthetic=True,mock_only=True,
            training_requests=0,optimizer_steps=0,native_requests_executed=0)
        self.journal.append("final",final)
        self.summary=dict(final,schema=subject.RESULT_SCHEMA,status="synthetic_complete_audit_pending",journal_head_sha256=self.journal.head,
            original_assessment_completed=False,model_promoted=False,execution_authorized=False,release_authorized=False)
        self.summary["fingerprint"]=canonical_hash(self.summary);write_new_json(self.root/"evaluation-result.json",self.summary)

    def write_packet(self,cohort,packet):
        stem=f"evaluation-{cohort:06d}";raw=json.dumps(packet,separators=(",",":"),allow_nan=False).encode();path=self.native/(stem+".packet.json.gz")
        path.write_bytes(gzip.compress(raw,mtime=0));receipt=dict(file=path.name,sha256=subject.digest(path),bytes=path.stat().st_size,
            uncompressed_sha256=hashlib.sha256(raw).hexdigest())
        (self.native/(stem+".receipt.json")).write_text(json.dumps(receipt))

    @staticmethod
    def fake_score(directory,packet,scene):
        terminal=[dict(request_id=choice["request_id"],released=row["invalid"] is None,decline_reason=row["invalid"],
            captured=row["route_success"],clean=row["route_quality"]==2,**{"return":reward})
            for choice,row,reward in zip(packet["choices"],packet["rows"],packet["rewards"])]
        return dict(terminal_results=terminal,requests=35,releases=sum(r["released"] for r in terminal),
            rejections=sum(not r["released"] for r in terminal),captures=sum(r["captured"] for r in terminal),clean_captures=sum(r["clean"] for r in terminal))

    def packet_check(self,directory,packet,**kwargs):
        self.packet_calls+=1
        # Exercise the actual strict native disabled-label/slot adapter; only
        # its already separately tested lower guard/scorer dependencies are mocks.
        with patch.object(original,"verify_guards",return_value=dict(mock_guard=True)),patch.object(original,"verify_score",side_effect=self.fake_score):
            return original.verify_assessment_packet(directory,packet,**kwargs)

    def audit(self,**changes):
        checks=dict(packet=self.packet_check,controls=lambda *a,**k:self.controls,
            startup=lambda *a,**k:dict(status="passed",startup_control_sha256=self.binding["startup_control_sha256"],synthetic_fixture=True))
        args=dict(plan=self.plan,summary=self.summary,expected_head_sha256=self.journal.head,
            expected_result_sha256=subject.digest(self.root/"evaluation-result.json"),model_receipt_path=self.model_path,
            expected_model_receipt_sha256=subject.digest(self.model_path),native_binding=self.binding,scene=self.scene,original_cells=self.cells,
            startup_directory=self.root/"native-setup",startup_cells=self.cells,synthetic=True,checks=checks)
        args.update(changes);return subject.audit_native_evaluation(self.root,**args)

    def test_complete_six_native_packet_joins_keep70_per_method_and_four_empty_failures(self):
        before={str(p):subject.digest(p) for p in self.root.rglob("*") if p.is_file()}
        result=self.audit();self.assertEqual(self.packet_calls,6)
        self.assertEqual(result["counts"]["requested"],210);self.assertEqual(len(result["batches"]),6)
        self.assertEqual(result["status"],"passed_synthetic");self.assertFalse(result["full_native_audit"])
        for method in subject.METHODS:self.assertEqual(result["comparison"]["methods"][method]["requested"],70)
        self.assertEqual(result["comparison"]["methods"]["continued"]["rejection_reasons"]["no_qualified_candidate_support"],2)
        self.assertEqual(result["comparison"]["methods"]["candidate_bank"]["rejection_reasons"]["no_qualified_candidate_support"],2)
        self.assertFalse(result["original_assessment_completed"]);self.assertEqual(result["optimizer_steps"],0)
        self.assertEqual(before,{p:subject.digest(Path(p)) for p in before})

    def test_real_audit_cannot_use_injected_synthetic_dependencies(self):
        with self.assertRaisesRegex(ValueError,"provenance"):self.audit(synthetic=False)

    def test_mutated_physics_file_rejected_before_scorer(self):
        (self.native/"evaluation-000000.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"physics bytes"):self.audit()
        self.assertEqual(self.packet_calls,0)

    def test_envelope_hash_cannot_change_original_slot_or_new_command(self):
        p=self.native/"evaluation-000000.envelope.json";value=json.loads(p.read_bytes())
        value["new_choices"][0]["action"]["gap_m"]+=.001;value.pop("fingerprint");value["fingerprint"]=canonical_hash(value);p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"envelope"):self.audit()

    def test_packet_byte_receipt_and_dispatch_are_independently_bound(self):
        p=self.native/"evaluation-000000.receipt.json";value=json.loads(p.read_bytes());value["bytes"]+=1;p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"compressed packet receipt"):self.audit()

    def test_failed_partial_or_repeated_native_artifacts_never_pass(self):
        p=self.native/"evaluation-000006.failure.json";p.write_text("{}")
        with self.assertRaisesRegex(ValueError,"Unlisted"):self.audit()
        p.unlink();(self.root/"application-failure.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"Failed or interrupted"):self.audit()

    def test_changed_model_proof_or_guard_bank_identity_rejected(self):
        value=copy.deepcopy(self.binding);value["guard_source_sha256"]={"changed.py":"0"*64};value.pop("fingerprint");value["fingerprint"]=canonical_hash(value)
        with self.assertRaisesRegex(ValueError,"guard"):self.audit(native_binding=value)

    def test_incomplete_journal_cannot_receive_native_pass(self):
        (self.root/"journal"/"event-000013.json").unlink()
        with self.assertRaisesRegex(ValueError,"head changed"):self.audit()

    def test_wrong_native_empty_label_rejected_by_actual_original_boundary_adapter(self):
        cohort=0;p=self.native/"evaluation-000000.packet.json.gz";value=json.loads(gzip.decompress(p.read_bytes()))
        index=next(i for i,c in enumerate(value["choices"]) if not c["enabled"])
        value["rows"][index]["native_invalid"]="inactive_split";self.write_packet(cohort,value)
        envelope_path=self.native/"evaluation-000000.envelope.json";e=json.loads(envelope_path.read_bytes());e["packet"]["sha256"]=subject.digest(p)
        e.pop("fingerprint");e["fingerprint"]=canonical_hash(e);envelope_path.write_text(json.dumps(e))
        with self.assertRaisesRegex(ValueError,"Empty-support"):self.audit()

    def test_terminal_claims_cannot_replace_independent_packet_scoring(self):
        events=[json.loads(p.read_bytes()) for p in sorted((self.root/"journal").iterdir())]
        value=events[2]["payload"]["results"][0];value.update(released=True,decline_reason=None,**{"return":-.5})
        head="0"*64
        for seq,event in enumerate(events):
            event["previous_sha256"]=head;event.pop("fingerprint");event["fingerprint"]=canonical_hash(event);head=event["fingerprint"]
            (self.root/"journal"/f"event-{seq:06d}.json").write_text(json.dumps(event))
        self.journal.head=head;self.summary["journal_head_sha256"]=head;self.summary.pop("fingerprint");self.summary["fingerprint"]=canonical_hash(self.summary)
        (self.root/"evaluation-result.json").write_text(json.dumps(self.summary))
        with self.assertRaises(ValueError):self.audit()

    def test_startup_and_model_proof_are_both_required_before_any_batch(self):
        checks=dict(packet=self.packet_check,controls=lambda *a,**k:self.controls,startup=lambda *a,**k:dict(status="failed"))
        with self.assertRaisesRegex(ValueError,"startup"):self.audit(checks=checks)
        self.assertEqual(self.packet_calls,0)


if __name__=="__main__":unittest.main()
