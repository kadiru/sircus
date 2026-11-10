"""Fake-only adapter checks against the original native shot_batch boundary."""

import ast
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_v4_native as subject
from candidate_continuation_journal import Journal,write_new_json
from candidate_continuation_ledger import canonical_hash
from test_candidate_v4_evaluation import fixture
from test_candidate_continuation_assessment import FakeNativeEnv


def model_receipt(plan):
    value=dict(schema="sircus-v4-independent-model-check-v1",status="passed_synthetic",synthetic=True,
        plan_fingerprint=plan["fingerprint"],original_plan_fingerprint=plan["original_plan"]["fingerprint"],
        final_checkpoint_sha256=plan["final_checkpoint_sha256"],parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
        model_hashes=plan["model_hashes"],evaluation_source_sha256=plan["bindings"]["evaluation_source_sha256"],
        original_source_sha256=plan["bindings"]["producer_source_sha256"],cpu_only=True,cuda_visible_devices="",
        source_and_input_bytes_unchanged=True,original_guard_source_sha256={"fake-source.py":"f"*64},
        commands_reconstructed=420,test_commands_verified=210,optimizer_steps=0,native_requests_executed=0,
        model_promoted=False,execution_authorized=False,release_authorized=False,original_assessment_completed=False,
        checkpoint_bindings_sha256="c"*64,forward_readouts_sha256="d"*64)
    binding=dict(source_sha256=value["original_source_sha256"],guard_source_sha256=value["original_guard_source_sha256"])
    binding["fingerprint"]=canonical_hash(binding);value["original_native_binding"]=binding
    value["fingerprint"]=canonical_hash(value);return value


class NativeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prepared=fixture()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.plan=copy.deepcopy(self.prepared)
        self.env=FakeNativeEnv(self.root/"startup",self.plan["original_plan"]["cells_by_split"]["train"])
        self.checked=0
        def check():self.checked+=1
        self.holder=types.SimpleNamespace(env=self.env,binding=dict(source_sha256=self.plan["bindings"]["evaluation_source_sha256"],
            fingerprint="b"*64,startup_control_sha256={}),original_cells=copy.deepcopy(self.env.base_cells),
            frozen_root=self.root.parent/("frozen-"+self.root.name),_validate_binding=check)
        control=patch("candidate_native_audit.verify_controls",return_value=dict(synthetic=True));control.start();self.addCleanup(control.stop)
        self.journal=Journal(self.root/"journal");self.journal.append("start",dict(synthetic=True))
        self.producer=subject.GuardedTestProducer(self.holder,self.plan,self.root,frozen_guard=check,synthetic=True)
    def reserve(self,index):
        c=self.plan["cohorts"][index]
        event=self.journal.append("reserve",dict(cohort=index,original_cohort=index+6,choices=c["choices"],
            requested=(index+1)*35,terminal=index*35,plan_fingerprint=self.plan["fingerprint"]))
        return c,copy.deepcopy(c["choices"]),dict(journal_head_sha256=event["fingerprint"],cohort=index,
            original_cohort=index+6,requested=(index+1)*35,request_ids=[x["request_id"] for x in c["choices"]])
    def execute(self,index):
        c,choices,res=self.reserve(index)
        return self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)

    def test_all210_original_batch_mock_slots_preserve_four_empty_declines(self):
        rows=[]
        for i in range(6):rows+=self.execute(i)
        self.assertEqual(len(rows),210)
        self.assertEqual(sum(r["decline_reason"]=="no_qualified_candidate_support" for r in rows),4)
        self.assertTrue(all(not r["released"] for r in rows));self.assertEqual(self.env.tick_calls,0)
        self.assertEqual(self.env.calls,["home","shot"]*6)
        self.assertGreaterEqual(self.checked,2+6*6)

    def test_raw_inner_old_ids_and_outer_new_ids_have_exact_hash_join(self):
        rows=self.execute(0);native=self.root/"native"
        envelope=json.loads((native/"evaluation-000000.envelope.json").read_bytes())
        packet=json.loads(gzip.decompress((native/envelope["packet"]["file"]).read_bytes()))
        self.assertEqual(envelope["schema"],subject.ENVELOPE_SCHEMA)
        self.assertEqual(packet["choices"],self.plan["original_plan"]["choices"][210:245])
        self.assertEqual(envelope["new_choices"],self.plan["choices"][:35])
        self.assertEqual(packet["request_ids"][0],"assessment-000210")
        self.assertEqual(rows[0]["request_id"],"test-evaluation-000000")
        self.assertEqual(rows[0]["source_sha256"],subject.digest(native/"evaluation-000000.envelope.json"))
        self.assertEqual(envelope["packet"]["sha256"],subject.digest(native/envelope["packet"]["file"]))
        self.assertEqual(packet["cohort"],6);self.assertEqual(packet["optimizer_updates"],0)

    def test_durable_dispatch_precedes_home_and_claim_cannot_retry(self):
        c,choices,res=self.reserve(0)
        self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)
        with self.assertRaisesRegex(ValueError,"Repeated"):
            self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)
        self.assertEqual(self.env.calls,["home","shot"])

    def test_storage_floor_is_checked_before_any_home(self):
        c,choices,res=self.reserve(0);self.producer.free_bytes=lambda:6*1024**3-1
        with self.assertRaisesRegex(ValueError,"6 GiB"):
            self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)
        self.assertFalse(self.env.calls)

    def test_wrong_or_missing_durable_reservation_never_calls_home(self):
        c,choices,res=self.reserve(0);res["requested"]=70
        with self.assertRaisesRegex(ValueError,"durable"):
            self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)
        self.assertFalse(self.env.calls)

    def test_in_memory_original_plan_mutation_cannot_change_executed_command(self):
        c,choices,res=self.reserve(0)
        self.producer.plan["original_plan"]["choices"][210]["action"]["height_m"]+=.01
        with self.assertRaisesRegex(ValueError,"Frozen native TEST plan"):
            self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)
        self.assertFalse(self.env.calls)

    def test_native_binding_memory_change_is_rejected_before_home(self):
        c,choices,res=self.reserve(0);self.producer.binding["fingerprint"]="0"*64
        with self.assertRaisesRegex(ValueError,"Frozen native TEST plan"):
            self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)
        self.assertFalse(self.env.calls)

    def test_modified_action_never_calls_home(self):
        c,choices,res=self.reserve(0);choices[0]["action"]["height_m"]+=.01
        with self.assertRaisesRegex(ValueError,"frozen"):
            self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)
        self.assertFalse(self.env.calls)

    def test_native_failure_keeps_claim_and_will_not_retry(self):
        c,choices,res=self.reserve(0);self.env.fail=True
        with self.assertRaisesRegex(RuntimeError,"interrupted"):
            self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)
        self.assertTrue((self.root/"native/evaluation-000000.failure.json").exists())
        with self.assertRaisesRegex(ValueError,"Repeated"):
            self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)
        self.assertEqual(self.env.calls,["home","shot"])

    def test_wrong_disabled_boundary_label_is_rejected_without_silent_relabel(self):
        # Original shot_batch is exercised; replace only its returned label to
        # prove the adapter requires native no_selected_setting at the boundary.
        original=self.env.shot_batch
        def changed(*args,**kwargs):
            rows,raw=original(*args,**kwargs)
            for i,use in enumerate(self.env.active_mask):
                if not use:rows[i]["invalid"]="inactive_split"
            return rows,raw
        self.env.shot_batch=changed
        with self.assertRaisesRegex(ValueError,"Empty TEST support"):
            self.execute(0)
        self.assertTrue((self.root/"native/evaluation-000000.failure.json").exists())

    def test_frozen_guard_rechecked_immediately_before_home(self):
        c,choices,res=self.reserve(0);calls=0
        def fail_second():
            nonlocal calls
            calls+=1
            if calls==2:raise ValueError("frozen input changed")
        self.producer.frozen_guard=fail_second
        with self.assertRaisesRegex(ValueError,"frozen input"):
            self.producer.execute_reserved(c,choices,res,journal_directory=self.journal.directory)
        self.assertFalse(self.env.calls)
        self.assertTrue((self.root/"native/evaluation-000000.failure.json").exists())


class ProductionInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prepared=fixture()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.plan=copy.deepcopy(self.prepared);self.events=[]
        self.inputs=dict(plan=self.plan,model_request=dict(receipt_path=str(self.root/"model-check.json")),
            verify_frozen=lambda:self.events.append("guard"))
    def checker(self,request):
        self.events.append("model")
        value=model_receipt(self.plan);write_new_json(request["receipt_path"],value);return value
    def startup(self,callback):
        self.events.append("native")
        return dict(records=[],parking=[])
    def components(self,checker=None):
        return subject.ProductionComponents(self.inputs,self.root,"a"*32,synthetic=True,
            hooks=dict(model_check=checker or self.checker,native_initialize=self.startup))
    def test_model_receipt_is_durable_and_verified_before_any_native_hook(self):
        c=self.components();c.initialize(lambda *args:None)
        self.assertLess(self.events.index("model"),self.events.index("native"))
        self.assertTrue((self.root/"model-check.json").exists())
        with self.assertRaisesRegex(ValueError,"no retry"):c.initialize(lambda *args:None)
    def test_verified_model_receipt_bytes_remain_frozen_after_startup(self):
        c=self.components();c.initialize(lambda *args:None)
        (self.root/"model-check.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"receipt bytes changed"):c.verify_frozen()

    def test_invalid_model_proof_prevents_native_hook(self):
        def bad(request):
            v=self.checker(request);v["optimizer_steps"]=1;v["fingerprint"]=canonical_hash({k:x for k,x in v.items() if k!="fingerprint"});return v
        with self.assertRaises(ValueError):self.components(bad).initialize(lambda *args:None)
        self.assertNotIn("native",self.events)
    def test_default_constructor_does_not_start_an_app_or_model(self):
        c=subject.ProductionComponents(self.inputs,self.root,"a"*32,synthetic=True)
        self.assertIsNone(c.app);self.assertFalse(self.events);self.assertFalse(c.started)
    def test_production_startup_source_contains_no_trainer_or_optimizer_call(self):
        tree=ast.parse(Path(subject.__file__).read_text())
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for forbidden in ("LearningEngine","ContinuationRunner","ProductionComponents.initialize","imitation_step","start_ppo"):
            self.assertFalse(any(forbidden in call for call in calls),forbidden)


if __name__=="__main__":unittest.main()
