"""CPU reader construction and fabricated reconstruction only; no model loads."""

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_v4_model_check as subject
from candidate_continuation_ledger import canonical_hash
from test_candidate_v4_evaluation import fixture


def model_receipt(plan,*,original_binding=None):
    binding=copy.deepcopy(original_binding) if original_binding is not None else dict(
        source_sha256=plan["bindings"]["producer_source_sha256"],guard_source_sha256={"unchanged.py":"e"*64})
    binding.pop("fingerprint",None);binding["fingerprint"]=canonical_hash(binding)
    value=dict(schema=subject.SCHEMA,status="passed_synthetic",synthetic=True,
        plan_fingerprint=plan["fingerprint"],original_plan_fingerprint=plan["original_plan"]["fingerprint"],
        final_checkpoint_sha256=plan["final_checkpoint_sha256"],parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
        model_hashes=plan["model_hashes"],evaluation_source_sha256=plan["bindings"]["evaluation_source_sha256"],
        original_source_sha256=plan["bindings"]["producer_source_sha256"],original_native_binding=binding,
        original_guard_source_sha256=binding["guard_source_sha256"],cpu_only=True,cuda_visible_devices="",
        source_and_input_bytes_unchanged=True,commands_reconstructed=420,test_commands_verified=210,
        optimizer_steps=0,native_requests_executed=0,model_promoted=False,execution_authorized=False,
        release_authorized=False,original_assessment_completed=False,checkpoint_bindings_sha256="a"*64,
        forward_readouts_sha256="b"*64,runtime=copy.deepcopy(subject.RUNTIME))
    value["fingerprint"]=canonical_hash(value)
    return value


class ModelCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prepared=fixture()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.inputs=self.root/"inputs";self.inputs.mkdir()
        self.old=self.root/"old-source";self.old.mkdir();(self.old/"proof.py").write_text("# fabricated\n")
        self.out=self.root/"new";self.out.mkdir();self.plan=copy.deepcopy(self.prepared)
        values={k:{"synthetic":True} for k in subject.ROLES}
        values["prepared_plan"]=self.plan;values["old_assessment_plan"]=self.plan["original_plan"]
        self.files={}
        for key,value in values.items():
            p=self.inputs/(key+".json");p.write_text(json.dumps(value,sort_keys=True))
            self.files[key]=dict(path=str(p),sha256=subject.digest(p))
        self.request=dict(schema=subject.REQUEST_SCHEMA,synthetic=True,files=self.files,
            original_source_dir=str(self.old),original_source_sha256=self.plan["bindings"]["producer_source_sha256"],
            evaluation_source_sha256=self.plan["bindings"]["evaluation_source_sha256"],protected_roots=[str(self.inputs)],
            receipt_path=str(self.out/"model-check.json"),expected_runtime=copy.deepcopy(subject.RUNTIME))
        self.rehash();self.calls=[]
        patcher=patch.object(subject,"source_hash",side_effect=lambda p:self.request["original_source_sha256"]
            if Path(p)==self.old else self.request["evaluation_source_sha256"])
        patcher.start();self.addCleanup(patcher.stop)
    def rehash(self):
        self.request.pop("fingerprint",None);self.request["fingerprint"]=canonical_hash(self.request)
    def fake_process(self,command,**kwargs):
        self.calls.append((command,kwargs));raw=kwargs["input"];request=json.loads(raw)
        value=model_receipt(self.plan)
        value.update(request_sha256=hashlib.sha256(raw).hexdigest(),input_sha256={k:r["sha256"] for k,r in request["files"].items()})
        value.pop("fingerprint");value["fingerprint"]=canonical_hash(value)
        subject._write_new(Path(request["receipt_path"]),value)
        return subprocess.CompletedProcess(command,0)

    def test_fixed_clean_cpu_child_and_exact_input_receipt(self):
        result=subject.check_models(self.request,run=self.fake_process)
        command,kwargs=self.calls[0]
        self.assertEqual(command[0],sys.executable);self.assertIn("--cpu-helper",command)
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"],"");self.assertEqual(kwargs["timeout"],600)
        self.assertTrue(kwargs["check"]);self.assertNotIn("shell",kwargs)
        self.assertEqual(result["status"],"passed_synthetic");self.assertEqual(result["optimizer_steps"],0)
        self.assertEqual(result["native_requests_executed"],0)

    def test_changed_checkpoint_bytes_are_rejected_before_child(self):
        Path(self.files["final_checkpoint"]["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError,"input bytes changed"):subject.check_models(self.request,run=self.fake_process)
        self.assertFalse(self.calls)

    def test_protected_receipt_path_and_existing_output_cannot_be_used(self):
        self.request["receipt_path"]=str(self.inputs/"new.json");self.rehash()
        with self.assertRaisesRegex(ValueError,"overlaps"):subject.check_models(self.request,run=self.fake_process)
        self.request["receipt_path"]=str(self.out/"existing.json");Path(self.request["receipt_path"]).write_text("keep");self.rehash()
        with self.assertRaisesRegex(ValueError,"untouched"):subject.check_models(self.request,run=self.fake_process)

    def test_changed_source_before_or_during_child_is_rejected(self):
        with patch.object(subject,"source_hash",return_value="0"*64):
            with self.assertRaisesRegex(ValueError,"producer source"):subject.check_models(self.request,run=self.fake_process)
        def changed(command,**kwargs):
            result=self.fake_process(command,**kwargs)
            Path(self.files["parent_checkpoint"]["path"]).write_text("late change")
            return result
        with self.assertRaisesRegex(ValueError,"changed during"):subject.check_models(self.request,run=changed)

    def test_subprocess_error_and_timeout_never_fabricate_receipt(self):
        for error in (subprocess.CalledProcessError(1,["fake"]),subprocess.TimeoutExpired(["fake"],600)):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):subject.check_models(self.request,run=lambda *a,**k:(_ for _ in ()).throw(error))
                self.assertFalse(Path(self.request["receipt_path"]).exists())

    def test_wrong_receipt_provenance_or_model_identity_is_rejected(self):
        for field,value in (("status","passed"),("model_hashes",{}),("optimizer_steps",1),("cpu_only",False),
                ("source_and_input_bytes_unchanged",False),("original_assessment_completed",True)):
            r=model_receipt(self.plan);r[field]=value;r.pop("fingerprint");r["fingerprint"]=canonical_hash(r)
            with self.subTest(field=field),self.assertRaises(ValueError):subject.validate_model_receipt(r,self.plan)

    def test_original_binding_cannot_be_relabelled_with_new_source(self):
        r=model_receipt(self.plan);r["original_native_binding"]["source_sha256"]=r["evaluation_source_sha256"]
        r["original_native_binding"].pop("fingerprint");r["original_native_binding"]["fingerprint"]=canonical_hash(r["original_native_binding"])
        r.pop("fingerprint");r["fingerprint"]=canonical_hash(r)
        with self.assertRaisesRegex(ValueError,"proof"):subject.validate_model_receipt(r,self.plan)

    def test_real_request_cannot_accept_fabricated_source_and_inputs(self):
        self.request["synthetic"]=False;self.rehash()
        with self.assertRaisesRegex(ValueError,"Original V4"):subject.check_models(self.request,run=self.fake_process)
        self.assertFalse(self.calls)

    def test_helper_rejects_visible_cuda_or_synthetic_request_before_model_import(self):
        raw=json.dumps(self.request,sort_keys=True,separators=(",",":")).encode();sha=hashlib.sha256(raw).hexdigest()
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}):
            with self.assertRaisesRegex(ValueError,"hide CUDA"):subject.cpu_helper(raw,sha)
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}):
            with self.assertRaisesRegex(ValueError,"Synthetic model checks"):subject.cpu_helper(raw,sha)

    def test_each_original_test_field_is_preserved_independently(self):
        subject._verify_suffix(self.plan,self.plan["original_plan"])
        for field,value in (("ik_seeds",{}),("enabled",False),("method","parent"),("original_request_id","changed")):
            p=copy.deepcopy(self.plan);p["choices"][0][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"command changed"):
                subject._verify_suffix(p,p["original_plan"])

    def test_module_has_no_simulator_or_optimizer_construction(self):
        tree=ast.parse(Path(subject.__file__).read_text())
        calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertFalse(set(calls)&{"AppLauncher","LearningEngine","ContinuationRunner","Adam","train"})


if __name__=="__main__":unittest.main()
