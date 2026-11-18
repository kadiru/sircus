"""Saved-data recovery proof and scientific parity using invented data only."""

from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import location_comparison_recovery_verify as recovery
import location_comparison_verify as original
import test_location_comparison_verify as original_tests
from location_comparison_watchdog import signed


class RecoveryVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_tests.FullVerificationTests.setUpClass()
        fixture = original_tests.FullVerificationTests
        cls.temporary = tempfile.TemporaryDirectory()
        cls.parent = Path(cls.temporary.name).resolve()
        cls.root = cls.parent/"native"
        shutil.copytree(fixture.root, cls.root)
        cls.manifest, cls.plan, cls.trace = copy.deepcopy((fixture.manifest, fixture.plan, fixture.trace))
        cls.checks = fixture.checks
        (cls.root/"child-exit.json").unlink()
        (cls.root/"events").mkdir()
        for index, event in enumerate(cls.trace["events"]):
            (cls.root/"events"/f"event-{index:06d}.json").write_text(json.dumps(event))
        cls.session = "f"*32
        owner = dict(pid=10,start_token="owner-start",cookie=cls.session)
        child = dict(pid=11,start_token="child-start",cookie=cls.session)
        cls.failed = signed(dict(schema="sircus-location-comparison-watchdog-v1",
            runtime_fingerprint=cls.manifest["fingerprint"],comparison_fingerprint=cls.plan["fingerprint"],
            session_id=cls.session,status="stopped_no_retry",reason="independent watchdog wall deadline reached",
            termination="stopped",lock_release_safe=True,native_application_starts=1,
            native_child_start_attempts=1,native_child_process_starts=1,owner=owner,native_owner=child,
            exit_code=None,final_snapshot=cls.trace["events"][-1]["payload"]["bounds"]))
        (cls.root/"comparison-watchdog-exit.json").write_text(json.dumps(cls.failed))
        state = dict(Status="exited",Pid=0,ExitCode=1,OOMKilled=False,
                     StartedAt="2026-11-17T22:15:51Z",FinishedAt="2026-11-18T13:44:38Z")
        cls.inventory = dict(schema=recovery.INVENTORY_SCHEMA,
            status="hash_inventory_only_not_a_scientific_audit",session_id=cls.session,
            source_commit=recovery.SOURCE_COMMIT,manifest_sha256="a"*64,container_id="b"*64,
            closed_container_state=state,files={})
        cls.inventory_path = cls.parent/"inventory.json"
        cls.retirement_path = cls.parent/"retirement.json"
        cls.refresh_inventory()
        cls.retirement = dict(schema=recovery.RETIREMENT_SCHEMA,status="owned_native_processes_retired",
            observed_at="2026-11-18T13:48:34Z",source_commit=recovery.SOURCE_COMMIT,
            manifest_fingerprint=cls.manifest["fingerprint"],comparison_fingerprint=cls.plan["fingerprint"],
            native_session_id=cls.session,native_exit_sha256=original.sha(cls.root/"comparison-watchdog-exit.json"),
            native_exit_fingerprint=cls.failed["fingerprint"],container_id="b"*64,container_state=state,
            owner=owner,native_owner=child,owned_processes_alive=[],owned_process_identity_check="verified_absent",
            original_lock_released=True,new_lock_owner_not_touched=True)
        cls.retirement_path.write_text(json.dumps(cls.retirement))
        cls.request = cls.make_request()
        cls.report = cls.run_audit()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        original_tests.FullVerificationTests.tearDownClass()

    @classmethod
    def refresh_inventory(cls):
        cls.inventory["files"] = {str(p.relative_to(cls.root)):dict(bytes=p.stat().st_size,sha256=original.sha(p))
                                 for p in cls.root.rglob("*") if p.is_file()}
        cls.inventory["file_count"] = len(cls.inventory["files"])
        cls.inventory["total_bytes"] = sum(v["bytes"] for v in cls.inventory["files"].values())
        cls.inventory_path.write_text(json.dumps(cls.inventory))

    @classmethod
    def make_request(cls):
        value = dict(schema=recovery.REQUEST_SCHEMA,native_output=str(cls.root),manifest_sha256="a"*64,
            source_commit=recovery.SOURCE_COMMIT,manifest_fingerprint=cls.manifest["fingerprint"],
            comparison_fingerprint=cls.plan["fingerprint"],native_session_id=cls.session,
            native_exit_fingerprint=cls.failed["fingerprint"],native_exit_sha256=original.sha(cls.root/"comparison-watchdog-exit.json"),
            inventory_path=str(cls.inventory_path),inventory_sha256=original.sha(cls.inventory_path),
            retirement_proof_path=str(cls.retirement_path),retirement_proof_sha256=original.sha(cls.retirement_path))
        return signed(value)

    @classmethod
    def run_audit(cls, checks=None, request=None):
        return recovery.audit_recovery(cls.root,cls.manifest,cls.plan,cls.trace,
            cls.request if request is None else request,synthetic=True,
            checks=cls.checks if checks is None else checks)

    @contextmanager
    def changed(self, path, raw):
        before = path.read_bytes() if path.exists() else None
        path.write_bytes(raw)
        try:
            yield path
        finally:
            if before is None:
                path.unlink()
            else:
                path.write_bytes(before)

    def test_scientific_body_is_original_except_explicit_exit_and_status(self):
        proof = recovery.verify_scientific_body()
        self.assertTrue(proof["scientific_body_preserved"])
        self.assertEqual(proof["original_verifier_sha256"],recovery.ORIGINAL_VERIFIER_SHA256)
        self.assertEqual(proof["recovery_scientific_body_sha256"],
                         hashlib.sha256(inspect.getsource(original.audit).encode()).hexdigest())

    def test_scientific_body_guard_change_fails_parity(self):
        getsource=inspect.getsource
        def changed_source(function):
            source=getsource(function)
            if function is recovery._audit_scientific:
                source=source.replace('paid_samples == accounting["completed"] == 18520',
                                      'paid_samples == accounting["completed"] == 1')
            return source
        with mock.patch.object(recovery.inspect,"getsource",side_effect=changed_source):
            with self.assertRaisesRegex(ValueError,"scientific audit body changed"):
                recovery.verify_scientific_body()

    def test_full_synthetic_saved_data_audit_covers_every_artifact_and_no_native_claim(self):
        r = self.report
        self.assertEqual(r["schema"],recovery.SCHEMA)
        self.assertEqual(r["status"],"synthetic_verified")
        self.assertTrue(r["synthetic"])
        self.assertEqual((r["paid_policy_samples"],r["terminal_batches"],r["update_artifacts"],r["checkpoint_count"]),
                         (18520,539,128,144))
        self.assertFalse(r["original_audit_passed"])
        self.assertFalse(r["clean_shutdown"])
        self.assertFalse(r["model_promoted"])
        self.assertNotIn("child_exit_sha256",r["exit_proof"])
        self.assertTrue(r["recovery_proof"]["inventory_before_after_verified"])
        self.assertEqual(r["recovery_proof"]["journal"]["file_count"],len(self.trace["events"]))
        self.assertEqual(r["fingerprint"],recovery.signed_hash({k:v for k,v in r.items() if k!="fingerprint"}))
        self.assertNotEqual(r["fingerprint"],recovery.process_digest({k:v for k,v in r.items() if k!="fingerprint"}))
        for field in ("checkpoint_bytes_independently_verified","model_likelihoods_independently_verified","native_physics_verified"):
            self.assertIs(r[field],False)

    def test_original_final_audit_remains_rejected(self):
        with self.assertRaisesRegex(ValueError,"child-exit"):
            original.audit(self.root,self.manifest,self.plan,self.trace,synthetic=True,checks=self.checks)

    def test_native_injected_checks_or_visible_cuda_rejected_before_readers(self):
        with self.assertRaisesRegex(ValueError,"Injected checks"):
            recovery.audit_recovery(self.root,self.manifest,self.plan,self.trace,self.request,checks=self.checks)
        with mock.patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}):
            with self.assertRaisesRegex(ValueError,"CPU-only"):
                recovery.audit_recovery(self.root,self.manifest,self.plan,self.trace,self.request)

    def test_request_resigning_does_not_relax_failed_receipt_conditions(self):
        changed = dict(self.failed,status="native_exit_zero_independent_audit_required",reason=None)
        changed = signed({k:v for k,v in changed.items() if k!="fingerprint"})
        with self.changed(self.root/"comparison-watchdog-exit.json",json.dumps(changed).encode()):
            request = dict(self.request,native_exit_sha256=original.sha(self.root/"comparison-watchdog-exit.json"),
                           native_exit_fingerprint=changed["fingerprint"])
            request = signed({k:v for k,v in request.items() if k!="fingerprint"})
            with self.assertRaisesRegex(ValueError,"deadline failure"):
                self.run_audit(request=request)

    def test_missing_failed_receipt_binding_or_changed_request_rejected(self):
        request = dict(self.request,native_exit_sha256="0"*64)
        with self.assertRaisesRegex(ValueError,"request fingerprint"):
            self.run_audit(request=request)
        request = signed({k:v for k,v in request.items() if k!="fingerprint"})
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            self.run_audit(request=request)

    def test_child_completion_or_failure_must_not_be_fabricated(self):
        for name in ("child-exit.json","child-failure.json"):
            with self.subTest(name=name),self.changed(self.root/name,b"{}"):
                with self.assertRaisesRegex(ValueError,"existing child"):
                    self.run_audit()

    def test_retirement_requires_exact_owned_identity_absence_and_safe_release(self):
        changes = [dict(owned_processes_alive=[1]),dict(original_lock_released=False),
                   dict(native_owner=dict(self.retirement["native_owner"],start_token="reused")),
                   dict(container_state=dict(self.retirement["container_state"],Pid=12)),
                   dict(observed_at="2026-11-17T22:00:00Z")]
        for change in changes:
            with self.subTest(change=change),self.changed(self.retirement_path,json.dumps(dict(self.retirement,**change)).encode()):
                request = signed({k:v for k,v in dict(self.request,retirement_proof_sha256=original.sha(self.retirement_path)).items() if k!="fingerprint"})
                with self.assertRaises(ValueError):
                    self.run_audit(request=request)

    def test_sealed_inventory_rejects_added_missing_changed_and_linked_evidence(self):
        with self.changed(self.root/"unlisted.txt",b"extra"):
            with self.assertRaisesRegex(ValueError,"Unlisted"):
                recovery.verify_inventory(self.root,self.inventory)
        sample = self.root/next(iter(self.inventory["files"]))
        with self.changed(sample,b"changed"):
            with self.assertRaisesRegex(ValueError,"bytes changed"):
                recovery.verify_inventory(self.root,self.inventory)
        link = self.root/"linked"
        link.symlink_to(sample)
        try:
            with self.assertRaisesRegex(ValueError,"Linked"):
                recovery.verify_inventory(self.root,self.inventory)
        finally:
            link.unlink()
        before=sample.read_bytes();sample.unlink()
        try:
            with self.assertRaisesRegex(ValueError,"Missing"):
                recovery.verify_inventory(self.root,self.inventory)
        finally:
            sample.write_bytes(before)

    def test_persisted_journal_cannot_disagree_with_trace(self):
        path=self.root/"events/event-000000.json"
        with self.changed(path,b"{}"):
            with self.assertRaisesRegex(ValueError,"journal differs"):
                recovery.verify_journal(self.root,self.trace)
        with self.changed(self.root/"events/extra.json",b"{}"):
            with self.assertRaisesRegex(ValueError,"journal membership"):
                recovery.verify_journal(self.root,self.trace)

    def test_original_startup_and_parent_scientific_failures_remain_failures(self):
        with self.assertRaisesRegex(ValueError,"startup physics"):
            self.run_audit(checks=dict(self.checks,startup=lambda *a,**k:dict(status="failed")))
        with self.assertRaisesRegex(ValueError,"original seed22"):
            self.run_audit(checks=dict(self.checks,parent=lambda *a:dict(weight_hash="0"*64,optimizer_hash="1"*64,optimizer_config={"synthetic":True})))

    def test_evidence_changed_by_reader_detected_after_scientific_pass(self):
        path=self.root/"poses.json"
        before=path.read_bytes()
        calls=[]
        def mutate(*args,**kwargs):
            if not calls:
                path.write_bytes(b'{"changed":true}')
                calls.append(True)
            return dict(synthetic=True,full_rate_reader="injected")
        try:
            with self.assertRaisesRegex(ValueError,"bytes changed"):
                self.run_audit(checks=dict(self.checks,native=mutate))
        finally:
            path.write_bytes(before)

    def test_recovery_request_and_original_evidence_are_not_written(self):
        self.assertEqual(original.sha(self.inventory_path),self.request["inventory_sha256"])
        self.assertEqual(original.sha(self.retirement_path),self.request["retirement_proof_sha256"])
        self.assertFalse((self.root/"child-exit.json").exists())
        self.assertFalse((self.root/"recovery-audit.json").exists())
        self.assertEqual(json.loads((self.root/"comparison-watchdog-exit.json").read_text()),self.failed)


if __name__ == "__main__":
    unittest.main()
