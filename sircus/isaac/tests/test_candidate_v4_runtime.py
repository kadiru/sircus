"""Production wiring rehearsed using synthetic archives/files and fake hosts."""
import copy
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_v4_runtime as subject
from candidate_v4_watchdog import ProgressWriter,SCHEMA,metadata_snapshot,signed
from candidate_continuation_journal import write_new_json
from candidate_continuation_ledger import canonical_hash
from test_candidate_v4_watchdog import successful_progress


class FakeHost:
    def __init__(self,workspace,plan):
        self.workspace,self.plan,self.t=workspace,plan,10.
        self.calls,self.started,self.stopped=[],[],[];self.free=6*1024**3+100
        self.locked=False;self.code=None;self.gpu_busy=False;self.container_running=True
        self.lock_available=True;self.lock_callback=None;self.after_start=None
        self.owner=dict(pid=901,start_token="boot:90",cookie=plan.session,ppid=1,sid=901,state="S")
        self.snapshots=successful_progress();self.next_snapshot=0;self.fail_start=False
    def monotonic(self):return self.t
    def sleep(self,s):self.t+=s
    def free_bytes(self,path):return self.free
    def acquire_lock(self,path,session):
        self.locked=self.lock_available
        if self.lock_callback:self.lock_callback()
        return self.locked
    def release_lock(self,path,session):self.locked=False
    def start(self,argv,log):
        self.started.append(argv)
        if self.fail_start:raise RuntimeError("fabricated start transport failure")
        path=self.workspace/Path(argv[-1]).relative_to("/workspace")
        payload=json.loads(path.read_text());self.payload=payload
        write_new_json(Path(self.plan.output)/"runtime-owner.json",signed(dict(schema=SCHEMA,
            session=self.plan.session,owner=self.owner,native_owner=dict(self.owner,pid=902),
            absolute_deadline=payload["absolute_deadline"],payload_sha256=payload["fingerprint"])))
        self.writer=ProgressWriter(Path(self.plan.output),self.plan.session,clock=self.monotonic)
        self.writer.publish(self.snapshots[0]);self.next_snapshot=1
        if self.after_start:self.after_start()
        return types.SimpleNamespace(pid=2000,poll=lambda:self.code)
    def exit_receipt(self,code=0):
        self.code=code;claimed=json.loads((Path(self.plan.output)/"runtime-owner.json").read_text())
        write_new_json(Path(self.plan.output)/"runtime-exit.json",signed(dict(schema=SCHEMA,session=self.plan.session,
            owner=claimed["owner"],native_owner=claimed["native_owner"],payload_sha256=claimed["payload_sha256"],
            termination="stopped" if code else None,exit_code=code,
            status="complete_audit_receipts_required" if code==0 else "failed_no_retry")))
    def run(self,argv):
        self.calls.append(argv)
        if argv[:2]==["docker","inspect"]:
            return json.dumps([dict(State=dict(Running=self.container_running),HostConfig=dict(DeviceRequests=[dict(DeviceIDs=[subject.GPU_UUID])]))])
        if argv[0]=="nvidia-smi":return (subject.GPU_UUID+", 999\n") if self.gpu_busy else ""
        def option(name):return argv[argv.index(name)+1]
        if "--snapshot" in argv:
            return json.dumps(metadata_snapshot(option("--snapshot"),expected_payload_sha256=option("--expected-payload-sha256"),
                payload_fingerprint=option("--payload-fingerprint"),session=option("--session"),nonce=option("--nonce"),
                local_path=lambda path:self.workspace/Path(path).relative_to("/workspace")))
        if "--inspect" in argv:
            if self.next_snapshot<len(self.snapshots):
                row=self.snapshots[self.next_snapshot];self.writer.publish(row);self.next_snapshot+=1
                if not row["running"]:self.exit_receipt()
            return json.dumps(self.owner)
        if "--stop" in argv:
            self.stopped.append(json.loads(argv[-1]));self.exit_receipt(1);return json.dumps(dict(status="stop_requested"))
        raise AssertionError("Unexpected mocked host operation: "+repr(argv))


class V4RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.workspace=self.root/"workspace";self.workspace.mkdir()
        self.source=self.workspace/"source";native=self.source/"sircus/isaac";native.mkdir(parents=True)
        for name in ("candidate_v4_entry.py","candidate_v4_watchdog.py","python.sh"):
            (native/name).write_text("# fabricated new source\n")
        self.original=self.workspace/"old-source";old=self.original/"sircus/isaac";old.mkdir(parents=True)
        (old/"immutable.py").write_text("# fabricated old producer\n")
        self.archive=self.workspace/"source.tar";self.commit="a"*40
        with tarfile.open(self.archive,"w",format=tarfile.PAX_FORMAT,pax_headers=dict(comment=self.commit)) as tar:
            for p in sorted(self.source.rglob("*")):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(self.source)))
        self.frozen=self.workspace/"old-partial";self.frozen.mkdir();self.old_witness=self.frozen/"keep.json"
        self.old_witness.write_text('{"charged":140,"unresolved":35}')
        self.inputs=self.workspace/"inputs";self.inputs.mkdir()
        self.parent=self.inputs/"parent.pt";self.parent.write_bytes(b"synthetic parent")
        self.final=self.inputs/"final.pt";self.final.write_bytes(b"synthetic final")
        self.legacy=self.workspace/"legacy";self.legacy.mkdir()
        for name in ("source-run","assets","frozen"):(self.legacy/name).mkdir()
        self.legacy_input=self.legacy/"bank.json";self.legacy_input.write_text("{}")
        self.oldmanifest=self.inputs/"original-manifest.json"
        self.oldvalue=dict(workspace_host=str(self.workspace),source_host=str(self.original),
            source_sha256=canonical_hash({p.name:subject.sha(p) for p in old.glob("*.py")}),
            source_run=str(self.legacy/"source-run"),assets_dir=str(self.legacy/"assets"),
            frozen_experiment_root=str(self.legacy/"frozen"),protected_roots=[str(self.legacy)],
            files=dict(bank=dict(path=str(self.legacy_input),sha256=subject.sha(self.legacy_input))))
        self.oldmanifest.write_text(json.dumps(self.oldvalue))
        self.patches=[patch.object(subject,"PARENT_CHECKPOINT_SHA256",subject.sha(self.parent)),
            patch.object(subject,"V4_CHECKPOINT_SHA256",subject.sha(self.final)),
            patch.object(subject,"V4_MANIFEST_SHA256",subject.sha(self.oldmanifest)),
            patch.object(subject,"V4_PRODUCER_SHA256",canonical_hash({p.name:subject.sha(p) for p in old.glob("*.py")}))]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        newsha=canonical_hash({p.name:subject.sha(p) for p in native.glob("*.py")})
        prepared=dict(requests=210,synthetic=False,final_checkpoint_sha256=subject.sha(self.final),
            parent_checkpoint_sha256=subject.sha(self.parent),native_execution_ready=False,
            execution_authorized=False,release_authorized=False,bindings=dict(evaluation_source_sha256=newsha))
        prepared=signed(prepared);self.prepared=self.inputs/"prepared.json";self.prepared.write_text(json.dumps(prepared))
        roles=dict(prepared_plan=self.prepared,parent_checkpoint=self.parent,final_checkpoint=self.final,original_manifest=self.oldmanifest)
        for role in ("original_learning_result","original_native_binding","old_assessment_plan","old_assessment_inventory","old_component_summary"):
            p=self.inputs/(role+".json");p.write_text("{}");roles[role]=p
        self.manifest=self.workspace/"manifest.json"
        self.value=dict(schema=subject.MANIFEST_SCHEMA,workspace_host=str(self.workspace),source_host=str(self.source),
            source_sha256=newsha,synthetic=False,expected_runtime=dict(python="synthetic"),original_source_host=str(self.original),original_source_sha256=subject.V4_PRODUCER_SHA256,
            prepared_plan_fingerprint=prepared["fingerprint"],protected_roots=list(map(str,(self.frozen,self.inputs,self.legacy))),
            files={k:dict(path=str(p),sha256=subject.sha(p)) for k,p in roles.items()},execution_authorized=False,release_authorized=False)
        self.manifest.write_text(json.dumps(self.value))
        self.spec=subject.ProductionSpec.create(workspace_host=self.workspace,source_host=self.source,archive_path=self.archive,
            source_commit=self.commit,manifest_path=self.manifest,manifest_sha256=subject.sha(self.manifest))
        self.plan=subject.EvaluationLaunchPlan.create(output=self.workspace/"new-evaluation",
            protected_roots=list(map(str,(self.source,self.original,self.frozen,self.inputs,self.legacy))),
            scene_lock=self.workspace/"scene.lock",session="1"*32,
            prepared_plan_fingerprint=prepared["fingerprint"],source_sha256=newsha)
        self.ops=FakeHost(self.workspace,self.plan)
        self.authorization=signed(dict(schema="sircus-v4-evaluation-operator-authorization-v1",action="evaluate_saved_v4_test_only",
            session=self.plan.session,output=self.plan.output,manifest_sha256=self.spec.manifest_sha256,
            prepared_plan_fingerprint=self.plan.prepared_plan_fingerprint,source_sha256=self.plan.source_sha256,
            execution_authorized=True,release_authorized=False))
    def runtime(self,**changes):
        options=dict(execution_authorized=True,authorization=self.authorization,ops=self.ops);options.update(changes)
        return subject.ProductionRuntime(self.spec,**options)

    def test_default_execution_disabled_makes_no_host_calls(self):
        r=subject.supervise(self.plan,self.runtime(execution_authorized=False),execute=False)
        self.assertEqual(r["status"],"execution_disabled");self.assertFalse(self.ops.calls);self.assertFalse(self.ops.started)
        self.assertFalse(Path(self.plan.output).exists())

    def test_unbound_operator_approval_fails_before_host_call(self):
        for changes in (dict(execution_authorized=False),dict(authorization=None),dict(authorization=dict(self.authorization,session="2"*32))):
            with self.subTest(changes=changes),self.assertRaises(ValueError):subject.supervise(self.plan,self.runtime(**changes),execute=True)
        self.assertFalse(self.ops.calls);self.assertFalse(self.ops.started)

    def test_source_archive_and_each_input_verified_without_process_operations(self):
        bindings,archive=self.spec.verify(self.plan)
        self.assertEqual(bindings["source_sha256"],self.plan.source_sha256)
        self.assertEqual(bindings["original_source_sha256"],subject.V4_PRODUCER_SHA256)
        self.assertEqual(archive["verified_source_files"],3);self.assertFalse(self.ops.calls)

    def test_complete_future_path_uses_one_watchdog_before_native_and_preserves_old_evidence(self):
        before=self.old_witness.read_bytes();runtime=self.runtime();r=subject.supervise(self.plan,runtime,execute=True)
        self.assertEqual(r["status"],"native_complete_audit_receipts_required",r)
        self.assertEqual(len(self.ops.started),1);self.assertFalse(self.ops.stopped);self.assertFalse(self.ops.locked)
        command=self.ops.started[0]
        self.assertEqual(command[:3],["docker","exec","-u"])
        self.assertIn("/workspace/source/sircus/isaac/candidate_v4_watchdog.py",command)
        self.assertNotIn("candidate_continuation_entry.py"," ".join(command))
        self.assertEqual(self.ops.payload["authorization_sha256"],self.authorization["fingerprint"])
        self.assertEqual(self.ops.payload["prepared_plan_fingerprint"],self.plan.prepared_plan_fingerprint)
        self.assertEqual(r["counts"]["evaluation_requests"],210)
        self.assertFalse(r["independent_audit_passed"]);self.assertFalse(r["model_promoted"])
        self.assertEqual((r["training_requests"],r["optimizer_steps"]),(0,0))
        self.assertEqual(self.old_witness.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,"already exists"):subject.supervise(self.plan,self.runtime(),execute=True)
        self.assertEqual(len(self.ops.started),1)

    def test_busy_gpu_stopped_container_or_foreign_scene_lock_never_start_or_restart(self):
        for field in ("gpu_busy","container_running","lock_available"):
            ops=FakeHost(self.workspace,self.plan);setattr(ops,field,field=="gpu_busy")
            r=subject.supervise(self.plan,self.runtime(ops=ops),execute=True)
            self.assertEqual(r["status"],"failed_no_retry");self.assertFalse(ops.started)
            self.assertFalse(any(x[:2] in (["docker","start"],["docker","restart"]) for x in ops.calls))
            self.assertFalse(Path(self.plan.output).exists())

    def test_source_input_mutation_before_start_is_rejected(self):
        self.final.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"input bytes"):subject.supervise(self.plan,self.runtime(),execute=True)
        self.assertFalse(self.ops.calls);self.assertFalse(self.ops.started)

    def test_mutation_after_lock_is_rejected_and_only_own_lease_released(self):
        self.ops.lock_callback=lambda:self.final.write_bytes(b"changed")
        r=subject.supervise(self.plan,self.runtime(),execute=True)
        self.assertEqual(r["status"],"failed_no_retry");self.assertFalse(self.ops.started);self.assertFalse(self.ops.locked)

    def test_readonly_metadata_refuses_nonce_or_payload_rebinding(self):
        runtime=self.runtime();r=subject.supervise(self.plan,runtime,execute=True)
        self.assertEqual(r["status"],"native_complete_audit_receipts_required")
        original=self.ops.run
        def wrong(argv):
            raw=original(argv)
            if "--snapshot" in argv:
                v=json.loads(raw);v["nonce"]="9"*32;v.pop("fingerprint");return json.dumps(signed(v))
            return raw
        self.ops.run=wrong
        with self.assertRaisesRegex(ValueError,"snapshot binding"):runtime._metadata()

    def test_reused_watchdog_pid_is_never_sent_stop(self):
        runtime=self.runtime();r=subject.supervise(self.plan,runtime,execute=True)
        # Fabricate still-running transport with foreign identity; no real process exists.
        self.ops.code=None;(Path(self.plan.output)/"runtime-exit.json").unlink()
        self.ops.owner["start_token"]="boot:reused"
        self.assertEqual(runtime.terminate_owned(runtime.owner),"ownership_mismatch")
        self.assertFalse(self.ops.stopped)

    def test_failed_start_consumes_intent_and_retains_unconfirmed_lock(self):
        self.ops.fail_start=True;r=subject.supervise(self.plan,self.runtime(),execute=True)
        self.assertEqual(r["status"],"failed_no_retry");self.assertEqual(len(self.ops.started),1)
        self.assertEqual(r["cleanup"],"unknown_child_identity_lock_retained");self.assertTrue(self.ops.locked)
        self.assertTrue((Path(self.plan.output)/"launch-intent.json").is_file())
        self.assertTrue((Path(self.plan.output)/"launch-result.json").is_file())

    def test_direct_start_rejects_partial_output_even_with_matching_authorization(self):
        runtime=self.runtime();runtime.bind_plan(self.plan);runtime.acquire_scene_lock(self.plan.scene_lock,self.plan.session)
        Path(self.plan.output).mkdir();(Path(self.plan.output)/"partial.npz").write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError,"new-only supervisor intent"):runtime.start_native(self.plan,1000.)
        self.assertFalse(self.ops.started)

    def test_original_bank_asset_and_source_run_roots_cannot_be_omitted(self):
        self.value["protected_roots"].remove(str(self.legacy));self.manifest.write_text(json.dumps(self.value))
        spec=subject.ProductionSpec.create(workspace_host=self.workspace,source_host=self.source,archive_path=self.archive,
            source_commit=self.commit,manifest_path=self.manifest,manifest_sha256=subject.sha(self.manifest))
        with self.assertRaisesRegex(ValueError,"omitted an original"):spec.verify(self.plan)
        self.assertFalse(self.ops.started)

    def test_manifest_cannot_relabel_old_source_or_turn_preparation_into_permission(self):
        self.value["execution_authorized"]=True;self.manifest.write_text(json.dumps(self.value))
        spec=subject.ProductionSpec.create(workspace_host=self.workspace,source_host=self.source,archive_path=self.archive,
            source_commit=self.commit,manifest_path=self.manifest,manifest_sha256=subject.sha(self.manifest))
        with self.assertRaisesRegex(ValueError,"authorization"):spec.verify(self.plan)


if __name__=="__main__":unittest.main()
