"""Production adapter tests inject every host/process operation; no native work."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_continuation_runtime as subject
from candidate_continuation_launcher import LaunchPlan,MIN_FREE_BYTES,supervise
from candidate_continuation_ledger import PARENT_CHECKPOINT_SHA256,canonical_hash
from candidate_continuation_watchdog import ProgressWriter,signed,SCHEMA,metadata_snapshot
from candidate_continuation_journal import write_new_json
from test_candidate_continuation_launcher import successful_progress

SOURCE=Path(__file__).resolve().parents[1]


class FakeHost:
    def __init__(self,workspace,plan):
        self.workspace,self.plan,self.t=workspace,plan,10.
        self.calls,self.started,self.stopped=[],[],[]
        self.free,self.locked,self.code=MIN_FREE_BYTES+100,False,None
        self.owner=dict(pid=901,start_token="boot:90",cookie=plan.session,ppid=1,sid=901,state="S")
        self.snapshots=successful_progress();self.next_snapshot=0
        self.in_container=False
    def monotonic(self):return self.t
    def sleep(self,s):self.t+=s
    def free_bytes(self,path):return self.free
    def acquire_lock(self,path,session):self.locked=True;return True
    def release_lock(self,path,session):self.locked=False
    def start(self,argv,log):
        self.started.append(argv)
        rawpath=argv[-1];payload=json.loads((self.workspace/Path(rawpath).relative_to("/workspace")).read_text())
        output=Path(self.plan.output)
        write_new_json(output/"runtime-owner.json",signed(dict(schema=SCHEMA,session=self.plan.session,
            owner=self.owner,native_owner=dict(self.owner,pid=902),absolute_deadline=payload["absolute_deadline"],
            payload_sha256=payload["fingerprint"])))
        self.writer=ProgressWriter(output,self.plan.session,clock=self.monotonic)
        self.writer.publish(self.snapshots[0]);self.next_snapshot=1
        return types.SimpleNamespace(pid=2000,poll=lambda:self.code)
    def exit_receipt(self,code=0):
        self.code=code
        claimed=json.loads((Path(self.plan.output)/"runtime-owner.json").read_text())
        write_new_json(Path(self.plan.output)/"runtime-exit.json",signed(dict(schema=SCHEMA,
            session=self.plan.session,owner=claimed["owner"],native_owner=claimed["native_owner"],
            payload_sha256=claimed["payload_sha256"],termination="stopped" if code else None,
            exit_code=code,status="complete_audit_receipts_required" if code==0 else "failed_no_retry")))
    def run(self,argv):
        self.calls.append(argv)
        self.in_container=True
        try:return self._run(argv)
        finally:self.in_container=False
    def _run(self,argv):
        if argv[:2]==["docker","inspect"]:
            return json.dumps([dict(State=dict(Running=True),HostConfig=dict(DeviceRequests=[dict(DeviceIDs=[subject.GPU_UUID])]))])
        if argv[0]=="nvidia-smi":return ""
        if "--snapshot" in argv:
            def option(name):return argv[argv.index(name)+1]
            return json.dumps(metadata_snapshot(option("--snapshot"),
                expected_payload_sha256=option("--expected-payload-sha256"),
                payload_fingerprint=option("--payload-fingerprint"),session=option("--session"),nonce=option("--nonce"),
                local_path=lambda path:self.workspace/Path(path).relative_to("/workspace")))
        if "--inspect" in argv:
            if self.next_snapshot<len(self.snapshots):
                snapshot=self.snapshots[self.next_snapshot];self.writer.publish(snapshot);self.next_snapshot+=1
                if not snapshot["running"]:self.exit_receipt()
            return json.dumps(self.owner)
        if "--stop" in argv:
            self.stopped.append(json.loads(argv[-1]));self.exit_receipt(1)
            return json.dumps(dict(status="stop_requested"))
        raise AssertionError("Unexpected mocked host operation: "+repr(argv))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.workspace=self.root/"workspace";self.workspace.mkdir()
        self.source=self.workspace/"source";(self.source/"sircus/isaac").mkdir(parents=True)
        for name in ("film_cylinder_farm.py","parallel_runtime.py"):
            shutil.copyfile(SOURCE/name,self.source/"sircus/isaac"/name)
        for name in ("candidate_continuation_entry.py","candidate_continuation_watchdog.py","python.sh"):
            (self.source/"sircus/isaac"/name).write_text("# synthetic source fixture\n")
        self.archive=self.workspace/"source.tar";self.commit="a"*40
        with tarfile.open(self.archive,"w",format=tarfile.PAX_FORMAT,pax_headers=dict(comment=self.commit)) as tar:
            for path in sorted(self.source.rglob("*")):
                if path.is_file():tar.add(path,arcname=str(path.relative_to(self.source)))
        roots=[self.workspace/n for n in ("frozen","bank","fresh")]
        for path in roots:path.mkdir()
        self.parent=roots[0]/"parent.pt";self.parent.write_bytes(b"synthetic parent")
        bank=roots[1]/"bank.json";bank.write_text("{}")
        split=roots[2]/"split.json";split.write_text("{}")
        self.manifest_path=self.workspace/"manifest.json"
        raw_sha=subject.sha
        def synthetic_sha(path):
            if Path(path)==self.parent and self.parent.read_bytes()==b"synthetic parent":return PARENT_CHECKPOINT_SHA256
            return raw_sha(path)
        self.sha_patch=patch.object(subject,"sha",side_effect=synthetic_sha);self.sha_patch.start();self.addCleanup(self.sha_patch.stop)
        py={p.name:raw_sha(p) for p in sorted((self.source/"sircus/isaac").glob("*.py"))}
        self.manifest=dict(schema=subject.MANIFEST_SCHEMA,workspace_host=str(self.workspace),
            files=dict(parent_checkpoint=dict(path=str(self.parent),sha256=PARENT_CHECKPOINT_SHA256),
                train_bank=dict(path=str(bank),sha256=raw_sha(bank)),fresh_split_manifest=dict(path=str(split),sha256=raw_sha(split))),
            native_contract=dict(synthetic=True),source_sha256=canonical_hash(py),protected_roots=list(map(str,roots)))
        self.write_manifest()
        bindings=dict(parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            train_bank_sha256=raw_sha(bank),fresh_split_sha256=raw_sha(split),native_binding_sha256=canonical_hash(self.manifest["native_contract"]),
            producer_source_sha256=self.manifest["source_sha256"],learning_protocol_sha256=subject.PROTOCOL_SHA256,
            **{n:raw_sha(self.source/"sircus/isaac"/n) for n in ("film_cylinder_farm.py","parallel_runtime.py")})
        (self.workspace/"scene.lock").write_text("mock lock; never opened by tests")
        self.plan=LaunchPlan.create(self.workspace/"out",protected_roots=[self.source,*roots],
            scene_lock=self.workspace/"scene.lock",input_sha256=bindings,session="1"*32)
        self.ops=FakeHost(self.workspace,self.plan)
        self.runtime=subject.ProductionRuntime(self.spec,execution_authorized=True,ops=self.ops)

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.spec=subject.ProductionSpec.create(workspace_host=self.workspace,source_host=self.source,
            archive_path=self.archive,source_commit=self.commit,manifest_path=self.manifest_path,
            manifest_sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())

    def start(self):
        Path(self.plan.output).mkdir()
        self.runtime.acquire_scene_lock(self.plan.scene_lock,self.plan.session)
        return self.runtime.start_native(self.plan,10810.)

    def test_import_construction_and_missing_authorization_do_not_call_host(self):
        runtime=subject.ProductionRuntime(self.spec,ops=self.ops)
        self.assertFalse(self.ops.calls);self.assertFalse(self.ops.started)
        with self.assertRaisesRegex(ValueError,"authorized"):
            supervise(self.plan,runtime,execute=True)
        self.assertFalse(Path(self.plan.output).exists())
        with self.assertRaisesRegex(ValueError,"authorization"):
            runtime.acquire_scene_lock(self.plan.scene_lock,self.plan.session)

    def test_explicit_production_call_with_fake_backend_runs_one_complete_supervisor_history(self):
        result=supervise(self.plan,self.runtime,execute=True)
        self.assertEqual(result["status"],"native_complete_independent_audit_required")
        self.assertEqual(result["native_application_starts"],1);self.assertEqual(len(self.ops.started),1)
        self.assertFalse(result["model_promoted"]);self.assertFalse(self.ops.locked)
        self.assertFalse(json.loads((Path(self.plan.output)/"launch-result.json").read_text())["mock_only"])

    def test_source_snapshot_checks_archive_raw_files_and_prestart_native_contract(self):
        self.assertEqual(self.runtime.source_snapshot(self.plan),dict(self.plan.bindings))
        self.parent.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError,"input bytes"):self.runtime.source_snapshot(self.plan)
        self.assertFalse(self.ops.started)

    def test_raw_manifest_source_archive_and_unarchived_python_changes_fail(self):
        original=self.manifest_path.read_bytes();self.manifest_path.write_text("{}")
        with self.assertRaisesRegex(ValueError,"manifest bytes"):self.spec.verify(self.plan)
        self.manifest_path.write_bytes(original)
        (self.source/"sircus/isaac/unarchived.py").write_text("# extra")
        with self.assertRaisesRegex(ValueError,"source identity"):self.spec.verify(self.plan)

    def test_mapped_paths_and_protected_roots_cannot_escape_or_be_omitted(self):
        with self.assertRaisesRegex(ValueError,"mount"):self.spec.container_path(self.root/"outside")
        bad=LaunchPlan.create(self.workspace/"out",protected_roots=[self.source],scene_lock=self.workspace/"scene.lock",
            input_sha256=dict(self.plan.bindings),session="2"*32)
        with self.assertRaisesRegex(ValueError,"protected root"):self.spec.verify(bad)

    def test_manifest_symlink_or_external_input_is_rejected(self):
        self.manifest["files"]["parent_checkpoint"]["path"]=str(self.root/"outside")
        (self.root/"outside").write_text("mock")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,"existing mount"):self.spec.verify(self.plan)

    def test_gpu_query_is_only_4090_and_never_restarts_a_container(self):
        state=self.runtime.gpu_state(subject.GPU_UUID)
        self.assertEqual(state["compute_pids"],[])
        self.assertIn("--id="+subject.GPU_UUID,self.ops.calls[-1])
        self.assertEqual(self.ops.calls[0],["docker","inspect",subject.CONTAINER])
        with self.assertRaisesRegex(ValueError,"4090"):self.runtime.gpu_state("GPU-4070")
        self.assertFalse(any("restart" in argv or "start" in argv for argv in self.ops.calls))

    def test_fixed_start_command_owns_watchdog_and_cannot_retry(self):
        owner=self.start()
        self.assertEqual(owner.pid,901);self.assertEqual(len(self.ops.started),1)
        argv=self.ops.started[0]
        self.assertIn(subject.CONTAINER,argv);self.assertIn("--execute",argv)
        self.assertNotIn("/bin/sh",argv)
        with self.assertRaisesRegex(ValueError,"retry"):self.runtime.start_native(self.plan,10810.)
        self.assertEqual(len(self.ops.started),1)

    def test_no_direct_start_without_owned_shared_lock(self):
        Path(self.plan.output).mkdir()
        with self.assertRaisesRegex(ValueError,"Owned shared scene lock"):
            self.runtime.start_native(self.plan,10810.)
        self.assertFalse(self.ops.started)

    def test_4090_race_is_rechecked_immediately_before_start(self):
        original=self.ops.run
        self.ops.run=lambda argv:subject.GPU_UUID+", 555\n" if argv[0]=="nvidia-smi" else original(argv)
        Path(self.plan.output).mkdir();self.runtime.acquire_scene_lock(self.plan.scene_lock,self.plan.session)
        with self.assertRaisesRegex(ValueError,"no longer idle"):
            self.runtime.start_native(self.plan,10810.)
        self.assertFalse(self.ops.started)

    def test_failed_owner_receipt_or_handshake_is_not_a_second_start(self):
        def no_receipt(argv,log):self.ops.started.append(argv);return types.SimpleNamespace(poll=lambda:1)
        self.ops.start=no_receipt
        Path(self.plan.output).mkdir()
        self.runtime.acquire_scene_lock(self.plan.scene_lock,self.plan.session)
        with self.assertRaisesRegex(ValueError,"before ownership"):self.runtime.start_native(self.plan,10810.)
        with self.assertRaisesRegex(ValueError,"retry"):self.runtime.start_native(self.plan,10810.)

    def test_owner_cookie_packet_or_progress_replacement_is_rejected(self):
        owner=self.start()
        path=Path(self.plan.output)/"runtime-owner.json";raw=json.loads(path.read_text());raw["session"]="2"*32;path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError,"session/fingerprint"):self.runtime._receipt(path.name)
        self.assertEqual(owner.pid,901)

    def test_reused_watchdog_pid_cannot_be_stopped(self):
        owner=self.start();self.ops.owner["start_token"]="boot:999"
        self.assertEqual(self.runtime.terminate_owned(owner),"ownership_mismatch")
        self.assertFalse(self.ops.stopped)

    def test_only_exact_owned_pid_can_be_inspected_or_stopped(self):
        owner=self.start()
        with self.assertRaisesRegex(ValueError,"owned watchdog"):self.runtime.identity(999)
        with self.assertRaisesRegex(ValueError,"Unowned"):self.runtime.terminate_owned(type(owner)(999,owner.start_token,owner.session))
        self.assertFalse(self.ops.stopped)

    def test_stop_goes_through_watchdog_cookie_start_token_and_waits_for_exit(self):
        owner=self.start();self.assertEqual(self.runtime.terminate_owned(owner),"stopped")
        self.assertEqual(self.ops.stopped,[dict(pid=owner.pid,start_token=owner.start_token,session=owner.session)])
        self.assertEqual(self.runtime.terminate_owned(owner),"already_exited")

    def test_rehashed_exit_receipt_cannot_rebind_another_native_owner(self):
        self.start();self.ops.exit_receipt()
        path=Path(self.plan.output)/"runtime-exit.json";raw=json.loads(path.read_text())
        raw["native_owner"]["pid"]=999;raw.pop("fingerprint");path.write_text(json.dumps(signed(raw)))
        with self.assertRaisesRegex(ValueError,"exact owned watchdog"):
            self.runtime._receipt("runtime-exit.json")

    def test_uncertain_native_cleanup_cannot_release_shared_lock(self):
        owner=self.start();self.ops.exit_receipt(1)
        path=Path(self.plan.output)/"runtime-exit.json";raw=json.loads(path.read_text())
        raw["termination"]="stop_unconfirmed";raw.pop("fingerprint");path.write_text(json.dumps(signed(raw)))
        with self.assertRaisesRegex(ValueError,"retain the shared lock"):
            self.runtime.terminate_owned(owner)
        self.assertTrue(self.ops.locked)

    def test_terminal_progress_is_not_completion_before_watchdog_and_transport_exit(self):
        owner=self.start()
        while self.ops.next_snapshot<len(self.ops.snapshots)-1:self.runtime.identity(owner.pid)
        # A child terminal message alone is not trusted as an OS exit.
        snapshot=self.ops.snapshots[-1];self.ops.writer.publish(snapshot)
        result=self.runtime.poll(owner);self.assertTrue(result["running"]);self.assertIsNone(result["exit_code"])

    def test_private_0600_receipts_and_progress_are_never_read_by_host(self):
        original_bytes,original_text=Path.read_bytes,Path.read_text
        host_reads=[]
        def private(path):return path.name in ("runtime-owner.json","runtime-exit.json") or path.parent.name=="runtime-progress"
        def read_bytes(path,*args,**kwargs):
            if private(path) and not self.ops.in_container:
                host_reads.append(str(path));raise PermissionError("mock root-owned 0600 metadata")
            return original_bytes(path,*args,**kwargs)
        def read_text(path,*args,**kwargs):
            if private(path) and not self.ops.in_container:
                host_reads.append(str(path));raise PermissionError("mock root-owned 0600 metadata")
            return original_text(path,*args,**kwargs)
        with patch.object(Path,"read_bytes",read_bytes),patch.object(Path,"read_text",read_text):
            result=supervise(self.plan,self.runtime,execute=True)
        self.assertEqual(result["status"],"native_complete_independent_audit_required")
        self.assertEqual(host_reads,[]);self.assertEqual(len(self.ops.started),1)
        output=Path(self.plan.output)
        for path in [output/"runtime-owner.json",output/"runtime-exit.json",*(output/"runtime-progress").glob("event-*.json")]:
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
        snapshots=[argv for argv in self.ops.calls if "--snapshot" in argv]
        self.assertTrue(snapshots);self.assertTrue(all("-B" in argv for argv in snapshots))
        self.assertTrue(all(argv[argv.index("-u")+1]=="0" for argv in snapshots))
        self.assertFalse(any("chmod" in argv or "chown" in argv for argv in self.ops.calls))

    def test_complete_transported_history_preserves_every_phase_boundary(self):
        owner=self.start()
        for snapshot in self.ops.snapshots[1:]:self.ops.writer.publish(snapshot)
        self.ops.next_snapshot=len(self.ops.snapshots);self.ops.exit_receipt()
        rows=[self.runtime.poll(owner) for _ in range(5)]
        self.assertEqual([row["phase"] for row in rows],["initialization","imitation","train","assessment","audit"])
        self.assertTrue(all(row["running"] for row in rows[:-1]));self.assertFalse(rows[-1]["running"])
        self.assertEqual(rows[-1]["assessment_requests"],420);self.assertEqual(rows[-1]["exit_code"],0)
        self.assertEqual(len(self.runtime.progress_fingerprints),len(self.ops.snapshots))

    def test_snapshot_binding_nonce_and_fingerprint_tamper_fail_closed(self):
        self.start();original=self.ops.run
        for key,replacement in (("session","2"*32),("nonce","3"*32),("payload_raw_sha256","a"*64),
                ("payload_fingerprint","b"*64),("output","/workspace/foreign"),("fingerprint","c"*64)):
            def changed(argv):
                raw=original(argv)
                if "--snapshot" not in argv:return raw
                value=json.loads(raw);value[key]=replacement
                if key!="fingerprint":value.pop("fingerprint");value=signed(value)
                return json.dumps(value)
            self.ops.run=changed
            with self.subTest(field=key),self.assertRaisesRegex(ValueError,"snapshot binding/fingerprint"):
                self.runtime._metadata()
        self.ops.run=original

    def test_even_rehashed_shortened_progress_and_replaced_owner_are_rejected(self):
        self.start();original=self.ops.run
        def shortened(argv):
            value=json.loads(original(argv));value["progress"]=[];value.pop("fingerprint")
            return json.dumps(signed(value))
        self.ops.run=shortened
        with self.assertRaisesRegex(ValueError,"shortened or replaced"):self.runtime._metadata()
        self.ops.run=original
        path=Path(self.plan.output)/"runtime-owner.json";value=json.loads(path.read_text())
        value["owner"]["start_token"]="boot:999";value.pop("fingerprint");path.write_text(json.dumps(signed(value)))
        with self.assertRaisesRegex(ValueError,"owner receipt changed"):self.runtime._metadata()

    def test_arbitrary_receipt_name_fails_before_any_container_query(self):
        self.start();before=len(self.ops.calls)
        with self.assertRaisesRegex(ValueError,"fixed runtime receipts"):self.runtime._receipt("learning/final.pt")
        self.assertEqual(len(self.ops.calls),before)

    def test_host_queries_have_finite_timeouts_and_use_no_shell(self):
        ops=subject.LinuxHostOps()
        with patch.object(subject.sys,"platform","linux"), \
             patch.object(subject.subprocess,"run",return_value=types.SimpleNamespace(stdout="mock")) as run:
            self.assertEqual(ops.run(["mock-query"]),"mock")
        self.assertEqual(run.call_args.kwargs["timeout"],10);self.assertNotIn("shell",run.call_args.kwargs)

    def test_linux_host_start_wires_popen_without_starting_a_real_child(self):
        ops=subject.LinuxHostOps();log=self.workspace/"mock-child.log"
        child=types.SimpleNamespace(pid=4242)
        with patch.object(subject.sys,"platform","linux"), \
             patch.object(subject.subprocess,"Popen",return_value=child) as start:
            self.assertIs(ops.start(["mock-child","--fake"],log),child)
            self.addCleanup(ops.log.close)
            self.assertFalse(ops.log.closed)
            self.assertEqual(start.call_args.args,(["mock-child","--fake"],))
            self.assertIs(start.call_args.kwargs["stdout"],ops.log)
            self.assertEqual(start.call_args.kwargs["stderr"],subject.subprocess.STDOUT)
            self.assertEqual(start.call_args.kwargs["stdin"],subject.subprocess.DEVNULL)
            self.assertTrue(start.call_args.kwargs["start_new_session"])
            self.assertNotIn("shell",start.call_args.kwargs)
            with self.assertRaises(FileExistsError):ops.start(["mock-child"],log)
            self.assertEqual(start.call_count,1)
        self.assertEqual(log.read_bytes(),b"")

    def test_linux_host_spawn_failure_closes_stream_and_keeps_new_log_evidence(self):
        ops=subject.LinuxHostOps();log=self.workspace/"failed-mock-child.log";streams=[]
        def fail(argv,**kwargs):
            streams.append(kwargs["stdout"]);raise OSError("synthetic spawn failure")
        with patch.object(subject.sys,"platform","linux"), \
             patch.object(subject.subprocess,"Popen",side_effect=fail) as start:
            with self.assertRaisesRegex(OSError,"synthetic spawn failure"):ops.start(["mock-child"],log)
        self.assertEqual(start.call_count,1);self.assertTrue(streams[0].closed)
        self.assertTrue(log.exists());self.assertEqual(log.read_bytes(),b"")

    def test_linux_host_non_linux_guard_runs_before_popen_or_log_creation(self):
        ops=subject.LinuxHostOps();log=self.workspace/"forbidden-mock-child.log"
        with patch.object(subject.sys,"platform","darwin"),patch.object(subject.subprocess,"Popen") as start:
            with self.assertRaisesRegex(ValueError,"Linux simulation host"):ops.start(["mock-child"],log)
        start.assert_not_called();self.assertFalse(log.exists())

    def test_linux_host_shared_lock_opens_existing_file_and_uses_mocked_flock_only(self):
        ops=subject.LinuxHostOps();path=self.workspace/"scene.lock";raw=path.read_bytes()
        with patch.object(subject.sys,"platform","linux"),patch.object(subject.fcntl,"flock") as flock:
            self.assertTrue(ops.acquire_lock(path,self.plan.session));stream=ops.locks[self.plan.session]
            self.assertTrue(ops.acquire_lock(path,self.plan.session));self.assertEqual(flock.call_count,1)
            self.assertEqual(flock.call_args.args[1],subject.fcntl.LOCK_EX|subject.fcntl.LOCK_NB)
            ops.release_lock(path,self.plan.session)
            self.assertEqual(flock.call_args.args[1],subject.fcntl.LOCK_UN)
        self.assertTrue(stream.closed);self.assertFalse(ops.locks);self.assertEqual(path.read_bytes(),raw)

    def test_linux_host_contended_or_missing_lock_is_never_created(self):
        ops=subject.LinuxHostOps();path=self.workspace/"scene.lock"
        with patch.object(subject.sys,"platform","linux"), \
             patch.object(subject.fcntl,"flock",side_effect=BlockingIOError) as flock:
            self.assertFalse(ops.acquire_lock(path,self.plan.session));self.assertFalse(ops.locks)
            missing=self.workspace/"missing.lock"
            with self.assertRaises(FileNotFoundError):ops.acquire_lock(missing,self.plan.session)
        self.assertEqual(flock.call_count,1);self.assertFalse(missing.exists())


if __name__=="__main__":unittest.main()
