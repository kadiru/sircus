"""V4 independent watchdog with fake process, procfs and clocks only."""
import copy
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_v4_watchdog as subject
from candidate_continuation_ledger import canonical_hash
from test_candidate_continuation_launcher import diagnostic_records


def successful_progress():
    def row(phase,req=0,terminal=0,released=0,running=True):
        return dict(phase=phase,diagnostics=diagnostic_records(),parking_resets=1,
            evaluation_requests=req,evaluation_terminal=terminal,evaluation_releases=released,
            train_requests=0,optimizer_steps=0,running=running,exit_code=None if running else 0,
            audit_cpu_only=phase=="audit")
    return [dict(row("initialization"),diagnostics=[],parking_resets=0),row("initialization"),
        row("evaluation"),row("evaluation",35),row("evaluation",210,210,200),
        row("audit",210,210,200),row("audit",210,210,200,False)]


class FakeOps:
    def __init__(self,root,session):
        self.root,self.session,self.t=root,session,10.
        self.records={777:dict(pid=777,start_token="boot:77",cookie=session,ppid=1,sid=777,state="R")}
        self.started,self.signalled=[],[];self.prepared=False;self.running=False;self.exit_code=None
        self.snapshots=successful_progress();self.on_sleep=None;self.free=subject.MIN_FREE_BYTES+100
    def local_path(self,path):return self.root/Path(path).relative_to("/workspace")
    def own_pid(self):return 777
    def monotonic(self):return self.t
    def free_bytes(self,path):return self.free
    def prepare(self):self.prepared=True
    def identity(self,pid):return copy.deepcopy(self.records.get(pid))
    def start(self,command,environment):
        self.started.append((command,environment));self.running=True
        self.records[888]=dict(pid=888,start_token="boot:88",cookie=self.session,ppid=777,sid=888,state="S")
        if self.snapshots is not None:
            writer=subject.ProgressWriter(self.local_path(command[command.index("--output")+1]),self.session,clock=self.monotonic)
            for row in self.snapshots:writer.publish(row)
            self.exit_code=0
        return 888
    def poll(self,pid):
        if self.exit_code is not None:self.running=False;self.records.pop(888,None)
        return self.exit_code
    def owned(self,pid,session):
        return [copy.deepcopy(self.records[888])] if self.running and 888 in self.records else []
    def signal_owned(self,record,signum,session):
        self.signalled.append((record,signum,session));self.running=False;self.records.pop(888,None)
        return "signalled"
    def sleep(self,s):
        self.t+=s
        if self.on_sleep:self.on_sleep(self)


class V4WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();(self.root/"out").mkdir()
        source=self.root/"source/sircus/isaac";source.mkdir(parents=True)
        for name in ("candidate_v4_entry.py","candidate_v4_watchdog.py","python.sh"):
            (source/name).write_text("# fabricated source\n")
        self.manifest=self.root/"manifest.json";self.manifest.write_text("{}")
        members={"sircus/isaac/"+p.name:subject.sha(p) for p in source.iterdir()}
        py={p.name:subject.sha(p) for p in source.glob("*.py")}
        self.session="1"*32
        self.payload=subject.signed(dict(schema=subject.SCHEMA,session=self.session,source_root="/workspace/source",
            manifest="/workspace/manifest.json",manifest_sha256=subject.sha(self.manifest),output="/workspace/out",
            absolute_deadline=3010.,source_members_sha256=members,prepared_plan_fingerprint="a"*64,
            evaluation_source_sha256=canonical_hash(py),authorization_sha256="b"*64))
        self.ops=FakeOps(self.root,self.session)
    def run_watchdog(self):return subject.run_guarded(self.payload,self.ops)

    def test_complete_one_start_fixed_entry_no_learning_and_no_native_audit_claim(self):
        r=self.run_watchdog();self.assertEqual(r["status"],"complete_audit_receipts_required")
        self.assertEqual(len(self.ops.started),1);self.assertFalse(self.ops.signalled)
        command,env=self.ops.started[0]
        self.assertIn("/workspace/source/sircus/isaac/candidate_v4_entry.py",command)
        self.assertIn("--execute-approved",command);self.assertEqual(env[subject.COOKIE],self.session)
        self.assertEqual((r["training_requests"],r["optimizer_steps"]),(0,0))
        self.assertFalse(r["independent_audit_passed"]);self.assertFalse(r["model_promoted"])
        with self.assertRaisesRegex(ValueError,"retry"):self.run_watchdog()
        self.assertEqual(len(self.ops.started),1)

    def test_independent_timer_preempts_child_without_progress_or_callback_return(self):
        self.ops.snapshots=None;self.ops.on_sleep=lambda o:setattr(o,"t",611.)
        r=self.run_watchdog();self.assertEqual(r["status"],"failed_no_retry")
        self.assertIn("phase wall",r["reason"]);self.assertEqual(r["termination"],"stopped")
        self.assertEqual([x[1] for x in self.ops.signalled],[signal.SIGTERM])

    def test_absolute_bound_never_adds_term_grace_after_deadline(self):
        self.ops.snapshots=None;self.ops.on_sleep=lambda o:setattr(o,"t",3011.)
        def ignore_term(record,signum,session):
            self.ops.signalled.append((record,signum,session))
            if signum==signal.SIGKILL:self.ops.running=False;self.ops.records.pop(888,None)
            return "signalled"
        self.ops.signal_owned=ignore_term;r=self.run_watchdog()
        self.assertIn("absolute",r["reason"])
        self.assertEqual([x[1] for x in self.ops.signalled],[signal.SIGTERM,signal.SIGKILL])
        self.assertEqual(self.ops.t,3011.)

    def test_reused_native_leader_pid_is_not_signalled(self):
        self.ops.snapshots=None
        def reuse(o):o.t=611.;o.records[888]["start_token"]="boot:999"
        self.ops.on_sleep=reuse;r=self.run_watchdog()
        self.assertEqual(r["termination"],"ownership_mismatch");self.assertFalse(self.ops.signalled)

    def test_future_event_clock_remains_rejected_after_fresh_sample(self):
        self.ops.snapshots=None;start=self.ops.start
        def begun(command,environment):
            pid=start(command,environment)
            w=subject.ProgressWriter(self.root/"out",self.session,clock=lambda:10.025)
            for row in successful_progress():w.publish(row)
            self.ops.exit_code=0;return pid
        self.ops.start=begun;r=self.run_watchdog()
        self.assertIn("future phase receipt",r["reason"])

    def test_concurrent_progress_read_resamples_time_and_keeps_actual_phase_start(self):
        self.ops.snapshots=None;start=self.ops.start;writer=[]
        def begun(command,environment):
            pid=start(command,environment);writer.append(subject.ProgressWriter(self.root/"out",self.session,clock=self.ops.monotonic))
            writer[0].publish(successful_progress()[0]);return pid
        self.ops.start=begun;old=Path.iterdir;changed=[]
        def iterdir(path):
            if path==self.root/"out/runtime-progress" and not changed:
                self.ops.t+=.025;changed.append(True)
                for row in successful_progress()[1:]:writer[0].publish(row)
                self.ops.exit_code=0
            return old(path)
        with patch.object(Path,"iterdir",iterdir):r=self.run_watchdog()
        self.assertEqual(r["status"],"complete_audit_receipts_required",r["reason"])
        self.assertFalse(self.ops.signalled)

    def test_delayed_phase_transition_cannot_reset_expired_initialization(self):
        self.ops.snapshots=None;start=self.ops.start
        def begun(command,environment):
            pid=start(command,environment);times=iter([10.,610.01])
            w=subject.ProgressWriter(self.root/"out",self.session,clock=lambda:next(times))
            for row in successful_progress()[:2]:w.publish(row)
            self.ops.t=611.;self.ops.exit_code=0;return pid
        self.ops.start=begun;r=self.run_watchdog();self.assertIn("Delayed",r["reason"])

    def test_audit_requires_all210_terminal_and_cpu_only(self):
        for kind in ("pending","cuda"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as unused:
                out=self.root/kind;out.mkdir();p=dict(self.payload,output="/workspace/"+kind);p.pop("fingerprint")
                ops=FakeOps(self.root,self.session)
                if kind=="pending":ops.snapshots[4]["evaluation_terminal"]=209;ops.snapshots[4]["evaluation_releases"]=200
                else:ops.snapshots[-1]["audit_cpu_only"]=False
                r=subject.run_guarded(subject.signed(p),ops)
                self.assertEqual(r["status"],"failed_no_retry")
                self.assertIn("complete TEST-only" if kind=="pending" else "CPU-only",r["reason"])

    def test_training_optimizer_work_or_unknown_phase_is_rejected_before_dispatch_progress(self):
        for field,value in (("train_requests",1),("optimizer_steps",1),("phase","imitation")):
            row=successful_progress()[0];row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):subject.validate_snapshot(row)

    def test_budget_overrun_and_decreasing_counts_fail_closed(self):
        self.ops.snapshots[4]["evaluation_requests"]=211
        r=self.run_watchdog();self.assertIn("budget",r["reason"])

    def test_source_tamper_or_storage_floor_preempts_once(self):
        self.ops.snapshots=None
        self.ops.on_sleep=lambda o:(setattr(o,"t",12.),setattr(o,"free",subject.MIN_FREE_BYTES-1))
        r=self.run_watchdog();self.assertIn("storage",r["reason"]);self.assertEqual(len(self.ops.signalled),1)
        self.assertTrue((self.root/"out/runtime-exit.json").exists())

    def test_unlisted_python_or_linked_source_appearing_after_start_preempts_owned_child(self):
        for kind in ("extra","linked"):
            with self.subTest(kind=kind):
                out=self.root/kind;out.mkdir();payload=dict(self.payload,output="/workspace/"+kind);payload.pop("fingerprint")
                ops=FakeOps(self.root,self.session);ops.snapshots=None
                source=self.root/"source/sircus/isaac/candidate_v4_entry.py";raw=source.read_bytes()
                linked=self.root/"source/sircus/isaac/linked-original"
                def mutate(o):
                    o.t=12.
                    if kind=="extra":(source.parent/"unlisted.py").write_text("# extra")
                    else:
                        source.rename(linked);source.symlink_to(linked)
                ops.on_sleep=mutate
                result=subject.run_guarded(subject.signed(payload),ops)
                self.assertEqual(result["status"],"failed_no_retry")
                self.assertEqual(len(ops.signalled),1)
                if kind=="extra":(source.parent/"unlisted.py").unlink()
                else:source.unlink();linked.rename(source)
                self.assertEqual(source.read_bytes(),raw)

    def test_manifest_and_all_python_inventory_change_rejected_before_start(self):
        self.manifest.write_text("changed")
        with self.assertRaisesRegex(ValueError,"manifest bytes"):self.run_watchdog()
        self.assertFalse(self.ops.started)
        self.manifest.write_text("{}")
        (self.root/"source/sircus/isaac/unbound.py").write_text("# extra")
        with self.assertRaisesRegex(ValueError,"inventory"):self.run_watchdog()
        self.assertFalse(self.ops.started)

    def test_missing_authorization_binding_or_arbitrary_command_is_rejected(self):
        for key in ("authorization_sha256","prepared_plan_fingerprint","evaluation_source_sha256"):
            p=copy.deepcopy(self.payload);p.pop(key);p.pop("fingerprint")
            with self.subTest(key=key),self.assertRaises(ValueError):subject.validate_payload(subject.signed(p))
        p=copy.deepcopy(self.payload);p["command"]=["arbitrary"];p.pop("fingerprint")
        with self.assertRaises(ValueError):subject.validate_payload(subject.signed(p))

    def test_metadata_route_is_bound_read_only_and_allowlisted(self):
        self.run_watchdog();path=self.root/"out/watchdog-input.json";subject.write_new_json(path,self.payload)
        args=dict(expected_payload_sha256=subject.sha(path),payload_fingerprint=self.payload["fingerprint"],
            session=self.session,nonce="a"*32,local_path=self.ops.local_path)
        with patch.object(subject.subprocess,"Popen") as started,patch.object(subject,"pidfd_send_signal") as sent:
            v=subject.metadata_snapshot("/workspace/out/watchdog-input.json",**args)
        self.assertEqual(set(v["receipts"]),set(subject.RECEIPT_NAMES));self.assertEqual(v["nonce"],"a"*32)
        self.assertEqual(v["progress"][-1]["snapshot"]["evaluation_requests"],210)
        started.assert_not_called();sent.assert_not_called()
        with self.assertRaises(ValueError):subject.metadata_snapshot("/workspace/out/watchdog-input.json",**dict(args,nonce="bad"))

    def test_metadata_rejects_linked_or_foreign_receipts(self):
        self.run_watchdog();path=self.root/"out/watchdog-input.json";subject.write_new_json(path,self.payload)
        args=dict(expected_payload_sha256=subject.sha(path),payload_fingerprint=self.payload["fingerprint"],session=self.session,nonce="a"*32,local_path=self.ops.local_path)
        owner=self.root/"out/runtime-owner.json";raw=owner.read_bytes();value=json.loads(raw);value["session"]="2"*32;value.pop("fingerprint")
        owner.write_text(json.dumps(subject.signed(value)))
        with self.assertRaises(ValueError):subject.metadata_snapshot("/workspace/out/watchdog-input.json",**args)
        owner.write_bytes(raw);target=owner.with_suffix(".saved");owner.rename(target);owner.symlink_to(target)
        with self.assertRaises(ValueError):subject.metadata_snapshot("/workspace/out/watchdog-input.json",**args)

    def test_pidfd_rechecks_identity_and_ancestry_without_pid_or_group_kill(self):
        record=dict(pid=900,start_token="boot:1",cookie=self.session,ppid=777,sid=900,state="R")
        ops=subject.LinuxProcessOps()
        with patch.object(ops,"identity",side_effect=[record,dict(record,start_token="boot:2")]), \
            patch.object(subject,"pidfd_open",return_value=44),patch.object(subject.os,"close") as closed, \
            patch.object(subject,"pidfd_send_signal") as sent:
            self.assertEqual(ops.signal_owned(record,signal.SIGTERM,self.session),"ownership_mismatch")
        closed.assert_called_once_with(44);sent.assert_not_called()
        with patch.object(ops,"identity",return_value=record),patch.object(ops,"owned",return_value=[record]), \
            patch.object(subject,"pidfd_open",return_value=44),patch.object(subject.os,"close"), \
            patch.object(subject,"pidfd_send_signal") as sent,patch.object(subject.os,"kill") as kill,patch.object(subject.os,"killpg") as group:
            self.assertEqual(ops.signal_owned(record,signal.SIGTERM,self.session),"signalled")
        sent.assert_called_once_with(44,signal.SIGTERM,None,0);kill.assert_not_called();group.assert_not_called()

    def test_one_reused_helper_does_not_skip_cleanup_of_other_owned_workers(self):
        owner=dict(pid=888,start_token="boot:88",cookie=self.session)
        live={888:dict(owner),999:dict(pid=999,start_token="boot:99",cookie=self.session)}
        calls=[]
        ops=type("CleanupOps",(),{})()
        ops.identity=lambda pid:copy.deepcopy(live.get(pid))
        ops.owned=lambda pid,session:[copy.deepcopy(r) for r in live.values()]
        ops.monotonic=lambda:100.
        ops.sleep=lambda seconds:None
        def signal_one(record,signum,session):
            calls.append((record["pid"],signum))
            if record["pid"]==999:return "ownership_mismatch"
            live.pop(record["pid"],None);return "signalled"
        ops.signal_owned=signal_one
        result=subject.stop_owned(ops,owner,self.session,deadline=100.)
        self.assertEqual(result,"ownership_mismatch")
        self.assertEqual(calls,[(999,signal.SIGTERM),(888,signal.SIGTERM)])
        self.assertNotIn(888,live);self.assertIn(999,live)

    def test_import_bootstrap_is_stdlib_only_and_does_no_process_work(self):
        source=str(Path(__file__).resolve().parents[1])
        code="import sys;sys.path.insert(0,sys.argv[1]);import candidate_v4_watchdog;import candidate_v4_runtime;assert not {'numpy','scipy','torch','isaaclab','isaacsim'} & sys.modules.keys()"
        subprocess.run([sys.executable,"-B","-S","-c",code,source],check=True,capture_output=True,text=True,timeout=10)


if __name__=="__main__":unittest.main()
