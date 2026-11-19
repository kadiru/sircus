"""Fake process/procfs tests; never starts or signals an OS process."""

import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_continuation_watchdog as subject
from candidate_continuation_launcher import MIN_FREE_BYTES
from test_candidate_continuation_launcher import diagnostic_records,successful_progress


class FakeOps:
    def __init__(self,root,session):
        self.root,self.session,self.t = root,session,10.
        self.started,self.signalled,self.prepared = [],[],False
        self.running,self.exit_code,self.on_sleep = False,None,None
        self.free = MIN_FREE_BYTES+100
        self.records = {777:dict(pid=777,start_token="boot:77",cookie=session,ppid=1,sid=777,state="R")}
        self.snapshots = successful_progress()
    def local_path(self,path): return self.root/Path(path).relative_to("/workspace")
    def own_pid(self): return 777
    def monotonic(self): return self.t
    def free_bytes(self,path): return self.free
    def prepare(self): self.prepared=True
    def identity(self,pid): return copy.deepcopy(self.records.get(pid))
    def start(self,command,environment):
        self.started.append((command,environment)); self.running=True
        self.records[888]=dict(pid=888,start_token="boot:88",cookie=self.session,ppid=777,sid=888,state="S")
        if self.snapshots is not None:
            writer=subject.ProgressWriter(self.root/"out",self.session,clock=self.monotonic)
            for snapshot in self.snapshots: writer.publish(snapshot)
            self.exit_code=0
        return 888
    def poll(self,pid):
        if self.exit_code is not None: self.running=False; self.records.pop(888,None)
        return self.exit_code
    def owned(self,pid,session):
        return [copy.deepcopy(self.records[888])] if self.running and 888 in self.records else []
    def signal_owned(self,record,signum,session):
        self.signalled.append((record,signum,session)); self.running=False; self.records.pop(888,None)
        return "signalled"
    def sleep(self,seconds):
        self.t+=seconds
        if self.on_sleep: self.on_sleep(self)


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve();(self.root/"out").mkdir();(self.root/"source/sircus/isaac").mkdir(parents=True)
        self.manifest=self.root/"manifest.json";self.manifest.write_text("{}")
        source=self.root/"source/sircus/isaac/candidate_continuation_entry.py";source.write_text("# synthetic only\n")
        self.session="1"*32
        self.payload=subject.signed(dict(schema=subject.SCHEMA,session=self.session,source_root="/workspace/source",
            manifest="/workspace/manifest.json",manifest_sha256=subject.sha(self.manifest),output="/workspace/out",
            absolute_deadline=10810.,source_members_sha256={"sircus/isaac/candidate_continuation_entry.py":subject.sha(source)}))
        self.ops=FakeOps(self.root,self.session)
    def run_guarded(self): return subject.run_guarded(self.payload,self.ops)

    def test_one_owned_start_fixed_command_and_completed_receipts(self):
        result=self.run_guarded()
        self.assertEqual(result["status"],"complete_audit_receipts_required")
        self.assertEqual(len(self.ops.started),1);self.assertFalse(self.ops.signalled)
        command,environment=self.ops.started[0]
        self.assertIn("--execute-approved",command);self.assertIn("--expected-manifest-sha256",command)
        self.assertEqual(environment[subject.COOKIE],self.session)
        self.assertTrue((self.root/"out/runtime-owner.json").is_file())
        self.assertTrue((self.root/"out/runtime-exit.json").is_file())
        self.assertFalse(result["model_promoted"]);self.assertFalse(result["independent_audit_passed"])
        with self.assertRaisesRegex(ValueError,"retry"):self.run_guarded()
        self.assertEqual(len(self.ops.started),1)

    def test_independent_phase_timer_stops_a_child_that_never_reports_or_returns(self):
        self.ops.snapshots=None;self.ops.on_sleep=lambda ops:setattr(ops,"t",611.)
        result=self.run_guarded()
        self.assertEqual(result["status"],"failed_no_retry")
        self.assertIn("phase wall",result["reason"]);self.assertEqual(len(self.ops.started),1)
        self.assertEqual([s[1] for s in self.ops.signalled],[signal.SIGTERM])

    def test_absolute_timer_is_independent_of_supervisor_and_stops_owned_child(self):
        self.ops.snapshots=None;self.ops.on_sleep=lambda ops:setattr(ops,"t",10811.)
        result=self.run_guarded();self.assertIn("absolute",result["reason"])
        self.assertEqual(len(self.ops.signalled),1)

    def test_hard_deadline_does_not_add_a_thirty_second_term_grace(self):
        self.ops.snapshots=None;self.ops.on_sleep=lambda ops:setattr(ops,"t",10811.)
        def ignore_term(record,signum,session):
            self.ops.signalled.append((record,signum,session))
            if signum==signal.SIGKILL:self.ops.running=False;self.ops.records.pop(888,None)
            return "signalled"
        self.ops.signal_owned=ignore_term
        result=self.run_guarded()
        self.assertEqual([s[1] for s in self.ops.signalled],[signal.SIGTERM,signal.SIGKILL])
        self.assertEqual(self.ops.t,10811.);self.assertEqual(result["termination"],"stopped")

    def test_source_or_storage_change_stops_once_and_preserves_evidence(self):
        for kind in ("source","storage"):
            with self.subTest(kind=kind):
                # Each scenario is a separate synthetic output history.
                out=self.root/(kind+"-out");out.mkdir();self.payload["output"]="/workspace/"+kind+"-out"
                self.payload=subject.signed({k:v for k,v in self.payload.items() if k!="fingerprint"})
                ops=FakeOps(self.root,self.session);ops.snapshots=None
                if kind=="source":
                    source=self.root/"source/sircus/isaac/candidate_continuation_entry.py";original=source.read_bytes()
                    def mutate(o):o.t=12.;source.write_text("changed")
                    ops.on_sleep=mutate
                else:ops.on_sleep=lambda o:(setattr(o,"t",12.),setattr(o,"free",MIN_FREE_BYTES-1))
                result=subject.run_guarded(self.payload,ops)
                self.assertEqual(result["status"],"failed_no_retry");self.assertEqual(len(ops.signalled),1)
                self.assertTrue((out/"runtime-exit.json").exists())
                if kind=="source":source.write_bytes(original)

    def test_future_or_reversed_phase_clock_and_counter_overrun_fail(self):
        self.ops.snapshots=successful_progress()
        self.ops.snapshots[2]["train_requests"]=1751
        result=self.run_guarded();self.assertIn("budget",result["reason"])

    def test_phase_skip_and_premature_assessment_never_finish(self):
        self.ops.snapshots=[successful_progress()[4]]
        result=self.run_guarded();self.assertIn("phase order",result["reason"])

    def test_checkpoint_boundary_and_cpu_only_audit_are_enforced(self):
        snapshots=successful_progress();snapshots[-1]["audit_cpu_only"]=False
        self.ops.snapshots=snapshots
        result=self.run_guarded();self.assertIn("CPU-only",result["reason"])

    def test_reused_native_pid_is_never_signalled(self):
        self.ops.snapshots=None
        def reused(ops):ops.t=611.;ops.records[888]["start_token"]="boot:999"
        self.ops.on_sleep=reused
        result=self.run_guarded()
        self.assertEqual(result["termination"],"ownership_mismatch");self.assertFalse(self.ops.signalled)

    def test_manifest_tamper_fails_before_any_start(self):
        self.manifest.write_text("changed")
        with self.assertRaisesRegex(ValueError,"manifest bytes"):self.run_guarded()
        self.assertFalse(self.ops.started)

    def test_payload_cannot_supply_arbitrary_command_or_escape_mount(self):
        bad=copy.deepcopy(self.payload);bad["command"]=["arbitrary"]
        with self.assertRaises(ValueError):subject.validate_payload(bad)
        bad=copy.deepcopy(self.payload);bad["source_root"]="/outside"
        bad=subject.signed({k:v for k,v in bad.items() if k!="fingerprint"})
        with self.assertRaisesRegex(ValueError,"workspace"):subject.validate_payload(bad)

    def test_progress_is_append_only_session_bound_and_rejects_rehash_tamper(self):
        writer=subject.ProgressWriter(self.root/"out",self.session,clock=lambda:10.)
        writer.publish(successful_progress()[0]);writer.publish(successful_progress()[1])
        self.assertEqual(len(subject.read_progress(self.root/"out",self.session)),2)
        with self.assertRaises(FileExistsError):subject.ProgressWriter(self.root/"out",self.session)
        with self.assertRaisesRegex(ValueError,"session/chain"):subject.read_progress(self.root/"out","2"*32)
        path=self.root/"out/runtime-progress/event-000001.json";raw=json.loads(path.read_text());raw["snapshot"]["phase"]="audit"
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError,"chain"):subject.read_progress(self.root/"out",self.session)

    def test_progress_gaps_symlinks_and_future_records_are_not_hidden(self):
        writer=subject.ProgressWriter(self.root/"out",self.session,clock=lambda:10.)
        writer.publish(successful_progress()[0]);writer.publish(successful_progress()[1])
        (self.root/"out/runtime-progress/event-000000.json").unlink()
        with self.assertRaisesRegex(ValueError,"gap"):subject.read_progress(self.root/"out",self.session)

    def test_pidfd_signal_pins_identity_and_refuses_reuse(self):
        record=dict(pid=900,start_token="boot:1",cookie=self.session,ppid=777,sid=900,state="R")
        ops=subject.LinuxProcessOps()
        with patch.object(ops,"identity",side_effect=[record,dict(record,start_token="boot:2")]), \
             patch.object(subject.os,"pidfd_open",return_value=44,create=True) as opened, \
             patch.object(subject.os,"close") as closed, \
             patch.object(subject.signal,"pidfd_send_signal",create=True) as signal_call:
            self.assertEqual(ops.signal_owned(record,signal.SIGTERM,self.session),"ownership_mismatch")
        opened.assert_called_once_with(900,0);closed.assert_called_once_with(44);signal_call.assert_not_called()

    def test_pidfd_signal_does_not_use_killpg_or_pid_signal(self):
        record=dict(pid=900,start_token="boot:1",cookie=self.session,ppid=777,sid=900,state="R")
        ops=subject.LinuxProcessOps()
        with patch.object(ops,"identity",return_value=record),patch.object(ops,"owned",return_value=[record]), \
             patch.object(subject.os,"pidfd_open",return_value=44,create=True),patch.object(subject.os,"close"), \
             patch.object(subject.signal,"pidfd_send_signal",create=True) as sent,patch.object(subject.os,"kill") as kill, \
             patch.object(subject.os,"killpg") as group:
            self.assertEqual(ops.signal_owned(record,signal.SIGTERM,self.session),"signalled")
        sent.assert_called_once_with(44,signal.SIGTERM,None,0);kill.assert_not_called();group.assert_not_called()

    def test_cookie_scrubbed_descendants_are_owned_but_foreign_processes_are_not(self):
        records={777:dict(pid=777,start_token="boot:77",cookie=self.session,ppid=1,sid=777,state="R"),
            888:dict(pid=888,start_token="boot:88",cookie=self.session,ppid=777,sid=888,state="S"),
            999:dict(pid=999,start_token="boot:99",cookie="",ppid=888,sid=888,state="S"),
            111:dict(pid=111,start_token="boot:11",cookie=self.session,ppid=1,sid=111,state="S")}
        ops=subject.LinuxProcessOps()
        with patch.object(ops,"identity",side_effect=lambda pid:copy.deepcopy(records.get(pid))), \
             patch.object(ops,"own_pid",return_value=777), \
             patch.object(Path,"iterdir",return_value=[Path("/proc")/str(pid) for pid in records]):
            owned=ops.owned(888,self.session)
        self.assertEqual({r["pid"] for r in owned},{888,999})

    def test_dead_adopted_workers_are_reaped_without_stealing_native_exit_status(self):
        records={777:dict(pid=777,start_token="boot:77",cookie=self.session,ppid=1,sid=777,state="R"),
            888:dict(pid=888,start_token="boot:88",cookie="",ppid=777,sid=888,state="Z"),
            999:dict(pid=999,start_token="boot:99",cookie="",ppid=777,sid=888,state="Z")}
        ops=subject.LinuxProcessOps();ops.child=types.SimpleNamespace(pid=888)
        with patch.object(ops,"identity",side_effect=lambda pid:copy.deepcopy(records.get(pid))), \
             patch.object(ops,"own_pid",return_value=777), \
             patch.object(Path,"iterdir",return_value=[Path("/proc")/str(pid) for pid in records]), \
             patch.object(subject.os,"waitpid",return_value=(999,0)) as wait:
            self.assertEqual(ops.owned(888,self.session),[])
        wait.assert_called_once_with(999,subject.os.WNOHANG)

    def test_watchdog_stop_rechecks_cookie_and_start_after_pidfd_open(self):
        value=dict(pid=900,start_token="boot:1",session=self.session)
        record=dict(pid=900,start_token="boot:1",cookie=self.session)
        with patch.object(subject,"inspect_watchdog",side_effect=[record,dict(record,start_token="boot:2")]), \
             patch.object(subject.os,"pidfd_open",return_value=45,create=True),patch.object(subject.os,"close"), \
             patch.object(subject.signal,"pidfd_send_signal",create=True) as sent:
            self.assertEqual(subject.stop_watchdog(value)["status"],"ownership_mismatch")
        sent.assert_not_called()

    def test_proc_stat_parser_handles_parenthesized_names_and_start_ticks(self):
        proc=self.root/"proc";(proc/"900").mkdir(parents=True);(proc/"sys/kernel/random").mkdir(parents=True)
        (proc/"sys/kernel/random/boot_id").write_text("boot")
        tail=["S","777","900","900"]+["0"]*15+["12345"]+["0"]*3
        (proc/"900/stat").write_text("900 (a tricky (name)) "+" ".join(tail))
        (proc/"900/environ").write_bytes((subject.COOKIE+"="+self.session).encode()+b"\0")
        value=subject.proc_identity(900,proc)
        self.assertEqual(value["start_token"],"boot:12345");self.assertEqual(value["ppid"],777)
        self.assertEqual(value["cookie"],self.session)


if __name__=="__main__":unittest.main()
