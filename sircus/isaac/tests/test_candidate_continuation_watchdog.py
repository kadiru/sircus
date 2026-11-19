"""Fake process/procfs tests plus an inert standard-library import probe."""

import copy
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
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
    def test_supervision_bootstrap_needs_no_site_packages(self):
        from candidate_continuation_contract import PROTOCOL_SHA256
        source = str(Path(__file__).resolve().parents[1])
        probe = "\n".join((
            "import sys",
            "sys.path.insert(0, sys.argv[1])",
            "import candidate_continuation_contract as contract",
            "import candidate_continuation_launcher",
            "import candidate_continuation_journal",
            "import candidate_continuation_watchdog",
            "import candidate_continuation_runtime",
            "assert not {'numpy', 'scipy', 'torch', 'isaaclab', 'isaacsim'} & sys.modules.keys()",
            "assert contract.PROTOCOL_SHA256 == '68d59f401efc411c9fd5590ac70819d7793d7d223d74d084e85b9b29c2f7ae2c'",
            "print(contract.PROTOCOL_SHA256)",
        ))
        result = subprocess.run([sys.executable, "-B", "-S", "-c", probe, source],
            check=True, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.stdout.strip(), PROTOCOL_SHA256)

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

    def test_event_published_during_real_progress_read_is_not_falsely_future(self):
        self.ops.snapshots=None
        start=self.ops.start;writer=[]
        def started(command,environment):
            pid=start(command,environment)
            writer.append(subject.ProgressWriter(self.root/"out",self.session,clock=self.ops.monotonic))
            writer[0].publish(successful_progress()[0])
            return pid
        self.ops.start=started
        original_iterdir=Path.iterdir;interleaved=[]
        def iterdir(path):
            if path==self.root/"out/runtime-progress" and not interleaved:
                # The real reader has started after the watchdog's old clock
                # sample. A concurrent writer publishes valid signed events.
                self.ops.t+=.025;interleaved.append(self.ops.t)
                for snapshot in successful_progress()[1:]:writer[0].publish(snapshot)
                self.ops.exit_code=0
            return original_iterdir(path)
        with patch.object(Path,"iterdir",iterdir):result=self.run_guarded()
        self.assertEqual(interleaved,[10.025])
        self.assertEqual(result["status"],"complete_audit_receipts_required",result["reason"])
        self.assertEqual(result["progress_events"],8);self.assertFalse(self.ops.signalled)

    def staged_event_history(self,snapshots,stamps,observed_time):
        """Publish real signed fixtures with event and reader clocks separated."""
        self.ops.snapshots=None;start=self.ops.start
        def started(command,environment):
            pid=start(command,environment);ticks=iter(stamps)
            writer=subject.ProgressWriter(self.root/"out",self.session,clock=lambda:next(ticks))
            for snapshot in snapshots:writer.publish(snapshot)
            self.ops.t=observed_time;self.ops.exit_code=0
            return pid
        self.ops.start=started

    def test_event_newer_than_fresh_post_read_clock_is_still_rejected(self):
        self.staged_event_history(successful_progress(),[10.025]*8,10.)
        result=self.run_guarded()
        self.assertEqual(result["status"],"failed_no_retry");self.assertIn("future phase receipt",result["reason"])
        self.assertEqual(result["progress_events"],0);self.assertEqual(result["termination"],"stopped")

    def test_delayed_phase_transition_cannot_extend_original_phase_cap(self):
        self.staged_event_history(successful_progress()[:2],[10.,610.025],611.)
        result=self.run_guarded()
        self.assertEqual(result["status"],"failed_no_retry");self.assertIn("Delayed or future phase receipt",result["reason"])
        self.assertEqual(result["phase"],"initialization");self.assertEqual(result["termination"],"stopped")

    def test_compliant_transition_observed_later_keeps_its_actual_event_time(self):
        self.staged_event_history(successful_progress(),[10.]+[609.5]*7,611.)
        result=self.run_guarded()
        self.assertEqual(result["status"],"complete_audit_receipts_required",result["reason"])
        self.assertFalse(self.ops.signalled)

    def test_late_observation_does_not_reset_new_phase_start_to_read_time(self):
        self.staged_event_history(successful_progress()[:2],[10.,11.],612.)
        result=self.run_guarded()
        self.assertEqual(result["status"],"failed_no_retry");self.assertIn("phase wall limit",result["reason"])
        self.assertEqual(result["phase"],"imitation");self.assertEqual(result["termination"],"stopped")

    def test_reversed_event_clock_is_still_rejected(self):
        self.staged_event_history(successful_progress()[:2],[10.,9.],10.)
        result=self.run_guarded()
        self.assertEqual(result["status"],"failed_no_retry");self.assertIn("progress clock moved backwards",result["reason"])
        self.assertEqual(result["termination"],"stopped")

    def assert_clock_failure(self,label,clock_values):
        out=self.root/label;out.mkdir()
        payload=subject.signed(dict({k:v for k,v in self.payload.items() if k!="fingerprint"},output="/workspace/"+label))
        ops=FakeOps(self.root,self.session);ops.snapshots=None
        with patch.object(ops,"monotonic",side_effect=clock_values):result=subject.run_guarded(payload,ops)
        self.assertEqual(result["status"],"failed_no_retry");self.assertIn("monotonic clock changed",result["reason"])
        self.assertEqual(result["termination"],"stopped");self.assertEqual(len(ops.started),1)

    def test_post_read_clock_must_remain_finite_and_non_decreasing(self):
        for label,bad in (("backwards",9.),("nonfinite-inf",float("inf")),("nonfinite-nan",float("nan"))):
            with self.subTest(clock=label):self.assert_clock_failure("read-"+label,[10.,10.,bad,10.])

    def test_post_validation_clock_must_remain_finite_and_non_decreasing(self):
        for label,bad in (("backwards",9.),("nonfinite-inf",float("inf")),("nonfinite-nan",float("nan"))):
            with self.subTest(clock=label):self.assert_clock_failure("validation-"+label,[10.,10.,10.,bad,10.])

    def test_progress_io_crossing_absolute_deadline_is_rejected_before_event_processing(self):
        original=subject.read_progress
        def read(output,session):
            events=original(output,session);self.ops.t=10810.025;return events
        with patch.object(subject,"read_progress",side_effect=read):result=self.run_guarded()
        self.assertEqual(result["status"],"failed_no_retry");self.assertIn("absolute native wall limit",result["reason"])
        self.assertEqual(result["progress_events"],0);self.assertEqual(result["termination"],"stopped")

    def validation_crosses_deadline(self,target):
        original=subject.DiagnosticLedger.observe
        def observe(ledger,records,resets):
            result=original(ledger,records,resets);self.ops.t=target;return result
        with patch.object(subject.DiagnosticLedger,"observe",observe):return self.run_guarded()

    def test_validation_time_crossing_absolute_deadline_is_rejected_before_exit_acceptance(self):
        result=self.validation_crosses_deadline(10810.025)
        self.assertEqual(result["status"],"failed_no_retry");self.assertIn("absolute native wall limit",result["reason"])
        self.assertEqual(result["progress_events"],8);self.assertEqual(result["termination"],"stopped")

    def test_validation_time_crossing_current_phase_deadline_is_rejected_before_exit_acceptance(self):
        result=self.validation_crosses_deadline(610.025)
        self.assertEqual(result["status"],"failed_no_retry");self.assertIn("phase wall limit",result["reason"])
        self.assertEqual(result["phase"],"audit");self.assertEqual(result["progress_events"],8)
        self.assertEqual(result["termination"],"stopped")

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
             patch.object(subject,"pidfd_open",return_value=44) as opened, \
             patch.object(subject.os,"close") as closed, \
             patch.object(subject,"pidfd_send_signal") as signal_call:
            self.assertEqual(ops.signal_owned(record,signal.SIGTERM,self.session),"ownership_mismatch")
        opened.assert_called_once_with(900,0);closed.assert_called_once_with(44);signal_call.assert_not_called()

    def test_self_pidfd_capability_is_checked_before_subreaper_or_application_start(self):
        import ctypes
        ops=subject.LinuxProcessOps();order=[]
        capability=dict(status="passed",self_only=True,signal=0,descriptor_closed=True)
        library=unittest.mock.Mock()
        library.prctl.side_effect=lambda *args:order.append("subreaper") or 0
        with patch.object(subject,"capability_preflight",side_effect=lambda:order.append("capability") or capability), \
             patch.object(ctypes,"CDLL",return_value=library),patch.object(Path,"exists",return_value=False), \
             patch.object(subject.subprocess,"Popen") as started:
            ops.prepare()
        self.assertEqual(order,["capability","subreaper"])
        self.assertEqual(ops.pidfd_capability,capability);started.assert_not_called()

    def test_failed_pidfd_capability_prevents_subreaper_and_application_start(self):
        import ctypes
        ops=subject.LinuxProcessOps()
        with patch.object(subject,"capability_preflight",side_effect=OSError("mock unsupported pidfd")), \
             patch.object(ctypes,"CDLL") as library,patch.object(subject.subprocess,"Popen") as started:
            with self.assertRaises(OSError):ops.prepare()
        library.assert_not_called();started.assert_not_called()

    def test_pidfd_signal_refuses_lost_ancestry_after_open_and_closes_descriptor(self):
        record=dict(pid=900,start_token="boot:1",cookie=self.session,ppid=777,sid=900,state="R")
        ops=subject.LinuxProcessOps()
        with patch.object(ops,"identity",return_value=record),patch.object(ops,"owned",return_value=[]), \
             patch.object(subject,"pidfd_open",return_value=44),patch.object(subject.os,"close") as closed, \
             patch.object(subject,"pidfd_send_signal") as sent:
            self.assertEqual(ops.signal_owned(record,signal.SIGTERM,self.session),"ownership_mismatch")
        closed.assert_called_once_with(44);sent.assert_not_called()

    def test_pidfd_signal_does_not_use_killpg_or_pid_signal(self):
        record=dict(pid=900,start_token="boot:1",cookie=self.session,ppid=777,sid=900,state="R")
        ops=subject.LinuxProcessOps()
        with patch.object(ops,"identity",return_value=record),patch.object(ops,"owned",return_value=[record]), \
             patch.object(subject,"pidfd_open",return_value=44),patch.object(subject.os,"close"), \
             patch.object(subject,"pidfd_send_signal") as sent,patch.object(subject.os,"kill") as kill, \
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
             patch.object(subject,"pidfd_open",return_value=45),patch.object(subject.os,"close"), \
             patch.object(subject,"pidfd_send_signal") as sent:
            self.assertEqual(subject.stop_watchdog(value)["status"],"ownership_mismatch")
        sent.assert_not_called()

    def metadata_fixture(self):
        # FakeOps fabricates one full history; every process boundary is fake.
        self.run_guarded()
        path=self.root/"out/watchdog-input.json";subject.write_new_json(path,self.payload)
        return dict(expected_payload_sha256=subject.sha(path),payload_fingerprint=self.payload["fingerprint"],
            session=self.session,nonce="a"*32,local_path=self.ops.local_path)

    def test_metadata_route_exports_only_bound_0600_receipts_and_complete_progress_read_only(self):
        args=self.metadata_fixture();before_started=len(self.ops.started);before_signals=len(self.ops.signalled)
        secret=self.root/"out/learning";secret.mkdir();(secret/"final.pt").write_bytes(b"private model must not be opened")
        allowed={self.root/"out/watchdog-input.json",self.root/"out/runtime-owner.json",self.root/"out/runtime-exit.json",
            *(self.root/"out/runtime-progress").glob("event-*.json")}
        original=Path.read_bytes;read=[]
        def guarded(path):
            self.assertIn(path,allowed);read.append(path);return original(path)
        with patch.object(Path,"read_bytes",guarded),patch.object(subject.subprocess,"Popen") as start, \
                patch.object(subject,"pidfd_open") as opened,patch.object(subject,"pidfd_send_signal") as sent:
            value=subject.metadata_snapshot("/workspace/out/watchdog-input.json",**args)
        self.assertEqual(set(value["receipts"]),set(subject.RECEIPT_NAMES));self.assertEqual(value["nonce"],"a"*32)
        self.assertEqual(len(value["progress"]),len(self.ops.snapshots));self.assertEqual(value["progress"][-1]["snapshot"]["assessment_requests"],420)
        self.assertEqual(value["receipts"]["runtime-exit.json"]["exit_code"],0)
        for path in allowed:self.assertEqual(path.stat().st_mode & 0o777,0o600)
        start.assert_not_called();opened.assert_not_called();sent.assert_not_called()
        self.assertEqual(len(self.ops.started),before_started);self.assertEqual(len(self.ops.signalled),before_signals)
        self.assertNotIn("private model",json.dumps(value));self.assertNotIn(secret/"final.pt",read)

    def test_metadata_payload_hash_fingerprint_session_nonce_and_target_cannot_rebind(self):
        args=self.metadata_fixture()
        for key,value in (("expected_payload_sha256","0"*64),("payload_fingerprint","0"*64),
                ("session","2"*32),("nonce","not-a-nonce")):
            with self.subTest(field=key),self.assertRaises(ValueError):
                subject.metadata_snapshot("/workspace/out/watchdog-input.json",**dict(args,**{key:value}))
        for target in ("/workspace/out/../out/watchdog-input.json","/workspace/out/./watchdog-input.json",
                "/etc/watchdog-input.json","/workspace/out/learning/final.pt","/workspace/out//watchdog-input.json"):
            with self.subTest(path=target),self.assertRaises(ValueError):subject.metadata_snapshot(target,**args)
        (self.root/"other").mkdir();subject.write_new_json(self.root/"other/watchdog-input.json",self.payload)
        with self.assertRaisesRegex(ValueError,"bound payload/session/output"):
            subject.metadata_snapshot("/workspace/other/watchdog-input.json",**args)

    def test_metadata_rejects_linked_payload_receipts_and_progress_directory(self):
        args=self.metadata_fixture()
        for relative in ("watchdog-input.json","runtime-owner.json","runtime-exit.json","runtime-progress"):
            target=self.root/"out"/relative;saved=target.with_name(target.name+"-saved")
            target.rename(saved);target.symlink_to(saved,target_is_directory=saved.is_dir())
            with self.subTest(path=relative),self.assertRaisesRegex(ValueError,"Linked|noncanonical"):
                subject.metadata_snapshot("/workspace/out/watchdog-input.json",**args)
            target.unlink();saved.rename(target)
        (self.root/"out/runtime-exit.json").unlink();(self.root/"out/runtime-exit.json").symlink_to("missing")
        with self.assertRaisesRegex(ValueError,"Linked|noncanonical"):
            subject.metadata_snapshot("/workspace/out/watchdog-input.json",**args)

    def test_metadata_rejects_foreign_rehashed_receipt_payload_and_progress(self):
        args=self.metadata_fixture();owner_path=self.root/"out/runtime-owner.json";original=owner_path.read_bytes()
        for key,value in (("session","2"*32),("payload_sha256","b"*64)):
            owner=json.loads(original);owner[key]=value;owner.pop("fingerprint");owner_path.write_text(json.dumps(subject.signed(owner)))
            with self.subTest(field=key),self.assertRaisesRegex(ValueError,"receipt session/fingerprint/payload"):
                subject.metadata_snapshot("/workspace/out/watchdog-input.json",**args)
        owner_path.write_bytes(original)
        path=self.root/"out/runtime-progress/event-000000.json";event=json.loads(path.read_bytes())
        event["session"]="2"*32;event.pop("fingerprint");path.write_text(json.dumps(subject.signed(event)))
        with self.assertRaisesRegex(ValueError,"progress session/chain"):
            subject.metadata_snapshot("/workspace/out/watchdog-input.json",**args)

    def test_snapshot_cli_is_cookie_bound_read_only_and_cannot_dispatch_an_application(self):
        argv=["watchdog","--snapshot","/workspace/out/watchdog-input.json","--session",self.session,
            "--expected-payload-sha256","a"*64,"--payload-fingerprint","b"*64,"--nonce","c"*32]
        with patch.object(sys,"argv",argv),patch.dict(subject.os.environ,{subject.COOKIE:self.session}), \
                patch.object(subject,"metadata_snapshot",return_value={"synthetic":"metadata only"}) as read, \
                patch.object(subject,"run_guarded") as run,patch.object(sys,"stdout",io.StringIO()) as output:
            self.assertEqual(subject.main(),0)
        run.assert_not_called();self.assertEqual(json.loads(output.getvalue()),{"synthetic":"metadata only"})
        read.assert_called_once_with("/workspace/out/watchdog-input.json",expected_payload_sha256="a"*64,
            payload_fingerprint="b"*64,session=self.session,nonce="c"*32)
        with patch.object(sys,"argv",argv),patch.dict(subject.os.environ,{subject.COOKIE:"2"*32}), \
                patch.object(subject,"metadata_snapshot") as read,patch.object(subject,"run_guarded") as run:
            with self.assertRaisesRegex(ValueError,"session cookie"):subject.main()
        read.assert_not_called();run.assert_not_called()

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
