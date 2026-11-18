"""CPU supervisor mocks only; no simulator, real child, GPU or service start."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import location_comparison_recovery_watchdog as subject

SESSION='a'*32

def request(root):
    return subject.signed(dict(schema='sircus-location-comparison-recovery-request-v1',
        native_output=str(root/'native'),inventory_path=str(root/'inventory.json'),
        retirement_proof_path=str(root/'retirement.json'),source_commit='1'*40,
        manifest_sha256='2'*64,manifest_fingerprint='3'*64,comparison_fingerprint='4'*64,
        native_session_id='5'*32,native_exit_fingerprint='6'*64,native_exit_sha256='7'*64,
        inventory_sha256='8'*64,retirement_proof_sha256='9'*64))

def report_signed(value):return dict(value,fingerprint=subject.scientific_digest(value))

def report(req):
    return report_signed(dict(schema=subject.REPORT_SCHEMA,status='data_verified_shutdown_incomplete',
        synthetic=False,model_promoted=False,original_audit_passed=False,clean_shutdown=False,
        recovery_request_fingerprint=req['fingerprint'],**{k:req[k] for k in subject.BINDINGS},
        checkpoint_bytes_independently_verified=True,model_likelihoods_independently_verified=True,
        native_physics_verified=True))

def save(path,value):
    path.write_text(json.dumps(value,sort_keys=True)+'\n')

def checksum(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class FakeOps:
    def __init__(self,root,req):
        self.root,self.req=root,req
        self.t=100.;self.free=subject.LIMITS['min_free_bytes']+1024
        self.records={77:dict(pid=77,ppid=1,sid=77,start_token='boot:77',cookie=SESSION,state='S')}
        self.started=[];self.signalled=[];self.running=False;self.code=None;self.polls=0
        self.after_sleep=None;self.after_start=None;self.finish_after=3;self.measured=True
        self.resource=dict(cgroup_memory_limit_bytes=8*2**30,cgroup_cpu_cores=4,
            memory_current_bytes=1024**2,cpu_usage_seconds=5.,visible_gpu_devices=0)
        self.storage_paths=[]
    def monotonic(self):return self.t
    def own_pid(self):return 77
    def free_bytes(self,path):self.storage_paths.append(path);return self.free
    def identity(self,pid):return copy.deepcopy(self.records.get(pid))
    def prepare(self):pass
    def resources(self):return copy.deepcopy(self.resource)
    def start(self,argv,env):
        self.started.append((argv,env));self.running=True
        self.records[88]=dict(pid=88,ppid=77,sid=88,start_token='boot:88',cookie=SESSION,state='S')
        if self.after_start:self.after_start(self)
        return 88
    def poll(self,pid):
        self.polls+=1
        if self.code is None and self.polls>=self.finish_after:
            save(self.root/'audit'/subject.REPORT_FILE,report(self.req));self.code=0
        if self.code is not None:
            self.running=False;self.records.pop(88,None)
        return self.code
    def owned(self,pid,session):return [copy.deepcopy(self.records[88])] if self.running and 88 in self.records else []
    def signal_owned(self,record,sig,session):
        self.signalled.append((record,sig,session));self.running=False;self.records.pop(88,None);self.code=-sig
        return 'signalled'
    def sleep(self,seconds):
        self.t+=seconds
        if self.after_sleep:self.after_sleep(self)
    def measurement(self):
        return dict(cpu_user_s=.2,cpu_system_s=.1,peak_rss_bytes=65536,
            measurement_scope='wait4_exact_audit_child_including_reaped_descendants') if self.measured else None

class RecoveryWatchdogTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();(self.root/'native').mkdir();(self.root/'disk').mkdir()
        self.req=request(self.root);self.ops=FakeOps(self.root,self.req)
    def run_audit(self,**kw):
        args=dict(output=self.root/'audit',storage_root=self.root/'disk',request=self.req,
                  session_id=SESSION,integrity_check=lambda:True,verify_report=lambda *a:True,
                  ops=self.ops,execute_approved=True)
        args.update(kw)
        return subject.supervise_recovery(['/fixed/python','/fixed/reader','--read-only'],**args)
    def verify(self,result):
        return subject.verify_exit(self.root/'audit',self.req,SESSION,
            expected_exit_sha256=checksum(self.root/'audit'/subject.EXIT_FILE),
            expected_report_sha256=checksum(self.root/'audit'/subject.REPORT_FILE))
    def test_success_is_separate_status_measured_exactly_and_bound_after_exit(self):
        result=self.run_audit()
        self.assertEqual(result['status'],'recovery_exit_zero_report_verified')
        self.assertEqual(result['cpu_user_s'],.2);self.assertEqual(result['peak_rss_bytes'],65536)
        self.assertTrue(self.verify(result)['independent_report_verified'])
        self.assertEqual(len(self.ops.started),1);self.assertFalse(self.ops.signalled)
        self.assertTrue(all(p==self.root/'disk' for p in self.ops.storage_paths))
        self.assertEqual(list((self.root/'native').iterdir()),[])
        env=self.ops.started[0][1]
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(env['NVIDIA_VISIBLE_DEVICES'],'void')
        self.assertEqual(result['native_application_starts'],0);self.assertEqual(result['optimizer_steps'],0)
    def test_import_has_no_process_start(self):
        with patch.object(subprocess,'Popen',side_effect=AssertionError('start')):
            import importlib
            importlib.reload(subject)
    def test_approval_existing_output_changed_request_and_same_session_fail_before_start(self):
        for changed in ({'execute_approved':False},{'session_id':self.req['native_session_id']},
                        {'request':dict(self.req,source_commit='f'*40)}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):self.run_audit(**changed)
        (self.root/'audit').mkdir()
        with self.assertRaisesRegex(ValueError,'exists'):self.run_audit()
        self.assertFalse(self.ops.started)
    def test_nested_evidence_and_linked_outputs_fail_without_writing(self):
        with self.assertRaisesRegex(ValueError,'overlaps'):self.run_audit(output=self.root/'native'/'audit')
        (self.root/'link').symlink_to(self.root/'native',target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'canonical'):self.run_audit(output=self.root/'link'/'audit')
        self.assertFalse(self.ops.started)
    def test_request_schema_and_required_hashes_are_checked(self):
        for field in ('schema','manifest_sha256','native_exit_sha256','source_commit'):
            req=dict(self.req);req[field]='wrong';req.pop('fingerprint');req=subject.signed(req)
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_audit(request=req)
        self.assertFalse(self.ops.started)
    def test_low_actual_filesystem_floor_blocks_start_even_with_empty_output(self):
        self.ops.free=subject.LIMITS['min_free_bytes']-1
        result=self.run_audit();self.assertEqual(result['audit_process_starts'],0)
        self.assertIn('storage floor',result['reason']);self.assertTrue((self.root/'audit'/subject.EXIT_FILE).is_file())
    def test_midrun_storage_decline_stops_owned_only_and_never_retries(self):
        self.ops.finish_after=999
        self.ops.after_sleep=lambda o:setattr(o,'free',0)
        result=self.run_audit();self.assertIn('storage floor',result['reason'])
        self.assertTrue(result['cleanup_safe']);self.assertEqual(len(self.ops.started),1)
        self.assertEqual(self.ops.signalled[0][0]['pid'],88)
        with self.assertRaises(ValueError):self.run_audit()
    def test_wall_deadline_does_not_depend_on_child_progress(self):
        self.ops.finish_after=999
        self.ops.after_sleep=lambda o:setattr(o,'t',7301.)
        result=self.run_audit();self.assertIn('deadline',result['reason'])
        self.assertTrue(result['cleanup_safe']);self.assertEqual(len(self.ops.started),1)
    def test_clock_reversal_and_nonfinite_fail_closed(self):
        for value in (99.,float('nan')):
            with self.subTest(value=value):
                self.ops=FakeOps(self.root,self.req);self.ops.finish_after=999
                self.ops.after_sleep=lambda o,v=value:setattr(o,'t',v)
                result=self.run_audit(output=self.root/('audit-'+str(value)))
                self.assertIn('clock',result['reason'])
    def test_cgroup_excess_memory_cpu_or_gpu_blocks_start(self):
        changes=[('cgroup_memory_limit_bytes',9*2**30),('cgroup_cpu_cores',5),
                 ('memory_current_bytes',9*2**30),('visible_gpu_devices',1)]
        for key,value in changes:
            with self.subTest(key=key):
                self.ops=FakeOps(self.root,self.req);self.ops.resource[key]=value
                result=self.run_audit(output=self.root/key)
                self.assertEqual(result['audit_process_starts'],0);self.assertEqual(result['status'],'recovery_stopped_no_retry')
    def test_cpu_time_budget_is_independent_and_cumulative(self):
        self.ops.finish_after=999
        self.ops.after_sleep=lambda o:o.resource.update(cpu_usage_seconds=28806.)
        result=self.run_audit();self.assertIn('CPU time',result['reason'])
        self.assertTrue(result['cleanup_safe'])
    def test_changed_integrity_stops_before_start(self):
        result=self.run_audit(integrity_check=lambda:False)
        self.assertIn('immutable',result['reason']);self.assertEqual(result['audit_process_starts'],0)
    def test_changed_child_identity_is_not_accepted(self):
        self.ops.after_start=lambda o:o.records[88].update(cookie='b'*32)
        result=self.run_audit();self.assertIn('ownership',result['reason'])
        self.assertEqual(result['termination'],'unknown_child_identity');self.assertFalse(self.ops.signalled)
    def test_unreadable_live_child_never_becomes_success(self):
        original=self.ops.identity
        self.ops.identity=lambda pid: (_ for _ in ()).throw(PermissionError()) if pid==88 else original(pid)
        self.ops.finish_after=999
        result=self.run_audit();self.assertEqual(result['status'],'recovery_stopped_no_retry')
        self.assertFalse(result['independent_report_verified']);self.assertFalse(self.ops.signalled)
    def test_nonzero_exit_report_callback_failure_and_missing_usage_are_not_passes(self):
        self.ops.code=1
        result=self.run_audit();self.assertEqual(result['status'],'recovery_stopped_no_retry')
        self.assertFalse(result['independent_report_verified'])
    def test_failed_independent_report_verifier_cannot_publish(self):
        result=self.run_audit(verify_report=lambda *a:False)
        self.assertIn('verification',result['reason']);self.assertFalse(result['independent_report_verified'])
    def test_absent_wait4_usage_prevents_success(self):
        self.ops.measured=False
        result=self.run_audit();self.assertIn('measurement missing',result['reason'])
        self.assertFalse(result['independent_report_verified'])
    def test_changed_report_between_verification_and_exit_fails(self):
        def verify(output,req):
            value=report(req);value['native_physics_verified']=False
            value.pop('fingerprint');save(output/subject.REPORT_FILE,report_signed(value));return True
        result=self.run_audit(verify_report=verify)
        self.assertIn('scientific',result['reason'])
    def test_resealed_clean_shutdown_synthetic_or_wrong_binding_reports_fail(self):
        (self.root/'audit').mkdir()
        for key,value in [('clean_shutdown',True),('original_audit_passed',True),('synthetic',True),
                          ('native_exit_sha256','f'*64),('model_promoted',True)]:
            value_report=report(self.req);value_report[key]=value;value_report.pop('fingerprint')
            save(self.root/'audit'/subject.REPORT_FILE,report_signed(value_report))
            with self.subTest(key=key),self.assertRaises(ValueError):subject.inspect_report(self.root/'audit',self.req)
    def test_exit_external_pins_and_owner_ancestry_cannot_be_resealed_away(self):
        result=self.run_audit();self.verify(result)
        original=(self.root/'audit'/subject.EXIT_FILE).read_bytes()
        for field,value in [('native_exit_sha256','0'*64),('cleanup_safe',False),('audit_process_starts',2),
                            ('peak_rss_bytes',10*2**30),('measurement_scope','invented')]:
            altered=copy.deepcopy(result);altered[field]=value;altered.pop('fingerprint')
            save(self.root/'audit'/subject.EXIT_FILE,subject.signed(altered))
            with self.subTest(field=field),self.assertRaises(ValueError):self.verify(result)
        (self.root/'audit'/subject.EXIT_FILE).write_bytes(original)
        owner=json.loads((self.root/'audit'/subject.OWNER_FILE).read_text());owner['child_owner']['ppid']=999
        owner.pop('fingerprint');save(self.root/'audit'/subject.OWNER_FILE,subject.signed(owner))
        with self.assertRaisesRegex(ValueError,'owner/exit'):self.verify(result)
    def test_foreign_exit_or_report_hash_rejected(self):
        result=self.run_audit()
        with self.assertRaises(ValueError):subject.verify_exit(self.root/'audit',self.req,SESSION,
            expected_exit_sha256='f'*64,expected_report_sha256=result['report_sha256'])
        with self.assertRaises(ValueError):subject.verify_exit(self.root/'audit',self.req,SESSION,
            expected_exit_sha256=checksum(self.root/'audit'/subject.EXIT_FILE),expected_report_sha256='f'*64)
    def test_linux_wait4_measures_exact_child_without_popen_poll(self):
        ops=subject.LinuxRecoveryOps();ops.child=types.SimpleNamespace(pid=123,returncode=None)
        usage=types.SimpleNamespace(ru_utime=2.25,ru_stime=.5,ru_maxrss=1234)
        with patch.object(os,'wait4',return_value=(0,0,None)) as wait:
            self.assertIsNone(ops.poll(123));wait.assert_called_once_with(123,os.WNOHANG)
        with patch.object(os,'wait4',return_value=(123,0,usage)):
            self.assertEqual(ops.poll(123),0)
        self.assertEqual(ops.measurement()['peak_rss_bytes'],1234*1024)
        self.assertEqual(ops.measurement()['cpu_user_s'],2.25)
        with self.assertRaises(ValueError):ops.poll(999)
    def test_linux_start_is_once_devnull_and_private_session(self):
        ops=subject.LinuxRecoveryOps()
        with patch.object(subprocess,'Popen',return_value=types.SimpleNamespace(pid=123)) as popen:
            self.assertEqual(ops.start(['/fixed/reader'],{}),123)
            self.assertEqual(popen.call_args.kwargs['stdout'],subprocess.DEVNULL)
            self.assertEqual(popen.call_args.kwargs['stderr'],subprocess.DEVNULL)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            with self.assertRaisesRegex(ValueError,'consumed'):ops.start(['/fixed/reader'],{})
            self.assertEqual(popen.call_count,1)
    def test_actual_cgroup_v2_parser_requires_finite_declared_limits(self):
        cg=self.root/'cg';cg.mkdir()
        values={'cgroup.controllers':'cpu memory','memory.max':str(8*2**30),'memory.current':'1024',
                'cpu.max':'400000 100000','cpu.stat':'usage_usec 2000000\nuser_usec 1000000\n'}
        for name,value in values.items():(cg/name).write_text(value)
        with patch.object(subject,'_cpu_only_environment'):
            measured=subject.LinuxRecoveryOps(cg).resources();self.assertEqual(measured['cpu_usage_seconds'],2)
            (cg/'cpu.max').write_text('max 100000')
            with self.assertRaises(ValueError):subject.LinuxRecoveryOps(cg).resources()
    def test_child_handshake_checks_direct_live_ancestry_and_request(self):
        self.run_audit()
        saved=json.loads((self.root/'audit'/subject.OWNER_FILE).read_text())
        ops=FakeOps(self.root,self.req)
        ops.records[88]=copy.deepcopy(saved['child_owner'])
        ops.own_pid=lambda:88
        with patch.object(subject,'_cpu_only_environment'),patch.object(os,'getppid',return_value=77):
            self.assertEqual(subject.validate_child_owner(self.root/'audit',self.req,SESSION,ops=ops)['fingerprint'],saved['fingerprint'])
            ops.records[88]['ppid']=99
            with self.assertRaisesRegex(ValueError,'ancestry'):
                subject.validate_child_owner(self.root/'audit',self.req,SESSION,ops=ops)
    def test_child_handshake_wait_is_bounded_and_never_starts_a_process(self):
        with patch.object(subject,'_cpu_only_environment'),self.assertRaisesRegex(ValueError,'timed out'):
            subject.validate_child_owner(self.root/'missing',self.req,SESSION,ops=self.ops)
        self.assertFalse(self.ops.started)
    def test_actual_cgroup_v1_parser_and_cpu_visibility_fail_closed(self):
        cg=self.root/'cg';cg.mkdir()
        values={'memory/memory.limit_in_bytes':str(8*2**30),'memory/memory.usage_in_bytes':'2048',
                'cpu/cpu.cfs_quota_us':'400000','cpu/cpu.cfs_period_us':'100000','cpuacct/cpuacct.usage':'3000000000'}
        for name,value in values.items():
            path=cg/name;path.parent.mkdir(exist_ok=True);path.write_text(value)
        with patch.object(subject,'_cpu_only_environment'):
            measured=subject.LinuxRecoveryOps(cg).resources();self.assertEqual(measured['cpu_usage_seconds'],3)
            (cg/'cpu/cpu.cfs_quota_us').write_text('-1')
            with self.assertRaises(ValueError):subject.LinuxRecoveryOps(cg).resources()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0','NVIDIA_VISIBLE_DEVICES':'void'}),self.assertRaisesRegex(ValueError,'CUDA'):
            subject._cpu_only_environment()
    def test_scientific_fingerprint_must_remain_default_separator_hash(self):
        (self.root/'audit').mkdir();value=report(self.req);value.pop('fingerprint')
        save(self.root/'audit'/subject.REPORT_FILE,subject.signed(value))
        with self.assertRaisesRegex(ValueError,'native data-verification'):
            subject.inspect_report(self.root/'audit',self.req)
    def test_success_receipt_rejects_bool_counters_and_report_nonfinite_or_duplicate_json(self):
        result=self.run_audit();altered=copy.deepcopy(result);altered['audit_process_starts']=True
        altered.pop('fingerprint');save(self.root/'audit'/subject.EXIT_FILE,subject.signed(altered))
        with self.assertRaises(ValueError):self.verify(result)
        for raw in ('{"fingerprint":"a","fingerprint":"b"}', '{"number":NaN}'):
            (self.root/'audit'/subject.REPORT_FILE).write_text(raw)
            with self.assertRaises(ValueError):subject.inspect_report(self.root/'audit',self.req)
    def test_original_process_helpers_remain_the_imported_implementations(self):
        import candidate_continuation_watchdog as original
        self.assertIs(subject.stop_owned,original.stop_owned)
        self.assertIs(subject.LinuxRecoveryOps.signal_owned,original.LinuxProcessOps.signal_owned)
        self.assertIs(subject.LinuxRecoveryOps.identity,original.LinuxProcessOps.identity)

if __name__=='__main__':unittest.main()
