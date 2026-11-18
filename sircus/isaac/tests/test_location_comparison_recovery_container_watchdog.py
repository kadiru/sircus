"""Only fake Docker records and clocks: never invokes a real container."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import location_comparison_recovery_container_watchdog as subject

IDENTIFIER='a'*64
IMAGE='sha256:'+'b'*64
SESSION='c'*32
NAME='saved-audit-new'
LABEL='sircus.recovery.session'
ENTRY=['/bin/sh']
COMMAND=['/recovery/entry.sh','--mode','audit','--request-sha256','d'*64]


def record(binds):
    return dict(Id=IDENTIFIER,Name='/'+NAME,Image=IMAGE,
        Config=dict(Labels={LABEL:SESSION},User='0:0',Entrypoint=ENTRY,Cmd=COMMAND,
            Env=['CUDA_VISIBLE_DEVICES=','NVIDIA_VISIBLE_DEVICES=void']),
        HostConfig=dict(Init=True,Runtime='runc',NetworkMode='none',ReadonlyRootfs=True,
            Privileged=False,PidMode='',Devices=[],DeviceRequests=[],AutoRemove=False,
            RestartPolicy=dict(Name='no',MaximumRetryCount=0),Memory=8*2**30,
            MemorySwap=8*2**30,NanoCpus=4*10**9,PidsLimit=128,CapDrop=['ALL'],CapAdd=['DAC_OVERRIDE'],
            SecurityOpt=['no-new-privileges'],LogConfig=dict(Type='local',Config={'max-size':'1m','max-file':'1','compress':'false'})),
        Mounts=[dict(Type='bind',Source=b['source'],Destination=b['destination'],RW=b['rw']) for b in binds],
        State=dict(Status='created',Running=False,Pid=0,ExitCode=0,OOMKilled=False,
            StartedAt='0001-01-01T00:00:00Z',FinishedAt='0001-01-01T00:00:00Z'))


class FakeOps:
    def __init__(self,initial):
        self.record=copy.deepcopy(initial);self.t=100.;self.free=subject.MIN_FREE_BYTES+1024
        self.started=[];self.stopped=[];self.killed=[];self.inspected=[];self.log_calls=[]
        self.storage_paths=[];self.sleeps=0;self.finish_after=2;self.after_sleep=None
        self.after_start=None;self.stop_works=True;self.kill_works=True;self.exit_code=0
        self.log=b'CPU-only audit output\n';self.log_truncated=False
        self.command_delay=0.;self.inspect_error=None
    def monotonic(self):return self.t
    def free_bytes(self,path):self.storage_paths.append(path);return self.free
    def inspect(self,identifier):
        self.inspected.append(identifier);self.t+=self.command_delay
        if self.inspect_error:raise self.inspect_error
        return copy.deepcopy(self.record)
    def start(self,identifier):
        self.started.append(identifier);self.t+=self.command_delay
        self.record['State'].update(Status='running',Running=True,Pid=123,StartedAt='2026-11-18T00:00:00Z')
        if self.after_start:self.after_start(self)
    def retire(self):
        self.record['State'].update(Status='exited',Running=False,Pid=0,ExitCode=self.exit_code,
            FinishedAt='2026-11-18T00:00:02Z')
    def stop(self,identifier):
        self.stopped.append(identifier);self.t+=self.command_delay
        if self.stop_works:self.retire()
        else:raise subprocess.TimeoutExpired('stop',5)
    def kill(self,identifier):
        self.killed.append(identifier);self.t+=self.command_delay
        if self.kill_works:self.retire()
        else:raise subprocess.TimeoutExpired('kill',3)
    def logs(self,identifier):
        self.log_calls.append(identifier);self.t+=self.command_delay
        return self.log,self.log_truncated
    def sleep(self,seconds):
        self.t+=seconds;self.sleeps+=1
        if self.sleeps>=self.finish_after:self.retire()
        if self.after_sleep:self.after_sleep(self)


class ContainerWatchdogTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        for name in ('readonly','writable','studies'):(self.root/name).mkdir()
        self.binds=[dict(source=str(self.root/'readonly'),destination='/evidence',rw=False),
                    dict(source=str(self.root/'writable'),destination='/audit-root',rw=True)]
        self.ops=FakeOps(record(self.binds))
    def run_container(self,**kwargs):
        arguments=dict(name=NAME,image=IMAGE,session_id=SESSION,label_key=LABEL,
            output=self.root/'host-proof',storage_root=self.root/'studies',expected_binds=self.binds,
            expected_entrypoint=ENTRY,expected_command=COMMAND,ops=self.ops,execute_approved=True)
        arguments.update(kwargs)
        return subject.supervise_container(IDENTIFIER,**arguments)
    def test_success_proves_owned_zero_exit_but_never_scientific_or_native_success(self):
        result=self.run_container()
        self.assertEqual(result['status'],'container_exit_zero_observed')
        self.assertTrue(result['cleanup_safe']);self.assertEqual(result['container_start_attempts'],1)
        self.assertEqual(result['state']['Pid'],0);self.assertEqual(result['state']['ExitCode'],0)
        self.assertFalse(result['scientific_audit_passed']);self.assertFalse(result['original_audit_passed'])
        self.assertEqual(result['original_native_shutdown_status'],'failed')
        self.assertEqual(result['native_application_starts'],0);self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual(result['new_shots'],0)
        self.assertEqual(self.ops.started,[IDENTIFIER]);self.assertFalse(self.ops.stopped+self.ops.killed)
        self.assertTrue(all(p==self.root/'studies' for p in self.ops.storage_paths))
        self.assertTrue(all(v==IDENTIFIER for v in self.ops.inspected))
        proof=self.root/'host-proof'
        owner=json.loads((proof/'recovery-container-owner.json').read_text())
        self.assertEqual(result['owner_receipt_sha256'],hashlib.sha256((proof/'recovery-container-owner.json').read_bytes()).hexdigest())
        self.assertEqual(owner['config_fingerprint'],result['config_fingerprint'])
        self.assertEqual((proof/'container.log').read_bytes(),self.ops.log)
        body=dict(result);fingerprint=body.pop('fingerprint');self.assertEqual(subject.digest(body),fingerprint)
        self.assertEqual(list((self.root/'readonly').iterdir()),[])
    def test_no_approval_or_reused_output_fails_before_start(self):
        with self.assertRaisesRegex(ValueError,'approval'):self.run_container(execute_approved=False)
        (self.root/'host-proof').mkdir()
        with self.assertRaisesRegex(ValueError,'Fresh'):self.run_container()
        self.assertFalse(self.ops.started)
    def test_ambiguous_identity_and_linked_paths_are_rejected_without_start(self):
        for kwargs in ({'session_id':'bad'},{'image':'tag:latest'},{'name':'invalid/name'},
                       {'expected_entrypoint':[],'expected_command':[]}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.run_container(**kwargs)
        (self.root/'link').symlink_to(self.root/'readonly',target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'canonical'):self.run_container(output=self.root/'link'/'output')
        self.assertFalse(self.ops.started)
    def test_previously_started_states_never_restart_or_signal(self):
        for state,running,pid in [('running',True,123),('exited',False,0),('restarting',True,123),('dead',False,0)]:
            self.ops=FakeOps(record(self.binds));self.ops.record['State'].update(Status=state,Running=running,Pid=pid)
            result=self.run_container(output=self.root/state)
            self.assertEqual(result['container_start_attempts'],0)
            self.assertFalse(self.ops.started+self.ops.stopped+self.ops.killed)
    def test_preflight_uses_its_own_approved_command_and_300_second_limit(self):
        command=['/recovery/entry.sh','--mode','preflight']
        self.ops.record['Config']['Cmd']=command
        result=self.run_container(mode='preflight',expected_command=command)
        self.assertEqual(result['wall_limit_s'],300)
        self.assertEqual(result['controls']['command'],command)
    def test_different_entrypoint_command_or_gpu_environment_blocks_start(self):
        changes=[('Entrypoint',['/bin/python']),('Cmd',['/other-script']),
                 ('Env',['CUDA_VISIBLE_DEVICES=0','NVIDIA_VISIBLE_DEVICES=all']),
                 ('Env',['CUDA_VISIBLE_DEVICES=','CUDA_VISIBLE_DEVICES=0','NVIDIA_VISIBLE_DEVICES=void'])]
        for i,(key,value) in enumerate(changes):
            self.ops=FakeOps(record(self.binds));self.ops.record['Config'][key]=value
            result=self.run_container(output=self.root/f'bad-command-{i}')
            self.assertEqual(result['container_start_attempts'],0);self.assertFalse(self.ops.started)
    def test_cpu_privilege_network_log_or_lifetime_control_changes_block_start(self):
        changes=[('Init',False),('Runtime','nvidia'),('NetworkMode','host'),('ReadonlyRootfs',False),
            ('Privileged',True),('PidMode','host'),('Devices',[{}]),('DeviceRequests',[{}]),
            ('AutoRemove',True),('Memory',9*2**30),('MemorySwap',-1),('NanoCpus',0),
            ('PidsLimit',0),('PidsLimit',256),
            ('CapDrop',[]),('CapAdd',['SYS_ADMIN']),('SecurityOpt',[]),
            ('LogConfig',{'Type':'json-file'}),('RestartPolicy',{'Name':'always'})]
        for i,(key,value) in enumerate(changes):
            self.ops=FakeOps(record(self.binds));self.ops.record['HostConfig'][key]=value
            result=self.run_container(output=self.root/f'bad-control-{i}')
            self.assertEqual(result['container_start_attempts'],0);self.assertFalse(self.ops.started)
    def test_explicit_no_capability_profile_allowed_but_unknown_capability_rejected(self):
        self.ops.record['HostConfig']['CapAdd']=None
        result=self.run_container(expected_cap_add=[])
        self.assertEqual(result['status'],'container_exit_zero_observed')
        self.ops=FakeOps(record(self.binds))
        result=self.run_container(expected_cap_add=['SYS_ADMIN'],output=self.root/'bad-cap')
        self.assertEqual(result['container_start_attempts'],0)
    def test_single_file_local_logs_require_explicit_no_compression(self):
        for i,value in enumerate((None,'true',True,False)):
            self.ops=FakeOps(record(self.binds))
            config=self.ops.record['HostConfig']['LogConfig']['Config']
            if value is None:config.pop('compress')
            else:config['compress']=value
            result=self.run_container(output=self.root/f'compression-{i}')
            self.assertEqual(result['container_start_attempts'],0)
            self.assertIn('uncompressed',result['reason'])
            self.assertFalse(self.ops.started)
    def test_wrong_container_identity_initially_never_signalled(self):
        for i,key in enumerate(('Id','Name','Image')):
            self.ops=FakeOps(record(self.binds));self.ops.record[key]='foreign'
            result=self.run_container(output=self.root/f'wrong-{i}')
            self.assertEqual(result['container_start_attempts'],0)
            self.assertFalse(self.ops.started+self.ops.stopped+self.ops.killed)
    def test_empty_writable_bind_and_exact_readonly_mounts_required(self):
        (self.root/'writable'/'existing').write_text('preserve')
        result=self.run_container();self.assertEqual(result['container_start_attempts'],0)
        self.assertEqual((self.root/'writable'/'existing').read_text(),'preserve')
        self.ops=FakeOps(record(self.binds));self.ops.record['Mounts'][0]['RW']=True
        result=self.run_container(output=self.root/'bad-mount')
        self.assertEqual(result['container_start_attempts'],0)
    def test_more_than_one_writable_bind_or_unexpected_volume_rejected(self):
        binds=copy.deepcopy(self.binds);binds[0]['rw']=True
        with self.assertRaisesRegex(ValueError,'Only a fresh'):self.run_container(expected_binds=binds)
        self.ops.record['Mounts'].append(dict(Type='volume',Source='/other',Destination='/other',RW=True))
        result=self.run_container();self.assertEqual(result['container_start_attempts'],0)
    def test_writable_mount_must_target_exact_audit_root(self):
        binds=copy.deepcopy(self.binds);binds[1]['destination']='/evidence-write'
        with self.assertRaisesRegex(ValueError,'/audit-root'):self.run_container(expected_binds=binds)
        self.assertFalse(self.ops.started)
    def test_host_receipt_cannot_write_inside_evidence_or_audit_bind(self):
        for output in (self.root/'readonly'/'host-proof',self.root/'writable'/'host-proof'):
            with self.subTest(output=output),self.assertRaisesRegex(ValueError,'overlaps'):
                self.run_container(output=output)
            self.assertFalse(output.exists())
        self.assertFalse(self.ops.started)
    def test_initial_storage_floor_blocks_start_and_records_actual_storage_root(self):
        self.ops.free=subject.MIN_FREE_BYTES-1
        result=self.run_container();self.assertIn('storage floor',result['reason'])
        self.assertEqual(result['container_start_attempts'],0)
        self.assertEqual(result['storage_root'],str(self.root/'studies'))
    def test_independent_storage_floor_stops_inner_hash_that_never_returns(self):
        self.ops.finish_after=10**6
        self.ops.after_sleep=lambda ops:setattr(ops,'free',subject.MIN_FREE_BYTES-1)
        result=self.run_container();self.assertIn('storage floor',result['reason'])
        self.assertEqual(self.ops.stopped,[IDENTIFIER]);self.assertTrue(result['cleanup_safe'])
        self.assertEqual(len(self.ops.started),1)
    def test_independent_deadline_stops_blocked_inner_callback_without_progress_dependency(self):
        self.ops.finish_after=10**6
        self.ops.after_sleep=lambda ops:setattr(ops,'t',100+7200-subject.CLEANUP_RESERVE_S)
        result=self.run_container();self.assertIn('deadline',result['reason'])
        self.assertEqual(self.ops.stopped,[IDENTIFIER]);self.assertTrue(result['cleanup_safe'])
        self.assertLess(result['wall_s'],7200)
    def test_preflight_deadline_has_the_same_independent_stop(self):
        self.ops.finish_after=10**6
        self.ops.after_sleep=lambda ops:setattr(ops,'t',100+300-subject.CLEANUP_RESERVE_S)
        result=self.run_container(mode='preflight');self.assertIn('deadline',result['reason'])
        self.assertLess(result['wall_s'],300);self.assertEqual(self.ops.stopped,[IDENTIFIER])
    def test_clock_reversal_and_nan_stop_owned_container(self):
        for i,value in enumerate((99.,float('nan'))):
            self.ops=FakeOps(record(self.binds));self.ops.finish_after=10**6
            self.ops.after_sleep=lambda ops,v=value:setattr(ops,'t',v)
            result=self.run_container(output=self.root/f'bad-clock-{i}')
            self.assertIn('clock',result['reason']);self.assertEqual(self.ops.stopped,[IDENTIFIER])
    def test_stop_timeout_escalates_only_same_id_to_kill(self):
        self.ops.finish_after=10**6;self.ops.stop_works=False
        self.ops.after_sleep=lambda ops:setattr(ops,'free',0)
        result=self.run_container();self.assertTrue(result['cleanup_safe'])
        self.assertEqual(self.ops.stopped,[IDENTIFIER]);self.assertEqual(self.ops.killed,[IDENTIFIER])
    def test_unconfirmed_retirement_is_not_success_and_is_not_retried(self):
        self.ops.finish_after=10**6;self.ops.stop_works=False;self.ops.kill_works=False
        self.ops.after_sleep=lambda ops:setattr(ops,'free',0)
        result=self.run_container();self.assertFalse(result['cleanup_safe'])
        self.assertEqual(result['status'],'container_stopped_no_retry');self.assertEqual(len(self.ops.started),1)
        self.assertFalse(self.ops.log_calls)
    def test_changed_identity_during_run_is_never_signalled(self):
        self.ops.finish_after=10**6
        self.ops.after_sleep=lambda ops:ops.record['Config']['Labels'].update({LABEL:'e'*32})
        result=self.run_container();self.assertFalse(result['cleanup_safe'])
        self.assertFalse(self.ops.stopped+self.ops.killed)
    def test_restarted_container_cannot_reuse_a_retired_success_observation(self):
        inspected=self.ops.inspect
        retired_reads=0
        def inspect(identifier):
            nonlocal retired_reads
            if self.ops.record['State']['Status']=='exited':
                retired_reads+=1
                if retired_reads==2:
                    self.ops.record['State'].update(Status='running',Running=True,Pid=999,
                        StartedAt='2026-11-18T01:00:00Z')
            return inspected(identifier)
        self.ops.inspect=inspect
        result=self.run_container()
        self.assertFalse(result['cleanup_safe']);self.assertEqual(result['status'],'container_stopped_no_retry')
        self.assertFalse(self.ops.log_calls)
    def test_changed_controls_during_run_stop_owned_container(self):
        self.ops.finish_after=10**6
        self.ops.after_sleep=lambda ops:ops.record['HostConfig'].update(NanoCpus=8*10**9)
        result=self.run_container();self.assertEqual(self.ops.stopped,[IDENTIFIER])
        self.assertEqual(result['status'],'container_stopped_no_retry')
    def test_nonzero_exit_or_oom_cannot_be_success(self):
        for i,(code,oom) in enumerate(((1,False),(0,True))):
            self.ops=FakeOps(record(self.binds));self.ops.exit_code=code;self.ops.record['State']['OOMKilled']=oom
            result=self.run_container(output=self.root/f'nonzero-{i}')
            self.assertTrue(result['cleanup_safe']);self.assertEqual(result['status'],'container_stopped_no_retry')
    def test_start_timeout_never_retries_and_retires_if_daemon_started_it(self):
        def fail(ops):raise subprocess.TimeoutExpired('start',3)
        self.ops.after_start=fail
        result=self.run_container();self.assertEqual(len(self.ops.started),1)
        self.assertEqual(self.ops.stopped,[IDENTIFIER]);self.assertTrue(result['cleanup_safe'])
    def test_worst_inflight_inspect_cleanup_fits_reserve_and_optional_logs_skip_near_end(self):
        self.ops.finish_after=10**6;self.ops.stop_works=False;self.ops.command_delay=3
        self.ops.after_sleep=lambda ops:setattr(ops,'t',100+7200-subject.CLEANUP_RESERVE_S-1)
        result=self.run_container();self.assertLess(result['wall_s'],7200)
        self.assertTrue(result['cleanup_safe']);self.assertEqual(self.ops.killed,[IDENTIFIER])
    def test_log_truncation_is_explicit_and_does_not_delete_original(self):
        self.ops.log=b'x'*subject.MAX_LOG_BYTES;self.ops.log_truncated=True
        result=self.run_container();self.assertTrue(result['logs']['truncated'])
        self.assertEqual(result['container_removals'],0)
    def test_docker_commands_are_bounded_exact_id_and_unattached(self):
        ops=subject.DockerOps()
        with patch.object(subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout=b'output',stderr=b'')) as call:
            ops.start(IDENTIFIER);self.assertEqual(call.call_args.args[0],['docker','start',IDENTIFIER])
            self.assertEqual(call.call_args.kwargs['timeout'],3)
            ops.stop(IDENTIFIER);self.assertEqual(call.call_args.args[0],['docker','stop','--time','2',IDENTIFIER])
            self.assertEqual(call.call_args.kwargs['timeout'],5)
            ops.kill(IDENTIFIER);self.assertEqual(call.call_args.args[0],['docker','kill',IDENTIFIER])
            self.assertEqual(call.call_args.kwargs['timeout'],3)
    def test_docker_log_snapshot_combines_stderr_and_is_size_bounded(self):
        ops=subject.DockerOps()
        with patch.object(subprocess,'run',return_value=types.SimpleNamespace(returncode=0,
                stdout=b'x'*(subject.MAX_LOG_BYTES+20)+b'traceback',stderr=None)) as call:
            raw,truncated=ops.logs(IDENTIFIER)
            self.assertEqual(call.call_args.kwargs['stderr'],subprocess.STDOUT)
            self.assertTrue(truncated);self.assertEqual(len(raw),subject.MAX_LOG_BYTES)
            self.assertTrue(raw.endswith(b'traceback'))
    def test_docker_inspect_requires_exact_single_bounded_record(self):
        ops=subject.DockerOps()
        with patch.object(ops,'_run',return_value=b'[]'),self.assertRaisesRegex(ValueError,'Ambiguous'):
            ops.inspect(IDENTIFIER)
        with patch.object(ops,'_run',return_value=b'x'*(256*1024+1)),self.assertRaisesRegex(ValueError,'bound'):
            ops.inspect(IDENTIFIER)
    def test_import_never_executes_docker_or_process(self):
        with patch.object(subprocess,'run',side_effect=AssertionError('process')):
            import importlib
            importlib.reload(subject)


if __name__=='__main__':unittest.main()
