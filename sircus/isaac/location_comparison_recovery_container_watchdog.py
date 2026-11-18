"""Independent host bound around one already-created CPU audit container.

No create, restart, remove, simulator or training route exists. Docker operations
use the full immutable container ID and recheck its unique name/image/session.
This outer clock/storage monitor does not invoke inner audit callbacks.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

SCHEMA = 'sircus-location-comparison-recovery-container-watchdog-v1'
LIMITS = {'audit':7200,'preflight':300}
MIN_FREE_BYTES = 6 * 2**30
MEMORY_BYTES = 8 * 2**30
NANO_CPUS = 4 * 10**9
CLEANUP_RESERVE_S = 30
CLI_TIMEOUT_S = 3
MAX_LOG_BYTES = 256 * 1024


def require(value,message):
    if not value:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def signed(value):return dict(value,fingerprint=digest(value))


def configuration_fingerprint(record):
    """Docker cgroup-v2 may render the enabled OOM killer as False or null.

    This is the sole normalized inspect value. Every other config/host/mount
    field, including list order, remains bound exactly as observed.
    """
    host=dict(record['HostConfig'])
    require('OomKillDisable' in host and
            (host['OomKillDisable'] is False or host['OomKillDisable'] is None),
            'Recovery container must retain the OOM killer')
    host['OomKillDisable']=False
    return digest(dict(Config=record['Config'],HostConfig=host,Mounts=record['Mounts']))


def canonical(value):
    path=Path(value)
    require(path.is_absolute() and path==path.resolve() and '..' not in path.parts
            and not any(p.is_symlink() for p in (path,*path.parents)), 'Expected canonical unlinked host path')
    return path


def write_new(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())


def valid_clock(value):return type(value) in (int,float) and math.isfinite(value) and value>=0


class DockerOps:
    def monotonic(self):return time.monotonic()
    def sleep(self,seconds):time.sleep(seconds)
    def free_bytes(self,path):return shutil.disk_usage(path).free
    def _run(self,argv,timeout=CLI_TIMEOUT_S,combined_output=False):
        result=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT if combined_output else subprocess.PIPE,
                              timeout=timeout,check=False)
        require(result.returncode==0,'Owned container command failed: '+
                (result.stdout if combined_output else result.stderr)[:1024].decode(errors='replace'))
        return result.stdout
    def inspect(self,identifier):
        raw=self._run(['docker','inspect',identifier])
        require(len(raw)<=256*1024,'Container inspection exceeds bound')
        value=json.loads(raw);require(type(value) is list and len(value)==1,'Ambiguous container inspection')
        return value[0]
    def start(self,identifier):self._run(['docker','start',identifier])
    def stop(self,identifier):self._run(['docker','stop','--time','2',identifier],timeout=5)
    def kill(self,identifier):self._run(['docker','kill',identifier])
    def logs(self,identifier):
        # The verified local driver has at most 1 MiB of retained logs. Only a
        # bounded suffix is copied; Docker's original retained log stays intact.
        raw=self._run(['docker','logs','--timestamps','--tail','200',identifier],combined_output=True)
        return raw[-MAX_LOG_BYTES:],len(raw)>MAX_LOG_BYTES


def _identity(record,identifier,name,image,session,label_key):
    require(type(record) is dict and record.get('Id')==identifier and record.get('Name')=='/'+name
            and record.get('Image')==image
            and record.get('Config',{}).get('Labels',{}).get(label_key)==session,
            'Owned recovery container identity changed')
    return record


def _normalized_binds(expected):
    require(type(expected) is list and expected,'Explicit expected bind mounts required')
    result=[]
    for item in expected:
        require(type(item) is dict and set(item)=={'source','destination','rw'} and type(item['rw']) is bool,
                'Invalid expected bind record')
        source=str(canonical(item['source']))
        destination=Path(item['destination'])
        require(destination.is_absolute() and '..' not in destination.parts,'Invalid container mount destination')
        result.append((source,str(destination),item['rw']))
    require(len(set(result))==len(result) and len({v[1] for v in result})==len(result),'Duplicate expected bind mounts')
    require(sum(v[2] for v in result)<=1,'Only a fresh audit destination may be a writable bind')
    require(all(not rw or destination=='/audit-root' for _,destination,rw in result),
            'Writable bind destination must be /audit-root')
    return sorted(result)


def validate_controls(record,expected_binds,expected_cap_add,expected_entrypoint,expected_command):
    host,config=record.get('HostConfig',{}),record.get('Config',{})
    require((config.get('Entrypoint') or [])==expected_entrypoint
            and (config.get('Cmd') or [])==expected_command,
            'Recovery container entrypoint/command differs from approved invocation')
    caps=lambda xs:sorted(str(v).removeprefix('CAP_') for v in (xs or []))
    require(expected_cap_add in ([],['DAC_OVERRIDE']),'Unreviewed added capability')
    require(host.get('Init') is True and host.get('Runtime')=='runc'
            and host.get('NetworkMode')=='none' and host.get('ReadonlyRootfs') is True
            and host.get('Privileged') is False and not host.get('PidMode')
            and not host.get('Devices') and not host.get('DeviceRequests')
            and host.get('AutoRemove') is False
            and host.get('RestartPolicy',{}).get('Name') in ('','no')
            and host.get('RestartPolicy',{}).get('MaximumRetryCount',0)==0,
            'Recovery container isolation or lifetime controls changed')
    require(type(host.get('Memory')) is int and host['Memory']==MEMORY_BYTES
            and host.get('MemorySwap')==MEMORY_BYTES and host.get('NanoCpus')==NANO_CPUS
            and type(host.get('PidsLimit')) is int and host['PidsLimit']==128
            and 'OomKillDisable' in host
            and (host['OomKillDisable'] is False or host['OomKillDisable'] is None)
            and caps(host.get('CapDrop'))==['ALL'] and caps(host.get('CapAdd'))==expected_cap_add
            and 'no-new-privileges' in host.get('SecurityOpt',[])
            and config.get('User')=='0:0','Recovery CPU/memory/privilege limits changed')
    env={}
    for value in config.get('Env',[]):
        if '=' in value:
            key,entry=value.split('=',1);require(key not in env,'Duplicate container environment key');env[key]=entry
    require(env.get('CUDA_VISIBLE_DEVICES')=='' and env.get('NVIDIA_VISIBLE_DEVICES')=='void',
            'Recovery container GPU environment changed')
    log=host.get('LogConfig',{})
    require(log.get('Type')=='local' and log.get('Config',{}).get('max-size')=='1m'
            and log.get('Config',{}).get('max-file')=='1'
            and log.get('Config',{}).get('compress')=='false',
            'Recovery container logs require bounded uncompressed local retention')
    binds=[]
    for item in record.get('Mounts',[]):
        require(item.get('Type') in ('bind','tmpfs'),'Unexpected writable volume in recovery container')
        if item['Type']=='bind':
            require(type(item.get('RW')) is bool,'Invalid mount write flag')
            binds.append((item['Source'],item['Destination'],item['RW']))
    require(sorted(binds)==expected_binds,'Recovery container mounts changed')
    return dict(cpu_only=True,network='none',readonly_root=True,init=True,
        cpu_cores=4,memory_bytes=MEMORY_BYTES,memory_swap_bytes=MEMORY_BYTES,
        pids_limit=128,
        cap_drop=['ALL'],cap_add=expected_cap_add,no_new_privileges=True,
        log_driver='local',log_max_size='1m',log_max_file=1,log_compress=False,
        entrypoint=expected_entrypoint,command=expected_command,
        bind_mounts=[dict(source=s,destination=d,rw=w) for s,d,w in expected_binds])


def _state(record):
    state=record.get('State',{})
    require(type(state.get('Running')) is bool and type(state.get('Pid')) is int
            and type(state.get('ExitCode')) is int and type(state.get('OOMKilled')) is bool,
            'Invalid owned container state')
    return {k:state.get(k) for k in ('Status','Running','Pid','ExitCode','OOMKilled','StartedAt','FinishedAt')}


def supervise_container(container_id,*,name,image,session_id,label_key,output,storage_root,
                        expected_binds,expected_entrypoint,expected_command,
                        expected_cap_add=None,mode='audit',ops=None,poll_s=1.,
                        execute_approved=False):
    """Start one verified CREATED container; independently bound its whole life."""
    require(execute_approved is True,'Explicit saved-data container approval required')
    require(type(container_id) is str and re.fullmatch('[a-f0-9]{64}',container_id)
            and type(image) is str and re.fullmatch('sha256:[a-f0-9]{64}',image)
            and type(session_id) is str and re.fullmatch('[a-f0-9]{32}',session_id)
            and type(name) is str and re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{1,100}',name)
            and type(label_key) is str and bool(label_key),'Explicit complete container identity required')
    require(mode in LIMITS and type(poll_s) in (int,float) and math.isfinite(poll_s) and 0<poll_s<=1,
            'Invalid independent container bounds')
    expected_cap_add=['DAC_OVERRIDE'] if expected_cap_add is None else expected_cap_add
    require(type(expected_entrypoint) is list and type(expected_command) is list
            and expected_entrypoint+expected_command
            and all(type(v) is str and '\0' not in v for v in expected_entrypoint+expected_command),
            'Explicit approved container invocation required')
    binds=_normalized_binds(expected_binds)
    output,storage_root=canonical(output),canonical(storage_root)
    require(output.parent.is_dir() and not output.exists() and storage_root.is_dir(),
            'Fresh host watchdog output and actual storage filesystem required')
    for source,_,_ in binds:
        source=Path(source)
        require(source.exists(),'Expected bind source is absent')
        require(output!=source and output not in source.parents and source not in output.parents,
                'Host watchdog output overlaps an evidence or audit bind')
    ops=ops or DockerOps()
    started=ops.monotonic();require(valid_clock(started),'Invalid outer watchdog clock')
    deadline=started+LIMITS[mode];execution_deadline=deadline-CLEANUP_RESERVE_S
    previous=started
    start_attempts=0;reason=error=None;safe=False;cleanup=[];last_state=None;controls=None
    config_fingerprint=None;log_info=None;owner_receipt_sha256=None
    output.mkdir(mode=0o700)
    def owned():
        return _identity(ops.inspect(container_id),container_id,name,image,session_id,label_key)
    def limits():
        nonlocal previous
        now=ops.monotonic()
        require(valid_clock(now) and previous<=now<execution_deadline,'Independent container deadline/clock reached')
        previous=now
        free=ops.free_bytes(storage_root)
        require(type(free) is int and free>=MIN_FREE_BYTES,'Independent container storage floor reached')
        return now
    def controls_unchanged(record):
        validate_controls(record,binds,expected_cap_add,expected_entrypoint,expected_command)
        require(configuration_fingerprint(record)
                ==config_fingerprint,'Owned recovery container configuration changed')
    try:
        limits();record=owned();controls=validate_controls(record,binds,expected_cap_add,expected_entrypoint,expected_command)
        last_state=_state(record)
        require(last_state['Status']=='created' and last_state['Running'] is False and last_state['Pid']==0,
                'Recovery container must be newly created; no retry/restart permitted')
        for source,_,rw in binds:
            if rw:require(Path(source).is_dir() and not any(Path(source).iterdir()),'Writable audit bind must be fresh and empty')
        config_fingerprint=configuration_fingerprint(record)
        limits()
        write_new(output/'recovery-container-owner.json',signed(dict(schema=SCHEMA,
            container_id=container_id,name=name,image=image,session_id=session_id,label_key=label_key,
            mode=mode,config_fingerprint=config_fingerprint,controls=controls,
            start_monotonic=started,deadline_monotonic=deadline,
            execution_deadline_monotonic=execution_deadline,storage_root=str(storage_root))))
        owner_receipt_sha256=hashlib.sha256((output/'recovery-container-owner.json').read_bytes()).hexdigest()
        start_attempts=1
        ops.start(container_id)
        while True:
            limits();record=owned();controls_unchanged(record);last_state=_state(record)
            limits()
            if last_state['Status']=='exited' and not last_state['Running'] and last_state['Pid']==0:
                safe=True
                require(last_state['ExitCode']==0 and last_state['OOMKilled'] is False,
                        'Recovery container exited unsuccessfully')
                break
            require(last_state['Status']=='running' and last_state['Running'] and last_state['Pid']>0,
                    'Unexpected recovery container lifecycle state')
            ops.sleep(poll_s)
    except BaseException as failure:
        error,reason=type(failure).__name__,(str(failure) or type(failure).__name__)[:2048]
        if start_attempts:
            try:
                record=owned();last_state=_state(record)
                if last_state['Running']:
                    try:ops.stop(container_id);cleanup.append('stop_sent')
                    except BaseException as exc:cleanup.append('stop_failed:'+type(exc).__name__)
                    record=owned();last_state=_state(record)
                if last_state['Running']:
                    try:ops.kill(container_id);cleanup.append('kill_sent')
                    except BaseException as exc:cleanup.append('kill_failed:'+type(exc).__name__)
                    record=owned();last_state=_state(record)
                safe=last_state['Status']=='exited' and last_state['Running'] is False and last_state['Pid']==0
            except BaseException as exc:
                cleanup.append('retirement_unconfirmed:'+type(exc).__name__);safe=False
    # Reconfirm retirement after the polling/cleanup path; a reused/restarted
    # container must not inherit a stale safe exit observation.
    if safe:
        try:
            record=owned();controls_unchanged(record)
            require(_state(record)==last_state,'Owned container retirement observation changed')
        except BaseException as exc:
            safe=False
            if reason is None:reason,error=str(exc)[:2048],type(exc).__name__
            cleanup.append('final_retirement_unconfirmed:'+type(exc).__name__)
    # Logs are copied only from the same retired container. This never removes
    # the original container/log. Loss or truncation is explicit, not a pass.
    if safe and valid_clock(ops.monotonic()) and ops.monotonic()<deadline-2*CLI_TIMEOUT_S:
        try:
            raw,truncated=ops.logs(container_id)
            require(type(raw) is bytes and len(raw)<=MAX_LOG_BYTES and type(truncated) is bool,'Invalid bounded log snapshot')
            with (output/'container.log').open('xb') as stream:stream.write(raw)
            log_info=dict(path='container.log',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),truncated=truncated)
        except BaseException as exc:log_info=dict(error=type(exc).__name__,original_docker_log_preserved=True)
    elif safe:log_info=dict(skipped='cleanup deadline reserve',original_docker_log_preserved=True)
    end=ops.monotonic()
    wall=end-started if valid_clock(end) and end>=started else None
    if reason is None and (wall is None or wall>=LIMITS[mode]):reason,error='Independent container total deadline exceeded','ValueError'
    result=signed(dict(schema=SCHEMA,
        status='container_exit_zero_observed' if reason is None else 'container_stopped_no_retry',
        container_id=container_id,name=name,image=image,session_id=session_id,label_key=label_key,
        mode=mode,config_fingerprint=config_fingerprint,controls=controls,
        owner_receipt_sha256=owner_receipt_sha256,
        container_start_attempts=start_attempts,container_creates=0,container_restarts=0,container_removals=0,
        state=last_state,reason=reason,error_type=error,cleanup_actions=cleanup,cleanup_safe=safe,
        start_monotonic=started,execution_deadline_monotonic=execution_deadline,
        deadline_monotonic=deadline,wall_s=wall,wall_limit_s=LIMITS[mode],cleanup_reserve_s=CLEANUP_RESERVE_S,
        min_free_storage_bytes=MIN_FREE_BYTES,storage_root=str(storage_root),logs=log_info,
        scientific_audit_passed=False,original_native_shutdown_status='failed',original_audit_passed=False,
        native_application_starts=0,optimizer_steps=0,new_shots=0))
    write_new(output/'recovery-container-exit.json',result)
    return result
