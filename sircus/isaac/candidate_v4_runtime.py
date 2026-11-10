"""Production host adapter for the bounded continuation supervisor.

Construction and imports are inert. Production operations require an explicit
future execution authorization and ``supervise(..., execute=True)``. Tests inject
fake host operations; no Docker, GPU, child, lock or signal operation is required.
"""

from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import uuid

from candidate_continuation_journal import write_new_json
from candidate_continuation_ledger import canonical_hash
from candidate_continuation_launcher import ChildIdentity, GPU_UUID, CONTAINER
from candidate_v4_evaluation import V4_CHECKPOINT_SHA256,V4_PRODUCER_SHA256,PARENT_CHECKPOINT_SHA256
NATIVE_SECONDS = 3000
V4_MANIFEST_SHA256 = "16c7c5edca352b192794505d5433643211a7bf16aca0cc631ff6409183340c47"
from candidate_v4_watchdog import (SCHEMA, COOKIE, RECEIPT_NAMES, sha,
    signed, strict_json, same_process, verify_metadata_snapshot)
from shot_benchmark_supervisor import verify_archive

SPEC_SCHEMA = "sircus-v4-evaluation-runtime-spec-v1"
MANIFEST_SCHEMA = "sircus-v4-evaluation-production-inputs-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(path):
    path = Path(path)
    require(path.is_absolute() and str(path.resolve()) == str(path), "Runtime paths must be canonical absolute paths")
    return path


def _plain_file(root, path):
    path = _canonical(path)
    require(path.is_relative_to(root) and path.is_file(), "Runtime input must be a file in the existing mount")
    require(not any(p.is_symlink() for p in (path,*path.parents) if p.is_relative_to(root)),
            "Linked runtime input paths are not admitted")
    return path


@dataclass(frozen=True)
class EvaluationLaunchPlan:
    output: str
    protected_roots: tuple[str, ...]
    scene_lock: str
    session: str
    prepared_plan_fingerprint: str
    source_sha256: str

    def __post_init__(self):
        output,lock = _canonical(self.output),_canonical(self.scene_lock)
        require(type(self.protected_roots) is tuple and self.protected_roots,"Protected roots required")
        require(re.fullmatch(r"[0-9a-f]{32}",self.session) is not None,"Fresh exact session required")
        require(all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}",v) for v in
            (self.prepared_plan_fingerprint,self.source_sha256)),"Exact evaluation plan/source fingerprints required")
        for raw in self.protected_roots:
            root = _canonical(raw)
            require(not output.is_relative_to(root) and not root.is_relative_to(output),
                "Evaluation output overlaps protected source/evidence")
        require(not lock.is_relative_to(output),"Shared lock cannot live inside evaluation output")

    @classmethod
    def create(cls,*,output,protected_roots,scene_lock,session,prepared_plan_fingerprint,source_sha256):
        return cls(str(_canonical(output)),tuple(str(_canonical(p)) for p in protected_roots),
            str(_canonical(scene_lock)),session,prepared_plan_fingerprint,source_sha256)


@dataclass(frozen=True)
class ProductionSpec:
    workspace_host: str
    source_host: str
    archive_path: str
    source_commit: str
    manifest_path: str
    manifest_sha256: str

    def __post_init__(self):
        root,source = _canonical(self.workspace_host),_canonical(self.source_host)
        require(root.is_dir() and source.is_dir() and source.is_relative_to(root),
                "Existing workspace and versioned source snapshot required")
        for path in (self.archive_path,self.manifest_path): _plain_file(root,path)
        require(re.fullmatch(r"[0-9a-f]{40}",self.source_commit)
            and re.fullmatch(r"[0-9a-f]{64}",self.manifest_sha256),"Exact source commit/manifest SHA required")

    @classmethod
    def create(cls,*,workspace_host,source_host,archive_path,source_commit,manifest_path,manifest_sha256):
        return cls(str(_canonical(workspace_host)),str(_canonical(source_host)),str(_canonical(archive_path)),
            source_commit,str(_canonical(manifest_path)),manifest_sha256)

    def container_path(self,path):
        path = _canonical(path)
        require(path.is_relative_to(Path(self.workspace_host)),"Mapped path leaves existing workspace mount")
        return str(PurePosixPath("/workspace")/path.relative_to(self.workspace_host).as_posix())

    def manifest(self):
        raw = Path(self.manifest_path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self.manifest_sha256,"External production manifest bytes changed")
        value = strict_json(raw)
        require(value.get("schema") == MANIFEST_SCHEMA and value.get("workspace_host") == self.workspace_host
            and value.get("source_host") == self.source_host and type(value.get("files")) is dict
            and value.get("synthetic") is False and type(value.get("expected_runtime")) is dict
            and bool(value["expected_runtime"])
            and type(value.get("protected_roots")) is list and value["protected_roots"]
            and value.get("execution_authorized") is value.get("release_authorized") is False,
            "Evaluation manifest schema/workspace/source/authorization changed")
        mandatory={"prepared_plan","parent_checkpoint","final_checkpoint","original_manifest",
            "original_learning_result","original_native_binding","old_assessment_plan",
            "old_assessment_inventory","old_component_summary"}
        require(mandatory == set(value["files"]),"Immutable V4 evaluation input role set changed")
        root=Path(self.workspace_host)
        for record in value["files"].values():
            require(type(record) is dict and set(record)=={"path","sha256"}
                and re.fullmatch(r"[0-9a-f]{64}",record["sha256"]),"Malformed input byte binding")
            _plain_file(root,record["path"])
        require(value["files"]["original_manifest"]["sha256"] == V4_MANIFEST_SHA256,
            "Original V4 manifest identity changed")
        require(value["files"]["parent_checkpoint"]["sha256"] == PARENT_CHECKPOINT_SHA256
            and value["files"]["final_checkpoint"]["sha256"] == V4_CHECKPOINT_SHA256,
            "Frozen parent or saved V4 model identity changed")
        original=_canonical(value["original_source_host"])
        require(original.is_dir() and original.is_relative_to(root) and original != Path(self.source_host)
            and value.get("original_source_sha256") == V4_PRODUCER_SHA256,
            "Original producer source must remain separately immutable")
        for path in value["protected_roots"]:
            require(_canonical(path).is_relative_to(root),"Protected runtime root leaves existing mount")
        return value

    def verify(self,plan):
        require(type(plan) is EvaluationLaunchPlan and plan.scene_lock == str(Path(self.workspace_host)/"scene.lock"),
            "Evaluation must use the original shared scene lock")
        source,root=Path(self.source_host),Path(self.workspace_host)
        manifest=self.manifest()
        original=Path(manifest["original_source_host"])
        protected=[source,original,*(Path(p) for p in manifest["protected_roots"])]
        output=Path(plan.output)
        require(output.is_relative_to(root) and all(not output.is_relative_to(p) and not p.is_relative_to(output)
            for p in protected),"Production output overlaps frozen inputs or source")
        require(set(map(str,protected)).issubset(plan.protected_roots),"Launch omitted required protected root")
        require(all(not Path(r["path"]).is_relative_to(output) for r in manifest["files"].values()),
            "Output overlaps a byte-bound input")
        old_manifest=strict_json(Path(manifest["files"]["original_manifest"]["path"]).read_bytes())
        require(old_manifest.get("workspace_host")==self.workspace_host
            and old_manifest.get("source_host")==str(original)
            and old_manifest.get("source_sha256")==manifest["original_source_sha256"]
            and type(old_manifest.get("protected_roots")) is list and old_manifest["protected_roots"]
            and type(old_manifest.get("files")) is dict and old_manifest["files"],
            "Original manifest workspace/source/protection scope changed")
        old_roots=[_canonical(old_manifest[key]) for key in
            ("source_host","source_run","assets_dir","frozen_experiment_root")]
        old_roots.extend(_canonical(p) for p in old_manifest["protected_roots"])
        for old_root in old_roots:
            require(old_root.is_relative_to(root) and any(old_root.is_relative_to(p) for p in protected),
                "New protection omitted an original source, bank, asset or evidence root")
            require(not output.is_relative_to(old_root) and not old_root.is_relative_to(output),
                "New output overlaps an original evidence root")
        for record in old_manifest["files"].values():
            require(type(record) is dict and set(record)=={"path","sha256"}
                and type(record["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}",record["sha256"]),
                "Malformed original input binding")
            old_path=_plain_file(root,record["path"])
            require(any(old_path.is_relative_to(p) for p in protected)
                and sha(old_path)==record["sha256"],"Original input bytes/protection coverage changed")
        archive=verify_archive(Path(self.archive_path),source,self.source_commit)
        py={p.name:sha(p) for p in sorted((source/"sircus/isaac").glob("*.py"))}
        require(py and canonical_hash(py)==manifest.get("source_sha256")==plan.source_sha256,
            "New evaluation all-Python source identity changed")
        archived={PurePosixPath(k).name:v for k,v in archive["source_members_sha256"].items()
            if PurePosixPath(k).parent==PurePosixPath("sircus/isaac") and k.endswith(".py")}
        require(py==archived,"Unarchived or missing evaluation Python source")
        for name in ("candidate_v4_entry.py","candidate_v4_watchdog.py","python.sh"):
            _plain_file(root,source/"sircus/isaac"/name)
        original_py={p.name:sha(p) for p in sorted((original/"sircus/isaac").glob("*.py"))}
        require(canonical_hash(original_py)==manifest["original_source_sha256"],"Original model producer source changed")
        for record in manifest["files"].values():
            require(sha(record["path"])==record["sha256"],"Evaluation input bytes changed")
        prepared=strict_json(Path(manifest["files"]["prepared_plan"]["path"]).read_bytes())
        require(prepared.get("fingerprint")==canonical_hash({k:v for k,v in prepared.items() if k!="fingerprint"})
            and prepared.get("fingerprint")==manifest.get("prepared_plan_fingerprint")==plan.prepared_plan_fingerprint
            and prepared.get("requests")==210 and prepared.get("synthetic") is False
            and prepared.get("release_authorized") is False and prepared.get("final_checkpoint_sha256")==V4_CHECKPOINT_SHA256
            and prepared.get("parent_checkpoint_sha256")==PARENT_CHECKPOINT_SHA256
            and prepared.get("native_execution_ready") is prepared.get("execution_authorized") is False
            and prepared.get("bindings",{}).get("evaluation_source_sha256")==plan.source_sha256,
            "Prepared TEST-only plan/model/source identity changed")
        return dict(manifest_sha256=self.manifest_sha256,prepared_plan_fingerprint=plan.prepared_plan_fingerprint,
            source_sha256=plan.source_sha256,original_source_sha256=manifest["original_source_sha256"],
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,final_checkpoint_sha256=V4_CHECKPOINT_SHA256),archive


class LinuxHostOps:
    """Lazy host operations, each subprocess query has a finite timeout."""

    def __init__(self): self.locks = {}
    def monotonic(self): return time.monotonic()
    def sleep(self,seconds): time.sleep(seconds)
    def free_bytes(self,path): return shutil.disk_usage(Path(path) if Path(path).exists() else Path(path).parent).free
    def run(self,argv):
        require(sys.platform == "linux","Production host operations require the Linux simulation host")
        return subprocess.run(argv,check=True,capture_output=True,text=True,timeout=10).stdout

    def acquire_lock(self,path,session):
        require(sys.platform == "linux","Production host operations require the Linux simulation host")
        if session in self.locks: return True
        stream = Path(path).open("rb")
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close(); return False
        self.locks[session] = stream
        return True

    def release_lock(self,path,session):
        stream = self.locks.pop(session)
        fcntl.flock(stream.fileno(),fcntl.LOCK_UN); stream.close()

    def start(self,argv,log):
        require(sys.platform == "linux","Production host operations require the Linux simulation host")
        stream = Path(log).open("xb")
        try:
            child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True)
        except BaseException:
            stream.close(); raise
        self.log = stream
        return child


class ProductionRuntime:
    """Future native wiring; no operation is authorized by construction/tests."""

    mock_only = False
    production_runtime = True

    def __init__(self,spec,*,execution_authorized=False,authorization=None,ops=None):
        require(type(spec) is ProductionSpec and type(execution_authorized) is bool,"Invalid production runtime setup")
        self.spec,self.execution_authorized = spec,execution_authorized
        self.authorization = authorization
        self.authorization_sha256 = None
        self.ops = ops or LinuxHostOps()
        self.child,self.owner,self.plan = None,None,None
        self.attempted_start = False
        self.progress_cursor = 0
        self.last_snapshot = None
        self.payload_sha256,self.native_owner = None,None
        self.payload_raw_sha256,self.payload_path = None,None
        self.progress_fingerprints = []
        self.lock_lease = None

    def bind_plan(self,plan):
        require(type(plan) is EvaluationLaunchPlan,"Expected fixed TEST-only launch plan")
        require(self.plan is None or self.plan==plan,"A runtime cannot rebind a different session or output")
        require(self.execution_authorized,"Explicit production execution authorization is missing")
        expected=dict(schema="sircus-v4-evaluation-operator-authorization-v1",action="evaluate_saved_v4_test_only",
            session=plan.session,output=plan.output,manifest_sha256=self.spec.manifest_sha256,
            prepared_plan_fingerprint=plan.prepared_plan_fingerprint,source_sha256=plan.source_sha256,
            execution_authorized=True,release_authorized=False)
        require(type(self.authorization) is dict and self.authorization==signed(expected),
            "Explicit operator authorization is not bound to exact session, output, manifest, plan and source")
        self.plan=plan;self.authorization_sha256=self.authorization["fingerprint"]
        return plan

    def _authorized(self):
        require(self.execution_authorized and self.plan is not None and self.authorization_sha256 is not None,
            "Explicit bound evaluation execution authorization is missing")
        self.bind_plan(self.plan)
    def monotonic(self): return self.ops.monotonic()
    def sleep(self,seconds): return self.ops.sleep(seconds)
    def free_bytes(self,path): return self.ops.free_bytes(path)
    def source_snapshot(self,plan): return self.spec.verify(plan)[0]
    def acquire_scene_lock(self,path,session):
        self._authorized()
        require(path == str(Path(self.spec.workspace_host)/"scene.lock")
            and re.fullmatch(r"[0-9a-f]{32}",session),"Only the declared shared scene lock/session may be acquired")
        acquired = self.ops.acquire_lock(path,session)
        if acquired:self.lock_lease = (path,session)
        return acquired
    def release_scene_lock(self,path,session):
        self._authorized();require(self.lock_lease == (path,session),"Cannot release an unowned scene lock lease")
        self.ops.release_lock(path,session);self.lock_lease = None

    def gpu_state(self,gpu_uuid):
        self._authorized(); require(gpu_uuid == GPU_UUID,"Only the declared 4090 may be queried")
        values = strict_json(self.ops.run(["docker","inspect",CONTAINER]))
        require(type(values) is list and len(values) == 1,"Existing container identity is ambiguous")
        value = values[0]
        devices = [d for request in value["HostConfig"].get("DeviceRequests",[]) for d in request.get("DeviceIDs",[])]
        pids = []
        output = self.ops.run(["nvidia-smi","--id="+GPU_UUID,"--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"])
        for line in output.splitlines():
            pieces = [x.strip() for x in line.split(",")]
            require(len(pieces) == 2 and pieces[0] == GPU_UUID and pieces[1].isdecimal(),"Unexpected GPU ownership response")
            pids.append(int(pieces[1]))
        return dict(gpu_uuid=GPU_UUID,container=CONTAINER,container_running=value["State"].get("Running") is True,
            container_gpu_uuids=devices,compute_pids=sorted(pids))

    def _exec(self,args,*,read_only=False):
        return ["docker","exec","-u","0","--env",COOKIE+"="+self.plan.session,
            CONTAINER,"/workspace/venv/bin/python",*(["-B"] if read_only else []),
            self.spec.container_path(Path(self.spec.source_host)/"sircus/isaac/candidate_v4_watchdog.py"),*args]

    def _metadata(self):
        """No host read of root-owned receipts or progress; use fixed metadata IPC."""
        self._authorized()
        require(self.plan is not None and self.payload_path is not None and self.payload_raw_sha256 is not None,
                "Runtime metadata requires the exact written watchdog input")
        nonce = uuid.uuid4().hex
        value = strict_json(self.ops.run(self._exec(["--snapshot",self.spec.container_path(self.payload_path),
            "--expected-payload-sha256",self.payload_raw_sha256,"--payload-fingerprint",self.payload_sha256,
            "--session",self.plan.session,"--nonce",nonce],read_only=True)))
        verify_metadata_snapshot(value,session=self.plan.session,nonce=nonce,payload_raw_sha256=self.payload_raw_sha256,
            payload_fingerprint=self.payload_sha256,output=self.spec.container_path(self.plan.output))
        fingerprints = [event["fingerprint"] for event in value["progress"]]
        require(fingerprints[:len(self.progress_fingerprints)] == self.progress_fingerprints,
                "Published runtime progress history was shortened or replaced")
        self.progress_fingerprints = fingerprints
        if self.owner is not None:
            require(self._receipt("runtime-owner.json",value) is not None,"Runtime owner receipt disappeared")
        return value

    def _receipt(self,name,metadata=None):
        require(name in RECEIPT_NAMES,"Only fixed runtime receipts may be read")
        metadata = self._metadata() if metadata is None else metadata
        value = metadata["receipts"][name]
        if value is None:return None
        require(value.get("schema") == SCHEMA and value.get("session") == self.plan.session
            and value.get("fingerprint") == canonical_hash({k:v for k,v in value.items() if k != "fingerprint"}),
            "Runtime receipt session/fingerprint changed")
        if name == "runtime-owner.json" and self.owner is not None:
            require(value["owner"]["pid"] == self.owner.pid
                and value["owner"]["start_token"] == self.owner.start_token
                and value["owner"]["cookie"] == self.owner.session
                and value.get("payload_sha256") == self.payload_sha256
                and value.get("native_owner") == self.native_owner,
                "Runtime owner receipt changed after start")
        if name == "runtime-exit.json":
            claimed = self._receipt("runtime-owner.json",metadata)
            require(claimed is not None and same_process(value.get("owner"),claimed.get("owner"))
                and value.get("native_owner") == claimed.get("native_owner")
                and value.get("payload_sha256") == claimed.get("payload_sha256")
                and type(value.get("exit_code")) is int,
                "Runtime exit is not bound to the exact owned watchdog/native child")
        return value

    def start_native(self,plan,absolute_deadline):
        self.bind_plan(plan);self._authorized()
        require(not self.attempted_start and self.child is None,"Production native start cannot retry")
        require(0 < absolute_deadline-self.monotonic() <= NATIVE_SECONDS,"Invalid production absolute deadline")
        require(self.lock_lease == (plan.scene_lock,plan.session),"Owned shared scene lock required immediately before native start")
        state = self.gpu_state(GPU_UUID)
        require(state["container_running"] and state["container_gpu_uuids"] == [GPU_UUID]
            and not state["compute_pids"],"Existing dedicated 4090/container is no longer idle; no native start")
        bindings,archive = self.spec.verify(plan)
        self.plan = plan
        require(Path(plan.output).is_dir(),"Supervisor must create the new output before its sole start")
        expected_intent=signed(dict(schema=SPEC_SCHEMA,session=plan.session,bindings=bindings,
            authorization_sha256=self.authorization_sha256,prepared_plan_fingerprint=plan.prepared_plan_fingerprint,
            status="single_start_intent_no_retry",execution_authorized=False,release_authorized=False,model_promoted=False))
        intent=Path(plan.output)/"launch-intent.json"
        require(sorted(p.name for p in Path(plan.output).iterdir())==["launch-intent.json"]
            and intent.is_file() and not intent.is_symlink() and strict_json(intent.read_bytes())==expected_intent,
            "Sole native start requires exact new-only supervisor intent, never partial old output")
        payload = signed(dict(schema=SCHEMA,session=plan.session,source_root=self.spec.container_path(self.spec.source_host),
            manifest=self.spec.container_path(self.spec.manifest_path),manifest_sha256=self.spec.manifest_sha256,
            output=self.spec.container_path(plan.output),absolute_deadline=absolute_deadline,
            prepared_plan_fingerprint=plan.prepared_plan_fingerprint,evaluation_source_sha256=plan.source_sha256,
            authorization_sha256=self.authorization_sha256,
            source_members_sha256={k:v for k,v in archive["source_members_sha256"].items()
                if k.startswith("sircus/isaac/") and (k.endswith(".py") or k == "sircus/isaac/python.sh")}))
        self.payload_sha256 = payload["fingerprint"]
        path = Path(plan.output)/"watchdog-input.json"
        write_new_json(path,payload)
        self.payload_path,self.payload_raw_sha256 = path,sha(path)
        self.attempted_start = True
        command = self._exec(["--execute",self.spec.container_path(path)])
        # Cookie is propagated to the watchdog itself and its one native child.
        env_index = command.index(CONTAINER)
        command[env_index:env_index] = ["--env","CUDA_VISIBLE_DEVICES=0","--env","PYTHONDONTWRITEBYTECODE=1"]
        self.child = self.ops.start(command,Path(plan.output)/"runtime-native.log")
        end = min(absolute_deadline,self.monotonic()+20)
        while self.monotonic() < end:
            receipt = self._receipt("runtime-owner.json")
            if receipt is not None:
                require(receipt.get("payload_sha256") == payload["fingerprint"],"Watchdog owner belongs to another payload")
                raw = receipt["owner"]
                require(raw.get("cookie") == plan.session,"Watchdog owner cookie changed")
                owner = ChildIdentity(raw["pid"],raw["start_token"],plan.session)
                require(owner.valid_for(plan),"Invalid watchdog ownership identity")
                self.owner = owner
                self.native_owner = receipt["native_owner"]
                return owner
            require(self.child.poll() is None,"Watchdog exited before ownership receipt; no retry")
            self.ops.sleep(.1)
        raise TimeoutError("Watchdog ownership handshake timed out; no retry")

    def identity(self,pid):
        self._authorized()
        require(self.owner is not None and pid == self.owner.pid,"Only the owned watchdog PID may be inspected")
        done = self._receipt("runtime-exit.json")
        if done is not None and self.child.poll() is not None:
            # An exit receipt is an attestation of the completed child, never a
            # permission to signal a possibly reused PID.
            return self.owner
        raw = strict_json(self.ops.run(self._exec(["--inspect",str(pid),"--session",self.plan.session])))
        if raw is None: return None
        if raw.get("cookie") != self.plan.session: return None
        return ChildIdentity(raw["pid"],raw["start_token"],self.plan.session)

    def poll(self,owner):
        self._authorized(); require(owner == self.owner,"Unowned production poll")
        metadata = self._metadata()
        events = metadata["progress"]
        if not events:
            return dict(phase="initialization",diagnostics=[],parking_resets=0,train_requests=0,optimizer_steps=0,
                evaluation_requests=0,evaluation_terminal=0,evaluation_releases=0,running=True,exit_code=None,audit_cpu_only=False)
        if self.progress_cursor < len(events):
            end = self.progress_cursor+1
            phase = events[self.progress_cursor]["snapshot"]["phase"]
            # Preserve every phase boundary even when the native child emits
            # several phases faster than host polling. Within one phase the
            # latest immutable counters replace redundant intermediate samples.
            while end < len(events) and events[end]["snapshot"]["phase"] == phase:
                end += 1
            snapshot = dict(events[end-1]["snapshot"])
            self.progress_cursor = end
            self.last_snapshot = snapshot
        else:
            snapshot = dict(self.last_snapshot)
        done,transport = self._receipt("runtime-exit.json",metadata),self.child.poll()
        if done is None or transport is None or (done["exit_code"] == 0 and transport == 0
                and self.progress_cursor < len(events)):
            snapshot.update(running=True,exit_code=None)
        else:
            code = done["exit_code"] if transport == 0 else transport
            snapshot.update(running=False,exit_code=code)
        return snapshot

    def terminate_owned(self,owner):
        self._authorized(); require(owner == self.owner,"Unowned production termination")
        done = self._receipt("runtime-exit.json")
        if done is not None and self.child.poll() is not None:
            require(done.get("status") == "complete_audit_receipts_required"
                or done.get("termination") in ("stopped","already_exited"),
                "Native termination remains unconfirmed; retain the shared lock and evidence")
            return "already_exited"
        if self.identity(owner.pid) != owner: return "ownership_mismatch"
        argument = json.dumps(dict(pid=owner.pid,start_token=owner.start_token,session=owner.session),separators=(",",":"))
        result = strict_json(self.ops.run(self._exec(["--stop",argument])))
        if result["status"] == "ownership_mismatch": return "ownership_mismatch"
        end = self.monotonic()+40
        while self.monotonic() < end:
            if self._receipt("runtime-exit.json") is not None and self.child.poll() is not None:
                return "stopped"
            self.ops.sleep(.1)
        raise TimeoutError("Owned watchdog stop unconfirmed; preserve lock and evidence")


def supervise(plan,runtime,*,execute=False):
    """One future explicitly authorized host start; mock operations in tests.

    The independent container watchdog enforces blocked-child deadlines. This
    host loop additionally refuses changed ownership, source/input bytes or
    counters, and cannot restart containers or signal a foreign process.
    """
    require(type(execute) is bool,"Explicit execution flag required")
    if not execute:
        return dict(schema=SPEC_SCHEMA,status="execution_disabled",native_application_starts=0,
            training_requests=0,optimizer_steps=0,model_promoted=False,execution_authorized=False)
    require(type(runtime) is ProductionRuntime and type(plan) is EvaluationLaunchPlan,
        "Explicit future production runtime/plan required")
    runtime.bind_plan(plan)
    bindings,_=runtime.spec.verify(plan)
    require(not Path(plan.output).exists() and not Path(plan.output).is_symlink(),
        "Evaluation output already exists; no retry or resume")
    start=runtime.monotonic()
    require(type(start) in (int,float) and math.isfinite(start),"Invalid supervisor monotonic clock")
    deadline=start+NATIVE_SECONDS
    lock=False;owner=None;result=None;cleanup=None
    try:
        free=runtime.free_bytes(plan.output)
        require(type(free) is int and free>=6*1024**3,"Original 6 GiB storage floor reached")
        state=runtime.gpu_state(GPU_UUID)
        require(state["container_running"] and state["container_gpu_uuids"]==[GPU_UUID]
            and not state["compute_pids"],"Existing 4090/container must already be idle and running")
        lock=runtime.acquire_scene_lock(plan.scene_lock,plan.session)
        require(lock,"Shared scene lock is owned by another project; no start")
        runtime.spec.verify(plan)
        require(not Path(plan.output).exists(),"Output appeared before single start")
        Path(plan.output).mkdir();write_new_json(Path(plan.output)/"launch-intent.json",signed(dict(
            schema=SPEC_SCHEMA,session=plan.session,bindings=bindings,authorization_sha256=runtime.authorization_sha256,
            prepared_plan_fingerprint=plan.prepared_plan_fingerprint,status="single_start_intent_no_retry",
            execution_authorized=False,release_authorized=False,model_promoted=False)))
        owner=runtime.start_native(plan,deadline)
        previous=dict(evaluation_requests=0,evaluation_terminal=0,evaluation_releases=0)
        while True:
            now=runtime.monotonic()
            require(type(now) in (int,float) and math.isfinite(now) and now>=start and now<=deadline,
                "Host evaluation wall limit reached")
            require(runtime.identity(owner.pid)==owner,"Owned watchdog identity changed; no PID-only action")
            snapshot=runtime.poll(owner)
            from candidate_v4_watchdog import validate_snapshot
            validate_snapshot(snapshot)
            require(all(snapshot[k]>=v for k,v in previous.items()),"Host evaluation counters moved backwards")
            require(snapshot["evaluation_requests"]<=210
                and snapshot["evaluation_releases"]<=snapshot["evaluation_terminal"]<=snapshot["evaluation_requests"],
                "Host TEST-only budget exceeded")
            previous={k:snapshot[k] for k in previous}
            if not snapshot["running"]:
                require(snapshot["exit_code"]==0 and snapshot["phase"]=="audit"
                    and snapshot["evaluation_requests"]==snapshot["evaluation_terminal"]==210,
                    "Native evaluation stopped before independent audit receipts")
                result=dict(schema=SPEC_SCHEMA,status="native_complete_audit_receipts_required",
                    session=plan.session,bindings=bindings,watchdog=dict(pid=owner.pid,start_token=owner.start_token,session=owner.session),
                    counts=previous,native_start_intents=1,native_process_starts=1,native_application_starts=None,
                    native_start_scope="One native entry process observed; App starts require the application receipt.",
                    training_requests=0,optimizer_steps=0,
                    model_promoted=False,independent_audit_passed=False,execution_authorized=False,
                    release_authorized=False)
                cleanup=runtime.terminate_owned(owner)
                require(cleanup=="already_exited","Successful native exit has unconfirmed owned workers")
                break
            runtime.sleep(.1)
    except BaseException as error:
        if owner is not None:
            try:cleanup=runtime.terminate_owned(owner)
            except BaseException as cleanup_error:cleanup="unconfirmed: "+repr(cleanup_error)
        elif runtime.attempted_start:
            cleanup="unknown_child_identity_lock_retained"
        else:
            cleanup="no_native_start"
        result=dict(schema=SPEC_SCHEMA,status="failed_no_retry",session=plan.session,error=repr(error),
            cleanup=cleanup,native_start_intents=int(runtime.attempted_start),
            native_process_starts=None if runtime.attempted_start else 0,
            native_application_starts=None if runtime.attempted_start else 0,
            native_start_scope="Start intent is not proof of simulator initialization.",training_requests=0,
            optimizer_steps=0,model_promoted=False,independent_audit_passed=False,
            execution_authorized=False,release_authorized=False)
    finally:
        if lock and cleanup in ("already_exited","stopped","no_native_start"):
            runtime.release_scene_lock(plan.scene_lock,plan.session)
        if result is not None and Path(plan.output).is_dir():
            write_new_json(Path(plan.output)/"launch-result.json",signed(result))
    return result
