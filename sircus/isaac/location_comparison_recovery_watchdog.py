"""One CPU-only saved-data audit; failed native evidence is never relabelled.

Imports do not launch anything. The explicit supervisor uses the original
identity/pidfd cleanup helpers and measures its exact child with wait4. It never
opens the scene lock, dispatches simulator work, or restarts a service.
"""

import hashlib
import math
import os
from pathlib import Path
import re
import subprocess

from candidate_continuation_journal import write_new_json
from candidate_continuation_ledger import canonical_hash
from candidate_continuation_watchdog import COOKIE, LinuxProcessOps, same_process, stop_owned, strict_json
from location_comparison_audit import digest as scientific_digest
from location_comparison_watchdog import _absolute, _clock, _final_wall, _observe_child, _sync_directory

SCHEMA = "sircus-location-comparison-recovery-watchdog-v1"
REPORT_SCHEMA = "sircus-location-comparison-recovery-audit-v1"
REPORT_FILE = "recovery-audit.json"
OWNER_FILE = "recovery-audit-owner.json"
EXIT_FILE = "recovery-audit-exit.json"
LIMITS = dict(wall_s=7200, min_free_bytes=6 * 2**30, cpu_cores=4,
              memory_bytes=8 * 2**30, cpu_seconds=28800)
BINDINGS = ("manifest_sha256", "manifest_fingerprint", "comparison_fingerprint", "native_session_id",
            "native_exit_fingerprint", "native_exit_sha256", "inventory_sha256",
            "retirement_proof_sha256", "source_commit")
MAX_REPORT_BYTES = 32 * 2**20


def require(value, message):
    if not value:
        raise ValueError(message)


def signed(value):
    return dict(value, fingerprint=canonical_hash(value))


def _sealed(value):
    return (type(value) is dict and value.get("fingerprint") ==
            canonical_hash({k: v for k, v in value.items() if k != "fingerprint"}))


def validate_request(request):
    require(_sealed(request) and request.get("schema") == "sircus-location-comparison-recovery-request-v1",
            "Recovery request fingerprint/schema changed")
    for key in BINDINGS:
        size = 32 if key == "native_session_id" else 40 if key == "source_commit" else 64
        require(type(request.get(key)) is str and re.fullmatch(r"[0-9a-f]{%d}" % size, request[key]),
                "Invalid recovery request binding: " + key)
    for key in ("native_output", "inventory_path", "retirement_proof_path"):
        _absolute(request.get(key, ""), key)
    require(request.get("synthetic", False) is False, "Synthetic recovery request cannot execute")


def _read_file(path, maximum=MAX_REPORT_BYTES):
    path = _absolute(path, "recovery evidence file")
    require(path.is_file(), "Missing recovery evidence file")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "Recovery report exceeds fixed size bound")
    return raw, strict_json(raw)


def inspect_report(output, request):
    """Independently bind completed report bytes; scientific checks live in reader."""
    raw, value = _read_file(Path(output) / REPORT_FILE)
    require(type(value) is dict and value.get("fingerprint") == scientific_digest(
                {k:v for k,v in value.items() if k != "fingerprint"})
            and value.get("schema") == REPORT_SCHEMA
            and value.get("status") == "data_verified_shutdown_incomplete"
            and value.get("synthetic") is False and value.get("model_promoted") is False
            and value.get("original_audit_passed") is False and value.get("clean_shutdown") is False,
            "Recovery report is not a native data-verification result")
    require(value.get("recovery_request_fingerprint") == request["fingerprint"]
            and all(value.get(k) == request[k] for k in BINDINGS),
            "Recovery report process/input identity changed")
    require(all(value.get(key) is True for key in ("checkpoint_bytes_independently_verified",
            "model_likelihoods_independently_verified", "native_physics_verified")),
            "Recovery report lacks complete scientific verification")
    return dict(report_sha256=hashlib.sha256(raw).hexdigest(),
                report_fingerprint=value["fingerprint"], report_status=value["status"])


def _receipt_bindings(request):
    return dict(manifest_sha256=request["manifest_sha256"], native_session_id=request["native_session_id"],
        runtime_fingerprint=request["manifest_fingerprint"],
        comparison_fingerprint=request["comparison_fingerprint"],
        recovery_request_fingerprint=request["fingerprint"],
        native_exit_fingerprint=request["native_exit_fingerprint"],
        native_exit_sha256=request["native_exit_sha256"],
        inventory_sha256=request["inventory_sha256"],
        retirement_proof_sha256=request["retirement_proof_sha256"], source_commit=request["source_commit"])


def _cpu_only_environment():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Recovery CUDA visibility must be empty")
    require(os.environ.get("NVIDIA_VISIBLE_DEVICES", "") in ("", "none", "void"),
            "Recovery NVIDIA visibility must be disabled")
    require(not list(Path("/dev").glob("nvidia[0-9]*")), "Recovery container exposes GPU devices")


class LinuxRecoveryOps(LinuxProcessOps):
    """Exact-child wait4 owns reaping; inherited pidfd operations own signalling."""

    def __init__(self, cgroup_root=Path("/sys/fs/cgroup")):
        self.cgroup_root = Path(cgroup_root)
        self.usage = None

    def prepare(self):
        _cpu_only_environment()
        require(hasattr(os, "wait4"), "Exact-child wait4 measurements unavailable")
        super().prepare()
        self.resources()

    def start(self, command, environment):
        require(getattr(self, "child", None) is None, "Recovery child start already consumed")
        self.child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return self.child.pid

    def poll(self, pid):
        require(pid == self.child.pid, "Cannot reap an unrelated process")
        if self.child.returncode is not None:
            return self.child.returncode
        try:
            found, status, usage = os.wait4(pid, os.WNOHANG)
        except ChildProcessError as exc:
            raise ValueError("Audit child was reaped outside measured wait4") from exc
        if not found:
            return None
        require(found == pid, "wait4 returned an unrelated child")
        self.child.returncode = os.waitstatus_to_exitcode(status)
        self.usage = dict(cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime,
            peak_rss_bytes=int(usage.ru_maxrss) * 1024,
            measurement_scope="wait4_exact_audit_child_including_reaped_descendants")
        return self.child.returncode

    def measurement(self):
        return self.usage

    def resources(self):
        _cpu_only_environment()
        root = self.cgroup_root
        if (root / "cgroup.controllers").is_file():
            memory = int((root / "memory.max").read_text().strip())
            used = int((root / "memory.current").read_text().strip())
            quota, period = (root / "cpu.max").read_text().split()
            cores = int(quota) / int(period)
            stats = dict(line.split() for line in (root / "cpu.stat").read_text().splitlines())
            cpu = int(stats["usage_usec"]) / 1e6
        else:
            memory = int((root / "memory/memory.limit_in_bytes").read_text().strip())
            used = int((root / "memory/memory.usage_in_bytes").read_text().strip())
            quota = int((root / "cpu/cpu.cfs_quota_us").read_text().strip())
            period = int((root / "cpu/cpu.cfs_period_us").read_text().strip())
            cores = quota / period
            cpu = int((root / "cpuacct/cpuacct.usage").read_text().strip()) / 1e9
        require(0 < memory <= LIMITS["memory_bytes"] and 0 < cores <= LIMITS["cpu_cores"],
                "Missing or excessive CPU/memory cgroup hard limits")
        return dict(cgroup_memory_limit_bytes=memory, cgroup_cpu_cores=cores,
                    memory_current_bytes=used, cpu_usage_seconds=cpu, visible_gpu_devices=0)


def validate_child_owner(output, request, session_id, *, ops=None):
    """Child handshake: independently verify its live direct watchdog ancestry."""
    validate_request(request)
    ops = ops or LinuxRecoveryOps()
    _cpu_only_environment()
    deadline = ops.monotonic() + 5
    path = _absolute(output, "recovery output") / OWNER_FILE
    while not path.exists():
        require(ops.monotonic() < deadline, "Recovery ownership handshake timed out")
        ops.sleep(.01)
    _, receipt = _read_file(path, 65536)
    own = ops.identity(ops.own_pid())
    parent = ops.identity(os.getppid())
    require(_sealed(receipt) and receipt.get("schema") == SCHEMA
            and receipt.get("session_id") == session_id
            and all(receipt.get(k) == v for k, v in _receipt_bindings(request).items()),
            "Recovery owner receipt binding changed")
    require(same_process(own, receipt.get("child_owner"))
            and same_process(parent, receipt.get("owner"))
            and own.get("cookie") == parent.get("cookie") == session_id
            and own.get("ppid") == parent.get("pid") and own.get("sid") == own.get("pid"),
            "Recovery child ancestry changed")
    require(receipt.get("limits") == LIMITS and receipt.get("cuda_visible_devices") == ""
            and ops.monotonic() < receipt["deadline_monotonic"], "Recovery child bounds changed")
    return receipt


def supervise_recovery(command, *, output, storage_root, request, session_id,
                       integrity_check, verify_report, ops=None, poll_s=.5,
                       execute_approved=False):
    """One fresh, explicitly approved CPU reader; never retries or opens a lock."""
    require(execute_approved is True, "Explicit saved-data audit approval required")
    validate_request(request)
    require(type(session_id) is str and re.fullmatch(r"[0-9a-f]{32}", session_id)
            and session_id != request["native_session_id"], "Distinct recovery process session required")
    require(type(command) is list and 0 < len(command) <= 32
            and all(type(v) is str and 0 < len(v) <= 4096 and "\0" not in v for v in command)
            and sum(map(len, command)) <= 16384 and Path(command[0]).is_absolute(), "Explicit absolute audit argv required")
    require(callable(integrity_check) and callable(verify_report), "Independent integrity/report verifiers required")
    require(type(poll_s) in (float, int) and math.isfinite(poll_s) and 0 < poll_s <= 1,
            "Invalid bounded polling interval")
    output, storage_root = _absolute(output, "fresh audit output"), _absolute(storage_root, "storage filesystem")
    native = _absolute(request["native_output"], "original native evidence")
    require(output.parent.is_dir() and not output.exists() and storage_root.is_dir(),
            "Audit output exists or parent/storage directory is missing")
    require(not output.is_relative_to(native) and not native.is_relative_to(output),
            "Audit output overlaps original native evidence")
    ops = ops or LinuxRecoveryOps()
    start = ops.monotonic()
    require(_clock(start), "Invalid recovery clock")
    deadline, previous, cpu_base, cpu_previous = start + LIMITS["wall_s"], start, None, None
    owner = child = None
    starts = process_starts = 0
    code = reason = error = cleanup = None
    safe = True
    report = dict(report_sha256=None, report_fingerprint=None, report_status=None)
    latest_resources, peak_memory, owner_sha = None, 0, None
    output.mkdir(mode=0o700)
    _sync_directory(output.parent)

    def check():
        nonlocal previous, cpu_base, cpu_previous, latest_resources, peak_memory
        now = ops.monotonic()
        require(_clock(now) and previous <= now < deadline, "Recovery wall deadline/clock failed")
        previous = now
        free = ops.free_bytes(storage_root)
        require(type(free) is int and free >= LIMITS["min_free_bytes"], "Recovery storage floor reached")
        require(integrity_check() is True, "Recovery immutable inputs changed")
        require(same_process(ops.identity(ops.own_pid()), owner), "Recovery watchdog identity changed")
        resources = ops.resources()
        require(type(resources.get("visible_gpu_devices")) is int and resources["visible_gpu_devices"] == 0,
                "Recovery exposes GPU devices")
        mem, cores, used, cpu = (resources.get(k) for k in ("cgroup_memory_limit_bytes", "cgroup_cpu_cores",
                                                         "memory_current_bytes", "cpu_usage_seconds"))
        require(type(mem) is int and 0 < mem <= LIMITS["memory_bytes"]
                and type(cores) in (int, float) and math.isfinite(cores) and 0 < cores <= LIMITS["cpu_cores"],
                "Recovery cgroup hard limits changed")
        require(type(used) is int and 0 <= used <= mem, "Recovery memory bound exceeded")
        require(_clock(cpu) and (cpu_previous is None or cpu >= cpu_previous), "Recovery CPU accounting invalid")
        if cpu_base is None:
            cpu_base = cpu
        require(cpu - cpu_base <= LIMITS["cpu_seconds"], "Recovery CPU time bound exceeded")
        cpu_previous, latest_resources = cpu, {k:resources[k] for k in ("cgroup_memory_limit_bytes",
            "cgroup_cpu_cores","memory_current_bytes","cpu_usage_seconds","visible_gpu_devices")}
        peak_memory = max(peak_memory, used)

    try:
        owner = ops.identity(ops.own_pid())
        require(owner is not None and owner.get("cookie") == session_id, "Recovery watchdog session cookie missing")
        check()
        ops.prepare()
        check()
        environment = dict(os.environ, **{COOKIE:session_id, "CUDA_VISIBLE_DEVICES":"",
            "NVIDIA_VISIBLE_DEVICES":"void", "PYTHONDONTWRITEBYTECODE":"1",
            "OMP_NUM_THREADS":"4", "MKL_NUM_THREADS":"4", "OPENBLAS_NUM_THREADS":"4",
            "SIRCUS_RECOVERY_REQUEST_FINGERPRINT":request["fingerprint"],
            "SIRCUS_RECOVERY_OWNER_FILE":str(output / OWNER_FILE),
            "SIRCUS_RECOVERY_DEADLINE_MONOTONIC":str(deadline)})
        starts = 1
        pid = ops.start(command, environment)
        process_starts = 1
        child, code = _observe_child(ops, pid, session_id, owner=None, check=check, label="Recovery audit child")
        require(child is not None and code is None, "Recovery child identity unavailable after sole start")
        write_new_json(output / OWNER_FILE, signed(dict(schema=SCHEMA,session_id=session_id,
            **_receipt_bindings(request),owner=owner,child_owner=child,command=command,
            limits=LIMITS,start_monotonic=start,deadline_monotonic=deadline,
            storage_root=str(storage_root),cuda_visible_devices="",visible_gpu_devices=0)))
        owner_sha = hashlib.sha256((output / OWNER_FILE).read_bytes()).hexdigest()
        while True:
            _, code = _observe_child(ops,pid,session_id,owner=child,check=check,label="Recovery audit child")
            if code is not None:
                check()
                require(type(code) is int and code == 0, "Recovery audit child exited unsuccessfully")
                require(not ops.owned(pid,session_id), "Recovery audit left owned workers alive")
                require(verify_report(output,request) is True, "Independent recovery report verification failed")
                report = inspect_report(output,request)
                check()
                require(inspect_report(output,request) == report, "Recovery report changed after verification")
                check()
                break
            ops.sleep(poll_s)
    except BaseException as failure:
        error, reason = type(failure).__name__, (str(failure) or type(failure).__name__)[:2048]
        if starts:
            safe = False
            try:
                cleanup = stop_owned(ops,child,session_id,deadline=min(deadline,ops.monotonic()+35)) \
                    if child is not None else "unknown_child_identity"
                safe = cleanup in ("stopped","already_exited")
                if child is not None:
                    code = ops.poll(child["pid"])
            except BaseException as stop_error:
                cleanup = "termination_unconfirmed:" + type(stop_error).__name__
    usage_raw = ops.measurement() if process_starts else None
    usage = ({k:usage_raw.get(k) for k in ("cpu_user_s","cpu_system_s","peak_rss_bytes","measurement_scope")}
             if type(usage_raw) is dict else None)
    if reason is None:
        if not (type(usage) is dict and all(_clock(usage.get(k)) for k in ("cpu_user_s","cpu_system_s"))
                and usage["cpu_user_s"] + usage["cpu_system_s"] <= LIMITS["cpu_seconds"]
                and type(usage.get("peak_rss_bytes")) is int and 0 <= usage["peak_rss_bytes"] <= LIMITS["memory_bytes"]
                and usage.get("measurement_scope") == "wait4_exact_audit_child_including_reaped_descendants"):
            reason,error = "Exact audit child resource measurement missing", "ValueError"
    result = signed(dict(schema=SCHEMA,session_id=session_id,**_receipt_bindings(request),
        owner=owner,child_owner=child,owner_receipt_sha256=owner_sha,command=command,limits=LIMITS,storage_root=str(storage_root),
        status="recovery_exit_zero_report_verified" if reason is None else "recovery_stopped_no_retry",
        audit_child_start_attempts=starts,audit_process_starts=process_starts,
        exit_code=code,reason=reason,error_type=error,termination=cleanup,cleanup_safe=safe,
        start_monotonic=start,deadline_monotonic=deadline,wall_s=_final_wall(ops,start),
        cuda_visible_devices="",visible_gpu_devices=0,native_application_starts=0,
        optimizer_steps=0,new_shots=0,model_promoted=False,independent_report_verified=reason is None,
        cgroup_limits=latest_resources,sampled_peak_cgroup_memory_bytes=peak_memory,
        measured_cgroup_cpu_s=None if cpu_base is None else cpu_previous-cpu_base,
        **(usage or dict(cpu_user_s=None,cpu_system_s=None,peak_rss_bytes=None,measurement_scope=None)),
        **report))
    write_new_json(output / EXIT_FILE,result)
    return result


def verify_exit(output, request, session_id, *, expected_exit_sha256, expected_report_sha256):
    """Publisher-facing independent process/report identity join after shutdown."""
    validate_request(request)
    raw, receipt = _read_file(Path(output)/EXIT_FILE,65536)
    require(hashlib.sha256(raw).hexdigest() == expected_exit_sha256 and _sealed(receipt)
            and receipt.get("schema") == SCHEMA
            and receipt.get("status") == "recovery_exit_zero_report_verified"
            and receipt.get("session_id") == session_id
            and all(receipt.get(k) == v for k,v in _receipt_bindings(request).items()),
            "Recovery exit receipt identity changed")
    require(receipt.get("exit_code") == 0 and type(receipt.get("exit_code")) is int
            and receipt.get("reason") is None and receipt.get("error_type") is None
            and receipt.get("termination") is None and receipt.get("cleanup_safe") is True
            and receipt.get("independent_report_verified") is True
            and type(receipt.get("audit_child_start_attempts")) is int
            and type(receipt.get("audit_process_starts")) is int
            and receipt.get("audit_child_start_attempts") == receipt.get("audit_process_starts") == 1
            and receipt.get("limits") == LIMITS and receipt.get("cuda_visible_devices") == ""
            and all(type(receipt.get(k)) is int for k in ("visible_gpu_devices","native_application_starts","optimizer_steps","new_shots"))
            and receipt.get("visible_gpu_devices") == receipt.get("native_application_starts") ==
                receipt.get("optimizer_steps") == receipt.get("new_shots") == 0
            and receipt.get("model_promoted") is False, "Recovery exit lacks CPU-only bounded success")
    owner,child = receipt.get("owner"),receipt.get("child_owner")
    require(type(owner) is dict and type(child) is dict
            and owner.get("cookie") == child.get("cookie") == session_id
            and child.get("ppid") == owner.get("pid") and child.get("sid") == child.get("pid")
            and all(type(r.get("pid")) is int and r["pid"] > 1 and type(r.get("start_token")) is str
                    and r["start_token"] for r in (owner,child)), "Recovery exit process ancestry incomplete")
    require(_clock(receipt.get("wall_s")) and receipt["wall_s"] < LIMITS["wall_s"]
            and all(_clock(receipt.get(k)) for k in ("cpu_user_s","cpu_system_s"))
            and receipt["cpu_user_s"]+receipt["cpu_system_s"] <= LIMITS["cpu_seconds"]
            and type(receipt.get("peak_rss_bytes")) is int and 0 <= receipt["peak_rss_bytes"] <= LIMITS["memory_bytes"],
            "Recovery exit resource measurement invalid")
    owner_raw, owner_receipt = _read_file(Path(output)/OWNER_FILE,65536)
    require(hashlib.sha256(owner_raw).hexdigest() == receipt.get("owner_receipt_sha256")
            and _sealed(owner_receipt) and owner_receipt.get("schema") == SCHEMA
            and all(owner_receipt.get(k) == receipt.get(k) for k in (*_receipt_bindings(request),
                "session_id","owner","child_owner","command","limits","start_monotonic",
                "deadline_monotonic","storage_root","cuda_visible_devices","visible_gpu_devices")),
            "Recovery owner/exit binding changed")
    controls = receipt.get("cgroup_limits",{})
    require(type(controls.get("cgroup_memory_limit_bytes")) is int
            and 0 < controls["cgroup_memory_limit_bytes"] <= LIMITS["memory_bytes"]
            and _clock(controls.get("cgroup_cpu_cores")) and 0 < controls["cgroup_cpu_cores"] <= LIMITS["cpu_cores"]
            and receipt.get("measurement_scope") == "wait4_exact_audit_child_including_reaped_descendants"
            and _clock(receipt.get("measured_cgroup_cpu_s")) and receipt["measured_cgroup_cpu_s"] <= LIMITS["cpu_seconds"]
            and type(receipt.get("sampled_peak_cgroup_memory_bytes")) is int
            and 0 <= receipt["sampled_peak_cgroup_memory_bytes"] <= LIMITS["memory_bytes"]
            and _clock(receipt.get("start_monotonic"))
            and receipt.get("deadline_monotonic") == receipt["start_monotonic"] + LIMITS["wall_s"],
            "Recovery measured resource/control binding changed")
    report = inspect_report(output,request)
    require(report["report_sha256"] == expected_report_sha256
            and all(receipt.get(k) == v for k,v in report.items()), "Recovery exit/report byte binding changed")
    return receipt
