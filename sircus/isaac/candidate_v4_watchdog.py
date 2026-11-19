"""Independent, explicitly invoked Linux watchdog for one owned native child.

Importing this module does not inspect processes, start a child or send a signal.
The production entry below is reached only by an explicitly authorized runtime.
Tests inject fake process/clock/procfs operations; no native operation is needed.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import subprocess
import time

from candidate_continuation_journal import write_new_json
from candidate_continuation_ledger import canonical_hash
from candidate_continuation_pidfd import capability_preflight, pidfd_open, pidfd_send_signal
from candidate_continuation_launcher import DiagnosticLedger

NATIVE_SECONDS = 3000
MIN_FREE_BYTES = 6 * 1024 ** 3
PHASE_LIMITS = (("initialization",600),("evaluation",1800),("audit",600))
PHASES = tuple(name for name,_ in PHASE_LIMITS)

SCHEMA = "sircus-v4-evaluation-watchdog-v1"
PROGRESS_SCHEMA = "sircus-v4-evaluation-progress-v1"
SNAPSHOT_SCHEMA = "sircus-v4-evaluation-metadata-snapshot-v1"
RECEIPT_NAMES = ("runtime-owner.json", "runtime-exit.json")
COOKIE = "SIRCUS_V4_EVALUATION_SESSION"
MAX_PROGRESS = 10000
TERM_GRACE_S = 30


def require(value, message):
    if not value:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate runtime JSON key")
            result[key] = value
        return result
    def reject(_):
        raise ValueError("Nonfinite runtime JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed(value):
    return dict(value, fingerprint=canonical_hash(value))


def validate_snapshot(value):
    fields = {"phase","diagnostics","parking_resets","evaluation_requests","evaluation_terminal",
        "evaluation_releases","train_requests","optimizer_steps","running","exit_code","audit_cpu_only"}
    require(type(value) is dict and set(value) == fields and value.get("phase") in PHASES,
            "Malformed V4 evaluation progress snapshot")
    require(type(value["diagnostics"]) is list and type(value["parking_resets"]) is int
        and value["parking_resets"] >= 0,"Malformed startup progress")
    require(all(type(value[k]) is int and value[k] >= 0 for k in
        ("evaluation_requests","evaluation_terminal","evaluation_releases","train_requests","optimizer_steps")),
        "Malformed evaluation-only counters")
    require(value["train_requests"] == value["optimizer_steps"] == 0,
        "Training or optimizer work is forbidden in evaluation")
    require(type(value["running"]) is bool and type(value["audit_cpu_only"]) is bool
        and ((value["running"] and value["exit_code"] is None)
             or (not value["running"] and type(value["exit_code"]) is int)),
        "Malformed progress lifecycle")
    return value


class ProgressWriter:
    """One append-only progress owner; cannot reopen or replace prior receipts."""

    def __init__(self, output, session, *, clock=time.monotonic):
        require(type(session) is str and re.fullmatch(r"[0-9a-f]{32}",session), "Invalid progress session")
        self.directory = Path(output)/"runtime-progress"
        self.directory.mkdir()
        self.session, self.clock, self.seq, self.head = session, clock, 0, "0"*64

    def publish(self, snapshot):
        require(self.seq < MAX_PROGRESS, "Runtime progress bound exceeded")
        validate_snapshot(snapshot)
        now = self.clock()
        require(type(now) in (float,int) and math.isfinite(now) and now >= 0, "Invalid progress clock")
        value = signed(dict(schema=PROGRESS_SCHEMA, session=self.session, seq=self.seq,
            previous_sha256=self.head, monotonic_s=now, snapshot=snapshot))
        write_new_json(self.directory/f"event-{self.seq:06d}.json", value)
        self.seq += 1; self.head = value["fingerprint"]
        return value


def read_progress(output, session):
    """Verify complete published records; pending atomic-write files are not read."""
    directory = Path(output)/"runtime-progress"
    require(not directory.is_symlink(), "Linked runtime progress directory")
    if not directory.exists():
        return []
    require(directory.is_dir() and not directory.is_symlink(), "Linked runtime progress directory")
    paths = sorted(p for p in directory.iterdir() if not p.name.startswith(".pending-"))
    require(len(paths) <= MAX_PROGRESS, "Runtime progress bound exceeded")
    events = []
    for i,path in enumerate(paths):
        require(path.name == f"event-{i:06d}.json" and path.is_file() and not path.is_symlink(),
                "Runtime progress gap, duplicate or unexpected file")
        event = strict_json(path.read_bytes())
        events.append(event)
    return verify_progress(events,session)


def verify_progress(events, session):
    """Recheck the complete chain after read-only container transport."""
    require(type(events) is list and len(events) <= MAX_PROGRESS,"Runtime progress bound exceeded")
    head, previous_time = "0"*64, -1.
    for i,event in enumerate(events):
        require(type(event) is dict,"Malformed runtime progress event")
        require(set(event) == {"schema","session","seq","previous_sha256","monotonic_s","snapshot","fingerprint"}
            and event["schema"] == PROGRESS_SCHEMA and event["session"] == session
            and type(event["seq"]) is int and event["seq"] == i and event["previous_sha256"] == head
            and event["fingerprint"] == canonical_hash({k:v for k,v in event.items() if k != "fingerprint"}),
            "Runtime progress session/chain changed")
        stamp = event["monotonic_s"]
        require(type(stamp) in (int,float) and math.isfinite(stamp) and stamp >= previous_time,
                "Runtime progress clock moved backwards")
        validate_snapshot(event["snapshot"])
        previous_time, head = stamp, event["fingerprint"]
    return events


def proc_identity(pid, proc_root=Path("/proc")):
    """Read start tick and cookie; no signalling and no PID-only ownership."""
    require(type(pid) is int and pid > 1, "Invalid owned PID")
    path = Path(proc_root)/str(pid)
    try:
        raw = (path/"stat").read_text()
        tail = raw[raw.rfind(")")+2:].split()
        boot = (Path(proc_root)/"sys/kernel/random/boot_id").read_text().strip()
        env = dict(item.split(b"=",1) for item in (path/"environ").read_bytes().split(b"\0") if b"=" in item)
        return dict(pid=pid, start_token=boot+":"+tail[19], ppid=int(tail[1]),
            sid=int(tail[3]), cookie=env.get(COOKIE.encode(),b"").decode(), state=tail[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(left, right):
    return left is not None and right is not None and all(left[k] == right[k] for k in ("pid","start_token","cookie"))


class LinuxProcessOps:
    """Linux operations are lazy and used only by the explicit execute path."""

    def monotonic(self): return time.monotonic()
    def sleep(self, seconds): time.sleep(seconds)
    def own_pid(self): return os.getpid()
    def identity(self, pid): return proc_identity(pid)
    def local_path(self, path): return Path(path)
    def free_bytes(self,path): return shutil.disk_usage(path).free

    def prepare(self):
        # Check the actual pidfd kernel path using only our own process and
        # signal zero; close its fd before subreaper setup or any app start.
        self.pidfd_capability = capability_preflight()
        # Adopt orphaned grandchildren so that failed native workers cannot
        # outlive this watchdog merely by losing their immediate parent.
        import ctypes
        require(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0) == 0,
                "Owned-child subreaper setup failed")
        offsets = Path("/proc/self/timens_offsets")
        if offsets.exists():
            require(all(line.split()[1:] == ["0","0"] for line in offsets.read_text().splitlines()),
                    "Unreviewed container clock namespace offset")

    def start(self, command, environment):
        self.child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
            start_new_session=True)
        return self.child.pid

    def poll(self, pid): return self.child.poll()

    def owned(self, owner_pid, session):
        """Only descendants/adopted children of this cookie-bound watchdog.

        A helper may scrub its own environment while remaining our descendant;
        immutable PID/start-time plus verified ancestry still prove ownership.
        """
        anchor = self.identity(self.own_pid())
        require(anchor is not None and anchor["cookie"] == session,"Watchdog ownership cookie changed")
        identities = {}
        for path in Path("/proc").iterdir():
            if path.name.isdecimal() and int(path.name) > 1:
                try:
                    record = self.identity(int(path.name))
                except (PermissionError, ValueError, IndexError):
                    continue
                if record is not None:
                    identities[record["pid"]] = record
        result = []
        for record in identities.values():
            parent, seen = record["pid"], set()
            while parent in identities and parent not in seen:
                seen.add(parent)
                if identities[parent]["ppid"] == self.own_pid():
                    if record["state"] in ("Z","X"):
                        # Popen owns the direct native child's wait status.
                        # Reap only already-dead adopted grandchildren here.
                        if record["ppid"] == self.own_pid() and record["pid"] != getattr(getattr(self,"child",None),"pid",None):
                            try:os.waitpid(record["pid"],os.WNOHANG)
                            except ChildProcessError:pass
                    else:
                        result.append(record)
                    break
                parent = identities[parent]["ppid"]
        return result

    def signal_owned(self, record, signum, session):
        current = self.identity(record["pid"])
        if current is None:
            return "already_exited"
        if not same_process(current,record):
            return "ownership_mismatch"
        try:descriptor = pidfd_open(record["pid"],0)
        except ProcessLookupError:return "already_exited"
        try:
            # Opening a pidfd pins one process identity. Recheck the start tick
            # and cookie after acquisition, then signal the fd, never a group.
            if not same_process(self.identity(record["pid"]),record):
                return "ownership_mismatch"
            owned = self.owned(record["pid"],session)
            if not any(r["pid"] == record["pid"] and r["start_token"] == record["start_token"] for r in owned):
                return "ownership_mismatch"
            pidfd_send_signal(descriptor,signum,None,0)
            return "signalled"
        except ProcessLookupError:
            return "already_exited"
        finally:
            os.close(descriptor)


def stop_owned(ops, owner, session, *, deadline=None):
    """Stop every independently proven descendant, never a reused identity.

    A helper race must not let other known workers escape their deadline. Keep
    the ownership-mismatch result (and host lock) while continuing safe cleanup.
    """
    expected=ops.identity(owner["pid"])
    leader_reused=expected is not None and expected["start_token"]!=owner["start_token"]
    mismatched=leader_reused
    refused=set()
    def remaining():
        records=ops.owned(owner["pid"],session)
        return [r for r in records if not (leader_reused and r["pid"]==owner["pid"])
            and (r["pid"],r["start_token"]) not in refused]
    for signum in (signal.SIGTERM,signal.SIGKILL):
        records=remaining()
        if not records:
            return "ownership_mismatch" if mismatched else "already_exited" if signum==signal.SIGTERM else "stopped"
        for record in reversed(records):
            result=ops.signal_owned(record,signum,session)
            if result=="ownership_mismatch":
                mismatched=True
                refused.add((record["pid"],record["start_token"]))
        if not remaining():return "ownership_mismatch" if mismatched else "stopped"
        end=ops.monotonic()+(TERM_GRACE_S if signum==signal.SIGTERM else 5)
        if deadline is not None:end=min(end,deadline)
        while ops.monotonic()<end:
            if not remaining():return "ownership_mismatch" if mismatched else "stopped"
            ops.sleep(.1)
    return "ownership_mismatch" if mismatched else "stop_unconfirmed"


def fixed_command(payload):
    source = PurePosixPath(payload["source_root"])
    require(source.is_absolute() and source.is_relative_to("/workspace") and ".." not in source.parts,
            "Native source is outside the existing workspace mount")
    return ["/bin/bash",str(source/"sircus/isaac/python.sh"),
        str(source/"sircus/isaac/candidate_v4_entry.py"),
        "--headless","--enable_cameras","--device","cuda:0",
        "--manifest",payload["manifest"],"--output",payload["output"],"--session",payload["session"],
        "--expected-manifest-sha256",payload["manifest_sha256"],"--execute-approved"]


def validate_payload(payload):
    require(type(payload) is dict and set(payload) == {"schema","session","source_root","manifest",
        "manifest_sha256","output","absolute_deadline","source_members_sha256",
        "prepared_plan_fingerprint","evaluation_source_sha256","authorization_sha256","fingerprint"},
        "Watchdog payload fields changed")
    require(payload["schema"] == SCHEMA and re.fullmatch(r"[0-9a-f]{32}",payload["session"])
        and payload["fingerprint"] == canonical_hash({k:v for k,v in payload.items() if k != "fingerprint"}),
        "Watchdog payload identity changed")
    for key in ("manifest","output"):
        path = PurePosixPath(payload[key])
        require(path.is_absolute() and path.is_relative_to("/workspace") and ".." not in path.parts,
                "Watchdog paths must remain in the existing mount")
    require(type(payload["absolute_deadline"]) in (float,int) and math.isfinite(payload["absolute_deadline"]),
            "Invalid absolute watchdog deadline")
    require(type(payload["source_members_sha256"]) is dict and payload["source_members_sha256"],
            "Missing watchdog source byte identities")
    for name, expected in payload["source_members_sha256"].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and re.fullmatch(r"[0-9a-f]{64}",expected),
                "Unsafe watchdog source member")
    require(all(type(payload[k]) is str and re.fullmatch(r"[0-9a-f]{64}",payload[k])
        for k in ("prepared_plan_fingerprint","evaluation_source_sha256","authorization_sha256")),
        "Watchdog source/plan/authorization binding missing")
    require({"sircus/isaac/candidate_v4_entry.py","sircus/isaac/candidate_v4_watchdog.py",
        "sircus/isaac/python.sh"}.issubset(payload["source_members_sha256"]),
        "Watchdog omitted mandatory native/bootstrap source")
    fixed_command(payload)
    return payload


def _metadata_path(raw, local_path):
    require(type(raw) is str,"Metadata paths must be canonical strings")
    relative = PurePosixPath(raw)
    require(relative.is_absolute() and relative.is_relative_to("/workspace")
        and ".." not in relative.parts and str(relative) == raw,
        "Metadata path leaves the canonical existing mount")
    root,path = Path(local_path("/workspace")),Path(local_path(raw))
    require(path.is_absolute() and path.is_relative_to(root) and str(path.resolve()) == str(path)
        and not any(p.is_symlink() for p in (path,*path.parents) if p.is_relative_to(root)),
        "Linked or noncanonical metadata path")
    return path


def verify_metadata_snapshot(value, *, session, nonce, payload_raw_sha256, payload_fingerprint, output):
    """Authenticate only the two runtime receipts and complete progress metadata."""
    require(type(value) is dict and set(value) == {"schema","session","nonce","payload_raw_sha256",
        "payload_fingerprint","output","receipts","progress","fingerprint"}
        and value["schema"] == SNAPSHOT_SCHEMA and value["session"] == session and value["nonce"] == nonce
        and value["payload_raw_sha256"] == payload_raw_sha256 and value["payload_fingerprint"] == payload_fingerprint
        and value["output"] == output
        and value["fingerprint"] == canonical_hash({k:v for k,v in value.items() if k != "fingerprint"}),
        "Runtime metadata snapshot binding/fingerprint changed")
    receipts = value["receipts"]
    require(type(receipts) is dict and set(receipts) == set(RECEIPT_NAMES),"Runtime metadata receipt allowlist changed")
    for name,receipt in receipts.items():
        if receipt is None:continue
        require(type(receipt) is dict and receipt.get("schema") == SCHEMA and receipt.get("session") == session
            and receipt.get("payload_sha256") == payload_fingerprint
            and receipt.get("fingerprint") == canonical_hash({k:v for k,v in receipt.items() if k != "fingerprint"}),
            "Runtime receipt session/fingerprint/payload changed")
        for key in ("owner","native_owner"):
            owner = receipt.get(key)
            require(type(owner) is dict and type(owner.get("pid")) is int and owner["pid"] > 1
                and type(owner.get("start_token")) is str and bool(owner["start_token"])
                and owner.get("cookie") == session,"Runtime metadata owner identity changed")
    owner,done = (receipts[name] for name in RECEIPT_NAMES)
    if done is not None:
        require(owner is not None and same_process(done["owner"],owner["owner"])
            and done["native_owner"] == owner["native_owner"] and type(done.get("exit_code")) is int,
            "Runtime exit is not bound to the exact owned watchdog/native child")
    verify_progress(value["progress"],session)
    return value


def metadata_snapshot(payload_path, *, expected_payload_sha256, payload_fingerprint, session, nonce, local_path=Path):
    """Read fixed private metadata through the existing root container route.

    No process inspection, signalling, application import or writable operation.
    ``local_path`` is a test-only mount mapper; the CLI uses literal /workspace.
    The path must be the exact new-only watchdog input under its own output.
    """
    require(type(session) is str and re.fullmatch(r"[0-9a-f]{32}",session)
        and type(nonce) is str and re.fullmatch(r"[0-9a-f]{32}",nonce),"Invalid metadata session/nonce")
    require(all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}",v)
        for v in (expected_payload_sha256,payload_fingerprint)),"Exact metadata payload digests required")
    path = _metadata_path(payload_path,local_path)
    require(path.name == "watchdog-input.json" and path.is_file(),"Only the fixed watchdog input may select metadata")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_payload_sha256,"Raw metadata payload bytes changed")
    payload = validate_payload(strict_json(raw))
    require(payload["session"] == session and payload["fingerprint"] == payload_fingerprint
        and payload_path == str(PurePosixPath(payload["output"])/"watchdog-input.json"),
        "Metadata target differs from the bound payload/session/output")
    output = _metadata_path(payload["output"],local_path)
    require(output.is_dir(),"Bound metadata output is missing")
    # Read exit before owner: a newly published exit may never be paired with an
    # earlier missing-owner observation. Progress remains an authenticated prefix.
    receipts = {}
    for name in reversed(RECEIPT_NAMES):
        target = _metadata_path(str(PurePosixPath(payload["output"])/name),local_path)
        require(not target.exists() or target.is_file(),"Runtime metadata receipt must be a plain file")
        receipts[name] = strict_json(target.read_bytes()) if target.exists() else None
    value = signed(dict(schema=SNAPSHOT_SCHEMA,session=session,nonce=nonce,
        payload_raw_sha256=expected_payload_sha256,payload_fingerprint=payload_fingerprint,
        output=payload["output"],receipts=receipts,progress=read_progress(output,session)))
    return verify_metadata_snapshot(value,session=session,nonce=nonce,payload_raw_sha256=expected_payload_sha256,
        payload_fingerprint=payload_fingerprint,output=payload["output"])


def verify_source_inventory(payload,ops):
    """Complete source membership and nonsymlink topology, including additions."""
    source=_metadata_path(payload["source_root"],ops.local_path)
    require(source.is_dir(),"Evaluation source root is missing")
    for name,expected in payload["source_members_sha256"].items():
        raw=str(PurePosixPath(payload["source_root"])/name)
        path=_metadata_path(raw,ops.local_path)
        require(path.is_file() and sha(path)==expected,"Native source bytes or topology changed")
    source_dir=source/"sircus/isaac"
    members=sorted(source_dir.glob("*.py"))
    require(members and all(p.is_file() and not p.is_symlink() for p in members),
        "Evaluation source inventory/topology changed")
    actual={p.name:sha(p) for p in members}
    declared={PurePosixPath(k).name:v for k,v in payload["source_members_sha256"].items()
        if PurePosixPath(k).parent==PurePosixPath("sircus/isaac") and k.endswith(".py")}
    require(actual==declared and canonical_hash(actual)==payload["evaluation_source_sha256"],
        "Evaluation source inventory changed")


def run_guarded(payload, ops=None):
    """One future authorized evaluation start, independent of the host loop.

    Tests supply fabricated process operations. Calling this with live Linux
    operations requires the separately bound runtime authorization; import alone
    performs no process inspection, simulator startup or signal.
    """
    validate_payload(payload)
    ops = ops or LinuxProcessOps()
    output, session = ops.local_path(payload["output"]), payload["session"]
    require(output.is_dir() and not (output/"runtime-owner.json").exists()
        and not (output/"runtime-exit.json").exists(), "Watchdog cannot retry or replace evidence")
    start = ops.monotonic()
    require(0 < payload["absolute_deadline"]-start <= NATIVE_SECONDS,
            "Native absolute deadline expired or exceeds 50 minutes")
    manifest_path=_metadata_path(payload["manifest"],ops.local_path)
    require(manifest_path.is_file() and sha(manifest_path)==payload["manifest_sha256"],"Native manifest bytes changed")
    verify_source_inventory(payload,ops)
    ops.prepare()
    owner = None
    pid = None
    phase, phase_start, consumed, last_time = 0, start, 0, start
    last_resource_check = start-1
    diagnostics = DiagnosticLedger()
    counters = dict(evaluation_requests=0,evaluation_terminal=0,evaluation_releases=0)
    termination, reason, code = None, None, None
    try:
        environment = dict(os.environ, **{COOKIE:session, "CUDA_VISIBLE_DEVICES":"0",
            "SIRCUS_ISAAC_SOURCE":str(Path(payload["source_root"])/"sircus/isaac"),
            "PYTHONDONTWRITEBYTECODE":"1"})
        pid = ops.start(fixed_command(payload),environment)
        owner = ops.identity(pid)
        require(owner is not None and owner["cookie"] == session and owner["ppid"] == ops.own_pid(),
                "Owned child identity unavailable after sole start; never retry")
        watchdog = ops.identity(ops.own_pid())
        require(watchdog is not None and watchdog["cookie"] == session, "Watchdog session cookie missing")
        write_new_json(output/"runtime-owner.json", signed(dict(schema=SCHEMA, session=session,
            owner=watchdog, native_owner=owner, absolute_deadline=payload["absolute_deadline"],
            payload_sha256=payload["fingerprint"])))
        while True:
            now = ops.monotonic()
            require(math.isfinite(now) and now >= last_time, "Watchdog monotonic clock changed")
            last_time = now
            require(now <= payload["absolute_deadline"], "Independent absolute native wall limit exceeded")
            if now-last_resource_check >= 1:
                free = ops.free_bytes(output)
                require(type(free) is int and free >= MIN_FREE_BYTES,"Independent 6 GiB storage floor reached")
                output=_metadata_path(payload["output"],ops.local_path)
                require(output.is_dir(),"Evaluation output path changed")
                manifest_path=_metadata_path(payload["manifest"],ops.local_path)
                require(manifest_path.is_file() and sha(manifest_path)==payload["manifest_sha256"],
                        "Manifest bytes changed during native application")
                verify_source_inventory(payload,ops)
                last_resource_check = now
            events = read_progress(output,session)
            # Source checks and the real reader can overlap publication. Bound
            # this frozen event list by a clock sampled after its read, rather
            # than misclassifying a newly published receipt against stale time.
            observed_now = ops.monotonic()
            require(math.isfinite(observed_now) and observed_now >= last_time, "Watchdog monotonic clock changed")
            last_time = observed_now
            require(observed_now <= payload["absolute_deadline"], "Independent absolute native wall limit exceeded")
            for event in events[consumed:]:
                snapshot = event["snapshot"]
                require(type(snapshot) is dict and snapshot.get("phase") in PHASES,
                        "Malformed watchdog progress snapshot")
                next_phase = PHASES.index(snapshot["phase"])
                require(next_phase in (phase,phase+1), "Watchdog phase order changed")
                stamp = event["monotonic_s"]
                require(start <= stamp <= observed_now and stamp-phase_start <= dict(PHASE_LIMITS)[PHASES[phase]],
                        "Delayed or future phase receipt cannot extend a deadline")
                validate_snapshot(snapshot)
                if next_phase != phase:
                    if next_phase == PHASES.index("audit"):
                        require(counters["evaluation_requests"] == counters["evaluation_terminal"] == 210,
                                "Watchdog audit before complete TEST-only evaluation")
                    phase, phase_start = next_phase, stamp
                diagnostics.observe(snapshot["diagnostics"],snapshot["parking_resets"])
                require(phase == 0 or diagnostics.complete, "Watchdog startup diagnostics incomplete")
                require(phase != PHASES.index("audit") or snapshot["audit_cpu_only"] is True,
                        "Audit phase must report CPU-only execution")
                for key,previous in counters.items():
                    require(snapshot[key] >= previous,"Watchdog task counters moved backwards")
                require(snapshot["evaluation_requests"] <= 210
                    and snapshot["evaluation_terminal"] <= snapshot["evaluation_requests"]
                    and snapshot["evaluation_releases"] <= snapshot["evaluation_terminal"],
                    "Watchdog TEST-only task budget exceeded")
                require(phase >= PHASES.index("evaluation") or all(snapshot[k] == 0 for k in counters),
                        "Watchdog saw TEST work before evaluation")
                require(phase <= PHASES.index("evaluation") or all(snapshot[k] == counters[k] for k in counters),
                        "Watchdog saw new task work after evaluation")
                counters = {k:snapshot[k] for k in counters}
            consumed = len(events)
            # Receipt validation time counts toward the same original caps.
            now = ops.monotonic()
            require(math.isfinite(now) and now >= last_time, "Watchdog monotonic clock changed")
            last_time = now
            require(now <= payload["absolute_deadline"], "Independent absolute native wall limit exceeded")
            require(now-phase_start <= dict(PHASE_LIMITS)[PHASES[phase]], "Independent phase wall limit exceeded")
            code = ops.poll(pid)
            if code is not None:
                require(type(code) is int, "Invalid native child exit code")
                require(not ops.owned(pid,session), "Native child exited with owned workers still alive")
                require(code == 0 and phase == len(PHASES)-1 and events
                    and events[-1]["snapshot"].get("running") is False
                    and events[-1]["snapshot"].get("exit_code") == 0,
                    "Native child ended before complete audited-phase receipts")
                break
            current = ops.identity(pid)
            require(current is not None and current["start_token"] == owner["start_token"] and current["cookie"] == session,
                    "Native leader ownership changed")
            ops.sleep(.1)
    except BaseException as error:
        reason = str(error) or type(error).__name__
        try:
            cleanup_deadline = min(payload["absolute_deadline"],phase_start+dict(PHASE_LIMITS)[PHASES[phase]])
            termination = stop_owned(ops,owner,session,deadline=cleanup_deadline) if owner is not None else "unknown_child_identity"
        except BaseException as stop_error:
            termination = "termination_unconfirmed: "+type(stop_error).__name__
        code = 1
    result = signed(dict(schema=SCHEMA, session=session, owner=ops.identity(ops.own_pid()),native_owner=owner,
        payload_sha256=payload["fingerprint"],
        status="complete_audit_receipts_required" if reason is None else "failed_no_retry",
        exit_code=code, reason=reason, termination=termination,
        native_start_intents=1,native_process_starts=1 if pid is not None else None,
        native_application_starts=None,
        native_start_scope="One native entry process at most; simulator App starts require the application receipt.",
        training_requests=0,optimizer_steps=0,
        prepared_plan_fingerprint=payload["prepared_plan_fingerprint"],
        evaluation_source_sha256=payload["evaluation_source_sha256"],progress_events=consumed,
        phase=PHASES[phase], wall_s=ops.monotonic()-start,
        model_promoted=False, independent_audit_passed=False))
    write_new_json(output/"runtime-exit.json",result)
    return result


def inspect_watchdog(pid,session):
    require(type(pid) is int and pid > 1 and re.fullmatch(r"[0-9a-f]{32}",session),"Invalid watchdog inspection identity")
    current = proc_identity(pid)
    if current is None or current["cookie"] != session: return None
    try:
        argv = (Path("/proc")/str(pid)/"cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return None
    if str(Path(__file__).resolve()).encode() not in argv or b"--execute" not in argv:
        return None
    return current


def stop_watchdog(value):
    require(type(value) is dict and set(value) == {"pid","start_token","session"},"Invalid watchdog stop fields")
    current = inspect_watchdog(value["pid"],value["session"])
    if current is None: return dict(status="ownership_mismatch")
    if current["start_token"] != value["start_token"]: return dict(status="ownership_mismatch")
    try:descriptor = pidfd_open(value["pid"],0)
    except ProcessLookupError:return dict(status="already_exited")
    try:
        if not same_process(inspect_watchdog(value["pid"],value["session"]),current):
            return dict(status="ownership_mismatch")
        pidfd_send_signal(descriptor,signal.SIGTERM,None,0)
        return dict(status="stop_requested")
    except ProcessLookupError:
        return dict(status="already_exited")
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--execute",type=Path)
    action.add_argument("--inspect",type=int)
    action.add_argument("--stop")
    action.add_argument("--snapshot")
    parser.add_argument("--session")
    parser.add_argument("--expected-payload-sha256")
    parser.add_argument("--payload-fingerprint")
    parser.add_argument("--nonce")
    args = parser.parse_args()
    if args.snapshot is not None:
        require(os.environ.get(COOKIE) == args.session,"Metadata snapshot session cookie missing")
        print(json.dumps(metadata_snapshot(args.snapshot,expected_payload_sha256=args.expected_payload_sha256,
            payload_fingerprint=args.payload_fingerprint,session=args.session,nonce=args.nonce),allow_nan=False))
        return 0
    if args.inspect is not None:
        require(os.environ.get(COOKIE) == args.session,"Watchdog inspection session missing")
        print(json.dumps(inspect_watchdog(args.inspect,args.session))); return 0
    if args.stop is not None:
        value = strict_json(args.stop)
        require(os.environ.get(COOKIE) == value.get("session"),"Watchdog stop session missing")
        print(json.dumps(stop_watchdog(value))); return 0
    payload = strict_json(args.execute.read_bytes())
    require(os.environ.get(COOKIE) == payload.get("session"), "Explicit watchdog session binding missing")
    def interrupted(signum,frame):
        raise KeyboardInterrupt("Owned watchdog received stop request")
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    return run_guarded(payload)["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
