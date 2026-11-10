"""Mock-only bounded orchestration for the separately proposed V4 TEST study.

This module has no simulator, policy loader, optimizer, subprocess or native
launcher. Its callbacks must explicitly be mocks. Wall/storage checks operate
at callback boundaries, so they do not preempt a blocked callback. A separately
reviewed production watchdog and native integration are still required before
any actual execution. Failed directories are evidence, never resume inputs.
"""

import copy
import math
from pathlib import Path
import re
import resource
import shutil
import sys
import time

from candidate_continuation_journal import Journal, write_new_json, _sync_directory
from candidate_continuation_ledger import canonical_hash

SCHEMA = "sircus-v4-test-evaluation-result-v1"
RESULT_FIELDS = frozenset(("request_id", "released", "decline_reason", "captured",
    "clean", "source_sha256", "return"))
FORBIDDEN_CALLBACKS = ("train", "imitation_step", "start_ppo", "optimizer", "learning_engine")


def require(value, message):
    if not value:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _usage():
    value = resource.getrusage(resource.RUSAGE_SELF)
    return dict(cpu_s=time.process_time(),
        lifetime_peak_rss_mib=value.ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024))


def _frozen(plan):
    return dict(final_checkpoint_sha256=plan["final_checkpoint_sha256"],
        parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
        bindings=copy.deepcopy(plan["bindings"]),
        model_hashes=copy.deepcopy(plan["original_plan"]["model_hashes"]))


def _output_path(output, protected_roots):
    path = Path(output)
    require(path.is_absolute() and path == path.resolve(), "Output must be canonical and absolute")
    require(type(protected_roots) in (tuple, list) and bool(protected_roots),
        "Original evidence/source roots must be explicitly protected")
    for raw in protected_roots:
        root = Path(raw)
        require(root.is_absolute() and root == root.resolve(), "Protected roots must be canonical")
        require(not path.is_relative_to(root) and not root.is_relative_to(path),
            "New evaluation output overlaps protected evidence or source")
    require(not path.exists() and not path.is_symlink(), "Output already exists; no retry or resume")
    require(path.parent.is_dir() and not any(p.is_symlink() for p in (path.parent, *path.parents)),
        "Output parent must exist without linked ancestors")
    return path


class _Costs:
    def __init__(self, clock, usage):
        self.clock, self.usage = clock, usage
        self.rows, self.current = {}, None

    def begin(self, phase):
        require(self.current is None and phase not in self.rows, "Repeated or overlapping cost phase")
        self.current = (phase, self.clock(), self.usage())

    def finish(self):
        if self.current is None:
            return
        phase, started, before = self.current
        now, after = self.clock(), self.usage()
        row = dict(wall_s=now-started, process_cpu_s=after["cpu_s"]-before["cpu_s"],
            lifetime_peak_rss_mib=after["lifetime_peak_rss_mib"])
        require(all(_finite(v) and v >= 0 for v in row.values()), "Invalid mock resource measurement")
        self.rows[phase], self.current = row, None

    def receipt(self):
        return dict(phases=copy.deepcopy(self.rows),
            scope="Mock callback phase wall and this Python process CPU only; no native measurements. RSS is process-lifetime high-water, not a summed phase peak. No GPU cost is measured. Callback-boundary checks cannot preempt blocked work.")


class MockEvaluation:
    """Single-use synthetic harness; no production execution is implemented."""

    def __init__(self, output, plan, runtime, *, protected_roots, clock=time.monotonic,
                 free_bytes=None, usage=_usage):
        from candidate_v4_evaluation import validate_prepared, protocol
        self.plan = validate_prepared(plan)
        require(self.plan.get("synthetic") is True and self.plan.get("execution_authorized") is False
            and self.plan.get("release_authorized") is False
            and self.plan.get("native_execution_ready") is False,
            "Only explicitly synthetic mock preparation may run; native execution is disabled")
        require(getattr(runtime, "synthetic", None) is True and getattr(runtime, "mock_only", None) is True,
            "Runtime must explicitly be mock-only and synthetic")
        require(not any(hasattr(runtime, name) for name in FORBIDDEN_CALLBACKS),
            "Evaluation runtime cannot expose a training or optimizer API")
        require(all(callable(getattr(runtime, name, None)) for name in
            ("initialize", "execute", "verify_frozen", "close", "audit")), "Incomplete mock runtime")
        self.output = _output_path(output, protected_roots)
        self.runtime, self.clock = runtime, clock
        self.free_bytes = free_bytes or (lambda: shutil.disk_usage(self.output.parent).free)
        self.bounds = protocol()
        require(self.bounds["phases"] == dict(initialization=600, evaluation=1800, audit=600)
            and self.bounds["wall_limit_s"] == 3000 and self.bounds["min_free_bytes"] == 6*1024**3,
            "Evaluation bounds changed")
        self.frozen = _frozen(self.plan)
        self.plan_hash = canonical_hash(self.plan)
        self.started = self.last_clock = clock()
        require(_finite(self.started), "Invalid initial monotonic clock")
        self.phase, self.phase_started = "initialization", self.started
        self.counts = dict(requested=0, terminal=0, released=0, rejected=0,
            captures=0, clean_captures=0, unresolved=0)
        self.pending = None
        self.used = self.closed = self.failed = self.finished = False
        self.initializations = 0
        self.costs = _Costs(clock, usage)
        self._guard()
        self.output.mkdir(); _sync_directory(self.output.parent)
        write_new_json(self.output/"evaluation-plan.json", self.plan)
        self.journal = Journal(self.output/"journal")
        self.journal.append("start", dict(plan_fingerprint=self.plan["fingerprint"],
            synthetic=True, mock_only=True, frozen=copy.deepcopy(self.frozen)))

    def _guard(self):
        require(not self.failed and not self.finished, "Evaluation is closed; no retry or resume")
        require(canonical_hash(self.plan) == self.plan_hash, "Prepared evaluation plan changed")
        now = self.clock()
        require(_finite(now) and now >= self.last_clock, "Monotonic clock moved backwards or is invalid")
        self.last_clock = now
        require(now-self.started <= self.bounds["wall_limit_s"]
            and now-self.phase_started <= self.bounds["phases"][self.phase],
            "Evaluation wall or phase limit reached")
        free = self.free_bytes()
        require(type(free) is int and free >= self.bounds["min_free_bytes"], "Original 6 GiB storage floor reached")
        actual = self.runtime.verify_frozen(copy.deepcopy(self.plan))
        require(actual == self.frozen, "Frozen checkpoint, source or input bytes changed")

    def _phase(self, name):
        self._guard(); self.costs.finish()
        self.phase = name; self.phase_started = self.clock()
        self.costs.begin(name); self._guard()

    def _close(self):
        if not self.closed:
            self.closed = True
            self.runtime.close()

    def _record(self, cohort, rows):
        require(self.pending is not None and type(rows) in (list, tuple) and bool(rows),
            "Expected an ordered terminal outcome prefix")
        choices = self.pending
        require(len(rows) <= len(choices), "Outcome prefix exceeds its charged reservation")
        checked = []
        for choice, row in zip(choices, rows):
            require(type(row) is dict and set(row) == RESULT_FIELDS
                and row["request_id"] == choice["request_id"], "Outcome request identity or order changed")
            require(all(type(row[k]) is bool for k in ("released", "captured", "clean"))
                and (not row["clean"] or row["captured"]) and (not row["captured"] or row["released"]),
                "Outcome flags contradict release")
            reason = row["decline_reason"]
            require((row["released"] and reason is None) or (not row["released"]
                and type(reason) is str and bool(reason.strip())), "Invalid release or rejection reason")
            require(choice["enabled"] or (not row["released"] and reason == choice["decline_reason"]),
                "Empty candidate support was released or replaced")
            value = row["return"]
            require(_finite(value) and ((not row["released"] and value == -1.)
                or (row["captured"] and value == (1.6 if row["clean"] else 1.5))
                or (row["released"] and not row["captured"] and -.8 <= value <= 0.)),
                "Outcome reward changed")
            require(type(row["source_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}",row["source_sha256"]),
                "Missing outcome source digest")
            checked.append(copy.deepcopy(row))
        next_counts = copy.deepcopy(self.counts)
        for row in checked:
            next_counts["terminal"] += 1
            next_counts["released"] += int(row["released"])
            next_counts["rejected"] += int(not row["released"])
            next_counts["captures"] += int(row["captured"])
            next_counts["clean_captures"] += int(row["clean"])
        next_counts["unresolved"] = next_counts["requested"]-next_counts["terminal"]
        self.journal.append("outcomes", dict(cohort=cohort["index"], results=checked,
            requested=next_counts["requested"], terminal=next_counts["terminal"]))
        self.counts = next_counts
        self.pending = self.pending[len(checked):] or None

    def _failure(self, error):
        self.failed = True
        resource_error = None
        try: self.costs.finish()
        except BaseException as issue: resource_error = repr(issue)
        value = dict(schema=SCHEMA, status="mock_incomplete_no_retry", synthetic=True, mock_only=True,
            phase=self.phase, error=repr(error), counts=copy.deepcopy(self.counts),
            plan_fingerprint=self.plan["fingerprint"], frozen=copy.deepcopy(self.frozen),
            training_requests=0, optimizer_steps=0, native_requests_executed=0,
            native_application_starts=0, mock_initializations=self.initializations,
            model_promoted=False, execution_authorized=False, release_authorized=False,
            resources=self.costs.receipt(), resource_error=resource_error)
        try: self.journal.append("failure", copy.deepcopy(value))
        finally: write_new_json(self.output/"evaluation-failure.json", value)

    def run(self):
        require(not self.used and not self.failed and not self.finished, "Mock evaluation already used; no retry")
        self.used = True
        try:
            self.costs.begin("initialization"); self._guard()
            self.initializations += 1
            self.runtime.initialize(copy.deepcopy(self.plan)); self._guard()
            self._phase("evaluation")
            for cohort in self.plan["cohorts"]:
                self._guard()
                require(self.pending is None and self.counts["requested"]+35 <= 210,
                    "Cannot replace unresolved slots or exceed fixed cap")
                choices = copy.deepcopy(cohort["choices"])
                new_requested = self.counts["requested"]+35
                event = self.journal.append("reserve", dict(cohort=cohort["index"],
                    original_cohort=cohort["original_cohort"], choices=choices,
                    requested=new_requested, terminal=self.counts["terminal"],
                    plan_fingerprint=self.plan["fingerprint"]))
                self.pending = choices
                self.counts["requested"] = new_requested
                self.counts["unresolved"] = new_requested-self.counts["terminal"]
                reservation = dict(journal_head_sha256=event["fingerprint"], cohort=cohort["index"],
                    original_cohort=cohort["original_cohort"], requested=new_requested,
                    request_ids=[c["request_id"] for c in choices])
                try:
                    rows = self.runtime.execute(copy.deepcopy(cohort), copy.deepcopy(choices), reservation)
                except BaseException as error:
                    prefix = getattr(error,"terminal_prefix",None)
                    if prefix is not None:
                        self._record(cohort, prefix)
                    raise
                self._record(cohort, rows)
                require(self.pending is None, "Partial cohort preserved; no replacement or retry")
                self._guard()
            require(self.counts["requested"] == self.counts["terminal"] == 210
                and self.counts["unresolved"] == 0, "Incomplete TEST-only budget")
            self._close(); self._guard()
            self.journal.append("final", dict(plan_fingerprint=self.plan["fingerprint"],
                frozen=copy.deepcopy(self.frozen), counts=copy.deepcopy(self.counts),
                synthetic=True, mock_only=True, training_requests=0, optimizer_steps=0,
                native_requests_executed=0))
            summary = dict(schema=SCHEMA, status="mock_evaluation_complete_audit_pending",
                plan_fingerprint=self.plan["fingerprint"], journal_head_sha256=self.journal.head,
                frozen=copy.deepcopy(self.frozen), counts=copy.deepcopy(self.counts),
                synthetic=True, mock_only=True, training_requests=0, optimizer_steps=0,
                native_requests_executed=0)
            self._phase("audit")
            audit = self.runtime.audit(self.output,copy.deepcopy(self.plan),copy.deepcopy(summary))
            self._guard()
            require(type(audit) is dict and audit.get("status") == "passed_mock_only"
                and audit.get("synthetic") is True and audit.get("mock_only") is True
                and audit.get("full_native_audit") is False
                and audit.get("plan_fingerprint") == self.plan["fingerprint"]
                and audit.get("journal_head_sha256") == self.journal.head,
                "Independent mock-only audit did not pass; no native performance claim")
            self.costs.finish()
            write_new_json(self.output/"mock-audit.json", audit)
            result = dict(summary, resources=self.costs.receipt(), audit=audit,
                native_application_starts=0,mock_initializations=self.initializations,
                model_promoted=False,execution_authorized=False,release_authorized=False,
                scope="Synthetic software rehearsal only. Original V4 assessment stays incomplete. No real TEST result, native audit, model improvement, execution permission or promotion is established.")
            result["fingerprint"] = canonical_hash(result)
            write_new_json(self.output/"evaluation-result.json", result)
            self.finished = True
            return copy.deepcopy(result)
        except BaseException as error:
            try: self._close()
            except BaseException as close_error:
                error = RuntimeError(repr(error)+"; close failed: "+repr(close_error))
            self._failure(error)
            raise


def run_mock_evaluation(output, plan, runtime, *, protected_roots, **kwargs):
    """Construct one new synthetic run; no production/native mode is accepted."""
    return MockEvaluation(output,plan,runtime,protected_roots=protected_roots,**kwargs).run()
