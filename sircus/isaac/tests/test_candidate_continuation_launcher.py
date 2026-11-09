"""Bounded-launch contract tests using fake clocks/processes, never native work."""

import copy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_continuation_launcher as launcher


def bindings():
    value = {name:"a"*64 for name in launcher.BINDING_NAMES}
    value.update(launcher.STARTUP_HELPERS)
    value["parent_checkpoint_sha256"] = launcher.PARENT_CHECKPOINT_SHA256
    value["learning_protocol_sha256"] = launcher.LEARNING_SHA256
    return value


def diagnostic_records(rims=1):
    records = []
    for index, angle in enumerate(launcher.RIM_ANGLES[:rims]):
        records.append(dict(kind="rim_ray", azimuth_deg=angle, placements=35,
            active_probes=35, physics_steps=48, wall_s=.25, process_cpu_s=.5,
            initialized_states_sha256="b"*64, passed=index == rims-1))
    for kind in launcher.CONTACT_KINDS:
        records.append(dict(kind=kind, placements=35, active_probes=35,
            physics_steps=240, wall_s=.5, process_cpu_s=1.,
            initialized_states_sha256="c"*64, passed=True))
    records.append(dict(kind="foreign_cell_collision_filter", placements=35,
        active_probes=1, physics_steps=240, wall_s=.5, process_cpu_s=1.,
        initialized_states_sha256="d"*64, passed=True))
    return records


def progress(phase, *, diagnostics=None, train=0, releases=0, assessment=0,
             assessment_releases=0, running=True, exit_code=None, cpu_only=True):
    return dict(phase=phase, diagnostics=diagnostics if diagnostics is not None else diagnostic_records(),
        parking_resets=2, train_requests=train, train_releases=releases,
        assessment_requests=assessment, assessment_releases=assessment_releases,
        running=running, exit_code=exit_code, audit_cpu_only=cpu_only)


def successful_progress(*, requests=1400, releases=1400):
    return [progress("initialization", diagnostics=[]), progress("imitation"),
        progress("train"), progress("train", train=requests, releases=releases),
        progress("assessment", train=requests, releases=releases),
        progress("assessment", train=requests, releases=releases, assessment=420,
                 assessment_releases=411),
        progress("audit", train=requests, releases=releases, assessment=420,
                 assessment_releases=411),
        progress("audit", train=requests, releases=releases, assessment=420,
                 assessment_releases=411, running=False, exit_code=0)]


class FakeRuntime:
    mock_only = True

    def __init__(self, plan, snapshots=None):
        self.plan = plan
        self.t = 0.
        self.snapshots = copy.deepcopy(snapshots if snapshots is not None else successful_progress())
        self.last_snapshot = self.snapshots[-1]
        self.source = dict(plan.bindings)
        self.free = launcher.MIN_FREE_BYTES
        self.locked = False
        self.lock_busy_until = 0.
        self.gpu_busy_until = 0.
        self.gpu_queries = []
        self.started = []
        self.signaled = []
        self.owner = launcher.ChildIdentity(321, "process-birth-7", plan.session)
        self.current_identity = self.owner
        self.after_sleep = None
        self.after_gpu = None
        self.after_poll = None
        self.after_start = None
        self.termination_response = "stopped"
        self.released_locks = 0
        self.start_error = None

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        if self.after_sleep:
            self.after_sleep(self)

    def source_snapshot(self, plan):
        return copy.deepcopy(self.source)

    def free_bytes(self, output_parent):
        return self.free

    def acquire_scene_lock(self, lock, session):
        assert lock == self.plan.scene_lock and session == self.plan.session
        assert not self.locked
        if self.t < self.lock_busy_until:
            return False
        self.locked = True
        return True

    def release_scene_lock(self, lock, session):
        assert lock == self.plan.scene_lock and session == self.plan.session
        assert self.locked
        self.locked = False
        self.released_locks += 1

    def gpu_state(self, gpu_uuid):
        self.gpu_queries.append(gpu_uuid)
        result = dict(gpu_uuid=gpu_uuid, container=launcher.CONTAINER, container_running=True,
            container_gpu_uuids=[launcher.GPU_UUID], compute_pids=[999] if self.t < self.gpu_busy_until else [])
        if self.after_gpu:
            self.after_gpu(self, result)
        return result

    def start_native(self, plan, absolute_deadline):
        assert self.locked
        assert (Path(plan.output)/"launch-plan.json").exists()
        start = json.loads((Path(plan.output)/"launch-000001.json").read_text())
        assert start["kind"] == "start_intent"
        self.started.append((plan.to_dict(), absolute_deadline))
        if self.start_error:
            raise self.start_error
        if self.after_start:
            self.after_start(self)
        return self.owner

    def identity(self, pid):
        assert pid == self.owner.pid
        return self.current_identity

    def poll(self, owner):
        assert owner == self.owner
        result = self.snapshots.pop(0) if self.snapshots else self.last_snapshot
        if self.after_poll:
            self.after_poll(self, result)
        return copy.deepcopy(result)

    def terminate_owned(self, owner):
        # Final signal boundary checks again, including a possible fake PID reuse.
        assert owner == self.owner
        if self.current_identity != owner:
            return "ownership_mismatch"
        if self.termination_response in ("stopped", "already_exited"):
            if self.termination_response == "stopped":
                self.signaled.append(owner)
        return self.termination_response


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.plan = launcher.LaunchPlan.create(self.root/"new-run",
            protected_roots=[self.root/"frozen", self.root/"source"],
            scene_lock=self.root/"existing.scene.lock", input_sha256=bindings(), session="1"*32)

    def tearDown(self):
        self.temporary.cleanup()

    def result(self):
        return json.loads((Path(self.plan.output)/"launch-result.json").read_text())

    def test_fixed_plan_round_trip_is_immutable_and_denies_permission(self):
        self.assertEqual(launcher.LaunchPlan.from_dict(self.plan.to_dict()), self.plan)
        with self.assertRaises(FrozenInstanceError):
            self.plan.session = "2"*32
        value = self.plan.to_dict()
        self.assertFalse(value["execution_authorized"])
        self.assertFalse(value["production_backend_implemented"])
        for field, changed in (("native_timeout_s", 10801), ("gpu_uuid", "GPU-4070"),
                ("max_application_starts", 2), ("execution_authorized", True)):
            bad = copy.deepcopy(value)
            bad[field] = changed
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "bounds or fingerprint"):
                launcher.LaunchPlan.from_dict(bad)

    def test_protected_sources_and_relative_paths_cannot_be_output(self):
        for path in (self.root/"frozen"/"new", self.root/"source"/"new", self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                replace(self.plan, output=str(path))
        with self.assertRaisesRegex(ValueError, "absolute"):
            replace(self.plan, output="relative")

    def test_parent_learning_startup_and_missing_bindings_rejected(self):
        for field in ("parent_checkpoint_sha256", "learning_protocol_sha256", "film_cylinder_farm.py"):
            value = bindings()
            value[field] = "0"*64
            with self.subTest(field=field), self.assertRaises(ValueError):
                launcher.LaunchPlan.create(self.root/"other", protected_roots=[self.root/"source"],
                    scene_lock=self.root/"lock", input_sha256=value)
        value = bindings()
        del value["native_binding_sha256"]
        with self.assertRaisesRegex(ValueError, "complete"):
            replace(self.plan, bindings=tuple(sorted(value.items())))

    def test_native_backend_is_explicitly_unavailable(self):
        runtime = FakeRuntime(self.plan)
        runtime.mock_only = False
        with self.assertRaisesRegex(ValueError, "mock-only"):
            launcher.supervise(self.plan, runtime)
        self.assertFalse(Path(self.plan.output).exists())
        self.assertFalse(runtime.started)

    def test_success_is_one_mock_start_pending_independent_audit(self):
        runtime = FakeRuntime(self.plan)
        result = launcher.supervise(self.plan, runtime)
        self.assertEqual(result["status"], "mock_complete_independent_audit_required")
        self.assertFalse(result["independent_audit_passed"])
        self.assertFalse(result["model_promoted"])
        self.assertEqual(len(runtime.started), 1)
        self.assertEqual(runtime.started[0][1], launcher.NATIVE_SECONDS)
        self.assertEqual(set(runtime.gpu_queries), {launcher.GPU_UUID})
        self.assertFalse(runtime.signaled)
        self.assertFalse(runtime.locked)
        self.assertEqual(sum(result["phase_wall_s"].values()), result["native_wall_s"])
        self.assertEqual(result["diagnostics"]["cohorts"], 7)
        self.assertEqual(result["diagnostics"]["initialized_trajectories"], 245)
        self.assertEqual(result["diagnostics"]["active_probes"], 211)
        self.assertEqual(result["diagnostics"]["parking_resets"], 2)
        self.assertEqual(result["diagnostics"]["task_requests"], 0)
        self.assertTrue(self.result()["mock_only"])

    def test_request_cap_with_rejections_also_allows_assessment(self):
        runtime = FakeRuntime(self.plan, successful_progress(requests=1750, releases=1333))
        result = launcher.supervise(self.plan, runtime)
        self.assertEqual(result["counters"]["train_requests"], 1750)
        self.assertEqual(result["counters"]["train_releases"], 1333)

    def test_existing_output_refuses_repeat_even_after_complete(self):
        runtime = FakeRuntime(self.plan)
        launcher.supervise(self.plan, runtime)
        before = (Path(self.plan.output)/"launch-result.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "untouched"):
            launcher.supervise(self.plan, runtime)
        self.assertEqual((Path(self.plan.output)/"launch-result.json").read_bytes(), before)
        self.assertEqual(len(runtime.started), 1)

    def test_symlink_output_never_overwrites_target(self):
        target = self.root/"target"
        target.mkdir()
        Path(self.plan.output).symlink_to(target)
        with self.assertRaisesRegex(ValueError, "canonical|untouched|fingerprint"):
            launcher.supervise(self.plan, FakeRuntime(self.plan))
        self.assertEqual(list(target.iterdir()), [])

    def test_busy_foreign_gpu_waits_without_signaling_it(self):
        runtime = FakeRuntime(self.plan)
        runtime.gpu_busy_until = 3
        result = launcher.supervise(self.plan, runtime)
        self.assertEqual(result["queue_wall_s"], 3)
        self.assertFalse(runtime.signaled)
        self.assertEqual(len(runtime.started), 1)
        self.assertGreaterEqual(runtime.released_locks, 4)

    def test_shared_lock_waits_without_even_querying_gpu(self):
        runtime = FakeRuntime(self.plan)
        runtime.lock_busy_until = 3
        result = launcher.supervise(self.plan, runtime)
        self.assertEqual(result["queue_wall_s"], 3)
        self.assertEqual(runtime.gpu_queries, [launcher.GPU_UUID])

    def test_wait_timeout_never_starts_or_signals_anything(self):
        runtime = FakeRuntime(self.plan)
        runtime.gpu_busy_until = 10000
        runtime.after_sleep = lambda rt: setattr(rt, "t", 7200)
        with self.assertRaisesRegex(ValueError, "two hours"):
            launcher.supervise(self.plan, runtime)
        self.assertFalse(runtime.started)
        self.assertFalse(runtime.signaled)
        self.assertFalse(runtime.locked)
        self.assertEqual(self.result()["termination"], "not_started")

    def test_rechecks_source_after_wait_and_after_lock_gpu_query(self):
        for where in ("after_sleep", "after_gpu"):
            with self.subTest(where=where):
                plan = replace(self.plan, output=str(self.root/where))
                runtime = FakeRuntime(plan)
                if where == "after_sleep":
                    runtime.gpu_busy_until = 1
                    runtime.after_sleep = lambda rt: rt.source.update(producer_source_sha256="e"*64)
                else:
                    runtime.after_gpu = lambda rt, state: rt.source.update(producer_source_sha256="e"*64)
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    launcher.supervise(plan, runtime)
                self.assertFalse(runtime.started)
                self.assertFalse(runtime.signaled)
                self.assertFalse(runtime.locked)

    def test_wrong_or_stopped_container_never_restarted(self):
        for field, value in (("container_running", False), ("container", "another"),
                             ("container_gpu_uuids", [launcher.GPU_UUID, "GPU-4070"])):
            with self.subTest(field=field):
                plan = replace(self.plan, output=str(self.root/field))
                runtime = FakeRuntime(plan)
                runtime.after_gpu = lambda rt, state: state.update({field:value})
                with self.assertRaisesRegex(ValueError, "already be running"):
                    launcher.supervise(plan, runtime)
                self.assertFalse(runtime.started)
                self.assertFalse(runtime.signaled)

    def test_storage_floor_before_start_and_during_run_preserves_files(self):
        for during in (False, True):
            plan = replace(self.plan, output=str(self.root/("during" if during else "before")))
            runtime = FakeRuntime(plan)
            if during:
                runtime.after_sleep = lambda rt: setattr(rt, "free", launcher.MIN_FREE_BYTES-1)
            else:
                runtime.free = launcher.MIN_FREE_BYTES-1
            with self.subTest(during=during), self.assertRaisesRegex(ValueError, "6 GiB"):
                launcher.supervise(plan, runtime)
            self.assertTrue((Path(plan.output)/"launch-plan.json").exists())
            self.assertEqual(len(runtime.started), int(during))
            self.assertEqual(len(runtime.signaled), int(during))
            self.assertFalse(runtime.locked)

    def test_midrun_source_change_stops_only_owned_child(self):
        runtime = FakeRuntime(self.plan)
        runtime.after_sleep = lambda rt: rt.source.update(train_bank_sha256="e"*64)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            launcher.supervise(self.plan, runtime)
        self.assertEqual(runtime.signaled, [runtime.owner])
        self.assertEqual(self.result()["native_application_starts"], 1)

    def test_phase_deadline_stops_owned_child_and_never_restarts(self):
        runtime = FakeRuntime(self.plan, [progress("initialization", diagnostics=[])])
        runtime.after_sleep = lambda rt: setattr(rt, "t", 601)
        with self.assertRaisesRegex(ValueError, "Phase wall limit.*initialization"):
            launcher.supervise(self.plan, runtime)
        self.assertEqual(runtime.signaled, [runtime.owner])
        self.assertEqual(len(runtime.started), 1)
        self.assertEqual(self.result()["termination"], "stopped")

    def test_callback_elapsed_time_cannot_evade_phase_or_absolute_cap(self):
        for absolute in (False, True):
            plan = replace(self.plan, output=str(self.root/("absolute" if absolute else "phase")))
            runtime = FakeRuntime(plan)
            runtime.after_poll = lambda rt, state: setattr(rt, "t", 10801 if absolute else 601)
            with self.subTest(absolute=absolute), self.assertRaisesRegex(ValueError, "wall limit"):
                launcher.supervise(plan, runtime)
            self.assertEqual(runtime.signaled, [runtime.owner])

    def test_each_phase_has_its_own_fixed_wall_bound(self):
        for phase, limit in launcher.PHASE_LIMITS:
            plan = replace(self.plan, output=str(self.root/f"limit-{phase}"))
            runtime = FakeRuntime(plan)
            runtime.observed_phase = None
            runtime.after_poll = lambda rt, state: setattr(rt, "observed_phase", state["phase"])
            def advance(rt):
                if rt.observed_phase == phase:
                    rt.t += limit+1
            runtime.after_sleep = advance
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, "wall limit"):
                launcher.supervise(plan, runtime)
            self.assertEqual(runtime.signaled, [runtime.owner])
            self.assertEqual(len(runtime.started), 1)

    def test_invalid_progress_is_preserved_with_nonfinite_tags(self):
        runtime = FakeRuntime(self.plan)
        runtime.snapshots[0]["diagnostics"] = diagnostic_records()
        runtime.snapshots[0]["diagnostics"][0]["wall_s"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            launcher.supervise(self.plan, runtime)
        raw = json.loads((Path(self.plan.output)/"launch-observation-000000.json").read_text())
        self.assertEqual(raw["observation"]["diagnostics"][0]["wall_s"],
                         {"nonfinite_numeric_repr":"nan"})
        self.assertEqual(self.result()["status"], "mock_failed")

    def test_startup_receipts_cannot_change_after_initialization(self):
        runtime = FakeRuntime(self.plan)
        runtime.snapshots[2]["diagnostics"][0]["wall_s"] += 1
        with self.assertRaisesRegex(ValueError, "changed after initialization"):
            launcher.supervise(self.plan, runtime)
        self.assertEqual(runtime.signaled, [runtime.owner])

    def test_pid_reuse_is_never_signaled_and_retains_scene_lease(self):
        runtime = FakeRuntime(self.plan)
        runtime.after_sleep = lambda rt: setattr(rt, "current_identity", replace(rt.owner, start_token="new-process"))
        with self.assertRaisesRegex(ValueError, "PID/start identity"):
            launcher.supervise(self.plan, runtime)
        self.assertFalse(runtime.signaled)
        self.assertTrue(runtime.locked)
        self.assertTrue(self.result()["scene_lock_retained"])
        self.assertEqual(self.result()["termination"], "ownership_mismatch")

    def test_atomic_signal_boundary_mismatch_does_not_claim_stopped(self):
        runtime = FakeRuntime(self.plan)
        runtime.termination_response = "ownership_mismatch"
        runtime.after_sleep = lambda rt: setattr(rt, "free", 0)
        with self.assertRaisesRegex(ValueError, "6 GiB"):
            launcher.supervise(self.plan, runtime)
        self.assertFalse(runtime.signaled)
        self.assertTrue(runtime.locked)
        self.assertTrue(self.result()["scene_lock_retained"])

    def test_failed_start_is_consumed_no_retry_unknown_owner_not_signaled(self):
        runtime = FakeRuntime(self.plan)
        runtime.start_error = RuntimeError("mock start interrupted after spawn")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            launcher.supervise(self.plan, runtime)
        self.assertEqual(len(runtime.started), 1)
        self.assertFalse(runtime.signaled)
        self.assertTrue(runtime.locked)
        self.assertEqual(self.result()["termination"], "unknown_child_identity")
        with self.assertRaisesRegex(ValueError, "untouched"):
            launcher.supervise(self.plan, runtime)
        self.assertEqual(len(runtime.started), 1)

    def test_keyboard_interrupt_preserves_receipt_and_stops_owned_child(self):
        runtime = FakeRuntime(self.plan)
        def interrupt(rt):
            raise KeyboardInterrupt()
        runtime.after_sleep = interrupt
        with self.assertRaises(KeyboardInterrupt):
            launcher.supervise(self.plan, runtime)
        self.assertEqual(runtime.signaled, [runtime.owner])
        self.assertFalse(runtime.locked)
        self.assertEqual(self.result()["exception_type"], "KeyboardInterrupt")
        self.assertEqual(self.result()["termination"], "stopped")

    def test_termination_callback_failure_preserves_original_error_and_lease(self):
        runtime = FakeRuntime(self.plan)
        runtime.after_sleep = lambda rt: setattr(rt, "free", 0)
        def failed_stop(owner):
            raise RuntimeError("mock termination failed")
        runtime.terminate_owned = failed_stop
        with self.assertRaisesRegex(ValueError, "6 GiB"):
            launcher.supervise(self.plan, runtime)
        self.assertTrue(runtime.locked)
        self.assertEqual(self.result()["termination"], "termination_failed")
        self.assertIn("mock termination failed", self.result()["termination_error"])
        self.assertFalse(runtime.signaled)

    def test_wrong_start_identity_refuses_foreign_termination(self):
        runtime = FakeRuntime(self.plan)
        runtime.owner = replace(runtime.owner, session="2"*32)
        with self.assertRaisesRegex(ValueError, "matching child identity"):
            launcher.supervise(self.plan, runtime)
        self.assertFalse(runtime.signaled)
        self.assertTrue(runtime.locked)

    def test_clock_nonfinite_or_backwards_stops_owned_run(self):
        for value in (-1., float("nan")):
            plan = replace(self.plan, output=str(self.root/("back" if value == -1 else "nan")))
            runtime = FakeRuntime(plan)
            runtime.after_sleep = lambda rt: setattr(rt, "t", value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "clock"):
                launcher.supervise(plan, runtime)
            self.assertEqual(runtime.signaled, [runtime.owner])

    def test_nonzero_exit_and_incomplete_phases_preserve_failure(self):
        for code in (0, 2):
            plan = replace(self.plan, output=str(self.root/f"exit-{code}"))
            runtime = FakeRuntime(plan, [progress("initialization", running=False, exit_code=code)])
            with self.subTest(code=code), self.assertRaises(ValueError):
                launcher.supervise(plan, runtime)
            self.assertEqual(json.loads((Path(plan.output)/"launch-result.json").read_text())["status"], "mock_failed")

    def test_phase_skip_cpu_audit_and_premature_assessment_fail(self):
        cases = [
            [progress("train")],
            [progress("imitation"), progress("train"), progress("assessment")],
            successful_progress()[:-2] + [progress("audit", train=1400, releases=1400,
                assessment=420, assessment_releases=411, cpu_only=False)],
        ]
        for index, snapshots in enumerate(cases):
            plan = replace(self.plan, output=str(self.root/f"bad-phase-{index}"))
            runtime = FakeRuntime(plan, snapshots)
            with self.subTest(index=index), self.assertRaises(ValueError):
                launcher.supervise(plan, runtime)
            self.assertEqual(runtime.signaled, [runtime.owner])

    def test_counter_overrun_regression_and_postfreeze_train_rejected(self):
        for index, bad in enumerate((
                progress("train", train=1751, releases=1400),
                progress("train", train=1400, releases=1401),
                progress("train", train=1399, releases=1399),
                progress("assessment", train=1401, releases=1400),
                progress("assessment", train=1400, releases=1400, assessment=421),
        )):
            plan = replace(self.plan, output=str(self.root/f"counter-{index}"))
            snapshots = successful_progress()[:4] + [bad]
            runtime = FakeRuntime(plan, snapshots)
            with self.subTest(index=index), self.assertRaises(ValueError):
                launcher.supervise(plan, runtime)
            self.assertEqual(runtime.signaled, [runtime.owner])


class DiagnosticTests(unittest.TestCase):
    def test_original_early_stop_or_full_eight_ray_bounds(self):
        for rims in (1, 8):
            ledger = launcher.DiagnosticLedger()
            ledger.observe(diagnostic_records(rims), 4)
            result = ledger.summary()
            self.assertTrue(result["complete"])
            self.assertEqual(result["cohorts"], rims+6)
            self.assertEqual(result["initialized_trajectories"], (rims+6)*35)
            self.assertEqual(result["physics_steps"], rims*48+6*240)
            self.assertEqual(result["parking_resets"], 4)

    def test_append_only_diagnostics_cannot_repeat_rewrite_or_shrink(self):
        for mode in ("repeat", "rewrite", "shrink", "parking"):
            ledger = launcher.DiagnosticLedger()
            records = diagnostic_records()
            ledger.observe(records[:1], 1)
            if mode == "repeat": records.insert(1, copy.deepcopy(records[0]))
            elif mode == "rewrite": records[0]["initialized_states_sha256"] = "e"*64
            elif mode == "shrink": records = []
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ledger.observe(records, 0 if mode == "parking" else 1)

    def test_extra_smoke_release_wrong_order_and_budget_tamper_rejected(self):
        for mode in ("smoke", "order", "steps", "placements", "isolation", "nonfinite", "boolcount"):
            ledger = launcher.DiagnosticLedger()
            records = diagnostic_records()
            if mode == "smoke": records.append({**records[-1], "kind":"smoke_release"})
            elif mode == "order": records[1], records[2] = records[2], records[1]
            elif mode == "steps": records[0]["physics_steps"] = 49
            elif mode == "placements": records[-1]["placements"] = 1
            elif mode == "isolation": records[-1]["active_probes"] = 35
            elif mode == "nonfinite": records[0]["wall_s"] = float("nan")
            elif mode == "boolcount": records[-1]["active_probes"] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ledger.observe(records, 0)

    def test_exhausted_or_failed_contacts_never_allow_task_start(self):
        exhausted = diagnostic_records(8)[:8]
        exhausted[-1]["passed"] = False
        contact = diagnostic_records()
        contact[1]["passed"] = False
        isolation = diagnostic_records()
        isolation[-1]["passed"] = False
        for records in (exhausted, contact, isolation):
            with self.assertRaises(ValueError):
                launcher.DiagnosticLedger().observe(records, 0)

    def test_full_startup_is_required_before_imitation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = launcher.LaunchPlan.create(root/"new", protected_roots=[root/"frozen"],
                scene_lock=root/"lock", input_sha256=bindings(), session="3"*32)
            runtime = FakeRuntime(plan, [progress("imitation", diagnostics=[])])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                launcher.supervise(plan, runtime)
            self.assertEqual(runtime.signaled, [runtime.owner])


if __name__ == "__main__":
    unittest.main()
