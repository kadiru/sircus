"""Fake clocks, resources and counters only; no process or simulator starts."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from location_comparison_bounds import (
    ARMS, ARM_WALL_S, ASSESSMENT_CAP, Bounds, BoundsExceeded, MIN_FREE_BYTES,
    SHARED_WALL_S, SnapshotWatchdog, TRAIN_CAP,
)


def plan():
    return dict(fingerprint="a" * 64, protocol=dict(
        proposed_train_requests_per_arm=TRAIN_CAP,
        proposed_total_train_requests=2 * TRAIN_CAP,
        proposed_wall_seconds_per_training_arm=ARM_WALL_S,
        proposed_shared_assessment_wall_seconds=SHARED_WALL_S,
        minimum_free_storage_bytes=MIN_FREE_BYTES,
        total_assessment_requests=ASSESSMENT_CAP))


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def advance(self, amount):
        self.now += amount


class Fixture:
    def __init__(self):
        self.clock = Clock()
        self.free = MIN_FREE_BYTES + 1000
        self.resources = lambda: dict(free_storage_bytes=self.free)
        self.bounds = Bounds(plan(), self.clock, self.resources)

    def batches(self, kind, count, arm=None):
        while count:
            n = min(35, count)
            reserved = self.bounds.reserve(kind, n, arm)
            self.bounds.complete(reserved)
            count -= n

    def training_arm(self, arm):
        self.bounds.enter_phase(arm)
        self.batches("train", TRAIN_CAP, arm)
        self.batches("assessment", 140, arm)


class CooperativeBoundsTests(unittest.TestCase):
    def test_exact_complete_counts_do_not_spend_assessment_as_training(self):
        f = Fixture()
        f.batches("assessment", 35)
        for arm in ARMS:
            f.training_arm(arm)
        f.bounds.enter_phase("shared")
        f.batches("assessment", 285)
        result = f.bounds.snapshot()
        self.assertEqual(result["train_requested"], dict.fromkeys(ARMS, 8960))
        self.assertEqual(result["train_completed"], dict.fromkeys(ARMS, 8960))
        self.assertEqual(result["assessment_requested"], 600)
        self.assertEqual(result["assessment_completed"], 600)
        self.assertIsNone(result["pending"])

    def test_shared_budget_accumulates_across_arms_and_cannot_reset(self):
        f = Fixture()
        f.clock.advance(2000)
        f.bounds.enter_phase("original_usable")
        f.batches("train", 8960, "original_usable")
        f.clock.advance(100)
        f.bounds.enter_phase("broader")
        f.batches("train", 8960, "broader")
        f.clock.advance(100)
        f.bounds.enter_phase("shared")
        f.clock.advance(1599)
        self.assertTrue(f.bounds.check())
        f.bounds.enter_phase("shared")  # Idempotent, never fresh credit.
        f.clock.advance(1)
        with self.assertRaisesRegex(BoundsExceeded, "shared"):
            f.bounds.check()
        self.assertEqual(f.bounds.snapshot()["phase_elapsed_s"]["shared"], 3600)

    def test_arm_monitoring_and_setup_consume_arm_clock(self):
        f = Fixture()
        f.bounds.enter_phase("original_usable")
        f.clock.advance(28799)
        r = f.bounds.reserve("assessment", 35, "original_usable")
        f.clock.advance(1)
        with self.assertRaisesRegex(BoundsExceeded, "original_usable"):
            f.bounds.complete(r)
        state = f.bounds.snapshot()
        self.assertEqual(state["assessment_completed"], 35)
        self.assertEqual(state["phase_elapsed_s"]["original_usable"], 28800)

    def test_incomplete_reserved_work_is_never_refunded_or_retried(self):
        f = Fixture()
        f.bounds.enter_phase("original_usable")
        reserved = f.bounds.reserve("train", 35, "original_usable")
        f.bounds.fail("fake execution failed")
        state = f.bounds.snapshot()
        self.assertEqual(state["train_requested"]["original_usable"], 35)
        self.assertEqual(state["train_completed"]["original_usable"], 0)
        self.assertEqual(state["pending"], reserved)
        with self.assertRaises(BoundsExceeded):
            f.bounds.reserve("train", 35, "original_usable")
        with self.assertRaises(BoundsExceeded):
            f.bounds.complete(reserved)

    def test_request_cap_and_duplicate_completion_fail_closed(self):
        f = Fixture()
        f.bounds.enter_phase("original_usable")
        f.batches("train", 8960, "original_usable")
        with self.assertRaisesRegex(BoundsExceeded, "cap"):
            f.bounds.reserve("train", 1, "original_usable")
        self.assertEqual(f.bounds.snapshot()["train_requested"]["original_usable"], 8960)
        f = Fixture()
        r = f.bounds.reserve("assessment", 1)
        f.bounds.complete(r)
        with self.assertRaisesRegex(BoundsExceeded, "completion"):
            f.bounds.complete(r)
        self.assertEqual(f.bounds.snapshot()["assessment_completed"], 1)

    def test_storage_failure_latches_even_when_storage_recovers(self):
        f = Fixture()
        f.free = MIN_FREE_BYTES - 1
        with self.assertRaisesRegex(BoundsExceeded, "storage"):
            f.bounds.reserve("assessment", 1)
        f.free = MIN_FREE_BYTES + 1000
        with self.assertRaises(BoundsExceeded):
            f.bounds.check()
        self.assertEqual(f.bounds.snapshot()["assessment_requested"], 0)

    def test_clock_regression_and_missing_resources_fail_closed(self):
        f = Fixture()
        f.clock.advance(-1)
        with self.assertRaisesRegex(BoundsExceeded, "clock"):
            f.bounds.check()
        with self.assertRaises(ValueError):
            Bounds(plan(), resources=None)
        with self.assertRaisesRegex(BoundsExceeded, "storage"):
            Bounds(plan(), Clock(), lambda: {})

    def test_phase_change_cannot_hide_partial_requests_or_monitor_time(self):
        f = Fixture()
        f.bounds.enter_phase("original_usable")
        f.batches("train", 35, "original_usable")
        with self.assertRaisesRegex(BoundsExceeded, "incomplete"):
            f.bounds.enter_phase("broader")
        f = Fixture()
        f.bounds.enter_phase("original_usable")
        with self.assertRaisesRegex(BoundsExceeded, "wrong phase"):
            f.bounds.reserve("assessment", 35)

    def test_modified_plan_cannot_widen_any_production_limit(self):
        for key in ("proposed_train_requests_per_arm", "proposed_total_train_requests",
                    "proposed_wall_seconds_per_training_arm", "proposed_shared_assessment_wall_seconds",
                    "total_assessment_requests", "minimum_free_storage_bytes"):
            value = plan()
            value["protocol"][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                Bounds(value, Clock(), lambda: dict(free_storage_bytes=MIN_FREE_BYTES))


class IndependentObserverTests(unittest.TestCase):
    def observer(self, f):
        return SnapshotWatchdog(plan(), started_monotonic=f.bounds.started,
                                clock=f.clock, resources=f.resources)

    def test_stalled_child_expires_without_another_child_callback(self):
        f = Fixture()
        f.bounds.enter_phase("original_usable")
        f.bounds.reserve("train", 35, "original_usable")
        observer = self.observer(f)
        self.assertTrue(observer.check(f.bounds.snapshot()))
        f.clock.advance(28800)
        with self.assertRaisesRegex(BoundsExceeded, "deadline"):
            observer.check()
        self.assertEqual(f.bounds.snapshot()["train_requested"]["original_usable"], 35)

    def test_missing_initial_snapshot_still_has_a_deadline(self):
        f = Fixture()
        observer = self.observer(f)
        f.clock.advance(3600)
        with self.assertRaisesRegex(BoundsExceeded, "initial"):
            observer.check()

    def test_counter_rollback_and_phase_clock_reset_are_rejected(self):
        for defect in ("counter", "clock"):
            f = Fixture()
            f.bounds.enter_phase("original_usable")
            f.clock.advance(100)
            f.batches("train", 35, "original_usable")
            observer = self.observer(f)
            initial = f.bounds.snapshot()
            observer.check(initial)
            changed = copy.deepcopy(initial)
            if defect == "counter":
                changed["train_requested"]["original_usable"] = 0
                changed["train_completed"]["original_usable"] = 0
            else:
                changed["phase_elapsed_s"]["original_usable"] -= 10
                changed["phase_elapsed_s"]["shared"] += 10
            with self.subTest(defect=defect), self.assertRaises(BoundsExceeded):
                observer.check(changed)

    def test_snapshot_tampering_and_independent_resources_fail(self):
        for mutate in (lambda s: s.update(comparison_fingerprint="b" * 64),
                       lambda s: s.update(started_monotonic=101),
                       lambda s: s.update(observed_monotonic=200),
                       lambda s: s.update(assessment_requested=601),
                       lambda s: s.update(phase_index=3, active_phase="shared")):
            f = Fixture()
            snapshot = f.bounds.snapshot()
            mutate(snapshot)
            with self.subTest(mutate=mutate), self.assertRaises(BoundsExceeded):
                self.observer(f).check(snapshot)
        f = Fixture()
        observer = self.observer(f)
        observer.check(f.bounds.snapshot())
        f.free = MIN_FREE_BYTES - 1
        with self.assertRaisesRegex(BoundsExceeded, "storage"):
            observer.check()


if __name__ == "__main__":
    unittest.main()
