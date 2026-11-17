"""Check that UI requests cannot release a stale or previously used plan."""

import threading
import time
import unittest
from unittest.mock import patch

from sircus.mjlab.interface import RequestError
from sircus.mjlab.interactive_physics_server import InteractivePhysicsState


class FakePlanner:
    planning_bucket_bounds = [[0.1, -0.65], [1.24, 0.25]]

    def __init__(self):
        self.releases = []

    def plan(self, xy, progress):
        preview = dict(bucket_xy_m=xy, status="feasible", configuration={"example": xy})
        return preview, preview

    def drop(self, preview, progress):
        self.releases.append(preview["bucket_xy_m"])
        return dict(preview, provenance={"substeps": 64})


class InteractivePhysicsServerTests(unittest.TestCase):
    def wait(self, state, job, phase):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            snap = state.snapshot(job)
            if snap["state"] == phase:
                return snap
            time.sleep(.005)
        self.fail(f"Job never reached {phase}: {state.snapshot(job)}")

    def test_explicit_modes_default_and_drop_routing(self):
        classic, fast = FakePlanner(), FakePlanner()
        legacy = InteractivePhysicsState(classic)
        self.assertEqual(legacy.capabilities()["default_planner"], "classic")
        with self.assertRaises(RequestError):
            legacy.submit_plan(dict(bucket_xy_m=[.4, -.3], planner="fast"))
        state = InteractivePhysicsState(classic, fast_planner=fast)
        self.assertEqual(state.capabilities()["default_planner"], "fast")
        for mode, explicit in (("fast", False), ("classic", True), ("fast", True)):
            payload = dict(bucket_xy_m=[.4, -.3])
            if explicit:
                payload["planner"] = mode
            job = state.submit_plan(payload)["id"]
            ready = self.wait(state, job, "ready")
            self.assertEqual(ready["planner"], mode)
            self.assertEqual(ready["preview"]["planner"], mode)
            self.assertGreaterEqual(ready["planning_elapsed_s"], 0.)
            state.submit_drop(job, {})
            done = self.wait(state, job, "completed")
            self.assertEqual(done["result"]["planner"], mode)
            self.assertEqual(done["result"]["timing"], ready["preview"]["timing"])
        self.assertEqual(len(fast.releases), 2)
        self.assertEqual(len(classic.releases), 1)

    def test_fast_no_setup_does_not_auto_fallback_and_can_choose_classic(self):
        class Unresolved(FakePlanner):
            def plan(self, xy, progress):
                return dict(bucket_xy_m=xy, status="unresolved", reason="No checked candidate"), None
        classic = FakePlanner()
        state = InteractivePhysicsState(classic, fast_planner=Unresolved())
        job = state.submit_plan(dict(bucket_xy_m=[.4, -.3]))["id"]
        self.wait(state, job, "no_setup")
        with self.assertRaises(RequestError):
            state.submit_drop(job, {})
        self.assertEqual(classic.releases, [])
        next_job = state.submit_plan(dict(bucket_xy_m=[.4, -.3], planner="classic"))["id"]
        self.wait(state, next_job, "ready")
        with self.assertRaises(RequestError) as stale:
            state.snapshot(job)
        self.assertEqual(stale.exception.status, 404)

    def test_native_timeout_does_not_change_planning_duration(self):
        class Busy(FakePlanner):
            def drop(self, prepared, progress):
                time.sleep(.03)
                raise TimeoutError("GPU occupied")
        state = InteractivePhysicsState(Busy())
        job = state.submit_plan(dict(bucket_xy_m=[.4, -.3]))["id"]
        ready = self.wait(state, job, "ready")
        with patch("sircus.mjlab.interactive_physics_server.logging.exception"):
            state.submit_drop(job, {})
            failed = self.wait(state, job, "failed")
        self.assertEqual(failed["planning_elapsed_s"], ready["planning_elapsed_s"])
        with self.assertRaises(RequestError):
            state.submit_drop(job, {})

    def test_active_planning_blocks_mode_change_and_close_closes_worker(self):
        entered, release = threading.Event(), threading.Event()
        class Waiting(FakePlanner):
            closed = False
            def plan(self, xy, progress):
                entered.set()
                release.wait(2)
                return super().plan(xy, progress)
            def close(self):
                self.closed = True
                release.set()
        fast = Waiting()
        state = InteractivePhysicsState(FakePlanner(), fast_planner=fast)
        job = state.submit_plan(dict(bucket_xy_m=[.4, -.3]))["id"]
        self.assertTrue(entered.wait(1))
        self.assertGreater(state.snapshot(job)["planning_elapsed_s"], 0.)
        with self.assertRaises(RequestError) as busy:
            state.submit_plan(dict(bucket_xy_m=[.5, -.3], planner="classic"))
        self.assertEqual(busy.exception.status, 409)
        closer = threading.Thread(target=state.close)
        closer.start()
        time.sleep(.01)
        self.assertTrue(closer.is_alive())
        self.assertFalse(fast.closed)
        release.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        self.assertTrue(fast.closed)
        with self.assertRaises(RequestError) as closed:
            state.submit_plan(dict(bucket_xy_m=[.5, -.3]))
        self.assertEqual(closed.exception.status, 503)

    def test_replaced_plan_and_concurrent_double_click_can_release_only_latest_once(self):
        planner = FakePlanner()
        state = InteractivePhysicsState(planner)

        def wait(job, phase):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if state.snapshot(job)["state"] == phase:
                    return
                time.sleep(0.005)
            self.fail(f"Job never reached {phase}")

        old = state.submit_plan({"bucket_xy_m": [0.2, -0.3]})["id"]
        wait(old, "ready")
        xy = [0.8, -0.3]
        current = state.submit_plan({"bucket_xy_m": xy})["id"]
        wait(current, "ready")
        with self.assertRaises(RequestError) as stale:
            state.submit_drop(old, {})
        self.assertEqual(stale.exception.status, 404)

        barrier = threading.Barrier(2)
        statuses = []

        def click():
            barrier.wait()
            try:
                state.submit_drop(current, {})
                statuses.append(202)
            except RequestError as exc:
                statuses.append(exc.status)

        threads = [threading.Thread(target=click) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wait(current, "completed")
        self.assertCountEqual(statuses, [202, 409])
        self.assertEqual(planner.releases, [xy])


if __name__ == "__main__":
    unittest.main()
