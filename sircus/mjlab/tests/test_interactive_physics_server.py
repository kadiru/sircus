"""Check that UI requests cannot release a stale or previously used plan."""

import threading
import time
import unittest

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
