"""Mock HTTP-state checks; no robot, native simulation or capture evidence."""
import time
import unittest
from sircus.mjlab.interactive_stalled_server import StalledPhysicsState
from sircus.mjlab.interface import RequestError


class FakePlanner:
    planning_bucket_bounds = [[.1, -.65], [1.24, .25]]

    def __init__(self, available=True):
        self.available = available
        self.releases = []
        self.closed = False

    def plan(self, xy, progress):
        preview = dict(bucket_xy_m=xy, status="feasible" if self.available else "unresolved",
                       configuration={"example": xy})
        return preview, preview if self.available else None

    def drop(self, preview, progress):
        self.releases.append(preview["bucket_xy_m"])
        return dict(preview, provenance={"substeps": 64})

    def close(self):
        self.closed = True


class StalledStateTests(unittest.TestCase):
    def wait(self, state, job, phase):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            snap = state.snapshot(job)
            if snap["state"] == phase:
                return snap
            time.sleep(.005)
        self.fail(str(state.snapshot(job)))

    def test_verified_hybrid_routing_retains_both_old_modes_and_single_release(self):
        classic, fast, hybrid = FakePlanner(), FakePlanner(), FakePlanner()
        state = StalledPhysicsState(classic, fast_planner=fast, hybrid_planner=hybrid,
                                    default_planner="hybrid")
        self.assertEqual([v["id"] for v in state.capabilities()["planners"]], ["hybrid", "fast", "classic"])
        self.assertEqual(state.capabilities()["default_planner"], "hybrid")
        try:
            for mode in ("hybrid", "fast", "classic"):
                payload = {"bucket_xy_m": [.6, -.4]}
                if mode != "hybrid": payload["planner"] = mode
                job = state.submit_plan(payload)["id"]
                self.wait(state, job, "ready")
                state.submit_drop(job, {})
                self.assertEqual(self.wait(state, job, "completed")["result"]["planner"], mode)
                with self.assertRaises(RequestError): state.submit_drop(job, {})
            self.assertEqual([len(p.releases) for p in (classic, fast, hybrid)], [1, 1, 1])
        finally:
            state.close()
        self.assertTrue(all(p.closed for p in (classic, fast, hybrid)))

    def test_unavailable_hybrid_cannot_be_selected_or_default(self):
        state = StalledPhysicsState(FakePlanner(), fast_planner=FakePlanner())
        self.assertEqual(state.capabilities()["default_planner"], "fast")
        with self.assertRaises(RequestError):
            state.submit_plan({"bucket_xy_m": [.6, -.4], "planner": "hybrid"})
        with self.assertRaises(ValueError):
            StalledPhysicsState(FakePlanner(), default_planner="hybrid")
        mismatch = FakePlanner(); mismatch.planning_bucket_bounds = [[0, 0], [1, 1]]
        with self.assertRaises(ValueError):
            StalledPhysicsState(FakePlanner(), hybrid_planner=mismatch)
        state.close()

    def test_failed_hybrid_never_silently_replans_or_drops_with_old_methods(self):
        classic, fast, hybrid = FakePlanner(), FakePlanner(), FakePlanner(False)
        state = StalledPhysicsState(classic, fast_planner=fast, hybrid_planner=hybrid,
                                    default_planner="hybrid")
        try:
            job = state.submit_plan({"bucket_xy_m": [.6, -.4]})["id"]
            self.wait(state, job, "no_setup")
            with self.assertRaises(RequestError): state.submit_drop(job, {})
            self.assertEqual(classic.releases + fast.releases + hybrid.releases, [])
            job = state.submit_plan({"bucket_xy_m": [.6, -.4], "planner": "fast"})["id"]
            self.wait(state, job, "ready")
            self.assertEqual(state.snapshot(job)["planner"], "fast")
        finally:
            state.close()


if __name__ == "__main__": unittest.main()
