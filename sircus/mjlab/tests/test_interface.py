"""Mock-only operator interface checks; never launch GPU jobs or robot clients."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from sircus.mjlab.interface import (
    DEFAULT_SCENE,
    InterfaceState,
    RequestError,
    handler_for,
    planner_ready,
    search_ready,
    study_ready,
    validate_request,
)


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.report_path = Path(self.temp.name) / "report.json"
        self.request = {
            "bucket_xy": [0.65, -0.4],
            "mode": "policy",
            "request_id": "test-request",
        }

    def tearDown(self):
        self.temp.cleanup()

    def completed(self):
        self.report_path.write_text(
            json.dumps({"status": "completed", "frozen_evaluation": {"trials": 10}})
        )

    def await_result(self, state):
        deadline = time.monotonic() + 2
        while state.snapshot()["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertNotEqual(state.snapshot()["status"], "running")
        return state.snapshot()

    def test_ready_requires_completed_frozen_evaluation(self):
        for report in (
            None,
            {},
            {"status": "running", "frozen_evaluation": {"trials": 1}},
            {"status": "completed"},
            {"status": "completed", "frozen_evaluation": {}},
            {"status": "completed", "frozen_evaluation": []},
        ):
            self.assertFalse(study_ready(report))
        self.assertTrue(
            study_ready({"status": "completed", "frozen_evaluation": {"trials": 1}})
        )

    def test_corrected_search_readiness_does_not_unlock_policy(self):
        report = {
            "status": "running",
            "planner": {"ready": True, "version": "sircus-corrected-sweep-planner-v1"},
        }
        self.report_path.write_text(json.dumps(report))
        self.assertTrue(planner_ready(report))
        self.assertTrue(search_ready(report))
        self.assertFalse(study_ready(report))
        calls = []

        def solve(request):
            calls.append(request)
            return {
                "method": "sweep_planner",
                "success": False,
                "status_native": "no_verified_solution_within_budget",
            }

        state = InterfaceState(self.report_path, solver=solve)
        public = state.report()
        self.assertTrue(public["search_ready"])
        self.assertFalse(public["interface_ready"])
        self.assertFalse(public["policy_ready"])
        with self.assertRaises(RequestError) as ctx:
            state.submit(self.request)
        self.assertEqual(ctx.exception.status, 423)
        state.submit({**self.request, "mode": "search", "search_budget": 32})
        result = self.await_result(state)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(calls[0]["search_budget"], 32)
        self.assertFalse(result["result"]["success"])

    def test_invalid_planner_capability_does_not_fall_back_to_legacy_search(self):
        ready = {"status": "completed", "frozen_evaluation": {"trials": 1}}
        self.assertTrue(search_ready(ready))
        for capability in (
            None,
            {},
            {"ready": True, "version": "other"},
            {"ready": "true", "version": "sircus-corrected-sweep-planner-v1"},
        ):
            self.assertFalse(search_ready({**ready, "planner": capability}))
        self.assertTrue(study_ready({**ready, "planner": {}}))

    def test_search_budget_is_bounded_and_not_a_policy_control(self):
        for budget in (16, 64, 128):
            request = validate_request(
                {**self.request, "mode": "search", "search_budget": budget},
                DEFAULT_SCENE,
            )
            self.assertEqual(request["search_budget"], budget)
        for budget in (True, 15, 129, 32.0, "64", float("nan")):
            with self.assertRaises(RequestError):
                validate_request(
                    {**self.request, "mode": "search", "search_budget": budget},
                    DEFAULT_SCENE,
                )
        with self.assertRaises(RequestError):
            validate_request({**self.request, "search_budget": 64}, DEFAULT_SCENE)

    def test_missing_study_never_calls_solver(self):
        calls = []
        state = InterfaceState(
            self.report_path, solver=lambda request: calls.append(request)
        )
        with self.assertRaises(RequestError) as ctx:
            state.submit(self.request)
        self.assertEqual(ctx.exception.status, 423)
        self.assertEqual(calls, [])

    def test_corrupted_report_fails_closed(self):
        for text in ("not JSON", "[]", "null"):
            self.report_path.write_text(text)
            state = InterfaceState(self.report_path)
            self.assertFalse(state.report()["interface_ready"])

    def test_public_report_omits_raw_test_trajectories_only(self):
        source = {
            "status": "completed",
            "frozen_evaluation": {"trials": 5, "captures": 2},
            "coverage": {"cells": [{"id": "a", "captures": 1}]},
            "learning_curve": [{"update": 1, "captures": 1}],
            "test_rows": [{"trajectory": [[0, 0, 0]]}],
            "bank_evaluation": {
                "status": "completed",
                "rows": [{"trajectory": [[1, 2, 3]]}],
                "coverage": [{"bucket_xy_m": [0.5, -0.3], "captures": 1}],
                "comparison": {"bank_selected": {"captures": 1, "attempts": 2}},
            },
        }
        self.report_path.write_text(json.dumps(source))
        public = InterfaceState(self.report_path).report()
        self.assertNotIn("test_rows", public)
        self.assertNotIn("rows", public["bank_evaluation"])
        self.assertEqual(
            public["bank_evaluation"]["coverage"], source["bank_evaluation"]["coverage"]
        )
        self.assertEqual(
            public["bank_evaluation"]["comparison"],
            source["bank_evaluation"]["comparison"],
        )
        self.assertIn(
            "rows", json.loads(self.report_path.read_text())["bank_evaluation"]
        )
        self.assertEqual(public["coverage"], source["coverage"])
        self.assertEqual(public["learning_curve"], source["learning_curve"])
        self.assertEqual(public["frozen_evaluation"], source["frozen_evaluation"])
        self.assertTrue(public["interface_ready"])
        self.assertIn("test_rows", json.loads(self.report_path.read_text()))
        self.assertEqual(DEFAULT_SCENE["board"]["size"], [0.2, 0.3, 0.005])

    def test_missing_solver_rejected(self):
        self.completed()
        with self.assertRaises(RequestError) as ctx:
            InterfaceState(self.report_path).submit(self.request)
        self.assertEqual(ctx.exception.status, 503)

    def test_position_locks_preserve_exact_values(self):
        payload = {
            **self.request,
            "constraints": {
                "board_position": [0.6, -0.1, 0.4],
                "dropper_position": [0.6, -0.1, 0.7],
            },
        }
        validated = validate_request(payload, DEFAULT_SCENE)
        self.assertEqual(validated["constraints"], payload["constraints"])
        for values in ([0.6, -0.1, 0.3], [0.6, -0.1, 0.4375]):
            payload["constraints"]["dropper_position"] = values
            with self.assertRaises(RequestError):
                validate_request(payload, DEFAULT_SCENE)

    def test_invalid_coordinates_and_unknown_controls_rejected(self):
        for xy in (
            [float("nan"), 0],
            [float("inf"), 0],
            [True, 0],
            [2, 0],
            [0],
            [".6", 0],
            None,
        ):
            with self.assertRaises(RequestError):
                validate_request({**self.request, "bucket_xy": xy}, DEFAULT_SCENE)
        for update in (
            {"mode": "physical"},
            {"robot": "sfhack10"},
            {"constraints": {"q10": [0] * 6}},
            {"constraints": {"board_position": [0, 0]}},
        ):
            with self.assertRaises(RequestError):
                validate_request({**self.request, **update}, DEFAULT_SCENE)

    def test_single_job_and_explicit_result_method(self):
        self.completed()
        entered, release = threading.Event(), threading.Event()

        def solver(request):
            entered.set()
            release.wait(2)
            return {"success": False, "method": "frozen_policy", "trajectory": []}

        state = InterfaceState(self.report_path, solver=solver)
        result = state.submit(self.request)
        self.assertEqual(result["job_id"], "test-request")
        self.assertTrue(entered.wait(1))
        with self.assertRaises(RequestError) as ctx:
            state.submit({**self.request, "request_id": "another"})
        self.assertEqual(ctx.exception.status, 409)
        release.set()
        result = self.await_result(state)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["method"], "frozen_policy")
        self.assertFalse(result["result"]["success"])

    def test_search_cannot_masquerade_as_policy(self):
        self.completed()
        state = InterfaceState(
            self.report_path,
            solver=lambda request: {"method": "search", "success": True},
        )
        state.submit(self.request)
        self.assertEqual(self.await_result(state)["status"], "error")
        state.submit({**self.request, "mode": "search"})
        self.assertEqual(self.await_result(state)["status"], "completed")

    def test_corrected_search_cannot_masquerade_as_policy(self):
        self.completed()
        for method in ("sweep_planner", "corrected_sweep"):
            state = InterfaceState(
                self.report_path,
                solver=lambda request: {"method": method, "success": True},
            )
            state.submit(self.request)
            self.assertEqual(self.await_result(state)["status"], "error")

    def test_wrong_request_and_nan_result_rejected(self):
        self.completed()
        for result in ({"request_id": "someone-else"}, {"value": float("nan")}):
            state = InterfaceState(self.report_path, solver=lambda request: result)
            state.submit(self.request)
            self.assertEqual(self.await_result(state)["status"], "error")

    def test_local_http_contract_static_allowlist_and_origin(self):
        state = InterfaceState(self.report_path)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        root = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(root + "/api/study") as response:
                self.assertFalse(json.load(response)["interface_ready"])
            with urlopen(root + "/") as response:
                self.assertIn(b"SIMULATION ONLY", response.read())
            for path in ("/../../README.md", "/api/robot", "/vendor/../../README.md"):
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(root + path)
                self.assertEqual(ctx.exception.code, 404)
                ctx.exception.close()
            request = Request(
                root + "/api/solve",
                json.dumps(self.request).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://example.invalid",
                },
            )
            with self.assertRaises(HTTPError) as ctx:
                urlopen(request)
            self.assertEqual(ctx.exception.code, 403)
            ctx.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
