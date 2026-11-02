"""Local-only study-gated browser interface for simulated two-arm shots.

No robot clients, cloud configuration, credentials, or background training are
loaded here. The numerical solver is an explicit injected callback and is only
imported after a completed frozen-policy evaluation is available.
"""

import argparse
import importlib
import json
import math
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

WEB = Path(__file__).with_name("web")
VENDOR = Path(__file__).resolve().parents[1] / "simulation" / "web" / "vendor"
DEFAULT_SCENE = {
    "version": "sircus-mjlab-interface-v1",
    "units": "m",
    "simulation_only": True,
    "table": {"center": [0.67, -0.25, -0.025], "size": [1.8, 1.25, 0.05]},
    "bases": [[0, 0, 0], [1.33731, 0, 0]],
    "board": {"size": [0.2, 0.3, 0.005], "center": [0.65, 0, 0.4]},
    "ball": {"radius": 0.0375},
    "bucket": {
        "xy": [0.65, -0.4],
        "inside_top_diameter": 0.177,
        "inside_bottom_diameter": 0.15,
        "outside_top_diameter": 0.2,
        "outside_bottom_diameter": 0.158,
        "reported_height": 0.175,
        "height_interpretation": "unconfirmed; simulated interior depth assumed",
        "bottom_thickness": 0.008,
    },
    "bucket_bounds": {"x": [-0.13, 1.47], "y": [-0.775, 0.275]},
    "release_center": [0.65, 0, 0.6875],
    "pose_display": "Schematic until native link positions are supplied",
}


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _read_json(path, fallback):
    if path is None or not Path(path).is_file():
        return fallback
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return fallback


def study_ready(report):
    """A report alone cannot enable inference before frozen evaluation exists."""
    return (
        isinstance(report, dict)
        and report.get("status") == "completed"
        and isinstance(report.get("frozen_evaluation"), dict)
        and bool(report["frozen_evaluation"])
        and (
            not report.get("bank_evaluation")
            or (isinstance(report["bank_evaluation"], dict) and report["bank_evaluation"].get("status") == "completed")
        )
    )


def _vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise RequestError(f"{label} must contain {length} finite numbers")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in value
    ):
        raise RequestError(f"{label} must contain {length} finite numbers")
    return [float(x) for x in value]


def validate_request(payload, scene):
    if not isinstance(payload, dict):
        raise RequestError("Expected a JSON object")
    if set(payload) - {"mode", "bucket_xy", "constraints", "request_id"}:
        raise RequestError("Unknown request fields")
    mode = payload.get("mode", "policy")
    if mode not in ("policy", "search"):
        raise RequestError("Mode must be policy or search")
    xy = _vector(payload.get("bucket_xy"), 2, "bucket_xy")
    bounds = scene.get("bucket_bounds", DEFAULT_SCENE["bucket_bounds"])
    for key, value in zip(("x", "y"), xy):
        if not bounds[key][0] <= value <= bounds[key][1]:
            raise RequestError(f"Bucket {key} lies outside the table bounds")
    constraints = payload.get("constraints", {})
    if not isinstance(constraints, dict) or set(constraints) - {
        "board_position",
        "dropper_position",
    }:
        raise RequestError("Unknown arm constraints")
    clean = {}
    for name in ("board_position", "dropper_position"):
        value = constraints.get(name)
        clean[name] = None if value is None else _vector(value, 3, name)
        if value is not None and any(abs(x) > 3 for x in clean[name]):
            raise RequestError(f"{name} lies outside the supported work volume")
    board, dropper = clean["board_position"], clean["dropper_position"]
    if board is not None and dropper is not None:
        radius = scene.get("ball", {}).get("radius", 0.0375)
        if dropper[2] <= board[2] + radius:
            raise RequestError("Locked release center must be above the board")
    request_id = payload.get("request_id", str(uuid.uuid4()))
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise RequestError("Invalid request_id")
    return dict(mode=mode, bucket_xy=xy, constraints=clean, request_id=request_id)


class InterfaceState:
    def __init__(self, study_report=None, scene_json=None, solver=None):
        self.study_report = study_report
        self.scene_json = scene_json
        self.solver = solver
        self.lock = threading.Lock()
        self.job = {"status": "idle", "result": None, "error": None}

    def report(self):
        report = _read_json(self.study_report, {"status": "not_started"})
        if not isinstance(report, dict):
            report = {"status": "invalid_report"}
        # Full frozen trajectories remain on disk; the browser needs summaries only.
        public = {key: value for key, value in report.items() if key != "test_rows"}
        if isinstance(public.get("bank_evaluation"), dict):
            public["bank_evaluation"] = {
                key: value
                for key, value in public["bank_evaluation"].items()
                if key != "rows"
            }
        return {**public, "interface_ready": study_ready(report)}

    def scene(self):
        supplied = _read_json(self.scene_json, {})
        if not isinstance(supplied, dict):
            supplied = {}
        return {**DEFAULT_SCENE, **supplied, "simulation_only": True}

    def snapshot(self):
        with self.lock:
            return dict(self.job)

    def submit(self, payload):
        if not study_ready(self.report()):
            raise RequestError(
                "Manual solves are locked until the learning study and frozen evaluation finish",
                423,
            )
        request = validate_request(payload, self.scene())
        if self.solver is None:
            raise RequestError("No compatible numerical solver is configured", 503)
        with self.lock:
            if self.job["status"] == "running":
                raise RequestError("A solve is already running", 409)
            self.job = {
                "job_id": request["request_id"],
                "status": "running",
                "mode": request["mode"],
                "result": None,
                "error": None,
            }
        threading.Thread(target=self._solve, args=(request,), daemon=True).start()
        return {"job_id": request["request_id"], "status": "running"}

    def _solve(self, request):
        try:
            solver = self.solver
            if isinstance(solver, str):
                module, name = solver.rsplit(":", 1)
                solver = getattr(importlib.import_module(module), name)
            result = solver(request)
            if not isinstance(result, dict):
                raise TypeError("Solver must return a result object")
            if result.get("request_id", request["request_id"]) != request["request_id"]:
                raise ValueError("Solver returned another request's result")
            if request["mode"] == "policy" and result.get("method") in {
                "search",
                "refinement",
                "search_refinement",
                "catalogue",
                "catalog_lookup",
                "cached",
                "dataset_lookup",
            }:
                raise ValueError(
                    "Search or lookup cannot be returned as policy inference"
                )
            result = {**result, "request_id": request["request_id"]}
            json.dumps(result, allow_nan=False)
            with self.lock:
                self.job.update(status="completed", result=result)
        except Exception as exc:
            with self.lock:
                self.job.update(status="error", error=f"{type(exc).__name__}: {exc}")


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/study":
                return self._json(200, state.report())
            if path == "/api/scene":
                return self._json(200, state.scene())
            if path in ("/api/solution", "/api/status"):
                return self._json(200, state.snapshot())
            routes = {
                "/": (WEB / "index.html", "text/html"),
                "/index.html": (WEB / "index.html", "text/html"),
                "/app.js": (WEB / "app.js", "text/javascript"),
                "/state.mjs": (WEB / "state.mjs", "text/javascript"),
                "/style.css": (WEB / "style.css", "text/css"),
                "/vendor/three.module.js": (
                    VENDOR / "three.module.js",
                    "text/javascript",
                ),
                "/vendor/OrbitControls.js": (
                    VENDOR / "OrbitControls.js",
                    "text/javascript",
                ),
            }
            entry = routes.get(path)
            if entry is None or not entry[0].is_file():
                return self._json(404, {"error": "Not found"})
            data = entry[0].read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", entry[1])
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if urlsplit(self.path).path != "/api/solve":
                return self._json(404, {"error": "Not found"})
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self._json(403, {"error": "Cross-origin requests are disabled"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._json(415, {"error": "Expected application/json"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise RequestError("Request is empty or too large", 413)
                payload = json.loads(self.rfile.read(size))
                return self._json(202, state.submit(payload))
            except RequestError as exc:
                return self._json(exc.status, {"error": str(exc)})
            except (ValueError, UnicodeDecodeError):
                return self._json(400, {"error": "Invalid JSON request"})

        def log_message(self, format, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-report", type=Path, required=True)
    parser.add_argument("--scene-json", type=Path)
    parser.add_argument(
        "--solver", help="Explicit Python module:function, never a robot adapter"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"]
    )
    parser.add_argument("--port", type=int, default=18781)
    args = parser.parse_args()
    state = InterfaceState(args.study_report, args.scene_json, args.solver)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(state))
    print(f"Sircus mjlab interface: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
