"""Serve the mjlab workcell with a frozen physics planner and explicit one-drop control."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json
import logging
from pathlib import Path
import signal
import threading
import time
import uuid
from urllib.parse import urlsplit

from .interface import InterfaceState, RequestError, WEB, _vector, handler_for

API = "/api/interactive-physics"


class InteractivePhysicsState:
    """Allow one active job and consume its exact planned command at most once."""

    def __init__(self, planner, *, fast_planner=None, default_planner=None):
        self.planner = planner
        self.planners = {"classic": planner}
        if fast_planner is not None:
            if fast_planner.planning_bucket_bounds != planner.planning_bucket_bounds:
                raise ValueError("Planner bounds must match")
            self.planners["fast"] = fast_planner
        self.default_planner = default_planner or ("fast" if fast_planner is not None else "classic")
        if self.default_planner not in self.planners:
            raise ValueError("Default planner is unavailable")
        lower, upper = planner.planning_bucket_bounds
        self.bounds = {"x": [lower[0], upper[0]], "y": [lower[1], upper[1]]}
        self.lock = threading.Lock()
        self.job = None
        self.prepared = None
        self.plan_started = None
        self.closed = False
        self.active_thread = None

    def capabilities(self):
        labels = {"fast": "Fast checked planner", "classic": "Original planner"}
        return {"enabled": True, "bucket_bounds": self.bounds, "default_planner": self.default_planner,
                "planners": [{"id": name, "label": labels[name]} for name in ("fast", "classic")
                             if name in self.planners]}

    def _current(self, job_id):
        if self.job is None or self.job["id"] != job_id:
            raise RequestError("This plan has expired. Plan a shot for the current bucket position.", 404)
        return self.job

    def snapshot(self, job_id):
        with self.lock:
            job = self._current(job_id).copy()
            if job["state"] in {"queued", "planning"}:
                job["planning_elapsed_s"] = time.perf_counter() - self.plan_started
            return job

    def submit_plan(self, payload):
        if (not isinstance(payload, dict) or "bucket_xy_m" not in payload
                or set(payload) - {"bucket_xy_m", "planner"}):
            raise RequestError("Expected bucket_xy_m with the bucket center in metres")
        mode = payload.get("planner", self.default_planner)
        if not isinstance(mode, str) or mode not in self.planners:
            raise RequestError("Choose an available planner")
        try:
            xy = _vector(payload["bucket_xy_m"], 2, "bucket_xy_m")
        except OverflowError as exc:
            raise RequestError("Bucket coordinates must be finite") from exc
        if any(not limits[0] <= value <= limits[1] for value, limits in zip(xy, self.bounds.values(), strict=True)):
            raise RequestError("Place the whole bucket on the tabletop")
        with self.lock:
            if self.closed:
                raise RequestError("The planner service is closing", 503)
            if self.job and self.job["state"] in {"queued", "planning", "running"}:
                raise RequestError("A shot is already being planned or simulated", 409)
            self.prepared = None
            self.plan_started = time.perf_counter()
            self.job = dict(id=uuid.uuid4().hex, state="queued", bucket_xy_m=xy, planner=mode,
                            planning_elapsed_s=0., message="Checking the two arm poses")
            submitted = self.job.copy()
            self.active_thread = threading.Thread(target=self._plan, args=(xy, mode), daemon=True)
            self.active_thread.start()
            return submitted

    def _progress(self, progress):
        with self.lock:
            self.job["progress"] = progress
            message = progress.get("message")
            if message:
                self.job["message"] = message

    def _plan(self, xy, mode):
        with self.lock:
            self.job["state"] = "planning"
        try:
            preview, prepared = self.planners[mode].plan(xy, self._progress)
            if preview["bucket_xy_m"] != xy or (prepared is not None) != (preview["status"] == "feasible"):
                raise ValueError("Planner returned a preview for another setup")
            json.dumps(preview, allow_nan=False)
            with self.lock:
                if self.closed:
                    return
                elapsed = time.perf_counter() - self.plan_started
                # Do not mutate the persisted preview consumed by native release.
                preview = dict(preview, planner=mode, timing=dict(preview.get("timing", {}), planning_wall_s=elapsed))
                self.job["planning_elapsed_s"] = elapsed
                self.prepared = prepared
                if prepared is None:
                    self.job.update(state="no_setup", preview=preview,
                                    message=preview.get("reason") or "No feasible setup found for this bucket position")
                else:
                    self.job.update(state="ready", preview=preview, message="Both arm poses are ready to preview")
        except Exception:
            self._fail("Planning failed. Plan a new shot to try again.")

    def submit_drop(self, job_id, payload):
        if payload != {}:
            raise RequestError("Run the previewed shot with an empty JSON object")
        with self.lock:
            job = self._current(job_id)
            if self.closed:
                raise RequestError("The planner service is closing", 503)
            if job["state"] != "ready" or self.prepared is None:
                raise RequestError("This plan is unavailable or its drop has already been requested", 409)
            prepared = self.prepared
            self.prepared = None
            job.update(state="running", message="Starting one simulated drop", progress={})
            submitted = job.copy()
            self.active_thread = threading.Thread(target=self._drop, args=(prepared, job["preview"], job["planner"]), daemon=True)
            self.active_thread.start()
            return submitted

    def _drop(self, prepared, preview, mode):
        try:
            result = self.planners[mode].drop(prepared, self._progress)
            if (result["bucket_xy_m"] != preview["bucket_xy_m"]
                    or result["configuration"] != preview["configuration"]
                    or result["provenance"]["substeps"] != 64):
                raise ValueError("Native result differs from the previewed plan")
            json.dumps(result, allow_nan=False)
            result = dict(result, planner=mode, timing=preview["timing"])
            with self.lock:
                self.job.update(state="completed", result=result, message="Drop complete")
        except TimeoutError:
            self._fail("The 4090 did not become available. Plan a new shot to try again.")
        except Exception:
            self._fail("The simulation could not finish this drop. Plan a new shot to try again.")

    def _fail(self, message):
        logging.exception(message)
        with self.lock:
            self.prepared = None
            if self.job["state"] in {"queued", "planning"}:
                self.job["planning_elapsed_s"] = time.perf_counter() - self.plan_started
            self.job.update(state="failed", message=message)

    def close(self):
        with self.lock:
            self.closed = True
            self.prepared = None
            active = self.active_thread
        # Both native release and CPU preparation have their own bounded
        # deadlines. Drain the owner before closing its process/pipe handles.
        if active is not None and active is not threading.current_thread():
            active.join()
        for planner in self.planners.values():
            if hasattr(planner, "close"):
                planner.close()


class PhysicsInterface(InterfaceState):
    def __init__(self, physics, *args):
        super().__init__(*args)
        self.physics = physics

    def scene(self):
        return {**super().scene(), **self.physics.planner.ui_scene,
                "bucket_bounds": self.physics.bounds, "simulation_only": True}

    def report(self):
        report = super().report()
        if self.solver is None:
            report.update(interface_ready=False, policy_ready=False, search_ready=False)
        return report


def physics_handler(interface, physics):
    class Handler(handler_for(interface)):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/interactive-physics.mjs":
                data = (WEB / "interactive-physics.mjs").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                return self.wfile.write(data)
            if path == API:
                return self._json(200, physics.capabilities())
            if path.startswith(API + "/"):
                try:
                    return self._json(200, physics.snapshot(path.removeprefix(API + "/")))
                except RequestError as exc:
                    return self._json(exc.status, {"error": str(exc)})
            return super().do_GET()

        def do_POST(self):
            path = urlsplit(self.path).path
            if not path.startswith(API + "/"):
                return super().do_POST()
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
                if path == API + "/plan":
                    return self._json(202, physics.submit_plan(payload))
                if path.endswith("/drop"):
                    job_id = path.removeprefix(API + "/").removesuffix("/drop")
                    return self._json(202, physics.submit_drop(job_id, payload))
                return self._json(404, {"error": "Not found"})
            except RequestError as exc:
                return self._json(exc.status, {"error": str(exc)})
            except (ValueError, UnicodeDecodeError):
                return self._json(400, {"error": "Invalid JSON request"})

    return Handler


def main():
    from .interactive_physics import PhysicsPlanner
    from .intergrid_flight_ranking_study import EXTRA_SOURCES
    from .intergrid_route_selector_study import SOURCE_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "parent-audit", "reference", "assets", "gpu-lock", "history-root", "rule",
                 "system-study", "system-audit", "evidence-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (*SOURCE_NAMES, *EXTRA_SOURCES, "prior-comparison"):
        for kind in ("study", "audit"):
            parser.add_argument(f"--{name}-{kind}", type=Path, required=True)
    parser.add_argument("--study-report", type=Path, help="Display a historical study alongside the planner")
    parser.add_argument("--scene-json", type=Path)
    parser.add_argument("--solver", help="Optional existing simulation solver for the historical modes")
    parser.add_argument("--gpu-wait-s", type=int, default=15, metavar="SECONDS",
                        help="Seconds to wait for the shared 4090, from 0 to 3600")
    parser.add_argument("--port", type=int, default=18781)
    for name in ("fast", "neural", "budget"):
        for kind in ("study", "audit"):
            parser.add_argument(f"--{name}-{kind}", type=Path)
    parser.add_argument("--default-planner", choices=("fast", "classic"))
    args = parser.parse_args()
    fast_paths = [getattr(args, f"{name}_{kind}") for name in ("fast", "neural", "budget") for kind in ("study", "audit")]
    if any(fast_paths) and not all(fast_paths):
        parser.error("Enabling fast planning requires all fast/neural/budget study and audit paths")
    if args.default_planner == "fast" and not all(fast_paths):
        parser.error("Fast default requires verified fast planning inputs")
    planner = PhysicsPlanner(args, args.evidence_dir)
    fast = None
    if all(fast_paths):
        from .interactive_fast_planner import FastPhysicsPlanner
        fast = FastPhysicsPlanner(args, args.evidence_dir / "fast", classic=planner)
    physics = InteractivePhysicsState(planner, fast_planner=fast, default_planner=args.default_planner)
    interface = PhysicsInterface(physics, args.study_report, args.scene_json, args.solver)
    def shutdown_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown_signal)
    with ThreadingHTTPServer(("127.0.0.1", args.port), physics_handler(interface, physics)) as server:
        print(f"Sircus mjlab physics planner: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            physics.close()


if __name__ == "__main__":
    main()
