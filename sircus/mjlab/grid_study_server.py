"""Separate loopback view of the sweep study and hash-checked shot replays."""

import argparse
import json
import os
from pathlib import Path
import re
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

from .grid_study import study_identity
from .grid_study_contract import validate_report
from .interface import InterfaceState, RequestError, handler_for
from .learning_contract import digest
from .sweep_server import public_state, ui_scene
from .sweep_worker import atomic, load_reference


class GridState(InterfaceState):
    def __init__(self, ref, study, report, scene, solver):
        super().__init__(report, scene, solver)
        self.study = Path(study)
        self.identity = study_identity(ref)

    def grid_report(self):
        try:
            report = validate_report(
                json.loads((self.study / "report.json").read_text())
            )
        except (OSError, ValueError) as exc:
            raise RequestError(
                "Study report is unavailable or failed verification", 503
            ) from exc
        if report["identity"] != self.identity:
            raise RequestError(
                "Study report belongs to another source/scene identity", 409
            )
        return report

    def report(self):
        report = super().report()
        try:
            report["planner_grid_study"] = self.grid_report()
        except RequestError as exc:
            report["planner_grid_error"] = str(exc)
        return report

    def recorded_result(self, cell_id):
        if not re.fullmatch(r"dev-grid-\d{2}-\d{2}", cell_id):
            raise RequestError("Unknown grid target", 404)
        report = self.grid_report()
        row = next((r for r in report["cells"] if r["id"] == cell_id), None)
        if row is None or not row.get("selected"):
            raise RequestError("No qualified recorded shot for this cell", 404)
        selected = row["selected"]
        try:
            result = json.loads(
                (self.study / "results" / (cell_id + ".json")).read_text()
            )
        except (OSError, ValueError) as exc:
            raise RequestError("Recorded shot is not yet available", 503) from exc
        attempt = next(
            (a for a in row["attempts"] if a["request_id"] == selected["request_id"]),
            None,
        )
        if (
            attempt is None
            or digest(result) != attempt["result_payload_sha256"]
            or digest(result.get("configuration")) != selected["configuration_sha256"]
            or result.get("bucket_xy_m") != row["bucket_xy_m"]
            or result.get("request_id") != selected["request_id"]
        ):
            raise RequestError("Recorded shot identity changed", 409)
        return result


def grid_handler(state):
    base = handler_for(state)

    class Handler(base):
        def do_GET(self):
            path = urlsplit(self.path).path
            prefix = "/api/planner-grid/result/"
            if path.startswith(prefix):
                try:
                    return self._json(200, state.recorded_result(path[len(prefix) :]))
                except RequestError as exc:
                    return self._json(exc.status, {"error": str(exc)})
            return super().do_GET()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "assets", "gpu-lock", "cache", "state-dir", "study-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"]
    )
    parser.add_argument("--port", type=int, default=18787)
    args = parser.parse_args()
    ref = load_reference(args.reference, args.assets)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    report, scene = (
        args.state_dir / "planner-report.json",
        args.state_dir / "ui-scene.json",
    )
    atomic(report, public_state(ref))
    atomic(scene, ui_scene(ref))
    for key, value in (
        ("REFERENCE", args.reference),
        ("ASSETS", args.assets),
        ("LOCK", args.gpu_lock),
        ("CACHE", args.cache),
    ):
        os.environ["SIRCUS_SWEEP_" + key] = str(value.resolve())
    state = GridState(
        ref, args.study_dir, report, scene, "sircus.mjlab.sweep_interface:solve"
    )
    # Starting a view must not silently show evidence from incompatible code.
    state.grid_report()
    server = ThreadingHTTPServer((args.host, args.port), grid_handler(state))
    print(
        json.dumps(dict(url=f"http://{args.host}:{args.port}/", simulation_only=True)),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
