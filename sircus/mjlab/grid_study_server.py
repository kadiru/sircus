"""Separate loopback view of the sweep study and hash-checked shot replays."""

import argparse
import json
import os
from pathlib import Path
import re
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

from .corrected_contract import sha
from .corrected_grid import home_layout_summary
from .grid_study import study_identity
from .grid_study_contract import validate_report
from .interface import InterfaceState, RequestError, handler_for
from .learning_contract import digest
from .sweep_server import public_state, ui_scene
from .sweep_worker import atomic, contained, load_reference


def followup_helpers():
    # The additive reader is imported only for an explicitly bound follow-up.
    from .grid_diverse_study import (
        load_parent_snapshot,
        study_identity as diverse_identity,
    )

    return diverse_identity, load_parent_snapshot


class GridState(InterfaceState):
    def __init__(self, ref, study, report, scene, solver, parent_study=None):
        super().__init__(report, scene, solver)
        self.study = Path(study)
        self.identity = study_identity(ref)
        self.parent_study, self.parent_metadata, self.parent_snapshot = None, None, None
        self.bound_contract, self.bound_contract_sha256 = None, None
        contract_path = self.study / "contract.json"
        try:
            current_contract = (
                json.loads(contract_path.read_text())
                if contract_path.is_file()
                else None
            )
        except (OSError, ValueError) as exc:
            raise RequestError(
                "Study contract is unavailable or failed verification", 503
            ) from exc
        if (
            isinstance(current_contract, dict)
            and current_contract.get("parent") is not None
        ):
            diverse_identity, self.load_parent_snapshot = followup_helpers()
            self.identity = diverse_identity(ref)
            unsigned = dict(current_contract)
            expected = unsigned.pop("fingerprint", None)
            if (
                expected != digest(unsigned)
                or current_contract.get("identity") != self.identity
            ):
                raise RequestError(
                    "Follow-up contract belongs to another source/scene identity", 409
                )
            if not isinstance(current_contract.get("parent"), dict) or not isinstance(
                current_contract.get("protocol"), dict
            ):
                raise RequestError(
                    "Follow-up contract lacks valid parent/protocol bindings", 409
                )
            self.parent_study = (
                Path(parent_study)
                if parent_study is not None
                else self.study / "parent"
            )
            self.parent_metadata = current_contract["parent"]
            self.bound_contract = current_contract
            self.bound_contract_sha256 = sha(contract_path)
        elif parent_study is not None:
            raise RequestError(
                "Parent evidence is not bound to this study contract", 409
            )

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
        protocol = (
            report.get("protocol") if isinstance(report.get("protocol"), dict) else {}
        )
        inherited = protocol.get("inherited_baseline") is True
        if inherited and self.parent_metadata is None:
            raise RequestError("Inherited study lacks a verified parent contract", 409)
        if self.parent_metadata is not None:
            try:
                if (
                    sha(self.study / "contract.json") != self.bound_contract_sha256
                    or report["protocol"] != self.bound_contract["protocol"]
                ):
                    raise ValueError("Follow-up contract or protocol changed")
                metadata = self.parent_metadata
                protocol = report["protocol"]
                if (
                    not inherited
                    or protocol.get("parent_contract_fingerprint")
                    != metadata.get("contract_fingerprint")
                    or protocol.get("parent_manifest_sha256")
                    != metadata.get("study_manifest_sha256")
                    or protocol.get("parent_audit_sha256")
                    != metadata.get("audit_sha256")
                ):
                    raise ValueError("Follow-up parent bindings changed")
                self.parent_snapshot = self.load_parent_snapshot(
                    self.parent_study, expected_metadata=self.parent_metadata
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise RequestError(
                    "Parent study evidence is unavailable or failed verification", 409
                ) from exc
        return report

    def report(self):
        report = super().report()
        try:
            report["planner_grid_study"] = self.grid_report()
            report["planner_grid_blockers"] = self.blocker_notes(
                report["planner_grid_study"]
            )
            if self.parent_snapshot is not None:
                report["planner_grid_parent"] = self.parent_snapshot["report"]
        except RequestError as exc:
            report["planner_grid_error"] = str(exc)
        return report

    def blocker_notes(self, report):
        """Explain hash-checked home failures without altering signed evidence."""
        notes = {}
        for row in report["cells"]:
            if row.get("status") != "blocked":
                continue
            attempts = [
                a
                for a in row.get("attempts", [])
                if a.get("stage") in {"baseline", "expanded"}
            ]
            if not attempts:
                continue
            attempt = attempts[-1]
            expected = attempt.get("evidence_sha256", {}).get("home.json")
            if not expected:
                continue
            try:
                path = contained(
                    self.study, f"cells/{row['id']}/{attempt['stage']}/home.json"
                )
                if sha(path) != expected:
                    continue
                home = json.loads(path.read_text())
                if not isinstance(home, dict) or home.get("passed") is not False:
                    continue
                checked = home_layout_summary(home)
                unresolved_base_clearance = any(
                    isinstance(item, dict)
                    and isinstance(item.get("pair"), list)
                    and len(item["pair"]) == 2
                    and all(isinstance(name, str) for name in item["pair"])
                    and any(
                        name.startswith(("Arm10/", "Arm14/")) and "/link_base/" in name
                        for name in item["pair"]
                    )
                    and any(name.startswith("bucket/") for name in item["pair"])
                    for item in home.get("unresolved_clearance_pairs", [])
                )
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            notes[row["id"]] = (
                "Bucket overlaps an arm base; another placement is needed."
                if checked["fixed_base_penetrations"]
                else (
                    "Clearance between the bucket and a fixed arm base is unproven; placement or geometry needs validation."
                    if unresolved_base_clearance
                    else "Prescribed starting pose is blocked; another staging strategy needs validation."
                )
            )
        return notes

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
    parser.add_argument("--parent-study-dir", type=Path)
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
        ref,
        args.study_dir,
        report,
        scene,
        "sircus.mjlab.sweep_interface:solve",
        parent_study=args.parent_study_dir,
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
