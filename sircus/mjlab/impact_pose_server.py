"""Separate read-only reader for validation-only impact-pose diagnostics.

The completed benchmark and prior validation audit remain byte-pinned. Neither
this reader nor a replay GET can schedule physics, training or robot commands.
"""

import argparse
import json
import os
from pathlib import Path
import re
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

from .corrected_contract import sha
from .interface import RequestError
from .learning_contract import digest
from .sweep_server import public_state, ui_scene
from .sweep_worker import atomic, load_reference
from .validation_failure_server import (
    REPLAY_ID, ValidationFailureState, pinned_json, validate_state_directory,
    validation_handler,
)


VERSION = "sircus-impact-pose-sweep-v1"


class ImpactPoseState(ValidationFailureState):
    def __init__(self, *args, impact_dir, impact_sha256, impact_audit_sha256, **kwargs):
        super().__init__(*args, **kwargs)
        self.impact_dir = Path(impact_dir)
        self.impact_sha256 = impact_sha256
        self.impact_audit_sha256 = impact_audit_sha256
        self.impact_report()

    def impact_report(self):
        try:
            diagnostic = self.diagnostic_report()
            value = pinned_json(self.impact_dir / "report.json", self.impact_sha256)
            audit = pinned_json(self.impact_dir / "audit.json", self.impact_audit_sha256)
            if not isinstance(value, dict) or not isinstance(audit, dict):
                raise ValueError("Report and independent audit must be objects")
            unsigned = dict(value)
            fingerprint = unsigned.pop("fingerprint", None)
            if (value.get("version") != VERSION or value.get("status") != "completed"
                    or value.get("synthetic_fixture") is not False or fingerprint != digest(unsigned)):
                raise ValueError("Completed native diagnostic with valid fingerprint required")
            if (audit.get("version") != "sircus-impact-pose-sweep-independent-audit-v1"
                    or audit.get("synthetic_fixture") is not False or audit.get("status") != "passed"
                    or audit.get("report_sha256") != self.impact_sha256
                    or audit.get("plan_sha256") != sha(self.impact_dir / "plan.json")
                    or audit.get("manifest_sha256") != sha(self.impact_dir / "manifest.json")
                    or type(audit.get("checks")) is not int or audit["checks"] <= 0
                    or audit.get("failures")):
                raise ValueError("Passed audit must bind exact report, plan and manifest bytes")
            source = value.get("source")
            if not isinstance(source, dict):
                raise ValueError("Frozen source identity required")
            expected = dict(
                benchmark_report_sha256=diagnostic["source"]["report_sha256"],
                benchmark_manifest_sha256=diagnostic["source"]["study_manifest_sha256"],
                benchmark_audit_sha256=diagnostic["source"]["audit_sha256"],
                validation_report_sha256=self.diagnostic_sha256,
                contract_fingerprint=diagnostic["source"]["contract_fingerprint"],
            )
            if any(source.get(k) != v for k, v in expected.items()):
                raise ValueError("Impact diagnostic lost original benchmark/validation binding")
            if audit.get("source") != source:
                raise ValueError("Independent audit source changed")
            seen = set()
            layouts = value.get("layouts")
            if not isinstance(layouts, list) or not layouts:
                raise ValueError("Explicit diagnostic cases required")
            for case in layouts:
                if not isinstance(case, dict) or not isinstance(case.get("trials"), list):
                    raise ValueError("Invalid case metadata")
                for trial in case["trials"]:
                    if not isinstance(trial, dict):
                        raise ValueError("Invalid trial metadata")
                    replay_id = trial.get("replay_id")
                    if trial.get("native_release") is True:
                        if (not isinstance(replay_id, str) or not REPLAY_ID.fullmatch(replay_id)
                                or replay_id in seen or not re.fullmatch(r"[a-f0-9]{64}", str(trial.get("replay_sha256", "")))):
                            raise ValueError("Native replay needs unique opaque ID and hash")
                        seen.add(replay_id)
                    elif replay_id or trial.get("replay_sha256"):
                        raise ValueError("Rejected trial cannot claim a native trajectory")
            # Attach only the independently pinned audit, never a producer claim.
            return dict(value, audit=dict(audit, audit_sha256=self.impact_audit_sha256))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RequestError("Impact diagnostic failed source/evidence verification", 409) from exc

    def report(self):
        value = super().report()
        try:
            value["impact_pose_sweep"] = self.impact_report()
        except RequestError as exc:
            value["impact_pose_sweep_error"] = str(exc)
        return value

    def impact_replay(self, replay_id):
        if not REPLAY_ID.fullmatch(replay_id):
            raise RequestError("Unknown impact diagnostic replay", 404)
        report = self.impact_report()
        pair = next(((case, row) for case in report["layouts"] for row in case["trials"]
                     if row.get("replay_id") == replay_id and row.get("native_release") is True), None)
        if pair is None:
            raise RequestError("Unknown impact diagnostic replay", 404)
        case, row = pair
        try:
            path = self.impact_dir / "replays" / (replay_id + ".json")
            if not path.resolve().is_relative_to(self.impact_dir.resolve()):
                raise ValueError("Replay leaves evidence root")
            value = pinned_json(path, row["replay_sha256"])
            result = value.get("result")
            if (value.get("version") != "sircus-impact-pose-replay-v1" or value.get("id") != replay_id
                    or value.get("role") != "validation_diagnostic" or value.get("source") != report["source"]
                    or value.get("source_row_id") != case.get("source_row_id")
                    or not isinstance(result, dict) or result.get("request_id") != row.get("id")
                    or result.get("method") != "impact_sweep_recorded_replay"
                    or result.get("bucket_xy_m") != case.get("bucket_xy_m")):
                raise ValueError("Native diagnostic replay identity changed")
            return result
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise RequestError("Impact diagnostic replay failed evidence verification", 409) from exc


def impact_handler(state):
    base = validation_handler(state)

    class Handler(base):
        def do_GET(self):
            path = urlsplit(self.path).path
            prefix = "/api/impact-sweep/replay/"
            if path.startswith(prefix):
                try:
                    return self._json(200, state.impact_replay(path[len(prefix):]))
                except RequestError as exc:
                    return self._json(exc.status, {"error": str(exc)})
            return super().do_GET()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "assets", "gpu-lock", "cache", "state-dir", "study-dir",
                 "benchmark-dir", "benchmark-audit", "diagnostic-dir", "impact-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("diagnostic-sha256", "impact-sha256", "impact-audit-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--parent-study-dir", type=Path)
    parser.add_argument("--historical-benchmark-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, default=18793)
    args = parser.parse_args()
    for key in ("diagnostic_sha256", "impact_sha256", "impact_audit_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", getattr(args, key)):
            parser.error("Pinned SHA256 required: " + key)
    try:
        validate_state_directory(args.state_dir, (args.benchmark_dir, args.diagnostic_dir, args.impact_dir,
            args.reference, args.assets, args.study_dir, args.parent_study_dir, args.historical_benchmark_dir))
    except ValueError as exc:
        parser.error(str(exc))
    ref = load_reference(args.reference, args.assets)
    report, scene = args.state_dir / "planner-report.json", args.state_dir / "ui-scene.json"
    for key, path in (("REFERENCE", args.reference), ("ASSETS", args.assets), ("LOCK", args.gpu_lock), ("CACHE", args.cache)):
        os.environ["SIRCUS_SWEEP_" + key] = str(path.resolve())
    state = ImpactPoseState(ref, args.study_dir, report, scene, None, args.benchmark_dir,
        parent_study=args.parent_study_dir, benchmark_audit=args.benchmark_audit,
        historical_benchmark=args.historical_benchmark_dir, diagnostic_dir=args.diagnostic_dir,
        diagnostic_sha256=args.diagnostic_sha256, impact_dir=args.impact_dir,
        impact_sha256=args.impact_sha256, impact_audit_sha256=args.impact_audit_sha256)
    args.state_dir.mkdir(parents=True, exist_ok=False)
    atomic(report, public_state(ref)); atomic(scene, ui_scene(ref))
    server = ThreadingHTTPServer((args.host, args.port), impact_handler(state))
    print(json.dumps(dict(url=f"http://{args.host}:{args.port}/#impactPoseSweep",
        simulation_only=True, read_only=True)), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
