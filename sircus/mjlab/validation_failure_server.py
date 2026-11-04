"""Separate read-only view of an immutable benchmark's validation diagnostics.

No training, native rollouts, solver commands or hardware clients. The diagnostic
report is pinned by bytes and bound to the benchmark's passed independent audit.
Replay requests serve only the report's hash-bound validation representatives.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

from .calibrated_benchmark_server import CalibratedBenchmarkState, calibrated_handler
from .corrected_contract import sha
from .interface import RequestError
from .learning_contract import digest
from .sweep_server import public_state, ui_scene
from .sweep_worker import atomic, load_reference

VERSION = "sircus-calibrated-validation-failure-audit-v1"
REPLAY_ID = re.compile(r"[a-z0-9][a-z0-9_-]{7,95}\Z")


def pinned_json(path, expected):
    if path.is_symlink():
        raise ValueError("Pinned evidence cannot be a symlink")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Pinned evidence bytes changed")
    return json.loads(raw)


def validate_state_directory(state, evidence):
    state = Path(state).resolve()
    if state.exists() or any(state.is_relative_to(Path(p).resolve()) for p in evidence if p is not None):
        raise ValueError("A new reader state directory outside every evidence root is required")


class ValidationFailureState(CalibratedBenchmarkState):
    def __init__(self, *args, diagnostic_dir, diagnostic_sha256, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_dir = Path(diagnostic_dir)
        self.diagnostic_sha256 = diagnostic_sha256
        self.diagnostic_report()

    def diagnostic_report(self):
        try:
            current = self.benchmark_report()
            if (current.get("status") != "completed" or current.get("test_disclosure") != "verified"
                    or current.get("audit", {}).get("status") != "passed"):
                raise ValueError("Completed, independently audited source required")
            path = self.diagnostic_dir / "report.json"
            value = pinned_json(path, self.diagnostic_sha256)
            if not isinstance(value, dict):
                raise ValueError("Diagnostic report must be an object")
            unsigned = dict(value)
            fingerprint = unsigned.pop("fingerprint", None)
            if (value.get("version") != VERSION or value.get("status") != "passed"
                    or value.get("synthetic_fixture") is not False or fingerprint != digest(unsigned)):
                raise ValueError("Diagnostic version/status/fingerprint changed")
            source = value["source"]
            if not isinstance(source, dict) or not isinstance(value.get("representatives"), list):
                raise ValueError("Diagnostic source and representative list are required")
            audit = pinned_json(self.benchmark_audit, source.get("audit_sha256"))
            for key in ("contract_fingerprint", "report_sha256", "study_manifest_sha256",
                        "source_sha256", "auditor_sha256", "auditor_dependencies"):
                if source.get(key) != audit.get(key):
                    raise ValueError("Diagnostic source binding changed: " + key)
            if (source.get("audit_sha256") != sha(self.benchmark_audit)
                    or source["report_sha256"] != sha(self.benchmark / "report.json")
                    or source["study_manifest_sha256"] != sha(self.benchmark / "manifest.json")
                    or source["contract_fingerprint"] != self.benchmark_contract["fingerprint"]):
                raise ValueError("Diagnostic source bytes changed")
            seen = set()
            for item in value.get("representatives", []):
                if not isinstance(item, dict):
                    raise ValueError("Representative metadata must be an object")
                replay_id = item.get("replay_id", item.get("id"))
                if (not isinstance(replay_id, str) or not REPLAY_ID.fullmatch(replay_id)
                        or replay_id in seen or type(item.get("recorded_replay_available")) is not bool):
                    raise ValueError("Invalid representative identity")
                if item["recorded_replay_available"] and not re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256", ""))):
                    raise ValueError("Available replay requires exact evidence hash")
                if not item["recorded_replay_available"] and (item.get("sha256") or item.get("path")):
                    raise ValueError("Unavailable replay cannot claim evidence")
                seen.add(replay_id)
            return value
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RequestError("Validation diagnostic failed source/evidence verification", 409) from exc

    def report(self):
        value = super().report()
        try:
            value["validation_failure_audit"] = self.diagnostic_report()
        except RequestError as exc:
            value["validation_failure_audit_error"] = str(exc)
        return value

    def validation_replay(self, replay_id):
        if not REPLAY_ID.fullmatch(replay_id):
            raise RequestError("Unknown validation replay", 404)
        report = self.diagnostic_report()
        item = next((r for r in report.get("representatives", [])
                     if r.get("replay_id", r.get("id")) == replay_id), None)
        if item is None or not item["recorded_replay_available"]:
            raise RequestError("Unknown validation replay", 404)
        try:
            path = self.diagnostic_dir / "replays" / (replay_id + ".json")
            if not path.resolve().is_relative_to(self.diagnostic_dir.resolve()):
                raise ValueError("Recorded replay leaves its evidence root")
            value = pinned_json(path, item["sha256"])
            if (not isinstance(value, dict) or value.get("version") != "sircus-recorded-validation-replay-v1"
                    or value.get("role") != "validation" or value.get("id") != replay_id
                    or not isinstance(value.get("source"), dict)
                    or value["source"] != report["source"]
                    or value.get("source_row_id") != item.get("source_row_id")
                    or not isinstance(value.get("result"), dict)
                    or value["result"].get("request_id") != item.get("source_row_id")
                    or value["result"].get("method") != "validation_recorded_replay"):
                raise ValueError("Recorded validation replay identity changed")
            return value["result"]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RequestError("Recorded validation replay failed evidence verification", 409) from exc


def validation_handler(state):
    base = calibrated_handler(state)

    class Handler(base):
        def do_GET(self):
            path = urlsplit(self.path).path
            prefix = "/api/validation-failures/replay/"
            if path.startswith(prefix):
                try:
                    return self._json(200, state.validation_replay(path[len(prefix):]))
                except RequestError as exc:
                    return self._json(exc.status, {"error": str(exc)})
            return super().do_GET()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "assets", "gpu-lock", "cache", "state-dir", "study-dir",
                 "benchmark-dir", "benchmark-audit", "diagnostic-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--diagnostic-sha256", required=True)
    parser.add_argument("--parent-study-dir", type=Path)
    parser.add_argument("--historical-benchmark-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, default=18792)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{64}", args.diagnostic_sha256):
        parser.error("A pinned diagnostic SHA256 is required")
    try:
        validate_state_directory(args.state_dir, (args.benchmark_dir, args.diagnostic_dir, args.reference, args.assets,
            args.study_dir, args.parent_study_dir, args.historical_benchmark_dir))
    except ValueError as exc:
        parser.error(str(exc))
    ref = load_reference(args.reference, args.assets)
    report, scene = args.state_dir / "planner-report.json", args.state_dir / "ui-scene.json"
    for key, path in (("REFERENCE", args.reference), ("ASSETS", args.assets), ("LOCK", args.gpu_lock), ("CACHE", args.cache)):
        os.environ["SIRCUS_SWEEP_" + key] = str(path.resolve())
    state = ValidationFailureState(ref, args.study_dir, report, scene, None, args.benchmark_dir,
        parent_study=args.parent_study_dir, benchmark_audit=args.benchmark_audit,
        historical_benchmark=args.historical_benchmark_dir, diagnostic_dir=args.diagnostic_dir,
        diagnostic_sha256=args.diagnostic_sha256)
    args.state_dir.mkdir(parents=True, exist_ok=False)
    atomic(report, public_state(ref)); atomic(scene, ui_scene(ref))
    server = ThreadingHTTPServer((args.host, args.port), validation_handler(state))
    print(json.dumps(dict(url=f"http://{args.host}:{args.port}/#validationFailureAudit",
                          simulation_only=True, read_only=True)), flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == "__main__": main()
