"""Read-only v2 benchmark view; historical v1 evidence keeps its original rule.

This loopback reader never launches training or a solver. TEST outcomes require
completed study bytes and an independently passed audit bound to those bytes.
"""

import argparse
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

from .benchmark_server import COUNT_FIELDS, RESOURCE_FIELDS, fields, public_benchmark
from .calibrated_benchmark_contract import AUDIT_VERSION, VERSION
from .corrected_contract import sha
from .grid_study_server import GridState, grid_handler
from .interface import RequestError
from .learning_contract import digest
from .sweep_server import public_state, ui_scene
from .sweep_worker import atomic, load_reference


def benchmark_helpers():
    from .calibrated_admission import validate_home
    from .calibrated_benchmark_contract import (
        checked_calibration, checked_setup_home, study_identity, validate_report,
    )
    return checked_calibration, checked_setup_home, study_identity, validate_report, validate_home


def admission_fields(value):
    result = fields(value, ("requested", "native_releases", "rejected", "fixed_base_rejected", "anchor_unresolved"))
    result.update(fields(value, ("invalid_physics",)))
    if isinstance(value, dict) and isinstance(value.get("reasons"), dict):
        result["reasons"] = {str(k): v for k, v in value["reasons"].items()
                             if type(v) is int and v >= 0}
    return result


def public_calibrated_benchmark(raw, identity, audit=None):
    # Reuse v1 aggregate projection, then impose the stricter v2 disclosure gate.
    value = public_benchmark(raw)
    value["final_test"] = []
    ready = raw.get("status") == "completed" and audit is not None and audit.get("status") == "passed"
    value["test_disclosure"] = "verified" if ready else "hidden_until_completed_and_audited"
    value["evidence_kind"] = "native_current_ball_v2"
    value["calibration"] = fields(identity["calibration"], (
        "calibration_version", "verification_version", "calibration_checks", "verification_checks", "score_config_sha256"))
    value["calibration"]["score_config"] = fields(identity["calibration"]["score_config"], (
        "version", "fingerprint", "wall_tolerance_m", "floor_tolerance_m", "geometry_epsilon_m",
        "max_speed_m_s", "retention_s", "native_dt_s"))
    value["setup"] = fields(identity["setup"], (
        "resolver", "geometry_candidate_budget", "rrt_nodes", "source_order",
        "no_flight_labels", "no_postprediction_action_fallback", "home_record_sha256"))
    value["setup"]["home"] = fields(identity["setup"]["home"], (
        "version", "sha256", "initial_state_assumption", "minimum_clearance_m"))
    value["protocol"].update(fields(raw.get("protocol"), (
        "study_kind", "scorer_version", "family", "anchor_rule", "observations", "setup_resolution", "evaluation_history")))
    for projected, source in zip(value["curves"], raw.get("curves", [])):
        projected["resources"] = fields(source.get("resources"), RESOURCE_FIELDS + (
            "path_s", "context_resolution_s", "admission_wall_s", "postprocess_s"))
    for projected, source in zip(value["runs"], raw.get("runs", [])):
        projected["resources"] = fields(source.get("resources"), RESOURCE_FIELDS + (
            "path_s", "context_resolution_s", "admission_wall_s", "postprocess_s"))
        projected["train_counts"] = fields(source.get("train_counts"), COUNT_FIELDS)
        projected["admission"] = admission_fields(source.get("admission"))
    if audit is not None:
        value["audit"] = fields(audit, ("version", "status", "checks", "scope", "limitations", "synthetic_fixture", "study_kind"))
    if ready:
        for source in raw.get("final_test", []):
            item = fields(source, ("method", "seed", "frozen_artifact_sha256", "query_s", "pre_prediction_plus_policy_s", "selected_samples",
                                  "geometry_search_at_inference", "ball_flight_search_at_inference"))
            item["counts"] = fields(source.get("counts"), COUNT_FIELDS)
            item["context_resolution_timing"] = fields(source.get("context_resolution_timing"), (
                "geometry_s", "path_s", "wall_s", "resolver_candidates", "resolver_cache_hits"))
            value["final_test"].append(item)
        value["audit"]["role_counts"] = {role: fields(audit.get("role_counts", {}).get(role), COUNT_FIELDS)
                                          for role in ("train", "validation", "test")}
    return value


class CalibratedBenchmarkState(GridState):
    def __init__(self, ref, study, report, scene, solver, benchmark, parent_study=None,
                 benchmark_audit=None, historical_benchmark=None):
        super().__init__(ref, study, report, scene, solver, parent_study=parent_study)
        self.benchmark = Path(benchmark)
        self.benchmark_audit = Path(benchmark_audit) if benchmark_audit else self.benchmark.with_name(self.benchmark.name + "-audit.json")
        self.historical_benchmark = Path(historical_benchmark) if historical_benchmark else None
        self.history_manifest_sha256 = None
        if self.historical_benchmark:
            self.history_manifest_sha256 = sha(self.historical_benchmark / "SHA256SUMS")
        try:
            calibration, setup, identity, self.validate_benchmark, validate_home = benchmark_helpers()
            self.calibration = calibration(self.benchmark / "calibration.json", self.benchmark / "verification.json", synthetic=False)
            self.setup = setup(self.benchmark / "setup-home.json")
            validate_home(self.setup["home"], ref["scene"])
            self.benchmark_identity = identity(ref, self.calibration, self.setup)
            contract = json.loads((self.benchmark / "contract.json").read_text())
            unsigned = dict(contract); fingerprint = unsigned.pop("fingerprint", None)
            if (contract.get("version") != VERSION or fingerprint != digest(unsigned)
                    or contract.get("identity") != self.benchmark_identity
                    or contract.get("protocol", {}).get("study_kind") != "native_current_ball_v2"):
                raise ValueError("Calibrated benchmark source/scene/control identity changed")
            self.benchmark_contract = contract
            self.frozen_files = {name: sha(self.benchmark / name) for name in (
                "contract.json", "calibration.json", "verification.json", "setup-home.json")}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RequestError("Calibrated benchmark controls are unavailable or incompatible", 409) from exc

    def _audit(self, raw, report_path):
        if not self.benchmark_audit.is_file(): return None
        audit = json.loads(self.benchmark_audit.read_text())
        if (audit.get("version") != AUDIT_VERSION
                or audit.get("contract_fingerprint") != self.benchmark_contract["fingerprint"]
                or audit.get("report_sha256") != sha(report_path)
                or audit.get("report_fingerprint") != raw["fingerprint"]
                or audit.get("study_manifest_sha256") != sha(self.benchmark / "manifest.json")
                or audit.get("source_sha256") != self.benchmark_identity["source_sha256"]
                or audit.get("auditor_sha256") != sha(Path(__file__).with_name("calibrated_benchmark_verify.py"))
                or audit.get("auditor_dependencies") != {"sircus/mjlab/benchmark_verify.py": sha(Path(__file__).with_name("benchmark_verify.py"))}
                or audit.get("synthetic_fixture") is not False
                or audit.get("study_kind") != "native_current_ball_v2"):
            raise ValueError("Calibrated independent audit binding changed")
        if audit.get("status") == "passed" and (
                type(audit.get("checks")) is not int or audit["checks"] <= 0 or audit.get("failures")):
            raise ValueError("Passed independent audit lacks completed checks")
        return audit

    def benchmark_report(self):
        try:
            if any(sha(self.benchmark / name) != expected for name, expected in self.frozen_files.items()):
                raise ValueError("Frozen calibrated controls changed after startup")
            path = self.benchmark / "report.json"
            raw = self.validate_benchmark(json.loads(path.read_text()))
            if (raw.get("contract_fingerprint") != self.benchmark_contract["fingerprint"]
                    or raw.get("protocol") != self.benchmark_contract["protocol"]
                    or raw.get("identity", self.benchmark_identity) != self.benchmark_identity):
                raise ValueError("Report belongs to another calibrated contract")
            return public_calibrated_benchmark(raw, self.benchmark_identity, self._audit(raw, path))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RequestError("Calibrated report failed source/control/audit verification", 409) from exc

    def history_report(self):
        if self.historical_benchmark is None: return None
        from .benchmark_contract import validate_report
        root = self.historical_benchmark
        if sha(root / "SHA256SUMS") != self.history_manifest_sha256:
            raise ValueError("Historical checksum manifest changed")
        manifest = dict(line.split("  ", 1)[::-1] for line in (root / "SHA256SUMS").read_text().splitlines())
        for name in ("report.json", "contract.json", "audit.json", "provenance.json"):
            if sha(root / name) != manifest.get(name): raise ValueError("Historical archive bytes changed")
        raw = validate_report(json.loads((root / "report.json").read_text()))
        contract = json.loads((root / "contract.json").read_text())
        unsigned = dict(contract); fingerprint = unsigned.pop("fingerprint", None)
        provenance = json.loads((root / "provenance.json").read_text())
        audit = json.loads((root / "audit.json").read_text())
        if (raw.get("version") != "sircus-sweep-rl-benchmark-v1"
                or contract.get("version") != raw["version"] or fingerprint != digest(unsigned)
                or fingerprint != raw["contract_fingerprint"] or raw.get("protocol") != contract.get("protocol")
                or provenance.get("report_sha256") != sha(root / "report.json")
                or provenance.get("contract_sha256") != sha(root / "contract.json")
                or audit.get("status") != "passed" or audit.get("report_sha256") != sha(root / "report.json")
                or type(audit.get("checks")) is not int or audit["checks"] <= 0 or audit.get("failures")
                or audit.get("report_fingerprint") != raw["fingerprint"]
                or audit.get("contract_fingerprint") != raw["contract_fingerprint"]):
            raise ValueError("Historical provenance changed")
        value = public_benchmark(raw)
        value.update(historical=True, metric=provenance.get("metric"), limitation=provenance.get("limitation"))
        return value

    def report(self):
        value = super().report()
        value["benchmark_read_only"] = True
        try: value["sweep_rl_benchmark"] = self.benchmark_report()
        except RequestError as exc: value["sweep_rl_error"] = str(exc)
        try:
            history = self.history_report()
            if history is not None: value["historical_sweep_rl_benchmark"] = history
        except (OSError, ValueError, TypeError, KeyError) as exc:
            value["historical_sweep_rl_error"] = "Historical v1 evidence failed verification"
        return value


def calibrated_handler(state):
    base = grid_handler(state)
    class Handler(base):
        def do_POST(self):
            return self._json(405, {"error": "This benchmark view is read-only; no solver or training command is available"})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "assets", "gpu-lock", "cache", "state-dir", "study-dir", "benchmark-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parent-study-dir", type=Path)
    parser.add_argument("--benchmark-audit", type=Path)
    parser.add_argument("--historical-benchmark-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, default=18790)
    args = parser.parse_args()
    ref = load_reference(args.reference, args.assets)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    report, scene = args.state_dir / "planner-report.json", args.state_dir / "ui-scene.json"
    atomic(report, public_state(ref)); atomic(scene, ui_scene(ref))
    for key, path in (("REFERENCE", args.reference), ("ASSETS", args.assets), ("LOCK", args.gpu_lock), ("CACHE", args.cache)):
        os.environ["SIRCUS_SWEEP_" + key] = str(path.resolve())
    state = CalibratedBenchmarkState(ref, args.study_dir, report, scene, None, args.benchmark_dir,
        parent_study=args.parent_study_dir, benchmark_audit=args.benchmark_audit,
        historical_benchmark=args.historical_benchmark_dir)
    state.grid_report(); state.benchmark_report()
    server = ThreadingHTTPServer((args.host, args.port), calibrated_handler(state))
    print(json.dumps(dict(url=f"http://{args.host}:{args.port}/#sweepRlBenchmark", simulation_only=True,
                          benchmark_read_only=True)), flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == "__main__": main()
