"""Read-only matched-budget benchmark alongside preserved planner results.

Serving this view never starts training, touches hardware, or changes the solver
route. Benchmark reports and optional independent audits must match the pinned
source/scene contract. Native arrays, checkpoint paths and host state are omitted.
"""

import argparse
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

from .corrected_contract import sha
from .grid_study_server import GridState, grid_handler
from .interface import RequestError
from .learning_contract import digest
from .sweep_server import public_state, ui_scene
from .sweep_worker import atomic, load_reference


VERSION = "sircus-sweep-rl-benchmark-v1"
RESOURCE_FIELDS = (
    "wall_s",
    "cpu_s",
    "geometry_s",
    "gpu_s",
    "peak_rss_mb",
    "peak_vram_mb",
    "torch_allocated_mb",
    "torch_reserved_mb",
    "setup_s",
    "evaluation_wall_s",
    "gpu_reservation_s",
)
COUNT_FIELDS = (
    "attempts",
    "captures",
    "native_releases",
    "invalid",
    "clean",
    "assisted",
    "mean_reward",
)


def benchmark_helpers():
    from .benchmark_contract import study_identity, validate_report

    return study_identity, validate_report


def fields(value, names):
    return (
        {key: value[key] for key in names if key in value}
        if isinstance(value, dict)
        else {}
    )


def public_benchmark(raw):
    """Expose measured aggregates, never raw records or local commands/paths."""
    value = fields(
        raw,
        (
            "version",
            "status",
            "phase",
            "contract_fingerprint",
            "fingerprint",
            "scope",
            "started_at",
            "updated_at",
        ),
    )
    protocol = fields(
        raw.get("protocol"),
        (
            "methods",
            "seeds",
            "milestones",
            "sample_budget",
            "scope",
            "budget_unit",
            "noise",
            "action_space",
            "actions",
            "train_layouts",
            "validation_layouts",
            "test_layouts",
            "evaluation_repeats",
            "capture_rule",
            "reward_version",
            "resource_provenance",
        ),
    )
    protocol["prior"] = fields(
        raw.get("protocol", {}).get("prior"),
        (
            "recipes",
            "latest_study_native_releases",
            "full_historical_cost_known",
            "scope",
            "source_sha256",
        ),
    )
    value["protocol"] = protocol
    value["curves"] = []
    for point in raw.get("curves", []):
        item = fields(point, ("method", "seed", "samples", "artifact_sha256"))
        item["validation"] = fields(point.get("validation"), COUNT_FIELDS)
        item["resources"] = fields(point.get("resources"), RESOURCE_FIELDS)
        value["curves"].append(item)
    value["runs"] = []
    for run in raw.get("runs", []):
        item = fields(
            run,
            (
                "method",
                "seed",
                "status",
                "train_requests",
                "train_native_releases",
                "selected_samples",
                "checkpoint_sha256",
                "artifact_sha256",
            ),
        )
        item["resources"] = fields(run.get("resources"), RESOURCE_FIELDS)
        value["runs"].append(item)
    # Test feedback is never used as a live training curve.
    value["final_test"] = []
    if raw.get("status") == "completed":
        for result in raw.get("final_test", []):
            item = fields(
                result, ("method", "seed", "frozen_artifact_sha256", "query_s")
            )
            item["counts"] = fields(result.get("counts"), COUNT_FIELDS)
            value["final_test"].append(item)
    return value


class BenchmarkState(GridState):
    def __init__(
        self,
        ref,
        study,
        report,
        scene,
        solver,
        benchmark,
        parent_study=None,
        benchmark_audit=None,
    ):
        super().__init__(ref, study, report, scene, solver, parent_study=parent_study)
        self.benchmark = Path(benchmark)
        self.benchmark_audit = (
            Path(benchmark_audit) if benchmark_audit is not None else None
        )
        identity, self.validate_benchmark = benchmark_helpers()
        self.benchmark_identity = identity(ref)
        path = self.benchmark / "contract.json"
        try:
            contract = json.loads(path.read_text())
            unsigned = dict(contract)
            fingerprint = unsigned.pop("fingerprint", None)
            if (
                fingerprint != digest(unsigned)
                or contract.get("identity") != self.benchmark_identity
            ):
                raise ValueError("Benchmark contract identity changed")
            if not isinstance(contract.get("protocol"), dict):
                raise ValueError("Benchmark protocol missing")
            self.benchmark_contract = contract
            self.benchmark_contract_sha256 = sha(path)
        except (OSError, ValueError, TypeError) as exc:
            raise RequestError(
                "Benchmark contract is unavailable or incompatible", 409
            ) from exc

    def benchmark_report(self):
        path = self.benchmark / "report.json"
        try:
            if sha(self.benchmark / "contract.json") != self.benchmark_contract_sha256:
                raise ValueError("Benchmark contract changed after startup")
            raw = self.validate_benchmark(json.loads(path.read_text()))
            if (
                raw.get("version") != VERSION
                or raw.get("contract_fingerprint")
                != self.benchmark_contract["fingerprint"]
                or raw.get("protocol") != self.benchmark_contract["protocol"]
                or ("identity" in raw and raw["identity"] != self.benchmark_identity)
            ):
                raise ValueError("Benchmark report belongs to another contract")
            value = public_benchmark(raw)
            if self.benchmark_audit is not None and self.benchmark_audit.is_file():
                audit = json.loads(self.benchmark_audit.read_text())
                if (
                    audit.get("contract_fingerprint")
                    != self.benchmark_contract["fingerprint"]
                    or audit.get("report_sha256") != sha(path)
                    or (
                        "report_fingerprint" in audit
                        and audit["report_fingerprint"] != raw.get("fingerprint")
                    )
                ):
                    raise ValueError("Benchmark independent audit binding changed")
                checks = audit.get("checks", audit.get("checks_count"))
                if audit.get("status") == "passed" and (
                    isinstance(checks, bool)
                    or not isinstance(checks, int)
                    or checks <= 0
                    or audit.get("failures")
                ):
                    raise ValueError("Passed benchmark audit lacks completed checks")
                value["audit"] = fields(
                    audit, ("status", "checks", "checks_count", "scope", "limitations")
                )
            return value
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RequestError(
                "Benchmark report is unavailable or failed source/contract verification",
                409,
            ) from exc

    def report(self):
        value = super().report()
        try:
            value["sweep_rl_benchmark"] = self.benchmark_report()
        except RequestError as exc:
            value["sweep_rl_error"] = str(exc)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference",
        "assets",
        "gpu-lock",
        "cache",
        "state-dir",
        "study-dir",
        "benchmark-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parent-study-dir", type=Path)
    parser.add_argument("--benchmark-audit", type=Path)
    parser.add_argument(
        "--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"]
    )
    parser.add_argument("--port", type=int, default=18788)
    args = parser.parse_args()
    ref = load_reference(args.reference, args.assets)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    report, scene = (
        args.state_dir / "planner-report.json",
        args.state_dir / "ui-scene.json",
    )
    atomic(report, public_state(ref))
    atomic(scene, ui_scene(ref))
    for key, path in (
        ("REFERENCE", args.reference),
        ("ASSETS", args.assets),
        ("LOCK", args.gpu_lock),
        ("CACHE", args.cache),
    ):
        os.environ["SIRCUS_SWEEP_" + key] = str(path.resolve())
    state = BenchmarkState(
        ref,
        args.study_dir,
        report,
        scene,
        "sircus.mjlab.sweep_interface:solve",
        args.benchmark_dir,
        parent_study=args.parent_study_dir,
        benchmark_audit=args.benchmark_audit,
    )
    state.grid_report()
    state.benchmark_report()
    server = ThreadingHTTPServer((args.host, args.port), grid_handler(state))
    print(
        json.dumps(
            dict(
                url=f"http://{args.host}:{args.port}/#sweepRlBenchmark",
                simulation_only=True,
                benchmark_read_only=True,
            )
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
