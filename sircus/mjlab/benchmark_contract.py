"""Matched warm-started sweep/PPO study; no hardware or implicit deployment."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .corrected_contract import PHYSICS, NOISE, grid_layouts, sha
from .expanded_actions import WIDE_BOUNDS
from .learning_contract import digest, reward

VERSION = "sircus-sweep-rl-benchmark-v1"
SPLIT_SEED = 20261303
SEEDS = [20261311, 20261312, 20261313]
METHODS = [
    {"id": "sweep", "label": "Guided sweep"},
    {"id": "ppo", "label": "Residual PPO"},
]
MILESTONES = [0, 256, 512, 1024]
RESIDUAL_SCALE = np.asarray([0.04, 0.04, 0.05, 8.0, 0.12])
NAMES = ["board_x_m", "board_y_m", "board_height_m", "tilt_deg", "gap_m"]
CRITICAL = (
    "benchmark_contract.py",
    "benchmark_adapter.py",
    "benchmark_policy.py",
    "benchmark_resources.py",
    "benchmark_run.py",
    "benchmark_study.py",
    "learning_policy.py",
)
SCOPE = (
    "Warm-started restricted continuous five-parameter shot selection with the "
    "current fixed ball. Same 17 historical anchors for both methods. Prescribed "
    "held endpoints and sampled common-home/local paths; no grasp, driven motion, "
    "impact compliance, perception or physical validation."
)


def study_identity(ref):
    source = dict(ref["source_hashes"])
    root = Path(__file__).resolve().parents[2]
    for name in CRITICAL:
        relative = "sircus/mjlab/" + name
        source[relative] = sha(root / relative)
    return dict(
        contract_fingerprint=ref["contract"]["fingerprint"],
        scene_sha256=digest(ref["scene"]),
        asset_sha256=ref["asset_hashes"],
        source_sha256=source,
        physics=copy.deepcopy(PHYSICS),
    )


def _read(path):
    return json.loads(Path(path).read_text())


def _contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Prior manifest path escapes its bundle")
    return path


def load_prior(bundle):
    """Read only the published, audited current bank; discard all test outcomes."""
    bundle = Path(bundle)
    for line in (bundle / "manifest.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha(_contained(bundle, name)) != expected:
            raise ValueError("Shared prior bundle changed: " + name)
    reports = {}
    for kind in ("parent", "followup"):
        directory = bundle / kind
        for name, expected in _read(directory / "manifest.json").items():
            if sha(_contained(directory, name)) != expected:
                raise ValueError("Prior compact manifest mismatch")
        report, audit = _read(directory / "report.json"), _read(
            directory / "audit.json"
        )
        contract = _read(directory / "contract.json")
        if (
            report["status"] != "completed"
            or audit["status"] != "passed"
            or audit["failures"]
            or audit["report_sha256"] != sha(directory / "report.json")
            or audit["contract_fingerprint"] != contract["fingerprint"]
        ):
            raise ValueError("Shared prior is incomplete or unaudited")
        reports[kind] = {c["id"]: c for c in report["cells"]}
    records = []
    expected_train = {tuple(c["bucket_xy_m"]) for c in grid_layouts()}
    for row in _read(bundle / "followup/selected-settings.json"):
        selected, configuration = row["selected"], row["configuration"]
        kind = "followup" if selected["source"] == "fresh_study" else "parent"
        source = reports[kind][row["id"]]
        expected = selected["configuration_sha256"]
        if (
            digest(configuration) != expected
            or source["selected"]["configuration_sha256"] != expected
        ):
            raise ValueError("Prior configuration binding changed")
        frozen = source["frozen_validation"]
        if (
            frozen["configuration_sha256"] != expected
            or frozen["status"] != "completed"
        ):
            raise ValueError("Prior frozen evidence does not match its configuration")
        if tuple(row["bucket_xy_m"]) not in expected_train:
            raise ValueError("Prior contains a non-training layout")
        values = [
            *configuration["board_center"],
            configuration["tilt_deg"],
            configuration["gap_m"],
        ]
        record = dict(
            id=row["id"],
            bucket_xy_m=row["bucket_xy_m"],
            values=values,
            q10=configuration["q10"],
            q14=configuration["q14"],
            configuration_sha256=expected,
            source_study=kind,
        )
        for key, size in (("bucket_xy_m", 2), ("values", 5), ("q10", 6), ("q14", 6)):
            vector = np.asarray(record[key], float)
            if vector.shape != (size,) or not np.isfinite(vector).all():
                raise ValueError("Invalid prior geometry")
        if not np.all(
            (np.asarray(values) >= WIDE_BOUNDS["low"])
            & (np.asarray(values) <= WIDE_BOUNDS["high"])
        ):
            raise ValueError("Prior lies outside matched global action bounds")
        record["source_record_sha256"] = digest(record)
        records.append(record)
    if len(records) != 17 or len({r["id"] for r in records}) != 17:
        raise ValueError("Expected exactly the 17 audited shared recipes")
    records.sort(key=lambda r: r["id"])
    return dict(
        records=records,
        records_sha256=digest(records),
        bundle_manifest_sha256=sha(bundle / "manifest.sha256"),
        recipes=len(records),
        latest_study_native_releases=4234,
        latest_study_requested_trials=4243,
        full_historical_cost_known=False,
        scope=(
            "Shared historical starting information, not 17 free cold-start labels. "
            "Latest two studies cost 4234 native releases; predecessor acquisition, "
            "contact tuning and human design costs are not fully quantified."
        ),
    )


def make_splits():
    train = [dict(c, split="train") for c in grid_layouts()]
    used = np.asarray([c["bucket_xy_m"] for c in train])
    result = {"train": train}
    for role, count, seed in (
        ("validation", 12, SPLIT_SEED),
        ("test", 24, SPLIT_SEED + 1),
    ):
        rng, rows = np.random.default_rng(seed), []
        for _ in range(100000):
            if len(rows) == count:
                break
            xy = rng.uniform([0.1, -0.65], [1.24, 0.25])
            if np.linalg.norm(used - xy, axis=1).min() < 0.005:
                continue
            rows.append(
                dict(
                    id=f"bench-{role}-{len(rows):02d}",
                    split=role,
                    bucket_xy_m=xy.tolist(),
                )
            )
            used = np.vstack([used, xy])
        if len(rows) != count:
            raise RuntimeError("Could not freeze separated layout pools")
        result[role] = rows
    return result


def anchor_for(bucket, records):
    xy = np.asarray(bucket, float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Finite bucket XY required")
    # No outcome query, candidate ranking, interpolation or retry at inference.
    return copy.deepcopy(
        min(
            records,
            key=lambda r: (
                float(np.linalg.norm(np.asarray(r["bucket_xy_m"]) - xy)),
                r["id"],
            ),
        )
    )


def observation(bucket, anchor):
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    return np.r_[
        (np.asarray(bucket) - [0.67, -0.20]) / [0.57, 0.45],
        (np.asarray(anchor["bucket_xy_m"]) - [0.67, -0.20]) / [0.57, 0.45],
        2 * (np.asarray(anchor["values"]) - low) / (high - low) - 1,
    ].astype(np.float32)


def action_from_latent(anchor_values, latent):
    return (
        np.asarray(anchor_values, float)
        + np.tanh(np.asarray(latent, float)) * RESIDUAL_SCALE
    )


def action_from_residual(anchor_values, residual):
    residual = np.asarray(residual, float)
    if (
        residual.shape != (5,)
        or not np.isfinite(residual).all()
        or np.max(np.abs(residual)) > 1 + 1e-12
    ):
        raise ValueError("Five bounded normalized residuals required")
    return np.asarray(anchor_values, float) + residual * RESIDUAL_SCALE


def schedule(seed, budget=1024):
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < budget:
        rows.extend(rng.permutation(35).tolist())
    return rows[:budget]


def perturbation(context_id, repeat):
    if repeat == 0:
        return np.zeros(5)
    seed = int(
        hashlib.sha256(f"{SPLIT_SEED}:{context_id}:{repeat}".encode()).hexdigest()[:16],
        16,
    )
    return np.random.default_rng(seed).normal(0, NOISE)


def statistics(rows):
    captures = sum(
        bool(
            r.get("native_release") and r.get("route_success") and not r.get("invalid")
        )
        for r in rows
    )
    return dict(
        attempts=len(rows),
        captures=captures,
        native_releases=sum(bool(r.get("native_release")) for r in rows),
        invalid=sum(bool(r.get("invalid")) for r in rows),
        clean=sum(
            bool(
                r.get("native_release")
                and not r.get("invalid")
                and r.get("route_quality") == 2
            )
            for r in rows
        ),
        mean_reward=float(np.mean([reward(r) for r in rows])) if rows else 0.0,
    )


def protocol(*, seeds=None, budget=1024, milestones=None):
    seeds = SEEDS if seeds is None else list(seeds)
    milestones = MILESTONES if milestones is None else list(milestones)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or milestones != sorted(set(milestones))
        or milestones[0] != 0
        or milestones[-1] != budget
    ):
        raise ValueError(
            "Unique seeds and ordered milestones from zero to equal budget required"
        )
    if budget < 32 or budget > 4096 or budget % 32 or any(m % 32 for m in milestones):
        raise ValueError("Finite budget/milestones must be multiples of native batch32")
    return dict(
        methods=copy.deepcopy(METHODS),
        seeds=seeds,
        milestones=milestones,
        sample_budget=budget,
        batch_size=32,
        run_order=[
            dict(method=method, seed=seed)
            for i, seed in enumerate(seeds)
            for method in (["sweep", "ppo"] if i % 2 == 0 else ["ppo", "sweep"])
        ],
        action_names=NAMES,
        action_bounds=copy.deepcopy(WIDE_BOUNDS),
        residual_scale=RESIDUAL_SCALE.tolist(),
        scope=SCOPE,
        family="Nearest shared TRAIN teacher, fixed source board XYZ/tilt/gap, query-facing heading; continuous bounded residuals, no global search or candidate-index policy",
        anchor_rule="Nearest source bucket Euclidean XY, tie source ID; fixed before outcomes; no fallback if geometry/path fails",
        observations="9 geometry-only values: bucketXY, source bucketXY, five source action values; fixed normalization",
        sweep="One request per candidate: zero, ten axis offsets, ten seeded space-filling samples, then TRAIN-best latent jitter. Heldout prediction transfers nearest TRAIN context's best normalized residual; no outcome query.",
        ppo=dict(
            network=[9, 64, 64, 5],
            critic=[9, 64, 64, 1],
            optimizer="CPU Adam",
            lr=0.0003,
            epochs=6,
            clip=0.2,
            value_weight=0.5,
            entropy_weight=0.001,
            gradient_norm=1.0,
            initial_logstd=-1.5,
            logstd_bounds=[-3.0, -0.7],
            episode="One terminal shot decision; no GAE/bootstrap; actual native reward updates",
        ),
        reward="Unchanged learning_contract.reward: invalid -1, native board/entry/proximity/retention shaping; success remains the independent board-route capture rule",
        sample_definition="Every requested training decision, including bounds/IK/home/path rejects; native releases and CPU proposals reported separately",
        evaluation=dict(
            validation_layouts=12,
            test_layouts=24,
            repeats=3,
            perturbations="One nominal + two paired Gaussian placement jitters per layout; same offsets across methods/seeds/milestones",
            selection="Validation captures, then mean reward, then earliest milestone; test only after selected artifact frozen",
            test="One deterministic base action per layout; no search/retry/adaptation at evaluation",
            full_denominator="All layouts and requested repetitions, including blocked/unresolved/rejected",
        ),
        resources=dict(
            wall_s="Setup + training wall time, excluding validation/test",
            cpu_s="Current worker process CPU time; external nvidia-smi telemetry subprocess CPU excluded",
            gpu_s="Synchronized native pipeline elapsed wall time, not active kernel utilization",
            gpu_reservation_s="Elapsed exclusive 4090 worker reservation including evaluation",
            memory="MiB; lifetime process peaks include evaluation; NVIDIA process VRAM sampled every1s, PyTorch allocator separately",
            storage="Algorithm artifacts separate from audit raw-data storage",
        ),
    )


def make_contract(ref, prior, *, seeds=None, budget=1024, milestones=None):
    p = protocol(seeds=seeds, budget=budget, milestones=milestones)
    p["prior"] = {k: copy.deepcopy(v) for k, v in prior.items() if k != "records"}
    value = dict(
        version=VERSION,
        identity=study_identity(ref),
        protocol=p,
        splits=make_splits(),
        prior_sha256=digest(prior),
        packages=copy.deepcopy(ref["packages"]),
    )
    value["fingerprint"] = digest(value)
    return value


def validate_report(raw):
    if not isinstance(raw, dict):
        raise ValueError("Benchmark report must be a mapping")
    report = copy.deepcopy(raw)
    unsigned = dict(report)
    fingerprint = unsigned.pop("fingerprint", None)
    if fingerprint != digest(unsigned) or report.get("version") != VERSION:
        raise ValueError("Benchmark report fingerprint/version mismatch")
    if report.get("status") not in {
        "queued",
        "running",
        "auditing",
        "completed",
        "failed",
    }:
        raise ValueError("Unknown benchmark status")
    contract_fp = report.get("contract_fingerprint", "")
    if len(contract_fp) != 64 or any(c not in "0123456789abcdef" for c in contract_fp):
        raise ValueError("Benchmark lacks a contract binding")
    p = report["protocol"]
    if (
        p["methods"] != METHODS
        or not p["seeds"]
        or p["milestones"][-1] != p["sample_budget"]
    ):
        raise ValueError("Benchmark protocol shape changed")
    seen = set()
    for point in report.get("curves", []):
        key = (point["method"], point["seed"], point["samples"])
        if (
            key in seen
            or key[0] not in {"sweep", "ppo"}
            or key[1] not in p["seeds"]
            or key[2] not in p["milestones"]
        ):
            raise ValueError("Duplicate or out-of-contract curve point")
        seen.add(key)
        counts = point["validation"]
        if (
            counts["attempts"]
            != p["evaluation"]["validation_layouts"] * p["evaluation"]["repeats"]
        ):
            raise ValueError("Validation curve lacks every requested layout/repetition")
        for name in ("attempts", "captures", "native_releases", "invalid", "clean"):
            value = counts[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Invalid benchmark count")
        if (
            counts["captures"] > counts["native_releases"]
            or counts["native_releases"] > counts["attempts"]
            or counts["clean"] > counts["captures"]
        ):
            raise ValueError("Benchmark count exceeds its denominator")
        for value in point.get("resources", {}).values():
            if (
                value is not None
                and isinstance(value, (int, float))
                and (not math.isfinite(value) or value < 0)
            ):
                raise ValueError("Invalid resource observation")
    if report["status"] == "completed":
        expected = len(p["methods"]) * len(p["seeds"])
        if len(report.get("final_test", [])) != expected or len(seen) != expected * len(
            p["milestones"]
        ):
            raise ValueError("Completed benchmark lacks every curve/final cohort")
        expected_pairs = {(m["id"], seed) for m in p["methods"] for seed in p["seeds"]}
        actual_pairs = set()
        for final in report["final_test"]:
            pair = (final["method"], final["seed"])
            if pair not in expected_pairs or pair in actual_pairs:
                raise ValueError("Duplicate or undeclared final method/seed")
            actual_pairs.add(pair)
            if final.get("selected_samples") not in p["milestones"]:
                raise ValueError("Final evaluation has an undeclared selected budget")
            counts = final["counts"]
            expected_attempts = (
                p["evaluation"]["test_layouts"] * p["evaluation"]["repeats"]
            )
            if counts["attempts"] != expected_attempts:
                raise ValueError("Final evaluation lacks every requested test trial")
            if any(
                isinstance(counts[k], bool)
                or not isinstance(counts[k], int)
                or counts[k] < 0
                for k in ("attempts", "captures", "native_releases", "invalid", "clean")
            ):
                raise ValueError("Invalid final evaluation counts")
            if (
                not counts["clean"]
                <= counts["captures"]
                <= counts["native_releases"]
                <= counts["attempts"]
            ):
                raise ValueError("Final counts exceed their denominator")
            h = final.get("frozen_artifact_sha256", "")
            if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("Final evaluation lacks its frozen artifact")
        if actual_pairs != expected_pairs:
            raise ValueError("Final evaluation method/seed coverage changed")
    return report
