"""Versioned calibrated current-ball benchmark; v1 evidence stays historical.

The scorer, reward, setup resolver and independently verified calibration are
explicit frozen inputs. Geometry-only setup resolution precedes prediction.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .benchmark_contract import (
    METHODS, MILESTONES, NAMES, RESIDUAL_SCALE, action_from_latent,
    action_from_residual, load_prior, observation,
)
from .benchmark_contract import protocol as v1_protocol
from .corrected_contract import PHYSICS, NOISE, grid_layouts, sha
from .contract import FORCE_NAMES
from .learning_contract import digest

VERSION = "sircus-calibrated-sweep-rl-benchmark-v2"
AUDIT_VERSION = "sircus-calibrated-sweep-rl-independent-audit-v2"
REWARD_VERSION = "sircus-calibrated-shot-reward-v2"
SCORER_VERSION = "sircus-calibrated-native-capture-v2"
SPLIT_SEED = 20261403
SEEDS = [20261411, 20261412, 20261413]
BUDGET = 1024
PRIOR_PATH = "baselines/mjlab-grid35-expanded-20261103"
V1_BASELINE = "baselines/mjlab-sweep-rl-benchmark-20261103"
CRITICAL = (
    "calibrated_benchmark_contract.py", "calibrated_benchmark_run.py",
    "calibrated_benchmark_study.py", "calibrated_benchmark_smoke.py",
    "calibrated_adapter.py", "calibrated_admission.py", "calibrated_env.py",
    "calibrated_score.py", "calibrated_score_oracle.py", "calibrated_native_log.py",
    "capture_calibration.py", "benchmark_adapter.py", "benchmark_contract.py", "benchmark_policy.py",
    "benchmark_resources.py", "learning_policy.py",
)
SCOPE = (
    "Current fixed-ball, privileged-geometry, five-parameter residual shot selection "
    "with independently calibrated capture scoring and geometry-only setup "
    "resolution before prediction. Same 17 historical recipes; prescribed held "
    "endpoints and sampled setup paths; no learned grasp/transit/compliance or "
    "physical validation."
)


def read(path):
    return json.loads(Path(path).read_text())


def checked_calibration(calibration, verification, *, synthetic=False):
    """Require separately hash-bound control calibration and verification.

    Input files are copied into every new study. Host paths do not enter the
    contract. A synthetic fixture is accepted only by explicit test-kind mode;
    production launchers never opt into it implicitly.
    """
    a, v = read(calibration), read(verification)
    for value, label in ((a, "calibration"), (v, "verification")):
        if value.get("status") != "passed" or value.get("failures") not in (None, []):
            raise ValueError(label + " controls did not pass")
        count = value.get("checks", value.get("checks_count"))
        if type(count) is not int or count <= 0:
            raise ValueError(label + " controls lack positive completed checks")
        if bool(value.get("synthetic_fixture", False)) != bool(synthetic):
            raise ValueError(label + " evidence kind does not match study kind")
    if v.get("calibration_sha256") != sha(calibration):
        raise ValueError("Verification does not bind the exact calibration bytes")
    config = copy.deepcopy(a.get("score_config"))
    if not isinstance(config, dict) or not config.get("version"):
        raise ValueError("Calibrated score configuration/version required")
    from .calibrated_score import validate_calibration
    validate_calibration(config)
    if not synthetic and config.get("native_dt_s", 1/15360) != 1/15360:
        raise ValueError("Native calibration sampling differs from the fixed engine")
    if a.get("score_config_sha256", digest(config)) != digest(config):
        raise ValueError("Calibration score configuration fingerprint changed")
    if v.get("score_config_sha256") != digest(config):
        raise ValueError("Verification score configuration differs")
    return dict(
        calibration_sha256=sha(calibration), verification_sha256=sha(verification),
        score_config=copy.deepcopy(config), score_config_sha256=digest(config),
        calibration_version=a.get("version"), verification_version=v.get("version"),
        calibration_checks=a.get("checks", a.get("checks_count")),
        verification_checks=v.get("checks", v.get("checks_count")),
        synthetic_fixture=bool(synthetic),
    )


def historical_coordinates(repository):
    """Exclude every v1 TRAIN/validation/test coordinate from fresh evaluation."""
    splits = read(Path(repository) / V1_BASELINE / "splits.json")
    rows = [list(c["bucket_xy_m"]) for role in ("train", "validation", "test") for c in splits[role]]
    if len(rows) != 71 or len({tuple(c) for c in rows}) != 71:
        raise ValueError("Published v1 coordinate exclusion set changed")
    return rows


def make_splits(repository):
    train = [dict(c, split="train") for c in grid_layouts()]
    excluded = historical_coordinates(repository)
    used = np.asarray(excluded, dtype=float)
    result = {"train": train}
    for role, count, seed in (("validation", 12, SPLIT_SEED), ("test", 24, SPLIT_SEED + 1)):
        rng, rows = np.random.default_rng(seed), []
        for _ in range(100000):
            if len(rows) == count:
                break
            xy = rng.uniform([0.1, -0.65], [1.24, 0.25])
            if np.linalg.norm(used - xy, axis=1).min() < 0.005:
                continue
            rows.append(dict(id=f"calibrated-{role}-{len(rows):02d}", split=role, bucket_xy_m=xy.tolist()))
            used = np.vstack([used, xy])
        if len(rows) != count:
            raise RuntimeError("Could not freeze fresh separated layout pools")
        result[role] = rows
    return result


def schedule(seed, budget=BUDGET):
    rng, rows = np.random.default_rng(seed), []
    while len(rows) < budget:
        rows.extend(rng.permutation(35).tolist())
    return rows[:budget]


def perturbation(context_id, repeat):
    if repeat == 0:
        return np.zeros(5)
    seed = int(hashlib.sha256(f"{SPLIT_SEED}:{context_id}:{repeat}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).normal(0, NOISE)


def reward(row):
    # Executed reward has its own versioned source; the auditor independently
    # reconstructs the same declared coefficients from calibrated row semantics.
    from .calibrated_score import reward as calibrated_reward
    return calibrated_reward(row)


def statistics(rows):
    captures = sum(bool(r.get("native_release") and r.get("route_success") and not r.get("invalid")) for r in rows)
    return dict(attempts=len(rows), captures=captures,
                native_releases=sum(bool(r.get("native_release")) for r in rows),
                invalid=sum(bool(r.get("invalid")) for r in rows),
                clean=sum(bool(r.get("native_release") and not r.get("invalid") and r.get("route_quality") == 2) for r in rows),
                mean_reward=float(np.mean([reward(r) for r in rows])) if rows else 0.0)


def admission_summary(rows):
    reasons={}
    for row in rows:
        if not row.get("native_release"):
            key=str(row.get("invalid") or "unspecified_setup_reject")
            reasons[key]=reasons.get(key,0)+1
    return dict(requested=len(rows),native_releases=sum(bool(r.get("native_release")) for r in rows),
                rejected=sum(not r.get("native_release") for r in rows),reasons=reasons,
                fixed_base_rejected=sum(bool(r.get("context_hard_reject")) for r in rows),
                anchor_unresolved=sum(not r.get("context_anchor_admitted") for r in rows),
                invalid_physics=sum(bool(r.get("invalid_physics")) for r in rows))


def protocol(*, seeds=None, budget=BUDGET, milestones=None, synthetic=False):
    seeds = SEEDS if seeds is None else list(seeds)
    milestones = MILESTONES if milestones is None else list(milestones)
    if not synthetic and (seeds != SEEDS or budget != BUDGET or milestones != MILESTONES):
        raise ValueError("Production v2 requires canonical three-seed 1024-request protocol")
    p = v1_protocol(seeds=seeds, budget=budget, milestones=milestones)
    p["evaluation"]["test"] = "One deterministic base action per layout; same bounded geometry-only setup resolution before policy; no ball-flight search/retry/adaptation"
    p.update(scope=SCOPE, study_kind="synthetic_fixture" if synthetic else "native_current_ball_v2",
             family="Geometry-resolved shared TRAIN anchor, including bounded geometry-only derivation when needed, before observation; bounded five-parameter residual; no action fallback after prediction",
             anchor_rule="Frozen nearest/id source order, 17 nominal anchors then declared coupled geometry offsets, at most64 endpoint/path candidates; select before prediction; original source chain and effective anchor recorded; no flight labels or postprediction replacement",
             observations="9 geometry-only values: query bucketXY, original source bucketXY, five effective anchor parameters; fixed normalization",
             reward_version=REWARD_VERSION, scorer_version=SCORER_VERSION,
             reward="Calibrated-shot reward-v2: same declared shaping coefficients, calibrated row semantics; invalid requests -1",
             setup_resolution="Geometry-only resolution before observation and action, separately record CPU/path/candidate costs; fresh per-method/seed caches; no hidden native query",
             evaluation_history="New coordinates exclude all 71 v1 coordinates by >=5mm; unconditional full-table sampling, no admission-conditioned resampling")
    return p


def checked_setup_home(path):
    home = read(path)
    if not isinstance(home, dict) or not isinstance(home.get("version"), str):
        raise ValueError("Versioned independently checked setup home required")
    for key in ("q10", "q14"):
        vector = np.asarray(home.get(key), dtype=float)
        if vector.shape != (6,) or not np.isfinite(vector).all():
            raise ValueError("Setup home requires finite six-joint poses")
    from .calibrated_admission import anchor_search_spec, VERSION as RESOLVER_VERSION
    return dict(home=copy.deepcopy(home), home_file_sha256=sha(path),
                home_record_sha256=digest(home),
                resolver=RESOLVER_VERSION, anchor_search=anchor_search_spec(),
                geometry_candidate_budget=64, rrt_nodes=192,
                source_order="nearest original source bucket XY then source ID",
                no_flight_labels=True, no_postprediction_action_fallback=True)


def trace_schema(*, synthetic=False):
    return dict(version="sircus-synthetic-interval-trace-fixture-v2" if synthetic else "sircus-dense-native-interval-trace-v2",
                position_m="(T+1,N,3), includes time zero",
                velocity_m_s="(T+1,N,3), includes time zero",
                forces_N="(T,N,6), interval contact-force observations",
                force_names=list(FORCE_NAMES), time_s="(T+1,), canonical dt-times-tick native times",
                native_time_s="(T+1,N), observed native GPU clock with drift retained",
                contact_distance_m="(T,N,2), nearest ball/wall and ball/base contact distance, +inf absent",
                contact_geom_ids="(T,N,2,2), native geom pair IDs, -1 absent",
                contact_position_m="(T,N,2,3), native contact positions, zeros absent",
                contact_normal="(T,N,2,3), native contact frame normals, zeros absent",
                contact_alignment="Interval contact fields correspond to PREstate position_m[k], verified independently against native geomDistance",
                time_provenance="time_s is tick-derived; native_time_s is the recorded engine clock",
                native_dt_s=.001 if synthetic else PHYSICS["native_dt_s"], horizon_s=1.6 if synthetic else PHYSICS["horizon_s"],
                scoring_dt_s=.001 if synthetic else PHYSICS["native_dt_s"],
                alignment="forces[i] belongs to native interval [time[i],time[i+1]]")


def study_identity(ref, calibration, setup, *, repository=None, synthetic=False):
    root = Path(repository) if repository is not None else Path(__file__).resolve().parents[2]
    sources = dict(ref["source_hashes"])
    for name in CRITICAL:
        relative = "sircus/mjlab/" + name
        sources[relative] = sha(root / relative)
    return dict(contract_fingerprint=ref["contract"]["fingerprint"],
                scene_sha256=digest(ref["scene"]), asset_sha256=ref["asset_hashes"],
                source_sha256=sources, physics=dict(copy.deepcopy(PHYSICS),
                    native_engine_version=PHYSICS["version"],
                    version="sircus-calibrated-articulated-passive-contact-15360hz-v2",
                    scoring_dt_s=PHYSICS["native_dt_s"], scorer=SCORER_VERSION),
                reward_version=REWARD_VERSION, scorer_version=SCORER_VERSION, calibration=copy.deepcopy(calibration),
                setup=copy.deepcopy(setup), trace_schema=trace_schema(synthetic=synthetic))


def make_contract(ref, prior, calibration, setup, *, repository=None, seeds=None, budget=BUDGET, milestones=None, synthetic=False):
    root = Path(repository) if repository is not None else Path(__file__).resolve().parents[2]
    p = protocol(seeds=seeds, budget=budget, milestones=milestones, synthetic=synthetic)
    p["prior"] = {k: copy.deepcopy(v) for k, v in prior.items() if k != "records"}
    excluded = historical_coordinates(root)
    from .calibrated_admission import validate_home
    validate_home(setup["home"], ref["scene"])
    value = dict(version=VERSION, identity=study_identity(ref, calibration, setup, repository=root, synthetic=synthetic),
                 protocol=p, splits=make_splits(root), prior_sha256=digest(prior),
                 historical_exclusion_sha256=digest(excluded), historical_exclusion_count=len(excluded),
                 packages=copy.deepcopy(ref["packages"]))
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
