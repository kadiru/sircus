"""CPU-only comparison of original and batched geometry checks on fixed inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def _poses(value, inherited_bucket):
    if isinstance(value, dict):
        bucket = value.get("bucket_xy_m", inherited_bucket)
        if "q10" in value and "q14" in value:
            q = np.r_[value["q10"], value["q14"]].astype(float)
            if q.shape == (12,) and np.isfinite(q).all():
                yield q, np.asarray(bucket, float)
        for child in value.values():
            yield from _poses(child, bucket)
    elif isinstance(value, list):
        for child in value:
            yield from _poses(child, inherited_bucket)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="Scene JSON or contract with scene key")
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--poses", type=Path, action="append", default=[])
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 4 or args.rounds < 1:
        parser.error("At least four samples and one timing round required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from .arm_check_speed import VERSION, install_vectorized_gate
    from .transit_audit import GeometricGate, audit_segment
    scene = json.loads(args.scene.read_text())
    scene = scene.get("scene", scene)
    home = json.loads(args.home.read_text())
    bucket = np.asarray(scene["bucket"]["center_xy_m"], float)
    home_q = np.r_[home["q10"], home["q14"]].astype(float)
    source_samples = [(home_q, bucket)]
    for path in args.poses:
        source_samples.extend(_poses(json.loads(path.read_text()), bucket))
    source_samples.extend(_poses(home, bucket))
    samples, seen = [], set()
    for q, xy in source_samples:
        key = (*q, *xy)
        if key not in seen:
            seen.add(key)
            samples.append((q, xy))
        if len(samples) >= args.samples // 2:
            break
    measured_count = len(samples)
    rng = np.random.default_rng(20261121)
    while len(samples) < args.samples:
        q, xy = samples[len(samples) % measured_count]
        samples.append((q + rng.uniform(-.12, .12, 12), xy.copy()))
    original = GeometricGate(scene, args.assets, bucket)
    optimized = install_vectorized_gate(GeometricGate(scene, args.assets, bucket))
    records, mismatches = [], []
    for index, (q, xy) in enumerate(samples):
        original.bucket_xy = optimized.bucket_xy = xy
        expected, actual = original.check(q), optimized.check(q)
        records.append(expected)
        if expected != actual:
            mismatches.append(dict(index=index, original=expected, optimized=actual))
    segment_mismatches = []
    # Both directions and unchanged sample spacing, around an existing input.
    delta = np.linspace(-.03, .03, 12)
    for index, (q, xy) in enumerate(samples[:4]):
        original.bucket_xy = optimized.bucket_xy = xy
        for reverse in (False, True):
            a, b = (q + delta, q) if reverse else (q, q + delta)
            expected = audit_segment(original, a, b)
            actual = audit_segment(optimized, a, b)
            if expected != actual:
                segment_mismatches.append(dict(index=index, reverse=reverse, original=expected, optimized=actual))
    timings = {"original": [], "vectorized": []}
    for round_index in range(args.rounds):
        order = (("original", original), ("vectorized", optimized))
        if round_index % 2:
            order = order[::-1]
        for name, gate in order:
            start = time.perf_counter()
            for q, xy in samples:
                gate.bucket_xy = xy
                gate.check(q)
            timings[name].append(time.perf_counter() - start)
    means = {name: float(np.mean(values)) for name, values in timings.items()}
    output = dict(version=VERSION, cpu_only=True, samples=len(samples), source_samples=measured_count,
        synthetic_joint_perturbation_samples=len(samples)-measured_count, rounds=args.rounds,
        checked_pairs=len(original.pairs), passed=sum(bool(row["passed"]) for row in records),
        rejected=sum(not row["passed"] for row in records),
        mismatch_count=len(mismatches), segment_mismatch_count=len(segment_mismatches),
        mismatches=mismatches, segment_mismatches=segment_mismatches,
        exact_full_record_equivalence=not mismatches and not segment_mismatches,
        timing_seconds=timings, mean_batch_seconds=means,
        mean_check_ms={name: seconds / len(samples)*1000 for name, seconds in means.items()},
        speedup=means["original"] / means["vectorized"],
        validation_model_sha256=original.validation_model_sha256,
        source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [args.scene, args.home, *args.poses]},
        limitations=["CPU geometry checks only; no simulation drops, policy performance or path dynamics claim",
            "Synthetic bounded joint perturbations supplement saved inputs; they are not demonstrated shot recipes"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: output[key] for key in ("samples", "passed", "rejected", "checked_pairs",
        "exact_full_record_equivalence", "mean_check_ms", "speedup")}, indent=2))
    if mismatches or segment_mismatches:
        raise SystemExit("Original and optimized checkers differed; do not deploy")


if __name__ == "__main__":
    main()
