"""CPU-only explanation of frozen VALIDATION retention, without relabeling shots.

This is an evidence diagnostic, not another evaluator or tolerance search. It
reads only published, complete validation cohorts and their saved traces. No
physics engine, Torch policy, GPU, rollout, test prediction or test outcome is
loaded. Output is a new external file; immutable study evidence is never edited.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .contract import FORCE_NAMES, ScoreStream
from ..isaac.board_route import FORCE_THRESHOLD_N
from ..isaac.rl_contract import GEOMETRY_EPS_M

VERSION = "sircus-validation-wall-diagnostic-v2"
DT_S = 1 / 240
HORIZON_STEPS = 960
ROOT = Path(__file__).resolve().parents[2]
SCORER_FILES = (
    "sircus/mjlab/contract.py",
    "sircus/isaac/rl_contract.py",
    "sircus/isaac/entry_quality.py",
    "sircus/isaac/board_route.py",
)
TRACE_SOURCE_FILES = (
    "sircus/mjlab/contact_env.py",
    "sircus/mjlab/sweep_worker.py",
)
ORIGINAL_FIELDS = (
    "success",
    "complete_success",
    "route_success",
    "route_quality",
    "board_bounce",
    "whole_ball_entry",
    "max_retention_s",
    "board_contact_episodes",
    "external_contact_steps",
    "single_bounce_success",
    "physics_steps",
    "entry_class",
    "rim_band_contact",
    "bucket_contact_before_entry",
    "exited_after_entry",
)


class DiagnosticError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DiagnosticError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _value_end(text, start):
    """Skip a JSON value lexically, without decoding excluded outcome objects."""
    depth, quoted, escaped, index = 0, False, False, start
    while index < len(text):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            if depth == 0:
                return index
            depth -= 1
        elif char == "," and depth == 0:
            return index
        index += 1
    require(not quoted and depth == 0, "Unterminated JSON value")
    return index


def selected_members(text, names):
    """Decode named top-level fields only; notably never decode final_test."""
    decoder = json.JSONDecoder()
    index, answer, seen = 0, {}, set()

    def whitespace(offset):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        return offset

    index = whitespace(index)
    require(index < len(text) and text[index] == "{", "Expected JSON object")
    index = whitespace(index + 1)
    while index < len(text) and text[index] != "}":
        key, index = decoder.raw_decode(text, index)
        require(
            isinstance(key, str) and key not in seen, "Duplicate/non-string JSON key"
        )
        seen.add(key)
        index = whitespace(index)
        require(text[index] == ":", "Missing JSON colon")
        start = whitespace(index + 1)
        end = _value_end(text, start)
        if key in names:
            answer[key] = json.loads(text[start:end])
        index = whitespace(end)
        require(index < len(text), "Unterminated JSON object")
        if text[index] == "}":
            break
        require(text[index] == ",", "Missing JSON comma")
        index = whitespace(index + 1)
    require(index < len(text) and text[index] == "}", "Unterminated JSON object")
    require(not text[index + 1 :].strip(), "Trailing JSON content")
    return answer


def _contained(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    result = (root / relative).resolve()
    require(
        not relative.is_absolute() and result.is_relative_to(root),
        "Evidence path escapes run",
    )
    return result


def _counts(rows):
    return dict(
        attempts=len(rows),
        native_releases=sum(bool(r.get("native_release")) for r in rows),
        invalid=sum(bool(r.get("invalid")) for r in rows),
        captures=sum(
            bool(
                r.get("native_release")
                and not r.get("invalid")
                and r.get("route_success")
            )
            for r in rows
        ),
        clean=sum(
            bool(
                r.get("native_release")
                and not r.get("invalid")
                and r.get("route_quality") == 2
            )
            for r in rows
        ),
    )


def _stats(values):
    values = np.asarray(values, dtype=float)
    return dict(
        count=int(values.size),
        min=float(values.min()) if values.size else None,
        median=float(np.median(values)) if values.size else None,
        max=float(values.max()) if values.size else None,
    )


def _runs(flags):
    """Consecutive true ranges; inclusive sample indices."""
    answer, start = [], None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            answer.append((start, index - 1))
            start = None
    if start is not None:
        answer.append((start, len(flags) - 1))
    return answer


def _run_summary(flags):
    ranges = _runs(flags)
    lengths = [end - start + 1 for start, end in ranges]
    return dict(
        count=len(ranges),
        failed_samples=int(sum(lengths)),
        duration_samples=lengths,
        durations_s=[length * DT_S for length in lengths],
    )


def wall_geometry(scene, bucket_xy, positions):
    """Original nominal polygon tests; not engine collision/contact distances."""
    bucket, radius = scene["bucket"], float(scene["ball"]["radius_m"])
    positions = np.asarray(positions, dtype=float)
    floor = bucket["assumed_bottom_thickness_m"]
    top = floor + bucket["assumed_interior_depth_m"]
    slope = (bucket["inner_top_diameter_m"] - bucket["inner_bottom_diameter_m"]) / (
        2 * bucket["assumed_interior_depth_m"]
    )
    cosine = math.cos(math.pi / bucket["segments"])
    angles = (np.arange(bucket["segments"]) + 0.5) * 2 * math.pi / bucket["segments"]
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    projection = np.max((positions[:, :2] - bucket_xy) @ normals.T, axis=1)
    inner = (
        bucket["inner_bottom_diameter_m"] / 2 + slope * (positions[:, 2] - floor)
    ) * cosine
    clearance = (inner - projection) / math.sqrt(1 + (slope * cosine) ** 2)
    above_rim = positions[:, 2] + radius > top + GEOMETRY_EPS_M
    below_floor = positions[:, 2] - radius < floor - 0.001
    wall_failed = clearance < radius - GEOMETRY_EPS_M
    return dict(
        whole=~(above_rim | below_floor | wall_failed),
        above_rim=above_rim,
        below_floor=below_floor,
        wall=wall_failed,
        nominal_wall_overlap_m=np.maximum(0, radius - clearance),
    )


def trace_sha256(identifier, positions, velocities, forces):
    """Bind the exact extracted arrays, including their saved dtype and shape."""
    hasher = hashlib.sha256(identifier.encode() + b"\0")
    for name, array in (
        ("position_m", positions),
        ("velocity_m_s", velocities),
        ("forces_N", forces),
    ):
        array = np.ascontiguousarray(array)
        hasher.update(
            json.dumps(
                [name, array.dtype.str, array.shape], separators=(",", ":")
            ).encode()
            + b"\0"
        )
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def native_cache_timing(arrays):
    """Check optional explicit cache timing; legacy caches have no dt field.

    Legacy timing is bound by frozen physics, hash-checked trace producers, and
    the exact960-row shape. The row's1/60s display trail metadata is unrelated.
    """
    names = set(arrays.files if hasattr(arrays, "files") else arrays)
    explicit = {}
    for name in ("dt_s", "scoring_dt_s", "trajectory_dt_s"):
        if name not in names:
            continue
        value = np.asarray(arrays[name])
        require(
            value.size == 1 and value.dtype.kind in "fiu",
            "Invalid NPZ cadence metadata",
        )
        dt = float(value.reshape(-1)[0])
        require(
            math.isfinite(dt) and math.isclose(dt, DT_S, rel_tol=1e-7, abs_tol=0),
            "NPZ cadence differs from frozen scoring cadence",
        )
        explicit[name] = dt
    for name in ("time_s", "times_s", "timestamps_s"):
        if name not in names:
            continue
        times = np.asarray(arrays[name])
        require(
            times.shape == (HORIZON_STEPS,) and times.dtype.kind == "f",
            "Invalid NPZ timestamp metadata",
        )
        expected = np.arange(1, HORIZON_STEPS + 1) * DT_S
        precision = np.finfo(times.dtype).eps * HORIZON_STEPS * DT_S
        require(
            np.allclose(times, expected, rtol=0, atol=precision),
            "NPZ timestamps differ from frozen post-step scoring cadence",
        )
        explicit[name] = dict(
            samples=int(times.size),
            first_time_s=float(times[0]),
            last_time_s=float(times[-1]),
        )
    return dict(
        scoring_dt_s=DT_S,
        horizon_steps=HORIZON_STEPS,
        horizon_s=HORIZON_STEPS * DT_S,
        explicit_npz_timing=explicit or None,
        basis="Frozen physics + hash-bound native trace producer +960-row shape; explicit NPZ timing additionally checked when present.",
    )


def diagnose_trace(
    scene, row, positions, velocities, forces, *, native_scoring_dt_s=DT_S
):
    """Reproduce the unchanged score, then explain resets in that scored prefix."""
    positions, velocities, forces = map(np.asarray, (positions, velocities, forces))
    require(
        positions.shape == velocities.shape == (HORIZON_STEPS, 3),
        "Wrong 4-second position/velocity trace",
    )
    require(forces.shape == (HORIZON_STEPS, len(FORCE_NAMES)), "Wrong force trace")
    require(
        all(np.isfinite(a).all() for a in (positions, velocities, forces)),
        "Nonfinite trace",
    )
    require((forces >= 0).all(), "Negative contact-force norm")
    require(
        scene["simulation"]["physics_dt_s"] == DT_S, "Changed original scoring cadence"
    )
    require(
        native_scoring_dt_s == DT_S, "Native trace cadence differs from original scorer"
    )
    require(
        row.get("native_release") and not row.get("invalid"),
        "Trace requires an admitted native release",
    )
    scorer = ScoreStream(scene, row["bucket_xy_m"], row["board_normal"])
    states = wall_geometry(scene, row["bucket_xy_m"], positions)
    first_entry, scored_steps = None, 0
    for index, (position, velocity, force) in enumerate(
        zip(positions, velocities, forces)
    ):
        if scorer.finished:
            break
        scorer.step(position, velocity, dict(zip(FORCE_NAMES, force)))
        scored_steps = index + 1
        if scorer.capture.entered and first_entry is None:
            first_entry = index
    measured = scorer.result()
    for key in ORIGINAL_FIELDS:
        require(key in row, "Missing original scored field: " + key)
        actual, original = measured[key], row[key]
        equal = (
            math.isclose(actual, original, rel_tol=1e-9, abs_tol=1e-10)
            if isinstance(actual, float)
            else actual == original
        )
        require(equal, "Original scorer mismatch: " + key)
    post = slice(first_entry, scored_steps) if first_entry is not None else slice(0, 0)
    inside = states["whole"][post]
    constraints = {
        name: _run_summary(states[name][post])
        for name in ("wall", "above_rim", "below_floor")
    }
    failures = states["wall"][post]
    overlaps = states["nominal_wall_overlap_m"][post]
    return dict(
        original_score_reproduced=True,
        scoring_trace_dt_s=native_scoring_dt_s,
        display_trajectory_dt_s=row.get("trajectory_dt_s"),
        first_whole_entry_sample=first_entry,
        first_whole_entry_time_s=(
            (first_entry + 1) * DT_S if first_entry is not None else None
        ),
        max_continuous_retention_s=measured["max_retention_s"],
        scoring_end_sample=scored_steps - 1,
        scoring_end_time_s=scored_steps * DT_S,
        scoring_end_whole_ball_inside=bool(states["whole"][scored_steps - 1]),
        trace_end_time_s=HORIZON_STEPS * DT_S,
        trace_end_whole_ball_inside=bool(states["whole"][-1]),
        retention_resets=_run_summary(~inside),
        post_entry_constraint_failures=constraints,
        failed_wall_nominal_overlap_m=_stats(overlaps[failures]),
        failed_wall_excess_beyond_original_epsilon_m=_stats(
            overlaps[failures] - GEOMETRY_EPS_M
        ),
        original_rim_and_assistance={
            key: row[key]
            for key in (
                "rim_band_contact",
                "bucket_contact_before_entry",
                "exited_after_entry",
                "external_contact_steps",
                "board_contact_episodes",
            )
        },
        scored_contact_observations={
            name: dict(
                samples_above_original_contact_threshold=int(
                    np.sum(forces[:scored_steps, column] > FORCE_THRESHOLD_N)
                ),
                peak_force_N=float(np.max(forces[:scored_steps, column])),
            )
            for column, name in enumerate(FORCE_NAMES)
        },
        trace_sha256=trace_sha256(row["id"], positions, velocities, forces),
    )


def diagnose_run(run, reference, milestones, *, request_ids=None):
    run, reference = Path(run).resolve(), Path(reference).resolve()
    require(
        run.parent.name == "runs",
        "Expected a benchmark runs/<method>-seed<seed> directory",
    )
    study = run.parent.parent
    contract_path, reference_path, report_path = (
        study / "contract.json",
        reference / "contract.json",
        run / "report.json",
    )
    contract, ref = json.loads(contract_path.read_text()), json.loads(
        reference_path.read_text()
    )
    unsigned = dict(contract)
    fingerprint = unsigned.pop("fingerprint", None)
    require(fingerprint == digest(unsigned), "Study contract fingerprint changed")
    require(
        contract["identity"]["scene_sha256"] == digest(ref["scene"]),
        "Reference scene differs from frozen study",
    )
    require(
        contract["identity"]["contract_fingerprint"] == ref["fingerprint"],
        "Reference contract binding differs",
    )
    unsigned_ref = dict(ref)
    unsigned_ref.pop("fingerprint")
    require(digest(unsigned_ref) == ref["fingerprint"], "Reference fingerprint changed")
    physics = contract["identity"]["physics"]
    require(
        physics["scoring_dt_s"] == DT_S
        and physics["horizon_s"] == DT_S * HORIZON_STEPS,
        "Frozen scoring cadence/horizon differs",
    )
    sources = {
        name: file_sha256(ROOT / name) for name in (*SCORER_FILES, *TRACE_SOURCE_FILES)
    }
    require(
        all(
            contract["identity"]["source_sha256"].get(name) == value
            for name, value in sources.items()
        ),
        "Installed original scorer/cache producer differs from frozen study",
    )
    report_text = report_path.read_text()
    report = selected_members(
        report_text, {"version", "method", "seed", "contract_fingerprint", "curves"}
    )
    require(
        report["contract_fingerprint"] == fingerprint,
        "Run/report contract binding changed",
    )
    require(
        run.name == f"{report['method']}-seed{report['seed']}",
        "Run directory identity differs",
    )
    requested = list(milestones)
    require(
        requested and len(requested) == len(set(requested)),
        "Choose unique completed validation milestones",
    )
    require(
        all(
            type(s) is int and s in contract["protocol"]["milestones"]
            for s in requested
        ),
        "Milestone outside frozen protocol",
    )
    curves = {}
    for point in report["curves"]:
        require(point["samples"] not in curves, "Duplicate published validation point")
        require(
            point["method"] == report["method"] and point["seed"] == report["seed"],
            "Curve identity differs",
        )
        curves[point["samples"]] = point
    require(
        all(s in curves for s in requested),
        "Requested validation milestone is not published complete",
    )
    contexts = {r["id"] for r in contract["splits"]["validation"]}
    repeats = contract["protocol"]["evaluation"]["repeats"]
    expected = {(s, c, k) for s in requested for c in contexts for k in range(repeats)}
    require(
        all(
            curves[s]["validation"]["attempts"] == len(contexts) * repeats
            for s in requested
        ),
        "Published validation cohort is incomplete",
    )
    rows, raw_hashes, seen = [], {}, set()
    with (run / "rows.jsonl").open() as stream:
        for line in stream:
            header = selected_members(line, {"role", "samples"})
            if header.get("role") == "test":
                raise DiagnosticError(
                    "Test boundary reached before requested validation cohorts completed"
                )
            if (
                header.get("role") != "validation"
                or header.get("samples") not in requested
            ):
                continue
            row = json.loads(line)
            key = (row["samples"], row["context_id"], row["repeat"])
            require(
                key in expected and key not in seen,
                "Duplicate/unexpected validation request",
            )
            require(
                row["method"] == report["method"] and row["seed"] == report["seed"],
                "Row identity differs",
            )
            require(
                row["frozen_artifact_sha256"]
                == curves[row["samples"]]["artifact_sha256"],
                "Validation frozen artifact differs",
            )
            seen.add(key)
            require(row["id"] not in raw_hashes, "Duplicate validation request ID")
            rows.append(row)
            raw_hashes[row["id"]] = hashlib.sha256(line.encode()).hexdigest()
            if seen == expected:
                break
    require(seen == expected, "Missing completed validation rows")
    cohorts = []
    for samples in sorted(requested):
        cohort = [r for r in rows if r["samples"] == samples]
        counts = _counts(cohort)
        require(
            all(
                curves[samples]["validation"][key] == value
                for key, value in counts.items()
            ),
            "Published original counts disagree with validation rows",
        )
        cohorts.append(
            dict(
                samples=samples,
                original_counts=curves[samples]["validation"],
                invalid_reasons=dict(
                    Counter(str(r["invalid"]) for r in cohort if r.get("invalid"))
                ),
            )
        )
    chosen = set(request_ids) if request_ids else set(raw_hashes)
    require(
        chosen <= set(raw_hashes) and chosen,
        "Requested IDs must belong to the completed validation cohorts",
    )
    native_ids = {r["id"] for r in rows if r.get("native_release")}
    cache_hashes, details = {}, []
    for row in rows:
        if row["id"] not in chosen:
            continue
        detail = dict(
            id=row["id"],
            samples=row["samples"],
            context_id=row["context_id"],
            repeat=row["repeat"],
            bucket_xy_m=row["bucket_xy_m"],
            command_values=row["command_values"],
            requested_row_sha256=raw_hashes[row["id"]],
            frozen_artifact_sha256=row["frozen_artifact_sha256"],
            original={
                key: row.get(key)
                for key in ("native_release", "invalid", *ORIGINAL_FIELDS)
            },
        )
        if row.get("native_release"):
            name = row["native_cache_file"]
            require(
                Path(name).name == name
                and name.startswith("native-")
                and name.endswith(".npz"),
                "Expected native cache basename",
            )
            path = _contained(run / "native", name)
            if name not in cache_hashes:
                cache_hashes[name] = file_sha256(path)
            with np.load(path, allow_pickle=False) as arrays:
                ids = arrays["record_ids"].tolist()
                require(
                    len(ids) == len(set(ids)) and set(ids) <= native_ids,
                    "Native cache includes unknown/non-selected validation IDs",
                )
                index = row["native_cache_row"]
                require(
                    type(index) is int
                    and 0 <= index < len(ids)
                    and ids[index] == row["id"],
                    "Native cache row ID differs",
                )
                require(
                    arrays["position_m"].shape
                    == arrays["velocity_m_s"].shape
                    == (HORIZON_STEPS, len(ids), 3),
                    "Native cache dimensions differ",
                )
                require(
                    arrays["forces_N"].shape
                    == (HORIZON_STEPS, len(ids), len(FORCE_NAMES)),
                    "Native force cache dimensions differ",
                )
                timing = native_cache_timing(arrays)
                metrics = diagnose_trace(
                    ref["scene"],
                    row,
                    arrays["position_m"][:, index],
                    arrays["velocity_m_s"][:, index],
                    arrays["forces_N"][:, index],
                    native_scoring_dt_s=timing["scoring_dt_s"],
                )
            detail.update(
                native_cache_file=name,
                native_cache_row=index,
                native_cache_sha256=cache_hashes[name],
                native_cache_timing=timing,
                diagnostic=metrics,
            )
        details.append(detail)
    result = dict(
        version=VERSION,
        status="completed",
        method=report["method"],
        seed=report["seed"],
        contract_fingerprint=fingerprint,
        run_report_sha256=hashlib.sha256(report_text.encode()).hexdigest(),
        reference_contract_sha256=file_sha256(reference_path),
        source_sha256=dict(
            sources,
            **{"sircus/mjlab/benchmark_wall_diagnostic.py": file_sha256(__file__)},
        ),
        scope="Completed validation only; original scores and labels unchanged; CPU replay of recorded arrays, no new rollout or tolerance search.",
        cadence=dict(
            scoring_dt_s=DT_S,
            horizon_steps=HORIZON_STEPS,
            horizon_s=DT_S * HORIZON_STEPS,
            original_geometry_epsilon_m=GEOMETRY_EPS_M,
            original_contact_threshold_N=FORCE_THRESHOLD_N,
        ),
        metric_definitions=dict(
            nominal_wall_overlap_m="Sphere radius minus perpendicular clearance to the nominal polygon wall planes, when positive; not a measured engine contact penetration.",
            wall_failure="Original nominal whole-sphere wall-clearance predicate fails using its unchanged geometry epsilon.",
            below_floor="Original bucket-interior bottom-height predicate, separate from the attributed simulation floor contact-force channel.",
            time_origin="Saved sample0 is the first post-step observation at1/240s; zero-based indices and sampled times are both reported.",
            display_trajectory_dt_s="Canonical row metadata describes the downsampled display trail (normally1/60s), not the960-sample native scoring arrays.",
        ),
        cohorts=cohorts,
        analyzed_request_ids=[r["id"] for r in details],
        requests=details,
        limitations=[
            "Whole-sphere wall-clearance failure is a geometric diagnostic, not proof of a ball escaping or certified physical penetration.",
            "Durations and first-entry times are sampled at240Hz; they are not substep collision measurements.",
            "Legacy native NPZ files contain no dt/timestamps: their scoring cadence is established by the frozen physics, hash-bound producer code and960-row dimensions, not independently measured from timestamps.",
            "Original scoring freezes on first success or ball-center Z below-0.2m; scored-prefix and full recorded-trace end states are distinct.",
            "Native GPU contact trajectories can differ with identical recorded commands; this diagnostic does not establish bitwise simulator determinism.",
            "This diagnostic does not replace the full independent study/provenance audit; it explains only the selected completed validation cohorts.",
            "No test outcomes, new tolerance, altered capture labels or training recommendations are produced.",
        ],
    )
    result["fingerprint"] = digest(result)
    return result


def write_external(result, output, run, reference):
    output, study, reference = (
        Path(output).resolve(),
        Path(run).resolve().parent.parent,
        Path(reference).resolve(),
    )
    require(
        not output.is_relative_to(study) and not output.is_relative_to(reference),
        "Diagnostic output must be outside immutable study/reference",
    )
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with output.open("x") as stream:
        stream.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--milestones", type=int, nargs="+", required=True)
    parser.add_argument(
        "--request-id",
        action="append",
        dest="request_ids",
        help="Optional subset of validation request IDs; full cohort counts remain reported",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New external JSON; never overwritten",
    )
    args = parser.parse_args(argv)
    result = diagnose_run(
        args.run, args.reference, args.milestones, request_ids=args.request_ids
    )
    write_external(result, args.output, args.run, args.reference)
    print(
        json.dumps(
            dict(
                status=result["status"],
                fingerprint=result["fingerprint"],
                analyzed_requests=len(result["requests"]),
            )
        )
    )


if __name__ == "__main__":
    main()
