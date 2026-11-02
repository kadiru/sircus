"""Compare fresh paired engine shots, or explicitly identify a historical reference."""

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .contract import verify_manifest
else:
    from contract import verify_manifest


def captured(row):
    return None if row.get("invalid") else bool(row["route_success"])


def validate_results(manifest, result):
    if (
        result["benchmark_fingerprint"] != manifest["fingerprint"]
        or not result["engine"]
    ):
        raise ValueError(
            "Engine result has the wrong benchmark fingerprint or identity"
        )
    lookup = {}
    for row in result["shots"]:
        key = row["shot_id"], row["repeat"]
        if key in lookup:
            raise ValueError("Duplicate engine shot/repeat")
        captured(row)
        lookup[key] = row
    expected = {
        (s["shot_id"], r) for s in manifest["shots"] for r in range(manifest["repeats"])
    }
    if set(lookup) != expected:
        raise ValueError("Engine result is missing or adds benchmark shots")
    return lookup


def summarize(rows):
    rows = list(rows)
    valid = [r for r in rows if captured(r) is not None]
    return dict(
        attempts=len(rows),
        valid=len(valid),
        initialization_failures=len(rows) - len(valid),
        captures=sum(bool(captured(r)) for r in valid),
        clean_captures=sum(captured(r) and r.get("route_quality") == 2 for r in valid),
        board_bounces=sum(bool(r.get("board_bounce")) for r in valid),
        capture_fraction=(
            sum(bool(captured(r)) for r in valid) / len(valid) if valid else None
        ),
    )


def trajectory_error(first, second):
    """Only compare positions sampled at exactly the same declared physics steps."""
    needed = ("trajectory_steps", "trajectory_m")
    if not all(k in first and k in second for k in needed):
        return None
    tracks = []
    for row in (first, second):
        steps, xyz = row["trajectory_steps"], row["trajectory_m"]
        if len(steps) != len(xyz) or len(set(steps)) != len(steps):
            raise ValueError("Ambiguous trajectory sample timing")
        p = np.asarray(xyz, dtype=float)
        if p.shape != (len(steps), 3) or not np.isfinite(p).all():
            raise ValueError("Invalid trajectory positions")
        tracks.append(dict(zip(steps, p)))
    common = sorted(set(tracks[0]) & set(tracks[1]))
    if not common:
        return None
    errors = [float(np.linalg.norm(tracks[0][i] - tracks[1][i])) * 1000 for i in common]
    return dict(
        common_samples=len(common),
        max_mm=max(errors),
        rms_mm=float(np.sqrt(np.mean(np.square(errors)))),
    )


def build_report(manifest, first, second=None):
    verify_manifest(manifest)
    substeps = first.get(
        "physics_substeps", first.get("versions", {}).get("physics_substeps", 1)
    )
    a = validate_results(manifest, first)
    historical = second is None
    if historical:
        second = dict(
            engine="historical_isaac_reference",
            versions={},
            shots=[
                dict(s["isaac_reference"], shot_id=s["shot_id"], repeat=i)
                for s in manifest["shots"]
                for i in range(manifest["repeats"])
            ],
            benchmark_fingerprint=manifest["fingerprint"],
        )
    b = validate_results(manifest, second)
    other_substeps = second.get(
        "physics_substeps", second.get("versions", {}).get("physics_substeps", 1)
    )
    if not historical and substeps != other_substeps:
        raise ValueError("Fresh engine pairs must use the same physics substeps")
    pairs = []
    for shot in manifest["shots"]:
        for repeat in range(manifest["repeats"]):
            key = shot["shot_id"], repeat
            x, y = a[key], b[key]
            one, two = captured(x), captured(y)
            pairs.append(
                dict(
                    shot_id=key[0],
                    repeat=repeat,
                    source_category=shot["category"],
                    first_capture=one,
                    second_capture=two,
                    capture_agrees=(
                        one == two if one is not None and two is not None else None
                    ),
                    trajectory_error=trajectory_error(x, y),
                )
            )
    comparable = [p for p in pairs if p["capture_agrees"] is not None]
    categories = sorted({s["category"] for s in manifest["shots"]})
    strata = []
    for category in categories:
        ids = {s["shot_id"] for s in manifest["shots"] if s["category"] == category}
        group = [p for p in comparable if p["source_category"] == category]
        strata.append(
            dict(
                category=category,
                unique_shots=len(ids),
                first=summarize(row for key, row in a.items() if key[0] in ids),
                second=summarize(row for key, row in b.items() if key[0] in ids),
                comparable_pairs=len(group),
                capture_agreements=sum(p["capture_agrees"] for p in group),
            )
        )
    repeatability = []
    for shot in manifest["shots"]:
        key = shot["shot_id"]
        values = {
            "first": [captured(a[key, i]) for i in range(manifest["repeats"])],
            "second": [
                captured(b[key, i])
                for i in range(manifest["repeats"])
                if not historical
            ],
        }
        repeatability.append(
            dict(
                shot_id=key,
                **{
                    name: (
                        dict(captures=vals, all_repeats_agree=len(set(vals)) == 1)
                        if vals
                        else None
                    )
                    for name, vals in values.items()
                },
            )
        )
    return dict(
        version="sircus-paired-engine-report-v1",
        benchmark_fingerprint=manifest["fingerprint"],
        physics_substeps=substeps,
        comparison=(
            "historical_reference"
            if historical
            else "fresh_paired_fixed_endpoint_contact_replay"
        ),
        unique_shots=len(manifest["shots"]),
        repeats=manifest["repeats"],
        first=dict(
            engine=first["engine"],
            versions=first.get("versions", {}),
            timing=first.get("timing"),
            **summarize(a.values()),
        ),
        second=dict(
            engine=second["engine"],
            versions=second.get("versions", {}),
            timing=second.get("timing"),
            **summarize(b.values()),
        ),
        comparable_pairs=len(comparable),
        capture_agreements=sum(p["capture_agrees"] for p in comparable),
        capture_agreement_fraction=(
            sum(p["capture_agrees"] for p in comparable) / len(comparable)
            if comparable
            else None
        ),
        strata=strata,
        pairs=pairs,
        repeatability=repeatability,
        limitations=[
            "Configurations were stratified by original Isaac outcome. Aggregate capture rates describe this diagnostic selection, not arbitrary-table policy performance.",
            "Unperturbed repeats test numerical repeatability. They do not establish robustness to perception, contact or calibration error.",
            "Fixed joint endpoints isolate release/contact physics. Grasping, path planning and interactive constrained policy search are separate tasks.",
        ]
        + (
            [
                "The reference is one historical Isaac observation per configuration, reused across repeats; it is not a fresh paired-engine measurement."
            ]
            if historical
            else []
        ),
    )


def markdown(report):
    a, b = report["first"], report["second"]
    lines = [
        "# Sircus engine contact comparison",
        "",
        f"Compared {report['unique_shots']} selected configurations with {report['repeats']} unperturbed repeats. Protocol is {report['comparison']}.",
        "",
        "| Source outcome stratum | Configurations | First captures / valid | Second captures / valid | Capture agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in report["strata"]:
        x, y = c["first"], c["second"]
        lines.append(
            f"| {c['category']} | {c['unique_shots']} | {x['captures']} / {x['valid']} | {y['captures']} / {y['valid']} | {c['capture_agreements']} / {c['comparable_pairs']} |"
        )
    lines.extend(
        [
            "",
            f"First engine is {a['engine']}; second engine is {b['engine']}. Initialization failures are {a['initialization_failures']} and {b['initialization_failures']} respectively.",
            "",
            "A complete success uses the shared board-rebound, whole-ball entry and continuous one-second retention scorer. Rim-assisted captures qualify. External table, floor, robot or unknown contact disqualifies the complete route.",
            "",
        ]
    )
    lines.extend(f"- {line}" for line in report["limitations"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text())
    report = build_report(
        load(args.manifest),
        load(args.first),
        load(args.second) if args.second else None,
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown(report))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("strata", "pairs", "repeatability", "limitations")
            }
        )
    )
