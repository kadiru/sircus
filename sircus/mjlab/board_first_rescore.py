"""CPU-only derivative labels from authenticated saved native evidence.

Original scores remain alongside the new board-first score. This does not
simulate missing time, manufacture rollouts, train a model, or promote a policy.
Only explicit source directories are indexed; copies of the same archive and
column count once. Recorded v2 and v3 scores must reproduce before relabeling.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from . import calibrated_score as v2
from . import table_assisted_rescore as evidence
from . import table_assisted_score as v3
from .learning_contract import digest

VERSION = "sircus-board-first-saved-rescore-v1"
AUDIT_VERSION = "sircus-board-first-saved-rescore-audit-v1"
SUPPORTED = (v2.VERSION, v3.VERSION)
OLD_FIELDS = (*evidence.OLD_FIELDS, "route_quality", "single_bounce_success",
              "board_contact_episodes", "physics_steps", "quality_score")
ROW_FIELDS = tuple(dict.fromkeys((*evidence.ROW_FIELDS, *OLD_FIELDS,
    "forbidden_contact_steps", "table_contact_steps", "direct_capture",
    "table_assisted_capture", "route_class", "single_board_bounce_success",
    "native_initial_ball_center_m", "historical_v2_score")))
sha = evidence.sha
read = evidence.read
require = evidence.require
same = evidence.same


def index(root, sources, scene_contracts=()):
    root = Path(root).resolve()
    require(bool(sources), "Explicit source directories required; never scan all storage")
    directories = [evidence.inside(root, root / name) for name in sources]
    inputs, scenes, contexts = {}, {}, {}
    for path in sorted({p.resolve() for d in directories for p in d.rglob("contract.json")}
                       | {evidence.inside(root, root / p) for p in scene_contracts}):
        value = read(path)
        if isinstance(value.get("scene"), dict):
            scenes[digest(value["scene"])] = value["scene"], path
    files = sorted({p.resolve() for d in directories for p in d.rglob("*.jsonl")
                    if p.name in ("shots.jsonl", "records.jsonl")},
                   key=lambda p: (p.name != "shots.jsonl", str(p)))
    require(bool(files), "No native shot indexes found")
    rows, native_ids, record_ids = [], {}, {}
    total = duplicates = skipped_files = 0
    for file in files:
        inputs[str(file.relative_to(root))] = sha(file)
        if file.parent not in contexts:
            try:
                directory = file.parent
                while directory.is_relative_to(root) and not (directory / "contract.json").exists() and directory != root:
                    directory = directory.parent
                contract_path = directory / "contract.json"
                if contract_path.exists() and read(contract_path).get("version") in (VERSION, evidence.VERSION):
                    inputs[str(contract_path.relative_to(root))] = sha(contract_path)
                    ctx = ("derivative", "existing_derivative_not_an_original_trial")
                else:
                    ctx = evidence.context(root, file, scenes, inputs)
                contexts[file.parent] = ctx
            except (ValueError, OSError) as exc:
                contexts[file.parent] = ("unscorable" if isinstance(exc, evidence.Unscorable)
                                         else "evidence_error", str(exc))
        ctx = contexts[file.parent]
        if isinstance(ctx, tuple) and ctx[0] == "derivative":
            skipped_files += 1
            continue
        with file.open() as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                total += 1
                source = dict(file=str(file.relative_to(root)), line=line_no,
                              metadata_sha256=inputs[str(file.relative_to(root))])
                try:
                    original = json.loads(line)
                    require(isinstance(original, dict), "Metadata must be an object")
                except ValueError as exc:
                    rows.append(dict(id=f"malformed:{source['file']}:{line_no}", source=source,
                                     original={}, aliases=[], status="evidence_error", reason=str(exc)))
                    continue
                row = {key: original[key] for key in ROW_FIELDS if key in original}
                ident = row.get("id") or f"unidentified:{source['file']}:{line_no}"
                source_study = ctx["source_study"] if isinstance(ctx, dict) else str(file.parent.relative_to(root))
                item = dict(id=ident, source=source, original=row, aliases=[],
                            **evidence.grouping(file, row, source_study))
                if not isinstance(ctx, dict):
                    item.update(status=ctx[0], reason=ctx[1]); rows.append(item)
                    continue
                item["context"] = ctx
                native_key = (row.get("native_cache_sha256"), row.get("native_cache_row"))
                native_key = native_key if native_key[0] is not None else None
                request_key = (source_study, ident, digest([row.get("bucket_xy_m"), row.get("values")]))
                previous = native_ids.get(native_key) if native_key else record_ids.get(request_key)
                if previous is not None:
                    keys = (*OLD_FIELDS, "id", "bucket_xy_m", "values", "board_normal")
                    if any(not same(previous["original"][key], row[key]) for key in keys
                           if key in previous["original"] and key in row):
                        previous.update(status="evidence_error", reason="Copied native evidence changes original labels or command")
                    if any(previous["context"][key] != ctx[key]
                           for key in ("scene_sha256", "calibration_fingerprint")):
                        previous.update(status="evidence_error", reason="Copied native evidence changes scene/calibration")
                    previous["aliases"].append(source); duplicates += 1
                    continue
                if row.get("native_release") is False:
                    item.update(status="rejected" if row.get("invalid") or row.get("policy_abstain") else "unscorable",
                                reason=row.get("invalid") or "nonrelease_without_scored_trajectory")
                elif row.get("scoring_version") not in SUPPORTED:
                    item.update(status="unscorable", reason="unsupported_original_scoring_version")
                elif not all(row.get(key) is not None for key in
                             ("native_cache_file", "native_cache_row", "native_cache_sha256")):
                    item.update(status="unscorable", reason="missing_dense_native_binding")
                else:
                    try:
                        path = evidence.archive_path(root, file, row, ctx)
                        require(type(row["native_cache_row"]) is int and row["native_cache_row"] >= 0,
                                "Native column must be nonnegative integer")
                        require(row.get("calibration_fingerprint") == ctx["calibration_fingerprint"],
                                "Row calibration differs from contract")
                        item.update(status="pending", native_file=str(path.relative_to(root)))
                    except (ValueError, OSError) as exc:
                        item.update(status="unscorable" if isinstance(exc, evidence.Unscorable)
                                    else "evidence_error", reason=str(exc))
                rows.append(item); record_ids[request_key] = item
                if native_key:
                    native_ids[native_key] = item
    return dict(root=root, rows=rows, inputs=inputs, sources=list(sources),
                metadata=dict(files=len(files)-skipped_files, indexed_rows=total,
                              duplicate_rows=duplicates, unique_requests=len(rows),
                              skipped_derivative_files=skipped_files))


def score_archive(job):
    from . import calibrated_score_oracle as v2_oracle
    from . import table_assisted_score_oracle as v3_oracle
    from . import board_first_score as new
    from . import board_first_score_oracle as oracle
    root, filename, members = job
    results = []
    try:
        path = Path(root) / filename
        archive_sha = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("record_ids", "position_m", "velocity_m_s", "forces_N", "time_s")}
            arrays.update({key: archive[key] for key in ("contact_distance_m", "dt_s") if key in archive})
        for item in members:
            row, ctx = item["original"], item["context"]
            try:
                require(archive_sha == row["native_cache_sha256"], "Native archive checksum differs")
                col = row["native_cache_row"]
                require(col < len(arrays["record_ids"]) and str(arrays["record_ids"][col]) == item["id"],
                        "Native record identity differs")
                trace = {key: arrays[key][:, col] for key in ("position_m", "velocity_m_s", "forces_N")}
                trace["time_s"] = arrays["time_s"]
                if "contact_distance_m" in arrays:
                    trace["contact_distance_m"] = arrays["contact_distance_m"][:, col]
                if "dt_s" in arrays:
                    trace["dt_s"] = arrays["dt_s"]
                if "native_initial_ball_center_m" in row:
                    require(np.array_equal(trace["position_m"][0], row["native_initial_ball_center_m"]),
                            "Native release center differs")
                step = item.get("substeps")
                if type(step) is int:
                    base_dt = ctx["scene"].get("simulation", {}).get("physics_dt_s")
                    if base_dt is not None:
                        require(np.allclose(np.diff(trace["time_s"]), float(base_dt)/step, rtol=0, atol=1e-12),
                                "Native cadence differs from substeps")
                args = ctx["scene"], row["bucket_xy_m"], row["board_normal"], ctx["calibration"], trace
                old, old_oracle = (v2, v2_oracle) if row["scoring_version"] == v2.VERSION else (v3, v3_oracle)
                original, independent_old = old.score_trace(*args), old_oracle.score_trace(*args)
                require(same(original, independent_old), "Original score disagrees with independent oracle")
                require(all(key in row and same(row[key], original[key]) for key in OLD_FIELDS),
                        "Original metadata score differs from dense evidence")
                updated, independent_new = new.score_trace(*args), oracle.score_trace(*args)
                require(same(updated, independent_new), "Board-first score disagrees with independent oracle")
                require(all(same(original[key], updated[key]) for key in
                            ("success", "invalid_physics", "whole_ball_entry", "max_retention_s", "physics_steps")),
                        "Containment, terminal or physics validity changed")
                item.update(status="scored", old_score=original, new_score=updated,
                            events=evidence.events(trace, updated), native_sha256=archive_sha, native_row=col,
                            checks_count=8)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                item.update(status="evidence_error", reason=str(exc))
            results.append(item)
    except (OSError, ValueError, KeyError) as exc:
        for item in members:
            item.update(status="evidence_error", reason="Native archive unavailable: " + str(exc))
            results.append(item)
    return results


def counts(rows):
    out = Counter(dict(requested=0, scorable_native=0, rejected=0, unscorable=0,
                       evidence_errors=0, invalid_native=0, old_captures=0, new_captures=0,
                       gained_captures=0, lost_captures=0, unchanged_captures=0, unchanged_misses=0))
    for row in rows:
        out["requested"] += 1
        if row["status"] != "scored":
            out[{"rejected": "rejected", "evidence_error": "evidence_errors"}.get(row["status"], "unscorable")] += 1
            continue
        old, new = row["old_score"], row["new_score"]
        before, after = bool(old["route_success"]), bool(new["route_success"])
        out.update(scorable_native=1, invalid_native=int(new["invalid_physics"]),
                   old_captures=int(before), new_captures=int(after))
        if new["invalid_physics"]:
            continue
        out["unchanged_captures" if before and after else "lost_captures" if before
            else "gained_captures" if after else "unchanged_misses"] += 1
    return dict(out)


def run(root, output, *, sources, scene_contracts=(), workers=2, index_only=False):
    output = Path(output).resolve()
    require(not output.exists(), "New derivative output directory required")
    require(1 <= workers <= 4, "One to four CPU workers")
    data = index(root, sources, scene_contracts); root = data["root"]
    require(all(not (root / name).is_relative_to(output) for name in data["inputs"]),
            "Derivative output overlaps source evidence")
    jobs = defaultdict(list)
    for row in data["rows"]:
        if row["status"] == "pending": jobs[row["native_file"]].append(row)
    if index_only:
        summary = dict(version=VERSION, status="index_only", metadata=data["metadata"],
                       statuses=dict(Counter(row["status"] for row in data["rows"])),
                       reasons=dict(Counter(row.get("reason") for row in data["rows"] if row.get("reason"))),
                       archives=len(jobs), inputs=data["inputs"])
        output.mkdir(parents=True); (output / "index.json").write_text(json.dumps(summary, indent=2)+"\n")
        return summary
    rows = [row for row in data["rows"] if row["status"] != "pending"]
    pending = [(str(root), name, group) for name, group in jobs.items()]
    def collect(groups):
        for i, group in enumerate(groups, 1):
            rows.extend(group)
            if i == 1 or i % 10 == 0 or i == len(pending):
                print(json.dumps(dict(phase="cpu_saved_rescore", archives_complete=i,
                                     archives_total=len(pending), rows_processed=len(rows))), flush=True)
    if workers == 1:
        collect(map(score_archive, pending))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(score_archive, pending))
    totals = counts(rows)
    require(totals["requested"] == data["metadata"]["unique_requests"], "Unique request accounting differs")
    require(totals["new_captures"]-totals["old_captures"] == totals["gained_captures"]-totals["lost_captures"],
            "Capture accounting differs")
    grouped, locations = defaultdict(list), defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "unknown") for k in evidence.GROUP_FIELDS)].append(row)
        xy = row["original"].get("bucket_xy_m")
        if xy is not None: locations[tuple(xy)].append(row)
        if row["status"] == "scored": data["inputs"][row["native_file"]] = row["native_sha256"]
    groups = [dict(zip(evidence.GROUP_FIELDS, key), counts=counts(value))
              for key,value in sorted(grouped.items(), key=lambda kv: tuple(map(str,kv[0])))]
    buckets = [dict(bucket_xy_m=list(key), counts=counts(value)) for key,value in sorted(locations.items())]
    for name, expected in data["inputs"].items():
        require(sha(root/name) == expected, "Source changed during rescoring: "+name)
    output.mkdir(parents=True)
    with (output/"records.jsonl").open("w") as stream:
        for row in rows: stream.write(json.dumps(evidence.public_row(row), allow_nan=False)+"\n")
    contract = dict(version=VERSION, sources=list(sources), scene_contracts=list(scene_contracts),
                    original_scoring_versions=list(SUPPORTED), no_new_native_steps=True,
                    new_trials=0, no_training=True, cpu_workers=workers,
                    deduplication="native archive SHA256 plus column; conflicting aliases fail",
                    source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("board_first*.py")})
    (output/"contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    manifest = dict(inputs=data["inputs"],artifacts={name:sha(output/name) for name in ("records.jsonl","contract.json")})
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    changes = [dict(id=row["id"],source_study=row["source_study"],
                    bucket_xy_m=row["original"].get("bucket_xy_m"),
                    old_capture=row["old_score"]["route_success"],new_capture=row["new_score"]["route_success"],
                    route=row["new_score"].get("route_class"),events=row["events"],
                    board_first_status=row["new_score"].get("board_first_status"),
                    first_contact_surfaces=row["new_score"].get("first_contact_surfaces"),
                    first_board_rebound_s=row["new_score"].get("first_board_rebound_s"),
                    assistance_surfaces=row["new_score"].get("assistance_surfaces"))
               for row in rows if row["status"] == "scored" and
               row["old_score"]["route_success"] != row["new_score"]["route_success"]]
    summary = dict(version=VERSION,status="verified" if not totals["evidence_errors"] else "completed_with_errors",
                   metadata=data["metadata"],totals=totals,groups=groups,buckets=buckets,changes=changes,
                   exclusions=dict(Counter(row.get("reason","unknown") for row in rows if row["status"] not in ("scored","rejected"))),
                   manifest_sha256=sha(output/"manifest.json"),
                   scope="Recorded trace interval only; original scores preserved. Missing/truncated future outcomes are not inferred. This is historical rescoring, not fresh accuracy or repeatability evidence.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    audit = dict(version=AUDIT_VERSION,status="passed" if not totals["evidence_errors"] else "failed",
                 full_checks_verified=not totals["evidence_errors"],
                 checks_count=sum(row.get("checks_count",0) for row in rows)+len(data["inputs"])+2,
                 failures=[dict(id=row["id"],reason=row["reason"]) for row in rows if row["status"]=="evidence_error"],
                 summary_sha256=sha(output/"summary.json"),manifest_sha256=sha(output/"manifest.json"),
                 source_note="Both original and revised score reproduced independently per authenticated native row; does not re-audit original simulator execution.")
    (output/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--source",action="append",required=True)
    parser.add_argument("--scene-contract",action="append",default=[])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--index-only",action="store_true")
    args=parser.parse_args()
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
    result=run(args.root,args.output,sources=args.source,scene_contracts=args.scene_contract,
               workers=args.workers,index_only=args.index_only)
    print(json.dumps({k:result[k] for k in ("version","status","metadata")} | {"totals":result.get("totals")}))
    if result["status"]=="completed_with_errors":raise SystemExit(2)


if __name__ == "__main__":main()
