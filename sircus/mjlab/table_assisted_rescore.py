"""CPU-only, derivative rescoring of authenticated saved v2 native shots.

No simulator, GPU, model fitting or runtime activation. Old labels are retained
verbatim. Incompatible historical records and missing evidence stay unknown;
source contradictions fail the audit instead of becoming successful outcomes.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from . import calibrated_score as old
from . import table_assisted_groups
from .learning_contract import digest

grouping = table_assisted_groups.grouping

VERSION = "sircus-table-assisted-saved-rescore-v1"
AUDIT_VERSION = "sircus-table-assisted-saved-rescore-audit-v1"
# Added after the initial saved-shot run frozen at producer 8eb5452. Its original
# source hash and outcomes remain historical; this only changes future indexing.
INDEX_POLICY_VERSION = "sircus-table-assisted-original-index-v2"
OLD_FIELDS = ("scoring_version", "route_version", "reward_version", "success",
    "route_success", "complete_success", "board_bounce", "whole_ball_entry",
    "max_retention_s", "invalid_physics", "reward", "external_contact_steps",
    "whole_ball_entry_time_s", "capture_time_s", "calibration_fingerprint")
ROW_FIELDS = (*OLD_FIELDS, "id", "context_id", "method", "phase", "role", "native_release",
    "invalid", "policy_abstain", "native_cache_file", "native_cache_row", "native_cache_sha256",
    "bucket_xy_m", "values", "board_normal", "board_center_m", "release_ball_center_m", "substeps", "contact_protocol")
GROUP_FIELDS = ("source_study", "method", "phase", "substeps")
COUNTS = ("requested", "scorable_native", "rejected", "unscorable", "evidence_errors",
    "invalid_native", "old_captures", "new_captures", "new_table_assisted_captures",
    "losses", "unchanged_misses", "old_reward_sum", "new_reward_sum", "reward_count")


class EvidenceError(ValueError):
    pass


class Unscorable(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, label):
    if not condition:
        raise EvidenceError(label)


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if type(a) in (float, int) and type(b) in (float, int):
        return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-8
    return type(a) is type(b) and a == b


def inside(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root), "Evidence path escapes source root")
    return path


def context(root, file, scenes, inputs):
    """Find the nearest frozen contract and bind calibration/scene identity."""
    directory = file.parent
    while directory.is_relative_to(root):
        cp = directory/"contract.json"
        if cp.exists():
            break
        if directory == root:
            raise Unscorable("missing_study_contract")
        directory = directory.parent
    c = read(cp); inputs[str(cp.relative_to(root))] = sha(cp)
    if "fingerprint" in c:
        unsigned = dict(c); expected = unsigned.pop("fingerprint")
        require(expected in (digest(unsigned), hashlib.sha256(json.dumps(unsigned, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()), "Contract fingerprint differs")
    identity = c.get("identity", {})
    declared = identity.get("calibration") or c.get("calibration") or {}
    cfg = declared.get("score_config")
    cf = directory/"calibration.json"
    if cf.exists():
        inputs[str(cf.relative_to(root))] = sha(cf)
        frozen = c.get("frozen_files_sha256", {}).get("calibration.json") or declared.get("calibration_sha256")
        if frozen:
            require(inputs[str(cf.relative_to(root))] == frozen, "Calibration file differs from contract")
        saved = read(cf).get("score_config")
        if cfg is not None and saved is not None:
            require(cfg == saved, "Calibration configuration differs from contract")
        cfg = cfg if cfg is not None else saved
    if cfg is None:
        raise Unscorable("no_compatible_contact_calibration")
    cfg = old.validate_calibration(cfg)
    if declared.get("score_config_sha256"):
        require(digest(cfg) == declared["score_config_sha256"], "Calibration configuration hash differs")
    scene = c.get("scene")
    expected_scene = identity.get("scene_sha256")
    scene_source = cp
    if scene is None and expected_scene in scenes:
        scene, scene_source = scenes[expected_scene]
    if scene is None:
        raise Unscorable("missing_matching_scene")
    if expected_scene:
        require(digest(scene) == expected_scene, "Scene fingerprint differs from contract")
    inputs[str(scene_source.relative_to(root))] = sha(scene_source)
    return dict(source_study=str(directory.relative_to(root)), scene=scene, calibration=cfg,
        contract_file=str(cp.relative_to(root)), contract_sha256=sha(cp),
        scene_source=str(scene_source.relative_to(root)), scene_sha256=digest(scene),
        calibration_fingerprint=cfg["fingerprint"])


def archive_path(root, file, row, ctx):
    """Resolve only documented study layouts; the worker still verifies SHA."""
    name = row["native_cache_file"]
    step = row.get("substeps") or (row.get("contact_protocol") or {}).get("substeps")
    bases = (file.parent, root/ctx["source_study"])
    candidates = [base/name for base in bases] + [base/"native"/name for base in bases]
    if type(step) is int and step > 0:
        candidates += [base/"native"/str(step)/name for base in bases]
    for candidate in candidates:
        candidate = inside(root, candidate)
        if candidate.is_file(): return candidate
    raise Unscorable("missing_dense_native_archive")


def index(root, sources=(), scene_contracts=()):
    root = Path(root).resolve(); inputs = {}; scene_catalog = {}
    directories = [inside(root, root/s) for s in sources] if sources else [root]
    files = sorted({p.resolve() for d in directories for p in d.rglob("*.jsonl")
        if p.name in ("records.jsonl", "shots.jsonl")}, key=lambda p: (p.name != "records.jsonl", str(p)))
    require(files, "No historical shot indexes found")
    contracts = {p.resolve() for d in directories for p in d.rglob("contract.json")}
    contracts.update(inside(root, root/p) for p in scene_contracts)
    for cp in sorted(contracts):
        c = read(cp)
        if isinstance(c.get("scene"), dict):
            scene_catalog[digest(c["scene"])] = (c["scene"], cp)
    contexts = {}; rows = []; native_ids = {}; record_ids = {}; duplicates = 0; total = 0
    ignored_files = 0; ignored_rows = 0; original_files = 0; derivative_directories = {}
    for file in files:
        inputs[str(file.relative_to(root))] = sha(file)
        if file.parent not in derivative_directories:
            directory = file.parent; derivative = None
            while directory.is_relative_to(root):
                cp = directory/"contract.json"
                if cp.exists():
                    if read(cp).get("version") == VERSION:
                        derivative = cp
                    break
                if directory == root: break
                directory = directory.parent
            derivative_directories[file.parent] = derivative
        derivative = derivative_directories[file.parent]
        if derivative is not None:
            # A prior derivative is neither a trial nor a failed/unknown trial.
            # Bind the ignored bytes because they still determine inventory counts.
            inputs[str(derivative.relative_to(root))] = sha(derivative)
            ignored_files += 1
            with file.open() as stream: ignored_rows += sum(bool(line.strip()) for line in stream)
            continue
        original_files += 1
        if file.parent not in contexts:
            try: contexts[file.parent] = context(root, file, scene_catalog, inputs)
            except (Unscorable, ValueError, OSError) as exc:
                contexts[file.parent] = ("unscorable" if isinstance(exc, Unscorable) else "evidence_error", str(exc))
        ctx = contexts[file.parent]
        with file.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip(): continue
                total += 1
                provenance = dict(file=str(file.relative_to(root)), line=line_number,
                    metadata_sha256=inputs[str(file.relative_to(root))])
                try:
                    original = json.loads(line)
                    if not isinstance(original, dict): raise ValueError("Metadata row must be an object")
                except ValueError as exc:
                    rows.append(dict(id=f"malformed:{provenance['file']}:{line_number}", status="evidence_error",
                        reason="Malformed metadata JSON: "+str(exc), **grouping(file, {}, str(file.parent.relative_to(root))),
                        source=provenance, original={}, aliases=[])); continue
                row = {k: original.get(k) for k in ROW_FIELDS if k in original}
                rid = row.get("id") or f"unidentified:{provenance['file']}:{line_number}"
                item = dict(id=rid, original=row, source=provenance, aliases=[],
                    **grouping(file, row, ctx["source_study"] if isinstance(ctx, dict) else str(file.parent.relative_to(root))))
                if not isinstance(ctx, dict):
                    item.update(status=ctx[0], reason=ctx[1]); rows.append(item); continue
                item["context"] = ctx
                # Cache identity prevents duplicated records/shots indexes and copied
                # archives from manufacturing trials. Conflicting request IDs fail.
                native_key = (row.get("native_cache_sha256"), row.get("native_cache_row"))
                if native_key[0] is None: native_key = None
                identity = (item["source_study"], rid, digest([row.get("bucket_xy_m"), row.get("values")]))
                previous = native_ids.get(native_key) if native_key else record_ids.get(identity)
                if previous is not None:
                    if previous["id"] != rid or not all(same(previous["original"].get(k), row.get(k)) for k in (*OLD_FIELDS,"bucket_xy_m","values","board_normal")
                               if k in previous["original"] and k in row):
                        previous.update(status="evidence_error", reason="Duplicate metadata changes original labels")
                    if native_key and not all(same(previous["original"].get(k), row.get(k)) for k in
                            ("id", "bucket_xy_m", "values", "board_normal", "calibration_fingerprint", "native_cache_sha256", "native_cache_row")):
                        previous.update(status="evidence_error", reason="Duplicate native identity changes shot binding")
                    if previous.get("context", {}).get("scene_sha256") != ctx["scene_sha256"] or previous.get("context", {}).get("calibration_fingerprint") != ctx["calibration_fingerprint"]:
                        previous.update(status="evidence_error", reason="Duplicate native identity changes scene or calibration")
                    if previous["status"] == "unscorable" and previous.get("reason") == "missing_dense_native_archive":
                        try:
                            located = archive_path(root, file, row, ctx)
                            previous.update(status="pending_native", native_file=str(located.relative_to(root)))
                            previous.pop("reason", None)
                        except Unscorable: pass
                    previous["aliases"].append(provenance); duplicates += 1; continue
                if identity in record_ids and native_key != record_ids[identity].get("native_identity"):
                    item.update(status="evidence_error", reason="Same request identity maps to different native evidence")
                elif row.get("native_release") is False:
                    item.update(status="rejected" if row.get("invalid") or row.get("policy_abstain") is True else "unscorable",
                        reason=row.get("invalid") or ("policy_abstention" if row.get("policy_abstain") is True else "unexplained_nonrelease"))
                elif row.get("scoring_version") != old.VERSION:
                    item.update(status="unscorable", reason="incompatible_legacy_scorer")
                elif not all(row.get(k) is not None for k in ("native_cache_file", "native_cache_row", "native_cache_sha256")):
                    item.update(status="unscorable", reason="missing_dense_native_binding")
                else:
                    try:
                        candidate = archive_path(root, file, row, ctx)
                        require(type(row["native_cache_row"]) is int and row["native_cache_row"] >= 0,
                            "Invalid native archive row")
                        require(re.fullmatch(r"[0-9a-f]{64}", row["native_cache_sha256"]) is not None,
                            "Invalid native archive checksum")
                        require(row.get("calibration_fingerprint") == ctx["calibration_fingerprint"],
                            "Row calibration differs from source contract")
                        item.update(status="pending_native", native_file=str(candidate.relative_to(root)))
                    except (Unscorable, ValueError, OSError) as exc:
                        item.update(status="unscorable" if isinstance(exc, Unscorable) else "evidence_error", reason=str(exc))
                item["native_identity"] = native_key; rows.append(item); record_ids[identity] = item
                if native_key: native_ids[native_key] = item
    require(original_files > 0, f"No original historical shot indexes found; ignored {ignored_files} known derivative files ({ignored_rows} rows)")
    return dict(root=root, rows=rows, inputs=inputs, metadata=dict(files=original_files, rows=total,
        duplicate_rows=duplicates, unique_requests=len(rows), ignored_derived_files=ignored_files,
        ignored_derived_rows=ignored_rows), sources=list(sources), scene_contracts=list(scene_contracts))


def events(trace, scored):
    steps = scored["physics_steps"]
    times, force = trace["time_s"][:steps], trace["forces_N"][:steps]
    result = {}
    for col, name in enumerate(("board", "bucket", "table", "floor", "robot", "unattributed")):
        hits = np.flatnonzero(force[:, col] > .01)
        result["first_"+name+"_contact_s"] = float(times[hits[0]]) if len(hits) else None
    result.update(whole_ball_entry_s=scored.get("whole_ball_entry_time_s"),
        capture_s=scored.get("capture_time_s"), continuous_retention_s=scored.get("max_retention_s"))
    return result


def score_archive(job):
    """Each process decompresses an archive once; no engine imports or stepping."""
    from . import calibrated_score_oracle as old_oracle
    from . import table_assisted_score as new
    from . import table_assisted_score_oracle as new_oracle
    root, filename, members = job; path = Path(root)/filename; results = []
    try:
        actual_sha = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in ("record_ids", "position_m", "velocity_m_s", "forces_N", "time_s")}
            for k in ("contact_distance_m", "dt_s"):
                if k in archive: arrays[k] = archive[k]
        for item in members:
            row, ctx = item["original"], item["context"]
            try:
                require(actual_sha == row["native_cache_sha256"], "Native archive checksum differs")
                col = row["native_cache_row"]
                require(col < len(arrays["record_ids"]) and str(arrays["record_ids"][col]) == item["id"],
                    "Native archive record identity differs")
                trace = {k: arrays[k][:, col] for k in ("position_m", "velocity_m_s", "forces_N")}
                trace["time_s"] = arrays["time_s"]
                if "contact_distance_m" in arrays: trace["contact_distance_m"] = arrays["contact_distance_m"][:, col]
                if "dt_s" in arrays: trace["dt_s"] = arrays["dt_s"]
                step = item.get("substeps")
                if type(step) is int:
                    base_dt = ctx["scene"].get("simulation", {}).get("physics_dt_s")
                    if base_dt is not None:
                        require(np.allclose(np.diff(trace["time_s"]), float(base_dt)/step, rtol=0, atol=1e-12),
                            "Native sample cadence differs from declared substeps")
                args = (ctx["scene"], row["bucket_xy_m"], row["board_normal"], ctx["calibration"], trace)
                a, b = old.score_trace(*args), old_oracle.score_trace(*args)
                require(same(a, b), "Original independent scorers disagree")
                require(all(k in row and same(row[k], a[k]) for k in OLD_FIELDS), "Saved v2 labels differ from dense rescore")
                c, d = new.score_trace(*args), new_oracle.score_trace(*args)
                require(same(c, d), "New independent scorers disagree")
                require(not a["route_success"] or c["route_success"], "v3 lost a previously valid task capture")
                require(a["invalid_physics"] == c["invalid_physics"] and a["success"] == c["success"],
                    "Containment or invalid-physics definition changed")
                gained = bool(c["route_success"] and not a["route_success"])
                require(not gained or c["table_assisted_capture"], "New capture is not explained by table allowance")
                item.update(status="scored", old_score=a, new_score=c, events=events(trace, c),
                    native_sha256=actual_sha, native_row=col, checks=[dict(label=item["id"]+"/"+label, passed=True)
                        for label in ("native archive checksum", "native record identity", "original scorer oracle agreement",
                            "original metadata score agreement", "v3 scorer oracle agreement", "no task capture lost",
                            "containment and physics guards unchanged", "new credit explained by table allowance")])
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                item.update(status="evidence_error", reason=str(exc))
            results.append(item)
    except (OSError, ValueError, KeyError) as exc:
        for item in members:
            item.update(status="evidence_error", reason="Native archive unavailable: "+str(exc)); results.append(item)
    return results


def counts(rows):
    result = dict.fromkeys(COUNTS, 0)
    for r in rows:
        result["requested"] += 1; status = r["status"]
        if status != "scored":
            result[{"rejected":"rejected", "evidence_error":"evidence_errors"}.get(status, "unscorable")] += 1
            continue
        a, b = r["old_score"], r["new_score"]
        result["scorable_native"] += 1
        result["invalid_native"] += int(b["invalid_physics"])
        result["old_captures"] += int(a["route_success"])
        result["new_captures"] += int(b["route_success"])
        result["new_table_assisted_captures"] += int(b["route_success"] and not a["route_success"])
        result["losses"] += int(a["route_success"] and not b["route_success"])
        result["unchanged_misses"] += int(not a["route_success"] and not b["route_success"] and not b["invalid_physics"])
        result["old_reward_sum"] += a["reward"]; result["new_reward_sum"] += b["reward"]
        result["reward_count"] += 1
    return result


def public_row(item):
    out = {k:v for k,v in item.items() if k not in ("context", "native_identity")}
    if "context" in item:
        out["provenance"] = {k:v for k,v in item["context"].items() if k not in ("scene", "calibration", "source_study")}
    return out


def run(root, output, *, sources=(), scene_contracts=(), workers=2, index_only=False):
    output = Path(output).resolve(); require(not output.exists(), "New derivative output directory required")
    data = index(root, sources, scene_contracts); root = data["root"]
    require(all(not (root/name).is_relative_to(output) for name in data["inputs"]), "Output overlaps original evidence")
    rows = data["rows"]
    if index_only:
        output.mkdir(parents=True)
        samples = collections.defaultdict(list)
        for row in rows:
            reason = row.get("reason")
            if reason and len(samples[reason]) < 12:
                samples[reason].append(dict(id=row["id"], source=row["source"],
                    native_cache_file=row["original"].get("native_cache_file"),
                    native_cache_sha256=row["original"].get("native_cache_sha256"), source_study=row["source_study"]))
        summary = dict(version=VERSION, index_policy_version=INDEX_POLICY_VERSION, status="index_only", metadata=data["metadata"],
            by_status=dict(collections.Counter(r["status"] for r in rows)),
            reasons=dict(collections.Counter(r.get("reason") for r in rows if r.get("reason"))),
            pending_archives=len({r["native_file"] for r in rows if r["status"]=="pending_native"}),
            sources=data["sources"], inputs=data["inputs"], issue_examples=dict(samples))
        (output/"index.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        return summary
    require(1 <= workers <= 4, "Use one to four CPU workers")
    jobs = collections.defaultdict(list)
    for row in rows:
        if row["status"] == "pending_native": jobs[row["native_file"]].append(row)
    pending = [(str(root), name, members) for name,members in jobs.items()]
    def collect(groups):
        processed = []
        for number, group in enumerate(groups, 1):
            processed.extend(group)
            if number == 1 or number % 10 == 0 or number == len(pending):
                print(json.dumps(dict(phase="cpu_saved_rescore", archives_complete=number,
                    archives_total=len(pending), native_rows_processed=len(processed),
                    evidence_errors=sum(r["status"]=="evidence_error" for r in processed))), flush=True)
        return processed
    if workers == 1:
        processed = collect(map(score_archive, pending))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            processed = collect(pool.map(score_archive, pending))
    rows = [r for r in rows if r["status"] != "pending_native"] + processed
    # Single-process scoring mutates the pending entries, so deduplicate objects.
    rows = list({id(r):r for r in rows}.values())
    totals = counts(rows)
    require(totals["requested"] == data["metadata"]["unique_requests"], "Request accounting changed")
    require(totals["losses"] == 0 and totals["new_captures"]-totals["old_captures"] == totals["new_table_assisted_captures"],
        "New task-capture accounting changed")
    grouped = collections.defaultdict(list)
    for row in rows: grouped[tuple(row[k] for k in GROUP_FIELDS)].append(row)
    groups = [dict(zip(GROUP_FIELDS, key), counts=counts(value)) for key,value in sorted(grouped.items(), key=lambda kv:tuple(map(str,kv[0])))]
    for k in COUNTS:
        require(abs(sum(g["counts"][k] for g in groups)-totals[k]) <= 1e-7, "Grouped counts do not partition totals: "+k)
    for row in rows:
        if row["status"] == "scored": data["inputs"][row["native_file"]] = row["native_sha256"]
    input_checks=[]
    for name, expected in data["inputs"].items():
        require(sha(root/name)==expected, "Input changed during rescore: "+name)
        input_checks.append(dict(label="unchanged input/"+name,passed=True))
    output.mkdir(parents=True)
    with (output/"records.jsonl").open("w") as stream:
        for row in rows: stream.write(json.dumps(public_row(row), allow_nan=False)+"\n")
    examples = []
    for r in rows:
        if r["status"] != "scored" or not r["new_score"]["route_success"] or r["old_score"]["route_success"]: continue
        examples.append(dict(id=r["id"], source_study=r["source_study"], method=r["method"], phase=r["phase"], substeps=r["substeps"],
            old_capture=False, new_capture=True, old_reward=r["old_score"]["reward"], new_reward=r["new_score"]["reward"],
            events=r["events"], native_sha256=r["native_sha256"], native_row=r["native_row"],
            values=r["original"].get("values"), bucket_xy_m=r["original"].get("bucket_xy_m")))
    from . import table_assisted_score as new
    from . import table_assisted_score_oracle as new_oracle
    from . import calibrated_score_oracle as old_oracle
    source_files = [Path(m.__file__) for m in (old, old_oracle, new, new_oracle, table_assisted_groups)] + [Path(__file__)]
    contract = dict(version=VERSION, index_policy_version=INDEX_POLICY_VERSION, sources=data["sources"], scene_contracts=data["scene_contracts"],
        old_scoring_version=old.VERSION, new_scoring_version=new.VERSION, cpu_workers=workers,
        source_sha256={p.name:sha(p) for p in source_files}, no_new_native_steps=True, no_training=True,
        new_trials=0, new_training_requests=0,
        deduplication="native archive checksum + row; conflicting repeated request identity fails",
        reward_scope="authenticated native shots only; setup-rejection rewards are not fabricated")
    (output/"contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    manifest = dict(inputs=data["inputs"], artifacts={name:sha(output/name) for name in ("records.jsonl", "contract.json")})
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    summary = dict(version=VERSION, index_policy_version=INDEX_POLICY_VERSION,
        status="verified" if not totals["evidence_errors"] else "completed_with_errors",
        comparison_available=bool(totals["scorable_native"]),
        comparison_denominator=totals["scorable_native"]+totals["rejected"],
        denominator_scope="Scorable compatible native attempts plus explicit compatible nonrelease rejections; incompatible/unknown evidence stays separate. Not a fresh policy-accuracy evaluation.",
        metadata=data["metadata"], totals=totals, groups=groups, examples=examples[:50],
        example_count=len(examples), examples_display_limit=50,
        exclusions=dict(collections.Counter(r.get("reason", "unknown") for r in rows if r["status"] not in ("scored", "rejected"))),
        contract_sha256=sha(output/"contract.json"), manifest_sha256=sha(output/"manifest.json"),
        scope="CPU-only rescoring of saved native traces. Original labels are unchanged. No retraining, new rollouts, reliability claim or policy activation.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    checks = [c for r in rows for c in r.get("checks", [])] + input_checks
    checks += [dict(label="accounting/"+k,passed=True) for k in ("requested denominator", "zero lost captures", "group partition")]
    audit = dict(version=AUDIT_VERSION, status="passed" if not totals["evidence_errors"] else "failed",
        full_checks_verified=not totals["evidence_errors"], checks_count=len(checks), checks=checks,
        new_trials=0, new_training_requests=0,
        failures=[dict(id=r["id"], reason=r["reason"]) for r in rows if r["status"]=="evidence_error"],
        summary_sha256=sha(output/"summary.json"), manifest_sha256=sha(output/"manifest.json"),
        contract_sha256=sha(output/"contract.json"), totals=totals, groups=groups,
        source_note="Each native row independently scored by production and oracle for both definitions; this is not a re-audit of original simulator execution.")
    (output/"audit.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", action="append", default=[]); p.add_argument("--scene-contract", action="append", default=[])
    p.add_argument("--workers", type=int, default=2); p.add_argument("--index-only", action="store_true")
    a = p.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[name] = "1"
    result = run(a.root, a.output, sources=a.source, scene_contracts=a.scene_contract,
        workers=a.workers, index_only=a.index_only)
    print(json.dumps({k:result[k] for k in ("version", "status", "metadata")} | {"totals":result.get("totals"), "by_status":result.get("by_status")}))
    if result["status"] == "completed_with_errors": raise SystemExit(2)


if __name__ == "__main__": main()
