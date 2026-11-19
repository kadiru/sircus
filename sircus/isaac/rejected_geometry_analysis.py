"""Durable CPU-only replay of rejected TRAIN proposals through frozen geometry.

This does not run a policy, simulator, task shot or training step. A repaired
geometric setting has no new capture or native-motion result. The original
continuous geometry layer and all of its budgets remain unchanged.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

import candidate_train_evidence as original
import continuous_geometry_analysis as success_analysis
from continuous_feasibility import ACTION_NAMES, decode_action

VERSION = "sircus-rejected-geometry-analysis-v1"
SOURCE = Path(__file__).resolve().parent
REJECTED_COUNT = 6344
REJECTED_EXPORT_SHA256 = "9f48315598ff581ca43f85a82ff9feaf579da9b58035f97aa642530f8d10e1ad"
REJECTED_EXPORT_FINGERPRINT = "ded7c9eb70a85bd9fd49f944be27a67c4b5114c0741be4be6dbbb494bf3b53fd"
EXPECTED_REASONS = dict(arm_table_proxy_overlap_Arm14=4,
    board_bucket_proxy_overlap=27, board_table_proxy_overlap=41,
    endpoint_ik_unresolved_Arm10=2186, endpoint_ik_unresolved_Arm14=2479,
    endpoints_not_settled=13, outer_link_length_bound=1259,
    setup_motion_not_quiet=147, setup_transit_incomplete=188)
NATIVE_REASONS = {"endpoints_not_settled", "setup_motion_not_quiet", "setup_transit_incomplete"}
CLASSIFICATIONS = ("unchanged", "adjusted", "rejected")
FROZEN_LAYER_SHA256 = "c6489a44e268610470edd2a4c37bd4cfb1a78c89989883113f76996bd8fbad23"
FROZEN_AUDIT_SHA256 = "a76de94d28e01594fb9d399ee56caf844e6a30fc20bb428f67257a1391b63ab3"


def require(value, message):
    if not value:
        raise ValueError(message)


def signed(value):
    return dict(value, fingerprint=original.canonical_hash(value))


def unsigned(value):
    result = copy.deepcopy(value)
    fingerprint = result.pop("fingerprint", None)
    require(fingerprint == original.canonical_hash(result), "Artifact fingerprint mismatch")
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(rejected_evidence, expected_rejected_sha256, success_evidence,
                benchmark_metadata, assets, published):
    """Verify original TRAIN-only receipts and exact rejected proposal coverage."""
    original._sha(expected_rejected_sha256)
    require(digest(rejected_evidence) == expected_rejected_sha256 == REJECTED_EXPORT_SHA256,
            "Rejected TRAIN export SHA mismatch")
    raw = success_analysis.plain_read(rejected_evidence)
    export = unsigned(raw)
    require(raw["fingerprint"] == REJECTED_EXPORT_FINGERPRINT
            and export["schema"] == "sircus-seed22-paid-train-rejections-v1"
            and export["seed"] == original.SEED, "Wrong rejected TRAIN evidence lineage")
    base = success_analysis.load_inputs(success_evidence, benchmark_metadata, assets, published)
    successful = success_analysis.plain_read(success_evidence)
    require(export["source_bindings"] == successful["source_bindings"]
            and export["source"] == successful["source"]
            and export["training_totals"] == successful["training_totals"]
            and export["train_trial_sha256"] == successful["train_trial_sha256"]
            and export["cells"] == successful["cells"]
            and export["successful_train_export_sha256"] == success_analysis.TRAIN_EXPORT_SHA256
            and export["successful_train_export_fingerprint"] == successful["fingerprint"],
            "Rejected export disagrees with independently bound original TRAIN evidence")
    require(export["rejected_count"] == export["training_totals"]["rejected"] == REJECTED_COUNT
            and export["rejection_reasons"] == EXPECTED_REASONS
            and export["rejection_stages"] == dict(Counter(
                "native_motion_or_settling" if reason in NATIVE_REASONS else "geometry_or_path"
                for reason,count in EXPECTED_REASONS.items() for _ in range(count))),
            "Original rejection population or cause counts changed")
    require(export["evaluated_trial_files_read"] == export["new_native_shots"]
            == export["optimizer_updates"] == 0, "Not TRAIN-only immutable evidence")
    require(export["scene"] == base["scene"], "Original scene changed")
    require(digest(SOURCE/"continuous_geometry_layer.py") == FROZEN_LAYER_SHA256
            and digest(SOURCE/"continuous_geometry_audit.py") == FROZEN_AUDIT_SHA256,
            "The previously tested geometry layer or independent reader changed")
    by_cell = {cell["id"]:cell for cell in successful["cells"]}
    cases, identities, reasons = [], set(), Counter()
    for row in export["rejections"]:
        key, cell, action = row["rejection_key"], row["cell"], row["action"]
        require(key not in identities and cell == by_cell.get(cell["id"])
                and cell["split"] == "train", "Duplicate or non-TRAIN rejection")
        require(original.TRAIN_PATH.fullmatch(row["trial_file"])
                and export["train_trial_sha256"][row["trial_file"]] == row["trial_sha256"]
                and type(row["slot"]) is int and 0 <= row["slot"] < 35
                and key == row["trial_file"]+"#"+str(row["slot"]), "Rejected TRAIN identity changed")
        reason = row["original_reason"]
        require(reason in EXPECTED_REASONS and type(row["home_path_eligible"]) is bool,
                "Malformed original rejection metadata")
        stage = "native_motion_or_settling" if reason in NATIVE_REASONS else "geometry_or_path"
        require(row["rejection_stage"] == stage, "Original rejection stage relabelled")
        if stage == "native_motion_or_settling":
            require(row["home_path_eligible"] is True
                    and row["qualification"]["accepted"] is True
                    and row["path_proxy"]["accepted"] is True,
                    "A native motion rejection lacks its original admitted geometry and path")
        decoded = decode_action(cell, [action[k] for k in ACTION_NAMES], action["id"])
        require(all(abs(action[k]-value) <= 1e-8 for k,value in decoded.items() if k != "id"),
                "Rejected TRAIN action geometry changed")
        starts = original._joints(row["recorded_starts_rad"])
        cases.append(dict(case_id=key, cell=copy.deepcopy(cell), action=copy.deepcopy(action),
            starts=starts, seeds=copy.deepcopy(starts), original_reason=reason, rejection_stage=stage,
            original_geometry_admitted=bool(row["home_path_eligible"]
                and (row.get("qualification") or {}).get("accepted") is True
                and (row.get("path_proxy") or {}).get("accepted") is True),
            home_path_eligible=row["home_path_eligible"], source_trial_file=row["trial_file"],
            source_trial_sha256=row["trial_sha256"], slot=row["slot"]))
        identities.add(key)
        reasons[reason] += 1
    require(len(cases) == REJECTED_COUNT and dict(reasons) == EXPECTED_REASONS,
            "Rejected TRAIN case coverage changed")
    paths = list(base["protected_before"])
    paths += [rejected_evidence] + [SOURCE/name for name in (
        "rejected_train_evidence.py", "rejected_geometry_analysis.py", "rejected_geometry_audit.py")]
    protected = success_analysis.snapshot(paths)
    require(all(protected[p] == sha for p,sha in base["protected_before"].items()),
            "Protected original input changed while loading rejected evidence")
    bindings = copy.deepcopy(base["bindings"])
    bindings.update(rejected_export_sha256=expected_rejected_sha256,
        rejected_export_fingerprint=raw["fingerprint"], original_rejection_reasons=EXPECTED_REASONS,
        geometry_layer_sha256=FROZEN_LAYER_SHA256, geometry_reader_sha256=FROZEN_AUDIT_SHA256,
        new_source_sha256={name:digest(SOURCE/name) for name in (
            "rejected_train_evidence.py", "rejected_geometry_analysis.py", "rejected_geometry_audit.py")})
    return dict(schema=VERSION+"-inputs", synthetic=False, scene=base["scene"], links=base["links"],
        cases=cases, expected_counts=dict(requested=REJECTED_COUNT), bindings=bindings,
        train_trial_sha256=copy.deepcopy(export["train_trial_sha256"]), protected_before=protected,
        protected_roots=base["protected_roots"])


def _counts(rows):
    totals = Counter(row["result"]["classification"] for row in rows)
    require(set(totals) <= set(CLASSIFICATIONS), "Unknown geometry classification")
    return dict(requested=len(rows), **{key:totals[key] for key in CLASSIFICATIONS})


def summarize(rows):
    """Geometry accounting; unchanged never establishes a repaired native run."""
    by_original = {}
    for reason in sorted({r["original_reason"] for r in rows}):
        by_original[reason] = _counts([r for r in rows if r["original_reason"] == reason])
    remaining, first, terminations = Counter(), Counter(), Counter()
    for row in rows:
        result = row["result"]
        first[result["original_preflight"]["reason"] or "admitted"] += 1
        repair = result["repair"]
        terminations[(repair.get("optimizer") or {}).get("termination", "not_attempted")] += 1
        if result["classification"] == "rejected":
            proof = repair.get("candidate_preflight")
            remaining[(proof or result["original_preflight"])["reason"]] += 1
    native = [r for r in rows if r["rejection_stage"] == "native_motion_or_settling"]
    return dict(counts=_counts(rows), by_original_reason=by_original,
        first_geometry_reasons=dict(sorted(first.items())),
        remaining_geometry_reasons=dict(sorted(remaining.items())),
        repair_terminations=dict(sorted(terminations.items())),
        unchanged_native_rejections=sum(r["result"]["classification"] == "unchanged" for r in native),
        adjusted_native_rejections=sum(r["result"]["classification"] == "adjusted" for r in native))


def evaluate_case(job):
    """The exact previously tested default layer, with fresh measured-start IK."""
    from continuous_geometry_layer import filter_shot
    scene, links, case, manifest_fingerprint = job
    wall, cpu = time.monotonic(), time.process_time()
    result = filter_shot(scene, case["cell"], case["action"], links, case["starts"])
    return signed(dict(case_id=case["case_id"], cell_id=case["cell"]["id"],
        original_reason=case["original_reason"], rejection_stage=case["rejection_stage"],
        home_path_eligible=case["home_path_eligible"], result=result,
        analysis_binding=manifest_fingerprint, wall_s=time.monotonic()-wall, cpu_s=time.process_time()-cpu))


def _write_new(path, value):
    """Fsync then atomically publish a new record; leave interrupted temps intact."""
    require(not path.exists(), "Refuse to replace saved geometry evidence")
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pending-", delete=False) as out:
        json.dump(value, out, separators=(",", ":"), allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
        pending = Path(out.name)
    require(not path.exists(), "Saved geometry record appeared concurrently")
    # link() publishes without replacing a destination that appeared after the
    # check. Remove only the temporary alias after the committed bytes exist.
    os.link(pending, path)
    pending.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _check_inputs(inputs):
    require(success_analysis.snapshot(inputs["protected_before"]) == inputs["protected_before"],
            "Protected input or source bytes changed")
    cases = inputs["cases"]
    require(len(cases) == inputs["expected_counts"]["requested"]
            and len({c["case_id"] for c in cases}) == len(cases), "Rejected case coverage mismatch")
    require(all(c["seeds"] == c["starts"] for c in cases), "Historical IK seeds forbidden")
    require(inputs.get("synthetic") is True or len(cases) == REJECTED_COUNT,
            "Real analysis requires every original rejected TRAIN proposal")


def _manifest(inputs):
    return signed(dict(schema=VERSION+"-manifest", bindings=inputs["bindings"],
        synthetic=inputs.get("synthetic") is True, case_ids=[c["case_id"] for c in inputs["cases"]],
        cases_fingerprint=original.canonical_hash(inputs["cases"]),
        protected_before=inputs["protected_before"], protected_roots=inputs.get("protected_roots", []),
        expected_counts=inputs["expected_counts"], layer_parameters=dict(
            repair=True, repair_max_nfev=24, repair_max_evaluations=512, max_normalized_adjustment=.10)))


def _output_path(output, inputs, resume):
    require(not Path(output).is_symlink(), "Geometry output cannot be a symlink")
    output = Path(output).resolve()
    for name in list(inputs["protected_before"])+list(inputs.get("protected_roots", [])):
        protected = Path(name).resolve()
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                "Output overlaps protected original input or experiment")
    require(type(resume) is bool and output.exists() is resume,
            "Output must be new, or explicitly resumed")
    if not resume:
        output.mkdir(parents=True)
        (output/"rows").mkdir()
        (output/"invocations").mkdir()
    require(output.is_dir() and (output/"rows").is_dir() and (output/"invocations").is_dir(),
            "Malformed existing geometry output")
    require(not any((output/name).is_symlink() for name in ("rows", "invocations", "writer.lock", "manifest.json"))
            and all((output/name).resolve().is_relative_to(output) for name in ("rows", "invocations")),
            "Geometry output children cannot be redirected")
    require(not (output/"analysis.json").exists(), "Completed geometry analysis cannot be reopened")
    return output


def analyze_cases(inputs, output, *, workers=3, resume=False, profile=False, evaluate=None, progress=None):
    """Compute absent records only; profile fixes one earliest row per old cause.

    Resume reuses exact source/input-bound completed CPU records, not simulation
    requests. Any changed input, source, case, setting or malformed record fails
    closed. A nonblocking file lock prevents concurrent writers to this output.
    """
    require(type(workers) is int and 1 <= workers <= 4 and type(profile) is bool,
            "Expected one to four CPU workers and explicit profile mode")
    require(evaluate is None or inputs.get("synthetic") is True,
            "Real geometry evaluation cannot be replaced")
    require(evaluate is None or workers == 1, "Synthetic hooks require one worker")
    _check_inputs(inputs)
    output = _output_path(output, inputs, resume)
    manifest = _manifest(inputs)
    with (output/"writer.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Geometry output already has an active writer") from exc
        if resume:
            require(success_analysis.plain_read(output/"manifest.json") == manifest,
                    "Resume source/input/case manifest changed")
        else:
            _write_new(output/"manifest.json", manifest)
        rows, record_hashes = {}, {}
        cases = inputs["cases"]
        allowed = {f"{index:06d}.json" for index in range(len(cases))}
        require(all(not p.is_symlink() and p.is_file() and (p.name in allowed or p.name.startswith(".pending-"))
                    for p in (output/"rows").iterdir()),
                "Unexpected durable geometry record")
        for index, case in enumerate(cases):
            path = output/"rows"/f"{index:06d}.json"
            if path.exists():
                row = success_analysis.plain_read(path)
                unsigned(row)
                require(row["case_id"] == case["case_id"] and row["analysis_binding"] == manifest["fingerprint"],
                        "Saved geometry row identity or binding changed")
                from rejected_geometry_audit import audit_row
                audit_row(row, case, inputs)
                rows[index], record_hashes[path.name] = row, digest(path)
        reused = len(rows)
        selected = set(range(len(cases)))
        if profile:
            selected, seen = set(), set()
            for index, case in enumerate(cases):
                if case["original_reason"] not in seen:
                    selected.add(index)
                    seen.add(case["original_reason"])
        missing = [index for index in sorted(selected) if index not in rows]
        started, cpu = time.monotonic(), time.process_time()
        jobs = [(inputs["scene"], inputs["links"], cases[i], manifest["fingerprint"]) for i in missing]
        def consume(iterator):
            for index, row in zip(missing, iterator):
                require(row["case_id"] == cases[index]["case_id"] and
                        row["analysis_binding"] == manifest["fingerprint"], "Computed row identity changed")
                unsigned(row)
                path = output/"rows"/f"{index:06d}.json"
                _write_new(path, row)
                rows[index], record_hashes[path.name] = row, digest(path)
                if progress is not None and (len(rows) % 25 == 0 or index == missing[-1]):
                    progress(dict(completed=len(rows), total=len(cases), computed=len(rows)-reused,
                        counts=summarize(list(rows.values()))["counts"], wall_s=time.monotonic()-started))
        if workers == 1:
            consume(map(evaluate_case if evaluate is None else evaluate, jobs))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                consume(pool.map(evaluate_case, jobs, chunksize=1))
        _check_inputs(inputs)
        invocation = signed(dict(schema=VERSION+"-invocation", analysis_binding=manifest["fingerprint"],
            profile=profile, rows_reused=reused, rows_computed=len(rows)-reused, completed=len(rows),
            wall_s=time.monotonic()-started, coordinator_cpu_s=time.process_time()-cpu,
            new_row_worker_cpu_s=sum(rows[i]["cpu_s"] for i in missing)))
        previous = list((output/"invocations").glob("[0-9]*.json"))
        require(sorted(p.name for p in previous) == [f"{i:06d}.json" for i in range(len(previous))]
                and all(not p.is_symlink() and p.is_file() for p in previous),
                "Durable resource receipt sequence changed")
        for path in previous:
            old = success_analysis.plain_read(path)
            unsigned(old)
            require(old["analysis_binding"] == manifest["fingerprint"], "Resource receipt binding changed")
        invocation_index = len(previous)
        _write_new(output/"invocations"/f"{invocation_index:06d}.json", invocation)
        ordered = [rows[i] for i in sorted(rows)]
        complete = len(rows) == len(cases)
        report = dict(schema=VERSION, status="geometry_complete_audit_pending" if complete else "geometry_profile_partial",
            seed=20261122, synthetic=inputs.get("synthetic") is True, rows=ordered, **summarize(ordered),
            bindings=copy.deepcopy(inputs["bindings"]), protected_before=inputs["protected_before"],
            protected_after=success_analysis.snapshot(inputs["protected_before"]),
            protected_roots=inputs.get("protected_roots", []), analysis_manifest_fingerprint=manifest["fingerprint"],
            durable_row_sha256=record_hashes, workers=workers, rows_reused=reused, rows_computed=len(rows)-reused,
            row_worker_cpu_s=sum(r["cpu_s"] for r in ordered),
            invocation_resources=invocation, optimizer_updates=0, new_native_shots=0,
            candidate_bank_read=False, model_loaded=False, held_out_shot_files_read=0,
            capture_replayed=False, native_execution_verified=False, release_authorized=False,
            scope="Geometry-only recheck and bounded repair of old rejected TRAIN proposals. Unchanged or adjusted geometry does not establish native settling, release, capture or a successful task. No saved endpoint lookup, model learning, simulator, new shot or held-out outcome.")
        return signed(report)


def write_results(output, report, audit):
    output = Path(output)
    require(report["status"] == "geometry_complete_audit_pending" and audit["status"] in ("passed", "passed_synthetic"),
            "Cannot finalize an incomplete or unaudited geometry analysis")
    _write_new(output/"analysis.json", report)
    _write_new(output/"audit.json", audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejected-evidence", type=Path, required=True)
    parser.add_argument("--expected-rejected-sha256", required=True)
    parser.add_argument("--success-evidence", type=Path, required=True)
    parser.add_argument("--benchmark-metadata", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    inputs = load_inputs(args.rejected_evidence, args.expected_rejected_sha256, args.success_evidence,
                         args.benchmark_metadata, args.assets, args.published)
    report = analyze_cases(inputs, args.output, workers=args.workers, resume=args.resume, profile=args.profile,
        progress=lambda value:print(json.dumps(value), flush=True))
    if report["status"] == "geometry_profile_partial":
        print(json.dumps(dict(status=report["status"], **summarize(report["rows"]),
            invocation_resources=report["invocation_resources"])), flush=True)
        return
    from rejected_geometry_audit import audit_analysis
    audit = audit_analysis(report, inputs)
    write_results(args.output, report, audit)
    print(json.dumps(dict(status=audit["status"], **summarize(report["rows"]))), flush=True)


if __name__ == "__main__":
    main()
