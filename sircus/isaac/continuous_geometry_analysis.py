"""CPU-only retention check for continuous shot geometry, without a shot bank.

Inputs are exact successful paid TRAIN requests from the original seed-22
collector. The default IK seeds are each request's measured arm starts, as in
the direct native route. Historical endpoint fits are not supplied to the
solver. No policy, optimizer, simulator, ball-flight scorer or catalogue is
loaded. Raw per-request records stay outside Git.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import xml.etree.ElementTree as ET

import candidate_train_evidence as collector
from continuous_feasibility import ACTION_NAMES, decode_action
from prepare_assets import chain

VERSION = "sircus-continuous-geometry-train-retention-v1"
SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
SUCCESS_COUNT = 1534
CLEAN_COUNT = 485
# These identify the historical TRAIN-only export, not a candidate catalogue.
TRAIN_EXPORT_SHA256 = "6053ff58e026bb288f9a9f18fd6700ee32ba613af771318d006def753e580233"
TRAIN_EXPORT_FINGERPRINT = "666ee9beb093412d12c2f5cc4a242c7e519969f3b3d1569a4ccea268dd5cdc25"
CORE_FILES = ("continuous_feasibility.py", "expanded_pose_menu.py",
              "feasibility_contract.py", "prepare_assets.py", "robust_ik.py",
              "expanded_pose_native.py")
LIVE_FILES = ("joint_feasibility.py", "settled_transit.py", "shot_benchmark_reach_audit.py",
              "candidate_train_evidence.py", "continuous_geometry_layer.py",
              "continuous_geometry_analysis.py", "continuous_geometry_audit.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain_read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda x: (
        _ for _ in ()).throw(ValueError("Nonfinite JSON " + x)))


def unsigned(value):
    result = copy.deepcopy(value)
    fingerprint = result.pop("fingerprint", None)
    require(fingerprint == collector.canonical_hash(result), "Evidence fingerprint mismatch")
    return result


def snapshot(paths):
    result = {}
    for path in paths:
        path = Path(path).resolve()
        require(path.is_file(), "Missing protected input: " + str(path))
        result[str(path)] = digest(path)
    return dict(sorted(result.items()))


def load_inputs(train_evidence, benchmark_metadata, assets, published, *,
                expected_export_sha256=TRAIN_EXPORT_SHA256):
    """Bind the TRAIN export, original receipts/model bytes and unchanged math.

    ``train_evidence`` may be freshly recollected over the read-only original
    collector. Its exact original raw SHA and canonical fingerprint are fixed.
    The collector verified all 289 TRAIN packets / 10,000 paid slots. This
    loader reads no evaluation packet, trajectory, saved IK cache or bank.
    """
    require(digest(train_evidence) == expected_export_sha256 == TRAIN_EXPORT_SHA256,
            "Original TRAIN-only export SHA mismatch")
    raw = plain_read(train_evidence)
    evidence = unsigned(raw)
    require(raw["fingerprint"] == TRAIN_EXPORT_FINGERPRINT
            and evidence["schema"] == collector.VERSION and evidence["seed"] == collector.SEED,
            "Expected original seed-22 TRAIN evidence")
    require(evidence["successful_shots"] == evidence["training_totals"]["captures"] == SUCCESS_COUNT
            and evidence["clean_successes"] == evidence["training_totals"]["clean"] == CLEAN_COUNT
            and evidence["training_totals"]["attempts"] == 10000
            and len(evidence["train_trial_sha256"]) == 289,
            "Original TRAIN request/success denominator mismatch")
    require(evidence["evaluated_trial_files_read"] == evidence["new_native_shots"]
            == evidence["optimizer_updates"] == 0, "Not a TRAIN-only no-execution export")
    reference = plain_read(published)
    runs = [r for r in reference["runs"] if r["method"] == "ppo" and r["seed"] == collector.SEED]
    require(len(runs) == 1, "Expected one original published seed-22 run")
    original = runs[0]
    binding = evidence["source_bindings"]
    require(binding["audit_sha256"] == original["independent_audit"]["sha256"]
            and binding["progress_sha256"] == original["independent_audit"]["source_progress_sha256"]
            and binding["checkpoint_sha256"] == original["checkpoint"]["sha256"]
            and binding["experiment_sha256"] == reference["experiment_sha256"]
            and binding["split_manifest_sha256"] == reference["split_manifest_sha256"]
            and evidence["source"] == reference["source"], "Original published lineage mismatch")
    benchmark, assets = Path(benchmark_metadata).resolve(), Path(assets).resolve()
    run = benchmark / "runs/ppo-seed20261122"
    audit_path = Path(str(run) + ".audit.json")
    audit = collector.read(audit_path, binding["audit_sha256"])
    require(audit["status"] == "passed" and audit["smoke_only"] is False
            and audit["paid_train_attempts"] == 10000 and audit["seed"] == collector.SEED
            and audit["source_progress_sha256"] == binding["progress_sha256"],
            "Original independent production audit not passed")
    experiment_path, progress_path = benchmark/"experiment.json", run/"progress.json"
    experiment = collector.read(experiment_path, binding["experiment_sha256"])
    splits_path = collector._inside(benchmark, experiment["split_manifest_file"])
    splits = collector.read(splits_path, binding["split_manifest_sha256"])
    progress = collector.read(progress_path, binding["progress_sha256"])
    contract_path, source_binding_path = run/"contract.json", run/"experiment-binding.json"
    contract = collector.read(contract_path, binding["contract_sha256"])
    source_binding = collector.read(source_binding_path, binding["binding_sha256"])
    require(evidence["scene"] == contract["scene"]
            and evidence["cells"] == [collector._cell(cell) for cell in splits["train"]]
            and progress["source"] == experiment["source"] == source_binding["source"] == evidence["source"],
            "Scene, TRAIN coordinates or frozen source mismatch")
    finals = [v for v in progress["checkpoints"] if v["training_attempts"] == 10000]
    require(len(finals) == 1, "Ambiguous original final model")
    model_path = collector._inside(run, finals[0]["checkpoint_file"])
    require(digest(model_path) == finals[0]["checkpoint_sha256"] == binding["checkpoint_sha256"],
            "Original model bytes changed")
    # Hash only: this module never imports Torch or loads the model.
    paths = [train_evidence, published, audit_path, experiment_path, splits_path,
             progress_path, contract_path, source_binding_path, model_path]
    for name in CORE_FILES:
        path = SOURCE/name
        require(digest(path) == audit["source_sha256"][name], "Original geometry math changed: " + name)
        paths.append(path)
    receipt_path = ROOT/"baselines/isaac-reach-ik-audit-20261106/summary.json"
    receipt = plain_read(receipt_path)
    require(receipt["snapshot"]["source"] == evidence["source"]
            and receipt["dependency_sha256"]["contract.json"] == binding["contract_sha256"],
            "Original geometry receipt mismatch")
    paths.append(receipt_path)
    asset_paths = {name: assets/file for name, file in
                   (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))}
    for path in asset_paths.values():
        require(digest(path) == receipt["dependency_sha256"]["assets/"+path.name], "Original arm asset changed")
    paths.extend(asset_paths.values())
    links = {name: chain(ET.fromstring(path.read_bytes())) for name, path in asset_paths.items()}
    cells = {c["id"]: c for c in evidence["cells"]}
    require(len(cells) == 35 and len(evidence["successes"]) == SUCCESS_COUNT,
            "Original TRAIN context coverage mismatch")
    cases, keys = [], set()
    for row in evidence["successes"]:
        key, cell, action = row["success_key"], row["cell"], row["action"]
        require(key not in keys and cell == cells.get(cell["id"]) and cell["split"] == "train",
                "Duplicate or non-TRAIN success")
        require(collector.TRAIN_PATH.fullmatch(row["trial_file"])
                and evidence["train_trial_sha256"][row["trial_file"]] == row["trial_sha256"]
                and key == row["trial_file"]+"#"+str(row["slot"])
                and type(row["slot"]) is int and 0 <= row["slot"] < 35
                and type(row["clean"]) is bool, "Successful TRAIN identity mismatch")
        collector._check_success(action, dict(action, invalid=None,
                                             qualification=row["qualification"],
                                             path_proxy=row["path_proxy"]))
        decoded = decode_action(cell, [action[name] for name in ACTION_NAMES], action["id"])
        require(all(abs(action[name]-value) <= 1e-8 for name,value in decoded.items() if name != "id"),
                "Successful TRAIN action derived geometry mismatch")
        starts = collector._joints(row["recorded_starts_rad"])
        cases.append(dict(case_id=key, cell=copy.deepcopy(cell), action=copy.deepcopy(action),
                          starts=starts, seeds=copy.deepcopy(starts), clean=row["clean"],
                          source_trial_file=row["trial_file"], source_trial_sha256=row["trial_sha256"],
                          slot=row["slot"]))
        keys.add(key)
    require(sum(c["clean"] for c in cases) == CLEAN_COUNT, "Clean success denominator mismatch")
    paths.extend(SOURCE/name for name in LIVE_FILES)
    protected = snapshot(paths)
    expected_files = {str(Path(train_evidence).resolve()):TRAIN_EXPORT_SHA256,
        str(audit_path):binding["audit_sha256"], str(experiment_path):binding["experiment_sha256"],
        str(splits_path):binding["split_manifest_sha256"], str(progress_path):binding["progress_sha256"],
        str(contract_path):binding["contract_sha256"], str(source_binding_path):binding["binding_sha256"],
        str(model_path):binding["checkpoint_sha256"]}
    expected_files.update({str((SOURCE/name).resolve()):audit["source_sha256"][name] for name in CORE_FILES})
    expected_files.update({str(path.resolve()):receipt["dependency_sha256"]["assets/"+path.name]
                           for path in asset_paths.values()})
    require(all(protected[path] == expected for path,expected in expected_files.items()),
            "Verified original input changed while loading")
    import numpy
    import scipy
    return dict(schema=VERSION+"-inputs", synthetic=False, scene=copy.deepcopy(evidence["scene"]),
                links=links, cases=cases, protected_before=protected,
                protected_roots=[str(benchmark), str(assets)],
                expected_counts=dict(requested=SUCCESS_COUNT, clean=CLEAN_COUNT),
                bindings=dict(train_export_sha256=digest(train_evidence),
                    train_export_fingerprint=raw["fingerprint"], original=copy.deepcopy(binding),
                    original_source=copy.deepcopy(evidence["source"]),
                    verified_paid_train_packets=289, verified_paid_train_requests=10000,
                    seed_strategy="actual_per_shot_measured_starts",
                    historical_endpoint_witnesses_used=False,
                    candidate_bank_read=False, model_loaded=False, held_out_shot_files_read=0,
                    runtime=dict(python=sys.version, executable=sys.executable,
                                 platform=platform.platform(), numpy=numpy.__version__, scipy=scipy.__version__),
                    geometry_source_sha256={name:digest(SOURCE/name) for name in CORE_FILES},
                    live_source_sha256={name:digest(SOURCE/name) for name in LIVE_FILES}))


def _evaluate(job):
    from continuous_geometry_layer import filter_shot
    scene, links, case = job
    began, cpu = time.monotonic(), time.process_time()
    result = filter_shot(scene, case["cell"], case["action"], links, case["starts"])
    return dict(case_id=case["case_id"], cell_id=case["cell"]["id"], clean=case["clean"],
                result=result, wall_s=time.monotonic()-began, cpu_s=time.process_time()-cpu)


def _counts(rows):
    counts = Counter(row["result"]["classification"] for row in rows)
    require(set(counts) <= {"unchanged", "adjusted", "rejected"}, "Unknown geometry classification")
    return dict(requested=len(rows), **{key: counts[key] for key in ("unchanged", "adjusted", "rejected")})


def analyze_cases(inputs, *, workers=1, evaluate=None, progress=None):
    """Freshly check each exact continuous action; test hooks are synthetic-only."""
    require(type(workers) is int and 1 <= workers <= 4, "Expected one to four CPU workers")
    require(evaluate is None or inputs.get("synthetic") is True, "Real evidence cannot replace geometry checks")
    require(evaluate is None or workers == 1, "Synthetic custom evaluation requires one worker")
    cases = inputs["cases"]
    require(len(cases) == inputs["expected_counts"]["requested"]
            and len({c["case_id"] for c in cases}) == len(cases)
            and sum(c["clean"] for c in cases) == inputs["expected_counts"]["clean"],
            "Case/clean coverage mismatch")
    require(all(c["seeds"] == c["starts"] for c in cases), "Primary analysis forbids historical IK lookup")
    before = snapshot(inputs["protected_before"])
    require(before == inputs["protected_before"], "Protected input changed before analysis")
    jobs = [(inputs["scene"], inputs["links"], case) for case in cases]
    began, cpu = time.monotonic(), time.process_time()
    rows = []
    def consume(iterator):
        for row in iterator:
            rows.append(row)
            if progress is not None and (len(rows) % 25 == 0 or len(rows) == len(cases)):
                progress(dict(completed=len(rows), total=len(cases), counts=_counts(rows),
                              wall_s=time.monotonic()-began))
    if workers == 1:
        consume(map(_evaluate if evaluate is None else evaluate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            consume(pool.map(_evaluate, jobs, chunksize=1))
    after = snapshot(inputs["protected_before"])
    require(after == before, "Protected input changed during analysis")
    report = dict(schema=VERSION, status="geometry_complete_audit_pending", seed=collector.SEED,
                  synthetic=inputs.get("synthetic") is True, rows=rows, counts=_counts(rows),
                  clean_counts=_counts([r for r in rows if r["clean"]]),
                  reasons=dict(sorted(Counter(r["result"]["original_preflight"]["reason"]
                      for r in rows if not r["result"]["original_preflight"]["accepted"]).items())),
                  final_rejection_reasons=dict(sorted(Counter(r["result"]["reason"] for r in rows
                      if r["result"]["classification"] == "rejected").items())),
                  bindings=copy.deepcopy(inputs["bindings"]), protected_before=before, protected_after=after,
                  protected_roots=copy.deepcopy(inputs.get("protected_roots", [])),
                  workers=workers, wall_s=time.monotonic()-began,
                  worker_cpu_s=sum(r["cpu_s"] for r in rows), coordinator_cpu_s=time.process_time()-cpu,
                  optimizer_updates=0, new_native_shots=0, model_loaded=False,
                  held_out_shot_files_read=0, candidate_bank_read=False, release_authorized=False,
                  scope="CPU geometry retention of recorded successful TRAIN actions at original measured starts; fresh original robust IK seeded from those starts. No historical endpoint warm starts, shot catalogue, model update, ball-flight replay, new capture or generalization measurement.")
    report["fingerprint"] = collector.canonical_hash(report)
    return report


def validate_output(output, values):
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to replace a geometry analysis output")
    for path in list(values["protected_before"])+list(values.get("protected_roots", [])):
        protected = Path(path).resolve()
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                "Output overlaps a protected input")
    return output


def write_results(output, report, audit):
    output = validate_output(output, report)
    output.mkdir(parents=True)
    for name, value in (("analysis.json", report), ("audit.json", audit)):
        (output/name).write_text(json.dumps(value, separators=(",", ":"), allow_nan=False)+"\n")
    with (output/"retention.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "cell_id", "clean", "classification", "original_reason", "final_reason"])
        writer.writeheader()
        for row in report["rows"]:
            result = row["result"]
            writer.writerow(dict(case_id=row["case_id"], cell_id=row["cell_id"], clean=int(row["clean"]),
                classification=result["classification"], original_reason=result["original_preflight"]["reason"] or "",
                final_reason=result["reason"] or ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-evidence", type=Path, required=True)
    parser.add_argument("--benchmark-metadata", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse to replace a geometry analysis output")
    inputs = load_inputs(args.train_evidence, args.benchmark_metadata, args.assets, args.published)
    validate_output(args.output, inputs)
    report = analyze_cases(inputs, workers=args.workers,
                           progress=lambda value: print(json.dumps(value), flush=True))
    from continuous_geometry_audit import audit_analysis
    audit_started, audit_cpu = time.monotonic(), time.process_time()
    audit = audit_analysis(report, inputs)
    resources = dict(analysis_wall_s=report["wall_s"], analysis_worker_cpu_s=report["worker_cpu_s"],
                     analysis_coordinator_cpu_s=report["coordinator_cpu_s"],
                     independent_audit_wall_s=time.monotonic()-audit_started,
                     independent_audit_cpu_s=time.process_time()-audit_cpu)
    require(audit["status"] == "passed", "Independent geometry analysis audit did not pass")
    write_results(args.output, report, audit)
    (args.output/"resources.json").write_text(json.dumps(resources, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status=audit["status"], counts=report["counts"], clean_counts=report["clean_counts"],
                          wall_s=report["wall_s"], worker_cpu_s=report["worker_cpu_s"])), flush=True)


if __name__ == "__main__":
    main()
