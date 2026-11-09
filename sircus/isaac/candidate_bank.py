"""Build and independently audit a TRAIN-only candidate bank without training.

Uses all successful paid TRAIN shots and exact final-parent TRAIN predictions.
The collector and this driver never read validation or TEST shot files.
"""

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from candidate_train_evidence import canonical_hash, digest, read, require
from continuous_feasibility import ACTION_NAMES, HIGH, LOW
from feasible_candidates import generate_candidates, qualify_candidates, unsigned
from prepare_assets import chain

VERSION = "sircus-seed22-train-candidate-bank-v1"
ROOT = Path(__file__).resolve().parents[2]


def signed(value):
    value["fingerprint"] = canonical_hash(value)
    return value


def plain_read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + x)))


def safe_path(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Expected relative frozen evidence path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Frozen evidence path escaped run")
    return path


def load_inputs(evidence_path, scale_path, benchmark, published_path, assets):
    """Verify lineage and read one final known-TRAIN packet; no held-out shots."""
    evidence = plain_read(evidence_path)
    unsigned(evidence)
    require(evidence["seed"] == 20261122 and evidence["training_totals"]["attempts"] == 10000
            and evidence["evaluated_trial_files_read"] == evidence["optimizer_updates"] == evidence["new_native_shots"] == 0,
            "Expected untouched paid-TRAIN evidence")
    published = plain_read(published_path)
    runs = [r for r in published["runs"] if r["method"] == "ppo" and r["seed"] == 20261122]
    require(len(runs) == 1, "Expected one published seed-22 run")
    original = runs[0]
    binding = evidence["source_bindings"]
    require(binding["audit_sha256"] == original["independent_audit"]["sha256"]
            and binding["checkpoint_sha256"] == original["checkpoint"]["sha256"]
            and binding["progress_sha256"] == original["independent_audit"]["source_progress_sha256"]
            and binding["experiment_sha256"] == published["experiment_sha256"]
            and binding["split_manifest_sha256"] == published["split_manifest_sha256"]
            and evidence["source"] == published["source"], "Collected evidence/published lineage mismatch")
    benchmark, assets = Path(benchmark), Path(assets)
    run = benchmark / "runs/ppo-seed20261122"
    audit = read(Path(str(run) + ".audit.json"), binding["audit_sha256"])
    require(audit["status"] == "passed" and audit["smoke_only"] is False, "Original independent audit did not pass")
    progress = read(run / "progress.json", binding["progress_sha256"])
    final = [c for c in progress["checkpoints"] if c["training_attempts"] == 10000]
    require(len(final) == 1, "Ambiguous prescribed parent")
    final = final[0]
    require(digest(safe_path(run, final["checkpoint_file"])) == binding["checkpoint_sha256"], "Frozen parent model bytes changed")
    proof = read(safe_path(run, final["proof_file"]), original["checkpoint"]["proof_sha256"])
    require(proof["checkpoint"]["sha256"] == binding["checkpoint_sha256"]
            and proof["state_hash_before"] == proof["state_hash_after"] == proof["checkpoint"]["state_hash"]
            and proof["no_optimizer_or_flight_search"] is True, "Parent frozen-state proof changed")
    train_files = [(p, sha) for p, sha in proof["source_trial_sha256"].items() if p.startswith("trials/eval-train-")]
    require(len(train_files) == 1, "Expected one final known-TRAIN packet")
    relative, expected = train_files[0]
    require(audit["source_trial_sha256"][relative] == progress["source_trial_sha256"][relative] == expected,
            "Parent TRAIN packet audit receipt mismatch")
    packet = read(safe_path(run, relative), expected)
    require(packet["mode"] == "eval-train" and packet["checkpoint"] == proof["checkpoint"]
            and packet["source"] == evidence["source"]
            and packet["split_manifest_sha256"] == binding["split_manifest_sha256"]
            and packet["learner_packet"]["deterministic"] is True,
            "Parent TRAIN prediction binding changed")
    cells = {c["id"]: c for c in evidence["cells"]}
    require(len(cells) == len(packet["cells"]) == len(packet["actions"]) == 35
            and packet["enabled"] == [True]*35, "Parent TRAIN position coverage changed")
    parent, means = {}, {}
    latent = np.asarray(packet["learner_packet"]["latent"], dtype=float)
    normalized = np.asarray(packet["actions_normalized"], dtype=float)
    require(latent.shape == normalized.shape == (35, 5) and np.isfinite(latent).all()
            and np.isfinite(normalized).all() and (abs(normalized) <= 1).all()
            and np.allclose(np.tanh(latent), normalized, atol=2e-7, rtol=0), "Parent mean/action mismatch")
    for i, (cell, action) in enumerate(zip(packet["cells"], packet["actions"])):
        require(cell["split"] == "train" and cell["id"] in cells and cell["id"] not in parent
                and cell["bucket_xy_m"] == cells[cell["id"]]["bucket_xy_m"], "Parent context is not original TRAIN geometry")
        require(np.array_equal(np.asarray([action[k] for k in ACTION_NAMES]), LOW+(normalized[i]+1)*(HIGH-LOW)/2),
                "Parent physical decode differs from original action")
        parent[cell["id"]] = copy.deepcopy(action)
        means[cell["id"]] = latent[i].tolist()
    scale = plain_read(scale_path)
    unsigned(scale)
    require(scale["schema"] == "sircus-frozen-parent-scale-export-v1" and scale["seed"] == 20261122
            and scale["checkpoint_sha256"] == binding["checkpoint_sha256"]
            and scale["weight_hash"] == proof["checkpoint"]["state_hash"]
            and scale["producer_source_sha256"] == audit["source_sha256"]["continuous_policy.py"]
            and scale["cpu_only"] is True and scale["actor_inference"] is False
            and scale["optimizer_updates"] == scale["native_shots"] == 0,
            "Frozen Gaussian scale export changed")
    logstd, scales = np.asarray(scale["logstd"], float), np.asarray(scale["latent_scale"], float)
    require(logstd.shape == scales.shape == (5,) and np.isfinite(logstd).all()
            and np.isfinite(scales).all() and (scales > 0).all()
            and np.allclose(scales, np.exp(np.clip(logstd, -3, 1)), atol=1e-7, rtol=0), "Invalid parent Gaussian scales")
    geometry_path = ROOT / "baselines/isaac-reach-ik-audit-20261106/summary.json"
    geometry = plain_read(geometry_path)
    require(geometry["snapshot"]["source"] == evidence["source"]
            and geometry["dependency_sha256"]["contract.json"] == binding["contract_sha256"], "Geometry receipt lineage mismatch")
    asset_files = {n: assets/f for n, f in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))}
    for path in asset_files.values():
        require(digest(path) == geometry["dependency_sha256"]["assets/" + path.name], "Frozen URDF bytes changed")
    links = {n: chain(ET.fromstring(p.read_bytes())) for n, p in asset_files.items()}
    successes = [dict(case_id=r["success_key"], cell=r["cell"], action=r["action"], ik=r["qualification"]["ik"], clean=r["clean"])
                 for r in evidence["successes"]]
    require(len(successes) == evidence["successful_shots"] == evidence["training_totals"]["captures"]
            and sum(r["clean"] for r in evidence["successes"]) == evidence["clean_successes"], "Successful TRAIN extraction accounting changed")
    starts = {name: row["recorded_starts_rad"] for name, row in evidence["canonical_starts"].items()}
    bindings = dict(train_evidence_sha256=digest(evidence_path), parent_scale_sha256=digest(scale_path),
                    parent_training_trial_sha256=expected, original=copy.deepcopy(binding),
                    original_geometry_receipt_sha256=digest(geometry_path),
                    assets_sha256={n: digest(p) for n,p in asset_files.items()},
                    parent_mean_source="Exact final frozen known-TRAIN GPU predictions; no actor inference or held-out shot files",
                    common_start_source="First paid TRAIN measured home at each original TRAIN position",
                    parent_scale_source="CPU-only read of unchanged final checkpoint")
    return evidence, parent, means, {name: scales.tolist() for name in cells}, successes, starts, links, bindings


def export_retention(path, evidence, qualification):
    history = {r["case_id"]: r for r in qualification["history_survival"]}
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["case_id", "cell_id", "clean", "candidate_id", "included_exactly", "qualified", "exclusion_reason"])
        writer.writeheader()
        for old in evidence["successes"]:
            row = history[old["success_key"]]
            writer.writerow(dict(case_id=row["case_id"], cell_id=row["cell_id"], clean=int(old["clean"]),
                                 candidate_id=row["candidate_id"], included_exactly=int(row["generator_retained"]),
                                 qualified=int(row["accepted"]), exclusion_reason=row["reason"] or ""))


def retention_summary(evidence, qualification, audit):
    originals = {r["success_key"]: r for r in evidence["successes"]}
    history = qualification["history_survival"]
    require(len(originals) == len(evidence["successes"]) == len(history)
            and {r["case_id"] for r in history} == set(originals), "Historical retention case coverage changed")
    require(all(type(r["accepted"]) is bool and r["generator_retained"] is True
                and r["cell_id"] == originals[r["case_id"]]["cell"]["id"]
                and (r["reason"] is None if r["accepted"] else isinstance(r["reason"], str) and bool(r["reason"]))
                for r in history), "Invalid historical retention status")
    clean = [r for r in history if originals[r["case_id"]]["clean"]]
    survived = sum(r["accepted"] for r in history)
    clean_survived = sum(r["accepted"] for r in clean)
    require(audit["successful_train_shots"] == evidence["successful_shots"] == len(history)
            and audit["clean_train_shots"] == evidence["clean_successes"] == len(clean)
            and audit["successful_train_survive"] == survived
            and audit["clean_train_survive"] == clean_survived,
            "Independent audit lost original TRAIN accounting")
    return dict(historical_successes=dict(requested=len(history), included_exactly=len(history),
                    qualified=survived, excluded=len(history)-survived,
                    reasons=dict(sorted(Counter(r["reason"] for r in history if not r["accepted"]).items()))),
                clean_historical_successes=dict(requested=len(clean), qualified=clean_survived,
                    excluded=len(clean)-clean_survived),
                historical_exclusions=[dict(r, clean=originals[r["case_id"]]["clean"]) for r in history if not r["accepted"]])


def build(evidence_path, scale_path, benchmark, published_path, assets, output, *, workers=4):
    from feasible_candidates_audit import audit_candidates
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to replace a candidate bank")
    require(not output.is_relative_to(Path(benchmark).resolve()), "Candidate bank must be outside frozen experiment")
    inputs = load_inputs(evidence_path, scale_path, benchmark, published_path, assets)
    evidence, parent, means, scales, successes, starts, links, bindings = inputs
    output.mkdir(parents=True)

    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

    plan = generate_candidates(evidence["cells"], parent, means, scales, successes)
    save("generation-input-bindings.json", bindings)
    save("plan.json", plan)

    def progress(value):
        if value["completed"] % 100 == 0 or value["completed"] == value["total"]:
            save("progress.json", value)
            print(json.dumps(value), flush=True)

    began = time.monotonic()
    qualification = qualify_candidates(plan, evidence["scene"], links, starts, workers=workers, progress=progress)
    save("qualification.json", qualification)
    audit = audit_candidates(plan, qualification, successes, evidence["scene"], links, starts,
                             parent_actions=parent, parent_latent_means=means, parent_latent_scales=scales)
    require(audit["status"] == "passed", "Candidate audit did not pass")
    retention = retention_summary(evidence, qualification, audit)
    save("audit.json", audit)
    export_retention(output/"successful-train-retention.csv", evidence, qualification)
    summary = dict(schema=VERSION, status="passed", seed=20261122, scope=plan["scope"],
                   input_bindings=bindings, generation_protocol=plan["protocol"],
                   raw_candidates=plan["requests"], unique_candidates=plan["unique_candidates"],
                   admitted=qualification["admitted"], declined=qualification["requests"]-qualification["admitted"],
                   source_survival=qualification["source_survival"], per_cell=qualification["per_cell"],
                   **retention,
                   supported_train_positions=sum(r["status"] == "supported" for r in qualification["per_cell"]),
                   train_positions=len(evidence["cells"]), paid_train_packets_verified=len(evidence["train_trial_sha256"]),
                   original_paid_train_requests=evidence["training_totals"]["attempts"],
                   original_training_counters=evidence["training_totals"],
                   artifacts_sha256={n: digest(output/n) for n in ("plan.json", "qualification.json", "audit.json", "generation-input-bindings.json", "successful-train-retention.csv")},
                   producer_source_sha256={p.name:digest(p) for p in (Path(__file__), Path(__file__).with_name("candidate_train_evidence.py"), Path(__file__).with_name("feasible_candidates.py"), Path(__file__).with_name("feasible_candidates_audit.py"))},
                   geometry_wall_s=qualification["wall_s"], audit_and_geometry_wall_s=time.monotonic()-began,
                   optimizer_updates=0, new_native_shots=0, held_out_shot_files_read=0,
                   release_authorized=False,
                   limits=["TRAIN success demonstrations inform proposal inclusion; geometry gates alone decide qualification.",
                           "Only finite exact candidates are qualified; no box interior or exhaustive reachability claim.",
                           "Historical IK witnesses warm-start fresh unchanged robust IK; all candidates use the same endpoint/path gates.",
                           "Common starts differ from original episode starts; capture/settlement are not replayed or guaranteed.",
                           "No native flight, policy choice, optimizer update or frozen benchmark change."])
    save("summary.json", signed(summary))
    print(json.dumps(dict(status="passed", historical_successes=summary["historical_successes"], admitted=summary["admitted"])), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-evidence", required=True, type=Path)
    parser.add_argument("--parent-scale", required=True, type=Path)
    parser.add_argument("--benchmark-metadata", required=True, type=Path)
    parser.add_argument("--published", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(args.train_evidence, args.parent_scale, args.benchmark_metadata, args.published,
          args.assets, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
