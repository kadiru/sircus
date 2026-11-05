"""Portable geometry-guided decoder export, separate from full-support identity.

Raw geometry probe histories remain outside Git. This compact artifact has its
own trusted fingerprint; it never masquerades as the original full support.
Export checks all saved policy contexts, including inactive training contexts.
This module starts no simulator, search, optimizer or GPU job.
"""

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

from continuous_feasibility import ACTION_NAMES, HIGH, LOW, canonical_hash, decode_action
from feasibility_guided_policy import POLICY_VERSION, load_checkpoint

SCHEMA = "sircus-guided-portable-decoder-v1"
RECEIPT_SCHEMA = "sircus-guided-portable-decoder-export-v1"
SCOPE = (
    "Portable decoder for 35 inherited geometry-assisted contexts only. Local "
    "boxes are not certified feasible interiors. Unresolved contexts remain "
    "explicit; IK warm starts are not verified endpoint commands. Grasp, paths "
    "and loading are not learned. Full raw support provenance is independently "
    "bound by its original SHA256/fingerprint, distinct from this compact hash."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid " + label + " SHA256")
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def signed(value, label):
    unsigned = dict(value)
    fingerprint = unsigned.pop("fingerprint", None)
    require(canonical_hash(unsigned) == fingerprint, label + " fingerprint changed")
    return fingerprint


def _decoder_core(decoder):
    return dict(action_names=decoder["action_names"], cells=decoder["cells"])


def _seed_core(decoder):
    return decoder["ik_seed_joint_rad"]


def validate_decoder(decoder, *, expected_fingerprint):
    """Expected compact hash must come from a trusted manifest, not this file.

    The compact file cannot recompute the original support fingerprint after
    omitted histories are removed. Faithful derivation is checked at export.
    """
    sha(expected_fingerprint, "trusted expected decoder")
    require(isinstance(decoder, dict) and decoder.get("schema") == SCHEMA,
            "Unsupported portable guided decoder")
    require(signed(decoder, "Portable decoder") == expected_fingerprint, "Trusted portable decoder fingerprint mismatch")
    for key in ("geometry_fingerprint", "study_fingerprint", "domain_fingerprint",
                "source_support_fingerprint", "source_support_sha256", "source_result_sha256",
                "decoder_core_fingerprint", "seed_core_fingerprint"):
        sha(decoder.get(key), key)
    require(decoder.get("policy_version") == POLICY_VERSION
            and decoder.get("action_names") == list(ACTION_NAMES), "Portable policy/action contract changed")
    require(decoder["decoder_core_fingerprint"] == canonical_hash(_decoder_core(decoder)), "Portable decoder core changed")
    require(decoder["seed_core_fingerprint"] == canonical_hash(_seed_core(decoder)), "Portable IK seed core changed")
    rows, seeds = decoder.get("cells"), decoder.get("ik_seed_joint_rad")
    require(isinstance(rows, list) and len(rows) == 35
            and all(isinstance(r.get("id"), str) for r in rows)
            and len({r["id"] for r in rows}) == 35, "Expected 35 distinct portable contexts")
    require(isinstance(seeds, dict) and set(seeds) == {r["id"] for r in rows}, "Portable seed/context IDs mismatch")
    coords = []
    for row in rows:
        require(set(row) == {"id", "bucket_xy_m", "status", "support_available", "low", "high"}, "Unexpected portable context fields")
        xy = np.asarray(row["bucket_xy_m"], dtype=float)
        lo, hi = np.asarray(row["low"], dtype=float), np.asarray(row["high"], dtype=float)
        require(xy.shape == (2,) and np.isfinite(xy).all(), "Invalid portable bucket coordinates")
        coords.append(tuple(xy))
        require(lo.shape == hi.shape == (5,) and np.isfinite(lo).all() and np.isfinite(hi).all()
                and (lo >= LOW).all() and (hi <= HIGH).all() and (hi > lo).all(), "Invalid portable support bounds")
        require(row["status"] in ("supported", "unresolved"), "Unknown portable support status")
        require(type(row.get("support_available")) is bool
                and row["support_available"] is (row["status"] == "supported"), "Portable eligibility disagrees with support status")
        seed = seeds[row["id"]]
        if row["status"] == "unresolved":
            require(seed is None and row["low"] == LOW.tolist() and row["high"] == HIGH.tolist(),
                    "Unresolved portable context was relabeled or seeded")
        else:
            require(isinstance(seed, dict) and set(seed) == {"Arm10", "Arm14"}, "Missing portable IK seeds")
            require(all(np.asarray(q).shape == (6,) and np.isfinite(q).all() for q in seed.values()),
                    "Invalid portable IK seed joints")
    require(len(set(coords)) == 35, "Portable bucket coordinates are not distinct")
    require(set(decoder.get("checkpoints", {})) == {"initial", "final"}, "Missing portable checkpoint binding")
    for stage, cp in decoder["checkpoints"].items():
        require(cp.get("file") == f"policy-{stage}.pt", "Unexpected portable checkpoint name")
        for key in ("sha256", "weight_hash", "actor_weight_hash"):
            sha(cp.get(key), "portable checkpoint " + key)
        require(type(cp.get("update_count")) is int and type(cp.get("attempted_shots")) is int
                and 0 <= cp["update_count"] <= cp["attempted_shots"], "Invalid portable checkpoint counters")
    return decoder


def load_decoder(path, *, expected_fingerprint):
    return validate_decoder(read(path), expected_fingerprint=expected_fingerprint)


def _selected(decoder, cells):
    by_id = {r["id"]: r for r in decoder["cells"]}
    require(len(cells) > 0 and len({c["id"] for c in cells}) == len(cells), "Expected unique nonempty requested contexts")
    result = []
    for cell in cells:
        require(cell.get("id") in by_id and cell.get("bucket_xy_m") == by_id[cell["id"]]["bucket_xy_m"],
                "Unknown bucket context: new geometry requires new support, not nearest lookup")
        result.append(by_id[cell["id"]])
    return result


def decode_actions(normalized, decoder, cells, *, expected_fingerprint, allow_unresolved=False):
    validate_decoder(decoder, expected_fingerprint=expected_fingerprint)
    if hasattr(normalized, "detach"):
        normalized = normalized.detach().cpu().numpy()
    values = np.asarray(normalized, dtype=float)
    require(values.shape == (len(cells), 5) and np.isfinite(values).all() and (np.abs(values) <= 1).all(),
            "Expected finite N by 5 normalized residual actions")
    require(type(allow_unresolved) is bool, "Expected Boolean audit/accounting override")
    selected = _selected(decoder, cells)
    require(allow_unresolved or all(r["support_available"] for r in selected),
            "Unresolved geometry support: ordinary inference is disabled; explicit accounting override only")
    low, high = (np.asarray([r[key] for r in selected], dtype=float) for key in ("low", "high"))
    return low+(values+1)*(high-low)/2


def ik_seeds(decoder, cells, *, expected_fingerprint, allow_unresolved=False):
    validate_decoder(decoder, expected_fingerprint=expected_fingerprint)
    require(type(allow_unresolved) is bool, "Expected Boolean audit/accounting override")
    rows = _selected(decoder, cells)
    require(allow_unresolved or all(r["support_available"] for r in rows), "Unresolved geometry support has no IK warm start")
    return [copy.deepcopy(decoder["ik_seed_joint_rad"][c["id"]]) for c in cells]


def load_policy(checkpoint, decoder, *, expected_decoder_fingerprint, stage="final", device="cpu"):
    """Use the unchanged guided loader with ORIGINAL full-support fingerprint."""
    require(stage in ("initial", "final"), "Unknown portable checkpoint stage")
    artifact = load_decoder(decoder, expected_fingerprint=expected_decoder_fingerprint)
    cp = artifact["checkpoints"][stage]
    require(Path(checkpoint).is_file() and digest(checkpoint) == cp["sha256"], "Portable checkpoint bytes changed")
    policy, saved = load_checkpoint(checkpoint,
        study_fingerprint=artifact["study_fingerprint"], geometry_fingerprint=artifact["geometry_fingerprint"],
        domain_fingerprint=artifact["domain_fingerprint"], support_fingerprint=artifact["source_support_fingerprint"],
        device=device, expected_updates=cp["update_count"], expected_attempted_shots=cp["attempted_shots"])
    require(saved["weight_hash"] == cp["weight_hash"] and saved["actor_weight_hash"] == cp["actor_weight_hash"],
            "Portable checkpoint model hashes changed")
    return policy, saved, artifact


def _file(run, name, *, trial=False, suffix=".json"):
    require(isinstance(name, str), "Invalid source evidence name")
    relative = Path(name)
    require(relative.parts == (("trials", relative.name) if trial else (relative.name,))
            and not relative.is_absolute() and relative.suffix == suffix, "Unsafe source evidence name")
    path = Path(run)/relative
    require(path.is_file() and not path.is_symlink()
            and not any(parent.is_symlink() for parent in path.parents if parent != Path(run).parent),
            "Missing or linked source evidence")
    return path


def build_decoder(run):
    """Bind raw protocol, support, weights, and every saved policy action."""
    from feasibility_guided_audit import validate_guided_study, validate_support_evidence
    from feasibility_guided_support import physical_actions

    run = Path(run)
    result, study, spec, support, feasibility = [read(_file(run, name))
        for name in ("result.json", "study.json", "contract.json", "support.json", "feasibility.json")]
    source = {name: digest(_file(run, name)) for name in ("result.json", "study.json", "contract.json", "support.json", "feasibility.json")}
    require(result.get("status") == "completed" and result.get("schema") == "sircus-native-feasibility-guided-pilot-v1"
            and result.get("study_fingerprint") == study.get("fingerprint")
            and result.get("geometry_fingerprint") == spec.get("fingerprint"), "Portable export requires completed matching guided result")
    validate_guided_study(study, spec, support, source["support.json"], source["feasibility.json"])
    support_audit = validate_support_evidence(support, feasibility, spec)
    cells = [{"id": r["cell_id"], "bucket_xy_m": copy.deepcopy(r["bucket_xy_m"]),
              "status": r["status"], "support_available": r["status"] == "supported", "low": copy.deepcopy(r["low"]), "high": copy.deepcopy(r["high"])}
             for r in support["cells"]]
    seeds = {r["cell_id"]: ({name: copy.deepcopy(r["anchor_ik"][name]["q"]) for name in ("Arm10", "Arm14")}
                           if r["status"] == "supported" else None) for r in support["cells"]}
    checkpoints = {}
    for stage in ("initial", "final"):
        item = result[stage]
        path = _file(run, item["file"], suffix=".pt")
        require(path.name == f"policy-{stage}.pt" and digest(path) == item["sha256"], "Raw checkpoint bytes changed")
        _, saved = load_checkpoint(path, study_fingerprint=study["fingerprint"], geometry_fingerprint=spec["fingerprint"],
            domain_fingerprint=study["domain"]["fingerprint"], support_fingerprint=support["fingerprint"],
            expected_updates=0 if stage == "initial" else study["training_batches"]//4,
            expected_attempted_shots=0 if stage == "initial" else study["training_batches"]*25)
        require(saved["weight_hash"] == item["policy_hash"] and saved["actor_weight_hash"] == item["actor_hash"],
                "Raw checkpoint model hashes changed")
        checkpoints[stage] = dict(file=path.name, sha256=digest(path), weight_hash=saved["weight_hash"],
                                  actor_weight_hash=saved["actor_weight_hash"], update_count=saved["update_count"],
                                  attempted_shots=saved["attempted_shots"])
        source[path.name] = digest(path)
    artifact = dict(schema=SCHEMA, policy_version=POLICY_VERSION,
                    geometry_fingerprint=spec["fingerprint"], study_fingerprint=study["fingerprint"],
                    domain_fingerprint=study["domain"]["fingerprint"],
                    source_support_fingerprint=support["fingerprint"], source_support_sha256=source["support.json"],
                    source_result_sha256=source["result.json"], action_names=list(ACTION_NAMES),
                    cells=cells, ik_seed_joint_rad=seeds, checkpoints=checkpoints, scope=SCOPE)
    artifact["decoder_core_fingerprint"] = canonical_hash(_decoder_core(artifact))
    artifact["seed_core_fingerprint"] = canonical_hash(_seed_core(artifact))
    artifact["fingerprint"] = canonical_hash(artifact)
    validate_decoder(artifact, expected_fingerprint=artifact["fingerprint"])
    require(len(result["batches"]) == study["training_batches"]
            and len(result["assessments"]) == 2*study["assessment_batches"]
            and len(result.get("coverage", [])) == study.get("coverage_batches", 0), "Incomplete raw trial schedule")
    references = result.get("coverage", [])+result["assessments"]+result["batches"]
    require(len({r["file"] for r in references}) == len(references), "Duplicate raw trial reference")
    all_contexts = active_contexts = inactive_contexts = policy_trials = coverage_trials = 0
    largest_physical = largest_tanh = 0.
    trial_hashes = {}
    for reference in references:
        path = _file(run, reference["file"], trial=True)
        require(digest(path) == reference["sha256"], "Raw trial bytes changed")
        trial = read(path)
        require(trial.get("schema") == study["schema"] and trial.get("study_fingerprint") == study["fingerprint"]
                and trial.get("support_fingerprint") == support["fingerprint"], "Trial protocol/support binding changed")
        trial_hashes[reference["file"]] = digest(path)
        if trial.get("mode") == "coverage":
            require(trial.get("policy_action_used") is False, "Coverage falsely marked policy-generated")
            coverage_trials += 1
            continue
        require(trial.get("mode") in ("train", "eval") and trial.get("policy_action_used") is True,
                "Unknown or disabled policy decode trial")
        latent, normalized = np.asarray(trial["latent"], dtype=np.float32), np.asarray(trial["actions_normalized"], dtype=float)
        require(latent.shape == normalized.shape == (35, 5) and np.isfinite(latent).all(), "Invalid saved policy latent")
        tanh_error = float(np.max(np.abs(np.tanh(latent)-normalized)))
        require(tanh_error <= 3e-7, "Saved latent-to-normalized action mismatch")
        largest_tanh = max(largest_tanh, tanh_error)
        portable = decode_actions(normalized, artifact, study["cells"], expected_fingerprint=artifact["fingerprint"], allow_unresolved=True)
        original = physical_actions(normalized, support, study["cells"])
        require(np.array_equal(portable, original), "Portable/raw affine decoder differs")
        stored = np.asarray([[a[k] for k in ACTION_NAMES] for a in trial["actions"]], dtype=float)
        require(stored.shape == (35, 5) and np.isfinite(stored).all(), "Invalid stored policy action")
        physical_error = float(np.max(np.abs(portable-stored)))
        require(physical_error <= 1e-12, "Stored trial action differs from portable decoder")
        largest_physical = max(largest_physical, physical_error)
        for cell, values, action in zip(study["cells"], portable, trial["actions"]):
            expected = decode_action(cell, values, action["id"])
            require(action == expected, "Stored derived shot geometry differs from portable decode")
        enabled = trial.get("enabled")
        require(isinstance(enabled, list) and len(enabled) == 35 and all(type(v) is bool for v in enabled), "Invalid active context mask")
        expected_enabled = [c["split"] == "train" for c in study["cells"]] if trial["mode"] == "train" else [True]*35
        require(enabled == expected_enabled
                and trial.get("active_cell_ids") == [c["id"] for c, use in zip(study["cells"], enabled) if use],
                "Policy trial active/held-out context membership changed")
        used = sum(enabled)
        all_contexts += 35
        active_contexts += used
        inactive_contexts += 35-used
        policy_trials += 1
    receipt = dict(schema=RECEIPT_SCHEMA, status="passed", exporter_code_sha256=digest(Path(__file__)), decoder_fingerprint=artifact["fingerprint"],
                   decoder_sha256=hashlib.sha256((json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()).hexdigest(),
                   decoder_core_fingerprint=artifact["decoder_core_fingerprint"], seed_core_fingerprint=artifact["seed_core_fingerprint"],
                   source_support_fingerprint=support["fingerprint"], source_support_sha256=source["support.json"],
                   source_files_sha256=source, trial_files_sha256=trial_hashes,
                   policy_trial_batches=policy_trials, excluded_coverage_batches=coverage_trials,
                   verified_decoded_contexts=all_contexts, verified_active_contexts=active_contexts,
                   verified_inactive_contexts=inactive_contexts,
                   max_stored_physical_action_error=largest_physical, max_latent_tanh_error=largest_tanh,
                   raw_and_compact_affine_outputs_bit_identical=True,
                   original_support_identity_recomputed_before_compaction=True,
                   geometry_support_evidence=support_audit,
                   scope="CPU-only compact extraction/action-equivalence audit, not a replacement for native outcome, policy likelihood or optimizer membership audits")
    receipt["fingerprint"] = canonical_hash(receipt)
    return artifact, receipt


def export(run, output):
    output = Path(output)
    require(not output.exists(), "Choose a new portable export directory")
    artifact, receipt = build_decoder(run)
    # All validation is complete before creating a new exclusive output.
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("decoder.json", artifact), ("decoder-export-audit.json", receipt)):
        with (output/name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    for item in artifact["checkpoints"].values():
        dest = output/item["file"]
        require(not dest.exists(), "Portable checkpoint output already exists")
        with (Path(run)/item["file"]).open("rb") as source, dest.open("xb") as target:
            shutil.copyfileobj(source, target)
        require(digest(dest) == item["sha256"], "Portable checkpoint copy checksum mismatch")
    return artifact, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    artifact, receipt = export(args.raw, args.output)
    print(json.dumps(dict(decoder_fingerprint=artifact["fingerprint"],
                          source_support_fingerprint=artifact["source_support_fingerprint"],
                          verified_contexts=receipt["verified_decoded_contexts"],
                          active_contexts=receipt["verified_active_contexts"],
                          inactive_contexts=receipt["verified_inactive_contexts"])))


if __name__ == "__main__":
    main()
