"""Extract all successful paid TRAIN shots from the frozen seed-22 run.

Standard library only, so this file can run over SSH stdin against read-only
original evidence. It verifies bound receipts before exporting exact geometry
and recorded starts/fits. Evaluation packets, trajectories and media are never
read. No solver, policy inference, optimizer or simulator is imported.
"""

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

VERSION = "sircus-seed22-paid-train-evidence-v1"
SEED = 20261122
TRAIN_REQUESTS = 10000
TRAIN_CELLS = 35
TRAIN_PACKETS = 289
EVALUATION_REQUESTS = 420
ARMS = ("Arm10", "Arm14")
TRAIN_PATH = re.compile(r"trials/train-([0-9]{5})\.json\.gz")
ACTION_NAMES = ("bearing_offset_rad", "radius_m", "height_m", "gap_m", "tilt_rad")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid expected SHA256")
    return value


def _inside(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Expected relative evidence path")
    result = (root / relative).resolve()
    require(result.is_relative_to(root.resolve()), "Evidence path escaped frozen directory")
    return result


def read(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == _sha(expected), "Evidence SHA mismatch: " + Path(path).name)
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + value)))


def _vector(values, length, label):
    require(isinstance(values, list) and len(values) == length
            and all(type(x) in (int, float) and math.isfinite(x) for x in values),
            "Expected finite " + label)
    return copy.deepcopy(values)


def _joints(values):
    require(isinstance(values, dict) and set(values) == set(ARMS), "Expected both measured arm starts")
    return {name: _vector(values[name], 6, "six-joint measured starts") for name in ARMS}


def _cell(value):
    require(isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"]
            and value.get("split") == "train", "Expected TRAIN bucket context")
    return dict(id=value["id"], split="train", bucket_xy_m=_vector(value["bucket_xy_m"], 2, "bucket XY"))


def _check_success(action, row):
    require(row["invalid"] is None and row["qualification"]["accepted"] is True
            and row["path_proxy"]["accepted"] is True,
            "A successful TRAIN shot did not pass original admission/release")
    require(row["qualification"].get("action") == action, "Saved qualification changed shot geometry")
    require(set(row["qualification"]["ik"]) == set(ARMS), "Successful shot missing a coupled arm fit")
    for name in ARMS:
        fit = row["qualification"]["ik"][name]
        require(fit["solved"] is True and type(fit["residual"]) in (int, float)
                and math.isfinite(fit["residual"]) and 0 <= fit["residual"] < 1e-4,
                "Successful shot has unresolved original IK")
        _vector(fit["q"], 6, "saved endpoint joints")
    for key in ACTION_NAMES:
        require(type(action[key]) in (int, float) and math.isfinite(action[key]), "Invalid saved physical action")
        require(row.get(key) == action[key], "Successful row/action geometry mismatch")


def collect(benchmark, *, audit_sha, progress_sha, experiment_sha, split_sha, checkpoint_sha):
    """Return all original paid-TRAIN captures, without evaluating any new shot."""
    root = Path(benchmark)
    for value in (audit_sha, progress_sha, experiment_sha, split_sha, checkpoint_sha):
        _sha(value)
    run = root / "runs" / f"ppo-seed{SEED}"
    audit = read(Path(str(run) + ".audit.json"), audit_sha)
    require(audit["status"] == "passed" and audit["smoke_only"] is False
            and audit["method"] == "ppo" and audit["seed"] == SEED
            and audit["paid_train_attempts"] == TRAIN_REQUESTS
            and audit["evaluation_attempts"] == EVALUATION_REQUESTS,
            "Expected original passed production seed-22 audit")
    require(audit["source_progress_sha256"] == progress_sha
            and audit["experiment_sha256"] == experiment_sha
            and audit["split_manifest_sha256"] == split_sha, "Audit binding mismatch")
    experiment = read(root / "experiment.json", experiment_sha)
    splits = read(_inside(root, experiment["split_manifest_file"]), split_sha)
    progress = read(run / "progress.json", progress_sha)
    binding = read(run / "experiment-binding.json", audit["source_binding_sha256"])
    contract = read(run / "contract.json", audit["source_contract_sha256"])
    require(experiment["source"] == audit["source"] == progress["source"] == binding["source"],
            "Frozen source identity changed")
    require(experiment["split_manifest_sha256"] == split_sha == progress["split_manifest_sha256"]
            == binding["split_sha256"] and binding["experiment_sha256"] == experiment_sha,
            "Frozen experiment/split identity changed")
    require(progress["status"] == "completed" and progress["method"] == "ppo"
            and progress["seed"] == SEED and progress["train"]["attempts"] == TRAIN_REQUESTS
            and progress["smoke"] is False, "Original seed-22 run incomplete or wrong role")
    require(experiment["protocol"]["native_training_calls"] == TRAIN_PACKETS
            and experiment["protocol"]["site_counts"]["train"] == TRAIN_CELLS,
            "Original TRAIN schedule changed")
    finals = [row for row in progress["checkpoints"] if row["training_attempts"] == TRAIN_REQUESTS]
    require(len(finals) == 1 and finals[0]["checkpoint_sha256"] == checkpoint_sha,
            "Prescribed final model identity changed")
    checkpoint_relative = finals[0]["checkpoint_file"]
    require(audit["checkpoint_sha256"][checkpoint_relative] == checkpoint_sha
            and digest(_inside(run, checkpoint_relative)) == checkpoint_sha,
            "Frozen final checkpoint bytes changed")
    cells = [_cell(value) for value in splits["train"]]
    require(len(cells) == TRAIN_CELLS and len({row["id"] for row in cells}) == TRAIN_CELLS
            and len({tuple(row["bucket_xy_m"]) for row in cells}) == TRAIN_CELLS,
            "Expected all distinct TRAIN coordinates")
    by_id = {row["id"]: row for row in cells}
    # Do not glob or open evaluation paths. Only the exact paid-TRAIN receipt
    # namespace is consumed, even though the original audit also binds evals.
    manifest = {}
    for relative, expected in audit["source_trial_sha256"].items():
        if relative.startswith("trials/train-"):
            require(TRAIN_PATH.fullmatch(relative), "Unexpected paid TRAIN packet path")
            require(progress["source_trial_sha256"].get(relative) == expected,
                    "TRAIN audit/progress receipt mismatch")
            manifest[relative] = _sha(expected)
    require(len(manifest) == TRAIN_PACKETS, "Original paid TRAIN packet coverage changed")
    successes, starts, all_keys = [], {}, set()
    counters = dict(attempts=0, captures=0, clean=0, rejected=0)
    reasons, per_cell = Counter(), {row["id"]: Counter() for row in cells}
    packet_numbers = []
    for relative, expected in sorted(manifest.items()):
        packet = read(_inside(run, relative), expected)
        packet_numbers.append(packet["batch"])
        require(packet["schema"] == "sircus-shot-benchmark-native-trial-v1" and packet["mode"] == "train"
                and packet["source"] == experiment["source"]
                and packet["split_manifest_sha256"] == split_sha,
                "Paid TRAIN packet role/source mismatch")
        require(packet["batch"] == int(TRAIN_PATH.fullmatch(relative).group(1)), "TRAIN packet identity mismatch")
        require(all(len(packet[key]) == TRAIN_CELLS for key in
                    ("cells", "actions", "rows", "enabled", "actions_normalized")), "Native farm slot count changed")
        require(all(type(value) is bool for value in packet["enabled"]), "Enabled request mask is not Boolean")
        order = packet["training_request_order"]
        require(isinstance(order, list) and all(type(i) is int and 0 <= i < TRAIN_CELLS for i in order)
                and len(order) == len(set(order))
                and set(order) == {i for i, value in enumerate(packet["enabled"]) if value},
                "Paid TRAIN request order/mask changed")
        checkpoint = packet["checkpoint"]
        require(audit["checkpoint_sha256"].get(checkpoint["file"]) == checkpoint["sha256"]
                and checkpoint["training_attempts"] == counters["attempts"], "TRAIN checkpoint lineage changed")
        home = packet["home"]
        require(home["episode_reset"] is True
                and home["protocol"] == "canonical-episode-home-reset-before-visible-drive-v1",
                "Original canonical episode reset changed")
        for i, (raw_cell, action, row) in enumerate(zip(packet["cells"], packet["actions"], packet["rows"])):
            cell = _cell(raw_cell)
            require(cell["id"] in by_id and cell == by_id[cell["id"]]
                    and row["cell_id"] == cell["id"] and row["split"] == "train",
                    "Non-TRAIN or changed bucket request in paid packet")
            if not packet["enabled"][i]:
                continue
            key = relative + "#" + str(i)
            require(key not in all_keys, "Duplicate paid TRAIN request")
            all_keys.add(key)
            measured = _joints({name: home["measured_joint_rad"][name][i] for name in ARMS})
            if cell["id"] not in starts:
                starts[cell["id"]] = dict(recorded_starts_rad=copy.deepcopy(measured),
                                         trial_file=relative, slot=i, trial_sha256=expected)
            require(type(row["route_success"]) is bool and type(row["route_quality"]) is int
                    and row["route_quality"] in (0, 1, 2)
                    and (row["invalid"] is None or isinstance(row["invalid"], str) and row["invalid"]),
                    "Invalid original outcome labels")
            captured, clean, rejected = row["route_success"], row["route_quality"] == 2, row["invalid"] is not None
            require(not captured or not rejected, "Rejected paid request labeled successful")
            require(not clean or captured, "Clean paid request was not successful")
            values = dict(attempts=1, captures=int(captured), clean=int(clean), rejected=int(rejected))
            for name, value in values.items():
                counters[name] += value
                per_cell[cell["id"]][name] += value
            reason = row["invalid"] or ("captured" if captured else
                       "external_contact" if row["external_contact_steps"] else "released_miss")
            reasons[reason] += 1
            if captured:
                _check_success(action, row)
                require(home["path_eligible"][i] is True, "Successful paid request had invalid home")
                successes.append(dict(success_key=key, trial_file=relative, trial_sha256=expected,
                    slot=i, cell=copy.deepcopy(cell), action=copy.deepcopy(action),
                    actions_normalized=_vector(packet["actions_normalized"][i], 5, "normalized action"),
                    recorded_starts_rad=measured, qualification=copy.deepcopy(row["qualification"]),
                    path_proxy=copy.deepcopy(row["path_proxy"]), clean=clean))
    require(len(packet_numbers) == len(set(packet_numbers)), "Duplicate paid TRAIN packet identity")
    require(counters["attempts"] == len(all_keys) == TRAIN_REQUESTS and len(starts) == TRAIN_CELLS,
            "Incomplete original paid TRAIN request coverage")
    require(all(counters[key] == progress["train"][key] for key in counters)
            and dict(reasons) == progress["train"]["failures"], "Original TRAIN totals or rejection reasons changed")
    result = dict(schema=VERSION, seed=SEED, source=experiment["source"],
                  source_bindings=dict(audit_sha256=audit_sha, progress_sha256=progress_sha,
                    experiment_sha256=experiment_sha, split_manifest_sha256=split_sha,
                    checkpoint_sha256=checkpoint_sha, binding_sha256=audit["source_binding_sha256"],
                    contract_sha256=audit["source_contract_sha256"]),
                  scene=copy.deepcopy(contract["scene"]), cells=cells, canonical_starts=starts,
                  train_trial_sha256=manifest, training_totals=dict(counters, failures=dict(sorted(reasons.items()))),
                  per_cell={key: dict(value) for key, value in per_cell.items()},
                  successes=successes, successful_shots=len(successes), clean_successes=counters["clean"],
                  evaluated_trial_files_read=0, new_native_shots=0, optimizer_updates=0,
                  scope="All original paid TRAIN captures, including assisted captures under the unchanged original scorer; no held-out trial, trajectory, media, policy inference or new outcome consumed.")
    result["fingerprint"] = canonical_hash(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    for name in ("audit", "progress", "experiment", "split", "checkpoint"):
        parser.add_argument("--" + name + "-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.output != "-":
        output = Path(args.output).resolve()
        require(not output.is_relative_to(args.benchmark.resolve()), "Collector output must be separate from frozen evidence")
        require(not output.exists(), "Refuse to overwrite collected evidence")
    value = collect(args.benchmark, **{name + "_sha": getattr(args, name + "_sha")
                                    for name in ("audit", "progress", "experiment", "split", "checkpoint")})
    text = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)


if __name__ == "__main__":
    main()
