"""Read-only CPU audit of the bounded mjlab learning study and frozen actors.

This verifies saved evidence, not independent physical repeatability. No native
GPU simulator, robot API, cloud client or service launcher is imported here.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = "sircus-mjlab-learning-audit-v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reward(row):
    """Independent transcription of the study's terminal learning reward."""
    if row["invalid"]:
        return -1.0
    shaped = 0.15 * float(row["board_bounce"]) + 0.5 * np.exp(
        -row["closest_opening_distance_m"] / 0.12
    )
    if row["external_contact_steps"]:
        return float(shaped - 0.5)
    return float(
        shaped
        + 0.5 * min(row["max_retention_s"], 1.0)
        + 3 * row["route_success"]
        + 0.3 * (row["route_quality"] == 2)
        + 0.1 * row["single_bounce_success"]
    )


def statistics(rows):
    return dict(
        attempts=len(rows),
        captures=sum(bool(r["route_success"]) and not r["invalid"] for r in rows),
        clean=sum(r["route_quality"] == 2 and not r["invalid"] for r in rows),
        invalid=sum(bool(r["invalid"]) for r in rows),
        board_bounces=sum(bool(r["board_bounce"]) and not r["invalid"] for r in rows),
        mean_reward=float(np.mean([reward(r) for r in rows])) if rows else 0.0,
    )


class Audit:
    def __init__(self):
        self.checks = []
        self.warnings = []

    def check(self, name, condition, detail=None):
        self.checks.append(dict(name=name, passed=bool(condition), detail=detail))
        return bool(condition)

    def stats(self, name, expected, actual):
        keys = (
            "attempts",
            "captures",
            "clean",
            "invalid",
            "board_bounces",
            "mean_reward",
        )
        return self.check(
            name,
            all(
                k in actual and np.isclose(expected[k], actual[k], atol=1e-8, rtol=1e-8)
                for k in keys
            ),
            dict(recomputed=expected, reported={k: actual.get(k) for k in keys}),
        )


def _xy(row):
    return tuple(row["bucket_xy_m"])


def audit_records(spec, report, rows, catalogue, audit=None):
    """Audit counts, spatial splits, scoring labels and train-only teachers."""
    a = audit or Audit()
    payload = dict(spec)
    fingerprint = payload.pop("fingerprint")
    a.check(
        "contract_fingerprint",
        digest(payload) == fingerprint == report["contract_fingerprint"],
    )
    pools = {
        key: [tuple(xy) for xy in spec[key + "_layouts"]]
        for key in ("train", "validation", "test")
    }
    merged = sum(pools.values(), [])
    a.check(
        "spatial_splits_disjoint",
        len(set(merged)) == len(merged),
        {k: len(v) for k, v in pools.items()},
    )
    a.check(
        "study_completed",
        report.get("status") == "completed" and report.get("phase") == "completed",
    )
    a.check("records_count", len(rows) == report["records"])
    a.check(
        "coverage_report_matches_catalogue",
        report.get("coverage", {}).get("cells") == catalogue,
    )
    a.check(
        "unique_sequential_trial_ids",
        [r["id"] for r in rows] == [f"shot-{i:07d}" for i in range(len(rows))],
    )
    grouped = {}
    for row in rows:
        grouped.setdefault(row["phase"], []).append(row)
    known = {
        "train-exploration",
        "train-confirmation",
        "train-ppo",
        "validation-warm",
        "test-initial",
        "test-warm",
        "test-selected",
        "test-lock-board",
        "test-lock-dropper",
    }
    known.update(
        f"validation-{x['update']}"
        for x in report["learning_curve"]
        if "validation" in x
    )
    a.check("known_phase_names", set(grouped).issubset(known), sorted(grouped))
    bad_membership = []
    for row in rows:
        phase = row["phase"]
        split = (
            "train"
            if phase.startswith("train-")
            else "validation" if phase.startswith("validation-") else "test"
        )
        if _xy(row) not in pools[split]:
            bad_membership.append(row["id"])
    a.check("phase_layout_membership", not bad_membership, bad_membership[:10])
    training_indices = [
        i
        for i, r in enumerate(rows)
        if r["phase"].startswith(("train-", "validation-"))
    ]
    test_indices = [i for i, r in enumerate(rows) if r["phase"].startswith("test-")]
    a.check(
        "jsonl_training_precedes_test",
        bool(test_indices)
        and (not training_indices or max(training_indices) < min(test_indices)),
    )
    a.check(
        "no_search_in_frozen_evaluation",
        report["frozen_evaluation"].get("search_at_inference") is False,
    )
    exploration = grouped.get("train-exploration", [])
    ppo = grouped.get("train-ppo", [])
    confirmation = grouped.get("train-confirmation", [])
    a.check(
        "exploration_budget",
        len(exploration) == spec["exploration_batches"] * spec["batch_size"],
    )
    a.check(
        "exploration_cyclic_layout_assignment",
        all(
            _xy(r) == pools["train"][i % len(pools["train"])]
            for i, r in enumerate(exploration)
        ),
    )
    a.check("ppo_budget", len(ppo) == spec["ppo_updates"] * spec["batch_size"])
    a.check(
        "training_count_excludes_validation_and_test",
        report["train_episodes"] == len(exploration) + len(confirmation) + len(ppo),
    )
    a.stats("exploration_statistics", statistics(exploration), report["exploration"])
    a.check(
        "optimizer_update_count",
        report["updates"] == spec["ppo_updates"] == len(report["learning_curve"]),
    )
    batch = spec["batch_size"]
    for i, curve in enumerate(report["learning_curve"]):
        a.check(f"update_{i+1}_number", curve["update"] == i + 1)
        a.stats(
            f"update_{i+1}_statistics",
            statistics(ppo[i * batch : (i + 1) * batch]),
            curve,
        )
        if "validation" in curve:
            val = grouped.get(f"validation-{i+1}", [])
            a.check(
                f"validation_{i+1}_layout_sequence",
                [_xy(r) for r in val] == pools["validation"],
            )
            a.stats(
                f"validation_{i+1}_statistics", statistics(val), curve["validation"]
            )
    warm_val = grouped.get("validation-warm", [])
    a.check(
        "warm_validation_layout_sequence",
        [_xy(r) for r in warm_val] == pools["validation"],
    )
    a.stats(
        "warm_validation_statistics", statistics(warm_val), report["warm_validation"]
    )
    best_value, selected_update = statistics(warm_val)["mean_reward"], 0
    for curve in report["learning_curve"]:
        if "validation" in curve and curve["validation"]["mean_reward"] > best_value:
            best_value, selected_update = (
                curve["validation"]["mean_reward"],
                curve["update"],
            )
    a.check(
        "validation_only_checkpoint_selection",
        report["selected_update"] == selected_update,
        dict(
            recomputed_update=selected_update, reported_update=report["selected_update"]
        ),
    )

    comparisons = report["frozen_evaluation"]["comparison"]
    for label in ("initial", "warm", "selected"):
        evaluation = grouped.get("test-" + label, [])
        metadata = comparisons[label]
        a.check(
            f"test_{label}_layout_and_repeat_sequence",
            [_xy(r) for r in evaluation] == pools["test"] * metadata["repeats"]
            and metadata["layouts"] == len(pools["test"]),
        )
        a.stats(f"test_{label}_statistics", statistics(evaluation), metadata)
    a.stats(
        "selected_top_level_statistics",
        statistics(grouped.get("test-selected", [])),
        report["frozen_evaluation"],
    )
    a.check(
        "selected_trial_count",
        report["frozen_evaluation"]["trials"] == len(grouped.get("test-selected", [])),
    )
    a.check(
        "selected_report_rows_match_jsonl",
        report.get("test_rows") == grouped.get("test-selected"),
    )
    for mode in ("board", "dropper"):
        locked = grouped.get("test-lock-" + mode, [])
        a.check(
            f"{mode}_lock_layout_sequence", [_xy(r) for r in locked] == pools["test"]
        )
        a.stats(
            f"{mode}_lock_statistics", statistics(locked), comparisons[mode + "_locked"]
        )
        for i, row in enumerate(locked):
            reference = grouped["test-selected"][i]
            expected = (
                reference["values"][:3]
                if mode == "board"
                else reference["release_ball_center_m"]
            )
            a.check(
                f"{row['id']}_constraint_source",
                row["locks"][mode] == expected
                and row["locks"]["dropper" if mode == "board" else "board"] is None,
            )
    score_violations = []
    for row in rows:
        route = bool(
            row["success"]
            and row["board_contact_episodes"] >= 1
            and not row["external_contact_steps"]
            and not row["invalid"]
        )
        okay = route == bool(row["route_success"]) == bool(row["complete_success"])
        okay &= bool(row["single_bounce_success"]) == (
            route and row["board_contact_episodes"] == 1
        )
        if row["success"]:
            okay &= bool(
                row["board_bounce"]
                and row["whole_ball_entry"]
                and row["max_retention_s"] >= 1 - 1e-8
            )
        if row["route_quality"] == 2:
            okay &= bool(
                route
                and not row["bucket_contact_before_entry"]
                and not row["rim_band_contact"]
                and not row["exited_after_entry"]
                and row["minimum_mouth_clearance_mm"] is not None
                and row["minimum_mouth_clearance_mm"] >= 1 - 1e-6
            )
        if not okay:
            score_violations.append(row["id"])
    a.check(
        "saved_scoring_label_invariants", not score_violations, score_violations[:10]
    )
    scalar_audits = [r for r in rows if "scalar_score_audit_passed" in r]
    a.check(
        "scalar_audit_evidence_present_and_passed",
        bool(scalar_audits)
        and all(r["scalar_score_audit_passed"] is True for r in scalar_audits),
        len(scalar_audits),
    )

    # JSONL was written before teacher_source_id was attached in memory. Restore
    # association independently from the protocol's stable layout/top3/rep order.
    references = []
    by_layout = {}
    for xy in pools["train"]:
        group = sorted(
            [r for r in exploration if _xy(r) == xy and not r["invalid"]],
            key=reward,
            reverse=True,
        )
        by_layout[xy] = group
        references.extend(
            (xy, candidate["id"]) for candidate in group[:3] for _ in range(3)
        )
    a.check("confirmation_count", len(confirmation) == len(references))
    repeats = {}
    reference_bad = []
    for row, (xy, source) in zip(confirmation, references):
        if _xy(row) != xy or (row.get("teacher_source_id") not in (None, source)):
            reference_bad.append(row["id"])
        repeats.setdefault(source, []).append(row)
    a.check(
        "confirmation_associations_reconstructed", not reference_bad, reference_bad[:10]
    )
    a.check(
        "catalogue_layout_sequence",
        [tuple(c["bucket_xy_m"]) for c in catalogue] == pools["train"],
    )
    teachers = []
    for cell in catalogue:
        xy = tuple(cell["bucket_xy_m"])
        group = by_layout.get(xy, [])
        raw = [r for r in exploration if _xy(r) == xy]
        a.check(
            cell["id"] + "_coverage_counts",
            cell["trials"] == len(raw)
            and cell["valid"] == len(group)
            and cell["captures"] == sum(r["route_success"] for r in group),
        )
        if not group:
            a.check(cell["id"] + "_no_teacher", cell.get("best") is None)
            continue
        selected = max(
            group,
            key=lambda r: (
                sum(
                    x["route_success"] and not x["invalid"]
                    for x in repeats.get(r["id"], [])
                ),
                reward(r),
            ),
        )
        teachers.append(selected)
        a.check(
            cell["id"] + "_teacher_selection",
            cell["best"]["id"] == selected["id"]
            and cell["best"]["values"] == selected["values"],
        )
        a.stats(
            cell["id"] + "_teacher_confirmation",
            statistics(repeats.get(selected["id"], [])),
            cell.get("confirmation", {}),
        )
    a.check(
        "imitation_train_only_sample_count",
        report["imitation"]["samples"] == 3 * len(teachers),
    )
    teacher_map = {tuple(c["bucket_xy_m"]): c.get("best") for c in catalogue}
    bad_training_locks = []
    for row in rows:
        locks = row["locks"]
        if row["phase"] != "train-ppo" and not row["phase"].startswith("test-lock-"):
            if any(value is not None for value in locks.values()):
                bad_training_locks.append(row["id"])
        if row["phase"] == "train-ppo":
            teacher = teacher_map.get(_xy(row))
            if sum(value is not None for value in locks.values()) > 1:
                bad_training_locks.append(row["id"])
            for mode, value in locks.items():
                if value is not None:
                    expected = (
                        None
                        if teacher is None
                        else (
                            teacher["values"][:3]
                            if mode == "board"
                            else teacher["release_ball_center_m"]
                        )
                    )
                    if value != expected:
                        bad_training_locks.append(row["id"])
    a.check(
        "ppo_constraints_anchored_to_train_only_teachers",
        not bad_training_locks,
        bad_training_locks[:10],
    )
    a.warnings.append(
        "Missing teacher_source_id fields were reconstructed from stable train-layout/top-three/three-repeat order; associations were not persisted before JSONL writing."
    )
    return a, grouped


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _observations(bucket, locks):
    xy = np.asarray(bucket)
    board, dropper = (
        np.array(
            [r[k] if r[k] is not None else [np.nan] * 3 for r in locks], dtype=float
        )
        for k in ("board", "dropper")
    )
    bm, dm = np.isfinite(board).all(1), np.isfinite(dropper).all(1)
    return np.column_stack(
        (
            (xy - [0.67, -0.20]) / [0.6, 0.5],
            bm,
            dm,
            np.nan_to_num(board) / [1, 0.7, 0.6],
            np.nan_to_num(dropper) / [1, 0.7, 1.2],
        )
    ).astype(np.float32)


def _project(values, locks, radius):
    values = values.copy()
    for i, lock in enumerate(locks):
        board, dropper = lock["board"], lock["dropper"]
        if board is not None:
            values[i, :3] = board
        if dropper is not None:
            if board is None:
                values[i, :2] = dropper[:2]
            values[i, 4] = dropper[2] - values[i, 2] - radius
    return values


def audit_checkpoints(study, spec, report, grouped, audit):
    import torch
    from .learning_policy import Policy

    torch.set_num_threads(2)
    states, hashes = {}, {}
    for name in ("initial", "warm", "selected", "latest"):
        checkpoint = torch.load(
            study / (name + ".pt"), map_location="cpu", weights_only=True
        )
        audit.check(name + "_checkpoint_contract", checkpoint["contract"] == spec)
        states[name] = checkpoint
        hashes[name] = state_hash(checkpoint["state"])
    audit.check("imitation_changes_weights", hashes["initial"] != hashes["warm"])
    previous = hashes["warm"]
    for curve in report["learning_curve"]:
        change = curve["weight_change"]
        audit.check(
            f"ppo_{curve['update']}_weight_chain",
            change["before"] == previous and change["after"] != change["before"],
        )
        epochs = change["epochs"]
        audit.check(
            f"ppo_{curve['update']}_finite_optimizer_logs",
            bool(epochs) and all(np.isfinite(list(e.values())).all() for e in epochs),
        )
        previous = change["after"]
    audit.check(
        "latest_matches_last_update",
        previous == hashes["latest"]
        and states["latest"]["updates"] == report["updates"],
    )
    selected_update = report["selected_update"]
    selected_hash = (
        hashes["warm"]
        if selected_update == 0
        else report["learning_curve"][selected_update - 1]["weight_change"]["after"]
    )
    audit.check(
        "selected_matches_validation_selected_weights",
        selected_hash == hashes["selected"] == states["selected"]["weight_sha256"]
        and states["selected"]["selected_update"] == selected_update,
    )
    audit.check(
        "selected_checkpoint_file_hash",
        file_hash(study / "selected.pt") == report["checkpoint_sha256"],
    )
    low, high = np.asarray(spec["low"]), np.asarray(spec["high"])
    radius = spec["scene"]["ball"]["radius_m"]
    maxima = np.zeros(5)
    for label in ("initial", "warm", "selected"):
        policy = Policy()
        policy.load_state_dict(states[label]["state"])
        policy.eval()
        audit.check(
            label + "_evaluation_weight_identity",
            report["frozen_evaluation"]["comparison"][label]["policy_weight_sha256"]
            == hashes[label],
        )
        labels = ["test-" + label] + (
            ["test-lock-board", "test-lock-dropper"] if label == "selected" else []
        )
        for phase in labels:
            rows = grouped[phase]
            locks = [r["locks"] for r in rows]
            obs = torch.tensor(_observations([r["bucket_xy_m"] for r in rows], locks))
            with torch.no_grad():
                latent = policy.actor(obs).numpy()
            expected = low + (np.tanh(latent) + 1) * (high - low) / 2
            expected = _project(expected, locks, radius)
            actual = np.asarray([r["values"] for r in rows])
            error = np.abs(actual - expected).max(0)
            maxima = np.maximum(maxima, error)
            audit.check(
                phase + "_cpu_frozen_actor_actions",
                (error <= [2e-6, 2e-6, 2e-6, 2e-4, 2e-6]).all(),
                error.tolist(),
            )
    return dict(weight_sha256=hashes, maximum_action_error=maxima.tolist())


def audit_fk(assets, spec, rows, audit):
    """Independent NumPy/SciPy URDF FK of stored endpoint joints, CPU only."""
    from scipy.spatial.transform import Rotation
    from .learning_ik import read_chain

    chains = {
        role: read_chain(assets / (role + ".urdf")) for role in ("dropper", "board_arm")
    }
    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_euler(
        "z", spec["scene"]["board"]["mount_yaw_rad"]
    ).as_matrix()
    mount[2, 3] = spec["scene"]["board"]["tool_offset_m"]
    base14 = np.diag([-1.0, -1.0, 1.0, 1.0])
    base14[0, 3] = spec["scene"]["arms"]["base_spacing_m"]
    maxima = dict(
        board_saved_position_m=0.0,
        board_requested_position_m=0.0,
        dropper_requested_position_m=0.0,
        locked_position_m=0.0,
    )
    bad = []
    for row in rows:
        poses = {}
        for role, key in (("dropper", "q10"), ("board_arm", "q14")):
            origins, axes, lower, upper = chains[role]
            q = np.asarray(row[key])
            if (
                not np.isfinite(q).all()
                or (q < lower - 1e-7).any()
                or (q > upper + 1e-7).any()
            ):
                bad.append((row["id"], "joint_limits"))
            pose = np.eye(4)
            for origin, axis, value in zip(origins, axes, q):
                rotate = np.eye(4)
                rotate[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                pose = pose @ origin @ rotate
            poses[role] = pose
        board = base14 @ poses["board_arm"] @ mount
        measured_error = np.linalg.norm(board[:3, 3] - row["board_center_m"])
        maxima["board_saved_position_m"] = max(
            maxima["board_saved_position_m"], measured_error
        )
        if measured_error > 2e-5:
            bad.append((row["id"], "saved_board_fk"))
        if row["invalid"]:
            continue
        requested_board = np.linalg.norm(board[:3, 3] - row["values"][:3])
        release = poses["dropper"][:3, 3] - [0, 0, 0.095]
        requested_release = np.linalg.norm(release - row["release_ball_center_m"])
        maxima["board_requested_position_m"] = max(
            maxima["board_requested_position_m"], requested_board
        )
        maxima["dropper_requested_position_m"] = max(
            maxima["dropper_requested_position_m"], requested_release
        )
        if requested_board > 0.002 or requested_release > 0.002:
            bad.append((row["id"], "accepted_endpoint_fk"))
        for role, pose_position in (("board", board[:3, 3]), ("dropper", release)):
            lock = row["locks"][role]
            if lock is not None:
                error = np.linalg.norm(pose_position - lock)
                maxima["locked_position_m"] = max(maxima["locked_position_m"], error)
                if error > 0.002:
                    bad.append((row["id"], role + "_lock_fk"))
    audit.check(
        "independent_urdf_fk_and_valid_lock_positions",
        not bad,
        dict(maxima=maxima, failures=bad[:10]),
    )
    return maxima


def verify(study, *, source=None, assets=None, checkpoints=True):
    study = Path(study)
    spec = json.loads((study / "contract.json").read_text())
    report = json.loads((study / "report.json").read_text())
    rows = [
        json.loads(line)
        for line in (study / "shots.jsonl").read_text().splitlines()
        if line.strip()
    ]
    catalogue = json.loads((study / "catalogue.json").read_text())
    audit, grouped = audit_records(spec, report, rows, catalogue)
    audit.check(
        "split_file_matches_contract",
        json.loads((study / "splits.json").read_text())
        == {k: spec[k + "_layouts"] for k in ("train", "validation", "test")},
    )
    evidence = dict(
        version=VERSION,
        study_contract_fingerprint=spec["fingerprint"],
        records=len(rows),
        phase_counts=dict(Counter(r["phase"] for r in rows)),
    )
    if source is not None:
        source = Path(source)
        hashes = report["source_sha256"]
        mismatched = [
            name
            for name, expected in hashes.items()
            if not (source / name).is_file() or file_hash(source / name) != expected
        ]
        audit.check(
            "executed_mjlab_source_hashes",
            bool(hashes) and not mismatched,
            dict(files=len(hashes), mismatched=mismatched),
        )
        audit.warnings.append(
            "Source hashes cover the mjlab Python files declared by the runner; shared Isaac scorer/geometry dependencies need their own artifact provenance manifest."
        )
    else:
        audit.warnings.append(
            "Executed source bytes were not supplied; source-hash verification is missing."
        )
    if checkpoints:
        evidence["checkpoints"] = audit_checkpoints(study, spec, report, grouped, audit)
    else:
        audit.warnings.append(
            "Checkpoint/action verification was explicitly disabled; this is a records-only audit."
        )
    if assets is not None:
        evidence["fk"] = audit_fk(Path(assets), spec, rows, audit)
    else:
        audit.warnings.append(
            "Generated URDF assets were not supplied; independent stored-joint FK/lock verification is missing."
        )
    audit.warnings.extend(
        [
            "Saved JSONL ordering and checkpoint identity support frozen evaluation; files alone cannot prove wall-clock freeze chronology.",
            "Weight changes and hash continuity are checked. Exact PPO gradient replay is unavailable because sampled latent actions, old log probabilities and optimizer state were not all persisted.",
            "This audit does not rerun ball dynamics or establish continuous tabletop coverage, grasping, collision-free transit or physical calibration.",
        ]
    )
    evidence.update(
        passed=all(x["passed"] for x in audit.checks),
        checks=audit.checks,
        warnings=audit.warnings,
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument(
        "--source", type=Path, help="Exact executed sircus/mjlab source directory"
    )
    parser.add_argument(
        "--assets",
        type=Path,
        help="Generated UF850 URDF directory for independent CPU FK",
    )
    parser.add_argument("--records-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(
        args.study,
        source=args.source,
        assets=args.assets,
        checkpoints=not args.records_only,
    )
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)
