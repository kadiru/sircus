"""Independent read-only CPU audit of a completed finite-bank mjlab PPO study."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .verify_learning import Audit, digest, file_hash, state_hash, statistics

VERSION = "sircus-mjlab-bank-audit-v1"


def materialize(bank, xy):
    """Independent stable-slot reconstruction; no eligibility filtering or labels."""
    xy = np.asarray(xy, dtype=float)
    values, metadata = [], []
    for turn in range(16):
        angle = math.radians(turn * 22.5)
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        for index, t in enumerate(bank["templates"]):
            absolute = xy + rotation @ np.asarray(t["relative_board_xy_m"])
            values.append(
                np.column_stack(
                    (
                        absolute,
                        np.tile(
                            [t["height_m"], t["tilt_deg"], t["gap_m"]], (len(xy), 1)
                        ),
                    )
                )
            )
            metadata.append(
                dict(
                    candidate_id=f"template-{index:03d}-rotation-{turn:02d}",
                    template_index=index,
                    rotation_index=turn,
                    rotation_deg=turn * 22.5,
                    source_trial_id=t["source_trial_id"],
                    source_cell_id=t.get("source_cell_id"),
                    qualification=t.get("qualification", "unknown"),
                )
            )
    return np.stack(values, axis=1), metadata


def _mask_hash(mask):
    import hashlib

    return hashlib.sha256(np.asarray(mask, dtype=np.bool_).tobytes()).hexdigest()


def audit_geometry(spec, bank, metadata, data, report, a):
    payload = dict(bank)
    expected = payload.pop("sha256")
    a.check(
        "bank_fingerprint",
        digest(payload)
        == expected
        == spec["proposal_bank_sha256"]
        == report["proposal_bank_sha256"],
    )
    pools = {k: spec[k + "_layouts"] for k in ("train", "validation", "test")}
    merged = [tuple(xy) for group in pools.values() for xy in group]
    a.check("spatial_splits_disjoint", len(set(merged)) == len(merged))
    a.check(
        "stable_candidate_ids",
        len(metadata) == len(bank["templates"]) * 16
        and len({r["candidate_id"] for r in metadata}) == len(metadata),
    )
    for split, group in data.items():
        values, expected_metadata = materialize(bank, pools[split])
        n, k = values.shape[:2]
        a.check(
            split + "_bucket_sequence", np.array_equal(group["bucket"], pools[split])
        )
        a.check(
            split + "_candidate_materialization",
            group["values"].shape == values.shape
            and np.allclose(group["values"], values, atol=1e-12, rtol=0)
            and metadata == expected_metadata,
        )
        a.check(
            split + "_array_shapes",
            group["mask"].shape == (n, k)
            and group["reasons"].shape == (n, k)
            and group["q10"].shape == (n, k, 6)
            and group["q14"].shape == (n, k, 6),
        )
        a.check(
            split + "_mask_reason_identity",
            group["mask"].dtype == np.bool_
            and np.array_equal(group["mask"], group["reasons"] == 0),
        )
        a.check(
            split + "_finite_geometry",
            all(
                np.isfinite(group[key]).all()
                for key in ("bucket", "values", "q10", "q14")
            ),
        )
        g = report["geometry"][split]
        a.check(
            split + "_geometry_counts",
            g["queries"] == n
            and g["candidates_per_query"] == k
            and g["eligible_candidates"] == group["mask"].sum()
            and g["queries_with_candidate"] == group["mask"].any(1).sum()
            and g["unresolved_queries"] == (~group["mask"].any(1)).sum(),
        )
    return pools


def audit_rows(spec, report, rows, metadata, data, a):
    a.check(
        "study_completed",
        report.get("status") == "completed" and report.get("phase") == "completed",
    )
    payload = dict(spec)
    fp = payload.pop("fingerprint")
    a.check(
        "contract_fingerprint", digest(payload) == fp == report["contract_fingerprint"]
    )
    a.check(
        "trial_count_and_ids",
        report["records"] == len(rows)
        and [r["id"] for r in rows] == [f"bank-shot-{i:07d}" for i in range(len(rows))],
    )
    batch, updates, repeats = (
        spec["batch_size"],
        spec["ppo_updates"],
        spec["test_repeats"],
    )
    train = [r for r in rows if r["phase"] == "train-ppo"]
    validation = [r for r in rows if r["phase"] == "validation-selected-0"]
    curves = report["learning_curve"]
    a.check(
        "training_budget",
        len(train) == batch * updates == report["train_episodes"]
        and len(curves) == updates == report["updates"],
    )
    a.check(
        "training_release_count",
        sum(r["native_release"] for r in train) == report["native_train_releases"],
    )
    eligible = data["train"]["mask"].any(1)
    sampling = report["train_sampling"]
    a.check(
        "eligible_training_context_counts",
        sampling["declared_contexts"] == len(eligible)
        and sampling["eligible_contexts"] == eligible.sum()
        and sampling["excluded_contexts"] == (~eligible).sum(),
    )
    val_updates = [0] + [c["update"] for c in curves if "validation" in c]
    vn = len(data["validation"]["bucket"])
    a.check("validation_budget", len(validation) == vn * len(val_updates))
    a.stats(
        "initial_validation_statistics",
        statistics(validation[:vn]),
        report["initial_validation"],
    )
    selected_update = 0
    best = (
        report["initial_validation"]["captures"],
        report["initial_validation"]["mean_reward"],
    )
    for i, c in enumerate(curves):
        a.check(f"update_{i+1}_number", c["update"] == i + 1)
        a.stats(
            f"update_{i+1}_statistics",
            statistics(train[i * batch : (i + 1) * batch]),
            c,
        )
        if "validation" in c:
            start = val_updates.index(c["update"]) * vn
            a.stats(
                f"validation_{i+1}_statistics",
                statistics(validation[start : start + vn]),
                c["validation"],
            )
            key = (c["validation"]["captures"], c["validation"]["mean_reward"])
            if key > best:
                selected_update, best = c["update"], key
    a.check(
        "validation_only_checkpoint_selection",
        report["selected_update"] == selected_update,
    )
    allowed = {"train-ppo", "validation-selected-0"} | {
        f"test-{tag}-{rep}"
        for tag in ("uniform", "selected", "parent-v2")
        for rep in range(repeats)
    }
    a.check("known_phases", all(r["phase"] in allowed for r in rows))
    test_at = [i for i, r in enumerate(rows) if r["phase"].startswith("test-")]
    dev_at = [i for i, r in enumerate(rows) if not r["phase"].startswith("test-")]
    a.check(
        "development_precedes_test_in_jsonl",
        bool(test_at) and max(dev_at) < min(test_at),
    )
    a.check(
        "single_shot_no_search_protocol",
        all(
            v.get("search_at_inference") is False
            for v in report["frozen_evaluation"]["comparison"].values()
        ),
    )
    comparisons = {}
    for tag, label in (
        ("uniform", "uniform_valid_bank"),
        ("selected", "selected_bank"),
        ("parent-v2", "parent_v2"),
    ):
        group = [r for r in rows if r["phase"].startswith("test-" + tag + "-")]
        comparisons[tag] = group
        target = np.asarray(spec["test_layouts"])
        a.check(
            label + "_all_query_repeat_denominator",
            len(group) == len(target) * repeats
            and all(
                np.array_equal(
                    [
                        r["bucket_xy_m"]
                        for r in group[rep * len(target) : (rep + 1) * len(target)]
                    ],
                    target,
                )
                for rep in range(repeats)
            ),
        )
        a.stats(
            label + "_statistics",
            statistics(group),
            report["frozen_evaluation"]["comparison"][label],
        )
        a.check(
            label + "_metadata",
            report["frozen_evaluation"]["comparison"][label]["layouts"] == len(target)
            and report["frozen_evaluation"]["comparison"][label]["repeats"] == repeats,
        )
    a.stats(
        "selected_top_level_statistics",
        statistics(comparisons["selected"]),
        report["frozen_evaluation"],
    )
    a.check(
        "selected_report_replay_matches_jsonl",
        report["test_rows"] == comparisons["selected"],
    )
    a.check(
        "selected_trial_denominator",
        report["frozen_evaluation"]["trials"] == len(comparisons["selected"]),
    )
    bad = []
    radius = spec["scene"]["ball"]["radius_m"]
    for row in rows:
        phase = row["phase"]
        if phase.startswith("test-parent-v2"):
            continue
        split = (
            "train"
            if phase == "train-ppo"
            else "validation" if phase.startswith("validation") else "test"
        )
        group = data[split]
        query = row["query_index"]
        index = row["candidate_index"]
        if not 0 <= query < len(group["bucket"]) or not 0 <= index < len(metadata):
            bad.append((row["id"], "index"))
            continue
        mask = group["mask"][query]
        eligible = bool(mask.any())
        if (
            row["bucket_xy_m"] != group["bucket"][query].tolist()
            or row["mask_sha256"] != _mask_hash(mask)
            or row["eligible_bank_endpoint"] != eligible
        ):
            bad.append((row["id"], "query_mask"))
        if not eligible:
            if (
                row["candidate_metadata"] is not None
                or row["native_release"]
                or row["invalid"] != "no_eligible_bank_endpoint"
                or row["route_success"]
                or row.get("behavior_logp") is not None
            ):
                bad.append((row["id"], "unresolved_query"))
            continue
        if not mask[index] or row["candidate_metadata"] != metadata[index]:
            bad.append((row["id"], "masked_selection"))
        value = group["values"][query, index]
        if not np.allclose(row["values"], value, atol=1e-12, rtol=0):
            bad.append((row["id"], "candidate_values"))
        if not np.allclose(
            row["q10"], group["q10"][query, index], atol=1e-12, rtol=0
        ) or not np.allclose(
            row["q14"], group["q14"][query, index], atol=1e-12, rtol=0
        ):
            bad.append((row["id"], "cached_joint_branch"))
        if not np.allclose(
            row["release_ball_center_m"],
            value[:3] + [0, 0, radius + value[4]],
            atol=1e-12,
            rtol=0,
        ):
            bad.append((row["id"], "release_pose"))
        if row["native_release"] != (not bool(row["invalid"])):
            bad.append((row["id"], "release_flag"))
        route = bool(
            row["success"]
            and row["board_contact_episodes"] >= 1
            and not row["external_contact_steps"]
            and not row["invalid"]
        )
        if row["route_success"] != route or row["complete_success"] != route:
            bad.append((row["id"], "route_score"))
        if row["success"] and not (
            row["board_bounce"]
            and row["whole_ball_entry"]
            and row["max_retention_s"] >= 1 - 1e-8
        ):
            bad.append((row["id"], "retention"))
    a.check("stable_candidate_mask_and_cached_branch_evidence", not bad, bad[:15])
    for tag in ("uniform", "selected"):
        group = comparisons[tag]
        n = len(data["test"]["bucket"])
        a.check(
            tag + "_same_frozen_candidate_across_repeats",
            all(
                [r["candidate_index"] for r in group[:n]]
                == [r["candidate_index"] for r in group[rep * n : (rep + 1) * n]]
                for rep in range(repeats)
            ),
        )
    return dict(
        train=train,
        validation=validation,
        validation_updates=val_updates,
        comparisons=comparisons,
    )


def audit_checkpoints(study, spec, report, records, data, a):
    import torch
    from .bank_policy import FEATURE_NAMES, Policy

    torch.set_num_threads(2)
    checkpoints = []
    hashes = []
    frozen_bank = json.loads((study / "proposal-bank.json").read_text())
    for update in range(spec["ppo_updates"] + 1):
        checkpoint = torch.load(
            study / f"update-{update:03d}.pt", map_location="cpu", weights_only=True
        )
        h = state_hash(checkpoint["state"])
        checkpoints.append(checkpoint)
        hashes.append(h)
        a.check(
            f"checkpoint_{update}_identity",
            checkpoint["contract"] == spec
            and checkpoint["weight_sha256"] == h
            and checkpoint["policy_class"] == "relative_bank_categorical"
            and checkpoint["bank"] == frozen_bank
            and (update == 0 or checkpoint["update"] == update),
        )
    selected = torch.load(study / "selected.pt", map_location="cpu", weights_only=True)
    selected_hash = state_hash(selected["state"])
    a.check(
        "selected_checkpoint_identity",
        selected_hash
        == hashes[report["selected_update"]]
        == selected["weight_sha256"]
        == report["policy_weight_sha256"]
        and selected["selected_update"] == report["selected_update"]
        and selected["contract"] == spec
        and selected["bank"] == frozen_bank
        and selected["policy_class"] == "relative_bank_categorical"
        and selected["feature_names"] == list(FEATURE_NAMES)
        and file_hash(study / "selected.pt") == report["checkpoint_sha256"],
    )
    a.check(
        "reported_evaluation_weight_identities",
        report["frozen_evaluation"]["comparison"]["uniform_valid_bank"][
            "policy_weight_sha256"
        ]
        == hashes[0]
        and report["frozen_evaluation"]["comparison"]["selected_bank"][
            "policy_weight_sha256"
        ]
        == selected_hash
        and report["frozen_evaluation"]["policy_weight_sha256"] == selected_hash,
    )
    maximum_logp_error = 0.0

    def check_rows(rows, checkpoint, split, label, argmax=False):
        nonlocal maximum_logp_error
        policy = Policy()
        policy.load_state_dict(checkpoint["state"])
        policy.eval()
        group = data[split]
        ids = np.asarray([r["query_index"] for r in rows])
        indices = torch.tensor([r["candidate_index"] for r in rows])
        xy = torch.tensor(group["bucket"][ids], dtype=torch.float32)
        values = torch.tensor(group["values"][ids], dtype=torch.float32)
        mask = torch.tensor(group["mask"][ids], dtype=torch.bool)
        with torch.no_grad():
            distribution, eligible = policy.distribution(xy, values, mask)
            expected = distribution.log_prob(indices).numpy()
            picks = distribution.probs.argmax(-1).numpy()
        errors = []
        bad = []
        h = state_hash(checkpoint["state"])
        for i, row in enumerate(rows):
            if row["behavior_policy_weight_sha256"] != h:
                bad.append((row["id"], "behavior_weight"))
            if eligible[i]:
                error = abs(expected[i] - row["behavior_logp"])
                errors.append(error)
                if error > 2e-5:
                    bad.append((row["id"], "behavior_logp"))
            if argmax and picks[i] != row["candidate_index"]:
                bad.append((row["id"], "frozen_argmax"))
        maximum_logp_error = max(maximum_logp_error, max(errors, default=0))
        a.check(
            label,
            not bad,
            dict(maximum_logp_error=max(errors, default=0), failures=bad[:10]),
        )

    batch = spec["batch_size"]
    for i, curve in enumerate(report["learning_curve"]):
        change = curve["weight_change"]
        a.check(
            f"ppo_{i+1}_weight_update_chain",
            change["before"] == hashes[i]
            and change["after"] == hashes[i + 1]
            and hashes[i] != hashes[i + 1]
            and change["optimizer_rows"] == batch
            and change["excluded_all_invalid"] == 0,
        )
        a.check(
            f"ppo_{i+1}_finite_epochs",
            len(change["epochs"]) == spec["epochs"]
            and all(np.isfinite(list(row.values())).all() for row in change["epochs"]),
        )
        check_rows(
            records["train"][i * batch : (i + 1) * batch],
            checkpoints[i],
            "train",
            f"ppo_{i+1}_cpu_behavior_logp",
        )
    vn = len(data["validation"]["bucket"])
    for chunk, update in enumerate(records["validation_updates"]):
        check_rows(
            records["validation"][chunk * vn : (chunk + 1) * vn],
            checkpoints[update],
            "validation",
            f"validation_{update}_cpu_argmax",
            argmax=True,
        )
    check_rows(
        records["comparisons"]["uniform"],
        checkpoints[0],
        "test",
        "uniform_cpu_behavior_logp",
    )
    check_rows(
        records["comparisons"]["selected"],
        selected,
        "test",
        "selected_cpu_frozen_argmax",
        argmax=True,
    )
    initial = checkpoints[0]["state"]
    a.check(
        "uniform_initial_actor",
        bool((initial["actor.4.weight"] == 0).all())
        and bool((initial["actor.4.bias"] == 0).all()),
    )
    return dict(
        update_weight_sha256=hashes,
        selected_weight_sha256=selected_hash,
        maximum_logp_error=float(maximum_logp_error),
    )


def audit_parent(parent, spec, report, bank, records, a, *, actions=True):
    from .learning_search import build_bank

    parent = Path(parent)
    contract = json.loads((parent / "contract.json").read_text())
    payload = dict(contract)
    fp = payload.pop("fingerprint")
    a.check(
        "parent_contract_checkpoint_unchanged",
        digest(payload) == fp == spec["parent_contract_fingerprint"]
        and file_hash(parent / "selected.pt")
        == spec["parent_checkpoint_sha256"]
        == report["parent_checkpoint_sha256"],
    )
    original = json.loads((parent / "catalogue.json").read_text())
    actual_bank = build_bank(
        original, contract["train_layouts"], file_hash(parent / "catalogue.json"), fp
    )
    a.check("bank_uses_only_frozen_parent_train_teachers", bank == actual_bank)
    old_eval = {
        tuple(xy)
        for key in ("validation_layouts", "test_layouts")
        for xy in contract[key]
    }
    new_train = {tuple(xy) for xy in spec["train_layouts"]}
    a.check(
        "new_training_excludes_parent_validation_and_test", not (new_train & old_eval)
    )
    old_all = old_eval | {tuple(xy) for xy in contract["train_layouts"]}
    new_eval = {
        tuple(xy) for key in ("validation_layouts", "test_layouts") for xy in spec[key]
    }
    a.check("new_validation_test_are_fresh_vs_parent_layouts", not (old_all & new_eval))
    a.check(
        "parent_training_grid_preserved",
        spec["train_layouts"][:35] == contract["train_layouts"],
    )
    if actions:
        import torch
        from .learning_policy import Policy
        from .verify_learning import _observations

        cp = torch.load(parent / "selected.pt", map_location="cpu", weights_only=True)
        h = state_hash(cp["state"])
        policy = Policy()
        policy.load_state_dict(cp["state"])
        policy.eval()
        rows = records["comparisons"]["parent-v2"]
        with torch.no_grad():
            latent = policy.actor(
                torch.tensor(
                    _observations(
                        [r["bucket_xy_m"] for r in rows],
                        [dict(board=None, dropper=None) for _ in rows],
                    )
                )
            ).numpy()
        expected = (
            np.asarray(contract["low"])
            + (np.tanh(latent) + 1)
            * (np.asarray(contract["high"]) - contract["low"])
            / 2
        )
        error = np.abs(expected - np.asarray([r["values"] for r in rows])).max(0)
        a.check(
            "parent_cpu_frozen_actor_actions",
            (error <= [2e-6, 2e-6, 2e-6, 2e-4, 2e-6]).all(),
            error.tolist(),
        )
        a.check(
            "parent_evaluation_checkpoint_identity",
            h
            == cp["weight_sha256"]
            == report["frozen_evaluation"]["comparison"]["parent_v2"][
                "policy_weight_sha256"
            ]
            and file_hash(parent / "selected.pt")
            == report["frozen_evaluation"]["comparison"]["parent_v2"][
                "checkpoint_sha256"
            ],
        )
    return dict(
        contract_fingerprint=fp,
        checkpoint_sha256=file_hash(parent / "selected.pt"),
        catalogue_sha256=file_hash(parent / "catalogue.json"),
    )


def verify(study, *, source=None, parent_study=None, checkpoints=True):
    study = Path(study)
    spec = json.loads((study / "contract.json").read_text())
    report = json.loads((study / "report.json").read_text())
    bank = json.loads((study / "proposal-bank.json").read_text())
    metadata = json.loads((study / "candidate-metadata.json").read_text())
    data = {
        label: dict(np.load(study / f"geometry-{label}.npz", allow_pickle=False))
        for label in ("train", "validation", "test")
    }
    rows = [
        json.loads(line)
        for line in (study / "shots.jsonl").read_text().splitlines()
        if line.strip()
    ]
    a = Audit()
    audit_geometry(spec, bank, metadata, data, report, a)
    records = audit_rows(spec, report, rows, metadata, data, a)
    a.check(
        "split_file_identity",
        json.loads((study / "splits.json").read_text())
        == {key: spec[key + "_layouts"] for key in ("train", "validation", "test")},
    )
    required = (
        {
            "contract.json",
            "splits.json",
            "proposal-bank.json",
            "candidate-metadata.json",
            "shots.jsonl",
            "selected.pt",
        }
        | {f"geometry-{split}.npz" for split in data}
        | {f"update-{i:03d}.pt" for i in range(spec["ppo_updates"] + 1)}
    )
    artifact_hashes = report["artifacts_sha256"]
    a.check("critical_artifacts_manifest_complete", required.issubset(artifact_hashes))
    mismatches = [
        name
        for name, sha in artifact_hashes.items()
        if Path(name).name != name
        or not (study / name).is_file()
        or file_hash(study / name) != sha
    ]
    a.check(
        "all_declared_artifact_hashes",
        not mismatches,
        dict(files=len(artifact_hashes), mismatches=mismatches),
    )
    if source is not None:
        source = Path(source)
        hashes = report["source_sha256"]
        changed = [
            name
            for name, sha in hashes.items()
            if not (source / name).is_file() or file_hash(source / name) != sha
        ]
        a.check(
            "executed_source_hashes",
            not changed
            and {"bank_geometry.py", "bank_policy.py", "bank_study.py"}.issubset(
                hashes
            ),
            dict(files=len(hashes), mismatches=changed),
        )
    else:
        a.warnings.append(
            "Executed source bytes not supplied; source verification is missing."
        )
    result = dict(
        version=VERSION,
        contract_fingerprint=spec["fingerprint"],
        records=len(rows),
        phase_counts=dict(Counter(r["phase"] for r in rows)),
    )
    if checkpoints:
        result["checkpoints"] = audit_checkpoints(study, spec, report, records, data, a)
    else:
        a.warnings.append(
            "Checkpoint/action verification explicitly disabled; records-only audit."
        )
    if parent_study is not None:
        result["parent"] = audit_parent(
            parent_study, spec, report, bank, records, a, actions=checkpoints
        )
    else:
        a.warnings.append(
            "Parent study unavailable; parent provenance, old held-out split isolation and parent actor reproduction are missing."
        )
    a.warnings.extend(
        [
            "Nominal frozen repeats test numerical repeatability, not perturbed reliability or continuous workspace coverage.",
            "Saved file ordering and identities support frozen evaluation but cannot independently prove wall-clock freezing chronology.",
            "PPO checkpoint changes, selected actions and pre-update behavior probabilities are verified; optimizer gradients and stochastic GPU RNG samples are not independently replayed.",
            "Native cached geometry was checksum-bound; this audit does not rerun ball dynamics or certify collision-free arm transit. Parent/shared-asset provenance requires the separate artifact verifier.",
        ]
    )
    result.update(
        passed=all(c["passed"] for c in a.checks), checks=a.checks, warnings=a.warnings
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--parent-study", type=Path)
    p.add_argument("--records-only", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = verify(
        args.study,
        source=args.source,
        parent_study=args.parent_study,
        checkpoints=not args.records_only,
    )
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    raise SystemExit(not result["passed"])
