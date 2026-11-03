"""Independent CPU evidence audit for the fixed-bucket gap-window study.

This checks saved requests, training provenance and observations, not new native
dynamics. ``--source`` names the immutable snapshot that executed the study;
later audit-only code may run from another checkout without relabelling it.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .verify_expansion import Audit, digest, sha256
from .verify_contact import SCORER_KEYS, NUMERIC_SCORER_KEYS

VERSION = "sircus-fixed-bucket-window-evidence-audit-v1"
STUDY_VERSION = "sircus-fixed-bucket-gap-study-v1"
PHYSICS = dict(version="sircus-articulated-passive-contact-15360hz-v1",
    board_mode="articulated", gravity_mode="passive_only", substeps=64,
    native_dt_s=1 / 15360, scoring_dt_s=1 / 240, horizon_s=4.,
    solref=[.004, .15], contact_friction=.35, solver="newton",
    solver_iterations=100, solver_tolerance=1e-8, cone="pyramidal",
    integrator="implicitfast")
METHODS = ("adaptive_surrogate", "sac")
MILESTONES = (504, 672, 840, 1008)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def compact_digest(value):
    """The model checkpoint format uses compact JSON, unlike study manifests."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def row_reward(row):
    """Reconstruct the historical terminal reward without runner helpers."""
    if row.get("invalid"):
        return -1.
    shaped = .15 * float(row["board_bounce"]) + .5 * np.exp(-row["closest_opening_distance_m"] / .12)
    if row["external_contact_steps"]:
        return float(shaped - .5)
    return float(shaped + .5 * min(row["max_retention_s"], 1.) + 3 * row["route_success"]
        + .3 * (row["route_quality"] == 2) + .1 * row["single_bounce_success"])


def stats(rows):
    return dict(attempts=len(rows), captures=sum(bool(r["route_success"]) and not r.get("invalid") for r in rows),
        clean=sum(r["route_quality"] == 2 and not r.get("invalid") for r in rows),
        invalid=sum(bool(r.get("invalid")) for r in rows),
        board_bounces=sum(bool(r["board_bounce"]) and not r.get("invalid") for r in rows),
        native_releases=sum(bool(r.get("native_release")) for r in rows),
        mean_reward=float(np.mean([row_reward(r) for r in rows])) if rows else 0.)


def audit_splits(a, spec, prepared):
    """Geometry contexts are fixed before labels; unsupported contexts stay put."""
    groups = spec["contexts"]
    declared = [r for role in ("train", "validation", "test") for r in groups[role]]
    ids = [r["id"] for r in declared]
    points = [(r["board_height_m"], r["tilt_deg"]) for r in declared]
    a.check("prespecified_24_6_8_disjoint_contexts", [len(groups[k]) for k in ("train", "validation", "test")] == [24, 6, 8]
        and len(ids) == len(set(ids)) == len(set(points)) == 38 and np.isfinite(points).all())
    contexts = {r["id"]: r for r in declared}
    result = {r["id"]: r for r in prepared}
    a.check("all_prepared_contexts_including_unsupported", Counter(ids) == Counter(r["id"] for r in prepared)
        and all(r.get("prepared") is True for r in prepared))
    bad = []
    for name, row in result.items():
        c = contexts.get(name)
        if c is None or any(row.get(k) != c[k] for k in ("board_height_m", "tilt_deg")):
            bad.append(name); continue
        if row.get("board_admitted"):
            q = np.asarray(row.get("q14", []), float)
            from .admission import VERIFIED_PARK_Q10
            park = spec.get("board_branch_park_q10", spec.get("parked_dropper_q10", list(VERIFIED_PARK_Q10)))
            proof = dict(version="sircus-fixed-board-gap-window-geometry-v1",
                bucket_xy_m=spec["bucket_xy_m"], board_world=row.get("board_world"),
                q14=row.get("q14"), park_q10=park, scene_sha256=digest(spec.get("scene", {})))
            if q.shape != (6,) or not np.isfinite(q).all() or digest(proof) != row.get("board_branch_sha256"):
                bad.append(name)
        elif row.get("q14") is not None or row.get("board_branch_sha256") is not None:
            bad.append(name)
    a.check("frozen_context_geometry_and_branch_identity", not bad, bad[:10])


def audit_rows(a, spec, prepared, rows):
    contexts = {r["id"]: (role, r) for role in ("train", "validation", "test") for r in spec["contexts"][role]}
    prepared = {r["id"]: r for r in prepared}
    a.check("unique_record_ids", len(rows) == len({r["id"] for r in rows}))
    bad_role = []; bad_pose = []; bad_branch = []; bad_release = []; bad_reward = []
    for row in rows:
        name = row.get("context_id")
        if row.get("role") == "control":
            # Controls are quarantined. Their exact anchor is checked separately.
            pass
        elif name not in contexts or row.get("role") != contexts[name][0]:
            bad_role.append(row["id"])
        value = np.asarray(row.get("values", []), float)
        if value.shape != (5,) or not np.isfinite(value).all():
            bad_pose.append(row["id"]); continue
        if row.get("role") != "control" and name in contexts:
            c = contexts[name][1]
            if not np.array_equal(value[:2], spec["board_xy_m"]) or value[2] != c["board_height_m"] or value[3] != c["tilt_deg"]:
                bad_pose.append(row["id"])
        if not np.array_equal(row.get("bucket_xy_m"), spec["bucket_xy_m"]) or value[4] != row.get("gap_m"):
            bad_pose.append(row["id"])
        p = prepared.get(name)
        if row.get("role") != "control" and p is not None:
            if row.get("board_branch_sha256") != p.get("board_branch_sha256"):
                bad_branch.append(row["id"])
            if p.get("board_admitted") and not np.array_equal(row.get("q14"), p["q14"]):
                bad_branch.append(row["id"])
        eligible = row.get("eligible", row.get("geometry_eligible"))
        within = spec["gap_bounds"][0] <= value[4] <= spec["gap_bounds"][1]
        if eligible is True:
            q10 = np.asarray(row.get("q10", []), float)
            margin = row.get("minimum_clearance_lower_bound_m")
            if not within or q10.shape != (6,) or not np.isfinite(q10).all() or margin is None or margin < .005:
                bad_release.append(row["id"])
        if (not eligible and (row.get("native_release") or row.get("route_success") or not row.get("invalid"))) or (
            row.get("native_release") and row.get("invalid")):
            bad_release.append(row["id"])
        if row.get("route_success") and (not row.get("native_release") or not row.get("board_bounce")
                or not row.get("whole_ball_entry") or row.get("max_retention_s", 0) < 1 - 1e-9
                or row.get("external_contact_steps")):
            bad_release.append(row["id"])
        if not np.isclose(row.get("learning_reward", np.nan), row_reward(row), atol=1e-9, rtol=1e-9):
            bad_reward.append(row["id"])
    for name, values in (("all_rows_keep_declared_split_roles", bad_role), ("fixed_bucket_xy_context_and_gap_identity", bad_pose),
            ("every_gap_retains_frozen_q14_branch", bad_branch), ("strict_release_and_capture_invariants", bad_release),
            ("unchanged_terminal_reward", bad_reward)):
        a.check(name, not values, values[:10])


def audit_training_ids(a, rows, checkpoints, *, seed_ids, adaptive_ids, sac_ids):
    """Explicit replay identities prevent a fair-looking count hiding leakage."""
    lookup = {r["id"]: r for r in rows}
    shared, adaptive, sac = map(list, (seed_ids, adaptive_ids, sac_ids))
    a.check("training_requests_504_shared_plus_504_each", len(shared) == len(set(shared)) == 504
        and len(adaptive) == len(set(adaptive)) == 504 and len(sac) == len(set(sac)) == 504
        and not (set(shared) & set(adaptive) or set(shared) & set(sac) or set(adaptive) & set(sac)))
    training = shared + adaptive + sac
    a.check("all_and_only_1512_unique_training_requests", set(training) == {r["id"] for r in rows if r.get("role") == "train"}
        and len(training) == len(set(training)) == 1512
        and all(i in lookup and lookup[i].get("role") == "train" for i in training))
    by_method = {m: [c for c in checkpoints if c["method"] == m] for m in METHODS}
    a.check("four_equal_training_milestones_each", all(sorted(c.get("training_requests", c.get("milestone")) for c in cc) == list(MILESTONES)
        for cc in by_method.values()) and len(checkpoints) == 8)
    bad = []
    for method, cc in by_method.items():
        extra = adaptive if method == "adaptive_surrogate" else sac
        allowed = shared + extra
        for c in cc:
            n = c.get("training_requests", c.get("milestone"))
            # Prefix identity also proves that validation/test/control labels
            # and the other learner's adaptive data cannot enter a checkpoint.
            if c.get("training_record_ids") != allowed[:n] or c.get("training_record_ids_sha256", digest(allowed[:n])) != digest(allowed[:n]):
                bad.append(c.get("id", method + "-" + str(n)))
    a.check("checkpoints_train_only_exact_shared_then_method_prefix", not bad, bad[:10])


def _inside(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Evidence path escapes its declared root: " + name)
    return path


def audit_checkpoint_files(a, study, fp, settings, rows, checkpoints):
    """Check actual serialized weights/replay against the declared TRAIN prefix."""
    by_id = {r["id"]: r for r in rows}
    for c in checkpoints:
        label = c["method"] + "-" + str(c["milestone"])
        path = _inside(study, c["path"])
        a.check(label + ":checkpoint_file_sha256", path.is_file() and sha256(path) == c["sha256"])
        training = [by_id[i] for i in c["training_record_ids"]]
        contexts = np.asarray([r["values"][2:4] for r in training], float)
        gaps = [r["gap_m"] for r in training]
        labels = [int(bool(r["route_success"])) for r in training]
        if c["method"] == "adaptive_surrogate":
            value = read(path); payload = dict(value); identity = payload.pop("fingerprint", None)
            log = value.get("fit_log", {})
            expected_data = compact_digest(dict(contexts=contexts.tolist(), gaps=gaps, labels=labels))
            a.check(label + ":actual_model_train_identity", compact_digest(payload) == identity
                and value.get("contract_fingerprint") == fp and value.get("data_roles") == ["train"]
                and value.get("version") == "sircus-window-rbf-logistic-v1"
                and log.get("rows") == len(training) and log.get("positive_labels") == sum(labels)
                and log.get("training_data_fingerprint") == expected_data)
        elif c["method"] == "sac":
            import torch
            value = torch.load(path, map_location="cpu", weights_only=True)
            weight_hash = hashlib.sha256()
            for prefix in ("actor", "q1", "q2"):
                for name, tensor in sorted(value[prefix].items()):
                    weight_hash.update((prefix + ":" + name).encode())
                    weight_hash.update(tensor.detach().cpu().numpy().tobytes())
            expected_updates = settings["sac"]["warmup_updates"] + ((c["milestone"] - 504) // 24) * settings["sac"]["updates_per_round"]
            a.check(label + ":actual_terminal_actor_critic_identity", value.get("contract_fingerprint") == fp
                and value.get("version") == "sircus-terminal-gap-sac-v1" and value.get("terminal") is True
                and value.get("data_roles") == ["train"] and value.get("context_names") == ["board_height_m", "tilt_deg"]
                and value.get("action") == "release_gap_m" and value.get("alpha") == settings["sac"]["alpha"]
                and value.get("total_seen") == len(training) and value.get("updates") == expected_updates
                and value.get("weight_hash") == weight_hash.hexdigest())
            replay = value.get("replay", {})
            cb, gb = np.asarray(value["context_bounds"]), np.asarray(value["gap_bounds"])
            expected_context = (2 * (contexts - cb[:, 0]) / (cb[:, 1] - cb[:, 0]) - 1).astype(np.float32)
            expected_actions = (2 * (np.asarray(gaps) - gb[0]) / (gb[1] - gb[0]) - 1).astype(np.float32)[:, None]
            expected_reward = np.asarray([row_reward(r) for r in training], np.float32)[:, None]
            a.check(label + ":actual_replay_exact_train_inputs_and_rewards", value.get("replay_included") is True
                and set(replay) == {"contexts", "actions", "rewards"}
                and np.array_equal(replay["contexts"].numpy(), expected_context)
                and np.array_equal(replay["actions"].numpy(), expected_actions)
                and np.array_equal(replay["rewards"].numpy(), expected_reward))


def audit_geometry_evidence(a, prepared, rows):
    """The inline geometry proof and repeated request cache cannot diverge."""
    bad = []; inconsistent = []; cache = {}
    frozen = {r["id"]: r for r in prepared}
    for row in rows:
        geometry = row.get("geometry")
        if not isinstance(geometry, dict):
            bad.append(row["id"]); continue
        keys = ("values", "bucket_xy_m", "q10", "q14", "board_branch_sha256", "eligible")
        if any(row.get(k) != geometry.get(k) for k in keys) or geometry.get("context_id") != row.get("context_id"):
            bad.append(row["id"])
        if geometry.get("geometry_version") != "sircus-fixed-board-gap-window-geometry-v1":
            bad.append(row["id"])
        if any(k in geometry for k in ("route_success", "learning_reward", "reward", "whole_ball_entry")):
            bad.append(row["id"])
        if row.get("role") != "control" and row["context_id"] not in frozen:
            bad.append(row["id"])
        key = (row["context_id"], float(row["gap_m"]).hex())
        identity = digest(geometry)
        if key in cache and cache[key] != identity:
            inconsistent.append(row["id"])
        cache[key] = identity
    a.check("inline_geometry_matches_exact_executed_request", not bad, bad[:10])
    a.check("identical_requests_have_method_independent_geometry", not inconsistent, inconsistent[:10])


def audit_request_budgets(a, settings, rows, report):
    """Regenerate the outcome-free seed and balanced per-method request counts."""
    roles = Counter(r["role"] for r in rows)
    a.check("all_2211_requests_including_invalids_and_controls", roles == dict(control=3, train=1512, validation=144, test=552)
        and len(rows) == 2211 and [r["id"] for r in rows] == [f"window-shot-{i:06d}" for i in range(2211)])
    shared = [r for r in rows if r["role"] == "train" and r["method"] == "shared"]
    adaptive = [r for r in rows if r["role"] == "train" and r["method"] == "adaptive_surrogate"]
    sac = [r for r in rows if r["role"] == "train" and r["method"] == "sac"]
    expected_seed = [(c["id"], float(g)) for c in settings["contexts"]["train"] for g in settings["seed_gaps_m"]]
    a.check("complete_same_504_seed_geometry_requests", [(r["context_id"], r["gap_m"]) for r in shared] == expected_seed
        and all(r["phase"] == "coarse_seed" for r in shared))
    expected = Counter({c["id"]: 21 for c in settings["contexts"]["train"]})
    a.check("balanced_additional_504_requests_each", Counter(r["context_id"] for r in adaptive) == expected
        and Counter(r["context_id"] for r in sac) == expected)
    a.check("fixed_adaptive_and_sac_round_denominators", Counter(r.get("round") for r in adaptive) == {1: 168, 2: 168, 3: 168}
        and Counter(r.get("round") for r in sac) == {i: 24 for i in range(21)}
        and all(r["phase"] == "adaptive_refinement" for r in adaptive)
        and all(r["phase"] == "sac_exploration" for r in sac))
    counts = report.get("counts", {})
    a.check("reported_full_denominators_and_no_free_training", counts.get("requests") == len(rows)
        and counts.get("native_releases") == sum(bool(r.get("native_release")) for r in rows)
        and counts.get("captures") == sum(bool(r["route_success"]) for r in rows)
        and counts.get("geometry_rejections") == sum(bool(r.get("invalid")) for r in rows)
        and counts.get("role_counts") == dict(roles)
        and counts.get("method_counts") == dict(Counter(r["method"] for r in rows))
        and counts.get("charged_training") == {"adaptive_surrogate": 1008, "sac": 1008})
    return dict(seed_ids=[r["id"] for r in shared], adaptive_ids=[r["id"] for r in adaptive], sac_ids=[r["id"] for r in sac])


def audit_noise_plans(a, study, settings, rows, checkpoints):
    """Regenerate every paired evaluation perturbation without using outcomes."""
    lookup = {r["id"]: r for r in rows}
    bad = []
    for checkpoint in checkpoints:
        stage = MILESTONES.index(checkpoint["milestone"])
        selected = [lookup[i] for i in checkpoint["validation_record_ids"]]
        noise = np.random.default_rng(settings["seed"] + 100 + stage)
        expected = []
        for c in settings["contexts"]["validation"]:
            expected += [(c["id"], repeat, "nominal" if repeat == 0 else "fresh", float(delta))
                for repeat, delta in enumerate([0.] + noise.normal(0, .002, 2).tolist())]
        if len(selected) != len(expected):
            bad.append(checkpoint["path"]); continue
        for row, (context, repeat, phase, delta) in zip(selected, expected):
            if row["context_id"] != context or row["repeat_index"] != repeat or row["phase"] != phase or row.get("gap_noise_m") != delta:
                bad.append(row["id"])
            if row.get("gap_m") != float(row["nominal_gap_m"] + delta) or row.get("requested_gap_m") != row.get("gap_m"):
                bad.append(row["id"])
    frozen = read(Path(study) / "frozen-test-plan.json")
    test = [r for r in rows if r["role"] == "test"]
    a.check("all_test_requests_match_pre_outcome_frozen_plan", len(frozen) == len(test) == 552
        and all(all(row.get(k) == value for k, value in p.items()) for row, p in zip(test, frozen)))
    for method in (*METHODS, "uniform_bounded"):
        rr = [r for r in test if r["method"] == method]
        rng = np.random.default_rng(settings["seed"] + 200)
        expected = []
        for c in settings["contexts"]["test"]:
            expected += [(c["id"], repeat, "nominal" if repeat < 3 else "fresh", float(delta))
                for repeat, delta in enumerate([0.] * 3 + rng.normal(0, .002, 20).tolist())]
        for row, (context, repeat, phase, delta) in zip(rr, expected):
            if row["context_id"] != context or row["repeat_index"] != repeat or row["phase"] != phase or row.get("gap_noise_m") != delta:
                bad.append(row["id"])
            if row.get("gap_m") != float(row["nominal_gap_m"] + delta) or row.get("requested_gap_m") != row.get("gap_m"):
                bad.append(row["id"])
    a.check("exact_paired_seeded_validation_and_test_noise_no_resampling", not bad, bad[:10])
    uniform = np.random.default_rng(settings["seed"] + 300).uniform(.02, .65, 8)
    rr = [r for r in test if r["method"] == "uniform_bounded"]
    expected = {c["id"]: float(g) for c, g in zip(settings["contexts"]["test"], uniform)}
    a.check("eight_outcome_free_uniform_actions_without_oracle_masking", len(rr) == 184
        and all(r["nominal_gap_m"] == expected[r["context_id"]] and r.get("source_checkpoint_sha256") is None for r in rr))


def audit_batches(a, study, spec, rows):
    study = Path(study); lookup = {r["id"]: r for r in rows}
    batches = read(study / "batches.json"); all_ids = []; rescored = []
    a.check("batch_packets_sequential", [b["packet"] for b in batches] == list(range(len(batches))))
    for batch in batches:
        label = "batch-" + str(batch["packet"])
        ids = batch["record_ids"]; all_ids.extend(ids)
        plan = read(_inside(study, batch["plan_file"]))
        cache = read(_inside(study, batch["geometry_cache_file"]))
        selected = [lookup[i] for i in ids]
        a.check(label + ":pre_outcome_plan_and_geometry_identity", ids == [r["id"] for r in plan] == cache["record_ids"]
            and len(ids) == len(cache["rows"]) and 0 < len(ids) <= spec["batch"]
            and all(all(row.get(k) == v for k, v in request.items()) for row, request in zip(selected, plan))
            and all(row["geometry"] == geometry and row["geometry_cache_file"] == batch["geometry_cache_file"]
                and row["geometry_cache_row"] == i for i, (row, geometry) in enumerate(zip(selected, cache["rows"]))))
        native = [r for r in selected if r.get("native_release")]
        native_ids = [r["id"] for r in native]
        a.check(label + ":all_admitted_and_only_admitted_release", native_ids == batch["native_record_ids"]
            == [r["id"] for r in selected if r["eligible"]])
        if native:
            file = batch["native_cache_file"]
            with np.load(_inside(study, file), allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
            a.check(label + ":native_cache_row_association", all(r.get("native_cache_file") == file
                and r.get("native_cache_row") == i for i, r in enumerate(native)))
            audit_native(a, spec["scene"], native, arrays, label)
            rescored.extend(native_ids)
        else:
            a.check(label + ":zero_releases_have_no_native_cache", batch["native_cache_file"] is None)
        a.check(label + ":rejected_requests_have_no_native_row", all(r.get("native_cache_file") is None
            and r.get("native_cache_row") is None for r in selected if not r.get("native_release")))
    a.check("every_requested_record_present_once_in_batches", all_ids == [r["id"] for r in rows]
        and len(all_ids) == len(set(all_ids)))
    a.check("every_released_record_independently_rescored_once", Counter(rescored)
        == Counter(r["id"] for r in rows if r.get("native_release")))
    return len(rescored)


def audit_evaluation(a, settings, rows, checkpoints, selected):
    """Selection sees validation only; test actions never search native outcomes."""
    by_id = {r["id"]: r for r in rows}; val_ids = []
    scores = {}; expected_validation_contexts = Counter({r["id"]: 3 for r in settings["contexts"]["validation"]})
    for c in checkpoints:
        label = c["method"] + "-" + str(c["milestone"])
        ids = c.get("validation_record_ids", [])
        vv = [by_id[i] for i in ids if i in by_id]; val_ids.extend(ids)
        okay = len(ids) == len(set(ids)) == len(vv) == 18
        okay &= Counter(r["context_id"] for r in vv) == expected_validation_contexts
        okay &= all(r["role"] == "validation" and r["method"] == c["method"] and r["milestone"] == c["milestone"]
            and r.get("source_checkpoint_sha256") == c["sha256"] for r in vv)
        a.check(label + ":same_18_validation_requests", okay)
        scores[(c["method"], c["milestone"])] = stats(vv)
    a.check("all_and_only_144_validation_requests", Counter(val_ids) == Counter(r["id"] for r in rows if r["role"] == "validation")
        and len(val_ids) == len(set(val_ids)) == 144)
    if isinstance(selected, list):
        chosen = {c["method"]: c for c in selected}
    else:
        chosen = selected.get("selected", selected)
    for method in METHODS:
        cc = [c for c in checkpoints if c["method"] == method]
        best = max(cc, key=lambda c: (scores[(method, c["milestone"])]["captures"],
            scores[(method, c["milestone"])]["mean_reward"], -c["milestone"]))
        selection = chosen.get(method, {})
        a.check(method + ":checkpoint_selected_by_validation_only", selection.get("milestone") == best["milestone"]
            and selection.get("sha256") == best["sha256"] and selection.get("path") == best["path"])
        rr = [r for r in rows if r["role"] == "test" and r["method"] == method]
        a.check(method + ":test_uses_only_frozen_selected_checkpoint", all(r.get("source_checkpoint_sha256") == best["sha256"] for r in rr))
    test_context_ids = {r["id"] for r in settings["contexts"]["test"]}
    a.check("test_has_only_two_frozen_learners_and_uniform", {r["method"] for r in rows if r["role"] == "test"}
        == set((*METHODS, "uniform_bounded")))
    noise_by_method = {}
    for method in (*METHODS, "uniform_bounded"):
        rr = [r for r in rows if r["role"] == "test" and r["method"] == method]
        expected = Counter({name: 23 for name in test_context_ids})
        a.check(method + ":all_8_test_contexts_and_184_requests", len(rr) == 184 and Counter(r["context_id"] for r in rr) == expected)
        bad = []; noise = {}
        for context in test_context_ids:
            cc = [r for r in rr if r["context_id"] == context]
            base = {r.get("nominal_gap_m") for r in cc}
            nominal = [r for r in cc if r.get("phase") == "nominal"]
            fresh = [r for r in cc if r.get("phase") == "fresh"]
            if len(base) != 1 or None in base or len(nominal) != 3 or len(fresh) != 20:
                bad.append(context); continue
            nominal_gap = next(iter(base))
            if any(r["gap_m"] != nominal_gap for r in nominal):
                bad.append(context)
            # Every attempted noisy action remains present; no clipping or retry
            # turns a rejected perturbation into a different successful request.
            if {r.get("repeat_index") for r in nominal} != set(range(3)) or {r.get("repeat_index") for r in fresh} != set(range(3, 23)):
                bad.append(context)
            fresh.sort(key=lambda r: r["repeat_index"])
            noise[context] = [float(r["gap_m"]) - float(nominal_gap) for r in fresh]
        noise_by_method[method] = noise
        a.check(method + ":frozen_gap_three_nominal_twenty_fresh", not bad, bad)
    reference = noise_by_method[METHODS[0]]
    a.check("same_gap_noise_schedule_all_test_methods", all(set(value) == set(reference) and all(
        np.allclose(value[k], reference[k], atol=2e-16, rtol=0) for k in reference)
        for value in noise_by_method.values()))


def audit_frozen_predictions(a, study, fp, settings, rows, checkpoints, selected):
    """Predict only from serialized frozen models; never evaluate a new shot."""
    from .window_learning import SuccessModel, SACAgent
    by_id = {r["id"]: r for r in rows}
    for checkpoint in checkpoints:
        method = checkpoint["method"]
        cls = SuccessModel if method == "adaptive_surrogate" else SACAgent
        model = cls.load(_inside(study, checkpoint["path"]), fp)
        contexts = np.asarray([[c["board_height_m"], c["tilt_deg"]] for c in settings["contexts"]["validation"]])
        gaps = model.propose(contexts)
        expected = {c["id"]: float(g) for c, g in zip(settings["contexts"]["validation"], gaps)}
        validation = [by_id[i] for i in checkpoint["validation_record_ids"]]
        a.check(method + "-" + str(checkpoint["milestone"]) + ":validation_frozen_model_actions", all(
            r["nominal_gap_m"] == expected[r["context_id"]] for r in validation))
        if selected[method]["sha256"] == checkpoint["sha256"]:
            contexts = np.asarray([[c["board_height_m"], c["tilt_deg"]] for c in settings["contexts"]["test"]])
            gaps = model.propose(contexts)
            expected = {c["id"]: float(g) for c, g in zip(settings["contexts"]["test"], gaps)}
            test = [r for r in rows if r["role"] == "test" and r["method"] == method]
            a.check(method + ":test_frozen_model_actions_without_native_search", all(
                r["nominal_gap_m"] == expected[r["context_id"]] for r in test))


def audit_native(a, scene, rows, arrays, label):
    """Recompute all canonical scorer fields, binding NPZ rows to exact IDs."""
    from .contract import ScoreStream, FORCE_NAMES
    ids = list(map(str, arrays.get("record_ids", [])))
    expected = [r["id"] for r in rows]
    a.check(label + ":record_ids", ids == expected and len(ids) == len(set(ids)))
    n = len(rows)
    good = all(k in arrays and arrays[k].shape == (960, n, width) and np.isfinite(arrays[k]).all()
        for k, width in (("position_m", 3), ("velocity_m_s", 3), ("forces_N", 6)))
    a.check(label + ":finite_full_trace_shapes", good)
    if not good or ids != expected:
        return
    a.check(label + ":released_corrected_protocol", all(r.get("native_release") is True and not r.get("invalid")
        and r.get("contact_protocol") == dict(substeps=64, board_mode="articulated", gravity_mode="passive_only") for r in rows))
    a.check(label + ":nonnegative_forces", bool((arrays["forces_N"] >= 0).all()))
    bad = []
    for i, row in enumerate(rows):
        scorer = ScoreStream(scene, row["bucket_xy_m"], row["board_normal"])
        for p, v, f in zip(arrays["position_m"][:, i], arrays["velocity_m_s"][:, i], arrays["forces_N"][:, i]):
            scorer.step(p, v, dict(zip(FORCE_NAMES, f)))
            if scorer.finished:
                break
        actual = scorer.result()
        discrete = all(actual[k] == row.get(k) for k in SCORER_KEYS)
        numeric = all((actual[k] is None and row.get(k) is None) or (actual[k] is not None and row.get(k) is not None
            and np.isclose(actual[k], row[k], atol=1e-3, rtol=1e-5)) for k in NUMERIC_SCORER_KEYS)
        if not discrete or not numeric:
            bad.append(row["id"])
    a.check(label + ":independent_scalar_rescore", not bad, bad[:10])


def audit_source(a, study, source, assets, spec, report):
    source = Path(source).resolve(); study = Path(study).resolve()
    for name, expected in spec["source_sha256"].items():
        path = (source / name).resolve()
        a.check("executed_source:" + name, path.is_relative_to(source) and path.is_file() and sha256(path) == expected)
    auditor = Path(__file__).resolve().parents[2]
    for name in ("sircus/mjlab/contract.py", "sircus/isaac/geometry.py", "sircus/isaac/rl_contract.py",
            "sircus/isaac/entry_quality.py", "sircus/isaac/board_route.py", "sircus/mjlab/learning_contract.py",
            "sircus/mjlab/window_learning.py", "sircus/mjlab/window_contract.py", "sircus/mjlab/admission.py"):
        a.check("matching_independent_scorer:" + name, sha256(auditor / name) == spec["source_sha256"].get(name))
    if assets is not None:
        root = Path(assets).resolve().parent
        for name, expected in spec["asset_sha256"].items():
            path = (root / name).resolve()
            a.check("native_asset:" + name, path.is_relative_to(root) and path.is_file() and sha256(path) == expected)
    else:
        a.check("native_assets_available_for_audit", False, "Pass --assets to verify recorded native model inputs")
    for name, expected in report.get("evidence_sha256", {}).items():
        path = (study / name).resolve()
        a.check("saved_evidence:" + name, path.is_relative_to(study) and path.is_file() and sha256(path) == expected)
    manifest = read(study / "evidence-manifest.json")
    actual_files = {str(p.relative_to(study)) for p in study.rglob("*") if p.is_file()
        and p.name not in ("report.json", "evidence-manifest.json") and not p.name.startswith("audit")}
    a.check("complete_evidence_manifest_identity", manifest == report.get("evidence_sha256")
        and set(manifest) == actual_files)


def audit(study, source, assets=None):
    """Audit one completed production evidence directory; never launch a job."""
    study = Path(study); spec = read(study / "contract.json"); report = read(study / "report.json")
    rows = lines(study / "records.jsonl"); prepared = read(study / "prepared-contexts.json")
    a = Audit(); payload = dict(spec); fp = payload.pop("fingerprint", None)
    a.check("contract_fingerprint", digest(payload) == fp == report.get("contract_fingerprint"))
    a.check("completed_window_study_version", spec.get("version") == report.get("version") == STUDY_VERSION
        and report.get("status") == report.get("phase") == "completed")
    a.check("corrected64_physics", spec.get("physics") == report.get("physics")
        and all(spec.get("physics", {}).get(k) == v for k, v in PHYSICS.items()))
    audit_source(a, study, source, assets, spec, report)
    from .window_contract import settings as frozen_settings
    settings = spec["settings"]
    a.check("prespecified_protocol_settings_and_report_identity", settings == json.loads(json.dumps(frozen_settings())) == report.get("settings"))
    joined = dict(settings, scene=spec["scene"], parked_dropper_q10=spec["parked_dropper_q10"])
    audit_splits(a, joined, prepared); audit_rows(a, joined, prepared, rows)
    a.check("prepared_context_file_report_identity", prepared == report.get("prepared_contexts"))
    audit_geometry_evidence(a, prepared, rows)
    identity = audit_request_budgets(a, settings, rows, report)
    checkpoints = read(study / "checkpoints.json"); selected = read(study / "selected-checkpoints.json")
    audit_training_ids(a, rows, checkpoints, **identity)
    audit_checkpoint_files(a, study, fp, settings, rows, checkpoints)
    audit_evaluation(a, settings, rows, checkpoints, selected)
    audit_noise_plans(a, study, settings, rows, checkpoints)
    audit_frozen_predictions(a, study, fp, settings, rows, checkpoints, selected)
    a.check("selected_checkpoint_report_identity", selected == report.get("selected_checkpoints"))
    for method in (*METHODS, "uniform_bounded"):
        rr = [r for r in rows if r["role"] == "test" and r["method"] == method]
        a.stats(method + ":reported_test_statistics", report["test"][method], stats(rr))
    a.check("all_8_test_contexts_in_report_including_unsupported", Counter(c["context_id"] for c in report["test_contexts"])
        == Counter(c["id"] for c in settings["contexts"]["test"]))
    for context in report["test_contexts"]:
        for method in (*METHODS, "uniform_bounded"):
            for phase, claim in ((None, "total"), ("nominal", "nominal"), ("fresh", "fresh")):
                rr = [r for r in rows if r["role"] == "test" and r["method"] == method and r["context_id"] == context["context_id"]
                    and (phase is None or r["phase"] == phase)]
                a.stats(context["context_id"] + ":" + method + ":" + claim, context["methods"][method][claim], stats(rr))
    controls = [r for r in rows if r["role"] == "control"]
    anchor = spec["anchor_geometry"]
    a.check("three_exact_quarantined_anchor_controls", len(controls) == 3 and rows[:3] == controls
        and all(r["method"] == "control" and r["phase"] == "quarantined_anchor"
            and all(r[k] == anchor[k] for k in ("values", "q10", "q14", "bucket_xy_m")) for r in controls)
        and sum(bool(r["route_success"]) for r in controls) >= 2)
    a.check("geometry_only_historical_anchor_no_inherited_outcomes", set(anchor) == {"source_id", "source_namespace", "source_record_sha256",
        "values", "q10", "q14", "bucket_xy_m", "use"})
    options = report.get("effective_native_options", {})
    a.check("effective_native_options_recorded", options.get("timestep_s") == 1 / 15360 and options.get("solver") == 2
        and options.get("cone") == 0 and options.get("integrator") == 3 and options.get("iterations") == 100
        and np.isclose(options.get("tolerance", 0), 1e-8, rtol=0, atol=1e-12) and options.get("gravity") == [0., 0., -9.81]
        and options.get("substeps") == 64 and options.get("board_mode") == "articulated" and options.get("gravity_mode") == "passive_only"
        and read(study / "effective-options.json") == options)
    package_errors = []
    for name, version in spec["packages"].items():
        try:
            if importlib.metadata.version(name) != version:
                package_errors.append(name)
        except importlib.metadata.PackageNotFoundError:
            package_errors.append(name)
    a.check("actual_auditor_packages_match_pinned_execution", not package_errors
        and spec["packages"].get("mjlab") == "1.6.0" and spec["packages"].get("mujoco") == "3.11.0", package_errors)
    rescored = audit_batches(a, study, spec, rows)
    failures = [c for c in a.checks if not c["passed"]]
    return dict(version=VERSION, status="passed" if not failures else "failed", failures=failures,
        checks=a.checks, checks_count=len(a.checks), contract_fingerprint=fp,
        auditor_source_sha256=sha256(__file__), executed_source_directory=str(source),
        records=len(rows), independently_rescored_throws=rescored,
        limitations=["Saved evidence/provenance and CPU scoring audit; native ball dynamics are not rerun.",
            "Fixed bucket and board XY; held-out board-height/tilt contexts do not establish new-bucket generalization.",
            "Strict endpoints/held poses do not establish actuator trajectories or common-start connectivity."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.study, args.source, args.assets)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        result = dict(version=VERSION, status="failed", checks_count=0, checks=[],
            failures=[dict(name="malformed_or_missing_evidence", passed=False, detail=str(exc))])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks_count", "independently_rescored_throws")}))
    if result["failures"]:
        raise RuntimeError("Window evidence audit failed: " + str([c["name"] for c in result["failures"]]))


if __name__ == "__main__":
    main()
