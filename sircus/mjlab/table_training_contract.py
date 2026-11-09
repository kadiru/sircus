"""Explicit TRAIN-only v3 bootstrap contract; no native/GPU side effects.

Historical source, labels and models remain immutable. The original observed
actions are joined to the separately audited v3 labels, never relabeled by ID
alone. Evaluation coordinates are drawn without consulting flight or IK labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import policy64_contract as historical
from .benchmark_contract import observation, action_from_residual
from .corrected_contract import sha
from .learning_contract import digest
from .table_assisted_score import reward, VERSION as SCORING_VERSION, ROUTE_VERSION, REWARD_VERSION

VERSION = "sircus-table-assisted-training-v1"
AUDIT_VERSION = "sircus-table-assisted-training-audit-v1"
SEED = 20261511
VALIDATION_SEED = 20262203
TEST_SEED = 20262204
MILESTONES = (0, 256, 512)
METHODS = ("old_bc", "old_ppo", "bc", "ppo")
SCOPE = ("Fixed64 privileged-geometry held-endpoint shot selection with explicit v3 table-assisted rewards. "
         "7168 historical TRAIN requests bootstrap imitation and critic;512 fresh requested shots fine-tune PPO. "
         "Frozen policies receive one decision at each fresh TEST bucket. All setup rejections count. "
         "No feasible-action resampling, flight search, learned arm paths/grasp, noise,128 qualification, "
         "perception or physical validation. Unequal historical training exposure is disclosed.")


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def captured(row):
    return bool(row.get("native_release") and row.get("route_success")
                and not row.get("invalid") and not row.get("invalid_physics"))


def clean(row):
    return captured(row) and row.get("route_quality") == 2


def statistics(rows):
    native = sum(r.get("native_release") is True for r in rows)
    captures = sum(captured(r) for r in rows)
    return dict(attempts=len(rows), captures=captures, native_releases=native,
        invalid=sum(bool(r.get("invalid")) for r in rows), clean=sum(clean(r) for r in rows),
        mean_reward=float(np.mean([reward(r) for r in rows])) if rows else 0.,
        direct_captures=sum(captured(r) and r.get("direct_capture") is True for r in rows),
        table_assisted_captures=sum(captured(r) and r.get("table_assisted_capture") is True for r in rows),
        setup_rejections=len(rows)-native, native_misses=native-captures,
        forbidden_contact_shots=sum(bool(r.get("forbidden_contact_steps")) for r in rows),
        invalid_physics=sum(bool(r.get("invalid_physics")) for r in rows))


def protocol():
    return dict(substeps=64, seed=SEED, validation_seed=VALIDATION_SEED, test_seed=TEST_SEED,
        train_layouts=35, validation_layouts=12, test_layouts=50, train_requests=512,
        batch_size=32, updates=16, milestones=list(MILESTONES), test_methods=list(METHODS),
        test_repeats=1, validation_requests=36, test_requests=200, control_requests=18,
        maximum_fresh_requests=766, noise=False, acquired_slot_limit_s=7200,
        actor_steps=800, critic_steps=400, bootstrap_lr=.001, ppo_lr=.0003,
        initial_logstd=-3., ppo_epochs=6, clip=.2, entropy=.001, gradient_norm=1.,
        minimum_split_distance_m=.005, scoring_version=SCORING_VERSION,
        route_version=ROUTE_VERSION, reward_version=REWARD_VERSION,
        selection="Validation captures, then mean v3 reward, then earliest milestone; freeze before TEST",
        source_selection="One captured TRAIN teacher per context: clean, v3 reward, stable ID; no incompatible-action averaging",
        control_roles=["direct_control", "table_control", "negative_control"], scope=SCOPE)


def schedule():
    rng, result = np.random.default_rng(SEED), []
    while len(result) < 512:
        result.extend(rng.permutation(35).tolist())
    return result[:512]


def select_teachers(rows):
    groups = {}
    for row in rows:
        require(row.get("role") == "train", "Only TRAIN may supply imitation teachers")
        if captured(row):
            groups.setdefault(row["context_id"], []).append(row)
    return [min(groups[key], key=lambda r: (-int(clean(r)), -reward(r), r["id"]))
            for key in sorted(groups)]


def contained(root, name):
    root, name = Path(root).resolve(), Path(name)
    result = (root/name).resolve()
    require(not name.is_absolute() and ".." not in name.parts and result.is_relative_to(root)
            and result.is_file(), "Bound relative evidence file required")
    return result


def _policy_parent(study, audit, repository):
    study, audit = Path(study), Path(audit)
    public = read(Path(repository)/"baselines/mjlab-policy64-20261105/summary.json")["provenance"]
    for path, field in ((study/"report.json", "report_sha256"), (study/"contract.json", "contract_sha256"),
                        (study/"evidence-manifest.json", "evidence_manifest_sha256"),
                        (audit, "independent_audit_sha256")):
        require(sha(path) == public[field], "Published policy64 binding differs: "+path.name)
    spec, report, proof, manifest = map(read, (study/"contract.json", study/"report.json", audit, study/"evidence-manifest.json"))
    unsigned = dict(spec); fp = unsigned.pop("fingerprint")
    require(digest(unsigned) == fp == public["contract_fingerprint"] and report["contract_fingerprint"] == fp,
            "Frozen policy64 contract fingerprint differs")
    require(report["status"] == "completed" and proof["status"] == "passed" and proof["failures"] == []
            and proof["checks_count"] == 346485 and len(proof["checks"]) == 346485
            and all(c.get("passed") is True for c in proof["checks"])
            and proof["report_sha256"] == sha(study/"report.json")
            and proof["evidence_manifest_sha256"] == sha(study/"evidence-manifest.json")
            and report["evidence_sha256"] == manifest, "Complete published policy64 audit required")
    needed = ["rows.jsonl", "prior.json", "bootstrap/teachers.json"]
    artifacts = {}
    for method, key in (("old_bc", "bc"), ("old_ppo", "selected")):
        item = copy.deepcopy(report["artifacts"][key]); needed.append(item["path"])
        require(item["sha256"] == manifest[item["path"]], "Old model/report manifest differs")
        item["relative_path"] = item["path"]; item["path"] = str(contained(study, item["path"]))
        artifacts[method] = item
    for name in needed:
        require(sha(contained(study, name)) == manifest[name] == public["evidence_sha256"][name],
                "Authenticated policy64 evidence changed: "+name)
    return spec, report, manifest, artifacts, public


def _rescore(rescore_dir, repository):
    directory = Path(rescore_dir)
    public = read(Path(repository)/"baselines/mjlab-table-assisted-scoring-20261108/results.json")
    for name, expected in (("summary.json", public["summary_sha256"]), ("manifest.json", public["manifest_sha256"]),
                           ("audit.json", public["audit"]["proof_sha256"])):
        require(sha(directory/name) == expected, "Published v3 rescore evidence differs: "+name)
    audit, manifest = read(directory/"audit.json"), read(directory/"manifest.json")
    require(audit["status"] == "passed" and audit["full_checks_verified"] is True and audit["failures"] == []
            and audit["checks_count"] == 64883 and len(audit["checks"]) == 64883
            and all(x["passed"] is True for x in audit["checks"]), "Complete v3 derivative audit required")
    for name, expected in manifest["artifacts"].items():
        require(sha(contained(directory, name)) == expected, "Derivative artifact changed: "+name)
    rows = {}
    with (directory/"records.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["source_study"] not in ("calibrated-benchmark-v2", "policy64-v1") or row["phase"] != "train":
                continue
            key = (row["source_study"], row["id"])
            require(key not in rows, "Duplicate v3 TRAIN evidence identity")
            rows[key] = row
    return rows, dict(rescore_summary_sha256=sha(directory/"summary.json"),
        rescore_manifest_sha256=sha(directory/"manifest.json"), rescore_audit_sha256=sha(directory/"audit.json"),
        rescore_records_sha256=sha(directory/"records.jsonl"))


def join_training_row(raw, derived, study, train_coordinates):
    """Exact native identity and geometric action binding precede label adoption."""
    require(raw.get("role") == "train" and raw.get("context_id") in train_coordinates
            and raw["bucket_xy_m"] == train_coordinates[raw["context_id"]]
            and raw.get("perturbation_mixed_units") == [0.]*5, "Source is not exact nominal TRAIN")
    require(derived["source_study"] == study and derived["phase"] == "train" and derived["id"] == raw["id"],
            "TRAIN derivative source association differs")
    original = derived["original"]
    for key in ("id", "bucket_xy_m", "values", "native_release"):
        require(original.get(key) == raw.get(key), "Derivative original action differs: "+key)
    if raw.get("native_release"):
        require(derived["status"] == "scored" and raw.get("contact_protocol", {}).get("substeps") == 64
                and raw.get("scoring_version") == "sircus-calibrated-native-capture-v2"
                and raw.get("scalar_score_audit_passed") is True, "Compatible audited native64 TRAIN required")
        for key in ("native_cache_sha256", "native_cache_row", "calibration_fingerprint"):
            require(raw.get(key) is not None and raw[key] == original.get(key), "Native evidence association differs: "+key)
        for key in ("route_success", "success", "invalid_physics", "board_bounce", "whole_ball_entry"):
            require(raw[key] == derived["old_score"][key], "Derivative historical score differs: "+key)
        labels = copy.deepcopy(derived["new_score"])
    else:
        require(derived["status"] == "rejected" and raw.get("invalid")
                and original.get("invalid") == raw["invalid"], "Rejected source must remain rejected")
        labels = dict(invalid=raw["invalid"], invalid_physics=False, native_release=False,
                      route_success=False, route_quality=0, direct_capture=False, table_assisted_capture=False)
    anchor = raw["effective_source_hint"]
    obs = np.asarray(raw["observation"], float); residual = np.asarray(raw["normalized_residual"], float)
    require(obs.shape == (9,) and residual.shape == (5,) and np.isfinite(obs).all()
            and np.isfinite(residual).all() and (np.abs(residual) <= 1).all(), "Finite nine-feature/five-action source required")
    require(np.allclose(obs, observation(raw["bucket_xy_m"], anchor), atol=1e-7, rtol=0)
            and np.allclose(raw["values"], action_from_residual(anchor["values"], residual), atol=1e-10, rtol=0),
            "Source observation/action no longer reconstructs from effective anchor")
    keys = ("id", "role", "context_id", "method", "seed", "train_index", "bucket_xy_m", "values",
            "observation", "normalized_residual", "effective_source_hint", "source_hint", "source_teacher_id",
            "native_release", "native_cache_sha256", "native_cache_row", "calibration_fingerprint", "invalid", "invalid_physics")
    row = {k: copy.deepcopy(raw[k]) for k in keys if k in raw}
    row.update(labels, source_study=study, source_record_sha256=digest(raw),
               original_route_success=bool(raw.get("route_success")),
               newly_credited_capture=bool(labels["route_success"] and not raw.get("route_success")))
    row["learning_reward"] = reward(row)
    return row


def load_inputs(parent_study, parent_audit, policy_study, policy_audit, rescore_dir, repository):
    parent_study, policy_study, repository = map(Path, (parent_study, policy_study, repository))
    parent, prior, _, old_teachers, _, parent_bindings, _ = historical.load_parent(parent_study, parent_audit, repository)
    policy_parent, policy_report, policy_manifest, old_artifacts, policy_public = _policy_parent(policy_study, policy_audit, repository)
    require(parent["splits"]["train"] == policy_parent["splits"]["train"]
            or {c["id"]:c["bucket_xy_m"] for c in parent["splits"]["train"]}
               == {c["id"]:c["bucket_xy_m"] for c in policy_parent["splits"]["train"]}, "Historical TRAIN coordinates differ")
    require(digest(read(policy_study/"prior.json")) == digest(prior)
            and parent["identity"]["scene_sha256"] == policy_parent["identity"]["scene_sha256"]
            and parent["identity"]["calibration"] == policy_parent["identity"]["calibration"], "Parent geometry/calibration/prior differs")
    derived, rescore_bindings = _rescore(rescore_dir, repository)
    train = {c["id"]: c["bucket_xy_m"] for c in parent["splits"]["train"]}
    sources = [(parent_study/f"runs/{method}-seed{seed}/rows.jsonl", "calibrated-benchmark-v2", method, seed)
               for method in ("ppo", "sweep") for seed in (20261411,20261412,20261413)]
    sources.append((policy_study/"rows.jsonl", "policy64-v1", "ppo", SEED))
    dataset, source_hashes = [], {}
    for path, study, method, seed in sources:
        source_hashes[f"{study}/{path.relative_to(parent_study if study == 'calibrated-benchmark-v2' else policy_study)}"] = sha(path)
        block = []
        for line in path.open():
            raw = json.loads(line)
            if raw.get("role") != "train":
                continue
            require(raw.get("method") == method and raw.get("seed") == seed, "TRAIN method/seed differs")
            key = (study, raw["id"])
            require(key in derived, "TRAIN row missing from authenticated v3 derivative")
            block.append(join_training_row(raw, derived[key], study, train))
        require(len(block) == 1024 and sorted(r["train_index"] for r in block) == list(range(1024)), "Complete TRAIN block required")
        dataset.extend(sorted(block, key=lambda r:r["train_index"]))
    require(len(dataset) == 7168 and len({r["id"] for r in dataset}) == 7168, "Exactly7168 unique TRAIN requests required")
    require(sum(r["newly_credited_capture"] for r in dataset) == 115, "Expected audited115 newly credited TRAIN captures")
    teachers = select_teachers(dataset)
    old_map = {r["context_id"]:r["id"] for r in old_teachers}
    changed = [dict(context_id=r["context_id"], old_id=old_map.get(r["context_id"]), new_id=r["id"],
                    table_assisted=bool(r.get("table_assisted_capture"))) for r in teachers if old_map.get(r["context_id"]) != r["id"]]
    summary = dict(requests=len(dataset), native_releases=sum(bool(r["native_release"]) for r in dataset),
        captures=sum(captured(r) for r in dataset), clean=sum(clean(r) for r in dataset), layouts=len(train),
        successful_layouts=len(teachers), teacher_count=len(teachers), newly_credited_training_count=115,
        table_teacher_count=sum(bool(r.get("table_assisted_capture")) for r in teachers),
        table_assisted_teachers=sum(bool(r.get("table_assisted_capture")) for r in teachers),
        newly_credited_captures=115, changed_teacher_count=len(changed),
        newly_credited_teacher_count=sum(r["newly_credited_capture"] for r in teachers), teacher_changes=changed,
        teacher_change_count=len(changed), original_teacher_count=len(old_teachers),
        labels_scope="TRAIN only; all7168 outcomes inform critic, one successful teacher/context informs actor.115new credits are not115actor teachers.")
    direct = min((r for r in dataset if captured(r) and r.get("direct_capture")), key=lambda r:(-int(clean(r)), -reward(r), r["id"]))
    table = min((r for r in dataset if captured(r) and r.get("table_assisted_capture")), key=lambda r:r["id"])
    miss = min((r for r in dataset if r.get("native_release") and not captured(r)
                and not r.get("invalid") and not r.get("invalid_physics") and not r.get("whole_ball_entry")), key=lambda r:r["id"])
    controls = [copy.deepcopy(r) for r in (direct, table, miss)]
    bindings = dict(parent=parent_bindings, policy_parent_contract_fingerprint=policy_parent["fingerprint"],
        policy_parent_report_sha256=sha(policy_study/"report.json"), policy_parent_audit_sha256=sha(policy_audit),
        policy_parent_manifest_sha256=sha(policy_study/"evidence-manifest.json"), source_files_sha256=source_hashes,
        source_train_record_sha256={r["id"]:r["source_record_sha256"] for r in dataset},
        teacher_ids=[r["id"] for r in teachers], old_artifacts=old_artifacts, **rescore_bindings)
    return dict(parent=parent, policy_parent=policy_parent, prior=prior, dataset=dataset, teachers=teachers,
        bootstrap_summary=summary, bindings=bindings, controls=controls, old_artifacts=old_artifacts,
        calibration=parent["identity"]["calibration"], home=parent["identity"]["setup"]["home"])


def _coordinates(value, result, *, coordinate_list=False):
    if coordinate_list and isinstance(value, (list, tuple)):
        a = np.asarray(value, dtype=object)
        if a.shape == (2,) and all(isinstance(x, (int,float)) and not isinstance(x,bool) and np.isfinite(x) for x in value):
            result.add(tuple(map(float, value))); return
    if isinstance(value, dict):
        for key, item in value.items():
            _coordinates(item, result, coordinate_list=key in ("bucket_xy_m", "bucket_xy", "exclusions", "excluded", "coordinates", "excluded_coordinates"))
    elif isinstance(value, list):
        for item in value:
            _coordinates(item, result, coordinate_list=coordinate_list)


def collect_exclusions(output_root, repository):
    """Read only coordinates from known metadata; no outcomes determine splits.

    Hash every scanned metadata file. Unknown unpublished/absent files cannot be
    certified, so this is explicitly novelty against the inventoried evidence.
    """
    coordinates, files = set(), {}
    names = {"contract.json", "splits.json", "exclusions.json", "report.json", "summary.json", "results.json",
             "records.jsonl", "rows.jsonl", "shots.jsonl", "requests.jsonl", "test-plan.json", "plan.json"}
    roots = (("output", Path(output_root)), ("baselines", Path(repository)/"baselines"))
    for label, root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and (p.name in names or (label == "baselines" and p.suffix == ".json"))):
            h = hashlib.sha256()
            if path.suffix == ".jsonl":
                with path.open("rb") as stream:
                    for line in stream:
                        h.update(line)
                        if line.strip():
                            _coordinates(json.loads(line), coordinates)
            else:
                data = path.read_bytes(); h.update(data)
                _coordinates(json.loads(data), coordinates, coordinate_list="exclusion" in path.name)
            files[label+"/"+str(path.relative_to(root))] = h.hexdigest()
    require(coordinates, "Historical coordinate inventory is empty")
    return dict(coordinates=[list(x) for x in sorted(coordinates)], files_sha256=files,
        scope="All bucket coordinates discovered in inventoried metadata and published baselines, all roles/outcomes; no IK/capture filtering or unpublished-history guarantee")


def make_splits(train_splits, excluded):
    train_rows = train_splits["train"] if isinstance(train_splits, dict) else train_splits
    train = [dict(id=c["id"], split="train", bucket_xy_m=copy.deepcopy(c["bucket_xy_m"])) for c in train_rows]
    require(len(train) == 35 and len({r["id"] for r in train}) == 35, "Original35 TRAIN contexts required")
    points = {tuple(map(float, x)) for x in excluded}
    points.update(tuple(c["bucket_xy_m"]) for c in train)
    exclusions = [list(x) for x in sorted(points)]
    used = np.asarray(exclusions, float)
    require(used.ndim == 2 and used.shape[1] == 2 and np.isfinite(used).all(), "Finite historical exclusion coordinates required")
    result = dict(train=train)
    for role, count, seed in (("validation",12,VALIDATION_SEED), ("test",50,TEST_SEED)):
        rng, rows = np.random.default_rng(seed), []
        for _ in range(100000):
            if len(rows) == count:
                break
            xy = rng.uniform([.1,-.65], [1.24,.25])
            if np.linalg.norm(used-xy, axis=1).min() < .005:
                continue
            rows.append(dict(id=f"table-training-{role}-{len(rows):02d}", split=role, bucket_xy_m=xy.tolist()))
            used = np.vstack([used,xy])
        require(len(rows) == count, "Unable to freeze fresh unfiltered layouts")
        result[role] = rows
    return result, exclusions
