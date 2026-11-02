"""Deterministic-IK audit and native development search; no policy training.

Former unresolved test coordinates remain development.  Source joint branches
guide a new audited IK solve; only fresh native flights and separately seeded
confirmations can qualify a new shot.  Original sources and scores are untouched.
"""

import argparse
import copy
import datetime
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from .expand_bank import atomic, qualified_template, sha256
from .expanded_actions import WIDE_BOUNDS
from .expand_far import far_proposals
from .learning_contract import digest, statistics

VERSION = "sircus-stable-ik-ten-target-study-v1"
SEED = 20261120
NOISE = [0.001, 0.001, 0.002, 0.35, 0.002]
EXPECTED = (8, 9, 10, 12, 21, 23, 26, 35, 39, 41)
CONTROL_INDICES = (10, 19, 21, 26, 39, 48)
INVARIANCE_TOLERANCE_RAD = 1e-9


def targets_from_cache(targets, bucket, mask):
    bucket, mask = np.asarray(bucket), np.asarray(mask, dtype=bool)
    if bucket.shape != (12, 2) or mask.ndim != 2 or mask.shape[0] != 12:
        raise ValueError("Expected completed twelve-coordinate development cache")
    if len(targets) != 12 or not np.array_equal(
        bucket, np.asarray([t["bucket_xy_m"] for t in targets])
    ):
        raise ValueError("Development coordinate/cache identity changed")
    selected = [copy.deepcopy(t) for t, m in zip(targets, mask) if not m.any()]
    if tuple(t["original_test_index"] for t in selected) != EXPECTED:
        raise ValueError("Expected precisely the ten unresolved v2 development positions")
    return selected


def acquisition(row):
    """Proposal acquisition, never an alternate capture definition.

    Native route captures dominate.  An unassisted partial retention dominates
    table-assisted entry; for assisted misses, only the measured airborne path
    before external contact informs proximity.  Prior assisted rows without that
    timing receive only a weak coarse proximity tie-break, no entry bonus.
    """
    if row.get("invalid") or not row.get("board_bounce"):
        return -100.0
    if row.get("route_success"):
        return 100.0 + 2 * (row.get("route_quality") == 2) + min(
            float(row.get("minimum_mouth_clearance_mm") or 0), 50
        ) / 50
    distance = float(row.get("closest_opening_distance_m", 2))
    if not np.isfinite(distance):
        return -100.0
    if not row.get("external_contact_steps"):
        return float(
            10
            + 5 * min(float(row.get("max_retention_s", 0)), 1)
            + 3 * bool(row.get("whole_ball_entry"))
            + 2 * np.exp(-max(distance, 0) / 0.12)
        )
    air = row.get("airborne_closest_opening_distance_m")
    if air is None:
        return float(0.05 * np.exp(-max(distance, 0) / 0.12))
    return float(
        6 * bool(row.get("bucket_contact_before_external"))
        + 2 * np.exp(-max(float(air), 0) / 0.12)
        - 0.25 * (row.get("contact_peaks_N", {}).get("robot", 0) > 0.01)
    )


def diverse_sources(rows, target_id, limit=12):
    if not 1 <= limit <= 32:
        raise ValueError("Finite seed count1..32 required")
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    pool, selected, seen = [], [], set()
    for row in sorted(rows, key=lambda r: (-acquisition(r), r["id"])):
        if row.get("target_id") != target_id or row.get("invalid"):
            continue
        if not row.get("board_bounce") or not all(k in row for k in ("q10", "q14")):
            continue
        value = np.asarray(row.get("values"), dtype=float)
        if value.shape != (5,) or not np.isfinite(value).all():
            continue
        if ((value < low - 1e-9) | (value > high + 1e-9)).any():
            continue
        key = tuple(np.round(value, 10))
        if key not in seen:
            seen.add(key)
            pool.append(row)
    while pool and len(selected) < limit:
        if not selected:
            index = 0
        else:
            prior = np.asarray([s["values"] for s in selected])
            score = [
                acquisition(s)
                + 0.2 * np.linalg.norm((prior - s["values"]) / (high - low), axis=1).min()
                for s in pool
            ]
            index = int(np.argmax(score))
        selected.append(pool.pop(index))
    return selected


def proposals(target, count, rng, scene, seeds, round_index):
    """Global full-domain coverage plus tightly branch-seeded continuation."""
    values, metadata = far_proposals(
        target["bucket_xy_m"], count, rng, scene, (), round_index=round_index
    )
    hints10 = np.full((count, 6), np.nan)
    hints14 = np.full((count, 6), np.nan)
    known_values = np.full((count, 5), np.nan)
    known_bucket = np.full((count, 2), np.nan)
    scales = np.asarray(
        [[.001, .001, .002, .25, .002], [.006, .006, .010, 1., .010],
         [.025, .030, .040, 4., .045]]
    )
    for i in range(count):
        if not seeds or (i % 5 >= 3 and not (round_index == 0 and i < len(seeds))):
            continue
        source = seeds[i % len(seeds)] if round_index == 0 else seeds[int(rng.integers(len(seeds)))]
        value = np.asarray(source["values"])
        exact = round_index == 0 and i < len(seeds)
        if not exact:
            value = value + rng.normal(0, scales[int(rng.integers(len(scales)))])
        values[i] = np.clip(value, WIDE_BOUNDS["low"], WIDE_BOUNDS["high"])
        hints10[i], hints14[i] = source["q10"], source["q14"]
        known_values[i], known_bucket[i] = source["values"], source["bucket_xy_m"]
        metadata[i] = dict(
            proposal_family="known-branch-exact" if exact else "known-branch-continuation",
            seed_source_id=source["id"], seed_source_record_sha256=digest(source),
            round_index=round_index,
        )
    return values, hints10, hints14, known_values, known_bucket, metadata


def flight_summary(forces, position, row, scene):
    """Read-only240Hz timing/proximity diagnostic, not terminal scoring."""
    count = min(len(forces), int(row.get("physics_steps", len(forces))))
    f, p = np.asarray(forces)[:count], np.asarray(position)[:count]
    names = ("board", "bucket", "table", "floor", "robot", "unattributed")
    first = {}
    for j, name in enumerate(names):
        ids = np.flatnonzero(f[:, j] > .01)
        first[name] = int(ids[0]) if len(ids) else None
    extern = [first[k] for k in names[2:] if first[k] is not None]
    cutoff = min(extern) if extern else count
    start = first["board"]
    distance = None
    if start is not None and cutoff > start:
        top = scene["bucket"]["assumed_interior_depth_m"] + scene["bucket"]["assumed_bottom_thickness_m"]
        center = np.asarray([*row["bucket_xy_m"], top - scene["ball"]["radius_m"]])
        distance = float(np.linalg.norm(p[start:cutoff] - center, axis=1).min())
    return dict(
        first_contact_s={name: None if i is None else (i + 1) / 240 for name, i in first.items()},
        first_external_contact_s=None if not extern else (cutoff + 1) / 240,
        airborne_closest_opening_distance_m=distance,
        bucket_contact_before_external=first["bucket"] is not None and first["bucket"] < cutoff,
        timing_scope="Observed force maxima and ballXYZ at240Hz, clipped to original scorer terminal; acquisition diagnostics only",
    )


def compare_arrays(reference, other):
    accepted = reference["mask"] & other["mask"]
    delta = max(
        (float(np.abs(reference[k][accepted] - other[k][accepted]).max())
         if accepted.any() else 0.0) for k in ("q10", "q14")
    )
    identity_fields = ("branch10", "branch14", "query10_sha256", "query14_sha256")
    identity_changes = sum(int((reference[k] != other[k]).sum()) for k in identity_fields if k in reference and k in other)
    return dict(
        mask_changes=int((reference["mask"] != other["mask"]).sum()),
        reason_changes=int((reference["reasons"] != other["reasons"]).sum()),
        max_q_delta_rad=delta,
        branch_or_query_identity_changes=identity_changes,
        passed=np.array_equal(reference["mask"], other["mask"])
        and np.array_equal(reference["reasons"], other["reasons"])
        and delta <= INVARIANCE_TOLERANCE_RAD and identity_changes == 0,
    )


def _arrays(mask, detail):
    keys = ("q10", "q14", "reasons", "branch10", "branch14", "query10_sha256", "query14_sha256", "known_pair_preserved", "known_request_match", "preserved_exact_nominal", "refined_continuation")
    return dict(mask=mask[:, 0], **{k: detail[k][:, 0] for k in keys if k in detail})


def solver_audit(scene, assets, queries, output):
    from .expand_geometry import audit_values as old_audit
    from .stable_geometry import audit_values as stable_audit

    xy = np.asarray([q["bucket_xy_m"] for q in queries])
    values = np.asarray([q["values"] for q in queries])
    known10 = np.asarray([q["q10"] for q in queries])
    known14 = np.asarray([q["q14"] for q in queries])
    order = np.arange(len(queries))[::-1]
    cache = dict(bucket=xy, values=values, known_values=values, known_bucket_xy=xy, known_q10=known10, known_q14=known14, reverse_order=order)
    comparisons, passes = [], []
    for method in ("old", "stable_matched", "stable_free", "stable_seeded"):
        function = old_audit if method == "old" else stable_audit
        kwargs = dict(bounds=WIDE_BOUNDS, starts=4 if method in ("old", "stable_matched") else 8, iterations=90 if method in ("old", "stable_matched") else 120)
        if method == "stable_seeded":
            kwargs.update(known_q10=known10[:, None], known_q14=known14[:, None], known_values=values[:, None], known_bucket_xy=xy[:, None])
        result = function(scene, assets, xy, values[:, None], **kwargs)
        direct = _arrays(*result)
        for key, value in direct.items():
            cache[method + "_" + key] = value
        reverse_kwargs = dict(bounds=WIDE_BOUNDS, starts=4 if method in ("old", "stable_matched") else 8, iterations=90 if method in ("old", "stable_matched") else 120)
        if method == "stable_seeded":
            reverse_kwargs.update(known_q10=known10[order, None], known_q14=known14[order, None], known_values=values[order, None], known_bucket_xy=xy[order, None])
        reverse = _arrays(*function(scene, assets, xy[order], values[order, None], **reverse_kwargs))
        partition = {key: [] for key in direct}
        for i in range(len(xy)):
            single_kwargs = dict(bounds=WIDE_BOUNDS, starts=4 if method in ("old", "stable_matched") else 8, iterations=90 if method in ("old", "stable_matched") else 120)
            if method == "stable_seeded":
                single_kwargs.update(known_q10=known10[i:i+1, None], known_q14=known14[i:i+1, None], known_values=values[i:i+1, None], known_bucket_xy=xy[i:i+1, None])
            one = _arrays(*function(scene, assets, xy[i:i+1], values[i:i+1, None], **single_kwargs))
            for key in direct:
                partition[key].append(one[key])
        partition = {key: np.concatenate(items) for key, items in partition.items()}
        for key in direct:
            cache[method + "_reverse_" + key] = reverse[key]
            cache[method + "_partition_" + key] = partition[key]
        reverse_normalized = {key: value[np.argsort(order)] for key, value in reverse.items()}
        a, b = compare_arrays(direct, reverse_normalized), compare_arrays(direct, partition)
        preserved = None
        if method == "stable_seeded":
            preserved = np.asarray(result[1]["preserved_exact_nominal"][:, 0], dtype=bool)
            cache["seeded_preserved"] = preserved
            passes.append(bool(preserved.all()))
        comparisons.append(dict(method=method, queries=len(xy), accepted=int(direct["mask"].sum()),
            order_mask_changes=a["mask_changes"], batch_mask_changes=b["mask_changes"],
            max_q_delta_rad=max(a["max_q_delta_rad"], b["max_q_delta_rad"]),
            preserved_known_branches=None if preserved is None else int(preserved.sum())))
        if method != "old":
            passes.extend([a["passed"], b["passed"]])
    np.savez_compressed(output / "solver-audit.npz", **cache)
    return dict(
        comparison=comparisons,
        order_batch_invariance=dict(passed=all(passes), tolerance_rad=INVARIANCE_TOLERANCE_RAD,
            variants="Full batch, reverse order and individual-query batches; raw cache arrays retained"),
        known_branch_preservation=dict(passed=bool(cache["seeded_preserved"].all()),
            known_valid_queries=len(xy), preserved=int(cache["seeded_preserved"].sum())),
        no_training=True,
    ), cache


def load_inputs(a):
    from .learning_solve import validate_artifacts

    _, cp, parent_sha, parent_fp = validate_artifacts(a.parent_study, a.assets)
    prior_report = json.loads((a.prior_study / "report.json").read_text())
    prior_contract = json.loads((a.prior_study / "contract.json").read_text())
    if prior_report.get("status") != "completed" or prior_contract["scene"] != cp["contract"]["scene"]:
        raise RuntimeError("Completed matching v2 study required")
    prior_payload=dict(prior_contract);prior_fp=prior_payload.pop("fingerprint")
    if digest(prior_payload)!=prior_fp or prior_report["contract_fingerprint"]!=prior_fp or prior_contract["parent_checkpoint_sha256"]!=parent_sha or prior_contract["parent_contract_fingerprint"]!=parent_fp:
        raise RuntimeError("Prior study fingerprint/parent identity changed")
    geometry = a.prior_study / "geometry-expanded-development.npz"
    if prior_report["artifacts_sha256"][geometry.name] != sha256(geometry):
        raise RuntimeError("Prior development endpoint cache changed")
    expansion_contract = json.loads((a.prior_expansion / "contract.json").read_text())
    expansion_report=json.loads((a.prior_expansion/"report.json").read_text())
    expansion_payload=dict(expansion_contract);expansion_fp=expansion_payload.pop("fingerprint")
    if digest(expansion_payload)!=expansion_fp or expansion_report["contract_fingerprint"]!=expansion_fp or expansion_report.get("status")!="completed":
        raise RuntimeError("Prior expansion contract identity changed")
    full_targets = expansion_contract["development_layouts"]
    with np.load(geometry, allow_pickle=False) as cached:
        targets = targets_from_cache(full_targets, cached["bucket"], cached["mask"])
    bank_path = a.prior_study / "proposal-bank.json"
    bank = json.loads(bank_path.read_text())
    p = dict(bank); recorded = p.pop("sha256")
    if digest(p) != recorded or prior_report["proposal_bank_sha256"] != recorded:
        raise RuntimeError("Preserved expanded bank identity changed")
    if sha256(bank_path) != prior_report["artifacts_sha256"]["proposal-bank.json"] or sha256(a.prior_expansion/"expanded-bank.json") != prior_contract["expanded_bank_artifact_sha256"]:
        raise RuntimeError("Preserved expanded bank file changed")
    origins, rows = [], []
    for study in a.source_study:
        report = json.loads((study / "report.json").read_text())
        contract = json.loads((study / "contract.json").read_text())
        if report.get("status") != "completed" or contract["scene"] != cp["contract"]["scene"]:
            raise RuntimeError("Only completed matching native development sources allowed")
        payload=dict(contract);fp=payload.pop("fingerprint")
        if digest(payload)!=fp or report["contract_fingerprint"]!=fp:
            raise RuntimeError("Prior native source contract identity changed")
        origin = dict(directory=str(study.resolve()), report_sha256=sha256(study / "report.json"),
            contract_sha256=sha256(study / "contract.json"), shots_sha256=sha256(study / "shots.jsonl"),
            contract_fingerprint=contract["fingerprint"])
        origins.append(origin)
        for line in (study / "shots.jsonl").open():
            row = json.loads(line)
            if row.get("phase") != "search" or row.get("invalid"):
                continue
            if row.get("target_id") not in {t["id"] for t in full_targets}:
                continue
            item = copy.deepcopy(row)
            item["prior_record_id"] = row["id"]
            item["id"] = study.name + "/" + row["id"]
            item["prior_origin"] = dict(study_directory=origin["directory"], record_sha256=digest(row),
                records_sha256=origin["shots_sha256"])
            rows.append(item)
    controls = []
    for index in CONTROL_INDICES:
        pool = [r for r in rows if r["target_id"] == f"old-test-{index:02d}" and r.get("route_success")]
        if not pool:
            raise RuntimeError("Missing actual captured control source")
        controls.append(max(pool, key=lambda r: (float(r.get("minimum_mouth_clearance_mm") or -1e6), acquisition(r))))
    seeds = {t["id"]: diverse_sources(rows, t["id"], a.seed_sources) for t in targets}
    if any(not values for values in seeds.values()):
        raise RuntimeError("Each target needs a measured geometry seed for this study")
    used = {r["id"]:r for r in controls + [s for values in seeds.values() for s in values]}
    return cp["contract"]["scene"], targets, full_targets, bank, list(used.values()), controls, seeds, dict(
        parent_study_directory=str(a.parent_study.resolve()), parent_checkpoint_sha256=parent_sha, parent_contract_fingerprint=parent_fp,
        prior_study_directory=str(a.prior_study.resolve()), prior_study_report_sha256=sha256(a.prior_study/"report.json"),
        prior_study_contract_sha256=sha256(a.prior_study/"contract.json"), prior_geometry_file=geometry.name, prior_geometry_sha256=sha256(geometry),
        base_bank_file="proposal-bank.json", base_bank_file_sha256=sha256(bank_path),
        prior_expansion_directory=str(a.prior_expansion.resolve()), prior_expansion_bank_sha256=sha256(a.prior_expansion/"expanded-bank.json"),
        base_bank_sha256=bank["sha256"], source_origins=origins,
    )


def run(a):
    scene, targets, full_targets, bank, prior_sources, controls, initial_seeds, provenance = load_inputs(a)
    from .stable_geometry import VERSION as GEOMETRY_VERSION
    from .stable_ik import VERSION as IK_VERSION

    spec = dict(version=VERSION, seed=SEED, scene=scene, **provenance,
        action_bounds=WIDE_BOUNDS, development_layouts=targets, original_development_layouts=full_targets,
        control_source_ids=[r["id"] for r in controls], selected_prior_source_ids=[r["id"] for r in prior_sources],
        selected_prior_source_hashes=[digest(r) for r in prior_sources],
        native_per_target=a.native_per_target, proposal_rounds=a.proposal_rounds,
        proposals_per_target_round=a.proposals, seed_sources_per_target=a.seed_sources,
        top_candidates=a.top, confirmations_per_candidate=10, final_confirmations_per_candidate=20,
        minimum_confirmation_rate=.8, confirmation_noise_std=NOISE,
        control_repeats=a.control_repeats, batch_size=a.batch, gpu_wait_timeout_s=a.gpu_wait_timeout,
        ik_version=IK_VERSION, geometry_version=GEOMETRY_VERSION, no_training=True,
        source_role="All ten previous empty-mask positions remain development; no new test/generalization claim",
        acquisition="Direct route capture first; direct partial retention next; airborne opening proximity before first external contact guides assisted misses, never their final capture label",
        confirmation_protocol="Native captured source plus>=8/10 independent small perturbations; freeze highest-qualified candidate; require>=16/20 separately seeded final perturbations before bank addition; CPU rejects remain failures",
        physics="Original native1920Hz integration/240Hz scorer and all contact/pose gates unchanged; prescribed release and held arm endpoints",
        scope="IK determinism/branch audit and finite development feasibility search, no PPO, no model training, no hardware or exhaustive coverage",
    )
    spec["fingerprint"] = digest(spec)
    a.output.mkdir(parents=True, exist_ok=False)
    atomic(a.output/"contract.json",spec)
    with (a.output/"prior-sources.jsonl").open("w") as stream:
        for source in prior_sources:stream.write(json.dumps(source,allow_nan=False)+"\n")
    (a.output/"shots.jsonl").write_text("")
    query_sources = controls + [initial_seeds[t["id"]][0] for t in targets]
    queries = [dict(source_id=s["id"], origin=s["prior_origin"], known_valid=True,
        target_id=s["target_id"], bucket_xy_m=s["bucket_xy_m"], values=s["values"],
        q10=s["q10"],q14=s["q14"]) for s in query_sources]
    atomic(a.output/"audit-queries.json",queries)
    report = dict(status="running",phase="cpu-ik-audit",contract_fingerprint=spec["fingerprint"],
        development_layouts=targets,no_training=True,native_search_attempts=0,
        confirmation_attempts=0,final_confirmation_attempts=0,control_attempts=0,
        geometry_proposals=0,cells=[],additions=0,solver_audit=None)
    records, searched, confirm, final, promoted = [], {t["id"]:[] for t in targets}, {}, {}, {}
    started = time.monotonic();cache_index=0

    def publish(phase):
        report.update(phase=phase,elapsed_s=time.monotonic()-started,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=len(records),additions=len(promoted))
        report["cells"]=[dict(t,search=statistics(searched[t["id"]]),
            observed_assisted_entries=sum(bool(r.get("whole_ball_entry")) and bool(r.get("external_contact_steps")) for r in searched[t["id"]]),
            promotion=promoted.get(t["id"]),unresolved_reason=None if t["id"] in promoted else "No independently final-confirmed route source in this finite study") for t in targets]
        atomic(a.output/"report.json",report)
        print(json.dumps({k:report[k] for k in ("phase","native_search_attempts","confirmation_attempts","final_confirmation_attempts","additions","elapsed_s")}),flush=True)

    publish("cpu-ik-audit")
    report["solver_audit"], audit_cache = solver_audit(scene,a.assets,queries,a.output)
    if not report["solver_audit"]["order_batch_invariance"]["passed"]:
        report["status"]="blocked";publish("ik-invariance-failed")
        raise RuntimeError("Stable IK invariance/known-branch audit failed; no GPU launch")
    publish("cpu-ik-audit-passed")
    if a.audit_only:
        report["status"]="audit_completed";publish("cpu-audit-only")
        return
    from .stable_geometry import audit_values
    from .expanded_env import ExpandedEnvironment
    from .bank_study import run_bank_rows
    from .repeat_diagnostic import ContactRecorder, ObservedSensor
    from .gpu_slot import gpu_slot

    def waiting(info):
        report["waiting_for_4090"]=info;publish("waiting-for-4090")

    def save_packet(environment,recorder,packet,phase,audited=False):
        nonlocal cache_index
        xy=np.asarray([p["bucket_xy_m"] for p in packet]);values=np.asarray([p["values"] for p in packet])
        hints10=np.asarray([p.get("known_q10",[np.nan]*6) for p in packet]);hints14=np.asarray([p.get("known_q14",[np.nan]*6) for p in packet])
        hint_values=np.asarray([p.get("known_values",[np.nan]*5) for p in packet]);hint_bucket=np.asarray([p.get("known_bucket_xy",[np.nan]*2) for p in packet])
        hint_used=np.asarray([p.get("known_hint_used",bool(np.isfinite(hints10[i]).all() and np.isfinite(hints14[i]).all())) for i,p in enumerate(packet)])
        if audited:
            q10=np.asarray([p["q10"] for p in packet]);q14=np.asarray([p["q14"] for p in packet]);eligible=np.asarray([p.get("eligible",True) for p in packet])
            branch10=np.asarray([p.get("branch10",-1) for p in packet]);branch14=np.asarray([p.get("branch14",-1) for p in packet]);preserved=np.asarray([p.get("known_pair_preserved",False) for p in packet])
            request_match=np.asarray([p.get("known_request_match",False) for p in packet]);exact_nominal=np.asarray([p.get("preserved_exact_nominal",False) for p in packet]);refined=np.asarray([p.get("refined_continuation",False) for p in packet])
        else:
            mask,detail=audit_values(scene,a.assets,xy,values[:,None],bounds=WIDE_BOUNDS,
                known_q10=hints10[:,None],known_q14=hints14[:,None],known_values=hint_values[:,None],known_bucket_xy=hint_bucket[:,None])
            q10,q14,eligible=detail["q10"][:,0],detail["q14"][:,0],mask[:,0]
            branch10,branch14=detail["branch10"][:,0],detail["branch14"][:,0];preserved=detail["known_pair_preserved"][:,0]
            request_match=detail["known_request_match"][:,0];exact_nominal=detail["preserved_exact_nominal"][:,0];refined=detail["refined_continuation"][:,0]
            report["geometry_proposals"]+=len(values)
        name=f"endpoint-batch-{cache_index:04d}.npz";cache_index+=1
        np.savez_compressed(a.output/name,bucket=xy,values=values,q10=q10,q14=q14,
            eligible=eligible,known_hint_used=hint_used,known_q10=hints10,known_q14=hints14,known_values=hint_values,known_bucket_xy=hint_bucket,branch10=branch10,branch14=branch14,known_pair_preserved=preserved,known_request_match=request_match,preserved_exact_nominal=exact_nominal,refined_continuation=refined)
        rows=run_bank_rows(environment,xy,values,q10,q14,eligible,record=phase.startswith("control"))
        if eligible.any():forces,position=recorder.arrays()
        native_index=0
        with (a.output/"shots.jsonl").open("a") as stream:
            for i,(row,p) in enumerate(zip(rows,packet)):
                if eligible[i]:
                    row.update(flight_summary(forces[:,native_index],position[:,native_index],row,scene));native_index+=1
                row.update(id=f"stable-shot-{len(records):07d}",phase=phase,bound_phase="wide",target_id=p["target_id"],
                    source_phase="targeted_development",values=p["values"],geometry_cache_file=name,geometry_cache_row=i,
                    geometry_eligible=bool(eligible[i]),branch10=int(branch10[i]),branch14=int(branch14[i]),
                    known_hint_used=bool(hint_used[i]),known_pair_preserved=bool(preserved[i]),known_request_match=bool(request_match[i]),preserved_exact_nominal=bool(exact_nominal[i]),refined_continuation=bool(refined[i]))
                for key in ("seed_source_id","seed_source_record_sha256","proposal_family","round_index","confirmation_source_id","control_source_id","repeat"):
                    if key in p:row[key]=p[key]
                records.append(row);stream.write(json.dumps(row,allow_nan=False)+"\n")
        return rows

    rng=np.random.default_rng(SEED);queue=[]
    with gpu_slot(a.gpu_lock,timeout_s=a.gpu_wait_timeout,notify=waiting):
        report.pop("waiting_for_4090",None);publish("gpu-acquired")
        import torch
        torch.set_num_threads(2)
        environment=ExpandedEnvironment(scene,a.assets,n=a.batch,bounds=WIDE_BOUNDS)
        recorder=ContactRecorder(environment);environment.sensors[-1]=ObservedSensor(environment.sensors[-1],recorder)
        for repeat in range(a.control_repeats):
            for mode in ("saved","legacy_fresh","stable_matched","stable_free","stable_seeded"):
                packet=[]
                for i,source in enumerate(controls):
                    p=dict(target_id=source["target_id"],bucket_xy_m=source["bucket_xy_m"],values=source["values"],control_source_id=source["id"],repeat=repeat,known_q10=source["q10"],known_q14=source["q14"],known_values=source["values"],known_bucket_xy=source["bucket_xy_m"],known_hint_used=mode=="stable_seeded")
                    if mode=="saved":p.update(q10=source["q10"],q14=source["q14"],eligible=True)
                    else:
                        cache_mode="old" if mode=="legacy_fresh" else mode
                        p.update(q10=audit_cache[cache_mode+"_q10"][i],q14=audit_cache[cache_mode+"_q14"][i],eligible=bool(audit_cache[cache_mode+"_mask"][i]))
                        for key in ("branch10","branch14","known_pair_preserved","known_request_match","preserved_exact_nominal","refined_continuation"):
                            if cache_mode+"_"+key in audit_cache:p[key]=audit_cache[cache_mode+"_"+key][i]
                    packet.append(p)
                for offset in range(0,len(packet),a.batch):
                    save_packet(environment,recorder,packet[offset:offset+a.batch],"control_"+mode,True)
                report["control_attempts"]+=len(packet);publish("native-controls")
        scheduled={t["id"]:0 for t in targets}
        def flush(force=False):
            while len(queue)>=a.batch or (force and queue):
                packet=queue[:a.batch];del queue[:len(packet)]
                rows=save_packet(environment,recorder,packet,"search",True)
                for row in rows:searched[row["target_id"]].append(row)
                report["native_search_attempts"]+=len(rows);publish("native-search")
        for round_index in range(a.proposal_rounds):
            for target in targets:
                tid=target["id"]
                if scheduled[tid]>=a.native_per_target:continue
                seeds=diverse_sources(initial_seeds[tid]+searched[tid],tid,a.seed_sources)
                values,h10,h14,hvalues,hbucket,metadata=proposals(target,a.proposals,rng,scene,seeds,round_index)
                xy=np.repeat([target["bucket_xy_m"]],len(values),axis=0)
                mask,detail=audit_values(scene,a.assets,xy,values[:,None],bounds=WIDE_BOUNDS,
                    known_q10=h10[:,None],known_q14=h14[:,None],known_values=hvalues[:,None],known_bucket_xy=hbucket[:,None])
                report["geometry_proposals"]+=len(values)
                for j in np.flatnonzero(mask[:,0]):
                    if scheduled[tid]>=a.native_per_target:break
                    queue.append(dict(target_id=tid,bucket_xy_m=target["bucket_xy_m"],values=values[j].tolist(),
                        q10=detail["q10"][j,0],q14=detail["q14"][j,0],
                        known_q10=h10[j],known_q14=h14[j],known_values=hvalues[j],known_bucket_xy=hbucket[j],branch10=detail["branch10"][j,0],branch14=detail["branch14"][j,0],
                        known_pair_preserved=detail["known_pair_preserved"][j,0],known_request_match=detail["known_request_match"][j,0],
                        preserved_exact_nominal=detail["preserved_exact_nominal"][j,0],refined_continuation=detail["refined_continuation"][j,0],**metadata[j]))
                    scheduled[tid]+=1
                flush();publish("cpu-search-geometry")
            flush(True)
            if all(n>=a.native_per_target for n in scheduled.values()):break
        captured=[]
        for target in targets:
            candidates=[r for r in searched[target["id"]] if r["route_success"] and not r["invalid"]]
            captured.extend(sorted(candidates,key=lambda r:(acquisition(r),r["id"]),reverse=True)[:a.top])
        def confirmation_packets(sources,count,seed):
            random=np.random.default_rng(seed);packets=[]
            for source in sources:
                vals=np.clip(np.asarray(source["values"])+random.normal(0,NOISE,(count,5)),WIDE_BOUNDS["low"],WIDE_BOUNDS["high"])
                packets.extend(dict(target_id=source["target_id"],bucket_xy_m=source["bucket_xy_m"],values=v.tolist(),
                    known_q10=source["q10"],known_q14=source["q14"],known_values=source["values"],known_bucket_xy=source["bucket_xy_m"],confirmation_source_id=source["id"]) for v in vals)
            return packets
        packets=confirmation_packets(captured,10,SEED+1)
        for offset in range(0,len(packets),a.batch):
            rows=save_packet(environment,recorder,packets[offset:offset+a.batch],"confirmation")
            for row in rows:confirm.setdefault(row["confirmation_source_id"],[]).append(row)
            report["confirmation_attempts"]+=len(rows);publish("first-confirmation")
        frozen=[]
        for target in targets:
            candidates=[r for r in captured if r["target_id"]==target["id"] and len(confirm.get(r["id"],[]))==10 and sum(bool(s["route_success"]) and not s["invalid"] for s in confirm[r["id"]])>=8]
            if candidates:frozen.append(max(candidates,key=lambda r:(sum(bool(s["route_success"]) and not s["invalid"] for s in confirm[r["id"]]),acquisition(r),r["id"])))
        atomic(a.output/"frozen-final-sources.json",[dict(id=r["id"],sha256=digest(r)) for r in frozen])
        packets=confirmation_packets(frozen,20,SEED+2)
        for offset in range(0,len(packets),a.batch):
            rows=save_packet(environment,recorder,packets[offset:offset+a.batch],"final_confirmation")
            for row in rows:final.setdefault(row["confirmation_source_id"],[]).append(row)
            report["final_confirmation_attempts"]+=len(rows);publish("final-confirmation")
        for source in frozen:
            rows=final.get(source["id"],[])
            if len(rows)!=20:continue
            template=qualified_template(source,rows,source["target_id"],WIDE_BOUNDS,.8)
            if template:
                template.update(source_record_sha256=digest(source),first_confirmation=statistics(confirm[source["id"]]),final_confirmation=statistics(rows),
                    source_study_contract_fingerprint=spec["fingerprint"],confirmation_study_contract_fingerprint=spec["fingerprint"],
                    nominal_source=dict(bucket_xy_m=source["bucket_xy_m"],requested_values=source["values"],
                        q10=source["q10"],q14=source["q14"],branch10=source["branch10"],branch14=source["branch14"],
                        geometry_cache_file=source["geometry_cache_file"],geometry_cache_row=source["geometry_cache_row"],
                        reuse_scope="Exact nominal source request may preserve these joints after strict/native checks; translated or changed targets use them only as continuation seeds"),
                    qualification="Stable IK native source,>=8/10 screening and>=16/20 independent final confirmations")
                promoted[source["target_id"]]=template
    output_bank=copy.deepcopy(bank);output_bank.pop("sha256")
    output_bank["templates"]+=list(promoted.values())
    output_bank.update(parent_bank_sha256=bank["sha256"],stable_ik_contract_fingerprint=spec["fingerprint"],
        action_bounds=WIDE_BOUNDS if promoted else bank["action_bounds"],
        stable_ik_development_layouts=targets,source_scope="Original17-template prefix preserved; additions use deterministic branch-seeded native development search, not held-out labels")
    output_bank["sha256"]=digest(output_bank);atomic(a.output/"confirmed-bank.json",output_bank)
    report.update(status="completed",confirmed_bank_sha256=sha256(a.output/"confirmed-bank.json"),
        proposal_bank_sha256=output_bank["sha256"],base_templates=len(bank["templates"]),confirmed_templates=len(output_bank["templates"]),
        native_search_statistics=statistics([r for r in records if r["phase"]=="search"]),
        confirmation_statistics=statistics([r for r in records if r["phase"]=="confirmation"]),
        final_confirmation_statistics=statistics([r for r in records if r["phase"]=="final_confirmation"]),
        control_statistics={mode:statistics([r for r in records if r["phase"]=="control_"+mode]) for mode in ("saved","legacy_fresh","stable_matched","stable_free","stable_seeded")},
        versions={name:importlib.metadata.version(name) for name in ("mjlab","mujoco","mujoco-warp","torch","warp-lang")},
        source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")},
        artifacts_sha256={p.name:sha256(p) for p in a.output.iterdir() if p.is_file() and p.name!="report.json"})
    for origin in provenance.get("source_origins",[]):
        directory=Path(origin["directory"])
        for filename,key in (("report.json","report_sha256"),("contract.json","contract_sha256"),("shots.jsonl","shots_sha256")):
            if sha256(directory/filename)!=origin[key]:raise RuntimeError("Prior source mutated during the study")
    publish("completed")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study","prior-study","prior-expansion","assets","output","gpu-lock"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source-study",type=Path,action="append",required=True)
    parser.add_argument("--batch",type=int,default=128)
    parser.add_argument("--native-per-target",type=int,default=512)
    parser.add_argument("--proposal-rounds",type=int,default=48)
    parser.add_argument("--proposals",type=int,default=128)
    parser.add_argument("--seed-sources",type=int,default=12)
    parser.add_argument("--top",type=int,default=8)
    parser.add_argument("--control-repeats",type=int,default=3)
    parser.add_argument("--gpu-wait-timeout",type=float,default=3600)
    parser.add_argument("--audit-only",action="store_true")
    args=parser.parse_args()
    if not(1<=args.batch<=256 and 64<=args.native_per_target<=2048 and 1<=args.proposal_rounds<=128
        and 16<=args.proposals<=1024 and 1<=args.seed_sources<=32 and 1<=args.top<=20
        and 1<=args.control_repeats<=10 and 0<args.gpu_wait_timeout<=86400):
        parser.error("Finite audit/search/confirmation budgets required")
    run(args)
