"""Independent CPU evidence audit for the versioned lower-board far study.

Saved requests/contacts are read without allocating a simulator, changing a
source parent, launching a policy or editing historical scoring definitions.
"""
import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from .distance_verify import (CORE_SOURCE, GEOMETRY_KEYS, audit_contexts, audit_footprint,
    audit_sources as audit_source_common, paired_sensitivity)
from .neighborhood_verify import audit_batches, inside, lines, read, statistics
from .verify_expansion import Audit, digest, sha256
from .window_verify import row_reward

VERSION = "sircus-low-board-far-evidence-audit-v1"
PARENT_FINGERPRINT = "6d94edc505db7201f603456116303cd14acd9dde26e364ea295b1739c78f8499"
SEED_IDS = ("recipe-00", "recipe-02", "recipe-03", "recipe-04", "recipe-08",
    "recipe-09", "recipe-10", "recipe-11")
PUBLICATION_OVERRIDE = {"sircus/mjlab/prepare_distance_report.py": (
    "7fb5088df5103de9155af365ecc85021c45d9e1aacdffa725fbbdad64ac29578",
    "9058f1873ed6aeabc348da0af2c6c290d403da27bb9d546fd4aded729cfa27a9")}


def audit_parent(a, spec, parent_study):
    root = Path(parent_study)
    parent, report, checked = (read(root / name) for name in
        ("contract.json", "report.json", "audit-distance.json"))
    payload = dict(parent); fingerprint = payload.pop("fingerprint", None)
    metadata = spec["parent_study"]
    checks = checked.get("checks", [])
    a.check("exact_completed_audited_distance_parent", digest(payload) == fingerprint
        == metadata["contract_fingerprint"] == report.get("contract_fingerprint")
        == checked.get("contract_fingerprint") == PARENT_FINGERPRINT
        and parent.get("version") == "sircus-bucket-distance-development-v1"
        and report.get("status") == report.get("phase") == "completed"
        and checked.get("status") == "passed" and not checked.get("failures")
        and bool(checks) and type(checked.get("checks_count")) is int
        and checked["checks_count"] == len(checks) and all(c.get("passed") is True for c in checks))
    required = {"contract.json", "report.json", "audit-distance.json", "evidence-manifest.json", "records.jsonl"}
    a.check("parent_declared_input_files", required <= set(metadata["files_sha256"])
        and all(sha256(inside(root, n)) == h for n, h in metadata["files_sha256"].items()))
    manifest = read(root / "evidence-manifest.json")
    a.check("parent_raw_evidence_unchanged", manifest == report.get("evidence_sha256")
        and all(sha256(inside(root, n)) == h for n, h in manifest.items()))
    seeds = spec["seeds"]
    a.check("exact_eight_shared_parent_geometry_seeds", seeds == parent["seeds"]
        and tuple(s["id"] for s in seeds) == SEED_IDS
        and all(set(s) == {"id", *GEOMETRY_KEYS} for s in seeds))
    controls = [r for r in lines(root / "records.jsonl") if r["role"] == "control"]
    provenance, bad = [], []
    for seed in seeds:
        group = [r for r in controls if r["source_anchor_id"] == seed["id"]]
        provenance.append(dict(id=seed["id"], parent_control_record_ids=[r["id"] for r in group],
            geometry_sha256=digest(seed)))
        if len(group) != 3 or any(r["substeps"] != 64 or not r["native_release"]
                or not r["route_success"] or r.get("invalid")
                or any(r[k] != seed[k] for k in GEOMETRY_KEYS) for r in group):
            bad.append(seed["id"])
    a.check("all_24_exact_parent_controls_and_source_lineage", not bad and len(controls) == 24
        and spec["seed_provenance"] == provenance, bad)
    a.check("same_parent_scene_native_assets_and_packages", all(spec[k] == parent[k]
        for k in ("scene", "asset_sha256", "packages")))
    differences, incompatible = {}, []
    for name, old in parent["source_sha256"].items():
        new = spec["source_sha256"].get(name)
        if new == old: continue
        if PUBLICATION_OVERRIDE.get(name) != (old, new): incompatible.append(name)
        differences[name] = dict(parent_sha256=old, current_sha256=new,
            purpose="Read-only display adapter published after parent execution; not imported by native physics")
    a.check("only_exact_publication_hash_override_and_no_physics_changes", not incompatible
        and spec["source_compatibility"] == differences, incompatible)
    return parent


def audit_sources(a, study, source, assets, spec, report):
    audit_source_common(a, study, source, assets, spec, report)
    own = Path(__file__).resolve().parents[2]
    for name in ("sircus/mjlab/lowboard_contract.py", "sircus/mjlab/lowboard_inputs.py",
            "sircus/mjlab/lowboard_study.py", "sircus/mjlab/lowboard_verify.py"):
        a.check("matching_lowboard_logic:" + name, sha256(own / name) == spec["source_sha256"].get(name))


def audit_targets_and_grid(a, spec, parent):
    """Reconstruct the finite requested grid and geometry-only source assignment."""
    seeds, settings = spec["seeds"], spec["settings"]
    expected_targets = [t for t in parent["settings"]["targets"]
        if t["id"] in ("corner-d01", "corner-d02", "corner-d03", "corner-d04")]
    a.check("four_exact_supported_parent_far_targets", settings["targets"] == expected_targets
        and len(expected_targets) == 4 and all(t["bucket_on_table"] for t in expected_targets))
    corner = dict(next(r for r in parent["settings"]["rays"] if r["id"] == "corner"))
    corner["distances_m"] = [t["distance_m"] for t in expected_targets]
    a.check("same_parent_corner_ray_without_layout_change", settings["rays"] == [corner])
    board = np.asarray(seeds[0]["values"][:2], float)
    radius = spec["scene"]["ball"]["radius_m"]
    ball_limit = float(.364 + np.sqrt(.842 ** 2 - float(board @ board)) - .095 - .09)
    expected_limit = dict(ball_center_height_m=ball_limit, board_height_plus_gap_m=ball_limit - radius,
        wrist_radius_m=.842, shoulder_height_m=.364, flange_above_ball_m=.095,
        prior_wrist_offset_m=.09,
        formula="Zball_max = 0.364 + sqrt(0.842^2 - X^2 - Y^2) - 0.095 - 0.09",
        scope="Unchanged cheap spherical workspace prior only; native IK and strict clearance are separate checks")
    a.check("independent_unchanged_outer_prior_release_ceiling", settings["reach_limit"] == expected_limit)
    control = settings["control_target"]
    control_expected = dict(id="parent-control", distance_id="parent-control", ray_id="original_control",
        distance_m=float(np.linalg.norm(np.asarray(seeds[0]["bucket_xy_m"]) - board)),
        board_xy_m=board.tolist(), bucket_xy_m=seeds[0]["bucket_xy_m"],
        scope="Exact source controls; excluded from far-target search tables")
    a.check("separate_original_control_bucket", control == control_expected)
    contexts = []
    for seed in seeds:
        contexts.append(dict(id="control:" + seed["id"], distance_id=control["id"], ray_id=control["ray_id"],
            distance_m=control["distance_m"], board_xy_m=control["board_xy_m"], bucket_xy_m=control["bucket_xy_m"],
            source_anchor_id=seed["id"], source_family=seed["id"], board_height_m=seed["values"][2],
            tilt_deg=seed["values"][3], proposed_anchor_gap_m=seed["values"][4], stage="control"))
    for target in expected_targets:
        for hi, height in enumerate(tuple(i / 1000 for i in range(220, 361, 20))):
            for ti, tilt in enumerate((28., 32., 36., 40.)):
                seed = min(seeds, key=lambda s: (((s["values"][2] - height) / .04) ** 2
                    + ((s["values"][3] - tilt) / 4.) ** 2, s["id"]))
                contexts.append(dict(id=f"{target['id']}:h{hi:02d}-t{ti:02d}",
                    **{k: target[k] for k in ("distance_id", "ray_id", "distance_m", "board_xy_m", "bucket_xy_m")},
                    source_anchor_id=seed["id"], source_family=seed["id"], board_height_m=height,
                    tilt_deg=tilt, proposed_anchor_gap_m=seed["values"][4], stage="lowboard"))
    return contexts


def audit_controls(a, study, spec, rows):
    target = spec["settings"]["control_target"]
    expected = [dict(role="control", phase="controls", source_anchor_id=s["id"], context_id="control:" + s["id"],
        distance_id=target["id"], ray_id=target["ray_id"], distance_m=target["distance_m"],
        gap_m=s["values"][4], repeat_index=i) for s in spec["seeds"] for i in range(3)]
    controls = [r for r in rows if r["role"] == "control"]
    a.check("exact_24_frozen_control_requests", read(study / "control-plan.json") == expected
        and len(controls) == 24 and all(r["substeps"] == 64
        and all(r.get(k) == v for k, v in p.items()) for r, p in zip(controls, expected)))
    a.check("each_parent_control_reproduces_three_exact_captures", all(
        len(group := [r for r in controls if r["source_anchor_id"] == s["id"]]) == 3
        and all(r["native_release"] and r["route_success"] and not r.get("invalid")
        and all(r[k] == s[k] for k in GEOMETRY_KEYS) for r in group) for s in spec["seeds"]))


def audit_plans(a, study, spec, declared, rows, recipes):
    from . import lowboard_contract as protocol
    a.check("all_136_frozen_geometry_only_contexts", read(study / "declared-contexts.json") == declared
        and len(declared) == 136 and sum(c["stage"] == "lowboard" for c in declared) == 128)
    limit = spec["settings"]["reach_limit"]["board_height_plus_gap_m"]
    expected_coarse = [dict(context_id=c["id"], distance_id=c["distance_id"], ray_id=c["ray_id"],
        distance_m=c["distance_m"], source_anchor_id=c["source_anchor_id"], role="development", phase="coarse",
        gap_m=float(limit - c["board_height_m"] - margin), release_headroom_m=margin,
        requested_ball_center_height_m=float(limit + spec["scene"]["ball"]["radius_m"] - margin))
        for c in declared if c["stage"] == "lowboard" for margin in (.005, .015, .030, .050, .080)]
    coarse = [r for r in rows if r["phase"] == "coarse"]
    a.check("all_640_exact_unclipped_near_ceiling_requests", read(study / "coarse-plan.json") == expected_coarse
        and len(coarse) == len(expected_coarse) == 640 and all(r["substeps"] == 64
        and all(r.get(k) == v for k, v in p.items()) for r, p in zip(coarse, expected_coarse)))
    refined = [r for r in rows if r["phase"] == "refinement"]
    expected_refined = protocol.refinement_plan(declared, coarse)
    a.check("four_refinements_only_for_captured_development_contexts",
        read(study / "refinement-plan.json") == expected_refined and len(refined) == len(expected_refined) <= 512
        and all(r["substeps"] == 64 and all(r.get(k) == v for k, v in p.items()) for r, p in zip(refined, expected_refined)))
    expected_recipes = protocol.select_recipes([r for r in rows if r["role"] == "development"], spec["settings"]["targets"])
    a.check("at_most_eight_actual_development_capture_recipes", recipes == expected_recipes and len(recipes) <= 8
        and all(r["source"]["native_release"] and r["source"]["route_success"] and not r["source"].get("invalid") for r in recipes))
    repeats = protocol.repeat_plan(recipes)
    a.check("frozen_seeded_33_requested_repeats_per_recipe", read(study / "frozen-repeat-plan.json") == repeats
        and len(repeats) == 33 * len(recipes))
    actual = [r for r in rows if r["role"] == "engineering_repeat"]
    a.check("complete_budget_and_no_hidden_retry_or_resampling", len(rows)
        == 24 + 640 + len(expected_refined) + len(repeats) * 3 <= protocol.MAX_REQUESTS == 1968
        and Counter((r["repeat_id"], r["substeps"]) for r in actual)
        == Counter((r["repeat_id"], step) for r in repeats for step in (32, 64, 128))
        and set(r["phase"] for r in rows) <= {"controls", "coarse", "refinement", "nominal", "fresh"})


def audit_options(a, study, report, rows):
    from .timestep_env import validate_options
    options = read(study / "effective-options.json")
    expected = {str(r["substeps"]) for r in rows if r["native_release"]}
    a.check("only_actually_used_native_engines_claimed", set(options) == expected
        and "64" in expected and expected <= {"32", "64", "128"}
        and options == report.get("effective_native_options"))
    for step, claim in options.items():
        try:
            w = claim["warp"]
            native = NS(timestep=claim["timestep_s"], tolerance=claim["tolerance"], gravity=claim["gravity"],
                solver=claim["solver"], integrator=claim["integrator"], cone=claim["cone"], iterations=claim["iterations"])
            warp = NS(timestep=np.asarray(w["timestep_s"], np.float32), tolerance=np.asarray(w["tolerance"], np.float32),
                gravity=np.asarray(w["gravity"], np.float32), solver=w["solver"], integrator=w["integrator"],
                cone=w["cone"], iterations=w["iterations"])
            passed = validate_options(native, warp, int(step)) == claim
        except (KeyError, ValueError, TypeError, RuntimeError): passed = False
        a.check("actual_allocated_native_warp_options_" + step, passed)

def audit_rows(a, spec, prepared, rows, recipes, repeat_plan, repeat_geometry, *, record_prefix="lowboard-shot-"):
    contexts = {r["id"]: r for r in prepared}
    selected = {r["id"]: r for r in recipes}
    repeat_lookup = {r["repeat_id"]: (r, g)
        for r, g in zip(repeat_geometry["requests"], repeat_geometry["rows"])}
    a.check("unique_sequential_request_ids", [r["id"] for r in rows]
        == [f"{record_prefix}{i:06d}" for i in range(len(rows))])
    a.check("frozen_repeat_geometry_request_identity", repeat_geometry["requests"] == repeat_plan
        and len(repeat_lookup) == len(repeat_plan) == len(repeat_geometry["rows"]))
    bad, release_bad, reward_bad = [], [], []
    paired = defaultdict(dict)
    bounds = spec["settings"]["bounds"]
    for row in rows:
        label = row["id"]
        values = np.asarray(row.get("values", []), float)
        geometry = row.get("geometry", {})
        fields = ("context_id", *GEOMETRY_KEYS, "eligible", "board_branch_sha256",
            "minimum_clearance_lower_bound_m")
        if values.shape != (5,) or not np.isfinite(values).all() or any(
                row.get(k) != geometry.get(k) for k in fields):
            bad.append(label); continue
        if not np.array_equal(values[:2], spec["settings"]["board_xy_m"]) \
                or values[4] != row["gap_m"]:
            bad.append(label)
        if row["role"] in ("control", "development"):
            context = contexts.get(row["context_id"])
            if context is None:
                bad.append(label); continue
            keys = ("bucket_xy_m", "board_branch_sha256", "source_anchor_id",
                "source_anchor_geometry_sha256", "distance_id", "ray_id", "distance_m")
            if any(geometry.get(k) != context.get(k) for k in keys) \
                    or values[2] != context["board_height_m"] \
                    or values[3] != context["tilt_deg"] \
                    or (row["eligible"] and row["q14"] != context["q14"]):
                bad.append(label)
            if geometry.get("wrapper_version") != "sircus-bucket-distance-frozen-branch-geometry-v1":
                bad.append(label)
            if any(row.get(k) != context.get(k) for k in ("distance_id", "ray_id", "distance_m")):
                bad.append(label)
        elif row["role"] == "engineering_repeat":
            matched = repeat_lookup.get(row.get("repeat_id"))
            recipe = selected.get(row.get("recipe_id"))
            if matched is None or recipe is None or any(
                    row.get(k) != v for k, v in matched[0].items()) or geometry != matched[1]:
                bad.append(label); continue
            nominal = {k: recipe["source"][k] for k in GEOMETRY_KEYS}
            if geometry.get("wrapper_version") != "sircus-nominal-branch-local-variation-v1" \
                    or geometry.get("nominal_geometry_sha256") != digest(nominal) \
                    or geometry.get("distance_wrapper_version") != "sircus-bucket-distance-local-variation-v1" \
                    or row["bucket_xy_m"] != recipe["bucket_xy_m"]:
                bad.append(label)
            if row["eligible"] and any(np.max(np.abs(np.asarray(row[k])
                    - recipe["source"][k])) > .10 for k in ("q10", "q14")):
                bad.append(label)
            if row["phase"] == "nominal" and any(row[k] != recipe["source"][k]
                    for k in GEOMETRY_KEYS):
                bad.append(label)
            paired[row["repeat_id"]][row["substeps"]] = digest(dict(
                values=row["values"], bucket_xy_m=row["bucket_xy_m"], q10=row["q10"],
                q14=row["q14"], geometry=geometry))
        else:
            bad.append(label)
        footprint = geometry.get("bucket_footprint", {})
        on_table = audit_footprint(a, spec["scene"], row["bucket_xy_m"], footprint, label)
        if row.get("native_release") != bool(row.get("eligible")) \
                or (not row.get("eligible") and (not row.get("invalid") or row["route_success"])):
            release_bad.append(label)
        if row.get("eligible") and (not on_table or row.get("invalid")
                or row.get("minimum_clearance_lower_bound_m") is None
                or row["minimum_clearance_lower_bound_m"] < .005
                or not ((values >= bounds["low"]).all() and (values <= bounds["high"]).all())
                or any(np.asarray(row[k]).shape != (6,) or not np.isfinite(row[k]).all()
                    for k in ("q10", "q14"))):
            release_bad.append(label)
        if row["route_success"] and (not row["board_bounce"] or not row["whole_ball_entry"]
                or row["max_retention_s"] < 1 - 1e-9 or row["external_contact_steps"]):
            release_bad.append(label)
        if not np.isclose(row.get("learning_reward", np.nan), row_reward(row), rtol=0, atol=1e-9):
            reward_bad.append(label)
    a.check("every_requested_target_and_frozen_source_branch_preserved", not bad, bad[:10])
    a.check("all_rejections_retained_and_only_admitted_endpoints_released", not release_bad, release_bad[:10])
    a.check("unchanged_terminal_reward", not reward_bad, reward_bad[:10])
    a.check("same_requested_geometry_and_joints_at_all_three_timesteps",
        len(paired) == len(repeat_plan) and all(set(v) == {32, 64, 128}
        and len(set(v.values())) == 1 for v in paired.values()))


def audit_report(a, spec, report, declared, prepared, rows, recipes, positions):
    expected_counts = dict(requests=len(rows), native_releases=sum(bool(r["native_release"]) for r in rows),
        captures=sum(bool(r["route_success"]) for r in rows), clean=sum(r["route_quality"] == 2 for r in rows),
        geometry_rejections=sum(bool(r.get("invalid")) for r in rows),
        phase_counts=dict(Counter(r["phase"] for r in rows)),
        substep_counts=dict(Counter(str(r["substeps"]) for r in rows)))
    a.check("reported_complete_request_and_release_denominators", report["counts"] == expected_counts)
    a.check("all_requested_contexts_reported_in_order", [c["id"] for c in report["contexts"]]
        == [c["id"] for c in declared])
    context_lookup = {p["id"]: p for p in prepared}
    declared_lookup = {c["id"]: c for c in declared}
    for context in report["contexts"]:
        rr = [r for r in rows if r["context_id"] == context["id"] and r["role"] == "development"]
        hits = [r for r in rr if r["route_success"]]
        geometry = context_lookup[context["id"]]
        expected = dict(geometry_requests=len(rr), eligible=sum(r["eligible"] for r in rr),
            search_attempts=len(rr), search_native_releases=sum(bool(r["native_release"]) for r in rr),
            search_captures=len(hits), clean=sum(r["route_quality"] == 2 for r in rr),
            best_gap_m=max(hits, key=lambda r: (r["route_quality"], row_reward(r)))["gap_m"] if hits else None,
            board_admitted=geometry["board_admitted"], reason=geometry.get("reason"),
            board_branch_sha256=geometry.get("board_branch_sha256"))
        a.check("context_report:" + context["id"], all(context.get(k) == v for k, v
            in {**declared_lookup[context["id"]], **expected}.items()))
    target_rows = spec["settings"]["targets"]
    a.check("all_four_far_targets_reported_in_order", [t["id"] for t in report["targets"]]
        == [t["id"] for t in target_rows])
    target_lookup = {t["id"]: t for t in target_rows}
    for target in report["targets"]:
        rr = [r for r in rows if r["distance_id"] == target["id"]]
        search = [r for r in rr if r["role"] == "development"]
        expected = dict(requests=len(rr), native_releases=sum(bool(r["native_release"]) for r in rr),
            captures=sum(bool(r["route_success"]) for r in rr), geometry_rejections=sum(bool(r.get("invalid")) for r in rr),
            search_attempts=len(search), search_native_releases=sum(bool(r["native_release"]) for r in search),
            search_captures=sum(bool(r["route_success"]) for r in search))
        a.check("target_report:" + target["id"], target["counts"] == expected
            and all(target.get(k) == v for k, v in target_lookup[target["id"]].items()))
    a.check("selected_recipe_report_identity", [r["id"] for r in report["selected_recipes"]]
        == [r["id"] for r in recipes])
    for claim, recipe in zip(report["selected_recipes"], recipes):
        rr = [r for r in rows if r.get("recipe_id") == recipe["id"]]
        keys = ("id", "context_id", "distance_id", "ray_id", "distance_m", "bucket_xy_m", "board_xy_m",
            "board_height_m", "tilt_deg", "gap_m", "selection", "sampled_run_width_m")
        a.check(recipe["id"] + ":source_identity", all(claim.get(k) == recipe[k] for k in keys)
            and claim["source_record_id"] == recipe["source"]["id"]
            and claim["release_center_height_m"] == recipe["board_height_m"] + recipe["gap_m"]
                + spec["scene"]["ball"]["radius_m"])
        for step in (32, 64, 128):
            for key, phase in (("by_substeps", None), ("nominal_by_substeps", "nominal"), ("fresh_by_substeps", "fresh")):
                selected = [r for r in rr if r["substeps"] == step and (phase is None or r["phase"] == phase)]
                a.stats(recipe["id"] + ":" + str(step) + ":" + key, claim[key][str(step)], statistics(selected))
        expected = paired_sensitivity(rr, positions)
        a.check(recipe["id"] + ":independent_paired_sensitivity",
            all(claim["sensitivity"].get(k) == v for k, v in expected.items()))
    lookup = {r["id"]: r for r in rows}
    for index, trace in enumerate(report["trace_examples"]):
        row = lookup[trace["record_id"]]
        keys = ("context_id", "distance_id", "ray_id", "distance_m", "bucket_xy_m",
            "substeps", "gap_m", "route_success")
        a.check("trace:" + str(index) + ":actual_bucket_associated_native_positions",
            row["native_release"] and all(trace.get(k) == row.get(k) for k in keys)
            and trace["board_xy_m"] == row["values"][:2]
            and trace["position_m"] == positions[row["id"]][::3].tolist())


def audit_display_selection(a, spec, report, rows, recipes):
    expected = []
    for recipe in recipes:
        for step in (64, 128):
            row = next(r for r in rows if r.get("recipe_id") == recipe["id"]
                and r["substeps"] == step and r["repeat_index"] == 0)
            if row["native_release"]: expected.append(row["id"])
    for target in spec["settings"]["targets"]:
        native = [r for r in rows if r["role"] == "development"
            and r["distance_id"] == target["id"] and r["native_release"]]
        if native:
            row = max(native, key=lambda r: (bool(r["route_success"]), bool(r["whole_ball_entry"]),
                r["max_retention_s"], -r["closest_opening_distance_m"], r["id"]))
            expected.append(row["id"])
    a.check("native_nominal_and_explicit_posthoc_development_examples", len(expected) <= 20
        and [t["record_id"] for t in report["trace_examples"]] == expected)


def audit(study, source, assets, parent_study):
    from . import lowboard_contract as protocol
    from .timestep_env import physics_settings
    study = Path(study)
    spec, report = read(study / "contract.json"), read(study / "report.json")
    rows, recipes = lines(study / "records.jsonl"), read(study / "recipes.json")
    a = Audit()
    payload = dict(spec); fingerprint = payload.pop("fingerprint", None)
    a.check("contract_identity", digest(payload) == fingerprint == report.get("contract_fingerprint"))
    a.check("completed_version", spec.get("version") == report.get("version") == protocol.VERSION
        and report.get("status") == report.get("phase") == "completed")
    a.check("prespecified_settings_and_scope", spec["settings"] == protocol.settings(spec["scene"], spec["seeds"])
        == report["settings"] and spec["scope"] == report["scope"] == protocol.SCOPE
        and report["max_requests"] == protocol.MAX_REQUESTS)
    a.check("all_three_declared_unchanged_timestep_contracts", spec["physics_by_substeps"]
        == {str(s): physics_settings(s) for s in (32, 64, 128)})
    parent = audit_parent(a, spec, parent_study)
    audit_sources(a, study, source, assets, spec, report)
    declared = audit_targets_and_grid(a, spec, parent)
    audit_plans(a, study, spec, declared, rows, recipes)
    prepared = read(study / "prepared-contexts.json")
    repeats, gates = read(study / "frozen-repeat-plan.json"), read(study / "repeat-geometry.json")
    audit_contexts(a, spec, declared, prepared)
    audit_controls(a, study, spec, rows)
    audit_rows(a, spec, prepared, rows, recipes, repeats, gates)
    audit_options(a, study, report, rows)
    positions, rescored = audit_batches(a, study, spec, rows)
    audit_report(a, spec, report, declared, prepared, rows, recipes, positions)
    audit_display_selection(a, spec, report, rows, recipes)
    failures = [r for r in a.checks if not r["passed"]]
    return dict(version=VERSION, status="failed" if failures else "passed", contract_fingerprint=fingerprint,
        checks_count=len(a.checks), checks=a.checks, failures=failures, records=len(rows),
        independently_rescored_throws=rescored, auditor_source_sha256=sha256(__file__),
        native_tested_substeps=sorted({r["substeps"] for r in rows if r["native_release"]}),
        scope="Saved source/geometry/scoring/provenance CPU audit; declared but unused timesteps are not native evidence; no new dynamics or policy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "assets", "parent-study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.study, args.source, args.assets, args.parent_study)
    except (KeyError, ValueError, TypeError, OSError, IndexError, AttributeError) as error:
        result = dict(version=VERSION, status="failed", checks_count=0, checks=[], failures=[dict(
            name="malformed_or_missing_evidence", passed=False, detail=str(error))])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks_count", "independently_rescored_throws")}))
    if result["status"] != "passed": raise RuntimeError("Lower-board evidence audit failed")


if __name__ == "__main__": main()
