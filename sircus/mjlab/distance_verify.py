"""Independent CPU evidence audit for the finite bucket-distance development study.

This module reads saved requests and traces; it never allocates a native engine,
advances dynamics, edits a source study, or activates a policy.
"""
import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .neighborhood_verify import (audit_batches, audit_options, inside, lines,
    read, statistics)
from .verify_expansion import Audit, digest, sha256
from .window_verify import row_reward

VERSION = "sircus-bucket-distance-evidence-audit-v1"
PARENT_FINGERPRINT = "12b2952c68b38d6bb392e3256d140cb7445ea97c73b090a6a39c5dfc4a4aa680"
PERFECT_PARENT_RECIPES = ("recipe-00", "recipe-02", "recipe-03", "recipe-04",
    "recipe-08", "recipe-09", "recipe-10", "recipe-11")
GEOMETRY_KEYS = ("values", "bucket_xy_m", "q10", "q14")
CORE_SOURCE = ("sircus/mjlab/timestep_env.py", "sircus/mjlab/contact_env.py",
    "sircus/mjlab/contact_model.py", "sircus/mjlab/contract.py",
    "sircus/mjlab/learning_contract.py", "sircus/isaac/geometry.py",
    "sircus/isaac/rl_contract.py", "sircus/isaac/entry_quality.py",
    "sircus/isaac/board_route.py")


def audit_parent(a, spec, parent_study):
    """Keep the audited parent and its eight exact geometry seeds immutable."""
    root = Path(parent_study)
    parent = read(root / "contract.json")
    report = read(root / "report.json")
    checked = read(root / "audit-neighborhood.json")
    metadata = spec["parent_study"]
    payload = dict(parent)
    fingerprint = payload.pop("fingerprint", None)
    a.check("exact_completed_audited_parent", digest(payload) == fingerprint
        == metadata["contract_fingerprint"] == report.get("contract_fingerprint")
        == checked.get("contract_fingerprint") == PARENT_FINGERPRINT
        and report.get("status") == report.get("phase") == "completed"
        and checked.get("status") == "passed" and not checked.get("failures")
        and checked.get("checks_count", 0) > 0
        and checked.get("checks_count") == len(checked.get("checks", []))
        and all(c.get("passed") is True for c in checked.get("checks", [])))
    required = {"contract.json", "report.json", "audit-neighborhood.json",
        "evidence-manifest.json", "recipes.json"}
    a.check("parent_declared_input_files", required <= set(metadata["files_sha256"]) and all(
        sha256(inside(root, name)) == expected
        for name, expected in metadata["files_sha256"].items()))
    manifest = read(root / "evidence-manifest.json")
    a.check("parent_raw_evidence_unchanged", manifest == report.get("evidence_sha256")
        and all(sha256(inside(root, name)) == expected for name, expected in manifest.items()))
    originals = read(root / "recipes.json")
    summaries = {r["id"]: r for r in report["selected_recipes"]}
    perfect = [r for r in originals if all(
        summaries[r["id"]]["by_substeps"][str(step)]["attempts"] == 23
        and summaries[r["id"]]["by_substeps"][str(step)]["captures"] == 23
        for step in (32, 64, 128))]
    expected = [dict(id=r["id"], **{k: r["source"][k] for k in GEOMETRY_KEYS}) for r in perfect]
    a.check("exact_eight_parent_geometry_seeds", tuple(r["id"] for r in perfect)
        == PERFECT_PARENT_RECIPES and spec["seeds"] == expected
        and all(set(s) == {"id", *GEOMETRY_KEYS} for s in spec["seeds"]))
    original_rows = {r["id"]: r for r in lines(root / "records.jsonl")}
    provenance = [dict(id=r["id"], parent_source_record_id=r["source"]["id"],
        parent_context_id=r["source"]["context_id"], geometry_sha256=digest(seed))
        for r, seed in zip(perfect, expected)]
    a.check("seed_sources_match_original_native_records", spec["seed_provenance"] == provenance
        and all(r["source"] == original_rows.get(r["source"]["id"])
        and r["source"]["id"] == summaries[r["id"]]["source_record_id"]
        and r["source"].get("native_release") and r["source"].get("route_success")
        and not r["source"].get("invalid") for r in perfect))
    a.check("unchanged_parent_scene_assets_packages", spec["scene"] == parent["scene"]
        and spec["asset_sha256"] == parent["asset_sha256"]
        and spec["packages"] == parent["packages"])
    a.check("unchanged_parent_native_scoring_source", all(
        spec["source_sha256"].get(name) == parent["source_sha256"].get(name)
        for name in CORE_SOURCE))
    a.check("all_parent_core_source_compatibility", all(
        spec["source_sha256"].get(name) == expected
        for name, expected in parent["source_sha256"].items()))
    return parent


def audit_sources(a, study, source, assets, spec, report):
    """Verify actual installed inputs, rather than trusting report metadata."""
    for name, expected in spec["source_sha256"].items():
        a.check("source:" + name, sha256(inside(source, name)) == expected)
    for name, expected in spec["asset_sha256"].items():
        a.check("asset:" + name, sha256(inside(Path(assets).parent, name)) == expected)
    own = Path(__file__).resolve().parents[2]
    for name in (*CORE_SOURCE, "sircus/mjlab/distance_contract.py",
            "sircus/mjlab/distance_geometry.py", "sircus/mjlab/distance_verify.py",
            "sircus/mjlab/neighborhood_verify.py"):
        a.check("matching_independent_logic:" + name,
            sha256(own / name) == spec["source_sha256"].get(name))
    for name, expected in spec["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        a.check("package:" + name, actual == expected)
    manifest = read(study / "evidence-manifest.json")
    actual_files = {str(p.relative_to(study)) for p in study.rglob("*")
        if p.is_file() and p.name not in ("report.json", "evidence-manifest.json")
        and not p.name.startswith("audit")}
    a.check("complete_saved_evidence_manifest", set(manifest) == actual_files
        and manifest == report.get("evidence_sha256"))
    for name, expected in manifest.items():
        a.check("evidence:" + name, sha256(inside(study, name)) == expected)


def table_footprint_clearance(scene, bucket_xy):
    """Signed margin of the outermost bucket radius inside the tabletop XY box."""
    point = np.asarray(bucket_xy, float)
    center = np.asarray(scene["table"]["center_m"][:2], float)
    half = np.asarray(scene["table"]["size_m"][:2], float) / 2
    radius = max(scene["bucket"]["outer_top_diameter_m"],
        scene["bucket"]["outer_bottom_diameter_m"]) / 2
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Finite requested bucket XY required")
    return float(np.min(half - np.abs(point - center) - radius))


def audit_footprint(a, scene, bucket_xy, proof, label):
    point = np.asarray(bucket_xy, float)
    center = np.asarray(scene["table"]["center_m"][:2], float)
    half = np.asarray(scene["table"]["size_m"][:2], float) / 2
    radius = max(scene["bucket"]["outer_top_diameter_m"],
        scene["bucket"]["outer_bottom_diameter_m"]) / 2
    low, high = center - half, center + half
    edges = np.r_[point - low - radius, high - point - radius]
    minimum = float(min(edges))
    expected = dict(version="sircus-full-outer-bucket-table-footprint-v1",
        bucket_xy_m=point.tolist(), outer_radius_m=radius,
        table_min_xy_m=low.tolist(), table_max_xy_m=high.tolist(),
        required_edge_margin_m=.005, edge_clearances_m=edges.tolist(),
        minimum_edge_clearance_m=minimum, numerical_tolerance_m=1e-12,
        passed=bool(minimum >= .005 - 1e-12))
    a.check(label + ":actual_bucket_full_footprint", proof == expected)
    return expected["passed"]


def audit_contexts(a, spec, declared, prepared):
    """Reconstruct source association and exact frozen board branch proofs."""
    from .admission import VERIFIED_PARK_Q10
    lookup = {c["id"]: c for c in declared}
    seeds = {s["id"]: s for s in spec["seeds"]}
    a.check("all_requested_contexts_prepared_once", len(lookup) == len(declared)
        and [p["id"] for p in prepared] == [c["id"] for c in declared])
    for row in prepared:
        label = "context:" + row["id"]
        context = lookup[row["id"]]
        seed = seeds[context["source_anchor_id"]]
        keys = ("source_anchor_id", "bucket_xy_m", "board_height_m", "tilt_deg",
            "distance_id", "ray_id", "distance_m")
        a.check(label + ":declared_geometry_identity", all(row.get(k) == context.get(k)
            for k in keys) and row["source_anchor_geometry_sha256"] == digest(seed)
            and row["wrapper_version"] == "sircus-bucket-distance-frozen-branch-geometry-v1")
        footprint_ok = audit_footprint(a, spec["scene"], context["bucket_xy_m"],
            row["bucket_footprint"], label)
        board_xy = np.asarray(seed["values"][:2], float)
        first = np.asarray(seed["bucket_xy_m"]) - board_xy
        second = np.asarray(context["bucket_xy_m"]) - board_xy
        product = float(np.linalg.norm(first) * np.linalg.norm(second))
        same = bool(np.dot(first, second) > 0
            and abs(first[0] * second[1] - first[1] * second[0]) <= 1e-12 * product)
        a.check(label + ":same_bearing_is_geometry_only",
            row.get("same_source_bearing") == same
            and row.get("bearing_comparison_tolerance") == 1e-12)
        if row.get("board_admitted"):
            q = np.asarray(row.get("q14", []), float)
            proof = dict(version="sircus-fixed-board-gap-window-geometry-v1",
                scene_sha256=digest(spec["scene"]), bucket_xy_m=context["bucket_xy_m"],
                board_world=row.get("board_world"), q14=row.get("q14"),
                park_q10=list(VERIFIED_PARK_Q10))
            a.check(label + ":admitted_frozen_branch", footprint_ok and q.shape == (6,)
                and np.isfinite(q).all() and digest(proof) == row.get("board_branch_sha256"))
            exact = [context["board_height_m"], context["tilt_deg"]] == seed["values"][2:4]
            a.check(label + ":same_ray_exact_source_branch", not (same and exact)
                or row["q14"] == seed["q14"])
            if not same:
                rotated = row.get("rotated_board_derivation", {})
                a.check(label + ":new_bearing_proved_before_freezing",
                    rotated.get("admitted") is True and rotated.get("preserve_known") is False
                    and rotated.get("old_joints_role") == "numerical_continuation_only"
                    and rotated.get("source_anchor_geometry_sha256") == digest(seed)
                    and rotated.get("bucket_xy_m") == context["bucket_xy_m"]
                    and any(c.get("passed") is True for c in rotated.get("branch_checks", [])))
        else:
            a.check(label + ":unresolved_kept_as_unresolved", row.get("q14") is None
                and row.get("board_branch_sha256") is None and bool(row.get("reason")))
        if not footprint_ok:
            a.check(label + ":off_table_not_repositioned", not row.get("board_admitted")
                and row.get("reason") == "bucket_outer_footprint_off_table")


def audit_rows(a, spec, prepared, rows, recipes, repeat_plan, repeat_geometry):
    contexts = {r["id"]: r for r in prepared}
    selected = {r["id"]: r for r in recipes}
    repeat_lookup = {r["repeat_id"]: (r, g)
        for r, g in zip(repeat_geometry["requests"], repeat_geometry["rows"])}
    a.check("unique_sequential_request_ids", [r["id"] for r in rows]
        == [f"distance-shot-{i:06d}" for i in range(len(rows))])
    a.check("frozen_repeat_geometry_request_identity", repeat_geometry["requests"] == repeat_plan
        and len(repeat_lookup) == len(repeat_plan) == len(repeat_geometry["rows"]))
    bad, release_bad, reward_bad = [], [], []
    paired = defaultdict(dict)
    bounds = spec["settings"]["bounds"]
    for row in rows:
        label = row["id"]
        values = np.asarray(row.get("values", []), float)
        geometry = row.get("geometry", {})
        fields = (*GEOMETRY_KEYS, "eligible", "board_branch_sha256",
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


def audit_development_plans(a, study, spec, rows, recipes):
    from . import distance_contract as protocol
    _, targets = protocol.targets(spec["scene"], spec["seeds"])
    coarse_contexts = protocol.contexts(spec["seeds"], targets)
    coarse = [r for r in rows if r["phase"] == "coarse"]
    extended = [r for r in rows if r["phase"] == "far_extension"]
    refined = [r for r in rows if r["phase"] == "refinement"]
    extension_contexts = protocol.extension_contexts(spec["seeds"], targets, coarse)
    contexts = coarse_contexts + extension_contexts
    a.check("declared_and_extension_contexts_frozen_exactly", read(study / "declared-contexts.json") == contexts
        and read(study / "extension-contexts.json") == extension_contexts)
    expected_coarse = protocol.coarse_plan(coarse_contexts)
    expected_extension = protocol.extension_plan(extension_contexts)
    expected_refinement = protocol.refinement_plan(contexts, coarse + extended)
    for phase, observed, expected, name in (
            ("coarse", coarse, expected_coarse, "coarse-plan.json"),
            ("far_extension", extended, expected_extension, "extension-plan.json"),
            ("refinement", refined, expected_refinement, "refinement-plan.json")):
        a.check(phase + ":complete_frozen_requests", read(study / name) == expected
            and len(observed) == len(expected) and all(r["substeps"] == 64
            and all(r.get(k) == v for k, v in p.items()) for r, p in zip(observed, expected)))
    a.check("exact_112_coarse_and_conditional_far_contexts", len(coarse_contexts) == 112
        and len(expected_coarse) == 1904 and len(extension_contexts) <= 96
        and len(expected_extension) <= 1344 and len(expected_refinement) <= 832)
    expected_recipes = protocol.select_recipes([r for r in rows if r["role"] == "development"], targets)
    a.check("recipes_selected_only_from_actual_development_captures", recipes == expected_recipes
        and len(recipes) <= 15 and all(r["source"]["native_release"]
        and r["source"]["route_success"] and not r["source"].get("invalid") for r in recipes))
    repeats = protocol.repeat_plan(recipes)
    a.check("seeded_33_repeat_requests_frozen_before_outcomes",
        read(study / "frozen-repeat-plan.json") == repeats and len(repeats) == 33 * len(recipes))
    actual_repeats = [r for r in rows if r["role"] == "engineering_repeat"]
    controls = [r for r in rows if r["role"] == "control"]
    a.check("no_hidden_retries_resampling_or_dropped_requests", len(rows)
        == 24 + 1904 + len(expected_extension) + len(expected_refinement) + len(repeats) * 3
        <= protocol.MAX_REQUESTS == 5589 and len(controls) == 24
        and Counter((r["repeat_id"], r["substeps"]) for r in actual_repeats)
        == Counter((r["repeat_id"], step) for r in repeats for step in (32, 64, 128))
        and set(r["phase"] for r in rows)
        <= {"controls", "coarse", "far_extension", "refinement", "nominal", "fresh"})
    return contexts


def audit_controls(a, study, spec, rows):
    seeds = spec["seeds"]
    target = next(t for t in spec["settings"]["targets"] if t["id"] == "original-d02")
    expected = [dict(role="control", phase="controls", source_anchor_id=s["id"],
        context_id=f"{target['id']}:{s['id']}", distance_id=target["id"], ray_id=target["ray_id"],
        distance_m=target["distance_m"], gap_m=s["values"][4], repeat_index=i)
        for s in seeds for i in range(3)]
    controls = [r for r in rows if r["role"] == "control"]
    a.check("frozen_exact_24_source_controls", read(study / "control-plan.json") == expected
        and len(controls) == 24 and all(r["substeps"] == 64
        and all(r.get(k) == v for k, v in p.items()) for r, p in zip(controls, expected)))
    a.check("each_seed_reproduces_three_captures_at_original_exact_joints", all(
        len(group := [r for r in controls if r["source_anchor_id"] == seed["id"]]) == 3
        and all(r["route_success"] and r["native_release"] and not r.get("invalid")
        and all(r[k] == seed[k] for k in GEOMETRY_KEYS) for r in group) for seed in seeds))


def audit_targets(a, spec):
    """Reconstruct both requested rays without using the runner's target helper."""
    board = np.asarray(spec["seeds"][0]["values"][:2], float)
    reference = np.asarray(spec["seeds"][0]["bucket_xy_m"], float)
    scene = spec["scene"]
    center = np.asarray(scene["table"]["center_m"][:2], float)
    half = np.asarray(scene["table"]["size_m"][:2], float) / 2
    bounds = np.c_[center - half, center + half]
    radius = max(scene["bucket"]["outer_top_diameter_m"], scene["bucket"]["outer_bottom_diameter_m"]) / 2
    original = reference - board; reference_distance = float(np.linalg.norm(original)); original /= reference_distance
    corner = bounds[:, 0] + radius + .005 - board; corner /= np.linalg.norm(corner)
    definitions = (("original", "Original bucket direction", original,
            [.35, .45, reference_distance, .55, .60, .64, .65]),
        ("corner", "Diagonal toward supported table corner", corner, [.55, .65, .70, .75, .77, .80, .95]))
    targets, rays = [], []
    for name, label, unit, distances in definitions:
        rays.append(dict(id=name, label=label, unit_xy=unit.tolist(),
            heading_deg=float(np.rad2deg(np.arctan2(unit[1], unit[0]))), distances_m=distances))
        for index, distance in enumerate(distances):
            xy = reference.copy() if name == "original" and index == 2 else board + unit * distance
            margin = float(min(np.r_[xy - bounds[:, 0] - radius, bounds[:, 1] - xy - radius]))
            ok = bool(margin >= .005 - 1e-12)
            targets.append(dict(id=f"{name}-d{index:02d}", distance_id=f"{name}-d{index:02d}",
                ray_id=name, distance_m=float(distance), board_xy_m=board.tolist(), bucket_xy_m=xy.tolist(),
                bucket_on_table=ok, table_margin_m=margin,
                reason=None if ok else "bucket_footprint_outside_table"))
    settings = spec["settings"]
    a.check("independent_two_rays_and_all_14_requested_bucket_positions", settings["rays"] == rays
        and settings["targets"] == targets and settings["table_bounds_xy_m"] == bounds.tolist()
        and settings["bucket_outer_radius_m"] == radius and settings["table_bucket_margin_m"] == .005)
    a.check("exact_original_reference_bucket_not_rounded", targets[2]["bucket_xy_m"] == reference.tolist())
    a.check("four_supported_far_targets_and_three_preserved_off_table_targets",
        sum(t["bucket_on_table"] and t["distance_m"] >= .65 for t in targets) == 4
        and sum(not t["bucket_on_table"] for t in targets) == 3)
    return targets


def paired_sensitivity(rows, positions):
    """Independently reconstruct the predeclared64/128 finite engineering gate."""
    paired = defaultdict(dict)
    for row in rows:
        paired[row["repeat_id"]][row["substeps"]] = row
    complete = all(set(pair) == {32, 64, 128} for pair in paired.values())
    agreements = sum(bool(pair[64]["route_success"]) == bool(pair[128]["route_success"])
        for pair in paired.values() if 64 in pair and 128 in pair)
    deltas = [float(np.linalg.norm(positions[pair[64]["id"]][83]
        - positions[pair[128]["id"]][83])) for pair in paired.values()
        if 64 in pair and 128 in pair and pair[64]["phase"] == "nominal"
        and pair[64]["native_release"] and pair[128]["native_release"]]
    nominal = {s: [r for r in rows if r["substeps"] == s and r["phase"] == "nominal"]
        for s in (64, 128)}
    fresh = {s: [r for r in rows if r["substeps"] == s and r["phase"] == "fresh"]
        for s in (64, 128)}
    passed = complete and len(paired) == 33 and agreements >= 30 and len(deltas) == 3 \
        and max(deltas) <= .005 and all(len(nominal[s]) == 3
        and sum(bool(r["route_success"]) for r in nominal[s]) == 3
        and len(fresh[s]) == 30 and sum(bool(r["route_success"]) for r in fresh[s]) >= 27
        for s in (64, 128))
    n32 = [r for r in rows if r["substeps"] == 32 and r["phase"] == "nominal"]
    f32 = [r for r in rows if r["substeps"] == 32 and r["phase"] == "fresh"]
    all_resolution = passed and len(n32) == 3 and all(r["route_success"] for r in n32) \
        and len(f32) == 30 and sum(bool(r["route_success"]) for r in f32) >= 27
    return dict(paired_attempts=len(paired), capture_agreement_64_128=agreements,
        nominal_position_delta_350ms_max_m=max(deltas) if deltas else None,
        paired_engineering_gate=bool(passed), all_resolution_repeat_gate=bool(all_resolution))


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
    a.check("all_14_targets_reported_in_order", [t["id"] for t in report["targets"]]
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


def audit(study, source, assets, parent_study):
    from . import distance_contract as protocol
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
    a.check("unchanged_distinct_timestep_native_contracts", spec["physics_by_substeps"]
        == {str(s): physics_settings(s) for s in (32, 64, 128)})
    audit_parent(a, spec, parent_study)
    audit_sources(a, study, source, assets, spec, report)
    audit_targets(a, spec)
    declared = audit_development_plans(a, study, spec, rows, recipes)
    prepared = read(study / "prepared-contexts.json")
    repeats, gates = read(study / "frozen-repeat-plan.json"), read(study / "repeat-geometry.json")
    audit_contexts(a, spec, declared, prepared)
    audit_controls(a, study, spec, rows)
    audit_rows(a, spec, prepared, rows, recipes, repeats, gates)
    audit_options(a, study, report)
    positions, rescored = audit_batches(a, study, spec, rows)
    audit_report(a, spec, report, declared, prepared, rows, recipes, positions)
    failures = [r for r in a.checks if not r["passed"]]
    return dict(version=VERSION, status="failed" if failures else "passed", contract_fingerprint=fingerprint,
        checks_count=len(a.checks), checks=a.checks, failures=failures, records=len(rows),
        independently_rescored_throws=rescored, auditor_source_sha256=sha256(__file__),
        scope="Saved source/geometry/scoring/provenance CPU audit; no new dynamics, policy or physical validation")


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
    if result["status"] != "passed":
        raise RuntimeError("Distance evidence audit failed")


if __name__ == "__main__":
    main()
