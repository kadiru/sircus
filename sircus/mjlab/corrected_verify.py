"""Independent CPU evidence checks for the corrected-contact spatial study.

No physics integration, GPU context, policy inference, hardware access or source
mutation is performed. Native outcomes are reconstructed from saved observations;
strict endpoint eligibility and common-staging connectivity are separate claims.
"""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .verify_expansion import Audit, digest, sha256
from .verify_contact import SCORER_KEYS, NUMERIC_SCORER_KEYS

VERSION = "sircus-corrected-grid-evidence-audit-v1"


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def audit_native_batch(audit, scene, records, arrays, record_ids, *, label):
    """Rescore every released row, preserving the NPZ's explicit ID association."""
    from .contract import ScoreStream, FORCE_NAMES

    ids = list(map(str, record_ids))
    expected_ids = [r["id"] for r in records]
    audit.check(label + "_record_id_identity", len(set(ids)) == len(ids) == len(records) and ids == expected_ids)
    shape = all(name in arrays and arrays[name].shape == (960, len(records), width)
        and np.isfinite(arrays[name]).all() for name, width in
        (("position_m", 3), ("velocity_m_s", 3), ("forces_N", 6)))
    audit.check(label + "_observation_shapes", shape)
    if not shape or ids != expected_ids:
        return
    audit.check(label + "_released_only", all(r.get("native_release") is True and not r.get("invalid") for r in records))
    audit.check(label + "_nonnegative_force_categories", bool((arrays["forces_N"] >= 0).all()))
    audit.check(label + "_corrected_articulated_protocol", all(r.get("contact_protocol") == dict(
        substeps=64, board_mode="articulated", gravity_mode="passive_only") for r in records))
    mismatch = []
    for index, row in enumerate(records):
        scorer = ScoreStream(scene, row["bucket_xy_m"], row["board_normal"])
        for p, v, f in zip(arrays["position_m"][:, index], arrays["velocity_m_s"][:, index], arrays["forces_N"][:, index]):
            scorer.step(p, v, dict(zip(FORCE_NAMES, f)))
            if scorer.finished:
                break
        actual = scorer.result()
        discrete = all(actual[k] == row[k] for k in SCORER_KEYS)
        numeric = all((actual[k] is None and row[k] is None) or (actual[k] is not None and row[k] is not None
            and np.isclose(actual[k], row[k], atol=1e-3, rtol=1e-5)) for k in NUMERIC_SCORER_KEYS)
        if not discrete or not numeric:
            mismatch.append(row["id"])
    audit.check(label + "_independent_scalar_rescore", not mismatch, mismatch[:10])


def audit_split_identity(audit, splits, grid_layouts, historical_xy):
    """Validate the declared new coordinates and five-millimetre exclusion radius."""
    declared = {key: np.asarray(splits.get(key, []), float) for key in ("train", "validation", "test")}
    good = all(a.ndim == 2 and a.shape[1] == 2 and len(a) > 0 and np.isfinite(a).all() for a in declared.values())
    audit.check("finite_nonempty_future_splits", good)
    if not good:
        return
    sets = {key: {tuple(x) for x in value} for key, value in declared.items()}
    audit.check("future_splits_unique_and_disjoint", all(len(sets[k]) == len(declared[k]) for k in sets)
        and sets["train"].isdisjoint(sets["validation"]) and sets["train"].isdisjoint(sets["test"])
        and sets["validation"].isdisjoint(sets["test"]))
    expected_grid = {tuple(row["bucket_xy_m"]) for row in grid_layouts}
    audit.check("training_grid_is_the_declared_development_grid", sets["train"] == expected_grid)
    history = {tuple(x) for x in historical_xy}
    audit.check("new_validation_test_exclude_historical_coordinates", sets["validation"].isdisjoint(history)
        and sets["test"].isdisjoint(history))
    forbidden = np.asarray(historical_xy, float).reshape(-1, 2)
    separated = True
    for role in ("validation", "test"):
        for point in declared[role]:
            if len(forbidden) and np.linalg.norm(forbidden - point, axis=1).min() < .005:
                separated = False
            forbidden = np.r_[forbidden, point[None]]
    audit.check("future_positions_separated_by_five_millimetres", separated)
    forbidden = np.asarray(historical_xy, float).reshape(-1, 2)
    regenerated = {}
    for role, count, seed in (("validation", 30, 20261228), ("test", 100, 20261229)):
        rng = np.random.default_rng(seed); chosen = []
        for _ in range(100000):
            if len(chosen) == count:
                break
            point = rng.uniform([.1, -.65], [1.24, .25])
            previous = np.r_[forbidden, np.asarray(chosen).reshape(-1, 2)]
            if len(previous) and np.linalg.norm(previous - point, axis=1).min() < .005:
                continue
            chosen.append(point.tolist())
        regenerated[role] = chosen
        forbidden = np.r_[forbidden, np.asarray(chosen)]
    # Small pure-helper fixtures need not carry the production 30/100 split.
    if len(declared["validation"]) == 30 and len(declared["test"]) == 100:
        audit.check("future_splits_independently_regenerated", all(np.array_equal(declared[role], regenerated[role]) for role in regenerated))


def audit_grid_denominators(audit, layouts, cells, *, expected=35):
    """All declared cells remain visible even when no action is admitted."""
    ids = [row["id"] for row in layouts]
    result_ids = [row["id"] for row in cells]
    xy = [tuple(row["bucket_xy_m"]) for row in layouts]
    audit.check("exact_35_development_layouts", len(ids) == len(set(ids)) == len(set(xy)) == expected)
    axes = [{point[k] for point in xy} for k in (0, 1)]
    audit.check("complete_seven_by_five_bucket_grid", len(axes[0]) == 7 and len(axes[1]) == 5
        and set(xy) == {(x, y) for x in axes[0] for y in axes[1]})
    audit.check("all_development_cells_in_report", Counter(ids) == Counter(result_ids))
    coordinates = {row["id"]: row["bucket_xy_m"] for row in layouts}
    audit.check("cell_bucket_identity", all(row["id"] in coordinates and row["bucket_xy_m"] == coordinates[row["id"]] for row in cells))


def audit_home_layouts(audit, study, spec, report):
    """Retain the original admission categories; supplementary probes are separate."""
    study = Path(study); records = read(study / "common-home-layouts.json")
    audit.check("common_home_layout_file_hash", sha256(study / "common-home-layouts.json") == spec["common_home_layouts_sha256"]
        == report["common_home_layouts_sha256"])
    declared = {r["id"]: r for r in spec["grid_layouts"]}
    audit.check("all_35_initial_home_layouts_preserved", Counter(r["id"] for r in records) == Counter(declared.keys()))
    summaries = {}; bad = []
    for row in records:
        raw = row["evidence"]; claimed = row["home_geometry"]
        fixed = [c for c in raw.get("margin_contacts", []) if len(c.get("pair", [])) == 2
            and c.get("distance_m", 0) < 0 and any("/link_base/" in n for n in c["pair"])
            and any(n.startswith("bucket/") for n in c["pair"])]
        category = ("home_clear" if raw.get("passed") else "fixed_base_bucket_penetration" if fixed else
            "home_unresolved" if raw.get("reason") == "common_home_unresolved" else
            "staging_clearance_unproven" if raw.get("reason") == "unproven_clearance" else "staging_clearance_blocked")
        expected = dict(passed=bool(raw.get("passed")), reason=raw.get("reason"), category=category,
            minimum_clearance_m=raw.get("minimum_clearance_m"), closest_pair=raw.get("closest_pair"),
            fixed_base_penetrations=fixed, common_home_sha256=raw.get("common_home_sha256"))
        if claimed != expected or row["bucket_xy_m"] != declared[row["id"]]["bucket_xy_m"] or claimed.get("common_home_sha256") != spec["common_home_sha256"]:
            bad.append(row["id"])
        summaries[row["id"]] = claimed
    audit.check("initial_home_categories_bound_to_actual_evidence", not bad, bad)
    audit.check("report_preserves_initial_home_layout_summaries", all(c.get("home_geometry") == summaries[c["id"]] for c in report["cells"]))
    audit.check("home_clear_position_denominator", report["counts"]["home_clear_positions"] == sum(bool(r["home_geometry"]["passed"]) for r in records)
        and report["counts"]["grid_positions"] == 35)


def audit_geometry_caches(audit, rows, caches):
    """Every outcome retains its exact request and selected CPU joint branches."""
    associated = []; bad = []; clearance_bad = []
    for row in rows:
        name = row.get("geometry_cache_file"); index = row.get("geometry_cache_row")
        cache = caches.get(name)
        if cache is None or not isinstance(index, int) or not 0 <= index < len(cache["record_ids"]):
            bad.append(row["id"]); continue
        associated.append(str(cache["record_ids"][index]))
        expected = dict(values=cache["values"][index], bucket_xy_m=cache["bucket"][index],
            q10=cache["q10"][index, 0], q14=cache["q14"][index, 0])
        if str(cache["record_ids"][index]) != row["id"] or not all(np.array_equal(row[k], v) for k, v in expected.items()):
            bad.append(row["id"])
        if row.get("geometry_eligible") != bool(cache["mask"][index, 0]) or row.get("geometry_reason") != int(cache["reasons"][index, 0]):
            bad.append(row["id"])
        if cache["mask"][index, 0] and cache["minimum_clearance_lower_bound_m"][index, 0] < .005:
            clearance_bad.append(row["id"])
    audit.check("all_record_request_joint_cache_associations", not bad, bad[:10])
    all_cached = [str(i) for c in caches.values() for i in c["record_ids"]]
    audit.check("no_cached_request_or_outcome_omitted", Counter(all_cached) == Counter(r["id"] for r in rows)
        and len(all_cached) == len(set(all_cached)))
    audit.check("strict_admitted_clearance_lower_bounds", not clearance_bad, clearance_bad[:10])


def independently_select(values, eligible, budget, seed):
    """Outcome-free geometric selection, reconstructed without runner helpers."""
    ids = np.flatnonzero(eligible); rng = np.random.default_rng(seed)
    ids = ids[rng.permutation(len(ids))]
    if not len(ids):
        return []
    low = np.asarray([.30, -.50, .12, 8., .05]); high = np.asarray([1.04, .35, .60, 60., .65])
    x = (np.asarray(values) - low) / (high - low)
    chosen = [int(ids[0])]; distance = np.linalg.norm(x[ids] - x[chosen[0]], axis=1)
    while len(chosen) < min(budget, len(ids)):
        distance[np.isin(ids, chosen)] = -1
        j = int(np.argmax(distance)); chosen.append(int(ids[j]))
        distance = np.minimum(distance, np.linalg.norm(x[ids] - x[ids[j]], axis=1))
    return chosen


def audit_proposals(audit, spec, bank, plan):
    """Regenerate fixed bank transforms and pre-outcome target-relative proposals."""
    from .bank_geometry import build_candidates
    from .expand_far import far_proposals

    layouts = spec["grid_layouts"]
    buckets = np.asarray([r["bucket_xy_m"] for r in layouts])
    bank_values, metadata = build_candidates(bank, buckets, spec["scene"])
    extra = spec["budgets"]["extra_proposals_per_cell"]
    audit.check("complete_fixed_proposal_denominator", len(plan) == 35 * (272 + extra)
        and len({p["id"] for p in plan}) == len(plan))
    mismatches = []
    for index, layout in enumerate(layouts):
        chosen = [p for p in plan if p["cell_id"] == layout["id"]]
        rng = np.random.default_rng(20261127 + index)
        more, _ = far_proposals(buckets[index], extra, rng, spec["scene"], seeds=()) if extra else (np.empty((0, 5)), [])
        expected = np.r_[bank_values[index], more]
        if len(chosen) != len(expected) or not np.array_equal([p["values"] for p in chosen], expected):
            mismatches.append(layout["id"]); continue
        if any(p["cell_index"] != index or p["candidate_index"] != i or p["bucket_xy_m"] != layout["bucket_xy_m"]
            or p["id"] != layout["id"] + "-" + p["candidate_id"] for i, p in enumerate(chosen)):
            mismatches.append(layout["id"])
        if [p["candidate_id"] for p in chosen[:272]] != [m["candidate_id"] for m in metadata]:
            mismatches.append(layout["id"])
    audit.check("all_geometry_proposals_regenerated_without_outcomes", not mismatches, mismatches)


def audit_search_accounting(audit, spec, report, plan, audited_plan, selected, rows):
    lookup = {r["id"]: r for r in rows}; audited = {r["id"]: r for r in audited_plan}
    search = [r for r in rows if r["phase"] == "search"]
    audit.check("each_search_request_has_one_outcome_status", Counter(p["id"] for p in plan) == Counter(r["id"] for r in search)
        and len(lookup) == len(rows) and set(audited) == {p["id"] for p in plan})
    selected_ids = {r["id"] for r in selected}; expected_selected = set()
    cell_claims = {c["id"]: c for c in report["cells"]}
    for index, layout in enumerate(spec["grid_layouts"]):
        original = [p for p in plan if p["cell_id"] == layout["id"]]
        derived = [audited[p["id"]] for p in original]
        eligible = np.asarray([r["geometry_eligible"] for r in derived])
        chosen = independently_select([p["values"] for p in original], eligible,
            spec["budgets"]["native_per_cell"], spec["seed"] + index)
        expected_selected.update(original[i]["id"] for i in chosen)
        matching = [r for r in search if r["cell_id"] == layout["id"]]
        claim = cell_claims[layout["id"]]
        audit.check(layout["id"] + "_geometry_counts", claim["geometry_requests"] == len(original)
            and claim["strict_eligible"] == int(eligible.sum()) and claim["geometry_rejected"] == int((~eligible).sum())
            and claim["skipped_eligible"] == int(eligible.sum()) - len(chosen))
        native = [r for r in matching if r["id"] in selected_ids]
        audit.check(layout["id"] + "_native_search_counts", claim["search_attempts"] == len(native)
            and claim["search_captures"] == sum(bool(r["route_success"]) for r in native))
    audit.check("selected_search_plan_is_outcome_free_geometry_selection", selected_ids == expected_selected and len(selected_ids) == len(selected))
    audit.check("unselected_eligible_not_mislabelled_native_misses", all(r.get("status") == "not_selected_budget"
        and r.get("native_release") is False and r.get("route_success") is False and r.get("invalid") is None
        for r in search if r["geometry_eligible"] and r["id"] not in selected_ids))
    audit.check("geometry_rejections_never_release_or_capture", all(r.get("native_release") is False
        and r.get("route_success") is False and bool(r.get("invalid")) for r in rows if not r["geometry_eligible"]))
    native_attempts = len(selected) + sum(r["phase"] != "search" for r in rows)
    audit.check("report_total_native_and_record_denominators", report["records"] == len(rows)
        and report["native_attempts"] == native_attempts
        and report["native_releases"] == sum(bool(r.get("native_release")) for r in rows))


def audit_confirmations(audit, study, spec, report, rows):
    """Qualification comes from fresh observations and fixed-home certificates."""
    study = Path(study); lookup = {r["id"]: r for r in rows}
    home = read(study / "common-home.json"); payload = dict(home); hp = payload.pop("sha256", None)
    audit.check("fixed_home_identity_frozen_in_contract", digest(payload) == hp == spec["common_home_sha256"]
        and spec["common_home"] == home and home["scene_sha256"] == digest(spec["scene"]))
    entries = [e for c in report["cells"] for e in c["confirmed_sources"]]
    expected_sources = []
    for cell in spec["grid_layouts"]:
        captured = [r for r in rows if r["phase"] == "search" and r["cell_id"] == cell["id"] and r.get("route_success")]
        captured.sort(key=lambda r: (bool(r.get("route_success")), int(r.get("route_quality", 0)),
            float(r["minimum_mouth_clearance_mm"] if r.get("minimum_mouth_clearance_mm") is not None else -1e9),
            -float(r.get("closest_opening_distance_m", 2)), r["id"]), reverse=True)
        expected_sources.extend(r["id"] for r in captured[:spec["budgets"]["top"]])
    attempts = report["transit_attempts"]
    audit.check("every_declared_top_capture_gets_transit_attempt_including_failures", [r["source_id"] for r in attempts] == expected_sources)
    passed_transit_ids = []
    for attempt in attempts:
        sid = attempt["source_id"]; source = lookup[sid]; proof = read(study / attempt["file"])
        body = proof.get("report", {})
        identity = dict(version="sircus-fixed-common-home-transit-admission-v1", scene_sha256=digest(spec["scene"]),
            common_home_sha256=hp, **{k: source[k] for k in ("bucket_xy_m", "values", "q10", "q14")})
        audit.check(sid + "_retained_transit_outcome_identity", proof.get("passed") == attempt.get("passed")
            and proof.get("common_home_sha256") == hp and proof.get("request_sha256") == digest(identity)
            and body.get("request") == identity and attempt["cell_id"] == source["cell_id"])
        if attempt.get("passed"):
            passed_transit_ids.append(sid)
    audit.check("every_passed_transit_source_has_full_nominal_and_confirmation_group", Counter(e["source"]["id"] for e in entries) == Counter(passed_transit_ids))
    expected_qualified = []; observed_confirmation_ids = []; source_ids = set()
    for entry in entries:
        source = entry["source"]; sid = source["id"]; source_ids.add(sid)
        audit.check(sid + "_new_search_source_identity", lookup.get(sid) == source and source.get("phase") == "search"
            and source.get("native_release") is True and source.get("route_success") is True)
        path = study / entry["transit_file"]; proof = read(path); body = proof.get("report", {})
        identity = dict(version="sircus-fixed-common-home-transit-admission-v1", scene_sha256=digest(spec["scene"]),
            common_home_sha256=hp, **{k: source[k] for k in ("bucket_xy_m", "values", "q10", "q14")})
        required_scope = "fixed_common_home_to_exact_selected_shot_and_reverse_sampled_geometry"
        audit.check(sid + "_exact_source_fixed_home_proof", proof.get("passed") is True and body.get("passed") is True
            and proof.get("common_home_sha256") == hp and proof.get("request_sha256") == digest(identity)
            and body.get("request") == identity and proof.get("proof_scope") == body.get("proof_scope") == required_scope
            and body.get("returned_home_joint_max_error_rad") == 0. and body.get("exact_selected_joint_max_error_rad") == 0.
            and body.get("minimum_clearance_lower_bound_m", 0) >= .005)
        nominal_ids, fresh_ids = entry["nominal_ids"], entry["fresh_ids"]
        observed_confirmation_ids.extend([*nominal_ids, *fresh_ids])
        nominal = [lookup[i] for i in nominal_ids]; fresh = [lookup[i] for i in fresh_ids]
        audit.check(sid + "_all_three_and_twenty_attempts", len(set(nominal_ids)) == len(nominal) == 3
            and len(set(fresh_ids)) == len(fresh) == 20 and set(nominal_ids).isdisjoint(fresh_ids)
            and all(r.get("confirmation_source_id") == sid for r in [*nominal, *fresh])
            and all(r["phase"] == "nominal" for r in nominal) and all(r["phase"] == "confirmation" for r in fresh))
        audit.check(sid + "_nominal_exact_geometry", all(r["values"] == source["values"] and r["bucket_xy_m"] == source["bucket_xy_m"] for r in nominal))
        cell = next(c for c in spec["grid_layouts"] if c["id"] == source["cell_id"])
        matches = [p for p in study.glob(f'confirmation-plan-{cell["id"]}-source-*.json') if read(p)["source_id"] == sid]
        audit.check(sid + "_one_confirmation_plan", len(matches) == 1)
        if matches:
            declared = read(matches[0]); rank = int(matches[0].stem.rsplit("-", 1)[1])
            seed = 20261127 + 1000 + 35 * cell["row"] + cell["column"] * 2 + rank
            rng = np.random.default_rng(seed)
            expected_values = [np.asarray(source["values"]) + rng.normal(0, [.001, .001, .002, .35, .002]) for _ in range(20)]
            audit.check(sid + "_fresh_noise_independently_regenerated", entry["noise_seed"] == declared["noise_seed"] == seed
                and [r["id"] for r in declared["nominal"]] == nominal_ids and [r["id"] for r in declared["fresh"]] == fresh_ids
                and all(np.array_equal(r["values"], expected) and r["bucket_xy_m"] == source["bucket_xy_m"]
                    for r, expected in zip(fresh, expected_values)))
        for row in [*nominal, *fresh]:
            if not row["geometry_eligible"]:
                continue
            variation = read(study / row["motion_admission_file"])
            request = {k: row[k] for k in ("bucket_xy_m", "values", "q10", "q14")}
            motion = variation.get("passed") is True
            audit.check(row["id"] + "_variation_exact_request_and_home", variation.get("common_home_sha256") == hp
                and variation.get("nominal_request_sha256") == proof["request_sha256"]
                and variation.get("request") == request and variation.get("request_sha256") == digest(request)
                and row.get("motion_admitted") == motion
                and (not motion or (len(variation.get("local_edges", [])) == 4
                    and all(e.get("passed") and e.get("minimum_clearance_m", 0) >= .005 for e in variation["local_edges"]))))
            audit.check(row["id"] + "_failed_motion_never_releases_or_scores", motion or
                (row.get("native_release") is False and row.get("route_success") is False and bool(row.get("invalid"))))
        actual = bool(proof.get("passed") and all(r.get("native_release") and r.get("route_success") for r in nominal)
            and sum(bool(r.get("native_release") and r.get("route_success")) for r in fresh) >= 18)
        audit.check(sid + "_qualification_and_full_denominators", entry["qualified"] == actual
            and entry["nominal_attempts"] == 3 and entry["fresh_attempts"] == 20
            and entry["nominal_captures"] == sum(r["route_success"] for r in nominal)
            and entry["fresh_captures"] == sum(r["route_success"] for r in fresh))
        if actual:
            expected_qualified.append(entry)
    audit.check("no_confirmation_failure_or_source_group_omitted", Counter(observed_confirmation_ids) == Counter(r["id"] for r in rows if r["phase"] != "search"))
    audit.check("qualified_sources_are_exact_fresh_qualifications", report["qualified_sources"] == expected_qualified)
    catalogue = read(study / "confirmed-catalogue.json"); payload = dict(catalogue); fp = payload.pop("fingerprint", None)
    audit.check("new_catalogue_fingerprint_and_identity", digest(payload) == fp == report["catalogue_fingerprint"]
        and catalogue["contract_fingerprint"] == spec["fingerprint"] and catalogue["physics"] == spec["physics"]
        and catalogue["sources"] == expected_qualified and sha256(study / "confirmed-catalogue.json") == report["catalogue_sha256"])
    audit.check("all_35_coverage_denominators", report["cells_with_native_capture"] == sum(c["search_captures"] > 0 for c in report["cells"])
        and report["cells_with_qualified_source"] == sum(any(e["qualified"] for e in c["confirmed_sources"]) for c in report["cells"]))


def verify(study, source, assets, source_studies, bank, parent_study, contact_reference, historical_studies=()):
    study, source, assets, bank, parent_study, contact_reference = map(Path,
        (study, source, assets, bank, parent_study, contact_reference))
    spec, report = read(study / "contract.json"), read(study / "report.json")
    rows = lines(study / "shots.jsonl"); plan = read(study / "plan.json")
    a = Audit(); payload = dict(spec); fp = payload.pop("fingerprint", None)
    a.check("contract_fingerprint", digest(payload) == fp == report.get("contract_fingerprint"))
    a.check("completed_corrected_version_without_training", spec.get("version") == report.get("version") == "sircus-corrected64-spatial-development-v1"
        and report.get("status") == report.get("phase") == "completed" and report.get("no_training") is True)
    physics = spec["physics"]
    expected_physics = dict(version="sircus-articulated-passive-contact-15360hz-v1", board_mode="articulated",
        gravity_mode="passive_only", substeps=64, native_dt_s=1 / 15360, scoring_dt_s=1 / 240,
        horizon_s=4., solref=[.004, .15], contact_friction=.35, solver="newton", solver_iterations=100,
        solver_tolerance=1e-8, cone="pyramidal", integrator="implicitfast")
    a.check("explicit_new_physics_contact_and_material", all(physics.get(k) == v for k, v in expected_physics.items()) and report.get("physics") == physics)
    a.check("strict_versioned_admission", spec.get("geometry_version") == "sircus-corrected-strict-endpoint-admission-v1"
        and spec.get("transit_version") == "sircus-fixed-common-home-transit-admission-v1"
        and spec["strict_admission"].get("clearance_m") == .005 and spec["strict_admission"].get("carried_ball") is True)
    a.check("fixed_confirmation_budgets", spec["budgets"]["nominal_repeats"] == 3
        and spec["budgets"]["fresh_confirmations"] == 20 and spec["budgets"]["required_fresh_captures"] == 18
        and spec["seed"] == 20261127 and spec["noise_std"] == [.001, .001, .002, .35, .002])
    options = report.get("effective_native_options", {})
    a.check("recorded_effective_native_solver_options", options.get("timestep") == 1 / 15360
        and options.get("solver") == 2 and options.get("cone") == 0 and options.get("integrator") == 3
        and options.get("iterations") == 100 and options.get("tolerance") == 1e-8 and options.get("gravity") == [0., 0., -9.81])
    a.check("recorded_runtime_packages_match_pinned_auditor", all(importlib.metadata.version(name) == version for name, version in spec["packages"].items())
        and spec["packages"].get("mjlab") == "1.6.0" and spec["packages"].get("mujoco") == "3.11.0")
    for name, expected in spec["source_sha256"].items():
        path = (source / name).resolve()
        a.check("executed_source:" + name, path.is_relative_to(source.resolve()) and path.is_file() and sha256(path) == expected)
    for name, expected in spec["asset_sha256"].items():
        path = (assets.parent / name).resolve()
        a.check("native_asset:" + name, path.is_relative_to(assets.parent.resolve()) and path.is_file() and sha256(path) == expected)
    auditor_root = Path(__file__).resolve().parents[2]
    for name in ("sircus/mjlab/contract.py", "sircus/isaac/geometry.py", "sircus/isaac/rl_contract.py", "sircus/isaac/entry_quality.py", "sircus/isaac/board_route.py"):
        a.check("independent_auditor_scoring_source:" + name, sha256(auditor_root / name) == spec["source_sha256"].get(name))
    parent = read(parent_study / "contract.json")
    a.check("original_parent_geometry_identity", parent["fingerprint"] == spec["parent_fingerprint"] and parent["scene"] == spec["scene"])
    reference = read(contact_reference)
    reference_spec = read(contact_reference.parent / "contract.json"); reference_payload = dict(reference_spec); reference_fp = reference_payload.pop("fingerprint", None)
    a.check("prior_corrected_reference_identity", sha256(contact_reference) == spec["contact_reference_sha256"]
        and reference["contract_fingerprint"] == spec["contact_reference_fingerprint"] == reference_fp
        and digest(reference_payload) == reference_fp and reference["status"] == "completed")
    for name in ("contact_env.py", "contact_model.py", "model.py", "learning_score.py", "contract.py"):
        a.check("unchanged_reference_contact_source:" + name,
            reference_spec["source_sha256"].get(name) == spec["source_sha256"].get("sircus/mjlab/" + name))
    original_bank = read(bank); copied_bank = read(study / "geometry-bank.json")
    bank_payload = dict(original_bank); bank_fp = bank_payload.pop("sha256", None)
    a.check("frozen_geometry_bank_identity", copied_bank == original_bank and len(original_bank["templates"]) == 17
        and digest(bank_payload) == bank_fp == spec["bank_geometry_fingerprint"] and sha256(bank) == spec["bank_file_sha256"])
    audit_proposals(a, spec, copied_bank, plan)
    source_paths = {Path(s).name: Path(s) for s in source_studies}; originals = {}; historical = []
    a.check("all_supplied_historical_studies_identified", set(source_paths) == {s["namespace"] for s in spec["source_studies"]})
    for s in spec["source_studies"]:
        path = source_paths[s["namespace"]] / "shots.jsonl"
        a.check("prior_geometry_source:" + s["namespace"], sha256(path) == s["records_sha256"])
        original_rows = lines(path)
        originals[s["namespace"]] = {r["id"]: r for r in original_rows}
        historical.extend(r["bucket_xy_m"] for r in original_rows if isinstance(r.get("bucket_xy_m"), list) and len(r["bucket_xy_m"]) == 2)
    history_paths = {Path(s).name: Path(s) for s in historical_studies}
    a.check("all_historical_split_sources_identified", set(history_paths) == {s["namespace"] for s in spec.get("historical_studies", [])})
    for history in spec.get("historical_studies", []):
        root = history_paths[history["namespace"]]
        for name, expected in history["files_sha256"].items():
            a.check("historical_split:" + history["namespace"] + "/" + name, sha256(root / name) == expected)
            data = read(root / name)
            for key in ("train_layouts", "validation_layouts", "test_layouts", "train", "validation", "test"):
                points = data.get(key)
                if isinstance(points, list) and all(isinstance(x, list) and len(x) == 2 for x in points):
                    historical.extend(points)
    for key in ("train_layouts", "validation_layouts", "test_layouts"):
        historical.extend(parent.get(key, []))
    historical.extend(r["bucket_xy_m"] for r in spec["grid_layouts"])
    historical = sorted({tuple(map(float, p)) for p in historical})
    a.check("historical_coordinate_exclusion_identity", np.array_equal(spec["historical_coordinates"], historical))
    audit_grid_denominators(a, spec["grid_layouts"], report["cells"])
    audit_home_layouts(a, study, spec, report)
    audit_split_identity(a, spec["future_splits"], spec["grid_layouts"], historical)
    a.check("future_split_fixed_sample_counts", len(spec["future_splits"]["train"]) == 35
        and len(spec["future_splits"]["validation"]) == 30 and len(spec["future_splits"]["test"]) == 100)
    hints_bad = []
    for request in plan:
        hint = request["geometry_hint"]
        original = originals.get(hint["source_namespace"], {}).get(hint["source_id"])
        allowed = {"source_id", "source_namespace", "source_record_sha256", "values", "q10", "q14", "bucket_xy_m", "use"}
        if original is None or set(hint) != allowed or digest(original) != hint["source_record_sha256"] or not all(hint[k] == original[k] for k in ("values", "q10", "q14", "bucket_xy_m")):
            hints_bad.append(request["id"])
    a.check("historical_hints_geometry_only_no_outcomes", not hints_bad, hints_bad[:10])
    caches = {}
    for file in sorted(study.glob("endpoint-batch-*.npz")):
        with np.load(file, allow_pickle=False) as data:
            caches[file.name] = {k: data[k].copy() for k in data.files}
    audit_geometry_caches(a, rows, caches)
    audit_search_accounting(a, spec, report, plan, read(study / "audited-plan.json"), read(study / "selected-search-plan.json"), rows)
    audit_confirmations(a, study, spec, report, rows)
    by_id = {r["id"]: r for r in rows}; rescored = []
    for file in sorted(study.glob("native-batch-*.npz")):
        with np.load(file, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        ids = list(map(str, arrays["record_ids"]))
        selected = [by_id[i] for i in ids]
        a.check(file.name + "_native_cache_membership", all(r.get("native_cache_file") == file.name and r.get("native_cache_row") == i for i, r in enumerate(selected)))
        audit_native_batch(a, spec["scene"], selected, arrays, ids, label=file.name)
        rescored.extend(ids)
    a.check("every_native_release_rescored_exactly_once", Counter(rescored) == Counter(r["id"] for r in rows if r.get("native_release")))
    for name, expected in report.get("evidence_sha256", {}).items():
        path = (study / name).resolve()
        a.check("saved_evidence:" + name, path.is_relative_to(study.resolve()) and path.is_file() and sha256(path) == expected)
    failures = [c for c in a.checks if not c["passed"]]
    return dict(version=VERSION, status="passed" if not failures else "failed", failures=failures,
        checks=a.checks, checks_count=len(a.checks), contract_fingerprint=spec["fingerprint"],
        auditor_source_sha256=sha256(__file__), executed_source_directory=str(source),
        independently_rescored_throws=len(rescored),
        limitations=["Stored-array scoring and provenance audit; native ball dynamics are not independently rerun.",
            "The35gridpositionsaredevelopment;futurevalidation/testcoordinatesarefrozenandnottestedhere.",
            "Common-staging certificate identities, endpoints and admission claims checked; full native geometric path checks are performed by the separately versioned admission runner, not repeated here."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "assets", "bank", "parent-study", "contact-reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-study", type=Path, action="append", required=True)
    parser.add_argument("--historical-study", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = verify(args.study, args.source, args.assets, args.source_study, args.bank, args.parent_study, args.contact_reference, args.historical_study)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_count", "independently_rescored_throws")}))
    if result["failures"]:
        raise RuntimeError("Corrected evidence audit failed: " + str([c["name"] for c in result["failures"]]))


if __name__ == "__main__":
    main()
