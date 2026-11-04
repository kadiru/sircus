"""Independent CPU evidence audit of the exact user-requested shallow candidate.

Literal requested parameters, discovery/selection and seeded repeats are rebuilt
here; physics/scoring, ancestry and cache checks reuse unchanged audited helpers.
No simulator, GPU, policy deployment or historical-evidence mutation occurs.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from . import lowboard_verify as shared
from .neighborhood_verify import audit_batches, inside, lines, read
from .verify_expansion import Audit, digest, sha256

VERSION = "sircus-shallow-candidate-evidence-audit-v1"


def audit_sources(a, study, source, assets, spec, report):
    shared.audit_sources(a, study, source, assets, spec, report)
    own = Path(__file__).resolve().parents[2]
    for name in ("sircus/mjlab/shallow_contract.py", "sircus/mjlab/shallow_study.py",
            "sircus/mjlab/shallow_verify.py", "sircus/mjlab/lowboard_study.py"):
        a.check("matching_exact_candidate_logic:" + name,
            sha256(own / name) == spec["source_sha256"].get(name))


def audit_candidate_and_contexts(a, spec, parent):
    seeds, settings, scene = spec["seeds"], spec["settings"], spec["scene"]
    targets = [t for t in parent["settings"]["targets"]
        if t["id"] in ("corner-d01", "corner-d02", "corner-d03", "corner-d04")]
    corner = copy.deepcopy(next(r for r in parent["settings"]["rays"] if r["id"] == "corner"))
    corner["distances_m"] = [t["distance_m"] for t in targets]
    a.check("four_exact_same_supported_parent_targets", len(targets) == 4
        and settings["targets"] == targets and settings["rays"] == [corner]
        and all(t["bucket_on_table"] for t in targets))
    board = np.asarray(seeds[0]["values"][:2], float)
    radius = scene["ball"]["radius_m"]
    ceiling = float(.364 + np.sqrt(.842 ** 2 - float(board @ board)) - .095 - .09)
    release = .380 + .3868 + radius
    candidate = dict(board_height_m=.380, tilt_deg=28., gap_m=.3868,
        release_center_height_m=release, ball_radius_m=radius, release_headroom_m=ceiling - release,
        gap_definition="Vertical clearance from released ball bottom to board-center height",
        input_precision="Exact 386.8 mm user input; not the earlier 386.824338 mm CPU proposal")
    a.check("literal380mm_28degrees_386point8mm_candidate", settings["candidate"] == candidate)
    a.check("new_seed_noise_and_finite_budget", settings["seed"] == 20261208
        and settings["budgets"] == dict(controls=24, coarse=4, refinement_max=0, repeats_max=396, maximum_requests=424)
        and settings["search"] == dict(control_contexts=8, coarse_contexts=4, coarse_requests=4,
            refinement_per_captured_context=0, maximum_recipes=4)
        and settings["repeatability"] == dict(nominal=3, fresh=30, noise_std=[.001, .25, .002],
            same_requests_and_joints=True, clipping=False, retry=False, max_local_joint_delta_rad=.10))
    a.check("singleton_grid_and_no_adaptive_candidate_changes", settings["grid"] == dict(
        board_heights_m=[.380], tilts_deg=[28.], release_headroom_m=[ceiling - release],
        source_anchor_selection="Nearest normalized H/40mm and tilt/4deg, then stable source ID; no outcome labels"))
    control = dict(id="parent-control", distance_id="parent-control", ray_id="original_control",
        distance_m=float(np.linalg.norm(np.asarray(seeds[0]["bucket_xy_m"]) - board)),
        board_xy_m=board.tolist(), bucket_xy_m=seeds[0]["bucket_xy_m"],
        scope="Exact source controls; excluded from far-target search tables")
    a.check("exact_original_control_target_separate_from_discovery", settings["control_target"] == control)
    declared = [dict(id="control:" + s["id"], distance_id=control["id"], ray_id=control["ray_id"],
        distance_m=control["distance_m"], board_xy_m=control["board_xy_m"], bucket_xy_m=control["bucket_xy_m"],
        source_anchor_id=s["id"], source_family=s["id"], board_height_m=s["values"][2],
        tilt_deg=s["values"][3], proposed_anchor_gap_m=s["values"][4], stage="control") for s in seeds]
    source = min(seeds, key=lambda s: (((s["values"][2] - .380) / .04) ** 2
        + ((s["values"][3] - 28.) / 4.) ** 2, s["id"]))
    for target in targets:
        declared.append(dict(id=target["id"] + ":exact-candidate",
            **{k: copy.deepcopy(target[k]) for k in ("distance_id", "ray_id", "distance_m", "board_xy_m", "bucket_xy_m")},
            source_anchor_id=source["id"], source_family=source["id"], board_height_m=.380,
            tilt_deg=28., proposed_anchor_gap_m=.3868, stage="shallow"))
    return declared


def select_observed_candidates(rows, target_rows):
    """One discovery request per target; select only its actual strict capture."""
    recipes = []
    for target in target_rows:
        candidates = [r for r in rows if r["distance_id"] == target["id"]]
        if len(candidates) != 1:
            raise ValueError("Exactly one discovery request per far target required")
        row = candidates[0]
        if not row["route_success"] or row.get("invalid") or not row["native_release"]: continue
        recipes.append(dict(source=copy.deepcopy(row), sampled_run_width_m=0., id=f"recipe-{len(recipes):02d}",
            **{k: copy.deepcopy(target[k]) for k in ("distance_id", "ray_id", "distance_m", "bucket_xy_m", "board_xy_m")},
            context_id=row["context_id"], board_height_m=.380, tilt_deg=28., gap_m=.3868,
            selection="Actual captured exact requested candidate; no search or parameter refinement"))
    return recipes


def expected_repeats(recipes):
    rng = np.random.default_rng(20261208)
    result = []
    for recipe in recipes:
        for i, noise in enumerate(np.r_[np.zeros((3, 3)), rng.normal(0, [.001, .25, .002], (30, 3))]):
            values = np.asarray(recipe["source"]["values"]).copy(); values[2:] += noise
            result.append(dict(repeat_id=f"{recipe['id']}-repeat-{i:02d}", recipe_id=recipe["id"],
                context_id=f"{recipe['id']}-local-{i:02d}", role="engineering_repeat",
                phase="nominal" if i < 3 else "fresh", repeat_index=i, noise=noise.tolist(), values=values.tolist(),
                board_height_m=float(values[2]), tilt_deg=float(values[3]), gap_m=float(values[4]),
                **{k: copy.deepcopy(recipe[k]) for k in ("distance_id", "ray_id", "distance_m", "bucket_xy_m", "board_xy_m")}))
    return result


def audit_plans(a, study, spec, declared, rows, recipes):
    candidate = spec["settings"]["candidate"]
    discovery = [dict(context_id=c["id"], distance_id=c["distance_id"], ray_id=c["ray_id"],
        distance_m=c["distance_m"], source_anchor_id=c["source_anchor_id"], role="development", phase="coarse",
        gap_m=.3868, release_headroom_m=candidate["release_headroom_m"],
        requested_ball_center_height_m=.380 + .3868 + spec["scene"]["ball"]["radius_m"])
        for c in declared if c["stage"] == "shallow"]
    observed = [r for r in rows if r["phase"] == "coarse"]
    a.check("all_twelve_frozen_contexts", len(declared) == 12
        and read(study / "declared-contexts.json") == declared)
    a.check("exact_four_discovery_requests_with_literal_gap", read(study / "coarse-plan.json") == discovery
        and len(observed) == len(discovery) == 4 and all(r["substeps"] == 64
        and all(r.get(k) == v for k, v in p.items()) for r, p in zip(observed, discovery)))
    a.check("no_refinement_or_other_development_requests", read(study / "refinement-plan.json") == []
        and all(r["phase"] == "coarse" for r in rows if r["role"] == "development"))
    expected = select_observed_candidates(observed, spec["settings"]["targets"])
    a.check("only_actual_exact_discovery_captures_selected", recipes == expected and len(recipes) <= 4
        and len({r["distance_id"] for r in recipes}) == len(recipes)
        and all(r["source"]["values"][2:] == [.380, 28., .3868] for r in recipes))
    repeats = expected_repeats(recipes)
    a.check("independent_seed20261208_3nominal_30fresh_plan", read(study / "frozen-repeat-plan.json") == repeats
        and len(repeats) == 33 * len(recipes))
    actual = [r for r in rows if r["role"] == "engineering_repeat"]
    a.check("complete_bounded_requests_no_retries_or_hidden_shots", len(rows) == 24 + 4 + len(repeats) * 3 <= 424
        and Counter((r["repeat_id"], r["substeps"]) for r in actual)
        == Counter((r["repeat_id"], step) for r in repeats for step in (32, 64, 128))
        and set(r["phase"] for r in rows) <= {"controls", "coarse", "nominal", "fresh"})


def audit_rows(a, spec, prepared, rows, recipes, repeat_plan, repeat_geometry):
    return shared.audit_rows(a, spec, prepared, rows, recipes, repeat_plan, repeat_geometry,
        record_prefix="shallow-shot-")


def audit(study, source, assets, parent_study):
    from . import shallow_contract as protocol
    from .timestep_env import physics_settings
    study = Path(study)
    spec, report = read(study / "contract.json"), read(study / "report.json")
    rows, recipes = lines(study / "records.jsonl"), read(study / "recipes.json")
    a = Audit(); payload = dict(spec); fingerprint = payload.pop("fingerprint", None)
    a.check("contract_identity", digest(payload) == fingerprint == report.get("contract_fingerprint"))
    a.check("completed_version", spec.get("version") == report.get("version") == "sircus-shallow-board-candidate-v1"
        and report.get("status") == report.get("phase") == "completed")
    a.check("prespecified_settings_and_scope", spec["settings"] == protocol.settings(spec["scene"], spec["seeds"])
        == report["settings"] and spec["scope"] == report["scope"] == protocol.SCOPE and report["max_requests"] == 424)
    a.check("all_declared_unchanged_timestep_physics", spec["physics_by_substeps"]
        == {str(s): physics_settings(s) for s in (32, 64, 128)})
    parent = shared.audit_parent(a, spec, parent_study)
    audit_sources(a, study, source, assets, spec, report)
    declared = audit_candidate_and_contexts(a, spec, parent)
    audit_plans(a, study, spec, declared, rows, recipes)
    prepared = read(study / "prepared-contexts.json")
    repeats, geometry = read(study / "frozen-repeat-plan.json"), read(study / "repeat-geometry.json")
    shared.audit_contexts(a, spec, declared, prepared)
    shared.audit_controls(a, study, spec, rows)
    audit_rows(a, spec, prepared, rows, recipes, repeats, geometry)
    shared.audit_options(a, study, report, rows)
    positions, rescored = audit_batches(a, study, spec, rows)
    shared.audit_report(a, spec, report, declared, prepared, rows, recipes, positions)
    shared.audit_display_selection(a, spec, report, rows, recipes)
    a.check("at_most_eight_nominal_plus_four_posthoc_traces", len(report["trace_examples"]) <= 12)
    failures = [r for r in a.checks if not r["passed"]]
    return dict(version=VERSION, status="failed" if failures else "passed", contract_fingerprint=fingerprint,
        checks_count=len(a.checks), checks=a.checks, failures=failures, records=len(rows),
        independently_rescored_throws=rescored, auditor_source_sha256=sha256(__file__),
        native_tested_substeps=sorted({r["substeps"] for r in rows if r["native_release"]}),
        scope="Saved exact candidate/source/geometry/scoring CPU audit; no new dynamics, policy or physical validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "assets", "parent-study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try: result = audit(args.study, args.source, args.assets, args.parent_study)
    except (KeyError, ValueError, TypeError, OSError, IndexError, AttributeError) as error:
        result = dict(version=VERSION, status="failed", checks_count=0, checks=[], failures=[dict(
            name="malformed_or_missing_evidence", passed=False, detail=str(error))])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks_count", "independently_rescored_throws")}))
    if result["status"] != "passed": raise RuntimeError("Shallow-candidate evidence audit failed")


if __name__ == "__main__": main()
