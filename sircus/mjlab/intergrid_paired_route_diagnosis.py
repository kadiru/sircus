"""Diagnose table-shot setup and flight failures from an audited matched test."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
from statistics import median

import numpy as np

from .benchmark_verify import contained, read, sha
from .calibrated_score import CONTACT_FREE_S, FORCE_THRESHOLD_N, REBOUND_NORMAL_SPEED_M_S, TIME_EPS_S
from .cell1050_followup_plan import geometry_key
from .intergrid_matched_route_study import load_model_inputs
from .intergrid_proposer import HIGH, LOW, RECIPE_SCALE, _features, _source, _wrap_degrees
from .intergrid_route_labels import TABLE_SPEED_M_S
from .intergrid_route_models import _coefficients
from .intergrid_route_study import route_library, source_menu
from .parallel_capture_grid_audit import observed_rows


def setup(study, layouts, slots, library, checkpoint, scene, assets, manifest):
    from .cell1050_heading_native import heading_targets, reach_models, wrist_bound

    matrices = _coefficients(checkpoint)
    arm_models = reach_models(str(assets))
    slots_by_case = {(s["context_id"], s["method"], s["intended_route"]): s for s in slots}
    candidates, failures, distances = (defaultdict(Counter), defaultdict(Counter), defaultdict(list))
    invalid_board_y, arms = Counter(), Counter()
    for layout in layouts:
        identity = layout["id"]
        path = f"test/{identity}.json"
        if sha(study / path) != manifest[path]:
            raise ValueError(f"Setup file changed: {path}")
        prepared = {r["proposal_id"]: r for r in read(study / path)["records"]}
        target = np.asarray(layout["bucket_xy_m"])
        sources = source_menu(layout, library)
        for method, coefficients in matrices.items():
            by_route = defaultdict(Counter)
            seen = set()
            for index, recipe in enumerate(sources):
                route = recipe["intended_route"]
                source = _source(recipe)
                values = source["values"] + (_features(target, source) @ coefficients[route]) * RECIPE_SCALE
                values[5:] = _wrap_degrees(values[5:])
                identifier = f"{identity}-{method}-{index:02d}"
                if method == next(iter(matrices)):
                    distances[route].append(math.dist(target, source["xy"]) * 1000)
                key = geometry_key(dict(values=values[:5], board_heading_offset_deg=values[5],
                                        board_twist_deg=values[6])) if np.isfinite(values).all() else None
                if key is None:
                    result = "nonfinite"
                elif not ((LOW <= values[:5]) & (values[:5] <= HIGH)).all():
                    result = "action_bounds"
                    if not LOW[1] <= values[1] <= HIGH[1]:
                        invalid_board_y[route] += 1
                elif np.linalg.norm(target - values[:2]) <= .15:
                    result = "board_too_close"
                elif key in seen or key == geometry_key(recipe):
                    result = "duplicate_or_unchanged"
                else:
                    seen.add(key)
                    result = prepared[identifier]["reason"] or "admitted"
                    if result == "proven_full_pose_wrist_reach_rejected":
                        targets = heading_targets(scene, target, values[:5], *values[5:])
                        failed = [arm for arm, name in (("dropper", "dropper_flange_world"), ("bouncer", "board_flange_local"))
                                  if not wrist_bound(arm_models[arm], targets[name])["passed"]]
                        if not failed:
                            raise ValueError(f"Could not reproduce wrist rejection: {identifier}")
                        arms[f"{route}/" + "+".join(failed)] += 1
                filtered = result in {"nonfinite", "action_bounds", "board_too_close", "duplicate_or_unchanged"}
                if filtered == (identifier in prepared):
                    raise ValueError(f"Frozen candidate presence differs: {identifier}")
                by_route[route][result] += 1
                candidates[route][result] += 1
            for route, counts in by_route.items():
                if slots_by_case[identity, method, route]["request"] is None:
                    failures[route]["+".join(sorted(counts))] += 1
    return dict(candidates=candidates, slot_failures=failures, board_y_out_of_bounds=invalid_board_y,
                wrist_failures=arms, median_source_distance_mm={route: median(values)
                                                              for route, values in distances.items()})


def flight(study, manifest):
    groups = defaultdict(list)
    for row in observed_rows(study / "rows.jsonl"):
        if row["phase"] == "test" and row["intended_route"] == "table":
            groups[row["native_cache_file"]].append(row)
    outcomes, events = Counter(), []
    for filename, rows in sorted(groups.items()):
        path = contained(study / "native-heading", filename)
        checksum = manifest[f"native-heading/{filename}"]
        if sha(path) != checksum:
            raise ValueError(f"Dense trajectory changed: {filename}")
        with np.load(path, allow_pickle=False) as archive:
            times, forces, velocities, positions, ids = (archive[key] for key in
                ("time_s", "forces_N", "velocity_m_s", "position_m", "record_ids"))
        for row in rows:
            steps, column = row["physics_steps"], row["native_cache_row"]
            if row["id"] != ids[column] or row["native_cache_sha256"] != checksum:
                raise ValueError(f"Dense trajectory does not match {row['id']}")
            t, v = times[:steps + 1], velocities[:steps + 1, column]
            board, bucket, table, _, robot, _ = (forces[:steps, column] > FORCE_THRESHOLD_N).T
            last = np.maximum.accumulate(np.where(board, t[1:], -np.inf))
            board_out = np.flatnonzero((~board) & (t[1:] - last <= CONTACT_FREE_S + TIME_EPS_S)
                                      & (v[1:].astype(float) @ np.asarray(row["board_normal"]) > REBOUND_NORMAL_SPEED_M_S))
            if bool(len(board_out)) != row["board_bounce"]:
                raise ValueError(f"Could not reproduce board rebound: {row['id']}")
            edges = np.flatnonzero(np.diff(np.r_[False, table, False]))
            rebounds = [(s, e) for s, e in zip(edges[::2], edges[1::2], strict=True)
                        if len(board_out) and s >= board_out[0] + 1 and not (board[s:e] | bucket[s:e]).any()
                        and v[s, 2] < -TABLE_SPEED_M_S and v[e, 2] > TABLE_SPEED_M_S]
            if row["shot_route"] == "table" and sum(t[e] <= row["whole_ball_entry_time_s"] for _, e in rebounds) != row["table_rebounds_before_entry"]:
                raise ValueError(f"Could not reproduce table capture: {row['id']}")
            first_bucket = next(iter(np.flatnonzero(bucket)), steps + 1)
            first_robot = next(iter(np.flatnonzero(robot)), steps + 1)
            start, end = rebounds[0] if rebounds else (None, steps + 1)
            if row["invalid_physics"]:
                result = "invalid_physics"
            elif row["shot_route"] == "table":
                result = "table_capture"
            elif first_robot < min(first_bucket, end):
                result = "robot_before_bucket_or_rebound"
            elif start is None:
                result = "no_verified_table_rebound"
            elif first_bucket < end:
                result = "bucket_before_rebound"
            elif first_bucket <= steps:
                result = "bucket_after_rebound"
            elif first_robot <= steps:
                result = "robot_after_rebound_no_bucket"
            else:
                result = "rebound_no_bucket_or_robot"
            outcomes[result] += 1
            event = dict(id=row["id"], outcome=result, whole_ball_entry=row["whole_ball_entry"],
                         measured_rebound=bool(rebounds), bucket_xy_m=row["bucket_xy_m"])
            if start is not None:
                center = np.asarray(row["bucket_xy_m"])
                impact_xy = positions[start, column, :2]
                event["rebound_distance_mm"] = math.dist(impact_xy, center) * 1000
                if end <= first_bucket < steps:
                    offset = positions[first_bucket, column, :2] - center
                    event["near_side"] = bool(offset @ (center - impact_xy) < 0)
                    event["ascending"] = bool(v[first_bucket, 2] > 0)
            events.append(event)
    return dict(outcomes=outcomes, shots=events)


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "audit", "model-study", "model-audit", "assets"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    study = args.study
    audit, manifest = read(args.audit), read(study / "evidence-manifest.json")
    if (audit["status"] != "passed" or audit["report_sha256"] != sha(study / "report.json")
            or audit["evidence_manifest_sha256"] != sha(study / "evidence-manifest.json")):
        raise ValueError("Expected a passed audit of the matched test")
    for name in ("contract.json", "layouts.json", "choices.json", "rows.jsonl"):
        if manifest[name] != sha(study / name):
            raise ValueError(f"Matched input changed: {name}")
    checkpoint, rows, _, _, binding = load_model_inputs(args.model_study, args.model_audit)
    contract = read(study / "contract.json")
    if contract["model_study"] != binding or audit["checkpoint_sha256"] != binding["checkpoint"]:
        raise ValueError("Matched test used a different checkpoint")
    layouts = read(study / "layouts.json")["layouts"]
    slots = read(study / "choices.json")["slots"]
    print(json.dumps(dict(setup=setup(study, layouts, slots, route_library(rows), checkpoint,
                                    contract["scene"], args.assets, manifest),
                          flight=flight(study, manifest)), indent=2))


if __name__ == "__main__":
    main()
