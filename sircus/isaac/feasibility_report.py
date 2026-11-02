"""Verify raw simulator evidence and export small development-search summaries."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from feasibility_contract import (
    ACTION_KEYS,
    baseline_actions,
    dynamic_cell,
    perturb,
    rank,
    refinement_actions,
    search_actions,
    select,
    specification,
)
from settling_contract import accepted

STAGES = (
    "baseline",
    "search",
    "refinement",
    "nominal",
    "fixed_xy_ablation",
    "confirmations",
)


def rate(rows):
    n = len(rows)
    k = sum(r["route_success"] for r in rows)
    z = 1.959963984540054
    center = (k / n + z * z / (2 * n)) / (1 + z * z / n) if n else 0
    half = (
        z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        if n
        else 0
    )
    return {
        "trials": n,
        "captures": k,
        "single_bounce": sum(r["single_bounce_success"] for r in rows),
        "clean": sum(r["route_quality"] == 2 for r in rows),
        "invalid": sum(bool(r["invalid"]) for r in rows),
        "failures": dict(
            Counter(
                r["invalid"]
                or (
                    "external_contact"
                    if r["external_contact_steps"]
                    else "released_miss"
                )
                for r in rows
                if not r["route_success"]
            )
        ),
        "wilson95": [max(0, center - half), min(1, center + half)] if n else None,
    }


def assert_action(row, action):
    for key in ACTION_KEYS:
        assert math.isclose(row[key], action[key], rel_tol=0, abs_tol=1e-12), key
    assert row["id"] == action["id"]
    if "seed" in action:
        assert row["seed"] == action["seed"]


def validate(d):
    assert d["phase"] == "completed" and d["validation_passed"]
    s = d["contract"]
    assert not s["smoke"]
    expected = specification(s["scene"], [c["index"] for c in s["cells"]])
    # JSON normalization accounts only for tuple/list serialization.
    for key, value in json.loads(json.dumps(expected)).items():
        assert s[key] == value, key
    assert d["max_bucket_drift_m"] <= 1e-5
    kinds = ["bucket", "rim", "table", "floor", "robot"] + (
        ["foreign_cell_collision_filter"] if s["num_envs"] > 1 else []
    )
    assert [p["kind"] for p in d["probes"]] == kinds
    assert all(p["all_passed"] for p in d["probes"])
    for p in d["probes"]:
        if p["kind"] == "foreign_cell_collision_filter":
            assert max(p["peak_N"]) <= 0.01 and p["foreign_ball_final_z_m"] < -1
        else:
            assert len(p["peak_N"]) == s["num_envs"] and all(p["per_cell_passed"])
            index = {"bucket": 1, "rim": 1, "table": 2, "floor": 3, "robot": 4}[
                p["kind"]
            ]
            for peaks in p["peak_N"]:
                assert len(peaks) == 6 and all(
                    math.isfinite(v) and v >= 0 for v in peaks
                )
                assert peaks[index] > 0.01
                if p["kind"] in ("table", "floor"):
                    assert max(peaks[:2]) <= 0.01
    assert len(d["cells"]) == s["num_envs"]
    all_rows = []
    for c, expected_cell in zip(d["cells"], s["cells"]):
        assert all(c[k] == v for k, v in expected_cell.items())
        assert len(c["baseline"]) == len(baseline_actions())
        assert len(c["search"]) == len(search_actions(s["scene"], c))
        valid = [r for r in c["baseline"] + c["search"] if not r["invalid"]]
        refine = refinement_actions(max(valid, key=rank, default=None))
        assert len(c["refinement"]) == len(refine)
        for key, plan in [
            ("baseline", baseline_actions()),
            ("search", search_actions(s["scene"], c)),
            ("refinement", refine),
        ]:
            for row, action in zip(c[key], plan):
                assert_action(row, action)
        winner = select(c["baseline"] + c["search"] + c["refinement"])
        expected_winner = (
            {k: winner[k] for k in (*ACTION_KEYS, "id", "trial_id")} if winner else None
        )
        assert c["winner"] == expected_winner
        for stage, count in [
            ("nominal", 3),
            ("fixed_xy_ablation", 3),
            ("confirmations", 30),
        ]:
            assert len(c[stage]) == (count if winner else 0)
            for rep, row in enumerate(c[stage]):
                action = perturb(winner, c["index"], rep, stage != "confirmations")
                if stage == "fixed_xy_ablation":
                    action.update(board_x_m=0.67, board_y_m=-0.10)
                assert_action(row, action)
        for stage in STAGES:
            for r in c[stage]:
                assert r["cell_id"] == c["id"] and r["stage"] == stage and r["included"]
                dc = dynamic_cell(c, r)
                assert math.isclose(r["distance_m"], dc["distance_m"], abs_tol=1e-12)
                assert math.isclose(r["heading_deg"], dc["heading_deg"], abs_tol=1e-12)
                if r["invalid"]:
                    assert not r["success"]
                else:
                    assert (
                        accepted(r["settled"])
                        and r["settled"]["dropper_position_error_mm"] <= 2
                    )
                    assert r["distance_m"] >= 0.28
                if r["success"]:
                    assert (
                        r["board_bounce"]
                        and r["whole_ball_entry"]
                        and r["max_retention_s"] >= 1 - 1e-9
                    )
                route = bool(
                    r["success"]
                    and r["board_contact_episodes"] >= 1
                    and not r["external_contact_steps"]
                )
                assert r["route_success"] == route
                assert r["single_bounce_success"] == bool(
                    route and r["board_contact_episodes"] == 1
                )
                assert r["route_quality"] == (r["quality_score"] if route else 0)
                all_rows.append(r)
    assert len(all_rows) == d["attempts"] == len({r["trial_id"] for r in all_rows})
    assert [r["cell_id"] for r in d["direct_drop_audit"]] == [
        c["id"] for c in s["cells"]
    ]
    for r in d["direct_drop_audit"]:
        assert r["beyond_geometric_outer_bound"] == (
            r["horizontal_distance_m"] > r["ball_outer_bound_m"]
        )
        assert math.isclose(r["ball_outer_bound_m"], r["flange_outer_bound_m"] + 0.095)
        assert [t["ball_center_z_m"] for t in r["tested_heights"]] == s[
            "direct_drop_audit_ball_heights_m"
        ]
    return all_rows


def compact(d):
    report = {
        k: d[k]
        for k in (
            "run",
            "phase",
            "contract",
            "timestamp_utc",
            "elapsed_s",
            "attempts",
            "max_bucket_drift_m",
            "probes",
            "artifact_sha256",
            "direct_drop_audit",
        )
    }
    report["cells"] = []
    for c in d["cells"]:
        row = {k: c[k] for k in d["contract"]["cells"][0]}
        row["winner"] = c["winner"]
        row["stages"] = {stage: rate(c[stage]) for stage in STAGES}
        for stage in ("nominal", "fixed_xy_ablation", "confirmations"):
            row[stage + "_replay"] = c.get(stage + "_replay")
            if c[stage]:
                row[stage + "_example"] = c[stage][0]
        if c["winner"]:
            row["selected_geometry"] = dynamic_cell(c, c["winner"])
        row["status"] = (
            (
                "30_of_30"
                if row["stages"]["confirmations"]["captures"] == 30
                else "tested_setting"
            )
            if c["winner"]
            else "unresolved"
        )
        report["cells"].append(row)
    report["totals"] = {
        stage: rate([r for c in d["cells"] for r in c[stage]]) for stage in STAGES
    }
    report["limits"] = [
        "Finite development search at 32 sampled bucket points, not continuous table coverage or RL policy performance.",
        "Different search budgets for fixed and expanded XY; their raw counts are not a causal comparison. The matched fixed-XY ablation uses frozen heights/tilt.",
        "30 fresh local perturbations per selected setting; failures and endpoint rejections remain in the denominator. No guarantee of reliability.",
        "Endpoint initialization and settling only. Arm paths, grasping, self-collision certification, perception and calibrated physical performance are not established.",
        "No successful setting found means unresolved, not impossible. Bucket height interpretation and contact/mount properties remain provisional.",
    ]
    return report


def export(source, dest, executed_source):
    d = json.loads((source / "summary.json").read_text())
    rows = validate(d)
    for name, expected in d["artifact_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, (
            name
        )
    for name, expected in d["contract"]["source_sha256"].items():
        assert (
            hashlib.sha256((executed_source / name).read_bytes()).hexdigest()
            == expected
        ), name
    batches = [
        json.loads(line) for line in (source / "trials.jsonl").read_text().splitlines()
    ]
    raw = [r for batch in batches for r in batch["rows"] if r["included"]]
    assert {r["trial_id"]: r for r in raw} == {r["trial_id"]: r for r in rows}
    frozen = json.loads((source / "frozen-settings.json").read_text())
    assert frozen == {c["id"]: c["winner"] for c in d["cells"]}
    trajectories = {
        r["trial_id"]: batch["raw"]["trajectory_m"][i]
        for batch in batches
        for i, r in enumerate(batch["rows"])
        if r["included"]
    }
    for cell in d["cells"]:
        for stage in ("nominal", "fixed_xy_ablation", "confirmations"):
            replay = cell.get(stage + "_replay")
            if replay:
                assert replay["trial_id"] == cell[stage][0]["trial_id"]
                assert replay["trajectory_m"] == trajectories[replay["trial_id"]]
                assert math.isclose(
                    replay["sample_dt_s"],
                    8 * d["contract"]["scene"]["simulation"]["physics_dt_s"],
                )
    report = compact(d)
    report["verification"] = {
        "raw_rows_matched": len(raw),
        "source_hashes_matched": len(d["contract"]["source_sha256"]),
        "artifact_hashes_matched": len(d["artifact_sha256"]),
        "contract_actions_selection_repeats_checked": True,
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (dest / "settings.csv").open("w") as f:
        fields = [
            "cell_id",
            "bucket_x_mm",
            "bucket_y_mm",
            "board_x_mm",
            "board_y_mm",
            "board_height_mm",
            "drop_gap_mm",
            "ball_bottom_release_mm",
            "ball_center_release_mm",
            "tilt_deg",
            "distance_mm",
            "nominal",
            "perturbed_captures",
            "perturbed_trials",
            "single_bounce",
            "clean",
            "invalid",
            "fixed_xy_ablation",
            "beyond_dropper_outer_bound",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c, reach in zip(report["cells"], report["direct_drop_audit"]):
            a, stats = c["winner"], c["stages"]["confirmations"]
            value = {
                "cell_id": c["id"],
                "bucket_x_mm": c["bucket_xy_m"][0] * 1000,
                "bucket_y_mm": c["bucket_xy_m"][1] * 1000,
                "status": c["status"],
                "beyond_dropper_outer_bound": reach["beyond_geometric_outer_bound"],
            }
            if a:
                value.update(
                    board_x_mm=a["board_x_m"] * 1000,
                    board_y_mm=a["board_y_m"] * 1000,
                    board_height_mm=a["height_m"] * 1000,
                    drop_gap_mm=a["gap_m"] * 1000,
                    ball_bottom_release_mm=(a["height_m"] + a["gap_m"]) * 1000,
                    ball_center_release_mm=(
                        a["height_m"]
                        + a["gap_m"]
                        + d["contract"]["scene"]["ball"]["radius_m"]
                    )
                    * 1000,
                    tilt_deg=a["tilt_deg"],
                    distance_mm=c["selected_geometry"]["distance_m"] * 1000,
                    nominal=c["stages"]["nominal"]["captures"],
                    perturbed_captures=stats["captures"],
                    perturbed_trials=stats["trials"],
                    single_bounce=stats["single_bounce"],
                    clean=stats["clean"],
                    invalid=stats["invalid"],
                    fixed_xy_ablation=c["stages"]["fixed_xy_ablation"]["captures"],
                )
            writer.writerow(value)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--executed-source", type=Path, required=True)
    a = p.parse_args()
    r = export(a.source, a.dest, a.executed_source)
    print(
        json.dumps(
            {
                "run": r["run"],
                "attempts": r["attempts"],
                "totals": r["totals"],
                "verification": r["verification"],
            }
        )
    )
