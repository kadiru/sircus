"""Validate frozen multi-cell evidence and export a credential-free compact report."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from bucket_grid_report import seeded_params
from parallel_contract import batch_actions, candidates, select, specification
from settling_contract import accepted


def validate(d):
    s = d["contract"]
    assert d["phase"] == "completed" and d["validation_passed"] and not s["smoke"]
    payload = {
        k: v
        for k, v in s.items()
        if k not in ("fingerprint", "source_sha256", "gpu", "smoke")
    }
    assert (
        hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        == s["fingerprint"]
    )
    expected = specification(
        s["scene"],
        s["grid_count"],
        [c["index"] for c in s["cells"]],
        s["batches"],
        s["confirmation_trials_per_selected_cell"],
    )
    for k, v in expected.items():
        if k not in ("fingerprint", "cells", "parent_grid"):
            assert s[k] == v, k
    for cell, e in zip(s["cells"], expected["cells"]):
        for k, v in e.items():
            if k in ("heading_deg", "distance_m"):
                assert math.isclose(cell[k], v, abs_tol=1e-10, rel_tol=0)
            else:
                assert cell[k] == v
    assert len(d["cells"]) == s["num_envs"]
    assert d["batches_completed"] == s["batches"]
    assert 0 <= d["max_bucket_drift_m"] <= 1e-5
    kinds = ["bucket", "rim", "table", "floor", "robot"] + (
        ["foreign_cell_collision_filter"] if s["num_envs"] > 1 else []
    )
    assert [p["kind"] for p in d["probes"]] == kinds
    for p in d["probes"]:
        assert p["all_passed"]
        peaks = np.array(p["peak_N"])
        assert np.isfinite(peaks).all() and (peaks >= 0).all()
        if p["kind"] == "foreign_cell_collision_filter":
            assert peaks.max() <= 0.01 and p["foreign_ball_final_z_m"] < -1
        else:
            assert peaks.shape == (s["num_envs"], 6) and all(p["per_cell_passed"])
            target = {"bucket": 1, "rim": 1, "table": 2, "floor": 3, "robot": 4}[
                p["kind"]
            ]
            assert (peaks[:, target] > 0.01).all()
            if p["kind"] in ("table", "floor"):
                assert (peaks[:, :2] <= 0.01).all()
    rows = []
    assert len(d["controls"]) == 2
    for ci, control in enumerate(d["controls"]):
        assert len(control["rows"]) == s["num_envs"]
        for cell, r in zip(s["cells"], control["rows"]):
            assert all(r[k] == v for k, v in candidates()[ci].items())
            assert r["cell_id"] == cell["id"]
            rows.append(r)
    for i, (cell, e) in enumerate(zip(d["cells"], s["cells"])):
        assert all(cell[k] == v for k, v in e.items())
        assert len(cell["rows"]) == s["batches"]
        for batch, r in enumerate(cell["rows"]):
            a = batch_actions(batch, s["cells"])[i]
            assert all(r[k] == v for k, v in a.items()) and r["cell_id"] == cell["id"]
        best = select(cell["rows"] + [c["rows"][i] for c in d["controls"]])
        winner = (
            {k: best[k] for k in ("id", "height_m", "gap_m", "tilt_deg")}
            if best
            else None
        )
        assert cell["winner"] == winner
        assert len(cell["confirmations"]) == (
            s["confirmation_trials_per_selected_cell"] if winner else 0
        )
        for rep, r in enumerate(cell["confirmations"]):
            seed = 20261102 + 1000 * cell["index"] + rep
            assert (
                r["seed"] == seed
                and r["cell_id"] == cell["id"]
                and r["id"] == winner["id"]
            )
            rng = np.random.default_rng(seed)
            a = dict(winner)
            for key, delta in s["perturbations"].items():
                a[key] += float(rng.uniform(-delta, delta))
            seeded_params(a, r)
        rows += cell["rows"] + cell["confirmations"]
    assert len(rows) == d["attempts"] and len({r["trial_id"] for r in rows}) == len(
        rows
    )
    for r in rows:
        if r["success"]:
            assert (
                r["board_bounce"]
                and r["whole_ball_entry"]
                and r["max_retention_s"] >= 1 - 1e-9
            )
        expected_route = bool(
            r["success"]
            and r["board_contact_episodes"] >= 1
            and not r["external_contact_steps"]
        )
        assert r["route_success"] == expected_route
        assert r["single_bounce_success"] == bool(
            expected_route and r["board_contact_episodes"] == 1
        )
        assert r["route_quality"] == (r["quality_score"] if expected_route else 0)
        if r["invalid"]:
            assert not r["success"] and not r["route_success"]
        else:
            assert (
                accepted(r["settled"])
                and r["settled"]["dropper_position_error_mm"] <= 2
            )
    return rows


def export(source, dest, executed_source):
    d = json.loads((source / "summary.json").read_text())
    rows = validate(d)
    for name, digest in d["artifact_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    for name, digest in d["contract"]["source_sha256"].items():
        assert (
            hashlib.sha256((executed_source / name).read_bytes()).hexdigest() == digest
        ), name
    saved = {
        r["trial_id"]: r
        for line in (source / "trials.jsonl").read_text().splitlines()
        for r in json.loads(line)["rows"]
        if "trial_id" in r
    }
    assert len(saved) == len(rows)
    assert all(saved[r["trial_id"]] == r for r in rows)
    d["verification"] = {
        "contract": True,
        "plans_and_seeds": True,
        "raw_rows": True,
        "executed_source_hashes": True,
        "distinct_fixed_targets": True,
        "contact_probes": True,
        "rows_checked": len(rows),
    }
    d["csv_url"] = f"/parallel-media/{d['run']}/settings.csv"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    fields = [
        "cell",
        "bucket_x_mm",
        "bucket_y_mm",
        "distance_mm",
        "board_height_mm",
        "gap_mm",
        "tilt_deg",
        "ball_bottom_z_mm",
        "fresh_trials",
        "route_captures",
        "single_bounce_captures",
        "clean",
        "invalid",
        "run",
        "fingerprint",
    ]
    with (dest / "settings.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for c in d["cells"]:
            tests = c["confirmations"]
            a = c["winner"]
            row = {
                "cell": c["id"],
                "bucket_x_mm": c["bucket_xy_m"][0] * 1000,
                "bucket_y_mm": c["bucket_xy_m"][1] * 1000,
                "distance_mm": c["distance_m"] * 1000,
                "fresh_trials": len(tests),
                "route_captures": sum(r["route_success"] for r in tests),
                "single_bounce_captures": sum(
                    r["single_bounce_success"] for r in tests
                ),
                "clean": sum(r["route_quality"] == 2 for r in tests),
                "invalid": sum(bool(r["invalid"]) for r in tests),
                "run": d["run"],
                "fingerprint": d["contract"]["fingerprint"],
            }
            if a:
                row.update(
                    board_height_mm=a["height_m"] * 1000,
                    gap_mm=a["gap_m"] * 1000,
                    tilt_deg=a["tilt_deg"],
                    ball_bottom_z_mm=(a["height_m"] + a["gap_m"]) * 1000,
                )
            w.writerow(row)
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--executed-source", required=True, type=Path)
    a = p.parse_args()
    export(a.source, a.destination, a.executed_source)
