"""Validate/export compact spatial-grid evidence; never include raw trajectories."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from bucket_grid_contract import (
    candidates,
    cells,
    choose,
    contract,
    perturb,
    refine,
    select,
    summary,
)
from distance_catalogue_report import params
from plane_contract import compact
from settling_contract import accepted


def seeded_params(expected, actual):
    # Fused arithmetic in NumPy's uniform generator can differ by one ULP across
    # architectures. Seeds, nominal actions and raw/compact equality stay exact.
    for key in ("height_m", "gap_m", "tilt_deg"):
        a, b = expected[key], actual[key]
        assert math.isfinite(a) and math.isfinite(b)
        assert abs(a - b) <= 8 * max(math.ulp(a), math.ulp(b)), (
            f"Seeded parameter differs: {key}"
        )


def validate(d):
    assert d["phase"] == "completed" and not d["contract"]["smoke"], (
        "Incomplete or smoke study"
    )
    cfg = d["contract"]["parent_quality"]["parent"]["scene"]
    expected = contract(cfg)
    stored = {
        k: v
        for k, v in d["contract"].items()
        if k not in ("fingerprint", "source_sha256", "contact_filters")
    }
    assert (
        hashlib.sha256(json.dumps(stored, sort_keys=True).encode()).hexdigest()
        == d["contract"]["fingerprint"]
    ), "Stored fingerprint changed"
    assert all(
        d["contract"][k] == v
        for k, v in expected.items()
        if k not in ("fingerprint", "cells")
    ), "Study contract mismatch"
    assert len(d["contract"]["cells"]) == len(expected["cells"])
    # libm atan2 differs by a few ULPs on macOS versus Linux. Preserve the original
    # hash and exact positions/actions; tolerate only derived display diagnostics.
    for saved, recomputed in zip(d["contract"]["cells"], expected["cells"]):
        assert (
            saved["id"] == recomputed["id"]
            and saved["bucket_xy_m"] == recomputed["bucket_xy_m"]
        )
        assert math.isclose(
            saved["distance_m"], recomputed["distance_m"], rel_tol=0, abs_tol=1e-12
        )
        assert math.isclose(
            saved["heading_deg"], recomputed["heading_deg"], rel_tol=0, abs_tol=1e-10
        )
    assert [p["kind"] for p in d["probes"]] == [
        "bucket",
        "rim",
        "table",
        "floor",
        "robot",
    ]
    for p in d["probes"]:
        assert p["passed"] and all(
            math.isfinite(v) and v >= 0 for v in p["peak_N"].values()
        )
        assert p["peak_N"]["bucket" if p["kind"] == "rim" else p["kind"]] > 0.01
        if p["kind"] in ("table", "floor"):
            assert p["peak_N"]["bucket"] <= 0.01 and p["peak_N"]["board"] <= 0.01
    assert len(d["controls"]) == 2 and d["controls"][0]["route_success"]
    assert (
        not d["controls"][1]["route_success"]
        and d["controls"][1]["external_contact_steps"]
    )
    assert len(d["cells"]) == len(cells())
    all_rows = list(d["controls"])
    for ci, (cell, e) in enumerate(zip(d["contract"]["cells"], d["cells"])):
        assert all(e[k] == v for k, v in cell.items())
        coarse = candidates()
        refined = refine(e["rows"][: len(coarse)])
        assert e["refinements"] == refined
        assert len(e["rows"]) == len(coarse) + len(refined)
        for a, r in zip(coarse + refined, e["rows"]):
            params(a, r)
        assert all(r["cell_id"] == cell["id"] for r in e["rows"])
        assert summary(e["rows"]) == e["search_summary"]
        assert select(e["rows"]) == [g["candidate"] for g in e["screened"]]
        all_rows.extend(e["rows"])
        for gi, g in enumerate(e["screened"]):
            assert len(g["trials"]) == 10 and summary(g["trials"]) == g["summary"]
            for i, r in enumerate(g["trials"]):
                sample = perturb(g["candidate"], ci, gi, i)
                seeded_params(sample, r)
                assert r["seed"] == sample["seed"]
            all_rows.extend(g["trials"])
        assert e["winner"] == choose(e["screened"])
        if e["winner"]:
            assert e["phase"] == "completed" and len(e["confirmation"]) == 30
            for i, r in enumerate(e["confirmation"]):
                sample = perturb(e["winner"], ci, 0, i, True)
                seeded_params(sample, r)
                assert r["seed"] == sample["seed"]
            assert summary(e["confirmation"]) == e["confirmation_summary"]
            all_rows.extend(e["confirmation"])
        else:
            assert e["phase"] == "no_route_setting_found" and not e["confirmation"]
    assert len(all_rows) == d["attempts"] and len(
        {r["trial_id"] for r in all_rows}
    ) == len(all_rows)
    for r in all_rows:
        assert bool(r["quality_score"]) == r["success"]
        assert bool(r["route_quality"]) == r["route_success"]
        assert r["route_quality"] in (0, 1, 2)
        assert r["single_bounce_success"] == bool(
            r["route_success"] and r["board_contact_episodes"] == 1
        )
        if r["route_success"]:
            assert (
                r["success"]
                and r["board_bounce"]
                and r["whole_ball_entry"]
                and r["retention_s"] >= 1 - 1e-9
            )
            assert r["board_contact_episodes"] >= 1 and not r["external_contact_steps"]
            assert all(
                r["contact_peaks_N"][k] <= 0.01
                for k in ("table", "floor", "robot", "unattributed")
            )
        if not r.get("invalid"):
            assert r["route_version"] == d["contract"]["route_version"]
            peaks = r["contact_peaks_N"]
            assert set(peaks) == {
                "board",
                "bucket",
                "table",
                "floor",
                "robot",
                "unattributed",
            }
            assert all(math.isfinite(v) and v >= 0 for v in peaks.values())
            assert bool(r["external_contact_steps"]) == any(
                peaks[k] > 0.01 for k in ("table", "floor", "robot", "unattributed")
            )
            assert r["route_success"] == bool(
                r["success"]
                and r["board_contact_episodes"] >= 1
                and not r["external_contact_steps"]
            )
            assert accepted(r["settled"])
    released = [r for r in all_rows if not r.get("invalid")]
    return dict(
        summary(all_rows),
        released=len(released),
        max_settled_position_error_mm=max(
            (r["settled"]["position_error_mm"] for r in released), default=None
        ),
        max_settled_normal_error_deg=max(
            (r["settled"]["normal_error_deg"] for r in released), default=None
        ),
        max_settled_joint_speed_rad_s=max(
            (r["settled"]["joint_speed_rad_s"] for r in released), default=None
        ),
    )


def records(d):
    rows = list(d["controls"])
    for e in d["cells"]:
        rows.extend(e["rows"])
        for group in e["screened"]:
            rows.extend(group["trials"])
        rows.extend(e["confirmation"])
    return rows


def verify_raw(d, source, source_dir):
    """Cross-check compact evidence against immutable local artifacts and code."""
    for name, digest in d["artifact_sha256"].items():
        assert Path(name).name == name
        assert hashlib.sha256((source.parent / name).read_bytes()).hexdigest() == digest
    assert json.loads((source.parent / "contract.json").read_text()) == d["contract"]
    for name, digest in d["contract"]["source_sha256"].items():
        assert Path(name).name == name
        assert hashlib.sha256((source_dir / name).read_bytes()).hexdigest() == digest
    raw = [
        json.loads(line)
        for line in (source.parent / "trials.jsonl").read_text().splitlines()
    ]
    rows = records(d)
    assert len(raw) == len(rows)
    for t, row in zip(raw, rows):
        result = t["result"]
        expected = compact(t["candidate"], result)
        expected.update(
            trial_id=t["trial_id"], cell_id=t["cell"]["id"], stage=t["stage"]
        )
        for key in (
            "quality_score",
            "route_success",
            "single_bounce_success",
            "route_quality",
            "route_version",
            "board_contact_episodes",
            "external_contact_steps",
            "contact_peaks_N",
        ):
            expected[key] = result.get(key)
        if result.get("invalid"):
            expected.update(
                route_success=False,
                single_bounce_success=False,
                route_quality=0,
                quality_score=0,
            )
        assert expected == row, "Raw/compact trial mismatch"
        if result.get("quality_score") == 2:
            assert result["minimum_mouth_clearance_mm"] >= 1
            assert not any(
                result[k]
                for k in (
                    "bucket_contact_before_entry",
                    "rim_band_contact",
                    "exited_after_entry",
                )
            )
    return {
        "raw_trials_matched": len(raw),
        "artifacts_matched": len(d["artifact_sha256"]),
        "executed_source_hashes_matched": len(d["contract"]["source_sha256"]),
    }


def export(source, dest, source_dir):
    d = json.loads(source.read_text())
    d["verification"] = dict(validate(d), **verify_raw(d, source, source_dir))
    d["source_summary_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps(d, separators=(",", ":")) + "\n")
    fields = [
        "bucket_x_mm",
        "bucket_y_mm",
        "distance_mm",
        "status",
        "board_height_mm",
        "tilt_deg",
        "gap_mm",
        "ball_bottom_Z_mm",
        "fresh_trials",
        "route_captures",
        "single_bounce_captures",
        "clean",
        "assisted",
        "historical_captures",
        "invalid",
        "run",
        "fingerprint",
    ]
    with (dest / "settings.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for e in d["cells"]:
            r = {
                "bucket_x_mm": e["bucket_xy_m"][0] * 1000,
                "bucket_y_mm": e["bucket_xy_m"][1] * 1000,
                "distance_mm": round(e["distance_m"] * 1000, 2),
                "status": e["phase"],
                "run": d["run"],
                "fingerprint": d["contract"]["fingerprint"],
            }
            c = e["winner"]
            if c:
                r.update(
                    board_height_mm=c["height_m"] * 1000,
                    tilt_deg=c["tilt_deg"],
                    gap_mm=c["gap_m"] * 1000,
                    ball_bottom_Z_mm=(c["height_m"] + c["gap_m"]) * 1000,
                )
                s = e["confirmation_summary"]
                r["fresh_trials"] = s["trials"]
                r.update(
                    {
                        k: s[k]
                        for k in [
                            "route_captures",
                            "single_bounce_captures",
                            "clean",
                            "assisted",
                            "historical_captures",
                            "invalid",
                        ]
                    }
                )
            w.writerow(r)
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--executed-source", type=Path, required=True)
    a = p.parse_args()
    export(a.source, a.destination, a.executed_source)
