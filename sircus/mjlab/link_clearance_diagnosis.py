"""Attribute saved robot impacts to reconstructed links without stepping physics.

The native robot-force channel is measured simulation evidence. Link names are
separate CPU geometry reconstructions at recorded ball centers and held joints.
Ambiguous or nonoverlapping reconstructions are retained, never forced to a link.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .calibrated_score import FORCE_NAMES, FORCE_THRESHOLD_N
from .hybrid_miss_diagnosis import episodes

VERSION = "sircus-remaining-link-diagnosis-v1"
CASE_IDS = ("parallel-board-miss-09", "parallel-board-miss-19")


def attribution(distances):
    """Name a unique overlapping collider; nearest is diagnostic when ambiguous."""
    ranked = sorted((float(distance), str(name)) for distance, name in distances)
    if not ranked or not all(np.isfinite(x[0]) for x in ranked):
        raise ValueError("Finite complete robot distances required")
    if len({x[1] for x in ranked}) != len(ranked):
        raise ValueError("Duplicate collider name")
    overlapping = [name for distance, name in ranked if distance < 0]
    return dict(status="unique_overlap" if len(overlapping) == 1 else
                "multiple_overlaps" if overlapping else "no_overlap_at_saved_sample",
                inferred_geometry=overlapping[0] if len(overlapping) == 1 else None,
                overlapping_geometries=overlapping,
                nearest_geometry=ranked[0][1], nearest_signed_distance_m=ranked[0][0],
                second_nearest_geometry=ranked[1][1] if len(ranked) > 1 else None,
                second_nearest_signed_distance_m=ranked[1][0] if len(ranked) > 1 else None)


def force_summary(t, p, v, f):
    found = episodes(t, p, v, f)
    robot = [event for event in found if event["channel"] == "robot"]
    if not robot:
        return dict(events=found, first_robot=None, peak_tick=None, robot_ticks=0)
    channel = f[:, FORCE_NAMES.index("robot")]
    return dict(events=found, first_robot=robot[0], peak_tick=int(np.argmax(channel)),
                robot_ticks=int(np.count_nonzero(channel > FORCE_THRESHOLD_N)))


def compact(cases):
    """Path-free UI summary; preserve uncertainty in reconstructed link labels."""
    result = []
    for case in cases:
        times = [r["measured_first_robot"]["start_s"] for r in case["results"] if r["measured_first_robot"]]
        result.append(dict(id=case["id"], bucket_xy_m=case["bucket_xy_m"], trials=case["shots"],
            robot_contacts=case["robot_contact_shots"], first_contact_link_counts=case["reconstructed_first_link_counts"],
            first_contact_time_s=dict(min=min(times) if times else None, max=max(times) if times else None),
            method="native_force_cpu_reconstruction"))
    return result


def diagnose(study, audit_path, assets, case_ids=CASE_IDS):
    from .arm_clearance_selector import ArmClearanceGate
    from .saved_shot_video import verify_assets
    study, audit_path, assets = map(Path, (study, audit_path, assets))
    checks = Checks()
    audit, report = read(audit_path), read(study / "report.json")
    checks.equal((audit["status"], report["status"]), ("passed", "completed"), "Audited complete source")
    checks.equal(audit["report_sha256"], sha(study / "report.json"), "Bound report")
    checks.equal(audit["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"), "Bound manifest")
    manifest = read(study / "evidence-manifest.json")
    verified = {}

    def bound(relative):
        path = contained(study, relative)
        if relative not in verified:
            checks.equal(sha(path), manifest[relative], "Bound artifact " + relative)
            verified[relative] = manifest[relative]
        return path

    contract = read(bound("contract.json"))
    checks.equal(audit["contract_sha256"], sha(study / "contract.json"), "Audited contract")
    fingerprint(contract, checks, "Source contract")
    checks.equal(contract["physics_substeps"], 64, "Source native64")
    checks.equal(digest(contract["scene"]), contract["identity"]["scene_sha256"], "Frozen scene")
    verify_assets(assets, contract["identity"]["asset_sha256"])
    root = Path(__file__).resolve().parents[2]
    for name in ("sircus/mjlab/model.py", "sircus/isaac/geometry.py"):
        checks.equal(sha(root / name), contract["identity"]["source_sha256"][name], "Historical geometry source")
    rows = [json.loads(line) for line in bound("rows.jsonl").read_text().splitlines()]
    preparation = read(bound("preparation.json"))
    gate = ArmClearanceGate(contract["scene"], assets)
    ball_id = gate.m.geom("ball/geom").id

    def distances(center):
        gate.d.qpos[gate.ball:gate.ball + 3] = center
        gate.mj.mj_kinematics(gate.m, gate.d)
        return [(float(gate.mj.mj_geomDistance(gate.m, gate.d, ball_id, item["index"], 10., None)),
                 item["name"]) for item in gate.local]

    cases = []
    for case_id in case_ids:
        selected = next(case for case in preparation["cases"] if case["id"] == case_id)
        candidates = {c["id"]: c for c in selected["candidates"]}
        found = [row for row in rows if row.get("context_id") == case_id and row.get("variant") == "search"]
        checks.equal(len(found), len(candidates), "All selected search drops " + case_id)
        checks.equal({row["candidate_id"] for row in found}, set(candidates), "Exact selected candidates")
        results = []
        for row in found:
            checks.equal(row["native_release"], True, "Native shot " + row["id"])
            checks.equal(row["contact_protocol"]["substeps"], 64, "Executed native64")
            for key in ("values", "bucket_xy_m", "selected_q10", "selected_q14"):
                checks.equal(row[key], candidates[row["candidate_id"]][key], "Exact selected " + key)
            relative = row["adapter_directory"] + "/" + row["native_cache_file"]
            trace_path = bound(relative)
            checks.equal(row["native_cache_sha256"], manifest[relative], "Row trace hash")
            if row["invalid_physics"]:
                results.append(dict(id=row["id"], candidate_id=row["candidate_id"], values=row["values"],
                    invalid_physics=True, invalid_physics_reasons=row.get("invalid_physics_reasons"),
                    measured_first_robot=None, reconstructed_onset=None,
                    attribution_omitted_reason="invalid_physics", trace_sha256=row["native_cache_sha256"],
                    trace_column=row["native_cache_row"], capture=False))
                continue
            with np.load(trace_path, allow_pickle=False) as trace:
                col, n = row["native_cache_row"], row["physics_steps"]
                checks.equal(str(trace["record_ids"][col]), row["id"], "Trace column identity")
                t, p, v, f = (trace["time_s"][:n + 1], trace["position_m"][:n + 1, col],
                              trace["velocity_m_s"][:n + 1, col], trace["forces_N"][:n, col])
            summary = force_summary(t, p, v, f)
            saved = row["experimental_route"]
            checks.equal(summary["robot_ticks"], saved["robot_contact_steps"], "Robot force tick count")
            first = summary["first_robot"]
            checks.equal(first["start_s"] if first else None, saved["first_robot_contact_s"], "Native force onset")
            gate.colliders(row["native_initial_q10"], row["native_initial_q14"], row["bucket_xy_m"])
            board = gate.m.body("Arm14/impact_board").id
            checks.numeric(gate.d.xpos[board], row["board_center_m"], "Reconstructed board center", 2e-6)
            checks.numeric(gate.d.xmat[board].reshape(3, 3)[:, 2], row["board_normal"], "Reconstructed board normal", 2e-6)
            checks.require(row["max_integrated_joint_excursion_rad"] < 1e-4, "Held endpoint reconstruction applicable")
            samples = []
            if first:
                ticks = sorted({first["start_tick"], min(first["start_tick"] + 1, n), summary["peak_tick"]})
                for tick in ticks:
                    samples.append(dict(tick=tick, time_s=float(t[tick]), center_m=p[tick].tolist(),
                                        **attribution(distances(p[tick]))))
            onset = next((s for s in samples if s["tick"] == first["start_tick"]), None) if first else None
            results.append(dict(id=row["id"], candidate_id=row["candidate_id"], values=row["values"],
                board_heading_offset_deg=row.get("board_heading_offset_deg", 0), board_twist_deg=row.get("board_twist_deg", 0),
                native_initial_q10=row["native_initial_q10"], native_initial_q14=row["native_initial_q14"],
                board_center_m=row["board_center_m"], board_normal=row["board_normal"],
                release_ball_center_m=row["release_ball_center_m"], capture=bool(row["route_success"]), invalid_physics=False,
                measured_events=summary["events"], measured_first_robot=first, reconstructed_onset=onset,
                reconstructed_samples=samples, trace_sha256=row["native_cache_sha256"], trace_column=row["native_cache_row"],
                max_integrated_joint_excursion_rad=row["max_integrated_joint_excursion_rad"]))
        cases.append(dict(id=case_id, bucket_xy_m=selected["original"]["bucket_xy_m"], shots=len(results),
            robot_contact_shots=sum(r["measured_first_robot"] is not None for r in results),
            invalid_physics_shots=sum(r["invalid_physics"] for r in results),
            reconstructed_first_link_counts=dict(Counter(r.get("attribution_omitted_reason") or
                 (r["reconstructed_onset"] or {}).get("inferred_geometry") or
                 (r["reconstructed_onset"] or {}).get("status", "no_robot_force") for r in results)), results=results))
    return dict(version=VERSION, status="completed", cpu_only=True, physics_substeps=64, checks_count=checks.count,
        source=dict(audit_sha256=sha(audit_path), report_sha256=sha(study / "report.json"),
                    manifest_sha256=sha(study / "evidence-manifest.json"), contract_sha256=sha(study / "contract.json")),
        verified_artifacts=verified, cases=cases, diagnosis_rows=compact(cases), model_sha256=gate.model_sha256,
        diagnostic_source_sha256=sha(__file__), mujoco_version=gate.mj.__version__,
        scope="Native force channels establish contact occurrence and timing. Link names are inferred from CPU "
              "collision geometry at saved ball centers and held initial joints; they are not logged robot geometry IDs. "
              "No physics steps, GPU allocation, new drops, collision changes or capture relabeling.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "audit", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Hide GPUs for this CPU-only diagnostic")
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):
        parser.error("Use a fresh output outside historical evidence")
    result = diagnose(args.study, args.audit, args.assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], checks=result["checks_count"],
                          cases=[{k:c[k] for k in ("id", "shots", "reconstructed_first_link_counts")} for c in result["cases"]])))


if __name__ == "__main__":
    main()
