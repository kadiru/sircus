"""Identify robot links in the four recorded case-33 misses on the CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .arm_clearance_selector import ArmClearanceGate
from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .calibrated_score import FORCE_NAMES, FORCE_THRESHOLD_N


VERSION = "sircus-intergrid-setup-contact-diagnosis-v1"
SOURCE_AUDIT_SHA256 = "64de6a57c730c40249875a4245db3b52549f5071974b3abdb39b8b461965974c"
CONTEXT_ID = "intergrid-route-fallback-test-33"


def diagnose(study, source_audit, assets):
    checks = Checks()
    audit = read(source_audit)
    report = read(study / "report.json")
    manifest_path = study / "evidence-manifest.json"
    audit_hash, report_hash, manifest_hash = map(sha, (source_audit, study / "report.json", manifest_path))
    checks.equal(audit_hash, SOURCE_AUDIT_SHA256, "Saved setup-drop audit")
    checks.equal((audit["status"], report["status"]), ("passed", "completed"), "Completed audited drops")
    checks.equal(audit["report_sha256"], report_hash, "Bound report bytes")
    checks.equal(audit["evidence_manifest_sha256"], manifest_hash, "Bound trace manifest")
    checks.equal(audit["result"], report["result"], "Matching independent outcomes")
    manifest = read(manifest_path)

    def bound(relative):
        path = contained(study, relative)
        checks.equal(sha(path), manifest[relative], f"Bound artifact: {relative}")
        return path

    spec = read(bound("contract.json"))
    fingerprint(spec, checks, "Frozen setup-drop contract")
    checks.equal(spec["physics_substeps"], 64, "64-substep source")
    checks.equal(digest(spec["scene"]), spec["identity"]["scene_sha256"], "Exact scene")
    for name, expected in spec["identity"]["asset_sha256"].items():
        checks.equal(sha(contained(assets.parent, name)), expected, f"Exact collision asset: {name}")

    case = next(item for item in audit["result"]["cases"] if item["id"] == CONTEXT_ID)
    expected_ids = [shot["id"] for shot in case["shots"]]
    with bound("rows.jsonl").open() as ledger:
        rows = [row for line in ledger if (row := json.loads(line)).get("context_id") == CONTEXT_ID]
    checks.equal(len(rows), 4, "Four released case-33 shots")
    checks.equal([row["id"] for row in rows], expected_ids, "Exact audited shot order")
    gate = ArmClearanceGate(spec["scene"], assets)
    ball_geom = gate.m.geom("ball/geom").id

    def nearest(center):
        gate.d.qpos[gate.ball : gate.ball + 3] = center
        gate.mj.mj_kinematics(gate.m, gate.d)
        return sorted(
            (float(gate.mj.mj_geomDistance(gate.m, gate.d, ball_geom, collider["index"], 0.5, None)),
             collider["name"]) for collider in gate.local
        )

    results = []
    for row in rows:
        name = row["id"]
        checks.equal((row["native_release"], row["invalid_physics"], row["route_success"]),
                     (True, False, False), "Valid miss: " + name)
        checks.equal(row["bucket_xy_m"], case["bucket_xy_m"], "Exact bucket center: " + name)
        checks.equal(row["adapter_directory"], "native-heading", "Expected archive class: " + name)
        relative = f"native-heading/{row['native_cache_file']}"
        trace_path = bound(relative)
        checks.equal(manifest[relative], row["native_cache_sha256"], "Row trace hash: " + name)
        with np.load(trace_path, allow_pickle=False) as trace:
            column = row["native_cache_row"]
            checks.equal(str(trace["record_ids"][column]), name, "Exact trace column")
            time = trace["time_s"]
            centers = trace["position_m"][:, column]
            forces = trace["forces_N"][:row["physics_steps"], column, FORCE_NAMES.index("robot")]
        contact = np.flatnonzero(forces > FORCE_THRESHOLD_N)
        saved = row["experimental_route"]
        checks.require(len(contact) > 0, "Measured robot force: " + name)
        checks.equal(len(contact), saved["robot_contact_steps"], "Matching robot force ticks: " + name)
        first, peak = int(contact[0]), int(contact[np.argmax(forces[contact])])
        checks.equal(float(time[first]), saved["first_robot_contact_s"], "Matching first contact timing")
        checks.require(time[first] > saved["first_board_rebound_s"], "Robot force follows board rebound")
        checks.numeric(float(forces[peak]), saved["peak_robot_force_N"], "Matching peak robot force", 1e-4)
        gate.colliders(row["native_initial_q10"], row["native_initial_q14"], row["bucket_xy_m"])
        initial, at_peak = nearest(centers[first]), nearest(centers[peak])
        distance, geometry = initial[0]
        checks.require(distance < 0 < initial[1][0] and at_peak[0][1] == geometry,
                       "Unique reconstructed overlapping link at onset with same nearest link at peak")
        results.append(dict(
            id=name, intended_route=row["intended_route"], board_rebound_s=saved["first_board_rebound_s"],
            first_robot_contact_s=float(time[first]), peak_robot_force_s=float(time[peak]),
            peak_robot_force_N=float(forces[peak]), geometry=geometry,
            initial_signed_distance_m=distance, next_nearest_distance_m=initial[1][0],
            next_nearest_geometry=initial[1][1], peak_signed_distance_m=at_peak[0][0],
            max_integrated_joint_excursion_rad=row["max_integrated_joint_excursion_rad"],
            trace_sha256=row["native_cache_sha256"],
        ))
    return dict(
        version=VERSION, status="completed", checks=checks.count, context_id=CONTEXT_ID,
        bucket_xy_m=case["bucket_xy_m"], physics_substeps=64, results=results,
        source=dict(audit_sha256=audit_hash, report_sha256=report_hash, evidence_manifest_sha256=manifest_hash),
        model_sha256=gate.model_sha256, diagnostic_source_sha256=sha(__file__), mujoco_version=gate.mj.__version__,
        scope="The saved native robot-force channel identifies collision instants. "
              "The native geometry-ID fields record bucket contacts only. Robot links are inferred with "
              "CPU MuJoCo signed-distance queries using the recorded ball centers and initial held arm joints.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source-audit", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):
        parser.error("Use a fresh output file outside the source study.")
    result = diagnose(args.study, args.source_audit, args.assets)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
