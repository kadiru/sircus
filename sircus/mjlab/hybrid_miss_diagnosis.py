"""CPU-only, evidence-bound diagnosis of the three stalled-reference misses.

Recorded force channels establish contact timing. A separate CPU geometry
reconstruction identifies the touching link; it is not a new native replay.
Conditional ballistic projections ignore intervening objects and cannot prove
capture, retention, or a safe corrected shot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark_verify import digest, read, sha
from .calibrated_score import FORCE_NAMES, FORCE_THRESHOLD_N
from .consistent_flight_selector import _descending_time, geometry, member_flight
from .intergrid_flight_ranker import rebound

VERSION = "sircus-hybrid-miss-diagnosis-v1"
CASE_IDS = tuple(f"stalled-reference-test-{i:02d}-hybrid" for i in (3, 9, 19))


def episodes(time, positions, velocities, forces):
    """Each force interval uses the pre-contact sample and post-contact sample."""
    time, positions, velocities, forces = map(np.asarray, (time, positions, velocities, forces))
    n = len(time) - 1
    if (n < 1 or positions.shape != (n + 1, 3) or velocities.shape != (n + 1, 3)
            or forces.shape != (n, len(FORCE_NAMES)) or not all(
                np.isfinite(x).all() for x in (time, positions, velocities, forces))
            or (np.diff(time) <= 0).any() or (forces < 0).any()):
        raise ValueError("Invalid recorded force/state interval alignment")
    result = []
    for channel, name in enumerate(FORCE_NAMES):
        edges = np.flatnonzero(np.diff(np.r_[False, forces[:, channel] > FORCE_THRESHOLD_N, False]))
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            result.append(dict(channel=name, start_tick=int(start), end_tick=int(end),
                start_s=float(time[start]), end_s=float(time[end]),
                position_m=positions[start].tolist(), velocity_before_m_s=velocities[start].tolist(),
                velocity_after_m_s=velocities[end].tolist(),
                peak_force_N=float(forces[start:end, channel].max())))
    return sorted(result, key=lambda x: (x["start_tick"], x["channel"]))


def conditional_flight(origin, velocity, bucket, geom, coefficients):
    """Gravity-only flight and one effective table rebound, with objects ignored."""
    origin, velocity = np.asarray(origin, float), np.asarray(velocity, float)
    gravity = np.asarray(geom["gravity_m_s2"], float)
    keys = ("first_descending_center_rim", "conservative_rim_band_clearance_m")
    direct = member_flight(origin, velocity, bucket, geom)
    out = dict(direct={key: direct[key] for key in keys}, table=None)
    duration = _descending_time(origin[2], velocity[2], geom["ball_radius_m"], gravity[2])
    if duration is not None:
        point = origin + velocity * duration + .5 * gravity * duration ** 2
        after = rebound(velocity + gravity * duration, np.array([0., 0., 1.]),
                        coefficients["table_normal_restitution"], coefficients["table_tangent_retention"])
        table = member_flight(point, after, bucket, geom)
        out["table"] = {key: table[key] for key in keys}
        out["table_contact_m"] = point.tolist()
    return out


def contact_geometry(scene, assets, row, position):
    """Reconstruct held endpoints at one recorded ball position; never mj_step."""
    import mujoco
    from .model import build_scene
    model = mujoco.MjModel.from_xml_string(build_scene(scene, assets, solref=(.004, .15), contact_friction=.35))
    data = mujoco.MjData(model)
    for role, key in (("Arm10", "q10"), ("Arm14", "q14")):
        for j, value in enumerate(row["native_initial_" + key], 1):
            data.qpos[model.jnt_qposadr[model.joint(f"{role}/joint{j}").id]] = value
    for side in ("left", "right"):
        data.qpos[model.jnt_qposadr[model.joint(f"Arm10/finger_{side}_joint").id]] = float(np.float32(.025))
    start = int(model.jnt_qposadr[model.joint("ball/free").id])
    data.qpos[start:start + 3], data.qpos[start + 3:start + 7] = position, [1., 0., 0., 0.]
    bucket = model.body_mocapid[model.body("bucket").id]
    data.mocap_pos[bucket], data.mocap_quat[bucket] = row["native_initial_bucket_center_m"], [1., 0., 0., 0.]
    mujoco.mj_forward(model, data)
    board = model.body("Arm14/impact_board").id
    if (not np.allclose(data.xpos[board], row["board_center_m"], atol=2e-6, rtol=0)
            or not np.allclose(data.xmat[board].reshape(3, 3)[:, 2], row["board_normal"], atol=2e-6, rtol=0)):
        raise ValueError("Reconstructed board differs from recorded endpoint")
    found = []
    for contact in data.contact:
        names = [model.geom(int(g)).name for g in contact.geom]
        if "ball/geom" in names:
            found.append(dict(geometries=names, distance_m=float(contact.dist)))
    return found


def diagnose(study, audit, assets):
    study, audit, assets = map(Path, (study, audit, assets))
    receipt, contract = read(audit), read(study / "contract.json")
    if (receipt["status"] != "passed" or receipt["report_sha256"] != sha(study / "report.json")
            or receipt["choices_sha256"] != sha(study / "choices.json")):
        raise ValueError("Need the unchanged audited parent report")
    if digest({k: v for k, v in contract.items() if k != "fingerprint"}) != contract["fingerprint"]:
        raise ValueError("Parent contract fingerprint changed")
    if sha(study / "calibration.json") != contract["frozen_inputs"]["calibration.json"]:
        raise ValueError("Parent calibration changed")
    from .saved_shot_video import verify_assets
    verify_assets(assets, contract["identity"]["asset_sha256"])
    root = Path(__file__).resolve().parents[2]
    for name in ("sircus/mjlab/model.py", "sircus/isaac/geometry.py"):
        if sha(root / name) != contract["identity"]["source_sha256"][name]:
            raise ValueError("Historical geometry source changed")
    geom = geometry(contract["scene"], read(study / "calibration.json")["score_config"])
    coefficients = contract["flight_ranking"]["rule"]["coefficients"]
    rows = {x["id"]: x for x in map(json.loads, (study / "rows.jsonl").read_text().splitlines())}
    slots = {x["id"]: x for x in read(study / "choices.json")["slots"]}
    cases = []
    for key in CASE_IDS:
        row, slot = rows[key], slots[key]
        for field in ("values", "bucket_xy_m", "selected_q10", "selected_q14"):
            if row[field] != slot["request"][field]:
                raise ValueError("Executed request differs from frozen choice: " + field)
        if (row["success"] or row["whole_ball_entry"] or row["invalid_physics"]
                or not row["native_release"] or row["contact_protocol"]["substeps"] != 64):
            raise ValueError("Expected valid native64 miss without entry")
        archive = study / row["adapter_directory"] / row["native_cache_file"]
        if sha(archive) != row["native_cache_sha256"]:
            raise ValueError("Native archive changed")
        with np.load(archive, allow_pickle=False) as saved:
            col, n = row["native_cache_row"], row["physics_steps"]
            if str(saved["record_ids"][col]) != key:
                raise ValueError("Native archive identity mismatch")
            t, p, v, f = (saved["time_s"][:n + 1], saved["position_m"][:n + 1, col],
                          saved["velocity_m_s"][:n + 1, col], saved["forces_N"][:n, col])
        events = episodes(t, p, v, f)
        first_board = next(e for e in events if e["channel"] == "board")
        first_robot = next(e for e in events if e["channel"] == "robot")
        end = first_board["end_tick"]
        forecast = conditional_flight(p[end], v[end], row["bucket_xy_m"], geom, coefficients)
        contacts = contact_geometry(contract["scene"], assets, row, p[first_robot["start_tick"]])
        cases.append(dict(id=key, bucket_xy_m=row["bucket_xy_m"], values=row["values"],
            release_ball_center_m=row["release_ball_center_m"], original_prediction=slot["prediction"],
            observed_events=events, reconstructed_first_robot_contacts=contacts,
            measured_post_board=dict(time_s=float(t[end]), position_m=p[end].tolist(), velocity_m_s=v[end].tolist()),
            conditional_no_robot_flight=forecast,
            success=False, whole_ball_entry=False, max_retention_s=row["max_retention_s"],
            trace=dict(file=str(archive), sha256=row["native_cache_sha256"], column=row["native_cache_row"],
                       terminal_tick=row["physics_steps"])))
    return dict(version=VERSION, status="completed", cpu_only=True, cases=cases,
        evidence=dict(study=str(study), report_sha256=sha(study / "report.json"),
            rows_sha256=sha(study / "rows.jsonl"), choices_sha256=sha(study / "choices.json"),
            contract_sha256=sha(study / "contract.json"), calibration_sha256=sha(study / "calibration.json"),
            audit_sha256=sha(audit)),
        scope="Saved native force/state diagnosis; link names from CPU endpoint geometry reconstruction; conditional projections ignore robot/bucket contacts and are not new drops or capture evidence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "audit", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing diagnosis; choose an unused output")
    result = diagnose(args.study, args.audit, args.assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], cases=len(result["cases"]), output=str(args.output))))


if __name__ == "__main__":
    main()
