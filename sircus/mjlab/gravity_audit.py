"""CPU-only forward-dynamics audit of the historical held-arm force balance.

No trajectory, GPU context, training, hardware connection or source mutation is
performed.  The ball is parked away from the robots so contact response cannot
explain the measured acceleration.  Historical model construction is imported
unchanged; its source, native XML and referenced assets are fingerprinted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

VERSION = "sircus-cpu-held-arm-gravity-audit-v1"
DEFAULT_SOURCES = (
    "stable-shot-0004711",  # The searched source that later scored 13/20.
    "stable-shot-0000000",  # Saved captured old-test-10 control.
    "stable-shot-0000001",  # Saved captured old-test-19 control.
)
PARKED_BALL_M = [0.0, 0.0, 2.0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure_condition(model, data, arm_dofs, jac_position, jac_rotation, forward, *, extra_bias):
    """Compare zero applied force with the old extra-bias injection.

    Only robot joint DOFs receive the extra force.  The ball remains a free body
    and is excluded from acceleration assertions and force injection.
    """
    data.qfrc_applied[:] = 0
    forward(model, data)
    bias = data.qfrc_bias.copy()
    if extra_bias:
        data.qfrc_applied[arm_dofs] = bias[arm_dofs]
        forward(model, data)
    return dict(
        extra_bias_applied=bool(extra_bias),
        arm_qfrc_bias_N_or_Nm=bias[arm_dofs].tolist(),
        arm_qfrc_passive_N_or_Nm=data.qfrc_passive[arm_dofs].tolist(),
        arm_qfrc_gravcomp_N_or_Nm=data.qfrc_gravcomp[arm_dofs].tolist(),
        arm_qfrc_applied_N_or_Nm=data.qfrc_applied[arm_dofs].tolist(),
        arm_qacc_rad_s2=data.qacc[arm_dofs].tolist(),
        max_abs_arm_qacc_rad_s2=float(np.abs(data.qacc[arm_dofs]).max()),
        board_linear_acceleration_m_s2=(jac_position @ data.qacc).tolist(),
        board_angular_acceleration_rad_s2=(jac_rotation @ data.qacc).tolist(),
        contacts=int(data.ncon),
    )


def checks(case):
    """Diagnostic conclusions with explicit tolerances; no contact claims."""
    zero, extra = case["without_extra_bias"], case["with_extra_bias"]
    bias = np.asarray(zero["arm_qfrc_bias_N_or_Nm"])
    passive = np.asarray(zero["arm_qfrc_passive_N_or_Nm"])
    gravcomp = np.asarray(zero["arm_qfrc_gravcomp_N_or_Nm"])
    return dict(
        contact_free=zero["contacts"] == extra["contacts"] == 0,
        passive_already_cancels_bias=bool(np.allclose(bias, passive, rtol=0, atol=1e-9)),
        passive_is_gravity_compensation=bool(np.allclose(passive, gravcomp, rtol=0, atol=1e-9)),
        no_extra_is_stationary=zero["max_abs_arm_qacc_rad_s2"] < 1e-8,
        extra_bias_causes_acceleration=extra["max_abs_arm_qacc_rad_s2"] > 1e-3,
    )


def probe_pose(model, row, mujoco):
    data = mujoco.MjData(model)
    joint = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    arm_dofs = []
    joint_names = []
    for role in (10, 14):
        for k, value in enumerate(row[f"q{role}"], 1):
            name = f"Arm{role}/joint{k}"
            j = joint(name)
            data.qpos[model.jnt_qposadr[j]] = value
            arm_dofs.append(int(model.jnt_dofadr[j]))
            joint_names.append(name)
    for side in ("left", "right"):
        data.qpos[model.jnt_qposadr[joint(f"Arm10/finger_{side}_joint")]] = .025
    ball_q = model.jnt_qposadr[joint("ball/free")]
    data.qpos[ball_q:ball_q + 3] = PARKED_BALL_M
    data.qpos[ball_q + 3:ball_q + 7] = [1, 0, 0, 0]
    bucket = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bucket")
    mocap = model.body_mocapid[bucket]
    data.mocap_pos[mocap] = [*row["bucket_xy_m"], 0]
    data.mocap_quat[mocap] = [1, 0, 0, 0]
    data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    board = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Arm14/impact_board")
    jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
    mujoco.mj_jacBody(model, data, jp, jr, board)
    result = dict(
        source_id=row["id"], target_id=row["target_id"],
        source_record_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        requested_values=row["values"], bucket_xy_m=row["bucket_xy_m"],
        q10_rad=row["q10"], q14_rad=row["q14"], joint_names=joint_names,
        board_body_gravcomp=float(model.body_gravcomp[board]),
    )
    for label, extra in (("without_extra_bias", False), ("with_extra_bias", True)):
        result[label] = measure_condition(model, data, arm_dofs, jp, jr, mujoco.mj_forward, extra_bias=extra)
    result["checks"] = checks(result)
    result["passed"] = all(result["checks"].values())
    return result


def run(args):
    import mujoco  # CPU library; no torch/mjlab/Warp import is needed.
    from . import model as model_source

    contract_path = args.study / "contract.json"
    records_path = args.study / "shots.jsonl"
    scene = json.loads(contract_path.read_text())["scene"]
    sources = set(args.source_id or DEFAULT_SOURCES)
    rows = {r["id"]: r for line in records_path.read_text().splitlines()
            if (r := json.loads(line))["id"] in sources}
    if rows.keys() != sources:
        raise ValueError("Missing requested saved source IDs: " + str(sources - rows.keys()))
    xml = model_source.build_scene(scene, args.assets, solref=(.004, .15), contact_friction=.35)
    native = mujoco.MjModel.from_xml_string(xml)
    asset_paths = {args.assets / "dropper.urdf", args.assets / "board_arm.urdf"}
    asset_paths.update(Path(e.attrib["file"]) for e in ET.fromstring(xml).findall("./asset/mesh") if "file" in e.attrib)
    cases = [probe_pose(native, rows[s], mujoco) for s in sorted(sources)]
    result = dict(
        version=VERSION, status="passed" if all(c["passed"] for c in cases) else "failed",
        execution="CPU-only MuJoCo mj_forward; no integration or GPU context",
        scope="Historical held-arm control diagnostic, not a ball-contact or transit validation",
        parked_ball_m=PARKED_BALL_M, joint_velocity_rad_s=0,
        solref=[.004, .15], friction=.35, mujoco_version=mujoco.__version__,
        model_source_sha256=sha(model_source.__file__),
        audit_source_sha256=sha(__file__), generated_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        contract_sha256=sha(contract_path), records_sha256=sha(records_path),
        assets_sha256={str(path): sha(path) for path in sorted(asset_paths)},
        tolerances=dict(force_balance_abs=1e-9, stationary_qacc_abs=1e-8, nonzero_qacc_min=1e-3),
        cases=cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(dict(status=result["status"], output=str(args.output), cases=len(cases))))
    if result["status"] != "passed":
        raise RuntimeError("CPU gravity diagnostic failed its declared checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", action="append")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
