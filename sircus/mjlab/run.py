"""GPU mjlab replay of matched two-arm shot states with the shared capture scorer."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch

from .model import build_scene
from .contract import ScoreStream, verify_manifest


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist(),
        )
    )
    tmp.replace(path)


def sensor_config(spec, model):
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    robot = tuple(
        n
        for n in body_names
        if n and n.startswith(("Arm10/", "Arm14/")) and n != "Arm14/impact_board"
    )
    groups = [
        ("board", "body", "Arm14/impact_board"),
        ("bucket", "body", "bucket"),
        ("table", "geom", "table"),
        ("floor", "geom", "floor"),
        ("robot", "body", robot),
    ]
    sensors = []
    for name, mode, pattern in groups:
        cfg = ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode=mode, pattern=pattern),
            secondary=ContactMatch(mode="geom", pattern="ball/geom"),
            fields=("force",),
            reduce="netforce",
        )
        s = cfg.build()
        s.edit_spec(spec, {})
        sensors.append(s)
    s = ContactSensorCfg(
        name="net",
        primary=ContactMatch(mode="geom", pattern="ball/geom"),
        fields=("force",),
        reduce="netforce",
    ).build()
    s.edit_spec(spec, {})
    sensors.append(s)
    return sensors


def run(a):
    started = time.perf_counter()
    manifest = json.loads(a.manifest.read_text())
    verify_manifest(manifest)
    a.output.mkdir(exist_ok=False, parents=True)
    calibration = dict(
        selected=dict(timeconst_s=a.solref[0], damping_ratio=a.solref[1]),
        scope="Frozen from separate flat-drop calibration; see paired study report.",
    )
    atomic(a.output / "normal-calibration.json", calibration)
    solref = (
        calibration["selected"]["timeconst_s"],
        calibration["selected"]["damping_ratio"],
    )
    xml = build_scene(manifest["scene"], a.assets, solref=solref, contact_friction=0.35)
    spec = mujoco.MjSpec.from_string(xml)
    basic = spec.compile()
    sensors = sensor_config(spec, basic)
    m = spec.compile()
    (a.output / "scene.xml").write_text(spec.to_xml())
    n = len(manifest["shots"])
    device = "cuda:0"
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(
            "Only the explicitly selected RTX 4090 may run this experiment"
        )
    settings = SimulationCfg(
        mujoco=MujocoCfg(timestep=manifest["dt_s"] / a.substeps),
        nconmax=512,
        njmax=2048,
    )
    sim = Simulation(
        num_envs=n,
        cfg=settings,
        model=m,
        device=device,
    )
    bucket = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bucket")
    mocap = int(m.body_mocapid[bucket])
    bucket_targets = torch.tensor(
        [[*r["bucket_xy_m"], 0] for r in manifest["shots"]], device=device
    )
    sim.data.mocap_pos[:, mocap, :] = bucket_targets
    sim.data.mocap_quat[:, mocap, :] = torch.tensor([1, 0, 0, 0], device=device)
    for s in sensors:
        s.initialize(sim.mj_model, sim.model, sim.data, device)
    ball_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball/free")
    ball_q = int(m.jnt_qposadr[ball_j])
    ball_v = int(m.jnt_dofadr[ball_j])
    board = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Arm14/impact_board")
    q = np.tile(m.qpos0, (n, 1))
    for i, row in enumerate(manifest["shots"]):
        for role, vals in row["final_joint_positions_rad"].items():
            names = [f"{role}/joint{k+1}" for k in range(6)]
            if role == "Arm10":
                names += [f"{role}/finger_left_joint", f"{role}/finger_right_joint"]
            for name, val in zip(names, vals):
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if j < 0:
                    raise ValueError(name)
                q[i, m.jnt_qposadr[j]] = val
        q[i, ball_q : ball_q + 3] = row["release_ball_center_m"]
        q[i, ball_q + 3 : ball_q + 7] = [1, 0, 0, 0]
    q = torch.tensor(q, device=device, dtype=torch.float32)
    sim.data.qpos[:] = q
    sim.data.qvel[:] = 0
    sim.forward()
    bucket_error = torch.linalg.vector_norm(
        sim.data.xpos[:, bucket, :] - bucket_targets, dim=-1
    )
    bucket_geoms = [
        i
        for i in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
            "bucket/"
        )
    ]
    expected = bucket_targets[:, None, :] + torch.tensor(
        m.geom_pos[bucket_geoms], device=device, dtype=torch.float32
    )
    geom_error = float(
        torch.linalg.vector_norm(
            sim.data.geom_xpos[:, bucket_geoms, :] - expected, dim=-1
        ).max()
    )
    if float(bucket_error.max()) > 1e-5 or geom_error > 1e-5:
        raise RuntimeError("GPU bucket pose differs from the scoring geometry")
    actuator_q = [int(m.jnt_qposadr[j]) for j in m.actuator_trnid[:, 0]]
    sim.data.ctrl[:] = q[:, actuator_q]
    centers = sim.data.xpos[:, board, :].cpu().numpy()
    matrices = sim.data.xmat[:, board, :].reshape(n, 3, 3).cpu().numpy()
    normals = matrices[:, :, 2]
    flange = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Arm10/link6")
    flange_xyz = sim.data.xpos[:, flange, :].cpu().numpy()
    gates = []
    for r, p, normal, tool in zip(manifest["shots"], centers, normals, flange_xyz):
        err = float(np.linalg.norm(p - r["settled_board_center_m"]) * 1000)
        deg = float(
            np.degrees(
                np.arccos(np.clip(np.dot(normal, r["settled_board_normal"]), -1, 1))
            )
        )
        drop_err = float(
            np.linalg.norm(
                tool - np.asarray(r["release_ball_center_m"]) - [0, 0, 0.095]
            )
            * 1000
        )
        gates.append(
            dict(
                board_position_error_mm=err,
                board_normal_error_deg=deg,
                dropper_position_error_mm=drop_err,
                passed=err <= 2 and deg <= 0.5 and drop_err <= 2,
            )
        )
    rows = []
    warm = time.perf_counter()
    setup_s = warm - started
    for repeat in range(manifest["repeats"]):
        sim.reset()
        sim.data.mocap_pos[:, mocap, :] = bucket_targets
        sim.data.mocap_quat[:, mocap, :] = torch.tensor([1, 0, 0, 0], device=device)
        sim.data.qpos[:] = q
        sim.data.qvel[:] = 0
        sim.data.ctrl[:] = q[:, actuator_q]
        sim.forward()
        for s in sensors:
            s.reset()
        streams = [
            ScoreStream(manifest["scene"], r["bucket_xy_m"], normals[i])
            for i, r in enumerate(manifest["shots"])
        ]
        trails = [[] for _ in streams]
        ticks = []
        first = [
            dict.fromkeys(
                ("board", "bucket", "table", "floor", "robot", "unattributed")
            )
            for _ in streams
        ]
        peaks = np.zeros((n, 6))
        max_drift = np.zeros(n)
        for step in range(manifest["horizon_steps"]):
            forces_gpu = torch.zeros((n, 6), device=device)
            for _ in range(a.substeps):
                sim.data.qpos[:, :ball_q] = q[:, :ball_q]
                sim.data.qvel[:, :ball_v] = 0
                sim.data.qfrc_applied[:, :ball_v] = sim.data.qfrc_bias[:, :ball_v]
                sim.step()
                for s in sensors:
                    s.update(manifest["dt_s"] / a.substeps)
                vectors = [s.data.force for s in sensors]
                blocks = [
                    torch.linalg.vector_norm(v, dim=-1).sum(dim=1) for v in vectors[:5]
                ]
                residual = torch.linalg.vector_norm(
                    vectors[-1].sum(1) + sum(v.sum(1) for v in vectors[:5]), dim=-1
                )
                forces_gpu = torch.maximum(
                    forces_gpu, torch.stack([*blocks, residual], dim=-1)
                )
                drift = sim.data.xpos[:, board, :].cpu().numpy()
                max_drift = np.maximum(
                    max_drift, np.linalg.norm(drift - centers, axis=1) * 1000
                )
            forces = forces_gpu.cpu().numpy()
            peaks = np.maximum(peaks, forces)
            xyz = sim.data.qpos[:, ball_q : ball_q + 3].cpu().numpy()
            vel = sim.data.qvel[:, ball_v : ball_v + 3].cpu().numpy()
            if not np.isfinite(xyz).all():
                raise RuntimeError("Nonfinite MuJoCo state")
            for i, s in enumerate(streams):
                if gates[i]["passed"]:
                    s.step(
                        xyz[i],
                        vel[i],
                        dict(
                            zip(
                                (
                                    "board",
                                    "bucket",
                                    "table",
                                    "floor",
                                    "robot",
                                    "unattributed",
                                ),
                                forces[i],
                            )
                        ),
                    )
                for j, k in enumerate(first[i]):
                    if first[i][k] is None and forces[i, j] > 0.01:
                        first[i][k] = step + 1
                if (step + 1) % 4 == 0:
                    trails[i].append(xyz[i].tolist())
            if (step + 1) % 4 == 0:
                ticks.append(step + 1)
        for i, (r, s) in enumerate(zip(manifest["shots"], streams)):
            rows.append(
                dict(
                    shot_id=r["shot_id"],
                    repeat=repeat,
                    **s.result(),
                    pose_gate=gates[i],
                    invalid=None if gates[i]["passed"] else "board_pose_gate",
                    trajectory_steps=ticks,
                    trajectory_m=trails[i],
                    board_center_m=centers[i].tolist(),
                    board_normal=normals[i].tolist(),
                    max_board_drift_mm=float(max_drift[i]),
                    first_contact_step=first[i],
                    all_horizon_contact_peaks_N=dict(zip(first[i], peaks[i].tolist())),
                )
            )
        atomic(a.output / "progress.json", dict(repeats=repeat + 1, shots=len(rows)))
    elapsed = time.perf_counter() - warm
    report = dict(
        benchmark_fingerprint=manifest["fingerprint"],
        engine="mjlab",
        versions={
            k: importlib.metadata.version(k)
            for k in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch")
        },
        protocol="fixed_endpoint_contact_replay",
        simulation_settings=asdict(settings),
        physics_substeps=a.substeps,
        physics_dt_s=manifest["dt_s"] / a.substeps,
        bucket_collider_max_error_m=geom_error,
        device=torch.cuda.get_device_name(0),
        visible_gpu_count=torch.cuda.device_count(),
        contact=dict(
            solref=solref,
            effective_friction=0.35,
            calibration="normal-calibration.json",
        ),
        timing=dict(
            setup_and_compile_s=setup_s,
            rollout_with_shared_scoring_s=elapsed,
            attempts_per_s=len(rows) / elapsed,
        ),
        shots=rows,
    )
    atomic(a.output / "result.json", report)
    print(
        json.dumps(
            dict(
                engine=report["engine"],
                shots=len(rows),
                captures=sum(r["complete_success"] for r in rows),
                seconds=elapsed,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--substeps", type=int, choices=(1, 4, 8, 16), default=8)
    p.add_argument("--solref", type=float, nargs=2, default=(0.004, 0.15))
    run(p.parse_args())
