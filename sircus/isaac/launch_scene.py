"""Run the isolated Isaac foundation and capture sensor-derived point clouds.

No Viam imports, robot endpoints, physical calibration or cloud identity.
Default is a stationary scene; --frames 0 keeps it running with bounded latest
preview files. A finite smoke run saves one timestamped synthetic observation.
"""

import argparse
import faulthandler
import json
from pathlib import Path

faulthandler.dump_traceback_later(60, repeat=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--frames", type=int, default=180, help="Render frames; 0 runs until stopped"
)
parser.add_argument("--output", default="/workspace/output")
parser.add_argument("--assets", default="/workspace/assets")
parser.add_argument("--capture-once", action="store_true")
parser.add_argument("--validation-layouts", type=int, default=0)
parser.add_argument("--validation-seed", type=int, default=20261031)
parser.add_argument("--shot-catalogue", action="store_true")
parser.add_argument("--distance-catalogue", action="store_true")
parser.add_argument("--catalogue-probe-only", action="store_true")
parser.add_argument(
    "--catalogue-confirm-source",
    help="Completed catalogue for100 fresh confirmation trials",
)
parser.add_argument("--plane-sweep", action="store_true")
parser.add_argument("--balanced-pilot", action="store_true")
parser.add_argument("--pilot-smoke", action="store_true")
parser.add_argument("--pilot-episodes", type=int, default=384)
parser.add_argument("--reference-policy", action="append", default=[])
parser.add_argument("--distance-sweep", action="store_true")
parser.add_argument("--sweep-policy", action="append", default=[])
parser.add_argument("--ik-audit")
parser.add_argument("--general-episodes", type=int, default=0)
parser.add_argument("--train-layouts", type=int, default=1000)
parser.add_argument("--study-smoke", action="store_true")
parser.add_argument("--study-preview", action="store_true")
parser.add_argument("--rl-episodes", type=int, default=0)
parser.add_argument("--rl-seed", type=int, default=42)
parser.add_argument(
    "--rl-checkpoint", help="Frozen fixed-scene checkpoint evaluation only; no updates"
)
parser.add_argument(
    "--verify-physics",
    action="store_true",
    help="Finite simulation-only bucket entry and articulation-response checks",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.distance_catalogue:
    if (
        args.shot_catalogue
        or args.catalogue_probe_only
        or args.catalogue_confirm_source
    ):
        parser.error(
            "Distance catalogue must run separately from single-distance catalogue"
        )
    args.shot_catalogue = True
if (
    args.catalogue_probe_only or args.catalogue_confirm_source
) and not args.shot_catalogue:
    parser.error("Catalogue options require --shot-catalogue")
if args.catalogue_probe_only and args.catalogue_confirm_source:
    parser.error("Choose contact probes or independent confirmation, not both")
if args.shot_catalogue:
    if (
        args.general_episodes
        or args.rl_episodes
        or args.validation_layouts
        or args.distance_sweep
        or args.balanced_pilot
        or args.plane_sweep
        or args.rl_checkpoint
    ):
        parser.error("Shot catalogue requires its own run")
    args.rl_episodes = 1
if args.plane_sweep:
    if (
        args.general_episodes
        or args.rl_episodes
        or args.validation_layouts
        or args.distance_sweep
        or args.balanced_pilot
        or args.rl_checkpoint
    ):
        parser.error("Plane sweep must run separately from other experiments")
    args.rl_episodes = 1
if args.balanced_pilot:
    if (
        args.general_episodes
        or args.rl_episodes
        or args.validation_layouts
        or args.distance_sweep
        or len(args.reference_policy) != 2
    ):
        parser.error("Balanced pilot requires two references and no other run mode")
    if (
        args.pilot_episodes < 48
        or args.pilot_episodes > 768
        or args.pilot_episodes % 48
    ):
        parser.error("Pilot budget must be a multiple of48, within48–768")
    args.rl_episodes = 1
if args.distance_sweep:
    if (
        args.general_episodes
        or args.rl_episodes
        or args.validation_layouts
        or len(args.sweep_policy) != 2
    ):
        parser.error(
            "Distance sweep requires two selected policies and no training flags"
        )
    args.rl_episodes = 1  # Enable contact sensors/materials; no training dispatch.
if args.general_episodes:
    if (
        args.rl_episodes
        or args.validation_layouts
        or not 1 <= args.train_layouts <= 1000
    ):
        parser.error("General learning must be separate; train layouts1..1000")
    args.rl_episodes = args.general_episodes
if args.validation_layouts < 0 or (args.validation_layouts and args.rl_episodes):
    parser.error("Use a positive validation count, separate from RL")
# A failed launch must never leave an earlier successful result looking current.
Path(args.output).mkdir(parents=True, exist_ok=True)
(Path(args.output) / "result.json").write_text(
    json.dumps({"passed": False, "state": "starting"})
)
app = AppLauncher(args).app
faulthandler.cancel_dump_traceback_later()

import datetime
import shutil
import time

import isaaclab.sim as sim_utils
import numpy as np
import torch
from geometry import WEDGE_FACES, board_world_transform, bucket_wedge, depth_to_points
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import quat_apply
from PIL import Image
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    try:
        raw = json.dumps(value, indent=2, allow_nan=False)
    except ValueError:

        def bad(v, name="root"):
            if isinstance(v, dict):
                return [p for k, item in v.items() for p in bad(item, name + "." + k)]
            if isinstance(v, list):
                return [
                    p for k, item in enumerate(v) for p in bad(item, name + f"[{k}]")
                ]
            return [name] if isinstance(v, float) and not np.isfinite(v) else []

        print("NONFINITE_METADATA", bad(value), flush=True)
        raise
    tmp.write_text(raw)
    tmp.replace(path)


def main():
    cfg = json.loads((Path(__file__).parent / "scene.json").read_text())
    poses = json.loads((Path(args.assets) / "poses.json").read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dt = cfg["simulation"]["physics_dt_s"]
    decimation = cfg["simulation"]["render_every_steps"]
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=dt,
            render_interval=decimation,
            device=args.device,
            physx=sim_utils.PhysxCfg(enable_ccd=False),
        )
    )
    sim.set_camera_view(cfg["camera"]["position_m"], cfg["camera"]["look_at_m"])
    stage = sim.stage
    sim_utils.DomeLightCfg(
        intensity=500, color=(0.85, 0.9, 1), visible_in_primary_ray=False
    ).func(
        "/World/Light",
        sim_utils.DomeLightCfg(
            intensity=500, color=(0.85, 0.9, 1), visible_in_primary_ray=False
        ),
    )
    table = cfg["table"]
    table_cfg = sim_utils.CuboidCfg(
        size=tuple(table["size_m"]),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.23, 0.31, 0.34)),
    )
    table_cfg.func("/World/Table", table_cfg, translation=tuple(table["center_m"]))
    # A floor below the table catches misses; not a substitute for bucket scoring.
    floor = sim_utils.CuboidCfg(
        size=(4, 4, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.11, 0.13, 0.17)),
    )
    floor.func("/World/Floor", floor, translation=(0.67, -0.25, -0.8))
    arms = {}
    for role, name, qkey, pos, rot in [
        ("dropper", "Arm10", "dropper_joint_rad", (0, 0, 0), (1, 0, 0, 0)),
        (
            "board_arm",
            "Arm14",
            "board_joint_rad",
            (cfg["arms"]["base_spacing_m"], 0, 0),
            (0, 0, 0, 1),
        ),
    ]:
        joints = {f"joint{i + 1}": q for i, q in enumerate(poses[qkey])}
        actuators = {
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=100,
                velocity_limit_sim=1.0,
                stiffness=1500,
                damping=100,
            )
        }
        if role == "dropper":
            joints.update({"finger_left_joint": 0.025, "finger_right_joint": 0.025})
            actuators["gripper"] = ImplicitActuatorCfg(
                joint_names_expr=["finger_.*_joint"],
                effort_limit_sim=30,
                velocity_limit_sim=0.1,
                stiffness=1000,
                damping=50,
            )
        asset = ArticulationCfg(
            prim_path="/World/" + name,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(args.assets) / role / (role + ".usd")),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=1
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos, rot=rot, joint_pos=joints
            ),
            actuators=actuators,
        )
        arms[name] = Articulation(asset)
    # Static convex wall wedges preserve the tapered hollow interior. Never use
    # a convex hull around the entire bucket (that would close the opening).
    b = cfg["bucket"]
    bx, by = b["center_xy_m"]
    root = UsdGeom.Xform.Define(stage, "/World/Bucket")
    root.AddTranslateOp().Set(Gf.Vec3d(bx, by, 0))
    for i in range(b["segments"]):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Bucket/Wall{i:02}")
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in bucket_wedge(b, i)])
        mesh.CreateFaceVertexCountsAttr([4] * len(WEDGE_FACES))
        mesh.CreateFaceVertexIndicesAttr([j for f in WEDGE_FACES for j in f])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([(0.91, 0.52, 0.12)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
            "convexHull"
        )
        PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()).CreateContactOffsetAttr(
            0.0005
        )
        PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()).CreateRestOffsetAttr(0.0)
    base = sim_utils.CylinderCfg(
        radius=b["outer_bottom_diameter_m"] / 2,
        height=b["assumed_bottom_thickness_m"],
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.0005, rest_offset=0
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.91, 0.52, 0.12)),
    )
    base.func(
        "/World/Bucket/Base",
        base,
        translation=(0, 0, b["assumed_bottom_thickness_m"] / 2),
    )
    if (
        args.general_episodes
        or args.distance_sweep
        or args.balanced_pilot
        or args.plane_sweep
        or args.shot_catalogue
    ):
        # Runtime USD transforms do not reliably move static PhysX colliders.
        # A kinematic compound body moves all 64 walls and its floor together.
        api = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        api.CreateKinematicEnabledAttr(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateDisableGravityAttr(
            True
        )
        sim._sircus_bucket = RigidObject(
            RigidObjectCfg(prim_path="/World/Bucket", spawn=None)
        )
    ball_cfg = cfg["ball"]
    ball = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Ball",
            spawn=sim_utils.SphereCfg(
                activate_contact_sensors=bool(args.rl_episodes),
                radius=ball_cfg["radius_m"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=ball_cfg["mass_kg"]),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.0005, rest_offset=0
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=ball_cfg["friction"],
                    dynamic_friction=ball_cfg["friction"],
                    restitution=ball_cfg["restitution"],
                    restitution_combine_mode="max" if args.rl_episodes else "average",
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.1, 0.45)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(bx, by, 0.4) if args.verify_physics else (0.18, -0.52, 0.20)
            ),
        )
    )
    c = cfg["camera"]
    camera = Camera(
        CameraCfg(
            prim_path="/World/RGBD",
            update_period=0,
            update_latest_camera_pose=True,
            height=c["height"],
            width=c["width"],
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=c["focal_length_mm"],
                horizontal_aperture=c["horizontal_aperture_mm"],
                clipping_range=(c["near_m"], c["far_m"]),
            ),
        )
    )
    contact = None
    if args.rl_episodes:
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/Ball",
                update_period=0.0,
                filter_prim_paths_expr=["/World/Arm14/impact_board"]
                + (["/World/Bucket"] if args.shot_catalogue else []),
            )
        )
    sim.reset()
    for arm in arms.values():
        arm.write_joint_state_to_sim(
            arm.data.default_joint_pos, arm.data.default_joint_vel
        )
        arm.set_joint_position_target(arm.data.default_joint_pos)
        arm.reset()
    camera.set_world_poses_from_view(
        torch.tensor([c["position_m"]], device=sim.device),
        torch.tensor([c["look_at_m"]], device=sim.device),
    )
    # Pose buffers are zero-initialized; refresh after changing the camera prim.
    camera.reset()
    print("SIRCUS_SCENE_READY", flush=True)
    if args.validation_layouts:
        from validation_study import run

        run(sim, arms, ball, camera, cfg, poses, args, atomic_json)
        return 0
    if args.shot_catalogue:
        if args.distance_catalogue:
            from distance_catalogue import run
        elif args.catalogue_confirm_source:
            from catalogue_confirmation import run
        else:
            from shot_catalogue import run

        run(sim, arms, ball, camera, contact, cfg, poses, args, atomic_json)
        return 0
    if args.plane_sweep:
        from plane_sweep import run

        run(sim, arms, ball, camera, contact, cfg, poses, args, atomic_json)
        return 0
    if args.balanced_pilot:
        from balanced_pilot import run

        run(sim, arms, ball, camera, contact, cfg, poses, args, atomic_json)
        return 0
    if args.distance_sweep:
        from distance_sweep import run

        run(sim, arms, ball, camera, contact, cfg, poses, args, atomic_json)
        return 0
    if args.general_episodes:
        from generalized_rl import train_scene

        train_scene(sim, arms, ball, camera, contact, cfg, poses, args, atomic_json)
        return 0
    if args.rl_episodes:
        from fixed_rl import train

        train(sim, arms, ball, camera, contact, cfg, poses, args, atomic_json)
        return 0
    frame = 0
    captured = False
    max_joint_error = 0.0
    started = time.monotonic()
    probe_initial = None
    board_probe = None
    if args.verify_physics and args.frames < 120:
        raise ValueError("Physics verification requires at least 120 frames")
    while app.is_running() and (args.frames == 0 or frame < args.frames):
        wall = time.monotonic()
        if args.verify_physics and frame == 60:
            probe_initial = {n: a.data.joint_pos.clone() for n, a in arms.items()}
            for arm in arms.values():
                target = arm.data.default_joint_pos.clone()
                idx = arm.joint_names.index("joint6")
                target[:, idx] += 0.05
                arm.set_joint_position_target(target)
        if frame % 300 == 0 and shutil.disk_usage(out).free < 3 * 1024**3:
            raise RuntimeError("Sircus stopped: less than 3 GiB free; no files deleted")
        for _ in range(decimation):
            for arm in arms.values():
                arm.write_data_to_sim()
            sim.step(render=False)
            for arm in arms.values():
                arm.update(dt)
            ball.update(dt)
        sim.render()
        camera.update(dt * decimation, force_recompute=True)
        frame += 1
        if frame < 20:
            continue
        if frame % 30 == 0 or (args.frames and frame == args.frames):
            if args.verify_physics and board_probe is None:
                board_arm = arms["Arm14"]
                idx = board_arm.body_names.index("impact_board")
                wrist = board_arm.body_names.index("link6")
                board_pos = board_arm.data.body_pos_w[0, idx].cpu().numpy()
                normal = (
                    quat_apply(
                        board_arm.data.body_quat_w[0, idx],
                        torch.tensor([0.0, 0.0, 1.0], device=sim.device),
                    )
                    .cpu()
                    .numpy()
                )
                expected_board = board_world_transform(cfg["board"])
                offset = float(
                    torch.linalg.vector_norm(
                        board_arm.data.body_pos_w[0, idx]
                        - board_arm.data.body_pos_w[0, wrist]
                    )
                )
                if (
                    np.linalg.norm(board_pos - expected_board[:3, 3]) > 0.005
                    or float(normal @ expected_board[:3, 2]) < 0.999
                    or abs(offset - cfg["board"]["tool_offset_m"]) > 0.002
                ):
                    raise RuntimeError(
                        f"Board attachment/pose check failed: {board_pos}, {normal}, {offset}"
                    )
                board_probe = {
                    "center_world_m": board_pos.tolist(),
                    "normal_world": normal.tolist(),
                    "wrist_offset_m": offset,
                }
            rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
            depth = (
                camera.data.output["distance_to_image_plane"][0, :, :, 0]
                .cpu()
                .numpy()
                .copy()
            )
            depth[(depth < c["near_m"]) | (depth > c["far_m"])] = np.nan
            k = camera.data.intrinsic_matrices[0].cpu().numpy()
            points, mask = depth_to_points(depth, k)
            if len(points) < 1000:
                raise RuntimeError("Insufficient valid rendered depth pixels")
            if not all(
                bool(torch.isfinite(arm.data.joint_pos).all()) for arm in arms.values()
            ):
                raise RuntimeError("Non-finite articulation state")
            image = Image.fromarray(rgb)
            image.save(out / "latest.tmp.jpg")
            (out / "latest.tmp.jpg").replace(out / "latest.jpg")
            shown = np.nan_to_num(
                (depth - c["near_m"]) / (c["far_m"] - c["near_m"]), nan=1
            )
            Image.fromarray(np.uint8((1 - np.clip(shown, 0, 1)) * 255)).save(
                out / "depth-preview.png"
            )
            arm_state = {
                name: {
                    "joint_names": arm.joint_names,
                    "joint_position": arm.data.joint_pos[0].cpu().tolist(),
                    "joint_velocity": arm.data.joint_vel[0].cpu().tolist(),
                    "joint_position_units": [
                        "m" if n.startswith("finger_") else "rad"
                        for n in arm.joint_names
                    ],
                    "body_names": arm.body_names,
                    "body_position_world_m": arm.data.body_pos_w[0].cpu().tolist(),
                    "body_quaternion_world_wxyz": arm.data.body_quat_w[0]
                    .cpu()
                    .tolist(),
                }
                for name, arm in arms.items()
            }
            errors = [
                float(
                    torch.max(
                        torch.abs(arm.data.joint_pos - arm.data.joint_pos_target)[:, :6]
                    )
                )
                for arm in arms.values()
            ]
            max_joint_error = max(max_joint_error, *errors)
            metadata = {
                "state": "running",
                "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "scene_id": cfg["scene_id"],
                "physics_contract": cfg["physics_contract"],
                "frame": frame,
                "physics_time_s": frame * dt * decimation,
                "gpu": torch.cuda.get_device_name(0),
                "camera": {
                    "intrinsics": k.tolist(),
                    "width": c["width"],
                    "height": c["height"],
                    "world_position_m": camera.data.pos_w[0].cpu().tolist(),
                    "world_quaternion_wxyz_ros": camera.data.quat_w_ros[0]
                    .cpu()
                    .tolist(),
                    "depth_type": "distance_to_image_plane: axial Z in metres",
                    "point_frame": "optical camera: X right,Y down,Z forward",
                    "valid_depth_pixels": int(mask.sum()),
                    "synchronization": "same simulator step; synthetic exact transforms",
                },
                "arms": arm_state,
                "ball_position_world_m": ball.data.root_pos_w[0].cpu().tolist(),
                "max_joint_target_error_rad": max_joint_error,
                "provenance": "synthetic renderer-derived observation; no physical measurements",
                "geometry": cfg,
                "limits": [
                    "No RL policy trained yet",
                    "Rigid board mount; compliance pending",
                    "Ideal depth; noise/occlusion study pending",
                    "Gripper proxy; grasp not validated",
                ],
            }
            atomic_json(out / "status.json", metadata)
            if args.capture_once and not captured:
                stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
                capture = out / stamp
                capture.mkdir()
                image.save(capture / "rgb.png")
                np.save(capture / "depth_m.npy", depth)
                colors = rgb[mask]
                with (capture / "pointcloud_camera.ply").open("w") as f:
                    f.write(
                        f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                    )
                    np.savetxt(
                        f,
                        np.column_stack([points, colors]),
                        fmt="%.5f %.5f %.5f %d %d %d",
                    )
                atomic_json(capture / "metadata.json", metadata)
                stage.Export(str(out / "scene.usda"))
                captured = True
                print(
                    f"SIRCUS_CAPTURE {capture} points={len(points)} joints={[a.num_joints for a in arms.values()]}",
                    flush=True,
                )
        if args.frames == 0:
            time.sleep(max(0, dt * decimation - (time.monotonic() - wall)))
    if args.verify_physics:
        bp = ball.data.root_pos_w[0].cpu().tolist()
        expected = b["assumed_bottom_thickness_m"] + ball_cfg["radius_m"]
        if (
            abs(bp[2] - expected) > 0.004
            or np.linalg.norm(np.array(bp[:2]) - [bx, by]) > 0.035
        ):
            raise RuntimeError(
                f"Bucket probe failed: {bp}, expected bottom center Z {expected}"
            )
        changes = {
            n: float(
                a.data.joint_pos[0, a.joint_names.index("joint6")]
                - probe_initial[n][0, a.joint_names.index("joint6")]
            )
            for n, a in arms.items()
        }
        if any(abs(d - 0.05) > 0.015 for d in changes.values()):
            raise RuntimeError(f"Arm command response failed: {changes}")
        atomic_json(
            out / "physics-verification.json",
            {
                "passed": True,
                "bucket_probe_final_ball_center_m": bp,
                "expected_floor_ball_center_z_m": expected,
                "joint6_command_delta_rad": 0.05,
                "joint6_measured_delta_rad": changes,
                "board_initial_pose_verified": board_probe,
                "note": "Synthetic open-top collider/articulation checks; not a board-bounce capture result",
            },
        )
        print("SIRCUS_PHYSICS_VERIFIED", flush=True)
    print(
        f"SIRCUS_COMPLETE frames={frame} max_joint_error={max_joint_error:.6f} elapsed={time.monotonic() - started:.1f}",
        flush=True,
    )
    atomic_json(
        out / "result.json",
        {
            "passed": True,
            "state": "completed",
            "frames": frame,
            "physics_verified": args.verify_physics,
        },
    )
    return 0


run_started_wall = time.time()
try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    pilot_report_path = Path(args.output) / "balanced-pilot.json"
    if (
        args.balanced_pilot
        and pilot_report_path.exists()
        and pilot_report_path.stat().st_mtime >= run_started_wall
    ):
        report = json.loads(pilot_report_path.read_text())
        if report.get("phase") != "blocked_pose_verification":
            report["phase"] = "failed"
        report["error"] = traceback.format_exc()
        atomic_json(pilot_report_path, report)
    if args.validation_layouts and (Path(args.output) / "validation.json").exists():
        report = json.loads((Path(args.output) / "validation.json").read_text())
        report.update(state="failed", error=traceback.format_exc())
        atomic_json(Path(args.output) / "validation.json", report)
    (Path(args.output) / "result.json").write_text(
        json.dumps(
            {"passed": False, "state": "failed", "error": traceback.format_exc()}
        )
    )
    raise
finally:
    # All observation writes above are synchronous. Sim 5.1's Replicator/stage
    # cleanup can hang here; use its supported immediate shutdown path.
    app.close(wait_for_replicator=False, skip_cleanup=True)
