"""Local USD assets and Isaac Lab clone configuration, imported after AppLauncher."""

from pathlib import Path

import isaaclab.sim as sim_utils
from geometry import WEDGE_FACES, bucket_wedge
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from parallel_contract import SPACING_M
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


def bucket_asset(path, bucket):
    path = Path(path)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Bucket")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateDisableGravityAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(1.0)
    for i in range(bucket["segments"]):
        mesh = UsdGeom.Mesh.Define(stage, f"/Bucket/Wall{i:02}")
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in bucket_wedge(bucket, i)])
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
    base = UsdGeom.Cylinder.Define(stage, "/Bucket/Base")
    base.CreateRadiusAttr(bucket["outer_bottom_diameter_m"] / 2)
    base.CreateHeightAttr(bucket["assumed_bottom_thickness_m"])
    base.AddTranslateOp().Set(Gf.Vec3d(0, 0, bucket["assumed_bottom_thickness_m"] / 2))
    base.CreateDisplayColorAttr([(0.91, 0.52, 0.12)])
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(base.GetPrim()).CreateContactOffsetAttr(0.0005)
    PhysxSchema.PhysxCollisionAPI.Apply(base.GetPrim()).CreateRestOffsetAttr(0.0)
    stage.GetRootLayer().Save()


def scene_config(cfg, poses, assets, bucket_usd, num_envs, robot_filters):
    scene = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=SPACING_M,
        replicate_physics=True,
        filter_collisions=True,
    )
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
            joints.update(finger_left_joint=0.025, finger_right_joint=0.025)
            actuators["gripper"] = ImplicitActuatorCfg(
                joint_names_expr=["finger_.*_joint"],
                effort_limit_sim=30,
                velocity_limit_sim=0.1,
                stiffness=1000,
                damping=50,
            )
        setattr(
            scene,
            name,
            ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(Path(assets) / role / (role + ".usd")),
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
            ),
        )
    for name, size, pos, color in [
        (
            "Table",
            tuple(cfg["table"]["size_m"]),
            tuple(cfg["table"]["center_m"]),
            (0.23, 0.31, 0.34),
        ),
        ("Floor", (4, 4, 0.05), (0.67, -0.25, -0.8), (0.11, 0.13, 0.17)),
    ]:
        setattr(
            scene,
            name,
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True, disable_gravity=True
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            ),
        )
    scene.Bucket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bucket",
        spawn=sim_utils.UsdFileCfg(usd_path=str(bucket_usd)),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(*cfg["bucket"]["center_xy_m"], 0)
        ),
    )
    b = cfg["ball"]
    scene.Ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=b["radius_m"],
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=b["mass_kg"]),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0005, rest_offset=0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=b["friction"],
                dynamic_friction=b["friction"],
                restitution=b["restitution"],
                restitution_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.45)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, -0.8, 0.04)),
    )
    filters = ["Arm14/impact_board", "Bucket", "Table", "Floor"] + robot_filters
    scene.contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        update_period=0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/" + f for f in filters],
    )
    scene.light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1200, color=(0.85, 0.9, 1), visible_in_primary_ray=False
        ),
    )
    return scene, filters


def robot_body_filters(assets):
    result = []
    for role, name in [("dropper", "Arm10"), ("board_arm", "Arm14")]:
        stage = Usd.Stage.Open(str(Path(assets) / role / (role + ".usd")))
        root = str(stage.GetDefaultPrim().GetPath())
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                relative = str(p.GetPath())[len(root) :].lstrip("/")
                if relative != "impact_board":
                    result.append(f"{name}/{relative}")
    return sorted(result)
